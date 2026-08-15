"""Classifier-based constrained BO for single-objective constrained problems.

Models the objective with a ``SingleTaskGP`` (input-normalized, output-standardized)
and models *feasibility* with a variational GP **classifier** rather than a regression
GP per constraint. In BoCoDe a point is feasible when every constraint is satisfied
(``c <= 0``); each observation is reduced to a single binary label (1 = feasible,
0 = infeasible) and a ``gpytorch`` ``ApproximateGP`` (``CholeskyVariationalDistribution``
+ ``VariationalStrategy`` + ``BernoulliLikelihood``) is trained on those labels with the
variational ELBO. The classifier predicts ``P(feasible | x)``.

The next point maximizes ``qLogExpectedImprovement`` on the objective, weighted by the
classifier's feasibility probability. This uses BoTorch's classifier-as-constraint path:
the constraint callback returns the Bernoulli likelihood probability directly (with a
small epsilon so ``log(0)`` is avoided) and is passed with ``fat=[None]``, so
``compute_smoothed_feasibility_indicator`` adds ``log(prob)`` instead of applying a
sigmoid to a signed constraint value. (BoTorch's default constraint convention is that
*negative* values imply feasibility; the ``fat=None`` path used here instead expects the
callable to deliver a feasibility probability in ``[0, 1]``.)

This is a direct BoCoDe port of the BoTorch community notebook
"Classifier-based Constrained BO":
https://github.com/meta-pytorch/botorch/blob/main/notebooks_community/clf_constrained_bo/clf_constrained_bo.ipynb

Run::

    python -m algorithms.single_obj_constrained.clf_cbo --problem ConstrainedHartmann --init 10 --iters 50

Sources:
BoTorch community notebook, Classifier-based Constrained BO: https://github.com/meta-pytorch/botorch/blob/main/notebooks_community/clf_constrained_bo/clf_constrained_bo.ipynb
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020. http://arxiv.org/abs/1910.06403
S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy. Unexpected Improvements to Expected Improvement for Bayesian Optimization. NeurIPS 2023. https://arxiv.org/abs/2310.20708
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import gpytorch
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

from .._bo_utils import (
    DTYPE,
    ProblemObjective,
    Result,
    add_common_args,
    default_n_init,
    finalize,
    initial_design,
    load_checkpoint,
    make_problem,
    resolve_device,
    robust_fit_mll,
    save_checkpoint,
    set_seed,
)


class _FeasibilityClassifier(ApproximateGP, GPyTorchModel):
    """Variational GP classifier over feasibility (the notebook's ``GP_vi``).

    A ``CholeskyVariationalDistribution`` + ``VariationalStrategy`` ApproximateGP with a
    ``BernoulliLikelihood``, trained on binary feasibility labels via the variational
    ELBO. Its posterior latent, pushed through the Bernoulli likelihood, gives
    ``P(feasible | x)``. ``_num_outputs = 1`` is required so it can live in a
    ``ModelListGP`` alongside the objective GP.
    """

    _num_outputs = 1

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.train_inputs = (train_x,)
        self.train_targets = train_y
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _pass_obj(Z: torch.Tensor, X=None) -> torch.Tensor:
    """MC objective: the objective output is index 0 of the ModelListGP samples."""
    return Z[..., 0]


def _pass_con(Z: torch.Tensor, model_con, X=None) -> torch.Tensor:
    """Classifier-as-constraint callback (the notebook's ``pass_con``).

    Index 1 of the ModelListGP samples is the classifier's latent draw; the Bernoulli
    likelihood maps it to a feasibility probability. A tiny epsilon keeps ``log(prob)``
    finite inside qLogEI (this value is used with ``fat=None``, i.e. it IS the
    feasibility probability, not a signed constraint).
    """
    y_con = Z[..., 1]
    prob = model_con.likelihood(y_con).probs
    return prob + 1e-8


def _fit_models(train_X, train_obj, train_feas):
    """Fit the objective GP and the feasibility classifier, wrapped in a ModelListGP.

    The objective GP is a ``SingleTaskGP`` with input normalization + output
    standardization (BoCoDe's convention, matching ``constrained_ei``/``single_task_gp``);
    the feasibility model is the variational GP classifier trained with the ELBO. Output
    index 0 is the objective, index 1 the classifier.
    """
    dim = train_X.shape[-1]
    model_obj = SingleTaskGP(
        train_X,
        train_obj,
        input_transform=Normalize(d=dim),
        outcome_transform=Standardize(m=1),
    )
    robust_fit_mll(
        ExactMarginalLogLikelihood(model_obj.likelihood, model_obj), label="clf_cbo"
    )

    model_con = _FeasibilityClassifier(train_X, train_feas).double().to(train_X.device)
    mll_con = VariationalELBO(
        model_con.likelihood, model_con, num_data=train_feas.size(0)
    )
    robust_fit_mll(mll_con, label="clf_cbo")

    return ModelListGP(model_obj, model_con)


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Classifier-based constrained BO. ``n_init`` defaults to the dim-scaled BoCoDe default.

    The objective GP fit, the classifier training and the acquisition optimization all
    run on ``device`` (default cuda when available); the objective/constraints are
    evaluated on CPU. With ``checkpoint`` set the run is resumable: it restores
    ``(X, obj, constraints, completed_iters, RNG)`` and saves after every iteration.
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = ProblemObjective(problem)
    assert obj.num_constraints > 0, "clf_cbo requires a constrained problem"
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result("clf_cbo", type(problem).__name__, seed, acquisition_function="qLogEI")

    def best_feasible(obj_v, con_v):
        feas = (con_v <= 0).all(dim=1)
        return obj_v[feas].max().item() if feas.any() else -float("inf")

    if checkpoint and Path(checkpoint).exists():
        train_X, train_obj, start_it, data = load_checkpoint(checkpoint, res)
        train_con = torch.tensor(data["con"], dtype=DTYPE)
        best = best_feasible(train_obj, train_con)
    else:
        train_X = initial_design(n_init, obj.dim, seed)
        train_obj, train_con = obj.evaluate_raw(train_X)
        best = best_feasible(train_obj, train_con)
        res.start(best)
        start_it = 0

    bounds_dev = obj.bounds.to(dev)

    for it in range(start_it, iters):
        # Binary feasibility labels for the classifier: 1 = feasible (all c <= 0), 0 = not.
        feas = (train_con <= 0).all(dim=1)
        train_feas = feas.to(DTYPE)
        model = _fit_models(train_X.to(dev), train_obj.to(dev), train_feas.to(dev))

        # best_f is the best FEASIBLE objective seen so far. Before any feasible point
        # exists the notebook's masked max is undefined; fall back to the worst observed
        # objective so qLogEI stays positive everywhere and the classifier's feasibility
        # weighting steers the search toward the feasible region.
        best_f = train_obj[feas].max().item() if feas.any() else train_obj.min().item()

        acqf = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1024])),
            objective=GenericMCObjective(_pass_obj),
            constraints=[partial(_pass_con, model_con=model.models[1])],
            fat=[None],
        )
        candidate, _ = optimize_acqf(
            acqf, bounds=bounds_dev, q=1, num_restarts=10, raw_samples=512
        )
        candidate = candidate.detach().to(device="cpu", dtype=DTYPE)
        o, c = obj.evaluate_raw(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_obj = torch.cat([train_obj, o], dim=0)
        train_con = torch.cat([train_con, c], dim=0)
        best = max(best, best_feasible(o, c))
        res.record(best)
        if checkpoint:
            res.set_history(train_X, train_obj, n_init, c=train_con)
            save_checkpoint(
                checkpoint,
                train_X,
                train_obj,
                res,
                it + 1,
                extra={"con": train_con.cpu().numpy()},
            )
    res.set_history(train_X, train_obj, n_init, c=train_con)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument(
        "--init",
        type=int,
        default=None,
        help="initial design size (default: dim-scaled)",
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--checkpoint", default=None, help="resumable checkpoint .npz path"
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args),
        args.init,
        args.iters,
        args.seed,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
