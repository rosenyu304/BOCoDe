"""Standard GP BO for single-objective problems (Xu et al., 2025).

"Standard Gaussian Process is All You Need for High-Dimensional Bayesian
Optimization" argues that a *standard* ARD Matern-5/2 GP is competitive with
specialized high-dimensional BO methods once its hyperparameter optimization is
started sensibly. The load-bearing choice is **Robust Initialization**: begin the
lengthscale optimization at ``l0 = sqrt(d)`` (Sec. 6.1; Lemma 5.1 shows the default
start of ``softplus(0) = 0.6931`` makes the marginal-likelihood gradient vanish in
high dimensions). The paper's main results use UCB with ``beta = 1.5``.

Note the paper's other finding: an RBF/SE kernel *fails* here — Matern is what makes
the standard GP work. The difference from ``vanilla_highdim_bo`` is the modeling choice
(initialization + Matern) rather than the acquisition-optimization tricks.

The reference implementation is ``XZT008/Standard-GP-is-all-you-need-for-HDBO``: the
paper's method is the ``GP_ARD_setls`` configuration, i.e. ``BO_loop_GP_MAP`` with
``set_ls=True, if_matern=True, ls_prior_type="Uniform", optim_type="LBFGS",
acqf_type="UCB", beta=1.5`` (``baselines/run_script.py``), whose model is
``GP_MAP_Wrapper`` in ``baselines/GP.py``. ``n_init`` there is 20 LHS points; BoCoDe
defaults to the dimension-scaled :func:`default_n_init` so every algorithm in the
suite gets the same initial-design budget (pass ``--init 20`` to reproduce the paper).

Run::

    python -m algorithms.single_obj.standard_gp --problem Rover --init 20 --iters 100
    python -m algorithms.single_obj.standard_gp --dataset P3HT --init 10 --iters 40

Sources:
Z. Xu, H. Wang, J. M. Phillips, and S. Zhe. Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization. ICLR 2025. https://openreview.net/forum?id=kX8h23UG6v
Official implementation: https://github.com/XZT008/Standard-GP-is-all-you-need-for-HDBO
M. Balandat et al. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, UniformPrior

from .._bo_utils import (
    DTYPE,
    DatasetObjective,
    ProblemObjective,
    Result,
    add_common_args,
    default_n_init,
    finalize,
    gp_stats,
    initial_design,  # noqa: E501
    load_checkpoint,
    make_problem,
    resolve_device,
    save_checkpoint,
    set_seed,
)

# Reference implementation (Xu et al., BO_loop.py:BO_loop_GP_MAP): UCB with beta=1.5,
# optimize_acqf(num_restarts=10, raw_samples=1000).
UCB_BETA = 1.5
NUM_RESTARTS = 10
RAW_SAMPLES = 1000


def _standard_gp(train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    """A standard ARD Matern-5/2 GP with the paper's Robust Initialization.

    The method of Xu et al. (2025) is a lengthscale *initialization*, not a prior:
    hyperparameter optimization starts from ``l0 = sqrt(dim)`` (Sec. 6.1, Lemma 5.1).
    GPyTorch otherwise starts at ``softplus(0) = 0.6931``, which is exactly the
    gradient-vanishing regime the paper diagnoses in high dimensions. The prior is a
    flat box (= MLE inside the box), widened for ``dim >= 100``, matching the reference
    implementation's ``GP_MAP_Wrapper``.
    """
    dim = train_X.shape[-1]
    # The reference's flat box is [1e-10, 10], widened to [1e-10, 30] for dim >= 100.
    # But the method itself *initializes* at l0 = sqrt(dim), and once dim > 900 that
    # initialization sits OUTSIDE the reference's own box: LassoLeukemia has dim = 7129,
    # so l0 = 84.4 > 30 and GPyTorch refused it outright ("Attempting to manually set a
    # parameter value that is out of bounds of its current constraints"). The reference
    # never ran a problem that large, so its two halves have never had to agree.
    # Widen the box to contain the initialization rather than clamping l0 down to 30:
    # clamping would silently discard the sqrt(d) Robust Initialization that IS the
    # method (Lemma 5.1), and it would do so precisely in the high-dimensional regime the
    # method exists for. NB this is an extrapolation beyond the published setting.
    #
    # l0 must land strictly INSIDE the box -- GPyTorch's Interval.inverse_transform is
    # infinite at either endpoint, so ub = l0 exactly still raises. The reference pins
    # ub = 3 * l0 at the dim = 100 boundary where it switches (l0 = 10, ub = 30), so
    # reuse that same 3:1 ratio when 30 is too small to hold l0. This leaves every
    # in-scope problem with sqrt(dim) < 30 on exactly the reference's ub = 30 -- only
    # LassoLeukemia (dim = 7129) is affected.
    ub = 30.0 if dim >= 100 else 10.0
    if math.sqrt(dim) >= ub:
        ub = 3.0 * math.sqrt(dim)
    # UniformPrior stores its bounds as plain tensor attributes, NOT registered buffers
    # (unlike GammaPrior), so the kernel's ``.to(train_X)`` below leaves them on CPU. When
    # a fit fails and BoTorch's fallback calls ``sample_all_priors``, the prior then samples
    # on CPU and assigns into the CUDA lengthscale -> a device-mismatch RuntimeError that
    # fires nondeterministically (only on the seeds whose L-BFGS fit needs the retry path).
    # Build the bounds on ``train_X``'s device so ``torch.distributions.Uniform.rsample``
    # (which draws with ``device=self.low.device``) lands the sample where the parameter is.
    ls_lb = torch.tensor(1e-10, device=train_X.device, dtype=train_X.dtype)
    ls_ub = torch.tensor(ub, device=train_X.device, dtype=train_X.dtype)
    base = MaternKernel(
        nu=2.5,
        ard_num_dims=dim,
        lengthscale_prior=UniformPrior(ls_lb, ls_ub),
        lengthscale_constraint=Interval(lower_bound=1e-10, upper_bound=ub),
    ).to(train_X)
    # *** the method: start the optimizer at l0 = sqrt(d) ***
    base._set_lengthscale(torch.full_like(base.lengthscale, math.sqrt(dim)))
    kernel = ScaleKernel(base, outputscale_prior=GammaPrior(2.0, 0.15)).to(train_X)
    unit_cube = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(train_X)
    model = SingleTaskGP(
        train_X=train_X.to(DTYPE),
        train_Y=train_Y.to(DTYPE),
        covar_module=kernel,
        # inputs already live in [0, 1]^d; do not re-learn the bounds from the data,
        # which would rescale the cube the sqrt(d) initialization is calibrated to.
        input_transform=Normalize(d=dim, bounds=unit_cube),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 100,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Continuous standard-GP BO over the unit cube.

    ``n_init`` defaults to the dimension-scaled BoCoDe default (:func:`default_n_init`).
    The GP fit and acquisition optimization run on ``device``; with ``checkpoint`` set
    the run is resumable per iteration.
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = ProblemObjective(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "standard_gp", type(problem).__name__, seed, acquisition_function="UCB"
    )

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, _ = load_checkpoint(checkpoint, res)
        best = train_Y.max().item()
    else:
        train_X = initial_design(n_init, obj.dim, seed)
        train_Y = obj(train_X)
        best = train_Y.max().item()
        res.start(best)
        start_it = 0

    bounds_dev = obj.bounds.to(dev)
    for it in range(start_it, iters):
        model = _standard_gp(train_X.to(dev), train_Y.to(dev))
        acqf = UpperConfidenceBound(model=model, beta=UCB_BETA)
        candidate, acq_value = optimize_acqf(
            acqf,
            bounds=bounds_dev,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )
        mean, var = gp_stats(model, candidate)
        candidate = candidate.detach().to(device="cpu", dtype=DTYPE)
        y = obj(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        best = max(best, y.item())
        res.record(best, mean=mean, variance=var, acq_value=acq_value.item())
        if checkpoint:
            res.set_history(train_X, train_Y, n_init)
            save_checkpoint(checkpoint, train_X, train_Y, res, it + 1)
    res.set_history(train_X, train_Y, n_init)
    return res


def optimize_dataset(
    dataset_problem, n_init: int = 10, iters: int = 50, seed: int = 0
) -> Result:
    """Discrete standard-GP BO over a candidate pool."""
    set_seed(seed)
    data = DatasetObjective(dataset_problem)
    res = Result(
        "standard_gp",
        type(dataset_problem).__name__,
        seed,
        acquisition_function="UCB",
    )

    perm = torch.randperm(data.n_candidates)
    observed = perm[:n_init].tolist()
    best = data.select(torch.tensor(observed)).max().item()
    res.start(best)

    for _ in range(iters):
        obs = torch.tensor(observed)
        model = _standard_gp(data.X[obs], data.Y[obs])
        mask = torch.ones(data.n_candidates, dtype=torch.bool)
        mask[obs] = False
        pool_idx = mask.nonzero(as_tuple=True)[0]
        if pool_idx.numel() == 0:
            break
        acqf = UpperConfidenceBound(model=model, beta=UCB_BETA)
        with torch.no_grad():
            scores = acqf(data.X[pool_idx].unsqueeze(1))
        choice = pool_idx[scores.argmax()].item()
        mean, var = gp_stats(model, data.X[choice].unsqueeze(0))
        observed.append(choice)
        best = max(best, data.select(torch.tensor(choice)).item())
        res.record(best, mean=mean, variance=var, acq_value=scores.max().item())
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem")
    group.add_argument("--dataset")
    parser.add_argument(
        "--init",
        type=int,
        default=None,
        help="initial design size (default: dim-scaled)",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--checkpoint", default=None, help="resumable checkpoint .npz path"
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    add_common_args(parser)
    args = parser.parse_args()

    if args.problem:
        res = optimize_problem(
            make_problem(args.problem, args),
            args.init,
            args.iters,
            args.seed,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    else:
        res = optimize_dataset(
            make_problem(args.dataset, args),
            args.init if args.init is not None else 10,
            args.iters,
            args.seed,
        )
    finalize(res, args)


if __name__ == "__main__":
    main()
