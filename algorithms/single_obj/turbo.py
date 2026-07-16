"""TuRBO: Trust-Region Bayesian Optimization (single trust region, TuRBO-1).

Implements the trust-region BO of Eriksson et al., 2019. A local GP is fit inside
an axis-aligned trust region centered on the best observation; the region is
rescaled by the GP's ARD lengthscales, grows after ``succtol`` consecutive
successes, shrinks after ``failtol`` failures, and the run restarts from a fresh
initial design when the region collapses below ``length_min``. Candidates are
drawn from a scrambled Sobol sequence inside the region with the TuRBO
perturbation mask, and the batch is picked by **Thompson sampling** on the GP
posterior (the paper's acquisition; §2 / Alg. 1).

Reference constants (uber-research/TuRBO ``turbo/turbo_1.py`` L105-113 and the
BoTorch tutorial ``tutorials/turbo_1``):

* ``length_init = 0.8``, ``length_min = 0.5 ** 7``, ``length_max = 1.6``
* ``succtol = 3`` (the paper's value; the BoTorch tutorial uses 10)
* ``failtol = ceil(max(4 / batch, dim / batch))``
* success test ``f_next > f_best + 1e-3 * |f_best|`` (maximization)
* ``n_cand = min(5000, max(2000, 200 * dim))`` (BoTorch tutorial)
* GP: ScaleKernel(Matern-5/2, ARD, lengthscale in [0.005, 4.0]) with a
  GaussianLikelihood noise in [1e-8, 1e-3], fit on standardized targets.

``n_init`` defaults to BoCoDe's dimension-scaled :func:`default_n_init` so the
method gets the same initial-design budget as the other algorithms in the suite
(the paper itself uses ``n_init = 2 * dim``; pass ``--init`` to reproduce that).

Run::

    python -m algorithms.single_obj.turbo --problem Rover --init 20 --iters 100
    python -m algorithms.single_obj.turbo --dataset CrossedBarrel --init 10 --iters 40

Sources:
D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek. Scalable Global Optimization via Local Bayesian Optimization. NeurIPS 2019. https://arxiv.org/abs/1910.01739
Official implementation: https://github.com/uber-research/TuRBO (turbo/turbo_1.py, turbo/gp.py)
BoTorch TuRBO tutorial: https://github.com/pytorch/botorch/blob/main/tutorials/turbo_1/turbo_1.ipynb
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import gpytorch
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from .._bo_utils import (
    DTYPE,
    DatasetObjective,
    ProblemObjective,
    Result,
    add_common_args,
    default_n_init,
    finalize,
    initial_design,  # noqa: E501
    load_checkpoint,
    make_problem,
    resolve_device,
    save_checkpoint,
    set_seed,
)

MAX_CHOLESKY_SIZE = float("inf")  # always use Cholesky (BoTorch tutorial)


@dataclass
class TrustRegion:
    """State of the single TuRBO trust region (turbo_1.py L105-113, L137-150)."""

    dim: int
    batch_size: int = 1
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 3  # the paper's succtol
    restart: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )

    def update(self, improved: bool) -> None:
        if improved:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1
        if self.success_counter == self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:
            self.length /= 2.0
            self.failure_counter = 0
        if self.length < self.length_min:
            self.restart = True


def _fit_turbo_gp(train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    """The TuRBO local GP: ARD Matern-5/2 with the paper's hyperparameter bounds.

    Targets are standardized (as in turbo_1.py ``_create_candidates`` and the
    BoTorch tutorial) so the [1e-8, 1e-3] noise interval is on the right scale;
    inputs already live in the unit cube.
    """
    dim = train_X.shape[-1]
    Y = (train_Y - train_Y.mean()) / train_Y.std().clamp_min(1e-9)
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3)).to(train_X)
    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
        )
    )
    model = SingleTaskGP(train_X, Y, covar_module=covar_module, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
        try:
            fit_gpytorch_mll(mll)
        except ModelFittingError:
            # A GP fit can genuinely fail on pathological data -- most often here when TuRBO's trust
            # region has collapsed and the candidate points are near-duplicate, so the covariance is
            # (near-)singular and the MLL optimisation diverges. Crashing loses every iteration already
            # computed and, on a cluster, looks like an infrastructure failure when it is really a
            # property of the local data. Fall back to the model's PRIOR (un-optimised) hyperparameters
            # -- a valid GP -- so the run continues; the trust region then registers a failure and
            # shrinks/restarts on a fresh design, which is the correct recovery. Mirrors scbo/baxus.
            warnings.warn(
                "turbo: GP fit failed (likely a collapsed trust region); "
                "falling back to prior hyperparameters for this iteration.",
                RuntimeWarning,
                stacklevel=2,
            )
    return model


def _generate_batch(
    tr: TrustRegion,
    model: SingleTaskGP,
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    """Thompson-sample ``batch_size`` points from the trust region around the best X.

    The region's side lengths are proportional to the GP's ARD lengthscales
    (normalized to a geometric mean of 1), as in turbo_1.py L182-186.
    """
    dim = X.shape[-1]
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.detach().view(-1)
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * tr.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * tr.length / 2.0, 0.0, 1.0)

    n_candidates = min(5000, max(2000, 200 * dim))
    sobol = SobolEngine(dim, scramble=True, seed=seed)
    pert = sobol.draw(n_candidates).to(X)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, device=X.device, dtype=X.dtype) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim, size=(len(ind),), device=X.device)] = True

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad(), gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
        return thompson_sampling(X_cand, num_samples=batch_size)


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 100,
    seed: int = 0,
    batch: int = 1,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Continuous TuRBO-1 over the unit cube for a problem.

    ``iters`` is the number of BO evaluations after the initial design; the
    initial designs of any *restarts* are drawn from the same budget (as in the
    reference, where a restart's initial design consumes evaluations).
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = ProblemObjective(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result("turbo", type(problem).__name__, seed, acquisition_function="ts")

    def restart_design(restart_idx: int):
        # A fresh space-filling design per restart (turbo_1.py L245-250 draws a new
        # Latin hypercube on every restart); a fixed seed would resample the same points.
        X = initial_design(n_init, obj.dim, seed + 9973 * restart_idx)
        return X, obj(X)

    if checkpoint and Path(checkpoint).exists():
        tr_X, tr_Y, evals, data = load_checkpoint(checkpoint, res)
        hist_X = torch.tensor(data["hist_X"], dtype=DTYPE)
        hist_Y = torch.tensor(data["hist_Y"], dtype=DTYPE)
        tr = TrustRegion(dim=obj.dim, batch_size=batch)
        tr.length = float(data["tr_length"])
        tr.success_counter = int(data["tr_success"])
        tr.failure_counter = int(data["tr_failure"])
        n_restarts = int(data["n_restarts"])
        best = hist_Y.max().item()
    else:
        tr_X, tr_Y = restart_design(0)
        hist_X, hist_Y = tr_X.clone(), tr_Y.clone()
        tr = TrustRegion(dim=obj.dim, batch_size=batch)
        n_restarts = 0
        evals = 0
        best = tr_Y.max().item()
        res.start(best)

    while evals < iters:
        model = _fit_turbo_gp(tr_X.to(dev), tr_Y.to(dev))
        cand = _generate_batch(
            tr, model, tr_X.to(dev), tr_Y.to(dev), batch, seed + evals
        )
        cand = cand.detach().to(device="cpu", dtype=DTYPE)

        y = obj(cand)
        # Success test against the current trust region's incumbent (turbo_1.py L138).
        f_best = tr_Y.max().item()
        improved = y.max().item() > f_best + 1e-3 * math.fabs(f_best)
        tr_X = torch.cat([tr_X, cand], dim=0)
        tr_Y = torch.cat([tr_Y, y], dim=0)
        hist_X = torch.cat([hist_X, cand], dim=0)
        hist_Y = torch.cat([hist_Y, y], dim=0)
        tr.update(improved)

        for j in range(cand.shape[0]):
            best = max(best, y[j].item())
            res.record(best)
            evals += 1
            if evals >= iters:
                break

        if tr.restart and evals < iters:
            n_restarts += 1
            tr_X, tr_Y = restart_design(n_restarts)
            hist_X = torch.cat([hist_X, tr_X], dim=0)
            hist_Y = torch.cat([hist_Y, tr_Y], dim=0)
            tr = TrustRegion(dim=obj.dim, batch_size=batch)
            for j in range(n_init):
                best = max(best, tr_Y[j].item())
                res.record(best)
                evals += 1
                if evals >= iters:
                    break

        if checkpoint:
            res.set_history(hist_X, hist_Y, n_init)
            save_checkpoint(
                checkpoint,
                tr_X,
                tr_Y,
                res,
                evals,
                extra=dict(
                    hist_X=hist_X.numpy(),
                    hist_Y=hist_Y.numpy(),
                    tr_length=tr.length,
                    tr_success=tr.success_counter,
                    tr_failure=tr.failure_counter,
                    n_restarts=n_restarts,
                ),
            )
    res.set_history(hist_X, hist_Y, n_init)
    return res


def optimize_dataset(
    dataset_problem, n_init: int = 10, iters: int = 50, seed: int = 0
) -> Result:
    """Discrete TuRBO over a candidate pool: Thompson-sample among the pool points
    that fall inside the trust region."""
    set_seed(seed)
    data = DatasetObjective(dataset_problem)
    res = Result(
        "turbo", type(dataset_problem).__name__, seed, acquisition_function="ts"
    )

    perm = torch.randperm(data.n_candidates)
    observed = perm[:n_init].tolist()
    best = data.select(torch.tensor(observed)).max().item()
    res.start(best)

    tr = TrustRegion(dim=data.dim)
    for _ in range(iters):
        obs = torch.tensor(observed)
        model = _fit_turbo_gp(data.X[obs], data.Y[obs])
        x_center = data.X[obs][data.Y[obs].argmax()]
        weights = model.covar_module.base_kernel.lengthscale.detach().view(-1)
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        within = ((data.X - x_center).abs() <= weights * tr.length / 2.0).all(dim=1)
        mask = within.clone()
        mask[obs] = False
        pool_idx = mask.nonzero(as_tuple=True)[0]
        if pool_idx.numel() == 0:  # region empty: fall back to all unobserved
            mask = torch.ones(data.n_candidates, dtype=torch.bool)
            mask[obs] = False
            pool_idx = mask.nonzero(as_tuple=True)[0]
            if pool_idx.numel() == 0:
                break

        with torch.no_grad():  # Thompson sampling over the candidate pool
            sample = model.posterior(data.X[pool_idx]).rsample(torch.Size([1]))
        choice = pool_idx[sample.reshape(-1).argmax()].item()

        y = data.select(torch.tensor(choice)).item()
        f_best = data.Y[obs].max().item()
        tr.update(y > f_best + 1e-3 * math.fabs(f_best))
        observed.append(choice)
        best = max(best, y)
        res.record(best)
        if tr.restart:
            tr = TrustRegion(dim=data.dim)
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
    parser.add_argument("--batch", type=int, default=1, help="TuRBO batch size q")
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
            batch=args.batch,
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
