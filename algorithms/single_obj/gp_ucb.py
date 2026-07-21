"""SingleTaskGP + Upper Confidence Bound (UCB) Bayesian optimization.

The plain BoTorch SingleTaskGP baseline (input normalization + output standardization,
fit with the marginal log-likelihood) but with the acquisition swapped from Expected
Improvement to the **Upper Confidence Bound** ``mean + beta * std`` (Srinivas et al.,
2010). BoTorch's ``UpperConfidenceBound`` computes ``mean + sqrt(beta) * std``, so the
module constant :data:`BETA` (the coefficient on the posterior std) is squared before
being handed to BoTorch, making the effective acquisition ``mean + BETA * std``.

Run::

    python -m algorithms.single_obj.gp_ucb --problem Branin --init 10 --iters 40

Sources:
N. Srinivas, A. Krause, S. Kakade, and M. Seeger. Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design. ICML 2010. https://arxiv.org/abs/0912.3995
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. NeurIPS 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from .._bo_utils import (
    DTYPE,
    ProblemObjective,
    Result,
    add_common_args,
    default_n_init,
    finalize,
    fit_gp,
    gp_stats,
    initial_design,
    load_checkpoint,
    make_problem,
    resolve_device,
    save_checkpoint,
    set_seed,
)

BETA = 2.0  # coefficient on the posterior std in mean + BETA * std


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 40,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Continuous SingleTaskGP + UCB BO over the unit cube.

    ``n_init`` defaults to the dimension-scaled BoCoDe default (:func:`default_n_init`).
    The GP fit and acquisition optimization run on ``device`` (default: cuda if
    available, else cpu); the objective is evaluated on CPU. With ``checkpoint`` set,
    the run is resumable: it loads ``(X, y, completed_iters, RNG)`` from the checkpoint
    if present and saves it after every iteration.
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = ProblemObjective(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "gp_ucb", type(problem).__name__, seed, acquisition_function="UCB"
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
        model = fit_gp(train_X.to(dev), train_Y.to(dev))
        # BoTorch's UCB is mean + sqrt(beta) * std, so square BETA to get mean + BETA * std.
        acqf = UpperConfidenceBound(model=model, beta=BETA**2)
        candidate, acq_value = optimize_acqf(
            acqf, bounds=bounds_dev, q=1, num_restarts=10, raw_samples=512
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument(
        "--init",
        type=int,
        default=None,
        help="initial design size (default: dim-scaled)",
    )
    parser.add_argument("--iters", type=int, default=40)
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
