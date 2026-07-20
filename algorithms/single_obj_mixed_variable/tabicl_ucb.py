"""TabICL-UCB: TabICL surrogate with a quantile UCB acquisition.

A deliberately *plain* TabICL baseline for mixed-variable BO -- no trust region, no
gradient subspace, no acquisition warping (the TabICL counterpart of ``tfm_ucb``). Each
iteration:

1. draw a Sobol candidate pool over the unit cube and **snap** its integer/categorical
   dimensions to their allowed values (the problem's ``resolved_variable_types`` grid),
   so every candidate is a valid mixed-variable configuration;
2. condition a frozen, pretrained TabICL regressor in-context on the observed ``(X, y)``
   (the penalized objective, so constraints fold into a single output);
3. score the pool with a **quantile UCB** -- the ``Phi(beta)`` quantile of TabICL's own
   predictive distribution, at the exploration level ``beta = 2.33`` -- and take the
   argmax.

This is the "TabICL + quantile-UCB, no modifications" reference baseline, mirroring
``tfm_ucb`` with TabICL swapped in for TabPFN.

Needs TabICL (``pip install tabicl``). Run::

    python -m algorithms.single_obj_mixed_variable.tabicl_ucb --problem AckleyCat --iters 50

Sources:
J. Qu, D. Holzmüller, G. Varoquaux, M. Le Morvan. TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. ICML 2025. https://github.com/soda-inria/tabicl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.quasirandom import SobolEngine

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
    save_checkpoint,
    set_seed,
)
from .._tabicl_utils import TabICLSurrogate
from .single_task_gp import _discrete_grids, _snap

N_CANDIDATES = 2000  # candidate pool scored per iteration
BETA = 2.33          # UCB exploration level (matches GIT-BO Sec. 3.3 / tfm_ucb)


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    device: str = "auto",
    checkpoint: str | None = None,
) -> Result:
    """TabICL-UCB over the unit cube (integer/categorical dims snapped to valid values)."""
    set_seed(seed)
    obj = ProblemObjective(problem)
    dim = obj.dim
    if n_init is None:
        n_init = default_n_init(dim)
    res = Result(
        "tabicl_ucb",
        type(problem).__name__,
        seed,
        acquisition_function=f"quantile UCB (TabICL predictive, beta={BETA})",
    )

    surrogate = TabICLSurrogate(device=device, seed=seed)
    grids = _discrete_grids(problem)
    sobol = SobolEngine(dim, scramble=True, seed=seed)

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, _ = load_checkpoint(checkpoint, res)
        best = train_Y.max().item()
    else:
        train_X = _snap(initial_design(n_init, dim, seed), grids)
        train_Y = obj(train_X)
        best = train_Y.max().item()
        res.start(best)
        start_it = 0

    for it in range(start_it, iters):
        cand = _snap(sobol.draw(N_CANDIDATES).to(DTYPE), grids)
        surrogate.fit(train_X, train_Y)
        mean, var, q = surrogate.score(cand)
        ucb = surrogate.ucb_from_quantiles(q, BETA)
        choice = int(torch.argmax(ucb).item())
        x_new = cand[choice : choice + 1]
        y_new = obj(x_new)
        train_X = torch.cat([train_X, x_new], dim=0)
        train_Y = torch.cat([train_Y, y_new], dim=0)
        best = max(best, y_new.item())
        res.record(
            best,
            mean=mean[choice].item(),
            variance=var[choice].item(),
            acq_value=ucb[choice].item(),
        )
        if checkpoint:
            res.set_history(train_X, train_Y, n_init)
            save_checkpoint(checkpoint, train_X, train_Y, res, it + 1)
    res.set_history(train_X, train_Y, n_init)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--init", type=int, default=None,
                        help="initial design size (default: dim-scaled)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--checkpoint", default=None, help="resumable checkpoint .npz path")
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args), args.init, args.iters, args.seed,
        device=args.device, checkpoint=args.checkpoint,
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
