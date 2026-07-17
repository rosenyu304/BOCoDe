"""TFM-UCB: tabular-foundation-model surrogate with the built-in quantile UCB.

A deliberately *plain* TabPFN baseline for mixed-variable BO -- no trust region, no
gradient subspace, no acquisition warping. Each iteration:

1. draw a Sobol candidate pool over the unit cube and **snap** its integer/categorical
   dimensions to their allowed values (the problem's ``resolved_variable_types`` grid),
   so every candidate is a valid mixed-variable configuration;
2. run a frozen, pretrained TabPFN in-context on the observed ``(X, y)`` (the penalized
   objective, so constraints fold into a single output);
3. score the pool with TabPFN's **built-in** ``BarDistribution.ucb`` -- the ``1 - rest_prob``
   quantile of the model's own non-Gaussian predictive distribution
   (``reg.predict(output_type="ucb")``), at the exploration level ``beta = 2.33`` -- and
   take the argmax.

This is the "TabPFN + quantile-UCB, no modifications" reference baseline: it isolates the
raw surrogate + acquisition from the trust-region / gradient machinery that GIT-BO and
TFM-TuRBO add, and it is cheap enough (one TabPFN forward per chunk, one column) to run on
a 24 GB card for the small mixed-variable problems.

Needs TabPFN >= v3 (see docs/tfm_setup.md). Run::

    python -m algorithms.single_obj_mixed_variable.tfm_ucb --problem AckleyCat --iters 50

Sources:
N. Hollmann, S. Müller, et al. TabPFN: accurate predictions on small data with a tabular foundation model. Nature, 2025.
"""

from __future__ import annotations

import argparse
import math
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
from .._tfm_utils import TabPFNSurrogate
from .single_task_gp import _discrete_grids, _snap

N_CANDIDATES = 2000  # candidate pool scored per iteration
MAX_CAND_PER_PASS = 512  # chunk width (peak memory scales with the chunk, not the pool)
BETA = 2.33  # UCB exploration level (matches GIT-BO Sec. 3.3)
# BarDistribution.ucb is a QUANTILE UCB: the (1 - rest_prob) quantile of the predictive.
# TabPFN's docstring gives the Gaussian-equivalent level beta = sqrt(2)*erfinv(2(1-rest_prob)-1);
# inverting at beta = 2.33 gives rest_prob = 1 - Phi(2.33).
REST_PROB = 0.5 * math.erfc(BETA / math.sqrt(2.0))


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    device: str = "auto",
    checkpoint: str | None = None,
) -> Result:
    """TFM-UCB over the unit cube (integer/categorical dims snapped to valid values)."""
    set_seed(seed)
    obj = ProblemObjective(problem)
    dim = obj.dim
    if n_init is None:
        n_init = default_n_init(dim)
    res = Result(
        "tfm_ucb",
        type(problem).__name__,
        seed,
        acquisition_function=f"quantile UCB (BarDistribution.ucb, beta={BETA})",
    )

    surrogate = TabPFNSurrogate(device=device)
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

    chunk = [min(N_CANDIDATES, MAX_CAND_PER_PASS)]

    def _score(cand: torch.Tensor):
        """UCB / mean / variance for a candidate chunk, from ONE TabPFN forward."""
        n_ctx = train_X.shape[0]
        X_full = torch.cat(
            [train_X.unsqueeze(1), cand.unsqueeze(1)], dim=0
        )  # (n_ctx+n_cand, 1, dim)
        Y_full = torch.cat(
            [train_Y, torch.zeros(cand.shape[0], 1, dtype=DTYPE)], dim=0
        ).unsqueeze(1)  # (n_ctx+n_cand, 1, 1)
        with torch.no_grad():
            logits = surrogate.forward(X_full, Y_full, single_eval_pos=n_ctx)
            ucb = surrogate.predict_ucb(logits, rest_prob=REST_PROB).reshape(-1)
            mean = surrogate.predict_mean(logits).reshape(-1)
            var = surrogate.predict_variance(logits).reshape(-1).clamp_min(0.0)
        return ucb.cpu(), mean.cpu(), var.cpu()

    def _chunked(cand: torch.Tensor):
        """Score the whole pool in slices of ``chunk[0]``; halve the chunk on CUDA OOM."""
        outs, i, n = [], 0, cand.shape[0]
        while i < n:
            c = min(chunk[0], n - i)
            try:
                outs.append(_score(cand[i : i + c]))
                i += c
            except torch.OutOfMemoryError:
                if chunk[0] == 1:
                    raise
                chunk[0] = max(1, chunk[0] // 2)
                if surrogate.device.type == "cuda":
                    torch.cuda.empty_cache()
        cat = lambda k: torch.cat([o[k] for o in outs])  # noqa: E731
        return cat(0), cat(1), cat(2)

    for it in range(start_it, iters):
        cand = _snap(sobol.draw(N_CANDIDATES).to(DTYPE), grids)
        ucb, mean, var = _chunked(cand)
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
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args),
        args.init,
        args.iters,
        args.seed,
        device=args.device,
        checkpoint=args.checkpoint,
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
