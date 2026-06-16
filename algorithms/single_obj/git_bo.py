"""GIT-BO: Gradient-Informed Bayesian Optimization with a tabular foundation model.

Uses a frozen, pretrained TabPFN as the surrogate (no GP training): at each step the
observed data are the model's in-context "training" rows and the candidates are query
rows scored by a bar-distribution acquisition (quantile-UCB by default). Candidates
are drawn inside the **gradient-informed active subspace** — the top eigenvectors of
the surrogate-gradient information matrix ``H = (1/n) sum_i (d mu/d x)(d mu/d x)^T`` —
which makes the method scale to high dimensions.

Two rank-selection rules for that subspace (``--rank``):
- ``5``      — fixed rank (the original GIT-BO default).
- ``marzouk`` — the **certified rank** r* = min{r : sum_{i>r} lambda_i <= 2*eps/kappa}
               on the trace-normalised spectrum (Zahm, Cui, Law, Spantini & Marzouk,
               Math. Comp. 2022), an error-bounded auto-selection of the rank.

Requires the TabPFN fork (see docs/tfm_setup.md). Run::

    python -m algorithms.single_obj.git_bo --problem Ackley --iters 50 --rank 5
    python -m algorithms.single_obj.git_bo --problem Ackley --iters 50 --rank marzouk

Sources:
S. Jiang et al. GIT-BO: high-dimensional Bayesian optimization with tabular foundation models. https://openreview.net/forum?id=9iTdKS4SRQ
N. Hollmann, S. Müller, et al. TabPFN: accurate predictions on small data with a tabular foundation model. Nature, 2025.
O. Zahm, T. Cui, K. Law, A. Spantini, Y. Marzouk. Certified dimension reduction in nonlinear Bayesian inverse problems. Mathematics of Computation 91:1789-1835, 2022.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.quasirandom import SobolEngine

import bocode

from .._bo_utils import (
    DTYPE,
    ProblemObjective,
    Result,
    add_common_args,
    finalize,
    initial_design,
    set_seed,
)
from .._tfm_utils import TabPFNSurrogate, sample_in_subspace

N_CANDIDATES = 2000  # candidate pool scored per iteration
UCB_REST_PROB = 0.05  # quantile-UCB upper-tail mass (95% quantile)


def optimize_problem(
    problem,
    n_init: int = 20,
    iters: int = 50,
    seed: int = 0,
    rank: str = "5",
    scale: float = 1.0,
    device: str = "auto",
) -> Result:
    """GIT-BO over the unit cube. ``rank`` is '5' (fixed) or 'marzouk' (certified)."""
    set_seed(seed)
    obj = ProblemObjective(problem)
    dim = obj.dim
    rank_mode = "marzouk" if str(rank).lower() == "marzouk" else "fixed"
    fixed_rank = 5 if rank_mode == "marzouk" else int(rank)
    res = Result(
        f"git_bo_{rank_mode if rank_mode == 'marzouk' else 'rank' + str(fixed_rank)}",
        type(problem).__name__,
        seed,
        acquisition_function="Quantile-UCB-95",
    )

    surrogate = TabPFNSurrogate(device=device)
    rng = np.random.default_rng(seed)
    sobol = SobolEngine(dim, scramble=True, seed=seed)

    train_X = initial_design(n_init, dim, seed)
    train_Y = obj(train_X)
    best = train_Y.max().item()
    res.start(best)

    grad_est = None
    for _it in range(iters):
        # candidate pool: gradient-informed subspace once we have a gradient estimate
        if grad_est is None:
            cand = sobol.draw(N_CANDIDATES).to(DTYPE)
        else:
            center = train_X[train_Y.argmax()].cpu().numpy()
            samples, _r = sample_in_subspace(
                center, grad_est, N_CANDIDATES, rank_mode, fixed_rank,
                eps=0.05, scale=scale, rng=rng,
            )
            cand = torch.from_numpy(samples).to(DTYPE)

        # build TabPFN inputs: context = observed, query = candidates (one batch col)
        n_ctx = train_X.shape[0]
        X_ctx = train_X.unsqueeze(1)  # (n_ctx, 1, dim)
        X_q = cand.unsqueeze(1).clone().requires_grad_(True)  # (n_cand, 1, dim)
        X_full = torch.cat([X_ctx.detach(), X_q], dim=0)
        Y_full = torch.cat(
            [train_Y, torch.zeros(N_CANDIDATES, 1, dtype=DTYPE)], dim=0
        ).unsqueeze(1)  # (n_ctx+n_cand, 1, 1)

        logits = surrogate.forward(X_full, Y_full, single_eval_pos=n_ctx)
        ucb = surrogate.predict_ucb(logits, train_Y.max(), UCB_REST_PROB)[n_ctx:].reshape(-1)
        mean = surrogate.predict_mean(logits)[n_ctx:].reshape(-1)

        # gradient of posterior mean wrt the candidates -> next subspace
        (grad_cand,) = torch.autograd.grad(mean.sum(), X_q, retain_graph=False)
        grad_est = -grad_cand.reshape(N_CANDIDATES, dim).detach().cpu().numpy()

        choice = int(torch.argmax(ucb).item())
        x_new = cand[choice : choice + 1]
        y_new = obj(x_new)
        train_X = torch.cat([train_X, x_new], dim=0)
        train_Y = torch.cat([train_Y, y_new], dim=0)
        best = max(best, y_new.item())
        res.record(
            best,
            mean=mean[choice].item(),
            variance=surrogate.predict_variance(logits)[n_ctx:].reshape(-1)[choice].item(),
            acq_value=ucb[choice].item(),
        )
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--init", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--rank", default="5", help="active-subspace rank: an integer (e.g. 5) or 'marzouk'"
    )
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        bocode.get_problem(args.problem)(), args.init, args.iters, args.seed,
        rank=args.rank, device=args.device,
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
