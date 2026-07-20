"""TabICL-BAxUS: BAxUS nested-subspace embedding with a TabICL surrogate.

BAxUS (Papenmeier et al., 2022) with **all of the nested-subspace machinery kept** -- the
HeSBO-style sign embedding, ``new_bins_on_split = 3``, the dimension-scaled initial target
dim, ``length_init = 0.8`` / ``length_min = 0.5**7`` / ``length_max = 1.6``,
``success_tolerance = 3``, the official failure-tolerance rule, and the split-on-collapse that
grows the subspace and carries every observation forward -- but with the **GP replaced by a
frozen, pretrained TabICL regressor**. This is the TabICL counterpart of ``baxus`` and the
sibling of ``tabicl_turbo`` in the embedded target space.

The BAxUS state machine, the embedding matrix, and the subspace-splitting logic are imported
from :mod:`algorithms.single_obj.baxus`, so the two methods share one implementation of them by
construction. Only the surrogate + the candidate acquisition change.

**Two BoCoDe design choices** (as in ``tabicl_turbo``), both because a PFN-style surrogate has
no GP posterior:

1. **Thompson sampling.** BAxUS scores its trust-region candidates with
   ``MaxPosteriorSampling`` on a GP posterior. TabICL gives a marginal predictive per row and no
   joint posterior, so we draw **independent samples from each candidate's own predictive**
   (``TabICLSurrogate.sample_from_quantiles``) and take the argmax.
2. **Isotropic trust region.** BAxUS weights each side of the trust region by the GP's ARD
   lengthscale; TabICL has no lengthscales, so the region is an isotropic cube -- the same
   honest choice ``tabicl_turbo`` / ``tfm_scbo`` make.

BAxUS internally works in ``[-1, 1]`` (target and embedded input space); points are mapped to
the unit cube before the BoCoDe objective is evaluated. TabICL is fit directly on the
target-space points (it standardizes its inputs internally).

Needs TabICL (``pip install tabicl``). Run::

    python -m algorithms.single_obj.tabicl_baxus --problem Ackley --init 10 --iters 100

Sources:
L. Papenmeier, L. Nardi, M. Poloczek. Increasing the Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces. NeurIPS 2022. https://arxiv.org/abs/2304.11468 (embedding, split logic, constants; official code https://github.com/LeoIV/BAxUS)
J. Qu, D. Holzmüller, G. Varoquaux, M. Le Morvan. TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. ICML 2025. https://github.com/soda-inria/tabicl (the surrogate)
The GP -> TabICL substitution and the independent-marginal Thompson sampling are BoCoDe's, with no paper behind them.
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
    load_checkpoint,
    make_problem,
    resolve_device,
    save_checkpoint,
    set_seed,
)
from .._tabicl_utils import TabICLSurrogate
from .baxus import (
    BaxusState,
    _embedding_matrix,
    _increase_embedding,
    _to_input,
    _update_state,
)


def _create_candidate(
    surrogate: TabICLSurrogate,
    state: BaxusState,
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """BAxUS's trust-region candidate set, scored by TabICL Thompson sampling.

    Isotropic trust region in the embedded target space (``x_center +/- L``, clamped to
    ``[-1, 1]``): TabICL has no ARD lengthscales to weight the sides by, so unlike the GP
    ``baxus`` this region is a cube.
    """
    x_center = X[Y.argmax(), :].clone()
    tr_lb = torch.clamp(x_center - state.length, -1.0, 1.0)
    tr_ub = torch.clamp(x_center + state.length, -1.0, 1.0)

    dim = X.shape[-1]
    n_candidates = min(5000, max(2000, 200 * dim))
    sobol = SobolEngine(dim, scramble=True, seed=seed)
    pert = tr_lb + (tr_ub - tr_lb) * sobol.draw(n_candidates).to(X)
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=X.dtype, device=X.device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim, size=(len(ind),), device=X.device)] = True
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    with torch.no_grad():
        surrogate.fit(X, Y.reshape(-1, 1))
        _, _, q = surrogate.score(X_cand)
        draw = surrogate.sample_from_quantiles(q).reshape(-1)
    choice = int(draw.argmax())
    return X_cand[choice : choice + 1]


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 100,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """TabICL-BAxUS over the unit cube for a (high-dimensional) problem."""
    set_seed(seed)
    dev = resolve_device(device)
    obj = ProblemObjective(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result("tabicl_baxus", type(problem).__name__, seed, acquisition_function="ts")

    surrogate = TabICLSurrogate(device=str(dev), seed=seed)

    def evaluate(X_input: torch.Tensor) -> torch.Tensor:
        # BAxUS works in [-1, 1]; map to the unit cube for the objective (on CPU).
        X_unit = (X_input.detach().cpu().to(DTYPE).clamp(-1.0, 1.0) + 1.0) / 2.0
        return obj(X_unit).to(dev)

    state = BaxusState(dim=obj.dim, eval_budget=iters)

    if checkpoint and Path(checkpoint).exists():
        X_target, Y, start_it, data = load_checkpoint(checkpoint, res)
        X_target, Y = X_target.to(dev), Y.to(dev)
        S = torch.tensor(data["S"], dtype=DTYPE, device=dev)
        state.target_dim = int(data["target_dim"])
        state.length = float(data["length"])
        state.success_counter = int(data["success_counter"])
        state.failure_counter = int(data["failure_counter"])
        state.best_value = float(data["best_value"])
    else:
        S = _embedding_matrix(obj.dim, state.d_init, dev)
        sobol = SobolEngine(state.d_init, scramble=True, seed=seed)
        X_target = 2.0 * sobol.draw(n_init).to(dtype=DTYPE, device=dev) - 1.0
        Y = evaluate(X_target @ S)
        state.best_value = Y.max().item()
        res.start(state.best_value)
        start_it = 0

    for it in range(start_it, iters):
        X_next_target = _create_candidate(surrogate, state, X_target, Y, seed + it)

        y_next = evaluate(X_next_target @ S)
        state = _update_state(state, y_next)
        X_target = torch.cat([X_target, X_next_target], dim=0)
        Y = torch.cat([Y, y_next], dim=0)
        res.record(state.best_value)

        if state.restart_triggered:
            state.restart_triggered = False
            S, X_target = _increase_embedding(S, X_target, state.new_bins_on_split)
            state.target_dim = len(S)
            state.length = state.length_init
            state.failure_counter = 0
            state.success_counter = 0

        if checkpoint:
            res.set_history(_to_input(X_target, S), Y, n_init)
            save_checkpoint(
                checkpoint,
                X_target,
                Y,
                res,
                it + 1,
                extra=dict(
                    S=S.detach().cpu().numpy(),
                    target_dim=state.target_dim,
                    length=state.length,
                    success_counter=state.success_counter,
                    failure_counter=state.failure_counter,
                    best_value=state.best_value,
                ),
            )

    res.set_history(_to_input(X_target, S), Y, n_init)
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
    parser.add_argument("--iters", type=int, default=100)
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
