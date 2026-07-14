"""BAxUS: Bayesian optimization in Adaptively eXpanding nested random Subspaces.

High-dimensional BO that optimizes in a low-dimensional random subspace embedded in the
input space, and grows the subspace (splitting bins of input dimensions) whenever the
trust region collapses, carrying all observations forward. Combines a HeSBO-style sign
embedding with a TuRBO trust region + Thompson sampling. Ported from the BoTorch BAxUS
tutorial, which mirrors Papenmeier et al. (2022) Alg. 1-2.

Reference constants (BoTorch ``tutorials/baxus``; LeoIV/BAxUS ``baxus/baxus.py``):

* ``new_bins_on_split = 3``, initial target dim ``d_init`` chosen so that
  ``n_splits = round(log_{b+1}(D))`` splits land as close as possible to ``D``
* ``length_init = 0.8``, ``length_min = 0.5 ** 7``, ``length_max = 1.6``,
  ``success_tolerance = 3``
* failure tolerance ``max(1, min(ceil(split_budget / gamma), max(4, target_dim)))`` with
  ``gamma = 2 * log_0.5(length_min / length_init)`` and the split budget of Eq. (4) --
  the **official repo's** rule (``LeoIV/BAxUS``, ``baxus/baxus.py::failtol``), which
  floors the tolerance at 4; the BoTorch tutorial caps it at ``target_dim`` instead
* GP fit on standardized targets with noise constrained to ``[1e-8, 1e-3]``;
  candidates by Thompson sampling in a lengthscale-scaled trust region.

Internally BAxUS works in ``[-1, 1]`` (target and embedded input space); points are
mapped to the unit cube before the BoCoDe objective is evaluated.

``n_init`` defaults to BoCoDe's dimension-scaled :func:`default_n_init` so the method
gets the same initial-design budget as the rest of the suite (the reference uses
``target_dim + 1`` and the BoTorch tutorial 10; pass ``--init`` to reproduce those).

Run::

    python -m algorithms.single_obj.baxus --problem Ackley --init 10 --iters 100

Sources:
L. Papenmeier, L. Nardi, M. Poloczek. Increasing the Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces. NeurIPS 2022. https://arxiv.org/abs/2304.11468
Official implementation: https://github.com/LeoIV/BAxUS
BoTorch BAxUS tutorial: https://github.com/pytorch/botorch/blob/main/tutorials/baxus/baxus.ipynb
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import botorch
import gpytorch
import numpy as np
import torch
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
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


@dataclass
class BaxusState:
    dim: int
    eval_budget: int
    new_bins_on_split: int = 3
    d_init: int = field(default=-1)
    target_dim: int = field(default=-1)
    n_splits: int = field(default=-1)
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        n_splits = round(math.log(self.dim, self.new_bins_on_split + 1))
        self.d_init = 1 + int(
            np.argmin(
                np.abs(
                    (1 + np.arange(self.new_bins_on_split))
                    * (1 + self.new_bins_on_split) ** n_splits
                    - self.dim
                )
            )
        )
        self.target_dim = self.d_init
        self.n_splits = n_splits

    @property
    def split_budget(self) -> float:
        return (
            -1
            * (self.new_bins_on_split * self.eval_budget * self.target_dim)
            / (self.d_init * (1 - (self.new_bins_on_split + 1) ** (self.n_splits + 1)))
        )

    @property
    def failure_tolerance(self) -> int:
        """The **official** BAxUS failure tolerance (LeoIV/BAxUS ``baxus/baxus.py::failtol``).

        The tolerance is *floored* at 4 through ``ft_max = max(4, target_dim)``. The
        BoTorch tutorial instead *caps* at ``target_dim`` and so can return 1 -- which
        halves the trust region after a single failed evaluation, and BAxUS starts at
        ``d_init in {1, 2, 3}``. Rosen's call (2026-07-13): follow the official repo.
        """
        ft_max = max(4.0, float(self.target_dim))
        if self.target_dim == self.dim:
            return int(ft_max)
        gamma = 2 * math.log(self.length_min / self.length_init, 0.5)
        ft = math.ceil(self.split_budget / gamma)
        return int(max(1, min(ft, ft_max)))


def _embedding_matrix(input_dim: int, target_dim: int, device) -> torch.Tensor:
    if target_dim >= input_dim:
        return torch.eye(input_dim, dtype=DTYPE, device=device)
    perm = torch.randperm(input_dim, device=device) + 1
    bins = torch.nn.utils.rnn.pad_sequence(
        list(torch.tensor_split(perm, target_dim)), batch_first=True
    )
    mtrx = torch.zeros((target_dim, input_dim + 1), dtype=DTYPE, device=device)
    mtrx = mtrx.scatter_(
        1,
        bins,
        2 * torch.randint(2, (target_dim, input_dim), dtype=DTYPE, device=device) - 1,
    )
    return mtrx[:, 1:]


def _increase_embedding(S: torch.Tensor, X: torch.Tensor, n_new_bins: int):
    S_update = S.clone()
    X_update = X.clone()
    for row_idx in range(len(S)):
        row = S[row_idx]
        idxs = torch.nonzero(row)
        idxs = idxs[torch.randperm(len(idxs), device=S.device)].reshape(-1)
        if len(idxs) <= 1:
            continue
        non_zero = row[idxs].reshape(-1)
        n_row_bins = min(n_new_bins + 1, len(idxs))
        new_bins = torch.tensor_split(idxs + 1, n_row_bins)[1:]
        els = torch.tensor_split(non_zero, n_row_bins)[1:]
        new_bins_p = torch.nn.utils.rnn.pad_sequence(list(new_bins), batch_first=True)
        els_p = torch.nn.utils.rnn.pad_sequence(list(els), batch_first=True)
        S_stack = torch.zeros(
            (n_row_bins - 1, len(row) + 1), dtype=DTYPE, device=S.device
        )
        S_stack = S_stack.scatter_(1, new_bins_p, els_p)
        S_update[row_idx, torch.hstack(new_bins) - 1] = 0
        X_update = torch.hstack(
            (X_update, X[:, row_idx].reshape(-1, 1).repeat(1, len(new_bins)))
        )
        S_update = torch.vstack((S_update, S_stack[:, 1:]))
    return S_update, X_update


def _update_state(state: BaxusState, Y_next: torch.Tensor) -> BaxusState:
    if Y_next.max() > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    state.best_value = max(state.best_value, Y_next.max().item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def _create_candidate(state, model, X, Y, seed) -> torch.Tensor:
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.lengthscale.detach().view(-1)
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length, -1.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length, -1.0, 1.0)

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

    ts = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad():
        return ts(X_cand, num_samples=1)


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 100,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """BAxUS over the unit cube for a (high-dimensional) problem."""
    set_seed(seed)
    dev = resolve_device(device)
    obj = ProblemObjective(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result("baxus", type(problem).__name__, seed, acquisition_function="ts")

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
        train_Y = (Y - Y.mean()) / (Y.std() + 1e-9)
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3)).to(dev)
        model = SingleTaskGP(X_target, train_Y, likelihood=likelihood).to(dev)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with (
            botorch.settings.validate_input_scaling(False),
            gpytorch.settings.max_cholesky_size(2000),
        ):
            try:
                fit_gpytorch_mll(mll)
            except ModelFittingError:
                # Right after a split the covariance can be indefinite: fall back to Adam.
                opt = torch.optim.Adam(model.parameters(), lr=0.1)
                for _ in range(100):
                    opt.zero_grad()
                    loss = -mll(model(X_target), train_Y.flatten())
                    loss.backward()
                    opt.step()
            X_next_target = _create_candidate(
                state, model, X_target, train_Y, seed + it
            )

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


def _to_input(X_target: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Map target-space points through the embedding back to the unit cube."""
    return (X_target @ S).clamp(-1, 1).add(1).div(2)


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
