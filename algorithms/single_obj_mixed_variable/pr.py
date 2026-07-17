"""PR: Probabilistic Reparameterization for mixed-variable BO (Daulton et al., 2022).

Instead of maximizing an acquisition function ``alpha(x, z)`` *directly* over the
discrete/categorical block ``z`` (where gradients do not pass through ``z``), PR puts a
**probability distribution** ``p(z | theta)`` over each discrete variable, parameterized
by *continuous* logits ``theta``, and maximizes the differentiable **probabilistic
objective** (Eq. 1-2)

    E_{z ~ p(z | theta)}[ alpha(x, z) ]

jointly over ``theta`` and the continuous variables ``x`` by Monte-Carlo gradient ascent.
The continuous ``x`` gets a pathwise gradient; the discrete ``theta`` gets the
**score-function / REINFORCE** estimator (Eq. 7)

    grad_theta E[alpha] = E[ alpha(x, z) grad_theta log p(z | theta) ]

with a batch-mean control-variate baseline (Eq. 8) to tame its variance. Both are
obtained from one surrogate loss ``(alpha - b).detach() * log p + alpha``. After the
ascent, the **mode** of ``p(z | theta)`` gives the discrete configuration and ``x`` the
continuous part; the point is snapped to the valid grid and evaluated. By Theorem 1 the
PO maximizer is an ``alpha`` maximizer, so PR inherits the acquisition's regret.

Reparameterizations (Table 2): a **softmax categorical** over the grid values for
unordered categoricals, and an **ordinal** ``Z = floor(theta) + Bernoulli(theta - floor)``
over two adjacent levels for integer dimensions. Each iteration fits a ``SingleTaskGP`` on
the observed ``(X_unit, penalized y)`` and uses ``LogExpectedImprovement`` as ``alpha``.
With no discrete dimensions the loop degenerates to standard EI gradient ascent.

This is the GP-based PR reference baseline (the classic Daulton setup; the TabPFN angle is
covered by the sibling ``tfm_ucb``). Run::

    python -m algorithms.single_obj_mixed_variable.pr --problem AckleyCat --iters 50

Sources:
S. Daulton, X. Wan, D. Eriksson, M. Balandat, M. A. Osborne, and E. Bakshy. Bayesian Optimization over Discrete and Mixed Spaces via Probabilistic Reparameterization. NeurIPS 35, 2022. https://arxiv.org/abs/2210.10199
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from botorch.acquisition import LogExpectedImprovement

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
    save_checkpoint,
    set_seed,
)
from .single_task_gp import _discrete_grids, _snap

N_MC = 128  # Monte-Carlo samples for the probabilistic-objective gradient (Eq. 7)
N_RESTARTS = 4  # random (x, theta) starts for the inner acquisition optimization
N_STEPS = 100  # Adam ascent steps per restart
LR = 0.05  # Adam step size


class _Categorical:
    """Softmax categorical over the ``K`` grid values of one unordered discrete dim."""

    def __init__(self, values: list[float]):
        self.values = torch.tensor(values, dtype=DTYPE)  # unit-cube grid values
        self.K = len(values)

    def init_params(self) -> torch.Tensor:
        return 0.01 * torch.randn(self.K, dtype=DTYPE)  # near-uniform logits

    def sample(self, logits: torch.Tensor, n: int) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, n, replacement=True)

    def log_prob(self, logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(logits, dim=-1)[idx]

    def mode(self, logits: torch.Tensor) -> int:
        return int(torch.argmax(logits).item())

    def clamp_(self, logits: torch.Tensor) -> None:
        pass  # softmax logits are unconstrained


class _Ordinal:
    """Ordinal ``Z = floor(theta) + Bernoulli(theta - floor)`` over adjacent levels.

    ``theta`` is a single continuous parameter in ``[0, K-1]`` (index space); the grid
    values are consecutive integers, so index adjacency is integer adjacency.
    """

    def __init__(self, values: list[float]):
        self.values = torch.tensor(values, dtype=DTYPE)  # unit-cube grid values
        self.K = len(values)

    def init_params(self) -> torch.Tensor:
        return (self.K - 1) * torch.rand(1, dtype=DTYPE)

    def _floor_frac(self, theta: torch.Tensor):
        floor = torch.floor(theta.detach()).clamp(0, self.K - 2)
        return floor, (theta - floor)  # frac carries the gradient (floor is detached)

    def sample(self, theta: torch.Tensor, n: int) -> torch.Tensor:
        floor, frac = self._floor_frac(theta)
        b = torch.bernoulli(frac.detach().clamp(0, 1).expand(n))
        return (floor + b).long()

    def log_prob(self, theta: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        floor, frac = self._floor_frac(theta)
        frac = frac.clamp(1e-6, 1 - 1e-6)
        return torch.where(idx == floor.long(), torch.log1p(-frac), torch.log(frac))

    def mode(self, theta: torch.Tensor) -> int:
        return int(theta.round().clamp(0, self.K - 1).item())

    def clamp_(self, theta: torch.Tensor) -> None:
        theta.clamp_(0.0, self.K - 1)


def _make_dists(problem):
    """Continuous-dim indices and the (dim, distribution) list for the discrete dims."""
    types = problem.resolved_variable_types()
    grids = _discrete_grids(problem)
    cont_dims = [i for i, t in enumerate(types) if t == "continuous"]
    discrete = [
        (i, _Ordinal(grids[i]) if types[i] == "integer" else _Categorical(grids[i]))
        for i in sorted(grids)
    ]
    return cont_dims, discrete


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    device: str = "auto",
    checkpoint: str | None = None,
) -> Result:
    """Probabilistic-reparameterization BO over the mixed space (GP surrogate + LogEI)."""
    set_seed(seed)
    obj = ProblemObjective(problem)
    dim = obj.dim
    if n_init is None:
        n_init = default_n_init(dim)
    res = Result(
        "pr",
        type(problem).__name__,
        seed,
        acquisition_function="EI (probabilistic reparameterization)",
    )

    grids = _discrete_grids(problem)
    cont_dims, discrete = _make_dists(problem)
    cont_set = set(cont_dims)
    n_cont = len(cont_dims)
    n_mc = N_MC if discrete else 1  # no discrete dims -> plain EI (one sample suffices)

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, _ = load_checkpoint(checkpoint, res)
        best = train_Y.max().item()
    else:
        train_X = _snap(initial_design(n_init, dim, seed), grids)
        train_Y = obj(train_X)
        best = train_Y.max().item()
        res.start(best)
        start_it = 0

    def _assemble(cont_block, disc_cols, n):
        """Stack continuous + discrete columns into an ``(n, dim)`` candidate batch."""
        columns, ci = [], 0
        for j in range(dim):
            if j in cont_set:
                columns.append(cont_block[:, ci])
                ci += 1
            else:
                columns.append(disc_cols[j])
        return torch.stack(columns, dim=1)

    def _optimize_acq(model):
        """Maximize E[LogEI] over (x, theta); return the best mode candidate + its acq."""
        acqf = LogExpectedImprovement(model=model, best_f=train_Y.max())
        best_cand, best_acq = None, -float("inf")
        for _ in range(N_RESTARTS):
            x = torch.rand(n_cont, dtype=DTYPE, requires_grad=n_cont > 0)
            params = [d.init_params().requires_grad_(True) for _, d in discrete]
            opt = torch.optim.Adam(([x] if n_cont else []) + params, lr=LR)
            for _ in range(N_STEPS):
                opt.zero_grad()
                idx = [
                    d.sample(p, n_mc)
                    for (_, d), p in zip(discrete, params, strict=False)
                ]
                cont_block = x.unsqueeze(0).expand(n_mc, -1)
                disc_cols = {
                    dim_i: d.values[j]
                    for (dim_i, d), j in zip(discrete, idx, strict=False)
                }
                cand = _assemble(cont_block, disc_cols, n_mc)
                acq = acqf(cand.unsqueeze(1))  # (n_mc,)
                logp = sum(
                    (
                        d.log_prob(p, j)
                        for (_, d), p, j in zip(discrete, params, idx, strict=False)
                    ),
                    torch.zeros(n_mc, dtype=DTYPE),
                )
                baseline = acq.detach().mean()
                surrogate = ((acq.detach() - baseline) * logp + acq).mean()
                (-surrogate).backward()
                opt.step()
                with torch.no_grad():
                    if n_cont:
                        x.clamp_(0.0, 1.0)
                    for (_, d), p in zip(discrete, params, strict=False):
                        d.clamp_(p)
            # mode configuration for this restart, scored deterministically
            with torch.no_grad():
                mode_cols = {
                    dim_i: d.values[d.mode(p)].reshape(1)
                    for (dim_i, d), p in zip(discrete, params, strict=False)
                }
                mode_cand = _assemble(x.detach().reshape(1, -1), mode_cols, 1)
                a = acqf(mode_cand.unsqueeze(1)).item()
            if a > best_acq:
                best_cand, best_acq = mode_cand, a
        return best_cand, best_acq

    for it in range(start_it, iters):
        model = fit_gp(train_X, train_Y)
        candidate, acq_value = _optimize_acq(model)
        candidate = _snap(candidate, grids)
        mean, var = gp_stats(model, candidate)
        y = obj(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        best = max(best, y.item())
        res.record(best, mean=mean, variance=var, acq_value=acq_value)
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
