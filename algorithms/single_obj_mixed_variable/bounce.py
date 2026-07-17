"""Bounce: trust-region BO over *nested* random embeddings of a mixed space (Papenmeier et al., 2023).

Bounce attacks high-dimensional combinatorial/mixed optimization with three ideas, all
kept here:

* **Nested random embedding.** The optimizer searches a low-dimensional *target* space
  that is embedded into the (higher-dimensional) input space, and *grows* that target
  dimension over the course of the run -- a **split**. The embedding is HeSBO-style: each
  continuous input dimension is tied to one *bin* (one target dimension) with a random
  sign, so a target coordinate drives a whole group of input dimensions at once. A split
  partitions one bin's members into two bins; crucially it is done so that **every point
  already observed decodes to exactly the same input** after the split (the new coordinate
  is seeded from its parent), so the target dimension increases *without discarding a
  single observation* -- this is what "nested" means (Papenmeier et al., Sec. 3). The
  target dimension climbs from a small start up to the full input dimension.
* **CoCaBO-style kernel.** The surrogate over the target space uses the mixed
  (categorical-overlap + continuous-Matern, summed *and* multiplied) kernel of CoCaBO /
  Casmopolitan. This module reuses that exact kernel + GP fit from ``casmopolitan.py``.
* **Trust region + EI/local-search acquisition.** A TuRBO trust region is maintained in
  the target space; expected improvement is maximized inside it by perturbing the
  continuous target dimensions within the TR box and doing local search over the discrete
  target dimensions. When the TR collapses, Bounce **splits** (grows the target dimension)
  instead of restarting, until the full input dimension is reached.

**Faithful-minimal scope.** The official code (github.com/LeoIV/bounce, a poetry/gin project
with heavily pinned dependencies) does not install cleanly into this environment, and the
PyPI name ``bounce`` is an unrelated package, so this is an in-repo reimplementation of the
method rather than a wrapper. It keeps Bounce's defining machinery -- the nested embedding
with observation-preserving splits, the CoCaBO kernel, and the TR/EI loop -- while embedding
only the *continuous* input block and modelling each discrete input dimension as its own
target dimension (the mixed problems in BoCoDe are low-dimensional, so this exercises the
embedding mechanism without the full binary/categorical bin machinery of the paper).

BoCoDe maximizes and its constraints fold into a single penalized objective, exactly as the
other single-objective mixed-variable baselines. CPU-friendly (the target-space GP is tiny).

Run::

    python -m algorithms.single_obj_mixed_variable.bounce --problem AckleyCat --init 8 --iters 50
    python -m algorithms.single_obj_mixed_variable.bounce --problem Ackley5Mixed --iters 40 --checkpoint bounce.npz

Sources:
L. Papenmeier, L. Nardi, and M. Poloczek. Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces. NeurIPS 2023. https://arxiv.org/abs/2307.00618
Official implementation: https://github.com/LeoIV/bounce
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

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
    save_checkpoint,
    set_seed,
)
from .casmopolitan import _copula_standardize, _ei, _train_gp
from .single_task_gp import _discrete_grids

# Trust-region hyperparameters (TuRBO defaults, mirrored into the maximization frame).
LENGTH_INIT = 0.8
LENGTH_MIN = 0.5**7
LENGTH_MAX = 1.6
TR_MULTIPLIER = 2.0
SUCCTOL = 3
FAILTOL = 10
INIT_TARGET_DIM = 2  #: continuous target dimension the nested embedding starts from


class _TargetSpace:
    """A lightweight ``space`` for :func:`casmopolitan._train_gp` / :func:`casmopolitan._ei`.

    The target space is ``[continuous target dims] ++ [discrete target dims]``; continuous
    dims are Matern, discrete dims are categorical (they hold an ordinal level index, so the
    overlap kernel's equality test is exact). No integer-wrapped continuous dims here.
    """

    def __init__(self, m_cont: int, n_categories: list[int]):
        self.dim = m_cont + len(n_categories)
        self.cont_dims = list(range(m_cont))
        self.cat_dims = list(range(m_cont, m_cont + len(n_categories)))
        self.int_pos: list[int] = []
        self.n_categories = list(n_categories)


class _Embedding:
    """The nested random embedding: target space <-> BoCoDe's unit-cube input space.

    Continuous input dims are HeSBO-embedded (each tied to one target *bin* with a random
    sign); discrete input dims are kept explicit (one target categorical dim each, holding
    an ordinal level index). ``split`` grows the continuous target dimension while leaving
    every past decode unchanged.
    """

    def __init__(self, problem, seed: int):
        types = problem.resolved_variable_types()
        self.grids = _discrete_grids(
            problem
        )  # unit-cube levels of each non-continuous dim
        self.dim = problem.dim
        self.cont_input = [i for i, t in enumerate(types) if t == "continuous"]
        self.disc_input = sorted(self.grids)  # integer + categorical input dims
        self.n_categories = [len(self.grids[i]) for i in self.disc_input]
        n_cont = len(self.cont_input)

        rng = np.random.default_rng(seed)
        self.signs = rng.choice([-1.0, 1.0], size=n_cont)
        self.m_cont = min(n_cont, INIT_TARGET_DIM) if n_cont else 0
        # contiguous groups -> no empty bins, so a split always has members to divide
        self.bins = np.zeros(n_cont, dtype=int)
        if self.m_cont:
            for b, g in enumerate(np.array_split(np.arange(n_cont), self.m_cont)):
                self.bins[g] = b

    @property
    def n_disc(self) -> int:
        return len(self.disc_input)

    def space(self) -> _TargetSpace:
        return _TargetSpace(self.m_cont, self.n_categories)

    def can_split(self) -> bool:
        """True while some continuous bin still holds >= 2 input dims (target dim < full)."""
        if self.m_cont == 0:
            return False
        counts = np.bincount(self.bins, minlength=self.m_cont)
        return bool((counts >= 2).any())

    def split(self, Zc: np.ndarray) -> np.ndarray:
        """Split the largest bin in two; extend the continuous history so decodes are unchanged.

        Returns the grown continuous history ``Zc`` (a new column, copied from the parent bin,
        is appended so every existing point still decodes to the same input).
        """
        counts = np.bincount(self.bins, minlength=self.m_cont)
        parent = int(np.argmax(counts))
        members = np.where(self.bins == parent)[0]
        half = members[len(members) // 2 :]  # second half moves to the new bin
        self.bins[half] = self.m_cont
        Zc = np.concatenate([Zc, Zc[:, parent : parent + 1]], axis=1)
        self.m_cont += 1
        return Zc

    def decode(self, Zc: np.ndarray, Zd: np.ndarray) -> torch.Tensor:
        """Target point(s) -> unit-cube input point(s) (shape ``(n, problem.dim)``)."""
        n = max(Zc.shape[0], Zd.shape[0]) if (Zc.size or Zd.size) else 1
        X = np.full((n, self.dim), 0.5)
        for p, j in enumerate(self.cont_input):
            z = Zc[:, self.bins[p]]
            X[:, j] = np.clip(0.5 + self.signs[p] * (z - 0.5), 0.0, 1.0)
        for q, i in enumerate(self.disc_input):
            idx = np.clip(np.rint(Zd[:, q]).astype(int), 0, self.n_categories[q] - 1)
            X[:, i] = np.asarray(self.grids[i])[idx]
        return torch.from_numpy(X).to(DTYPE)

    def sample_design(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """A (continuous LHS, discrete uniform) initial design in the *target* space."""
        Zc = (
            initial_design(n, self.m_cont, seed).numpy()
            if self.m_cont
            else np.zeros((n, 0))
        )
        rng = np.random.default_rng(seed + 1)
        Zd = (
            np.stack(
                [rng.integers(0, k, size=n) for k in self.n_categories], axis=1
            ).astype(float)
            if self.n_disc
            else np.zeros((n, 0))
        )
        return Zc, Zd


def _candidates(emb, Zc_c, Zd_c, length, n_cand, rng):
    """EI candidates inside the trust region around the incumbent target point.

    Continuous target dims are perturbed within the TR box (TuRBO-style partial perturbation);
    discrete target dims get local search (each resampled to a random level with small
    probability). Returns ``(Zc, Zd)`` candidate blocks.
    """
    if emb.m_cont:
        lb = np.clip(Zc_c - length / 2.0, 0.0, 1.0)
        ub = np.clip(Zc_c + length / 2.0, 0.0, 1.0)
        pert = lb + (ub - lb) * rng.random((n_cand, emb.m_cont))
        prob = min(1.0, 20.0 / emb.m_cont)
        mask = rng.random((n_cand, emb.m_cont)) <= prob
        empty = np.where(mask.sum(axis=1) == 0)[0]
        mask[empty, rng.integers(0, emb.m_cont, size=len(empty))] = True
        Zc = np.tile(Zc_c, (n_cand, 1))
        Zc[mask] = pert[mask]
    else:
        Zc = np.zeros((n_cand, 0))

    Zd = np.tile(Zd_c, (n_cand, 1)) if emb.n_disc else np.zeros((n_cand, 0))
    if emb.n_disc:
        prob = min(1.0, 1.0 / emb.n_disc)  # change ~one discrete dim per candidate
        cmask = rng.random((n_cand, emb.n_disc)) < prob
        for q in range(emb.n_disc):
            rows = np.where(cmask[:, q])[0]
            if len(rows):
                Zd[rows, q] = rng.integers(0, emb.n_categories[q], size=len(rows))
    return Zc, Zd


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Bounce over the problem's mixed space: nested embedding + CoCaBO GP + TR/EI."""
    set_seed(seed)
    resolve_device(
        device
    )  # honored for parity; the target-space GP is tiny and runs on CPU
    obj = ProblemObjective(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result("bounce", type(problem).__name__, seed, acquisition_function="EI")

    emb = _Embedding(problem, seed)

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, data = load_checkpoint(checkpoint, res)
        emb.m_cont = int(data["m_cont"])
        emb.bins = data["bins"].astype(int)
        emb.signs = data["signs"].astype(float)
        Zc = data["Zc"].reshape(len(train_X), emb.m_cont)
        Zd = data["Zd"].reshape(len(train_X), emb.n_disc)
        length = float(data["length"])
        succ, fail = int(data["succ"]), int(data["fail"])
        best = train_Y.max().item()
    else:
        Zc, Zd = emb.sample_design(n_init, seed)
        train_X = emb.decode(Zc, Zd)
        train_Y = obj(train_X)
        best = train_Y.max().item()
        res.start(best)
        length, succ, fail = LENGTH_INIT, 0, 0
        start_it = 0

    for it in range(start_it, iters):
        # Seed the candidate RNG from (seed, it) only, so a resumed run reproduces the same
        # candidates as an uninterrupted one (the numpy Generator state is not checkpointed).
        rng = np.random.default_rng([seed, it])
        space = emb.space()
        Z = np.concatenate([Zc, Zd], axis=1)
        Zt = torch.from_numpy(Z).to(DTYPE)
        y_std = torch.tensor(_copula_standardize(train_Y.numpy()), dtype=DTYPE)
        model = _train_gp(Zt, y_std, space)

        b = int(train_Y.argmax().item())
        Zc_c, Zd_c = Zc[b], Zd[b]
        with torch.no_grad():
            mu_star = model.likelihood(model(Zt[b : b + 1])).mean

        n_cand = min(100 * max(space.dim, 1), 2000)
        Cc, Cd = _candidates(emb, Zc_c, Zd_c, length, n_cand, rng)
        cand = torch.from_numpy(np.concatenate([Cc, Cd], axis=1)).to(DTYPE)
        with torch.no_grad():
            ei = _ei(model, cand, mu_star).reshape(-1)
            choice = int(torch.argmax(ei).item())
        zc_new, zd_new = Cc[choice : choice + 1], Cd[choice : choice + 1]

        candidate = emb.decode(zc_new, zd_new)
        with torch.no_grad():
            post = model(cand[choice : choice + 1])
            mean, var = post.mean.item(), post.variance.item()
        y = obj(candidate)

        # TuRBO success/failure counters (maximization frame).
        if y.item() >= best + 1e-3 * math.fabs(best):
            succ, fail = succ + 1, 0
        else:
            succ, fail = 0, fail + 1
        if succ == SUCCTOL:
            length, succ = min(length * TR_MULTIPLIER, LENGTH_MAX), 0
        elif fail == FAILTOL:
            length, fail = max(length / TR_MULTIPLIER, LENGTH_MIN / 2), 0

        Zc = np.concatenate([Zc, zc_new], axis=0)
        Zd = np.concatenate([Zd, zd_new], axis=0)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        best = max(best, y.item())

        # On TR collapse, GROW the target dimension (a nested split) rather than restart,
        # until the full input dimension is reached; then just re-open the TR.
        if length <= LENGTH_MIN:
            if emb.can_split():
                Zc = emb.split(Zc)
            length, succ, fail = LENGTH_INIT, 0, 0

        res.record(best, mean=mean, variance=var, acq_value=ei[choice].item())
        if checkpoint:
            res.set_history(train_X, train_Y, n_init)
            save_checkpoint(
                checkpoint,
                train_X,
                train_Y,
                res,
                it + 1,
                extra=dict(
                    m_cont=emb.m_cont,
                    bins=emb.bins,
                    signs=emb.signs,
                    Zc=Zc,
                    Zd=Zd,
                    length=length,
                    succ=succ,
                    fail=fail,
                ),
            )

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
