"""Bounce for mixed-variable problems -- a thin adapter around the **official** package.

This file does not reimplement Bounce: it drives Papenmeier et al.'s official ``bounce``
optimizer (``from bounce.bounce import Bounce``) on a BoCoDe problem. All of the method --
and, crucially, its defining contribution -- lives in that package:

* the **nested random embedding** over the *full* mixed space (``bounce/projection.py``,
  ``AxUS``): binary, categorical (one-hot) *and* continuous input dimensions are grouped
  into a small number of shared *target* bins, so the optimizer searches a low-dimensional
  target space whose dimension is *grown by observation-preserving splits* over the run
  (Sec. 3). This is the combinatorial dimensionality reduction the paper is about;
* the **CoCaBO-style mixed GP** over that target space (``bounce/gaussian_process.py``);
* the **trust region + interleaved EI search** that proposes points and triggers a split
  when it collapses (``bounce/trust_region.py``, ``bounce/bounce.py``).

What this file supplies is only the adapter: BoCoDe's ``variable_types`` -> Bounce's
``Parameter`` list, and BoCoDe's unit-cube/maximization conventions -> Bounce's native
minimization over the representation space.

**Search space.** Each BoCoDe dimension becomes one Bounce ``Parameter``: ``continuous``
-> ``CONTINUOUS`` on ``[0, 1]`` (the unit cube BoCoDe optimizes over), and every discrete
dimension -- both ``integer`` and an explicit list of levels -- becomes a ``CATEGORICAL``
parameter over the *index* of its allowed values. Categorical parameters are the ones
Bounce places into the nested embedding, so this is what makes the discrete dimensions
genuinely embedded (grouped into shared target bins) rather than mapped one-to-one.
Integer dimensions are modelled as categorical because the official ``sample_init`` raises
``NotImplementedError`` for ``ORDINAL`` parameters; this drops the ordering of integer
levels but keeps them inside the same combinatorial embedding.

**Sign / decode.** Bounce minimizes over the one-hot *representation* space; each point it
evaluates is decoded here (categorical block -> ``argmax`` index -> the level's unit-cube
value; continuous block -> the value itself), the BoCoDe (maximization) objective is
evaluated on the decoded unit-cube point, and its negation is handed back to Bounce. Only
the sign flips at that boundary; everything the harness sees stays in BoCoDe's convention.

**Install.** The official package pins old dependencies (``torch==2.0.0``, ``botorch
^0.8.2``, ``pandas<2.0.0``) via poetry, so a plain ``pip install`` fails to build on this
environment's Python 3.12. It was therefore installed from the official source tree with
``pip install --no-deps --no-build-isolation <bounce repo>`` (plus ``gin-config``, its one
otherwise-missing pure-Python import); ``--no-deps`` leaves the rest of the environment
untouched. Bounce 0.1.0 then runs unmodified against this environment's torch 2.x /
botorch 0.18 / gpytorch 1.15.

Note: Bounce runs its whole budget inside a single ``run()`` call whose optimizer state is
not externally step-drivable, so the checkpoint here is completion-only (a finished run is
not re-run on resume); it does not resume a partially completed run. Per-iteration ``mean``
/ ``variance`` / acquisition diagnostics are recorded as NaN rather than reaching into
Bounce's internals.

Run::

    python -m algorithms.single_obj_mixed_variable.bounce --problem AckleyCat --init 8 --iters 50
    python -m algorithms.single_obj_mixed_variable.bounce --problem Ackley5Mixed --iters 40 --checkpoint bounce.npz

Sources:
L. Papenmeier, L. Nardi, and M. Poloczek. Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces. NeurIPS 2023. https://arxiv.org/abs/2307.00618
Official implementation (wrapped here): https://github.com/LeoIV/bounce
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
from bounce.bounce import Bounce
from bounce.benchmarks import Benchmark
from bounce.util.benchmark import Parameter, ParameterType

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
from .single_task_gp import _discrete_grids

# Official gin defaults (configs/default.gin); the start dimension is clamped per problem.
INIT_TARGET_DIM = 5  #: Bounce's initial target dimensionality
NEW_BINS_ON_SPLIT = 2  #: number of new bins created at each split

logging.getLogger("bounce").setLevel(logging.ERROR)


class _BoCoDeBenchmark(Benchmark):
    """A BoCoDe problem exposed as a Bounce ``Benchmark`` (over the one-hot representation).

    Continuous dims -> ``CONTINUOUS`` on ``[0, 1]``; every discrete dim (integer or an
    explicit list of levels) -> ``CATEGORICAL`` over the *index* of its allowed values, so
    the discrete dimensions enter Bounce's nested embedding. ``__call__`` receives points
    in Bounce's representation space (one-hot categorical blocks), decodes them to the unit
    cube, and returns the *negated* BoCoDe objective (Bounce minimizes).
    """

    def __init__(self, problem, obj: ProblemObjective):
        self._obj = obj
        types = problem.resolved_variable_types()
        self.grids = _discrete_grids(problem)  # unit-cube levels of each non-continuous dim
        self._dim = problem.dim

        params: list[Parameter] = []
        for i, t in enumerate(types):
            if t == "continuous":
                params.append(
                    Parameter(
                        name=f"x{i}",
                        type=ParameterType.CONTINUOUS,
                        lower_bound=0.0,
                        upper_bound=1.0,
                    )
                )
            else:  # integer or explicit level list -> categorical over level indices
                k = len(self.grids[i])
                params.append(
                    Parameter(
                        name=f"x{i}",
                        type=ParameterType.CATEGORICAL,
                        lower_bound=0,
                        upper_bound=k - 1,
                    )
                )
        super().__init__(parameters=params, noise_std=None)

    def decode(self, X_repr: np.ndarray) -> torch.Tensor:
        """Representation-space points (n, representation_dim) -> unit-cube (n, dim)."""
        X_repr = np.atleast_2d(np.asarray(X_repr, dtype=float))
        out = np.empty((X_repr.shape[0], self._dim), dtype=float)
        for r, row in enumerate(X_repr):
            start = 0
            for i, p in enumerate(self.parameters):
                width = p.dims_required
                block = row[start : start + width]
                if p.type == ParameterType.CATEGORICAL:
                    out[r, i] = self.grids[i][int(np.argmax(block))]
                else:
                    out[r, i] = float(np.clip(block[0], 0.0, 1.0))
                start += width
        return torch.from_numpy(out).to(DTYPE)

    def __call__(self, X_repr) -> torch.Tensor:
        X_unit = self.decode(X_repr)
        y = self._obj(X_unit).reshape(-1)  # BoCoDe maximization objective
        return (-y).to(torch.double)  # Bounce minimizes


def _run_bounce(problem, obj, n_init, iters, seed, device):
    """Drive the official Bounce over ``n_init + iters`` evaluations; return unit-cube history.

    Returns ``(train_X, train_Y)`` in BoCoDe's convention (unit cube, maximization),
    truncated to exactly ``n_init + iters`` evaluations in chronological order.
    """
    set_seed(seed)
    bench = _BoCoDeBenchmark(problem, obj)
    n_types = len(bench.unique_parameter_types)
    # start dim must be >= number of parameter types and <= benchmark dim
    init_target = max(n_types, min(INIT_TARGET_DIM, bench.dim))

    dev = resolve_device(device)
    with tempfile.TemporaryDirectory(prefix="bounce_") as results_dir:
        opt = Bounce(
            benchmark=bench,
            number_initial_points=n_init,
            initial_target_dimensionality=init_target,
            number_new_bins_on_split=NEW_BINS_ON_SPLIT,
            maximum_number_evaluations=n_init + iters,
            batch_size=1,
            results_dir=results_dir,
            device=str(dev),
            dtype="float64",
        )
        opt.run()

    # x_up_global / fx_global are the chronological representation-space points and their
    # (minimization) values. Decode to the unit cube and flip the sign back to maximization.
    X_repr = opt.x_up_global.detach().cpu().numpy()
    fx = opt.fx_global.detach().cpu().numpy().reshape(-1)
    keep = min(len(fx), n_init + iters)
    train_X = bench.decode(X_repr[:keep])
    train_Y = torch.from_numpy(-fx[:keep]).to(DTYPE).reshape(-1, 1)
    return train_X, train_Y


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Official Bounce over the problem's mixed space (nested embedding + CoCaBO GP + TR/EI)."""
    obj = ProblemObjective(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result("bounce", type(problem).__name__, seed, acquisition_function="EI")

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, _, _ = load_checkpoint(checkpoint, res)
        res.set_history(train_X, train_Y, n_init)
        return res

    train_X, train_Y = _run_bounce(problem, obj, n_init, iters, seed, device)

    # Rebuild the per-iteration best-so-far trace: index 0 is the initial design, then one
    # entry per subsequent evaluation (running maximum).
    y = train_Y.reshape(-1)
    res.start(y[:n_init].max().item())
    running = y[:n_init].max().item()
    for k in range(n_init, len(y)):
        running = max(running, y[k].item())
        res.record(running)

    res.set_history(train_X, train_Y, n_init)
    if checkpoint:
        save_checkpoint(checkpoint, train_X, train_Y, res, len(y) - n_init)
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
