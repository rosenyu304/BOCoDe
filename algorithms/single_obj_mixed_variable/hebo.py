"""HEBO-MCBO for mixed-variable problems -- HEBO's ingredients inside the MCBO framework.

This is **not** vanilla ``hebo.optimizers.hebo.HEBO``. It drives Huawei Noah's **MCBO**
library (*Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian
Optimization*, Dreczkowski, Grosnit & Bou-Ammar, NeurIPS 2023 D&B; lives in the HEBO repo
under ``MCBO/``). MCBO composes a BO method from five interchangeable primitives -- search
space, surrogate/kernel, acquisition function, acquisition optimizer, and (optional) trust
region -- via ``mcbo.optimizers.bo_builder.BoBuilder``.

**"HEBO in MCBO".** MCBO ships named recipes for the *combinatorial* baselines it
benchmarks (Casmopolitan, COMBO, BODi, BOSS, CoCaBO, BOiLS, BOCS, RDUCB) but **no** named
HEBO recipe -- HEBO is not one of the algorithms in the MCBO paper. HEBO's two signature
components, moreover, are exactly the two things MCBO does *not* implement: HEBO's
input-warped GP surrogate (a learned Kumaraswamy CDF on each input) and its **MACE**
acquisition (EI/PI/UCB optimized *simultaneously* with NSGA-II, the next query drawn from
the Pareto front). MACE is explicitly on MCBO's roadmap as unimplemented, and MCBO has no
input-warping surrogate. So "HEBO in MCBO" is a ``BoBuilder`` configured with the closest
available primitives to HEBO's ingredients:

* **surrogate** ``gp_to`` -- an exact GP whose numeric dims use an ARD Matern-5/2 kernel
  (as HEBO's GP does) and whose nominal dims use a transformed-overlap kernel;
* **acquisition optimizer** ``ga`` -- an evolutionary (genetic-algorithm) acquisition
  optimizer, the MCBO analogue of HEBO's NSGA-II acquisition optimization (and the trait
  MCBO's own ablations found most beneficial);
* **acquisition** ``ei`` -- Expected Improvement, MCBO's closest single-objective stand-in
  for HEBO's multi-objective MACE ensemble;
* **trust region** ``None`` -- HEBO uses no trust region.

**Search space.** Each BoCoDe dimension becomes one MCBO parameter over the unit cube
BoCoDe optimizes: ``continuous`` -> ``num`` on ``[0, 1]``, ``integer`` -> ``int`` carrying
the *index* into the dimension's allowed integer grid, and an explicit list of levels ->
``nominal`` carrying the *index* into that dimension's allowed levels. Indexing the
non-continuous dims means every point MCBO proposes decodes to an allowed level by
construction; the decode maps each index back to its unit-cube value.

**Sign.** BoCoDe maximizes; MCBO always *minimizes*, so the objective handed to
``observe``/``initialize`` is negated, and only there.

**Initial design.** BoCoDe's dimension-scaled LHS initial design is injected into the
optimizer via MCBO's ``initialize`` (which consumes the optimizer's own would-be random
init), so every algorithm in the suite starts from the same budget.

**Install.** MCBO is not on PyPI; it is a sub-tree of the HEBO repo. It was installed into
this environment from the local source tree with
``pip install -e <HEBO repo>/MCBO --no-deps --no-build-isolation`` (``--no-deps`` leaves
the env's torch/botorch/gpytorch untouched; MCBO pins old versions it does not actually
need). One pure-Python dependency, ``seaborn`` (a plotting import pulled in at package
import), was added with ``--no-deps``. MCBO targets gpytorch < 1.10, which still exposed
``gpytorch.lazify``/``delazify``; those moved into ``linear_operator`` in the modern stack,
so a tiny import-time shim aliases them back (see ``_install_gpytorch_compat_shim``).

Note: MCBO's ``suggest`` does not expose its surrogate's posterior at the chosen point, so
the per-iteration ``mean`` / ``variance`` / acquisition-value diagnostics are recorded as
NaN rather than fabricated.

Run::

    python -m algorithms.single_obj_mixed_variable.hebo --problem AckleyCat --iters 40
    python -m algorithms.single_obj_mixed_variable.hebo --problem Ackley5Mixed --iters 40 --checkpoint hebo.npz

Sources:
K. Dreczkowski, A. Grosnit, and H. Bou-Ammar. Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization. NeurIPS 2023 Datasets & Benchmarks. https://arxiv.org/abs/2306.09803
A. I. Cowen-Rivers et al. HEBO: Pushing the Limits of Sample-Efficient Hyper-parameter Optimisation. JAIR 74, 2022. https://arxiv.org/abs/2012.03826
MCBO implementation (wrapped here): https://github.com/huawei-noah/HEBO/tree/master/MCBO
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _install_gpytorch_compat_shim() -> None:
    """Bridge MCBO's gpytorch < 1.10 API onto this env's gpytorch + linear_operator.

    MCBO calls ``gpytorch.lazify`` / ``gpytorch.delazify`` at import time
    (``mcbo/models/gp/kernels.py``); both were removed when gpytorch split ``LazyTensor``
    out into the standalone ``linear_operator`` package. Alias them back so MCBO imports
    unchanged. Idempotent.
    """
    import gpytorch

    if not hasattr(gpytorch, "lazify"):
        from linear_operator.operators import to_linear_operator

        gpytorch.lazify = to_linear_operator
    if not hasattr(gpytorch, "delazify"):

        def _delazify(obj):
            if hasattr(obj, "to_dense"):
                return obj.to_dense()
            if hasattr(obj, "evaluate"):
                return obj.evaluate()
            return obj

        gpytorch.delazify = _delazify


_install_gpytorch_compat_shim()

import pandas as pd
import torch
from mcbo.optimizers.bo_builder import BoBuilder
from mcbo.search_space import SearchSpace

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
from .single_task_gp import _discrete_grids, _snap

#: HEBO's ingredients expressed with MCBO's primitives (see the module docstring):
#: ARD-Matern/transformed-overlap GP + genetic-algorithm acquisition optimization + EI,
#: no trust region.
MCBO_HEBO = dict(model_id="gp_to", acq_opt_id="ga", acq_func_id="ei", tr_id=None)


class _Space:
    """BoCoDe's ``variable_types`` <-> an MCBO ``SearchSpace`` (over the unit cube).

    Non-continuous dimensions are parameterized by the *index* of their allowed value, so
    decoding an MCBO proposal can only ever produce an allowed level.
    """

    def __init__(self, problem):
        types = problem.resolved_variable_types()
        self.dim = problem.dim
        self.grids = _discrete_grids(problem)  # unit values of each non-continuous dim
        params = []
        for i, t in enumerate(types):
            name = f"x{i}"
            if t == "continuous":
                params.append({"name": name, "type": "num", "lb": 0.0, "ub": 1.0})
            elif t == "integer":
                # index into the dimension's integer grid
                params.append(
                    {"name": name, "type": "int", "lb": 0, "ub": len(self.grids[i]) - 1}
                )
            else:  # explicit list of allowed levels -> nominal (index into the levels)
                params.append(
                    {
                        "name": name,
                        "type": "nominal",
                        "categories": list(range(len(t))),
                    }
                )
        self.names = [f"x{i}" for i in range(self.dim)]
        self.search_space = SearchSpace(params=params, dtype=DTYPE)

    def to_unit(self, df: pd.DataFrame) -> torch.Tensor:
        """An MCBO suggestion frame -> unit-cube points."""
        X = torch.zeros((len(df), self.dim), dtype=DTYPE)
        for i in range(self.dim):
            col = df[f"x{i}"].to_numpy()
            if i in self.grids:  # int / nominal dims carry the index into the grid
                X[:, i] = torch.tensor(
                    [self.grids[i][int(round(float(v)))] for v in col], dtype=DTYPE
                )
            else:
                X[:, i] = torch.tensor([float(v) for v in col], dtype=DTYPE)
        return X

    def to_frame(self, X_unit: torch.Tensor) -> pd.DataFrame:
        """Unit-cube points -> an MCBO observation frame (for seeding the initial design)."""
        data = {}
        for i in range(self.dim):
            if i in self.grids:
                g = torch.tensor(self.grids[i], dtype=DTYPE)
                idx = (X_unit[:, i].unsqueeze(1) - g.unsqueeze(0)).abs().argmin(dim=1)
                # both int and nominal params carry an integer index/label
                data[f"x{i}"] = [int(k) for k in idx.tolist()]
            else:
                data[f"x{i}"] = [float(v) for v in X_unit[:, i].clamp(0.0, 1.0).tolist()]
        return pd.DataFrame(data, columns=self.names)


def _build_optimizer(space: _Space, n_init: int):
    """A fresh MCBO 'HEBO' optimizer over the problem's search space (CPU, float64)."""
    builder = BoBuilder(**MCBO_HEBO)
    return builder.build_bo(
        search_space=space.search_space,
        n_init=n_init,
        device=torch.device("cpu"),
        dtype=DTYPE,
    )


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 40,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """HEBO-MCBO over the problem's mixed categorical/integer/continuous space."""
    set_seed(seed)
    resolve_device(device)  # honored for parity; MCBO's mixed GP is tiny and runs on CPU
    obj = ProblemObjective(problem)
    space = _Space(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result("hebo", type(problem).__name__, seed, acquisition_function="EI")

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, _ = load_checkpoint(checkpoint, res)
    else:
        # BoCoDe's dimension-scaled LHS initial design (rather than MCBO's own random
        # phase), so every algorithm in the suite starts from the same budget.
        train_X = _snap(initial_design(n_init, obj.dim, seed), space.grids)
        train_Y = obj(train_X)
        res.start(train_Y.max().item())
        start_it = 0

    # With no trust region, the MCBO optimizer's state is exactly its observed data, so a
    # resumed run is rebuilt by replaying the whole history into a fresh instance: the
    # first n_init points seed the (skipped) init phase via `initialize`, the rest are
    # `observe`d. MCBO minimizes, so every objective is negated.
    opt = _build_optimizer(space, n_init)
    opt.initialize(
        space.to_frame(train_X[:n_init]), -train_Y[:n_init].numpy().reshape(-1, 1)
    )
    if len(train_X) > n_init:
        opt.observe(
            space.to_frame(train_X[n_init:]), -train_Y[n_init:].numpy().reshape(-1, 1)
        )

    for it in range(start_it, iters):
        rec = opt.suggest(n_suggestions=1)
        candidate = space.to_unit(rec)
        y = obj(candidate)
        opt.observe(rec, -y.numpy().reshape(-1, 1))  # MCBO minimizes

        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        res.record(train_Y.max().item())

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
