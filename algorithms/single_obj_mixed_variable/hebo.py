"""HEBO for mixed-variable problems -- a thin adapter around the **official** package.

This file does not reimplement HEBO: it drives Huawei Noah's official ``HEBO`` optimizer
(``from hebo.optimizers.hebo import HEBO``) on a BoCoDe problem. All of the method lives
in that package:

* the **non-linear output transform** (Box-Cox when the objective is strictly positive,
  Yeo-Johnson otherwise, Sec. 4.1 of the paper),
* the **input-warped GP** surrogate (a Kumaraswamy CDF on each input, learned jointly with
  the GP hyperparameters, Sec. 4.2), and
* **MACE**, the multi-objective acquisition ensemble that *is* the method: EI, PI and UCB
  are optimized *simultaneously* with NSGA-II and the next query is drawn from the
  resulting Pareto front (Sec. 5).

What this file supplies is only the adapter: BoCoDe's ``variable_types`` -> HEBO's
``DesignSpace``, and BoCoDe's unit-cube/maximization conventions -> HEBO's native
minimization.

**Search space.** Each dimension becomes one HEBO parameter: ``continuous`` -> ``num`` on
``[0, 1]`` (the unit cube BoCoDe optimizes over), ``integer`` -> ``int``, and a list of
allowed levels -> ``cat``. The integer/categorical parameters carry the *index* into the
dimension's allowed values, so every point HEBO proposes decodes to an allowed level by
construction -- no rounding or snapping is applied to its proposals.

**Sign.** BoCoDe maximizes and HEBO minimizes, so the objective handed to
``HEBO.observe`` is negated, and only there; everything the harness sees stays in
BoCoDe's maximization convention.

**Install.** The official package is on PyPI as ``HEBO``, but its PyPI sdist cannot build
its metadata on Python 3.12 (it pins a NumPy that no longer compiles). It was therefore
installed into this environment from the official source tree with
``pip install --no-deps --no-build-isolation <HEBO repo>/HEBO`` (plus ``disjoint_set``,
its one missing pure-Python dependency); ``--no-deps`` leaves the rest of the environment
untouched. HEBO 0.3.6 then runs unmodified against this environment's torch/gpytorch.

Note: HEBO's ``suggest()`` does not expose its surrogate, so the per-iteration ``mean`` /
``variance`` / acquisition-value diagnostics are recorded as NaN rather than fabricated.

Run::

    python -m algorithms.single_obj_mixed_variable.hebo --problem Ackley5Mixed --init 20 --iters 40
    python -m algorithms.single_obj_mixed_variable.hebo --problem CoCaBOFunc2C --iters 40 --checkpoint hebo.npz

Sources:
A. I. Cowen-Rivers, W. Lyu, R. Tutunov, Z. Wang, A. Grosnit, R. R. Griffiths, A. M. Maraval, H. Jianye, J. Wang, J. Peters, and H. Bou-Ammar. HEBO: Pushing the Limits of Sample-Efficient Hyper-parameter Optimisation. JAIR 74, 2022. https://arxiv.org/abs/2012.03826
Official implementation (wrapped here): https://github.com/huawei-noah/HEBO
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

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


class _Space:
    """BoCoDe's ``variable_types`` <-> HEBO's ``DesignSpace`` (over the unit cube).

    Non-continuous dimensions are parameterized by the *index* of their allowed value, so
    decoding a HEBO proposal can only ever produce an allowed level.
    """

    def __init__(self, problem):
        types = problem.resolved_variable_types()
        self.dim = problem.dim
        self.grids = _discrete_grids(problem)  # unit values of each non-continuous dim
        specs = []
        for i, t in enumerate(types):
            name = f"x{i}"
            if t == "continuous":
                specs.append({"name": name, "type": "num", "lb": 0.0, "ub": 1.0})
            elif t == "integer":
                specs.append(
                    {"name": name, "type": "int", "lb": 0, "ub": len(self.grids[i]) - 1}
                )
            else:  # explicit list of allowed levels -> nominal
                specs.append(
                    {
                        "name": name,
                        "type": "cat",
                        "categories": [str(k) for k in range(len(t))],
                    }
                )
        self.design_space = DesignSpace().parse(specs)
        self.names = [f"x{i}" for i in range(self.dim)]

    def to_unit(self, df: pd.DataFrame) -> torch.Tensor:
        """A HEBO recommendation -> unit-cube points."""
        X = torch.zeros((len(df), self.dim), dtype=DTYPE)
        for i in range(self.dim):
            col = df[f"x{i}"].values
            if i in self.grids:
                X[:, i] = torch.tensor(
                    [self.grids[i][int(v)] for v in col], dtype=DTYPE
                )
            else:
                X[:, i] = torch.tensor([float(v) for v in col], dtype=DTYPE)
        return X

    def to_frame(self, X_unit: torch.Tensor) -> pd.DataFrame:
        """Unit-cube points -> a HEBO observation frame (for seeding the initial design)."""
        data = {}
        for i in range(self.dim):
            if i in self.grids:
                g = torch.tensor(self.grids[i], dtype=DTYPE)
                idx = (X_unit[:, i].unsqueeze(1) - g.unsqueeze(0)).abs().argmin(dim=1)
                # ``cat`` levels are strings in HEBO, ``int`` levels are ints
                is_cat = self.design_space.paras[f"x{i}"].is_categorical
                data[f"x{i}"] = [
                    str(int(k)) if is_cat else int(k) for k in idx.tolist()
                ]
            else:
                data[f"x{i}"] = X_unit[:, i].clamp(0.0, 1.0).tolist()
        return pd.DataFrame(data, columns=self.names)


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 40,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Official HEBO over the problem's mixed categorical/continuous space."""
    set_seed(seed)
    resolve_device(
        device
    )  # honored for parity; HEBO's surrogate is tiny and runs on CPU
    obj = ProblemObjective(problem)
    space = _Space(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result("hebo", type(problem).__name__, seed, acquisition_function="MACE")

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, _ = load_checkpoint(checkpoint, res)
    else:
        # BoCoDe's dimension-scaled LHS initial design (rather than HEBO's own quasi-random
        # phase), so every algorithm in the suite starts from the same budget.
        train_X = _snap(initial_design(n_init, obj.dim, seed), space.grids)
        train_Y = obj(train_X)
        res.start(train_Y.max().item())
        start_it = 0

    # HEBO is a pure ask/tell optimizer with no state beyond its observations, so a
    # resumed run is rebuilt exactly by replaying the history into a fresh instance.
    opt = HEBO(space.design_space, rand_sample=2, scramble_seed=seed)
    opt.observe(space.to_frame(train_X), -train_Y.numpy())  # HEBO minimizes

    for it in range(start_it, iters):
        rec = opt.suggest(n_suggestions=1)
        candidate = space.to_unit(rec)
        y = obj(candidate)
        opt.observe(rec, -y.numpy())  # HEBO minimizes

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
