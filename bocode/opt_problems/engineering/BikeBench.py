"""BikeBench parametric bicycle design — 64-D with all four variable types.

BikeBench (MIT DeCoDE lab) is a benchmark of parametric bicycle designs — frame geometry,
tube dimensions, wheels, materials and components — scored by trained surrogates
(structural FEM, aerodynamics), analytic kinematics and 40 constraints. It is natively a
multi-objective / generative benchmark; the single-objective instance below
(**BikeBench-Mass**) is defined by us, which the benchmark's own documentation invites.

Design variables (64-D; this is the *un-one-hot* form of BikeBench's 83-column one-hot
layout ``bikebench.transformation.ordered_columns.bike_bench_columns``, verified against
the installed package rather than the paper text)::

    dims  0..47 : 48 continuous frame/tube/wheel/component parameters. The bounds are the
                  per-column min/max over the shipped 3036-row ``bike_bench.csv`` training
                  set -- the only published range information -- hardcoded in
                  ``_CONTINUOUS`` below so the benchmark instance is stable across
                  BikeBench releases (a mismatch against the loaded dataset warns).
    dims 48..51 : 4 integers  Number of cogs (0..11), Number of chainrings (0..2),
                  SPOKES composite front (0..6), SPOKES composite rear (0..5)
    dims 52..54 : 3 booleans  BELTorCHAIN, SEATSTAYbrdgCheck, CHAINSTAYbrdgCheck
    dims 55..63 : 9 categoricals with cardinalities [3, 4, 3, 3, 3, 3, 3, 3, 3]
                  (MATERIAL, Head tube type, Down tube type, RIM_STYLE front,
                  RIM_STYLE rear, Handlebar style, Stem kind, Fork type, Seat tube type)

  .. note:: The BikeBench paper's summary tables say "9 categoricals with 3 classes"; the
     shipped package in fact gives *Head tube type* four classes. We follow the package.

**Scalarization** (ours), minimized::

    f(x) = Mass(kg)  +  LAMBDA * (1/K) * sum_k relu(g_k(x)) / s_k

i.e. structural mass penalized by the ``K = 34`` constraints returned by
``StructuralEvaluator`` (2 safety factors) + ``FrameValidityEvaluator`` (1 learned
validity) + ``ValidationEvaluator`` (31 analytic geometric checks). BikeBench's sign
convention is ``g_k > 0 == VIOLATED`` (``benchmarking/scoring.py``: feasibility is
``all(constraint_scores <= 0)``), so ``relu(g_k)`` is the violation magnitude.
``evaluate()`` returns ``-f`` because BoCoDe maximizes. The problem is an unconstrained
box in BoCoDe terms (the constraints are folded into the objective).

``s_k = mean_x |g_k(x)|`` over a *fixed* reference sample of 2000 designs drawn from
``np.random.default_rng(0).random((2000, 64))`` in the unit cube and mapped through the
problem's own bounds/rounding — this is exactly BikeBench's own "meanabs" criterion
scaling (``scoring.get_ref_point``), recomputed locally so we do not depend on the 5
criteria whose names changed between the shipped ``default_weights.csv`` and the current
evaluator. Sanity-checked: on the 29 criteria whose names *do* still match, our ``s_k``
agrees with the shipped ``default_weights.csv`` to a median factor of 1.7.

``LAMBDA = 100`` kg: over that reference sample mass spans 1.5-32.1 kg (median 10.9) and
the normalized penalty spans 0-9.89 (median 0.125), so a median random design pays
~12.5 kg — the same order as the mass itself, i.e. mass and feasibility matter equally at
typical designs while grossly infeasible designs are decisively rejected. Only 1 of those
2000 random designs is fully feasible, so this is a genuinely hard constrained instance.

Deterministic (the surrogates are dropout-free MLPs with no BatchNorm). CPU by default: a
batch of 200 designs takes ~2.3 ms, so a GPU buys nothing (override with
``BOCODE_BIKEBENCH_DEVICE=cuda:0`` if ever wanted).

**Setup.** BikeBench is not on PyPI at a stable name; install it from source
(``pip install git+https://github.com/Lyleregenwetter/BikeBench``) or point
``BOCODE_BIKEBENCH_SRC`` at the ``src/`` directory of a clone. Its model weights and
dataset are downloaded by the package itself on first use. All heavy work (dataset load,
surrogate weights, reference sample) happens on the first ``evaluate()`` call;
constructing the problem needs nothing installed.

Sources:
L. Regenwetter, C. Y. Weaver, F. Ahmed, et al. BikeBench: A Benchmark for Engineering Design Generative Models. arXiv:2508.00830, 2025. https://github.com/Lyleregenwetter/BikeBench
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import warnings

import numpy as np
import torch

from ...base import BenchmarkProblem

_LAMBDA = 100.0  # kg per unit of mean normalized constraint violation
_REF_N = 2000  # reference sample size for the "meanabs" constraint scaling
_REF_SEED = 0

# (column name, lower bound, upper bound) for the 48 continuous variables: the per-column
# min/max of the shipped 3036-row bike_bench.csv training set (see module docstring).
_CONTINUOUS = [
    ("Seatpost LENGTH", 0.0, 700.0),
    ("CS textfield", 213.74, 1000.0),
    ("BB textfield", -163.0, 185.0),
    ("Stack", 242.22821, 802.26215),
    ("Head angle", 37.0, 90.0),
    ("Head tube length textfield", 34.4, 366.7),
    ("Seat stay junction0", -10.0, 430.0),
    ("Seat tube length", 77.0, 810.0),
    ("Seat angle", 34.0, 106.0),
    ("DT Length", 338.72543, 1243.389),
    ("FORK0R", -35.0, 145.0),
    ("BB diameter", 0.0, 80.0),
    ("ttd", 0.0, 93.6),
    ("dtd", 0.0, 120.0),
    ("csd", 0.0, 61.45),
    ("std", 0.0, 80.0),
    ("htd", 5.0, 80.0),
    ("ssd", 3.25, 51.0),
    ("Chain stay position on BB", 0.0, 125.0),
    ("SSTopZOFFSET", -2.0, 30.0),
    ("Head tube upper extension2", -26.1, 165.0),
    ("Seat tube extension2", -128.4, 491.0),
    ("Head tube lower extension2", -5.0, 230.0),
    ("SEATSTAYbrdgshift", 290.0, 380.0),
    ("CHAINSTAYbrdgshift", 260.0, 550.0),
    ("SEATSTAYbrdgdia1", 7.0, 30.0),
    ("CHAINSTAYbrdgdia1", 10.0, 27.0),
    ("Dropout spacing", 109.0, 190.0),
    ("Wall thickness Bottom Bracket", 0.20669384, 30.81698),
    ("Wall thickness Top tube", 0.08183249, 16.174728),
    ("Wall thickness Head tube", 0.07993562, 16.060106),
    ("Wall thickness Down tube", 0.06815186, 22.351868),
    ("Wall thickness Chain stay", 0.111149274, 15.135947),
    ("Wall thickness Seat stay", 0.07505079, 12.831684),
    ("Wall thickness Seat tube", 0.04927965, 11.62011),
    ("Wheel diameter front", 304.0, 1067.0),
    ("RDBSD", 1.0, 226.0),
    ("Wheel diameter rear", 304.0, 1067.0),
    ("FDBSD", -17.2, 226.0),
    ("BB length", 0.0, 730.0),
    ("Wheel cut", 500.0, 842.0),
    ("FIRST color R_RGB", 0.0, 255.0),
    ("FIRST color G_RGB", 0.0, 255.0),
    ("FIRST color B_RGB", 0.0, 255.0),
    ("SBLADEW front", 20.0, 140.0),
    ("SBLADEW rear", 20.0, 140.0),
    ("Saddle length", 130.0, 500.0),
    ("Saddle height", 0.0, 1702.7),
]

# (column name, cardinality) for the 4 integer variables: the shipped dataset's ranges.
_INTEGERS = [
    ("Number of cogs", 12),
    ("Number of chainrings", 3),
    ("SPOKES composite front", 7),
    ("SPOKES composite rear", 6),
]

_BOOLEANS = ["BELTorCHAIN", "SEATSTAYbrdgCheck", "CHAINSTAYbrdgCheck"]

# One-hot group cardinalities, in bike_bench_columns group order (MATERIAL, Head tube
# type, Down tube type, RIM_STYLE front/rear, Handlebar style, Stem kind, Fork type,
# Seat tube type). Verified against the package at evaluation time.
_CATEGORICAL_CARD = [3, 4, 3, 3, 3, 3, 3, 3, 3]


def _import_bikebench() -> None:
    """Import ``bikebench``, adding a vendored ``src/`` to ``sys.path`` when it is not
    pip-installed, and shimming ``transformers``.

    ``design_evaluation`` imports the CLIP module at file top, but the structural /
    validity / validation evaluators never touch it, so the shim avoids pulling
    ``transformers`` into the environment for nothing.
    """
    if importlib.util.find_spec("bikebench") is None:
        src = os.environ.get("BOCODE_BIKEBENCH_SRC", "")
        if not src or not os.path.isdir(src):
            raise ImportError(
                "The BikeBench64 problem requires the 'bikebench' package, which is not "
                "on PyPI: install it from source with "
                "`pip install git+https://github.com/Lyleregenwetter/BikeBench`, or set "
                "BOCODE_BIKEBENCH_SRC to the src/ directory of a clone."
            )
        if src not in sys.path:
            sys.path.insert(0, src)
    if importlib.util.find_spec("transformers") is None:
        shim = types.ModuleType("transformers")
        for name in ("CLIPTokenizer", "CLIPImageProcessor", "CLIPModel"):
            setattr(shim, name, None)
        sys.modules.setdefault("transformers", shim)


class BikeBench64(BenchmarkProblem):
    """BikeBench-Mass: 64-D mixed bicycle design (48 cont + 4 int + 3 bool + 9 cat).

    Maximize ``-(mass + LAMBDA * mean normalized constraint violation)``. See the module
    docstring for the variable layout, the constraint set, the penalty normalization and
    the setup instructions.
    """

    available_dimensions = 64
    num_objectives = 1
    num_constraints = 0

    tags = {"single_objective", "unconstrained", "64D", "mixed", "design"}

    def __init__(self, device: str | None = None) -> None:
        self.variable_types = (
            ["continuous"] * 48
            + ["integer"] * 4
            + ["integer"] * 3
            + [[float(i) for i in range(k)] for k in _CATEGORICAL_CARD]
        )
        super().__init__(
            dim=64,
            num_objectives=1,
            num_constraints=0,
            bounds=[(lo, hi) for _c, lo, hi in _CONTINUOUS]
            + [(0, k - 1) for _c, k in _INTEGERS]
            + [(0, 1)] * 3
            + [(0, k - 1) for k in _CATEGORICAL_CARD],
        )
        self._device = device or os.environ.get("BOCODE_BIKEBENCH_DEVICE", "cpu")
        self._state = None

    # -- lazy heavy initialization ---------------------------------------------
    def _ensure(self):
        if self._state is not None:
            return self._state
        _import_bikebench()
        from bikebench.data_loading import data_loading
        from bikebench.design_evaluation.design_evaluation import (
            FrameValidityEvaluator,
            StructuralEvaluator,
            ValidationEvaluator,
            construct_tensor_evaluator,
        )
        from bikebench.transformation import ordered_columns as oc

        cols = list(oc.bike_bench_columns)
        oh_groups = [list(g) for g in oc.oh_columns]
        if [len(g) for g in oh_groups] != _CATEGORICAL_CARD or len(cols) != 83:
            raise RuntimeError(
                "The installed BikeBench has a different design space than the one "
                f"BikeBench64 is defined against ({len(cols)} columns, one-hot groups "
                f"{[len(g) for g in oh_groups]}; expected 83 and {_CATEGORICAL_CARD})."
            )
        col_idx = {c: i for i, c in enumerate(cols)}

        train = data_loading.load_bike_bench_train()
        if list(train.columns) != cols:
            raise RuntimeError("bike_bench.csv column order != ordered_columns")
        lo = train[[c for c, _l, _h in _CONTINUOUS]].min().to_numpy(dtype=np.float64)
        hi = train[[c for c, _l, _h in _CONTINUOUS]].max().to_numpy(dtype=np.float64)
        ours = np.array([[a, b] for _c, a, b in _CONTINUOUS], dtype=np.float64)
        if not (
            np.allclose(lo, ours[:, 0], rtol=1e-6, atol=1e-9)
            and np.allclose(hi, ours[:, 1], rtol=1e-6, atol=1e-9)
        ):
            warnings.warn(
                "The installed BikeBench dataset's continuous column ranges differ from "
                "the ones BikeBench64's bounds were derived from; the hardcoded bounds "
                "are kept so the benchmark instance stays comparable.",
                RuntimeWarning,
                stacklevel=2,
            )

        # BikeBench's surrogates are float32 networks. ``BenchmarkProblem.evaluate``
        # raises the default dtype to float64 for the duration of a call, and this lazy
        # setup runs inside that window, so the modules would be built in float64 and
        # then reject the float32 design tensor ("mat1 and mat2 must have the same
        # dtype"). Build them under the package's own float32 default and restore.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            evaluators = [
                StructuralEvaluator(device=self._device),
                FrameValidityEvaluator(device=self._device),
                ValidationEvaluator(device=self._device),
            ]
            for e in evaluators:
                if hasattr(e, "model"):
                    e.model.eval()
            fn, names, is_obj, _ = construct_tensor_evaluator(
                evaluators, cols, device=self._device
            )
        finally:
            torch.set_default_dtype(prev_dtype)
        self._state = {
            "fn": fn,
            "cols": cols,
            "col_idx": col_idx,
            "oh_groups": oh_groups,
            "mass_i": names.index("Mass (kg)"),
            "con_i": np.where(~np.asarray(is_obj, dtype=bool))[0],
        }

        # Fixed reference sample -> per-constraint "meanabs" scale (see docstring). The
        # unit sample is mapped through this problem's own bounds/rounding, so the scale
        # is reproducible from the class definition alone.
        rng = np.random.default_rng(_REF_SEED)
        ref_unit = torch.from_numpy(rng.random((_REF_N, self.dim))).to(torch.float64)
        ref = self.enforce_variable_types(self.scale(ref_unit))
        _mass, cons = self._raw(ref)
        self._state["scale"] = np.maximum(np.abs(cons).mean(axis=0), 1e-12)
        return self._state

    # -- decode + raw evaluation -----------------------------------------------
    def _decode(self, x: np.ndarray) -> np.ndarray:
        """Snapped natural design ``(n, 64)`` -> the ``(n, 83)`` one-hot design matrix."""
        state = self._state
        cols, col_idx, oh_groups = state["cols"], state["col_idx"], state["oh_groups"]
        n = x.shape[0]
        out = np.zeros((n, len(cols)), dtype=np.float32)
        for j, (c, lo, hi) in enumerate(_CONTINUOUS):
            out[:, col_idx[c]] = np.clip(x[:, j], lo, hi)
        for j, (c, k) in enumerate(_INTEGERS):
            out[:, col_idx[c]] = np.clip(np.rint(x[:, 48 + j]), 0, k - 1)
        for j, c in enumerate(_BOOLEANS):
            out[:, col_idx[c]] = (x[:, 52 + j] >= 0.5).astype(np.float32)
        for j, group in enumerate(oh_groups):
            k = len(group)
            cat = np.clip(np.rint(x[:, 55 + j]).astype(int), 0, k - 1)
            idx = np.array([col_idx[c] for c in group])
            out[np.arange(n), idx[cat]] = 1.0
        return out

    def _raw(self, X: torch.Tensor):
        """Natural design tensor -> ``(mass (n,), constraints (n, K))``."""
        state = self._ensure()
        x = X.detach().cpu().numpy().astype(np.float64)
        t = torch.from_numpy(self._decode(x)).to(self._device)
        with torch.no_grad():
            out = state["fn"](t).detach().cpu().numpy().astype(np.float64)
        return out[:, state["mass_i"]], out[:, state["con_i"]]

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        X = self.enforce_variable_types(X.to(torch.float64))
        state = self._ensure()
        mass, cons = self._raw(X)
        pen = (np.maximum(cons, 0.0) / state["scale"][None, :]).mean(axis=1)
        fx = torch.tensor(-(mass + _LAMBDA * pen), dtype=torch.float64).reshape(-1, 1)
        return None, fx  # maximize
