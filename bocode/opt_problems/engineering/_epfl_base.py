"""Shared implementation of the EPFL AIG logic-synthesis sequence-optimization problems.

See :mod:`bocode.opt_problems.engineering.EPFLSeqOpt` for the benchmark problems, the
search space and the setup instructions. This module holds the (dependency-free) static
description of the 56-D search space plus the glue that drives Berkeley ABC through
MCBO's EDA utility modules.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import torch

from ...base import BenchmarkProblem

LUT_INPUTS = 6  # MCBO task_factory default
REF_SEQ = "resyn2"  # reference sequence for the ratio objective
PENALTY_OBJ = 2.0  # score of an abc crash: lut/ref_lut + lev/ref_lev with both == ref

# ---------------------------------------------------------------------------
# STATIC SEARCH SPACE (56-D), extracted from MCBO's OptimHyperSpace for
# operator space 'basic', sequence pattern 'basic_w_post_map' and hyperparameter
# space 'boils_hyp_op_space'. It is identical for every circuit, and it is
# cross-checked against the live MCBO objects on the first evaluation.
# ---------------------------------------------------------------------------
# The 14 operators of the 'basic' operator space, in ``all_operators`` order: indices
# 0-9 are pre-mapping, 10-11 mapping, 12-13 post-mapping.
OPERATORS = [
    "rewrite_wo_z",
    "rewrite_w_z",
    "refactor_wo_z",
    "refactor_w_z",
    "resub",
    "balance",
    "&blut",
    "&sopb",
    "&dsdb",
    "fraig",
    "if",
    "if -a",
    "speedup_if",
    "mfs2_lutpack",
]

# 'basic_w_post_map' = ([pre]*7 + [map] + [post, post]) * 2 -> 20 sequence slots.
SEQ_PATTERN = ([0] * 7 + [1] + [2, 2]) * 2
# Operator indices allowed in a slot of each type.
TYPE_CATEGORIES = {0: list(range(0, 10)), 1: [10, 11], 2: [12, 13]}

# The 36 operator hyperparameters, in MCBO's order (grouped by operator, following
# ``all_operators``): ``(name, "bool")`` or ``(name, "int", lb, ub)``.
HYPERPARAMS = [
    ("rewrite_wo_z -l", "bool"),
    ("rewrite_w_z -l", "bool"),
    ("refactor_wo_z N", "int", 2, 15),
    ("refactor_wo_z -l", "bool"),
    ("refactor_w_z N", "int", 2, 15),
    ("refactor_w_z -l", "bool"),
    ("resub K", "int", 5, 8),
    ("resub N", "int", 0, 3),
    ("resub -l", "bool"),
    ("resub -z", "bool"),
    ("balance -l", "bool"),
    ("balance -d", "bool"),
    ("balance -s", "bool"),
    ("balance -x", "bool"),
    ("&blut C", "int", 1, 8),
    ("&blut -a", "bool"),
    ("&sopb C", "int", 8, 100),
    ("&dsdb C", "int", 8, 100),
    ("fraig -r", "bool"),
    ("if C", "int", 4, 1024),
    ("if F", "int", 0, 2),
    ("if A", "int", 0, 4),
    ("if -a C", "int", 4, 1024),
    ("if -a F", "int", 0, 2),
    ("if -a A", "int", 0, 4),
    ("speedup_if speedup P", "int", 5, 20),
    ("speedup_if speedup N", "int", 1, 5),
    ("speedup_if if C", "int", 4, 256),
    ("speedup_if if F", "int", 0, 2),
    ("mfs2_lutpack mfs2 W", "int", 2, 200),
    ("mfs2_lutpack mfs2 M", "int", 300, 1000),
    ("mfs2_lutpack mfs2 D", "int", 0, 20),
    ("mfs2_lutpack mfs2 -a", "bool"),
    ("mfs2_lutpack lutpack N", "int", 2, 16),
    ("mfs2_lutpack lutpack S", "int", 0, 3),
    ("mfs2_lutpack lutpack -z", "bool"),
]

SEQ_LEN = len(SEQ_PATTERN)  # 20
DIM = SEQ_LEN + len(HYPERPARAMS)  # 56


def _space():
    """``(bounds, variable_types)`` of the 56-D space, in natural units.

    Sequence slots take the *operator index* itself as their level (e.g. ``{10, 11}`` for
    a mapping slot), integer hyperparameters their own ``[lb, ub]`` range, booleans
    ``{0, 1}``.
    """
    bounds, vtypes = [], []
    for op_type in SEQ_PATTERN:
        cats = TYPE_CATEGORIES[op_type]
        bounds.append((float(min(cats)), float(max(cats))))
        vtypes.append([float(c) for c in cats])
    for spec in HYPERPARAMS:
        if spec[1] == "bool":
            bounds.append((0, 1))
        else:
            bounds.append((spec[2], spec[3]))
        vtypes.append("integer")
    return bounds, vtypes


# ---------------------------------------------------------------------------
# MCBO / ABC glue.
# ---------------------------------------------------------------------------
_MODULES: dict = {}  # lazy import cache (shared by all 20 circuits)


def _resolve(env_var: str, default: str, check) -> str | None:
    path = os.environ.get(env_var, "") or default
    return os.path.abspath(path) if path and check(path) else None


def get_modules() -> dict:
    """Import MCBO's gpytorch-free EDA utility modules (cached).

    MCBO reads three path files *at import time*, so they are written first. MCBO itself
    is not importable in a modern environment (its package ``__init__`` pins gpytorch-1.6
    era APIs), so the heavy ``mcbo`` / ``mcbo.tasks`` package ``__init__``s are stubbed
    out; every other ``__init__`` on the path is empty.
    """
    if _MODULES:
        return _MODULES
    import importlib

    circuits = _resolve("BOCODE_EPFL_CIRCUITS", "", os.path.isdir)
    abc_bin = _resolve("BOCODE_ABC_BIN", "", os.path.isfile)
    mcbo_root = _resolve("BOCODE_MCBO_ROOT", "", os.path.isdir)
    if circuits is None or abc_bin is None or mcbo_root is None:
        raise ImportError(
            "The AIG logic-synthesis problems need three external assets that are too "
            "large to ship with BoCoDe:\n"
            "  BOCODE_EPFL_CIRCUITS -> a directory of EPFL .blif circuits "
            "(https://github.com/lsils/benchmarks)\n"
            "  BOCODE_ABC_BIN       -> a compiled Berkeley ABC binary "
            "(https://github.com/berkeley-abc/abc)\n"
            "  BOCODE_MCBO_ROOT     -> the MCBO checkout that provides the EDA utilities "
            "(https://github.com/huawei-noah/HEBO/tree/master/MCBO)\n"
            f"Resolved: circuits={circuits}, abc={abc_bin}, mcbo={mcbo_root}."
        )
    results_root = os.environ.get(
        "BOCODE_EPFL_RESULTS", os.path.join(os.path.dirname(circuits), "epfl_results")
    )
    os.makedirs(results_root, exist_ok=True)
    # MCBO reads these txt files AT IMPORT TIME (utils_operators.py); rewrite only when
    # stale so concurrent workers do not race.
    eda_dir = os.path.join(mcbo_root, "mcbo", "tasks", "eda_seq_opt")
    for fname, val in [
        ("circuits_path.txt", circuits),
        ("abc_release_path.txt", abc_bin),
        ("results_root_path.txt", results_root),
    ]:
        fpath = os.path.join(eda_dir, fname)
        try:
            cur = open(fpath).readline().strip() if os.path.exists(fpath) else None
        except OSError:
            cur = None
        if cur != val:
            with open(fpath, "w") as fh:
                fh.write(val + "\n")
    if mcbo_root not in sys.path:
        sys.path.insert(0, mcbo_root)
    for name, sub in [("mcbo", "mcbo"), ("mcbo.tasks", os.path.join("mcbo", "tasks"))]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [os.path.join(mcbo_root, sub)]
            mod.__package__ = name
            sys.modules[name] = mod
    base = "mcbo.tasks.eda_seq_opt.utils."
    _MODULES.update(
        utils=importlib.import_module(base + "utils"),
        uops=importlib.import_module(base + "utils_operators"),
        uhyp=importlib.import_module(base + "utils_operators_hyp"),
        ubis=importlib.import_module(base + "utils_build_in_seq"),
        udg=importlib.import_module(base + "utils_design_groups"),
    )
    return _MODULES


def _convert_array_to_operator_seq(sequence, hyperparams, operator_space, final_map_op):
    """Verbatim copy of ``EDASeqOptimization.convert_array_to_operator_seq``
    (MCBO ``eda_seq_opt_task.py``) — the class itself imports gpytorch-era dependencies,
    so it cannot be imported. Only the name resolution differs."""
    uops = get_modules()["uops"]
    operator_sequence = []
    print_stat_stages = [0]
    for ind in sequence:
        op_class = operator_space.all_operators[int(round(float(ind)))]
        op_hyp = {}
        for kw in op_class.hyperparams:
            if kw in hyperparams:
                op_hyp[kw] = hyperparams[kw]
            if uops.is_lut_mapping_hyperparam(kw):
                op_hyp[kw] = LUT_INPUTS
        operator_sequence.append(op_class(**op_hyp))
        if (
            operator_sequence[-1].op_type
            in [uops.MAPPING_OPERATOR_TYPE, uops.POST_MAPPING_OPERATOR_TYPE]
        ) or final_map_op:
            print_stat_stages.append(print_stat_stages[-1] + 1)
        else:
            print_stat_stages.append(print_stat_stages[-1])
    valid_seq_operator, is_new_op = uops.make_operator_sequence_valid(
        operator_sequence, final_mapping_op=final_map_op
    )
    return valid_seq_operator, np.array(print_stat_stages), is_new_op


class EPFLSeqOptProblem(BenchmarkProblem):
    """One EPFL circuit as a 56-D mixed logic-synthesis problem.

    Subclasses set ``circuit`` (the EPFL benchmark name). See
    :mod:`bocode.opt_problems.engineering.EPFLSeqOpt` for the full description.
    """

    available_dimensions = DIM
    num_objectives = 1
    num_constraints = 0

    circuit: str = ""

    def __init__(self) -> None:
        bounds, self.variable_types = _space()
        super().__init__(dim=DIM, num_objectives=1, num_constraints=0, bounds=bounds)
        self._state = None

    # -- lazy backend ----------------------------------------------------------
    def _ensure(self):
        if self._state is not None:
            return self._state
        m = get_modules()
        op_space = m["uops"].get_operator_space("basic")
        pattern = m["uops"].get_seq_operators_pattern("basic_w_post_map")
        hyp_space = m["uhyp"].get_operator_hyperparms_space("boils_hyp_op_space")
        design_files = m["udg"].get_designs_path(self.circuit)
        if not design_files or not os.path.exists(design_files[0]):
            raise ImportError(
                f"The EPFL circuit '{self.circuit}.blif' was not found under "
                f"BOCODE_EPFL_CIRCUITS. Get the circuits from "
                "https://github.com/lsils/benchmarks."
            )
        # 'basic_w_post_map' ends with a post-mapping operator -> final_mapping_op None.
        if pattern.not_mapped_at_the_end():
            raise RuntimeError("unexpected MCBO pattern: not mapped at the end")
        self._check_space(op_space, pattern, hyp_space)
        self._state = {
            "op_space": op_space,
            "design_file": design_files[0],
            "refs": None,
        }
        return self._state

    @staticmethod
    def _check_space(op_space, pattern, hyp_space) -> None:
        """Guard the hardcoded search space against MCBO drift."""
        live_ops = [op.op_id for op in op_space.all_operators]
        if live_ops != OPERATORS or list(pattern.pattern) != SEQ_PATTERN:
            raise RuntimeError(
                "The MCBO operator space has changed; the AIG problems' hardcoded 56-D "
                f"search space no longer matches it (operators {live_ops}, pattern "
                f"{list(pattern.pattern)})."
            )
        live_hyps = []
        for algo in op_space.all_operators:
            for p in hyp_space.all_hyps[algo.op_id]:
                if p["type"] == "bool":
                    live_hyps.append((p["name"], "bool"))
                else:
                    live_hyps.append((p["name"], "int", int(p["lb"]), int(p["ub"])))
        if live_hyps != HYPERPARAMS:
            raise RuntimeError(
                "The MCBO 'boils_hyp_op_space' hyperparameters have changed; the AIG "
                "problems' hardcoded 56-D search space no longer matches them."
            )

    def _refs(self):
        """``(ref_lut, ref_level)`` of the resyn2 reference sequence (disk-cached)."""
        state = self._ensure()
        if state["refs"] is None:
            m = get_modules()
            ref1, ref2, _ = m["ubis"].get_ref(
                design_file=state["design_file"],
                lut_inputs=LUT_INPUTS,
                ref_abc_seq=REF_SEQ,
                evaluator="abc",
                n_eval_ref=1,
            )
            state["refs"] = (float(ref1), float(ref2))
        return state["refs"]

    @staticmethod
    def _unmap(row: np.ndarray):
        """A snapped design (natural units) -> ``(operator index sequence, hyperparams)``."""
        seq = np.rint(row[:SEQ_LEN]).astype(int)
        hyps = {}
        for j, spec in enumerate(HYPERPARAMS):
            value = row[SEQ_LEN + j]
            if spec[1] == "bool":
                hyps[spec[0]] = bool(int(round(float(value))))
            else:
                hyps[spec[0]] = int(np.clip(round(float(value)), spec[2], spec[3]))
        return seq, hyps

    def _one(self, row: np.ndarray) -> float:
        """One design -> the MCBO objective ``lut/ref_lut + level/ref_level`` (minimize)."""
        from subprocess import CalledProcessError

        m = get_modules()
        state = self._ensure()
        ref1, ref2 = self._refs()
        seq, hyps = self._unmap(row)
        ops, _stages, is_new = _convert_array_to_operator_seq(
            seq, hyps, state["op_space"], None
        )
        try:
            # obj_func -> keep the best intermediate print_stats stage (MCBO's
            # return_best_intermediate=True default). print_stat_stages is None because
            # 'basic_w_post_map' has contains_free=False.
            lut, lev, _extra = m["utils"].get_design_prop(
                seq=[op.op_str() for op in ops],
                design_file=state["design_file"],
                evaluator="abc",
                print_stat_stages=None,
                new_op=is_new,
                obj_func=lambda a, b: a / ref1 + b / ref2,
            )
        except CalledProcessError as exc:  # abc crashed on an unlucky sequence
            print(
                f"[AIG_{self.circuit}] abc failed (rc={exc.returncode}); "
                f"penalizing with the reference sequence's score ({PENALTY_OBJ})"
            )
            return PENALTY_OBJ
        return lut / ref1 + lev / ref2

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        X = self.enforce_variable_types(X.to(torch.float64))
        x = X.detach().cpu().numpy().astype(np.float64)
        out = np.array([-self._one(row) for row in x], dtype=np.float64)  # maximize
        return None, torch.tensor(out, dtype=torch.float64).reshape(-1, 1)
