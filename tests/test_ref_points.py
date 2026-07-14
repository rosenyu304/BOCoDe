"""Hypervolume reference points must be a fixed property of the problem.

A hypervolume is only meaningful relative to a reference point. If the reference
point is derived from the data an algorithm happened to observe, then every
algorithm is scored against a *different* reference point and the hypervolumes are
not comparable -- which is the entire purpose of the metric. These tests pin that
down:

1. every multi-objective problem exposes a well-formed ``ref_point``;
2. the reference point is actually dominated by (worse than) sampled points, so it
   sits inside the attainable objective region and the hypervolume is not trivially
   zero for every algorithm;
3. the hypervolume of a *fixed* dataset is identical no matter which algorithm
   object computes it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

import bocode
from algorithms._bo_utils import MultiObjectiveProblem

#: Problems that legitimately have no reference point yet. Anything added here
#: must come with a documented reason -- an entry means "hypervolume for this
#: problem is not comparable across algorithms and must not be reported".
NO_REF_POINT = {
    # EngiBench + a native ngspice are needed to evaluate it at all, so no
    # reference point could be derived; its objective sign convention is also
    # unverified. See PowerElectronics' docstring.
    "PowerElectronics",
}

#: Problems whose *published* reference point is tight enough that a plain random
#: sample of the box does not reach it (the reference point sits just outside the
#: Pareto front, as Tanabe & Ishibuchi intend, so only near-Pareto points score).
#: Their hypervolume is still perfectly comparable across algorithms -- it is just
#: 0 until an algorithm finds a good point -- so they are exempted only from the
#: "a uniform sample scores > 0" check, not from any of the others.
TIGHT_REF_POINT = {"RE42", "CRE32"}

MO_PROBLEMS = sorted(
    set(bocode.get_multi_objective_unconstrained())
    | set(bocode.get_multi_objective_constrained())
)


def _make(name):
    try:
        return bocode.get_problem(name)()
    except ImportError as exc:  # optional extra not installed in this environment
        pytest.skip(f"{name} requires an optional dependency: {exc}")


def _sample(problem, n=256, seed=0):
    """A fixed, seeded sample of the box, and its objectives (maximization frame)."""
    X = problem.sample(n, seed=seed)
    Y, C = problem.evaluate(X)
    return Y.double(), C.double()


def _big_sample(problem, n=50_000, seed=0):
    """A large uniform sample, for problems whose reference point is deliberately tight."""
    torch.manual_seed(seed)
    X = torch.rand(n, problem.dim, dtype=torch.double)
    Y, C = problem.evaluate(problem.scale(X))
    return Y.double(), C.double()


@pytest.mark.parametrize("name", MO_PROBLEMS)
def test_ref_point_is_defined(name):
    """Every multi-objective problem defines a fixed, well-formed reference point."""
    problem = _make(name)
    if name in NO_REF_POINT:
        pytest.xfail(f"{name} has no reference point yet (see NO_REF_POINT)")

    assert problem.ref_point is not None, (
        f"{name} has no ref_point. Without one the algorithm harness infers one "
        f"per run, so its hypervolume is not comparable across algorithms."
    )
    rp = torch.as_tensor(problem.ref_point, dtype=torch.double)
    assert rp.shape == (problem.num_objectives,)
    assert torch.isfinite(rp).all(), f"{name}: ref_point is not finite: {rp}"


@pytest.mark.parametrize("name", sorted(set(MO_PROBLEMS) - NO_REF_POINT))
def test_ref_point_is_dominated_by_sampled_points(name):
    """The reference point must be worse than points that are actually attainable.

    Otherwise it lies outside the objective region and every algorithm scores a
    hypervolume of exactly 0.
    """
    problem = _make(name)
    # Deliberately tight published reference points sit just outside the Pareto
    # front, so reach for them with a much larger sample rather than exempting them.
    Y, _ = _big_sample(problem) if name in TIGHT_REF_POINT else _sample(problem, n=256)
    rp = torch.as_tensor(problem.ref_point, dtype=torch.double)

    dominating = (Y > rp).all(dim=1)
    assert dominating.any(), (
        f"{name}: no sampled point dominates ref_point={rp.tolist()}, so its "
        f"hypervolume is 0 for every algorithm. The reference point is outside "
        f"the attainable objective region."
    )


@pytest.mark.parametrize("name", sorted(set(MO_PROBLEMS) - NO_REF_POINT))
def test_hypervolume_is_computable_and_positive(name):
    """A random sample yields a computable, strictly positive hypervolume.

    For *constrained* problems the hypervolume is taken over the feasible points
    only, exactly as the algorithms compute it. Several of the constrained suites
    (MODAct, Mazda, TwoBarTruss, CRE21) have feasibility ratios so low that a plain
    random sample of a few hundred points finds no feasible point at all -- there
    the check is that the hypervolume is computable and 0, not that it is positive,
    because that 0 comes from infeasibility and not from a broken reference point
    (which :func:`test_ref_point_is_dominated_by_sampled_points` already rules out).
    """
    problem = _make(name)
    Y, C = _big_sample(problem) if name in TIGHT_REF_POINT else _sample(problem, n=256)
    rp = torch.as_tensor(problem.ref_point, dtype=torch.double)

    feasible = (
        (C <= 0).all(dim=1) if C.numel() else torch.ones(len(Y), dtype=torch.bool)
    )
    scoring = feasible & (Y > rp).all(dim=1)

    hv = (
        DominatedPartitioning(ref_point=rp, Y=Y[scoring]).compute_hypervolume().item()
        if scoring.any()
        else 0.0
    )
    assert np.isfinite(hv), f"{name}: hypervolume is not finite ({hv})"
    assert hv >= 0.0

    if scoring.any():
        assert hv > 0.0, f"{name}: feasible points dominate the ref point but HV == 0"
    else:
        # No feasible & ref-dominating point in this sample -> HV is legitimately 0.
        assert not feasible.any() or problem.num_constraints > 0, (
            f"{name}: unconstrained problem scored no points against its ref point"
        )


@pytest.mark.parametrize("name", ["RE33", "Penicillin", "CarSideImpact", "DiscBrake"])
def test_hypervolume_is_algorithm_independent(name):
    """The same dataset gets the same hypervolume, whichever algorithm scores it.

    This is the regression test for the bug this module exists for: the reference
    point used to come from ``nadir - 0.1 * span`` of whatever data the *calling
    algorithm* had seen, so a BO method (which had only its n_init design) and
    random search (which had all its samples) scored the *same* Pareto set
    differently.
    """
    problem = _make(name)
    Y, C = _sample(problem, n=128)
    feasible = (
        (C <= 0).all(dim=1) if C.numel() else torch.ones(len(Y), dtype=torch.bool)
    )

    def hv_as_seen_by(observed: torch.Tensor) -> float:
        """Score the *same* Pareto set, from an algorithm that observed ``observed``."""
        obj = MultiObjectiveProblem(problem)
        rp = obj.hv_ref_point(observed)  # the only run-dependent input there ever was
        return (
            DominatedPartitioning(ref_point=rp, Y=Y[feasible])
            .compute_hypervolume()
            .item()
        )

    # Three algorithms with three very different views of the data: a 20-point
    # initial design, a 64-point one, and the full sample.
    hvs = [hv_as_seen_by(Y[:20]), hv_as_seen_by(Y[:64]), hv_as_seen_by(Y)]
    assert hvs[0] == hvs[1] == hvs[2], (
        f"{name}: hypervolume depends on which algorithm computes it: {hvs}. The "
        f"reference point is being inferred from run data instead of taken from "
        f"the problem."
    )


def test_result_roundtrip_saves_constraints(tmp_path):
    """``Result`` persists constraint values, so feasible HV is recomputable offline."""
    from algorithms._bo_utils import Result

    problem = _make("CarSideImpact")
    Y, C = _sample(problem, n=32)
    X = problem.sample(32, seed=0)

    res = Result("test", "CarSideImpact", seed=0)
    res.record(0.0)
    res.set_history(X, Y, n_init=8, c=C)

    path = tmp_path / "run.npz"
    res.to_npz(str(path))
    data = np.load(path)

    assert "c" in data.files, "Result.to_npz must save the constraint values"
    assert data["c"].shape == (32, problem.num_constraints)
    np.testing.assert_allclose(data["c"], C.numpy(), rtol=1e-6)

    # The saved run is enough to recompute the feasible hypervolume offline.
    rp = torch.as_tensor(problem.ref_point, dtype=torch.double)
    y = torch.as_tensor(data["y"], dtype=torch.double)
    c = torch.as_tensor(data["c"], dtype=torch.double)
    feasible = (c <= 0).all(dim=1)
    hv = DominatedPartitioning(ref_point=rp, Y=y[feasible]).compute_hypervolume().item()
    assert np.isfinite(hv) and hv >= 0.0


def test_unconstrained_result_still_roundtrips():
    """Backwards compatibility: an unconstrained run saves an empty (n, 0) block."""
    from algorithms._bo_utils import Result

    res = Result("test", "RE21", seed=0)
    res.record(0.0)
    res.set_history(np.zeros((5, 4)), np.zeros((5, 2)), n_init=2)
    d = res.to_dict()
    assert d["c"].shape == (5, 0)
