"""Tests for the HPO-B continuous-surrogate pools (``HPOBSurr_*``).

Same tasks / mixed-integer search spaces as the discrete ``HPOB_*`` pools, but ``evaluate`` is an
XGBoost surrogate over the pool instead of nearest-config lookup — a smooth, non-flat landscape
for continuous-relaxation / mixed-integer BO. Requires ``xgboost``.
"""
import numpy as np
import pytest
import torch

import bocode
from bocode.registry import PROBLEM_REGISTRY

xgboost = pytest.importorskip("xgboost")

SURR_NAMES = sorted(n for n in PROBLEM_REGISTRY if n.startswith("HPOBSurr_"))


def test_all_surrogate_tasks_registered():
    # one surrogate class per discrete HPOB_* task
    disc = {n[len("HPOB_"):] for n in PROBLEM_REGISTRY if n.startswith("HPOB_")}
    surr = {n[len("HPOBSurr_"):] for n in SURR_NAMES}
    assert surr == disc and len(surr) == 92


def test_surrogate_is_mixed_and_same_space_as_discrete():
    s = bocode.HPOBSurr_5636_146064()
    d = bocode.HPOB_5636_146064()
    assert s.dim == d.dim
    assert s.bounds == d.bounds
    assert s.is_mixed_variable
    assert s.resolved_variable_types() == d.resolved_variable_types()


def test_surrogate_evaluate_deterministic():
    p = bocode.HPOBSurr_5636_146064()
    X = torch.tensor(np.random.RandomState(0).rand(64, p.dim))
    y1, _ = p.evaluate(X)
    y2, _ = p.evaluate(X)
    assert torch.allclose(y1, y2)                     # fixed-seed surrogate, cached
    assert y1.shape == (64, 1)


def test_surrogate_is_not_flat():
    # a smooth surrogate over random points must span a real accuracy range (not collapse)
    p = bocode.HPOBSurr_5636_146064()
    X = torch.tensor(np.random.RandomState(1).rand(400, p.dim))
    y, _ = p.evaluate(X)
    assert float(y.max() - y.min()) > 0.05


def test_surrogate_snaps_discrete_dims():
    # integer/discrete dims are snapped to allowed levels before the surrogate query,
    # so two inputs differing only within a discrete cell give the same value
    p = bocode.HPOBSurr_5636_146064()
    types = p.resolved_variable_types()
    disc = next(j for j, t in enumerate(types) if t != "continuous")
    base = torch.tensor(np.random.RandomState(2).rand(1, p.dim))
    a = base.clone(); b = base.clone()
    allowed = sorted(float(v) for v in types[disc])
    a[0, disc] = allowed[0] + 1e-4
    b[0, disc] = allowed[0] - 1e-4 if allowed[0] > 1e-4 else allowed[0] + 1e-4
    ya, _ = p.evaluate(a)
    yb, _ = p.evaluate(b)
    assert torch.allclose(ya, yb)
