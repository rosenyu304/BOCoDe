"""Tests for the LCBench and FixedHPO-B discrete mixed-variable HPO pools.

Both suites are candidate-pool benchmarks: construction loads a small in-repo
``.npz`` table (no network), and ``evaluate`` is a nearest-configuration lookup,
so it must be deterministic and only ever return values from the pool.
"""

import torch

import bocode
from bocode.registry import PROBLEM_REGISTRY

# All 13 mixed HPO-B-v3 search spaces + the two glmnet spaces meeting the
# 1000-eval bar. 5860 (glmnet, purely continuous, all pools < 1000) is skipped.
HPOB_SPACES = {
    "4796",
    "5527",
    "5636",
    "5859",
    "5889",
    "5891",
    "5906",
    "5965",
    "5970",
    "5971",
    "6766",
    "6767",
    "6794",
    "7607",
    "7609",
}
GLMNET_SPACES = {"5970", "6766"}  # alpha/lambda only: continuous, not mixed


def test_lcbench_all_35_datasets_registered():
    names = [n for n in PROBLEM_REGISTRY if n.startswith("LCBench")]
    assert len(names) == 35


def test_hpob_all_v3_spaces_covered():
    spaces = {n.split("_")[1] for n in PROBLEM_REGISTRY if n.startswith("HPOB_")}
    assert spaces == HPOB_SPACES


def test_lcbench_is_mixed_integer():
    p = bocode.LCBenchAdult()
    assert p.dim == 7
    vt = p.resolved_variable_types()
    # batch_size, max_units, num_layers are integers (ifBO, Table 3)
    assert [i for i, t in enumerate(vt) if t == "integer"] == [0, 3, 5]
    assert p.is_mixed_variable
    s = p.sample(8, seed=0)
    for j in (0, 3, 5):
        assert torch.equal(s[:, j], s[:, j].round())


def test_lcbench_evaluate_deterministic_pool_lookup():
    p = bocode.LCBenchAdult()
    X = p.scale(torch.rand(64, p.dim, dtype=torch.double))
    v1, c1 = p.evaluate(X)
    v2, _ = p.evaluate(X)
    assert torch.equal(v1, v2)
    assert v1.shape == (64, 1) and c1.shape == (64, 0)
    assert torch.isin(v1.flatten(), p.values.to(v1.dtype)).all()


def test_hpob_mixed_space_variable_types():
    p = bocode.HPOB_5527_31()  # svm: cost, gamma continuous; rest binary/degree grid
    assert p.dim == 8
    assert p.bounds == [(0.0, 1.0)] * 8
    assert p.is_mixed_variable
    vt = p.resolved_variable_types()
    assert vt[0] == "continuous" and vt[1] == "continuous"  # cost, gamma
    for j in (2, 4, 5, 6, 7):  # gamma.na, degree.na, kernel one-hots
        assert vt[j] != "continuous" and set(vt[j]) <= {0.0, 1.0}
    assert vt[3] != "continuous" and len(vt[3]) <= 4  # degree grid
    assert p.variable_names[0] == "cost"


def test_hpob_fallback_space_4796():
    """rpart.preproc has no 1000-eval task; included via the largest-pool fallback."""
    p = bocode.HPOB_4796_3549()
    assert p.dim == 3
    assert p.is_mixed_variable
    vt = p.resolved_variable_types()
    assert vt[0] != "continuous" and vt[1] != "continuous"  # minsplit, minbucket
    assert vt[2] == "continuous"  # cp
    assert len(p.values) == 300


def test_hpob_glmnet_spaces_are_continuous():
    """The two kept glmnet spaces are documented as NOT mixed (alpha, lambda)."""
    p = bocode.HPOB_5970_37()
    assert p.dim == 2 and not p.is_mixed_variable


def test_hpob_evaluate_deterministic_pool_lookup():
    p = bocode.HPOB_5906_3918()  # 16-dim xgboost, fallback pool
    X = torch.rand(64, p.dim, dtype=torch.double)
    v1, _ = p.evaluate(X)
    v2, _ = p.evaluate(X)
    assert torch.equal(v1, v2)
    assert torch.isin(v1.flatten(), p.values.to(v1.dtype)).all()
    # snapping to variable types keeps evaluation valid
    Xs = p.enforce_variable_types(X)
    v3, _ = p.evaluate(Xs)
    assert torch.isin(v3.flatten(), p.values.to(v3.dtype)).all()


def test_hpob_metadata_marks_mixed():
    meta = bocode.get_metadata("HPOB_6767_31")
    assert meta["input_type"] == "mixed"
    assert meta["dim"] == 18
    meta_glmnet = bocode.get_metadata("HPOB_6766_31")
    assert meta_glmnet["input_type"] == "continuous"
