"""Tests for the NEC2 Yagi-Uda antenna problem (YagiUda21)."""

import pytest
import torch

pytest.importorskip("PyNEC", reason="YagiUda21 needs the optional 'nec' extra")

import bocode  # noqa: E402

# A feasible 8-element design (6 directors, 9.5 mm tube) found by differential evolution.
X_REF = [
    6.0, 9.5, 0.4992319838135769, 0.4402378995664684, 0.19811749781490942,
    0.42102666972243563, 0.42304976281074586, 0.41768020483901, 0.3854281894124139,
    0.36489994486384164, 0.42162379958600044, 0.40679360272386134, 0.41725488460154975,
    0.3074604827141408, 0.1367890596846978, 0.27121001017706786, 0.2795884678413855,
    0.1340160118467726, 0.34452357578424536, 0.12480325941355241, 0.26463120729591244,
]  # fmt: skip
G_REF = 11.3556  # forward gain of X_REF, dBi


@pytest.fixture(scope="module")
def problem():
    return bocode.get_problem("YagiUda21")()


def test_metadata(problem):
    meta = bocode.get_metadata("YagiUda21")
    assert meta["dim"] == problem.dim == 21
    assert meta["num_constraints"] == problem.num_constraints == 3
    assert meta["extra"] == "nec"
    assert problem.is_mixed_variable


def test_shapes_dtype_and_types(problem):
    X = problem.sample(6, seed=0)
    assert torch.equal(X[:, 0], X[:, 0].round())  # director count is integer
    assert torch.isin(X[:, 1], torch.tensor([3.2, 4.8, 6.35, 9.5], dtype=X.dtype)).all()
    f, g = problem.evaluate(X)
    assert f.shape == (6, 1) and g.shape == (6, 3)
    assert f.dtype == g.dtype == torch.float64
    assert torch.isfinite(f).all() and torch.isfinite(g).all()


def test_deterministic(problem):
    X = problem.sample(4, seed=1)
    f1, g1 = problem.evaluate(X)
    f2, g2 = problem.evaluate(X)
    assert torch.equal(f1, f2) and torch.equal(g1, g2)


def test_inactive_directors_are_ignored(problem):
    x = torch.tensor([X_REF], dtype=torch.float64)
    x[0, 0] = 3.0  # only directors 1..3 active
    y = x.clone()
    y[0, 8:13] = 0.36  # lengths of directors 4..8
    y[0, 16:21] = 0.39  # gaps of directors 4..8
    assert torch.equal(problem.evaluate(x)[0], problem.evaluate(y)[0])
    assert torch.equal(problem.evaluate(x)[1], problem.evaluate(y)[1])


def test_boom_constraint_is_geometric(problem):
    x = torch.tensor([X_REF], dtype=torch.float64)
    _, g = problem.evaluate(x)
    n_dir = int(x[0, 0])
    boom = x[0, 4] + x[0, 13 : 13 + n_dir].sum()
    assert g[0, 2].item() == pytest.approx(boom.item() - 2.0, abs=1e-12)


def test_reference_design(problem):
    x = torch.tensor([X_REF], dtype=torch.float64)
    f, g = problem.evaluate(x)
    assert f.item() == pytest.approx(G_REF, abs=0.05)
    assert (g <= 0).all()  # feasible
    sim = problem.simulate(X_REF)
    assert sim["max_vswr"] <= 2.0 and sim["front_to_back_db"] >= 15.0
    assert sim["n_elements"] == 8


def test_more_directors_raise_gain_on_a_fixed_design(problem):
    # Yagi physics sanity: on a tidy uniform design, adding directors (longer boom)
    # increases forward gain, and the antenna radiates forwards (FB > 0).
    base = [1, 6.35, 0.50, 0.47, 0.20] + [0.43] * 8 + [0.25] * 8
    gains = []
    for n in (1, 3, 6):
        x = list(base)
        x[0] = n
        sim = problem.simulate(x)
        assert sim["front_to_back_db"] > 0
        gains.append(sim["gain_dbi"])
    assert gains[0] < gains[1] < gains[2]


def test_out_of_bounds_inputs_are_clipped(problem):
    x = torch.tensor([X_REF], dtype=torch.float64)
    y = x.clone()
    y[0, 13] = -0.5  # first director gap below its 0.10 lower bound
    z = x.clone()
    z[0, 13] = 0.10
    fy, gy = problem.evaluate(y)
    fz, gz = problem.evaluate(z)
    assert torch.equal(fy, fz) and torch.equal(gy, gz)


@pytest.fixture(scope="module")
def hifi():
    return bocode.get_problem("YagiUda21_HiFi")()


def test_hifi_shares_the_search_space(problem, hifi):
    assert hifi.dim == problem.dim and hifi.num_constraints == problem.num_constraints
    assert hifi.bounds == problem.bounds
    assert hifi.variable_types == problem.variable_types
    assert bocode.get_metadata("YagiUda21_HiFi")["extra"] == "nec"


def test_hifi_tracks_low_fidelity(problem, hifi):
    # Same specification at higher accuracy: the reference design stays feasible and
    # its gain moves by well under 1 dB; the boom constraint is purely geometric.
    x = torch.tensor([X_REF], dtype=torch.float64)
    f_lo, g_lo = problem.evaluate(x)
    f_hi, g_hi = hifi.evaluate(x)
    assert (g_hi <= 0).all()
    assert abs(f_hi.item() - f_lo.item()) < 0.5
    assert g_hi[0, 2].item() == pytest.approx(g_lo[0, 2].item(), abs=1e-12)


def test_front_to_rear_is_stricter(hifi):
    strict = bocode.get_problem("YagiUda21_HiFi")(front_to_rear=True)
    sim = strict.simulate(X_REF)
    assert "front_to_rear_db" not in hifi.simulate(X_REF)
    assert sim["front_to_back_db"] == pytest.approx(
        hifi.simulate(X_REF)["front_to_back_db"]
    )  # still the 180 deg value
    assert sim["front_to_rear_db"] <= sim["front_to_back_db"] + 1e-9
    _, g = strict.evaluate(torch.tensor([X_REF], dtype=torch.float64))
    assert g[0, 1].item() == pytest.approx(15.0 - sim["front_to_rear_db"])


def test_sample_gives_discrete_choices_equal_shares(problem):
    X = problem.sample(8000, seed=0)
    for d in (3.2, 4.8, 6.35, 9.5):
        assert (X[:, 1] == d).double().mean().item() == pytest.approx(0.25, abs=0.01)
    for k in range(1, 9):
        assert (X[:, 0] == k).double().mean().item() == pytest.approx(0.125, abs=0.01)
    assert torch.equal(problem.enforce_variable_types(X), X)  # already valid
    b = torch.tensor(problem.bounds, dtype=torch.float64)
    lo, hi = b[:, 0], b[:, 1]
    assert ((X >= lo) & (X <= hi)).all()
