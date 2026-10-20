"""Tests for the harder multi-objective synthetic suites.

ZDT4/ZDT6, DTLZ6/DTLZ7, WFG1-9, MW2/4-6/8-14, DAS-CMOP1-9 and LIR-CMOP1-14.

* every problem is reachable by name, evaluates to finite float64 tensors of the
  right shape, and has a fixed, well-formed reference point;
* ``PINNED`` holds each problem's objectives/constraints (minimization frame,
  constraints feasible ``<= 0``) at one fixed point of the box, computed with the
  reference implementations: pymoo 0.6.1.3 (ZDT, DTLZ, WFG, MW, DAS-CMOP; DAS-CMOP at
  its default difficulty 8) and an independent transcription of the PlatEMO code for
  LIR-CMOP. They pin BoCoDe's port to the reference values;
* analytically known Pareto-optimal points land on the published fronts and dominate
  the reference point.
"""

from __future__ import annotations

import math

import pytest
import torch

import bocode

NAMES = (
    ["ZDT4", "ZDT6", "DTLZ6", "DTLZ7"]
    + [f"WFG{i}" for i in range(1, 10)]
    + [f"MW{i}" for i in (2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)]
    + [f"DASCMOP{i}" for i in range(1, 10)]
    + [f"LIRCMOP{i}" for i in range(1, 15)]
)

# fmt: off
PINNED = {
    "ZDT4": ((0.6180339887498949, 158.8194535826237,), ()),
    "ZDT6": ((0.9789148148973197, 8.40190829952836,), ()),
    "DTLZ6": ((5.296021058479618, 2.310187114770798, 8.445785009797833,), ()),
    "DTLZ7": ((0.6180339887498949, 0.2360679774997898, 19.41848761817126,), ()),
    "WFG1": ((2.8820873384856114, 0.9820773521324854, 0.9921051907023839,), ()),
    "WFG2": ((0.8744161236948389, 0.7783685592908863, 4.532968267822618,), ()),
    "WFG3": ((1.178800159356488, 1.330071264060726, 4.09751665134239,), ()),
    "WFG4": ((0.7164043494316825, 1.388997249094538, 6.088042986368188,), ()),
    "WFG5": ((1.7982201592063998, 2.757207291173043, 3.640614501947697,), ()),
    "WFG6": ((2.0044489777202337, 2.047140885485833, 4.644977213537713,), ()),
    "WFG7": ((0.8752037041313457, 1.3091538148488437, 6.293818700639212,), ()),
    "WFG8": ((1.7040758448489868, 1.8861001486550797, 5.330787939850939,), ()),
    "WFG9": ((2.1129904917216638, 3.808725346157165, 1.259112706890673,), ()),
    "MW2": ((0.6180339887498949, 2.549736870120473,), (2.1616835471656866,)),
    "MW4": ((0.5559880615056348, 0.17180975967482479, 1.177601611608102,), (0.9052208567609589,)),
    "MW5": ((1.1419528035108297, 1.4525864048834538,), (1.147073442287553, -1.16656162965276, -3.110738285174307)),
    "MW6": ((2.768726510678754, 3.521874525713899,), (18.82853199864051,)),
    "MW8": ((3.0567184848379316, 1.1884543891920312, 4.793911715966998,), (32.410184475264096,)),
    "MW9": ((1.1419528035108297, 0.4633770366114982,), (-0.02001268051962891,)),
    "MW10": ((0.7478092598362996, 2.9912370393451977,), (-17.64158755242165, 20.96749721878322, 6.2799857788665845)),
    "MW11": ((1.4550767253338468, 1.8508861861752386,), (-2.987065360882575, 2.3811369957782653, -0.6062028013576174, 0.36671621556666406)),
    "MW12": ((1.1419528035108297, 0.646655243029369,), (-0.1280942725381113, 0.11936626936419759)),
    "MW13": ((0.9714509761692324, 2.258881423614119,), (-0.15305091002055948, -0.040946232641302704)),
    "MW14": ((0.9270509831248424, 0.3541019662496847, 4.962224355497945,), (1.192009507298149,)),
    "DASCMOP1": ((5.885510618437334, 5.885510618437334,), (-0.9057342725556126, 19.424270430362625, -176.51062874721742, -175.67729541388408, -145.60722462430186, -143.10722462430184, -143.94055795763518, -118.870487168053, -114.70382050138629, -113.87048716805296, -116.37048716805293)),
    "DASCMOP2": ((5.885510618437334, 5.481325251930016,), (-0.9057342725556126, 19.424270430362625, -163.51821761712168, -162.01124200627615, -134.29891918798657, -131.12527691047435, -131.2849679662955, -109.24628742551818, -104.40597848133929, -102.89900287049376, -104.72536059298157)),
    "DASCMOP3": ((5.885510618437334, 5.621077132591602,), (-0.9057342725556126, 19.424270430362625, -167.93350800089297, -166.65945219115005, -138.13191006900132, -135.19118759259175, -135.5837984495155, -112.4969788037763, -107.88958966070008, -106.6155338509572, -108.67481137454759)),
    "DASCMOP4": ((32.91618319507969, 32.91618319507969,), (-0.9057342725556126, 989.0814954786202, -6898.443108929514, -6897.609775596181, -6687.335220962315, -6684.835220962315, -6685.668554295648, -6480.393999661786, -6476.2273329951195, -6475.393999661785, -6477.893999661786)),
    "DASCMOP5": ((32.91618319507969, 32.511997828572376,), (-0.9057342725556126, 989.0814954786202, -6812.614682450556, -6811.107706839711, -6603.190900177137, -6600.017257899625, -6600.176948955446, -6397.933784570387, -6393.0934756262095, -6391.586500015363, -6393.412857737852)),
    "DASCMOP6": ((32.91618319507969, 32.65174970923396,), (-0.9057342725556126, 989.0814954786202, -6842.213888355217, -6840.939832545474, -6632.207806579041, -6629.267084102631, -6629.659694959555, -6426.368391469534, -6421.761002326458, -6420.486946516715, -6422.546224040305)),
    "DASCMOP7": ((30.733677982615365, 30.67794989261452, 31.35171197136526,), (-0.9057342725556126, 0.6407091449636951, 884.4192424104617, -2808.0955595230807, -2808.207015703082, -2806.8594915455806, -2762.4490370255903)),
    "DASCMOP8": ((31.114037970622306, 30.792389453766212, 31.413120754254095,), (-0.9057342725556126, 0.6407091449636951, 884.4192424104617, -2841.7481866898443, -2842.391483723556, -2841.1500211225803, -2796.2201301079012)),
    "DASCMOP9": ((7.436702142633111, 7.115053625777019, 7.735784926264902,), (-0.9057342725556126, 0.6407091449636951, 36.650412558349636, -151.83499099608767, -152.47828802979984, -151.2368254288241, -140.97296004206314)),
    "LIRCMOP1": ((3.0854072176489598, 1.8035020048462753,), (3.8508836894997414, 0.4630117209301438)),
    "LIRCMOP2": ((3.0854072176489598, 1.399316638338957,), (3.8508836894997414, 0.4630117209301438)),
    "LIRCMOP3": ((3.0854072176489598, 1.8035020048462753,), (3.8508836894997414, 0.4630117209301438, -0.40573427255561256)),
    "LIRCMOP4": ((3.0854072176489598, 1.399316638338957,), (3.8508836894997414, 0.4630117209301438, -0.40573427255561256)),
    "LIRCMOP5": ((16.36223008771866, 31.17910695887387,), (-252.52989139165777, -227.83582471565273)),
    "LIRCMOP6": ((16.36223008771866, 31.583292325381187,), (-247.52567550855193, -225.85291430200198)),
    "LIRCMOP7": ((16.36223008771866, 31.17910695887387,), (-257.66670324168865, -148.86682669252784, -132.48569996310815)),
    "LIRCMOP8": ((16.36223008771866, 31.583292325381187,), (-262.4171144803114, -151.7055204185326, -135.1812681327469)),
    "LIRCMOP9": ((16.907471033502294, 32.953219169386536,), (-495.63337543328686, -32.32224674947734)),
    "LIRCMOP10": ((16.907471033502294, 11.402286356588634,), (-85.44525041337086, -19.993238555522254)),
    "LIRCMOP11": ((16.907471033502294, 11.402286356588634,), (-149.68736961395436, -18.893238555522256)),
    "LIRCMOP12": ((16.907471033502294, 32.953219169386536,), (-487.30258606388446, -31.822246749477337)),
    "LIRCMOP13": ((13.01471604812101, 5.060131146587242, 20.4112351374677,), (-366147.3611493845, -369884.4361767404)),
    "LIRCMOP14": ((13.01471604812101, 5.060131146587242, 20.4112351374677,), (-366147.3611493845, -369884.4361767404, -370631.3264865094)),
}
# fmt: on


def _point(problem) -> torch.Tensor:
    """The fixed probe point used for ``PINNED``: a golden-ratio sweep of the box."""
    b = torch.tensor(problem.bounds, dtype=torch.double)
    frac = (
        torch.arange(1, problem.dim + 1, dtype=torch.double) * 0.6180339887498949
    ) % 1.0
    return (b[:, 0] + (b[:, 1] - b[:, 0]) * frac).unsqueeze(0)


@pytest.mark.parametrize("name", NAMES)
def test_listed_and_shapes(name):
    assert name in bocode.list_synthetic()
    p = bocode.get_problem(name)()
    assert p.num_objectives == type(p).num_objectives
    assert p.num_constraints == type(p).num_constraints
    X = p.sample(16, seed=0)
    Y, C = p.evaluate(X)
    assert Y.shape == (16, p.num_objectives) and Y.dtype == torch.float64
    assert C.shape == (16, p.num_constraints) and C.dtype == torch.float64
    assert torch.isfinite(Y).all() and torch.isfinite(C).all()
    rp = torch.as_tensor(p.ref_point, dtype=torch.double)
    assert rp.shape == (p.num_objectives,) and torch.isfinite(rp).all()


@pytest.mark.parametrize("name", NAMES)
def test_matches_reference_implementation(name):
    p = bocode.get_problem(name)()
    Y, C = p.evaluate(_point(p))
    f_ref, g_ref = PINNED[name]
    torch.testing.assert_close(
        -Y[0], torch.tensor(f_ref, dtype=torch.double), rtol=1e-9, atol=1e-12
    )
    if p.num_constraints:
        torch.testing.assert_close(
            C[0], torch.tensor(g_ref, dtype=torch.double), rtol=1e-9, atol=1e-9
        )


def _dominates_ref(p, Y):
    rp = torch.as_tensor(p.ref_point, dtype=torch.double)
    return bool((Y > rp).all(dim=1).all())


def test_zdt_fronts():
    x1 = torch.linspace(0.01, 1.0, 50, dtype=torch.double)
    for name, front in [("ZDT4", lambda f: 1 - f.sqrt()), ("ZDT6", None)]:
        p = bocode.get_problem(name)()
        X = torch.zeros(50, p.dim, dtype=torch.double)
        X[:, 0] = x1
        Y, _ = p.evaluate(X)
        f1, f2 = -Y[:, 0], -Y[:, 1]
        expected = front(f1) if front else 1 - f1**2  # g = 1 on the front
        torch.testing.assert_close(f2, expected)
        assert _dominates_ref(p, Y)


def test_dtlz6_front_is_unit_sphere_curve():
    p = bocode.get_problem("DTLZ6")()
    X = torch.zeros(64, p.dim, dtype=torch.double)
    X[:, :2] = torch.rand(
        64, 2, dtype=torch.double, generator=torch.Generator().manual_seed(0)
    )
    Y, _ = p.evaluate(X)
    torch.testing.assert_close((Y**2).sum(dim=1), torch.ones(64, dtype=torch.double))
    # degenerate: with g = 0 every theta_i (i >= 2) collapses to pi/4, so f1 == f2
    torch.testing.assert_close(Y[:, 0], Y[:, 1])
    assert _dominates_ref(p, Y)


def test_dtlz7_front():
    p = bocode.get_problem("DTLZ7")()
    X = torch.zeros(64, p.dim, dtype=torch.double)
    X[:, :2] = torch.rand(
        64, 2, dtype=torch.double, generator=torch.Generator().manual_seed(1)
    )
    f = -p.evaluate(X)[0]
    h = 3 - (f[:, :2] / 2 * (1 + torch.sin(3 * math.pi * f[:, :2]))).sum(dim=1)
    torch.testing.assert_close(f[:, 2], 2 * h)  # g = 1 on the front


def _wfg_optimal(p, n=64, seed=0):
    """Pareto-optimal WFG1-7 point: random position vars, distance vars at 0.35 * 2i."""
    g = torch.Generator().manual_seed(seed)
    z = torch.full((n, p.dim), 0.35, dtype=torch.double)
    z[:, : p.k] = torch.rand(n, p.k, dtype=torch.double, generator=g)
    return z * torch.arange(2, 2 * p.dim + 1, 2, dtype=torch.double)


@pytest.mark.parametrize("name", ["WFG4", "WFG5", "WFG6", "WFG7"])
@pytest.mark.parametrize("m", [2, 3])
def test_wfg_concave_fronts(name, m):
    p = bocode.get_problem(name)(num_objectives=m)
    Y, _ = p.evaluate(_wfg_optimal(p))
    S = torch.arange(2, 2 * m + 1, 2, dtype=torch.double)
    torch.testing.assert_close(
        ((-Y / S) ** 2).sum(dim=1), torch.ones(len(Y), dtype=torch.double)
    )
    assert _dominates_ref(p, Y)


def test_wfg3_linear_degenerate_front():
    p = bocode.get_problem("WFG3")()
    Y, _ = p.evaluate(_wfg_optimal(p))
    S = torch.arange(2, 2 * p.num_objectives + 1, 2, dtype=torch.double)
    torch.testing.assert_close(
        (-Y / S).sum(dim=1), torch.ones(len(Y), dtype=torch.double)
    )


def test_lircmop1_band_point_is_feasible_and_on_shifted_front():
    """g1 = g2 = 0.505 (inside the [0.5, 0.51] band) -> f = (x1, 1 - x1^2) + 0.505."""
    p = bocode.get_problem("LIRCMOP1")()
    x1 = torch.tensor([0.1, 0.3, 0.5], dtype=torch.double)
    X = torch.zeros(3, p.dim, dtype=torch.double)
    X[:, 0] = x1
    n_odd, n_even = len(range(2, p.dim, 2)), len(range(1, p.dim, 2))
    X[:, 2::2] = (torch.sin(0.5 * math.pi * x1) + math.sqrt(0.505 / n_odd)).unsqueeze(
        -1
    )
    X[:, 1::2] = (torch.cos(0.5 * math.pi * x1) - math.sqrt(0.505 / n_even)).unsqueeze(
        -1
    )
    Y, C = p.evaluate(X)
    assert (C <= 0).all()
    torch.testing.assert_close(-Y, torch.stack([x1 + 0.505, 1 - x1**2 + 0.505], dim=-1))
    assert _dominates_ref(p, Y)


@pytest.mark.parametrize("name,radius", [("LIRCMOP13", 1.7057), ("LIRCMOP14", None)])
def test_lircmop_sphere_fronts(name, radius):
    p = bocode.get_problem(name)()
    X = torch.full((32, p.dim), 0.5, dtype=torch.double)
    X[:, :2] = torch.rand(
        32, 2, dtype=torch.double, generator=torch.Generator().manual_seed(2)
    )
    Y, C = p.evaluate(X)
    torch.testing.assert_close(
        Y.norm(dim=1), torch.full((32,), 1.7057, dtype=torch.double)
    )
    # LIR-CMOP13: the g = 0 sphere is feasible. LIR-CMOP14: its third shell makes it
    # infeasible, so the true front is the shell boundary at radius 1.75.
    assert bool((C <= 0).all()) == (radius is not None)


@pytest.mark.parametrize("name", ["DASCMOP1", "DASCMOP7"])
def test_dascmop_difficulty_controls_feasibility(name):
    easy = bocode.get_problem(name)(difficulty=(0.0, 0.0, 0.0))
    hard = bocode.get_problem(name)()
    X = easy.sample(256, seed=0)
    feas = lambda p: float((p.evaluate(X)[1] <= 0).all(dim=1).double().mean())  # noqa: E731
    assert feas(easy) > feas(hard)
    with pytest.raises(ValueError):
        bocode.get_problem(name)(difficulty=17)


def test_scalable_objectives():
    for name in ["WFG1", "WFG9", "DTLZ6", "MW4", "MW8", "MW14"]:
        p = bocode.get_problem(name)(num_objectives=4)
        Y, _ = p.evaluate(p.sample(4, seed=0))
        assert Y.shape == (4, 4) and len(p.ref_point) == 4
    with pytest.raises(ValueError):
        bocode.get_problem("WFG2")(dim=23)  # l = dim - k must be even
    with pytest.raises(ValueError):
        bocode.get_problem("MW2")(num_objectives=3)
