"""Yagi-Uda wire antenna, NEC2 method of moments: 21-D mixed, 3 constraints.

Design a 2 m band (144-148 MHz) Yagi-Uda antenna for maximum forward gain under three
common build specifications: a direct 50 ohm feed with VSWR <= 2 across the whole band,
a front-to-back ratio (at 180 deg) of at least 15 dB, and a boom no longer than 2
wavelengths. Every evaluation is a full-wave method-of-moments solve of the wire
structure with NEC2 (via the ``PyNEC`` bindings to NEC2++), so mutual coupling between
the elements, the feed-point impedance and the radiation pattern all come from the
solver rather than from an array-factor approximation. This is the solver-backed
counterpart of the analytic ``AntennaArray200`` problem, so it needs the optional
``nec`` dependency::

    pip install 'bocode[nec]'

Two fidelities share one search space: ``YagiUda21`` (fast, ~12 ms: 15 segments,
VSWR at 3 frequencies) and ``YagiUda21_HiFi`` (~0.14 s: 31 segments, VSWR at 9
frequencies). Over 1500 random designs both have the same 0.93 % feasible fraction, agree
on feasibility for 99.9 % of designs and rank gain almost identically (Spearman 0.996,
median difference 0.07 dB). The pair still disagrees where it matters: of the two best
low-fidelity designs below, one stays feasible at high fidelity (13.53 dBi) and the other
just fails its 9-frequency VSWR check (2.03).

Geometry: straight, parallel, centre-fed thin wires along ``z``; the boom runs along
``x``. The reflector sits behind the driven element (at ``x = -S_ref``), the driven
element at ``x = 0``, and the directors in front of it. Free space, perfectly
conducting wires, 15 segments per element (odd, so the feed is the centre segment).
Lengths and spacings are in wavelengths at the design frequency ``f0 = 146 MHz``
(``lambda0 = 2.053 m``).

Design variables (21-D)::

    dim 0      : n_dir in {1, ..., 8}         number of directors        (integer)
    dim 1      : D in {3.2, 4.8, 6.35, 9.5}   element tube diameter, mm  (categorical:
                                              1/8, 3/16, 1/4 and 3/8 inch aluminium)
    dim 2      : L_ref in [0.45, 0.58]        reflector length            (lambda0)
    dim 3      : L_drv in [0.40, 0.54]        driven element length       (lambda0)
    dim 4      : S_ref in [0.10, 0.35]        reflector to driven spacing (lambda0)
    dims 5..12 : L_dir_k in [0.35, 0.48]      director k length           (lambda0)
    dims 13..20: S_dir_k in [0.10, 0.40]      gap from element k-1 to director k

The space is **conditional**: only the first ``n_dir`` director lengths and gaps are
used, and the rest are inactive, so a director count change switches whole blocks of
dimensions on or off (a hierarchical, mixed search space).

Objective (maximized): forward gain ``G_fwd`` in dBi at ``f0``, in the endfire
direction ``(theta, phi) = (90, 0)`` deg.

Constraints (feasible when ``<= 0``)::

    g0 = max_{f in {144, 146, 148} MHz} |Gamma(f)| - 1/3     VSWR <= 2 against 50 ohm
    g1 = 15 - FB                                              FB = G(0 deg) - G(180 deg),
                                                              clipped to [-40, 40] dB
    g2 = L_boom - 2.0                                         L_boom = S_ref + sum of the
                                                              active director gaps

The reflection coefficient form ``|Gamma| <= 1/3`` is used instead of VSWR because it
is bounded in ``[0, 1]`` (VSWR diverges as ``|Gamma| -> 1``), which keeps the
constraint surface well scaled for surrogate models. FB is the classic front-to-back
ratio at exactly 180 deg, not the stricter front-to-rear ratio (worst lobe over the rear
half plane), so optimizers may place a narrow null at 180 deg; this is the standard F/B
spec, kept for simplicity and speed. Continuous inputs are clipped to their bounds.

Reference points (measured with this implementation, PyNEC 2.3.4):

* Solver check: a lone 0.473 lambda half-wave dipole in the 4.8 mm tube (15 segments)
  gives 2.13 dBi and 71.5 - j1.2 ohm, against the textbook 2.15 dBi and ~73 ohm.
* Random designs: 3000 uniform samples give ``G_fwd`` from -18.7 to 13.0 dBi
  (median 7.4); only about 1 % satisfy all three constraints (|Gamma| alone ~10 %,
  FB alone ~9 %), so the feasible region is narrow and much of the initial design is
  usually infeasible.
* Best known feasible: ``G_fwd = 13.50 dBi`` (10 elements, all 8 directors, boom
  1.99 lambda, VSWR 1.66, FB 15.4 dB), from scipy differential evolution (popsize 15,
  150 generations, feasibility rule); a second seed reached 13.47 dBi. Both use the
  whole boom budget, so the boom constraint is active at the optimum.
* Discretization: 15 segments per element keeps every segment shorter than
  0.04 lambda and its length/radius ratio above 10 for every tube size (the NEC2
  thin-wire guidelines). Against a 121-segment reference, gain differs by a median
  0.12 dB over random designs and by at most 0.13 dB near the feasible region; FB and
  |Gamma| move more (up to ~2 dB and ~0.03), so the constraints, like the objective,
  are defined by this 15-segment model rather than by the converged physics.

Each evaluation is three NEC2 matrix solves (one per frequency) plus two far-field
directions:
about 12 ms on one CPU core, deterministic.

Sources:
G. J. Burke and A. J. Poggio. Numerical Electromagnetics Code (NEC) - Method of Moments. Technical Report UCID-18834, Lawrence Livermore National Laboratory, 1981.
T. C. A. Molteno. NEC2++: An NEC-2 compatible Numerical Electromagnetics Code. Electronics Technical Reports No. 2014-3, University of Otago, 2014.
E. A. Jones and W. T. Joines. Design of Yagi-Uda antennas using genetic algorithms. IEEE Transactions on Antennas and Propagation 45(9):1386-1392, 1997.
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem

_C0 = 299.792458  # speed of light [m * MHz]
_F0 = 146.0  # design (centre) frequency [MHz]
_F_LO, _F_HI = 144.0, 148.0  # band edges [MHz]
_LAMBDA0 = _C0 / _F0  # design wavelength [m]
_Z0 = 50.0  # feed-line impedance [ohm]

_N_DIR_MAX = 8  # maximum number of directors
_N_SEG = 15  # segments per element (odd -> centre-segment feed)
_DIAMETERS_MM = (3.2, 4.8, 6.35, 9.5)  # 1/8, 3/16, 1/4, 3/8 inch tube

_GAMMA_MAX = 1.0 / 3.0  # |Gamma| for VSWR = 2
_FB_MIN_DB = 15.0  # minimum front-to-back ratio [dB]
_BOOM_MAX = 2.0  # maximum boom length [lambda0]
_FB_CLIP_DB = 40.0  # FB clipped to [-40, 40] dB (deep nulls -> huge FB)
_G_FLOOR_DBI = -40.0  # NEC reports -999.99 dBi in exact nulls; floor it

_B_LREF = (0.45, 0.58)
_B_LDRV = (0.40, 0.54)
_B_SREF = (0.10, 0.35)
_B_LDIR = (0.35, 0.48)
_B_SDIR = (0.10, 0.40)
_CONT_BOUNDS = (
    [_B_LREF, _B_LDRV, _B_SREF] + [_B_LDIR] * _N_DIR_MAX + [_B_SDIR] * _N_DIR_MAX
)
_LO = np.array([b[0] for b in _CONT_BOUNDS])
_HI = np.array([b[1] for b in _CONT_BOUNDS])


def _load_pynec():
    try:
        from PyNEC import nec_context
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "The YagiUda21 problem requires the optional 'nec' dependency (PyNEC). "
            "Install it with: pip install 'bocode[nec]'"
        ) from exc
    return nec_context


class YagiUda21(BenchmarkProblem):
    """2 m band Yagi-Uda antenna, NEC2 method of moments (21 vars: 1 int, 1 cat, 19 cont).

    Maximize forward gain (dBi) subject to VSWR <= 2 over 144-148 MHz, front-to-back
    >= 15 dB and boom <= 2 wavelengths. Directors beyond ``n_dir`` are inactive.
    Needs ``pip install 'bocode[nec]'``.
    """

    available_dimensions = 21
    num_objectives = 1
    num_constraints = 3

    tags = {"single_objective", "constrained", "21D", "mixed", "conditional"}

    def __init__(self) -> None:
        self.variable_types = ["integer", list(_DIAMETERS_MM)] + ["continuous"] * (
            3 + 2 * _N_DIR_MAX
        )
        super().__init__(
            dim=21,
            num_objectives=1,
            num_constraints=3,
            bounds=[
                (1, _N_DIR_MAX),
                (min(_DIAMETERS_MM), max(_DIAMETERS_MM)),
                _B_LREF,
                _B_LDRV,
                _B_SREF,
            ]
            + [_B_LDIR] * _N_DIR_MAX
            + [_B_SDIR] * _N_DIR_MAX,
        )

    # -- one design -> (G_fwd, max |Gamma|, FB, boom) -------------------------------
    @staticmethod
    def _decode(x: np.ndarray):
        n_dir = int(np.clip(np.round(x[0]), 1, _N_DIR_MAX))
        diam_mm = float(x[1])
        x = x.copy()
        x[2:] = np.clip(x[2:], _LO, _HI)  # continuous dims to their bounds
        l_ref, l_drv, s_ref = float(x[2]), float(x[3]), float(x[4])
        l_dir = x[5 : 5 + n_dir]
        s_dir = x[5 + _N_DIR_MAX : 5 + _N_DIR_MAX + n_dir]
        lengths = np.concatenate([[l_ref, l_drv], l_dir])
        positions = np.concatenate([[-s_ref, 0.0], np.cumsum(s_dir)])
        boom = s_ref + float(np.sum(s_dir))
        return lengths, positions, diam_mm, boom

    # Fidelity knobs (overridden by YagiUda21_HiFi). With phi_step = 180 the pattern is
    # sampled at phi = 0 and 180 only, so "rear" is exactly 180 deg (classic F/B).
    _n_seg = _N_SEG  # segments per element (odd)
    _n_freq = 3  # frequencies spread evenly over 144-148 MHz (odd, centre = f0)
    _phi_step = 180.0  # azimuth step of the theta = 90 deg pattern cut [deg]

    def _solve(self, lengths, positions, diam_mm):
        nec_context = _load_pynec()
        n_seg, n_freq = self._n_seg, self._n_freq
        n_phi = int(round(360.0 / self._phi_step))
        ctx = nec_context()
        geo = ctx.get_geometry()
        radius = diam_mm / 2000.0  # mm diameter -> m radius
        for tag, (length, pos) in enumerate(
            zip(lengths, positions, strict=True), start=1
        ):
            half = 0.5 * length * _LAMBDA0
            xm = pos * _LAMBDA0
            geo.wire(tag, n_seg, xm, 0.0, -half, xm, 0.0, half, radius, 1.0, 1.0)
        ctx.geometry_complete(0)
        ctx.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)  # free space
        # voltage source on the centre segment of the driven element (tag 2)
        ctx.ex_card(0, 2, n_seg // 2 + 1, 0, 1.0, 0, 0, 0, 0, 0)
        # n_freq frequencies evenly over the band (3 -> 144, 146, 148 MHz)
        ctx.fr_card(0, n_freq, _F_LO, (_F_HI - _F_LO) / (n_freq - 1))
        # horizontal-plane cut theta = 90 deg, phi = 0, phi_step, ... (phi = 0 forward)
        ctx.rp_card(0, 1, n_phi, 0, 5, 0, 0, 90.0, 0.0, 0.0, self._phi_step, 0, 0)

        gamma = 0.0
        for k in range(n_freq):
            z = ctx.get_input_parameters(k).get_impedance()[0]
            gamma = max(gamma, abs((z - _Z0) / (z + _Z0)))
        k0 = (n_freq - 1) // 2  # index of f0
        gains = np.asarray(ctx.get_radiation_pattern(k0).get_gain(), float).reshape(-1)
        gains = np.maximum(gains, _G_FLOOR_DBI)
        phi = np.arange(n_phi) * self._phi_step
        g_fwd = float(gains[0])
        g_back = float(gains[np.isclose(phi, 180.0)][0])
        g_rear = float(gains[(phi >= 90.0) & (phi <= 270.0)].max())
        fb = float(np.clip(g_fwd - g_back, -_FB_CLIP_DB, _FB_CLIP_DB))  # 180 deg only
        fr = float(np.clip(g_fwd - g_rear, -_FB_CLIP_DB, _FB_CLIP_DB))  # rear half
        # the constrained pattern ratio: F/B by default, F/R when phi is finely sampled
        ratio = fr if self.front_to_rear else fb
        return g_fwd, gamma, ratio, fb, fr

    @property
    def front_to_rear(self) -> bool:
        """True when the pattern constraint uses the front-to-rear ratio."""
        return self._phi_step < 180.0

    def sample(self, n: int, seed: int | None = None) -> torch.Tensor:
        """Latin-hypercube initial design with uniform discrete choices.

        The base class scales a continuous coordinate and snaps it to the nearest
        allowed value, which gives the four tube diameters (unevenly spaced) and the
        end values of ``n_dir`` unequal shares. Here each director count and each
        diameter gets an equal share, while keeping the Latin-hypercube stratification.
        """
        from scipy.stats import qmc

        u = qmc.LatinHypercube(d=self.dim, seed=seed).random(n)
        X = self.scale(torch.from_numpy(u).to(torch.double))
        n_dir = np.floor(u[:, 0] * _N_DIR_MAX) + 1
        idx = np.minimum((u[:, 1] * len(_DIAMETERS_MM)).astype(int), 3)
        X[:, 0] = torch.from_numpy(n_dir)
        X[:, 1] = torch.tensor(_DIAMETERS_MM, dtype=torch.double)[idx]
        return X

    def simulate(self, x) -> dict:
        """Full NEC2 result for one design (useful for inspection and plotting)."""
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        X = torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(1, -1))
        x = self.enforce_variable_types(X)[0].numpy()  # same snapping as evaluate()
        lengths, positions, diam_mm, boom = self._decode(x)
        g_fwd, gamma, _, fb, fr = self._solve(lengths, positions, diam_mm)
        vswr = (1 + gamma) / (1 - gamma) if gamma < 1 else float("inf")
        out = {
            "gain_dbi": g_fwd,
            "max_abs_gamma": float(gamma),
            "max_vswr": float(vswr),
            "front_to_back_db": fb,  # always the 180 deg ratio
            "boom_lambda": boom,
            "n_elements": len(lengths),
        }
        if self.front_to_rear:
            out["front_to_rear_db"] = fr  # worst lobe over the rear half plane
        return out

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        X = self.enforce_variable_types(X.to(torch.float64))
        x = X.detach().cpu().numpy().astype(np.float64)

        n = x.shape[0]
        obj = np.empty(n)
        cons = np.empty((n, 3))
        for i in range(n):
            lengths, positions, diam_mm, boom = self._decode(x[i])
            g_fwd, gamma, fb, _, _ = self._solve(lengths, positions, diam_mm)
            obj[i] = g_fwd
            cons[i] = (gamma - _GAMMA_MAX, _FB_MIN_DB - fb, boom - _BOOM_MAX)

        fx = torch.tensor(obj, dtype=torch.float64).reshape(-1, 1)  # maximize gain
        gx = torch.tensor(cons, dtype=torch.float64)
        return gx, fx


class YagiUda21_HiFi(YagiUda21):
    """High-fidelity YagiUda21: same 21 variables, bounds, objective and constraints.

    Same specification, more accurate physics: 31 segments per element instead of 15,
    and the VSWR limit checked at 9 frequencies across 144-148 MHz instead of 3. About
    10x the cost of ``YagiUda21``, the fast low-fidelity member of the pair, so the two
    form a natural multi-fidelity benchmark. ``front_to_rear=True`` replaces the 180 deg
    front-to-back ratio with the stricter front-to-rear ratio (forward gain minus the
    strongest lobe anywhere in phi = 90..270 deg, 5 deg steps); that is a harder,
    different specification (about 0.4 % of random designs pass it). Needs
    ``pip install 'bocode[nec]'``.
    """

    tags = {"single_objective", "constrained", "21D", "mixed", "conditional"}

    _n_seg = 31
    _n_freq = 9

    def __init__(self, front_to_rear: bool = False) -> None:
        super().__init__()
        self._phi_step = 5.0 if front_to_rear else 180.0
