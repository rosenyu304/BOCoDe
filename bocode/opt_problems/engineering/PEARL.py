"""PEARL marine-platform conceptual design — minimize platform mass.

PEARL is an autonomous, solar-powered ocean-sensing surface platform (a buoy that
recharges underwater vehicles). Its conceptual design couples six analysis
disciplines — geometry, hydrostatics, mass, propulsion, communications, and power —
through a feedback loop, so each design evaluation runs an inner multidisciplinary
(MDA) solve to consistency before the objective and constraints are formed. Adapted
from the ``minimdo`` PEARL application and Norheim's PhD thesis (§7.4).

Seven free design variables drive the platform: forward speed ``v``, comms
energy-per-bit ratio ``EN``, float draft ``h_f``, and the float/spar/damper-plate
thicknesses ``t_f, t_s`` and diameters ``D_s, D_d``. The float diameter ``D_f``
(from solar-array sizing), the damper-plate thickness ``t_d`` (from the structural
mass balance), the total mass, and the battery mass are *coupling* variables solved
inside the MDA: ``A_solar`` and ``t_d`` close the loop, after which the platform
displaces ``m_tot = rho_w * V_displaced``.

Objective: minimize total platform mass ``m_tot`` [kg].
Constraints (feasible <= 0): geometric proportions ``h_f <= 0.9 t_f``,
``D_s <= 0.9 D_f``, ``D_s <= 0.9 D_d``; minimum propulsion power ``P_move >= 0.1 W``;
minimum comms power ``P_comms >= 50 W``; minimum damper thickness ``t_d >= 0.1 m``.

Reference optimum (thesis Table 7.14): ``m_tot ~ 585.3 kg`` with ``D_f ~ 1.98``,
``D_s ~ 0.19``, ``D_d ~ 0.21``, ``t_f = t_d = 0.1``, ``t_s = 10`` (m).

Sources:
P. T. Norheim. A computational framework for systems engineering models (PhD thesis, MIT AeroAstro), 2022, Ch. 7 (PEARL platform). https://github.com/norheim/minimdo
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import fsolve

from ...base import BenchmarkProblem, DataType

# Physical constants / parameters (from minimdo pearl_initial_formulation.py).
_RHOW = 1023.6  # seawater density [kg/m^3]
_RHO = 700.0  # float/spar material density [kg/m^3]
_RHOH = 2700.0  # damper-plate material density [kg/m^3]
_G = 9.81
_ALPHA = 0.2  # solar-area packing factor
_ETA_SOLAR = 10.0  # solar areal mass [kg/m^2]
_M_PROP = 50.0  # propulsion subsystem mass [kg]
_M_COMMS = 50.0  # comms subsystem mass [kg]
_C_D = 1.0  # drag coefficient
_ETA_M = 0.75  # motor efficiency
# comms
_K = 1.38065e-23  # Boltzmann
_C_LIGHT = 3e8
_F = 2.2e9  # carrier frequency [Hz]
_ETA_PARAB = 0.55
_D_R = 0.3  # ground-station dish diameter [m]
_H_ORBIT = 780e3
_RE = 6378e3
_T_S = 135.0
_R_RATE = 10e6  # data rate [bit/s]
_LA = 10 ** (-0.3 / 10)
_LL = 10 ** (-1.0 / 10)
_LP = 10 ** (-0.1 / 10)
# power / energy budget
_P_HOTEL = 50.0
_E_AUV_WH = 1.9e3  # AUV battery to recharge [Wh]
_GAMMA = 1.0
_T_MISSION, _T_COMMS, _T_MOVE, _T_RECHARGE = 24.0, 1.0, 1.0, 12.0
_ETA_S = 0.27
_PHI_S = 800.0
_THETA = np.deg2rad(55.0)
_I_DEG = 0.9
_D_DEG = 0.005
_L_SOLAR = 10.0
_DOD = 0.7
_N_BATT = 1.0
_ETA_BATT = 0.85
_MU_BATT = 200.0  # [Wh/kg]
_M_BATT0 = 5.0


def _disciplines(A_s, td, v, EN, hf, tf, ts, Ds, Dd):
    """Evaluate all disciplines for a guess of the coupling variables (A_s, t_d)."""
    A_s = abs(A_s)
    Df = np.sqrt(4 * A_s / (np.pi * (1 - _ALPHA)))
    d = _ALPHA * Df

    # hydrostatics: displaced volume -> buoyancy -> total mass
    Vd = np.pi / 4 * (Df**2 * hf + Ds**2 * ts + Dd**2 * td)
    mtot = _RHOW * Vd

    # propulsion: wetted surface -> drag power
    Swd = np.pi * ((Dd / 2) ** 2 - (Ds / 2) ** 2 + (Dd / 2) ** 2 + 2 * (Dd / 2) * td)
    Sws = 2 * np.pi * (Ds / 2) * ts
    Swf = np.pi * ((Df / 2) ** 2 - (Ds / 2) ** 2 + 2 * (Df / 2) * hf)
    Sw = Swd + Sws + Swf
    P_move = _RHOW * _C_D * Sw * v**3 / (2 * _ETA_M)

    # communications: link budget -> comms power
    lam = _C_LIGHT / _F
    Gt = _ETA_PARAB * (np.pi * d / lam) ** 2
    Gr = _ETA_PARAB * (np.pi * _D_R / lam) ** 2
    S = np.sqrt(_H_ORBIT * (_H_ORBIT + 2 * _RE))
    Ls = (lam / (4 * np.pi * S)) ** 2
    Pcomms = EN / (_LA * Ls * _LL * _LP * Gr * Gt) * (_K * _T_S * _R_RATE)

    # power / energy budget (Wh)
    E_required = (
        _P_HOTEL * _T_MISSION + P_move * _T_MOVE + _E_AUV_WH * _GAMMA + Pcomms * _T_COMMS
    )
    P_recharge = E_required / _T_RECHARGE
    A_s_new = P_recharge / (_ETA_S * _PHI_S * np.cos(_THETA) * _I_DEG * (1 - _D_DEG) ** _L_SOLAR)
    cap = E_required / (_DOD * _N_BATT * _ETA_BATT)
    mbatt = cap / _MU_BATT + _M_BATT0

    # mass balance closes the damper-plate thickness
    msolar = _ETA_SOLAR * A_s
    mstruct = mtot - mbatt - msolar - _M_COMMS - _M_PROP
    td_new = (4 / np.pi * mstruct - Df**2 * tf * _RHO - Ds**2 * ts * _RHO) / (Dd**2 * _RHOH)

    return dict(Df=Df, mtot=mtot, P_move=P_move, Pcomms=Pcomms, A_s_new=A_s_new, td_new=td_new)


def _solve_one(v, EN, hf, tf, ts, Ds, Dd):
    """Solve the MDA for one design; return (mtot, Df, td, P_move, Pcomms)."""

    def residual(u):
        out = _disciplines(u[0], u[1], v, EN, hf, tf, ts, Ds, Dd)
        return [u[0] - out["A_s_new"], u[1] - out["td_new"]]

    u = fsolve(residual, [2.0, 0.1], full_output=True)[0]
    out = _disciplines(u[0], u[1], v, EN, hf, tf, ts, Ds, Dd)
    return out["mtot"], out["Df"], u[1], out["P_move"], out["Pcomms"]


class PEARL(BenchmarkProblem):
    """Minimize PEARL platform mass (7 continuous vars, 6 constraints, inner MDA).

    Variables (in order): ``v`` [0, 2] m/s, ``EN`` [0.1, 100], ``h_f`` [0, 5] m,
    ``t_f`` [0.1, 10] m, ``t_s`` [0.1, 10] m, ``D_s`` [0.1, 10] m, ``D_d`` [0.1, 10] m.
    """

    available_dimensions = 7
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 6

    def __init__(self) -> None:
        super().__init__(
            dim=7,
            num_objectives=1,
            num_constraints=6,
            bounds=[
                (0.0, 2.0),
                (0.1, 100.0),
                (0.0, 5.0),
                (0.1, 10.0),
                (0.1, 10.0),
                (0.1, 10.0),
                (0.1, 10.0),
            ],
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)

        mtot = np.empty(x.shape[0])
        gx = np.empty((x.shape[0], 6))
        for i, (v, EN, hf, tf, ts, Ds, Dd) in enumerate(x):
            m, Df, td, P_move, Pcomms = _solve_one(v, EN, hf, tf, ts, Ds, Dd)
            mtot[i] = m
            gx[i] = [
                hf - 0.9 * tf,  # float draft vs thickness
                Ds - 0.9 * Df,  # spar vs float diameter
                Ds - 0.9 * Dd,  # spar vs damper diameter
                0.1 - P_move,  # propulsion power >= 0.1 W
                50.0 - Pcomms,  # comms power >= 50 W
                0.1 - td,  # damper thickness >= 0.1 m
            ]

        gx = torch.tensor(gx, dtype=torch.float64)
        fx = torch.tensor(-mtot, dtype=torch.float64).reshape(-1, 1)  # maximize -mass
        return gx, fx
