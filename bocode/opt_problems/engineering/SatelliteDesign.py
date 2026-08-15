"""Satellite conceptual design — minimize total mass (SMAD / minimdo formulation).

A multidisciplinary conceptual design of a small Earth-observation satellite,
adapted from the `minimdo` satellite application
(https://github.com/norheim/minimdo, `applications/satellite/sat_initial.py`),
which is based on Wertz & Larson, *Space Mission Analysis and Design* (SMAD).

Four continuous design variables — orbit altitude, solar-array area, payload
ground resolution, and propellant mass — drive coupled orbit / power / payload /
communications / structure / propulsion disciplines. Total mass ``m_t`` is a
coupling variable; because the structural mass is linear in ``m_t``
(``m_s = η_S · m_t``) the multidisciplinary system has the closed form
``m_t = (m_T + m_p + m_b + m_A + m_pr) / (1 - η_S)``, so no inner iteration is
needed. The objective is to minimize ``m_t`` subject to mission constraints.

Sources:
J. R. Wertz and W. J. Larson (eds.). Space Mission Analysis and Design (SMAD), 3rd ed., Microcosm/Springer, 1999.
P. Norheim, minimdo (satellite application). https://github.com/norheim/minimdo
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem

# Physical constants (SI).
_MU = 3.986005e14  # Earth gravitational parameter [m^3/s^2]
_R = 6378e3  # Earth radius [m]
_Q = 1367.0  # solar flux [W/m^2]
_K = 1.38064852e-23  # Boltzmann [J/K]
_C = 2.99e8  # speed of light [m/s]
_G = 9.81  # surface gravity [m/s^2]
_YEAR_S = 365.25 * 24 * 3600  # seconds per year

# Atmospheric density / scale-height tables (altitude in m), from minimdo constants.
_H_TABLE = (
    np.array(
        [
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            800,
            850,
            900,
            950,
            1000,
            1250,
            1500,
        ]
    )
    * 1e3
)
_RHO_TABLE = np.array(
    [
        4.79e-07,
        1.81e-09,
        2.53e-10,
        6.24e-11,
        1.95e-11,
        6.98e-12,
        2.72e-12,
        1.13e-12,
        4.89e-13,
        2.21e-13,
        1.04e-13,
        5.15e-14,
        2.72e-14,
        1.55e-14,
        9.63e-15,
        6.47e-15,
        4.66e-15,
        3.54e-15,
        2.79e-15,
        1.11e-15,
        5.21e-16,
    ]
)
_SCALEHEIGHT_TABLE = (
    np.array(
        [
            5.9,
            25.5,
            37.5,
            44.8,
            50.3,
            54.8,
            58.2,
            61.3,
            64.5,
            68.7,
            74.8,
            84.4,
            99.3,
            121,
            151,
            188,
            226,
            263,
            296,
            408,
            516,
        ]
    )
    * 1e3
)


def _loglog_interp(alt_m, table):
    """Log-log interpolate ``table`` against altitude (clamped to the table range)."""
    alt = np.clip(alt_m, _H_TABLE[0], _H_TABLE[-1])
    return np.exp(np.interp(np.log(alt), np.log(_H_TABLE), np.log(table)))


def _db_to_linear(db):
    return 10.0 ** (db / 10.0)


class SatelliteDesign(BenchmarkProblem):
    """Minimize satellite total mass (4 continuous vars, 3 constraints).

    Variables: ``h`` orbit altitude [km], ``A`` solar-array area [m^2], ``X_r``
    payload ground resolution [m], ``m_pr`` propellant mass [kg]. Constraints
    (feasible when <= 0): mission lifetime >= 10 yr, link-budget margin
    ``E_b/N_0 >= 1``, transmit power ``P_T >= 0``.
    """

    available_dimensions = 4
    num_objectives = 1
    num_constraints = 3

    def __init__(self) -> None:
        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=3,
            bounds=[(300.0, 800.0), (0.01, 2.0), (1.0, 20.0), (0.1, 5.0)],
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        h_km, A, X_r, m_pr = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        # parameters
        eta, eta_A, eta_S = 0.55, 0.3, 0.2
        rho_A, rho_b, rho_p, rho_T = 10.0, 0.002, 2.0, 0.2
        P_l, B, N, l_v, f = 12.0, 8.0, 2e3, 500e-9, 2.2e9
        D_r, C_D, I_sp, T_s = 5.3, 2.2, 70.0, 135.0
        G_T = _db_to_linear(16.5)
        L = _db_to_linear(1 + 8.5 + 0.3 + 0.1)

        h_m = h_km * 1e3
        a = h_m + _R  # orbit radius [m]
        T = 2 * np.pi * np.sqrt(a**3 / _MU)  # period [s]
        g = (1.0 / np.pi) * np.arccos(np.clip(_R / a, -1.0, 1.0))  # eclipse fraction
        d = g + 0.5
        r = np.sqrt(h_m**2 + 2 * _R * h_m)  # slant range [m]

        # power
        m_A = rho_A * A
        P_c = d * A * _Q * eta_A
        P_T = P_c - P_l
        E_b_kJ = P_c * T / d / 1e3
        m_b = rho_b * E_b_kJ

        # payload
        D_p = 1.22 * l_v * h_m / X_r
        m_p = rho_p * np.abs(D_p) ** 1.5
        data_bits = 2 * np.pi * _R * B * N / X_r
        b_rate = data_bits / (g * T)  # bits/s

        # communications
        lam = _C / f
        D_T = lam * np.sqrt(G_T / eta) / np.pi
        m_T = rho_T * D_T**1.5
        G_r = eta * (np.pi * D_r / lam) ** 2
        EN = P_T * G_r * G_T / (L * _K * T_s * b_rate) * (lam / (4 * np.pi * r)) ** 2

        # closed-form total mass (m_s = eta_S * m_t)
        m_t = (m_T + m_p + m_b + m_A + m_pr) / (1 - eta_S)

        # propulsion lifetime
        rho_atm = _loglog_interp(h_m, _RHO_TABLE)
        H_sc = _loglog_interp(h_m, _SCALEHEIGHT_TABLE)
        L_n = H_sc * m_t / (2 * np.pi * C_D * A * rho_atm * a**2) * T / _YEAR_S
        L_p = m_pr * I_sp * _G * a / (0.5 * C_D * A * rho_atm * _MU) / _YEAR_S
        L_t = L_n + L_p

        # constraints (feasible <= 0)
        g1 = 10.0 - L_t  # lifetime >= 10 yr
        g2 = 1.0 - EN  # link-budget margin E_b/N_0 >= 1
        g3 = -P_T  # transmit power >= 0
        gx = torch.tensor(np.stack([g1, g2, g3], axis=1), dtype=torch.float64)

        fx = torch.tensor(-m_t, dtype=torch.float64).reshape(-1, 1)  # maximize -mass
        return gx, fx
