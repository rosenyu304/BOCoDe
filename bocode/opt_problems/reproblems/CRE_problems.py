"""
CRE benchmark problems (Tanabe & Ishibuchi, 2020).

8 constrained multi-objective problems with explicit constraints.

Sign convention
---------------
The CRE problems are defined for **minimization**. BoCoDe **maximizes**, so every
``_evaluate_implementation`` here returns ``-fx`` (the negated original
objectives) and ``ref_point`` is the negated reference point.

Reference points (DERIVED, not published for CRE)
-------------------------------------------------
Tanabe & Ishibuchi publish approximated ideal/nadir points -- and hence
hypervolume reference points -- only for the 16 *RE* problems; the paper
explicitly does not analyze the eight CRE problems, and
https://github.com/ryojitanabe/reproblems/tree/master/ideal_nadir_points
contains no CRE files.

Each CRE problem is, however, the constrained original of one RE problem, and its
objectives are *exactly* the leading objectives of that RE problem (the RE variant
appends the total constraint violation as one extra objective). This is verified
numerically: ``CRE_k(x) == RE_j(x)[:m]`` to machine precision on random inputs.

The reference point of each CRE problem is therefore taken as the leading ``m``
components of the corresponding RE problem's published reference point
(``RE_problems._REF_POINT_MIN``, i.e. ``z_ideal + 1.1 * (z_nadir - z_ideal)``).
This is a **derivation from published RE data, not a published CRE reference
point**. Caveat: the RE ideal/nadir points are approximated over the RE problem's
*unconstrained* domain, so relative to the CRE feasible region this reference
point is somewhat pessimistic (it can only make the hypervolume larger by a fixed
offset -- it stays identical across algorithms, which is what matters for a fair
comparison).

Reference:
    Ryoji Tanabe, Hisao Ishibuchi, "An Easy-to-use Real-world Multi-objective
    Problem Suite" Applied Soft Computing. 89: 106078 (2020)
    https://arxiv.org/pdf/2009.12867

Note: Original reproblems use g >= 0 = satisfied.
      BOCoDe uses g <= 0 = satisfied. Constraints are negated accordingly.
"""

import math

import torch

from ...base import BenchmarkProblem
from .RE_problems import _REF_POINT_MIN

#: CRE problem -> (RE problem it is the constrained original of, number of objectives).
_CRE_TO_RE = {
    "CRE21": ("RE31", 2),
    "CRE22": ("RE32", 2),
    "CRE23": ("RE33", 2),
    "CRE24": ("RE35", 2),
    "CRE25": ("RE36", 2),
    "CRE31": ("RE41", 3),
    "CRE32": ("RE42", 3),
    "CRE51": ("RE61", 5),
}


class _CREProblem(BenchmarkProblem):
    """Base for the CRE suite: attaches the reference point derived from RE.

    Stored in BoCoDe's maximization frame (negated from the minimization-frame
    value), matching the negated objectives returned by
    ``_evaluate_implementation``.
    """

    def __init__(self, **kwargs) -> None:
        re_name, m = _CRE_TO_RE[type(self).__name__]
        super().__init__(ref_point=[-r for r in _REF_POINT_MIN[re_name][:m]], **kwargs)


class CRE21(_CREProblem):
    available_dimensions = 3
    num_objectives = 2
    num_constraints = 3

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=2,
            num_constraints=3,
            bounds=[
                (0.00001, 100.0),
                (0.00001, 100.0),
                (1.0, 3.0),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, self.num_constraints), device=X.device)

        fx[:, 0] = x1 * torch.sqrt(16.0 + (x3 * x3)) + x2 * torch.sqrt(1.0 + x3 * x3)
        fx[:, 1] = (20.0 * torch.sqrt(16.0 + (x3 * x3))) / (x1 * x3)

        # Negate for BOCoDe convention (g <= 0 = satisfied)
        gx[:, 0] = -(0.1 - fx[:, 0])
        gx[:, 1] = -(100000.0 - fx[:, 1])
        gx[:, 2] = -(100000 - ((80.0 * torch.sqrt(1.0 + x3 * x3)) / (x3 * x2)))

        return gx, -fx  # CRE is minimization; BoCoDe maximizes


class CRE22(_CREProblem):
    available_dimensions = 4
    num_objectives = 2
    num_constraints = 4

    def __init__(self):
        super().__init__(
            dim=4,
            num_objectives=2,
            num_constraints=4,
            bounds=[
                (0.125, 5.0),
                (0.1, 10.0),
                (0.1, 10.0),
                (0.125, 5.0),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        P = 6000
        L = 14
        E = 30 * 1e6
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, self.num_constraints), device=X.device)

        fx[:, 0] = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        fx[:, 1] = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + torch.pow((x1 + x3) / 2.0, 2)
        R = torch.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + torch.pow((x1 + x3) / 2.0, 2)
        J = 2 * math.sqrt(2) * x1 * x2 * tmpVar

        tauDashDash = (M * R) / J
        tauDash = P / (math.sqrt(2) * x1 * x2)
        tmpVar = (
            tauDash * tauDash
            + ((2 * tauDash * tauDashDash * x2) / (2 * R))
            + (tauDashDash * tauDashDash)
        )
        tau = torch.sqrt(tmpVar)
        sigma = (6 * P * L) / (x4 * x3 * x3)
        tmpVar = (
            4.013
            * E
            * torch.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0)
            / (L * L)
        )
        tmpVar2 = (x3 / (2 * L)) * math.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)

        # Negate for BOCoDe convention (g <= 0 = satisfied)
        gx[:, 0] = -(tauMax - tau)
        gx[:, 1] = -(sigmaMax - sigma)
        gx[:, 2] = -(x4 - x1)
        gx[:, 3] = -(PC - P)

        return gx, -fx  # CRE is minimization; BoCoDe maximizes


class CRE23(_CREProblem):
    available_dimensions = 4
    num_objectives = 2
    num_constraints = 4

    def __init__(self):
        super().__init__(
            dim=4,
            num_objectives=2,
            num_constraints=4,
            bounds=[
                (55.0, 80.0),
                (75.0, 110.0),
                (1000.0, 3000.0),
                (11.0, 20.0),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, self.num_constraints), device=X.device)

        fx[:, 0] = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        fx[:, 1] = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (
            x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)
        )

        # Negate for BOCoDe convention (g <= 0 = satisfied)
        gx[:, 0] = -((x2 - x1) - 20.0)
        gx[:, 1] = -(0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1))))
        gx[:, 2] = -(
            1.0
            - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1))
            / torch.pow((x2 * x2 - x1 * x1), 2)
        )
        gx[:, 3] = -(
            (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))
            / (x2 * x2 - x1 * x1)
            - 900.0
        )

        return gx, -fx  # CRE is minimization; BoCoDe maximizes


class CRE24(_CREProblem):
    available_dimensions = 7
    num_objectives = 2
    num_constraints = 11

    def __init__(self):
        super().__init__(
            dim=7,
            num_objectives=2,
            num_constraints=11,
            bounds=[
                (2.6, 3.6),
                (0.7, 0.8),
                (17.0, 28.0),
                (7.3, 8.3),
                (7.3, 8.3),
                (2.9, 3.9),
                (5.0, 5.5),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = torch.round(X[:, 2])
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, self.num_constraints), device=X.device)

        fx[:, 0] = (
            0.7854
            * x1
            * (x2 * x2)
            * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934)
            - 1.508 * x1 * (x6 * x6 + x7 * x7)
            + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7)
            + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
        )

        tmpVar = torch.pow((745.0 * x4) / (x2 * x3), 2.0) + 1.69 * 1e7
        fx[:, 1] = torch.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

        # Negate for BOCoDe convention (g <= 0 = satisfied)
        gx[:, 0] = -(-(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0)
        gx[:, 1] = -(-(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5)
        gx[:, 2] = -(-(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93)
        gx[:, 3] = -(-(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93)
        gx[:, 4] = -(-(x2 * x3) + 40.0)
        gx[:, 5] = -(-(x1 / x2) + 12.0)
        gx[:, 6] = -(-5.0 + (x1 / x2))
        gx[:, 7] = -(-1.9 + x4 - 1.5 * x6)
        gx[:, 8] = -(-1.9 + x5 - 1.1 * x7)
        gx[:, 9] = -(-fx[:, 1] + 1300.0)
        tmpVar = torch.pow((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
        gx[:, 10] = -(-torch.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0)

        return gx, -fx  # CRE is minimization; BoCoDe maximizes


class CRE25(_CREProblem):
    available_dimensions = 4
    num_objectives = 2
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=4,
            num_objectives=2,
            num_constraints=1,
            bounds=[(12.0, 60.0)] * 4,
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = torch.round(X[:, 0])
        x2 = torch.round(X[:, 1])
        x3 = torch.round(X[:, 2])
        x4 = torch.round(X[:, 3])

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, self.num_constraints), device=X.device)

        fx[:, 0] = torch.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        fx[:, 1] = torch.stack([x1, x2, x3, x4], dim=1).max(dim=1).values

        # Negate for BOCoDe convention (g <= 0 = satisfied)
        gx[:, 0] = -(0.5 - (fx[:, 0] / 6.931))

        return gx, -fx  # CRE is minimization; BoCoDe maximizes


class CRE31(_CREProblem):
    available_dimensions = 7
    num_objectives = 3
    num_constraints = 10

    def __init__(self):
        super().__init__(
            dim=7,
            num_objectives=3,
            num_constraints=10,
            bounds=[
                (0.5, 1.5),
                (0.45, 1.35),
                (0.5, 1.5),
                (0.5, 1.5),
                (0.875, 2.625),
                (0.4, 1.2),
                (0.4, 1.2),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, self.num_constraints), device=X.device)

        fx[:, 0] = (
            1.98
            + 4.9 * x1
            + 6.67 * x2
            + 6.98 * x3
            + 4.01 * x4
            + 1.78 * x5
            + 0.00001 * x6
            + 2.73 * x7
        )
        fx[:, 1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3

        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        fx[:, 2] = 0.5 * (Vmbp + Vfd)

        # Negate for BOCoDe convention (g <= 0 = satisfied)
        gx[:, 0] = -(1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3))
        gx[:, 1] = -(
            0.32
            - (
                0.261
                - 0.0159 * x1 * x2
                - 0.06486 * x1
                - 0.019 * x2 * x7
                + 0.0144 * x3 * x5
                + 0.0154464 * x6
            )
        )
        gx[:, 2] = -(
            0.32
            - (
                0.214
                + 0.00817 * x5
                - 0.045195 * x1
                - 0.0135168 * x1
                + 0.03099 * x2 * x6
                - 0.018 * x2 * x7
                + 0.007176 * x3
                + 0.023232 * x3
                - 0.00364 * x5 * x6
                - 0.018 * x2 * x2
            )
        )
        gx[:, 3] = -(
            0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        )
        gx[:, 4] = -(
            32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        )
        gx[:, 5] = -(
            32
            - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        )
        gx[:, 6] = -(32 - (46.36 - 9.9 * x2 - 4.4505 * x1))
        gx[:, 7] = -(4 - fx[:, 1])
        gx[:, 8] = -(9.9 - Vmbp)
        gx[:, 9] = -(15.7 - Vfd)

        return gx, -fx  # CRE is minimization; BoCoDe maximizes


class CRE32(_CREProblem):
    available_dimensions = 6
    num_objectives = 3
    num_constraints = 9

    def __init__(self):
        super().__init__(
            dim=6,
            num_objectives=3,
            num_constraints=9,
            bounds=[
                (150.0, 274.32),
                (20.0, 32.31),
                (13.0, 25.0),
                (10.0, 11.71),
                (14.0, 18.0),
                (0.63, 0.75),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x_L = X[:, 0]
        x_B = X[:, 1]
        x_D = X[:, 2]
        x_T = X[:, 3]
        x_Vk = X[:, 4]
        x_CB = X[:, 5]

        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g_const = 9.8065
        Fn = V / torch.pow(g_const * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (torch.pow(displacement, 2.0 / 3.0) * torch.pow(x_Vk, 3.0)) / (
            a + (b * Fn)
        )
        outfit_weight = (
            1.0
            * torch.pow(x_L, 0.8)
            * torch.pow(x_B, 0.6)
            * torch.pow(x_D, 0.3)
            * torch.pow(x_CB, 0.1)
        )
        steel_weight = (
            0.034
            * torch.pow(x_L, 1.7)
            * torch.pow(x_B, 0.7)
            * torch.pow(x_D, 0.4)
            * torch.pow(x_CB, 0.5)
        )
        machinery_weight = 0.17 * torch.pow(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * (
            (2000.0 * torch.pow(steel_weight, 0.85))
            + (3500.0 * outfit_weight)
            + (2400.0 * torch.pow(power, 0.8))
        )
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * torch.pow(torch.clamp(DWT, min=1e-10), 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * torch.pow(torch.clamp(DWT, min=1e-10), 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * torch.pow(torch.clamp(DWT, min=1e-10), 0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, self.num_constraints), device=X.device)

        fx[:, 0] = annual_costs / annual_cargo
        fx[:, 1] = light_ship_weight
        fx[:, 2] = -annual_cargo

        # Negate for BOCoDe convention (g <= 0 = satisfied)
        gx[:, 0] = -((x_L / x_B) - 6.0)
        gx[:, 1] = -(-(x_L / x_D) + 15.0)
        gx[:, 2] = -(-(x_L / x_T) + 19.0)
        gx[:, 3] = -(0.45 * torch.pow(torch.clamp(DWT, min=1e-10), 0.31) - x_T)
        gx[:, 4] = -(0.7 * x_D + 0.7 - x_T)
        gx[:, 5] = -(500000.0 - DWT)
        gx[:, 6] = -(DWT - 3000.0)
        gx[:, 7] = -(0.32 - Fn)

        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        gx[:, 8] = -((KB + BMT - KG) - (0.07 * x_B))

        return gx, -fx  # CRE is minimization; BoCoDe maximizes


class CRE51(_CREProblem):
    available_dimensions = 3
    num_objectives = 5
    num_constraints = 7

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=5,
            num_constraints=7,
            bounds=[
                (0.01, 0.45),
                (0.01, 0.10),
                (0.01, 0.10),
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        n = X.size(0)

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        fx = torch.zeros((n, self.num_objectives), device=X.device)
        gx = torch.zeros((n, self.num_constraints), device=X.device)

        fx[:, 0] = 106780.37 * (x2 + x3) + 61704.67
        fx[:, 1] = 3000 * x1
        fx[:, 2] = 305700 * 2289 * x2 / (0.06 * 2289) ** 0.65
        fx[:, 3] = 250 * 2289 * torch.exp(-39.75 * x2 + 9.9 * x3 + 2.74)
        fx[:, 4] = 25 * (1.39 / (x1 * x2) + 4940 * x3 - 80)

        # Negate for BOCoDe convention (g <= 0 = satisfied)
        gx[:, 0] = -(1 - (0.00139 / (x1 * x2) + 4.94 * x3 - 0.08))
        gx[:, 1] = -(1 - (0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986))
        gx[:, 2] = -(50000 - (12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02))
        gx[:, 3] = -(16000 - (2.098 / (x1 * x2) + 8046.33 * x3 - 696.71))
        gx[:, 4] = -(10000 - (2.138 / (x1 * x2) + 7883.39 * x3 - 705.04))
        gx[:, 5] = -(2000 - (0.417 * x1 * x2 + 1721.26 * x3 - 136.54))
        gx[:, 6] = -(550 - (0.164 / (x1 * x2) + 631.13 * x3 - 54.48))

        return gx, -fx  # CRE is minimization; BoCoDe maximizes
