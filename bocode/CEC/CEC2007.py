"""
The CEC2007 benchmark suite.

Sources:
(1) V. L. Huang, A. K. Qin, K. Deb, E. Zitzler, P. N. Suganthan, J. J. Liang, M. Preuss and S. Huband (2007). Problem Definitions for Performance Assessment of Multi-objective Optimization Algorithms. Special Session on Constrained Real-Parameter Optimization, Technical Report, Nanyang Technological University, Singapore, 2007. https://github.com/P-N-Suganthan/CEC2007/blob/master/CEC-07-TR-13-Feb.pdf
(2) Simon Wessing. Towards Optimal Parameterizations of the S-Metric Selection Evolutionary Multi-Objective Algorithm. Diploma thesis, Algorithm Engineering Report TR09-2-006, Technische Universitaet Dortmund, 2009. https://ls11-www.cs.uni-dortmund.de/_media/techreports/tr09-06.pdf
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from ..base import BenchmarkProblem, DataType
from ..exceptions import FunctionDefinitionAssertionError


# =============================================================================
# OKA2
# =============================================================================
class CEC2007_OKA2(BenchmarkProblem):
    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 2
    num_constraints = 0

    def __init__(self):
        dim = 3
        super().__init__(
            dim,
            num_objectives=2,
            num_constraints=0,
            bounds=[(-math.pi, math.pi), (-5.0, 5.0), (-5.0, 5.0)],
        )

    def _evaluate_implementation(self, X: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        # X: (batch, 3)
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        f1 = x1
        f2 = (
            1.0
            - (x1 + math.pi) ** 2 / (4 * math.pi**2)
            + torch.abs(x2 - 5.0 * torch.cos(x1)) ** (1.0 / 3.0)
            + torch.abs(x3 - 5.0 * torch.sin(x1)) ** (1.0 / 3.0)
        )
        fx = torch.stack((f1, f2), dim=1)
        return None, -fx


# =============================================================================
# SYMPART
# =============================================================================
class CEC2007_SYMPART(BenchmarkProblem):
    # requires at least 2 variables (even-length rotations), unlimited upper
    available_dimensions = (2, None)
    input_type = DataType.CONTINUOUS
    num_objectives = 2
    num_constraints = 0

    def __init__(self, dim: int):
        if dim < 2 or dim % 2 != 0:
            raise FunctionDefinitionAssertionError(
                "SYMPART requires an even dimension ≥ 2"
            )
        super().__init__(
            dim,
            num_objectives=2,
            num_constraints=0,
            bounds=[(-100.0, 100.0)] * dim,
        )

    def _evaluate_implementation(self, X: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        # Constants
        a, b, c = 1.0, 10.0, 8.0
        c1 = a + c / 2.0  # 5
        c2 = c + 2 * a  # 10
        b1 = b / 2.0  # 5
        ω = math.pi / 4.0
        si, co = math.sin(ω), math.cos(ω)

        # rotate pairs
        x = X.clone()
        for i in range(0, self.dim, 2):
            xi, xj = x[:, i], x[:, i + 1]
            x[:, i] = co * xi - si * xj
            x[:, i + 1] = si * xi + co * xj

        # tile for the first two dims
        x1, x2 = x[:, 0], x[:, 1]
        t1 = torch.where(
            torch.abs(x1) < c1,
            torch.zeros_like(x1),
            torch.ceil((torch.abs(x1) - c1) / c2),
        )
        t2 = torch.where(
            torch.abs(x2) < b1,
            torch.zeros_like(x2),
            torch.ceil((torch.abs(x2) - b1) / b),
        )
        t1 = t1 * torch.sign(x1)
        t2 = t2 * torch.sign(x2)
        t1 = torch.clamp(t1, -1, 1)
        t2 = torch.clamp(t2, -1, 1)

        # objectives
        f0 = torch.zeros(X.shape[0], device=X.device)
        f1 = torch.zeros_like(f0)
        for k in range(self.dim):
            xk = x[:, k]
            if k % 2 == 0:
                f0 = f0 + (xk + a - t1 * c2) ** 2
                f1 = f1 + (xk - a - t1 * c2) ** 2
            else:
                f0 = f0 + (xk - t2 * b) ** 2
                f1 = f1 + (xk - t2 * b) ** 2

        f0 = f0 / self.dim
        f1 = f1 / self.dim
        fx = torch.stack((f0, f1), dim=1)
        return None, -fx


# =============================================================================
# S_ZDT1
# =============================================================================
class CEC2007_S_ZDT1(BenchmarkProblem):
    available_dimensions = 100
    input_type = DataType.CONTINUOUS
    num_objectives = 2
    num_constraints = 0

    # shift & scaling parameters (100-dim)
    _o = torch.tensor(
        [
            0.950,
            0.231,
            0.607,
            0.486,
            0.891,
            0.762,
            0.457,
            0.019,
            0.821,
            0.445,
            0.615,
            0.792,
            0.922,
            0.738,
            0.176,
            0.406,
            0.936,
            0.917,
            0.410,
            0.894,
            0.058,
            0.353,
            0.813,
            0.010,
            0.139,
            0.203,
            0.199,
            0.604,
            0.272,
            0.199,
            0.015,
            0.747,
            0.445,
            0.932,
            0.466,
            0.419,
            0.846,
            0.525,
            0.203,
            0.672,
            0.838,
            0.020,
            0.681,
            0.380,
            0.832,
            0.503,
            0.710,
            0.429,
            0.305,
            0.190,
            0.193,
            0.682,
            0.303,
            0.542,
            0.151,
            0.698,
            0.378,
            0.860,
            0.854,
            0.594,
            0.497,
            0.900,
            0.822,
            0.645,
            0.818,
            0.660,
            0.342,
            0.290,
            0.341,
            0.534,
            0.727,
            0.309,
            0.839,
            0.568,
            0.370,
            0.703,
            0.547,
            0.445,
            0.695,
            0.621,
            0.795,
            0.957,
            0.523,
            0.880,
            0.173,
            0.980,
            0.271,
            0.252,
            0.876,
            0.737,
            0.137,
            0.012,
            0.894,
            0.199,
            0.299,
            0.661,
            0.284,
            0.469,
            0.065,
            0.988,
        ],
        dtype=torch.float32,
    )  # fill with the 100 values from C code
    _d_l = torch.tensor(
        [
            0.155,
            0.119,
            0.185,
            0.064,
            0.070,
            0.203,
            0.166,
            0.151,
            0.219,
            0.083,
            0.161,
            0.057,
            0.169,
            0.072,
            0.135,
            0.114,
            0.241,
            0.129,
            0.230,
            0.181,
            0.195,
            0.058,
            0.152,
            0.107,
            0.119,
            0.219,
            0.203,
            0.212,
            0.126,
            0.238,
            0.164,
            0.236,
            0.086,
            0.087,
            0.188,
            0.244,
            0.146,
            0.124,
            0.218,
            0.196,
            0.120,
            0.140,
            0.120,
            0.207,
            0.205,
            0.114,
            0.107,
            0.228,
            0.097,
            0.208,
            0.105,
            0.120,
            0.135,
            0.196,
            0.183,
            0.111,
            0.081,
            0.236,
            0.183,
            0.171,
            0.063,
            0.160,
            0.246,
            0.091,
            0.064,
            0.188,
            0.222,
            0.082,
            0.174,
            0.114,
            0.155,
            0.209,
            0.179,
            0.130,
            0.184,
            0.200,
            0.108,
            0.143,
            0.116,
            0.155,
            0.078,
            0.189,
            0.128,
            0.171,
            0.168,
            0.226,
            0.061,
            0.250,
            0.201,
            0.226,
            0.065,
            0.194,
            0.114,
            0.159,
            0.135,
            0.239,
            0.167,
            0.204,
            0.095,
            0.091,
        ],
        dtype=torch.float32,
    )  # fill with the 100 values
    _lam = torch.tensor(
        [
            3.236,
            4.201,
            2.701,
            7.775,
            7.148,
            2.465,
            3.020,
            3.322,
            2.285,
            6.004,
            3.106,
            8.801,
            2.956,
            6.918,
            3.708,
            4.403,
            2.077,
            3.884,
            2.171,
            2.760,
            2.567,
            8.636,
            3.299,
            4.666,
            4.208,
            2.285,
            2.469,
            2.362,
            3.963,
            2.099,
            3.052,
            2.120,
            5.809,
            5.753,
            2.654,
            2.053,
            3.427,
            4.046,
            2.297,
            2.548,
            4.151,
            3.577,
            4.165,
            2.417,
            2.445,
            4.405,
            4.668,
            2.192,
            5.147,
            2.398,
            4.743,
            4.157,
            3.710,
            2.553,
            2.733,
            4.519,
            6.159,
            2.121,
            2.738,
            2.926,
            7.918,
            3.130,
            2.034,
            5.495,
            7.846,
            2.666,
            2.253,
            6.068,
            2.881,
            4.388,
            3.220,
            2.389,
            2.791,
            3.832,
            2.711,
            2.496,
            4.624,
            3.505,
            4.297,
            3.223,
            6.428,
            2.644,
            3.920,
            2.917,
            2.978,
            2.215,
            8.138,
            2.000,
            2.490,
            2.215,
            7.711,
            2.574,
            4.395,
            3.135,
            3.705,
            2.091,
            3.003,
            2.456,
            5.241,
            5.499,
        ],
        dtype=torch.float32,
    )  # fill with the 100 values

    def __init__(self):
        dim = 100
        super().__init__(
            dim,
            num_objectives=2,
            num_constraints=0,
            bounds=[(0.0, 1.0)] * dim,
        )

    def _evaluate_implementation(self, X: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        # apply shift
        z = X - self._o.to(X.device)

        # f0
        z0 = z[:, 0]
        mask = z0 >= 0
        zz0 = torch.where(mask, z0, -self._lam[0] * z0)
        f0 = torch.where(
            mask,
            zz0 + 1.0,
            2.0 / (1.0 + torch.exp(z0 / self._d_l[0])) * (zz0 + 1.0),
        )

        # rest
        z_rest = z[:, 1:]
        lam_rest = self._lam[1:].to(X.device)
        d_l_rest = self._d_l[1:].to(X.device)

        zz_rest = torch.where(z_rest < 0, -lam_rest * z_rest, z_rest)
        p_rest = torch.where(z_rest < 0, -z_rest / d_l_rest, torch.zeros_like(z_rest))

        sum_z = zz_rest.sum(dim=1)
        psum = torch.sqrt((p_rest**2).sum(dim=1))

        g = 1.0 + 9.0 * sum_z / (self.dim - 1)
        f1 = (g * (1.0 - torch.sqrt(zz0 / g)) + 1.0) * (2.0 / (1.0 + torch.exp(-psum)))

        fx = torch.stack((f0, f1), dim=1)
        return None, -fx


# =============================================================================
# S_ZDT2
# =============================================================================
class CEC2007_S_ZDT2(CEC2007_S_ZDT1):
    def __init__(self):
        super().__init__()
        # same shift+bounds as S_ZDT1

    def _evaluate_implementation(self, X: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        # reuse z, zz0, f0 from parent
        z = X - self._o.to(X.device)
        z0 = z[:, 0]
        mask = z0 >= 0
        zz0 = torch.where(mask, z0, -self._lam[0] * z0)
        f0 = torch.where(
            mask,
            zz0 + 1.0,
            2.0 / (1.0 + torch.exp(z0 / self._d_l[0])) * (zz0 + 1.0),
        )

        z_rest = z[:, 1:]
        lam_rest = self._lam[1:].to(X.device)
        d_l_rest = self._d_l[1:].to(X.device)

        zz_rest = torch.where(z_rest < 0, -lam_rest * z_rest, z_rest)
        p_rest = torch.where(z_rest < 0, -z_rest / d_l_rest, torch.zeros_like(z_rest))

        sum_z = zz_rest.sum(dim=1)
        psum = torch.sqrt((p_rest**2).sum(dim=1))

        g = 1.0 + 9.0 * sum_z / (self.dim - 1)
        f1 = (g * (1.0 - (zz0 / g) ** 2) + 1.0) * (2.0 / (1.0 + torch.exp(-psum)))

        fx = torch.stack((f0, f1), dim=1)
        return None, -fx


# =============================================================================
# S_ZDT4
# =============================================================================
class CEC2007_S_ZDT4(BenchmarkProblem):
    available_dimensions = 100
    input_type = DataType.CONTINUOUS
    num_objectives = 2
    num_constraints = 0

    _o = torch.tensor(
        [
            0.957,
            0.436,
            2.092,
            5.523,
            5.686,
            3.616,
            1.646,
            9.461,
            0.881,
            7.606,
            4.401,
            4.251,
            5.182,
            6.320,
            9.136,
            9.871,
            7.308,
            6.021,
            1.941,
            0.640,
            0.581,
            4.970,
            4.677,
            4.436,
            3.997,
            1.971,
            0.071,
            8.880,
            9.464,
            4.152,
            1.318,
            4.620,
            9.296,
            2.804,
            9.034,
            1.787,
            5.197,
            7.792,
            5.364,
            7.301,
            0.953,
            6.922,
            5.955,
            5.000,
            1.437,
            1.800,
            2.796,
            2.448,
            0.499,
            2.813,
            3.784,
            5.816,
            7.544,
            9.607,
            0.634,
            7.079,
            6.864,
            9.367,
            2.498,
            3.362,
            5.484,
            8.693,
            2.720,
            0.246,
            1.878,
            7.354,
            4.399,
            8.886,
            1.394,
            4.045,
            7.694,
            1.343,
            4.430,
            4.077,
            1.512,
            5.488,
            7.547,
            3.081,
            7.321,
            7.537,
            3.430,
            1.710,
            9.287,
            3.121,
            5.341,
            1.471,
            5.165,
            3.627,
            7.946,
            1.710,
            9.013,
            7.844,
            9.240,
            6.567,
            4.996,
            3.462,
            1.847,
            2.767,
            9.231,
            8.492,
        ],
        dtype=torch.float32,
    )
    _d_l = torch.tensor(
        [
            0.099,
            1.905,
            2.486,
            1.323,
            0.823,
            1.519,
            1.737,
            1.969,
            2.072,
            1.949,
            1.812,
            1.895,
            0.571,
            2.378,
            1.079,
            0.673,
            1.300,
            1.929,
            2.052,
            1.499,
            2.282,
            1.721,
            0.675,
            1.275,
            1.282,
            2.080,
            1.178,
            1.539,
            2.319,
            0.672,
            1.243,
            0.883,
            0.939,
            2.239,
            1.249,
            1.833,
            1.154,
            1.773,
            1.743,
            2.152,
            2.445,
            1.783,
            0.753,
            1.610,
            1.248,
            0.749,
            0.703,
            1.544,
            2.203,
            2.355,
            1.373,
            1.570,
            1.330,
            0.834,
            1.183,
            0.731,
            1.142,
            1.991,
            2.101,
            1.163,
            1.817,
            0.849,
            1.631,
            0.934,
            1.672,
            1.313,
            1.488,
            0.826,
            1.907,
            2.250,
            0.676,
            0.593,
            1.953,
            0.699,
            1.340,
            1.880,
            0.690,
            1.655,
            1.804,
            2.296,
            1.826,
            0.856,
            1.924,
            1.652,
            1.501,
            0.903,
            1.852,
            1.661,
            2.351,
            2.107,
            1.819,
            0.574,
            0.803,
            1.662,
            2.390,
            2.402,
            1.007,
            0.654,
            1.845,
            2.116,
        ],
        dtype=torch.float32,
    )
    _lam = torch.tensor(
        [
            5.055,
            2.625,
            2.011,
            3.779,
            6.077,
            3.291,
            2.878,
            2.540,
            2.413,
            2.566,
            2.760,
            2.639,
            8.752,
            2.103,
            4.633,
            7.434,
            3.847,
            2.592,
            2.437,
            3.336,
            2.191,
            2.905,
            7.409,
            3.922,
            3.901,
            2.404,
            4.245,
            3.249,
            2.156,
            7.441,
            4.024,
            5.665,
            5.327,
            2.233,
            4.003,
            2.727,
            4.334,
            2.820,
            2.869,
            2.323,
            2.045,
            2.804,
            6.644,
            3.105,
            4.007,
            6.676,
            7.116,
            3.238,
            2.269,
            2.123,
            3.642,
            3.185,
            3.759,
            5.997,
            4.228,
            6.837,
            4.378,
            2.512,
            2.380,
            4.299,
            2.752,
            5.893,
            3.066,
            5.353,
            2.990,
            3.808,
            3.360,
            6.055,
            2.622,
            2.222,
            7.394,
            8.426,
            2.560,
            7.155,
            3.732,
            2.660,
            7.246,
            3.022,
            2.772,
            2.178,
            2.738,
            5.842,
            2.599,
            3.026,
            3.332,
            5.538,
            2.700,
            3.010,
            2.126,
            2.374,
            2.748,
            8.707,
            6.230,
            3.008,
            2.092,
            2.081,
            4.963,
            7.649,
            2.710,
            2.363,
        ],
        dtype=torch.float32,
    )

    def __init__(self):
        dim = 100
        super().__init__(
            dim,
            num_objectives=2,
            num_constraints=0,
            bounds=[(0.0, 1.0)] + [(-5.0, 5.0)] * (dim - 1),
        )

    def _evaluate_implementation(self, X: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        z = X - self._o.to(X.device)

        # first var
        z0 = z[:, 0]
        mask = z0 >= 0
        zz0 = torch.where(mask, z0, -self._lam[0] * z0)
        p0 = torch.where(mask, torch.zeros_like(z0), -z0 / self._d_l[0])
        f0 = torch.where(
            mask,
            zz0 + 1.0,
            2.0 / (1.0 + torch.exp(-p0)) * (zz0 + 1.0),
        )

        # rest
        z_rest = z[:, 1:]
        lam_rest = self._lam[1:].to(X.device)
        d_l_rest = self._d_l[1:].to(X.device)

        # piecewise
        under = z_rest < -5.0
        zz_rest = torch.where(under, -5.0 - lam_rest * (5.0 + z_rest), z_rest)
        p_rest = torch.where(
            under, (-5.0 - z_rest) / d_l_rest, torch.zeros_like(z_rest)
        )

        sum_term = (zz_rest**2 - 10.0 * torch.cos(4.0 * math.pi * zz_rest)).sum(dim=1)
        psum = torch.sqrt(p0**2 + (p_rest**2).sum(dim=1))

        g = 1.0 + 10.0 * (self.dim - 1) + sum_term
        f1 = (g * (1.0 - torch.sqrt(zz0 / g)) + 1.0) * (2.0 / (1.0 + torch.exp(-psum)))

        fx = torch.stack((f0, f1), dim=1)
        return None, -fx


# =============================================================================
# S_ZDT6
# =============================================================================
class CEC2007_S_ZDT6(CEC2007_S_ZDT1):
    _o = torch.tensor(
        [
            0.950,
            0.231,
            0.607,
            0.486,
            0.891,
            0.762,
            0.457,
            0.019,
            0.821,
            0.445,
            0.615,
            0.792,
            0.922,
            0.738,
            0.176,
            0.406,
            0.936,
            0.917,
            0.410,
            0.894,
            0.058,
            0.353,
            0.813,
            0.010,
            0.139,
            0.203,
            0.199,
            0.604,
            0.272,
            0.199,
            0.015,
            0.747,
            0.445,
            0.932,
            0.466,
            0.419,
            0.846,
            0.525,
            0.203,
            0.672,
            0.838,
            0.020,
            0.681,
            0.380,
            0.832,
            0.503,
            0.710,
            0.429,
            0.305,
            0.190,
            0.193,
            0.682,
            0.303,
            0.542,
            0.151,
            0.698,
            0.378,
            0.860,
            0.854,
            0.594,
            0.497,
            0.900,
            0.822,
            0.645,
            0.818,
            0.660,
            0.342,
            0.290,
            0.341,
            0.534,
            0.727,
            0.309,
            0.839,
            0.568,
            0.370,
            0.703,
            0.547,
            0.445,
            0.695,
            0.621,
            0.795,
            0.957,
            0.523,
            0.880,
            0.173,
            0.980,
            0.271,
            0.252,
            0.876,
            0.737,
            0.137,
            0.012,
            0.894,
            0.199,
            0.299,
            0.661,
            0.284,
            0.469,
            0.065,
            0.988,
        ],
        dtype=torch.float32,
    )
    _d_l = CEC2007_S_ZDT1._d_l
    _lam = CEC2007_S_ZDT1._lam

    def __init__(self):
        super().__init__()
        # same bounds as S_ZDT1

    def _evaluate_implementation(self, X: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        z = X - self._o.to(X.device)
        # first var special
        z0 = z[:, 0]
        mask = z0 >= 0
        zz0 = torch.where(mask, z0, -self._lam[0] * z0)
        sin6 = torch.sin(6.0 * math.pi * zz0) ** 6
        base = 1.0 - torch.exp(-4.0 * zz0) * sin6 + 1.0
        f0 = torch.where(
            mask,
            base,
            2.0 / (1.0 + torch.exp(z0 / self._d_l[0])) * base,
        )

        # rest
        z_rest = z[:, 1:]
        lam_rest = self._lam[1:].to(X.device)
        d_l_rest = self._d_l[1:].to(X.device)
        zz_rest = torch.where(z_rest < 0, -lam_rest * z_rest, z_rest)
        p_rest = torch.where(z_rest < 0, -z_rest / d_l_rest, torch.zeros_like(z_rest))

        sum_z = zz_rest.sum(dim=1)
        psum = torch.sqrt((p_rest**2).sum(dim=1))

        g = 1.0 + 9.0 * (sum_z / (self.dim - 1)) ** 0.25
        f1 = (g * (1.0 - (base / g) ** 2) + 1.0) * (2.0 / (1.0 + torch.exp(-psum)))

        fx = torch.stack((f0, f1), dim=1)
        return None, -fx


# =============================================================================
# S_DTLZ2 & S_DTLZ3 (dynamic objectives & dims)
# =============================================================================
class CEC2007_S_DTLZ2(BenchmarkProblem):
    available_dimensions = (2, None)
    input_type = DataType.CONTINUOUS
    num_objectives = (2, None)
    num_constraints = 0

    _o = torch.tensor(
        [
            0.366,
            0.303,
            0.852,
            0.759,
            0.950,
            0.558,
            0.014,
            0.596,
            0.816,
            0.977,
            0.222,
            0.704,
            0.522,
            0.933,
            0.713,
            0.228,
            0.450,
            0.172,
            0.969,
            0.356,
            0.049,
            0.755,
            0.895,
            0.286,
            0.251,
            0.933,
            0.131,
            0.941,
            0.702,
            0.848,
        ],
        dtype=torch.float32,
    )  # length 30
    _d_l = torch.tensor(
        [
            0.155,
            0.119,
            0.185,
            0.064,
            0.07,
            0.203,
            0.166,
            0.151,
            0.219,
            0.083,
            0.161,
            0.057,
            0.169,
            0.072,
            0.135,
            0.114,
            0.241,
            0.129,
            0.23,
            0.181,
            0.195,
            0.058,
            0.152,
            0.107,
            0.119,
            0.219,
            0.203,
            0.212,
            0.126,
            0.238,
        ],
        dtype=torch.float32,
    )  # length 30
    _lam = torch.tensor(
        [
            3.236,
            4.201,
            2.701,
            7.775,
            7.148,
            2.465,
            3.02,
            3.322,
            2.285,
            6.004,
            3.106,
            8.801,
            2.956,
            6.918,
            3.708,
            4.403,
            2.077,
            3.884,
            2.171,
            2.76,
            2.567,
            8.636,
            3.299,
            4.666,
            4.208,
            2.285,
            2.469,
            2.362,
            3.963,
            2.099,
        ],
        dtype=torch.float32,
    )  # length 30

    def __init__(self, num_objectives: int = 3, dim: Optional[int] = None):
        if num_objectives < 2:
            raise FunctionDefinitionAssertionError("DTLZ2 requires ≥2 objectives")
        if dim is None:
            dim = num_objectives + 9
        if dim < num_objectives + 1 or dim > 30:
            raise FunctionDefinitionAssertionError(
                f"Dimension must be in [{num_objectives + 1}..30]"
            )
        super().__init__(
            dim,
            num_objectives=num_objectives,
            num_constraints=0,
            bounds=[(0.0, 1.0)] * dim,
        )

    def _evaluate_implementation(self, X: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        n, m = self.dim, self.num_objectives
        o = self._o[:n].to(X.device)
        d_l = self._d_l[:n].to(X.device)
        lam = self._lam[:n].to(X.device)
        z = X - o

        # compute zz and p
        zz = torch.where(z < 0, -lam * z, z)
        p = torch.where(z < 0, -z / d_l, torch.zeros_like(z))

        # g and psums
        k = n - m + 1
        g_term = ((zz[:, n - k :] - 0.5) ** 2).sum(dim=1)
        psum = torch.zeros(X.shape[0], m, device=X.device)
        for i in range(m):
            # sum p for relevant vars
            if i == 0:
                psum[:, i] = torch.sqrt((p[:, : n - k] ** 2).sum(dim=1))
            else:
                psum[:, i] = torch.sqrt((p[:, : n - k + i] ** 2).sum(dim=1))

        # assemble f
        fx = []
        for i in range(m):
            ff = (1 + g_term).unsqueeze(1)
            for j in range(m - i - 1):
                ff = ff * torch.cos(zz[:, j] * math.pi / 2.0).unsqueeze(1)
            if i > 0:
                j = m - i - 1
                ff = ff * torch.sin(zz[:, j] * math.pi / 2.0).unsqueeze(1)
            fx.append((2.0 / (1.0 + torch.exp(-psum[:, i]))) * (ff.squeeze(1) + 1.0))
        fx = torch.stack(fx, dim=1)
        return None, -fx


class CEC2007_S_DTLZ3(CEC2007_S_DTLZ2):
    # identical shift+bounds to S_DTLZ2

    def _evaluate_implementation(self, X: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        # same as DTLZ2 but g_term uses cos(20π(zz[i]-0.5)) and scaled by 100
        n, m = self.dim, self.num_objectives
        o = self._o[:n].to(X.device)
        d_l = self._d_l[:n].to(X.device)
        lam = self._lam[:n].to(X.device)
        z = X - o

        zz = torch.where(z < 0, -lam * z, z)
        p = torch.where(z < 0, -z / d_l, torch.zeros_like(z))

        k = n - m + 1
        term = (
            (zz[:, n - k :] - 0.5) ** 2
            - torch.cos(20.0 * math.pi * (zz[:, n - k :] - 0.5))
        ).sum(dim=1)
        g_term = 100.0 * (k + term)

        psum = torch.zeros(X.shape[0], m, device=X.device)
        for i in range(m):
            psum[:, i] = torch.sqrt((p[:, : n - k + i] ** 2).sum(dim=1))

        fx = []
        for i in range(m):
            ff = (1 + g_term).unsqueeze(1)
            for j in range(m - i - 1):
                ff = ff * torch.cos(zz[:, j] * math.pi / 2.0).unsqueeze(1)
            if i > 0:
                j = m - i - 1
                ff = ff * torch.sin(zz[:, j] * math.pi / 2.0).unsqueeze(1)
            fx.append((2.0 / (1.0 + torch.exp(-psum[:, i]))) * (ff.squeeze(1) + 1.0))
        fx = torch.stack(fx, dim=1)
        return None, -fx


# =============================================================================
# R_ZDT4
# =============================================================================


class CEC2007_R_ZDT4(BenchmarkProblem):
    available_dimensions = [10, 30]
    input_type = DataType.CONTINUOUS
    num_objectives = 2
    num_constraints = 0

    def __init__(self, dim: int):
        if dim not in (10, 30):
            raise FunctionDefinitionAssertionError(
                "R_ZDT4 only supports dim=10 or dim=30"
            )
        super().__init__(
            dim,
            num_objectives=2,
            num_constraints=0,
            bounds=[(0.0, 1.0)] + [(-5.0, 5.0)] * (dim - 1),
        )
        # load rotation matrix M and lam for chosen dim
        if dim == 10:
            self.M = torch.tensor(
                [
                    [0.522, -0.230, 0.087, 0.806, 0.131, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.009, 0.648, -0.707, 0.229, 0.167, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.404, 0.391, 0.160, -0.036, -0.811, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-0.455, -0.450, -0.448, 0.303, -0.546, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-0.598, 0.415, 0.515, 0.452, -0.016, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
            self.lam = torch.tensor(
                [0.042, 0.483, 0.510, 0.390, 0.459, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=torch.float32,
            )
        else:
            # 30x30 rotation and lambda
            self.M = torch.tensor(
                [
                    [
                        -0.087,
                        0.057,
                        -0.403,
                        0.349,
                        0.114,
                        -0.206,
                        0.014,
                        0.335,
                        0.341,
                        0.100,
                        -0.504,
                        -0.127,
                        -0.230,
                        -0.186,
                        0.230,
                    ]
                    + [0.0] * 15,
                    [
                        0.124,
                        0.496,
                        0.007,
                        -0.121,
                        -0.168,
                        0.130,
                        0.270,
                        -0.210,
                        0.222,
                        0.135,
                        0.232,
                        -0.498,
                        -0.013,
                        -0.003,
                        0.439,
                    ]
                    + [0.0] * 15,
                    [
                        0.047,
                        -0.215,
                        -0.117,
                        -0.107,
                        -0.152,
                        0.442,
                        -0.229,
                        -0.172,
                        0.155,
                        -0.591,
                        -0.076,
                        0.025,
                        -0.048,
                        -0.432,
                        0.239,
                    ]
                    + [0.0] * 15,
                    [
                        0.220,
                        -0.082,
                        0.078,
                        -0.130,
                        0.530,
                        -0.100,
                        0.123,
                        -0.206,
                        -0.071,
                        0.048,
                        -0.004,
                        -0.382,
                        -0.213,
                        -0.489,
                        -0.368,
                    ]
                    + [0.0] * 15,
                    [
                        -0.418,
                        -0.423,
                        -0.261,
                        -0.271,
                        -0.219,
                        -0.449,
                        -0.044,
                        -0.134,
                        0.263,
                        0.065,
                        0.304,
                        -0.217,
                        0.120,
                        -0.089,
                        -0.020,
                    ]
                    + [0.0] * 15,
                    [
                        0.371,
                        -0.463,
                        0.229,
                        0.087,
                        0.312,
                        -0.209,
                        0.059,
                        -0.074,
                        0.027,
                        0.093,
                        0.049,
                        0.111,
                        0.191,
                        0.060,
                        0.612,
                    ]
                    + [0.0] * 15,
                    [
                        -0.034,
                        -0.066,
                        0.148,
                        0.416,
                        -0.178,
                        -0.172,
                        0.005,
                        0.043,
                        -0.172,
                        -0.143,
                        0.454,
                        0.029,
                        -0.682,
                        -0.060,
                        0.101,
                    ]
                    + [0.0] * 15,
                    [
                        -0.398,
                        -0.152,
                        0.434,
                        -0.122,
                        0.043,
                        0.282,
                        -0.437,
                        0.056,
                        -0.073,
                        0.380,
                        -0.198,
                        -0.274,
                        -0.208,
                        0.023,
                        0.178,
                    ]
                    + [0.0] * 15,
                    [
                        -0.010,
                        -0.376,
                        -0.311,
                        -0.070,
                        -0.099,
                        0.497,
                        0.508,
                        0.147,
                        -0.156,
                        0.393,
                        0.086,
                        0.100,
                        -0.133,
                        -0.081,
                        0.008,
                    ]
                    + [0.0] * 15,
                    [
                        0.256,
                        -0.066,
                        0.423,
                        -0.245,
                        -0.181,
                        0.003,
                        0.101,
                        0.511,
                        0.564,
                        0.003,
                        0.085,
                        0.087,
                        -0.088,
                        -0.093,
                        -0.195,
                    ]
                    + [0.0] * 15,
                    [
                        0.014,
                        0.257,
                        -0.063,
                        -0.320,
                        -0.081,
                        -0.222,
                        -0.139,
                        0.395,
                        -0.448,
                        0.115,
                        0.112,
                        0.173,
                        0.120,
                        -0.510,
                        0.256,
                    ]
                    + [0.0] * 15,
                    [
                        -0.381,
                        0.121,
                        0.040,
                        -0.462,
                        0.399,
                        -0.062,
                        0.332,
                        0.020,
                        0.037,
                        -0.303,
                        -0.071,
                        0.246,
                        -0.322,
                        0.217,
                        0.203,
                    ]
                    + [0.0] * 15,
                    [
                        0.283,
                        -0.162,
                        -0.275,
                        -0.214,
                        0.069,
                        0.033,
                        -0.191,
                        0.421,
                        -0.237,
                        -0.264,
                        0.070,
                        -0.481,
                        -0.086,
                        0.429,
                        0.002,
                    ]
                    + [0.0] * 15,
                    [
                        0.366,
                        0.079,
                        -0.309,
                        -0.318,
                        -0.022,
                        -0.040,
                        -0.398,
                        -0.294,
                        0.171,
                        0.329,
                        0.009,
                        0.322,
                        -0.398,
                        0.127,
                        -0.006,
                    ]
                    + [0.0] * 15,
                    [
                        -0.171,
                        0.138,
                        -0.187,
                        0.195,
                        0.512,
                        0.280,
                        -0.264,
                        0.204,
                        0.266,
                        0.067,
                        0.556,
                        0.105,
                        0.175,
                        -0.029,
                        -0.002,
                    ]
                    + [0.0] * 15,
                ]
                + [[1.0 if i == j else 0.0 for j in range(30)] for i in range(15, 30)],
                dtype=torch.float32,
            )
            self.lam = torch.tensor(
                [
                    0.011,
                    0.125,
                    0.128,
                    0.128,
                    0.115,
                    0.132,
                    0.151,
                    0.117,
                    0.128,
                    0.134,
                    0.120,
                    0.117,
                    0.118,
                    0.118,
                    0.124,
                ]
                + [1.0] * 15,
                dtype=torch.float32,
            )

    def _evaluate_implementation(self, X: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        n = self.dim
        M = self.M.to(X.device)  # rotation matrix
        lam = self.lam.to(X.device)  # lambda vector

        # 1) rotate
        Z = X @ M.T

        # 2) first variable
        z0 = Z[:, 0]
        in01 = (z0 >= 0.0) & (z0 <= 1.0)

        # zz0: clipped & scaled
        zz0 = torch.where(
            in01, z0, torch.where(z0 < 0.0, -lam[0] * z0, 1.0 - lam[0] * (z0 - 1.0))
        )

        # p0: penalty exponent
        p0 = torch.where(
            in01, torch.zeros_like(z0), torch.where(z0 < 0.0, -z0, z0 - 1.0)
        )

        # f0
        f0 = torch.where(in01, zz0 + 1.0, 2.0 / (1.0 + torch.exp(-p0)) * (zz0 + 1.0))

        # 3) the remaining variables
        Z_rest = Z[:, 1:]
        zz_rest = torch.empty_like(Z_rest)
        p_rest = torch.empty_like(Z_rest)

        for i in range(1, n):
            zi = Z[:, i]
            under = zi < -5.0
            over = zi > +5.0

            # zz_rest[:, i-1]
            zz_rest[:, i - 1] = torch.where(
                under,
                -5.0 - lam[i] * (5.0 + zi),
                torch.where(over, 5.0 - lam[i] * (zi - 5.0), zi),
            )

            # p_rest[:, i-1]
            p_rest[:, i - 1] = torch.where(
                under, -5.0 - zi, torch.where(over, zi - 5.0, torch.zeros_like(zi))
            )

        # 4) g and psum
        sum_term = (zz_rest**2 - 10.0 * torch.cos(4.0 * math.pi * zz_rest)).sum(dim=1)
        psum = torch.sqrt((p_rest**2).sum(dim=1))
        g = 1.0 + 10.0 * (n - 1) + sum_term

        # 5) second objective uses zz0, not raw z0
        f1 = (g * (1.0 - torch.sqrt(zz0 / g)) + 1.0) * (2.0 / (1.0 + torch.exp(-psum)))

        fx = torch.stack((f0, f1), dim=1)
        return None, -fx


# =============================================================================
# R_DTLZ2 (rotated DTLZ2)
# =============================================================================
class CEC2007_R_DTLZ2(BenchmarkProblem):
    available_dimensions = [10, 30]
    input_type = DataType.CONTINUOUS
    num_objectives = (2, None)
    num_constraints = 0

    def __init__(self, num_objectives: int = 3, dim: Optional[int] = None):
        if dim is None:
            dim = num_objectives + 7  # common default
        if dim not in (10, 30):
            raise FunctionDefinitionAssertionError(
                "R_DTLZ2 supports only dim=10 or dim=30"
            )
        if num_objectives < 2 or num_objectives > dim:
            raise FunctionDefinitionAssertionError("num_objectives must be in [2..dim]")
        self.dim = dim
        self.M = num_objectives
        super().__init__(
            dim,
            num_objectives=num_objectives,
            num_constraints=0,
            bounds=[(0.0, 1.0)] * dim,
        )

        # load rotation matrix and lambda for chosen dim
        if dim == 10:
            self.Mat = torch.tensor(
                [
                    [-0.444, -0.380, -0.510, 0.124, 0.619, 0, 0, 0, 0, 0],
                    [0.214, -0.570, -0.445, 0.239, -0.612, 0, 0, 0, 0, 0],
                    [-0.675, 0.462, -0.336, -0.093, -0.458, 0, 0, 0, 0, 0],
                    [0.526, 0.376, -0.644, -0.379, 0.154, 0, 0, 0, 0, 0],
                    [0.160, 0.419, -0.120, 0.880, 0.097, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
            )
            self.lam = torch.tensor(
                [0.313, 0.312, 0.321, 0.316, 0.456, 1, 1, 1, 1, 1], dtype=torch.float32
            )
        else:  # dim == 30
            first15 = [
                [
                    -0.376,
                    0.392,
                    -0.034,
                    0.074,
                    -0.124,
                    -0.013,
                    -0.430,
                    0.168,
                    0.144,
                    0.334,
                    0.054,
                    -0.486,
                    0.255,
                    -0.081,
                    0.163,
                ]
                + [0.0] * 15,
                [
                    -0.202,
                    -0.401,
                    0.426,
                    0.136,
                    0.123,
                    -0.437,
                    -0.297,
                    -0.182,
                    0.417,
                    0.119,
                    0.150,
                    0.190,
                    -0.129,
                    -0.026,
                    0.085,
                ]
                + [0.0] * 15,
                [
                    -0.218,
                    -0.035,
                    0.074,
                    -0.107,
                    -0.412,
                    -0.093,
                    0.659,
                    0.181,
                    0.291,
                    0.148,
                    -0.102,
                    -0.082,
                    -0.185,
                    -0.200,
                    0.300,
                ]
                + [0.0] * 15,
                [
                    0.053,
                    -0.211,
                    -0.173,
                    -0.130,
                    0.579,
                    0.034,
                    0.065,
                    0.171,
                    0.132,
                    -0.143,
                    -0.081,
                    -0.311,
                    -0.039,
                    0.279,
                    0.562,
                ]
                + [0.0] * 15,
                [
                    -0.104,
                    0.331,
                    -0.031,
                    -0.347,
                    0.036,
                    -0.345,
                    0.077,
                    0.236,
                    0.130,
                    -0.088,
                    0.323,
                    -0.020,
                    -0.280,
                    0.503,
                    -0.340,
                ]
                + [0.0] * 15,
                [
                    -0.219,
                    -0.212,
                    0.294,
                    -0.514,
                    -0.332,
                    0.373,
                    -0.264,
                    0.050,
                    -0.058,
                    -0.428,
                    0.134,
                    0.006,
                    0.062,
                    0.047,
                    0.155,
                ]
                + [0.0] * 15,
                [
                    -0.165,
                    0.324,
                    0.386,
                    -0.425,
                    0.366,
                    -0.075,
                    0.033,
                    -0.112,
                    -0.335,
                    0.261,
                    -0.344,
                    0.189,
                    -0.100,
                    -0.176,
                    0.096,
                ]
                + [0.0] * 15,
                [
                    -0.135,
                    0.328,
                    -0.066,
                    0.268,
                    0.062,
                    0.093,
                    0.011,
                    -0.154,
                    -0.203,
                    -0.167,
                    0.562,
                    0.202,
                    -0.360,
                    -0.181,
                    0.417,
                ]
                + [0.0] * 15,
                [
                    0.238,
                    -0.064,
                    0.093,
                    0.111,
                    -0.143,
                    -0.131,
                    -0.303,
                    0.296,
                    -0.185,
                    -0.138,
                    -0.272,
                    -0.317,
                    -0.655,
                    -0.206,
                    -0.041,
                ]
                + [0.0] * 15,
                [
                    0.623,
                    0.161,
                    0.259,
                    -0.132,
                    -0.110,
                    0.340,
                    -0.070,
                    -0.132,
                    0.288,
                    0.413,
                    0.165,
                    -0.020,
                    -0.137,
                    0.175,
                    0.146,
                ]
                + [0.0] * 15,
                [
                    -0.008,
                    -0.130,
                    -0.578,
                    -0.273,
                    -0.154,
                    -0.064,
                    -0.313,
                    0.199,
                    0.028,
                    0.305,
                    -0.087,
                    0.504,
                    -0.119,
                    -0.032,
                    0.195,
                ]
                + [0.0] * 15,
                [
                    -0.346,
                    -0.347,
                    0.078,
                    0.209,
                    -0.024,
                    0.315,
                    0.083,
                    0.056,
                    -0.381,
                    0.452,
                    0.069,
                    -0.061,
                    -0.231,
                    0.425,
                    -0.093,
                ]
                + [0.0] * 15,
                [
                    0.168,
                    -0.274,
                    -0.170,
                    -0.349,
                    -0.098,
                    -0.361,
                    0.059,
                    -0.355,
                    -0.355,
                    0.190,
                    0.351,
                    -0.392,
                    0.068,
                    -0.178,
                    0.031,
                ]
                + [0.0] * 15,
                [
                    0.259,
                    0.026,
                    0.291,
                    0.178,
                    -0.206,
                    -0.358,
                    0.011,
                    0.419,
                    -0.374,
                    -0.001,
                    0.059,
                    0.182,
                    0.363,
                    0.237,
                    0.326,
                ]
                + [0.0] * 15,
                [
                    0.064,
                    -0.173,
                    0.104,
                    -0.088,
                    0.327,
                    0.178,
                    0.075,
                    0.577,
                    0.031,
                    0.149,
                    0.402,
                    0.035,
                    0.096,
                    -0.463,
                    -0.248,
                ]
                + [0.0] * 15,
            ]
            # append identity for rows 16..30
            id_rows = [
                [1.0 if j == i else 0.0 for j in range(30)] for i in range(15, 30)
            ]
            self.Mat = torch.tensor(first15 + id_rows, dtype=torch.float32)

            self.lam = torch.tensor(
                [
                    0.113,
                    0.105,
                    0.117,
                    0.119,
                    0.108,
                    0.110,
                    0.101,
                    0.107,
                    0.111,
                    0.109,
                    0.120,
                    0.108,
                    0.101,
                    0.105,
                    0.116,
                ]
                + [1.0] * 15,
                dtype=torch.float32,
            )

    def _evaluate_implementation(self, X: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        n, m = self.dim, self.M
        M = self.Mat.to(X.device)
        lam = self.lam.to(X.device)

        # rotate
        Z = X @ M.T

        # compute zz and p
        zz = torch.empty_like(Z)
        p = torch.empty_like(Z)
        for i in range(n):
            zi = Z[:, i]
            # three cases
            in01 = (zi >= 0.0) & (zi <= 1.0)
            below = zi < 0.0
            # above = zi > 1.0
            zz[:, i] = torch.where(
                in01, zi, torch.where(below, -lam[i] * zi, 1.0 - lam[i] * (zi - 1.0))
            )
            p[:, i] = torch.where(
                in01, torch.zeros_like(zi), torch.where(below, -zi, zi - 1.0)
            )

        # compute g and psum per objective
        k = n - m + 1
        g = (
            (zz[:, n - k :] - 0.5) ** 2
            - torch.cos(20.0 * math.pi * (zz[:, n - k :] - 0.5))
        ).sum(dim=1)
        g = 100.0 * (k + g)

        psum = torch.zeros((X.shape[0], m), device=X.device)
        for i in range(m):
            # accumulate p[0..(n-k+i-1)]
            length = n - k + i
            if length > 0:
                psum[:, i] = torch.sqrt((p[:, :length] ** 2).sum(dim=1))
            else:
                psum[:, i] = 0.0

        # assemble f
        fx = []
        for i in range(m):
            F = (1.0 + g).unsqueeze(1)
            # cos terms
            for j in range(m - i - 1):
                F = F * torch.cos(zz[:, j].unsqueeze(1) * math.pi / 2.0)
            # sin term
            if i > 0:
                idx = m - i - 1
                F = F * torch.sin(zz[:, idx].unsqueeze(1) * math.pi / 2.0)
            F = F.squeeze(1)
            fx.append((2.0 / (1.0 + torch.exp(-psum[:, i]))) * (F + 1.0))
        fx = torch.stack(fx, dim=1)
        return None, -fx


# =============================================================================
# WFG1
# =============================================================================


class CEC2007_WFG1(BenchmarkProblem):
    available_dimensions = (2, None)
    input_type = DataType.CONTINUOUS
    num_objectives = (2, None)
    num_constraints = 0

    def __init__(self, num_objectives: int = 3, dim: Optional[int] = None):
        if num_objectives < 2:
            raise FunctionDefinitionAssertionError("WFG1 requires ≥2 objectives")
        k = 2 * (num_objectives - 1)
        if dim is None:
            dim = k + 10  # default tail length = 10
        if dim < k + 1:
            raise FunctionDefinitionAssertionError(f"Dimension must be ≥ k+1 = {k + 1}")
        self.k = k
        self.M = num_objectives
        super().__init__(
            dim,
            num_objectives=num_objectives,
            num_constraints=0,
            # bounds: zi ∈ [0, 2*(i+1)]
            bounds=[(0.0, 2.0 * (i + 1)) for i in range(dim)],
        )

    @staticmethod
    def _s_linear(y: Tensor, A: float) -> Tensor:
        # |y - A| / |floor(A - y) + A|
        return torch.abs(y - A) / torch.abs(torch.floor(A - y) + A)

    @staticmethod
    def _b_flat(y: Tensor, A: float, B: float, C: float) -> Tensor:
        # tmp1 = min(0, floor(y - B)) * A*(B - y)/B
        tmp1 = torch.minimum(torch.zeros_like(y), torch.floor(y - B)) * (
            A * (B - y) / B
        )
        # tmp2 = min(0, floor(C - y)) * (1-A)*(y - C)/(1-C)
        tmp2 = torch.minimum(torch.zeros_like(y), torch.floor(C - y)) * (
            (1 - A) * (y - C) / (1 - C)
        )
        return A + tmp1 - tmp2

    @staticmethod
    def _b_poly(y: Tensor, alpha: float) -> Tensor:
        return y**alpha

    @staticmethod
    def _r_sum(y: Tensor, w: Tensor) -> Tensor:
        # weighted sum over last dim, returns (batch,)
        return (y * w).sum(dim=1) / w.sum()

    @staticmethod
    def _convex(x: Tensor, m: int, M: int) -> Tensor:
        # x: (batch, M), m: 1..M
        batch = x.shape[0]
        ones = torch.ones(batch, device=x.device)
        result = ones.clone()
        for i in range(1, M - m + 1):
            result = result * (1.0 - torch.cos(x[:, i - 1] * math.pi / 2.0))
        if m != 1:
            result = result * (1.0 - torch.sin(x[:, M - m] * math.pi / 2.0))
        return result

    @staticmethod
    def _mixed(x0: Tensor, A: int, alpha: float) -> Tensor:
        # mixed for last objective
        tmp = 2.0 * A * math.pi
        return (1.0 - x0 - torch.cos(tmp * x0 + math.pi / 2) / tmp) ** alpha

    def _evaluate_implementation(self, Z: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        n, M, k = self.dim, self.M, self.k
        batch = Z.shape[0]

        # 1) normalize: y[i] = z[i] / (2*(i+1))
        denom = torch.arange(2.0, 2.0 * (n + 1), 2.0, device=Z.device)  # [2,4,...,2n]
        Y = Z / denom

        # 2) t1: copy first k, then s_linear on tail
        T1 = Y.clone()
        if k < n:
            T1[:, k:] = self._s_linear(Y[:, k:], 0.35)

        # 3) t2: copy first k, then b_flat on tail
        T2 = T1.clone()
        if k < n:
            T2[:, k:] = self._b_flat(T1[:, k:], 0.8, 0.75, 0.85)

        # 4) t3: b_poly on all
        T3 = self._b_poly(T2, 0.02)

        # 5) t4: group-wise r_sum into M values
        w = denom.clone()  # same weights
        T4 = torch.zeros((batch, M), device=Z.device)
        # first M-1 groups
        for i in range(1, M):
            head = (i - 1) * k // (M - 1)
            tail = i * k // (M - 1)
            T4[:, i - 1] = self._r_sum(T3[:, head:tail], w[head:tail])
        # last group
        T4[:, M - 1] = self._r_sum(T3[:, k:], w[k:])

        # 6) shape -> x
        X = torch.zeros_like(T4)
        for i in range(M - 1):
            tmp = torch.max(T4[:, M - 1], torch.ones(batch, device=Z.device))
            X[:, i] = tmp * (T4[:, i] - 0.5) + 0.5
        X[:, M - 1] = T4[:, M - 1]

        # 7) compute h
        H = torch.zeros_like(X)
        for m in range(1, M):
            H[:, m - 1] = self._convex(X, m, M)
        H[:, M - 1] = self._mixed(X[:, 0], 5, 1.0)

        # 8) final f
        S = torch.arange(2.0, 2.0 * (M + 1), 2.0, device=Z.device)
        f = X[:, -1].unsqueeze(1) + S * H  # shape (batch, M)

        return None, -f


# =============================================================================
# WFG8
# =============================================================================
class CEC2007_WFG8(BenchmarkProblem):
    available_dimensions = (2, None)
    input_type = DataType.CONTINUOUS
    num_objectives = (2, None)
    num_constraints = 0

    def __init__(self, num_objectives: int = 3, dim: Optional[int] = None):
        if num_objectives < 2:
            raise FunctionDefinitionAssertionError("WFG8 requires ≥2 objectives")
        if dim is None:
            k = 2 * (num_objectives - 1)
            dim = k + 10
        k = 2 * (num_objectives - 1)
        if dim < k + 1:
            raise FunctionDefinitionAssertionError("Dimension must be ≥ k+1")
        self.k = k
        self.M = num_objectives
        super().__init__(
            dim,
            num_objectives=num_objectives,
            num_constraints=0,
            bounds=[(0.0, 2.0 * (i + 1)) for i in range(dim)],
        )

    def _evaluate_implementation(self, Z: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        n, M, k = self.dim, self.M, self.k
        device = Z.device

        # 1) normalize
        scale = torch.arange(2.0, 2.0 * (n + 1), 2.0, device=device)
        Y = Z / scale

        # 2) t1: apply b_param for i >= k
        T1 = Y.clone()
        eps = 1e-12
        A, B, C = 0.98 / 49.98, 0.02, 50.0
        for i in range(k, n):
            yi = Y[:, i].clamp(min=eps)
            u = (Y[:, :i].sum(dim=1)) / i  # w = 1 for all, so sum/ i
            floor_term = torch.floor(0.5 - u)
            v = A - (1.0 - 2.0 * u) * torch.abs(floor_term + A)
            alpha = B + (C - B) * v
            T1[:, i] = yi.pow(alpha)

        # 3) t2: same as WFG1_t1 → s_linear
        T2 = T1.clone()
        for i in range(k, n):
            yi = T1[:, i]
            # s_linear(y, A) = |y - A| / |floor(A - y) + A|
            num = torch.abs(yi - 0.35)
            den = torch.abs(torch.floor(0.35 - yi) + 0.35).clamp(min=eps)
            T2[:, i] = (num / den).clamp(0.0, 1.0)

        # 4) t3: b_poly with α=0.02
        T3 = T2.pow(0.02)

        # 5) t4: r_sum reduce to M values
        w = scale
        T4 = torch.zeros((Z.shape[0], M), device=device)
        for m in range(1, M):
            head = (m - 1) * k // (M - 1)
            tail = m * k // (M - 1)
            Ysub, Wsub = T3[:, head:tail], w[head:tail]
            T4[:, m - 1] = (Ysub * Wsub).sum(dim=1) / Wsub.sum()
        # last group
        Ysub, Wsub = T3[:, k:], w[k:]
        T4[:, M - 1] = (Ysub * Wsub).sum(dim=1) / Wsub.sum()

        # 6) shape (same as WFG1 convex/concave)
        X = torch.zeros_like(T4)
        for m in range(M - 1):
            tmp = torch.max(T4[:, M - 1], torch.ones_like(T4[:, 0]))
            X[:, m] = tmp * (T4[:, m] - 0.5) + 0.5
        X[:, M - 1] = T4[:, M - 1]

        H = torch.zeros_like(X)
        # convex for m=1..M-1
        for m in range(1, M):
            prod = torch.ones(Z.shape[0], device=device)
            for j in range(1, M - m + 1):
                prod = prod * (1 - torch.cos(X[:, j - 1] * math.pi / 2))
            if m != 1:
                prod = prod * (1 - torch.sin(X[:, M - m] * math.pi / 2))
            H[:, m - 1] = prod
        # mixed for last
        A_mix, alpha_mix = 5, 1.0
        tmp = 2.0 * A_mix * math.pi
        H[:, M - 1] = (1 - X[:, 0] - torch.cos(tmp * X[:, 0] + math.pi / 2) / tmp).pow(
            alpha_mix
        )

        S = torch.arange(2.0, 2.0 * (M + 1), 2.0, device=device)
        F = X[:, -1].unsqueeze(1) + S * H

        return None, -F


# =============================================================================
# WFG9
# =============================================================================
class CEC2007_WFG9(CEC2007_WFG1):
    def _evaluate_implementation(self, Z: Tensor) -> Tuple[Optional[Tensor], Tensor]:
        n, M, k = self.dim, self.M, self.k
        device = Z.device

        # 1) normalize
        weights = torch.arange(2.0, 2.0 * (n + 1), 2.0, device=device)
        Y = Z / weights

        # 2) t1: b_param
        # C b_param: v = A - (1-2u)*|floor(0.5 - u) + A|
        #           exponent = B + (C - B)*v
        #           t1[i] = y[i] ** exponent
        A0, B0, C0 = 0.98 / 49.98, 0.02, 50.0
        T1 = Y.clone()
        for i in range(n - 1):
            # compute u = r_sum over Y[i+1 .. end]
            u = (Y[:, i + 1 :] * weights[i + 1 :]).sum(dim=1) / weights[i + 1 :].sum()
            v = A0 - (1.0 - 2.0 * u) * torch.abs(torch.floor(0.5 - u) + A0)
            exp = B0 + (C0 - B0) * v
            T1[:, i] = torch.pow(Y[:, i].clamp(min=1e-12), exp)  # clamp to avoid 0**neg
        T1[:, -1] = Y[:, -1]

        # 3) t2: deceptive (first k) and multi (rest)
        T2 = T1.clone()

        # s_decept for i < k
        A1, B1, C1 = 0.35, 0.001, 0.05
        for i in range(k):
            yi = T1[:, i]
            tmp1 = torch.floor(yi - A1 + B1)
            tmp2 = torch.floor(A1 + B1 - yi)
            coef1 = (1.0 - C1 + (A1 - B1) / B1) / (A1 - B1)
            coef2 = (1.0 - C1 + (1.0 - A1 - B1) / B1) / (1.0 - A1 - B1)
            T2[:, i] = 1.0 + (torch.abs(yi - A1) - B1) * (
                tmp1 * coef1 + tmp2 * coef2 + 1.0 / B1
            )

        # s_multi for i >= k
        A2, B2, C2 = 30.0, 95.0, 0.35
        for i in range(k, n):
            yi = T1[:, i]
            tmp1 = torch.abs(yi - C2) / (2.0 * (torch.floor(C2 - yi) + C2))
            tmp2 = (4.0 * A2 + 2.0) * math.pi * (0.5 - tmp1)
            T2[:, i] = (1.0 + torch.cos(tmp2) + 4.0 * B2 * tmp1 * tmp1) / (B2 + 2.0)

        # 4) now the same tail-reduction (r_sum) + shape steps as WFG1, but on T2
        # (we inline rather than call super() to skip re-normalizing)
        # r_sum to produce M values per sample:
        S = torch.arange(2.0, 2.0 * (n + 1), 2.0, device=device)
        T4 = torch.zeros((Z.shape[0], M), device=device)
        for m in range(1, M):
            head = (m - 1) * k // (M - 1)
            tail = m * k // (M - 1)
            Ysub = T2[:, head:tail]
            Wsub = S[head:tail]
            T4[:, m - 1] = (Ysub * Wsub).sum(dim=1) / Wsub.sum()
        # last group
        Ysub = T2[:, k:]
        Wsub = S[k:]
        T4[:, M - 1] = (Ysub * Wsub).sum(dim=1) / Wsub.sum()

        # 5) shape
        X = torch.zeros_like(T4)
        for m in range(M - 1):
            X[:, m] = (
                torch.max(T4[:, M - 1], torch.ones_like(T4[:, 0])) * (T4[:, m] - 0.5)
                + 0.5
            )
        X[:, M - 1] = T4[:, M - 1]

        H = torch.zeros_like(X)
        # convex for m=1..M-1
        for m in range(1, M):
            prod = torch.ones(Z.shape[0], device=device)
            for j in range(1, M - m + 1):
                prod = prod * (1.0 - torch.cos(X[:, j - 1] * math.pi / 2.0))
            if m != 1:
                prod = prod * (1.0 - torch.sin(X[:, M - m] * math.pi / 2.0))
            H[:, m - 1] = prod
        # mixed for last objective
        A3, alpha3 = 5.0, 1.0
        tmp = 2.0 * A3 * math.pi * X[:, 0]
        H[:, M - 1] = torch.pow(
            1.0 - X[:, 0] - torch.cos(tmp + math.pi / 2.0) / tmp.clamp(min=1e-12),
            alpha3,
        )

        # final f
        S_m = torch.arange(2.0, 2.0 * (M + 1), 2.0, device=device)
        f = X[:, -1].unsqueeze(1) + S_m * H

        return None, -f
