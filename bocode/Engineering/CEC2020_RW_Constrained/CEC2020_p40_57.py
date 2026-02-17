"""
https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation/
"""

from pathlib import Path

import numpy as np
import torch

from ...base import BenchmarkProblem, DataType
from .helperFuncs import ybus, Fitness


class CEC2020_p40(BenchmarkProblem):
    """
    CEC2020 Problem 40
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 76
    num_constraints = 76

    def __init__(self):
        super().__init__(
            dim=76,
            num_objectives=1,
            num_constraints=76,
            #  X_opt=[[0] * 76],
            optimum=[0.0],
            bounds=[(-1.0, 1.0)] * 76,
        )
        self.initialized = False
        self.P = None
        self.Q = None
        self.L = None

    def _evaluate_implementation(self, X, scaling=False):
        if not self.initialized:
            script_dir = Path(__file__).parent
            self.P = np.loadtxt(script_dir / "input_data" / "FunctionPS2_P.txt")
            self.Q = np.loadtxt(script_dir / "input_data" / "FunctionPS2_Q.txt")
            self.L = np.loadtxt(script_dir / "input_data" / "FunctionPS14_linedata.txt")
            self.initialized = True
        P, Q, L = self.P, self.Q, self.L

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        f = np.zeros((n_samples, 1))
        h = np.zeros((n_samples, 76))

        # Voltage initialization
        V = np.zeros(38, dtype=complex)
        V[0] = 1
        Pc = np.zeros(38)
        Qc = np.zeros(38)

        for i in range(n_samples):
            V[1:38] = X[i, 0:37] + 1j * X[i, 37:74]
            Pc[[33, 34, 35, 36, 37]] = 1 / np.array(
                [5.102e-03, 1.502e-03, 4.506e-03, 2.253e-03, 2.253e-03]
            )
            Qc[[33, 34, 35, 36, 37]] = 1 / np.array([0.05, 0.03, 0.05, 0.01, 0.1])
            w = X[i, 74]
            V[0] = X[i, 75] + 1e-5

            # Current calculation
            Y = ybus(L, w)
            I = Y @ V
            Ir = np.real(I)
            Im = np.imag(I)
            Vr = np.real(V)
            Vm = np.imag(V)
            Psp = Pc * (1 - w) - P[:, 0] * (np.abs(V) / P[:, 4]) ** P[:, 5]
            Qsp = (
                Qc * (1 - np.sqrt(Vr**2 + Vm**2))
                - Q[:, 0] * (np.abs(V) / Q[:, 4]) ** Q[:, 5]
            )
            spI = np.conj((Psp + 1j * Qsp) / V)
            spIr = np.real(spI)
            spIm = np.imag(spI)
            delIr = Ir - spIr
            delIm = Im - spIm
            delP2 = Psp - (Vr * Ir + Vm * Im)
            delQ2 = Qsp - (Vm * Ir - Vr * Im)

            # Objective calculation
            f[i, 0] = np.sum(delP2[0:38] ** 2) + np.sum(delQ2[0:38] ** 2)
            h[i, :] = np.concatenate((delIr[0:38], delIm[0:38]))

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f),
        )


class CEC2020_p41(BenchmarkProblem):
    """
    CEC2020 Problem 41
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 74
    num_constraints = 74

    def __init__(self):
        super().__init__(
            dim=74,
            num_objectives=1,
            num_constraints=74,
            #  X_opt=[[0] * 74],
            optimum=[0.0],
            bounds=[(-1.0, 1.0)] * 74,
        )
        self.initialized = False
        self.G = None
        self.B = None
        self.P = None
        self.Q = None

    def _evaluate_implementation(self, X, scaling=False):
        if not self.initialized:
            script_dir = Path(__file__).parent
            self.G = np.loadtxt(script_dir / "input_data" / "FunctionPS2_G.txt")
            self.B = np.loadtxt(script_dir / "input_data" / "FunctionPS2_B.txt")
            self.P = np.loadtxt(script_dir / "input_data" / "FunctionPS2_P.txt")
            self.Q = np.loadtxt(script_dir / "input_data" / "FunctionPS2_Q.txt")
            self.initialized = True
        G, B, P, Q = self.G, self.B, self.P, self.Q

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        f = np.zeros((n_samples, 1))
        h = np.zeros((n_samples, 74))

        Y = G + 1j * B

        # Voltage initialization
        V = np.zeros(38, dtype=complex)
        V[0] = 1
        Pdg = np.zeros(38)
        Qdg = np.zeros(38)
        Pdg[[33, 34, 35, 36, 37]] = 0.2
        Qdg[[33, 34, 35, 36, 37]] = 0.18

        for i in range(n_samples):
            V[1:38] = X[i, 0:37] + 1j * X[i, 37:74]

            # Current calculation
            I = Y @ V
            Ir = np.real(I)
            Im = np.imag(I)
            Vr = np.real(V)
            Vm = np.imag(V)
            Psp = Pdg - P[:, 0] * (np.abs(V) / P[:, 4]) ** P[:, 5]
            Qsp = Qdg - Q[:, 0] * (np.abs(V) / Q[:, 4]) ** Q[:, 5]
            spI = np.conj((Psp + 1j * Qsp) / V)
            spIr = np.real(spI)
            spIm = np.imag(spI)
            delIr = Ir - spIr
            delIm = Im - spIm
            delP = Psp - (Vr * Ir + Vm * Im)
            delQ = Qsp - (Vm * Ir - Vr * Im)

            # Objective calculation and equality constraints
            f[i, 0] = np.sum(delP[1:38] ** 2) + np.sum(delQ[1:38] ** 2)
            h[i, :] = np.concatenate((delIr[1:38], delIm[1:38]))

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f),
        )


class CEC2020_p42(BenchmarkProblem):
    """
    CEC2020 Problem 42
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 86
    num_constraints = 76

    def __init__(self):
        super().__init__(
            dim=86,
            num_objectives=1,
            num_constraints=76,
            #  X_opt=[[0] * 86],
            optimum=[0.077027102],
            bounds=[(-1.0, 1.0)] * 86,
        )
        self.initialized = False
        self.P = None
        self.Q = None
        self.L = None

    def _evaluate_implementation(self, X, scaling=False):
        if not self.initialized:
            # Load the files
            script_dir = Path(__file__).parent
            self.P = np.loadtxt(script_dir / "input_data" / "FunctionPS2_P.txt")
            self.Q = np.loadtxt(script_dir / "input_data" / "FunctionPS2_Q.txt")
            self.L = np.loadtxt(script_dir / "input_data" / "FunctionPS14_linedata.txt")
            self.initialized = True
        P, Q, L = self.P, self.Q, self.L

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        f = np.zeros((n_samples, 1))
        h = np.zeros((n_samples, 76))

        # Voltage initialization
        V = np.zeros(38, dtype=complex)
        V[0] = 1
        Pc = np.zeros(38)
        Qc = np.zeros(38)

        for i in range(n_samples):
            V[1:38] = X[i, 0:37] + 1j * X[i, 37:74]
            w = X[i, 74]
            V[0] = X[i, 75] + 1e-5
            Pc[33:38] = X[i, 76:81]
            Qc[33:38] = X[i, 81:86]

            # Current calculation
            Y = ybus(L, w)
            I = np.dot(Y, V)
            Ir = np.real(I)
            Im = np.imag(I)
            Vr = np.real(V)
            Vm = np.imag(V)
            Psp = Pc * (1 - w) - P[:, 0] * (np.abs(V) / P[:, 4]) ** P[:, 5]
            Qsp = (
                Qc * (1 - np.sqrt(Vr**2 + Vm**2))
                - Q[:, 0] * (np.abs(V) / Q[:, 4]) ** Q[:, 5]
            )
            spI = np.conj((Psp + 1j * Qsp) / V)
            spIr = np.real(spI)
            spIm = np.imag(spI)
            delIr = Ir - spIr
            delIm = Im - spIm

            # Objective calculation and equality constraints
            f[i, 0] = np.sum(Psp)
            h[i, :] = np.hstack([delIr[0:38], delIm[0:38]])

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f),
        )


class CEC2020_p43(BenchmarkProblem):
    """
    CEC2020 Problem 43
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 86
    num_constraints = 76

    def __init__(self):
        super().__init__(
            dim=86,
            num_objectives=1,
            num_constraints=76,
            #  X_opt=[[0] * 86],
            optimum=[0.07983597],
            bounds=[(-1.0, 1.0)] * 86,
        )
        self.initialized = False
        self.P = None
        self.Q = None
        self.L = None

    def _evaluate_implementation(self, X, scaling=False):
        if not self.initialized:
            script_dir = Path(__file__).parent
            self.P = np.loadtxt(script_dir / "input_data" / "FunctionPS2_P.txt")
            self.Q = np.loadtxt(script_dir / "input_data" / "FunctionPS2_Q.txt")
            self.L = np.loadtxt(script_dir / "input_data" / "FunctionPS14_linedata.txt")
            self.initialized = True
        P, Q, L = self.P, self.Q, self.L

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        f = np.zeros((n_samples, 1))

        # Voltage initialization
        V = np.zeros(38, dtype=complex)
        V[0] = 1
        Pc = np.zeros(38)
        Qc = np.zeros(38)
        h = np.zeros((n_samples, 76))

        for i in range(n_samples):
            V[1:38] = X[i, 0:37] + 1j * X[i, 37:74]
            w = X[i, 74]
            V[0] = X[i, 75] + 1e-5
            Pc[33:38] = X[i, 76:81]
            Qc[33:38] = X[i, 81:86]

            # Current calculation
            Y = ybus(L, w)
            I = np.dot(Y, V)
            Ir = np.real(I)
            Im = np.imag(I)
            Vr = np.real(V)
            Vm = np.imag(V)
            Psp = Pc * (1 - w) - P[:, 0] * (np.abs(V) / P[:, 4]) ** P[:, 5]
            Qsp = (
                Qc * (1 - np.sqrt(Vr**2 + Vm**2))
                - Q[:, 0] * (np.abs(V) / Q[:, 4]) ** Q[:, 5]
            )
            spI = np.conj((Psp + 1j * Qsp) / V)
            spIr = np.real(spI)
            spIm = np.imag(spI)
            delIr = Ir - spIr
            delIm = Im - spIm

            # Objective calculation and equality constraints
            f[i, 0] = 0.5 * (np.sum(Qsp) + np.sum(Psp))
            h[i, :] = np.hstack([delIr[0:38], delIm[0:38]])

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f),
        )


class CEC2020_p44(BenchmarkProblem):
    """
    CEC2020 Problem 44
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 30
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=30,
            num_objectives=1,
            num_constraints=0,
            #  X_opt=[[0] * 30],
            optimum=[-6273.1715],
            bounds=[(40.0, 1960.0)] * 30,
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        D = n_samples
        f = np.zeros((n_samples, 1))

        interval = 15
        interval_num = 360 // interval
        cut_in_speed = 3.5
        rated_speed = 14
        cut_out_speed = 25
        R = 40

        CT = 0.8
        a = 1 - np.sqrt(1 - CT)
        kappa = 0.01

        N = 15
        # X = 2000

        k = np.full(interval_num, 2)
        c = [
            7,
            5,
            5,
            5,
            5,
            4,
            5,
            6,
            7,
            7,
            8,
            9.5,
            10,
            8.5,
            8.5,
            6.5,
            4.6,
            2.6,
            8,
            5,
            6.4,
            5.2,
            4.5,
            3.9,
        ]
        fre = [
            0.0003,
            0.0072,
            0.0237,
            0.0242,
            0.0222,
            0.0301,
            0.0397,
            0.0268,
            0.0626,
            0.0801,
            0.1025,
            0.1445,
            0.1909,
            0.1162,
            0.0793,
            0.0082,
            0.0041,
            0.0008,
            0.0010,
            0.0005,
            0.0013,
            0.0031,
            0.0085,
            0.0222,
        ]

        # Objective Function
        for i in range(n_samples):
            f[i, 0] = -Fitness(
                interval_num,
                interval,
                fre,
                N,
                X[i, :],
                a,
                kappa,
                R,
                k,
                c,
                cut_in_speed,
                rated_speed,
                cut_out_speed,
                "origin",
            )

        # Constraint Violation
        XX = np.zeros((n_samples, D // 2))
        YY = np.zeros((n_samples, D // 2))
        for i in range(D // 2):
            XX[:, i] = X[:, 2 * i]
            YY[:, i] = X[:, 2 * i + 1]

        k = 0
        g = np.zeros((n_samples, (D // 2) * ((D // 2) - 1) // 2))
        for i in range(D // 2):
            for j in range(i + 1, D // 2):
                g[:, k] = 5 * R - np.sqrt(
                    (XX[:, i] - XX[:, j]) ** 2 + (YY[:, i] - YY[:, j]) ** 2
                )
                k += 1

        # No equality constraints
        h = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f),
        )


class CEC2020_p45(BenchmarkProblem):
    """
    CEC2020 Problem 45
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 25
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=25,
            num_objectives=1,
            num_constraints=1,
            #  X_opt=[[0] * 25],
            optimum=[0.03073936],
            bounds=[(0.0, 90.0)] * 25,
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        D = n_samples
        f = np.zeros((n_samples, 1))

        m = 0.32
        s = (-np.ones(25)) ** np.arange(2, 27)
        k = np.array(
            [
                5,
                7,
                11,
                13,
                17,
                19,
                23,
                25,
                29,
                31,
                35,
                37,
                41,
                43,
                47,
                49,
                53,
                55,
                59,
                61,
                65,
                67,
                71,
                73,
                77,
                79,
                83,
                85,
                91,
                95,
                97,
            ]
        )

        # Objective function
        for i in range(n_samples):
            su = 0
            for j in range(31):
                su2 = 0
                for l in range(D):
                    su2 += s[l] * np.cos(k[j] * X[i, l] * np.pi / 180)
                su += su2**2 / k[j] ** 4
            f[i, 0] = (su) ** 0.5 / (np.sum(1 / k**4)) ** 0.5

        # Inequality constraints
        g = np.zeros((n_samples, D - 1))
        for i in range(D - 1):
            g[:, i] = X[:, i] - X[:, i + 1] + 1e-6

        # Equality onstraints
        h = np.sum(s * np.cos(X * np.pi / 180), axis=1) - m

        return (
            torch.from_numpy(np.abs(h) - 1e-4).unsqueeze(-1),
            torch.from_numpy(g),
            -torch.from_numpy(f),
        )


class CEC2020_p46(BenchmarkProblem):
    """
    CEC2020 Problem 46
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 25
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=25,
            num_objectives=1,
            num_constraints=1,
            #  X_opt=[[0] * 25],
            optimum=[0.020240335],
            bounds=[(0.0, 90.0)] * 25,
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        D = n_samples
        f = np.zeros((n_samples, 1))

        m = 0.32
        s = [
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
        ]
        k = [
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            25,
            29,
            31,
            35,
            37,
            41,
            43,
            47,
            49,
            53,
            55,
            59,
            61,
            65,
            67,
            71,
            73,
            77,
            79,
            83,
            85,
            91,
            95,
            97,
        ]

        # Objective function
        for i in range(n_samples):
            su = 0
            for j in range(31):
                su2 = 0
                for l in range(D):
                    su2 += s[l] * np.cos(k[j] * X[i, l] * np.pi / 180)
                su += su2**2 / k[j] ** 4
            f[i, 0] = 0.5 * (su) ** 0.5 / (np.sum(1 / np.array(k) ** 4)) ** 0.5

        # Inequality constraints
        g = np.zeros((n_samples, D - 1))
        for i in range(D - 1):
            g[:, i] = X[:, i] - X[:, i + 1] + 1e-6

        # Equality constraints
        h = np.sum(np.array(s) * np.cos(X * np.pi / 180), axis=1) - 2 * m

        return (
            torch.from_numpy(np.abs(h) - 1e-4).unsqueeze(-1),
            torch.from_numpy(g),
            -torch.from_numpy(f),
        )


class CEC2020_p47(BenchmarkProblem):
    """
    CEC2020 Problem 47
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 25
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=25,
            num_objectives=1,
            num_constraints=1,
            #  X_opt=[[0] * 25],
            optimum=[0.012783068],
            bounds=[(0.0, 90.0)] * 25,
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        D = n_samples
        f = np.zeros((n_samples, 1))

        m = 0.36
        s = [
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
        ]
        k = [
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            25,
            29,
            31,
            35,
            37,
            41,
            43,
            47,
            49,
            53,
            55,
            59,
            61,
            65,
            67,
            71,
            73,
            77,
            79,
            83,
            85,
            91,
            95,
            97,
        ]

        # Objective function
        for i in range(n_samples):
            su = 0
            for j in range(31):
                su2 = 0
                for l in range(D):
                    su2 += s[l] * np.cos(k[j] * X[i, l] * np.pi / 180)
                su += su2**2 / k[j] ** 4
            f[i, 0] = (1 / 3) * (su) ** 0.5 / (np.sum(1 / np.array(k) ** 4)) ** 0.5

        # Inequality constraints
        g = np.zeros((n_samples, D - 1))
        for i in range(D - 1):
            g[:, i] = X[:, i] - X[:, i + 1] + 1e-6

        # Equality constraints
        h = np.sum(np.array(s) * np.cos(X * np.pi / 180), axis=1) - 3 * m

        return (
            torch.from_numpy(np.abs(h) - 1e-4).unsqueeze(-1),
            torch.from_numpy(g),
            -torch.from_numpy(f),
        )


class CEC2020_p48(BenchmarkProblem):
    """
    CEC2020 Problem 48
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 30
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=30,
            num_objectives=1,
            num_constraints=1,
            #  X_opt=[[0] * 30],
            optimum=[0.016787535766],
            bounds=[(0.0, 90.0)] * 30,
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        D = n_samples
        f = np.zeros((n_samples, 1))

        m = 0.32
        s = [
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
        ]
        k = [
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            25,
            29,
            31,
            35,
            37,
            41,
            43,
            47,
            49,
            53,
            55,
            59,
            61,
            65,
            67,
            71,
            73,
            77,
            79,
            83,
            85,
            91,
            95,
            97,
        ]

        # Objective function
        for i in range(n_samples):
            su = 0
            for j in range(31):
                su2 = 0
                for l in range(D):
                    su2 += s[l] * np.cos(k[j] * X[i, l] * np.pi / 180)
                su += su2**2 / k[j] ** 4
            f[i, 0] = (1 / 4) * (su) ** 0.5 / (np.sum(1 / np.array(k) ** 4)) ** 0.5

        # Inequality constraints
        g = np.zeros((n_samples, D - 1))
        for i in range(D - 1):
            g[:, i] = X[:, i] - X[:, i + 1] + 1e-6

        # Equality constraints
        h = np.sum(np.array(s) * np.cos(X * np.pi / 180), axis=1) - 4 * m

        return (
            torch.from_numpy(np.abs(h) - 1e-4).unsqueeze(-1),
            torch.from_numpy(g),
            -torch.from_numpy(f),
        )


class CEC2020_p49(BenchmarkProblem):
    """
    CEC2020 Problem 49
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 30
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=30,
            num_objectives=1,
            num_constraints=1,
            #  X_opt=[[0] * 30],
            optimum=[0.00931187418],
            bounds=[(0.0, 90.0)] * 30,
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        D = n_samples
        f = np.zeros((n_samples, 1))

        m = 0.3333
        s = [
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
        ]
        k = [
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            25,
            29,
            31,
            35,
            37,
            41,
            43,
            47,
            49,
            53,
            55,
            59,
            61,
            65,
            67,
            71,
            73,
            77,
            79,
            83,
            85,
            91,
            95,
            97,
        ]

        # Objective function
        for i in range(n_samples):
            su = 0
            for j in range(31):
                su2 = 0
                for l in range(D):
                    su2 += s[l] * np.cos(k[j] * X[i, l] * np.pi / 180)
                su += su2**2 / k[j] ** 4
            f[i, 0] = (1 / 5) * (su) ** 0.5 / (np.sum(1 / np.array(k) ** 4)) ** 0.5

        # Inequality constraints
        g = np.zeros((n_samples, D - 1))
        for i in range(D - 1):
            g[:, i] = X[:, i] - X[:, i + 1] + 1e-6

        # Equality constraints
        h = np.sum(np.array(s) * np.cos(X * np.pi / 180), axis=1) - 5 * m

        return (
            torch.from_numpy(np.abs(h) - 1e-4).unsqueeze(-1),
            torch.from_numpy(g),
            -torch.from_numpy(f),
        )


class CEC2020_p50(BenchmarkProblem):
    """
    CEC2020 Problem 50
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 30
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=30,
            num_objectives=1,
            num_constraints=1,
            #  X_opt=[[0] * 30],
            optimum=[0.01505147],
            bounds=[(0.0, 90.0)] * 30,
        )

    def _evaluate_implementation(self, X, scaling=False):
        import numpy as np

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        D = n_samples
        f = np.zeros((n_samples, 1))

        m = 0.32
        s = [
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
        ]
        k = [
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            25,
            29,
            31,
            35,
            37,
            41,
            43,
            47,
            49,
            53,
            55,
            59,
            61,
            65,
            67,
            71,
            73,
            77,
            79,
            83,
            85,
            91,
            95,
            97,
        ]

        # Objective function
        for i in range(n_samples):
            su = 0
            for j in range(31):
                su2 = 0
                for l in range(D):
                    su2 += s[l] * np.cos(k[j] * X[i, l] * np.pi / 180)
                su += su2**2 / k[j] ** 4
            f[i, 0] = (1 / 6) * (su) ** 0.5 / (np.sum(1 / np.array(k) ** 4)) ** 0.5

        # Inequality constraints
        g = np.zeros((n_samples, D - 1))
        for i in range(D - 1):
            g[:, i] = X[:, i] - X[:, i + 1] + 1e-6

        # Equality constraints
        h = np.sum(np.array(s) * np.cos(X * np.pi / 180), axis=1) - 6 * m

        return (
            torch.from_numpy(np.abs(h) - 1e-4).unsqueeze(-1),
            torch.from_numpy(g),
            -torch.from_numpy(f),
        )


class CEC2020_p51(BenchmarkProblem):
    """
    CEC2020 Problem 51
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 59
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=59,
            num_objectives=1,
            num_constraints=1,
            #  X_opt=[[0] * 59],
            optimum=[4550.8511497],
            bounds=[(0.0, 10.0)] * 59,
        )
        self.initialized = False
        self.P = None

    def _evaluate_implementation(self, X, scaling=False):
        if not self.initialized:
            script_dir = Path(__file__).parent
            self.P = np.loadtxt(script_dir / "input_data" / "FunctionRM_feed.txt")
            self.initialized = True
        P = self.P

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        f = np.zeros((n_samples, 1))

        # Objective function
        f = np.sum(X * np.tile(P[0, :], (n_samples, 1)), axis=1)

        # Inequality constraints
        g = np.zeros((n_samples, 14))
        g[:, 0] = -np.sum(X * np.tile(P[4, :], (n_samples, 1)), axis=1) + 1.090
        g[:, 1] = np.sum(X * np.tile(P[4, :], (n_samples, 1)), axis=1) - 2.170
        g[:, 2] = -np.sum(X * np.tile(P[3, :], (n_samples, 1)), axis=1) + 4.870
        g[:, 3] = np.sum(X * np.tile(P[3, :], (n_samples, 1)), axis=1) - 5.200
        g[:, 4] = -np.sum(X * np.tile(P[5, :], (n_samples, 1)), axis=1) + 0.043
        g[:, 5] = np.sum(X * np.tile(P[5, :], (n_samples, 1)), axis=1) - 0.086
        g[:, 6] = -np.sum(X * np.tile(P[6, :], (n_samples, 1)), axis=1) + 0.023
        g[:, 7] = np.sum(X * np.tile(P[6, :], (n_samples, 1)), axis=1) - 0.046
        g[:, 8] = -np.sum(X[:, :17], axis=1) / np.sum(X, axis=1) + 0.295
        g[:, 9] = np.sum(X[:, :17], axis=1) / np.sum(X, axis=1) - 0.36
        g[:, 10] = (
            -np.sum(X * np.tile(P[2, :], (n_samples, 1)), axis=1) / np.sum(X, axis=1)
            + 0.3
        )
        g[:, 11] = (
            np.sum(X * np.tile(P[2, :], (n_samples, 1)), axis=1) / np.sum(X, axis=1)
            - 0.4712
        )
        g[:, 12] = -np.sum(X[:, 33:59], axis=1) + 9.2
        g[:, 13] = np.sum(X[:, 33:59], axis=1) - 11.5

        # Equality constraints
        h = np.sum(X * np.tile(P[1, :], (n_samples, 1)), axis=1) - 6.9

        return (
            torch.from_numpy(np.abs(h) - 1e-4).unsqueeze(-1),
            torch.from_numpy(g).unsqueeze(-1),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p52(BenchmarkProblem):
    """
    CEC2020 Problem 52
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 59
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=59,
            num_objectives=1,
            num_constraints=1,
            #  X_opt=[[0] * 59],
            optimum=[3348.9821493],
            bounds=[(0.0, 10.0)] * 59,
        )
        self.initialized = False
        self.P = None

    def _evaluate_implementation(self, X, scaling=False):
        if not self.initialized:
            script_dir = Path(__file__).parent
            self.P = np.loadtxt(script_dir / "input_data" / "FunctionRM_feed.txt")
            self.initialized = True
        P = self.P

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        f = np.zeros((n_samples, 1))

        # Objective function
        f = np.sum(X * np.tile(P[0, :], (n_samples, 1)), axis=1)

        # Inequality constraints
        g = np.zeros((n_samples, 14))
        g[:, 0] = -np.sum(X * np.tile(P[4, :], (n_samples, 1)), axis=1) + 1.280
        g[:, 1] = np.sum(X * np.tile(P[4, :], (n_samples, 1)), axis=1) - 2.560
        g[:, 2] = -np.sum(X * np.tile(P[3, :], (n_samples, 1)), axis=1) + 7.300
        g[:, 3] = np.sum(X * np.tile(P[3, :], (n_samples, 1)), axis=1) - 7.810
        g[:, 4] = -np.sum(X * np.tile(P[5, :], (n_samples, 1)), axis=1) + 0.005
        g[:, 5] = np.sum(X * np.tile(P[5, :], (n_samples, 1)), axis=1) - 0.094
        g[:, 6] = -np.sum(X * np.tile(P[6, :], (n_samples, 1)), axis=1) + 0.031
        g[:, 7] = np.sum(X * np.tile(P[6, :], (n_samples, 1)), axis=1) - 0.062
        g[:, 8] = -np.sum(X[:, :17], axis=1) / np.sum(X, axis=1) + 0.2
        g[:, 9] = np.sum(X[:, :17], axis=1) / np.sum(X, axis=1) - 0.24
        g[:, 10] = (
            -np.sum(X * np.tile(P[2, :], (n_samples, 1)), axis=1) / np.sum(X, axis=1)
            + 0.3
        )
        g[:, 11] = (
            np.sum(X * np.tile(P[2, :], (n_samples, 1)), axis=1) / np.sum(X, axis=1)
            - 0.4
        )
        g[:, 12] = -np.sum(X[:, 33:59], axis=1) + 9.8
        g[:, 13] = np.sum(X[:, 33:59], axis=1) - 16.4

        # Equality constraints
        h = np.sum(X * np.tile(P[1, :], (n_samples, 1)), axis=1) - 9.8

        return (
            torch.from_numpy(np.abs(h) - 1e-4).unsqueeze(-1),
            torch.from_numpy(g).unsqueeze(-1),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p53(BenchmarkProblem):
    """
    CEC2020 Problem 53
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 59
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=59,
            num_objectives=1,
            num_constraints=1,
            #  X_opt=[[0] * 59],
            optimum=[4997.606929],
            bounds=[(0.0, 10.0)] * 59,
        )
        self.initialized = False
        self.P = None

    def _evaluate_implementation(self, X, scaling=False):
        if not self.initialized:
            script_dir = Path(__file__).parent
            self.P = np.loadtxt(script_dir / "input_data" / "FunctionRM_feed.txt")
            self.initialized = True
        P = self.P

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        f = np.zeros((n_samples, 1))

        # Objective function
        f = np.sum(X * np.tile(P[0, :], (n_samples, 1)), axis=1)

        # Inequality constraints
        g = np.zeros((n_samples, 14))

        g[:, 0] = -np.sum(X * np.tile(P[4, :], (n_samples, 1)), axis=1) + 1.170
        g[:, 1] = np.sum(X * np.tile(P[4, :], (n_samples, 1)), axis=1) - 2.340
        g[:, 2] = -np.sum(X * np.tile(P[3, :], (n_samples, 1)), axis=1) + 6.940
        g[:, 3] = np.sum(X * np.tile(P[3, :], (n_samples, 1)), axis=1) - 7.430
        g[:, 4] = -np.sum(X * np.tile(P[5, :], (n_samples, 1)), axis=1) + 0.038
        g[:, 5] = np.sum(X * np.tile(P[5, :], (n_samples, 1)), axis=1) - 0.076
        g[:, 6] = -np.sum(X * np.tile(P[6, :], (n_samples, 1)), axis=1) + 0.034
        g[:, 7] = np.sum(X * np.tile(P[6, :], (n_samples, 1)), axis=1) - 0.068
        g[:, 8] = -np.sum(X[:, :17], axis=1) / np.sum(X, axis=1) + 0.085
        g[:, 9] = np.sum(X[:, :17], axis=1) / np.sum(X, axis=1) - 0.111
        g[:, 10] = (
            -np.sum(X * np.tile(P[2, :], (n_samples, 1)), axis=1) / np.sum(X, axis=1)
            + 0.25
        )
        g[:, 11] = (
            np.sum(X * np.tile(P[2, :], (n_samples, 1)), axis=1) / np.sum(X, axis=1)
            - 0.4
        )
        g[:, 12] = -np.sum(X[:, 33:59], axis=1) + 11.6
        g[:, 13] = np.sum(X[:, 33:59], axis=1) - 14.5

        # Equality constraints
        h = np.sum(X * np.tile(P[1, :], (n_samples, 1)), axis=1) - 8.7

        return (
            torch.from_numpy(np.abs(h) - 1e-4).unsqueeze(-1),
            torch.from_numpy(g).unsqueeze(-1),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p54(BenchmarkProblem):
    """
    CEC2020 Problem 54
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 59
    num_constraints = 1

    def __init__(self):
        super().__init__(
            dim=59,
            num_objectives=1,
            num_constraints=1,
            #  X_opt=[[0] * 59],
            optimum=[4240.5482538],
            bounds=[(0.0, 10.0)] * 59,
        )
        self.initialized = False
        self.P = None

    def _evaluate_implementation(self, X, scaling=False):
        if not self.initialized:
            script_dir = Path(__file__).parent
            self.P = np.loadtxt(script_dir / "input_data" / "FunctionRM_feed.txt")
            self.initialized = True
        P = self.P

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        f = np.zeros((n_samples, 1))

        # Objective function
        f = np.sum(X * np.tile(P[0, :], (n_samples, 1)), axis=1)

        # Inequality constraints
        g = np.zeros((n_samples, 14))
        g[:, 0] = -np.sum(X * np.tile(P[4, :], (n_samples, 1)), axis=1) + 0.56
        g[:, 1] = np.sum(X * np.tile(P[4, :], (n_samples, 1)), axis=1) - 1.12
        g[:, 2] = -np.sum(X * np.tile(P[3, :], (n_samples, 1)), axis=1) + 3.23
        g[:, 3] = np.sum(X * np.tile(P[3, :], (n_samples, 1)), axis=1) - 3.46
        g[:, 4] = -np.sum(X * np.tile(P[5, :], (n_samples, 1)), axis=1) + 0.018
        g[:, 5] = np.sum(X * np.tile(P[5, :], (n_samples, 1)), axis=1) - 0.036
        g[:, 6] = -np.sum(X * np.tile(P[6, :], (n_samples, 1)), axis=1) + 0.0116
        g[:, 7] = np.sum(X * np.tile(P[6, :], (n_samples, 1)), axis=1) - 0.040
        g[:, 8] = -np.sum(X[:, :17], axis=1) / np.sum(X, axis=1) + 0.25
        g[:, 9] = np.sum(X[:, :17], axis=1) / np.sum(X, axis=1) - 0.9
        g[:, 10] = (
            -np.sum(X * np.tile(P[2, :], (n_samples, 1)), axis=1) / np.sum(X, axis=1)
            + 0.3
        )
        g[:, 11] = (
            np.sum(X * np.tile(P[2, :], (n_samples, 1)), axis=1) / np.sum(X, axis=1)
            - 0.4384
        )
        g[:, 12] = -np.sum(X[:, 33:59], axis=1) + 7.470
        g[:, 13] = np.sum(X[:, 33:59], axis=1) - 9.340

        # Equality constraints
        h = np.sum(X * np.tile(P[1, :], (n_samples, 1)), axis=1) - 5.6

        return (
            torch.from_numpy(np.abs(h) - 1e-4).unsqueeze(-1),
            torch.from_numpy(g).unsqueeze(-1),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p55(BenchmarkProblem):
    """
    CEC2020 Problem 55
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 64
    num_constraints = 6

    def __init__(self):
        super().__init__(
            dim=64,
            num_objectives=1,
            num_constraints=6,
            #  X_opt=[[0] * 64],
            optimum=[6696.4145128],
            bounds=[(0.0, 10.0)] * 64,
        )
        self.initialized = False
        self.P = None

    def _evaluate_implementation(self, X, scaling=False):
        if not self.initialized:
            script_dir = Path(__file__).parent
            self.P = np.loadtxt(script_dir / "input_data" / "FunctionRM_dairy.txt")
            self.initialized = True
        P = self.P

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        f = np.zeros((n_samples, 1))
        # Objective function
        f = np.sum(X * np.tile(P[0, :], (n_samples, 1)), axis=1)

        # Equality constraints
        h = np.zeros((n_samples, 6))
        h[:, 0] = np.sum(X * np.tile(P[11, :], (n_samples, 1)), axis=1) - 25.67
        h[:, 1] = np.sum(X * np.tile(P[1, :], (n_samples, 1)), axis=1) - 0.0218
        h[:, 2] = np.sum(X * np.tile(P[2, :], (n_samples, 1)), axis=1) - 0.062
        h[:, 3] = np.sum(X * np.tile(P[12, :], (n_samples, 1)), axis=1) - 0.034
        h[:, 4] = np.sum(X * np.tile(P[13, :], (n_samples, 1)), axis=1) - 0.021
        h[:, 5] = (
            np.sum(np.tile(np.sum(P[1:11, :], axis=0), (n_samples, 1)) * X, axis=1)
            - 0.999
        )

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p56(BenchmarkProblem):
    """
    CEC2020 Problem 56
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 64
    num_constraints = 6

    def __init__(self):
        super().__init__(
            dim=64,
            num_objectives=1,
            num_constraints=6,
            #  X_opt=[[0] * 64],
            optimum=[14746.58],
            bounds=[(0.0, 10.0)] * 64,
        )
        self.initialized = False
        self.P = None

    def _evaluate_implementation(self, X, scaling=False):
        if not self.initialized:
            script_dir = Path(__file__).parent
            self.P = np.loadtxt(script_dir / "input_data" / "FunctionRM_dairy.txt")
            self.initialized = True
        P = self.P

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        f = np.zeros((n_samples, 1))
        # Objective function
        f = np.sum(X * np.tile(P[0, :], (n_samples, 1)), axis=1)

        # Equality constraints
        h = np.zeros((n_samples, 6))
        h[:, 0] = np.sum(X * np.tile(P[11, :], (n_samples, 1)), axis=1) - 65.24
        h[:, 1] = np.sum(X * np.tile(P[1, :], (n_samples, 1)), axis=1) - 0.066
        h[:, 2] = np.sum(X * np.tile(P[2, :], (n_samples, 1)), axis=1) - 0.159
        h[:, 3] = np.sum(X * np.tile(P[12, :], (n_samples, 1)), axis=1) - 0.103
        h[:, 4] = np.sum(X * np.tile(P[13, :], (n_samples, 1)), axis=1) - 0.052
        h[:, 5] = (
            np.sum(np.tile(np.sum(P[1:11, :], axis=0), (n_samples, 1)) * X, axis=1)
            - 2.644
        )

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )


class CEC2020_p57(BenchmarkProblem):
    """
    CEC2020 Problem 57
    """

    num_objectives = 1
    input_type = DataType.CONTINUOUS
    available_dimensions = 64
    num_constraints = 6

    def __init__(self):
        super().__init__(
            dim=64,
            num_objectives=1,
            num_constraints=6,
            #  X_opt=[[0] * 64],
            optimum=[3213.2917019],
            bounds=[(0.0, 10.0)] * 64,
        )
        self.initialized = False
        self.P = None

    def _evaluate_implementation(self, X, scaling=False):
        if not self.initialized:
            script_dir = Path(__file__).parent
            self.P = np.loadtxt(script_dir / "input_data" / "FunctionRM_dairy.txt")
            self.initialized = True
        P = self.P

        if scaling:
            X = super().scale(X)
        X = X.numpy()

        n_samples = X.shape[0]

        f = np.zeros((n_samples, 1))
        # Objective function
        f = np.sum(X * np.tile(P[0, :], (n_samples, 1)), axis=1)

        # Equality constraints
        h = np.zeros((n_samples, 6))
        h[:, 0] = np.sum(X * np.tile(P[11, :], (n_samples, 1)), axis=1) - 30.05
        h[:, 1] = np.sum(X * np.tile(P[1, :], (n_samples, 1)), axis=1) - 0.0259
        h[:, 2] = np.sum(X * np.tile(P[2, :], (n_samples, 1)), axis=1) - 0.077
        h[:, 3] = np.sum(X * np.tile(P[12, :], (n_samples, 1)), axis=1) - 0.096
        h[:, 4] = np.sum(X * np.tile(P[13, :], (n_samples, 1)), axis=1) - 0.025
        h[:, 5] = (
            np.sum(np.tile(np.sum(P[1:11, :], axis=0), (n_samples, 1)) * X, axis=1)
            - 1.214
        )

        # No inequality constraints
        g = np.zeros((n_samples, 0))

        return (
            torch.from_numpy(np.abs(h) - 1e-4),
            torch.from_numpy(g),
            -torch.from_numpy(f).unsqueeze(-1),
        )
