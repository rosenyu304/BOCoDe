from typing import Tuple

import numpy as np
import torch

from ..base import BenchmarkProblem, DataType


class Beale(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/beale.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "Beale",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 2-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-4.5, 4.5)] * 2,
            optimum=[[0]],
            x_opt=[[3, 0.5]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import Beale as Beale_imported

        fun = Beale_imported(negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)


class Cosine8(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/beale.html
    """

    available_dimensions = 8
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "Cosine8",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 8-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=8,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-1.0, 1.0)] * 8,
            optimum=[[-0.8]],
            x_opt=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import Cosine8 as Cosine8_imported

        fun = Cosine8_imported(negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)


class DropWave(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/drop.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "DropWave",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 2-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-5.12, 5.12)] * 2,
            optimum=[[1]],
            x_opt=[[0, 0]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import DropWave as DropWave_imported

        fun = DropWave_imported(negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)


class EggHolder(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/egg.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "EggHolder",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 2-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-512, 512)] * 2,
            optimum=[[959.6407]],
            x_opt=[[512, 404.2319]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import EggHolder as EggHolder_imported

        fun = EggHolder_imported(negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)


class Hartmann3D(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/hart3.html
    """

    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "Hartmann",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 3-Dim",
            "IMPORTS: None",
        ]

        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0, 1)] * 3,
            optimum=[[-3.86278]],
            x_opt=[[0.114614, 0.555649, 0.852547]],
            tags=tags,
        )

    def hart3(self, X):
        # Parameters
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        P = 10 ** (-4) * np.array(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        )

        outer = 0

        for ii in range(4):
            inner = 0

            for jj in range(3):
                xj = X[:, jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]

                # Compute the inner sum
                inner += Aij * (xj - Pij) ** 2

            # Update the outer sum
            new = alpha[ii] * np.exp(-inner)
            outer += new

        # Return the negative of the outer sum
        y = -outer
        return y

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        return None, self.hart3(X).to(dtype=torch.float32).unsqueeze(-1)


class Hartmann6D(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/hart6.html
    """

    available_dimensions = 6
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "Hartmann",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 6-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=6,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0, 1)] * 6,
            optimum=[[3.32237]],
            x_opt=[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import Hartmann as Hartmann_imported

        fun = Hartmann_imported(dim=self.dim, negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)


class HolderTable(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/holder.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "HolderTable",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 2-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-10, 10)] * 2,
            optimum=[[19.2085]],
            x_opt=[[8.05502, 9.66459]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import HolderTable as HolderTable_imported

        fun = HolderTable_imported(negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)


class BaseShekel(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/shekel.html
    """

    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self, m: int, optimum):
        tags = [
            "Shekel",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 4-Dim",
            "IMPORTS: BoTorch",
        ]
        self.m = m
        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0, 10)] * 4,
            optimum=optimum,
            x_opt=[[4.0, 4.0, 4.0, 4.0]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import Shekel as Shekel_imported

        fun = Shekel_imported(m=self.m, negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)


class Shekelm5(BaseShekel):
    def __init__(self):
        super().__init__(m=5, optimum=[10.1532])


class Shekelm7(BaseShekel):
    def __init__(self):
        super().__init__(m=7, optimum=[10.4029])


class Shekelm10(BaseShekel):
    def __init__(self):
        super().__init__(m=10, optimum=[10.5364])


class Shekel(BaseShekel):
    def __init__(self, m):
        super().__init__(m)


class SixHumpCamel(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/camel6.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "SixHumpCamel",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 2-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-3, 3)] * 2,
            optimum=[[1.0316], [1.0316]],
            x_opt=[[0.0898, -0.7126], [-0.0898, 0.7126]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import (
            SixHumpCamel as SixHumpCamel_imported,
        )

        fun = SixHumpCamel_imported(negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)


class ThreeHumpCamel(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/camel3.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        tags = [
            "ThreeHumpCamel",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 2-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-5, 5)] * 2,
            optimum=[[0]],
            x_opt=[[0, 0]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import (
            ThreeHumpCamel as ThreeHumpCamel_imported,
        )

        fun = ThreeHumpCamel_imported(negate=True)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return None, fun(X).unsqueeze(-1)


class ConstrainedGramacy(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/camel3.html
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 2

    def __init__(self):
        tags = [
            "ConstrainedGramacy",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: 2",
            "SPACE: Continuous",
            "SCALABLE: 2-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=2,
            bounds=[(0, 1)] * 2,
            optimum=[[-0.5998]],
            x_opt=[[0.1954, 0.4044]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import (
            ConstrainedGramacy as ConstrainedGramacy_imported,
        )

        fun = ConstrainedGramacy_imported(negate=True)

        gx = fun.evaluate_slack(X)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return gx, fun(X).unsqueeze(-1)


class ConstrainedHartmann(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/hart6.html
    """

    available_dimensions = 6
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 1

    def __init__(self):
        tags = [
            "ConstrainedHartmann",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: 1",
            "SPACE: Continuous",
            "SCALABLE: 6-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=6,
            num_objectives=1,
            num_constraints=1,
            bounds=[(0, 1)] * 6,
            optimum=[[3.32237]],
            x_opt=[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import (
            ConstrainedHartmann as Hartmann_imported,
        )

        fun = Hartmann_imported(dim=self.dim, negate=True)

        gx = fun.evaluate_slack(X)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return gx, fun(X).unsqueeze(-1)


class ConstrainedHartmannSmooth(BenchmarkProblem):
    """
    https://www.sfu.ca/~ssurjano/hart6.html
    """

    available_dimensions = 6
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 1

    def __init__(self):
        tags = [
            "ConstrainedHartmannSmooth",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: 1",
            "SPACE: Continuous",
            "SCALABLE: 6-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=6,
            num_objectives=1,
            num_constraints=1,
            bounds=[(0, 1)] * 6,
            optimum=[[3.32237]],
            x_opt=[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import (
            ConstrainedHartmannSmooth as Hartmann_imported,
        )

        fun = Hartmann_imported(dim=self.dim, negate=True)

        gx = fun.evaluate_slack(X)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return gx, fun(X).unsqueeze(-1)


class PressureVessel(BenchmarkProblem):
    """ """

    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 4

    def __init__(self):
        tags = [
            "PressureVessel",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: 4",
            "SPACE: Continuous",
            "SCALABLE: 4-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=4,
            bounds=[(0, 10), (0, 10), (10, 50), (150, 200)],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import (
            PressureVessel as PressureVessel_imported,
        )

        fun = PressureVessel_imported(negate=True)

        gx = fun.evaluate_slack(X)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return gx, fun(X).unsqueeze(-1)


class WeldedBeamSO(BenchmarkProblem):
    """ """

    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 4

    def __init__(self):
        tags = [
            "WeldedBeam",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: 4",
            "SPACE: Continuous",
            "SCALABLE: 4-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=4,
            bounds=[(0.125, 2)] * 4,
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import WeldedBeamSO as WeldedBeam_imported

        fun = WeldedBeam_imported(negate=True)

        gx = fun.evaluate_slack(X)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return gx, fun(X).unsqueeze(-1)


class TensionCompressionString(BenchmarkProblem):
    """ """

    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 2

    def __init__(self):
        tags = [
            "TensionCompressionString",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: 2",
            "SPACE: Continuous",
            "SCALABLE: 3-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=2,
            bounds=[(0.01, 1), (0.01, 1), (0.01, 20)],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import (
            TensionCompressionString as TensionCompressionString_imported,
        )

        fun = TensionCompressionString_imported(negate=True)

        gx = fun.evaluate_slack(X)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return gx, fun(X).unsqueeze(-1)


class SpeedReducer(BenchmarkProblem):
    """ """

    available_dimensions = 7
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 11

    def __init__(self):
        tags = [
            "SpeedReducer",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: 11",
            "SPACE: Continuous",
            "SCALABLE: 7-Dim",
            "IMPORTS: BoTorch",
        ]

        super().__init__(
            dim=7,
            num_objectives=1,
            num_constraints=11,
            bounds=[
                (2.6, 3.6),
                (0.7, 0.8),
                (17, 28),
                (7.3, 8.3),
                (7.8, 8.3),
                (2.9, 3.9),
                (5, 5.5),
            ],
            tags=tags,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        from botorch.test_functions.synthetic import (
            SpeedReducer as SpeedReducer_imported,
        )

        fun = SpeedReducer_imported(negate=True)

        gx = fun.evaluate_slack(X)

        fun.bounds = self.torch_bounds.to(dtype=torch.float32).T

        return gx, fun(X).unsqueeze(-1)
