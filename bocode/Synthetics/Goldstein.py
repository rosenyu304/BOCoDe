from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


class Goldstein(BenchmarkProblem):
    """
    (Alt Name: Goldstein-Price)
    LVGP paper: https://www.nature.com/articles/s41598-020-60652-9
    """

    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        bounds, tags, optimum, x_opt = self._get_defaults()

        super().__init__(
            dim=2,
            num_objectives=1,
            num_constraints=0,
            optimum=optimum,
            x_opt=x_opt,
            bounds=bounds,
            tags=tags,
        )

    # Returns default values for the generic continuous problem. Can be overridden by subclasses.
    def _get_defaults(self):
        tags = [
            "Goldstein",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Continuous",
            "SCALABLE: 2-Dim",
            "IMPORTS: N/A",
        ]
        # bounds, tags, optimum, x_opt
        return ([(-2, 2), (-2, 2)], tags, [-3], [[0, -1]])

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x0: [-2, 2]
        # x1: {-2, -1, 0, 1, 2}

        fx = -(
            (
                1
                + (X[:, 0] + X[:, 1] + 1) ** 2
                * (
                    19
                    - 14 * X[:, 0]
                    + 3 * X[:, 0] ** 2
                    - 14 * X[:, 1]
                    + 6 * X[:, 0] * X[:, 1]
                    + 3 * X[:, 1] ** 2
                )
            )
            * (
                30
                + (2 * X[:, 0] - 3 * X[:, 1]) ** 2
                * (
                    18
                    - 32 * X[:, 0]
                    + 12 * X[:, 0] ** 2
                    + 48 * X[:, 1]
                    - 36 * X[:, 0] * X[:, 1]
                    + 27 * X[:, 1] ** 2
                )
            )
        )

        n = X.size(0)
        fx = fx.reshape((n, 1))

        return None, fx


class Goldstein_Discrete(Goldstein):
    """
    (Alt Name: Goldstein-Price)
    LVGP paper: https://www.nature.com/articles/s41598-020-60652-9
    """

    input_type = DataType.MIXED

    def __init__(self):
        super().__init__()

    def _get_defaults(self):
        tags = [
            "Goldstein Discrete",
            "-----------------------------",
            "OBJECTIVES: Single Objective (1)",
            "CONSTRAINTS: N/A",
            "SPACE: Mixed (Continuous/Discrete)",
            "SCALABLE: 2-Dim",
            "IMPORTS: N/A",
        ]
        return ([(-2, 2), {-2, -1, 0, 1, 2}], tags, None, None)

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x0: [-2, 2]
        # x1: {-2, -1, 0, 1, 2}

        # def cont_to_disc(x, disc_values):
        #     # Convert continuous value to discrete value
        #     # Input:
        #     #   x: continuous value in [0, 1]
        #     #   disc_values: discrete values
        #     # Output: discrete value
        #     idx = torch.floor(x * len(disc_values)).long()
        #     return disc_values[torch.clamp(idx, 0, len(disc_values)-1)]

        # # Second column is discrete
        # sorted_disc_values = torch.tensor(list(self.bounds[1]))
        # sorted_disc_values.sort()

        # X[:,1] = cont_to_disc(X[:,1], sorted_disc_values)

        fx = -(
            (
                1
                + (X[:, 0] + X[:, 1] + 1) ** 2
                * (
                    19
                    - 14 * X[:, 0]
                    + 3 * X[:, 0] ** 2
                    - 14 * X[:, 1]
                    + 6 * X[:, 0] * X[:, 1]
                    + 3 * X[:, 1] ** 2
                )
            )
            * (
                30
                + (2 * X[:, 0] - 3 * X[:, 1]) ** 2
                * (
                    18
                    - 32 * X[:, 0]
                    + 12 * X[:, 0] ** 2
                    + 48 * X[:, 1]
                    - 36 * X[:, 0] * X[:, 1]
                    + 27 * X[:, 1] ** 2
                )
            )
        )

        return None, fx.unsqueeze(1)
