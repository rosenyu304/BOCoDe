import math

import torch

from ...base import BenchmarkProblem, DataType


class PressureVessel(BenchmarkProblem):
    """
    Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly algorithm.
    Computers & Structures 89(23-24):2325–2336
    """

    # 4D objective, 4 constraints, X = n-by-4

    available_dimensions = 4
    input_type = DataType.MIXED
    num_objectives = 1
    num_constraints = 4
    # x1, x2 (shell/head thickness, indices 0,1) are integer multiples of 0.0625
    # inch; x3, x4 (radius, length) are continuous.
    _GAUGE = [round(0.0625 * k, 4) for k in range(1, 100)]

    tags = {"single_objective", "constrained", "4D"}

    def __init__(self, is_discrete: bool = True):
        # is_discrete=True (default) is the firefly mixed-variable formulation
        # (thicknesses snap to the 0.0625" gauge grid); is_discrete=False is the
        # fully-continuous relaxation.
        self.is_discrete = is_discrete
        self.variable_types = (
            [self._GAUGE, self._GAUGE, "continuous", "continuous"]
            if is_discrete
            else None
        )
        if not is_discrete:
            self.input_type = DataType.CONTINUOUS
        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=4,
            bounds=[(99 * 0.0625, 0.0625), (99 * 0.0625, 0.0625), (200, 10), (200, 10)],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        n = X.size(0)

        C1, C2, C3, C4 = (0.6224, 1.7781, 3.1661, 19.84)

        gx = torch.zeros((n, self.num_constraints))

        Ts = X[:, 0]
        Th = X[:, 1]
        R = X[:, 2]
        L = X[:, 3]

        fx = -(C1 * Ts * R * L + C2 * Th * R * R + C3 * Ts * Ts * L + C4 * Ts * Ts * R)

        gx[:, 0] = -Ts + 0.0193 * R
        gx[:, 1] = -Th + 0.00954 * R
        gx[:, 2] = (
            (-1) * math.pi * R * R * L + (-1) * 4 / 3 * math.pi * R * R * R + 750 * 1728
        )
        gx[:, 3] = L - 240

        return gx, fx.reshape(n, 1)
