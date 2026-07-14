"""QPowerModel — quadrant power flattening for the HOLOS-Quad microreactor (8-D).

The HOLOS-Quad high-temperature gas-cooled microreactor has 8 cylindrical control
drums. A surrogate (a small neural net, run here through ONNX Runtime) maps the 8 drum
angles to the fraction of total core power produced in each of the 4 core quadrants.
The design goal is a FLAT power distribution: every quadrant should carry 1/4 of the
power. NEORL's example 11 calls this the power objective ``hatfp`` and MINIMIZES it::

    def hatfp(x):
        powers = pm.eval(thetas)          # 4 quadrant power fractions, sum == 1
        targets = np.zeros(4) + 0.25
        return np.abs(powers - targets).sum()

BoCoDe maximizes, so ``_evaluate_implementation`` returns ``-hatfp(x)``. The best
attainable value is 0 (a perfectly flat quadrant power split).

PREVIOUS BUG: the vendored ``QPowerModel.eval`` had been reduced to
``float(unorm.sum())`` -- the sum of a *normalized* power distribution, which is 1.0 by
construction. The objective was therefore effectively CONSTANT (range 2.9e-3 over 256
random points) and optimizing it was meaningless. ``eval`` now returns the 4-vector, as
upstream NEORL does, and this class rebuilds the real load-balancing objective.

DIMENSION: NEORL's ex11 immobilizes one drum and optimizes the remaining 7. BoCoDe
keeps all 8 drum angles free, so this is the 8-D (no-malfunction) variant of the same
objective. Each angle is in [-pi, pi]. Unconstrained.

Sources:
NEORL example 11 (Microreactor Control with Malfunction): https://neorl.readthedocs.io/en/latest/examples/ex11.html
Upstream model: https://github.com/aims-umich/neorl/blob/master/neorl/benchmarks/qpower_model.py
Price, D., Radaideh, M. I., & Kochunas, B. (2022). Multiobjective optimization of nuclear microreactor reactivity control system operation with swarm and evolutionary algorithms. Nuclear Engineering and Design, 393, 111776.
"""

import math

import numpy as np
import torch

from ...base import BenchmarkProblem
from .._vendor.neorl_lib.qpower_model import QPowerModel as Q_Power_Model

# Perfectly flat split: each of the 4 quadrants carries 1/4 of the core power.
_TARGET_QUADRANT_POWER = 0.25


class QPowerModel(BenchmarkProblem):
    """Flatten the quadrant power split of an 8-drum microreactor (8 drum angles)."""

    available_dimensions = 8
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        self.model = Q_Power_Model()

        super().__init__(
            dim=8,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-math.pi, math.pi)] * 8,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param X: tensor of shape (batch_size, 8) containing drum angles in radians
        :returns: ``(None, fx)`` with ``fx[i] = -sum_q |P_q(X[i]) - 1/4|``, the negated
            NEORL ``hatfp`` power-flattening objective (BoCoDe maximizes).
        """
        batch_size = X.shape[0]
        fx = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            angles = X[i].detach().cpu().numpy()
            powers = np.asarray(self.model.eval(angles), dtype=float)
            imbalance = np.abs(powers - _TARGET_QUADRANT_POWER).sum()
            fx[i] = float(-imbalance)  # BoCoDe maximizes; NEORL minimizes hatfp

        return None, fx.unsqueeze(-1)
