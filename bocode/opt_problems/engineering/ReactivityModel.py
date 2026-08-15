"""ReactivityModel — criticality targeting for the HOLOS-Quad microreactor (8-D).

The HOLOS-Quad high-temperature gas-cooled microreactor controls its reactivity with 8
cylindrical control drums. ``ReactivityModel.eval(thetas)`` returns the total reactivity
worth (rho) inserted by a given set of drum angles -- a SIGNED quantity, not a cost. The
design goal is not "as much reactivity as possible" but "exactly the reactivity that
makes the core critical". NEORL's example 11 calls this the criticality objective
``hatfc`` and MINIMIZES the distance to the target reactivity ``rho_tgt = 0.03308``::

    def hatfc(x):
        react = rm.eval(thetas)
        return np.abs(react - 0.03308)

BoCoDe maximizes, so ``_evaluate_implementation`` returns ``-hatfc(x)``. The best
attainable value is 0 (target reactivity hit exactly).

PREVIOUS BUG: this class returned the raw signed reactivity worth ``rm.eval(x)``. With
no target and no negation the optimization direction was undefined -- BO simply drove
the reactivity as high as it would go, which is not the problem NEORL poses (and, as a
reactor-control objective, is the opposite of what you want).

DIMENSION: NEORL's ex11 immobilizes one drum and optimizes the remaining 7. BoCoDe keeps
all 8 drum angles free, so this is the 8-D (no-malfunction) variant of the same
objective. Each angle is in [-pi, pi]. Unconstrained.

Sources:
NEORL example 11 (Microreactor Control with Malfunction): https://neorl.readthedocs.io/en/latest/examples/ex11.html
Upstream model: https://github.com/aims-umich/neorl/blob/master/neorl/benchmarks/reactivity_model.py
Price, D., Radaideh, M. I., & Kochunas, B. (2022). Multiobjective optimization of nuclear microreactor reactivity control system operation with swarm and evolutionary algorithms. Nuclear Engineering and Design, 393, 111776.
"""

import math

import torch

from ...base import BenchmarkProblem
from .._vendor.neorl_lib.reactivity_model import ReactivityModel as Reactivity_Model

# Target reactivity for a critical core (NEORL ex11: ``rho_tgt = 0.03308``).
_TARGET_REACTIVITY = 0.03308


class ReactivityModel(BenchmarkProblem):
    """Hit the target reactivity of an 8-drum microreactor (8 drum angles)."""

    available_dimensions = 8
    num_objectives = 1
    num_constraints = 0

    def __init__(self, typ: str = "wtd"):
        """
        typ can be "abs", "wtd" or "refl"
        """
        self.model = Reactivity_Model(typ)

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
        :returns: ``(None, fx)`` with ``fx[i] = -|rho(X[i]) - rho_tgt|``, the negated
            NEORL ``hatfc`` criticality objective (BoCoDe maximizes).
        """
        batch_size = X.shape[0]
        fx = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            angles = X[i].detach().cpu().numpy()
            reactivity = float(self.model.eval(angles))
            # BoCoDe maximizes; NEORL minimizes the distance to the target reactivity.
            fx[i] = -abs(reactivity - _TARGET_REACTIVITY)

        return None, fx.unsqueeze(-1)
