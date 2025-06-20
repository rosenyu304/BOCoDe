from ..base import *
import torch
import numpy as np
from typing import Tuple
import math
from .neorl_lib.reactivity_model import ReactivityModel

class Reactivity8Drums(BenchmarkProblem):
    """
    Eight-drum reactivity insertion problem.
    Uses the ReactivityModel.eval(pert) to compute the total reactivity worth
    of 8 control-drum angles (in radians).
    https://neorl.readthedocs.io/en/latest/examples/ex11.html#problem-description
    """

    available_dimensions = 8
    num_objectives = 1
    num_constraints = 0

    def __init__(self, typ: str = "wtd"):
        self.model = ReactivityModel(typ)

        super().__init__(
            dim=8,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-math.pi, math.pi)] * 8,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param X: tensor of shape (batch_size, 8) containing drum angles in radians
        :returns: (None, fx) where fx[i] = self.model.eval( X[i].numpy() )
        """
        batch_size = X.shape[0]
        fx = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            angles = X[i].detach().cpu().numpy()
            val = self.model.eval(angles)
            fx[i] = float(val)

        return None, fx.unsqueeze(-1)