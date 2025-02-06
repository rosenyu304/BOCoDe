import torch
from ..base import *

class Levy(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/levy.html
    '''

    def __init__(self, dim: int = 2):

        tags = ["Levy",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: N-Dim", 
                "IMPORTS: BoTorch",
               ]

        super().__init__(dim, 
                         num_objectives = 1, 
                         num_constraints = 0, 
                         bounds = [[-10, 10]]*dim,
                         tags = tags
                        )

    def _evaluate_implementation(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        from botorch.test_functions.synthetic import Levy as Levy_imported

        fun = Levy_imported(dim=self.dim, negate=True)

        fun.bounds[0, :] = torch.tensor([b[0] for b in self.bounds], dtype=torch.float32)
        fun.bounds[1, :] = torch.tensor([b[1] for b in self.bounds], dtype=torch.float32)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx
