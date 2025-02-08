import torch
from ..base import *

class Rosenbrock(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/rosen.html
    '''

    def __init__(self, dim: int = 2):

        tags = ["Rosenbrock",
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
                         bounds = [(-5, 10)]*dim,
                         optimum = [[0]],
                         x_opt = [[1]*dim],
                         tags = tags
                        )

    def _evaluate_implementation(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        from botorch.test_functions.synthetic import Rosenbrock as Rosenbrock_imported

        fun = Rosenbrock_imported(dim=self.dim, negate=True)

        fun.bounds[0, :] = torch.tensor([b[0] for b in self.bounds], dtype=torch.float32)
        fun.bounds[1, :] = torch.tensor([b[1] for b in self.bounds], dtype=torch.float32)

        # Previous method of setting bounds:
        # fun.bounds[0, :].fill_(self.bounds[0][0])
        # fun.bounds[1, :].fill_(self.bounds[0][1])

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx
