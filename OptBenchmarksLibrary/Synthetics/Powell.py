import torch
from ..base import *

class Powell(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/powell.html
    '''

    def __init__(self, dim: int = 4):

        tags = ["Powell",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: N-Dim (at least 4)", 
                "IMPORTS: BoTorch",
               ]
        
        if dim < 4:
            raise ValueError("Powell function is only defined for n >= 4")
        
        super().__init__(dim, 
                         num_objectives = 1, 
                         num_constraints = 0, 
                         bounds = [(-4, 5)]*dim,
                         optimum = [[0]],
                         x_opt = [[0]*dim],
                         tags = tags
                        )

    def _evaluate_implementation(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        from botorch.test_functions.synthetic import Powell as Powell_imported

        fun = Powell_imported(dim=self.dim, negate=True)

        fun.bounds[0, :] = torch.tensor([b[0] for b in self.bounds], dtype=torch.float32)
        fun.bounds[1, :] = torch.tensor([b[1] for b in self.bounds], dtype=torch.float32)

        # Previous method of setting bounds:
        # fun.bounds[0, :].fill_(-4)
        # fun.bounds[1, :].fill_(5)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx
    