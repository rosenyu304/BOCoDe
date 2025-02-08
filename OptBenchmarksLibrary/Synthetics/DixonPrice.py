import torch
from ..base import *

class DixonPrice(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/dixonpr.html
    '''

    def __init__(self, dim: int = 2):
        
        tags = ["DixonPrice",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: N/A", 
                "SPACE: Continuous", 
                "SCALABLE: N-Dim", 
                "IMPORTS: BoTorch",
               ]
        
        x_opt = torch.tensor([[2**(-(2**i - 2) / 2**i) for i in range(1, dim + 1)]], dtype=torch.float32).tolist()
        
        super().__init__(dim, 
                         num_objectives = 1, 
                         num_constraints = 0, 
                         bounds = [(-10, 10)]*dim,
                         optimum = [[0]],
                         x_opt = x_opt,
                         tags = tags
                        )
        
    def _evaluate_implementation(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        from botorch.test_functions.synthetic import DixonPrice as DixonPrice_imported

        fun = DixonPrice_imported(dim=self.dim, negate=True)

        fun.bounds[0, :] = torch.tensor([b[0] for b in self.bounds], dtype=torch.float32)
        fun.bounds[1, :] = torch.tensor([b[1] for b in self.bounds], dtype=torch.float32)

        n = X.size(0)

        fx = fun(X)
        fx = fx.reshape((n, 1))

        return None, fx
