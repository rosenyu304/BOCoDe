import torch
from ..base import *

class Ackley(BenchmarkProblem):

    r'''
    Sources:
    (1) https://www.sfu.ca/~ssurjano/ackley.html
    (2) Eriksson D, Poloczek M (2021) Scalable constrained bayesian optimization.
    In: International Conference on Artificial Intelligence and Statistics, PMLR, pp 730–738
    '''

    available_dimensions = (1,None)
    num_objectives = 1

    def __init__(self, 
                 dim: int = 2, 
                 CONSTRAINTS = ConstraintConfig(type='CONSTRAINTS'),
                 mute = False,
                 ):
        
        tags = ["Ackley",
                "-----------------------------",
                "OBJECTIVES: Single Objective (1)", 
                "CONSTRAINTS: Constrained (2)", 
                "SPACE: Continuous", 
                "SCALABLE: N-Dim", 
                "IMPORTS: BoTorch",
               ]
        
        super().__init__(dim, 
                         num_objectives = 1, 
                         num_constraints = 2, 
                         optimum = [[0]],
                         x_opt=[[0]*dim], 
                         bounds = [(-5, 10)]*dim,
                         CONSTRAINTS = CONSTRAINTS,
                         tags = tags,
                         mute = mute,
                        )
        
        if not self.mute:
            print(f"Function info:\n",
                f"Number of objectives: {self.num_objectives}\n",
                f"Number of constraints: {self.num_constraints}\n",
                f"Number of dimensions: {self.dim} (setable)\n",
                f"Bounds: {self.bounds}\n",
                )

    def _evaluate_implementation(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        from botorch.test_functions import Ackley as Ackley_imported
        
        n = X.size(0)

        gx = torch.zeros((n, self.num_constraints))

        fun = Ackley_imported(dim=self.dim, negate=False)

        fun.bounds = torch.tensor(self.bounds, dtype=torch.float32).T
        
        gx[:, 0] = torch.sum(X,1)
        gx[:, 1] = (torch.norm(X, p=2, dim=1)-5)

        return gx, fun(X).unsqueeze(1)
