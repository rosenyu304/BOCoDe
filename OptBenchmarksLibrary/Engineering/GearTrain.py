import torch
from .base import BenchmarkProblem

class GearTrain(BenchmarkProblem):

    r'''
    Sandgren, E. (1990). Nonlinear Integer and Discrete Programming in Mechanical Design Optimization."
    ASME. J. Mech. Des. June 1990; 112(2): 223–229.
    '''

    # 4D objective, 0 constraints, X = n-by-4

    tags = {"single_objective", "unconstrained", "mixed", "4D"}

    def __init__(self, is_mixed = True, to_print_Xscaled = False):
        super().__init__(dim = 4, 
                         num_obj = 1, 
                         num_cons = 0, 
                         bounds = [[0, 1]],
                         is_mixed = is_mixed,
                         to_print_Xscaled = to_print_Xscaled,
                        )

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        if self.is_mixed:
            X = super().cont_to_disc(X, torch.tensor(range(12, 61))) # x0, x1, x2, x3: {12, 13, ..., 60}

        if self.to_print_Xscaled:
            print(f'X: {X}')

        fx = -((1/6.931 - (X[:,0]*X[:,1])/(X[:,2]*X[:,3]))**2).reshape(-1, 1)

        return None, fx



