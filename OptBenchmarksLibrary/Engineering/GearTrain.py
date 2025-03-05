import torch
from ..base import BenchmarkProblem

class GearTrain(BenchmarkProblem):

    r'''
    Sandgren, E. (1990). Nonlinear Integer and Discrete Programming in Mechanical Design Optimization."
    ASME. J. Mech. Des. June 1990; 112(2): 223–229.
    '''

    # 4D objective, 0 constraints, X = n-by-4

    tags = {"single_objective", "unconstrained", "mixed", "4D"}

    def __init__(self, is_mixed = True):

        self.is_mixed = is_mixed

        super().__init__(dim = 4, 
                         num_objectives = 1, 
                         num_constraints = 0, 
                         bounds = [(0, 1)]*4,
                        )

    def _evaluate_implementation(self, X):
        # X = super().scale(X, to_verify)

        def cont_to_disc(x, disc_values):
            # Convert continuous value to discrete value
            # Input:
            #   x: continuous value in [0, 1]
            #   disc_values: discrete values
            # Output: discrete value
            idx = torch.floor(x * len(disc_values)).long()
            return disc_values[torch.clamp(idx, 0, len(disc_values)-1)]

        if self.is_mixed:
            X = cont_to_disc(X, torch.tensor(range(12, 61))) # x0, x1, x2, x3: {12, 13, ..., 60}

        fx = -((1/6.931 - (X[:,0]*X[:,1])/(X[:,2]*X[:,3]))**2).reshape(-1, 1)

        return None, fx



