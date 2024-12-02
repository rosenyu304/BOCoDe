import torch
from base import BenchmarkProblem

class StyblinskiTang_mixed(BenchmarkProblem):

    r'''

    '''

    # 10D objective, 0 constraints, X = n-by-10

    tags = {"single_objective", "unconstrained", "mixed", "10D"}

    def __init__(self):
        super().__init__(dim = 10, num_obj = 1, num_cons = 0, bounds = [[0, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        def cont_to_disc(x, disc_values):
            # Convert continuous value to discrete value
            # Input:
            #   x: continuous value in [0, 1]
            #   disc_values: discrete values
            # Output: discrete value
            idx = torch.floor(x * len(disc_values)).long()
            return disc_values[torch.clamp(idx, 0, len(disc_values)-1)]

        # X: {-5, -2.5, 0, 2.5, 5}^dim
        X = cont_to_disc(X, torch.tensor([-5, -2.5, 0, 2.5, 5]))

        from botorch.test_functions.synthetic import StyblinskiTang as StyblinskiTang_imported

        return None, -StyblinskiTang_imported(X).view(-1, 1)
