import modact.problems as pb
from ..base import *
from typing import Tuple
import numpy as np

class BaseModactProblem(BenchmarkProblem):

    def __init__(self, problem: str, optimum=None, x_opt=None):

        self.problem = pb.get_problem(problem)

        bounds = list(zip(*self.problem.bounds()))
        dim = len(self.problem.bounds()[0])
        num_obj = len(self.problem.weights)
        num_cons = len(self.problem.c_weights)

        super().__init__(dim=dim, 
                         num_objectives=num_obj, 
                         num_constraints=num_cons, 
                         bounds=bounds, 
                         x_opt=x_opt, 
                         optimum=optimum)

    def _evaluate_implementation(self, X: torch.Tensor, scaling=True) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if scaling:
            X = super().scale(X)

        X = np.array(X)

        fx = np.zeros((X.shape[0], self.num_objectives))
        gx = np.zeros((X.shape[0], self.num_constraints))

        for i, w in enumerate(self.problem.weights):
            fx[i, :], gx[i, :] = self.problem(X[i, :])

        for i, w in enumerate(self.problem.weights):
            # Objective weights: -1 --> minimization / 1 --> maximization
            # Convert everything to minimization
            if w == 1:
                fx[:, i] = -fx[:, i]
        
        for i, w in enumerate(self.problem.c_weights):
            # Constraints weights: -1 --> g(x) >= 0 / 1 --> g(x) <= 0
            # Convert everything to g(x) <= 0
            if w == -1:
                gx[:, i] = -gx[:, i]

        return torch.Tensor(gx), torch.Tensor(fx)