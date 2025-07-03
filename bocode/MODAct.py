from typing import Tuple

import modact.problems as pb
import torch

from .base import BenchmarkProblem, DataType


class BaseModactProblem(BenchmarkProblem):
    input_type = DataType.CONTINUOUS

    def __init_subclass__(subcls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        subcls.problem_name = subcls.__name__.lower()
        subcls.problem = pb.get_problem(subcls.problem_name)
        subcls.available_dimensions = len(subcls.problem.bounds()[0])
        subcls.num_objectives = len(subcls.problem.weights)
        subcls.num_constraints = len(subcls.problem.c_weights)

    def __init__(self, optimum=None, x_opt=None):
        bounds = list(zip(*self.problem.bounds()))
        dim = len(self.problem.bounds()[0])
        num_obj = len(self.problem.weights)
        num_cons = len(self.problem.c_weights)

        super().__init__(
            dim=dim,
            num_objectives=num_obj,
            num_constraints=num_cons,
            bounds=bounds,
            x_opt=x_opt,
            optimum=optimum,
        )

    def _evaluate_implementation(
        self, X: torch.Tensor, scaling=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scaling:
            X = super().scale(X)

        X = X.numpy()

        fx = torch.zeros((X.shape[0], self.num_objectives))
        gx = torch.zeros((X.shape[0], self.num_constraints))

        for i in range(X.shape[0]):
            f, g = self.problem(X[i, :])
            fx[i, :], gx[i, :] = torch.tensor(f), torch.tensor(g)

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

        return gx, fx


class CS1(BaseModactProblem):
    pass


class CT1(BaseModactProblem):
    pass


class CTS1(BaseModactProblem):
    pass


class CTSE1(BaseModactProblem):
    pass


class CTSEI1(BaseModactProblem):
    pass


class CS2(BaseModactProblem):
    pass


class CT2(BaseModactProblem):
    pass


class CTS2(BaseModactProblem):
    pass


class CTSE2(BaseModactProblem):
    pass


class CTSEI2(BaseModactProblem):
    pass


class CS3(BaseModactProblem):
    pass


class CT3(BaseModactProblem):
    pass


class CTS3(BaseModactProblem):
    pass


class CTSE3(BaseModactProblem):
    pass


class CTSEI3(BaseModactProblem):
    pass


class CS4(BaseModactProblem):
    pass


class CT4(BaseModactProblem):
    pass


class CTS4(BaseModactProblem):
    pass


class CTSE4(BaseModactProblem):
    pass


class CTSEI4(BaseModactProblem):
    pass
