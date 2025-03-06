import torch
from ..base import *
import numpy as np

import cocoex

r'''
    N. Hansen, A. Auger, R. Ros, O. Mersmann, T. Tušar, D. Brockhoff. COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting, Optimization Methods and Software, 36(1), pp. 114-144, 2021.
'''

class BaseBBOB(BenchmarkProblem):

    def __init__(self,
                 dim,
                 suite,
                 function_number,
                 instance_number):
        problem = cocoex.Suite(suite, "", "").get_problem_by_function_dimension_instance(function_number, dim, instance_number)
        num_objectives = problem.number_of_objectives
        num_constraints = problem.number_of_constraints

        lower_bounds = torch.tensor(problem.lower_bounds)
        upper_bounds = torch.tensor(problem.upper_bounds)
        bounds = torch.cat( (lower_bounds.unsqueeze(-1), 
                             upper_bounds.unsqueeze(-1)) , dim=1)
        
        x_opt = torch.tensor(problem._best_parameter()) if problem._best_parameter() is not None else None
        optimum = torch.tensor(problem.best_observed_fvalue1) if problem.best_observed_fvalue1 is not None else None
        
        super().__init__(dim=dim,
                        num_objectives=num_objectives,
                        num_constraints=num_constraints,
                        bounds=bounds,
                        x_opt=x_opt,
                        optimum= optimum,
                        )
        
        self.problem = problem
        self.suite = suite
        self.function_number = function_number
        self.instance_number = instance_number

    def _evaluate_implementation(self, X: torch.Tensor, scaling=True):
        
        if scaling:
            X = super().scale(X)

        X_np = X.detach().cpu().numpy()
        fx = []
        gx = []

        fx = torch.tensor([self.problem(x) for x in X_np])

        if self.problem.number_of_constraints > 0:
            gx = torch.tensor([self.problem.constraint(x) for x in X_np])
        else:
            gx = None

        if self.num_objectives == 1:
            fx = fx.unsqueeze(-1)

        return gx, fx

class BBOB(BaseBBOB):

    def __init__(self, 
                 dim=2,
                 function_number=1,
                 instance_number=1,
                 ):
        suite = "bbob"
        super().__init__(dim,
                         suite,
                         function_number,
                         instance_number)


class BBOB_Biobj(BaseBBOB):

    def __init__(self, 
                 dim=2,
                 function_number=1,
                 instance_number=1,
                 ):
        suite = "bbob-biobj"
        super().__init__(dim,
                         suite,
                         function_number,
                         instance_number)

class BBOB_BiobjMixInt(BaseBBOB):

    def __init__(self, 
                 dim=2,
                 function_number=1,
                 instance_number=1,
                 ):
        suite = "bbob-biobj-mixint"
        super().__init__(dim,
                         suite,
                         function_number,
                         instance_number)

class BBOB_Boxed(BaseBBOB):

    def __init__(self, 
                 dim=2,
                 function_number=1,
                 instance_number=1,
                 ):
        suite = "bbob-boxed"
        super().__init__(dim,
                         suite,
                         function_number,
                         instance_number)

class BBOB_Constrained(BaseBBOB):

    def __init__(self, 
                 dim,
                 function_number,
                 instance_number=1,
                 ):
        suite = "bbob-constrained"
        super().__init__(dim,
                         suite,
                         function_number,
                         instance_number)

class BBOB_LargeScale(BaseBBOB):

    def __init__(self, 
                 dim,
                 function_number,
                 instance_number=1,
                 ):
        suite = "bbob-largescale"
        super().__init__(dim,
                         suite,
                         function_number,
                         instance_number)

class BBOB_MixInt(BaseBBOB):

    def __init__(self, 
                 dim,
                 function_number,
                 instance_number=1,
                 ):
        suite = "bbob-mixint"
        super().__init__(dim,
                         suite,
                         function_number,
                         instance_number)

class BBOB_Noisy(BaseBBOB):

    def __init__(self, 
                 dim,
                 function_number,
                 instance_number=1,
                 ):
        suite = "bbob-noisy"
        super().__init__(dim,
                         suite,
                         function_number,
                         instance_number)