# import torch
# from ..base import *
# import numpy as np
# from cec2019comp100digit import cec2019comp100digit

# """
# CEC 2019 Benchmark Functions
# K. V. Price, N. H. Awad, M. Z. Ali, P. N. Suganthan, "Problem Definitions and Evaluation Criteria for the 100-Digit Challenge Special Session and Competition on Single Objective Numerical Optimization," Technical Report, Nanyang Technological University, Singapore, November 2018.

# +-----+--------------------------------------------------------------+----------------+-----+------------------+
# | No. | Functions                                                    | Fi* = Fi(x*)   | D   | Search Range     |
# +-----+--------------------------------------------------------------+----------------+-----+------------------+
# |  1  | Storn's Chebyshev Polynomial Fitting Problem                 |        1       |  9  | [-8192, 8192]    |
# |  2  | Inverse Hilbert Matrix Problem                               |        1       | 16  | [-16384, 16384]  |
# |  3  | Lennard-Jones Minimum Energy Cluster                         |        1       | 18  | [-4, 4]          |
# |  4  | Rastrigin’s Function                                         |        1       | 10  | [-100, 100]      |
# |  5  | Griewangk’s Function                                         |        1       | 10  | [-100, 100]      |
# |  6  | Weierstrass Function                                         |        1       | 10  | [-100, 100]      |
# |  7  | Modified Schwefel’s Function                                 |        1       | 10  | [-100, 100]      |
# |  8  | Expanded Schaffer’s F6 Function                              |        1       | 10  | [-100, 100]      |
# |  9  | Happy Cat Function                                           |        1       | 10  | [-100, 100]      |
# | 10  | Ackley Function                                              |        1       | 10  | [-100, 100]      |
# +-----+--------------------------------------------------------------+----------------+-----+------------------+
# """

# class BaseCEC2019(BenchmarkProblem):

#     available_dimensions = None
#     num_objectives = 1

#     CEC2019Problem = None

#     def __init__(self, dim: int, bounds):

#         self.problem = cec2019comp100digit
#         self.problem.init(self.__class__.CEC2019Problem, dim)

#         super().__init__(dim = dim, 
#                          num_objectives = 1, 
#                          num_constraints = 0, 
#                          bounds = bounds,
#                          )

#     def _evaluate_implementation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

#         fx = torch.zeros(x.shape[0], self.num_objectives)
        
#         for i, el in enumerate(x):
#             f = self.problem.eval(el.double().numpy())
#             fx[i, :] = torch.tensor(f)

#         return None, torch.tensor(fx)

# class CEC2019_1(BaseCEC2019):
#     """
#     Storn's Chebyshev Polynomial Fitting Problem
#     """

#     CEC2019Problem = 1
#     available_dimensions = 9

#     def __init__(self):

#         super().__init__(dim = 9, bounds = [(-8192, 8192)]*9)

# class CEC2019_2(BaseCEC2019):
#     """
#     Inverse Hilbert Matrix Problem
#     """

#     CEC2019Problem = 2
#     available_dimensions = 16

#     def __init__(self):

#         super().__init__(dim = 16, bounds = [(-16384, 16384)]*16)

# class CEC2019_3(BaseCEC2019):
#     """
#     Lennard-Jones Minimum Energy Cluster
#     """

#     CEC2019Problem = 3
#     available_dimensions = 18

#     def __init__(self):

#         super().__init__(dim = 18, bounds = [(-4, 4)]*18)

# class CEC2019_4(BaseCEC2019):
#     """
#     Rastrigin’s Function
#     """

#     CEC2019Problem = 4
#     available_dimensions = 10

#     def __init__(self):

#         super().__init__(dim = 10, bounds = [(-100, 100)]*10)

# class CEC2019_5(BaseCEC2019):
#     """
#     Griewangk’s Function
#     """

#     CEC2019Problem = 5
#     available_dimensions = 10

#     def __init__(self):

#         super().__init__(dim = 10, bounds = [(-100, 100)]*10)

# class CEC2019_6(BaseCEC2019):
#     """
#     Weierstrass Function
#     """

#     CEC2019Problem = 6
#     available_dimensions = 10

#     def __init__(self):

#         super().__init__(dim = 10, bounds = [(-100, 100)]*10)

# class CEC2019_7(BaseCEC2019):
#     """
#     Modified Schwefel’s Function
#     """

#     CEC2019Problem = 7
#     available_dimensions = 10

#     def __init__(self):

#         super().__init__(dim = 10, bounds = [(-100, 100)]*10)

# class CEC2019_8(BaseCEC2019):
#     """
#     Expanded Schaffer’s F6 Function
#     """

#     CEC2019Problem = 8
#     available_dimensions = 10

#     def __init__(self):

#         super().__init__(dim = 10, bounds = [(-100, 100)]*10)

# class CEC2019_9(BaseCEC2019):
#     """
#     Happy Cat Function
#     """

#     CEC2019Problem = 9
#     available_dimensions = 10

#     def __init__(self):

#         super().__init__(dim = 10, bounds = [(-100, 100)]*10)

# class CEC2019_10(BaseCEC2019):
#     """
#     Ackley Function
#     """

#     CEC2019Problem = 10
#     available_dimensions = 10

#     def __init__(self):

#         super().__init__(dim = 10, bounds = [(-100, 100)]*10)