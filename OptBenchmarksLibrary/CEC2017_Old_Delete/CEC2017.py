# import os
# import math
# import torch
# from scipy.io import loadmat
# from typing import Tuple
# from ..base import BenchmarkProblem

# """
# CEC 2017 Benchmark Functions

# Sources:
# (1) G. Wu, R. Mallipeddi, and P. N. Suganthan. Problem definitions and evaluation criteria for the CEC 2017 competition on constrained real-parameter optimization. National University of Defense Technology, China, 2016.
# (2) Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform for evolutionary multi-objective optimization [educational forum], IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87
# """

# _mat_data = loadmat(os.path.join(os.path.dirname(__file__), 'CEC2017.mat'))['Data']

# class CEC2017_p1(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         O = _mat_data[0][0].flatten()
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=1,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         Z = X - self.O[: X.size(1)]
#         fx = torch.sum(torch.cumsum(Z, dim=1) ** 2, dim=1, keepdim=True)
#         gx = torch.sum(Z ** 2 - 5000 * torch.cos(0.1 * math.pi * Z) - 4000, dim=1, keepdim=True)
#         return gx, fx


# class CEC2017_p2(BenchmarkProblem):
#     available_dimensions = [10, 30, 50, 100]
#     num_objectives = 1

#     def __init__(self):
#         data2 = _mat_data[0][1]
#         O = data2['o'].flatten()[0][0]
#         # pick appropriate D & rotation matrix
#         requested = getattr(self, 'D', None)
#         if requested is None or requested < 30:
#             D, M = 10, data2['M_10'][0][0]
#         elif requested < 50:
#             D, M = 30, data2['M_30'][0][0]
#         elif requested < 100:
#             D, M = 50, data2['M_50'][0][0]
#         else:
#             D, M = 100, data2['M_100'][0][0]
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=1,
#             # optimum=[[0.0]],
#             # x_opt=[O[:D].tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)
#         self.Mat = torch.tensor(M, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z = X - self.O[: X.size(1)]
#         fx = torch.sum(torch.cumsum(Z, dim=1) ** 2, dim=1, keepdim=True)
#         Y = Z @ self.Mat.T
#         gx = torch.sum(Y ** 2 - 5000 * torch.cos(0.1 * math.pi * Y) - 4000, dim=1, keepdim=True)
#         return gx, fx


# class CEC2017_p3(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         O = _mat_data[0][2].flatten()
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z = X - self.O[: X.size(1)]
#         fx = torch.sum(torch.cumsum(Z, dim=1) ** 2, dim=1, keepdim=True)
#         c1 = torch.sum(Z ** 2 - 5000 * torch.cos(0.1 * math.pi * Z) - 4000, dim=1, keepdim=True)
#         c2 = torch.abs(torch.sum(Z * torch.sin(0.1 * math.pi * Z), dim=1, keepdim=True)) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p4(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         O = _mat_data[0][3].flatten()
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-10.0, 10.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z = X - self.O[: X.size(1)]
#         fx = torch.sum(Z ** 2 - 10 * torch.cos(2 * math.pi * Z) + 10, dim=1, keepdim=True)
#         c1 = -torch.sum(Z * torch.sin(2 * Z), dim=1, keepdim=True)
#         c2 = torch.sum(Z * torch.sin(Z), dim=1, keepdim=True)
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p5(BenchmarkProblem):
#     available_dimensions = [10, 30, 50, 100]
#     num_objectives = 1

#     def __init__(self):
#         data5 = _mat_data[0][4]
#         O = data5['o'].flatten()[0][0]
#         requested = getattr(self, 'D', None)
#         if requested is None or requested < 30:
#             D, M1, M2 = 10, data5['M1_10'][0][0], data5['M2_10'][0][0]
#         elif requested < 50:
#             D, M1, M2 = 30, data5['M1_30'][0][0], data5['M2_30'][0][0]
#         elif requested < 100:
#             D, M1, M2 = 50, data5['M1_50'][0][0], data5['M2_50'][0][0]
#         else:
#             D, M1, M2 = 100, data5['M1_100'][0][0], data5['M2_100'][0][0]
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             # optimum=[[0.0]],
#             # x_opt=[O[:D].tolist()],
#             bounds=[(-10.0, 10.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)
#         self.Mat1 = torch.tensor(M1, dtype=torch.float32)
#         self.Mat2 = torch.tensor(M2, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z = X - self.O[: X.size(1)]
#         fx = torch.sum(100 * (Z[:, :-1] ** 2 - Z[:, 1:]) ** 2 + (Z[:, :-1] - 1) ** 2, dim=1, keepdim=True)
#         Y = Z @ self.Mat1.T
#         W = Z @ self.Mat2.T
#         c1 = torch.sum(Y ** 2 - 50 * torch.cos(2 * math.pi * Y) - 40, dim=1, keepdim=True)
#         c2 = torch.sum(W ** 2 - 50 * torch.cos(2 * math.pi * W) - 40, dim=1, keepdim=True)
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p6(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         O = _mat_data[0][5].flatten()
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=5,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-20.0, 20.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z = X - self.O[: X.size(1)]
#         fx = torch.sum(Z ** 2 - 10 * torch.cos(2 * math.pi * Z) + 10, dim=1, keepdim=True)
#         c = []
#         c.append(torch.abs(torch.sum(Z * torch.sin(Z), dim=1, keepdim=True)) - 1e-4)
#         c.append(torch.abs(torch.sum(Z * torch.sin(math.pi * Z), dim=1, keepdim=True)) - 1e-4)
#         c.append(torch.abs(torch.sum(Z * torch.cos(Z), dim=1, keepdim=True)) - 1e-4)
#         c.append(torch.abs(torch.sum(Z * torch.cos(math.pi * Z), dim=1, keepdim=True)) - 1e-4)
#         c.append(torch.abs(torch.sum(Z * torch.sin(2 * torch.sqrt(torch.abs(Z))), dim=1, keepdim=True)) - 1e-4)
#         gx = torch.cat(c, dim=1)
#         return gx, fx


# class CEC2017_p7(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         O = _mat_data[0][6].flatten()
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=1,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-50.0, 50.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z = X - self.O[: X.size(1)]
#         fx = torch.sum(Z * torch.sin(Z), dim=1, keepdim=True)
#         gx = torch.abs(torch.sum(Z - 100 * torch.cos(0.5 * Z) + 100, dim=1, keepdim=True)) - 1e-4
#         return gx, fx


# class CEC2017_p8(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         O = _mat_data[0][7].flatten()
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z = X - self.O[: X.size(1)]
#         fx = torch.max(Z, dim=1, keepdim=True)[0]
#         Y = Z[:, ::2]
#         W = Z[:, 1::2]
#         c1 = torch.abs(torch.sum(torch.cumsum(Y, dim=1) ** 2, dim=1, keepdim=True)) - 1e-4
#         c2 = torch.abs(torch.sum(torch.cumsum(W, dim=1) ** 2, dim=1, keepdim=True)) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p9(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         O = _mat_data[0][8].flatten()
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-10.0, 10.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z = X - self.O[: X.size(1)]
#         fx = torch.max(Z, dim=1, keepdim=True)[0]
#         Y = Z[:, ::2]
#         W = Z[:, 1::2]
#         c1 = torch.prod(W, dim=1, keepdim=True)
#         c2 = torch.abs(torch.sum((Y[:, :-1] ** 2 - Y[:, 1:]) ** 2, dim=1, keepdim=True)) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p10(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         O = _mat_data[0][9].flatten()
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z = X - self.O[: X.size(1)]
#         fx = torch.max(Z, dim=1, keepdim=True)[0]
#         c1 = torch.abs(torch.sum(torch.cumsum(Z, dim=1) ** 2, dim=1, keepdim=True)) - 1e-4
#         c2 = torch.abs(torch.sum((Z[:, :-1] - Z[:, 1:]) ** 2, dim=1, keepdim=True)) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p11(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         O = _mat_data[0][10].flatten()
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z = X - self.O[: X.size(1)]
#         fx = torch.sum(Z, dim=1, keepdim=True)
#         c1 = torch.prod(Z, dim=1, keepdim=True)
#         c2 = torch.abs(torch.sum((Z[:, :-1] - Z[:, 1:]) ** 2, dim=1, keepdim=True)) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p12(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Y = X - self.O[: X.size(1)]
#         fx = torch.sum(Y ** 2 - 10 * torch.cos(2 * math.pi * Y) + 10, dim=1, keepdim=True)
#         c1 = 4.0 - torch.sum(torch.abs(Y), dim=1, keepdim=True)
#         c2 = torch.sum(Y ** 2, dim=1, keepdim=True) - 4.0
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p13(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         # Note: F13 reuses Data{12}.o
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=3,
#             # optimum=[[0.0]],
#             # x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Y = X - self.O[: X.size(1)]
#         fx = torch.sum(100 * (Y[:, :-1] ** 2 - Y[:, 1:]) ** 2 + (Y[:, :-1] - 1) ** 2, dim=1, keepdim=True)
#         c1 = torch.sum(Y ** 2 - 10 * torch.cos(2 * math.pi * Y) + 10, dim=1, keepdim=True) - 100.0
#         c2 = torch.sum(Y, dim=1, keepdim=True) - 2.0 * Y.size(1)
#         c3 = 5.0 - torch.sum(Y, dim=1, keepdim=True)
#         gx = torch.cat([c1, c2, c3], dim=1)
#         return gx, fx


# class CEC2017_p14(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         # Note: F14 also reuses Data{12}.o
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Y = X - self.O[: X.size(1)]
#         fx = (
#             -20.0 * torch.exp(-0.2 * torch.sqrt(torch.mean(Y ** 2, dim=1)))
#             + 20.0
#             - torch.exp(torch.mean(torch.cos(2 * math.pi * Y), dim=1))
#             + math.e
#         ).unsqueeze(1)
#         c1 = torch.sum(Y[:, 1:] ** 2, dim=1, keepdim=True) + 1.0 - torch.abs(Y[:, :1])
#         c2 = torch.abs(torch.sum(Y ** 2, dim=1, keepdim=True) - 4.0) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx

# class CEC2017_p15(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Y = X - self.O[: X.size(1)]
#         fx = torch.max(torch.abs(Y), dim=1, keepdim=True)[0]
#         c1 = torch.sum(Y ** 2, dim=1, keepdim=True) - 100.0 * Y.size(1)
#         c2 = torch.abs(torch.cos(fx) + torch.sin(fx)) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p16(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Y = X - self.O[: X.size(1)]
#         fx = torch.sum(torch.abs(Y), dim=1, keepdim=True)
#         c1 = torch.sum(Y ** 2, dim=1, keepdim=True) - 100.0 * Y.size(1)
#         tmp = fx
#         c2 = torch.abs((torch.cos(tmp) + torch.sin(tmp)) ** 2 - torch.exp(torch.cos(tmp) + torch.sin(tmp)) - 1 + math.e) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p17(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)
#         # precompute denominator sqrt(1..D)
#         idx = torch.arange(1, D + 1, dtype=torch.float32)
#         self.sqrt_idx = torch.sqrt(idx)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Y = X - self.O[: X.size(1)]
#         scaled = Y / self.sqrt_idx[: X.size(1)]
#         fx = (0.25 * torch.sum(Y ** 2, dim=1, keepdim=True) / 1000.0
#               + 1.0
#               - torch.prod(torch.cos(scaled), dim=1, keepdim=True))
#         # constraints
#         sumY2 = torch.sum(Y ** 2, dim=1, keepdim=True)
#         c1 = 1.0 - torch.sum(torch.sign(torch.abs(Y) - sumY2 + Y ** 2 - 1.0), dim=1, keepdim=True)
#         c2 = torch.abs(sumY2 - 4.0 * Y.size(1)) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p18(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=3,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Y = X - self.O[: X.size(1)]
#         Z = torch.where(
#             torch.abs(Y) < 0.5,
#             Y,
#             0.5 * torch.round(2.0 * Y)
#         )
#         fx = torch.sum(Z ** 2 - 10 * torch.cos(2 * math.pi * Z) + 10, dim=1, keepdim=True)
#         c1 = 1.0 - torch.sum(torch.abs(Y), dim=1, keepdim=True)
#         c2 = torch.sum(Y ** 2, dim=1, keepdim=True) - 100.0 * Y.size(1)
#         term = torch.sum(100 * (Y[:, :-1] ** 2 - Y[:, 1:]) ** 2, dim=1, keepdim=True)
#         term += torch.prod(torch.sin(math.pi * (Y - 1.0)) ** 2, dim=1, keepdim=True)
#         c3 = torch.abs(term) - 1e-4
#         gx = torch.cat([c1, c2, c3], dim=1)
#         return gx, fx


# class CEC2017_p19(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-50.0, 50.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Y = X - self.O[: X.size(1)]
#         fx = torch.sum(torch.sqrt(torch.abs(Y)) + 2 * torch.sin(Y ** 3), dim=1, keepdim=True)
#         # c1
#         A = Y[:, :-1] ** 2 + Y[:, 1:] ** 2
#         c1 = torch.sum(-10 * torch.exp(-0.2 * torch.sqrt(A)), dim=1, keepdim=True)
#         c1 += (Y.size(1) - 1) * 10.0 / math.exp(-5.0)
#         # c2
#         c2 = torch.sum(torch.sin(2.0 * Y) ** 2, dim=1, keepdim=True) - 0.5 * Y.size(1)
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p20(BenchmarkProblem):
#     available_dimensions = (10, None)
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D = len(O)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O.tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Y = X - self.O[: X.size(1)]
#         Y2 = Y ** 2
#         # pairwise terms
#         T = Y2[:, :-1] + Y2[:, 1:]
#         term = 0.5 + (torch.sin(torch.sqrt(T)) ** 2 - 0.5) / (1.0 + 0.001 * torch.sqrt(T)) ** 2
#         fx = torch.sum(term, dim=1)
#         # wrap-around last term
#         wrap = Y2[:, -1] + Y2[:, 0]
#         fx += 0.5 + (torch.sin(torch.sqrt(wrap)) ** 2 - 0.5) / (1.0 + 0.001 * torch.sqrt(wrap)) ** 2
#         fx = fx.unsqueeze(1)
#         # constraints on sum over dims
#         S = torch.sum(Y, dim=1, keepdim=True)
#         c1 = torch.cos(S) ** 2 - 0.25 * torch.cos(S) - 0.125
#         c2 = torch.exp(torch.cos(S)) - math.exp(0.25)
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# # utility for rotated problems 21–28
# def _select_rotated(D_requested, data12):
#     if D_requested is None or D_requested < 30:
#         return 10, data12['M_10'][0][0]
#     elif D_requested < 50:
#         return 30, data12['M_30'][0][0]
#     elif D_requested < 100:
#         return 50, data12['M_50'][0][0]
#     else:
#         return 100, data12['M_100'][0][0]


# class CEC2017_p21(BenchmarkProblem):
#     available_dimensions = [10, 30, 50, 100]
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D, M = _select_rotated(getattr(self, 'D', None), data12)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O[:D].tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)
#         self.Mat = torch.tensor(M, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z0 = X - self.O[: X.size(1)]
#         Z = Z0 @ self.Mat.T
#         fx = torch.sum(Z ** 2 - 10 * torch.cos(2 * math.pi * Z) + 10, dim=1, keepdim=True)
#         c1 = 4.0 - torch.sum(torch.abs(Z), dim=1, keepdim=True)
#         c2 = torch.sum(Z ** 2, dim=1, keepdim=True) - 4.0
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p22(BenchmarkProblem):
#     available_dimensions = [10, 30, 50, 100]
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D, M = _select_rotated(getattr(self, 'D', None), data12)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=3,
#             optimum=[[0.0]],
#             # x_opt=[O[:D].tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)
#         self.Mat = torch.tensor(M, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z0 = X - self.O[: X.size(1)]
#         Z = Z0 @ self.Mat.T
#         fx = torch.sum(100 * (Z[:, :-1] ** 2 - Z[:, 1:]) ** 2 + (Z[:, :-1] - 1.0) ** 2, dim=1, keepdim=True)
#         c1 = torch.sum(Z ** 2 - 10 * torch.cos(2 * math.pi * Z) + 10, dim=1, keepdim=True) - 100.0
#         c2 = torch.sum(Z, dim=1, keepdim=True) - 2.0 * Z.size(1)
#         c3 = 5.0 - torch.sum(Z, dim=1, keepdim=True)
#         gx = torch.cat([c1, c2, c3], dim=1)
#         return gx, fx


# class CEC2017_p23(BenchmarkProblem):
#     available_dimensions = [10, 30, 50, 100]
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D, M = _select_rotated(getattr(self, 'D', None), data12)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O[:D].tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)
#         self.Mat = torch.tensor(M, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z0 = X - self.O[: X.size(1)]
#         Z = Z0 @ self.Mat.T
#         fx = (
#             -20.0 * torch.exp(-0.2 * torch.sqrt(torch.mean(Z ** 2, dim=1)))
#             + 20.0
#             - torch.exp(torch.mean(torch.cos(2 * math.pi * Z), dim=1))
#             + math.e
#         ).unsqueeze(1)
#         c1 = torch.sum(Z[:, 1:] ** 2, dim=1, keepdim=True) + 1.0 - torch.abs(Z[:, :1])
#         c2 = torch.abs(torch.sum(Z ** 2, dim=1, keepdim=True) - 4.0) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p24(BenchmarkProblem):
#     available_dimensions = [10, 30, 50, 100]
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D, M = _select_rotated(getattr(self, 'D', None), data12)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O[:D].tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)
#         self.Mat = torch.tensor(M, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z0 = X - self.O[: X.size(1)]
#         Z = Z0 @ self.Mat.T
#         fx = torch.max(torch.abs(Z), dim=1, keepdim=True)[0]
#         c1 = torch.sum(Z ** 2, dim=1, keepdim=True) - 100.0 * Z.size(1)
#         c2 = torch.abs(torch.cos(fx) + torch.sin(fx)) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p25(BenchmarkProblem):
#     available_dimensions = [10, 30, 50, 100]
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D, M = _select_rotated(getattr(self, 'D', None), data12)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O[:D].tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)
#         self.Mat = torch.tensor(M, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z0 = X - self.O[: X.size(1)]
#         Z = Z0 @ self.Mat.T
#         fx = torch.sum(torch.abs(Z), dim=1, keepdim=True)
#         c1 = torch.sum(Z ** 2, dim=1, keepdim=True) - 100.0 * Z.size(1)
#         tmp = fx
#         c2 = torch.abs((torch.cos(tmp) + torch.sin(tmp)) ** 2
#                        - torch.exp(torch.cos(tmp) + torch.sin(tmp)) - 1 + math.e) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p26(BenchmarkProblem):
#     available_dimensions = [10, 30, 50, 100]
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D, M = _select_rotated(getattr(self, 'D', None), data12)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O[:D].tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)
#         self.Mat = torch.tensor(M, dtype=torch.float32)
#         idx = torch.arange(1, D + 1, dtype=torch.float32)
#         self.sqrt_idx = torch.sqrt(idx)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z0 = X - self.O[: X.size(1)]
#         Z = Z0 @ self.Mat.T
#         scaled = Z / self.sqrt_idx[: X.size(1)]
#         fx = (0.25 * torch.sum(Z ** 2, dim=1, keepdim=True) / 1000.0
#               + 1.0
#               - torch.prod(torch.cos(scaled), dim=1, keepdim=True))
#         sumZ2 = torch.sum(Z ** 2, dim=1, keepdim=True)
#         c1 = 1.0 - torch.sum(torch.sign(torch.abs(Z) - sumZ2 + Z ** 2 - 1.0), dim=1, keepdim=True)
#         c2 = torch.abs(sumZ2 - 4.0 * Z.size(1)) - 1e-4
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx


# class CEC2017_p27(BenchmarkProblem):
#     available_dimensions = [10, 30, 50, 100]
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D, M = _select_rotated(getattr(self, 'D', None), data12)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=3,
#             optimum=[[0.0]],
#             x_opt=[O[:D].tolist()],
#             bounds=[(-100.0, 100.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)
#         self.Mat = torch.tensor(M, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Y0 = X - self.O[: X.size(1)]
#         Y = Y0 @ self.Mat.T
#         Z = torch.where(
#             torch.abs(Y) < 0.5,
#             Y,
#             0.5 * torch.round(2.0 * Y)
#         )
#         fx = torch.sum(Z ** 2 - 10 * torch.cos(2 * math.pi * Z) + 10, dim=1, keepdim=True)
#         c1 = 1.0 - torch.sum(torch.abs(Y), dim=1, keepdim=True)
#         c2 = torch.sum(Y ** 2, dim=1, keepdim=True) - 100.0 * Y.size(1)
#         term = torch.sum(100 * (Y[:, :-1] ** 2 - Y[:, 1:]) ** 2, dim=1, keepdim=True)
#         term += torch.prod(torch.sin(math.pi * (Y - 1.0)) ** 2, dim=1, keepdim=True)
#         c3 = torch.abs(term) - 1e-4
#         gx = torch.cat([c1, c2, c3], dim=1)
#         return gx, fx


# class CEC2017_p28(BenchmarkProblem):
#     available_dimensions = [10, 30, 50, 100]
#     num_objectives = 1

#     def __init__(self):
#         data12 = _mat_data[0][11]
#         O = data12['o'].flatten()[0][0]
#         D, M = _select_rotated(getattr(self, 'D', None), data12)
#         super().__init__(
#             D,
#             num_objectives=1,
#             num_constraints=2,
#             optimum=[[0.0]],
#             x_opt=[O[:D].tolist()],
#             bounds=[(-50.0, 50.0)] * D,
#         )
#         self.O = torch.tensor(O, dtype=torch.float32)
#         self.Mat = torch.tensor(M, dtype=torch.float32)

#     def _evaluate_implementation(self, X: torch.Tensor):
#         Z0 = X - self.O[: X.size(1)]
#         Z = Z0 @ self.Mat.T
#         fx = torch.sum(torch.sqrt(torch.abs(Z)) + 2 * torch.sin(Z ** 3), dim=1, keepdim=True)
#         A = Z[:, :-1] ** 2 + Z[:, 1:] ** 2
#         c1 = torch.sum(-10 * torch.exp(-0.2 * torch.sqrt(A)), dim=1, keepdim=True)
#         c1 += (Z.size(1) - 1) * 10.0 / math.exp(-5.0)
#         c2 = torch.sum(torch.sin(2.0 * Z) ** 2, dim=1, keepdim=True) - 0.5 * Z.size(1)
#         gx = torch.cat([c1, c2], dim=1)
#         return gx, fx
