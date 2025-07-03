"""
Y.-K. Tsai and R. J. Malak Jr, “Surrogate-assisted constraint-handling technique for constrained parametric multi-objective optimization,” Structural and Multidisciplinary Optimization, 2024.
https://link.springer.com/article/10.1007/s00158-024-03859-y
"""

import math
from typing import Tuple

import torch

from ..base import BenchmarkProblem, DataType


def parameterfunA(x: torch.Tensor, p: int) -> torch.Tensor:
    """
    This function computes the parameters based on the decision variables x.

    Args:
        x (torch.Tensor): Input tensor of decision variables.
        p (int): Number of parameters.

    Returns:
        torch.Tensor: The calculated parameter values.  Shape: (batch_size, p)
    """

    p = min(p, x.shape[1])
    return x[:, :p]


def g_fun(x: torch.Tensor, m: int, p: int, n: int) -> torch.Tensor:
    """
    Calculates the g function value.

    Args:
        x (torch.Tensor): Input tensor of decision variables.
        m (int): Number of objectives.
        p (int): Number of parameters.
        n (int): Total number of decision variables.

    Returns:
        torch.Tensor: The computed g value. Shape: (batch_size, 1)
    """

    if n <= 1:
        x_opt = torch.full_like(x[:, :1], 0.5)
    else:
        x_opt = 0.5 + 0.01 / (n - 1) * torch.sin(5 * torch.pi * x[:, :-1]).sum(
            dim=1, keepdim=True
        )

    last_var_index = min(n - 1, x.shape[1] - 1)
    g = 1 + 5 * (x[:, last_var_index : last_var_index + 1] - x_opt) ** 2
    return g


def objectivefunA(x: torch.Tensor, m: int, p: int) -> torch.Tensor:
    """
    This function computes the objective function values based on decision variables x.

    Args:
        x (torch.Tensor): Input tensor of decision variables.
        m (int): Number of objectives.
        p (int): Number of parameters.

    Returns:
        torch.Tensor: The computed objective function values.  Shape: (batch_size, m)
    """
    nvars = x.size(1)
    batch_size = x.shape[0]
    f = torch.zeros((batch_size, m), device=x.device, dtype=x.dtype)

    if m == 1:
        theta = parameterfunA(x, p)
        g = g_fun(x, m, p, nvars)
        h = 3 * (m + p)
        h_term2 = 2 * theta + torch.sin(3 * math.pi * theta)
        h_term2 = h_term2.sum(dim=1, keepdim=True)
        h = h + h_term2 / (5 * g)
        f[:, 0] = g.squeeze() * h.squeeze()

    else:
        theta = parameterfunA(x, p)

        valid_cols = min(m - 1, x.shape[1] - p)
        f[:, :valid_cols] = x[:, p : p + valid_cols]
        g = g_fun(x, m, p, nvars)
        h = 3 * (m + p)

        h_term2 = 2 * theta + torch.sin(3 * math.pi * theta)
        h_term2 = h_term2.sum(dim=1, keepdim=True)

        sqrt_term_cols = min(m - 1, f.shape[1])
        h_term2 += 10 * torch.sqrt(1 - f[:, :sqrt_term_cols] ** 2).sum(
            dim=1, keepdim=True
        )

        h = h + h_term2 / (5 * g)
        f[:, m - 1] = (g * h).squeeze()
    return f


def objectivefunB(x: torch.Tensor, m: int, p: int) -> torch.Tensor:
    """
    Computes the objective function values for problem B.

    Args:
        x (torch.Tensor): Input tensor (batch_size, n_vars).
        m (int): Number of objectives.
        p (int): Number of parameters.

    Returns:
        torch.Tensor: Objective function values (batch_size, m).
    """
    nvars = x.shape[1]
    batch_size = x.shape[0]
    f = torch.zeros((batch_size, m), dtype=x.dtype, device=x.device)

    if m == 1:
        theta = parameterfunA(x, p)
        g = g_fun(x, m, p, nvars)
        h = 3 * (m + p)
        h_term2 = 2 * theta + torch.sin(3 * torch.pi * theta)
        h += h_term2.sum(dim=1, keepdim=True) / (5 * g)
        f[:, 0] = (g * h).squeeze()
    else:
        theta = parameterfunA(x, p)
        valid_cols = min(m - 1, x.shape[1] - p)
        f[:, :valid_cols] = x[:, p : p + valid_cols]
        g1 = g_fun(x, m, p, nvars)
        h = 3 * (m + p)
        h_term2 = (2 * theta + torch.sin(3 * torch.pi * theta)).sum(dim=1, keepdim=True)
        sqrt_term_cols = min(m - 1, f.shape[1])
        h_term2 += 10 * torch.sqrt(1 - f[:, :sqrt_term_cols] ** 2).sum(
            dim=1, keepdim=True
        )
        h += h_term2 / (5 * g1)
        f[:, -1] = (g1 * h).squeeze()

    return f


class NonLinearConstraintProblemA3(BenchmarkProblem):
    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = (1, None)
    num_constraints = 0

    def __init__(
        self,
        dim: int,
        num_objectives: int = 1,
        bounds=None,
        num_constraints: int = 0,
    ) -> None:
        super().__init__(dim, bounds, num_objectives, num_constraints)
        self.m = num_objectives
        self.p = dim

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the problem. This method returns the constraint values (gx) and objective function values (fx).

        Args:
            x (torch.Tensor): Input tensor of decision variables.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple (gx, fx) where gx are the inequality constraints and
                                                fx are the objective function values.
        """

        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor.")

        batch_size = x.shape[0]
        cm = torch.full((batch_size, 1), 0.6, device=x.device, dtype=x.dtype)

        if self.p >= 1:
            theta = parameterfunA(x, self.p)
            cm += (
                0.2 / self.p * torch.sin(1.5 * math.pi * theta).sum(dim=1, keepdim=True)
            )
            gx = -(x[:, self.p : self.p + 1] - cm)
        else:
            f = objectivefunA(x, self.m, self.p)

            cm += 0.2 / (x.shape[1] - 1) * torch.sin(torch.pi * (f[:, :1] + 0.5))
            gx = -(x[:, :1] - cm)

        fx = objectivefunA(x, self.m, self.p)
        return gx, fx

    def g_fun(self, x: torch.Tensor) -> torch.Tensor:
        """
        The g function computes a constraint based on the input decision variables x.

        Args:
            x (torch.Tensor): Input tensor of decision variables.

        Returns:
            torch.Tensor: The computed g value.
        """
        nvars = x.size(1)
        g = torch.tensor(1.0)
        g += 5 * (x[:, nvars - 1] - 0.5) ** 2
        return g


class NonLinearConstraintProblemA4(BenchmarkProblem):
    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = (1, None)
    num_constraints = 0

    def __init__(
        self, dim: int, num_objectives: int = 1, bounds=None, num_constraints: int = 0
    ) -> None:
        super().__init__(dim, bounds, num_objectives, num_constraints)
        self.m = num_objectives
        self.p = dim

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the constraints and return (gx, fx).

        Args:
            x (torch.Tensor): Input tensor of decision variables.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: gx (inequality constraints), fx (objective function values).
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor.")

        batch_size = x.shape[0]
        cm = torch.full((batch_size, 1), 0.75, device=x.device, dtype=x.dtype)

        if self.p >= 1:
            theta = parameterfunA(x, self.p)
            cm += 0.1 / self.p * torch.sin(2 * math.pi * theta).sum(dim=1, keepdim=True)
            gx = -(x[:, self.p : self.p + 1] - cm)
        else:
            f = objectivefunA(x, self.m, self.p)
            cm += 0.2 / (x.shape[1] - 1) * torch.sin(torch.pi * (f[:, :1] + 0.5))
            gx = -(x[:, :1] - cm)

        fx = objectivefunA(x, self.m, self.p)
        return gx, fx


class NonLinearConstraintProblemA7(BenchmarkProblem):
    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = (1, None)
    num_constraints = [1, 3]

    def __init__(
        self,
        dim: int,
        num_objectives: int = 1,
        bounds=None,
        # num_constraints: int = 1,
        x_opt=None,
        optimum=None,
    ) -> None:
        num_constraints = 1 if (num_objectives == 1) else 3
        super().__init__(dim, bounds, num_objectives, num_constraints)
        self.m = num_objectives
        self.p = dim
        self.x_opt = x_opt
        self.optimum = optimum

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the problem by computing inequality constraints (gix < 0),
        equality constraints (gex = 0), and objective function values (fx).

        Args:
            x (torch.Tensor): Input tensor of decision variables.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple (gix, fx), where:
                - gix: Inequality constraints (should be < 0)
                - fx: Objective function values
        """
        batch_size = x.shape[0]
        n = x.shape[1]

        gix = []

        if self.p >= 1:
            cm = 0.8 + 0.1 / self.p * torch.sin(2 * math.pi * x[:, : self.p]).sum(
                dim=1, keepdim=True
            )

            last_index = min(self.p + self.m - 1, x.shape[1] - 1)
            gix.append(-(x[:, last_index : last_index + 1] - cm))

        if self.m > 1:
            cm = torch.full((batch_size, 1), 0.85, device=x.device, dtype=x.dtype)
            cm2 = torch.full((batch_size, 1), 0.65, device=x.device, dtype=x.dtype)

            if self.p >= 1:
                cm += (
                    0.1
                    / (n - 1)
                    * torch.cos(4 * math.pi * x[:, : self.p]).sum(dim=1, keepdim=True)
                )
                cm2 -= (
                    0.1
                    / (n - 1)
                    * torch.cos(4 * math.pi * x[:, : self.p]).sum(dim=1, keepdim=True)
                )

            fx = objectivefunA(x, self.m, self.p)

            cols = min(self.m - 1, fx.shape[1])
            cm += (
                0.1
                / (n - 1)
                * torch.cos(4 * math.pi * fx[:, :cols]).sum(dim=1, keepdim=True)
            )
            cm2 -= (
                0.1
                / (n - 1)
                * torch.cos(4 * math.pi * fx[:, :cols]).sum(dim=1, keepdim=True)
            )

            last_index = min(self.p + self.m - 1, x.shape[1] - 1)
            gix.append(-cm + x[:, last_index : last_index + 1])
            gix.append(cm2 - x[:, last_index : last_index + 1])
        else:
            fx = objectivefunA(x, self.m, self.p)

        gix = (
            torch.cat(gix, dim=1)
            if gix
            else torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        )

        return gix, fx


class NonLinearConstraintProblemA8(BenchmarkProblem):
    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = (1, None)
    num_constraints = [2, 4]

    def __init__(
        self,
        dim: int,
        num_objectives: int = 1,
        bounds=None,
        # num_constraints: int = 2,
    ):
        num_constraints = 2 if (num_objectives == 1) else 4
        super().__init__(dim, bounds, num_objectives, num_constraints)
        self.p = dim
        self.m = num_objectives

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the problem and returns the inequality constraints (gix),  and function values (fx).

        Args:
            x (torch.Tensor): Input tensor of decision variables.

        Returns:
            Tuple[torch.Tensor,  torch.Tensor]:
                gix - Inequality constraints (should be negative),
                fx - Objective function values.
        """
        batch_size = x.shape[0]
        n = x.shape[1]
        cm_old = -0.1 * x[:, :1] + 0.6
        cm_old2 = torch.full((batch_size, 1), 0.4, device=x.device, dtype=x.dtype)

        gix1 = torch.empty((batch_size, 0), device=x.device, dtype=x.dtype)
        gix2 = torch.empty((batch_size, 0), device=x.device, dtype=x.dtype)
        gix3 = torch.empty((batch_size, 0), device=x.device, dtype=x.dtype)
        gix4 = torch.empty((batch_size, 0), device=x.device, dtype=x.dtype)
        if self.p >= 1:
            cm_old += (
                0.3
                / self.p
                * torch.sin(1.5 * math.pi * x[:, : self.p]).sum(dim=1, keepdim=True)
            )

            if self.m > 1:
                f = objectivefunA(x, self.m, self.p)

                f_cols = min(self.m - 1, f.shape[1])
                cm_old2 += (
                    0.2
                    * torch.sin(math.pi * x[:, :1])
                    / (self.m - 1)
                    * torch.cos(math.pi / 2 * f[:, :f_cols])
                ).sum(dim=1, keepdim=True)

            last_index = min(self.p + self.m - 1, x.shape[1] - 1)
            gix1 = -(x[:, last_index : last_index + 1] - cm_old)
            gix2 = -(x[:, last_index : last_index + 1] - cm_old2)

        if self.m > 1:
            cm = -0.3 * x[:, :1] + 0.8
            cm2 = -0.3 * x[:, :1] + 0.7
            f = objectivefunA(x, self.m, self.p)

            if self.p >= 1:
                cm += (
                    0.1
                    / (n - 1)
                    * torch.cos(4 * math.pi * x[:, : self.p]).sum(dim=1, keepdim=True)
                )
                cm2 -= (
                    0.1
                    / (n - 1)
                    * torch.cos(4 * math.pi * x[:, : self.p]).sum(dim=1, keepdim=True)
                )

            f_cols = min(self.m - 1, f.shape[1])
            cm += (
                0.1
                / (n - 1)
                * torch.cos(4 * math.pi * f[:, :f_cols]).sum(dim=1, keepdim=True)
            )
            cm2 -= (
                0.1
                / (n - 1)
                * torch.cos(4 * math.pi * f[:, :f_cols]).sum(dim=1, keepdim=True)
            )

            last_index = min(self.p + self.m - 1, x.shape[1] - 1)
            gix3 = -cm + x[:, last_index : last_index + 1]
            gix4 = cm2 - x[:, last_index : last_index + 1]
        else:
            f = objectivefunA(x, self.m, self.p)

        gix_list = [gix1, gix2, gix3, gix4]
        gix_list = [t for t in gix_list if t.numel() > 0]
        gix = (
            torch.cat(gix_list, dim=1)
            if gix_list
            else torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        )

        return gix, f


class NonLinearConstraintProblemB3(BenchmarkProblem):
    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = (1, None)
    num_constraints = 0

    def __init__(
        self,
        dim: int,
        num_objectives: int = 1,
        bounds=None,
        num_constraints: int = 0,
    ):
        super().__init__(
            dim=dim,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            bounds=bounds,
        )
        self.p = dim
        self.m = num_objectives

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fx = objectivefunB(x, self.m, self.p)
        batch_size = x.shape[0]

        cm = torch.full((batch_size, 1), 0.6, device=x.device, dtype=x.dtype)
        if self.p >= 1:
            theta = parameterfunA(x, self.p)
            cm += (0.3 / self.p) * torch.sin(1.5 * torch.pi * theta).sum(
                dim=1, keepdim=True
            )
            gix = -(x[:, self.p : self.p + 1] - cm)
        else:
            fx_cols = min(self.m - 1, fx.shape[1])
            cm += (0.3 / (x.shape[1] - 1)) * torch.sin(
                torch.pi * (fx[:, :fx_cols] + 0.5)
            ).sum(dim=1, keepdim=True)

            obj_index = min(self.m, x.shape[1] - 1)
            gix = -(x[:, obj_index : obj_index + 1] - cm)

        return gix, fx


class NonLinearConstraintProblemB4(BenchmarkProblem):
    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = (1, None)
    num_constraints = 1

    def __init__(
        self, dim: int, num_objectives: int = 1, bounds=None, num_constraints: int = 1
    ) -> None:
        super().__init__(
            dim=dim,
            bounds=bounds,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
        )
        self.p = dim
        self.m = num_objectives

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(x):
            raise TypeError("Error: X must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Error: X must be a 2D tensor")

        batch_size = x.shape[0]
        n = x.shape[1]
        cm = torch.full((batch_size, 1), 0.75, device=x.device, dtype=x.dtype)

        if self.p >= 1:
            theta = parameterfunA(x, self.p)
            cm += (0.1 / self.p * torch.sin(2 * torch.pi * theta)).sum(
                dim=1, keepdim=True
            )
            p_index = min(self.p, x.shape[1] - 1)
            gix = -(x[:, p_index : p_index + 1] - cm)
        else:
            f = objectivefunB(x, self.m, self.p)

            f_cols = min(self.m - 1, f.shape[1])
            cm += (
                0.2
                / (n - 1)
                * torch.sin(torch.pi * (f[:, :f_cols] + 0.5)).sum(dim=1, keepdim=True)
            )
            m_index = min(self.m, x.shape[1] - 1)
            gix = -(x[:, m_index : m_index + 1] - cm)

        fx = objectivefunB(x, self.m, self.p)

        return gix, fx


class NonLinearConstraintProblemB7(BenchmarkProblem):
    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = (1, None)
    num_constraints = [1, 2]

    def __init__(
        self,
        dim: int,
        num_objectives: int = 1,
        bounds=None,
        # num_constraints: int = 1,
    ):
        num_constraints = 1 if (num_objectives == 1) else 2
        super().__init__(
            dim=dim,
            bounds=bounds,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
        )
        self.n = dim
        self.m = num_objectives

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the problem constraints and objective function.

        Args:
            X (torch.Tensor): A 2D tensor of decision variables.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - gix: Inequality constraints
                - fx: Objective function values
        """
        batch_size = x.shape[0]
        n = self.n
        m = self.m
        p = n - m if n > m else 0

        gix = []
        if m > 1:
            cm = torch.full((batch_size, 1), 0.85, device=x.device, dtype=x.dtype)
            cm2 = torch.full((batch_size, 1), 0.65, device=x.device, dtype=x.dtype)

            if p >= 1:
                cm += (
                    0.15
                    / (n - 1)
                    * torch.sin(10 * torch.pi * x[:, :p]).sum(dim=1, keepdim=True)
                )
                cm2 -= (
                    0.15
                    / (n - 1)
                    * torch.sin(10 * torch.pi * x[:, :p]).sum(dim=1, keepdim=True)
                )

            f = objectivefunB(x, m, p)

            f_cols = min(m - 1, f.shape[1])
            cm += (0.05 / (n - 1) * torch.sin(10 * torch.pi * f[:, :f_cols])).sum(
                dim=1, keepdim=True
            )
            cm2 -= (0.05 / (n - 1) * torch.sin(10 * torch.pi * f[:, :f_cols])).sum(
                dim=1, keepdim=True
            )

            last_index = min(p + m - 1, x.shape[1] - 1)
            gix.append(-cm + x[:, last_index : last_index + 1])
            gix.append(cm2 - x[:, last_index : last_index + 1])
        else:
            cm = 0.8
            if p >= 1:
                cm += (
                    0.1
                    / p
                    * torch.sin(2 * torch.pi * x[:, :p]).sum(dim=1, keepdim=True)
                )
            last_index = min(p + m - 1, x.shape[1] - 1)
            gix.append(-(x[:, last_index : last_index + 1] - cm))
            f = objectivefunB(x, m, p)

        gix = (
            torch.cat(gix, dim=1)
            if gix
            else torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        )

        return gix, f


class NonLinearConstraintProblemB8(BenchmarkProblem):
    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = (1, None)
    num_constraints = [2, 4]

    def __init__(
        self,
        dim: int,
        num_objectives: int = 1,
        bounds=None,
        # num_constraints: int = 2,
    ):
        num_constraints = 2 if (num_objectives == 1) else 4
        super().__init__(
            dim=dim,
            bounds=bounds,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
        )
        self.m = num_objectives
        self.p = dim

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        n = x.shape[1]
        cm_old = -0.1 * x[:, :1] + 0.6
        cm_old2 = torch.full((batch_size, 1), 0.4, device=x.device, dtype=x.dtype)

        gix1 = torch.empty((batch_size, 0), device=x.device, dtype=x.dtype)
        gix2 = torch.empty((batch_size, 0), device=x.device, dtype=x.dtype)
        gix3 = torch.empty((batch_size, 0), device=x.device, dtype=x.dtype)
        gix4 = torch.empty((batch_size, 0), device=x.device, dtype=x.dtype)

        if self.p >= 1:
            cm_old += (0.3 / self.p * torch.sin(1.5 * torch.pi * x[:, : self.p])).sum(
                dim=1, keepdim=True
            )

            if self.m > 1:
                f = objectivefunB(x, self.m, self.p)

                f_cols = min(self.m - 1, f.shape[1])
                cm_old2 += (
                    0.2
                    * torch.sin(torch.pi * x[:, :1])
                    / (self.m - 1)
                    * torch.cos(torch.pi / 2 * f[:, :f_cols])
                ).sum(dim=1, keepdim=True)

            last_index = min(self.p + self.m - 1, x.shape[1] - 1)
            gix1 = -(x[:, last_index : last_index + 1] - cm_old)
            gix2 = -(x[:, last_index : last_index + 1] - cm_old2)

        if self.m > 1:
            cm = -0.3 * x[:, :1] + 0.8
            cm2 = -0.3 * x[:, :1] + 0.7
            f = objectivefunB(x, self.m, self.p)

            if self.p >= 1:
                cm += (0.1 / (n - 1) * torch.cos(4 * torch.pi * x[:, : self.p])).sum(
                    dim=1, keepdim=True
                )
                cm2 -= (0.1 / (n - 1) * torch.cos(4 * torch.pi * x[:, : self.p])).sum(
                    dim=1, keepdim=True
                )

            f_cols = min(self.m - 1, f.shape[1])
            cm += (0.1 / (n - 1) * torch.cos(4 * torch.pi * f[:, :f_cols])).sum(
                dim=1, keepdim=True
            )
            cm2 -= (0.1 / (n - 1) * torch.cos(4 * torch.pi * f[:, :f_cols])).sum(
                dim=1, keepdim=True
            )

            last_index = min(self.p + self.m - 1, x.shape[1] - 1)
            gix3 = -cm + x[:, last_index : last_index + 1]
            gix4 = cm2 - x[:, last_index : last_index + 1]
        else:
            f = objectivefunB(x, self.m, self.p)

        gix_list = [gix1, gix2, gix3, gix4]
        gix_list = [t for t in gix_list if t.numel() > 0]
        gix = (
            torch.cat(gix_list, dim=1)
            if gix_list
            else torch.empty((batch_size, 0), device=x.device, dtype=x.dtype)
        )

        return gix, f
