import warnings
from functools import cached_property
from typing import List, Optional, Set, Tuple, Union

import torch

from .exceptions import DimensionException, RangeException, TypeException

warnings.filterwarnings("ignore")  # Ignore all warnings


class DataType:
    """
    Data types for the decision variables.
    Available DataTypes:
        - DataType.CONTINUOUS
        - DataType.DISCRETE
        - DataType.CATEGORICAL
    """

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    MIXED = "mixed"


class BenchmarkProblem:
    available_dimensions = None
    num_objectives = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the BenchmarkProblem class.
        This method checks if the class has implemented all required attributes.
        If any of the required attributes are not implemented, it raises a NotImplementedError.
        """
        if any(
            getattr(cls, attr, None) is None
            for attr in [
                "available_dimensions",
                "num_objectives",
                "num_constraints",
                "input_type",
            ]
        ):
            raise NotImplementedError(
                "This benchmark problem is not fully implemented yet."
            )
        return super().__new__(cls)

    def __init__(
        self,
        dim: int = 1,
        bounds: Union[List[Union[Tuple, Set]], None] = None,
        num_objectives: int = 1,
        num_constraints: int = 0,
        x_opt: Optional[torch.Tensor] = None,
        optimum: Optional[torch.Tensor] = None,
        ref_point: Optional[torch.Tensor] = None,
        tags: Optional[List[str]] = None,
        debug: bool = False,
    ) -> None:
        """Initialize the BenchmarkProblem class.

        Args:
            dim (int, optional): Dimension of the decision space. Defaults to 1.
            bounds (Union[List[Union[Tuple, Set]], None]): Bounds of the decision variables.
            num_objectives (int, optional): Number of objective functions. Defaults to 1.
            num_constraints (int, optional): Number of constraint functions. Defaults to 0.
            x_opt (torch.Tensor, optional): The decision variables that maximize the objective function(s). Defaults to None.
            optimum (torch.Tensor, optional): The optimal objective values corresponding to the x_opt. Defaults to None.
            ref_point (torch.Tensor, optional): Reference point for calculating hypervolume. Defaults to None.
            tags (List[str], optional): More information for the benchmark problem. Defaults to None.
            debug (bool, optional): Debugging flag. Defaults to False.
        """
        self.dim = dim
        self.bounds = bounds
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.is_constrained = num_constraints > 0
        self.x_opt = x_opt
        self.optimum = optimum
        self.ref_point = ref_point
        self.tags = tags
        self.debug = debug

    @cached_property
    def torch_bounds(self) -> torch.Tensor:
        """
        Converts the bounds to a torch tensor.
        If bounds are not provided, returns a tensor with zeros and ones.
        """
        if self.bounds is None:
            return None
        if torch.is_tensor(self.bounds):
            return self.bounds
        return torch.tensor(self.bounds)

    def evaluate(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the objective and constraint functions.
        Enforces X to be casted to float.

        Returns ``(values, constraints)``. ``values`` has shape
        ``(n, num_objectives)`` and ``constraints`` has shape
        ``(n, num_constraints)`` with the convention that a constraint is
        satisfied when its value is ``<= 0``.

        Problems may implement either a 2-element return ``(constraints, values)``
        or, for problems with separate equality and inequality constraints (e.g.
        the CEC2020 suite), a 3-element return
        ``(equality_constraints, inequality_constraints, values)``. In the latter
        case the two constraint blocks are concatenated here into a single
        ``constraints`` tensor, so that ``constraints.shape[1] == num_constraints``
        for every problem.
        """
        output = self._evaluate_implementation(X.clone().to(torch.float))
        n = X.shape[0]
        if len(output) == 2:
            constraints, values = output
            return values, self._normalize_constraints(constraints, n)
        # len(output) == 3: (equality, inequality, values) for CEC2020 functions.
        equality, inequality, values = output
        eq = self._normalize_constraints(equality, n)
        ineq = self._normalize_constraints(inequality, n)
        return values, torch.cat([eq, ineq], dim=1)

    @staticmethod
    def _normalize_constraints(constraints, n: int) -> torch.Tensor:
        """Coerce a constraint block to a 2-D ``(n, k)`` tensor.

        Handles the shape inconsistencies present across the problem suite:
        ``None`` becomes an empty ``(n, 0)`` block, a stray trailing singleton
        dimension from an erroneous ``unsqueeze`` is removed, and a 1-D block is
        treated as a single constraint column.
        """
        if constraints is None:
            return torch.zeros((n, 0))
        if constraints.dim() == 3 and constraints.shape[-1] == 1:
            constraints = constraints.squeeze(-1)
        if constraints.dim() == 1:
            constraints = constraints.unsqueeze(-1)
        return constraints

    def _evaluate_implementation(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the objective and constraint functions.
        """
        raise NotImplementedError(
            "This benchmark problem is not fully implemented yet."
        )

    def scale(self, X) -> torch.Tensor:
        """
        Scales a fully continuous X to the problem's bounds.

        Input Args:
            X (torch.Tensor): continuous data in range of (0, 1)

        Returns:
            X (torch.Tensor): continuous data scaled to bounds
        """

        if not torch.is_tensor(X):
            raise TypeException("Error: X in scale() is not torch tensor")

        if X.size(1) != self.dim:
            raise DimensionException("Error: Incorrect X dimensions.")
        if torch.max(X) > 1 or torch.min(X) < 0:
            raise RangeException("Error: Incorrect X range: must be (0, 1).")

        bounds = self.torch_bounds.to(X)

        X_scaled = torch.add(torch.mul(X, (bounds[:, 1] - bounds[:, 0])), bounds[:, 0])

        return X_scaled

    def show_info(self) -> None:
        """
        Prints the information about the benchmark problem.
        """
        print(
            "Function info:\n",
            f"Number of objectives: {self.num_objectives}\n",
            f"Number of constraints: {self.num_constraints}\n",
            f"Number of dimensions: {self.dim}\n",
            f"Optimum Value: {self.optimum}\n",
            f"Optimal Decision Variables: {self.x_opt}\n",
            f"Bounds: {self.bounds}\n",
        )

    def visualize_function(self, sampling_density: int = 50) -> None:
        """
        Visualizes the benchmark problem function.
        Decrease sampling_density for faster rendering. Default is 50. Increase for better resolution.
        -----
        sampling_density: sampling density per axis. Number of evaluated points = sampling_density^2
        """
        from .visualization import visualize_function

        visualize_function(self, sampling_density)
