"""Shekel-10 with two integer dimensions (2 integer + 2 continuous).

The 4-D Shekel function (``m = 10`` attraction basins) with its first two inputs
restricted to the integers ``{0, ..., 10}`` and the last two left continuous.
Shekel's many narrow basins make the integer restriction bite: the unconstrained
minimizer sits at ``x ~ (4.000747, 3.99951, 4.00075, 3.99951)``, so the integer
dimensions must land exactly on ``4``.

Objective (minimize), the standard Shekel-10 form::

    f(x) = - sum_{i=1..10} ( sum_{j=1..4} (x_j - C_ji)^2 + beta_i )^{-1}

with BoTorch's constants: ``beta = [1, 2, 2, 4, 4, 6, 3, 7, 5, 5] / 10`` and

    C = [[4, 1, 8, 6, 3, 2, 5, 8, 6, 7  ],
         [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
         [4, 1, 8, 6, 3, 2, 5, 8, 6, 7  ],
         [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]]

The math is written out here rather than delegated to ``botorch``'s ``Shekel``
because that class *raises* on inputs outside ``[0, 10]^4``, which would make the
problem brittle for optimizers that probe past the bounds. The constants are
identical; the two agree to ~5e-8 inside the bounds (BoTorch stores these buffers
in float32, so its ``3.6`` entry is not exactly representable).

Variables: ``x0, x1`` integer in ``[0, 10]``; ``x2, x3`` continuous in ``[0, 10]``.

The unconstrained minimum is ``-10.536443``; restricted to integer ``x0 = x1 = 4``
the reachable minimum is ``f* = -10.536363`` (minimization sense) at
``x2, x3 ~ (4.000746, 3.999510)``. ``evaluate()`` returns ``-f`` because BoCoDe
maximizes.

Sources:
J. Shekel. Test functions for multimodal search techniques. Proc. 5th Annual Princeton Conf. on Information Science and Systems, 1971.
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
J. Qian. Discretised mixed-variable benchmark suite, MIT MEng thesis (Fig. 3.2 benchmarks).
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem

_BETA = [1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 3.0, 7.0, 5.0, 5.0]
_C = [
    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
]


class ShekelMixed(BenchmarkProblem):
    """Shekel-10 with the first two dimensions restricted to integers (4-D, mixed)."""

    available_dimensions = 4
    num_objectives = 1
    num_constraints = 0

    def __init__(self) -> None:
        self.variable_types = ["integer", "integer", "continuous", "continuous"]
        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0.0, 10.0)] * 4,
            x_opt=[[4.0, 4.0, 4.000746, 3.999510]],
            optimum=[-10.536363],  # best with x0, x1 integer; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = X.to(torch.float64)
        beta = torch.tensor(_BETA, dtype=torch.float64) / 10.0  # (m,)
        C = torch.tensor(_C, dtype=torch.float64).T  # (m, 4): C[i] is basin i's centre

        # dist[n, i] = sum_j (x_nj - C_ij)^2
        dist = ((x.unsqueeze(1) - C) ** 2).sum(dim=2)
        f = -(1.0 / (dist + beta)).sum(dim=1)

        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
