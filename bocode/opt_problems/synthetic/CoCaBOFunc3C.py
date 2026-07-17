"""CoCaBO Func3C: 2 continuous + 3 categorical (K=3, K=5, K=4).

The Func3C benchmark of the CoCaBO paper: Func2C plus a third categorical that
adds a *weighted* sub-function, making the categorical block interact even more
strongly with the continuous search.

The two continuous inputs live in ``[-1, 1]`` and are scaled by 2 internally. The
three scaled sub-functions ``ros``, ``six`` and ``beale`` are as in
:mod:`~bocode.opt_problems.synthetic.CoCaBOFunc2C`.

Objective (minimize)::

    f = [ros, six, beale][c0]
      + (ros if c1 == 0 else six if c1 == 1 else beale)
      + (5 six if c2 == 0 else 2 ros if c2 == 1 else c2 * beale)

Variables: ``x0, x1`` continuous in ``[-1, 1]``; ``c0`` categorical K=3; ``c1``
categorical K=5; ``c2`` categorical K=4. All three categoricals are nominal, so
the decision variable carries the level *index*.

No closed-form optimum is published. ``evaluate()`` returns ``-f`` because BoCoDe
maximizes.

Sources:
B. Ru, A. S. Alvi, V. Nguyen, M. A. Osborne, S. J. Roberts. Bayesian Optimisation over Multiple Continuous and Categorical Inputs. ICML 2020 (arXiv:1906.08878).
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem
from ._common import cat_index, cocabo_subfunctions


class CoCaBOFunc3C(BenchmarkProblem):
    """CoCaBO Func3C: 2 continuous + 3 nominal categoricals (K=3, K=5, K=4)."""

    available_dimensions = 5
    num_objectives = 1
    num_constraints = 0

    def __init__(self) -> None:
        self.variable_types = [
            "continuous",
            "continuous",
            [0.0, 1.0, 2.0],  # c0: K=3
            [0.0, 1.0, 2.0, 3.0, 4.0],  # c1: K=5
            [0.0, 1.0, 2.0, 3.0],  # c2: K=4
        ]
        super().__init__(
            dim=5,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0), (0.0, 4.0), (0.0, 3.0)],
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = X.to(torch.float64)
        x1, x2 = x[:, 0] * 2, x[:, 1] * 2
        c0 = cat_index(x[:, 2], 3)
        c1 = cat_index(x[:, 3], 5)
        c2 = cat_index(x[:, 4], 4)

        ros, six, beale = cocabo_subfunctions(x1, x2)
        subs = torch.stack([ros, six, beale], dim=1)  # (n, 3)

        f = subs.gather(1, c0.unsqueeze(1)).squeeze(1)
        # c1: level 0 -> ros, level 1 -> six, every other level -> beale
        f = f + torch.where(c1 == 0, ros, torch.where(c1 == 1, six, beale))
        # c2: level 0 -> 5*six, level 1 -> 2*ros, every other level -> c2 * beale
        third = torch.where(
            c2 == 0,
            5.0 * six,
            torch.where(c2 == 1, 2.0 * ros, c2.to(torch.float64) * beale),
        )
        f = f + third

        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
