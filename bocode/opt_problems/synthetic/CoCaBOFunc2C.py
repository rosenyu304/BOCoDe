"""CoCaBO Func2C: 2 continuous + 2 categorical (K=3, K=5).

The Func2C benchmark of the CoCaBO paper. Each categorical level selects one of
three 2-D sub-functions (Rosenbrock / six-hump camel / Beale), so the categorical
choice is *non-separable* from the continuous optimization — unlike toy
additive-offset categoricals.

The two continuous inputs live in ``[-1, 1]`` and are scaled by 2 internally
(giving effective coordinates in ``[-2, 2]``). The scaled sub-functions are::

    ros(x1, x2)   = (100 (x2 - x1^2)^2 + (x1 - 1)^2) / 300
    six(x1, x2)   = ((4 - 2.1 x1^2 + x1^4 / 3) x1^2 + x1 x2 + (-4 + 4 x2^2) x2^2) / 10
    beale(x1, x2) = ((1.5 - x1 + x1 x2)^2 + (2.25 - x1 + x1 x2^2)^2 + (2.625 - x1 + x1 x2^3)^2) / 50

Objective (minimize)::

    f = [ros, six, beale][c0]  +  (ros if c1 == 0 else six if c1 == 1 else beale)

Variables: ``x0, x1`` continuous in ``[-1, 1]``; ``c0`` categorical with K=3
levels (indices ``{0, 1, 2}``); ``c1`` categorical with K=5 levels (indices
``{0, ..., 4}``). Both categoricals are nominal, so the decision variable carries
the level *index*.

No closed-form optimum is published. ``evaluate()`` returns ``-f`` because BoCoDe
maximizes.

Sources:
B. Ru, A. S. Alvi, V. Nguyen, M. A. Osborne, S. J. Roberts. Bayesian Optimisation over Multiple Continuous and Categorical Inputs. ICML 2020 (arXiv:1906.08878).
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem
from ._common import cat_index, cocabo_subfunctions


class CoCaBOFunc2C(BenchmarkProblem):
    """CoCaBO Func2C: 2 continuous + 2 nominal categoricals (K=3, K=5)."""

    available_dimensions = 4
    num_objectives = 1
    num_constraints = 0

    def __init__(self) -> None:
        self.variable_types = [
            "continuous",
            "continuous",
            [0.0, 1.0, 2.0],  # c0: K=3
            [0.0, 1.0, 2.0, 3.0, 4.0],  # c1: K=5
        ]
        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=0,
            bounds=[(-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0), (0.0, 4.0)],
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = X.to(torch.float64)
        x1, x2 = x[:, 0] * 2, x[:, 1] * 2
        c0 = cat_index(x[:, 2], 3)
        c1 = cat_index(x[:, 3], 5)

        ros, six, beale = cocabo_subfunctions(x1, x2)
        subs = torch.stack([ros, six, beale], dim=1)  # (n, 3)

        f = subs.gather(1, c0.unsqueeze(1)).squeeze(1)
        # c1: level 0 -> ros, level 1 -> six, every other level -> beale
        second = torch.where(c1 == 0, ros, torch.where(c1 == 1, six, beale))
        f = f + second

        fx = (-f).reshape(-1, 1)  # maximize -f
        return None, fx
