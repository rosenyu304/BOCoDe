"""Ackley-53: 50 binary categorical + 3 continuous (Casmopolitan's mixed benchmark).

The 53-D Ackley function with the first 50 variables binary categorical and the last
3 continuous — the reference *high-dimensional mixed* benchmark introduced with
Casmopolitan (Wan et al. 2021). With ``2^50 ~ 1.1e15`` categorical configurations it
is the standard stress test for mixed-variable BO, and the largest categorical space
in this package by a wide margin.

Objective (minimize), with ``a = 20``, ``b = 0.2``, ``c = 2 pi`` and ``n = 53``::

    f(x) = -a exp(-b sqrt(sum x_i^2 / n)) - exp(sum cos(c x_i) / n) + a + e

Variables: ``x0..x49`` binary categorical over the two levels ``{0, 32.768}``;
``x50, x51, x52`` continuous in ``[-32.768, 32.768]``. The categorical levels are the
physical values of ``x_i``, so they are used directly by the objective.

The grid follows the MCBO formulation of the task (``num_dims=[50, 3]``,
``variable_type=["nominal", "num"]``, ``num_categories=[2]*50``) over the SFU Ackley
box ``[-32.768, 32.768]``, with the binary levels placed on the *one-sided* grid
``linspace(0, 32.768, 2)`` (see ``_common.onesided_levels``). Reference optimum
``f* = 0`` (minimization sense) with every categorical at level 0 and every
continuous variable at ``0``.

.. note::
   **The one-sided grid is what makes this problem well-posed, not a cosmetic
   choice.** Ackley is *even* in every coordinate, so on the symmetric binary grid
   ``{-32.768, +32.768}`` the two levels contribute *identically* — both give
   ``x^2 = 1073.7`` and the same ``cos(2 pi x)``. All 50 categorical variables would
   then be completely **inert** (every one of the ``2^50`` configurations scores
   exactly ``21.5068`` at the continuous origin) and this "mixed" benchmark would
   collapse to a 3-D continuous Ackley — precisely the ``Ackley5Mixed`` pathology,
   which would defeat the purpose of the benchmark. On the one-sided grid ``{0,
   32.768}`` the two levels are distinct and the categorical structure is real.

   Note this means the box differs from Casmopolitan's own Ackley-53, which uses
   binary ``{0, 1}`` and continuous ``[-1, 1]``; the levels/box here follow MCBO's
   SFU parameterisation, so absolute ``f`` values are not comparable with the
   Casmopolitan paper's, only the ``f* = 0`` optimum and the search-space structure.

``evaluate()`` returns ``-f`` because BoCoDe maximizes.

Sources:
X. Wan, V. Nguyen, H. Ha, B. Ru, C. Lu, M. A. Osborne. Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces. Proceedings of the 38th International Conference on Machine Learning, PMLR 139:10663-10674, 2021. https://github.com/xingchenwan/Casmopolitan
S. Surjanovic, D. Bingham. Virtual Library of Simulation Experiments: Test Functions and Datasets — Ackley Function. https://www.sfu.ca/~ssurjano/ackley.html
K. Dreczkowski, A. Grosnit, H. Bou-Ammar. Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization. Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks), 2023. arXiv:2306.09803 (the ``num_dims=[50, 3]`` Ackley task formulation used here).
"""

from __future__ import annotations

import torch

from ...base import BenchmarkProblem
from ._common import onesided_levels, sfu_ackley


class Ackley53(BenchmarkProblem):
    """53-D Ackley over 50 binary categorical + 3 continuous variables (mixed)."""

    available_dimensions = 53
    num_objectives = 1
    num_constraints = 0

    NUM_CATEGORICAL = 50
    NUM_CONTINUOUS = 3
    DIM = NUM_CATEGORICAL + NUM_CONTINUOUS
    UB = 32.768  # the SFU Ackley box is [-UB, UB]

    def __init__(self) -> None:
        self.LEVELS = onesided_levels(self.UB, 2)  # binary: {0.0, 32.768}
        self.variable_types = [list(self.LEVELS)] * self.NUM_CATEGORICAL + [
            "continuous"
        ] * self.NUM_CONTINUOUS
        super().__init__(
            dim=self.DIM,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0.0, self.UB)] * self.NUM_CATEGORICAL
            + [(-self.UB, self.UB)] * self.NUM_CONTINUOUS,
            x_opt=[[0.0] * self.DIM],  # all categoricals at level 0, continuous at 0
            optimum=[0.0],  # on-grid global minimum; evaluate() returns -f
        )

    def _evaluate_implementation(self, X: torch.Tensor):
        x = self.enforce_variable_types(X.to(torch.float64))
        # keep the continuous coordinates inside the box the landscape is defined on
        bounds = self.torch_bounds.to(x)
        x = torch.clamp(x, bounds[:, 0], bounds[:, 1])
        fx = (-sfu_ackley(x)).reshape(-1, 1)  # maximize -f
        return None, fx
