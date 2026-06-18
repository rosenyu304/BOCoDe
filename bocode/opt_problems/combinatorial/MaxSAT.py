"""Weighted MaxSAT — 60 binary variables (generated instance, self-contained).

A weighted Maximum-Satisfiability benchmark in the style used for combinatorial
Bayesian optimization (COMBO, Casmopolitan, Bounce): given weighted clauses over
``n`` binary variables, maximize the total weight of satisfied clauses. Unlike the
MaxSAT-Evaluation competition instances wrapped by Bencher, this uses a reproducibly
generated random weighted 3-SAT instance (fixed seed), so it is fully self-contained
(no external ``.wcnf`` files). 60 binary variables, 300 weighted 3-literal clauses.

Sources:
C. Oh, J. M. Tomczak, E. Gavves, M. Welling. Combinatorial Bayesian Optimization using the Graph Cartesian Product. Advances in Neural Information Processing Systems 32, 2019.
X. Wan, V. Nguyen, H. Ha, B. Ru, C. Lu, M. A. Osborne. Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces. International Conference on Machine Learning, 2021.
L. Papenmeier, L. Nardi. Bencher: simple and reproducible benchmarking for black-box optimization, 2025 (wraps the MaxSAT-Evaluation maxsat60/maxsat125 instances).
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem

_N_VARS = 60
_N_CLAUSES = 300
_CLAUSE_LEN = 3


def _generate_instance(n_vars: int, n_clauses: int, clause_len: int, seed: int = 0):
    """Reproducible weighted 3-SAT: list of (var indices, literal signs, weight)."""
    rng = np.random.RandomState(seed)
    clauses = []
    for _ in range(n_clauses):
        variables = rng.choice(n_vars, size=clause_len, replace=False)
        signs = rng.randint(0, 2, size=clause_len)  # 1 = positive literal, 0 = negated
        weight = float(rng.randint(1, 100))
        clauses.append((variables, signs, weight))
    return clauses


_CLAUSES = _generate_instance(_N_VARS, _N_CLAUSES, _CLAUSE_LEN)


class MaxSAT(BenchmarkProblem):
    """Maximize satisfied-clause weight of a weighted 3-SAT instance (60 binary vars)."""

    available_dimensions = _N_VARS
    num_objectives = 1
    num_constraints = 0

    def __init__(self) -> None:
        self.variable_types = [[0, 1]] * _N_VARS  # binary (categorical)
        super().__init__(
            dim=_N_VARS,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0, 1)] * _N_VARS,
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = (X.detach().cpu().numpy() > 0.5).astype(int)
        sat_weight = np.zeros(x.shape[0])
        for variables, signs, weight in _CLAUSES:
            # a clause is satisfied if any literal is true: literal true when x == sign
            satisfied = (x[:, variables] == signs[None, :]).any(axis=1)
            sat_weight += weight * satisfied
        fx = torch.tensor(sat_weight, dtype=torch.float64).reshape(-1, 1)  # maximize
        return None, fx
