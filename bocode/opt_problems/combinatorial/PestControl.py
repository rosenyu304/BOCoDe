"""Pest control — 25-stage categorical pesticide scheduling (COMBO benchmark).

A standard categorical combinatorial-BO benchmark: choose, at each of 25 sequential
stages, one of 5 actions (do nothing, or apply one of 4 pesticide types). Pests
spread stage to stage (a stochastic Markov model averaged over 100 simulations);
each pesticide has a price, develops tolerance with repeated use, and earns a
volume discount. Minimize the total cost = pesticide spending + the fraction of
stages whose pest level exceeds a threshold. All 25 variables are categorical
(5 unordered choices each). Ported verbatim from the COMBO reference implementation
(Oh et al., NeurIPS 2019); the simulation RNG is seeded so the objective is
deterministic.

Sources:
C. Oh, J. M. Tomczak, E. Gavves, M. Welling. Combinatorial Bayesian Optimization using the Graph Cartesian Product. Advances in Neural Information Processing Systems 32, 2019. https://github.com/QUVA-Lab/COMBO
L. Papenmeier, L. Nardi. Bencher: simple and reproducible benchmarking for black-box optimization. 2025 (wraps the COMBO pest-control benchmark).
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem

_N_STAGES = 25
_N_CHOICE = 5  # 0 = no control; 1-4 = pesticide types


def _pest_spread(curr, spread_rate, control_rate, apply_control):
    if apply_control:
        return (1.0 - control_rate) * curr
    return spread_rate * (1.0 - curr) + curr


def _pest_control_score(x: np.ndarray, rng: np.random.RandomState) -> float:
    """COMBO's pest-control cost for one action vector ``x`` in {0..4}^25."""
    U = 0.1
    n_stages = x.size
    n_sim = 100
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0.0
    above_threshold = 0.0
    curr = rng.beta(1.0, 30.0, size=n_sim)
    for i in range(n_stages):
        spread_rate = rng.beta(1.0, 17.0 / 3.0, size=n_sim)
        if x[i] > 0:
            control_rate = rng.beta(1.0, control_beta[x[i]], size=n_sim)
            nxt = _pest_spread(curr, spread_rate, control_rate, True)
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            payed_price_sum += control_price[x[i]] * (
                1.0
                - control_price_max_discount[x[i]]
                / float(n_stages)
                * float(np.sum(x == x[i]))
            )
        else:
            nxt = _pest_spread(curr, spread_rate, 0.0, False)
        above_threshold += float(np.mean(curr > U))
        curr = nxt
    return payed_price_sum + above_threshold


class PestControl(BenchmarkProblem):
    """Minimize 25-stage pest-control cost (25 categorical vars, 5 choices each)."""

    available_dimensions = _N_STAGES
    num_objectives = 1
    num_constraints = 0

    def __init__(self) -> None:
        self.variable_types = [list(range(_N_CHOICE))] * _N_STAGES
        super().__init__(
            dim=_N_STAGES,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0, _N_CHOICE - 1)] * _N_STAGES,
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy()
        scores = np.empty(x.shape[0])
        for r in range(x.shape[0]):
            xi = np.rint(x[r]).astype(int).clip(0, _N_CHOICE - 1)
            # fixed seed -> deterministic objective (common random numbers across points)
            scores[r] = _pest_control_score(xi, np.random.RandomState(0))
        fx = torch.tensor(-scores, dtype=torch.float64).reshape(-1, 1)  # maximize -cost
        return None, fx
