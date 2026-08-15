"""Cardinality-constrained portfolio optimization — OR-Library ``port2`` / ``port5``.

Beasley's OR-Library "portfolio optimisation" data set (weekly returns, March 1992 –
September 1997, of five real stock indices), introduced with the cardinality-constrained
mean-variance model of Chang et al. (2000) and reused by hundreds of follow-up papers.
Two instances are exposed here, the two largest ones:

- :class:`PortfolioPort2` — DAX 100, ``N = 85`` assets  -> 170-D;
- :class:`PortfolioPort5` — Nikkei 225, ``N = 225`` assets -> 450-D.

**Data format** (verified against ``portinfo.html`` and against the shipped files:
``port2.txt`` has ``1 + 85 + 85*86/2 = 3741`` data lines and ``port5.txt``
``1 + 225 + 225*226/2 = 25651`` — both match exactly)::

    line 1              : N, the number of assets
    next N lines        : mean return_i , standard deviation of return_i
    next N(N+1)/2 lines : i  j  correlation(i, j)      (1-based, i <= j)

The covariance is reconstructed as ``Sigma_ij = rho_ij * s_i * s_j``. Both files are
shipped in ``Portfolio_Data/`` (467 KB total); ``BOCODE_ORLIB_PORTFOLIO_DIR`` overrides
the directory they are read from.

**Mixed encoding** (``2N`` dimensions, ``N`` binary + ``N`` continuous)::

    dims 0     .. N-1  : z_i in {0, 1}, "is asset i selected?"   (binary categorical)
    dims N     .. 2N-1 : raw weight r_i in [0, 1]                (continuous)

Decoding: ``w_i = z_i * r_i / sum_{j: z_j = 1} r_j`` — the weights are renormalized over
the *selected* assets, so ``sum_i w_i = 1`` exactly and the budget constraint holds by
construction. This is the standard "repair" of the metaheuristic portfolio literature.
If every selected raw weight is zero the weights fall back to uniform-over-selected; if
nothing is selected at all the design is fully penalized.

**Objective** (Chang et al. Eq. (1) with ``lambda = 0.5``), minimized::

    f(w) = lambda * w' Sigma w  -  (1 - lambda) * mu' w

**Constraints as additive penalties** (Chang et al. use ``K = 10``, ``eps = 0.01``,
``delta = 1``), so the problem is an unconstrained box in BoCoDe terms::

    cardinality    sum_i z_i = K          ->  RHO_CARD  * |sum_i z_i - K|
    floor/ceiling  eps <= w_i <= delta    ->  RHO_BOUND * sum_i violation_i

``evaluate()`` returns ``-(f + penalty)`` because BoCoDe maximizes.

*Penalty calibration* (fixed constants, recorded for reproducibility): measured over
2000 random *feasible* (exactly-``K``) designs the objective spans ``2.99e-3`` (port2) /
``3.13e-3`` (port5). ``RHO_CARD = 1e-3`` therefore makes one asset of cardinality
deviation cost about a third of the whole feasible objective range — a strong but smooth
pull towards ``|z| = K`` that does not flatten the weight landscape — and
``RHO_BOUND = 1e-2`` makes a full floor violation (``w_i = 0`` instead of ``0.01``) cost
``1e-4``. Unrestricted random designs select ~``N/2`` assets and therefore score around
``-0.02`` (port2) / ``-0.10`` (port5), far below any feasible design.

Sources:
T.-J. Chang, N. Meade, J. E. Beasley, Y. M. Sharaiha. Heuristics for cardinality constrained portfolio optimisation. Computers & Operations Research 27(13):1271-1302, 2000.
J. E. Beasley. OR-Library: distributing test problems by electronic mail. Journal of the Operational Research Society 41(11):1069-1072, 1990. Portfolio instances: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ...base import BenchmarkProblem

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Portfolio_Data")

_LAMBDA = 0.5  # risk / return trade-off of Chang et al. Eq. (1)
_K = 10  # target cardinality
_EPS = 0.01  # floor on a selected asset's weight
_DELTA = 1.0  # ceiling on a selected asset's weight
_RHO_CARD = 1e-3  # penalty per unit of cardinality deviation
_RHO_BOUND = 1e-2  # penalty per unit of floor/ceiling violation


def _parse_orlib_portfolio(path: str):
    """Parse an OR-Library ``portN.txt`` into ``(mu (N,), Sigma (N, N))``."""
    with open(path) as fh:
        tok = fh.read().split()
    pos = 0
    n = int(tok[pos])
    pos += 1
    mu = np.empty(n, dtype=np.float64)
    sd = np.empty(n, dtype=np.float64)
    for i in range(n):
        mu[i] = float(tok[pos])
        sd[i] = float(tok[pos + 1])
        pos += 2
    rho = np.eye(n, dtype=np.float64)
    for _ in range(n * (n + 1) // 2):
        i = int(tok[pos]) - 1
        j = int(tok[pos + 1]) - 1
        r = float(tok[pos + 2])
        pos += 3
        rho[i, j] = r
        rho[j, i] = r
    if pos != len(tok):
        raise ValueError(f"{path}: {len(tok) - pos} trailing tokens (format mismatch)")
    return mu, rho * np.outer(sd, sd)


class _PortfolioProblem(BenchmarkProblem):
    """Shared implementation of the OR-Library cardinality-constrained portfolio.

    Subclasses set ``stem`` (the OR-Library file stem) and ``n_assets``.
    """

    num_objectives = 1
    num_constraints = 0

    stem: str = ""
    n_assets: int = 0

    def __init__(self) -> None:
        n = self.n_assets
        # N binary selectors + N continuous raw weights, both on [0, 1].
        self.variable_types = [[0.0, 1.0]] * n + ["continuous"] * n
        super().__init__(
            dim=2 * n,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0.0, 1.0)] * (2 * n),
        )
        self._mu = None
        self._sigma = None

    # -- data (lazy: keeps construction cheap for the metadata generator) --------
    def _data(self):
        if self._mu is None:
            directory = os.environ.get("BOCODE_ORLIB_PORTFOLIO_DIR", _DATA_DIR)
            path = os.path.join(directory, f"{self.stem}.txt")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"OR-Library instance {self.stem}.txt not found at {path}. It ships "
                    "with BoCoDe in bocode/opt_problems/engineering/Portfolio_Data/; set "
                    "BOCODE_ORLIB_PORTFOLIO_DIR to read it from elsewhere (files are at "
                    "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/)."
                )
            mu, sigma = _parse_orlib_portfolio(path)
            if mu.shape[0] != self.n_assets:
                raise RuntimeError(
                    f"{path}: expected {self.n_assets} assets, found {mu.shape[0]}"
                )
            self._mu, self._sigma = mu, sigma
        return self._mu, self._sigma

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        X = self.enforce_variable_types(X.to(torch.float64))
        x = X.detach().cpu().numpy().astype(np.float64)
        mu, sigma = self._data()
        n = self.n_assets
        z = x[:, :n] > 0.5
        raw = np.clip(x[:, n:], 0.0, 1.0)

        out = np.empty(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            sel = z[i]
            k = int(sel.sum())
            if k == 0:
                # Nothing selected: no portfolio exists. Penalize the full cardinality
                # deviation on a zero-return, zero-risk portfolio.
                out[i] = -(_RHO_CARD * _K)
                continue
            w = np.zeros(n, dtype=np.float64)
            r = raw[i] * sel
            s = r.sum()
            w[sel] = (r[sel] / s) if s > 0 else (1.0 / k)
            f = _LAMBDA * float(w @ sigma @ w) - (1.0 - _LAMBDA) * float(mu @ w)
            wv = w[sel]
            pen = _RHO_CARD * abs(k - _K) + _RHO_BOUND * (
                np.maximum(0.0, _EPS - wv).sum() + np.maximum(0.0, wv - _DELTA).sum()
            )
            out[i] = -(f + pen)

        fx = torch.tensor(out, dtype=torch.float64).reshape(-1, 1)  # maximize
        return None, fx


class PortfolioPort2(_PortfolioProblem):
    """DAX 100 cardinality-constrained portfolio (OR-Library ``port2``), 170-D mixed.

    85 binary asset selectors + 85 continuous raw weights; maximize
    ``-(0.5 * risk - 0.5 * return + cardinality/bound penalties)``. See the module
    docstring for the encoding and the penalty calibration.
    """

    available_dimensions = 170
    num_objectives = 1
    num_constraints = 0

    tags = {"single_objective", "unconstrained", "170D", "mixed", "high_dimensional"}

    stem = "port2"
    n_assets = 85


class PortfolioPort5(_PortfolioProblem):
    """Nikkei 225 cardinality-constrained portfolio (OR-Library ``port5``), 450-D mixed.

    225 binary asset selectors + 225 continuous raw weights; maximize
    ``-(0.5 * risk - 0.5 * return + cardinality/bound penalties)``. See the module
    docstring for the encoding and the penalty calibration.
    """

    available_dimensions = 450
    num_objectives = 1
    num_constraints = 0

    tags = {"single_objective", "unconstrained", "450D", "mixed", "high_dimensional"}

    stem = "port5"
    n_assets = 225
