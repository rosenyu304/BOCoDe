"""Shared utilities for the TabICL-surrogate BO algorithms.

These algorithms use a pretrained tabular foundation model (TabICL) as the surrogate
instead of a Gaussian process, mirroring the TabPFN-based ``tfm_*`` methods (see
:mod:`algorithms._tfm_utils`). TabICL is reported ~3x faster than TabPFN with a very
similar in-context-learning API.

**Feasibility.** BO needs a predictive *mean* AND *uncertainty* for EI/UCB. TabICL
ships a native ``TabICLRegressor`` (``tabicl >= 2``) that exposes exactly this: one
``predict`` call returns the predictive ``mean``, ``variance``, and a grid of
``quantiles`` of its predictive distribution (internally a ``QuantileDistribution``,
the moral equivalent of TabPFN's bar/Riemann distribution over a binned target). The
quantile grid is the primary uncertainty signal here -- in a held-out check it stays
tight in-data and widens sharply out-of-distribution, exactly as a calibrated
predictive should -- so EI, quantile-UCB and Thompson sampling are all read off the
grid, the same way the ``tfm_*`` methods read them off TabPFN's bar distribution.

Contents:
- ``TabICLSurrogate``: a thin fit/predict wrapper around ``TabICLRegressor`` that
  exposes the predictive mean/variance and computes quantile-UCB, EI, and Thompson
  samples from a fixed equal-probability quantile grid of the predictive distribution.

Source:
J. Qu, D. Holzmüller, G. Varoquaux, M. Le Morvan. TabICL: A Tabular Foundation Model
for In-Context Learning on Large Data. ICML 2025. https://github.com/soda-inria/tabicl
"""

from __future__ import annotations

import bisect
import math

import torch

DTYPE = torch.double

#: Resolution of the equal-probability quantile grid used to represent the predictive
#: distribution for EI / Thompson sampling. Grid level k is ``(k + 0.5) / K``, so the K
#: quantiles are the midpoints of K equal-mass bins and a plain mean over them is an
#: unbiased Monte-Carlo estimate of any expectation under the predictive (e.g. EI).
QUANTILE_K = 64

#: Chunk the candidate rows through TabICL so a large pool does not build one giant
#: forward. Query rows attend only to the (fixed) context, so scoring them in chunks
#: returns the same per-row predictions.
MAX_CAND_PER_PASS = 2000


def _require_tabicl():
    try:
        import tabicl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "TabICL algorithms need TabICL (>= 2, e.g. tabicl 2.1.1), which carries the "
            "native TabICLRegressor with mean/variance/quantile predictive outputs. "
            "Install it with `pip install tabicl`."
        ) from exc


class TabICLSurrogate:
    """Inference-only TabICL regressor exposing mean/variance/UCB/EI/samples.

    Wraps a native :class:`tabicl.TabICLRegressor`. Each optimization iteration calls
    :meth:`fit` on the observed ``(X, y)`` (TabICL re-encodes the context in-context;
    nothing is gradient-trained) and then :meth:`score` on a candidate pool, which
    returns the predictive mean, variance, and an equal-probability quantile grid of
    the predictive distribution. Quantile-UCB, EI, and Thompson samples are computed
    from that grid, the same way the ``tfm_*`` methods read them off TabPFN's bar
    distribution.

    ``X`` are unit-cube inputs and ``y`` the maximization-frame objective (BoCoDe's
    sign convention); TabICL standardizes the target internally, so no external
    Normalize/Standardize is needed.
    """

    def __init__(self, device: str = "auto", seed: int = 0, n_estimators: int = 1):
        # n_estimators=1 (single forward, no ensemble) to match the TabPFN wrapper's
        # single-pass cost -- the default 8-member ensemble made TabICL ~8x slower per
        # iteration (8 model evals + context re-encoding) than tfm_* on the same problem.
        _require_tabicl()
        from tabicl import TabICLRegressor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.reg = TabICLRegressor(
            device=str(self.device),
            random_state=int(seed),
            n_estimators=int(n_estimators),
            allow_auto_download=True,
        )
        # Equal-probability quantile levels: midpoints of K equal-mass bins.
        self._alphas = [(k + 0.5) / QUANTILE_K for k in range(QUANTILE_K)]
        self._alpha_t = torch.tensor(self._alphas, dtype=DTYPE)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> TabICLSurrogate:
        """Condition TabICL on the observed data ``(X, y)`` (in-context, no training)."""
        Xn = X.detach().cpu().to(DTYPE).numpy()
        yn = y.detach().cpu().to(DTYPE).reshape(-1).numpy()
        self.reg.fit(Xn, yn)
        return self

    def score(self, cand: torch.Tensor):
        """Predictive mean, variance, and quantile grid for a candidate pool.

        Returns ``(mean, var, q)`` with ``mean``/``var`` of shape ``(n,)`` and ``q`` the
        quantile grid of shape ``(n, QUANTILE_K)`` (increasing along the last axis, at
        the levels in ``self._alphas``). One TabICL ``predict`` call yields all three.
        """
        means, vars_, qs = [], [], []
        for i in range(0, cand.shape[0], MAX_CAND_PER_PASS):
            Xn = cand[i : i + MAX_CAND_PER_PASS].detach().cpu().to(DTYPE).numpy()
            out = self.reg.predict(
                Xn, output_type=["mean", "variance", "quantiles"], alphas=self._alphas
            )
            means.append(torch.as_tensor(out["mean"], dtype=DTYPE).reshape(-1))
            vars_.append(torch.as_tensor(out["variance"], dtype=DTYPE).reshape(-1))
            qs.append(torch.as_tensor(out["quantiles"], dtype=DTYPE))
        return (
            torch.cat(means),
            torch.cat(vars_).clamp_min(0.0),
            torch.cat(qs, dim=0),
        )

    def ucb_from_quantiles(self, q: torch.Tensor, beta: float) -> torch.Tensor:
        """Quantile UCB: the ``Phi(beta)`` quantile of the predictive distribution.

        The direct analog of TabPFN's built-in ``BarDistribution.ucb`` -- a *quantile*
        UCB, not ``mu + beta*sigma`` -- read off TabICL's own (non-Gaussian) predictive.
        ``beta`` is the Gaussian-equivalent exploration level, so the probability level
        is ``Phi(beta)``.
        """
        level = 0.5 * (1.0 + math.erf(beta / math.sqrt(2.0)))
        return self._interp_quantile(q, level)

    def ei_from_quantiles(self, q: torch.Tensor, best_f: float) -> torch.Tensor:
        """Expected improvement ``E[max(Y - best_f, 0)]`` under the predictive.

        Monte-Carlo estimate over the equal-probability quantile grid: because the grid
        levels are the midpoints of K equal-mass bins, a plain mean over the grid is an
        unbiased estimate of the expectation. Uses TabICL's own predictive (no Gaussian
        assumption), the quantile analog of TabPFN's ``BarDistribution.ei``.
        """
        return (q - best_f).clamp_min(0.0).mean(dim=1)

    def sample_from_quantiles(self, q: torch.Tensor) -> torch.Tensor:
        """One independent draw from each candidate's predictive (Thompson sampling).

        Inverse-CDF sampler on the equal-probability quantile grid: draw ``u ~ U(0, 1)``
        and linearly interpolate the grid at ``u`` (grid point ``k`` sits at probability
        level ``(k + 0.5)/K``). Draws are independent per candidate -- TabICL gives a
        marginal predictive per row, no joint posterior -- exactly like TabPFN's
        ``predict_sample``.

        ``u`` is drawn on the global (CPU) generator on purpose: ``save_checkpoint``
        persists ``torch.get_rng_state()`` (the CPU RNG), so a resumed run matches an
        uninterrupted one.
        """
        n, K = q.shape
        u = torch.rand(n, dtype=DTYPE)
        pos = (u * K - 0.5).clamp(0.0, K - 1.0)
        j = pos.floor().long().clamp(0, K - 2)
        w = (pos - j.to(DTYPE)).clamp(0.0, 1.0)
        lo = q.gather(1, j.unsqueeze(1)).squeeze(1)
        hi = q.gather(1, (j + 1).unsqueeze(1)).squeeze(1)
        return lo * (1.0 - w) + hi * w

    def _interp_quantile(self, q: torch.Tensor, level: float) -> torch.Tensor:
        """Linear interpolation of the quantile grid at probability ``level``."""
        a = self._alphas
        level = min(max(level, a[0]), a[-1])
        j = min(max(bisect.bisect_right(a, level) - 1, 0), len(a) - 2)
        lo, hi = a[j], a[j + 1]
        w = (level - lo) / (hi - lo) if hi > lo else 0.0
        return q[:, j] * (1.0 - w) + q[:, j + 1] * w
