"""Shared utilities for the transformer-foundation-model (TFM) BO algorithms.

These algorithms use a pretrained tabular foundation model (TabPFN) as the
surrogate instead of a Gaussian process. They require a TabPFN fork (see docs/tfm_setup.md), which
installs the TabPFN fork carrying the bar-distribution acquisition methods
(``ei``/``ucb``/``mean``/``variance``) that GIT-BO and PFN-CEI rely on. The model
weights are downloaded from Hugging Face on first use, and inference runs on GPU
if available (CPU otherwise, slower).

Contents:
- ``TabPFNSurrogate``: a thin inference-only wrapper around TabPFN that evaluates
  many candidate "contexts" in parallel (the model's batch dimension) and exposes
  posterior mean / variance / EI / UCB from the bar distribution.
- ``gradient_information_matrix`` / ``select_rank`` / ``sample_in_subspace``: the
  gradient-informed active-subspace machinery of GIT-BO, with two rank rules —
  a fixed rank (default GIT-BO) and the **Marzouk certified rank** (Zahm, Cui,
  Law, Spantini & Marzouk, Math. Comp. 2022): r* = min{r : sum_{i>r} lambda_i <=
  2*eps/kappa} on the trace-normalised spectrum.

Sources:
N. Hollmann, S. Müller, et al. TabPFN: accurate predictions on small data with a tabular foundation model. Nature, 2025. https://github.com/PriorLabs/TabPFN
S. Jiang, et al. GIT-BO: high-dimensional Bayesian optimization with tabular foundation models. https://openreview.net/forum?id=9iTdKS4SRQ
O. Zahm, T. Cui, K. Law, A. Spantini, Y. Marzouk. Certified dimension reduction in nonlinear Bayesian inverse problems. Mathematics of Computation 91:1789-1835, 2022.
"""

from __future__ import annotations

import numpy as np
import torch


def _require_tabpfn():
    try:
        import tabpfn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "TFM algorithms (GIT-BO, PFN-CEI) need the TabPFN fork that carries the "
            "bar-distribution acquisitions (ei/ucb). That fork pins scikit-learn and "
            "Python <3.12, so it does NOT coexist with BoCoDe's core env — run the TFM "
            "algorithms in a dedicated environment. See docs/tfm_setup.md."
        ) from exc


class TabPFNSurrogate:
    """Inference-only TabPFN regressor that scores candidate contexts in parallel.

    The model takes ``X`` of shape ``(context+query, batch, dim)`` and ``Y`` of
    shape ``(context+query, batch, 1)`` with a ``single_eval_pos`` split: the first
    ``single_eval_pos`` rows are the observed (context) data, the rest are queries.
    The ``batch`` dimension lets us evaluate many candidates (or many candidate
    contexts) in a single forward pass. Targets are standardized on the context
    block; predictions are de-standardized back to the original scale.
    """

    def __init__(self, device: str = "auto"):
        _require_tabpfn()
        from tabpfn.base import initialize_tabpfn_model

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.config, self.bardist = initialize_tabpfn_model(
            model_path="auto",
            which="regression",
            fit_mode="fit_preprocessors",
            static_seed=0,
        )
        self.model.to(self.device).eval()
        self.bardist.borders = self.bardist.borders.to(self.device)
        self._y_mean = None
        self._y_std = None

    def forward(self, X: torch.Tensor, Y: torch.Tensor, single_eval_pos: int):
        """Run TabPFN; returns the standard-output logits (bar-distribution params)."""
        X = X.to(self.device)
        Y = Y.to(self.device)
        y_ctx = Y[:single_eval_pos]
        self._y_mean = y_ctx.mean(dim=0, keepdim=True)
        self._y_std = y_ctx.std(dim=0, keepdim=True) + 1e-9
        Y_std = Y.clone()
        Y_std[:single_eval_pos] = (Y[:single_eval_pos] - self._y_mean) / self._y_std
        out = self.model(
            None,
            X,
            Y_std,
            single_eval_pos=single_eval_pos,
            only_return_standard_out=False,
        )
        return out["standard"]

    def predict_mean(self, logits):
        m = self.bardist.mean(logits)
        return m * self._y_std.squeeze(-1) + self._y_mean.squeeze(-1)

    def predict_variance(self, logits):
        return self.bardist.variance(logits) * self._y_std.squeeze(-1) ** 2

    def predict_ei(self, logits, best_f):
        # best_f given in original scale -> standardize for the bar distribution
        bf = (best_f - self._y_mean.squeeze(-1)) / self._y_std.squeeze(-1)
        return self.bardist.ei(logits, bf)

    def predict_ucb(self, logits, best_f, rest_prob=0.05):
        bf = (best_f - self._y_mean.squeeze(-1)) / self._y_std.squeeze(-1)
        return self.bardist.ucb(logits=logits, best_f=bf, rest_prob=rest_prob)


def gradient_information_matrix(grads: np.ndarray) -> np.ndarray:
    """Gradient-information (surrogate-Fisher) matrix H = (1/n) sum_i g_i g_i^T."""
    n = grads.shape[0]
    return (grads.T @ grads) / n


def select_rank(
    eigenvalues: np.ndarray,
    mode: str = "fixed",
    rank: int = 5,
    eps: float = 0.05,
    kappa: float = 1.0,
) -> int:
    """Choose the active-subspace rank from the (descending) eigenvalue spectrum.

    mode='fixed'   -> return ``rank`` (capped at the dimension); the default GIT-BO.
    mode='marzouk' -> the certified rank r* = min{r : sum_{i>r} lambda_i <= 2*eps/kappa}
                      on the trace-normalised spectrum (Zahm-Marzouk 2022). This keeps
                      >= (1 - 2*eps/kappa) of the gradient-information energy, with a
                      KL bound on the subspace approximation.
    """
    d = len(eigenvalues)
    lam = np.clip(np.sort(eigenvalues)[::-1], 0.0, None)
    if mode == "fixed":
        return int(min(max(rank, 1), d))
    if mode == "marzouk":
        total = lam.sum()
        if total <= 1e-300:
            return d
        lam_n = lam / total
        thresh = 2.0 * eps / kappa
        for r in range(1, d + 1):
            if lam_n[r:].sum() <= thresh:
                return r
        return d
    raise ValueError(f"unknown rank mode {mode!r}")


def sample_in_subspace(
    center: np.ndarray,
    grads: np.ndarray,
    n_samples: int,
    rank_mode: str,
    rank: int,
    eps: float,
    scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Sample ``n_samples`` candidates in the top-``r`` gradient-information subspace.

    Returns ``(candidates, r)`` where candidates are clamped to the unit cube and
    ``r`` is the selected rank (fixed or Marzouk-certified).
    """
    H = gradient_information_matrix(grads)
    eigvals, eigvecs = np.linalg.eigh(H)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    r = select_rank(eigvals, mode=rank_mode, rank=rank, eps=eps)
    U_r = eigvecs[:, :r]  # (d, r)
    alpha = rng.uniform(-scale, scale, size=(n_samples, r))
    samples = center[None, :] + alpha @ U_r.T
    return np.clip(samples, 0.0, 1.0), r
