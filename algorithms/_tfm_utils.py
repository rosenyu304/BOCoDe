"""Shared utilities for the transformer-foundation-model (TFM) BO algorithms.

These algorithms use a pretrained tabular foundation model (TabPFN) as the
surrogate instead of a Gaussian process. **Stock TabPFN is enough**: since v3 the
bar-distribution acquisitions GIT-BO and PFN-CEI rely on (``ei``/``pi``/``ucb``/
``mean``/``variance``) are upstream, so the old TabPFN fork is no longer needed.
Verified against ``tabpfn 8.0.6`` (v3). The weights are downloaded from Hugging Face
on first use -- which needs a Prior-Labs licence token -- so point
``BOCODE_TABPFN_CKPT`` at a local ``.ckpt`` to skip that and to pin the exact weights
a batch run used. Inference runs on GPU if available (CPU otherwise, slower).

Contents:
- ``TabPFNSurrogate``: a thin inference-only wrapper around TabPFN that evaluates
  many candidate "contexts" in parallel (the model's batch dimension) and exposes
  posterior mean / variance / EI / PI / UCB from the bar distribution.
- ``TabPFNModel``: ``TabPFNSurrogate`` dressed up as a **BoTorch multi-output**
  ``Model``, so BoTorch's stock acquisition functions (qLogNEHVI, qLogNEI, ...) run
  on TabPFN unchanged. Used by the TFM multi-objective algorithms.
- ``gradient_information_matrix`` / ``select_rank`` / ``sample_in_subspace``: the
  gradient-informed active-subspace machinery of GIT-BO, with two rank rules —
  a fixed rank (default GIT-BO) and the **Marzouk certified rank** (Zahm, Cui,
  Law, Spantini & Marzouk, Math. Comp. 2022): r* = min{r : sum_{i>r} lambda_i <=
  2*eps/kappa} on the trace-normalised spectrum.

Sources:
N. Hollmann, S. Müller, et al. TabPFN: accurate predictions on small data with a tabular foundation model. Nature, 2025. https://github.com/PriorLabs/TabPFN
R. T.-Y. Yu, C. Picard, F. Ahmed. GIT-BO: high-dimensional Bayesian optimization with tabular foundation models. https://arxiv.org/abs/2505.20685
O. Zahm, T. Cui, K. Law, A. Spantini, Y. Marzouk. Certified dimension reduction in nonlinear Bayesian inverse problems. Mathematics of Computation 91:1789-1835, 2022.
BoTorch's own PFN-as-a-Model wrapper, the template for ``TabPFNModel``: ``botorch_community.models.prior_fitted_network.PFNModel``. https://github.com/meta-pytorch/botorch/blob/main/botorch_community/models/prior_fitted_network.py
"""

from __future__ import annotations

import os

import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from linear_operator.operators import DiagLinearOperator


def _require_tabpfn():
    try:
        import tabpfn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "TFM algorithms (GIT-BO, PFN-CEI) need TabPFN (>= v3, e.g. tabpfn 8.0.6), "
            "which carries the bar-distribution acquisitions (ei/pi/ucb). Install it in "
            "a dedicated environment and set BOCODE_TABPFN_CKPT to a local .ckpt to "
            "avoid the Hugging Face licence token. See docs/tfm_setup.md."
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
        import inspect

        from tabpfn.base import initialize_tabpfn_model

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # TabPFN v2 (the GIT-BO fork) and v3 (upstream, which now carries the ei/ucb
        # bar-distribution acquisitions we need) disagree on this call: v3 dropped
        # ``static_seed``, renamed the task to "regressor", and returns lists of
        # architectures plus an extra inference config. Support both.
        # "auto" makes TabPFN v3 fetch the checkpoint from HuggingFace, which needs a
        # Prior-Labs licence token. Point BOCODE_TABPFN_CKPT at a local .ckpt to skip that
        # (and to pin the exact weights a batch run used).
        model_path = os.environ.get("BOCODE_TABPFN_CKPT", "auto")

        params = inspect.signature(initialize_tabpfn_model).parameters
        kwargs = {"model_path": model_path, "fit_mode": "fit_preprocessors"}
        kwargs["which"] = "regressor" if "static_seed" not in params else "regression"
        if "static_seed" in params:
            kwargs["static_seed"] = 0
        out = initialize_tabpfn_model(**kwargs)
        model, config, self.bardist = out[0], out[1], out[2]
        self.model = model[0] if isinstance(model, list) else model
        self.config = config[0] if isinstance(config, list) else config

        # v3's forward() dropped ``single_eval_pos`` (it infers the split from len(y))
        # and returns only the query rows; v2's took it and returned every row.
        self._v3 = (
            "single_eval_pos" not in inspect.signature(self.model.forward).parameters
        )
        self.model.to(self.device).eval()
        self.bardist.borders = self.bardist.borders.to(self.device)
        self._y_mean = None
        self._y_std = None

    def forward(self, X: torch.Tensor, Y: torch.Tensor, single_eval_pos: int):
        """Run TabPFN; returns the bar-distribution logits for the CANDIDATE rows.

        ``X`` holds context rows followed by candidate rows; only ``Y[:single_eval_pos]``
        (the observed targets) conditions the model.
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        y_ctx = Y[:single_eval_pos]
        self._y_mean = y_ctx.mean(dim=0, keepdim=True)
        self._y_std = y_ctx.std(dim=0, keepdim=True) + 1e-9
        y_norm = (y_ctx - self._y_mean) / self._y_std

        if self._v3:
            # v3 infers the context/query split from len(y) and returns only the query
            # rows. This mirrors the reference notebook
            # (Knowledge_Base/MILP/vanillatabpfn_script_tabpfnv3.ipynb,
            # ``VanillaDirectTabPFNRegressor.forward``): full X, train-only normalized Y,
            # no ``single_eval_pos``/``style``, and the logits read out of
            # ``out["standard"]``. ``categorical_inds`` is left at None -- upstream
            # indexes it per batch item (transformer.py:411-418), so a single ``[[]]``
            # raises IndexError as soon as the batch has more than one column (PFN-CEI
            # stacks 1 + n_constraints columns).
            # Cast to the model's float32; the cast is differentiable, so gradients still
            # reach the float64 candidate leaf that GIT-BO needs.
            out = self.model(
                X.float(),
                y_norm.float(),
                only_return_standard_out=False,
            )
            return out["standard"]

        Y_std = Y.clone()
        Y_std[:single_eval_pos] = y_norm
        out = self.model(
            None,
            X,
            Y_std,
            single_eval_pos=single_eval_pos,
            only_return_standard_out=False,
        )
        logits = out["standard"]
        # v2 returns every row; keep the same contract as v3 (candidate rows only).
        return logits[single_eval_pos:] if logits.shape[0] == X.shape[0] else logits

    def predict_mean(self, logits):
        m = self.bardist.mean(logits)
        return m * self._y_std.squeeze(-1) + self._y_mean.squeeze(-1)

    def predict_variance(self, logits):
        return self.bardist.variance(logits) * self._y_std.squeeze(-1) ** 2

    def _best_f(self, logits, best_f):
        """Standardize ``best_f`` and broadcast it to one value per (candidate, column).

        ``best_f`` is in the original target scale and is either a scalar or one value
        per output column; it is standardized with the same context statistics as the
        targets, so it lands in the bar distribution's space.
        """
        best_f = torch.as_tensor(best_f, device=self._y_mean.device)
        bf = (best_f - self._y_mean.squeeze(-1)) / self._y_std.squeeze(-1)
        # the bar distribution asserts best_f matches logits[..., 0] exactly, so a
        # single incumbent has to be expanded across the candidate rows.
        return bf.to(logits.dtype).expand(logits.shape[:-1])

    def predict_ei(self, logits, best_f):
        # best_f given in original scale -> standardize for the bar distribution
        return self.bardist.ei(logits, self._best_f(logits, best_f))

    def predict_pi(self, logits, threshold):
        """P(y > threshold) under the bar distribution (threshold in original scale).

        PFN-CEI's feasibility factor: with constraints negated so that feasible means
        "large", this is exactly P(c <= 0). Uses the model's own (non-Gaussian)
        predictive distribution rather than a Gaussian approximation of it.
        """
        return self.bardist.pi(logits, self._best_f(logits, threshold))

    def predict_sample(self, logits) -> torch.Tensor:
        """One independent draw from each candidate's bar distribution (Thompson sampling).

        Vectorized ``BarDistribution.sample``: this is upstream's own inverse-CDF sampler,
        ``icdf(logits, u)`` with ``u ~ U(0, 1)``, written as one batched op. Upstream's
        ``sample`` is a **Python loop over the evaluation points**, which is unusable for the
        thousands of trust-region candidates TuRBO/SCBO score each iteration. The arithmetic
        is the same: find the bucket where the cumulative mass reaches ``u``, then interpolate
        linearly inside it. De-standardized back to the Y scale, like ``predict_mean``.

        ``u`` is drawn on the **CPU** generator on purpose: ``_bo_utils.save_checkpoint``
        persists ``torch.get_rng_state()`` (the CPU RNG) only, so drawing on the CUDA
        generator would make a resumed run diverge from an uninterrupted one.

        Note the draws are **independent per candidate** (marginal), because TabPFN's bar
        distribution only gives a marginal predictive per row -- there is no joint
        posterior to draw a correlated sample path from, as a GP's Thompson sampling does.
        """
        probs = logits.softmax(-1)
        cum = probs.cumsum(-1).contiguous()
        u = torch.rand(probs.shape[:-1], dtype=probs.dtype).to(probs.device)
        idx = (
            torch.searchsorted(cum, u.unsqueeze(-1))
            .squeeze(-1)
            .clamp(0, probs.shape[-1] - 1)
        )
        cum0 = torch.cat([torch.zeros_like(cum[..., :1]), cum], dim=-1)
        rest = u - cum0.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        # 0 <= rest <= p[idx] by construction; clamp only to keep the division finite when a
        # bucket carries (almost) no mass -- upstream divides unguarded.
        p_idx = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
        lo, hi = self.bardist.borders[idx], self.bardist.borders[idx + 1]
        z = lo + (hi - lo) * rest / p_idx
        return z * self._y_std.squeeze(-1) + self._y_mean.squeeze(-1)

    def predict_ucb(self, logits, best_f=0.0, rest_prob=(1 - 0.682) / 2):
        """TabPFN's **built-in** bar-distribution UCB, de-standardized to the Y scale.

        ``BarDistribution.ucb`` is a *quantile* UCB: it returns the ``1 - rest_prob``
        quantile of the model's own (non-Gaussian) predictive distribution, i.e.
        ``icdf(logits, 1 - rest_prob)`` -- not ``mu + beta * sigma``. Upstream's docstring
        gives the Gaussian-equivalent exploration level as
        ``beta = sqrt(2) * erfinv(2 * (1 - rest_prob) - 1)``, so ``rest_prob = 1 - Phi(beta)``.

        ``best_f`` is accepted for parity with the reference notebook's
        ``predict_ucb(logits, best_f, percentile)`` but is **ignored by TabPFN's ucb**
        (upstream marks the argument unused).

        The quantile lives in the bar distribution's standardized space; de-standardizing
        is a positive affine map, so it leaves the argmax unchanged and puts the reported
        acquisition value in the same units as ``predict_mean``.
        """
        u = self.bardist.ucb(logits=logits, best_f=best_f, rest_prob=rest_prob)
        return u * self._y_std.squeeze(-1) + self._y_mean.squeeze(-1)


class TabPFNModel(Model):
    """TabPFN as a **BoTorch multi-output** ``Model``: stock acquisitions work unchanged.

    Modeled on BoTorch's own PFN-as-a-Model wrapper
    (``botorch_community.models.prior_fitted_network.PFNModel``), which cannot be reused
    directly here: it is hard-wired to the ``pfns4bo`` checkpoints (its module imports
    ``pfns.train``, not installed) and, more importantly, it is **single-output** -- it
    returns a ``BoundedRiemannPosterior``, which BoTorch's multi-objective machinery
    cannot sample jointly across outputs. So we keep its contract (a ``Model`` whose
    ``posterior()`` conditions the frozen network on the training data in-context) and
    change the posterior type.

    **Multi-output.** TabPFN is single-output, so each of the ``m`` outputs (the
    objectives, followed by the constraints in the constrained variants) gets its own
    TabPFN predictor -- the ``ModelListGP`` pattern. They are not ``m`` separate forward
    passes though: the same frozen network scores all ``m`` outputs in **one** pass by
    putting them in TabPFN's *batch* dimension (identical ``X`` columns, one ``Y`` column
    per output), the same trick PFN-CEI uses for objective + constraints. Each output is
    conditioned and standardized independently, exactly as independent models would be.

    **The posterior.** ``posterior(X)`` returns a ``GPyTorchPosterior`` over a
    ``MultitaskMultivariateNormal`` built from the per-output *mean* and *variance* of
    TabPFN's bar distribution, with a **diagonal** covariance. Two approximations, both
    inherent to a PFN surrogate and both stated plainly:

    * *Gaussian*: the bar distribution is discretized and non-Gaussian; we keep only its
      first two moments. This is the price of using the stock MC acquisitions, which
      sample a ``Posterior`` -- the moments are TabPFN's own, not a refit Gaussian.
    * *Diagonal*: a PFN gives marginal predictives, never a joint covariance across query
      points, so points are independent under this posterior (a GP's would be
      correlated). qNEHVI/qNEI stay valid MC estimators; they just see no cross-point
      correlation. For the same reason the acquisitions must be built with
      ``cache_root=False``: the cached-Cholesky path exists to exploit a joint GP
      covariance this model does not have.

    Nothing is fitted: TabPFN is frozen and conditions on ``(train_X, train_Y)``
    in-context at every ``posterior()`` call, so there is no ``fit_gpytorch_mll`` step.
    Inputs are expected to already live in the unit cube (BoCoDe's search space), and the
    targets are standardized inside :class:`TabPFNSurrogate` on the context block, so no
    ``Normalize`` / ``Standardize`` transforms are needed.

    Signs are BoCoDe's: ``train_Y`` is the maximization-frame objective (and, in the
    constrained variants, the raw constraints, feasible when ``<= 0``). The model applies
    no negation whatsoever, so the acquisition sees exactly the values the problem
    returned.

    Args:
        surrogate: a loaded :class:`TabPFNSurrogate`.
        train_X: ``(n, d)`` observed inputs (unit cube).
        train_Y: ``(n, m)`` observed outputs, ``m >= 2`` (objectives, then constraints).
    """

    def __init__(
        self,
        surrogate: TabPFNSurrogate,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
    ):
        super().__init__()
        if train_Y.ndim != 2 or train_Y.shape[-1] < 2:
            raise ValueError(
                "TabPFNModel is the multi-output wrapper and needs train_Y of shape "
                f"(n, m) with m >= 2; got {tuple(train_Y.shape)}. For a single output "
                "use TabPFNSurrogate directly (see git_bo / pfn_cei)."
            )
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must have the same number of rows.")
        self.surrogate = surrogate
        self.train_X = train_X
        self.train_Y = train_Y
        self._num_outputs = train_Y.shape[-1]

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    def posterior(
        self,
        X: torch.Tensor,
        output_indices=None,
        observation_noise=False,
        posterior_transform=None,
        **kwargs,
    ) -> GPyTorchPosterior:
        """Independent per-output TabPFN predictives at ``X`` as one BoTorch posterior.

        ``X`` is ``(..., q, d)``; the returned posterior has event shape ``(..., q, m)``.
        Gradients flow back to ``X`` (TabPFN's float32 cast is differentiable), which is
        what lets ``optimize_acqf`` run its usual gradient-based multi-start on this
        model.
        """
        if output_indices is not None:
            raise NotImplementedError("TabPFNModel does not support output_indices.")
        m, n = self._num_outputs, self.train_X.shape[0]
        lead, d = X.shape[:-1], X.shape[-1]
        X_q = X.reshape(-1, d)
        n_q = X_q.shape[0]

        # One TabPFN batch column per output: same X, one Y column each. The candidate
        # rows' Y is ignored (they sit past single_eval_pos), so zeros are fine.
        X_full = (
            torch.cat([self.train_X.to(X_q), X_q], dim=0).unsqueeze(1).expand(-1, m, -1)
        )
        Y_ctx = self.train_Y.to(X_q)
        Y_full = torch.cat(
            [Y_ctx, torch.zeros(n_q, m, dtype=X_q.dtype, device=X_q.device)], dim=0
        ).unsqueeze(-1)

        logits = self.surrogate.forward(
            X_full, Y_full, single_eval_pos=n
        )  # (n_q, m, b)
        mean = self.surrogate.predict_mean(logits).to(X.dtype).reshape(*lead, m)
        var = (
            self.surrogate.predict_variance(logits)
            .to(X.dtype)
            .reshape(*lead, m)
            .clamp_min(1e-9)  # a degenerate (constant) output would give var == 0
        )

        # Independent outputs with a diagonal covariance over the q points, combined the
        # way ModelListGPyTorchModel combines its per-output GPs.
        mvns = [
            MultivariateNormal(mean[..., i], DiagLinearOperator(var[..., i]))
            for i in range(m)
        ]
        posterior = GPyTorchPosterior(
            MultitaskMultivariateNormal.from_independent_mvns(mvns)
        )
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior


def gradient_information_matrix(grads: np.ndarray) -> np.ndarray:
    """Gradient-information (surrogate-Fisher) matrix H = (1/n) sum_i g_i g_i^T."""
    n = grads.shape[0]
    return (grads.T @ grads) / n


def select_rank(
    eigenvalues: np.ndarray,
    mode: str = "fixed",
    rank: int = 10,
    eps: float = 0.05,
    kappa: float = 1.0,
) -> int:
    """Choose the active-subspace rank from the (descending) eigenvalue spectrum.

    mode='fixed'   -> return ``rank`` (capped at the dimension). GIT-BO uses a fixed
                      r = 10 for every experiment (paper Appendix G).
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
