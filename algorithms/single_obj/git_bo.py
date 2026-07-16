"""GIT-BO: Gradient-Informed Bayesian Optimization with a tabular foundation model.

Uses a frozen, pretrained TabPFN as the surrogate (no GP training): the observed data
are the model's in-context "training" rows and candidates are query rows. Each iteration
follows the paper's Algorithm 1:

1. Draw a Sobol candidate set over the domain and run TabPFN in-context on it.
2. Backprop through TabPFN's predictive mean to get ``g(x) = d mu / d x`` on those points.
3. Form the empirical Fisher / gradient-information matrix ``H = E[g(x) g(x)^T]`` and take
   its top-``r`` eigenvectors as the gradient-informed subspace ``V_r``.
4. Generate candidates ``X_GI = x_ref + V_r z`` with ``z ~ U([-1, 1]^r)``, centered on the
   **centroid of the observed data** ``x_ref = mean(X_obs)`` (the paper ablates this against
   centering on ``argmax y`` and finds the centroid better, Appendix B.6).
5. Pick ``argmax`` of **TabPFN's own** ``BarDistribution.ucb`` -- the ``1 - rest_prob``
   quantile of the model's non-Gaussian predictive distribution -- at the exploration
   level of the paper's ``beta = 2.33`` (Sec. 3.3), i.e. ``rest_prob = 1 - Phi(2.33)``.
   Deliberately *not* a hand-rolled ``mu + beta * sigma``, which would Gaussianize away
   the predictive shape the tabular foundation model exists to provide.

Note the gradients are computed on the *Sobol* set and the acquisition on the *subspace*
set, so an iteration costs two TabPFN forward passes.

``--rank`` selects the subspace rank:
- ``10``      — fixed rank, the GIT-BO default used for every experiment in the paper.
- ``marzouk`` — a BoCoDe extension: the **certified rank** r* = min{r : sum_{i>r} lambda_i
               <= 2*eps/kappa} on the trace-normalised spectrum (Zahm, Cui, Law, Spantini &
               Marzouk, Math. Comp. 2022). Not part of the paper.

Needs TabPFN >= v3 (see docs/tfm_setup.md). Run::

    python -m algorithms.single_obj.git_bo --problem Ackley --iters 50
    python -m algorithms.single_obj.git_bo --problem Ackley --iters 50 --rank marzouk

Sources:
R. T.-Y. Yu, C. Picard, F. Ahmed. GIT-BO: High-Dimensional Bayesian Optimization with Tabular Foundation Models. https://arxiv.org/abs/2505.20685
N. Hollmann, S. Müller, et al. TabPFN: accurate predictions on small data with a tabular foundation model. Nature, 2025.
O. Zahm, T. Cui, K. Law, A. Spantini, Y. Marzouk. Certified dimension reduction in nonlinear Bayesian inverse problems. Mathematics of Computation 91:1789-1835, 2022.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .._bo_utils import (
    DTYPE,
    ProblemObjective,
    Result,
    add_common_args,
    default_n_init,
    finalize,
    initial_design,
    load_checkpoint,
    make_problem,
    save_checkpoint,
    set_seed,
)
from .._tfm_utils import TabPFNSurrogate, sample_in_subspace

N_CANDIDATES = 2000  # candidate pool scored per iteration ("m" in the paper)

# How many candidate rows to push through TabPFN per forward pass. The pool is scored in
# chunks rather than all 2000 at once. In TabPFN a query row attends only to the context
# rows, never to another query row, so a candidate's predictive mean -- and its gradient
# d mu / d x -- depends only on the context and on that row. Splitting the pool is
# therefore an EXACT reordering of the same computation, not an approximation.
#
# It has to be chunked because the un-chunked cost is O(N_CANDIDATES * dim): the gradient
# pass keeps the autograd graph over an (n_ctx + 2000) x 1 x dim activation, which on
# AntPolicySearchProblem (dim 840) and LassoLeukemia (dim 7129) asked CUDA for 0.82 and
# 6.82 GiB in a single allocation and OOMed. With chunking, peak memory scales with the
# chunk size and NOT with the pool size, so the method runs on any card.
MAX_CAND_PER_PASS = 512
BETA = 2.33  # UCB exploration level (paper Sec. 3.3)
DEFAULT_RANK = 10  # fixed subspace rank used for every paper experiment (Appendix G)

# The acquisition is TabPFN's **built-in** ``BarDistribution.ucb`` -- a quantile UCB that
# returns the ``1 - rest_prob`` quantile of the model's own non-Gaussian predictive
# distribution -- not a hand-rolled ``mu + beta * sigma``. TabPFN's docstring gives the
# Gaussian-equivalent level as ``beta = sqrt(2) * erfinv(2 * (1 - rest_prob) - 1)``;
# inverting it at the paper's ``beta = 2.33`` gives ``rest_prob = 1 - Phi(2.33) ~ 0.0099``.
REST_PROB = 0.5 * math.erfc(BETA / math.sqrt(2.0))

ANCHOR_SIGMA = 0.1  # Vanilla BO's sample_around_best_sigma (configs/acq_opt/highdim.yaml)


def _sample_around_best(
    x_best: torch.Tensor, n: int, sigma: float, prob_perturb: float
) -> torch.Tensor:
    """Vanilla BO's ``sample_around_best`` trick, as a discrete candidate generator.

    Perturbs the best observed point with truncated-Gaussian noise on a RANDOM SUBSET of
    the dimensions -- each dim is perturbed with probability ``prob_perturb`` (BoTorch's
    default ``min(20/d, 1)``); dims left unperturbed keep the incumbent's value. This is
    exactly ``botorch.utils.sampling.sample_perturbed_subset_dims`` (Regis & Shoemaker's
    subset-perturbation idea that Hvarfner et al. 2024 rely on), reduced to the single
    incumbent case: rows that would perturb nothing get a few forced random dims. The
    Gaussian is truncated to ``[0, 1]`` by clamping.

    ``x_best`` is ``(dim,)`` in ``[0, 1]``; returns ``(n, dim)`` in ``[0, 1]``.
    """
    dim = x_best.shape[0]
    base = x_best.unsqueeze(0).expand(n, dim).clone()
    pert = (base + sigma * torch.randn(n, dim, dtype=x_best.dtype)).clamp(0.0, 1.0)
    mask = torch.rand(n, dim, dtype=x_best.dtype) <= prob_perturb
    n_perturb = max(1, math.ceil(dim * prob_perturb))
    for i in (~mask).all(dim=1).nonzero(as_tuple=True)[0].tolist():
        mask[i, torch.randperm(dim)[:n_perturb]] = True
    return torch.where(mask, pert, base)


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    rank: str = str(DEFAULT_RANK),
    scale: float = 1.0,
    anchor: float = 0.0,
    anchor_sigma: float = ANCHOR_SIGMA,
    acq: str = "gaussian",
    device: str = "auto",
    checkpoint: str | None = None,
) -> Result:
    """GIT-BO over the unit cube. ``n_init`` defaults to the dim-scaled BoCoDe default.

    ``rank`` is an integer (fixed rank, paper default 10) or 'marzouk' (certified rank).
    ``scale`` is the half-width of the subspace sampling box: 1.0 reproduces the paper's
    ``z ~ U([-1, 1]^r)``.
    ``anchor`` (default 0.0 = plain GIT-BO) mixes a fraction of the candidate pool from
    Vanilla BO's ``sample_around_best`` generator -- Gaussian perturbations of a random
    subset of the best point's dimensions -- in place of that many gradient-subspace
    candidates. This is the "mixed anchor" variant: the incumbent-anchored local sampler
    of Vanilla BO grafted onto GIT-BO's global gradient-informed subspace pool.
    ``acq`` selects the acquisition scored on the subspace candidates:
    * ``gaussian`` (default) -- the paper's ``mu + beta * sigma`` with beta = 2.33;
    * ``ts``       -- Monte-Carlo Thompson sampling: one draw from TabPFN's own bar
                      distribution per candidate (``TabPFNSurrogate.predict_sample``);
    * ``ucb``      -- TabPFN's built-in QUANTILE UCB (``BarDistribution.ucb``, the
                      ``1 - rest_prob`` quantile of the non-Gaussian predictive), i.e.
                      ``reg.predict(output_type="ucb")`` at the paper's beta = 2.33.
    """
    set_seed(seed)
    obj = ProblemObjective(problem)
    dim = obj.dim
    if n_init is None:
        n_init = default_n_init(dim)
    acq = acq.lower()
    assert acq in ("gaussian", "ts", "ucb"), f"unknown acq {acq!r}"
    rank_mode = "marzouk" if str(rank).lower() == "marzouk" else "fixed"
    fixed_rank = DEFAULT_RANK if rank_mode == "marzouk" else int(rank)
    base_name = rank_mode if rank_mode == "marzouk" else "rank" + str(fixed_rank)
    anchor_tag = "_anchor" if anchor > 0.0 else ""
    # gaussian keeps the legacy names (git_bo_rank10 / git_bo_anchor_rank10); ts/ucb are
    # tagged so the 2 acq x 2 rank x 2 anchor grid gives eight distinct method names.
    if acq == "gaussian":
        method = f"git_bo{anchor_tag}_{base_name}"
    else:
        method = f"git_bo_{acq}_{base_name}{anchor_tag}"
    acq_str = {
        "gaussian": f"UCB(mu + {BETA} * sigma)  [GIT-BO Sec. 3.3]",
        "ts": "Thompson sampling (TabPFN bar-distribution draw)",
        "ucb": f"quantile UCB (BarDistribution.ucb, beta={BETA})",
    }[acq]
    res = Result(method, type(problem).__name__, seed, acquisition_function=acq_str)

    surrogate = TabPFNSurrogate(device=device)
    rng = np.random.default_rng(seed)
    sobol = SobolEngine(dim, scramble=True, seed=seed)

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, _ = load_checkpoint(checkpoint, res)
        best = train_Y.max().item()
    else:
        train_X = initial_design(n_init, dim, seed)
        train_Y = obj(train_X)
        best = train_Y.max().item()
        res.start(best)
        start_it = 0

    def _forward(cand: torch.Tensor, grad: bool):
        """One TabPFN pass: observed rows as context, ``cand`` as queries."""
        n_ctx = train_X.shape[0]
        X_q = cand.unsqueeze(1)
        if grad:
            X_q = X_q.clone().requires_grad_(True)
        X_full = torch.cat([train_X.unsqueeze(1).detach(), X_q], dim=0)
        Y_full = torch.cat(
            [train_Y, torch.zeros(cand.shape[0], 1, dtype=DTYPE)], dim=0
        ).unsqueeze(1)
        return surrogate.forward(X_full, Y_full, single_eval_pos=n_ctx), X_q

    chunk = [min(N_CANDIDATES, MAX_CAND_PER_PASS)]

    def _chunked(fn, n):
        """Apply ``fn(lo, hi)`` over ``[0, n)`` in slices of ``chunk[0]`` rows.

        On CUDA OOM the chunk width is halved (down to 1) and the failed slice retried,
        so a shared-GPU spike shrinks the pass instead of killing the run. The width
        stays shrunk for the rest of the run. Results are returned in order.
        """
        out, i = [], 0
        while i < n:
            c = min(chunk[0], n - i)
            try:
                out.append(fn(i, i + c))
                i += c
            except torch.OutOfMemoryError:
                if chunk[0] == 1:
                    raise
                chunk[0] = max(1, chunk[0] // 2)
                if surrogate.device.type == "cuda":
                    torch.cuda.empty_cache()
        return out

    for it in range(start_it, iters):
        # (1-2) Sobol pool -> TabPFN -> gradients of the predictive mean, in candidate
        # chunks. Each row's gradient is independent (queries do not cross-attend), so the
        # concatenation equals the single-pass grad exactly.
        pool = sobol.draw(N_CANDIDATES).to(DTYPE)

        def _grad_slice(lo, hi):
            logits, X_q = _forward(pool[lo:hi], grad=True)
            mean_sub = surrogate.predict_mean(logits).reshape(-1)
            (g,) = torch.autograd.grad(mean_sub.sum(), X_q)
            return g.reshape(hi - lo, dim).detach().cpu()

        grads = torch.cat(_chunked(_grad_slice, N_CANDIDATES)).numpy()

        # (3-4) H = E[g g^T] -> top-r eigenvectors -> candidates in that subspace,
        # centered on the centroid of the observed data (x_ref = mean(X_obs)).
        x_ref = train_X.mean(dim=0).cpu().numpy()
        samples, _r = sample_in_subspace(
            x_ref,
            grads,
            N_CANDIDATES,
            rank_mode,
            fixed_rank,
            eps=0.05,
            scale=scale,
            rng=rng,
        )
        cand = torch.from_numpy(samples).to(DTYPE)

        # Mixed-anchor variant: swap a fraction of the gradient-subspace pool for
        # sample_around_best candidates centered on the current incumbent. The subspace
        # pool covers the globally informative directions; the anchor pool refines locally
        # around the best point, which the pure centroid-anchored subspace can under-serve.
        if anchor > 0.0:
            n_anchor = int(round(anchor * N_CANDIDATES))
            if n_anchor > 0:
                x_best = train_X[int(train_Y.argmax())].to(DTYPE)
                anchor_cand = _sample_around_best(
                    x_best, n_anchor, anchor_sigma, min(20.0 / dim, 1.0)
                )
                cand = torch.cat([cand[: N_CANDIDATES - n_anchor], anchor_cand], dim=0)

        # (5) Score the subspace candidates with the selected acquisition. mu and sigma
        # are always read off TabPFN for logging; the ranking `score` depends on `acq`:
        #   gaussian -> mu + beta*sigma (paper Sec. 3.3);
        #   ts       -> one bar-distribution draw per candidate (Thompson);
        #   ucb      -> TabPFN's built-in quantile UCB (BarDistribution.ucb), the
        #               (1 - rest_prob) quantile of its non-Gaussian predictive.
        # The bar-distribution ``sample``/``ucb`` are read from the SAME logits as mu/sigma,
        # so switching acquisitions costs no extra forward pass.
        def _acq_slice(lo, hi):
            with torch.no_grad():
                logits_gi, _ = _forward(cand[lo:hi], grad=False)
                m = surrogate.predict_mean(logits_gi).reshape(-1)
                v = surrogate.predict_variance(logits_gi).reshape(-1).clamp_min(0.0)
                if acq == "ts":
                    s = surrogate.predict_sample(logits_gi).reshape(-1)
                elif acq == "ucb":
                    s = surrogate.predict_ucb(logits_gi, rest_prob=REST_PROB).reshape(-1)
                else:
                    s = m + BETA * v.sqrt()
            return m.cpu(), v.cpu(), s.cpu()

        parts = _chunked(_acq_slice, cand.shape[0])
        mean = torch.cat([m for m, _, _ in parts])
        var = torch.cat([v for _, v, _ in parts])
        score = torch.cat([s for _, _, s in parts])

        choice = int(torch.argmax(score).item())
        x_new = cand[choice : choice + 1]
        y_new = obj(x_new)
        train_X = torch.cat([train_X, x_new], dim=0)
        train_Y = torch.cat([train_Y, y_new], dim=0)
        best = max(best, y_new.item())
        res.record(
            best,
            mean=mean[choice].item(),
            variance=var[choice].item(),
            acq_value=score[choice].item(),
        )
        if checkpoint:
            res.set_history(train_X, train_Y, n_init)
            save_checkpoint(checkpoint, train_X, train_Y, res, it + 1)
    res.set_history(train_X, train_Y, n_init)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument(
        "--init",
        type=int,
        default=None,
        help="initial design size (default: dim-scaled)",
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--rank",
        default=str(DEFAULT_RANK),
        help="active-subspace rank: an integer (paper default 10) or 'marzouk'",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="half-width of the subspace sampling box (paper: 1.0, i.e. z ~ U([-1,1]^r))",
    )
    parser.add_argument(
        "--anchor",
        type=float,
        default=0.0,
        help="fraction of the candidate pool drawn from Vanilla BO's sample_around_best "
        "(0.0 = plain GIT-BO; e.g. 0.5 = the mixed-anchor variant)",
    )
    parser.add_argument(
        "--anchor-sigma",
        type=float,
        default=ANCHOR_SIGMA,
        help="sample_around_best Gaussian sigma (Vanilla BO default 0.1)",
    )
    parser.add_argument(
        "--acq",
        default="gaussian",
        choices=("gaussian", "ts", "ucb"),
        help="acquisition: gaussian (mu+beta*sigma), ts (Thompson), ucb (quantile UCB)",
    )
    parser.add_argument(
        "--checkpoint", default=None, help="resumable checkpoint .npz path"
    )
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args),
        args.init,
        args.iters,
        args.seed,
        rank=args.rank,
        scale=args.scale,
        anchor=args.anchor,
        anchor_sigma=args.anchor_sigma,
        acq=args.acq,
        device=args.device,
        checkpoint=args.checkpoint,
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
