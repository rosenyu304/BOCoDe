"""TFM-TuRBO: TuRBO with a tabular foundation model (TabPFN) as the local surrogate.

TuRBO-1 (Eriksson et al., 2019) with **every piece of the trust-region machinery kept**
-- ``length_init = 0.8``, ``length_min = 0.5**7``, ``length_max = 1.6``, ``succtol = 3``,
``failtol = ceil(max(4/q, d/q))``, the success test ``f_next > f_best + 1e-3*|f_best|``,
the Sobol candidate set with the ``p_perturb = min(1, 20/d)`` mask, restart on collapse --
but with the local **GP replaced by a frozen, pretrained TabPFN**: the trust region's
observed points are the model's in-context rows and the candidates are query rows. No
surrogate is trained; each iteration is one TabPFN forward pass (plus one backward pass
for the trust-region weights, see below).

The trust-region state machine is imported from :mod:`algorithms.single_obj.turbo`, so the
two methods share one implementation of it by construction.

**This method is not from a paper.** It is a BoCoDe ablation ("what does TuRBO do if its
surrogate is a TFM?"), and two things TuRBO gets from a GP have no TabPFN counterpart. Both
substitutes below are **our design choices**:

1. **Thompson sampling.** TuRBO's acquisition draws a realization of the GP posterior over
   the candidate set and takes its argmax (``MaxPosteriorSampling``). TabPFN gives a
   *marginal* predictive distribution per row (the bar distribution) and no joint posterior,
   so there is no correlated sample path to draw. We draw **independent samples from each
   candidate's own bar distribution** (``TabPFNSurrogate.predict_sample``, the vectorized
   form of TabPFN's ``BarDistribution.sample``) and take the argmax. This is Thompson
   sampling under an independence assumption; it is noisier / more exploratory than a GP's
   correlated draw, and that difference is a property of the surrogate, not a bug.

2. **The trust-region side lengths.** TuRBO scales side length ``i`` by the GP's ``i``-th
   ARD lengthscale (normalized to geometric mean 1), so directions the GP finds smooth get
   a longer side. TabPFN has no lengthscales. ``--tr-weights``:

   * ``grad`` (default) -- a **gradient-sensitivity pseudo-lengthscale**. Backprop through
     TabPFN's predictive mean at the trust region's own observed points (the very data
     TuRBO's GP fits its ARD lengthscales to) to get ``s_i = mean_x |d mu / d x_i|``, scale
     it to mean 1, and set ``ell_i = 1 / s_i``: a coordinate the model's mean barely
     responds to behaves like a long lengthscale and gets a long side, and vice versa.
     ``ell`` is clamped to ``[0.005, 4.0]`` -- the *same* interval TuRBO constrains its ARD
     lengthscales to (``turbo/gp.py``) -- and then normalized to geometric mean 1 with
     TuRBO's own two lines. This is an *analogy*, not a derivation: ``1/|d mu/dx|`` is a
     first-order sensitivity, not a kernel lengthscale, and nothing guarantees the two agree.
   * ``isotropic`` -- ``weights = 1``, i.e. a cube. The honest null option; use it to
     measure what the gradient weighting actually buys.

Reproducibility caveat (``--tr-weights grad`` only)
---------------------------------------------------
Backpropagating through TabPFN on CUDA is **not bit-deterministic** (float32 atomics in the
backward reductions): repeated backward passes on identical inputs disagree by ~1e-8 in the
sensitivities. That is far too small to matter for the weights themselves, but the
Thompson-sampling argmax over thousands of near-tied candidates *amplifies* it, so two runs
with the same seed can query different (statistically equivalent) points, and a checkpoint
resume need not reproduce an uninterrupted run exactly. Measured on this repo:
``--tr-weights grad`` differs run-to-run; ``--tr-weights isotropic`` and ``tfm_scbo`` (neither
of which does a backward pass) are **bit-identical** across runs and across checkpoint
resumes. Use ``isotropic`` if you need exact reproducibility.

Needs TabPFN >= v3 (see docs/tfm_setup.md). Run::

    python -m algorithms.single_obj.tfm_turbo --problem Ackley --init 20 --iters 50
    python -m algorithms.single_obj.tfm_turbo --problem Ackley --iters 50 --tr-weights isotropic

Sources:
D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, and M. Poloczek. Scalable Global Optimization via Local Bayesian Optimization. NeurIPS 2019. https://arxiv.org/abs/1910.01739 (trust-region logic, constants; official code https://github.com/uber-research/TuRBO)
N. Hollmann, S. Müller, et al. TabPFN: accurate predictions on small data with a tabular foundation model. Nature, 2025. https://github.com/PriorLabs/TabPFN (the surrogate)
The GP -> TabPFN substitution, the independent-marginal Thompson sampling and the gradient-sensitivity trust-region weights are BoCoDe's, with no paper behind them.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

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
from .._tfm_utils import TabPFNSurrogate
from .turbo import TrustRegion

# TuRBO's ARD lengthscale box (uber-research/TuRBO ``turbo/gp.py``). The gradient-derived
# pseudo-lengthscales are clamped to it so a TabPFN sensitivity of ~0 cannot blow one side
# of the trust region up (or a huge one collapse it) beyond what TuRBO's own GP could.
LENGTHSCALE_MIN, LENGTHSCALE_MAX = 0.005, 4.0


def _tabpfn_forward(surrogate, train_X, train_Y, cand, grad: bool):
    """One TabPFN pass: the trust region's data as context, ``cand`` as queries."""
    n_ctx = train_X.shape[0]
    X_q = cand.unsqueeze(1)
    if grad:
        X_q = X_q.clone().requires_grad_(True)
    X_full = torch.cat([train_X.unsqueeze(1).detach(), X_q], dim=0)
    Y_full = torch.cat(
        [train_Y, torch.zeros(cand.shape[0], 1, dtype=DTYPE)], dim=0
    ).unsqueeze(1)
    return surrogate.forward(X_full, Y_full, single_eval_pos=n_ctx), X_q


def tr_weights(surrogate, train_X, train_Y, mode: str) -> torch.Tensor:
    """Per-dimension trust-region side-length weights (TuRBO's ARD-lengthscale stand-in).

    ``mode='isotropic'`` returns ones (a cube). ``mode='grad'`` returns the
    gradient-sensitivity pseudo-lengthscales described in the module docstring: a BoCoDe
    design choice, **not** something TuRBO or TabPFN prescribes.
    """
    dim = train_X.shape[-1]
    if mode == "isotropic":
        return torch.ones(dim, dtype=DTYPE)

    # d mu / d x at the trust region's own observed points (the data TuRBO's GP would fit
    # its ARD lengthscales to). One forward + one backward pass over n_ctx rows.
    logits, X_q = _tabpfn_forward(surrogate, train_X, train_Y, train_X, grad=True)
    mean = surrogate.predict_mean(logits).reshape(-1)
    (grads,) = torch.autograd.grad(mean.sum(), X_q)
    s = grads.reshape(-1, dim).abs().mean(dim=0).detach().to(device="cpu", dtype=DTYPE)

    s = s / s.mean().clamp_min(1e-12)  # scale-free: mean sensitivity 1
    ell = (1.0 / s.clamp_min(1e-12)).clamp(LENGTHSCALE_MIN, LENGTHSCALE_MAX)
    # TuRBO's own normalization (turbo.py:149-151 / turbo_1.py L182-186).
    ell = ell / ell.mean()
    return ell / torch.prod(ell.pow(1.0 / dim))


def _generate_batch(
    surrogate,
    tr: TrustRegion,
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int,
    seed: int,
    weight_mode: str,
) -> torch.Tensor:
    """TuRBO's candidate set inside the trust region, scored by TabPFN Thompson sampling."""
    dim = X.shape[-1]
    x_center = X[Y.argmax(), :].clone()
    weights = tr_weights(surrogate, X, Y, weight_mode)
    tr_lb = torch.clamp(x_center - weights * tr.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * tr.length / 2.0, 0.0, 1.0)

    n_candidates = min(5000, max(2000, 200 * dim))  # TuRBO's n_cand
    sobol = SobolEngine(dim, scramble=True, seed=seed)
    pert = tr_lb + (tr_ub - tr_lb) * sobol.draw(n_candidates).to(DTYPE)

    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=DTYPE) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim, size=(len(ind),))] = True
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Thompson sampling on TabPFN's bar distribution: one independent draw per candidate,
    # argmax; repeat without replacement for a batch (TuRBO's MaxPosteriorSampling does the
    # same, but from a *correlated* GP sample path -- see the module docstring).
    with torch.no_grad():
        logits, _ = _tabpfn_forward(surrogate, X, Y, X_cand, grad=False)
        chosen: list[int] = []
        for _ in range(batch_size):
            draw = surrogate.predict_sample(logits).reshape(-1).to("cpu")
            if chosen:
                draw[torch.tensor(chosen)] = -float("inf")
            chosen.append(int(draw.argmax()))
    return X_cand[chosen]


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 100,
    seed: int = 0,
    batch: int = 1,
    tr_weight_mode: str = "grad",
    device: str = "auto",
    checkpoint: str | None = None,
) -> Result:
    """TFM-TuRBO over the unit cube. ``n_init`` defaults to the dim-scaled BoCoDe default.

    The trust-region bookkeeping is TuRBO's (restarts included: a restart's initial design
    consumes evaluations from ``iters``); only the surrogate and the two substitutes named
    in the module docstring differ.
    """
    set_seed(seed)
    obj = ProblemObjective(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "tfm_turbo",
        type(problem).__name__,
        seed,
        acquisition_function=f"TabPFN bar-distribution Thompson sampling (tr_weights={tr_weight_mode})",
    )
    surrogate = TabPFNSurrogate(device=device)

    def restart_design(restart_idx: int):
        X = initial_design(n_init, obj.dim, seed + 9973 * restart_idx)
        return X, obj(X)

    if checkpoint and Path(checkpoint).exists():
        tr_X, tr_Y, evals, data = load_checkpoint(checkpoint, res)
        hist_X = torch.tensor(data["hist_X"], dtype=DTYPE)
        hist_Y = torch.tensor(data["hist_Y"], dtype=DTYPE)
        tr = TrustRegion(dim=obj.dim, batch_size=batch)
        tr.length = float(data["tr_length"])
        tr.success_counter = int(data["tr_success"])
        tr.failure_counter = int(data["tr_failure"])
        n_restarts = int(data["n_restarts"])
        best = hist_Y.max().item()
    else:
        tr_X, tr_Y = restart_design(0)
        hist_X, hist_Y = tr_X.clone(), tr_Y.clone()
        tr = TrustRegion(dim=obj.dim, batch_size=batch)
        n_restarts = 0
        evals = 0
        best = tr_Y.max().item()
        res.start(best)

    while evals < iters:
        cand = _generate_batch(
            surrogate, tr, tr_X, tr_Y, batch, seed + evals, tr_weight_mode
        )
        cand = cand.detach().to(device="cpu", dtype=DTYPE)

        y = obj(cand)
        f_best = tr_Y.max().item()  # success is judged against the TR's own incumbent
        improved = y.max().item() > f_best + 1e-3 * math.fabs(f_best)
        tr_X = torch.cat([tr_X, cand], dim=0)
        tr_Y = torch.cat([tr_Y, y], dim=0)
        hist_X = torch.cat([hist_X, cand], dim=0)
        hist_Y = torch.cat([hist_Y, y], dim=0)
        tr.update(improved)

        for j in range(cand.shape[0]):
            best = max(best, y[j].item())
            res.record(best)
            evals += 1
            if evals >= iters:
                break

        if tr.restart and evals < iters:
            n_restarts += 1
            tr_X, tr_Y = restart_design(n_restarts)
            hist_X = torch.cat([hist_X, tr_X], dim=0)
            hist_Y = torch.cat([hist_Y, tr_Y], dim=0)
            tr = TrustRegion(dim=obj.dim, batch_size=batch)
            for j in range(n_init):
                best = max(best, tr_Y[j].item())
                res.record(best)
                evals += 1
                if evals >= iters:
                    break

        if checkpoint:
            res.set_history(hist_X, hist_Y, n_init)
            save_checkpoint(
                checkpoint,
                tr_X,
                tr_Y,
                res,
                evals,
                extra=dict(
                    hist_X=hist_X.numpy(),
                    hist_Y=hist_Y.numpy(),
                    tr_length=tr.length,
                    tr_success=tr.success_counter,
                    tr_failure=tr.failure_counter,
                    n_restarts=n_restarts,
                ),
            )
    res.set_history(hist_X, hist_Y, n_init)
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
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1, help="TuRBO batch size q")
    parser.add_argument(
        "--tr-weights",
        dest="tr_weights",
        choices=["grad", "isotropic"],
        default="grad",
        help="trust-region side-length weights: TabPFN gradient sensitivity (default) "
        "or an isotropic cube. Both are BoCoDe design choices (TabPFN has no lengthscales).",
    )
    parser.add_argument(
        "--checkpoint", default=None, help="resumable checkpoint .npz path"
    )
    parser.add_argument("--device", default="auto", help="cuda / cpu (default: auto)")
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args),
        args.init,
        args.iters,
        args.seed,
        batch=args.batch,
        tr_weight_mode=args.tr_weights,
        device=args.device,
        checkpoint=args.checkpoint,
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
