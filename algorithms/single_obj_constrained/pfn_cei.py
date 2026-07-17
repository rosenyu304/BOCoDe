"""PFN-CEI: Constrained Expected Improvement with a tabular foundation model.

Constrained BO that swaps the GP surrogate for a frozen, pretrained TabPFN regressor,
otherwise following BoTorch's closed-loop constrained-EI recipe
(https://botorch.org/docs/tutorials/closed_loop_botorch_only) -- q-Noisy Constrained EI
in style: the acquisition is ``EI(f) * prod_i P(c_i <= 0)`` with the incumbent set to the
**best feasible** observed objective (or, before any point is feasible, the best observed
objective). The objective and every inequality constraint are each modeled by the *same*
network as a **zero-shot in-context regression** -- one column of the model's batch
dimension each -- so a single in-context pass returns the objective's EI and every
constraint's feasibility probability. That single-surrogate design is the method's
contribution (no per-constraint GP to fit).

Deliberately NOT following the PFN-CEI paper: there is **no power transform** on the
objective or the constraints. TabPFN standardizes each target column internally, so raw
targets go straight in; the objective and each constraint are simply the observed values.

The objective and each constraint are scored as SEPARATE in-context passes --
``1 + num_constraints`` forwards per iteration, one output column each -- rather than
stacked into the batch dimension all at once. A transformer's batch dimension does not
mix its items, so looping returns the same numbers; it is done this way because a single
pass over ``1 + num_constraints`` columns is O(num_constraints) in activation memory and
OOMed on every high-constraint problem in the suite, whereas one-at-a-time is O(1) in the
constraint count. ICL is cheap for TabPFN, so the extra forwards cost little.

The acquisition ``EI(f) * prod_i P(c_i <= 0)`` is evaluated as
``log EI(f) + sum_i log P(c_i <= 0)`` -- the same function (log is monotone, so the argmax
is identical), but the product form underflows float32 to exactly 0 once there are enough
constraints, which silently reduces the method to random search. Both factors are read off
the PFN's own **bar distribution** -- a discretized, non-Gaussian predictive distribution:
constraints are negated (``g = -c``) so that "feasible" means "large", and the feasibility
factor is the bar distribution's ``pi`` (probability of improvement) above the zero
threshold -- *not* a Gaussian ``Phi((0 - mu)/sigma)`` read off a predictive mean/variance,
which would throw away the non-Gaussian shape the PFN exists to model.

BoCoDe substitutes TabPFN (>= v3) for the reference's PFNs4BO HEBO checkpoint.

Needs TabPFN >= v3 (see docs/tfm_setup.md). Run::

    python -m algorithms.single_obj_constrained.pfn_cei --problem PressureVessel --iters 50

Sources:
R. T. Y. Yu, C. Picard, and F. Ahmed. Fast and accurate Bayesian optimization with pre-trained transformers for constrained engineering problems. Structural and Multidisciplinary Optimization, 2025. https://link.springer.com/article/10.1007/s00158-025-03987-z
S. Müller, M. Feurer, N. Hollmann, F. Hutter. PFNs4BO: In-Context Learning for Bayesian Optimization. ICML 2023. https://arxiv.org/abs/2305.17535
N. Hollmann, S. Müller, et al. TabPFN: accurate predictions on small data with a tabular foundation model. Nature, 2025.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

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

N_CANDIDATES = 1000  # reference draws 1000 uniform random candidates per iteration

# How many output columns (objective + constraints) to push through TabPFN per forward.
#
# ONE. Each objective/constraint is run through TabPFN as its own zero-shot in-context
# regression -- ``1 + num_constraints`` separate forward passes per iteration -- rather
# than stacked into TabPFN's batch dimension. This is Rosen's instruction: ICL is cheap
# for TabPFN, so handle the constraints by looping the forward, not by widening the batch.
#
# It is also the only thing that makes peak memory independent of the constraint count.
# A single stacked pass is O(1 + nc) in activation memory: CEC2020_p35 (148 constraints)
# -> 149 columns x (n_ctx + 1000) candidate rows asked CUDA for 7.09 GiB in one allocation
# and died; nine CEC2020 problems and both Truss72D variants OOMed the same way. At one
# column per pass, peak memory is O(1) in nc, so the method runs on any card.
#
# A transformer's batch dimension does not mix its items -- attention runs only over a
# column's own rows -- so one-at-a-time returns the SAME logits a stacked pass would (see
# _score_columns); the only thing that changes is peak activation memory. Measured cost on
# CEC2020_p35 (148 constraints), 1000 candidates, one RTX 4090: chunk=1 -> ~2 GiB, and the
# extra forwards are cheap because the 1000-candidate ROW dimension already saturates the
# GPU, so a wider column batch buys no throughput and only costs memory.
MAX_COLS_PER_PASS = 1


def _score_columns(surrogate, X_all, Y_full, n_ctx, thr_full, chunk):
    """Score the ``m`` output columns through TabPFN in groups of ``chunk``.

    Returns ``(ei, pi, mean, var)``, each ``(n_cand, m)`` -- exactly what a single
    forward over all ``m`` columns at once would have returned.

    Why this is not a change to the method: the columns live in TabPFN's batch
    dimension, and attention never crosses batch items, so column ``j``'s logits depend
    only on column ``j``'s targets. The per-column target standardization TabPFNSurrogate
    applies is likewise computed per column (``mean``/``std`` over the row axis), so it is
    unchanged by how the columns are grouped. Grouping is therefore a pure memory/layout
    choice; only peak activation memory changes.
    """
    m = Y_full.shape[1]
    ei, pi, mean, var = [], [], [], []
    for c0 in range(0, m, chunk):
        c1 = min(c0 + chunk, m)
        X_c = X_all.unsqueeze(1).expand(-1, c1 - c0, -1).contiguous()
        Y_c = Y_full[:, c0:c1]
        thr_c = thr_full[c0:c1]
        logits = surrogate.forward(X_c, Y_c, single_eval_pos=n_ctx)
        ei.append(surrogate.predict_ei(logits, thr_c))
        pi.append(surrogate.predict_pi(logits, thr_c))
        mean.append(surrogate.predict_mean(logits))
        var.append(surrogate.predict_variance(logits))
    cat = lambda xs: torch.cat(xs, dim=1)  # noqa: E731
    return cat(ei), cat(pi), cat(mean), cat(var)


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    device: str = "auto",
    checkpoint: str | None = None,
) -> Result:
    """PFN-CEI. ``n_init`` defaults to the dim-scaled BoCoDe default."""
    set_seed(seed)
    obj = ProblemObjective(problem)
    nc = obj.num_constraints
    assert nc > 0, "pfn_cei requires a constrained problem"
    dim = obj.dim
    if n_init is None:
        n_init = default_n_init(dim)
    res = Result(
        "pfn_cei", type(problem).__name__, seed, acquisition_function="Constrained-EI"
    )

    surrogate = TabPFNSurrogate(device=device)
    # Columns per TabPFN forward. Shrinks on OOM (below) and stays shrunk: a card that
    # could not fit 32 columns this iteration will not fit them the next one either.
    chunk = min(1 + nc, MAX_COLS_PER_PASS)

    def best_feasible(o, c):
        feas = (c <= 0).all(dim=1)
        return o[feas].max().item() if feas.any() else -float("inf")

    if checkpoint and Path(checkpoint).exists():
        train_X, train_obj, start_it, data = load_checkpoint(checkpoint, res)
        train_con = torch.tensor(data["con"], dtype=DTYPE)
        best = best_feasible(train_obj, train_con)
    else:
        train_X = initial_design(n_init, dim, seed)
        train_obj, train_con = obj.evaluate_raw(train_X)
        best = best_feasible(train_obj, train_con)
        res.start(best)
        start_it = 0

    for it in range(start_it, iters):
        cand = torch.rand(N_CANDIDATES, dim, dtype=DTYPE)
        # TabPFN cannot ingest a non-finite target either: the _power_transform fallback
        # (apply - train.mean()) propagates an -inf into the context and NaNs the forward. Some
        # benchmark objectives are non-finite on a sliver of the domain -- CEC2020_p12's
        # -log(y4+1) at y4=round(x7)=-1 is -inf, a verbatim property of the official suite -- so
        # build the in-context set from the rows where the objective and constraints are all finite.
        # The non-finite evaluations stay in the run history (best_feasible and the saved trace);
        # they only sit out of the surrogate. On a well-behaved problem every row is finite, so the
        # context is identical to before.
        fin = torch.isfinite(train_obj).all(dim=1) & torch.isfinite(train_con).all(
            dim=1
        )
        ctx_X, ctx_obj, ctx_con = train_X[fin], train_obj[fin], train_con[fin]
        n_ctx = ctx_X.shape[0]
        m = 1 + nc  # objective + constraints, as batch columns

        # Raw targets -- no power transform. TabPFN standardizes each target column
        # internally, so the objective column is the observed objective as-is and each
        # constraint column is its negation g = -c, chosen so that "feasible" (c <= 0)
        # means g is large. X is the same for every column; Y differs per column.
        X_all = torch.cat([ctx_X, cand], dim=0)  # (n_ctx+n_cand, dim)
        Y_cols = torch.cat([ctx_obj, -ctx_con], dim=1)  # (n_ctx, m)
        Y_full = torch.cat(
            [Y_cols, torch.zeros(N_CANDIDATES, m, dtype=DTYPE)], dim=0
        ).unsqueeze(-1)  # (n_ctx+n_cand, m, 1)

        # One threshold per output column. Objective column: the constrained-EI incumbent
        # -- the best FEASIBLE observed objective (q-Noisy Constrained EI style), or the
        # best observed objective before any point is feasible so EI stays well-defined.
        # Constraint columns: the zero feasibility threshold (g = -c, so P(g > 0) = P(c <= 0)).
        feas_ctx = (ctx_con <= 0).all(dim=1)
        incumbent = ctx_obj[feas_ctx].max() if feas_ctx.any() else ctx_obj.max()
        thr_full = torch.cat(
            [incumbent.reshape(1), torch.zeros(nc, dtype=DTYPE)]
        )  # (m,)

        with torch.no_grad():
            # Score the m columns in chunks. On OOM, halve the chunk and retry: the
            # result is identical either way, so this only trades kernel launches for
            # peak memory. At chunk == 1 there is nothing left to shrink -- re-raise
            # rather than pretend the iteration succeeded.
            while True:
                try:
                    ei_all, pi_all, mean, var = _score_columns(
                        surrogate, X_all, Y_full, n_ctx, thr_full, chunk
                    )
                    break
                except torch.OutOfMemoryError:
                    if chunk == 1:
                        raise
                    chunk = max(1, chunk // 2)
                    if surrogate.device.type == "cuda":
                        torch.cuda.empty_cache()
            ei = ei_all[:, 0]  # objective column
            # feasibility straight from the bar distribution: P(g_i > transformed 0)
            p_feas = pi_all[:, 1:]  # (n_cand, nc)
            # Score in LOG space: log CEI = log EI + sum_i log P(c_i <= 0).
            #
            # This is the same acquisition -- log is strictly increasing, so the argmax is
            # unchanged -- but the product form underflows. The factors are float32 and
            # there is one per constraint, so on a problem with many constraints their
            # product falls below float32's smallest normal (1.2e-38) and rounds to
            # EXACTLY 0. On CEC2020_p7 (38 constraints) that happened for all 1000
            # candidates at once: every p_feas was fine (min 1.5e-6, none zero) yet
            # prod() was 0.0 for every candidate, so cei was the constant 0, argmax
            # returned index 0, and PFN-CEI silently degenerated into "return the first
            # uniform random candidate" on every iteration -- while still reporting a
            # successful run. The log-sum of the same numbers is a healthy -218..-161 and
            # ranks all 1000 candidates distinctly.
            log_cei = torch.log(ei.double()) + torch.log(p_feas.double()).sum(dim=1)

        choice = int(torch.argmax(log_cei).item())
        x_new = cand[choice : choice + 1]
        o_new, c_new = obj.evaluate_raw(x_new)
        train_X = torch.cat([train_X, x_new], dim=0)
        train_obj = torch.cat([train_obj, o_new], dim=0)
        train_con = torch.cat([train_con, c_new], dim=0)
        best = max(best, best_feasible(o_new, c_new))
        res.record(
            best,
            mean=mean[choice, 0].item(),
            variance=var[choice, 0].item(),
            # NB: the recorded acquisition is now log(CEI), not CEI. The linear value is
            # not representable -- it underflows to 0 on high-constraint problems, which
            # is the whole reason selection moved to log space.
            acq_value=log_cei[choice].item(),
        )
        if checkpoint:
            res.set_history(train_X, train_obj, n_init, c=train_con)
            save_checkpoint(
                checkpoint,
                train_X,
                train_obj,
                res,
                it + 1,
                extra={"con": train_con.cpu().numpy()},
            )
    res.set_history(train_X, train_obj, n_init, c=train_con)
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
        device=args.device,
        checkpoint=args.checkpoint,
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
