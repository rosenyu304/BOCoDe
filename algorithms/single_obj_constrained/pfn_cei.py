"""PFN-CEI: Constrained Expected Improvement with a tabular foundation model.

Constrained BO that replaces the GP surrogate with a frozen, pretrained PFN regressor.
The objective and every inequality constraint are modeled by the *same* network, scored
in **one parallel forward pass** -- the same candidate inputs are paired with the
objective and each constraint as separate columns of the model's batch dimension -- so a
single call returns the objective's EI and every constraint's feasibility probability
together. That single-surrogate, single-pass design is the method's contribution (no
per-constraint GP to fit).

The acquisition is the classic constrained EI ``EI(f) * prod_i P(c_i <= 0)``, but *both*
factors are read off the PFN's own **bar distribution** -- a discretized, non-Gaussian
predictive distribution. Following the reference implementation:

* constraints are negated (``g = -c``) so that "feasible" means "large", and the
  feasibility factor is the bar distribution's ``pi`` (probability of improvement) above
  the transformed zero threshold -- *not* a Gaussian ``Phi((0 - mu)/sigma)`` read off the
  predictive mean/variance, which would throw away the non-Gaussian shape the PFN exists
  to model;
* the objective and the constraints are warped with a Yeo-Johnson power transform, and
  the zero feasibility threshold is mapped through the *same* transform;
* the EI incumbent is ``tau = max(y_observed)`` over all observations (reference
  ``PFN_CEI.py:313``), not the best feasible value.

Reference: https://github.com/rosenyu304/BOEngineeringBenchmark (``PFN_CEI.py``,
``Tutorial_PFN_CEI.ipynb``). BoCoDe substitutes TabPFN (>= v3) for the reference's
PFNs4BO HEBO checkpoint; the acquisition logic is otherwise the reference's.

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
from sklearn.preprocessing import PowerTransformer

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


def _power_transform(train: torch.Tensor, apply: torch.Tensor) -> torch.Tensor:
    """Yeo-Johnson power transform fit on ``train``, applied to ``apply`` (column-wise).

    Mirrors the reference's ``general_power_transform`` with ``eps=0``; on failure it
    falls back to mean-centering, as the reference does.
    """
    pt = PowerTransformer(method="yeo-johnson")
    try:
        pt.fit(train.cpu().double().numpy())
        out = pt.transform(apply.cpu().double().numpy())
        out = torch.as_tensor(out, dtype=DTYPE)
        if not torch.isfinite(out).all():
            raise ValueError("power transform produced non-finite values")
        return out
    except Exception:
        return (apply - train.mean(dim=0)).to(DTYPE)


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
        n_ctx = train_X.shape[0]
        m = 1 + nc  # objective + constraints, as batch columns

        # Warp objective and (negated) constraints, and push the zero feasibility
        # threshold through the constraints' own transform.
        y_t = _power_transform(train_obj, train_obj)
        g = -train_con  # feasible (c <= 0) now means "large"
        g_t = _power_transform(g, g)
        thr = _power_transform(g, torch.zeros(1, nc, dtype=DTYPE))[0]  # (nc,)

        # X is the same for every column; Y differs per column (objective / each con).
        X_full = (
            torch.cat([train_X, cand], dim=0)
            .unsqueeze(1)
            .expand(-1, m, -1)
            .contiguous()
        )
        Y_cols = torch.cat([y_t, g_t], dim=1)  # (n_ctx, m)
        Y_full = torch.cat(
            [Y_cols, torch.zeros(N_CANDIDATES, m, dtype=DTYPE)], dim=0
        ).unsqueeze(-1)  # (n_ctx+n_cand, m, 1)

        # One threshold per output column: the EI incumbent for the objective column,
        # the transformed zero feasibility threshold for each constraint column.
        # tau = max over ALL observed objective values (reference PFN_CEI.py:313).
        thr_full = torch.cat([y_t.max().reshape(1), thr])  # (m,)

        with torch.no_grad():
            # forward() returns the candidate rows' logits (context rows already dropped)
            logits = surrogate.forward(X_full, Y_full, single_eval_pos=n_ctx)
            ei = surrogate.predict_ei(logits, thr_full)[:, 0]  # objective column
            # feasibility straight from the bar distribution: P(g_i > transformed 0)
            p_feas = surrogate.predict_pi(logits, thr_full)[:, 1:]  # (n_cand, nc)
            cei = ei * p_feas.prod(dim=1)
            mean = surrogate.predict_mean(logits)
            var = surrogate.predict_variance(logits)

        choice = int(torch.argmax(cei).item())
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
            acq_value=cei[choice].item(),
        )
        if checkpoint:
            res.set_history(train_X, train_obj, n_init)
            save_checkpoint(
                checkpoint,
                train_X,
                train_obj,
                res,
                it + 1,
                extra={"con": train_con.cpu().numpy()},
            )
    res.set_history(train_X, train_obj, n_init)
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
