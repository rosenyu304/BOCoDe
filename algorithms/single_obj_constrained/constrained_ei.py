"""Constrained Expected Improvement for single-objective constrained problems.

Models the objective and each inequality constraint with independent GPs (a
``ModelListGP``) and maximizes the log Constrained Expected Improvement: EI on the
objective weighted by the probability that every constraint is satisfied
(``c <= 0``). This is the analytic constrained-BO baseline from the BoTorch
closed-loop tutorial, using the numerically stable log form.

Before any feasible point has been observed the EI target (the best *feasible*
value) does not exist, so -- following Gelbart et al. (2014), Sec. 3.2 "Finding the
feasible region", Eq. (9) -- the acquisition drops the EI factor and maximizes the
probability of feasibility alone, ``prod_k P(c_k(x) <= 0)``, until the feasible
region is found.

Run::

    python -m algorithms.single_obj_constrained.constrained_ei --problem PressureVessel --init 10 --iters 50

Sources:
J. R. Gardner, M. J. Kusner, Z. Xu, K. Q. Weinberger, and J. P. Cunningham. Bayesian Optimization with Inequality Constraints. ICML 2014.
M. Gelbart, J. Snoek, and R. P. Adams. Bayesian Optimization with Unknown Constraints. UAI 2014. https://arxiv.org/abs/1403.5607
S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy. Unexpected Improvements to Expected Improvement for Bayesian Optimization. NeurIPS 2023. https://arxiv.org/abs/2310.20708
BoTorch closed-loop tutorial: https://botorch.org/docs/tutorials/closed_loop_botorch_only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from botorch.acquisition.analytic import (
    LogConstrainedExpectedImprovement,
    LogProbabilityOfFeasibility,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import SumMarginalLogLikelihood

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
    resolve_device,
    save_checkpoint,
    set_seed,
)


def _fit_model_list(train_X, train_obj, train_con):
    """Fit one GP for the objective and one per constraint."""
    dim = train_X.shape[-1]
    models = [
        SingleTaskGP(
            train_X,
            train_obj,
            input_transform=Normalize(d=dim),
            outcome_transform=Standardize(m=1),
        )
    ]
    for i in range(train_con.shape[-1]):
        models.append(
            SingleTaskGP(
                train_X,
                train_con[:, i : i + 1],
                input_transform=Normalize(d=dim),
                outcome_transform=Standardize(m=1),
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Constrained EI BO. ``n_init`` defaults to the dim-scaled BoCoDe default.

    GP fits + acquisition optimization run on ``device`` (default cuda when available);
    the objective/constraints are evaluated on CPU. With ``checkpoint`` set the run is
    resumable: it restores ``(X, obj, constraints, completed_iters, RNG)`` and saves
    after every iteration.
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = ProblemObjective(problem)
    assert obj.num_constraints > 0, "constrained_ei requires a constrained problem"
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "constrained_ei", type(problem).__name__, seed, acquisition_function="LogCEI"
    )

    def best_feasible(obj_v, con_v):
        feas = (con_v <= 0).all(dim=1)
        return obj_v[feas].max().item() if feas.any() else -float("inf")

    if checkpoint and Path(checkpoint).exists():
        train_X, train_obj, start_it, data = load_checkpoint(checkpoint, res)
        train_con = torch.tensor(data["con"], dtype=DTYPE)
        best = best_feasible(train_obj, train_con)
    else:
        train_X = initial_design(n_init, obj.dim, seed)
        train_obj, train_con = obj.evaluate_raw(train_X)
        best = best_feasible(train_obj, train_con)
        res.start(best)
        start_it = 0

    # BoTorch's constraint dict maps each constraint model's output index to
    # (lower, upper) feasible bounds; here constraints are feasible when <= 0.
    constraint_dict = {i + 1: (None, 0.0) for i in range(obj.num_constraints)}
    bounds_dev = obj.bounds.to(dev)

    for it in range(start_it, iters):
        model = _fit_model_list(train_X.to(dev), train_obj.to(dev), train_con.to(dev))
        # With no feasible observation the EI target does not exist, so the acquisition
        # reduces to the probability of feasibility alone (Gelbart et al. 2014, Eq. 9);
        # once a feasible point exists, best_f is the best *feasible* objective.
        feas = (train_con <= 0).all(dim=1)
        if feas.any():
            acqf = LogConstrainedExpectedImprovement(
                model=model,
                best_f=train_obj[feas].max().to(dev),
                objective_index=0,
                constraints=constraint_dict,
            )
        else:
            acqf = LogProbabilityOfFeasibility(model=model, constraints=constraint_dict)
        candidate, _ = optimize_acqf(
            acqf, bounds=bounds_dev, q=1, num_restarts=10, raw_samples=512
        )
        candidate = candidate.detach().to(device="cpu", dtype=DTYPE)
        o, c = obj.evaluate_raw(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_obj = torch.cat([train_obj, o], dim=0)
        train_con = torch.cat([train_con, c], dim=0)
        best = max(best, best_feasible(o, c))
        res.record(best)
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
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args),
        args.init,
        args.iters,
        args.seed,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
