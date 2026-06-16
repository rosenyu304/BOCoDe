"""PFN-CEI: Constrained Expected Improvement with a tabular foundation model.

Constrained BO that replaces the GP surrogate with a frozen, pretrained TabPFN
regressor (the "vanilla regressor"). The objective and every inequality constraint
are modeled by TabPFN, and the next point maximizes the constrained Expected
Improvement: ``EI(objective) * prod_i P(constraint_i <= 0)``.

All outputs are scored in **one parallel TabPFN forward**: the same candidate inputs
are paired with the objective and each constraint as separate columns of the model's
batch dimension, so one call returns the objective EI and every constraint's
feasibility probability together (Gaussian feasibility from the bar-distribution
mean/variance).

Requires the TabPFN fork (see docs/tfm_setup.md). Run::

    python -m algorithms.single_obj_constrained.pfn_cei --problem PressureVessel --iters 50

Sources:
PFN-CEI: constrained Bayesian optimization with prior-data fitted networks. Structural and Multidisciplinary Optimization, 2025. https://link.springer.com/article/10.1007/s00158-025-03987-z
N. Hollmann, S. Müller, et al. TabPFN: accurate predictions on small data with a tabular foundation model. Nature, 2025.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch.quasirandom import SobolEngine

import bocode

from .._bo_utils import (
    DTYPE,
    ProblemObjective,
    Result,
    add_common_args,
    finalize,
    initial_design,
    set_seed,
)
from .._tfm_utils import TabPFNSurrogate

N_CANDIDATES = 2000


def _normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def optimize_problem(problem, n_init: int = 20, iters: int = 50, seed: int = 0,
                     device: str = "auto") -> Result:
    set_seed(seed)
    obj = ProblemObjective(problem)
    nc = obj.num_constraints
    assert nc > 0, "pfn_cei requires a constrained problem"
    dim = obj.dim
    res = Result("pfn_cei", type(problem).__name__, seed, acquisition_function="Constrained-EI")

    surrogate = TabPFNSurrogate(device=device)
    sobol = SobolEngine(dim, scramble=True, seed=seed)

    train_X = initial_design(n_init, dim, seed)
    train_obj, train_con = obj.evaluate_raw(train_X)

    def best_feasible(o, c):
        feas = (c <= 0).all(dim=1)
        return o[feas].max().item() if feas.any() else -float("inf")

    best = best_feasible(train_obj, train_con)
    res.start(best)

    for _ in range(iters):
        cand = sobol.draw(N_CANDIDATES).to(DTYPE)
        n_ctx = train_X.shape[0]
        m = 1 + nc  # objective + constraints, as batch columns

        # X is the same for every column; Y differs per column (objective / each con).
        X_full = torch.cat([train_X, cand], dim=0).unsqueeze(1).expand(-1, m, -1).contiguous()
        Y_cols = torch.cat([train_obj, train_con], dim=1)  # (n_ctx, m)
        Y_full = torch.cat(
            [Y_cols, torch.zeros(N_CANDIDATES, m, dtype=DTYPE)], dim=0
        ).unsqueeze(-1)  # (n_ctx+n_cand, m, 1)

        with torch.no_grad():
            logits = surrogate.forward(X_full, Y_full, single_eval_pos=n_ctx)
            feas_best = best if best != -float("inf") else train_obj.max().item()
            ei = surrogate.predict_ei(logits, torch.tensor(feas_best))[n_ctx:, 0]  # objective column
            mean = surrogate.predict_mean(logits)[n_ctx:]  # (n_cand, m)
            var = surrogate.predict_variance(logits)[n_ctx:].clamp_min(1e-9)

        # feasibility probability per constraint column: P(c_i <= 0) = Phi((0 - mu)/sigma)
        mu_c = mean[:, 1:]
        sig_c = var[:, 1:].sqrt()
        p_feas = _normal_cdf((0.0 - mu_c) / sig_c).prod(dim=1)  # (n_cand,)
        cei = ei * p_feas

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
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--init", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="auto")
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        bocode.get_problem(args.problem)(), args.init, args.iters, args.seed, device=args.device
    )
    finalize(res, args)


if __name__ == "__main__":
    main()
