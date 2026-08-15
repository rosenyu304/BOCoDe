"""TFM-CqNParEGO: constrained qNParEGO on a tabular-foundation-model surrogate.

Constrained qParEGO exactly as in ``multi_obj_constrained/constrained_qparego`` -- each
step draws a random augmented-Chebyshev scalarization of the objectives and maximizes
q-Log Noisy Expected Improvement on it, feasibility-weighted by the constraint outputs
(``c <= 0`` is feasible) -- but the GPs are replaced by a **frozen, pretrained TabPFN**.
There is no model fitting: TabPFN conditions on the observed data in-context at every
acquisition evaluation.

The surrogate is :class:`algorithms._tfm_utils.TabPFNModel`, which dresses TabPFN up as a
BoTorch multi-output ``Model``, so BoTorch's stock ``qLogNoisyExpectedImprovement`` +
``GenericMCObjective(chebyshev)`` and ``optimize_acqf`` run on it unchanged (gradients
flow through the network to the candidate).

**Multi-output.** TabPFN is single-output, so every objective *and every constraint* gets
its own TabPFN predictor -- the ``ModelListGP`` pattern, objectives first -- but all
``m + n_c`` outputs are scored in **one** forward pass by putting them in TabPFN's *batch*
dimension (identical ``X`` columns, one ``Y`` column per output), the same trick PFN-CEI
uses. Each output is conditioned and standardized independently. As in the GP version, the
scalarization is applied only to the objective outputs ``Z[..., :m]`` and the constraint
outputs ``m..m+n_c-1`` are passed as ``constraints=`` callables (BoTorch's convention --
feasible when the callable returns ``<= 0`` -- matches BoCoDe's ``c <= 0``, so the
constraints are modeled raw, with no negation anywhere).

The Chebyshev scalarization is normalized with the surrogate's **posterior mean at the
training inputs**, following BoTorch's MOBO tutorial and matching both BoCoDe ParEGO
variants.

The posterior is a ``GPyTorchPosterior`` whose mean and variance are the first two moments
of TabPFN's bar distribution, with a **diagonal** covariance (a PFN gives marginal
predictives, never a joint covariance across query points); the acquisition is therefore
built with ``cache_root=False``. See :class:`TabPFNModel` for the full statement of the
two approximations.

Needs TabPFN >= v3 (see docs/tfm_setup.md). Run::

    python -m algorithms.multi_obj_constrained.tfm_cqnparego --problem WeldedBeam --init 12 --iters 50

Sources:
N. Hollmann, S. Müller, et al. TabPFN: accurate predictions on small data with a tabular foundation model. Nature, 2025. https://github.com/PriorLabs/TabPFN
S. Daulton, M. Balandat, and E. Bakshy. Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement. NeurIPS 2021. https://arxiv.org/abs/2105.08195
J. Knowles. ParEGO: a hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems. IEEE TEVC 2006.
BoTorch's PFN-as-a-Model wrapper (the template for TabPFNModel): https://github.com/meta-pytorch/botorch/blob/main/botorch_community/models/prior_fitted_network.py
BoTorch constrained multi-objective tutorial: https://botorch.org/docs/tutorials/constrained_multi_objective_bo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex

from .._bo_utils import (
    DTYPE,
    MultiObjectiveProblem,
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
from .._tfm_utils import TabPFNModel, TabPFNSurrogate


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """TFM-CqNParEGO constrained multi-objective BO over the unit cube.

    ``n_init`` defaults to the dimension-scaled BoCoDe default (:func:`default_n_init`).
    TabPFN inference + acquisition optimization run on ``device`` (default cuda when
    available); the objective is evaluated on CPU. With ``checkpoint`` set, the run is
    resumable: it restores ``(X, Y, C, completed_iters, RNG, ref_point)`` and saves after
    every iter (the ``ref_point`` is checkpointed so resume keeps the same reference).
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = MultiObjectiveProblem(problem)
    m, nc = obj.num_objectives, obj.num_constraints
    assert nc > 0, "tfm_cqnparego requires a constrained problem"
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "tfm_cqnparego",
        type(problem).__name__,
        seed,
        acquisition_function="qLogNEI",
    )

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, data = load_checkpoint(checkpoint, res)
        train_C = torch.tensor(data["C"], dtype=DTYPE)
        ref_point = torch.tensor(data["ref_point"], dtype=DTYPE)
    else:
        train_X = initial_design(n_init, obj.dim, seed)
        train_Y, train_C = obj.evaluate_raw(train_X)
        ref_point = obj.hv_ref_point(train_Y)
        start_it = 0

    # Constraint callables index the constraint model outputs (m..m+nc-1); feasible when
    # the value is <= 0, matching BoCoDe's convention (no negation).
    constraint_callables = [(lambda Z, i=m + j: Z[..., i]) for j in range(nc)]

    def feasible_hv(Y, C):
        feas = (C <= 0).all(dim=1)
        if not feas.any():
            return 0.0
        return (
            DominatedPartitioning(ref_point=ref_point.cpu(), Y=Y[feas].cpu())
            .compute_hypervolume()
            .item()
        )

    if start_it == 0:
        res.start(feasible_hv(train_Y, train_C))

    surrogate = TabPFNSurrogate(device=str(dev))
    bounds_dev = obj.bounds.to(dev)
    for it in range(start_it, iters):
        X_dev = train_X.to(dev)
        # One TabPFN output per objective, then one per constraint (objectives first).
        # No fitting: TabPFN is frozen and conditions on (X, Y) in-context.
        outputs = torch.cat([train_Y, train_C], dim=1).to(dev)
        model = TabPFNModel(surrogate, X_dev, outputs)
        # Normalize the Chebyshev scalarization with the surrogate's posterior mean at
        # the observed inputs. The model is objectives-first, so take columns [..., :m].
        with torch.no_grad():
            pred = model.posterior(X_dev).mean[..., :m]
        weights = sample_simplex(m, dtype=DTYPE, device=dev).squeeze()
        scalarization = get_chebyshev_scalarization(weights=weights, Y=pred)

        # Apply the scalarization only to the objective outputs (0..m-1); the remaining
        # outputs are the constraints, handled by `constraints=` below.
        def scalarized_objective(Z, X=None, s=scalarization, mm=m):
            return s(Z[..., :mm])

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acqf = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_dev,
            objective=GenericMCObjective(scalarized_objective),
            constraints=constraint_callables,
            sampler=sampler,
            prune_baseline=True,
            # TabPFN's posterior has no joint covariance across points (see TabPFNModel),
            # which is exactly what the cached-Cholesky root exists to exploit.
            cache_root=False,
        )
        candidate, acq_value = optimize_acqf(
            acqf,
            bounds=bounds_dev,
            q=1,
            num_restarts=10,  # tutorial's NUM_RESTARTS
            raw_samples=512,  # tutorial's RAW_SAMPLES
            options={"batch_limit": 5, "maxiter": 200},
        )
        candidate = candidate.detach().to(device="cpu", dtype=DTYPE)
        y, c = obj.evaluate_raw(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        train_C = torch.cat([train_C, c], dim=0)
        res.record(feasible_hv(train_Y, train_C), acq_value=acq_value.item())
        if checkpoint:
            res.set_history(train_X, train_Y, n_init, c=train_C)
            save_checkpoint(
                checkpoint,
                train_X,
                train_Y,
                res,
                it + 1,
                extra={
                    "C": train_C.cpu().numpy(),
                    "ref_point": ref_point.cpu().numpy(),
                },
            )
    res.set_history(train_X, train_Y, n_init, c=train_C)
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
