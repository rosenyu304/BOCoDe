"""DGEMO for multi-objective (unconstrained) problems.

Diversity-Guided Efficient Multi-Objective Optimization (Konakovic Lukovic, Tian &
Matusik, NeurIPS 2020). DGEMO is a *batch* MOBO method: each iteration it (1) fits an
independent GP per objective, (2) uses the GP posterior mean as the acquisition
``f_tilde_j = mu_j`` and approximates the Pareto set/front over ``f_tilde`` with an
evolutionary solver, (3) partitions that approximate Pareto set into *diversity
regions* in joint design+performance space, and (4) greedily selects a batch that
maximizes hypervolume improvement while spreading the picks across the regions as
evenly as possible (Algorithm 1 + the greedy Algorithm 2 of the paper).

Fidelity / adaptation notes
---------------------------
This is a faithful adaptation of the DGEMO *algorithm* to BoCoDe's stack, NOT a verbatim
port of the official code (github.com/yunshengtian/DGEMO). Two components of the
reference implementation cannot be vendored additively here:

* The reference's Pareto-front approximation is the first-order manifold "ParetoDiscovery"
  of Schulz et al. (2018), built on pymoo 0.4.x internals and autograd Jacobians/Hessians.
  BoCoDe ships pymoo 0.6.2 (whose API is incompatible) and every other MO method imports
  ``pymoo.algorithms.moo.nsga2`` from it, so downgrading is not additive. We instead
  approximate the Pareto set/front by running NSGA-II over the GP-mean surrogate problem
  ("NSGA-II-style Pareto exploration"), which is one of the solvers the reference framework
  itself supports.
* The reference groups Pareto points into diversity regions with a performance-buffer +
  graph-cut (needs the ``pygco`` C++ wrapper, not installed). We instead cluster the
  approximate Pareto set with KMeans in the joint (normalized design, normalized
  performance) space -- exactly the paper's stated grouping objective ("group the optimal
  points based on their design properties and performance").

The headline contribution -- the diversity-guided greedy batch selection (Algorithm 2:
greedily take the max-HVI point, then keep taking the max-HVI point from a *not-yet-visited*
region until every region is covered, then reset and repeat) -- is ported directly from
``mobo/solver/pareto_discovery/utils.py::propose_next_batch``, with the hypervolume computed
in BoCoDe's maximization frame via BoTorch's ``DominatedPartitioning`` (the reference uses
pymoo's HV indicator; the greedy logic is identical).

Handles the MO-UNCONSTRAINED problem class only.

Run::

    python -m algorithms.multi_obj.dgemo --problem BraninCurrin --iters 40 --batch_size 5

Sources:
M. Konakovic Lukovic, Y. Tian, and W. Matusik. Diversity-Guided Multi-Objective Bayesian Optimization With Batch Evaluations. NeurIPS 2020. https://cdfg.mit.edu/assets/files/DGEMO.pdf
Reference implementation: https://github.com/yunshengtian/DGEMO
A. Schulz et al. Interactive Exploration of Design Trade-Offs. ACM TOG 37(4), 2018 (the Pareto-discovery method DGEMO builds on).
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import SumMarginalLogLikelihood
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from sklearn.cluster import KMeans

from .._bo_utils import (
    DTYPE,
    MultiObjectiveProblem,
    Result,
    add_common_args,
    default_n_init,
    initial_design,
    make_problem,
    resolve_device,
    robust_fit_mll,
    set_seed,
)


def _fit(train_X, train_Y):
    """One independent GP per objective (Normalize inputs, Standardize outputs).

    The Standardize outcome transform means ``posterior(X).mean`` comes back in the RAW
    maximization frame, so the GP mean is directly usable as DGEMO's acquisition
    ``f_tilde_j = mu_j`` and its hypervolume is comparable to the other MO baselines.
    """
    dim = train_X.shape[-1]
    models = [
        SingleTaskGP(
            train_X,
            train_Y[:, i : i + 1],
            input_transform=Normalize(d=dim),
            outcome_transform=Standardize(m=1),
        )
        for i in range(train_Y.shape[-1])
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    robust_fit_mll(mll, label="dgemo")
    return model


class _SurrogateProblem(Problem):
    """The GP-mean surrogate as a pymoo minimization problem over the unit cube.

    DGEMO's acquisition is the posterior mean ``f_tilde_j = mu_j`` (exploitation of the
    surrogate); NSGA-II explores its Pareto front. BoCoDe maximizes, pymoo minimizes, so
    the returned objectives are negated.
    """

    def __init__(self, model, dim, n_obj, dev):
        super().__init__(n_var=dim, n_obj=n_obj, n_ieq_constr=0, xl=0.0, xu=1.0)
        self._model = model
        self._dev = dev

    def _evaluate(self, X, out, *args, **kwargs):
        Xt = torch.from_numpy(np.asarray(X)).to(DTYPE).to(self._dev)
        with torch.no_grad():
            mean = self._model.posterior(Xt).mean  # raw maximization frame, (n, m)
        out["F"] = -mean.detach().cpu().numpy()  # negate: pymoo minimizes


def _approximate_pareto(model, dim, n_obj, dev, pop_size, n_gen, seed):
    """NSGA-II over the GP-mean surrogate -> approximate Pareto set (X) and front (Y).

    Returns the final non-dominated population in the unit cube ``X`` and its predicted
    objectives ``Y`` in BoCoDe's maximization frame.
    """
    surr = _SurrogateProblem(model, dim, n_obj, dev)
    result = pymoo_minimize(
        surr, NSGA2(pop_size=pop_size), ("n_gen", n_gen), seed=seed, verbose=False
    )
    X = np.atleast_2d(result.pop.get("X"))
    Y = -np.atleast_2d(result.pop.get("F"))  # back to maximization frame
    return X, Y


def _diversity_regions(approx_X, approx_Y, n_regions, seed):
    """Split the approximate Pareto set into diversity regions (Section 4.3).

    Clusters in the joint (min-max normalized design, min-max normalized performance)
    space so points are grouped by both design properties and performance -- the paper's
    grouping objective. Substitutes for the reference's performance-buffer + graph-cut.
    Returns an integer label per point.
    """
    n = len(approx_X)
    k = int(max(1, min(n_regions, n)))
    if k <= 1:
        return np.zeros(n, dtype=int)

    def _norm(A):
        A = np.asarray(A, dtype=float)
        lo = A.min(axis=0, keepdims=True)
        span = np.clip(A.max(axis=0, keepdims=True) - lo, 1e-12, None)
        return (A - lo) / span

    feats = np.hstack([_norm(approx_X), _norm(approx_Y)])
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(feats)
    return labels


def _hv(ref_point, Y):
    """Dominated hypervolume above ``ref_point`` (maximization frame)."""
    if Y.numel() == 0:
        return 0.0
    return (
        DominatedPartitioning(ref_point=ref_point.cpu(), Y=Y.cpu())
        .compute_hypervolume()
        .item()
    )


def _propose_batch(curr_pfront, ref_point, pred_Y, pred_X, labels, batch_size, rng):
    """Diversity-guided greedy batch selection -- DGEMO Algorithm 2.

    Ported from the reference ``propose_next_batch``: greedily take the candidate with the
    largest hypervolume improvement over the current Pareto front, add it, then keep taking
    the largest-HVI candidate from a region not yet visited this cycle until every region
    is covered; reset the visited set and repeat until the batch is full. HVI is computed
    with BoTorch's ``DominatedPartitioning`` in the maximization frame.

    Returns the selected unit-cube points (up to ``batch_size``).
    """
    labels = np.asarray(labels)
    n_cand = len(pred_X)
    if n_cand == 0:
        return np.zeros((0, pred_X.shape[1] if pred_X.ndim == 2 else 0))
    batch_size = min(batch_size, n_cand)

    curr = curr_pfront.clone()  # (k, m) maximization frame
    ref = ref_point
    pred_Y_t = torch.as_tensor(pred_Y, dtype=DTYPE)

    selected_mask = np.zeros(n_cand, dtype=bool)  # globally chosen
    visited_mask = np.zeros(n_cand, dtype=bool)  # visited this diversity cycle
    chosen = []

    for _ in range(batch_size):
        # start a new cycle once every not-yet-selected candidate has been visited
        avail = np.where(~selected_mask & ~visited_mask)[0]
        if len(avail) == 0:
            visited_mask[:] = selected_mask
            avail = np.where(~selected_mask)[0]
            if len(avail) == 0:
                break

        curr_hv = _hv(ref, curr)
        best_idx, best_contrib = -1, 0.0
        for idx in avail:
            cand = pred_Y_t[idx : idx + 1]
            new_hv = _hv(ref, torch.cat([curr, cand], dim=0))
            contrib = new_hv - curr_hv
            if contrib > best_contrib:
                best_contrib = contrib
                best_idx = idx
        if best_idx == -1:  # no positive HVI: pick a random unvisited candidate
            best_idx = int(rng.choice(avail))

        selected_mask[best_idx] = True
        curr = torch.cat([curr, pred_Y_t[best_idx : best_idx + 1]], dim=0)
        chosen.append(best_idx)
        # mark the whole region of the chosen point as visited for this cycle
        visited_mask[labels == labels[best_idx]] = True

    return pred_X[np.asarray(chosen, dtype=int)]


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    batch_size: int = 5,
    pop_size: int = 100,
    n_gen: int = 50,
    device: str | None = None,
) -> Result:
    """DGEMO multi-objective BO over the unit cube (unconstrained problems).

    ``iters`` is the *function-evaluation* budget beyond the initial design (matching the
    single-point MO baselines): DGEMO consumes it in batches of ``batch_size`` points, so
    it runs ``ceil(iters / batch_size)`` BO iterations. ``per_iteration_value`` records the
    running hypervolume in BoCoDe's maximization frame after each evaluation, so the trace
    is directly comparable, at equal evaluation budget, to qnehvi/qnparego. ``pop_size`` /
    ``n_gen`` size the NSGA-II Pareto-front approximation of the GP-mean surrogate.
    """
    set_seed(seed)
    dev = resolve_device(device)
    obj = MultiObjectiveProblem(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    rng = np.random.default_rng(seed)

    res = Result("dgemo", type(problem).__name__, seed, acquisition_function="mean")

    train_X = initial_design(n_init, obj.dim, seed)
    train_Y, _ = obj.evaluate_raw(train_X)
    ref_point = obj.hv_ref_point(train_Y)

    def hv_all(Y):
        return _hv(ref_point, Y)

    res.start(hv_all(train_Y))

    n_batches = math.ceil(iters / batch_size)
    recorded = 0  # evaluations recorded into the trace so far
    for _ in range(n_batches):
        if recorded >= iters:
            break
        model = _fit(train_X.to(dev), train_Y.to(dev))

        # (2) approximate Pareto set/front over the GP-mean acquisition f_tilde = mu
        approx_X, approx_Y = _approximate_pareto(
            model, obj.dim, obj.num_objectives, dev, pop_size, n_gen, seed
        )

        # (3) split into diversity regions (joint design+performance clustering)
        labels = _diversity_regions(approx_X, approx_Y, batch_size, seed)

        # (4) diversity-guided greedy batch selection over the current Pareto front
        nd = is_non_dominated(train_Y)
        curr_pfront = train_Y[nd]
        this_batch = min(batch_size, iters - recorded)
        cand = _propose_batch(
            curr_pfront, ref_point, approx_Y, approx_X, labels, this_batch, rng
        )
        if len(cand) == 0:  # degenerate: fall back to random unit-cube points
            cand = rng.random((this_batch, obj.dim))

        X_next = torch.from_numpy(np.asarray(cand)).to(DTYPE)
        Y_next, _ = obj.evaluate_raw(X_next)
        train_X = torch.cat([train_X, X_next], dim=0)
        train_Y = torch.cat([train_Y, Y_next], dim=0)

        # record one HV entry per new evaluation (per-evaluation-comparable trace)
        for j in range(len(X_next)):
            if recorded >= iters:
                break
            k = n_init + recorded + j + 1
            res.record(hv_all(train_Y[:k]))
        recorded += len(X_next)

    res.set_history(train_X, train_Y, n_init)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument(
        "--init", type=int, default=None, help="initial design size (default: dim-scaled)"
    )
    parser.add_argument(
        "--iters", type=int, default=50, help="function-eval budget beyond the init design"
    )
    parser.add_argument("--batch_size", type=int, default=5, help="DGEMO batch size b")
    parser.add_argument("--pop_size", type=int, default=100, help="NSGA-II population size")
    parser.add_argument(
        "--n_gen", type=int, default=50, help="NSGA-II generations for Pareto approximation"
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    add_common_args(parser)
    args = parser.parse_args()
    res = optimize_problem(
        make_problem(args.problem, args),
        args.init,
        args.iters,
        args.seed,
        batch_size=args.batch_size,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        device=args.device,
    )
    from .._bo_utils import finalize

    finalize(res, args)


if __name__ == "__main__":
    main()
