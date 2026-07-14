"""Shared evaluation harness for the BoCoDe algorithm scripts.

Following the CleanRL philosophy, the optimization *logic* of each algorithm lives
in its own single file; this module only holds the shared plumbing that every
script needs: reproducible seeding, the two search-space adapters
(problem-optimization over the unit cube vs. dataset-optimization over a discrete
candidate pool), a GP-fitting helper, and a small results container.

Conventions
-----------
* BoCoDe maximizes the returned objective, so every adapter exposes a single
  objective ``Y`` (shape ``(n, 1)``) that is *maximized*.
* The continuous search space is the unit cube ``[0, 1]^d``; the adapter scales
  proposals to the problem's true bounds before evaluating.
* Constraints follow BoCoDe's convention (``c <= 0`` is feasible). Unconstrained
  algorithms fold them into a feasibility-aware objective via :func:`penalized`.

Sources:
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020. http://arxiv.org/abs/1910.06403
"""

from __future__ import annotations

import inspect
import random
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

import bocode

DTYPE = torch.double


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def initial_design(n: int, dim: int, seed: int) -> torch.Tensor:
    """Latin Hypercube initial design over the unit cube ``[0, 1]^d``.

    Uses SciPy's ``qmc.LatinHypercube`` for a space-filling, seeded initial set of
    points (the design from which every BO loop starts before fitting a GP).
    """
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    return torch.from_numpy(sampler.random(n)).to(DTYPE)


def _scale_clamped(problem, X_unit: torch.Tensor) -> torch.Tensor:
    """Scale unit-cube proposals to the problem bounds, clamped to stay inside.

    ``optimize_acqf`` can return points a hair outside ``[0, 1]`` and the scaling
    multiply can overshoot the bound by a float epsilon; some problems (BoTorch
    test functions) validate their input bounds strictly, so we clamp to be safe.
    """
    X_unit = X_unit.clamp(0.0, 1.0).to(DTYPE)
    X = problem.scale(X_unit)
    bounds = problem.torch_bounds.to(X)
    # Some problems store bounds as (upper, lower); take the true low/high so the
    # inset clamp below doesn't collapse every value to one corner.
    lo = torch.minimum(bounds[:, 0], bounds[:, 1])
    hi = torch.maximum(bounds[:, 0], bounds[:, 1])
    # Inset by a tiny relative epsilon so the point stays strictly inside the
    # bounds even after the float32 cast inside evaluate() (BoTorch test problems
    # validate their input bounds strictly).
    eps = 1e-6 * (hi - lo).clamp_min(1e-12)
    X = torch.maximum(torch.minimum(X, hi - eps), lo + eps)
    # Mixed-variable problems: project the continuous proposal onto valid points
    # (round integers / snap categoricals). No-op for continuous problems.
    if getattr(problem, "is_mixed_variable", False):
        X = problem.enforce_variable_types(X)
    return X


def penalized(values: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
    """Fold constraint violations into the objective for unconstrained solvers.

    Subtracts the total positive violation (``sum(max(c, 0))``) from the objective
    so infeasible points are penalized. Feasible points are unchanged.
    """
    if constraints is None or constraints.numel() == 0:
        return values
    violation = constraints.clamp(min=0).sum(dim=1, keepdim=True)
    return values - violation


class ProblemObjective:
    """Continuous single-objective maximization over the unit cube ``[0, 1]^d``.

    Proposals in ``[0, 1]^d`` are scaled to the problem's bounds and evaluated.
    ``__call__`` returns the (penalized) objective to maximize; ``evaluate_raw``
    returns the unmodified ``(values, constraints)`` for constraint-aware solvers.
    """

    def __init__(self, problem):
        self.problem = problem
        self.dim = problem.dim
        self.num_constraints = problem.num_constraints
        self.bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]).to(
            DTYPE
        )

    def evaluate_raw(self, X_unit: torch.Tensor):
        X = _scale_clamped(self.problem, X_unit)
        values, constraints = self.problem.evaluate(X)
        return values[:, :1].to(DTYPE), (
            None if constraints is None else constraints.to(DTYPE)
        )

    def __call__(self, X_unit: torch.Tensor) -> torch.Tensor:
        values, constraints = self.evaluate_raw(X_unit)
        return penalized(values, constraints)


class DatasetObjective:
    """Discrete single-objective maximization over a fixed candidate pool.

    Wraps a :class:`bocode.opt_problems.materials._dataset_problem.MaterialsDatasetProblem`
    (or any problem exposing ``candidates`` and ``values``). Features are min-max
    normalized to ``[0, 1]`` so a GP sees a unit-cube input. ``select`` returns the
    objective for chosen candidate indices.
    """

    def __init__(self, dataset_problem):
        X = dataset_problem.candidates.to(DTYPE)
        self._lo = X.min(dim=0).values
        self._span = (X.max(dim=0).values - self._lo).clamp_min(1e-12)
        self.X = (X - self._lo) / self._span
        self.Y = dataset_problem.values.to(DTYPE).reshape(-1, 1)
        self.dim = self.X.shape[1]
        self.n_candidates = self.X.shape[0]

    def select(self, idx: torch.Tensor) -> torch.Tensor:
        return self.Y[idx]


class MultiObjectiveProblem:
    """Continuous multi-objective maximization over the unit cube ``[0, 1]^d``.

    Exposes the full objective vector (shape ``(n, m)``, maximized) plus the
    constraints, and the hypervolume reference point.

    The reference point **must** be a fixed property of the problem, never of the
    run: if it were derived from the data a given algorithm happened to observe,
    every algorithm would be scored against a different reference point and the
    hypervolumes would not be comparable — which is the whole point of the metric.
    Every multi-objective problem in BoCoDe therefore defines ``ref_point``; see
    :meth:`hv_ref_point`.
    """

    def __init__(self, problem):
        self.problem = problem
        self.dim = problem.dim
        self.num_objectives = problem.num_objectives
        self.num_constraints = problem.num_constraints
        self.bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]).to(
            DTYPE
        )
        rp = getattr(problem, "ref_point", None)
        self.ref_point = None if rp is None else torch.as_tensor(rp, dtype=DTYPE)

    def evaluate_raw(self, X_unit: torch.Tensor):
        X = _scale_clamped(self.problem, X_unit)
        values, constraints = self.problem.evaluate(X)
        return values.to(DTYPE), (
            None if constraints is None else constraints.to(DTYPE)
        )

    def hv_ref_point(self, Y: torch.Tensor) -> torch.Tensor:
        """The problem's fixed hypervolume reference point.

        ``Y`` is only used for the *fallback*: if the problem does not define a
        ``ref_point``, one is inferred from the observed data (slightly below the
        nadir) and a loud warning is emitted, because a data-dependent reference
        point makes the resulting hypervolumes incomparable across algorithms.
        """
        if self.ref_point is not None:
            return self.ref_point
        warnings.warn(
            f"{type(self.problem).__name__} defines no ref_point: inferring one "
            "from this run's observed data. The resulting hypervolume is NOT "
            "COMPARABLE with any other algorithm's — every algorithm sees "
            "different data and so gets a different reference point. Give the "
            "problem a fixed ref_point before reporting its hypervolume.",
            RuntimeWarning,
            stacklevel=2,
        )
        nadir = Y.min(dim=0).values
        span = (Y.max(dim=0).values - nadir).clamp_min(1e-9)
        return nadir - 0.1 * span

    # Backwards-compatible alias for the pre-fix name. The method no longer
    # "infers" anything unless the problem is missing its ref_point.
    infer_ref_point = hv_ref_point


def fit_gp(train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    """Fit a SingleTaskGP with input normalization and output standardization."""
    model = SingleTaskGP(
        train_X=train_X.to(DTYPE),
        train_Y=train_Y.to(DTYPE),
        input_transform=Normalize(d=train_X.shape[-1]),
        outcome_transform=Standardize(m=train_Y.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


@dataclass
class Result:
    """Full per-iteration trace of one optimization run.

    Index 0 of every per-iteration list is the **initial design** (the best value
    observed in the initial sample, at wall-time 0); index ``k`` is the state after
    BO iteration ``k``. So a run of ``iters`` iterations has ``iters + 1`` entries.
    """

    algorithm: str
    problem: str
    seed: int
    acquisition_function: str = "none"
    per_iteration_value: list = field(default_factory=list)  # best-so-far per iter
    wall_time: list = field(default_factory=list)  # seconds since trial start
    mean: list = field(default_factory=list)  # GP posterior mean at the chosen point
    variance: list = field(default_factory=list)  # GP posterior var at the chosen point
    per_iteration_acquisition_function_value: list = field(default_factory=list)
    # full evaluation history: X are the queried points (unit cube [0,1]^d, the
    # first n_init rows are the initial design), y the corresponding objective in
    # the maximized convention (same as `best`), c the constraint values at those
    # points (feasible when <= 0), None for unconstrained problems; n_init is the
    # initial-design size.
    X: object = None
    y: object = None
    c: object = None
    n_init: int = 0
    _t0: float = field(default=0.0, repr=False)

    @property
    def best(self) -> float:
        return (
            self.per_iteration_value[-1] if self.per_iteration_value else float("nan")
        )

    @property
    def iterations(self) -> list:
        """Iteration indices ``[0, 1, ..., n]`` (0 = initial design)."""
        return list(range(len(self.per_iteration_value)))

    @property
    def best_history(self) -> list:
        """Backwards-compatible alias for the best-so-far trace."""
        return self.per_iteration_value

    def start(self, init_best: float) -> None:
        """Record the initial design (iteration 0) and start the wall-clock."""
        import time

        self._t0 = time.time()
        self.per_iteration_value.append(float(init_best))
        self.wall_time.append(0.0)
        self.mean.append(float("nan"))
        self.variance.append(float("nan"))
        self.per_iteration_acquisition_function_value.append(float("nan"))

    def record(
        self,
        best: float,
        mean: float = float("nan"),
        variance: float = float("nan"),
        acq_value: float = float("nan"),
    ) -> None:
        """Record one BO iteration (best-so-far, GP mean/variance, acquisition value).

        The first call (when no entries exist yet) acts as :meth:`start` — it
        starts the wall-clock and records iteration 0 — so solvers without a
        separate initial-design phase (e.g. random search) need only call
        ``record`` each step.
        """
        if not self.per_iteration_value:
            self.start(best)
            return

        import time

        self.per_iteration_value.append(float(best))
        self.wall_time.append(time.time() - self._t0)
        self.mean.append(float(mean))
        self.variance.append(float(variance))
        self.per_iteration_acquisition_function_value.append(float(acq_value))

    def set_history(self, X, y, n_init: int, c=None) -> None:
        """Store the full evaluation history: queried points ``X`` (n, dim), their
        objective ``y`` -- (n,) single-objective, (n, m) multi-objective -- their
        constraint values ``c`` (n, n_constraints; feasible when ``<= 0``, ``None``
        for unconstrained problems) and the initial-design size ``n_init`` (so
        ``X[:n_init]`` / ``y[:n_init]`` are the initial sample and the rest are the
        BO queries).

        Recording ``c`` is what makes the *feasible* hypervolume of a constrained
        multi-objective run recomputable offline from the saved ``.npz``.
        """
        import numpy as np

        def _arr(a):
            if hasattr(a, "detach"):
                a = a.detach().cpu().numpy()
            return np.asarray(a, dtype=float)

        self.X = _arr(X)
        y = _arr(y)
        # Flatten only single-objective histories; flattening an (n, m) multi-objective
        # y would interleave the objectives into one unusable vector.
        self.y = y.reshape(-1) if y.ndim < 2 or y.shape[1] == 1 else y
        self.c = None if c is None else _arr(c).reshape(len(self.X), -1)
        self.n_init = int(n_init)

    # Backwards-compatible single-argument logger (records best only).
    def log(self, current_best: float) -> None:
        if not self.per_iteration_value:
            self.start(current_best)
        else:
            self.record(current_best)

    def to_dict(self) -> dict:
        import numpy as np

        d = {
            "algorithm": self.algorithm,
            "problem": self.problem,
            "seed": self.seed,
            "acquisition_function": self.acquisition_function,
            "best": self.best,
            "n_init": self.n_init,
            "iterations": np.asarray(self.iterations),
            "per_iteration_value": np.asarray(self.per_iteration_value),
            "wall_time": np.asarray(self.wall_time),
            "mean": np.asarray(self.mean),
            "variance": np.asarray(self.variance),
            "per_iteration_acquisition_function_value": np.asarray(
                self.per_iteration_acquisition_function_value
            ),
        }
        # full evaluation history (queried points + objectives + constraints), when
        # recorded. ``c`` is an (n, n_constraints) block, feasible when <= 0; it is
        # (n, 0) for unconstrained problems. Saving it is what lets the *feasible*
        # hypervolume of a constrained multi-objective run be recomputed offline.
        # Backwards compatibility: .npz files written before this key existed simply
        # have no "c" entry, so readers should check ``"c" in data.files``.
        d["X"] = np.asarray(self.X) if self.X is not None else np.empty((0, 0))
        d["y"] = np.asarray(self.y) if self.y is not None else np.empty((0,))
        n = len(d["X"])
        d["c"] = np.asarray(self.c) if self.c is not None else np.empty((n, 0))
        return d

    def to_npz(self, path: str) -> None:
        import os

        import numpy as np

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.savez(path, **self.to_dict())


def gp_stats(model, X) -> tuple[float, float]:
    """Return the GP posterior (mean, variance) at a single point X (first output)."""
    with torch.no_grad():
        post = model.posterior(X)
        mean = post.mean.reshape(-1)[0].item()
        var = post.variance.reshape(-1)[0].item()
    return mean, var


def resolve_device(device: str | None = None) -> torch.device:
    """Compute device: explicit name, else cuda when available, else cpu."""
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def default_n_init(dim: int) -> int:
    """BoCoDe default initial-design size, scaled by problem dimension.

    20 for ``dim < 10``, 50 for ``10 <= dim < 20``, 100 for ``20 <= dim < 100``,
    200 for ``dim >= 100``.
    """
    if dim < 10:
        return 20
    if dim < 20:
        return 50
    if dim < 100:
        return 100
    return 200


def save_checkpoint(
    path: str, train_X, train_Y, res, completed_iters: int, extra: dict | None = None
) -> None:
    """Atomically write a resumable checkpoint: BO data + torch RNG + Result trace.

    ``extra`` adds named arrays to persist alongside (e.g. qNEHVI's ``ref_point``,
    which must be restored rather than recomputed from the resumed data).
    """
    import os

    import numpy as np

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = dict(
        X=train_X.detach().cpu().numpy(),
        Y=train_Y.detach().cpu().numpy(),
        completed_iters=int(completed_iters),
        per_iteration_value=np.asarray(res.per_iteration_value),
        wall_time=np.asarray(res.wall_time),
        mean=np.asarray(res.mean),
        variance=np.asarray(res.variance),
        acq=np.asarray(res.per_iteration_acquisition_function_value),
        torch_rng=torch.get_rng_state().numpy(),
    )
    if extra:
        payload.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **payload)
    os.replace(tmp, path)  # atomic: a half-written checkpoint never replaces a good one


def load_checkpoint(path: str, res):
    """Restore a checkpoint into ``res`` in place.

    Returns ``(train_X, train_Y, completed_iters, data)`` where ``data`` is the loaded
    npz (so callers can read any ``extra`` arrays, e.g. ``data["ref_point"]``).
    """
    import time

    import numpy as np

    d = np.load(path)
    res.per_iteration_value = [float(v) for v in d["per_iteration_value"]]
    res.wall_time = [float(v) for v in d["wall_time"]]
    res.mean = [float(v) for v in d["mean"]]
    res.variance = [float(v) for v in d["variance"]]
    res.per_iteration_acquisition_function_value = [float(v) for v in d["acq"]]
    res._t0 = time.time() - (res.wall_time[-1] if res.wall_time else 0.0)
    torch.set_rng_state(torch.from_numpy(d["torch_rng"]))
    return (
        torch.tensor(d["X"], dtype=DTYPE),
        torch.tensor(d["Y"], dtype=DTYPE),
        int(d["completed_iters"]),
        d,
    )


def make_problem(name: str, args=None):
    """Instantiate a registered problem, honoring ``--discrete/--continuous`` and ``--seed``.

    Problems that expose an ``is_discrete`` constructor argument (e.g. PressureVessel,
    SpeedReducer, Car, GearTrain) can be switched between their fully-continuous and
    mixed-variable formulations at launch. ``args.discrete`` is ``True`` (``--discrete``),
    ``False`` (``--continuous``), or ``None`` (use the problem's own default).

    Problems that expose a ``seed`` constructor argument are handed the RUN's seed
    (``args.seed``). A problem needs this when its objective is *defined* by a random
    draw -- e.g. SVM subsamples a 10,000-row train/test split, and a different split is
    a different objective. Seeding it from the run seed makes the whole run
    reproducible, and keeps the objective identical across algorithms at the same seed.
    """
    cls = bocode.get_problem(name)
    params = inspect.signature(cls.__init__).parameters
    kwargs = {}

    discrete = getattr(args, "discrete", None) if args is not None else None
    if discrete is not None and "is_discrete" in params:
        kwargs["is_discrete"] = discrete

    seed = getattr(args, "seed", None) if args is not None else None
    if seed is not None and "seed" in params:
        kwargs["seed"] = int(seed)

    return cls(**kwargs)


def add_common_args(parser) -> None:
    """Add --seed, --show_progress, --saved_full_experiment, and --discrete/--continuous."""
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--discrete",
        dest="discrete",
        action="store_const",
        const=True,
        default=None,
        help="force mixed/discrete-variable mode (is_discrete=True) where the problem supports it",
    )
    mode.add_argument(
        "--continuous",
        dest="discrete",
        action="store_const",
        const=False,
        help="force fully-continuous mode (is_discrete=False) where the problem supports it",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="print the best-so-far value at each iteration",
    )
    parser.add_argument(
        "--saved_full_experiment",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="save the full per-iteration trace to an .npz file "
        "(default name: <algorithm>_<problem>_seed<seed>.npz)",
    )


def finalize(res: Result, args) -> None:
    """Handle --show_progress (per-iteration print) and --saved_full_experiment (.npz)."""
    if getattr(args, "show_progress", False):
        for i, (v, t) in enumerate(
            zip(res.per_iteration_value, res.wall_time, strict=True)
        ):
            print(f"  iter {i:3d}: best = {v:.6g}   (t = {t:.2f}s)")
    print(
        f"{res.algorithm} on {res.problem}: best={res.best:.6g} "
        f"after {len(res.per_iteration_value) - 1} iterations (seed {res.seed})"
    )
    save = getattr(args, "saved_full_experiment", None)
    if save:
        path = (
            f"{res.algorithm}_{res.problem}_seed{res.seed}.npz"
            if save == "auto"
            else save
        )
        res.to_npz(path)
        print(f"saved full experiment to {path}")
