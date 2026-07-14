"""Casmopolitan: trust-region BO over mixed categorical/continuous spaces (Wan et al., 2021).

"Think Global and Act Local" builds a GP whose kernel is *tailored* to a mixed space
and restricts the acquisition optimization to a **trust region defined jointly** over
the categorical and the continuous variables:

* **Kernel** (``bo/kernels.py`` in the official code). The categorical dimensions get a
  *transformed overlap* (exponentiated Hamming) kernel
  ``k_h(h, h') = exp( sum_i l_i * 1[h_i = h'_i] / sum_i l_i )`` with ARD lengthscales,
  the continuous dimensions get an ARD Matern-5/2 kernel, and the two are combined as
  **both a sum and a product** (the CoCaBO-style mixture, Eq. 5 of the paper)::

      k((h, x), (h', x')) = (1 - lam) * (k_h + k_x) + lam * k_h * k_x

  with the mixture weight ``lam in [0, 1]`` learned by marginal likelihood.
* **Trust region** (``bo/localbo_cat.py``, ``bo/localbo_mixed.py``). The TR is the
  intersection of a *Hamming ball* of radius ``L_h`` around the incumbent's categorical
  configuration and a *box* of side ``L_x`` (rescaled per dimension by the continuous
  kernel's lengthscales) around the incumbent's continuous part. The usual TuRBO
  success/failure counters expand/shrink both radii by ``1.5x``; when both collapse the
  run restarts, with the restart centre chosen by a *guided restart*: an auxiliary GP is
  fit over the best point of every previous restart and the new centre maximizes its UCB.
* **Acquisition** (EI, in-fill variant) is maximized *inside* the TR by **interleaved
  search**: Adam steps on the continuous dimensions (clipped to the TR box) alternating
  with local search over categorical neighbours (Hamming distance <= ``L_h``).

Observations are Gaussian-copula standardized (rank -> normal quantile), as in the
official code. BoCoDe maximizes, so the whole file is written in the maximization
convention (the official code minimizes); the copula transform is rank-based and
monotone, so this is an exact mirror of it.

Integer dimensions are treated as continuous dimensions with the Garrido-Merchan &
Hernandez-Lobato rounding ("wrapping") inside the Matern kernel, exactly as the official
code's ``int_constrained_dims`` does; only ``variable_types`` entries that are an
explicit list of levels are modelled as categorical.

Run::

    python -m algorithms.single_obj_mixed_variable.casmopolitan --problem Ackley5Mixed --init 20 --iters 40
    python -m algorithms.single_obj_mixed_variable.casmopolitan --problem BraninLVGP --iters 40 --checkpoint cas.npz

Sources:
X. Wan, V. Nguyen, H. Ha, B. Ru, C. Lu, and M. A. Osborne. Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces. ICML 2021. https://arxiv.org/abs/2102.07188
Official implementation: https://github.com/xingchenwan/Casmopolitan
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm, rankdata
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
    resolve_device,
    save_checkpoint,
    set_seed,
)
from .single_task_gp import _discrete_grids, _snap

# Trust-region hyperparameters (bo/localbo_cat.py defaults in the official code).
LENGTH_INIT = 0.8  #: initial side of the continuous TR box
LENGTH_MIN = 0.5**7
LENGTH_MAX = 1.6
#: initial Hamming radius; effectively capped at the number of categorical dims when the
#: TR is sampled, so a >20-dim radius simply means "the whole categorical space".
LENGTH_INIT_DISCRETE = 20
LENGTH_MIN_DISCRETE = 1
LENGTH_MAX_DISCRETE = 30
TR_MULTIPLIER = 1.5
SUCCTOL = 3
FAILTOL = 40
#: GP hyperparameter-fitting steps (Adam), as in the official ``train_gp`` calls.
N_TRAINING_STEPS = 300
#: interleaved search: restarts / alternation interval / steps (official defaults).
N_RESTART = 3
INTERVAL = 1
N_STEPS = 200


# --------------------------------------------------------------------------------------
# search-space bookkeeping
# --------------------------------------------------------------------------------------
class _Space:
    """The problem's mixed space, and the map between the unit cube and the model space.

    BoCoDe's unit cube is the interface to the objective; the GP instead sees the
    *model space*, in which a categorical dimension holds the **ordinal index** of its
    level (0, 1, ..., K-1) so that the overlap kernel's equality test is exact regardless
    of how the levels happen to be spaced in the box.
    """

    def __init__(self, problem):
        types = problem.resolved_variable_types()
        grids = _discrete_grids(problem)  # unit-cube values of every non-continuous dim
        self.dim = problem.dim
        self.cat_dims = [i for i, t in enumerate(types) if not isinstance(t, str)]
        self.cont_dims = [i for i, t in enumerate(types) if isinstance(t, str)]
        self.levels = [torch.tensor(grids[i], dtype=DTYPE) for i in self.cat_dims]
        self.n_categories = [len(v) for v in self.levels]
        # integer dims, as positions *within the continuous block* (for kernel wrapping)
        self.int_pos = [
            k for k, i in enumerate(self.cont_dims) if types[i] == "integer"
        ]
        self.int_grids = {i: grids[i] for i in self.cont_dims if types[i] == "integer"}

    def to_model(self, X_unit: torch.Tensor) -> torch.Tensor:
        """Unit cube -> model space (categorical dims become ordinal indices)."""
        X = X_unit.clone()
        for k, d in enumerate(self.cat_dims):
            v = self.levels[k].to(X)
            X[:, d] = (X[:, d].unsqueeze(-1) - v).abs().argmin(dim=-1).to(X.dtype)
        return X

    def to_unit(self, X_model: torch.Tensor) -> torch.Tensor:
        """Model space -> unit cube (ordinal indices become their level values)."""
        X = X_model.clone()
        for k, d in enumerate(self.cat_dims):
            v = self.levels[k].to(X)
            idx = X[:, d].round().long().clamp(0, len(v) - 1)
            X[:, d] = v[idx]
        return _snap(X, self.int_grids)  # integer dims -> nearest integer


def _copula_standardize(y: np.ndarray) -> np.ndarray:
    """Gaussian-copula transform: rank the observations, then map to normal quantiles."""
    y = np.nan_to_num(np.asarray(y, dtype=float).ravel())
    return norm.ppf((rankdata(y) - 0.5) / len(y))


# --------------------------------------------------------------------------------------
# the tailored mixed kernel (bo/kernels.py)
# --------------------------------------------------------------------------------------
class _TransformedOverlap(Kernel):
    """Exponentiated-Hamming ("transformed overlap") kernel over categorical dims."""

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # 1 where the two configurations agree on a dimension, 0 where they differ
        same = ((x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs() < 1e-5).to(x1.dtype)
        ls = self.lengthscale
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k = torch.exp((same * ls).sum(dim=-1) / ls.sum())
        else:
            k = torch.exp(ls.squeeze(-1) * same.sum(dim=-1) / x1.shape[-1])
        return k.diagonal(dim1=-2, dim2=-1) if diag else k


class _WrappedMatern(MaternKernel):
    """Matern kernel that rounds the integer dimensions of its inputs before use.

    The "wrapping" transformation of Garrido-Merchan & Hernandez-Lobato (2020), used by
    Casmopolitan for its integer-constrained continuous dimensions.
    """

    def __init__(self, int_pos=None, **kwargs):
        super().__init__(**kwargs)
        self.int_pos = list(int_pos or [])

    def _wrap(self, x):
        if not self.int_pos:
            return x
        x = x.clone()
        x[..., self.int_pos] = x[..., self.int_pos].round()
        return x

    def forward(self, x1, x2, diag=False, **params):
        return super().forward(self._wrap(x1), self._wrap(x2), diag=diag, **params)


class _MixtureKernel(Kernel):
    """``(1 - lam) * (k_cat + k_cont) + lam * k_cat * k_cont`` (Eq. 5 of the paper)."""

    def __init__(self, cat_dims, cont_dims, int_pos, ard=True, **kwargs):
        super().__init__(**kwargs)
        self.cat_dims, self.cont_dims = list(cat_dims), list(cont_dims)
        ls_constraint = Interval(0.01, 0.5) if ard else Interval(0.01, 2.5)
        self.categorical_kern = _TransformedOverlap(
            ard_num_dims=len(cat_dims) if ard else None,
            lengthscale_constraint=ls_constraint,
        )
        self.continuous_kern = _WrappedMatern(
            nu=2.5,
            int_pos=int_pos,
            ard_num_dims=len(cont_dims) if ard else None,
            lengthscale_constraint=ls_constraint,
        )
        self.register_parameter("raw_lamda", torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_lamda", Interval(0.0, 1.0))

    @property
    def lamda(self):
        return self.raw_lamda_constraint.transform(self.raw_lamda)

    def forward(self, x1, x2, diag=False, **params):
        x1_cont, x2_cont = x1[..., self.cont_dims], x2[..., self.cont_dims]
        # the categorical kernel is not differentiable w.r.t. its inputs: detach so the
        # interleaved search's Adam steps only differentiate through the continuous part.
        x1_cat, x2_cat = (
            x1[..., self.cat_dims].detach(),
            x2[..., self.cat_dims].detach(),
        )
        k_cat = self.categorical_kern.forward(x1_cat, x2_cat, diag=diag, **params)
        k_cont = self.continuous_kern.forward(x1_cont, x2_cont, diag=diag, **params)
        lam = self.lamda.to(k_cat)
        return (1.0 - lam) * (k_cat + k_cont) + lam * k_cat * k_cont


class _CasmopolitanGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            kernel, outputscale_constraint=Interval(0.5, 5.0)
        )

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def _make_kernel(space: _Space, ard: bool) -> Kernel:
    """The mixed kernel, degenerating gracefully to a pure categorical/continuous one."""
    ls_constraint = Interval(0.01, 0.5) if ard else Interval(0.01, 2.5)
    if not space.cat_dims:
        return _WrappedMatern(
            nu=2.5,
            int_pos=space.int_pos,
            ard_num_dims=len(space.cont_dims) if ard else None,
            lengthscale_constraint=ls_constraint,
        )
    if not space.cont_dims:
        return _TransformedOverlap(
            ard_num_dims=len(space.cat_dims) if ard else None,
            lengthscale_constraint=ls_constraint,
        )
    return _MixtureKernel(space.cat_dims, space.cont_dims, space.int_pos, ard=ard)


def _train_gp(X: torch.Tensor, y: torch.Tensor, space: _Space, ard=True, steps=None):
    """Fit the Casmopolitan GP by Adam on the exact marginal likelihood."""
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 0.1)).to(X)
    model = _CasmopolitanGP(X, y, likelihood, _make_kernel(space, ard)).to(X)
    model.initialize(**{"likelihood.noise": 0.005, "covar_module.outputscale": 1.0})
    mll = ExactMarginalLogLikelihood(likelihood, model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(steps if steps is not None else N_TRAINING_STEPS):
        optimizer.zero_grad()
        loss = -mll(model(X), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# trust region + interleaved search (bo/localbo_utils.py)
# --------------------------------------------------------------------------------------
def _hamming(x_cat: np.ndarray, y_cat: np.ndarray) -> int:
    return int((np.abs(x_cat - y_cat) > 1e-5).sum())


def _sample_neighbour(x_cat: np.ndarray, n_categories: list) -> np.ndarray:
    """One categorical neighbour: re-draw a single dimension to a different level."""
    x = x_cat.copy()
    j = random.randint(0, len(n_categories) - 1)
    options = [v for v in range(n_categories[j]) if v != int(round(x[j]))]
    if options:
        x[j] = random.choice(options)
    return x


def _sample_within_hamming_tr(
    x_cat: np.ndarray, radius: int, n_categories: list
) -> np.ndarray:
    """A uniform sample from the Hamming ball of the given radius around ``x_cat``."""
    x = x_cat.copy()
    n_change = int(min(max(radius, 1), len(n_categories)))
    for j in random.sample(range(len(n_categories)), n_change):
        x[j] = float(random.randrange(n_categories[j]))
    return x


def _ei(model, X: torch.Tensor, mu_star: torch.Tensor) -> torch.Tensor:
    """Augmented expected improvement, in the *maximization* convention."""
    with gpytorch.settings.fast_pred_var():
        preds = model(X)
    mean, std = preds.mean, preds.stddev.clamp_min(1e-9)
    u = (mean - mu_star) / std
    gauss = torch.distributions.Normal(
        torch.zeros(1, dtype=X.dtype), torch.ones(1, dtype=X.dtype)
    )
    ei = std * torch.exp(gauss.log_prob(u)) + (mean - mu_star) * gauss.cdf(u)
    sigma_n = model.likelihood.noise
    return ei * (1.0 - sigma_n.sqrt() / (sigma_n + std**2).sqrt())


def _interleaved_search(space, model, x_center, lb, ub, radius, mu_star, dtype):
    """Maximize EI inside the TR: Adam on the continuous dims, local search on the cats.

    ``x_center`` is the incumbent in model space (numpy); ``lb``/``ub`` bound the
    continuous TR box; ``radius`` is the Hamming radius of the categorical TR.
    """
    cat, cont = space.cat_dims, space.cont_dims

    def acq(x_np_or_torch) -> torch.Tensor:
        X = x_np_or_torch
        if not torch.is_tensor(X):
            X = torch.tensor(np.asarray(X), dtype=dtype)
        return _ei(model, X.reshape(1, -1), mu_star).reshape(())

    lb_t, ub_t = torch.tensor(lb, dtype=dtype), torch.tensor(ub, dtype=dtype)
    sobol = SobolEngine(max(len(cont), 1), scramble=True, seed=random.randint(0, 10**6))
    starts_cont = lb + (ub - lb) * sobol.draw(N_RESTART).to(dtype).numpy()

    best_x, best_acq = None, -float("inf")
    for r in range(N_RESTART):
        x = x_center.copy()
        if cat:
            x[cat] = _sample_within_hamming_tr(
                x_center[cat], radius, space.n_categories
            )
        if cont:
            x[cont] = starts_cont[r]
        x_cat, x_cont = x[cat].copy(), x[cont].copy()
        acq_x = float(acq(x))

        for _ in range(0, N_STEPS, INTERVAL):
            # (a) continuous dims, categorical part frozen
            if cont:
                x_cont_t = torch.tensor(x_cont, dtype=dtype).requires_grad_(True)
                x_cat_t = torch.tensor(x_cat, dtype=dtype)
                opt = torch.optim.Adam([x_cont_t], lr=0.1)
                for _ in range(INTERVAL):
                    opt.zero_grad()
                    full = torch.empty(space.dim, dtype=dtype)
                    full[cat] = x_cat_t
                    full[cont] = x_cont_t
                    (-acq(full)).backward()
                    opt.step()
                    with torch.no_grad():
                        x_cont_t.data = torch.max(torch.min(x_cont_t, ub_t), lb_t)
                x_cont = x_cont_t.detach().numpy()

            # (b) categorical dims, continuous part frozen
            for _ in range(INTERVAL if cat else 0):
                neighbour = _sample_neighbour(x_cat, space.n_categories)
                if _hamming(x_center[cat], neighbour) > radius:
                    continue
                cand = x.copy()
                cand[cat], cand[cont] = neighbour, x_cont
                acq_n = float(acq(cand))
                if acq_n > acq_x:
                    x_cat, acq_x = neighbour, acq_n

        x[cat], x[cont] = x_cat, x_cont
        acq_x = float(acq(x))
        if acq_x > best_acq:
            best_x, best_acq = x.copy(), acq_x
    return best_x, best_acq


def _tr_bounds(model, space, x_center, length):
    """The continuous TR box: a hyper-rectangle rescaled by the Matern lengthscales."""
    if not space.cont_dims:
        return np.zeros(0), np.zeros(0)
    kern = model.covar_module.base_kernel
    cont_kern = getattr(kern, "continuous_kern", kern)
    w = cont_kern.lengthscale.detach().cpu().numpy().ravel()
    w = w / w.mean()
    w = w / np.prod(np.power(w, 1.0 / len(w)))  # so that w.prod() == 1
    c = x_center[space.cont_dims]
    return (
        np.clip(c - w * length / 2.0, 0.0, 1.0),
        np.clip(c + w * length / 2.0, 0.0, 1.0),
    )


# --------------------------------------------------------------------------------------
# the BO loop
# --------------------------------------------------------------------------------------
def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 40,
    seed: int = 0,
    checkpoint: str | None = None,
    device: str | None = None,
) -> Result:
    """Casmopolitan over the problem's mixed categorical/continuous space."""
    set_seed(seed)
    resolve_device(device)  # honored for parity; the mixed GP is tiny and runs on CPU
    obj = ProblemObjective(problem)
    space = _Space(problem)
    if n_init is None:
        n_init = default_n_init(obj.dim)
    res = Result(
        "casmopolitan", type(problem).__name__, seed, acquisition_function="EI"
    )

    n_cand = min(100 * space.dim, 5000)

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, data = load_checkpoint(checkpoint, res)
        length = float(data["length"])
        length_discrete = int(data["length_discrete"])
        succcount, failcount = int(data["succcount"]), int(data["failcount"])
        restart_at = int(data["restart_at"])
        queue = list(data["queue"].reshape(-1, space.dim))
        best_X_restart = data["best_X_restart"]
        best_Y_restart = data["best_Y_restart"]
    else:
        # LHS over the unit cube, snapped onto the valid levels / integers
        train_X = _snap(initial_design(n_init, obj.dim, seed), _discrete_grids(problem))
        train_Y = obj(train_X)
        res.start(train_Y.max().item())
        length, length_discrete = LENGTH_INIT, LENGTH_INIT_DISCRETE
        succcount = failcount = 0
        restart_at = (
            0  # index in the history where the current trust region's data starts
        )
        queue: list = []  # points of a fresh (re)start design, still to be evaluated
        best_X_restart = np.zeros((0, space.dim))
        best_Y_restart = np.zeros((0,))
        start_it = 0

    for it in range(start_it, iters):
        mean = var = acq_value = float("nan")
        if queue:
            x_model = queue.pop(0)
            candidate = space.to_unit(torch.tensor(x_model, dtype=DTYPE).reshape(1, -1))
        else:
            X_tr = space.to_model(train_X[restart_at:])
            Y_tr = torch.tensor(
                _copula_standardize(train_Y[restart_at:].numpy()), dtype=DTYPE
            )
            model = _train_gp(X_tr, Y_tr, space)

            x_center = X_tr[Y_tr.argmax().item()].numpy().copy()
            lb, ub = _tr_bounds(model, space, x_center, length)
            with torch.no_grad():
                mu_star = model.likelihood(
                    model(torch.tensor(x_center, dtype=DTYPE).reshape(1, -1))
                ).mean
            x_model, acq_value = _interleaved_search(
                space, model, x_center, lb, ub, length_discrete, mu_star, DTYPE
            )
            candidate = space.to_unit(torch.tensor(x_model, dtype=DTYPE).reshape(1, -1))
            with torch.no_grad():
                post = model(space.to_model(candidate))
                mean, var = post.mean.item(), post.variance.item()

        y = obj(candidate)
        # The counters only run once the current trust region's (re)start design has been
        # fully observed -- as in the official ``observe()``, which guards ``_adjust_length``
        # with ``len(_fX) >= n_init`` and calls it *before* appending the new point.
        tr_observed = len(train_X) - restart_at
        best_before = (
            train_Y[restart_at:].max().item() if tr_observed >= n_init else None
        )
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)

        if best_before is not None:
            # TuRBO-style counters, mirrored into the maximization convention.
            if y.item() >= best_before + 1e-3 * math.fabs(best_before):
                succcount, failcount = succcount + 1, 0
            else:
                succcount, failcount = 0, failcount + 1
            if succcount == SUCCTOL:
                length = min(length * TR_MULTIPLIER, LENGTH_MAX)
                length_discrete = int(
                    min(length_discrete * TR_MULTIPLIER, LENGTH_MAX_DISCRETE)
                )
                succcount = 0
            elif failcount == FAILTOL:
                length = max(length / TR_MULTIPLIER, LENGTH_MIN)
                length_discrete = int(length_discrete / TR_MULTIPLIER)
                failcount = 0

        if length <= LENGTH_MIN or length_discrete <= LENGTH_MIN_DISCRETE:
            queue, best_X_restart, best_Y_restart = _guided_restart(
                space,
                train_X[restart_at:],
                train_Y[restart_at:],
                best_X_restart,
                best_Y_restart,
                n_init,
                n_cand,
            )
            length, length_discrete = LENGTH_INIT, LENGTH_INIT_DISCRETE
            succcount = failcount = 0
            restart_at = len(train_X)

        res.record(train_Y.max().item(), mean=mean, variance=var, acq_value=acq_value)
        if checkpoint:
            res.set_history(train_X, train_Y, n_init)
            save_checkpoint(
                checkpoint,
                train_X,
                train_Y,
                res,
                it + 1,
                extra=dict(
                    length=length,
                    length_discrete=length_discrete,
                    succcount=succcount,
                    failcount=failcount,
                    restart_at=restart_at,
                    queue=np.asarray(queue, dtype=float).reshape(len(queue), space.dim),
                    best_X_restart=best_X_restart,
                    best_Y_restart=best_Y_restart,
                ),
            )

    res.set_history(train_X, train_Y, n_init)
    return res


def _guided_restart(space, X_unit, Y, best_X_restart, best_Y_restart, n_init, n_cand):
    """Pick the next TR centre with the auxiliary GP, and lay out the restart design.

    The auxiliary GP is fit on the best point of every restart so far and the new centre
    is the candidate maximizing its UCB; the restart design is then drawn inside a fresh
    trust region around that centre.
    """
    X_model = space.to_model(X_unit).numpy()
    best = int(Y.argmax().item())
    best_X_restart = np.vstack([best_X_restart, X_model[best]])
    best_Y_restart = np.concatenate([best_Y_restart, [Y[best].item()]])

    aux = _train_gp(
        torch.tensor(best_X_restart, dtype=DTYPE),
        torch.tensor(_copula_standardize(best_Y_restart), dtype=DTYPE)
        if len(best_Y_restart) > 1
        else torch.zeros(1, dtype=DTYPE),
        space,
        ard=False,
        steps=N_TRAINING_STEPS,
    )
    cand = np.random.rand(n_cand, space.dim)
    for k, d in enumerate(space.cat_dims):
        cand[:, d] = np.random.randint(0, space.n_categories[k], size=n_cand)
    with torch.no_grad():
        post = aux(torch.tensor(cand, dtype=DTYPE))
        ucb = (post.mean + 1.96 * post.variance.sqrt()).numpy()
    centre = cand[int(np.argmax(ucb))]

    # Sobol perturbations around the centre inside a fresh continuous TR (as in TuRBO),
    # and uniform samples inside a fresh Hamming ball for the categorical part.
    design = np.tile(centre, (n_init, 1))
    if space.cont_dims:
        c = centre[space.cont_dims]
        lb = np.clip(c - LENGTH_INIT / 2.0, 0.0, 1.0)
        ub = np.clip(c + LENGTH_INIT / 2.0, 0.0, 1.0)
        sobol = SobolEngine(
            len(space.cont_dims), scramble=True, seed=random.randint(0, 10**6)
        )
        pert = lb + (ub - lb) * sobol.draw(n_init).to(DTYPE).numpy()
        prob_perturb = min(20.0 / len(space.cont_dims), 1.0)
        mask = np.random.rand(n_init, len(space.cont_dims)) <= prob_perturb
        empty = np.where(mask.sum(axis=1) == 0)[0]
        mask[empty, np.random.randint(0, len(space.cont_dims), size=len(empty))] = True
        block = design[:, space.cont_dims]
        block[mask] = pert[mask]
        design[:, space.cont_dims] = block
    for i in range(n_init):
        if space.cat_dims:
            design[i, space.cat_dims] = _sample_within_hamming_tr(
                centre[space.cat_dims], LENGTH_INIT_DISCRETE, space.n_categories
            )
    return list(design), best_X_restart, best_Y_restart


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument(
        "--init",
        type=int,
        default=None,
        help="initial design size (default: dim-scaled)",
    )
    parser.add_argument("--iters", type=int, default=40)
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
