"""BODi: Bayesian Optimization via Dictionary-based Embeddings (Deshwal et al., 2023).

BODi models a function over a high-dimensional combinatorial (categorical / integer)
space by first mapping every discrete configuration into a **continuous** feature vector
through a *Hamming Embedding via Dictionaries* (HED), then fitting an ordinary GP on that
embedding. Concretely, a **dictionary** ``A`` of ``m`` binary/categorical *anchor*
configurations is drawn, and a point ``z`` is embedded as its vector of normalized Hamming
distances to the anchors (Sec. 3.1, Prop. 1)::

    phi_A(z) = [ d_H(z, a_1), ..., d_H(z, a_m) ] / n_discrete   in  [0, 1]^m

so ``phi_A(z)_i`` is just the fraction of discrete coordinates on which ``z`` disagrees
with anchor ``a_i``. A ``SingleTaskGP`` with a **Matern-5/2 ARD** kernel is fit on this
``m``-dimensional embedding. The key theoretical result (Thm. 4) is that the regret scales
with the *reduced cardinality* of the embedded space (governed by ``m``), not with the raw
combinatorial dimensionality, which is what lets a plain GP work in high dimensions.

Two design choices from the paper are reproduced here:

* **Diverse random dictionary** (Sec. 3.2). Each anchor row is drawn with its own sparsity:
  a Bernoulli rate ``theta ~ Uniform(0, 1)`` is sampled per row, and each discrete
  coordinate is set to a random non-baseline level with probability ``theta`` (else to the
  baseline level 0). Rows therefore span a range of Hamming radii, giving the embedding
  informative, non-degenerate features. The dictionary is **regenerated every iteration**.
* **Mixed spaces** (Sec. 3.4) use a **product kernel** ``k_HED(phi) * k_cont(x)`` -- a
  Matern-5/2 ARD on the discrete embedding times a Matern-5/2 ARD on the continuous block --
  and the acquisition is optimized by **alternating** gradient ascent on the continuous
  variables with **local (Hamming-1) hill-climbing search** over the discrete variables,
  restarted from several random configurations (Alg. 2). With no discrete dimensions the
  method degenerates to a standard Matern-ARD GP + gradient-ascent EI.

Each iteration fits the GP on the current embedding and maximizes ``LogExpectedImprovement``.
The chosen configuration is snapped to the valid grid and evaluated. Run::

    python -m algorithms.single_obj_mixed_variable.bodi --problem AckleyCat --iters 50

Sources:
A. Deshwal, S. Ament, M. Balandat, E. Bakshy, J. R. Doppa, and D. Eriksson. Bayesian Optimization over High-Dimensional Combinatorial Spaces via Dictionary-based Embeddings. AISTATS 2023. https://arxiv.org/abs/2303.01774
Reference implementation: https://github.com/aryandeshwal/BODi
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from .._bo_utils import (
    DTYPE,
    ProblemObjective,
    Result,
    add_common_args,
    default_n_init,
    finalize,
    gp_stats,
    initial_design,
    load_checkpoint,
    make_problem,
    resolve_device,
    save_checkpoint,
    set_seed,
)
from .single_task_gp import _discrete_grids, _snap

N_DICT = 128  # dictionary size m (number of anchors / embedding dimension), the paper's default
N_RESTARTS = 4  # random restarts for the alternating acquisition search
N_ALT = 3  # alternating (continuous-ascent / discrete-local-search) rounds per restart
N_LS_SWEEPS = 10  # max steepest-ascent Hamming-1 sweeps per local search
N_CONT_STEPS = 60  # Adam steps for the continuous inner optimization
LR = 0.05  # Adam step size for the continuous inner optimization


class _Space:
    """The problem's mixed space: discrete grids as ordinal indices + continuous block."""

    def __init__(self, problem):
        types = problem.resolved_variable_types()
        grids = _discrete_grids(problem)  # unit-cube values of every non-continuous dim
        self.dim = problem.dim
        self.grids = grids
        self.disc_dims = sorted(grids)  # positions of discrete (integer/categorical) dims
        self.cont_dims = [i for i, t in enumerate(types) if t == "continuous"]
        # per-discrete-dim allowed unit-cube values and their cardinality
        self.levels = [torch.tensor(grids[d], dtype=DTYPE) for d in self.disc_dims]
        self.cards = [len(v) for v in self.levels]
        self.n_disc = len(self.disc_dims)
        self.n_cont = len(self.cont_dims)

    def to_indices(self, X_unit: torch.Tensor) -> torch.Tensor:
        """Unit-cube points -> (n, n_disc) ordinal indices of the discrete dims."""
        if self.n_disc == 0:
            return torch.zeros(len(X_unit), 0, dtype=torch.long)
        cols = []
        for k, d in enumerate(self.disc_dims):
            v = self.levels[k].to(X_unit)
            cols.append((X_unit[:, d].unsqueeze(-1) - v).abs().argmin(dim=-1))
        return torch.stack(cols, dim=1)

    def assemble(self, idx: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        """(discrete indices, continuous block) -> a unit-cube point (1, dim)."""
        x = torch.zeros(self.dim, dtype=DTYPE)
        for k, d in enumerate(self.disc_dims):
            x[d] = self.levels[k][int(idx[k])]
        for c, d in enumerate(self.cont_dims):
            x[d] = cont[c]
        return x.reshape(1, -1)


def _make_dictionary(space: _Space) -> torch.Tensor:
    """A diverse random dictionary of ``N_DICT`` anchors (Sec. 3.2), as ordinal indices.

    Each anchor row samples its own sparsity ``theta ~ U(0, 1)``: a coordinate is set to a
    uniformly random non-baseline level with probability ``theta``, else to level 0. This
    spreads the rows across a range of Hamming radii, as in the paper's diverse dictionary.
    """
    A = torch.zeros(N_DICT, space.n_disc, dtype=torch.long)
    for i in range(N_DICT):
        theta = torch.rand(1).item()
        active = torch.rand(space.n_disc) < theta
        for k in range(space.n_disc):
            if active[k] and space.cards[k] > 1:
                A[i, k] = torch.randint(1, space.cards[k], (1,)).item()
    return A


def _embed(idx: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Hamming embedding phi_A(z): normalized Hamming distance to each anchor.

    ``idx`` is ``(n, n_disc)`` ordinal indices, ``A`` is ``(m, n_disc)``; returns
    ``(n, m)`` in ``[0, 1]`` (the fraction of discrete coordinates on which they differ).
    """
    if A.shape[1] == 0:
        return torch.zeros(len(idx), N_DICT, dtype=DTYPE)
    mismatch = idx.unsqueeze(1) != A.unsqueeze(0)  # (n, m, n_disc)
    return mismatch.to(DTYPE).mean(dim=-1)


def _features(space: _Space, idx: torch.Tensor, cont: torch.Tensor, A: torch.Tensor):
    """Concatenate the discrete embedding and the continuous block into GP inputs.

    ``idx`` is ``(n, n_disc)`` and ``cont`` is ``(n, n_cont)`` (rows aligned).
    """
    phi = _embed(idx, A)
    if space.n_cont == 0:
        return phi
    return torch.cat([phi, cont.to(DTYPE)], dim=-1)


def _matern(ard_num_dims: int, offset: int):
    """A Matern-5/2 ARD sub-kernel (the paper's kernel) over ``active_dims``.

    The only regularization is a lengthscale *floor* (>= 0.025): it keeps the
    high-dimensional embedding's marginal likelihood well-conditioned so the L-BFGS fit is
    stable, without the strong large-lengthscale prior that would over-smooth the Hamming
    features and stall the search.
    """
    return MaternKernel(
        nu=2.5,
        ard_num_dims=ard_num_dims,
        active_dims=list(range(offset, offset + ard_num_dims)),
        lengthscale_constraint=GreaterThan(2.5e-2),
    )


def _fit_bodi_gp(features: torch.Tensor, train_Y: torch.Tensor, space: _Space):
    """SingleTaskGP with the HED (x continuous) product Matern-5/2 ARD kernel."""
    m = N_DICT
    k_emb = _matern(m, 0)
    base = k_emb if space.n_cont == 0 else k_emb * _matern(space.n_cont, m)
    model = SingleTaskGP(
        train_X=features.to(DTYPE),
        train_Y=train_Y.to(DTYPE),
        covar_module=ScaleKernel(base),
        input_transform=Normalize(d=features.shape[-1]),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except ModelFittingError:
        # Bounded-Adam safety net if the L-BFGS fit still fails on a degenerate embedding.
        fit_gpytorch_mll(
            mll, optimizer=fit_gpytorch_mll_torch, optimizer_kwargs={"step_limit": 300}
        )
    return model


def optimize_problem(
    problem,
    n_init: int | None = None,
    iters: int = 50,
    seed: int = 0,
    device: str = "auto",
    checkpoint: str | None = None,
) -> Result:
    """BODi over the problem's mixed space (dictionary embedding + GP + LogEI)."""
    set_seed(seed)
    resolve_device(device)  # honored for parity; the embedding GP is tiny and runs on CPU
    obj = ProblemObjective(problem)
    dim = obj.dim
    if n_init is None:
        n_init = default_n_init(dim)
    res = Result(
        "bodi",
        type(problem).__name__,
        seed,
        acquisition_function="LogEI (dictionary embedding)",
    )

    space = _Space(problem)
    grids = space.grids

    if checkpoint and Path(checkpoint).exists():
        train_X, train_Y, start_it, _ = load_checkpoint(checkpoint, res)
        best = train_Y.max().item()
    else:
        train_X = _snap(initial_design(n_init, dim, seed), grids)
        train_Y = obj(train_X)
        best = train_Y.max().item()
        res.start(best)
        start_it = 0

    n_alt = N_ALT if space.n_cont else 1  # no continuous block -> one local search suffices

    def _acq_at(acqf, idx, cont, A):
        feat = _features(space, idx.reshape(1, -1), cont.reshape(1, -1), A)
        return acqf(feat.unsqueeze(1))  # (1,)

    def _neighbours(idx: torch.Tensor) -> torch.Tensor:
        """All Hamming-1 neighbours of ``idx`` (one flipped discrete coordinate each)."""
        rows = []
        for k in range(space.n_disc):
            for level in range(space.cards[k]):
                if level != int(idx[k]):
                    t = idx.clone()
                    t[k] = level
                    rows.append(t)
        return torch.stack(rows) if rows else idx.reshape(1, -1)

    def _opt_cont(acqf, idx, cont, A):
        """Gradient-ascent EI on the continuous block with the discrete part frozen."""
        if space.n_cont == 0:
            return cont
        x = cont.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=LR)
        for _ in range(N_CONT_STEPS):
            opt.zero_grad()
            (-_acq_at(acqf, idx, x, A)).backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(0.0, 1.0)
        return x.detach()

    def _local_search(acqf, idx, cont, A):
        """Steepest-ascent Hamming-1 hill climbing; each sweep is one batched EI call."""
        idx = idx.clone()
        with torch.no_grad():
            cur = _acq_at(acqf, idx, cont, A).item()
            for _ in range(N_LS_SWEEPS):
                cand = _neighbours(idx)  # (N, n_disc)
                cont_rep = cont.reshape(1, -1).expand(len(cand), -1)
                feats = _features(space, cand, cont_rep, A)
                vals = acqf(feats.unsqueeze(1))  # (N,) -- all neighbours at once
                b = int(vals.argmax())
                if vals[b].item() <= cur:
                    break
                idx, cur = cand[b], vals[b].item()
        return idx, cur

    def _optimize_acq(model, A):
        """Alternating restarts: return the best mixed candidate and its acquisition."""
        acqf = LogExpectedImprovement(model=model, best_f=train_Y.max())
        best_cand, best_acq = None, -float("inf")
        for _ in range(N_RESTARTS):
            idx = torch.tensor(
                [torch.randint(0, c, (1,)).item() for c in space.cards],
                dtype=torch.long,
            )
            cont = torch.rand(space.n_cont, dtype=DTYPE)
            for _ in range(n_alt):
                cont = _opt_cont(acqf, idx, cont, A)
                idx, _ = _local_search(acqf, idx, cont, A)
            with torch.no_grad():
                acq = _acq_at(acqf, idx, cont, A).item()
            if acq > best_acq:
                best_cand, best_acq = space.assemble(idx, cont), acq
        return best_cand, best_acq

    for it in range(start_it, iters):
        A = _make_dictionary(space)  # fresh dictionary each iteration (Sec. 3.2)
        idx_train = space.to_indices(train_X)
        cont_train = train_X[:, space.cont_dims] if space.n_cont else train_X[:, :0]
        features = _features(space, idx_train, cont_train, A)
        model = _fit_bodi_gp(features, train_Y, space)

        candidate, acq_value = _optimize_acq(model, A)
        candidate = _snap(candidate, grids)
        cand_idx = space.to_indices(candidate)
        cand_cont = candidate[:, space.cont_dims] if space.n_cont else candidate[:, :0]
        cand_feat = _features(space, cand_idx, cand_cont, A)
        mean, var = gp_stats(model, cand_feat)

        y = obj(candidate)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, y], dim=0)
        best = max(best, y.item())
        res.record(best, mean=mean, variance=var, acq_value=acq_value)
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
