"""Tests for the TFM algorithms.

The TabPFN-dependent runs are skipped unless the fork is importable (see
docs/tfm_setup.md); the rank-selection logic (the GIT-BO/Marzouk contribution) is
pure NumPy and is tested directly.
"""

import importlib.util

import numpy as np
import pytest

from algorithms._tfm_utils import (
    gradient_information_matrix,
    sample_in_subspace,
    select_rank,
)

HAS_TABPFN = importlib.util.find_spec("tabpfn") is not None


def test_fixed_rank_capped_at_dim():
    eig = np.array([3.0, 2.0, 1.0])
    assert select_rank(eig, mode="fixed", rank=5) == 3  # capped at dim
    assert select_rank(eig, mode="fixed", rank=2) == 2
    assert select_rank(eig, mode="fixed", rank=0) == 1  # at least 1


def test_marzouk_certified_rank():
    # one dominant direction: tail after r=1 is tiny -> r* = 1
    eig = np.array([100.0, 0.5, 0.3, 0.1])
    assert select_rank(eig, mode="marzouk", eps=0.05, kappa=1.0) == 1
    # flat spectrum: need almost all directions to drive the tail below 2*eps
    flat = np.ones(10)
    r = select_rank(flat, mode="marzouk", eps=0.05, kappa=1.0)
    assert r >= 9  # tail of 10 equal eigenvalues only drops below 0.1 near the end


def test_marzouk_tighter_eps_keeps_more_rank():
    eig = np.array([10.0, 3.0, 1.0, 0.5, 0.2])
    loose = select_rank(eig, mode="marzouk", eps=0.10)
    tight = select_rank(eig, mode="marzouk", eps=0.01)
    assert tight >= loose


def test_gradient_information_matrix_shape_and_psd():
    grads = np.random.RandomState(0).randn(50, 4)
    H = gradient_information_matrix(grads)
    assert H.shape == (4, 4)
    assert np.allclose(H, H.T)
    assert np.linalg.eigvalsh(H).min() >= -1e-9  # PSD


def test_sample_in_subspace_in_unit_cube():
    rng = np.random.default_rng(0)
    center = np.full(6, 0.5)
    grads = rng.standard_normal((40, 6))
    samples, r = sample_in_subspace(
        center, grads, n_samples=32, rank_mode="marzouk", rank=5, eps=0.05, scale=0.3, rng=rng
    )
    assert samples.shape == (32, 6)
    assert (samples >= 0.0).all() and (samples <= 1.0).all()
    assert 1 <= r <= 6


@pytest.mark.slow
@pytest.mark.skipif(not HAS_TABPFN, reason="TabPFN fork not installed (see docs/tfm_setup.md)")
@pytest.mark.parametrize("rank", ["5", "marzouk"])
def test_git_bo_runs(rank):
    import bocode
    from algorithms.single_obj import git_bo

    res = git_bo.optimize_problem(bocode.get_problem("Branin")(), n_init=6, iters=2, seed=42, rank=rank)
    assert len(res.per_iteration_value) == 3


@pytest.mark.slow
@pytest.mark.skipif(not HAS_TABPFN, reason="TabPFN fork not installed (see docs/tfm_setup.md)")
def test_pfn_cei_runs():
    import bocode
    from algorithms.single_obj_constrained import pfn_cei

    res = pfn_cei.optimize_problem(bocode.PressureVessel(), n_init=6, iters=2, seed=42)
    assert len(res.per_iteration_value) == 3
