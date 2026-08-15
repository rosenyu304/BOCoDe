"""Tests for the clean-room weighted-Lasso HPO problems.

Construction is offline (no data fetch); evaluation downloads a dataset from
OpenML, so the eval test is marked ``slow`` and skipped if there is no network.
"""

import socket

import pytest
import torch

import bocode

LASSO = {
    "LassoDiabetes": 8,
    "LassoBreastCancer": 9,
    "LassoDNA": 180,
    "LassoLeukemia": 7129,
    "LassoRCV1": 47236,
}


@pytest.mark.parametrize("name,dim", list(LASSO.items()))
def test_lasso_constructs(name, dim):
    """Problems construct without fetching data (lazy)."""
    p = bocode.get_problem(name)()
    assert p.dim == dim
    assert p.num_objectives == 1
    assert p.num_constraints == 0
    assert len(p.bounds) == dim


def _has_network(host="api.openml.org", port=443, timeout=3.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


@pytest.mark.slow
def test_lasso_diabetes_evaluates():
    """The smallest Lasso problem returns a finite validation-MSE objective."""
    if not _has_network():
        pytest.skip("no network: cannot fetch the OpenML dataset")
    p = bocode.LassoDiabetes()
    values, constraints = p.evaluate(torch.rand(3, p.dim) * 2 - 1)
    assert values.shape == (3, 1)
    assert torch.isfinite(values).all()
    assert values.max() <= 0  # objective is the negated MSE
