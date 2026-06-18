"""Borehole — water flow rate through a borehole (8-D), continuous or mixed.

The classic 8-dimensional borehole function: the flow rate of water through a
borehole that connects two aquifers. Widely used as a benchmark. The mixed-variable
form (``is_discrete=True``, from the GP+ project, Bostanabad Research Group) turns
the borehole radius ``rw`` and the casing length ``L`` into categorical variables
chosen from a small set of discrete levels.

Variables (order): ``rw`` [0.05, 0.15], ``r`` [100, 50000], ``Tu`` [63070, 115600],
``Hu`` [990, 1110], ``Tl`` [63.1, 116], ``Hl`` [700, 820], ``L`` [1120, 1680],
``Kw`` [9855, 12045]. Objective (maximize): the flow rate. Unconstrained.

Sources:
A. Yousefpour, Z. Zanjani Foumani, M. Shishehbor, C. Mora, R. Bostanabad. GP+: a Python library for kernel-based learning via Gaussian processes. Advances in Engineering Software, 2024. https://github.com/Bostanabad-Research-Group/GP-Plus
Borehole function: https://www.sfu.ca/~ssurjano/borehole.html
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem, DataType

# (lower, upper) per variable, in order rw, r, Tu, Hu, Tl, Hl, L, Kw.
_BOUNDS = [
    (0.05, 0.15),
    (100.0, 50000.0),
    (63070.0, 115600.0),
    (990.0, 1110.0),
    (63.1, 116.0),
    (700.0, 820.0),
    (1120.0, 1680.0),
    (9855.0, 12045.0),
]
# Mixed form: rw (col 0) and L (col 6) become categorical with these discrete levels.
_RW_LEVELS = list(np.linspace(0.05, 0.15, 5))
_L_LEVELS = list(np.linspace(1120.0, 1680.0, 3))


class Borehole(BenchmarkProblem):
    """Maximize borehole water flow rate (8 vars). ``is_discrete`` -> mixed (rw, L)."""

    available_dimensions = 8
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self, is_discrete: bool = False) -> None:
        if is_discrete:
            self.variable_types = ["continuous"] * 8
            self.variable_types[0] = _RW_LEVELS
            self.variable_types[6] = _L_LEVELS
            self.input_type = DataType.MIXED
        else:
            self.variable_types = None
        super().__init__(dim=8, num_objectives=1, num_constraints=0, bounds=_BOUNDS)

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        rw, r, Tu, Hu, Tl, Hl, L, Kw = (x[:, i] for i in range(8))

        ln_r_rw = np.log(r / rw)
        flow = (2 * np.pi * Tu * (Hu - Hl)) / (
            ln_r_rw * (1.0 + 2 * L * Tu / (ln_r_rw * rw**2 * Kw) + Tu / Tl)
        )
        fx = torch.tensor(flow, dtype=torch.float64).reshape(-1, 1)  # maximize flow
        return None, fx
