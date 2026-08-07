"""Thinned + phased linear antenna array — 200-D mixed (100 binary + 100 continuous).

A classic electromagnetics design benchmark: switch the elements of a uniformly-spaced
linear array on or off ("thinning": fewer active elements means less cost, weight and
feed-network complexity) and set each element's excitation *phase*, so as to push the
peak sidelobe level (PSLL) of the radiation pattern as far below the main beam as
possible. Haupt (1994) thins a 200-element (2 x 100 symmetric) half-wavelength-spaced
linear array with a genetic algorithm and reports the maximum sidelobe level in dB; we
use his geometry (``N = 100`` elements, ``d = lambda/2``, broadside steering, uniform
amplitude) and add the per-element phase as the continuous half of the mixed space,
which is the standard "thinned + phased" extension.

Physics (pure numpy — no optional dependency, microseconds per evaluation)::

    u  = cos(theta) in [-1, 1];   k*d = pi   for d = lambda/2
    AF(u) = sum_n a_n * exp(1j * (n * pi * u + phi_n)),     a_n in {0, 1}
    PSLL(dB) = 20 * log10( max_{u in sidelobe region} |AF| / max_u |AF| )

The main lobe is located at the *global* pattern peak and delimited by walking outwards
from that peak to the first local minimum (the first null) on each side; everything else
is the sidelobe region. That definition is steering-agnostic, so it stays correct when
the phases tilt the beam away from broadside.

Design variables (200-D)::

    dims 0..99    : a_n in {0, 1}       element on/off        (binary categorical)
    dims 100..199 : phi_n in [0, 2*pi]  excitation phase      (continuous)

``evaluate()`` returns ``-PSLL(dB)`` because BoCoDe maximizes, so larger is better.
Degenerate designs with fewer than 4 active elements (no meaningful pattern) get the
worst possible score, ``0 dB`` (sidelobes as strong as the main beam).

Reference points (measured with this implementation): the all-on / all-zero-phase
uniform broadside array scores ``-PSLL = +13.27 dB``, i.e. the textbook ``-13.2 dB``
uniform-illumination sidelobe level; 200 scrambled-Sobol designs score ``0.00 .. 3.78``
dB (random phases destroy the pattern coherence), so the landscape has a large,
structured gap to close.

Sources:
R. L. Haupt. Thinned arrays using genetic algorithms. IEEE Transactions on Antennas and Propagation 42(7):993-999, 1994.
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem

_N_ELEM = 100  # elements in the (half-)array
_N_U = 2001  # u-grid resolution (odd, so u = 0 lies on the grid)
_MIN_ON = 4  # fewer active elements -> no meaningful pattern
_BAD_DB = 0.0  # PSLL(dB) assigned to degenerate designs (== main-beam level)


class AntennaArray200(BenchmarkProblem):
    """Thinned + phased 100-element linear array (Haupt 1994): 100 binary + 100 phases.

    Maximize ``-PSLL(dB)``. Pure numpy, deterministic, no optional dependency.
    """

    available_dimensions = 200
    num_objectives = 1
    num_constraints = 0

    tags = {"single_objective", "unconstrained", "200D", "mixed", "high_dimensional"}

    def __init__(self, n_elem: int = _N_ELEM, n_u: int = _N_U) -> None:
        self._n = int(n_elem)
        self._n_u = int(n_u)
        self._steer_cache = None
        self.variable_types = [[0.0, 1.0]] * self._n + ["continuous"] * self._n
        super().__init__(
            dim=2 * self._n,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0.0, 1.0)] * self._n + [(0.0, 2.0 * np.pi)] * self._n,
        )

    @property
    def _steer(self) -> np.ndarray:
        """``exp(1j * n * pi * u)`` on the u-grid, shape ``(n_u, n_elem)`` (cached)."""
        if self._steer_cache is None:
            u = np.linspace(-1.0, 1.0, self._n_u)
            self._steer_cache = np.exp(1j * np.pi * np.outer(u, np.arange(self._n)))
        return self._steer_cache

    @staticmethod
    def _psll_db(mag: np.ndarray) -> float:
        """Peak sidelobe level in dB of one ``|AF|`` curve over the u-grid."""
        peak_i = int(np.argmax(mag))
        peak = mag[peak_i]
        if peak <= 0:
            return _BAD_DB
        # Walk out from the peak to the first local minimum on each side.
        lo = peak_i
        while lo > 0 and mag[lo - 1] <= mag[lo]:
            lo -= 1
        hi = peak_i
        while hi < mag.size - 1 and mag[hi + 1] <= mag[hi]:
            hi += 1
        side = np.concatenate([mag[:lo], mag[hi + 1 :]])
        if side.size == 0:  # the main lobe fills the whole visible region
            return _BAD_DB
        smax = float(side.max())
        if smax <= 0:
            return -300.0  # a perfect null everywhere else
        return float(20.0 * np.log10(smax / peak))

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        X = self.enforce_variable_types(X.to(torch.float64))
        x = X.detach().cpu().numpy().astype(np.float64)
        n = self._n
        a = (x[:, :n] > 0.5).astype(np.float64)
        phi = np.clip(x[:, n:], 0.0, 2.0 * np.pi)
        exc = a * np.exp(1j * phi)  # (batch, n_elem)

        out = np.empty(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            if a[i].sum() < _MIN_ON:  # degenerate: no meaningful pattern
                out[i] = -_BAD_DB
                continue
            out[i] = -self._psll_db(np.abs(self._steer @ exc[i]))

        fx = torch.tensor(out, dtype=torch.float64).reshape(-1, 1)  # maximize -PSLL
        return None, fx
