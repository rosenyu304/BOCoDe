"""EngiBench Power Electronics — DC-DC converter parameter optimization (2 objectives).

Wraps the EngiBench ``power_electronics`` problem: a fixed-topology DC-DC converter
(5 switches, 4 diodes, 3 inductors, 6 capacitors) simulated with ngspice. Twenty
continuous design parameters — 6 capacitances, 3 inductances, 1 duty cycle, and 10
switch-signal parameters — drive two competing objectives: minimize ``DcGain`` and
maximize ``Voltage_Ripple``. It is a black-box circuit simulation with no usable
gradient, which makes it a genuine Bayesian-optimization benchmark (unlike EngiBench's
high-dimensional topology problems; see ``Research_Plan.md`` "EngiBench problem
selection" for why those are excluded).

Requires the optional ``engibench`` extra AND a native ``ngspice`` (engibench needs
ngspice **v42-v45**; conda-forge's 41 is too old, so install a recent build)::

    pip install 'bocode[engibench]'                 # engibench + networkx
    # then install ngspice 42-45 (e.g. from ngspice.sourceforge.net or a recent apt/brew)

EngiBench is GPLv3; BoCoDe does not vendor it — this is a thin wrapper over its public
``Problem.simulate`` API, so BoCoDe itself stays permissively licensed.

Sources:
EngiBench: A Framework for Engineering Design Optimization Benchmarks. arXiv:2509.17677, 2025. https://github.com/IDEALLab/EngiBench
"""

from __future__ import annotations

import tempfile

import numpy as np
import torch

from ...base import BenchmarkProblem

# EngiBench power_electronics v0 design space: 6 caps, 3 inductors, duty, 10 switches.
_LOW = [1e-6] * 6 + [1e-6] * 3 + [0.1] + [0.0] * 10
_HIGH = [2e-5] * 6 + [1e-3] * 3 + [0.9] + [1.0] * 10


class PowerElectronics(BenchmarkProblem):
    """Minimize DcGain and maximize Voltage_Ripple of a DC-DC converter (20-D, 2 obj)."""

    available_dimensions = 20
    num_objectives = 2
    num_constraints = 0

    def __init__(self) -> None:
        try:
            from engibench.problems.power_electronics.v0 import (
                PowerElectronics as _EngiPowerElectronics,
            )
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "PowerElectronics requires the optional 'engibench' extra and a native "
                "ngspice (v42-v45). Install with: pip install 'bocode[engibench]' and a "
                "recent ngspice (conda-forge's v41 is too old for engibench)."
            ) from exc

        self._prob = _EngiPowerElectronics(
            target_dir=tempfile.mkdtemp(prefix="bocode_pe_")
        )
        super().__init__(
            dim=20,
            num_objectives=2,
            num_constraints=0,
            bounds=list(zip(_LOW, _HIGH, strict=True)),
        )

    def _evaluate_implementation(self, X, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        x = X.detach().cpu().numpy().astype(float)
        out = np.empty((x.shape[0], 2))
        for i, row in enumerate(x):
            dc_gain, v_ripple = self._prob.simulate(row)
            # BoCoDe maximizes: negate the minimize objective, keep the maximize one.
            out[i] = (-float(dc_gain), float(v_ripple))
        return None, torch.tensor(out, dtype=torch.float64)
