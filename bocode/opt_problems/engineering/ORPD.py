"""Optimal Reactive Power Dispatch (ORPD) on the IEEE 118-bus system — 77-D mixed.

The classic MINLP volt-VAr control problem of the power-systems literature: choose the
voltage-control assets of a transmission grid so as to minimize the active power lost in
the network, without violating the bus-voltage limits. The grid model is
``pandapower.networks.case118()`` (the MATPOWER / IEEE 118-bus test case), so this
problem needs the optional ``pandapower`` dependency::

    pip install 'bocode[pandapower]'

Control split — the literature-standard IEEE-118 ORPD split, ``54 + 9 + 14 = 77``::

    dims 0..53  : 54 continuous generator terminal-voltage setpoints in [0.94, 1.06] pu
                  (the slack ``net.ext_grid`` first, then the 53 ``net.gen`` units)
    dims 54..62 : 9  integer transformer tap positions in {-8, ..., +8}
                  (the 9 of case118's 13 transformers that carry a tap changer)
    dims 63..76 : 14 integer switched shunt-bank steps in {0, 1}
                  (``net.shunt``; every bank in case118 has ``max_step = 1``)

**Tap model** (a documented deviation from the raw case data): case118 ships fixed tap
ratios with heterogeneous ``tap_step_percent`` (1.5 / 4.0 / 6.5 %) and no
``tap_min``/``tap_max``. We adopt the ORPD-literature convention instead — tap ratio in
``[0.90, 1.10]`` in 1.25 % steps, i.e. ``tap_pos`` in ``{-8, ..., +8}``, 17 discrete
positions per tap changer — by setting ``tap_step_percent = 1.25`` on those 9
transformers. This is the standard "17-step tap" model quoted in IEEE-118 ORPD studies
and gives a uniform, well-scaled integer cardinality.

**Objective** (minimized)::

    P_loss = sum(res_line.pl_mw) + sum(res_trafo.pl_mw)                        [MW]
           + RHO_V * sum_b [ max(0, V_b - V_max)^2 + max(0, V_min - V_b)^2 ]

with ``V_min / V_max = 0.94 / 1.06`` pu on every bus, so the problem is an unconstrained
box in BoCoDe terms. ``evaluate()`` returns the negated penalized loss because BoCoDe
maximizes.

*Penalty calibration*: measured over random controls, losses span ~177-250 MW (base case
133 MW) and the squared-violation sum spans ``0 .. 8.5e-4``, so ``RHO_V = 1e5`` makes the
worst observed violation cost ~85 MW — comparable to the entire loss range, i.e. decisive
but not landscape-flattening. A power flow that does not converge scores
``DIVERGED_LOSS_MW = 1000`` MW (~4x the worst converged loss); the count is exposed as
``problem.n_diverged``.

Each evaluation is one AC power flow: ~18 ms (Newton-Raphson, no numba). ``init="dc"`` is
forced so the result never depends on evaluation order (verified: repeated evaluations
agree to 0.0 MW).

Sources:
L. Thurner, A. Scheidler, F. Schafer, J.-H. Menke, J. Dollichon, F. Meier, S. Meinecke, M. Braun. pandapower — an Open Source Python Tool for Convenient Modeling, Analysis and Optimization of Electric Power Systems. IEEE Transactions on Power Systems 33(6):6510-6521, 2018.
IEEE 118-bus test case, as distributed with MATPOWER / pandapower.networks.case118.
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem

_VM_MIN, _VM_MAX = 0.94, 1.06  # bus-voltage limits [pu]
_TAP_STEP_PERCENT = 1.25  # ORPD-standard uniform tap step
_TAP_RANGE = 8  # tap_pos in {-8, ..., +8} -> 17 positions
_RHO_V = 1e5  # MW per (pu)^2 of squared voltage violation
_DIVERGED_LOSS_MW = 1000.0

# Structure of pandapower's case118, verified at load time (see _net()).
_N_GEN = 53  # net.gen rows (plus 1 ext_grid -> 54 voltage setpoints)
_N_TAP = 9  # transformers with a tap changer
_N_SHUNT = 14  # switched shunt banks, all with max_step = 1


class ORPD118(BenchmarkProblem):
    """IEEE 118-bus optimal reactive power dispatch, 77-D mixed (54 cont + 23 integer).

    Maximize ``-(active power loss + voltage-violation penalty)``. Requires the optional
    ``pandapower`` dependency (``pip install 'bocode[pandapower]'``); the import is lazy,
    so constructing the problem and reading its bounds works without it.
    """

    available_dimensions = 77
    num_objectives = 1
    num_constraints = 0

    tags = {"single_objective", "unconstrained", "77D", "mixed", "power_systems"}

    def __init__(self) -> None:
        n_v = 1 + _N_GEN  # ext_grid + generators
        self.variable_types = ["continuous"] * n_v + ["integer"] * (_N_TAP + _N_SHUNT)
        super().__init__(
            dim=n_v + _N_TAP + _N_SHUNT,
            num_objectives=1,
            num_constraints=0,
            bounds=[(_VM_MIN, _VM_MAX)] * n_v
            + [(-_TAP_RANGE, _TAP_RANGE)] * _N_TAP
            + [(0, 1)] * _N_SHUNT,
        )
        self._cache = None
        self.n_diverged = 0

    # -- grid model (lazy: keeps construction dependency-free) ------------------
    def _net(self):
        if self._cache is None:
            try:
                import pandapower as pp
            except ImportError as exc:  # pragma: no cover - exercised without the extra
                raise ImportError(
                    "The ORPD118 problem requires the optional 'pandapower' dependency. "
                    "Install it with: pip install 'bocode[pandapower]'"
                ) from exc
            net = pp.networks.case118()
            tap_rows = net.trafo.index[net.trafo.tap_pos.notna()]
            if (len(net.gen), len(tap_rows), len(net.shunt)) != (
                _N_GEN,
                _N_TAP,
                _N_SHUNT,
            ):
                raise RuntimeError(
                    "pandapower's case118 has an unexpected structure "
                    f"({len(net.gen)} gens / {len(tap_rows)} tap changers / "
                    f"{len(net.shunt)} shunts; expected {_N_GEN}/{_N_TAP}/{_N_SHUNT}). "
                    "The ORPD118 search space is defined against the classic case; "
                    "please report this with your pandapower version."
                )
            # Adopt the ORPD-standard uniform 1.25 % tap step (see module docstring).
            net.trafo.loc[tap_rows, "tap_step_percent"] = _TAP_STEP_PERCENT
            net.trafo.loc[tap_rows, "tap_min"] = -_TAP_RANGE
            net.trafo.loc[tap_rows, "tap_max"] = _TAP_RANGE
            max_step = net.shunt["max_step"].to_numpy().astype(int)
            self._cache = (pp, net, tap_rows, max_step)
        return self._cache

    def _one(self, row: np.ndarray) -> float:
        """One design (natural units) -> penalized loss in MW (to minimize)."""
        pp, net, tap_rows, max_step = self._net()
        n_v = 1 + _N_GEN
        vm = np.clip(row[:n_v], _VM_MIN, _VM_MAX)
        taps = np.clip(np.rint(row[n_v : n_v + _N_TAP]), -_TAP_RANGE, _TAP_RANGE)
        steps = np.clip(np.rint(row[n_v + _N_TAP :]), 0, max_step)
        net.ext_grid.loc[:, "vm_pu"] = float(vm[0])
        net.gen.loc[:, "vm_pu"] = vm[1:]
        net.trafo.loc[tap_rows, "tap_pos"] = taps
        net.shunt.loc[:, "step"] = steps
        try:
            pp.runpp(net, numba=False, init="dc")
        except Exception as exc:  # noqa: BLE001 - any solver failure = divergence
            self.n_diverged += 1
            print(
                f"[ORPD118] power flow did not converge (#{self.n_diverged}): "
                f"{type(exc).__name__}; penalizing with {_DIVERGED_LOSS_MW} MW"
            )
            return _DIVERGED_LOSS_MW
        loss = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
        v = net.res_bus.vm_pu.to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        viol = float(
            np.sum(
                np.maximum(0.0, v - _VM_MAX) ** 2 + np.maximum(0.0, _VM_MIN - v) ** 2
            )
        )
        return loss + _RHO_V * viol

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        X = self.enforce_variable_types(X.to(torch.float64))
        x = X.detach().cpu().numpy().astype(np.float64)
        out = np.array([-self._one(row) for row in x], dtype=np.float64)  # maximize
        return None, torch.tensor(out, dtype=torch.float64).reshape(-1, 1)
