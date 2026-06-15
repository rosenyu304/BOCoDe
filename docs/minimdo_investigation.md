# minimdo investigation — what to add to BoCoDe

Investigated `https://github.com/norheim/minimdo` (cloned to
`Bocode_dev/minimdo`). Summary and recommendation per Research_Plan ToDo #3.

## What minimdo is

minimdo is **not** a benchmark suite — it is a *framework* for formulating
conceptual-design equations symbolically and reconfiguring/solving the resulting
system of equations under different MDO (multidisciplinary design optimization)
formulations (MDF, IDF, etc.), in the spirit of cvxpy/pyomo/gekko. So we cannot
"import a problem"; we'd port the underlying design equations into a standalone
BoCoDe problem (`evaluate(x) -> (objective, constraints)`), dropping minimdo's
symbolic/MDA machinery.

The reusable engineering content lives in `applications/`:

| Application | What it is | Fit for BoCoDe |
|---|---|---|
| `satellite/` (`sat_initial.py`, `sat_opt.ipynb`) | SMAD-based conceptual **satellite design** (coupled orbit / power / payload / comms / structure / propulsion) | **Strong** — real, citable, continuous, distinct from current problems |
| `pearl/` (`pearl_initial_formulation.py`) | **Wave-energy-converter buoy** design (geometry + hydrodynamics + structures) | Good secondary candidate |
| `bliss/` | BLISS supersonic business-jet MDO demo | Possible, but heavier to port |
| `thesis_coffee/`, `synthetic/`, `random_presolve/` | thesis/demo and synthetic sparsity studies | Not benchmark-worthy |

## Recommendation: add the satellite conceptual-design problem

**`SatelliteDesign`** — minimize total satellite mass for a small earth-observation
satellite, subject to mission constraints. It is a multidisciplinary-feasible (MDF)
problem: total mass `m_t` is a coupling variable solved by a fixed-point/MDA loop
(`m_s = η_S · m_t`), so the wrapper runs that inner solve and returns the converged
mass — a natural "simulator" for BoCoDe.

- **Design variables (continuous):** orbit altitude `h` (km), solar-array area `A`
  (m²), payload ground resolution `X_r` (m), propellant mass `m_pr` (kg). (4-D;
  could be extended with antenna diameter etc.)
- **Objective (minimize → BoCoDe maximizes `-m_t`):** total mass
  `m_t = m_T + m_p + m_b + m_A + m_s + m_pr`.
- **Constraints:** mission lifetime `L_t ≥ L_min` (e.g. 10 yr), link-budget margin
  `E_b/N_0 ≥ threshold`, transmit-power balance `P_T ≥ 0`, data-rate target.
- **Source:** Wertz & Larson, *Space Mission Analysis and Design* (SMAD), 3rd ed.;
  formulation as used in the minimdo satellite application.

Why it's worth adding: BoCoDe currently has no **aerospace conceptual-design /
MDO** problem; this adds a real, coupled, constrained design problem with a
genuine inner fixed-point solve, complementing the structural/mechanical problems.

**Porting effort:** moderate — re-express the ~25 design equations in
`sat_initial.py` as plain numpy, implement the `m_t` fixed-point solve (a few
Gauss–Seidel iterations or `scipy.optimize.fsolve`), define the bounds, and return
`(-m_t, constraints)`. No dependency on minimdo or sympy at runtime.

**Secondary:** `PEARL` wave-energy buoy (6 continuous geometry variables — three
diameters, three thicknesses — with buoyancy/hydrodynamic constraints) is a clean
second addition if a marine-energy problem is desired.

## Proposed next step

Implement `bocode/opt_problems/engineering/SatelliteDesign.py` (and optionally
`PEARL.py`) as standalone numpy problems with the SMAD citation, pending your
go-ahead. They would be continuous-variable, constrained, single-objective
(mass), with an inner MDA solve.
