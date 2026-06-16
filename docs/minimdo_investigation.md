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

## Black-box-suitability survey of all `applications/`

A problem is "black-box-portable" if it has a self-contained `.py` formulation that
maps design variables to outputs (so it can become a BoCoDe `evaluate(x) -> y`).

| Application | Form | Black-box-portable? | Notes |
|---|---|---|---|
| `satellite/sat_initial.py` | `.py` | ✅ **DONE** | implemented as `SatelliteDesign` |
| `pearl/pearl_initial_formulation.py` | `.py` (150 lines) | ✅ **yes — next candidate** | autonomous marine/ocean sensing platform: coupled geometry / hydrodynamics / mass / propulsion / communications / power; ~6 continuous geometry variables (three diameters Df/Ds/Dd, three thicknesses tf/ts/td) plus operational vars (speed `v`, data rate `R`); natural objective = minimize total mass with power/buoyancy constraints. Ports the same way as satellite. |
| `bliss/bliss_gen4.ipynb` | notebook only | ⚠️ needs extraction | **BLISS / SSBJ** (supersonic business jet) — a *recognized* MDO benchmark (Sobieszczanski-Sobieski et al.), so high value, but the equations live in a Jupyter notebook and would need extraction + the BLISS coupling solve. |
| `thesis_coffee/app_coffee_v*.ipynb` | notebook only | ✖ | thesis demonstration, not a standard design benchmark |
| `synthetic/`, `random_presolve/` | notebooks / CSVs | ✖ | synthetic sparsity studies and MDO-reconfiguration method experiments, not design problems |

**Recommended order:** (1) `SatelliteDesign` — done; (2) **PEARL** marine platform —
clean `.py`, ports like satellite; (3) **BLISS/SSBJ** — recognized aircraft MDO
benchmark, worth the notebook extraction if an aircraft problem is wanted.

## Full inventory (notebooks × Norheim PhD thesis, 2022)

Cross-referencing every `applications/*.ipynb` with Norheim's thesis (Ch. 7 case
studies). **Already in BoCoDe:** `SatelliteDesign` (= thesis §7.1 spacecraft) and
`SpeedReducer` (= Golinski, `speedreducer/`) — do not re-port.

**Tier 1 — strong new real-engineering black boxes (each needs an inner MDA solve):**
- **PEARL marine platform** (`pearl/`, thesis §7.4): offshore autonomous surface
  platform; 6 continuous geometry vars (Df, Ds, Dd, tf, ts, td ∈ [0.1, 10] m);
  minimize platform mass (thesis optimum ≈585 kg); 4 inequality constraints; inner
  Newton solve over feedback vars. **The flagship next port.**
- **Sobieski SSBJ / BLISS** (`bliss/`): supersonic business jet, 10 vars
  (aero/struct/prop), maximize range, ~8–10 stress/pressure/temp/ESF constraints;
  Gauss–Seidel MDA per eval. (Bounds not in the notebook — use the canonical
  Sobieski bounds.)
- **miniaero GP wing** (`miniaero/cvxaircraft.ipynb`): cleanest — convex geometric
  program, 3 vars (aspect ratio, wing area, speed), minimize drag.

**Tier 2 — academic MDO benchmarks (low-dim, fully posed):** Sellar (`sellar_opt/`,
3+2 vars), Allison (`thesis_allison/`, 3-var coupled quadratic, 3 equality
couplings), HVAC air-quality (`dprohvac/`, 2 vars min infection probability),
satmini and miniaerostruct (small synthetic coupled toys).

**Tier 3 — high-altitude balloon** (`balloon/`, thesis §7.2–7.3): portable but the
posed 1-D version's optimum sits on the lower bound (thesis calls it trivial);
worth it only with extra design variables exposed.

**Skip:** rocket staging, cold-gas thruster, the "coffee"/space-telescope model
(coupled analyses with no objective/constraints declared); pump and beam
(incomplete stubs); the `synthetic/` and `random_presolve/` random-polynomial
systems (regenerated each run — not fixed benchmarks).

**Porter note:** unlike `SatelliteDesign` (closed-form mass coupling), the Tier-1/2
problems are genuinely coupled — each `x → (objective, constraints)` must run an
inner fixed-point/Newton MDA to convergence (wrap `scipy.optimize.fsolve` around the
discipline residuals and guard against non-convergence at extreme designs).

## Status (ported + verified against reference optima)

Five minimdo-derived problems are now in `bocode/opt_problems/engineering/`, each
verified to reproduce its published optimum:

| Problem | Vars / constraints | Inner solve | Reference optimum | Verified |
|---|---|---|---|---|
| `SatelliteDesign` | 4 / 3 | closed-form mass coupling | — | ✅ (earlier) |
| `PEARL` | 7 / 6 | 2-D MDA fixed point (`A_solar`, `t_d`) via `fsolve` | `m_tot` 585.3 kg (thesis Tbl 7.14) | ✅ 585.39 kg at reported design |
| `Sellar` | 3 / 2 | 2-discipline fixed point | f* 3.1834 | ✅ 3.1834 at [1.978,0,0] |
| `Allison` | 3 / 0 | linear 3×3 coupling solve | f* 0.5698 | ✅ 0.5698 at [-0.507,0.047,0.179] |
| `MiniAeroWing` | 3 / 0 | weight fixed point (closed form) | D* 242.3 N | ✅ 242.27 N at [18.2,5.3,49.2] |

**Deferred / skipped:**
- **Sobieski SSBJ / BLISS** — the `bliss_gen4.ipynb` formulation is incomplete
  (polynomial-surrogate coefficient tables not fully in the notebook, no design-
  variable bounds). Port from the *canonical* Sobieski SSBJ definition instead of
  this notebook if/when an aircraft MDO problem is wanted — flagged, not done.
- **HVAC** (`dprohvac`) — checked numerically: the objective is monotone in both
  variables, so the optimum sits on the corner (η=1, Q=max). Trivial for
  optimization benchmarking (same reason the 1-D balloon was skipped) — skipped.
- Tier-2 toys (satmini, miniaerostruct) and Tier-3 balloon remain skipped.

Sellar and Allison are classic *coupled-MDA* test problems (synthetic couplings,
standard in the MDO literature) rather than physical designs; they live in
`engineering/` as MDO benchmarks and are valuable for testing MDA-aware BO.
