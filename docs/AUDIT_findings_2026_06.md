# Problem-suite audit (dev/2026_06)

Cross-check of the library against the two source papers
(`Guidelines_Real_World_Constrained.pdf` = CEC2020 RW-Constrained suite,
`An Easy-to-use Real-world Multi-objective.pdf` = Tanabe & Ishibuchi RE/CRE
suite). **Coverage is complete**: all 57 CEC2020-RW (RC01–RC57), all 16 RE, and
all 8 CRE problems are implemented and registered. The Tanabe paper's Table 4
simulation problems (radar waveform, MarioGAN, etc.) are intentionally not part
of the RE suite and are correctly absent.

The audit surfaced correctness issues. Status below (FIXED items resolved in the
2026-06 constraint-semantics cleanup; see commit history).

## Bugs
1. **[FIXED] `CEC2020_p3` runtime bug** (`cec2020_rw/CEC2020_p1_20.py`):
   `h = (n_samples, 0)` assigned a tuple, so the constraint return became a
   shape-(2,) array. Now `h = np.zeros((n_samples, 0))`.
2. **[FIXED] `D = n_samples` bug in p44–p50** (`CEC2020_p40_57.py`): `D` was set to
   the batch size instead of the problem dimension, which silently corrupted both
   the objective (summed over the wrong number of dimensions) and the constraint
   count (e.g. p45 produced 2 of 24 spacing constraints). Now `D = X.shape[1]`.
   After the fix p45–p50 match the paper exactly (25/25/25/30/30/30 constraints).
3. **[FIXED] `num_constraints` counts only equality constraints across CEC2020**:
   every problem now declares the total constraint count (equality + inequality),
   and `base.evaluate()` concatenates the two blocks into one `(n, num_constraints)`
   constraint tensor, consistent with every other problem. Verified: all 57
   problems have `num_constraints == evaluate() constraint width`, and the widths
   match the paper's `g + h` except p31 and p44 (below).
4. **[FIXED] Return-shape inconsistencies**: handled centrally in
   `base.evaluate()` / `_normalize_constraints` — `None` → `(n, 0)`, a stray
   trailing singleton (`(n, k, 1)` from p51–p54's `unsqueeze`) is squeezed, and a
   1-D block becomes `(n, 1)`. p34–p39's `None` equality blocks are normalized.
5. **[FIXED] `CEC2020_p4`**: `available_dimensions` corrected from 4 to 6 (paper D=6).
6. **[OPEN] `CEC2020_p33` MATLAB→Python port**: `n1 = (nely+1)*(elx-1) + ely` keeps
   MATLAB 1-based indexing with 0-based `elx`, producing negative indices into `U`.
   The problem evaluates (Python negative indexing wraps) but may not match the
   reference; needs validation against the original MATLAB. Constraint count (30)
   matches the paper.
7. **[OPEN / documented] `CEC2020_p31`** (gear train) implemented as unconstrained
   (paper: g=1, h=1). The classic gear-train problem is unconstrained; resolving
   needs the original MATLAB source. Left at 0 rather than fabricated.
8. **[OPEN / documented] `CEC2020_p44`** produces the complete pairwise spacing
   constraint set `C(15,2)=105`; paper table says 91. 105 is the mathematically
   complete set.

## Decisions to confirm with maintainer
- Constraint-count deltas vs the guideline table where the library matches the
  official MATLAB suite source instead: p17 (4 vs 3), p21 (8 vs 7). Likely the
  paper table undercounts; library is probably correct.
- p26 (87 vs 86 g) and p44 (105 = C(15,2) vs 91 g): recheck against official source.
- p25 dimension: library dim=7 vs paper D=4 (hydrostatic thrust bearing is
  conventionally 4-var).
- Sign convention: CEC2020 classes return `-f` (maximization), RE/CRE return raw
  `fx` (minimization). Downstream algorithms must be told which, or one suite
  normalized.
- RE/CRE classes carry no reference Pareto-front / paper-name metadata (e.g.
  "RE2-4-1"); could be added to the metadata JSON.
