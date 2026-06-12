# Problem-suite audit (dev/2026_06)

Cross-check of the library against the two source papers
(`Guidelines_Real_World_Constrained.pdf` = CEC2020 RW-Constrained suite,
`An Easy-to-use Real-world Multi-objective.pdf` = Tanabe & Ishibuchi RE/CRE
suite). **Coverage is complete**: all 57 CEC2020-RW (RC01–RC57), all 16 RE, and
all 8 CRE problems are implemented and registered. The Tanabe paper's Table 4
simulation problems (radar waveform, MarioGAN, etc.) are intentionally not part
of the RE suite and are correctly absent.

The audit also surfaced correctness issues to fix in a follow-up (kept separate
from the foundation refactor so the restructure stays reviewable):

## Bugs (should fix)
1. **`CEC2020_p3` runtime bug** (`cec2020_rw/CEC2020_p1_20.py`): `h = (n_samples, 0)`
   assigns a tuple, so the constraint return becomes a shape-(2,) array instead
   of an `(n, 0)` empty matrix. Will raise/return garbage when evaluated.
2. **`CEC2020_p31` (gear train) constraints missing**: paper specifies g=1, h=1;
   implementation returns neither and does not round the integer teeth counts.
3. **`num_constraints` counts only equality constraints across CEC2020**: every
   inequality-only problem (p3, p8, p10, p12–p21, p24–p33, p44, …) declares
   `num_constraints=0`, so `is_constrained` is wrong and metadata understates the
   constraint count. Paper counts are g+h. **This propagates into the generated
   metadata JSON** — the constraint counts for those problems reflect the code's
   current (understated) attributes, not the paper. Flagged for the constraint-
   semantics cleanup.
4. **Return-shape inconsistencies**: p51–p54 and p33 `unsqueeze(-1)` an already-2D
   constraint tensor → `(n, k, 1)`; p34–p39 return `None` for g instead of `(n, 0)`.
5. **`CEC2020_p4`**: class attribute `available_dimensions = 4` contradicts the
   actual `dim = 6` (paper D=6).
6. **`CEC2020_p33` MATLAB→Python port**: `n1 = (nely+1)*(elx-1) + ely` keeps
   MATLAB 1-based indexing with 0-based `elx`, producing negative indices into `U`.

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
