# Variable-type audit (mixed-variable formulations)

Source-by-source check of whether each problem's *original* formulation is
continuous, integer, categorical, or mixed, vs. how BoCoDe currently treats it.
Drives the per-dimension `variable_types` metadata and the optional
`BenchmarkProblem.enforce_variable_types()` enforcement layer (rounds integers /
snaps categoricals). Indices below are 0-based.

## Design decision

`variable_types` records the **original** formulation so the metadata and
CATEGORIZATION are transparent ("democracy with guardrails": the user is told
which variables are meant to be integer/categorical). Enforcement is **opt-in**:
`enforce_variable_types(X)` snaps a continuously-sampled `X` to the declared types;
`evaluate()` itself does not force it (so existing continuous usage is unchanged).
Some problems already round internally (noted below) — for those `variable_types`
is documentary and `enforce_variable_types` is an idempotent no-op.

## MODAct (CS/CT/CTS/CTSE/CTSEI 1–4) — CONTINUOUS (do NOT round)

The `[0, 4.999999]` / `[9, 40.999999]` bounds are **not** a rounding request. The
modact library uses `math.modf()` to split such a variable into an integer part (a
discrete selector — motor catalog index, pinion/gear teeth) AND a fractional part
(a continuous profile-shift / fill-factor). Both halves carry meaning, so rounding
would destroy information. MODAct is faithfully continuous; the discrete structure
is *within* a dimension and cannot be expressed by per-dimension snapping. (Paper
paywalled; verified against the authoritative modact source.)

## CEC2020 RW-Constrained (code `pN` = paper `RC0N`)

Already rounded inside `_evaluate_implementation` (documentary only):
p8 idx1, p9 idx2, p10 idx2 (binary); p11 idx4,5; p12 idx3-6 (binary); p13 idx3,4
(integer); p14 idx0-2 (integer); p18 idx0,1 (0.0625·int); p22 idx0-5 (int teeth),
idx6 ∈{3,4,5}, idx7,8 ∈{1.75..3.0}; p26 idx0-7 int + idx8-21 discrete; p28 idx2
(int balls); p30 idx0 (int coils), idx2 (43-value wire gauge).

Original integer but code does NOT round → candidates for enforcement:
**p21** (all 5 integer), **p31** (4 teeth ∈[12,60] integer).

Likely code bug (rounds although the paper is continuous) — flagged, not fixed
here: **p19** (welded beam) and **p20** (three-bar truss) apply `0.0625·round`.

## RE / CRE (Tanabe & Ishibuchi 2020) — already rounded in code (documentary)

RE22 idx0 (76-value rebar table) → mixed; RE23 idx0,1 (0.0625·int) → mixed;
RE25 idx0 (int) + idx2 (42-value wire table) → mixed; RE35 idx2 (int teeth) →
mixed; RE36 idx0-3 (int) → integer; CRE24 (=RE35) idx2 → mixed; CRE25 (=RE36)
idx0-3 → integer. All other RE/CRE continuous.

## Engineering standalone — original mixed, code does NOT enforce

These get `variable_types` set so users can opt into `enforce_variable_types`:
- **PressureVessel** (4): idx0,1 → nearest 0.0625 gauge; idx2,3 continuous.
- **SpeedReducer** (7): idx2 → integer teeth [17,28].
- **GearTrain** (4): idx0-3 → integer teeth [12,60].
- **ReinforcedConcreteBeam** (3): idx0 (rebar-area table), idx2 (discrete width);
  idx1 continuous.
- **Car** (11, Gu/Deb): idx7,8 → {0.192, 0.345} (two-level material).
- **EulerBernoulliBeamBending** (3): idx2 → 12-value cross-section catalog
  (catalog present but commented out in code).
- **Mazda / Mazda_SCA** (222 / 148): the single largest omission — every variable
  has a sheet-metal-gauge catalog (column G of the xlsx); the 2nd objective
  (number of common gauge parts) only makes sense under gauge discretization.
  Continuous treatment is a defensible standard usage but under-specifies the
  problem; the full per-variable catalogs are recorded in the design xlsx.

Fully continuous (correct as-is): CompressionSpring (relaxed Gandomi 3-var),
WeldedBeam, CantileverBeam, HeatExchanger, CarSideImpact, BotorchCarSideImpact,
DiscBrake, VehicleSafety, Penicillin, WaterProblem, WaterResources, ThreeTruss,
TwoBarTruss, QPowerModel, ReactivityModel, RobotPush, Rover, MOPTA08Car.

## Materials & TSP — joint/permutation (not per-dimension)

- **Materials** (AgNP, AutoAM, CrossedBarrel, P3HT, Perovskite): joint discrete —
  `evaluate()` snaps the whole vector to the nearest measured candidate row. Already
  `DataType.DISCRETE`; per-dimension `variable_types` cannot express this, and the
  candidate-set lookup is the correct semantics (already implemented).
- **TSP_51Cities / TSP_100Cities**: permutation/combinatorial (a valid solution is
  a permutation of city indices), already `DataType.DISCRETE`; needs
  permutation-aware handling, not integer rounding.

## Uncertain (need original source)

Trusses 10D/25D/72D/120D/200D: classic trusses exist in both continuous-area and
discrete-AISC-catalog variants; the code uses continuous areas with no citation in
those files. Left continuous pending a source.
