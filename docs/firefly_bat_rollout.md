# Firefly / BAT engineering-problem variant rollout

Plan + verified specs for the named-variant engineering problems from the Firefly
(Gandomi, Yang & Alavi 2011, mixed-variable) and BAT (Yang & Gandomi 2012, mostly
continuous) papers. Each problem is added with the variants below; continuous↔mixed
is selected with an `is_discrete` constructor flag (snapping the discrete dims via
`variable_types`), matching the existing pattern (PressureVessel, SpeedReducer).
"Where botorch has the same problem, we cite it" — botorch's engineering set
(`PressureVessel`, `WeldedBeamSO`, `TensionCompressionString`, `SpeedReducer`) is in
`botorch.test_functions.synthetic`.

Verification rule (as for Sellar/PEARL/MiniAeroWing): each variant must reproduce its
paper optimum (differential-evolution + penalty) before it lands.

## Status

| Problem | Variants | Reference optimum | Status |
|---|---|---|---|
| **WeldedBeamSO** | SO continuous / SO mixed (`is_discrete`) | cont 1.7249 ✓ / mixed 1.7312 | **done + verified** (continuous f*=1.7249); complements existing MO `WeldedBeam` |
| **HelicalSpring** | SO mixed (D cont, n int, d 42-discrete) | 2.6586 | **done + verified** (f*=2.6586, all 8 constraints match Table 9) |
| **SteppedCantileverBeam** | SO continuous / SO mixed (`is_discrete`) | cont 61915 / mixed 63894 | **done + verified** (cont 61914.8 gap 0.05; mixed 63896 gap 2.7 from reported-design rounding; 11 constraints from both papers) |
| **ReinforcedConcreteBeam** | SO continuous / SO mixed (`is_discrete`) | 359.208 | **done + verified** (f*=359.208 exact at As=6.32,b=34,h=8.5; added `is_discrete` to the existing class — As from Table 10, b integer) |
| CarSideImpact | already in BoCoDe (x8,x9 discrete) | 22.84 | check existing matches Firefly Case VI |
| PressureVessel / SpeedReducer | already have `is_discrete` (cont + mixed) | PV 5850.4 / SR 2994.5 | done earlier |
| ThreeTruss | already in BoCoDe (BAT three-bar truss) | 263.90 | done |

## Per-variable integral/continuous specs (from the two PDFs)

### HelicalSpring (Firefly Case III) — the richest mixed case
- Vars: `D` coil diameter (continuous), `d` wire diameter (**discrete**, 42 standard
  gauge values), `n` number of coils (**integer**).
- Objective (minimize): spring volume `f = π² D d² (n + 2) / 4`.
- 8 constraints: shear stress, surge frequency / free length, min wire diameter, max
  outer diameter, coil ratio `D/d >= 3`, deflection, free-length, working deflection.
- Optimum f* ≈ 2.6586 (d=0.283, D=1.223, n=9).

### SteppedCantileverBeam (BAT Case 6 / Firefly Case V)
- 10 vars: widths b1..b5, heights h1..h5; 5 equal segments of length l=100 cm (L=500).
- Continuous (BAT): 1≤b_i≤5, 30≤h_j≤65 → opt ≈ 61914.9.
- Mixed (Firefly): b1∈{1..5}, b2,b3∈{2.4,2.6,2.8,3.1}, h1,h2∈{45,50,55,60},
  h3∈{30..65} int, b4,b5,h4,h5 continuous → opt ≈ 63893.5.
- Constants: P=50000 N, σ_d=14000 N/cm², E=2e7 N/cm², Δ_max=2.7 cm.
- Objective: minimize volume `V = l · Σ b_i h_i`.
- 11 constraints: bending stress per segment (g1–g5), tip deflection (g6),
  aspect ratio `h_i/b_i <= 20` (g7–g11).

### ReinforcedConcreteBeam (Firefly Case IV)
- Vars: `As` rebar area (**discrete**, ACI table), `b` width (**integer**, in
  {28..40}), `h` depth (continuous, [5,10]).
- Objective: `f = 29.4 As + 0.6 b h`.
- 2 constraints: `h/b <= 4`; ACI strength requirement.
- Optimum f* ≈ 359.208 (As=6.32, b=34, h=8.5).

## Naming convention

`<Name>` (existing multi-objective, e.g. `WeldedBeam`) is kept; new single-objective
versions get the `SO` suffix (`WeldedBeamSO`). Continuous vs mixed is the
`is_discrete` flag on the same class (not a separate name), matching PressureVessel /
SpeedReducer. botorch counterparts are cited in each class docstring.
