# BoCoDe Problem Categorization

This page categorizes every optimization problem in BoCoDe. The big table below
is **generated from the per-problem metadata** in `bocode/opt_problems_metadata/`
(one JSON per problem) by `tools/render_categorization.py`, so it never drifts
from the code. Regenerate it with:

```bash
python tools/generate_metadata.py      # refresh metadata from the classes
python tools/render_categorization.py  # rewrite the table below
```

Each problem's metadata records: number of objectives, number of constraints,
dimension (or scalable dimension range), variable type (continuous / discrete /
mixed), search bounds, application area, and the convexity / NP-hardness
assessment described below.

## Columns

- **Application** — Engineering, Hyperparameter Optimization, Control, Materials,
  or Combinatorial.
- **Suite** — the benchmark family the problem comes from (e.g. MODAct, CEC2020
  RW-Constrained, RE/CRE, MuJoCo control, PV-Lab materials), so grouped problems
  like `CS1`/`CT1` (MODAct) or `RE21` are identifiable.
- **Dim** — decision-space dimension; `a+` or `a–b` for scalable problems.
- **#Obj / #Constr** — objective and (inequality + equality) constraint counts as
  declared by the problem class. See the caveat under *Known limitations*.
- **f_opt** — the known optimal objective value (natural minimization sense) where
  the problem class records one; `?` when not known. Per-problem JSON metadata also
  carries the full `variable_types`, `scalable`, and `input_type` fields.
- **Convex / NP-hard** — see methodology below; `unknown` where not determined.

## Convexity assessment

A problem is marked `convex` only when the objective(s) and feasible region are
provably convex from the closed-form definition (e.g. a convex quadratic with
linear constraints). Black-box simulators (MuJoCo, Mazda, MOPTA08, the materials
datasets) and multimodal engineering objectives are marked `unknown` or `no`:
convexity cannot be certified from oracle access, and most of these objectives
are known to be multimodal. When in doubt we record `unknown` rather than guess.

## NP-hardness methodology

NP-hardness is a property of **decision-problem classes**, not of individual
continuous instances, so we assign the `np_hard` flag using the following rules,
each grounded in the complexity-theory literature:

1. **Documented NP-hard classes → `true`.** A problem is marked NP-hard when it
   belongs to a class with a published hardness proof. In BoCoDe this currently
   applies to the **Traveling Salesman Problem** (TSP_51Cities, TSP_100Cities),
   NP-hard by reduction from Hamiltonian Cycle (Karp, *Reducibility Among
   Combinatorial Problems*, 1972). Each `true` entry carries a citation in its
   metadata `source` field.

2. **Discrete / mixed-integer nonlinear structure → usually `unknown`, `true`
   only with a citation.** Integer or categorical decision variables over a
   nonconvex objective place a problem in the MINLP family, which is NP-hard in
   general (Kannan & Monma, 1978; Murty & Kabadi, *Some NP-complete problems in
   quadratic and nonlinear programming*, 1987). We do **not** auto-label every
   mixed-integer problem `true`, because a specific instance may be polynomially
   solvable; we mark `unknown` unless a concrete reduction is documented for that
   problem.

3. **Continuous black-box simulators → `unknown` / not-applicable.** For
   oracle-access continuous problems (MuJoCo control, Mazda, MOPTA08, the truss
   simulators, the materials datasets) classical NP-hardness does not directly
   apply — there is no finite combinatorial instance to reduce to. General
   nonconvex continuous optimization is itself NP-hard (Murty & Kabadi, 1987),
   but that is a statement about the worst case of a class, not about these
   specific oracles, so we record `unknown` with this explanation rather than an
   unjustified `true`.

4. **Convex continuous problems → `false`.** Where a problem is provably convex
   it is not NP-hard (convex programs are polynomially solvable to fixed
   accuracy).

Because most BoCoDe problems are continuous engineering simulators, the default
honest answer is `unknown`; the table reflects that, with `true` reserved for the
combinatorial problems that have explicit hardness proofs.

### References

- R. M. Karp. Reducibility among combinatorial problems. *Complexity of Computer
  Computations*, 1972.
- R. Kannan and C. L. Monma. On the computational complexity of integer
  programming problems. *Lecture Notes in Economics and Mathematical Systems*,
  1978.
- K. G. Murty and S. N. Kabadi. Some NP-complete problems in quadratic and
  nonlinear programming. *Mathematical Programming* 39:117–129, 1987.

## Known limitations

The CEC2020 `num_constraints` semantics were corrected so that the count is the
total number of constraints (equality + inequality), matching the source paper's
`g + h`, and `evaluate()` now returns a single concatenated constraint tensor of
that width. Two residual discrepancies versus the guideline paper remain and are
tracked in `docs/AUDIT_findings_2026_06.md`:

- **CEC2020_p31** (gear train) is implemented as unconstrained (0 constraints);
  the paper lists `g=1, h=1`. The classic gear-train formulation is unconstrained,
  so resolving this needs the original MATLAB source to confirm the intended
  constraints. Marked `0` rather than fabricated.
- **CEC2020_p44** (wind-farm layout) generates the complete set of pairwise
  turbine-spacing constraints, `C(15,2) = 105`; the paper table lists `91`. The
  105-count is the mathematically complete pairwise set.

## Problem table

The leading `#` column numbers the problems 1…N, so the last row shows the total
problem count (currently **284**).

<!-- TABLE:START -->

| # | Problem | Application | Suite | Dim | #Obj | #Constr | f_opt | Convex | NP-hard |
|---|---|---|---|---|---|---|---|---|---|
| 1 | MaxSAT | Combinatorial | TSP / NEORL | 60 | 1 | 0 | ? | unknown | unknown |
| 2 | PestControl | Combinatorial | TSP / NEORL | 25 | 1 | 0 | ? | unknown | unknown |
| 3 | TSP_100Cities | Combinatorial | TSP / NEORL | 100 | 1 | 0 | ? | no | yes |
| 4 | TSP_51Cities | Combinatorial | TSP / NEORL | 51 | 1 | 0 | ? | no | yes |
| 5 | PD4CartPole | Control | Classic control | 4 | 1 | 0 | ? | unknown | unknown |
| 6 | PID4Acrobot | Control | Classic control | 3 | 1 | 0 | ? | unknown | unknown |
| 7 | AntPolicySearchProblem | Control | MuJoCo control | 840 | 1 | 0 | ? | unknown | unknown |
| 8 | AntProblem | Control | MuJoCo control | 8 | 1 | 0 | ? | unknown | unknown |
| 9 | HalfCheetahPolicySearchProblem | Control | MuJoCo control | 102 | 1 | 0 | ? | unknown | unknown |
| 10 | HalfCheetahProblem | Control | MuJoCo control | 6 | 1 | 0 | ? | unknown | unknown |
| 11 | HopperPolicySearchProblem | Control | MuJoCo control | 33 | 1 | 0 | ? | unknown | unknown |
| 12 | HopperProblem | Control | MuJoCo control | 3 | 1 | 0 | ? | unknown | unknown |
| 13 | HumanoidProblem | Control | MuJoCo control | 17 | 1 | 0 | ? | unknown | unknown |
| 14 | HumanoidStandupProblem | Control | MuJoCo control | 17 | 1 | 0 | ? | unknown | unknown |
| 15 | InvertedDoublePendulumProblem | Control | MuJoCo control | 1 | 1 | 0 | ? | unknown | unknown |
| 16 | InvertedPendulumProblem | Control | MuJoCo control | 1 | 1 | 0 | ? | unknown | unknown |
| 17 | PusherProblem | Control | MuJoCo control | 7 | 1 | 0 | ? | unknown | unknown |
| 18 | ReacherProblem | Control | MuJoCo control | 2 | 1 | 0 | ? | unknown | unknown |
| 19 | SwimmerPolicySearchProblem | Control | MuJoCo control | 16 | 1 | 0 | ? | unknown | unknown |
| 20 | SwimmerProblem | Control | MuJoCo control | 2 | 1 | 0 | ? | unknown | unknown |
| 21 | Walker2DPolicySearchProblem | Control | MuJoCo control | 102 | 1 | 0 | ? | unknown | unknown |
| 22 | Walker2DProblem | Control | MuJoCo control | 6 | 1 | 0 | ? | unknown | unknown |
| 23 | PowerElectronics | EngiBench | EngiBench | 20 | 2 | 0 | ? | unknown | unknown |
| 24 | CEC2020_p1 | Engineering | CEC2020 RW-Constrained | 9 | 1 | 8 | 189.31163 | unknown | unknown |
| 25 | CEC2020_p10 | Engineering | CEC2020 RW-Constrained | 3 | 1 | 3 | 1.076543 | unknown | unknown |
| 26 | CEC2020_p11 | Engineering | CEC2020 RW-Constrained | 7 | 1 | 8 | 99.238464 | unknown | unknown |
| 27 | CEC2020_p12 | Engineering | CEC2020 RW-Constrained | 7 | 1 | 9 | 2.924831 | unknown | unknown |
| 28 | CEC2020_p13 | Engineering | CEC2020 RW-Constrained | 5 | 1 | 3 | 26887.0 | unknown | unknown |
| 29 | CEC2020_p14 | Engineering | CEC2020 RW-Constrained | 10 | 1 | 10 | 53638.942722 | unknown | unknown |
| 30 | CEC2020_p15 | Engineering | CEC2020 RW-Constrained | 7 | 1 | 11 | 2994.424466 | unknown | unknown |
| 31 | CEC2020_p16 | Engineering | CEC2020 RW-Constrained | 14 | 1 | 15 | 0.032213 | unknown | unknown |
| 32 | CEC2020_p17 | Engineering | CEC2020 RW-Constrained | 3 | 1 | 4 | 0.012665 | unknown | unknown |
| 33 | CEC2020_p18 | Engineering | CEC2020 RW-Constrained | 4 | 1 | 4 | ? | unknown | unknown |
| 34 | CEC2020_p19 | Engineering | CEC2020 RW-Constrained | 4 | 1 | 5 | 1.670218 | unknown | unknown |
| 35 | CEC2020_p2 | Engineering | CEC2020 RW-Constrained | 11 | 1 | 9 | 7049.036954 | unknown | unknown |
| 36 | CEC2020_p20 | Engineering | CEC2020 RW-Constrained | 2 | 1 | 3 | 263.895843 | unknown | unknown |
| 37 | CEC2020_p21 | Engineering | CEC2020 RW-Constrained | 5 | 1 | 8 | 0.235242 | unknown | unknown |
| 38 | CEC2020_p22 | Engineering | CEC2020 RW-Constrained | 9 | 1 | 11 | 0.525769 | unknown | unknown |
| 39 | CEC2020_p23 | Engineering | CEC2020 RW-Constrained | 5 | 1 | 11 | 16.069869 | unknown | unknown |
| 40 | CEC2020_p24 | Engineering | CEC2020 RW-Constrained | 7 | 1 | 7 | 2.528792 | unknown | unknown |
| 41 | CEC2020_p25 | Engineering | CEC2020 RW-Constrained | 7 | 1 | 7 | 1616.119765 | unknown | unknown |
| 42 | CEC2020_p26 | Engineering | CEC2020 RW-Constrained | 22 | 1 | 87 | 35.359232 | unknown | unknown |
| 43 | CEC2020_p27 | Engineering | CEC2020 RW-Constrained | 10 | 1 | 3 | 524.450761 | unknown | unknown |
| 44 | CEC2020_p28 | Engineering | CEC2020 RW-Constrained | 10 | 1 | 9 | 14614.135715 | unknown | unknown |
| 45 | CEC2020_p29 | Engineering | CEC2020 RW-Constrained | 4 | 1 | 1 | 2964895.4173 | unknown | unknown |
| 46 | CEC2020_p3 | Engineering | CEC2020 RW-Constrained | 7 | 1 | 14 | -4529.119739 | unknown | unknown |
| 47 | CEC2020_p30 | Engineering | CEC2020 RW-Constrained | 3 | 1 | 8 | 2.613884 | unknown | unknown |
| 48 | CEC2020_p31 | Engineering | CEC2020 RW-Constrained | 4 | 1 | 0 | 0.0 | unknown | unknown |
| 49 | CEC2020_p32 | Engineering | CEC2020 RW-Constrained | 5 | 1 | 6 | -30665.538672 | unknown | unknown |
| 50 | CEC2020_p33 | Engineering | CEC2020 RW-Constrained | 30 | 1 | 30 | 2.639346 | unknown | unknown |
| 51 | CEC2020_p34 | Engineering | CEC2020 RW-Constrained | 118 | 1 | 108 | 0.0 | unknown | unknown |
| 52 | CEC2020_p35 | Engineering | CEC2020 RW-Constrained | 153 | 1 | 148 | 0.079964 | unknown | unknown |
| 53 | CEC2020_p36 | Engineering | CEC2020 RW-Constrained | 158 | 1 | 148 | 0.047734 | unknown | unknown |
| 54 | CEC2020_p37 | Engineering | CEC2020 RW-Constrained | 126 | 1 | 116 | 0.018594 | unknown | unknown |
| 55 | CEC2020_p38 | Engineering | CEC2020 RW-Constrained | 126 | 1 | 116 | 2.713937 | unknown | unknown |
| 56 | CEC2020_p39 | Engineering | CEC2020 RW-Constrained | 126 | 1 | 116 | 2.751591 | unknown | unknown |
| 57 | CEC2020_p4 | Engineering | CEC2020 RW-Constrained | 6 | 1 | 5 | -0.38826 | unknown | unknown |
| 58 | CEC2020_p40 | Engineering | CEC2020 RW-Constrained | 76 | 1 | 76 | 0.0 | unknown | unknown |
| 59 | CEC2020_p41 | Engineering | CEC2020 RW-Constrained | 74 | 1 | 74 | 0.0 | unknown | unknown |
| 60 | CEC2020_p42 | Engineering | CEC2020 RW-Constrained | 86 | 1 | 76 | 0.077027 | unknown | unknown |
| 61 | CEC2020_p43 | Engineering | CEC2020 RW-Constrained | 86 | 1 | 76 | 0.079836 | unknown | unknown |
| 62 | CEC2020_p44 | Engineering | CEC2020 RW-Constrained | 30 | 1 | 105 | -6273.1715 | unknown | unknown |
| 63 | CEC2020_p45 | Engineering | CEC2020 RW-Constrained | 25 | 1 | 25 | 0.030739 | unknown | unknown |
| 64 | CEC2020_p46 | Engineering | CEC2020 RW-Constrained | 25 | 1 | 25 | 0.02024 | unknown | unknown |
| 65 | CEC2020_p47 | Engineering | CEC2020 RW-Constrained | 25 | 1 | 25 | 0.012783 | unknown | unknown |
| 66 | CEC2020_p48 | Engineering | CEC2020 RW-Constrained | 30 | 1 | 30 | 0.016788 | unknown | unknown |
| 67 | CEC2020_p49 | Engineering | CEC2020 RW-Constrained | 30 | 1 | 30 | 0.009312 | unknown | unknown |
| 68 | CEC2020_p5 | Engineering | CEC2020 RW-Constrained | 9 | 1 | 6 | -400.0056 | unknown | unknown |
| 69 | CEC2020_p50 | Engineering | CEC2020 RW-Constrained | 30 | 1 | 30 | 0.015051 | unknown | unknown |
| 70 | CEC2020_p51 | Engineering | CEC2020 RW-Constrained | 59 | 1 | 15 | 4550.85115 | unknown | unknown |
| 71 | CEC2020_p52 | Engineering | CEC2020 RW-Constrained | 59 | 1 | 15 | 3348.982149 | unknown | unknown |
| 72 | CEC2020_p53 | Engineering | CEC2020 RW-Constrained | 59 | 1 | 15 | 4997.606929 | unknown | unknown |
| 73 | CEC2020_p54 | Engineering | CEC2020 RW-Constrained | 59 | 1 | 15 | 4240.548254 | unknown | unknown |
| 74 | CEC2020_p55 | Engineering | CEC2020 RW-Constrained | 64 | 1 | 6 | 6696.414513 | unknown | unknown |
| 75 | CEC2020_p56 | Engineering | CEC2020 RW-Constrained | 64 | 1 | 6 | 14746.58 | unknown | unknown |
| 76 | CEC2020_p57 | Engineering | CEC2020 RW-Constrained | 64 | 1 | 6 | 3213.291702 | unknown | unknown |
| 77 | CEC2020_p6 | Engineering | CEC2020 RW-Constrained | 38 | 1 | 32 | 1.86383 | unknown | unknown |
| 78 | CEC2020_p7 | Engineering | CEC2020 RW-Constrained | 48 | 1 | 38 | 1.567045 | unknown | unknown |
| 79 | CEC2020_p8 | Engineering | CEC2020 RW-Constrained | 2 | 1 | 2 | 2.0 | unknown | unknown |
| 80 | CEC2020_p9 | Engineering | CEC2020 RW-Constrained | 3 | 1 | 2 | 2.557655 | unknown | unknown |
| 81 | CRE21 | Engineering | CRE (Tanabe-Ishibuchi) | 3 | 2 | 3 | ? | unknown | unknown |
| 82 | CRE22 | Engineering | CRE (Tanabe-Ishibuchi) | 4 | 2 | 4 | ? | unknown | unknown |
| 83 | CRE23 | Engineering | CRE (Tanabe-Ishibuchi) | 4 | 2 | 4 | ? | unknown | unknown |
| 84 | CRE24 | Engineering | CRE (Tanabe-Ishibuchi) | 7 | 2 | 11 | ? | unknown | unknown |
| 85 | CRE25 | Engineering | CRE (Tanabe-Ishibuchi) | 4 | 2 | 1 | ? | unknown | unknown |
| 86 | CRE31 | Engineering | CRE (Tanabe-Ishibuchi) | 7 | 3 | 10 | ? | unknown | unknown |
| 87 | CRE32 | Engineering | CRE (Tanabe-Ishibuchi) | 6 | 3 | 9 | ? | unknown | unknown |
| 88 | CRE51 | Engineering | CRE (Tanabe-Ishibuchi) | 3 | 5 | 7 | ? | unknown | unknown |
| 89 | Allison | Engineering | Engineering (standalone) | 3 | 1 | 0 | 0.5698 | unknown | unknown |
| 90 | Borehole | Engineering | Engineering (standalone) | 8 | 1 | 0 | ? | unknown | unknown |
| 91 | BotorchCarSideImpact | Engineering | Engineering (standalone) | 7 | 4 | 0 | ? | unknown | unknown |
| 92 | CantileverBeam | Engineering | Engineering (standalone) | 10 | 1 | 11 | 61914.87 | unknown | unknown |
| 93 | Car | Engineering | Engineering (standalone) | 11 | 1 | 10 | ? | unknown | unknown |
| 94 | CarSideImpact | Engineering | Engineering (standalone) | 7 | 3 | 10 | ? | unknown | unknown |
| 95 | ColumnBuckling | Engineering | Engineering (standalone) | 4 | 1 | 0 | ? | unknown | unknown |
| 96 | CompressionSpring | Engineering | Engineering (standalone) | 3 | 1 | 4 | ? | unknown | unknown |
| 97 | DiscBrake | Engineering | Engineering (standalone) | 4 | 2 | 4 | ? | unknown | unknown |
| 98 | EulerBeamMixed | Engineering | Engineering (standalone) | 3 | 1 | 0 | 1286.97 | unknown | unknown |
| 99 | EulerBernoulliBeamBending | Engineering | Engineering (standalone) | 3 | 1 | 0 | -1287.385 | unknown | unknown |
| 100 | GearTrain | Engineering | Engineering (standalone) | 4 | 1 | 0 | ? | unknown | unknown |
| 101 | HeatExchanger | Engineering | Engineering (standalone) | 8 | 1 | 6 | ? | unknown | unknown |
| 102 | HelicalSpring | Engineering | Engineering (standalone) | 3 | 1 | 8 | 2.6586 | unknown | unknown |
| 103 | MOPTA08Car | Engineering | Engineering (standalone) | 124 | 1 | 68 | ? | no | unknown |
| 104 | Mazda | Engineering | Engineering (standalone) | 222 | 5 | 54 | ? | unknown | unknown |
| 105 | Mazda_SCA | Engineering | Engineering (standalone) | 148 | 4 | 36 | ? | unknown | unknown |
| 106 | MiniAeroWing | Engineering | Engineering (standalone) | 3 | 1 | 0 | 242.27 | unknown | unknown |
| 107 | PEARL | Engineering | Engineering (standalone) | 7 | 1 | 6 | 585.3 | unknown | unknown |
| 108 | Penicillin | Engineering | Engineering (standalone) | 7 | 3 | 0 | ? | unknown | unknown |
| 109 | PressureVessel | Engineering | Engineering (standalone) | 4 | 1 | 4 | ? | unknown | unknown |
| 110 | QPowerModel | Engineering | Engineering (standalone) | 8 | 1 | 0 | ? | unknown | unknown |
| 111 | RE21 | Engineering | Engineering (standalone) | 4 | 2 | 0 | ? | unknown | unknown |
| 112 | RE22 | Engineering | Engineering (standalone) | 3 | 2 | 0 | ? | unknown | unknown |
| 113 | RE23 | Engineering | Engineering (standalone) | 4 | 2 | 0 | ? | unknown | unknown |
| 114 | RE24 | Engineering | Engineering (standalone) | 2 | 2 | 0 | ? | unknown | unknown |
| 115 | RE25 | Engineering | Engineering (standalone) | 3 | 2 | 0 | ? | unknown | unknown |
| 116 | RE31 | Engineering | Engineering (standalone) | 3 | 3 | 0 | ? | unknown | unknown |
| 117 | RE32 | Engineering | Engineering (standalone) | 4 | 3 | 0 | ? | unknown | unknown |
| 118 | RE33 | Engineering | Engineering (standalone) | 4 | 3 | 0 | ? | unknown | unknown |
| 119 | RE34 | Engineering | Engineering (standalone) | 5 | 3 | 0 | ? | unknown | unknown |
| 120 | RE35 | Engineering | Engineering (standalone) | 7 | 3 | 0 | ? | unknown | unknown |
| 121 | RE36 | Engineering | Engineering (standalone) | 4 | 3 | 0 | ? | unknown | unknown |
| 122 | RE37 | Engineering | Engineering (standalone) | 4 | 3 | 0 | ? | unknown | unknown |
| 123 | RE41 | Engineering | Engineering (standalone) | 7 | 4 | 0 | ? | unknown | unknown |
| 124 | RE42 | Engineering | Engineering (standalone) | 6 | 4 | 0 | ? | unknown | unknown |
| 125 | RE61 | Engineering | Engineering (standalone) | 3 | 6 | 0 | ? | unknown | unknown |
| 126 | RE91 | Engineering | Engineering (standalone) | 7 | 9 | 0 | ? | unknown | unknown |
| 127 | ReactivityModel | Engineering | Engineering (standalone) | 8 | 1 | 0 | ? | unknown | unknown |
| 128 | ReinforcedConcreteBeam | Engineering | Engineering (standalone) | 3 | 1 | 2 | 359.208 | unknown | unknown |
| 129 | RobotPush | Engineering | Engineering (standalone) | 14 | 1 | 0 | ? | no | unknown |
| 130 | Rover | Engineering | Engineering (standalone) | 100 | 1 | 0 | ? | unknown | unknown |
| 131 | SatelliteDesign | Engineering | Engineering (standalone) | 4 | 1 | 3 | ? | unknown | unknown |
| 132 | Sellar | Engineering | Engineering (standalone) | 3 | 1 | 2 | 3.18339 | unknown | unknown |
| 133 | SpeedReducer | Engineering | Engineering (standalone) | 7 | 1 | 9 | ? | unknown | unknown |
| 134 | SteppedCantileverBeam | Engineering | Engineering (standalone) | 10 | 1 | 11 | 63893.53 | unknown | unknown |
| 135 | ThreeTruss | Engineering | Engineering (standalone) | 2 | 1 | 3 | ? | unknown | unknown |
| 136 | Truss10D | Engineering | Engineering (standalone) | 10 | 1 | 14 | ? | unknown | unknown |
| 137 | Truss120D | Engineering | Engineering (standalone) | 120 | 1 | 121 | ? | unknown | unknown |
| 138 | Truss200D | Engineering | Engineering (standalone) | 200 | 1 | 200 | ? | unknown | unknown |
| 139 | Truss25D | Engineering | Engineering (standalone) | 25 | 1 | 31 | ? | unknown | unknown |
| 140 | Truss72D_FourForces | Engineering | Engineering (standalone) | 72 | 1 | 88 | ? | unknown | unknown |
| 141 | Truss72D_SingleForce | Engineering | Engineering (standalone) | 72 | 1 | 88 | ? | unknown | unknown |
| 142 | TwoBarTruss | Engineering | Engineering (standalone) | 2 | 2 | 5 | ? | unknown | unknown |
| 143 | VehicleSafety | Engineering | Engineering (standalone) | 5 | 3 | 0 | ? | unknown | unknown |
| 144 | WaterProblem | Engineering | Engineering (standalone) | 3 | 5 | 7 | ? | unknown | unknown |
| 145 | WaterResources | Engineering | Engineering (standalone) | 3 | 5 | 7 | ? | unknown | unknown |
| 146 | WeldedBeam | Engineering | Engineering (standalone) | 4 | 2 | 4 | ? | unknown | unknown |
| 147 | WeldedBeamSO | Engineering | Engineering (standalone) | 4 | 1 | 7 | ? | unknown | unknown |
| 148 | Wing | Engineering | Engineering (standalone) | 10 | 1 | 0 | ? | unknown | unknown |
| 149 | CS1 | Engineering | MODAct (actuator design) | 20 | 2 | 7 | ? | unknown | unknown |
| 150 | CS2 | Engineering | MODAct (actuator design) | 20 | 2 | 8 | ? | unknown | unknown |
| 151 | CS3 | Engineering | MODAct (actuator design) | 20 | 2 | 10 | ? | unknown | unknown |
| 152 | CS4 | Engineering | MODAct (actuator design) | 20 | 2 | 9 | ? | unknown | unknown |
| 153 | CT1 | Engineering | MODAct (actuator design) | 20 | 2 | 7 | ? | unknown | unknown |
| 154 | CT2 | Engineering | MODAct (actuator design) | 20 | 2 | 8 | ? | unknown | unknown |
| 155 | CT3 | Engineering | MODAct (actuator design) | 20 | 2 | 10 | ? | unknown | unknown |
| 156 | CT4 | Engineering | MODAct (actuator design) | 20 | 2 | 9 | ? | unknown | unknown |
| 157 | CTS1 | Engineering | MODAct (actuator design) | 20 | 3 | 7 | ? | unknown | unknown |
| 158 | CTS2 | Engineering | MODAct (actuator design) | 20 | 3 | 8 | ? | unknown | unknown |
| 159 | CTS3 | Engineering | MODAct (actuator design) | 20 | 3 | 10 | ? | unknown | unknown |
| 160 | CTS4 | Engineering | MODAct (actuator design) | 20 | 3 | 9 | ? | unknown | unknown |
| 161 | CTSE1 | Engineering | MODAct (actuator design) | 20 | 4 | 7 | ? | unknown | unknown |
| 162 | CTSE2 | Engineering | MODAct (actuator design) | 20 | 4 | 8 | ? | unknown | unknown |
| 163 | CTSE3 | Engineering | MODAct (actuator design) | 20 | 4 | 10 | ? | unknown | unknown |
| 164 | CTSE4 | Engineering | MODAct (actuator design) | 20 | 4 | 9 | ? | unknown | unknown |
| 165 | CTSEI1 | Engineering | MODAct (actuator design) | 20 | 5 | 7 | ? | unknown | unknown |
| 166 | CTSEI2 | Engineering | MODAct (actuator design) | 20 | 5 | 8 | ? | unknown | unknown |
| 167 | CTSEI3 | Engineering | MODAct (actuator design) | 20 | 5 | 10 | ? | unknown | unknown |
| 168 | CTSEI4 | Engineering | MODAct (actuator design) | 20 | 5 | 9 | ? | unknown | unknown |
| 169 | HPOBSurr_4796_23 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 3 | 1 | 0 | ? | unknown | unknown |
| 170 | HPOBSurr_4796_3549 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 3 | 1 | 0 | ? | unknown | unknown |
| 171 | HPOBSurr_4796_3918 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 3 | 1 | 0 | ? | unknown | unknown |
| 172 | HPOBSurr_4796_9903 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 3 | 1 | 0 | ? | unknown | unknown |
| 173 | HPOBSurr_4796_9906 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 3 | 1 | 0 | ? | unknown | unknown |
| 174 | HPOBSurr_4796_9946 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 3 | 1 | 0 | ? | unknown | unknown |
| 175 | HPOBSurr_5527_10101 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 176 | HPOBSurr_5527_145804 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 177 | HPOBSurr_5527_146064 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 178 | HPOBSurr_5527_146065 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 179 | HPOBSurr_5527_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 180 | HPOBSurr_5527_9914 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 181 | HPOBSurr_5636_10101 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 182 | HPOBSurr_5636_145804 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 183 | HPOBSurr_5636_146064 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 184 | HPOBSurr_5636_146065 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 185 | HPOBSurr_5636_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 186 | HPOBSurr_5636_9914 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 187 | HPOBSurr_5859_125923 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 188 | HPOBSurr_5859_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 189 | HPOBSurr_5859_37 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 190 | HPOBSurr_5859_3902 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 191 | HPOBSurr_5859_9977 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 192 | HPOBSurr_5859_9983 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 193 | HPOBSurr_5889_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 194 | HPOBSurr_5889_3493 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 195 | HPOBSurr_5889_3918 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 196 | HPOBSurr_5889_3950 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 197 | HPOBSurr_5889_49 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 198 | HPOBSurr_5889_9971 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 199 | HPOBSurr_5891_3492 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 200 | HPOBSurr_5891_3891 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 201 | HPOBSurr_5891_3899 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 202 | HPOBSurr_5891_6566 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 203 | HPOBSurr_5891_9889 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 204 | HPOBSurr_5891_9980 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 205 | HPOBSurr_5906_3889 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 206 | HPOBSurr_5906_3896 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 207 | HPOBSurr_5906_3918 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 208 | HPOBSurr_5906_9970 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 209 | HPOBSurr_5906_9971 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 210 | HPOBSurr_5906_9977 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 211 | HPOBSurr_5965_10101 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 212 | HPOBSurr_5965_145836 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 213 | HPOBSurr_5965_3903 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 214 | HPOBSurr_5965_49 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 215 | HPOBSurr_5965_9889 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 216 | HPOBSurr_5965_9914 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 217 | HPOBSurr_5965_9946 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 218 | HPOBSurr_5970_14951 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 219 | HPOBSurr_5970_34536 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 220 | HPOBSurr_5970_3492 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 221 | HPOBSurr_5970_37 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 222 | HPOBSurr_5970_49 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 223 | HPOBSurr_5970_9952 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 224 | HPOBSurr_5971_10093 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 225 | HPOBSurr_5971_34536 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 226 | HPOBSurr_5971_3954 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 227 | HPOBSurr_5971_43 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 228 | HPOBSurr_5971_6566 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 229 | HPOBSurr_5971_9970 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 230 | HPOBSurr_6766_10101 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 231 | HPOBSurr_6766_145804 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 232 | HPOBSurr_6766_145953 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 233 | HPOBSurr_6766_146064 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 234 | HPOBSurr_6766_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 235 | HPOBSurr_6766_3903 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 236 | HPOBSurr_6767_145804 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 18 | 1 | 0 | ? | unknown | unknown |
| 237 | HPOBSurr_6767_146064 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 18 | 1 | 0 | ? | unknown | unknown |
| 238 | HPOBSurr_6767_146065 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 18 | 1 | 0 | ? | unknown | unknown |
| 239 | HPOBSurr_6767_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 18 | 1 | 0 | ? | unknown | unknown |
| 240 | HPOBSurr_6767_9914 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 18 | 1 | 0 | ? | unknown | unknown |
| 241 | HPOBSurr_6767_9967 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 18 | 1 | 0 | ? | unknown | unknown |
| 242 | HPOBSurr_6794_10101 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 243 | HPOBSurr_6794_145804 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 244 | HPOBSurr_6794_146065 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 245 | HPOBSurr_6794_3 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 246 | HPOBSurr_6794_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 247 | HPOBSurr_6794_9914 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 248 | HPOBSurr_7607_145976 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 249 | HPOBSurr_7607_3896 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 250 | HPOBSurr_7607_3903 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 251 | HPOBSurr_7607_3913 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 252 | HPOBSurr_7607_9946 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 253 | HPOBSurr_7607_9967 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 254 | HPOBSurr_7609_125923 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 255 | HPOBSurr_7609_145853 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 256 | HPOBSurr_7609_145854 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 257 | HPOBSurr_7609_145878 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 258 | HPOBSurr_7609_34537 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 259 | HPOBSurr_7609_3903 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 260 | HPOBSurr_7609_9967 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 261 | HPOB_4796_23 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 3 | 1 | 0 | ? | unknown | unknown |
| 262 | HPOB_4796_3549 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 3 | 1 | 0 | ? | unknown | unknown |
| 263 | HPOB_4796_3918 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 3 | 1 | 0 | ? | unknown | unknown |
| 264 | HPOB_4796_9903 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 3 | 1 | 0 | ? | unknown | unknown |
| 265 | HPOB_4796_9906 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 3 | 1 | 0 | ? | unknown | unknown |
| 266 | HPOB_4796_9946 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 3 | 1 | 0 | ? | unknown | unknown |
| 267 | HPOB_5527_10101 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 268 | HPOB_5527_145804 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 269 | HPOB_5527_146064 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 270 | HPOB_5527_146065 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 271 | HPOB_5527_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 272 | HPOB_5527_9914 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 273 | HPOB_5636_10101 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 274 | HPOB_5636_145804 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 275 | HPOB_5636_146064 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 276 | HPOB_5636_146065 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 277 | HPOB_5636_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 278 | HPOB_5636_9914 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 279 | HPOB_5859_125923 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 280 | HPOB_5859_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 281 | HPOB_5859_37 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 282 | HPOB_5859_3902 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 283 | HPOB_5859_9977 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 284 | HPOB_5859_9983 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 285 | HPOB_5889_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 286 | HPOB_5889_3493 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 287 | HPOB_5889_3918 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 288 | HPOB_5889_3950 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 289 | HPOB_5889_49 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 290 | HPOB_5889_9971 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 6 | 1 | 0 | ? | unknown | unknown |
| 291 | HPOB_5891_3492 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 292 | HPOB_5891_3891 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 293 | HPOB_5891_3899 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 294 | HPOB_5891_6566 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 295 | HPOB_5891_9889 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 296 | HPOB_5891_9980 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 297 | HPOB_5906_3889 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 298 | HPOB_5906_3896 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 299 | HPOB_5906_3918 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 300 | HPOB_5906_9970 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 301 | HPOB_5906_9971 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 302 | HPOB_5906_9977 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 303 | HPOB_5965_10101 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 304 | HPOB_5965_145836 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 305 | HPOB_5965_3903 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 306 | HPOB_5965_49 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 307 | HPOB_5965_9889 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 308 | HPOB_5965_9914 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 309 | HPOB_5965_9946 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 310 | HPOB_5970_14951 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 311 | HPOB_5970_34536 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 312 | HPOB_5970_3492 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 313 | HPOB_5970_37 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 314 | HPOB_5970_49 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 315 | HPOB_5970_9952 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 316 | HPOB_5971_10093 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 317 | HPOB_5971_34536 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 318 | HPOB_5971_3954 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 319 | HPOB_5971_43 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 320 | HPOB_5971_6566 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 321 | HPOB_5971_9970 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 16 | 1 | 0 | ? | unknown | unknown |
| 322 | HPOB_6766_10101 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 323 | HPOB_6766_145804 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 324 | HPOB_6766_145953 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 325 | HPOB_6766_146064 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 326 | HPOB_6766_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 327 | HPOB_6766_3903 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 2 | 1 | 0 | ? | unknown | unknown |
| 328 | HPOB_6767_145804 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 18 | 1 | 0 | ? | unknown | unknown |
| 329 | HPOB_6767_146064 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 18 | 1 | 0 | ? | unknown | unknown |
| 330 | HPOB_6767_146065 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 18 | 1 | 0 | ? | unknown | unknown |
| 331 | HPOB_6767_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 18 | 1 | 0 | ? | unknown | unknown |
| 332 | HPOB_6767_9914 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 18 | 1 | 0 | ? | unknown | unknown |
| 333 | HPOB_6767_9967 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 18 | 1 | 0 | ? | unknown | unknown |
| 334 | HPOB_6794_10101 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 335 | HPOB_6794_145804 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 336 | HPOB_6794_146065 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 337 | HPOB_6794_3 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 338 | HPOB_6794_31 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 339 | HPOB_6794_9914 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 10 | 1 | 0 | ? | unknown | unknown |
| 340 | HPOB_7607_145976 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 341 | HPOB_7607_3896 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 342 | HPOB_7607_3903 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 343 | HPOB_7607_3913 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 344 | HPOB_7607_9946 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 345 | HPOB_7607_9967 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 346 | HPOB_7609_125923 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 347 | HPOB_7609_145853 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 348 | HPOB_7609_145854 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 349 | HPOB_7609_145878 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 350 | HPOB_7609_34537 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 351 | HPOB_7609_3903 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 352 | HPOB_7609_9967 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 353 | LCBenchAPSFailure | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 354 | LCBenchAdult | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 355 | LCBenchAirlines | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 356 | LCBenchAlbert | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 357 | LCBenchAmazonEmployeeAccess | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 358 | LCBenchAustralian | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 359 | LCBenchBankMarketing | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 360 | LCBenchBloodTransfusionServiceCenter | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 361 | LCBenchCar | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 362 | LCBenchChristine | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 363 | LCBenchCnae9 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 364 | LCBenchConnect4 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 365 | LCBenchCovertype | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 366 | LCBenchCreditG | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 367 | LCBenchDionis | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 368 | LCBenchFabert | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 369 | LCBenchFashionMNIST | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 370 | LCBenchHelena | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 371 | LCBenchHiggs | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 372 | LCBenchJannis | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 373 | LCBenchJasmine | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 374 | LCBenchJungleChess2pcsRawEndgameComplete | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 375 | LCBenchKDDCup09Appetency | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 376 | LCBenchKc1 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 377 | LCBenchKrVsKp | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 378 | LCBenchMfeatFactors | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 379 | LCBenchMiniBooNE | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 380 | LCBenchNomao | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 381 | LCBenchNumerai286 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 382 | LCBenchPhoneme | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 383 | LCBenchSegment | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 384 | LCBenchShuttle | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 385 | LCBenchSylvine | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 386 | LCBenchVehicle | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 387 | LCBenchVolkert | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7 | 1 | 0 | ? | unknown | unknown |
| 388 | LassoBreastCancer | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 9 | 1 | 0 | ? | unknown | unknown |
| 389 | LassoDNA | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 180 | 1 | 0 | ? | unknown | unknown |
| 390 | LassoDiabetes | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 8 | 1 | 0 | ? | unknown | unknown |
| 391 | LassoLeukemia | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 7129 | 1 | 0 | ? | unknown | unknown |
| 392 | LassoRCV1 | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 47236 | 1 | 0 | ? | unknown | unknown |
| 393 | SVM | Hyperparameter Optimization | HPO (LassoBench / HPO-B) | 388 | 1 | 0 | ? | no | unknown |
| 394 | NASBench201 | Hyperparameter Optimization | NAS-Bench-201 | 6 | 1 | 0 | ? | unknown | unknown |
| 395 | AgNP | Materials | PV-Lab materials | 5 | 1 | 0 | ? | unknown | unknown |
| 396 | AutoAM | Materials | PV-Lab materials | 4 | 1 | 0 | ? | unknown | unknown |
| 397 | CrossedBarrel | Materials | PV-Lab materials | 4 | 1 | 0 | ? | unknown | unknown |
| 398 | HOIP | Materials | PV-Lab materials | 3 | 1 | 0 | ? | unknown | unknown |
| 399 | P3HT | Materials | PV-Lab materials | 5 | 1 | 0 | ? | unknown | unknown |
| 400 | Perovskite | Materials | PV-Lab materials | 3 | 1 | 0 | ? | unknown | unknown |
| 401 | Ackley53 | Synthetic | Categorical SFU grids | 53 | 1 | 0 | 0.0 | no | unknown |
| 402 | AckleyCat | Synthetic | Categorical SFU grids | 20 | 1 | 0 | 0.0 | no | unknown |
| 403 | RastriginCat | Synthetic | Categorical SFU grids | 20 | 1 | 0 | 0.0 | no | unknown |
| 404 | SchwefelCat | Synthetic | Categorical SFU grids | 20 | 1 | 0 | 7256.810531 | no | unknown |
| 405 | Ackley5Mixed | Synthetic | Mixed-variable synthetic | 5 | 1 | 0 | 0.0 | unknown | unknown |
| 406 | BraninCategorical | Synthetic | Mixed-variable synthetic | 3 | 1 | 0 | 0.397887 | unknown | unknown |
| 407 | BraninLVGP | Synthetic | Mixed-variable synthetic | 2 | 1 | 0 | 2.79118 | unknown | unknown |
| 408 | CoCaBOFunc2C | Synthetic | Mixed-variable synthetic | 4 | 1 | 0 | ? | unknown | unknown |
| 409 | CoCaBOFunc3C | Synthetic | Mixed-variable synthetic | 5 | 1 | 0 | ? | unknown | unknown |
| 410 | GoldsteinLVGP | Synthetic | Mixed-variable synthetic | 2 | 1 | 0 | 3.0 | unknown | unknown |
| 411 | GoldsteinMixed | Synthetic | Mixed-variable synthetic | 2 | 1 | 0 | 3.0 | unknown | unknown |
| 412 | HartmannCat | Synthetic | Mixed-variable synthetic | 6 | 1 | 0 | -3.32237 | unknown | unknown |
| 413 | MixedAckley | Synthetic | Mixed-variable synthetic | 5 | 1 | 0 | 0.0 | unknown | unknown |
| 414 | Rosenbrock5Mixed | Synthetic | Mixed-variable synthetic | 5 | 1 | 0 | 0.0 | unknown | unknown |
| 415 | ShekelMixed | Synthetic | Mixed-variable synthetic | 4 | 1 | 0 | -10.536363 | unknown | unknown |
| 416 | StyblinskiTangCat | Synthetic | Mixed-variable synthetic | 10 | 1 | 0 | -367.1875 | unknown | unknown |
| 417 | StyblinskiTangMixed | Synthetic | Mixed-variable synthetic | 5 | 1 | 0 | -156.66464 | unknown | unknown |
| 418 | WeldedBeamCategorical | Synthetic | Mixed-variable synthetic | 5 | 1 | 5 | 1.7249 | unknown | unknown |

<!-- TABLE:END -->

## Problems by category

The same problems grouped by the filters used in `bocode.list_problems(...)`. Each
heading shows the exact call. (Generated; rerun `python tools/render_categorization.py`.)

<!-- CATEGORIES:START -->

### Single-objective, unconstrained, continuous (64)

`list_problems(num_objectives=1, constrained=False, input_type='continuous')`

`AgNP`, `Allison`, `AntPolicySearchProblem`, `AntProblem`, `AutoAM`, `Borehole`, `CEC2020_p31`, `EulerBernoulliBeamBending`, `HPOBSurr_5970_14951`, `HPOBSurr_5970_34536`, `HPOBSurr_5970_3492`, `HPOBSurr_5970_37`, `HPOBSurr_5970_49`, `HPOBSurr_5970_9952`, `HPOBSurr_6766_10101`, `HPOBSurr_6766_145804`, `HPOBSurr_6766_145953`, `HPOBSurr_6766_146064`, `HPOBSurr_6766_31`, `HPOBSurr_6766_3903`, `HPOB_5970_14951`, `HPOB_5970_34536`, `HPOB_5970_3492`, `HPOB_5970_37`, `HPOB_5970_49`, `HPOB_5970_9952`, `HPOB_6766_10101`, `HPOB_6766_145804`, `HPOB_6766_145953`, `HPOB_6766_146064`, `HPOB_6766_31`, `HPOB_6766_3903`, `HalfCheetahPolicySearchProblem`, `HalfCheetahProblem`, `HopperPolicySearchProblem`, `HopperProblem`, `HumanoidProblem`, `HumanoidStandupProblem`, `InvertedDoublePendulumProblem`, `InvertedPendulumProblem`, `LassoBreastCancer`, `LassoDNA`, `LassoDiabetes`, `LassoLeukemia`, `LassoRCV1`, `MiniAeroWing`, `P3HT`, `PD4CartPole`, `PID4Acrobot`, `Perovskite`, `PusherProblem`, `QPowerModel`, `ReacherProblem`, `ReactivityModel`, `RobotPush`, `Rover`, `SVM`, `SwimmerPolicySearchProblem`, `SwimmerProblem`, `TSP_100Cities`, `TSP_51Cities`, `Walker2DPolicySearchProblem`, `Walker2DProblem`, `Wing`

### Single-objective, unconstrained, mixed-variable (210)

`list_problems(num_objectives=1, constrained=False, input_type='mixed')`

`Ackley53`, `Ackley5Mixed`, `BraninCategorical`, `BraninLVGP`, `CoCaBOFunc2C`, `CoCaBOFunc3C`, `ColumnBuckling`, `EulerBeamMixed`, `GoldsteinLVGP`, `GoldsteinMixed`, `HPOBSurr_4796_23`, `HPOBSurr_4796_3549`, `HPOBSurr_4796_3918`, `HPOBSurr_4796_9903`, `HPOBSurr_4796_9906`, `HPOBSurr_4796_9946`, `HPOBSurr_5527_10101`, `HPOBSurr_5527_145804`, `HPOBSurr_5527_146064`, `HPOBSurr_5527_146065`, `HPOBSurr_5527_31`, `HPOBSurr_5527_9914`, `HPOBSurr_5636_10101`, `HPOBSurr_5636_145804`, `HPOBSurr_5636_146064`, `HPOBSurr_5636_146065`, `HPOBSurr_5636_31`, `HPOBSurr_5636_9914`, `HPOBSurr_5859_125923`, `HPOBSurr_5859_31`, `HPOBSurr_5859_37`, `HPOBSurr_5859_3902`, `HPOBSurr_5859_9977`, `HPOBSurr_5859_9983`, `HPOBSurr_5889_31`, `HPOBSurr_5889_3493`, `HPOBSurr_5889_3918`, `HPOBSurr_5889_3950`, `HPOBSurr_5889_49`, `HPOBSurr_5889_9971`, `HPOBSurr_5891_3492`, `HPOBSurr_5891_3891`, `HPOBSurr_5891_3899`, `HPOBSurr_5891_6566`, `HPOBSurr_5891_9889`, `HPOBSurr_5891_9980`, `HPOBSurr_5906_3889`, `HPOBSurr_5906_3896`, `HPOBSurr_5906_3918`, `HPOBSurr_5906_9970`, `HPOBSurr_5906_9971`, `HPOBSurr_5906_9977`, `HPOBSurr_5965_10101`, `HPOBSurr_5965_145836`, `HPOBSurr_5965_3903`, `HPOBSurr_5965_49`, `HPOBSurr_5965_9889`, `HPOBSurr_5965_9914`, `HPOBSurr_5965_9946`, `HPOBSurr_5971_10093`, `HPOBSurr_5971_34536`, `HPOBSurr_5971_3954`, `HPOBSurr_5971_43`, `HPOBSurr_5971_6566`, `HPOBSurr_5971_9970`, `HPOBSurr_6767_145804`, `HPOBSurr_6767_146064`, `HPOBSurr_6767_146065`, `HPOBSurr_6767_31`, `HPOBSurr_6767_9914`, `HPOBSurr_6767_9967`, `HPOBSurr_6794_10101`, `HPOBSurr_6794_145804`, `HPOBSurr_6794_146065`, `HPOBSurr_6794_3`, `HPOBSurr_6794_31`, `HPOBSurr_6794_9914`, `HPOBSurr_7607_145976`, `HPOBSurr_7607_3896`, `HPOBSurr_7607_3903`, `HPOBSurr_7607_3913`, `HPOBSurr_7607_9946`, `HPOBSurr_7607_9967`, `HPOBSurr_7609_125923`, `HPOBSurr_7609_145853`, `HPOBSurr_7609_145854`, `HPOBSurr_7609_145878`, `HPOBSurr_7609_34537`, `HPOBSurr_7609_3903`, `HPOBSurr_7609_9967`, `HPOB_4796_23`, `HPOB_4796_3549`, `HPOB_4796_3918`, `HPOB_4796_9903`, `HPOB_4796_9906`, `HPOB_4796_9946`, `HPOB_5527_10101`, `HPOB_5527_145804`, `HPOB_5527_146064`, `HPOB_5527_146065`, `HPOB_5527_31`, `HPOB_5527_9914`, `HPOB_5636_10101`, `HPOB_5636_145804`, `HPOB_5636_146064`, `HPOB_5636_146065`, `HPOB_5636_31`, `HPOB_5636_9914`, `HPOB_5859_125923`, `HPOB_5859_31`, `HPOB_5859_37`, `HPOB_5859_3902`, `HPOB_5859_9977`, `HPOB_5859_9983`, `HPOB_5889_31`, `HPOB_5889_3493`, `HPOB_5889_3918`, `HPOB_5889_3950`, `HPOB_5889_49`, `HPOB_5889_9971`, `HPOB_5891_3492`, `HPOB_5891_3891`, `HPOB_5891_3899`, `HPOB_5891_6566`, `HPOB_5891_9889`, `HPOB_5891_9980`, `HPOB_5906_3889`, `HPOB_5906_3896`, `HPOB_5906_3918`, `HPOB_5906_9970`, `HPOB_5906_9971`, `HPOB_5906_9977`, `HPOB_5965_10101`, `HPOB_5965_145836`, `HPOB_5965_3903`, `HPOB_5965_49`, `HPOB_5965_9889`, `HPOB_5965_9914`, `HPOB_5965_9946`, `HPOB_5971_10093`, `HPOB_5971_34536`, `HPOB_5971_3954`, `HPOB_5971_43`, `HPOB_5971_6566`, `HPOB_5971_9970`, `HPOB_6767_145804`, `HPOB_6767_146064`, `HPOB_6767_146065`, `HPOB_6767_31`, `HPOB_6767_9914`, `HPOB_6767_9967`, `HPOB_6794_10101`, `HPOB_6794_145804`, `HPOB_6794_146065`, `HPOB_6794_3`, `HPOB_6794_31`, `HPOB_6794_9914`, `HPOB_7607_145976`, `HPOB_7607_3896`, `HPOB_7607_3903`, `HPOB_7607_3913`, `HPOB_7607_9946`, `HPOB_7607_9967`, `HPOB_7609_125923`, `HPOB_7609_145853`, `HPOB_7609_145854`, `HPOB_7609_145878`, `HPOB_7609_34537`, `HPOB_7609_3903`, `HPOB_7609_9967`, `HartmannCat`, `LCBenchAPSFailure`, `LCBenchAdult`, `LCBenchAirlines`, `LCBenchAlbert`, `LCBenchAmazonEmployeeAccess`, `LCBenchAustralian`, `LCBenchBankMarketing`, `LCBenchBloodTransfusionServiceCenter`, `LCBenchCar`, `LCBenchChristine`, `LCBenchCnae9`, `LCBenchConnect4`, `LCBenchCovertype`, `LCBenchCreditG`, `LCBenchDionis`, `LCBenchFabert`, `LCBenchFashionMNIST`, `LCBenchHelena`, `LCBenchHiggs`, `LCBenchJannis`, `LCBenchJasmine`, `LCBenchJungleChess2pcsRawEndgameComplete`, `LCBenchKDDCup09Appetency`, `LCBenchKc1`, `LCBenchKrVsKp`, `LCBenchMfeatFactors`, `LCBenchMiniBooNE`, `LCBenchNomao`, `LCBenchNumerai286`, `LCBenchPhoneme`, `LCBenchSegment`, `LCBenchShuttle`, `LCBenchSylvine`, `LCBenchVehicle`, `LCBenchVolkert`, `MixedAckley`, `Rosenbrock5Mixed`, `ShekelMixed`, `StyblinskiTangMixed`

### Single-objective, constrained, continuous (71)

`list_problems(num_objectives=1, constrained=True, input_type='continuous')`

`CEC2020_p1`, `CEC2020_p10`, `CEC2020_p11`, `CEC2020_p12`, `CEC2020_p13`, `CEC2020_p14`, `CEC2020_p15`, `CEC2020_p16`, `CEC2020_p17`, `CEC2020_p18`, `CEC2020_p19`, `CEC2020_p2`, `CEC2020_p20`, `CEC2020_p21`, `CEC2020_p22`, `CEC2020_p23`, `CEC2020_p24`, `CEC2020_p25`, `CEC2020_p26`, `CEC2020_p27`, `CEC2020_p28`, `CEC2020_p29`, `CEC2020_p3`, `CEC2020_p30`, `CEC2020_p32`, `CEC2020_p33`, `CEC2020_p34`, `CEC2020_p35`, `CEC2020_p36`, `CEC2020_p37`, `CEC2020_p38`, `CEC2020_p39`, `CEC2020_p4`, `CEC2020_p40`, `CEC2020_p41`, `CEC2020_p42`, `CEC2020_p43`, `CEC2020_p44`, `CEC2020_p45`, `CEC2020_p46`, `CEC2020_p47`, `CEC2020_p48`, `CEC2020_p49`, `CEC2020_p5`, `CEC2020_p50`, `CEC2020_p51`, `CEC2020_p52`, `CEC2020_p53`, `CEC2020_p54`, `CEC2020_p55`, `CEC2020_p56`, `CEC2020_p57`, `CEC2020_p6`, `CEC2020_p7`, `CEC2020_p8`, `CEC2020_p9`, `CantileverBeam`, `CompressionSpring`, `HeatExchanger`, `MOPTA08Car`, `PEARL`, `SatelliteDesign`, `Sellar`, `ThreeTruss`, `Truss10D`, `Truss120D`, `Truss200D`, `Truss25D`, `Truss72D_FourForces`, `Truss72D_SingleForce`, `WeldedBeamSO`

### Single-objective, constrained, mixed-variable (7)

`list_problems(num_objectives=1, constrained=True, input_type='mixed')`

`Car`, `HelicalSpring`, `PressureVessel`, `ReinforcedConcreteBeam`, `SpeedReducer`, `SteppedCantileverBeam`, `WeldedBeamCategorical`

### Single-objective, unconstrained, discrete (10)

`list_problems(num_objectives=1, constrained=False, input_type='discrete')`

`AckleyCat`, `CrossedBarrel`, `GearTrain`, `HOIP`, `MaxSAT`, `NASBench201`, `PestControl`, `RastriginCat`, `SchwefelCat`, `StyblinskiTangCat`

### Single-objective, constrained, discrete (0)

`list_problems(num_objectives=1, constrained=True, input_type='discrete')`

_(none)_

### Multi-objective, unconstrained, continuous (20)

`list_problems(constrained=False, input_type='continuous')  # >=2 objectives`

`BotorchCarSideImpact`, `Penicillin`, `PowerElectronics`, `RE21`, `RE22`, `RE23`, `RE24`, `RE25`, `RE31`, `RE32`, `RE33`, `RE34`, `RE35`, `RE36`, `RE37`, `RE41`, `RE42`, `RE61`, `RE91`, `VehicleSafety`

### Multi-objective, constrained, continuous (36)

`list_problems(constrained=True, input_type='continuous')  # >=2 objectives`

`CRE21`, `CRE22`, `CRE23`, `CRE24`, `CRE25`, `CRE31`, `CRE32`, `CRE51`, `CS1`, `CS2`, `CS3`, `CS4`, `CT1`, `CT2`, `CT3`, `CT4`, `CTS1`, `CTS2`, `CTS3`, `CTS4`, `CTSE1`, `CTSE2`, `CTSE3`, `CTSE4`, `CTSEI1`, `CTSEI2`, `CTSEI3`, `CTSEI4`, `CarSideImpact`, `DiscBrake`, `Mazda`, `Mazda_SCA`, `TwoBarTruss`, `WaterProblem`, `WaterResources`, `WeldedBeam`

### Multi-objective, mixed-variable (any constraints) (0)

`list_problems(input_type='mixed')  # >=2 objectives`

_(none)_

<!-- CATEGORIES:END -->
