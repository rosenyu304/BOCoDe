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
- **Variables** — continuous, discrete, or mixed.
- **Scalable** — whether the dimension can be chosen by the user.
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
problem count (currently **164**).

<!-- TABLE:START -->

| # | Problem | Application | Suite | Dim | #Obj | #Constr | Variables | Scalable | Convex | NP-hard |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TSP_100Cities | Combinatorial | TSP / NEORL | 100 | 1 | 0 | discrete | no | no | yes |
| 2 | TSP_51Cities | Combinatorial | TSP / NEORL | 51 | 1 | 0 | discrete | no | no | yes |
| 3 | PD4CartPole | Control | Classic control | 4 | 1 | 0 | continuous | no | unknown | unknown |
| 4 | PID4Acrobot | Control | Classic control | 3 | 1 | 0 | continuous | no | unknown | unknown |
| 5 | AntPolicySearchProblem | Control | MuJoCo control | 840 | 1 | 0 | continuous | no | unknown | unknown |
| 6 | AntProblem | Control | MuJoCo control | 8 | 1 | 0 | continuous | no | unknown | unknown |
| 7 | HalfCheetahPolicySearchProblem | Control | MuJoCo control | 102 | 1 | 0 | continuous | no | unknown | unknown |
| 8 | HalfCheetahProblem | Control | MuJoCo control | 6 | 1 | 0 | continuous | no | unknown | unknown |
| 9 | HopperPolicySearchProblem | Control | MuJoCo control | 33 | 1 | 0 | continuous | no | unknown | unknown |
| 10 | HopperProblem | Control | MuJoCo control | 3 | 1 | 0 | continuous | no | unknown | unknown |
| 11 | HumanoidProblem | Control | MuJoCo control | 17 | 1 | 0 | continuous | no | unknown | unknown |
| 12 | HumanoidStandupProblem | Control | MuJoCo control | 17 | 1 | 0 | continuous | no | unknown | unknown |
| 13 | InvertedDoublePendulumProblem | Control | MuJoCo control | 1 | 1 | 0 | continuous | no | unknown | unknown |
| 14 | InvertedPendulumProblem | Control | MuJoCo control | 1 | 1 | 0 | continuous | no | unknown | unknown |
| 15 | PusherProblem | Control | MuJoCo control | 7 | 1 | 0 | continuous | no | unknown | unknown |
| 16 | ReacherProblem | Control | MuJoCo control | 2 | 1 | 0 | continuous | no | unknown | unknown |
| 17 | SwimmerPolicySearchProblem | Control | MuJoCo control | 16 | 1 | 0 | continuous | no | unknown | unknown |
| 18 | SwimmerProblem | Control | MuJoCo control | 2 | 1 | 0 | continuous | no | unknown | unknown |
| 19 | Walker2DPolicySearchProblem | Control | MuJoCo control | 102 | 1 | 0 | continuous | no | unknown | unknown |
| 20 | Walker2DProblem | Control | MuJoCo control | 6 | 1 | 0 | continuous | no | unknown | unknown |
| 21 | CEC2020_p1 | Engineering | CEC2020 RW-Constrained | 9 | 1 | 8 | continuous | no | unknown | unknown |
| 22 | CEC2020_p10 | Engineering | CEC2020 RW-Constrained | 3 | 1 | 3 | continuous | no | unknown | unknown |
| 23 | CEC2020_p11 | Engineering | CEC2020 RW-Constrained | 7 | 1 | 8 | continuous | no | unknown | unknown |
| 24 | CEC2020_p12 | Engineering | CEC2020 RW-Constrained | 7 | 1 | 9 | continuous | no | unknown | unknown |
| 25 | CEC2020_p13 | Engineering | CEC2020 RW-Constrained | 5 | 1 | 3 | continuous | no | unknown | unknown |
| 26 | CEC2020_p14 | Engineering | CEC2020 RW-Constrained | 10 | 1 | 10 | continuous | no | unknown | unknown |
| 27 | CEC2020_p15 | Engineering | CEC2020 RW-Constrained | 7 | 1 | 11 | continuous | no | unknown | unknown |
| 28 | CEC2020_p16 | Engineering | CEC2020 RW-Constrained | 14 | 1 | 15 | continuous | no | unknown | unknown |
| 29 | CEC2020_p17 | Engineering | CEC2020 RW-Constrained | 3 | 1 | 4 | continuous | no | unknown | unknown |
| 30 | CEC2020_p18 | Engineering | CEC2020 RW-Constrained | 4 | 1 | 4 | continuous | no | unknown | unknown |
| 31 | CEC2020_p19 | Engineering | CEC2020 RW-Constrained | 4 | 1 | 5 | continuous | no | unknown | unknown |
| 32 | CEC2020_p2 | Engineering | CEC2020 RW-Constrained | 11 | 1 | 9 | continuous | no | unknown | unknown |
| 33 | CEC2020_p20 | Engineering | CEC2020 RW-Constrained | 2 | 1 | 3 | continuous | no | unknown | unknown |
| 34 | CEC2020_p21 | Engineering | CEC2020 RW-Constrained | 5 | 1 | 8 | continuous | no | unknown | unknown |
| 35 | CEC2020_p22 | Engineering | CEC2020 RW-Constrained | 9 | 1 | 11 | continuous | no | unknown | unknown |
| 36 | CEC2020_p23 | Engineering | CEC2020 RW-Constrained | 5 | 1 | 11 | continuous | no | unknown | unknown |
| 37 | CEC2020_p24 | Engineering | CEC2020 RW-Constrained | 7 | 1 | 7 | continuous | no | unknown | unknown |
| 38 | CEC2020_p25 | Engineering | CEC2020 RW-Constrained | 7 | 1 | 7 | continuous | no | unknown | unknown |
| 39 | CEC2020_p26 | Engineering | CEC2020 RW-Constrained | 22 | 1 | 87 | continuous | no | unknown | unknown |
| 40 | CEC2020_p27 | Engineering | CEC2020 RW-Constrained | 10 | 1 | 3 | continuous | no | unknown | unknown |
| 41 | CEC2020_p28 | Engineering | CEC2020 RW-Constrained | 10 | 1 | 9 | continuous | no | unknown | unknown |
| 42 | CEC2020_p29 | Engineering | CEC2020 RW-Constrained | 4 | 1 | 1 | continuous | no | unknown | unknown |
| 43 | CEC2020_p3 | Engineering | CEC2020 RW-Constrained | 7 | 1 | 14 | continuous | no | unknown | unknown |
| 44 | CEC2020_p30 | Engineering | CEC2020 RW-Constrained | 3 | 1 | 8 | continuous | no | unknown | unknown |
| 45 | CEC2020_p31 | Engineering | CEC2020 RW-Constrained | 4 | 1 | 0 | continuous | no | unknown | unknown |
| 46 | CEC2020_p32 | Engineering | CEC2020 RW-Constrained | 5 | 1 | 6 | continuous | no | unknown | unknown |
| 47 | CEC2020_p33 | Engineering | CEC2020 RW-Constrained | 30 | 1 | 30 | continuous | no | unknown | unknown |
| 48 | CEC2020_p34 | Engineering | CEC2020 RW-Constrained | 118 | 1 | 108 | continuous | no | unknown | unknown |
| 49 | CEC2020_p35 | Engineering | CEC2020 RW-Constrained | 153 | 1 | 148 | continuous | no | unknown | unknown |
| 50 | CEC2020_p36 | Engineering | CEC2020 RW-Constrained | 158 | 1 | 148 | continuous | no | unknown | unknown |
| 51 | CEC2020_p37 | Engineering | CEC2020 RW-Constrained | 126 | 1 | 116 | continuous | no | unknown | unknown |
| 52 | CEC2020_p38 | Engineering | CEC2020 RW-Constrained | 126 | 1 | 116 | continuous | no | unknown | unknown |
| 53 | CEC2020_p39 | Engineering | CEC2020 RW-Constrained | 126 | 1 | 116 | continuous | no | unknown | unknown |
| 54 | CEC2020_p4 | Engineering | CEC2020 RW-Constrained | 6 | 1 | 5 | continuous | no | unknown | unknown |
| 55 | CEC2020_p40 | Engineering | CEC2020 RW-Constrained | 76 | 1 | 76 | continuous | no | unknown | unknown |
| 56 | CEC2020_p41 | Engineering | CEC2020 RW-Constrained | 74 | 1 | 74 | continuous | no | unknown | unknown |
| 57 | CEC2020_p42 | Engineering | CEC2020 RW-Constrained | 86 | 1 | 76 | continuous | no | unknown | unknown |
| 58 | CEC2020_p43 | Engineering | CEC2020 RW-Constrained | 86 | 1 | 76 | continuous | no | unknown | unknown |
| 59 | CEC2020_p44 | Engineering | CEC2020 RW-Constrained | 30 | 1 | 105 | continuous | no | unknown | unknown |
| 60 | CEC2020_p45 | Engineering | CEC2020 RW-Constrained | 25 | 1 | 25 | continuous | no | unknown | unknown |
| 61 | CEC2020_p46 | Engineering | CEC2020 RW-Constrained | 25 | 1 | 25 | continuous | no | unknown | unknown |
| 62 | CEC2020_p47 | Engineering | CEC2020 RW-Constrained | 25 | 1 | 25 | continuous | no | unknown | unknown |
| 63 | CEC2020_p48 | Engineering | CEC2020 RW-Constrained | 30 | 1 | 30 | continuous | no | unknown | unknown |
| 64 | CEC2020_p49 | Engineering | CEC2020 RW-Constrained | 30 | 1 | 30 | continuous | no | unknown | unknown |
| 65 | CEC2020_p5 | Engineering | CEC2020 RW-Constrained | 9 | 1 | 6 | continuous | no | unknown | unknown |
| 66 | CEC2020_p50 | Engineering | CEC2020 RW-Constrained | 30 | 1 | 30 | continuous | no | unknown | unknown |
| 67 | CEC2020_p51 | Engineering | CEC2020 RW-Constrained | 59 | 1 | 15 | continuous | no | unknown | unknown |
| 68 | CEC2020_p52 | Engineering | CEC2020 RW-Constrained | 59 | 1 | 15 | continuous | no | unknown | unknown |
| 69 | CEC2020_p53 | Engineering | CEC2020 RW-Constrained | 59 | 1 | 15 | continuous | no | unknown | unknown |
| 70 | CEC2020_p54 | Engineering | CEC2020 RW-Constrained | 59 | 1 | 15 | continuous | no | unknown | unknown |
| 71 | CEC2020_p55 | Engineering | CEC2020 RW-Constrained | 64 | 1 | 6 | continuous | no | unknown | unknown |
| 72 | CEC2020_p56 | Engineering | CEC2020 RW-Constrained | 64 | 1 | 6 | continuous | no | unknown | unknown |
| 73 | CEC2020_p57 | Engineering | CEC2020 RW-Constrained | 64 | 1 | 6 | continuous | no | unknown | unknown |
| 74 | CEC2020_p6 | Engineering | CEC2020 RW-Constrained | 38 | 1 | 32 | continuous | no | unknown | unknown |
| 75 | CEC2020_p7 | Engineering | CEC2020 RW-Constrained | 48 | 1 | 38 | continuous | no | unknown | unknown |
| 76 | CEC2020_p8 | Engineering | CEC2020 RW-Constrained | 2 | 1 | 2 | continuous | no | unknown | unknown |
| 77 | CEC2020_p9 | Engineering | CEC2020 RW-Constrained | 3 | 1 | 2 | continuous | no | unknown | unknown |
| 78 | CRE21 | Engineering | CRE (Tanabe-Ishibuchi) | 3 | 2 | 3 | continuous | no | unknown | unknown |
| 79 | CRE22 | Engineering | CRE (Tanabe-Ishibuchi) | 4 | 2 | 4 | continuous | no | unknown | unknown |
| 80 | CRE23 | Engineering | CRE (Tanabe-Ishibuchi) | 4 | 2 | 4 | continuous | no | unknown | unknown |
| 81 | CRE24 | Engineering | CRE (Tanabe-Ishibuchi) | 7 | 2 | 11 | continuous | no | unknown | unknown |
| 82 | CRE25 | Engineering | CRE (Tanabe-Ishibuchi) | 4 | 2 | 1 | continuous | no | unknown | unknown |
| 83 | CRE31 | Engineering | CRE (Tanabe-Ishibuchi) | 7 | 3 | 10 | continuous | no | unknown | unknown |
| 84 | CRE32 | Engineering | CRE (Tanabe-Ishibuchi) | 6 | 3 | 9 | continuous | no | unknown | unknown |
| 85 | CRE51 | Engineering | CRE (Tanabe-Ishibuchi) | 3 | 5 | 7 | continuous | no | unknown | unknown |
| 86 | BotorchCarSideImpact | Engineering | Engineering (standalone) | 7 | 4 | 0 | continuous | no | unknown | unknown |
| 87 | CantileverBeam | Engineering | Engineering (standalone) | 10 | 1 | 11 | continuous | no | unknown | unknown |
| 88 | Car | Engineering | Engineering (standalone) | 11 | 1 | 10 | continuous | no | unknown | unknown |
| 89 | CarSideImpact | Engineering | Engineering (standalone) | 7 | 3 | 10 | continuous | no | unknown | unknown |
| 90 | CompressionSpring | Engineering | Engineering (standalone) | 3 | 1 | 4 | continuous | no | unknown | unknown |
| 91 | DiscBrake | Engineering | Engineering (standalone) | 4 | 2 | 4 | continuous | no | unknown | unknown |
| 92 | EulerBernoulliBeamBending | Engineering | Engineering (standalone) | 3 | 1 | 0 | continuous | no | unknown | unknown |
| 93 | GearTrain | Engineering | Engineering (standalone) | 4 | 1 | 0 | mixed | no | unknown | unknown |
| 94 | HeatExchanger | Engineering | Engineering (standalone) | 8 | 1 | 6 | continuous | no | unknown | unknown |
| 95 | MOPTA08Car | Engineering | Engineering (standalone) | 124 | 1 | 68 | continuous | no | no | unknown |
| 96 | Mazda | Engineering | Engineering (standalone) | 222 | 5 | 54 | continuous | no | unknown | unknown |
| 97 | Mazda_SCA | Engineering | Engineering (standalone) | 148 | 4 | 36 | continuous | no | unknown | unknown |
| 98 | Penicillin | Engineering | Engineering (standalone) | 7 | 3 | 0 | continuous | no | unknown | unknown |
| 99 | PressureVessel | Engineering | Engineering (standalone) | 4 | 1 | 4 | continuous | no | unknown | unknown |
| 100 | QPowerModel | Engineering | Engineering (standalone) | 8 | 1 | 0 | continuous | no | unknown | unknown |
| 101 | ReactivityModel | Engineering | Engineering (standalone) | 8 | 1 | 0 | continuous | no | unknown | unknown |
| 102 | ReinforcedConcreteBeam | Engineering | Engineering (standalone) | 3 | 1 | 2 | continuous | no | unknown | unknown |
| 103 | RobotPush | Engineering | Engineering (standalone) | 14 | 1 | 0 | continuous | no | no | unknown |
| 104 | Rover | Engineering | Engineering (standalone) | 100 | 1 | 0 | continuous | no | unknown | unknown |
| 105 | SpeedReducer | Engineering | Engineering (standalone) | 7 | 1 | 9 | continuous | no | unknown | unknown |
| 106 | ThreeTruss | Engineering | Engineering (standalone) | 2 | 1 | 3 | continuous | no | unknown | unknown |
| 107 | Truss10D | Engineering | Engineering (standalone) | 10 | 1 | 14 | continuous | no | unknown | unknown |
| 108 | Truss120D | Engineering | Engineering (standalone) | 120 | 1 | 121 | continuous | no | unknown | unknown |
| 109 | Truss200D | Engineering | Engineering (standalone) | 200 | 1 | 200 | continuous | no | unknown | unknown |
| 110 | Truss25D | Engineering | Engineering (standalone) | 25 | 1 | 31 | continuous | no | unknown | unknown |
| 111 | Truss72D_FourForces | Engineering | Engineering (standalone) | 72 | 1 | 88 | continuous | no | unknown | unknown |
| 112 | Truss72D_SingleForce | Engineering | Engineering (standalone) | 72 | 1 | 88 | continuous | no | unknown | unknown |
| 113 | TwoBarTruss | Engineering | Engineering (standalone) | 2 | 2 | 5 | continuous | no | unknown | unknown |
| 114 | VehicleSafety | Engineering | Engineering (standalone) | 5 | 3 | 0 | continuous | no | unknown | unknown |
| 115 | WaterProblem | Engineering | Engineering (standalone) | 3 | 5 | 7 | continuous | no | unknown | unknown |
| 116 | WaterResources | Engineering | Engineering (standalone) | 3 | 5 | 7 | continuous | no | unknown | unknown |
| 117 | WeldedBeam | Engineering | Engineering (standalone) | 4 | 2 | 4 | continuous | no | unknown | unknown |
| 118 | CS1 | Engineering | MODAct (actuator design) | 20 | 2 | 7 | continuous | no | unknown | unknown |
| 119 | CS2 | Engineering | MODAct (actuator design) | 20 | 2 | 8 | continuous | no | unknown | unknown |
| 120 | CS3 | Engineering | MODAct (actuator design) | 20 | 2 | 10 | continuous | no | unknown | unknown |
| 121 | CS4 | Engineering | MODAct (actuator design) | 20 | 2 | 9 | continuous | no | unknown | unknown |
| 122 | CT1 | Engineering | MODAct (actuator design) | 20 | 2 | 7 | continuous | no | unknown | unknown |
| 123 | CT2 | Engineering | MODAct (actuator design) | 20 | 2 | 8 | continuous | no | unknown | unknown |
| 124 | CT3 | Engineering | MODAct (actuator design) | 20 | 2 | 10 | continuous | no | unknown | unknown |
| 125 | CT4 | Engineering | MODAct (actuator design) | 20 | 2 | 9 | continuous | no | unknown | unknown |
| 126 | CTS1 | Engineering | MODAct (actuator design) | 20 | 3 | 7 | continuous | no | unknown | unknown |
| 127 | CTS2 | Engineering | MODAct (actuator design) | 20 | 3 | 8 | continuous | no | unknown | unknown |
| 128 | CTS3 | Engineering | MODAct (actuator design) | 20 | 3 | 10 | continuous | no | unknown | unknown |
| 129 | CTS4 | Engineering | MODAct (actuator design) | 20 | 3 | 9 | continuous | no | unknown | unknown |
| 130 | CTSE1 | Engineering | MODAct (actuator design) | 20 | 4 | 7 | continuous | no | unknown | unknown |
| 131 | CTSE2 | Engineering | MODAct (actuator design) | 20 | 4 | 8 | continuous | no | unknown | unknown |
| 132 | CTSE3 | Engineering | MODAct (actuator design) | 20 | 4 | 10 | continuous | no | unknown | unknown |
| 133 | CTSE4 | Engineering | MODAct (actuator design) | 20 | 4 | 9 | continuous | no | unknown | unknown |
| 134 | CTSEI1 | Engineering | MODAct (actuator design) | 20 | 5 | 7 | continuous | no | unknown | unknown |
| 135 | CTSEI2 | Engineering | MODAct (actuator design) | 20 | 5 | 8 | continuous | no | unknown | unknown |
| 136 | CTSEI3 | Engineering | MODAct (actuator design) | 20 | 5 | 10 | continuous | no | unknown | unknown |
| 137 | CTSEI4 | Engineering | MODAct (actuator design) | 20 | 5 | 9 | continuous | no | unknown | unknown |
| 138 | RE21 | Engineering | RE (Tanabe-Ishibuchi) | 4 | 2 | 0 | continuous | no | unknown | unknown |
| 139 | RE22 | Engineering | RE (Tanabe-Ishibuchi) | 3 | 2 | 0 | continuous | no | unknown | unknown |
| 140 | RE23 | Engineering | RE (Tanabe-Ishibuchi) | 4 | 2 | 0 | continuous | no | unknown | unknown |
| 141 | RE24 | Engineering | RE (Tanabe-Ishibuchi) | 2 | 2 | 0 | continuous | no | unknown | unknown |
| 142 | RE25 | Engineering | RE (Tanabe-Ishibuchi) | 3 | 2 | 0 | continuous | no | unknown | unknown |
| 143 | RE31 | Engineering | RE (Tanabe-Ishibuchi) | 3 | 3 | 0 | continuous | no | unknown | unknown |
| 144 | RE32 | Engineering | RE (Tanabe-Ishibuchi) | 4 | 3 | 0 | continuous | no | unknown | unknown |
| 145 | RE33 | Engineering | RE (Tanabe-Ishibuchi) | 4 | 3 | 0 | continuous | no | unknown | unknown |
| 146 | RE34 | Engineering | RE (Tanabe-Ishibuchi) | 5 | 3 | 0 | continuous | no | unknown | unknown |
| 147 | RE35 | Engineering | RE (Tanabe-Ishibuchi) | 7 | 3 | 0 | continuous | no | unknown | unknown |
| 148 | RE36 | Engineering | RE (Tanabe-Ishibuchi) | 4 | 3 | 0 | continuous | no | unknown | unknown |
| 149 | RE37 | Engineering | RE (Tanabe-Ishibuchi) | 4 | 3 | 0 | continuous | no | unknown | unknown |
| 150 | RE41 | Engineering | RE (Tanabe-Ishibuchi) | 7 | 4 | 0 | continuous | no | unknown | unknown |
| 151 | RE42 | Engineering | RE (Tanabe-Ishibuchi) | 6 | 4 | 0 | continuous | no | unknown | unknown |
| 152 | RE61 | Engineering | RE (Tanabe-Ishibuchi) | 3 | 6 | 0 | continuous | no | unknown | unknown |
| 153 | RE91 | Engineering | RE (Tanabe-Ishibuchi) | 7 | 9 | 0 | continuous | no | unknown | unknown |
| 154 | LassoBreastCancer | Hyperparameter Optimization | HPO (LassoBench / GP-HDBO) | 9 | 1 | 0 | continuous | no | unknown | unknown |
| 155 | LassoDNA | Hyperparameter Optimization | HPO (LassoBench / GP-HDBO) | 180 | 1 | 0 | continuous | no | unknown | unknown |
| 156 | LassoDiabetes | Hyperparameter Optimization | HPO (LassoBench / GP-HDBO) | 8 | 1 | 0 | continuous | no | unknown | unknown |
| 157 | LassoLeukemia | Hyperparameter Optimization | HPO (LassoBench / GP-HDBO) | 7129 | 1 | 0 | continuous | no | unknown | unknown |
| 158 | LassoRCV1 | Hyperparameter Optimization | HPO (LassoBench / GP-HDBO) | 47236 | 1 | 0 | continuous | no | unknown | unknown |
| 159 | SVM | Hyperparameter Optimization | HPO (LassoBench / GP-HDBO) | 388 | 1 | 0 | continuous | no | no | unknown |
| 160 | AgNP | Materials | PV-Lab materials | 5 | 1 | 0 | discrete | no | unknown | unknown |
| 161 | AutoAM | Materials | PV-Lab materials | 4 | 1 | 0 | discrete | no | unknown | unknown |
| 162 | CrossedBarrel | Materials | PV-Lab materials | 4 | 1 | 0 | discrete | no | unknown | unknown |
| 163 | P3HT | Materials | PV-Lab materials | 5 | 1 | 0 | discrete | no | unknown | unknown |
| 164 | Perovskite | Materials | PV-Lab materials | 3 | 1 | 0 | discrete | no | unknown | unknown |

<!-- TABLE:END -->
