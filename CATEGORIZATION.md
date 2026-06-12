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

The CEC2020 real-world constrained suite currently declares `num_constraints`
counting only equality constraints for several inequality-only problems; the
table therefore understates the constraint count for those problems. This is
tracked in `docs/AUDIT_findings_2026_06.md` and will be corrected in the
constraint-semantics cleanup.

## Problem table

The leading `#` column numbers the problems 1…N, so the last row shows the total
problem count (currently **164**).

<!-- TABLE:START -->

| # | Problem | Application | Dim | #Obj | #Constr | Variables | Scalable | Convex | NP-hard |
|---|---|---|---|---|---|---|---|---|---|
| 1 | TSP_100Cities | Combinatorial | 100 | 1 | 0 | discrete | no | no | yes |
| 2 | TSP_51Cities | Combinatorial | 51 | 1 | 0 | discrete | no | no | yes |
| 3 | AntPolicySearchProblem | Control | 840 | 1 | 0 | continuous | no | unknown | unknown |
| 4 | AntProblem | Control | 8 | 1 | 0 | continuous | no | unknown | unknown |
| 5 | HalfCheetahPolicySearchProblem | Control | 102 | 1 | 0 | continuous | no | unknown | unknown |
| 6 | HalfCheetahProblem | Control | 6 | 1 | 0 | continuous | no | unknown | unknown |
| 7 | HopperPolicySearchProblem | Control | 33 | 1 | 0 | continuous | no | unknown | unknown |
| 8 | HopperProblem | Control | 3 | 1 | 0 | continuous | no | unknown | unknown |
| 9 | HumanoidProblem | Control | 17 | 1 | 0 | continuous | no | unknown | unknown |
| 10 | HumanoidStandupProblem | Control | 17 | 1 | 0 | continuous | no | unknown | unknown |
| 11 | InvertedDoublePendulumProblem | Control | 1 | 1 | 0 | continuous | no | unknown | unknown |
| 12 | InvertedPendulumProblem | Control | 1 | 1 | 0 | continuous | no | unknown | unknown |
| 13 | PD4CartPole | Control | 4 | 1 | 0 | continuous | no | unknown | unknown |
| 14 | PID4Acrobot | Control | 3 | 1 | 0 | continuous | no | unknown | unknown |
| 15 | PusherProblem | Control | 7 | 1 | 0 | continuous | no | unknown | unknown |
| 16 | ReacherProblem | Control | 2 | 1 | 0 | continuous | no | unknown | unknown |
| 17 | SwimmerPolicySearchProblem | Control | 16 | 1 | 0 | continuous | no | unknown | unknown |
| 18 | SwimmerProblem | Control | 2 | 1 | 0 | continuous | no | unknown | unknown |
| 19 | Walker2DPolicySearchProblem | Control | 102 | 1 | 0 | continuous | no | unknown | unknown |
| 20 | Walker2DProblem | Control | 6 | 1 | 0 | continuous | no | unknown | unknown |
| 21 | BotorchCarSideImpact | Engineering | 7 | 4 | 0 | continuous | no | unknown | unknown |
| 22 | CEC2020_p1 | Engineering | 9 | 1 | 8 | continuous | no | unknown | unknown |
| 23 | CEC2020_p10 | Engineering | 3 | 1 | 0 | continuous | no | unknown | unknown |
| 24 | CEC2020_p11 | Engineering | 7 | 1 | 4 | continuous | no | unknown | unknown |
| 25 | CEC2020_p12 | Engineering | 7 | 1 | 0 | continuous | no | unknown | unknown |
| 26 | CEC2020_p13 | Engineering | 5 | 1 | 0 | continuous | no | unknown | unknown |
| 27 | CEC2020_p14 | Engineering | 10 | 1 | 0 | continuous | no | unknown | unknown |
| 28 | CEC2020_p15 | Engineering | 7 | 1 | 0 | continuous | no | unknown | unknown |
| 29 | CEC2020_p16 | Engineering | 14 | 1 | 0 | continuous | no | unknown | unknown |
| 30 | CEC2020_p17 | Engineering | 3 | 1 | 0 | continuous | no | unknown | unknown |
| 31 | CEC2020_p18 | Engineering | 4 | 1 | 0 | continuous | no | unknown | unknown |
| 32 | CEC2020_p19 | Engineering | 4 | 1 | 0 | continuous | no | unknown | unknown |
| 33 | CEC2020_p2 | Engineering | 11 | 1 | 9 | continuous | no | unknown | unknown |
| 34 | CEC2020_p20 | Engineering | 2 | 1 | 0 | continuous | no | unknown | unknown |
| 35 | CEC2020_p21 | Engineering | 5 | 1 | 0 | continuous | no | unknown | unknown |
| 36 | CEC2020_p22 | Engineering | 9 | 1 | 1 | continuous | no | unknown | unknown |
| 37 | CEC2020_p23 | Engineering | 5 | 1 | 3 | continuous | no | unknown | unknown |
| 38 | CEC2020_p24 | Engineering | 7 | 1 | 0 | continuous | no | unknown | unknown |
| 39 | CEC2020_p25 | Engineering | 7 | 1 | 0 | continuous | no | unknown | unknown |
| 40 | CEC2020_p26 | Engineering | 22 | 1 | 0 | continuous | no | unknown | unknown |
| 41 | CEC2020_p27 | Engineering | 10 | 1 | 0 | continuous | no | unknown | unknown |
| 42 | CEC2020_p28 | Engineering | 10 | 1 | 0 | continuous | no | unknown | unknown |
| 43 | CEC2020_p29 | Engineering | 4 | 1 | 0 | continuous | no | unknown | unknown |
| 44 | CEC2020_p3 | Engineering | 7 | 1 | 0 | continuous | no | unknown | unknown |
| 45 | CEC2020_p30 | Engineering | 3 | 1 | 0 | continuous | no | unknown | unknown |
| 46 | CEC2020_p31 | Engineering | 4 | 1 | 0 | continuous | no | unknown | unknown |
| 47 | CEC2020_p32 | Engineering | 5 | 1 | 0 | continuous | no | unknown | unknown |
| 48 | CEC2020_p33 | Engineering | 30 | 1 | 0 | continuous | no | unknown | unknown |
| 49 | CEC2020_p34 | Engineering | 118 | 1 | 108 | continuous | no | unknown | unknown |
| 50 | CEC2020_p35 | Engineering | 153 | 1 | 148 | continuous | no | unknown | unknown |
| 51 | CEC2020_p36 | Engineering | 158 | 1 | 148 | continuous | no | unknown | unknown |
| 52 | CEC2020_p37 | Engineering | 126 | 1 | 116 | continuous | no | unknown | unknown |
| 53 | CEC2020_p38 | Engineering | 126 | 1 | 116 | continuous | no | unknown | unknown |
| 54 | CEC2020_p39 | Engineering | 126 | 1 | 116 | continuous | no | unknown | unknown |
| 55 | CEC2020_p4 | Engineering | 6 | 1 | 4 | continuous | no | unknown | unknown |
| 56 | CEC2020_p40 | Engineering | 76 | 1 | 76 | continuous | no | unknown | unknown |
| 57 | CEC2020_p41 | Engineering | 74 | 1 | 74 | continuous | no | unknown | unknown |
| 58 | CEC2020_p42 | Engineering | 86 | 1 | 76 | continuous | no | unknown | unknown |
| 59 | CEC2020_p43 | Engineering | 86 | 1 | 76 | continuous | no | unknown | unknown |
| 60 | CEC2020_p44 | Engineering | 30 | 1 | 0 | continuous | no | unknown | unknown |
| 61 | CEC2020_p45 | Engineering | 25 | 1 | 1 | continuous | no | unknown | unknown |
| 62 | CEC2020_p46 | Engineering | 25 | 1 | 1 | continuous | no | unknown | unknown |
| 63 | CEC2020_p47 | Engineering | 25 | 1 | 1 | continuous | no | unknown | unknown |
| 64 | CEC2020_p48 | Engineering | 30 | 1 | 1 | continuous | no | unknown | unknown |
| 65 | CEC2020_p49 | Engineering | 30 | 1 | 1 | continuous | no | unknown | unknown |
| 66 | CEC2020_p5 | Engineering | 9 | 1 | 4 | continuous | no | unknown | unknown |
| 67 | CEC2020_p50 | Engineering | 30 | 1 | 1 | continuous | no | unknown | unknown |
| 68 | CEC2020_p51 | Engineering | 59 | 1 | 1 | continuous | no | unknown | unknown |
| 69 | CEC2020_p52 | Engineering | 59 | 1 | 1 | continuous | no | unknown | unknown |
| 70 | CEC2020_p53 | Engineering | 59 | 1 | 1 | continuous | no | unknown | unknown |
| 71 | CEC2020_p54 | Engineering | 59 | 1 | 1 | continuous | no | unknown | unknown |
| 72 | CEC2020_p55 | Engineering | 64 | 1 | 6 | continuous | no | unknown | unknown |
| 73 | CEC2020_p56 | Engineering | 64 | 1 | 6 | continuous | no | unknown | unknown |
| 74 | CEC2020_p57 | Engineering | 64 | 1 | 6 | continuous | no | unknown | unknown |
| 75 | CEC2020_p6 | Engineering | 38 | 1 | 32 | continuous | no | unknown | unknown |
| 76 | CEC2020_p7 | Engineering | 48 | 1 | 38 | continuous | no | unknown | unknown |
| 77 | CEC2020_p8 | Engineering | 2 | 1 | 0 | continuous | no | unknown | unknown |
| 78 | CEC2020_p9 | Engineering | 3 | 1 | 1 | continuous | no | unknown | unknown |
| 79 | CRE21 | Engineering | 3 | 2 | 3 | continuous | no | unknown | unknown |
| 80 | CRE22 | Engineering | 4 | 2 | 4 | continuous | no | unknown | unknown |
| 81 | CRE23 | Engineering | 4 | 2 | 4 | continuous | no | unknown | unknown |
| 82 | CRE24 | Engineering | 7 | 2 | 11 | continuous | no | unknown | unknown |
| 83 | CRE25 | Engineering | 4 | 2 | 1 | continuous | no | unknown | unknown |
| 84 | CRE31 | Engineering | 7 | 3 | 10 | continuous | no | unknown | unknown |
| 85 | CRE32 | Engineering | 6 | 3 | 9 | continuous | no | unknown | unknown |
| 86 | CRE51 | Engineering | 3 | 5 | 7 | continuous | no | unknown | unknown |
| 87 | CS1 | Engineering | 20 | 2 | 7 | continuous | no | unknown | unknown |
| 88 | CS2 | Engineering | 20 | 2 | 8 | continuous | no | unknown | unknown |
| 89 | CS3 | Engineering | 20 | 2 | 10 | continuous | no | unknown | unknown |
| 90 | CS4 | Engineering | 20 | 2 | 9 | continuous | no | unknown | unknown |
| 91 | CT1 | Engineering | 20 | 2 | 7 | continuous | no | unknown | unknown |
| 92 | CT2 | Engineering | 20 | 2 | 8 | continuous | no | unknown | unknown |
| 93 | CT3 | Engineering | 20 | 2 | 10 | continuous | no | unknown | unknown |
| 94 | CT4 | Engineering | 20 | 2 | 9 | continuous | no | unknown | unknown |
| 95 | CTS1 | Engineering | 20 | 3 | 7 | continuous | no | unknown | unknown |
| 96 | CTS2 | Engineering | 20 | 3 | 8 | continuous | no | unknown | unknown |
| 97 | CTS3 | Engineering | 20 | 3 | 10 | continuous | no | unknown | unknown |
| 98 | CTS4 | Engineering | 20 | 3 | 9 | continuous | no | unknown | unknown |
| 99 | CTSE1 | Engineering | 20 | 4 | 7 | continuous | no | unknown | unknown |
| 100 | CTSE2 | Engineering | 20 | 4 | 8 | continuous | no | unknown | unknown |
| 101 | CTSE3 | Engineering | 20 | 4 | 10 | continuous | no | unknown | unknown |
| 102 | CTSE4 | Engineering | 20 | 4 | 9 | continuous | no | unknown | unknown |
| 103 | CTSEI1 | Engineering | 20 | 5 | 7 | continuous | no | unknown | unknown |
| 104 | CTSEI2 | Engineering | 20 | 5 | 8 | continuous | no | unknown | unknown |
| 105 | CTSEI3 | Engineering | 20 | 5 | 10 | continuous | no | unknown | unknown |
| 106 | CTSEI4 | Engineering | 20 | 5 | 9 | continuous | no | unknown | unknown |
| 107 | CantileverBeam | Engineering | 10 | 1 | 11 | continuous | no | unknown | unknown |
| 108 | Car | Engineering | 11 | 1 | 10 | continuous | no | unknown | unknown |
| 109 | CarSideImpact | Engineering | 7 | 3 | 10 | continuous | no | unknown | unknown |
| 110 | CompressionSpring | Engineering | 3 | 1 | 4 | continuous | no | unknown | unknown |
| 111 | DiscBrake | Engineering | 4 | 2 | 4 | continuous | no | unknown | unknown |
| 112 | EulerBernoulliBeamBending | Engineering | 3 | 1 | 0 | continuous | no | unknown | unknown |
| 113 | GearTrain | Engineering | 4 | 1 | 0 | mixed | no | unknown | unknown |
| 114 | HeatExchanger | Engineering | 8 | 1 | 6 | continuous | no | unknown | unknown |
| 115 | MOPTA08Car | Engineering | 124 | 1 | 68 | continuous | no | no | unknown |
| 116 | Mazda | Engineering | 222 | 5 | 54 | continuous | no | unknown | unknown |
| 117 | Mazda_SCA | Engineering | 148 | 4 | 36 | continuous | no | unknown | unknown |
| 118 | Penicillin | Engineering | 7 | 3 | 0 | continuous | no | unknown | unknown |
| 119 | PressureVessel | Engineering | 4 | 1 | 4 | continuous | no | unknown | unknown |
| 120 | QPowerModel | Engineering | 8 | 1 | 0 | continuous | no | unknown | unknown |
| 121 | RE21 | Engineering | 4 | 2 | 0 | continuous | no | unknown | unknown |
| 122 | RE22 | Engineering | 3 | 2 | 0 | continuous | no | unknown | unknown |
| 123 | RE23 | Engineering | 4 | 2 | 0 | continuous | no | unknown | unknown |
| 124 | RE24 | Engineering | 2 | 2 | 0 | continuous | no | unknown | unknown |
| 125 | RE25 | Engineering | 3 | 2 | 0 | continuous | no | unknown | unknown |
| 126 | RE31 | Engineering | 3 | 3 | 0 | continuous | no | unknown | unknown |
| 127 | RE32 | Engineering | 4 | 3 | 0 | continuous | no | unknown | unknown |
| 128 | RE33 | Engineering | 4 | 3 | 0 | continuous | no | unknown | unknown |
| 129 | RE34 | Engineering | 5 | 3 | 0 | continuous | no | unknown | unknown |
| 130 | RE35 | Engineering | 7 | 3 | 0 | continuous | no | unknown | unknown |
| 131 | RE36 | Engineering | 4 | 3 | 0 | continuous | no | unknown | unknown |
| 132 | RE37 | Engineering | 4 | 3 | 0 | continuous | no | unknown | unknown |
| 133 | RE41 | Engineering | 7 | 4 | 0 | continuous | no | unknown | unknown |
| 134 | RE42 | Engineering | 6 | 4 | 0 | continuous | no | unknown | unknown |
| 135 | RE61 | Engineering | 3 | 6 | 0 | continuous | no | unknown | unknown |
| 136 | RE91 | Engineering | 7 | 9 | 0 | continuous | no | unknown | unknown |
| 137 | ReactivityModel | Engineering | 8 | 1 | 0 | continuous | no | unknown | unknown |
| 138 | ReinforcedConcreteBeam | Engineering | 3 | 1 | 2 | continuous | no | unknown | unknown |
| 139 | RobotPush | Engineering | 14 | 1 | 0 | continuous | no | no | unknown |
| 140 | Rover | Engineering | 100 | 1 | 0 | continuous | no | unknown | unknown |
| 141 | SpeedReducer | Engineering | 7 | 1 | 9 | continuous | no | unknown | unknown |
| 142 | ThreeTruss | Engineering | 2 | 1 | 3 | continuous | no | unknown | unknown |
| 143 | Truss10D | Engineering | 10 | 1 | 14 | continuous | no | unknown | unknown |
| 144 | Truss120D | Engineering | 120 | 1 | 121 | continuous | no | unknown | unknown |
| 145 | Truss200D | Engineering | 200 | 1 | 200 | continuous | no | unknown | unknown |
| 146 | Truss25D | Engineering | 25 | 1 | 31 | continuous | no | unknown | unknown |
| 147 | Truss72D_FourForces | Engineering | 72 | 1 | 88 | continuous | no | unknown | unknown |
| 148 | Truss72D_SingleForce | Engineering | 72 | 1 | 88 | continuous | no | unknown | unknown |
| 149 | TwoBarTruss | Engineering | 2 | 2 | 5 | continuous | no | unknown | unknown |
| 150 | VehicleSafety | Engineering | 5 | 3 | 0 | continuous | no | unknown | unknown |
| 151 | WaterProblem | Engineering | 3 | 5 | 7 | continuous | no | unknown | unknown |
| 152 | WaterResources | Engineering | 3 | 5 | 7 | continuous | no | unknown | unknown |
| 153 | WeldedBeam | Engineering | 4 | 2 | 4 | continuous | no | unknown | unknown |
| 154 | LassoBreastCancer | Hyperparameter Optimization | 10 | 1 | 0 | continuous | no | unknown | unknown |
| 155 | LassoDNA | Hyperparameter Optimization | 180 | 1 | 0 | continuous | no | unknown | unknown |
| 156 | LassoDiabetes | Hyperparameter Optimization | 8 | 1 | 0 | continuous | no | unknown | unknown |
| 157 | LassoLeukemia | Hyperparameter Optimization | 7129 | 1 | 0 | continuous | no | unknown | unknown |
| 158 | LassoRCV1 | Hyperparameter Optimization | 47236 | 1 | 0 | continuous | no | unknown | unknown |
| 159 | SVM | Hyperparameter Optimization | 388 | 1 | 0 | continuous | no | no | unknown |
| 160 | AgNP | Materials | 5 | 1 | 0 | discrete | no | unknown | unknown |
| 161 | AutoAM | Materials | 4 | 1 | 0 | discrete | no | unknown | unknown |
| 162 | CrossedBarrel | Materials | 4 | 1 | 0 | discrete | no | unknown | unknown |
| 163 | P3HT | Materials | 5 | 1 | 0 | discrete | no | unknown | unknown |
| 164 | Perovskite | Materials | 3 | 1 | 0 | discrete | no | unknown | unknown |

<!-- TABLE:END -->
