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

<!-- TABLE:START -->

| Problem | Application | Dim | #Obj | #Constr | Variables | Scalable | Convex | NP-hard |
|---|---|---|---|---|---|---|---|---|
| TSP_100Cities | Combinatorial | 100 | 1 | 0 | discrete | no | no | yes |
| TSP_51Cities | Combinatorial | 51 | 1 | 0 | discrete | no | no | yes |
| AntPolicySearchProblem | Control | 840 | 1 | 0 | continuous | no | unknown | unknown |
| AntProblem | Control | 8 | 1 | 0 | continuous | no | unknown | unknown |
| HalfCheetahPolicySearchProblem | Control | 102 | 1 | 0 | continuous | no | unknown | unknown |
| HalfCheetahProblem | Control | 6 | 1 | 0 | continuous | no | unknown | unknown |
| HopperPolicySearchProblem | Control | 33 | 1 | 0 | continuous | no | unknown | unknown |
| HopperProblem | Control | 3 | 1 | 0 | continuous | no | unknown | unknown |
| HumanoidProblem | Control | 17 | 1 | 0 | continuous | no | unknown | unknown |
| HumanoidStandupProblem | Control | 17 | 1 | 0 | continuous | no | unknown | unknown |
| InvertedDoublePendulumProblem | Control | 1 | 1 | 0 | continuous | no | unknown | unknown |
| InvertedPendulumProblem | Control | 1 | 1 | 0 | continuous | no | unknown | unknown |
| PD4CartPole | Control | 4 | 1 | 0 | continuous | no | unknown | unknown |
| PID4Acrobot | Control | 3 | 1 | 0 | continuous | no | unknown | unknown |
| PusherProblem | Control | 7 | 1 | 0 | continuous | no | unknown | unknown |
| ReacherProblem | Control | 2 | 1 | 0 | continuous | no | unknown | unknown |
| SwimmerPolicySearchProblem | Control | 16 | 1 | 0 | continuous | no | unknown | unknown |
| SwimmerProblem | Control | 2 | 1 | 0 | continuous | no | unknown | unknown |
| Walker2DPolicySearchProblem | Control | 102 | 1 | 0 | continuous | no | unknown | unknown |
| Walker2DProblem | Control | 6 | 1 | 0 | continuous | no | unknown | unknown |
| BotorchCarSideImpact | Engineering | 7 | 4 | 0 | continuous | no | unknown | unknown |
| CEC2020_p1 | Engineering | 9 | 1 | 8 | continuous | no | unknown | unknown |
| CEC2020_p10 | Engineering | 3 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p11 | Engineering | 7 | 1 | 4 | continuous | no | unknown | unknown |
| CEC2020_p12 | Engineering | 7 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p13 | Engineering | 5 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p14 | Engineering | 10 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p15 | Engineering | 7 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p16 | Engineering | 14 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p17 | Engineering | 3 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p18 | Engineering | 4 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p19 | Engineering | 4 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p2 | Engineering | 11 | 1 | 9 | continuous | no | unknown | unknown |
| CEC2020_p20 | Engineering | 2 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p21 | Engineering | 5 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p22 | Engineering | 9 | 1 | 1 | continuous | no | unknown | unknown |
| CEC2020_p23 | Engineering | 5 | 1 | 3 | continuous | no | unknown | unknown |
| CEC2020_p24 | Engineering | 7 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p25 | Engineering | 7 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p26 | Engineering | 22 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p27 | Engineering | 10 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p28 | Engineering | 10 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p29 | Engineering | 4 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p3 | Engineering | 7 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p30 | Engineering | 3 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p31 | Engineering | 4 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p32 | Engineering | 5 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p33 | Engineering | 30 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p34 | Engineering | 118 | 1 | 108 | continuous | no | unknown | unknown |
| CEC2020_p35 | Engineering | 153 | 1 | 148 | continuous | no | unknown | unknown |
| CEC2020_p36 | Engineering | 158 | 1 | 148 | continuous | no | unknown | unknown |
| CEC2020_p37 | Engineering | 126 | 1 | 116 | continuous | no | unknown | unknown |
| CEC2020_p38 | Engineering | 126 | 1 | 116 | continuous | no | unknown | unknown |
| CEC2020_p39 | Engineering | 126 | 1 | 116 | continuous | no | unknown | unknown |
| CEC2020_p4 | Engineering | 6 | 1 | 4 | continuous | no | unknown | unknown |
| CEC2020_p40 | Engineering | 76 | 1 | 76 | continuous | no | unknown | unknown |
| CEC2020_p41 | Engineering | 74 | 1 | 74 | continuous | no | unknown | unknown |
| CEC2020_p42 | Engineering | 86 | 1 | 76 | continuous | no | unknown | unknown |
| CEC2020_p43 | Engineering | 86 | 1 | 76 | continuous | no | unknown | unknown |
| CEC2020_p44 | Engineering | 30 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p45 | Engineering | 25 | 1 | 1 | continuous | no | unknown | unknown |
| CEC2020_p46 | Engineering | 25 | 1 | 1 | continuous | no | unknown | unknown |
| CEC2020_p47 | Engineering | 25 | 1 | 1 | continuous | no | unknown | unknown |
| CEC2020_p48 | Engineering | 30 | 1 | 1 | continuous | no | unknown | unknown |
| CEC2020_p49 | Engineering | 30 | 1 | 1 | continuous | no | unknown | unknown |
| CEC2020_p5 | Engineering | 9 | 1 | 4 | continuous | no | unknown | unknown |
| CEC2020_p50 | Engineering | 30 | 1 | 1 | continuous | no | unknown | unknown |
| CEC2020_p51 | Engineering | 59 | 1 | 1 | continuous | no | unknown | unknown |
| CEC2020_p52 | Engineering | 59 | 1 | 1 | continuous | no | unknown | unknown |
| CEC2020_p53 | Engineering | 59 | 1 | 1 | continuous | no | unknown | unknown |
| CEC2020_p54 | Engineering | 59 | 1 | 1 | continuous | no | unknown | unknown |
| CEC2020_p55 | Engineering | 64 | 1 | 6 | continuous | no | unknown | unknown |
| CEC2020_p56 | Engineering | 64 | 1 | 6 | continuous | no | unknown | unknown |
| CEC2020_p57 | Engineering | 64 | 1 | 6 | continuous | no | unknown | unknown |
| CEC2020_p6 | Engineering | 38 | 1 | 32 | continuous | no | unknown | unknown |
| CEC2020_p7 | Engineering | 48 | 1 | 38 | continuous | no | unknown | unknown |
| CEC2020_p8 | Engineering | 2 | 1 | 0 | continuous | no | unknown | unknown |
| CEC2020_p9 | Engineering | 3 | 1 | 1 | continuous | no | unknown | unknown |
| CRE21 | Engineering | 3 | 2 | 3 | continuous | no | unknown | unknown |
| CRE22 | Engineering | 4 | 2 | 4 | continuous | no | unknown | unknown |
| CRE23 | Engineering | 4 | 2 | 4 | continuous | no | unknown | unknown |
| CRE24 | Engineering | 7 | 2 | 11 | continuous | no | unknown | unknown |
| CRE25 | Engineering | 4 | 2 | 1 | continuous | no | unknown | unknown |
| CRE31 | Engineering | 7 | 3 | 10 | continuous | no | unknown | unknown |
| CRE32 | Engineering | 6 | 3 | 9 | continuous | no | unknown | unknown |
| CRE51 | Engineering | 3 | 5 | 7 | continuous | no | unknown | unknown |
| CS1 | Engineering | 20 | 2 | 7 | continuous | no | unknown | unknown |
| CS2 | Engineering | 20 | 2 | 8 | continuous | no | unknown | unknown |
| CS3 | Engineering | 20 | 2 | 10 | continuous | no | unknown | unknown |
| CS4 | Engineering | 20 | 2 | 9 | continuous | no | unknown | unknown |
| CT1 | Engineering | 20 | 2 | 7 | continuous | no | unknown | unknown |
| CT2 | Engineering | 20 | 2 | 8 | continuous | no | unknown | unknown |
| CT3 | Engineering | 20 | 2 | 10 | continuous | no | unknown | unknown |
| CT4 | Engineering | 20 | 2 | 9 | continuous | no | unknown | unknown |
| CTS1 | Engineering | 20 | 3 | 7 | continuous | no | unknown | unknown |
| CTS2 | Engineering | 20 | 3 | 8 | continuous | no | unknown | unknown |
| CTS3 | Engineering | 20 | 3 | 10 | continuous | no | unknown | unknown |
| CTS4 | Engineering | 20 | 3 | 9 | continuous | no | unknown | unknown |
| CTSE1 | Engineering | 20 | 4 | 7 | continuous | no | unknown | unknown |
| CTSE2 | Engineering | 20 | 4 | 8 | continuous | no | unknown | unknown |
| CTSE3 | Engineering | 20 | 4 | 10 | continuous | no | unknown | unknown |
| CTSE4 | Engineering | 20 | 4 | 9 | continuous | no | unknown | unknown |
| CTSEI1 | Engineering | 20 | 5 | 7 | continuous | no | unknown | unknown |
| CTSEI2 | Engineering | 20 | 5 | 8 | continuous | no | unknown | unknown |
| CTSEI3 | Engineering | 20 | 5 | 10 | continuous | no | unknown | unknown |
| CTSEI4 | Engineering | 20 | 5 | 9 | continuous | no | unknown | unknown |
| CantileverBeam | Engineering | 10 | 1 | 11 | continuous | no | unknown | unknown |
| Car | Engineering | 11 | 1 | 10 | continuous | no | unknown | unknown |
| CarSideImpact | Engineering | 7 | 3 | 10 | continuous | no | unknown | unknown |
| CompressionSpring | Engineering | 3 | 1 | 4 | continuous | no | unknown | unknown |
| DiscBrake | Engineering | 4 | 2 | 4 | continuous | no | unknown | unknown |
| EulerBernoulliBeamBending | Engineering | 3 | 1 | 0 | continuous | no | unknown | unknown |
| GearTrain | Engineering | 4 | 1 | 0 | mixed | no | unknown | unknown |
| HeatExchanger | Engineering | 8 | 1 | 6 | continuous | no | unknown | unknown |
| MOPTA08Car | Engineering | 124 | 1 | 68 | continuous | no | no | unknown |
| Mazda | Engineering | 222 | 5 | 54 | continuous | no | unknown | unknown |
| Mazda_SCA | Engineering | 148 | 4 | 36 | continuous | no | unknown | unknown |
| Penicillin | Engineering | 7 | 3 | 0 | continuous | no | unknown | unknown |
| PressureVessel | Engineering | 4 | 1 | 4 | continuous | no | unknown | unknown |
| QPowerModel | Engineering | 8 | 1 | 0 | continuous | no | unknown | unknown |
| RE21 | Engineering | 4 | 2 | 0 | continuous | no | unknown | unknown |
| RE22 | Engineering | 3 | 2 | 0 | continuous | no | unknown | unknown |
| RE23 | Engineering | 4 | 2 | 0 | continuous | no | unknown | unknown |
| RE24 | Engineering | 2 | 2 | 0 | continuous | no | unknown | unknown |
| RE25 | Engineering | 3 | 2 | 0 | continuous | no | unknown | unknown |
| RE31 | Engineering | 3 | 3 | 0 | continuous | no | unknown | unknown |
| RE32 | Engineering | 4 | 3 | 0 | continuous | no | unknown | unknown |
| RE33 | Engineering | 4 | 3 | 0 | continuous | no | unknown | unknown |
| RE34 | Engineering | 5 | 3 | 0 | continuous | no | unknown | unknown |
| RE35 | Engineering | 7 | 3 | 0 | continuous | no | unknown | unknown |
| RE36 | Engineering | 4 | 3 | 0 | continuous | no | unknown | unknown |
| RE37 | Engineering | 4 | 3 | 0 | continuous | no | unknown | unknown |
| RE41 | Engineering | 7 | 4 | 0 | continuous | no | unknown | unknown |
| RE42 | Engineering | 6 | 4 | 0 | continuous | no | unknown | unknown |
| RE61 | Engineering | 3 | 6 | 0 | continuous | no | unknown | unknown |
| RE91 | Engineering | 7 | 9 | 0 | continuous | no | unknown | unknown |
| ReactivityModel | Engineering | 8 | 1 | 0 | continuous | no | unknown | unknown |
| ReinforcedConcreteBeam | Engineering | 3 | 1 | 2 | continuous | no | unknown | unknown |
| RobotPush | Engineering | 14 | 1 | 0 | continuous | no | no | unknown |
| Rover | Engineering | 100 | 1 | 0 | continuous | no | unknown | unknown |
| SpeedReducer | Engineering | 7 | 1 | 9 | continuous | no | unknown | unknown |
| ThreeTruss | Engineering | 2 | 1 | 3 | continuous | no | unknown | unknown |
| Truss10D | Engineering | 10 | 1 | 14 | continuous | no | unknown | unknown |
| Truss120D | Engineering | 120 | 1 | 121 | continuous | no | unknown | unknown |
| Truss200D | Engineering | 200 | 1 | 200 | continuous | no | unknown | unknown |
| Truss25D | Engineering | 25 | 1 | 31 | continuous | no | unknown | unknown |
| Truss72D_FourForces | Engineering | 72 | 1 | 88 | continuous | no | unknown | unknown |
| Truss72D_SingleForce | Engineering | 72 | 1 | 88 | continuous | no | unknown | unknown |
| TwoBarTruss | Engineering | 2 | 2 | 5 | continuous | no | unknown | unknown |
| VehicleSafety | Engineering | 5 | 3 | 0 | continuous | no | unknown | unknown |
| WaterProblem | Engineering | 3 | 5 | 7 | continuous | no | unknown | unknown |
| WaterResources | Engineering | 3 | 5 | 7 | continuous | no | unknown | unknown |
| WeldedBeam | Engineering | 4 | 2 | 4 | continuous | no | unknown | unknown |
| LassoBreastCancer | Hyperparameter Optimization | 10 | 1 | 0 | continuous | no | unknown | unknown |
| LassoDNA | Hyperparameter Optimization | 180 | 1 | 0 | continuous | no | unknown | unknown |
| LassoDiabetes | Hyperparameter Optimization | 8 | 1 | 0 | continuous | no | unknown | unknown |
| LassoLeukemia | Hyperparameter Optimization | 7129 | 1 | 0 | continuous | no | unknown | unknown |
| LassoRCV1 | Hyperparameter Optimization | 47236 | 1 | 0 | continuous | no | unknown | unknown |
| SVM | Hyperparameter Optimization | 388 | 1 | 0 | continuous | no | no | unknown |
| AgNP | Materials | 5 | 1 | 0 | discrete | no | unknown | unknown |
| AutoAM | Materials | 4 | 1 | 0 | discrete | no | unknown | unknown |
| CrossedBarrel | Materials | 4 | 1 | 0 | discrete | no | unknown | unknown |
| P3HT | Materials | 5 | 1 | 0 | discrete | no | unknown | unknown |
| Perovskite | Materials | 3 | 1 | 0 | discrete | no | unknown | unknown |

<!-- TABLE:END -->
