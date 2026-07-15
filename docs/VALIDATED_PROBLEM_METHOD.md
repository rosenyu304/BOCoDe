# Validated problems and methods

_Last updated 2026-07-14 · code commit `3ab1995` (dev/2026_06). Basis: the full runnability sweep (every method smoke-tested on its target hardware — TFM on H100/H200, GP/CPU locally) plus the reproduction study against published anchors. See `RUNNABILITY_MATRIX.csv` for the full per-cell grid and `REPRODUCTION_REPORT.md` for the anchor comparisons._

**Bottom line: 298 of 300 real-world problems run cleanly with every method that applies to their category.** The only two residual caveats are hardware/physics, not bugs:
- `LassoLeukemia × git_bo` — needs an 80 GB H100 (7,129-dim); runs there, not on a 24 GB card.
- `RE91 × {qnehvi, mesmo, tfm_qnehvi}` — 9 objectives, exact hypervolume is intractable at m≥5; use `qnparego`/`tfm_qnparego` instead (both OK).


## Column 1 — Methods we are 100% confident in

| Method | Validation basis |
|---|---|
| random_search / mo_ / con_ / mocon_ (per category) | Baseline — correct by construction; every problem has exactly one applicable variant. |
| single_task_gp | Standard BoTorch ARD GP + qEI/UCB; reproduction-anchored, runs on all problems. |
| standard_gp | Xu et al. 2025 — Ackley-150D band **MATCH** (run on CPU; GPU device bug fixed in c94ba64). |
| vanilla_highdim_bo | Hvarfner et al. 2024 — Hartmann-6D regret ~0.1, **CLOSE** to the paper curve. |
| turbo | BoTorch TuRBO tutorial — Ackley-20D, **CLOSE** (multi-seed spread of the single-run anchor). |
| baxus | BAxUS paper — Branin-500D regret **1.1e-4**, textbook **MATCH**. |
| qnehvi | BoTorch MOO tutorial — BraninCurrin HV **MATCH**; NaN on badly-scaled objectives fixed (objective-standardized frame). |
| qnparego | Same MOO tutorial — HV **CLOSE** (report with enough seeds; higher variance). |
| scbo | SCBO tutorial/paper — Ackley-10D 2-constraint **CLOSE**; budget accounting verified exact. |
| penalty / mocon_penalty | Penalty-method baselines; budget-parity verified. |
| constrained_ei | Constrained EI (CPU); one GP per constraint. Runs; no published per-problem anchor. |
| constrained_qnehvi / constrained_qparego | BoTorch constrained-MO acquisitions; run on all MO-constrained problems. |
| mesmo | Runs (LB2 estimator); does NOT reproduce Belakaria's exact numbers (BoTorch lower-bound estimator) — use for ranking, not digit-match. |
| git_bo, pfn_cei, tfm_turbo, tfm_scbo, tfm_qnehvi, tfm_qnparego, tfm_cqnehvi, tfm_cqnparego | TabPFN-v3 methods — run on **H100/H200 only**. Ballpark by design (they use TabPFN-v3, not the paper checkpoints), verified to beat random search and behave sensibly. |

## Column 2 — Problems we are 100% sure run

All problems below run with **every method listed for their category** in Column 1 (GP/CPU baselines everywhere; TabPFN methods on H100/H200). Counts: SO-unconstrained 171, MO-unconstrained 19, SO-constrained 75, MO-constrained 34.


### Single-objective, unconstrained  (171 problems)
_Methods: `random_search`, `single_task_gp`, `standard_gp`, `vanilla_highdim_bo`, `turbo`, `baxus` + TabPFN(H100): `git_bo`, `tfm_turbo`_

| # | Problem |
|---|---|
| 1 | AgNP |
| 2 | Allison |
| 3 | AntPolicySearchProblem |
| 4 | AntProblem |
| 5 | AutoAM |
| 6 | Borehole |
| 7 | CEC2020_p31 |
| 8 | ColumnBuckling |
| 9 | CrossedBarrel |
| 10 | EulerBeamMixed |
| 11 | EulerBernoulliBeamBending |
| 12 | GearTrain |
| 13 | HOIP |
| 14 | HPOB_4796_23 |
| 15 | HPOB_4796_3549 |
| 16 | HPOB_4796_3918 |
| 17 | HPOB_4796_9903 |
| 18 | HPOB_4796_9906 |
| 19 | HPOB_4796_9946 |
| 20 | HPOB_5527_10101 |
| 21 | HPOB_5527_145804 |
| 22 | HPOB_5527_146064 |
| 23 | HPOB_5527_146065 |
| 24 | HPOB_5527_31 |
| 25 | HPOB_5527_9914 |
| 26 | HPOB_5636_10101 |
| 27 | HPOB_5636_145804 |
| 28 | HPOB_5636_146064 |
| 29 | HPOB_5636_146065 |
| 30 | HPOB_5636_31 |
| 31 | HPOB_5636_9914 |
| 32 | HPOB_5859_125923 |
| 33 | HPOB_5859_31 |
| 34 | HPOB_5859_37 |
| 35 | HPOB_5859_3902 |
| 36 | HPOB_5859_9977 |
| 37 | HPOB_5859_9983 |
| 38 | HPOB_5889_31 |
| 39 | HPOB_5889_3493 |
| 40 | HPOB_5889_3918 |
| 41 | HPOB_5889_3950 |
| 42 | HPOB_5889_49 |
| 43 | HPOB_5889_9971 |
| 44 | HPOB_5891_3492 |
| 45 | HPOB_5891_3891 |
| 46 | HPOB_5891_3899 |
| 47 | HPOB_5891_6566 |
| 48 | HPOB_5891_9889 |
| 49 | HPOB_5891_9980 |
| 50 | HPOB_5906_3889 |
| 51 | HPOB_5906_3896 |
| 52 | HPOB_5906_3918 |
| 53 | HPOB_5906_9970 |
| 54 | HPOB_5906_9971 |
| 55 | HPOB_5906_9977 |
| 56 | HPOB_5965_10101 |
| 57 | HPOB_5965_145836 |
| 58 | HPOB_5965_3903 |
| 59 | HPOB_5965_49 |
| 60 | HPOB_5965_9889 |
| 61 | HPOB_5965_9914 |
| 62 | HPOB_5965_9946 |
| 63 | HPOB_5970_14951 |
| 64 | HPOB_5970_34536 |
| 65 | HPOB_5970_3492 |
| 66 | HPOB_5970_37 |
| 67 | HPOB_5970_49 |
| 68 | HPOB_5970_9952 |
| 69 | HPOB_5971_10093 |
| 70 | HPOB_5971_34536 |
| 71 | HPOB_5971_3954 |
| 72 | HPOB_5971_43 |
| 73 | HPOB_5971_6566 |
| 74 | HPOB_5971_9970 |
| 75 | HPOB_6766_10101 |
| 76 | HPOB_6766_145804 |
| 77 | HPOB_6766_145953 |
| 78 | HPOB_6766_146064 |
| 79 | HPOB_6766_31 |
| 80 | HPOB_6766_3903 |
| 81 | HPOB_6767_145804 |
| 82 | HPOB_6767_146064 |
| 83 | HPOB_6767_146065 |
| 84 | HPOB_6767_31 |
| 85 | HPOB_6767_9914 |
| 86 | HPOB_6767_9967 |
| 87 | HPOB_6794_10101 |
| 88 | HPOB_6794_145804 |
| 89 | HPOB_6794_146065 |
| 90 | HPOB_6794_3 |
| 91 | HPOB_6794_31 |
| 92 | HPOB_6794_9914 |
| 93 | HPOB_7607_145976 |
| 94 | HPOB_7607_3896 |
| 95 | HPOB_7607_3903 |
| 96 | HPOB_7607_3913 |
| 97 | HPOB_7607_9946 |
| 98 | HPOB_7607_9967 |
| 99 | HPOB_7609_125923 |
| 100 | HPOB_7609_145853 |
| 101 | HPOB_7609_145854 |
| 102 | HPOB_7609_145878 |
| 103 | HPOB_7609_34537 |
| 104 | HPOB_7609_3903 |
| 105 | HPOB_7609_9967 |
| 106 | HalfCheetahPolicySearchProblem |
| 107 | HalfCheetahProblem |
| 108 | HopperPolicySearchProblem |
| 109 | HopperProblem |
| 110 | HumanoidProblem |
| 111 | HumanoidStandupProblem |
| 112 | InvertedDoublePendulumProblem |
| 113 | InvertedPendulumProblem |
| 114 | LCBenchAPSFailure |
| 115 | LCBenchAdult |
| 116 | LCBenchAirlines |
| 117 | LCBenchAlbert |
| 118 | LCBenchAmazonEmployeeAccess |
| 119 | LCBenchAustralian |
| 120 | LCBenchBankMarketing |
| 121 | LCBenchBloodTransfusionServiceCenter |
| 122 | LCBenchCar |
| 123 | LCBenchChristine |
| 124 | LCBenchCnae9 |
| 125 | LCBenchConnect4 |
| 126 | LCBenchCovertype |
| 127 | LCBenchCreditG |
| 128 | LCBenchDionis |
| 129 | LCBenchFabert |
| 130 | LCBenchFashionMNIST |
| 131 | LCBenchHelena |
| 132 | LCBenchHiggs |
| 133 | LCBenchJannis |
| 134 | LCBenchJasmine |
| 135 | LCBenchJungleChess2pcsRawEndgameComplete |
| 136 | LCBenchKDDCup09Appetency |
| 137 | LCBenchKc1 |
| 138 | LCBenchKrVsKp |
| 139 | LCBenchMfeatFactors |
| 140 | LCBenchMiniBooNE |
| 141 | LCBenchNomao |
| 142 | LCBenchNumerai286 |
| 143 | LCBenchPhoneme |
| 144 | LCBenchSegment |
| 145 | LCBenchShuttle |
| 146 | LCBenchSylvine |
| 147 | LCBenchVehicle |
| 148 | LCBenchVolkert |
| 149 | LassoBreastCancer |
| 150 | LassoDNA |
| 151 | LassoDiabetes |
| 152 | LassoLeukemia ⚠️ git_bo needs 80 GB H100 |
| 153 | MaxSAT |
| 154 | MiniAeroWing |
| 155 | NASBench201 |
| 156 | P3HT |
| 157 | PD4CartPole |
| 158 | PID4Acrobot |
| 159 | Perovskite |
| 160 | PestControl |
| 161 | PusherProblem |
| 162 | QPowerModel |
| 163 | ReacherProblem |
| 164 | ReactivityModel |
| 165 | RobotPush |
| 166 | Rover |
| 167 | SwimmerPolicySearchProblem |
| 168 | SwimmerProblem |
| 169 | Walker2DPolicySearchProblem |
| 170 | Walker2DProblem |
| 171 | Wing |

### Multi-objective, unconstrained  (19 problems)
_Methods: `mo_random_search`, `qnehvi`, `qnparego`, `mesmo` + TabPFN(H100): `tfm_qnehvi`, `tfm_qnparego`_

| # | Problem |
|---|---|
| 1 | BotorchCarSideImpact |
| 2 | Penicillin |
| 3 | RE21 |
| 4 | RE22 |
| 5 | RE23 |
| 6 | RE24 |
| 7 | RE25 |
| 8 | RE31 |
| 9 | RE32 |
| 10 | RE33 |
| 11 | RE34 |
| 12 | RE35 |
| 13 | RE36 |
| 14 | RE37 |
| 15 | RE41 |
| 16 | RE42 |
| 17 | RE61 |
| 18 | RE91 ⚠️ use qnparego (exact-HV intractable at m=9) |
| 19 | VehicleSafety |

### Single-objective, constrained  (75 problems)
_Methods: `con_random_search`, `constrained_ei`, `scbo`, `penalty` + TabPFN(H100): `pfn_cei`, `tfm_scbo`_

| # | Problem |
|---|---|
| 1 | CEC2020_p1 |
| 2 | CEC2020_p10 |
| 3 | CEC2020_p11 |
| 4 | CEC2020_p12 |
| 5 | CEC2020_p13 |
| 6 | CEC2020_p14 |
| 7 | CEC2020_p15 |
| 8 | CEC2020_p16 |
| 9 | CEC2020_p17 |
| 10 | CEC2020_p18 |
| 11 | CEC2020_p19 |
| 12 | CEC2020_p2 |
| 13 | CEC2020_p20 |
| 14 | CEC2020_p21 |
| 15 | CEC2020_p22 |
| 16 | CEC2020_p23 |
| 17 | CEC2020_p24 |
| 18 | CEC2020_p25 |
| 19 | CEC2020_p26 |
| 20 | CEC2020_p27 |
| 21 | CEC2020_p28 |
| 22 | CEC2020_p29 |
| 23 | CEC2020_p3 |
| 24 | CEC2020_p30 |
| 25 | CEC2020_p32 |
| 26 | CEC2020_p33 |
| 27 | CEC2020_p34 |
| 28 | CEC2020_p35 |
| 29 | CEC2020_p36 |
| 30 | CEC2020_p37 |
| 31 | CEC2020_p38 |
| 32 | CEC2020_p39 |
| 33 | CEC2020_p4 |
| 34 | CEC2020_p40 |
| 35 | CEC2020_p41 |
| 36 | CEC2020_p42 |
| 37 | CEC2020_p43 |
| 38 | CEC2020_p44 |
| 39 | CEC2020_p45 |
| 40 | CEC2020_p46 |
| 41 | CEC2020_p47 |
| 42 | CEC2020_p48 |
| 43 | CEC2020_p49 |
| 44 | CEC2020_p5 |
| 45 | CEC2020_p50 |
| 46 | CEC2020_p51 |
| 47 | CEC2020_p52 |
| 48 | CEC2020_p53 |
| 49 | CEC2020_p54 |
| 50 | CEC2020_p55 |
| 51 | CEC2020_p56 |
| 52 | CEC2020_p57 |
| 53 | CEC2020_p6 |
| 54 | CEC2020_p7 |
| 55 | CEC2020_p8 |
| 56 | CEC2020_p9 |
| 57 | CantileverBeam |
| 58 | Car |
| 59 | CompressionSpring |
| 60 | HeatExchanger |
| 61 | HelicalSpring |
| 62 | PEARL |
| 63 | PressureVessel |
| 64 | ReinforcedConcreteBeam |
| 65 | SatelliteDesign |
| 66 | Sellar |
| 67 | SpeedReducer |
| 68 | SteppedCantileverBeam |
| 69 | ThreeTruss |
| 70 | Truss10D |
| 71 | Truss120D |
| 72 | Truss25D |
| 73 | Truss72D_FourForces |
| 74 | Truss72D_SingleForce |
| 75 | WeldedBeamSO |

### Multi-objective, constrained  (34 problems)
_Methods: `mocon_random_search`, `constrained_qnehvi`, `constrained_qparego`, `mocon_penalty` + TabPFN(H100): `tfm_cqnehvi`, `tfm_cqnparego`_

| # | Problem |
|---|---|
| 1 | CRE21 |
| 2 | CRE22 |
| 3 | CRE23 |
| 4 | CRE24 |
| 5 | CRE25 |
| 6 | CRE31 |
| 7 | CRE32 |
| 8 | CRE51 |
| 9 | CS1 |
| 10 | CS2 |
| 11 | CS3 |
| 12 | CS4 |
| 13 | CT1 |
| 14 | CT2 |
| 15 | CT3 |
| 16 | CT4 |
| 17 | CTS1 |
| 18 | CTS2 |
| 19 | CTS3 |
| 20 | CTS4 |
| 21 | CTSE1 |
| 22 | CTSE2 |
| 23 | CTSE3 |
| 24 | CTSE4 |
| 25 | CTSEI1 |
| 26 | CTSEI2 |
| 27 | CTSEI3 |
| 28 | CTSEI4 |
| 29 | CarSideImpact |
| 30 | DiscBrake |
| 31 | Mazda |
| 32 | Mazda_SCA |
| 33 | WaterResources |
| 34 | WeldedBeam |
