# Benchmark Problem Dictionary
So that we know what is found and what we have yet found.
--Rosen


# Problems that are coded in our set:

## Botorch:
| Problem Name | num of Objectives | dim of X |  num of Constraints | Problem Type |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Ackley | 1  | N  | 2  | Single Objective, Constrained, Continuous  |
| Branin | 1  | 2  | 0  | Single Objective (is there a mix-variable or constrained form?)  |
| Bukin | 1  | 2  | 0  | Single Objective (is there a more variance?)  |
| Dixon Price | 1  | N  | 0  | Single Objective, Continuous (is there a constrained form?)  |
| Griewank | 1  | N  | 0  | Single Objective, Continuous (is there a constrained form?)  |
| Hartmann 6D | 1  | 6  | 0  | Single Objective, Continuous (is there a constrained form?)  |
| Levy | 1  | N  | 0  | Single Objective, Continuous (is there a constrained form?)   |
| Michalewicz | 1 | N  | 0  | Single Objective, Continuous (is there a constrained form?) |
| Michalewicz | 1 | N  | 0  | Single Objective, Continuous (is there a constrained form?) |
| BraninCurrin | 2 | 2  | 0  | Multi Objective |
| DH1 | 2 | N (min_dim = 2)  | 0  | Multi Objective |
| DH2 | 2 | N  | 0  | Multi Objective |
| DH3 | 2 | N (min_dim = 3)  | 0  | Multi Objective |
| DH4 | 2 | N  | 0  | Multi Objective |
| DTLZ1 | 2 | N  | 0  | Multi Objective |
| DTLZ2 | 2 | N  | 0  | Multi Objective |
| DTLZ3 | 2 | N  | 0  | Multi Objective |
| DTLZ4 | 2 | N  | 0  | Multi Objective |
| DTLZ5 | 2 | N  | 0  | Multi Objective |
| DTLZ7 | 2 | N  | 0  | Multi Objective |
| GMM | 2 | 2  | 0  | Multi Objective |
| Penicillin | 3 | 7  | 0  | Multi Objective |
| ToyRobust | 2 | 1  | 0  | Multi Objective |
| VehicleSafety | 3 | 5  | 0  | Multi Objective |
| ZDT1 | 2 | 2 or more  | 0  | Multi Objective |
| ZDT2 | 2 | 2 or more  | 0  | Multi Objective |
| ZDT3 | 2 | 2 or more  | 0  | Multi Objective |
| CarSideImpact | 4 | 7  | 0  | Multi Objective |
| BNH | 2 | 2  | 2  | Multi Objective, Constrained |
| CONSTR | 2 | 2  | 2  | Multi Objective |
| ConstrainedBraninCurrin | 2 | 2  | 1  | Multi Objective, Constrained |
| C2DTLZ2 | 2 | 2 or more  | 1  | Multi Objective, Constrained |
| DiscBrake | 2 | 4 | 4  | Multi Objective, Constrained |
| MW7 | 2 | 1 or more | 2 | Multi Objective, Constrained |
| OSY | 2 | 6 | 6 | Multi Objective, Constrained |
| SRN | 2 | 2 | 2 | Multi Objective, Constrained |
| WeldedBeam | 2 | 4 | 4 | Multi Objective, Constrained |


## Pymoo:
| Problem Name | num of Objectives | dim of Objectives |  num of Constraints | Problem Type |
| ------------- | ------------- | ------------- | ------------- | ------------- |

## CEC:
| Problem Name | num of Objectives | dim of Objectives |  num of Constraints | Problem Type |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| HappyCat | 1 | D | 0 | Single Objective, Continuous |
| Discus | 1 | D | 0 | Single Objective, Continuous |
| HGBat | 1 | D | 0 | Single Objective, Continuous |
| Schaffer | 1 | D | 0 | Single Objective, Continuous |

## Others:

| Problem Name | num of Objectives | dim of Objectives |  num of Constraints | Problem Type |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Cantilever Step Beam  (engineering) | 1  | 10  | 11  | Single Objective, Constrained (is there a discrete form? and I think we can expand to N steps (right now it's five steps))  |
| Car crashed simplified  (engineering) | 1  | 11  | 10  | Single Objective, Constrained (is there a mix-variable or multi-obj form?)  |
| Compression Spring  (engineering) | 1  | 8  | 6  | Single Objective, Constrained (is there a mix-variable or multi-obj form?)  |
| GKXWC1  (synthetic) | 1  | 2  | 1  | Single Objective, Constrained, Continuous  |
| GKXWC2  (synthetic) | 1  | 2  | 1  | Single Objective, Constrained, Continuous  |
| JLH1 (synthetic) | 1  | 2  | 1  | Single Objective, Constrained, Continuous  |
| JLH2 (synthetic) | 1  | 2  | 1  | Single Objective, Constrained, Continuous  |
| Keane Bump (synthetic) | 1  | N  | 2  | Single Objective, Constrained, Continuous  |
| Mazda (engineering) | m | 222  | 54  | Single/Multi Objective, Constrained, Continuous (is there a mix-variable form?) (I haven't upload the executables)  |
| MOPTA08_Car (engineering) | 1 | 124  | 68  | Single Objective, Constrained, Continuous (is there a mix-variable form?) (I haven't upload the executables)   |
| PressureVessel (engineering) | 1 | 4  | 4  | Single Objective, Constrained, Continuous (is there a mix-variable or multi-obj form?)   |
| ReinforcedConcreteBeam (engineering) | 1 | 3  | 9  | Single Objective, Constrained, Continuous (We should implement the mix-variable form)   |
| Rosenbrock (botorch) | 1 | N  | 0  | Single Objective, Continuous (is there a constrained form?)   |
| Speed Reducer (engineering) | 1 | 7  | 9  | Single Objective, Constrained, Continuous (is there a mix-variable or multi-obj form?)   |
| Three-bar Truss (engineering) | 1 | 2  | 3  | Single Objective, Constrained, Continuous  |
| TrussSolvers (engineering, slientruss3d) | 1 | 10, 25, 72, 120  | depends  | Single Objective, Continuous (is there a mix-variable or multi-obj form?) |
| WeldedBeam (engineering) | 1 | 4  | 5  | Single Objective, Constrained, Continuous (is there a mix-variable or multi-obj form?)  |
| Sharp Ride | 1 | D | 0 | Single Objective, Continuous |
| Different Powers | 1 | D | 0 | Single Objective, Continuous |


# Literature, Reference, and URL for the current set of problems:
* Optimization Test Problems: https://www.sfu.ca/~ssurjano/optimization.html
* Computational Optimization, Methods and Algorithms, Chapter 12 Benchmark Problems in Structural Optimization (Amir Hossein Gandomi and Xin-She Yang)
* Bat algorithm for constrained optimization tasks (Amir H Gandomi, Xin-She Yang, Amir H. Alavi, Siamak Talatahari)
* Mixed variable structural optimization using Firefly Algorithm, https://www.sciencedirect.com/science/article/pii/S0045794911002185
* Constraining the Feasible Design Space in Bayesian Optimization With User Feedback, Christopher Hoyle, https://asmedigitalcollection.asme.org/mechanicaldesign/article/146/4/041703/1169762/Constraining-the-Feasible-Design-Space-in-Bayesian
* slientruss3d : Python for stable truss analysis and deep learning research, https://github.com/leo27945875/Python_Stable_3D_Truss_Analysis
* Mopta08 Executables: https://leonard.papenmeier.io/2023/02/09/mopta08-executables.html
* Mazda car benchmark: https://ladse.eng.isas.jaxa.jp/benchmark/jpn/index.html



# ToDos (What have we found but not yet coded. Please add to the list. Check the existed set before adding new ones):

## Step 0: Always check if a problem already exist here before next step
1. BoTorch library
2. Pymoo library
3. Optimization Test Problems: https://www.sfu.ca/~ssurjano/optimization.html
4. Blackbox optimization benchmarking (BBOB) on COCO(COmparing Continuous Optimizer): http://numbbo.github.io/coco/testsuites/bbob
5. **Honestly just go look at the CEC problems each year:**
6. PlatEMO's Github: https://github.com/BIMK/PlatEMO/tree/master/PlatEMO/Problems
7. CEC 2017 , https://www.researchgate.net/publication/317228117_Problem_Definitions_and_Evaluation_Criteria_for_the_CEC_2017_Competition_and_Special_Session_on_Constrained_Single_Objective_Real-Parameter_Optimization?enrichId=rgreq-b8f8213db02831458225d2c8ba3fe09d-XXX&enrichSource=Y292ZXJQYWdlOzMxNzIyODExNztBUzo1MjMxNjg4MDI1MDA2MDhAMTUwMTc0NDU3MDUwMQ%3D%3D&el=1_x_3&_esc=publicationCoverPdf
8. CEC 2014, https://github.com/P-N-Suganthan/CEC2014/tree/master

## Step 0.5: 
1. Cyril's ultimate multi-objective constrained problem! : https://github.com/epfl-lamd/modact

## Step 1+:
1. Openmdao: https://openmdao.org/newdocs/versions/latest/examples/examples.html
2. Electric motors: https://www.pyleecan.org/index.html
3. Parameterized Quantum Circuits and Bayesian Optimization: https://github.com/w00zie/pqc_chsh/tree/main
4. ICML2018 paper "Batch Bayesian Optimization via Multi-objective Acquisition Ensemble for Automated Analog Circuit Design": https://github.com/Alaya-in-Matrix/pyMACE/tree/master
5. 2 Engineering FEA problems: https://github.com/TsaiYK/BayesianCHT/tree/main
6. Neorl: https://neorl.readthedocs.io/en/latest/
7. Multi-objective Bayesian Optimization Supported by an Expected Pareto Distance Change (Homero Valladares and Andres Tovar): https://github.com/edrl-purdue/jmd-epdc
8. MDO: https://mdolab.engin.umich.edu/software
9. Re-Examining Linear Embeddings for High-Dimensional Bayesian Optimization (can we find the robot problem?): https://proceedings.neurips.cc/paper_files/paper/2020/file/10fb6cfa4c990d2bad5ddef4f70e8ba2-Paper.pdf
10. Standard Gaussian Process Can Be Excellent for High-Dimensional Bayesian Optimization (we have some, but not all): https://arxiv.org/pdf/2402.02746
11. Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB (Can we steal any of these: https)://github.com/MariaLauraSantoni/IOH-Profiler-HDBO-Comparison
12. BOUNCE: https://arxiv.org/pdf/2307.00618 and https://neurips.cc/virtual/2023/poster/71554
13. HEBO: https://neurips.cc/virtual/2023/poster/73451 and https://github.com/huawei-noah/HEBO/tree/master/MCBO
14. Bayesian optimization for mixed-variable, multi-objective problems: https://link.springer.com/article/10.1007/s00158-022-03382-y
15. Uber's benchmark: https://github.com/uber/bayesmark/tree/master
* Kailey
15. Bayesian optimization for robust design of steel frames with joint and individual probabilistic constraints (multi-objective): https://www.sciencedirect.com/science/article/abs/pii/S0141029621010099
16. Genetic evolution vs. function approximation: Benchmarking algorithms for architectural design optimization (discrete/continuous): https://academic.oup.com/jcde/article/6/3/414/5732355
17. Bayesian optimization with known experimental and design constraints for chemistry applications (discrete/continuous, some overlap) https://pubs.rsc.org/en/content/articlehtml/2022/dd/d2dd00028h
8. Gaussian Process Assisted Particle Swarm Optimization (Rastrigin, Schwefel, some overlap): https://link.springer.com/chapter/10.1007/978-3-642-13800-3_11
* (More Possibilities)
19. A Benchmark-Suite of real-World constrained multi-objective optimization problems and some baseline results: https://www.sciencedirect.com/science/article/abs/pii/S2210650221001231
20. Bayesian optimization with known experimental and design constraints for chemistry applications (uses 8 benchmarks but might not actually be able to find ourselves
): https://pubs.rsc.org/en/content/articlehtml/2022/dd/d2dd00028h
21. Bayesian optimization of pump operations in water distribution systems: https://link.springer.com/article/10.1007/s10898-018-0641-2 
22. Archimedes screw turbines: https://www.sciencedirect.com/science/article/pii/S0306261916313861
23. Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization: https://proceedings.neurips.cc/paper_files/paper/2023/file/dbc4b67c6430c22460623186c3d3fdc2-Paper-Datasets_and_Benchmarks.pdf
24. Aircraft design: https://hal.science/hal-03346341/
25. Design point: https://www.sciencedirect.com/science/article/abs/pii/S0951832023005276
26. Multi-objective Bayesian optimization of chemical reactor design using computational fluid dynamics: https://www.sciencedirect.com/science/article/pii/S0098135418301236 
27. Water resources management: https://link.springer.com/article/10.1007/s11269-013-0350-z
28. Bayesian Optimization for Materials Science - https://link.springer.com/content/pdf/10.1007/978-981-10-6781-5.pdf
29. Revolutionizing Membrane Design Using Machine Learning-Bayesian Optimization: https://pubs.acs.org/doi/full/10.1021/acs.est.1c04373 
30. Bayesian optimization with hidden constraints for aircraft design: https://link.springer.com/article/10.1007/s00158-024-03833-8 
31. Genetic evolution vs. function approximation: Benchmarking algorithms for architectural design optimization: https://academic.oup.com/jcde/article/6/3/414/5732355 

# What Rosen think would be great:
* Make all the synthetic (numerical) function scalable 
* To my knowledge, a lot of my engineering problems have a multi-objective form (plz see the CEC 2017 pdf link)
* Each problem can have more than one variance among these six types: Single/Multi objective, With/Without constraint, Continuous/Mix-variable


**Reminder to Rosen: read these papers**
* Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB: https://arxiv.org/pdf/2303.00890
* Benchmarking in Optimization: Best Practice and Open Issues: https://arxiv.org/pdf/2007.03488

