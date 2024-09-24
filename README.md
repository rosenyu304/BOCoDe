# Benchmark Problem Dictionary
So that we know what is found and what we have yet found.
--Rosen


## Literature, Reference, and URL for the current set of problems:
* Optimization Test Problems: https://www.sfu.ca/~ssurjano/optimization.html
* Computational Optimization, Methods and Algorithms, Chapter 12 Benchmark Problems in Structural Optimization (Amir Hossein Gandomi and Xin-She Yang)
* Bat algorithm for constrained optimization tasks (Amir H Gandomi, Xin-She Yang, Amir H. Alavi, Siamak Talatahari)
* Mixed variable structural optimization using Firefly Algorithm, https://www.sciencedirect.com/science/article/pii/S0045794911002185
* Constraining the Feasible Design Space in Bayesian Optimization With User Feedback, Christopher Hoyle, https://asmedigitalcollection.asme.org/mechanicaldesign/article/146/4/041703/1169762/Constraining-the-Feasible-Design-Space-in-Bayesian
* slientruss3d : Python for stable truss analysis and deep learning research, https://github.com/leo27945875/Python_Stable_3D_Truss_Analysis
* Mopta08 Executables: https://leonard.papenmeier.io/2023/02/09/mopta08-executables.html
* Mazda car benchmark: https://ladse.eng.isas.jaxa.jp/benchmark/jpn/index.html



## Problems with code:
| Problem Name | num of Objectives | dim of Objectives |  num of Constraints | Problem Type |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Ackley  (botorch) | 1  | N  | 2  | Single Objective, Constrained, Continuous  |
| Branin  (botorch) | 1  | 2  | 0  | Single Objective (is there a discrete or constrained form?)  |
| Bukin  (botorch) | 1  | 2  | 0  | Single Objective (is there a more variance?)  |
| Cantilever Step Beam  (engineering) | 1  | 10  | 11  | Single Objective, Constrained (is there a discrete form? and I think we can expand to N steps (right now it's five steps))  |

## ToDos (What have we found but not yet coded. Please add to the list):
1. BoTorch library (always check if a problem already exist here)
2. Pymoo library (always check if a problem already exist here)
3. Optimization Test Problems: https://www.sfu.ca/~ssurjano/optimization.html (always check if a problem already exist here)
4. CEC 2017 , https://www.researchgate.net/publication/317228117_Problem_Definitions_and_Evaluation_Criteria_for_the_CEC_2017_Competition_and_Special_Session_on_Constrained_Single_Objective_Real-Parameter_Optimization?enrichId=rgreq-b8f8213db02831458225d2c8ba3fe09d-XXX&enrichSource=Y292ZXJQYWdlOzMxNzIyODExNztBUzo1MjMxNjg4MDI1MDA2MDhAMTUwMTc0NDU3MDUwMQ%3D%3D&el=1_x_3&_esc=publicationCoverPdf
5. Openmdao: https://openmdao.org/newdocs/versions/latest/examples/examples.html
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

## What Rosen think would be great:
* Make all the synthetic (numerical) function scalable
* Each problem can have more than one variance among these six types: Single/Multi objective, With/Without constraint, Continuous/Mix-variable


**Reminder to Rosen: read these papers**
* Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB: https://arxiv.org/pdf/2303.00890
* Benchmarking in Optimization: Best Practice and Open Issues: https://arxiv.org/pdf/2007.03488

