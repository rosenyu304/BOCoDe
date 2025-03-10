# March 10 Notes:
To-Dos:
1. Add Cyril's MODAct to Engineering: See Kailey/MODAct.py and the original implementation: https://github.com/epfl-lamd/modact. As the original MODAct support PyMOO framework, the objective is set to be minimization. However, here we want maximization, so double-check this logic while implementation:
```
# Pymoo: f = np.array(f)*-1*cs1.weights
f = np.array(f)*self.prob.weights # BO is maximizing
g = np.array(g)*self.prob.c_weights
```
We would like to see the full implementation of all MODAct problems as multi-objective problems. Confirm the implemetation with original paper: https://ieeexplore.ieee.org/document/9179777

2. Add all BoTorch to BoTorch: See https://github.com/pytorch/botorch/tree/main/botorch/test_functions. Think about whether we want to follow their directory structure when categorizing them

3. Add these paper's functions to Engineering: I think some of it might require ABAQUS or MATLAB. See if you can convert these code to Python using ChatGPT and let us know if you need access to any of these softwares. 
- "MACHINE LEARNING-GUIDED DESIGN OF NON-RECIPROCAL AND ASYMMETRIC ELASTIC CHIRAL METAMATERIALS": https://arxiv.org/pdf/2404.13215
- "Surrogate-assisted constraint-handling technique for parametric multi-objective optimization": https://link.springer.com/article/10.1007/s00158-024-03859-y

4. Add "Vectorized BBOB": As you might see BBOB's optimization function is not vectorized as it can only process one x -> f(x) at a time. Write the torch version for the 24 function here (https://coco-platform.org/testsuites/bbob/overview.html) with their first instance. Consider storing them in a separate folder (maybe named BBOB_Vectorized?) 


# Feb 5 Notes
1. The OptBenchmarksLibrary/base.py have the BenchmarkProblem class
2. The actual f(x) in the most commonly used continuous version should be be a subclass of BenchmarkProblem
3. The other variant of f(x) should be a subclass of the actual f(x)
