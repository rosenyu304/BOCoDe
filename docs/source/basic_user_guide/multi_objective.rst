.. _multiobj_userguide:


Multi-Objective Optimization
========================

This guide shows how to work with multi-objective optimization problems in BOCoDe.

Basic Example
------------

Here's an example of using BOCoDe with a multi-objective problem:

.. code-block:: python

    import bocode
    import numpy as np
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.core.problem import Problem
    import torch

    # Create a multi-objective benchmark problem
    problem = bocode.Engineering.CarSideImpact()

    # Wrap the problem as a PyMOO Problem
    class CarSideImpactProblem(Problem):
        def __init__(self):
            super().__init__(n_var=problem.dim,
                            n_obj=problem.num_objectives,
                            n_constr=problem.num_constraints,
                            xl=np.array([b[0] for b in problem.bounds], dtype=float),
                            xu=np.array([b[1] for b in problem.bounds], dtype=float),
                            )
        
        def _evaluate(self, x, out, *args, **kwargs):
            values, constraints = problem.evaluate(torch.Tensor(x), scaling=False)
            out["F"] = values.numpy()
            out["G"] = constraints.numpy()

    # Setup the algorithm
    algorithm = NSGA2(pop_size=100)

    # Optimize
    res = minimize(CarSideImpactProblem(),
                  algorithm,
                  verbose=True)

    # Get Pareto front
    pareto_front = res.F

Features for Multi-Objective Problems
----------------------------------

* Pareto front visualization
* Performance metrics (hypervolume, IGD)
* Constraint handling
* Integration with multi-objective optimization frameworks