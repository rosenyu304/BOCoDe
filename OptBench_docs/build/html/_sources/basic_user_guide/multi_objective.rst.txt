.. _multiobj_userguide:


Multi-Objective Optimization
========================

This guide shows how to work with multi-objective optimization problems in OptBench.

Basic Example
------------

Here's an example of using OptBench with a multi-objective problem:

.. code-block:: python

    import optbench
    import numpy as np
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    # Create a multi-objective benchmark problem
    problem = optbench.create_problem("zdt1")

    # Setup the algorithm
    algorithm = NSGA2(pop_size=100)

    # Optimize
    res = minimize(problem,
                  algorithm,
                  ('n_gen', 200),
                  verbose=True)

    # Get Pareto front
    pareto_front = res.F

Features for Multi-Objective Problems
----------------------------------

* Pareto front visualization
* Performance metrics (hypervolume, IGD)
* Constraint handling
* Integration with multi-objective optimization frameworks