.. _singleobj_userguide:


Single Objective Optimization
=========================

This guide demonstrates how to use OptBench for single-objective optimization problems.

Basic Example
------------

Here's a complete example of using OptBench with a single-objective optimization problem:

.. code-block:: python

    import optbench
    import numpy as np
    from scipy.optimize import minimize

    # Create a benchmark problem
    problem = optbench.create_problem("sphere", dim=2)

    # Get problem bounds
    bounds = problem.bounds

    # Define objective function for optimizer
    def objective(x):
        return problem.evaluate(x)

    # Starting point
    x0 = np.zeros(2)

    # Optimize using SciPy
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    print(f"Optimal value found: {result.fun}")
    print(f"Optimal point: {result.x}")

Advanced Features
---------------

* Custom callback functions
* Progress tracking
* Multiple starting points
* Integration with other optimization frameworks