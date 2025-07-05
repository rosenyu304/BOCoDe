.. _singleobj_userguide:


Single Objective Optimization
=========================

This guide demonstrates how to use BOCoDe for single-objective optimization problems.

Basic Example
------------

Here's a complete example of using BOCoDe with a single-objective optimization problem:

.. code-block:: python

    import bocode
    import numpy as np
    import torch
    from scipy.optimize import minimize

    # Create a benchmark problem
    problem = bocode.Synthetics.Michalewicz(dim=2)

    problem.visualize_function()

    # Get problem bounds
    bounds = problem.bounds

    # Define objective function for optimizer
    def objective(x):
        x = torch.Tensor([x])
        fx, _ = problem.evaluate(x)
        fx = -fx # Negate the objective function for MINIMIZATION
        return fx.numpy()[0][0]

    # Starting point (2-dimensional)
    x0 = np.zeros(2)

    # Optimize using SciPy
    result = minimize(objective, x0, method='Powell', bounds=bounds)

    print(f"Optimal value found: {result.fun}")
    print(f"Optimal point found: {result.x}")

    print(f"Actual optimal value: {-problem.optimum[0]}")
    print(f"Actual optimal point: {problem.x_opt[0]}")

Advanced Features
---------------

* Custom callback functions
* Progress tracking
* Multiple starting points
* Integration with other optimization frameworks