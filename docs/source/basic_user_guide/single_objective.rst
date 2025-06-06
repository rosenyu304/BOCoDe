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
    import torch
    from scipy.optimize import minimize

    # Create a benchmark problem
    problem = optbench.Synthetics.Michalewicz()

    # Get problem bounds
    bounds = problem.bounds

    # Define objective function for optimizer
    def objective(x):
        x = torch.Tensor([x])
        _, fx = problem._evaluate_implementation(x)
        return fx.numpy()[0][0]

    # Starting point 2 dimensional
    x0 = np.zeros(2)

    # Optimize using SciPy
    result = minimize(objective, x0, method='Powell', bounds=bounds)

    print(f"Optimal value found: {result.fun}")
    print(f"Optimal point: {result.x}")

Advanced Features
---------------

* Custom callback functions
* Progress tracking
* Multiple starting points
* Integration with other optimization frameworks