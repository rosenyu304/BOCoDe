.. _lasso_benchmarks:

Lasso Benchmarks
=================

The Lasso benchmark collection is for high-dimensional hyperparameter optimization benchmarks based on Weighted Lasso regression.
More information about the functions is available in the `LassoBench <https://github.com/ksehic/LassoBench>`_ docs.

Available Problems
----------------

* :code:`optbench.LassoBench.LassoBreastCancer`
* :code:`optbench.LassoBench.LassoDiabetes`
* :code:`optbench.LassoBench.LassoDNA`
* :code:`optbench.LassoBench.LassoLeukemia`
* :code:`optbench.LassoBench.LassoRCV1`
* :code:`optbench.LassoBench.LassoSyntHard`
* :code:`optbench.LassoBench.LassoSyntHigh`
* :code:`optbench.LassoBench.LassoSyntMedium`
* :code:`optbench.LassoBench.LassoSyntSimple`

Example Usage
------------

.. code-block:: python

    import optbench
    import torch

    # Create a Botorch benchmark problem
    problem = optbench.LassoBench.LassoBreastCancer()
    
    # Get problem information
    bounds = problem.bounds
    optimum_function_value = problem.optimum
    optimum_input_value = problem.x_opt
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    constraints, values = problem._evaluate_implementation(x)
    
    print(f"Lasso Breast Cancer function value at origin: {values[0]}")

Output:

.. code-block:: console

    Lasso Breast Cancer function value at origin: tensor([-0.2626])