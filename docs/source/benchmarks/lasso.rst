.. _lasso_benchmarks:

LassoBench Benchmarks
=====================

The LassoBench collection provides high-dimensional hyperparameter optimization benchmarks based on Weighted Lasso regression.

More information about the functions is available in the `LassoBench <https://github.com/ksehic/LassoBench>`_ documentation.

Available Problems
----------------

* :code:`bocode.LassoBench.LassoBreastCancer`
* :code:`bocode.LassoBench.LassoDiabetes`
* :code:`bocode.LassoBench.LassoDNA`
* :code:`bocode.LassoBench.LassoLeukemia`
* :code:`bocode.LassoBench.LassoRCV1`
* :code:`bocode.LassoBench.LassoSyntHard`
* :code:`bocode.LassoBench.LassoSyntHigh`
* :code:`bocode.LassoBench.LassoSyntMedium`
* :code:`bocode.LassoBench.LassoSyntSimple`

Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Create a LassoBench problem
    problem = bocode.LassoBench.LassoBreastCancer()
    
    # Get problem information
    bounds = problem.bounds
    optimum_function_value = problem.optimum
    optimum_input_value = problem.x_opt
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    values, constraints = problem.evaluate(x)
    
    print(f"Lasso Breast Cancer function value at origin: {values[0]}")

Output:

.. code-block:: console

    Lasso Breast Cancer function value at origin: tensor([-0.2626])