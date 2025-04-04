.. _botorch_benchmarks:

Synthetic Benchmarks
=================

The Botorch benchmark collection includes synthetic test problems commonly used in Bayesian optimization research.

Available Problems
----------------

* :code:`optbench.Synthetics.Ackley`
* :code:`optbench.Synthetics.Bukin`
* :code:`optbench.Synthetics.DixonPrice`
* :code:`optbench.Synthetics.Goldstein`
* :code:`optbench.Synthetics.Goldstein_Discrete`
* :code:`optbench.Synthetics.Griewank`
* :code:`optbench.Synthetics.Levy`
* :code:`optbench.Synthetics.Michalewicz`
* :code:`optbench.Synthetics.Powell`
* :code:`optbench.Synthetics.Rastrigin`
* :code:`optbench.Synthetics.Rosenbrock`
* :code:`optbench.Synthetics.Styblinski-Tang`

Example Usage
------------

.. code-block:: python

    import optbench
    import torch

    # Create a Botorch benchmark problem
    problem = optbench.Synthetics.Goldstein_Discrete()
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    constraints, values = problem._evaluate_implementation(x)
    
    print(f"Goldstein Discrete function value at origin: {values[0]}")

Output:

.. code-block:: console

    Goldstein Discrete function value at origin: tensor([-600.])