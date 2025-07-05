.. _neorl_benchmarks:

NEORL Benchmarks
=================

The NEORL (NeuroEvolution Optimization with Reinforcement Learning) benchmark collection of various mathematical and engineering problems.
More information about the functions is available in the `NEORL GitHub Page <https://github.com/aims-umich/neorl/tree/master/docs/source/examples>`_.

Source:

 M. I. Radaideh, K. Du, P. Seurin, D. Seyler, X. Gu, H. Wang, and K. Shirvan, “NEORL: NeuroEvolution Optimization with Reinforcement Learning—Applications to carbon-free energy systems,” Nuclear Engineering and Design, vol. 412, p. 112423, 2023.

Available Problems
----------------

* :code:`bocode.NEORL.TSP_51Cities` (Ex 1)
* :code:`bocode.NEORL.TSP_100Cities` (Ex 1)
* :code:`bocode.NEORL.ReactivityModel` (Ex 11)
* :code:`bocode.NEORL.QPowerModel` (Ex 11)

Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Create a NEORL benchmark problem
    problem = bocode.NEORL.TSP_100Cities()

    # Get problem information
    bounds = problem.bounds  # Each input is an integer between 1 and 100

    # Get the optimum input value
    optimum_input_value = problem.x_opt[0]

    # Generate a random test point
    X = torch.rand((1, problem.dim))

    # Scale to integers between 1 and 100
    X = (X * 100).long() + 1

    values, constraints = problem.evaluate(X)

    print(f"Travelling Salesman Problem (TSP) with 100 cities function value at random point: {values[0]}")

Example Output:

.. code-block:: console

    Travelling Salesman Problem (TSP) with 100 cities function value at random point: tensor([-5540.])