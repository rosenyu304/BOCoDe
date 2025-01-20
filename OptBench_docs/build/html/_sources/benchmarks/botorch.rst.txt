.. _botorch_benchmarks:

Botorch Benchmarks
=================

The Botorch benchmark collection includes synthetic test problems commonly used in Bayesian optimization research.

Available Problems
----------------

* Ackley Function
* Levy Function
* Rosenbrock Function
* Sphere Function

Example Usage
------------

.. code-block:: python

    import optbench

    # Create a Botorch benchmark problem
    problem = optbench.botorch.create_problem("ackley", dim=2)
    
    # Evaluate at a point
    x = [0.0, 0.0]
    value = problem.evaluate(x)
    
    print(f"Ackley function value at origin: {value}")