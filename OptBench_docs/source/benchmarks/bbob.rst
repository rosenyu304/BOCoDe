.. _bbob_benchmarks:

BBOB Benchmarks
==============

The Black-Box Optimization Benchmarking (BBOB) collection provides a comprehensive set of continuous optimization problems.

Available Functions
-----------------

* Sphere Function
* Rastrigin Function
* Schwefel Function
* Griewank Function

Example Usage
------------

.. code-block:: python

    import optbench

    # Create a BBOB benchmark problem
    problem = optbench.bbob.create_problem("sphere", dim=5)
    
    # Get problem information
    bounds = problem.bounds
    
    # Evaluate at a point
    x = [0.0] * 5
    value = problem.evaluate(x)
    
    print(f"Sphere function value at origin: {value}")