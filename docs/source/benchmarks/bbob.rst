.. _bbob_benchmarks:

BBOB Benchmarks
==============

The Black-Box Optimization Benchmarking (BBOB) collection provides a comprehensive set of continuous optimization problems.
Visit the `COCO platform <https://numbbo.github.io/coco/testsuites/bbob>`_ for a complete list of functions.

Available Suites
-----------------
- bbob: :code:`bocode.BBOB_Problems.BBOB`
- bbob-biobj: :code:`bocode.BBOB_Problems.BBOB_Biobj`
- bbob-biobj-mixint: :code:`bocode.BBOB_BiobjMixInt`
- bbob-boxed: :code:`bocode.BBOB_Problems.BBOB_Boxed`
- bbob-constrained: :code:`bocode.BBOB_Problems.BBOB_Constrained`
- bbob-largescale: :code:`bocode.BBOB_Problems.BBOB_LargeScale`
- bbob-mixint: :code:`bocode.BBOB_Problems.BBOB_MixInt`
- bbob-noisy: :code:`bocode.BBOB_Problems.BBOB_Noisy`

Available Functions
-----------------

* Sphere Function
* Rastrigin Function
* Schwefel Function
* Griewank Function



Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Create a BBOB benchmark problem
    problem = bocode.BBOB_Problems.BBOB(dim=5, function_number=2, instance_number=1) # Separable 5-dimensional ellipsoidal function
    
    # Get problem information
    bounds = problem.bounds
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * 5])
    constraints, values = problem._evaluate_implementation(x)
    
    print(f"Sphere function value at origin: {values[0]}")

Output:

.. code-block:: console

    Sphere function value at origin: tensor([42420381.6772])