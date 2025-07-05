.. _bbob_benchmarks:

BBOB Benchmarks
==============

The Black-Box Optimization Benchmarking (BBOB) collection provides a comprehensive set of continuous optimization problems.
Visit the `COCO platform <https://numbbo.github.io/coco/testsuites/bbob>`_ for a complete list of functions.

Available Suites
-----------------
- bbob: :code:`bocode.BBOB.BBOB`
- bbob-biobj: :code:`bocode.BBOB.BBOB_Biobj`
- bbob-biobj-mixint: :code:`bocode.BBOB.BBOB_BiobjMixInt`
- bbob-boxed: :code:`bocode.BBOB.BBOB_Boxed`
- bbob-constrained: :code:`bocode.BBOB.BBOB_Constrained`
- bbob-largescale: :code:`bocode.BBOB.BBOB_LargeScale`
- bbob-mixint: :code:`bocode.BBOB.BBOB_MixInt`
- bbob-noisy: :code:`bocode.BBOB.BBOB_Noisy`

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
    problem = bocode.BBOB.BBOB(dim=5, function_number=2, instance_number=1) # Separable 5-dimensional ellipsoidal function

    # Get problem information
    bounds = problem.bounds
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * 5])
    values, constraints = problem.evaluate(x)
    
    print(f"BBOB function value at origin: {values[0]}")

Output:

.. code-block:: console

    BBOB function value at origin: tensor([42420381.6772])