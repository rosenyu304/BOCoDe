.. _bbob_benchmarks:

BBOB Benchmarks
==============

The Black-Box Optimization Benchmarking (BBOB) collection provides a comprehensive set of continuous optimization problems.

Available Suites
-----------------
- bbob: :code:`optbench.BBOB`
- bbob-biobj: :code:`optbench.BBOB_Biobj`
- bbob-biobj-mixint: :code:`optbench.BBOB_BiobjMixInt`
- bbob-boxed: :code:`optbench.BBOB_Boxed`
- bbob-constrained: :code:`optbench.BBOB_Constrained`
- bbob-largescale: :code:`optbench.BBOB_LargeScale`
- bbob-mixint: :code:`optbench.BBOB_MixInt`
- bbob-noisy: :code:`optbench.BBOB_Noisy`

Available Functions
-----------------

* Sphere Function
* Rastrigin Function
* Schwefel Function
* Griewank Function

Visit https://numbbo.github.io/coco/testsuites/bbob for a complete list of functions.

Example Usage
------------

.. code-block:: python

    import optbench
    import torch

    # Create a BBOB benchmark problem
    problem = optbench.BBOB(dim=5, function_number=2, instance_number=1) # Separable 5-dimensional ellipsoidal function
    
    # Get problem information
    bounds = problem.bounds
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * 5])
    constraints, values = problem._evaluate_implementation(x)
    
    print(f"Sphere function value at origin: {values[0]}")

Output:

.. code-block:: console

    Sphere function value at origin: tensor([42420381.6772], dtype=torch.float64)