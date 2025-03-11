.. _engineering_benchmarks:

Engineering Benchmarks
=================

The Engineering benchmark collection contains various Engineering-related functions.

Available Problems
----------------

* :code:`optbench.Engineering.CarSideImpact`
* :code:`optbench.Engineering.EulerBernoulliBeamBending`
* :code:`optbench.Engineering.GearTrain`
* :code:`optbench.Engineering.Mazda_SCA`
* :code:`optbench.Engineering.Mazda`
* :code:`optbench.Engineering.MOPTA08Car`
* :code:`optbench.Engineering.RobotPush`
* :code:`optbench.Engineering.Rover`
* :code:`optbench.Engineering.Truss10D`
* :code:`optbench.Engineering.Truss25D`
* :code:`optbench.Engineering.TwoBarTruss`
* :code:`optbench.Engineering.WaterProblem`
* :code:`optbench.Engineering.WaterResources`

Example Usage
------------

.. code-block:: python

    import optbench
    import torch

    # Create a Botorch benchmark problem
    problem = optbench.Engineering.GearTrain()
    
    # Get problem information
    bounds = problem.bounds
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    constraints, values = problem._evaluate_implementation(x)
    
    print(f"Gear Train function value at [0.5]*4: {values[0]}")

Output:

.. code-block:: console

    Gear Train function value at [0.5]*4: tensor([-0.7323])