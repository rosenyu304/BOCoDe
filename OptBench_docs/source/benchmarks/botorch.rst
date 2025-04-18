.. _botorch_benchmarks:

Botorch Benchmarks
=================

The Botorch benchmark collection includes synthetic test problems commonly used in Bayesian optimization research.

Single Objective Problems
----------------

* :code:`optbench.BoTorch.AugmentedBranin`
* :code:`optbench.BoTorch.AugmentedHartmann`
* :code:`optbench.BoTorch.AugmentedRosenbrock`
* :code:`optbench.BoTorch.Ishigami`
* :code:`optbench.BoTorch.Gsobol`
* :code:`optbench.BoTorch.Morris`

Multi Objective Problems
----------------

* :code:`optbench.BoTorch.MOMFBraninCurrin`
* :code:`optbench.BoTorch.MOMFPark1`
* :code:`optbench.BoTorch.BraninCurrin`
* :code:`optbench.BoTorch.DH1`
* :code:`optbench.BoTorch.DH2`
* :code:`optbench.BoTorch.DH3`
* :code:`optbench.BoTorch.DH4`
* :code:`optbench.BoTorch.DTLZ1`
* :code:`optbench.BoTorch.DTLZ2`
* :code:`optbench.BoTorch.DTLZ3`
* :code:`optbench.BoTorch.DTLZ4`
* :code:`optbench.BoTorch.DTLZ5`
* :code:`optbench.BoTorch.DTLZ7`
* :code:`optbench.BoTorch.GMM`
* :code:`optbench.BoTorch.Penicillin`
* :code:`optbench.BoTorch.ToyRobust`
* :code:`optbench.BoTorch.VehicleSafety`
* :code:`optbench.BoTorch.ZDT1`
* :code:`optbench.BoTorch.ZDT2`
* :code:`optbench.BoTorch.ZDT3`
* :code:`optbench.BoTorch.CarSideImpact`
* :code:`optbench.BoTorch.BNH`
* :code:`optbench.BoTorch.CONSTR`
* :code:`optbench.BoTorch.ConstrainedBraninCurrin`
* :code:`optbench.BoTorch.C2DTLZ2`
* :code:`optbench.BoTorch.DiscBrake`
* :code:`optbench.BoTorch.MW7`
* :code:`optbench.BoTorch.OSY`
* :code:`optbench.BoTorch.SRN`
* :code:`optbench.BoTorch.WeldedBeam`

Single Objective Example Usage
------------

.. code-block:: python

    import optbench
    import torch

    # Create a Botorch benchmark problem
    problem = optbench.BoTorch.AugmentedBranin()

    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    constraints, values = problem._evaluate_implementation(x)

    print(f"AugmentedBranin function value at origin: {values[0]}")

Output:

.. code-block:: console

    AugmentedBranin function value at origin: tensor([228.4423])

Multi Objective Example Usage
------------

.. code-block:: python

    import optbench
    import torch

    # Create a Botorch benchmark problem
    problem = optbench.BoTorch.MOMFBraninCurrin()

    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    constraints, values = problem._evaluate_implementation(x)

    print(f"MOMFBraninCurrin function value at origin: {values[0]}")

Output:

.. code-block:: console

    MOMFBraninCurrin function value at origin: tensor([11.8986, -0.7333])
