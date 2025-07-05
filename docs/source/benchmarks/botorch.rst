.. _botorch_benchmarks:

BoTorch Benchmarks
==================

The Botorch benchmark collection includes synthetic test problems commonly used in Bayesian optimization research.

Single Objective Problems
----------------

* :code:`bocode.BoTorch.AugmentedBranin`
* :code:`bocode.BoTorch.AugmentedHartmann`
* :code:`bocode.BoTorch.AugmentedRosenbrock`
* :code:`bocode.BoTorch.Ishigami`
* :code:`bocode.BoTorch.Gsobol`
* :code:`bocode.BoTorch.Morris`

Multi Objective Problems
----------------

* :code:`bocode.BoTorch.MOMFBraninCurrin`
* :code:`bocode.BoTorch.MOMFPark1`
* :code:`bocode.BoTorch.BraninCurrin`
* :code:`bocode.BoTorch.DH1`
* :code:`bocode.BoTorch.DH2`
* :code:`bocode.BoTorch.DH3`
* :code:`bocode.BoTorch.DH4`
* :code:`bocode.BoTorch.DTLZ1`
* :code:`bocode.BoTorch.DTLZ2`
* :code:`bocode.BoTorch.DTLZ3`
* :code:`bocode.BoTorch.DTLZ4`
* :code:`bocode.BoTorch.DTLZ5`
* :code:`bocode.BoTorch.DTLZ7`
* :code:`bocode.BoTorch.GMM`
* :code:`bocode.BoTorch.Penicillin`
* :code:`bocode.BoTorch.ToyRobust`
* :code:`bocode.BoTorch.VehicleSafety`
* :code:`bocode.BoTorch.ZDT1`
* :code:`bocode.BoTorch.ZDT2`
* :code:`bocode.BoTorch.ZDT3`
* :code:`bocode.BoTorch.CarSideImpact`
* :code:`bocode.BoTorch.BNH`
* :code:`bocode.BoTorch.CONSTR`
* :code:`bocode.BoTorch.ConstrainedBraninCurrin`
* :code:`bocode.BoTorch.C2DTLZ2`
* :code:`bocode.BoTorch.DiscBrake`
* :code:`bocode.BoTorch.MW7`
* :code:`bocode.BoTorch.OSY`
* :code:`bocode.BoTorch.SRN`
* :code:`bocode.BoTorch.WeldedBeam`

Single Objective Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Create a BoTorch benchmark problem
    problem = bocode.BoTorch.AugmentedBranin()

    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    values, constraints = problem.evaluate(x)

    print(f"AugmentedBranin function value at origin: {values[0]}")

Output:

.. code-block:: console

    AugmentedBranin function value at origin: tensor([228.4423])

Multi Objective Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Create a BoTorch benchmark problem
    problem = bocode.BoTorch.MOMFBraninCurrin()

    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    values, constraints = problem.evaluate(x)

    print(f"MOMFBraninCurrin function value at origin: {values[0]}")

Output:

.. code-block:: console

    MOMFBraninCurrin function value at origin: tensor([11.8986, -0.7333])
