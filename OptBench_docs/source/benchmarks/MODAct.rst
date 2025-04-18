.. _modact_benchmarks:

MODAct (Multi-Objective Design of electro-mechanical Actuators) Benchmarks
=================

The `MODAct <https://github.com/epfl-lamd/modact>`_ benchmark collection includes 20 benchmark problems for constrained multi-objective optimization.

Available Problems
----------------

* :code:`optbench.MODAct.CS1`
* :code:`optbench.MODAct.CT1`
* :code:`optbench.MODAct.CTS1`
* :code:`optbench.MODAct.CTSE1`
* :code:`optbench.MODAct.CTSEI1`
* :code:`optbench.MODAct.CS2`
* :code:`optbench.MODAct.CT2`
* :code:`optbench.MODAct.CTS2`
* :code:`optbench.MODAct.CTSE2`
* :code:`optbench.MODAct.CTSEI2`
* :code:`optbench.MODAct.CS3`
* :code:`optbench.MODAct.CT3`
* :code:`optbench.MODAct.CTS3`
* :code:`optbench.MODAct.CTSE3`
* :code:`optbench.MODAct.CTSEI3`
* :code:`optbench.MODAct.CS4`
* :code:`optbench.MODAct.CT4`
* :code:`optbench.MODAct.CTS4`
* :code:`optbench.MODAct.CTSE4`
* :code:`optbench.MODAct.CTSEI4`

Example Usage
------------

.. code-block:: python

    import optbench
    import torch

    # Create a Botorch benchmark problem
    problem = optbench.MODAct.CS1()

    # Evaluate at a point
    x = torch.Tensor([[0.5]*problem.dim])
    constraints, values = problem._evaluate_implementation(x)

    print(f"CS1 objective function values at origin: {values[0]}")

Output:

.. code-block:: console

    CS1 objective function values at origin: tensor([0.3887, -50.4243])