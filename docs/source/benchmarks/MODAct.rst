.. _modact_benchmarks:

MODAct Benchmarks
=================

The `MODAct <https://github.com/epfl-lamd/modact>`_ (Multi-Objective Design of electro-mechanical Actuators) benchmark collection includes 20 benchmark problems for constrained multi-objective optimization.

Available Problems
----------------

* :code:`bocode.Engineering.MODAct.CS1`
* :code:`bocode.Engineering.MODAct.CT1`
* :code:`bocode.Engineering.MODAct.CTS1`
* :code:`bocode.Engineering.MODAct.CTSE1`
* :code:`bocode.Engineering.MODAct.CTSEI1`
* :code:`bocode.Engineering.MODAct.CS2`
* :code:`bocode.Engineering.MODAct.CT2`
* :code:`bocode.Engineering.MODAct.CTS2`
* :code:`bocode.Engineering.MODAct.CTSE2`
* :code:`bocode.Engineering.MODAct.CTSEI2`
* :code:`bocode.Engineering.MODAct.CS3`
* :code:`bocode.Engineering.MODAct.CT3`
* :code:`bocode.Engineering.MODAct.CTS3`
* :code:`bocode.Engineering.MODAct.CTSE3`
* :code:`bocode.Engineering.MODAct.CTSEI3`
* :code:`bocode.Engineering.MODAct.CS4`
* :code:`bocode.Engineering.MODAct.CT4`
* :code:`bocode.Engineering.MODAct.CTS4`
* :code:`bocode.Engineering.MODAct.CTSE4`
* :code:`bocode.Engineering.MODAct.CTSEI4`

Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Create a MODAct benchmark problem
    problem = bocode.Engineering.MODAct.CS1()

    # Evaluate at a point
    x = torch.tensor([[0.5] * problem.dim])
    values, constraints = problem.evaluate(x)

    print(f"CS1 objective function values at [0.5]*dim: {values[0]}")

Output:

.. code-block:: console

    CS1 objective function values at [0.5]*dim: tensor([0.3887, -50.4243])
