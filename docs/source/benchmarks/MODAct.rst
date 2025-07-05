.. _modact_benchmarks:

MODAct Benchmarks
=================

The `MODAct <https://github.com/epfl-lamd/modact>`_ (Multi-Objective Design of electro-mechanical Actuators) benchmark collection includes 20 benchmark problems for constrained multi-objective optimization.

Available Problems
----------------

* :code:`bocode.MODAct.CS1`
* :code:`bocode.MODAct.CT1`
* :code:`bocode.MODAct.CTS1`
* :code:`bocode.MODAct.CTSE1`
* :code:`bocode.MODAct.CTSEI1`
* :code:`bocode.MODAct.CS2`
* :code:`bocode.MODAct.CT2`
* :code:`bocode.MODAct.CTS2`
* :code:`bocode.MODAct.CTSE2`
* :code:`bocode.MODAct.CTSEI2`
* :code:`bocode.MODAct.CS3`
* :code:`bocode.MODAct.CT3`
* :code:`bocode.MODAct.CTS3`
* :code:`bocode.MODAct.CTSE3`
* :code:`bocode.MODAct.CTSEI3`
* :code:`bocode.MODAct.CS4`
* :code:`bocode.MODAct.CT4`
* :code:`bocode.MODAct.CTS4`
* :code:`bocode.MODAct.CTSE4`
* :code:`bocode.MODAct.CTSEI4`

Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Create a MODAct benchmark problem
    problem = bocode.MODAct.CS1()

    # Evaluate at a point
    x = torch.tensor([[0.5] * problem.dim])
    values, constraints = problem.evaluate(x)

    print(f"CS1 objective function values at [0.5]*dim: {values[0]}")

Output:

.. code-block:: console

    CS1 objective function values at [0.5]*dim: tensor([0.3887, -50.4243])
