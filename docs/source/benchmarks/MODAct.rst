.. _modact_benchmarks:

MODAct Benchmarks
=================

The `MODAct <https://github.com/epfl-lamd/modact>`_ (Multi-Objective Design of electro-mechanical Actuators) benchmark collection includes 20 benchmark problems for constrained multi-objective optimization.

Available Problems
----------------

* :code:`bocode.CS1`
* :code:`bocode.CT1`
* :code:`bocode.CTS1`
* :code:`bocode.CTSE1`
* :code:`bocode.CTSEI1`
* :code:`bocode.CS2`
* :code:`bocode.CT2`
* :code:`bocode.CTS2`
* :code:`bocode.CTSE2`
* :code:`bocode.CTSEI2`
* :code:`bocode.CS3`
* :code:`bocode.CT3`
* :code:`bocode.CTS3`
* :code:`bocode.CTSE3`
* :code:`bocode.CTSEI3`
* :code:`bocode.CS4`
* :code:`bocode.CT4`
* :code:`bocode.CTS4`
* :code:`bocode.CTSE4`
* :code:`bocode.CTSEI4`

Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Create a MODAct benchmark problem
    problem = bocode.CS1()

    # Evaluate at a point
    x = torch.tensor([[0.5] * problem.dim])
    values, constraints = problem.evaluate(x)

    print(f"CS1 objective function values at [0.5]*dim: {values[0]}")

Output:

.. code-block:: console

    CS1 objective function values at [0.5]*dim: tensor([0.3887, -50.4243])
