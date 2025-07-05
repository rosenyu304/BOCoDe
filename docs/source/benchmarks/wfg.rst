.. _wfg_benchmarks:

WFG Benchmarks
=================

The WFG (Walking Fish Group) benchmark collection contains all functions from the WFG benchmark suite. Python implementation of the original C++ code is derived from the `optproblems python library <https://www.simonwessing.de/optproblems/doc/wfg.html>`_.

Sources:

 Huband, S.; Hingston, P.; Barone, L.; While, L. (2006). A review of multiobjective test problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation, vol.10, no.5, pp. 477-506.

Available Problems
----------------

* :code:`bocode.WFG.WFG1`
* :code:`bocode.WFG.WFG2`
* :code:`bocode.WFG.WFG3`
* :code:`bocode.WFG.WFG4`
* :code:`bocode.WFG.WFG5`
* :code:`bocode.WFG.WFG6`
* :code:`bocode.WFG.WFG7`
* :code:`bocode.WFG.WFG8`
* :code:`bocode.WFG.WFG9`

Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Retrieve available dimensions for instantiation
    available_dimensions = bocode.WFG.WFG1.available_dimensions

    # Create a WFG benchmark problem
    problem = bocode.WFG.WFG1(dim=5)

    # Get problem information
    bounds = problem.bounds

    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    values, constraints = problem.evaluate(x)

    print(f"First WFG function values at origin: {values[0]}")

Output:

.. code-block:: console

    First WFG function values at origin: tensor([1., 5.])