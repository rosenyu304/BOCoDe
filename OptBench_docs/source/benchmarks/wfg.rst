.. _wfg_benchmarks:

WFG Benchmarks
=================

The WFG benchmark collection contains all functions from the WFG (Walking Fish Group) benchmark suite. Python implementation of the original C++ code is derived from the `optproblems python library <https://www.simonwessing.de/optproblems/doc/wfg.html>`_.

Sources:

 Huband, S.; Hingston, P.; Barone, L.; While, L. (2006). A review of multiobjective test problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation, vol.10, no.5, pp. 477-506.

Available Problems
----------------

* :code:`optbench.WFG.WFG1`
* :code:`optbench.WFG.WFG2`
* :code:`optbench.WFG.WFG3`
* :code:`optbench.WFG.WFG4`
* :code:`optbench.WFG.WFG5`
* :code:`optbench.WFG.WFG6`
* :code:`optbench.WFG.WFG7`
* :code:`optbench.WFG.WFG8`
* :code:`optbench.WFG.WFG9`

Example Usage
------------

.. code-block:: python

    import optbench
    import torch

    # Retrieve available dimensions for instantiation
    available_dimensions = optbench.WFG.WFG1.available_dimensions

    # Create a Botorch benchmark problem
    problem = optbench.WFG.WFG1(dim=5)

    # Get problem information
    bounds = problem.bounds

    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    constraints, values = problem._evaluate_implementation(x)

    print(f"First WFG function values at origin: {values[0]}")

Output:

.. code-block:: console

    First WFG function values at origin: tensor([1., 5.])