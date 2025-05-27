.. _zdt_benchmarks:

ZDT Benchmarks
=================

The ZDT benchmark collection contains all functions from the ZDT (Zitzler, Deb, and Thiele) benchmark suite. Python implementation is derived from the `optproblems python library <https://www.simonwessing.de/optproblems/doc/zdt.html>`_.

Sources:

 Zitzler, E., Deb, K., and Thiele, L. (2000). Comparison of Multiobjective Evolutionary Algorithms: Empirical Results. Evolutionary Computation 8(2).

Available Problems
----------------

* :code:`optbench.ZDT.ZDT1`
* :code:`optbench.ZDT.ZDT2`
* :code:`optbench.ZDT.ZDT3`
* :code:`optbench.ZDT.ZDT4`
* :code:`optbench.ZDT.ZDT5`
    * Accepts 80 bits as input, automatically splitting it into the necessary sublists. See example below.
* :code:`optbench.ZDT.ZDT6`

Example Usage
------------

.. code-block:: python

    import optbench
    import torch

    # Retrieve available dimensions for instantiation
    available_dimensions = optbench.ZDT.ZDT1.available_dimensions

    # Create a Botorch benchmark problem
    problem = optbench.ZDT.ZDT1(dim=5)

    # Get problem information
    bounds = problem.bounds

    # Evaluate at a point
    x = torch.Tensor([[0.5] * problem.dim])
    constraints, values = problem._evaluate_implementation(x)

    print(f"First ZDT function values at [0.5]*5: {values[0]}")

Output:

.. code-block:: console

    First ZDT function values at [0.5]*5: tensor([0.5000, 3.8417])

.. _ref-zdt5:
Example Usage of ZDT5
------------
.. code-block:: python

    import optbench
    import torch

    # Create a Botorch benchmark problem
    problem = optbench.ZDT.ZDT5()

    # Get problem information
    bounds = problem.bounds

    # Evaluate using 80 random bits of 0s and 1s
    x = torch.randint(0, 2, (1, 80))
    constraints, values = problem._evaluate_implementation(x)

    print(f"ZDT5 function values at x: {values[0]}")

Output:

.. code-block:: console

    ZDT5 function values at x: tensor([10.0000,  4.5000])