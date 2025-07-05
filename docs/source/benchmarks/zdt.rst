.. _zdt_benchmarks:

ZDT Benchmarks
=================

The ZDT (Zitzler, Deb, and Thiele) benchmark collection contains all functions from the ZDT benchmark suite. Python implementation is derived from the `optproblems python library <https://www.simonwessing.de/optproblems/doc/zdt.html>`_.

Sources:

 Zitzler, E., Deb, K., and Thiele, L. (2000). Comparison of Multiobjective Evolutionary Algorithms: Empirical Results. Evolutionary Computation 8(2).

Available Problems
----------------

* :code:`bocode.ZDT.ZDT1`
* :code:`bocode.ZDT.ZDT2`
* :code:`bocode.ZDT.ZDT3`
* :code:`bocode.ZDT.ZDT4`
* :code:`bocode.ZDT.ZDT5`
    * ZDT 5 accepts 80 bits as input, automatically splitting it into the necessary sublists. See example below.
* :code:`bocode.ZDT.ZDT6`

Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Retrieve available dimensions for instantiation
    available_dimensions = bocode.ZDT.ZDT1.available_dimensions

    # Create a ZDT benchmark problem
    problem = bocode.ZDT.ZDT1(dim=5)

    # Get problem information
    bounds = problem.bounds

    # Evaluate at a point
    x = torch.Tensor([[0.5] * problem.dim])
    values, constraints = problem.evaluate(x)

    print(f"First ZDT function values at [0.5]*5: {values[0]}")

Output:

.. code-block:: console

    First ZDT function values at [0.5]*5: tensor([0.5000, 3.8417])

.. _ref-zdt5:

Example Usage of ZDT5
------------
.. code-block:: python

    import bocode
    import torch

    # Create a ZDT5 benchmark problem
    problem = bocode.ZDT.ZDT5()

    # Get problem information
    bounds = problem.bounds

    # Evaluate using 80 random bits of 0s and 1s
    x = torch.randint(0, 2, (1, 80))
    values, constraints = problem.evaluate(x)

    print(f"ZDT5 function values at x: {values[0]}")

Output:

.. code-block:: console

    ZDT5 function values at x: tensor([10.0000,  4.5000])