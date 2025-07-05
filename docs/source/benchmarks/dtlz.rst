.. _dtlz_benchmarks:

DTLZ Benchmarks
=================

The DTLZ (Deb, Thiele, Laumanns, Zitzler) benchmark collection contains all functions from the DTLZ benchmark suite. Python implementation of the original C++ code is derived from the `optproblems python library <https://www.simonwessing.de/optproblems/doc/wfg.html>`_.

Sources:

(1) Deb, K.; Thiele, L.; Laumanns, M.; Zitzler, E. (2001). Scalable Test Problems for Evolutionary Multi-Objective Optimization, Technical Report, Computer Engineering and Networks Laboratory (TIK), Swiss Federal Institute of Technology (ETH). https://dx.doi.org/10.3929/ethz-a-004284199
(2) Deb, K.; Thiele, L.; Laumanns, M.; Zitzler, E. (2002). Scalable multi-objective optimization test problems, Proceedings of the IEEE Congress on Evolutionary Computation, pp. 825-830

Available Problems
----------------

* :code:`bocode.DTLZ.DTLZ1`
* :code:`bocode.DTLZ.DTLZ2`
* :code:`bocode.DTLZ.DTLZ3`
* :code:`bocode.DTLZ.DTLZ4`
* :code:`bocode.DTLZ.DTLZ5`
* :code:`bocode.DTLZ.DTLZ6`
* :code:`bocode.DTLZ.DTLZ7`

Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Retrieve available dimensions for instantiation
    available_dimensions = bocode.DTLZ.DTLZ1.available_dimensions

    # Create a DTLZ benchmark problem
    problem = bocode.DTLZ.DTLZ1(dim=10)

    # Get problem information
    bounds = problem.bounds

    # Evaluate at a point
    x = torch.Tensor([[0.5] * problem.dim])
    values, constraints = problem.evaluate(x)

    print(f"First DTLZ function values at [0.5]*dim: {values[0]}")

Output:

.. code-block:: console

    First DTLZ function values at [0.5]*dim: tensor([0.2500, 0.2500])