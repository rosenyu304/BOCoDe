.. _synthetics_benchmarks:

Synthetic Benchmarks
===================

The Synthetic benchmark collection includes classic mathematical test functions commonly used in optimization research.

Available Problems
----------------

* :code:`bocode.Synthetics.Ackley`
* :code:`bocode.Synthetics.Bukin`
* :code:`bocode.Synthetics.DixonPrice`
* :code:`bocode.Synthetics.Goldstein`
* :code:`bocode.Synthetics.Goldstein_Discrete`
* :code:`bocode.Synthetics.Griewank`
* :code:`bocode.Synthetics.Levy`
* :code:`bocode.Synthetics.Michalewicz`
* :code:`bocode.Synthetics.Powell`
* :code:`bocode.Synthetics.Rastrigin`
* :code:`bocode.Synthetics.Rosenbrock`
* :code:`bocode.Synthetics.Styblinski-Tang`
* :code:`bocode.Synthetics.Beale`
* :code:`bocode.Synthetics.Cosine8`
* :code:`bocode.Synthetics.DropWave`
* :code:`bocode.Synthetics.EggHolder`
* :code:`bocode.Synthetics.Hartmann3D`
* :code:`bocode.Synthetics.Hartmann6D`
* :code:`bocode.Synthetics.HolderTable`
* :code:`bocode.Synthetics.Shekelm5`
* :code:`bocode.Synthetics.Shekelm7`
* :code:`bocode.Synthetics.Shekelm10`
* :code:`bocode.Synthetics.Shekel`
* :code:`bocode.Synthetics.SixHumpCamel`
* :code:`bocode.Synthetics.ThreeHumpCamel`
* :code:`bocode.Synthetics.ConstrainedGramacy`
* :code:`bocode.Synthetics.ConstrainedHartmann`
* :code:`bocode.Synthetics.ConstrainedHartmannSmooth`
* :code:`bocode.Synthetics.PressureVessel`
* :code:`bocode.Synthetics.WeldedBeamSO`
* :code:`bocode.Synthetics.TensionCompressionString`
* :code:`bocode.Synthetics.SpeedReducer`
* :code:`bocode.Synthetics.SVM`

Example Usage
------------

.. code-block:: python

    import bocode
    import torch

    # Create a synthetic benchmark problem
    problem = bocode.Synthetics.Goldstein_Discrete()
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    values, constraints = problem.evaluate(x)
    
    print(f"Goldstein Discrete function value at origin: {values[0]}")

Output:

.. code-block:: console

    Goldstein Discrete function value at origin: tensor([-600.])