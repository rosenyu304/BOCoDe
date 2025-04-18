.. _synthetics_benchmarks:

Synthetic Benchmarks
=================

The Botorch benchmark collection includes synthetic test problems commonly used in Bayesian optimization research.

Available Problems
----------------

* :code:`optbench.Synthetics.Ackley`
* :code:`optbench.Synthetics.Bukin`
* :code:`optbench.Synthetics.DixonPrice`
* :code:`optbench.Synthetics.Goldstein`
* :code:`optbench.Synthetics.Goldstein_Discrete`
* :code:`optbench.Synthetics.Griewank`
* :code:`optbench.Synthetics.Levy`
* :code:`optbench.Synthetics.Michalewicz`
* :code:`optbench.Synthetics.Powell`
* :code:`optbench.Synthetics.Rastrigin`
* :code:`optbench.Synthetics.Rosenbrock`
* :code:`optbench.Synthetics.Styblinski-Tang`
* :code:`optbench.Synthetics.Beale`
* :code:`optbench.Synthetics.Cosine8`
* :code:`optbench.Synthetics.DropWave`
* :code:`optbench.Synthetics.EggHolder`
* :code:`optbench.Synthetics.Hartmann3D`
* :code:`optbench.Synthetics.Hartmann6D`
* :code:`optbench.Synthetics.HolderTable`
* :code:`optbench.Synthetics.Shekelm5`
* :code:`optbench.Synthetics.Shekelm7`
* :code:`optbench.Synthetics.Shekelm10`
* :code:`optbench.Synthetics.Shekel`
* :code:`optbench.Synthetics.SixHumpCamel`
* :code:`optbench.Synthetics.ThreeHumpCamel`
* :code:`optbench.Synthetics.ConstrainedGramacy`
* :code:`optbench.Synthetics.ConstrainedHartmann`
* :code:`optbench.Synthetics.ConstrainedHartmannSmooth`
* :code:`optbench.Synthetics.PressureVessel`
* :code:`optbench.Synthetics.WeldedBeamSO`
* :code:`optbench.Synthetics.TensionCompressionString`
* :code:`optbench.Synthetics.SpeedReducer`


Example Usage
------------

.. code-block:: python

    import optbench
    import torch

    # Create a Botorch benchmark problem
    problem = optbench.Synthetics.Goldstein_Discrete()
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    constraints, values = problem._evaluate_implementation(x)
    
    print(f"Goldstein Discrete function value at origin: {values[0]}")

Output:

.. code-block:: console

    Goldstein Discrete function value at origin: tensor([-600.])