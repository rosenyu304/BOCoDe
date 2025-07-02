.. _Getting_Started:

Getting Started
===============

Installation
------------

To use BOCode, first install it using pip:

.. code-block:: console

   (.venv) $ pip install bocode

Quick Start
----------

Here's a simple example of how to use BOCode:

.. code-block:: python

    import bocode
    import torch

    # Retrieve all available benchmark problems in bocode by searching with no filters
    all_problems = bocode.filter_functions()
    print(all_problems)

    # Instantiate a Synthetic benchmark problem
    problem = bocode.Synthetics.Ackley()
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    values, constraints = problem.evaluate(x)
    
    print(f"Ackley function value at origin: {values[0]}")

Basic Concepts
-------------

BOCode provides a standardized interface for various benchmark problems. Each problem has:

* An :ref:`objective function <evaluate_implementation>`
* Input :ref:`dimension <benchmark_attributes>`
* :ref:`Bounds <benchmark_attributes>` on the variables
* Optional :ref:`constraints <benchmark_attributes>`
* Known :ref:`optimum <benchmark_attributes>` value (for some problems)