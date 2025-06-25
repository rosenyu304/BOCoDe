.. _Getting_Started:

Getting Started
===============

Installation
------------

To use OptBench, first install it using pip:

.. code-block:: console

   (.venv) $ pip install optbench

Quick Start
----------

Here's a simple example of how to use OptBench:

.. code-block:: python

    import optbench
    import torch

    # Retrieve all available benchmark problems in optbench by searching with no filters
    all_problems = optbench.filter_functions()
    print(all_problems)

    # Instantiate a Synthetic benchmark problem
    problem = optbench.Synthetics.Ackley()
    
    # Evaluate at a point
    x = torch.Tensor([[0.0] * problem.dim])
    constraints, values = problem._evaluate_implementation(x)
    
    print(f"Ackley function value at origin: {values[0]}")

Basic Concepts
-------------

OptBench provides a standardized interface for various benchmark problems. Each problem has:

* An objective function
* Input dimension
* Bounds on the variables
* Optional constraints
* Known optimal value (for most problems)