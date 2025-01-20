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

    # Create a benchmark problem
    problem = optbench.create_problem("sphere")

    # Get problem information
    dim = problem.dimension
    bounds = problem.bounds
    
    # Evaluate the objective function
    x = [0.0] * dim
    value = problem.evaluate(x)

    print(f"Function value at origin: {value}")

Basic Concepts
-------------

OptBench provides a standardized interface for various benchmark problems. Each problem has:

* An objective function
* Input dimension
* Bounds on the variables
* Optional constraints
* Known optimal value (for most problems)