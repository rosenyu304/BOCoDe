.. _Getting_Started:

Getting Started
===============

Installation
------------

To use BOCoDe, first install it using pip:

.. code-block:: console

   (.venv) $ pip install bocode

Quick Start
----------

Here's a simple example of how to use BOCoDe:

.. code-block:: python

    import bocode
    import torch

    # List all available problems (optionally filter by metadata)
    print(bocode.list_problems())
    print(bocode.list_problems(application="Engineering"))

    # Instantiate a real-world benchmark problem by name
    problem = bocode.CantileverBeam()

    # Evaluate at random points scaled into the problem bounds
    x = problem.scale(torch.rand(5, problem.dim))
    values, constraints = problem.evaluate(x)

    print("objective values:", values.flatten())
    print("inspect metadata:", bocode.get_metadata("CantileverBeam"))

Basic Concepts
-------------

BOCoDe provides a standardized interface for various benchmark problems. Each problem has:

* An :ref:`objective function <evaluate-x>` to be optimized
* Input :ref:`dimension <benchmark_attributes>` (number of decision variables)
* :ref:`Bounds <benchmark_attributes>` on the decision variables
* Optional :ref:`constraints <benchmark_attributes>` that must be satisfied
* Known :ref:`optimum <benchmark_attributes>` value (for some problems)