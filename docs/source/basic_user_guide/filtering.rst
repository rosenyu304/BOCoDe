.. _function_filtering:

Search for problems with filters
================================

Problems are indexed by metadata. Use :code:`bocode.list_problems` to search by
application area, objective count, constraints, variable type, or scalability.

Example Usage
-------------

.. code-block:: python

    import bocode

    # All problems
    bocode.list_problems()

    # Filter by metadata fields
    bocode.list_problems(application="Engineering")
    bocode.list_problems(num_objectives=1, constrained=False)
    bocode.list_problems(input_type="continuous", scalable=True)

    # Convenience selectors for algorithm benchmarking
    bocode.get_single_objective_unconstrained()
    bocode.get_single_objective_constrained()
    bocode.get_multi_objective_unconstrained()
    bocode.get_multi_objective_constrained()

Each name resolves to a problem class via :code:`bocode.get_problem(name)` (or the
flat attribute :code:`bocode.<Name>`), and its metadata is available through
:code:`bocode.get_metadata(name)`.

Output:

.. code-block:: console

    >>> bocode.list_problems(application="Materials")
    ['AgNP', 'AutoAM', 'CrossedBarrel', 'P3HT', 'Perovskite']

    >>> bocode.get_metadata("CrossedBarrel")["num_objectives"]
    1
