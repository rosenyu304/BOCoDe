.. _api:

API Reference
=============

BoCoDe's public Python API. The registry functions discover and load problems by
name from metadata; :class:`~bocode.base.BenchmarkProblem` is the base class that
every benchmark problem implements.

Problem registry
----------------

.. currentmodule:: bocode

.. autosummary::
   :toctree: _generated/api
   :nosignatures:

   list_problems
   get_problem
   get_metadata
   list_metadata
   list_synthetic
   filter_functions
   get_single_objective_unconstrained
   get_single_objective_constrained
   get_multi_objective_unconstrained
   get_multi_objective_constrained

Benchmark problem base class
----------------------------

Every problem subclasses :class:`bocode.base.BenchmarkProblem` and implements
``_evaluate_implementation``. Key attributes: ``available_dimensions``,
``num_objectives``, ``num_constraints``, and ``variable_types`` (per-dimension
``"continuous"`` / ``"integer"`` / list-of-allowed-values, the source of truth for
mixed-variable handling).

.. autoclass:: bocode.base.BenchmarkProblem
   :members:
   :undoc-members:
   :show-inheritance:
