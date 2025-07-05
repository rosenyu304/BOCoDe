.. _benchmark_problem:

BenchmarkProblem Class
=====================

The ``BenchmarkProblem`` class is the parent class for all benchmark problems in the library.

.. note::
   See the :ref:`glossary` for definitions of key terms like 'objective function', 'constraint', and 'bounds'.

Class Attributes
----------------

Every BenchmarkProblem class must define the following class attributes:

.. list-table:: Class Attributes
   :widths: 20 50 65
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``available_dimensions``
     - ``Union[int, Tuple[int, Optional[int]], Set[int], List[int]]``
     - Supported dimensions for the problem.
   * - ``num_objectives``
     - ``Union[int, Tuple[int, Optional[int]], Set[int], List[int]]``
     - Supported number of objective functions
   * - ``num_constraints``
     - ``Union[int, Tuple[int, Optional[int]], Set[int], List[int]]``
     - Supported number of constraint functions

Definition Scheme:

- ``int``: A single possible value for the problem. E.g. ``1`` 
- ``Tuple[int, Optional[int]]``: A range of possible values for the problem. E.g. ``(1, 3)`` means 1, 2, or 3.
- ``Set[int] or List[int]``: A set of possible values for the problem. E.g. ``{1, 3, 4}`` means 1, 3, or 4.

Example:

.. code-block:: python

    import bocode

    print(bocode.Synthetics.Ackley.available_dimensions)
    # Output: (1, None)
    # This means that the problem can be defined with 1 or more dimensions.
    print(bocode.Synthetics.Ackley.num_objectives)
    # Output: 1
    # This means that the problem has 1 objective function.
    print(bocode.Synthetics.Ackley.num_constraints)
    # Output: 2
    # This means that the problem has 2 constraint functions.

Instance Attributes
------------------

After initialization, a ``BenchmarkProblem`` instance contains the following attributes:

.. _benchmark_attributes:
.. list-table:: Instance Attributes
   :widths: 20 15 65
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``dim``
     - ``int``
     - Dimension of the decision space
   * - ``num_objectives``
     - ``int``
     - Number of objective functions
   * - ``num_constraints``
     - ``int``
     - Number of constraint functions
   * - ``bounds``
     - ``Optional[List[Union[Tuple, Set]]]``
     - Bounds for each decision variable. A tuple denotes a continuous bound, while a set or list denotes a discrete bound. Can be modified for certain problems.
   * - ``x_opt``
     - ``Optional[torch.Tensor]``
     - Optimal decision variables.
   * - ``optimum``
     - ``Optional[torch.Tensor]``
     - Optimal objective value(s)


Methods
-------

.. evaluate:
evaluate(X)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluates the objective and constraint functions at the given input points.

**Parameters:**

- ``X`` (``torch.Tensor``): Input data in range of the bounds with shape (n_samples, dim).

**Returns:**

- ``Tuple[torch.Tensor, torch.Tensor]:``: Objective and constraint values (in that order) with shape ``(n_samples, num_objectives)`` and ``(n_samples, num_constraints)`` respectively.

.. note::

    Some problems (like the CEC2020 functions) may have additional return values for equality constraints. See :ref:`benchmarks` for more details and examples. 

scale(X)
~~~~~~~~

Scales continuous input data from the normalized range (0, 1) to the problem's actual bounds.

**Parameters:**

- ``X`` (``torch.Tensor``): Input data in range (0, 1) with shape (n_samples, dim)

**Returns:**

- ``torch.Tensor``: Scaled data within the problem bounds

**Raises:**

- ``TypeException``: If X is not a torch tensor
- ``DimensionException``: If X dimensions don't match problem dimension
- ``RangeException``: If X values are outside (0, 1) range

**Example:**

.. code-block:: python

    import bocode
    import torch

    # Create a problem
    problem = bocode.Engineering.KeaneBump(dim=2)

    # Print problem bounds
    print(problem.bounds)
    # Output: [(0, 10), (0, 10)]

    # Make a normalized input in (0, 1) range
    X_normalized = torch.tensor([[0.5, 0.3]])

    # Scale to actual bounds
    X_scaled = problem.scale(X_normalized)
    print(X_scaled)
    # Result: tensor([[5.0, 3.0]])

show_info()
~~~~~~~~~~~

Prints information about the benchmark problem.

**Example:**

.. code-block:: python

    import bocode

    problem = bocode.Synthetics.Ackley(dim=2)
    problem.show_info()
    
Output:

.. code-block:: console

    Function info:
     Number of objectives: 1
     Number of constraints: 2
     Number of dimensions: 2
     Optimum Value: [[0]]
     Optimal Decision Variables: [[0, 0]]
     Bounds: [(-5, 10), (-5, 10)]

visualize_function(sampling_density=50)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates interactive visualizations of the objective function(s).

**Parameters:**
- ``sampling_density`` (``int``, optional): Sampling density per axis. Default is 50.

**Features:**

- **1D problems**: 2D line plots for each objective
- **2D problems**: 3D surface plots for each objective
- **Higher dimensions**: Interactive cross-sectional 3D plots with sliders for fixed dimensions

**Limitations:**

- Not supported for discrete or mixed variable types
- May be slow for problems with high dimensionality

**Example:**

.. code-block:: python

    # Create and visualize a 2D problem
    problem = bocode.Synthetics.Rastrigin(dim=5)
    problem.visualize_function()

You may need to open a browser to http://127.0.0.1:8050/ to fully see the visualization for problems with more than 2 dimensions. See :ref:`function_visualization` for examples.

.. rubric:: Glossary

.. _glossary:

Glossary
--------

- **Objective function**: The function(s) to be optimized (maximized or minimized) in a benchmark problem.
- **Constraint**: A function that restricts the feasible region of the decision variables to be **g(x) ≤ 0**. Some problems may have equality constraints. See :ref:`benchmarks` for more details on specific problems.
- **Dimension**: The number of decision variables in the problem.
- **Bounds**: A type of constraint that restricts the decision variables to a certain range (continuous) or a set of values (discrete/categorical).
- **Decision variable**: A variable whose value is to be determined by the optimization process. This is the input to the objective and constraint functions, also referred to as `X`.
- **Optimum**: The maximum value of the objective function(s) for the problem within the feasible region.
- **Feasible region**: The set of all points that satisfy the constraints and bounds.