.. _function_filtering:

Search for functions with custom filters
=================

You can search for benchmark functions in BOCoDe using custom filters for dimensionality, number of objectives, number of constraints, and other criteria.

Example Usage
------------

.. code-block:: python

    import bocode
    from bocode import DataType

    # Available DataTypes:
    #    DataType.CONTINUOUS
    #    DataType.DISCRETE
    #    DataType.CATEGORICAL

    # Retrieve all available categories
    available_categories = bocode.filter_functions().keys()

    # Example of filtering functions using all available filters
    filtered_functions_list = bocode.filter_functions(
        dimension_filter = lambda dim: dim>=5, # Include only functions with dimensionality of 5 or greater
        input_type_filter = lambda input_type: input_type==DataType.CONTINUOUS, # Include only continuous functions
        objectives_filter = lambda n: n==1, # Include only single-objective functions
        constraints_filter = lambda c: c==0, # Include functions with 0 constraints (Functions with only simple bound constraints)
        category_filter = lambda x: x!="CEC.CEC2020_RW_Constrained" # Exclude all CEC2020 functions
        )

    print(filtered_functions_list)

Output:

.. code-block:: console

    {'Synthetics': ['bocode.Synthetics.DixonPrice', 'bocode.Synthetics.Griewank', 'bocode.Synthetics.Levy', 'bocode.Synthetics.Michalewicz', 'bocode.Synthetics.Powell', 'bocode.Synthetics.Rastrigin', 'bocode.Synthetics.Rosenbrock', 'bocode.Synthetics.StyblinskiTang', 'bocode.Synthetics.Cosine8', 'bocode.Synthetics.Hartmann6D', 'bocode.Synthetics.SVM'],
     'LassoBench': ['bocode.LassoBench.LassoBreastCancer', 'bocode.LassoBench.LassoDiabetes', 'bocode.LassoBench.LassoDNA', 'bocode.LassoBench.LassoLeukemia', 'bocode.LassoBench.LassoRCV1', 'bocode.LassoBench.LassoSyntHard', 'bocode.LassoBench.LassoSyntHigh', 'bocode.LassoBench.LassoSyntMedium', 'bocode.LassoBench.LassoSyntSimple'], 
     'Engineering': ['bocode.Engineering.RobotPush', 'bocode.Engineering.Rover'],
     ...
     }