.. _function_filtering:

Search for functions with custom filters
=================

You can search for benchmark functions with filters for category, dimensionality, and number of objectives.

Example Usage
------------

.. code-block:: python

    import optbench
    import optbench.DataType as DataType

    # Retrieve all available categories
    available_categories = optbench.filter_functions().keys()

    # Example of filtering functions
    filtered_functions_list = optbench.filter_functions(
        dimension_filter = lambda dim: dim>=5, # Include only functions with dimensionality of 5 or greater
        input_type_filter = lambda input_type: input_type==DataType.CONTINUOUS, # Include only continuous functions
        objectives_filter = lambda n: n==1, # Include only single-objective functions
        constraints_filter = lambda c: c==0, # Include functions with 0 constraints (Functions with only simple bound constraints)
        category_filter = lambda x: x!="CEC.CEC2020_RW_Constrained" # Exclude all CEC2020 functions
        )

    print(filtered_functions_list)

Output:

.. code-block:: console

    {'Synthetics': ['optbench.Synthetics.DixonPrice', 'optbench.Synthetics.Griewank', 'optbench.Synthetics.Levy', 'optbench.Synthetics.Michalewicz', 'optbench.Synthetics.Powell', 'optbench.Synthetics.Rastrigin', 'optbench.Synthetics.Rosenbrock', 'optbench.Synthetics.StyblinskiTang', 'optbench.Synthetics.Cosine8', 'optbench.Synthetics.Hartmann6D', 'optbench.Synthetics.SVM'],
     'LassoBench': ['optbench.LassoBench.LassoBreastCancer', 'optbench.LassoBench.LassoDiabetes', 'optbench.LassoBench.LassoDNA', 'optbench.LassoBench.LassoLeukemia', 'optbench.LassoBench.LassoRCV1', 'optbench.LassoBench.LassoSyntHard', 'optbench.LassoBench.LassoSyntHigh', 'optbench.LassoBench.LassoSyntMedium', 'optbench.LassoBench.LassoSyntSimple'], 
     'Engineering': ['optbench.Engineering.RobotPush', 'optbench.Engineering.Rover', 'optbench.Engineering.NonLinearConstraintProblemA3', 'optbench.Engineering.NonLinearConstraintProblemA4', 'optbench.Engineering.NonLinearConstraintProblemB3', 'optbench.Engineering.AntProblem', 'optbench.Engineering.HalfCheetahProblem', 'optbench.Engineering.HumanoidProblem', 'optbench.Engineering.HumanoidStandupProblem', 'optbench.Engineering.PusherProblem', 'optbench.Engineering.Walker2DProblem'], 
     ...
     }