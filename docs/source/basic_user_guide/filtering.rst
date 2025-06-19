.. _function_filtering:

Search for functions with custom filters
=================

You can search for benchmark functions with filters for category, dimensionality, and number of objectives.

Example Usage
------------

.. code-block:: python

    import optbench

    # Retrieve all available categories
    available_categories = optbench.filter_functions().keys()

    # Call the function filter method
    filtered_functions_list = optbench.filter_functions(
        dimension_filter = lambda x: x>=5, # Include only functions with dimensionality of 5 or greater
        objectives_filter = lambda x: x==1, # Include only single-objective functions
        constraints_filter = lambda x: x==0, # Include functions with 0 constraints (Functions with only simple bound constraints)
        category_filter = lambda x: x!="CEC2020_RW_Constrained" # Exclude all CEC2020 functions
        )

    print(filtered_functions_list)

Output:

.. code-block:: console

    {
    'Synthetics': ['DixonPrice', 'Griewank', 'Levy', 'Michalewicz', 'Powell', 'Rastrigin', 'Rosenbrock', 'StyblinskiTang', 'Cosine8', 'Hartmann6D', 'SVM'], 
    'LassoBench': ['LassoBreastCancer', 'LassoDiabetes', 'LassoDNA', 'LassoLeukemia', 'LassoRCV1', 'LassoSyntHard', 'LassoSyntHigh', 'LassoSyntMedium', 'LassoSyntSimple'], 
    'Engineering': ['RobotPush', 'Rover', 'NonLinearConstraintProblemA3', 'NonLinearConstraintProblemA4', 'NonLinearConstraintProblemB3', 'AntProblem', 'HalfCheetahProblem', 'HumanoidProblem', 'HumanoidStandupProblem', 'PusherProblem', 'Walker2DProblem'], 
    ...
    }