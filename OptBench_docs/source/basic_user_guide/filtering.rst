.. _function_filtering:

Search for functions with custom filters
=================

You can search for benchmark functions with filters for category, dimensionality, and number of objectives.

Example Usage
------------

.. code-block:: python

    import optbench

    # Available Categories: ["Synthetics", "LassoBench", "Engineering", "CEC2020_RW_Constrained", "BBOB", "BoTorch"]

    # Call the function filter method
    filtered_functions_list = optbench.filter_functions(
        dimension_filter = lambda x: x>=5, # Include only functions with dimensionality of 5 or greater
        objectives_filter = lambda x: x==1, # Include only single-objective functions
        category_filter = lambda x: x!="CEC2020_RW_Constrained" # Exclude all CEC2020 functions
        )
    
    print(filtered_functions_list)

Output:

.. code-block:: console

    {
    'Synthetics': ['Ackley', 'DixonPrice', 'Griewank', 'Levy', 'Michalewicz', 'Powell', 'Rastrigin', 'Rosenbrock', 'StyblinskiTang'], 
    'LassoBench': ['LassoBreastCancer', 'LassoDiabetes', 'LassoDNA', 'LassoLeukemia', 'LassoRCV1', 'LassoSyntHard', 'LassoSyntHigh', 'LassoSyntMedium', 'LassoSyntSimple'], 
    'Engineering': ['MOPTA08Car', 'RobotPush', 'Rover', 'Truss10D', 'Truss25D'], 
    'BBOB': ['BBOB', 'BBOB_Boxed', 'BBOB_Constrained', 'BBOB_LargeScale', 'BBOB_MixInt', 'BBOB_Noisy']
    }