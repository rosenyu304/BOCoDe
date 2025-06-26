.. _function_visualization:

Visualize functions
=================

You can visualize any non-discrete function in OptBench. This is useful for understanding the function's behavior and characteristics.

Example Usage
------------

.. code-block:: python

    import bocode

    # Initialize the function
    problem = bocode.Synthetics.Powell() 

    # Visualize the function
    problem.visualize_function()

Example Output at http://127.0.0.1:8050/:

 .. image:: /basic_user_guide/example_visualization.png
    :width: 600px
    :align: center
    :alt: Example visualization of Powell function