.. BOCoDe documentation master file, created by
   sphinx-quickstart on Mon Jan 13 13:02:18 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/bocode_logo.png
   :align: center
   :width: 360px
   :alt: BoCoDe

BOCoDe Documentation
====================

BOCoDe is a comprehensive Python library containing optimization benchmark problems for testing and evaluating optimization algorithms.

.. note::

   This project is under active development.

Features
--------

* **Diverse Benchmark Collection**: Problems from various sources including Botorch, BBOB, CEC competitions, and engineering applications
* **Multiple Problem Types**: Single and multi-objective, with and without constraints
* **Standardized Interface** across all benchmark problems
* **Function Filtering** to find problems matching specific criteria
* **Visualization Tools** for understanding problem landscapes and behavior
* **Framework Integration**

Citing
--------

1. If you use BOCoDe in your research, please cite the following paper (full report coming soon):

   @misc{yu2025bocode,
   
   author={Rosen Ting-Ying Yu, Advaith Narayanan, Cyril Picard, Faez Ahmed},
   
   title = {{BOCoDe}: Benchmarks for Optimization and Computational Design},
   
   year={2025},
   
   url={https://github.com/rosenyu304/BOCoDe}
   }


2. If you use the the BOCoDe interfaces to other libraries or open source code functions (ex: BoTorch, BBOB, NEORL, MODAct, LassoBench, ...), please cite them accordingly.


License
--------
BOCoDe is MIT licensed.

Contents
--------

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   :glob:

   getting_started
   benchmark_table
   basic_user_guide/index
   benchmarks/index
   api

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
