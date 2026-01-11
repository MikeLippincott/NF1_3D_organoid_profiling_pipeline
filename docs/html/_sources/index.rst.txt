================================================================================
3D Feature Extraction for Cell Painting Performed on Organoids
================================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   process
   features/index
   extraction_math
   libraries

Welcome to the documentation for the 3D feature extraction pipeline. This guide
explains the features being extracted and the methodology behind extraction from
3D image sets.

An image set is a collection of 3D images representing the same object but
captured using different light spectra (channels).

Quick Start
===========

To get started with understanding the features:

1. Read the :doc:`overview` for a high-level introduction
2. Review the :doc:`process` to understand the extraction workflow
3. Explore :doc:`features/index` to learn about each feature type
4. Check :doc:`extraction_math` for the mathematical formulation

Installation
============

To build this documentation locally:

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme
   cd docs
   make html

Then open ``docs/build/html/index.html`` in your browser.

