========
Overview
========

Pipeline Architecture
=====================

The feature extraction pipeline is adapted from CellProfiler's approach and
follows standard image-based profiling practices.

Key Characteristics
===================

* **3D Processing**: Operates on volumetric data across multiple light spectra
* **Multi-compartment**: Segments cells into distinct biological compartments
* **Distributed Computing**: Optimized for multi-core CPU and GPU execution
* **Functional Approach**: Adapted from CellProfiler with functional programming paradigm
* **Memory Efficient**: Uses generators instead of list declarations

Supported Compartments
======================

The pipeline segments images into the following biological compartments:

* **Organoid**: Whole organoid structure
* **Nucleus**: Nuclear region
* **Cell**: Individual cell boundaries
* **Cytoplasm**: Cytoplasmic region

Supported Image Channels
========================

The pipeline supports multi-channel imaging with 5 default channels:

* DAPI (nucleus stain)
* Phalloidin (actin cytoskeleton)
* Additional protein-specific channels

