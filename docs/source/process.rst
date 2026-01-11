=======
Process
=======

Extraction Workflow
===================

The feature extraction process follows these steps:

1. **Image Loading**: Load 3D image sets from different spectral channels
2. **Segmentation**: Identify and segment biological compartments
3. **Feature Calculation**: Extract features from segmented regions
4. **Output Formatting**: Generate feature matrices for downstream analysis

Segmentation Strategy
=====================

The pipeline uses automated segmentation to identify:

- Object boundaries and spatial extent
- Subcellular compartments
- Relationship between compartments

Feature Extraction Strategy
===========================

Features are extracted for each compartment-channel combination:

* **Per-object features**: Properties of individual segmented objects
* **Channel-specific features**: Measurement within specific spectral channels
* **Relational features**: Properties between different channels (colocalization)

Computing Approaches
====================

The implementation supports multiple computing strategies:

CPU-Based Processing
---------------------

* Distributed across multiple CPU cores
* Single image set - channel combination per core
* Scalable with available processors

GPU Acceleration
----------------

* Single GPU execution when cluster unavailable
* Accelerated via RAPIDS libraries (cuPy, cuCIM)
* Maintains API compatibility with CPU version

Libraries Used
==============

The extraction uses these Python scientific libraries:

* `scikit-image <https://scikit-image.org/>`_: Image processing and analysis
* `scipy <https://www.scipy.org/>`_: Scientific computing
* `mahotas <https://mahotas.readthedocs.io/>`_: Image processing operations
* `numpy <https://numpy.org/>`_: Numerical computing
* `cupy <https://docs.cupy.dev/>`_: GPU-accelerated arrays
* `cucim <https://docs.rapids.ai/api/cucim/>`_: GPU-accelerated image processing

