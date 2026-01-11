===========================================================
Texture Features
===========================================================

Description
=================================

Texture features quantify spatial patterns and local intensity variations
within segmented objects. These features are computed from the Gray-Level
Co-occurrence Matrix (GLCM).

Calculation Method
=========================================================================

Texture features are derived from the Gray-Level Co-occurrence Matrix (GLCM),
which captures the frequency of intensity pair relationships at specified offsets.

Parameters
--------------------------------------------------

* **Gray Levels**: 256 (quantization of intensity values)
* **Offset**: 1 voxel (distance for co-occurrence pairs)

These parameters can be adjusted to capture texture patterns at different scales.

Features Extracted
==============================================================

.. list-table:: Texture Feature Measurements
   :header-rows: 1
   :widths: 45 55

   * - Feature
     - Description
   * - Angular.Second.Moment
     - Textural uniformity (GLCM homogeneity)
   * - Contrast
     - Local variation in intensity
   * - Correlation
     - Linear dependency of gray levels
   * - Entropy
     - Randomness/disorder of texture
   * - Difference.Entropy
     - Entropy of intensity differences
   * - Difference.Variance
     - Variance of intensity differences
   * - Information.Measure.of.Correlation.1
     - Correlation measure 1
   * - Information.Measure.of.Correlation.2
     - Correlation measure 2
   * - Inverse.Difference.Moment
     - Local homogeneity metric
   * - Sum.Average
     - Average of co-occurrence sum
   * - Sum.Entropy
     - Entropy of co-occurrence sum
   * - Sum.Variance
     - Variance of co-occurrence sum
   * - Variance
     - Variance of GLCM

Multi-Scale Analysis
=====================================================

Different offsets reveal texture patterns at different scales:

* **Offset = 1**: Fine-grained local texture
* **Offset > 1**: Coarser regional patterns

Adjusting parameters allows for multi-scale texture characterization.

