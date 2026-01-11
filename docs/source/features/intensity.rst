=======================================================================
Intensity Features
=======================================================================

Description
=================================================================

Intensity features quantify the pixel/voxel value distributions within
segmented objects. These measurements capture both overall intensity levels
and spatial intensity patterns.

Features Extracted
========================================================================

Location-Based Intensity
----------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Description
   * - CM.X / CM.Y / CM.Z
     - Center of mass in each spatial dimension
   * - CMI.X / CMI.Y / CMI.Z
     - Inverse center of mass in each dimension
   * - I.X / I.Y / I.Z
     - Integrated intensity along each axis
   * - MAX.X / MAX.Y / MAX.Z
     - Coordinates of maximum intensity

Statistical Measures
------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Description
   * - MEAN.INTENSITY
     - Average intensity within object
   * - MEDIAN.INTENSITY
     - Median intensity value
   * - MAX.INTENSITY
     - Maximum intensity value
   * - MIN.INTENSITY
     - Minimum intensity value
   * - STD.INTENSITY
     - Standard deviation of intensity
   * - MAD.INTENSITY
     - Median absolute deviation
   * - LOWER.QUARTILE.INTENSITY
     - 25th percentile
   * - UPPER.QUARTILE.INTENSITY
     - 75th percentile

Edge-Based Measurements
---------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Feature
     - Description
   * - INTEGRATED.INTENSITY.EDGE
     - Sum of edge pixel intensities
   * - MEAN.INTENSITY.EDGE
     - Average edge intensity
   * - MAX.INTENSITY.EDGE
     - Maximum edge intensity
   * - MIN.INTENSITY.EDGE
     - Minimum edge intensity
   * - STD.INTENSITY.EDGE
     - Standard deviation of edge intensities
   * - EDGE.COUNT
     - Number of edge voxels

Other Measurements
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Description
   * - VOLUME
     - Object volume (included for reference)
   * - DIFF.X / DIFF.Y / DIFF.Z
     - Spatial intensity gradient in each dimension
   * - MASS.DISPLACEMENT
     - Distance between geometric and intensity centers

Calculation Method
========================================================================

Intensity features are extracted from 3D voxel data:

1. **Voxel Extraction**: Extract all voxel values within segmented region
2. **Statistical Computation**: Calculate mean, median, std, quartiles
3. **Spatial Analysis**: Determine center of mass and intensity gradients
4. **Edge Detection**: Identify and measure boundary voxel properties

Data Source
=================================================================

All intensity features are computed from the 3D voxel intensity data
of the selected channel within the segmented object boundary.

