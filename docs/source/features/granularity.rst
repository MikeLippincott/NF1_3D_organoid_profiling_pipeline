====================
Granularity Features
====================

Description
===========

Granularity features measure the size distribution of texture elements
(granules) at multiple scales. These features reveal the characteristic
scale of intensity variations within objects.

Spectrum Approach
=================

Granularity is calculated across a spectrum of scales (1 to 16).

Features Extracted
==================

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Feature
     - Description
   * - GRANULARITY.1
     - Granularity at scale 1 (finest scale)
   * - GRANULARITY.2
     - Granularity at scale 2
   * - ...
     - ...
   * - GRANULARITY.16
     - Granularity at scale 16 (coarsest scale)

Interpretation
==============

* **High granularity at small scales**: Fine-grained texture details
* **High granularity at large scales**: Coarse regional intensity variations
* **Granularity profile**: Overall texture scale characteristics

Applications
=============

Granularity features are useful for:

* Identifying cellular texture scale patterns
* Detecting subcellular compartment granularity
* Characterizing organelle size distributions
* Quantifying spatial heterogeneity

