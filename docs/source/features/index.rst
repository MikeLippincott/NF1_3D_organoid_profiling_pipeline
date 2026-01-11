=====================================================================
Feature Types
=====================================================================

.. toctree::
   :maxdepth: 2

   areasizeshape
   intensity
   colocalization
   texture
   granularity
   neighbors
   sammed3d

Overview
==============================================================

The pipeline extracts seven distinct feature types from segmented biological
compartments. Each feature type captures different aspects of cellular and
subcellular morphology and intensity.

Feature Count Summary
===========================================================================

For a typical imaging experiment with 5 channels and 4 compartment types:

- **Area.Size.Shape features per object**: 15
- **Intensity features per object**: 32
- **Colocalization features per object pair**: 32
- **Texture features per object**: 13
- **Granularity features per object**: 1
- **Neighbors features**: 2
- **SAM-Med3D features**: Segmentation masks

Total theoretical features extracted: **2,502**

This is calculated as:

.. math::

   n_{features} = n_{objects} \times n_{channels} \times n_{features\_per\_object}
   + n_{objects} \times \frac{n_{channels}(n_{channels} - 1)}{2} \times n_{colocalization}
   + n_{neighbors}

