====================================================
Mathematical Formulation
====================================================

Feature Extraction Equations
===============================================================

Single-Channel Features
-----------------------------------------------------------

For features that depend on a single channel within a compartment:

.. math::

   F_{c,o,f} = \text{extract}(I_c, M_o)

Where:

* :math:`F_{c,o,f}` = feature :math:`f` from channel :math:`c` in object :math:`o`
* :math:`I_c` = 3D image intensity array for channel :math:`c`
* :math:`M_o` = 3D binary mask for object :math:`o`

Multi-Channel Features (Colocalization)
-------------------------------------------------------------------------

For features comparing two channels:

.. math::

   F_{c_1,c_2,o,f} = \text{colocalize}(I_{c_1}, I_{c_2}, M_o)

Total Feature Count
---------------------------------------------------------------

The total number of features extracted is:

.. math::

   N_{\text{total}} = N_{\text{single}} + N_{\text{colocalization}} + N_{\text{special}}

Where:

**Single-channel features:**

.. math::

   N_{\text{single}} = n_o \times n_c \times n_f

**Colocalization features:**

.. math::

   N_{\text{colocalization}} = n_o \times \binom{n_c}{2} \times n_{f,coloc}

**Complete formula:**

.. math::

   N_{\text{total}} = n_o \times n_c \times n_f + n_o \times \frac{n_c(n_c - 1)}{2} \times n_{f,coloc} + n_{\text{special}}

Symbol Definitions
======================================================================================

.. list-table::
   :header-rows: 1
   :widths: 15 70

   * - Symbol
     - Definition
   * - :math:`N_{\text{total}}`
     - Total number of features
   * - :math:`n_o`
     - Number of object types (Organoid, Nucleus, Cell, Cytoplasm = 4)
   * - :math:`n_c`
     - Number of image channels (typically 5)
   * - :math:`n_f`
     - Number of single-channel features per object (~61)
   * - :math:`n_{f,coloc}`
     - Number of colocalization features per channel pair (32)

Calculation Example
======================================================================

With standard parameters:

.. math::

   N_{\text{single}} &= 4 \times 5 \times 61 = 1,220

   N_{\text{colocalization}} &= 4 \times 10 \times 32 = 1,280

   N_{\text{total}} &= 1,220 + 1,280 + 2 = 2,502

