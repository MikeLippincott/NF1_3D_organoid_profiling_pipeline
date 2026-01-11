============================================
Introduction
============================================

Overview
========================================

The NF1 3D Organoid Profiling Pipeline is a comprehensive computational framework for processing, segmenting, and analyzing three-dimensional microscopy imaging data from patient-derived organoids. This pipeline was developed to support high-throughput drug screening and characterization studies of NF1-related conditions.

Key Features
============================================

- **3D Image Processing**: Full support for volumetric microscopy data
- **Automated Segmentation**: Deep learning-based nuclei and organoid segmentation
- **Feature Extraction**: 150+ morphological and intensity-based features
- **High-Throughput**: Scalable processing on HPC clusters (SLURM)
- **Quality Control**: Comprehensive image QC and filtering
- **Statistical Analysis**: Built-in tools for hit detection and significance testing
- **Interactive Visualization**: Web-based Shiny application for result exploration

Applications
====================================

- **Drug Screening**: Identify compounds affecting organoid morphology
- **Disease Modeling**: Compare morphological signatures across patient genotypes
- **Mechanistic Studies**: Relate phenotypes to specific organelles
- **Biomarker Discovery**: Identify predictive features of treatment response

Technical Specifications
=========================================

**Imaging Requirements:**

- Microscope: Olympus IX83 with 60x/1.35 NA oil immersion objective
- Channels: 5 fluorescent channels (405, 488, 555, 568, 640 nm)
- Z-depth: ~50-100 µm with 0.5 µm z-spacing
- Typical image size: 2048 × 2048 × 100 voxels

**Computational Requirements:**

- **CPU Processing**: 8-16 cores recommended
- **GPU Processing**: NVIDIA GPU with 8+ GB VRAM (optional, for acceleration)
- **Storage**: ~500 GB - 1 TB per patient dataset
- **Runtime**: ~2-4 weeks for full analysis of 1000 well FOVs

**Software Stack:**

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- CellProfiler 4.x
- Cellpose 2.x
- SAM-Med3D (optional)
- R 4.x (for visualization)

Citation
================================

If you use this pipeline in your research, please cite:

*[Citation information to be added]*

Contact & Support
=========================================

For questions or issues, please contact:

- **Primary Developer**: [Contact information]
- **GitHub Issues**: [Repository URL]
- **Documentation**: See individual module READMEs
