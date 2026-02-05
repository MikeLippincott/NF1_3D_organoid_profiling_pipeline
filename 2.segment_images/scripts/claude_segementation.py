#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import json
import os
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy
import tifffile
import torch
from arg_parsing_utils import check_for_missing_args, parse_args
from cellpose import models
from file_reading import *
from file_reading import read_zstack_image
from general_segmentation_utils import *
from notebook_init_utils import bandicoot_check, init_notebook
from organoid_segmentation import *
from segmentation_decoupling import *

# In[2]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[3]:


root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[4]:


if not in_notebook:
    args = parse_args()
    clip_limit = args["clip_limit"]
    well_fov = args["well_fov"]
    patient = args["patient"]
    input_subparent_name = args["input_subparent_name"]
    mask_subparent_name = args["mask_subparent_name"]
    check_for_missing_args(
        well_fov=well_fov,
        patient=patient,
        clip_limit=clip_limit,
        input_subparent_name=input_subparent_name,
        mask_subparent_name=mask_subparent_name,
    )
else:
    print("Running in a notebook")
    patient = "NF0014_T1"
    well_fov = "F4-1"
    clip_limit = 0.01
    input_subparent_name = "zstack_images"
    mask_subparent_name = "segmentation_masks"


window_size = 2
input_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{input_subparent_name}/{well_fov}"
).resolve(strict=True)
mask_path = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{mask_subparent_name}/{well_fov}"
).resolve()
mask_path.mkdir(exist_ok=True, parents=True)


# In[5]:


# look up the morphology of the organoid from json file
# json_file_path = pathlib.Path("./organoid_image_labels.json").resolve(strict=True)
# with open(json_file_path, "r") as f:
#     organoid_image_labels = json.load(f)
# organoid_image_labels_df = pd.DataFrame(organoid_image_labels)
# # look up the morphology for this well_fov
# morphology = organoid_image_labels_df.loc[
#     organoid_image_labels_df["well_fov"] == well_fov, "label"
# ].values[0]
morphology = "globular"


# In[6]:


return_dict = read_in_channels(
    find_files_available(input_dir),
    channel_dict={
        "nuclei": "405",
        "cyto1": "488",
        "cyto2": "555",
        "cyto3": "640",
        "brightfield": "TRANS",
    },
    channels_to_read=["cyto2"],
)
cyto2_raw = return_dict["cyto2"]
del return_dict
nuclei_mask_output = pathlib.Path(f"{mask_path}/nuclei_mask.tiff")
nuclei_mask = read_zstack_image(nuclei_mask_output)
# run clip_limit here
cyto2 = skimage.exposure.equalize_adapthist(
    cyto2_raw, clip_limit=clip_limit, kernel_size=None
)
del cyto2_raw


# In[7]:


"""
3D Nuclei-Aware Cell Segmentation Pipeline
============================================
Segments 3D cell stacks using nuclei as guides for cell boundaries.
Works without ground truth data using morphological + seed-based methods.

Supports:
- Multi-channel 3D microscopy data (nuclei + cytoplasm/membrane)
- Automated parameter estimation
- Visualization and validation metrics
"""

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from skimage import filters, measure, morphology, restoration
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries, watershed

warnings.filterwarnings("ignore")


class NucleiAware3DSegmentation:
    """
    3D cell segmentation guided by nuclei channel.

    Pipeline:
    1. Preprocess nuclei and cytoplasm channels
    2. Segment nuclei robustly
    3. Extract nuclei seeds
    4. Propagate boundaries using cytoplasm channel + watershed
    5. Post-process and validate
    """

    def __init__(
        self,
        nuclei_data: np.ndarray,
        cytoplasm_data: np.ndarray,
        voxel_size: Tuple[float, float, float] = (0.5, 0.2, 0.2),
        min_nucleus_volume: int = 50,
        min_cell_volume: int = 100,
    ):
        """
        Parameters
        ----------
        nuclei_data : np.ndarray
            3D array of nuclei channel (Z, Y, X)
        cytoplasm_data : np.ndarray
            3D array of cytoplasm/membrane channel (Z, Y, X)
        voxel_size : tuple
            (Z, Y, X) voxel sizes in micrometers for scale-aware processing
        min_nucleus_volume : int
            Minimum nucleus voxels to keep (filters debris)
        min_cell_volume : int
            Minimum cell voxels to keep
        """
        self.nuclei_raw = nuclei_data.astype(np.float32)
        self.cytoplasm_raw = cytoplasm_data.astype(np.float32)
        self.voxel_size = voxel_size
        self.min_nucleus_volume = min_nucleus_volume
        self.min_cell_volume = min_cell_volume

        # Will store results
        self.nuclei_mask = None
        self.nuclei_labels = None
        self.cell_labels = None
        self.cell_boundaries = None
        self.stats = {}

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] range."""
        data_min = np.percentile(data, 1)
        data_max = np.percentile(data, 99)
        return np.clip((data - data_min) / (data_max - data_min + 1e-8), 0, 1)

    def _preprocess_3d(
        self, data: np.ndarray, sigma_z: float = 1.0, sigma_xy: float = 1.0
    ) -> np.ndarray:
        """Gaussian filtering with anisotropic kernels."""
        # Handle anisotropic voxel sizes
        from scipy.ndimage import gaussian_filter

        sigmas = (sigma_z, sigma_xy, sigma_xy)
        return gaussian_filter(data, sigma=sigmas)

    def segment_nuclei(
        self,
        threshold_method: str = "otsu",
        min_nucleus_size: Optional[int] = None,
        closing_radius: int = 2,
    ) -> np.ndarray:
        """
        Segment nuclei from nuclei channel.

        Parameters
        ----------
        threshold_method : str
            'otsu' - Otsu's thresholding (good for most cases)
            'li' - Li's method (more sensitive)
            'manual' - Use percentile (90th)
        min_nucleus_size : int
            Override min_nucleus_volume for this run
        closing_radius : int
            Morphological closing radius to fill small holes

        Returns
        -------
        nuclei_labels : np.ndarray
            Labeled nuclei (0 = background, 1,2,3... = nuclei)
        """
        print("[1/5] Segmenting nuclei...")

        # Normalize
        nuclei_norm = self._normalize(self.nuclei_raw)

        # Preprocess
        nuclei_smooth = self._preprocess_3d(nuclei_norm, sigma_z=1.0, sigma_xy=1.0)

        # Threshold
        if threshold_method == "otsu":
            threshold = filters.threshold_otsu(nuclei_smooth)
        elif threshold_method == "li":
            threshold = filters.threshold_li(nuclei_smooth)
        else:  # percentile
            threshold = np.percentile(nuclei_smooth, 90)

        nuclei_binary = nuclei_smooth > threshold
        print(f"   - Threshold: {threshold:.3f}")

        # Morphological closing (fill holes)
        if closing_radius > 0:
            nuclei_binary = binary_dilation(nuclei_binary, iterations=closing_radius)
            nuclei_binary = binary_erosion(nuclei_binary, iterations=closing_radius)

        # Label connected components
        nuclei_labels, n_nuclei = ndimage.label(nuclei_binary)
        print(f"   - Found {n_nuclei} nuclei before size filtering")

        # Filter by size
        min_size = min_nucleus_size or self.min_nucleus_volume
        nuclei_labels = self._filter_by_size(nuclei_labels, min_size, max_size=None)

        self.nuclei_labels = nuclei_labels
        self.nuclei_mask = nuclei_labels > 0
        n_final = nuclei_labels.max()
        print(f"   - Kept {n_final} nuclei after size filtering")
        self.stats["n_nuclei"] = n_final

        return nuclei_labels

    def segment_cells(
        self,
        use_distance_transform: bool = True,
        watershed_compactness: float = 1.0,
        dilation_iterations: int = 3,
    ) -> np.ndarray:
        """
        Segment cells using nuclei as seeds and cytoplasm as guidance.

        Parameters
        ----------
        use_distance_transform : bool
            If True, use distance transform from nuclei for smoother propagation
        watershed_compactness : float
            Compactness parameter for watershed (higher = more compact cells)
        dilation_iterations : int
            How many iterations to dilate nuclei before seeding watershed

        Returns
        -------
        cell_labels : np.ndarray
            Labeled cells
        """
        print("\n[2/5] Segmenting cells using nuclei as seeds...")

        if self.nuclei_labels is None:
            raise ValueError("Must run segment_nuclei() first!")

        # Normalize cytoplasm
        cyto_norm = self._normalize(self.cytoplasm_raw)
        cyto_smooth = self._preprocess_3d(cyto_norm, sigma_z=1.0, sigma_xy=1.0)

        # Create seeds from nuclei (optionally dilated)
        if dilation_iterations > 0:
            seeds = binary_dilation(self.nuclei_mask, iterations=dilation_iterations)
            seeds_labeled, n_seeds = ndimage.label(seeds)
        else:
            seeds_labeled = self.nuclei_labels
            n_seeds = self.nuclei_labels.max()

        print(f"   - Using {n_seeds} nuclei as seeds")

        # Create markers for watershed
        markers = seeds_labeled

        # Prepare image for watershed
        # Invert cytoplasm signal (watershed uses low values as targets)
        # Combine with distance transform for smoother segmentation
        if use_distance_transform:
            # Distance from nuclei centers guides cell expansion
            dist_nuclei = distance_transform_edt(~self.nuclei_mask)
            # Normalize distance
            dist_nuclei = dist_nuclei / (dist_nuclei.max() + 1e-8)
            # Invert cytoplasm intensity for watershed
            cyto_inverted = 1.0 - cyto_smooth
            # Combine: favor expansion in high cytoplasm regions
            watershed_image = cyto_inverted + 0.5 * dist_nuclei
        else:
            watershed_image = 1.0 - cyto_smooth

        # Run watershed
        print("   - Running 3D watershed...")
        cell_labels = watershed(
            watershed_image, markers=markers, compactness=watershed_compactness
        )

        print(f"   - Found {cell_labels.max()} cell regions")

        # Filter by size
        cell_labels = self._filter_by_size(
            cell_labels, self.min_cell_volume, max_size=None
        )

        self.cell_labels = cell_labels
        n_final = cell_labels.max()
        print(f"   - Kept {n_final} cells after size filtering")
        self.stats["n_cells"] = n_final

        return cell_labels

    def _filter_by_size(
        self, labels: np.ndarray, min_size: int, max_size: Optional[int] = None
    ) -> np.ndarray:
        """Remove labeled regions by size constraints."""
        unique_labels = np.unique(labels)[1:]  # Exclude background (0)

        filtered = np.zeros_like(labels)
        new_label = 1

        for old_label in unique_labels:
            mask = labels == old_label
            size = mask.sum()

            if size >= min_size and (max_size is None or size <= max_size):
                filtered[mask] = new_label
                new_label += 1

        return filtered

    def separate_touching_cells(
        self, use_h_minima: bool = True, h_value: float = 0.1
    ) -> np.ndarray:
        """
        Further separate touching/merging cells using morphological operations.

        Parameters
        ----------
        use_h_minima : bool
            Use H-minima transform to find weak cell boundaries
        h_value : float
            H parameter for H-minima (higher = more aggressive separation)

        Returns
        -------
        cell_labels : np.ndarray
            Re-labeled cells with improved separation
        """
        print("\n[3/5] Separating touching cells...")

        if self.cell_labels is None:
            raise ValueError("Must run segment_cells() first!")

        cyto_norm = self._normalize(self.cytoplasm_raw)

        if use_h_minima:
            # H-minima transform to suppress shallow minima (reduce over-segmentation)
            from skimage.morphology import h_minima

            try:
                minima = h_minima(cyto_norm, h=h_value)
                markers = ndimage.label(minima)[0]
                # Re-run watershed with refined markers
                watershed_image = 1.0 - cyto_norm
                refined_labels = watershed(watershed_image, markers=markers)
                refined_labels = self._filter_by_size(
                    refined_labels, self.min_cell_volume
                )
                self.cell_labels = refined_labels
                print(f"   - Refined to {refined_labels.max()} cells")
            except Exception as e:
                print(f"   - H-minima skipped ({e}), using original segmentation")

        return self.cell_labels

    def get_cell_boundaries(self) -> np.ndarray:
        """Get boundaries of segmented cells."""
        if self.cell_labels is None:
            raise ValueError("Must segment cells first!")

        boundaries = find_boundaries(self.cell_labels, mode="thick")
        self.cell_boundaries = boundaries
        return boundaries

    def validate_segmentation(self) -> Dict:
        """
        Compute validation metrics for segmentation quality.

        Returns
        -------
        metrics : dict
            Dictionary of validation metrics
        """
        print("\n[4/5] Computing validation metrics...")

        metrics = {}

        if self.nuclei_labels is not None:
            # Nuclei statistics
            nuclei_sizes = ndimage.sum(
                self.nuclei_mask,
                self.nuclei_labels,
                range(self.nuclei_labels.max() + 1),
            )
            nuclei_sizes = nuclei_sizes[1:]  # Exclude background
            metrics["nuclei_count"] = len(nuclei_sizes)
            metrics["nuclei_mean_volume"] = np.mean(nuclei_sizes)
            metrics["nuclei_std_volume"] = np.std(nuclei_sizes)
            print(
                f"   - Nuclei: count={metrics['nuclei_count']}, "
                f"mean_vol={metrics['nuclei_mean_volume']:.1f}±{metrics['nuclei_std_volume']:.1f}"
            )

        if self.cell_labels is not None:
            # Cell statistics
            cell_mask = self.cell_labels > 0
            cell_sizes = ndimage.sum(
                cell_mask, self.cell_labels, range(self.cell_labels.max() + 1)
            )
            cell_sizes = cell_sizes[1:]  # Exclude background
            metrics["cell_count"] = len(cell_sizes)
            metrics["cell_mean_volume"] = np.mean(cell_sizes)
            metrics["cell_std_volume"] = np.std(cell_sizes)
            metrics["cell_min_volume"] = np.min(cell_sizes)
            metrics["cell_max_volume"] = np.max(cell_sizes)
            print(
                f"   - Cells: count={metrics['cell_count']}, "
                f"mean_vol={metrics['cell_mean_volume']:.1f}±{metrics['cell_std_volume']:.1f}, "
                f"range=[{metrics['cell_min_volume']:.1f}, {metrics['cell_max_volume']:.1f}]"
            )

            # Nuclei per cell
            if self.nuclei_labels is not None:
                nuclei_per_cell = []
                for cell_id in range(1, self.cell_labels.max() + 1):
                    cell_mask = self.cell_labels == cell_id
                    nuclei_in_cell = np.unique(self.nuclei_labels[cell_mask])
                    nuclei_in_cell = nuclei_in_cell[nuclei_in_cell > 0]
                    nuclei_per_cell.append(len(nuclei_in_cell))

                metrics["nuclei_per_cell_mean"] = np.mean(nuclei_per_cell)
                metrics["nuclei_per_cell_std"] = np.std(nuclei_per_cell)
                metrics["cells_with_nuclei"] = sum(1 for n in nuclei_per_cell if n > 0)
                metrics["binucleate_cells"] = sum(1 for n in nuclei_per_cell if n >= 2)
                print(
                    f"   - Nuclei distribution: mean={metrics['nuclei_per_cell_mean']:.2f}±{metrics['nuclei_per_cell_std']:.2f}, "
                    f"cells_with_nuclei={metrics['cells_with_nuclei']}, "
                    f"binucleate={metrics['binucleate_cells']}"
                )

        self.stats.update(metrics)
        return metrics

    def run_pipeline(
        self, visualize: bool = True, save_path: Optional[str] = None
    ) -> Dict:
        """
        Run complete segmentation pipeline.

        Parameters
        ----------
        visualize : bool
            Generate visualization plots
        save_path : str, optional
            Path to save segmentation results (NPZ format)

        Returns
        -------
        results : dict
            Dictionary with 'nuclei_labels', 'cell_labels', 'metrics'
        """
        print("\n" + "=" * 60)
        print("3D NUCLEI-AWARE CELL SEGMENTATION PIPELINE")
        print("=" * 60)

        # Step 1: Segment nuclei
        self.segment_nuclei(threshold_method="otsu", closing_radius=2)

        # Step 2: Segment cells
        self.segment_cells(
            use_distance_transform=True,
            watershed_compactness=1.0,
            dilation_iterations=3,
        )

        # Step 3: Refine cell boundaries
        self.separate_touching_cells(use_h_minima=False)

        # Step 4: Get boundaries and validate
        self.get_cell_boundaries()
        metrics = self.validate_segmentation()

        # Step 5: Save results
        print("\n[5/5] Saving results...")
        if save_path:
            np.savez(
                save_path,
                nuclei_labels=self.nuclei_labels,
                cell_labels=self.cell_labels,
                cell_boundaries=self.cell_boundaries,
            )
            print(f"   - Saved to {save_path}")

        # Visualization
        if visualize:
            self.visualize()

        results = {
            "nuclei_labels": self.nuclei_labels,
            "cell_labels": self.cell_labels,
            "cell_boundaries": self.cell_boundaries,
            "metrics": metrics,
        }

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60 + "\n")

        return results

    def visualize(
        self, slice_z: Optional[int] = None, figsize: Tuple[int, int] = (16, 12)
    ):
        """
        Visualize segmentation results at a middle Z-slice.

        Parameters
        ----------
        slice_z : int, optional
            Which Z-slice to visualize (default: middle slice)
        figsize : tuple
            Figure size
        """
        if slice_z is None:
            slice_z = self.nuclei_raw.shape[0] // 2

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(
            f"3D Nuclei-Aware Cell Segmentation (Z-slice {slice_z})", fontsize=14
        )

        # Row 1: Input data
        axes[0, 0].imshow(self.nuclei_raw[slice_z], cmap="hot")
        axes[0, 0].set_title("Raw Nuclei Channel")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(self.cytoplasm_raw[slice_z], cmap="viridis")
        axes[0, 1].set_title("Raw Cytoplasm Channel")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(self.nuclei_raw[slice_z], cmap="hot", alpha=0.6)
        axes[0, 2].imshow(self.cytoplasm_raw[slice_z], cmap="viridis", alpha=0.4)
        axes[0, 2].set_title("Overlay (Nuclei + Cytoplasm)")
        axes[0, 2].axis("off")

        # Row 2: Segmentation results
        if self.nuclei_labels is not None:
            axes[1, 0].imshow(self.nuclei_labels[slice_z], cmap="nipy_spectral")
            axes[1, 0].set_title(
                f"Nuclei Segmentation ({self.nuclei_labels.max()} nuclei)"
            )
            axes[1, 0].axis("off")

        if self.cell_labels is not None:
            axes[1, 1].imshow(self.cell_labels[slice_z], cmap="nipy_spectral")
            axes[1, 1].set_title(f"Cell Segmentation ({self.cell_labels.max()} cells)")
            axes[1, 1].axis("off")

        if self.cell_boundaries is not None:
            axes[1, 2].imshow(self.cytoplasm_raw[slice_z], cmap="gray", alpha=0.7)
            axes[1, 2].imshow(self.cell_boundaries[slice_z], cmap="Reds", alpha=0.5)
            axes[1, 2].set_title("Cell Boundaries on Cytoplasm")
            axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig("nuclei_aware_segmentation.png", dpi=150, bbox_inches="tight")
        print("\nVisualization saved to 'nuclei_aware_segmentation.png'")
        plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING SYNTHETIC 3D MICROSCOPY DATA FOR DEMO")
    print("=" * 60)

    # Create synthetic data for demonstration
    np.random.seed(42)
    Z, Y, X = 30, 256, 256

    # Generate synthetic nuclei
    nuclei_data = np.zeros((Z, Y, X), dtype=np.uint16)
    for _ in range(15):  # 15 nuclei
        z, y, x = (
            np.random.randint(5, Z - 5),
            np.random.randint(20, Y - 20),
            np.random.randint(20, X - 20),
        )
        radius_z, radius_xy = np.random.randint(3, 6), np.random.randint(8, 12)
        for dz in range(-radius_z, radius_z + 1):
            for dy in range(-radius_xy, radius_xy + 1):
                for dx in range(-radius_xy, radius_xy + 1):
                    if (
                        dz**2 / (radius_z**2 + 0.1)
                        + (dy**2 + dx**2) / (radius_xy**2 + 0.1)
                        <= 1
                    ):
                        nuclei_data[z + dz, y + dy, x + dx] = np.random.randint(
                            100, 255
                        )

    # Add noise
    nuclei_data = nuclei_data + np.random.poisson(5, nuclei_data.shape)

    # Generate synthetic cytoplasm (halos around nuclei)
    cytoplasm_data = ndimage.gaussian_filter(nuclei_data.astype(float), sigma=3) * 0.6
    cytoplasm_data = cytoplasm_data + np.random.poisson(3, cytoplasm_data.shape)
    cytoplasm_data = np.clip(cytoplasm_data, 0, 255).astype(np.uint16)

    print(
        f"Created synthetic data: nuclei {nuclei_data.shape}, cytoplasm {cytoplasm_data.shape}\n"
    )

    # Initialize segmentation
    segmenter = NucleiAware3DSegmentation(
        nuclei_data=nuclei_data,
        cytoplasm_data=cytoplasm_data,
        voxel_size=(0.5, 0.2, 0.2),
        min_nucleus_volume=50,
        min_cell_volume=100,
    )

    # Run pipeline
    results = segmenter.run_pipeline(
        visualize=True, save_path="segmentation_results.npz"
    )

    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 60)
    for key, value in results["metrics"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


# In[ ]:


# In[8]:


nuclei = nuclei_mask.copy()
cytoplasm = cyto2.copy()
print(len(np.unique(nuclei)))
seg = NucleiAware3DSegmentation(nuclei, cytoplasm)
results = seg.run_pipeline()
