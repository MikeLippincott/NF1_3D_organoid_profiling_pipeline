#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import pathlib
import sys
import time

import psutil

sys.path.append("../featurization_utils")
import multiprocessing
import os
from functools import partial
from itertools import product

import cucim
import cupy as cp
import numpy
import numpy as np
import pandas as pd
import psutil
import scipy
import skimage
from granularity_utils import measure_3D_granularity

# from granularity import measure_3D_granularity
from loading_classes import ImageSetLoader, ObjectLoader
from resource_profiling_util import get_mem_and_time_profiling

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# In[2]:


def process_combination(
    args: tuple[str, str],
    image_set_loader: ImageSetLoader,
    output_parent_path: pathlib.Path,
):
    """
    Process a single combination of compartment and channel.

    Parameters
    ----------
    args : tuple
        Args that contain the compartment and channel.
        Ordered as (compartment, channel).
        Yes, order matters here.
        channel : str
            The channel name.
        compartment : str
            The compartment name.
    image_set_loader : Class ImageSetLoader
        This contains the image information needed to retreive the objects.
    output_parent_path : pathlib.Path
        The parent path to save the output files to.
        This is used to create the output directory if it doesn't exist.
    Returns
    -------
    str
        A string indicating the compartment and channel that was processed.
    """
    compartment, channel = args
    object_loader = ObjectLoader(
        image_set_loader.image_set_dict[channel],
        image_set_loader.image_set_dict[compartment],
        channel,
        compartment,
    )
    object_measurements = measure_3D_granularity(
        object_loader=object_loader,
        radius=10,  # radius of the sphere to use for granularity measurement
        granular_spectrum_length=16,  # usually 16 but 2 is used for testing for now
        subsample_size=0.25,  # subsample to 25% of the image to reduce computation time
        image_name=channel,
    )
    final_df = pd.DataFrame(object_measurements)
    # get the mean of each value in the array
    # melt the dataframe to wide format
    final_df = final_df.pivot_table(
        index=["object_id"], columns=["feature"], values=["value"]
    )
    final_df.columns = final_df.columns.droplevel()
    final_df = final_df.reset_index()
    # prepend compartment and channel to column names
    for col in final_df.columns:
        if col == "object_id":
            continue
        else:
            final_df.rename(
                columns={col: f"Granularity_{compartment}_{channel}_{col}"},
                inplace=True,
            )
    final_df.insert(0, "image_set", image_set_loader.image_set_name)
    output_file = pathlib.Path(
        output_parent_path / f"Granularity_{compartment}_{channel}_features.parquet"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_file)

    return f"Processed {compartment} - {channel}"


# In[3]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--well_fov",
        type=str,
        default="None",
        help="Well and field of view to process, e.g. 'A01_1'",
    )

    argparser.add_argument(
        "--patient",
        type=str,
        help="Patient ID, e.g. 'NF0014'",
    )

    args = argparser.parse_args()
    well_fov = args.well_fov
    patient = args.patient
    if well_fov == "None":
        raise ValueError(
            "Please provide a well and field of view to process, e.g. 'A01_1'"
        )

else:
    well_fov = "C4-2"
    patient = "NF0014"

image_set_path = pathlib.Path(f"../../data/{patient}/cellprofiler/{well_fov}/")
output_parent_path = pathlib.Path(
    f"../../data/{patient}/extracted_features/{well_fov}/"
)
output_parent_path.mkdir(parents=True, exist_ok=True)


# In[4]:


channel_mapping = {
    "DNA": "405",
    "AGP": "488",
    "ER": "555",
    "Mito": "640",
    "BF": "TRANS",
    "Nuclei": "nuclei_",
    "Cell": "cell_",
    "Cytoplasm": "cytoplasm_",
    "Organoid": "organoid_",
}


# In[5]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[6]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[7]:


# Generate all combinations of compartments and channels
combinations = list(
    product(image_set_loader.compartments, image_set_loader.image_names)
)
# cores = multiprocessing.cpu_count()
cores = 4
print(f"Using {cores} cores for processing.")
# Use multiprocessing to process combinations in parallel
with multiprocessing.Pool(processes=cores) as pool:
    results = list(
        tqdm(
            pool.imap(
                partial(
                    process_combination,
                    image_set_loader=image_set_loader,
                    output_parent_path=output_parent_path,
                ),
                combinations,
            ),
            desc="Processing combinations",
        )
    )

print("Processing complete.")


# In[8]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
get_mem_and_time_profiling(
    start_mem=start_mem,
    end_mem=end_mem,
    start_time=start_time,
    end_time=end_time,
    feature_type="Granularity",
    well_fov=well_fov,
    patient_id=patient,
    CPU_GPU="CPU",
    output_file_dir=pathlib.Path(
        f"../../data/{patient}/extracted_features/run_stats/{well_fov}_Granularity_CPU.parquet"
    ),
)


# In[9]:


pd.read_parquet(
    "../../data/NF0014/extracted_features/C4-2/Granularity_Nuclei_Mito_features.parquet"
)
