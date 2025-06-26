#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import sys
import time

import psutil

sys.path.append("../featurization_utils")
import os
from itertools import product

import cucim
import cupy as cp
import numpy
import numpy as np
import pandas as pd
import psutil
import scipy
import skimage
from featurization_parsable_arguments import parse_featurization_args
from granularity_utils import measure_3D_granularity, measure_3D_granularity_gpu

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


if not in_notebook:
    arguments_dict = parse_featurization_args()
    patient = arguments_dict["patient"]
    well_fov = arguments_dict["well_fov"]
    channel = arguments_dict["channel"]
    compartment = arguments_dict["compartment"]
    processor_type = arguments_dict["processor_type"]

else:
    well_fov = "C4-2"
    patient = "NF0014"
    channel = "DNA"
    compartment = "Nuclei"
    processor_type = "CPU"

image_set_path = pathlib.Path(f"../../data/{patient}/cellprofiler/{well_fov}/")
output_parent_path = pathlib.Path(
    f"../../data/{patient}/extracted_features/{well_fov}/"
)
output_parent_path.mkdir(parents=True, exist_ok=True)


# In[3]:


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


# In[4]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[5]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[6]:


object_loader = ObjectLoader(
    image_set_loader.image_set_dict[channel],
    image_set_loader.image_set_dict[compartment],
    channel,
    compartment,
)
if processor_type == "GPU":
    object_measurements = measure_3D_granularity_gpu(
        object_loader=object_loader,
        radius=10,  # radius of the sphere to use for granularity measurement
        granular_spectrum_length=16,  # usually 16 but 2 is used for testing for now
        subsample_size=0.25,  # subsample to 25% of the image to reduce computation time
        image_name=channel,
    )
elif processor_type == "CPU":
    object_measurements = measure_3D_granularity(
        object_loader=object_loader,
        radius=10,  # radius of the sphere to use for granularity measurement
        granular_spectrum_length=16,  # usually 16 but 2 is used for testing for now
        subsample_size=0.25,  # subsample to 25% of the image to reduce computation time
        image_name=channel,
    )
else:
    raise ValueError(
        f"Processor type {processor_type} is not supported. Use 'CPU' or 'GPU'."
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


# In[7]:


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
    channel=channel,
    compartment=compartment,
    CPU_GPU=processor_type,
    output_file_dir=pathlib.Path(
        f"../../data/{patient}/extracted_features/run_stats/{well_fov}_{channel}_{compartment}_Granularity_{processor_type}.parquet"
    ),
)
