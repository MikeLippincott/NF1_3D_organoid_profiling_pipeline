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
import itertools
import multiprocessing
from functools import partial
from itertools import product

import pandas as pd
from colocalization_utils import (
    measure_3D_colocalization,
    prepare_two_images_for_colocalization,
)
from loading_classes import ImageSetLoader, TwoObjectLoader
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

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# In[2]:


def process_combination(
    args: tuple[str, str],
    image_set_loader: ImageSetLoader,
    output_parent_path: pathlib.Path,
) -> str:
    """
    Process a single combination of compartment and channel pair for colocalization analysis.

    Parameters
    ----------
    args : tuple
        A tuple containing the compartment, channel1, and channel2.
        Yes, order matters.
        args = (compartment, channel1, channel2)
        Where:
        compartment : str
            The compartment to process.
        channel1 : str
            The first channel to process.
        channel2 : str
            The second channel to process.

    image_set_loader : ImageSetLoader
        An instance of the ImageSetLoader class that loads the images and metadata.

    output_parent_path : pathlib.Path
        The parent directory where the output files will be saved.
    Returns
    -------
    str
        A message indicating the completion of processing for the given combination.
    """
    compartment, channel1, channel2 = args
    coloc_loader = TwoObjectLoader(
        image_set_loader=image_set_loader,
        compartment=compartment,
        channel1=channel1,
        channel2=channel2,
    )

    output_dir = pathlib.Path(
        output_parent_path
        / f"Colocalization_{compartment}_{channel1}.{channel2}_features.parquet"
    )
    list_of_dfs = []
    for object_id in coloc_loader.object_ids:
        cropped_image1, cropped_image2 = prepare_two_images_for_colocalization(
            label_object1=coloc_loader.label_image,
            label_object2=coloc_loader.label_image,
            image_object1=coloc_loader.image1,
            image_object2=coloc_loader.image2,
            object_id1=object_id,
            object_id2=object_id,
        )
        colocalization_features = measure_3D_colocalization(
            cropped_image_1=cropped_image1,
            cropped_image_2=cropped_image2,
            thr=15,
            fast_costes="Accurate",
        )
        coloc_df = pd.DataFrame(colocalization_features, index=[0])
        coloc_df.columns = [
            f"Colocalization_{compartment}_{channel1}.{channel2}_{col}"
            for col in coloc_df.columns
        ]
        coloc_df.insert(0, "object_id", object_id)
        coloc_df.insert(1, "image_set", image_set_loader.image_set_name)
        list_of_dfs.append(coloc_df)
    coloc_df = pd.concat(list_of_dfs, ignore_index=True)
    coloc_df.to_parquet(output_dir)

    return f"Processed {compartment} - {channel1}.{channel2}"


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


# get all channel combinations
channel_combinations = list(itertools.combinations(image_set_loader.image_names, 2))


# runs upon converted script execution

# In[8]:


combinations = [
    (compartment, channel1, channel2)
    for compartment, (channel1, channel2) in product(
        image_set_loader.compartments, channel_combinations
    )
]


# In[9]:


# Specify the number of cores to use
cores_to_use = multiprocessing.cpu_count()  # Adjust the number of cores as needed
print(f"Using {cores_to_use} cores for processing.")

# Use multiprocessing to process combinations in parallel
with multiprocessing.Pool(processes=cores_to_use) as pool:
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


# In[10]:


results


# In[11]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
get_mem_and_time_profiling(
    start_mem=start_mem,
    end_mem=end_mem,
    start_time=start_time,
    end_time=end_time,
    feature_type="Colocalization",
    well_fov=well_fov,
    patient_id=patient,
    CPU_GPU="CPU",
    output_file_dir=pathlib.Path(
        f"../../data/{patient}/extracted_features/run_stats/{well_fov}_Colocalization_CPU.parquet"
    ),
)
