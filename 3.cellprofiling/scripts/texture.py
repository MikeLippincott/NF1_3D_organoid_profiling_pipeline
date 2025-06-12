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
import gc
import multiprocessing
import pathlib
from functools import partial
from itertools import product

import pandas as pd
import tqdm
from loading_classes import ImageSetLoader, ObjectLoader
from resource_profiling_util import get_mem_and_time_profiling
from texture_utils import measure_3D_texture
from tqdm import tqdm

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
) -> str:
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
        The parent path where the output files will be saved.
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
    output_texture_dict = measure_3D_texture(
        object_loader=object_loader,
        distance=1,
    )
    final_df = pd.DataFrame(output_texture_dict)

    final_df = final_df.pivot(
        index="object_id",
        columns="texture_name",
        values="texture_value",
    )
    final_df.reset_index(inplace=True)
    for col in final_df.columns:
        if col == "object_id":
            continue
        else:
            final_df.rename(
                columns={col: f"Texture_{compartment}_{channel}_{col}"},
                inplace=True,
            )
    final_df.insert(0, "image_set", image_set_loader.image_set_name)
    final_df.columns.name = None

    output_file = pathlib.Path(
        output_parent_path / f"Texture_{compartment}_{channel}_features.parquet"
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


if __name__ == "__main__":
    # Generate all combinations of compartments and channels
    combinations = list(
        product(image_set_loader.compartments, image_set_loader.image_names)
    )
    cores = multiprocessing.cpu_count()
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
    feature_type="Texture",
    well_fov=well_fov,
    patient_id=patient,
    CPU_GPU="CPU",
    output_file_dir=pathlib.Path(
        f"../../data/{patient}/extracted_features/run_stats/{well_fov}_Texture_CPU.parquet"
    ),
)
