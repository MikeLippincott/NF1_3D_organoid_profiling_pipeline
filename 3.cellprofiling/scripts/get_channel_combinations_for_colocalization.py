#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import pathlib
import sys
from itertools import product

import pandas as pd

sys.path.append("../featurization_utils")
from loading_classes import ImageSetLoader

# In[2]:


well_fov = "C4-2"
patient = "NF0014"
channel = "DNA"
compartment = "Nuclei"
processor_type = "CPU"

image_set_path = pathlib.Path(f"../../data/{patient}/cellprofiler/{well_fov}/")
output_channel_combinations_path = pathlib.Path(
    "../load_data/output_channel_combinations.parquet"
)
output_channel_combinations_path.parent.mkdir(parents=True, exist_ok=True)


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


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[5]:


# get all channel combinations
channel_combinations = list(itertools.combinations(image_set_loader.image_names, 2))


# In[6]:


combinations = [
    (compartment, channel1, channel2)
    for compartment, (channel1, channel2) in product(
        image_set_loader.compartments, channel_combinations
    )
]
channel_combinations_df = pd.DataFrame(
    combinations, columns=["compartment", "channel1", "channel2"]
)
channel_combinations_df.to_parquet(
    output_channel_combinations_path,
    index=False,
    engine="pyarrow",
)
