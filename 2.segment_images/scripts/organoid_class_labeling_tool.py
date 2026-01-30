#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import json
import os
import pathlib
import sys
import time

import matplotlib.image as mpimg
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
from skimage.filters import sobel

# In[2]:


def save_labels(dictionary: dict, outfile: str):
    """
    Description
    ----------
    Save labels to a parquet file.
    Parameters
    ----------
    dictionary : dict
        Dictionary containing labels to save.
    outfile : str
        Path to the output parquet file.
    Returns
    -------
    None

    """

    try:
        pathlib.Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(dictionary)
        df.to_parquet(outfile, index=False)
    except Exception as e:
        print(f"Error saving labels to {outfile}: {e}")
        return False
    return True


def read_labels(infile: str) -> dict:
    """
    Description
    ----------
    Read labels from a parquet file.
    Parameters
    ----------
    infile : str
        Path to the input parquet file.
    Returns
    -------
    dict
        Dictionary containing the labels.
    """
    data = pd.read_parquet(infile).to_dict(orient="list")
    return data


def check_for_image_labels(
    dictionary: dict,
    patient: str,
    well_fov: str,
    annotator: str,
) -> bool:
    """
    Description
    ----------
    Check if an image has already been labeled.
    Parameters
    ----------
    dictionary : dict
        Dictionary containing existing labels.
    patient : str
        Patient identifier.
    well_fov : str
        Well FOV identifier.
    annotator : str
        Annotator name.
    Returns
    -------
    bool
        True if the image has been labeled, False otherwise.
    """
    for i in range(len(dictionary["patient"])):
        if (
            dictionary["patient"][i] == patient
            and dictionary["well_fov"][i] == well_fov
        ):
            return True
    return False


def label_images_keypress(
    image_dict: dict, label_map: dict, labels_save_file: pathlib.Path
) -> dict:
    """
    Label images using keyboard input.

    Parameters
    ----------
    image_paths : list of str
    label_map : dict
        Mapping from key press (str) to label value

    Returns
    -------
    dict
    """
    annotator = input("Enter annotator name: ")

    labels = {"patient": [], "well_fov": [], "label": [], "annotator": []}
    if labels_save_file.exists():
        labels = read_labels(labels_save_file)
    for i, image_path in enumerate(image_dict["image_path"]):
        if check_for_image_labels(
            dictionary=labels,
            patient=image_dict["patient"][i],
            well_fov=image_dict["well_fov"][i],
            annotator=annotator,
        ):
            continue
        image = read_zstack_image(image_path)
        # load the middle slice to check if there is anything there
        mid_slice = image.shape[0] // 2
        image_mid = image[mid_slice, :, :]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image_mid, cmap="inferno")
        ax.axis("off")
        plt.show(block=False)
        key = input("Press key for label: ")
        plt.close(fig)
        labels["annotator"].append(annotator)
        labels["patient"].append(image_dict["patient"][i])
        labels["well_fov"].append(image_dict["well_fov"][i])
        labels["label"].append(label_map.get(key, None))
        save_labels(labels, labels_save_file)

    return labels


# In[3]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[4]:


root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[5]:


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
    input_subparent_name = "zstack_images"
    mask_subparent_name = "segmentation_masks"


input_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{input_subparent_name}/"
).resolve(strict=True)
mask_path = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{mask_subparent_name}/"
).resolve()
mask_path.mkdir(exist_ok=True, parents=True)
labels_save_file = pathlib.Path(
    "../image_labels/organoid_image_labels.parquet"
).resolve()
labels_save_file.parent.mkdir(exist_ok=True, parents=True)


# In[6]:


# get all well fovs for this patient
all_well_fovs = input_dir.glob("*")
all_well_fovs = sorted([wf.name for wf in all_well_fovs if wf.is_dir()])
images_to_process = {"patient": [], "well_fov": [], "image_path": []}
for well_fov in tqdm.tqdm(all_well_fovs):
    image_path = pathlib.Path(f"{input_dir}/{well_fov}/").resolve(strict=True)
    image_to_load = [x for x in image_path.glob("*.tif") if "555" in x.name]
    images_to_process["patient"].append(patient)
    images_to_process["well_fov"].append(well_fov)
    images_to_process["image_path"].append(image_to_load)


# In[7]:


label_map = {"1": "globular", "2": "dissociated", "3": "small", "4": "elongated"}


# In[8]:


labels = label_images_keypress(images_to_process, label_map, labels_save_file)


# In[9]:


labels = read_labels(labels_save_file)
# show stats for the labeling
df = pd.DataFrame(labels)
print("Label counts:")
print(df["label"].value_counts())
# print a list of well fovs for each label
for label in label_map.values():
    well_fovs = df[df["label"] == label][["patient", "well_fov"]]
    print(f"\nWell FOVs for label '{label}':")
    counter = 0
    for index, row in well_fovs.iterrows():
        if counter >= 10:
            break
        print(f"{row['well_fov']}")
        counter += 1
