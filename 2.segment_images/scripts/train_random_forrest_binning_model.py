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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# In[2]:


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


# In[3]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[4]:


root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)
patient_list_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(
    strict=True
)


# In[5]:


labels_save_file = pathlib.Path(
    "../image_labels/organoid_image_labels.parquet"
).resolve()
sammed_features_save_path = pathlib.Path(
    f"../../3.cellprofiling/results/sammed_features.parquet"
).resolve()
labels = read_labels(labels_save_file)
labels_df = pd.DataFrame(labels)
labels_df
sammed_features_df = pd.read_parquet(sammed_features_save_path)
sammed_features_df
df = pd.merge(
    sammed_features_df,
    labels_df,
    on=["patient", "well_fov"],
    how="right",
)
# drop rows with na
df = df.dropna(subset=["label"])
df


# In[8]:


# set up data splits
# train: 70%, val: 15%, test: 15%
# stratify by label, patient

train_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df[["label"]],
)
train_df, val_df = train_test_split(
    train_df,
    test_size=0.1765,  # 0.1765 * 0.85 = 0.15
    random_state=42,
    stratify=train_df[["label"]],
)
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")


# In[9]:


# train a random forest classifier for the organoid labels


rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(
    train_df.drop(columns=["patient", "well_fov", "label", "annotator"]),
    train_df["label"],
)
val_preds = rf_model.predict(
    val_df.drop(columns=["patient", "well_fov", "label", "annotator"])
)
print("Validation Classification Report:")
print(classification_report(val_df["label"], val_preds))
print("Validation Confusion Matrix:")
print(confusion_matrix(val_df["label"], val_preds))
test_preds = rf_model.predict(
    test_df.drop(columns=["patient", "well_fov", "label", "annotator"])
)
print("Test Classification Report:")
print(classification_report(test_df["label"], test_preds))
print("Test Confusion Matrix:")
print(confusion_matrix(test_df["label"], test_preds))


# In[ ]:
