#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copairs import Matcher, MatcherMultilabel, map
from copairs.matching import assign_reference_index

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")


# In[2]:


paths_to_process_dict = {
    "organoid_fs": {
        "mAP": {
            "inter_patient_input": pathlib.Path(
                f"{root_dir}/5.EDA/results/mAP/organoid_fs_inter_patient_mAP.parquet"
            ).resolve(strict=True),
            "intra_patient_input": pathlib.Path(
                f"{root_dir}/5.EDA/results/mAP/organoid_fs_intra_patient_mAP.parquet"
            ).resolve(strict=True),
        },
        "distance_metrics": {
            "inter_patient_input": pathlib.Path(
                f"{root_dir}/5.EDA/results/distance_metrics/organoid_fs_inter_patient_distance_metrics.parquet"
            ).resolve(strict=True),
            "intra_patient_input": pathlib.Path(
                f"{root_dir}/5.EDA/results/distance_metrics/organoid_fs_intra_patient_distance_metrics.parquet"
            ).resolve(strict=True),
        },
        "inter_patient_output": pathlib.Path(
            f"{root_dir}/5.EDA/results/drug_hit_merged_metrics/organoid_fs_inter_patient_drug_hit_merged_metrics.parquet"
        ).resolve(),
        "intra_patient_output": pathlib.Path(
            f"{root_dir}/5.EDA/results/drug_hit_merged_metrics/organoid_fs_intra_patient_drug_hit_merged_metrics.parquet"
        ).resolve(),
        "inter_drug_hit_list_output": pathlib.Path(
            f"{root_dir}/5.EDA/results/drug_hit_lists/organoid_fs_inter_patient_drug_hit_list.parquet"
        ).resolve(),
        "intra_drug_hit_list_output": pathlib.Path(
            f"{root_dir}/5.EDA/results/drug_hit_lists/organoid_fs_intra_patient_drug_hit_list.parquet"
        ).resolve(),
        "metadata_columns": [
            "patient",
            "object_id",
            "unit",
            "dose",
            "treatment",
            "Target",
            "Class",
            "Therapeutic Categories",
            "image_set",
            "Well",
            "single_cell_count",
            "reference_index",
        ],
    },
    "single_cell_fs": {
        "mAP": {
            "inter_patient_input": pathlib.Path(
                f"{root_dir}/5.EDA/results/mAP/sc_fs_inter_patient_mAP.parquet"
            ).resolve(strict=True),
            "intra_patient_input": pathlib.Path(
                f"{root_dir}/5.EDA/results/mAP/sc_fs_intra_patient_mAP.parquet"
            ).resolve(strict=True),
        },
        "distance_metrics": {
            "inter_patient_input": pathlib.Path(
                f"{root_dir}/5.EDA/results/distance_metrics/sc_fs_inter_patient_distance_metrics.parquet"
            ).resolve(strict=True),
            "intra_patient_input": pathlib.Path(
                f"{root_dir}/5.EDA/results/distance_metrics/sc_fs_intra_patient_distance_metrics.parquet"
            ).resolve(strict=True),
        },
        "inter_patient_output": pathlib.Path(
            f"{root_dir}/5.EDA/results/drug_hit_merged_metrics/sc_fs_inter_patient_drug_hit_merged_metrics.parquet"
        ).resolve(),
        "intra_patient_output": pathlib.Path(
            f"{root_dir}/5.EDA/results/drug_hit_merged_metrics/sc_fs_intra_patient_drug_hit_merged_metrics.parquet"
        ).resolve(),
        "inter_drug_hit_list_output": pathlib.Path(
            f"{root_dir}/5.EDA/results/drug_hit_lists/sc_fs_inter_patient_drug_hit_list.parquet"
        ).resolve(),
        "intra_drug_hit_list_output": pathlib.Path(
            f"{root_dir}/5.EDA/results/drug_hit_lists/sc_fs_intra_patient_drug_hit_list.parquet"
        ).resolve(),
        "metadata_columns": [
            "patient",
            "object_id",
            "unit",
            "dose",
            "treatment",
            "Target",
            "Class",
            "Therapeutic Categories",
            "image_set",
            "Well",
            "reference_index",
        ],
    },
}


# ## organoid fs inter patient mAP and distance metrics

# In[ ]:


organoid_inter_metric_df = pd.read_parquet(
    paths_to_process_dict["organoid_fs"]["distance_metrics"]["inter_patient_input"]
)
organoid_inter_mAP_df = pd.read_parquet(
    paths_to_process_dict["organoid_fs"]["mAP"]["inter_patient_input"]
)
# merge the two DataFrames on 'treament'
organoid_fs_merged_inter_patient_df = organoid_inter_mAP_df.merge(
    organoid_inter_metric_df,
    on=["treatment"],
    how="left",
).drop(columns=["indices"])


# ## organoid fs intra patient mAP and distance metrics

# In[ ]:


organoid_intra_metric_df = pd.read_parquet(
    paths_to_process_dict["organoid_fs"]["distance_metrics"]["intra_patient_input"]
)
organoid_intra_mAP_df = pd.read_parquet(
    paths_to_process_dict["organoid_fs"]["mAP"]["intra_patient_input"]
)
# merge the two DataFrames on 'treament'
organoid_fs_merged_intra_patient_df = organoid_intra_mAP_df.merge(
    organoid_intra_metric_df,
    on=["treatment", "patient"],
    how="left",
).drop(columns=["indices"])


# ## Single cell fs inter patient mAP and distance metrics

# In[ ]:


sc_inter_metric_df = pd.read_parquet(
    paths_to_process_dict["single_cell_fs"]["distance_metrics"]["inter_patient_input"]
)
sc_inter_mAP_df = pd.read_parquet(
    paths_to_process_dict["single_cell_fs"]["mAP"]["inter_patient_input"]
)
# merge the two DataFrames on 'treament'
sc_fs_merged_inter_patient_df = sc_inter_mAP_df.merge(
    sc_inter_metric_df,
    on=["treatment"],
    how="left",
).drop(columns=["indices"])


# ## Single cell fs intra patient mAP and distance metrics

# In[ ]:


sc_intra_patient_metric_df = pd.read_parquet(
    paths_to_process_dict["single_cell_fs"]["distance_metrics"]["intra_patient_input"]
)
sc_intra_patient_mAP_df = pd.read_parquet(
    paths_to_process_dict["single_cell_fs"]["mAP"]["intra_patient_input"]
)
# merge the two DataFrames on 'treament'
sc_fs_merged_intra_patient_df = sc_intra_patient_mAP_df.merge(
    sc_intra_patient_metric_df,
    on=["treatment", "patient"],
    how="left",
).drop(columns=["indices"])


# ## Save the merged metrics

# In[7]:


paths_to_process_dict["organoid_fs"]["inter_patient_output"].parent.mkdir(
    parents=True, exist_ok=True
)
organoid_fs_merged_inter_patient_df.to_parquet(
    paths_to_process_dict["organoid_fs"]["inter_patient_output"],
    index=False,
)
organoid_fs_merged_intra_patient_df.to_parquet(
    paths_to_process_dict["organoid_fs"]["intra_patient_output"],
    index=False,
)
sc_fs_merged_inter_patient_df.to_parquet(
    paths_to_process_dict["single_cell_fs"]["inter_patient_output"],
    index=False,
)
sc_fs_merged_intra_patient_df.to_parquet(
    paths_to_process_dict["single_cell_fs"]["intra_patient_output"],
    index=False,
)


# ## Threshold and gate the metrics to generate the drug hits
#
# This should be done at the drug and patient level for the intra patient metrics
# and just the drug level for the inter patient metrics

# ### Organoid fs inter patient drug hit merged metrics

# In[8]:


# first the drug only level metrics
organoid_fs_merged_inter_patient_df.head()
mAP_mean = organoid_fs_merged_inter_patient_df["mean_average_precision"].mean()
cosine_distance_mean_mean = organoid_fs_merged_inter_patient_df[
    "cosine_distance_mean"
].mean()
cosine_distance_std_mean = organoid_fs_merged_inter_patient_df[
    "cosine_distance_std"
].mean()
# find all drugs that have a mAP > mAP mean
# cosine_distance_mean > cosine_distance_mean_mean
# cosine_distance_std < cosine_distance_std_mean
drug_hits = organoid_fs_merged_inter_patient_df[
    (organoid_fs_merged_inter_patient_df["mean_average_precision"] > mAP_mean)
    & (
        organoid_fs_merged_inter_patient_df["cosine_distance_mean"]
        > cosine_distance_mean_mean
    )
    & (
        organoid_fs_merged_inter_patient_df["cosine_distance_std"]
        < cosine_distance_std_mean
    )
]
paths_to_process_dict["organoid_fs"]["inter_drug_hit_list_output"].parent.mkdir(
    parents=True, exist_ok=True
)

pd.DataFrame(drug_hits["treatment"].unique(), columns=["treatment"]).to_parquet(
    paths_to_process_dict["organoid_fs"]["inter_drug_hit_list_output"],
    index=False,
)


# ## Organoid fs intra patient drug hit merged metrics

# In[ ]:


# intra patient metrics
# generate drug hits on the per patient basis
intra_patient_drug_hit_dict = {
    "patient": [],
    "treatment": [],
}
for patient in organoid_fs_merged_intra_patient_df["patient"].unique():
    patient_df = organoid_fs_merged_intra_patient_df[
        organoid_fs_merged_intra_patient_df["patient"] == patient
    ]
    mAP_mean = patient_df["mean_average_precision"].mean()
    cosine_distance_mean_mean = patient_df["cosine_distance_mean"].mean()
    cosine_distance_std_mean = patient_df["cosine_distance_std"].mean()
    # find all drugs that have a mAP > mAP mean
    # cosine_distance_mean > cosine_distance_mean_mean
    # cosine_distance_std < cosine_distance_std_mean
    drug_hits = patient_df[
        (patient_df["mean_average_precision"] > mAP_mean)
        & (patient_df["cosine_distance_mean"] > cosine_distance_mean_mean)
        & (patient_df["cosine_distance_std"] < cosine_distance_std_mean)
    ]["treatment"].unique()
    intra_patient_drug_hit_dict["patient"].extend([patient] * len(drug_hits))
    intra_patient_drug_hit_dict["treatment"].extend(drug_hits)
intra_patient_drug_hits_df = pd.DataFrame(intra_patient_drug_hit_dict)
intra_patient_drug_hits_df.to_parquet(
    paths_to_process_dict["organoid_fs"]["intra_drug_hit_list_output"],
    index=False,
)


# ### Single cell fs inter patient mAP and distance metrics

# In[10]:


sc_fs_merged_inter_patient_df.head()
mAP_mean = sc_fs_merged_inter_patient_df["mean_average_precision"].mean()
cosine_distance_mean_mean = sc_fs_merged_inter_patient_df["cosine_distance_mean"].mean()
cosine_distance_std_mean = sc_fs_merged_inter_patient_df["cosine_distance_std"].mean()
# find all drugs that have a mAP > mAP mean
# cosine_distance_mean > cosine_distance_mean_mean
# cosine_distance_std < cosine_distance_std_mean
sc_drug_hits = sc_fs_merged_inter_patient_df[
    (sc_fs_merged_inter_patient_df["mean_average_precision"] > mAP_mean)
    & (
        sc_fs_merged_inter_patient_df["cosine_distance_mean"]
        > cosine_distance_mean_mean
    )
    & (sc_fs_merged_inter_patient_df["cosine_distance_std"] < cosine_distance_std_mean)
]
pd.DataFrame(sc_drug_hits["treatment"].unique(), columns=["treatment"]).to_parquet(
    paths_to_process_dict["single_cell_fs"]["inter_drug_hit_list_output"],
    index=False,
)


# ### Single cell fs intra patient mAP and distance metrics

# In[11]:


sc_fs_merged_intra_patient_df.head()
intra_patient_sc_drug_hit_dict = {
    "patient": [],
    "treatment": [],
}
for patient in sc_fs_merged_intra_patient_df["patient"].unique():
    patient_df = sc_fs_merged_intra_patient_df[
        sc_fs_merged_intra_patient_df["patient"] == patient
    ]
    mAP_mean = patient_df["mean_average_precision"].mean()
    cosine_distance_mean_mean = patient_df["cosine_distance_mean"].mean()
    cosine_distance_std_mean = patient_df["cosine_distance_std"].mean()
    # find all drugs that have a mAP > mAP mean
    # cosine_distance_mean > cosine_distance_mean_mean
    # cosine_distance_std < cosine_distance_std_mean
    drug_hits = patient_df[
        (patient_df["mean_average_precision"] > mAP_mean)
        & (patient_df["cosine_distance_mean"] > cosine_distance_mean_mean)
        & (patient_df["cosine_distance_std"] < cosine_distance_std_mean)
    ]["treatment"].unique()
    intra_patient_sc_drug_hit_dict["patient"].extend([patient] * len(drug_hits))
    intra_patient_sc_drug_hit_dict["treatment"].extend(drug_hits)
intra_patient_sc_drug_hits_df = pd.DataFrame(intra_patient_sc_drug_hit_dict)
intra_patient_sc_drug_hits_df.to_parquet(
    paths_to_process_dict["single_cell_fs"]["intra_drug_hit_list_output"],
    index=False,
)


# In[12]:


# read the drug hit lists
organoid_fs_inter_patient_drug_hits = pd.read_parquet(
    paths_to_process_dict["organoid_fs"]["inter_drug_hit_list_output"]
)
organoid_fs_intra_patient_drug_hits = pd.read_parquet(
    paths_to_process_dict["organoid_fs"]["intra_drug_hit_list_output"]
)
sc_fs_inter_patient_drug_hits = pd.read_parquet(
    paths_to_process_dict["single_cell_fs"]["inter_drug_hit_list_output"]
)
sc_fs_intra_patient_drug_hits = pd.read_parquet(
    paths_to_process_dict["single_cell_fs"]["intra_drug_hit_list_output"]
)
# print the drug hit lists
print("Organoid FS Inter Patient Drug Hits:")
print(organoid_fs_inter_patient_drug_hits)
print("\nOrganoid FS Intra Patient Drug Hits:")
print(organoid_fs_intra_patient_drug_hits)
print("\nSingle Cell FS Inter Patient Drug Hits:")
print(sc_fs_inter_patient_drug_hits)
print("\nSingle Cell FS Intra Patient Drug Hits:")
print(sc_fs_intra_patient_drug_hits)
