#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import pandas as pd

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
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


patients = pd.read_csv(
    pathlib.Path(f"{root_dir}/data/patient_IDs.txt"),
    header=None,
    names=["patient"],
    dtype=str,
)["patient"].tolist()
list_of_sc_profile_paths = []
list_of_organoid_profile_paths = []
for patient_id in patients:
    list_of_sc_profile_paths.append(
        pathlib.Path(
            f"{root_dir}/data/{patient_id}/image_based_profiles/2.annotated_profiles/sc_anno.parquet"
        )
    )
    list_of_organoid_profile_paths.append(
        pathlib.Path(
            f"{root_dir}/data/{patient_id}/image_based_profiles/2.annotated_profiles/organoid_anno.parquet"
        )
    )


# In[3]:


sc_metadata_columns = [
    "patient",
    "object_id",
    "unit",
    "dose",
    "treatment",
    "image_set",
    "Well",
    "parent_organoid",
]
organoid_metadata_columns = [
    "patient",
    "object_id",
    "unit",
    "dose",
    "treatment",
    "image_set",
    "Well",
    "single_cell_count",
    "Area.Size.Shape_Organoid_VOLUME",
]


# In[4]:


single_cell_counts = pd.concat(
    [
        pd.read_parquet(path)
        for path in tqdm(list_of_sc_profile_paths, desc="Reading sc profiles")
    ],
    ignore_index=True,
)[sc_metadata_columns]


# In[5]:


organoid_counts = pd.concat(
    [
        pd.read_parquet(path)
        for path in tqdm(
            list_of_organoid_profile_paths, desc="Reading organoid profiles"
        )
    ],
    ignore_index=True,
)
organoid_counts1 = organoid_counts
organoid_counts = organoid_counts[organoid_metadata_columns]


# In[6]:


# replace the single cell count NAN with 0

organoid_counts = organoid_counts.fillna(0)
sc_counts = single_cell_counts.fillna(0)
sc_counts.drop(columns=["object_id"], inplace=True, errors="ignore")
organoid_counts.drop(columns=["object_id"], inplace=True, errors="ignore")


# In[7]:


print("Single cell counts shape:", single_cell_counts.shape)
print("Organoid counts shape:", organoid_counts.shape)


# In[8]:


single_cell_counts


# In[9]:


organoid_counts.drop_duplicates(
    subset=["patient", "image_set", "single_cell_count"],
    inplace=True,
    ignore_index=True,
)
organoid_counts = (
    organoid_counts.groupby(["patient", "unit", "dose", "treatment", "image_set"])
    .sum(numeric_only=True)
    .reset_index()
)
organoid_counts["cell_density"] = (
    organoid_counts["single_cell_count"]
    / organoid_counts["Area.Size.Shape_Organoid_VOLUME"]
)


# In[10]:


sc_counts.drop_duplicates(
    subset=["patient", "Well", "parent_organoid"], inplace=True, ignore_index=True
)
sc_counts = (
    sc_counts.groupby(["patient", "unit", "dose", "treatment", "image_set"])
    .size()
    .reset_index(name="organoid_count")
)


# In[11]:


sc_and_organoid_counts = pd.merge(
    organoid_counts,
    sc_counts,
    how="inner",
    on=[
        "patient",
        "unit",
        "dose",
        "treatment",
        "image_set",
    ],
)


# In[12]:


# save the merged profile counts
pathlib.Path(f"{root_dir}/figures/cell_count_plots/results/").mkdir(
    parents=True, exist_ok=True
)
sc_and_organoid_counts.to_parquet(
    pathlib.Path(
        f"{root_dir}/figures/cell_count_plots/results/sc_and_organoid_counts.parquet"
    ),
    index=False,
)
sc_and_organoid_counts.shape
