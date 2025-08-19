#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import pandas as pd
import umap

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

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


# paths to data
data_dict = {
    "sc": {
        "input": pathlib.Path(
            f"{root_dir}/data/all_patient_profiles/sc_profiles.parquet"
        ).resolve(strict=True),
        "output": pathlib.Path(f"{root_dir}/5.EDA/results/sc_umap.parquet").resolve(),
    },
    "sc_fs": {
        "input": pathlib.Path(
            f"{root_dir}/data/all_patient_profiles/sc_fs_profiles.parquet"
        ).resolve(strict=True),
        "output": pathlib.Path(
            f"{root_dir}/5.EDA/results/sc_fs_umap.parquet"
        ).resolve(),
    },
    "sc_agg": {
        "input": pathlib.Path(
            f"{root_dir}/data/all_patient_profiles/sc_agg_profiles.parquet"
        ).resolve(strict=True),
        "output": pathlib.Path(
            f"{root_dir}/5.EDA/results/sc_agg_umap.parquet"
        ).resolve(),
    },
    "organoid": {
        "input": pathlib.Path(
            f"{root_dir}/data/all_patient_profiles/organoid_profiles.parquet"
        ).resolve(strict=True),
        "output": pathlib.Path(
            f"{root_dir}/5.EDA/results/organoid_umap.parquet"
        ).resolve(),
    },
    "organoid_fs": {
        "input": pathlib.Path(
            f"{root_dir}/data/all_patient_profiles/organoid_fs_profiles.parquet"
        ).resolve(strict=True),
        "output": pathlib.Path(
            f"{root_dir}/5.EDA/results/organoid_fs_umap.parquet"
        ).resolve(),
    },
    "organoid_agg": {
        "input": pathlib.Path(
            f"{root_dir}/data/all_patient_profiles/organoid_agg_profiles.parquet"
        ).resolve(strict=True),
        "output": pathlib.Path(
            f"{root_dir}/5.EDA/results/organoid_agg_umap.parquet"
        ).resolve(),
    },
    "sc_consensus": {
        "input": pathlib.Path(
            f"{root_dir}/data/all_patient_profiles/sc_consensus_profiles.parquet"
        ).resolve(strict=True),
        "output": pathlib.Path(
            f"{root_dir}/5.EDA/results/sc_consensus_umap.parquet"
        ).resolve(),
    },
    "organoid_consensus": {
        "input": pathlib.Path(
            f"{root_dir}/data/all_patient_profiles/organoid_consensus_profiles.parquet"
        ).resolve(strict=True),
        "output": pathlib.Path(
            f"{root_dir}/5.EDA/results/organoid_consensus_umap.parquet"
        ).resolve(),
    },
}

data_dict["organoid"]["output"].parent.mkdir(parents=True, exist_ok=True)


# In[3]:


metadata_columns = [
    "patient",
    "object_id",
    "unit",
    "dose",
    "treatment",
    "image_set",
    "Well",
    "single_cell_count",
    "parent_organoid",
    "Treatment",
    "Target",
    "Function",
    "Class",
    "Therapeutic Categories",
]


# In[4]:


umap_object = umap.UMAP(
    n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0
)

for dataset, paths in data_dict.items():
    # Load the data
    df = pd.read_parquet(data_dict[dataset]["input"])

    metadata_df = df.copy()
    metadata_subset = []
    for col in metadata_columns:
        if col in df.columns:
            metadata_subset.append(col)

    metadata_df = df[metadata_subset]
    features_df = df.drop(columns=metadata_columns, errors="ignore")
    print(features_df.shape)
    # remove NaN values
    features_df = features_df.dropna(axis=0, how="any")
    print(f"Data shape after dropping NaN values: {features_df.shape}")
    # Extract features and apply UMAP

    umap_embedding = umap_object.fit_transform(features_df)

    # Create a DataFrame with UMAP results
    umap_df = pd.DataFrame(umap_embedding, columns=["UMAP1", "UMAP2"])
    umap_df = pd.concat([metadata_df.reset_index(drop=True), umap_df], axis=1)
    # Save the UMAP results
    umap_df.to_parquet(data_dict[dataset]["output"], index=False)


# Individual umaps

# In[5]:


patients = pd.read_csv(
    pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True),
    header=None,
    names=["patient"],
)["patient"].to_list()


# In[6]:


file_dict = {}
for patient in patients:
    file_dict[patient] = {
        "fs": {
            "sc": {
                "input": pathlib.Path(
                    f"{root_dir}/data/{patient}/image_based_profiles/4.feature_selected_profiles/sc_fs.parquet"
                ).resolve(strict=True),
                "output": pathlib.Path(
                    f"{root_dir}/5.EDA/results/patient_results/{patient}_sc_fs_umap.parquet"
                ).resolve(),
            },
            "organoid": {
                "input": pathlib.Path(
                    f"{root_dir}/data/{patient}/image_based_profiles/4.feature_selected_profiles/organoid_fs.parquet"
                ).resolve(strict=True),
                "output": pathlib.Path(
                    f"{root_dir}/5.EDA/results/patient_results/{patient}_organoid_fs_umap.parquet"
                ).resolve(),
            },
        },
        "agg": {
            "sc_parent_organoid_level": {
                "input": pathlib.Path(
                    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/sc_agg_parent_organoid_level.parquet"
                ).resolve(strict=True),
                "output": pathlib.Path(
                    f"{root_dir}/5.EDA/results/patient_results/{patient}_sc_agg_parent_organoid_level_umap.parquet"
                ).resolve(),
            },
            "sc_well_level": {
                "input": pathlib.Path(
                    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/sc_agg_well_level.parquet"
                ).resolve(strict=True),
                "output": pathlib.Path(
                    f"{root_dir}/5.EDA/results/patient_results/{patient}_sc_agg_well_level_umap.parquet"
                ).resolve(),
            },
            "sc_consensus": {
                "input": pathlib.Path(
                    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/sc_consensus.parquet"
                ).resolve(strict=True),
                "output": pathlib.Path(
                    f"{root_dir}/5.EDA/results/patient_results/{patient}_sc_consensus_umap.parquet"
                ).resolve(),
            },
            "organoid_well_level": {
                "input": pathlib.Path(
                    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/organoid_agg_well_level.parquet"
                ).resolve(strict=True),
                "output": pathlib.Path(
                    f"{root_dir}/5.EDA/results/patient_results/{patient}_organoid_agg_well_level_umap.parquet"
                ).resolve(),
            },
            "organoid_consensus": {
                "input": pathlib.Path(
                    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/organoid_consensus.parquet"
                ).resolve(strict=True),
                "output": pathlib.Path(
                    f"{root_dir}/5.EDA/results/patient_results/{patient}_organoid_consensus_umap.parquet"
                ).resolve(),
            },
        },
    }


# In[7]:


for patient in file_dict.keys():
    for level in file_dict[patient].keys():
        for profile_type in file_dict[patient][level].keys():
            for dataset, paths in file_dict[patient][level][profile_type].items():
                print(f"Processing {patient} - {level} - {profile_type} - {dataset}")
                df = pd.read_parquet(file_dict[patient][level][profile_type]["input"])

                metadata_df = df.copy()
                metadata_subset = []
                for col in metadata_columns:
                    if col in df.columns:
                        metadata_subset.append(col)

                metadata_df = df[metadata_subset]
                features_df = df.drop(columns=metadata_columns, errors="ignore")
                print(features_df.shape)
                # remove NaN values
                features_df = features_df.dropna(axis=0, how="any")
                print(f"Data shape after dropping NaN values: {features_df.shape}")
                # Extract features and apply UMAP

                umap_embedding = umap_object.fit_transform(features_df)

                # Create a DataFrame with UMAP results
                umap_df = pd.DataFrame(umap_embedding, columns=["UMAP1", "UMAP2"])
                umap_df = pd.concat(
                    [metadata_df.reset_index(drop=True), umap_df], axis=1
                )
                # Save the UMAP results
                file_dict[patient][level][profile_type]["output"].parent.mkdir(
                    parents=True, exist_ok=True
                )
                umap_df.to_parquet(
                    file_dict[patient][level][profile_type]["output"], index=False
                )


# In[ ]:
