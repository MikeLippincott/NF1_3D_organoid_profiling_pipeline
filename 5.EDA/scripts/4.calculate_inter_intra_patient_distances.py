#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
from typing import Union

import numpy as np
import pandas as pd
import scipy
import scipy.spatial.distance
import sklearn.metrics.pairwise

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


# ## Define the functions

# In[2]:


def calculate_inter_patient_metrics(
    df: pd.DataFrame, metadata_columns: list, results_path: str = None
) -> Union[pd.DataFrame, None]:
    """
    Description
    -----------
    Calculate inter-patient distance metrics for each treatment in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing patient data with a 'treatment' column and other features.
    metadata_columns : list
        List of metadata columns to be excluded from distance calculations.
    results_path : str, optional
        Path to save the results as a parquet file. If None, the DataFrame is returned.
    Returns
    -------
    pd.DataFrame or None
        DataFrame containing inter-patient distance metrics for each treatment, or None if results_path is
        provided and the DataFrame is saved to that path.
    """
    inter_patient_distance_metric_dict = {
        "treatment": [],
        "euclidean_distance_mean": [],
        "euclidean_distance_std": [],
        "cosine_distance_mean": [],
        "cosine_distance_std": [],
        "correlation_distance_mean": [],
        "correlation_distance_std": [],
    }
    treatments = df["treatment"].unique()
    for drug in treatments:
        if drug == "DMSO":
            continue
        dmso_df = df.loc[df["treatment"] == "DMSO"]
        drug_df = df.loc[df["treatment"] == drug]
        dmso_df = dmso_df.drop(columns=metadata_columns)
        drug_df = drug_df.drop(columns=metadata_columns)
        # drop NaN values
        dmso_df = dmso_df.dropna(axis=0, how="any")
        drug_df = drug_df.dropna(axis=0, how="any")
        euclidean_distance = sklearn.metrics.pairwise.euclidean_distances(
            dmso_df.values, drug_df.values
        ).reshape(-1)

        cosine_distance = sklearn.metrics.pairwise.cosine_distances(
            dmso_df.values, drug_df.values
        ).reshape(-1)

        correlation_distance = sklearn.metrics.pairwise.pairwise_distances(
            dmso_df.values, drug_df.values, metric="correlation"
        ).reshape(-1)

        inter_patient_distance_metric_dict["treatment"].append(drug)
        inter_patient_distance_metric_dict["euclidean_distance_mean"].append(
            euclidean_distance.mean()
        )
        inter_patient_distance_metric_dict["euclidean_distance_std"].append(
            euclidean_distance.std()
        )
        inter_patient_distance_metric_dict["cosine_distance_mean"].append(
            cosine_distance.mean()
        )
        inter_patient_distance_metric_dict["cosine_distance_std"].append(
            cosine_distance.std()
        )
        inter_patient_distance_metric_dict["correlation_distance_mean"].append(
            correlation_distance.mean()
        )
        inter_patient_distance_metric_dict["correlation_distance_std"].append(
            correlation_distance.std()
        )

    inter_patient_metric_df = pd.DataFrame(inter_patient_distance_metric_dict)
    if results_path is not None:
        inter_patient_metric_df.to_parquet(results_path, index=False)
    else:
        return inter_patient_metric_df


def calculate_intra_patient_metrics(
    df: pd.DataFrame, metadata_columns: list, results_path: str = None
) -> Union[pd.DataFrame, None]:
    """
    Description
    -----------
    Calculate intra-patient distance metrics for each treatment in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing patient data with a 'treatment' column and other features.
    metadata_columns : list
        List of metadata columns to be excluded from distance calculations.
    results_path : str, optional
        Path to save the results as a parquet file. If None, the DataFrame is returned.
    Returns
    -------
    pd.DataFrame or None
        DataFrame containing inter-patient distance metrics for each treatment, or None if results_path is
        provided and the DataFrame is saved to that path.
    """
    intra_patient_distance_metric_dict = {
        "patient": [],
        "treatment": [],
        "euclidean_distance_mean": [],
        "euclidean_distance_std": [],
        "cosine_distance_mean": [],
        "cosine_distance_std": [],
        "correlation_distance_mean": [],
        "correlation_distance_std": [],
    }
    patients = df["patient"].unique()
    for patient in patients:
        patient_df = df[df["patient"] == patient]
        # get the dmso data
        for drug in patient_df["treatment"].unique():
            if drug == "DMSO":
                continue

            # get both dmso and drug data
            test_df = patient_df.loc[patient_df["treatment"] == drug]
            dmso_df = patient_df.loc[patient_df["treatment"] == "DMSO"]
            test_df = test_df.drop(columns=metadata_columns)
            dmso_df = dmso_df.drop(columns=metadata_columns)
            # drop NaN values
            test_df = test_df.dropna(axis=0, how="any")
            dmso_df = dmso_df.dropna(axis=0, how="any")
            # check the test_df shape
            if test_df.shape[0] == 0 or dmso_df.shape[0] == 0:
                print(
                    f"Skipping patient {patient} and drug {drug} due to empty DataFrame."
                )
                continue

            # calculate the distance between the drug and the DMSO
            euclidean_distance = sklearn.metrics.pairwise.euclidean_distances(
                dmso_df.values, test_df.values
            ).reshape(-1)
            cosine_distance = sklearn.metrics.pairwise.cosine_distances(
                dmso_df.values, test_df.values
            ).reshape(-1)
            correlation_distance = sklearn.metrics.pairwise.pairwise_distances(
                dmso_df.values, test_df.values, metric="correlation"
            ).reshape(-1)
            intra_patient_distance_metric_dict["patient"].append(patient)
            intra_patient_distance_metric_dict["treatment"].append(drug)
            intra_patient_distance_metric_dict["euclidean_distance_mean"].append(
                euclidean_distance.mean()
            )
            intra_patient_distance_metric_dict["euclidean_distance_std"].append(
                euclidean_distance.std()
            )
            intra_patient_distance_metric_dict["cosine_distance_mean"].append(
                cosine_distance.mean()
            )
            intra_patient_distance_metric_dict["cosine_distance_std"].append(
                cosine_distance.std()
            )
            intra_patient_distance_metric_dict["correlation_distance_mean"].append(
                correlation_distance.mean()
            )
            intra_patient_distance_metric_dict["correlation_distance_std"].append(
                correlation_distance.std()
            )
    intra_patient_distance_metric_df = pd.DataFrame(intra_patient_distance_metric_dict)
    if results_path is not None:
        intra_patient_distance_metric_df.to_parquet(results_path, index=False)
    else:
        return intra_patient_distance_metric_df


# ## Set paths

# In[3]:


paths_to_process_dict = {
    "organoid_fs": {
        "input": pathlib.Path(
            f"{root_dir}/data/all_patient_profiles/organoid_fs_profiles.parquet"
        ).resolve(strict=True),
        "inter_patient_results_path": pathlib.Path(
            f"{root_dir}/5.EDA/results/distance_metrics/organoid_fs_inter_patient_distance_metrics.parquet"
        ).resolve(),
        "intra_patient_results_path": pathlib.Path(
            f"{root_dir}/5.EDA/results/distance_metrics/organoid_fs_intra_patient_distance_metrics.parquet"
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
        ],
    },
    "single_cell_fs": {
        "input": pathlib.Path(
            f"{root_dir}/data/all_patient_profiles/sc_fs_profiles.parquet"
        ).resolve(strict=True),
        "inter_patient_results_path": pathlib.Path(
            f"{root_dir}/5.EDA/results/distance_metrics/sc_fs_inter_patient_distance_metrics.parquet"
        ).resolve(),
        "intra_patient_results_path": pathlib.Path(
            f"{root_dir}/5.EDA/results/distance_metrics/sc_fs_intra_patient_distance_metrics.parquet"
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
        ],
    },
}


# ## Run the functions

# In[4]:


for key, paths in paths_to_process_dict.items():
    # make the parent directories for the results paths if they do not exist
    paths["intra_patient_results_path"].parent.mkdir(parents=True, exist_ok=True)
    calculate_intra_patient_metrics(
        df=pd.read_parquet(paths["input"]),
        metadata_columns=paths.get("metadata_columns", []),
        results_path=paths["intra_patient_results_path"],
    )
    calculate_inter_patient_metrics(
        df=pd.read_parquet(paths["input"]),
        metadata_columns=paths.get("metadata_columns", []),
        results_path=paths["inter_patient_results_path"],
    )
