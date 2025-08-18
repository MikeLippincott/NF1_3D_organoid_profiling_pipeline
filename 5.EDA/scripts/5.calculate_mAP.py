#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
from typing import Union

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


# ## Paths to compute

# In[2]:


paths_to_process_dict = {
    "organoid_fs": {
        "input": pathlib.Path(
            f"{root_dir}/data/all_patient_profiles/organoid_fs_profiles.parquet"
        ).resolve(strict=True),
        "inter_patient_results_path": pathlib.Path(
            f"{root_dir}/5.EDA/results/mAP/organoid_fs_inter_patient_mAP.parquet"
        ).resolve(),
        "intra_patient_results_path": pathlib.Path(
            f"{root_dir}/5.EDA/results/mAP/organoid_fs_intra_patient_mAP.parquet"
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
        "input": pathlib.Path(
            f"{root_dir}/data/all_patient_profiles/sc_fs_profiles.parquet"
        ).resolve(strict=True),
        "inter_patient_results_path": pathlib.Path(
            f"{root_dir}/5.EDA/results/mAP/sc_fs_inter_patient_mAP.parquet"
        ).resolve(),
        "intra_patient_results_path": pathlib.Path(
            f"{root_dir}/5.EDA/results/mAP/sc_fs_intra_patient_mAP.parquet"
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
paths_to_process_dict["single_cell_fs"]["intra_patient_results_path"].parent.mkdir(
    parents=True, exist_ok=True
)


# In[3]:


def calculate_mAP(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "treatment",
    reference_group: str = "DMSO",
) -> Union[None, pd.DataFrame]:
    """
    Description
    -------
    Calculate intra-patient mean Average Precision (mAP) for a given DataFrame.

    Parameters
    -------
    df : pd.DataFrame
        DataFrame containing the profiles and metadata.
    metadata_columns : list
        List of metadata columns to be used for grouping.
    col_for_reference : str
        Column name to be used for reference profiles.
    reference_group : str
        The value in `col_for_reference` that indicates the reference profiles (default is 'DMSO').
    Returns
    -------
    Union[None, pd.DataFrame]
        If output_path is provided, returns None. Otherwise, returns a DataFrame with the mAP results.
    """
    reference_col = "reference_index"

    df = assign_reference_index(
        df,
        f"{col_for_reference} == '{reference_group}'",  # condition to get reference profiles (neg controls)
        reference_col=reference_col,
        default_value=-1,
    )
    df.dropna(inplace=True)
    metadata = df[metadata_columns]
    profiles = df.drop(columns=metadata_columns).values
    # positive pairs are replicates of the same treatment
    pos_sameby = [col_for_reference, reference_col]
    pos_diffby = []
    neg_sameby = []
    # negative pairs are replicates of different treatments
    neg_diffby = [col_for_reference, reference_col]
    try:
        df_ap = map.average_precision(
            metadata, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby
        )
        df_ap = df_ap.query(
            f"{col_for_reference} != '{reference_group}'"
        )  # remove DMSO
        activity_map = map.mean_average_precision(
            df_ap, pos_sameby, null_size=1000000, threshold=0.05, seed=0
        )
        activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(
            np.log10
        )
        return activity_map
    except Exception as e:
        print(f"Error calculating mAP: {e}")
        return None


def calculate_intra_patient_mAP(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "treatment",
    reference_group: str = "DMSO",
    output_path: Union[None, pathlib.Path] = None,
):
    list_of_dfs = []
    for patient in df["patient"].unique():
        patient_df = df.loc[df["patient"] == patient, :].copy()
        for drug in patient_df["treatment"].unique():
            # subset the DataFrame for the current drug and DMSO
            drug_df = patient_df.loc[patient_df["treatment"] == drug, :].copy()
            dmso_df = patient_df.loc[patient_df["treatment"] == "DMSO", :].copy()
            drug_df = pd.concat([drug_df, dmso_df], ignore_index=True)
            # calculate mAP for the current drug
            mAP = calculate_mAP(
                drug_df,
                metadata_columns=metadata_columns,
                col_for_reference=col_for_reference,
                reference_group=reference_group,
            )
            if mAP is not None:
                mAP["patient"] = patient
                mAP["treatment"] = drug
                list_of_dfs.append(mAP)

    output_df = pd.concat(list_of_dfs, ignore_index=True)
    if output_path is not None:
        output_df.to_parquet(output_path, index=False)
    return output_df


def calculate_inter_patient_mAP(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "treatment",
    reference_group: str = "DMSO",
    output_path: Union[None, pathlib.Path] = None,
):
    """
    Calculate inter-patient mAP for the given DataFrame.

    Parameters
    -------
    df : pd.DataFrame
        DataFrame containing the profiles and metadata.
    """
    list_of_dfs = []
    for drug in df["treatment"].unique():
        drug_df = df.loc[df["treatment"] == drug, :].copy()
        dmso_df = df.loc[df["treatment"] == "DMSO", :].copy()
        drug_df = pd.concat([drug_df, dmso_df], ignore_index=True)
        # calculate mAP for the current drug
        mAP = calculate_mAP(
            drug_df,
            metadata_columns=metadata_columns,
            col_for_reference=col_for_reference,
            reference_group=reference_group,
        )
        if mAP is not None:
            mAP["treatment"] = drug
            mAP["patient"] = "all_patients"
            list_of_dfs.append(mAP)
    output_df = pd.concat(list_of_dfs, ignore_index=True)
    if output_path is not None:
        output_df.to_parquet(output_path, index=False)
    return output_df


# In[4]:


for key, paths in paths_to_process_dict.items():
    print(f"Processing {key}")
    calculate_intra_patient_mAP(
        df=pd.read_parquet(paths["input"]),
        metadata_columns=paths["metadata_columns"],
        col_for_reference="treatment",
        reference_group="DMSO",
        output_path=paths["intra_patient_results_path"],
    )
    calculate_inter_patient_mAP(
        df=pd.read_parquet(paths["input"]),
        metadata_columns=paths["metadata_columns"],
        col_for_reference="treatment",
        reference_group="DMSO",
        output_path=paths["inter_patient_results_path"],
    )
