#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import warnings

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")  # Ignore all warnings
warnings.simplefilter("ignore")  # Additional suppression method

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


profile_dict = {
    "organoid_fs": {
        "input_profile_path": pathlib.Path(
            root_dir, "data/all_patient_profiles/organoid_fs_profiles.parquet"
        ),
        "output_profile_path": pathlib.Path(
            root_dir, "5.EDA/results/linear_modeling/organoid_fs.parquet"
        ),
        "metadata_columns": [
            "patient",
            "object_id",
            "unit",
            "dose",
            "treatment",
            "Target",
            "Class",
            "image_set",
            "Well",
            "Therapeutic Categories",
            "single_cell_count",
        ],
    },
    "single_cell_fs": {
        "input_profile_path": pathlib.Path(
            root_dir, "data/all_patient_profiles/sc_fs_profiles.parquet"
        ),
        "output_profile_path": pathlib.Path(
            root_dir, "5.EDA/results/linear_modeling/sc_fs.parquet"
        ),
        "metadata_columns": [
            "patient",
            "object_id",
            "unit",
            "dose",
            "treatment",
            "Target",
            "Class",
            "image_set",
            "Well",
            "Therapeutic Categories",
            "parent_organoid",
        ],
    },
}


# ## Linear modeling
#
# We want to predict each feature given some information about the organoid per patient. We will use linear regression to do this.
# $y = X_0 * \beta_0 + X_1 * \beta_1 + ... + X_n * \beta_n + \epsilon$
#
# Where:
# $y$ = feature to predict
# $\beta_0$ = Intercept
# $X_1$ = The treatment (e.g. control, drug)
# $\beta_1$ = The coefficient for the treatment
# $\epsilon$ = The error term
#
# For each model(feature), we get the following statistics:
# - **R-squared**: Proportion of variance explained by the model.
# - **p-value**: Significance of the model.
# - **F-statistic**: Overall significance of the model.
# - **Coefficients**: Effect size of each predictor.

# In[3]:


for profile in tqdm(profile_dict.keys(), desc="Loading profiles"):
    # set the output dictionary for linear modeling results
    # per profile
    linear_modeling_results_dict = {
        "patient": [],
        "treatment": [],
        "feature": [],
        "rsquared": [],
        "rsquared_adj": [],
        "fvalue": [],
        "pvalue": [],
        "coefficient": [],
        "intercept": [],
    }
    metadata_columns = profile_dict[profile]["metadata_columns"]
    df = pd.read_parquet(profile_dict[profile]["input_profile_path"])
    # rename feature columns as the "." dod not play nice with the formula
    for col in df.columns:
        new_col = col.replace(
            ".",
            "",  # we replace the "." with an empty string because it causes issues in the formula
            # the linear model interprets the "." as an operator and not as part of the column name
        )  # Replace . with empty string for compatibility in formula
        df.rename(columns={col: new_col}, inplace=True)

    for patient in tqdm(
        df["patient"].unique(), desc="Processing patients", unit="patient", leave=False
    ):
        df_patient = df.loc[df["patient"] == patient]

        # Filter for specific treatments
        df_patient_trt = df_patient.loc[df_patient["treatment"].isin(["DMSO"])]
        combo_list = [
            ("DMSO", i) for i in df_patient["treatment"].unique() if i != "DMSO"
        ]
        for combo in tqdm(
            combo_list,
            desc="Processing treatment combinations",
            unit="combo",
            leave=False,
        ):
            df_patient_trt = df_patient.loc[df_patient["treatment"].isin(combo)]
            # order the treatment column to ensure DMSO is first
            df_patient_trt["treatment"] = pd.Categorical(
                df_patient_trt["treatment"],
                categories=["DMSO"]
                + [
                    other_treatment
                    for other_treatment in df_patient["treatment"].unique()
                    if other_treatment != "DMSO"
                ],
            )
            for col in df_patient_trt.columns:
                if col not in metadata_columns:
                    # Prepare the formula for the linear model
                    formula = f"{col} ~ C(treatment)"
                    # Import statsmodels and run the linear model
                    model = smf.ols(formula=formula, data=df_patient_trt)
                    results = model.fit()

                    linear_modeling_results_dict["patient"].append(patient)
                    linear_modeling_results_dict["treatment"].append(combo[1])
                    linear_modeling_results_dict["feature"].append(col)
                    linear_modeling_results_dict["rsquared"].append(results.rsquared)
                    linear_modeling_results_dict["rsquared_adj"].append(
                        results.rsquared_adj
                    )
                    linear_modeling_results_dict["fvalue"].append(results.fvalue)
                    linear_modeling_results_dict["pvalue"].append(
                        results.pvalues[f"C(treatment)[T.{combo[1]}]"]
                    )
                    linear_modeling_results_dict["coefficient"].append(
                        results.params[f"C(treatment)[T.{combo[1]}]"].item()
                    )
                    linear_modeling_results_dict["intercept"].append(
                        results.params["Intercept"].item()
                    )
    linear_modeling_results_df = pd.DataFrame(linear_modeling_results_dict)
    # split the feature column into multiple columns
    linear_modeling_results_df["feature"].str.split("_", expand=True)
    linear_modeling_results_df[
        ["Feature_type", "Compartment", "Channel", "Measurement", "Extra_info"]
    ] = linear_modeling_results_df["feature"].str.split("_", expand=True)

    # if feature type is area shape then make the measurement the channel and
    # set the channel to None
    # this because area size shape features are not channel specific
    linear_modeling_results_df.loc[
        linear_modeling_results_df["Feature_type"] == "AreaSizeShape", "Measurement"
    ] = linear_modeling_results_df["Channel"]
    linear_modeling_results_df.loc[
        linear_modeling_results_df["Feature_type"] == "AreaSizeShape", "Channel"
    ] = None
    # set compartment to None if is adjacent
    # this is because adjacent features are not compartment specific
    linear_modeling_results_df.loc[
        linear_modeling_results_df["Compartment"] == "adjacent", "Compartment"
    ] = None

    # run FDR on the p-values
    pvals = linear_modeling_results_df["pvalue"].values
    _, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")
    linear_modeling_results_df["pvalue_fdr"] = pvals_fdr
    # Save the updated DataFrame with FDR p-values
    linear_modeling_results_df.to_parquet(
        profile_dict[profile]["output_profile_path"], index=False
    )
    profile_dict[profile]["output_profile_path"].parent.mkdir(
        parents=True, exist_ok=True
    )
    linear_modeling_results_df.to_parquet(
        profile_dict[profile]["output_profile_path"], index=False
    )
