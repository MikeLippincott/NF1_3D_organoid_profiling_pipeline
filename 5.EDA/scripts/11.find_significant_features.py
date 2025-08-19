#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import warnings

import pandas as pd
import statsmodels.formula.api as smf

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


# ## Filter significant features
# pvalue threshold is set to 0.05 - statistically significant features
# rsquared threshold is set to 0.5 - the explained variance is at least 50% of the total variance
# rsquared adjusted threshold is set to positive values - the model performs better than the mean
#

# ### Single Cell

# In[3]:


df = pd.read_parquet(
    profile_dict["single_cell_fs"]["input_profile_path"],
)
print(df.shape)


# In[4]:


pvalue_threshold_max = 0.05  # significance threshold for p-values
rsquared_threshold_min = 0.5  # 50% of variance explained by the model
rsquared_adj_threshold_min = 0  # the model performs better than the null model
coefficient_threshold_min = 1  # minimum effect size of 1


# In[5]:


# filter significant features
df_filtered = df[
    (df["pvalue"] < pvalue_threshold_max)
    & (df["rsquared"] > rsquared_threshold_min)
    & (df["rsquared_adj"] > rsquared_adj_threshold_min)
    & (df["coefficient"].abs() > coefficient_threshold_min)
].copy()
print(df_filtered.shape)
df_filtered.head()


# In[6]:


df_filtered["treatment"].unique()


# In[7]:


df_filtered["patient"].unique()


# In[8]:


df_filtered["feature"].unique()


# ### Organoid

# In[9]:


df = pd.read_parquet(
    profile_dict["organoid_fs"]["input_profile_path"],
)
print(df.shape)


# In[10]:


pvalue_threshold_max = 0.05
rsquared_threshold_min = 0.4
rsquared_adj_threshold_min = 0
coefficient_threshold_min = 1


# In[11]:


# filter significant features
df_filtered = df[
    (df["pvalue"] < pvalue_threshold_max)
    & (df["rsquared"] > rsquared_threshold_min)
    & (df["rsquared_adj"] > rsquared_adj_threshold_min)
    & (df["coefficient"].abs() > coefficient_threshold_min)
].copy()
print(df_filtered.shape)
df_filtered.head()


# In[12]:


df_filtered["treatment"].unique()


# In[13]:


df_filtered["patient"].unique()


# In[14]:


df_filtered["feature"].unique()
