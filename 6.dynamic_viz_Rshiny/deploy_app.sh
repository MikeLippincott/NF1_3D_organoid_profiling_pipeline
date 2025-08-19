#!/bin/bash

test_run="TRUE"

# establish the git root directory
git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    return 1
fi

# check if r_shiny_env is available
if ! conda env list | grep -q "r_shiny_env"; then
    mamba env create -f "$git_root/environments/R_shiny.yml"
    # shellcheck disable=SC2181
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create the r_shiny_env environment."
        return 1
    fi
fi

conda activate r_shiny_env

if [ "$test_run" = "TRUE" ]; then
    # check for faults - use direct exit code checking
    if ! Rscript "$git_root/6.dynamic_viz_Rshiny/deploy_app.R" --testing; then
        echo "Deployment failed. Please check the logs for errors."
        return 1
    fi
else
    # deploy the app
    if ! Rscript "$git_root/6.dynamic_viz_Rshiny/deploy_app.R"; then
        echo "Deployment failed. Please check the logs for errors."
        return 1
    fi
fi

conda deactivate


echo "Deployment completed successfully."
