#!/bin/bash

# Array of datasets
datasets=("dv_neut" "dv_ioni" "ioni_gas-temp" "ioni_gas" "neut_gas" "dark_mat")

# Loop through each dataset and create a new screen session
for dataset in "${datasets[@]}"; do
    screen -dmS $dataset bash -c "python nearest-voxel-id-search-cKDTree.py $dataset"
done
