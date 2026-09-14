#!/bin/bash

#$ -M tluo@nd.edu
#$ -m abe
#$ -N POINT2_GREA_500
#$ -q gpu@qa-xp-021 ## Use gpu@qa-xp-020 if the 021 node is full
#$ -pe smp 4        # Maximum 16, suggest 4 CPUs for each GPU card
#$ -l gpu_card=1

# Activate the conda environment
conda activate POINT2

# Configuration arrays
target_properties=("Tg" "Tm" "TC" "FFV") #"O2_msa_log10" "N2_msa_log10" "H2_msa_log10" "CO2_msa_log10" "CH4_msa_log10" "Bulk_modulus_GPa"
models=("torch-GREA")  # Only the desired models

# Run all combinations
for target_property in "${target_properties[@]}"; do
  for model in "${models[@]}"; do
    echo "Running model: --target_property $target_property --model $model"
    python train.py --target_property "$target_property" --model "$model" --n_search 500
  done
done
