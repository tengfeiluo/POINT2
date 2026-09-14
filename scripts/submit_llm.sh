#!/bin/bash

#$ -M jxu24@nd.edu
#$ -m abe
#$ -N POINT2_gpt
#$ -q gpu@qa-xp-020 ## Use gpu@qa-xp-020 if the 021 node is full
#$ -pe smp 4        # Maximum 16, suggest 4 CPUs for each GPU card
#$ -l gpu_card=1

# Activate the conda environment
conda activate POINT2
# conda activate llm-huggingface

# Configuration arrays
target_properties=("Tm") # "Tm" "Tg" "TC" "FFV" "density" "O2_msa_log10" "N2_msa_log10" "H2_msa_log10" "CO2_msa_log10" "CH4_msa_log10"
models=("llm-gpt-fewshot-20")  # Only the desired models "llm-gpt-zeroshot" "llm-gpt-fewshot-5" "llm-gpt-fewshot-10" 

# Run all combinations
for target_property in "${target_properties[@]}"; do
  for model in "${models[@]}"; do
    echo "Running model: --target_property $target_property --model $model"
    python train.py --target_property "$target_property" --model "$model" --n_search 500
  done
done
