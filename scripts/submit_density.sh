#!/bin/bash

#$ -M jxu24@nd.edu
#$ -m abe
#$ -N POINT2_density
#$ -q gpu@qa-xp-021 ## gpu@qa-xp-020 if the 021 node is full
#$ -pe smp 4 # maximum 16, suggest 4 cpu for each GPU card
#$ -l gpu_card=1

 
#setenv CUDA_VISIBLE_DEVICES 0   ##pick 0,1,2,3 for the GPU card #setenv CUDA_VISIBLE_DEVICES "0,1" for using 2 GPU card
conda activate POINT2

# Configuration arrays
fpmethods=("Morgan" "RDKit" "MACCS" "TopologicalTorsion" "AtomPair")
radii=("2")  # Default radius for methods that use it
n_bits="2048"
target_properties=("density")
models=("QuantileRandomForest" "DropoutMLP" "torch-GREA" "torch-GNN")
special_models=("torch-GREA" "torch-GNN") # Models that don't use fpmethod, radius, n_bits

# Run all combinations
for target_property in "${target_properties[@]}"; do
  for model in "${models[@]}"; do
    if [[ " ${special_models[@]} " =~ " ${model} " ]]; then
      # Special model logic
      echo "Running special model: --target_property $target_property --model $model"
      python train.py --target_property "$target_property" --model "$model"
    else
      # Standard model logic
      for fpmethod in "${fpmethods[@]}"; do
        # Handle conditional radius setting for RDKit
        local_radii=("2")
        if [ "$fpmethod" == "RDKit" ]; then
          local_radii+=("6")
        fi
        
        for radius in "${local_radii[@]}"; do
          echo "Running: --fpmethod $fpmethod --radius $radius --n_bits $n_bits --target_property $target_property --model $model"
          python train.py --fpmethod "$fpmethod" --radius "$radius" --n_bits "$n_bits" --target_property "$target_property" --model "$model"
        done
      done
    fi
  done
done


#Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz
