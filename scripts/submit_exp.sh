#!/bin/bash

#$ -M jxu24@nd.edu
#$ -m abe
#$ -N POINT2_explain
#$ -q gpu@qa-xp-020
#$ -pe smp 4
#$ -l gpu_card=1

conda activate POINT2

# Arrays of parameters
target_properties=("N2_msa_log10" "H2_msa_log10" "CO2_msa_log10" "CH4_msa_log10") #"Tg" "Tm" "TC" "FFV" "O2_msa_log10"
models=("QuantileRandomForest" "DropoutMLP")
fpmethods=("Morgan" "RDKit" "MACCS" "TopologicalTorsion" "AtomPair")
n_bits="2048"

# Loop through each combination of target property, model, and fingerprint method
for target_property in "${target_properties[@]}"; do
  for model in "${models[@]}"; do
    for fpmethod in "${fpmethods[@]}"; do
      # Default radius
      radius="2"
      
      # Adjust radius for RDKit fingerprint method
      if [ "$fpmethod" == "RDKit" ]; then
        radius="6"
      fi

      echo "Running: --target_property $target_property --model $model --fpmethod $fpmethod --radius $radius --n_bits $n_bits"
      python explain.py --target_property "$target_property" --model "$model" --fpmethod "$fpmethod" --radius "$radius" --n_bits "$n_bits"
    done
  done
done

