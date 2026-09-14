#!/bin/bash

#$ -M jxu24@nd.edu
#$ -m abe
#$ -N POINT2_pi1m
#$ -q gpu@qa-xp-021
#$ -pe smp 16
#$ -l gpu_card=1

conda activate POINT2

# Path to the Python script
python_script="predict.py"

# Run the Python script
python $python_script
