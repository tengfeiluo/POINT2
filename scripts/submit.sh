#!/bin/bash

#$ -M jxu24@nd.edu
#$ -m abe
#$ -N POINT2_GNN
#$ -q gpu@qa-xp-021 ## gpu@qa-xp-020 if the 021 node is full
#$ -pe smp 8 # maximum 16, suggest 4 cpu for each GPU card
#$ -l gpu_card=1

 
#setenv CUDA_VISIBLE_DEVICES 0   ##pick 0,1,2,3 for the GPU card #setenv CUDA_VISIBLE_DEVICES "0,1" for using 2 GPU card
conda activate POINT2

#python train.py --fpmethod Morgan --radius 2 --n_bits 2048 --target_property Tm --model torch-GNN
echo "Running special model: --target_property Tm --model torch-GREA"
python train.py --fpmethod Morgan --radius 2 --n_bits 2048 --target_property Tm --model torch-GREA
echo "Running special model: --target_property Tg --model torch-GNN"
python train.py --fpmethod Morgan --radius 2 --n_bits 2048 --target_property Tg --model torch-GNN
echo "Running special model: --target_property Tg --model torch-GREA"
python train.py --fpmethod Morgan --radius 2 --n_bits 2048 --target_property Tg --model torch-GREA
echo "Running special model: --target_property TC --model torch-GNN"
python train.py --fpmethod Morgan --radius 2 --n_bits 2048 --target_property TC --model torch-GNN
echo "Running special model: --target_property TC --model torch-GREA"
python train.py --fpmethod Morgan --radius 2 --n_bits 2048 --target_property TC --model torch-GREA



#Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz
