#!/bin/bash
#$ -q gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -o /users/tluo/POINT2_r3/logs/$JOB_NAME.$JOB_ID.out
#$ -j y
TARGET=$1; MODEL=$2
cd /users/tluo/POINT2_r3
module load cuda/11.8
export PATH=/users/tluo/POINT2_r3/env/bin:$PATH
echo "start $(date) $TARGET $MODEL n_search=200 (control for search budget)"
python train_r3.py --target_property $TARGET --model $MODEL --seed 0 --n_search 200 --out_root ./results_r3_search200 || exit 1
for s in 1 2 3 4; do python train_r3.py --target_property $TARGET --model $MODEL --seed $s --out_root ./results_r3_search200 || exit 1; done
echo "end $(date)"
