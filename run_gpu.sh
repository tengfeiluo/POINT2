#!/bin/bash
#$ -q gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -N p2r3gpu
#$ -o /users/tluo/POINT2_r3/logs/$JOB_NAME.$JOB_ID.out
#$ -j y
# usage: qsub -N gpu_<target>_<model> run_gpu.sh <target> <torch-GNN|torch-GREA> [n_search]
TARGET=$1; MODEL=$2; NSEARCH=${3:-50}
cd /users/tluo/POINT2_r3
module load cuda/11.8
export PATH=/users/tluo/POINT2_r3/env/bin:$PATH
echo "start $(date) host $(hostname) $TARGET $MODEL n_search=$NSEARCH"; nvidia-smi -L
python train_r3.py --target_property $TARGET --model $MODEL --seed 0 --n_search $NSEARCH --out_root ./results_r3 || exit 1
for s in 1 2 3 4; do
  python train_r3.py --target_property $TARGET --model $MODEL --seed $s --out_root ./results_r3 || exit 1
done
echo "end $(date)"
