#!/bin/bash
#$ -q long
#$ -pe smp 4
#$ -N p2r3cpu
#$ -o /users/tluo/POINT2_r3/logs/$JOB_NAME.$JOB_ID.out
#$ -j y
# usage: qsub -N cpu_<target>_<fp>_<model> run_cpu.sh <target> <fp> <model>
TARGET=$1; FP=$2; MODEL=$3
RADIUS=2; [ "$FP" = "RDKit" ] && RADIUS=6
cd /users/tluo/POINT2_r3
export PATH=/users/tluo/POINT2_r3/env/bin:$PATH
export OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
echo "start $(date) host $(hostname) $TARGET $FP $MODEL"
python train_r3.py --target_property $TARGET --model $MODEL --fpmethod $FP --radius $RADIUS --seed 0 --search --n_jobs 4 --out_root ./results_r3 || exit 1
for s in 1 2 3 4; do
  python train_r3.py --target_property $TARGET --model $MODEL --fpmethod $FP --radius $RADIUS --seed $s --n_jobs 4 --out_root ./results_r3 || exit 1
done
echo "end $(date)"
