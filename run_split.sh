#!/bin/bash
#$ -q long
#$ -pe smp 4
#$ -N p2split
#$ -o /users/tluo/POINT2_r3/logs/$JOB_NAME.$JOB_ID.out
#$ -j y
cd /users/tluo/POINT2_r3
export PATH=/users/tluo/POINT2_r3/env/bin:$PATH
export OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_CPP_MIN_LOG_LEVEL=2
EXTRA=""; [ "$4" = "cluster" ] && EXTRA="--cluster"
echo "start $(date) $1 $2 $3 $EXTRA"
python split_study.py --target_property $1 --model $2 --fpmethod $3 $EXTRA --n_jobs 4 --out_root ./results_split
echo "end $(date)"
