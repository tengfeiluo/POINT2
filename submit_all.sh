#!/bin/bash
# submit every CPU and GPU job for the R3 re-runs
cd /users/tluo/POINT2_r3; mkdir -p logs
TARGETS="Tg Tm density TC FFV Bulk_modulus_GPa O2_msa_log10 N2_msa_log10 H2_msa_log10 CO2_msa_log10 CH4_msa_log10"
FPS="Morgan RDKit MACCS TopologicalTorsion AtomPair"
if [ "$1" = "cpu" ] || [ -z "$1" ]; then
  for t in $TARGETS; do for f in $FPS; do for m in QuantileRandomForest DropoutMLP; do
    qsub -N cpu_${t}_${f}_${m} run_cpu.sh $t $f $m
  done; done; done
fi
if [ "$1" = "gpu" ] || [ -z "$1" ]; then
  for t in $TARGETS; do for m in torch-GNN torch-GREA; do
    qsub -N gpu_${t}_${m} run_gpu.sh $t $m ${2:-50}
  done; done
fi
qstat | grep -c p2r3 ; qstat | head -3
