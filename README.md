# POINT²: POlymer INformatics Training and Testing

Code, data splits, and reproduction scripts for
*POINT²: A Polymer Informatics Training and Testing Database* (Xu, Liu, Guo, Jiang, Luo; *Digital Discovery*, 2026).

POINT² is a benchmark protocol for polymer property prediction with uncertainty quantification,
interpretability and template-based synthesizability: ten property tasks (T_g, T_m, density, thermal
conductivity, fractional free volume, and O2/N2/H2/CO2/CH4 permeability), fixed train/test splits,
and a model suite of quantile random forests (QRF), MC-dropout MLPs (MLP-D), graph neural networks
(GIN/GCN) and GREA, plus GPT-4o-mini in-context-learning baselines.

## Contents

| Path | What |
|---|---|
| `data/benchmark/` | One CSV per task with the fixed 80/20 split (see `data/README.md` for what each file contains and the PoLyInfo redistribution policy) |
| `train.py`, `model.py`, `utils.py`, `predict.py`, `explain.py`, `llm_prediction.py` | The original training / screening / SHAP code used for the results in the paper (as archived from the group cluster) |
| `train_r3.py` | The revised training protocol: internal 10 % validation subset drawn from the training split, seeded runs, grid search for QRF/MLP-D, early stopping, and ECE; see below |
| `aggregate_r3.py`, `run_cpu.sh`, `run_gpu.sh`, `submit_all.sh` | Aggregation into mean ± std tables and the SGE job scripts used for the multi-seed re-runs |
| `results_r3/` | Per-task, per-model mean ± std metrics from the multi-seed runs (CSV) |
| `retrosynthesis/` | Template library (82 SMARTS templates), PolyScore, the curated polymerization-reaction set, and evaluation outputs |
| `scripts/` | The original SGE submission scripts |
| `torch-molecule/` | Git submodule pinned to v0.1.1 (commit `0185b95`), the version used for the GNN/GREA results |

Trained models, fingerprint caches and the PI1M screening predictions (≈1.5 GB) are archived on Zenodo
(DOI to be added on acceptance) rather than in this repository.

## Installation

```bash
git clone --recurse-submodules https://github.com/tengfeiluo/POINT2.git && cd POINT2
conda env create -f environment.yml && conda activate point2       # CPU models
pip install -r requirements-gpu.txt && pip install -e ./torch-molecule   # GNN / GREA (CUDA 11.8)
```

## Reproducing a table entry

All commands read `data/benchmark/<task>.csv`. For the PoLyInfo tasks (T_g, T_m, density) first add the
`SMILES` and label columns from your own PoLyInfo export by joining on `PID` (the split column is provided).

```bash
# QRF + Morgan on T_g: grid search on the validation subset with seed 0, then seeds 1–4 at the chosen setting
python train_r3.py --target_property Tg --model QuantileRandomForest --fpmethod Morgan --seed 0 --search
for s in 1 2 3 4; do python train_r3.py --target_property Tg --model QuantileRandomForest --fpmethod Morgan --seed $s; done

# GREA on T_g (GPU): Optuna search (100 trials) on the validation subset, then seeds 1–4
python train_r3.py --target_property Tg --model torch-GREA --seed 0 --n_search 100
for s in 1 2 3 4; do python train_r3.py --target_property Tg --model torch-GREA --seed $s; done

python aggregate_r3.py ./results_r3       # -> summary_mean_std.csv, table_rmse.csv, table_mae.csv, table_r2.csv, table_spearman.csv, table_ece.csv
```

Each run writes `results_r3/<task>/<model>_<fp>/seed<k>/metrics.json` (RMSE, MAE, R², Spearman rank
correlation between |error| and predicted σ, 90 % interval coverage, and ECE) together with the
predictions, σ, interval bounds and the SMILES of the train/validation/test rows.

**ECE.** For confidence levels α ∈ {0.1, 0.2, …, 0.9} the central prediction interval ŷ ± z_(1+α)/2 · σ is
formed from the model's predictive σ (QRF: half the 5–95 % quantile width divided by 1.645; MLP-D: standard
deviation over 100 MC-dropout passes; GREA: square root of the rationale-environment variance) and
ECE = mean_α |coverage(α) − α|.

The original single-run protocol (`train.py`, Optuna with the training set as validation set, no seed loop)
is kept for provenance; `scripts/` holds the original job scripts.

## Citation

Xu, J.; Liu, G.; Guo, R.; Jiang, M.; Luo, T. POINT²: A Polymer Informatics Training and Testing Database. *Digital Discovery* 2026, DOI to be added.

## License

MIT (see `LICENSE`). PoLyInfo-derived values are not redistributed; see `data/README.md`.
