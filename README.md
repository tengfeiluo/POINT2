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
| `train_r3.py` | The benchmark training protocol: internal 10 % validation subset drawn from the training split, seeded runs, grid search for QRF/MLP-D, early stopping, Optuna search for GNN/GREA, and ECE; see below |
| `aggregate_r3_guard.py`, `make_tables.py` | Aggregation into the mean ± std tables of the paper (predictions clipped to the training-label range; see below) and the LaTeX table bodies |
| `run_cpu.sh`, `run_gpu200.sh`, `submit_all.sh`, `split_study.py`, `run_split.sh` | SGE job scripts for the multi-seed runs and the split-sensitivity study (Appendix A.1) |
| `aggregate_r3.py` | Earlier aggregation script (convergence filter), kept for reference; superseded by `aggregate_r3_guard.py` |
| `results_r3/` | Per-run and mean ± std metrics behind Tables 2, 3 and A.9–A.11, and the split-sensitivity results (`split/`, Table A.12); see `results_r3/README.md` |
| `retrosynthesis/` | Template library (82 SMARTS templates), PolyScore, the curated polymerization-reaction set, and per-entry evaluation outputs (Table 4) |
| `figures/` | Scripts that draw the two case-study figures (Fig. 5 and Fig. A.6) from SMILES |
| `scripts/` | The original SGE submission scripts |
| `torch-molecule/` | Git submodule pinned to v0.1.1 (commit `0185b95`), the version used for the GNN/GREA results |

The QRF and MLP-D models used for the PI1M screening and the case studies, and their predictions for the 199,160-polymer screening subset (~2.1 GB), are archived in a separate Zenodo record (DOI: [10.5281/zenodo.23252833](https://doi.org/10.5281/zenodo.23252833)). These single-seed models were trained by `train.py` with fixed hyperparameters; the benchmark tables are produced by `train_r3.py`.

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

# GREA on T_g (GPU): Optuna search (200 trials) on the validation subset, then seeds 1–4 at the chosen setting
python train_r3.py --target_property Tg --model torch-GREA --seed 0 --n_search 200 --out_root ./results_r3_search200
for s in 1 2 3 4; do python train_r3.py --target_property Tg --model torch-GREA --seed $s --out_root ./results_r3_search200; done

# fingerprint runs from ./results_r3, graph runs from ./results_r3_search200 -> ./results_final/summary_mean_std.csv
python aggregate_r3_guard.py ./results_r3 ./results_r3_search200 ./results_final
```

Each run writes `results_r3/<task>/<model>_<fp>/seed<k>/metrics.json` (RMSE, MAE, R², Spearman rank
correlation between |error| and predicted σ, 90 % interval coverage, and ECE) together with the
predictions, σ, interval bounds and the SMILES of the train/validation/test rows.

**ECE.** For confidence levels α ∈ {0.1, 0.2, …, 0.9} the central prediction interval ŷ ± z_(1+α)/2 · σ is
formed from the model's predictive σ (QRF: half the 5–95 % quantile width divided by 1.645; MLP-D: standard
deviation over 100 MC-dropout passes; GREA: square root of the rationale-environment variance) and
ECE = mean_α |coverage(α) − α|.

**Prediction-range guard.** Before any metric is computed, each run's predictions and interval bounds are clipped to the range of labels in its own training split. This uses training information only and is applied identically to every model, task and seed; no run or test point is removed. It is a no-op for QRF. `results_r3/clipped_runs.csv` lists every run in which a prediction was clipped.

The original single-run code (`train.py`, with `scripts/` holding its job scripts) is kept for provenance; it produced the models used for the PI1M screening and the case studies.

## Citation

Xu, J.; Liu, G.; Guo, R.; Jiang, M.; Luo, T. POINT²: A Polymer Informatics Training and Testing Database. *Digital Discovery* 2026, DOI to be added.

## License

MIT (see `LICENSE`). PoLyInfo-derived values are not redistributed; see `data/README.md`.
