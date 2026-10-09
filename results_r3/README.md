# Multi-seed results (R3 protocol)

Produced by `train_r3.py` and aggregated by `aggregate_r3_guard.py`. Fingerprint models (QRF, MLP-D) come from the main run set; graph models (GNN, GREA) from the 200-trial Optuna runs. Every model is trained under the
protocol described in the manuscript: hyperparameters and early stopping are selected on a 10%
internal validation subset drawn from the training split, the final model of every family is
trained on the remaining 90%, and the fixed benchmark test split is used once for the final score.
Five random seeds per configuration.

| File | Contents |
|---|---|
| `all_runs.csv` | one row per run per split: task, model, fingerprint, seed, RMSE, MAE, R², Spearman(\|error\|, σ), ECE, 90% interval coverage, runtime |
| `summary_mean_std.csv` | mean and standard deviation over seeds for every (task, model, fingerprint) |
| `table_rmse.csv`, `table_mae.csv`, `table_r2.csv`, `table_spearman.csv`, `table_ece.csv` | the manuscript tables in "mean ± std" form |
| `clipped_runs.csv` | runs in which at least one test prediction was clipped by the prediction-range guard, with RMSE before and after |
| `split/` | the split-sensitivity study (Appendix A.1, Table A.12): benchmark vs repeated random vs scaffold vs cluster splits, one JSON per task and model |

## Prediction-range guard

Every run's predictions and interval bounds are clipped to the range of labels observed in its own
training split before metrics are computed. The guard uses training information only, is applied identically
to all runs, and removes no run and no test point. On the ten benchmark tasks it changes one reported cell by
more than 1% (GNN on T_m). It cannot affect QRF, whose predictions are quantiles of training labels.

## ECE

For confidence level α the central interval is ŷ ± z_((1+α)/2)·σ, with σ the standard deviation over
100 MC-dropout passes (MLP-D), half the 5–95% quantile width divided by 1.645 (QRF), or the square
root of the rationale–environment variance (GREA). ECE is the mean of |coverage(α) − α| over
α ∈ {0.1, …, 0.9}. Vanilla GNNs and the LLM baselines give point predictions and have no ECE.
