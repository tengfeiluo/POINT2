# Multi-seed results (R3 protocol)

Produced by `train_r3.py` and aggregated by `aggregate_r3.py`. Every model is trained under the
protocol described in the manuscript: hyperparameters and early stopping are selected on a 10%
internal validation subset drawn from the training split, the final model of every family is
trained on the remaining 90%, and the fixed benchmark test split is used once for the final score.
Five random seeds per configuration.

| File | Contents |
|---|---|
| `all_runs.csv` | one row per run per split: task, model, fingerprint, seed, RMSE, MAE, R², Spearman(\|error\|, σ), ECE, 90% interval coverage, runtime |
| `summary_mean_std.csv` | mean and standard deviation over seeds for every (task, model, fingerprint) |
| `table_rmse.csv`, `table_mae.csv`, `table_r2.csv`, `table_spearman.csv`, `table_ece.csv` | the manuscript tables in "mean ± std" form |
| `excluded_runs.csv` | runs excluded by the convergence filter, with the reason |
| `split/` | the split-sensitivity study: benchmark vs repeated random vs scaffold vs cluster splits |

## Convergence filter

A run is excluded from the reported means if any of its test predictions falls outside the observed
label range extended by five times its span, i.e. it did not produce a usable regression model. The
criterion is applied identically to all 660 runs and excluded exactly one (GREA on Tg, one seed, which
emitted predictions of order 10³–10⁴ °C against an observed range of −123 to 495 °C). **Individual test
points are never dropped:** every metric is computed over the complete test split.

## ECE

For confidence level α the central interval is ŷ ± z_((1+α)/2)·σ, with σ the standard deviation over
100 MC-dropout passes (MLP-D), half the 5–95% quantile width divided by 1.645 (QRF), or the square
root of the rationale–environment variance (GREA). ECE is the mean of |coverage(α) − α| over
α ∈ {0.1, …, 0.9}. Vanilla GNNs and the LLM baselines give point predictions and have no ECE.
