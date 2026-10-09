#!/usr/bin/env python
"""Collect results_r3/<target>/<tag>/seed*/metrics.json into mean +/- std tables (CSV + LaTeX-ready)."""
import glob, json, os, sys
import numpy as np, pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else './results_r3'

def diverged(run_dir, y_test_path='y_test.npy', pred='test_y_pred.npy'):
    """Objective, pre-stated convergence filter, applied identically to every run.

    A run is treated as non-convergent if any test prediction falls outside the observed label
    range extended by five times its span. Such a run has not produced a usable regression model
    (the excursions are orders of magnitude beyond any physical value) and its RMSE reflects the
    divergence rather than the method. NOTE: the filter operates on whole RUNS. Individual test
    points are never dropped -- every reported metric is computed over the complete test split.
    """
    try:
        y = np.load(os.path.join(run_dir, pred)); yt = np.load(os.path.join(run_dir, y_test_path))
    except Exception:
        return False, None
    lo, hi = float(yt.min()), float(yt.max()); span = hi - lo
    n_ext = int(((y < lo - 5 * span) | (y > hi + 5 * span)).sum())
    return n_ext > 0, n_ext

rows = []
excluded = []
for f in glob.glob(f'{root}/*/*/seed*/metrics.json'):
    j = json.load(open(f))
    bad, n_ext = diverged(os.path.dirname(f))
    if bad:
        excluded.append(dict(target=j['target'], model=j['model'],
                             fp=j['fpmethod'] if not j['model'].startswith('torch') else 'graph',
                             seed=j['seed'], n_extreme=n_ext,
                             rmse=j['metrics']['test']['rmse'], r2=j['metrics']['test']['r2']))
        continue
    for split in ('test', 'train'):
        m = j['metrics'][split]
        rows.append(dict(target=j['target'], model=j['model'], fp=j['fpmethod'] if not j['model'].startswith('torch') else 'graph',
                         seed=j['seed'], split=split, rmse=m['rmse'], mae=m['mae'], r2=m['r2'],
                         spearman=m.get('spearman_err_sigma'), ece=m.get('ece'), cov90=m.get('coverage_90_interval'),
                         seconds=j['seconds']))
df = pd.DataFrame(rows)
if df.empty:
    sys.exit('no metrics found')
n_runs_total = len(glob.glob(f'{root}/*/*/seed*/metrics.json'))
if excluded:
    pd.DataFrame(excluded).to_csv(f'{root}/excluded_runs.csv', index=False)
    print(f'EXCLUDED {len(excluded)} of {n_runs_total} runs as non-convergent '
          f'(prediction outside the label range extended by 5x its span):')
    for e in excluded:
        print(f"  {e['target']} / {e['model']} / seed {e['seed']}: {e['n_extreme']} extreme "
              f"prediction(s), RMSE {e['rmse']:.1f}, R2 {e['r2']:.2f}")
else:
    print(f'no runs excluded ({n_runs_total} runs, all convergent)')
df.to_csv(f'{root}/all_runs.csv', index=False)
g = df.groupby(['split', 'target', 'model', 'fp'])
summary = g.agg(n_seeds=('seed', 'count'), **{f'{k}_{s}': (k, s) for k in ('rmse', 'mae', 'r2', 'spearman', 'ece', 'cov90') for s in ('mean', 'std')}).reset_index()
summary.to_csv(f'{root}/summary_mean_std.csv', index=False)
print(f'{len(df)//2} runs, {len(summary)//2} (target, model, fp) combos with test metrics')
te = summary[summary.split == 'test']
for metric in ('rmse', 'mae', 'r2', 'spearman', 'ece'):
    piv = te.pivot_table(index=['model', 'fp'], columns='target', values=f'{metric}_mean')
    pivs = te.pivot_table(index=['model', 'fp'], columns='target', values=f'{metric}_std')
    fmt = piv.round(3).astype(str) + ' ± ' + pivs.round(3).astype(str)
    fmt.to_csv(f'{root}/table_{metric}.csv')
    rel = (pivs / piv.abs()).replace([np.inf, -np.inf], np.nan)
    print(f'{metric}: median std/|mean| across combos = {np.nanmedian(rel.values):.3f}, max = {np.nanmax(rel.values):.3f}')
print('wrote', f'{root}/summary_mean_std.csv and table_*.csv')
