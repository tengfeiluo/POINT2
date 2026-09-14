#!/usr/bin/env python
"""Collect results_r3/<target>/<tag>/seed*/metrics.json into mean +/- std tables (CSV + LaTeX-ready)."""
import glob, json, os, sys
import numpy as np, pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else './results_r3'
rows = []
for f in glob.glob(f'{root}/*/*/seed*/metrics.json'):
    j = json.load(open(f))
    for split in ('test', 'train'):
        m = j['metrics'][split]
        rows.append(dict(target=j['target'], model=j['model'], fp=j['fpmethod'] if not j['model'].startswith('torch') else 'graph',
                         seed=j['seed'], split=split, rmse=m['rmse'], mae=m['mae'], r2=m['r2'],
                         spearman=m.get('spearman_err_sigma'), ece=m.get('ece'), cov90=m.get('coverage_90_interval'),
                         seconds=j['seconds']))
df = pd.DataFrame(rows)
if df.empty:
    sys.exit('no metrics found')
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
