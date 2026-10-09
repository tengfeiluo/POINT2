#!/usr/bin/env python
"""Aggregate R3 runs with the prediction-range guard.

Every run's predictions (and interval bounds) are clipped to the range of labels observed in that
run's TRAINING split before any metric is computed. Uses training information only; applied
identically to every model, task and seed; no run is dropped. Fingerprint rows come from
results_r3 (the main 660-run set), graph rows from results_r3_search200 (200 Optuna trials, the
budget used in the original work). Writes merged summary_mean_std.csv for make_tables.py.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_r3 import uq_metrics

OUT = sys.argv[3] if len(sys.argv) > 3 else './results_final'
os.makedirs(OUT, exist_ok=True)
SRC = [(sys.argv[1] if len(sys.argv) > 1 else './results_r3', lambda m: not m.startswith('torch')),
       (sys.argv[2] if len(sys.argv) > 2 else './results_r3_search200', lambda m: m.startswith('torch'))]

rows, touched = [], []
for root, keep in SRC:
    for f in sorted(glob.glob(f'{root}/*/*/seed*/metrics.json')):
        j = json.load(open(f)); d = os.path.dirname(f)
        if not keep(j['model']): continue
        ytr = np.load(f'{d}/y_train.npy'); lo, hi = float(ytr.min()), float(ytr.max())
        for split in ('test', 'train'):
            y = np.load(f'{d}/y_{split}.npy'); yh = np.load(f'{d}/{split}_y_pred.npy')
            sg = np.load(f'{d}/{split}_sigma.npy'); lb = np.load(f'{d}/{split}_lb.npy'); ub = np.load(f'{d}/{split}_ub.npy')
            n_clip = int(((yh < lo) | (yh > hi)).sum())
            yc, lbc, ubc = np.clip(yh, lo, hi), np.clip(lb, lo, hi), np.clip(ub, lo, hi)
            m = uq_metrics(y, yc, sg, lbc, ubc)
            if split == 'test' and n_clip: touched.append(dict(target=j['target'], model=j['model'], fp=j['fpmethod'], seed=j['seed'], n_clipped=n_clip, rmse_before=j['metrics']['test']['rmse'], rmse_after=m['rmse']))
            rows.append(dict(target=j['target'], model=j['model'], fp=j['fpmethod'] if not j['model'].startswith('torch') else 'graph',
                             seed=j['seed'], split=split, n_search=j.get('n_search'), rmse=m['rmse'], mae=m['mae'], r2=m['r2'],
                             spearman=m.get('spearman_err_sigma'), ece=m.get('ece'), cov90=m.get('coverage_90_interval'), n_clipped=n_clip))
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/all_runs.csv', index=False)
pd.DataFrame(touched).to_csv(f'{OUT}/clipped_runs.csv', index=False)
g = df.groupby(['split', 'target', 'model', 'fp'])
summary = g.agg(n_seeds=('seed', 'count'), **{f'{k}_{s}': (k, s) for k in ('rmse', 'mae', 'r2', 'spearman', 'ece', 'cov90') for s in ('mean', 'std')}).reset_index()
summary.to_csv(f'{OUT}/summary_mean_std.csv', index=False)
te = df[df.split == 'test']
print(f'{len(te)} test runs; graph n_search values: {sorted(te[te.model.str.startswith("torch")].n_search.unique())}')
print(f'runs with >=1 clipped test prediction: {len(touched)}; runs where clipping moved RMSE by >1%: '
      f'{sum(1 for t in touched if abs(t["rmse_after"]-t["rmse_before"])/t["rmse_before"] > 0.01)}')
for t in touched:
    if abs(t['rmse_after']-t['rmse_before'])/t['rmse_before'] > 0.01: print('  ', t)
