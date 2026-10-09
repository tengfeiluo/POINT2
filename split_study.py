#!/usr/bin/env python
"""
POINT2 -- R3 split-sensitivity study (answers Referee 1, comment 5).

For a given (task, model, fingerprint) this reports test RMSE under three split regimes:
  1. 'benchmark'  the official fixed split (random_state=42), 5 model seeds  -- the paper's number
  2. 'repeated'   5 DIFFERENT random 80/20 splits (split seeds 0-4), 1 model seed each
                  -> quantifies sensitivity to the partition itself, which the model-seed study does not
  3. 'scaffold'   a deterministic Bemis-Murcko scaffold split (largest scaffold groups to train
                  until 80%, remainder to test) -> chemical extrapolation rather than interpolation
  4. 'cluster'    (optional, small tasks) Butina clustering on Morgan/Tanimoto, whole clusters held out

Writes one JSON per (task, model, fp) to <out_root>/<task>/<model>_<fp>.json
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

import train_r3 as T   # reuse DATA map, fingerprints, model fitting


def scaffold_groups(smiles):
    """Bemis-Murcko scaffold per repeat unit; dummy atoms stripped first. Acyclic -> own bucket."""
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        core = ''
        if mol is not None:
            ed = Chem.RWMol(mol)
            for a in [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0][::-1]:
                ed.RemoveAtom(a)
            try:
                m2 = ed.GetMol(); Chem.SanitizeMol(m2)
                core = MurckoScaffold.MurckoScaffoldSmiles(mol=m2)
            except Exception:
                core = ''
        groups[core or f'__acyclic_{i}'].append(i)
    return groups


def scaffold_split(smiles, test_frac=0.2):
    g = scaffold_groups(smiles)
    order = sorted(g.values(), key=len, reverse=True)      # deterministic: big scaffolds to train
    n_test_target = int(round(test_frac * len(smiles)))
    test, train = [], []
    for idx in order:
        (test if len(test) + len(idx) <= n_test_target and len(train) > 0 else train).extend(idx)
    return np.array(sorted(train)), np.array(sorted(test)), len(g)


def cluster_split(smiles, test_frac=0.2, cutoff=0.6):
    from rdkit.ML.Cluster import Butina
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in smiles]
    n = len(fps); dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1 - np.asarray(sims))
    cl = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    cl = sorted(cl, key=len, reverse=True)
    n_test_target = int(round(test_frac * n))
    test, train = [], []
    for idx in cl:
        (test if len(test) + len(idx) <= n_test_target and len(train) > 0 else train).extend(idx)
    return np.array(sorted(train)), np.array(sorted(test)), len(cl)


def run_once(df, target, idx_tr, idx_te, model, fp, radius, n_bits, seed, n_jobs):
    s_tr = df['SMILES'].iloc[idx_tr].reset_index(drop=True)
    s_te = df['SMILES'].iloc[idx_te].reset_index(drop=True)
    y_tr = df[target].to_numpy(float)[idx_tr]; y_te = df[target].to_numpy(float)[idx_te]
    s_fit, s_va, y_fit, y_va = train_test_split(s_tr, y_tr, test_size=0.1, random_state=seed)
    s_fit, s_va = s_fit.reset_index(drop=True), s_va.reset_index(drop=True)
    T.set_seed(seed)
    X = lambda s: T.smiles_to_fingerprint(s, fp, radius, n_bits)
    if model == 'QuantileRandomForest':
        _, _, predict = T.fit_qrf(X(s_fit), y_fit, X(s_va), y_va, seed, False, n_jobs)
    else:
        _, _, predict = T.fit_mlp(X(s_fit), y_fit, X(s_va), y_va, seed, False)
    yhat = predict(X(s_te))[0]
    return float(np.sqrt(mean_squared_error(y_te, yhat)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_property', required=True)
    ap.add_argument('--model', default='QuantileRandomForest')
    ap.add_argument('--fpmethod', default='Morgan')
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--n_bits', type=int, default=2048)
    ap.add_argument('--n_repeats', type=int, default=5)
    ap.add_argument('--cluster', action='store_true', help='also run a Butina cluster split (small tasks only)')
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--data_root', default='.')
    ap.add_argument('--out_root', default='./results_split')
    a = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(os.path.join(a.data_root, T.DATA[a.target_property]))
    smiles = df['SMILES'].astype(str).tolist(); n = len(df); idx = np.arange(n)
    out = dict(target=a.target_property, model=a.model, fp=a.fpmethod, n=n)

    # 1. the official benchmark split, 5 model seeds
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    v = [run_once(df, a.target_property, tr, te, a.model, a.fpmethod, a.radius, a.n_bits, s, a.n_jobs) for s in range(5)]
    out['benchmark'] = dict(rmse_mean=float(np.mean(v)), rmse_std=float(np.std(v)), values=v, n_test=len(te))
    print('benchmark split:', out['benchmark']['rmse_mean'], '+/-', out['benchmark']['rmse_std'], flush=True)

    # 2. repeated random splits
    v = []
    for k in range(a.n_repeats):
        tr, te = train_test_split(idx, test_size=0.2, random_state=1000 + k)
        v.append(run_once(df, a.target_property, tr, te, a.model, a.fpmethod, a.radius, a.n_bits, 0, a.n_jobs))
        print(f'  repeated split {k}: {v[-1]:.4f}', flush=True)
    out['repeated'] = dict(rmse_mean=float(np.mean(v)), rmse_std=float(np.std(v)), values=v)

    # 3. scaffold split
    tr, te, n_groups = scaffold_split(smiles)
    r = run_once(df, a.target_property, tr, te, a.model, a.fpmethod, a.radius, a.n_bits, 0, a.n_jobs)
    out['scaffold'] = dict(rmse=r, n_scaffolds=n_groups, n_train=len(tr), n_test=len(te))
    print('scaffold split:', r, f'({n_groups} scaffolds)', flush=True)

    # 4. cluster split
    if a.cluster:
        tr, te, n_cl = cluster_split(smiles)
        r = run_once(df, a.target_property, tr, te, a.model, a.fpmethod, a.radius, a.n_bits, 0, a.n_jobs)
        out['cluster'] = dict(rmse=r, n_clusters=n_cl, n_train=len(tr), n_test=len(te))
        print('cluster split:', r, f'({n_cl} clusters)', flush=True)

    out['seconds'] = round(time.time() - t0, 1)
    d = os.path.join(a.out_root, a.target_property); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f'{a.model}_{a.fpmethod}.json'), 'w'), indent=2)
    print(json.dumps({k: (v.get('rmse_mean') or v.get('rmse')) for k, v in out.items() if isinstance(v, dict)}), flush=True)


if __name__ == '__main__':
    main()
