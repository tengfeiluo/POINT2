#!/usr/bin/env python
"""
POINT2 -- R3 re-run script (2026-09-14).

Same data, same fixed 80/20 benchmark split (random_state=42) as train.py, but:
  * an internal validation subset (default 10 %) is drawn from the TRAINING split and used for
    hyperparameter selection / early stopping; the test split is never touched before final scoring;
  * every run takes --seed (model initialisation, MC dropout, validation draw);
  * QRF and MLP-D can run a small documented grid search on the validation subset (--search);
  * torch-GNN / torch-GREA pass the validation subset (not the training set) to autofit();
  * RMSE, MAE, R2, Spearman(|err|, sigma), 90 % interval coverage and ECE are written to metrics.json.

Output: <out_root>/<target>/<model>_<fp>_<radius>_<nbits>/seed<seed>/
"""
import argparse, os, sys, json, time, random, itertools
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, MACCSkeys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, norm
RDLogger.DisableLog('rdApp.*')

DATA = {
    'Tg': 'data/labeled/polyinfo/Tg_SMILES_class_pid_polyinfo_median.csv',
    'Tm': 'data/labeled/polyinfo/Tm_SMILES_class_pid_polyinfo_median.csv',
    'density': 'data/labeled/polyinfo/density_SMILES_class_pid_polyinfo_median.csv',
    'TC': 'data/labeled/nd/MD_TC_Oct21_2024_wSMILES.csv',
    'Bulk_modulus_GPa': 'data/labeled/nd/modulus_yuhan_Nov3_2024.csv',
    'FFV': 'data/labeled/uwm/YingLi_FFV_MD_homopolymer_polyamides_combined.csv',
}
for g in ['O2', 'N2', 'H2', 'CO2', 'CH4']:
    DATA[f'{g}_msa'] = DATA[f'{g}_msa_log10'] = f'data/labeled/msa/{g}_raw.csv'

QRF_GRID = dict(n_estimators=[100, 300, 500], max_depth=[None, 10, 20, 40], min_samples_split=[2, 5, 10])
MLP_GRID = [  # (hidden layer widths), dropout 0.2, Adam lr 1e-3, batch 32, <=100 epochs, early stopping (patience 10)
    (512, 128), (128,), (256,), (512,), (128, 128), (256, 256), (512, 512),
    (128, 128, 128), (256, 256, 256), (512, 512, 512),
]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf; tf.random.set_seed(seed)
    except Exception:
        pass
    try:
        import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def smiles_to_fingerprint(smiles_list, method='Morgan', radius=2, n_bits=2048):
    """Identical bit definitions to train.py (Morgan/RDKit/MACCS/TopologicalTorsion/AtomPair)."""
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if not mol:
            raise ValueError(f'invalid SMILES: {s}')
        if method == 'Morgan':
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif method == 'RDKit':
            fp = Chem.RDKFingerprint(mol, maxPath=radius, fpSize=n_bits)
        elif method == 'MACCS':
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif method == 'TopologicalTorsion':
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
        elif method == 'AtomPair':
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        else:
            raise ValueError(method)
        fps.append(np.array(fp))
    return np.array(fps)


def load_split(data_root, target, test_size=0.2):
    df = pd.read_csv(os.path.join(data_root, DATA[target]))
    smiles = df['SMILES'].astype(str)
    y = df[target].to_numpy(dtype=float)
    if np.any(~np.isfinite(y)):
        raise ValueError('non-finite labels')
    idx = np.arange(len(df))
    # identical call signature/order to train.py -> identical benchmark split
    idx_tr, idx_te, y_tr, y_te, s_tr, s_te = train_test_split(idx, y, smiles, test_size=test_size, random_state=42)
    return s_tr.reset_index(drop=True), y_tr, s_te.reset_index(drop=True), y_te


def uq_metrics(y, yhat, sigma, lb, ub):
    err = np.abs(y - yhat)
    out = dict(rmse=float(np.sqrt(mean_squared_error(y, yhat))), mae=float(mean_absolute_error(y, yhat)),
               r2=float(r2_score(y, yhat)))
    if sigma is not None and np.nanstd(sigma) > 0:
        out['spearman_err_sigma'] = float(spearmanr(err, sigma)[0])
        # ECE: mean over alpha in {0.1..0.9} of |empirical coverage of the central alpha-interval - alpha|,
        # intervals = yhat +/- z_{(1+alpha)/2} * sigma (Gaussian predictive distribution)
        alphas = np.arange(0.1, 0.95, 0.1)
        cov = [float(np.mean(err <= norm.ppf((1 + a) / 2) * sigma)) for a in alphas]
        out['ece'] = float(np.mean(np.abs(np.array(cov) - alphas)))
        out['coverage_by_alpha'] = dict(zip([f'{a:.1f}' for a in alphas], cov))
    if lb is not None:
        out['coverage_90_interval'] = float(np.mean((y >= lb) & (y <= ub)))
    return out


def fit_qrf(Xtr, ytr, Xva, yva, seed, search, n_jobs, cfg=None):
    from quantile_forest import RandomForestQuantileRegressor as QRF
    if cfg is None:
        cfg = dict(n_estimators=100, max_depth=None, min_samples_split=2)
        if search:
            best = None
            for ne, md, ms in itertools.product(*QRF_GRID.values()):
                m = QRF(n_estimators=ne, max_depth=md, min_samples_split=ms, random_state=seed, n_jobs=n_jobs).fit(Xtr, ytr)
                rmse = np.sqrt(mean_squared_error(yva, m.predict(Xva)))
                print(f'  QRF search n_est={ne} depth={md} mss={ms}: val RMSE {rmse:.4f}', flush=True)
                if best is None or rmse < best[0]:
                    best, cfg = (rmse,), dict(n_estimators=ne, max_depth=md, min_samples_split=ms)
    # final fit on the full training split (train + val) with the chosen config
    m = QRF(random_state=seed, n_jobs=n_jobs, **cfg).fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))

    def predict(X):
        yhat = m.predict(X)
        q05, q95 = m.predict(X, quantiles=0.05), m.predict(X, quantiles=0.95)
        half = (q95 - q05) / 2                      # 'std' as defined in the paper (half the 90 % interval)
        sigma = half / norm.ppf(0.95)               # Gaussian-equivalent sigma used for ECE
        return yhat, sigma, q05, q95, half
    return m, cfg, predict


def fit_mlp(Xtr, ytr, Xva, yva, seed, search, cfg=None):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping

    class MCDropout(Dropout):
        def call(self, inputs, training=None):
            return super().call(inputs, training=True)

    def build(widths, dropout=0.2, lr=1e-3):
        layers = [Input(shape=(Xtr.shape[1],))]
        for w in widths:
            layers += [Dense(w, activation='relu'), MCDropout(dropout)]
        layers += [Dense(1)]
        m = Sequential(layers)
        m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
        return m

    def train(widths):
        tf.keras.utils.set_random_seed(seed)
        m = build(widths)
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=100, batch_size=32, verbose=0, callbacks=[es])
        return m, float(min(h.history['val_loss'])), len(h.history['val_loss'])

    if cfg is None:
        cfg = dict(widths=[512, 128], dropout=0.2, lr=1e-3, batch=32, max_epochs=100, patience=10)
        if search:
            best = None
            for widths in MLP_GRID:
                _, vl, ep = train(widths)
                print(f'  MLP search widths={widths}: val MSE {vl:.4f} ({ep} epochs)', flush=True)
                if best is None or vl < best:
                    best, cfg['widths'] = vl, list(widths)
    m, vl, ep = train(tuple(cfg['widths']))
    cfg['epochs_trained'] = ep

    def predict(X, n_iter=100):
        preds = np.array([m.predict(X, verbose=0).ravel() for _ in range(n_iter)])
        yhat, sigma = preds.mean(0), preds.std(0)
        q05, q95 = np.percentile(preds, 5, axis=0), np.percentile(preds, 95, axis=0)
        return yhat, sigma, q05, q95, (q95 - q05) / 2
    return m, cfg, predict


def fit_torch(model_type, s_tr, y_tr, s_va, y_va, seed, n_trials, target, epochs=500, prior_cfg=None):
    from torch_molecule import GREAMolecularPredictor, GNNMolecularPredictor
    from torch_molecule.utils.search import ParameterType, ParameterSpec
    Cls = GREAMolecularPredictor if model_type == 'torch-GREA' else GNNMolecularPredictor
    common = dict(num_tasks=1, task_type='regression', batch_size=512, epochs=epochs,
                  evaluate_criterion='r2', evaluate_higher_better=True, verbose=True,
                  model_name=f"{'GREA' if model_type == 'torch-GREA' else 'GNN'}_{target}")
    # search space copied verbatim from Jiaxin's train.py (torch-molecule v0.1.1)
    space = {
        'gnn_type': ParameterSpec(ParameterType.CATEGORICAL, ['gin-virtual', 'gcn-virtual', 'gin', 'gcn']),
        'norm_layer': ParameterSpec(ParameterType.CATEGORICAL, ['batch_norm', 'layer_norm', 'size_norm']),
        'num_layer': ParameterSpec(ParameterType.INTEGER, value_range=(2, 5)),
        'emb_dim': ParameterSpec(ParameterType.INTEGER, value_range=(256, 512)),
        'learning_rate': ParameterSpec(ParameterType.FLOAT, value_range=(1e-4, 1e-2)),
        'drop_ratio': ParameterSpec(ParameterType.FLOAT, value_range=(0.05, 0.5)),
        'augmented_feature': ParameterSpec(ParameterType.CATEGORICAL, ['maccs,morgan', 'maccs', 'morgan', None]),
    }
    if model_type == 'torch-GREA':
        space['num_layer'] = ParameterSpec(ParameterType.INTEGER, value_range=(2, 6))
        space['gamma'] = ParameterSpec(ParameterType.FLOAT, value_range=(0.25, 0.75))
    keep = list(space)
    if prior_cfg is not None:
        # seeds 1..4: same hyperparameters as the seed-0 search, fresh initialisation, early stopping on the val subset
        m = Cls(**common, **{k: prior_cfg[k] for k in keep if k in prior_cfg})
        m.fit(s_tr.tolist(), y_tr, s_va.tolist(), y_va)
    else:
        m = Cls(**common)
        # THE FIX: validation subset (drawn from the training split) instead of the training set itself
        m.autofit(X_train=s_tr.tolist(), y_train=y_tr, X_val=s_va.tolist(), y_val=y_va,
                  search_parameters=space, n_trials=n_trials)
    p = m.get_params()
    cfg = {k: p.get(k) for k in keep}
    cfg['fitting_epoch'] = p.get('fitting_epoch')

    def predict(S):
        r = m.predict(list(S))
        yhat = np.asarray(r['prediction']).reshape(-1)
        var = np.asarray(r['variance']).reshape(-1) if r.get('variance') is not None and np.size(r.get('variance')) else np.zeros_like(yhat)
        sigma = np.sqrt(np.clip(var, 0, None))
        return yhat, sigma, yhat - 1.96 * sigma, yhat + 1.96 * sigma, 1.96 * sigma
    return m, cfg, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_property', required=True, choices=sorted(DATA))
    ap.add_argument('--model', required=True, choices=['QuantileRandomForest', 'DropoutMLP', 'torch-GNN', 'torch-GREA'])
    ap.add_argument('--fpmethod', default='Morgan', choices=['Morgan', 'RDKit', 'MACCS', 'TopologicalTorsion', 'AtomPair'])
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--n_bits', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--search', action='store_true', help='grid search (QRF/MLP) on the validation subset')
    ap.add_argument('--n_search', type=int, default=50, help='Optuna trials for torch models')
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--data_root', default='.')
    ap.add_argument('--out_root', default='./results_r3')
    a = ap.parse_args()

    t0 = time.time(); set_seed(a.seed)
    tag = f'{a.model}_{a.fpmethod}_{a.radius}_{a.n_bits}' if not a.model.startswith('torch') else a.model
    run_dir = os.path.join(a.out_root, a.target_property, tag, f'seed{a.seed}')
    cfg_path = os.path.join(a.out_root, a.target_property, tag, 'best_config.json')
    os.makedirs(run_dir, exist_ok=True)

    s_tr, y_tr, s_te, y_te = load_split(a.data_root, a.target_property)
    # internal validation subset from the TRAINING split only
    s_fit, s_va, y_fit, y_va = train_test_split(s_tr, y_tr, test_size=a.val_frac, random_state=a.seed)
    s_fit, s_va = s_fit.reset_index(drop=True), s_va.reset_index(drop=True)
    print(f'{a.target_property} {tag} seed={a.seed}: train {len(s_tr)} (fit {len(s_fit)} / val {len(s_va)}), test {len(s_te)}', flush=True)

    prior_cfg = json.load(open(cfg_path)) if (os.path.exists(cfg_path) and a.seed != 0) else None
    if a.model in ('QuantileRandomForest', 'DropoutMLP'):
        X_fit = smiles_to_fingerprint(s_fit, a.fpmethod, a.radius, a.n_bits)
        X_va = smiles_to_fingerprint(s_va, a.fpmethod, a.radius, a.n_bits)
        X_tr = smiles_to_fingerprint(s_tr, a.fpmethod, a.radius, a.n_bits)
        X_te = smiles_to_fingerprint(s_te, a.fpmethod, a.radius, a.n_bits)
        if a.model == 'QuantileRandomForest':
            model, cfg, predict = fit_qrf(X_fit, y_fit, X_va, y_va, a.seed, a.search, a.n_jobs, prior_cfg)
        else:
            model, cfg, predict = fit_mlp(X_fit, y_fit, X_va, y_va, a.seed, a.search, prior_cfg)
        P_te, P_tr = predict(X_te), predict(X_tr)
    else:
        model, cfg, predict = fit_torch(a.model, s_fit, y_fit, s_va, y_va, a.seed, a.n_search, a.target_property, a.epochs, prior_cfg)
        P_te, P_tr = predict(s_te), predict(s_tr)

    res = {}
    for name, (yv, P) in {'test': (y_te, P_te), 'train': (y_tr, P_tr)}.items():
        yhat, sigma, lb, ub, half = P
        res[name] = uq_metrics(yv, yhat, sigma, lb, ub)
        res[name]['spearman_err_halfinterval'] = float(spearmanr(np.abs(yv - yhat), half)[0]) if np.nanstd(half) > 0 else None
        for k, v in dict(y_pred=yhat, sigma=sigma, lb=lb, ub=ub).items():
            np.save(os.path.join(run_dir, f'{name}_{k}.npy'), v)
    s_tr.to_csv(os.path.join(run_dir, 'smiles_train.csv'), index=False); s_te.to_csv(os.path.join(run_dir, 'smiles_test.csv'), index=False)
    s_va.to_csv(os.path.join(run_dir, 'smiles_val.csv'), index=False)
    np.save(os.path.join(run_dir, 'y_train.npy'), y_tr); np.save(os.path.join(run_dir, 'y_test.npy'), y_te)
    out = dict(target=a.target_property, model=a.model, fpmethod=a.fpmethod, radius=a.radius, n_bits=a.n_bits,
               seed=a.seed, val_frac=a.val_frac, search=a.search, n_search=a.n_search, config=cfg,
               n_train=int(len(s_tr)), n_val=int(len(s_va)), n_test=int(len(s_te)),
               metrics=res, seconds=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=2, default=str)
    if a.seed == 0 or not os.path.exists(cfg_path):
        json.dump(cfg, open(cfg_path, 'w'), indent=2, default=str)
    print(json.dumps({k: res['test'][k] for k in ('rmse', 'mae', 'r2') if k in res['test']} | {'ece': res['test'].get('ece'), 'spearman': res['test'].get('spearman_err_sigma')}), flush=True)
    print(f'done in {out["seconds"]} s -> {run_dir}', flush=True)


if __name__ == '__main__':
    main()
