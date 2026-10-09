#!/usr/bin/env python
"""Emit LaTeX table bodies for Tables 2 / A.7-A.10 from the R3 summary, preserving the published layout.

Rows: QRF-<fp> x5, QRF (avg), MLP-D-<fp> x5, MLP-D (avg), GNN, GREA, then the GPT-4o-mini rows
carried over verbatim from the previous version (the LLM baselines were not re-run).
Columns: Tg, Tm, TC, FFV, density, P(O2), P(N2), P(H2), P(CH4), P(CO2).
Values are mean +/- std over the converged seeds; the top two per column are underlined.
"""
import sys, pandas as pd, numpy as np

COLS = ['Tg', 'Tm', 'TC', 'FFV', 'density', 'O2_msa_log10', 'N2_msa_log10', 'H2_msa_log10', 'CH4_msa_log10', 'CO2_msa_log10']
FPS = [('Morgan', 'Morgan'), ('MACCS', 'MACCS'), ('RDKit', 'RDKit'), ('TopologicalTorsion', 'TT'), ('AtomPair', 'AP')]
# LLM rows are not re-run; carried over from the R2 version so the comparison stays in the table.
LLM = {
 'RMSE': {'GPT-4o-mini-0': [100.92,110.47,0.112,0.178,0.189,2.949,3.023,4.684,3.212,2.033],
          'GPT-4o-mini-5': [95.54,114.75,0.096,0.039,0.182,1.440,1.629,1.198,1.656,1.327],
          'GPT-4o-mini-10':[91.11,111.03,0.092,0.035,0.172,1.320,1.520,1.170,1.533,1.290],
          'GPT-4o-mini-20':[85.98,105.65,0.083,0.031,0.169,1.267,1.381,1.064,1.342,1.257]}}

def fmt(m, s, dec):
    if m is None or (isinstance(m, float) and np.isnan(m)): return '{--}'
    return f'{m:.{dec}f} $\\pm$ {s:.{dec}f}' if s is not None and not np.isnan(s) else f'{m:.{dec}f}'

def build(metric, dec_map, lower_is_better=True, include_llm=False):
    s = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'results_r3/summary_mean_std.csv')
    te = s[s.split == 'test']
    lines, table = [], {}
    def get(model, fp, col):
        r = te[(te.model == model) & (te.fp == fp) & (te.target == col)]
        if r.empty: return None, None
        return float(r[f'{metric}_mean'].iloc[0]), float(r[f'{metric}_std'].iloc[0])
    for model, label, fp_list in [('QuantileRandomForest', 'QRF', FPS), ('DropoutMLP', 'MLP-D', FPS)]:
        for fp, short in fp_list:
            table[f'{label}-{short}'] = [get(model, fp, c) for c in COLS]
        vals = [[table[f'{label}-{sh}'][i][0] for _, sh in fp_list if table[f'{label}-{sh}'][i][0] is not None] for i in range(len(COLS))]
        table[f'{label} (avg)'] = [(float(np.mean(v)) if v else None, None) for v in vals]
    for model, label in [('torch-GNN', 'GNN'), ('torch-GREA', 'GREA')]:
        table[label] = [get(model, 'graph', c) for c in COLS]
    if include_llm and metric.upper() in LLM:
        for k, v in LLM[metric.upper()].items():
            table[k] = [(x, None) for x in v]
    # underline the best two per column among the non-avg model rows
    body_rows = [k for k in table if '(avg)' not in k and not k.startswith('GPT')]
    best = {}
    for i in range(len(COLS)):
        vals = [(table[k][i][0], k) for k in body_rows if table[k][i][0] is not None]
        vals.sort(reverse=not lower_is_better)
        best[i] = {k for _, k in vals[:2]}
    for k, cells in table.items():
        cs = []
        for i, (m, sd) in enumerate(cells):
            dec = dec_map[COLS[i]]
            txt = fmt(m, sd, dec)
            if k in best.get(i, set()) and txt != '{--}': txt = '\\underline{' + txt + '}'
            cs.append('{' + txt + '}' if txt != '{--}' else txt)
        pre = '\\rowcolor{gray!20}\n' if '(avg)' in k else ''
        lines.append(f'{pre}{k:16s} & ' + ' & '.join(cs) + ' \\\\')
        if k in ('QRF (avg)', 'MLP-D (avg)', 'GREA'): lines.append('\\hline')
    return '\n'.join(lines)

DEC = {'Tg':2,'Tm':2,'TC':3,'FFV':3,'density':3,'O2_msa_log10':3,'N2_msa_log10':3,'H2_msa_log10':3,'CH4_msa_log10':3,'CO2_msa_log10':3}
if __name__ == '__main__':
    which = sys.argv[2] if len(sys.argv) > 2 else 'rmse'
    lower = which not in ('r2', 'spearman')
    print(build(which, DEC, lower_is_better=lower, include_llm=(which in ('rmse', 'mae', 'r2'))))
