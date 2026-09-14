# Benchmark data and fixed splits

One file per property task in `benchmark/`. Every file carries the **fixed 80/20 benchmark split**
used throughout the paper (`split` = `train` | `test`). The split is `sklearn.model_selection.train_test_split(..., test_size=0.2, random_state=42)`
applied to the rows of the source table in the order given here, i.e. exactly the partition in `train.py` / `train_r3.py`.

| Task | File | Columns | Source | Redistribution |
|---|---|---|---|---|
| T_g (°C) | `Tg.csv` | PID, Polymer Class, split | PoLyInfo (NIMS), one value per PoLyInfo entry (median of reported values) | **IDs and split only** — PoLyInfo terms of use do not permit redistribution of structures or property values. Registered PoLyInfo users can retrieve SMILES and T_g by PID. |
| T_m (°C) | `Tm.csv` | PID, Polymer Class, split | PoLyInfo | IDs and split only (as above) |
| density (g/cm³) | `density.csv` | PID, Polymer Class, split | PoLyInfo | IDs and split only (as above) |
| TC (W m⁻¹ K⁻¹) | `TC.csv` | SMILES, PID, TC, source, split | MD simulations, Luo group (Ma & Luo, *Mater. Today Phys.* 2022, DOI 10.1016/j.mtphys.2022.100850) | full |
| Bulk modulus (GPa) | `Bulk_modulus_GPa.csv` | SMILES, PID, Bulk_modulus_GPa, source, split | MD simulations, Luo group (unpublished set used in the case studies) | full |
| FFV (–) | `FFV.csv` | SMILES, FFV, source, split | MD simulations, Ying Li group (UW-Madison), homopolymers + polyamides | full |
| P(O2), P(N2), P(H2), P(CO2), P(CH4) (log10 Barrer) | `<gas>_msa_log10.csv` | SMILES, <gas>_msa_log10, source, split | Membrane Society of Australasia database as curated in *Cell Rep. Phys. Sci.* 2024, DOI 10.1016/j.xcrp.2024.102067 | full |

`_summary.csv` lists, per task: rows, train/test counts, the number of rows whose repeat-unit SMILES is
duplicated within the table, and the number of SMILES that occur in both splits.

## About duplicated repeat units

The PoLyInfo-derived tables are aggregated **per PoLyInfo entry (PID)**, not per repeat-unit SMILES.
A small number of distinct PIDs share the same canonical repeat unit (different polymer entries,
different measured values), so those SMILES appear more than once and can fall on both sides of the
split: T_g 11, T_m 6, density 9, TC 3, FFV 41 repeat units (see `_summary.csv`). The gas-permeability
and modulus tables contain no duplicates. The split is left as is so that all published numbers remain
reproducible; users who want a strictly de-duplicated variant can collapse on SMILES before splitting.

PI1M (the unlabeled screening pool, 995,800 p-SMILES) is available from https://github.com/RUIMINMA1996/PI1M.
