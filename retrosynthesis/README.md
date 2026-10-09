# Template-based polymer retrosynthesis and PolyScore

* `retro.py` — the complete implementation: the SMARTS reaction-template library (`templates` dict, three
  polymerization classes — addition, condensation, ring-opening — **82 templates**), depolymerization of a
  repeat unit into candidate monomer sets, SA-score based monomer scoring (`SA_Score/`, Ertl & Schuffenhauer),
  and **PolyScore** (harmonic mean of the monomer scores of a route; the polymer's PolyScore is the minimum over routes).
  To add a template, add an entry `"<name>": "<reaction SMARTS>"` under its polymerization class in the `templates` dict; the termination rule
  used by `depolymerize()` is the string `"<class>-<name>"`.
* `data/polymerization_reactions_training.csv`, `data/polymerization_reactions_testing.csv` — the curated
  polymerization-reaction set (PoLyInfo PID, PubChem CIDs of the monomers, monomer and polymer SMILES,
  polymerization type, polymer class). Training rows: 3,460 (3,237 PIDs); testing rows: 388 (360 PIDs).
* `results/reaction_accuracy_coverage_{train,test,case}.csv` — per-entry outcome of applying the template
  library (`has_success` = at least one route found; `correct` = a found route matches the recorded monomers).
  Table 4 of the paper is the held-out set aggregated per PoLyInfo entry: 360 entries, Prediction Accuracy 59.7 %,
  Applicability Rate 91.9 % (one entry occupies two rows in the file).

Run: `python retro.py --help` (RDKit required).
