# Template-based polymer retrosynthesis and PolyScore

* `retro.py` — the complete implementation: the SMARTS reaction-template library (`templates` dict, three
  polymerization classes — addition, condensation, ring-opening — **82 templates**), depolymerization of a
  repeat unit into candidate monomer sets, SA-score based monomer scoring (`SA_Score/`, Ertl & Schuffenhauer),
  and **PolyScore** (harmonic mean of the monomer scores of a route; the polymer's PolyScore is the minimum over routes).
  To add a template, append a SMARTS entry to the `templates` dict; each entry is `name: (reaction SMARTS, termination rule)`.
* `data/polymerization_reactions_training.csv`, `data/polymerization_reactions_testing.csv` — the curated
  polymerization-reaction set (PoLyInfo PID, PubChem CIDs of the monomers, monomer and polymer SMILES,
  polymerization type, polymer class). Training rows: 3,460 (3,237 PIDs); testing rows: 388 (360 PIDs).
* `results/reaction_accuracy_coverage_{train,test,case}.csv` — per-PID outcome of applying the template
  library (`has_success` = at least one route found; `correct` = a found route matches the recorded monomers).

Run: `python retro.py --help` (RDKit required).
