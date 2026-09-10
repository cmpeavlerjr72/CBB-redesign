# Scripts

Flat directory, no subpackages. Every script is prefixed by what it does so the
directory listing tells you the pipeline stage without opening a file. Same
convention as `cfb-props-sim`.

| Prefix | Meaning |
|---|---|
| `pull_` | Fetch raw data from an external source (hoopR, KenPom, odds APIs) into `data/raw/` |
| `build_` | Deterministic transform of raw/reference data into `data/processed/` (feature tables, lookups, crosswalks) |
| `train_` | Fit a model artifact and write it under `data/processed/models/` |
| `run_` | Execute the sim engine (a slate, a backtest, a sweep) |
| `exp_` | One-off experiment / bake-off arm, not part of the standing pipeline |
| `diag_` | Diagnostics — inspect an artifact, a data file, or an engine run without changing anything |
| `grade_` | Score sim output against actuals or market lines |
| `leak_` | Leak-detection checks (created_at < tipoff, temporal integrity) |
| `chain_` | Orchestrates a sequence of other scripts end-to-end |
| `inv{N}_` | Investigation-numbered script tied to an entry in `docs/models/change_ledger.md` or `docs/plans/` |

## Rule: trainers are versioned filenames, never overwritten

A training script's filename is part of the record. Once `train_shot_make_v3.py`
has produced a shipped or evaluated artifact, it is never edited in place —
a change gets a new filename (`train_shot_make_v4.py`). This keeps every
artifact under `data/processed/models/` reproducible from a script that still
exists and still means what it meant when the artifact was built.
