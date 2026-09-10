# L4 PLAYER ATTRIBUTION: how to finish the bake-off

Written 2026-09-10 at the session's hard stop. Everything except the **run** is
done: the module, the trainer, the tests, `model.md`, `features.md`, the
pre-registration in `experiments.md` section 1, and the registrations in
`docs/models/README.md` and `docs/models/change_ledger.md`.

## State on disk

| Path | State |
|---|---|
| `src/cbb_sim/models/attribution.py` | complete, imports clean |
| `scripts/train_attribution_v1.py` | complete |
| `tests/test_attribution.py` | complete — **27 passed** (`.venv/Scripts/python.exe -m pytest tests/test_attribution.py -q`) |
| `docs/models/attribution/{experiments,model,features}.md` | written; `experiments.md` section 1 is the verbatim pre-registration, the results section carries a PARTIAL marker |
| `data/processed/models/attribution/events_{reb_off,reb_def,made_fga,tov,miss_fga}_v2.parquet` | **BUILT** (2024 + 2025), 222,775 / 526,310 / 546,514 / 250,065 / 689,375 rows |
| `data/processed/models/attribution/asof_v2.parquet` | **BUILT**, 206,169 player-games |
| `data/processed/models/attribution/team_asof_v2.parquet` | **BUILT**, 21,246 team-games |
| `data/processed/models/attribution/build_report_v2.json` | **BUILT** (coverage per target and season) |
| `data/processed/models/attribution/results_v1.json`, `attribution_params_v1.json`, `report_v1.md` | whatever the interrupted run had written; the trainer rewrites all three after **every** target, so they are always internally consistent |

The tables are the expensive part (113 s) and they are cached: the trainer reuses
them unless `--rebuild` is passed, so a resumed run starts straight into fitting.

## A run may still be in flight

The interrupted launch (pid was 285103, log `/tmp/attr_f1.log`) was left running
rather than killed. It rewrites `results_v1.json`, `attribution_params_v1.json`
and `report_v1.md` after **every** target, so those three files are always
internally consistent -- but if it got further than the PARTIAL section of
`experiments.md` says, **the files on disk are ahead of the doc**. Check
`report_v1.md` and `train_log_v1.txt` first, and note that anything it produced
carries the reduced flags listed above, so it is triage material, not a result to
carry into a decision.

## The command that finishes it

```bash
cd /c/Users/devuser/CBB-clean-sheet
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/train_attribution_v1.py \
    --version v2 > /tmp/attr_full.log 2>&1 &
```

That is the full pre-registration: both folds, all eight targets, the three-rung
LightGBM grid, three seed refits, 40 game-level draws, 200 bootstrap reps.

### If time is short, run it in this order

```bash
# 1. the three binaries -- minutes, not hours (logistic ridge + one tree each)
.venv/Scripts/python.exe scripts/train_attribution_v1.py --version v2 \
    --targets assisted stolen blocked

# 2. the two cheap choice targets
.venv/Scripts/python.exe scripts/train_attribution_v1.py --version v2 \
    --targets steal block

# 3. the three expensive ones (REB_def and assist are the largest populations)
.venv/Scripts/python.exe scripts/train_attribution_v1.py --version v2 \
    --targets REB_off assist REB_def
```

**Each invocation overwrites `results_v1.json` with only the targets it ran.**
Either run all eight in one invocation, or merge the JSONs before quoting a
combined table. Running one target at a time is for triage, not for the record.

## What was NOT run, and the flags that were used

The interrupted launch was:

```
--skip-wf --skip-composed --lgbm-grid 2 --lgbm-seeds 2 --sim-draws 20 --boot-reps 100
```

so relative to the pre-registration it is missing:

1. **the within-2025 robustness fold** (`--skip-wf`) — required by the
   pre-registration;
2. **the composed per-player diagnostic** (`--skip-composed`) — reported only,
   never a gate;
3. **the third LightGBM parameter rung** and the **third seed refit**, which
   makes the tree's own noise floor a 2-point SD rather than a 3-point one;
4. game-level draws at 20 instead of 40 and bootstrap reps at 100 instead of 200.

None of those change what is compared; all of them change how tight the floors
are. A number carried into a decision must come from a run without them.

## The one thing to watch: P3 runtime

The grouped-softmax objective is a Python callback invoked once per boosting
iteration over K·n rows, and `REB_def` (≈ 480 k modelled events × 5) and `assist`
(≈ 265 k × 4) are the largest choice populations in the project. Measured on this
machine: roughly two minutes per parameter rung on `REB_off` (≈ 84 k train
events), so the full grid over all five choice targets and both folds is hours,
not minutes.

Levers, in order of preference, none of which changes what is compared:

1. `--lgbm-grid 1` — the grid's first rung won the inner split on `REB_off`
   (1.408099) and is the rung `usage` adopted on every one of its five classes.
2. Subsample whole **games** for the parameter search only (the search already
   runs on the inner split, so this only shrinks the search).
3. Move `group_softmax_objective` to a compiled form. `CLAUDE.md` already allows
   numba in the sim loop; this is the same shape of fix.

Do **not** reduce `--lgbm-seeds` below 2: the tree's noise floor is a seed-varied
refit SD and with one seed there is no floor to compare against, which is the
thing the decision rule needs most.

## Where the decision happens

`train_attribution_v1.decide` implements the pre-registered rule verbatim and runs
per target automatically; the winner and its reason are printed, stored in
`results_v1.json` and rendered into `report_v1.md`. Appending the results to
`experiments.md` is a copy of `report_v1.md` under a dated heading — the same
convention `usage` and `rebound` use. Replace the PARTIAL marker when the run is
complete, and move the status lines in `model.md`, `docs/models/README.md` and the
change ledger in the same commit.
