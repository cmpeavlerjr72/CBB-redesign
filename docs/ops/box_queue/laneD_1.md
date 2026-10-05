# laneD_1: full retrain on the clean foundation, F_R first, then F_T, each followed by its full-size gate read (verified truth)

- **Commit:** `58348b35e02082e880a13a44a18521a9620a89f1` (pushed to `main`; contains every script below)
- **Lane:** D (overnight 2026-09-30). Doc: `docs/ops/full_retrain_chain_2026-09-30.md`. Nothing is adopted and no
  served file is written; all outputs go to `data/processed/models/full_retrain_v1/<tag>/` and `results/engine_v0/fr1_*`.
- **Measured cost:** retrain about 3.0 core-hours for F_R and about 2.0 for F_T (it reuses F_R's shared stages). The gate
  read is 5,710 games x 200 seeds, about 11-16 min at 90 workers, as tonight's S0 read was.
- **Expected wall time:** about 30-35 min for F_R, about 25-30 min for F_T. The long poles are one PO `first` fit (about
  8 min single-thread locally) and one rotation window (about 13 min).
- **Memory:** about 60 single-thread fit processes at once per variant. PO fits read about 2.6 M rows each, and the
  per-fit RSS was not measured. Run the two commands one after the other. If `free -g` shows under 40% used while F_R's
  trainers run, F_T may start alongside.

## 0. Setup (once)
```
git fetch && git checkout 58348b35e02082e880a13a44a18521a9620a89f1
chmod +x scripts/*.sh
# inputs (all used on the box tonight; skip what is already there):
#   HF: raw, model_artifacts (fg/rebound designs + shooter cache), engine_inputs_v3, team_rate_tables
# possessions_v4 has NO HF key: the chain's first stage rebuilds it from raw (about 1-2 min per season, single process)
scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_T --tag FR_box_v1 --cores 90 --preflight-only
# must print "preflight: 25/25 inputs present" (exit 0); any MISSING line names its HF key
```

## 1. F_R (served expanding-mean team-rate features) + full gate read
```
scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_R --tag FR_box_v1 --cores 90 --parallel \
  --gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers 90 \
  --gate-ref docs/tests/v3box_grades_2026-09-30/v3full_S0_s200_o0__verified.md \
  --gate-noise docs/tests/v3box_grades_2026-09-30/v3full_S0f1_s200_o1000__verified.md,docs/tests/v3box_grades_2026-09-30/v3full_S0f2_s200_o2000__verified.md,docs/tests/v3box_grades_2026-09-30/v3full_S0f3_s200_o3000__verified.md,docs/tests/v3box_grades_2026-09-30/v3full_S0f4_s200_o4000__verified.md
```

## 2. F_T (E3 v4 features) + full gate read: reuses F_R's rotation, designs, foul state and clock
```
scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_T --tag FT_box_v1 --reuse-from FR_box_v1 --cores 90 --parallel \
  --gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers 90 \
  --gate-ref docs/tests/v3box_grades_2026-09-30/v3full_S0_s200_o0__verified.md \
  --gate-noise <the same four S0f1..S0f4 paths as above>
```
- **Resume after a spot reclaim:** rerun the same command. Finished stages are skipped, and the trainers resume from
  their per-fit checkpoints.
- **Flags / env:** the chain sets `CBB_TRUTH=verified_v1`, the served `ENGINE_*` pins and the thread pins to 1 in every
  subprocess. Do not set any other `ENGINE_*` variable. The anchor switch stays off. The chain serves artifacts
  in-process (`scripts/run_engine_overlay_v1.py`), so no extra docker mounts are needed.
- **If time allows, optional (Decision 12 floors on the arm itself):** rerun either command with
  `--gate-offsets 0,1000,2000,3000,4000`. The finished offset 0 is skipped.

## Expected outputs (please sync back to the same local paths)
- `results/engine_v0/fr1_FR_box_v1_full_s200_o0/` and `results/engine_v0/fr1_FT_box_v1_full_s200_o0/` (the `results` key)
- `data/processed/models/full_retrain_v1/FR_box_v1/` and `.../FT_box_v1/`: every stage's `.done.json`, `stage.log`,
  artifacts, and `gate/*__verified.md` plus `gate/pair_ref_vs_*_n{0..3}.md`. These ride the `model_artifacts` key,
  because the directory is gitignored under `data/processed/models/`.

## The one line to decide
Run F_R first and F_T second, one after the other unless the memory check allows both at once. If only one fits
before 02:00, run F_R.
