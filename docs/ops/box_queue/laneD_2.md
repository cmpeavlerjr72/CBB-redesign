# laneD_2: full retrain gate reads on v4-replay engine inputs (corrects laneD_1's inputs; retrain outputs reused)

- **Commit:** `7d38de5f4fee49d1149bbe2cc5c4f4cf9555b741` (pushed to `main`)
- **Lane:** D. Doc: `docs/ops/full_retrain_chain_2026-09-30.md`, section "2026-09-30 22:00".
- **What changed since laneD_1:** the chain has a new stage, `inputs_base`. It replays the possession_outcome round-2
  event block of the engine inputs on the **v4** chance tables, so the block served to the retrained PO matches what
  it trains on. The cost is about 200 s single-core for 151 dates; the box shards it across 16 processes.
- **Nothing else changed.** The corrected foul state already applied to BOTH `first` and `cont` in laneD_1, per the
  PM ruling (the overlay covers every design row, 3,037,203). The retrain stages are byte-for-byte the same code.

## Case A: laneD_1 has NOT started. Run this instead of laneD_1 (laneD_1 is then void)
Setup and preflight are the same as laneD_1 section 0, at the commit above.
```
scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_R --tag FR_box_v1 --cores 90 --parallel \
  --gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers 90 \
  --gate-ref docs/tests/v3box_grades_2026-09-30/v3full_S0_s200_o0__verified.md \
  --gate-noise docs/tests/v3box_grades_2026-09-30/v3full_S0f1_s200_o1000__verified.md,docs/tests/v3box_grades_2026-09-30/v3full_S0f2_s200_o2000__verified.md,docs/tests/v3box_grades_2026-09-30/v3full_S0f3_s200_o3000__verified.md,docs/tests/v3box_grades_2026-09-30/v3full_S0f4_s200_o4000__verified.md
scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_T --tag FT_box_v1 --reuse-from FR_box_v1 --cores 90 --parallel \
  <the same five --gate-* arguments>
```

## Case B: laneD_1 has run, or is running. Let it finish, then `git checkout` the commit above and run
```
scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_R --tag FR_box_v1 --cores 90 --parallel --redo inputs,gate \
  <the same five --gate-* arguments>
scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_T --tag FT_box_v1 --reuse-from FR_box_v1 --cores 90 --parallel --redo inputs,gate \
  <the same five --gate-* arguments>
```
- **Reused unchanged from laneD_1:** every finished stage. That is `possessions_v4`, `rotation`, `po_design`,
  `foul_state`, `po_train`, `clock`, `fg_design`, `fg_train`, `rb_design` and `rb_train`, for both tags.
- **Rerun:** only `inputs_base` (new; built once under FR_box_v1, and FT_box_v1 reuses it), `inputs`, and the full gate
  read. `--redo` renames the old `inputs/` and `gate/` to `*.prev_<timestamp>/`; nothing is deleted.
- **Gate result names:** the new reads are `results/engine_v0/fr1_{FR,FT}_box_v1_full_s200_o0_ev4`. laneD_1's v2-input
  reads, `..._o0` without the suffix, stay where they are.
- **Cost of case B:** about 3 min for inputs plus about 15 min for each gate read, per variant.

## Expected outputs (sync back to the same paths)
- `results/engine_v0/fr1_FR_box_v1_full_s200_o0_ev4/` and `fr1_FT_box_v1_full_s200_o0_ev4/` (the `results` key)
- `data/processed/models/full_retrain_v1/{FR_box_v1,FT_box_v1}/`: `inputs_base/`, `inputs/`, `gate/` (`*__verified.md`
  and `pair_ref_vs_*_n{0..3}.md`), plus everything else in case A (the `model_artifacts` key)

## The one line to decide
F_R before F_T. If only one variant fits, run F_R.
