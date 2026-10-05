# laneD_3 (SHIP-DECISION priority): F_T rerun with the fg_make train/serve skew fixed, tag FT_box_v2, full gate read

- **Commit:** `e998b22e83159c3ce84d915b8f6c60e5c4e0baa5` (pushed). Doc: `docs/ops/full_retrain_chain_2026-09-30.md`
  section 3.
- **Why:** lane A's skew. Under `--team-rate-table`, fg_make trained `shooter_shrunk_dev_c` from a stale raw rate.
  Local parity, served vs trained:
  - skewed F_T: corr 0.943-0.970 (FAIL);
  - fixed F_T: corr 0.99992-0.999995 (PASS);
  - F_R: corr 0.9998 (PASS).
  PO and rebound are not affected. Rebound's served arm reads no rate-derived column; lane A found PO `T` parity exact.
  So only `fg_train` is retrained.
- **Measured cost on this box (laneD_2.done):** fg_train 50 s, inputs about 5 s, parity about 2 s, gate read 514 s.
  **About 10 minutes in total.**

## Commands (in `~/cbb4` or the clone that ran laneD_2; `git fetch && git checkout e998b22...`)
```
# 1. start FT_box_v2 as a copy of FT_box_v1, so PO and rebound F_T artifacts are reused unchanged (no retrain)
cp -r data/processed/models/full_retrain_v1/FT_box_v1 data/processed/models/full_retrain_v1/FT_box_v2
# 2. redo from the first affected stage: fg_train (now trainer v2), then inputs, the new parity stage, and the gate
scripts/box_run.sh scripts/chain_full_retrain_v1.py --variant F_T --tag FT_box_v2 --reuse-from FR_box_v1 --cores 90 --parallel \
  --redo fg_train,inputs,gate \
  --gate-mode full --gate-seeds 200 --gate-offsets 0 --gate-workers 90 \
  --gate-ref docs/tests/v3box_grades_2026-09-30/v3full_S0_s200_o0__verified.md \
  --gate-noise docs/tests/v3box_grades_2026-09-30/v3full_S0f1_s200_o1000__verified.md,docs/tests/v3box_grades_2026-09-30/v3full_S0f2_s200_o2000__verified.md,docs/tests/v3box_grades_2026-09-30/v3full_S0f3_s200_o3000__verified.md,docs/tests/v3box_grades_2026-09-30/v3full_S0f4_s200_o4000__verified.md
# 3. the second pairing: against FR_box_v1 (the same S0 floor draws)
for k in 1 2 3 4; do scripts/box_run.sh scripts/diag_pair_gate_reports.py \
  --a data/processed/models/full_retrain_v1/FR_box_v1/gate/fr1_FR_box_v1_full_s200_o0_ev4__verified.md \
  --b data/processed/models/full_retrain_v1/FT_box_v2/gate/fr1_FT_box_v2_full_s200_o0_ev4__verified.md \
  --label-a FR_box_v1 --label-b FT_box_v2 \
  --noise docs/tests/v3box_grades_2026-09-30/v3full_S0f${k}_s200_o${k}000__verified.md \
  --out data/processed/models/full_retrain_v1/FT_box_v2/gate/pair_FR_box_v1_vs_FT_box_v2_n${k}.md; done
# 4. if the operator's Decision-12 tool is at hand: scripts/ops_pair_bootstrap_v1.py for fr1_FT_box_v2_full_s200_o0_ev4
#    against v3full_S0_s200_o0 AND against fr1_FR_box_v1_full_s200_o0_ev4 (as for laneD_2)
```
- **Hard stop:** the chain FAILS at the `parity` stage, before the gate, if the trained and served fg_make shooter
  feature disagree (corr < 0.999 or p99 |diff| > 1e-3). If that happens, report `FT_box_v2/parity/parity_fg_rb.json`
  and do not run the gate.
- **Reused unchanged from FT_box_v1 / FR_box_v1:** possessions_v4, rotation, all designs, foul_state, clock,
  inputs_base, po_train (F_T) and rb_train (F_T). **Rerun:** fg_train, inputs, parity, gate.

## Expected outputs (sync back to the same local paths)
- `results/engine_v0/fr1_FT_box_v2_full_s200_o0_ev4/` (the `results` key)
- `data/processed/models/full_retrain_v1/FT_box_v2/`: `gate/` (`*__verified.md`, `pair_ref_vs_FT_box_v2_n{0..3}.md`,
  `pair_FR_box_v1_vs_FT_box_v2_n{1..4}.md`), `parity/parity_fg_rb.json`, `fg_train/`, `inputs/`, and every
  `.done.json` and `stage.log`
- The bootstrap tables, if step 4 ran.

## The one line to decide
None. Run it as soon as possible: it is the F_T ship-decision read.
