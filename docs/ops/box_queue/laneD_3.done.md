# laneD_3 DONE (FT_box_v2 with the skew-free fg trainer, full gate read) (AWS operator, 2026-10-01)

- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66). New clone `~/cbb9` at `e998b22`. Engine flag-off parity vs v6: PASS bit-identical (04:38Z). The laneD_2 outputs (`full_retrain_v1/FR_box_v1`, `FT_box_v1`, and every data file of the laneD_2 clone) were copied in.
- Commands as written (steps 1-4), 04:37:42Z -> 04:50:05Z.
  - Step 1: `cp -r FT_box_v1 FT_box_v2`.
  - Step 2: chain `--redo fg_train,inputs,gate`. Stage times: fg_train 44.5 s, inputs 3.6 s, parity 0.6 s, gate 537 s.
- **Parity stage PASSED**, so the gate ran. From `FT_box_v2/parity/parity_fg_rb.json`, the trained vs served shooter feature:

| class | corr | p99 abs diff |
|---|---|---|
| rim | 0.999917 | 2.6e-8 |
| jump2 | 0.999949 | 5.4e-9 |
| three | 0.999995 | (see the json) |

  Overall: 142,777 served cells, 91,258 matched.
- Step 3: the four `pair_FR_box_v1_vs_FT_box_v2_n{1..4}.md` were written.
- Step 4: `ops_pair_bootstrap_v1.py` ran twice:
  - `gate/boot_vs_S0.{md,json}`: arms FT_box_v2, FT_box_v1 and FR_box_v1 against `v3full_S0_s200_o0`, with the four S0 draws.
  - `gate/boot_vs_FR_box_v1.{md,json}`: FT_box_v2 against FR_box_v1. **Caveat:** in this pairing the draw SD is taken over [FR_box_v1, S0f1..S0f4]. FR differs from S0 by about 0.77 possessions and 1.5 points of total bias, so the "draw SD" there is not a floor. Example: its G1 possessions floor reads 0.69 against a true seed floor of about 0.01. Use only the bootstrap CI column of that table, or recompute the floor from the S0 draws.
- Results (operator's Decision 12 table vs S0, FT_box_v2, move (floor)):
  - Beyond the floor:
    - G1 possessions -0.771 (0.010);
    - G5 total SD ratio -0.0265 (0.0032); its within-game SD is beyond, its residual SD is inside;
    - G5 margin SD ratio -0.0078 (0.0062);
    - home/away corr -0.019 (0.004);
    - G9 total bias -1.492 (0.057);
    - TOV count -0.176 (0.017);
    - FTA/FGA +0.0111 (0.0006).
  - Inside the floor: G9 calibration slope +0.0075 (0.0105) and G9 margin bias -0.0003 (0.064).
- Synced:
  - scp: `results/engine_v0/fr1_FT_box_v2_full_s200_o0_ev4/{games.parquet,run_meta.json}`;
  - tar: `data/processed/models/full_retrain_v1/FT_box_v2/` `gate/` (all 14 files), `parity/`, every `.done.json`, `stage.log` and `*report*.json` under 5 MB;
  - HF: the full dir and results were pushed (`results`, `model_artifacts`; push rc=0).
