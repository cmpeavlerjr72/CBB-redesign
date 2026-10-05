# pm_1 DONE: COMB9GKD at full size (AWS operator, 2026-10-01)

**Run setup**
- Request received 01:50 EDT. Read started 05:51:21Z (01:51 EDT), ahead of everything still queued. To free the cores, I stopped the operator's own section-18 S1K2O read (5 of 8 chunks were done; the partial chunk was deleted) and set `STOP_NEW`, so the remaining section-18 reads did not start.
- Clone `~/cbb10` at `6edbe94`. Engine flag-off parity vs v6: PASS bit-identical (05:21Z).
- Command: a local copy of `scripts/box_chance_time_v1.sh` (box only, not committed). It differs from the original in three places: `-e ENGINE_SHARED_SHOOTING` is added to the docker args, `G` is inserted in the tag, and the env echo is extended. Invocation: `ENGINE_SHARED_SHOOTING=G3 bash scripts/box_chance_time_pm1_local.sh COMB9 KD 150 0 200`, 150 workers.

**Flags and check**
- `[env] COMB9+G+KD ENGINE_SHARED_SHOOTING=G3 ENGINE_CLOCK=v5b_r6L2_glat_pmean ENGINE_SHOT_BLOCK=K2_Ocell ENGINE_FOUL_JOINT=R9ao3 ENGINE_CHANCE_TIME=KD`, plus `ENGINE_EVENT=round2_s1 ENGINE_ROTATION=reference ENGINE_FG3=decision8 CBB_TRUTH=verified_v1`. Inputs: the S0 tag of `engine_v3`, exactly as for laneI_2.
- Row-difference check: rows differ from `v3full_COMB9CTKD_s200_o0` in 80.2% of 1,142,000 (game, seed) rows on `home_pts`.

**Timing:** run 05:51:21Z -> 06:01:24Z. `eval_gates.py` (verified and legacy) done 06:01:44Z; both bootstraps done 06:01:57Z (02:02 EDT).

**Pairings**
- `ops_pair_bootstrap_v1.py` vs `v3full_S0_s200_o0` with the four S0 floor draws. Arms COMB9GKD, COMB9G, COMB9KD and COMB9 are side by side.
- vs `v3full_COMB9G_s200_o0`. In this pairing the draw SD mixes in the COMB9G reference, so only its bootstrap CI column is a valid floor.

**Synced locally**
- `results/engine_v0/v3full_COMB9GCTKD_s200_o0/{games.parquet,run_meta.json,players.parquet}` (players kept for G8);
- `results/engine_v0/v3full_grade/v3full_COMB9GCTKD_s200_o0__{verified,current}.md`;
- `results/pm1/pair_COMB9GKD_vs_{S0,COMB9G}.{md,json}`.

HF push of `results` started from the clone at 06:02Z.

**Gate doc:** `docs/tests/engine_gates_F2_2025_s200_v3_COMB9GKD_full_2026-10-01.md`.

**Lines vs S0, verified truth: move (Decision 12 floor), verdict ref -> arm**
- G1 possessions mean 69.775 -> 68.823, -0.952 (0.010), **FAIL -> PASS**. G1 by month: 0/5 -> 4/5 powered months inside (still FAIL).
- G2: 3/9 -> 6/9 cells inside (still FAIL).
- G4:
  - OREB% 0.2829 -> 0.2887, **FAIL -> PASS**;
  - eFG% 0.5014 -> 0.5080, +0.0066 (0.0001), PASS -> PASS;
  - FTA/FGA 0.3169 -> 0.3271, PASS -> PASS;
  - TOV count per team-game -0.222 (0.003).
- G5:
  - margin SD ratio 1.0500 -> 1.0419, -0.0080 (0.0035), PASS -> PASS;
  - total SD ratio 0.9306 -> 0.9254, -0.0052 (0.0018), FAIL -> FAIL. Its within-game SD component moved -0.089 (0.021); the residual SD is inside the floor;
  - home/away corr 0.1174 -> 0.1260, +0.0086 (0.0021), FAIL -> FAIL;
  - PIT p 0.018 -> 0.002, FAIL -> FAIL.
- G9:
  - margin bias -0.052 -> -0.261, -0.209 (0.042), PASS -> PASS;
  - total bias -0.258 -> -0.337, -0.080 (0.029), PASS -> PASS;
  - calibration slope 0.917 -> 0.948, +0.031 (0.005), FAIL -> FAIL.
- G6 non-neutral: 5.852 -> 5.626, PASS -> PASS.
- OT rate: inside the floor.
