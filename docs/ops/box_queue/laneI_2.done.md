# laneI_2 DONE (COMB9 + KD at 200 seeds, graded, both pairings) (AWS operator, 2026-10-01)

- Accepted: filed 01:07 EDT, before the 01:30 note.
- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66). Clone `~/cbb10` at `6edbe94`. Engine flag-off parity vs v6: PASS bit-identical (05:21Z). S0 tag built in that clone. The `v3full_S0`, S0f1..f4, `v3full_COMB9` and `v3full_COMB9CTK` reads were copied in.
- `bash scripts/box_chance_time_v1.sh COMB9 KD 90 0 200`, then `eval_gates.py` (verified), then both `ops_pair_bootstrap_v1.py` commands exactly as written. Ran 05:20:45Z -> 05:38:46Z.
- Flag check:
  - `[env] COMB9+KD ENGINE_CLOCK=v5b_r6L2_glat_pmean ENGINE_SHOT_BLOCK=K2_Ocell ENGINE_FOUL_JOINT=R9ao3 ENGINE_CHANCE_TIME=KD`.
  - Rows differ from `v3full_COMB9_s200_o0` in 66.2% of the 1,142,000 rows.
  - Total points per game +1.141 vs COMB9 (your local estimate was +1.10).
- In the COMB9-referenced pairing, the draw SD mixes the ref with S0 reruns (as in laneI_1).
- Synced (scp), to the same paths:
  - `results/engine_v0/v3full_COMB9CTKD_s200_o0/{games.parquet,run_meta.json}`;
  - `results/engine_v0/v3full_grade/v3full_COMB9CTKD_s200_o0__verified.md`;
  - `results/ppp_decomp/full/pair_COMB9KD_vs_{S0,COMB9}.{md,json}`.
