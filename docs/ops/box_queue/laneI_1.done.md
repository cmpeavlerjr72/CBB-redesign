# laneI_1 DONE, tier A (COMB9 + K at 200 seeds, graded, both pairings); tier B status at the bottom (AWS operator, 2026-10-01)

- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66). Clone `~/cbb8` at `38eafdc`. Engine flag-off parity vs v6: PASS bit-identical (04:36Z). S0 tag built in that clone. The `v3full_S0` / S0f1..f4 / `v3full_COMB9` reads were copied in.
- `bash scripts/box_chance_time_v1.sh COMB9 K 90 0 200`: 04:36Z -> about 04:51Z. The script printed `[env] COMB9+K ENGINE_CLOCK=v5b_r6L2_glat_pmean ENGINE_SHOT_BLOCK=K2_Ocell ENGINE_FOUL_JOINT=R9ao3 ENGINE_CHANCE_TIME=K`.
- Flag check: the run differs from `v3full_COMB9_s200_o0` in 91.1% of the 1,142,000 (game, seed) rows on `home_pts`. Mean total points per game is +0.929 vs COMB9 (your local 500 x 32 estimate was +0.93).
- `eval_gates.py` (verified) and both `ops_pair_bootstrap_v1.py` commands ran as written; the job ended 04:53:09Z.
  - In the COMB9-referenced table, the draw SD is over [COMB9, S0f1..S0f4], i.e. the ref mixed with S0 reruns. As you said, recompute it locally from the S0 draws.
- Synced (scp), to the same paths:
  - `results/engine_v0/v3full_COMB9CTK_s200_o0/{games.parquet,run_meta.json}`;
  - `results/engine_v0/v3full_grade/v3full_COMB9CTK_s200_o0__verified.md`;
  - `results/ppp_decomp/full/pair_COMB9K_vs_{COMB9,S0}.{md,json}`.

Tier B DONE (C12): `bash scripts/box_chance_time_v1.sh COMB9 C12 90 0 200` ran 04:50:05Z -> 05:05:48Z, with the same clone, grade and both pairings (`pair_COMB9C12_vs_{COMB9,S0}`). Synced (scp): `results/engine_v0/v3full_COMB9CT12_s200_o0/{games.parquet,run_meta.json}`, `v3full_grade/v3full_COMB9CT12_s200_o0__verified.md`, `results/ppp_decomp/full/pair_COMB9C12_*`.
