# laneB_2 DONE (COMB9G at 200 seeds, graded, both pairings) (AWS operator, 2026-10-01)

- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66). Clone `~/cbb6` at `7840d40` (contains `daa759a`). Engine flag-off parity vs v6: PASS bit-identical (03:16Z).
- laneC_2 tier A (COMB9) ran first on the other worker: 03:15:41Z -> 03:33:51Z.
- COMB9G: your loop exactly, run inside the image with every flag passed by `-e`. The flags were `ENGINE_CLOCK=v5b_r6L2_glat_pmean ENGINE_SHOT_BLOCK=K2_Ocell ENGINE_FOUL_JOINT=R9ao3 ENGINE_SHARED_SHOOTING=G3` plus `ENGINE_EVENT=round2_s1 ENGINE_ROTATION=reference ENGINE_FG3=decision8 CBB_TRUTH=verified_v1 PYTHONIOENCODING=utf-8`, with `scripts/run_engine_v3evb_v1.py ... --workers 90 --input-dir data/processed/models/engine_v3`. Chunks 03:24:50Z -> 03:40:35Z (about 2 min each), then concat and `eval_gates.py` (verified).
- Env check: on seeds 0-24, `home_pts` differs from `v3full_COMB9_s200_o0` in 80.3% of the 142,750 rows.
- Pairings: both `ops_pair_bootstrap_v1.py` commands ran as written, after COMB9 finished.
  - The second (`--ref v3full_COMB9_s200_o0`) takes the draw SD over [COMB9, S0f1..S0f4]. That mixes the COMB9 ref with S0 reruns, so recompute that floor locally if you need it clean.
- Synced (scp), to the same paths:
  - `results/engine_v0/v3full_COMB9G_s200_o0/{games.parquet,run_meta.json}`;
  - `results/engine_v0/v3full_grade/v3full_COMB9G_s200_o0__verified.md`;
  - `results/shared_shooting/full/pair_COMB9G_vs_{S0,COMB9}.{md,json}`;
  - `results/engine_v0/v3full_COMB9_s200_o0/{games.parquet,run_meta.json}` (from laneC_2).
- U1 (laneB_1 tier 2): a first attempt started at 03:14:51Z. I stopped it at 03:15:40Z (my own job; no chunk had finished, and the partial dir was removed) to let the ship-decision reads go first. It is re-queued at LOW priority. See section 19 of `docs/ops/aws_launch_chain.md`; if it runs, a line is added to `laneB_1.done.md`.
