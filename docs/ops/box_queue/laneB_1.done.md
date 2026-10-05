# laneB_1 DONE, tier 1 (G3); tier 2 (U1) status at the bottom (AWS operator, 2026-10-01)

- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66), clone `~/cbb4` at `7d38de5` (contains `0ac56fd`; engine parity vs `parity_reference_windows_v6.json` PASS bit-identical on this clone, 02:03Z).
- Run exactly as written, python inside the `cbb-sweep` image with `-e ENGINE_SHARED_SHOOTING=G3` passed explicitly plus `PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 ENGINE_EVENT=round2_s1 ENGINE_CLOCK=v5b_glat_pmean ENGINE_ROTATION=reference ENGINE_FG3=decision8`; `scripts/run_engine_v3evb_v1.py ... --workers 90 --input-dir data/processed/models/engine_v3`, 8 chunks of 25 seeds (offsets 0-175), then `concat_engine_runs.py`.
- Timing: 02:15:45Z -> 02:31:24Z (about 1 min 45 s per 25-seed chunk at 90 workers, sharing the box with lane D's retrain).
- Env check: the run_meta does not record the flag (as you warned), so I checked the output instead: on seeds 0-24, `home_pts` differs from `v3full_S0_s200_o0_off0_n25` in 80.7% of the 142,750 (game, seed) rows, so the flag was live.
- Synced (scp) to `results/engine_v0/laneB_v3full_G3_s200_o0/{games.parquet,run_meta.json}`. The 25-seed chunk dirs and players.parquet stay on the box (and partly on HF under `results`).
- Not graded on the box (per the request).

Tier 2 DONE (U1): ran on `~/cbb4` with the same loop and `-e ENGINE_SHARED_SHOOTING=U1`, 04:53:09Z -> 05:08:32Z. Two earlier attempts (03:14Z, 04:36Z) were stopped by the operator before any chunk finished, to let ship-decision reads go first. Their partial dirs were removed, so no chunk mixes runs. Env check: `home_pts` differs from `v3full_S0_s200_o0` in 68.9% of the 1,142,000 rows. Synced (scp): `results/engine_v0/laneB_v3full_U1_s200_o0/{games.parquet,run_meta.json}`.
