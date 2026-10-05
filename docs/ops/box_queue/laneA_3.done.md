# laneA_3 DONE (AWS operator, 2026-10-01)

- Ran on the on-demand c7a.48xlarge (i-0393f93baca3f0f66), clone `~/cbb4` at `7d38de5` (contains `30cf4fb`; engine code identical, parity vs `parity_reference_windows_v6.json` PASS bit-identical on this clone 02:03Z).
- Env as requested (`PYTHONIOENCODING=utf-8`, `CBB_TRUTH=verified_v1`, served flags, no mounts), inside the `cbb-sweep` image with the clone bind-mounted at `/app`.
- Builds (all rc=0, 02:03Z): `X_F_laneA`, `X_PR_laneA` (and, for laneA_1/laneA_2 later, `S0_laneA`, `S1_laneA`, `R2_laneA`).
- Reads (90 workers, 200 seeds, all 5,710 games):
  - `X_F FULL`: 02:03:53Z -> 02:09:40Z (5 min 47 s)
  - `X_PR FULL`: 02:09:40Z -> 02:15:45Z (6 min 05 s)
- Synced to the local paths (scp): `results/aggregation_v1/X_F_FULL_s200_o0/{games.parquet,run_meta.json}`, `results/aggregation_v1/X_PR_FULL_s200_o0/{games.parquet,run_meta.json}`. Not pushed to HF.

Builder `outputs_sha256` on the box (first 12 hex):

| tag | arrays_F2_2025.npz | event_block_F2_2025.npz |
|---|---|---|
| X_F_laneA | e0317445ef99 | 0e4c9568d0e5 |
| X_PR_laneA | b1e9d915b94d | c0217cb73284 |
| S0_laneA | e914337b0973 (matches your local) | 228794fc0d93 (matches) |
| S1_laneA | **32f6a992a7bb (your local: 556b66547f5c) DIFFERS** | **c0217cb73284 (your local: 21ef52ea6f86) DIFFERS** |
| R2_laneA | e914337b0973 (same as S0_laneA) | 228794fc0d93 (same as S0_laneA) |

The input tables are identical on both sides (sha256 prefixes: E3 v4 `f7adbf34ef26`, O1a v3 `7c58eb61a013`, local = box), so the difference sits in the Stage B `T` artifacts pulled from HF or in the builder environment. The S1_laneA difference is reported, not investigated further (box inputs: HF `team_rate_tables` E3 v4 and the variance table `team_rate_variance_O1a_v3.parquet` scp'd from the local tree). X_PR's event block equals the box S1_laneA's, so X_PR inherits whatever S1_laneA differs by.
