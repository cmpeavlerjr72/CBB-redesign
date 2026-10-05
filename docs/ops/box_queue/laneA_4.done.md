# laneA_4 DONE (all three reads) (AWS operator, 2026-10-01)

- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66). Clone `~/cbb7` at `acad375`. Engine flag-off parity vs v6: PASS bit-identical (03:57Z).
- Artifacts: `hf_sync_data.py pull --dirs model_artifacts --only 'fg_make/round_aggfix/Tfix_seed*/**'`, rc=0. No retrain on the box.
- Builds: all three ran as written, rc=0. Builder hashes (first 12 hex):

| tag | arrays | event block |
|---|---|---|
| `X_Tfix_laneA` | `f2e09929d3f0` | `0e4c9568d0e5` |
| `S1fix_laneA` | `46ce9aeb240c` | `c0217cb73284` |
| `X_Tfix1_laneA` | `f2e09929d3f0` | `0e4c9568d0e5` |

  X_Tfix and X_Tfix1 have identical arrays: only the fg artifacts differ, and those are served in-process.
- Reads (200 seeds, 90 workers), in your priority order:
  - X_Tfix: 03:57:40Z -> 04:04:35Z
  - S1fix: 04:04:35Z -> 04:12:04Z
  - X_Tfix1: 04:12:04Z -> 04:19:23Z
- Synced (scp): `results/aggregation_v1/{X_Tfix,S1fix,X_Tfix1}_FULL_s200_o0/{games.parquet,run_meta.json}`.
