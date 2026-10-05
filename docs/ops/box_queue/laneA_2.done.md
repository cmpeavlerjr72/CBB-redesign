# laneA_2 DONE (AWS operator, 2026-10-01)

- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66). Clone `~/cbb4` at `7d38de5` (contains `3f25bac`), engine parity v6 PASS bit-identical. `R2_laneA` was built at 02:03Z, rc=0. Builder hashes: arrays `e914337b0973`, event block `228794fc0d93`, both identical to `S0_laneA`'s (see `laneA_3.done.md`).
- `exp_aggregation_swap_v1.py --stack R2 --arm FULL --all-games --seeds 200 --workers 90 --games-per-block 16`: 03:00:20Z -> 03:07:29Z (7 min 09 s).
- Synced (scp): `results/aggregation_v1/R2_FULL_s200_o0/{games.parquet,run_meta.json}`.
