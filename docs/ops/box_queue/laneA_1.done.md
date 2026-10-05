# laneA_1 DONE, tier 1 (16 runs); tier 2 status at the bottom (AWS operator, 2026-10-01)

- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66). Clone `~/cbb4` at `7d38de5` (contains `f14886f`), engine parity v6 PASS bit-identical.
- Step 1 builds at 02:03Z, rc=0. Builder `outputs_sha256`:

| tag | arrays | event block | vs your local |
|---|---|---|---|
| `S0_laneA` | `e914337b0973` | `228794fc0d93` | match |
| `S1_laneA` | `32f6a992a7bb` | `c0217cb73284` | **differ** (yours: `556b66547f5c`, `21ef52ea6f86`) |

  Ran anyway, as you instructed. The input tables are byte-identical on the box and locally (E3 v4 `f7adbf34ef26`, O1a v3 `7c58eb61a013`), so the difference comes from the Stage B `T` artifacts pulled from HF or from the builder environment.
- Tier 1: `exp_aggregation_swap_v1.py --stack {S0,S1} --arm {TEAM,OFF,DEF,ALL,PO,FG,RB,RAT} --all-games --seeds 48 --workers 90 --games-per-block 16`.
  - Run order: all S0 arms first (02:39:28Z -> 02:56:20Z), then all S1 arms (02:56:20Z -> 03:14:51Z).
  - About 2.1 min per run. No failures.
  - laneA_3 (X_F, X_PR) and laneA_2 (R2) ran ahead of or between these, per your priority note.
- Synced (scp) for all 16: `results/aggregation_v1/{S0,S1}_{TEAM,OFF,DEF,ALL,PO,FG,RB,RAT}_s48_o0/{games.parquet,run_meta.json}`. `_adapter_dirs/` was not synced.

Tier 2 DONE: `{S0,S1} x {RAT_PO, PACE, PLY}` at 48 seeds, 04:03:54Z -> 04:20:21Z (after laneC_2 / laneB_2 / laneA_4), no failures. Synced (scp) `results/aggregation_v1/{S0,S1}_{RAT_PO,PACE,PLY}_s48_o0/{games.parquet,run_meta.json}`.
