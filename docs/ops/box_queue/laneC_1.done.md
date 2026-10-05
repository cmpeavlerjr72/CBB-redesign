# laneC_1 DONE, tier 1 (all 12 runs); tier 2 status at the bottom (AWS operator, 2026-10-01)

- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66). Clone `~/cbb4` at `7d38de5`, which contains `222a570`, `525fabd` and `bd03ea9`. Engine flag-off parity vs `parity_reference_windows_v6.json`: PASS bit-identical (02:03Z). The S0 tag (`engine_v3_S0` + `docker_mounts.txt`) was built in that clone with `build_engine_inputs_v3_tag_v1.py --tag S0`.
- Ran exactly as written: `bash scripts/box_r9_v1.sh {sample|tap} <ARM> 90 <OFF>` for the 12 tier-1 lines (sample R9ao1, R9ao3; tap S0, R8b, R9ao1, R9ao3; sample S0 at 3000 and 4000; tap S0 at 1000-4000). All exited 0, no traceback in any log. The other queue worker (lane A / lane D) shared the box.
- Timing: 02:31:24Z -> 03:00:20Z (12 runs, about 2.4 min each).
- No box parity check of `sample R8b` was run (optional; it was not in the tier-1 list).
- Synced locally (scp), to the same paths:
  - `results/engine_v0/v3box_{R9ao1,R9ao3}_s200_o0/` and `v3box_S0_s200_o{3000,4000}/` (games.parquet, run_meta.json; players.parquet left on the box);
  - `results/engine_v0/v3box_grade/v3box_{R9ao1,R9ao3}_s200_o0.md`, `v3box_S0_s200_o{3000,4000}.md`;
  - `results/engine_v0/r9taph_{S0,R8b,R9ao1,R9ao3}_s200_o0/` and `r9taph_S0_s200_o{1000,2000,3000,4000}/` (games.parquet, games_v2.parquet, half_agg.parquet, run_meta.json).

Tier 2 (full R9ao1, R9ao3): NOT RUN. It was cancelled by your 23:20 EDT amendment and superseded by laneC_2 (see `laneC_2.done.md`).
