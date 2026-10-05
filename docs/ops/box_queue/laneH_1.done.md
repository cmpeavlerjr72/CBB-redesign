# laneH_1 DONE, tier 1 (A2 at 200 seeds); tier 2 status at the bottom (AWS operator, 2026-10-01)

- Box: on-demand c7a.48xlarge (i-0393f93baca3f0f66). Clone `~/cbb5` at `759be01`. Engine flag-off parity vs `parity_reference_windows_v6.json`: PASS bit-identical (03:08Z). S0 tag built in that clone.
- **Same command as `v3full_L2_s200_o0`, only `ENGINE_CLOCK` and the tag changed.** That was NOT the laneB-style loop. L2 was produced by `scripts/box_fullread_v2.sh run L2 96 0 200 25`, which is `scripts/box_run_v2.sh scripts/run_engine.py --fold F2 --season 2025 --seeds 25 --seed-offset <o> --workers <W> --games-per-block 60 --seeds-per-block 25 --tag <sub> --results-dir results/engine_v0 --input-dir data/processed/models/engine_v3_S0` with the S0 tag's overlay mounts (`docker_mounts.txt`) and `ENGINE_EVENT=round2_s1 ENGINE_ROTATION=reference ENGINE_FG3=decision8`. A2 ran exactly that, with `ENGINE_CLOCK=v5b_r7A2_glat_pmean`, at 90 workers (L2 used 96; worker count does not change results).
- Chunks 03:08:29Z -> 03:22:44Z + concat, job end 03:24:50Z. That is about 2 min 05 s per 25-seed chunk on 90 workers while another 90-worker job shared the box; L2 took about 1.5 min per chunk at 96 workers under similar load.
- Check: `run_meta.json` `adapter_flags.ENGINE_CLOCK` = `v5b_r7A2_glat_pmean`.
- Synced (scp): `results/engine_v0/laneH_v3full_A2_s200_o0/{games.parquet,run_meta.json}`. Not graded on the box.

Tier 2 DONE: A2 floor draw, seeds 1000-1199, same command with offsets 1000-1175, tag `laneH_v3full_A2f1_s200_o1000`. It ran 04:20:21Z -> 04:35:26Z on the same `~/cbb5` clone, and `run_meta` `ENGINE_CLOCK` = `v5b_r7A2_glat_pmean`. Synced (scp): `results/engine_v0/laneH_v3full_A2f1_s200_o1000/{games.parquet,run_meta.json}`.
