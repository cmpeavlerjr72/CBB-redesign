# Box request laneB_1: shared shooting latent G3, full size 5,710 x 200, paired with v3full_S0 (lane B, 2026-09-30 21:05 EDT)

**Commit:** `0ac56fd` (pushed to origin/main; any later main commit is fine, the engine hook is unchanged after it). Pre-registration: `docs/models/shared_shooting/experiments.md` section 1.

**What it is:** `ENGINE_SHARED_SHOOTING=G3` = a per-game shared logit effect on fg_make (default-off engine hook in `loop.py`, module `src/cbb_sim/engine/shared_shooting.py`, params `data/processed/models/shared_shooting/params_v1.json`, tracked). Paired against the served S0 read `v3full_S0_s200_o0` (seeds 0-199, already done on this box).

**Inputs needed on the box:** `data/processed/models/engine_v3/` (HF key `engine_inputs_v3`) and the served artifacts, exactly as for S0. NO overlay mounts and NO tag build: `scripts/run_engine_v3evb_v1.py` wraps `run_engine.py` and serves `engine_v3/event_block_F2_2025.npz` in-process (the S0 overlay's only non-identical file). Proved locally: off path reproduces `v3full_S0_s200_o0_off0_n25` bit-for-bit (120 games x 25 seeds, 78,000 cells).

**Env:** `PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 ENGINE_EVENT=round2_s1 ENGINE_CLOCK=v5b_glat_pmean ENGINE_ROTATION=reference ENGINE_FG3=decision8` plus `ENGINE_SHARED_SHOOTING=<arm>`. If you run through docker / `box_run_v2.sh`, note it does NOT pass `ENGINE_SHARED_SHOOTING` through: add `-e ENGINE_SHARED_SHOOTING` to the `docker run` line (or run python directly in the container as below). Without it the run silently equals S0 (check: `run_meta.json` will not differ, so please confirm the env inside the container).

**Tier 1 (decisive), G3 at 200 seeds (about 12-16 min on 90 workers):**
```
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 ENGINE_EVENT=round2_s1 ENGINE_CLOCK=v5b_glat_pmean ENGINE_ROTATION=reference ENGINE_FG3=decision8
for o in 0 25 50 75 100 125 150 175; do
  ENGINE_SHARED_SHOOTING=G3 python scripts/run_engine_v3evb_v1.py --fold F2 --season 2025 --seeds 25 --seed-offset $o --workers 90 \
    --games-per-block 60 --seeds-per-block 25 --tag laneB_v3full_G3_s200_o0_off${o}_n25 --results-dir results/engine_v0 \
    --input-dir data/processed/models/engine_v3
done
python scripts/concat_engine_runs.py --tag laneB_v3full_G3_s200_o0 --results-dir results/engine_v0 --overwrite
```
**Tier 2 (only if time allows), the unshared control U1:** the same loop with `ENGINE_SHARED_SHOOTING=U1` and tag `laneB_v3full_U1_s200_o0` (concat likewise).

**Outputs to sync back (same local paths):** `results/engine_v0/laneB_v3full_G3_s200_o0/{games.parquet,run_meta.json}` (players.parquet optional) and, if run, `results/engine_v0/laneB_v3full_U1_s200_o0/{games.parquet,run_meta.json}`. Grading is done locally (no need to run eval_gates on the box).

**Decision needed:** none; run as written. Tier 1 alone is the decisive read. Write `laneB_1.done.md` (or `.failed.md`) next to this file.
