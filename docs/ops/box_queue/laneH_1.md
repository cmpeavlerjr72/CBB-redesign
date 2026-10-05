# Box request laneH_1: clock round 7 arm A2, full size 5,710 x 200, paired with v3full_L2_s200_o0 (lane H, 2026-09-30 22:42 EDT)

**Commit:** `759be01` (pushed to origin/main; any later main commit is fine). Pre-registration: `docs/models/clock/experiments.md` sections 32-33.

**What it is:** `ENGINE_CLOCK=v5b_r7A2_glat_pmean`, a new default-off clock mode in `src/cbb_sim/engine/clock_adapter_v3.py` (dict keys only). It uses the module `src/cbb_sim/models/clock_r7.py` and the artifacts `data/processed/models/clock/r7_A2/F2/*` + `r7_A2/v5b_bakeoff/v5b_bakeoff_report.json`. **These are TRACKED in git (5.7 MB)**, so a `git pull` to `759be01` or later brings everything. No HF key or scp needed. Flag-off parity 60 x 5 PASS locally (`0d4ddccc...029f`).

**Inputs on the box:** exactly those of your `v3full_L2_s200_o0` read (engine_v3 inputs, served artifacts). **Please run A2 with the SAME command that produced `v3full_L2_s200_o0`, changing ONLY `ENGINE_CLOCK` from `v5b_r6L2_glat_pmean` to `v5b_r7A2_glat_pmean` and the tag.** If that was the laneB_1-style loop, it is:

```
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 ENGINE_EVENT=round2_s1 ENGINE_CLOCK=v5b_r7A2_glat_pmean ENGINE_ROTATION=reference ENGINE_FG3=decision8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for o in 0 25 50 75 100 125 150 175; do
  python scripts/run_engine_v3evb_v1.py --fold F2 --season 2025 --seeds 25 --seed-offset $o --workers 90 \
    --games-per-block 60 --seeds-per-block 25 --tag laneH_v3full_A2_s200_o0_off${o}_n25 --results-dir results/engine_v0 \
    --input-dir data/processed/models/engine_v3
done
python scripts/concat_engine_runs.py --tag laneH_v3full_A2_s200_o0 --results-dir results/engine_v0 --overwrite
```

**Check:** `run_meta.json` `adapter_flags.ENGINE_CLOCK` must read `v5b_r7A2_glat_pmean`. If the docker wrapper drops ENGINE_CLOCK, the run silently equals L2/S0.

**Speed:** A2's pmf adds a per-row CDF rescale, so expect about 1.3-2x L2's time per chunk (locally, a 30-game smoke ran at 351 poss/s on 3 workers).

**Tier 2 (only if time allows):** an A2 floor draw at seeds 1000-1199. Use the same loop with `o` in 1000 1025 ... 1175 and tag `laneH_v3full_A2f1_s200_o1000`.

**Outputs to sync back (same local paths):** `results/engine_v0/laneH_v3full_A2_s200_o0/{games.parquet,run_meta.json}` (and the Tier-2 one if run). Grading is done locally.

**Decision needed:** none; run as written. Write `laneH_1.done.md` (or `.failed.md`) next to this file.
