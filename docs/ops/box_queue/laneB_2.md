# Box request laneB_2 -- **SHIP-DECISION priority**: full-size COMB9G read (lane B, 2026-09-30 23:15 EDT)

**Commit:** `daa759a` (origin/main) or later (needs lane C's `7840d40`, already an ancestor). No new artifacts beyond COMB9's: G3 params `data/processed/models/shared_shooting/params_v1.json` are tracked.

**Arm COMB9G** = COMB9 (`ENGINE_CLOCK=v5b_r6L2_glat_pmean` + `ENGINE_SHOT_BLOCK=K2_Ocell` + `ENGINE_FOUL_JOINT=R9ao3`) **+ `ENGINE_SHARED_SHOOTING=G3`**. Paired against `v3full_S0_s200_o0` AND lane C's `v3full_COMB9_s200_o0` (laneC_2 tier A). **If COMB9 is not on the box yet, run laneC_2 tier A first**, then this.

**Composition checked locally (60 games x 5 seeds, v3 inputs):** default path parity v6 digest PASS at HEAD; G3 alone at HEAD bit-identical to the box's `laneB_v3full_G3_s200_o0` (300 rows); COMB9 and COMB9G both run (COMB9G differs from COMB9 in 79% of rows, as expected).

**How to run:** NOT via `box_r9_v1.sh` (it does not pass `ENGINE_SHARED_SHOOTING`). Use `scripts/run_engine_v3evb_v1.py`, which serves the S0 overlay's v3 event block in-process (no mounts needed; proved bit-identical to `v3full_S0` rows). Inside the image (pass every flag with `-e` if using docker):
```
export PYTHONIOENCODING=utf-8 CBB_TRUTH=verified_v1 ENGINE_EVENT=round2_s1 ENGINE_ROTATION=reference ENGINE_FG3=decision8 \
       ENGINE_CLOCK=v5b_r6L2_glat_pmean ENGINE_SHOT_BLOCK=K2_Ocell ENGINE_FOUL_JOINT=R9ao3 ENGINE_SHARED_SHOOTING=G3
for o in 0 25 50 75 100 125 150 175; do
  python scripts/run_engine_v3evb_v1.py --fold F2 --season 2025 --seeds 25 --seed-offset $o --workers 90 \
    --games-per-block 60 --seeds-per-block 25 --tag v3full_COMB9G_s200_o0_off${o}_n25 --results-dir results/engine_v0 \
    --input-dir data/processed/models/engine_v3
done
python scripts/concat_engine_runs.py --tag v3full_COMB9G_s200_o0 --results-dir results/engine_v0 --overwrite
```
Env check (run_meta does not record ENGINE_SHARED_SHOOTING): seeds 0-24 `home_pts` must differ from `v3full_COMB9_s200_o0` in most rows.

**Grade + Decision 12 pairing** (same tool lane C asked for):
```
CBB_TRUTH=verified_v1 python scripts/eval_gates.py --results results/engine_v0/v3full_COMB9G_s200_o0 --season 2025 --out results/engine_v0/v3full_grade/v3full_COMB9G_s200_o0__verified.md
CBB_TRUTH=verified_v1 python scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_S0_s200_o0 \
  --floors v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000 \
  --arms v3full_COMB9G_s200_o0,v3full_COMB9_s200_o0 --out-json results/shared_shooting/full/pair_COMB9G_vs_S0.json --out-md results/shared_shooting/full/pair_COMB9G_vs_S0.md
CBB_TRUTH=verified_v1 python scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_COMB9_s200_o0 \
  --floors v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000 \
  --arms v3full_COMB9G_s200_o0 --out-json results/shared_shooting/full/pair_COMB9G_vs_COMB9.json --out-md results/shared_shooting/full/pair_COMB9G_vs_COMB9.md
```
(G5 by component is computed locally by lane B with `scripts/grade_shared_shooting_loop_v1.py`; no need on the box.)

**LOW priority, only after everything else:** U1 control at 200 seeds = laneB_1 tier 2 (served stack, `ENGINE_SHARED_SHOOTING=U1`, tag `laneB_v3full_U1_s200_o0`, commands in `laneB_1.md`).

**Sync back (same paths):** `results/engine_v0/v3full_COMB9G_s200_o0/{games.parquet,run_meta.json}`, its `__verified.md`, `results/shared_shooting/full/*`; `v3full_COMB9_s200_o0/games.parquet` if not already synced for lane C; U1 dir if run.

**Decision needed:** none; run as written.
