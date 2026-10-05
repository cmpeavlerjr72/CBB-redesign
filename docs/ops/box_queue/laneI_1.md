# Box request laneI_1: COMB9 + chance-time feed `K` at full size, paired with COMB9 (lane I, 2026-10-01 00:20 EDT)

**Label: POST-HOC DIAGNOSTIC read for a PM ruling (not a registered SHIP-DECISION).** Arm `K` passes its pre-registered primary by 56.7 floors and F1 confirms, but it fails two registered guards (`docs/models/chance_time/experiments.md` s3). The PM rules on those guards; this read gives the full-size gate picture for that ruling. Priority: after any registered SHIP-DECISION requests.

**Commit:** `38eafdc` (origin/main) or later. Engine hook `ENGINE_CHANCE_TIME` (default off, flag-off parity PASS bit-identical vs `parity_reference_windows_v6.json`, digest `0d4ddccc`). Tables `data/processed/models/chance_time/F2/lut_v2.npz` are tracked (git pull brings them). Inputs and overlay exactly as for COMB9 (`engine_v3_S0` + `docker_mounts.txt`). `chmod +x scripts/*.sh`.

**What it is:** COMB9 (`ENGINE_CLOCK=v5b_r6L2_glat_pmean` + `ENGINE_SHOT_BLOCK=K2_Ocell` + `ENGINE_FOUL_JOINT=R9ao3`) plus `ENGINE_CHANCE_TIME=K`. `K` changes only the `chance_elapsed_s` / `is_transition_f` values fed to fg_make, drawn from training-season tables on a new rng family. `box_run_v2.sh` does not pass `ENGINE_CHANCE_TIME`, so the new script appends `-e ENGINE_CHANCE_TIME` to `BOX_DOCKER_ARGS`.

**Check that the flag reached the container:** the script prints `[env] COMB9+K ... ENGINE_CHANCE_TIME=K`. `run_meta.json` does NOT record this flag. If the flag got through, the games rows must DIFFER from `v3full_COMB9_s200_o0` on the same (game, seed). Local 500 x 32: points per game +0.93.

**Tier A (decisive, about 16 min at 90 workers, the same cost as COMB9):**
```
bash scripts/box_chance_time_v1.sh COMB9 K 90 0 200
CBB_TRUTH=verified_v1 python scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_COMB9_s200_o0 \
  --floors v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000 \
  --arms v3full_COMB9CTK_s200_o0 --out-json results/ppp_decomp/full/pair_COMB9K_vs_COMB9.json --out-md results/ppp_decomp/full/pair_COMB9K_vs_COMB9.md
CBB_TRUTH=verified_v1 python scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_S0_s200_o0 \
  --floors v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000 \
  --arms v3full_COMB9CTK_s200_o0,v3full_COMB9_s200_o0 --out-json results/ppp_decomp/full/pair_COMB9K_vs_S0.json --out-md results/ppp_decomp/full/pair_COMB9K_vs_S0.md
```
-> `results/engine_v0/v3full_COMB9CTK_s200_o0/` and `results/engine_v0/v3full_grade/v3full_COMB9CTK_s200_o0__verified.md`. (Draw SD with ref = COMB9 mixes the ref with S0 reruns, as in laneC_2; lane I recomputes it from the S0 draws locally.)

**Tier B (only if time allows):** the same with arm `C12` (`bash scripts/box_chance_time_v1.sh COMB9 C12 90 0 200`) -> `v3full_COMB9CT12_s200_o0`. This arm also changes possession_outcome's transition input.

**Sync back (same paths):** `results/engine_v0/v3full_COMB9CTK_s200_o0/{games.parquet,run_meta.json}` (players optional), its `__verified.md` grade, `results/ppp_decomp/full/*`; if run, the C12 equivalents.

**Decision needed:** none; run as written, at the operator's priority, after registered SHIP-DECISION requests. Write `laneI_1.done.md` (or `.failed.md`) next to this file.
