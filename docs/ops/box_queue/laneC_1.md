**AMENDED 2026-09-30 23:20 EDT: Tier 2 (full R9ao1 / R9ao3) is CANCELLED, superseded by laneC_2.md (COMB9, SHIP-DECISION).**

# Box request laneC_1: foul round 9 (and-one model) paired reads vs R8b and S0 (lane C, 2026-09-30 ~21:45 EDT)

**Commit:** `222a570b7e05994cc44637cb2fdfb4e99a8ae98d` (pushed to origin/main) or any later main. Pre-registration:
`docs/models/possession_outcome/experiments.md` section 26 (commit `265e6fd`). Engine wiring `525fabd`
(default-off `ENGINE_FOUL_JOINT=R9ao1|R9ao3`), LUTs force-added in `bd03ea9`
(`data/processed/models/possession_outcome/round9/lut_ao_AO{1,3}_F2.npz`, `ao_team_prior_v1.parquet`, tracked in git,
so a `git pull` brings them).

**Inputs on the box:** exactly those of tonight's `R8b` reads: `data/processed/models/engine_v3_S0/` (+ its
`docker_mounts.txt`), served artifacts, round-7 LUTs `data/processed/models/possession_outcome/round7/lut_acc_A2_F2.npz`
and `lut_trip_T2c_F2.npz` (R8b needs them already), `data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet`.

**Env:** set by the script (`CBB_TRUTH=verified_v1`, `ENGINE_FOUL_JOINT=<ARM>` except for S0). `chmod +x scripts/*.sh` after the pull.

**Tier 1 (sample, 500 verified games x 200 seeds; about 2.5-3 min each on 90 workers):**
```
for A in R9ao1 R9ao3; do bash scripts/box_r9_v1.sh sample $A 90 0; done      # gates, paired with v3box_{S0,R8b}_s200_o0
for O in 3000 4000;  do bash scripts/box_r9_v1.sh sample S0 90 $O; done       # the two lost S0 floor draws (S0f3, S0f4)
for A in S0 R8b R9ao1 R9ao3; do bash scripts/box_r9_v1.sh tap $A 90 0; done   # half lines (H1/H2 FTA/FGA), tapped
for O in 1000 2000 3000 4000; do bash scripts/box_r9_v1.sh tap S0 90 $O; done # floor draws for the half lines
```
**Tier 2 (full size 5,710 x 200, about 15 min each), only after tier 1:**
```
bash scripts/box_r9_v1.sh full R9ao1 90 0
bash scripts/box_r9_v1.sh full R9ao3 90 0
```
(paired against your full-size `v3full_R8b_s200_o0` and `v3full_S0_s200_o0` and the S0 floor draws f1..f4).

**Outputs to sync back (same local paths):**
- `results/engine_v0/v3box_{R9ao1,R9ao3}_s200_o0/`, `results/engine_v0/v3box_S0_s200_o{3000,4000}/` (games.parquet,
  run_meta.json; players.parquet optional), `results/engine_v0/v3box_grade/*.md` for those tags;
- `results/engine_v0/r9taph_{S0,R8b,R9ao1,R9ao3}_s200_o0/` and `r9taph_S0_s200_o{1000,2000,3000,4000}/`
  (`half_agg.parquet` ~5 MB, games*.parquet, run_meta.json);
- tier 2: `results/engine_v0/v3full_{R9ao1,R9ao3}_s200_o0/` (games.parquet, run_meta.json) and
  `results/engine_v0/v3full_grade/v3full_{R9ao1,R9ao3}_s200_o0__verified.md`; if not already local, also
  `v3full_{S0,R8b}_s200_o0` games + grades and the S0 full floor draws.

**Parity:** local runs of S0 and R8b on this code are bit-identical to `v3box_{S0,R8b}_s200_o0` on the shared rows
(26 columns; `scripts/diag_foul_r9_parity_v1.py`). A quick box check if wanted: the `sample R8b` rows of seeds 0-199
must equal `v3box_R8b_s200_o0`.

**Decision needed:** none; run as written. If time is short, tier 1 lines 1 and 3 alone (6 runs) are the decisive minimum.
