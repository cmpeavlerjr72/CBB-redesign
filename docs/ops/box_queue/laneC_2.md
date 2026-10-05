# Box request laneC_2 -- **SHIP-DECISION priority**: full-size COMB9 read (lane C, 2026-09-30 ~23:20 EDT)

PM ruling 23:12 EDT: `R9ao3` supersedes `R8b` in the Decision 11 set. This request REPLACES laneC_1 tier 2:
**drop the standalone full-size R9ao1 read** (and R9ao3 standalone; COMB9 is the read that matters).

**Commit:** `7840d4057b45e6c01299fc6957bf214edaf3772c` (origin/main) or later. LUTs `data/processed/models/possession_outcome/round9/lut_ao_AO3_F2.npz` + `ao_team_prior_v1.parquet` are tracked (git pull brings them); round-7 LUTs `round7/lut_acc_A2_F2.npz`, `lut_trip_T2c_F2.npz` as for R8b/COMB; clock round-6 L2 artifacts as for COMB. Inputs: `engine_v3_S0` + `docker_mounts.txt`. `chmod +x scripts/*.sh`.

**Arm COMB9** = `ENGINE_CLOCK=v5b_r6L2_glat_pmean` + `ENGINE_SHOT_BLOCK=K2_Ocell` + `ENGINE_FOUL_JOINT=R9ao3` on the S0 tag (the script sets all three; `[env]` line printed). Identical to the operator's COMB except the foul member.

**Tier A (decisive; ~15 min):** full size 5,710 x 200, seeds 0-199, chunked, resumable, graded under verified and legacy truth:
```
bash scripts/box_r9_v1.sh full COMB9 90 0 200
```
-> `results/engine_v0/v3full_COMB9_s200_o0/`, `results/engine_v0/v3full_grade/v3full_COMB9_s200_o0__{verified,current}.md`.
Then the Decision 12 pairing (four S0 floor draws + paired game bootstrap), against S0 AND against COMB:
```
CBB_TRUTH=verified_v1 python scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_S0_s200_o0 \
  --floors v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000 \
  --arms v3full_COMB9_s200_o0,v3full_COMB_s200_o0 --out-json results/foul_r9/full/pair_COMB9_vs_S0.json --out-md results/foul_r9/full/pair_COMB9_vs_S0.md
CBB_TRUTH=verified_v1 python scripts/ops_pair_bootstrap_v1.py --results-dir results/engine_v0 --ref v3full_COMB_s200_o0 \
  --floors v3full_S0f1_s200_o1000,v3full_S0f2_s200_o2000,v3full_S0f3_s200_o3000,v3full_S0f4_s200_o4000 \
  --arms v3full_COMB9_s200_o0 --out-json results/foul_r9/full/pair_COMB9_vs_COMB.json --out-md results/foul_r9/full/pair_COMB9_vs_COMB.md
```
(If `ops_pair_bootstrap_v1.py` takes the draw SD over [ref, floors] with ref = COMB, note it in the .done file: lane C will recompute the COMB-referenced floor from the S0 draws locally.)

**Tier B (FTA/FGA by half; ~15 min each, tapped, half aggregates only, no players):**
```
bash scripts/box_r9_v1.sh tapfull COMB9 90 0 200
bash scripts/box_r9_v1.sh tapfull COMB 90 0 200
bash scripts/box_r9_v1.sh tapfull S0 90 0 200
```
-> `results/engine_v0/r9tapfull_{COMB9,COMB,S0}_s200_o0_off{0,25,...,175}_n25/` (half_agg.parquet ~0.1 GB total per arm, games.parquet, games_v2.parquet, run_meta.json). The tapped games must be bit-identical to the untapped `v3full_*` games (lane C checks locally with `scripts/diag_foul_r9_parity_v1.py`).
Optional Tier C (half-line draw SD): `bash scripts/box_r9_v1.sh tapfull S0 90 <O> 200` for O = 1000, 2000, 3000, 4000.

**Sync back (same paths):** `results/engine_v0/v3full_COMB9_s200_o0/` (games.parquet, run_meta.json; players optional), its two grade md files, `results/foul_r9/full/*`, all `r9tapfull_*` chunk dirs; and, if not already local, `v3full_{S0,COMB}_s200_o0` + `v3full_S0f{1..4}_*` games and their `__verified.md` grades.

**Lane C grades afterwards:** half lines `python scripts/grade_foul_r9_closed_loop_v1.py results/foul_r9/full/half --ref r9tapfull_COMB_s200_o0 --arms r9tapfull_COMB9_s200_o0 --extra r9tapfull_S0_s200_o0 --floors r9tapfull_S0_s200_o0[,..o1000..o4000]`; gate deltas `scripts/diag_foul_r9_gate_delta_v1.py`.

**Decision needed:** none; run Tier A first, then B, C if time.
