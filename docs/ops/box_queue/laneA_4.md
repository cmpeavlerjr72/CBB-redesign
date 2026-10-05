# Box request laneA_4: closed loop of the skew-free fg_make E3 arm Tfix (lane A, filed 2026-09-30 23:19 EDT)

**Commit:** `acad375800aaa6db0e98241575ff60689d835811` (origin/main). Registered in `docs/models/aggregation/experiments.md` section 2.5 (addendum G).

**Why:** Stage B's fg_make `T` was trained with `shooter_shrunk_dev_c` built from the STALE served `off_make_raw` (the `--team-rate-table` path does not re-derive raw), but is served a dev built from the E3 rate. That train/serve skew carries most of fg_make's G9 damage in the harness. `Tfix` removes the skew: harness 1 - slope is X_Tfix +0.012 / +0.004 (seeds 0 / 1) vs S0, against X_F's +0.035; S1fix is +0.021 vs S1's +0.043. These three reads say whether that holds in the sim. **This request outranks laneA_1.**

**Artifacts (already on HF, `model_artifacts` key):** `model_artifacts/fg_make/round_aggfix/Tfix_seed{0,1}/team_rate_features_E3_v4/{B1/*, m_fitted.json}` -> local path `data/processed/models/fg_make/round_aggfix/Tfix_seed{0,1}/team_rate_features_E3_v4/`.
- Alternative: retrain on the box (about 2 min each, deterministic; the local retrain of `T` reproduced the box `T` exactly):
  `python scripts/train_fg_make_v4_par_rawfix_v1.py --mode run --arms B1 --seed 0 --no-floor --no-leak --n-jobs 18 --team-rate-table data/processed/team_rate_features_E3_v4.parquet --team-rate-missing raise --extra-cache data/processed/models/fg_make/round_aggfix/design_v4_extra_E3_rawfix_s0.parquet --out-dir data/processed/models/fg_make/round_aggfix/Tfix_seed0` (seed 1: `--seed 1`, cache `..._s1.parquet`, out-dir `Tfix_seed1`).
- Plus everything laneA_3 used: E3 v4 table and the Stage B `T` PO / rebound artifacts.

**Env:** `PYTHONIOENCODING=utf-8`, `CBB_TRUTH=verified_v1`; served flags; no mounts (in-process overlay).

**Commands (priority order):**
```
FG=data/processed/models/fg_make/round_aggfix; S=team_rate_features_E3_v4; TAB="--team-rate-table data/processed/$S.parquet --team-rate-missing raise"
python scripts/build_engine_inputs_v3_tag_v1.py --tag X_Tfix_laneA $TAB --no-table-for po,rb --fg-artifacts $FG/Tfix_seed0/$S/B1
python scripts/build_engine_inputs_v3_tag_v1.py --tag S1fix_laneA $TAB --po-artifacts data/processed/models/possession_outcome/round_stageb/T/$S --fg-artifacts $FG/Tfix_seed0/$S/B1 --rb-artifacts data/processed/models/rebound/round_stageb/T/$S/artifacts/s2_F2_A0B0C0_seed0
python scripts/build_engine_inputs_v3_tag_v1.py --tag X_Tfix1_laneA $TAB --no-table-for po,rb --fg-artifacts $FG/Tfix_seed1/$S/B1
python scripts/exp_aggregation_swap_v1.py --stack X_Tfix --arm FULL --all-games --seeds 200 --workers 90 --games-per-block 16
python scripts/exp_aggregation_swap_v1.py --stack S1fix --arm FULL --all-games --seeds 200 --workers 90 --games-per-block 16
python scripts/exp_aggregation_swap_v1.py --stack X_Tfix1 --arm FULL --all-games --seeds 200 --workers 90 --games-per-block 16
```
About 6 min per read on 90 workers (laneA_3 measured 5 min 47 s and 6 min 05 s).

**Outputs to sync back:** `results/aggregation_v1/{X_Tfix,S1fix,X_Tfix1}_FULL_s200_o0/` (`games.parquet`, `run_meta.json`).

**Decision needed:** none. If only one read fits, run X_Tfix.
