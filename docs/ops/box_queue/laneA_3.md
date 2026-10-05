# Box request laneA_3: closed-loop confirmation that fg_make's E3 retrain owns the S1 slope drop (lane A, 2026-09-30 21:53 EDT)

**Commit:** `30cf4fb4440cc463034ae109a51c47789541293f` (origin/main). Registered in `docs/models/aggregation/experiments.md` section 1.11 (addendum D).

**Why:** the deterministic harness factorial puts +0.035 of S1's +0.043 harness slope loss on fg_make's Stage B `T` retrain alone (PO and rebound inside their retrain-seed floors, interaction +0.002). Two full reads test it in the sim. **This is the most useful of my three requests; if only one runs, run this one (X_F first).**

**Inputs:** as laneA_1 (engine_v3, E3 v4 table, Stage B `T` artifacts for possession_outcome, fg_make, rebound).

**Env:** `PYTHONIOENCODING=utf-8`, `CBB_TRUTH=verified_v1`; served flags; no docker mounts needed (overlay served in-process).

**Commands:**
```
python scripts/build_engine_inputs_v3_tag_v1.py --tag X_F_laneA --team-rate-table data/processed/team_rate_features_E3_v4.parquet --team-rate-missing raise --no-table-for po,rb --fg-artifacts data/processed/models/fg_make/round_stageb/T/team_rate_features_E3_v4/B1
python scripts/build_engine_inputs_v3_tag_v1.py --tag X_PR_laneA --team-rate-table data/processed/team_rate_features_E3_v4.parquet --team-rate-missing raise --no-table-for fg --po-artifacts data/processed/models/possession_outcome/round_stageb/T/team_rate_features_E3_v4 --rb-artifacts data/processed/models/rebound/round_stageb/T/team_rate_features_E3_v4/artifacts/s2_F2_A0B0C0_seed0
python scripts/exp_aggregation_swap_v1.py --stack X_F --arm FULL --all-games --seeds 200 --workers 90 --games-per-block 16
python scripts/exp_aggregation_swap_v1.py --stack X_PR --arm FULL --all-games --seeds 200 --workers 90 --games-per-block 16
```
Each read is one 200-seed full read (11-16 min on 90 workers).

**Outputs to sync back:** `results/aggregation_v1/X_F_FULL_s200_o0/` and `results/aggregation_v1/X_PR_FULL_s200_o0/` (`games.parquet`, `run_meta.json`).

**Priority:** ahead of laneA_1 and laneA_2. **Decision needed:** none.
