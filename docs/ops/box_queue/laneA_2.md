# Box request laneA_2: sim-level retrain-seed floor for the S1 vs S0 G9 slope (lane A, 2026-09-30 21:25 EDT)

**Commit:** `3f25baccb850717f21c1a4d623ad4252a71765d3` (origin/main). Registered in `docs/models/aggregation/experiments.md` section 1.8.

**Why:** the Stage B `R2` artifacts (served features, retrain seed 1) move the deterministic harness slope by -0.014 versus the served/`R` stack, about a third of the `T` (E3) effect. The box's S1 vs S0 "15 floor-SD" used Monte Carlo floors only. One full read of the `R2` stack gives the spec-identical-retrain floor at sim level.

**Inputs:** as laneA_1, plus the Stage B `R2_seed1` artifacts (HF `model_artifacts`): `data/processed/models/possession_outcome/round_stageb/R2_seed1/`, `data/processed/models/fg_make/round_stageb/R2_seed1/B1/` (+ `m_fitted.json`), `data/processed/models/rebound/round_stageb/R2_seed1/artifacts/s2_F2_A0B0C0_seed1/`.

**Env:** `PYTHONIOENCODING=utf-8`, `CBB_TRUTH=verified_v1`; served flags; no docker mounts needed (in-process overlay).

**Commands:**
```
python scripts/build_engine_inputs_v3_tag_v1.py --tag R2_laneA --po-artifacts data/processed/models/possession_outcome/round_stageb/R2_seed1 --fg-artifacts data/processed/models/fg_make/round_stageb/R2_seed1/B1 --rb-artifacts data/processed/models/rebound/round_stageb/R2_seed1/artifacts/s2_F2_A0B0C0_seed1
python scripts/exp_aggregation_swap_v1.py --stack R2 --arm FULL --all-games --seeds 200 --workers 90 --games-per-block 16
```
About one 200-seed full read (11-16 min on 90 workers).

**Output to sync back:** `results/aggregation_v1/R2_FULL_s200_o0/games.parquet` + `run_meta.json`.

**Priority:** after laneA_1 tier 1. **Decision needed:** none.
