# Box request laneA_1: aggregation over-spread swap arms, full size (lane A, 2026-09-30 21:05 EDT)

**Commit:** `f14886f5572b512b72450c2216a40d3b249caac3` (pushed to origin/main). Pre-registration: `docs/models/aggregation/experiments.md` section 1.

**Inputs needed on the box (all already used by tonight's / the 18:23 UTC session):**
- `data/processed/models/engine_v3/` (HF key `engine_inputs_v3`)
- `data/processed/team_rate_features_E3_v4.parquet`, `data/processed/team_rate_variance_O1a_v3.parquet` (HF `team_rate_tables`; the variance file is NOT matched by that key's glob, section 18 note 3: copy it directly if missing)
- Stage B `T` artifacts (HF `model_artifacts`): `data/processed/models/possession_outcome/round_stageb/T/team_rate_features_E3_v4/`, `data/processed/models/fg_make/round_stageb/T/team_rate_features_E3_v4/B1/` (+ its `m_fitted.json`), `data/processed/models/rebound/round_stageb/T/team_rate_features_E3_v4/artifacts/s2_F2_A0B0C0_seed0/`
- the served artifacts (`engine_inputs`, `model_artifacts`) as for any served sim.

**Env:** `PYTHONIOENCODING=utf-8`, `CBB_TRUTH=verified_v1`. No `ENGINE_*` flags (served defaults). No docker mounts are needed: the runner serves each stack's overlay IN-PROCESS (`run_engine_live.prepare_from_overlay`); it writes scratch adapter dirs under `results/aggregation_v1/_adapter_dirs/` (do not sync those).

**Step 1, build the two input sets (seconds each):**
```
python scripts/build_engine_inputs_v3_tag_v1.py --tag S0_laneA
python scripts/build_engine_inputs_v3_tag_v1.py --tag S1_laneA --team-rate-table data/processed/team_rate_features_E3_v4.parquet --team-rate-missing raise --variance-table data/processed/team_rate_variance_O1a_v3.parquet --po-artifacts data/processed/models/possession_outcome/round_stageb/T/team_rate_features_E3_v4 --fg-artifacts data/processed/models/fg_make/round_stageb/T/team_rate_features_E3_v4/B1 --rb-artifacts data/processed/models/rebound/round_stageb/T/team_rate_features_E3_v4/artifacts/s2_F2_A0B0C0_seed0
```
Check against the local builds (`builder_report.json` `outputs_sha256`): S1_laneA arrays `556b66547f5c...`, event block `21ef52ea6f86...`; S0_laneA arrays `e914337b0973...`, event block `228794fc0d93...`. If they differ, still run, and say so in the .done file.

**Step 2, the sims (tier 1 first, in this order; tier 2 only if time allows):**
```
for A in TEAM OFF DEF ALL PO FG RB RAT; do for S in S0 S1; do
  python scripts/exp_aggregation_swap_v1.py --stack $S --arm $A --all-games --seeds 48 --workers 90 --games-per-block 16
done; done
# tier 2
for A in RAT_PO PACE PLY; do for S in S0 S1; do
  python scripts/exp_aggregation_swap_v1.py --stack $S --arm $A --all-games --seeds 48 --workers 90 --games-per-block 16
done; done
```
Each run is 5,710 games x 48 seeds = 274k game-seeds (about a quarter of one 200-seed full read, so about 3-4 min on 90 workers plus about 1 min of model loading). Tier 1 = 16 runs (about 65 min), tier 2 = 6 runs (about 25 min). The runner keeps no player rows.

**Outputs to sync back (same local paths):** `results/aggregation_v1/{S0,S1}_{ARM}_s48_o0/games.parquet` and `run_meta.json` (about 15 MB per run). Via HF `results` key or scp, whichever is quicker.

**Decision needed:** none; run as written. If the queue is long, tier 1 alone is enough for the decisive read; a partial tier 1 is still useful if it covers TEAM, OFF, DEF, ALL for both stacks.
