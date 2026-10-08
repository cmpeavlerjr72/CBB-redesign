# Serving-path seal exemption + real sim_4seed (ops worker, 2026-10-08)

Seal flag unchanged (`seal_lift_approved` false). No model or default change.

## 1. Exemption
- `cbb_sim.data.seal.assert_not_sealed_serving(seasons, serving_season, context)`. Allows 2025-26 only as the PRIOR season when the live season served is 2027; logs and prints `[seal] EXEMPTION USED ...`. Any other serving season falls through to `assert_not_sealed`.
- Single call site: `scripts/build_engine_inputs_live.py::build_live` (the only raise site on the serving path). No env var, no global state; `assert_not_sealed` and all trainers/experiments/eval loaders are untouched and still raise.
- Caveat: any experiment script that calls `build_live(season=2027)` would also be exempt; none exists today.

## 2. Tests
`tests/test_serving_seal_exemption.py` (4): serving 2027 reads 2026 prior while sealed; other serving seasons still raise; generic loader and `eval.reference` reads of 2026 still raise; exemption does not set CBB_UNSEAL.

## 3. sim_4seed
`scripts/ops_seal_week_v1.py`: under `--execute` the stage runs `chain_daily_v3.py --seeds 4 --slate-date ...` (no dry-run flags); the dry run remains the default.

## 4. Real 4-seed sim of 2026-11-02: STILL BLOCKED (second blocker)
build_live now passes (adapter written under `results/serving_sim_v1/sim/2026-11-02/s4_o0/`). The sim then stops in `engine/foul_r9.py::_team_term`: `ao_team_prior_v1.parquet` has no 2027 rows (has 2023-2026). Season 2027 rows are the 2025-26 and-one rates and must be built from the SEALED 2025-26 events (`scripts/build_ao_team_prior_v2.py --season 2027`, a builder that reads sealed events, outside the serving-path scope of this ruling). No sanity numbers produced. Same pattern as the shot-block prior built in the seal-week build d7a4798.

## 5. Regression
Parity smoke vs parity_reference_windows_v10.json: PASS, bit-identical. Full suite: 751 passed, 1 skipped, 2 failed. Both failures are pre-existing stale expectations (fail on the stashed baseline too): `test_hard_stops_2027.py::test_shot_block_live_prior_missing_names_file_and_seal` and `test_shot_block_prior_config_v1.py::test_prior_2025_loads_and_2027_stops_naming_file` expect the 2027 shot-block prior to be missing, but the seal-week build now created it.
