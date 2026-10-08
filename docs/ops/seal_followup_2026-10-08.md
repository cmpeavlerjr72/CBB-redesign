# Seal follow-up 2026-10-08 (ops worker)

Seal closed (`seal_lift_approved` false). No model or default change. `preseason_dir.json` NOT switched.

## 1. Preseason rosters
- Both pulls have empty CBBD rosters: `roster_players_2027.parquet` has 0 rows in 2027_v2_20260930 AND 2027_v2_20261008 (roster_teams: 1535 rows, n_players all 0 in 1008; report.json `roster_teams_with_players` 0).
- Consumers of preseason_dir (via `cbb_sim.live.preseason`): chain_daily_v2/v3, ops_seal_week_v1, grade_daily_v1, diag_live_day1_v1, diag_day1_census_v1, build_team_crosswalk_v2_westflorida, pull_preseason_refresh_v2.
  Only two read the roster file: `chain_daily_v3.sim_warnings` (warns only if CBBD file empty AND `data/raw/cbbd/rosters/roster_2027.parquet` (ESPN, 4410 players, 307 teams) is absent; ESPN exists so no warning) and `diag_live_day1_v1` (informational line). Everything else reads games_2027 / teams_2027 / hoopr schedule.
  The sim's rosters come from the ESPN file (`--roster`, stage rosters) and R1 fallback, not from the preseason dir. build_roster_continuity reads `data/raw/preseason/2027/` (portal, recruiting), not preseason_dir.
- Content diff 0930 vs 1008: games_2027, hoopr schedule, conference_changes identical; teams_2027 gains one column; portal_2026 differs in content (not investigated).
- Recommendation: **switch is safe but not needed for rosters** (no consumer needs the CBBD rosters). Switch only if the PM wants the 1008 teams_2027 column / portal refresh; it fixes nothing about rosters. No fix required.

## 2. Real 4-seed 2027 sim: BLOCKED by the seal
- The chain's ratings stage reports `seal_lift_approved` false, so inputs / sim stages block (dry chain 2026-11-02: 118 mappable games, 38 non-D1 unmapped, 91 placeholder tips).
- Direct run `scripts/run_daily_sim_v1.py --slate-date 2026-11-02 --season 2027 --seeds 4 --ratings-dir data/processed/ratings_asof/2026-11-02 --tips table --pass evening ...` stops in `build_live` with `SealedSeasonError` (prior-season carry reads 2025-26 tables). Overriding needs CBB_UNSEAL=1, which I did not set (seal decision belongs to the user / PM). No sim output produced; no sanity numbers.
- To run it: during an approved seal window run the command above with `CBB_UNSEAL=1`, `--root results/seal_followup_sim_v1`, `--players`, `--schedule-source cbbd --schedule-path <preseason_dir>/games_2027.parquet --crosswalk data/reference/team_crosswalk_v2.parquet`, `--now 2026-11-02T01:00:00Z`; or the chain without `--dry-run` once the flag is true. Note the `sim_4seed` stage in ops_seal_week_v1 hard-codes `--dry-run --dry-run-sim` even under --execute.

## 3. Loud failures
- Cause: `chain_daily_v3.main()` (and `chain_daily_v2.main()`) always `return 0`; `run_stage` converts exceptions into a status `error` result and continues. `ops_seal_week_v1` already propagates the child's returncode, and `build_own_ratings_asof_v1.py` already returns/raises properly; the swallow was the chain's exit code.
- Fix: v3 `exit_code(results)` returns 1 if any stage has status `error` (blocked / skipped are not errors); a live chain breaks after the first errored stage (dry runs still run all stages for the census); v2 returns 1 likewise. Test: `tests/test_daily_chain_v3.py::test_chain_exit_code_is_nonzero_when_a_stage_crashes`.
- Checks: test_daily_chain_v3, test_chain_daily_v2, test_chain_daily, test_ops_seal_week: 28 passed. Parity smoke vs parity_reference_windows_v10.json: PASS bit-identical.

## 4. Rotation prior "0 of 236"
- Source: `build_engine_inputs_live.py` line 237, count of slate team-games present in `LF.rotation_priors(ctx, fit, min_prior_games=1)`, which is built from the team's own in-season games. On an opening day nobody has a game yet, so 0 is the expected value (same as the documented "0 of 222 on 2024-11-04"). Not the parked rotation model and not a path bug.
- The gap is handled by the A3 day-1 seed (acts only on all-anonymous team-games) + R1 fallback, which is what builds the names. Confirm seed coverage in the A3 build diag, not this line.
