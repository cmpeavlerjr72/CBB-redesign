# Seal-week runbook rehearsal, 2026-10-07 (ops worker)

Non-sealed steps only. No SEAL_OK stage executed, no 2025-26 data read, seal_lift_approved stays false. Outputs under `results/rehearsal_2026-10-07/` (gitignored) and `results/engine_v0/rehearsal_parity_v9_20261007`.

| stage | status | min | note |
|---|---|---|---|
| parity v9 smoke (run_engine F2 2025, 60 games x 5 seeds, engine_v3, 4 workers; digest_engine_run --compare parity_reference_windows_v9.json) | PASS bit-identical | 0.4 | not a stage of ops_seal_week_v1; run by hand, 26 s total |
| ops_seal_week_v1 dry run, all 17 stages | 12 OK, 4 SEAL_OK, 1 MANUAL, 0 FAIL | 0.4 total (24 s) | script prints no per-stage times (suggest adding); preflight 31 tests pass in ~1 s |
| SEAL_OK: ratings, r9ao3_prior, shot_block_prior, a3_day1_priors | hard stops fire as designed | within the 24 s | runnable only after the lift |
| reseal | MANUAL | - | PM / user edits the flag |
| ESPN 2027 roster count (re-pull to a versioned sibling, 365 requests, 4 workers) | OK | 0.2 | see below |

## Findings
- `--fallback R1` is passed by the a3_day1_priors stage; the builder accepts it (FALLBACKS = None/R1/R2) and `chain_daily_v2` sets `LIVE_ROSTER_FALLBACK = "R1"` for live 2027 and hard-stops if the S-1 roster file is missing. Static and dry-run only: the A3 `tables()` call itself is sealed, so the 2027 inputs stage was not exercised end to end.
- ESPN 2027 roster coverage: 307 / 365 teams (was 296), 4410 players, 4072 cbbd-mapped, 345 transfers in. 58 teams still return stale season.year 2026 and are excluded (they fall to R1 on Oct 10). Coverage rises as ESPN rolls teams to 2027; re-pull on the day. Threshold in the puller is 250 teams / 3000 players: passes.
- The served `data/raw/cbbd/rosters/roster_2027.parquet` was NOT overwritten (rehearsal wrote a sibling).

## Blockers / manual steps for Oct 10
1. seal_lift_approved must be set true by user / PM; `--execute` refuses otherwise.
2. After the fresh preseason pull, PM switches `data/overrides/preseason_dir.json`.
3. Re-pull rosters on the day (the stale-team list shrinks); the 58 stale teams depend on R1.
4. sim_4seed and inputs_census were only statically checked (need ratings + priors from the sealed stages).
5. Parity smoke is not wired into the runbook; add it as a stage (or keep as a manual first step). hf_sync dry check shows 2750 files pending upload.
6. Dry run found no FAILs.
