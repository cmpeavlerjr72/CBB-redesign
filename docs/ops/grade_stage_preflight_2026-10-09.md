# Grade and bias/CLV stage preflight (2026-10-09, ops only, model freeze on)

The grade stage (`scripts/grade_daily_v1.py`) and the bias/CLV stage (`scripts/diag_daily_bias_clv_v1.py`) have only ever run in replay.
On 2026-27 the chain reports them SKIPPED ("season not running"). This preflight runs them against fabricated finals so a crash cannot first
appear on Nov 3. Test file: `tests/test_grade_stage_preflight.py` (6 tests, 1.3 s). Fabricated truth lives in `tmp_path` and is deleted with it.

## (a) Synthetic end to end

Fabricated slate: 20 games on 2026-11-02, season 2027, 40 sim seeds, lines present, published 2026-10-09 13:00Z. Game 5 is an OT game
(`n_periods` 3, 84-82). Game 7 carries a 00:00 ET placeholder tip (the "tip time unknown" case). Finals are fabricated in the schema
`run_grade_stage` takes (`game_id, home_score, away_score, n_periods, finals_source`; the ingest schema is `finals_verified_2027.parquet`).
A sixth test (`test_real_served_slate_grades_with_fabricated_finals`) repeats the run on 20 games of the real served 11-02 publication
(`results/daily/publish/2026-11-02/s200_o0/*/slate.parquet`, real sim numbers, fabricated finals) and skips when that file is absent.

Result, no error anywhere:
- Grade: `_status ok`, 20 graded, 0 pending, ledger 20 rows. Per-game rows satisfy `margin_err = sim_margin_mean - actual margin` and
  `total_err = sim_total_mean - actual total` exactly. ATS / O/U / ML P&L settled at the stored lines. `report.md`, `summary.json`,
  `pending.parquet` written. Re-grading is idempotent (0 new rows).
- OT game: `n_periods` 3 reaches the ledger; margin 2.
- Placeholder-tip game: graded, `pre_tip_status = placeholder_unverified`, `pre_tip_verified = False` (no real tip yet, so it stays flagged,
  never assumed pre-tip). If a real tip later shows the build was after tip, the row is `violated` and goes to pending
  `created_after_real_tip` (existing test in `test_tip_guard_placeholder.py`).
- Bias/CLV: `ok`, 20 ledger games, `report.md`, `monitor.json`, `clv.parquet` (20 rows, all CLV columns finite) written. Leak verdict reads
  UNDERPOWERED, as it should at 20 games.
- A game with no final stays pending with reason `no_verified_final`.

## (b) Second-source rule for finals

Two layers, both exist.
1. Ingest (`scripts/pull_daily_ingest_v1.py::verify_finals`): a game is `final_verified` only when hoopR and CBBD both say final and agree on
   both scores. Disagreement, flipped sides, missing CBBD row, or either source not final goes to `pending_{season}.parquet` with a reason
   (`score_disagree`, `sides_flipped`, `awaiting_*`). A HARD box cross-check (points / fgm / tpm / ftm vs CBBD `/games/teams`) can also push a
   game back to pending (`box_disagree`). Only verified games are written to `finals_verified_{season}.parquet`.
2. Grade (`grade_daily_v1.finals_ingest`): re-checks `hoopr_home == cbbd_home` and `hoopr_away == cbbd_away` and non-null, non-0-0, so a bad
   row in the file still cannot be graded.
In replay (`--finals truth`) the truth is `load_actual_games(verified_finals=True)` (CBB_TRUTH=verified_v1), the harness's verified table.
`finals_frame=` (used by tests) bypasses both layers by design; the chain never passes it.

Behaviour on a one-point disagreement (tested): the game is absent from `finals_ingest`, is not in the ledger, and appears in
`pending.parquet`. CHANGE MADE: the pending reason was a generic `no_verified_final`; it is now `finals_sources_disagree` when the finals
table contains the game with disagreeing or missing second-source scores (`finals_ingest_disagree_ids`), so a disagreement is visible in the
report instead of looking like a late final. Tested for a +1 CBBD away score and a null CBBD score. No grading number changes.

## (c) Early-season totals known-bias label (PM decision 2026-10-08)

Before today the label existed nowhere in publish, grade or bias output (the d0-14 bucket only existed in diagnostics). Added, label only:
- Publish rows: boolean column `early_season_totals_flag` (true when the game date is 0-14 days after the season's first D-I game date; NA when
  the season start is unknown, never silently false). Also added: `inputs_hash`, `config_hash`, `engine_tag` from the sim `run_meta.json`.
- Grade: the ledger inherits the column from the publication. Publications written before the label existed are backfilled in the grade stage
  by the same rule. `summary.json` gains `n_early_season_totals_flag`, `total_bias_early_flagged`, `total_bias_unflagged` (a split of the
  existing total bias, nothing adjusted).
- Bias/CLV: `clv.parquet` rows carry the column (tested), so totals rows can be filtered.
Files: `src/cbb_sim/live/daily.py`, `scripts/run_daily_publish_v1.py`, `scripts/grade_daily_v1.py`. Existing signatures unchanged;
`publish_id` and all numbers are bit-identical (tested).

## Open items
- Real finals ingestion and the real CBBD close have still never run on 2026-27 data; this preflight covers the stage code, not the feeds.
- The 10-09 morning publication (27 games) was built before the cache-key fix, so its sim `run_meta.json` has no `inputs_hash`; its publish
  rows will show `inputs_hash` empty if re-published. New passes carry it.
