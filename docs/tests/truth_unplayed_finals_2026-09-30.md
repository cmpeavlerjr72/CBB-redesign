# Grading truth: unplayed and disputed finals (Lane H, 2026-09-30)

Scope: second-source verification of finals for every season on disk, where the bad rows flow, an opt-in loader fix (default unchanged), and a no-sim regrade of the served-stack 200-seed read under current and corrected truth. Season 2025-26 (universe `season == 2026`) is SEALED: counted by status / 0-0 / missing only, nothing model-related computed.

Season convention: universe / hoopR `season` = ending year. Fold 2 = `season 2025` (2024-25). Fold 1 test = `season 2024`.
Scripts: `scripts/diag_truth_finals_audit_v1.py`, `diag_truth_flow_scan_v1.py`, `diag_truth_gate_targets_v1.py`, `grade_truth_regrade_v1.py`. Outputs: `data/processed/truth/truth_audit_unplayed_v1.parquet` (+ `_summary.json`), `truth_flow_scan_v1.csv`, `truth_gate_targets_check_v1.csv`. Full side-by-side gate detail (every table row that changes): `docs/tests/truth_unplayed_finals_2026-09-30_side_by_side.md`.

## 1. Audit: sources and result

Sources joined on `game_id == CBBD sourceId`: (A) hoopR schedule (score, status), (B) hoopR team box (`team_score`, and points rebuilt as 2*FGM + 3PM + FTM), (C) CBBD games (points, status, per-period points). Note B's `team_score` is the same ESPN feed as A (0 disagreements in every season), so it is not an independent check; only C is. The only third source ever used is `data/processed/truth/diag_finals_resolution_2025.json` (ESPN game pages, 4 games, 2026-09-10). I did not fetch any new third source.

### 1.1 Rows by class, per season (all games in the hoopR schedule; CBBD count in brackets where it differs)

| class | 2021-22 (2022) | 2022-23 (2023) | 2023-24 (2024) | 2024-25 (2025, fold 2) | 2025-26 (2026, sealed) |
|---|---:|---:|---:|---:|---:|
| games in hoopR schedule / CBBD | 5,976 / 6,387 | 6,261 / 6,261 | 6,249 / 6,249 | 6,299 / 6,299 | 6,318 / 6,317 |
| hoopR status FINAL | 5,966 | 6,222 | 6,243 | 6,292 | 6,300 |
| hoopR non-final (forfeit / canceled / postponed / scheduled) | 10 (10 forfeit) | 39 (6 / 21 / 12 / 0) | 6 (0 / 2 / 4 / 0) | 7 (0 / 1 / 5 / 1) | 18 (0 / 4 / 14 / 0) |
| hoopR 0-0 score | 0 | 33 | 6 | 7 | 18 |
| hoopR forfeit scored 2-0 (nominal) | 10 | 6 | 0 | 0 | 0 |
| hoopR missing score | 0 | 0 | 0 | 0 | 0 |
| CBBD non-final | 421 (218 scheduled, 193 cancelled, 10 postponed) | 39 | 6 | 7 | 18 |
| CBBD 0-0 | 410 | 33 | 6 | 7 | 18 |
| CBBD missing score | 0 | 0 | 0 | 1 (game 401726454, cancelled) | 0 |
| in CBBD only (not in hoopR schedule) | 411 (all unplayed) | 0 | 0 | 0 | 1 |
| in hoopR only (not in CBBD) | 0 | 0 | 0 | 0 | 2 |
| hoopR FINAL but no team box | 1 | 2 | 3 | 7 | 1 |
| **unverified final inside the graded rule (D-I, non-truncated)** | **10** | **34** | **5** | **5** | 15 (status count only) |
| of which 0-0 | 0 | 28 | 5 | 5 | 15 |
| of which forfeit 2-0 | 10 | 6 | 0 | 0 | 0 |
| graded-rule games (`is_d1_game & ~pbp_truncated`) | 5,406 | 5,658 | 5,640 | 5,710 | 5,762 |

Every hoopR non-FINAL row is also non-final in CBBD, every hoopR FINAL row is CBBD final (status agreement 100%), and none of the unverified graded-rule games has a team box or pbp. The one status inconsistency is game 401726454 (2025, non-D-I, hoopR "scheduled" 0-0, CBBD "cancelled" with no points) which nevertheless has a team box of 121-49, i.e. hoopR's status is stale for a played game; it is outside the graded set.

The five fold-2 0-0 games are 401714278, 401722532, 401706691, 401700283, 401716154 (hoopR POSTPONED; CBBD cancelled or scheduled). The 2022 forfeits (2-0 or 0-2, both sources agree) are 401370817, 401370829, 401370830, 401370856, 401370857, 401370893, 401371464, 401373159, 401373175, 401402244; they are graded as 2-0 finals today.

### 1.2 Finals agreement between the two scoring sources (hoopR schedule vs CBBD, both FINAL)

| season | games both final | score disagreements | agreement |
|---|---:|---:|---:|
| 2022 | 5,966 | 0 | 100% |
| 2023 | 6,222 | 0 | 100% |
| 2024 | 6,243 | 0 | 100% |
| 2025 | 6,292 | 5 | 99.921% |
| all four | 24,723 | 5 | **99.980%** |

The five disagreements (all 2025), with both values and the resolution on file:

| game_id | hoopR (home-away) | CBBD | class | resolution | in graded rule? |
|---|---|---|---|---|---|
| 401745889 | 79-59 | 77-59 | CBBD one side off by 2 | ESPN: hoopR right | yes |
| 401746100 | 80-67 | 80-65 | CBBD one side off by 2 | ESPN: hoopR right | yes |
| 401723767 | 69-81 | 0-81 | CBBD home points = 0 (data gap; per-period sum also disagrees with CBBD's own points) | ESPN: hoopR right | yes |
| 401722537 | 62-60 | 60-62 | sides flipped | ESPN: **CBBD right, hoopR flipped** | yes |
| 401706625 | 80-72 | 80-71 | CBBD one side off by 1; hoopR schedule and box agree on 80-72 | no third source fetched; box favours hoopR | no (pbp-truncated) |

Overtime accumulation: 321 / 361 / 346 / 325 games (2022-2025) are OT in one or both sources; **0** of them disagree on score. CBBD per-period points sum to its own final in every final game except 401723767. Universe `n_periods` vs CBBD period count disagrees in 1 game (2022). `n_periods` is NaN in the universe for 124 / 11 / 4 / 5 graded-rule games (2022-2025; 2026: 13); in 2025 those five are exactly the 0-0 games, in 2022 the rest have no linescore or pbp, so 2022 OT is undercounted in the universe (311 vs 320 in CBBD). No OT score-accumulation error found.

**Defect in the graded truth beyond the 0-0 rows:** `games_universe.parquet` (and `_v2`) carries hoopR's flipped sides for game 401722537 (62-60), while `game_finals_v2.parquet` has the ESPN-resolved 60-62. `eval.reference.load_actual_games` reads the universe, so fold 2 is graded with the wrong sign of margin for that game. `game_finals_v2` was resolved on 2026-09-10 but never fed back into the graded truth. Only 4 games have `finals_third_source_checked=True`; the 54 unverified-final games in the graded rule carry `finals_third_source_checked=False` (their non-play is established by two agreeing statuses, not by a third source).

## 2. Where the rows flow

| table | 0-0 / forfeit / non-final games inside | verdict |
|---|---|---|
| `games_universe.parquet`, `games_universe_v2.parquet` | 62 unplayed ids (2022-2025), 54 inside the graded rule (10 / 34 / 5 / 5) | source of the defect |
| `eval.reference.load_actual_games` (all G1-G9 and G10 callers) | same 54 (fold 2: 5; fold-1 test 2024: 5) | **grading defect** |
| `data/reference/gate_targets_{2022..2025}.parquet` game-level rows | contaminated (below) | grading targets, not features |
| `data/processed/truth/game_finals_v1/v2.parquet` | 54 | same rows, flagged |
| `data/processed/lines/lines_close_v1.parquet` | 30 unplayed games with line rows (43 rows: 25 games in 2023, 2 in 2024, 3 in 2025; by provider 24 teamrankings, 15 ESPN BET, 4 consensus) | market graders (`grade_market_games_v2`, `grade_market_props`) join lines to `load_actual_games`, so a line on a 0-0 game is graded as a 0-0 result |
| `data/processed/models/engine/games_F2_2025.parquet`, `_v2`, `engine_fgm4/games_F2_2025.parquet` (engine inputs, built from `load_actual_games`) | 5 each | the sim runs the 5 games (harmless), but the graded set then includes them |
| every `results/**/games.parquet` for fold 2 | 34 full-season runs (5,710 games): 5 each. 500-game stride samples: **84 of 101** dirs contain 1 (game 401714278, including `po4b_R_s25`); 250-game samples: 4 of 8 contain 1. Other sizes (458, 5,195, 5,700, 60, ...): 0 | closed-loop reference and arm carry the same rows, so paired deltas are unaffected but absolute G5 / G9 lines shift |
| **features and training**: `own_ratings_*`, `possessions_v3/*`, `possessions_pbp`, `models/{attribution,clock,fg_make,free_throw,pace,possession_outcome,rebound,rotation,usage*,late_game}`, `clock_censoring`, `truth/player_game_*`, `team_game_shots_*`, all other parquet with a `game_id` column (424 files scanned) | **0** | **clean** |

Why the features are clean: `own_ratings.load_team_games` (the feature builder for own ratings, pace, gate tables) requires exactly two usable team-box rows, and none of the 54 has a box row (or pbp). A 0-0 game therefore cannot appear in any team's as-of history. The one box-having unplayed-status row (401726454) is non-D-I and outside every table. `train_fg_make_v4_shooter_block.py` reads `load_actual_games` for a margin panel but inner-joins to shot rows, which the unplayed games lack, so they drop out. Caveat: the scan covers parquet tables with a `game_id` column; npz arrays under `models/engine` are built from the `games_F2_2025*.parquet` listed above.

### 2.1 `gate_targets` contamination (game-level rows only; team-level rows come from the team box and are clean)

Stored season/all values equal a recompute from the graded rule including the unplayed games (verified to 4 decimals), so the stored targets are contaminated. Removing the unverified-final games:

| season | n removed | total_points_mean | total_points_sd | margin_sd | home/away score corr | home margin (non-neutral) |
|---|---:|---|---|---|---|---|
| 2022 | 10 | 139.800 -> 140.056 | 19.294 -> 18.376 | 14.133 -> 14.144 | 0.3017 -> 0.2560 | 4.978 -> 4.989 |
| 2023 | 34 | 140.530 -> 141.377 | 21.468 -> 18.551 | 13.870 -> 13.906 | **0.4113 -> 0.2806** | 5.376 -> 5.414 |
| 2024 | 5 | 145.128 -> 145.257 | 19.046 -> 18.557 | 14.339 -> 14.345 | 0.2767 -> 0.2522 | 5.364 -> 5.370 |
| 2025 | 5 | 145.509 -> 145.637 | 18.954 -> 18.466 | 14.633 -> 14.639 | 0.2532 -> 0.2283 | 5.738 -> 5.744 |

Month and tier rows of the same metrics are touched in 2 / 5 / 2 / 4 months. Neutral home margin, OT rate, possessions and all four-factor rows do not move. Consumers: `gates.py` (G1/G7 targets, both unaffected in value), `grade_control.py`, `grade_late_game_r2_v1.py`. Nothing trains on them. A rebuild (`scripts/build_gate_reference.py`, not run today) would change only these game-level rows in `gate_targets_2022..2025` and `gate_targets_fold2_train`; the 2025-26 file has 15 more unplayed games in the same rows (not computed).

## 3. The fix (opt-in, default unchanged)

`eval.reference.load_actual_games(season, universe_path=..., verified_finals=False)`.

- Default `False`: byte-for-byte the previous behaviour (test asserts 5,710 rows and the five 0-0 rows).
- `True`: drops every game in the new `eval.reference.unverified_final_game_ids(season)` (hoopR status not FINAL, or hoopR 0-0, or CBBD status not "final", or CBBD 0-0), and takes scores from the third-source-resolved `game_finals_v2.parquet` where they differ from the universe (fixes 401722537). Fold 2 becomes 5,705 games; every other row's scores are identical to the default (tested).
- Tests: `tests/test_eval.py::test_load_actual_games_verified_finals_is_opt_in` and `::test_unverified_final_game_ids_2025` (24 eval tests pass).
- Not changed: `build_gate_reference.py`, the reference and engine-input tables, any feature table, `gates.py`. When the PM flips the default, all `load_actual_games` callers (gates, market graders, engine-input builder, closed-loop graders) pick it up together. Two consequences to plan for: `build_engine_inputs.py` would then simulate 5,705 not 5,710 games, and the market graders need `lines_close_v1` joined against the same verified set (they already inner-join truth).

## 4. Regrade: served stack, 200 seeds, stored results

Results `results/engine_v0/F2_2025_s200_v5b_A_full` are local (206 MB), so this is the full 200-seed A stream regraded, no pull, no sim. `scripts/grade_truth_regrade_v1.py` runs the same gate code path as `eval_gates.py` twice; the corrected pass patches only `load_actual_games(verified_finals=True)`. G1/G7 targets are read from the un-rebuilt `gate_targets_2025` (game-level rows do not enter those two). G8 depends only on the player truth table and was run once. The "current" column reproduces the 2026-09-18 report exactly (G1 69.866, G5 0.8983 / 0.2532, G9 -0.8617).

Games graded: current 5,710, corrected 5,705.

| gate | line | current truth | corrected truth | status cur -> corr |
|---|---|---|---|---|
| G1 | possessions/game mean | 69.866 vs 67.875 | 69.865 vs 67.875 | FAIL -> FAIL |
| G1 | possessions/game SD | 5.552 vs 5.474 | 5.552 vs 5.474 | PASS -> PASS |
| G1 | by month | 0/5 powered months inside | 0/5 | FAIL -> FAIL |
| G2 | PPP by offense x defense tercile | 2/9 cells inside | 2/9 cells inside | FAIL -> FAIL |
| G3 | by team / by team rim / pooled 3PA share, FTA/FGA, rim share | NEEDS-INSTR. / PASS x3 | identical | unchanged |
| G4 | by team; pooled tov% PASS, oreb% FAIL, ft_rate PASS, eFG% FAIL | as 2026-09-18 | identical | unchanged |
| G5 | margin SD ratio | 1.0396 | 1.0391 | PASS -> PASS |
| G5 | **total SD ratio** | 0.8983 | **0.9233** | FAIL -> FAIL (band 0.95-1.05) |
| G5 | **home/away score corr (sim vs actual)** | 0.1170 vs 0.2532 (gap 0.136) | 0.1169 vs **0.2283** (gap 0.111) | FAIL -> FAIL (tol 0.05) |
| G5 | PIT K-S p | 0.016 | 0.0144 | FAIL -> FAIL |
| G6 | home margin non-neutral | +5.691 vs +5.738 | +5.695 vs +5.743 | PASS -> PASS |
| G6 | home margin neutral | +2.105 vs +3.288 | +2.105 vs +3.288 | FAIL -> FAIL |
| G7 | OT rate; 1H/2H share | 0.0304 vs 0.0557; n/a | identical | FAIL; NEEDS-INSTR. |
| G8 | all four lines | as 2026-09-18 | identical | unchanged |
| G9 | margin bias | -0.1932 | -0.1949 | PASS -> PASS |
| G9 | **total bias** | -0.8617 | **-0.9836** (tol +/-1.0) | PASS -> PASS (0.016 from the edge) |
| G9 | calibration slope | 0.9096 | 0.9096 | FAIL -> FAIL |
| G9 | bias by month (cells outside tolerance) | 4/10 | **5/10** (Dec total bias -0.967 -> -1.115) | FAIL -> FAIL |
| G9 | bias by home tier | 3/6 | **4/6** (bottom tercile total bias -0.907 -> -1.080) | FAIL -> FAIL |
| G9 | bias by predicted-total tercile | 3/6 | 3/6 (bottom tercile total bias -1.383 -> -1.674) | FAIL -> FAIL |

**No gate-level verdict changes; no PASS line becomes FAIL at the headline level and no FAIL becomes PASS.** Changes that matter for reading the gates:

1. The total SD ratio moves 0.898 -> 0.923 and the score correlation gap narrows from 0.136 to 0.111. Both still fail; the sim's own correlation (0.117) is the larger part of the gap.
2. Total bias worsens to -0.984 against a +/-1.0 tolerance: the headline G9 total-bias PASS is now within noise of failing, and two sub-cells flip PASS to FAIL on total bias (December, and the bottom home tier); the bottom predicted-total tercile, already failing, goes further out (-1.383 to -1.674). The sim under-predicts totals by more, not less, once the zeros are removed (the actual total mean rises 0.13).
3. G2 tercile cells shift by up to 0.002 PPP (terciles are built from each team's own scoring average, which the 0-0 games depressed); the count of cells inside +/-0.02 stays 2/9.
4. Sim-side numbers move by at most 0.001-0.003 because five sim games leave the pool. Nothing here touches the engine.

The engine-side G1, G2 (levels), G3, G4, G6-neutral, G7, G8 conclusions are unchanged.

## 5. What is and is not established

- Established: 54 graded-rule games (10 / 34 / 5 / 5 in 2022-2025) have no verified played final in both sources; fold 2 has 5 zero-score ones plus 1 wrong-sided final. Features and training tables are clean (0 hits in 424 tables). The contamination is confined to grading truth, `gate_targets` game-level rows, `lines_close_v1` line rows on unplayed games, and the closed-loop game samples (1 unplayed game in 84 of the 101 500-game stride samples).
- Not done: no default flipped, no reference or engine-input rebuild, no new third-source fetch for the 54 (non-play rests on two agreeing statuses, no box, no pbp). Fold-1 test (2024) and any regrade of fold-1 runs are affected by 5 zero games and were not regraded. The 2025-26 season was only counted.
- Recommended after the paired-loop lanes report: flip the default to `verified_finals=True`, rebuild `gate_targets_2022..2026` game-level rows through the same filter (2026 by the PM under the seal rules), drop unplayed ids from `lines_close_v1` consumers by joining the verified truth, and rerun the 500-game references so the reference and arm stay paired.
