# Truth tables v2 -- technical FTs and the 2025 finals resolutions folded in (2026-09-10)

Built by `.venv/Scripts/python.exe scripts/build_truth_tables_v2.py`, seasons
2022-2025 (2026 sealed, never read). `scripts/build_truth_tables.py` (v1) is
untouched and still owns the `*_v1.parquet` files; this script imports its
team/box/player/finals builders directly and adds the technical-free-throw
event walk plus the finals resolutions on top. Answers the open question left
by `docs/tests/truth_tables_v1_2026-09-10.md` and diagnosed in
`docs/tests/ft_trip_reconciliation_2026-09-10.md` / `docs/LEARNINGS.md` L24:
netting technical FTs out of the event-vs-box FTA comparison should push
exact-match rate from ~90-93% to well above 97%.

    data/processed/truth/team_game_shots_v2.parquet   44,714 rows,  1.64 MB
    data/processed/truth/player_game_v2.parquet      707,336 rows, 10.36 MB
    data/processed/truth/game_finals_v2.parquet       22,414 rows,  0.43 MB
    data/processed/truth/build_report_v2.json          (the numbers below)

All three under the 50 MB cap; row counts identical to v1 (this is an
additive/corrective pass, not a new universe cut).

## 1. `team_game_shots_v2` -- technical FTs identified from the event layer

`fta_tech`/`ftm_tech` are computed the way `cbb_sim.pbp.possessions
._handle_technical` + `_collect_trip` identify a technical free-throw trip: a
`technical` play row immediately followed (allowing an administrative
OREB/DeadBallReb skip, the same rule real trips use) by a run of FT rows on
one team -- every FT in that run is a technical attempt. `ev_fta_plus_tech` /
`ev_ftm_plus_tech` add them to the possession-layer `ev_fta`/`ev_ftm` (which
deliberately exclude technicals, correctly, per L24); `diff_fta_tech` /
`diff_ftm_tech` compare that sum to `box_fta`/`box_ftm`. `box_points` (hoopR
`team_box.team_score`) backs a naive points identity with and without
technicals (`ev_points_naive` = 2*2PM + 3*3PM + `ev_ftm`; `ev_points_tech`
adds `ftm_tech`).

### FTA exact-match rate, before vs after netting technicals (event+box matched rows)

| season | n | raw exact | raw 2+ off | tech-adj exact | tech-adj 2+ off |
|---|---|---|---|---|---|
| 2022 | 10,564 | 91.59% | 6.42% | **99.25%** | 0.37% |
| 2023 | 11,080 | 88.91% | 7.23% | **99.56%** | 0.15% |
| 2024 | 11,102 | 92.02% | 6.71% | **99.76%** | 0.04% |
| 2025 | 11,179 | 92.09% | 7.08% | **99.38%** | 0.37% |

### FTM exact-match rate, same comparison

| season | raw exact | raw 2+ off | tech-adj exact | tech-adj 2+ off |
|---|---|---|---|---|
| 2022 | 92.54% | 4.60% | **99.58%** | 0.29% |
| 2023 | 90.25% | 5.44% | **99.76%** | 0.14% |
| 2024 | 92.73% | 4.80% | **99.93%** | 0.03% |
| 2025 | 92.66% | 5.37% | **99.60%** | 0.27% |

Every season clears the pre-registered ">97% for FTA" bar by a wide margin
(99.25-99.76%), confirming L24's headline finding (94.3% of the ORIGINAL
disagreeing rows explained) at the table level: the flat ~90-93% agreement
that tracked nothing else in the feed was overwhelmingly this one mechanism.

### Residual disagreement, after netting `fta_tech`

A coarser 2-way split than `ft_trip_reconciliation`'s 5-way waterfall (this
table doesn't re-derive orphan front-end chances / end-of-period trips /
blank-shooter rows -- see that doc's section 2 for those), computed directly
from columns now on the table:

| season | n residual | tech present, still off | tech absent, unexplained |
|---|---|---|---|
| 2022 | 79 | 8 | 71 |
| 2023 | 49 | 7 | 42 |
| 2024 | 27 | 2 | 25 |
| 2025 | 69 | 6 | 63 |

Residual counts are small (27-79 of 10,564-11,179 matched team-games/season,
0.25-0.74%) and dominated by "tech absent, unexplained" -- the same
population `ft_trip_reconciliation` attributes to front-end one-and-one
orphans, end-of-period trips and blank-shooter rows, none of which recur here
at a rate that changes the headline number.

### Points identity, before vs after netting `tech_ftm`

`diff_points_notech` / `diff_points_tech` = naive event-layer points minus
`box_points` (hoopR `team_score`), on rows with a resolved box score:

| season | n | mismatch (no tech) | mismatch (tech) |
|---|---|---|---|
| 2022 | 10,564 | 18.55% | 12.61% |
| 2023 | 11,080 | 20.08% | 12.02% |
| 2024 | 11,102 | 11.15% | 4.00% |
| 2025 | 11,179 | 8.48% | 1.61% |

Matches `ft_trip_reconciliation`'s independently-computed section 3 numbers
exactly (12.6/12.0/4.0/1.6%), confirming the two implementations of the
technical-FT walk agree.

## 2. `game_finals_v2` -- the four 2025 resolutions applied

Per `data/processed/truth/diag_finals_resolution_2025.json` (ESPN third
source): hoopR right on three, CBBD right on one. `home_score`/`away_score`
are corrected only where the resolution requires it; `finals_source`,
`finals_third_source_checked` and `finals_resolution_note` are new columns.
All 22,410 non-flagged rows are byte-identical to v1 (columns, dtypes and
values all checked).

| game_id | v1 home-away | v2 home-away | finals_source | note |
|---|---|---|---|---|
| 401745889 | 79-59 | 79-59 (unchanged) | hoopr | CBBD home score off by 2 |
| 401723767 | 69-81 | 69-81 (unchanged) | hoopr | CBBD homePoints=0 is a data gap |
| 401722537 | 62-60 | **60-62 (corrected)** | cbbd | hoopR carried the side-flip, not CBBD |
| 401746100 | 80-67 | 80-67 (unchanged) | hoopr | CBBD away score off by 2 |

## 3. `player_game_v2` -- per-player technical FTA/FTM

Same technical-FT walk keyed on `shot_shooter_id`, joined on `cbbd_player_id`
for the two seasons with a roster crosswalk. 2022-2023: `fta_tech`/`ftm_tech`
are null on every row (no crosswalk at all, same treatment v1 gives `ev_fga`
etc.) -- stated, not hidden. 2024: 999 player-games have `fta_tech > 0`; 2025:
939. Every other v1 column is unchanged.

## 4. `reference.py` loaders

`load_team_shot_truth` / `load_player_game_truth` / `load_game_finals_truth`
now default to `version="v2"` (new keyword arg, positional signature
unchanged so `gates.py`'s existing `(season, truth_dir)` calls are
unaffected); pass `version="v1"` for the frozen original. `gates.py` and
`eval_gates.py` were not touched (another worker owns the G4 eFG% contract
there this session) -- v2 is a strict column superset of v1 so no gate logic
needs to change to read it.

`pytest tests/test_eval.py -q`: **22 passed** (0 failed) in 2.0s, confirming
the loader default swap does not break the eval harness. The broader
`tests/` directory is shared with many other in-flight workers' new model
code this session (clock v3, fg_make round2, usage v2, rotation, etc. --
`git status` shows a dozen unrelated dirty files); a full-suite run was
started, found to be slow under that concurrent load, and was stopped
rather than tie up shared compute for a result outside this task's scope.

## 5. Limitations, stated rather than hidden

- The residual-disagreement split here is coarser than
  `ft_trip_reconciliation`'s 5-way waterfall; it reuses only what the v2
  table itself computes (whether a technical trip was found at all), not a
  full re-derivation of orphan-front-end/end-of-period/blank-shooter classes.
- `game_finals_v2`'s corrected `home_score`/`away_score` for 401722537 is not
  yet consumed anywhere -- same as v1, no grading script reads this table
  directly yet.
- Nothing here is a bake-off decision (`CLAUDE.md`'s bake-off rule covers
  model/feature choices); this is a second-source reconciliation refinement
  of existing truth data, same footing as v1.

## Change ledger

Row added to `docs/models/change_ledger.md` section D, same commit series as
this doc (PM commits after review; not committed by this worker).
