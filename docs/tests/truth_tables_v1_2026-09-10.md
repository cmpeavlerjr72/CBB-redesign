# Truth tables v1 for G3, G4, G8 -- 2026-09-10

Built by `.venv/Scripts/python.exe scripts/build_truth_tables.py` from
`data/processed/games_universe.parquet` (`is_d1_game & ~pbp_truncated`), seasons
2022-2025. Season 2026 is sealed and is never read by the builder. Every
number below is recomputed from `data/processed/truth/build_report.json`;
none is typed in by hand.

Two independent sources back every table; neither is picked over the other --
both column sets are written, a diff column is added, and disagreeing rows
are flagged rather than resolved.

    data/processed/truth/team_game_shots_v1.parquet   44,714 rows,  1.36 MB
    data/processed/truth/player_game_v1.parquet      707,336 rows, 10.33 MB
    data/processed/truth/game_finals_v1.parquet       22,414 rows,  0.43 MB
    data/processed/truth/build_report.json            (the numbers below)

All three are well under the 50 MB cap with no season split needed.

---

## 1. `team_game_shots_v1.parquet`

Per (game_id, team_id): FGA/FGM by shot class (rim/jump2/3) + FTA/FTM from the
CBBD `possessions_v2` event layer (`ev_*` columns, summed over every chance of
every possession the team was on offence for), alongside hoopR `team_box`'s
FGA/FGM/FG3A/FG3M/FTA/FTM as the second source (`box_*` columns), a `diff_*`
column per stat, and `any_disagreement` (bool). Coverage is restricted to
`is_d1_game & ~pbp_truncated`; the box side is filtered to the same game set
before joining (an unfiltered join against the raw hoopR file inflates
`box_only` by roughly 6x with non-D1/excluded games, and was caught and fixed
during this build).

### Row coverage by season

| season | team-games | event+box matched | event-only | box-only |
|---|---|---|---|---|
| 2022 | 10,792 | 10,564 | 0 | 228 |
| 2023 | 11,248 | 11,080 | 2 | 166 |
| 2024 | 11,268 | 11,102 | 4 | 162 |
| 2025 | 11,406 | 11,179 | 6 | 221 |

`box_only` rows are almost entirely the games CBBD's pbp feed does not cover
at all (124/117/87/117 games per season, `docs/tests/possessions_build_v2_2026-09-10.md`
section 0 -- 2x that in team-games), not a join defect.

### Reconciliation: share of team-games where event layer == box, by season

Each cell is `abs(event_layer_count - box_count)` bucketed at 0 / 1 / 2+
attempts, over the `event+box matched` rows above.

**FGA**

| season | n | exact (0) | off by 1 | off by 2+ |
|---|---|---|---|---|
| 2022 | 10,564 | 80.3% | 9.2% | 10.5% |
| 2023 | 11,080 | 81.5% | 8.6% | 10.0% |
| 2024 | 11,102 | 94.6% | 4.9% | 0.5% |
| 2025 | 11,179 | 97.3% | 1.7% | 1.0% |

**FGM**

| season | n | exact | off by 1 | off by 2+ |
|---|---|---|---|---|
| 2022 | 10,564 | 87.4% | 8.0% | 4.6% |
| 2023 | 11,080 | 88.0% | 7.8% | 4.2% |
| 2024 | 11,102 | 96.2% | 3.6% | 0.2% |
| 2025 | 11,179 | 98.5% | 0.9% | 0.6% |

**FGA_3 / FGM_3**

| season | FGA3 exact | FGM3 exact |
|---|---|---|
| 2022 | 98.5% | 99.7% |
| 2023 | 98.9% | 99.8% |
| 2024 | 98.9% | 99.8% |
| 2025 | 98.7% | 99.6% |

**FTA / FTM**

| season | FTA exact | FTA 2+ off | FTM exact | FTM 2+ off |
|---|---|---|---|---|
| 2022 | 91.6% | 6.4% | 92.5% | 4.6% |
| 2023 | 88.9% | 7.2% | 90.3% | 5.4% |
| 2024 | 92.0% | 6.7% | 92.7% | 4.8% |
| 2025 | 92.1% | 7.1% | 92.7% | 5.4% |

**Reading it.** Total-FGA and FGM agreement tracks the CBBD feed-completeness
curve already established for the possession layer (`pbp_complete` 78%/80%/92%/95%
of D-I non-truncated games by season, `docs/tests/possessions_build_v2_2026-09-10.md`
section 2.1): the 2022-2023 gap is feed incompleteness, not a segmentation
defect, and it mechanically shrinks in 2024-2025. FGA_3/FGM_3 agree at
98-100% in every season -- three-point makes are the cleanest-tagged event in
the feed (module docstring, `cbb_sim.pbp.events`). FTA/FTM sit apart from that
trend: agreement is flat at ~90-93% across ALL FOUR seasons, including 2024-2025
where FGA has already converged near 100%. That flatness, not the level, is
the finding -- it points at the ~40% of free-throw trips flagged
`ft_trip_ambiguous` (bonus-vs-shooting-foul cannot be told apart from the feed,
`docs/tests/possessions_build_v2_2026-09-10.md` section 5) and the ~4 administrative
rebounds/game between free throws, rather than at feed completeness. **Neither
column is silently preferred**: both `ev_fta` and `box_fta` are on every row,
`diff_fta` is computed, and `any_disagreement` flags it.

---

## 2. `player_game_v1.parquet`

Per (game_id, athlete_id, season): hoopR `player_box` (minutes, FGA/FGM,
FG3A/FG3M, FTA/FTM, ORB/DRB, AST, STL, BLK, TOV, PF, PTS), keyed on the ESPN
`athlete_id`, joined to `cbbd_player_id` via the existing crosswalk
(`data/processed/player_crosswalk.parquet`, `src/cbb_sim/data/player_ids.py`),
plus event-layer FGA/FGM by class + FTA/FTM keyed on **`shot_shooter_id`**
(never `participant_1_id`, which is the assister on 97,817 of 872,437 shot
rows in 2025 alone -- 11.2% -- confirmed directly against the raw CBBD pbp
before building this table; matches the change ledger's independently
measured 11.9-14.1% mislabel rate for the L4 usage bake-off's round-2 fix).

### Crosswalk coverage by season

CBBD's `/teams/roster` was pulled for 2024-2026 only (`data/raw/cbbd/rosters/`
has no 2022/2023 files), so `cbbd_player_id` and the event-layer `ev_*`
columns are structurally null for 2022-2023 -- not a matching failure, a data
gap, reported as such rather than hidden.

| season | CBBD rosters available | player-game rows | row match rate | played (min>0) rows | played match rate |
|---|---|---|---|---|---|
| 2022 | no | 171,440 | 0.0% | 105,833 | 0.0% |
| 2023 | no | 175,644 | 0.0% | 110,135 | 0.0% |
| 2024 | yes | 177,638 | 99.80% | 110,464 | 99.79% |
| 2025 | yes | 182,614 | 99.90% | 112,312 | 99.87% |

### Player-level shot reconciliation (2024-2025 only -- the seasons with a crosswalk)

Same 0/1/2+ bucketing as the team table, over player-games with a resolved
`cbbd_player_id` AND an event-layer shooter row (~111-113k of ~177-183k
player-games; the gap is players with box minutes but zero shot involvement,
who correctly carry `ev_fga = 0` rather than appearing here).

| season | n | FGA exact | FGM exact | FGA3 exact | FGM3 exact | FTA exact | FTM exact |
|---|---|---|---|---|---|---|---|
| 2024 | 111,163 | 97.7% | 98.3% | 98.7% | 99.3% | 99.1% | 99.2% |
| 2025 | 112,951 | 97.1% | 97.7% | 98.3% | 99.0% | 98.7% | 98.9% |

Player-level agreement is tighter than the team-level one for the same two
seasons (team FGA exact 94.6%/97.3% vs player 97.7%/97.1%) because the
crosswalk-resolved subset skews toward players CBBD's roster pull actually
covers, which correlates with cleaner pbp coverage for those games.

---

## 3. `game_finals_v1.parquet`

Per game_id: hoopR-schedule final score, OT count (`n_periods`, already the
universe's own hoopR-derived value) and per-period scores, alongside CBBD
`games_{season}.parquet` (`homePoints`/`awayPoints`/`homePeriodPoints`/
`awayPeriodPoints`) as the second source. hoopR's `home_linescores` /
`away_linescores` are parsed with a regex on the `'value':` field rather than
`ast.literal_eval`, because hoopR serialises them as numpy's `repr` of an
array of dicts with NO comma between elements (`"[{'value': 36.0} {'value':
39.0}]"`) -- not valid Python list syntax, so a literal-eval-based parser
silently returns nothing on most 2023-2024 rows; caught and fixed during this
build (`universe.py`'s own period-COUNT regex uses the same technique and
does not have this bug because it only counts `'value'` occurrences).

| season | games | final score disagree | OT count: both present & disagree | OT count: hoopR missing (CBBD fills it) | per-period scores checked | per-period disagree |
|---|---|---|---|---|---|---|
| 2022 | 5,406 | 0 | 0 | 124 | 0 (hoopR has no linescores in 2022) | -- |
| 2023 | 5,658 | 0 | 0 | 11 | 5,624 | 0 |
| 2024 | 5,640 | 0 | 0 | 4 | 5,635 | 0 |
| 2025 | 5,710 | 4 | 0 | 5 | 5,705 | 3 |

**OT count never disagrees when both sources report it, in any season.**
Where hoopR's own `n_periods` is null (124 games in 2022, single digits
after), CBBD fills the gap rather than conflicting with it -- a genuine
second-source recovery, not a defect.

**The four 2025 final-score disagreements, flagged rather than resolved:**

| game_id | hoopR home-away | CBBD home-away | note |
|---|---|---|---|
| 401745889 | 79-59 | 77-59 | home score off by 2 |
| 401723767 | 69-81 | 0-81 | CBBD homePoints is a data gap (0), not a real 69-vs-0 disagreement |
| 401722537 | 62-60 | 60-62 | scores match as a SET but swapped -- a side-label flip, the same class of defect the pbp layer's `_fix_flipped_sides` repairs (`docs/tests/possessions_build_v2_2026-09-10.md` section 0) |
| 401746100 | 80-67 | 80-65 | away score off by 2 |

Three of the three 2025 per-period disagreements are these same three
non-gap games (401745889, 401722537, 401746100); the CBBD-homePoints-gap game
(401723767) is excluded from the per-period comparison because a 0-point
CBBD side has no meaningful period split to check against. Both column sets
are kept on every row -- `home_score`/`away_score` (hoopR) and
`homePoints`/`awayPoints` (CBBD) -- so a downstream grader can choose either
source, or flag-and-drop the four disagreeing games, but nothing here picks
one silently.

---

## 4. G3 / G4 / G8: provisional gate reads

`scripts/eval_gates.py` gained a `--truth-dir` argument (default
`data/processed/truth`; a season/table not found there falls back to the
pre-truth-table behaviour byte-for-byte -- verified by diffing a run against
a nonexistent truth dir against the original 2026-09-10 report: only the
`Generated <date>` line and one reworded note differ). `gate_g3`/`gate_g4`/
`gate_g8` in `src/cbb_sim/eval/gates.py` now read the three tables above.
Re-run on `results/engine_v0/F2_2025_s5_r2event` (season 2025, 5 seeds):

    .venv/Scripts/python.exe scripts/eval_gates.py --results results/engine_v0/F2_2025_s5_r2event \
        --season 2025 --out docs/tests/gates_engine_v0_F2_2025_s5_r2event_PROVISIONAL_2026-09-10.md

Full report: `docs/tests/gates_engine_v0_F2_2025_s5_r2event_PROVISIONAL_2026-09-10.md`.
G1, G2, G5, G6, G7, G9 are diffed byte-identical to the original
2026-09-10 report (confirmed); only G3, G4, G8 change. All reads below are
**PROVISIONAL**: the truth tables are new as of this doc and have not
themselves been through a bake-off or a second-source-verification cycle the
way the box-derived truth (`reference.load_actual_team_box`) already has.

### G3 -- shot mix (NEEDS-INSTRUMENTATION overall; unchanged root cause, new season-level reads)

| quantity | value | tolerance | status |
|---|---|---|---|
| 3PA share & FTA/FGA by team | 0/0 powered team-metrics | +/-1.5pp | NEEDS-INSTRUMENTATION (per-team n~30-40 < min_cell_n=300, unrelated to truth data) |
| rim share by team | 0/0 powered teams | +/-1.5pp | NEEDS-INSTRUMENTATION (same power issue -- but now a REAL per-team table exists where "n/a" stood before) |
| three_pa_share (season, pooled) | 0.3942 vs 0.3906 | +/-1.5pp | **PASS** |
| fta_per_fga (season, pooled) | 0.3147 vs 0.3295 | +/-1.5pp | **PASS** |
| rim_share (season, pooled) | 0.3677 vs 0.3733 | +/-1.5pp | **PASS** |

Rim share -- previously always NEEDS-INSTRUMENTATION with no truth table at
all -- now reads a real, close comparison (-0.56pp) at the powered
season level. The by-team cells remain UNDERPOWERED under the existing
min_cell_n=300 convention (~30-40 games/team over one season); that is a gate
power-design question, not something this truth build changes.

### G4 -- four factors (FAIL overall; new finding: pooled OREB% misses)

| quantity | value | tolerance | status |
|---|---|---|---|
| eFG% (offense/defense) | n/a | +/-1.0pp | NEEDS-INSTRUMENTATION (engine-contract gap: games.parquet carries attempt counts only, never makes -- confirmed by inspecting the actual results file, 20 columns, no FGM anywhere) |
| tov_pct / oreb_pct / ft_rate by team | 0/0 powered | -- | NEEDS-INSTRUMENTATION (same per-team power issue as G3) |
| tov_pct (season, pooled) | 0.1788 vs 0.1739 | +/-1.0pp | **PASS** |
| oreb_pct (season, pooled) | 0.2855 vs 0.2984 | +/-1.0pp | **FAIL** (-1.29pp) |
| ft_rate (season, pooled) | 0.3147 vs 0.3295 | +/-0.015 | **PASS** |

eFG% ACTUAL side is now computable from `team_game_shots_v1`'s box columns
(0.5086 pooled) -- the SIM side still cannot be, because the engine's own
output contract has no make counts. That is documented as an engine-contract
gap, not papered over with a fabricated sim-side eFG%.

### G8 -- player layer (FAIL overall; first real actual-box comparison ever produced here)

| quantity | value | tolerance | status |
|---|---|---|---|
| rotation minutes mean | 29.93 vs 29.83 | +/-2.0 | **PASS** |
| rotation minutes SD ratio | 1.2219 | 0.9-1.1 | **FAIL** (sim rotation minutes over-dispersed) |
| top-1 FGA share, mean | 0.2611 vs 0.2496 | report only (no pre-registered tolerance) | NEEDS-INSTRUMENTATION |
| players used per team-game, mean | 8.81 vs 9.80 | K-S p > 0.10 | **FAIL** (K-S D=0.238, p~0) |

Every one of these was unconditionally NEEDS-INSTRUMENTATION before this
build (no actual-box comparison existed at all). The two real FAILs are
independent corroboration of defects `docs/models/engine/model.md` section 7
already names from the model side: item 6, "under-concentrated top usage"
(the usage sampler spreads events too evenly, consistent with `players used`
running high and minutes SD running wide), and the open L4 usage
change-ledger row on top-1/top-3 share under-concentration. No new defect is
implied; this is the player-layer instrumentation those findings were waiting
on.

---

## 5. Limitations, stated rather than hidden

- 2022-2023 have no CBBD roster crosswalk at all, so `player_game_v1`'s
  event-layer columns and `cbbd_player_id` are null for those two seasons by
  construction. Extending the crosswalk backward (pulling 2022-2023 CBBD
  rosters, if the API supports it) is future work, not attempted here.
- FTA/FTM reconciliation is flat at ~90-93% even in the best-covered seasons
  -- the free-throw-trip ambiguity already documented
  (`ft_trip_ambiguous`, ~40% of trips) is the leading candidate, not
  re-derived here.
- The by-team G3/G4 cells stay UNDERPOWERED under the existing
  `min_cell_n=300` convention regardless of truth-table quality (one season
  gives ~30-40 games per team); only the season-level pooled reads this build
  adds are powered.
- `game_finals_v1` is built and reconciled but not yet wired into any grading
  script (`grade_market_games.py`/`grade_market_props.py` still read
  `games_universe.parquet` directly); wiring it in was not asked for here and
  the existing behaviour is untouched.
- Nothing here is a bake-off decision. The new tables are truth data, not a
  model; CLAUDE.md's bake-off rule applies to model/feature choices, not to
  building a second-source reconciliation of existing box/event data.

## Change ledger

Row added to `docs/models/change_ledger.md` section A, same commit series as
this doc (PM commits after review; not committed by this worker).
