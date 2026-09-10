# L5 clock consumption — feature inventory

Companion to `model.md` and `experiments.md`. Every column below is built by
`cbb_sim.models.clock.build_design`, which is the ONLY feature builder for this
model — the trainer, the tests and the sim all call it, so there is no second
implementation to drift.

Provenance conventions follow `docs/models/DOCUMENTATION_STANDARD.md`:
source file → computation → fallback → why it is leak-free.

---

## 0. The population

| filter | source | effect |
|---|---|---|
| D-I game, hoopR feed not truncated | `data/processed/games_universe.parquet` (`is_d1_game & ~pbp_truncated`) | the project-wide universe |
| CBBD-complete | `cbb_sim.models.clock.cbbd_complete_games` — the possession table's own `points + tech_points_off` must equal the schedule's final score for BOTH teams | drops the games whose CBBD event stream is short of the final score (`docs/tests/possessions_build_2026-09-10.md` section 2; 19% of 2022-2023 games). `games_universe.pbp_truncated` is a hoopR-side flag and does not catch these |
| `duration_s <= DURATION_CAP` (90 s) | the possession row | drops ~0.01% of rows whose "duration" is a CBBD feed gap (a period whose first logged event is also its last, giving a possession stamped with the full period length). Excluded, never clipped |
| season in the fold | `cbb_sim.data.seal.assert_not_sealed` on both slices | season 2026 can never enter a fold |

Clock accounting is reported as a diagnostic and NOT used as a filter: the
signed deviation of a half's summed possession durations from 1200 s is
−0.75 s on 2025 (0.06% of a half). Filtering on it at ±2 s would remove two
thirds of the universe to fix an effect two orders of magnitude below the G1
tolerance.

---

## 1. Target and censoring

| column | source | notes |
|---|---|---|
| `duration_s` | `possessions_{season}.parquet` | `start_clock − end_clock`, clipped at 0 by the builder. Integer seconds, support 0..90 |
| `censored` | `terminal_event == "end_period"` | RIGHT-CENSORED: the possession ended at the horn, so the true duration is strictly larger. 0.22% of rows. Every arm states its handling (module docstring of `clock.py`) |

---

## 2. A_state — pre-registered: previous end type, period, seconds remaining, chance number

| feature | source column | computation | fallback | leak-free because |
|---|---|---|---|---|
| `prev_end_DREB`, `prev_end_TOV`, `prev_end_made_FG`, `prev_end_made_FT`, `prev_end_other` | `start_reason` | one-hot; `period_start` is the reference level | none — `build_design` raises on an unmapped `start_reason` rather than bucketing it into `other` | `start_reason` is set by the possession builder from the PREVIOUS possession's terminal event (`possessions.py::_close(reason_next=...)`), i.e. it is known before this possession's first event |
| `period` | `period` | as float | none | period is a property of the game clock |
| `is_ot` | `period` | `period >= 3` | none | same |
| `seconds_remaining` | `start_clock` | seconds left in the period at the possession's first event | none | the clock at the possession's start |
| `chance_number_at_start` | constant | identically 1.0 | n/a | **ZERO VARIANCE.** A possession begins on its first chance by the segmentation rule, so the chance number at a possession's start carries no information. Kept in the list so the pre-registered bundle is visible, dropped by every arm as a zero-variance column, and the drop is printed by the trainer and recorded in `experiments.md`. See `model.md` section 9 |

---

## 3. B_plus_teams — A + tempo priors + ratings + site

| feature | source | computation | fallback | leak-free because |
|---|---|---|---|---|
| `off_tempo_rel`, `def_tempo_rel` | `cbb_sim.models.pace.build_pace_table` → `home_tempo_rel` / `away_tempo_rel`, assigned to the offence/defence by `offense_is_home` | as-of own-ridge tempo ratio to the as-of league mean | 1.0 (which IS the league mean on a ratio scale, not a fabricated level) | `own_ratings.join_as_of`'s window is built strictly before the game's own date; leak-tested at `data/processed/ratings/own_ratings_leak_test.csv` (L9) |
| `tempo_prior_game` | same | `home_tempo_rel × away_tempo_rel × league_tempo_mean_asof` — the **L2 bake-off winner verbatim** (`pace.MultiplicativeModel`), used exactly as `ARCHITECTURE_DECISIONS.md` Decision 7 says it must be: a pregame PRIOR FEATURE, never a sampler | the as-of league tempo mean | same as above |
| `off_rating_off_c`, `off_rating_def_c` | `own_ratings.join_as_of(team_col="offense_team_id")` | league-centred as-of offensive / defensive efficiency of the team with the ball | 0.0 (the league mean on a centred scale) | as-of join, strictly before |
| `def_rating_off_c`, `def_rating_def_c` | same, `defense_team_id` | the same two ratings for the team defending | 0.0 | as-of join, strictly before |
| `site_home`, `site_away` | `games_universe.neutral_site` × `offense_is_home` | from the OFFENCE's point of view; neutral is the reference level | none | schedule metadata |

Home/away/neutral is a first-class feature here per the `CLAUDE.md` modelling
rule ("audit each model's feature list for it").

---

## 4. C_plus_score — B + score diff, its interaction with seconds remaining, bonus

| feature | source | computation | leak-free because |
|---|---|---|---|
| `score_diff` | `start_score_diff` | offence score − defence score from the feed's running score AFTER the previous possession's terminal event | it is the score before this possession's first event |
| `x_score_diff__seconds_remaining` | derived | `score_diff × seconds_remaining / 1200` — the scale divisor is fixed, never fitted | both inputs are known at the possession's start |
| `in_bonus` | `off_in_bonus` | the DEFENCE's period foul count ≥ 6 at possession start, the NCAA threshold derived and validated in `possessions.py` | foul counts are accumulated over strictly earlier events |

---

## 5. D_plus_season — C + season index and days since season start

| feature | source | computation | why it is here |
|---|---|---|---|
| `season_idx` | `season − 2022` (`pace.SEASON_INDEX_ANCHOR`, shared with L2 so folds share one scale) | float | L4/L11: pooled-season training with no season term under-shoots the current year |
| `days_since_start` | `pace.build_pace_table` → `days_since_start` | days from the season's first game | L5 (the learning): pace is ~3 possessions faster in November than in February, so a within-season progress term is a candidate, not an assumption |

---

## 6. Banned columns — present in the possession table, never a feature

`cbb_sim.models.clock.BANNED_FEATURES`, asserted by
`tests/test_clock.py::test_no_banned_column_is_ever_a_feature` against every
feature set AND against every fitted arm's stored feature list:

```
terminal_event, end_clock, duration_s, n_chances, oreb_count, points,
fga_rim, fgm_rim, fga_jump2, fgm_jump2, fga_3, fgm_3, fta, ftm, and_one,
stolen, ft_trip_ambiguous, is_transition, tech_points_off, tech_points_def,
censored
```

Two of these deserve naming:

- **`is_transition`** is defined by the possession builder as
  `duration_s <= 8 AND start_reason in {DREB, TOV}`. It is a FUNCTION OF THE
  TARGET. `docs/models/change_ledger.md` already carries it as a
  CONFIRMED-DEFECT for L3 because it is contemporaneous with the outcome it
  predicts there; for L5 it is a perfect leak, so it is banned outright and the
  test proves it never appears.
- **`terminal_event`** is banned by the pre-registration's own design: the
  engine draws the duration FIRST and the terminal event SECOND, so the
  terminal event does not exist when this model is called. The pre-registered
  "mean and SD of duration by terminal event class" diagnostic exists precisely
  to check how much of that unseen information the state recovers on its own.

---

## 7. Reporting cells (not features)

`prev_end`, `secs_bucket` (`0-34 / 35-119 / 120-299 / 300-599 / 600+`),
`score_bucket`, `pit_cell` (`prev_end | secs_bucket`) and `month` are carried
on the design table for the pre-registered PIT-by-cell table, the empirical
arm's cell grid, and the by-month G1 read. They never enter a model matrix.

---

## 8. Round-2 state representation (pre-registration section 5, 2026-09-10)

Round 1's diagnosis was that `seconds_remaining` was represented too coarsely
near a period boundary (`experiments.md` section 3.1). Round 2 replaces that
representation. Everything else — target, universe, censoring, the banned list,
the team and score blocks — is unchanged, so only the changed columns are
documented here. Built by `cbb_sim.models.clock.add_r2_state`, which is called
once by `build_design` and again by `apply_clock_override` for every simulated
second.

### 8.1 The new columns

| feature | computation | why |
|---|---|---|
| `sr_bucket_id` | `searchsorted` on edges `(0, 5, 10, 20, 30, 45, 60, 90, 150, 300, 1201)` → 0..9 | ten FINE buckets instead of round 1's five, four of them inside the last 45 seconds where the conditional mean actually bends |
| `period_type` | `0` if period == 1 else `1` | {first half, second half/OT}, so end-of-first-half and end-of-game are separate states. OT is POOLED with the second half exactly as the pre-registration writes it; the consequence (a 300-second OT and a 300+-second second half share a bucket) is recorded rather than silently patched |
| `score_state` | `0` trailing / `1` tied / `2` leading, from `score_diff` | makes the late TRAILING team's short possessions representable — a trailing team fouls and shoots quickly, a leading team runs clock, and round 1 could express neither |
| `srx_{bucket}__{period_type}__{score_state}` | one-hot of the crossed factor, 60 levels, level 0 (`srx_0-4__H1__trailing`) dropped as the reference | the crossing is the point: the bend is not the same shape at the end of the first half as at the end of the game, nor for a trailing team as for a leading one |
| `last_shot_window` | `seconds_remaining <= 30` | one shot clock: below it the offence can hold for the last shot |
| `two_for_one_window` | `30 < seconds_remaining <= 45` | the two-for-one window, where the offence shoots EARLY to get the ball back |

`last_shot_window` and `two_for_one_window` are nearly, but not exactly,
functions of `sr_bucket_id` (the boundaries at 30 and 45 seconds fall inside
buckets `30-44` and `45-59`). They are near-collinear with the crossed dummies,
which leaves a flat direction in the parametric arms' likelihood; the fitted
VALUES are still identified, and the arms record whether L-BFGS converged. They
are included because the pre-registration names them.

### 8.2 The two encodings

| feature set | who uses it | contents |
|---|---|---|
| `R2_dummy` (78 cols) | `empirical` (as a cell grid), `lognormal`, `gamma`, `hazard` | previous-end dummies + the 59 crossed dummies + the two window flags + the round-1 team block + `score_diff`, `x_score_diff__seconds_remaining`, `in_bonus`. The raw `seconds_remaining` main effect and the `period`/`is_ot` columns are REPLACED by the crossed factor |
| `R2_tree` (23 cols) | `lgbm_quantile` | the same, but with raw `seconds_remaining` plus `sr_bucket_id`, `period_type` and `score_state` as ordered integers instead of 59 one-hot columns — a tree splits on an ordered id by itself, and 59 one-hots would only make the splits harder to find. This is what the pre-registration asks for ("the tree arm takes the raw seconds_remaining plus the bucket id") |

`chance_number_at_start` is gone from both: round 1 measured it as
zero-variance at a possession's start and the finding is recorded, so carrying
it forward would be ceremony.

### 8.3 The round-2 empirical cell grid

`prev end type (6) × fine bucket (10) × period type (2) × score state (3) ×
tempo tercile (3)` = **1,080 cells**, in that order, exactly as pre-registered.
Minimum cell size 300 training rows; the hierarchical fallback drops dimensions
from the RIGHT (tempo tercile first, then score state, then period type, then
the fine bucket, then down to the global cell), so the fine bucket — the thing
round 2 exists to add — survives longer than any coarser dimension. The share
of test rows served at each fallback level is reported by the trainer.

### 8.4 Leak safety of the new columns

Every round-2 column is a deterministic function of `seconds_remaining`,
`period` and `score_diff`, all three of which are known at the possession's
start and all three of which round 1 already established as leak-free. The
column that would be a NEW leak channel is a stale one: if the emergent
simulator carried the REAL possession's bucket instead of recomputing it from
the simulated clock, the sim would be reading the answer. `apply_clock_override`
is the single place that recomputes them, and
`tests/test_clock.py::test_chained_sim_does_not_inherit_the_real_clock_bucket`
proves it end to end by corrupting every round-2 clock column in the design to
a constant and requiring the chained possession counts to be bit-identical.
