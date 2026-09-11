# L3 FIELD-GOAL MAKE: features

Companion to `model.md` and `experiments.md`. Every number lives in
`experiments.md`; this file is provenance only.

Builder: `cbb_sim.models.fg_make.build_design`. Every as-of quantity is an
expanding mean over games **strictly before** the current one within the season
(`prob_metrics.expanding_asof`, which is `cumsum() - value`) and is then centred
on the **league's own as-of rate on the same date** — never on a season
constant, never on a raw level (`CLAUDE.md`, "every rating feature is expressed
relative to its own snapshot's league mean").

The three shot classes each get their own model, so every per-class column below
is **wide in the source table and selected per row for that row's own class**
(`_pick_class`). "This class" always means the class of the attempt in the row.

---

## 1. Source manifest

| Feature | Dtype | Source file | Computation | Fallback | Notes |
|---|---|---|---|---|---|
| `off_make_c` | float32 | `data/raw/cbbd/pbp/plays_{season}.parquet` via `event_stream.build_stream` -> `fg_make.build_fg_events` -> `team_shot_form` | offence's expanding make rate on this shot class over its games strictly before this one, minus the league's as-of rate on this date | **exactly 0.0** (= the league mean on a centred scale) when the team has no prior attempts of the class | The zero is a branch, not a subtraction: see section 4 |
| `def_allow_c` | float32 | same | the defence's expanding make rate ALLOWED on this class, centred the same way | exactly 0.0 | The second responsiveness driver: the pre-registration applies the matchup rule to the defence too |
| `off_rating_off_c` | float32 | `data/processed/ratings/own_ratings_{season}.parquet` via `ratings.own_ratings.join_as_of` | the offensive team's as-of offensive rating, already league-centred by that builder | 0.0 | Passed the INV-45 change-form leak test (ledger section B) |
| `off_rating_def_c` | float32 | same | the offensive team's as-of defensive rating | 0.0 | |
| `def_rating_off_c` | float32 | same | the defensive team's as-of offensive rating | 0.0 | |
| `def_rating_def_c` | float32 | same | the defensive team's as-of defensive rating | 0.0 | |
| `site_home` | float32 | `data/processed/games_universe.parquet` | 1 when the shooting team is home and the game is not neutral | 0.0 | Neutral is the reference level (`CLAUDE.md`: home/away/neutral is first-class in every scoring-stage model) |
| `site_away` | float32 | same | 1 when the shooting team is away and the game is not neutral | 0.0 | |
| `season_idx` | float32 | universe | `season - 2022` | — | The season-level term L11 requires be present and reported, not assumed to work |
| `shooter_make_c` | float32 | pbp `participant_1_id` (default) or `shot_shooter_id` (`build_fg_events(shooter_key=...)`, round 3) via `shooter_form` | the shooter's expanding make rate on this class over his games strictly before this one, minus the league as-of rate | **exactly 0.0** when he has no prior attempt of the class | The first responsiveness driver. Rows where it is undefined are reported as their own cell, never bucketed on the imputation. **`participant_1_id` is the ASSISTER on 47.7-50.2% of assisted made FGAs (`docs/tests/shooter_key_audit_2026-09-10.md`); every SERVED artifact (`winner_*`, `round2/`, `round2b/`) still uses it.** Round 3 (`experiments.md` sections 17-18) measured the correction and found the reference arm's `FGA_3` Decision-8 quintile span (35.9 pp) collapses to 2.8 pp once corrected -- NOT ADOPTED per the pre-registered rule; `docs/models/fg_make/model.md` item 8 |
| `shooter_att_c` | float32 | same | the shooter's attempts of this class to date, this season | 0.0 | Also the `defined` gate for the shooter responsiveness driver |
| `prior_season_make_c` | float32 | same | the shooter's COMPLETED previous season's make rate on this class, centred | exactly 0.0 when there is no prior season | Entirely in the past by construction |
| `has_prior_season` | float32 | same | 1 when the shooter has a completed prior season in the data | 0.0 | Makes the fallback above legible to a linear arm |
| `pos_G` / `pos_F` / `pos_C` | float32 | `data/raw/cbbd/rosters/roster_{season}.parquet` via `free_throw.load_positions` | roster position mapped to guard / forward / centre | all three 0 (`UNK` is the reference level) | Rosters exist for 2024-2026 only; the CBBD player id is stable across seasons, so a player on any of those rosters carries his position back to 2022. Coverage is a reported number |
| `shooter_games_asof` | float32 | `shooter_form` | games this season in which the shooter attempted a field goal, strictly before this one | 0.0 | Stand-in for `minutes-to-date`, section 3 |
| `shooter_fga_asof` | float32 | `shooter_form` | the shooter's total field-goal attempts to date this season, all classes | 0.0 | Same |
| `period` | float32 | pbp | period index | — | Kept alongside `seconds_remaining`, which is PER PERIOD in the feed (1200 in halves, 300 in overtime) and does not identify late-game without it |
| `seconds_remaining` | float32 | pbp | seconds left in the period at the attempt | — | |
| `score_diff` | float32 | pbp running score | score from the SHOOTING team's point of view | — | **POST-OUTCOME, CONFIRMED-DEFECT 2026-09-10.** The feed's score on the attempt's own row already includes the attempt's own points. ROUND 1 ONLY; section 5 |
| `score_diff_pre` | float32 | `fg_make.add_round2_state` | `score_diff` minus the attempt's own points when it went in: the margin BEFORE the attempt | — | The corrected column. Round 2 arms S-D/S-E; NOT in the adopted arm. Section 5 |
| `gt_flag` | float32 | `fg_make.add_round2_state` | garbage time: `abs(score_diff_pre) >= 15` AND at most 480 s left in regulation AND `period <= 2` | 0.0 | Thresholds fixed in the round-2 pre-registration from the step-1 evidence. Round-2 arm S-D only; NOT in the adopted arm |
| `eg_trail` | float32 | same | end game trailing: `-9 <= score_diff_pre <= -1` AND at most 120 s left AND `period <= 2` | 0.0 | Same |
| `eg_lead` | float32 | same | end game leading: `+1 <= score_diff_pre <= +9` AND at most 120 s left AND `period <= 2` | 0.0 | Same |
| `in_bonus` | float32 | `event_stream._attach_team_fouls` | the DEFENCE's prior personal-foul count in the period is at or past the one-and-one threshold | 0.0 | Thresholds are `possessions`' constants, themselves derived from the trip-length distribution (L18) |
| `chance_number` | float32 | `fg_make.chance_state` | 1 on a possession's first chance, +1 per offensive rebound before this attempt | 1.0 | Section 4 |
| `chance_elapsed_s` | float32 | `fg_make.chance_state` | seconds between the event that STARTED this chance and the attempt, clipped to 60 s | period length minus the clock, when the chance starts a period | Section 4 |
| `is_transition_f` | float32 | `fg_make.chance_state` | the chance started on a live defensive rebound or a turnover AND the attempt came within 8 s | 0.0 | NOT the banned L5 `is_transition`; section 4 |
| `def5_rim_allow_c` | float32 | CBBD pbp `onFloor` via `defender_rates` | mean of the five on-floor defenders' as-of rim make rate ALLOWED, each shrunk toward the league as-of rate with a FITTED number of pseudo-attempts, minus the league rate | 0.0, and the row is excluded from the lineup fold via `lineup_on_floor_ok` | 2024+ only (L13); `D_plus_lineup` fold only |
| `def5_three_allow_c` | float32 | same | the same for three-point attempts (perimeter defence) | 0.0 | |

### 1.1 Columns the design carries that are NOT features

Reported, or consumed by an arm's own algebra, and never in a feature matrix:
`lg_make_asof`, `off_make_raw`, `def_allow_raw`, `shooter_make_raw`,
`position_make_raw`, `prior_season_make_raw` (the EB arm's priors and its
league anchor), `shooter_mk_c`, `off_att_prior`, `def_att_prior` (the
denominators, which gate the responsiveness drivers), `is_transfer` (the L15
subset), `espn_athlete_id` (the L4 join), `position_group`, `chance_side_ok`,
`chance_elapsed_clipped` (derivation diagnostics), and `blocked` / `and_one`
(reported population rules, and both **banned**, section 3).

---

## 2. Feature sets tested

| Set name | Included features | Rationale |
|---|---|---|
| `A_team` | `off_make_c`, `def_allow_c`, the four own ratings, `site_home`, `site_away`, `season_idx` | The pre-registered team-level floor: what a model that has never heard of the shooter can do. Also the feature set of the `team_baseline` arm |
| `B_plus_shooter` | A + `shooter_make_c`, `shooter_att_c`, `prior_season_make_c`, `has_prior_season`, `pos_G/F/C`, `shooter_games_asof`, `shooter_fga_asof` | L15: player identity dominates every player-game rate stat. This is the block that tests whether it dominates a MAKE as well |
| `C_plus_state` | B + `period`, `seconds_remaining`, `score_diff`, `in_bonus`, `chance_number`, `chance_elapsed_s`, `is_transition_f` | The full bundle, and the one the `ridge` and `lgbm` arms are scored on |
| `D_plus_lineup` | C + `def5_rim_allow_c`, `def5_three_allow_c` | 2024+ only. Run on its own fold (train 2024, test 2025) to answer one question: is the defence a lineup-level quantity or a team-level one |

### 2.1 ROUND 2 feature sets (2026-09-10, `experiments.md` section 13)

Every round-2 set is `B_plus_shooter` plus a state block; the team and shooter
blocks are byte-identical across all five, which is what makes a log-loss
difference attributable to the state parametrisation and nothing else.
`fg_make.R2_FEATURE_SETS` is the source of truth and `feature_set()` resolves
these names alongside the round-1 ones.

| set name | arm | added to `B_plus_shooter` | status |
|---|---|---|---|
| `R2_A_round1_leaked` | S-A | round 1's `C_plus_state` block, **including the post-outcome `score_diff`** | INELIGIBLE, reference only |
| `R2_B_no_state` | S-B | nothing (identical to `B_plus_shooter`) | rejected -- fails calibration on the jumper and Decision 8 on the three |
| **`R2_C_safe_state`** | **S-C** | `period`, `seconds_remaining`, `in_bonus`, `chance_number`, `chance_elapsed_s`, `is_transition_f` | **ADOPTED, all three classes, rounds 2 and 2b** |
| `R2_D_safe_plus_indicators` | S-D | S-C + `gt_flag`, `eg_trail`, `eg_lead` | rejected -- 0.04-0.64 floors from S-C, inside the floor, simpler arm wins |
| `R2_E_safe_plus_continuous` | S-E | S-C + `score_diff_pre` | rejected -- inside the floor on rim and jumper, fails Decision 8 on the three |

None of the five columns of section 1's round-2 block is in the adopted set.
They are kept because the evidence that produced them is real (`gt_flag` is
+4.79 pp on rim attempts and flat across a 3x3 threshold grid; `eg_trail` is
-7.78 pp on threes) and because a later round -- or the possession-outcome and
clock models, which own the end-game SHOT MIX that carries most of that effect
-- may want them. `add_round2_state` builds all five unconditionally so the
comparison stays re-runnable.

The four sets are strictly nested, and `tests/test_fg_make.py` asserts it, so a
gain between two of them is the block's own contribution and nothing else.

---

## 3. Rejected features, and why

| Candidate | Status | Reason |
|---|---|---|
| **assisted flag** | REJECTED by the pre-registration, and NOT BUILT | In this feed an assist is logged as a property of a MADE basket, so the flag is knowable only after the ball goes in. It would be a near-perfect predictor of its own target. It is not derived, not stored, and `fg_make.BANNED_FEATURES` names it so a later reader cannot add it by accident |
| **`blocked`** | BANNED | A `Block Shot` row exists only because the attempt missed. It is carried on the attempt (the pre-registration's "blocked shots are misses" rule is checked with it) and it can never enter a feature matrix: `design_matrix` raises on it |
| **`and_one`** | BANNED | The foul that turns a make into an and-one is logged after the ball goes in. Same treatment: derived for the population check, refused as a feature |
| **`minutes-to-date`** | PRE-REGISTERED, COULD NOT BE BUILT, substituted | Minutes live in hoopR `player_box` keyed on the ESPN athlete id. The CBBD -> ESPN crosswalk exists only for 2024-2026 (CBBD rosters were pulled for those seasons only), so it resolves 0% of two of the three training seasons, and CBBD `onFloor` cannot substitute because it is empty at the source before 2024 (L13). A column that is real in 2024-2025 and structurally zero in 2022-2023 is a season dummy wearing a minutes label. `shooter_games_asof` and `shooter_fga_asof` carry the exposure instead; the ESPN-id coverage per season is reported in `experiments.md` section 2.1 so the substitution is auditable. When the L4 player layer lands, minutes are the obvious first addition and need their own pre-registration |
| **`score_diff` (the round-1 column)** | **BANNED IN PRACTICE from 2026-09-10, not yet in `BANNED_FEATURES`** | It is POST-OUTCOME: `_season_events` reads `homeScore`/`awayScore` on the attempt's own row and that column is the score AFTER the play, so a made three already carries its own three points. Proved three independent ways off the raw feed in `docs/tests/fg_make_state_confound_2026-09-10.md` section 0 (playText; the shooting team's own-row score delta equalling the shot's value on 93.6-94.5% of makes against 99.5% of misses moving by zero; the collapse of the effect when the shot's own points are removed). It manufactures 62.0 / 83.3 / 83.5% of the apparent margin effect and, because the ENGINE always fed the correct pre-shot margin, round 1 also shipped a train/serve skew -- the mechanism behind L23. It is deliberately NOT in `BANNED_FEATURES` because round 2's arm S-A has to reproduce round 1's numbers exactly; adding the ban is a PM decision and would break `train_fg_make_v1.py`. The adopted round-2/2b arm carries no margin term at all |
| **shot coordinates / release distance** | NOT USED AS A FEATURE | `shot_location_x/y` is populated on 78-88% of shooting rows in 2022-2024 and 97-99% in 2025-2026 (`cbb_sim.pbp.events`), so a distance feature would carry a season-shaped missingness pattern straight into a model whose whole problem is season drift (L11). Location is used at the EVENT layer only, to repair the L16 mistag, where a missing coordinate simply leaves the feed's own label alone |
| **defender identity / closest defender** | NOT AVAILABLE | The feed names the shooter, the rebounder and the fouler. The five on-floor defenders are known (2024+) and enter as `D_plus_lineup`; which of them contested the shot is not in any source we have |
| **the shooter's as-of rate on the OTHER two classes** | NOT PRE-REGISTERED | A rim-shooting big man's three-point rate is a style signal, not a skill signal for the shot he is taking. Adding it is a defensible experiment and would need its own pre-registration |
| **explicit offence x defence interaction products** | NOT PRE-REGISTERED here | L3 round 1 measured them at 240x BELOW its noise floor on the possession-outcome target (ledger section B) and rejected them on both linear arms; the tree arm represents interactions natively |

---

## 4. The chance-state block, and why it is not the banned `is_transition`

The change ledger records `is_transition` as a CONFIRMED-DEFECT at L3 and as
BANNED OUTRIGHT at L5. The definition it bans is
`duration_s <= 8 AND start_reason in {DREB, TOV}`, where `duration_s` is the
chance's own duration — i.e. a function of when the chance **ends**. At L5 that
is the target itself; at L3 it is contemporaneous with the terminal event being
predicted.

This model's version is a different quantity built from different rows.
`chance_state` walks the event stream and records, for every row, the event that
**started** the current chance (a made basket, a made last free throw, a
defensive rebound, a turnover, a dead-ball rebound or a period boundary ends the
previous one; an offensive rebound ends the chance but keeps the possession,
which is what `chance_number` counts). `chance_elapsed_s` is then the gap
between that start event and the attempt's own clock — the time on the ball when
the shooter released it. Every input is at-or-before the release, and the
attempt's own class and outcome are excluded from its own chance by construction
(`cumsum() - self`).

Three things make that claim checkable rather than rhetorical:

* `tests/test_fg_make.py::test_flipping_an_attempts_outcome_cannot_move_its_own_chance_state`
  flips an attempt's `made` flag and requires chance number, elapsed, start
  reason and transition flag to be bit-identical.
* The derivation is validated against the feed: the offence implied by the
  chance's own start event is compared with the shooting team's own side on
  every attempt, and the match rate is reported in `experiments.md` section 2.2.
* Elapsed time is clipped at 60 s (a longer gap means a missing chance-start
  event, not a 61-second possession) and the share that hits the clip is
  reported rather than silently absorbed.

The centring convention is worth one more line, because it caused the only real
bug in this build. Every centred rate is constructed as
`where(the rate exists, rate - league, 0.0)`. The zero branch is not a
convenience: "no prior attempts" means the feature's value **is** the league
mean, which on a centred scale is exactly 0.0. Writing it as a subtraction
instead leaves a float32 rounding residue of ~1e-8 in a column that is supposed
to be constant; a standardiser then divides by that residue, turns a
zero-information column into a 1e7-sized input and the linear arms diverge. That
is what the first smoke run of this trainer did. Both ends are now fixed — the
branch here, and a relative constant-column threshold in `RidgeArm`.

---

## 5. The `score_diff` leak, and why the correction is itself safe

`_season_events` builds `score_diff` as
`np.where(off_home, hs - as_, as_ - hs)` from the stream's `home_score` /
`away_score` on the ATTEMPT'S OWN ROW. In the CBBD/ESPN feed that column is the
score after the play: on 93.6-94.5% of made field goals in 2022-2025 the
shooting team's own score moves by exactly the shot's value on that row, against
99.5% of missed attempts where it does not move at all.

That makes the round-1 `C_plus_state` block carry a post-outcome column of
exactly the kind this file's section 3 bans by name, and it is the first-order
cause of L23: margin SD 34.6 and a home/away score correlation of -0.64 in the
engine, because the engine correctly feeds the PRE-shot margin and the model had
learned a coefficient that partly means "this shot went in".

The correction, `fg_make.add_round2_state`:

```
score_diff_pre = score_diff - (3 if a made three else 2 if a made two else 0)
```

is safe in the one direction that matters. A MISS is unchanged by construction,
so the correction never injects outcome information; it only removes the
information the feed had already injected. The three indicators are then
functions of `score_diff_pre`, the period and the clock alone.

On the simulated side `cbb_sim.engine.loop._state_block` writes the live
pre-shot margin into `score_diff_pre` and computes the three indicators from
`fg_make`'s own constants and its own `regulation_seconds_remaining`, so the
trained and simulated definitions cannot drift;
`tests/test_engine.py::test_round2_state_definitions_agree_between_training_and_the_engine`
pins them bit for bit.

Full evidence, including the confound decomposition and the threshold grids:
`docs/tests/fg_make_state_confound_2026-09-10.md`.
