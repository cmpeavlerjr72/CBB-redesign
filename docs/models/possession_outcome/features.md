# L3 POSSESSION-OUTCOME: features

Companion to `model.md` (what and why) and `experiments.md` (the grid). Builder:
`cbb_sim.models.possession_outcome.build_design`. Event layer:
`cbb_sim.pbp.possessions` / `cbb_sim.pbp.events`, validated in
`docs/tests/possessions_build_2026-09-10.md` (v1) and
`docs/tests/possessions_build_v2_2026-09-10.md` (v2).

**Round 2 changed where two things come from, and section 1.1 is the record of it.** The style rates
are now built from FIRST chances only rather than from the possession table, and the event layer they
read carries the rim-location override. Both are data fixes upstream of the model; neither is an
adjustment to its output. The round-1 source is still reachable
(`build_design(style_source="all_chances")`) so that `scripts/train_possession_outcome_v1.py`
reproduces its own recorded numbers exactly.

**Round 3 (2026-09-11) added three candidate columns and adopted NONE of them on the `first`
population.** `is_conf_game` (the conference-game flag, Decision 9b) and the opponent-adjusted
style rates `off_*_a1` / `opp_def_*_a1` (one pass) and `off_*_a3` / `opp_def_*_a3` (alternating
least squares) are built by `cbb_sim.features.conference` and
`cbb_sim.features.opponent_adjust` and are present in
`data/processed/models/possession_outcome/round3/design_v3.parquet`, but the feature ladder's best
arm gained 0.000225 log loss against a noise floor of 0.000804, so `F0` -- the round-2 bundle
below -- stands unchanged. All thirteen columns PASS the change-form leak test (|as-joined corr|
0.003-0.023 against a 0.15 gate); `experiments.md` section 7.5. `own_ratings`' `off_c` / `def_c`
are ALREADY opponent-adjusted (a jointly fitted offence/defence ridge) and were deliberately not
re-adjusted.

Three rules govern every row of the table below and are worth stating before it:

- **Strictly-before, always.** Every team-form feature is an expanding mean over that team's own
  games *before* the current one, built by `cumsum() - own_value` inside `(season, team_id)` ordered
  by `(game_date, game_id)`. A game's own events cannot enter its own features;
  `tests/test_possession_outcome.py::test_a_games_own_events_cannot_change_its_own_features` proves
  it by corrupting the game's own rows and requiring the feature to be bit-identical.
- **Centred, never raw.** Every rate is a deviation from the *league's* as-of mean on the same date
  (`CLAUDE.md`: "Every rating feature is expressed relative to its own snapshot's league mean. Raw
  levels are banned"; L2 is the measured cost of not doing this). A team with no prior games gets
  exactly `0.0`, which IS the league mean -- not an imputed guess.
- **A feature may not be built from the population it predicts, or from a population whose labels are
  known to be defective.** Added in round 2, and it is the rule the old `off_rim_c` broke: it was
  built over the possession table, which sums attempts across continuation chances, so an upstream
  labelling defect in 2025's continuation chances moved a predictor of the `first` population --
  whose targets that defect cannot touch. Section 1.1.

---

## 1. Source manifest

| Feature | Dtype | Source file | Computation | Fallback | Notes |
|---|---|---|---|---|---|
| `off_3pa_c` | float32 | FIRST chances of `data/processed/possessions_v2/chances_{season}.parquet` | `100 * sum(fga_3) / sum(poss)` over the offence's earlier games this season, minus the same ratio over all earlier league games. `poss` is the count of first chances, which IS the possession count | `0.0` (= league mean) | three-point attempts per 100 possessions. hoopR `team_box` carries an independent version of this stat; section 1.1 says why the event source is used anyway |
| `off_rim_c` | float32 | same | `100 * sum(fga_rim) / sum(fga)`, centred | `0.0` | rim share of first-chance field-goal attempts. **hoopR `team_box` cannot express this at all** -- it carries no rim/jumper split -- so this rate has to come from events, and it is also the rate the 2025 defect attacked |
| `off_tov_c` | float32 | same | `100 * sum(tov) / sum(poss)`, centred; `tov` counts FIRST chances with `terminal_event == 'TOV'` | `0.0` | turnover rate |
| `off_ftr_c` | float32 | same | `100 * sum(fta) / sum(fga)`, centred | `0.0` | free-throw rate |
| `opp_def_3pa_c` | float32 | same | the identical four ratios computed over what the DEFENCE's earlier opponents did to it, centred | `0.0` | defence-allowed 3PA rate |
| `opp_def_rim_c` | float32 | same | as above | `0.0` | defence-allowed rim share |
| `opp_def_tov_c` | float32 | same | as above | `0.0` | defence-allowed turnover rate |
| `opp_def_ftr_c` | float32 | same | as above | `0.0` | defence-allowed FT rate |
| `off_rating_off_c` | float32 | `data/processed/ratings/own_ratings_{season}.parquet` | `own_ratings.join_as_of(team=offense_team_id, date=game_date)` -> `off_c` | `0.0` | own ridge offensive rating, points/100 relative to the as-of league mean |
| `off_rating_def_c` | float32 | same | -> `def_c` for the offence team | `0.0` | the offence team's own defensive rating (carried because a team's style correlates with both ends) |
| `def_rating_off_c` | float32 | same | `join_as_of(team=defense_team_id)` -> `off_c` | `0.0` | the defence team's offensive rating |
| `def_rating_def_c` | float32 | same | -> `def_c` for the defence team | `0.0` | the defence the offence is facing |
| `site_home` | float32 | `data/processed/games_universe.parquet` | `(not neutral_site) and offense_is_home` | `0.0` | neutral site is the REFERENCE level |
| `site_away` | float32 | same | `(not neutral_site) and not offense_is_home` | `0.0` | home/away/neutral is a first-class feature in every scoring-stage model (`CLAUDE.md`) |
| `season_idx` | float32 | derived | `season - 2022` | n/a | the season-drift term L11 says outcome models need |
| `days_since_start` | float32 | `games_universe.game_date` | days since that season's first D-I game date | n/a | pace and shot mix drift within a season (L5) |
| `period` | float32 | `possessions_v2/chances_{season}.parquet` | `period` | n/a | 1-2 regulation, 3+ overtime |
| `seconds_remaining` | float32 | same | `start_clock`, seconds left IN THE PERIOD at the chance's start | n/a | per-period clock: 1200 in periods 1-2, 300 in overtime |
| `score_diff` | float32 | same | `start_score_diff`, offence minus defence at the chance's start | n/a | offence's point of view, so it is symmetric across the two teams |
| `in_bonus` | float32 | same | `off_in_bonus`: the defence has committed >= 6 fouls this period, so the next common foul is the 7th and shoots | n/a | derived by counting, validated against the NCAA thresholds in the possessions build report |
| `is_transition` | float32 | same | `duration_s <= 8` and the possession started on a defensive rebound or turnover | n/a | a proxy, labelled as one |
| `chance_number` | float32 | same | `1` for the first chance, `2, 3, ...` after each offensive rebound | n/a | only used in the `cont` population -- it is identically 1 in `first`, and a constant column is not a feature |
| `x_*__*` | float32 | derived | elementwise products, see section 2 | n/a | linear arms only |

Target: `terminal_event` of the chance, mapped to the fixed class order
`(TOV, FGA_rim, FGA_jump2, FGA_3, FT_trip_shooting, FT_trip_bonus)`. Rows with
`terminal_event` in `{end_period, unknown}` are dropped -- `end_period` belongs to the clock model
(L5) per the pre-registration, `unknown` is the mismatch-guard data gap and is never imputed.

Universe, from round 2 on: `is_d1_game & ~pbp_truncated & pbp_complete`
(`build_design(require_pbp_complete=True)`). The `pbp_complete` restriction is applied to the design
AND to the games the as-of rates are accumulated over, because a rate built partly from games whose
event stream is short of the box score is a rate of a different quantity. It costs 11.5% of the
round-1 rows, almost all of them in 2022 and 2023 (`cbb_sim.data.universe`, module docstring;
per-season shares in `docs/tests/possessions_build_v2_2026-09-10.md` section 2.1).

## 1.1 Which source each rate uses, and why

The round-2 pre-registration offered two ways to decontaminate the style rates: build them from
first-chance events only, or take them from hoopR `team_box` where the stat exists. The four rates
were resolved individually rather than as a block, and all four landed on the same answer for
reasons that differ by rate:

| Rate | Does hoopR `team_box` carry it? | Source used | Why |
|---|---|---|---|
| `rim` (rim share of FGA) | **No.** `team_box` has no rim/jumper split. Its nearest column, `points_in_paint`, is points rather than attempts, and the paint is not the rim | first-chance events | There is no alternative. This is also the one rate the 2025 defect actually moved, so it is the rate the decontamination is for |
| `3pa` (3PA / 100 poss) | Yes (`three_point_field_goals_attempted`) | first-chance events | See below |
| `tov` (TOV / 100 poss) | Yes (`turnovers`) | first-chance events | See below |
| `ftr` (FTA / FGA) | Yes (`free_throws_attempted`, `field_goals_attempted`) | first-chance events | See below |

**Why the three available-from-`team_box` rates still come from events.** Three reasons, in order of
weight:

1. **A mixed-source bundle is not a bundle of comparable features.** `rim` has to come from events.
   If the other three came from `team_box`, the offence's four style rates would be three
   whole-game box rates and one first-chance event rate, and the model's coefficient on `off_rim_c`
   would be conditioned on a differently-scoped set of neighbours. The point of the change is that
   every style rate describes the same population as the chance being predicted.
2. **`team_box` is whole-game and cannot be restricted to first chances**, so it carries exactly the
   contamination the change exists to remove -- just from a different feed. A 2025 putback is in
   `team_box`'s 3PA/FGA/TOV totals as surely as it is in the possession table's.
3. **The first-chance rate is still measuring the same underlying team style**, so preferring it
   costs nothing that matters. Measured by `scripts/diag_style_rate_sources.py` per team-game over
   `is_d1_game & ~pbp_truncated` games (10,564-11,433 team-games a season), written to
   `data/processed/models/possession_outcome/round2/style_rate_sources.json`, and stated here
   because it is the one place the claim could be misread as stronger than it is:

   | | correlation with the hoopR `team_box` rate |
   |---|---|
   | all-chances event rate | 0.994-0.997 in every season, on all three of `3pa`, `tov`, `ftr` |
   | first-chance event rate | 0.947-0.961 in every season, on all three |

   The first-chance rate is **visibly less correlated with the box score, and it is supposed to be**.
   The all-chances rate is nearly an identity with the box rate because it is the same quantity
   computed from a different feed -- that near-1.0 is a data-agreement check, not a quality score.
   The first-chance rate deliberately measures a sub-population (86-87% of chances) with a different
   composition, so a correlation of ~0.95 over ~11,000 team-games is the evidence that wants
   collecting: the two move together strongly, which says the first-chance rate ranks teams' styles
   the way the whole-game rate does, while remaining a different number.

**What the first-chance rate is NOT.** It is not an estimate of the whole-game rate. A first-chance
rim share is systematically lower than an all-chances one, because continuation chances are putbacks
and putbacks are rim attempts (`docs/tests/possessions_build_v2_2026-09-10.md` section 3). That is
the intended difference and not an error: the `first` model wants a first-chance descriptor, and the
`cont` model wants a predictor its own labels cannot have written.

**The `cont` population uses the same first-chance rates.** Not a first-chance rate for `first` and a
continuation-chance rate for `cont`: a continuation-chance style rate would be built from exactly the
labels the `cont` model is graded on, which is the circularity this rule was added to prevent.

## 2. Feature sets tested

| Set name | Included features | Rationale |
|---|---|---|
| `A_team` | the 8 form rates + the 4 own-rating columns + `site_home`, `site_away` | The pure matchup bundle: who is playing whom, and where. This is the set that has to demonstrate responsiveness -- if the model cannot slope with the offence's own prior rates, nothing downstream will. |
| `B_plus_season` | A + `season_idx`, `days_since_start` | L11: fold-2 train mean total 141.8 vs test 145.5, and the Control inherited 77% of the drift because its spec deliberately had no season term. This set is the pre-registered test of whether an explicit season term recovers it. |
| `C_plus_state` | B + `period`, `seconds_remaining`, `score_diff`, `in_bonus`, `is_transition` (+ `chance_number` in `cont`) | Game state changes shot selection mechanically -- a bonus possession can end in free throws that a non-bonus one cannot, late-clock possessions shoot more threes, transition possessions get to the rim. If these do not help, the engine can carry a simpler lookup. |
| `D_plus_interactions` | C + `off_3pa_c x opp_def_3pa_c`, `off_rim_c x opp_def_rim_c`, `off_tov_c x opp_def_tov_c`, `off_ftr_c x opp_def_ftr_c`, `off_rating_off_c x def_rating_def_c` | Linear arms only. A tree finds interactions itself, so giving them to LightGBM would make `D` a duplicate of `C`; the pre-registration says so explicitly. |

## 3. Rejected features

Recorded per the pre-registration ("State features that do not reduce log loss beyond the floor are
recorded as rejected in features.md"). The measured numbers are in `experiments.md` section 3.5 and
in `data/processed/models/possession_outcome/rejected_blocks.json`; they are not duplicated here,
per the documentation standard's "don't copy-paste numbers between files".

Deliberately excluded from every set, with the reason:

| Considered | Excluded because |
|---|---|
| Lineup / on-floor features (the ten `on_floor_*` ids the possessions table carries from 2024) | The pre-registration scopes this bake-off to team level. Lineups exist only from 2023-24 (L13), so including them would shrink the training window from four seasons to two and confound the model-class comparison with a sample-size change. They belong to L4. |
| Shot coordinates (`shot_location_x/y`) as a MODEL FEATURE | Still excluded, and for the original reason: they are 78-88% populated before 2025 and 97-99% after, so a coordinate feature would change meaning across the fold boundary. Note that this is not the same question as the rim-location OVERRIDE added in round 2 (`cbb_sim.pbp.events`), which uses coordinates in the EVENT LAYER to repair a vendor mislabel and can only ever move a two-point jumper toward `FGA_rim`. A label repair applied to every season alike and a predictor whose coverage jumps mid-window are different things; the first is a data fix, the second would be a fold artefact. |
| CBBD season ratings (`/ratings/adjusted`, `/srs`, `/elo`) | End-of-season snapshots with no date field: banned as pregame features by L7 and the SIM_GUARDRAILS meta-gates. |
| KenPom `adj_o`/`adj_t` | Available and leak-tested, but the L1 anchor bake-off (`docs/models/control_engine/experiments.md`) found our own ridge ratings tie centred KenPom within seed noise (L9). Adding a second anchor here would test the anchor, not the possession model. `off_c`/`def_c` are the pre-registered anchor. |
| Raw (uncentred) team rates | Banned by `CLAUDE.md`; L2 measures the cost (KenPom's league-mean AdjO drifted 100 -> 109.3 and inflated all six of last year's sub-models). |
| `n_prior_off` (games of form behind the feature) | Computed and carried on the form table for diagnostics, but not in any pre-registered bundle. Adding it after seeing the grid would be exactly the post-hoc feature search the standing rule bans. |
| Embedded ESPN win probability (`homeWinProbability`) | A market-derived column. `CLAUDE.md` data rules: embedded ESPN market columns in pbp are stripped from features. |

## 4. Known gaps

- `ft_trip_ambiguous` marks the ~40% of free-throw-trip chances where the feed cannot separate a
  two-shot shooting foul from a bonus trip (possessions build report, section 5). It is on the
  possessions and chances tables and available to condition on, but it is a property of the LABEL,
  not of the pre-game state, so it is not a feature and must never become one.
- The defence-allowed rates use the opponent's raw production, not opponent-adjusted production. The
  own ridge ratings in the same bundle carry the strength-of-schedule adjustment, so the two
  together are not redundant; a properly adjusted style rate is a followup.
- `is_transition` is a duration proxy computed from the chance's OWN duration, so it is not available at
  the chance's start. It is a confirmed defect (`change_ledger.md` section B) and must be replaced by a
  pre-chance definition before this model is wired into the engine.
- The `FGA_rim` / `FGA_jump2` split of CONTINUATION chances in season 2025 was not comparable in v1:
  the rim share of putbacks is 53.9-55.0% in 2022-2024, 44.5% in 2025, and 56.3% in 2026, because
  ESPN moved `LayUpShot` putbacks into `TipShot` and, for that one season, into two-point `JumpShot`.
  **REPAIRED in possessions v2** by the rim-location override
  (`docs/tests/possessions_build_v2_2026-09-10.md` section 3.1). The residual gap is under a tenth of
  a percentage point against a 6.3 pp defect, and the repair is a threshold derived from the feed's
  own rim rows rather than a relabelling rule chosen by eye. What remains is the part the override
  cannot reach: rows the shot-chart subsystem never located (12-22% of two-point jumper rows in
  2022-2024) keep the feed's label, which is a miss rather than a false positive.
- Feature contamination from that defect is closed by construction rather than repaired: see section
  1.1. The two fixes are independent, and the second would have been necessary even if the first had
  never been possible.
