# L3 REBOUND — features

Companion to [`model.md`](model.md) and [`experiments.md`](experiments.md).
Doc layout: `docs/models/DOCUMENTATION_STANDARD.md`.

---

## 1. Source manifest

Every feature that goes into ANY version of the model. "Source file" is the
authoritative upstream table; "computation" names the function that produces it.

| Feature | Dtype | Source file | Computation | Fallback value | Notes |
|---|---|---|---|---|---|
| `off_oreb_c` | float32 | `data/raw/cbbd/pbp/plays_{season}.parquet` -> rebound opportunities | `rebound.team_rebound_form` — offence's as-of OREB% over FIRST-CHANCE live opportunities in strictly earlier games, minus the league's as-of OREB% on the same date | `0.0` (= exactly the league mean on a centred scale) | The centring is the `CLAUDE.md` rule "every rating feature is expressed relative to its own snapshot's league mean". First-chance-only is the pre-registration's "first-chance-safe sources" clause; see §3 |
| `opp_def_dreb_c` | float32 | same | `rebound.team_rebound_form` — the DEFENCE's as-of DREB% (1 minus the OREB% its opponents took against it) over strictly earlier games, league-centred | `0.0` | The defence-allowed side of the same table |
| `off_rating_off_c` | float32 | `data/processed/ratings/own_ratings_{season}.parquet` | `own_ratings.join_as_of`, offence team, `off_c` | `0.0` | As-of ridge rating, already league-centred at source; passed the INV-45 change-form leak test (ledger row B2) |
| `off_rating_def_c` | float32 | same | offence team, `def_c` | `0.0` | |
| `def_rating_off_c` | float32 | same | defence team, `off_c` | `0.0` | |
| `def_rating_def_c` | float32 | same | defence team, `def_c` | `0.0` | |
| `site_home` | float32 | `data/processed/games_universe.parquet` | 1 when the OFFENCE is the home team and the game is not neutral | `0.0` | Neutral is the reference level, so home/away/neutral is three states in two columns. `CLAUDE.md` modelling rule: home/away/neutral is a first-class feature in every scoring-stage model |
| `site_away` | float32 | same | 1 when the offence is the away team and not neutral | `0.0` | |
| `miss_rim` | float32 | pbp | `classify_frame` class of the missed attempt == `FGA_rim` | `0.0` | `ft` (a missed last free throw) is the reference level, so four miss types are three columns |
| `miss_jump2` | float32 | pbp | class == `FGA_jump2` | `0.0` | **This is the one feature the possessions version moves** (L16); see §4 |
| `miss_three` | float32 | pbp | class == `FGA_3` | `0.0` | |
| `blocked_f` | float32 | pbp | a `Block Shot` row is adjacent to the attempt at the same `secondsRemaining` (`event_stream.build_stream`) | `0.0` | 10.1% of missed FGAs in 2024. Block rows are then dropped from the stream so that "the next event is the rebound" stays true |
| `period` | float32 | pbp | `period` of the opportunity | n/a (always present) | 1, 2, then 3+ for overtimes |
| `seconds_remaining` | float32 | pbp | `secondsRemaining` at the opportunity | n/a | PER PERIOD in the feed (1200 in halves, 300 in overtime), which is why `period` travels with it |
| `score_diff` | float32 | pbp running score | offence score minus defence score at the opportunity | n/a | Offence's point of view, so it is symmetric between the two teams |
| `in_bonus` | float32 | pbp | the DEFENCE's prior personal-foul count in this period is at or above the one-and-one threshold (`event_stream.in_bonus`) | `0.0` | The threshold is `possessions.BONUS_PRIOR_FOULS`, which that module derived from the free-throw-trip length distribution rather than assuming |
| `lineup_oreb_c` | float32 | pbp `onFloor` + rebound opportunities | `rebound.player_rebound_rates` + `attach_lineup_features` — sum over the OFFENCE's five on-floor players of each one's as-of individual OREB rate, minus five times the league's as-of per-player rate | `0.0` when any of the ten ids is missing | 2024+ only (L13: CBBD `onFloor` is empty at the source in 2022-2023) |
| `lineup_dreb_c` | float32 | same | sum over the DEFENCE's five of their as-of individual DREB rate, league-centred the same way | `0.0` | |

### Carried on the row but NOT a feature

| Column | Why it is there |
|---|---|
| `outcome` | the target (`OREB` / `DREB` / `DEAD`, plus `unresolved` which is dropped) |
| `chance_index` | how many offensive rebounds this offence has already taken in the possession. Used to build the first-chance-only team rates and reported as a diagnostic; see §3 of `experiments.md`. NOT in any pre-registered bundle |
| `rebounder_id` | CBBD player id of the rebounder, used to build the individual rates |
| `on_floor_h1..5`, `on_floor_a1..5` | CBBD on-floor ids, used to build the lineup features |
| `lineup_on_floor_ok` | all ten ids resolved to an as-of rate; the lineup fold is scored on these rows only, for BOTH C and D |
| `game_id`, `game_date`, `season` | join keys and the block-bootstrap unit |

---

## 2. Feature sets tested

| Set name | Included features | Rationale |
|---|---|---|
| `A_team` | `off_oreb_c`, `opp_def_dreb_c`, the four own-rating columns, `site_home`, `site_away` | The matchup, with nothing about the shot. Establishes how much of a rebound is decided before the ball leaves the shooter's hand |
| `B_plus_miss` | A + `miss_rim`, `miss_jump2`, `miss_three`, `blocked_f` | The shot itself. Rim misses are rebounded by the offence 37.6% of the time against 12.8% for a missed free throw, so this is expected to be the largest single block |
| `C_plus_state` | B + `period`, `seconds_remaining`, `score_diff`, `in_bonus` | Game state. The L3 round-1 analogue of this block was worth 30-140x the noise floor, so it is tested here too rather than assumed either way |
| `D_plus_lineup` | C + `lineup_oreb_c`, `lineup_dreb_c` | Whether rebounding is a LINEUP-level quantity in the engine or a team-level one. Has its own fold (train 2024, test 2025) because the features do not exist before 2024 |

---

## 3. Leak safety, in detail

Three separate hazards, each closed by construction rather than by convention:

1. **The game's own rows.** `prob_metrics.expanding_asof` is `cumsum() - value`
   within (season, team), so a team-game's own contribution is subtracted from
   its own feature. Proved twice in `tests/test_rebound.py`: by recomputing the
   strictly-earlier average independently, and by corrupting the game's own
   rows and requiring the feature not to move (while a LATER game's feature
   does move, so the test cannot pass for a builder that ignores the data).

2. **Continuation chances.** The team rates use `chance_index == 0`
   opportunities only. Pooling continuation chances would make "as-of OREB%"
   partly a function of how many second chances a team happened to get, and it
   is the exact channel by which L16's 2025 labelling defect reached a
   first-chance model's predictors in L3 round 1
   (`docs/tests/shot_classification_diag_2026-09-10.md` section 6). The pooled
   variant is available as `first_chance_only=False` and `experiments.md`
   reports what the restriction costs.

3. **No prior games.** A team with no earlier game sits at a centred value of
   exactly `0.0` -- the league mean -- and `n_prior_off` / `n_prior_def` are
   carried so the case is visible rather than indistinguishable from an
   average team.

The dead-ball and unresolved outcomes are excluded from the rate DENOMINATOR
(the rates are over live rebounds only) so that OREB% keeps meaning "share of
contested rebounds the offence took". Their shares are reported separately.

---

## 4. What the possessions version changes

`--version v1|v2` selects the possessions build. For this model it moves
**exactly one thing**: whether ESPN's 2025 putback mistag is repaired before a
missed two-point attempt is labelled `miss_rim` or `miss_jump2` (L16). The
threshold is read out of the chosen build's own `build_report.json`
(`event_stream.rim_override_for_version`, 0.0 when the build predates the
override), so the PM's re-run is one flag and no edit.

Nothing else can move, and `tests/test_rebound.py::
test_the_override_moves_only_the_miss_type_never_the_outcome` asserts it on a
synthetic play the override fires on: the override can only turn an `FGA_jump2`
into an `FGA_rim`, and the rebound outcome, the opportunity set and every team
rate are read off events the override never touches.

---

## 5. Rejected features, and why

| Candidate | Status | Reason |
|---|---|---|
| `chance_index` (this is the Nth chance of the possession) | NOT IN ANY PRE-REGISTERED BUNDLE | It is a real predictor -- pooled 2024, OREB% runs 28.7% / 31.1% / 31.8% / 32.9% at chance index 0 / 1 / 2 / 3 -- but the pre-registration's four bundles do not contain it and this round does not add features to a pre-registered grid. Recorded here so the next round can pre-register it deliberately |
| Shooter identity / rebounder identity as a feature | OUT OF SCOPE | The lineup bundle is the pre-registered way identity enters this model. Per-player rebound rates are the L4 layer's job |
| Offence's as-of DREB% and defence's as-of OREB% (the other diagonal) | NOT PRE-REGISTERED | The pre-registration names "offense as-of OREB% and defense as-of DREB% allowed" specifically. The own-ratings block already carries a general team-strength signal |
| Explicit offence x defence interaction products | NOT PRE-REGISTERED HERE | L3 round 1 measured them at 3e-06 log loss, 240x BELOW its noise floor, and rejected them on both linear arms (`possession_outcome/experiments.md` section 3.9). The tree arm finds interactions itself |
| Shot distance / location | REJECTED | Coordinates cover 6-26% of shots before 2024-25 (`SIM_GUARDRAILS.md` section 4). A feature present on a quarter of the training rows and all of the test rows is a season-drift generator, not a feature. The rim-location override uses coordinates only as an OVERRIDE on a label that already exists |
| `is_transition` | REJECTED | Confirmed-defect row B7 of `docs/models/change_ledger.md`: as currently defined it is derived from the chance's own duration and so is not available at the chance's start |
