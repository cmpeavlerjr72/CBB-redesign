# L4 SHOT ALLOCATION (usage): feature inventory

Every feature the allocator can see, with its upstream file, its computation, and
its fallback. Pre-registration: `experiments.md` section 1. Code:
`src/cbb_sim/models/usage.py`. Trainer: `scripts/train_usage_v1.py`.

The model answers "which of these five took it", so almost every feature is
**per alternative**: one value for each of the five on the floor, carried on the
design as `<feature>_1 .. <feature>_5` in ascending CBBD-player-id order. The
handful of **event-state** features are alternative-invariant and are listed
separately, because that distinction decides whether a feature can be identified
at all (section 4).

---

## 1. Source manifest

### 1.1 Per-alternative, pregame (as-of)

Every row in this block is an expanding sum over that player's games **strictly
before** the current game, within the season
(`prob_metrics.expanding_asof`, called from `usage.build_player_asof`).

| Feature | Dtype | Source file | Computation | Fallback | Notes |
|---|---|---|---|---|---|
| `exposure_asof` | float | `data/raw/cbbd/pbp/plays_{season}.parquet` via `event_stream.build_stream` | count of credited events of ANY of the five classes that occurred while this player was one of the OFFENSIVE five, strictly earlier in the season | 0 (his first game) | The denominator of every rate. A per-possession ON-FLOOR exposure, which is what makes a bench player's rate comparable to a starter's. This is the pre-registration's "per-possession-on-floor rates, not per game". |
| `ev_{class}` | float | same | count of those events credited to him, per class, strictly earlier | 0 | `class` runs over `FGA_rim`, `FGA_jump2`, `FGA_3`, `TOV`, `FT_trip`. |
| `ev_total` | float | same | `sum_class ev_{class}` | 0 | The "as-of usage rank" of the role rule is ranked on this, shrunk by the same formula as a real class, so the role cannot be assigned off a differently-constructed rate. |
| `rate_{class}` | float | derived | `ev_{class} / exposure_asof` | NaN when exposure is 0; never consumed raw -- `shrunk_rate` is | The raw as-of per-possession-on-floor rate. |
| `lg_rate_{class}` | float | derived | league total `ev_{class}` / total `exposure` over all games strictly earlier in the season (by calendar date) | back-filled from the season's own next available date on opening day; then the column median | The league shrinkage prior. The opening-day back-fill is the only forward-looking value in the module and touches opening day alone (same rule as `free_throw`). |
| `pos_rate_{class}` | float | `data/raw/cbbd/rosters/roster_*.parquet` (position) + pbp | same ratio restricted to the player's position group, strictly earlier | as `lg_rate_{class}` | The position shrinkage prior, and the fallback the pre-registration names for players with no prior season. |
| `prev_rate_{class}` | float | derived | `ev_{class} / exposure` over the player's COMPLETED previous season | `pos_rate_{class}` when there is no previous season | Exists only for 2025 rows: on-floor ids start in 2024 (L13). See section 4. |
| `has_prior_season` | float | derived | 1 when the previous season carries any on-floor exposure for him | 0 | Identically 0 on the whole F1 training fold. |
| `prev_exposure` | float | derived | previous season's `exposure` | 0 | How much the prior-season rate is worth. |
| `is_transfer` | int8 | derived | his modal team in the previous season differs from this game's team AND `has_prior_season` | 0 | The transfer-subset key, and the L15 attenuation flag. |
| `minutes_asof` | float | `data/raw/hoopr/player_box/player_box_{season}.parquet` joined through `data/processed/player_crosswalk.parquet` | cumulative hoopR `minutes` over strictly earlier games | 0, with the join rate reported rather than imputed | The pre-registration's "minutes-to-date". The crosswalk covers 2024-2026, which is exactly this model's window -- unlike at FT-2, where it could not span the fit window. |
| `minutes_per_game_asof` | float | derived | `minutes_asof / (games with a matched box line)` | 0 | |
| `games_asof` | float | derived | count of strictly earlier games in which he was on the floor for at least one credited event | 0 | |
| `position_code` | int8 | `data/raw/cbbd/rosters/roster_*.parquet` | `free_throw.POSITION_GROUPS` mapping of CBBD `position` to G / F / C, then to an index into `usage.POSITION_LEVELS` | `UNK` (index 3) | Imported from `free_throw.load_positions` rather than re-implemented, so the two models cannot disagree about a player's position. |

### 1.2 Per-alternative, derived at fit time

| Feature | Dtype | Computation | Notes |
|---|---|---|---|
| shrunk rate | float | `(m * prior + ev_{class}) / (m + exposure_asof)`, `usage.shrunk_rate` | `m` is in pseudo ON-FLOOR EVENTS, the same unit as the denominator, so "m = 50" means "50 events of history before a player's own rate outweighs the prior". Both `prior` (league / position / prior_season) and `m` are FITTED on the training fold (`usage.fit_shrinkage`). |
| `share` | float | the shrunk rate normalised over the five (`usage.normalise`) | U1's prediction. A lineup in which nobody carries any rate falls back to the uniform over the five. |
| `usage_rank` | float | rank of `share` within the five | Tree feature. |
| role | int8 | centre -> `big`; among the non-centres the top `HANDLER_RANK` by shrunk `rate_total` -> `handler`; the rest -> `wing` (`usage.assign_roles`) | The between-role level of U3. Position group and as-of usage rank are exactly the two inputs the pre-registration names. The rank is taken over the lineup, which is the candidate set the sampler is handed; the rate is entirely pregame. |

### 1.3 Event state (alternative-invariant)

| Feature | Dtype | Source | Computation | Notes |
|---|---|---|---|---|
| `score_diff` | int16 | pbp `homeScore` / `awayScore` | offence score minus defence score on the event's own row | |
| `sec_remaining` | int32 | pbp `secondsRemaining`, `period` | `(2 - period) * 1200 + sec` in regulation; `sec` in overtime | `secondsRemaining` is PER PERIOD in the feed, so it does not identify late-game on its own; `period` is carried alongside for the same reason `free_throw` carries it. |
| `period` | int16 | pbp | | |
| `chance_number` | int16 | pbp, `usage._chance_number` | `1 + (offensive rebounds since the last possession-ending event)` | **A PROXY, labelled as one.** The possession state machine owns the authoritative `chance_number`, but the chance table it writes carries no on-floor five to join an event to, and the possession table carries the five only once per possession (from its first event), which would mis-attribute any lineup change inside a possession. Possession-ending events are taken as DREB, TOV, a made field goal, and the last made free throw of a trip. |

### 1.4 Target and universe keys

| Column | Source | Notes |
|---|---|---|
| `event_class` | `event_stream.cls`, plus `FT_trip` | `FGA_rim` / `FGA_jump2` / `FGA_3` / `TOV` straight off `events.classify_frame`; `FT_trip` is the FIRST attempt of a foul-caused free-throw trip (`trip_pos == 1 and trip_cause == 'foul'`). |
| `player_id` | pbp `participant_1_id` | The credited player: the shooter, the turnover, the fouled shooter. CBBD player id. |
| `alt_1 .. alt_5` | pbp `home_on_*` / `away_on_*` through `event_stream.ON_FLOOR_COLS`, side-resolved | The OFFENSIVE five, sorted ascending so the alternative order is a function of the lineup and never of the feed's column order. |
| `y` | derived | the slot index of `player_id` inside the sorted five. |
| `five_ok`, `in_five` | derived | coverage flags, KEPT on the table so the trainer reports the filter instead of silently losing rows (`usage.coverage_report`). |
| rim/jumper label | `data/processed/possessions_v2/build_report.json` -> `event_stream.rim_override_for_version` | The L16 putback repair threshold is READ from the chosen possessions build's own report, not assumed. It moves only which of `FGA_rim` / `FGA_jump2` an attempt belongs to -- it cannot move a credited identity. |

---

## 2. Feature sets tested

The pre-registration fixes one feature bundle per arm rather than a ladder of
bundles, so there is no feature-set grid here; the grid is over arms, priors,
shrinkage strengths and concentrations (`experiments.md` section 3).

| Set | Arms | Included |
|---|---|---|
| `S_share` | U1, U2, U3 | the shrunk as-of class rate alone, normalised over the five. Everything else enters only through the fitted prior and `m`. U2/U3 add the role vector. |
| `S_logit` | U4 | `log_share`, `log_prior_share`, `log_exposure`, `log_minutes`, `is_G`, `is_F`, `is_C`, `is_transfer`, and the state interactions `log_share x {score_diff, sec_remaining, chance_number}` and `is_C x {score_diff, sec_remaining}` -- all centred within the choice set. |
| `S_tree` | U5 | `share`, `rate`, `prior_rate`, `exposure_asof`, `minutes_asof`, `minutes_per_game_asof`, `games_asof`, `position_code`, `is_transfer`, `usage_rank`, plus the RAW state variables `score_diff`, `sec_remaining`, `period`, `chance_number`. |

---

## 3. Why the state enters U4 only as an interaction

A conditional logit over a choice set differences out anything constant across
the alternatives: `exp(v_i + c) / sum_j exp(v_j + c)` does not depend on `c`. So
`score_diff`, `sec_remaining` and `chance_number` as plain columns are not
merely weak in U4, they are **unidentified** -- their coefficients are not
estimable. They are pre-registered features, so they are kept, and they enter the
only way they can carry information about WHO gets the ball: multiplied by an
alternative's own share and by its role. The raw columns go to U5 unchanged,
because a tree scores each alternative separately and can interact a state
variable with an alternative feature on its own.

---

## 4. The prior-season asymmetry, and what is dropped because of it

On-floor ids are empty at the source in 2022-2023 (L13), so:

- a **2024** row has no prior season of on-floor history at all
  (`has_prior_season` = 0 on 100% of 2024 player-games),
- a **2025** row does.

Fold F1 trains on 2024 and tests on 2025, so the prior-season block is
identically absent in training and present in test. That is worse than a constant
column: `build_usage_design` falls `prev_rate_{class}` back to the POSITION
prior, so the column still varies across the five and a naive fit happily
estimates a coefficient for a variable whose MEANING changes between the folds.
Measured cost of not catching it: the F1 conditional logit on `FGA_3` scored
3.74 against U1's 1.46.

`usage.unidentified_features` therefore returns `log_prior_share` (U4) and
`prior_rate` (U5) on any fold whose training slice has no prior-season history,
the trainer drops them, and the drop is RECORDED in the results table -- the same
treatment `cbb_sim.models.clock` gives its degenerate chance-number column. The
shrinkage grid handles the same problem by reporting the `prior_season` rung as
`unidentified` and excluding it from the choice.

The within-2025 walk-forward fold has the block active on BOTH sides (2024 is by
then a completed season of on-floor history), and that fold is where its value is
actually measured.

`is_transfer` is dropped on F1 for the same root cause by the ordinary
constant-column rule: a transfer flag needs a previous season to compare against.

---

## 5. Rejected features

| Feature | Why not |
|---|---|
| on-floor five from `possessions_{season}.parquet` | It is recorded once per possession, from the possession's FIRST event, so any substitution inside a possession attributes the event to the wrong five. The event stream carries the five on the event's own row. |
| `is_transition` | Derived from the chance's own `duration_s`, i.e. not available when the chance starts, and at L5 it is a function of the target outright. Banned upstream (`cbb_sim.models.clock.BANNED_FEATURES`, change-ledger row B7); not used here. |
| CBBD season ratings (`/ratings/adjusted`, `/ratings/srs`, `/ratings/elo`) | End-of-season snapshots with no date field -- banned as pregame features (L7). |
| ESPN market columns embedded in pbp | Stripped from every feature table by the data rules (`CLAUDE.md`). |
| shot coordinates as a direct feature | 6-26% coverage before 2024-25. They enter only through the L16 rim-location override, which is applied at the event layer and read from the possessions build's own report. |
| hoopR `active` | A placeholder before 2025-26; `did_not_play` is the historical availability field and availability is the ROTATION model's input, not the allocator's -- the allocator is handed a five. |
| opponent identity / opponent defensive rates | Not in the pre-registration. The allocation is conditioned on a five that the rotation model already chose against this opponent, and a defensive term belongs to the possession-outcome model that decides WHICH class of event happens. Listed here so the omission is a recorded choice and not an oversight; see `model.md` section 9. |
| the player's own in-game counts so far | Not a pregame quantity. It is computed in `usage.sequential_probs` as the Polya-urn DIAGNOSTIC only, is labelled as not pregame, and never enters a decision. |
