# L4 PLAYER ATTRIBUTION: feature inventory

Code: `src/cbb_sim/models/attribution.py`. Trainer:
`scripts/train_attribution_v1.py`. Pre-registration and results:
[`experiments.md`](experiments.md). Model doc: [`model.md`](model.md).

Every feature in this model is **pregame or state**: an as-of quantity computed
from games strictly earlier in the season, a completed prior season, a roster
position, or the game state at the event. Nothing on the event's own row enters a
feature except the state (period, clock, score) and the shot class, and the
event's own recorded attribution never does —
`tests/test_attribution.py` group A proves that two ways.

---

## 1. Source tables

| Table | Built by | What it is |
|---|---|---|
| `data/raw/cbbd/pbp/plays_{season}.parquet` | `scripts/pull_cbbd_pbp.py` | the CBBD play-by-play feed |
| the attribution stream | `attribution.build_attr_stream` | the cleaned stream with the four credits attached to the rows they belong to |
| `events_{pop}_{version}.parquet` | `attribution.build_attr_events` | the five population tables |
| `asof_{version}.parquet` | `attribution.build_player_asof` | one row per (season, player, game), every as-of player input |
| `team_asof_{version}.parquet` | `attribution.build_team_asof` | one row per (season, team, game), the binaries' team inputs |
| `data/processed/player_crosswalk.parquet` | `scripts/build_player_crosswalk.py` | CBBD ↔ ESPN player ids, the bridge to hoopR minutes |
| CBBD rosters | `free_throw.load_positions` | the position group prior |

### 1.1 Why the stream is rebuilt rather than imported

`cbb_sim.models.event_stream` drops the `Block Shot` rows after folding them into
a `blocked` flag (so the blocker's id is gone), carries no assist columns, and
keeps no play `id` to join a stealer back onto its turnover. So
`build_attr_stream` re-does the same cleaning on the same source with three extra
columns (`participant_1_id`, `shot_assisted`, `shot_assisted_by_id`) and the same
repairs: `possessions._fix_flipped_sides` for the `isHomeTeam` inversion
(data-defect D2), the administrative-rebound drop inside free-throw trips, and
the same-second adjacency rule for the defensive credits. `event_stream.py` is a
shared prerequisite two shipped models train against and is not edited.

### 1.2 THE SHOOTER IS `shot_shooter_id`, NOT `participant_1_id`

Measured on 2025 made field goals: `participant_1_id` equals `shot_shooter_id` on
**74.5%** of rows and on only **50.9%** of **assisted** ones, where it is the
**assister** on the other 48.9% — CBBD's participant order is not stable on a
two-participant row. Reading the shooter off `participant_1_id` makes the
assister "the shooter" on half the assist population, removes the wrong man from
the choice set, and hands the model an impossible target (measured before the
fix: 51% of assisted field goals were modellable; after it, 98.7%). This model
reads the shooter off the dedicated `shot_shooter_id` column on every field-goal
row and `participant_1_id` only where it is the sole participant (the rebounder,
the charged player on a turnover, the stealer, the blocker).

### 1.3 The defensive credits come from an adjacency rule, not a join key

A `Steal` row follows its `Lost Ball Turnover` row and a `Block Shot` row sits
adjacent to its attempt, at the same `secondsRemaining` in the same game.
`attribution._adjacent_credit` matches in **both** directions and prefers the
NEXT row when both sides carry a credit, because the feed emits the decoration
after the event on the large majority of rows. This is `event_stream`'s `blocked`
rule with the player id carried through rather than discarded. The share of
flagged events with no resolvable id is a reported number
(`coverage_report`), never a silent drop.

---

## 2. The per-player as-of rate — and its denominator

`attribution.build_player_asof`. One row per (season, player, game). Every column
is an expanding sum over that player's games **strictly before** the current one
within the season (`prob_metrics.expanding_asof`).

**The denominator is opportunities of the target's own population, on the
floor.** A player's `REB_def` exposure is the number of live defensive rebounds
his team won while he was one of the defensive five; his numerator is how many he
secured. Consequences:

- the candidates' rates sum to approximately one over the candidate set, so P1's
  normalisation is a re-scaling rather than a re-interpretation;
- the fitted shrinkage strength `m` reads directly as "pseudo **opportunities**
  of history before a player's own rate outweighs the prior";
- it is **not** the same definition as `rebound.player_rebound_rates`, which uses
  every miss while on the floor as the denominator and so measures a share of
  *available* rebounds. Both are defensible; the difference is a denominator, not
  a leak, and the two numbers are not interchangeable.

| Target | exposure (denominator) | numerator |
|---|---|---|
| `REB_off` | live offensive rebounds his team won while he was one of the offensive five | he secured it |
| `REB_def` | live defensive rebounds his team won while he was one of the defensive five | he secured it |
| `assist` | assisted made field goals by his team while he was on the floor and **not the shooter** | he assisted it |
| `steal` | turnovers the opponent lost to a steal while he was one of the defensive five | he got the steal |
| `block` | missed field goals that were blocked while he was one of the defensive five | he got the block |

### 2.1 The three priors, and the shrinkage formula

`shrunk_rate = (m * prior + credits) / (m + opportunities)`, with the prior one
of:

| `prior_kind` | value | notes |
|---|---|---|
| `league` | the league's as-of per-candidate rate on the same date | back-filled on opening day from that season's next date (the only forward-looking value in the module; it touches opening day alone) |
| `position` | the same, within the player's roster position group | `free_throw.load_positions`; `UNK` falls back to the league |
| `prior_season` | his own **completed** prior season's rate, falling back to the position prior when he has none | this is the pre-registration's "prior-season rate for returners, position group prior for new players" |

`(prior_kind, m)` is **fitted** on each fold's training slice by
`fit_shrinkage`, never assumed (L13: "shrinkage strength is a fitted
parameter"). On F1 the `prior_season` rung is **unidentified** — on-floor ids
start in 2024, so a 2024 row has no prior season and a 2025 row does — and it is
reported as such rather than scored.

### 2.2 Reported-only player columns

`games_asof`, `minutes_asof`, `minutes_per_game_asof` (hoopR `player_box`
minutes, joined through the player crosswalk), `prev_team_id` and the derived
`is_transfer_{k}` are carried onto the design for the **reported subsets**
(transfer / continuing / no-prior-season) and are **not** features of any arm.
The pre-registration's player input list is the as-of rate, the prior-season rate
and the position group; exposure and minutes are deliberately absent from every
arm, and that is recorded as a known gap in `model.md` section 9 rather than
quietly added.

---

## 3. The per-team as-of rates (the three binaries)

`attribution.build_team_asof`. One row per (season, team, game), expanding over
that team's strictly earlier games, built from the **same** population tables the
targets are — so a team's denominator is exactly the set of events the binary is
graded on.

| binary | offence's own rate | defence's rate allowed |
|---|---|---|
| `assisted` | assisted made FGA / made FGA | assists allowed / opponent made FGA |
| `stolen` | turnovers lost to a steal / turnovers | steals / opponent turnovers |
| `blocked` | blocked misses / missed FGA | blocks / opponent missed FGA |

Both are shrunk toward the **league as-of rate of the same date** with a fitted
`m_team` (`team_shrunk`), then **centred on that league rate** before entering a
feature matrix — `CLAUDE.md`: "every rating feature is expressed relative to its
own snapshot's league mean; raw levels are banned". A team with no history yet
sits exactly at the league rate rather than at a NaN.

The **own-share** block is the individual offensive player's own as-of share for
that binary (his assisted share of his own made field goals, his stolen share of
his own turnovers, his blocked share of his own misses), shrunk with its own
fitted `(own_prior, m_own)` and centred the same way.

---

## 4. What each arm actually sees

### 4.1 Choice arms

| feature | P1 | P2 (cond. logit) | P3 (LightGBM) |
|---|:--:|:--:|:--:|
| shrunk as-of rate, normalised over the candidate set | the model | `log_share` | `share`, `rate`, `rate_rank` |
| prior-season rate | inside the prior | `log_prior_share` | `prior_rate` |
| position group | inside the prior | `is_G`, `is_F`, `is_C` | `position_code` |
| shot class | — | `log_share_x_{rim,three,ft}`, `is_C_x_{rim,three,ft}` | `shot_class_code` |
| score diff | — | `log_share_x_scorediff`, `is_C_x_scorediff` | `score_diff` |
| seconds remaining | — | `log_share_x_sec`, `is_C_x_sec` | `sec_remaining` |

**Why the state enters P2 only as an interaction.** A conditional logit
differences out anything constant across the alternatives, so `score_diff`,
`sec_remaining` and the shot class as plain columns are not weak but
**unidentified**. They are pre-registered, so they are kept and interacted with
the alternative's own log share and with `is_C`. The raw columns go to P3
unchanged, where a tree can interact them itself (`usage` decision 9).

**Why `share` and `rate_rank` are not extra inputs.** Both are monotone
transforms of the shrunk as-of rate *within the choice set*. They are listed
separately because the tree sees them as separate columns, not because a new
quantity entered.

**The `steal` target has no shot class.** Its three shot-class interactions are
identically zero and `drop_constant_features` removes them; the drop is printed
in that target's block in `experiments.md` rather than passing silently.

**`log_prior_share` and `prior_rate` are dropped on F1** by
`unidentified_features`, because the prior-season block's *meaning* differs
between a fold with no prior season and one with (`build_choice_design` falls it
back to the position prior, so it still *varies* and a constant-column check does
not catch it). `usage` measured the cost of not catching this at a log loss of
3.74 against 1.46 on one class.

### 4.2 Binary arms

| feature | `team_ridge` | `aware_ridge` | `lgbm` |
|---|:--:|:--:|:--:|
| `off_rate_c` (offence's shrunk as-of rate, league-centred) | ✓ | ✓ | ✓ |
| `def_rate_c` (defence's shrunk as-of rate allowed, league-centred) | ✓ | ✓ | ✓ |
| `score_diff`, `sec_remaining` | ✓ | ✓ | ✓ |
| `cand_is_home`, `neutral_site` | ✓ | ✓ | ✓ |
| `own_share_c` (the offensive player's own shrunk as-of share) | — | ✓ | ✓ |
| `sc_rim`, `sc_three` (shot / miss class) | — | ✓ | ✓ |

Home/away/neutral is in every arm per `CLAUDE.md`'s standing modelling rule.
Ridge features are standardised on the **train slice only** and the fitted
means/SDs travel with the model, so the sim applies the identical transform
(`rebound.RidgeLogitArm`'s contract).

---

## 5. The game state on the event's own row

| column | provenance |
|---|---|
| `period` | CBBD `period` |
| `sec_remaining` | seconds left in regulation for periods 1-2, seconds left in the period in overtime |
| `score_diff` | the **candidate team's** margin at the event, from the forward-filled running score |
| `shot_class` | the attempt's own class on a field-goal row; the class of the **immediately preceding miss** on a rebound row; `none` on a turnover |
| `cand_is_home`, `neutral_site` | from `games_universe.parquet` |

`shot_class` on a rebound row is the pre-registration's "shot class where
relevant" for the two R targets: an offensive board off a rim miss is a different
event from one off a three.

---

## 6. Leak safety, stated as tests

| test | what it proves |
|---|---|
| `test_a_asof_rate_equals_an_independent_strictly_earlier_recomputation` | every as-of numerator and denominator, recomputed from the event table without calling `expanding_asof`, equals the shipped column |
| `test_a_the_events_own_attribution_never_enters_its_own_features` | re-crediting every event of a game moves **no** feature of that game and none of a later game, but does move a later game's features when the corrupted game is earlier |
| `test_a_prior_season_columns_come_from_a_completed_season` | the prior-season block is identically zero on the first season present and non-zero on the second |
| `test_b_the_shooter_is_never_an_assist_candidate` | the eligible-set rule, on the real derivation path |
| `test_b_a_duplicated_or_missing_on_floor_id_disqualifies_the_choice_set` | a short or degenerate five is rejected, not silently padded |
