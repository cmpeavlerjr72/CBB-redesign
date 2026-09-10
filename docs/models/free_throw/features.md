# L3 FREE THROW — features

Companion to [`model.md`](model.md) and [`experiments.md`](experiments.md).
Doc layout: `docs/models/DOCUMENTATION_STANDARD.md`.

The model has two parts and only the second one has features. FT-1 (how many
attempts a foul produces) is a RULE TABLE checked against the data, not a fit;
its inputs are listed in §4.

---

## 1. Source manifest — FT-2 (make probability per attempt)

| Feature | Dtype | Source file | Computation | Fallback value | Notes |
|---|---|---|---|---|---|
| `shooter_ft_asof` | float32 | `data/raw/cbbd/pbp/plays_{season}.parquet` free-throw rows | `free_throw.build_ft_design` — the shooter's made/attempted over games STRICTLY BEFORE this one within the season, minus the league's as-of rate on the same date | `0.0` (= exactly the league mean on a centred scale) when the shooter has no earlier game | L15: player identity is the dominant term in every player-game rate stat, so this is the feature the whole model turns on |
| `shooter_fta_asof` | float32 | same | the shooter's attempts to date this season | `0.0` | The reliability of `shooter_ft_asof`. It is what makes the shrinkage arm's strength interpretable and what the ridge arm needs to know how far to trust the rate |
| `prior_season_ft` | float32 | same, previous season | the shooter's COMPLETED previous-season make rate, league-centred | `0.0` (league mean) for a player with no prior season | A completed prior season is entirely in the past, so it is leak-free by construction |
| `has_prior_season` | float32 | same | 1 when the shooter has a previous season in the table | `0.0` | Distinguishes "prior-season rate equals the league mean" from "there is no prior season", which the centred column alone cannot |
| `season_idx` | float32 | derived | `season - 2022` | n/a | L4/L11: scoring has trended up through free-throw rate; a pooled fit with no season term under-shoots the current year |
| `seconds_remaining` | float32 | pbp | `secondsRemaining` at the attempt | n/a | PER PERIOD in the feed |
| `period` | float32 | pbp | `period` of the attempt | n/a | **The one addition to the pre-registered list.** `seconds_remaining` is per period (1200 in halves, 300 in overtime), so on its own it does not identify late-game at all: 30 seconds left in period 1 and in period 2 are different situations with the same value. Recorded here rather than slipped in |
| `score_diff` | float32 | pbp running score | shooting team's score minus the opponent's at the attempt | n/a | The pressure axis of the pre-registration's "late-game/pressure state" |
| `in_bonus` | float32 | pbp | the trip's foul class is `bonus_one_and_one` or `double_bonus` | `0.0` | The pre-registration's "bonus vs shooting". Derived from the fouling team's running period foul count, never from the attempt count |

### Used by the arms but not in the feature matrix

| Column | Used by | What it is |
|---|---|---|
| `team_ft_raw` / `team_ft_asof` | arm (a) `team_asof` | The shooting TEAM's as-of make rate (raw, and league-centred). The team-level floor: the arm every shooter-level arm has to beat to justify carrying player identity into the engine |
| `lg_ft_asof` | arm (b) prior `league` | The league's as-of make rate on the attempt's date |
| `position_ft_asof` | arm (b) prior `position` | The as-of make rate of the shooter's position group (G / F / C / UNK), from CBBD rosters |
| `prior_season_ft_raw` | arm (b) prior `prior_season` | The shooter's completed prior-season rate, uncentred |
| `shooter_ftm_prior`, `shooter_fta_prior` | arm (b) | The numerator and denominator the shrinkage formula needs |
| `is_transfer` | the transfer-subset check | The shooter's modal team changed since the prior season |
| `espn_athlete_id` | the L4 handoff, and the reported match rate | From `data/processed/player_crosswalk.parquet` where it resolves (2024-2026 only) |
| `foul_class`, `trip_len`, `trip_pos`, `trip_prior_fouls` | FT-1, and the bonus segment | Trip structure |
| `game_id` | the block bootstrap | The resampling unit |

---

## 2. Feature sets tested

The pre-registration lists ARMS rather than feature bundles, so the grid is over
arms and the feature matrix is fixed:

| Arm | Inputs | Rationale |
|---|---|---|
| (a) `team_asof` | `team_ft_raw` only | The team-level floor. No shooter identity at all. If nothing beats it, free throws are a team quantity and the engine does not need a shooter layer for them |
| (b) `eb_shrink` | `shooter_ftm_prior`, `shooter_fta_prior`, and one of {`lg_ft_asof`, `position_ft_asof`, `prior_season_ft_raw`} | Shooter identity with a fitted amount of scepticism. Both the prior and the strength are fitted on the training fold |
| (c) `ridge` | the nine columns of §1 | Shooter identity plus the state, linearly |
| (d) `lgbm` | the same nine | Shooter identity plus the state, with interactions found rather than specified |

---

## 3. Leak safety, in detail

1. **The shooter's own game.** `shooter_ft_asof` and `shooter_fta_asof` are
   built by `prob_metrics.expanding_asof` over (season, shooter) at GAME
   granularity: `cumsum() - value`, so the current game contributes nothing.
   An attempt therefore cannot see the other attempts in its own trip either.
   Proved twice in `tests/test_free_throw.py` (independent recomputation, and
   invariance to corrupting the shooter's own game while a LATER game does
   move).
2. **The league and position rates are built from PER-GAME counts, not from the
   as-of columns.** Expanding a column that is already an expanding sum would
   square the history. `build_ft_design` keeps `pg_raw` for exactly this reason
   and the test above is what catches a regression: an early version of this
   file made that mistake and the strictly-earlier-average test failed on it.
3. **The prior season is a completed season.** No as-of logic needed.
4. **Opening day.** The season's first date has nothing strictly earlier, so
   the league as-of rate is back-filled from the season's own next date, on the
   date-sorted table. This is the only forward-looking value in the module, it
   touches opening day alone, and it is preferred to a typed-in constant
   because a typed-in constant would be a hand-tuned number
   (`CLAUDE.md`, no hand tuning).

---

## 4. FT-1 inputs (the rule check)

FT-1 fits nothing. Its inputs are the trip's CONTEXT, and the foul class is
derived from context alone -- never from the attempt count, because deriving
the class from the count and then checking the count against the class would
verify nothing:

| Input | Source | Used for |
|---|---|---|
| the row before the trip's first attempt | pbp | `technical` vs `foul` vs a feed gap |
| a MADE field goal by the shooting team at the same `secondsRemaining`, two rows back | pbp | the and-one signature (validated in `possessions.py`: of 1,982 fouls both preceded by a shot at the same clock and followed by free throws, 1,942 follow a MADE shot) |
| the fouling team's PRIOR personal-foul count in the period | pbp, counted by `event_stream._attach_team_fouls` | which side of the bonus thresholds the foul is on |
| the bonus thresholds themselves | RE-DERIVED per season by `free_throw.derive_bonus_thresholds` | the era check. They are arguments to `classify_foul`, not constants, so a rule change is representable and detectable rather than silently mis-labelled |

The derived thresholds are written to
`data/processed/models/free_throw/bonus_era.json`. **The engine reads that into
GameState** — per `CLAUDE.md` ("rule-era flags live in GameState, not baked
into sub-models"), no fitted object in `cbb_sim.models.free_throw` carries a
season-specific rule constant.

---

## 5. What the possessions version changes

Nothing. `--version` is accepted for symmetry with the rebound model and with
the PM's one-flag re-run, and it selects the rim-location override that is
applied when the event stream is classified — but the override can only turn an
`FGA_jump2` into an `FGA_rim`, and neither a free-throw attempt nor a trip's
structure reads a field-goal class. `train_free_throw_v1.py --check-versions`
rebuilds the attempt table under both versions and asserts the two are
identical rather than leaving it as prose; the result is in `experiments.md`
section 2.

---

## 6. Rejected features, and why

| Candidate | Status | Reason |
|---|---|---|
| Technical free-throw attempts | EXCLUDED FROM THE UNIVERSE | The shooter on a technical is chosen by the coach rather than by who was fouled, so the attempt is drawn from a different shooter distribution. Pooling would bias both. Their count and make rate are reported and the engine needs a separate rule for them (`model.md` section 9) |
| `trip_pos` (1st vs 2nd attempt of a trip) | NOT PRE-REGISTERED | A real and well-known effect. Adding it would be a feature invented after seeing the grid; recorded here so the next round can pre-register it |
| Opponent identity / defensive rating | NOT PRE-REGISTERED, and implausible | A free throw is uncontested. The pre-registration deliberately keys on the shooter |
| Home / away / neutral | NOT PRE-REGISTERED HERE | `CLAUDE.md` makes site a first-class feature in every scoring-stage model, and this one does not carry it, because the pre-registration's feature list does not. This is a genuine gap and it is listed in `model.md` section 9 rather than closed by an unregistered addition |
| ESPN athlete id as the model's key | REJECTED as the KEY, kept as an attachment | The crosswalk covers 2024-2026 only (CBBD rosters were pulled for those seasons), so it cannot span the 2022-2025 fit window. The CBBD player id can, and it maps to the SAME ESPN athlete id on 100.0% of the 9,421 players the crosswalk covers in both 2024 and 2025 and the 3,730 covered in both 2025 and 2026 |
| Shot-clock / possession context | UNAVAILABLE | A free-throw row carries no possession-level context beyond period, clock and score, all of which are already here |
