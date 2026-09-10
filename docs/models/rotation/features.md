# Rotation model (L4) — features

Companion to `model.md` and `experiments.md`. Per `docs/models/DOCUMENTATION_STANDARD.md`.

Everything here is in **CBBD player-id space**, because that is the only id space
the on-floor data exists in. `src/cbb_sim/data/player_ids.py` bridges to ESPN
`athlete_id` for the hoopR box-score fields; the bridge's measured match rate is
in `model.md` section 4 and in `data/processed/player_crosswalk_report.json`.

---

## 1. Source manifest (canonical table)

### 1.1 Pregame, per (game_id, team_id, cbbd_player_id) — all as-of, strictly earlier games

Built by `cbb_sim.models.rotation.build_asof_player_features`. The candidate pool
for a game is exactly the set of players with at least one **earlier** appearance
for that team in that season, so a player who joins the roster in January is
invisible to a November game by construction. Every column is a `shift(1)`
expanding statistic over the team's own game order.

| Feature | Dtype | Source file | Computation | Fallback | Notes |
|---|---|---|---|---|---|
| `mpg_asof_raw` | float | `data/processed/possessions/possessions_{season}.parquet` | cumulative on-floor minutes / cumulative team games, both shifted one game | 0.0 | a game he missed counts as a zero, which is how availability enters the mean |
| `mpg_adj` | float | derived | `shrink_mpg(mpg_asof_raw, n_games, rank, role_prior, k0)` then `* w_dnp` if the last game was a DNP | role prior | computed in `add_asof_ranks`, not stored on the feature frame |
| `games_played_asof` | float | same | shifted cumulative count of games with > 0 minutes | 0.0 | |
| `team_games_asof` | float | same | shifted count of the team's own games | 0.0 | games with < 3 are excluded from the test universe |
| `start_freq_asof` | float | same | shifted expanding mean of the starter flag | 0.0 | flat mean; kept for reporting, **not** the start ordering |
| `start_ewma_{15,30,50,80}` | float | same | shifted EWMA of the starter flag at decay α ∈ {0.15, 0.30, 0.50, 0.80} | 0.0 | the fitted `start_alpha` picks one; this is the start ordering |
| `dnp_rate_asof` | float | same | shifted expanding mean of `minutes == 0` | 0.0 | CBBD-derived availability |
| `last_game_dnp` | float | same | shifted `minutes == 0` of the previous team game | 0.0 | the pre-registered "last game's DNP" |
| `cum_fouls_asof` | float | `data/raw/cbbd/pbp/plays_{season}.parquet` | shifted cumulative `PersonalFoul` rows with this participant id | 0.0 | 99.93% of 2025 foul rows carry a participant id |
| `cum_minutes_asof` | float | possessions | shifted cumulative on-floor minutes | 0.0 | denominator of the foul rate |
| `fouls_per_min_asof` | float | derived | raw ratio `cum_fouls / max(cum_minutes, 1)` | 0.0 | **reporting only** — the modelling rate is the shrunk one below |
| `fpm_shrunk` | float | derived | `(cum_fouls + M0·μ) / (cum_minutes + M0)`, `μ = fpm_league`, `M0 = fpm_prior_min` | `fpm_league` | computed in `build_priors`; the unshrunk ratio gives a player with one prior minute and one foul a rate of 1.0 fouls/min and produced ~5x too many four-foul players |
| `minutes_rank_asof` / `minutes_rank_adj` | int | derived | within-team-game rank by `mpg_asof_raw` / `mpg_adj`, descending | — | `minutes_rank_adj` indexes `p_play` |
| `start_rank_asof` | int | derived | within-team-game rank by (`start_ewma_{α}`, `mpg_adj`) descending | — | positions 1–5 are the model's starting five and index the state tilt |
| `rotation_depth_asof` | int | derived | count of team candidates with `mpg_asof_raw >= 10` | 0 | the pre-registered team-level rotation depth |
| `did_not_play`, `starter` (hoopR) | bool | `data/raw/hoopr/player_box/player_box_{season}.parquet` | joined through the CBBD↔ESPN crosswalk by `attach_hoopr_availability` | CBBD-derived value | the guardrails' named availability field; used to *validate* the CBBD-derived DNP (agreement reported in `experiments.md` §2.1), and `active` is a placeholder before 2025-26 so it is not used |

### 1.2 Game state, per possession

Built by `load_team_possessions` and carried on `GameScript`.

| Feature | Dtype | Source | Computation | Notes |
|---|---|---|---|---|
| `period` | int | possessions | as recorded | ≥ 3 is overtime |
| `start_clock` | int | possessions | `secondsRemaining` at the possession's start | per period |
| `duration_s` | float | possessions | `start_clock - end_clock`, clipped at 0 | sums to 40.35 min per game on 2025 |
| `margin` | int | possessions | `start_score_diff` re-signed to this team | own perspective |
| `time_bucket` | int | derived | 0 = 1st half, 1 = 2nd half > 8:00, 2 = 8:00–2:00, 3 = final 2:00, 4 = OT | |
| `margin_bucket` | int | derived | 0 = \|m\| ≤ 5, 1 = 6–15, 2 = > 15 | the G8 blowout bands |
| player's current fouls | int | simulated | per-possession Bernoulli at `fpm_shrunk × duration × foul_rate_scale` | **simulated, never read from the game being simulated** — reading the real foul events would be a leak |
| minutes played so far vs target | float | simulated | `played_i` vs the drawn target | |
| recent on-floor rate | float | simulated | EWMA at horizon `ema_horizon` | replaces "time since last substitution", which it strictly generalises |

### 1.3 Fitted objects (train season only)

| Object | Shape | Fitted how |
|---|---|---|
| `role_prior` | 15 | mean realised minutes by within-team-game minutes rank |
| `k0` | scalar | grid, minimum next-game minutes MAE |
| `w_dnp` | scalar | realised / predicted minutes of last-game-DNP players |
| `start_alpha` | scalar | grid over the four EWMA decays, maximum overlap with the real starting five |
| `p_play[rank]` | 15 | P(the as-of rank-r candidate records any minutes) |
| `w_avail_dnp` | scalar | realised play rate of last-game-DNP players / their rank-matched expectation |
| `fpm_league`, `fpm_prior_min` | scalars | pooled fouls per on-floor minute; grid on next-game foul MAE |
| `min_share` | scalar | mean share of the smallest nonzero-minutes player in a team-game |
| `tail_ratio` | scalar | median share ratio of successive tail ranks (7→13) |
| `alpha`, `alpha_family`, `alpha_starters`, `alpha_bench` | scalars | method of moments on the share residual variance, `Var(s_i) = p_i(1−p_i)/(α+1)` |
| `tilt.state[rank, time, margin]` | 10×5×3 | share of on-floor slot-seconds by rank bucket per state cell, over that bucket's exposure-weighted share across all cells |
| `tilt.foul[fouls, time]` | 6×5 | within-player event study: on-floor rate in the 20 possessions after picking up foul *f* over the 20 before, chained across fouls |
| `n_profile`, `ema_horizon`, `swap_threshold`, `lam_deficit` | scalars | one joint grid against four training-season targets (substitution rate, distinct lineups per team-game, top-5 minutes share, players with > 0 minutes) |
| `hazard_exit`, `hazard_enter` | 18 coefs each | logistic regressions on substitution events derived from on-floor set transitions (R3 only) |
| donor bank | per team-game | the team's own last 5 games' possession-by-possession on-floor sets and state bands (R4 only) |

---

## 2. Feature sets tested

The pre-registration fixes one feature bundle and varies the *sampling scheme*,
so there is no feature-set grid here; the arms differ in how they turn the same
pregame profile and the same game state into five players.

| Set | Included | Used by |
|---|---|---|
| `P_pregame` | the whole of §1.1 | every arm (the profile, availability, starters, foul rate) |
| `S_state` | the whole of §1.2 | R1/R2 through the fitted tilt tables; R3 as hazard features; R4 through the splice band |
| `H_hazard` | `P_pregame` + `S_state` + the 4 player×state interactions | R3 only |

`H_hazard`'s interactions (`is_starter × |margin|`, `is_starter × late`,
`target_share × |margin| × late`, `fouls × late`) are load-bearing rather than
decoration: every feature that is constant across the players of one possession
cancels in the entry softmax and cannot move *who* comes in, so without them the
arm was flat across margin bands in the final eight minutes.

---

## 3. Rejected features

| Feature | Why not |
|---|---|
| hoopR `active` | a placeholder before 2025-26 (`docs/SIM_GUARDRAILS.md` §4); `did_not_play` is the historical availability field |
| CBBD `/teams/roster` membership as the candidate pool | the endpoint returns one season-level snapshot with no as-of date, so a January addition would be visible to a November game. The deep tail is repaired with anonymous fitted tail slots (`extend_profile`) instead |
| CBBD `Substitution` pbp rows | **absent entirely in 2024** and only ~35 per game in 2025 against ~145 in 2026 (see `model.md` §9). Substitution events are derived from on-floor set transitions instead |
| the game's own `PersonalFoul` events as the sim's foul state | contemporaneous with the outcome being simulated — the same defect `is_transition` has at L3 |
| CBBD season ratings / lineup ratings (`/lineups/game`) | end-of-season snapshots, banned as pregame features (L7) |
| a flat expanding `start_freq_asof` as the start ordering | measured 4.22 of 5 starters correct against 4.60 for the fitted EWMA; the metric counts the five the model started, so a wrong fifth man is worth ~2–4 pp of the late-window starters' share on its own |
| `is_transition` / possession-outcome features | the rotation model is called *before* the possession is resolved |
