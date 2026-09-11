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

---

## 4. Round 4: the per-player substitution-hazard feature set (`rotation_v4.SUB_FEATURES`)

Round 4 changed model family (L25, `experiments.md` §10). The family is two
per-player discrete-time hazards evaluated at every possession boundary — a
sub-out hazard over the five on the floor and a sub-in hazard over the eligible
bench — so its feature set is a *decision* feature set, not a budget feature
set, and it replaces §1.2's "minutes played so far vs target / recent on-floor
rate" pair rather than adding to it.

**45 features, one design function.** `rotation_v4.design()` is written over
(M, S) arrays and is called with M = 1 by the offline sampler and M = 2N by
`engine/rotation_adapter.py`, so the bake-off and the engine cannot drift apart;
`tests/test_rotation_v4.py` pins that equality, the feature order, and the
`period_boundary` coding.

| group | features | source at fit time | source in the engine |
|---|---|---|---|
| player role | `is_starter` (as-of `start_rank_asof` ≤ 5), `share` (as-of minutes share × 5, normalised over **all** candidates) | §1.1 | `inp.rot_srank`, `inp.rot_share` |
| player fouls | `fouls`, `foul_out`, `fouls × is_starter`, `fouls × late` | the training game's own `PersonalFoul` events | the (2N, S) `GameState.player_fouls` block, **simulated** |
| player fatigue | `state_min` (minutes in the current on/off state — stint if on, rest if off), `half_min` (minutes played so far in this half), `half_min_dev` = `half_min − share ×` elapsed half minutes | on-floor stream | rotation-batch state |
| clock | `sec_left_frac`, `is_ot`, eight **time-cell dummies** (the nine audit cells, `H1 20:00–10:00` the reference) | `period`, `start_clock` | `st.period`, `st.seconds_remaining` |
| margin | `abs_margin`, `mb_6_15`, `mb_gt15` | `start_score_diff` re-signed | `st.home_score_diff()` re-signed |
| dead ball | `period_boundary`, `dead_made_ft`, `dead_tov`, `dead_other` — the `start_reason` levels, which are exactly `engine.state.PREV_END_LEVELS` | possessions `start_reason` | `st.prev_end` |
| team fouls | `team_fouls_frac` (own team fouls / 5) | possessions `off_team_fouls`/`def_team_fouls` re-signed to the team | `st.team_fouls[:, side]` |
| interactions | `is_starter ×` each time-cell dummy, each margin-band dummy, `period_boundary`, `sec_left_frac`, `is_close × late`, `is_blowout × late`, `abs_margin`; `share ×` `late`, `mb_gt15`, `period_boundary` | — | — |

**Why the design is saturated in (time cell × margin band × is_starter).** The
audit (`docs/tests/rotation_sub_hazard_audit_2026-09-10.md` §1) measures a
starter's exit hazard running 0.035 → 0.020 → 0.054 → 0.027 → 0.037 across the
nine cells — non-monotone — a bench player's exit hazard jumping to 0.244 in
H2 20:00–16:00, and the starter/bench ordering reversing sign in the final two
minutes of a blowout. A linear time term cannot represent that shape. The gate
reads *occupancy* in some of those cells, which is a different functional of the
process (an equilibrium under the five-on-the-floor constraint), and §10.9's
reachability probe is what makes the claim falsifiable.

### 4.1 Round-4 rejected features (measured, then excluded)

| Feature | Measured effect | Why not |
|---|---|---|
| timeout at this stoppage (CBBD `OfficialTVTimeOut` / `ShortTimeOut` / `RegularTimeOut`) | multiplies every hazard 3–4× (starter exit 0.031 → 0.136, starter entry 0.068 → 0.243, n = 4.0M / 162k) | the engine has no timeout model, so a hazard conditioned on it could be fitted offline and could never be evaluated in simulation — the same exclusion class as `is_transition`. Its cost is visible: the arms' substitutions are more uniform in time than real ones, which shows up in the substitution rate and distinct-lineup counts |
| prior-season minutes share (hoopR `player_box` of season − 1 through the crosswalk) | a real gradient, same sign as the as-of share and about half its size (P5 − P1: −0.9 pp on starter exit, +1.8 pp on starter entry, against −1.3 / +2.8 for the as-of share) | a near-duplicate of a feature already in the design, and not expressible in the engine without a new per-roster-slot input array, i.e. without rebuilding `arrays_F2_2025.npz` while other workers read it |
| the R1/R2 scheduler's `minutes played so far vs target` and EMA on-floor rate | — | superseded: `half_min_dev` carries the budget deviation and `state_min` carries the stint, both as hazard covariates rather than as a scheduler utility |

---

## 5. Round 5: the wave cell (`rotation_v5.wave_cell`)

Round 5 changed the DRAW, not the family (L30, `experiments.md` §12). Round 4's
45 hazard features of §4 are reused **byte for byte** as the composition rule —
nothing in §4 is refitted, re-specified or re-ordered — and round 5 adds exactly
one new object: a cell index over which a per-(team, boundary) wave probability
and a wave-size distribution are tabulated.

**The cell: 324 = 6 × 9 × 3 × 2.** Every component is already in §4's table, so
round 5 adds no new *source* and no new join.

| component | levels | source at fit time | source in the engine |
|---|---|---|---|
| `prev_end` | 6 (`period_start`, `DREB`, `TOV`, `made_FG`, `made_FT`, `other`) | possessions `start_reason` | `st.prev_end` |
| time cell | 9 (`rotation_v4.time_cell`, the audit's nine cells) | `period`, `start_clock` | `st.period`, `st.seconds_remaining` |
| margin band | 3 (`rotation.margin_bucket`) | `start_score_diff` re-signed | `st.home_score_diff()` re-signed |
| foul state | 2: any player ON THE FLOOR carrying ≥ 4 personal fouls | the training game's own `PersonalFoul` events | the (2N, S) `GameState.player_fouls` block, **simulated** |

**The two tabulated objects.** `p_wave[cell]` (a Bernoulli) and
`p_size[cell, 1..5]` (a categorical), both fitted by two-level shrinkage —
cell → (`prev_end` × time cell) → `prev_end` → root — with the shrinkage
constant fixed at `k = 300`, the project's UNDERPOWERED threshold, declared in
the pre-registration and never tuned. Each parent is a marginalisation of the
same counts, so every level is the maximum-likelihood estimate of its own
coarser model. Both are lookup tables by construction: the sim loop makes no
model call.

**One fitted scalar beyond the tables.** `rho`, the probability that the two
teams of a game share the dead-ball uniform at a boundary, fitted by moment
matching on the training window's joint counts. It changes no team's marginal.

**Why the foul state is a team-level indicator and not a per-player count.** The
per-player foul count is already in §4's design and reaches the composition
step, where it belongs (it decides *who* moves). The cell decides *whether the
bench moves at all*, which the audit measures as a team-level event: P(wave) is
0.1504 with no 4-foul player on the floor and 0.1707 with one
(`docs/tests/rotation_wave_audit_2026-09-11.md` §1.2). A per-player count in the
cell would multiply the table by five for a 2 pp effect already carried once.

### 5.1 Round-5 rejected features (measured, then excluded)

| Feature | Measured effect | Why not |
|---|---|---|
| timeout at this stoppage | P(wave) **0.1371 → 0.4967**, and P(size ≥ 2 \| wave) 0.3165 → 0.4146; timeouts carry **13.3%** of all waves (audit §2) | unchanged from §4.1: the engine has no timeout model. Round 5 is the test of whether the wave draw makes it unnecessary — the bunching a timeout produces is now in the draw itself rather than left to five independent coins |
| own team fouls (the bonus state) as a cell axis | P(wave) 0.1214 → 0.1898 across 0-2 → 12+ (audit §1.2) | mostly the same information seen twice: team fouls buy free throws and `made_FT` is already the high-wave `prev_end` level (0.393 against 0.108 at `made_FG`). It stays where round 4 put it, as one linear term inside the composition hazards |
| a per-`prev_end` size distribution finer than the cell | the period boundary IS a different distribution (0.41 / 0.41 / 0.15 against 0.68 / 0.25 / 0.06), and waves shrink through the game (0.595 single swaps in H1 20:00–10:00 → 0.773 in the last two minutes → 0.884 in OT) | already carried: the size categorical is indexed by the SAME cell as the wave probability, so both facts are in the table without a second object |
