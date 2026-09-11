# L7 LATE-GAME REGIME -- features

**STATUS: PROPOSED. No feature below has been built, joined or fitted.** This
is the inventory the pre-registered arms in `experiments.md` section 1 would
consume, written before any modelling so that the bundles can be named in the
spec. Companion files: `model.md`, `experiments.md`.

Three rules govern every row and are stated before the table.

- **Strictly-before, always.** Every team-form feature is an expanding mean
  over that team's own prior games only. Every backtest row must satisfy
  `created_at < tipoff`, enforced in code.
- **Relative to the snapshot's own league mean.** Raw rating levels are banned
  (CLAUDE.md modelling rules).
- **Live state, not post-outcome.** Every state column below is read at the
  START of the possession, before the outcome resolves, exactly as
  `_state_block` in `src/cbb_sim/engine/loop.py` assembles it. `duration_s` in
  `possessions_v2` is POST-OUTCOME (the L5 ban) and is a TARGET for the
  duration half, never an input to anything.

## 1. Source manifest

| Feature | Dtype | Source file | Computation | Fallback | Notes |
|---|---|---|---|---|---|
| `period` | float | `possessions_v2/chances_*.parquet` | as-is | -- | already in the engine state block |
| `seconds_remaining` | float | same | `start_clock` | -- | already in the state block |
| `score_diff` | float | same | `start_score_diff`, OFFENCE perspective | -- | already in the state block |
| `in_bonus` | float | same | `off_in_bonus` | 0 | already in the state block |
| `in_double_bonus` | float | same | `off_in_double_bonus` | 0 | **not currently in the state block**; would need adding to `STATE_INDEX` |
| `is_home` | float | `games_universe_v2.parquet` | `offense_is_home` | -- | CLAUDE.md: home/away/neutral is first class in every scoring-stage model |
| `is_neutral` | float | `games_universe_v2.parquet` | neutral-site flag | 0 | the third level of the same feature |
| **`role`** | int8 | derived | `sign(score_diff)` at the possession start: -1 trailing, 0 tied, +1 leading | -- | **the single most important proposed column**; the measured trailing-minus-leading split is -0.158 on three-point share and +0.265 on the bonus-FT rate |
| **`poss_deficit`** | float | derived | `ceil(|score_diff| / 3)`, the number of possessions the trailing team is behind by | -- | the quantity coaches actually act on; 1/2/3-possession games are different regimes at the same clock |
| **`trail_must_foul`** | float | derived | `role == -1 and poss_deficit >= 1 and seconds_remaining <= k`, `k` a pre-declared grid member | 0 | the intentional-foul gate; `k` is a SPEC choice fixed before any fit, never tuned on the outcome |
| **`lead_can_hold`** | float | derived | `role == +1 and seconds_remaining <= shot_clock_remaining_equivalent` | 0 | the hold-the-ball gate |
| **`gt_margin`** | float | derived | `|score_diff|` clipped to [0, 6] | -- | the tie/go-ahead distance; a 3 ties from 3 and a FT ties from 1 |
| `x_role__sec` | float | derived | `role * seconds_remaining / 120` | -- | the interaction the flat sim role split says is missing |
| `x_gt_margin__sec` | float | derived | `gt_margin * seconds_remaining / 120` | -- | |
| `prev_end_*` (5 dummies) | float | same | as the clock model defines them | -- | already in the state block; carried so the duration half keeps its existing conditioning |
| `is_transition` | float | same | as-is | 0 | already in the state block |
| `tempo_prior_game` | float | team block | as the served clock builds it | league mean | already used by the clock's cell grid |
| `off_*` / `opp_def_*` style rates | float | `possession_outcome` round-2 design | as-is, own-snapshot centred | league mean | carried unchanged so arm B is a strict superset of the served bundle |
| `own_ratings` `off_c` / `def_c` | float | `data/processed/ratings` | jointly fitted offence/defence ridge, already opponent-adjusted | league mean | not re-adjusted, per the round-3 note in `possession_outcome/features.md` |
| `is_conf_game` | float | `cbb_sim.features.conference` | as-is | 0 | Decision 9 mandatory bake-off arm wherever team rates are consumed |
| `team_late_foul_rate` | float | derived, as-of | expanding mean of that team's bonus-FT-conceded rate inside the window, strictly prior games | league mean | the responsiveness column for the team-quintile gate; **must pass the leak test** |
| `team_late_3pa_share` | float | derived, as-of | expanding mean of that team's window three-point share, strictly prior games | league mean | same |

**Leak test, mandatory before any of the last two enter a feature table:**
change-form correlation with own-week margin, `|corr| <= 0.15`, honest baseline
0.04-0.08 (CLAUDE.md backtest rules). No external feed is proposed here, so no
KenPom-style leak test applies to the rest.

## 2. Feature sets tested

| Set | Included | Rationale |
|---|---|---|
| `L0_reference` | exactly what the served `possession_outcome` `C_plus_state` and the served clock's cell grid carry today | the reference arm; the thing every candidate must beat beyond the floor |
| `L1_role` | `L0_reference` + `role`, `gt_margin`, `x_role__sec` | the minimum hypothesis: the model is missing WHO IS AHEAD, nothing else. Simplest arm, wins every tie |
| `L2_role_poss` | `L1_role` + `poss_deficit`, `in_double_bonus`, `x_gt_margin__sec` | adds the possession arithmetic a coach acts on |
| `L3_gates` | `L2_role_poss` + `trail_must_foul`, `lead_can_hold` | adds the explicit behavioural gates; the most structured and the least likely to win a tie-break |
| `L4_team` | `L3_gates` + `team_late_foul_rate`, `team_late_3pa_share`, `is_conf_game` | the responsiveness bundle; required for the team-quintile gate to be reachable at all |

Every set carries `is_home` and `is_neutral`. Each is run against the SAME
regime gate; the gate is not a member of any bundle.

## 3. Rejected features

| Feature | Why not |
|---|---|
| a fitted or learned regime BOUNDARY | a free parameter tuned on the gate it is graded against; the hand-tuning pattern CLAUDE.md bans |
| any function of the FINAL score, the number of periods, or the realised OT | outcome leakage; the whole target is the distribution of that quantity |
| `duration_s` as an input | POST-OUTCOME (L5). It is a target, never a feature |
| market lines, spreads, totals, embedded ESPN market columns in pbp | stripped from features by the data rules; and the market scorecard grades this model, so it cannot also feed it |
| timeouts remaining | not in `possessions_v2`; would require a new pbp extraction and is deferred rather than assumed |
| a direct `P(tie)` or "closeness" feature of any kind | would let an arm hit the secondary metric without representing the behaviour; banned by construction |
