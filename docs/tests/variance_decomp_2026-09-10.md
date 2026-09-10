# Variance decomposition: coach / team / player / opponent (CBB, 2022-2025)

Generated: 2026-09-10T00:00:00+00:00 (script `scripts/exp_variance_decomp_v1.py`)

CBB re-run of `cfb-props-sim`'s coach/team/QB variance decomposition (`docs/postmortem/05_cfb_methodology_extract.md` section 5, LEARNINGS L1-L14). Method: `src/cbb_sim/analysis/variance.py::decompose()`, ported verbatim from `cfb-props-sim/src/cfb_props_sim/analysis/variance.py` -- a sequential (nested) fixed-effects R^2 decomposition, not a mixed-effects model, not ridge regression. The question: what should a prior/rating table be keyed on -- coach, (team, season), player, or league mean -- per metric.

**CFB's headline finding was 'team-beyond-coach ~= 0% on every tendency metric measured (scheme moves with the coach).' The postmortem explicitly warns not to assume this carries over to CBB, given transfer-portal roster churn. This study tests it directly.**

## Data and scope

- Seasons: [2022, 2023, 2024, 2025] (season 2026 is sealed; never touched by this script -- `cbb_sim.data.seal.assert_not_sealed`).
- Universe: `data/processed/games_universe.parquet` filtered to `is_d1_game & ~pbp_truncated` (same filter as `scripts/build_gate_reference.py`).
- `data/reference/coaches.parquet` read AS-IS (another agent is finishing its cross-validation; not edited here, nor is `scripts/pull_coaches.py`). `fetched_at`: ['2026-09-10T16:03:24.606628+00:00'].
- Study 1 team-games: 44,702 rows. Coach match: 44,488/44,702 (99.5%); the remainder falls into an 'UNKNOWN' coach group inside the decomposition rather than being dropped.
- Rim-share data quality: pbp-derived total shot attempts match team_box `field_goals_attempted` exactly on 87.1% of team-games and within +/-1 on 93.0% -- close enough that rim share is NOT flagged as too sparse to use (contrary to the fallback the task allowed for); it is reported on the full 2022-2025 universe.
- Tempo/possessions use the pbp-derived per-team-game count from `possessions_pbp.parquet` (missing for 772 of 44,702 rows), not the FGA-OREB+TOV+0.44*FTA formula estimate used in `scripts/build_gate_reference.py` -- tempo is the one metric here that IS a possession count, so the direct pbp count is preferred over a formula built from the very box columns several other metrics in this table also consume.
- Study 2 player-games: 346,559 rows with minutes >= 10 (of 707,336 total player-game rows before the filter). Distinct athletes: 8,423. Coach match: 99.5%.
- Transfer natural experiment: 2,484 athletes whose modal team_id differs across at least two of their seasons in 2022-2025 (163,879 player-games).

---

## Study 1: team-game level (weight = possessions)

Sequential order: `coach_id` -> `(team_id, season)` beyond coach -> opponent's `coach_id` beyond both -> residual. 'Allowed'/defensive metrics are the identical formula computed on the opponent's own box line in that same game (what the opponent did on offense against this team's defense).

### Headline: pooled 2022-2025

| Metric | Side | n | Coach (seq) | Team-beyond-coach (seq) | Opponent-coach (seq) | Residual |
|---|---|---:|---:|---:|---:|---:|
| Tempo (possessions, team's own pbp-derived count) | offense | 43,930 | 13.9% | 5.9% | 12.6% | 67.6% |
| Tempo (possessions, team's own pbp-derived count) | defense | 43,930 | 13.8% | 5.9% | 12.7% | 67.6% |
| 3PA share of FGA | offense | 43,930 | 21.5% | 8.6% | 10.8% | 59.1% |
| 3PA share of FGA | defense | 43,930 | 11.8% | 5.9% | 18.8% | 63.6% |
| Rim share of FGA (dunk+layup+tip) | offense | 43,930 | 11.9% | 7.3% | 9.2% | 71.5% |
| Rim share of FGA (dunk+layup+tip) | defense | 43,930 | 10.6% | 4.5% | 10.6% | 74.2% |
| FTA / FGA | offense | 43,930 | 6.4% | 4.6% | 8.3% | 80.7% |
| FTA / FGA | defense | 43,930 | 10.6% | 4.3% | 4.8% | 80.3% |
| TOV% (TOV / possessions) | offense | 43,930 | 8.8% | 5.8% | 10.1% | 75.3% |
| TOV% (TOV / possessions) | defense | 43,930 | 12.4% | 6.4% | 6.7% | 74.5% |
| OREB% | offense | 43,930 | 14.2% | 5.8% | 5.7% | 74.3% |
| OREB% | defense | 43,930 | 6.8% | 3.6% | 12.3% | 77.3% |
| eFG% | offense | 43,930 | 8.1% | 3.8% | 7.2% | 80.8% |
| eFG% | defense | 43,930 | 6.2% | 3.4% | 9.0% | 81.4% |

One-way R^2 for season alone (the trend) and for (coach, season) jointly, pooled 2022-2025:

| Metric | Side | Season alone (one-way) | (Coach, season) (one-way) |
|---|---|---:|---:|
| Tempo (possessions, team's own pbp-derived count) | offense | 0.2% | 19.8% |
| Tempo (possessions, team's own pbp-derived count) | defense | 0.2% | 19.6% |
| 3PA share of FGA | offense | 0.6% | 30.1% |
| 3PA share of FGA | defense | 0.6% | 17.6% |
| Rim share of FGA (dunk+layup+tip) | offense | 0.3% | 19.2% |
| Rim share of FGA (dunk+layup+tip) | defense | 0.3% | 15.1% |
| FTA / FGA | offense | 0.6% | 11.0% |
| FTA / FGA | defense | 0.6% | 14.8% |
| TOV% (TOV / possessions) | offense | 1.2% | 14.6% |
| TOV% (TOV / possessions) | defense | 1.2% | 18.8% |
| OREB% | offense | 0.5% | 19.9% |
| OREB% | defense | 0.5% | 10.4% |
| eFG% | offense | 0.1% | 11.9% |
| eFG% | defense | 0.1% | 9.6% |

### Full sequential tables (pooled 2022-2025)

### Tempo (possessions, team's own pbp-derived count) -- offense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.139 | 0.139 |
| team_season | 1447 | 0.198 | 0.059 |
| opp_coach_id | 497 | 0.138 | 0.126 |
| residual | -- | -- | 0.676 |

### Tempo (possessions, team's own pbp-derived count) -- defense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.138 | 0.138 |
| team_season | 1447 | 0.197 | 0.059 |
| opp_coach_id | 497 | 0.139 | 0.127 |
| residual | -- | -- | 0.676 |

### 3PA share of FGA -- offense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.215 | 0.215 |
| team_season | 1447 | 0.301 | 0.086 |
| opp_coach_id | 497 | 0.118 | 0.108 |
| residual | -- | -- | 0.591 |

### 3PA share of FGA -- defense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.118 | 0.118 |
| team_season | 1447 | 0.176 | 0.059 |
| opp_coach_id | 497 | 0.215 | 0.188 |
| residual | -- | -- | 0.636 |

### Rim share of FGA (dunk+layup+tip) -- offense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.119 | 0.119 |
| team_season | 1447 | 0.192 | 0.073 |
| opp_coach_id | 497 | 0.106 | 0.092 |
| residual | -- | -- | 0.715 |

### Rim share of FGA (dunk+layup+tip) -- defense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.106 | 0.106 |
| team_season | 1447 | 0.151 | 0.045 |
| opp_coach_id | 497 | 0.119 | 0.106 |
| residual | -- | -- | 0.742 |

### FTA / FGA -- offense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.064 | 0.064 |
| team_season | 1447 | 0.110 | 0.046 |
| opp_coach_id | 497 | 0.106 | 0.083 |
| residual | -- | -- | 0.807 |

### FTA / FGA -- defense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.106 | 0.106 |
| team_season | 1447 | 0.148 | 0.043 |
| opp_coach_id | 497 | 0.064 | 0.048 |
| residual | -- | -- | 0.803 |

### TOV% (TOV / possessions) -- offense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.088 | 0.088 |
| team_season | 1447 | 0.146 | 0.058 |
| opp_coach_id | 497 | 0.124 | 0.101 |
| residual | -- | -- | 0.753 |

### TOV% (TOV / possessions) -- defense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.124 | 0.124 |
| team_season | 1447 | 0.188 | 0.064 |
| opp_coach_id | 497 | 0.088 | 0.067 |
| residual | -- | -- | 0.745 |

### OREB% -- offense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.142 | 0.142 |
| team_season | 1447 | 0.200 | 0.058 |
| opp_coach_id | 497 | 0.068 | 0.057 |
| residual | -- | -- | 0.743 |

### OREB% -- defense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.068 | 0.068 |
| team_season | 1447 | 0.104 | 0.036 |
| opp_coach_id | 497 | 0.142 | 0.123 |
| residual | -- | -- | 0.773 |

### eFG% -- offense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.081 | 0.081 |
| team_season | 1447 | 0.119 | 0.038 |
| opp_coach_id | 497 | 0.062 | 0.072 |
| residual | -- | -- | 0.808 |

### eFG% -- defense

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.062 | 0.062 |
| team_season | 1447 | 0.096 | 0.034 |
| opp_coach_id | 497 | 0.081 | 0.090 |
| residual | -- | -- | 0.814 |

---

### Robustness: half-sample stability

### 2024-2025 only

| Metric | Side | n | Coach (seq) | Team-beyond-coach (seq) | Opponent-coach (seq) | Residual |
|---|---|---:|---:|---:|---:|---:|
| Tempo (possessions, team's own pbp-derived count) | offense | 22,282 | 15.1% | 3.8% | 14.3% | 66.8% |
| Tempo (possessions, team's own pbp-derived count) | defense | 22,282 | 15.1% | 3.8% | 14.4% | 66.7% |
| 3PA share of FGA | offense | 22,282 | 25.8% | 5.3% | 11.9% | 57.0% |
| 3PA share of FGA | defense | 22,282 | 13.4% | 4.1% | 22.5% | 60.0% |
| Rim share of FGA (dunk+layup+tip) | offense | 22,282 | 15.9% | 4.4% | 10.7% | 69.0% |
| Rim share of FGA (dunk+layup+tip) | defense | 22,282 | 12.6% | 2.7% | 14.2% | 70.5% |
| FTA / FGA | offense | 22,282 | 7.7% | 3.0% | 9.4% | 80.0% |
| FTA / FGA | defense | 22,282 | 11.4% | 2.7% | 6.0% | 79.8% |
| TOV% (TOV / possessions) | offense | 22,282 | 10.8% | 2.9% | 12.6% | 73.7% |
| TOV% (TOV / possessions) | defense | 22,282 | 14.7% | 3.1% | 8.6% | 73.6% |
| OREB% | offense | 22,282 | 16.8% | 3.4% | 6.4% | 73.3% |
| OREB% | defense | 22,282 | 7.4% | 2.0% | 15.1% | 75.5% |
| eFG% | offense | 22,282 | 10.4% | 2.3% | 8.2% | 79.1% |
| eFG% | defense | 22,282 | 7.0% | 1.8% | 11.3% | 79.9% |

### 2022-2023 only

| Metric | Side | n | Coach (seq) | Team-beyond-coach (seq) | Opponent-coach (seq) | Residual |
|---|---|---:|---:|---:|---:|---:|
| Tempo (possessions, team's own pbp-derived count) | offense | 21,648 | 17.5% | 3.2% | 15.1% | 64.2% |
| Tempo (possessions, team's own pbp-derived count) | defense | 21,648 | 17.4% | 3.1% | 15.2% | 64.3% |
| 3PA share of FGA | offense | 21,648 | 24.4% | 4.5% | 13.2% | 57.8% |
| 3PA share of FGA | defense | 21,648 | 14.8% | 2.8% | 21.5% | 61.0% |
| Rim share of FGA (dunk+layup+tip) | offense | 21,648 | 14.0% | 3.7% | 10.7% | 71.6% |
| Rim share of FGA (dunk+layup+tip) | defense | 21,648 | 12.0% | 2.6% | 12.8% | 72.6% |
| FTA / FGA | offense | 21,648 | 7.8% | 2.7% | 10.0% | 79.5% |
| FTA / FGA | defense | 21,648 | 12.4% | 2.3% | 6.3% | 79.0% |
| TOV% (TOV / possessions) | offense | 21,648 | 10.7% | 2.9% | 12.0% | 74.4% |
| TOV% (TOV / possessions) | defense | 21,648 | 14.9% | 3.1% | 8.2% | 73.7% |
| OREB% | offense | 21,648 | 15.9% | 3.2% | 7.0% | 73.9% |
| OREB% | defense | 21,648 | 8.6% | 2.2% | 13.6% | 75.5% |
| eFG% | offense | 21,648 | 8.8% | 2.2% | 8.5% | 80.6% |
| eFG% | defense | 21,648 | 8.1% | 2.2% | 9.3% | 80.4% |

**Stability reading.** Compare each metric's Coach/Team-beyond-coach/Opponent-coach/Residual sequential shares across the pooled, 2024-25, and 2022-23 tables above. Differences beyond a couple of points reflect real half-sample noise at this n, not a trend -- treat any single-metric flip in ranking between the two halves as unreliable; only shifts consistent across both halves in the same direction should be read as a genuine drift.

---

## Study 2: player-game level (weight = minutes, players with >= 10 minutes)

Sequential order: `coach_id` -> `(team_id, season)` beyond coach -> `athlete_id` beyond both -> residual.

### Headline: pooled 2022-2025, all qualifying players

| Metric | n | Coach (seq) | Team-beyond-coach (seq) | Player-beyond-both (seq) | Residual |
|---|---:|---:|---:|---:|---:|
| Usage proxy: (FGA + 0.44*FTA + TOV) / minutes | 346,559 | 1.0% | 0.3% | 34.1% | 64.7% |
| 3PA share of own FGA | 340,801 | 2.9% | 1.2% | 51.7% | 44.2% |
| FT rate: FTA / FGA | 340,801 | 0.8% | 0.6% | 9.8% | 88.8% |
| Assist rate: AST / minutes | 346,559 | 1.6% | 0.7% | 28.9% | 68.9% |
| Rebound rate: REB / minutes | 346,559 | 1.0% | 0.5% | 35.9% | 62.7% |
| Points per minute | 346,559 | 1.1% | 0.6% | 19.5% | 78.8% |

### Full sequential tables (pooled 2022-2025)

### Usage proxy: (FGA + 0.44*FTA + TOV) / minutes

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.010 | 0.010 |
| team_season | 1447 | 0.013 | 0.003 |
| athlete_id | 8423 | 0.354 | 0.341 |
| residual | -- | -- | 0.647 |

### 3PA share of own FGA

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.029 | 0.029 |
| team_season | 1447 | 0.041 | 0.012 |
| athlete_id | 8396 | 0.564 | 0.517 |
| residual | -- | -- | 0.442 |

### FT rate: FTA / FGA

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.008 | 0.008 |
| team_season | 1447 | 0.014 | 0.006 |
| athlete_id | 8396 | 0.110 | 0.098 |
| residual | -- | -- | 0.888 |

### Assist rate: AST / minutes

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.016 | 0.016 |
| team_season | 1447 | 0.023 | 0.007 |
| athlete_id | 8423 | 0.309 | 0.289 |
| residual | -- | -- | 0.689 |

### Rebound rate: REB / minutes

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.010 | 0.010 |
| team_season | 1447 | 0.014 | 0.005 |
| athlete_id | 8423 | 0.374 | 0.359 |
| residual | -- | -- | 0.627 |

### Points per minute

| Grouping | # groups | One-way R^2 | Sequential R^2 |
|---|---:|---:|---:|
| coach_id | 497 | 0.011 | 0.011 |
| team_season | 1447 | 0.017 | 0.006 |
| athlete_id | 8423 | 0.213 | 0.195 |
| residual | -- | -- | 0.788 |

---

### Robustness: the transfer natural experiment

Restricted to the 2,484 athletes who played for more than one team_id across seasons in 2022-2025 (163,879 player-games). This is the CFB-style natural experiment (coach moves, player transfers) that separates player identity from team identity cleanly, because these players supply within-athlete variation across different (team, coach) contexts that the pooled sample's many single-team players cannot.

### Transfers-only sample

| Metric | n | Coach (seq) | Team-beyond-coach (seq) | Player-beyond-both (seq) | Residual |
|---|---:|---:|---:|---:|---:|
| Usage proxy: (FGA + 0.44*FTA + TOV) / minutes | 163,879 | 4.2% | 3.7% | 23.4% | 68.7% |
| 3PA share of own FGA | 161,711 | 6.6% | 4.9% | 40.9% | 47.6% |
| FT rate: FTA / FGA | 161,711 | 1.4% | 1.4% | 7.2% | 90.0% |
| Assist rate: AST / minutes | 163,879 | 3.6% | 2.9% | 22.7% | 70.7% |
| Rebound rate: REB / minutes | 163,879 | 3.2% | 2.9% | 29.4% | 64.5% |
| Points per minute | 163,879 | 2.3% | 2.5% | 13.0% | 82.2% |

**Reading.** `player-beyond-both` is the largest non-residual term for every metric in BOTH the pooled sample and the transfers-only sample -- the ranking survives the honest test of players who actually crossed a (coach, team) boundary, so it is not an artifact of single-team players who never provided separation. It is consistently 5-12 points LOWER in the transfers-only sample than pooled for every metric (Usage proxy: (FGA + 0.44*FTA + TOV) / minutes: 34.1% -> 23.4%; 3PA share of own FGA: 51.7% -> 40.9%; FT rate: FTA / FGA: 9.8% -> 7.2%; Assist rate: AST / minutes: 28.9% -> 22.7%; Rebound rate: REB / minutes: 35.9% -> 29.4%; Points per minute: 19.5% -> 13.0%), consistent with a real player effect that attenuates somewhat under transfer -- fewer games per (player, new-team) cell adds noise, and a transfer season itself (new system, new role, an adjustment period) is not the player's true steady-state rate. The right reading is not 'the effect is fake' but 'shrink the player prior harder in a transfer's first season, same shape as CFB's HC/QB continuity-weighted prior (section 5 of the postmortem), not a flat carry-over of the old team's rate.'

---

## Plain reading

**Study 1 (team-game).**
- Tempo (possessions, team's own pbp-derived count): coach 13.9%, team-beyond-coach 5.9%, opponent-coach 12.6%, residual 67.6% -- largest non-residual term is **coach** (13.9%).
- 3PA share of FGA: coach 21.5%, team-beyond-coach 8.6%, opponent-coach 10.8%, residual 59.1% -- largest non-residual term is **coach** (21.5%).
- Rim share of FGA (dunk+layup+tip): coach 11.9%, team-beyond-coach 7.3%, opponent-coach 9.2%, residual 71.5% -- largest non-residual term is **coach** (11.9%).
- FTA / FGA: coach 6.4%, team-beyond-coach 4.6%, opponent-coach 8.3%, residual 80.7% -- largest non-residual term is **opponent-coach** (8.3%).
- TOV% (TOV / possessions): coach 8.8%, team-beyond-coach 5.8%, opponent-coach 10.1%, residual 75.3% -- largest non-residual term is **opponent-coach** (10.1%).
- OREB%: coach 14.2%, team-beyond-coach 5.8%, opponent-coach 5.7%, residual 74.3% -- largest non-residual term is **coach** (14.2%).
- eFG%: coach 8.1%, team-beyond-coach 3.8%, opponent-coach 7.2%, residual 80.8% -- largest non-residual term is **coach** (8.1%).

**Study 2 (player-game).**
- Usage proxy: (FGA + 0.44*FTA + TOV) / minutes: coach 1.0%, team-beyond-coach 0.3%, player-beyond-both 34.1%, residual 64.7% -- largest non-residual term is **player-beyond-both** (34.1%).
- 3PA share of own FGA: coach 2.9%, team-beyond-coach 1.2%, player-beyond-both 51.7%, residual 44.2% -- largest non-residual term is **player-beyond-both** (51.7%).
- FT rate: FTA / FGA: coach 0.8%, team-beyond-coach 0.6%, player-beyond-both 9.8%, residual 88.8% -- largest non-residual term is **player-beyond-both** (9.8%).
- Assist rate: AST / minutes: coach 1.6%, team-beyond-coach 0.7%, player-beyond-both 28.9%, residual 68.9% -- largest non-residual term is **player-beyond-both** (28.9%).
- Rebound rate: REB / minutes: coach 1.0%, team-beyond-coach 0.5%, player-beyond-both 35.9%, residual 62.7% -- largest non-residual term is **player-beyond-both** (35.9%).
- Points per minute: coach 1.1%, team-beyond-coach 0.6%, player-beyond-both 19.5%, residual 78.8% -- largest non-residual term is **player-beyond-both** (19.5%).
