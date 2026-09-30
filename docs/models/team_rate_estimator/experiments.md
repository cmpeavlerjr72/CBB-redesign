# Team-rate estimator: experiments (append-only)

**Owner:** Lane F (2026-09-30).

**What this is.** The estimator that turns a team's season-to-date box counts into the as-of, league-centred team rates. possession_outcome, fg_make and rebound consume those rates as features.

- **Origin:** the G9 diagnostic, `docs/tests/g9_g6_margin_slope_home_diagnostic_2026-09-30.md` section 6. There the served estimator (an expanding mean from 0.0) owns 0.064 of the 0.091 close-referenced slope miss, through over-reaction to recent games (lag-1 movement ratio 0.38-0.83).
- **Status:** NOTHING in this file adopts anything or changes a served default. The served feature tables are untouched. The PM decides from the results section.
- **Other docs:** `model.md` and `features.md` are written if and when an arm is adopted. This folder holds only the pre-registration and results until then.

---

## 1. Pre-registration: Stage A estimator bake-off

Written 2026-09-30 about 11:40 EDT. It is COMMITTED BEFORE ANY FIT.

### 1.1 Scope and stages

- **Stage A (this round).** An offline estimator bake-off. There is no sub-model retrain and no sim.
- **Stage B (later).** Retrain possession_outcome, fg_make and rebound under S1 on the Stage A winner's feature build. This is sequenced by the PM after Lane C (season-drift anchor) reports.
- **Stage C (later).** A paired 200-seed closed loop. Primary: G9 slope. Co-primary: slope(close on X) and the close-referenced k_within, recomputed with `scripts/diag_g9_g6_margin_v1.py --part close`.
- **Constraint.** Damp unjustified within-season movement AT THE ESTIMATE. Team levels must not shrink. No sim-output damping of any kind.

### 1.2 Targets (rates x side)

Per team-game:

- **Box counts:** hoopR team box via `cbb_sim.eval.reference.load_actual_team_box`.
- **Rim/jump split:** the 2PT attempts and makes are split by the CBBD event-layer shares (`load_team_shot_truth`, v2). Where the event layer is missing, the league split is used.
- **Seasons:** 2022-2025. 2026 is SEALED and not loaded.

| rate | numerator / denominator | likelihood |
|---|---|---|
| tov | TOV / P, with P = FGA - OREB + TOV + 0.44 FTA | binomial (quasi, non-integer P) |
| ftr | FTA / FGA | Poisson with exposure FGA |
| share3 | 3PA / FGA | binomial |
| share_rim | rim FGA / FGA | binomial |
| make_rim | rim FGM / rim FGA | binomial |
| make_jump | jump2 FGM / jump2 FGA | binomial |
| make3 | 3PM / 3PA | binomial |
| oreb | OREB / (OREB + opponent DREB) | binomial |

**Sides:**

- **off:** the team's own rate.
- **def:** what opponents did against the team (for oreb, the opponent's OREB% against it).

That gives 16 rate-sides.

### 1.3 League centring (CLAUDE.md modelling rule; amendment 5)

Every estimate is expressed relative to L(s, t), the league's cumulative rate in season s over team-games strictly before date t.

**On day 0** (no games yet), every arm uses L = the prior season's FINAL league level L_end(s-1). This is the default. For 2022, which has no prior season in the panel, the day-0 level is 2022's own first-week pooled level. 2022 rows are used only as a prior source and are never scored.

Lane C is testing alternative anchors, and its result will be folded in at Stage B. No arm here varies the league-level treatment. Arms differ only in the team-level estimate c (centred), with prediction p = L + c.

### 1.4 Arms

Notation:

- N, D are the team's season-to-date numerator and denominator (games strictly before), and n is the number of games.
- c_prev is the team's prior-season FINAL centred rate (its season rate minus that season's league rate).

Every free parameter is FITTED on the fold's TRAINING seasons that have a prior season: fold 1 trains on 2023, fold 2 on 2023 and 2024. The fit maximises next-game likelihood of the side's realised rate. It is never fitted on a test season.

| arm | estimate c | free parameters | simplicity rank |
|---|---|---|---|
| **E0** (served reference) | D > 0: N/D - L; D = 0: 0 | none | 0 |
| **E1** reliability-weighted | D (N/D - L) / (D + k) | k | 1 |
| **E2** E1 plus prior-season carry | (D (N/D - L) + k rho c_prev) / (D + k) | k, rho | 2 |
| **E2c** E2 with continuity-dependent carry | as E2, with rho = rho0 + rho1 (cont - mean cont) + rho2 coach_change | k, rho0, rho1, rho2 | 3 |
| **E4** exponentially weighted with the E2 prior | weights lambda^(games ago) on each past game's (num, den), plus the E2 prior pseudo-count | lambda, k, rho | 4 |
| **E3** local-level state-space (Kalman) | c_0 = rho c_prev with variance P0. Each game adds an observation of the centred rate with binomial / Poisson variance at L; process variance q per game. c is the filtered mean strictly before tipoff. | q, P0, rho | 5 |
| **E9** (optional, amendment 6) | opponent adjustment on top of the Stage A winner | as the winner | - |

For E9, each past game's observation is corrected by the opponent's as-of estimate on the opposite side. It runs only if time remains.

**Carry inputs (E2, E2c, E3, E4)**

- If a team has no prior season in the panel, c_prev = 0.
- **cont (E2c):** the team's returning share of prior-season minutes. Built as-of preseason from hoopR `player_box`, for every panel season: the team's s-1 minutes played by athletes listed on the team's box (including DNP rows) in any of its first 3 games of season s, divided by the team's total s-1 minutes. This is a roster proxy, because the listing is public preseason. It is built for 2023-2025, and 2022 has no prior season.
- **coach_change (E2c):** `data/reference/coaches.parquet`, where head_coach(s) differs from head_coach(s-1) or `interim` is set.
- If this cannot be built cleanly today, E2c is registered-not-run and is NOT approximated.

### 1.5 Primary metric, segments, guards

**Primary.** Fold 2 (test 2025) next-game predictive deviance, summed over the 16 rate-sides:

- the pooled deviance gain of an arm is sum(D_E0) - sum(D_arm), over all team-games with a scored target;
- it is also reported per rate-side and per weeks band (0-3, 4-7, 8-15, 16+);
- fold 1 (test 2024) is the confirmation.

**Mandatory segment: day-0 / first-games cell (amendment 1).** A team's games 1, 2-3 and 4-6 of the season are scored separately, per arm, rate and side, with:

- the deviance gain vs E0;
- the team slope (realised on predicted, weighted).

E0 and E1 are identical at game 1 by construction (c = 0), so their game-1 slope is undefined, and that is reported as such.

**Noise floor (amendment 3).** A team-block bootstrap of the paired deviance difference vs E0: resample teams with replacement, 1,000 draws. It gives a 95% interval per rate-side and pooled. The fold-1-to-fold-2 transfer is the confirmation line.

**Guards (amendment 4).** Every one is reported per rate-side.

- **G-A1 (levels not shrunk).** Ratio R = SD across teams of the team's mean as-of estimate over its season, divided by SD across teams of the realised team season rate. R(arm) must not fall below R(E0) pooled; the per-rate exceptions are listed.
- **G-A2 (movement).** The lag-1 movement ratio b_move / b_level from y = a + b_level c_lag1 + b_move (c - c_lag1) + L must move toward 1 relative to E0, pooled over rate-sides (median over rate-sides reported).
- **G-A3 (day-0 cell).** The team slope in the games-1 and 2-3 cells is reported.

**Decision rule (amendment 7).** A winner must satisfy all of:

- its pooled fold-2 bootstrap interval of the gain vs E0 excludes zero on the positive side;
- the fold-1 pooled gain has the same sign;
- G-A1 and G-A2 hold.

Among qualifying arms, the higher pooled fold-2 gain wins, unless its interval overlaps the next-simpler qualifying arm's point gain. Ties go to the simpler arm, in the order E1, E2, E2c, E4, E3. Selection is on fold 2, and 2025-26 is SEALED.

### 1.6 Execution

- **Fitting and estimates:** `scripts/exp_team_rate_estimator_v1.py`.
- **Grading, blind across all arms:** `scripts/grade_team_rate_estimator_v1.py`. One grader scores every arm from the same estimate table, keyed by arm.
- **Outputs:** versioned files under `results/team_rate_estimator/`. No served table, engine input or other lane's file is written.
- **Results:** appended as section 2 of this file, once, when final, plus `docs/tests/team_rate_estimator_stageA_2026-09-30.md` for long tables.

---

## 2. Results: Stage A (run 2026-09-30, about 11:33-11:37 EDT; 1 core; about 2 minutes of fitting plus grading)

Full tables: `docs/tests/team_rate_estimator_stageA_2026-09-30.md`.

**What ran.** Every arm registered in section 1.4 except E9. The E2c continuity build ran cleanly:

- **Source:** hoopR `player_box` 2022-2025, with the roster read from box listings (including DNP rows) in each team's first 3 games.
- **Filter:** restricted to teams with at least 2,000 prior-season minutes, i.e. full D-I seasons.
- **Median returning-minutes share by season:** 2023 0.495, 2024 0.457, 2025 0.407.
- **Coach change rate:** about 9% of teams per season.

**Not run: E9 (opponent adjustment).** It is registered on top of a qualifying winner, and none qualifies (2.3).

### 2.1 Pooled primary: fold-2 next-game deviance gain vs E0 (16 rate-sides)

| arm | F2 gain (%) | F2 95% team-block interval | F1 gain (%) | level-SD ratio (G-A1, median) | lag-1 move ratio (G-A2, median) | day-0 g1 gain | g1 slope (median) | g2-3 slope |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| E0 | 0 | - | 0 | 1.133 | 0.694 | 0 | undefined | 0.179 |
| E1 | 28,731 (8.37) | [27,035, 30,778] | 29,492 (8.62) | 0.611 | 1.424 | 0 (= E0 by construction) | undefined | 1.031 |
| E2 | 31,281 (9.11) | [29,524, 33,487] | 32,531 (9.50) | 0.694 | 1.465 | 475 | 0.916 | 1.068 |
| E2c | 31,504 (9.18) | [29,793, 33,687] | 32,653 (9.54) | 0.688 | 1.442 | **534** | **0.997** | 1.104 |
| E4 | **32,068 (9.34)** | [30,309, 34,255] | 33,497 (9.79) | 0.673 | **0.940** | 476 | 0.924 | 0.989 |
| E3 | 32,045 (9.33) | [30,219, 34,232] | 33,537 (9.80) | 0.704 | 1.240 | 476 | 0.915 | 1.074 |

**Gains are positive everywhere.** Every arm beats E0:

- on every one of the 16 rate-sides;
- in both folds;
- with no rate-side interval touching zero.

**Where the gain comes from.**

- By weeks: the gain is concentrated early (weeks 0-3: 22-24%; 4-7: 6-7%; 8-15: 3%; 16+: 1-2%).
- E3 and E4 keep about 2% in weeks 16+, where E1, E2 and E2c keep about 1.2%. This is the part a recency weight adds.
- **E0 over-reacts badly early.** Its team slope in games 2-3 is 0.18, and 0.28 in games 4-6.

### 2.2 Per-rate exceptions

**ftr off, game 1.** E2, E4 and E3 LOSE 0.36-0.39% vs E0; the carried prior-season FT rate is worse than the league level. E2c is the exception at +0.12%.

**make3 def.** This side carries almost no team signal.
- Game-1 slope is 0.30. Game 2-3 slopes are negative in every arm (-0.03 to -1.0).
- The level-SD ratio collapses to 0.17-0.28, which is heavy but correct shrinkage of a noise-dominated rate.
- Its lag-1 move ratio overshoots to 2.3-3.5 in E1, E2, E2c and E3, and is 1.19 in E4.

**make_rim off.** The move ratio overshoots in every arm (1.45-2.02). The prior dominates too long.

**E2c's continuity gain is small.** On the pooled score it is +0.07 pp over E2 on fold 2 and +0.04 pp on fold 1.
- It is concentrated at game 1 (534 vs 475 deviance units): share3 off 9.3% vs 7.4%, tov def 5.6% vs 4.0%, make_jump off 1.2% vs 0.6%.
- It brings the game-1 slope to 0.997. It is the best arm for the day-0 launch condition.

### 2.3 Decision under the registered rule: NO ARM QUALIFIES

**Interval and transfer: pass for every arm.** Each arm clears zero on the pooled fold-2 interval, and the fold-1 gain has the same sign.

**G-A1 (level-SD ratio must not fall below E0): FAIL for every arm on every rate-side** (E0 1.13 vs 0.61-0.70 for the other arms).

- As registered, this guard compares against E0.
- E0's ratio above 1 is itself few-game noise: its season-mean estimate is MORE dispersed than the realised season rates.
- Against RAW realised rates, which also carry sampling noise, any estimator that stops chasing noise lowers the ratio.
- A post-hoc, noise-aware variant is below. It is labelled as post-hoc and was not used here.
  - It compares the estimate built on 20 or more games against a split-half true SD.
  - E0 reads 1.18, meaning it is over-dispersed against truth.
  - The arms read 0.83 (E4), 0.86-0.88 (E1, E2, E2c) and 0.92 (E3; last estimate 0.96).

**G-A2 (lag-1 movement ratio moves toward 1 relative to E0's 0.694): pass for E4 (0.94) and E3 (1.24). Fail for E1 (1.42), E2 (1.47) and E2c (1.44).** Those three over-shoot: after reliability weighting, the movement they do make is MORE predictive than their level, so they are too sluggish.

**What the PM would get with an amended G-A1.** This is the PM's call, not a selection here. If G-A1 were amended to a noise-aware comparison, E4 and E3 would qualify.
- Their pooled fold-2 gains are tied: 9.342% vs 9.335%, with each interval containing the other's point estimate.
- Under the tie rule the simpler arm is **E4**.
- E3 holds levels better (0.92 vs 0.84 noise-aware) and has the best fold-1 gain.
- Neither carries continuity. E2c's day-0 advantage would argue for adding its continuity-dependent carry to the E4 prior, which would be a new arm and a new registration.

### 2.4 Stage B resume plan (sequenced after Lane C; nothing retrained today)

1. **Register the G-A1 amendment** (PM) and select.
2. **Emit the winner's as-of centred rates** as a versioned sibling feature table (`data/processed/team_rate_features_<arm>_v1.parquet`), in the exact column names that possession_outcome (`off_/opp_def_{tov,ftr,3pa,rim}_c`), fg_make (`off_make_c__*`, `def_allow_c__*`) and rebound (`off_oreb_c`, `opp_def_dreb_c`) consume.
   - This needs one new `build_` script plus an estimator module.
   - Estimate: about 2 h of coding, and minutes to run.
3. **Retrain under S1 on the box**, fold 2 then fold 1:
   - possession_outcome round2 S1: 6 refits x 2 populations, about 1-2 h;
   - fg_make round4 B1: 6 refits x 3 classes, about 2 h;
   - rebound S1_weekly: 23 refits, about 3 h per cell per PROJECT_STATUS.
   - Each is graded on its own registered primary plus quintile responsiveness.
4. **Build engine inputs v3** as a sibling (about 30 min), then run **Stage C**: a paired 200-seed closed loop (about 35 min on AWS at 70 workers per stream), with the G9 slope, slope(close on X) and k_within via `scripts/diag_g9_g6_margin_v1.py --part close`.

**Total Stage B+C:** about 1 working day, of which about 7-8 h is box compute.
