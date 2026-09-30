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

---

## 3. Amendment (PM ruling 2026-09-30, registered about 11:45 EDT, COMMITTED BEFORE THE RUNS IT ADDS)

**POST-HOC STATUS.** This amendment was written AFTER the section 2 results were read. Every re-score of E0, E1, E2, E2c, E4 and E3 under the new guards is therefore POST-HOC. Only E4c, E3c and E9 are new arms whose results were not seen before registration.

### 3.1 G-A1 is withdrawn

The registered guard was "the level-SD ratio must not fall below E0". It rewards noise.

- **E0 is noisier than the truth.** E0's season-mean estimates are MORE spread than the realised season rates: 1.13, and 1.18 on the noise-aware read (section 2.3). So any estimator that removes noise fails the guard by construction.
- **A calibrated estimate must be less spread than the truth.** It is a posterior mean, so its dispersion across teams is below the true team rates'. The spread the point estimate gives up is estimation uncertainty. It must be carried as a variance, not smuggled back in as noise in the mean.
- **Reconciliation with the refused possession-outcome G2 arm (2026-09-18).** G2 compressed the spread of team estimates with nothing carrying the variance. The engine then lost between-game spread.
- **Responsibility.** The mis-specification was the PM's amendment 4, not the section 1 draft.

### 3.2 Replacement guards

**G-A1a (calibration).** The slope of realised on predicted (WLS by denominator, over team-games) must be within 0.90-1.10:

- pooled, and per games band (game 1, games 2-3, 4-6, 7+), per rate-side;
- a cell whose slope SE exceeds 0.10 is labelled UNDERPOWERED and does not count as a failure.

**G-A1b (variance accounting).** Every arm emits an estimation variance v for each team-game-rate. Write sigma^2 = L(1-L) for binomial rates and L for Poisson rates.

| arm | v |
|---|---|
| E3, E3c | the filter's posterior variance before tipoff |
| E1, E2, E2c | sigma^2 / (D + k), the posterior variance implied by the effective sample size D and the fitted prior strength k |
| E4, E4c | sigma^2 / (D_w + k), with D_w the exponentially weighted denominator |
| E0 | sigma^2 / D; undefined at D = 0, so those rows are excluded and the exclusion is counted |

The arm must pass two checks:

- **(i) Between-team variance.** [var(c across team-games) + mean(v)] / (split-half true between-team variance of centred season rates) is within 0.90-1.10. This is checked pooled over the season and per games band.
- **(ii) Standardised residuals.** The SD of z = (y - p) / sqrt(v + s^2) is within 0.90-1.10, pooled and per games band. Here s^2 is the next-game sampling variance at p (binomial p(1-p)/den or Poisson p/den).

An arm whose v is not calibrated cannot later feed the engine an honest uncertainty draw.

**G-A2 (lag-1 move ratio toward 1 relative to E0) stays as registered.**

### 3.3 New arms

| arm | definition | free parameters |
|---|---|---|
| **E4c** | E4 with E2c's carry, rho = rho0 + rho1 (cont - mean cont) + rho2 coach_change | lambda, k, rho0, rho1, rho2 |
| **E3c** | E3 with the same carry | q, P0, rho0, rho1, rho2 |

**Reason.** E2c is the best arm at game 1, with slope 0.997 vs about 0.92. Game 1 is the launch condition. But E2c is sluggish later (move ratio 1.44), and E4 and E3 are the opposite.

Fitting is as in section 1: training seasons only, by next-game likelihood.

### 3.4 Decision rule (amended)

- **Primary metric and noise floor:** as registered in section 1.5.
- **To qualify, an arm must:**
  - clear zero on the pooled fold-2 interval;
  - show the same sign on fold 1;
  - pass G-A1a (pooled, over the rate-sides that are not underpowered);
  - pass G-A1b (i) and (ii), pooled;
  - pass G-A2.
- **Ties** go to the simpler arm, in the order E1, E2, E2c, E4, E4c, E3, E3c.
- **Calibrated variance is a requirement, not a tiebreak.** If the E4 and E3 families tie on gain but only one passes G-A1b, the passing one wins.

### 3.5 E9

E9 is opponent adjustment on the amended winner, as registered: a single arm, run if time remains. Each past game's observation is corrected by the opponent's as-of estimate on the opposite side, taken from the winner's unadjusted estimates. The winner's parameters are then refitted on training seasons.

### 3.6 Stage C pre-registration (spec only; nothing runs today)

The paired 200-seed closed loop on fold 2 compares two arms, both on the Stage B retrained sub-models:

- **C-point:** the winner's point estimates alone as the team-rate features;
- **C-draw:** the winner plus a per-game draw of each team rate from N(c, v), its own estimation variance. It is drawn once per simulated game, per team-rate and side, on the RNG family `team_rate` seeded on (seed, game_id, family).

C-draw follows the CLAUDE.md dispersion rule: the engine restores spread from the estimator's own variance function.

- **Primary:** G9 calibration slope.
- **Must not regress beyond the paired A/B floor:**
  - G5 total SD ratio;
  - G5 home/away score correlation.
- **Also reported:**
  - slope(close on X) and k_within (`scripts/diag_g9_g6_margin_v1.py --part close`);
  - G5 margin SD ratio;
  - G1.
- **Reference:** the served stack (E0 features) is carried as the reference row.

### 3.7 Execution

- **Fitter:** `scripts/exp_team_rate_estimator_v2.py`.
- **Grader:** `scripts/grade_team_rate_estimator_v2.py`, the section 1 grader extended with G-A1a and G-A1b. Its section 1 metrics are unchanged.
- **Outputs:** `results/team_rate_estimator/*_v2.*`. The v1 scripts and outputs are not overwritten.
- **Results:** section 4, written once when final.

---

## 4. Results of the section 3 amendment (run 2026-09-30, about 11:45-11:50 EDT; 1 core)

**POST-HOC LABEL.** Every guard number for E0, E1, E2, E2c, E4 and E3 below is a POST-HOC re-score: those arms' section 2 results were seen before section 3 was written. E4c, E3c and E9 are new arms, registered in section 3 before they ran.

**Scripts and outputs**

- Fitter: `scripts/exp_team_rate_estimator_v2.py`. It reproduces every v1 estimate exactly (maximum absolute difference 0.0) and adds v, E4c, E3c and `--e9`.
- Grader: `scripts/grade_team_rate_estimator_v2.py`. It is the v1 grader plus `section3()`.
- Outputs: `results/team_rate_estimator/*_v2.*`. The v1 outputs are untouched.

**How the grader decides a guard.** Each check is decided on the fold-2 median over the 16 rate-sides, with the G-A1a median taken over rate-sides that are not underpowered. The count of rate-sides inside the band is reported alongside.

### 4.1 All arms, fold 2 (fold 1 in brackets)

Gain is the pooled next-game deviance gain vs E0 over the 16 rate-sides. Intervals are team-block bootstrap with 1,000 draws.

| arm | gain (%) | 95% interval | fold-1 gain (%) | G-A1a slope, median (rate-sides in band) | G-A1b(i) var ratio, median (in band) | G-A1b(ii) z SD, median | G-A2 move ratio | game-1 slope | qualifies |
|---|---:|---|---:|---|---|---:|---:|---:|---|
| E0 | 0 | - | 0 | 0.41 (0/16) FAIL | 3.32 (0/16) FAIL | 1.096 | 0.694 (ref) | undefined | no |
| E1 | 28,731 (8.37) | [27,035, 30,778] | 8.62 | 0.96 (11/13) | 0.910 (9/16) | 1.091 | 1.424 FAIL | undefined | no (G-A2) |
| E2 | 31,281 (9.11) | [29,524, 33,487] | 9.50 | 0.98 (14/15) | 0.917 (7/16) | 1.091 | 1.465 FAIL | 0.916 | no (G-A2) |
| E2c | 31,504 (9.18) | [29,793, 33,687] | 9.54 | 0.99 (15/15) | 0.903 (7/16) | 1.091 | 1.442 FAIL | 0.997 | no (G-A2) |
| E4 | 32,068 (9.34) | [30,309, 34,255] | 9.79 | 0.98 (15/15) | **1.134 FAIL** (5/16) | 1.076 | 0.940 | 0.924 | no (G-A1b) |
| E4c | 32,281 (9.40) | [30,534, 34,454] | 9.82 | 0.99 (15/15) | **1.106 FAIL** (5/16) | 1.076 | 0.926 | 1.005 | no (G-A1b) |
| **E3** | 32,045 (9.33) | [30,219, 34,232] | 9.80 | 0.94 (13/15) | 1.061 (7/16) | 1.085 | 1.240 | 0.915 | **yes** |
| E3c | 32,274 (9.40) | [30,448, 34,406] | 9.83 | 0.95 (15/15) | 1.045 (9/16) | 1.085 | 1.232 | 0.995 | yes |

**Fold 1 agrees on every verdict** except G-A1b(i) for the E4 family, which is at 1.083 and 1.079 on fold 1, inside the band. The rule reads fold 2.

**Decision under the section 3.4 rule: E3.**

- E3 and E3c both qualify.
- E3c's pooled gain is higher, but its interval [30,448, 34,406] contains E3's point gain of 32,045. So this is a registered tie, and the tie goes to the simpler E3.
- The E4 family passes every other guard and has gains tied with E3's. Only the E3 family passes G-A1b(i), so the "calibrated variance is a requirement" clause decides between the families as well.

**POST-HOC INFORMATION, not in the rule.** On the paired difference, E3c beats E3 on both folds:

| fold | E3c gain over E3 | 95% interval |
|---|---:|---|
| 2 | +229 | [112, 344] |
| 1 | +118 | [29, 211] |

The E4c-over-E4 difference is the same size. Almost all of it is game 1, where E3c's slope is 0.995 vs E3's 0.915. The registered tie test (interval of each arm's gain vs E0) cannot see a paired difference this small. If the PM wants E3c, that needs a registered amendment to the tie test; it is not selected here.

### 4.2 Emitted-variance calibration of the winner (E3), fold 2, by games band

| band | slope (median over rate-sides) | underpowered rate-sides | G-A1b(i) var ratio | G-A1b(ii) z SD |
|---|---:|---:|---:|---:|
| game 1 | 0.915 | 16/16 | **0.739** | 1.214 |
| games 2-3 | 1.074 | 15/16 | **0.779** | 1.148 |
| games 4-6 | 0.995 | 12/16 | 0.825 | 1.115 |
| games 7+ | 0.933 | 1/16 | 1.092 | 1.069 |
| pooled | 0.942 | 1/16 | 1.061 | 1.085 |

- **E3's v is calibrated pooled but under-covers early.** At game 1 it carries about 74% of the needed between-team variance, and 78% at games 2-3.
- **The cause is the fitted P0.** It is too small relative to the true spread left unexplained by the prior-season carry.
- **Stage C's draw arm must not use it uncorrected before games 7+.** This is a Stage B item: a registered P0-by-carry fit, not a tweak.
- **The z SD above 1 is mostly the sampling model, not v.** Without any v, the z SD is 1.11 (median). Game-level shot-mix and FT-rate counts are over-dispersed relative to binomial or Poisson per FGA:
  - z SD for ftr is about 1.8;
  - share3 1.26-1.38;
  - share_rim 1.41-1.46.
- **E3's per-rate G-A1b(i) exceptions on fold 2:**
  - under-covered: ftr off 0.77, ftr def 0.83;
  - over-covered: tov def 1.28, make3 def 1.22, make_rim off 1.17, oreb def 1.15, tov off 1.14, make_rim def 1.12.

### 4.3 E9 (opponent adjustment on E3, single arm)

**Registered primary: FAILS.** The single-side next-game deviance change vs E3 is:

| fold | change | 95% interval |
|---|---:|---|
| 2 | -748 (-0.24%) | [-1,038, -473] |
| 1 | -813 (-0.26%) | [-1,047, -601] |

On guards, E9 passes G-A1a and G-A1b(i) (1.014) but has a worse G-A2 than E3 (1.355).

**POST-HOC SECONDARY: scored the way the sub-models consume the features.** With p = L + c(team) + c(opponent, opposite side), both from the same arm, E9 GAINS:

| fold | change | 95% interval |
|---|---:|---|
| 2 | +1,449 (+1.01%) | [1,226, 1,672] |
| 1 | +1,538 (+1.08%) | [1,288, 1,779] |

**Why the registered primary is the wrong test for this arm.** A single-side prediction that omits the next opponent penalises an estimator for removing opponent strength from the team's own rate. E9 is NOT selected. Stage B can carry it as an arm under a two-sided primary if the PM registers one.

### 4.4 Weak-signal rate-sides: prior strength, and whether "league level plus variance" is honest

**3P% allowed (make3 def).** The split-half true between-team SD is 1.18 pp.

- **Fitted prior strength:**
  - E1: k = 1,780 3PA, about 78 games of about 22.7 3PA, so the data weight after 30 games is about 28%;
  - E2: k = 2,446, rho 0.25;
  - E3: prior SD 0.82 pp with process SD 0.23 pp per game;
  - E4: half-life about 4.4 games, k = 940.
- **What that means.** Every arm has learned that this side is almost all noise. Estimate spread is 0.38-0.56 of the true SD, and the slope is UNDERPOWERED in every arm (SE 0.13-0.18).
- **Verdict.** Yes: league level plus variance is the honest estimate here. E1, which is nearly exactly that, carries the variance best (G-A1b(i) 0.90). E3's v over-states it (1.22) and E4's badly so (1.80). The winner's v on this side should be treated as conservative.

**Rim FG% offence (make_rim off).** The true SD is 3.28 pp, which is real signal.

- **Fitted prior strength:** E1 k = 244 rim FGA (about 11 games); E3 prior SD 2.3 pp; rho 0.56.
- **Every arm is short of calibration here:**
  - slopes 0.87-0.92 (not underpowered: SE about 0.04);
  - G-A1b(i) 1.09-1.24;
  - lag-1 move ratios 1.45-2.02.
- **What that means.** The level is slightly over-spread, and the within-season movement is more predictive than the level. That points to genuine within-season drift of rim finishing plus an over-weighted prior-season carry (rho 0.56 on a rate whose team signal changes year to year).
- **Verdict.** No: league level plus variance would discard 3.3 pp of real team signal. This side needs a better model of drift and carry (a rate-specific q and rho in E3 already exists; the carry input is the suspect), not a flatter estimate.

### 4.5 Stage B inputs (not built today)

The winner, E3, writes one versioned sibling table: `data/processed/team_rate_features_E3_v1.parquet`.

- **Rows:** one per (season, game_id, team_id) for 2022-2025, about 45,000 team-games (2026 SEALED; the live season is built as-of on the day).
- **Columns:** for each of the 16 rate-sides, the centred estimate c, its variance v and the league level L. That is 48 value columns plus keys: about 2.2M values, about 10-15 MB of parquet.
- **Names:** mapped to the consumers' existing names:
  - possession_outcome: `off_/opp_def_{tov,ftr,3pa,rim}_c`;
  - fg_make: `off_make_c__{rim,jump2,three}`, `def_allow_c__*`;
  - rebound: `off_oreb_c`, `opp_def_dreb_c`.
- **Resume plan:** as in section 2.4. Plus the early-season P0 fix (4.2) and the PM's choice on E3c (4.1) and E9 (4.3), all to be registered before Stage B.

---

## 5. Round 2 pre-registration (PM ruling 2026-09-30; registered about 11:55 EDT, COMMITTED BEFORE ANY ROUND-2 FIT)

The E3 (state-space) family is SELECTED by the PM (ledger entry is the PM's). Round 2 holds the family fixed and asks three separate questions, so that Stage B retrains once, on a final table.

### 5.1 Rules common to Q1-Q3

- Hyper-parameters are fitted on training seasons only: fold 1 on 2023, fold 2 on 2023+2024.
- Fold 2 selects. Fold 1 must agree in sign.
- **Test for every comparison:** a PAIRED team-block bootstrap of the difference between the two arms (teams resampled with replacement, 1,000 draws, 95% interval). This replaces the round-1 tie test.
- A comparison between arms already seen in round 1 is labelled POST-HOC.
- **Gaussian predictive likelihood (used by V1, V2 and O1):** log N(y; p, v + phi s^2), where s^2 is the binomial (or Poisson) next-game sampling variance at p and phi = 1 unless the arm fits it.
- **Scripts:** `scripts/exp_team_rate_estimator_v3.py` (fitting, emission) and `scripts/grade_team_rate_estimator_v3.py` (the one grader).
- **Estimator core:** a NEW module, `src/cbb_sim/live/team_rate_estimator.py`. It is shared by the historical emission and the live-slate function; no existing file under `src/cbb_sim/live/` is edited.

### 5.2 Q1: variance

In every arm below, q and rho are E3's own fitted values; only the stated parameters are fitted.

| arm | definition |
|---|---|
| V0 | E3 as fitted in round 1 (q, P0, rho) |
| V1 | P0 refitted as its own parameter per rate-side, by maximising the Gaussian predictive likelihood of each team's FIRST 6 games (j = 0-5) on training seasons |
| V2 | as V1, with log P0 = a0 + a1 (cont - mean cont) + a2 coach_change (a1 is expected negative) |
| O1 | on top of the better of V1/V2 by the Q1 primary: a per-rate-side overdispersion factor phi, applied to the filter's observation variance and to s^2. phi and that arm's P0 parameters are fitted jointly by the Gaussian predictive likelihood over all training rows |

**Primary.** Per rate-side and pooled (median over the 16 rate-sides), at games bands 1, 2-3, 4-6 and 7+, and pooled over bands:

- the variance ratio [var(c) + mean(v)] / (split-half true between-team variance);
- the z SD, with z = (y - p) / sqrt(v + phi s^2).

Both must be inside 0.90-1.10.

**Deviance guard.** An arm may not lose deviance to V0: it fails if its paired interval vs V0 lies entirely below zero.

**Decision.**

- An arm PASSES the primary if both pooled medians are inside the band at all four games bands.
- Among passing arms, the first in the order V0, V1, V2 wins. O1 is eligible only if no V arm passes because of the z SD.
- If no arm passes, the winner is the arm with the most (rate-side x band x check) cells inside the band, ties to the simpler arm.

### 5.3 Q2: prior mean at day 0

Both arms use the Q1 winner's variance specification.

| arm | carry |
|---|---|
| P0 | E3's prior-season carry, rho c_prev |
| P1 | E3c's carry, rho = r0 + r1 (cont - mean) + r2 coach_change (POST-HOC pair) |

**Primary.** The paired deviance difference P1 - P0 on the game-1 cell and on the games 2-3 cell. The game-1 team slope is reported.

**Decision.** P1 is selected only if both hold:

- its fold-2 paired interval on game 1 excludes zero on the favourable side;
- fold 1 agrees, with its own interval also excluding zero.

**Diagnostic: rim FG% offence carry.**

- Report the fitted carry weight rho per rate-side.
- For make_rim off, also report a per-rate carry variant with rho fitted by the first-6-games predictive likelihood, and whether it repairs the G-A1a slope (target 0.90-1.10 pooled, and at 7+).
- This variant is diagnostic only and is not selectable in round 2.

### 5.4 Q3: opponent adjustment (optional arm; POST-HOC)

**Primary (two-sided, as the sub-models consume features).** The matchup-level next-game deviance on offence-side rows, with p = L + c_team(side) + c_opponent(opposite side), both estimates from the same arm. The comparison is E3-final vs E3-final+E9, paired bootstrap, on both folds.

**Decision.** If both folds' intervals exclude zero in favour of E9:

- E9 is carried into Stage B as ONE extra arm on possession_outcome only, where that sub-model's own unseen primary decides;
- it is NOT adopted here.

### 5.5 Emission (after Q1-Q3)

The round-2 winner (the Q1 winner combined with the Q2 winner) writes `data/processed/team_rate_features_<arm>_v1.parquet`. If Q3 clears, a second table with the `+opp` suffix is also written for the possession_outcome arm.

**Contents**

- One row per (fold, season, game_id, team_id).
- For each of the 16 rate-sides: the centred estimate `c`, variance `v` and league level `L`.
- Fold F1 rows: seasons 2022-2024, parameters fitted on 2023.
- Fold F2 rows: seasons 2022-2025, parameters fitted on 2023+2024.
- 2026 is SEALED and not computed.

**Strictly as-of.** An assertion in code checks, for every row, that every performance input has a date strictly before the row's game date. Roster continuity uses roster LISTINGS (athlete membership, not performance) from the team's first 3 box scores; that is documented as the one non-performance input and is flagged if it is used.

**Live function:** `estimate_asof(...)` in the new module, for a future slate. It is not run for 2026-27 today.

**Trainer consumption:** a name-for-name mapping of which existing columns the table replaces is written in section 6.

---

## 6. Round 2 results (run 2026-09-30, about 11:55-12:10 EDT; 1 core)

**Code**

- Fitter: `scripts/exp_team_rate_estimator_v3.py`. V0 reproduces round-1 E3 exactly (maximum absolute difference 0.0 in c and v).
- Grader: `scripts/grade_team_rate_estimator_v3.py`.
- Estimator core: `src/cbb_sim/live/team_rate_estimator.py`, a new module.

**Outputs:** `results/team_rate_estimator/*_v3.*`.

**Test.** Every comparison below is a PAIRED team-block bootstrap (1,000 draws, 95% interval). A positive difference favours the second arm.

### 6.1 Q1: variance. Winner under the rule: V0

Medians over the 16 rate-sides, fold 2. The target band is 0.90-1.10.

| arm | variance ratio g1 / g2-3 / g4-6 / 7+ | z SD g1 / g2-3 / g4-6 / 7+ | cells in band (of 128) | paired deviance vs V0, F2 | F1 |
|---|---|---|---:|---|---|
| V0 | 0.739 / 0.779 / 0.825 / 1.092 | 1.214 / 1.148 / 1.115 / 1.069 | 45 | - | - |
| V1 | 1.702 / 1.757 / 1.799 / 1.535 | 1.163 / 1.107 / 1.093 / 1.069 | 32 | -4,258 [-4,707, -3,835] | -4,519 [-4,901, -4,103] |
| V2 | 1.753 / 1.819 / 1.845 / 1.560 | 1.159 / 1.103 / 1.092 / 1.069 | 32 | -4,445 [-4,897, -4,016] | -4,582 [-4,963, -4,168] |
| O1a (V1 base) | **0.956 / 0.982 / 1.021** / 1.163 | **1.065 / 1.043 / 1.027 / 0.993** | **90** | -101 [-186, -20] | -104 [-193, -13] |
| O1b (V2 base) | 0.914 / 0.919 / 0.959 / 1.145 | 1.051 / 1.038 / 1.026 / 0.993 | 84 | -1,909 [-2,426, -1,439] | -3,370 [-3,963, -2,871] |

**How the rule reads.**

- No arm passes the primary at all four bands.
- V1, V2, O1a and O1b all fail the deviance guard: each paired interval vs V0 lies entirely below zero.
- The most-cells fallback therefore has only V0 left. **V0 wins.** It is what the emitted table carries.

**Why V1 and V2 over-shoot.** They fitted the starting variance by a Gaussian likelihood with BINOMIAL sampling variance. Game-level counts are over-dispersed, so the fit absorbed that excess into P0: the variance ratio is about 1.7 and the calibration slope about 0.72. This is the case for O1.

**O1a for the PM.**

- It is the only arm that calibrates the variance at games 1-6 and the z SD at every band.
- Its fitted overdispersion factors are:

| rate-side | phi |
|---|---:|
| ftr | about 3.2-3.4 |
| share3 | 1.6-1.9 |
| share_rim | 1.1-2.2 |
| oreb | 1.1-1.3 |
| make classes | 1.0-1.2 |

- Its deviance cost is 0.03% (-101 [-186, -20]).
- It still over-covers at 7+ (1.16), because q was held at E3's deviance-fitted value.
- Adopting O1a for Stage C's draw arm needs an amendment to the deviance guard (for example a tolerance of 0.1%), or a registered joint refit of q, P0 and phi. It is NOT selected here.

**Final arm's guard line (V0 + P0 = E3).**

| fold | band | calibration slope | underpowered rate-sides | variance ratio | z SD |
|---|---|---:|---:|---:|---:|
| F2 | g1 | 0.915 | 16/16 | 0.739 | 1.214 |
| F2 | g2-3 | 1.074 | 15/16 | 0.779 | 1.148 |
| F2 | g4-6 | 0.995 | 12/16 | 0.825 | 1.115 |
| F2 | 7+ | 0.933 | 1/16 | 1.092 | 1.069 |
| F2 | pooled | 0.942 | 1/16 | 1.061 | 1.085 |
| F1 | g1 | 0.992 | 16/16 | 0.710 | 1.179 |
| F1 | g2-3 | 1.145 | 15/16 | 0.772 | 1.144 |
| F1 | g4-6 | 1.009 | 13/16 | 0.847 | 1.116 |
| F1 | 7+ | 0.966 | 2/16 | 1.047 | 1.077 |
| F1 | pooled | 0.966 | 1/16 | 0.998 | 1.087 |

**Stage C consequence.** The draw arm, run with V0's v, would restore only about 74-83% of the needed early-season spread in games 1-6.

### 6.2 Q2: prior mean at day 0 (POST-HOC pair). Winner under the rule: P0

| cell | fold 2: P1 - P0 paired | fold 1 |
|---|---|---|
| game 1 | +55.9 [+10.7, +100.1] (0.50%) | -0.6 [-30.9, +32.4] |
| games 2-3 | +55.7 [+18.3, +98.5] | +33.2 [+1.1, +67.3] |
| all games | +236 [+133, +349] | +125 [+44, +219] |

- **Game-1 team slope (fold 2, median):** P0 0.915, P1 0.995.
- **Verdict.** P1 is better on fold 2 at game 1, but fold 1's game-1 interval spans zero. The registered condition (both folds excluding zero) is not met, so **P0 stands**.
- P1 does win on every games-2+ cell in both folds. The continuity carry helps, just not detectably at game 1 on fold 1.

**Fitted carry weight rho (P0, fold 2).**

| rate | off | def |
|---|---:|---:|
| tov | 0.50 | 0.58 |
| ftr | 0.45 | 0.65 |
| share3 | 0.58 | 0.57 |
| share_rim | 0.50 | 0.61 |
| make_rim | 0.56 | 0.38 |
| make_jump | 0.32 | 0.39 |
| make3 | 0.21 | 0.25 |
| oreb | 0.61 | 0.58 |

**Rim FG% offence: the per-rate carry does not repair the slope.**

- Carry fits: P0 rho 0.56 (fold 2) and 0.53 (fold 1). P1 is 0.58 + 0.07 x continuity - 0.10 x coach change. Pd, with rho fitted by the first-6-games likelihood, is 0.65.
- Fold-2 pooled slope: P0 0.896, P1 0.900, Pd 0.846 (worse). At 7+ the slope is 0.84, not underpowered (SE 0.04).
- Fold-1 pooled slope: 1.02, calibrated.
- **Reading.** The fold-2 miss is a 2024-25 within-season shift in rim finishing that no carry weight addresses. It is not a carry mis-weight. It stays an open exception for the season-drift work (Lane C).

### 6.3 Q3: opponent adjustment, two-sided (POST-HOC). CLEARS both folds

| scoring | fold 2: final+opp - final | fold 1 |
|---|---|---|
| **two-sided (registered)** | **+1,465 [+1,247, +1,701] (+1.02%)** | **+1,552 [+1,315, +1,812] (+1.09%)** |
| one-sided (round-1 primary, for reference) | -788 [-1,073, -518] | -870 [-1,097, -660] |

E9 is therefore carried into Stage B as ONE extra arm, on possession_outcome only, using table `E3opp`. Its parameters are identical to the final arm's. The only difference is that each past observation is centred on the opponent's as-of opposite-side estimate. It is not adopted here.

### 6.4 Emitted Stage B tables

**Files**

- `data/processed/team_rate_features_E3_v1.parquet`: the round-2 winner, E3 = V0 + P0.
- `data/processed/team_rate_features_E3opp_v1.parquet`: the possession_outcome-only E9 arm.
- Each is 78,004 rows x 53 columns, about 26 MB.
- **Not committed.** Each is over the 20 MB guidance for processed artifacts, so tracking or HF sync is the PM's call. They regenerate in about 1 minute with:

```
.venv/Scripts/python.exe scripts/exp_team_rate_estimator_v3.py --part emit --final q2:P0 --label E3
.venv/Scripts/python.exe scripts/exp_team_rate_estimator_v3.py --part emit --final q2:P0 --opp --label E3opp
```

**Rows.** One per (fold, season, game_id, team_id):

| fold | seasons | rows per season |
|---|---|---|
| F1 | 2022-2024 | 10,792 / 11,246 / 11,264 |
| F2 | 2022-2025 | 10,792 / 11,246 / 11,264 / 11,400 |

- F1 rows use parameters fitted on 2023; F2 rows use parameters fitted on 2023+2024.
- A Stage B trainer for fold k reads `fold == k` only.
- 2026 is not computed.

**Schema**

- Keys: `fold`, `season`, `game_id`, `team_id`, `game_date`.
- Then, for rate in {tov, ftr, share3, share_rim, make_rim, make_jump, make3, oreb} and side in {off, def}, three columns:
  - `<rate>_<side>_c`: the centred estimate entering the game;
  - `<rate>_<side>_v`: its estimation variance;
  - `<rate>_<side>_L`: the as-of league level. It is identical for off and def.
- Everything is on a 0-1 rate scale.

**Strictly as-of.** The emitter asserts that within every team-season the game index order is strictly increasing in tipoff time. One 2023 team played twice on the calendar date 2022-11-25, so a date-only check was insufficient and tipoff time is used. Given that, the estimate at index j uses only games 0..j-1. The league level uses only dates strictly before the game.

- The final arm (P0) uses no roster-continuity input, so the only non-performance input in section 5.5 is NOT present in this table.
- `c_prev` is the prior season's final.

**Live function.** `cbb_sim.live.team_rate_estimator.estimate_asof(team_games, as_of, params, league_level, carry)`.

- It asserts that every game used has game_date < as_of.
- Given per-game `L_<rate>` columns, it reproduces the emitted F2 table exactly: maximum absolute difference 0.0 on four 2024-25 dates (Nov 4, Dec 10, Jan 15, Mar 1).
- It was not run for 2026-27.
- Parameters are in `results/team_rate_estimator/params_emit_E3_v3.json`, keyed `F2|<rate>|<side>`.

### 6.5 How Stage B trainers consume the table (name for name)

**Joins**

- For a row where team T is on offence against opponent O in game g, take T's `_off_` columns and O's `_def_` columns from the same game_id.
- possession_outcome's served features are on a x100 scale (`RATE_SCALE`), so multiply its columns by 100.
- fg_make and rebound are on the 0-1 scale.

| served column (consumer) | replaced by | notes |
|---|---|---|
| `off_tov_c` (PO) | 100 x `tov_off_c` (T) | served tov is per pbp possession; the table uses the box possession formula |
| `opp_def_tov_c` (PO) | 100 x `tov_def_c` (O) | |
| `off_ftr_c` (PO) | 100 x `ftr_off_c` (T) | FTA / FGA, the same definition |
| `opp_def_ftr_c` (PO) | 100 x `ftr_def_c` (O) | |
| `off_rim_c` (PO) | 100 x `share_rim_off_c` (T) | rim FGA / FGA; the table uses the box 2PT count split by event-layer shares |
| `opp_def_rim_c` (PO) | 100 x `share_rim_def_c` (O) | |
| `off_3pa_c` (PO) | **NO DIRECT REPLACEMENT** | see below |
| `opp_def_3pa_c` (PO) | **NO DIRECT REPLACEMENT** | as above |
| `off_make_c__rim` / `__jump2` / `__three` (fg_make) | `make_rim_off_c` / `make_jump_off_c` / `make3_off_c` (T) | |
| `def_allow_c__rim` / `__jump2` / `__three` (fg_make) | `make_rim_def_c` / `make_jump_def_c` / `make3_def_c` (O) | |
| `off_oreb_c` (rebound) | `oreb_off_c` (T) | the table uses box OREB / (OREB + opponent DREB); rebound's served version uses live rebound opportunities |
| `opp_def_dreb_c` (rebound) | **minus** `oreb_def_c` (O) | the defence's DREB% minus the league's = -(OREB% allowed minus the league's) |

**The 3PA gap.** Served `3pa` is 3PA per POSSESSION. The table's `share3` is 3PA per FGA. Stage B must choose one of two options, and that choice must be registered:

- (a) add a `three_per_poss` rate-side to the estimator (one more rate, same code, about 1 minute to re-emit);
- (b) use `share3` as a changed feature definition.

**Untouched columns.** The rating features (`off_rating_*`, `def_rating_*`, which are ridge and site-adjusted), site, date, tempo and player/slot features are not touched. The `_v` and `_L` columns are new, for Stage C's draw arm, and are not model features.

**Other definitional deltas to note in each trainer's Stage B registration:**

- possessions: box formula vs pbp;
- rebound opportunities: box vs live;
- rim split: box plus event-layer shares vs pure event layer.

---

## 7. Stage B and Stage C pre-registration, plus the 3PA-per-possession rate and the S3 variance pre-check (written 2026-09-30 about 12:15 EDT; COMMITTED BEFORE ANY STAGE B OR C RUN)

The PM rulings are recorded in the ledger (`d88642d`):

- Q1: V0 is the POINT estimate.
- Q2: P0 stands.
- Q3: E3opp goes to Stage B as the possession_outcome-only extra arm.
- O1a's variance becomes Stage C's registered draw arm S3.

**Where this runs.** Nothing below runs on this machine today. Stages B and C run on the box tonight.

**Scope.** The whole Stage B/C registration lives in this file, so no other lane's experiments.md is touched.

### 7.0 Done today, before this registration (inputs to B and C, not selections)

**Adapter** (`2ccf964`): `src/cbb_sim/team_rate_adapter.py`, `apply(frame, table_path, submodel, fold, missing)`.

- It uses the name-for-name mapping of section 6.5, and `off_3pa_c`/`opp_def_3pa_c` now map to `pa3`.
- It recomputes possession_outcome's `x_off_<r>_c__opp_def_<r>_c` products from the replaced factors.
- It asserts row count, order, no new NaN and key coverage. Missing keys raise unless `missing="keep_served"` is passed explicitly, and the count kept is stored in `frame.attrs`.
- Tests: `tests/test_team_rate_adapter.py` substitutes REAL design samples for fg_make, rebound and possession_outcome round 2, and covers the refusal and missing-key paths. 5 passed.

**3PA per possession (`pa3`), like for like with possession_outcome's served `3pa`.**

- Procedure: the identical E3 procedure (`scripts/exp_team_rate_estimator_v4.py --part fit`).
- Fitted rho: 0.60 (off) and 0.58 (def).
- Guard line (medians over off/def):

| band | F2 slope | F2 variance ratio | F2 z SD | F1 slope | F1 variance ratio | F1 z SD |
|---|---:|---:|---:|---:|---:|---:|
| game 1 | 0.652 (underpowered) | 0.608 | 1.462 | 0.990 | 0.575 | 1.439 |
| games 2-3 | 0.832 | 0.648 | 1.504 | 1.097 | 0.628 | 1.389 |
| games 4-6 | 1.094 | 0.705 | 1.355 | 0.983 | 0.708 | 1.404 |
| 7+ | 0.980 | 1.001 | 1.365 | 0.979 | 1.020 | 1.365 |
| pooled | 0.972 | 0.937 | 1.377 | 0.984 | 0.950 | 1.372 |

For comparison, the 16 other rate-sides' pooled medians (V0) are slope 0.942 / 0.966, variance ratio 1.061 / 0.998 and z SD 1.085 / 1.087 (F2 / F1).

- **Point estimate:** calibrated pooled.
- **Variance:** under-covers early, like the other rates.
- **z SD:** 1.37, the same binomial-overdispersion signature as share3.

**Tables v2** (the v1 files stay):

- `data/processed/team_rate_features_E3_v2.parquet` and `..._E3opp_v2.parquet`.
- 78,004 rows x 59 columns (17 rate-sides x {c, v, L}).
- The 48 shared columns are identical to v1 (maximum absolute difference 0.0, asserted).
- As-of is asserted by tipoff order.
- About 29 MB each, NOT committed. Regenerate with `exp_team_rate_estimator_v4.py --part emit [--opp]`.

**S3's variance source:** `data/processed/team_rate_variance_O1a_v1.parquet`.

- 78,004 rows; `<rate>_<side>_v_o1a` and `_phi_o1a` for all 17 rate-sides (pa3 phi 1.11 off, 2.03 def).
- The Kalman variance path depends only on exposures, L, q, P0 and phi, never on outcomes, so v is as-of by construction.
- Regenerate with `--part o1a`.

**S3 pre-check** (`scripts/grade_team_rate_estimator_v4.py`): the O1a variance around the V0 MEANS. Medians over the 16 round-2 rate-sides; figures are variance ratio / z SD.

| band | S2: V0 mean + V0 v, F2 | F1 | S3: V0 mean + O1a (v, phi), F2 | F1 |
|---|---|---|---|---|
| game 1 | 0.739 / 1.214 | 0.710 / 1.179 | **0.956 / 1.065** | **0.907 / 1.052** |
| games 2-3 | 0.779 / 1.148 | 0.772 / 1.144 | **0.961 / 1.044** | **0.937 / 1.028** |
| games 4-6 | 0.825 / 1.115 | 0.847 / 1.116 | **0.983 / 1.027** | **0.964 / 1.021** |
| 7+ | 1.092 / 1.069 | 1.047 / 1.077 | 1.142 / 0.993 | 1.114 / 1.000 |
| pooled | 1.061 / 1.085 | 0.998 / 1.087 | 1.099 / 1.001 | 1.079 / 1.004 |

- O1a's variance IS calibrated around the V0 point estimates at games 1-6 on both folds. The z SD is in band at every band, with 15/16 rate-sides in band.
- It over-covers at 7+ by 11-14%. S3 therefore slightly over-restores spread late in the season. That is recorded here as a known property of the arm, not tuned.

### 7.1 Stage B (per sub-model, box, tonight)

**Arms, for each sub-model** (possession_outcome `first` and `cont`, fg_make rim / jump2 / three, rebound):

- **R:** the served model on the served features, graded today by the same grader.
- **R2:** R retrained spec-identically under another seed. This is the noise floor.
- **T:** the same spec, scheme (S1) and seed as R, on E3 **v2** features through `team_rate_adapter.apply(..., fold=<fold>)`.
- **Topp (possession_outcome only):** as T, with `team_rate_features_E3opp_v2.parquet`.

Everything else in the design (ratings, site, state, player/slot blocks, derived columns not in the mapping) is the served value.

- Any arm the trainer refuses on an overlaid table (for example rebound round-3's derived-column arms) is not part of Stage B.
- Only the served spec is retrained.

**Primary and floor: each sub-model's OWN registered primary, quoted.**

- **possession_outcome** (served event `round2_s1`: `first` = lgbm, `cont` = cascade, S1_monthly).
  - `docs/models/possession_outcome/experiments.md` section 1: "Primary metric | multiclass log loss on F2, `first` population". `cont` is graded on the same metric separately.
  - Floor: the section 7.2 second-seed floors, "Applied floor **0.000804**" (`first`) and "**0.001982**" (`cont`). Tonight's R2 replaces the floor if larger.
- **fg_make** (served `round4_B1`).
  - `docs/models/fg_make/experiments.md` section 20: "**Primary:** attempt-level log loss", per class.
  - Floor: section 20.3, "second-seed refit of B1's whole S1 schedule: **4.386e-5 / 1.1216e-4 / 1.813e-5** (rim / jumper / three)". Tonight's R2 replaces it if larger.
- **rebound** (served `S1_weekly`).
  - `docs/models/rebound/experiments.md` section 1: "Primary metric | three-class log loss on F2".
  - Floor: the operative floor used by that file's latest graded round (section 11): "max(spread, published 5-seed SD 6.7e-05) = **0.000067**". Tonight's R2 spread is reported beside it.

**Selection.**

- Fold 2 selects and fold 1 confirms: same sign, and on fold 1 T does not lose beyond the floor either.
- **T advances to Stage C unless it LOSES to R beyond the floor.** A tie advances T, because the change is motivated by the closed-loop slope that Stage C tests, and T is not more complex than R.
- **A sub-model where T loses keeps its served features.** This is reported, and Stage C then runs the MIXED stack.
- **Topp advances instead of T only if it beats T beyond the floor** (it is the more complex arm).

**Mandatory lines, per arm and fold:**

- quintile responsiveness slope (each sub-model's registered form);
- team-level slope of realised on predicted, and the SD of team predictions vs realised;
- cells: weeks 0-3, and each team's games 1-6;
- level calibration on the held-out season (predicted vs actual class/make/OREB share).

**Compute estimate.** The PO first + cont refits run on lane J's box-parallel trainers (`scripts/train_*_par*`, when committed).

### 7.2 Stage C (paired closed loop, fold 2, 200 seeds, box)

**Stacks:**

| arm | stack |
|---|---|
| S0 | the served stack, on the v3 inputs rebuilt tonight |
| S1 | Stage B winners (T or Topp where they advanced, served features elsewhere), point estimates only |
| S2 | S1 plus a per-game draw of every team rate from its E3 (V0) estimation variance v |
| S3 | S1 plus the draw from the O1a variance (`team_rate_variance_O1a_v1.parquet`). The means stay V0, and the pre-check above passed at games 1-6 |

**Primary.** The G9 calibration slope.

**Co-primary.**

- slope(close on sim mean margin);
- the close-referenced within-season term k_within (`scripts/diag_g9_g6_margin_v1.py --part close`, section 1.2 method).

**Must not regress beyond the paired A/B floors:**

- G5 margin SD ratio and total SD ratio;
- G5 home/away score correlation;
- G9 margin bias and total bias;
- G1 possessions mean and SD;
- the G4 lines.

**Decision.**

- Each of S1, S2 and S3 is compared with S0, and S2 and S3 with S1, on paired seeds.
- An arm wins only if it improves the primary beyond the A/B floor with no must-not-regress line broken.
- Ties go to S1 (no draw), then S2, then S3.

**Draw mechanics (keeps the engine on lookup tables):**

- **K = 64** perturbed input sets per game, precomputed by the inputs builder.
  - Why 64: a set of K draws reproduces the draw variance to about sqrt(2/K) = 18% for one game, and to under 1% pooled over 5,710 games.
  - Memory: K x G x 2 sides x (27 team + 16 PO-block columns) x 4 bytes = about 126 MB, which fits the box workers.
  - With 200 seeds each draw is used about 3 times per game.
  - 32 is the fallback if memory binds; 128 buys nothing measurable at 200 seeds.
- **Joint draw per game.**
  - For each team, each of its 17 rate-sides is drawn as c_k = c + sqrt(v) e_k (S3: v = v_o1a), with e_k ~ N(0, 1).
  - Draws are **independent** across a team's rates, sides and the two teams. No as-of estimate of the between-rate covariance exists today. Estimating one from the filter innovations is a follow-up, not in this round.
  - Each drawn team rate is then mapped into every model feature that consumes it, exactly as the adapter maps point estimates, so the offence row and the opponent's opposite-side row see the same draw. PO interaction products are recomputed.
- **Selection in the engine.**
  - Each simulated game picks its draw index once: k = floor(K u), with u from the RNG family `team_rate` seeded on (seed, game_id, family).
  - It is a separate family, so no other stream advances. Paired arms (S2 vs S3) use the same k for the same (seed, game_id) and differ only in the drawn values.
- **K = 1 reproduces S1 bit for bit.** The single set is the point estimate itself, and the hook makes NO RNG call when K = 1. This is asserted by a parity run against S1's digest before any S2/S3 seed is read.
- **Files (to write tomorrow; default-off; not written today):**
  - `src/cbb_sim/engine/team_rate_draw.py` (new): builds the (K, G, 2, F) arrays from a features table plus a variance table, and does the draw-index lookup.
  - `scripts/build_engine_inputs_trdraw_v1.py` (new): writes a versioned sibling of the v3 inputs with the K axis.
  - A minimal hook in `src/cbb_sim/engine/loop.py`, and `EventAdapter.predict` in `src/cbb_sim/engine/adapters.py`. Where the loop gathers `inp.team_static[gidx, off]` and the event adapter gathers `team_block[gidx, off]`, the hook gathers `[k_g, gidx, off]` when `ENGINE_TEAM_RATE_DRAW` is not `off`.
  - One new family name in `src/cbb_sim/engine/rng.py`.
  - Default `off`, so the served path is untouched (parity reference v6/v3 must pass unchanged).

---

## 7a. Addendum to section 7 (PM items 2026-09-30 about 12:20; written about 12:30 EDT; COMMITTED BEFORE ANY STAGE B RUN)

Section 7 was already committed (`d5a7abb`), so these items are added here rather than edited in.

### 7a.1 Key coverage: root cause, fix, and FINAL table names

**Root cause (one cause only).** Six D-I, non-truncated games that have pbp have NO hoopR team-box rows at all. The design frames include them because designs are built from pbp; the estimator's table did not, because it is built from the box.

- **The games:** 401492245 (2022-11-26), 401587235, 401603093, 401706980, 401714520 and 401711714. That is 12 team-games.
- **Missing keys per design:** 773 possession_outcome design rows, 572 fg_make and 388 rebound, on the offence and defence keys alike.
- **Candidates ruled out** (no missing key traces to any of them):
  - teams with no history;
  - non-D-I opponents;
  - the 2022-11-25 double-header;
  - crosswalk gaps.
- The list is in `results/team_rate_estimator/missing_keys_v2.csv`.

**Fix** (`scripts/exp_team_rate_estimator_v5.py`). Those team-games enter the panel as SCHEDULE-ONLY rows with zero counts, so they give no observation and no Kalman update.

- The game still takes one index step, so process variance q accrues once, as for any game played.
- Each such row gets exactly what the estimator says entering that game: the filtered state from strictly earlier games.
- A team with no history at all would get the estimator's own prior (rho c_prev, P0). No other rule was added.
- Parameters are unchanged: the six games carry no outcome, so no fit sees them.
- Changes vs v2:
  - E3: 18 new rows; 442 later rows of the affected teams change, through the extra q step;
  - E3opp: 4,329 rows change, because opponent adjustment propagates.

**Coverage is asserted in code: 0 missing keys** for every one of these six design-fold combinations:

| sub-model | F1 rows (seasons <= 2024) | F2 rows (seasons <= 2025) |
|---|---:|---:|
| possession_outcome (round-2 design) | 2,184,697 | 3,038,628 |
| fg_make (`design_v2_shotshooter`) | 1,608,774 | 2,239,678 |
| rebound (`design_round3`) | 1,144,032 | 1,536,570 |

These are the same design files lane J's `train_*_par_v1.py` trainers default to. So `--team-rate-missing raise` is the correct setting, and the box should hit no missing keys.

**FINAL Stage B / C table names**

| file | contents | used by |
|---|---|---|
| `data/processed/team_rate_features_E3_v3.parquet` | 78,022 rows x 59 columns | arms T and TO |
| `data/processed/team_rate_features_E3opp_v3.parquet` | same shape | arm Topp, possession_outcome only |
| `data/processed/team_rate_variance_O1a_v2.parquet` | 78,022 x 41 | S3's draw variance |

- v1 and v2 of each file stay on disk and are superseded for Stage B/C.
- None of them is committed (about 29 MB each).
- Regenerate with `exp_team_rate_estimator_v5.py --part features | opp | o1a`. These need the v4 pa3 fit outputs in `results/`.

### 7a.2 Stage B: additional arm TO (rebound and possession_outcome only)

**Why.** The season-drift lane reported (`docs/tests/season_drift_anchor_round_2026-09-30.md`, `docs/models/season_drift/experiments.md`) that anchor O was the only anchor passing the drift-stops check.

- Anchor O models the target relative to the as-of in-season league level, as a logit offset, with the prior season's end level on day 0.
- It fixed rebound's held-out level (-1.14 -> -0.08 pp) and improved possession_outcome calibration.
- It was NOT selected standalone: rebound's 2.0 pp decile calibration gate fails in the top OREB decile, a team-responsiveness defect already present in the reference. That gate is not waived.

**The arm.** TO = arm T (E3 v3 features through the adapter) plus anchor O, applied through `src/cbb_sim/season_anchor.py` (the season-drift lane's module).

- Registered for rebound and possession_outcome (`first` and `cont`).
- fg_make gets no TO arm, because the anchor was not tested there.

**Gates: all unchanged**, and explicitly including:

- rebound's 2.0 pp worst-decile calibration gate;
- the held-out level line;
- each sub-model's quintile responsiveness gate.

**Decision.** T vs TO is decided by the sub-model's own primary (quoted in 7.1) and its gates, beyond the floor from R2. Ties go to T, the simpler arm. TO is compared with T, not with R. Whether any team-rate arm advances at all is still decided by T vs R under 7.1.

### 7a.3 Definition of the table's `L` columns (for comparison with the season-drift anchor)

`<rate>_<side>_L` is the league level that `<rate>_<side>_c` is centred on. The row's prediction is L + c.

- **In season:** the league's cumulative rate in the row's season, i.e. the sum of the numerator divided by the sum of the denominator. It is taken over all D-I team-box team-games (both teams of every game) with game_date strictly before the row's game_date. Schedule-only rows contribute zero counts.
- **Day 0** (no earlier game that season): the PRIOR season's final league level, i.e. the full-season sum of the numerator divided by the sum of the denominator. For 2022, which has no prior season in the panel and is never scored, the day-0 value is 2022's own first-week pooled level.
- **Off vs def:** L is identical for the off and def columns of a rate. The league's rate against itself is one number.
- **Counting:** it is a pooled rate, not a mean of team rates, and it is not smoothed. Its numerators and denominators are the box counts of section 1.2. For example, tov / P uses P = FGA - OREB + TOV + 0.44 FTA from the box, not pbp possessions.
- **Built by:** `exp_team_rate_estimator_v2.league_asof`.

**Relation to anchor O.** The day-0 rule is the same as anchor O's (prior season's end level). In season, both are as-of cumulative levels, but over different counting bases: the box here, the sub-model's own target events there. So the two can differ by that base's definitional gap (e.g. live rebound opportunities vs box OREB + opponent DREB).

**No double counting.** Under TO, L centres the FEATURES (a team relative to the league), while anchor O offsets the TARGET's intercept. These are different objects.

---

## 7b. Local Stage B smoke test (run 2026-09-30, 12:14-12:37 EDT; NOT a Stage B result)

Script: `scripts/diag_team_rate_stageb_smoke_v1.py`. Output: `results/team_rate_estimator/stageb_smoke_v1.json`.

**Setup**

- One refit date (2025-01-01), fold F2.
- Training rows: a 25% sample of the rows before the cut. Test: the following 31 days.
- LightGBM runs at n_jobs = 1, monkeypatched in-process only.
- The served modules' fit functions are called UNMODIFIED. Lane J's box trainers were not yet committed when this started.
- Table: E3 **v2** with `missing="keep_served"`, because the v3 coverage fix (7a.1) landed while the test was running. The rows kept served are exactly the box-less games that v3 now covers.

**Results.** Every sub-model runs end to end on substituted features.

| sub-model | replaced columns | rows kept served (v2 gap) | test log loss, served / E3 | wall time |
|---|---|---:|---|---:|
| possession_outcome `first` (lgbm) | the 8 PO style columns, plus the 4 recomputed interactions | 673 | 1.51942 / 1.52018 | 610 s |
| fg_make FGA_rim (lgbm, B1 features) | `off_make_c`, `def_allow_c` | 188 | 0.67441 / 0.67435 | 642 s |
| rebound (lgbm, S1 features) | `off_oreb_c`, `opp_def_dreb_c` | 388 | 0.63655 / 0.63727 | 125 s |

These log losses come from one cut on a 25% sample. They are plumbing checks and decide nothing: Stage B's arms, floors and folds are in sections 7 and 7a.

---

## 7c. Stage C draw code, proofs and box runbook (written 2026-09-30 about 13:00 EDT; DEFAULT OFF, nothing adopted)

### 7c.1 Files

| file | role |
|---|---|
| `src/cbb_sim/engine/team_rate_draw.py` (new) | `ENGINE_TEAM_RATE_DRAW=<path.npz>` loads K sets, aligned to the inputs' game order (asserted). The index is k = floor(K u), with u the first draw of the new family `team_rate`; with K = 1 there is no RNG call. |
| `src/cbb_sim/engine/loop.py` | Chunk start: `trd`, `ksim`. Step: `kk`. The gather sites for the clock team row, the event team row, fg_make and rebound take `trd.team_static_k[kk, gidx, off]` instead of `inp.team_static[gidx, off]`. |
| `src/cbb_sim/engine/adapters.py` | `EventAdapter.predict(..., kidx=None)` reads `team_block_k[kidx, gidx, off]`. |
| `src/cbb_sim/engine/rng.py` | Adds the family `team_rate`. |
| `scripts/build_engine_inputs_trdraw_v1.py` (new) | Builds the draw files. It defaults to the v3 replay inputs (`data/processed/models/engine_v3`, the path imported from `build_engine_inputs_v3_replay.OUT`) and does not edit them. Every input is a path argument: the table, variance, K, seed and `--artifact` (recorded in the meta). |
| `tests/test_team_rate_draw.py` | 4 tests pass. |
| `scripts/diag_team_rate_draw_proofs_v1.py` | The proofs below. |

Hooks were committed as `725e8f1`. The engine tests passed 20/20 after the hooks.

- The FT gather site is untouched, because FT features hold no team rate.
- **Incompatibility:** `ENGINE_LATE_GAME` wraps the event adapter, so combining it with the draw is not supported until that wrapper forwards `kidx`. Both are default off.

**Key coverage for the engine universe.** The F2 engine inputs hold 7 games with neither pbp nor box, which are in no design. They are covered by `scripts/exp_team_rate_estimator_v6.py` in the same way as 7a.1 (schedule-only rows), and asserted: 0 missing of 5,710 x 2.

**Final tables for Stages B and C:**

- `team_rate_features_E3_v4.parquet` and `..._E3opp_v4.parquet` (78,036 rows);
- `team_rate_variance_O1a_v3.parquet`.

**v4 supersedes v3.** They differ only in the 14 new rows and in 233 later rows of the teams that played those games (1,744 rows for E3opp). Stage B should read v4 as well, so that B and C use one table.

### 7c.2 Proofs (local, 2 cores, on the served v2 inputs; v3 does not exist yet)

**(a) Flag off.** 60 games x 5 seeds against `docs/ops/parity_reference_windows_v6.json`: **PASS, bit-identical**, sha256 `0d4ddccc64d7...`.

**(b) K = 1.**

- `k_from_book` leaves the `team_rate` counter at 0, so there is no RNG call.
- A K = 1 identity file (base values) run through the hook gives the same digest, **PASS `0d4ddccc...`**.
- The hook is engaged: a K = 1 file carrying the substituted S1 point estimates changes 310 of 320 (game, seed) scores in the end-to-end run.

**(c) Moments of the perturbed sets, K = 64, all 5,710 games, the 16 mapped columns x 2 sides.**

- Standardised mean error (mean over K minus the estimate, divided by sqrt(v/K)): its mean lies in [-0.025, +0.036] and its SD in [0.984, 1.023], against the target N(0, 1).
- SD over K divided by sqrt(v): its mean lies in [0.993, 0.999]. The small-sample bias at K = 64 predicts 0.996.
- This holds for BOTH the E3 variance (S2) and the O1a variance (S3). The two share the same normals by construction, so the arms are paired.

**Support clip.** Rates are clipped to [1e-4, 1 - 1e-4] (binomial) or >= 1e-4 (the Poisson FT rate). The clip **never binds**: 0 of about 1.46M draws per rate, in both files.

**(d) Paired streams.**

- A K = 4 zero-variance file consumes the `team_rate` stream (k counts 88/65/84/63 over 300 simulations) and reproduces the reference digest, **PASS `0d4ddccc...`**.
- Every other family's keys and counters are identical with and without the draw.
- `k_from_book` equals the offline `k_index` exactly.
- So a game's margin under seed s changes only through the drawn inputs.

**(e) End-to-end run: 40 F2 games x 8 seeds, the served models.**

| arm | ran (games x seeds) |
|---|---|
| S0 | 40 x 8 |
| S1 (K = 1 substituted) | 40 x 8 |
| S2 (K = 64, E3 v) | 40 x 8 |
| S3 (K = 64, O1a v) | 40 x 8 |

The between-seed spread of a game's drawn inputs matches the draw SD. Median SD across seeds vs SD over K:

| input | S2 | S3 |
|---|---|---|
| `off_tov_c` | 1.06 vs 1.03 | 1.11 vs 1.08 |
| `off_make_c__rim` | 0.0169 vs 0.0174 | 0.0180 vs 0.0184 |
| `off_oreb_c` | 0.0186 vs 0.0190 | 0.0192 vs 0.0197 |
| `off_3pa_c` | 1.47 vs 1.59 | 1.51 vs 1.63 |

This is a mechanical check. No gate is read from it, and these are not the Stage B models.

### 7c.3 Box runbook (tonight)

**Prerequisites that other lanes or ops must provide; none of these exist yet:**

1. **The v3 replay inputs** from the replay lane: `data/processed/models/engine_v3/{games,arrays,names}_F2_2025.*` and `event_block_F2_2025.npz`, in the v2 game order. There must also be a way for `run_engine.py --input-dir data/processed/models/engine_v3` to load them. `EngineInputs.resolve_tag` must find the bare `F2_2025` tag, and S0 needs the replay lane's event-block mechanism, because the served `EventAdapter` still reads its own `team_block.npz` when the draw is off. With the draw on, the event block comes from the draw file.
2. **The Stage B retrained artifacts,** served through `ENGINE_*` modes or directories that the adapters can load. The draw file does not depend on them; `--artifact` only records the paths.
3. **These files on the box** (about 43 MB). They are local and untracked, so push them through an HF bulk key or copy them:
   - `data/processed/team_rate_features_E3_v4.parquet` (plus `E3opp_v4` if possession_outcome's Topp advances);
   - `data/processed/team_rate_variance_O1a_v3.parquet`.

**Build the draw files** (about 20 s each on one core; outputs about 2 MB for S1 and about 126 MB each for S2 and S3). Use the same seed for S2 and S3, so they are paired:

```
python scripts/build_engine_inputs_trdraw_v1.py --variance none --K 1 \
    --table data/processed/team_rate_features_E3_v4.parquet --out data/processed/models/engine_v3_trdraw/S1_K1
python scripts/build_engine_inputs_trdraw_v1.py --variance e3 --K 64 --seed 20260930 \
    --table data/processed/team_rate_features_E3_v4.parquet --out data/processed/models/engine_v3_trdraw/S2_e3_K64
python scripts/build_engine_inputs_trdraw_v1.py --variance o1a --K 64 --seed 20260930 \
    --table data/processed/team_rate_features_E3_v4.parquet \
    --variance-table data/processed/team_rate_variance_O1a_v3.parquet --out data/processed/models/engine_v3_trdraw/S3_o1a_K64
```

For a mixed stack, where a sub-model's T lost in Stage B, rebuild with that sub-model's columns kept at their served values. The builder needs a `--keep-served <submodel>` flag for that, and it does not exist yet. It is a small follow-up, to be written only if Stage B returns a mixed verdict.

**Parity.** Run the parity gate first, with the draw UNSET: `bash scripts/run_aws_sweep.sh --tag TRD_parity --parity only`.

**Sims.** Paired seeds, each arm on streams A (0-199) and B (1000-1199), with the same v3 input dir and the same retrained-model flags:

```
bash scripts/run_aws_sweep.sh --tag TRC_S0_A --parity skip --input-dir data/processed/models/engine_v3 --seeds 200 --seed-offset-start 0
ENGINE_TEAM_RATE_DRAW=$PWD/data/processed/models/engine_v3_trdraw/S1_K1.npz \
  bash scripts/run_aws_sweep.sh --tag TRC_S1_A --parity skip --input-dir data/processed/models/engine_v3 --seeds 200 --seed-offset-start 0
ENGINE_TEAM_RATE_DRAW=$PWD/data/processed/models/engine_v3_trdraw/S2_e3_K64.npz \
  bash scripts/run_aws_sweep.sh --tag TRC_S2_A --parity skip --input-dir data/processed/models/engine_v3 --seeds 200 --seed-offset-start 0
ENGINE_TEAM_RATE_DRAW=$PWD/data/processed/models/engine_v3_trdraw/S3_o1a_K64.npz \
  bash scripts/run_aws_sweep.sh --tag TRC_S3_A --parity skip --input-dir data/processed/models/engine_v3 --seeds 200 --seed-offset-start 0
# repeat each with _B and --seed-offset-start 1000 (the A/B noise floor)
```

**Expected wall time and memory.**

- The v5b read (`engine_v1_gates_F2_2025_s200_v5b_full_2026-09-18.md`) did 125 seeds per stream in about 21 min at 70 workers per stream. On 192 vCPUs, one 200-seed arm-stream is therefore about 13-15 min at about 180 workers.
- The whole plan is 4 arms x 2 streams = 8 runs, about 1.8-2 h if run back to back, or about 1 h with two concurrent at about 90 workers each.
- The draw adds one gather per team-rate site; its cost is negligible.
- Each worker memory-maps nothing: it loads the whole npz, about 126 MB per worker for S2 and S3, which is about 23 GB at 180 workers and fits a 384 GB box.

**Grading.** Grade each arm with `scripts/eval_gates.py` and the close-referenced split (`scripts/diag_g9_g6_margin_v1.py --part close`, pointing `RUN_A`/`RUN_B` at the arm's A/B dirs). Apply the section 7.2 rule.
