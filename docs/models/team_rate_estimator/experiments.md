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
