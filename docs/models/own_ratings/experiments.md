# Own team ratings (`own_ratings`): day-1 priors, experiments (append-only)

**Owner:** Lane N (2026-09-30).

**What this is.** Our own ridge team ratings (`cbb_sim.ratings.own_ratings`, reproduced as-of any date by `scripts/build_own_ratings_asof_v1.py`) start every season from a prior built out of last season's final fit. The day-1 build makes several choices without evidence (`docs/ops/own_ratings_daily_2026-09-30.md` section 4). This file bakes off the ones that determine the early-season rating.

- **Origin of the current rule:** control_engine bake-off (`docs/models/control_engine/experiments.md` R1): lambda 5, carry weight w = 0.8 for efficiency and tempo, selected by one-step-ahead RMSE over the whole of 2022-23 (one season-to-season transition).
- **Relation to the team-rate estimator lane** (`docs/models/team_rate_estimator/experiments.md` sections 1-6): same questions (carry, continuity, coach change), answered there for box RATES. Ratings are a different object (a joint ridge over opponents), so the questions are re-asked here. The continuity table is REUSED from that lane (`results/team_rate_estimator/continuity_v2.parquet`, built by `exp_team_rate_estimator_v2.continuity()`), not rebuilt.
- **Status:** NOTHING here adopts anything, changes the manifest, or touches `data/processed/ratings` or engine code. The PM decides.

---

## 1. Pre-registration (written 2026-09-30 about 13:55 EDT; COMMITTED BEFORE ANY RUN)

### 1.1 Object and what is held fixed

- **Rating:** the efficiency ridge (off / def dummies, intercept, home, away; points per 100 possessions), ridge toward a prior `lambda * ||b - b_prior||^2`, fitted on the season's D-I games strictly before date D. Every team effect is re-centred on the mean over the season's team set on every date (the league-mean-relative rule); the league level sits in the intercept.
- **Home/away/neutral:** as the builder does: `site_home`, `site_away` indicator terms, neutral is the reference; the fixed-term priors (intercept, home, away) are last season's final values in EVERY arm (not varied here; see 1.8).
- **Tempo:** held at the current rule (lambda 5, w 0.8) in every arm. Predicted possessions for each game come from the library's own as-of tempo fit. Arms vary only the efficiency prior and lambda. Margin depends on tempo only through a scale.
- **Prior chain:** the previous season's FINAL fit is the library's current-rule chain (2022 unprimed, then 2023, 2024 chained at w 0.8) for every arm. A full season (~30 games at lambda 5) makes the final fit nearly prior-free; varying the chain per arm is not done.
- **Team set:** the season's played D-I team set (`tg`, the library default and parity mode). It is membership, not performance. Live day 1 uses the schedule / CBBD set; that choice is not tested here (1.8).
- **Games:** `load_team_games` (D-I, non-truncated, two usable box rows). Truth: verified finals `data/processed/truth/game_finals_v2.parquet` (home_score - away_score, OT included). Games without a verified final are dropped and counted.

### 1.2 Folds (transitions are the unit)

| fold | fit (training transitions, weeks 0-7 of the later season) | test (weeks 0-7) | role |
|---|---|---|---|
| F1 | 2022 -> 2023 (games of 2022-23) | 2023 -> 2024 (games of 2023-24) | confirmation |
| F2 | 2022 -> 2023 and 2023 -> 2024 | 2024 -> 2025 (games of 2024-25) | selection |

2025-26 is SEALED; nothing from season 2026 is loaded. **Week w** = days [7w, 7w+7) from the season's first D-I game date. Bands: 0-1, 2-3, 4-7.

### 1.3 Arms

Notation: `c_prev` = team's previous-season final centred off / def effect (0 if the team has no previous rating); `cm` = mean of `c_prev` over the season-S members of the team's season-S conference (conference id from the hoopR season-S schedule, the team's modal `*_conference_id`; public preseason); `cont` = returning share of previous-season minutes, `coach` = head-coach change flag (both from `continuity_v2.parquet`; `cont` centred on the training mean, missing -> 0 after centring). Scalar weights are common to off and def.

| arm | team-effect prior | lambda | free params (fitted on training) | simplicity rank |
|---|---|---|---|---:|
| **Z** floor | 0 (league mean) | 5 | none | 0 |
| **R** current rule | 0.8 c_prev | 5 | none | 1 |
| **F** fitted carry | w c_prev | 5 | w | 2 |
| **C** conference carry | cm + w_c (c_prev - cm) | 5 | w_c | 3 |
| **K** continuity/coach carry | (r0 + r1 cont + r2 coach) c_prev | 5 | r0, r1, r2 | 4 |
| **D** fitted decay | 0.8 c_prev | lambda_b per band 0-1 / 2-3 / 4-7 | 3 lambdas | 5 |

- **Fitting.** The predicted margin is LINEAR in (w, w_c, r0, r1, r2) for fixed lambda (the solve and the re-centring are linear in the prior), so F, C and K are fitted by least squares of realised margin on the prior components over the training games of weeks 0-7. D picks each band's lambda from {0.5, 1, 2.5, 5, 10, 20, 40, 80} by training squared error (bands are separable: each date's fit depends only on that date's lambda). Nothing is fitted on a test season.
- **New-to-D-I prior (stated rule, every arm).** A team with no previous-season rating gets `c_prev = 0`: the league mean in Z, R, F, K, D; the conference mean `cm` in C (that is C's definition). Reported as a separate cell (underpowered; labelled).
- **Continuity input caveat** (as in the estimator lane): the roster is the athletes listed in the team's first 3 box scores of season S (membership, not performance). It is the one non-performance input from inside the season and is flagged.

### 1.4 Primary metric, segments, noise floor

- **Predicted home margin** for a game on date D: `poss_hat * (eff_home - eff_away) / 100`, with `eff_home - eff_away = off_h + def_a + site_h - (off_a + def_h + site_a)` from the arm's rating as of D, `poss_hat` the as-of tempo prediction.
- **Primary: fold-2 margin MAE, pooled weeks 0-7**, also by band 0-1, 2-3, 4-7.
- **Co-reported:** log loss of the home-win probability `Phi(pred / sigma)`, sigma fitted per arm and band on the training games by win-outcome likelihood (grid 5.0-20.0 by 0.1).
- **Reference (never a feature):** MAE of the arm's prediction against the closing home margin (-median closing spread across providers, `lines_close_v2_verified.parquet`), on the games with a line; the close's own MAE vs the final is printed as a benchmark.
- **Responsiveness:** teams bucketed into quintiles of their day-1 net rating under R (off_c - def_c, common buckets across arms); per band, team-perspective mean predicted and mean realised margin per quintile; slope of the 5 realised means on the 5 predicted means. Also the game-level OLS slope of realised on predicted.
- **Noise floor:** paired team-block bootstrap of (arm - R) per game loss, 1,000 draws, seed 20260930: each game's paired difference is split half to each of its two teams; teams are resampled with replacement; the statistic is the sum of drawn teams' differences over the sum of their game shares. 95% interval. The model is deterministic, so there is no seed retrain; the bootstrap is the floor. Fold-1 sign is the confirmation.
- **New-D-I cell:** games involving a team with no previous rating, per band where populated; underpowered and labelled.

### 1.5 Decision rule

An arm qualifies if ALL hold:
1. its fold-2 pooled weeks 0-7 MAE difference vs R has a 95% interval entirely below 0;
2. its fold-1 pooled difference has the same sign (negative);
3. its fold-2 pooled log-loss interval vs R is not entirely above 0.

Among qualifiers, the lowest fold-2 pooled MAE wins, unless its interval (vs R) contains the point difference of a simpler qualifying arm, in which case the simpler one wins (order: rank column). If no arm qualifies, R stands. Z qualifying would be a finding (carry harms), not a recommendation to drop the prior without the PM. Combining components (e.g. F + D) is not an arm here; a follow-up only if two components qualify separately.

### 1.6 Execution

- Fitter: `scripts/exp_own_ratings_day1_priors_v1.py` (reuses `cbb_sim.ratings.own_ratings` data loading, ridge accumulator and the current-rule chain; asserts that its R arm reproduces the library's as-of `pred_off_eff` to 1e-8 on the test seasons). Writes versioned predictions under `results/own_ratings_day1/`.
- Grader: `scripts/grade_own_ratings_day1_priors_v1.py`, one blind pass over the prediction table keyed by arm.
- 1 core per process; threads pinned to 1.
- Results: section 2 of this file, once, plus `docs/tests/own_ratings_day1_priors_2026-09-30.md` if the tables are long.

### 1.7 Honesty

Every rating for date D uses games strictly before D (asserted). Previous-season finals, conference membership and the continuity roster listing are preseason-public or membership-only. Lines are a grading reference only.

### 1.8 Not tested here (named, not chosen)

The fixed-term priors (intercept / home / away carry), the re-centring team set, the early-season D-I membership rule, reduced-weight non-D-I games, separate off / def / tempo weights, the tempo carry, and a KenPom preseason arm (no 2027 snapshot).

---

## 2. Results (run 2026-09-30, about 13:53-13:56 EDT; 1 core; fitter 25 s, grader under 1 min)

**Code and outputs.** Fitter `scripts/exp_own_ratings_day1_priors_v1.py`, grader `scripts/grade_own_ratings_day1_priors_v1.py`; `results/own_ratings_day1/{basis,preds,day1}_v1.parquet`, `params_v1.json`, `grade_v1.json`, logs. The R arm reproduced the library's as-of prediction to < 1e-8 on every date (asserted). Every game in weeks 0-7 had a verified final (0 dropped). Test games: F2 2,078 (all with a close), F1 2,112 (1,946 with a close). Teams with no previous rating: 5 (2023), 1 (2024), 2 (2025); all have a season-S conference.

### 2.1 Fitted parameters (training transitions only)

| fold | w_F | w_C | K: r0 / r1 (cont) / r2 (coach) | D: lambda 0-1 / 2-3 / 4-7 |
|---|---:|---:|---|---|
| F1 (fit 2023) | 1.001 | 0.634 | 1.032 / -0.141 / -0.184 | 2.5 / 5 / 5 |
| F2 (fit 2023+2024) | 1.008 | 0.642 | 1.051 / +0.074 / -0.240 | 5 / 5 / 5 (= R) |

### 2.2 Primary: fold-2 margin MAE by week band, paired vs R (team-block bootstrap, 1,000 draws, 95%)

R absolute MAE: 10.217 / 9.844 / 9.038 / 9.629 (0-1 / 2-3 / 4-7 / all). Negative = better than R.

| arm | 0-1 | 2-3 | 4-7 | all (primary) | F1 all | F1 sign |
|---|---|---|---|---|---|---|
| Z | +2.660 [+2.292, +3.006] | +0.923 [+0.704, +1.161] | +0.820 [+0.638, +0.981] | +1.380 [+1.230, +1.541] | +1.196 [+1.063, +1.336] | same (worse) |
| F | -0.155 [-0.264, -0.048] | -0.021 [-0.092, +0.044] | -0.046 [-0.096, +0.003] | -0.069 [-0.109, -0.028] | -0.075 [-0.110, -0.039] | same |
| **C** | **-0.267 [-0.360, -0.164]** | **-0.124 [-0.188, -0.060]** | **-0.078 [-0.116, -0.038]** | **-0.147 [-0.183, -0.113]** | **-0.148 [-0.185, -0.109]** | same |
| K | -0.186 [-0.303, -0.064] | -0.056 [-0.124, +0.009] | -0.045 [-0.096, +0.006] | -0.089 [-0.134, -0.044] | -0.085 [-0.127, -0.047] | same |
| D | 0 (identical to R on F2) | 0 | 0 | 0 | +0.019 [-0.003, +0.042] | - |

Fold-1 C by band: -0.264 [-0.354, -0.176] / -0.106 [-0.174, -0.042] / -0.091 [-0.127, -0.054].

### 2.3 Log loss of the win probability (sigma per arm and band fitted on training), fold 2, paired vs R

| arm | 0-1 | 2-3 | 4-7 | all | F1 all |
|---|---|---|---|---|---|
| R (absolute) | 0.4272 | 0.5421 | 0.4602 | 0.4765 | 0.4988 |
| Z | +0.109 [+0.091, +0.126] | +0.033 | +0.039 | +0.057 [+0.050, +0.064] | +0.044 |
| F | -0.005 [-0.011, +0.000] | -0.000 | -0.001 | -0.002 [-0.004, -0.000] | -0.002 |
| **C** | **-0.010 [-0.015, -0.005]** | **-0.007 [-0.010, -0.003]** | -0.001 [-0.003, +0.001] | **-0.005 [-0.008, -0.003]** | **-0.004 [-0.006, -0.002]** |
| K | -0.007 [-0.013, -0.002] | -0.001 | +0.001 | -0.002 [-0.004, -0.000] | -0.002 |

### 2.4 Reference: MAE of the arm's margin against the closing margin (lines are never a feature)

The close's own MAE vs the final: F2 9.44 / 9.29 / 8.72 / 9.10; F1 9.18 / 9.10 / 8.86 / 9.03 (R trails the close by about 0.5 points on F2 pooled; C by about 0.38).

| arm | F2 0-1 | 2-3 | 4-7 | all | paired vs R, F2 all | F1 all (paired) |
|---|---:|---:|---:|---:|---|---|
| R | 4.173 | 2.918 | 2.343 | 3.048 | - | 3.073 |
| Z | 9.002 | 5.344 | 4.678 | 6.127 | +3.079 | 5.546 (+2.473) |
| F | 3.812 | 2.827 | 2.200 | 2.859 | -0.189 [-0.239, -0.140] | 2.915 (-0.158) |
| **C** | **3.458** | **2.577** | **2.016** | **2.606** | **-0.442 [-0.485, -0.394]** | **2.711 (-0.362)** |
| K | 3.763 | 2.793 | 2.189 | 2.830 | -0.218 [-0.266, -0.170] | 2.923 (-0.150) |

### 2.5 Responsiveness (slope of realised on predicted; 1.0 = calibrated spread)

Quintiles: teams bucketed by their R day-1 net rating in the test season (common buckets); slope of the 5 team-perspective realised means on the 5 predicted means. Game: OLS slope, home perspective.

| arm | F2 quintile 0-1 / 2-3 / 4-7 / all | F2 game all | F1 quintile all | F1 game all |
|---|---|---:|---:|---:|
| Z | 1.923 / 1.531 / 1.488 / 1.645 | 0.958 | 1.614 | 0.969 |
| R | 1.087 / 1.003 / 1.063 / 1.054 | 1.011 | 1.025 | 1.007 |
| F | 0.976 / 0.920 / 0.989 / 0.964 | 0.931 | 0.938 | 0.934 |
| **C** | **1.054 / 1.001 / 1.058 / 1.041** | **1.011** | **1.018** | **1.008** |
| K | 0.984 / 0.936 / 0.996 / 0.974 | 0.940 | 0.936 | 0.932 |
| D | = R | = R | 1.017 | 0.981 |

Every carry arm slopes with actuals across quintiles. Z (no carry) is flat early: game slope 0.27 on F2 weeks 0-1, and its quintile slope of 1.9 means its predictions barely separate the quintiles. F and K over-spread (game slope 0.93-0.94: a full carry, w about 1.0, exaggerates spread); C stays calibrated (1.01).

### 2.6 New-to-D-I cell (UNDERPOWERED: F2 24 games, 2 teams; F1 11 games, 1 team)

| fold | R MAE | C - R (weeks 0-7) | F - R | K - R | Z - R |
|---|---:|---|---|---|---|
| F2 | 8.74 | -0.564 [-1.073, -0.121] | +0.099 | +0.175 | +1.765 [+0.410, +3.133] |
| F1 | 10.65 | -0.119 [-1.077, +0.815] | -0.137 | -0.099 | +0.548 |

C gives a new team its conference mean; every other arm gives 0 (league mean). Direction favours C on both folds, but with 1-2 teams per test season this is an anecdote, not evidence. These games are about 1% of the pool and do not drive the pooled ranking.

### 2.7 Decision under the registered rule (1.5): C qualifies and wins; nothing adopted

| arm | F2 pooled MAE interval < 0 | F1 sign negative | F2 log-loss interval not above 0 | qualifies |
|---|---|---|---|---|
| Z | no | no | no | no |
| F | yes | yes | yes | yes |
| C | yes | yes | yes | yes |
| K | yes | yes | yes | yes |
| D | no (identical to R) | no | - | no |

Lowest F2 MAE among qualifiers: C (-0.147). Its interval [-0.183, -0.113] contains neither F's point (-0.069) nor K's (-0.089), so the simpler-arm clause does not apply. **Winner under the rule: C** (prior = cm + 0.64 (c_prev - cm): carry 64% of last season's final rating and shrink the rest toward the team's current-conference mean instead of the league mean). The PM decides adoption.

### 2.8 Interpretation

- **The carry is worth about 1.4 points of weeks 0-7 MAE** (Z vs R); carrying last season is not in question.
- **The league mean is the wrong shrinkage target.** A single fitted weight goes to w about 1.0 on both folds (F): it helps weeks 0-1 but over-spreads (slope 0.93). Shrinking toward the current-conference mean fixes both. The part of a rating that does not persist is largely the within-conference part; the conference level persists. C gains at every band, most in weeks 0-1 (-0.27 points, about 2.6%), and is much closer to the market (-0.44 points vs the close on F2).
- **Roster continuity and coach change add little for ratings** (K vs F: -0.02 on F2 pooled; r1 changes sign between folds; r2 about -0.2 on both folds, so a coach change lowers the carry). This mirrors the estimator lane's P1 result. K was not combined with C (combinations are a follow-up under 1.5).
- **The in-season decay schedule (D) found nothing.** F2 training picked lambda 5 in every band; F1 picked 2.5 for weeks 0-1, which was worse on test (+0.06). The constant lambda stands on this evidence.

### 2.9 What did not run / caveats

- C + K (conference carry with continuity/coach) and C with a refit lambda: not arms (1.5); a registered follow-up if the PM wants them.
- Tempo carry, fixed-term (intercept/home) carry, team-set and D-I-membership rules: held at R (1.8).
- The prior chain is the current rule for every arm (1.1). If C is adopted, the chain changes slightly; a chained-C rebuild and parity check are needed before serving.
- The gain is at the rating level only; there is no sim-level or gate evidence. Adoption goes through the usual paired closed-loop check.
- The conference mean excludes teams with no previous rating (they would otherwise pull it toward 0). Conference = the modal conference id on the season-S hoopR schedule.

---

## 3. Chained-C rebuild and paired closed loop (PM ruling 2026-09-30: C is VALIDATED-PENDING-SHIP-ACTION; registered about 14:05 EDT, COMMITTED BEFORE THE CLOSED LOOP RUNS)

### 3.1 Conference-mean definition (confirmed; no rerun needed)

`cm_i` = mean of the previous-season FINAL off / def effects over the members of team i's NEW (season-S) conference that have a previous-season rating. Membership = the team's conference id on the hoopR season-S schedule: a per-season team attribute (0 teams carry more than one id in 2024 or 2025), known preseason; only previous-season finals enter. This is exactly what the section-2 run used (`conference_map(season)` with season = the test season). Realignment check, 2024 -> 2025: 21 teams changed conference (for example UCLA 26: 21 -> 7, Stanford 24: 21 -> 2); each is pooled with its 2025 conference.

### 3.2 Chained-C ratings (built before this registration; mechanics, not a selection)

- `scripts/build_own_ratings_C_v1.py` -> `data/processed/ratings_C_v1/own_ratings_{2022..2025}.parquet` + manifest (versioned sibling; the stored `data/processed/ratings` is untouched). Strictly walk-forward: each season's day-0 prior is built from the CHAINED-C final of the previous season. w_c = 0.6419 (fold-2 fit, section 2.1) for every season. 2022 has no prior and equals R (max 3e-14). 2026 SEALED, not built.
- `scripts/build_own_ratings_asof_C_v1.py` (sibling daily entry point): parity vs the sibling on 3 dates per season (day 1, about 1/3 and 2/3 of the season), 12 dates: team sets equal, max abs difference 0.0 on all 10 numeric columns (`docs/ops/own_ratings_C_asof_parity_2026-09-30.json`).
- Caveat: w_c was fitted on transitions whose later seasons (2023, 2024) are rebuilt with it, so chained-C 2023-24 ratings are in-sample for w_c; the fold-2 test season 2025 is not.

**Chained-C vs R across the season** (net = off_c - def_c; SD across teams averaged over dates; |C-R| = mean absolute net difference):

| season | band | SD R | SD C | mean abs C-R | share of the week-0 gap left |
|---|---|---:|---:|---:|---:|
| 2025 | 0-1 | 10.32 | 11.59 | 2.07 | 0.93 |
| 2025 | 2-3 | 11.44 | 12.46 | 1.55 | 0.70 |
| 2025 | 4-7 | 12.04 | 12.88 | 1.17 | 0.53 |
| 2025 | 8-15 | 12.73 | 13.42 | 0.89 | 0.40 |
| 2025 | 16+ | 12.95 | 13.62 | 0.81 | 0.36 |
| 2024 | 0-1 / 4-7 / 16+ | 9.60 / 10.95 / 12.01 | 10.46 / 11.53 / 12.49 | 1.73 / 0.94 / 0.64 | 0.92 / 0.50 / 0.34 |
| 2023 | 0-1 / 4-7 / 16+ | 8.63 / 10.17 / 10.96 | 8.95 / 10.46 / 11.25 | 1.26 / 0.66 / 0.44 | 0.91 / 0.47 / 0.32 |

- C is MORE spread than R at every week, and the gap grows along the chain (2023 +0.3, 2025 +0.7 to +1.3 SD points): R's 0.8 toward the league mean compresses every season's prior; C shrinks less overall (only the within-conference part).
- The prior does NOT wash out in either arm: lambda 5 is identical, so the gap decays on the same schedule (half gone by week 4-5), and about a third of the week-0 difference is still there at season end (the ridge keeps pulling toward the prior all season; the final fit therefore also carries it into the next season's prior). This is the same structure as R, not new to C.
- Offline re-grade with the CHAINED ratings (same games, same grader functions as section 2): fold-2 weeks 0-7 MAE R 9.629, chained-C 9.445, difference -0.184 [-0.230, -0.137] (bands -0.333 / -0.157 / -0.098); game slope 0.996. Fold 1 -0.176 [-0.218, -0.131] (in-sample for w_c; confirmation only). `results/own_ratings_day1/chainC_offline_regrade_v1.txt`.

### 3.3 Closed-loop design

- **Arms.** R = the served ratings; C = chained-C. Everything else is identical: base inputs `data/processed/models/engine_v3` (honest fold-2 replay), served artifacts, `--arm round2_s1`, pinned served stack. Only the four own-rating team columns (`off_rating_off_c, off_rating_def_c, def_rating_off_c, def_rating_def_c`) change, in `team_static` and in the round-2 event block (columns 8-11). Tempo columns are unchanged (C's tempo equals R's, asserted).
- **Inputs.** `engine_v3_N_R` = `build_engine_inputs_v3_tag_v1.py --tag N_R` (defaults: bit-identical to engine_v3 plus its overlay). `engine_v3_N_C` = `build_engine_inputs_v3_ratings_tag_v1.py --tag N_C --ratings-dir data/processed/ratings_C_v1` (new sibling builder, since the tag builder has no ratings parameter; its recipe applied to the served ratings reproduces engine_v3's eight rating channels exactly, parity 0.0, checked before substituting). Sub-models are NOT retrained: they were trained on R-distributed ratings, so this loop prices C as a served-input swap only.
- **Runner.** `scripts/run_po4b_closed_loop_sample_overlay_v1.py` (new wrapper around the unedited sample runner): serves the tag's overlay in-process (no Docker locally) by rebinding `adapters.ENGINE_DIR` in the parent and every worker; asserts the overlay's event block equals the input dir's. Smoke: 1 seed, workers confirmed bound to the overlay dir.
- **Sample.** `data/processed/truth/stride500_verified_v1_F2_2025.parquet` (500 games, 2024-11-04..2025-03-15). 25 seeds (0-24), 4 workers.
- **Floors (Decision 12).** R re-run at seed offsets 1000, 2000 (required) and 3000, 4000 (if time allows; if they do not finish, the report says two draws, not four). Floor (a) = max over floor runs of |floor - R| per line; floor (b) = paired game bootstrap (1,000 draws, games resampled with their 25 paired seeds kept together) of C - R on the lines computed from game rows. A line's floor is the larger of the two where both exist.
- **Grading.** `eval_gates.py --season 2025` on every run under `CBB_TRUTH=verified_v1`; `diag_pair_gate_reports.py` C vs R with each floor; `scripts/grade_own_ratings_closed_loop_v1.py` for the primaries and the weeks 0-7 cell (verified finals `game_finals_v2`).

### 3.4 Primary, vetoes, decision

- **Primary (fold 2, weeks 0-7 cell of the sample, i.e. games before 2024-12-30):** G9 calibration slope (`polyfit(sim_margin_mean, margin)`, the gates.py definition) and margin MAE of the seed-mean margin. The cell has about 40% of the sample (its n is reported; it is underpowered for slope if n < 150). The full-sample G9 slope and MAE are co-reported.
- **Vetoes:** every line of the full gate list (G1-G9 headline tables) must not move AWAY from its target by more than its floor. Per Decision 12, a 25-seed loop does not decide G5 ratio / correlation lines; they are reported, marked PROVISIONAL.
- **Pass:** the weeks 0-7 margin MAE difference C - R is negative with its game-bootstrap interval excluding 0 OR within the floor, AND the weeks 0-7 slope does not move away from 1.0 beyond its floor, AND no veto fires. The PM decides serving; nothing is adopted here.
