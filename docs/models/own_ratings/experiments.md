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
