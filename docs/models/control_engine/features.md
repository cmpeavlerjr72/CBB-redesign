# control_engine — features

Layout per `docs/models/DOCUMENTATION_STANDARD.md`. Absolute numbers live in
`experiments.md` and `docs/tests/control_engine_F2_2026-09-10.md`; this file is
the "exactly what goes in" reference for whoever wires the sim.

Everything here is **pregame by construction**. Two independent guards:
`cbb_sim.control.features` only ever performs strictly-before as-of joins, and
`tests/test_control.py` corrupts one game's box line, refits the season, and
requires that game's own features to be bit-identical while later games move.

---

## 1. Source manifest (canonical table)

### 1a. Per team-game (offence side). Used by the rate and make-rate models.

| Feature | Dtype | Source file | Computation | Fallback value | Notes |
|---|---|---|---|---|---|
| `own_off_c` | float64 | `data/processed/ratings/own_ratings_{season}.parquet` | ridge team-dummy offensive efficiency (pts per 100 poss) relative to the as-of league mean, fitted on that season's games **strictly before** the game date, with a prior-season shrinkage prior. `scripts/build_own_ratings.py` | train-time median (≈0 ≡ league mean) | Higher = better offence. Leak-tested, change-form corr +0.054 |
| `own_def_c` | float64 | same | ridge team-dummy defensive efficiency relative to the as-of league mean | train-time median | **Positive = worse defence (allows more)**, the KenPom AdjD sign convention. Change-form corr −0.062 |
| `own_tempo_rel` | float64 | same | team tempo / as-of league mean tempo (KenPom AdjT convention) | train-time median (≈1.0) | Change-form corr +0.002 |
| `opp_own_off_c` | float64 | same | the opponent's `own_off_c`, same as-of date | train-time median | |
| `opp_own_def_c` | float64 | same | the opponent's `own_def_c` | train-time median | The defence this offence is facing |
| `opp_own_tempo_rel` | float64 | same | the opponent's `own_tempo_rel` | train-time median | |
| `own_kp_adj_o_c` | float64 | `data/processed/kenpom_snapshots.parquet` | `adj_o_c` from the latest snapshot **strictly before** the game date (`pd.merge_asof(..., allow_exact_matches=False)`, `cbb_sim.data.kenpom.as_of` semantics) | train-time median (≈0 ≡ that snapshot's league mean) | Centred column only. Raw `adj_o` is banned (CLAUDE.md: KenPom's league-mean AdjO drifted 100 → 109.3) |
| `own_kp_adj_d_c` | float64 | same | `adj_d_c`, same join | train-time median | Positive = worse defence |
| `own_kp_adj_t_rel` | float64 | same | `adj_t_rel` (= adj_t / snapshot mean adj_t), same join | train-time median | |
| `opp_kp_adj_o_c` | float64 | same | the opponent's `adj_o_c` | train-time median | |
| `opp_kp_adj_d_c` | float64 | same | the opponent's `adj_d_c` | train-time median | |
| `opp_kp_adj_t_rel` | float64 | same | the opponent's `adj_t_rel` | train-time median | |
| `site_home` | float64 {0,1} | `data/processed/games_universe.parquet` | 1 if this team is the home team **and** `neutral_site` is false | 0 | Neutral site is the reference level |
| `site_away` | float64 {0,1} | same | 1 if this team is the away team **and** `neutral_site` is false | 0 | Home/away/neutral is a first-class feature in every component model (CLAUDE.md modelling rule 2) |
| `days_since_start` | float64 | same | `game_date − min(game_date)` within the season, in days, over D-I non-truncated games | train-time median | 0–155. Carries within-season drift only, never a season level |

### 1b. Per game (symmetric). Used by the pace model only.

| Feature | Dtype | Source file | Computation | Fallback value | Notes |
|---|---|---|---|---|---|
| `pace_own_tempo_sum` | float64 | own ratings | `own_tempo_rel(home) + own_tempo_rel(away)` | train-time median | Sum, not two separate coefficients: one pace realisation per game with both teams scaled by it (CLAUDE.md modelling rule 3), and the own-ratings tempo model is itself additive with equal team weights. Also removes any dependence on which side is nominally "home" at a neutral site |
| `pace_own_off_sum` | float64 | own ratings | `own_off_c(home) + own_off_c(away)` | train-time median | |
| `pace_own_def_sum` | float64 | own ratings | `own_def_c(home) + own_def_c(away)` | train-time median | Slow defences shorten games |
| `pace_kp_t_sum` | float64 | KenPom snapshots | `adj_t_rel(home) + adj_t_rel(away)` | train-time median | |
| `pace_kp_o_sum` | float64 | KenPom snapshots | `adj_o_c(home) + adj_o_c(away)` | train-time median | |
| `pace_kp_d_sum` | float64 | KenPom snapshots | `adj_d_c(home) + adj_d_c(away)` | train-time median | |
| `neutral` | float64 {0,1} | games universe | `neutral_site` | 0 | The only site distinction a game-level quantity has |
| `days_since_start` | float64 | games universe | as above | train-time median | |

### 1c. Targets (not features)

| Target | Model | Source | Computation |
|---|---|---|---|
| `game_poss` | pace | hoopR `team_box` | `FGA − OREB + TOV + 0.44·FTA`, averaged over both teams (identical to `scripts/build_gate_reference.py`; includes overtime possessions) |
| `tpa`, `fg2a`, `fta`, `tov` | rates | hoopR `team_box` | counts, with offset `log(game_poss / 100)` so the coefficient scale is per 100 possessions. `fg2a = FGA − 3PA` |
| `tp_pct`, `fg2_pct`, `ft_pct` | pcts | hoopR `team_box` | two-column (makes, misses) Binomial endog — that IS the trials weighting. `fg2m = FGM − 3PM` |

### 1d. Fallback policy

Every model persists `medians`, the **train-time median of each of its own
features**, and `FittedModel.design()` fills NaN with it. For the centred
columns the median is ≈0, which is the league mean by construction — not an
invented constant. Missing rates in practice: 15 of 44,702 team-games
(2022–2025) have no KenPom snapshot before their date; own ratings are never
missing because a season's first date carries the prior-season carryover.

`cbb_sim.control.models.preflight` runs before a single draw and raises
`FeaturePreflightError` if any model feature is absent from the sim rows.

---

## 2. Feature sets tested

The three anchor bundles are the pre-registered L1 arms, run inside the Control
for free. Every bundle always includes the site and day-of-season terms.

| Set name | Included features | Rationale |
|---|---|---|
| `A_own` | own ridge ratings (own + opponent) + site + day | The compliant, self-contained arm: hoopR box scores only, no third-party feed, refreshable daily by us |
| `B_kp` | centred KenPom (own + opponent) + site + day | Last year's anchor, kept honest by using only centred columns and a strictly-before snapshot |
| `C_both` | union of the two | Tests whether the two ratings carry independent information |

Pace bundles use the corresponding `pace_*` sums (section 1b).

**Chosen on fold 2: `A_own`** — see `experiments.md` for the decision rule and
the seed noise floor that produced it.

---

## 3. Rejected features

| Feature | Why not |
|---|---|
| Raw KenPom `adj_o`, `adj_d`, `adj_t`, `rank_net` | CLAUDE.md bans raw rating levels: KenPom's league-mean AdjO drifted 100 → 109.3 across last year's data and inflated every model. Only the centred/relative columns are used |
| Own-ratings `league_off_mean` / `league_tempo_mean` (the as-of season level) | Pregame and leak-safe, and it WOULD close most of the −2.83 total bias, but it is not in the pre-registered feature list. Adding it after seeing the gate result is exactly the post-hoc move the standing rules ban. Recorded in `experiments.md` as the top follow-up for L2/L3 |
| `own_n_games` / `opp_own_n_games` | Kept in the feature table for diagnostics but out of every model: it is a deterministic function of the date and the team's schedule and would partly duplicate `days_since_start` |
| Season-of-play as a categorical | Same objection as the league mean, and it cannot be evaluated out of sample: the test season's level is unknown at train time by definition |
| Team tier (own-season point-differential tercile) | **Not leak-free** — a team's tier uses its full season including games after the one being described (`docs/tests/gate_reference_2026-09-10.md`). Grading-only, never joined onto a game |
| CBBD season ratings endpoints | End-of-season snapshots; banned as pregame features by `docs/SIM_GUARDRAILS.md` |
| Embedded ESPN market columns in hoopR pbp | Stripped from features by the data rules (CLAUDE.md) |
| pbp-derived shot location / rim-vs-jumper split | Coordinates cover only 6–26% of shots before 2024-25; the Control pools 2PA deliberately. L3 replaces this |

---

## 4. Provenance chain

```
data/raw/hoopr/{team_box,schedules}/*.parquet
        |
        +-> data/processed/games_universe.parquet          (cbb_sim.data.universe)
        |        is_d1_game & ~pbp_truncated & completed  -> 5,406-5,762 games/season
        |
        +-> data/processed/ratings/own_ratings_{season}.parquet
        |        (cbb_sim.ratings.own_ratings, scripts/build_own_ratings.py)
        |        one row per (season, as_of_date, team_id)
        |
data/processed/kenpom_snapshots.parquet                    (cbb_sim.data.kenpom)
        |
        v
cbb_sim.control.features.build_team_game_features(seasons)
        -> (team_game, game) feature + target tables
        -> cbb_sim.control.models.fit_all(...)  -> data/processed/models/control_engine/
        -> cbb_sim.control.simulate.prepare/simulate -> results/control/<fold>_<anchor>/
```

---

## 5. Leak evidence

`scripts/build_own_ratings.py` runs `cbb_sim.analysis.leak_test.run_leak_test`
on the finished as-of table and writes
`data/processed/ratings/own_ratings_leak_test.csv`. Pooled over 2022–2025
(43,255 team-games), change-form ("as-joined") correlation with own-game margin:

| column | as-joined | legit-update | level | verdict |
|---|---|---|---|---|
| `off_c` | +0.054 | +0.440 | +0.283 | pass |
| `def_c` | −0.062 | −0.437 | −0.268 | pass |
| `tempo_rel` | +0.002 | −0.003 | −0.006 | pass |

Gate is \|as-joined\| ≤ 0.15 with an honest baseline of 0.04–0.08; all three sit
inside that baseline. The large `legit-update` correlation is the positive
control: it is what the as-joined figure *would* read if the join were one game
late, and it confirms the ratings really do update on the game they just saw.
The harness also prints a level-form verdict for every column; per its own
docstring that verdict is only meaningful for STATIC columns, and none of these
are static (43,253 of 43,255 deltas are non-zero) — a level correlation of 0.28
for a rating is ordinary predictive power, not leakage.

KenPom's own leak status is inherited from `cbb_sim.data.kenpom`'s
strictly-before `as_of` semantics; `tests/test_control.py` verifies on real data
that a game played on a snapshot day receives the **previous** snapshot.
