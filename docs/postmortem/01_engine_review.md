# CBB-Monte Postmortem - Simulation Engine & Model Training

Code: `C:\Users\devuser\CBB-Monte` | Data/models/outputs: `C:\Users\devuser\CBB-Monte-storage`
Scope: the production sim (`run_sims_by_date.py`), its feature builders, all trainers, every experimental branch.

Where I state an empirical number I computed it myself from the shipped artifacts
(`out/cbb-sims-2026/2026/days/*/games/*/summary.json` + `final.json`; n = 5,229 graded games with closing
lines) or by re-running the production prediction path on real `data/sim_input/2026/*_sim_rows.csv`.
Those are marked **[measured]**.

---

## 0. Executive verdict (read this first)

The engine shipped a **team-level box-score Monte Carlo** built on six independent GLMs. It was,
measurably, a net-losing model on every market it priced.

| **[measured]** over 5,229 graded 2025-26 games | Sim | Market | Actual |
|---|---|---|---|
| Mean game total | 158.8 | 148.9 | 149.3 |
| Mean home margin | +2.19 | +5.57 | +5.31 |
| Total: mean error | **+9.44** | -0.48 | - |
| Margin: mean error | **-3.12** | +0.26 | - |
| ATS record following the sim | **2668-2533 (51.3%)** | - | 52.4% needed at -110 |
| O/U record following the sim | **2538-2677 (48.7%)**, Over picked 95.9% of the time | - | |

Regressing the actual result on `[1, market_line, model_line]` **[measured]**:

```
MARGIN: const -0.14 (t=-0.66) | market  0.9647 (t=24.63) | model  0.0361 (t= 0.86)   R2=0.435  n=5229
TOTAL : const  7.00 (t= 1.96) | market  0.9884 (t=19.96) | model -0.0302 (t=-0.74)   R2=0.234  n=5229
```

**After orthogonalising against the closing line the model's edge has zero predictive content**
(beta=0.036, t=0.86 on margin; beta=-0.030, t=-0.74 on total). The two large level biases below are
therefore *not* the whole story - patching them flat still leaves ATS at 51.0% and O/U at 49.2%
**[measured]**. The clean sheet needs a different architecture, not a recalibration.

The two level biases have clean single-line causes:

1. **No home-court advantage exists anywhere in the pipeline.** `Site` is explicitly discarded at
   `merge_kp_and_enrich_gamelogs.py:21`, and no home flag is ever built
   (`build_training_file_multi.py:411-433`, `build_sim_rows_by_date.py:426-432`). The sim projects the
   home team 3.1 pts too low and picks the home side ATS only **16.6%** of the time **[measured]**.
2. **Raw KenPom levels feed every model with no league centering, and KenPom's scale drifted.**
   League-mean AdjO was 100-103 in 2022, 104.5 across the training window, and **109.3 in 2025-26**
   (from `2022_kenpom.csv` ... `data/kenpom/2026/*.csv`) **[measured]**. `team_or`/`opp_dr` enter every
   GLM as raw levels with positive coefficients, so all six sub-models inflated together.

---

## 1. ARCHITECTURE

### 1.1 Unit of simulation

**Aggregate team-level box-score draws.** Not possession-by-possession, not shot-by-shot, not
player-level. Each team's whole-game shot profile is drawn once per simulation from six marginal
distributions, then converted to points arithmetically. The core is 45 lines:
`run_sims_by_date.py:1240-1325`.

### 1.2 Data flow

```
sports-reference team gamelogs                       KenPom weekly snapshots
  scrape_team_gamelogs_all.py                          pull_kenpom_table.py
        |                                                     |
        v                                                     |
  build_pregame_averages.py                                   |
    per-team CSV; x.shift(1) then expanding mean/std          |
    and rolling-5 mean/std   (build_pregame_averages.py:93-101)
        |                                                     |
        |  TRAINING PATH                                      |
        |    merge_kp_and_enrich_gamelogs.py                  |
        |      joins KP as-of prior Monday       (:213-341)   |
        |      DROPS Site / OT / Tm / Opp        (:21)        |
        |    build_training_file_multi.py                     |
        |      2 rows per game (teama, teamb)                 |
        |      ratings_features()                (:235-266)   |
        |      early-season blend                (:177-233)   |
        |      targets y_{3pa,2pa,fta,3p_pct,2p_pct,ft_pct}   |
        |    -> all_training_stats_rows_2022_2025.csv         |
        |       45,436 rows, 104 cols, 2021-11-09..2025-04-07 |
        |    train_models_multi.py  (chrono 60/20/20)         |
        |    -> models/all_glm_y_*.pkl + *.medians.json       |
        |                                                     |
        v  SERVING PATH                                       v
            build_sim_rows_by_date.py
              pick KP snapshot on/before date       (:94-100)
              ratings_features()                    (:251-276)
              exp_possessions = mean(AdjT_A, AdjT_B)(:275)
              early-season blend                    (:201-248)
            -> data/sim_input/{season}/{date}_sim_rows.csv
                        |
                        v
            run_sims_by_date.py::_simulate_one_game (:1240)
              12 GLM point predictions              (:1250-1262)
              6 dispersion lookups from the row     (:1265-1277)
              draw -> round -> binomial -> points   (:1286-1325)
                        |
                        v
            sims.parquet / summary.json / sims_compact.json / index.json
                        |
            hf_publish.py -> HuggingFace dataset -> website
            compute_daily_results.py -> daily_results.json -> calibration/profit scripts
```

Entry point: `daily_run.bat` -> `daily_run.py` (data refresh only) -> `hf_publish.py:210-237` runs
`run_sims_by_date.py --nsims 10000 --processes auto`. `daily_run.py:49-60` **patches the string constant
`RUN_DATE` inside `hf_publish.py` on disk** and reverts it afterwards - self-modifying source as a
parameter-passing mechanism.

### 1.3 The generative model, exactly

Per team side, independently (`run_sims_by_date.py:1279-1325`):

```python
# attempts, for each of 3PA / 2PA / FTA
mu   = glm.predict(X, offset=log(exp_possessions))          # :742
sd   = mean(team_ytd_std, team_roll5_std,
            opp_allowed_ytd_std, opp_allowed_roll5_std)     # :696-703
A_xa = round(clip(Normal(mu, sd), 0, None)).astype(int)     # :1280-1288

# percentages, for each of 3P% / 2P% / FT%
p_mu = glm.predict(X)                                       # :748  (logit-link Binomial)
p_sd = clip(mean(same four *_std columns), 1e-3, 0.20)      # :705-713
p    = Beta(matched to mean p_mu and var p_sd**2)           # :1294-1301

# makes and points
A_xm  = Binomial(A_xa, p)                                   # :1319-1321
A_pts = 3*A_3pm + 2*A_2pm + A_ftm                           # :1324
```

**Stochastic:** three Normal attempt draws, three Beta percentage draws, three Binomial make draws per
team - 12 random vectors of length `nsims` per game.

**Deterministic:** every GLM mean, every dispersion (read straight off the feature row), the pace
(`exp_possessions` is a fixed number), the points arithmetic, the market comparison.

**Not modelled at all:** possessions as a random quantity, turnovers, rebounds, fouls, game state,
overtime, correlation between the two teams, correlation among a team's own six stats, garbage time,
rest/travel/altitude, home court.

### 1.4 Structural consequences of this design

- **The two teams are statistically independent.** Nothing links A's draw to B's. The sim's own output
  proves it: implied SD of margin = **24.61** and implied SD of total = **24.71** - identical, the
  signature of `Var(A-B) = Var(A+B) = Var(A)+Var(B)` **[measured]**. Realised SD of the margin residual
  is **12.36**, of the total residual **17.91**. The sim is 2x too wide on margin (4x the variance) and
  1.4x too wide on totals.
- **Nothing constrains the shot profile to the possession count.** Each of 3PA/2PA/FTA is an independent
  GLM with a `log(exp_possessions)` offset; their sum is unconstrained. On real 2026 rows the models
  predict 24.2 3PA + 35.6 2PA + 21.5 FTA **[measured]**, implying ~72 possessions while
  `exp_possessions` says 68.9. The offset is decorative.
- **Shooting variance is double-counted.** `_pct_sd_from_row` (`:705-713`) uses the *observed
  game-to-game SD of the team's shooting percentage*, ~0.10 for 3P%. The pure binomial SD of 22 threes
  at 34% is `sqrt(.34*.66/22) = 0.101`. Essentially all of that observed SD *is* binomial noise. The sim
  injects it as true-talent variation and then adds the binomial layer again at `:1319`. Measured
  sim-time SDs on real rows: 3P% 0.0998, 2P% 0.0981, FT% 0.1140 **[measured]**.
- **Ties are silently discarded.** `A_win_prob = (margin > 0).mean()` (`:1390`). Mean sim tie rate 1.46%
  **[measured]**, so `A_win_prob + B_win_prob = 0.985`, and `_american_from_prob` (`:752-760`) emits
  fair odds that under-price both sides by ~1.5%. Downstream +EV filters consume these directly
  (`compute_daily_results.py:251-264`). No overtime model exists.
- **Sim RNG is unseeded** (`:1216` `np.random.default_rng()`), so runs are not reproducible.
- **`priors.json` is a stub.** `:1367-1378` writes `"targets": {}` with the comment
  "Paste the exact targets population block from your previous file here". Every shipped `priors.json`
  is 176 bytes of nothing, yet `manifest.json` advertises `"priors_file": "priors.json"`.
- **Models are re-unpickled per game.** `_load_model_and_medians` (`:644`) is called inside
  `_predict_attempts_mu` / `_predict_pct_mu` (`:737`, `:746`) - **12 `sm.load()` calls of 6.5-6.9 MB
  pickles per game**, ~80 MB of I/O per matchup. A cache exists (`_MEDIANS_CACHE`, `_get_medians`,
  `:30-54`) but is **never called**. `rank_round_robin.py:636-655` has the correct `_init_worker`
  caching; it was never back-ported.
- **Output volume:** 15 GB for one season. Per game: `sims.parquet` 790 KB + `sims_compact.json` 1.87 MB
  (10,000 raw draws x 13 arrays, JSON, shipped to the browser).

---

## 2. SUB-MODELS

### 2.1 Shipped in production (`models/all_glm_*.pkl`, loaded at `run_sims_by_date.py:446-453`)

| Target | Family / link | Offset | Features | n_obs | Trainer | Use in sim |
|---|---|---|---|---|---|---|
| `y_3pa` | Poisson / log | `log(exp_possessions)` | team_3pa {ytd,roll5}x{avg,std}, opp_allowed_3pa x4, `team_or`, `opp_dr`, `adjt_mean` (12) | 25,929 | `train_models_multi.py:389-420` | mean of Normal draw |
| `y_2pa` | Poisson / log | same | analogous 2pa block (12) | 25,929 | `:389-420` | same |
| `y_fta` | NegativeBinomial (alpha grid 0.1/0.3/1.0/3.0, picked on **val** deviance) / log | same | analogous fta block (12) | 25,929 | `:423-443`, `:299-308` | same |
| `y_3p_pct` | Binomial / logit, **unweighted** | none | team_3ppct x4, opp_allowed_3ppct x4, `team_or`, `opp_dr` (10) | 27,183 | `:448-484`, fit at `:348` | mean of Beta draw |
| `y_2p_pct` | Binomial / logit, **unweighted** | none | analogous (10) | 27,183 | same | same |
| `y_ft_pct` | Binomial / logit, **unweighted** | none | analogous (10) | 27,180 | same | same |

Deployed coefficients read from the pickles, prediction at the training medians **[measured]**:

```
y_3pa    Poisson/Log    pred@median 21.30   team_or +0.0016  opp_dr +0.0002  adjt_mean -0.0079
y_2pa    Poisson/Log    pred@median 35.85   team_or +0.0004  opp_dr +0.0005  adjt_mean -0.0114
y_fta    NegBin/Log     pred@median 17.66   team_or +0.0041  opp_dr +0.0041  adjt_mean -0.0022
y_3p_pct Binomial/Logit pred@median 0.3365  team_3ppct_ytd +0.543  opp_allowed_3ppct_ytd +0.138
y_2p_pct Binomial/Logit pred@median 0.5030  team_2ppct_ytd +1.558  opp_allowed_2ppct_ytd +1.155
y_ft_pct Binomial/Logit pred@median 0.7135  team_ftpct_ytd +1.456  opp_allowed_ftpct_ytd +0.286
```

Note `y_2p_pct`: coefficients on the team's own 2P% (1.558) **and** the opponent's allowed 2P% (1.155)
are both large and positive, summing to 2.71 on the logit scale, applied to *raw levels*. No centering,
no zero-sum constraint. When the league-wide level of both features rises, the prediction runs away.
That is exactly what happened.

### 2.2 Trained but never used

| Files | What | Why dead |
|---|---|---|
| `models/all_glm_y_*_tw.pkl` (+ medians) | Identical to production except `freq_weights=n_train` on the Binomial fits (`train_models_multi_pct_trials_weighted.py:351`) - the *statistically correct* version | `MODEL_FILES` never references `_tw`. The `_tw` medians are byte-identical to the non-`_tw` ones, and the `_tw` *attempt* models are re-runs of the same unweighted fit - pure duplication |
| `models/2025_glm_y_*.pkl` | Single-season 2025 version (`train_models_2025.py`, otherwise a near-clone of `train_models_multi.py`) | Superseded; **no `.medians.json` companions**, so `_load_model_and_medians` (`:649-654`) would hard-fail on them |
| `models/improved/glm_y_*.pkl` | NegBin for all three attempt targets + weighted Binomial pcts on a 30-feature interaction set (`train_models_improved.py`) | Only `run_sims_improved.py` reads them; never wired into `daily_run`/`hf_publish` |
| `models/pace_glm_exp_possessions.pkl` | Gaussian/identity GLM predicting `pace_target = 3PA + 2PA + 0.44*FTA` (`train_pace_model_multi.py:311-315, 223-288`) | Only `run_sims_by_date_pace_experiment.py:412, 1760-1786`, writing to `out/experiments/` |
| `models/ppp_ridge_model.pkl` | StandardScaler + Ridge(alpha=5) on ~100 features predicting `y_ppp = y_pts / exp_possessions` (`train_models_ppp_poss.py:96-107, 192-198`) | No sim loads it. Its "PPP" is points divided by *predicted* pace, so it is a reparametrisation, not true PPP |
| `data/models/boosted/*.pkl` (16 files) | XGBoost for attempts; XGB + LightGBM + RandomForest simple-average ensemble for percentages (`final_boosted_trainer.py:84-150`) | Only `boosted_simulator_backtest.py` reads them, into `out/boosted_sims/` |

### 2.3 GLM vs boosted - was a comparison ever done?

**Partially, and never against the thing that shipped.**

- `percent_model_comparison.py` is a real 8-way shootout for the **percentage** targets only:
  XGBoost_Tuned, LightGBM, RandomForest, Ridge, Ensemble_Avg, XGBoost+Isotonic, LightGBM+Isotonic, and a
  regularised OLS on 6 key features (`:68-171`), scored MAE/R2 on val 2024 / test 2025 (`:177-189`).
  **The production statsmodels Binomial GLM is not one of the eight contenders.**
- No equivalent shootout exists for the three **attempt** models. `boosted_training.py` trains XGB for
  all six targets but **never saves a model** (`:72-86` trains, `:121-122` calls `plt.show()`, script
  ends) - a notebook in .py clothing.
- `compare_models.py` compares the *original* sim tree against the *improved* sim tree end-to-end on
  betting outcomes (`:17-19`, `:189-433`).
- `compare_sims.py` compares original vs boosted sim outputs on CLV/profit (`:1-4`, window
  2025-11-03 to 2025-12-18).

So the boosted branch was explored, produced saved artifacts, and was never adopted or benchmarked
against production.

---

## 3. TEAM STRENGTH / OPPONENT ADJUSTMENT

### 3.1 Where quality enters

Entirely via **KenPom**, through four raw numbers per team, at `build_sim_rows_by_date.py:251-276` /
`build_training_file_multi.py:235-266`:

```python
team_or, team_dr, team_net, team_adjt      # this team's KP AdjO / AdjD / NetRtg / AdjT
opp_or,  opp_dr,  opp_net,  opp_adjt       # opponent's
net_diff  = team_net - opp_net
or_vs_dr  = team_or  - opp_dr
dr_vs_or  = team_dr  - opp_or
adjt_mean = mean(team_adjt, opp_adjt)
adjt_diff = team_adjt - opp_adjt
exp_possessions = mean(team_adjt, opp_adjt)         # :275 - identical to adjt_mean
```

Of these, **the shipped GLMs use only three**: `team_or`, `opp_dr`, `adjt_mean`
(`train_models_multi.py:132-166`; the pct models drop `adjt_mean`). `net_diff`, `or_vs_dr`, `dr_vs_or`,
`adjt_diff`, `team_dr`, `opp_or`, `team_net`, `opp_net` are computed and then never used by production.
**The difference features - the ones that actually encode a matchup - are all discarded.**

There are no own-built ratings. There is no opponent adjustment of the box-score features: the
`opp_allowed_*` block is the opponent's raw allowed counts/percentages from their own gamelog
(`build_pregame_averages.py:63-114`, reading the `Opponent *` columns), un-adjusted for who they played
and un-normalised for pace.

### 3.2 The KenPom scale drift - the single largest technical defect

KenPom recomputes ratings daily; the league-average AdjO/AdjD is **not stable**, in-season or across
seasons. From the raw snapshot files **[measured]**:

| Season | first snapshot (ORtg, AdjT) | last snapshot (ORtg, AdjT) |
|---|---|---|
| 2022 | 100.0, 72.2 | 103.05, 67.03 |
| 2023 | 99.1, 71.5 | 105.22, 67.25 |
| 2024 | 99.0, 71.0 | 106.12, 67.44 |
| 2025 | 101.5, 71.0 | 107.18, 67.43 |
| **2026** | **104.5, 71.0** (2025-10-13) | **109.31, 67.39** |

Two separate problems:

1. **Within-season drift.** In early November the sim uses `exp_possessions ~ 71`; by February the same
   two teams get `~ 67`. Since `exp_possessions` is the multiplicative offset on every attempt model
   (`run_sims_by_date.py:742`), the *same* team is projected ~6% more attempts in November than in
   February for identical form. Real tempo does not move like that; it is a rating-scale artifact.
2. **Cross-era drift.** Training-file mean `team_or` = 104.53; 2026 serving mean = **107.97**;
   `opp_dr` 103.61 -> 107.76 **[measured]**. Coefficients fit on raw levels around 104 are applied to
   levels around 108-110. For `y_fta` (`team_or` +0.0041, `opp_dr` +0.0041) that alone is
   `exp(0.0041*3.44 + 0.0041*4.15) ~ +3.2%`; for `y_2p_pct` it is +0.078 on the logit, about **+1.9 pp**
   of 2P%.

**No feature is ever expressed relative to its snapshot's league mean.** This is the root cause to fix
first in the clean sheet.

### 3.3 Home court

**Absent.** In detail:

- `merge_kp_and_enrich_gamelogs.py:21` -
  `EXCLUDE_COLS = {"Rk","Gtm","Date","Site","Opp","Type","Rslt","Tm","Opp","OT"}`.
  `Site` is the `@` / blank / `N` home-away-neutral field from sports-reference. It is dropped before the
  training file is built. So are `OT` and the actual score (`Tm`).
- Verified: `2025_games_with_kp_and_gamelogs.csv` has 62 columns and **zero** matching
  `site|home|away|ha|neutral|loc` **[measured]**.
- `build_training_file_multi.py:411-433` - the schema has `side` in {`teama`,`teamb`} only.
- `build_sim_rows_by_date.py:6-7` - the docstring lists `team1_ha`/`team2_ha` as available input columns;
  **the code never reads them** (the only grep hits are the docstring).
- `run_sims_by_date.py` mentions "home" only in odds matching (`:289`, `:1497-1499`).
- `2026_massey_schedule_enriched.csv` has `team1_ha`/`team2_ha` populated: H/A 3247, A/H 1796, N/N 709
  **[measured]** - the data was sitting right there.
- `side_encoded` in the boosted branch (`boost_feature_builder.py:25`, `final_boosted_trainer.py:59`
  `home_vs_dr = side_encoded * opp_dr`) is *named* as if it were home/away but is really "is this
  schedule row's team2", which is home only ~64% of the time.

Measured cost: mean home margin +2.19 vs actual +5.31 and market +5.57; the sim picks the home side ATS
**16.6%** of the time **[measured]**.

### 3.4 Pace / tempo

`exp_possessions = mean(AdjT_team, AdjT_opp)` (`build_sim_rows_by_date.py:275`) - a plain arithmetic
mean, not the standard `AdjT_A * AdjT_B / league_AdjT` form, and not corrected for the drifting league
baseline (3.2). It is:

- identical for both sides of a game (correct sign of intent);
- used only as a `log()` offset (`run_sims_by_date.py:739-742`) - never itself drawn as a random quantity;
- inconsistent with the attempts the models predict (1.4).

The `run_sims_improved.py:106-114` variant makes it **worse**: `0.6*team_adjt + 0.4*opp_adjt` is
*asymmetric*, so team A and team B in the same game get different possession counts.

A real pace model exists (`train_pace_model_multi.py`, Gaussian GLM on
`pace_target = 3PA + 2PA + 0.44*FTA`) and `run_sims_by_date_pace_experiment.py:1760-1786` correctly
averages the two teams' predictions into one shared value - but it writes to `out/experiments/` and was
never promoted.

### 3.5 Recency weighting

Three mechanisms, all crude:

- **roll5 vs ytd**: both the 5-game rolling mean/std and the season-to-date mean/std are supplied as
  *separate features* and the GLM picks the weights. Fitted weights are lopsided - e.g. `y_3pa`:
  `team_3pa_ytd_avg` 0.0278 vs `team_3pa_roll5_avg` 0.0064, i.e. ~81% on season-to-date and ~19% on
  recent form. There is no explicit exponential decay anywhere.
- **Early-season blending** toward a prior: `w = 0 if game 1 else min(1,(g-1)/5)`, blending the current
  YTD value with `0.65*prev_season_final + 0.35*league_median`
  (`build_sim_rows_by_date.py:201-248`, `build_training_file_multi.py:177-233`; constants at `:44-48` /
  `:17-22`). Fully live from game 6 onward.
- **No season weighting in training.** All four seasons contribute equally; there is no recency weight
  on training rows at all, despite the 2022 and 2025 scoring environments differing materially.

Downstream, `pick_best_bets.py:36-37` applies a 21-day window with a 7-day half-life to *calibration*
statistics - recency weighting applied to the wrong object (see 5, 8).

---

## 4. PLAYER LEVEL

**None. The engine is 100% team-level.** No minutes model, no usage model, no rotation, no injury
handling, no player props.

The only player-shaped thing in the repo is `scrape_cbb_sr_boxscores_with_ids.py`, which advertises
`data/boxscores/YYYY-MM-DD.parquet` (player-level box scores, `:7`). That directory exists and is
**empty** (0 files) **[measured]**, and no training or sim script references `boxscores`. Players appear
only as free text in the LLM commentary path (`hybrid_post.py:203` asks Grok about "key injuries /
suspensions / lineup changes"), which never touches a number.

Consequence: a starter being out has literally no effect on the projection.

---

## 5. MAGIC NUMBERS / HAND-TUNING INVENTORY

### 5.1 Production sim (`run_sims_by_date.py`) - surprisingly few, and that is the problem

| file:line | Value / expression | What it patches | Best inference of why |
|---|---|---|---|
| `:713` | `np.clip(sd, 1e-3, 0.20)` on percentage SD | Caps shooting-% dispersion at 20 pp | Guard against a team with 1-2 games whose `*_std` is garbage; the 1e-3 lower bound also hides zero-std rows |
| `:1280` | `max(sd, 1e-6)` | Prevents `rng.normal(mu, 0)` | Same missing-data guard |
| `:1281` | `np.clip(v, 0, None)` on attempts | Truncates negative attempt draws | The Normal draw is unbounded; sd~6 on a mean of ~18 FTA makes negatives reachable. Truncation biases the mean slightly up |
| `:1286-1291` | `np.round(...).astype(int)` | Makes attempts integers | Required by `rng.binomial`; introduces a half-attempt rounding artifact |
| `:1295` | `np.clip(mu, 1e-6, 1-1e-6)` | Keeps the Beta parameterisation defined | Logit GLM can emit exactly 0/1 |
| `:1298` | `var = min(var, mu*(1-mu) - 1e-12)` | Forces the Beta to exist | The supplied `sd` frequently exceeds the maximum possible variance for that mean - **the code knows the dispersion source is too large** and silently truncates it |
| `:1229` | `lo, hi = lo-0.5, hi+0.5` in `_hist_counts` | Degenerate-histogram guard | Cosmetic |
| `:1311-1316` | Six commented-out `draws(..., clip01=True)` lines, superseded by `draw_pct_beta` at `:1294` with the comment "Percentages (Alert to testing)" | The old truncated-Normal percentage draw | Documents an untracked switch from clipped-Normal to Beta; left in place |
| `:1756` | `--nsims` default 1000 | - | Production overrides to 10000 (`hf_publish.py:30`) |
| `:1762` | `--model_version` default `"cbb-2026-10-15"` | Stamped into every manifest | A hard-coded string, not derived from the model files (`_sha()` at `:637` exists but is never called) |
| `:1873-1874` | `sys.argv += ["--season","2026","--date","2025-11-04", ...]` when run bare | Convenience default | Silently simulates the wrong date if invoked with no args |

**Conspicuously absent: there is no post-hoc calibration of any kind on the sim output.** No shrink
toward the market, no bias offset, no variance rescale, no HCA constant. Given the measured +9.4 total
and -3.1 margin bias, the *absence* of even a crude patch is itself the finding - the biases were never
diagnosed at the projection level, only at the betting-outcome level.

### 5.2 Feature builders (shared by training and serving)

| file:line | Value | What it patches | Inference |
|---|---|---|---|
| `build_sim_rows_by_date.py:46-47`, `build_training_file_multi.py:19-20`, `rank_round_robin.py:79-80` | `ALPHA_PRIOR_TEAM = ALPHA_PRIOR_OPP = 0.65` | Weight on last season's final value vs league median in the early-season prior | Pure guess; no shrinkage estimated from data. Applied identically to a 3PA count and an FT% |
| `:45` / `:18` | `BLEND_GAMES = 5` | Blend fully expires after game 6 | Guess; a real prior would decay by *attempts*, not games, and would never fully expire for FT% |
| `:44` | `DATE_TOL_DAYS = 1` | Pregame-row lookup window | Papers over date mismatches between the Massey schedule and SR gamelogs |
| `:48` / `:21` | `TREAT_ZERO_AS_MISSING = True` | Treats a real 0 as missing | A team that genuinely attempted 0 FT in a game gets its prior substituted |
| `build_sim_rows_by_date.py:296-304` | `pick_with_fallback` returns `same_day.iloc[0]` or the nearest +/-1-day row when the opponent slug does not match | Name-matching failures | **Silently attaches another game's pregame block.** No counter, no warning |
| `build_pregame_averages.py:28, 103-105` | `FILL_STD_ZERO = True` -> `std.fillna(0.0)` | Games 1-2 have no std | A 0 std then trips `TREAT_ZERO_AS_MISSING` upstream - two patches interacting |
| `train_pace_model_multi.py:314` | `0.44 * FTA` in the pace target | Possession formula | NCAA convention is 0.475; minor, and this model is unused |

### 5.3 `build_improved_training_file.py` / `run_sims_improved.py` (the "improved" branch)

This branch is where the hand-tuning really lives.

| file:line | Value | What it patches | Inference |
|---|---|---|---|
| `build_improved_training_file.py:41-42`, `run_sims_improved.py:75-76` | `ALPHA_START=0.65`, `ALPHA_DECAY=0.15` -> `alpha = 0.65*(1-0.15*(g-1))`, clipped to `[0.1, 0.65]` | "Dynamic" early-season blend | Linear decay chosen by eye |
| `:45-46` / `:78-79` | `PACE_WEIGHT_TEAM=0.6`, `PACE_WEIGHT_OPP=0.4` | "Improved" pace | **Asymmetric - the two teams in one game get different possession counts.** Physically impossible |
| `:145` / `:109` | `pace_adjust = net_diff * 0.01` | Adds +/-0.4 possessions for a +/-40 net-rating gap | Numerically a no-op dressed as an adjustment |
| `:146` / `:110` | `np.clip(pace_base + pace_adjust, 50, 90)` | Bounds pace | Hides bad AdjT values instead of fixing them |
| `:49` / `:81` | `OUTLIER_CLIP_PCT = [0.01, 0.99]` | Winsorises every avg feature | In training the percentiles come from the **whole 4-season dataset** (`:97-113`); at serve time from **that single day's ~60-100 rows** (`run_sims_improved.py:83-96`). Same name, different operation - a train/serve skew |
| `:219` / `:164` | `np.clip(ratio, 0.5, 2.0)` on volume/permission ratios | Divide-by-near-zero | Also destroys the tails, which is where matchup edges live |
| `:250-251` / `:192-193` | `adjustment = (ortg_dr_diff*0.1 + pace_factor*0.2) * base_expect * 0.1`; `clip(base+adj, 0, 100)` | `pace_adj_{shot}_expect` | Three unmotivated constants; a made-up quantity fed to the model as a feature |
| `:261` / `:201` | `def_{X}_difficulty = df[col2].fillna(0.3)` | - | **The feature is a verbatim copy of `opp_allowed_{X}pct_ytd_avg`**, already in the feature list. Perfectly collinear. The `0.3` fill is 3P%-shaped and applied to 2P% and FT% too |
| `:270-273` / `:210-213` | `np.clip(std_ratio, 0.1, 10.0)`, default `0.5` | Variance ratio | The 0.5 default is outside the typical range - a sentinel masquerading as a value |
| `:320-323` / `:252-255` | `np.clip((opp_or - team_dr)/adjt_mean, -0.5, 0.5)` | `opp_strength_factor` | The clip bites for essentially no real matchup |
| `:249` / `:191` | `.fillna(20)` on the base attempt expectation | Missing volume | 20 is 3PA-shaped, applied to 2PA (~36) and FTA (~18) alike |
| `run_sims_improved.py:339-340` | `corr = -0.2`; `cov = [[0.05, -0.01],[-0.01, 0.05]]` for the two teams' shooting % | Correlates A and B percentages | **`var = 0.05` implies SD = 0.224** - a 22-pp SD on 3P%/2P%/FT%, identical for all three, ignoring the per-team dispersion the row already carries. Then clipped to [0,1] at `:346-352`. The negative sign has no stated justification |
| `run_sims_improved.py:325-329` | `shared_poss = Poisson(mean); adj = shared_poss / team_pace` | Shared pace draw | **Genuinely the right idea** (see 7), buried in this branch |

### 5.4 Boosted branch

| file:line | Value | What it patches | Inference |
|---|---|---|---|
| `boost_feature_builder.py:29` | `df.fillna(0)` on everything | Missing pregame data | Destroys missingness; a 0 FT% is indistinguishable from "no data" |
| `boost_feature_builder.py:45, 83` and ~40 sites | `epsilon = 1e-6` in ratio denominators | Divide-by-zero | Combined with `fillna(0)` above, a 0 denominator yields ~1e6 fed to a tree |
| `boost_feature_builder.py:89-90`, `final_boosted_trainer.py:51` | `weight_roll5 = min(game_num/10, 1)` | Hand-built recency blend | 10 games is a guess; competes with the model's own ability to weight roll5 vs ytd |
| `boosted_perc_training.py:165-178`, `quick_pct_eval.py:80-84` | `blended = 0.4*model + 0.4*opp_allowed_pct + 0.2*league_mean` | Post-hoc "blending with priors" | **A 60% shrink of the model toward two priors** - an admission the model's spread was untrustworthy. Weights never optimised |
| `boosted_perc_training.py:38-42` | `league_means` computed from the **full dataset including test** | Blend anchor | Leak (see 6) |
| `quick_pct_eval.py:16-20` | Those same league means hard-coded to 16 decimal places | Reproducing a prior run | Fossilised leaked constants |
| `boosted_perc_training.py:181-185` | `IsotonicRegression(out_of_bounds='clip')` fit on **train** predictions | "Calibration" | Isotonic on in-sample train predictions is near-degenerate and overstates its own benefit |
| `final_boosted_trainer.py:74-78` | `pct_xgb_params` hard-coded per target (`max_depth` 3/5/5, `min_child_weight` 20/5/10, `reg_lambda` 10/5/10 ...), comment "from shootout winner" | Frozen hyperparameters | Copy-pasted from a `GridSearchCV` run using `cv=3` (`boosted_perc_training.py:145`) - **random folds over pooled 2022+2023 rows, i.e. temporally leaky tuning** |
| `boosted_simulator_backtest.py:37-41` | `sd_dict = {3P%: 0.105, 2P%: 0.09, FT%: 0.127}` | The entire variance model | Three constants replace the whole dispersion layer. Attempts are **not** randomised at all (`:238-239`), so team scores are `attempts x drawn_pct` - continuous, no binomial, no volume variance |
| `boosted_simulator_backtest.py:193` | `df[feat] = 0` for any feature missing at serve time | Feature-name drift between builder and server | Silently zeroes real inputs |
| `boosted_simulator_backtest.py:281-282` | `start_utc = date+1 at 00:00:00Z` | Fabricated tip time | Written into an output that looks like production |

### 5.5 Downstream selection (post-sim, but part of the tuning story)

| file:line | Value | Inference |
|---|---|---|
| `compute_daily_results.py:26-27` | `SPREAD_LOSS_RISK = TOTAL_LOSS_RISK = 1.1` | Assumes -110 everywhere rather than reading prices |
| `compare_models.py:21-22` | `VIG_PROB = 110/210`; `ODDS_DEFAULT = -110` | Same |
| `pick_best_bets.py:36-37` | `WINDOW_DAYS = 21`, `HALF_LIFE_DAYS = 7` | Recency-weighted calibration window |
| `pick_best_bets.py:41` | `MIN_EV_UNITS = 0.02` | EV floor |
| `pick_best_bets.py:42` | `BRIER_MULT_MAX = 1.05` | **Only bet buckets where the model's recent Brier is <= 1.05x its overall Brier** |
| `pick_best_bets.py:43-44` | `MIN_BUCKET_SAMPLES = 10`, `MIN_BUCKET_ROI = 0.0` | Bucket trust gates at **n >= 10** |
| `pick_best_bets.py:45-46` | `ALPHA_ROI_BOOST = 0.50`, `ALPHA_CALIB_BOOST = 0.20`, i.e. `score *= (1 + 0.5*max(0,ROI)) * (1 + 0.2*max(0, overall_brier/bucket_brier - 1))` | **Explicitly up-weights buckets that recently made money.** With a 21-day window and n>=10 buckets this is momentum-chasing on pure noise |
| `pick_best_bets.py:50` | `PER_MARKET_LIMITS = "ml:6,spread:10"` | Daily bet caps |
| `compute_calibration_multi_days.py:25-48` | 10 ML price buckets, 6 spread buckets | Bucket edges chosen by eye |

The whole `pick_best_bets.py` gating layer is a *symptom*: it exists because the sim's probabilities were
known to be unreliable, so a second, entirely heuristic layer was bolted on to decide which of the sim's
probabilities to believe today. It never fixes the projection.

---

## 6. VALIDATION AND LEAKAGE

### 6.1 What validation existed - more than you would expect

- **Chronological 60/20/20 split** in every GLM trainer (`train_models_multi.py:74-83, 382`;
  `train_pace_model_multi.py:70-88`). Actual boundaries **[measured]**: train `2021-11-09 -> 2024-01-07`,
  val `2024-01-07 -> 2024-12-01`, test `2024-12-01 -> 2025-04-07`.
- **Season-based splits** in the boosted branch: train 2022+2023, val 2024, test 2025
  (`boosted_training.py:36-42`, `final_boosted_trainer.py:36-37`, `percent_model_comparison.py:27-29`).
- **Game-level chronological split** in `train_models_ppp_poss.py:181-189` - splits by `game_id` so both
  sides of a game land in the same fold. The cleanest split in the repo.
- **Live daily grading**: `compute_daily_results.py` writes per-game W/L/P + profit + EV against
  `final.json` (SBR closing lines and scores).
- **Probability calibration**: `compute_calibration_multi_days.py` computes Brier and log-loss overall and
  bucketed by ML price and spread size (`:137-263`) -> `multi_day_calibration.json`.
- **Profit/ROI analysis**: `analyze_profit_range.py` -> `2025-11-19_2025-12-16.json`.
- **CLV**: `compute_clv_by_date.py`.
- **A/B of sim variants**: `compare_models.py` (original vs improved), `compare_sims.py` (original vs
  boosted).

### 6.2 What the validation actually said (from the checked-in artifacts)

`multi_day_calibration.json`, window 2025-11-19 to 2025-12-16:

```
moneyline  n=898   Brier 0.1951  logloss 0.5751
spread     n=975   Brier 0.2507  logloss 0.6951      <- coin flip is 0.2500 / 0.6931
over       n=972   Brier 0.2930  logloss 0.7913      <- much worse than a coin flip
under      n= 15   Brier 0.2597  logloss 0.7136      <- only 15 Under picks in 987 games
all        n=2860  Brier 0.2477  logloss 0.6902
```

`2025-11-19_2025-12-16.json`, same window:

```
spread.all       533-440-1   54.8%   ROI  +4.9%
total.all        478-508-1   48.5%   ROI  -7.5%
ml.all           648-250     72.2%   ROI  -0.02%
ml.evchase.dog   173-571     23.3%   ROI -18.6%
```

The `over: 972 / under: 15` split is a 98.5% one-sided pick rate and should have been read as a
level-bias alarm on day one. It was instead absorbed into the bucket-gating heuristics in
`pick_best_bets.py`.

Note also that the +4.9% spread ROI in that four-week window was not real: **over the full 5,229-game
season the same pick rule went 2668-2533 = 51.3%, a loss at -110 [measured]**. `pick_best_bets.py` was
then tuned on exactly that lucky window.

### 6.3 Leakage - clean paths (credit where due)

- **Pregame features are correctly lagged.** `build_pregame_averages.py:93` does `pre_x = x.shift(1)`
  before the expanding and rolling windows, so row *i* only ever sees games `< i`. This is right, and it
  is the best-engineered piece in the repo.
- **KenPom is joined as-of the prior Monday**, with a *backward* `merge_asof` fallback
  (`merge_kp_and_enrich_gamelogs.py:213-341`). No end-of-season ratings on early-season games. Serving
  uses `pick_snapshot_on_or_before` (`build_sim_rows_by_date.py:94-100`). Correct.
- **Feature medians are computed on train only** and persisted next to the model
  (`train_models_multi.py:91-95, 401`), then reused at serve time (`run_sims_by_date.py:679`). Correct.
- **Early-season priors come from the *previous* season plus previous-season league medians**
  (`build_training_file_multi.py:135-157, 377-382`). Correct.
- **The PPP trainer splits by game, not by row** (`train_models_ppp_poss.py:181-189`). Correct.

### 6.4 Leakage and near-leakage - actual problems

1. **The production models were never refit on the val/test data, and were badly stale at serve time.**
   `train_models_multi.py` saves the model fit on the *train slice only* (`:397-401`, `:427-430`,
   `:459-463`) and never refits on the full history. Deployed `nobs` = 25,929 / 27,183 of 45,436 rows
   **[measured]**; the newest training example is **2024-01-07**. Those models priced the entire 2025-26
   season, two scoring environments later. Not leakage, but the same class of error: the deployed
   artifact is not the artifact that was validated, and 40% of the labelled data was discarded.
2. **`build_improved_training_file.py` has genuine forward-looking leakage in three places:**
   - `:97-113` outlier percentiles computed on the **whole 4-season dataset**, then applied to
     train/val/test.
   - `:156-208` `dynamic_early_season_blending` uses `df.groupby(season).median()` (`:178`) as the
     *prior* for games 1-5 - **a team's first game is blended toward that same season's full-season
     median**, which is not knowable in November.
   - `:278-298` imputation uses same-season medians over the full season (`:285`); `:306`
     `pd.qcut(net_diff, 4)` uses full-dataset quantiles.
3. **`run_sims_improved.py` recomputes all of those statistics from the current day's ~60-100 rows**
   (`:83-96`, `:130`, `:222`, `:238`). Same function names, completely different distributions from
   training. Severe train/serve skew even setting leakage aside.
4. **`boosted_perc_training.py:38-42`** computes `league_means` on the full dataframe (train+val+test) and
   blends predictions toward it at `:177`. `quick_pct_eval.py:16-20` hard-codes those leaked values.
5. **`boosted_perc_training.py:145`** uses `GridSearchCV(..., cv=3)` - random, non-temporal folds over
   pooled 2022+2023 rows. Both sides of a game and adjacent games of the same team land in different
   folds. The winning hyperparameters were then frozen into `final_boosted_trainer.py:74-78`.
6. **`train_models_improved.py:125-143`** selects the NegBin `alpha` by **train-set** deviance
   (`res.mu` at `:134-135`), not validation - unlike `train_models_multi.py:299-308`, which correctly
   uses `val_pdev`.
7. **`train_models_improved.py:211, 246`** rebuild the imputation medians from each evaluation split
   (`build_feature_matrix(split_clean, feats)`) instead of reusing the train medians. Mild but real.
8. **Both sides of every game are separate rows** in all trainers, and the chronological split cuts by
   row position, not by game (`train_models_multi.py:74-83`). At the split boundary, team A of a game can
   be in train while team B is in val. Small and bounded, but present - the PPP trainer shows the author
   knew the fix.
9. **`_pair_games` (`run_sims_by_date.py:721-733`) silently drops any game whose two rows do not both
   appear** (`if len(rows) < 2: continue`) - no count, no warning.

---

## 7. WHAT WORKED - worth carrying forward

1. **The pregame-feature construction is correct and reusable.**
   `build_pregame_averages.py:63-114`: per-team gamelog -> `shift(1)` -> expanding mean/std + rolling-5
   mean/std for every box-score column, for both the team's own line and the opponent's line.
   Leakage-free by construction and cheap. Keep this file almost as-is.
2. **As-of KenPom joining.** `merge_kp_and_enrich_gamelogs.py:213-341` (prior-Monday exact join with a
   backward `merge_asof` fallback) and `build_sim_rows_by_date.py:94-100` (snapshot on-or-before). The
   storage layout `data/kenpom/{season}/{YYYY-MM-DD}_kenpom.csv` with 153 snapshots for 2026 is exactly
   what a point-in-time feature store should look like.
3. **Persisting train-time medians next to the model** (`train_models_multi.py:38-40`) and *refusing to
   run* if a medians file is missing or lacks a model feature (`run_sims_by_date.py:644-660`). Also the
   pre-flight check that every model feature exists as a column in the sim rows before any game is
   simulated (`run_sims_by_date.py:1788-1799`). Real engineering discipline; keep it.
4. **Counts x percentages -> points is the right decomposition** for basketball. Modelling volume and
   efficiency separately and combining them binomially gives correct integer scores, natural push
   handling, and interpretable per-component diagnostics. The failure was in *which* covariates and
   *which* dispersion, not in the decomposition.
5. **Trials-weighted Binomial GLM for percentages** - `train_models_multi_pct_trials_weighted.py:351`
   (`freq_weights=n_train`) and `train_models_improved.py:155`. Correct; it just never shipped.
6. **The shared-possession draw in `run_sims_improved.py:325-329`:**
   ```python
   shared_poss = np.random.poisson(shared_mean, nsims)
   adj_A = shared_poss / rowA["exp_possessions_improved"]
   A3 = np.random.poisson(l3a_A * adj_A)
   ```
   One pace realisation per simulation, both teams' volumes scaled by it. This is the correct way to
   induce the positive score correlation the production sim is missing. Take the idea, drop the code
   around it.
7. **The whole evaluation harness.** `compute_daily_results.py` -> `daily_results.json` ->
   `compute_calibration_multi_days.py` (Brier/log-loss, bucketed by price and by spread size) +
   `analyze_profit_range.py` (ROI by market/subset) + `compute_clv_by_date.py`. Scoring *probabilities*
   rather than only W/L, and bucketing by favourite/underdog and line size, is the right instrument. It
   correctly diagnosed the model as bad; nobody acted on the diagnosis.
8. **Per-game artifact layout.** `out/{season}/days/{date}/games/{game_id}/{sims.parquet, summary.json,
   manifest.json}` plus a day-level `index.json` (`run_sims_by_date.py:1168-1180, 1858-1868`) is a clean,
   reproducible contract between engine and consumers.
9. **Model-agnostic pairing and market evaluation.** `_pair_games` (`:721-733`) and the vectorised
   cover/over tallies (`:1474-1554`) are simulator-agnostic - a new engine can reuse this layer verbatim.
10. **The multiprocessing worker-init caching in `rank_round_robin.py:636-655`** is the fix the
    production engine needs.

---

## 8. WHAT WAS WEAK - structural, not cosmetic

**S1. No home-court advantage, and the data was deliberately thrown away.**
`merge_kp_and_enrich_gamelogs.py:21` puts `"Site"` in `EXCLUDE_COLS`. The largest, cheapest,
best-understood effect in college basketball is absent from both training and serving. Measured cost:
-3.12 pts of margin; the sim picks the road side ATS 83% of the time **[measured]**. `OT` and the actual
scores `Tm`/`Opp` are dropped in the same line, so overtime games silently inflate the attempt targets
and the model can never learn to exclude them.

**S2. Raw, un-centered rating features against a drifting scale.** See 3.2.
`team_or`/`opp_dr`/`adjt_mean` are consumed as absolute levels. KenPom's league-mean AdjO moved
100 -> 109.3 across the data used, and moves ~6 points *within* each season. Every feature should have
been expressed relative to the league mean of its own snapshot. This one defect propagates into all six
sub-models simultaneously and in the same direction.

**S3. The six sub-models are independent, and their errors compound.**
Measured predictions on real 2026 rows vs the actual 2026 D-I averages **[measured]**:

| | model | actual 2026 | delta |
|---|---|---|---|
| 3PA | 24.25 | 23.25 | +1.00 |
| 2PA | 35.58 | 35.51 | +0.07 |
| FTA | 21.47 | 20.48 | +0.99 |
| 3P% | .3546 | .339 | +1.6 pp |
| 2P% | .5413 | .526 | +1.5 pp |
| FT% | .7251 | .721 | +0.4 pp |
| **points/team** | **79.93** | **75.82** | **+4.11** |

Six small same-signed errors multiply into +8.2 points of game total. With a joint model of points (or a
possessions x efficiency factorisation) these errors would partially cancel and would be visible in a
single residual. Here they are invisible until you look at the totals line.

**S4. The dispersion model is not a model.**
`_attempts_sd_from_row` / `_pct_sd_from_row` (`:696-713`) take the *arithmetic mean of four different
standard-deviation columns* - the team's own YTD std, its rolling-5 std, and the opponent's allowed YTD
and rolling-5 stds - and call the result the predictive SD. Those quantities measure different things,
they already contain binomial sampling noise, and averaging them has no probabilistic meaning. It is so
large that `:1298` must truncate it to keep the Beta distribution defined - the code knows. Consequence:
implied margin SD 24.6 vs realised 12.4 **[measured]**. Over-wide distributions push every probability
toward 0.50, which is why the spread Brier is 0.2507.

**S5. Full independence between the two teams in a game.**
Nothing is shared: no shared pace realisation, no shared officiating/foul environment, no game-flow
correlation. This is why `SD(margin) == SD(total)` exactly in the sim output. In reality
`SD(margin) ~ 12` and `SD(total) ~ 17`. Any engine that draws A and B independently cannot get both
right.

**S6. Pace is a constant, and the wrong constant.**
`exp_possessions = mean(AdjT_A, AdjT_B)` (`build_sim_rows_by_date.py:275`): not the multiplicative form,
not centered on league tempo, never a random variable, and inconsistent with the attempts the models
actually predict. It is the offset on all three attempt models, so the error is multiplicative on
everything.

**S7. Features are raw counts, not rates, and are not opponent-adjusted.**
`opp_allowed_3pa_ytd_avg` is "3PA the opponent's opponents attempted per game", with no adjustment for
who those opponents were and no normalisation by pace. A slow team with a weak schedule and a fast team
with a strong one are indistinguishable. There is no strength-of-schedule term anywhere except the three
KenPom scalars.

**S8. The deployed models are the 60% train slice, two seasons stale.**
`train_models_multi.py` never refits on all data (6.4-1). Newest training row: 2024-01-07. No retraining
cadence exists - `daily_run.py` refreshes data and publishes, but never retrains.

**S9. Five parallel forks of the engine, none reconciled.**
`run_sims_by_date.py` (1,875 lines), `run_sims_by_date_alt.py` (1,606 - an older copy; the diff is only
ESPN-odds plumbing plus NaN sanitising, and the sim core at `:1037-1083` is byte-identical),
`run_sims_by_date_pace_experiment.py` (1,855), `run_sims_improved.py` (435),
`boosted_simulator_backtest.py` (374), plus `rank_round_robin.py` (a sixth copy of the sim core at
`:725-900`). Each has a different variance model. Fixes made in one (worker-level model caching in
`rank_round_robin.py`, the shared pace draw in `run_sims_improved.py`, the pace model in the experiment
file) never reached production. Bug-fix cost is 5x.

**S10. Dead and duplicated code inside the production file.**
`_ctx_tag_booleans` is defined twice with different signatures (`:119` and `:219`) - the second silently
shadows the first, so `:119-160` is unreachable. `_model_info_dict` is defined at module level (`:1197`)
and again inside `_simulate_one_game` (`:1349`), and neither is ever called; `priors.json` is therefore
an empty stub. `_MEDIANS_CACHE`/`_get_medians` (`:30-54`) are dead. `_sha()` (`:637`) is dead - nothing
fingerprints the models that produced a given output. Roughly 400 of 1,875 lines in the "engine" are
LLM-commentary scaffolding, name-alias tables, and logo/ESPN plumbing.

**S11. Model selection was never done against the model that shipped.**
`percent_model_comparison.py` benchmarks 8 methods for the percentage targets - and omits the production
statsmodels Binomial GLM. No comparison at all exists for the three attempt models.
`boosted_training.py` trains six XGBoost models and never saves one (`:72-86`, then `plt.show()` at
`:122`).

**S12. The betting layer papers over the projection layer.**
`pick_best_bets.py:42-46` gates on recent bucket Brier and multiplies the score by
`(1 + 0.5*recent_ROI)` on a 21-day window with `MIN_BUCKET_SAMPLES = 10`. This is selecting on noise, and
it was tuned on the one window (2025-11-19 to 2025-12-16) where spreads happened to run 54.8% - a rate
that did not survive the season (51.3% **[measured]**).

**S13. Operational fragility.**
`daily_run.py:49-60` rewrites string constants inside `hf_publish.py` and `save_finals_from_sbr.py` on
disk to pass parameters. `daily_run.py:113-124` aborts the whole pipeline if any single task fails - the
tail of `daily_run.log` shows the run dying because `save_finals_from_sbr` returned 1 on an off-season
date. `daily_run.log` is **780 MB**, almost entirely pandas `PerformanceWarning` spam from
`build_pregame_averages.py:109-112`. Output is 15 GB per season, with 1.87 MB of raw draws per game
serialised to JSON for the browser.

---

## 9. OTHER THINGS THAT ARE SURPRISING

1. **The one-sided Over rate was staring at everyone.** `multi_day_calibration.json` records
   `over: n=972` vs `under: n=15`. Season-wide the sim picked Over **95.9%** of the time **[measured]**.
   A model that takes the same side of a market 96% of the time is broken in a way that requires no
   statistics to see.
2. **The correct fix for the percentage models was built, saved to disk, and never wired in.** The `_tw`
   (trials-weighted) models sit next to the production ones in `models/` with identical medians files.
   The only difference is `freq_weights=n_train` at `train_models_multi_pct_trials_weighted.py:351`.
3. **The `_tw` attempt models are pointless duplicates.** `freq_weights` only affects the Binomial fits,
   but the script also re-saves `all_glm_y_3pa_tw.pkl` / `_2pa_tw` / `_fta_tw` (`:402`, `:431`), which
   are re-runs of the identical unweighted Poisson/NegBin fit - about 20 MB of byte-equivalent models.
4. **Both `2025_glm_*.pkl` and the `improved/` models would crash the production loader.**
   `_load_model_and_medians` raises `FileNotFoundError` if `*.medians.json` is absent
   (`run_sims_by_date.py:649-654`), and the `2025_glm_*` files have no medians companions.
5. **The pace model's own feature comments are wrong.** `train_pace_model_multi.py:177-180` labels
   `team_or` as "offensive rebounding" and `opp_dr` as "opponent defensive rebounding". They are KenPom
   AdjO and AdjD. Whoever wrote the pace model did not know what its inputs were.
6. **`def_3_difficulty` / `def_2_difficulty` / `def_ft_difficulty` are literal copies of an existing
   feature.** `build_improved_training_file.py:259-261` and `run_sims_improved.py:195-201`:
   `df_out[new_col] = df[col2].fillna(0.3)` where `col2` is `opp_allowed_{X}pct_ytd_avg`, already in the
   feature list. Three perfectly collinear features shipped into `models/improved/`.
7. **The "boosted simulator backtest" is not a backtest.** `boosted_simulator_backtest.py` never loads a
   final score and never grades anything (`main()` at `:322-372`). It also treats attempts as
   deterministic (`:238-239`) and produces continuous, non-integer scores (`:249-250`), so pushes are
   impossible and the total distribution carries only shooting-percentage variance.
8. **`run_sims_by_date_alt.py` is a 70 KB stale copy.** The full diff against production is ESPN-odds
   plumbing, a NaN-to-null JSON sanitiser, and two back-compat summary keys. The simulation core is
   identical. It is still on disk and still runnable.
9. **`run_sims_by_date.py:1873-1874`**: run with no arguments, it silently simulates 2025-11-04 with
   10,000 sims. `build_sim_rows_by_date.py:34-35` similarly hard-codes `SEASON=2026, DATE='2025-11-21'`,
   and `run_sims_by_date_pace_experiment.py:35` has a stray module-level `DATE = '2025-12-02'`.
10. **A player box-score scraper exists and its output directory is empty.**
    `scrape_cbb_sr_boxscores_with_ids.py` writes `data/boxscores/YYYY-MM-DD.parquet`; the directory has
    0 files and nothing reads it. The player-level ambition was started and abandoned.
11. **Fair odds do not sum to 100%.** Because ties are dropped (`:1390`) and the sim ties 1.46% of the
    time **[measured]**, `A_win_prob + B_win_prob = 0.985`. The +EV filters in
    `compute_daily_results.py:251-264` and `pick_best_bets.py` consume these directly, so every bet is
    scored ~1.5% under its own model's implied probability.
12. **`SBR_TO_KP_ALIASES` (`run_sims_by_date.py:544-570`) contains contradictory entries** -
    `"umass":"massachuesetts"` (typo) alongside `"massachusetts":"umass"`, and
    `'north carolina state':'nc st'` at `:555` overwritten by `'north carolina state':'n c st'` at
    `:563`. Silent name-matching failures mean silently missing odds, which means silently dropped games.
13. **KenPom snapshots exist for a 2027 season** (`data/kenpom/2027/`, 45 files) that are byte-identical
    copies of the final 2026 ratings - the scraper kept running through the off-season, and
    `pick_snapshot_on_or_before` would happily serve them.

---

## 10. Concrete implications for the clean sheet

Ordered by measured impact, not by effort:

1. **Center every rating feature on its own snapshot's league mean** - `team_or - mean(ORtg)`,
   `opp_dr - mean(DRtg)`, `adjt / mean(AdjT)`. Non-negotiable given 3.2.
2. **Home/away/neutral as a first-class feature**, from `team1_ha`/`team2_ha` (already populated) or from
   the SR `Site` column (stop dropping it at `merge_kp_and_enrich_gamelogs.py:21`). Keep `OT` too, so
   overtime games can be excluded from or modelled in the volume targets.
3. **One shared possession draw per simulation**, both teams scaled by it
   (`run_sims_improved.py:325-329` has the shape). Fixes the `SD(margin) == SD(total)` pathology.
4. **Fit the dispersion; do not read it off a feature column.** Use the model's own variance function
   (NegBin alpha, Beta-Binomial overdispersion) and validate against realised residual SD. Target:
   margin SD ~ 12, total SD ~ 17.
5. **Model rates, not counts** - 3PA per possession, 3PM/3PA - so the pace offset is structurally honest
   and the shot profile sums to the possession count.
6. **Weight the percentage models by trials** (`freq_weights`) - the code already exists.
7. **Refit on all available data before deploying**, with a scheduled retrain; keep the chronological
   split for *selection* only.
8. **Add a bias monitor to the daily run**: mean(proj_total - market_total) and
   mean(proj_margin - market_margin) over the last 200 games, with a hard alert at +/-2 points. Either
   number would have caught this in week one.
9. **Keep**: `build_pregame_averages.py`, the as-of KenPom join, the medians + preflight contract, the
   per-game artifact layout, and the entire calibration/ROI harness.
10. **One engine.** Delete the forks; put variants behind flags over a shared core.
