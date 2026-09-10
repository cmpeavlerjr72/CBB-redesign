# control_engine experiments (append-only)

## Pre-registration (2026-09-10, before any code)

Grid:

| Dimension | Values |
|---|---|
| Anchor feature set | A_own (own ridge ratings, as-of daily) ; B_kp (centred KenPom as-of snapshot) ; C_both |
| Rate dispersion | Poisson ; NegBin (chosen per target by training-residual overdispersion test, recorded) |
| Folds | F1 train 2022-2023 test 2024 ; F2 train 2022-2024 test 2025 (selection) |
| Seeds | 200 per game; noise floor = same spec, seed offset +1000 |

Metrics: margin MAE/bias, total MAE/bias, Brier, calibration slope, G1/G5/G6/G9, G10 vs ESPN BET close (2024, 2025).

Decision rules, committed now:
- Anchor set with the lowest F2 margin MAE wins for the Control; if A_own and B_kp are within the seed noise floor, C_both is used only if it beats both by more than the floor, else A_own (compliant, self-contained).
- A rate target uses NegBin only if the Poisson deviance/df on training residuals exceeds 1.2.
- Any G9 bias outside tolerance on F2 is reported as FAIL and traced to a component; nothing is adjusted.

Results: (appended when run)


---

# Results (appended 2026-09-10, after the run)

Full gate report: `docs/tests/control_engine_F2_2026-09-10.md`.
Reproduce with:

```
.venv/Scripts/python.exe scripts/build_own_ratings.py
.venv/Scripts/python.exe scripts/train_control.py --fold {F1,F2} --anchor {A_own,B_kp,C_both}
.venv/Scripts/python.exe scripts/run_control.py   --fold {F1,F2} --anchor ... --seeds 200 [--seed-offset 1000]
.venv/Scripts/python.exe scripts/grade_control.py
```

Population: `is_d1_game & ~pbp_truncated & completed`, minus the handful of games
missing a `team_box` row (10 in 2025, 8 in 2024). F1 test = 5,632 games,
F2 test = 5,700 games. `assert_not_sealed` is called on the season list and on
the materialised training slice in every trainer/selection path; season 2026 was
never read by a fitted or selection artifact.

## R1. Own ridge ratings — fitted hyperparameters

Selected on the **fold-1 training seasons only** (2022, 2023) by one-step-ahead
walk-forward RMSE — the same quantity the ratings are used to predict. Nothing
later than 2023 was read, so the values are honest for both folds.

| model | lambda | prior-season shrinkage weight | walk-forward RMSE | grid |
|---|---|---|---|---|
| offence/defence (pts per 100 poss) | **5.0** | **0.8** | 13.7061 | lambda in {0.5, 1, 2.5, 5, 10, 20, 40, 80, 160, 320, 640} x w in {0, 0.2, 0.4, 0.6, 0.8, 1.0} |
| tempo (game possessions) | **5.0** | **0.8** | 4.7320 | lambda in {0.5, 1, 2.5, 5, 10, 20, 40, 80, 160} x w in {0, 0.2, 0.4, 0.6, 0.8, 1.0} |

Both optima are interior (the eff grid was extended downward from 5.0 after the
first sweep put the winner on the boundary; 2.5 and 1.0 are worse). The surface
is flat: the worst corner of the eff grid is 14.891 vs 13.706 at the optimum, and
w = 1.0 costs only +0.003 over w = 0.8, so the prior weight is weakly identified
and should be re-estimated whenever a season is added. Full grids in
`data/processed/ratings/own_ratings_manifest.json`.

Leak test (`cbb_sim.analysis.leak_test.run_leak_test`, harness was importable),
pooled 2022-2025, 43,255 team-games:

| column | as-joined corr | legit-update corr | level corr | verdict |
|---|---|---|---|---|
| `off_c` | +0.054 | +0.440 | +0.283 | pass (gate 0.15, honest baseline 0.04-0.08) |
| `def_c` | -0.062 | -0.437 | -0.268 | pass |
| `tempo_rel` | +0.002 | -0.003 | -0.006 | pass |

Per-season figures are in `data/processed/ratings/own_ratings_leak_test.csv`;
the worst single season is 2022 at +0.084 / -0.079, still inside the baseline.

## R2. Rate dispersion — the pre-registered NegBin rule applied

Rule as committed: *a rate target uses NegBin only if the Poisson deviance/df on
training residuals exceeds 1.2*. Measured on each fold's own training window;
the two folds agree on every target, so the choice is stable.

| target | Poisson deviance/df (F1 / F2) | decision | fitted NegBin alpha (F1 / F2) |
|---|---|---|---|
| 3PA | 1.587 / 1.597 | **NegBin** | 0.02639 / 0.02677 |
| 2PA | 1.276 / 1.266 | **NegBin** | 0.00725 / 0.00695 |
| FTA | 2.560 / 2.586 | **NegBin** | 0.08375 / 0.08319 |
| TOV | 1.044 / 1.065 | **Poisson** | n/a |

## R3. Every fitted dispersion parameter (F2, anchor A_own)

No dispersion, offset or adjustment anywhere in the engine is hand-set.

| component | family | fitted dispersion | value |
|---|---|---|---|
| pace (game possessions) | Gaussian GLM, identity link | residual SD | **4.6805** (R^2 = 0.282) |
| 3PA per 100 poss | NegBin, log link | alpha (Var = mu + alpha*mu^2) | **0.026774** |
| 2PA per 100 poss | NegBin, log link | alpha | **0.006952** |
| FTA per 100 poss | NegBin, log link | alpha | **0.083192** |
| TOV per 100 poss | Poisson, log link | none needed (dev/df 1.065) | -- |
| 3P% | Binomial (trials-weighted) + Beta-Binomial | rho | **0.000116** |
| 2P% | Binomial (trials-weighted) + Beta-Binomial | rho | **0.002924** |
| FT% | Binomial (trials-weighted) + Beta-Binomial | rho | **0.009709** |

rho is Williams' moment estimator from
`sum (y - n p)^2 = sum n p (1-p) [1 + (n-1) rho]`. The Binomial Pearson
dispersions it is derived from are 1.003 (3P%), 1.103 (2P%), 1.183 (FT%): team
3P% in a game is essentially binomial once the model knows who is shooting,
while FT% carries real extra-binomial spread. Per-fold, per-anchor values in
`data/processed/models/control_engine/train_{fold}_{anchor}.json`.

## R4. Fold-2 scorecard, the three anchors (selection metric)

200 seeds, 5,700 games, paired (seed, game_id) streams.

| anchor | margin MAE | margin bias | total MAE | total bias | Brier | calib slope |
|---|---|---|---|---|---|---|
| A_own | 9.1462 | +0.147 | 13.2038 | -2.826 | 0.18568 | 0.960 |
| B_kp | 9.0754 | -0.390 | 13.1596 | -2.888 | 0.18679 | 1.040 |
| C_both | **9.0591** | -0.285 | **13.1404** | -2.866 | 0.18590 | 1.010 |

Fold 1 (drift check), same spec one season earlier:

| anchor | margin MAE | margin bias | total MAE | total bias | Brier |
|---|---|---|---|---|---|
| A_own | 9.2048 | +0.549 | 13.5450 | -4.077 | 0.18908 |
| B_kp | 9.1634 | -0.237 | 13.5652 | -4.112 | 0.19207 |
| C_both | 9.1450 | -0.156 | 13.5302 | -4.118 | 0.19115 |

Ordering is identical on both folds, which is the drift read: no anchor's edge
is a one-season artifact.

## R5. Seed noise floor

Spec-identical re-simulation at seed offset +1000 (`--seed-offset 1000`):

| anchor | margin MAE, seeds 0-199 | seeds 1000-1199 | \|delta\| |
|---|---|---|---|
| A_own | 9.14619 | 9.22304 | 0.07685 |
| B_kp | 9.07540 | 9.12482 | 0.04942 |
| C_both | 9.05907 | 9.10574 | 0.04667 |

Floor = 0.047-0.077 margin-MAE points, mean 0.058. This is larger than it looks:
because the simulated margin SD is ~19.6, the Monte-Carlo SE of a game's
predicted margin at 200 seeds is 19.6/sqrt(200) = 1.39 points, and that noise
propagates straight into MAE. **200 seeds is not enough to separate these
anchors on the unpaired floor.** The formal seed-count study (FRAMEWORK_PLAN
week 6) should read this number, and the floor will shrink once the G5
over-dispersion below is fixed, because the MC SE scales with the sim SD.

## R6. Decision — anchor

**Chosen: A_own**, by the rule committed above.

Applying it literally: |A_own - B_kp| = 0.0708 margin-MAE points, inside the
measured floor of 0.047-0.077, so the two are *within the seed noise floor* and
the tie clause governs. C_both is then adopted only if it beats both by more
than the floor; it beats the better of them (B_kp) by 0.0163, roughly a quarter
of the floor. The clause falls through to A_own -- the compliant, self-contained
arm, and also the simpler model under the standing tie rule.

Recorded for the PM, not acted on: the arms share their (seed, game_id) streams,
so a paired per-game comparison cancels most of the MC noise and is far more
powerful than the unpaired floor. Paired differences in margin MAE:

| comparison | delta | paired SE | t |
|---|---|---|---|
| A_own - B_kp | +0.0708 | 0.0299 | 2.37 |
| A_own - C_both | +0.0871 | 0.0247 | 3.52 |
| B_kp - C_both | +0.0163 | 0.0068 | 2.40 |

On the paired statistic KenPom carries a small but real edge over our own
ratings, and the union a smaller one again. Both edges are ~0.9% and ~0.2% of
margin MAE, i.e. worth about 0.07 and 0.02 points of spread -- real, tiny, and
below the pre-registered floor. The honest reading is that **our own ridge
ratings are competitive with KenPom as a pregame anchor**, which is the
result CLAUDE.md's data rules were hoping for. If a future decision rule wants
to resolve differences this small it needs the paired SE, not the unpaired
seed-offset floor.

## R7. Decision — rate dispersion

**NegBin for 3PA, 2PA and FTA; Poisson for TOV**, by the deviance/df > 1.2 rule,
on both folds independently. Recorded above with the test statistic and the
fitted alphas.

## R8. Gate results (fold 2, A_own)

| gate | quantity | value | tolerance | status |
|---|---|---|---|---|
| G1 | possessions/game mean | 68.268 vs 67.875 | +/-1.0 | PASS |
| G1 | possessions/game SD | 5.702 vs 5.474 | +/-0.75 | PASS |
| G1 | by month, mean and SD | 0/5 powered months out (April, n=17, UNDERPOWERED) | all inside | PASS |
| G5 | margin SD ratio | 1.678 | 0.95-1.05 | FAIL |
| G5 | total SD ratio | 1.318 | 0.95-1.05 | FAIL |
| G5 | home/away score corr | 0.084 vs 0.229 | +/-0.05 | FAIL |
| G5 | PIT K-S p | 2.4e-79 | > 0.1 | FAIL |
| G6 | home margin, non-neutral | +6.058 vs +5.732 | +/-1.0 | PASS |
| G6 | home margin, neutral | +2.229 vs +3.288 | +/-1.0 | FAIL |
| G9 | margin bias | +0.147 | +/-0.5 | PASS |
| G9 | total bias | -2.826 | +/-1.0 | FAIL |
| G9 | calibration slope | 0.960 | 0.95-1.05 | PASS |
| G9 | responsiveness slope | 0.977, monotone | monotone, 0.85-1.15 | PASS |
| G9 | bias by month | 8/10 scored cells out | all inside | FAIL |
| G9 | bias by tier | 6/6 scored cells out | all inside | FAIL |
| G9 | bias by predicted-total tercile | 3/6 scored cells out | all inside | FAIL |
| G10 | 2024 margin MAE vs close | 9.233 vs 8.870 | report only | PASS |
| G10 | 2024 leak screen | surprise corr +0.010 | <= 0.15 | PASS |
| G10 | 2025 margin MAE vs close | 9.106 vs 8.746 | report only | PASS |
| G10 | 2025 leak screen | surprise corr -0.013, CLV agreement 0.530 | <= 0.15 | PASS |

Nothing was adjusted in response to any failure. Diagnoses are in section 10 of
the gate report; the short form:

- **G5 (all four lines) -> the attempt-count layer.** The four per-100-possession
  count models are individually right (fitted marginal SD 5.84/6.72/6.75/3.50 vs
  actual 5.97/7.20/7.36/3.98), but the simulator draws them independently while
  in the data their Pearson residuals correlate -0.56 (3PA/2PA), -0.36
  (2PA/TOV), -0.29 (3PA/FTA): they compete for the same finite possessions.
  Computed from the fitted parameters alone, independent components imply a
  team-points SD of 13.87 against an actual residual SD of 9.22, and
  sqrt(2) x 13.87 = 19.61 against a measured simulated margin SD of 19.62. The
  same inflated per-team variance is what dilutes the home/away score
  correlation and drives the PIT failure and the under-confident win-probability
  deciles. The model class cannot represent the negative dependence, so this is a
  rebuild, not an adjustment: the L3 possession-outcome model allocates each
  possession to one outcome and gets it for free. **The Control's point estimates
  are usable; its intervals are not.**
- **G9 total bias and its month/tier breakdowns -> the GLM level (intercept).**
  Fold-2 training seasons average 141.82 total points, test season 145.51, drift
  +3.69; the pre-registered feature list is entirely centred ratings plus site
  plus day-of-season, so nothing carries a season level and the engine inherits
  77% of the drift as -2.83. Fold 1 shows the same mechanism at -4.08. The close
  is essentially unbiased on totals over the same games (-0.56), so this is the
  model, not the grading truth. SIM_GUARDRAILS section 4 predicted it verbatim.
- **G9 bias by tier -> mostly the tier definition.** Tiers are terciles of each
  team's own full-season margin and are explicitly not leak-free. Running the
  same breakdown on the closing line gives +1.32 / +0.11 / -1.40 against the
  model's +1.96 / +0.46 / -1.49, so most of the pattern is selection on the
  outcome. The residual the engine owns is +0.64 / +0.35 / -0.09.
- **G6 neutral -> the site feature is too coarse.** Non-neutral passes at +0.33,
  so the home term is wired correctly everywhere; the neutral bucket is
  heterogeneous (sub-regionals in a team's home state, in-season tournaments in a
  team's own market) and the nominal home team there is systematically the
  stronger side.
- **G1 sign -> the overtime stub.** The pace target is observed possessions,
  which already include any overtime played, and the stub adds a further 5/40 of
  a game to every tied sim: about +0.16 possessions per game at the simulated OT
  rate of 0.0183. That simulated OT rate is itself far below the actual 0.0558,
  because a Beta-Binomial/NegBin score has no end-game mechanics to pile
  probability onto an exact tie. Both are the known L5 gap (model.md section 9).

## R9. G10 — market scorecard vs ESPN BET closes

| season | n with line | model margin MAE | close margin MAE | model margin bias | close margin bias | model total MAE | close total MAE | model total bias | close total bias |
|---|---|---|---|---|---|---|---|---|---|
| 2024 (F1) | 5,223 | 9.2330 | 8.8697 | +0.559 | +0.015 | 13.5322 | 12.9114 | -4.070 | -0.828 |
| 2025 (F2) | 5,375 | 9.1058 | 8.7460 | +0.090 | -0.161 | 13.2181 | 12.6799 | -2.846 | -0.562 |

ATS by disagreement bucket (bet the model's side, -110):

| season | >= 1 | >= 2 | >= 3 | >= 5 |
|---|---|---|---|---|
| 2024 | 1943-1949, 49.9% | 1366-1370, 49.9% | 908-885, 50.6% | 333-322, 50.8% |
| 2025 | 2002-2009, 49.9% | 1395-1380, 50.3% | 886-891, 49.9% | 318-324, 49.5% |

ROI at -110 is -3.2% to -6.0% in every bucket of both seasons. Brier against the
de-vigged moneyline: 2024 model 0.18947 vs market 0.17841 (n = 5,204, mean vig
0.0388); 2025 model 0.18818 vs market 0.17536 (n = 5,222, mean vig 0.0431).

**Reading: the Control trails the close by ~0.36 points of margin MAE and ~0.54
points of total MAE and has no ATS edge at any disagreement threshold.** That is
the intended result -- it is the yardstick, not a candidate -- and it sets the
bar: the possession engine has to close a 0.36-point margin gap before any
market claim is worth making. The leak screen is clean in both seasons
(surprise corr +0.010 and -0.013 against a 0.15 gate; 2025 CLV sign agreement
0.530 on 1,085 moved lines, at the 0.53 threshold and therefore not a
LEAK-SUSPECT under the AND rule, but also no evidence of line-move skill).
2024 has no opening lines in the CBBD pull, so its CLV cell is unmeasurable.

## R10. Implementation notes / deviations from `model.md`

1. **CLI shape.** `model.md` section 7 describes
   `run_control.py --season 2025 --seeds 200` writing `results/control/<season>/`.
   Because the anchor comparison is run inside the Control, the shipped CLI is
   `--fold {F1,F2} --anchor {A_own,B_kp,C_both}` writing
   `results/control/<fold>_<anchor>[_seedoff{N}]/`, and the model artifacts are
   `{pace,rates,pcts}_{fold}_{anchor}.pkl`. Same contents, one more axis.
2. **Pace features are symmetric sums** of the two teams' ratings plus a neutral
   indicator, not separate home/away coefficients. Rationale in `features.md`
   section 1b: one pace realisation per game with both teams scaled by it, and
   asymmetric pace weights are on the banned list (`SIM_GUARDRAILS.md` section 1
   cites last year's 0.6/0.4 patch).
3. **Calibration slope is reported twice.** The gate value is OLS of actual
   margin on the SIM MEAN margin. That regressor is a Monte-Carlo estimate, so
   the slope is attenuated by Var(MC) = mean(sim SD^2)/n_seeds; at 200 seeds and
   a sim SD of 19.6 the attenuation is ~2%. The report also prints an
   errors-in-variables-corrected slope (0.982 for A_own) clearly labelled a
   diagnostic. The deterministic expected-margin slope on fold 1 is 0.964, so the
   mean predictions themselves are well calibrated; both slope numbers move
   toward 1.0 once the G5 over-dispersion is fixed.
4. **KenPom as-of falls back across the season boundary.** `merge_asof` is keyed
   on the team only, not (team, season) -- the same semantics as
   `cbb_sim.data.kenpom.join_as_of` -- so a game played before that season's
   first snapshot gets the previous season's last snapshot rather than NaN. Still
   strictly pregame, and comparable because every snapshot column is centred on
   its own snapshot's league mean. Without it, B_kp would have lost ~2.5% of
   games to the median fallback on opening weekend and the anchor comparison
   would have been unfair.
5. **RNG.** `(seed, game_id, "control")` is realised as a counter-based
   splitmix64 stream rather than a per-game `Generator`, so a game's draws depend
   on nothing but that triple -- verified in `tests/test_control.py` by
   re-simulating a 37-game subset and requiring bit-identical results. 200 seeds
   x 5,700 games runs in ~18 s single-process.

## R11. Follow-ups this run generated

| # | Item | Owner layer |
|---|---|---|
| 1 | Attempt counts must come from a possession-outcome allocation, not four independent overdispersed marginals. This is the single largest defect in the Control and it invalidates every interval it produces | L3 |
| 2 | Something must carry the season scoring level. The as-of league mean from `own_ratings` is already built, pregame and leak-safe; a preseason refit is the alternative. Pre-registration first | L2/L3 |
| 3 | The site feature needs more than home/away/neutral -- a designated-home or venue-distance term for the neutral bucket | feature layer |
| 4 | Pace target should probably be regulation-only possessions once an overtime model exists, otherwise the OT stub double-counts | L5 |
| 5 | Decision rules that need to resolve differences smaller than ~0.05 margin MAE must use the paired per-game SE, not the unpaired seed-offset floor | process |
| 6 | Seed-count study should read R5's Monte-Carlo SE of 1.39 points at 200 seeds | week 6 |
