# Control engine gate report -- fold F2 (train 2022-2024, test 2024-25)

Generated 2026-09-10 by `scripts/grade_control.py`. Spec: `docs/models/control_engine/model.md` (pre-registered 2026-09-10). Tolerances: `docs/SIM_GUARDRAILS.md` section 3 (provisional until the seed-noise study). Every gate line ends in a literal PASS / FAIL / NEEDS-INSTRUMENTATION.

The Control is a yardstick, not a candidate. Nothing in this report is used to adjust it: a failed gate is reported with the component responsible and left alone (CLAUDE.md "no hand tuning on engine output").

## 0. Runs graded

| fold | anchor | seed_offset | seeds | games | test_season | backtest | created_at |
|---|---|---|---|---|---|---|---|
| F1 | A_own | 0 | 200 | 5632 | 2024 | True | 2026-09-10 |
| F1 | A_own | 1000 | 200 | 5632 | 2024 | True | 2026-09-10 |
| F1 | B_kp | 0 | 200 | 5632 | 2024 | True | 2026-09-10 |
| F1 | C_both | 0 | 200 | 5632 | 2024 | True | 2026-09-10 |
| F1 | C_both | 1000 | 200 | 5632 | 2024 | True | 2026-09-10 |
| F2 | A_own | 0 | 200 | 5700 | 2025 | True | 2026-09-10 |
| F2 | A_own | 1000 | 200 | 5700 | 2025 | True | 2026-09-10 |
| F2 | B_kp | 0 | 200 | 5700 | 2025 | True | 2026-09-10 |
| F2 | B_kp | 1000 | 200 | 5700 | 2025 | True | 2026-09-10 |
| F2 | C_both | 0 | 200 | 5700 | 2025 | True | 2026-09-10 |
| F2 | C_both | 1000 | 200 | 5700 | 2025 | True | 2026-09-10 |

`backtest=True` on every row: these are completed seasons, so `created_at < tipoff` is impossible and is stamped rather than faked (`scripts/run_control.py` asserts the inequality only for `--live` runs).

## 1. Headline scorecard

Fold 2 is the selection metric; fold 1 is shown for drift. `calib_slope` is OLS of actual margin on the SIM MEAN margin; because the sim mean is itself a Monte-Carlo estimate, that slope is attenuated by Var(MC) = mean(sim SD^2)/n_seeds. `slope_MC_corrected` divides that attenuation out and is a DIAGNOSTIC, not the gate.

| fold | anchor | n | margin_MAE | margin_bias | total_MAE | total_bias | Brier | calib_slope | slope_MC_corrected |
|---|---|---|---|---|---|---|---|---|---|
| F1 | A_own | 5632 | 9.2048 | 0.5489 | 13.5450 | -4.0770 | 0.1891 | 0.9486 | 0.9722 |
| F1 | B_kp | 5632 | 9.1634 | -0.2366 | 13.5652 | -4.1123 | 0.1921 | 1.0278 | 1.0572 |
| F1 | C_both | 5632 | 9.1450 | -0.1555 | 13.5302 | -4.1178 | 0.1912 | 1.0107 | 1.0384 |
| F2 | A_own | 5700 | 9.1462 | 0.1472 | 13.2038 | -2.8263 | 0.1857 | 0.9595 | 0.9820 |
| F2 | B_kp | 5700 | 9.0754 | -0.3904 | 13.1596 | -2.8876 | 0.1868 | 1.0404 | 1.0679 |
| F2 | C_both | 5700 | 9.0591 | -0.2853 | 13.1404 | -2.8661 | 0.1859 | 1.0098 | 1.0348 |

Fold-2 ranking on the pre-registered primary metric (margin MAE): C_both 9.0591 < B_kp 9.0754 < A_own 9.1462.

## 2. Seed noise floor and the anchor decision

Noise floor = the same spec re-simulated at seed offset +1000 (`--seed-offset 1000`), differenced against the base run on the same games.

| anchor | margin_MAE_seed0 | margin_MAE_seed1000 | abs_delta | total_MAE_delta | Brier_delta |
|---|---|---|---|---|---|
| A_own | 9.14619 | 9.22304 | 0.07685 | 0.03933 | 0.00147 |
| B_kp | 9.07540 | 9.12482 | 0.04942 | 0.03897 | 0.00124 |
| C_both | 9.05907 | 9.10574 | 0.04667 | 0.03669 | 0.00107 |

Paired per-game differences (the two arms share the same (seed, game_id) streams, so Monte-Carlo noise largely cancels):

| comparison | delta_margin_MAE | paired_SE | t |
|---|---|---|---|
| A_own - B_kp | 0.0708 | 0.0299 | 2.3656 |
| A_own - C_both | 0.0871 | 0.0247 | 3.5205 |
| B_kp - C_both | 0.0163 | 0.0068 | 2.3982 |

**Decision (pre-registered rule, `experiments.md`): A_own.** |A_own - B_kp| = 0.0708 is within the measured seed noise floor (0.0467-0.0769, mean 0.0576), so the pre-registered tie clause applies: C_both is adopted only if it beats BOTH by more than the floor. It beats the better of them by 0.0163, below the floor.

Runner-up on the raw metric is C_both (9.0591) ahead of B_kp (9.0754); the paired table above shows that gap is statistically detectable once MC noise is cancelled, but the pre-registered floor is the unpaired seed-offset difference and it is coarser than the paired SE. Recorded as an observation for the PM, not acted on here.

## 3. G1 -- possessions per game (A_own, F2, test season 2025)

`sim` pools every (game, seed) row; `actual` is the same games' box-derived possessions (FGA - OREB + TOV + 0.44 FTA, averaged over both teams); `ref` is `data/reference/gate_targets_{season}.parquet` over the full season. Tolerance: mean +/- 1.0, SD +/- 0.75.

| breakdown | group | n_games | sim_mean | actual_mean | ref_mean | sim_sd | actual_sd | ref_sd | d_mean | d_sd | status_mean | status_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| season | all | 5700 | 68.268 | 67.875 | 67.875 | 5.702 | 5.474 | 5.474 | 0.393 | 0.228 | PASS | PASS |
| month | 1 | 1420 | 68.158 | 67.434 | 67.434 | 5.720 | 5.482 | 5.482 | 0.724 | 0.238 | PASS | PASS |
| month | 2 | 1364 | 67.677 | 67.234 | 67.234 | 5.701 | 5.160 | 5.160 | 0.443 | 0.541 | PASS | PASS |
| month | 3 | 765 | 67.387 | 67.483 | 67.483 | 5.774 | 5.352 | 5.352 | -0.095 | 0.422 | PASS | PASS |
| month | 4 | 17 | 66.120 | 69.136 | 69.136 | 5.761 | 5.669 | 5.669 | -3.016 | 0.093 | UNDERPOWERED | UNDERPOWERED |
| month | 11 | 1219 | 69.223 | 69.070 | 69.070 | 5.522 | 5.582 | 5.582 | 0.153 | -0.060 | PASS | PASS |
| month | 12 | 915 | 68.824 | 68.229 | 68.229 | 5.601 | 5.599 | 5.599 | 0.595 | 0.002 | PASS | PASS |

G1 possessions mean: sim 68.268 vs actual 67.875 (delta +0.393, tol +/-1.0) -- PASS

G1 possessions SD: sim 5.702 vs actual 5.474 (delta +0.228, tol +/-0.75) -- PASS

G1 by month: 5/5 powered months inside both tolerances (1 month(s) below n=50 labelled UNDERPOWERED and not scored) -- PASS

## 4. G5 -- dispersion (A_own, F2)

SD ratio = mean(sim SD) / SD(actual - sim mean). >1 means the engine is too wide. Tolerance 0.95-1.05; score correlation +/-0.05; PIT K-S p > 0.1.

| quantity | mean_sim_SD | SD(actual - sim mean) | ratio | status |
|---|---|---|---|---|
| margin | 19.6215 | 11.6919 | 1.6782 | FAIL |
| total | 22.0260 | 16.7133 | 1.3179 | FAIL |

G5 margin SD ratio 1.6782 -- FAIL

G5 total SD ratio 1.3179 -- FAIL

G5 home/away score correlation: sim 0.0836 vs actual 0.2285 on the same 5700 games (delta -0.1449) -- FAIL

(The season reference table gives 0.2532 over all 5710 D-I non-truncated games; the 10 games this run drops for a missing team_box row happen to be scoring outliers, which is why the like-for-like actual is lower. The gate compares sim to actual on the SAME games.)

G5 PIT K-S vs Uniform(0,1): D = 0.1260, p = 2.38e-79 -- FAIL

PIT decile histogram (a correctly dispersed engine is flat at 10% per decile; a mass in the middle deciles means the engine is too WIDE):

| decile | share |
|---|---|
| 1 | 0.0170 |
| 2 | 0.0609 |
| 3 | 0.1079 |
| 4 | 0.1540 |
| 5 | 0.1747 |
| 6 | 0.1723 |
| 7 | 0.1296 |
| 8 | 0.1039 |
| 9 | 0.0582 |
| 10 | 0.0214 |

## 5. G6 -- home margin, non-neutral vs neutral, same games (A_own, F2)

Tolerance +/-1.0 points.

| site | n | sim | actual | delta | status |
|---|---|---|---|---|---|
| non-neutral | 4964 | 6.0577 | 5.7317 | 0.3260 | PASS |
| neutral | 736 | 2.2295 | 3.2880 | -1.0586 | FAIL |

G6 home margin (non-neutral): sim +6.058 vs actual +5.732 (delta +0.326) -- PASS

G6 home margin (neutral): sim +2.229 vs actual +3.288 (delta -1.059) -- FAIL

## 6. G9 -- spread and total accuracy (A_own, F2)

Tolerance: margin bias +/-0.5, total bias +/-1.0, calibration slope 0.95-1.05.

G9 margin MAE 9.1462, bias +0.1472 -- PASS

G9 total MAE 13.2038, bias -2.8263 -- FAIL

G9 calibration slope 0.9595 (MC-corrected 0.9820, MC noise SD 1.390) -- PASS

G9 win-probability Brier 0.18568

Win-probability calibration by decile:

| decile | n | pred | actual | delta |
|---|---|---|---|---|
| 1 | 586 | 0.3134 | 0.2014 | -0.1121 |
| 2 | 558 | 0.4283 | 0.4247 | -0.0035 |
| 3 | 609 | 0.4919 | 0.5008 | 0.0090 |
| 4 | 596 | 0.5399 | 0.5688 | 0.0288 |
| 5 | 564 | 0.5859 | 0.6383 | 0.0524 |
| 6 | 523 | 0.6271 | 0.7075 | 0.0803 |
| 7 | 557 | 0.6696 | 0.7343 | 0.0646 |
| 8 | 589 | 0.7198 | 0.8370 | 0.1173 |
| 9 | 561 | 0.7808 | 0.8913 | 0.1105 |
| 10 | 557 | 0.8764 | 0.9659 | 0.0895 |

### G9 by month

| month | n | margin_mae | margin_bias | total_mae | total_bias | slope | status_margin_bias | status_total_bias |
|---|---|---|---|---|---|---|---|---|
| 1 | 1420 | 8.9973 | 1.3810 | 13.4259 | -1.9151 | 0.8907 | FAIL | FAIL |
| 2 | 1364 | 8.8231 | 0.8219 | 13.0943 | -2.3406 | 0.8657 | FAIL | FAIL |
| 3 | 765 | 8.7496 | -0.4818 | 12.7988 | -2.7861 | 0.9356 | PASS | FAIL |
| 4 | 17 | 8.6397 | 1.7815 | 15.8868 | -10.2144 | n/a | UNDERPOWERED | UNDERPOWERED |
| 11 | 1219 | 10.0355 | -1.5914 | 13.8971 | -4.1075 | 1.0128 | FAIL | FAIL |
| 12 | 915 | 9.0152 | 0.0385 | 12.3876 | -3.1538 | 0.9573 | PASS | FAIL |

G9 by month: 8 of 10 scored bias cells outside tolerance -- FAIL

### G9 by tier

| home_tier | n | margin_mae | margin_bias | total_mae | total_bias | slope | status_margin_bias | status_total_bias |
|---|---|---|---|---|---|---|---|---|
| bottom_tercile | 1604 | 9.0695 | 1.8449 | 13.1957 | -3.0240 | 0.8060 | FAIL | FAIL |
| middle_tercile | 1827 | 8.9586 | 0.5145 | 13.5386 | -2.3564 | 0.7866 | FAIL | FAIL |
| top_tercile | 2269 | 9.3514 | -1.3486 | 12.9400 | -3.0649 | 0.9238 | FAIL | FAIL |

G9 by tier: 6 of 6 scored bias cells outside tolerance -- FAIL

### G9 by pred_total_tercile

| pred_total_tercile | n | margin_mae | margin_bias | total_mae | total_bias | slope | status_margin_bias | status_total_bias |
|---|---|---|---|---|---|---|---|---|
| bottom_tercile | 1900 | 8.9604 | 0.3179 | 12.5696 | -3.4995 | 0.9551 | PASS | FAIL |
| middle_tercile | 1900 | 9.0839 | 0.1512 | 13.3195 | -2.6384 | 0.9552 | PASS | FAIL |
| top_tercile | 1900 | 9.3942 | -0.0274 | 13.7224 | -2.3409 | 0.9636 | PASS | FAIL |

G9 by pred_total_tercile: 3 of 6 scored bias cells outside tolerance -- FAIL

## 7. Responsiveness (A_own, F2)

Games bucketed by pregame rating differential (a prior, not the prediction). The standing rule is that predictions must SLOPE with actuals, not sit flat at the mean.

| quintile | n | rating_diff | predicted_margin | actual_margin | delta |
|---|---|---|---|---|---|
| 1 | 1140 | -13.4068 | -6.2099 | -5.7982 | -0.4116 |
| 2 | 1140 | -3.6158 | 0.7813 | 1.1035 | -0.3222 |
| 3 | 1140 | 2.3382 | 4.9830 | 4.0807 | 0.9023 |
| 4 | 1140 | 8.6100 | 9.6296 | 9.0649 | 0.5647 |
| 5 | 1140 | 20.4847 | 18.6328 | 18.6298 | 0.0030 |

Responsiveness: actual margin monotone across quintiles = True; slope of actual on predicted across quintile means = 0.9765 -- PASS

## 8. G10 -- market scorecard vs ESPN BET closes

CBBD `lines_{season}.parquet`, provider ESPN BET, joined on `cbbd_game_id`. `spread` is home-perspective, so the market's expected home margin is `-spread`. G10 is report-only except the leak screen (LEAK-SUSPECT if surprise corr > 0.15 AND CLV agreement < 0.53).

| fold | season | n_with_line | model_margin_MAE | close_margin_MAE | model_margin_bias | close_margin_bias | model_total_MAE | close_total_MAE | model_total_bias | close_total_bias |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 2024 | 5223 | 9.2330 | 8.8697 | 0.5589 | 0.0148 | 13.5322 | 12.9114 | -4.0696 | -0.8282 |
| F2 | 2025 | 5375 | 9.1058 | 8.7460 | 0.0897 | -0.1614 | 13.2181 | 12.6799 | -2.8457 | -0.5623 |

G10 2024 margin: model MAE 9.2330 vs close 8.8697 (model trails the close by 0.3633) -- report-only, PASS

G10 2024 total: model MAE 13.5322 vs close 12.9114; model total bias -4.0696 vs close -0.8282 -- report-only, PASS

### ATS by disagreement bucket, 2024

| bucket | n | wins | losses | pushes | win_pct | roi_at_-110 |
|---|---|---|---|---|---|---|
| >= 1 | 3892 | 1943 | 1949 | 0 | 0.4992 | -0.0516 |
| >= 2 | 2736 | 1366 | 1370 | 0 | 0.4993 | -0.0515 |
| >= 3 | 1793 | 908 | 885 | 0 | 0.5064 | -0.0365 |
| >= 5 | 655 | 333 | 322 | 0 | 0.5084 | -0.0324 |

G10 2024 Brier vs de-vigged moneyline (n = 5204, mean vig 0.0388): model 0.18947 vs market 0.17841 -- report-only, PASS

G10 2024 leak screen: surprise corr 0.0103 (gate 0.15), CLV sign agreement n/a (no opening lines) on 0 moved lines -- PASS

G10 2025 margin: model MAE 9.1058 vs close 8.7460 (model trails the close by 0.3598) -- report-only, PASS

G10 2025 total: model MAE 13.2181 vs close 12.6799; model total bias -2.8457 vs close -0.5623 -- report-only, PASS

### ATS by disagreement bucket, 2025

| bucket | n | wins | losses | pushes | win_pct | roi_at_-110 |
|---|---|---|---|---|---|---|
| >= 1 | 4011 | 2002 | 2009 | 0 | 0.4991 | -0.0518 |
| >= 2 | 2775 | 1395 | 1380 | 0 | 0.5027 | -0.0443 |
| >= 3 | 1777 | 886 | 891 | 0 | 0.4986 | -0.0530 |
| >= 5 | 642 | 318 | 324 | 0 | 0.4953 | -0.0598 |

G10 2025 Brier vs de-vigged moneyline (n = 5222, mean vig 0.0431): model 0.18818 vs market 0.17536 -- report-only, PASS

G10 2025 leak screen: surprise corr -0.0129 (gate 0.15), CLV sign agreement 0.5300 on 1085 moved lines -- PASS

## 9. Fold drift (F1 vs F2)

Same spec, one season earlier. Both folds train on pooled seasons and test on the next one, so the size and sign of the total bias is the direct read on season-level drift.

| anchor | margin_MAE_F1 | margin_MAE_F2 | margin_bias_F1 | margin_bias_F2 | total_bias_F1 | total_bias_F2 | Brier_F1 | Brier_F2 |
|---|---|---|---|---|---|---|---|---|
| A_own | 9.2048 | 9.1462 | 0.5489 | 0.1472 | -4.0770 | -2.8263 | 0.1891 | 0.1857 |
| B_kp | 9.1634 | 9.0754 | -0.2366 | -0.3904 | -4.1123 | -2.8876 | 0.1921 | 0.1868 |
| C_both | 9.1450 | 9.0591 | -0.1555 | -0.2853 | -4.1178 | -2.8661 | 0.1912 | 0.1859 |

## 10. Diagnosis of every failed gate

The first response to a failed gate is *which sub-model is producing the wrong distribution*, never *what adjustment closes the gap* (`docs/SIM_GUARDRAILS.md` core principle). Nothing below has been changed.

### D1. G5 margin/total SD ratio, home-away score correlation, PIT -- the attempt-count layer

The four per-100-possession count models are each individually well calibrated: their fitted marginal SD matches the data.

| count | actual_mean | actual_SD | model_SD | family | poisson_deviance_df |
|---|---|---|---|---|---|
| tpa | 21.6067 | 5.9673 | 5.8415 | negbin | 1.5968 |
| fg2a | 36.1127 | 7.2012 | 6.7186 | negbin | 1.2657 |
| fta | 18.2110 | 7.3552 | 6.7533 | negbin | 2.5863 |
| tov | 12.2875 | 3.9782 | 3.4985 | poisson | 1.0646 |

But the simulator draws those four counts INDEPENDENTLY given the shared possession draw, and in the data they are strongly negatively correlated -- they compete for the same finite possessions (a possession that ends in a turnover is not a shot; a three is not a two). Pearson-residual correlation on the fold-2 training team-games:

| count | tpa | fg2a | fta | tov |
|---|---|---|---|---|
| tpa | 1.000 | -0.561 | -0.287 | -0.157 |
| fg2a | -0.561 | 1.000 | -0.158 | -0.360 |
| fta | -0.287 | -0.158 | 1.000 | -0.019 |
| tov | -0.157 | -0.360 | -0.019 | 1.000 |

Consequence, computed from the fitted parameters alone: the independent-component variance implies a team-points SD of 13.87, while the actual residual team-points SD is 9.22 (unconditional 11.94). sqrt(2) x 13.87 = 19.61 is the implied simulated margin SD, and the report above measures 19.62. The same inflated per-team variance is what dilutes the home/away score correlation (0.0836 simulated vs 0.2285 actual): the shared pace draw contributes about the right covariance, but it is divided by two SDs that are ~50% too large. The PIT failure and the under-confident win-probability deciles are the same defect seen through two more lenses.

**Responsible component: the attempt-count layer (3PA / 2PA / FTA / TOV drawn as four independent overdispersed counts).** The marginals are right and every dispersion parameter is fitted, so this is not a tuning error -- the model class cannot represent the negative dependence, exactly the situation SIM_GUARDRAILS section 5 says requires a rebuilt model rather than an adjustment. The pre-registered replacement is the L3 possession-outcome model, which allocates each possession to one outcome and therefore gets the competition for possessions for free. Until then the Control's point estimates are usable and its intervals are not.

### D2. G9 total bias (and its month/tier breakdowns) -- pooled-season training against a rising scoring level

Scoring per game has risen every season in the data. Reference totals:

| season | total_points_mean | n |
|---|---|---|
| 2022 | 139.800 | 5406 |
| 2023 | 140.530 | 5658 |
| 2024 | 145.128 | 5640 |
| 2025 | 145.509 | 5710 |

The fold-2 training seasons average 141.82 total points; the test season is 145.51, a drift of +3.69. The pre-registered feature list is entirely CENTRED ratings plus site plus day-of-season, so nothing in it carries a season LEVEL: the GLM intercepts are pooled over the training seasons and the engine inherits the shortfall. The measured total bias is -2.826, i.e. 77% of the raw drift (the rest is absorbed by the as-of rating and day-of-season terms). Fold 1 shows the same mechanism one season earlier at a larger magnitude, which is the drift check in section 9. The closing line over the same games is essentially unbiased on totals (-0.562), confirming this is the model, not the grading truth.

**Responsible component: the level (intercept) of the per-100-possession rate and make-rate GLMs.** No adjustment is applied: the pre-registered Control feature list has no season-level term by design. The fix belongs upstream -- a season-aware level term (the as-of league mean is already produced by `own_ratings`, is pregame and is leak-safe) or a preseason refit, both of which are L2/L3 decisions, not Control patches. SIM_GUARDRAILS section 4 predicted exactly this: *any model trained on pooled seasons without season-aware features will under-shoot the current year*.

### D3. G9 bias by tier -- partly selection, partly real

Team tiers are terciles of each team's OWN full-season margin, which `docs/tests/gate_reference_2026-09-10.md` flags as not leak-free. Grouping on a quantity computed from the outcomes themselves guarantees some regression-to-the-mean bias for ANY pregame predictor. The control is to run the same breakdown on the closing line:

| home_tier | n | model_margin_bias | close_margin_bias | model_minus_close |
|---|---|---|---|---|
| bottom_tercile | 1501 | 1.9571 | 1.3175 | 0.6397 |
| middle_tercile | 1705 | 0.4592 | 0.1135 | 0.3457 |
| top_tercile | 2169 | -1.4931 | -1.4009 | -0.0922 |

The closing line shows the same sign and most of the same magnitude, so most of the tier-wise bias is the tier definition, not the engine. The `model_minus_close` column is the part the engine actually owns.

### D4. G6 neutral-site home margin

Simulated +2.229 vs actual +3.288 on 736 neutral games. The site term is a single `site_home`/`site_away` pair with neutral as the reference level, so every neutral game gets exactly zero site effect and the residual gap has to come from the ratings. In reality *neutral* covers a wide range -- an NCAA sub-regional in a team's home state, an in-season tournament in a team's own market -- and the nominal home team at a neutral site is systematically the stronger or higher-seeded one. **Responsible component: the site feature, which is too coarse.** The non-neutral cell passes (+0.326), so the home effect itself is wired correctly in every scoring-stage model; it is the neutral bucket that is heterogeneous. A venue-distance or designated-home feature is the honest fix and belongs to the feature layer, not to a post-hoc neutral-site offset.

### D5. G1 possessions -- the overtime stub

Sim 68.268 vs actual 67.875 possessions per game (+0.393) passes, but the sign is explained: the pace target is the OBSERVED possession count, which already contains whatever overtime a real game played, and the stub then adds a further 5/40 of a game to every tied sim. That double count is worth about +0.16 possessions per game at the simulated OT rate of 0.0183 (actual OT rate on these games: 0.0558). The simulated OT rate is itself too low because a Beta-Binomial / NegBin score has no end-game mechanics to pile probability mass onto an exact tie. Both are the known L5 gap, listed in model.md section 9 and left alone.

## 11. Gate summary

| gate | quantity | value | tolerance | status |
|---|---|---|---|---|
| G1 | possessions/game mean | 68.268 vs 67.875 | +/-1.0 | PASS |
| G1 | possessions/game SD | 5.702 vs 5.474 | +/-0.75 | PASS |
| G1 | by month (mean and SD) | 0/5 powered months out | all inside | PASS |
| G5 | margin SD ratio | 1.6782 | 0.95-1.05 | FAIL |
| G5 | total SD ratio | 1.3179 | 0.95-1.05 | FAIL |
| G5 | home/away score corr | 0.0836 vs 0.2285 | +/-0.05 | FAIL |
| G5 | PIT K-S p | 2.38e-79 | > 0.1 | FAIL |
| G6 | home margin non-neutral | +6.058 vs +5.732 | +/-1.0 | PASS |
| G6 | home margin neutral | +2.229 vs +3.288 | +/-1.0 | FAIL |
| G9 | margin bias | +0.1472 | +/-0.5 | PASS |
| G9 | total bias | -2.8263 | +/-1.0 | FAIL |
| G9 | calibration slope | 0.9595 | 0.95-1.05 | PASS |
| G9 | responsiveness slope | 0.9765 | monotone, 0.85-1.15 | PASS |
| G9 | bias by month | 8/10 scored cells out | all inside | FAIL |
| G9 | bias by tier | 6/6 scored cells out | all inside | FAIL |
| G9 | bias by pred_total_tercile | 3/6 scored cells out | all inside | FAIL |
| G10 | 2024 margin MAE vs close | 9.233 vs 8.870 | report only | PASS |
| G10 | 2024 leak screen | surprise corr 0.010 | <= 0.15 | PASS |
| G10 | 2025 margin MAE vs close | 9.106 vs 8.746 | report only | PASS |
| G10 | 2025 leak screen | surprise corr -0.013 | <= 0.15 | PASS |

