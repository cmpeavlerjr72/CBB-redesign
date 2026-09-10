# Gate report -- F2_A_own (season 2025)

Generated 2026-09-10 by `scripts/eval_gates.py`. Results: `results/control/F2_A_own`. Tolerances: `docs\gates.yaml` (provisional until the seed-noise study, `docs/SIM_GUARDRAILS.md` section 3). Every gate line ends in a literal PASS / FAIL / NEEDS-INSTRUMENTATION.

Games graded: 5700 (of 5700 in the run; the difference is games the truth tables exclude as non-D-I or pbp-truncated, or that the run itself does not cover). Seeds per game: 200. run_meta: fold=F2, backtest=True, sealed_touched=False.

## Contract notices

- run_meta.json missing 'engine_tag'; defaulted to results dir name 'F2_A_own' (pre-contract Control run_meta.json).
- run_meta.json missing 'sealed_touched'; defaulted to False (pre-contract Control run_meta.json).
- games.parquet has no 'n_periods' column; derived it from the legacy 'n_ot' column as n_periods = 2 + n_ot (pre-contract Control shape).
- players.parquet not found; player-level gates (G8) and grade_market_props.py will report NEEDS-INSTRUMENTATION.
- games.parquet missing optional box column pair(s) for: fga3, fga2_rim, fga2_jump, fta, tov, oreb, dreb; the gates that need them report NEEDS-INSTRUMENTATION.

## G1 -- Possessions per game, mean and SD (overall, by month)

| quantity | value | target | tolerance | status |
|---|---|---|---|---|
| possessions/game mean | 68.268 vs 67.875 | 67.875 | +/-1.0 | PASS |
| possessions/game SD | 5.702 vs 5.474 | 5.474 | +/-0.75 | PASS |
| by month (mean and SD) | 5/5 powered months inside | all inside | see per-month table | PASS |

1 month(s) below n=300 labelled UNDERPOWERED and not scored.

### by breakdown

| breakdown | group | n_games | sim_mean | actual_mean | ref_mean | sim_sd | actual_sd | ref_sd | d_mean | d_sd | status_mean | status_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| season | all | 5700 | 68.2680 | 67.8753 | 67.8753 | 5.7017 | 5.4741 | 5.4741 | 0.3927 | 0.2276 | PASS | PASS |
| month | 1 | 1420 | 68.1576 | 67.4339 | 67.4339 | 5.7201 | 5.4817 | 5.4817 | 0.7237 | 0.2384 | PASS | PASS |
| month | 2 | 1364 | 67.6774 | 67.2344 | 67.2344 | 5.7010 | 5.1604 | 5.1604 | 0.4430 | 0.5406 | PASS | PASS |
| month | 3 | 765 | 67.3872 | 67.4826 | 67.4826 | 5.7739 | 5.3518 | 5.3518 | -0.0954 | 0.4220 | PASS | PASS |
| month | 4 | 17 | 66.1201 | 69.1365 | 69.1365 | 5.7613 | 5.6686 | 5.6686 | -3.0164 | 0.0927 | UNDERPOWERED | UNDERPOWERED |
| month | 11 | 1219 | 69.2232 | 69.0697 | 69.0697 | 5.5223 | 5.5818 | 5.5818 | 0.1535 | -0.0595 | PASS | PASS |
| month | 12 | 915 | 68.8236 | 68.2291 | 68.2291 | 5.6011 | 5.5994 | 5.5994 | 0.5945 | 0.0017 | PASS | PASS |

**G1 overall: PASS**

## G2 -- Points per possession, by offense tercile x defense tercile

| quantity | value | target | tolerance | status |
|---|---|---|---|---|
| PPP by offense x defense tercile (9 cells) | 3/9 powered cells inside +/-0.02 | all inside | +/-0.02 per cell | FAIL |

Terciles are each team's OWN season points-scored / points-allowed average (`reference.team_quality_terciles`), grading-only, never a model feature.

### by tercile cell

| offense_tercile | defense_tercile | n_team_games | sim_ppp | actual_ppp | delta | status |
|---|---|---|---|---|---|---|
| bottom_tercile | bottom_tercile | 1275 | 0.9680 | 0.9659 | 0.0021 | PASS |
| bottom_tercile | middle_tercile | 1247 | 1.0019 | 1.0139 | -0.0120 | PASS |
| bottom_tercile | top_tercile | 1147 | 1.0348 | 1.0584 | -0.0236 | FAIL |
| middle_tercile | bottom_tercile | 1254 | 1.0059 | 1.0189 | -0.0131 | PASS |
| middle_tercile | middle_tercile | 1298 | 1.0453 | 1.0737 | -0.0284 | FAIL |
| middle_tercile | top_tercile | 1262 | 1.0725 | 1.1189 | -0.0464 | FAIL |
| top_tercile | bottom_tercile | 1429 | 1.0564 | 1.0851 | -0.0287 | FAIL |
| top_tercile | middle_tercile | 1252 | 1.0966 | 1.1346 | -0.0380 | FAIL |
| top_tercile | top_tercile | 1236 | 1.1294 | 1.1908 | -0.0615 | FAIL |

**G2 overall: FAIL**

## G3 -- Shot mix per possession: 3PA share, rim share, FTA/FGA (by team)

| quantity | value | target | tolerance | status |
|---|---|---|---|---|
| shot mix by team | n/a | n/a | +/-1.5pp | NEEDS-INSTRUMENTATION |

games.parquet lacks optional box column pair(s) for: fga3, fga2_rim, fga2_jump, fta.

**G3 overall: NEEDS-INSTRUMENTATION**

## G4 -- Four factors, offense and defense (by team, by tier)

| quantity | value | target | tolerance | status |
|---|---|---|---|---|
| eFG% (offense/defense) | n/a | n/a | +/-1.0pp | NEEDS-INSTRUMENTATION |
| TOV% / OREB% / FT rate | n/a | n/a | see SIM_GUARDRAILS G4 row | NEEDS-INSTRUMENTATION |

eFG% needs MAKE counts (FGM/3PM); the contract's optional box columns are attempt counts only, so eFG% is unconditionally NEEDS-INSTRUMENTATION under the current contract (a documented gap, not a missing-file accident).

games.parquet also lacks optional box column pair(s) for: fga3, fga2_rim, fga2_jump, fta, tov, oreb, dreb.

**G4 overall: NEEDS-INSTRUMENTATION**

## G5 -- Dispersion: SD ratio, score correlation, PIT histogram

| quantity | value | target | tolerance | status |
|---|---|---|---|---|
| margin SD ratio | 1.6782 | 1.0 | 0.95-1.05 | FAIL |
| total SD ratio | 1.3179 | 1.0 | 0.95-1.05 | FAIL |
| home/away score correlation | 0.0836 vs 0.2285 | 0.2285 | +/-0.05 | FAIL |
| PIT K-S p | 2.38e-79 | > 0.1 | > 0.1 | FAIL |

SD ratio = mean(sim SD) / SD(actual - sim mean); >1 means the engine is too wide.

PIT K-S statistic D = 0.1260.

### SD ratio

| quantity | mean_sim_SD | SD(actual - sim mean) | ratio | status |
|---|---|---|---|---|
| margin | 19.6215 | 11.6919 | 1.6782 | FAIL |
| total | 22.0260 | 16.7133 | 1.3179 | FAIL |

### PIT decile histogram

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

**G5 overall: FAIL**

## G6 -- Home margin, non-neutral vs neutral, same games

| quantity | value | target | tolerance | status |
|---|---|---|---|---|
| home margin (non-neutral) | +6.058 vs +5.732 | +5.732 | +/-1.0 | PASS |
| home margin (neutral) | +2.229 vs +3.288 | +3.288 | +/-1.0 | FAIL |

### by site

| site | n | sim | actual | delta | status |
|---|---|---|---|---|---|
| non-neutral | 4964 | 6.0577 | 5.7317 | 0.3260 | PASS |
| neutral | 736 | 2.2295 | 3.2880 | -1.0586 | FAIL |

**G6 overall: FAIL**

## G7 -- Overtime rate; first-half vs second-half scoring share

| quantity | value | target | tolerance | status |
|---|---|---|---|---|
| OT rate | 0.0183 vs 0.0558 | 0.0558 | +/-1.0pp | FAIL |
| first/second half scoring share | n/a (contract has no per-period sim score) | 1H 0.4731 / 2H 0.5269 | +/-1.0pp | NEEDS-INSTRUMENTATION |

games.parquet carries only a whole-game home_pts/away_pts; a per-half score column is not in the contract, so the half-share leg is always NEEDS-INSTRUMENTATION.

**G7 overall: FAIL**

## G8 -- Player layer: minutes, usage share, distribution tails

| quantity | value | target | tolerance | status |
|---|---|---|---|---|
| player layer | n/a | n/a | see SIM_GUARDRAILS G8 row | NEEDS-INSTRUMENTATION |

players.parquet not found for this engine.

**G8 overall: NEEDS-INSTRUMENTATION**

## G9 -- Spread and total accuracy: MAE, signed bias, calibration slope

| quantity | value | target | tolerance | status |
|---|---|---|---|---|
| margin bias | +0.1472 | 0 | +/-0.5 | PASS |
| total bias | -2.8263 | 0 | +/-1.0 | FAIL |
| calibration slope | 0.9595 | 1.0 | 0.95-1.05 | PASS |
| bias by month | 8/10 scored cells outside tolerance | all inside | see per-breakdown table | FAIL |
| bias by tier | 6/6 scored cells outside tolerance | all inside | see per-breakdown table | FAIL |
| bias by pred_total_tercile | 3/6 scored cells outside tolerance | all inside | see per-breakdown table | FAIL |

margin MAE 9.1462, total MAE 13.2038, win-probability Brier 0.18568, slope_MC_corrected 0.9820 (MC noise SD 1.390, a diagnostic, not the gate).

### win-probability calibration by decile

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

### by month

| month | n | margin_mae | margin_bias | total_mae | total_bias | slope | status_margin_bias | status_total_bias |
|---|---|---|---|---|---|---|---|---|
| 1 | 1420 | 8.9973 | 1.3810 | 13.4259 | -1.9151 | 0.8907 | FAIL | FAIL |
| 2 | 1364 | 8.8231 | 0.8219 | 13.0943 | -2.3406 | 0.8657 | FAIL | FAIL |
| 3 | 765 | 8.7496 | -0.4818 | 12.7988 | -2.7861 | 0.9356 | PASS | FAIL |
| 4 | 17 | 8.6397 | 1.7815 | 15.8868 | -10.2144 | n/a | UNDERPOWERED | UNDERPOWERED |
| 11 | 1219 | 10.0355 | -1.5914 | 13.8971 | -4.1075 | 1.0128 | FAIL | FAIL |
| 12 | 915 | 9.0152 | 0.0385 | 12.3876 | -3.1538 | 0.9573 | PASS | FAIL |

### by tier

| home_tier | n | margin_mae | margin_bias | total_mae | total_bias | slope | status_margin_bias | status_total_bias |
|---|---|---|---|---|---|---|---|---|
| bottom_tercile | 1604 | 9.0695 | 1.8449 | 13.1957 | -3.0240 | 0.8060 | FAIL | FAIL |
| middle_tercile | 1827 | 8.9586 | 0.5145 | 13.5386 | -2.3564 | 0.7866 | FAIL | FAIL |
| top_tercile | 2269 | 9.3514 | -1.3486 | 12.9400 | -3.0649 | 0.9238 | FAIL | FAIL |

### by pred_total_tercile

| pred_total_tercile | n | margin_mae | margin_bias | total_mae | total_bias | slope | status_margin_bias | status_total_bias |
|---|---|---|---|---|---|---|---|---|
| bottom_tercile | 1900 | 8.9604 | 0.3179 | 12.5696 | -3.4995 | 0.9551 | PASS | FAIL |
| middle_tercile | 1900 | 9.0839 | 0.1512 | 13.3195 | -2.6384 | 0.9552 | PASS | FAIL |
| top_tercile | 1900 | 9.3942 | -0.0274 | 13.7224 | -2.3409 | 0.9636 | PASS | FAIL |

**G9 overall: FAIL**

## Summary

| status | count |
|---|---|
| PASS | 1 |
| FAIL | 5 |
| NEEDS-INSTRUMENTATION | 3 |

| gate | status |
|---|---|
| G1 | PASS |
| G2 | FAIL |
| G3 | NEEDS-INSTRUMENTATION |
| G4 | NEEDS-INSTRUMENTATION |
| G5 | FAIL |
| G6 | FAIL |
| G7 | FAIL |
| G8 | NEEDS-INSTRUMENTATION |
| G9 | FAIL |

