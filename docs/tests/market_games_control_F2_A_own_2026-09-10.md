# Market scorecard (G10) -- F2_A_own (season 2025)

Generated 2026-09-10 by `scripts/grade_market_games.py`. Results: `results/control/F2_A_own`. Lines: `data/raw/cbbd/lines_2025.parquet`, provider preference DraftKings > ESPN BET > Bovada > consensus (rows used, by provider: {'ESPN BET': 5375}). `spread` is home-perspective, so the market's expected home margin is `-spread`. G10 is report-only except the leak screen (LEAK-SUSPECT if surprise corr > 0.15 AND CLV agreement < 0.53). SETTLEMENT is always at the real posted line/odds; de-vig is used only to build the probability for Brier/calibration.

n games with a usable spread line: 5375

## Margin and total: model vs close

| n | model_margin_MAE | close_margin_MAE | model_margin_bias | close_margin_bias | model_total_MAE | close_total_MAE | model_total_bias | close_total_bias |
|---|---|---|---|---|---|---|---|---|
| 5375 | 9.1058 | 8.7460 | 0.0897 | -0.1614 | 13.2181 | 12.6799 | -2.8457 | -0.5623 |

Margin: model MAE 9.1058 vs close 8.7460 (model trails the close by 0.3598) -- report-only, PASS

## ATS by disagreement bucket (settled at the real spread, -110)

| bucket | n | wins | losses | pushes | win_pct | roi_at_-110 |
|---|---|---|---|---|---|---|
| >= 1 | 4011 | 2002 | 2009 | 0 | 0.4991 | -0.0518 |
| >= 2 | 2775 | 1395 | 1380 | 0 | 0.5027 | -0.0443 |
| >= 3 | 1777 | 886 | 891 | 0 | 0.4986 | -0.0530 |
| >= 5 | 642 | 318 | 324 | 0 | 0.4953 | -0.0598 |

ATS >= 1 pt bootstrap (1000 reps, 4011 games): mean ROI -0.0511, 95% CI [-0.0833, -0.0183], P(ROI<=0) = 0.997

## OU by disagreement bucket (settled at the real total, -110)

| bucket | n | wins | losses | pushes | win_pct | roi_at_-110 |
|---|---|---|---|---|---|---|
| >= 1 | 4382 | 2152 | 2230 | 0 | 0.4911 | -0.0687 |
| >= 2 | 3483 | 1724 | 1759 | 0 | 0.4950 | -0.0606 |
| >= 3 | 2617 | 1268 | 1349 | 0 | 0.4845 | -0.0825 |
| >= 5 | 1301 | 638 | 663 | 0 | 0.4904 | -0.0702 |

OU >= 1 pt bootstrap (1000 reps, 4382 games): mean ROI -0.0688, 95% CI [-0.1003, -0.0371], P(ROI<=0) = 1.000

## Moneyline edge buckets (settled at REAL posted odds; edge = model P(home) - de-vigged market P(home))

| edge_bucket | n | win_pct | roi |
|---|---|---|---|
| [0.000, 0.025) | 856 | 0.4755 | -0.0428 |
| [0.025, 0.050) | 908 | 0.3678 | -0.1603 |
| [0.050, 0.100) | 1686 | 0.2877 | -0.1632 |
| [0.100, 1.000) | 1772 | 0.1986 | -0.2420 |

ML edge (all buckets pooled) bootstrap (1000 reps, 5222 games): mean ROI -0.1702, 95% CI [-0.2129, -0.1261], P(ROI<=0) = 1.000

## Brier vs de-vigged moneyline (n = 5222, mean vig 0.0431)

model 0.18818 vs market 0.17536 -- report-only, PASS

### Calibration deciles (model P(home) vs de-vigged market P(home) vs actual)

| decile | n | model_p | market_p | actual_rate | model_delta | market_delta |
|---|---|---|---|---|---|---|
| 1 | 549 | 0.3133 | 0.2486 | 0.1985 | -0.1148 | -0.0500 |
| 2 | 514 | 0.4285 | 0.4050 | 0.4280 | -0.0005 | 0.0231 |
| 3 | 504 | 0.4889 | 0.4877 | 0.4841 | -0.0048 | -0.0035 |
| 4 | 551 | 0.5339 | 0.5508 | 0.5808 | 0.0469 | 0.0300 |
| 5 | 545 | 0.5799 | 0.6068 | 0.6000 | 0.0201 | -0.0068 |
| 6 | 483 | 0.6218 | 0.6694 | 0.7101 | 0.0883 | 0.0408 |
| 7 | 517 | 0.6643 | 0.7212 | 0.7389 | 0.0745 | 0.0176 |
| 8 | 522 | 0.7122 | 0.7942 | 0.8238 | 0.1116 | 0.0295 |
| 9 | 543 | 0.7708 | 0.8620 | 0.8803 | 0.1095 | 0.0183 |
| 10 | 494 | 0.8555 | 0.9317 | 0.9636 | 0.1081 | 0.0319 |

## Leak screen

surprise corr -0.0129 (gate 0.15), CLV sign agreement 0.5300 on 1085 moved lines -- **PASS**

