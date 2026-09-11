# Market scorecard (G10) -- engine_v0/F2_2025_s5_r2event (season 2025)

Generated 2026-09-10 by `scripts/grade_market_games.py`. Results: `results/engine_v0/F2_2025_s5_r2event`. Lines: `data/raw/cbbd/lines_2025.parquet`, provider preference DraftKings > ESPN BET > Bovada > consensus (rows used, by provider: {'ESPN BET': 5383}). `spread` is home-perspective, so the market's expected home margin is `-spread`. G10 is report-only except the leak screen (LEAK-SUSPECT if surprise corr > 0.15 AND CLV agreement < 0.53). SETTLEMENT is always at the real posted line/odds; de-vig is used only to build the probability for Brier/calibration.

n games with a usable spread line: 5383

## Margin and total: model vs close

| n | model_margin_MAE | close_margin_MAE | model_margin_bias | close_margin_bias | model_total_MAE | close_total_MAE | model_total_bias | close_total_bias |
|---|---|---|---|---|---|---|---|---|
| 5383 | 16.0332 | 8.7419 | -1.8185 | -0.1636 | 14.4678 | 12.7297 | -0.1338 | -0.5182 |

Margin: model MAE 16.0332 vs close 8.7419 (model trails the close by 7.2913) -- report-only, PASS

## ATS by disagreement bucket (settled at the real spread, -110)

| bucket | n | wins | losses | pushes | win_pct | roi_at_-110 |
|---|---|---|---|---|---|---|
| >= 1 | 5117 | 2575 | 2542 | 0 | 0.5032 | -0.0432 |
| >= 2 | 4862 | 2448 | 2414 | 0 | 0.5035 | -0.0427 |
| >= 3 | 4605 | 2322 | 2283 | 0 | 0.5042 | -0.0411 |
| >= 5 | 4089 | 2061 | 2028 | 0 | 0.5040 | -0.0415 |

ATS >= 1 pt bootstrap (1000 reps, 5117 games): mean ROI -0.0434, 95% CI [-0.0724, -0.0153], P(ROI<=0) = 1.000

## OU by disagreement bucket (settled at the real total, -110)

| bucket | n | wins | losses | pushes | win_pct | roi_at_-110 |
|---|---|---|---|---|---|---|
| >= 1 | 4844 | 2364 | 2480 | 0 | 0.4880 | -0.0751 |
| >= 2 | 4367 | 2131 | 2236 | 0 | 0.4880 | -0.0752 |
| >= 3 | 3908 | 1895 | 2013 | 0 | 0.4849 | -0.0817 |
| >= 5 | 2906 | 1398 | 1508 | 0 | 0.4811 | -0.0897 |

OU >= 1 pt bootstrap (1000 reps, 4844 games): mean ROI -0.0759, 95% CI [-0.1038, -0.0461], P(ROI<=0) = 1.000

## Moneyline edge buckets (settled at REAL posted odds; edge = model P(home) - de-vigged market P(home))

| edge_bucket | n | win_pct | roi |
|---|---|---|---|
| [0.000, 0.025) | 300 | 0.5000 | -0.0373 |
| [0.025, 0.050) | 326 | 0.5307 | -0.0510 |
| [0.050, 0.100) | 674 | 0.4659 | -0.1006 |
| [0.100, 1.000) | 3930 | 0.3168 | -0.1596 |

ML edge (all buckets pooled) bootstrap (1000 reps, 5230 games): mean ROI -0.1380, 95% CI [-0.1786, -0.1010], P(ROI<=0) = 1.000

## Brier vs de-vigged moneyline (n = 5230, mean vig 0.0431)

model 0.27145 vs market 0.17540 -- report-only, PASS

### Calibration deciles (model P(home) vs de-vigged market P(home) vs actual)

| decile | n | model_p | market_p | actual_rate | model_delta | market_delta |
|---|---|---|---|---|---|---|
| 1 | 905 | 0.1644 | 0.5522 | 0.5580 | 0.3936 | 0.0058 |
| 2 | 1392 | 0.4000 | 0.6029 | 0.5941 | 0.1941 | -0.0088 |
| 3 | 1649 | 0.6000 | 0.6342 | 0.6592 | 0.0592 | 0.0250 |
| 4 | 999 | 0.8000 | 0.6768 | 0.6937 | -0.1063 | 0.0169 |
| 5 | 285 | 1.0000 | 0.7329 | 0.7789 | -0.2211 | 0.0460 |

## Leak screen

surprise corr 0.0017 (gate 0.15), CLV sign agreement 0.5405 on 1086 moved lines -- **PASS**

