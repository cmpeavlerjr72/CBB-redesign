# Early-season total bias, served v2 vs clock K2 (2026-10-07)

Diagnostic only. Same method and game set as `docs/tests/total_bias_decomp_2026-10-05.md` (`scripts/diag_total_bias_decomp_v1.py`, imported unchanged; a thin wrapper
ran both arms, 5,551 F1 / 5,589 F2 games with event shot classes, truth = verified finals). Buckets d0-14 / d15-45 / d46+. Bias = sim mean total minus actual, points.
Arms: F1 `f1c_V2_full_s200_o0` vs `f1c_K2_full_s200_o0`; F2 `v3full_COMB9GCTKD_s200_o0` (the served-v2 read used by the decomp doc) vs `laneH_v3full_K2_s200_o0`.
The v2 columns reproduce the decomp doc exactly.

## Total and margin bias

SE(bias) is the across-game SE (dominated by real game noise). SE(K2-v2) is the paired per-game difference SE (sim noise only), so the arm difference is far better resolved than either level.

| fold | bucket | n games | v2 total bias | K2 total bias | SE(bias) | K2 - v2 (SE) | v2 margin bias | K2 margin bias | SE(margin) |
|---|---|---|---|---|---|---|---|---|---|
| F1 | all | 5551 | -1.72 | -1.47 | 0.23 | +0.25 (0.02) | +0.05 | +0.05 | 0.16 |
| F1 | d0-14 | 657 | -5.19 | -5.22 | 0.68 | -0.03 (0.05) | -0.60 | -0.67 | 0.48 |
| F1 | d15-45 | 1171 | -2.08 | -1.85 | 0.51 | +0.23 (0.03) | -0.10 | -0.12 | 0.34 |
| F1 | d46+ | 3723 | -1.00 | -0.69 | 0.28 | +0.31 (0.02) | +0.21 | +0.23 | 0.19 |
| F2 | all | 5589 | -0.29 | -0.08 | 0.23 | +0.21 (0.02) | -0.22 | -0.22 | 0.16 |
| F2 | d0-14 | 572 | -4.20 | -4.46 | 0.76 | -0.26 (0.05) | -0.50 | -0.56 | 0.54 |
| F2 | d15-45 | 1190 | -0.37 | -0.31 | 0.48 | +0.06 (0.03) | -0.61 | -0.59 | 0.35 |
| F2 | d46+ | 3827 | +0.32 | +0.64 | 0.27 | +0.32 (0.02) | -0.05 | -0.05 | 0.18 |

Underpowered: all margin cells in d0-14 and d15-45 (|bias| below 2 SE except F2 d15-45 at 1.7 SE); the d0-14 total-bias levels carry SE 0.7, so v2 vs K2 levels are not separable, only the paired difference is.

## Channels (Shapley, points per game, d0-14; v2 / K2)

| channel | F1 d0-14 | F2 d0-14 |
|---|---|---|
| pace (P) | -0.23 / -0.57 | -0.28 / -0.91 |
| TOV | -2.12 / -2.18 | -0.52 / -0.57 |
| OREB | -0.54 / -0.65 | -0.12 / -0.23 |
| FTA rate | -0.99 / -0.91 | -0.77 / -0.70 |
| FT make | -1.08 / -1.07 | -1.37 / -1.36 |
| rim share | -0.62 / -0.56 | -0.42 / -0.36 |
| rim make | +0.55 / +0.69 | -0.21 / -0.07 |
| jump2 make | -0.51 / -0.44 | -0.70 / -0.61 |
| 3 make | +0.19 / +0.31 | +0.14 / +0.28 |

K2 moves possession count down in the opening window (-0.3 F1, -0.6 F2 points), offsetting small make-rate gains. FT make, FTA rate and TOV (the owners named in the decomp doc) are unchanged.
Later buckets: K2 adds about +0.3 points at d46+ in both folds (pace and make-rate channels), moving the full-season bias toward zero.

## Per-team evidence, d0-14 (teams with at least 2 window games; team total bias = mean bias of the games it played)

| fold | arm | teams | share with negative bias | quintile means by prior-season scoring env, Q1 (low) to Q5 (high) | slope (pts per prior total pt), corr |
|---|---|---|---|---|---|
| F1 | v2 | 358 | 70.1% | -3.72 -4.46 -5.33 -7.62 -6.31 | -0.165, -0.130 |
| F1 | K2 | 358 | 69.8% | -3.63 -4.45 -5.46 -7.74 -6.36 | -0.171, -0.135 |
| F2 | v2 | 354 | 65.8% | -3.79 -4.30 -3.60 -4.21 -5.25 | -0.075, -0.050 |
| F2 | K2 | 354 | 66.9% | -3.98 -4.46 -4.10 -4.40 -5.54 | -0.077, -0.052 |

About 71 teams per quintile, quintile SE 1.0-1.5 points (each cell is individually underpowered). The gap is broad: roughly two thirds of teams are negative, and every quintile in both folds is negative.
It is not concentrated in low-prior teams; if anything it is larger for high-scoring-environment teams (slope negative, F1 mild, F2 about 1 SE across the range and not distinguishable from flat).
The prior used is last season's scoring environment, not roster continuity; an anonymous-roster split was not cut here.

## Read

K2 leaves the d0-14 total gap unchanged to slightly worse (F1 -0.03, F2 -0.26 points, the F2 shift is 5 paired SE but under 0.4 of the cross-game SE and via lower pace); K2 shrinks the gap only from d15 onward (+0.2 to +0.3), so the launch-window shortfall of about 4 to 5 points stays with the FT, FTA, TOV and shot-class owners, not the clock.
