# Heuermann close-bucket compression and the yardstick test, on our served engine (fold 2, 2024-25), 2026-11-11

DIAGNOSTIC ONLY. Nothing changed. Script `scripts/diag_heuermann_jqmk2_closegrade_v1.py` (tables also in `results/heuermann_jqmk2_closegrade/`, gitignored). Fold 2 (season 2024-25) only; 2025-26 and 2026-27 not read.

## Inputs

- Sim: served v3 stack (clock K2, control of clock round 10), **200 seeds** x 5,710 games: `results/clock_env10/full/clk10full_ctrl_o0_s200/games.parquet` (the largest on disk; the 50-seed extract `results/diag_c4/f2all50_*` was the 10-09 grading input). No new sim was run.
- Truth: verified finals (`fin_h`, `fin_a` in `results/diag_c4/f2all50_pregame.parquet`, the same verified finals the 10-09 grading used; OT included). Closing spread: same file, ESPN BET close from `lines_close_v2_verified`. **Deviation:** `data/processed/lines/lines_open_close_v1.parquet` holds only the live 2026-27 captures (893 rows), so it has no fold-2 closes and was not used.
- n games 5,710; n lined games 5,380 (pick-em spread 0 -> home treated as favourite). Tests 1, 3, 4 use lined games; test 2 uses the lined games too (all-games IV slope is given at the end).
- Seed-noise handling: odd/even-seed instrument (100 seeds each) for every slope with the sim mean as regressor; 200-seed mean noise is small (var 0.75 of 55.4 in fav orientation). Bootstrap: 2,000 game-level resamples (the 3b bootstrap redoes the cross-fit every resample).
- Experience split: in-season prior games per team counted from `games_universe.parquet` (season 2025 rows only, so early-season games count low; this is a cheap proxy, not a cross-season count). Cells under 150 games are labelled UNDERPOWERED and not read.
- NOT RUN: the first-half / second-half split of the 15-20 and 20+ gaps. The 200-seed output and the cached extracts carry no points by half (only final points, possession and duration by half), so it cannot be derived without re-simulating.

## Summary (plain)

1. (A) Yes, in the top bucket. Close 20+: sim - real favourite margin -3.29 [-4.73, -1.95], n = 403, compression present by the fixed rule. 15-20: -0.95 [-2.12, +0.26], n = 349, not present (CI includes 0, |gap| < 1). All buckets under 15: no.
2. The 20+ gap is all favourite points (-3.25 [-4.39, -2.13]); underdog points are unbiased (+0.04). Total is low by the same amount (-3.22): the engine under-scores for heavy favourites, not an underdog-side effect.
3. Slope of sim margin on close 0.944 [0.929, 0.960] vs real 1.051 [1.005, 1.098] (Heuermann: 0.79 vs 1.03). Ours is much milder than his.
4. By our own prediction (test 2, no market): real - sim is within +/-0.8 in every |s| bucket, +0.13 at 15-20 and +0.18 at 20+ (CIs include 0, wide). IV slope of real on sim 0.962 [0.926, 0.997] in home orientation, 1.006 in sim-favourite orientation. So the engine is roughly calibrated against its own predictions; the top-bucket gap appears only when bucketing on the market.
5. (B) The yardstick is biased in the sense the PM described, but only partly explains our gap. Under nested information (calibrated sim knowing less than a calibrated market) the home-orientation slope of s on the close would be var(s)/var(m) = 0.975 [0.954, 0.997]; measured 0.927 [0.915, 0.938]. So about a third of the shortfall from 1 is explained by information deficit; the rest is cov(error, close) > 0 (the real margin tracks the close more than the sim does). In fav orientation the nested prediction is 1.10 (above 1) because the sim mean has more spread than the close; measured 0.944.
6. Constructed calibrated s* (cross-fit, slope of r on s* = 0.998): 20+ gap -3.71 [-4.76, -2.76], 15-20 -1.17 [-2.33, -0.05]. By the pre-registered rule claim (B) is SHOWN (a calibrated predictor still shows a negative gap at 20+ and 15-20). Read it carefully: calibration removed none of our 20+ gap (it grew from -3.29 to -3.71), so for us the gap is not something a slope correction would fix.
7. (3c) Stretching s by k = 1.113 to match the real slope on the close gives IV slope of r on it 0.803 (over-spread by 20%) and margin MAE 9.233 vs 9.130 for s (paired +0.103 [+0.048, +0.169]): it costs accuracy. s* MAE 9.122 (-0.009, n.s.). The close itself: 8.743 (-0.387 [-0.468, -0.308] vs s).
8. Experienced vs rest: experienced games show no compression in any powered bucket (15-20 and 20+ cells are n = 85 and 23, UNDERPOWERED, not read); the 20+ gap sits in the rest group (-3.39 [-4.83, -1.91], n = 380), which is mostly the early season and mismatch games.
9. Not proven: that the 20+ gap is a sub-model defect (it could be the market's information, the early-season rating, or neither); the cause is not located here. Half split not run.
10. Nothing was tuned or adjusted; these are measurements.

# Tables

Sim: served v3, 200 seeds, 5710 games; lined games 5380. Truth: verified finals (OT included). Margins are favourite-minus-underdog; favourite from the close. Bootstrap 2000 game-level resamples.

Experienced (both teams >= 20 prior in-season games in games_universe): 2020 lined games; rest 3360 (games with an unknown prior count are in the rest: 0).

### ALL lined games: n lined games = 5380

| close bucket | n | mean close | real fav margin | sim - real fav margin [95% CI] (SE) | compression? | total sim - real [95%] (SE) | fav pts sim - real [95%] | dog pts sim - real [95%] |
|---|---:|---:|---:|---|---|---|---|---|
| 0-3 | 1222 | 1.84 | +1.55 | -0.36 [-0.97, +0.23] (0.31) | no | +0.02 [-0.94, +1.00] (0.50) | -0.17 [-0.72, +0.39] | +0.19 [-0.39, +0.78] |
| 3-6 | 1382 | 4.42 | +3.85 | +0.10 [-0.54, +0.73] (0.33) | no | +0.87 [-0.04, +1.79] (0.48) | +0.48 [-0.06, +1.04] | +0.39 [-0.20, +0.95] |
| 6-10 | 1243 | 7.81 | +8.15 | -0.39 [-1.00, +0.26] (0.32) | no | +0.20 [-0.71, +1.15] (0.47) | -0.10 [-0.63, +0.45] | +0.30 [-0.30, +0.87] |
| 10-15 | 781 | 12.31 | +12.55 | -0.32 [-1.09, +0.49] (0.40) | no | -0.85 [-2.04, +0.31] (0.61) | -0.58 [-1.29, +0.10] | -0.26 [-1.02, +0.45] |
| 15-20 | 349 | 17.14 | +17.50 | -0.95 [-2.12, +0.26] (0.62) | no | -1.38 [-3.18, +0.38] (0.90) | -1.17 [-2.24, -0.10] | -0.22 [-1.34, +0.86] |
| 20+ | 403 | 26.84 | +27.53 | -3.29 [-4.73, -1.95] (0.71) | COMPRESSION | -3.22 [-4.69, -1.56] (0.80) | -3.25 [-4.39, -2.13] | +0.04 [-0.92, +1.03] |

Slope of sim fav margin on close: 0.944 [0.929, 0.960]; slope of real fav margin on close: 1.051 [1.005, 1.098]. (Regressor is the close, no seed-noise issue.)

Test 2 (ALL lined games): buckets on |s| (favourite by the SIM), real - sim:

| |s| bucket | n | mean sim fav margin | mean real | real - sim [95% CI] (SE) |
|---|---:|---:|---:|---|
| 0-3 | 1347 | 1.48 | +1.34 | -0.14 [-0.76, +0.51] (0.32) |
| 3-6 | 1226 | 4.43 | +4.23 | -0.20 [-0.85, +0.44] (0.33) |
| 6-10 | 1152 | 7.78 | +7.38 | -0.41 [-1.05, +0.22] (0.32) |
| 10-15 | 865 | 12.31 | +11.54 | -0.76 [-1.49, +0.01] (0.39) |
| 15-20 | 413 | 17.20 | +17.33 | +0.13 [-0.96, +1.23] (0.57) |
| 20+ | 377 | 25.96 | +26.13 | +0.18 [-1.19, +1.61] (0.72) |

IV slope of real home margin on sim home margin (odd/even instrument): 0.962 [0.926, 0.997] (other direction 0.965); sim-favourite orientation: 1.006 [0.956, 1.057].

## Test 3a: algebra and numbers

Let e = r - s. Calibration means E[r | s] = s, so E[e | s] = 0 and cov(s, r) = var(s) (var(s) = noise-free variance of the sim mean).
cov(s, c) = cov(r - e, c) = cov(r, c) - cov(e, c). Calibration alone does not fix cov(e, c); an assumption is needed.
ASSUMPTION N (nested information): s = E[r | F_s], c = E[r | F_c], F_s a subset of F_c. Then E[c | F_s] = s, so cov(s, c) = var(s), and
the slope of s on c is var(s) / var(c) < 1 whenever the market has more variance than the sim (it knows more). Form B uses cov(r, c) in place of var(c) (does not assume c calibrated).
Seed noise: var(s) is measured as cov(s_odd, s_even), which is free of the independent seed noise.

| quantity | value [95%] |
|---|---|
| var_s_obs | 55.42 [52.53, 58.33] |
| var_noise | 0.75 [0.72, 0.78] |
| var_s_true | 54.67 [51.79, 57.58] |
| var_c | 49.60 [46.62, 52.65] |
| cov_sr | 49.03 [45.28, 52.64] |
| cov_rc | 52.15 [48.23, 56.09] |
| cov_sc | 46.84 [44.07, 49.70] |
| check calibration: cov(s,r) vs var(s_true) (diff) | -5.64 |
| IV slope of r on s (fav-by-close orientation) | 0.894 [0.849, 0.938] |
| real slope of r on c (= cov(r,c)/var(c)) | 1.051 [1.005, 1.096] |
| MEASURED slope of s on c | 0.944 [0.928, 0.960] |
| IMPLIED slope of s on c, form A: var(s_true)/var(c) | 1.102 [1.068, 1.138] |
| IMPLIED slope of s on c, form B: var(s_true)/cov(r,c) | 1.048 [0.992, 1.108] |

Same algebra in HOME orientation (regressor m = -close spread, signed; this is the orientation the cross-fit below uses):

| quantity | value [95%] |
|---|---|
| h_var_s_true | 86.88 [82.84, 90.87] |
| h_var_m | 89.11 [84.96, 93.29] |
| h_cov_sr | 83.71 [78.75, 88.50] |
| h_cov_rm | 90.58 [85.28, 95.82] |
| IV slope of r on s | 0.962 [0.925, 0.998] |
| real slope of r on m | 1.016 [0.982, 1.049] |
| MEASURED slope of s on m | 0.927 [0.915, 0.938] |
| IMPLIED slope of s on m (nested + m calibrated): var(s_true)/var(m) | 0.975 [0.954, 0.997] |

## Test 3b: constructed calibrated predictor s* (2-fold cross-fit, fitted on home margin, applied to fav orientation)

IV slope of r on s* = 0.998 [0.994, 1.001]. Gaps below are s* - real fav margin (bootstrap redoes the cross-fit each resample).

| close bucket | n | s* - real [95% CI] | compression-rule flag | s - real (test 1, for comparison) |
|---|---:|---|---|---|
| 0-3 | 1222 | -0.33 [-0.93, +0.28] | includes 0 | -0.36 |
| 3-6 | 1382 | +0.10 [-0.53, +0.73] | includes 0 | +0.10 |
| 6-10 | 1243 | -0.47 [-1.06, +0.13] | includes 0 | -0.39 |
| 10-15 | 781 | -0.44 [-1.19, +0.28] | includes 0 | -0.32 |
| 15-20 | 349 | -1.17 [-2.33, -0.05] | excludes 0 | -0.95 |
| 20+ | 403 | -3.71 [-4.76, -2.76] | excludes 0 | -3.29 |

Claim (B) by the pre-registered rule (s* gap CI excludes 0 at 15-20 or 20+): **SHOWN**.

## Test 3c: stretch s linearly (about 0, fav orientation) so its slope on c equals the real slope of r on c

Stretch factor k = real slope / sim slope = 1.113 [1.059, 1.166]. IV slope of r on the stretched predictor = 0.803 [0.781, 0.823] (= 0.898 x IV slope of r on s by construction; 1.0 would mean 'fixing' the close slope kept calibration).

## Test 3d: margin MAE vs real (fav orientation)

| predictor | MAE overall [95%] |
|---|---|
| s (sim mean) | 9.130 [8.936, 9.323] |
| s* (calibrated) | 9.122 [8.930, 9.321] |
| s stretched (3c) | 9.233 [9.026, 9.445] |
| c (close, zero HA offset) | 8.743 [8.558, 8.928] |

Paired differences (bootstrap), MAE(predictor) - MAE(s): s* -0.009 [-0.030, +0.012]; stretched +0.103 [+0.048, +0.169]; c -0.387 [-0.468, -0.308]

| close bucket | n | MAE s | MAE s* | MAE stretched | MAE c |
|---|---:|---:|---:|---:|---:|
| 0-3 | 1222 | 8.62 | 8.60 | 8.68 | 8.39 |
| 3-6 | 1382 | 9.28 | 9.28 | 9.38 | 8.95 |
| 6-10 | 1243 | 8.94 | 8.93 | 9.01 | 8.64 |
| 10-15 | 781 | 8.76 | 8.74 | 8.97 | 8.37 |
| 15-20 | 349 | 9.13 | 9.12 | 9.33 | 8.60 |
| 20+ | 403 | 11.46 | 11.49 | 11.52 | 10.24 |

Stretched-predictor gap (k*s - real) per bucket, for reference: 0-3 -0.23 [-0.86, +0.40]; 3-6 +0.54 [-0.18, +1.25]; 6-10 +0.49 [-0.26, +1.24]; 10-15 +1.07 [+0.14, +2.04]; 15-20 +0.92 [-0.37, +2.18]; 20+ -0.54 [-1.40, +0.33]

## Test 4: experienced vs rest

### EXPERIENCED (both teams >= 20 prior in-season games): n lined games = 2020

| close bucket | n | mean close | real fav margin | sim - real fav margin [95% CI] (SE) | compression? | total sim - real [95%] (SE) | fav pts sim - real [95%] | dog pts sim - real [95%] |
|---|---:|---:|---:|---|---|---|---|---|
| 0-3 | 537 | 1.81 | +1.77 | -0.65 [-1.53, +0.22] (0.44) | no | -0.14 [-1.58, +1.34] (0.75) | -0.39 [-1.21, +0.43] | +0.25 [-0.65, +1.13] |
| 3-6 | 550 | 4.37 | +3.53 | +0.42 [-0.50, +1.38] (0.49) | no | +0.90 [-0.51, +2.39] (0.75) | +0.66 [-0.13, +1.54] | +0.24 [-0.66, +1.15] |
| 6-10 | 559 | 7.78 | +8.28 | -0.40 [-1.27, +0.51] (0.46) | no | +0.25 [-1.07, +1.65] (0.69) | -0.08 [-0.88, +0.74] | +0.33 [-0.46, +1.14] |
| 10-15 | 266 | 12.23 | +13.04 | -0.48 [-1.73, +0.71] (0.64) | no | -0.18 [-2.17, +1.81] (1.02) | -0.33 [-1.52, +0.89] | +0.15 [-1.04, +1.31] |
| 15-20 UNDERPOWERED | 85 | 17.04 | +18.47 | -0.48 [-2.93, +1.90] (1.23) | n/a (underpowered) | +1.12 [-2.34, +4.36] (1.69) | +0.32 [-1.88, +2.47] | +0.80 [-1.13, +2.69] |
| 20+ UNDERPOWERED | 23 | 24.41 | +25.83 | -1.61 [-7.03, +4.16] (2.85) | n/a (underpowered) | -1.54 [-6.50, +3.56] (2.62) | -1.58 [-5.51, +2.23] | +0.04 [-3.65, +3.71] |

Slope of sim fav margin on close: 1.084 [1.054, 1.115]; slope of real fav margin on close: 1.132 [1.027, 1.236]. (Regressor is the close, no seed-noise issue.)

Test 2 (EXPERIENCED (both teams >= 20 prior in-season games)): buckets on |s| (favourite by the SIM), real - sim:

| |s| bucket | n | mean sim fav margin | mean real | real - sim [95% CI] (SE) |
|---|---:|---:|---:|---|
| 0-3 | 601 | 1.49 | +1.42 | -0.06 [-0.92, +0.84] (0.45) |
| 3-6 | 500 | 4.45 | +4.00 | -0.45 [-1.46, +0.55] (0.50) |
| 6-10 | 474 | 7.82 | +8.04 | +0.22 [-0.78, +1.19] (0.50) |
| 10-15 | 294 | 12.18 | +11.00 | -1.19 [-2.43, +0.00] (0.61) |
| 15-20 UNDERPOWERED | 108 | 17.08 | +17.05 | -0.04 [-2.04, +2.03] (1.04) |
| 20+ UNDERPOWERED | 43 | 23.39 | +22.77 | -0.62 [-4.48, +3.34] (1.98) |

IV slope of real home margin on sim home margin (odd/even instrument): 0.942 [0.873, 1.004] (other direction 0.951); sim-favourite orientation: 0.989 [0.886, 1.086].

### REST (at least one team < 20 prior games, or unknown): n lined games = 3360

| close bucket | n | mean close | real fav margin | sim - real fav margin [95% CI] (SE) | compression? | total sim - real [95%] (SE) | fav pts sim - real [95%] | dog pts sim - real [95%] |
|---|---:|---:|---:|---|---|---|---|---|
| 0-3 | 685 | 1.86 | +1.37 | -0.14 [-1.02, +0.71] (0.44) | no | +0.15 [-1.20, +1.49] (0.68) | +0.01 [-0.78, +0.77] | +0.15 [-0.67, +1.00] |
| 3-6 | 832 | 4.45 | +4.06 | -0.12 [-0.94, +0.73] (0.42) | no | +0.85 [-0.25, +1.97] (0.59) | +0.37 [-0.33, +1.06] | +0.48 [-0.19, +1.24] |
| 6-10 | 684 | 7.84 | +8.06 | -0.38 [-1.34, +0.56] (0.48) | no | +0.16 [-1.09, +1.45] (0.66) | -0.11 [-0.89, +0.71] | +0.27 [-0.49, +1.06] |
| 10-15 | 515 | 12.35 | +12.29 | -0.24 [-1.22, +0.71] (0.50) | no | -1.19 [-2.69, +0.25] (0.75) | -0.71 [-1.60, +0.11] | -0.48 [-1.37, +0.44] |
| 15-20 | 264 | 17.17 | +17.19 | -1.10 [-2.55, +0.32] (0.72) | no | -2.19 [-4.25, -0.05] (1.09) | -1.65 [-2.85, -0.42] | -0.54 [-1.86, +0.83] |
| 20+ | 380 | 26.99 | +27.63 | -3.39 [-4.83, -1.91] (0.75) | COMPRESSION | -3.32 [-4.97, -1.68] (0.85) | -3.35 [-4.56, -2.19] | +0.04 [-1.04, +1.07] |

Slope of sim fav margin on close: 0.923 [0.904, 0.942]; slope of real fav margin on close: 1.041 [0.989, 1.094]. (Regressor is the close, no seed-noise issue.)

Test 2 (REST (at least one team < 20 prior games, or unknown)): buckets on |s| (favourite by the SIM), real - sim:

| |s| bucket | n | mean sim fav margin | mean real | real - sim [95% CI] (SE) |
|---|---:|---:|---:|---|
| 0-3 | 746 | 1.47 | +1.27 | -0.20 [-1.05, +0.66] (0.44) |
| 3-6 | 726 | 4.41 | +4.39 | -0.02 [-0.88, +0.90] (0.44) |
| 6-10 | 678 | 7.76 | +6.92 | -0.84 [-1.71, +0.01] (0.43) |
| 10-15 | 571 | 12.37 | +11.83 | -0.55 [-1.49, +0.43] (0.50) |
| 15-20 | 305 | 17.24 | +17.43 | +0.20 [-1.20, +1.50] (0.68) |
| 20+ | 334 | 26.29 | +26.57 | +0.28 [-1.21, +1.77] (0.74) |

IV slope of real home margin on sim home margin (odd/even instrument): 0.970 [0.928, 1.012] (other direction 0.970); sim-favourite orientation: 1.011 [0.954, 1.069].

All 5710 games (no line needed): IV slope of real home margin on sim home margin = 0.955.

