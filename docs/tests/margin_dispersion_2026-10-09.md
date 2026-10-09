# Margin dispersion: is the sim's per-game margin too wide, and does that make its win probabilities under-confident? (fold 2 / 2024-25), 2026-10-09

DIAGNOSTIC ONLY. Nothing changed; no retrain; parity v10 untouched. Script `scripts/diag_margin_dispersion_v1.py` (reuses the `results/diag_c4/f2all50_*` extracts from the fold-2 x 50-seed trajectory run, 5,710 games; nothing re-simulated; 2025-26 sealed and unread). The raw tables are reproduced below and also written to `results/diag_margin_dispersion/` (gitignored).

Sim: served stack v3. Truth: verified finals (OT included). Market: ESPN BET close from `lines_close_v2_verified` (5,380 games with a spread, 5,227 with a moneyline, power de-vig primary). Margin = home minus away. Every residual variance that uses the 50-seed sim mean is corrected for the mean's MC noise (minus mean seed-variance / 50). CIs are game-cluster bootstraps (1000 reps; 400 for slopes). Underpowered = n < 60 games.

## Verdict

**Too wide, modestly, and it does not show up as win-probability under-confidence.**

1. **Width.** Sim own margin SD 12.22 vs realised residual SD (real minus sim mean, MC-corrected) 11.65: ratio **1.049 [1.028, 1.070]**, n = 5,710; 1.054 [1.031, 1.076] on the 5,380 games with a close. About +10% in variance (+13.5 pts^2). Seed noise is not the story (seed-bootstrap band [1.018, 1.035]; odd/even seed halves 1.050 / 1.047). The PIT agrees: central mass (0.25-0.75) 0.534 vs 0.50 (+5.1 se), tail mass 0.180 vs 0.20 (-3.8 se), KS p = 2e-4. A mild hump, not a gross one.
2. **Where.** Worst powered cells (ratio [95%], n): month Feb 1.136 [1.083, 1.194] n = 1,115 (games with close); neutral site 1.124 [1.070, 1.191] n = 736; |spread| Q1 (closest) 1.122 [1.074, 1.174] n = 1,222; March 1.108; conference games 1.081 [1.055, 1.110] n = 3,613. Correct: non-conference 1.001 [0.968, 1.036] n = 2,097; |spread| Q5 0.981 [0.936, 1.036]; home-rating Q5 1.026. Too narrow: November 0.952 [0.913, 0.996] n = 1,222. The pattern: **the sim's own SD is flat (12.1-12.5 across |spread| quintiles and across months) while realised residual SD is not** (10.8 in the closest |spread| quintile to 12.8 in the widest; 13.0 in Nov to 10.9 in Feb/Mar). The market residual shows the same shape (10.6 to 11.7), so the heteroscedasticity is real and the sim's variance function does not carry it. April is underpowered (n = 17).
3. **Centre vs width vs market.** Var(real - sim mean) 134.5 = Var(real - close) 123.5 + Var(close - sim mean) 10.8 + 2Cov 0.3 (5,380 games). The sim's centre costs about 11 pts^2 over the market's (residual SD 11.60 vs 11.11); its claimed SD 12.22 is 1.100 [1.078, 1.123] times the market's realised error SD. So both: wrong centre (adds 8% to error variance vs market) and too wide (claims about 10% more variance than it realises). Centre spread is also slightly too wide: realised margin regressed on sim mean has slope **0.934 [0.900, 0.971]** (market 1.016 [0.979, 1.048]); jointly the sim mean adds nothing beyond the close (sim coefficient -0.001 [-0.083, 0.089], market 1.018 [0.928, 1.104]).
4. **Win probability.** Reliability of realised home win on sim P(home win), n = 5,227: **IV slope 1.017 [0.964, 1.068]** (odd-seed P instrumenting even-seed P to remove 50-seed noise); probit slope on z = mean/SD 1.027 [0.971, 1.087] (implied width multiplier 0.973); logistic on smooth sim P 0.997 [0.935, 1.074]. The naive OLS slope on the 50-seed empirical P is 0.948 [0.901, 0.993]: that apparent over-confidence is MC attenuation, not sim behaviour (do not read a moneyline edge off a 50-seed P without smoothing). Market: slope 1.002 [0.962, 1.036] (OLS), 1.015 [0.951, 1.081] (logistic, power de-vig); proportional de-vig reads 1.165 [1.094, 1.236] (that method compresses the market; power is the one used downstream). So **no detectable under-confidence at the win-probability level**: the roughly 5% excess width is offset by the roughly 7% excess centre spread, net calibration about 1.0. That is two errors cancelling, which the project rules treat as a defect, not a pass.
5. **Versus the market, probabilities.** The sim is less extreme than the market in the regression sense (sim P on market P slope 0.860, SD of P 0.227 vs 0.237, corr 0.898), as expected of a noisier predictor, and is worse on the same games: log loss 0.5460 vs 0.5172 (paired diff +0.029 [+0.023, +0.035]); Brier 0.1852 vs 0.1748. Where the sim disagrees with the market, the realised rate sits on the market (realised - market within +/-0.013 in every disagreement quintile; realised - sim -0.13 to +0.17). Joint logistic: sim coefficient 0.041 [-0.103, 0.177], market 0.983 [0.851, 1.106]. For sides and moneyline: the width excess is not what costs edge; the sim centre carries no information beyond the close.
6. **Owner.** Sim margin variance is 99.4% per-possession efficiency gap, 0.2% pace latent (pace SD ratio 1.009 [0.984, 1.033]), 0.5% overtime. Inside regulation the excess is in the **late-game regime**: Var(minutes 0-36) +2.7 [-2.9, +8.1] (n.s.), Var(last 4:00) +1.4 [+0.8, +2.2], and 2Cov(minute-36 margin, last-4:00 increment) is -12.0 in the sim vs -20.9 realised (diff +8.9 [+5.7, +11.7]): in real games late play pulls the margin back toward the middle more than the sim's does. Margin through minute 36 is on width (ratio 1.010 [0.990, 1.032]). Secondary: the efficiency variance function is flat in |spread| and season stage (item 2), and the known +1.3% possession-level bias (sim 68.39 vs real 67.54 per team) inflates margin points by about 1.3% of the 4.7%. Overtime is too NARROW (increment SD ratio 0.829 [0.768, 0.900]; sim OT rate 3.5% vs 5.6% real, known from D4) and works the other way.
   **Owners, ranked:** (a) late-game regime (D3/D4 round: margin-conditional clock, foul state), (b) efficiency variance function (no dependence on |spread| or season stage), (c) pace level (clock round), (d) OT rate (opposite sign). **The pace latent SD is NOT the owner**; the clock scoring-environment pace spec needs only a no-regression dispersion check, written as a sibling note: `docs/models/clock/experiments_DRAFT_scoring_env_pace_2026-10-09_dispersion_note.md` (the draft is unedited).

Nothing was fixed or adjusted. The implied width multiplier and slope fits are diagnosis only; nothing is applied to sim output.

## Caveats

- The realised residual is measured against the sim's own mean, so it contains the sim's centre error. That is the right comparison for "is the predictive distribution calibrated", but one residual cannot separate width from centre: part 2 and the slopes do that.
- Cut cells (|spread|, month, conference, site, rating) overlap and are not independent evidence; single season.
- Spread lines carry no per-line timestamp (see `lines_cbbd_validation_2026-09-10.md`); "close" is the line on file.
- Part 4b component ratios use 5,499 games whose pbp final equals the verified final and whose regulation margin matches. The sim possession and late-game tables are regulation-only, so pace and PPP components are regulation; OT enters through the OT increment.
- Win-probability bucket rows have about 261 games each; bucket deviations of +/-0.05 are within Wilson bands. Read the bucket tables as shape and the slopes as the test.

# Tables

## 1. Sim own margin SD vs realised residual SD (real - sim mean)

Games: 5710 simulated; 5380 with an ESPN BET close; 5227 with a de-vigged moneyline. All 5710 have verified finals. Each sim game is 50 seeds; MC correction subtracts mean(s2)/50.

### 1a. All games with verified final

| cell | n games | sim own SD | realised residual SD (real - sim mean, MC-corrected) [95%] | ratio sim/real [95% game bootstrap] | mean(real - sim mean) | verdict (CI vs 1) |
|---|---:|---:|---|---|---:|---|
| ALL | 5710 | 12.22 | 11.65 [11.43, 11.88] | 1.049 [1.028, 1.070] | +0.22 | wide |
| |spread| Q1 (mean 1.8) | 1222 | 12.10 | 10.78 [10.30, 11.27] | 1.122 [1.074, 1.174] | +0.20 | wide |
| |spread| Q2 (mean 4.0) | 975 | 12.20 | 11.64 [11.06, 12.25] | 1.048 [0.997, 1.102] | -0.21 | n.s. |
| |spread| Q3 (mean 6.4) | 1123 | 12.13 | 11.61 [11.08, 12.15] | 1.045 [0.997, 1.096] | -0.27 | n.s. |
| |spread| Q4 (mean 10.2) | 1034 | 12.20 | 11.29 [10.74, 11.86] | 1.081 [1.025, 1.136] | -0.05 | wide |
| |spread| Q5 (mean 20.1) | 1026 | 12.52 | 12.75 [12.09, 13.35] | 0.981 [0.936, 1.036] | +1.57 | n.s. |
| month Nov | 1222 | 12.36 | 12.98 [12.42, 13.54] | 0.952 [0.913, 0.996] | +0.91 | narrow |
| month Dec | 916 | 12.46 | 11.64 [11.12, 12.20] | 1.070 [1.021, 1.122] | +0.36 | wide |
| month Jan | 1423 | 12.10 | 11.47 [10.97, 11.95] | 1.055 [1.013, 1.103] | -0.81 | wide |
| month Feb | 1367 | 12.07 | 10.94 [10.50, 11.38] | 1.104 [1.062, 1.151] | +0.18 | wide |
| month Mar | 765 | 12.14 | 10.96 [10.42, 11.53] | 1.108 [1.053, 1.163] | +0.98 | wide |
| month Apr UNDERPOWERED | 17 | 12.26 | 11.12 [7.52, 14.45] | 1.102 [0.854, 1.628] | -1.66 | n.s. |
| conference game | 3613 | 12.10 | 11.19 [10.92, 11.47] | 1.081 [1.055, 1.110] | -0.12 | wide |
| non-conference game | 2097 | 12.41 | 12.39 [11.99, 12.81] | 1.001 [0.968, 1.036] | +0.82 | n.s. |
| neutral site | 736 | 12.26 | 10.90 [10.30, 11.44] | 1.124 [1.070, 1.191] | +1.13 | wide |
| home/away site | 4974 | 12.21 | 11.75 [11.49, 12.02] | 1.039 [1.014, 1.061] | +0.09 | wide |
| home rating Q1 (mean net -14.2) | 1120 | 11.81 | 11.19 [10.68, 11.69] | 1.055 [1.009, 1.104] | +0.38 | wide |
| home rating Q2 (mean net -5.4) | 1120 | 12.13 | 11.36 [10.86, 11.85] | 1.068 [1.023, 1.117] | -0.08 | wide |
| home rating Q3 (mean net +0.7) | 1119 | 12.23 | 11.66 [11.14, 12.18] | 1.049 [1.004, 1.098] | -0.20 | wide |
| home rating Q4 (mean net +8.4) | 1120 | 12.37 | 11.52 [10.93, 12.08] | 1.074 [1.024, 1.131] | -0.11 | wide |
| home rating Q5 (mean net +21.2) | 1120 | 12.46 | 12.14 [11.62, 12.67] | 1.026 [0.982, 1.074] | +0.77 | n.s. |
| home rating missing | 111 | 12.84 | 14.70 [12.38, 16.90] | 0.874 [0.764, 1.042] | +3.72 | n.s. |

### 1b. Same cuts, restricted to games with a market close (the sample parts 2-3 use)

| cell | n games | sim own SD | realised residual SD (real - sim mean, MC-corrected) [95%] | ratio sim/real [95% game bootstrap] | mean(real - sim mean) | verdict (CI vs 1) |
|---|---:|---:|---|---|---:|---|
| ALL | 5380 | 12.22 | 11.60 [11.36, 11.85] | 1.054 [1.031, 1.076] | +0.24 | wide |
| |spread| Q1 (mean 1.8) | 1222 | 12.10 | 10.78 [10.33, 11.23] | 1.122 [1.076, 1.171] | +0.20 | wide |
| |spread| Q2 (mean 4.0) | 975 | 12.20 | 11.64 [11.07, 12.22] | 1.048 [0.998, 1.101] | -0.21 | n.s. |
| |spread| Q3 (mean 6.4) | 1123 | 12.13 | 11.61 [11.08, 12.11] | 1.045 [1.001, 1.097] | -0.27 | wide |
| |spread| Q4 (mean 10.2) | 1034 | 12.20 | 11.29 [10.75, 11.87] | 1.081 [1.028, 1.136] | -0.05 | wide |
| |spread| Q5 (mean 20.1) | 1026 | 12.52 | 12.75 [12.17, 13.37] | 0.981 [0.936, 1.029] | +1.57 | n.s. |
| month Nov | 1221 | 12.36 | 12.98 [12.40, 13.60] | 0.952 [0.909, 0.998] | +0.92 | narrow |
| month Dec | 915 | 12.46 | 11.65 [11.02, 12.23] | 1.070 [1.020, 1.130] | +0.36 | wide |
| month Jan | 1350 | 12.11 | 11.38 [10.89, 11.87] | 1.064 [1.020, 1.113] | -0.79 | wide |
| month Feb | 1115 | 12.07 | 10.62 [10.14, 11.14] | 1.136 [1.083, 1.194] | +0.20 | wide |
| month Mar | 762 | 12.14 | 10.96 [10.40, 11.51] | 1.108 [1.053, 1.168] | +0.95 | wide |
| month Apr UNDERPOWERED | 17 | 12.26 | 11.12 [7.68, 14.34] | 1.102 [0.863, 1.601] | -1.66 | n.s. |
| conference game | 3285 | 12.11 | 11.07 [10.77, 11.38] | 1.094 [1.066, 1.125] | -0.13 | wide |
| non-conference game | 2095 | 12.41 | 12.40 [11.98, 12.78] | 1.001 [0.971, 1.035] | +0.83 | n.s. |
| neutral site | 734 | 12.26 | 10.91 [10.35, 11.51] | 1.124 [1.065, 1.186] | +1.13 | wide |
| home/away site | 4646 | 12.22 | 11.71 [11.44, 11.98] | 1.044 [1.020, 1.069] | +0.10 | wide |
| home rating Q1 (mean net -14.1) | 1035 | 11.84 | 11.11 [10.57, 11.62] | 1.065 [1.017, 1.120] | +0.29 | wide |
| home rating Q2 (mean net -5.5) | 1043 | 12.13 | 11.21 [10.69, 11.73] | 1.083 [1.034, 1.137] | -0.22 | wide |
| home rating Q3 (mean net +0.7) | 1038 | 12.24 | 11.72 [11.15, 12.30] | 1.045 [0.996, 1.098] | -0.13 | n.s. |
| home rating Q4 (mean net +8.5) | 1069 | 12.36 | 11.54 [10.96, 12.18] | 1.071 [1.015, 1.128] | -0.01 | wide |
| home rating Q5 (mean net +21.2) | 1084 | 12.46 | 12.02 [11.46, 12.54] | 1.037 [0.995, 1.090] | +0.89 | n.s. |
| home rating missing | 111 | 12.84 | 14.70 [12.47, 16.89] | 0.874 [0.756, 1.030] | +3.72 | n.s. |

### 1c. Seed-noise band on the overall ratio

| seeds used | sim own SD | realised residual SD | ratio |
|---|---:|---:|---:|
| all 50 seeds | 12.215 | 11.648 | 1.0487 |
| even seeds (25) | 12.183 | 11.635 | 1.0471 |
| odd seeds (25) | 12.244 | 11.665 | 1.0496 |

Seed bootstrap (300 reps of resampling the 50 seeds, games fixed): ratio 95% band [1.0175, 1.0350]. This is the pure sim-noise band; it is small next to the game-sampling CI above.

Equivalent view, standardised residual z = (real - sim mean)/sim own game SD: SD(z) = 0.984 (calibrated width = 1.0; below 1 means the sim is too wide); mean z = +0.019.

## 2. Versus the market: wrong centre or wrong width?

Sample: 5380 games with a close. e_sim = real - sim mean; e_mkt = real - m (m = -close spread); d = m - sim mean, so e_sim = e_mkt + d exactly. Var(e_sim) and Var(d) are MC-corrected (minus mean(s2)/50); Cov(e_mkt, d) is not affected by MC noise (independent of the realised margin). Variances are about each cell's mean; the means are shown separately.

| cell | n | SD(e_mkt) market resid [95%] | SD(e_sim) sim resid [95%] | sim own SD | Var(e_mkt) | Var(d) centre gap | 2Cov | Var(e_sim) | sim own SD / market resid SD [95%] | sim own SD / sim resid SD [95%] | mean e_mkt | mean e_sim |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| ALL (with close) | 5380 | 11.11 [10.88, 11.34] | 11.60 [11.34, 11.84] | 12.22 | 123.5 | 10.8 | +0.3 | 134.5 | 1.100 [1.078, 1.123] | 1.054 [1.032, 1.078] | +0.17 | +0.24 |
| |spread| Q1 | 1222 | 10.55 [10.07, 11.00] | 10.78 [10.30, 11.21] | 12.10 | 111.4 | 8.3 | -3.5 | 116.2 | 1.147 [1.100, 1.201] | 1.122 [1.078, 1.177] | +0.40 | +0.20 |
| |spread| Q2 | 975 | 11.34 [10.79, 11.87] | 11.64 [11.09, 12.17] | 12.20 | 128.7 | 8.8 | -2.0 | 135.4 | 1.075 [1.026, 1.131] | 1.048 [1.001, 1.100] | -0.18 | -0.21 |
| |spread| Q3 | 1123 | 11.28 [10.76, 11.76] | 11.60 [11.09, 12.10] | 12.13 | 127.3 | 7.9 | -0.6 | 134.6 | 1.075 [1.033, 1.128] | 1.046 [1.001, 1.094] | -0.08 | -0.27 |
| |spread| Q4 | 1034 | 10.69 [10.16, 11.21] | 11.29 [10.71, 11.85] | 12.20 | 114.3 | 10.3 | +2.8 | 127.4 | 1.141 [1.089, 1.202] | 1.081 [1.030, 1.139] | +0.12 | -0.05 |
| |spread| Q5 | 1026 | 11.74 [11.20, 12.27] | 12.66 [12.04, 13.24] | 12.52 | 137.8 | 18.1 | +4.3 | 160.2 | 1.066 [1.018, 1.119] | 0.989 [0.944, 1.039] | +0.53 | +1.57 |
| month Nov | 1221 | 11.87 [11.38, 12.31] | 12.95 [12.42, 13.43] | 12.36 | 140.9 | 23.5 | +3.4 | 167.7 | 1.042 [1.004, 1.087] | 0.955 [0.920, 0.997] | +0.39 | +0.92 |
| month Dec | 915 | 11.08 [10.56, 11.62] | 11.64 [11.05, 12.21] | 12.46 | 122.9 | 11.2 | +1.6 | 135.6 | 1.124 [1.071, 1.180] | 1.070 [1.021, 1.127] | +0.15 | +0.36 |
| month Jan | 1350 | 10.99 [10.51, 11.47] | 11.35 [10.86, 11.87] | 12.11 | 120.8 | 5.9 | +2.2 | 128.9 | 1.101 [1.056, 1.154] | 1.066 [1.022, 1.116] | -0.45 | -0.79 |
| month Feb | 1115 | 10.67 [10.23, 11.15] | 10.61 [10.16, 11.11] | 12.07 | 113.9 | 5.5 | -6.8 | 112.7 | 1.131 [1.081, 1.182] | 1.137 [1.086, 1.189] | +0.31 | +0.20 |
| month Mar | 762 | 10.71 [10.16, 11.25] | 10.92 [10.37, 11.45] | 12.14 | 114.7 | 5.7 | -1.2 | 119.2 | 1.134 [1.077, 1.195] | 1.112 [1.060, 1.172] | +0.75 | +0.95 |
| month Apr UNDERPOWERED | 17 | 9.83 [5.99, 12.60] | 11.00 [5.88, 14.17] | 12.26 | 96.7 | 8.8 | +15.5 | 121.0 | 1.247 [0.973, 2.056] | 1.115 [0.857, 2.072] | -1.09 | -1.66 |
| conference game | 3285 | 10.85 [10.55, 11.14] | 11.06 [10.76, 11.37] | 12.11 | 117.7 | 5.9 | -1.2 | 122.4 | 1.116 [1.087, 1.147] | 1.094 [1.065, 1.125] | +0.03 | -0.13 |
| non-conference game | 2095 | 11.51 [11.15, 11.89] | 12.37 [11.93, 12.80] | 12.41 | 132.6 | 18.2 | +2.2 | 153.0 | 1.077 [1.043, 1.113] | 1.003 [0.970, 1.040] | +0.38 | +0.83 |
| neutral site | 734 | 10.52 [9.98, 11.05] | 10.85 [10.29, 11.42] | 12.26 | 110.7 | 10.4 | -3.3 | 117.8 | 1.165 [1.110, 1.230] | 1.130 [1.073, 1.192] | +0.74 | +1.13 |
| home/away site | 4646 | 11.20 [10.95, 11.44] | 11.71 [11.45, 11.98] | 12.22 | 125.5 | 10.8 | +0.7 | 137.0 | 1.091 [1.068, 1.115] | 1.044 [1.019, 1.068] | +0.08 | +0.10 |
| home rating Q1 | 1035 | 10.88 [10.38, 11.40] | 11.11 [10.57, 11.67] | 11.84 | 118.4 | 8.6 | -3.6 | 123.4 | 1.088 [1.039, 1.142] | 1.065 [1.013, 1.120] | +0.05 | +0.29 |
| home rating Q2 | 1043 | 10.84 [10.32, 11.31] | 11.20 [10.66, 11.67] | 12.13 | 117.5 | 8.1 | -0.1 | 125.5 | 1.119 [1.074, 1.176] | 1.083 [1.039, 1.139] | -0.24 | -0.22 |
| home rating Q3 | 1038 | 11.23 [10.70, 11.73] | 11.71 [11.12, 12.30] | 12.24 | 126.2 | 9.2 | +1.8 | 137.2 | 1.089 [1.043, 1.141] | 1.045 [0.997, 1.099] | -0.11 | -0.13 |
| home rating Q4 | 1069 | 11.13 [10.58, 11.67] | 11.54 [10.95, 12.13] | 12.36 | 123.9 | 12.5 | -3.2 | 133.2 | 1.111 [1.059, 1.169] | 1.071 [1.019, 1.129] | -0.09 | -0.01 |
| home rating Q5 | 1084 | 11.25 [10.76, 11.73] | 11.98 [11.45, 12.49] | 12.46 | 126.5 | 13.1 | +4.0 | 143.6 | 1.108 [1.062, 1.158] | 1.040 [0.996, 1.089] | +0.91 | +0.89 |

Reading: if sim own SD is larger than SD(e_mkt) the sim claims more per-game uncertainty than the market's own realised error. Var(d) is the extra variance the sim's centre adds over the market's (a wrong-centre cost); a ratio of sim own SD to sim resid SD above 1 means the sim is too wide *given its own centre*.

### 2b. Centre responsiveness: realised margin regressed on each centre (slope 1 = right spread of means)

| regressor | n | slope [95%] | intercept | R^2 |
|---|---:|---|---:|---:|
| sim mean margin | 5380 | 0.934 [0.900, 0.971] | +0.59 | 0.364 |
| market m = -close | 5380 | 1.016 [0.979, 1.048] | +0.08 | 0.427 |

Joint fit real = a + b1*m + b2*mu: b1 (market) = 1.018 [0.928, 1.104], b2 (sim) = -0.001 [-0.083, 0.089]. SD of the centres across games: sim mean 9.48, market 9.44 (mean-noise-corrected sim: 9.32).

## 3. Win-probability calibration and PIT

Sample: 5227 games with a de-vigged moneyline (power de-vig primary; proportional shown for slopes). Sim P(home win) = share of 50 seeds with margin > 0 (MC SD of that estimate is sqrt(p(1-p)/50) <= 0.071; this attenuates naive slopes, so an odd/even-seed IV slope is reported). Realised home win rate 0.6377; sim mean 0.6242; market mean 0.6344.

### 3a. 20 equal-count buckets on SIM P(home win)

| bucket | n | mean sim P | mean market P | realised rate [Wilson 95%] | realised - sim | realised - market |
|---|---:|---:|---:|---|---:|---:|
| 1 | 262 | 0.141 | 0.174 | 0.164 [0.124, 0.214] | +0.023 | -0.010 |
| 2 | 261 | 0.265 | 0.305 | 0.257 [0.208, 0.313] | -0.009 | -0.048 |
| 3 | 261 | 0.340 | 0.379 | 0.398 [0.341, 0.459] | +0.058 | +0.019 |
| 4 | 262 | 0.392 | 0.428 | 0.405 [0.347, 0.465] | +0.013 | -0.023 |
| 5 | 261 | 0.440 | 0.475 | 0.479 [0.419, 0.539] | +0.039 | +0.004 |
| 6 | 261 | 0.487 | 0.500 | 0.559 [0.499, 0.618] | +0.072 | +0.059 |
| 7 | 262 | 0.523 | 0.533 | 0.515 [0.455, 0.575] | -0.008 | -0.018 |
| 8 | 261 | 0.558 | 0.565 | 0.567 [0.506, 0.626] | +0.009 | +0.002 |
| 9 | 261 | 0.591 | 0.611 | 0.690 [0.631, 0.743] | +0.098 | +0.079 |
| 10 | 262 | 0.627 | 0.618 | 0.641 [0.581, 0.697] | +0.014 | +0.023 |
| 11 | 261 | 0.659 | 0.657 | 0.617 [0.557, 0.674] | -0.042 | -0.040 |
| 12 | 261 | 0.690 | 0.692 | 0.709 [0.651, 0.761] | +0.019 | +0.017 |
| 13 | 261 | 0.723 | 0.718 | 0.720 [0.663, 0.771] | -0.003 | +0.003 |
| 14 | 262 | 0.754 | 0.756 | 0.752 [0.696, 0.800] | -0.002 | -0.004 |
| 15 | 261 | 0.788 | 0.790 | 0.785 [0.732, 0.831] | -0.003 | -0.004 |
| 16 | 261 | 0.822 | 0.821 | 0.870 [0.823, 0.905] | +0.047 | +0.049 |
| 17 | 262 | 0.863 | 0.866 | 0.828 [0.778, 0.869] | -0.035 | -0.037 |
| 18 | 261 | 0.900 | 0.904 | 0.897 [0.854, 0.928] | -0.004 | -0.008 |
| 19 | 261 | 0.937 | 0.932 | 0.923 [0.885, 0.950] | -0.014 | -0.009 |
| 20 | 262 | 0.980 | 0.966 | 0.977 [0.951, 0.989] | -0.002 | +0.011 |

### 3b. 20 equal-count buckets on MARKET de-vigged P(home win)

| bucket | n | mean sim P | mean market P | realised rate [Wilson 95%] | realised - sim | realised - market |
|---|---:|---:|---:|---|---:|---:|
| 1 | 262 | 0.192 | 0.128 | 0.130 [0.094, 0.176] | -0.063 | +0.002 |
| 2 | 261 | 0.296 | 0.259 | 0.241 [0.193, 0.297] | -0.054 | -0.017 |
| 3 | 261 | 0.380 | 0.340 | 0.322 [0.268, 0.381] | -0.058 | -0.018 |
| 4 | 262 | 0.423 | 0.399 | 0.420 [0.362, 0.480] | -0.003 | +0.021 |
| 5 | 261 | 0.478 | 0.447 | 0.456 [0.397, 0.517] | -0.022 | +0.009 |
| 6 | 261 | 0.499 | 0.489 | 0.502 [0.442, 0.562] | +0.003 | +0.013 |
| 7 | 262 | 0.529 | 0.533 | 0.546 [0.485, 0.605] | +0.017 | +0.013 |
| 8 | 261 | 0.562 | 0.567 | 0.628 [0.568, 0.685] | +0.067 | +0.062 |
| 9 | 261 | 0.587 | 0.599 | 0.590 [0.529, 0.648] | +0.003 | -0.009 |
| 10 | 262 | 0.614 | 0.629 | 0.584 [0.523, 0.642] | -0.030 | -0.045 |
| 11 | 261 | 0.640 | 0.661 | 0.674 [0.615, 0.728] | +0.034 | +0.014 |
| 12 | 261 | 0.684 | 0.698 | 0.716 [0.659, 0.768] | +0.032 | +0.019 |
| 13 | 261 | 0.714 | 0.730 | 0.701 [0.643, 0.753] | -0.012 | -0.029 |
| 14 | 262 | 0.734 | 0.768 | 0.771 [0.716, 0.818] | +0.037 | +0.003 |
| 15 | 261 | 0.771 | 0.809 | 0.847 [0.798, 0.885] | +0.075 | +0.038 |
| 16 | 261 | 0.805 | 0.857 | 0.866 [0.819, 0.902] | +0.061 | +0.009 |
| 17 | 262 | 0.836 | 0.895 | 0.874 [0.828, 0.909] | +0.038 | -0.021 |
| 18 | 261 | 0.883 | 0.932 | 0.931 [0.894, 0.956] | +0.048 | -0.001 |
| 19 | 261 | 0.910 | 0.962 | 0.958 [0.926, 0.976] | +0.048 | -0.004 |
| 20 | 262 | 0.947 | 0.988 | 0.996 [0.979, 0.999] | +0.049 | +0.008 |

### 3c. Reliability slopes (1.0 = calibrated; >1 = under-confident; <1 = over-confident)

n = 5227. 95% = game bootstrap, 400 reps.

| fit | slope [95%] |
|---|---|
| OLS realised ~ sim P (naive; attenuated by 50-seed noise) | 0.948 [0.901, 0.993] |
| IV realised ~ sim P (even-seed P instrumented by odd-seed P; noise-free) | 1.017 [0.964, 1.068] |
| OLS realised ~ market P | 1.002 [0.962, 1.036] |
| logistic slope, logit(sim P) (50-seed empirical, clipped) | 0.938 [0.875, 1.008] |
| logistic slope, logit(Phi(mu/s)) (smooth sim P) | 0.997 [0.935, 1.074] |
| logistic slope, logit(market P, power de-vig) | 1.015 [0.951, 1.081] |
| logistic slope, logit(market P, proportional de-vig) | 1.165 [1.094, 1.236] |
| probit slope of win on z = sim mean / sim own SD (width-implied; slope b => calibrated width = sim SD / b) | 1.027 [0.971, 1.087] |

Joint logistic y ~ logit(sim Phi) + logit(market): sim coefficient 0.041 [-0.103, 0.177], market coefficient 0.983 [0.851, 1.106]. The market carries the weight if the sim coefficient is near zero.

Implied width multiplier from the probit slope: sim SD would need to be x0.973 to make win probabilities calibrated *with the sim's own centre* (diagnostic only; nothing is applied to any output). Because the sim centre is noisier than the market's, a slope below 1 can reflect centre error as well as width.

Sim-vs-market dispersion of probabilities: OLS slope of sim P on market P = 0.860 [0.849, 0.870]; slope of logit(Phi(mu/s)) on logit(market P) = 0.762 [0.750, 0.774]. SD of sim P 0.2267 vs market P 0.2368; corr 0.898. Slope > 1 means the sim is MORE confident than the market, < 1 less.

### 3d. Scores on the same games

| predictor | log loss | Brier | 95% (paired vs market, log loss diff) |
|---|---:|---:|---|
| sim empirical P (50 seeds) | 0.5460 | 0.1852 | diff +0.0289 [+0.0226, +0.0361] |
| sim Phi(mu/s) | 0.5427 | 0.1841 | diff +0.0256 [+0.0195, +0.0317] |
| market power de-vig | 0.5172 | 0.1748 | diff +0.0000 [+0.0000, +0.0000] |
| market proportional | 0.5202 | 0.1752 | diff +0.0030 [+0.0015, +0.0045] |

### 3e. Sim minus market probability: who is right when they disagree

| bucket of (sim P - market P) | n | mean sim - mkt | realised - market | realised - sim |
|---|---:|---:|---:|---:|
| Q1 | 1046 | -0.155 | +0.013 | +0.168 |
| Q2 | 1045 | -0.059 | -0.007 | +0.052 |
| Q3 | 1045 | -0.011 | -0.006 | +0.005 |
| Q4 | 1045 | +0.037 | +0.009 | -0.028 |
| Q5 | 1046 | +0.137 | +0.008 | -0.129 |

If the sim were merely under-confident (same centre, too wide) its disagreements with the market would be centred on the market: realised - market ~ 0. A realised - market gap that tracks the sign of sim - market would mean the sim has real information; a gap of the opposite sign would mean the disagreement is noise.

### 3f. PIT of the realised margin under each game's 50-seed sim margin distribution

PIT is rank-based on the 50 seeds with randomised tie-breaking (margins are integers), so it is exactly uniform if the real margin is exchangeable with the sim draws. Expected per decile = n/10; the 95% band on a decile count is +/- 1.96 sqrt(n*0.09) (n=5710: expected 571 +/- 44). Central mass = P(0.25 < PIT < 0.75), expected 0.50; a hump (too wide) gives > 0.50. Tail mass = P(PIT < 0.1 or > 0.9), expected 0.20; too wide gives < 0.20.

| cell | n | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | central mass (z vs .5) | tail mass (z vs .2) | KS p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| ALL games | 5710 | 493 | 529 | 547 | 621 | 673 | 613 | 595 | 551 | 555 | 533 | 0.534 (+5.1 se) | 0.180 (-3.8 se) | 0.000248 |
| with close | 5380 | 459 | 490 | 517 | 586 | 636 | 583 | 565 | 526 | 520 | 498 | 0.536 (+5.3 se) | 0.178 (-4.1 se) | 0.000106 |
| |spread| Q1 | 1222 | 89 | 103 | 126 | 139 | 157 | 130 | 128 | 125 | 124 | 101 | 0.570 (+4.9 se) | 0.155 (-3.9 se) | 0.00646 |
| |spread| Q2 | 975 | 96 | 84 | 102 | 97 | 125 | 94 | 111 | 88 | 87 | 91 | 0.528 (+1.8 se) | 0.192 (-0.6 se) | 0.313 |
| |spread| Q3 | 1123 | 113 | 97 | 100 | 122 | 134 | 133 | 106 | 117 | 103 | 98 | 0.533 (+2.2 se) | 0.188 (-1.0 se) | 0.288 |
| |spread| Q4 | 1034 | 83 | 96 | 111 | 128 | 115 | 108 | 120 | 98 | 93 | 82 | 0.557 (+3.7 se) | 0.160 (-3.2 se) | 0.0593 |
| |spread| Q5 | 1026 | 78 | 110 | 78 | 100 | 105 | 118 | 100 | 98 | 113 | 126 | 0.487 (-0.8 se) | 0.199 (-0.1 se) | 0.0279 |
| month Nov | 1222 | 114 | 117 | 116 | 105 | 127 | 134 | 117 | 116 | 126 | 150 | 0.493 (-0.5 se) | 0.216 (+1.4 se) | 0.145 |
| month Dec | 916 | 69 | 91 | 82 | 105 | 109 | 99 | 97 | 89 | 94 | 81 | 0.536 (+2.2 se) | 0.164 (-2.7 se) | 0.169 |
| month Jan | 1423 | 129 | 155 | 152 | 163 | 169 | 147 | 146 | 134 | 131 | 97 | 0.541 (+3.1 se) | 0.159 (-3.9 se) | 0.000681 |
| month Feb | 1367 | 112 | 118 | 132 | 152 | 170 | 153 | 149 | 131 | 128 | 122 | 0.557 (+4.2 se) | 0.171 (-2.7 se) | 0.0325 |
| month Mar | 765 | 68 | 45 | 63 | 93 | 95 | 80 | 84 | 81 | 75 | 81 | 0.541 (+2.3 se) | 0.195 (-0.4 se) | 0.000903 |
| conference | 3613 | 318 | 326 | 355 | 415 | 443 | 386 | 382 | 348 | 336 | 304 | 0.546 (+5.6 se) | 0.172 (-4.2 se) | 0.00204 |
| non-conference | 2097 | 175 | 203 | 192 | 206 | 230 | 227 | 213 | 203 | 219 | 229 | 0.512 (+1.1 se) | 0.193 (-0.8 se) | 0.0133 |
| neutral | 736 | 50 | 53 | 67 | 83 | 89 | 89 | 78 | 78 | 76 | 73 | 0.548 (+2.6 se) | 0.167 (-2.2 se) | 0.000349 |
| home/away | 4974 | 443 | 476 | 480 | 538 | 584 | 524 | 517 | 473 | 479 | 460 | 0.532 (+4.5 se) | 0.182 (-3.3 se) | 0.0233 |

Reference: PIT of the realised margin under N(market, SD=11.11) (a constant-width market model) has central mass 0.487 on the same 5380 games (Normal-shape baseline: margins are not exactly normal, so this is the honest ceiling for 'shape' error, not 0.50 exactly).

## 4. Where the width comes from

Real-side counterpart games: 5499 of 5710 (pbp present and its final equals the verified final).

### 4a. Within-game variance of the sim margin (across 50 seeds), attributed by covariance share (shares sum to 1)

Mean within-game margin variance = 149.22 pts^2 (SD 12.22). regulation margin = P*D + small (sim_tg is regulation-only; OT is its own term), P = mean team possessions, D = PPP_home - PPP_away; pace term = mean(D)*dP, efficiency term = mean(P)*dD, interaction = dP*dD, remainder = home/away possession imbalance. Mean |D| across games 0.120 PPP, so pace only moves the margin through a game's expected point gap.

| term | variance of term (pts^2) | share of margin variance (Cov(term, margin)/Var(margin)) |
|---|---:|---:|
| pace (possession-count latent) x expected PPP gap | 0.510 | 0.0019 |
| per-possession efficiency (PPP gap) x mean possessions | 150.145 | 0.9940 |
| pace x efficiency interaction | 0.645 | -0.0018 |
| possession imbalance remainder (regulation) | 0.565 | 0.0010 |
| overtime increment | 1.025 | 0.0049 |
| total | 149.22 | 1.0000 |

Time-sequence attribution of the same variance (increments of home margin): first 36 min of regulation (to the last possession starting with > 4:00 left in H2), last 4:00 of regulation, overtime.

| segment | mean within-game variance | share of margin variance (Cov(incr, margin)/Var) | share of games with any activity |
|---|---:|---:|---:|
| minutes 0-36 | 140.78 | 0.9012 | 1.000 |
| last 4:00 of regulation | 20.09 | 0.0939 | 0.912 |
| overtime | 1.02 | 0.0049 | 0.035 |

Sim OT rate 0.0354 per game-seed (OT with exact tie after OT periods would show as 0 increment, so this is a floor) / sim `ot` flag mean 0.0354; real OT rate 0.0560 (pbp games), verified-final OT rate 0.0557.

### 4b. Component SD: sim within-game SD vs realised residual SD (real - sim mean, MC-corrected), games with matching pbp

| component | n | sim within-game SD | realised residual SD [95%] | ratio sim/real [95%] | mean(real - sim mean) | note |
|---|---:|---:|---|---|---:|---|
| final margin (OT incl.) | 5499 | 12.2181 | 11.6650 [11.4348, 11.9222] | 1.047 [1.026, 1.067] | +0.1821 | heavy-tailed / mostly zero: SD ratio is a weak summary |
| regulation margin | 5499 | 12.2001 | 11.6532 [11.3757, 11.8994] | 1.047 [1.025, 1.072] | +0.1271 |  |
| margin through minute 36 | 5499 | 11.8648 | 11.7489 [11.5080, 11.9947] | 1.010 [0.989, 1.031] | +0.2809 |  |
| last 4:00 regulation increment | 5499 | 4.4843 | 4.3223 [4.2405, 4.4037] | 1.037 [1.017, 1.058] | -0.1538 | heavy-tailed / mostly zero: SD ratio is a weak summary |
| OT increment | 5499 | 1.0148 | 1.2242 [1.1191, 1.3184] | 0.829 [0.768, 0.907] | +0.0550 | heavy-tailed / mostly zero: SD ratio is a weak summary |
| regulation possessions per team (pace) | 5498 | 4.5821 | 4.5405 [4.4355, 4.6484] | 1.009 [0.985, 1.033] | -0.8527 |  |
| regulation PPP home | 5498 | 0.1360 | 0.1366 [0.1339, 0.1392] | 0.996 [0.977, 1.015] | +0.0064 |  |
| regulation PPP away | 5499 | 0.1369 | 0.1371 [0.1344, 0.1398] | 0.999 [0.980, 1.019] | +0.0048 |  |
| regulation PPP gap D = home - away | 5498 | 0.1794 | 0.1750 [0.1712, 0.1785] | 1.025 [1.005, 1.048] | +0.0016 |  |
| regulation total points | 5498 | 16.1179 | 16.2157 [15.8814, 16.5558] | 0.994 [0.973, 1.015] | -1.0087 |  |

Caveat: the realised residual of a component is measured against the sim's own mean for that component, so it includes centre error; a ratio below 1 means the sim's within-game spread is smaller than real-vs-sim error for that component, above 1 larger. Components are not independent and their ratios do not combine linearly; use 4a for shares and 4b for per-component width checks.

### 4c. Margin, pace and PPP-gap ratios by |spread| quintile (games with close and matching pbp)

| cell | n | margin ratio [95%] | pace P ratio [95%] | PPP gap D ratio [95%] | reg margin ratio [95%] |
|---|---:|---|---|---|---|
| |spread| Q1 | 1187 | 1.121 [1.078, 1.170] | 1.009 [0.959, 1.060] | 1.093 [1.050, 1.140] | 1.126 [1.079, 1.179] |
| |spread| Q2 | 937 | 1.042 [0.989, 1.100] | 1.026 [0.981, 1.073] | 1.019 [0.970, 1.071] | 1.045 [0.990, 1.097] |
| |spread| Q3 | 1074 | 1.049 [1.000, 1.097] | 1.022 [0.966, 1.084] | 1.017 [0.975, 1.063] | 1.051 [1.005, 1.103] |
| |spread| Q4 | 991 | 1.070 [1.019, 1.134] | 1.044 [0.991, 1.097] | 1.032 [0.986, 1.087] | 1.065 [1.014, 1.122] |
| |spread| Q5 | 992 | 0.989 [0.944, 1.038] | 0.933 [0.885, 0.984] | 0.994 [0.947, 1.041] | 0.985 [0.938, 1.038] |

### 4e. Regulation margin variance split into minutes 0-36, last 4:00, and their covariance (sim within-game vs realised residual, MC-corrected)

Sim column = mean across games of the across-seed (co)variance. Realised column = mean of (real - sim mean) products, minus the MC-noise share (mean sim (co)variance / 50). Games with matching pbp only.

| term | sim (pts^2) | realised (pts^2) [95%] | sim - real [95%] |
|---|---:|---|---|
| Var(minutes 0-36) | 140.77 | 138.04 [131.87, 143.05] | +2.74 [-2.64, +8.83] |
| Var(last 4:00) | 20.11 | 18.68 [17.93, 19.48] | +1.43 [+0.62, +2.22] |
| 2 Cov(0-36, last 4:00) | -12.04 | -20.92 [-23.80, -17.98] | +8.88 [+5.75, +11.73] |
| Var(regulation margin) | 148.84 | 135.80 [129.89, 141.21] | +13.05 [+7.73, +18.76] |

### 4f. How much of the margin SD ratio is the known possession-count level bias

Mean regulation possessions per team: sim 68.39, real 67.54 (ratio 1.0126). Margin = possessions x PPP gap, so a sim that runs 1.3% more possessions scales margin SD by about the same factor at fixed PPP-gap SD. Regulation margin SD ratio 1.047; PPP-gap SD ratio 1.025; possession-level factor 1.013; product 1.038. (Pace *variance* is not the issue - pace SD ratio 1.01 - but the pace *level* bias, owned by the clock round, also widens the margin in points.)

### 4d. Excess variance accounting

Overall sim own variance 149.22 vs MC-corrected realised residual variance 135.67; excess = +13.54 pts^2 (+10.0% of the realised residual variance).
