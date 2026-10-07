# Early-season total gap in points, possession-replacement accounting (2026-10-07)

Diagnostic only; nothing fitted or adopted; 2025-26 sealed (not read). Worker analysis for the PM.

## Method

- Sim: served-v3 50-seed control taps `results/engine_v0/fcal_F{1,2}_ctrl_s50` (the control of foul-accrual round 2; F1 = 2023-24, F2 = 2024-25). Truth: verified finals, hoopR team box (TOV, OREB, FTA, FTM), event shot classes. Games without event shot classes are dropped (as in the 2026-10-05 decomp), so total-bias levels differ slightly from the loop grader's (F1 d0-14 -5.06 here vs -4.93; F2 -4.59 vs -5.16; the dropped box-only games carry roughly 0.6 pts of the F2 early gap).
- Identity per team-game, with possessions P fixed: `FGA = P (1 - tov_r + oreb_r - 0.44 fta_r)`, `pts = 2 FGA (1-s3) (r p_rim + (1-r) p_jump) + 3 FGA s3 p3 + P fta_r p_ft`. Possession replacement is inside the identity: raising TOV, OREB or FTA rate at fixed P removes or adds shots, so the FTA channel is already net of displaced shots, TOV is net of the lost shot, and so on.
- Split: exact Shapley over 10 factors (P, tov_r, oreb_r, fta_r, s3, rim share, p_rim, p_jump, p3, p_ft) on bucket-mean counts; parts sum to the identity gap; residual = total gap minus identity gap (and-ones, technicals, class drift). Script `scripts/diag_early_total_points_decomp_v1.py` (reuses `scripts/diag_total_bias_decomp_v1.py`), outputs in `results/early_total_points/` (gitignored). A one-factor-at-a-time swap (sim to actual, others held at sim; sign reversed) agrees with the Shapley values within about 0.1 pt for every d0-14 channel above; Shapley is reported.
- Channel mapping: "possessions" = P (pace); "shot mix" = 3PA share + rim share; "second chances" = OREB rate (net of the displaced shot).

## 1. Points per game, sim minus actual (negative = sim low)

| channel | F2 d0-14 | F1 d0-14 | F2 d46+ | F1 d46+ |
|---|---|---|---|---|
| games | 572 | 657 | 3827 | 3723 |
| **total gap** | **-4.59** | **-5.06** | **+0.65** | **-0.67** |
| possessions (pace) | -0.92 | -0.51 | +0.79 | +0.55 |
| TOV rate | -0.57 | **-2.20** | -0.17 | -0.87 |
| OREB / second chances | -0.23 | -0.64 | -0.31 | +0.33 |
| FTA rate (net of displaced shots) | -0.70 | -0.90 | +0.13 | -0.02 |
| FT% | **-1.37** | -1.06 | -0.20 | -0.41 |
| shot mix: rim share | -0.38 | -0.56 | -0.10 | +0.11 |
| shot mix: 3PA share | +0.06 | -0.02 | 0.00 | 0.00 |
| rim make | -0.02 | +0.69 | +0.05 | +0.22 |
| jump2 make | -0.67 | -0.42 | +0.51 | -0.36 |
| 3 make | +0.21 | +0.37 | +0.15 | -0.38 |
| residual | 0.00 | +0.18 | -0.19 | +0.16 |

d15-45 (not asked): F2 -0.30, F1 -1.83. Sum of the three field-goal make-rate channels: F2 d0-14 -0.48, F1 +0.64, so make rates are not the largest family; FT% is the only make rate that is large and shared.

Rates behind the table (sim / actual, d0-14, F2 | F1): TOV/P 0.185/0.181 | 0.198/0.182; FTA/P 0.272/0.292 | 0.258/0.284; FT% 0.673/0.708 | 0.675/0.703; OREB/P 0.153/0.154 | 0.149/0.154; rim share of 2PA 0.598/0.611 | 0.578/0.596.

### Reading
1. The earlier attribution is confirmed in points: the whole FTA-rate channel is -0.70 / -0.90, in line with the 0.7-0.9 implied by s34.0. FT% (make rate on the FTs actually taken) is a separate -1.4 / -1.1.
2. The shared block (both folds, similar size) is FT% -1.1 to -1.4, FTA rate -0.7 to -0.9, possessions -0.5 to -0.9, rim share -0.4 to -0.6, jump2 make -0.4 to -0.7: about -3.6 (F2) and -3.0 (F1). F1 adds TOV -2.2 and OREB -0.6, which is the cross-season TOV level step (sim 0.198 vs 0.182 early; d46+ still -0.87), not an early-season effect; F2 TOV is only -0.57.
3. Controls: d46+ is small and mixed in sign (F2 +0.65 with possessions +0.79 and jump2 +0.51; F1 -0.67 with TOV -0.87 and possessions +0.55). FT% (-1.4 / -1.1 early, -0.2 / -0.4 late) and FTA (about -0.8 early, about 0 late) are calendar effects. The possession shortfall early flips to an overshoot late (-0.9 to +0.8), so pace is mis-timed.

## 2. Per-team evidence

Team bucket = all games the team played in the window (both sides), at least 2 games, 352-362 teams per cell. Each team holds only 2-3 games in d0-14, so single-team values are underpowered; shares and slopes are the usable evidence. Top two channels by two-fold d0-14 size: TOV (-2.77 summed) and FT% (-2.43).

| channel | fold | share of teams negative | team mean (SD) | slope per prior-season total pt (corr) | quintile means Q1 (low prior) to Q5 |
|---|---|---|---|---|---|
| total gap | F1 | 69% | -5.39 (9.74) | -0.170 (-0.13) | -3.42 -4.39 -5.45 -7.64 -6.05 |
| total gap | F2 | 67% | -4.65 (10.77) | -0.075 (-0.05) | -4.09 -4.70 -4.08 -4.78 -5.59 |
| TOV | F1 | 79% | -2.21 (2.81) | +0.002 (+0.01) | -2.24 -1.78 -2.59 -2.48 -1.96 |
| TOV | F2 | 59% | -0.66 (3.14) | +0.014 (+0.03) | -0.80 -0.44 -0.96 -0.55 -0.54 |
| FT% | F1 | 75% | -1.09 (1.71) | -0.013 (-0.06) | -1.04 -0.95 -0.95 -1.31 -1.21 |
| FT% | F2 | 78% | -1.43 (1.92) | -0.017 (-0.06) | -1.18 -1.12 -1.69 -1.82 -1.34 |
| possessions | F1 / F2 | 53% / 53% | -0.70 / -0.92 | -0.075 / -0.052 | F1 Q1-Q2 about 0, Q3-Q5 -1.0 to -1.4 |
| FTA rate | F1 / F2 | 74% / 64% | -0.93 / -0.77 | +0.002 / -0.002 | flat |

Control d46+ (361 / 362 teams, F1 / F2): total gap negative in 56% / 43%; TOV negative 76% / 59% (mean -0.88 / -0.16); FT% negative 69% / 59% (mean -0.41 / -0.20).

- TOV (F1) and FT% (both folds) are broad: 75-79% of teams negative. They are flat across prior-environment quintiles (slopes within 0.02 of zero, quintile SE about 0.3-0.4), so they are level offsets, not a responsiveness failure, and they do not explain the mild total-gap slope (-0.17 F1, -0.075 F2). The channels that do carry slope are possessions (-0.075 / -0.052) and rim share (-0.027 F1), each within about 1 SE of flat (underpowered).
- Quintile cells (70 teams of 2-3 games each) are individually underpowered; only the shared sign and the flat slope are read.

## 3. Per-player cut for FT% (largest shared make-rate channel)

The FT sub-model and players are unchanged between served v2 and v3 (v3 differs in the clock), and the v3 control taps carry no `players.parquet`, so this cut uses the 200-seed served-v2 reads (`f1c_V2_full_s200_o0`, `v3full_COMB9GCTKD_s200_o0`); script `scripts/diag_early_ft_player_cut_v1.py`. Benchmarks: named = shrunk (20 att) full-season actual FT% of the same players (grading only); anonymous = actual window FT%. Anonymous FTA = game total minus named players (anonymous slots are absent from `players.parquet`). This share (43% / 37% in d0-14) is larger than the 16-21% quoted in the 2026-10-05 doc, so treat the anonymous row as an upper bound that may include unattributed FTA. Points per game = group FTA x (sim FT% - benchmark). Returner = player with FTA in the prior season's truth.

| group | F2 d0-14: FTA share, sim FT% vs bench, pts | F1 d0-14: same | F2 d46+ pts | F1 d46+ pts |
|---|---|---|---|---|
| anonymous slots | 37%, 0.645 vs 0.708, **-0.87** | 43%, 0.646 vs 0.702, **-0.87** | -0.02 | -0.13 |
| named newcomers | 17%, 0.658 vs 0.713, -0.36 | 15%, 0.666 vs 0.702, -0.19 | -0.08 | -0.10 |
| named returners | 46%, 0.702 vs 0.723, -0.37 | 42%, 0.706 vs 0.723, -0.26 | +0.01 | -0.02 |
| sum | -1.59 | -1.31 | -0.08 | -0.25 |

(The v2 Shapley FT% line is -1.37 / -1.08; the group sums run 0.2 pt larger because the benchmark is talent-based per group, not the window league mean.) Anonymous slots carry about 55-65% of the early FT% points, newcomers 15-25%, returners 20-25%. By d46+ the anonymous share of FTA is under 5% and the channel is near zero. Per FTA, anonymous and newcomer shortfalls are 4-6 pp against 1.5-2 pp for returners; the returner gap is small but nonzero, consistent with the offline FT model under-predicting early attempts (-1.4 pp returners in the 2026-10-05 doc).

## 4. Owner map

| channel | points (F2 / F1, d0-14) | owner sub-model |
|---|---|---|
| FT% | -1.4 / -1.1 | free-throw model (`shooter_fta_asof` raw-count defect, FT experiments section 18) plus player-layer day-1 / anonymous-slot shooter prior (A3, FT A1) |
| TOV | -0.6 / -2.2 | possession_outcome TOV; the F1 level is the cross-season `season_drift` anchor, F2 is small |
| FTA rate | -0.7 / -0.9 | foul accrual / FT trip rate (round 2: not recoverable through accrual in points) |
| possessions | -0.9 / -0.5 | clock (K2 pace; flips to +0.6 to +0.8 at d46+) |
| jump2 make, rim share | -0.7, -0.4 / -0.4, -0.6 | shot-make and shot-selection models (each small, 1-2 SE of game noise) |

Game-level SE of the total gap is about 0.7 per fold window, so channel values below about 0.5 are underpowered against game noise (the paired sim noise at 50 seeds is much smaller, but the actual-side noise is the same).
