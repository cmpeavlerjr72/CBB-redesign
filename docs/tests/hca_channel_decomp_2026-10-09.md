# D1 home margin: level, channel decomposition, slope (fold 2 / 2024-25), 2026-10-09

DIAGNOSTIC ONLY. Nothing changed, no retrain. Script `scripts/diag_c4_d1_hca_v1.py` (library `diag_c4_lib_v1.py`, extraction `diag_c4_extract_v1.py`).

Sim: served stack v3, `engine_v3` inputs, fold 2 / season 2025, **5710 games x 50 seeds** per-possession trajectories (`results/trajectories/_runs/f2all50_p*`). Real: verified finals (`game_finals_v2`) for margins; hoopR/CBBD possession table `possessions_v4otc` for channels. 2025-26 not read. Fold 1 not run (the F1 engine inputs/overlay are not in the served `engine_v3` set; see report).

Definitions. margin = home - away at the final whistle including OT. Sim cell = mean over seeds of the per-game margin, then mean over games; `seed SE` = SD over seeds of the game-set mean / sqrt(S) (Monte Carlo error of the sim mean only). real - sim: paired by game, SE = SD of the per-game difference / sqrt(n) (includes real-game sampling noise and sim seed noise). Underpowered = n < 60 games.

## (a) Mean home margin, sim vs real

| cell | n games | real | sim | seed SE | real - sim | SE (paired) | 95% | flag |
|---|---:|---:|---:|---:|---:|---:|---|---|
| ALL games | 5710 | 5.42 | 5.20 | 0.025 | 0.22 | 0.16 | [-0.08, 0.53] |  |
| non-neutral | 4974 | 5.74 | 5.65 | 0.027 | 0.09 | 0.17 | [-0.24, 0.42] |  |
| neutral (listed home) | 736 | 3.29 | 2.16 | 0.065 | 1.13 | 0.41 | [0.34, 1.92] | excludes 0 |
| non-neutral, 2024-11 | 921 | 12.24 | 11.44 | 0.069 | 0.79 | 0.44 | [-0.08, 1.66] |  |
| non-neutral, 2024-12 | 841 | 8.58 | 8.35 | 0.063 | 0.23 | 0.41 | [-0.58, 1.04] |  |
| non-neutral, 2025-01 | 1422 | 2.49 | 3.29 | 0.045 | -0.80 | 0.31 | [-1.40, -0.20] | excludes 0 |
| non-neutral, 2025-02 | 1364 | 3.33 | 3.11 | 0.044 | 0.22 | 0.30 | [-0.37, 0.80] |  |
| non-neutral, 2025-03 | 426 | 4.62 | 3.78 | 0.068 | 0.85 | 0.56 | [-0.26, 1.95] |  |
| non-neutral, conference game | 3348 | 3.06 | 3.28 | 0.029 | -0.22 | 0.20 | [-0.61, 0.17] |  |
| non-neutral, non-conference game | 1626 | 11.24 | 10.52 | 0.053 | 0.72 | 0.32 | [0.10, 1.34] | excludes 0 |
| non-neutral, regular season | 4947 | 5.74 | 5.65 | 0.027 | 0.10 | 0.17 | [-0.23, 0.43] |  |
| non-neutral, postseason | 27 | 4.56 | 6.32 | 0.307 | -1.76 | 2.14 | [-5.96, 2.43] | UNDERPOWERED |
| the 5-date window of the trajectory doc (all games) | 394 | 4.04 | 2.56 | 0.086 | 1.48 | 0.57 | [0.36, 2.60] | excludes 0 |

Reading: a pooled mean-margin gap is partly a strength-mix effect (home teams are the favourite more often); the intercept at equal rating is in (c).

## (b) Channel decomposition of the home-minus-away offence gap (non-neutral games)

Games kept: 4791 of 4974 non-neutral (real pbp path final = verified final, >= 90 parsed possessions). Regulation possessions only (period <= 2) so OT does not enter the rates. Sim rates are computed from seed-summed counts per game side; real from the same counts in the possession table; the channel definitions are identical (one code path).

| channel | real home-off | real away-off | real gap (H-A) | sim gap (H-A) | real - sim gap | 95% (game bootstrap) | flag |
|---|---:|---:|---:|---:|---:|---|---|
| poss / team | 67.565 | 67.604 | -0.039 | -0.008 | -0.031 | [-0.069, 0.005] |  |
| PPP | 1.106 | 1.023 | 0.083 | 0.083 | 0.001 | [-0.004, 0.006] |  |
| eFG% | 52.48 | 49.47 | 3.01 pp | 3.19 pp | -0.18 pp | [-0.48, 0.11] |  |
| 2P% | 52.79 | 49.32 | 3.47 pp | 3.54 pp | -0.07 pp | [-0.41, 0.28] |  |
| 3P% | 34.68 | 33.14 | 1.54 pp | 1.76 pp | -0.22 pp | [-0.64, 0.17] |  |
| 3PA share | 39.43 | 38.76 | 0.66 pp | 0.80 pp | -0.14 pp | [-0.45, 0.18] |  |
| FTA / poss | 29.03 | 25.48 | 3.55 pp | 2.93 pp | 0.62 pp | [0.29, 0.95] | excludes 0 |
| FT% | 72.44 | 71.50 | 0.94 pp | 1.48 pp | -0.54 pp | [-0.98, -0.11] | excludes 0 |
| TOV / poss | 16.63 | 18.14 | -1.51 pp | -1.34 pp | -0.17 pp | [-0.35, 0.03] |  |
| OREB / poss | 15.58 | 14.98 | 0.60 pp | 0.44 pp | 0.15 pp | [-0.08, 0.35] |  |
| OREB % of misses | 33.59 | 30.97 | 2.62 pp | 2.36 pp | 0.26 pp | [-0.14, 0.58] |  |
| sec / poss | 17.569 | 17.894 | -0.325 | -0.236 | -0.088 | [-0.162, -0.006] | excludes 0 |

Gap = pooled home-offence rate minus pooled away-offence rate over the same games. Home teams are stronger on average, so the raw gap mixes site and strength; the sim and real share the same games, so a strength bias in the sim is common to both only if the sim is unbiased on strength (checked in D2). The FE-adjusted version follows.

### Margin contribution of each channel (points of home margin per game; LMDI exact decomposition of the PPP gap x mean possessions)

Attempts per possession regressed on TOV, OREB and FTA rates (real team-games, n=9582): beta TOV -0.997, OREB 0.984, FTA -0.489; used only to split the attempts-per-possession term.

| channel | real contribution | sim contribution | real - sim | 95% (game bootstrap) | flag |
|---|---:|---:|---:|---|---|
| 2P% make | 2.430 | 2.499 | -0.069 | [-0.31, 0.19] |  |
| 3P% make | 1.036 | 1.181 | -0.145 | [-0.42, 0.12] |  |
| 3PA share | -0.001 | -0.001 | -0.001 | [-0.00, 0.00] |  |
| FT% | 0.173 | 0.281 | -0.108 | [-0.19, -0.03] | excludes 0 |
| FTA rate (direct) | 1.725 | 1.425 | 0.299 | [0.14, 0.46] | excludes 0 |
| TOV (via attempts) | 1.038 | 0.936 | 0.102 | [-0.03, 0.23] |  |
| OREB (via attempts) | 0.405 | 0.305 | 0.099 | [-0.06, 0.23] |  |
| FTA-trip (via attempts) | -1.194 | -0.999 | -0.195 | [-0.31, -0.08] | excludes 0 |
| other attempts/poss | 0.014 | 0.016 | -0.002 | [-0.04, 0.04] |  |
| possession parity (home - away off. possessions) | -0.041 | -0.009 | -0.033 | [-0.07, 0.00] | |
| **total (home-off PPP gap x poss + parity)** | 5.584 | 5.636 | -0.052 | [-0.39, 0.31] | |

Sanity: real regulation home-minus-away points per game in these games = 5.584; sim = 5.636. (Contributions sum to the PPP-gap identity x mean possessions; the identity closes up to the exactness of LMDI and the pooled-vs-mean weighting.)

### Team-FE-adjusted site effect (rate ~ offence team + defence team + home-offence indicator; non-neutral games kept above)

| channel | real FE site effect | sim FE site effect | real - sim | 95% (game bootstrap, 100 reps) |
|---|---:|---:|---:|---|
| PPP | 0.043 | 0.049 | -0.005 | [-0.012, -0.000] |
| eFG% | 1.365 pp | 1.900 pp | -0.535 pp | [-0.885, -0.312] |
| 2P% | 1.522 pp | 2.065 pp | -0.544 pp | [-0.898, -0.272] |
| 3P% | 0.768 pp | 1.093 pp | -0.325 pp | [-0.713, 0.034] |
| FTA / poss | 2.505 pp | 2.247 pp | 0.258 pp | [-0.103, 0.638] |
| FT% | 0.541 pp | 0.867 pp | -0.325 pp | [-0.858, 0.196] |
| TOV / poss | -1.043 pp | -0.890 pp | -0.152 pp | [-0.362, 0.062] |
| OREB / poss | 0.279 pp | 0.077 pp | 0.202 pp | [0.005, 0.397] |
| poss / team | -0.032 | -0.006 | -0.026 | [-0.063, 0.012] |

FE-site PPP effect x 67.9 possessions = real 2.95 pts, sim 3.32 pts of home advantage per side (x2 is not applied: the indicator is home-offence minus away-offence, i.e. the full home-minus-away gap of one team's offence); poss/team FE site effect real -0.032, sim -0.006.

## (c) Slope check: home margin by home-team rating quintile and by |spread| quintile (non-neutral)

| home-rating quintile (Q1 weakest home team) | n | mean extra | real margin | sim margin | real - sim | 95% | flag |
|---|---:|---:|---:|---:|---:|---|---|
| Q1 | 974 | -14.64 | -1.66 | -1.89 | 0.23 | [-0.49, 0.94] |  |
| Q2 | 974 | -6.07 | 2.44 | 2.57 | -0.13 | [-0.86, 0.59] |  |
| Q3 | 974 | 0.02 | 5.34 | 5.33 | 0.01 | [-0.71, 0.73] |  |
| Q4 | 974 | 7.44 | 7.83 | 8.19 | -0.36 | [-1.12, 0.40] |  |
| Q5 | 974 | 19.89 | 13.44 | 13.10 | 0.34 | [-0.45, 1.12] |  |

| |close spread| quintile (Q1 closest) | n | mean extra | real margin | sim margin | real - sim | 95% | flag |
|---|---:|---:|---:|---:|---:|---|---|
| Q1 | 909 | 1.74 | 0.73 | 0.42 | 0.31 | [-0.40, 1.02] |  |
| Q2 | 908 | 3.77 | 0.98 | 1.31 | -0.33 | [-1.11, 0.44] |  |
| Q3 | 908 | 6.44 | 2.42 | 3.06 | -0.64 | [-1.40, 0.12] |  |
| Q4 | 908 | 10.19 | 5.53 | 6.01 | -0.48 | [-1.23, 0.27] |  |
| Q5 | 909 | 20.12 | 18.40 | 17.13 | 1.27 | [0.45, 2.10] | excludes 0 |

Regression of final margin on the pre-game predictor, real and sim fitted on the same games (intercept = home margin at zero predictor; slope = points of margin per unit).

| predictor | n | real intercept | sim intercept | int. diff 95% | real slope | sim slope | slope diff 95% |
|---|---:|---:|---:|---|---:|---:|---|
| own rating diff (home_net - away_net) | 4870 | 3.54 | 3.54 | [-0.28, 0.34] | 0.723 | 0.717 | [-0.02, 0.03] |
| -close spread (market) | 4542 | -0.08 | 0.37 | [-0.80, -0.11] | 1.017 | 0.932 | [0.04, 0.12] |

### Per home team: (real - sim) home margin

363 home teams with >= 8 non-neutral home games (mean n 13.7). SD across teams of the mean (real - sim) = 3.30; expected SD from game-level noise alone = 3.22 (ratio 1.03; ratio near 1 means no team-specific component beyond noise). Share of teams with (real - sim) > 0: 0.477. Per-team cells are individually underpowered (SE about 3.2 points).

## (d) Verdict

**No scoring-stage site term is under-responsive on net; the home-margin level is not identified as a defect.** On all of fold 2 at 50 seeds (5,710 games x 50 seeds; non-neutral n = 4,974, powered) the sim's mean home margin is +5.65 against +5.74 real (real - sim +0.09, 95% [-0.24, 0.42]); all games +0.22 [-0.08, 0.53]. The +1.2/+1.5 gap in the 394-game trajectory window (real 4.04 vs sim 2.56, z about 2.6) does not survive the full fold: it was a sampling draw of one window, not a level offset. Month cells run -0.80 (Jan, excludes 0) to +0.85 (Mar) with alternating sign (chi-square of the five monthly z's = 13 on 5 df: some calendar heterogeneity, no stable sign); conference games -0.22 [-0.61, 0.17], non-conference +0.72 [0.10, 1.34] (a strength-mix cell, see Q5 of |spread| +1.27). Channel decomposition (4,791 games with a matched pbp path): the total home-minus-away points identity is 5.58 real vs 5.64 sim (diff -0.05 [-0.39, 0.31]); the only channels excluding 0 are the free-throw block (FTA rate gap +0.62 pp [0.29, 0.95] = +0.30 pts; FT% -0.54 pp = -0.11 pts; FTA-trip via attempts -0.20 pts), and they net to about 0. The team-FE-adjusted site effect is, if anything, slightly LARGER in the sim than in reality on make rates (eFG -0.54 pp real - sim [-0.89, -0.31], 2P% -0.54 pp, PPP -0.005 [-0.012, 0.000], about -0.37 pts), partly offset by a smaller sim OREB site effect (+0.20 pp [0.005, 0.40]) and FTA (+0.26 pp, not significant): fg_make's 2P site term is mildly over-responsive and possession_outcome's FT-trip / rebound site terms mildly under-responsive, each worth well under half a point and cancelling. Slope: margin on our own rating has identical intercept (3.54 vs 3.54) and slope (0.723 vs 0.717); on the market line the sim's slope is 0.932 vs 1.017 real (diff [0.04, 0.12]) and its intercept 0.37 vs -0.08, i.e. the sim follows OUR rating faithfully and is less responsive than reality to the market's information (D2), not to site. Per home team the (real - sim) dispersion is 1.03 x what game noise alone gives (363 teams): no team-specific home component. The one powered level gap left is the listed-home margin at NEUTRAL sites (+1.13 [0.34, 1.92], n = 736), the known G6 neutral line, which is a listing/strength question and not a home-court term. Fold 1 was not run (fold-1 engine inputs for the served v3 stack do not exist). Nothing to draft for D1.
