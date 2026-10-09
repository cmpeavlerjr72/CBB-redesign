# D2 responsiveness to strength: totals, possessions vs PPP, rating compression, lead-change slopes (fold 2 / 2024-25), 2026-10-09

DIAGNOSTIC ONLY. Nothing changed. Script `scripts/diag_c4_d2_strength_v1.py` on `diag_c4_extract_v1.py` caches.

Sim: served stack v3, fold 2 / season 2025, 5710 games x 50 seeds. Games analysed: 5424 with own as-of ratings, real pbp path = verified final and >= 90 possessions (neutral included unless stated). Rating = own as-of net (off_c - def_c), day-before snapshot. Spread = ESPN BET close (n with line = 5106). Totals include OT unless 'regulation'; channels are regulation (period <= 2). Real game sampling noise is in every CI (game-cluster bootstrap, 300 reps); the sim is seed-averaged (seed MC SE of a quintile mean is shown where relevant). Underpowered = n < 60.

## (a) By home-team rating quintile (Q1 weakest home team): total points gap = possessions + points per possession; PPP by strong / opposing offence; PPP by channel

Total here is REGULATION points (both teams) so the channel split closes; the OT-inclusive total gap is in the first table's last column.

| cell | n | mean home_net | real total (incl OT) | sim total (incl OT) | real - sim [95%] | seed SE | reg total real | reg total sim | gap reg | of which possessions | of which PPP |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Q1 | 1085 | -14.13 | 143.64 | 144.40 | -0.76 [-1.75, 0.22] | 0.071 | 141.64 | 143.43 | -1.79 | -2.30 | 0.51 |
| Q2 | 1085 | -5.34 | 144.92 | 145.42 | -0.50 [-1.56, 0.56] | 0.079 | 142.92 | 144.49 | -1.58 | -2.24 | 0.67 |
| Q3 | 1084 | 0.88 | 145.58 | 145.96 | -0.38 [-1.41, 0.66] | 0.075 | 143.68 | 145.10 | -1.41 | -2.04 | 0.63 |
| Q4 | 1085 | 8.57 | 145.75 | 145.83 | -0.08 [-1.07, 0.92] | 0.076 | 144.07 | 145.09 | -1.02 | -1.87 | 0.85 |
| Q5 | 1085 | 21.27 | 147.94 | 146.76 | 1.18 [0.16, 2.21] | 0.081 | 146.28 | 146.09 | 0.20 | -0.86 | 1.06 |

| cell | side | n team-games | poss/team real | sim | PPP real | sim | PPP gap [95% game bootstrap] | eFG% real/sim | FTA/poss real/sim | TOV/poss real/sim | OREB/poss real/sim |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| Q1 | strong-side offence | 1085 | 67.21 | 68.29 | 1.085 | 1.083 | 0.002 [-0.006, 0.009] | 51.87 / 51.89 | 28.35 / 29.12 | 17.23 / 17.26 | 15.55 / 15.40 |
| Q1 | opposing offence | 1085 | 67.20 | 68.31 | 1.022 | 1.017 | 0.006 [-0.003, 0.013] | 49.56 / 49.51 | 26.96 / 27.84 | 18.01 / 18.11 | 14.63 / 14.42 |
| Q2 | strong-side offence | 1085 | 67.50 | 68.50 | 1.086 | 1.087 | -0.001 [-0.008, 0.008] | 51.85 / 52.30 | 28.43 / 28.89 | 17.17 / 17.16 | 15.71 / 14.97 |
| Q2 | opposing offence | 1084 | 67.45 | 68.51 | 1.033 | 1.023 | 0.010 [0.003, 0.020] | 49.47 / 49.68 | 26.84 / 27.53 | 18.02 / 18.04 | 15.36 / 14.58 |
| Q3 | strong-side offence | 1084 | 67.39 | 68.38 | 1.111 | 1.108 | 0.003 [-0.005, 0.011] | 52.98 / 53.05 | 28.66 / 29.04 | 16.64 / 16.64 | 15.40 / 14.96 |
| Q3 | opposing offence | 1084 | 67.46 | 68.40 | 1.020 | 1.014 | 0.006 [-0.003, 0.013] | 48.83 / 49.28 | 26.39 / 26.43 | 17.54 / 17.78 | 15.12 / 14.40 |
| Q4 | strong-side offence | 1085 | 67.58 | 68.51 | 1.133 | 1.126 | 0.007 [-0.001, 0.016] | 53.69 / 53.66 | 28.90 / 28.85 | 16.26 / 16.39 | 15.57 / 15.22 |
| Q4 | opposing offence | 1085 | 67.70 | 68.53 | 0.997 | 0.992 | 0.006 [-0.001, 0.013] | 48.37 / 48.49 | 25.30 / 25.54 | 18.25 / 18.20 | 14.56 / 14.16 |
| Q5 | strong-side offence | 1085 | 67.83 | 68.27 | 1.169 | 1.156 | 0.013 [0.004, 0.021] | 54.54 / 54.12 | 29.11 / 29.10 | 15.59 / 15.71 | 16.52 / 16.32 |
| Q5 | opposing offence | 1085 | 67.92 | 68.28 | 0.986 | 0.983 | 0.003 [-0.004, 0.011] | 47.64 / 47.92 | 25.05 / 25.35 | 18.47 / 18.24 | 14.61 / 14.19 |

PPP gap (real - sim) by channel, points per team-game (LMDI x possessions), all offences in the cell:

| cell | 2P% make | 3P% make | 3PA share | FT% | FTA rate (direct) | TOV (via attempts) | OREB (via attempts) | FTA-trip (via attempts) | other attempts/poss | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.00 | 0.01 | 0.00 | 0.18 | -0.40 | 0.05 | 0.12 | 0.27 | 0.01 | 0.25 |
| Q2 | -0.23 | -0.15 | 0.00 | 0.17 | -0.28 | -0.00 | 0.52 | 0.19 | 0.11 | 0.33 |
| Q3 | -0.25 | -0.05 | 0.00 | 0.08 | -0.10 | 0.09 | 0.39 | 0.07 | 0.09 | 0.31 |
| Q4 | 0.01 | -0.07 | -0.00 | 0.14 | -0.05 | 0.03 | 0.26 | 0.03 | 0.07 | 0.43 |
| Q5 | 0.20 | -0.11 | -0.01 | 0.08 | -0.07 | -0.04 | 0.22 | 0.05 | 0.22 | 0.53 |

## (a) By |close spread| quintile (Q1 closest): total points gap = possessions + points per possession; PPP by strong / opposing offence; PPP by channel

Total here is REGULATION points (both teams) so the channel split closes; the OT-inclusive total gap is in the first table's last column.

| cell | n | mean |spread| | real total (incl OT) | sim total (incl OT) | real - sim [95%] | seed SE | reg total real | reg total sim | gap reg | of which possessions | of which PPP |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Q1 | 1085 | 1.78 | 145.51 | 145.72 | -0.20 [-1.25, 0.85] | 0.069 | 143.13 | 144.70 | -1.57 | -2.28 | 0.71 |
| Q2 | 1085 | 3.93 | 144.46 | 145.44 | -0.98 [-2.03, 0.07] | 0.076 | 142.33 | 144.40 | -2.07 | -2.51 | 0.44 |
| Q3 | 766 | 6.19 | 145.42 | 145.69 | -0.28 [-1.45, 0.89] | 0.103 | 143.55 | 144.79 | -1.24 | -1.97 | 0.73 |
| Q4 | 1085 | 9.30 | 146.18 | 146.15 | 0.03 [-0.98, 1.05] | 0.073 | 144.64 | 145.38 | -0.73 | -1.67 | 0.94 |
| Q5 | 1085 | 19.09 | 146.48 | 145.35 | 1.13 [0.14, 2.13] | 0.068 | 145.28 | 144.91 | 0.37 | -0.74 | 1.11 |

| cell | side | n team-games | poss/team real | sim | PPP real | sim | PPP gap [95% game bootstrap] | eFG% real/sim | FTA/poss real/sim | TOV/poss real/sim | OREB/poss real/sim |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| Q1 | strong-side offence | 1085 | 67.05 | 68.13 | 1.066 | 1.067 | -0.001 [-0.010, 0.008] | 50.99 / 51.16 | 26.84 / 27.96 | 17.25 / 17.11 | 15.30 / 14.96 |
| Q1 | opposing offence | 1085 | 67.08 | 68.13 | 1.068 | 1.057 | 0.011 [0.004, 0.019] | 50.83 / 50.88 | 28.20 / 28.44 | 17.13 / 17.22 | 15.42 / 14.70 |
| Q2 | strong-side offence | 1085 | 67.22 | 68.36 | 1.073 | 1.075 | -0.002 [-0.009, 0.007] | 51.11 / 51.60 | 27.95 / 28.49 | 17.10 / 17.17 | 15.54 / 15.03 |
| Q2 | opposing offence | 1084 | 67.19 | 68.37 | 1.045 | 1.037 | 0.008 [0.001, 0.016] | 49.64 / 50.05 | 27.81 / 27.73 | 17.28 / 17.46 | 15.47 / 14.62 |
| Q3 | strong-side offence | 766 | 67.50 | 68.43 | 1.100 | 1.101 | -0.000 [-0.010, 0.009] | 52.43 / 52.60 | 28.97 / 29.18 | 16.70 / 16.80 | 15.23 / 15.05 |
| Q3 | opposing offence | 766 | 67.52 | 68.45 | 1.026 | 1.015 | 0.011 [0.001, 0.020] | 49.35 / 49.32 | 26.20 / 26.63 | 17.55 / 17.89 | 14.55 / 14.37 |
| Q4 | strong-side offence | 1085 | 67.68 | 68.49 | 1.137 | 1.127 | 0.010 [0.003, 0.018] | 53.82 / 53.62 | 29.75 / 29.50 | 16.26 / 16.37 | 15.87 / 15.38 |
| Q4 | opposing offence | 1085 | 67.75 | 68.51 | 0.999 | 0.995 | 0.004 [-0.004, 0.013] | 48.33 / 48.57 | 25.89 / 26.01 | 18.32 / 18.23 | 14.73 / 14.30 |
| Q5 | strong-side offence | 1085 | 68.32 | 68.72 | 1.203 | 1.186 | 0.017 [0.010, 0.026] | 56.39 / 55.85 | 30.27 / 29.89 | 15.66 / 15.81 | 16.75 / 16.39 |
| Q5 | opposing offence | 1085 | 68.45 | 68.74 | 0.922 | 0.923 | -0.001 [-0.010, 0.007] | 45.58 / 45.99 | 22.68 / 23.86 | 19.94 / 19.62 | 14.19 / 13.74 |

PPP gap (real - sim) by channel, points per team-game (LMDI x possessions), all offences in the cell:

| cell | 2P% make | 3P% make | 3PA share | FT% | FTA rate (direct) | TOV (via attempts) | OREB (via attempts) | FTA-trip (via attempts) | other attempts/poss | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | -0.17 | 0.05 | 0.00 | 0.16 | -0.33 | -0.02 | 0.36 | 0.23 | 0.08 | 0.35 |
| Q2 | -0.19 | -0.33 | -0.00 | 0.13 | -0.11 | 0.09 | 0.46 | 0.08 | 0.09 | 0.22 |
| Q3 | -0.16 | 0.08 | 0.00 | 0.08 | -0.16 | 0.15 | 0.12 | 0.11 | 0.15 | 0.37 |
| Q4 | -0.04 | 0.02 | -0.00 | 0.11 | 0.03 | 0.01 | 0.32 | -0.02 | 0.05 | 0.47 |
| Q5 | 0.23 | -0.17 | -0.01 | 0.19 | -0.20 | -0.06 | 0.28 | 0.14 | 0.15 | 0.56 |

### Where in the game the possession gap lives: possessions per team-game and seconds per possession, by half

| cut | quintile | n games | H1 poss/team real / sim | H2 poss/team real / sim | H1 sec/poss real / sim | H2 sec/poss real / sim |
|---|---|---:|---|---|---|---|
| home-rating | Q1 | 1085 | 33.13 / 33.63 | 34.08 / 34.67 | 18.09 / 17.84 | 17.64 / 17.30 |
| home-rating | Q2 | 1085 | 33.38 / 33.77 | 34.09 / 34.73 | 17.96 / 17.77 | 17.54 / 17.28 |
| home-rating | Q3 | 1084 | 33.42 / 33.70 | 34.01 / 34.68 | 17.92 / 17.80 | 17.57 / 17.30 |
| home-rating | Q4 | 1085 | 33.60 / 33.80 | 34.04 / 34.72 | 17.85 / 17.75 | 17.62 / 17.28 |
| home-rating | Q5 | 1085 | 33.69 / 33.69 | 34.19 / 34.59 | 17.80 / 17.81 | 17.52 / 17.35 |
| |spread| | Q1 | 1085 | 33.23 / 33.57 | 33.84 / 34.56 | 18.04 / 17.87 | 17.68 / 17.36 |
| |spread| | Q2 | 1085 | 33.28 / 33.69 | 33.92 / 34.68 | 18.02 / 17.81 | 17.66 / 17.30 |
| |spread| | Q3 | 766 | 33.45 / 33.73 | 34.06 / 34.71 | 17.89 / 17.79 | 17.66 / 17.28 |
| |spread| | Q4 | 1085 | 33.51 / 33.77 | 34.21 / 34.73 | 17.89 / 17.77 | 17.57 / 17.28 |
| |spread| | Q5 | 1085 | 33.88 / 33.93 | 34.50 / 34.80 | 17.69 / 17.68 | 17.31 / 17.24 |

## (b) Is the compression in the rating itself or in the rating -> rate mapping?

Regression of the game outcome on our pre-game rating, real outcome vs sim seed-mean outcome on the SAME games and SAME predictor. If the sim passed the rating's information through faithfully, the two slopes agree; a sim slope BELOW the real slope means the strength spread is compressed between the rating and the simulated game. G9 (realised on sim) is shown for reference.

| outcome ~ predictor | n | real slope [95%] | sim slope [95%] | sim / real | real int | sim int | slope diff (real - sim) 95% |
|---|---:|---|---|---:|---:|---:|---|
| home margin ~ rating diff (non-neutral) | 4724 | 0.723 [0.693, 0.748] | 0.716 [0.708, 0.725] | 0.992 | 3.55 | 3.56 | [-0.022, 0.031] |
| total points ~ sum of off ratings - s x def ratings (all) | 5424 | 0.606 [0.553, 0.667] | 0.408 [0.387, 0.429] | 0.674 | 145.51 | 145.63 | [0.146, 0.251] |

G9-style: realised margin on sim seed-mean margin, non-neutral, n=4724: slope 0.929 (1 = no over-spread). Realised margin on rating diff alone: see row 1; if the sim slope on the rating is about equal to the real slope, the rating -> margin mapping is not compressed, and the over-spread seen in G9 is not a strength-compression issue.

### Per-channel slopes: team-game rate on the offence's own strength and the opponent's defence quality

x_off = off_c(team) - s x def_c(opponent) (expected offensive strength of this team-game, rating units). Slope of each channel on x_off with team-game level OLS, real vs sim (sim = seed-mean per team-game), game-cluster bootstrap of the DIFFERENCE. Slope ratio sim/real < 1 means the sim channel responds less to strength than reality.

| channel | n team-games | real slope per rating unit | sim slope | sim / real | diff (real - sim) 95% | flag |
|---|---:|---:|---:|---:|---|---|
| PPP | 10847 | 0.0098 | 0.0096 | 0.97 | [-0.0001, 0.0006] |  |
| eFG% | 10847 | 0.3923 pp | 0.3639 | 0.93 | [0.0031, 0.0495] | excludes 0 |
| 2P% | 10847 | 0.4514 pp | 0.4142 | 0.92 | [0.0126, 0.0543] | excludes 0 |
| 3P% | 10847 | 0.2071 pp | 0.1917 | 0.93 | [-0.0150, 0.0404] |  |
| 3PA share | 10847 | 0.0809 pp | 0.0984 | 1.22 | [-0.0375, -0.0001] | excludes 0 |
| FTA/poss | 10847 | 0.1887 pp | 0.1504 | 0.80 | [0.0174, 0.0605] | excludes 0 |
| FT% | 10847 | 0.1463 pp | 0.2234 | 1.53 | [-0.1044, -0.0442] | excludes 0 |
| TOV/poss | 10847 | -0.1709 pp | -0.1685 | 0.99 | [-0.0127, 0.0077] |  |
| OREB/poss | 10847 | 0.0539 pp | 0.0646 | 1.20 | [-0.0268, 0.0055] |  |
| poss/team | 10847 | 0.0445 | 0.0027 | 0.06 | [0.0325, 0.0515] | excludes 0 |
| sec/poss | 10847 | -0.0705 | -0.0200 | 0.28 | [-0.0553, -0.0451] | excludes 0 |

## (c) Lead-change and time-of-decision slopes against the pre-game |spread| (continuous, per point of spread)

n = 5106 games with a close line. Real = one realised path per game; sim = seed-mean per game (so sim noise is small). CI = game-cluster bootstrap (300 reps); difference CI is of (real - sim).

| statistic | real slope [95%] | sim slope [95%] | sim / real | diff (real - sim) 95% | real mean | sim mean |
|---|---|---|---:|---|---:|---:|
| lead changes | -0.1180 [-0.1318, -0.1029] | -0.0934 [-0.0959, -0.0908] | 0.79 | [-0.0385, -0.0084] | 4.593 | 4.440 |
| ties | -0.0793 [-0.0884, -0.0706] | -0.0625 [-0.0642, -0.0609] | 0.79 | [-0.0258, -0.0076] | 2.997 | 2.978 |
| time of decision (s elapsed) | -37.8842 [-40.8640, -35.4137] | -28.7972 [-29.4426, -28.1675] | 0.76 | [-12.3490, -6.4887] | 1165.617 | 1110.969 |
| largest lead | 0.6193 [0.5756, 0.6606] | 0.5029 [0.4855, 0.5203] | 0.81 | [0.0684, 0.1607] | 17.989 | 18.359 |
| OT share | -0.0027 [-0.0034, -0.0021] | -0.0014 [-0.0015, -0.0014] | 0.53 | [-0.0020, -0.0006] | 0.056 | 0.035 |

By |spread| quintile (means):

| quintile | n | lead changes real | sim | t_decision real | sim | largest lead real | sim | OT share real | sim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 (|spread| 1.7) | 1022 | 5.23 | 4.94 | 1417 | 1272 | 15.00 | 16.01 | 0.081 | 0.043 |
| Q2 (|spread| 3.7) | 1021 | 5.20 | 4.88 | 1357 | 1248 | 15.46 | 16.29 | 0.071 | 0.045 |
| Q3 (|spread| 6.2) | 1021 | 4.79 | 4.68 | 1226 | 1186 | 16.57 | 17.02 | 0.054 | 0.039 |
| Q4 (|spread| 9.7) | 1021 | 4.56 | 4.37 | 1119 | 1081 | 17.89 | 18.39 | 0.047 | 0.032 |
| Q5 (|spread| 19.5) | 1021 | 3.19 | 3.32 | 709 | 768 | 25.03 | 24.09 | 0.027 | 0.018 |

### Variance budget around the closing line by |spread| quintile (non-neutral games)

Real residual = final margin - (-close spread). Sim between = SD over games of (seed-mean margin - (-close spread)) (includes a little MC noise: SD/sqrt(50) of a seed mean is about 1.7); sim within = root mean over games of the across-seed variance of the final margin; sim total = sqrt(between^2 + within^2). If the sim's within-game spread is wider than the real residual SD the sim is over-dispersed per game (consistent with the G9 slope < 1: the spread is the right size but part of it carries no signal).

| quintile | n | real residual SD | sim between SD | sim within SD | sim total SD | sim total / real |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 882 | 10.65 | 3.13 | 12.09 | 12.49 | 1.172 |
| Q2 | 881 | 11.43 | 3.53 | 12.13 | 12.64 | 1.105 |
| Q3 | 882 | 11.38 | 3.15 | 12.13 | 12.53 | 1.101 |
| Q4 | 881 | 10.74 | 3.60 | 12.20 | 12.72 | 1.185 |
| Q5 | 882 | 11.72 | 4.58 | 12.49 | 13.31 | 1.135 |

### Supporting regressions (run for the verdict; script `scripts/diag_c4_d2_supp_v1.py`)

Game level, 5,510 games with ratings, regulation possessions per game (both teams), env = sum of both teams' off_c plus both defences' def_c (the scoring environment of the matchup; units are rating points, SD 8.3), gap = |home net - away net| (SD 8.0):

| regression of game possessions on | real | sim |
|---|---:|---:|
| env | +0.152 per unit | +0.011 |
| gap | +0.050 per unit | -0.008 |
| env and gap jointly | +0.152 / +0.050 | +0.011 / -0.008 |

Team points on own scoring strength x_h and the opponent's x_a (x = off_c of the team + def_c of the opponent), fold 2: real own +0.698 (home) / +0.608 (away), cross -0.065 / -0.043; sim own +0.554 / +0.564, cross -0.151 / -0.151. Total points on x_h: real 0.655, sim 0.403. Game possessions on x_h / x_a: real +0.199 / +0.104, sim +0.025 / -0.003. Reading: the sim's per-possession efficiency response is within 6-8% of real (own 0.55 vs 0.59 after removing the possession part, cross -0.15 vs -0.17); about 0.2 of the 0.25 points-per-unit total-slope gap is the possession count.

## (d) Verdict

**The compression lives in the pace/clock side (possession count and duration do not respond to the scoring environment or to mismatch), not in the rating estimator and mostly not in the efficiency model; owner: the clock model (served K2 duration law).** On all of fold 2 (5,424 rated games, 5,106 with a line) the window finding shrinks: the top home-rating quintile total is 147.9 real vs 146.8 sim (+1.18 [0.16, 2.21], n = 1,085), not 152.0 vs 146.3 (n = 74); the real-minus-sim total gap still rises from -0.76 (Q1) to +1.18 (Q5) and from -0.20 to +1.13 across |spread| quintiles. In regulation points that rise splits into possessions (-2.30 to -0.86 per game) and points per possession (+0.51 to +1.06): about 3/4 possessions. The rating is not compressed: realised margin on our rating has slope 0.723 real vs 0.716 sim (ratio 0.992, diff [-0.022, 0.031], n = 4,724) and the rating-implied PPP response is 0.97 of real (eFG 0.93, 2P% 0.92, FTA rate 0.80, TOV 0.99, OREB 1.20; those with CIs excluding 0 are worth a few tenths of a point). The mapping that is compressed is possessions: game possessions rise +0.152 per env unit and +0.050 per mismatch unit in the data but +0.011 / -0.008 in the sim, team possessions per unit of offensive strength have a sim/real slope ratio of 0.06 and seconds per possession of 0.28 (CIs of the difference exclude 0), and the game total's slope on env is 0.408 sim vs 0.600 real (ratio 0.67, diff [0.146, 0.251], n = 5,424). First-half seconds per possession already fall 18.09 -> 17.80 across home-rating quintiles in reality and 17.84 -> 17.81 in the sim, so it is not an end-of-game artefact. The lead-change slope against |spread| is -0.118 real vs -0.093 sim (ratio 0.79, diff [-0.0385, -0.0084], n = 5,106, powered; earlier -0.188 vs -0.107 was a 371-game estimate), time of decision 0.76, OT share 0.53; the sim also decides close-spread games earlier (Q1: 1,272 vs 1,417 s) and plays per-game margins 10-18% more dispersed than the real residual around the closing line (variance budget above): a second, untested candidate (game-level variance structure) for the path-shape part. Pre-registered owner for the possession part: clock (draft `docs/models/clock/experiments_DRAFT_scoring_env_pace_2026-10-09.md`). Fold 1 not run.
