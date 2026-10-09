# D4 overtime rate and the endgame (fold 2 / 2024-25), 2026-10-09

DIAGNOSTIC ONLY. Nothing changed. Same script and data as D3.

Sim: 5500 games x 50 seeds. Real: 5500 games (pbp path = verified final); the season-wide real OT rate over all 5710 fold-2 games in the extract is 5.57% (verified finals, period count > 2). Seed SE = SD over seeds / sqrt(S). CIs: game-cluster bootstrap of real - sim (300 reps).

Grading-truth check: real regulation-end margin == 0 (from the pbp scoreboard) agrees with verified-final `n_periods > 2` in 100.00% of the 5500 games.

## (a) Regulation-end margin distribution (|margin| at the end of 40:00)

| |margin| at end of regulation | real share (n=5500) | sim share | real - sim [95%] | sim seed SE | flag |
|---|---:|---:|---|---:|---|
| 0 (tie -> OT) | 5.62% | 3.54% | 2.08 pp [1.49, 2.71] | 0.038 pp | excludes 0 |
| 1 | 3.64% | 5.72% | -2.08 pp [-2.54, -1.61] | 0.043 pp | excludes 0 |
| 2 | 5.15% | 5.35% | -0.20 pp [-0.80, 0.39] | 0.037 pp |  |
| 3 | 5.40% | 4.78% | 0.62 pp [0.05, 1.29] | 0.037 pp | excludes 0 |
| 4-6 | 15.36% | 12.38% | 2.98 pp [2.05, 3.86] | 0.061 pp | excludes 0 |
| 7-9 | 14.25% | 12.74% | 1.52 pp [0.63, 2.41] | 0.057 pp | excludes 0 |
| 10+ | 50.58% | 55.49% | -4.91 pp [-6.24, -3.74] | 0.086 pp | excludes 0 |
| <= 3 (cum.) | 19.80% | 19.39% | 0.41 pp [-0.50, 1.56] | 0.076 pp |  |
| <= 6 (cum.) | 35.16% | 31.77% | 3.39 pp [2.36, 4.67] | 0.083 pp | excludes 0 |
| <= 1 (cum.) | 9.25% | 9.26% | -0.00 pp [-0.70, 0.76] | 0.050 pp |  |

### OT = sum over the margin state at 2:00 of P(state) x P(tie | state)

State = |margin| before the first H2 possession starting with <= 120 s left (home perspective sign dropped). Real games in cell shown; `U` = underpowered.

| |margin| at 2:00 | real games | P(state) real / sim | P(tie | state) real / sim | contribution to OT rate (pp) real / sim | diff in contribution [95%] |
|---|---:|---|---|---|---|
| 0 | 163 | 0.030 / 0.028 (0.001) | 0.252 / 0.167 (0.085*)  | 0.75 / 0.48 | 0.27 [0.04, 0.50] |
| 1-2 | 670 | 0.122 / 0.109 (0.012*) | 0.190 / 0.136 (0.054*)  | 2.31 / 1.49 | 0.82 [0.45, 1.24] |
| 3-5 | 950 | 0.173 / 0.156 (0.016*) | 0.113 / 0.078 (0.034*)  | 1.95 / 1.22 | 0.72 [0.38, 1.09] |
| 6-9 | 1001 | 0.182 / 0.184 (-0.002) | 0.033 / 0.019 (0.014*)  | 0.60 / 0.34 | 0.26 [0.07, 0.48] |
| 10+ | 2716 | 0.494 / 0.522 (-0.028*) | 0.000 / 0.000 (0.000)  | 0.02 / 0.01 | 0.00 [-0.02, 0.04] |

Totals: real 5.62% , sim 3.54% (tie at end of regulation).

## (b) Endgame behaviour in the last 2 minutes of H2, by the CURRENT trailing margin at the start of each possession

Rows are possessions starting with <= 120 s left in H2. `trailer` = team with the ball is behind (margin before the possession 1-3, 4-6, 7+); `leader` = team with the ball is ahead by that margin. Cells show `real / sim (diff)`; `*` = CI excludes 0; `U` = underpowered (< 60 real possessions).

| role | |margin| | real poss | sim poss/seed | sec/poss | 3PA share (pp) | TOV rate (pp) | FT-trip share (pp) | FTA/poss | PPP |
|---|---|---:|---:|---|---|---|---|---|---|
| leader | 1-3 | 4307 | 3599 | 13.70 / 13.30 (0.40*) | 9.7 / 12.4 (-2.7*) | 14.1 / 14.6 (-0.5) | 54.2 / 48.4 (5.8*) | 1.040 / 0.924 (0.116*) | 1.210 / 1.120 (0.090*) |
| leader | 4-6 | 3954 | 3751 | 10.33 / 10.57 (-0.24) | 5.9 / 8.7 (-2.7*) | 13.5 / 13.2 (0.4) | 65.8 / 60.1 (5.7*) | 1.269 / 1.170 (0.099*) | 1.257 / 1.237 (0.020) |
| leader | 7+ | 11393 | 16523 | 14.30 / 11.82 (2.48*) | 16.8 / 15.4 (1.5*) | 18.8 / 22.2 (-3.4*) | 39.3 / 38.4 (0.9) | 0.747 / 0.742 (0.006) | 1.135 / 1.123 (0.013) |
| tied | tied | 1206 | 1244 | 18.16 / 14.75 (3.41*) | 23.4 / 22.7 (0.7) | 14.3 / 16.4 (-2.1) | 19.7 / 23.1 (-3.4*) | 0.375 / 0.431 (-0.056*) | 0.988 / 1.086 (-0.098*) |
| trailer | 1-3 | 3977 | 3652 | 13.71 / 14.74 (-1.03*) | 30.1 / 27.0 (3.1*) | 13.8 / 14.5 (-0.7) | 23.7 / 24.1 (-0.4) | 0.449 / 0.453 (-0.003) | 1.035 / 1.129 (-0.094*) |
| trailer | 4-6 | 4433 | 3735 | 10.62 / 11.04 (-0.42*) | 38.5 / 35.7 (2.9*) | 10.2 / 13.0 (-2.8*) | 21.8 / 21.6 (0.2) | 0.412 / 0.403 (0.010) | 1.141 / 1.135 (0.006) |
| trailer | 7+ | 14535 | 18121 | 11.46 / 11.08 (0.39*) | 36.3 / 36.0 (0.3) | 11.1 / 11.9 (-0.8*) | 21.3 / 20.7 (0.6) | 0.398 / 0.383 (0.015*) | 1.149 / 1.104 (0.045*) |

Foul STATE at the start of each possession (last 2 min of H2): share of possessions where the team with the ball is in the bonus, and the team-foul counts of the defence and the offence. This is the state the foul-accrual law feeds into possession_outcome.

| role | |margin| | real poss | in bonus (pp) real / sim (diff) | defence team fouls | offence team fouls |
|---|---|---:|---|---|---|
| leader | 1-3 | 4307 | 94.1 / 88.6 (5.5*) | 9.39 / 9.09 (0.30*) | 8.85 / 8.92 (-0.07) |
| leader | 4-6 | 3954 | 95.8 / 89.6 (6.2*) | 10.09 / 9.54 (0.56*) | 8.99 / 8.88 (0.10) |
| leader | 7+ | 11393 | 92.3 / 89.1 (3.3*) | 9.54 / 9.38 (0.16*) | 8.85 / 8.67 (0.18*) |
| tied | tied | 1206 | 92.4 / 88.1 (4.3*) | 8.84 / 8.87 (-0.04) | 8.83 / 8.95 (-0.12) |
| trailer | 1-3 | 3977 | 91.7 / 88.0 (3.7*) | 8.89 / 8.79 (0.10) | 8.87 / 9.16 (-0.29*) |
| trailer | 4-6 | 4433 | 92.5 / 88.1 (4.4*) | 8.96 / 8.77 (0.18*) | 9.53 / 9.67 (-0.14*) |
| trailer | 7+ | 14535 | 91.4 / 86.5 (4.9*) | 8.94 / 8.59 (0.35*) | 9.51 / 9.68 (-0.18*) |

The trailing team's FOUL rate: fouls by the trailing team per leader possession. Sim = team fouls the defender (trailing team) adds during the leader's possession (engine counters). Real = PersonalFoul events by the trailing team in the raw pbp per leader possession in the possession table (includes offensive fouls and fouls with no free throws, so the real definition is the broader one; treat as an upper bound on the comparison). Leader possessions here start in the last 2 minutes of H2.

| trailing by | leader poss real | sim/seed | trailing-team fouls per leader poss: real (pbp events) | sim (engine counters) | diff [95%] |
|---|---:|---:|---|---|---|
| 1-3 | 4307 | 3599 | 0.647 | 0.480 | 0.166 [0.148, 0.185] |
| 4-6 | 3954 | 3751 | 0.746 | 0.610 | 0.136 [0.119, 0.154] |
| 7+ | 11393 | 16523 | 0.462 | 0.399 | 0.064 [0.051, 0.075] |

## (c) Final-possession outcomes when the team with the ball is down 1-3 with < 30 s left in H2

Class rule: TOV > FT trip (the offence was fouled and shot free throws) > 3PA > 2PA > NONE (no shot, no TOV). First possession of a team in the window only is NOT separated; all possessions starting with <= 30 s are pooled (this includes second chances after a made basket by the other team). Shares of possessions; `*` = CI excludes 0.

**down 1-3**: real possessions 1589, sim possessions per seed 1224.1.

| outcome | real share | sim share | diff (pp) [95%] |
|---|---:|---:|---|
| 3PA | 36.1% | 31.1% | 5.0 [2.6, 7.6] |
| 2PA | 26.0% | 29.7% | -3.7 [-5.8, -1.4] |
| FT | 24.3% | 24.6% | -0.3 [-2.4, 1.8] |
| TOV | 13.3% | 14.6% | -1.3 [-2.9, 0.6] |
| NONE | 0.3% | 0.0% | 0.3 [0.1, 0.6] |

**down 1**: real possessions 415, sim possessions per seed 413.3.

| outcome | real share | sim share | diff (pp) [95%] |
|---|---:|---:|---|
| 3PA | 22.9% | 23.6% | -0.7 [-4.9, 3.2] |
| 2PA | 38.8% | 36.0% | 2.8 [-1.8, 7.3] |
| FT | 22.4% | 23.9% | -1.5 [-5.2, 2.9] |
| TOV | 15.2% | 16.5% | -1.3 [-4.4, 2.5] |
| NONE | 0.7% | 0.0% | 0.7 [0.0, 1.7] |

**down 2-3**: real possessions 1174, sim possessions per seed 810.9.

| outcome | real share | sim share | diff (pp) [95%] |
|---|---:|---:|---|
| 3PA | 40.7% | 34.9% | 5.8 [2.9, 8.8] |
| 2PA | 21.5% | 26.4% | -5.0 [-7.2, -2.6] |
| FT | 25.0% | 25.0% | -0.1 [-2.6, 2.6] |
| TOV | 12.7% | 13.6% | -1.0 [-2.8, 1.1] |
| NONE | 0.2% | 0.0% | 0.2 [0.0, 0.4] |

Points per such possession: real 0.893, sim 1.104, diff -0.211 [-0.266, -0.153]. Seconds per such possession: real 8.22, sim 9.00, diff -0.78 [-1.13, -0.46].

## (d) Verdict

**The missing overtimes are a conversion defect inside the last two minutes (the leader is fouled too little and the trailer scores too well), owned jointly by foul accrual (the bonus state) and the late-game final-possession behaviour; it is neither variance nor arrival, and the near-zero band as a whole is right.** (a) Over 5,500 games the sim ties at the end of regulation 3.54% of the time against 5.62% real (2.08 pp [1.49, 2.71], seed SE 0.04 pp; season-wide verified OT rate 5.57%); the sim's one-point margins are 5.72% against 3.64%, a mirror-image 2.08 pp. The cumulative cells are equal: |margin| <= 1 is 9.26% sim vs 9.25% real and <= 3 is 19.39% vs 19.80% (diff 0.41 pp [-0.50, 1.56]), so only exact ties are under (it is a tie-vs-one-point redistribution); the sim also has too many 10+ blowouts (55.5% vs 50.6%) and too few 4-9 margins. Real regulation-end margin 0 agrees with the verified-final period count in 100% of games. (b) By state at 2:00 the arrival is close (P(|m| 0 / 1-2 / 3-5 / 6-9) real 0.030 / 0.122 / 0.173 / 0.182 vs sim 0.028 / 0.109 / 0.156 / 0.184), while P(tie | state) is 0.55-0.75 of real in every state (0.252 vs 0.167, 0.190 vs 0.136, 0.113 vs 0.078, 0.033 vs 0.019); the OT-rate gap decomposes into +0.27, +0.82, +0.72, +0.26 pp from those four states: conversion, not arrival. Behaviour: the leader's free-throw trips per possession in the last two minutes are 1.040 vs 0.924 (leading by 1-3) and 1.269 vs 1.170 (by 4-6), its PPP 1.210 vs 1.120; the trailing team's PPP when trailing 1-3 is 1.035 real vs 1.129 sim; the sim's team with the ball is in the bonus 88.6-89.6% of the time when leading by 1-6 against 94.1-95.8% real (+5.5 to +6.2 pp, defence team fouls 9.09 vs 9.39 and 9.54 vs 10.09); trailing teams commit 0.65 / 0.75 / 0.46 fouls per leader possession (trailing by 1-3 / 4-6 / 7+) in the pbp against 0.48 / 0.61 / 0.40 in the engine (the real definition is broader, so read as an upper bound on the gap). (c) On 1,589 real vs 1,224 per-seed sim possessions with the team down 1-3 and under 30 s, three-point attempts are 36.1% vs 31.1% (+5.0 pp [2.6, 7.6]) and two-point attempts 26.0% vs 29.7% (-3.7 pp [-5.8, -1.4]) with TOV and FT-trip shares within 1.3 pp; points per such possession are 0.893 real vs 1.104 sim (-0.211 [-0.266, -0.153]) and seconds per possession 8.22 vs 9.00, i.e. the sim's last possessions score too well, the horn-possession finding of late-game rounds 3-6. Verdict: not variance; owners are the foul accrual state (possession_outcome A2; the pre-registered but never run late-foul-accrual round, possession_outcome/experiments.md section 29, targets exactly the trailing-defence-below-bonus gap measured here) and the late-game final-possession law (rounds 3-6 arms were run and not adopted; their vetoes fail on G1/G9, not on this evidence). Draft: `docs/models/late_game/experiments_DRAFT_endgame_margin_clock_foul_2026-10-09.md`.
