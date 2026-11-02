# Opening-day totals, 2026-11-02: why the sim sits about 11 points under the market

This is a diagnostic round. It only reports: the model is frozen, and nothing was changed, fitted, adjusted or adopted.
- Script: `scripts/diag_opening_day_totals_v1.py`.
- Outputs: `results/diag/opening_day_totals_2026-11-02/` (gitignored).
- 2025-26 (season 2026) is SEALED. No 2025-26 finals, box or lines were read. The only 2025-26 quantity used is the one the serving path already carries: `league_off_mean` / `league_tempo_mean` in `own_ratings_2027.parquet`. Q2 therefore uses 2022-23, 2023-24 and 2024-25. Reading the 2025-26 opening day (2025-11-03) would be the sharpest test, and it needs a user seal ruling.

## Answer

**Most of the -11 is a real opening-day pace effect that the engine lacks, about -7.6 points.**
- In all three seasons, real opening-day games run about +3.1 possessions per team faster than the rest of the season: 71.0 vs 67.8 possessions per team, by season +3.5 / +2.7 / +3.1.
- The engine's own past opening days were also -3.2 / -3.3 possessions short. On those days it was -11.1 (2023-24) and -12.3 (2024-25) points under the actual finals.
- **Mechanism.** The clock's pregame tempo prior is `tempo_prior_game = tempo_rel_A x tempo_rel_B x as-of league tempo mean`. On day 0 the as-of league mean is the prior season's FINAL value (68.4 for 2024-25 d0; 68.7 today), which is a late-season level. From day 1 the as-of mean jumps to about 70.8 and the engine's pace recovers: 69.7-70.4 on d1-3.

**The remaining about -3.3 points is per-possession efficiency.** At the real opening-day pace, the market implies 1.064 points per possession (PPP); the sim gives 1.040. That residual splits into three parts:
- **The known early-season efficiency bias.** On past opening days the engine's PPP was 0.033-0.041 low, owned by FT make, FTA rate, jump2 make and TOV, as already documented.
- **The 2027 build's level lag, about -0.8 points (estimate).** The sub-models are the 2025-04-01 refit and the event model's `season_idx` saturates. They miss the 2025-26 level rise of +0.55% in the ratings' league offence.
- **The market being high: about 0, plausible range up to -2.6.** History points the other way: on opening day the market sat 2.9 points (pooled) and 4.0 / 5.0 points (last two seasons) BELOW the actual totals.

Today's -11 is therefore not new to the 2027 build:
- On past opening days the frozen engine was -11 to -12 vs the finals, but only -7.0 / -7.1 vs the market, because the market itself was low.
- Today the engine is +2.6 points above its own 2024-25 opening day, mostly from FT make now that A3 priors exist. The market is +6.5 points above its 2024-25 opening-day level.
- A history-based projection of today's real total lands on the market: real opening-day pace x 2024-25 opening-day PPP x the ratings' league drift gives 150.8, against a market of 150.9.

**Single largest contributor: opening-day pace, -7.6 points.**
- Confidence in the sign and the order of magnitude is HIGH: three seasons of truth, two seasons of engine replay, and the mechanism found in code.
- Confidence in the size is MEDIUM: ±1.5 points, from the season-to-season spread of opening-day pace.

## Q1. Today's gap, pace vs efficiency (117 games with a market total; DraftKings)

**Definitions.**
- P is box possessions per team: FGA - OREB + TOV + 0.44 FTA, from the per-seed sim output `results/daily/sim/2026-11-02/s2500_o0/games.parquet`.
- The engine's own `possessions` counter runs 0.9% above box P: 67.86 vs 67.24 today.
- Sim PPP = sim total / (2 P).

**Implied quantities.** Market PPP at the sim's P is total / (2 P_sim). Market P at the sim's PPP is total / (2 PPP_sim).

**The split.** A market total alone does not split into pace and efficiency, so the split uses a reference pace: real opening-day P, pooled over the 3 seasons and matched by matchup class (Q2). The pace part is 2 (P_sim - P_ref) PPP_sim. The efficiency part is the rest.

| quantity (per game) | mean | 10% | median | 90% |
|---|---|---|---|---|
| sim total - market | -10.97 | -17.31 | -10.63 | -5.69 |
| sim PPP | 1.040 | 1.020 | 1.037 | 1.062 |
| market PPP at sim P | 1.122 | 1.073 | 1.123 | 1.171 |
| sim P (box) | 67.24 | 64.66 | 67.85 | 70.72 |
| market P at sim PPP | 72.51 | 69.13 | 72.44 | 75.91 |

Overall: of the -10.97 points, **pace accounts for -7.65 (P 67.24 vs a reference of 70.92) and efficiency for -3.32**. 99% of games are under the line.

| dim | bucket | n | sim - mkt | sim total | market | sim P | sim PPP | mkt PPP @ sim P | mkt P @ sim PPP | ref d0 P | pace part | eff part | mean abs spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | all | 117 | -10.97 | 139.9 | 150.9 | 67.24 | 1.040 | 1.122 | 72.51 | 70.92 | -7.65 | -3.32 | 16.8 |
| home rating quintile (own ratings 2027 net) | Q1 low | 24 | -11.09 | 139.8 | 150.9 | 67.40 | 1.037 | 1.119 | 72.75 | 70.74 | -6.90 | -4.19 | 5.2 |
| | Q2 | 23 | -10.75 | 140.7 | 151.4 | 67.72 | 1.038 | 1.118 | 72.88 | 71.04 | -6.87 | -3.87 | 8.7 |
| | Q3 | 23 | -10.30 | 139.6 | 149.9 | 67.33 | 1.037 | 1.113 | 72.30 | 70.94 | -7.47 | -2.82 | 21.8 |
| | Q4 | 23 | -9.62 | 139.7 | 149.3 | 67.12 | 1.040 | 1.112 | 71.76 | 71.11 | -8.26 | -1.37 | 24.0 |
| | Q5 high | 24 | -12.97 | 139.9 | 152.8 | 66.65 | 1.049 | 1.147 | 72.85 | 70.80 | -8.71 | -4.26 | 24.4 |
| abs spread quintile | S1 (2.8) | 29 | -11.18 | 141.7 | 152.8 | 68.17 | 1.039 | 1.121 | 73.55 | 70.47 | -4.76 | -6.41 | 2.8 |
| | S2 | 20 | -11.79 | 137.5 | 149.2 | 66.49 | 1.034 | 1.123 | 72.20 | 71.09 | -9.49 | -2.30 | 9.2 |
| | S3 | 21 | -10.94 | 139.8 | 150.8 | 67.49 | 1.036 | 1.117 | 72.78 | 71.14 | -7.54 | -3.40 | 17.1 |
| | S4 | 24 | -10.14 | 140.6 | 150.8 | 67.35 | 1.044 | 1.120 | 72.22 | 71.08 | -7.75 | -2.39 | 24.6 |
| | S5 (32.6) | 23 | -10.86 | 139.1 | 150.0 | 66.37 | 1.048 | 1.131 | 71.54 | 70.97 | -9.67 | -1.19 | 32.6 |
| site | home-court | 104 | -10.78 | 139.8 | 150.6 | 67.11 | 1.042 | 1.122 | 72.29 | 71.01 | -8.11 | -2.68 | 18.0 |
| | neutral (UNDERPOWERED) | 13 | -12.43 | 140.7 | 153.1 | 68.29 | 1.030 | 1.121 | 74.32 | 70.24 | -3.98 | -8.45 | 6.9 |
| matchup (conference tier) | low-low | 29 | -10.00 | 141.6 | 151.6 | 68.06 | 1.040 | 1.114 | 72.86 | 70.37 | -4.79 | -5.21 | 4.9 |
| | mid-low | 29 | -11.05 | 139.2 | 150.3 | 67.05 | 1.038 | 1.121 | 72.38 | 71.47 | -9.16 | -1.89 | 13.8 |
| | power-low | 48 | -10.85 | 139.7 | 150.6 | 66.99 | 1.043 | 1.124 | 72.19 | 70.91 | -8.17 | -2.67 | 27.1 |
| | power-mid / mid-mid / power-power (UNDERPOWERED, n 5/3/3; history refs n 9/3/2) | 11 | -13.7 / -12.5 / -15.3 | | | | | | | | | | |

**Reading.**
- The gap is a slate-wide level shift.
  - It is flat across home-rating quintiles (-9.6 to -13.0; each cell SE about 1.0, so Q5 vs Q4 is about 2.4 SE, one cell of 20, underpowered) and across spread quintiles (-10.1 to -11.8).
  - It is flat across site and matchup classes.
- Sim totals are matchup-responsive:
  - Across games, the correlation with the market total is 0.70.
  - The SD across games is 5.4 for the sim (117 lined games) vs 6.2 for the market.
- The pace/efficiency split varies by bucket mainly through the matchup-specific reference pace. The neutral and small-matchup cells are underpowered.

Tiers: power = ACC, Big Ten, Big 12, SEC, Big East; mid = MWC, WCC, A-10, American, Pac-12; low = the rest. Conferences come from `data/raw/preseason/2027_v2_20261008/games_2027.parquet`.

## Q2. History by day of season (truth = verified finals + hoopR box; market = CBBD close, ESPN BET, else the provider median)

d0 is the first date with at least 50 D-I games: 2022-11-07, 2023-11-06, 2024-11-04. The 2025-26 season is sealed and was not read.

| season | bucket | n | actual total | actual P | actual PPP | n lined | market total | actual - market (SE) | abs spread |
|---|---|---|---|---|---|---|---|---|---|
| 2022-23 | d0 | 123 | 141.76 | 71.16 | 0.996 | 123 | 141.66 | +0.10 (1.52) | 16.0 |
| | d1-3 | 51 | 144.00 | 70.93 | 1.015 | 51 | 141.87 | +2.13 (2.68) | 14.5 |
| | d4-7 | 184 | 142.02 | 69.58 | 1.021 | 184 | 141.33 | +0.69 (1.21) | 11.7 |
| | d8-14 | 357 | 141.01 | 69.01 | 1.022 | 357 | 141.93 | -0.92 (0.92) | 8.9 |
| | d15-30 | 739 | 139.38 | 67.90 | 1.026 | 739 | 139.11 | +0.26 (0.59) | 8.4 |
| | d31+ | 4169 | 141.70 | 67.59 | 1.048 | 4169 | 140.21 | +1.49 (0.25) | 6.8 |
| 2023-24 | d0 | 112 | 145.41 | 70.94 | 1.025 | 93 | 141.21 | **+4.03 (1.66)** | 14.9 |
| | d1-3 | 56 | 147.16 | 71.88 | 1.024 | 48 | 144.06 | +4.08 (2.35) | 13.0 |
| | d4-7 | 172 | 146.12 | 69.81 | 1.047 | 161 | 144.74 | +1.48 (1.19) | 11.9 |
| | d8-14 | 356 | 143.34 | 68.99 | 1.039 | 327 | 143.90 | -0.01 (0.94) | 9.9 |
| | d15-30 | 691 | 145.79 | 69.11 | 1.055 | 637 | 143.89 | +1.57 (0.64) | 8.8 |
| | d31+ | 4245 | 145.27 | 68.05 | 1.067 | 3957 | 144.65 | +0.64 (0.26) | 7.2 |
| 2024-25 | d0 | 109 | 149.56 | 70.75 | 1.057 | 108 | 144.41 | **+5.00 (1.47)** | 15.2 |
| | d1-3 | 54 | 150.46 | 70.70 | 1.064 | 54 | 146.83 | +3.63 (1.96) | 17.1 |
| | d4-7 | 164 | 147.20 | 70.08 | 1.050 | 164 | 148.86 | -1.66 (1.28) | 13.3 |
| | d8-14 | 291 | 146.85 | 69.36 | 1.059 | 291 | 147.64 | -0.79 (1.01) | 11.0 |
| | d15-30 | 743 | 144.85 | 68.18 | 1.062 | 743 | 145.15 | -0.30 (0.59) | 9.8 |
| | d31+ | 4339 | 145.46 | 67.53 | 1.077 | 4014 | 144.75 | +0.75 (0.26) | 7.3 |
| pooled | d0 | 344 | 145.42 | 70.96 | 1.025 | 324 | 142.45 | **+2.86 (0.90)** | 15.4 |
| | d1-3 | 161 | 147.27 | 71.18 | 1.034 | 153 | 144.31 | +3.27 (1.34) | 14.9 |
| | d4-7 | 520 | 145.01 | 69.81 | 1.039 | 509 | 144.83 | +0.18 (0.71) | 12.3 |
| | d8-14 | 1004 | 143.53 | 69.11 | 1.038 | 975 | 144.29 | -0.58 (0.55) | 9.9 |
| | d15-30 | 2173 | 143.29 | 68.38 | 1.048 | 2119 | 142.66 | +0.46 (0.35) | 9.0 |
| | d31+ | 12753 | 144.17 | 67.73 | 1.064 | 12140 | 143.16 | +0.97 (0.15) | 7.1 |

Opening day vs the rest of the season (d15+), with a midpoint split: total = 2 P PPP.

| season | d0 total | d15+ total | difference | pace part | PPP part | d0 P / rest P | d0 PPP / rest PPP | d0 actual - market | rest actual - market |
|---|---|---|---|---|---|---|---|---|---|
| 2022-23 | 141.76 | 141.35 | +0.41 | +7.19 | -6.78 | 71.16 / 67.64 | 0.996 / 1.045 | +0.10 | +1.31 |
| 2023-24 | 145.41 | 145.35 | +0.07 | +5.72 | -5.66 | 70.94 / 68.20 | 1.025 / 1.066 | +4.03 | +0.77 |
| 2024-25 | 149.56 | 145.37 | +4.19 | +6.65 | -2.46 | 70.75 / 67.63 | 1.057 / 1.075 | +5.00 | +0.59 |
| pooled | 145.42 | 144.04 | +1.38 | +6.55 | -5.17 | 70.96 / 67.82 | 1.025 / 1.062 | +2.86 | +0.89 |

**Reading.**
1. Real opening-day totals are about the season mean: +1.4 pooled, +4.2 in 2024-25. Underneath, two large opposite effects nearly cancel: pace +6.5 and PPP -5.2.
2. The pace effect is not a mismatch artefact:
   - d0 P is 70.4-71.5 in every matchup class with n >= 83, vs 67.7-68.9 for the same classes later in the season.
   - d0 P is 70.4-71.3 in every abs-spread bin (`q2_d0_by_abs_spread.csv`).
3. The market does NOT run high on opening day. It runs low:
   - Pooled, the market is 2.9 points (SE 0.9) under actual on d0.
   - It is +4.0 and +5.0 in the last two seasons, each about 2.5-3.5 SE.
   - It is 3.3 points under on d1-3.
4. The real opening-day PPP rose by about 0.030 per season (0.996 / 1.025 / 1.057), faster than the season level (1.045 / 1.066 / 1.075).

## Q3. What the frozen engine did on past opening days

Sources:
- Served v3 (clock K2) closed-loop reads at 200 seeds: `results/engine_v0/f1c_K2_full_s200_o0` (2023-24, F1) and `results/engine_v0/laneH_v3full_K2_s200_o0` (2024-25, F2, the K2 early-total doc's sample).
- **Replay through the current daily path**, run for this round:
  - Command: `run_daily_sim_v1.run_sim_stage("2024-11-04", 2025, "F2", 200 seeds, replay=True, jobs=8)`, via `diag_opening_day_totals_v1.py --run-replay`.
  - Output: `results/diag/opening_day_totals_2026-11-02/replay_2024-11-04/`.
  - It is bit-identical to the closed-loop read on all 111 games (max abs difference 0.0 in total and P).
  - The old 3-seed day-1 rehearsal from 2026-09-30 (P 69.1) predates the current stack and is superseded.
- 2025-26 is sealed: no engine read exists for it, and none was made.

| season | bucket | n | sim total | actual | sim - actual (SE) | market | sim - market (SE) | sim P / actual P | sim PPP / actual PPP | pace part | PPP part |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023-24 | d0 | 112 | 134.27 | 145.41 | **-11.14 (1.63)** | 141.21 | **-6.99 (0.53)** | 67.65 / 70.94 | 0.992 / 1.025 | -6.63 | -4.51 |
| | d1-3 | 56 | 139.55 | 147.16 | -7.62 (2.42) | 144.06 | -4.38 (0.79) | 70.41 / 71.88 | 0.991 / 1.024 | -2.97 | -4.65 |
| | d4-7 | 172 | 140.58 | 146.12 | -5.54 (1.20) | 144.74 | -4.13 (0.47) | 70.19 / 69.81 | 1.001 / 1.047 | +0.77 | -6.31 |
| | d8-14 | 356 | 140.67 | 143.34 | -2.67 (0.93) | 143.90 | -3.24 (0.32) | 69.77 / 68.99 | 1.008 / 1.039 | +1.58 | -4.25 |
| | d15-30 | 691 | 142.81 | 145.79 | -2.98 (0.65) | 143.89 | -1.06 (0.23) | 69.28 / 69.11 | 1.031 / 1.055 | +0.35 | -3.33 |
| | d31+ | 4245 | 144.66 | 145.27 | -0.61 (0.27) | 144.65 | +0.00 (0.08) | 68.31 / 68.05 | 1.059 / 1.067 | +0.55 | -1.16 |
| 2024-25 | d0 | 109 | 137.30 | 149.56 | **-12.26 (1.59)** | 144.41 | **-7.13 (0.51)** | 67.56 / 70.75 | 1.016 / 1.057 | -6.62 | -5.65 |
| | d1-3 | 54 | 142.14 | 150.46 | -8.32 (2.05) | 146.83 | -4.69 (0.73) | 69.73 / 70.70 | 1.019 / 1.064 | -2.02 | -6.30 |
| | d4-7 | 164 | 143.78 | 147.20 | -3.43 (1.36) | 148.86 | -5.08 (0.52) | 69.75 / 70.08 | 1.031 / 1.050 | -0.69 | -2.74 |
| | d8-14 | 291 | 144.29 | 146.85 | -2.56 (1.08) | 147.64 | -3.35 (0.35) | 69.52 / 69.36 | 1.038 / 1.059 | +0.33 | -2.89 |
| | d15-30 | 743 | 144.93 | 144.85 | +0.08 (0.63) | 145.15 | -0.22 (0.20) | 68.91 / 68.18 | 1.052 / 1.062 | +1.55 | -1.46 |
| | d31+ | 4339 | 146.00 | 145.46 | +0.54 (0.26) | 144.75 | +1.28 (0.08) | 67.80 / 67.53 | 1.077 / 1.077 | +0.57 | -0.04 |

Shapley channels, engine - actual, points per game. Games with event shot classes only: 2024-25 d0 has 71 of 109.

| sample | n | gap | pace | TOV | OREB | FTA rate | 3PA share | rim share | rim make | jump2 make | 3 make | FT make |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023-24 d0 | 112 | -11.14 | **-6.66** | -1.81 | -1.01 | -0.89 | -0.01 | -1.08 | +0.40 | +0.55 | +0.23 | -1.04 |
| 2024-25 d0 | 71 | -11.43 | **-5.91** | -0.52 | -0.30 | -0.89 | +0.10 | +0.44 | -0.94 | -1.50 | +0.14 | -2.07 |
| 2023-24 d1-3 | 56 | -7.62 | -3.04 | -0.72 | -0.07 | -1.39 | -0.08 | -0.81 | +1.30 | -1.53 | +0.03 | -1.42 |
| 2024-25 d1-3 | 54 | -8.32 | -2.00 | -1.73 | +0.67 | -0.13 | +0.02 | -0.06 | -1.25 | +0.24 | -2.16 | -1.93 |
| 2023-24 d0-14 | 657 | -5.22 | -0.58 | -2.18 | -0.65 | -0.91 | -0.02 | -0.56 | +0.69 | -0.44 | +0.31 | -1.07 |
| 2024-25 d0-14 | 572 | -4.46 | -0.91 | -0.57 | -0.23 | -0.70 | +0.06 | -0.36 | -0.07 | -0.62 | +0.28 | -1.36 |

**Reading.**
- The frozen engine has always been 11-12 points under the finals on opening day.
  - About 6.6 points of that is pace, and only on d0. On d0 the engine's P is 67.6, and from d1 it is 69.7-70.4.
  - The rest is the documented early-season PPP owners (FT make, FTA rate, jump2 / rim make, TOV).
  - The d0-14 block hides this: there pace is only -0.6 / -0.9, because d4-14 sim pace runs slightly fast.
- Against the market it was only -7.0 / -7.1 on d0, because the market was itself 4-5 points low.
- The served v2 clock gives the same d0 pace (67.97 F2, 67.84 F1). K2 takes off a further 0.2-0.4 possessions. The defect predates K2.

## Q4. 2027-build checks

**(a) Ratings level.** Own ratings are league-centred:
- `off_c` and `def_c` have mean 0.000 and `tempo_rel` has mean 1.000 over the 365 teams in `own_ratings_2027.parquet`.
- Day-1 teams carry the 2025-26 final fit at weight 0.8 (`prior_carry`). New teams get the league mean.
- `league_off_mean` = 107.006 and `league_tempo_mean` = 68.718. Both are the 2025-26 FINAL values.
- The engine's efficiency level does not come from `league_off_mean`. It comes from sub-model intercepts, the 2025-04-01 refit trained through 2025-03; the event model's `season_idx` saturates at its trained range.
- Tempo DOES enter as an absolute level: `tempo_prior_game` multiplies by the as-of league tempo mean.

| season | truth PPP x100, season / Nov / d0 | truth P, season / Nov | ratings league off, d0 / final | ratings league tempo, d0 (carried) / d1 / final |
|---|---|---|---|---|
| 2022-23 | 104.12 / 101.89 / 99.60 | 67.90 / 68.99 | 101.75 / 102.84 | 68.11 / 70.89 / 67.93 |
| 2023-24 | 106.20 / 104.54 / 102.49 | 68.39 / 69.57 | 102.86 / 104.59 | 67.94 / 70.76 / 68.40 |
| 2024-25 | 107.27 / 105.79 / 105.70 | 67.88 / 69.07 | 104.59 / 106.42 | 68.40 / 70.76 / 67.88 |
| 2026-27 day 1 | sealed / - / - | - | 107.01 (= 2025-26 final) | 68.72 (= 2025-26 final) |

The d1 column is the as-of league tempo the morning after opening day (`data/processed/ratings/own_ratings_{season}.parquet`). On d0 the clock is fed a late-season tempo level; from d1 it gets an early-season one, about 2.4-2.8 higher.

On the ratings' scale, the 2025-26 league offence is +0.55% over 2024-25. The truth level runs 0.9-1.6 per 100 above the ratings' final in each past season. That puts the 2025-26 truth PPP at about 108. The 2027 sub-models never saw this level. Estimated cost: about -0.8 points per game. The estimate is indirect and cannot be verified while 2025-26 is sealed.

**(b) Channel rates today vs real opening days and vs the engine's own past opening days.**

Shapley, today's slate minus the reference, points per game. Today's slate runs 117 games' team rows against each reference.

| reference | n ref | gap | pace | TOV | OREB | FTA rate | 3PA share | rim share | rim make | jump2 make | 3 make | FT make |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real d0 2024-25 | 71 | -8.39 | **-6.20** | -0.14 | +0.15 | -0.91 | -0.02 | +0.24 | -0.44 | -1.09 | +0.43 | -0.42 |
| real d0 2023-24 | 112 | -5.56 | **-7.71** | +0.32 | +0.09 | -0.48 | -0.08 | -0.36 | +0.11 | +1.16 | +0.65 | +0.56 |
| real d0 pooled 3 seasons | 306 | -4.75 | **-7.19** | +0.93 | +0.69 | -0.39 | -0.07 | +0.04 | -0.15 | +0.55 | +0.83 | +0.24 |
| real d15+ 2024-25 | 5017 | -5.58 | -0.73 | -1.46 | +1.10 | -0.48 | -0.03 | -0.14 | +0.28 | -0.77 | -1.95 | -1.25 |
| ENGINE d0 2024-25 (closed loop) | 109 | +2.56 | -0.69 | +0.38 | +0.43 | +0.04 | -0.08 | -0.18 | +0.33 | +0.51 | +0.27 | **+1.55** |
| ENGINE d0 2023-24 (closed loop) | 112 | +5.59 | -0.87 | +2.09 | +1.08 | +0.44 | -0.06 | +0.66 | -0.28 | +0.63 | +0.40 | +1.50 |

Today's rates compared with real d0 2024-25:

| channel | today | real d0 2024-25 | verdict |
|---|---|---|---|
| P per team | 67.22 | 70.18 | short |
| TOV/P | 0.184 | 0.183 | at reality |
| OREB/P | 0.161 | 0.159 | at reality |
| FTA/P | 0.267 | 0.293 | short, -0.9 pts |
| 3PA share | 0.401 | 0.399 | at reality |
| rim share of 2PA | 0.608 | 0.600 | at reality |
| rim make | 0.587 | 0.592 | slightly short |
| jump2 make | 0.378 | 0.397 | short, -1.1 pts |
| 3 make | 0.325 | 0.322 | at reality |
| FT make | 0.690 | 0.701 | short, -0.4 pts |
| PPP | 1.040 | 1.056 | short |

- Pace is the only large channel against every opening-day reference (-6.2 to -7.7 points).
- Against the engine's own 2024-25 opening day, today is +2.6 points, mostly FT make (+1.5): FT% 0.690 now vs 0.647 when every slot was anonymous. Pace is nearly unchanged (67.2 vs 67.6).

**(c) Anonymous shooters and missing rotation priors.**
- The log line "rotation priors: 0 of 232" refers to in-season rotation history, which no team has on day 0. The A3 day-1 player priors then seed the slots ("day-1 player priors (reads season S-1 minutes)").
- Anonymous minutes per team-game, from the slate `notes`: mean 26.4 of 200, median 18.6, max 170.
- Team-game regression of sim PPP x100 on `off_c`, opponent `def_c`, site and anonymous minutes, n = 236:
  - The anonymous-minutes coefficient is -0.013 per minute (SE 0.007).
  - At the mean that is about -0.35 per 100, or **about -0.5 points per game**. It is 1.9 SE, confounded with team tier, and should be read as an upper bound.
- This agrees with the earlier estimate of 0.2-0.3 points, and with the +1.5-point FT gain over the fully anonymous 2024-25 opening day. It is not a material owner of today's gap.

**(d) Rule constants.** The 2027 audit (`docs/ops/readiness_gaps_2026-10-05.md`):
- 17 constants: 11 verified.
- 2 changed and not encoded: the 19 s backcourt count and consecutive media timeouts.
- 3 unverified: foul-out 5, FT trip lengths, possession arrow.

None of these is a plausible per-game points-level owner on the scale of a point. The level piece that IS 2027-specific is the carried sub-model level (`season_idx` saturation, refit through 2025-03), about -0.8 points, estimated in (a). Confidence that the rule constants matter for totals: low.

## Q5. Actuals so far (UNDERPOWERED, n = 1)

`finals_verified_2027.parquet` has no game from today's slate; its one row is from 11-01. The chain's 12:00Z CBBD schedule pull (`data/raw/schedule_daily/2026-11-02.parquet`, single source, UNVERIFIED) shows one slate game final:

| game | sim mean (q05-q95) | market | final | source |
|---|---|---|---|---|
| Northern Illinois vs Grambling (401920912) | 132.3 (107-160) | 141.5 | 176 | CBBD status=final, unverified |

One more game (Merrimack vs App State) was in progress at the pull. n = 1 says nothing about level; it is listed only for completeness.

## Recommended next step (a candidate round for the PM, not an adjustment)

**Sub-model: the clock's pregame tempo prior on opening day.** This is the `tempo_prior_game` input, as built by the ratings chain's day-1 branch (`prior_carry` of `league_tempo_mean`) and by `pace.build_pace_table` for training rows.

**Defect.** The d0 input is the prior season's FINAL league tempo, a late-season level. From d1 the as-of league mean is an early-season level, about 2.4-2.8 higher. The engine's d0 pace is therefore 3.2 possessions per team short in both replay seasons, and fine from d1.

**Arms** (pre-registered in `docs/models/clock/experiments.md` before any run):
- **A0 (reference).** d0 league tempo = prior-season final.
- **A1.** d0 league tempo = the prior season's own d1 as-of league mean. This is data-defined, has no new parameter, and is the same construction the engine already sees from d1.
- **A2.** d0 league tempo = the mean of the d1 as-of league means over the 2-3 prior seasons.
- **A3 (optional).** A1, with the clock retrained on d0 training rows rebuilt the same way, for train/serve consistency.

A1 and A2 change only d0 inputs, so d1+ output is identical by construction.

**Primary metric.** Fold-2 d0 (2024-11-04) possession bias and total bias vs verified finals. Fold 1 (2023-11-06) confirms.

**Segments.** Matchup class, abs-spread bin, tempo quintile (responsiveness), and site.

**Noise floor.** A seed-paired rerun. The d0 cell has about 110 games, so the per-game P SE is about 0.5 possessions; the effect to detect is about 3.

**Guardrail.** No d1+ change. The PM should rule that a prior-construction arm like this is not a calendar term (the 10-07 calendar arm was not adopted).

**Second, already queued.** The early-season PPP owners: FT make, FTA rate, jump2 make. They are worth about -2.4 points today.

**Until then.** The owner's practice of not betting d0-14 totals stays correct. On d0 the engine sits about 7.6 points of pace under reality, before any efficiency bias.

## Files

- Script: `scripts/diag_opening_day_totals_v1.py`. With `--run-replay` it reruns the 200-seed d0 replay, about 3 minutes at 8 jobs.
- Outputs: `results/diag/opening_day_totals_2026-11-02/` (`q1_*`, `q2_*`, `q3_*`, `q4a/b/c_*`, `q5_finals.csv`, `meta.json`, `replay_2024-11-04/`).
- Inputs:
  - `results/daily/publish/2026-11-02/s2500_o0/da1e4c7591/slate.parquet`
  - `results/daily/sim/2026-11-02/s2500_o0/games.parquet`
  - `data/processed/truth/game_finals_v2.parquet` and `team_game_shots_v2.parquet`, via `cbb_sim.eval.reference`
  - `data/processed/lines/lines_close_v2_verified.parquet`
  - `data/processed/ratings/own_ratings_{2023..2025}.parquet`
  - `data/processed/ratings_asof/2026-11-02/own_ratings_2027.parquet`
- Code for the mechanism:
  - `src/cbb_sim/live/features.py` (the `tempo_prior_game` build)
  - `src/cbb_sim/models/clock.py` (`build_design`, the same product for training rows)
- Prior docs: `docs/tests/early_total_bias_K2_2026-10-07.md` and `docs/tests/total_bias_decomp_2026-10-05.md`. The "team openers" row there (pace -4.14 / -2.27 on v2) was the first sign of this.
