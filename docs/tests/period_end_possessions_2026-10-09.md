# D3 period-end possession surplus and garbage-time scoring (fold 2 / 2024-25), 2026-10-09

DIAGNOSTIC ONLY. Nothing changed. Script `scripts/diag_c4_d34_endgame_v1.py` (library `diag_c4_lib_v1.py`).

Sim: served stack v3, fold 2 / season 2025, 5500 games x 50 seeds; real: 5500 games whose parsed pbp path ends at the verified final. Sim columns are seed-averaged. CIs: game-cluster bootstrap (300 reps) of real - sim; `*` marks a CI that excludes 0; `U` = underpowered (real possessions or games in cell < 60).

## (a) Rule out segmentation: independent recount of real possessions in the last 4 minutes of each half

Three real counts and the same counts on the sim:

- **M-start / M-end**: possession-table count (`possessions_v4otc`, the pbp module the earlier doc used) of possessions that START / END with <= 240 s left in the period.
- **E1 (Oliver)**: FGA + 0.44 FTA + TOV - OREB from RAW hoopR pbp events with <= 240 s left (does not touch the possession module; dead-ball team rebounds between the two free throws of one trip are removed).
- **E2 (same-team runs)**: from raw pbp events (shots, free throws, turnovers, rebounds with a team id), the number of maximal runs of consecutive events by the same team, counted when the run's last event has <= 240 s left. A run is a possession-change marker count: no rebound or free-throw heuristics. Over full regulation it gives 133.52 possessions per game against the module's 133.53 (394-game check) and Oliver's 132.9, so it is a valid second method.
- Sim: M-start/M-end from the trajectory; E1 from the engine's own FGA/FTA/TOV/OREB counts of possessions ending in the window; E2 = number of possessions ending in the window (every engine possession ends in a terminal event, the last one is forced to resolve).

| half window | count | real per game (both teams) | sim per game | real - sim [95%] | seed band excl. | flag |
|---|---|---:|---:|---|---|---|
| min 16-20 (end of H1) | M-start | 12.990 | 13.393 | -0.403 [-0.455, -0.339] | | excludes 0 |
| min 16-20 (end of H1) | M-end | 13.989 | 14.393 | -0.404 [-0.456, -0.340] | | excludes 0 |
| min 16-20 (end of H1) | M-end Oliver (same formula on module / engine counts) | 13.732 | 14.178 | -0.446 [-0.494, -0.383] | | excludes 0 |
| min 16-20 (end of H1) | E1 (Oliver, raw events) | 13.622 | 14.178 | -0.556 [-0.607, -0.494] | | excludes 0 |
| min 16-20 (end of H1) | E2 (same-team runs, raw events) | 14.243 | 14.393 | -0.150 [-0.206, -0.086] | | excludes 0 |
| min 36-40 (end of H2) | M-start | 14.522 | 15.747 | -1.225 [-1.314, -1.152] | | excludes 0 |
| min 36-40 (end of H2) | M-end | 15.520 | 16.747 | -1.226 [-1.314, -1.153] | | excludes 0 |
| min 36-40 (end of H2) | M-end Oliver (same formula on module / engine counts) | 15.044 | 16.243 | -1.199 [-1.284, -1.125] | | excludes 0 |
| min 36-40 (end of H2) | E1 (Oliver, raw events) | 14.925 | 16.243 | -1.318 [-1.403, -1.245] | | excludes 0 |
| min 36-40 (end of H2) | E2 (same-team runs, raw events) | 15.817 | 16.747 | -0.930 [-1.019, -0.855] | | excludes 0 |

Agreement test (the instruction: if two independent real counts disagree by more than the sim-real gap, D3 stops).

| half window | sim - real (M-start) | sim - real (M-end) | sim - real (E1) | sim - real (E2) | real E1 - real M-end | real E2 - real M-end | agrees? |
|---|---:|---:|---:|---:|---:|---:|---|
| min 16-20 (end of H1) | 0.403 | 0.404 | 0.556 | 0.150 | -0.367 | 0.255 | YES |
| min 36-40 (end of H2) | 1.225 | 1.226 | 1.318 | 0.930 | -0.595 | 0.296 | YES |

`agrees? = YES` when both independent real counts sit closer to the module's real count than the sim does (|real E - real M-end| < |sim - real M-end|), i.e. the segmentation is not what separates the sim from reality.

Context, the period-end terminal event in the real table (possessions whose terminal event is `end_period`/`unknown`): see the possession-class table below (NONE row).

## (b) By margin state at 4:00 and 2:00 (|margin| buckets 0-3, 4-7, 8-12, 13+), leader vs trailer

Reference state = home margin before the first possession that starts with <= 240 s (or <= 120 s) left in the half; every possession of the window inherits that reference bucket and the role (leader / trailer = the team ahead / behind at the reference; tied = tied). Class rule (identical both sides): TOV > FT trip (fta > 0) > 3PA > 2PA. Cells show `real / sim (real - sim)`.

### Half 2, window = last 4 min, reference state at 4:00

| bucket | role | real poss | sim poss/seed | sec/poss (real / sim, diff) | FT-trip share (pp) | 3PA share (pp) | TOV rate (pp) | PPP |
|---|---|---:|---:|---|---|---|---|---|
| 0-3 | leader | 6946 | 6854 | 16.44 / 15.12 (1.32*) | 34.8 / 33.8 (1.0) | 20.7 / 20.0 (0.8) | 14.4 / 14.9 (-0.5) | 1.136 / 1.144 (-0.008) |
| 0-3 | tied | 2199 | 2319 | 16.20 / 15.28 (0.92*) | 30.0 / 29.2 (0.8) | 24.2 / 23.3 (0.9) | 12.9 / 14.8 (-1.9*) | 1.098 / 1.129 (-0.031) |
| 0-3 | trailer | 7106 | 6884 | 15.08 / 15.11 (-0.03) | 26.7 / 25.9 (0.8) | 27.6 / 26.9 (0.7) | 12.8 / 14.0 (-1.2*) | 1.115 / 1.133 (-0.019) |
| 13+ | leader | 13845 | 17100 | 17.92 / 14.94 (2.98*) | 23.8 / 25.0 (-1.2*) | 24.8 / 22.1 (2.6*) | 19.0 / 22.1 (-3.1*) | 1.108 / 1.090 (0.018*) |
| 13+ | trailer | 14746 | 17112 | 13.87 / 14.27 (-0.40*) | 22.2 / 21.2 (1.0*) | 31.5 / 32.2 (-0.7) | 12.9 / 12.8 (0.1) | 1.120 / 1.086 (0.034*) |
| 4-7 | leader | 8553 | 8769 | 15.98 / 14.31 (1.67*) | 41.8 / 40.7 (1.0) | 15.0 / 15.9 (-0.9*) | 15.4 / 15.8 (-0.4) | 1.157 / 1.165 (-0.008) |
| 4-7 | trailer | 8929 | 8859 | 13.51 / 14.41 (-0.90*) | 23.9 / 22.7 (1.2*) | 31.9 / 31.6 (0.4) | 12.0 / 13.0 (-1.0*) | 1.135 / 1.135 (0.000) |
| 8-12 | leader | 8548 | 9317 | 15.86 / 13.87 (2.00*) | 42.3 / 41.4 (0.9) | 13.7 / 14.9 (-1.1*) | 17.9 / 17.9 (-0.0) | 1.170 / 1.164 (0.006) |
| 8-12 | trailer | 9000 | 9397 | 12.33 / 14.01 (-1.68*) | 22.7 / 21.2 (1.5*) | 33.4 / 33.7 (-0.4) | 11.5 / 12.4 (-0.9*) | 1.171 / 1.127 (0.043*) |

### Half 2, window = last 2 min, reference state at 2:00

| bucket | role | real poss | sim poss/seed | sec/poss (real / sim, diff) | FT-trip share (pp) | 3PA share (pp) | TOV rate (pp) | PPP |
|---|---|---:|---:|---|---|---|---|---|
| 0-3 | leader | 3875 | 3649 | 13.96 / 13.22 (0.74*) | 46.0 / 42.0 (4.0*) | 15.5 / 16.1 (-0.6) | 13.4 / 14.6 (-1.1*) | 1.173 / 1.144 (0.029) |
| 0-3 | tied | 1231 | 1225 | 14.16 / 13.85 (0.30) | 34.6 / 32.4 (2.2) | 23.1 / 22.2 (0.9) | 12.7 / 14.3 (-1.6) | 1.104 / 1.125 (-0.021) |
| 0-3 | trailer | 4036 | 3697 | 12.77 / 13.32 (-0.55*) | 24.7 / 26.0 (-1.4) | 32.0 / 28.9 (3.2*) | 12.4 / 13.8 (-1.5*) | 1.084 / 1.125 (-0.041*) |
| 13+ | leader | 6465 | 10125 | 17.17 / 12.76 (4.41*) | 24.2 / 26.7 (-2.5*) | 24.2 / 19.8 (4.5*) | 21.1 / 25.4 (-4.3*) | 1.050 / 1.051 (-0.001) |
| 13+ | trailer | 7381 | 10129 | 12.51 / 11.20 (1.30*) | 22.0 / 21.1 (0.9) | 33.6 / 34.2 (-0.6) | 11.9 / 12.2 (-0.2) | 1.122 / 1.086 (0.036*) |
| 4-7 | leader | 5158 | 5222 | 11.90 / 11.25 (0.65*) | 59.0 / 55.8 (3.3*) | 8.0 / 10.0 (-1.9*) | 14.2 / 14.3 (-0.1) | 1.209 / 1.215 (-0.006) |
| 4-7 | trailer | 5597 | 5334 | 10.81 / 11.19 (-0.37*) | 22.7 / 21.9 (0.8) | 37.1 / 35.5 (1.6*) | 10.1 / 12.4 (-2.3*) | 1.149 / 1.135 (0.014) |
| 8-12 | leader | 4796 | 5580 | 10.62 / 10.71 (-0.09) | 59.5 / 55.4 (4.1*) | 7.0 / 9.1 (-2.1*) | 16.4 / 17.4 (-0.9) | 1.242 / 1.215 (0.026*) |
| 8-12 | trailer | 5266 | 5662 | 10.20 / 10.88 (-0.68*) | 20.9 / 20.6 (0.2) | 37.7 / 37.3 (0.4) | 10.8 / 11.7 (-0.9) | 1.180 / 1.131 (0.049*) |

### Half 1, window = last 4 min, reference state at 4:00

| bucket | role | real poss | sim poss/seed | sec/poss (real / sim, diff) | FT-trip share (pp) | 3PA share (pp) | TOV rate (pp) | PPP |
|---|---|---:|---:|---|---|---|---|---|
| 0-3 | leader | 8776 | 8883 | 17.77 / 17.11 (0.66*) | 16.2 / 15.5 (0.7) | 29.5 / 29.9 (-0.4) | 17.1 / 17.5 (-0.4) | 1.025 / 1.039 (-0.013) |
| 0-3 | tied | 3215 | 3066 | 17.58 / 17.14 (0.44*) | 14.9 / 15.6 (-0.7) | 30.3 / 29.7 (0.6) | 16.6 / 17.4 (-0.8) | 1.040 / 1.033 (0.007) |
| 0-3 | trailer | 8789 | 8888 | 17.58 / 17.15 (0.43*) | 15.7 / 15.8 (-0.1) | 29.2 / 29.7 (-0.5) | 17.3 / 17.5 (-0.1) | 1.018 / 1.031 (-0.013) |
| 13+ | leader | 6963 | 7254 | 17.31 / 16.75 (0.56*) | 15.5 / 15.2 (0.3) | 30.8 / 31.3 (-0.5) | 17.0 / 17.9 (-0.9) | 1.085 / 1.090 (-0.004) |
| 13+ | trailer | 6992 | 7274 | 16.94 / 17.20 (-0.27*) | 17.1 / 16.7 (0.4) | 28.7 / 29.0 (-0.3) | 17.8 / 17.7 (0.1) | 0.986 / 0.979 (0.007) |
| 4-7 | leader | 9775 | 10294 | 17.80 / 17.05 (0.75*) | 15.6 / 15.6 (0.1) | 30.0 / 30.0 (-0.0) | 17.2 / 17.7 (-0.5) | 1.037 / 1.047 (-0.011) |
| 4-7 | trailer | 9832 | 10305 | 17.47 / 17.20 (0.27*) | 16.6 / 16.0 (0.6) | 30.0 / 29.7 (0.3) | 17.4 / 17.3 (0.1) | 1.017 / 1.023 (-0.006) |
| 8-12 | leader | 8551 | 8840 | 17.78 / 16.97 (0.81*) | 15.1 / 15.4 (-0.4) | 31.2 / 30.2 (1.0) | 17.0 / 18.0 (-1.1*) | 1.035 / 1.056 (-0.021) |
| 8-12 | trailer | 8552 | 8860 | 17.13 / 17.22 (-0.09) | 16.7 / 16.2 (0.4) | 29.0 / 29.6 (-0.6) | 17.9 / 17.2 (0.7) | 1.002 / 1.008 (-0.006) |

### Half 1, window = last 2 min, reference state at 2:00

| bucket | role | real poss | sim poss/seed | sec/poss (real / sim, diff) | FT-trip share (pp) | 3PA share (pp) | TOV rate (pp) | PPP |
|---|---|---:|---:|---|---|---|---|---|
| 0-3 | leader | 3897 | 4154 | 17.52 / 16.43 (1.08*) | 16.1 / 16.2 (-0.1) | 30.7 / 30.1 (0.5) | 16.4 / 17.5 (-1.1*) | 1.007 / 1.030 (-0.023) |
| 0-3 | tied | 1462 | 1408 | 17.11 / 16.55 (0.56*) | 16.2 / 16.3 (-0.1) | 30.6 / 30.3 (0.3) | 16.7 / 17.1 (-0.5) | 0.983 / 1.031 (-0.048) |
| 0-3 | trailer | 3936 | 4154 | 17.09 / 16.50 (0.59*) | 16.8 / 16.5 (0.3) | 29.9 / 30.3 (-0.4) | 16.7 / 17.3 (-0.5) | 1.008 / 1.030 (-0.021) |
| 13+ | leader | 3833 | 4130 | 17.05 / 16.11 (0.94*) | 15.9 / 15.5 (0.4) | 30.4 / 31.9 (-1.5) | 17.2 / 18.0 (-0.8) | 1.061 / 1.081 (-0.020) |
| 13+ | trailer | 3831 | 4148 | 16.51 / 16.60 (-0.09) | 17.5 / 17.1 (0.4) | 29.7 / 29.8 (-0.1) | 17.8 / 17.4 (0.4) | 0.955 / 0.979 (-0.024) |
| 4-7 | leader | 4680 | 4863 | 17.35 / 16.38 (0.97*) | 15.5 / 16.6 (-1.1*) | 30.7 / 30.4 (0.4) | 17.0 / 17.6 (-0.6) | 1.005 / 1.041 (-0.036) |
| 4-7 | trailer | 4706 | 4880 | 16.87 / 16.54 (0.32*) | 16.9 / 16.5 (0.4) | 30.0 / 30.5 (-0.4) | 16.9 / 17.2 (-0.3) | 1.005 / 1.021 (-0.016) |
| 8-12 | leader | 4081 | 4345 | 17.59 / 16.26 (1.33*) | 16.4 / 16.3 (0.1) | 31.8 / 30.6 (1.2) | 16.4 / 18.1 (-1.7*) | 1.037 / 1.056 (-0.019) |
| 8-12 | trailer | 4076 | 4361 | 16.90 / 16.65 (0.26*) | 17.2 / 16.5 (0.6) | 30.2 / 30.6 (-0.4) | 17.0 / 16.9 (0.1) | 0.994 / 1.010 (-0.016) |

### Last-2-minute points and possessions per game (both teams), by margin bucket at 2:00 (half 2)

| bucket | games real (U if < 60) | share real | share sim | poss/game real / sim (diff) | points/game real / sim (diff) | PPP real / sim (diff) |
|---|---:|---:|---:|---|---|---|
| 0-3 | 1154 | 0.210 | 0.192 | 7.92 / 8.12 (-0.20*) | 8.91 / 9.20 (-0.29*) | 1.124 / 1.133 (-0.009) |
| 4-7 | 1163 | 0.211 | 0.198 | 9.25 / 9.67 (-0.42*) | 10.89 / 11.36 (-0.47*) | 1.178 / 1.175 (0.003) |
| 8-12 | 1060 | 0.193 | 0.203 | 9.49 / 10.06 (-0.57*) | 11.48 / 11.80 (-0.32) | 1.209 / 1.173 (0.036*) |
| 13+ | 2123 | 0.386 | 0.406 | 6.52 / 9.06 (-2.54*) | 7.10 / 9.68 (-2.58*) | 1.089 / 1.069 (0.020*) |

### Possession-class mix in the last 4 minutes of H2 (all states): real vs sim

| class | real share | sim share |
|---|---:|---:|
| 2PA | 31.20% | 31.21% |
| 3PA | 25.44% | 25.18% |
| FT | 28.70% | 27.85% |
| TOV | 14.61% | 15.77% |
| NONE | 0.05% | 0.00% |

## (c) Verdict

**Segmentation is ruled out for the second half and underdetermined for the first; the second-half surplus is real, lives in lopsided end-of-half states, and is owned by the clock model (a margin-magnitude x time term in the duration law that the late-game regime does not cover).** (a) Over 5,500 fold-2 games the sim plays 1.23 more possessions than the possession table in minutes 36-40 (M-end, CI [-1.31, -1.15]); an independent Oliver count from raw pbp events gives 1.32 (the same formula on the engine's own counts) and an independent count of same-team event runs gives 0.93 (that method reproduces the module to 133.52 vs 133.53 possessions per full regulation game on the 394-game check). The two independent real counts sit 0.30-0.60 from the module's, inside the 0.93-1.32 gap, so the H2 surplus stands under every method. The first-half end (minutes 16-20) is smaller and method-dependent: 0.40 (module), 0.56 (Oliver), 0.15 (runs; CI still excludes 0): not resolvable as a count defect, labelled as such. (b) The H2 surplus is concentrated where reality stops playing: in the 13+ state at 2:00 (39% of games) the sim plays 9.06 possessions in the last two minutes against 6.52 real and scores 9.68 against 7.10 points (-2.58 pts/game in the cell, about -1.0 per game overall); the 0-3, 4-7 and 8-12 states are -0.29, -0.47 and -0.32 points. Per-possession PPP is close in every state (real - sim -0.01 to +0.04). The sim's LEADER is the fast one: seconds per possession with the ball when leading by 13+ are 17.17 real vs 12.76 sim at the 2:00 reference (4:00 reference: 17.92 vs 14.94), by 8-12 15.86 vs 13.87, by 4-7 15.98 vs 14.31 at 4:00; the trailer is faster in reality at 4-12 points (4:00 reference: 13.51 vs 14.41, 12.33 vs 14.01) and slower at 13+ at 2:00 (12.51 vs 11.20): reality has a leader-minus-trailer clock gap of 4.7 s in blowouts at 2:00 against 1.6 s in the sim. The leader's TOV rate in 13+ states is also too high in the sim (22.1% vs 19.0% at 4:00, 25.4% vs 21.1% at 2:00) while FT-trip shares are within about 1 to 4 pp: foul accrual is not the owner of the possession surplus. The existing late-game regime is declared on |margin| <= 6 in the last 2:00 (late_game section 1.1, rounds 2-6), which excludes the 8-12 and 13+ states and the 4:00-2:00 span where the effect starts, so no closed arm covers it. Verdict: clock (duration law lacks a margin-magnitude x time term for the leader) with the leader TOV overshoot as a reported co-symptom. Draft: `docs/models/late_game/experiments_DRAFT_endgame_margin_clock_foul_2026-10-09.md`.
