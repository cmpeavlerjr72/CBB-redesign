# Total-bias decomposition of served stack v2 by season and time-in-season (2026-10-05)

Scoring-level worker. Diagnostic only, nothing fitted or adopted. Sims: existing 200-seed served-v2 reads,
`v3full_COMB9GCTKD_s200_o0` (fold 2, 2024-25) and `f1c_V2_full_s200_o0` (fold 1, 2023-24, pulled from HF `results`).
Truth: verified finals, hoopR team box (TOV, OREB, FTA, FTM), event shot classes (`team_game_shots_v2`). Games whose box
has no event shot classes are dropped (81 F1, 111 F2), which leaves the identity residual at 0.0-0.2 points. 2025-26 was not read.

**Method.** Accounting identity per team: P = FGA - OREB + TOV + 0.44 FTA (box possessions, same in sim and truth),
pts = f(P, TOV/P, OREB/P, FTA/P, 3PA share, rim share of 2PA, rim / jump2 / 3 / FT make). Exact Shapley over the 10
factors on bucket-mean counts gives each factor's contribution in points per game (sim - actual).
Script `scripts/diag_total_bias_decomp_v1.py`, FT slot split `scripts/diag_total_bias_ft_slots_v1.py`.
Tables are in `results/total_bias_decomp/` (gitignored).

## 1. Decomposition (points per game, sim - actual)

| season | bucket | games | total bias | pace | TOV | OREB | FTA rate | FT make | rim share | rim make | jump2 make | 3 make |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023-24 (F1) | all | 5551 | **-1.72** | +0.39 | **-1.00** | +0.14 | -0.28 | -0.50 | -0.08 | +0.16 | -0.43 | -0.30 |
| | d0-14 | 657 | **-5.19** | -0.23 | **-2.12** | -0.54 | -0.99 | -1.08 | -0.62 | +0.55 | -0.51 | +0.19 |
| | d15-45 | 1171 | -2.08 | +0.18 | -0.91 | -0.36 | -0.40 | -0.49 | -0.22 | +0.31 | -0.32 | -0.04 |
| | d46+ | 3723 | -1.00 | +0.58 | -0.83 | +0.42 | -0.11 | -0.42 | +0.05 | +0.05 | -0.44 | -0.46 |
| | team openers | 218 | -9.79 | -4.14 | -2.00 | -0.26 | -1.07 | -1.29 | -1.07 | +0.46 | -0.38 | -0.16 |
| 2024-25 (F2) | all | 5589 | **-0.29** | +0.82 | -0.16 | -0.25 | -0.11 | -0.35 | -0.21 | -0.15 | +0.19 | +0.04 |
| | d0-14 | 572 | **-4.20** | -0.28 | -0.52 | -0.12 | -0.77 | **-1.37** | -0.42 | -0.21 | -0.70 | +0.14 |
| | d15-45 | 1190 | -0.37 | +1.41 | -0.04 | -0.45 | -0.19 | -0.36 | -0.28 | -0.29 | -0.17 | -0.02 |
| | d46+ | 3827 | +0.32 | +0.81 | -0.14 | -0.20 | +0.03 | -0.20 | -0.16 | -0.10 | +0.44 | +0.04 |
| | team openers | 229 | -7.26 | -2.27 | -0.14 | -0.08 | -0.82 | -1.98 | -0.18 | -0.81 | -0.31 | -0.74 |

3PA share contributes 0.00 everywhere. Rates (F2, d0-14, sim / actual): FT% 0.673 / 0.708, FTA/P 0.270 / 0.292, TOV/P 0.185 / 0.181,
P per team 69.6 / 69.8. Later (d46+): FT% 0.720 / 0.725, FTA/P 0.282 / 0.281.

**Site** (home-court / neutral): F1 -1.90 / -0.49, F2 -0.30 / -0.24. Neutral cells hold 693 / 710 games and the gap is not separable from mix.
In home-court games of the d0-14 window, home teams carry -3.07 / -2.26 (F1 / F2) and away teams -2.60 / -1.87. Both sides are low, so margin is unaffected.
**Responsiveness** (quintile of predicted total, Q1 to Q5): F1 -2.08 -2.38 -2.22 -1.44 -0.51. F2 -0.74 -0.45 -0.67 +0.97 -0.55.
By prior-season scoring quintile (S-1 actual totals, grading only): F1 -0.55 to -2.81 and F2 +1.56 to -1.50, falling from low to high.
So high-scoring-environment teams are under-predicted most. This is the G9 slope < 1 seen from the total side. Cells hold about 1,100 games each.

## 2. Readings

1. **Pace is not the owner.** The clock over-produces possessions slightly in both folds: +0.39 / +0.82 points for the full season, and pace is about 0 in the opening window.
   The "clock / PPP level term" attribution of the cross-season drift is refuted at the clock end. All of the under-prediction is per possession.
2. **Cross-season drift (F1 -1.72 vs F2 -0.29) is mostly TOV level.** Sim TOV/P is 0.179 against 0.172 actual in F1, and matches in F2 (0.175 / 0.174).
   Actual TOV/P stepped from 0.1835 (2022-23) to 0.1724 (2023-24). The F1 possession-outcome model trains through 2022-23 and carries the old level.
   This is the season-level drift object (`season_drift`, anchor `O`; possession_outcome TOV), not an early-season one.
   Secondary F1-only pieces: jump2 make -0.43 and 3 make -0.30.
3. **Opening window (d0-14, -4.2 / -5.2): the free-throw channel is the largest single owner that both folds share.**
   FT make plus FTA rate costs -2.14 (F2) / -2.07 (F1). The rest is spread across TOV (F1 -2.1, the same level drift amplified), rim share, jump2 make and OREB.
4. **FT make in the window splits into two owners, with matching sizes in both folds:**

| | F2 d0-14 | F1 d0-14 |
|---|---|---|
| sim FT% / actual | 0.673 / 0.708 | 0.674 / 0.703 |
| anonymous-slot FTA share, sim FT% on them | 16.4%, 0.589 | 21.3%, 0.595 |
| anon-slot contribution to the FT% gap | about -1.9 pp | about -2.3 pp |
| named players: sim FT% | 0.690 | 0.696 |
| **offline served FT model on the real window attempts, mean p - mean y** | **-1.73 pp** (newcomers -2.7, returners -1.4) | **-0.62 pp** (-1.2 / -0.4) |
| offline gap d15-45 / d46+ | -0.39 / +0.20 pp | 0.00 / -0.19 pp |

   (a) **Anonymous-slot shooter prior.** The all-zero block plays at 0.59 against 0.695 for real off-roster shooters.
   Owners: player-layer day-1 (A3 selected) and FT section 13.3 `A1` (put forward 2026-10-01).
   (b) **The FT model itself under-predicts early-season attempts.** `shooter_fta_asof` is a raw count. In training, a low count pools early-season shooters with rarely-fouled, weak late-season shooters.
   No feature tells the model which of the two it is looking at, so in November every shooter is scored like the low-volume pool.
   This is the testable sub-model defect taken to Phase 2 (`free_throw/experiments.md` section 18).
5. **Not owned here and left open:** the FTA-rate deficit in the window (sim FTA/P 0.270 vs 0.292; actual early-season FTA/P runs above the season level, sim runs below it); window jump2 make; F1 TOV level.
