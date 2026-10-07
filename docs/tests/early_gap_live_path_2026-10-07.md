# Early-season total gap under the LIVE 2026-27 path (A3 day-1 priors + R1), 2026-10-07

Measurement only; no model changed; 2025-26 sealed (not read). Worker analysis for the PM.

## Method
- Control: served v3 (current defaults, parity v10; F1 = `engine_v3_f1` + `docs/ops/box_queue/d1007_K2F1_overrides.json`, `ENGINE_ROTATION_SCHEME=static`; F2 = `engine_v3`) with historical in-season priors, i.e. a team's first game anonymous, later games named from earlier games this season.
- Live path: `build_live` UNPATCHED with the A3 seed (`scripts/build_engine_inputs_day1prior_v1.py` `build`, `--fallback R1`), built by the new `scripts/build_engine_inputs_d1p_live_early_v1.py`. The seed acts only on all-anonymous team-games (a team's first game); every later game keeps its served in-season rotation prior. This differs from the 2026-10-05 bake-off window (which suppressed the in-season prior for all 14 days): here days 0-45 are built as 2026-27 will be served. Seeded: F2 497 team-games in 323 games, F1 738 in 550. Shot-block LUT shooter/known zeroed on seeded sides only (bake-off convention; conservative). R1 fired for 0 teams (every team had a season-S roster in replay), so the arm is A3 plus an inactive fallback; the roster-less case is covered only by `roster_fallback_2026-10-05.md`.
- Runs: `scripts/run_engine_window_v1.py`, days 0-45 games (1836 F2, 1879 F1), 50 seeds each, paired streams (identical RNG keys), 8 workers. The F2 control matches the earlier 50-seed taps (decomp d0-14 -4.59/-5.06 reproduced exactly). Verified finals via `reference.load_actual_games`.
- Grader/decomp: `scripts/diag_early_gap_live_path_v1.py` (per-game bias = per-game seed-mean minus actual; paired diff SE over games; Shapley decomposition by `scripts/diag_early_total_points_decomp_v1.py` unedited, on each arm). Games without event shot classes are dropped in the decomposition only. Outputs `results/early_gap_live_path/` (gitignored).

## 1. Bias (sim minus actual, points), paired
| fold | bucket | metric | n | control bias | A3+R1 bias | paired diff (SE) |
|---|---|---|---|---|---|---|
| F2 | d0-14 | total | 620 | -5.16 | -4.78 | +0.38 (0.04) |
| F2 | d15-45 | total | 1215 | -0.13 | -0.09 | +0.04 (0.01) |
| F2 | d0-45 | total | 1835 | -1.83 | -1.68 | +0.15 (0.01) |
| F1 | d0-14 | total | 697 | -4.93 | -4.26 | +0.67 (0.05) |
| F1 | d15-45 | total | 1182 | -1.80 | -1.46 | +0.35 (0.03) |
| F1 | d0-45 | total | 1879 | -2.96 | -2.50 | +0.47 (0.03) |
| F2 | d0-14 | margin | 620 | -0.71 | -0.60 | +0.11 (0.04) |
| F2 | d15-45 | margin | 1215 | -0.55 | -0.55 | -0.00 (0.01) |
| F1 | d0-14 | margin | 697 | -0.64 | -0.53 | +0.11 (0.06) |
| F1 | d15-45 | margin | 1182 | -0.09 | -0.18 | -0.09 (0.03) |

By seeding (d0-14, total): games with a seeded side F2 n=300, -8.70 -> -7.92 (+0.78, SE 0.07); F1 n=364, -6.90 -> -5.61 (+1.29, SE 0.08). Unseeded games identical by construction (F2 -1.84, F1 -2.79). Total MAE d0-14: F2 14.82 -> 14.70, F1 14.27 -> 14.14. Margin MAE d0-45 changes <= 0.07.

Reading: the live path closes only 0.4 (F2) to 0.7 (F1) of the 4.3-5.2 pt d0-14 gap, 7-14%. The gap that remains under the live path is -4.8 (F2) and -4.3 (F1) in d0-14; the opening-game fix is small because the seeded games are only about half of d0-14 games and the seeded games stay 5.6-7.9 low. The d15-45 F1 gain (+0.35, 183 seeded games) is mostly second games of teams whose first game was seeded, not a persistent state; F2 d15-45 has only 22 seeded games (underpowered cell, the +0.04 is not a read). The paired SEs are over games with 50 seeds averaged in; they do not include sampling error of the actuals, which is shared and cancels in the diff, so the diff is well determined but the control bias SE (0.5-0.9) is the right scale for how well the level itself is known.

## 2. Shapley points channels, d0-14 (sim minus actual, points; games with event shot classes: F2 572, F1 657)
| channel | F2 ctrl | F2 A3+R1 | F1 ctrl | F1 A3+R1 |
|---|---|---|---|---|
| total gap | -4.59 | -4.23 | -5.06 | -4.37 |
| possessions | -0.92 | -0.92 | -0.51 | -0.59 |
| TOV rate | -0.57 | -0.56 | -2.20 | -2.18 |
| OREB | -0.23 | -0.28 | -0.64 | -0.76 |
| FTA rate (net) | -0.70 | -0.71 | -0.90 | -0.93 |
| FT% | -1.37 | -1.04 | -1.06 | -0.60 |
| rim share | -0.38 | -0.38 | -0.56 | -0.57 |
| rim make | -0.02 | -0.03 | +0.69 | +0.80 |
| jump2 make | -0.67 | -0.65 | -0.42 | -0.36 |
| 3 make | +0.21 | +0.27 | +0.37 | +0.65 |

Top four channels by size. F2 control: FT% -1.37, possessions -0.92, FTA -0.70, jump2 make -0.67. F2 A3+R1: FT% -1.04, possessions -0.92, FTA -0.71, jump2 make -0.65. F1 control: TOV -2.20, FT% -1.06, FTA -0.90, rim make +0.69 (OREB -0.64 next). F1 A3+R1: TOV -2.18, FTA -0.93, rim make +0.80, OREB -0.76 (FT% -0.60, 3 make +0.65).

The live path moves FT% (+0.33 F2, +0.46 F1) and 3-point/rim make rates up (F1 3 make +0.28, rim +0.11), as the anonymous-slot mechanism predicts. It does not touch the possession, TOV, FTA-rate or shot-mix channels: those, the larger part of the gap, are team-level and not driven by anonymous player slots. Residual after the live path is about -4.2 / -4.4 in d0-14.

## 3. Verdict for 2026-27 opening weeks
Under the live path the d0-14 total is still about 4-5 pts low (F2 -4.8, F1 -4.3 on all games); d0-45 about -1.7 (F2) and -2.5 (F1). Of the 10-05 anonymous-slot size (about -2.6) most is recovered only on the first games, and the live path never reaches the old anonymous state beyond day one. The remaining d0-14 gap is owned by possessions (pace), TOV (F1 level), FTA rate and residual FT%, none of which A3 or R1 addresses. Nothing here is a model change.

Caveats: 2 folds, 50 seeds; per-fold d0-14 team cells are 2-3 games (underpowered, not cut here); R1 inactive in replay; F1 uses static rotation (stated deviation, as in the 10-05 doc).
