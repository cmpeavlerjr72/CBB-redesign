# Market scorecard v2 (accepted lines source) -- engine_v0/F2_2025_s200_v1_clockv3c_A (season 2025)

Generated 2026-09-11 by `scripts/grade_market_games_v2.py`. Results: `results/engine_v0/F2_2025_s200_v1_clockv3c_A`. Lines: `data\processed\lines\lines_close_v1.parquet` (provider `ESPN BET`, 0 duplicate (game_id) rows dropped). See the module docstring for the four differences from `scripts/grade_market_games.py` (v1, unmodified).

## 1. Seed-count and backtest-honesty guard

This run has **200 seeds**, at or above the 200-seed gate-report floor.

This run has **200** seeds, at or above the 200-seed floor: section 3's calibration/reliability tables print below, labelled PROVISIONAL unless 200 also clears the 2000-seed floor.

**ROI, Brier and edge-bucket hit-rate numbers are REFUSED below 2000 seeds.** `docs/tests/engine_seed_count_2026-09-10.md` measured that floor for a per-game ROI/Brier read (game-level margin-mean SE is still 2.144 pts and win-prob SE 0.0298 at 200 seeds; at 5 seeds a sim win probability can only take the values {0, 0.2, 0.4, 0.6, 0.8, 1.0}). This run has **200** seeds, below that threshold, so section 3's Brier line and all of section 4 (edge buckets/ROI) print the seed count and threshold ONLY -- no numbers.

backtest: run created post hoc by construction; artifact `max_train_date < tipoff` asserted on possession_outcome.first: 5710/5710 rows pass; possession_outcome.cont: 5710/5710 rows pass; clock.clock: 5710/5710 rows pass; fg_make.FGA_rim: 5710/5710 rows pass; fg_make.FGA_jump2: 5710/5710 rows pass; fg_make.FGA_3: 5710/5710 rows pass; free_throw.FT2_lgbm: 5710/5710 rows pass; rebound.C_plus_state: 5710/5710 rows pass; rotation.R2: 5710/5710 rows pass. **artifact dates not recorded** for 1 other sub-model artifact path(s) (usage (static, no per-artifact dates)) -- not checked, not fabricated, not assumed to pass.

### Second-source truth accounting (game_finals_v2 only)

| n_schedule_games | n_no_finals_row | n_unresolved_excluded | n_truth_games | finals_source_counts |
|---|---|---|---|---|
| 5710 | 0 | 0 | 5710 | {"hoopr": 5709, "cbbd": 1} |

n games with a usable `ESPN BET` close line, after joining truth x sim x lines: 5383 (of 5710 truth games).

## 2. Sim vs market: margin and total point estimates

Model-vs-actual and market-vs-actual MAE/bias (context), then a DIRECT model-vs-market comparison (MAE/bias/corr between the two point estimates themselves).

| n | model_margin_MAE_vs_actual | market_margin_MAE_vs_actual | model_margin_bias | market_margin_bias | model_total_MAE_vs_actual | market_total_MAE_vs_actual | model_total_bias | market_total_bias |
|---|---|---|---|---|---|---|---|---|
| 5383 | 9.2483 | 8.7413 | -0.2128 | -0.1628 | 13.5520 | 12.7297 | -0.9307 | -0.5182 |

| n | margin_MAE(model,market) | margin_bias(model-market) | margin_corr(model,market) | total_MAE(model,market) | total_bias(model-market) | total_corr(model,market) |
|---|---|---|---|---|---|---|
| 5383 | 2.7974 | -0.0500 | 0.9182 | 4.5135 | -0.4332 | 0.7537 |

seed-noise SE of the sim mean at 200 seeds: margin 2.144 pts, total 0.838 pts (extrapolated via the studys c/sqrt(k) law).

## 3. Sim win probability vs de-vigged close probability

**PROVISIONAL** (200 seeds, below the 2000-seed floor for a fully trusted read) -- calibration/reliability tables below are directional only.

n = 5230. mean |prop - power| de-vig difference: 0.01534 (proportional is primary throughout; power is reported for comparison only).

### Calibration vs realised outcome (20 buckets, sim probability)

| bucket | n | mean_p | actual_rate | delta |
|---|---|---|---|---|
| 1.0000 | 267 | 0.1507 | 0.1348 | -0.0159 |
| 2.0000 | 271 | 0.2762 | 0.2694 | -0.0068 |
| 3.0000 | 248 | 0.3411 | 0.4113 | 0.0702 |
| 4.0000 | 276 | 0.3974 | 0.4601 | 0.0628 |
| 5.0000 | 267 | 0.4453 | 0.4607 | 0.0154 |
| 6.0000 | 262 | 0.4875 | 0.4924 | 0.0049 |
| 7.0000 | 254 | 0.5260 | 0.5748 | 0.0488 |
| 8.0000 | 295 | 0.5606 | 0.5932 | 0.0326 |
| 9.0000 | 241 | 0.5929 | 0.5892 | -0.0037 |
| 10.0000 | 267 | 0.6253 | 0.6704 | 0.0451 |
| 11.0000 | 234 | 0.6573 | 0.6752 | 0.0179 |
| 12.0000 | 264 | 0.6883 | 0.7083 | 0.0200 |
| 13.0000 | 256 | 0.7172 | 0.7031 | -0.0141 |
| 14.0000 | 287 | 0.7495 | 0.7526 | 0.0032 |
| 15.0000 | 257 | 0.7873 | 0.8210 | 0.0337 |
| 16.0000 | 267 | 0.8246 | 0.7978 | -0.0268 |
| 17.0000 | 250 | 0.8608 | 0.8640 | 0.0032 |
| 18.0000 | 249 | 0.8950 | 0.8795 | -0.0155 |
| 19.0000 | 279 | 0.9355 | 0.9570 | 0.0215 |
| 20.0000 | 239 | 0.9772 | 0.9791 | 0.0019 |

### Calibration vs realised outcome (20 buckets, market probability)

| bucket | n | mean_p | actual_rate | delta |
|---|---|---|---|---|
| 1.0000 | 266 | 0.1569 | 0.1278 | -0.0291 |
| 2.0000 | 264 | 0.2753 | 0.2538 | -0.0215 |
| 3.0000 | 265 | 0.3507 | 0.3132 | -0.0375 |
| 4.0000 | 273 | 0.4072 | 0.4469 | 0.0397 |
| 5.0000 | 274 | 0.4541 | 0.4599 | 0.0057 |
| 6.0000 | 233 | 0.4932 | 0.4807 | -0.0125 |
| 7.0000 | 268 | 0.5319 | 0.5522 | 0.0204 |
| 8.0000 | 294 | 0.5664 | 0.6293 | 0.0628 |
| 9.0000 | 239 | 0.5971 | 0.5774 | -0.0197 |
| 10.0000 | 271 | 0.6247 | 0.5867 | -0.0380 |
| 11.0000 | 229 | 0.6536 | 0.6856 | 0.0320 |
| 12.0000 | 295 | 0.6884 | 0.7085 | 0.0200 |
| 13.0000 | 251 | 0.7204 | 0.7211 | 0.0007 |
| 14.0000 | 248 | 0.7518 | 0.7581 | 0.0063 |
| 15.0000 | 297 | 0.7911 | 0.8519 | 0.0607 |
| 16.0000 | 238 | 0.8315 | 0.8655 | 0.0340 |
| 17.0000 | 294 | 0.8709 | 0.8912 | 0.0202 |
| 18.0000 | 217 | 0.9031 | 0.9217 | 0.0185 |
| 19.0000 | 253 | 0.9285 | 0.9605 | 0.0319 |
| 20.0000 | 261 | 0.9613 | 0.9962 | 0.0349 |

### Reliability: (sim prob - market prob) deciles vs (realised - market prob)

| decile | n | mean_sim_minus_market | mean_actual_minus_market |
|---|---|---|---|
| 1 | 524 | -0.1856 | 0.0308 |
| 2 | 522 | -0.0889 | 0.0080 |
| 3 | 524 | -0.0541 | 0.0161 |
| 4 | 522 | -0.0298 | 0.0023 |
| 5 | 523 | -0.0072 | 0.0248 |
| 6 | 528 | 0.0114 | 0.0151 |
| 7 | 519 | 0.0294 | 0.0055 |
| 8 | 523 | 0.0515 | -0.0104 |
| 9 | 522 | 0.0833 | 0.0079 |
| 10 | 523 | 0.1608 | 0.0206 |

**Brier REFUSED**: 200 seeds < 2000-seed floor. No Brier number printed.

## 4. Edge buckets: spread, total, moneyline (settled at real -110 / real posted odds)

**REFUSED**: 200 seeds < 2000-seed floor. No ROI or hit-rate numbers printed. Breakdowns (season / month / conference / own-rating quintile) are likewise withheld.

## 5. Leak cross-check: does edge-at-open predict open-to-close movement?

CLAUDE.md: "an edge that beats the close but cannot predict line movement is presumed leaked." Only 2025 has usable `open_spread_home` coverage (`docs/tests/lines_cbbd_validation_2026-09-10.md` addendum A: 0% in 2023-2024, 33.5% in 2025).

n games with an open: 1802; n with a nonzero close-vs-open move: 1086.
corr(edge at open, open-to-close movement) = 0.1683 (the sim mean feeding `edge_at_open` carries 2.14 pts of seed noise at 200 seeds, which attenuates any correlation read here toward zero).

2x2 (edge sign at open vs movement sign, nonzero moves only):

| edge_at_open | moved away | moved home |
|---|---|---|
| edge<0 (away) | 309 | 220 |
| edge=0 | 0 | 1 |
| edge>0 (home) | 220 | 336 |

CLV sign agreement: 0.5939 on 1086 moved lines (gate 0.53: near-coinflip = presumed-leaked signature if the surprise correlation above is also real).

