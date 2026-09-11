# Market scorecard v2 (accepted lines source) -- engine_v0/F2_2025_s200_v5b_A (season 2025)

Generated 2026-09-10 by `scripts/grade_market_games_v2.py`. Results: `results/engine_v0/F2_2025_s200_v5b_A`. Lines: `data\processed\lines\lines_close_v1.parquet` (provider `ESPN BET`, 0 duplicate (game_id) rows dropped). See the module docstring for the four differences from `scripts/grade_market_games.py` (v1, unmodified).

## 1. Seed-count and backtest-honesty guard

**EVERY TABLE IN THIS REPORT IS PROVISIONAL.** This run has **75 seeds**, below the `docs/tests/engine_seed_count_2026-09-10.md` floor of 200 seeds for even the slate-level gate report.

**Section 3 (probability calibration) is REFUSED below 200 seeds** -- this run has **75**, below that floor. No calibration/reliability numbers printed.

**ROI, Brier and edge-bucket hit-rate numbers are REFUSED below 2000 seeds.** `docs/tests/engine_seed_count_2026-09-10.md` measured that floor for a per-game ROI/Brier read (game-level margin-mean SE is still 3.496 pts and win-prob SE 0.0514 at 75 seeds; at 5 seeds a sim win probability can only take the values {0, 0.2, 0.4, 0.6, 0.8, 1.0}). This run has **75** seeds, below that threshold, so section 3's Brier line and all of section 4 (edge buckets/ROI) print the seed count and threshold ONLY -- no numbers.

backtest: run created post hoc by construction; artifact `max_train_date < tipoff` asserted on possession_outcome.first: 5710/5710 rows pass; possession_outcome.cont: 5710/5710 rows pass; clock.clock: 5710/5710 rows pass; fg_make.FGA_rim: 5710/5710 rows pass; fg_make.FGA_jump2: 5710/5710 rows pass; fg_make.FGA_3: 5710/5710 rows pass; free_throw.FT2_lgbm: 5710/5710 rows pass; rebound.C_plus_state: 5710/5710 rows pass; rotation.R2: 5710/5710 rows pass. **artifact dates not recorded** for 1 other sub-model artifact path(s) (usage (static, no per-artifact dates)) -- not checked, not fabricated, not assumed to pass.

### Second-source truth accounting (game_finals_v2 only)

| n_schedule_games | n_no_finals_row | n_unresolved_excluded | n_truth_games | finals_source_counts |
|---|---|---|---|---|
| 5710 | 0 | 0 | 5710 | {"hoopr": 5709, "cbbd": 1} |

n games with a usable `ESPN BET` close line, after joining truth x sim x lines: 5383 (of 5710 truth games).

## 2. Sim vs market: margin and total point estimates

PROVISIONAL. Model-vs-actual and market-vs-actual MAE/bias (context), then a DIRECT model-vs-market comparison (MAE/bias/corr between the two point estimates themselves).

| n | model_margin_MAE_vs_actual | market_margin_MAE_vs_actual | model_margin_bias | market_margin_bias | model_total_MAE_vs_actual | market_total_MAE_vs_actual | model_total_bias | market_total_bias |
|---|---|---|---|---|---|---|---|---|
| 5383 | 9.2651 | 8.7413 | -0.2154 | -0.1628 | 13.5923 | 12.7297 | -0.9537 | -0.5182 |

| n | margin_MAE(model,market) | margin_bias(model-market) | margin_corr(model,market) | total_MAE(model,market) | total_bias(model-market) | total_corr(model,market) |
|---|---|---|---|---|---|---|
| 5383 | 2.9274 | -0.0526 | 0.9130 | 4.7004 | -0.4561 | 0.7360 |

seed-noise SE of the sim mean at 75 seeds: margin 3.496 pts, total 1.530 pts (extrapolated via the studys c/sqrt(k) law).

## 3. Sim win probability vs de-vigged close probability

**REFUSED**: 75 seeds < 200-seed floor for even a PROVISIONAL per-game probability read. No calibration/reliability/Brier numbers printed.

## 4. Edge buckets: spread, total, moneyline (settled at real -110 / real posted odds)

**REFUSED**: 75 seeds < 2000-seed floor. No ROI or hit-rate numbers printed. Breakdowns (season / month / conference / own-rating quintile) are likewise withheld.

## 5. Leak cross-check: does edge-at-open predict open-to-close movement?

CLAUDE.md: "an edge that beats the close but cannot predict line movement is presumed leaked." Only 2025 has usable `open_spread_home` coverage (`docs/tests/lines_cbbd_validation_2026-09-10.md` addendum A: 0% in 2023-2024, 33.5% in 2025).

n games with an open: 1802; n with a nonzero close-vs-open move: 1086.
corr(edge at open, open-to-close movement) = 0.1530 (PROVISIONAL -- the sim mean feeding `edge_at_open` carries 3.50 pts of seed noise at 75 seeds, which attenuates any correlation read here toward zero).

2x2 (edge sign at open vs movement sign, nonzero moves only):

| edge_at_open | moved away | moved home |
|---|---|---|
| edge<0 (away) | 306 | 222 |
| edge>0 (home) | 223 | 335 |

CLV sign agreement: 0.5902 on 1086 moved lines (gate 0.53: near-coinflip = presumed-leaked signature if the surprise correlation above is also real).

