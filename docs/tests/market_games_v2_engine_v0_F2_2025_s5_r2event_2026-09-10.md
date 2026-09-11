# Market scorecard v2 (accepted lines source) -- engine_v0/F2_2025_s5_r2event (season 2025)

Generated 2026-09-10 by `scripts/grade_market_games_v2.py`. Results: `results/engine_v0/F2_2025_s5_r2event`. Lines: `data\processed\lines\lines_close_v1.parquet` (provider `ESPN BET`, 0 duplicate (game_id) rows dropped). See the module docstring for the four differences from `scripts/grade_market_games.py` (v1, unmodified).

## 1. Seed-count and backtest-honesty guard

**EVERY TABLE IN THIS REPORT IS PROVISIONAL.** This run has **5 seeds**, below the `docs/tests/engine_seed_count_2026-09-10.md` floor of 200 seeds for even the slate-level gate report.

**Section 3 (probability calibration) is REFUSED below 200 seeds** -- this run has **5**, below that floor. No calibration/reliability numbers printed.

**ROI, Brier and edge-bucket hit-rate numbers are REFUSED below 2000 seeds.** `docs/tests/engine_seed_count_2026-09-10.md` measured that floor for a per-game ROI/Brier read (game-level margin-mean SE is still 14.850 pts and win-prob SE 0.2164 at 5 seeds; at 5 seeds a sim win probability can only take the values {0, 0.2, 0.4, 0.6, 0.8, 1.0}). This run has **5** seeds, below that threshold, so section 3's Brier line and all of section 4 (edge buckets/ROI) print the seed count and threshold ONLY -- no numbers.

backtest: run created post hoc by construction; artifact `max_train_date < tipoff` asserted on event.first: 5710/5710 rows pass; event.cont: 5710/5710 rows pass. **artifact dates not recorded** for 9 other sub-model artifact path(s) (event.team_block, clock, fg_make.FGA_rim, fg_make.FGA_jump2, fg_make.FGA_3, free_throw, rebound, usage, rotation) -- not checked, not fabricated, not assumed to pass.

### Second-source truth accounting (game_finals_v2 only)

| n_schedule_games | n_no_finals_row | n_unresolved_excluded | n_truth_games | finals_source_counts |
|---|---|---|---|---|
| 5710 | 0 | 0 | 5710 | {"hoopr": 5709, "cbbd": 1} |

n games with a usable `ESPN BET` close line, after joining truth x sim x lines: 5383 (of 5710 truth games).

## 2. Sim vs market: margin and total point estimates

PROVISIONAL. Model-vs-actual and market-vs-actual MAE/bias (context), then a DIRECT model-vs-market comparison (MAE/bias/corr between the two point estimates themselves).

| n | model_margin_MAE_vs_actual | market_margin_MAE_vs_actual | model_margin_bias | market_margin_bias | model_total_MAE_vs_actual | market_total_MAE_vs_actual | model_total_bias | market_total_bias |
|---|---|---|---|---|---|---|---|---|
| 5383 | 16.0325 | 8.7413 | -1.8177 | -0.1628 | 14.4920 | 12.7297 | -0.1124 | -0.5182 |

| n | margin_MAE(model,market) | margin_bias(model-market) | margin_corr(model,market) | total_MAE(model,market) | total_bias(model-market) | total_corr(model,market) |
|---|---|---|---|---|---|---|
| 5383 | 13.4312 | -1.6549 | 0.3292 | 6.6494 | 0.3844 | 0.6685 |

seed-noise SE of the sim mean at 5 seeds: margin 14.850 pts, total 6.434 pts (exact study rung).

## 3. Sim win probability vs de-vigged close probability

**REFUSED**: 5 seeds < 200-seed floor for even a PROVISIONAL per-game probability read. No calibration/reliability/Brier numbers printed.

## 4. Edge buckets: spread, total, moneyline (settled at real -110 / real posted odds)

**REFUSED**: 5 seeds < 2000-seed floor. No ROI or hit-rate numbers printed. Breakdowns (season / month / conference / own-rating quintile) are likewise withheld.

## 5. Leak cross-check: does edge-at-open predict open-to-close movement?

CLAUDE.md: "an edge that beats the close but cannot predict line movement is presumed leaked." Only 2025 has usable `open_spread_home` coverage (`docs/tests/lines_cbbd_validation_2026-09-10.md` addendum A: 0% in 2023-2024, 33.5% in 2025).

n games with an open: 1802; n with a nonzero close-vs-open move: 1086.
corr(edge at open, open-to-close movement) = 0.0674 (PROVISIONAL -- the sim mean feeding `edge_at_open` carries 14.85 pts of seed noise at 5 seeds, which attenuates any correlation read here toward zero).

2x2 (edge sign at open vs movement sign, nonzero moves only):

| edge_at_open | moved away | moved home |
|---|---|---|
| edge<0 (away) | 288 | 258 |
| edge>0 (home) | 241 | 299 |

CLV sign agreement: 0.5405 on 1086 moved lines (gate 0.53: near-coinflip = presumed-leaked signature if the surprise correlation above is also real).

