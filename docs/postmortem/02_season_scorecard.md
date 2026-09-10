# CBB-Monte 2025-26 Season Scorecard (sonnet worker, 2026-09-10)
Scripts: results_load_data.py, results_analysis.py, results_variants.py; outputs results_out.json, results_variants_out.json, master_games.parquet etc. in this folder.

## Bottom line
- Only base `cbb-sims-2026` (frozen model_version cbb-2026-10-15) ran live pregame daily. 5,685 logged, 5,229 graded (92%), Nov 3 2025 - Mar 15 2026, 130/133 days. created_at precedes start_utc, mean lead ~9h. Clean.
- Did NOT beat close on either market.

## Spread
Model MAE 9.74 vs close 9.10. Bias +0.31 (close +0.44).
ATS by |model - close|: <1: 51.6% (n724); >=1: 51.8% (n4505, -1.2% ROI); >=2: 51.9%; >=3: 52.4% (n2986, ~0); >=5: 53.7% (n1712, +48.2u, +2.8% ROI, ~1.1 SD above breakeven).

## Totals (headline problem)
Model MAE 16.47 vs close 13.36. Bias +9.44 (close -0.48).
By predicted tercile: low(126-153) +5.73 / mid +9.15 / high(164-212) +13.82. Bias grows with predicted total.
By month: Nov +8.60, Dec +9.48, Jan +10.74, Feb +9.31, Mar +7.82. Present all season.
O/U vs close >=1pt: n5126, 48.6%, -408.9u (-8.0% ROI). All buckets 48.2-48.6%.
Independently confirmed by comparison_2025-11-03_to_2025-12-16.json: 92.5% Over picks, 48.4%, -8.4% ROI.

## Win prob calibration
Brier 0.2013 vs de-vigged close ML 0.1795. Underconfident at tails:
0.1-0.2 pred 0.160 actual 0.018 (n110); 0.2-0.3 0.261 -> 0.106 (n360); 0.4-0.5 0.452 -> 0.413 (n1421); 0.6-0.7 0.644 -> 0.782 (n693); 0.7-0.8 0.742 -> 0.954 (n280); 0.8-0.9 0.841 -> 0.984 (n63).
=> sim margin distribution far too wide (variance too high), not a "needs Platt scaling" problem.

## CLV
Spread: corr(model-open edge, close move)=0.120, sign agree 58.0% (n3636). Totals: corr 0.361 but sign agree 51.5% (n4052) — artifact of chronic over-bias.

## Variants — NOT valid backtests
boosted_sims: entire Nov3-Dec17 backlog created_at 2025-12-18/19. improved: all files mtime 2026-03-08. pace experiment: 2 days, created a month late.
Side by side on joined games (var/base): boosted spread MAE 12.08/10.51, total MAE 13.90/16.42, total bias -0.66/+8.93, Brier 0.190/0.179. improved: 10.25/10.49, 15.43/16.56, +6.40/+9.07, 0.191/0.179.
improved ML picks (comparison json): 180-647 (21.8%), -254u, 803/827 bets on dogs — broken.

## Picks record
pick_best_bets output exists for one day (2025-11-30, 6-4, +1.45u). tweet_meta 12 dates, no sides. No season-long tracked pick/ROI record exists.

## Pipeline health
daily_run.log last write 2026-06-14, died on save_finals_from_sbr RuntimeError (off-season, no games) from ThreadPoolExecutor with no try/retry. 164 [FAIL] / 411 [DONE]; 82 hard crashes, 81 from save_finals_from_sbr. 31 games backfilled 3-49h late (0.6%).
NCAA tournament sims (162 games) never graded (0 final.json). dk_raw/oddsapi_raw empty.

## Output inventory
cbb-sims-2026: 41,221 files 16GB, per game summary.json (A_win_prob, margin_p25/50/75, total_p25/50/75, odds, market_eval, fair) + final.json (scores, closing) + sims.parquet + priors.json + manifest.json. boosted_sims 1,868; improved 1,768 rows; experiments 431; cbb-tourney-sims 480; bracketology RR 52; espn_lines 185 (2025-11-28..2026-06-14); comparison 1.

## Worker recommendations (PM note: #2 as worded — Platt/isotonic on output — is banned by no-hand-tuning rule; the fix is sim variance at the sub-model level)
1 fix totals bias at source; 2 win prob tails; 3 realistic spread expectations; 4 CI gate created_at < start_utc for every backtest; 5 harden finals scraper; 6 continuous pick/CLV tracking from day one.
