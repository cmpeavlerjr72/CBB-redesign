# march-madness fork + CBB-Monte prop/publishing scan (haiku worker, 2026-09-10)

## calibration_layer.py
POST-HOC adjustment: 9-point calibration curve from 5,216 full-season games, maps raw sim win prob -> empirical win rate via linear interpolation (e.g. 57.5% -> 64.3%; model underconfident on favorites). Applied to tournament round odds, +/-10-20% shifts. This is exactly a compensating layer over a sim defect.

## calibration_analysis.py
Season audit: SU accuracy by confidence tier, Brier ~0.17, ATS record, daily P&L, margin MAE 3-5, total MAE 4-7, monthly breakdowns.

## compare_tourney.py
Production vs tournament-specific model on B10/SEC conf tourney games (Mar 11-15 2026): margin MAE 2.9 vs 3.2, total MAE 3.8 vs 4.1 (tiny sample); ~60% SU R1 declining to 50% QF.

## prizepicks_analysis.py / prizepicks_evaluate.py
100K bracket sims -> expected tournament games per team; ESPN per-game season stats (27 players, 16 teams) x expected games = projection; edge% vs PrizePicks lines from Book1.xlsx. 30 props ranked. No player-level simulation; PPG x games.

## analyze_underdog_ml(_actual).py
2021-2025 tournament ML backtest (335 games). Claimed 5v12 underdog +14.7% edge, 8v9 +7%. Small samples; treat as anecdote.

## CBB-Monte publishing stack
grok_context.py -> generate_why_grok.py (Grok API) -> hybrid_post.py (Grok facts + GPT-5 tweet) -> post_daily_picks.py (EV filters 0.04 favs / 0.06 dogs) -> render_tweet_cards.py (Playwright PNG). Publishes to X.
boosted_training.py: XGBoost on team shooting stats (3PA,3P%,2PA,2P%,FTA,FT%).
