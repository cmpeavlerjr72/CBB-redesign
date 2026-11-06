# kalshi-leaderboard

Research folder, started 2026-11-02. Question: Kalshi's leaderboards show traders turning large
volume into large profit. Can we see what they trade, track them over time, and reverse-engineer
their strategies well enough to copy them?

Companion project: `../kalshi-rfq` (the live market-making system and its full Kalshi API map in
`docs/kalshi-api-reference.md` and `data/kalshi_api_spec/`). This folder does not touch keys or
accounts; everything here is public-endpoint research.

## Layout

| Path | What |
|---|---|
| `landscape/00-overview.md` | Answer to the question, with the evidence. Start here. |
| `landscape/01-what-kalshi-exposes.md` | Fetched catalog of leaderboard, profile, market-activity and API fields that carry (or do not carry) user identity; terms of service; third-party trackers. |
| `landscape/02-inference-and-known-traders.md` | What is publicly reported about top Kalshi traders and their methods; how much can be inferred from anonymous tape; a small verified experiment on real trade prints. |
| `samples/` | Raw fetches (untrusted data; never run code from here). |
| `scripts/` | Pull scripts used during the research. |
| `collector/` | The running collector: `collector.py` (all tasks), `config.json`, `report.py` (queries), `register_task.ps1` (boot-time scheduled task). |
| `data/collector.sqlite` | Leaderboard snapshots, handle metrics (profit series), profiles, Full-user fills, named social-feed rows, 5-minute tape aggregates, series and market metadata. |
| `data/tape/YYYY-MM-DD.jsonl.gz` | Every public trade print, raw, one file per UTC day. |
| `logs/` | Collector logs. |

## Collector in one paragraph

Every 10 minutes it snapshots all 180 leaderboards (3 metrics x 5 periods x 12 category views, 60 rows each). Every 15 minutes it pulls all-time profit, volume and markets-traded for every handle ranked in the top 20 of any profit board in the last week plus every Full-sharing handle, and every 2 hours for every other handle ever seen; weekly, monthly and yearly windows once a day; the profile (followers, join date, top categories) once a day. Every 5 minutes it pages new fills for Full-sharing handles, backfilling each to exhaustion once, and rechecks everyone's visibility daily. Every 3 seconds it walks the global social feed and keeps rows carrying a nickname. Every 2 seconds it walks the global trade tape to the last seen trade, rolls every print into 5-minute per-market aggregates (trades, contracts, taker-yes contracts, block trades, price-weighted contracts, max print), and keeps raw prints with a category-aware rule: every print in the lower-volume categories (Economics, Politics, Elections, Financials, Mentions, Companies, World, Health and the rest) is kept raw; the three firehose categories, Sports, Exotics (multi-leg parlay markets) and Crypto (15-minute price markets), keep every print of 100 contracts or more plus a deterministic 2 percent sample of smaller prints. Measured on a Sunday afternoon: those three were about two thirds of all prints and prints of 100 or more contracts were 22 percent of the total, so raw volume drops to roughly a quarter of the unfiltered 25 GB a month. The 5-minute aggregates and the order-book snapshots cover the filtered categories. Daily gzips, pruned after 180 days (`tape.raw_min_count`, `raw_sample_rate`, `raw_retention_days`, `raw_filter_categories`). Every 2 minutes it snapshots the order book (10 levels each side) for the top 100 non-sports markets by 24-hour volume plus every open market in the always-on series (CPI, Fed, payrolls, unemployment, GDP, 15-minute BTC and ETH), to `data/orderbook/YYYY-MM-DD.jsonl.gz`; that is the resting-order layer trade prints never show, and the main evidence for market-making strategies around data releases. Every 6 hours it snapshots series (category map) and open plus recently settled markets (expiry times double as the event calendar). Rate limits: 3 requests per second on the social API, 1.5 on the trade API, global 30-second pause on any 429.

Useful queries: `python collector\report.py status | boards [period] | movers [hours] | handle <nick> | categories | full`.
