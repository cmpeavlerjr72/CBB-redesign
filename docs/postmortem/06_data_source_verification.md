# NCAA MBB data source verification (worker report, fetched 2026-09-10)

## Verified usable
- **sportsdataverse/hoopR-mbb-data (GitHub, CC BY 4.0, daily cron).** Files committed on `main`, not releases:
  `https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/<dataset>/parquet/<stem>_<season>.parquet`
  - pbp: play_by_play_{season} 2003, 2006-2026 (2022: 54MB; 2025: 63MB). Rows: 2021 1.30M, 2022 1.86M, 2023 1.96M, 2024 2.0M, 2025 2.19M, 2026 2.92M. Columns incl. athlete_id_1/2, shooting_play, coordinate_x/y, period/clock, AND embedded ESPN market fields (game_spread, home_team_spread, home_win_prob, pregame_home_prob) — strip from features (leak) but probe as a historical line proxy.
  - player_box: 2003-2026 (PM verified 2025: 207,613 rows x 55 cols, 6,290 games, 2024-11-04..2025-04-07; minutes, starter, did_not_play, active, full shooting/reb/ast/stl/blk/to/fouls).
  - team_box, schedules (2003-2027), shots (coords), player_core (bio/position per season).
  - rosters / game_rosters: 2025 and 2026 only.
- **ESPN site.api / core.api JSON** (undocumented). summary has plays[] (~458/game), boxscore.players (minutes), pickcenter (DraftKings spread/total/ML, live snapshot only). teams/{id}/roster current season. teams/{id}/injuries exists (empty off-season). www.espn.com robots.txt disallows AI crawlers; api hosts have none. Use lightly in-season; hoopR mirrors the bulk.
- **The Odds API**: NCAAB spread/total/ML historical from mid-2020; player points & rebounds props (assists/threes unconfirmed) historical from May 2023. $30/mo 20K credits, $59/mo 100K. Free tier impractical for backtests.
- **Injuries**: Covers.com NCAAB injuries page (open robots), RotoWire (needs browser headers), ESPN endpoint. Unstructured.
- **Historical closing lines 2021-2025**: sportsbookreviewsonline per-season xlsx (URL needs manual check); Scottfree LLC CSV $119 (85,033 rows, open+close spread/ML/total). Kaggle leads unverified.
- **Point-in-time ratings**: on disk already — KenPom weekly 2022-2025 + daily 2026 (153 snaps). Wayback CDX has 288 kenpom.com and 417 barttorvik trank.php snapshots 2021-2026 (content unverified). Warren Nolan open but no history.
- **Season**: 2026-27 runs Nov 1 2026 - Mar 14 2027; most D-I openers Nov 2-3, 2026.

## Do NOT use in the pipeline
- Sports-Reference CBB: data_use.html bans ML/AI use of content and bots; 20 req/min. Last year's gamelog scraper is retired.
- barttorvik.com: robots.txt Disallow / for ClaudeBot & anthropic-ai; trank-time-machine, results, playerstat, *.json disallowed for all.
- masseyratings.com: 403 + robots Disallow for AI crawlers.
- PrizePicks (DataDome 403), Underdog (no API), DraftKings (no public API).
