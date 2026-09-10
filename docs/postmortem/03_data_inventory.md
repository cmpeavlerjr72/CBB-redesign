# CBB-Monte Data Sources & Pipeline Inventory (worker report, 2026-09-10)

Daily pipeline graph (daily_run.py): enrich_massey_schedule_with_ids -> scrape_team_gamelogs_all -> build_pregame_averages; pull_kenpom_table; espn_cbb_lines (today); save_finals_from_sbr (yesterday); -> hf_publish (runs sims + uploads). dk_pull / oddsapi_pull NOT in the graph; their output folders are empty.
.env keys: HF_TOKEN, ODDS_API_KEY, OPENAI_API_KEY, XAI_API_KEY. KenPom/SR/Massey/ESPN/SBR/DK are unauthenticated scrapes.

## Sources
- KenPom: scrape kenpom.com/index.php?y=season, requests->Playwright fallback. Daily -> data/kenpom/{season}/{date}_kenpom.csv. 153 daily snaps 2026 (Oct25-Apr26), 45 for 2027 preseason. History = weekly Monday snapshots only (20XX_kenpom.csv, 19-20 rows/season, 2022-2025). Fragile "biggest table" parse.
- Sports-Reference team gamelogs: per-game TEAM box (shooting splits both sides). 2s/request, 3 threads. Hard-coded slug alias table duplicated 3x with drift. 6 seasons on disk: 2021 (347 teams) .. 2026 (365).
- SR player box scores: scrape_cbb_sr_boxscores_with_ids.py fully written (per-player per-game + player_id registry + roster history -> data/boxscores/*.parquet). ZERO output files exist. Never run to completion.
- Massey: headerless Matlab Games CSV via cloudscraper; used as team-ID crosswalk. 2026_massey_schedule_enriched.csv present.
- ESPN team metadata: espn_teams.json one-off snapshot.
- ESPN lines: scoreboard->summary pickcenter (ML/spread/total), 16 workers. 185 files, 183 dates 2025-11-28 -> 2026-06-14 (post-March stale).
- SBR: consensus spread/total/ML via headless Playwright from Next.js JSON. Pregame {date}_sbr.csv and closing {date}_sbr.post.csv (+score). 2,602 files, 130 dates 2025-11-03 -> 2026-03-15. backfill_finals.py hard-codes 14 failed dates.
- DraftKings / OddsAPI: coded, never produced persisted output.
- HuggingFace: sink. mvpeav/cbb-sims-2026.

## Historical depth
6 seasons team box (2021-2026). 4 complete backtest seasons 2022-2025 with games + weekly KenPom + gamelogs + training rows (all_training_stats_rows_2022_2025.csv = 45,436 rows; ~11k/season). Betting lines: ONE season only (2025-26). No PBP. No player box. No historical odds 2022-2025.

## Player data
Effectively none. All modeling is team-level shot volume/efficiency (y_3pa/y_2pa/y_fta/y_3p_pct/y_2p_pct/y_ft_pct). No rosters, injuries, minutes, lineups.

## Lines
Spread/total/ML only. No 1H, no props. Team totals synthesized from total+spread (backfill_team_totals.py). SBR consensus median + raw per-book. ESPN prefers Caesars/ESPN BET w/ inconsistent fallback. Finals-to-lines: canonicalized name lookups with hand alias dicts; update_lines_by_date.py has best matcher (ID-based via espn_to_kp_from_massy_matches.csv, then name canon, then opponent disambiguation); save_finals_from_sbr.py silently drops unmatched.

## Name/ID mapping
{season}_teams.csv, massy_ids.csv (2022-2026 stable IDs), massy_to_kp.csv (368 rows Massey->SR->KenPom), espn_to_kp_from_massy_matches.csv + manual override CSV (0 unmatched). 3+ drifted copies of SR slug alias table (sports_reference_aliases.py, scrape_team_gamelogs_all.py, merge_massy_names.py). Bugs: circular typo mapping massachuesetts<->umass; duplicate dict key "north carolina state" silently overwrites.

## Processed / sim_input
data/sim_input/{season}/{date}_sim_rows.csv (131 daily files): one row per team-side per game, 104 cols — KenPom ratings, pace, ytd/roll5 avg+std+has_data for 6 shot-rate/efficiency stats x team/opp-allowed, y_* targets. data/processed/{season}_training_stats_rows.csv same schema (~11-11.5k rows/season). data/pregameaverages/.../{slug}.csv (2,156 files) leak-free pregame-shifted rolling features. data/models/boosted/*.pkl LightGBM/RF/XGBoost variants for shot-rate targets.

## Gaps for possession/player rebuild
Missing: PBP (ESPN summary `plays` array = unverified lead), player box/minutes (scraper exists, unused — highest-leverage fix), rosters/depth (ESPN roster endpoint unverified), injury reports (no source), shot location (likely paid), player prop lines (never scraped; unknown if SBR/DK carry NCAAB props), multi-season odds history (paid archive needed for 2022-2025). Consolidate alias tables into one crosswalk on the stable numeric ID before building anything.
