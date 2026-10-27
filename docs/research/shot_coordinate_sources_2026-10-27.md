# Shot-coordinate sources survey (2026-10-27)

Worker: Sonnet. Scope: where else can NCAA men's shot x/y be obtained, legally and free. On-disk coordinate quality is being audited by another worker and is not redone here. Scripts: `scripts/pull_hoopr_hist_pbp_v1.py`, `scripts/diag_cbbd_shot_coverage_v1.py`. Raw outputs: `data/raw/hoopr_hist/coverage.json` (gitignored), `docs/tests/cbbd_shot_coverage_2026-10-27.json` (local, not committed per brief).

## Facts the PM should know (no recommendations)

1. hoopR-mbb-data `mbb/pbp/parquet` holds 2003 and 2006-2026. Pre-2022 files (2003, 2006-2021, 17 files, about 0.5 GB) are downloaded to `data/raw/hoopr_hist/pbp/` with a manifest.
2. hoopR coordinate columns (`coordinate_x/_y`, `_raw`) exist in every historical file, but are EMPTY before the 2014 season (2003, 2006-2013: 0.00% of FGA located). From 2014 to 2021 they are populated for only 10.8-20.8% of FGA, concentrated in a minority of games (in located games about 92% of FGA carry coordinates; 579-1,204 games per season have any).
3. The historical hoopR coordinate frame differs from our 2022+ CBBD frame: hoopR `coordinate_x` is in feet centred on the court (about -47 to +47), `coordinate_x_raw` is 0-50 (feet). CBBD uses tenths of a foot, 0-940 x 0-500. A frame conversion would be required.
4. The hoopR `shots` release (`mbb/shots/parquet`, 2003-2026) is derived from the same pbp (`source_endpoint` = "derived from espn_mbb pbp"), so it adds no coordinates.
5. CBBD `/plays/game/{id}` carries `shotInfo.location.x/y` for later seasons than hoopR hist: in 20-game samples, 0% located for 2013-2018, 28.5% (5 of 20 games) for 2019, 71.4% (16/20) for 2020, 65.1% (11/20) for 2021. Small samples, indicative only. In a follow-up look at the 2020 sample, most located CBBD games had zero coordinates in hoopR hist, i.e. CBBD appears to have a different or fuller location source for 2019-21 than hoopR.
6. On-disk 2022-2025, CBBD and hoopR coordinates are complementary at the game level (shared games, located = more than 50% of FGA with coordinates). CBBD: 79.5/84.0/87.6/99.4% of games (2022-25); hoopR (non-null, non-zero): 12.0/8.8/6.2/27.1%; union 91.5/92.7/93.9/99.9%. Games located in hoopR only: 699/533/384/33; in CBBD only: 4,633/5,131/5,389/4,468. So hoopR could add roughly 12 points of located games in 2022, 9 in 2023, 6 in 2024, almost none in 2025 (subject to frame conversion and to the quality audit; quick desk check, not a validated join).
7. ESPN is the origin of both hoopR and (apparently) the gaps. A one-request probe per game of ESPN's public `site.api.espn.com .../mens-college-basketball/summary?event=<id>` shows games unlocated in hoopR are also unlocated at the origin today (4 of 4 unlocated games probed, 2016, 2019, 2021, 2024: 0 coordinates; 1 located 2016 game: 118 of 118). The gap is not a hoopR scraping artefact and re-fetching does not look like it would fill it (n=5 probes, indicative).
8. NCAA's own site: the bigballR author (page dated 2026-09-30) reports the NCAA added Terms of Service language in early September 2026 prohibiting robots, scraping and harvesting, and stopped updating the package. Anything via stats.ncaa.org is therefore off-limits.
9. No free, open, licensed NCAA men's shot x/y dataset on Kaggle or Hugging Face turned up in standard or extended search. The one candidate with real x/y is the Google BigQuery public dataset `bigquery-public-data.ncaa_basketball` (table `mbb_pbp_sr`, Sportradar-sourced, columns `event_coord_x/y`, reported start 2013); it needs a GCP account to query and its licence and exact season coverage were NOT verified (listing page not readable by my tools).
10. The CBBD docs readable by my tools (getting-started) document only `/games`; no statement of shot-coordinate coverage and no dedicated shots endpoint were found (the OpenAPI JSON was not reachable at usual paths). The Plays API carries `shot_info.location` (10 pts per foot, per the CBBD blog).
11. Season labels follow the file name (2021 = 2020-21).

## Summary table

| Source | Type | Seasons | Access | Licence / terms | Cost | On do-not-scrape list | Verdict |
|---|---|---|---|---|---|---|---|
| hoopR-mbb-data pbp (ESPN-derived) | x/y, feet | coords 2014-2026 only; 10.8-20.8% of FGA 2014-21 | raw GitHub parquet (already our primary) | GitHub tags licence "Other"/NOASSERTION; CC BY 4.0 per CLAUDE.md | free | no | USABLE WITH CAVEAT: sparse pre-2022, different frame |
| hoopR-mbb-data shots release | x/y | 2003-2026 | GitHub / release tag | same | free | no | NOT USABLE as new info: derived from the same pbp |
| CBBD API plays (`shotInfo.location`) | x/y, tenths ft | sample: 2019 28%, 2020 71%, 2021 65%, 2013-18 0% (20 games each) | REST, existing key | CBBD API terms (not re-read here) | free tier (existing key) | no | USABLE WITH CAVEAT: needs larger sample and full-season pull to confirm 2019-21; call budget |
| ESPN site.api.espn.com summary | x/y (origin of hoopR) | pre-2022 served (2016, 2019, 2021 tested) | undocumented public JSON | no published terms for these endpoints found | free | no | USABLE WITH CAVEAT: nothing beyond hoopR (unlocated games stay unlocated); terms unclear |
| NCAA LiveStats / stats.ncaa.org | shot charts exist internally (Genius Sports) | 2018-19 onward capture | web pages / feeds | ToS (Sept 2026) bans scraping | n/a | not listed, but ToS forbids | NOT USABLE |
| bigballR `get_shot_locations` | x/y, rim distance | stats.ncaa.org games | R pkg scraping NCAA | MIT code, data ToS bans scraping; author stopped updates | free | n/a | NOT USABLE |
| StatBroadcast XML | live stats XML feeds | per school | school athletics sites | no shot x/y schema found; per-school terms | free | n/a | NOT USABLE (no evidence of x/y) |
| hoopR `ncaa_mbb_pbp` / `espn_mbb_pbp` | wrappers over NCAA and ESPN | n/a | R package | NCAA route inherits NCAA ToS | free | n/a | espn_mbb_pbp = same ESPN data (USABLE WITH CAVEAT); ncaa route NOT USABLE |
| wehoop / cbbdata | women's / Torvik-backed | n/a | R pkgs | wehoop is women's; cbbdata is barttorvik-linked | free | cbbdata relates to barttorvik | NOT USABLE |
| gamezoneR (STATS LLC GameZone) | x/y (`loc_x`) | 2017-18 onward | scrapes GameZone | licence not stated on page; scraping a vendor site | free | not listed | NOT USABLE without terms review |
| BigQuery public `ncaa_basketball.mbb_pbp_sr` | x/y (Sportradar) | reported from 2013; unverified | BigQuery SQL, GCP account | marketplace licence NOT verified | free query tier | no | USABLE WITH CAVEAT: unverified; needs schema/licence check by someone with GCP access |
| Kaggle (March Mania, others) | box and aggregate; no NCAA shot x/y found | n/a | download | per-competition | free | no | NOT USABLE (nothing found) |
| Hugging Face | no NCAA shot dataset found | n/a | n/a | n/a | n/a | no | NOT USABLE (nothing found) |
| Bart Torvik | shot data on site | n/a | web | robots disallow AI crawlers | n/a | YES | NOT USABLE (listed only) |
| Synergy / Sportradar | tracked or charted shots | multi-year | licensed API | commercial | custom quote, no public price | no | NOT USABLE (paid) |
| ShotQuality | charted xP | multi-year | licensed | commercial | no public price | no | NOT USABLE (paid) |
| Stats Perform | commercial | n/a | licensed | commercial | no public price | no | NOT USABLE (paid) |
| Second Spectrum | optical tracking | NBA mostly | licensed | commercial | no public price | no | NOT USABLE (paid) |
| DunkAPI | claims college PBP; shot x/y stated for NBA only | n/a | API, 500 free credits/month | unknown | free tier; paid unknown | no | NOT USABLE until a free college x/y claim is verified |
| Apify NCAA scraper | scraper of NCAA/ESPN pbp | since 2006 | paid actor | scraper; ToS risk | paid | n/a | NOT USABLE |
| Academic / hand-labelled sets | e.g. about 8,000 shots from about 60 games | tiny | unpublished | n/a | n/a | no | NOT USABLE (too small) |

Searches: standard mode, plus extended for BigQuery, Kaggle and open data. Nothing newer than the NCAA ToS change (Sept 2026) and the hoopR shots release (2026-08-12) turned up.

## hoopR historical coverage (downloaded, read with `python -I`)

FGA = `shooting_play` true and neither `type_text` nor `text` contains "Free Throw". Located = coordinate non-null and non-zero for both x and y. Coordinate columns exist in all files.

| season | rows | games | FGA | games with FGA | located % (coordinate_x/y) | games with any located | located % (raw) |
|---|---|---|---|---|---|---|---|
| 2003 | 5,182 | 18 | 2,104 | 18 | 0.00 | 0 | 0.00 |
| 2006 | 314,119 | 1,104 | 122,889 | 1,104 | 0.00 | 0 | 0.00 |
| 2007 | 457,893 | 1,623 | 179,997 | 1,623 | 0.00 | 0 | 0.00 |
| 2008 | 798,658 | 2,790 | 310,014 | 2,790 | 0.00 | 0 | 0.00 |
| 2009 | 1,141,127 | 3,650 | 407,184 | 3,650 | 0.00 | 0 | 0.00 |
| 2010 | 1,271,470 | 3,836 | 429,946 | 3,836 | 0.00 | 0 | 0.00 |
| 2011 | 1,131,683 | 3,469 | 383,334 | 3,469 | 0.00 | 0 | 0.00 |
| 2012 | 1,118,161 | 3,448 | 378,354 | 3,448 | 0.00 | 0 | 0.00 |
| 2013 | 1,257,727 | 3,895 | 427,291 | 3,895 | 0.00 | 0 | 0.00 |
| 2014 | 1,670,308 | 5,084 | 560,608 | 5,084 | 20.84 | 1,151 | 21.27 |
| 2015 | 1,737,302 | 5,280 | 588,564 | 5,280 | 20.59 | 1,200 | 20.99 |
| 2016 | 1,803,531 | 5,390 | 625,276 | 5,390 | 20.66 | 1,204 | 20.94 |
| 2017 | 1,818,853 | 5,479 | 635,456 | 5,479 | 17.07 | 1,009 | 17.22 |
| 2018 | 1,827,395 | 5,645 | 642,064 | 5,569 | 15.83 | 940 | 15.81 |
| 2019 | 1,787,332 | 5,474 | 633,445 | 5,474 | 14.21 | 841 | 14.51 |
| 2020 | 1,746,980 | 5,394 | 620,024 | 5,394 | 10.80 | 629 | 10.95 |
| 2021 | 1,296,252 | 4,015 | 463,370 | 4,015 | 13.48 | 579 | 13.68 |

Files: `data/raw/hoopr_hist/pbp/play_by_play_{2003,2006..2021}.parquet`; manifest `data/raw/hoopr_hist/manifest.json` (url, bytes, sha256, downloaded_at). Not synced to HF. About 22% of games are located in 2014 falling to 14% in 2021; within a located game about 92% of FGA have coordinates. Selection into the located set is unexplained (plausibly games ESPN charted, e.g. higher-profile games; not tested), so it is probably non-random.

## CBBD sample (20 random final games per season, `/plays/game/{id}`)

Shot plays = shooting plays excluding free throws (by playType, playText, shotInfo.range). Located = `shotInfo.location.x` and `.y` non-null. About 190 calls of a 400 cap.

| season | games | shot plays | located | located % | games with any located |
|---|---|---|---|---|---|
| 2013 | 20 | 1,745 | 0 | 0.00 | 0 |
| 2014 | 20 | 2,121 | 0 | 0.00 | 0 |
| 2015 | 20 | 2,109 | 0 | 0.00 | 0 |
| 2016 | 20 | 2,101 | 0 | 0.00 | 0 |
| 2017 | 20 | 1,975 | 0 | 0.00 | 0 |
| 2018 | 20 | 2,275 | 0 | 0.00 | 0 |
| 2019 | 20 | 1,970 | 561 | 28.48 | 5 |
| 2020 | 20 | 2,210 | 1,577 | 71.36 | 16 |
| 2021 | 20 | 1,500 | 976 | 65.07 | 11 |

Game-level sampling error is large (20 games of a whole-game binary: SE roughly 10 points). In the 2020 sample, CBBD x ranged 18.8-930.6 and y 10-490 (tenths of feet, same frame as on-disk 2022+). hoopR hist locates 2014-2018 games where CBBD located none in the sample, so the two feeds may complement each other for 2014-2018 (not tested beyond these samples).

## Imputation angle (desk reasoning, no fitting; no recommendation)

Reference (`docs/tests/fg_make_pae_coverage_2026-10-25.json`): about 1,800 FGA per team-season (672k-724k FGA over 369-378 teams). Per-team located share, median / p10 / min: 2022 89.5% / 55.9% / 0%; 2023 92.0 / 65.0 / 0; 2024 93.7 / 78.9 / 0; 2025 99.2 / 96.8 / 88.3. Share of teams below 40% located: 6.0 / 4.3 / 2.1 / 0.0%. Per game: share with zero located 5.1 / 3.6 / 2.6 / 0.2%; fully located 56 / 59 / 57 / 75%. So a typical 2022-24 team-season has about 1,600 located attempts, a p10 team-season in 2022 about 1,000, and teams under 40% located have under about 700.

(a) Estimating each team's as-of location distribution. This needs located counts per team before the as-of date; the located share per se matters less. The estimate is a multinomial over K location cells. Illustrative SE of a cell share p = 0.2 is sqrt(p(1-p)/n): 0.013 at n = 1,000, 0.023 at n = 300, 0.04 at n = 100. Early in a season n is a fraction of the season total (about 60 located shots after 10 games at roughly 6 per game per team-class), so shrinkage toward a league or prior-season team distribution carries most of the weight early. With a conjugate weight w = n/(n+k), the pseudo-count k and the between-team variance of cell shares (not measured here) decide how many located shots let the data dominate. If missingness is within-team random (whole games missing, not selected shots), the located subset is an unbiased draw and only effective n shrinks. The risk is composition: location is missing mostly by whole game (2022 per-game p10 is 9.7% located, median 99%), so a team whose located games skew to televised, home or high-profile games gives a biased as-of mix. Home/away split is checkable (`by_season_site` in the coverage file); opponent or venue selection is not.

(b) Training location-conditioned make / foul / rebound models. These need located shots with outcomes, and the question is whether missingness is informative. The coverage file shows located vs unlocated make rates differ: 2022 rim 58.4% located vs 61.4% unlocated; 2025 rim 58.1% vs 81.5% (n = 5,510 unlocated, plausibly a different shot population such as putbacks or dunks without a coordinate) and jump2 38.8% vs 58.6%. Inside partially located games the gap is small (2025 rim 60.9 vs 59.8, jump2 38.2 vs 37.7, 3PT 33.8 vs 34.2), which suggests part of the whole-game difference is selection into games and part is shot type. A model trained only on located shots inherits that selection; a model that also uses unlocated shots needs a location-free fallback (shot class only) and a stated rule for what is imputed. Foul and rebound outcomes need the following event linked to the located shot, so coverage of the shot, not the rebound, is the constraint.

Coverage arithmetic only: with about 1,600 located attempts per full-season team, a cell with p = 0.1 has SE 0.0075; at 700 located (about 40% coverage) SE is 0.011; at 200 it is 0.021. Per-game coverage, not the per-team share, determines whether the located subset is representative. For the historical hoopR seasons (2014-2021) the league-wide located counts are 65k-130k FGA per season, i.e. on the order of 200-400 per team if spread evenly over about 350 teams, and fewer for teams ESPN rarely charts; per-team counts for those seasons were not computed, so the spread across teams is unknown.
