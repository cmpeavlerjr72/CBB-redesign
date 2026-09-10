# CBBD data audit -- 2026-09-10

Source: CollegeBasketballData API (`https://api.collegebasketballdata.com`). Pulled by `scripts/pull_cbbd.py` into `data/raw/cbbd/`. This report is generated read-only by `scripts/diag_cbbd_audit.py`.

## 1. Lines (`/lines`)

Per-season line coverage:

|   season |   games |   games_with_line | pct_with_line   | spread_nonnull_%   | ou_nonnull_%   | ml_home_nonnull_%   | ml_away_nonnull_%   | spreadOpen_nonnull_%   | ouOpen_nonnull_%   | date_min   | date_max   |
|---------:|--------:|------------------:|:----------------|:-------------------|:---------------|:--------------------|:--------------------|:-----------------------|:-------------------|:-----------|:-----------|
|     2013 |    5838 |              5395 | 92.4%           | 85.3%              | 39.3%          | 56.1%               | 56.6%               | 0.0%                   | 0.0%               | 2012-11-09 | 2013-04-09 |
|     2014 |    5995 |              5541 | 92.4%           | 85.5%              | 54.5%          | 54.9%               | 54.9%               | 0.0%                   | 0.0%               | 2013-11-08 | 2014-04-08 |
|     2015 |    5949 |              5508 | 92.6%           | 87.1%              | 36.7%          | 52.4%               | 52.4%               | 0.0%                   | 0.0%               | 2014-11-14 | 2015-04-07 |
|     2016 |    5991 |              5532 | 92.3%           | 86.9%              | 59.9%          | 53.7%               | 53.7%               | 0.0%                   | 0.0%               | 2015-11-13 | 2016-04-05 |
|     2017 |    5982 |              5510 | 92.1%           | 87.1%              | 60.9%          | 54.6%               | 54.6%               | 0.0%                   | 0.0%               | 2016-11-11 | 2017-04-04 |
|     2018 |    5987 |              5518 | 92.2%           | 88.3%              | 88.5%          | 80.5%               | 80.5%               | 0.0%                   | 0.0%               | 2017-11-10 | 2018-04-03 |
|     2019 |    6066 |              5612 | 92.5%           | 98.5%              | 99.6%          | 84.9%               | 84.9%               | 0.0%                   | 0.0%               | 2018-11-06 | 2019-04-09 |
|     2020 |    5856 |              5390 | 92.0%           | 99.4%              | 100.0%         | 90.1%               | 90.1%               | 0.0%                   | 0.0%               | 2019-11-05 | 2020-03-14 |
|     2021 |    5282 |              4629 | 87.6%           | 96.0%              | 96.2%          | 95.1%               | 95.1%               | 0.0%                   | 0.0%               | 2020-11-25 | 2021-04-06 |
|     2022 |    6387 |              5773 | 90.4%           | 97.6%              | 98.2%          | 97.8%               | 97.8%               | 0.0%                   | 0.0%               | 2021-11-09 | 2022-04-05 |
|     2023 |    6261 |              5853 | 93.5%           | 99.1%              | 99.1%          | 98.9%               | 98.9%               | 0.0%                   | 0.0%               | 2022-11-07 | 2023-04-04 |
|     2024 |    6249 |              5311 | 85.0%           | 100.0%             | 100.0%         | 99.6%               | 99.6%               | 0.0%                   | 0.0%               | 2023-11-06 | 2024-04-09 |
|     2025 |    6299 |              5440 | 86.4%           | 100.0%             | 100.0%         | 97.2%               | 97.2%               | 33.3%                  | 32.1%              | 2024-11-04 | 2025-04-08 |
|     2026 |    6317 |              5754 | 91.1%           | 99.8%              | 99.8%          | 65.9%               | 66.0%               | 98.7%                  | 98.1%              | 2025-11-03 | 2026-04-07 |

Games-per-provider (distinct gameId with a non-null line row), by season:

- **2013**: teamrankings=5316, numberfire=4021, consensus=3505
- **2014**: teamrankings=5509, consensus=3895, numberfire=3894
- **2015**: teamrankings=5498, numberfire=3956, consensus=3934
- **2016**: teamrankings=5519, numberfire=3883, consensus=3744
- **2017**: teamrankings=5502, numberfire=3900, consensus=3499
- **2018**: teamrankings=5497, consensus=3903, numberfire=3602
- **2019**: teamrankings=5582, numberfire=5443, consensus=2063
- **2020**: teamrankings=5384, consensus=5343, numberfire=4942
- **2021**: teamrankings=4511, numberfire=4067, consensus=3934
- **2022**: teamrankings=5733, consensus=5119
- **2023**: ESPN BET=5833, teamrankings=5732, consensus=5602
- **2024**: ESPN BET=5311
- **2025**: ESPN BET=5440
- **2026**: Bovada=4952, Draft Kings=4505, ESPN BET=1256

### 2026 open vs. close

- Line-rows with a provider quote: 10713
- Rows with BOTH spread and spreadOpen: 10576 (98.7%)
- Rows with BOTH overUnder and overUnderOpen: 10505 (98.1%)

`spread - spreadOpen` distribution:

|       |             0 |
|:------|--------------:|
| count | 10576         |
| mean  |    -0.0414618 |
| std   |     1.64454   |
| min   |   -26         |
| 25%   |    -1         |
| 50%   |     0         |
| 75%   |     1         |
| max   |    26         |

`overUnder - overUnderOpen` distribution:

|       |             0 |
|:------|--------------:|
| count | 10505         |
| mean  |     0.0198001 |
| std   |     2.39231   |
| min   |   -14         |
| 25%   |    -1.5       |
| 50%   |     0         |
| 75%   |     1.5       |
| max   |    17         |

### Join to hoopR schedules (2025, 2026)

- **2025**: matched 6246/6299 CBBD games (99.2%) to a hoopR schedule row on (date, home team name, away team name).
  - Sample unmatched (CBBD homeTeam/awayTeam/date), up to 10 shown:
    - 2024-11-05 | Dayton vs St. Francis (PA)
    - 2024-11-05 | Mount St. Mary's vs Notre Dame (Md)
    - 2024-11-05 | Kansas City vs Hannibal-LaGrange
    - 2024-11-05 | Jacksonville vs Trinity Baptist
    - 2024-11-09 | Clemson vs St. Francis (PA)
    - 2024-11-10 | Loyola Maryland vs Lancaster Bible College
    - 2024-11-10 | Campbell vs St. Francis (PA)
    - 2024-11-13 | Penn State vs St. Francis (PA)
    - 2024-11-14 | Mercer vs Trinity Baptist
    - 2024-11-16 | Stony Brook vs Saint Joseph's Long Island
- **2026**: matched 6090/6317 CBBD games (96.4%) to a hoopR schedule row on (date, home team name, away team name).
  - Sample unmatched (CBBD homeTeam/awayTeam/date), up to 10 shown:
    - 2025-11-03 | Prairie View A&M vs College Of Biblical Studies
    - 2025-11-03 | Stonehill vs Thomas (ME)
    - 2025-11-03 | VMI vs Johnson & Wales (NC)
    - 2025-11-04 | Morehead State vs Midway University
    - 2025-11-04 | Florida Gulf Coast vs New College (FL)
    - 2025-11-04 | Central Connecticut vs Vermont State - Johnson
    - 2025-11-04 | Chattanooga vs Union (KY)
    - 2025-11-04 | Grambling vs Huston-Tillotson
    - 2025-11-04 | Alabama A&M vs Blue Mountain
    - 2025-11-04 | Army vs SUNY-Maritime

## 2. Games (`/games`)

|   season |   games | neutralSite_%   | seasonTypes                          |   games_with_period_scores |   OT_games(periods>2) | OT_%_of_with_periods   | score_present_%   |
|---------:|--------:|:----------------|:-------------------------------------|---------------------------:|----------------------:|:-----------------------|:------------------|
|     2013 |    5838 | 11.8%           | {'regular': 5726, 'postseason': 112} |                       5766 |                   322 | 5.6%                   | 100.0%            |
|     2014 |    5995 | 11.3%           | {'regular': 5849, 'postseason': 146} |                       5948 |                   368 | 6.2%                   | 100.0%            |
|     2015 |    5949 | 10.8%           | {'regular': 5804, 'postseason': 145} |                       5932 |                   357 | 6.0%                   | 100.0%            |
|     2016 |    5991 | 11.1%           | {'regular': 5844, 'postseason': 147} |                       5951 |                   343 | 5.8%                   | 100.0%            |
|     2017 |    5982 | 10.9%           | {'regular': 5856, 'postseason': 126} |                       5964 |                   342 | 5.7%                   | 100.0%            |
|     2018 |    5987 | 11.0%           | {'regular': 5853, 'postseason': 134} |                       5969 |                   396 | 6.6%                   | 100.0%            |
|     2019 |    6066 | 11.0%           | {'regular': 5926, 'postseason': 140} |                       6049 |                   349 | 5.8%                   | 100.0%            |
|     2020 |    5856 | 9.2%            | {'regular': 5856}                    |                       5768 |                   301 | 5.2%                   | 100.0%            |
|     2021 |    5282 | 10.1%           | {'regular': 5192, 'postseason': 90}  |                       4285 |                   220 | 5.1%                   | 100.0%            |
|     2022 |    6387 | 11.0%           | {'regular': 6255, 'postseason': 132} |                       5967 |                   320 | 5.4%                   | 100.0%            |
|     2023 |    6261 | 12.3%           | {'regular': 6148, 'postseason': 113} |                       6222 |                   361 | 5.8%                   | 100.0%            |
|     2024 |    6249 | 11.9%           | {'regular': 6129, 'postseason': 120} |                       6243 |                   346 | 5.5%                   | 100.0%            |
|     2025 |    6299 | 11.9%           | {'regular': 6176, 'postseason': 123} |                       6292 |                   325 | 5.2%                   | 100.0%            |
|     2026 |    6317 | 10.9%           | {'regular': 6212, 'postseason': 105} |                       6299 |                   328 | 5.2%                   | 100.0%            |

## 3. Ratings (`/ratings/adjusted`, `/ratings/srs`, `/ratings/elo`)

### `ratings_adjusted` -- adjusted (offensive/defensive/net rating)

Fields returned: `season, teamId, team, conference, offensiveRating, defensiveRating, netRating, rankings`
Any date/asOf field present: **False**

Sample rows for ['Duke', 'Houston', 'Auburn', 'Florida', 'Tennessee']:

|   season |   teamId | team      | conference   |   offensiveRating |   defensiveRating |   netRating | rankings                                |
|---------:|---------:|:----------|:-------------|------------------:|------------------:|------------:|:----------------------------------------|
|     2025 |       72 | Duke      | ACC          |             130.3 |              90.5 |        39.8 | {'offense': 1, 'defense': 4, 'net': 1}  |
|     2025 |      113 | Houston   | Big 12       |             123.3 |              86.9 |        36.4 | {'offense': 11, 'defense': 1, 'net': 2} |
|     2025 |       16 | Auburn    | SEC          |             128.4 |              93.4 |        35.1 | {'offense': 2, 'defense': 8, 'net': 3}  |
|     2025 |       87 | Florida   | SEC          |             126.5 |              91.9 |        34.6 | {'offense': 4, 'defense': 5, 'net': 4}  |
|     2025 |      292 | Tennessee | SEC          |             120.8 |              90.3 |        30.5 | {'offense': 17, 'defense': 3, 'net': 6} |

### `ratings_srs` -- SRS

Fields returned: `season, teamId, team, conference, rating`
Any date/asOf field present: **False**

Sample rows for ['Duke', 'Houston', 'Auburn', 'Florida', 'Tennessee']:

|   season |   teamId | team      | conference   |   rating |
|---------:|---------:|:----------|:-------------|---------:|
|     2025 |       72 | Duke      | ACC          |     22.2 |
|     2025 |       16 | Auburn    | SEC          |     18.9 |
|     2025 |       87 | Florida   | SEC          |     18.5 |
|     2025 |      113 | Houston   | Big 12       |     18.4 |
|     2025 |      292 | Tennessee | SEC          |     14.7 |

**Anomaly**: `ratings_srs_2026.parquet` has **0 rows** (other seasons/endpoints for 2026 are populated -- e.g. `ratings_elo_2026` and `ratings_adjusted_2026` both have ~365 rows). This looks like a CBBD data-availability gap for this specific endpoint+season, not a pull bug -- flagging for awareness before relying on `ratings_srs` for 2026.

### `ratings_elo` -- Elo (season-level)

Fields returned: `season, teamId, team, conference, elo`
Any date/asOf field present: **False**

Sample rows for ['Duke', 'Houston', 'Auburn', 'Florida', 'Tennessee']:

|   season |   teamId | team      | conference                |   elo |
|---------:|---------:|:----------|:--------------------------|------:|
|     2025 |       16 | Auburn    | Southeastern Conference   |  2308 |
|     2025 |       72 | Duke      | Atlantic Coast Conference |  2497 |
|     2025 |       87 | Florida   | Southeastern Conference   |  2461 |
|     2025 |      113 | Houston   | Big 12 Conference         |  2426 |
|     2025 |      292 | Tennessee | Southeastern Conference   |  2228 |

### Point-in-time verdict

`/ratings/adjusted`, `/ratings/srs`, and `/ratings/elo` (queried by `season` only) each return **exactly one row per team-season** with no date/asOf/game-id field anywhere in the payload -- verified by inspecting the raw JSON keys above. These are **end-of-season snapshots**, computed over the team's full slate of games for that season. Joining any of them onto a game as a *pregame* feature (for any game before the season's last game) is a **look-ahead leak**: the rating already reflects games that, for an early- or mid-season game, have not been played yet.

By contrast, `/games` carries `homeTeamEloStart` / `homeTeamEloEnd` / `awayTeamEloStart` / `awayTeamEloEnd` **per game** -- this IS point-in-time (the Elo value entering that specific game) and is safe to use as a pregame feature. `/ratings/elo` (season-level) is a different, later snapshot and should not be confused with the per-game Elo embedded in `/games`.

**Practical implication**: for a pregame model, do not join `/ratings/adjusted` or `/ratings/srs` by (team, season) onto in-season games. Either (a) use only the per-game Elo from `/games`, or (b) pull `/ratings/adjusted`/`srs` incrementally after each date and reconstruct a time series (not attempted in this pull), or (c) restrict use of the season-end adjusted/SRS ratings to season-level or postseason analysis where the leak window is closed.

## 4. Samples (`/plays`, `/substitutions`, `/lineups`, `/games/players`, `/teams/roster`, `/recruiting/portal`)

Sample games (Duke, season 2025): [9, 298, 512]

### `/plays/game/{id}` structure

- 327 play records for game 9.
- Top-level fields: `gameId, gameSourceId, gameStartDate, season, seasonType, gameType, tournament, id, sourceId, playType, isHomeTeam, teamId, team, conference, teamSeed, opponentId, opponent, opponentConference, opponentSeed, homeScore, awayScore, homeWinProbability, period, clock, secondsRemaining, scoringPlay, shootingPlay, scoreValue, wallclock, playText, participants, shotInfo, onFloor`
- `onFloor`: list of 10 dicts (`id`, `name`, `team`) -- the full 5-vs-5 lineup on court at the moment of the play, for BOTH teams. This is present on every play record, not just shots/subs.
- `shotInfo` (only on `shootingPlay=true` rows): `{"shooter": {"id": 206, "name": "Kon Knueppel"}, "made": true, "range": "jumper", "assisted": false, "assistedBy": {"id": null, "name": null}, "location": {"x": 874.2, "y": 285}}` -- shooter id/name, made bool, `range` (e.g. jumper/layup/dunk), assisted flag + assister, and x/y shot `location`.
- Non-shooting plays (fouls, subs-as-plays, turnovers, etc.) have `shotInfo: null`.

### `/substitutions/game/{id}` structure

- 77 substitution records for game 9 (one row per player per stint on court).
- Fields: `gameId, startDate, teamId, team, conference, athleteId, athlete, position, opponentId, opponent, opponentConference, subIn, subOut`
- `subIn` / `subOut` are nested `{period, secondsRemaining, teamPoints, opponentPoints}` -- gives exact game-clock and score context for when a player entered/left, i.e. a stint table, not raw substitution events.

### `/lineups/game/{id}` structure

- 31 five-man-lineup records for game 9 (one row per distinct 5-man unit that appeared, per team).
- Top-level fields: `teamId, team, conference, idHash, athletes, totalSeconds, pace, offenseRating, defenseRating, netRating, teamStats, opponentStats`
- `athletes`: list of 5 `{id, name}` dicts identifying the unit (`idHash` is a stable sorted-id key for the lineup).
- Per-lineup minutes/possessions: `totalSeconds`=527, `pace`=68.3 -- YES, minutes and pace (~possessions proxy) are present.
- Ratings: `offenseRating`, `defenseRating`, `netRating` per lineup.
- `teamStats` / `opponentStats`: full box-score-style splits for the lineup (`points, possessions, assists, steals, turnovers, blocks, defensiveRebounds, offensiveRebounds, trueShooting, fieldGoals, freeThrows, twoPointers, threePointers, fourFactors`), including a `fourFactors` sub-block (`effectiveFieldGoalPct`, `turnoverRatio`, `offensiveReboundPct`, `freeThrowRate`).

### `/games/players` structure

- 39 game-rows for Duke, season 2025 (one row per game, `players` nested list per game).
- Per-game top fields: `gameId, season, seasonLabel, seasonType, tournament, startDate, startTimeTbd, conferenceGame, neutralSite, isHome, gameType, notes, teamId, team, conference, teamSeed, opponentId, opponent, opponentConference, opponentSeed, gameMinutes, gamePace, players`
- Per-player fields: `athleteId, athleteSourceId, name, position, starter, ejected, minutes, points, turnovers, fouls, assists, steals, blocks, gameScore, offensiveRating, defensiveRating, netRating, usage, effectiveFieldGoalPct, trueShootingPct, assistsTurnoverRatio, freeThrowRate, offensiveReboundPct, fieldGoals, twoPointFieldGoals, threePointFieldGoals, freeThrows, rebounds`
- Includes box score + advanced per-player: minutes, usage, offensive/defensive/net rating, gameScore, true shooting%, effective FG%, four-factor-style rates, and split shooting (FG/2P/3P/FT) with makes/attempts/pct.

### `/teams/roster` structure

- 15 players on the Duke 2025 roster.
- Team-level fields: `teamId, teamSourceId, team, conference, season, players`
- Per-player fields: `id, sourceId, name, firstName, lastName, jersey, position, height, weight, hometown, dateOfBirth, startSeason, endSeason`
- `hometown` is a nested dict (city/state/country/lat/long/countyFips); `dateOfBirth` was null for the sampled players; `startSeason`/`endSeason` bound eligibility on this roster snapshot.

### `/recruiting/portal` structure

- 1611 transfer-portal entries pulled with `year=2025` (**note**: the endpoint's real filter param is `year`, not `season` -- passing `season=2025` is silently ignored and returns all years unfiltered [verified: 6679 rows spanning years 2021-2026 vs. 1611 rows with `year=2025`]).
- Fields: `id, sourceId, year, firstName, lastName, position, origin, destination, stars, rating, eligibility, yearsRemaining`
- `origin`/`destination` are nested `{id, name, conference}`; includes `stars`, `rating`, `eligibility` (e.g. 'Immediate'), and `yearsRemaining`.

### CBBD vs. hoopR as the primary event source for lineup-on-floor modelling

**Verdict: CBBD is the primary source.** The hoopR MBB play-by-play parquet (`data/raw/hoopr/pbp/play_by_play_2025.parquet`, 63 columns) carries only `athlete_id_1/2/3` (the play's direct participants -- shooter/assister/etc.), with **no on-court roster field at all**; reconstructing which 10 players were on the floor at a given moment from hoopR alone would require inferring stints from a separate substitution/box signal that hoopR's MBB pbp does not appear to expose either. CBBD's `/plays/game/{id}` embeds a 10-player `onFloor` list (5 per team, with team label) on **every single play record**, and CBBD additionally ships a purpose-built `/lineups/game/{id}` endpoint that pre-aggregates minutes (`totalSeconds`), pace, offensive/defensive/net rating, and a full four-factors box line per distinct 5-man unit -- i.e. the lineup-level modelling target is largely pre-computed. hoopR remains useful for shot coordinates/shot-chart detail (`shots_*.parquet`) and player/team box scores, but for on-floor lineup composition and lineup-level efficiency, CBBD's `onFloor` + `/lineups` are the right primary source.

## 5. Call budget

- Total API calls made this pull: **393**
- `X-CallLimit-Remaining` at end of pull: **29527**
- Manifest generated at: 2026-09-10T12:58:23.137365+00:00
- Manifest entries: 57 (see `data/raw/cbbd/manifest.json` for full detail)
