# hoopR MBB data audit

Generated: 2026-09-10T13:02:17+00:00  
Source: `sportsdataverse/hoopR-mbb-data` (GitHub, CC BY 4.0), pulled via `scripts/pull_hoopr.py` into `data/raw/hoopr`  
Seasons audited: [2022, 2023, 2024, 2025, 2026] (hoopR season = ending year of the season, e.g. 2022 = 2021-22 season)  
Schedule preview also pulled for season 2027 (upcoming, mostly-unplayed -- reported separately, excluded from gate-reference stats).

---

## 1. Schedules

Games and date coverage per season (2027 shown separately as a forward-looking, mostly-unplayed schedule):

| season | games | status_completed=True | min game_date | max game_date |
|---|---|---|---|---|
| 2022 | 5976 | 5976 | 2021-11-09 | 2022-04-04 |
| 2023 | 6261 | 6228 | 2022-11-07 | 2023-04-03 |
| 2024 | 6249 | 6243 | 2023-11-06 | 2024-04-08 |
| 2025 | 6299 | 6292 | 2024-11-04 | 2025-04-07 |
| 2026 | 6318 | 6300 | 2025-11-03 | 2026-04-06 |
| 2027 | 1629 | 0 | 2026-11-02 | 2027-03-06 |

Schema drift note: schedule column counts differ by season -- {2022: 76, 2023: 85, 2024: 84, 2025: 87, 2026: 86}. 2022 lacks `home_linescores`/`away_linescores`/`home_records`/`away_records`/`*_current_rank`/`broadcast_market`/`broadcast_name`/`play_by_play_available` (all present from 2023+); 2025+ add `broadcast`/`highlights`; 2025 uniquely adds a near-empty `away_non_div1_team` column (see below). Treat the schedules schema as evolving, not fixed, across seasons.

### 2022

- games: **5,976**
- `season_type` value counts: {2: 5846, 3: 130}  (2=regular season, 3=postseason/tournament per ESPN convention; cross-check against `type_abbreviation`: {'N/A': 5642, nan: 172, 'STD': 39, 'Bowl Game': 17, 'RD128': 15, 'QRR': 13, 'RD32': 13, 'CC': 10, 'TRNMNT': 10, 'SEMI': 10, 'QUALIFY': 9, 'FINAL': 7, 'EXH': 7, 'Conference Championship': 3, 'Championship': 3, 'RD64': 3, 'QTR': 3})
- `neutral_site` counts: {False: 5282, True: 694}; `conference_competition` counts: {True: 3508, False: 2468}
- venues: 437 distinct `venue_id`; `venue_indoor` counts: {True: 5975, False: 1}; `attendance` populated for 100.0% of games
- D-I identification field used: `home_conference_id` / `away_conference_id` (null => team not in an ESPN-tracked D-I conference for that game). Both-teams-D-I games: **5,485** / 5,976. Games with >=1 non-D-I side: **491** (home-side null: 6, away-side null: 485, both null: 0). Distinct non-D-I `team_id`s involved: **322**.
  - home-side nulls are a mix of legitimate non-D-I host venues (e.g. Chaminade 'Silverswords' at the Maui Invitational) and a couple of apparent data gaps for nominally D-I hosts; sample: [{'game_id': 401370158, 'game_date': datetime.date(2021, 12, 11), 'home_name': 'Cougars', 'away_name': 'Eagles', 'type_abbreviation': 'N/A'}, {'game_id': 401376727, 'game_date': datetime.date(2021, 11, 24), 'home_name': 'Silverswords', 'away_name': 'Bulldogs', 'type_abbreviation': 'N/A'}, {'game_id': 401376723, 'game_date': datetime.date(2021, 11, 23), 'home_name': 'Silverswords', 'away_name': 'Fighting Irish', 'type_abbreviation': 'N/A'}, {'game_id': 401369812, 'game_date': datetime.date(2021, 11, 22), 'home_name': 'Silverswords', 'away_name': 'Ducks', 'type_abbreviation': 'N/A'}, {'game_id': 401385916, 'game_date': datetime.date(2021, 11, 13), 'home_name': 'Uvi', 'away_name': 'Lions', 'type_abbreviation': 'N/A'}]
- `status_type_name` counts: {'STATUS_FINAL': 5966, 'STATUS_FORFEIT': 10}; games with a final score (`status_type_completed`==True): **5,976** / 5,976
- OT indicator: `home_linescores`/`away_linescores` columns **do not exist** in this season's schedule file (schema drift -- see below); OT is instead derived from pbp `period_number` in Section 4/6.
- `format_regulation_periods` is constant at [2.0] for all seasons (halves format metadata, not an OT flag by itself).
- **no embedded market fields** (spread / over_under / moneyline) exist on the schedules table itself. The market-ish columns (`game_spread`, `home_team_spread`, `home_favorite`, `game_spread_available`, `pregame_home_prob`, `home_win_prob`) live on the **pbp** table only -- see Section 4 for non-null rates and date coverage, including an important frozen-field defect there.

### 2023

- games: **6,261**
- `season_type` value counts: {2: 6148, 3: 113}  (2=regular season, 3=postseason/tournament per ESPN convention; cross-check against `type_abbreviation`: {'N/A': 5853, nan: 236, 'STD': 51, 'TRNMNT': 23, 'RD128': 15, 'CC': 10, 'RD32': 10, 'SEMI': 10, 'QTR': 10, 'QUALIFY': 9, 'Championship': 9, 'RD64': 8, 'FINAL': 7, 'EXH': 7, 'Conference Championship': 3})
- `neutral_site` counts: {False: 5492, True: 769}; `conference_competition` counts: {True: 3612, False: 2649}
- venues: 460 distinct `venue_id`; `venue_indoor` counts: {True: 6258, False: 3}; `attendance` populated for 100.0% of games
- D-I identification field used: `home_conference_id` / `away_conference_id` (null => team not in an ESPN-tracked D-I conference for that game). Both-teams-D-I games: **5,749** / 6,261. Games with >=1 non-D-I side: **512** (home-side null: 3, away-side null: 509, both null: 0). Distinct non-D-I `team_id`s involved: **346**.
  - home-side nulls are a mix of legitimate non-D-I host venues (e.g. Chaminade 'Silverswords' at the Maui Invitational) and a couple of apparent data gaps for nominally D-I hosts; sample: [{'game_id': 401494594, 'game_date': datetime.date(2022, 11, 26), 'home_name': 'Firehawks', 'away_name': 'Bulldogs', 'type_abbreviation': 'N/A'}, {'game_id': 401489621, 'game_date': datetime.date(2022, 11, 8), 'home_name': 'Mighty Macs', 'away_name': 'Great Danes', 'type_abbreviation': 'N/A'}, {'game_id': 401487408, 'game_date': datetime.date(2022, 11, 7), 'home_name': 'Maroon Tigers', 'away_name': 'Crimson', 'type_abbreviation': 'N/A'}]
- `status_type_name` counts: {'STATUS_FINAL': 6222, 'STATUS_CANCELED': 21, 'STATUS_POSTPONED': 12, 'STATUS_FORFEIT': 6}; games with a final score (`status_type_completed`==True): **6,228** / 6,261
  - non-completed games (33) have home_score/away_score == 0-0 in 100.0% of cases (placeholder, not a real result)
- OT indicator: derived by counting period entries in `home_linescores` (regulation = 2 halves). Periods-per-game distribution: {2.0: 5888, 3.0: 312, 4.0: 44, 5.0: 4, 6.0: 1, nan: 12}. Games that went to OT: **361** (5.8%).
- `format_regulation_periods` is constant at [2.0] for all seasons (halves format metadata, not an OT flag by itself).
- **no embedded market fields** (spread / over_under / moneyline) exist on the schedules table itself. The market-ish columns (`game_spread`, `home_team_spread`, `home_favorite`, `game_spread_available`, `pregame_home_prob`, `home_win_prob`) live on the **pbp** table only -- see Section 4 for non-null rates and date coverage, including an important frozen-field defect there.

### 2024

- games: **6,249**
- `season_type` value counts: {2: 6129, 3: 120}  (2=regular season, 3=postseason/tournament per ESPN convention; cross-check against `type_abbreviation`: {'STD': 5570, 'TRNMNT': 679})
- `neutral_site` counts: {False: 5503, True: 746}; `conference_competition` counts: {True: 3565, False: 2684}
- venues: 454 distinct `venue_id`; `venue_indoor` counts: {True: 6248, False: 1}; `attendance` populated for 100.0% of games
- D-I identification field used: `home_conference_id` / `away_conference_id` (null => team not in an ESPN-tracked D-I conference for that game). Both-teams-D-I games: **5,731** / 6,249. Games with >=1 non-D-I side: **518** (home-side null: 5, away-side null: 513, both null: 0). Distinct non-D-I `team_id`s involved: **355**.
  - home-side nulls are a mix of legitimate non-D-I host venues (e.g. Chaminade 'Silverswords' at the Maui Invitational) and a couple of apparent data gaps for nominally D-I hosts; sample: [{'game_id': 401622999, 'game_date': datetime.date(2023, 12, 27), 'home_name': 'Flames', 'away_name': 'Lopes', 'type_abbreviation': 'STD'}, {'game_id': 401581601, 'game_date': datetime.date(2023, 11, 22), 'home_name': 'Silverswords', 'away_name': 'Orange', 'type_abbreviation': 'TRNMNT'}, {'game_id': 401581597, 'game_date': datetime.date(2023, 11, 21), 'home_name': 'Silverswords', 'away_name': 'Bruins', 'type_abbreviation': 'TRNMNT'}, {'game_id': 401575875, 'game_date': datetime.date(2023, 11, 20), 'home_name': 'Silverswords', 'away_name': 'Jayhawks', 'type_abbreviation': 'TRNMNT'}, {'game_id': 401584390, 'game_date': datetime.date(2023, 11, 18), 'home_name': 'Storm', 'away_name': 'Trailblazers', 'type_abbreviation': 'STD'}]
- `status_type_name` counts: {'STATUS_FINAL': 6243, 'STATUS_POSTPONED': 4, 'STATUS_CANCELED': 2}; games with a final score (`status_type_completed`==True): **6,243** / 6,249
  - non-completed games (6) have home_score/away_score == 0-0 in 100.0% of cases (placeholder, not a real result)
- OT indicator: derived by counting period entries in `home_linescores` (regulation = 2 halves). Periods-per-game distribution: {2.0: 5899, 3.0: 295, 4.0: 42, 5.0: 9, nan: 4}. Games that went to OT: **346** (5.5%).
- `format_regulation_periods` is constant at [2.0] for all seasons (halves format metadata, not an OT flag by itself).
- **no embedded market fields** (spread / over_under / moneyline) exist on the schedules table itself. The market-ish columns (`game_spread`, `home_team_spread`, `home_favorite`, `game_spread_available`, `pregame_home_prob`, `home_win_prob`) live on the **pbp** table only -- see Section 4 for non-null rates and date coverage, including an important frozen-field defect there.

### 2025

- games: **6,299**
- `season_type` value counts: {2: 6176, 3: 123}  (2=regular season, 3=postseason/tournament per ESPN convention; cross-check against `type_abbreviation`: {'STD': 5477, 'TRNMNT': 822})
- `neutral_site` counts: {False: 5552, True: 747}; `conference_competition` counts: {True: 3650, False: 2649}
- venues: 458 distinct `venue_id`; `venue_indoor` counts: {True: 6298, False: 1}; `attendance` populated for 100.0% of games
- D-I identification field used: `home_conference_id` / `away_conference_id` (null => team not in an ESPN-tracked D-I conference for that game). Both-teams-D-I games: **5,769** / 6,299. Games with >=1 non-D-I side: **530** (home-side null: 7, away-side null: 523, both null: 0). Distinct non-D-I `team_id`s involved: **338**.
  - home-side nulls are a mix of legitimate non-D-I host venues (e.g. Chaminade 'Silverswords' at the Maui Invitational) and a couple of apparent data gaps for nominally D-I hosts; sample: [{'game_id': 401729953, 'game_date': datetime.date(2024, 12, 5), 'home_name': 'Saints', 'away_name': 'Chippewas', 'type_abbreviation': 'STD'}, {'game_id': 401732174, 'game_date': datetime.date(2024, 11, 27), 'home_name': 'Panthers', 'away_name': 'Salukis', 'type_abbreviation': 'TRNMNT'}, {'game_id': 401732172, 'game_date': datetime.date(2024, 11, 26), 'home_name': 'Panthers', 'away_name': 'Cardinals', 'type_abbreviation': 'TRNMNT'}, {'game_id': 401732302, 'game_date': datetime.date(2024, 11, 24), 'home_name': 'Gallitos', 'away_name': 'Roos', 'type_abbreviation': 'TRNMNT'}, {'game_id': 401721929, 'game_date': datetime.date(2024, 11, 23), 'home_name': 'Pirates', 'away_name': 'Eagles', 'type_abbreviation': 'TRNMNT'}]
- `status_type_name` counts: {'STATUS_FINAL': 6292, 'STATUS_POSTPONED': 5, 'STATUS_SCHEDULED': 1, 'STATUS_CANCELED': 1}; games with a final score (`status_type_completed`==True): **6,292** / 6,299
  - non-completed games (7) have home_score/away_score == 0-0 in 100.0% of cases (placeholder, not a real result)
- OT indicator: derived by counting period entries in `home_linescores` (regulation = 2 halves). Periods-per-game distribution: {2.0: 5969, 3.0: 268, 4.0: 46, 5.0: 10, 6.0: 1, nan: 5}. Games that went to OT: **325** (5.2%).
- `format_regulation_periods` is constant at [2.0] for all seasons (halves format metadata, not an OT flag by itself).
- note: this season's schedule also carries an `away_non_div1_team` column, but it is populated for only 4 row(s) out of 6,299 -- far fewer than the 523 rows with a null `away_conference_id`. It looks like an incompletely-populated one-off field (also absent from every other season's schema) and should **not** be relied on as the D-I flag; `*_conference_id` nullness is the more consistent proxy used throughout this report.
- **no embedded market fields** (spread / over_under / moneyline) exist on the schedules table itself. The market-ish columns (`game_spread`, `home_team_spread`, `home_favorite`, `game_spread_available`, `pregame_home_prob`, `home_win_prob`) live on the **pbp** table only -- see Section 4 for non-null rates and date coverage, including an important frozen-field defect there.

### 2026

- games: **6,318**
- `season_type` value counts: {2: 6213, 3: 105}  (2=regular season, 3=postseason/tournament per ESPN convention; cross-check against `type_abbreviation`: {'STD': 5654, 'TRNMNT': 664})
- `neutral_site` counts: {False: 5609, True: 709}; `conference_competition` counts: {True: 3673, False: 2645}
- venues: 463 distinct `venue_id`; `venue_indoor` counts: {True: 6317, False: 1}; `attendance` populated for 100.0% of games
- D-I identification field used: `home_conference_id` / `away_conference_id` (null => team not in an ESPN-tracked D-I conference for that game). Both-teams-D-I games: **5,767** / 6,318. Games with >=1 non-D-I side: **551** (home-side null: 3, away-side null: 548, both null: 0). Distinct non-D-I `team_id`s involved: **363**.
  - home-side nulls are a mix of legitimate non-D-I host venues (e.g. Chaminade 'Silverswords' at the Maui Invitational) and a couple of apparent data gaps for nominally D-I hosts; sample: [{'game_id': 401831203, 'game_date': datetime.date(2025, 11, 26), 'home_name': 'Silverswords', 'away_name': 'Broncos', 'type_abbreviation': 'TRNMNT'}, {'game_id': 401831199, 'game_date': datetime.date(2025, 11, 25), 'home_name': 'Silverswords', 'away_name': 'Longhorns', 'type_abbreviation': 'TRNMNT'}, {'game_id': 401824137, 'game_date': datetime.date(2025, 11, 24), 'home_name': 'Silverswords', 'away_name': 'Cougars', 'type_abbreviation': 'TRNMNT'}]
- `status_type_name` counts: {'STATUS_FINAL': 6300, 'STATUS_POSTPONED': 14, 'STATUS_CANCELED': 4}; games with a final score (`status_type_completed`==True): **6,300** / 6,318
  - non-completed games (18) have home_score/away_score == 0-0 in 100.0% of cases (placeholder, not a real result)
- OT indicator: derived by counting period entries in `home_linescores` (regulation = 2 halves). Periods-per-game distribution: {2.0: 5977, 3.0: 276, 4.0: 42, 5.0: 9, nan: 14}. Games that went to OT: **327** (5.2%).
- `format_regulation_periods` is constant at [2.0] for all seasons (halves format metadata, not an OT flag by itself).
- **no embedded market fields** (spread / over_under / moneyline) exist on the schedules table itself. The market-ish columns (`game_spread`, `home_team_spread`, `home_favorite`, `game_spread_available`, `pregame_home_prob`, `home_win_prob`) live on the **pbp** table only -- see Section 4 for non-null rates and date coverage, including an important frozen-field defect there.

## 2. Team box

Columns (54): `game_id, season, season_type, game_date, game_date_time, team_id, team_uid, team_slug, team_location, team_name, team_abbreviation, team_display_name, team_short_display_name, team_color, team_alternate_color, team_logo, team_home_away, team_score, team_winner, assists, blocks, defensive_rebounds, field_goal_pct, field_goals_made, field_goals_attempted, flagrant_fouls, fouls, free_throw_pct, free_throws_made, free_throws_attempted, largest_lead, offensive_rebounds, steals, team_turnovers, technical_fouls, three_point_field_goal_pct, three_point_field_goals_made, three_point_field_goals_attempted, total_rebounds, total_technical_fouls, total_turnovers, turnovers, opponent_team_id, opponent_team_uid, opponent_team_slug, opponent_team_location, opponent_team_name, opponent_team_abbreviation, opponent_team_display_name, opponent_team_short_display_name, opponent_team_color, opponent_team_alternate_color, opponent_team_logo, opponent_team_score`

| season | rows | games covered | games w/ !=2 team rows | sched games missing from team_box | team_box games not in sched | home_score mismatch | away_score mismatch |
|---|---|---|---|---|---|---|---|
| 2022 | 11930 | 5965 | 0 | 11 | 0 | 0/5965 | 0/5965 |
| 2023 | 12440 | 6220 | 0 | 41 | 0 | 0/6220 | 0/6220 |
| 2024 | 12480 | 6240 | 0 | 9 | 0 | 0/6240 | 0/6240 |
| 2025 | 12572 | 6286 | 0 | 13 | 0 | 0/6285 | 0/6285 |
| 2026 | 12598 | 6299 | 0 | 19 | 0 | 0/6299 | 0/6299 |

`team_score` in team_box matches `home_score`/`away_score` on the schedules table for essentially all completed games in every season (see mismatch columns above); team_box is treated as internally consistent with schedules for scoring.

## 3. Player box

| season | rows | games | players | minutes null rate | did_not_play | active | starter |
|---|---|---|---|---|---|---|---|
| 2022 | 191548 | 5965 | 12234 | 37.2% | {False: 120233, True: 71315} | {False: 191548} | {False: 131904, True: 59644} |
| 2023 | 196589 | 6221 | 12697 | 36.2% | {False: 125343, True: 71246} | {False: 196589} | {False: 134409, True: 62180} |
| 2024 | 198586 | 6241 | 12600 | 36.6% | {False: 125918, True: 72668} | {False: 198586} | {False: 136325, True: 62261} |
| 2025 | 207613 | 6290 | 15235 | 38.6% | {False: 127421, True: 80192} | {None: 207392, False: 221} | {False: 144754, True: 62859} |
| 2026 | 196876 | 6300 | 12481 | 35.2% | {False: 127532, True: 69344} | {False: 133801, True: 62246, None: 829} | {False: 133876, True: 63000} |

- **DEFECT: `active` is unreliable before 2026.** It is constant `False` for every row in 2022-2024, ~100% `None`/null in 2025 (only 221/207,613 rows populated), and only becomes a real, informative True/False signal in 2026 (True correlates with `did_not_play`==False as expected: 62,202/62,246 True rows are NOT DNP). Do not use `active` as a roster-availability feature for seasons before 2026 -- use `did_not_play` instead, which is populated and behaves sensibly in every season.

### 2022

- per-game team point-sum vs `team_score`: **4** / 11,930 team-game rows mismatch
| game_id | team_id | team_name | sum(points) | team_score | diff |
|---|---|---|---|---|---|
| 401371515 | 2547 | Redhawks | 65.0 | 67 | -2.0 |
| 401372396 | 331 | Eagles | 74.0 | 76 | -2.0 |
| 401373049 | 111926 | Royals | 32.0 | 30 | 2.0 |
| 401377765 | 151 | Pirates | 64.0 | 66 | -2.0 |

- per-team-game sum(minutes) vs expected (200 + 25*OT periods): exact match (within 0.5 min) for **11,690** / 11,930 team-games. Diff distribution (minutes over/under expected, rounded), top 10 buckets by frequency: {-200.0: 2, -25.0: 2, -7.0: 1, -3.0: 4, -2.0: 35, -1.0: 100, 0.0: 11690, 1.0: 65, 2.0: 7, 25.0: 20}

### 2023

- per-game team point-sum vs `team_score`: **12** / 12,442 team-game rows mismatch
| game_id | team_id | team_name | sum(points) | team_score | diff |
|---|---|---|---|---|---|
| 401483337 | 2197 | Panthers | 61.0 | 57 | 4.0 |
| 401491692 | 110031 | Redhawks | 53.0 | 49 | 4.0 |
| 401473471 | 161 | Knights | 80.0 | 78 | 2.0 |
| 401470409 | 2492 | Waves | 87.0 | 89 | -2.0 |
| 401486976 | 2099 | Golden Griffins | 62.0 | 64 | -2.0 |
| 401486976 | 2363 | Jaspers | 55.0 | 57 | -2.0 |
| 401490314 | 30 | Trojans | 78.0 | 80 | -2.0 |
| 401491722 | 2272 | Panthers | 109.0 | 111 | -2.0 |
| 401514303 | 314 | Gaels | 74.0 | 76 | -2.0 |
| 401500699 | 2167 | Senators | 64.0 | 66 | -2.0 |

- per-team-game sum(minutes) vs expected (200 + 25*OT periods): exact match (within 0.5 min) for **12,173** / 12,442 team-games. Diff distribution (minutes over/under expected, rounded), top 10 buckets by frequency: {-25.0: 2, -12.0: 1, -3.0: 4, -2.0: 25, -1.0: 112, 0.0: 12173, 1.0: 91, 2.0: 9, 25.0: 14, 50.0: 4}

### 2024

- per-game team point-sum vs `team_score`: **18** / 12,482 team-game rows mismatch
| game_id | team_id | team_name | sum(points) | team_score | diff |
|---|---|---|---|---|---|
| 401597895 | 3169 | Johnson & Wales (Nc) | 15.0 | 61 | -46.0 |
| 401597376 | 2170 | Statesmen | 56.0 | 71 | -15.0 |
| 401576679 | 506 | Eagles | 51.0 | 64 | -13.0 |
| 401604275 | 81 | Pride | 62.0 | 74 | -12.0 |
| 401597922 | 2893 | Bears | 69.0 | 61 | 8.0 |
| 401600073 | 2176 | Blue Devils | 58.0 | 65 | -7.0 |
| 401584376 | 2592 | Knights | 20.0 | 26 | -6.0 |
| 401598007 | 2723 | Falcons | 46.0 | 51 | -5.0 |
| 401606714 | 2927 | Bulldogs | 49.0 | 54 | -5.0 |
| 401608976 | 2726 | Bulldogs | 72.0 | 75 | -3.0 |

- per-team-game sum(minutes) vs expected (200 + 25*OT periods): exact match (within 0.5 min) for **12,387** / 12,482 team-games. Diff distribution (minutes over/under expected, rounded), top 10 buckets by frequency: {-50.0: 2, -29.0: 3, -16.0: 2, -10.0: 3, -4.0: 3, -2.0: 5, -1.0: 37, 0.0: 12387, 1.0: 21, 25.0: 6}

### 2025

- per-game team point-sum vs `team_score`: **21** / 12,580 team-game rows mismatch
| game_id | team_id | team_name | sum(points) | team_score | diff |
|---|---|---|---|---|---|
| 401722338 | 62 | Rainbow Warriors | 110.0 | 96 | 14.0 |
| 401722030 | 130198 | Nittany Lions | 25.0 | 30 | -5.0 |
| 401727325 | 2565 | Cougars | 91.0 | 95 | -4.0 |
| 401720664 | 2491 | Tigers | 38.0 | 42 | -4.0 |
| 401706293 | 2529 | Pioneers | 72.0 | 74 | -2.0 |
| 401715320 | 36 | Rams | 67.0 | 69 | -2.0 |
| 401713720 | 2261 | Pirates | 80.0 | 82 | -2.0 |
| 401712995 | 2711 | Broncos | 63.0 | 65 | -2.0 |
| 401708357 | 251 | Longhorns | 67.0 | 69 | -2.0 |
| 401721227 | 2225 | Bulldogs | 67.0 | 69 | -2.0 |

- per-team-game sum(minutes) vs expected (200 + 25*OT periods): exact match (within 0.5 min) for **12,459** / 12,580 team-games. Diff distribution (minutes over/under expected, rounded), top 10 buckets by frequency: {-25.0: 1, -10.0: 1, -9.0: 1, -7.0: 1, -5.0: 3, -2.0: 5, -1.0: 38, 0.0: 12459, 1.0: 52, 25.0: 14}

### 2026

- per-game team point-sum vs `team_score`: **14** / 12,600 team-game rows mismatch
| game_id | team_id | team_name | sum(points) | team_score | diff |
|---|---|---|---|---|---|
| 401826937 | 495 | Ramblin' Rams | 56.0 | 76 | -20.0 |
| 401826091 | 594 | PACIFIC UNION | 47.0 | 52 | -5.0 |
| 401812169 | 160 | Wildcats | 61.0 | 63 | -2.0 |
| 401820790 | 97 | Cardinals | 90.0 | 92 | -2.0 |
| 401823784 | 328 | Aggies | 70.0 | 72 | -2.0 |
| 401809415 | 2390 | Hurricanes | 100.0 | 102 | -2.0 |
| 401824356 | 104 | Terriers | 74.0 | 76 | -2.0 |
| 401828154 | 249 | Mean Green | 46.0 | 48 | -2.0 |
| 401828286 | 166 | Aggies | 68.0 | 70 | -2.0 |
| 401828468 | 2325 | Explorers | 74.0 | 76 | -2.0 |

- per-team-game sum(minutes) vs expected (200 + 25*OT periods): exact match (within 0.5 min) for **12,510** / 12,600 team-games. Diff distribution (minutes over/under expected, rounded), top 10 buckets by frequency: {-34.0: 1, -25.0: 2, -12.0: 2, -10.0: 1, -2.0: 4, -1.0: 47, 0.0: 12510, 1.0: 23, 2.0: 4, 25.0: 2}

## 4. Play-by-play

Overview:

| season | rows | games covered | sched games missing from pbp |
|---|---|---|---|
| 2022 | 1860561 | 5830 | 146 |
| 2023 | 1957375 | 6116 | 145 |
| 2024 | 2004997 | 6151 | 98 |
| 2025 | 2190101 | 6136 | 163 |
| 2026 | 2915731 | 6275 | 43 |

Coordinate coverage trend (rises sharply over time -- treat pre-2025 pbp coordinates as unreliable/sparse, use the `shots` dataset instead for those seasons):

| season | coord coverage, all rows | coord coverage, shooting_play rows |
|---|---|---|
| 2022 | 12.1% | 12.1% |
| 2023 | 8.8% | 8.8% |
| 2024 | 6.3% | 6.3% |
| 2025 | 33.5% | 25.9% |
| 2026 | 100.0% | 100.0% |

### 2022

- rows: **1,860,561**; games covered: **5,830** / 5,976 scheduled; games in schedule missing from pbp: **146**
  - sample missing game_ids: [401365184, 401365215, 401365236, 401365264, 401365510, 401365521, 401365525, 401365534, 401365544, 401365548]
  - 15.1% of missing-pbp games involve a non-D-I side (null `home_conference_id`/`away_conference_id`) -- most pbp gaps look like exhibition/buy games against non-D-I opponents rather than random data loss, though not all of them are.
- of games present in pbp, **89** / 5,830 (1.5%) have a **truncated/partial pbp feed**: the running `home_score`/`away_score` at the last recorded pbp row never reaches the schedules final score (e.g. game 401725959 in 2025 cuts off mid-1st-half at 38-26 vs a 97-61 final). These are not full game_id-level gaps but partial-coverage games; treat `game_id` presence in pbp as necessary but not sufficient for full-game coverage.
- `shooting_play` counts: {False: 987342, True: 873219}
- `score_value` counts: {0: 987362, 1: 201132, 2: 415891, 3: 256176} (0=non-scoring event, 1=FT, 2=2pt FG, 3=3pt FG; note `type_text`='JumpShot' covers both 2pt and 3pt jumpers -- distinguish makes/attempts by `score_value`, not by `type_text` alone)
- coordinate coverage: 12.1% of all rows, 12.1% of `shooting_play`==True rows have non-null `coordinate_x`/`coordinate_y` -- **shot coordinates are sparse this season** (looks like only a subset of broadcast/tracked games carry ESPN shot-chart coordinates in the pbp table; the separate `shots` dataset should be checked as the primary coordinate source instead of relying on pbp coordinates).
- clock/period fields: `period_number` (int, 1/2=halves, 3+=OT), `period_display_value` (str, e.g. '1st Half'/'2nd Half'/'OT'/'2OT'), `clock_display_value` (str 'MM:SS' counting down), `clock_minutes`/`clock_seconds` (ints, same info split out), `start_period_seconds_remaining`/`end_period_seconds_remaining`/`start_game_seconds_remaining`/`end_game_seconds_remaining` (numeric countdown clocks).
- periods-per-game distribution (from pbp max `period_number`): {2: 5519, 3: 264, 4: 33, 5: 10, 6: 4}
- participants coverage on shooting plays: `athlete_id_1` populated for 100.0% of `shooting_play`==True rows
- substitution events (`type_text` matching 'Substitution'/'Enters'): **0** rows -- **none found this season**
- duplicate (`game_id`,`sequence_number`) pairs: **2**
- embedded market columns non-null rate: `game_spread`=100.0%, `home_team_spread`=100.0%, `home_favorite`=100.0%, `game_spread_available`=100.0%, `pregame_home_prob`=100.0%, `home_win_prob`=100.0%
  - real, varying spread data this season: `game_spread` non-null 100.0%, `game_spread_available`==True for 90.5% of rows, earliest game_date with a real spread 2021-11-09, latest 2022-04-04.

### 2023

- rows: **1,957,375**; games covered: **6,116** / 6,261 scheduled; games in schedule missing from pbp: **145**
  - sample missing game_ids: [401469706, 401469723, 401469756, 401469774, 401469832, 401469845, 401469848, 401469874, 401469984, 401470021]
  - 20.7% of missing-pbp games involve a non-D-I side (null `home_conference_id`/`away_conference_id`) -- most pbp gaps look like exhibition/buy games against non-D-I opponents rather than random data loss, though not all of them are.
- of games present in pbp, **98** / 6,116 (1.6%) have a **truncated/partial pbp feed**: the running `home_score`/`away_score` at the last recorded pbp row never reaches the schedules final score (e.g. game 401725959 in 2025 cuts off mid-1st-half at 38-26 vs a 97-61 final). These are not full game_id-level gaps but partial-coverage games; treat `game_id` presence in pbp as necessary but not sufficient for full-game coverage.
- `shooting_play` counts: {False: 1036780, True: 920595}
- `score_value` counts: {0: 1036892, 1: 218854, 2: 436782, 3: 264847} (0=non-scoring event, 1=FT, 2=2pt FG, 3=3pt FG; note `type_text`='JumpShot' covers both 2pt and 3pt jumpers -- distinguish makes/attempts by `score_value`, not by `type_text` alone)
- coordinate coverage: 8.8% of all rows, 8.8% of `shooting_play`==True rows have non-null `coordinate_x`/`coordinate_y` -- **shot coordinates are sparse this season** (looks like only a subset of broadcast/tracked games carry ESPN shot-chart coordinates in the pbp table; the separate `shots` dataset should be checked as the primary coordinate source instead of relying on pbp coordinates).
- clock/period fields: `period_number` (int, 1/2=halves, 3+=OT), `period_display_value` (str, e.g. '1st Half'/'2nd Half'/'OT'/'2OT'), `clock_display_value` (str 'MM:SS' counting down), `clock_minutes`/`clock_seconds` (ints, same info split out), `start_period_seconds_remaining`/`end_period_seconds_remaining`/`start_game_seconds_remaining`/`end_game_seconds_remaining` (numeric countdown clocks).
- periods-per-game distribution (from pbp max `period_number`): {2: 5764, 3: 304, 4: 43, 5: 4, 6: 1}
- participants coverage on shooting plays: `athlete_id_1` populated for 100.0% of `shooting_play`==True rows
- substitution events (`type_text` matching 'Substitution'/'Enters'): **0** rows -- **none found this season**
- duplicate (`game_id`,`sequence_number`) pairs: **1**
- embedded market columns non-null rate: `game_spread`=100.0%, `home_team_spread`=100.0%, `home_favorite`=100.0%, `game_spread_available`=100.0%, `pregame_home_prob`=100.0%, `home_win_prob`=100.0%
  - real, varying spread data this season: `game_spread` non-null 100.0%, `game_spread_available`==True for 89.7% of rows, earliest game_date with a real spread 2022-11-07, latest 2023-04-03.

### 2024

- rows: **2,004,997**; games covered: **6,151** / 6,249 scheduled; games in schedule missing from pbp: **98**
  - sample missing game_ids: [401574119, 401574638, 401576633, 401577535, 401577699, 401577708, 401577721, 401578964, 401579017, 401579041]
  - 11.2% of missing-pbp games involve a non-D-I side (null `home_conference_id`/`away_conference_id`) -- most pbp gaps look like exhibition/buy games against non-D-I opponents rather than random data loss, though not all of them are.
- of games present in pbp, **101** / 6,151 (1.6%) have a **truncated/partial pbp feed**: the running `home_score`/`away_score` at the last recorded pbp row never reaches the schedules final score (e.g. game 401725959 in 2025 cuts off mid-1st-half at 38-26 vs a 97-61 final). These are not full game_id-level gaps but partial-coverage games; treat `game_id` presence in pbp as necessary but not sufficient for full-game coverage.
- `shooting_play` counts: {False: 1047824, True: 957173}
- `score_value` counts: {0: 1051430, 1: 233443, 2: 449347, 3: 270777} (0=non-scoring event, 1=FT, 2=2pt FG, 3=3pt FG; note `type_text`='JumpShot' covers both 2pt and 3pt jumpers -- distinguish makes/attempts by `score_value`, not by `type_text` alone)
- coordinate coverage: 6.3% of all rows, 6.3% of `shooting_play`==True rows have non-null `coordinate_x`/`coordinate_y` -- **shot coordinates are sparse this season** (looks like only a subset of broadcast/tracked games carry ESPN shot-chart coordinates in the pbp table; the separate `shots` dataset should be checked as the primary coordinate source instead of relying on pbp coordinates).
- clock/period fields: `period_number` (int, 1/2=halves, 3+=OT), `period_display_value` (str, e.g. '1st Half'/'2nd Half'/'OT'/'2OT'), `clock_display_value` (str 'MM:SS' counting down), `clock_minutes`/`clock_seconds` (ints, same info split out), `start_period_seconds_remaining`/`end_period_seconds_remaining`/`start_game_seconds_remaining`/`end_game_seconds_remaining` (numeric countdown clocks).
- periods-per-game distribution (from pbp max `period_number`): {2: 5807, 3: 293, 4: 42, 5: 9}
- participants coverage on shooting plays: `athlete_id_1` populated for 99.9% of `shooting_play`==True rows
- substitution events (`type_text` matching 'Substitution'/'Enters'): **0** rows -- **none found this season**
- duplicate (`game_id`,`sequence_number`) pairs: **2**
- embedded market columns non-null rate: `game_spread`=100.0%, `home_team_spread`=100.0%, `home_favorite`=100.0%, `game_spread_available`=100.0%, `pregame_home_prob`=100.0%, `home_win_prob`=100.0%
  - **DEFECT: frozen placeholder values.** `game_spread`/`home_team_spread`/`home_favorite` take a single constant value this season (`game_spread`=[2.5]) and `game_spread_available` is always False -- this is NOT real betting-line data for this season, it is a dummy default. `home_win_prob`/`pregame_home_prob` remain real, varying model win-probabilities (ESPN BPI-style), not sportsbook odds.

### 2025

- rows: **2,190,101**; games covered: **6,136** / 6,299 scheduled; games in schedule missing from pbp: **163**
  - sample missing game_ids: [401700235, 401700236, 401700283, 401700289, 401700294, 401700299, 401700304, 401700306, 401700400, 401700407]
  - 28.2% of missing-pbp games involve a non-D-I side (null `home_conference_id`/`away_conference_id`) -- most pbp gaps look like exhibition/buy games against non-D-I opponents rather than random data loss, though not all of them are.
- of games present in pbp, **71** / 6,135 (1.2%) have a **truncated/partial pbp feed**: the running `home_score`/`away_score` at the last recorded pbp row never reaches the schedules final score (e.g. game 401725959 in 2025 cuts off mid-1st-half at 38-26 vs a 97-61 final). These are not full game_id-level gaps but partial-coverage games; treat `game_id` presence in pbp as necessary but not sufficient for full-game coverage.
- `shooting_play` counts: {False: 1253591, True: 936510}
- `score_value` counts: {0: 1243632, 1: 232504, 2: 433458, 3: 280507} (0=non-scoring event, 1=FT, 2=2pt FG, 3=3pt FG; note `type_text`='JumpShot' covers both 2pt and 3pt jumpers -- distinguish makes/attempts by `score_value`, not by `type_text` alone)
- coordinate coverage: 33.5% of all rows, 25.9% of `shooting_play`==True rows have non-null `coordinate_x`/`coordinate_y` -- **shot coordinates are sparse this season** (looks like only a subset of broadcast/tracked games carry ESPN shot-chart coordinates in the pbp table; the separate `shots` dataset should be checked as the primary coordinate source instead of relying on pbp coordinates).
- clock/period fields: `period_number` (int, 1/2=halves, 3+=OT), `period_display_value` (str, e.g. '1st Half'/'2nd Half'/'OT'/'2OT'), `clock_display_value` (str 'MM:SS' counting down), `clock_minutes`/`clock_seconds` (ints, same info split out), `start_period_seconds_remaining`/`end_period_seconds_remaining`/`start_game_seconds_remaining`/`end_game_seconds_remaining` (numeric countdown clocks).
- periods-per-game distribution (from pbp max `period_number`): {1: 1, 2: 5817, 3: 261, 4: 46, 5: 10, 6: 1}
- participants coverage on shooting plays: `athlete_id_1` populated for 100.0% of `shooting_play`==True rows
- substitution events (`type_text` matching 'Substitution'/'Enters'): **195,930** rows -- present this season
- duplicate (`game_id`,`sequence_number`) pairs: **2**
- embedded market columns non-null rate: `game_spread`=100.0%, `home_team_spread`=100.0%, `home_favorite`=100.0%, `game_spread_available`=100.0%, `pregame_home_prob`=100.0%, `home_win_prob`=100.0%
  - **DEFECT: frozen placeholder values.** `game_spread`/`home_team_spread`/`home_favorite` take a single constant value this season (`game_spread`=[2.5]) and `game_spread_available` is always False -- this is NOT real betting-line data for this season, it is a dummy default. `home_win_prob`/`pregame_home_prob` remain real, varying model win-probabilities (ESPN BPI-style), not sportsbook odds.

### 2026

- rows: **2,915,731**; games covered: **6,275** / 6,318 scheduled; games in schedule missing from pbp: **43**
  - sample missing game_ids: [401805178, 401806378, 401806434, 401808646, 401808712, 401808713, 401812370, 401812393, 401817253, 401817640]
  - 37.2% of missing-pbp games involve a non-D-I side (null `home_conference_id`/`away_conference_id`) -- most pbp gaps look like exhibition/buy games against non-D-I opponents rather than random data loss, though not all of them are.
- of games present in pbp, **6** / 6,275 (0.1%) have a **truncated/partial pbp feed**: the running `home_score`/`away_score` at the last recorded pbp row never reaches the schedules final score (e.g. game 401725959 in 2025 cuts off mid-1st-half at 38-26 vs a 97-61 final). These are not full game_id-level gaps but partial-coverage games; treat `game_id` presence in pbp as necessary but not sufficient for full-game coverage.
- `shooting_play` counts: {False: 1923895, True: 991836}
- `score_value` counts: {0: 1923895, 1: 253589, 2: 445829, 3: 292418} (0=non-scoring event, 1=FT, 2=2pt FG, 3=3pt FG; note `type_text`='JumpShot' covers both 2pt and 3pt jumpers -- distinguish makes/attempts by `score_value`, not by `type_text` alone)
- coordinate coverage: 100.0% of all rows, 100.0% of `shooting_play`==True rows have non-null `coordinate_x`/`coordinate_y` -- **coverage is essentially complete this season** (a marked improvement vs earlier seasons -- see trend across seasons below).
- clock/period fields: `period_number` (int, 1/2=halves, 3+=OT), `period_display_value` (str, e.g. '1st Half'/'2nd Half'/'OT'/'2OT'), `clock_display_value` (str 'MM:SS' counting down), `clock_minutes`/`clock_seconds` (ints, same info split out), `start_period_seconds_remaining`/`end_period_seconds_remaining`/`start_game_seconds_remaining`/`end_game_seconds_remaining` (numeric countdown clocks).
- periods-per-game distribution (from pbp max `period_number`): {2: 5948, 3: 276, 4: 42, 5: 9}
- participants coverage on shooting plays: `athlete_id_1` populated for 99.9% of `shooting_play`==True rows
- substitution events (`type_text` matching 'Substitution'/'Enters'): **829,632** rows -- present this season
- duplicate (`game_id`,`sequence_number`) pairs: **0**
- embedded market columns non-null rate: `game_spread`=100.0%, `home_team_spread`=100.0%, `home_favorite`=100.0%, `game_spread_available`=100.0%, `pregame_home_prob`=100.0%, `home_win_prob`=100.0%
  - **DEFECT: frozen placeholder values.** `game_spread`/`home_team_spread`/`home_favorite` take a single constant value this season (`game_spread`=[2.5]) and `game_spread_available` is always False -- this is NOT real betting-line data for this season, it is a dummy default. `home_win_prob`/`pregame_home_prob` remain real, varying model win-probabilities (ESPN BPI-style), not sportsbook odds.

### Combined type_id / type_text vocabulary (all seasons) -- the possession-outcome event dictionary

| type_id | type_text | total (all seasons) | present in / count by season |
|---|---|---|---|
| 558 | JumpShot | 2262285 | {2022: 436359, 2023: 450026, 2024: 461065, 2025: 462176, 2026: 452659} |
| 587 | Defensive Rebound | 1517800 | {2022: 295190, 2023: 304446, 2024: 310497, 2025: 302396, 2026: 305271} |
| 540 | MadeFreeThrow | 1138946 | {2022: 201130, 2023: 218846, 2024: 233110, 2025: 232271, 2026: 253589} |
| 572 | LayUpShot | 1131575 | {2022: 210696, 2023: 225498, 2024: 236934, 2025: 220717, 2026: 237730} |
| 584 | Substitution | 1025562 | {2025: 195930, 2026: 829632} |
| 519 | PersonalFoul | 1025550 | {2022: 193171, 2023: 205490, 2024: 206379, 2025: 204379, 2026: 216131} |
| 598 | Lost Ball Turnover | 746771 | {2022: 149731, 2023: 155236, 2024: 147524, 2025: 146825, 2026: 147455} |
| 586 | Offensive Rebound | 692612 | {2022: 129253, 2023: 138197, 2024: 147021, 2025: 143720, 2026: 134421} |
| 607 | Steal | 405286 | {2022: 76673, 2023: 79583, 2024: 81293, 2025: 83339, 2026: 84398} |
| 580 | OfficialTVTimeOut | 248470 | {2022: 46405, 2023: 49263, 2024: 50365, 2025: 50343, 2026: 52094} |
| 618 | Block Shot | 200335 | {2022: 37696, 2023: 38780, 2024: 41578, 2025: 40563, 2026: 41718} |
| 574 | DunkShot | 126608 | {2022: 23499, 2023: 24963, 2024: 25303, 2025: 24468, 2026: 28375} |
| 579 | ShortTimeOut | 126419 | {2022: 21061, 2023: 25804, 2024: 26356, 2025: 26314, 2026: 26884} |
| 449 | Dead Ball Rebound | 99032 | {2022: 14575, 2023: 14811, 2024: 12493, 2025: 18304, 2026: 38849} |
| 412 | End Period | 58496 | {2022: 10675, 2023: 11388, 2024: 11546, 2025: 11947, 2026: 12940} |
| 437 | TipShot | 32121 | {2022: 1509, 2023: 1193, 2024: 761, 2025: 9177, 2026: 19481} |
| 615 | Jumpball | 28752 | {2022: 2190, 2023: 1645, 2024: 1248, 2025: 4856, 2026: 18813} |
| 402 | End Game | 28628 | {2022: 5334, 2023: 5611, 2024: 5638, 2025: 5768, 2026: 6277} |
| 578 | RegularTimeOut | 20550 | {2022: 3446, 2023: 3784, 2024: 3780, 2025: 4720, 2026: 4820} |
| 521 | Technical Foul | 10966 | {2022: 1942, 2023: 2742, 2024: 2106, 2025: 1888, 2026: 2288} |
| 216 | Coach's Challenge (Stands) | 1163 | {2026: 1163} |
| 215 | Coach's Challenge (Overturned) | 741 | {2026: 741} |
| 0 | Not Available | 95 | {2022: 26, 2023: 69} |
| 91 | Shot | 2 | {2026: 2} |

- **24** distinct (type_id, type_text) pairs observed across seasons [2022, 2023, 2024, 2025, 2026].
- **Schema drift across seasons (vocabulary is NOT stable):**
  - `Coach's Challenge (Overturned)` present only in season(s) [2026]
  - `Coach's Challenge (Stands)` present only in season(s) [2026]
  - `Not Available` present only in season(s) [2022, 2023]
  - `Shot` present only in season(s) [2026]
  - `Substitution` present only in season(s) [2025, 2026]

## 5. Cross-check: recomputed final score from pbp scoring plays vs team_box (2025, 30 random games)

- sampled 30 distinct game_ids from pbp 2025 (seed=42); recomputed each team's final score as `sum(score_value)` over rows where `scoring_play`==True, grouped by (game_id, team_id).
- mismatches vs team_box `team_score`: **3** / 60 team-game rows
| game_id | team_id | team_name | recomputed_score | team_score | diff |
|---|---|---|---|---|---|
| 401706391 | 189 | Falcons | 61 | 61 | 0 |
| 401706391 | 2050 | Cardinals | 52 | 52 | 0 |
| 401706411 | 231 | Paladins | 90 | 90 | 0 |
| 401706411 | 2717 | Catamounts | 61 | 61 | 0 |
| 401706441 | 231 | Paladins | 82 | 82 | 0 |
| 401706441 | 2678 | Keydets | 91 | 91 | 0 |
| 401706673 | 204 | Beavers | 61 | 61 | 0 |
| 401706673 | 2351 | Lions | 78 | 82 | -4 |
| 401708398 | 2 | Tigers | 67 | 67 | 0 |
| 401708398 | 8 | Razorbacks | 60 | 60 | 0 |
| 401711625 | 526 | Eagles | 74 | 74 | 0 |
| 401711625 | 2198 | Colonels | 92 | 92 | 0 |
| 401716880 | 366 | Panthers | 55 | 55 | 0 |
| 401716880 | 2244 | Patriots | 100 | 100 | 0 |
| 401719169 | 2550 | Pirates | 61 | 61 | 0 |
| 401719169 | 2599 | Red Storm | 71 | 71 | 0 |
| 401720523 | 2692 | Wildcats | 62 | 62 | 0 |
| 401720523 | 3084 | Wolverines | 64 | 64 | 0 |
| 401720572 | 2193 | Buccaneers | 87 | 87 | 0 |
| 401720572 | 2724 | Shockers | 96 | 96 | 0 |
| 401720680 | 357 | Islanders | 56 | 56 | 0 |
| 401720680 | 2320 | Cardinals | 67 | 67 | 0 |
| 401720743 | 2443 | Privateers | 71 | 71 | 0 |
| 401720743 | 2837 | Lions | 73 | 73 | 0 |
| 401721101 | 161 | Knights | 66 | 66 | 0 |
| 401721101 | 2115 | Blue Devils | 87 | 87 | 0 |
| 401721181 | 2385 | Lakers | 69 | 69 | 0 |
| 401721181 | 2681 | Seahawks | 65 | 65 | 0 |
| 401721317 | 26 | Bruins | 73 | 73 | 0 |
| 401721317 | 2483 | Ducks | 71 | 71 | 0 |
| 401721444 | 127 | Spartans | 75 | 75 | 0 |
| 401721444 | 130 | Wolverines | 62 | 62 | 0 |
| 401722240 | 36 | Rams | 66 | 66 | 0 |
| 401722240 | 2440 | Wolf Pack | 64 | 64 | 0 |
| 401722291 | 2433 | Warhawks | 64 | 64 | 0 |
| 401722291 | 2655 | Green Wave | 80 | 80 | 0 |
| 401722588 | 71 | Braves | 107 | 107 | 0 |
| 401722588 | 546 | Eagles | 41 | 41 | 0 |
| 401724362 | 179 | Bonnies | 82 | 82 | 0 |
| 401724362 | 2325 | Explorers | 83 | 83 | 0 |
| 401724473 | 257 | Spiders | 60 | 60 | 0 |
| 401724473 | 2244 | Patriots | 64 | 64 | 0 |
| 401724848 | 52 | Seminoles | 67 | 67 | 0 |
| 401724848 | 87 | Fighting Irish | 60 | 60 | 0 |
| 401724917 | 103 | Eagles | 69 | 69 | 0 |
| 401724917 | 228 | Tigers | 78 | 78 | 0 |
| 401725514 | 58 | Bulls | 75 | 75 | 0 |
| 401725514 | 151 | Pirates | 69 | 69 | 0 |
| 401725535 | 5 | Blazers | 81 | 81 | 0 |
| 401725535 | 2226 | Owls | 75 | 76 | -1 |
| 401725770 | 248 | Cougars | 73 | 73 | 0 |
| 401725770 | 2132 | Bearcats | 64 | 64 | 0 |
| 401726060 | 290 | Eagles | 85 | 87 | -2 |
| 401726060 | 2729 | Tribe | 102 | 102 | 0 |
| 401727092 | 2145 | Mountain Lions | 87 | 87 | 0 |
| 401727092 | 2172 | Pioneers | 94 | 94 | 0 |
| 401732302 | 140 | Roos | 88 | 88 | 0 |
| 401732302 | 435 | Gallitos | 55 | 55 | 0 |
| 401745819 | 193 | RedHawks | 74 | 74 | 0 |
| 401745819 | 2006 | Zips | 76 | 76 | 0 |

- for reference, across **all** 2025 games (not just the 30-game sample): **1364** / 12572 team-game rows mismatch between recomputed pbp score and team_box team_score.

## 6. Possessions-per-game and rate gate-reference targets

Season-level gate targets (possessions/game estimated as mean over the two teams of `FGA - OREB + TOV + 0.44*FTA`, from team_box):

| season | mean poss/gm | SD poss/gm | mean pts/team/gm | mean 3PA share of FGA | mean FTA/FGA |
|---|---|---|---|---|---|
| 2022 | 68.59 | 5.88 | 70.40 | 37.9% | 0.305 |
| 2023 | 68.43 | 5.85 | 71.10 | 37.6% | 0.318 |
| 2024 | 68.91 | 5.80 | 72.96 | 37.4% | 0.331 |
| 2025 | 68.41 | 5.79 | 73.16 | 39.2% | 0.332 |
| 2026 | 68.74 | 5.89 | 75.05 | 39.6% | 0.352 |

By month (calendar month of game_date; November/December/etc. -- note season-year wraps, e.g. Nov/Dec belong to the season's first calendar year, Jan-Apr to the second):

| season | month | mean poss/gm | SD poss/gm | games |
|---|---|---|---|---|
| 2022 | 1 | 67.95 | 5.58 | 1420 |
| 2022 | 2 | 67.61 | 5.46 | 1472 |
| 2022 | 3 | 67.20 | 5.48 | 633 |
| 2022 | 4 | 66.19 | 6.31 | 4 |
| 2022 | 11 | 70.42 | 6.33 | 1343 |
| 2022 | 12 | 69.28 | 5.85 | 1093 |
| 2023 | 1 | 67.42 | 5.42 | 1472 |
| 2023 | 2 | 67.54 | 5.50 | 1398 |
| 2023 | 3 | 67.45 | 5.37 | 554 |
| 2023 | 4 | 64.77 | 2.68 | 3 |
| 2023 | 11 | 70.10 | 6.22 | 1498 |
| 2023 | 12 | 69.03 | 5.95 | 1295 |
| 2024 | 1 | 68.22 | 5.48 | 1405 |
| 2024 | 2 | 67.74 | 5.19 | 1397 |
| 2024 | 3 | 67.62 | 5.27 | 800 |
| 2024 | 4 | 67.65 | 5.92 | 6 |
| 2024 | 11 | 70.55 | 6.22 | 1415 |
| 2024 | 12 | 69.98 | 5.99 | 1217 |
| 2025 | 1 | 67.50 | 5.54 | 1456 |
| 2025 | 2 | 67.24 | 5.17 | 1375 |
| 2025 | 3 | 67.49 | 5.35 | 766 |
| 2025 | 4 | 69.14 | 5.67 | 17 |
| 2025 | 11 | 70.13 | 6.04 | 1531 |
| 2025 | 12 | 69.29 | 6.08 | 1141 |
| 2026 | 1 | 67.63 | 5.36 | 1558 |
| 2026 | 2 | 67.24 | 5.13 | 1370 |
| 2026 | 3 | 67.29 | 5.45 | 613 |
| 2026 | 4 | 70.33 | 6.88 | 13 |
| 2026 | 11 | 70.99 | 6.35 | 1570 |
| 2026 | 12 | 69.71 | 5.87 | 1175 |

Home-court margin (home_score - away_score), completed games only, non-neutral vs neutral sites:

| season | mean margin (non-neutral) | SD (non-neutral) | n (non-neutral) | mean margin (neutral) | SD (neutral) | n (neutral) |
|---|---|---|---|---|---|---|
| 2022 | 7.94 | 17.44 | 5282 | 1.09 | 13.29 | 694 |
| 2023 | 8.38 | 17.32 | 5459 | 2.77 | 13.42 | 769 |
| 2024 | 8.34 | 17.55 | 5497 | 2.64 | 13.70 | 746 |
| 2025 | 8.93 | 18.18 | 5545 | 3.08 | 12.99 | 747 |
| 2026 | 9.13 | 18.74 | 5592 | 2.77 | 13.71 | 708 |

## 7. ID hygiene

- distinct `team_id` values across all seasons (team_box): **1,070**
- `team_id`s whose `team_display_name` changed across seasons: **213**
| team_id | name history |
|---|---|
| 40 | 2024=Curry College Colonels; 2026=Curry Colonels |
| 45 | 2022=George Washington Colonials; 2023=George Washington Revolutionaries; 2024=George Washington Revolutionaries; 2025=George Washington Revolutionaries; 2026=George Washington Revolutionaries |
| 53 | 2022=Florida Tech Panthers; 2023=Florida Inst Of Tech Panthers; 2025=Florida Tech Panthers |
| 60 | 2022=Morehouse College Tigers; 2023=Morehouse Maroon Tigers; 2024=Morehouse Maroon Tigers; 2025=Morehouse Maroon Tigers; 2026=Morehouse Maroon Tigers |
| 83 | 2022=DePauw Tigers; 2024=Depauw Tigers |
| 85 | 2022=IUPUI Jaguars; 2023=IUPUI Jaguars; 2024=IUPUI Jaguars; 2025=IU Indianapolis Jaguars; 2026=IU Indianapolis Jaguars |
| 113 | 2022=UMass Minutemen; 2023=Massachusetts Minutemen; 2024=Massachusetts Minutemen; 2025=Massachusetts Minutemen; 2026=Massachusetts Minutemen |
| 129 | 2022=Saginaw Valley Cardinals; 2024=Saginaw Valley State Cardinals |
| 177 | 2024=Plattsburgh St Cardinals; 2025=Plattsburgh St Cardinals; 2026=Plattsburgh Cardinals |
| 193 | 2022=Miami (OH) Redhawks; 2023=Miami (OH) RedHawks; 2024=Miami (OH) RedHawks; 2025=Miami (OH) RedHawks; 2026=Miami (OH) RedHawks |
| 212 | 2022=John Jay College of Criminal Justice  Bloodhounds; 2023=John Jay College Bloodhounds; 2024=John Jay College Bloodhounds; 2025=John Jay College Bloodhounds; 2026=John Jay Bloodhounds |
| 224 | 2022=Westminster (UT) Griffins; 2023=Westminster (UT) Parson; 2025=Westminster (UT) Griffins; 2026=Westminster (UT) Griffins |
| 289 | 2023=Mount Saint Mary (NY) Mt St. Mary (Ny); 2024=Mount Saint Mary (NY) Mt St. Mary (Ny); 2025=Mount Saint Mary (NY) Knights; 2026=Mount St Mary Knights |
| 316 | 2022=Dickinson State Blue Hawks; 2025=Dickinson State (ND) Blue Hawks |
| 329 | 2022=Mount St. Vincent Dolphins; 2023=Mount Saint Vincent; 2024=Mount Saint Vincent |

  - sample multi-team athlete_ids in 2026: [{'athlete_id': 5112945.0, 'athlete_display_name': 'Jabez Williamson', 'team_id': 108806, 'team_name': 'Mustangs', 'game_date': datetime.date(2025, 11, 13)}, {'athlete_id': 5112945.0, 'athlete_display_name': 'Jabez Williamson', 'team_id': 3126, 'team_name': 'Evangels', 'game_date': datetime.date(2025, 11, 29)}, {'athlete_id': 5112952.0, 'athlete_display_name': 'Ronnie Lopez', 'team_id': 108806, 'team_name': 'Mustangs', 'game_date': datetime.date(2025, 11, 13)}, {'athlete_id': 5112952.0, 'athlete_display_name': 'Ronnie Lopez', 'team_id': 3126, 'team_name': 'Evangels', 'game_date': datetime.date(2025, 11, 29)}, {'athlete_id': 5116505.0, 'athlete_display_name': 'Ikechi Ajukwa', 'team_id': 494, 'team_name': 'Toppers', 'game_date': datetime.date(2025, 11, 3)}, {'athlete_id': 5116505.0, 'athlete_display_name': 'Ikechi Ajukwa', 'team_id': 126824, 'team_name': 'Pomeroys', 'game_date': datetime.date(2025, 12, 2)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 11, 3)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 11, 6)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 11, 10)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 11, 14)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 11, 18)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 11, 21)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 11, 23)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 11, 28)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 12, 1)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 12, 5)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 502, 'team_name': 'Crimson Wave', 'game_date': datetime.date(2025, 12, 6)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 12, 12)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 12, 17)}, {'athlete_id': 5152658.0, 'athlete_display_name': "Pape N'Diaye", 'team_id': 2752, 'team_name': 'Musketeers', 'game_date': datetime.date(2025, 12, 20)}]
- `athlete_id` appearing under >1 `team_id` within the same season (mid-season team change; true in-season D-I transfers are not supposed to happen under NCAA eligibility rules, so a non-zero count here is either a rare real case -- e.g. a withdrawal/re-enrollment -- or an athlete_id collision/reuse):

| season | athlete_ids on >1 team_id |
|---|---|
| 2022 | 73 |
| 2023 | 28 |
| 2024 | 10 |
| 2025 | 23 |
| 2026 | 21 |

## 8. Notable defects / flags summary

- schedules 2022: 6 games have null home_conference_id even though the home team can be a D-I school (data-gap, not real non-D-I) -- spot check before using conference_id as a hard D-I filter on the home side.
- schedules 2022: home_linescores/away_linescores columns are absent from the schedule schema this season (present from 2023 onward)
- schedules 2023: 3 games have null home_conference_id even though the home team can be a D-I school (data-gap, not real non-D-I) -- spot check before using conference_id as a hard D-I filter on the home side.
- schedules 2024: 5 games have null home_conference_id even though the home team can be a D-I school (data-gap, not real non-D-I) -- spot check before using conference_id as a hard D-I filter on the home side.
- schedules 2025: 7 games have null home_conference_id even though the home team can be a D-I school (data-gap, not real non-D-I) -- spot check before using conference_id as a hard D-I filter on the home side.
- schedules 2025: away_non_div1_team column exists but is populated for only 4 rows (inconsistent with 523 away_conference_id nulls) and is absent from other seasons -- unreliable, do not use as the D-I filter
- schedules 2026: 3 games have null home_conference_id even though the home team can be a D-I school (data-gap, not real non-D-I) -- spot check before using conference_id as a hard D-I filter on the home side.
- player_box: `active` column is a frozen/near-empty placeholder in 2022-2025 (constant False, or ~100% null in 2025) and only becomes real in 2026 -- use `did_not_play` instead for historical seasons
- player_box 2022: 4 team-games where sum(player points) != team_score (worst diff 2)
- player_box 2022: 27 team-games have summed player minutes off by >5 minutes from the 200(+25/OT) expectation
- player_box 2023: 12 team-games where sum(player points) != team_score (worst diff 4)
- player_box 2023: 24 team-games have summed player minutes off by >5 minutes from the 200(+25/OT) expectation
- player_box 2024: 18 team-games where sum(player points) != team_score (worst diff 46)
- player_box 2024: 25 team-games have summed player minutes off by >5 minutes from the 200(+25/OT) expectation
- player_box 2025: 21 team-games where sum(player points) != team_score (worst diff 14)
- player_box 2025: 22 team-games have summed player minutes off by >5 minutes from the 200(+25/OT) expectation
- player_box 2026: 14 team-games where sum(player points) != team_score (worst diff 20)
- player_box 2026: 10 team-games have summed player minutes off by >5 minutes from the 200(+25/OT) expectation
- pbp 2022: 89 games (1.5% of games present) have a truncated/partial pbp feed whose final running score never matches the schedules final score
- pbp 2022: only 12.1% of shooting plays have coordinates in the pbp table -- do not assume shot-location coverage from pbp alone
- pbp 2022: 2 duplicate (game_id, sequence_number) key collisions -- sequence_number is not a fully unique per-game key
- pbp 2023: 98 games (1.6% of games present) have a truncated/partial pbp feed whose final running score never matches the schedules final score
- pbp 2023: only 8.8% of shooting plays have coordinates in the pbp table -- do not assume shot-location coverage from pbp alone
- pbp 2023: 1 duplicate (game_id, sequence_number) key collisions -- sequence_number is not a fully unique per-game key
- pbp 2024: 101 games (1.6% of games present) have a truncated/partial pbp feed whose final running score never matches the schedules final score
- pbp 2024: only 6.3% of shooting plays have coordinates in the pbp table -- do not assume shot-location coverage from pbp alone
- pbp 2024: 2 duplicate (game_id, sequence_number) key collisions -- sequence_number is not a fully unique per-game key
- pbp 2024: game_spread/home_team_spread/home_favorite/game_spread_available are frozen placeholders (spread constant, availability flag always False) -- not usable as historical line data
- pbp 2025: 71 games (1.2% of games present) have a truncated/partial pbp feed whose final running score never matches the schedules final score
- pbp 2025: only 25.9% of shooting plays have coordinates in the pbp table -- do not assume shot-location coverage from pbp alone
- pbp 2025: 2 duplicate (game_id, sequence_number) key collisions -- sequence_number is not a fully unique per-game key
- pbp 2025: game_spread/home_team_spread/home_favorite/game_spread_available are frozen placeholders (spread constant, availability flag always False) -- not usable as historical line data
- pbp 2026: 6 games (0.1% of games present) have a truncated/partial pbp feed whose final running score never matches the schedules final score
- pbp 2026: game_spread/home_team_spread/home_favorite/game_spread_available are frozen placeholders (spread constant, availability flag always False) -- not usable as historical line data
- pbp: type_text vocabulary is not stable across seasons -- 'Substitution' events only start appearing in 2025+, 'Not Available' placeholder disappears after 2024, and 2026 adds new event types ('Shot', "Coach's Challenge (...)"). Any downstream feature relying on a fixed event-type set will silently break on older/newer seasons.
- pbp/team_box 2025: 1364 team-games where sum(scoring_play score_value) != team_box team_score
- team_id -> name is not fully stable: 213 team_id(s) show a display-name change across seasons (rebrand or ID reuse) -- see Section 7 table
- player_box: athlete_id reuse across teams within a season detected in some seasons -- counts: [[2022, 73], [2023, 28], [2024, 10], [2025, 23], [2026, 21]]
