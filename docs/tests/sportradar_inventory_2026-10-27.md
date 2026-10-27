# Sportradar men's pbp (BigQuery public dataset) inventory (2026-10-27)

Research-only data (user ruling 2026-10-27): never synced to the public HF dataset or rfq-collab. Descriptive; nothing is fit or adopted. Script `scripts/diag_sportradar_inventory_v1.py`; tables are also in `docs/tests/sportradar_inventory_2026-10-27.json`. Files `data/raw/bigquery_ncaa/mbb_pbp_sr/mbb_pbp_sr_<season>.parquet`, season 2013 = 2013-14 ... 2017 = 2017-18. FGA = `event_type` in twopointmade / twopointmiss / threepointmade / threepointmiss, de-duplicated on `event_id`. Distances are feet from the textbook basket (5.25 ft from the baseline, mid-width) after folding the court onto one half.

## 1. Files as delivered

| season | rows | games | date_min | date_max | teams | fga_rows_raw | fga_rows | fga_duplicate_event_ids_dropped | fga_per_game | located_pct_fga | distinct_x | distinct_y | xy_integer_pct | zero_zero_rows | three_flag_vs_event_agree_pct | team_basket_null_pct_fga |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2013 | 636362 | 1745 | 2013-11-08 | 2014-04-07 | 349 | 194962 | 194962 | 0 | 111.726 | 100 | 650 | 417 | 100 | 0 | 100 | 0 |
| 2014 | 837937 | 2375 | 2014-11-14 | 2015-04-06 | 366 | 258353 | 258353 | 0 | 108.780 | 100 | 718 | 583 | 100 | 0 | 100 | 0 |
| 2015 | 961955 | 2598 | 2015-11-13 | 2016-04-04 | 384 | 301516 | 301516 | 0 | 116.057 | 100 | 815 | 585 | 100 | 0 | 100 | 0 |
| 2016 | 808550 | 2210 | 2016-11-11 | 2017-04-03 | 363 | 255966 | 255966 | 0 | 115.822 | 100 | 824 | 587 | 100 | 0 | 100 | 0 |
| 2017 | 915589 | 2235 | 2017-11-10 | 2018-04-02 | 361 | 261175 | 260205 | 970 | 116.423 | 100 | 788 | 580 | 100 | 0 | 100 | 0 |

Games per season are 1,745-2,598, roughly a third of Division I (CBBD has about 6,000 per season): Sportradar here is a sample of games, and the file includes non-D-I exhibition opponents. The 2017-18 file holds 970 exact duplicate FGA rows in five games (same event_id, different load_timestamp), removed.

### Vocabularies

**event_type (all rows, count)**

| season | rebound | twopointmiss | turnover | twopointmade | freethrowmade | assist | threepointmiss | personalfoul | shootingfoul | threepointmade | freethrowmiss | block | attemptblocked | tvtimeout | teamtimeout | deadball | offensivefoul | lineupchange | endperiod | opentip | officialtimeout | openinbound | jumpball | technicalfoul | review | flagrantone | kickball | substitution | possession | laneviolation | flagranttwo | minortechnicalfoul | delay | clearpathfoul | ejection | defensivethreeseconds | freethrow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2013 | 122672 | 68341 | 62749 | 63380 | 52993 | 44322 | 41743 | 34072 | 26450 | 21498 | 23138 | 13562 | 13562 | 11114 | 11853 | 9642 | 4757 | 0 | 3626 | 1486 | 3061 | 0 | 815 | 764 | 11 | 328 | 216 | 0 | 97 | 51 | 26 | 0 | 8 | 12 | 12 | 1 | 0 |
| 2014 | 163023 | 90080 | 86946 | 81186 | 64224 | 59407 | 57597 | 47925 | 29861 | 29490 | 28908 | 17340 | 17340 | 15951 | 15853 | 12027 | 7214 | 0 | 4929 | 2357 | 2823 | 0 | 1043 | 1144 | 445 | 354 | 231 | 1 | 144 | 42 | 28 | 0 | 9 | 5 | 10 | 0 | 0 |
| 2015 | 188803 | 101105 | 95614 | 94540 | 75591 | 69211 | 69429 | 53592 | 36097 | 36442 | 32466 | 18857 | 18857 | 19255 | 13032 | 13259 | 9234 | 0 | 5374 | 2686 | 674 | 2470 | 1561 | 1534 | 749 | 374 | 416 | 465 | 149 | 63 | 24 | 0 | 17 | 5 | 7 | 1 | 2 |
| 2016 | 157536 | 83766 | 82228 | 80026 | 62421 | 59181 | 59914 | 42089 | 31493 | 32260 | 25675 | 16064 | 16064 | 16277 | 11390 | 10529 | 7792 | 0 | 4545 | 2286 | 571 | 2162 | 1196 | 1254 | 1003 | 290 | 446 | 0 | 0 | 46 | 14 | 0 | 12 | 8 | 10 | 2 | 0 |
| 2017 | 158456 | 82130 | 82629 | 81516 | 61408 | 61282 | 63252 | 65935 | 59640 | 34277 | 24069 | 16040 | 16040 | 17363 | 11485 | 9792 | 19716 | 35926 | 4654 | 2318 | 399 | 2215 | 1542 | 1354 | 1162 | 432 | 351 | 0 | 0 | 40 | 26 | 89 | 14 | 23 | 14 | 0 | 0 |

**type (all rows)**

| season | fieldgoal | rebound | freethrow | personalfoul | (null) | turnover | twopointmade | twopointmiss | steal | threepointmade | fouldrawn | assist | offensivefoul | threepointmiss | technicalfoul | block | attemptblocked | flagrantfoul | ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2013 | 194962 | 122672 | 76131 | 65849 | 41993 | 41645 | 26240 | 25932 | 21104 | 18082 | 1 | 0 | 0 | 1192 | 382 | 0 | 0 | 177 | 0 |
| 2014 | 258353 | 163023 | 93132 | 85768 | 55865 | 58001 | 34486 | 33060 | 28945 | 24921 | 0 | 0 | 0 | 1620 | 572 | 0 | 0 | 191 | 0 |
| 2015 | 301515 | 188802 | 108057 | 99894 | 60175 | 64402 | 38843 | 35970 | 31212 | 30368 | 0 | 0 | 0 | 1744 | 767 | 0 | 0 | 199 | 7 |
| 2016 | 255966 | 157536 | 88096 | 82161 | 50465 | 55667 | 32348 | 30606 | 26561 | 26833 | 0 | 0 | 0 | 1522 | 627 | 0 | 0 | 152 | 10 |
| 2017 | 261175 | 158456 | 85477 | 80549 | 87261 | 55788 | 27059 | 24748 | 26841 | 22771 | 57492 | 11452 | 8243 | 1474 | 766 | 2929 | 2929 | 165 | 14 |

**shot_type (FGA rows)**

| season | jump shot | layup | (null) | dunk | hook shot | tip shot |
|---|---|---|---|---|---|---|
| 2013 | 1 | 0 | 194961 | 0 | 0 | 0 |
| 2014 | 166045 | 82285 | 0 | 10023 | 0 | 0 |
| 2015 | 193186 | 97176 | 1 | 11153 | 0 | 0 |
| 2016 | 158942 | 78469 | 0 | 9982 | 5579 | 2994 |
| 2017 | 161307 | 79540 | 0 | 10815 | 5429 | 3114 |

**shot_subtype (FGA rows)**

| season | (null) | driving | pullup | floating | putback | turnaround | reverse | step back | fadeaway | alley-oop | finger roll |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2013 | 194962 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2014 | 258353 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2015 | 301516 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2016 | 225266 | 9944 | 7000 | 3800 | 2945 | 1476 | 1498 | 1346 | 1018 | 970 | 703 |
| 2017 | 217134 | 15062 | 7773 | 5000 | 4179 | 3358 | 1899 | 2030 | 1475 | 1366 | 929 |

**Derived class (FGA rows; rim = shot_type layup/dunk/tip or description text, jump2 includes hook shots, 2013 has no shot type)**

| season | three | rim | jump2 | two_untyped |
|---|---|---|---|---|
| 2013 | 63241 | 0 | 0 | 131721 |
| 2014 | 87087 | 92308 | 78958 | 0 |
| 2015 | 105871 | 108336 | 87309 | 0 |
| 2016 | 92174 | 91445 | 72347 | 0 |
| 2017 | 97131 | 93469 | 69605 | 0 |

**Derived rim sub-type (FGA rows)**

| season | (none) | layup | dunk | tip |
|---|---|---|---|---|
| 2013 | 194962 | 0 | 0 | 0 |
| 2014 | 166045 | 82285 | 10023 | 0 |
| 2015 | 193176 | 97161 | 11179 | 0 |
| 2016 | 164521 | 78469 | 9982 | 2994 |
| 2017 | 166736 | 79540 | 10815 | 3114 |

## 2. Coordinates and the frame

| season | x_min | x_max | y_min | y_max | x_p01 | x_p99 | y_p01 | y_p99 | located_pct_fga |
|---|---|---|---|---|---|---|---|---|---|
| 2013 | 13 | 1120 | 8 | 589 | 61 | 1062 | 29 | 564 | 100 |
| 2014 | 2 | 1123 | 6 | 593 | 60 | 1062 | 27 | 565 | 100 |
| 2015 | 0 | 1123 | 6 | 595 | 54 | 1069 | 26 | 565 | 100 |
| 2016 | 5 | 1119 | 6 | 593 | 51 | 1071 | 24 | 567 | 100 |
| 2017 | 2 | 1124 | 6 | 593 | 53 | 1075 | 26 | 568 | 100 |

Frame facts per season (inches unless the name says ft):

| season | dunks | x_right_half_pct | team_basket_matches_x_side_pct | team_games_p1_p2 | switch_at_half_team_basket_pct | switch_at_half_coord_side_pct | home_attacks_right_p1_pct | dunk_centroid_x_in | dunk_centroid_y_in | dunk_centroid_x_ft | dunk_centroid_y_ft | dunk_median_dist_textbook_ft | dunk_median_dist_at_own_centroid_ft | dunk_dist_q10_q25_q50_q75_q90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2013 | 0 | 50.046 | 99.800 | 3488 | 99.427 | 99.455 | 54.782 |  |  |  |  |  |  |  |
| 2014 | 10023 | 50.133 | 99.736 | 4750 | 98.737 | 98.758 | 55.678 | 92 | 296 | 7.667 | 24.667 | 3.030 | 1.924 | [1.24, 2.02, 3.03, 4.42, 6.01] |
| 2015 | 11179 | 50.013 | 99.919 | 5196 | 100 | 100 | 58.750 | 84 | 297 | 7 | 24.750 | 2.417 | 1.583 | [0.9, 1.48, 2.42, 3.64, 4.75] |
| 2016 | 9982 | 50.041 | 99.911 | 4420 | 99.842 | 99.842 | 58.563 | 88 | 296 | 7.333 | 24.667 | 2.800 | 1.833 | [1.05, 1.77, 2.8, 3.98, 5.11] |
| 2017 | 10815 | 50.265 | 99.789 | 4470 | 99.754 | 99.754 | 59.344 | 87 | 297 | 7.250 | 24.750 | 2.534 | 1.618 | [1.0, 1.62, 2.53, 3.6, 4.69] |


### Three-point arc check (arc of the era = 20.75 ft; the 22.15 ft modern arc is shown for reference only)

| season | n_three | n_two | three_inside_20.75_pct | three_inside_19.75_pct(1ft slack) | three_dist_q01_q05_q50 | two_outside_20.75_pct | two_outside_21.75_pct(1ft slack) | three_inside_22.15_pct(modern arc, reference) |
|---|---|---|---|---|---|---|---|---|
| 2013 | 63241 | 131721 | 1.210 | 1.021 | [19.66, 21.44, 22.84] | 0.810 | 0.677 | 22.694 |
| 2014 | 87087 | 171266 | 1.176 | 1.068 | [19.26, 21.57, 23.2] | 0.778 | 0.685 | 16.172 |
| 2015 | 105871 | 195645 | 0.417 | 0.328 | [21.07, 21.54, 23.1] | 0.456 | 0.359 | 18.291 |
| 2016 | 92174 | 163792 | 0.394 | 0.279 | [21.05, 21.62, 23.33] | 0.397 | 0.303 | 14.892 |
| 2017 | 97131 | 163074 | 0.240 | 0.167 | [21.2, 21.69, 23.34] | 0.350 | 0.266 | 14.294 |


### Per-team located attempts per season (offence = shooter's team; defence = the opponent)

| season | side | teams | min | p10 | p50 | p90 | teams_under_100 |
|---|---|---|---|---|---|---|---|
| 2013 | offence | 349 | 42 | 58 | 171 | 1814.400 | 68 |
| 2013 | defence | 349 | 43 | 58.800 | 170 | 1799.200 | 66 |
| 2014 | offence | 366 | 43 | 106.500 | 318 | 1826 | 30 |
| 2014 | defence | 366 | 41 | 104 | 313.500 | 1829.500 | 31 |
| 2015 | offence | 384 | 45 | 105.300 | 360 | 1976.600 | 34 |
| 2015 | defence | 384 | 45 | 110.600 | 351 | 1978.400 | 36 |
| 2016 | offence | 363 | 46 | 108.400 | 336 | 1805.600 | 32 |
| 2016 | defence | 363 | 44 | 109.400 | 336 | 1827.800 | 32 |
| 2017 | offence | 361 | 50 | 114 | 307 | 1843 | 29 |
| 2017 | defence | 361 | 47 | 111 | 311 | 1853 | 29 |


## 3. Match to the ESPN hoopR history files

Sportradar season s is the season starting in autumn of s; the ESPN file `play_by_play_<s+1>.parquet` is the same season. Games match on the UNORDERED team pair (so neutral-site home/away swaps do not matter) and date within one day (Sportradar `scheduled_date` converted UTC -> America/New_York; ESPN `game_date`). Names are matched after a normaliser (lower case, `&` -> `and`, apostrophes and dots dropped, `Saint`/`State` -> `st`, non-alphanumerics dropped) plus an explicit alias map for the Division I names the two vendors spell differently (the map is in the json under `name_aliases`: Arkansas-Little Rock = Little Rock, Brigham Young = BYU, Bryant University = Bryant, Central Connecticut State = Central Connecticut, Citadel = The Citadel, Connecticut = UConn, Detroit = Detroit Mercy, Grambling State = Grambling, Houston Baptist = Houston Christian, IPFW = Purdue Fort Wayne, Illinois-Chicago = UIC, LIU Brooklyn = Long Island University, ...). 'Resolvable' means both Sportradar team names exist in that ESPN file's name universe (the rest are mostly non-D-I exhibition opponents).

| sr_season | espn_file | sr_games | sr_games_both_names_in_espn | matched | espn_games | dup_espn_claims_dropped | match_rate_all_sr_games_pct | match_rate_resolvable_pct | matched_date_diff_0_pct | matched_games_espn_fga_located_gt0 | matched_games_espn_fga_located_ge50pct | matched_games_espn_fga_located_ge90pct | espn_fga_in_matched_games | espn_fga_located_in_matched_games | espn_located_pct_matched | espn_located_pct_all_games | espn_dunk_median_dist_ft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2013 | 2014 | 1745 | 1734 | 1594 | 5084 | 0 | 91.347 | 91.926 | 99.875 | 981 | 981 | 981 | 177292 | 108787 | 61.360 | 22.740 | 2.236 |
| 2014 | 2015 | 2375 | 2352 | 2213 | 5280 | 0 | 93.179 | 94.090 | 99.819 | 1151 | 1151 | 1151 | 244228 | 126434 | 51.769 | 22.432 | 2.236 |
| 2015 | 2016 | 2598 | 2550 | 2417 | 5390 | 0 | 93.033 | 94.784 | 99.793 | 1159 | 1159 | 1159 | 280296 | 134984 | 48.158 | 22.425 | 2.236 |
| 2016 | 2017 | 2210 | 2177 | 2060 | 5479 | 0 | 93.213 | 94.626 | 99.757 | 955 | 955 | 955 | 238351 | 111006 | 46.572 | 18.511 | 2.236 |
| 2017 | 2018 | 2235 | 2207 | 2155 | 5645 | 0 | 96.421 | 97.644 | 99.582 | 860 | 860 | 860 | 235376 | 95739 | 40.675 | 16.688 | 2.236 |


Sportradar locates every attempt it records, so the number of games BOTH vendors locate equals the number of matched games in which ESPN locates something: `matched_games_espn_fga_located_gt0` (any), `_ge50pct`, `_ge90pct`. The play-level counts (plays both vendors locate) are in the rim-asymmetry doc, part (e).


## 4. Bands (PART 3): short2 / long2 and corner3 / wing3 / top3, per team-season and drift 2013-2017

Band definitions are section 2 of `docs/models/shot_location/experiments_DRAFT_shot_location_stage_2026-10-27.md`, applied in the folded frame with the textbook basket: short2 d <= 10 ft, long2 d > 10 ft (jump2 = non-rim two-point attempts, hook shots included); corner3 = within 4.0 ft of a sideline (|lateral offset| >= 21 ft) and d <= 24 ft; wing3 = not corner and |theta| >= 30 deg from the centre line; top3 = |theta| < 30 deg. Season 2013 carries no shot type, so jump2 cannot be separated from layups and dunks: its jump2 bands are not computed (threes are). Rows are located attempts only (Sportradar locates 100%). The CBBD rows are the 2022-2025 located attempts from the coordinate-audit master tables (2026 stays sealed and is not read); CBBD located attempts are a non-random 80-99% of attempts, so they are a reference, not a truth. `teams_ge50` counts teams with at least 50 attempts in the class. Sportradar coverage is lopsided by team (per-team located attempts per season: median 171-360, p90 about 1,800-2,000, so a minority of teams have nearly a full schedule and most have a few games), so team shares are noisy for most teams. `binomial_sd_if_equal_pct` is the team-share SD expected if every team had the pooled share; `excess_sd_pct` = sqrt(max(observed var - binomial var, 0)) is the real between-team spread.

**Reading (drift).** Corner3 steps from 18.1 / 18.2% of located threes in 2013-14 and 2014-15 to 21.5 / 21.5 / 22.6% in 2015-16 to 2017-18, landing on CBBD's 21.6-21.9% for 2022-25; wing3 falls 47.9 -> 41.5% (CBBD 44.3-44.8%) and top3 stays at 34.0-35.8% (CBBD 33.6-33.8%). Short2 (non-rim twos within 10 ft) is 39.4 / 44.3 / 42.0 / 43.2% in 2014-15 to 2017-18 against CBBD's 42.8-45.7%. The step between the 2014-15 and 2015-16 files coincides with the frame shift seen in the dunk centroid (92 in in 2014-15, 84-88 in later files), so a vendor-side re-charting or recalibration is a candidate explanation as well as a real change in shot selection; the two earliest seasons are the least comparable. Team-level spread is real in both eras (`excess_sd_pct` 3.7-4.4 points for corner3, 7.4-8.2 for short2/long2 in Sportradar; CBBD 4.2-4.5 and 9.4-10.7), but Sportradar has far fewer attempts per team, so its team shares are noisy and it is a population-level, not a team-level, window.

### Pooled shares by season

| source | season | class | attempts | corner3_pct | wing3_pct | top3_pct | short2_pct | long2_pct |
|---|---|---|---|---|---|---|---|---|
| cbbd_located | 2022 | jump2 | 140453 |  |  |  | 42.791 | 57.209 |
| cbbd_located | 2023 | jump2 | 151653 |  |  |  | 42.920 | 57.080 |
| cbbd_located | 2024 | jump2 | 162194 |  |  |  | 45.153 | 54.847 |
| cbbd_located | 2025 | jump2 | 167040 |  |  |  | 45.669 | 54.331 |
| sportradar | 2014 | jump2 | 78958 |  |  |  | 39.373 | 60.627 |
| sportradar | 2015 | jump2 | 87309 |  |  |  | 44.301 | 55.699 |
| sportradar | 2016 | jump2 | 72347 |  |  |  | 42.046 | 57.954 |
| sportradar | 2017 | jump2 | 69605 |  |  |  | 43.214 | 56.786 |
| cbbd_located | 2022 | three | 206705 | 21.639 | 44.800 | 33.561 |  |  |
| cbbd_located | 2023 | three | 224841 | 21.592 | 44.786 | 33.622 |  |  |
| cbbd_located | 2024 | three | 238613 | 21.842 | 44.458 | 33.700 |  |  |
| cbbd_located | 2025 | three | 278323 | 21.875 | 44.344 | 33.781 |  |  |
| sportradar | 2013 | three | 63241 | 18.135 | 47.880 | 33.984 |  |  |
| sportradar | 2014 | three | 87087 | 18.175 | 46.681 | 35.144 |  |  |
| sportradar | 2015 | three | 105871 | 21.523 | 43.889 | 34.587 |  |  |
| sportradar | 2016 | three | 92174 | 21.515 | 43.981 | 34.504 |  |  |
| sportradar | 2017 | three | 97131 | 22.645 | 41.525 | 35.830 |  |  |


### Team-season share distribution (teams with >= 50 attempts in the class)

| source | season | class | band | teams_ge50 | median_attempts_per_team | share_p10_pct | share_median_pct | share_p90_pct | share_min_pct | team_share_sd_pct | binomial_sd_if_equal_pct | excess_sd_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cbbd_located | 2022 | jump2 | long2 | 364 | 385.500 | 43.111 | 57.408 | 68.759 | 25.688 | 10.054 | 2.830 | 9.647 |
| cbbd_located | 2023 | jump2 | long2 | 373 | 408 | 43.458 | 57.213 | 68.347 | 28.723 | 9.819 | 2.818 | 9.405 |
| cbbd_located | 2024 | jump2 | long2 | 376 | 421.500 | 40.100 | 55.503 | 67.031 | 22.193 | 10.423 | 2.739 | 10.057 |
| cbbd_located | 2025 | jump2 | long2 | 403 | 418 | 39.661 | 54.101 | 67.872 | 13.968 | 11.151 | 3.056 | 10.724 |
| sportradar | 2014 | jump2 | long2 | 275 | 170 | 48.567 | 61.905 | 71.311 | 33.696 | 9.226 | 4.321 | 8.151 |
| sportradar | 2015 | jump2 | long2 | 307 | 136 | 46.113 | 56.604 | 67.341 | 25.926 | 8.570 | 4.350 | 7.384 |
| sportradar | 2016 | jump2 | long2 | 268 | 149.500 | 45.837 | 58.739 | 68.749 | 26.667 | 9.302 | 4.356 | 8.219 |
| sportradar | 2017 | jump2 | long2 | 271 | 147 | 45.333 | 57.143 | 67.606 | 29.664 | 8.868 | 4.426 | 7.685 |
| cbbd_located | 2022 | jump2 | short2 | 364 | 385.500 | 31.241 | 42.592 | 56.889 | 13.846 | 10.054 | 2.830 | 9.647 |
| cbbd_located | 2023 | jump2 | short2 | 373 | 408 | 31.653 | 42.787 | 56.542 | 13.924 | 9.819 | 2.818 | 9.405 |
| cbbd_located | 2024 | jump2 | short2 | 376 | 421.500 | 32.969 | 44.497 | 59.900 | 11.864 | 10.423 | 2.739 | 10.057 |
| cbbd_located | 2025 | jump2 | short2 | 403 | 418 | 32.128 | 45.899 | 60.339 | 11.765 | 11.151 | 3.056 | 10.724 |
| sportradar | 2014 | jump2 | short2 | 275 | 170 | 28.689 | 38.095 | 51.433 | 16.379 | 9.226 | 4.321 | 8.151 |
| sportradar | 2015 | jump2 | short2 | 307 | 136 | 32.659 | 43.396 | 53.887 | 13.115 | 8.570 | 4.350 | 7.384 |
| sportradar | 2016 | jump2 | short2 | 268 | 149.500 | 31.251 | 41.261 | 54.163 | 19.298 | 9.302 | 4.356 | 8.219 |
| sportradar | 2017 | jump2 | short2 | 271 | 147 | 32.394 | 42.857 | 54.667 | 20 | 8.868 | 4.426 | 7.685 |
| cbbd_located | 2022 | three | corner3 | 381 | 571 | 15.502 | 21.892 | 27.866 | 8 | 5.011 | 2.259 | 4.473 |
| cbbd_located | 2023 | three | corner3 | 389 | 615 | 15.208 | 21.573 | 27.693 | 6.780 | 4.725 | 2.234 | 4.164 |
| cbbd_located | 2024 | three | corner3 | 393 | 640 | 15.852 | 21.521 | 28.081 | 7.345 | 4.922 | 2.221 | 4.392 |
| cbbd_located | 2025 | three | corner3 | 424 | 703 | 15.293 | 21.888 | 27.880 | 3.846 | 4.956 | 2.401 | 4.335 |
| sportradar | 2013 | three | corner3 | 204 | 125.500 | 10.811 | 17.366 | 23.502 | 3.704 | 5.126 | 3.559 | 3.689 |
| sportradar | 2014 | three | corner3 | 297 | 155 | 10.464 | 17.179 | 24.596 | 1.205 | 5.525 | 3.332 | 4.408 |
| sportradar | 2015 | three | corner3 | 330 | 156 | 14.147 | 20.988 | 27.695 | 6.818 | 5.261 | 3.387 | 4.027 |
| sportradar | 2016 | three | corner3 | 311 | 144 | 14.607 | 20.792 | 27.723 | 6.422 | 5.435 | 3.547 | 4.119 |
| sportradar | 2017 | three | corner3 | 304 | 155.500 | 14.779 | 21.374 | 28.280 | 6.579 | 5.466 | 3.512 | 4.188 |
| cbbd_located | 2022 | three | top3 | 381 | 571 | 26.942 | 33.678 | 39.305 | 10 | 5.258 | 2.591 | 4.576 |
| cbbd_located | 2023 | three | top3 | 389 | 615 | 27.502 | 33.185 | 40.354 | 14.103 | 5.185 | 2.565 | 4.506 |
| cbbd_located | 2024 | three | top3 | 393 | 640 | 27.952 | 33.654 | 39.796 | 20 | 4.795 | 2.541 | 4.066 |
| cbbd_located | 2025 | three | top3 | 424 | 703 | 26.878 | 33.573 | 39.988 | 16.071 | 5.267 | 2.747 | 4.493 |
| sportradar | 2013 | three | top3 | 204 | 125.500 | 26.169 | 33.847 | 43.193 | 13.462 | 7.311 | 4.375 | 5.857 |
| sportradar | 2014 | three | top3 | 297 | 155 | 28.597 | 34.851 | 45.051 | 14.035 | 6.487 | 4.125 | 5.007 |
| sportradar | 2015 | three | top3 | 330 | 156 | 27.436 | 34.999 | 42.505 | 16.667 | 5.986 | 3.919 | 4.524 |
| sportradar | 2016 | three | top3 | 311 | 144 | 28.125 | 34.444 | 42.361 | 16.418 | 5.775 | 4.103 | 4.064 |
| sportradar | 2017 | three | top3 | 304 | 155.500 | 28.652 | 35.930 | 43.433 | 19.084 | 5.721 | 4.024 | 4.066 |
| cbbd_located | 2022 | three | wing3 | 381 | 571 | 39.299 | 44.828 | 50.177 | 30.377 | 4.290 | 2.728 | 3.311 |
| cbbd_located | 2023 | three | wing3 | 389 | 615 | 40.115 | 44.700 | 50.694 | 26.582 | 4.621 | 2.700 | 3.750 |
| cbbd_located | 2024 | three | wing3 | 393 | 640 | 39.717 | 44.556 | 49.500 | 33.228 | 3.986 | 2.671 | 2.959 |
| cbbd_located | 2025 | three | wing3 | 424 | 703 | 39.429 | 44.393 | 50.716 | 30.435 | 4.698 | 2.886 | 3.707 |
| sportradar | 2013 | three | wing3 | 204 | 125.500 | 40.977 | 48.211 | 56.020 | 23.333 | 6.723 | 4.615 | 4.889 |
| sportradar | 2014 | three | wing3 | 297 | 155 | 39.321 | 46.586 | 55.047 | 31.655 | 6.404 | 4.311 | 4.737 |
| sportradar | 2015 | three | wing3 | 330 | 156 | 37.820 | 43.600 | 50.993 | 27.473 | 5.589 | 4.089 | 3.811 |
| sportradar | 2016 | three | wing3 | 311 | 144 | 37.802 | 44.068 | 51.190 | 21.839 | 5.962 | 4.284 | 4.147 |
| sportradar | 2017 | three | wing3 | 304 | 155.500 | 35.297 | 41.878 | 49.532 | 26.415 | 5.386 | 4.135 | 3.451 |
