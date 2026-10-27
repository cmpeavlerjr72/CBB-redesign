# GameZone (gamezoneR-data) inventory (2026-10-27)

Private internal research data (user ruling 2026-10-27): research input only, not synced to the public HF dataset, not in rfq-collab. Descriptive; nothing is fit or adopted. Script `scripts/diag_gamezone_inventory_v1.py`; pull `scripts/pull_gamezone_v1.py` (16 `.rds` files read with pyreadr 0.5.8, all converted, no failures). Tables are also in `docs/tests/gamezone_inventory_2026-10-27.json`. Season = ending year. "File season" = file of origin; "date season" = from the game date (Aug or later = next year). Attempt cells under 500 are marked UNDERPOWERED; game-level cells in section 5 hold about 130 attempts per game, so a cell is flagged only below 5 games.

## 1. Files as delivered

Rows are all pbp rows (lineups, subs, fouls, shots). FGA = rows with a shot outcome and `free_throw == False`. Free-throw rows never carry coordinates. "Window" = 25 Oct (y-1) to 15 Apr (y).

| file_season | rows | games | date_min | date_max | games_outside_window | games_dateseason_ne_file | fga_rows | fga_located_pct | ft_rows | ft_located_pct | fga_games | games_ge90_located_pct | games_zero_located |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2009 | 77150 | 312 | 2009-02-19 | 2009-03-15 | 0 | 0 | 34847 | 99.983 | 11815 | 0 | 312 | 100 | 0 |
| 2010 | 469694 | 1930 | 2009-11-09 | 2010-04-05 | 0 | 0 | 217420 | 99.972 | 78892 | 0 | 1930 | 100 | 0 |
| 2011 | 462010 | 1904 | 2010-11-08 | 2013-12-27 | 1 | 1 | 212330 | 99.993 | 76137 | 0 | 1904 | 100 | 0 |
| 2012 | 514459 | 1967 | 2011-11-07 | 2013-12-27 | 1 | 1 | 216506 | 99.985 | 76056 | 0 | 1967 | 100 | 0 |
| 2013 | 544432 | 1976 | 2012-11-09 | 2013-04-06 | 0 | 0 | 219501 | 99.978 | 76267 | 0 | 1976 | 100 | 0 |
| 2014 | 637821 | 2197 | 2013-11-08 | 2014-04-07 | 0 | 0 | 245447 | 99.967 | 97278 | 0 | 2197 | 100 | 0 |
| 2015 | 620635 | 2195 | 2013-12-27 | 2015-04-06 | 1 | 1 | 241108 | 99.967 | 86256 | 0 | 2195 | 100 | 0 |
| 2016 | 717500 | 2223 | 2013-12-27 | 2016-04-04 | 1 | 1 | 259693 | 99.973 | 92579 | 0 | 2223 | 100 | 0 |
| 2017 | 685571 | 2147 | 2013-12-27 | 2017-04-03 | 1 | 1 | 248279 | 99.979 | 84349 | 0 | 2147 | 100 | 0 |
| 2018 | 690289 | 2182 | 2017-11-10 | 2018-04-02 | 0 | 0 | 252659 | 99.981 | 82469 | 0 | 2182 | 100 | 0 |
| 2019 | 705927 | 2205 | 2018-11-06 | 2019-04-08 | 0 | 0 | 255276 | 99.987 | 83772 | 0 | 2205 | 100 | 0 |
| 2020 | 671978 | 2127 | 2019-11-05 | 2020-03-11 | 0 | 0 | 245853 | 99.991 | 78292 | 0 | 2127 | 100 | 0 |
| 2021 | 534716 | 1620 | 2020-11-25 | 2021-04-05 | 0 | 0 | 188448 | 99.992 | 58244 | 0 | 1620 | 100 | 0 |
| 2022 | 748083 | 2240 | 2021-11-09 | 2022-04-04 | 0 | 0 | 260164 | 99.983 | 76791 | 0 | 2240 | 100 | 0 |
| 2023 | 773600 | 2235 | 2022-11-07 | 2023-04-03 | 0 | 0 | 258945 | 99.988 | 78557 | 0 | 2235 | 100 | 0 |
| 2024 | 458814 | 1294 | 2023-11-06 | 2024-01-15 | 0 | 0 | 153823 | 99.969 | 49156 | 0 | 1294 | 100 | 0 |

Dates outside the window: file 2011: 2013-12-27; file 2012: 2013-12-27; file 2015: 2013-12-27; file 2016: 2013-12-27; file 2017: 2013-12-27.

Duplicate game_ids across files: game 1360559 dated 2013-12-27 (date season 2014) appears in files [2011, 2012, 2014, 2015, 2016, 2017]. Each duplicate is a single game copied into other files; that one stray game is what makes the minimum date of several files look wrong (e.g. 2013-12-27 in the 2014-15 file). It is not mixed-season data: after de-duplicating on game_id (keep the copy in the file matching the date season) no date/season mismatch remains. The 2008-09 file holds only part of a season (see its date range); 2019-20 ends in March 2020; 2023-24 ends 15 Jan 2024 (feed stops).

## 2. Shot-type vocabulary (percent of FGA rows per file; `three_pt_pct` = vendor three flag)

Vocabulary: Jump Shot, Layup Shot, Dunk Shot, Hook Shot, Tip Shot (plus 3 `Slam Dunk Shot` rows in all files and a few null `shot_desc`). Threes are flagged by `three_pt`; a few non-jumpers carry that flag (json `three_flag_on_nonjump`).

| file_season | (null) | Dunk Shot | Hook Shot | Jump Shot | Layup Shot | Slam Dunk Shot | Tip Shot | three_pt_pct |
|---|---|---|---|---|---|---|---|---|
| 2009 | 0.020 | 3.790 | 2.320 | 66.720 | 25.570 | 0 | 1.600 | 33.395 |
| 2010 | 0.030 | 3.910 | 2.400 | 64.590 | 27.190 | 0 | 1.870 | 32.687 |
| 2011 | 0.010 | 3.620 | 2.720 | 65.370 | 26.420 | 0 | 1.860 | 32.992 |
| 2012 | 0.020 | 3.820 | 2.610 | 65.080 | 26.670 | 0 | 1.800 | 32.723 |
| 2013 | 0.020 | 3.970 | 2.350 | 64.990 | 26.870 | 0 | 1.790 | 32.819 |
| 2014 | 0.030 | 3.880 | 2.130 | 62.710 | 29.470 | 0 | 1.780 | 32.722 |
| 2015 | 0.030 | 3.800 | 1.830 | 65 | 27.620 | 0 | 1.720 | 33.910 |
| 2016 | 0.030 | 3.710 | 1.590 | 65.680 | 27.340 | 0 | 1.650 | 34.912 |
| 2017 | 0.020 | 3.850 | 1.410 | 66.470 | 26.820 | 0 | 1.430 | 35.800 |
| 2018 | 0.020 | 4.200 | 1.290 | 65.860 | 27.300 | 0 | 1.330 | 37.400 |
| 2019 | 0.010 | 4.330 | 1.750 | 64.150 | 28.690 | 0 | 1.080 | 38.857 |
| 2020 | 0.010 | 3.910 | 1.850 | 64.230 | 29.160 | 0 | 0.830 | 37.609 |
| 2021 | 0.010 | 3.820 | 1.870 | 65.310 | 28.290 | 0 | 0.700 | 37.451 |
| 2022 | 0.020 | 4.300 | 2.070 | 63 | 29.990 | 0 | 0.620 | 37.630 |
| 2023 | 0.010 | 4.300 | 2 | 62.610 | 30.550 | 0 | 0.520 | 37.251 |
| 2024 | 0.030 | 4.150 | 1.820 | 61.580 | 31.950 | 0 | 0.470 | 37.557 |

## 3. Frame

Feet, half-court frame, one basket for every attempt: `loc_x` 0 to 50 across the court, `loc_y` from the baseline outward. There is no half-time side switching: attempts of both teams in both halves are placed at the same basket near the baseline (see `y_min`, dunk centroid below, and the half/home split in the json `side_check`).

Basket. The possessions v2 calibration for CBBD checked that the median dunk lands 2-2.5 ft from a stated basket. That check is degenerate here because GameZone dunks (and most rim attempts) are not spread around the rim: they sit on two canned pixels, A = (25.0, 5.25) and B = (26.0, 6.25), so the dunk median is 0 or 1.41 ft from A by construction. The per-date-season located `Dunk Shot` centroid is {'2009': [25.0, 5.25], '2010': [25.0, 5.25], '2011': [25.0, 5.25], '2012': [25.5, 5.8], '2013': [25.45, 5.75], '2014': [25.5, 5.8], '2015': [25.5, 5.75], '2016': [25.5, 5.75], '2017': [25.5, 5.75], '2018': [25.5, 5.75], '2019': [25.5, 5.8], '2020': [25.5, 5.75], '2021': [25.5, 5.75], '2022': [25.5, 5.75], '2023': [25.5, 5.75], '2024': [25.5, 5.75]} (pooled (25.442, 5.702) over 139,527 dunks): it equals pixel A in 2008-09 to 2010-11 (A dominates) and sits about 0.7 ft to the upper right of it from 2011-12 (A and B used about equally). PRIMARY basket for every GameZone distance here and in the vendor comparison = the textbook rim centre (25.0, 5.25) = pixel A; the dunk centroid is carried as a sensitivity. The text-threes-inside-the-arc yardstick (vendor comparison, part d) is lower with the textbook basket from 2012-13 on (2.2-6.6% vs 5.0-12.0% for 2012-13 onward), which supports it. Distances are therefore not on the same footing as CBBD's calibrated frame, and any band built from them inherits the vendor's frame offset. Frame by date season:

| date_season | n_located_fga | x_min | x_max | y_min | y_max | y_gt_47_pct | dunk_mean_x | dunk_mean_y | three_median_dist | three_p05_dist |
|---|---|---|---|---|---|---|---|---|---|---|
| 2009 | 34841 | 0.700 | 49.500 | 0.950 | 46.950 | 0 | 25.000 | 5.252 | 23.098 | 20.815 |
| 2010 | 217359 | 0.300 | 49.700 | 0.650 | 46.950 | 0 | 25.003 | 5.263 | 23.053 | 20.854 |
| 2011 | 212181 | 0 | 49.800 | 0.650 | 46.950 | 0 | 25.008 | 5.269 | 23.197 | 20.961 |
| 2012 | 216340 | 0 | 50 | 0.250 | 46.950 | 0 | 25.513 | 5.775 | 23.355 | 21.174 |
| 2013 | 219452 | -0.100 | 50.100 | 0.750 | 46.950 | 0 | 25.474 | 5.735 | 23.509 | 22.045 |
| 2014 | 245367 | -0.100 | 50 | 1.050 | 46.950 | 0 | 25.516 | 5.783 | 23.548 | 22.077 |
| 2015 | 240492 | -0.100 | 50.100 | 1.050 | 46.850 | 0 | 25.503 | 5.763 | 23.582 | 22.042 |
| 2016 | 259355 | -0.100 | 50.100 | 1.050 | 46.950 | 0 | 25.511 | 5.770 | 23.753 | 22.189 |
| 2017 | 248092 | -0.100 | 50.100 | 1.050 | 46.950 | 0 | 25.512 | 5.773 | 23.745 | 22.204 |
| 2018 | 252612 | -0.100 | 50.100 | 1.050 | 46.950 | 0 | 25.492 | 5.749 | 23.892 | 22.227 |
| 2019 | 255244 | 0 | 50.100 | 1.250 | 46.950 | 0 | 25.516 | 5.776 | 23.853 | 22.208 |
| 2020 | 245830 | -0.100 | 50.100 | 1.050 | 46.950 | 0 | 25.504 | 5.764 | 23.755 | 22.203 |
| 2021 | 188433 | 0 | 49.900 | 1.250 | 46.850 | 0 | 25.509 | 5.767 | 23.865 | 22.299 |
| 2022 | 260121 | -0.100 | 50 | 0.750 | 46.950 | 0 | 25.499 | 5.755 | 23.854 | 22.390 |
| 2023 | 258913 | -0.100 | 50.100 | 1.250 | 46.850 | 0 | 25.506 | 5.768 | 23.955 | 22.438 |
| 2024 | 153776 | -0.100 | 50 | 1.550 | 46.850 | 0 | 25.507 | 5.766 | 23.780 | 22.319 |

Distance to the primary (textbook) basket for dunks and layups (ft; quantiles; the CBBD calibration had a dunk median of 2.27 ft):

| date_season | type | n | p05 | p10 | p25 | p50 | p75 | p90 | p95 |
|---|---|---|---|---|---|---|---|---|---|
| 2009 | Dunk | 1319 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2009 | Layup | 8909 | 0 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 |
| 2010 | Dunk | 8507 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2010 | Layup | 59115 | 0 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 |
| 2011 | Dunk | 7671 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2011 | Layup | 56073 | 0 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 |
| 2012 | Dunk | 8259 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2012 | Layup | 57706 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2013 | Dunk | 8725 | 0 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 |
| 2013 | Layup | 58976 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2014 | Dunk | 9517 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2014 | Layup | 72332 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2015 | Dunk | 9147 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2015 | Layup | 66477 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2016 | Dunk | 9615 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2016 | Layup | 70946 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2017 | Dunk | 9565 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2017 | Layup | 66552 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2018 | Dunk | 10615 | 0 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 |
| 2018 | Layup | 68967 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2019 | Dunk | 11051 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2019 | Layup | 73228 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2020 | Dunk | 9623 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2020 | Layup | 71697 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2021 | Dunk | 7192 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2021 | Layup | 53320 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2022 | Dunk | 11193 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2022 | Layup | 78021 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2023 | Dunk | 11140 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2023 | Layup | 79097 | 0 | 0 | 1.414 | 2.377 | 4.264 | 5.941 | 7.043 |
| 2024 | Dunk | 6388 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 | 1.414 |
| 2024 | Layup | 49146 | 0 | 0 | 0 | 0 | 1.414 | 1.414 | 1.414 |

### Canned rim pixels

Two pixels, A = (25.0, 5.25) and B = (26.0, 6.25), hold a large share of ALL located attempts and most rim-text attempts. Which of the two is used does not follow half, home/away or neutral (columns `A_share_*` near 50%), so the pair behaves like a rim placeholder rather than a frame switch.

| date_season | pixelA_pct_of_located | pixelB_pct_of_located | canned_pair_pct_of_rim_text | A_share_of_pair | A_share_half1 | A_share_half2 | A_share_home | A_share_away |
|---|---|---|---|---|---|---|---|---|
| 2009 | 18.174 | 11.156 | 94.696 | 61.963 | 61.154 | 62.793 | 63.905 | 59.882 |
| 2010 | 19.337 | 11.753 | 94.139 | 62.197 | 60.923 | 63.425 | 65.357 | 58.756 |
| 2011 | 18.570 | 11.440 | 93.980 | 61.879 | 60.198 | 63.464 | 65.093 | 58.419 |
| 2012 | 15.118 | 15.306 | 94.127 | 49.691 | 49.436 | 49.888 | 49.783 | 49.592 |
| 2013 | 15.391 | 15.429 | 94.308 | 49.938 | 50.006 | 49.894 | 49.807 | 50.078 |
| 2014 | 16.563 | 16.759 | 94.751 | 49.705 | 49.780 | 49.620 | 49.734 | 49.674 |
| 2015 | 15.690 | 15.687 | 94.518 | 50.005 | 49.756 | 50.311 | 50.275 | 49.722 |
| 2016 | 15.483 | 15.541 | 94.816 | 49.907 | 50.109 | 49.739 | 49.737 | 50.083 |
| 2017 | 15.138 | 15.498 | 95.351 | 49.411 | 49.426 | 49.385 | 49.162 | 49.674 |
| 2018 | 15.692 | 15.759 | 95.750 | 49.894 | 49.700 | 50.080 | 50.099 | 49.677 |
| 2019 | 16.432 | 16.510 | 96.569 | 49.881 | 50.137 | 49.637 | 49.782 | 49.985 |
| 2020 | 16.378 | 16.637 | 97.313 | 49.608 | 49.653 | 49.578 | 49.487 | 49.733 |
| 2021 | 15.903 | 16.162 | 97.662 | 49.595 | 49.647 | 49.515 | 49.956 | 49.218 |
| 2022 | 17.080 | 17.166 | 98.023 | 49.874 | 49.513 | 50.173 | 49.791 | 49.963 |
| 2023 | 8.811 | 8.950 | 50.189 | 49.610 | 49.339 | 49.818 | 49.691 | 49.522 |
| 2024 | 18.027 | 18.005 | 98.476 | 50.031 | 49.856 | 50.173 | 49.887 | 50.186 |

## 4. Match to our game universe

Our universe: CBBD `games_<season>.parquet`, status final. D1 proxy = both conferences non-null (section 5 uses `games_universe_v2` `is_d1_game`, as the audit did). CBBD games exist on disk from season 2013, so earlier GameZone files cannot be matched (NaN). Matching is on date (+0 then +-1 day; US/Eastern date of the CBBD start time) and team names in either orientation, in three tiers: (1) exact normalised names; (2) one side matches by normalised or learned name and the date window holds exactly one candidate (this learns the other side's name); (3) both names unknown: unique candidate with the same neutral flag and both-side difflib ratio >= 0.70. The learned name map has 99 non-identical entries (json key `name_map_gz_to_cbbd_normalised_nonidentical`) and 530 identical ones.

| date_season | gz_games | matched | match_rate_pct | tier1 | tier2 | tier3_fuzzy | date_offset_nonzero | our_final_d1proxy_games | our_matched | matched_share_of_ours_pct | matched_to_nonD1proxy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2009 | 312 | 0 |  | 0 | 0 | 0 | 0 | 0 | 0 |  | 0 |
| 2010 | 1930 | 0 |  | 0 | 0 | 0 | 0 | 0 | 0 |  | 0 |
| 2011 | 1903 | 0 |  | 0 | 0 | 0 | 0 | 0 | 0 |  | 0 |
| 2012 | 1966 | 0 |  | 0 | 0 | 0 | 0 | 0 | 0 |  | 0 |
| 2013 | 1976 | 1974 | 99.899 | 1762 | 212 | 0 | 2 | 5470 | 1942 | 35.503 | 32 |
| 2014 | 2197 | 2197 | 100 | 1930 | 267 | 0 | 2 | 5503 | 2148 | 39.033 | 49 |
| 2015 | 2194 | 2194 | 100 | 1927 | 267 | 0 | 5 | 5494 | 2151 | 39.152 | 43 |
| 2016 | 2222 | 2222 | 100 | 1959 | 263 | 0 | 4 | 5514 | 2186 | 39.645 | 36 |
| 2017 | 2146 | 2146 | 100 | 1915 | 231 | 0 | 4 | 5520 | 2107 | 38.170 | 39 |
| 2018 | 2182 | 2149 | 98.488 | 1921 | 228 | 0 | 8 | 5501 | 2112 | 38.393 | 37 |
| 2019 | 2205 | 2205 | 100 | 2009 | 196 | 0 | 2 | 5597 | 2171 | 38.789 | 34 |
| 2020 | 2127 | 2127 | 100 | 1928 | 199 | 0 | 0 | 5328 | 2088 | 39.189 | 39 |
| 2021 | 1620 | 1620 | 100 | 1483 | 137 | 0 | 3 | 3944 | 1579 | 40.035 | 41 |
| 2022 | 2240 | 2240 | 100 | 2032 | 208 | 0 | 4 | 5475 | 2200 | 40.183 | 40 |
| 2023 | 2235 | 2235 | 100 | 2041 | 194 | 0 | 6 | 5715 | 2207 | 38.618 | 28 |
| 2024 | 1294 | 1294 | 100 | 1144 | 150 | 0 | 3 | 5726 | 1247 | 21.778 | 47 |

`match_rate_pct` = matched GameZone games / GameZone games. `matched_share_of_ours_pct` = our D1-proxy games with a GameZone counterpart. `matched_to_nonD1proxy` = matches to CBBD games with a null conference. Unmatched GameZone games (name list and 40 examples) are in the json.

## 5. Overlap with our unlocated games, 2022-2025

Our located game = more than 50% of FGA rows located in CBBD or hoopR (audit master tables); universe = `games_universe_v2` D1 games (a game with no CBBD pbp rows counts as unlocated).

| season | d1_games | ours_unlocated | ours_unlocated_pct | gz_covers_all_games_pct | gz_covers_unlocated | gz_covers_unlocated_pct |
|---|---|---|---|---|---|---|
| 2022 | 5485 | 457 | 8.332 | 40.109 | 114 | 24.945 |
| 2023 | 5749 | 393 | 6.836 | 38.389 | 101 | 25.700 |
| 2024 | 5731 | 316 | 5.514 | 21.759 | 73 | 23.101 |
| 2025 | 5769 | 120 | 2.080 | 0 | 0 | 0 |

Breakouts (power = home conference in the audit's power list; neutral; postseason = season_type 3; March = game month 3):

| season | cell | games | gz_cover_all_pct | unlocated | unlocated_pct | gz_covers_unlocated | gz_covers_unlocated_pct | flag |
|---|---|---|---|---|---|---|---|---|
| 2022 | power=True | 1508 | 99.934 | 66 | 4.377 | 66 | 100 |  |
| 2022 | power=False | 3977 | 17.425 | 391 | 9.832 | 48 | 12.276 |  |
| 2022 | neutral=True | 688 | 53.343 | 111 | 16.134 | 44 | 39.640 |  |
| 2022 | neutral=False | 4797 | 38.211 | 346 | 7.213 | 70 | 20.231 |  |
| 2022 | postseason=True | 130 | 100 | 12 | 9.231 | 12 | 100 |  |
| 2022 | postseason=False | 5355 | 38.655 | 445 | 8.310 | 102 | 22.921 |  |
| 2022 | march=True | 633 | 51.185 | 39 | 6.161 | 13 | 33.333 |  |
| 2022 | march=False | 4852 | 38.664 | 418 | 8.615 | 101 | 24.163 |  |
| 2023 | power=True | 1519 | 99.737 | 58 | 3.818 | 58 | 100 |  |
| 2023 | power=False | 4230 | 16.359 | 335 | 7.920 | 43 | 12.836 |  |
| 2023 | neutral=True | 765 | 55.425 | 110 | 14.379 | 47 | 42.727 |  |
| 2023 | neutral=False | 4984 | 35.774 | 283 | 5.678 | 54 | 19.081 |  |
| 2023 | postseason=True | 113 | 100 | 14 | 12.389 | 14 | 100 |  |
| 2023 | postseason=False | 5636 | 37.154 | 379 | 6.725 | 87 | 22.955 |  |
| 2023 | march=True | 554 | 56.859 | 24 | 4.332 | 16 | 66.667 |  |
| 2023 | march=False | 5195 | 36.420 | 369 | 7.103 | 85 | 23.035 |  |
| 2024 | power=True | 1635 | 54.679 | 60 | 3.670 | 41 | 68.333 |  |
| 2024 | power=False | 4096 | 8.618 | 256 | 6.250 | 32 | 12.500 |  |
| 2024 | neutral=True | 741 | 28.745 | 132 | 17.814 | 45 | 34.091 |  |
| 2024 | neutral=False | 4990 | 20.721 | 184 | 3.687 | 28 | 15.217 |  |
| 2024 | postseason=True | 120 | 0 | 0 | 0 | 0 |  | UNDERPOWERED (<5 games) |
| 2024 | postseason=False | 5611 | 22.224 | 316 | 5.632 | 73 | 23.101 |  |
| 2024 | march=True | 800 | 0 | 16 | 2 | 0 | 0 |  |
| 2024 | march=False | 4931 | 25.289 | 300 | 6.084 | 73 | 24.333 |  |
| 2025 | power=True | 1618 | 0 | 34 | 2.101 | 0 | 0 |  |
| 2025 | power=False | 4151 | 0 | 86 | 2.072 | 0 | 0 |  |
| 2025 | neutral=True | 739 | 0 | 27 | 3.654 | 0 | 0 |  |
| 2025 | neutral=False | 5030 | 0 | 93 | 1.849 | 0 | 0 |  |
| 2025 | postseason=True | 123 | 0 | 0 | 0 | 0 |  | UNDERPOWERED (<5 games) |
| 2025 | postseason=False | 5646 | 0 | 120 | 2.125 | 0 | 0 |  |
| 2025 | march=True | 766 | 0 | 1 | 0.131 | 0 | 0 | UNDERPOWERED (<5 games) |
| 2025 | march=False | 5003 | 0 | 119 | 2.379 | 0 | 0 |  |

## 6. 2019-2021: games that would gain coordinates if the training window were extended

| season | our_final_d1proxy_games | gz_matched | gz_matched_pct | ours_hoopr_hist_located_games | ours_located_pct | gain_games_gz_and_not_located | gain_pct_of_ours | gain_pct_of_unlocated |
|---|---|---|---|---|---|---|---|---|
| 2019 | 5597 | 2171 | 38.789 | 840 | 15.008 | 1438 | 25.692 | 30.229 |
| 2020 | 5328 | 2088 | 39.189 | 627 | 11.768 | 1489 | 27.947 | 31.674 |
| 2021 | 3944 | 1579 | 40.035 | 578 | 14.655 | 1051 | 26.648 | 31.224 |

CBBD plays are on disk only from 2021-11 (season 2022), so for 2019-2021 the only on-disk coordinates are hoopR_hist pbp (ESPN, roughly 11-14% located); 'unlocated' there means hoopR_hist located share <= 0.5. CBBD coordinates for those seasons (sampled earlier at 28-71%) are not on disk, so the true gain over a full CBBD pull would be smaller.

## Verdict, inventory (plain words)

All 16 files read without R and all carry coordinates on 99.97-99.99% of field-goal attempts (free throws never), in every game (no game below 90% located). The one odd minimum date (2013-12-27) is a single game copied into several files, not mixed seasons. GameZone games match our CBBD universe by date and names at 98.5-100.0% per season (2013-2024; 0 fuzzy matches in total), and it covers about 38-40% of our D1 games each full season (40.2% in 2022, 38.6% in 2023, 21.8% in 2024 because the feed stops 15 Jan 2024, 0% in 2025 and 2026). Coverage is skewed to the games we lack: it covers 100% of our unlocated power-conference games and 100% of unlocated postseason games in 2022 and 2023, 34-43% of unlocated neutral-site games, but only 23-26% of all our unlocated games (114, 101, 73 games in 2022-2024), because most of our unlocated games are low-tier games GameZone does not chart (power-conference home games: GameZone covers about 100%; the rest about 8-17%). Nothing for 2024-25 or 2025-26. The placement caveat that matters for any use: rim attempts sit on two canned pixels (about 94-98% of rim-text attempts in most files, 50% in the 2022-23 file), and the CBBD-style dunk-median basket calibration is degenerate (the textbook rim centre is used instead). If the window were extended to 2019-2021, GameZone would add coordinates to 1,051-1,489 games per season (26-28% of our D1 games) over what hoopR_hist locates today, with the caveat above on CBBD coverage for those years.
