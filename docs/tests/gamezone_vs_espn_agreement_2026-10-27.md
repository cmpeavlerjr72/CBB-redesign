# GameZone vs ESPN (CBBD) agreement diagnostic (2026-10-27)

Descriptive only: no fit, no recommendation. Private internal research data (user ruling 2026-10-27). Script `scripts/diag_gamezone_vs_espn_v1.py` (needs `scripts/diag_gamezone_inventory_v1.py` first); tables also in `docs/tests/gamezone_vs_espn_agreement_2026-10-27.json`. "ESPN" below means the CBBD located shots (ESPN-sourced placement).

## Scope, definitions, floors

- Seasons: (a), (c): 2022, 2023, 2024 (the co-located seasons; GameZone ends 15 Jan 2024; 2025 and 2026 are not in GameZone and 2026 is untouched). (b), (d): all GameZone seasons by date season. Season = ending year.
- Games: GameZone games matched to CBBD games (inventory doc section 4). CBBD side: the audit master tables (FGA rows, de-duplicated, rim override 2.27 ft, classes rim / jump2 / three; text classes from `playText`, as the audit). GameZone side: FGA = shot rows with `free_throw == False`; `gz_cls` = three if `three_pt`, rim if `shot_desc` in {Layup, Dunk, Tip}, else jump2 (hook shots are jump2 here); text classes from the vendor `desc` (3-point; layup / dunk / tip; jump shot).
- Frames: CBBD folded to the left half, basket (52.5, 250) tenths of ft. GameZone: PRIMARY basket = textbook rim centre (25.0, 5.25), which is canned pixel A (inventory section 3 explains why the dunk-median calibration is degenerate); the per-date-season dunk centroid is shown as a sensitivity (`dunkcentroid` columns). Distances in feet. Arc 22.15 ft; slack variants as in the audit.
- Join (a): one-to-one on (game, period, clock within 10 s, shooter, outcome). Period: CBBD period = GameZone half (3+ = overtime). Clock: CBBD `secondsRemaining` vs GameZone `half_secs_remaining`, exact first, then nearest-first up to +-10 s (the two clocks differ by 3-10 s on many shots, symmetric, no per-game constant). Shooter: CBBD `shot_shooter_name` vs GameZone `shooter`, normalised (accents, punctuation, suffix removed) to last name + first initial. Outcome must match. Leftover pairs agreeing on everything except outcome are counted.
- Floors: paired comparisons are within game (weights n_a n_b / (n_a + n_b)) with a 300-resample game bootstrap, as the audit. Any cell under 500 attempts is UNDERPOWERED and is not read as signal or as absence of signal.

## (a) Play-level join on matched games, 2022-2024

Join rate (of CBBD attempts, of GameZone attempts, of CBBD located attempts), and the clock offsets that were needed ({0: 216038, 1: 79710, -1: 73988, 2: 50351, -2: 45573, 3: 28872, -3: 28611, -4: 17910, 4: 16869, -5: 11482, 5: 10223, -6: 7553, 6: 6827, -7: 5266, 7: 4766, -8: 3861, 8: 3701, 9: 2945, -9: 2786, 10: 2611, -10: 2166}; 39 leftover pairs agree on game, period, clock and shooter but not on outcome):

| season | matched_games | cbbd_fga | cbbd_located | gz_fga | joined | join_rate_of_cbbd_pct | join_rate_of_gz_pct | join_rate_of_cbbd_located_pct | joined_both_located |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 2219 | 257062 | 178894 | 260164 | 242398 | 94.296 | 93.171 | 94.066 | 168278 |
| 2023 | 2224 | 256252 | 193919 | 258945 | 239882 | 93.612 | 92.638 | 94.065 | 182409 |
| 2024 | 1281 | 152697 | 136557 | 153823 | 139829 | 91.573 | 90.903 | 91.439 | 124867 |

Distance-to-basket agreement on joined attempts located by BOTH vendors (475,554 attempts), |CBBD distance - GameZone distance|, by CBBD shot class and by outcome separately:

All: | n | within_1ft_pct | within_3ft_pct | gt_6ft_pct | median_abs_diff_ft | flag |
|---|---|---|---|---|---|
| 475554 | 30.435 | 70.972 | 7.394 | 1.790 |  |

By class:

| cls_c | n | within_1ft_pct | within_3ft_pct | gt_6ft_pct | median_abs_diff_ft | flag |
|---|---|---|---|---|---|---|
| jump2 | 123790 | 21.395 | 55.705 | 17.028 | 2.606 |  |
| rim | 170304 | 22.393 | 64.521 | 6.666 | 2.270 |  |
| three | 181460 | 44.149 | 87.441 | 1.505 | 1.163 |  |

By outcome:

| outcome | n | within_1ft_pct | within_3ft_pct | gt_6ft_pct | median_abs_diff_ft | flag |
|---|---|---|---|---|---|---|
| made | 206696 | 30.835 | 73.203 | 5.614 | 1.694 |  |
| missed | 268858 | 30.127 | 69.256 | 8.763 | 1.836 |  |

Class x outcome:

| cls_c | outcome | n | within_1ft_pct | within_3ft_pct | gt_6ft_pct | median_abs_diff_ft | flag |
|---|---|---|---|---|---|---|---|
| jump2 | made | 45540 | 22.734 | 58.566 | 14.930 | 2.422 |  |
| jump2 | missed | 78250 | 20.616 | 54.040 | 18.249 | 2.716 |  |
| rim | made | 100548 | 25.615 | 70.560 | 4.133 | 1.963 |  |
| rim | missed | 69756 | 17.749 | 55.815 | 10.316 | 2.721 |  |
| three | made | 60608 | 45.583 | 88.584 | 1.069 | 1.119 |  |
| three | missed | 120852 | 43.430 | 86.867 | 1.724 | 1.185 |  |

By season x class:

| season | cls_c | n | within_1ft_pct | within_3ft_pct | gt_6ft_pct | median_abs_diff_ft | flag |
|---|---|---|---|---|---|---|---|
| 2022 | jump2 | 44737 | 20.987 | 54.892 | 17.596 | 2.654 |  |
| 2022 | rim | 59005 | 19.744 | 61.615 | 7.145 | 2.368 |  |
| 2022 | three | 64536 | 44.041 | 87.308 | 1.452 | 1.168 |  |
| 2023 | jump2 | 47231 | 20.971 | 55.576 | 16.767 | 2.629 |  |
| 2023 | rim | 65725 | 26.275 | 68.943 | 6.314 | 1.976 |  |
| 2023 | three | 69453 | 43.595 | 87.003 | 1.507 | 1.180 |  |
| 2024 | jump2 | 31822 | 22.598 | 57.039 | 16.617 | 2.501 |  |
| 2024 | rim | 45574 | 20.224 | 61.904 | 6.552 | 2.368 |  |
| 2024 | three | 47471 | 45.108 | 88.262 | 1.574 | 1.132 |  |

Signed difference (CBBD minus GameZone distance, ft) by class and outcome: json `a_mean_signed_diff_cbbd_minus_gz_ft`. Class agreement on the joined attempts: {'n_joined_both_located': 475554, 'three_flag_disagree_pct': 0.21743061776370295, 'rim_class_agree_pct': 94.90657212430133}.

Text consistency of each vendor on the SAME joined, both-located attempts (text class from the CBBD play text; threes charted inside 22.15 ft, text jumpers charted outside it, layups/dunks/tips charted more than 6 ft from the rim), by outcome:

| outcome | vendor | n_three | n_rimtext | n_twojump | three_inside_arc_pct | twojump_outside_arc_pct | rimtext_gt6ft_pct | flag |
|---|---|---|---|---|---|---|---|---|
| all | cbbd | 181460 | 168405 | 124908 | 1.714 | 0.301 | 9.227 |  |
| all | gamezone | 181460 | 168405 | 124908 | 2.843 | 0.274 | 7.383 |  |
| made | cbbd | 60608 | 99442 | 46364 | 1.625 | 0.125 | 5.555 |  |
| made | gamezone | 60608 | 99442 | 46364 | 2.686 | 0.060 | 5.097 |  |
| missed | cbbd | 120852 | 68963 | 78544 | 1.758 | 0.405 | 14.521 |  |
| missed | gamezone | 120852 | 68963 | 78544 | 2.922 | 0.400 | 10.678 |  |

Rim-text charted > 6 ft from the rim, missed minus made, within game with game-bootstrap floor, same attempts, per vendor:

| vendor | miss_gt6_rate | make_gt6_rate | pooled_diff | within_game_diff | boot_se | z | n_miss | n_make | n_games_paired | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| cbbd | 0.145 | 0.056 | 0.090 | 0.088 | 0.002 | 50.065 | 68963 | 99442 | 5324 |  |
| gamezone | 0.107 | 0.051 | 0.056 | 0.055 | 0.002 | 32.417 | 68963 | 99442 | 5324 |  |

Robustness (same measure; only joins with clock offset <= 1 s; only attempts GameZone also classes as a rim shot):

| subset | vendor | miss_gt6_pct | make_gt6_pct | within_game_diff_pp | boot_se_pp | n_miss | n_make | flag |
|---|---|---|---|---|---|---|---|---|
| all joined rim-text | cbbd | 14.521 | 5.555 | 8.782 | 0.172 | 68963 | 99442 |  |
| all joined rim-text | gamezone | 10.678 | 5.097 | 5.462 | 0.171 | 68963 | 99442 |  |
| clock offset <= 1 s | cbbd | 14.223 | 5.610 | 8.564 | 0.208 | 41236 | 66761 |  |
| clock offset <= 1 s | gamezone | 10.190 | 5.081 | 4.967 | 0.218 | 41236 | 66761 |  |
| GameZone also calls it a rim shot | cbbd | 13.484 | 5.058 | 8.282 | 0.173 | 62098 | 94236 |  |
| GameZone also calls it a rim shot | gamezone | 4.123 | 2.811 | 1.680 | 0.099 | 62098 | 94236 |  |

Verdict (a): The independent vendor does not reproduce ESPN rim charting by outcome at ESPN size, but it also cannot test it well, because it puts almost every rim attempt on one of two fixed pixels. Join: 91.6-94.3% of CBBD attempts per season join one-to-one to a GameZone attempt (475,554 joined attempts located by both vendors). On those, the two vendors' distances to the basket are within 1 ft for 30% of attempts, within 3 ft for 71%, and more than 6 ft apart for 7.4%; threes agree best (44% within 1 ft, 87% within 3 ft, 1.5% over 6 ft), two-point jumpers worst (21%, 56%, 17.0%), rim shots in between (22%, 65%, 6.7%). By outcome the disagreement is not symmetric: rim attempts are over 6 ft apart for 10.3% of misses against 4.1% of makes, jumpers 18.3% vs 14.9%, threes 1.7% vs 1.1%. Where the rim error sits: for layups, dunks and tips (CBBD play text), ESPN charts 14.5% of misses against 5.6% of makes more than 6 ft from the rim (gap 8.8 points within game, floor 0.17); GameZone on the same attempts charts 10.7% of misses against 5.1% of makes (gap 5.5 points). Restricted to attempts GameZone itself calls rim shots, ESPN keeps its gap (13.5% vs 5.1%, 8.3 points) while GameZone shows 4.1% vs 2.8% (1.7 points, floor 0.10). So the outcome-dependent far placement is an ESPN pattern, and GameZone own rim placement barely moves with outcome (mostly because it is canned). Three-point placement does not depend on outcome in either vendor (text threes charted inside 22.15 ft: ESPN 1.6% made vs 1.8% missed, GameZone 2.7% vs 2.9%).

## (b) Rim asymmetry re-measured on GameZone alone, all seasons

Layups, dunks and tips (`shot_desc`), located. For each date season: share charted > 6 ft from the rim, share sitting exactly on one of the two canned rim pixels (25.0, 5.25) / (26.0, 6.25), and the same with the per-season dunk-centroid basket, for misses vs makes, within game (the audit found 6-7% for makes vs 15-16% for misses on ESPN):

| date_season | measure | n_miss | n_make | miss_rate_pct | make_rate_pct | pooled_diff_pp | within_game_diff_pp | boot_se_pp | z | games_paired | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2009 | gt6ft_from_rim | 4241 | 6543 | 0.141 | 0.015 | 0.126 | 0.136 | 0.069 | 1.965 | 306 |  |
| 2009 | on_canned_pixel | 4241 | 6543 | 94.671 | 94.712 | -0.041 | -0.183 | 0.539 | -0.339 | 306 |  |
| 2009 | gt6ft_dunk_centroid_basket | 4241 | 6543 | 0.141 | 0.015 | 0.126 | 0.136 | 0.067 | 2.028 | 306 |  |
| 2010 | gt6ft_from_rim | 28234 | 43460 | 0.298 | 0.108 | 0.189 | 0.193 | 0.036 | 5.296 | 1892 |  |
| 2010 | on_canned_pixel | 28234 | 43460 | 93.671 | 94.443 | -0.772 | -0.670 | 0.207 | -3.238 | 1892 |  |
| 2010 | gt6ft_dunk_centroid_basket | 28234 | 43460 | 0.298 | 0.108 | 0.189 | 0.193 | 0.036 | 5.304 | 1892 |  |
| 2011 | gt6ft_from_rim | 26867 | 40827 | 0.234 | 0.118 | 0.117 | 0.130 | 0.034 | 3.770 | 1833 |  |
| 2011 | on_canned_pixel | 26867 | 40827 | 93.397 | 94.364 | -0.967 | -0.819 | 0.215 | -3.817 | 1833 |  |
| 2011 | gt6ft_dunk_centroid_basket | 26867 | 40827 | 0.234 | 0.118 | 0.117 | 0.130 | 0.035 | 3.685 | 1833 |  |
| 2012 | gt6ft_from_rim | 28069 | 41797 | 0.349 | 0.100 | 0.249 | 0.253 | 0.042 | 6.058 | 1899 |  |
| 2012 | on_canned_pixel | 28069 | 41797 | 93.630 | 94.461 | -0.831 | -0.782 | 0.191 | -4.094 | 1899 |  |
| 2012 | gt6ft_dunk_centroid_basket | 28069 | 41797 | 0.242 | 0.069 | 0.173 | 0.184 | 0.033 | 5.495 | 1899 |  |
| 2013 | gt6ft_from_rim | 28816 | 42807 | 0.538 | 0.290 | 0.248 | 0.274 | 0.053 | 5.185 | 1934 |  |
| 2013 | on_canned_pixel | 28816 | 42807 | 93.944 | 94.552 | -0.608 | -0.501 | 0.203 | -2.472 | 1934 |  |
| 2013 | gt6ft_dunk_centroid_basket | 28816 | 42807 | 0.371 | 0.192 | 0.180 | 0.202 | 0.042 | 4.769 | 1934 |  |
| 2014 | gt6ft_from_rim | 35340 | 50868 | 0.580 | 0.429 | 0.152 | 0.164 | 0.051 | 3.214 | 2130 |  |
| 2014 | on_canned_pixel | 35340 | 50868 | 94.624 | 94.840 | -0.216 | -0.149 | 0.167 | -0.890 | 2130 |  |
| 2014 | gt6ft_dunk_centroid_basket | 35340 | 50868 | 0.405 | 0.271 | 0.133 | 0.146 | 0.042 | 3.457 | 2130 |  |
| 2015 | gt6ft_from_rim | 33118 | 46632 | 0.710 | 0.358 | 0.351 | 0.375 | 0.055 | 6.840 | 2103 |  |
| 2015 | on_canned_pixel | 33118 | 46632 | 94.480 | 94.545 | -0.064 | 0.039 | 0.174 | 0.223 | 2103 |  |
| 2015 | gt6ft_dunk_centroid_basket | 33118 | 46632 | 0.553 | 0.229 | 0.323 | 0.356 | 0.047 | 7.578 | 2103 |  |
| 2016 | gt6ft_from_rim | 33943 | 50890 | 0.598 | 0.334 | 0.264 | 0.258 | 0.048 | 5.334 | 2122 |  |
| 2016 | on_canned_pixel | 33943 | 50890 | 94.738 | 94.867 | -0.129 | -0.066 | 0.181 | -0.366 | 2122 |  |
| 2016 | gt6ft_dunk_centroid_basket | 33943 | 50890 | 0.448 | 0.224 | 0.224 | 0.229 | 0.046 | 5.014 | 2122 |  |
| 2017 | gt6ft_from_rim | 31437 | 48222 | 0.471 | 0.280 | 0.191 | 0.189 | 0.051 | 3.729 | 2063 |  |
| 2017 | on_canned_pixel | 31437 | 48222 | 95.588 | 95.197 | 0.391 | 0.489 | 0.168 | 2.904 | 2063 |  |
| 2017 | gt6ft_dunk_centroid_basket | 31437 | 48222 | 0.359 | 0.209 | 0.150 | 0.157 | 0.042 | 3.747 | 2063 |  |
| 2018 | gt6ft_from_rim | 32302 | 50645 | 0.452 | 0.274 | 0.178 | 0.173 | 0.048 | 3.582 | 2130 |  |
| 2018 | on_canned_pixel | 32302 | 50645 | 95.821 | 95.705 | 0.115 | 0.244 | 0.153 | 1.596 | 2130 |  |
| 2018 | gt6ft_dunk_centroid_basket | 32302 | 50645 | 0.359 | 0.211 | 0.148 | 0.147 | 0.042 | 3.501 | 2130 |  |
| 2019 | gt6ft_from_rim | 34138 | 52890 | 0.395 | 0.229 | 0.167 | 0.187 | 0.040 | 4.680 | 2198 |  |
| 2019 | on_canned_pixel | 34138 | 52890 | 96.743 | 96.457 | 0.286 | 0.295 | 0.131 | 2.257 | 2198 |  |
| 2019 | gt6ft_dunk_centroid_basket | 34138 | 52890 | 0.264 | 0.172 | 0.092 | 0.113 | 0.034 | 3.341 | 2198 |  |
| 2020 | gt6ft_from_rim | 33459 | 49902 | 0.317 | 0.234 | 0.082 | 0.077 | 0.040 | 1.922 | 2125 |  |
| 2020 | on_canned_pixel | 33459 | 49902 | 97.427 | 97.237 | 0.190 | 0.311 | 0.117 | 2.655 | 2125 |  |
| 2020 | gt6ft_dunk_centroid_basket | 33459 | 49902 | 0.272 | 0.166 | 0.106 | 0.104 | 0.034 | 3.038 | 2125 |  |
| 2021 | gt6ft_from_rim | 24767 | 37072 | 0.266 | 0.235 | 0.032 | 0.023 | 0.044 | 0.515 | 1595 |  |
| 2021 | on_canned_pixel | 24767 | 37072 | 97.985 | 97.446 | 0.540 | 0.631 | 0.123 | 5.144 | 1595 |  |
| 2021 | gt6ft_dunk_centroid_basket | 24767 | 37072 | 0.174 | 0.165 | 0.009 | 0.002 | 0.038 | 0.047 | 1595 |  |
| 2022 | gt6ft_from_rim | 37083 | 53741 | 0.321 | 0.208 | 0.112 | 0.119 | 0.036 | 3.267 | 2220 |  |
| 2022 | on_canned_pixel | 37083 | 53741 | 98.134 | 97.946 | 0.188 | 0.285 | 0.110 | 2.600 | 2220 |  |
| 2022 | gt6ft_dunk_centroid_basket | 37083 | 53741 | 0.272 | 0.151 | 0.122 | 0.126 | 0.032 | 3.931 | 2220 |  |
| 2023 | gt6ft_from_rim | 36957 | 54630 | 10.474 | 7.135 | 3.339 | 4.357 | 0.220 | 19.786 | 2231 |  |
| 2023 | on_canned_pixel | 36957 | 54630 | 45.372 | 53.449 | -8.077 | -7.606 | 0.244 | -31.115 | 2231 |  |
| 2023 | gt6ft_dunk_centroid_basket | 36957 | 54630 | 8.142 | 5.748 | 2.394 | 3.392 | 0.177 | 19.172 | 2231 |  |
| 2024 | gt6ft_from_rim | 23471 | 32779 | 0.183 | 0.070 | 0.113 | 0.104 | 0.031 | 3.310 | 1292 |  |
| 2024 | on_canned_pixel | 23471 | 32779 | 98.671 | 98.337 | 0.333 | 0.390 | 0.116 | 3.357 | 1292 |  |
| 2024 | gt6ft_dunk_centroid_basket | 23471 | 32779 | 0.162 | 0.058 | 0.104 | 0.100 | 0.028 | 3.571 | 1292 |  |

Spike pixels: top 20 pixels per season (exact vendor coordinate), share of located attempts, make rate versus the attempts in the 3x3 block of 1-ft cells around it (excluding itself), in the json `b_top20_pixels`. Pixel A = (25.0, 5.25) and B = (26.0, 6.25): make rate at each, and which pixel makes and misses use:

| date_season | n_A | A_make_rate | n_B | B_make_rate | rim_text_make_rate_not_on_A_or_B | share_of_pair_that_is_A_among_makes_pct | share_of_pair_that_is_A_among_misses_pct |
|---|---|---|---|---|---|---|---|
| 2009 | 6332 | 0.979 | 3887 | 0.000 | 0.605 | 99.984 | 3.285 |
| 2010 | 42030 | 0.977 | 25546 | 0.002 | 0.575 | 99.891 | 3.646 |
| 2011 | 39402 | 0.978 | 24274 | 0.002 | 0.565 | 99.881 | 3.524 |
| 2012 | 32707 | 0.587 | 33114 | 0.614 | 0.564 | 48.540 | 51.420 |
| 2013 | 33776 | 0.623 | 33860 | 0.576 | 0.572 | 51.916 | 46.980 |
| 2014 | 40639 | 0.602 | 41122 | 0.580 | 0.580 | 50.626 | 48.375 |
| 2015 | 37734 | 0.597 | 37727 | 0.573 | 0.582 | 51.019 | 48.574 |
| 2016 | 40157 | 0.619 | 40306 | 0.582 | 0.594 | 51.463 | 47.572 |
| 2017 | 37555 | 0.595 | 38450 | 0.614 | 0.625 | 48.615 | 50.627 |
| 2018 | 39641 | 0.609 | 39809 | 0.612 | 0.617 | 49.755 | 50.113 |
| 2019 | 41941 | 0.597 | 42141 | 0.617 | 0.628 | 49.024 | 51.204 |
| 2020 | 40261 | 0.588 | 40898 | 0.608 | 0.616 | 48.791 | 50.823 |
| 2021 | 29966 | 0.593 | 30455 | 0.603 | 0.655 | 49.171 | 50.227 |
| 2022 | 44429 | 0.593 | 44653 | 0.589 | 0.615 | 50.044 | 49.629 |
| 2023 | 22814 | 0.631 | 23173 | 0.639 | 0.557 | 49.300 | 50.149 |
| 2024 | 27721 | 0.582 | 27687 | 0.582 | 0.636 | 50.017 | 50.050 |

Top three pixels per season:

| date_season | make_rate_rank1 | make_rate_rank2 | make_rate_rank3 | share_of_located_pct_rank1 | share_of_located_pct_rank2 | share_of_located_pct_rank3 |
|---|---|---|---|---|---|---|
| 2009 | 0.979 | 0 | 0.400 | 18.174 | 11.156 | 0.029 |
| 2010 | 0.977 | 0.002 | 0.419 | 19.337 | 11.753 | 0.020 |
| 2011 | 0.978 | 0.002 | 0.356 | 18.570 | 11.440 | 0.042 |
| 2012 | 0.614 | 0.587 | 0.412 | 15.306 | 15.118 | 0.047 |
| 2013 | 0.576 | 0.623 | 0.423 | 15.429 | 15.391 | 0.051 |
| 2014 | 0.580 | 0.602 | 0.458 | 16.759 | 16.563 | 0.039 |
| 2015 | 0.597 | 0.573 | 0.453 | 15.690 | 15.687 | 0.040 |
| 2016 | 0.582 | 0.619 | 0.326 | 15.541 | 15.483 | 0.035 |
| 2017 | 0.614 | 0.595 | 0.643 | 15.498 | 15.138 | 0.023 |
| 2018 | 0.612 | 0.609 | 0.385 | 15.759 | 15.692 | 0.021 |
| 2019 | 0.617 | 0.597 | 0.397 | 16.510 | 16.432 | 0.029 |
| 2020 | 0.608 | 0.588 | 0.418 | 16.637 | 16.378 | 0.022 |
| 2021 | 0.603 | 0.593 | 0.435 | 16.162 | 15.903 | 0.024 |
| 2022 | 0.589 | 0.593 | 0.387 | 17.166 | 17.080 | 0.024 |
| 2023 | 0.639 | 0.631 | 0.543 | 8.950 | 8.811 | 0.067 |
| 2024 | 0.582 | 0.582 | 0.300 | 18.027 | 18.005 | 0.033 |

Verdict (b): On its own, GameZone shows almost no outcome-dependent rim charting, except in one file, and its rim placement is mostly not a measurement. In 15 of 16 date seasons 93-99% of layups, dunks and tips sit on one of two fixed pixels, (25.0, 5.25) and (26.0, 6.25), which together hold 29-36% of ALL located attempts (18% in the 2022-23 file); only 0.0-0.7% of rim-text attempts are charted more than 6 ft from the rim, and misses minus makes is 0.02-0.37 points within game (floors 0.03-0.07; many are statistically nonzero because n is huge, but the size is a few tenths of a point against ESPN 9 points). The exception is the 2022-23 file (date season 2023): only 45% of misses but 53% of makes are on the fixed pixels, the rest carry real coordinates, and there misses are charted over 6 ft from the rim 10.5% of the time against 7.1% for makes (4.4 points within game, floor 0.22; about half of ESPN gap on its own data). Which rim attempts get a real coordinate in that file depends on the outcome (makes are more often canned, -7.6 points, z -31). A different problem: in the three earliest date seasons (2008-09 to 2010-11 files) the pixel itself encodes the outcome: (25.0, 5.25) has a 97.8% make rate and (26.0, 6.25) a 0.2% make rate (99.9% of makes on the pair use the first pixel, 96-97% of misses use the second), then from 2011-12 the two pixels are used 50/50 with make rates 57-64% each. Spike pixels exist in every season (the top two pixels are the pair; the third pixel holds at most 0.07% of attempts); from 2011-12 the two pixels make rate is within 5 points of their neighbours (a third pixel holds at most 0.07% of attempts and is too small to read).

## (c) Band shares per team-game, GameZone vs CBBD, co-located games 2022-2024

Bands from spec section 2 (short2 <= 10 ft; long2; corner3 = within 4 ft of a sideline and <= 24 ft; wing3 |theta| >= 30 deg; top3), each vendor in its own frame and own class. Pooled shares over all co-located located attempts:

| vendor | short2 | long2 | corner3 | wing3 | top3 | n_jump2 | n_three |
|---|---|---|---|---|---|---|---|
| cbbd | 41.286 | 58.714 | 21.785 | 44.281 | 33.934 | 132825 | 193870 |
| gamezone | 31.688 | 68.312 | 18.089 | 50.917 | 30.994 | 182259 | 252101 |
| gamezone (dunk-centroid basket sensitivity) | 34.365 | 65.635 | 17.819 | 52.020 | 30.161 | 182259 | 252101 |

Per team-game (both vendors with at least 5 located attempts of the class), share of the class in each band:

| class | band | min_attempts_each_vendor | team_games | mean_share_cbbd_pct | mean_share_gz_pct | mean_abs_diff_pp | mean_signed_diff_cbbd_minus_gz_pp | corr_team_game | flag |
|---|---|---|---|---|---|---|---|---|---|
| jump2 | short2 | 5 | 8595 | 41.046 | 31.355 | 19.501 | 9.691 | 0.298 |  |
| jump2 | long2 | 5 | 8595 | 58.954 | 68.645 | 19.501 | -9.691 | 0.298 |  |
| three | corner3 | 5 | 9017 | 21.689 | 18.142 | 10.782 | 3.547 | 0.350 |  |
| three | wing3 | 5 | 9017 | 44.137 | 50.982 | 16.912 | -6.845 | 0.173 |  |
| three | top3 | 5 | 9017 | 34.174 | 30.876 | 14.580 | 3.298 | 0.252 |  |

Per team-season (both vendors with at least 100 attempts of the class), correlation of the band share between vendors:

| class | band | min_attempts_each_vendor | team_seasons | corr_team_season | sd_share_cbbd_pp | sd_share_gz_pp | mean_abs_diff_pp | flag |
|---|---|---|---|---|---|---|---|---|
| jump2 | short2 | 100 | 361 | 0.639 | 9.734 | 8.714 | 10.568 |  |
| jump2 | long2 | 100 | 361 | 0.639 | 9.734 | 8.714 | 10.568 |  |
| three | corner3 | 100 | 474 | 0.509 | 5.406 | 4.871 | 4.878 |  |
| three | wing3 | 100 | 474 | 0.257 | 5.008 | 6.136 | 7.702 |  |
| three | top3 | 100 | 474 | 0.389 | 5.546 | 5.509 | 5.409 |  |

Share of individual joined attempts whose band differs (both located, same class under both vendors):

| class | scope | n | band_differs_pct | flag |
|---|---|---|---|---|
| jump2 | same class under both vendors | 112306 | 26.674 |  |
| jump2 | either vendor says short2 | 50595 | 59.207 |  |
| jump2 | either vendor says long2 | 91667 | 32.679 |  |
| three | same class under both vendors | 180956 | 37.462 |  |
| three | either vendor says corner3 | 50914 | 56.884 |  |
| three | either vendor says wing3 | 116388 | 52.825 |  |
| three | either vendor says top3 | 81443 | 55.418 |  |
| jump2 | same class, made | 41193 | 25.609 |  |
| three | same class, made | 60547 | 36.615 |  |
| jump2 | same class, missed | 71113 | 27.290 |  |
| three | same class, missed | 120409 | 37.888 |  |

Three-point band confusion on joined attempts (rows CBBD, columns GameZone): {'corner3': {'corner3': 21952, 'top3': 4383, 'wing3': 13141}, 'top3': {'corner3': 1924, 'top3': 36309, 'wing3': 23308}, 'wing3': {'corner3': 9514, 'top3': 15519, 'wing3': 54906}}. Jump2 (rows CBBD, columns GameZone): {'long2': {'long2': 61711, 'short2': 8911}, 'short2': {'long2': 21045, 'short2': 20639}}.

Verdict (c): The two vendors do not agree on where jumpers and threes are taken at the band level, though they agree on which teams lean one way for some bands. Pooled over co-located games, GameZone has fewer short two-point jumpers (31.7% of two-point jumpers vs 41.3% in CBBD) and more wing threes (50.9% vs 44.3%); corner threes 18.1% vs 21.8%, top threes 31.0% vs 33.9% (using the dunk-centroid basket instead moves GameZone short2 to 34.4%, so part of the gap is frame calibration, which cannot be resolved here). Per team-game the mean absolute difference in band share is 19.5 points for short2/long2, 10.8 for corner3, 16.9 for wing3 and 14.6 for top3, with team-game correlations of 0.30, 0.35, 0.17 and 0.25. Per team-season (at least 100 attempts each) the band shares correlate 0.64 for short2, 0.51 for corner3, 0.39 for top3 and only 0.26 for wing3, with similar spread across teams in the two vendors (SD 9.7 vs 8.7 points for short2). On individual joined attempts of the same class, the band differs for 26.7% of two-point jumpers and 37.5% of threes (of the attempts either vendor puts in a given band, 56.9% corner, 52.8% wing, 55.4% top and 59.2% short2 differ), with no difference by outcome (made 36.6% vs missed 37.9% for threes). The main confusions are wing vs top and corner vs wing for threes, short vs long for jumpers. What the vendors share is a team-level tendency, not an attempt-level band.

## (d) Three-point placement accuracy under GameZone

The men's three-point line was 20.75 ft through 2018-19 and 22.15 ft from 2019-20; the audit used 22.15 ft for every season by instruction, so both are shown (`rule_arc_ft` columns use the line in force; the 22.15 columns are the audit's convention and the only like-for-like comparison with the audit for 2022-2024). Text threes (vendor `desc` contains 3-point) charted inside the arc, and text two-point jumpers (`desc` contains "jump shot", no 3-point) charted outside it, by date season, with the textbook (primary) and dunk-centroid baskets and the audit's slack variants (CBBD on the same co-located attempts is in (a)):

| date_season | n_text_three | n_text_twojump | rule_arc_ft | three_inside_rule_arc_textbook_pct | twojump_outside_rule_arc_textbook_pct | three_inside_22.15_textbook_pct | three_inside_21.65_textbook_pct | three_inside_20.65_textbook_pct | three_inside_20.75_textbook_pct | twojump_outside_22.15_textbook_pct | twojump_outside_22.65_textbook_pct | twojump_outside_23.65_textbook_pct | three_inside_22.15_dunkcentroid_pct | three_inside_21.65_dunkcentroid_pct | three_inside_20.65_dunkcentroid_pct | three_inside_20.75_dunkcentroid_pct | twojump_outside_22.15_dunkcentroid_pct | twojump_outside_22.65_dunkcentroid_pct | twojump_outside_23.65_dunkcentroid_pct | vendor_three_flag_vs_text_disagree_pct | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2009 | 11636 | 11618 | 20.750 | 4.031 | 0.869 | 29.160 | 18.314 | 2.836 | 4.031 | 0.172 | 0.103 | 0 | 29.160 | 18.314 | 2.836 | 4.031 | 0.172 | 0.103 | 0 | 0 |  |
| 2010 | 71053 | 69445 | 20.750 | 3.562 | 1.071 | 30.054 | 18.872 | 2.621 | 3.562 | 0.160 | 0.068 | 0.012 | 30.054 | 18.872 | 2.621 | 3.562 | 0.160 | 0.068 | 0.012 | 0 |  |
| 2011 | 69989 | 68776 | 20.750 | 2.780 | 1.067 | 27.942 | 17.314 | 2.053 | 2.780 | 0.140 | 0.084 | 0.012 | 27.942 | 17.314 | 2.053 | 2.780 | 0.140 | 0.084 | 0.012 | 0 |  |
| 2012 | 70781 | 70076 | 20.750 | 1.801 | 0.953 | 18.978 | 10.662 | 1.320 | 1.801 | 0.094 | 0.063 | 0.004 | 24.193 | 13.932 | 3.111 | 3.764 | 0.050 | 0.013 | 0 | 0 |  |
| 2013 | 72026 | 70696 | 20.750 | 0.097 | 0.699 | 6.631 | 1.509 | 0.096 | 0.097 | 0 | 0 | 0 | 11.736 | 3.234 | 0.133 | 0.169 | 0 | 0 | 0 | 0 |  |
| 2014 | 80301 | 73702 | 20.750 | 0.080 | 0.544 | 6.151 | 1.326 | 0.080 | 0.080 | 0 | 0 | 0 | 12.021 | 3.613 | 0.126 | 0.162 | 0 | 0 | 0 | 0 |  |
| 2015 | 81510 | 74877 | 20.750 | 0.075 | 0.573 | 6.490 | 1.523 | 0.075 | 0.075 | 0 | 0 | 0 | 11.980 | 3.654 | 0.108 | 0.142 | 0 | 0 | 0 | 0 |  |
| 2016 | 90524 | 79874 | 20.750 | 0.007 | 0.427 | 4.664 | 0.925 | 0.007 | 0.007 | 0 | 0 | 0 | 9.145 | 2.647 | 0.040 | 0.078 | 0 | 0 | 0 | 0 |  |
| 2017 | 88812 | 76127 | 20.750 | 0.008 | 0.436 | 4.359 | 0.742 | 0.007 | 0.008 | 0 | 0 | 0 | 8.401 | 2.379 | 0.034 | 0.054 | 0 | 0 | 0 | 0 |  |
| 2018 | 94476 | 71988 | 20.750 | 0.075 | 0.357 | 4.063 | 0.862 | 0.074 | 0.075 | 0 | 0 | 0 | 7.621 | 2.243 | 0.098 | 0.126 | 0 | 0 | 0 | 0 |  |
| 2019 | 99178 | 64593 | 20.750 | 0.006 | 0.529 | 4.276 | 0.916 | 0.006 | 0.006 | 0 | 0 | 0 | 8.289 | 2.491 | 0.054 | 0.090 | 0 | 0 | 0 | 0 |  |
| 2020 | 92453 | 65478 | 22.150 | 4.443 | 0 | 4.443 | 0.887 | 0.009 | 0.009 | 0 | 0 | 0 | 8.348 | 2.400 | 0.050 | 0.084 | 0 | 0 | 0 | 0 |  |
| 2021 | 70568 | 52562 | 22.150 | 3.391 | 0 | 3.391 | 0.607 | 0.075 | 0.077 | 0 | 0 | 0 | 6.683 | 1.740 | 0.103 | 0.126 | 0 | 0 | 0 | 0 |  |
| 2022 | 97889 | 66052 | 22.150 | 2.697 | 0 | 2.697 | 0.457 | 0.036 | 0.036 | 0 | 0 | 0 | 5.755 | 1.407 | 0.053 | 0.063 | 0 | 0 | 0 | 0 |  |
| 2023 | 96451 | 65697 | 22.150 | 2.230 | 0 | 2.230 | 0.339 | 0.003 | 0.004 | 0 | 0 | 0 | 5.047 | 1.170 | 0.019 | 0.030 | 0 | 0 | 0 | 0 |  |
| 2024 | 57761 | 37083 | 22.150 | 3.328 | 0 | 3.328 | 0.738 | 0.220 | 0.220 | 0 | 0 | 0 | 6.918 | 2.022 | 0.251 | 0.279 | 0 | 0 | 0 | 0 |  |

By outcome (primary textbook basket):

| date_season | outcome | rule_arc_ft | n_three | three_inside_22.15_pct | three_inside_rule_arc_pct | n_twojump | twojump_outside_22.15_pct | twojump_outside_rule_arc_pct | flag |
|---|---|---|---|---|---|---|---|---|---|
| 2009 | made | 20.750 | 3999 | 31.658 | 4.801 | 4320 | 0.162 | 0.926 |  |
| 2010 | made | 20.750 | 24196 | 31.699 | 3.968 | 24644 | 0.138 | 1.250 |  |
| 2011 | made | 20.750 | 23985 | 30.265 | 3.031 | 24803 | 0.149 | 1.286 |  |
| 2012 | made | 20.750 | 24038 | 20.160 | 1.851 | 25134 | 0.092 | 1.166 |  |
| 2013 | made | 20.750 | 24058 | 7.000 | 0.249 | 24517 | 0 | 0.938 |  |
| 2014 | made | 20.750 | 27187 | 6.157 | 0.210 | 26277 | 0 | 0.693 |  |
| 2015 | made | 20.750 | 27529 | 6.706 | 0.196 | 26907 | 0 | 0.740 |  |
| 2016 | made | 20.750 | 31404 | 4.716 | 0.003 | 29115 | 0 | 0.464 |  |
| 2017 | made | 20.750 | 31052 | 4.428 | 0.006 | 28174 | 0 | 0.564 |  |
| 2018 | made | 20.750 | 33175 | 4.006 | 0.072 | 26585 | 0 | 0.459 |  |
| 2019 | made | 20.750 | 33513 | 4.377 | 0.006 | 22604 | 0 | 0.668 |  |
| 2020 | made | 22.150 | 30266 | 4.493 | 4.493 | 23039 | 0 | 0 |  |
| 2021 | made | 22.150 | 23772 | 3.492 | 3.492 | 19580 | 0 | 0 |  |
| 2022 | made | 22.150 | 32598 | 2.795 | 2.795 | 23872 | 0 | 0 |  |
| 2023 | made | 22.150 | 32202 | 2.192 | 2.192 | 23833 | 0 | 0 |  |
| 2024 | made | 22.150 | 19200 | 3.542 | 3.542 | 14000 | 0 | 0 |  |
| 2009 | missed | 20.750 | 7637 | 27.851 | 3.627 | 7298 | 0.178 | 0.836 |  |
| 2010 | missed | 20.750 | 46857 | 29.204 | 3.353 | 44801 | 0.172 | 0.973 |  |
| 2011 | missed | 20.750 | 46004 | 26.730 | 2.650 | 43973 | 0.134 | 0.944 |  |
| 2012 | missed | 20.750 | 46743 | 18.371 | 1.776 | 44942 | 0.096 | 0.834 |  |
| 2013 | missed | 20.750 | 47968 | 6.446 | 0.021 | 46179 | 0 | 0.572 |  |
| 2014 | missed | 20.750 | 53114 | 6.147 | 0.013 | 47425 | 0 | 0.462 |  |
| 2015 | missed | 20.750 | 53981 | 6.380 | 0.013 | 47970 | 0 | 0.479 |  |
| 2016 | missed | 20.750 | 59120 | 4.636 | 0.008 | 50759 | 0 | 0.406 |  |
| 2017 | missed | 20.750 | 57760 | 4.321 | 0.009 | 47953 | 0 | 0.361 |  |
| 2018 | missed | 20.750 | 61301 | 4.095 | 0.077 | 45403 | 0 | 0.297 |  |
| 2019 | missed | 20.750 | 65665 | 4.224 | 0.006 | 41989 | 0 | 0.455 |  |
| 2020 | missed | 22.150 | 62187 | 4.419 | 4.419 | 42439 | 0 | 0 |  |
| 2021 | missed | 22.150 | 46796 | 3.340 | 3.340 | 32982 | 0 | 0 |  |
| 2022 | missed | 22.150 | 65291 | 2.648 | 2.648 | 42180 | 0 | 0 |  |
| 2023 | missed | 22.150 | 64249 | 2.249 | 2.249 | 41864 | 0 | 0 |  |
| 2024 | missed | 22.150 | 38561 | 3.221 | 3.221 | 23083 | 0 | 0 |  |

Verdict (d): GameZone places text-threes on the right side of the line almost always, with a few percent inside it. Against the line in force (20.75 ft through 2018-19, 22.15 ft after) charted from the textbook rim, 0.01-0.10% of text-threes fall inside it in 2012-13 to 2018-19, 1.8-4.0% in 2008-09 to 2011-12 (a different vintage), and 2.2-4.4% in 2019-20 to 2023-24 (2.7%, 2.2%, 3.3% in 2022, 2023, 2024). The audit convention of 22.15 ft for every season gives 4.1-6.6% for 2012-13 to 2018-19, which reflects the old line, not a placement error. ESPN (CBBD) on the same joined 2022-2024 attempts is at 1.7% inside 22.15 ft, GameZone at 2.8%; neither depends on outcome (GameZone made vs missed 2.2-3.5% vs 2.3-3.2% by season). Text two-point jumpers charted outside the line: 0.00% in 2019-20 to 2023-24 and 0.36-1.07% in 2008-09 to 2018-19 against the 20.75 ft line. The vendor three flag and its own text agree on every row. With the dunk-centroid basket instead of the textbook rim the inside-line share roughly doubles in 2012-13 to 2023-24 (5.8% vs 2.7% in 2022), so basket choice matters as much as vendor.
