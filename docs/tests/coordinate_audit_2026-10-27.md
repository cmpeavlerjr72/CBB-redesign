# Coordinate-placement audit (2026-10-27)

Diagnostic worker output for PM ruling 4 of 2026-10-25 (`PROJECT_STATUS.md`, fg_make PAE round): can shot x/y be trusted as an upstream input, or is the location signal partly outcome-dependent charting? Script `scripts/diag_coordinate_audit_v1.py`; tables below are also in `docs/tests/coordinate_audit_2026-10-27.json`. Descriptive; nothing is adopted or wired.

## Scope, definitions, floors

- **Diagnostic only.** No model is fit, nothing is wired into the engine. Script: `scripts/diag_coordinate_audit_v1.py` (`--part master|1|2|3|report|all`). Intermediate tables: `results/coordinate_audit/` (gitignored).
- **Seasons.** season = ending year (2025 = 2024-25). Outcome tables use 2022-2025 only. Season 2026 (2025-26) is sealed for selection: it is read for non-outcome structure only (coverage, pixel and distance structure, text-vs-position accuracy), `made` is set to NaN for it at load time, and no make rate, no outcome comparison and no rating test touches it. Rows labelled 2026 are descriptive and play no role in any verdict.
- **Universe.** CBBD FGA rows (de-duplicated, `E.classify_frame`), the same universe and classes as `build_pae_shot_rating_v1.py` (rim includes jumpers located within 2.27 ft of the rim: "build" class). The "feed" class has no location override and is used where the located share is the outcome (it cannot depend on location). hoopR rows are joined by ESPN play id (hoopR `id` == CBBD `sourceId`; 99.9-100% of CBBD FGA rows have a hoopR row).
- **Frame.** CBBD x is folded to the left half (`x > 470 -> 940 - x`), so the basket is at (52.5, 250) in tenths of feet; distances are Euclidean feet from that point. hoopR distances are `sqrt(raw_y^2 + y^2)` (hoopR's raw frame is basket-relative; verified by the dunk medians). hoopR vs CBBD position comparisons use folded baseline distance and |sideline offset| because the two frames flip left/right differently.
- **Pixel.** A (folded x, y) pair on CBBD's 101 x 101 grid. **Canned** (as instructed) = a pixel holding more than 0.5% of a season's located attempts (all classes pooled). **Spike pixel** = a canned pixel whose count is at least 3 times the mean of its 8 grid neighbours; **ring** = those 8 neighbours.
- **Floors.** Paired comparisons are within game: the difference is computed in every game that holds both groups and averaged with weights n_a n_b / (n_a + n_b); the floor is the standard deviation of that estimate over 300 bootstrap resamples of games (slopes: 200 resamples). Pooled (unpaired) rates are reported next to it. Slope differences add the two bootstrap SEs in quadrature, which is approximate because the groups share games.
- **Underpowered.** Any cell with fewer than 500 attempts in either group is flagged UNDERPOWERED in the `flag` column and is not read as signal or as absence of signal.
- **Text classes (3e).** From `playText` only, independent of position: "three point" -> three; layup / dunk / tip -> rim text; "jumper" without "three point" -> two-point jumper. Arc = 22.15 ft for all seasons (instructed); the 20.75 ft and 0.5 / 1.5 ft slack variants are shown because coordinates sit on a 0.94 x 0.5 ft grid.

## Part 1. Inventory and reconciliation of on-disk coordinate sources

**Verdict, part 1 (what the on-disk sources are).** There are two independent coordinate sources on disk, not three. `data/raw/hoopr/shots/shots_YYYY.parquet` is a verbatim copy of the hoopR `pbp` rows with `shooting_play == True` (same 12 key columns identical as a multiset in all five seasons; it came from `sportsdataverse/hoopR-mbb-data` `mbb/shots/parquet/`, pulled by `scripts/pull_hoopr.py` on 2026-09-10); it adds no coordinates and no play id, and it includes free-throw rows. So the two sources are hoopR pbp (ESPN values, integer feet, centre-court frame, basket at |x| = 41.75 ft, x sign fixed by home/away so there is no half-time switching) and CBBD plays (a 101 x 101 grid on 94 x 50 ft, steps of 0.94 ft by 0.5 ft, x switching ends at half-time in 99.98% of team-games, per the 10-25 coverage report). They are NOT copies of each other. In 2022-2024 they locate different attempts (game-level coverage correlation -0.66 to -0.70) and on the 7-11k plays both locate, only 9-10% agree within 1 ft and 43-48% disagree by more than 3 ft; in 2025-26 they agree within 1 ft on 91-95% of 190k-650k pairs (the same ESPN value, re-quantised), but the 2025 agreement exists only from the week of 2025-02-10 (hoopR's own located share jumps from 3.9% to 87.6% that week; part 3d). The share of exactly equal pairs is 0.0-0.9%, which is a quantisation fact, not disagreement. Neither source locates every attempt: the UNION coverage is 90.5 / 91.9 / 92.8 / 99.2% for 2022-2025 (CBBD alone 80.0 / 84.3 / 87.5 / 98.6%; the 2025-26 sealed season 99.9% union vs 87.8% CBBD, descriptive only). hoopR-only attempts add 10.5 / 7.6 / 5.3 / 0.6 points, but in 2022-2024 they are a different measurement vintage than CBBD's values, so a union table would mix two incompatible placements. On the same plays, hoopR's 2022-24 values are far more consistent with the play text than CBBD's (layups/dunks/tips more than 6 ft from the rim: 0.1-0.2% vs 10.5-13.7%; text-threes charted inside the arc: 0.1-0.3% vs 2.4-3.1%); in 2025-26 the two sources are equally (im)precise. Which source is closer to physical truth cannot be settled with files on disk; text consistency is the only truth-independent yardstick and it favours hoopR in 2022-24 on the 6-12% of attempts it covers.

### 1a. hoopR pbp coordinate frame and FGA coverage

| season | fga_rows | located_pct | int32_sentinel_rows | null_pct | team_periods | x_sign_matches_home_flag_pct | share_team_periods_one_side_ge95pct | share_team_games_switch_side_at_half |
|---|---|---|---|---|---|---|---|---|
| 2022 | 672063 | 12.110 | 0 | 87.890 | 2796 | 99.910 | 0.999 | 0 |
| 2023 | 701680 | 8.870 | 0 | 91.130 | 2144 | 99.920 | 0.997 | 0 |
| 2024 | 724063 | 6.280 | 0 | 93.720 | 1536 | 99.930 | 0.999 | 0 |
| 2025 | 716538 | 26.860 | 5 | 73.140 | 6644 | 99.620 | 1.000 | 0 |
| 2026 | 738245 | 100 | 0 | 0 | 25099 | 99.880 | 0.999 | 0.000 |

| season | x_min | x_max | y_min | y_max | raw_x_min | raw_x_max | raw_y_min | raw_y_max | dunk_median_abs_x | dunk_median_abs_y | distinct_x | distinct_y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | -47.250 | 47.250 | -25 | 25 | 0 | 50 | -5 | 89 | 40.750 | 1 | 120 | 51 |
| 2023 | -46.750 | 47.250 | -25 | 25 | 0 | 50 | -5 | 89 | 40.750 | 1 | 116 | 51 |
| 2024 | -46.750 | 45.750 | -25 | 25 | 1 | 50 | -5 | 83 | 40.750 | 1 | 104 | 51 |
| 2025 | -46.750 | 46.750 | -25 | 25 | 0 | 50 | -5 | 85 | 39.750 | 1 | 141 | 51 |
| 2026 | -46.750 | 46.750 | -25 | 25 | 0 | 50 | -5 | 87 | 39.750 | 1 | 180 | 51 |

### 1b. hoopR shots_YYYY.parquet vs pbp shooting rows

| season | shots_rows | pbp_shooting_play_rows | multiset_identical_on_12_cols | shots_rows_in_pbp_share | shots_coordinate_x_nonnull_pct | shots_coordinate_x_sentinel_pct | shots_has_play_id |
|---|---|---|---|---|---|---|---|
| 2022 | 873219 | 873219 | True | 1 | 12.100 | 0.025 | False |
| 2023 | 920595 | 920595 | True | 1 | 8.830 | 0.015 | False |
| 2024 | 957173 | 957173 | True | 1 | 6.290 | 0.009 | False |
| 2025 | 936510 | 936510 | True | 1 | 25.930 | 0.001 | False |
| 2026 | 991836 | 991836 | True | 1 | 100 | 0.202 | False |

### 1c. CBBD plays: coverage and frame

| season | fga | cbbd_located_pct | hoopr_joined_pct | hoopr_located_pct |
|---|---|---|---|---|
| 2022 | 672277 | 80.005 | 99.972 | 12.102 |
| 2023 | 701446 | 84.335 | 99.979 | 8.822 |
| 2024 | 724063 | 87.545 | 100 | 6.283 |
| 2025 | 716446 | 98.627 | 99.993 | 26.864 |
| 2026 | 736935 | 87.758 | 99.911 | 99.911 |

| season | x_min | x_max | y_min | y_max | distinct_x | distinct_y | x_on_9.4_grid_pct | y_on_5_grid_pct | dunk_median_folded_x | dunk_median_y | share_x_left_half |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 0 | 940 | 0 | 500 | 101 | 101 | 100 | 100 | 65.800 | 250 | 0.499 |
| 2023 | 0 | 940 | 0 | 500 | 101 | 101 | 100 | 100 | 65.800 | 250 | 0.499 |
| 2024 | 0 | 940 | 0 | 500 | 101 | 101 | 100 | 100 | 65.800 | 250 | 0.500 |
| 2025 | 0 | 940 | 0 | 500 | 101 | 101 | 100 | 100 | 65.800 | 250 | 0.500 |
| 2026 | 0 | 940 | 0 | 500 | 101 | 101 | 100 | 100 | 65.800 | 250 | 0.500 |

### 1d. Reconciliation (CBBD attempts are the universe; join key hoopR `id` == CBBD `sourceId`)

| season | attempts | cbbd_located | hoopr_located | both | cbbd_only | hoopr_only | neither | cbbd_pct | hoopr_pct | union_pct | pair_exact_pct | pair_le0.5ft_pct | pair_le1ft_pct | pair_1_to_3ft_pct | pair_gt3ft_pct | pair_gt10ft_pct | pair_dist_corr | pair_median_abs_dist_diff_ft | cbbd_rows_without_hoopr_row |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 672277 | 537854 | 81360 | 10843 | 527011 | 70517 | 63906 | 80 | 12.100 | 90.490 | 0.050 | 1.900 | 10.230 | 43.190 | 46.580 | 4.660 | 0.919 | 1.310 | 188 |
| 2023 | 701446 | 591564 | 61885 | 8678 | 582886 | 53207 | 56675 | 84.330 | 8.820 | 91.920 | 0.010 | 1.990 | 9.350 | 42.730 | 47.930 | 4.390 | 0.937 | 1.363 | 145 |
| 2024 | 724063 | 633884 | 45491 | 7285 | 626599 | 38206 | 51973 | 87.550 | 6.280 | 92.820 | 0.010 | 2.060 | 10.030 | 46.620 | 43.350 | 2.770 | 0.950 | 1.248 | 0 |
| 2025 | 716446 | 706606 | 192467 | 188125 | 518481 | 4342 | 5498 | 98.630 | 26.860 | 99.230 | 0.860 | 27.340 | 91.250 | 5.960 | 2.790 | 1.620 | 0.954 | 0.326 | 50 |
| 2026 | 736935 | 646718 | 736281 | 646531 | 187 | 89750 | 467 | 87.760 | 99.910 | 99.940 | 0.890 | 27.880 | 94.740 | 5.240 | 0.020 | 0.010 | 0.995 | 0.318 | 654 |

Text-consistency of each source on the SAME attempts (both located)

| season | subset | source | n_three | n_rimtext | n_twojump | three_inside_arc_pct | twojump_outside_arc_pct | rimtext_gt6ft_pct |
|---|---|---|---|---|---|---|---|---|
| 2022 | all | cbbd | 3637 | 4347 | 2859 | 2.750 | 0.590 | 13.140 |
| 2022 | all | hoopr | 3637 | 4347 | 2859 | 0.110 | 0.350 | 0.210 |
| 2022 | makes | cbbd | 1177 | 2696 | 1063 | 2.460 | 0 | 8.090 |
| 2022 | makes | hoopr | 1177 | 2696 | 1063 | 0.080 | 0.090 | 0.040 |
| 2022 | misses | cbbd | 2460 | 1651 | 1796 | 2.890 | 0.950 | 21.380 |
| 2022 | misses | hoopr | 2460 | 1651 | 1796 | 0.120 | 0.500 | 0.480 |
| 2023 | all | cbbd | 2920 | 3415 | 2343 | 2.360 | 0.550 | 13.650 |
| 2023 | all | hoopr | 2920 | 3415 | 2343 | 0.340 | 0.380 | 0.200 |
| 2023 | makes | cbbd | 977 | 2102 | 854 | 1.540 | 0 | 7.520 |
| 2023 | makes | hoopr | 977 | 2102 | 854 | 0.410 | 0.350 | 0.140 |
| 2023 | misses | cbbd | 1943 | 1313 | 1489 | 2.780 | 0.870 | 23.460 |
| 2023 | misses | hoopr | 1943 | 1313 | 1489 | 0.310 | 0.400 | 0.300 |
| 2024 | all | cbbd | 2553 | 2866 | 1866 | 3.130 | 0.430 | 10.500 |
| 2024 | all | hoopr | 2553 | 2866 | 1866 | 0.200 | 0.380 | 0.070 |
| 2024 | makes | cbbd | 865 | 1704 | 751 | 2.660 | 0 | 5.580 |
| 2024 | makes | hoopr | 865 | 1704 | 751 | 0.120 | 0.130 | 0 |
| 2024 | misses | cbbd | 1688 | 1162 | 1115 | 3.380 | 0.720 | 17.730 |
| 2024 | misses | hoopr | 1688 | 1162 | 1115 | 0.240 | 0.540 | 0.170 |
| 2025 | all | cbbd | 73416 | 69900 | 44728 | 1.840 | 0.150 | 8.570 |
| 2025 | all | hoopr | 73416 | 69900 | 44728 | 5.620 | 0.200 | 9.090 |
| 2025 | makes | cbbd | 25049 | 40741 | 17181 | 1.760 | 0.080 | 5.280 |
| 2025 | makes | hoopr | 25049 | 40741 | 17181 | 5.760 | 0.150 | 5.750 |
| 2025 | misses | cbbd | 48367 | 29159 | 27547 | 1.890 | 0.200 | 13.170 |
| 2025 | misses | hoopr | 48367 | 29159 | 27547 | 5.550 | 0.230 | 13.750 |
| 2026 | all | cbbd | 256165 | 249370 | 128605 | 1.660 | 0.170 | 8.870 |
| 2026 | all | hoopr | 256165 | 249370 | 128605 | 2.910 | 0.280 | 9.790 |

Pair disagreement split by whether the CBBD pixel is canned

| season | cbbd_pixel_canned | n_pairs | pair_le1ft_pct | pair_gt3ft_pct |
|---|---|---|---|---|
| 2022 | True | 1266 | 24.170 | 14.850 |
| 2022 | False | 9577 | 8.380 | 50.780 |
| 2023 | True | 950 | 20.420 | 16.110 |
| 2023 | False | 7728 | 7.980 | 51.840 |
| 2024 | True | 914 | 19.800 | 13.680 |
| 2024 | False | 6371 | 8.630 | 47.610 |
| 2025 | True | 25484 | 97.010 | 0.630 |
| 2025 | False | 162641 | 90.340 | 3.130 |
| 2026 | True | 90504 | 99.950 | 0.020 |
| 2026 | False | 556027 | 93.890 | 0.030 |

Game-level coverage, hoopR vs CBBD

| season | games | hoopr_games_zero_pct | hoopr_games_ge90_pct | cbbd_games_zero_pct | cbbd_games_ge90_pct | corr_game_cov_hoopr_vs_cbbd |
|---|---|---|---|---|---|---|
| 2022 | 5832 | 88.014 | 11.986 | 5.110 | 76.286 | -0.703 |
| 2023 | 6115 | 91.284 | 8.716 | 3.630 | 81.145 | -0.693 |
| 2024 | 6151 | 93.757 | 6.243 | 2.569 | 84.767 | -0.664 |
| 2025 | 6135 | 72.926 | 27.074 | 0.163 | 98.957 | -0.080 |
| 2026 | 6263 | 0.064 | 99.920 | 11.432 | 88.440 | 0.052 |


## Part 2. Coverage structure

**Verdict, part 2 (which games lack coordinates).** CBBD is the best source for 2022-2025 (hoopR in the sealed 2026 season), so tables use CBBD with the union alongside. Missingness is NOT decided per game alone. Whole games at 0% are rare (5.1 / 3.6 / 2.6 / 0.2%) and fully located games are only 30-44% of games in 2022-2025; 56-67% of games are partially located. But missingness is strongly clustered at every nested level: the share of attempts in units that are entirely located or entirely missing is 37% for games (0.0% expected if each attempt were located independently), 55% for game x period, 72% for game x team x period, 76% for game x period x 5-minute block and 86% for game x shooter (20% expected) in 2022. In other words it comes in runs inside a game (a period, a team's stretch, a player's stretch). The missing set is a biased subset in 2022-2024 and nearly unbiased in 2025. By CBBD coverage (2022 / 2023 / 2024 / 2025): power-conference home teams 65.5 / 73.5 / 81.0 / 97.9% vs low-tier 84.8 / 87.8 / 90.0 / 98.9%; neutral-site games 51.7 / 60.9 / 61.9 / 98.0% vs 83.7 / 87.6 / 90.8 / 98.7%; postseason 31.7 / 27.1 / 46.5 / 94.5%; March 63.7 / 64.4 / 75.7 / 98.5% vs January 84.8 / 89.3 / 92.4 / 98.2%; games with no line in the lines files 58.1 / 62.0 / 76.2 / 98.6% vs 82.1 / 85.9 / 89.6 / 98.6% (and 29.6% of no-line games in 2022 are entirely unlocated vs 2.9% of lined games). The games CBBD lacks are disproportionately the high-profile games (power conference, neutral site, March, tournament) and those are the games hoopR does locate (union coverage for power-conference games 95.9 / 96.4 / 96.0%), while the lowest-profile games with no line are the ones neither source locates. Per team, the located offence attempts are p10 1,035 / 1,283 / 1,482 / 1,672, median 1,536 / 1,667 / 1,739 / 1,860 (minimum 0 / 0 / 0 / 761, and 29 / 18 / 10 / 0 teams below 50% located); defence p10 1,051 / 1,265 / 1,460 / 1,663 (minimum 510 / 676 / 525 / 1,009). Only 2025 is close to complete and even; before that, any per-game or per-team location feature is built on a non-random 80-88%.

### 2a. Is missingness decided per game?

| season | games | attempt_cov_pct | games_0pct | games_1_99pct | games_100pct | attempts_in_partial_games_pct | obs_over_indep_variance_ratio |
|---|---|---|---|---|---|---|---|
| 2022 | 5832 | 80.005 | 5.110 | 62.397 | 32.493 | 62.704 | 89.303 |
| 2023 | 6115 | 84.335 | 3.630 | 63.729 | 32.641 | 64.124 | 86.162 |
| 2024 | 6151 | 87.545 | 2.569 | 67.127 | 30.304 | 67.330 | 84.078 |
| 2025 | 6135 | 98.627 | 0.163 | 56.153 | 43.684 | 56.519 | 36.965 |
| 2026 | 6263 | 87.758 | 11.432 | 15.376 | 73.192 | 15.357 | 110.736 |

Share of attempts that sit in a unit located 0% or 100% (observed vs expected if each attempt were located independently at the season rate)

| season | unit | units | mean_attempts_per_unit | pure_share_observed_pct | pure_share_if_independent_pct |
|---|---|---|---|---|---|
| 2022 | game | 5832 | 115.274 | 37.296 | 0.000 |
| 2022 | game x period | 11974 | 56.145 | 54.851 | 0.028 |
| 2022 | game x team | 11664 | 57.637 | 58.592 | 0.001 |
| 2022 | game x team x period | 23948 | 28.072 | 72.167 | 0.350 |
| 2022 | game x period x 5-min block | 46971 | 14.313 | 76.231 | 4.357 |
| 2022 | game x shooter | 104682 | 6.418 | 85.958 | 19.744 |
| 2023 | game | 6115 | 114.709 | 35.876 | 0.000 |
| 2023 | game x period | 12579 | 55.763 | 54.967 | 0.071 |
| 2023 | game x team | 12228 | 57.364 | 58.865 | 0.010 |
| 2023 | game x team x period | 25156 | 27.884 | 73.301 | 1.102 |
| 2023 | game x period x 5-min block | 49267 | 14.238 | 78.042 | 8.974 |
| 2023 | game x shooter | 109383 | 6.410 | 87.608 | 26.892 |
| 2024 | game | 6151 | 117.715 | 32.670 | 0.000 |
| 2024 | game x period | 12646 | 57.256 | 52.981 | 0.149 |
| 2024 | game x team | 12302 | 58.857 | 55.677 | 0.054 |
| 2024 | game x team x period | 25292 | 28.628 | 72.138 | 2.445 |
| 2024 | game x period x 5-min block | 49560 | 14.610 | 77.986 | 14.278 |
| 2024 | game x shooter | 110492 | 6.540 | 88.177 | 33.666 |
| 2025 | game | 6135 | 116.780 | 43.481 | 19.828 |
| 2025 | game x period | 12587 | 56.920 | 64.965 | 44.812 |
| 2025 | game x team | 12269 | 58.395 | 64.235 | 44.259 |
| 2025 | game x team x period | 25172 | 28.462 | 79.547 | 66.618 |
| 2025 | game x period x 5-min block | 49385 | 14.507 | 88.699 | 81.316 |
| 2025 | game x shooter | 110492 | 6.478 | 92.793 | 87.933 |
| 2026 | game | 6263 | 117.665 | 84.643 | 0.000 |
| 2026 | game x period | 12850 | 57.349 | 91.339 | 0.153 |
| 2026 | game x team | 12517 | 58.875 | 91.230 | 0.063 |
| 2026 | game x team x period | 25681 | 28.696 | 95.276 | 2.584 |
| 2026 | game x period x 5-min block | 50431 | 14.613 | 97.361 | 14.772 |
| 2026 | game x shooter | 113449 | 6.488 | 98.350 | 34.608 |

### 2b. coverage by_season

| season | games | attempts | cbbd_located_pct | union_located_pct | games_0pct | games_1_99pct | games_100pct |
|---|---|---|---|---|---|---|---|
| 2022 | 5832 | 672277 | 80.005 | 90.494 | 5.110 | 62.397 | 32.493 |
| 2023 | 6115 | 701446 | 84.335 | 91.920 | 3.630 | 63.729 | 32.641 |
| 2024 | 6151 | 724063 | 87.545 | 92.822 | 2.569 | 67.127 | 30.304 |
| 2025 | 6135 | 716446 | 98.627 | 99.233 | 0.163 | 56.153 | 43.684 |
| 2026 | 6263 | 736935 | 87.758 | 99.937 | 11.432 | 15.376 | 73.192 |

### 2b. coverage by_season_month

| season | month | games | attempts | cbbd_located_pct | union_located_pct | games_0pct | games_1_99pct | games_100pct |
|---|---|---|---|---|---|---|---|---|
| 2022 | 1 | 1384 | 157713 | 84.807 | 94.401 | 2.818 | 61.705 | 35.477 |
| 2022 | 2 | 1445 | 163936 | 86.470 | 94.320 | 2.353 | 61.453 | 36.194 |
| 2022 | 3 | 625 | 71798 | 63.665 | 94.102 | 4.800 | 72.160 | 23.040 |
| 2022 | 4 | 4 | 507 | 17.554 | 100 | 0 | 100 | 0 |
| 2022 | 11 | 1297 | 152367 | 77.029 | 83.706 | 8.867 | 61.141 | 29.992 |
| 2022 | 12 | 1077 | 125956 | 78.742 | 86.740 | 7.428 | 60.260 | 32.312 |
| 2023 | 1 | 1447 | 163047 | 89.309 | 95.155 | 2.004 | 63.994 | 34.001 |
| 2023 | 2 | 1384 | 157154 | 88.187 | 95.826 | 1.301 | 63.656 | 35.043 |
| 2023 | 3 | 549 | 63506 | 64.377 | 95.807 | 2.914 | 73.224 | 23.862 |
| 2023 | 4 | 3 | 340 | 10 | 100 | 0 | 100 | 0 |
| 2023 | 11 | 1460 | 170410 | 81.793 | 85.153 | 6.849 | 61.507 | 31.644 |
| 2023 | 12 | 1272 | 146989 | 86.440 | 90.303 | 4.638 | 61.871 | 33.491 |
| 2024 | 1 | 1390 | 161752 | 92.408 | 96.058 | 0.863 | 68.633 | 30.504 |
| 2024 | 2 | 1380 | 159921 | 92.386 | 96.641 | 0.435 | 65.797 | 33.768 |
| 2024 | 3 | 799 | 92947 | 75.736 | 97.861 | 0.626 | 69.337 | 30.038 |
| 2024 | 4 | 6 | 733 | 16.508 | 100 | 0 | 100 | 0 |
| 2024 | 11 | 1363 | 163432 | 84.971 | 86.072 | 5.503 | 67.351 | 27.146 |
| 2024 | 12 | 1213 | 145278 | 87.613 | 89.348 | 4.946 | 65.045 | 30.008 |
| 2025 | 1 | 1437 | 165556 | 98.158 | 99.117 | 0 | 59.708 | 40.292 |
| 2025 | 2 | 1362 | 156556 | 98.820 | 99.693 | 0.073 | 42.878 | 57.048 |
| 2025 | 3 | 765 | 89157 | 98.468 | 99.996 | 0.784 | 54.379 | 44.837 |
| 2025 | 4 | 17 | 2064 | 99.467 | 100 | 0 | 47.059 | 52.941 |
| 2025 | 11 | 1431 | 169865 | 98.714 | 98.719 | 0.140 | 62.683 | 37.177 |
| 2025 | 12 | 1123 | 133248 | 98.964 | 98.968 | 0.089 | 60.730 | 39.181 |
| 2026 | 1 | 1556 | 180478 | 90.894 | 99.978 | 8.933 | 10.283 | 80.784 |
| 2026 | 2 | 1370 | 157983 | 98.008 | 99.994 | 1.752 | 22.993 | 75.255 |
| 2026 | 3 | 613 | 71881 | 99.083 | 99.993 | 0.163 | 46.819 | 53.018 |
| 2026 | 4 | 13 | 1603 | 84.342 | 100 | 15.385 | 23.077 | 61.538 |
| 2026 | 11 | 1540 | 185179 | 82.013 | 99.855 | 16.818 | 7.727 | 75.455 |
| 2026 | 12 | 1171 | 139811 | 73.953 | 99.897 | 24.851 | 6.746 | 68.403 |

### 2b. coverage by_season_tier

| season | tier | games | attempts | cbbd_located_pct | union_located_pct | games_0pct | games_1_99pct | games_100pct |
|---|---|---|---|---|---|---|---|---|
| 2022 | low | 4122 | 474823 | 84.804 | 88.267 | 6.696 | 58.054 | 35.250 |
| 2022 | mid | 202 | 22597 | 91.052 | 95.800 | 1.980 | 59.901 | 38.119 |
| 2022 | power | 1508 | 174857 | 65.544 | 95.857 | 1.194 | 74.602 | 24.204 |
| 2023 | low | 4388 | 503127 | 87.767 | 90.222 | 4.717 | 60.552 | 34.731 |
| 2023 | mid | 209 | 23474 | 91.787 | 95.267 | 1.435 | 66.507 | 32.057 |
| 2023 | power | 1518 | 174845 | 73.459 | 96.357 | 0.791 | 72.530 | 26.680 |
| 2024 | low | 4312 | 506791 | 89.987 | 91.698 | 3.409 | 66.234 | 30.357 |
| 2024 | mid | 205 | 23475 | 88.720 | 90.756 | 3.415 | 57.561 | 39.024 |
| 2024 | power | 1634 | 193797 | 81.019 | 96.012 | 0.245 | 70.685 | 29.070 |
| 2025 | low | 4345 | 506633 | 98.861 | 99.192 | 0.207 | 56.364 | 43.429 |
| 2025 | mid | 204 | 23286 | 99.055 | 99.287 | 0 | 55.882 | 44.118 |
| 2025 | power | 1586 | 186527 | 97.936 | 99.337 | 0.063 | 55.612 | 44.325 |
| 2026 | low | 4399 | 515162 | 84.652 | 99.940 | 14.299 | 14.867 | 70.834 |
| 2026 | mid | 233 | 26649 | 88.394 | 99.917 | 11.159 | 12.017 | 76.824 |
| 2026 | power | 1630 | 195055 | 95.905 | 99.930 | 3.681 | 17.239 | 79.080 |

### 2b. coverage by_season_neutral

| season | neutralSite | games | attempts | cbbd_located_pct | union_located_pct | games_0pct | games_1_99pct | games_100pct |
|---|---|---|---|---|---|---|---|---|
| 2022 | False | 5158 | 595207 | 83.673 | 91.019 | 4.886 | 61.051 | 34.064 |
| 2022 | True | 674 | 77070 | 51.673 | 86.437 | 6.825 | 72.700 | 20.475 |
| 2023 | False | 5364 | 615693 | 87.603 | 92.531 | 3.561 | 62.435 | 34.004 |
| 2023 | True | 751 | 85753 | 60.870 | 87.534 | 4.128 | 72.969 | 22.903 |
| 2024 | False | 5448 | 641629 | 90.837 | 93.553 | 2.625 | 66.134 | 31.241 |
| 2024 | True | 703 | 82434 | 61.928 | 87.134 | 2.134 | 74.822 | 23.044 |
| 2025 | False | 5414 | 632700 | 98.712 | 99.234 | 0.055 | 55.929 | 44.016 |
| 2025 | True | 721 | 83746 | 97.978 | 99.224 | 0.971 | 57.836 | 41.193 |
| 2026 | False | 5594 | 658230 | 87.517 | 99.930 | 11.691 | 14.015 | 74.294 |
| 2026 | True | 668 | 78636 | 89.851 | 99.996 | 9.132 | 26.796 | 64.072 |

### 2b. coverage by_season_has_line

| season | has_line | games | attempts | cbbd_located_pct | union_located_pct | games_0pct | games_1_99pct | games_100pct |
|---|---|---|---|---|---|---|---|---|
| 2022 | False | 473 | 59354 | 58.075 | 58.075 | 29.598 | 52.008 | 18.393 |
| 2022 | True | 5359 | 612923 | 82.128 | 93.633 | 2.948 | 63.314 | 33.738 |
| 2023 | False | 375 | 46880 | 62.005 | 62.005 | 26.667 | 54.667 | 18.667 |
| 2023 | True | 5740 | 654566 | 85.934 | 94.063 | 2.125 | 64.321 | 33.554 |
| 2024 | False | 913 | 112527 | 76.224 | 78.528 | 13.363 | 60.898 | 25.739 |
| 2024 | True | 5238 | 611536 | 89.629 | 95.452 | 0.687 | 68.213 | 31.100 |
| 2025 | False | 810 | 99101 | 98.633 | 98.872 | 0.247 | 54.691 | 45.062 |
| 2025 | True | 5325 | 617345 | 98.626 | 99.291 | 0.150 | 56.376 | 43.474 |
| 2026 | False | 540 | 69357 | 25.040 | 99.624 | 74.630 | 2.407 | 22.963 |
| 2026 | True | 5722 | 667509 | 94.284 | 99.969 | 5.453 | 16.603 | 77.945 |

### 2b. coverage by_season_seasontype

| season | seasonType | games | attempts | cbbd_located_pct | union_located_pct | games_0pct | games_1_99pct | games_100pct |
|---|---|---|---|---|---|---|---|---|
| 2022 | postseason | 130 | 15265 | 31.739 | 90.783 | 9.231 | 78.462 | 12.308 |
| 2022 | regular | 5702 | 657012 | 81.126 | 90.487 | 5.016 | 62.031 | 32.953 |
| 2023 | postseason | 112 | 13405 | 27.132 | 87.848 | 11.607 | 78.571 | 9.821 |
| 2023 | regular | 6003 | 688041 | 85.449 | 92.000 | 3.482 | 63.452 | 33.067 |
| 2024 | postseason | 120 | 14431 | 46.525 | 99.744 | 0 | 82.500 | 17.500 |
| 2024 | regular | 6031 | 709632 | 88.380 | 92.681 | 2.620 | 66.821 | 30.559 |
| 2025 | postseason | 123 | 14801 | 94.480 | 100 | 4.878 | 51.220 | 43.902 |
| 2025 | regular | 6012 | 701645 | 98.714 | 99.216 | 0.067 | 56.254 | 43.679 |
| 2026 | postseason | 105 | 12763 | 96.419 | 100 | 2.857 | 43.810 | 53.333 |
| 2026 | regular | 6157 | 724103 | 87.614 | 99.936 | 11.564 | 14.894 | 73.542 |

### 2b. coverage by_season_conference_game

| season | conferenceGame | games | attempts | cbbd_located_pct | union_located_pct | games_0pct | games_1_99pct | games_100pct |
|---|---|---|---|---|---|---|---|---|
| 2022 | False | 2402 | 282184 | 74.427 | 84.522 | 8.743 | 61.699 | 29.559 |
| 2022 | True | 3430 | 390093 | 84.040 | 94.814 | 2.566 | 62.886 | 34.548 |
| 2023 | False | 2565 | 299227 | 79.767 | 86.178 | 7.057 | 62.612 | 30.331 |
| 2023 | True | 3550 | 402219 | 87.733 | 96.192 | 1.155 | 64.535 | 34.310 |
| 2024 | False | 2622 | 314835 | 83.778 | 87.589 | 5.301 | 66.972 | 27.727 |
| 2024 | True | 3529 | 409228 | 90.444 | 96.848 | 0.538 | 67.243 | 32.219 |
| 2025 | False | 2527 | 300902 | 98.595 | 98.871 | 0.356 | 61.377 | 38.267 |
| 2025 | True | 3608 | 415544 | 98.649 | 99.494 | 0.028 | 52.494 | 47.478 |
| 2026 | False | 2605 | 313455 | 79.098 | 99.869 | 19.655 | 9.021 | 71.324 |
| 2026 | True | 3658 | 423480 | 94.168 | 99.987 | 5.577 | 19.902 | 74.522 |

### 2b. coverage by_season_tier_has_line

| season | tier | has_line | games | attempts | cbbd_located_pct | union_located_pct | games_0pct | games_1_99pct | games_100pct |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | low | False | 456 | 57217 | 58.434 | 58.434 | 29.167 | 52.851 | 17.982 |
| 2022 | low | True | 3666 | 417606 | 88.417 | 92.354 | 3.901 | 58.702 | 37.398 |
| 2022 | mid | False | 10 | 1258 | 61.924 | 61.924 | 20 | 40 | 40 |
| 2022 | mid | True | 192 | 21339 | 92.769 | 97.797 | 1.042 | 60.938 | 38.021 |
| 2022 | power | False | 7 | 879 | 29.238 | 29.238 | 71.429 | 14.286 | 14.286 |
| 2022 | power | True | 1501 | 173978 | 65.728 | 96.193 | 0.866 | 74.883 | 24.250 |
| 2023 | low | False | 362 | 45235 | 61.972 | 61.972 | 26.796 | 54.420 | 18.785 |
| 2023 | low | True | 4026 | 457892 | 90.315 | 93.013 | 2.732 | 61.103 | 36.165 |
| 2023 | mid | False | 9 | 1141 | 74.145 | 74.145 | 11.111 | 77.778 | 11.111 |
| 2023 | mid | True | 200 | 22333 | 92.688 | 96.346 | 1 | 66 | 33 |
| 2023 | power | False | 4 | 504 | 37.500 | 37.500 | 50 | 25 | 25 |
| 2023 | power | True | 1514 | 174341 | 73.563 | 96.527 | 0.661 | 72.655 | 26.684 |
| 2024 | low | False | 785 | 97436 | 76.284 | 77.029 | 14.395 | 61.274 | 24.331 |
| 2024 | low | True | 3527 | 409355 | 93.248 | 95.190 | 0.964 | 67.338 | 31.698 |
| 2024 | mid | False | 31 | 3651 | 64.311 | 64.311 | 22.581 | 45.161 | 32.258 |
| 2024 | mid | True | 174 | 19824 | 93.215 | 95.627 | 0 | 59.770 | 40.230 |
| 2024 | power | False | 97 | 11440 | 79.519 | 95.830 | 2.062 | 62.887 | 35.052 |
| 2024 | power | True | 1537 | 182357 | 81.113 | 96.023 | 0.130 | 71.178 | 28.692 |
| 2025 | low | False | 744 | 91326 | 98.821 | 98.826 | 0.269 | 55.242 | 44.489 |
| 2025 | low | True | 3601 | 415307 | 98.870 | 99.272 | 0.194 | 56.595 | 43.210 |
| 2025 | mid | False | 13 | 1618 | 99.382 | 99.382 | 0 | 53.846 | 46.154 |
| 2025 | mid | True | 191 | 21668 | 99.031 | 99.280 | 0 | 56.021 | 43.979 |
| 2025 | power | False | 53 | 6157 | 95.647 | 99.415 | 0 | 47.170 | 52.830 |
| 2025 | power | True | 1533 | 180370 | 98.014 | 99.334 | 0.065 | 55.903 | 44.031 |
| 2026 | low | False | 528 | 67829 | 24.842 | 99.615 | 74.811 | 2.462 | 22.727 |
| 2026 | low | True | 3871 | 447333 | 93.721 | 99.989 | 6.045 | 16.559 | 77.396 |
| 2026 | mid | False | 9 | 1141 | 45.311 | 100 | 55.556 | 0 | 44.444 |
| 2026 | mid | True | 224 | 25508 | 90.321 | 99.914 | 9.375 | 12.500 | 78.125 |
| 2026 | power | False | 3 | 387 | 0 | 100 | 100 | 0 | 0 |
| 2026 | power | True | 1627 | 194668 | 96.096 | 99.930 | 3.503 | 17.271 | 79.226 |

### 2b. Composition of games by located status (0% / partial / full)

| season | group | games | power_pct | mid_pct | low_pct | neutral_pct | has_line_pct | march_apr_pct | mean_attempts |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | full | 1895 | 19.261 | 4.063 | 76.675 | 7.282 | 95.409 | 7.599 | 113.687 |
| 2022 | partial | 3639 | 30.915 | 3.325 | 65.760 | 13.465 | 93.240 | 12.503 | 115.841 |
| 2022 | zero | 298 | 6.040 | 1.342 | 92.617 | 15.436 | 53.020 | 10.067 | 118.436 |
| 2023 | full | 1996 | 20.291 | 3.357 | 76.353 | 8.617 | 96.493 | 6.563 | 112.752 |
| 2023 | partial | 3897 | 28.253 | 3.567 | 68.181 | 14.062 | 94.740 | 10.393 | 115.421 |
| 2023 | zero | 222 | 5.405 | 1.351 | 93.243 | 13.964 | 54.955 | 7.207 | 119.802 |
| 2024 | full | 1864 | 25.483 | 4.292 | 70.225 | 8.691 | 87.393 | 12.876 | 116.501 |
| 2024 | partial | 4129 | 27.973 | 2.858 | 69.169 | 12.739 | 86.534 | 13.563 | 118.071 |
| 2024 | zero | 158 | 2.532 | 4.430 | 93.038 | 9.494 | 22.785 | 3.165 | 122.728 |
| 2025 | full | 2680 | 26.231 | 3.358 | 70.410 | 11.082 | 86.381 | 13.134 | 115.794 |
| 2025 | partial | 3445 | 25.602 | 3.309 | 71.089 | 12.104 | 87.141 | 12.308 | 117.540 |
| 2025 | zero | 10 | 10 | 0 | 90 | 70 | 80 | 60 | 119.300 |
| 2026 | full | 4584 | 28.120 | 3.905 | 67.976 | 9.337 | 97.295 | 7.264 | 116.806 |
| 2026 | partial | 963 | 29.180 | 2.908 | 67.913 | 18.588 | 98.650 | 30.114 | 117.521 |
| 2026 | zero | 716 | 8.380 | 3.631 | 87.849 | 8.531 | 43.575 | 0.419 | 123.358 |

### 2c. Per-team located attempts per season (teams with >= 500 attempts)

| season | side | teams | located_attempts_min | located_attempts_p10 | located_attempts_p50 | located_pct_min | located_pct_p10 | located_pct_p50 | teams_below_50pct |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | offence | 360 | 0 | 1034.800 | 1536 | 0 | 57.080 | 89.444 | 29 |
| 2022 | defence | 360 | 510 | 1050.500 | 1495 | 20.808 | 59.295 | 85.996 | 22 |
| 2023 | offence | 363 | 0 | 1282.600 | 1667 | 0 | 67.552 | 91.884 | 18 |
| 2023 | defence | 363 | 676 | 1265 | 1603 | 35.139 | 68.497 | 89.173 | 9 |
| 2024 | offence | 365 | 0 | 1482 | 1739 | 0 | 78.134 | 93.482 | 10 |
| 2024 | defence | 365 | 525 | 1460.200 | 1686 | 46.976 | 77.578 | 90.323 | 3 |
| 2025 | offence | 365 | 761 | 1671.600 | 1860 | 88.305 | 96.748 | 99.151 | 0 |
| 2025 | defence | 365 | 1009 | 1662.800 | 1844 | 88.490 | 96.642 | 99.166 | 0 |
| 2026 | offence | 367 | 0 | 1477.600 | 1743 | 0 | 80.552 | 92.104 | 2 |
| 2026 | defence | 368 | 0 | 1479 | 1759 | 0 | 79.952 | 93.016 | 3 |


## Part 3. Placement audit

**Verdict, part 3 (is x/y trustworthy as an upstream input?).** Partly. Three- and two-point jumpers are placed consistently with the play text (text-threes charted inside 22.15 ft: 1.83 / 1.62 / 1.81 / 1.86%, inside 20.75 ft only 0.13-0.14%; text two-point jumpers charted outside 22.15 ft: 0.39 / 0.34 / 0.21 / 0.16%), and make rate falls with distance as it should. The rim is the problem, and the problem is outcome-dependent. (i) Charting accuracy depends on the result: layups/tips charted more than 6 ft from the rim are 6.1-6.8% of makes but 14.6-15.7% of misses in every season (within-game difference -8.4 to -8.9 points against a bootstrap floor of 0.2 points); the same asymmetry appears in hoopR's values for 2025-26 (5.8% vs 13.8% in 2025), so it is in the ESPN feed and not a CBBD processing artefact. (ii) Which attempts get a coordinate depends on the result: the located share of rim makes is 1.3 / 1.6 / 1.7 / 2.1 points lower than that of rim misses within game (floor 0.1 point; z = -13 to -36; 2.1 to 3.6 points in games with partial coverage); jump2 is smaller (0.3-1.1 points) and threes are outcome-blind (within +/-0.2 points). (iii) The 2025 gap is reproduced and explained: located rim attempts make 58.1% vs 81.5% for the 5,510 unlocated ones (2022-24: 58.4 / 58.3 / 57.7% vs 61.4 / 62.3 / 61.7%, sign opposite and 3-4 points). It lives entirely in layups (unlocated 81.1% vs located 54.0%; dunks are equal, 91.1% vs 87.8%), is NOT a few games (2,860 games, top 10 hold 8.2%; excluding them 83.3%), not a few teams (468), not a month, a tier or a half, and not shot mix (located rate re-weighted to the unlocated play-type mix is 56.6%). In 2025 about 3.0% of rim makes lack a coordinate against 1.0% of rim misses. (iv) Canned pixels. Using the instructed rule (> 0.5% of located attempts) there are 15 / 14 / 16 / 19 canned pixels holding 10.8 / 10.6 / 11.8 / 13.6% of located attempts (28-35% of located rim attempts, 1.3-2.2% of jump2, 0.03% of threes); 0.5% is 20 times the uniform pixel share (0.025%) and sits just under the 99.9th percentile of pixel shares (0.60-0.70%). Only ONE pixel per season is an isolated spike (8-9 times its eight neighbours; 5.64 ft from the baseline, 25.0 ft from the sideline, 0.39 ft from the basket centre): 2.7 / 2.9 / 2.9 / 3.0% of located attempts and about 7.6-8.0% of located rim attempts; the other canned pixels have neighbour ratios of mostly 0.6-1.8 and are the dense layup region. Canned rim attempts make 66.7-68.1% vs 53.4-54.5% elsewhere (+13 pooled, +14 to +15 within game), but matched on distance from the basket (0.5 ft bins to 4 ft) the gap falls to +1.4 to +1.9 points (SE 0.003), and the spike pixel makes LESS than its own eight neighbours (-2.3 to -3.2 points within game, z -3.6 to -5.9). The spike pixel (about 70% vs 57% elsewhere in the PM's numbers) is therefore mostly geometry. (v) Response to team skill: for rim shots on the 2022-25 as-of offence-PAE rating, make rate rises by 5.5 / 6.4 / 6.9 / 6.7 points from quintile 1 to 5 (slope 0.70 / 0.59 / 0.63 / 0.52 per rating point); canned (> 0.5%) pixels respond like the rest in 2023-25 (slope difference -0.07 / +0.02 / -0.02, all |z| < 1; 2022 -0.29, z -2.8), but the isolated spike pixel responds at about 0.2-0.36 against 0.63-0.96 for its own ring (spike minus ring z = -2.0 / -2.2 / -4.3 / -4.8): a damped response, consistent with a mixture of true at-the-rim shots and uninformative default placements, though a physical explanation cannot be excluded. Jump2 canned pixels (1.5-3k attempts) have slopes of -1.2 / -0.2 / -0.7 / +0.3, below the non-canned 0.24-0.35 in 2022 and 2024. (vi) Regime change: nothing in the play-text accuracy moves between 2023-24 and 2024-25 (text-three inside the arc 1.7-2.0% in every month of both seasons; layups > 6 ft 9.8-10.7% vs 8.3-10.5%, drifting down through 2025), and no break is visible inside 2024-25. What does step is the season boundary: CBBD located share (Nov 85.0% -> Nov 98.7%), canned share (11.9% -> 13.3% in November), and the distance histogram (total-variation distance 2024 vs 2025 = 0.022 against a split-half noise floor of 0.005-0.007 and 0.009-0.011 for the two earlier transitions). The hoopR/CBBD agreement break on 2025-02-10 is a hoopR scrape-vintage artefact (hoopR located share 3.9% -> 87.6% -> 99.99% in consecutive weeks) and not a charting change. We cannot tell from these tables whether ESPN changed charting at the start of 2024-25 or CBBD changed what it ingests; both are consistent with a step at the season boundary, and the higher 2025 coverage alone changes the composition of the located set.

### 3a. Pixel-frequency distribution (threshold justification)

| season | located | distinct_pixels | uniform_pixel_share_pct | pixel_share_p50_pct | pixel_share_p99_pct | pixel_share_p999_pct | max_pixel_share_pct | pixels_over_0.5pct | attempts_on_canned_pct | median_neighbour_spike_ratio_canned | max_neighbour_ratio_noncanned(n>=20) | p99_neighbour_ratio_noncanned(n>=20) | pixels_canned_with_ratio_ge3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 537854 | 3867 | 0.026 | 0.008 | 0.320 | 0.600 | 2.684 | 15 | 10.798 | 1.218 | 2.038 | 1.654 | 1 |
| 2023 | 591564 | 3894 | 0.026 | 0.007 | 0.336 | 0.611 | 2.874 | 14 | 10.592 | 1.263 | 1.976 | 1.605 | 1 |
| 2024 | 633884 | 3948 | 0.025 | 0.007 | 0.340 | 0.636 | 2.933 | 16 | 11.766 | 1.261 | 2.078 | 1.607 | 1 |
| 2025 | 706606 | 4032 | 0.025 | 0.006 | 0.327 | 0.638 | 2.978 | 19 | 13.613 | 1.216 | 2.053 | 1.663 | 1 |
| 2026 | 646718 | 3957 | 0.025 | 0.006 | 0.347 | 0.695 | 3.118 | 19 | 13.999 | 1.220 | 2.353 | 1.693 | 1 |

Sensitivity of the canned set to the threshold

| season | threshold_pct | pixels | attempts_on_them_pct |
|---|---|---|---|
| 2022 | 0.250 | 63 | 27.012 |
| 2022 | 0.500 | 15 | 10.798 |
| 2022 | 1 | 1 | 2.684 |
| 2023 | 0.250 | 61 | 27.094 |
| 2023 | 0.500 | 14 | 10.592 |
| 2023 | 1 | 1 | 2.874 |
| 2024 | 0.250 | 64 | 28.327 |
| 2024 | 0.500 | 16 | 11.766 |
| 2024 | 1 | 1 | 2.933 |
| 2025 | 0.250 | 67 | 29.537 |
| 2025 | 0.500 | 19 | 13.613 |
| 2025 | 1 | 1 | 2.978 |
| 2026 | 0.250 | 62 | 29.090 |
| 2026 | 0.500 | 19 | 13.999 |
| 2026 | 1 | 1 | 3.118 |

### 3a. Top 20 pixels per season (folded frame, feet from baseline / from sideline)

**Season 2022**

| neighbour_ratio | rank | x_ft_folded | y_ft | dist_from_basket_ft | n | share_of_located_pct | pct_rim | pct_jump2 | pct_three | pct_dunk_playtype | make_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8.302 | 1 | 5.640 | 25 | 0.390 | 14437 | 2.684 | 99.792 | 0 | 0.208 | 14.539 | 0.699 |
| 1.559 | 2 | 6.580 | 23 | 2.402 | 3692 | 0.686 | 92.335 | 7.638 | 0.027 | 7.828 | 0.668 |
| 1.491 | 3 | 6.580 | 26.500 | 2.005 | 3583 | 0.666 | 99.888 | 0 | 0.112 | 9.350 | 0.658 |
| 1.426 | 4 | 6.580 | 27 | 2.402 | 3495 | 0.650 | 91.102 | 8.870 | 0.029 | 5.722 | 0.624 |
| 1.208 | 5 | 7.520 | 23 | 3.025 | 3185 | 0.592 | 91.115 | 8.823 | 0.063 | 7.347 | 0.631 |
| 1.409 | 6 | 6.580 | 22.500 | 2.832 | 3121 | 0.580 | 91.733 | 8.267 | 0 | 5.799 | 0.631 |
| 1.528 | 7 | 6.580 | 27.500 | 2.832 | 3109 | 0.578 | 89.900 | 10.068 | 0.032 | 4.342 | 0.571 |
| 0.796 | 8 | 6.580 | 25.500 | 1.421 | 3063 | 0.569 | 99.967 | 0 | 0.033 | 21.548 | 0.721 |
| 0.782 | 9 | 6.580 | 24.500 | 1.421 | 3058 | 0.569 | 100 | 0 | 0 | 28.483 | 0.756 |
| 1.333 | 10 | 6.580 | 24 | 1.664 | 3042 | 0.566 | 99.836 | 0 | 0.164 | 18.376 | 0.720 |
| 0.778 | 11 | 6.580 | 25 | 1.330 | 3030 | 0.563 | 100 | 0 | 0 | 28.845 | 0.750 |
| 1.218 | 12 | 6.580 | 26 | 1.664 | 2928 | 0.544 | 99.932 | 0 | 0.068 | 13.764 | 0.668 |
| 1.160 | 13 | 6.580 | 23.500 | 2.005 | 2874 | 0.534 | 100 | 0 | 0 | 12.422 | 0.685 |
| 1.102 | 14 | 7.520 | 24 | 2.481 | 2746 | 0.511 | 91.005 | 8.886 | 0.109 | 14.858 | 0.676 |
| 1.087 | 15 | 7.520 | 22.500 | 3.377 | 2716 | 0.505 | 89.654 | 10.272 | 0.074 | 5.228 | 0.595 |
| 1.186 | 16 | 7.520 | 27.500 | 3.377 | 2661 | 0.495 | 87.411 | 12.514 | 0.075 | 3.796 | 0.542 |
| 1.061 | 17 | 7.520 | 26.500 | 2.721 | 2646 | 0.492 | 90.703 | 9.221 | 0.076 | 8.541 | 0.597 |
| 0.986 | 18 | 7.520 | 27 | 3.025 | 2585 | 0.481 | 90.135 | 9.826 | 0.039 | 5.609 | 0.593 |
| 0.932 | 19 | 7.520 | 23.500 | 2.721 | 2519 | 0.468 | 91.346 | 8.615 | 0.040 | 10.639 | 0.660 |
| 1.001 | 20 | 7.520 | 24.500 | 2.324 | 2518 | 0.468 | 90.429 | 9.571 | 0 | 17.554 | 0.686 |

**Season 2023**

| neighbour_ratio | rank | x_ft_folded | y_ft | dist_from_basket_ft | n | share_of_located_pct | pct_rim | pct_jump2 | pct_three | pct_dunk_playtype | make_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8.249 | 1 | 5.640 | 25 | 0.390 | 17004 | 2.874 | 99.829 | 0 | 0.171 | 13.850 | 0.698 |
| 1.540 | 2 | 6.580 | 23 | 2.402 | 4064 | 0.687 | 91.929 | 8.071 | 0 | 8.120 | 0.656 |
| 1.492 | 3 | 6.580 | 26.500 | 2.005 | 3948 | 0.667 | 99.924 | 0 | 0.076 | 8.435 | 0.649 |
| 1.401 | 4 | 6.580 | 27 | 2.402 | 3821 | 0.646 | 91.154 | 8.846 | 0 | 5.444 | 0.623 |
| 1.568 | 5 | 6.580 | 27.500 | 2.832 | 3589 | 0.607 | 90.164 | 9.780 | 0.056 | 4.514 | 0.593 |
| 1.338 | 6 | 6.580 | 24 | 1.664 | 3505 | 0.592 | 99.943 | 0 | 0.057 | 18.716 | 0.715 |
| 0.770 | 7 | 6.580 | 25 | 1.330 | 3502 | 0.592 | 99.943 | 0 | 0.057 | 26.499 | 0.735 |
| 0.787 | 8 | 6.580 | 25.500 | 1.421 | 3491 | 0.590 | 99.943 | 0 | 0.057 | 22.801 | 0.710 |
| 1.404 | 9 | 6.580 | 22.500 | 2.832 | 3449 | 0.583 | 91.911 | 8.031 | 0.058 | 5.857 | 0.631 |
| 0.750 | 10 | 6.580 | 24.500 | 1.421 | 3421 | 0.578 | 99.912 | 0 | 0.088 | 25.197 | 0.728 |
| 1.162 | 11 | 7.520 | 23 | 3.025 | 3380 | 0.571 | 91.213 | 8.728 | 0.059 | 7.249 | 0.630 |
| 1.188 | 12 | 6.580 | 23.500 | 2.005 | 3314 | 0.560 | 99.909 | 0 | 0.091 | 12.674 | 0.702 |
| 1.186 | 13 | 6.580 | 26 | 1.664 | 3200 | 0.541 | 99.938 | 0 | 0.062 | 11.938 | 0.685 |
| 1.074 | 14 | 7.520 | 22.500 | 3.377 | 2969 | 0.502 | 89.525 | 10.441 | 0.034 | 5.490 | 0.600 |
| 1.060 | 15 | 7.520 | 24 | 2.481 | 2937 | 0.496 | 91.488 | 8.444 | 0.068 | 15.322 | 0.673 |
| 1.004 | 16 | 7.520 | 27 | 3.025 | 2866 | 0.484 | 90.614 | 9.386 | 0 | 6.490 | 0.579 |
| 1.054 | 17 | 7.520 | 26.500 | 2.721 | 2856 | 0.483 | 89.881 | 10.084 | 0.035 | 7.913 | 0.608 |
| 1.141 | 18 | 7.520 | 27.500 | 3.377 | 2855 | 0.483 | 88.511 | 11.419 | 0.070 | 4.063 | 0.542 |
| 0.990 | 19 | 7.520 | 24.500 | 2.324 | 2792 | 0.472 | 90.723 | 9.241 | 0.036 | 17.371 | 0.679 |
| 0.914 | 20 | 7.520 | 23.500 | 2.721 | 2736 | 0.463 | 89.912 | 9.978 | 0.110 | 10.892 | 0.639 |

**Season 2024**

| neighbour_ratio | rank | x_ft_folded | y_ft | dist_from_basket_ft | n | share_of_located_pct | pct_rim | pct_jump2 | pct_three | pct_dunk_playtype | make_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8.519 | 1 | 5.640 | 25 | 0.390 | 18589 | 2.933 | 99.849 | 0 | 0.151 | 13.368 | 0.694 |
| 1.673 | 2 | 6.580 | 27 | 2.402 | 4537 | 0.716 | 90.611 | 9.345 | 0.044 | 6.260 | 0.614 |
| 1.592 | 3 | 6.580 | 26 | 1.664 | 4217 | 0.665 | 99.953 | 0 | 0.047 | 12.734 | 0.668 |
| 0.848 | 4 | 6.580 | 25 | 1.330 | 4079 | 0.643 | 100 | 0 | 0 | 27.458 | 0.743 |
| 1.358 | 5 | 6.580 | 23 | 2.402 | 4029 | 0.636 | 90.618 | 9.357 | 0.025 | 8.141 | 0.654 |
| 1.583 | 6 | 6.580 | 22.500 | 2.832 | 3994 | 0.630 | 89.960 | 9.990 | 0.050 | 5.358 | 0.619 |
| 1.268 | 7 | 6.580 | 23.500 | 2.005 | 3719 | 0.587 | 99.839 | 0 | 0.161 | 11.992 | 0.677 |
| 1.285 | 8 | 6.580 | 24 | 1.664 | 3681 | 0.581 | 99.919 | 0 | 0.081 | 18.908 | 0.717 |
| 0.737 | 9 | 6.580 | 24.500 | 1.421 | 3666 | 0.578 | 99.918 | 0 | 0.082 | 24.905 | 0.714 |
| 1.382 | 10 | 6.580 | 27.500 | 2.832 | 3599 | 0.568 | 88.691 | 11.253 | 0.056 | 3.862 | 0.571 |
| 1.255 | 11 | 7.520 | 22.500 | 3.377 | 3587 | 0.566 | 88.932 | 11.068 | 0 | 5.018 | 0.587 |
| 1.152 | 12 | 6.580 | 26.500 | 2.005 | 3491 | 0.551 | 99.943 | 0 | 0.057 | 9.138 | 0.649 |
| 1.048 | 13 | 7.520 | 23 | 3.025 | 3445 | 0.543 | 89.405 | 10.595 | 0 | 6.676 | 0.621 |
| 1.179 | 14 | 7.520 | 27 | 3.025 | 3441 | 0.543 | 88.928 | 10.898 | 0.174 | 6.190 | 0.581 |
| 1.031 | 15 | 7.520 | 23.500 | 2.721 | 3257 | 0.514 | 89.714 | 10.286 | 0 | 9.303 | 0.638 |
| 0.649 | 16 | 6.580 | 25.500 | 1.421 | 3254 | 0.513 | 99.939 | 0 | 0.061 | 20.867 | 0.695 |
| 1.145 | 17 | 7.520 | 26 | 2.481 | 3149 | 0.497 | 90.473 | 9.432 | 0.095 | 11.686 | 0.626 |
| 1.020 | 18 | 7.520 | 24 | 2.481 | 3090 | 0.487 | 89.676 | 10.291 | 0.032 | 13.430 | 0.654 |
| 1.053 | 19 | 7.520 | 25 | 2.270 | 3074 | 0.485 | 89.395 | 10.507 | 0.098 | 17.664 | 0.654 |
| 1.064 | 20 | 7.520 | 27.500 | 3.377 | 3035 | 0.479 | 87.348 | 12.619 | 0.033 | 4.613 | 0.540 |

**Season 2025**

| neighbour_ratio | rank | x_ft_folded | y_ft | dist_from_basket_ft | n | share_of_located_pct | pct_rim | pct_jump2 | pct_three | pct_dunk_playtype | make_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8.005 | 1 | 5.640 | 25 | 0.390 | 21041 | 2.978 | 99.810 | 0 | 0.190 | 6.064 | 0.691 |
| 1.806 | 2 | 6.580 | 27 | 2.402 | 5363 | 0.759 | 92.057 | 7.943 | 0 | 6.321 | 0.611 |
| 1.712 | 3 | 6.580 | 26 | 1.664 | 5131 | 0.726 | 99.922 | 0 | 0.078 | 13.876 | 0.662 |
| 0.891 | 4 | 6.580 | 25 | 1.330 | 5003 | 0.708 | 99.920 | 0 | 0.080 | 27.963 | 0.741 |
| 1.650 | 5 | 6.580 | 22.500 | 2.832 | 4510 | 0.638 | 92.328 | 7.650 | 0.022 | 6.053 | 0.651 |
| 0.775 | 6 | 6.580 | 24.500 | 1.421 | 4477 | 0.634 | 99.955 | 0 | 0.045 | 25.463 | 0.730 |
| 1.339 | 7 | 6.580 | 23.500 | 2.005 | 4406 | 0.624 | 99.909 | 0 | 0.091 | 13.005 | 0.685 |
| 1.304 | 8 | 6.580 | 23 | 2.402 | 4372 | 0.619 | 92.978 | 6.976 | 0.046 | 8.120 | 0.646 |
| 1.265 | 9 | 6.580 | 24 | 1.664 | 4258 | 0.603 | 99.977 | 0 | 0.023 | 19.117 | 0.705 |
| 0.672 | 10 | 6.580 | 25.500 | 1.421 | 3932 | 0.556 | 99.924 | 0 | 0.076 | 21.109 | 0.706 |
| 1.309 | 11 | 6.580 | 27.500 | 2.832 | 3880 | 0.549 | 90.670 | 9.227 | 0.103 | 3.969 | 0.566 |
| 1.243 | 12 | 7.520 | 22.500 | 3.377 | 3840 | 0.543 | 90.312 | 9.635 | 0.052 | 5.599 | 0.597 |
| 1.216 | 13 | 7.520 | 27 | 3.025 | 3839 | 0.543 | 89.711 | 10.237 | 0.052 | 5.913 | 0.584 |
| 1.108 | 14 | 7.520 | 25 | 2.270 | 3826 | 0.541 | 92.342 | 7.554 | 0.105 | 17.825 | 0.666 |
| 1.050 | 15 | 7.520 | 23 | 3.025 | 3812 | 0.539 | 91.632 | 8.342 | 0.026 | 7.975 | 0.623 |
| 1.188 | 16 | 7.520 | 26 | 2.481 | 3666 | 0.519 | 91.408 | 8.565 | 0.027 | 10.993 | 0.612 |
| 1.030 | 17 | 7.520 | 23.500 | 2.721 | 3627 | 0.513 | 91.784 | 8.133 | 0.083 | 10.394 | 0.636 |
| 0.991 | 18 | 7.520 | 24.500 | 2.324 | 3610 | 0.511 | 90.693 | 9.252 | 0.055 | 16.094 | 0.672 |
| 1.018 | 19 | 6.580 | 26.500 | 2.005 | 3600 | 0.509 | 99.944 | 0 | 0.056 | 8.556 | 0.639 |
| 0.988 | 20 | 7.520 | 24 | 2.481 | 3484 | 0.493 | 91.734 | 8.180 | 0.086 | 13.921 | 0.654 |

**Season 2026**

| neighbour_ratio | rank | x_ft_folded | y_ft | dist_from_basket_ft | n | share_of_located_pct | pct_rim | pct_jump2 | pct_three | pct_dunk_playtype |
|---|---|---|---|---|---|---|---|---|---|---|
| 9.004 | 1 | 5.640 | 25 | 0.390 | 20165 | 3.118 | 99.980 | 0 | 0.020 | 7.325 |
| 1.590 | 2 | 6.580 | 27 | 2.402 | 4787 | 0.740 | 96.282 | 3.718 | 0 | 6.998 |
| 1.461 | 3 | 6.580 | 23 | 2.402 | 4622 | 0.715 | 96.928 | 3.072 | 0 | 9.368 |
| 1.551 | 4 | 6.580 | 26 | 1.664 | 4560 | 0.705 | 99.978 | 0 | 0.022 | 16.316 |
| 0.888 | 5 | 6.580 | 25 | 1.330 | 4495 | 0.695 | 100 | 0 | 0 | 29.366 |
| 1.483 | 6 | 6.580 | 24 | 1.664 | 4233 | 0.655 | 99.976 | 0 | 0.024 | 20.387 |
| 1.238 | 7 | 7.520 | 23 | 3.025 | 4219 | 0.652 | 95.710 | 4.266 | 0.024 | 9.220 |
| 1.404 | 8 | 6.580 | 22.500 | 2.832 | 4086 | 0.632 | 96.476 | 3.524 | 0 | 5.384 |
| 1.204 | 9 | 7.520 | 22.500 | 3.377 | 3867 | 0.598 | 94.130 | 5.844 | 0.026 | 6.103 |
| 1.365 | 10 | 6.580 | 27.500 | 2.832 | 3745 | 0.579 | 95.434 | 4.566 | 0 | 5.154 |
| 1.178 | 11 | 7.520 | 27 | 3.025 | 3679 | 0.569 | 94.401 | 5.572 | 0.027 | 6.524 |
| 1.250 | 12 | 7.520 | 25 | 2.270 | 3652 | 0.565 | 95.318 | 4.682 | 0 | 18.757 |
| 1.076 | 13 | 6.580 | 26.500 | 2.005 | 3645 | 0.564 | 100 | 0 | 0 | 10.782 |
| 1.220 | 14 | 7.520 | 24 | 2.481 | 3627 | 0.561 | 95.203 | 4.770 | 0.028 | 14.640 |
| 1.053 | 15 | 6.580 | 23.500 | 2.005 | 3514 | 0.543 | 100 | 0 | 0 | 13.717 |
| 1.178 | 16 | 7.520 | 26 | 2.481 | 3493 | 0.540 | 95.792 | 4.208 | 0 | 12.940 |
| 0.626 | 17 | 6.580 | 25.500 | 1.421 | 3452 | 0.534 | 100 | 0 | 0 | 24.218 |
| 1.161 | 18 | 7.520 | 27.500 | 3.377 | 3347 | 0.518 | 93.726 | 6.274 | 0 | 4.541 |
| 0.616 | 19 | 6.580 | 24.500 | 1.421 | 3347 | 0.518 | 100 | 0 | 0 | 28.354 |
| 0.871 | 20 | 7.520 | 23.500 | 2.721 | 3061 | 0.473 | 95.165 | 4.770 | 0.065 | 10.552 |

### 3a. Make rate on canned vs non-canned pixels (a = canned, b = non-canned). within_game_diff is the paired, within-game estimate; boot_se is the game bootstrap floor

| season | class | a | b | n_a | n_b | rate_a | rate_b | pooled_diff | games_paired | within_game_diff | boot_se | z | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | rim | canned | non-canned | 56060 | 134636 | 0.681 | 0.544 | 0.136 | 5227 | 0.148 | 0.003 | 55.757 |  |
| 2022 | jump2 | canned | non-canned | 1967 | 138486 | 0.546 | 0.373 | 0.173 | 1207 | 0.157 | 0.011 | 14.012 |  |
| 2022 | three | canned | non-canned | 52 | 206653 | 0.038 | 0.337 | -0.298 | 50 | -0.294 | 0.028 | -10.416 | UNDERPOWERED |
| 2023 | rim | canned | non-canned | 60705 | 154365 | 0.680 | 0.545 | 0.134 | 5640 | 0.144 | 0.003 | 56.081 |  |
| 2023 | jump2 | canned | non-canned | 1899 | 149754 | 0.531 | 0.378 | 0.154 | 1159 | 0.144 | 0.013 | 11.089 |  |
| 2023 | three | canned | non-canned | 53 | 224788 | 0.057 | 0.340 | -0.284 | 53 | -0.291 | 0.032 | -9.213 | UNDERPOWERED |
| 2024 | rim | canned | non-canned | 71449 | 161628 | 0.667 | 0.538 | 0.129 | 5830 | 0.139 | 0.003 | 55.376 |  |
| 2024 | jump2 | canned | non-canned | 3077 | 159117 | 0.548 | 0.386 | 0.161 | 1659 | 0.148 | 0.010 | 14.532 |  |
| 2024 | three | canned | non-canned | 59 | 238554 | 0.051 | 0.338 | -0.287 | 59 | -0.291 | 0.030 | -9.622 | UNDERPOWERED |
| 2025 | rim | canned | non-canned | 92364 | 168879 | 0.667 | 0.534 | 0.133 | 6110 | 0.145 | 0.002 | 64.270 |  |
| 2025 | jump2 | canned | non-canned | 3747 | 163293 | 0.553 | 0.384 | 0.169 | 1918 | 0.160 | 0.008 | 18.789 |  |
| 2025 | three | canned | non-canned | 82 | 278241 | 0.049 | 0.338 | -0.289 | 79 | -0.276 | 0.029 | -9.550 | UNDERPOWERED |

Canned share of located attempts by class

| season | class | located | canned_share_pct |
|---|---|---|---|
| 2022 | rim | 190696 | 29.398 |
| 2022 | jump2 | 140453 | 1.400 |
| 2022 | three | 206705 | 0.025 |
| 2023 | rim | 215070 | 28.226 |
| 2023 | jump2 | 151653 | 1.252 |
| 2023 | three | 224841 | 0.024 |
| 2024 | rim | 233077 | 30.655 |
| 2024 | jump2 | 162194 | 1.897 |
| 2024 | three | 238613 | 0.025 |
| 2025 | rim | 261243 | 35.356 |
| 2025 | jump2 | 167040 | 2.243 |
| 2025 | three | 278323 | 0.029 |

Spike pixel (canned AND >= 3x its 8 neighbours) vs the ring of its 8 neighbours, same class, within game

| season | class | a | b | n_a | n_b | rate_a | rate_b | pooled_diff | games_paired | within_game_diff | boot_se | z | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | rim | spike pixel | its 8-neighbour ring | 14407 | 13908 | 0.701 | 0.745 | -0.044 | 3041 | -0.032 | 0.007 | -4.952 |  |
| 2022 | three | spike pixel | its 8-neighbour ring | 30 | 3 | 0 | 0 | 0 | 0 |  |  |  | UNDERPOWERED |
| 2023 | rim | spike pixel | its 8-neighbour ring | 16975 | 16482 | 0.699 | 0.731 | -0.031 | 3439 | -0.023 | 0.006 | -3.615 |  |
| 2023 | three | spike pixel | its 8-neighbour ring | 29 | 9 | 0 | 0.111 | -0.111 | 0 |  |  |  | UNDERPOWERED |
| 2024 | rim | spike pixel | its 8-neighbour ring | 18561 | 17448 | 0.695 | 0.722 | -0.027 | 3683 | -0.028 | 0.006 | -4.337 |  |
| 2024 | three | spike pixel | its 8-neighbour ring | 28 | 9 | 0 | 0 | 0 | 0 |  |  |  | UNDERPOWERED |
| 2025 | rim | spike pixel | its 8-neighbour ring | 21001 | 21011 | 0.693 | 0.728 | -0.035 | 4256 | -0.032 | 0.005 | -5.893 |  |
| 2025 | three | spike pixel | its 8-neighbour ring | 40 | 16 | 0 | 0.062 | -0.062 | 0 |  |  |  | UNDERPOWERED |

Distance-matched: rim attempts, canned vs non-canned within 0.5 ft distance-from-basket bins

| season | dist_bin_ft | n_canned | n_non_canned | make_canned | make_non_canned | diff | flag |
|---|---|---|---|---|---|---|---|
| 2022 | [1.0, 1.5) | 9150 | 4202 | 0.742 | 0.697 | 0.045 |  |
| 2022 | [1.5, 2.0) | 5963 | 4591 | 0.695 | 0.680 | 0.015 |  |
| 2022 | [2.0, 2.5) | 15545 | 13371 | 0.666 | 0.661 | 0.005 |  |
| 2022 | [2.5, 3.0) | 5658 | 8191 | 0.610 | 0.632 | -0.022 |  |
| 2022 | [3.0, 4.0) | 5337 | 32356 | 0.623 | 0.576 | 0.047 |  |
| 2022 | STANDARDISED (weights = canned n, bins >= 500 both) | 41653 |  |  |  | 0.017 | binomial SE 0.0034 |
| 2023 | [1.0, 1.5) | 10407 | 5075 | 0.724 | 0.703 | 0.021 |  |
| 2023 | [1.5, 2.0) | 6701 | 5536 | 0.701 | 0.679 | 0.022 |  |
| 2023 | [2.0, 2.5) | 14475 | 18014 | 0.661 | 0.664 | -0.003 |  |
| 2023 | [2.5, 3.0) | 6406 | 8756 | 0.622 | 0.626 | -0.004 |  |
| 2023 | [3.0, 4.0) | 5741 | 36464 | 0.626 | 0.576 | 0.050 |  |
| 2023 | STANDARDISED (weights = canned n, bins >= 500 both) | 43730 |  |  |  | 0.014 | binomial SE 0.0032 |
| 2024 | [1.0, 1.5) | 10994 | 5970 | 0.719 | 0.687 | 0.032 |  |
| 2024 | [1.5, 2.0) | 7893 | 5555 | 0.691 | 0.659 | 0.032 |  |
| 2024 | [2.0, 2.5) | 14964 | 19675 | 0.651 | 0.650 | 0.001 |  |
| 2024 | [2.5, 3.0) | 9707 | 6613 | 0.614 | 0.610 | 0.004 |  |
| 2024 | [3.0, 4.0) | 9330 | 36500 | 0.607 | 0.576 | 0.031 |  |
| 2024 | STANDARDISED (weights = canned n, bins >= 500 both) | 52888 |  |  |  | 0.018 | binomial SE 0.0030 |
| 2025 | [1.0, 1.5) | 13403 | 6872 | 0.728 | 0.685 | 0.042 |  |
| 2025 | [1.5, 2.0) | 9384 | 6253 | 0.681 | 0.662 | 0.019 |  |
| 2025 | [2.0, 2.5) | 27160 | 12799 | 0.653 | 0.652 | 0.001 |  |
| 2025 | [2.5, 3.0) | 11011 | 7331 | 0.625 | 0.602 | 0.023 |  |
| 2025 | [3.0, 4.0) | 10405 | 40659 | 0.605 | 0.575 | 0.030 |  |
| 2025 | STANDARDISED (weights = canned n, bins >= 500 both) | 71363 |  |  |  | 0.019 | binomial SE 0.0029 |

Rim canned-vs-non-canned within-game gap under other canned thresholds

| season | threshold_pct | class | n_a | n_b | rate_a | rate_b | pooled_diff | games_paired | within_game_diff | boot_se | z | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 0.250 | rim | 132024 | 58672 | 0.638 | 0.465 | 0.173 | 5243 | 0.200 | 0.003 | 62.865 |  |
| 2023 | 0.250 | rim | 146366 | 68704 | 0.640 | 0.462 | 0.178 | 5645 | 0.206 | 0.003 | 74.895 |  |
| 2024 | 0.250 | rim | 162168 | 70909 | 0.630 | 0.456 | 0.174 | 5803 | 0.198 | 0.003 | 70.144 |  |
| 2025 | 0.250 | rim | 187199 | 74044 | 0.630 | 0.457 | 0.173 | 6060 | 0.196 | 0.003 | 77.613 |  |
| 2022 | 0.500 | rim | 56060 | 134636 | 0.681 | 0.544 | 0.136 | 5227 | 0.148 | 0.003 | 48.891 |  |
| 2023 | 0.500 | rim | 60705 | 154365 | 0.680 | 0.545 | 0.134 | 5640 | 0.144 | 0.003 | 54.120 |  |
| 2024 | 0.500 | rim | 71449 | 161628 | 0.667 | 0.538 | 0.129 | 5830 | 0.139 | 0.003 | 55.208 |  |
| 2025 | 0.500 | rim | 92364 | 168879 | 0.667 | 0.534 | 0.133 | 6110 | 0.145 | 0.002 | 66.506 |  |
| 2022 | 1 | rim | 14407 | 176289 | 0.701 | 0.575 | 0.126 | 4167 | 0.137 | 0.005 | 29.148 |  |
| 2023 | 1 | rim | 16975 | 198095 | 0.699 | 0.573 | 0.126 | 4669 | 0.134 | 0.004 | 34.268 |  |
| 2024 | 1 | rim | 18561 | 214516 | 0.695 | 0.567 | 0.128 | 5001 | 0.136 | 0.004 | 35.823 |  |
| 2025 | 1 | rim | 21001 | 240242 | 0.693 | 0.572 | 0.121 | 5473 | 0.130 | 0.004 | 36.698 |  |

### 3b(i). Located share, MAKES vs MISSES (outcome-blind charting => equal within the floor)

| season | class(feed) | a | b | metric | n_a | n_b | rate_a | rate_b | pooled_diff | games_paired | within_game_diff | boot_se | z | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | rim | makes | misses | located share | 139177 | 96600 | 0.787 | 0.807 | -0.020 | 5827 | -0.013 | 0.001 | -12.275 |  |
| 2022 | jump2 | makes | misses | located share | 68120 | 112144 | 0.800 | 0.794 | 0.007 | 5830 | -0.003 | 0.001 | -3.441 |  |
| 2022 | three | makes | misses | located share | 86103 | 170133 | 0.808 | 0.806 | 0.002 | 5832 | 0.002 | 0.001 | 3.029 |  |
| 2023 | rim | makes | misses | located share | 148117 | 103444 | 0.831 | 0.854 | -0.022 | 6111 | -0.016 | 0.001 | -18.211 |  |
| 2023 | jump2 | makes | misses | located share | 71175 | 113975 | 0.840 | 0.837 | 0.003 | 6111 | -0.006 | 0.001 | -5.926 |  |
| 2023 | three | makes | misses | located share | 89916 | 174819 | 0.851 | 0.849 | 0.002 | 6113 | 0.001 | 0.001 | 1.424 |  |
| 2024 | rim | makes | misses | located share | 153066 | 109932 | 0.862 | 0.881 | -0.019 | 6149 | -0.017 | 0.001 | -19.213 |  |
| 2024 | jump2 | makes | misses | located share | 75598 | 114712 | 0.869 | 0.877 | -0.008 | 6149 | -0.011 | 0.001 | -12.569 |  |
| 2024 | three | makes | misses | located share | 91442 | 179313 | 0.882 | 0.881 | 0.001 | 6151 | -0.000 | 0.001 | -0.688 |  |
| 2025 | rim | makes | misses | located share | 148080 | 106248 | 0.970 | 0.990 | -0.021 | 6135 | -0.021 | 0.001 | -38.178 |  |
| 2025 | jump2 | makes | misses | located share | 74364 | 107304 | 0.983 | 0.992 | -0.009 | 6132 | -0.009 | 0.000 | -21.088 |  |
| 2025 | three | makes | misses | located share | 94834 | 185616 | 0.992 | 0.993 | -0.001 | 6135 | -0.001 | 0.000 | -2.635 |  |

Same, restricted to games with partial coverage (the only games that can show the effect)

| season | class(feed) | a | b | metric | n_a | n_b | rate_a | rate_b | pooled_diff | games_paired | within_game_diff | boot_se | z | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | rim | makes | misses | located share (partial games only) | 88597 | 60493 | 0.753 | 0.782 | -0.029 | 3637 | -0.021 | 0.002 | -12.496 |  |
| 2022 | jump2 | makes | misses | located share (partial games only) | 42894 | 71097 | 0.756 | 0.750 | 0.006 | 3639 | -0.005 | 0.002 | -3.219 |  |
| 2022 | three | makes | misses | located share (partial games only) | 53133 | 105332 | 0.777 | 0.772 | 0.005 | 3639 | 0.004 | 0.001 | 3.002 |  |
| 2023 | rim | makes | misses | located share (partial games only) | 96098 | 66537 | 0.801 | 0.832 | -0.031 | 3896 | -0.025 | 0.001 | -19.790 |  |
| 2023 | jump2 | makes | misses | located share (partial games only) | 45946 | 73258 | 0.805 | 0.804 | 0.001 | 3896 | -0.009 | 0.001 | -6.341 |  |
| 2023 | three | makes | misses | located share (partial games only) | 56951 | 111007 | 0.826 | 0.822 | 0.003 | 3897 | 0.001 | 0.001 | 1.369 |  |
| 2024 | rim | makes | misses | located share (partial games only) | 103419 | 74244 | 0.838 | 0.864 | -0.026 | 4127 | -0.025 | 0.001 | -18.183 |  |
| 2024 | jump2 | makes | misses | located share (partial games only) | 51722 | 77294 | 0.847 | 0.857 | -0.010 | 4127 | -0.016 | 0.001 | -12.234 |  |
| 2024 | three | makes | misses | located share (partial games only) | 61008 | 119827 | 0.862 | 0.862 | -0.001 | 4129 | -0.001 | 0.001 | -0.639 |  |
| 2025 | rim | makes | misses | located share (partial games only) | 84895 | 60775 | 0.950 | 0.986 | -0.036 | 3445 | -0.036 | 0.001 | -44.149 |  |
| 2025 | jump2 | makes | misses | located share (partial games only) | 41858 | 59960 | 0.972 | 0.987 | -0.016 | 3444 | -0.016 | 0.001 | -20.331 |  |
| 2025 | three | makes | misses | located share (partial games only) | 53041 | 104396 | 0.989 | 0.990 | -0.001 | 3445 | -0.001 | 0.000 | -2.728 |  |

### 3b(ii). Make rate, LOCATED (a) vs UNLOCATED (b)

| season | class | class_def | a | b | n_a | n_b | rate_a | rate_b | pooled_diff | games_paired | within_game_diff | boot_se | z | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | rim | build(rim override) | located | unlocated | 190696 | 48179 | 0.584 | 0.614 | -0.030 | 3168 | -0.073 | 0.006 | -11.399 |  |
| 2022 | jump2 | build(rim override) | located | unlocated | 140453 | 36713 | 0.375 | 0.370 | 0.005 | 1690 | -0.029 | 0.007 | -4.016 |  |
| 2022 | three | build(rim override) | located | unlocated | 206705 | 49531 | 0.337 | 0.333 | 0.004 | 1245 | 0.018 | 0.005 | 3.334 |  |
| 2022 | rim | feed | located | unlocated | 187598 | 48179 | 0.584 | 0.614 | -0.030 | 3167 | -0.072 | 0.006 | -12.336 |  |
| 2022 | jump2 | feed | located | unlocated | 143551 | 36713 | 0.380 | 0.370 | 0.010 | 1695 | -0.023 | 0.007 | -3.309 |  |
| 2022 | three | feed | located | unlocated | 206705 | 49531 | 0.337 | 0.333 | 0.004 | 1245 | 0.018 | 0.006 | 2.961 |  |
| 2023 | rim | build(rim override) | located | unlocated | 215070 | 40082 | 0.583 | 0.623 | -0.040 | 3386 | -0.100 | 0.006 | -16.038 |  |
| 2023 | jump2 | build(rim override) | located | unlocated | 151653 | 29906 | 0.379 | 0.380 | -0.000 | 1640 | -0.052 | 0.008 | -6.666 |  |
| 2023 | three | build(rim override) | located | unlocated | 224841 | 39894 | 0.340 | 0.337 | 0.004 | 1160 | 0.009 | 0.006 | 1.365 |  |
| 2023 | rim | feed | located | unlocated | 211479 | 40082 | 0.582 | 0.623 | -0.040 | 3385 | -0.100 | 0.006 | -15.583 |  |
| 2023 | jump2 | feed | located | unlocated | 155244 | 29906 | 0.385 | 0.380 | 0.006 | 1642 | -0.047 | 0.007 | -6.478 |  |
| 2023 | three | feed | located | unlocated | 224841 | 39894 | 0.340 | 0.337 | 0.004 | 1160 | 0.009 | 0.006 | 1.445 |  |
| 2024 | rim | build(rim override) | located | unlocated | 233077 | 34115 | 0.577 | 0.617 | -0.040 | 3470 | -0.107 | 0.006 | -17.183 |  |
| 2024 | jump2 | build(rim override) | located | unlocated | 162194 | 23922 | 0.390 | 0.413 | -0.023 | 1864 | -0.099 | 0.009 | -11.607 |  |
| 2024 | three | build(rim override) | located | unlocated | 238613 | 32142 | 0.338 | 0.336 | 0.002 | 1126 | -0.004 | 0.007 | -0.647 |  |
| 2024 | rim | feed | located | unlocated | 228883 | 34115 | 0.577 | 0.617 | -0.041 | 3469 | -0.107 | 0.006 | -16.594 |  |
| 2024 | jump2 | feed | located | unlocated | 166388 | 23922 | 0.395 | 0.413 | -0.018 | 1866 | -0.092 | 0.008 | -11.518 |  |
| 2024 | three | feed | located | unlocated | 238613 | 32142 | 0.338 | 0.336 | 0.002 | 1126 | -0.004 | 0.007 | -0.628 |  |
| 2025 | rim | build(rim override) | located | unlocated | 261243 | 5510 | 0.581 | 0.815 | -0.233 | 2850 | -0.283 | 0.007 | -42.415 |  |
| 2025 | jump2 | build(rim override) | located | unlocated | 167040 | 2203 | 0.388 | 0.586 | -0.199 | 995 | -0.315 | 0.016 | -19.734 |  |
| 2025 | three | build(rim override) | located | unlocated | 278323 | 2127 | 0.338 | 0.353 | -0.015 | 379 | -0.048 | 0.019 | -2.554 |  |
| 2025 | rim | feed | located | unlocated | 248818 | 5510 | 0.577 | 0.815 | -0.238 | 2850 | -0.287 | 0.007 | -42.218 |  |
| 2025 | jump2 | feed | located | unlocated | 179465 | 2203 | 0.407 | 0.586 | -0.179 | 995 | -0.293 | 0.016 | -18.048 |  |
| 2025 | three | feed | located | unlocated | 278323 | 2127 | 0.338 | 0.353 | -0.015 | 379 | -0.048 | 0.018 | -2.631 |  |

### 3b(ii). Explaining the rim located-vs-unlocated make-rate gap: by_playtype

| season | playType | unlocated_n | located_n | unlocated_make | located_make |
|---|---|---|---|---|---|
| 2022 | DunkShot | 5114 | 18391 | 0.888 | 0.878 |
| 2022 | JumpShot | 0 | 3098 |  | 0.594 |
| 2022 | LayUpShot | 41653 | 169110 | 0.578 | 0.553 |
| 2022 | TipShot | 1412 | 97 | 0.678 | 0.196 |
| 2023 | DunkShot | 4063 | 20894 | 0.884 | 0.878 |
| 2023 | JumpShot | 0 | 3587 |  | 0.633 |
| 2023 | LayUpShot | 34903 | 190515 | 0.590 | 0.550 |
| 2023 | Not Available | 0 | 4 |  | 0.250 |
| 2023 | TipShot | 1116 | 70 | 0.711 | 0.329 |
| 2024 | DunkShot | 3304 | 21999 | 0.876 | 0.878 |
| 2024 | JumpShot | 0 | 4194 |  | 0.607 |
| 2024 | LayUpShot | 30103 | 206831 | 0.588 | 0.545 |
| 2024 | TipShot | 708 | 53 | 0.681 | 0.151 |
| 2025 | DunkShot | 347 | 24113 | 0.911 | 0.878 |
| 2025 | JumpShot | 0 | 12425 |  | 0.666 |
| 2025 | LayUpShot | 4931 | 215762 | 0.811 | 0.540 |
| 2025 | TipShot | 232 | 8943 | 0.750 | 0.666 |

### 3b(ii). Explaining the rim located-vs-unlocated make-rate gap: mix_adjusted

| season | unlocated_n | unlocated_make | located_make | located_make_at_unlocated_playtype_mix | located_close_jumper_share_pct |
|---|---|---|---|---|---|
| 2022 | 48179 | 0.614 | 0.584 | 0.577 | 1.625 |
| 2023 | 40082 | 0.623 | 0.583 | 0.577 | 1.670 |
| 2024 | 34115 | 0.617 | 0.577 | 0.569 | 1.799 |
| 2025 | 5510 | 0.815 | 0.581 | 0.566 | 4.756 |

### 3b(ii). Explaining the rim located-vs-unlocated make-rate gap: concentration

| season | unlocated_rim | games_with_unlocated_rim | games_in_season | top10_games_share_pct | unlocated_make_excl_top10_games | n_excl_top10 | teams_with_unlocated_rim | top10_teams_share_pct | mean_make_rate_of_games_with_unlocated_rim_all_attempts | make_rate_games_without_unlocated_rim |
|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 48179 | 3499 | 5832 | 1.405 | 0.615 | 47502 | 587 | 13.904 | 0.594 | 0.585 |
| 2023 | 40082 | 3630 | 6113 | 1.714 | 0.624 | 39395 | 587 | 16.781 | 0.592 | 0.586 |
| 2024 | 34115 | 3644 | 6151 | 2.172 | 0.619 | 33374 | 594 | 18.124 | 0.584 | 0.581 |
| 2025 | 5510 | 2860 | 6135 | 8.167 | 0.833 | 5060 | 468 | 10.399 | 0.588 | 0.585 |

### 3b(ii). Explaining the rim located-vs-unlocated make-rate gap: by_month

| season | month | unlocated_n | unlocated_make | located_n | located_make |
|---|---|---|---|---|---|
| 2022 | 1 | 8791 | 0.623 | 47196 | 0.582 |
| 2022 | 2 | 8027 | 0.629 | 50088 | 0.584 |
| 2022 | 3 | 9068 | 0.603 | 16225 | 0.585 |
| 2022 | 4 | 139 | 0.496 | 33 | 0.697 |
| 2022 | 11 | 12695 | 0.616 | 41440 | 0.585 |
| 2022 | 12 | 9459 | 0.601 | 35714 | 0.587 |
| 2023 | 1 | 6595 | 0.648 | 52614 | 0.578 |
| 2023 | 2 | 6749 | 0.646 | 50499 | 0.583 |
| 2023 | 3 | 7809 | 0.604 | 14729 | 0.588 |
| 2023 | 4 | 93 | 0.452 | 12 | 0.667 |
| 2023 | 11 | 11442 | 0.604 | 50761 | 0.586 |
| 2023 | 12 | 7394 | 0.632 | 46455 | 0.585 |
| 2024 | 1 | 4722 | 0.650 | 54555 | 0.572 |
| 2024 | 2 | 4833 | 0.651 | 53770 | 0.573 |
| 2024 | 3 | 8110 | 0.593 | 26246 | 0.576 |
| 2024 | 4 | 217 | 0.590 | 46 | 0.761 |
| 2024 | 11 | 9259 | 0.604 | 51228 | 0.586 |
| 2024 | 12 | 6974 | 0.619 | 47232 | 0.579 |
| 2025 | 1 | 1512 | 0.795 | 60094 | 0.573 |
| 2025 | 2 | 973 | 0.780 | 58269 | 0.581 |
| 2025 | 3 | 776 | 0.820 | 32321 | 0.584 |
| 2025 | 4 | 8 | 1 | 666 | 0.604 |
| 2025 | 11 | 1329 | 0.829 | 60980 | 0.583 |
| 2025 | 12 | 912 | 0.857 | 48913 | 0.587 |

### 3b(ii). Explaining the rim located-vs-unlocated make-rate gap: by_period(3=OT)

| season | period | unlocated_n | unlocated_make | located_n | located_make |
|---|---|---|---|---|---|
| 2022 | 1 | 22668 | 0.601 | 90725 | 0.573 |
| 2022 | 2 | 25169 | 0.625 | 98400 | 0.595 |
| 2022 | 3 | 342 | 0.611 | 1571 | 0.584 |
| 2023 | 1 | 18737 | 0.612 | 102302 | 0.572 |
| 2023 | 2 | 21071 | 0.633 | 110986 | 0.594 |
| 2023 | 3 | 274 | 0.584 | 1782 | 0.563 |
| 2024 | 1 | 16059 | 0.607 | 111469 | 0.564 |
| 2024 | 2 | 17835 | 0.627 | 119732 | 0.590 |
| 2024 | 3 | 221 | 0.615 | 1876 | 0.561 |
| 2025 | 1 | 2347 | 0.810 | 124432 | 0.566 |
| 2025 | 2 | 3133 | 0.819 | 134868 | 0.595 |
| 2025 | 3 | 30 | 0.800 | 1943 | 0.584 |

### 3b(ii). Explaining the rim located-vs-unlocated make-rate gap: by_tier

| season | tier | unlocated_n | unlocated_make | located_n | located_make |
|---|---|---|---|---|---|
| 2022 | power | 20709 | 0.619 | 40453 | 0.588 |
| 2022 | mid | 697 | 0.620 | 6896 | 0.603 |
| 2022 | low | 26773 | 0.610 | 143347 | 0.582 |
| 2023 | power | 16379 | 0.625 | 45998 | 0.587 |
| 2023 | mid | 767 | 0.626 | 7307 | 0.597 |
| 2023 | low | 22936 | 0.621 | 161765 | 0.582 |
| 2024 | power | 13343 | 0.612 | 56337 | 0.588 |
| 2024 | mid | 1049 | 0.628 | 7889 | 0.593 |
| 2024 | low | 19723 | 0.620 | 168851 | 0.573 |
| 2025 | power | 1793 | 0.769 | 65111 | 0.595 |
| 2025 | mid | 161 | 0.863 | 8462 | 0.598 |
| 2025 | low | 3556 | 0.836 | 187670 | 0.576 |

### 3c. Make rate by as-of OFFENCE-PAE quintile, canned vs non-canned (located attempts, rated teams)

| season | class | group | quintile | n | mean_rating | make_rate | flag |
|---|---|---|---|---|---|---|---|
| 2022 | rim | spike pixel | 1 | 2039 | -0.040 | 0.689 |  |
| 2022 | rim | spike pixel | 2 | 1945 | -0.014 | 0.694 |  |
| 2022 | rim | spike pixel | 3 | 1979 | -0.000 | 0.691 |  |
| 2022 | rim | spike pixel | 4 | 1921 | 0.015 | 0.708 |  |
| 2022 | rim | spike pixel | 5 | 1939 | 0.041 | 0.712 |  |
| 2022 | rim | ring of spike pixel | 1 | 1836 | -0.039 | 0.723 |  |
| 2022 | rim | ring of spike pixel | 2 | 1886 | -0.014 | 0.743 |  |
| 2022 | rim | ring of spike pixel | 3 | 1830 | 0.000 | 0.738 |  |
| 2022 | rim | ring of spike pixel | 4 | 2067 | 0.014 | 0.758 |  |
| 2022 | rim | ring of spike pixel | 5 | 1743 | 0.041 | 0.784 |  |
| 2022 | rim | canned (>0.5%) | 1 | 7601 | -0.039 | 0.663 |  |
| 2022 | rim | canned (>0.5%) | 2 | 7458 | -0.014 | 0.676 |  |
| 2022 | rim | canned (>0.5%) | 3 | 7462 | 0.000 | 0.669 |  |
| 2022 | rim | canned (>0.5%) | 4 | 7735 | 0.014 | 0.683 |  |
| 2022 | rim | canned (>0.5%) | 5 | 7540 | 0.042 | 0.707 |  |
| 2022 | rim | non-canned | 1 | 18350 | -0.039 | 0.510 |  |
| 2022 | rim | non-canned | 2 | 18496 | -0.014 | 0.530 |  |
| 2022 | rim | non-canned | 3 | 18484 | 0.000 | 0.537 |  |
| 2022 | rim | non-canned | 4 | 18209 | 0.014 | 0.557 |  |
| 2022 | rim | non-canned | 5 | 18401 | 0.042 | 0.571 |  |
| 2022 | rim | all located | 1 | 25951 | -0.039 | 0.555 |  |
| 2022 | rim | all located | 2 | 25954 | -0.014 | 0.572 |  |
| 2022 | rim | all located | 3 | 25946 | 0.000 | 0.575 |  |
| 2022 | rim | all located | 4 | 25944 | 0.014 | 0.595 |  |
| 2022 | rim | all located | 5 | 25941 | 0.042 | 0.610 |  |
| 2023 | rim | spike pixel | 1 | 2622 | -0.053 | 0.683 |  |
| 2023 | rim | spike pixel | 2 | 2380 | -0.020 | 0.678 |  |
| 2023 | rim | spike pixel | 3 | 2477 | -0.001 | 0.707 |  |
| 2023 | rim | spike pixel | 4 | 2308 | 0.018 | 0.708 |  |
| 2023 | rim | spike pixel | 5 | 2171 | 0.054 | 0.705 |  |
| 2023 | rim | ring of spike pixel | 1 | 2555 | -0.054 | 0.696 |  |
| 2023 | rim | ring of spike pixel | 2 | 2456 | -0.020 | 0.714 |  |
| 2023 | rim | ring of spike pixel | 3 | 2304 | -0.000 | 0.735 |  |
| 2023 | rim | ring of spike pixel | 4 | 2261 | 0.018 | 0.746 |  |
| 2023 | rim | ring of spike pixel | 5 | 2009 | 0.054 | 0.771 |  |
| 2023 | rim | canned (>0.5%) | 1 | 8894 | -0.054 | 0.648 |  |
| 2023 | rim | canned (>0.5%) | 2 | 8509 | -0.020 | 0.665 |  |
| 2023 | rim | canned (>0.5%) | 3 | 8659 | -0.001 | 0.679 |  |
| 2023 | rim | canned (>0.5%) | 4 | 8281 | 0.018 | 0.685 |  |
| 2023 | rim | canned (>0.5%) | 5 | 8102 | 0.055 | 0.709 |  |
| 2023 | rim | non-canned | 1 | 20953 | -0.053 | 0.507 |  |
| 2023 | rim | non-canned | 2 | 21329 | -0.020 | 0.529 |  |
| 2023 | rim | non-canned | 3 | 21181 | -0.001 | 0.544 |  |
| 2023 | rim | non-canned | 4 | 21548 | 0.018 | 0.556 |  |
| 2023 | rim | non-canned | 5 | 21737 | 0.055 | 0.577 |  |
| 2023 | rim | all located | 1 | 29847 | -0.053 | 0.549 |  |
| 2023 | rim | all located | 2 | 29838 | -0.020 | 0.568 |  |
| 2023 | rim | all located | 3 | 29840 | -0.001 | 0.583 |  |
| 2023 | rim | all located | 4 | 29829 | 0.018 | 0.592 |  |
| 2023 | rim | all located | 5 | 29839 | 0.055 | 0.613 |  |
| 2024 | rim | spike pixel | 1 | 3211 | -0.057 | 0.669 |  |
| 2024 | rim | spike pixel | 2 | 3160 | -0.022 | 0.695 |  |
| 2024 | rim | spike pixel | 3 | 2951 | -0.001 | 0.690 |  |
| 2024 | rim | spike pixel | 4 | 3071 | 0.020 | 0.700 |  |
| 2024 | rim | spike pixel | 5 | 2643 | 0.058 | 0.705 |  |
| 2024 | rim | ring of spike pixel | 1 | 3052 | -0.058 | 0.671 |  |
| 2024 | rim | ring of spike pixel | 2 | 2822 | -0.022 | 0.694 |  |
| 2024 | rim | ring of spike pixel | 3 | 2923 | -0.001 | 0.714 |  |
| 2024 | rim | ring of spike pixel | 4 | 2914 | 0.020 | 0.742 |  |
| 2024 | rim | ring of spike pixel | 5 | 2751 | 0.059 | 0.775 |  |
| 2024 | rim | canned (>0.5%) | 1 | 12080 | -0.056 | 0.625 |  |
| 2024 | rim | canned (>0.5%) | 2 | 11920 | -0.022 | 0.648 |  |
| 2024 | rim | canned (>0.5%) | 3 | 11540 | -0.001 | 0.668 |  |
| 2024 | rim | canned (>0.5%) | 4 | 11837 | 0.020 | 0.678 |  |
| 2024 | rim | canned (>0.5%) | 5 | 11390 | 0.059 | 0.696 |  |
| 2024 | rim | non-canned | 1 | 26329 | -0.055 | 0.506 |  |
| 2024 | rim | non-canned | 2 | 26498 | -0.022 | 0.520 |  |
| 2024 | rim | non-canned | 3 | 26867 | -0.001 | 0.530 |  |
| 2024 | rim | non-canned | 4 | 26563 | 0.020 | 0.551 |  |
| 2024 | rim | non-canned | 5 | 27001 | 0.060 | 0.577 |  |
| 2024 | rim | all located | 1 | 38409 | -0.055 | 0.543 |  |
| 2024 | rim | all located | 2 | 38418 | -0.022 | 0.560 |  |
| 2024 | rim | all located | 3 | 38407 | -0.001 | 0.572 |  |
| 2024 | rim | all located | 4 | 38400 | 0.020 | 0.590 |  |
| 2024 | rim | all located | 5 | 38391 | 0.060 | 0.612 |  |
| 2025 | rim | spike pixel | 1 | 3826 | -0.063 | 0.675 |  |
| 2025 | rim | spike pixel | 2 | 3513 | -0.024 | 0.695 |  |
| 2025 | rim | spike pixel | 3 | 3573 | -0.001 | 0.695 |  |
| 2025 | rim | spike pixel | 4 | 3452 | 0.023 | 0.691 |  |
| 2025 | rim | spike pixel | 5 | 3484 | 0.064 | 0.706 |  |
| 2025 | rim | ring of spike pixel | 1 | 4059 | -0.066 | 0.672 |  |
| 2025 | rim | ring of spike pixel | 2 | 3738 | -0.024 | 0.711 |  |
| 2025 | rim | ring of spike pixel | 3 | 3441 | -0.000 | 0.740 |  |
| 2025 | rim | ring of spike pixel | 4 | 3332 | 0.023 | 0.769 |  |
| 2025 | rim | ring of spike pixel | 5 | 3493 | 0.063 | 0.760 |  |
| 2025 | rim | canned (>0.5%) | 1 | 16298 | -0.063 | 0.627 |  |
| 2025 | rim | canned (>0.5%) | 2 | 15912 | -0.024 | 0.652 |  |
| 2025 | rim | canned (>0.5%) | 3 | 15762 | -0.000 | 0.671 |  |
| 2025 | rim | canned (>0.5%) | 4 | 15468 | 0.023 | 0.677 |  |
| 2025 | rim | canned (>0.5%) | 5 | 15824 | 0.064 | 0.699 |  |
| 2025 | rim | non-canned | 1 | 28296 | -0.063 | 0.500 |  |
| 2025 | rim | non-canned | 2 | 28692 | -0.024 | 0.521 |  |
| 2025 | rim | non-canned | 3 | 28832 | -0.001 | 0.526 |  |
| 2025 | rim | non-canned | 4 | 29111 | 0.023 | 0.548 |  |
| 2025 | rim | non-canned | 5 | 28766 | 0.064 | 0.567 |  |
| 2025 | rim | all located | 1 | 44594 | -0.063 | 0.547 |  |
| 2025 | rim | all located | 2 | 44604 | -0.024 | 0.567 |  |
| 2025 | rim | all located | 3 | 44594 | -0.001 | 0.577 |  |
| 2025 | rim | all located | 4 | 44579 | 0.023 | 0.593 |  |
| 2025 | rim | all located | 5 | 44590 | 0.064 | 0.614 |  |

Slopes (make probability per rating point; SE = game bootstrap)

| season | class | group | n | slope_make_per_rating_pt | boot_se | make_Q5_minus_Q1 | flag |
|---|---|---|---|---|---|---|---|
| 2022 | rim | spike pixel | 9823 | 0.213 | 0.175 | 0.024 |  |
| 2022 | rim | ring of spike pixel | 9362 | 0.684 | 0.163 | 0.060 |  |
| 2022 | rim | canned (>0.5%) | 37796 | 0.496 | 0.081 | 0.044 |  |
| 2022 | rim | non-canned | 91940 | 0.785 | 0.066 | 0.061 |  |
| 2022 | rim | all located | 129736 | 0.703 | 0.048 | 0.055 |  |
| 2022 | jump2 | spike pixel | 0 |  |  |  | UNDERPOWERED |
| 2022 | jump2 | ring of spike pixel | 0 |  |  |  | UNDERPOWERED |
| 2022 | jump2 | canned (>0.5%) | 1509 | -1.229 | 0.465 | -0.091 |  |
| 2022 | jump2 | non-canned | 97805 | 0.342 | 0.054 | 0.025 |  |
| 2022 | jump2 | all located | 99314 | 0.313 | 0.057 | 0.023 |  |
| 2022 | three | spike pixel | 16 |  |  | 0 | UNDERPOWERED |
| 2022 | three | ring of spike pixel | 2 |  |  |  | UNDERPOWERED |
| 2022 | three | canned (>0.5%) | 28 |  |  | 0 | UNDERPOWERED |
| 2022 | three | non-canned | 140954 | 0.309 | 0.045 | 0.026 |  |
| 2022 | three | all located | 140982 | 0.309 | 0.043 | 0.026 |  |
| 2023 | rim | spike pixel | 11958 | 0.281 | 0.115 | 0.022 |  |
| 2023 | rim | ring of spike pixel | 11585 | 0.633 | 0.113 | 0.074 |  |
| 2023 | rim | canned (>0.5%) | 42445 | 0.571 | 0.067 | 0.061 |  |
| 2023 | rim | non-canned | 106748 | 0.639 | 0.045 | 0.070 |  |
| 2023 | rim | all located | 149193 | 0.586 | 0.039 | 0.064 |  |
| 2023 | jump2 | spike pixel | 0 |  |  |  | UNDERPOWERED |
| 2023 | jump2 | ring of spike pixel | 0 |  |  |  | UNDERPOWERED |
| 2023 | jump2 | canned (>0.5%) | 1466 | -0.198 | 0.339 | -0.007 |  |
| 2023 | jump2 | non-canned | 108238 | 0.238 | 0.040 | 0.027 |  |
| 2023 | jump2 | all located | 109704 | 0.227 | 0.040 | 0.027 |  |
| 2023 | three | spike pixel | 23 |  |  | 0 | UNDERPOWERED |
| 2023 | three | ring of spike pixel | 5 |  |  |  | UNDERPOWERED |
| 2023 | three | canned (>0.5%) | 39 |  |  | 0 | UNDERPOWERED |
| 2023 | three | non-canned | 157216 | 0.166 | 0.033 | 0.018 |  |
| 2023 | three | all located | 157255 | 0.166 | 0.029 | 0.019 |  |
| 2024 | rim | spike pixel | 15036 | 0.355 | 0.098 | 0.036 |  |
| 2024 | rim | ring of spike pixel | 14462 | 0.956 | 0.101 | 0.105 |  |
| 2024 | rim | canned (>0.5%) | 58767 | 0.666 | 0.046 | 0.072 |  |
| 2024 | rim | non-canned | 133258 | 0.641 | 0.035 | 0.071 |  |
| 2024 | rim | all located | 192025 | 0.630 | 0.027 | 0.069 |  |
| 2024 | jump2 | spike pixel | 0 |  |  |  | UNDERPOWERED |
| 2024 | jump2 | ring of spike pixel | 0 |  |  |  | UNDERPOWERED |
| 2024 | jump2 | canned (>0.5%) | 2181 | -0.663 | 0.255 | -0.083 |  |
| 2024 | jump2 | non-canned | 129586 | 0.353 | 0.037 | 0.040 |  |
| 2024 | jump2 | all located | 131767 | 0.330 | 0.033 | 0.037 |  |
| 2024 | three | spike pixel | 20 |  |  | 0 | UNDERPOWERED |
| 2024 | three | ring of spike pixel | 6 |  |  |  | UNDERPOWERED |
| 2024 | three | canned (>0.5%) | 42 |  |  | 0 | UNDERPOWERED |
| 2024 | three | non-canned | 194773 | 0.258 | 0.024 | 0.029 |  |
| 2024 | three | all located | 194815 | 0.259 | 0.025 | 0.029 |  |
| 2025 | rim | spike pixel | 17848 | 0.195 | 0.078 | 0.031 |  |
| 2025 | rim | ring of spike pixel | 18063 | 0.727 | 0.080 | 0.088 |  |
| 2025 | rim | canned (>0.5%) | 79264 | 0.523 | 0.041 | 0.072 |  |
| 2025 | rim | non-canned | 143697 | 0.538 | 0.031 | 0.066 |  |
| 2025 | rim | all located | 222961 | 0.518 | 0.024 | 0.067 |  |
| 2025 | jump2 | spike pixel | 0 |  |  |  | UNDERPOWERED |
| 2025 | jump2 | ring of spike pixel | 0 |  |  |  | UNDERPOWERED |
| 2025 | jump2 | canned (>0.5%) | 3063 | 0.345 | 0.222 | 0.064 |  |
| 2025 | jump2 | non-canned | 139095 | 0.340 | 0.031 | 0.045 |  |
| 2025 | jump2 | all located | 142158 | 0.330 | 0.028 | 0.044 |  |
| 2025 | three | spike pixel | 29 |  |  | 0 | UNDERPOWERED |
| 2025 | three | ring of spike pixel | 12 |  |  | 0 | UNDERPOWERED |
| 2025 | three | canned (>0.5%) | 61 | 0.162 | 0.208 | 0 | UNDERPOWERED |
| 2025 | three | non-canned | 236453 | 0.265 | 0.021 | 0.036 |  |
| 2025 | three | all located | 236514 | 0.265 | 0.022 | 0.036 |  |

Group minus non-canned slope (SE adds the two bootstrap SEs in quadrature; the groups share games, so this is approximate)

| season | class | group | slope_group | slope_reference | diff | se_diff_(indep. approx.) | z | n_group | flag |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | jump2 | canned (>0.5%) | -1.229 | 0.342 | -1.571 | 0.468 | -3.358 | 1509 |  |
| 2022 | jump2 | spike pixel |  | 0.342 |  |  |  | 0 | UNDERPOWERED |
| 2022 | jump2 | ring of spike pixel |  | 0.342 |  |  |  | 0 | UNDERPOWERED |
| 2022 | rim | canned (>0.5%) | 0.496 | 0.785 | -0.289 | 0.105 | -2.764 | 37796 |  |
| 2022 | rim | spike pixel | 0.213 | 0.785 | -0.572 | 0.187 | -3.062 | 9823 |  |
| 2022 | rim | ring of spike pixel | 0.684 | 0.785 | -0.101 | 0.176 | -0.577 | 9362 |  |
| 2022 | three | canned (>0.5%) |  | 0.309 |  |  |  | 28 | UNDERPOWERED |
| 2022 | three | spike pixel |  | 0.309 |  |  |  | 16 | UNDERPOWERED |
| 2022 | three | ring of spike pixel |  | 0.309 |  |  |  | 2 | UNDERPOWERED |
| 2023 | jump2 | canned (>0.5%) | -0.198 | 0.238 | -0.436 | 0.341 | -1.278 | 1466 |  |
| 2023 | jump2 | spike pixel |  | 0.238 |  |  |  | 0 | UNDERPOWERED |
| 2023 | jump2 | ring of spike pixel |  | 0.238 |  |  |  | 0 | UNDERPOWERED |
| 2023 | rim | canned (>0.5%) | 0.571 | 0.639 | -0.069 | 0.081 | -0.846 | 42445 |  |
| 2023 | rim | spike pixel | 0.281 | 0.639 | -0.358 | 0.124 | -2.898 | 11958 |  |
| 2023 | rim | ring of spike pixel | 0.633 | 0.639 | -0.006 | 0.122 | -0.050 | 11585 |  |
| 2023 | three | canned (>0.5%) |  | 0.166 |  |  |  | 39 | UNDERPOWERED |
| 2023 | three | spike pixel |  | 0.166 |  |  |  | 23 | UNDERPOWERED |
| 2023 | three | ring of spike pixel |  | 0.166 |  |  |  | 5 | UNDERPOWERED |
| 2024 | jump2 | canned (>0.5%) | -0.663 | 0.353 | -1.017 | 0.257 | -3.948 | 2181 |  |
| 2024 | jump2 | spike pixel |  | 0.353 |  |  |  | 0 | UNDERPOWERED |
| 2024 | jump2 | ring of spike pixel |  | 0.353 |  |  |  | 0 | UNDERPOWERED |
| 2024 | rim | canned (>0.5%) | 0.666 | 0.641 | 0.024 | 0.058 | 0.414 | 58767 |  |
| 2024 | rim | spike pixel | 0.355 | 0.641 | -0.286 | 0.105 | -2.737 | 15036 |  |
| 2024 | rim | ring of spike pixel | 0.956 | 0.641 | 0.315 | 0.107 | 2.933 | 14462 |  |
| 2024 | three | canned (>0.5%) |  | 0.258 |  |  |  | 42 | UNDERPOWERED |
| 2024 | three | spike pixel |  | 0.258 |  |  |  | 20 | UNDERPOWERED |
| 2024 | three | ring of spike pixel |  | 0.258 |  |  |  | 6 | UNDERPOWERED |
| 2025 | jump2 | canned (>0.5%) | 0.345 | 0.340 | 0.006 | 0.224 | 0.026 | 3063 |  |
| 2025 | jump2 | spike pixel |  | 0.340 |  |  |  | 0 | UNDERPOWERED |
| 2025 | jump2 | ring of spike pixel |  | 0.340 |  |  |  | 0 | UNDERPOWERED |
| 2025 | rim | canned (>0.5%) | 0.523 | 0.538 | -0.016 | 0.052 | -0.302 | 79264 |  |
| 2025 | rim | spike pixel | 0.195 | 0.538 | -0.344 | 0.084 | -4.071 | 17848 |  |
| 2025 | rim | ring of spike pixel | 0.727 | 0.538 | 0.188 | 0.086 | 2.200 | 18063 |  |
| 2025 | three | canned (>0.5%) | 0.162 | 0.265 | -0.103 | 0.209 | -0.494 | 61 | UNDERPOWERED |
| 2025 | three | spike pixel |  | 0.265 |  |  |  | 29 | UNDERPOWERED |
| 2025 | three | ring of spike pixel |  | 0.265 |  |  |  | 12 | UNDERPOWERED |
| 2022 | rim | spike pixel MINUS its ring | 0.213 | 0.684 | -0.471 | 0.239 | -1.972 | 9823 |  |
| 2023 | rim | spike pixel MINUS its ring | 0.281 | 0.633 | -0.352 | 0.161 | -2.182 | 11958 |  |
| 2024 | rim | spike pixel MINUS its ring | 0.355 | 0.956 | -0.601 | 0.141 | -4.255 | 15036 |  |
| 2025 | rim | spike pixel MINUS its ring | 0.195 | 0.727 | -0.532 | 0.112 | -4.757 | 17848 |  |

### 3d. Regime: season-level structure of located shots

| season | located | located_share_of_attempts_pct | canned_share_pct | spike_pixel_share_pct | spike_share_of_rim_located_pct | top3_pixel_share_pct | distinct_pixels_per_10k_located | within_1ft_of_basket_pct | rim_mean_dist_ft | jump2_mean_dist_ft | three_mean_dist_ft |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 537854 | 80.005 | 10.798 | 2.684 | 7.555 | 4.037 | 71.897 | 3.569 | 3.407 | 11.470 | 24.834 |
| 2023 | 591564 | 84.335 | 10.592 | 2.874 | 7.893 | 4.229 | 65.826 | 3.902 | 3.372 | 11.392 | 24.855 |
| 2024 | 633884 | 87.545 | 11.766 | 2.933 | 7.963 | 4.314 | 62.283 | 3.951 | 3.363 | 11.115 | 24.818 |
| 2025 | 706606 | 98.627 | 13.613 | 2.978 | 8.039 | 4.463 | 57.062 | 4.055 | 3.291 | 11.006 | 24.846 |
| 2026 | 646718 | 87.758 | 13.999 | 3.118 | 8.038 | 4.573 | 61.186 | 4.142 | 3.273 | 11.199 | 24.833 |

Distance-from-basket histogram of located attempts (% of season's located)

| dist_bin_ft | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| [0, 1) | 3.570 | 3.900 | 3.950 | 4.060 | 4.140 |
| [1, 2) | 4.450 | 4.690 | 4.800 | 5.090 | 5.120 |
| [2, 3) | 8.540 | 8.640 | 8.700 | 8.800 | 9.120 |
| [3, 4) | 7.970 | 8.080 | 8.280 | 8.150 | 8.630 |
| [4, 6) | 11.470 | 11.470 | 11.850 | 11.510 | 11.340 |
| [6, 8) | 6.220 | 6.190 | 6.380 | 5.930 | 5.820 |
| [8, 10) | 4.180 | 4.150 | 4.150 | 4.040 | 3.910 |
| [10, 14) | 6.370 | 6.430 | 6.310 | 6.130 | 6.020 |
| [14, 18) | 5.660 | 5.600 | 5.380 | 4.840 | 4.440 |
| [18, 20) | 2.020 | 1.900 | 1.750 | 1.420 | 1.240 |
| [20, 22) | 1.090 | 0.900 | 0.820 | 0.690 | 0.620 |
| [22, 24) | 12.660 | 12.380 | 12.540 | 13.170 | 13.530 |
| [24, 26) | 18.150 | 17.930 | 17.690 | 18.140 | 18.220 |
| [26, 30) | 7.220 | 7.280 | 6.970 | 7.510 | 7.320 |
| [30, 40) | 0.420 | 0.440 | 0.420 | 0.520 | 0.520 |
| [40, 100) | 0.020 | 0.020 | 0.020 | 0.020 | 0.020 |

Total-variation distance of the histogram between consecutive seasons vs a split-half (random halves of games) noise floor

| from | to | tv_distance | split_half_noise_mean | split_half_noise_p95 |
|---|---|---|---|---|
| 2022 | 2023 | 0.009 | 0.006 | 0.008 |
| 2023 | 2024 | 0.011 | 0.006 | 0.008 |
| 2024 | 2025 | 0.022 | 0.006 | 0.008 |
| 2025 | 2026 | 0.014 | 0.006 | 0.007 |

Month by month (months 11-3; the `hoopr_pair` columns compare CBBD with hoopR's ESPN value on the same plays)

| season | month | attempts | located_pct | canned_pct | spike_pct | within_1ft_basket_pct | rim_mean_dist | three_mean_dist | text3_inside_arc_pct | layup_gt6ft_pct | hoopr_pair_n | hoopr_pair_le1ft_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 1 | 157713 | 84.807 | 10.771 | 2.534 | 3.387 | 3.403 | 24.816 | 1.909 | 10.073 | 2496 | 10.657 |
| 2022 | 2 | 163936 | 86.470 | 10.733 | 2.564 | 3.446 | 3.401 | 24.851 | 1.721 | 10.226 | 2041 | 10.289 |
| 2022 | 3 | 71798 | 63.665 | 10.770 | 2.689 | 3.745 | 3.396 | 24.970 | 1.616 | 11.092 | 3439 | 10.294 |
| 2022 | 11 | 152367 | 77.029 | 10.778 | 2.929 | 3.817 | 3.399 | 24.824 | 1.923 | 10.743 | 1315 | 10.114 |
| 2022 | 12 | 125956 | 78.742 | 10.966 | 2.766 | 3.616 | 3.432 | 24.787 | 1.855 | 10.867 | 1463 | 9.501 |
| 2023 | 1 | 163047 | 89.309 | 10.479 | 2.738 | 3.743 | 3.382 | 24.870 | 1.585 | 10.551 | 1600 | 9.250 |
| 2023 | 2 | 157154 | 88.187 | 10.701 | 2.799 | 3.889 | 3.336 | 24.870 | 1.442 | 10.018 | 1577 | 11.034 |
| 2023 | 3 | 63506 | 64.377 | 10.393 | 2.691 | 3.747 | 3.343 | 24.964 | 1.679 | 9.228 | 3686 | 8.953 |
| 2023 | 11 | 170410 | 81.793 | 10.391 | 2.978 | 3.916 | 3.439 | 24.840 | 1.742 | 11.253 | 851 | 7.286 |
| 2023 | 12 | 146989 | 86.440 | 10.890 | 3.059 | 4.132 | 3.338 | 24.804 | 1.696 | 10.273 | 930 | 9.892 |
| 2024 | 1 | 161752 | 92.408 | 11.624 | 2.809 | 3.740 | 3.404 | 24.807 | 1.763 | 10.725 | 1200 | 9.667 |
| 2024 | 2 | 159921 | 92.386 | 11.676 | 2.881 | 3.919 | 3.332 | 24.851 | 1.687 | 9.873 | 1370 | 9.416 |
| 2024 | 3 | 92947 | 75.736 | 11.765 | 2.911 | 4.064 | 3.339 | 24.881 | 1.844 | 9.844 | 3933 | 10.120 |
| 2024 | 11 | 163432 | 84.971 | 11.900 | 3.025 | 4.053 | 3.360 | 24.818 | 1.870 | 10.343 | 284 | 11.268 |
| 2024 | 12 | 145278 | 87.613 | 11.892 | 3.049 | 4.061 | 3.370 | 24.758 | 1.923 | 10.468 | 377 | 11.671 |
| 2025 | 1 | 165556 | 98.158 | 13.587 | 2.960 | 4.044 | 3.307 | 24.843 | 1.890 | 10.038 | 2321 | 6.894 |
| 2025 | 2 | 156556 | 98.820 | 13.760 | 2.945 | 4.070 | 3.287 | 24.867 | 1.857 | 9.306 | 94657 | 91.517 |
| 2025 | 3 | 89157 | 98.468 | 13.395 | 2.990 | 4.146 | 3.239 | 24.928 | 1.735 | 8.265 | 87790 | 94.406 |
| 2025 | 11 | 169865 | 98.714 | 13.269 | 2.986 | 3.955 | 3.344 | 24.833 | 1.830 | 10.537 | 708 | 5.085 |
| 2025 | 12 | 133248 | 98.964 | 14.080 | 3.019 | 4.108 | 3.250 | 24.770 | 1.961 | 9.639 | 596 | 3.859 |
| 2026 | 1 | 180478 | 90.894 | 14.111 | 3.052 | 4.002 | 3.300 | 24.839 | 1.657 | 9.005 | 164036 | 94.783 |
| 2026 | 2 | 157983 | 98.008 | 13.883 | 2.955 | 3.969 | 3.281 | 24.834 | 1.561 | 8.874 | 154833 | 94.737 |
| 2026 | 3 | 71881 | 99.083 | 13.827 | 3.291 | 4.527 | 3.146 | 24.917 | 1.703 | 7.632 | 71092 | 94.614 |
| 2026 | 11 | 185179 | 82.013 | 13.711 | 3.154 | 4.190 | 3.302 | 24.830 | 1.622 | 9.219 | 151825 | 94.774 |
| 2026 | 12 | 139811 | 73.953 | 14.565 | 3.285 | 4.274 | 3.264 | 24.764 | 1.838 | 8.983 | 103393 | 94.710 |

Weekly, seasons 2023-24 and 2024-25: the CBBD value vs the ESPN (hoopR) value on the same plays

| season | week_start | attempts | cbbd_located_pct | hoopr_located_pct | pairs | pair_le1ft_pct | layup_gt6ft_pct |
|---|---|---|---|---|---|---|---|
| 2024 | 2023-11-06 | 55907 | 85.764 | 0.453 | 31 | 16.129 | 11.532 |
| 2024 | 2023-11-13 | 44200 | 86.577 | 0.835 | 46 | 6.522 | 8.540 |
| 2024 | 2023-11-20 | 43588 | 79.247 | 2.517 | 169 | 11.243 | 11.040 |
| 2024 | 2023-11-27 | 39360 | 91.717 | 1.186 | 42 | 14.286 | 10.175 |
| 2024 | 2023-12-04 | 37666 | 86.001 | 3.520 | 119 | 9.244 | 11.167 |
| 2024 | 2023-12-11 | 27683 | 87.498 | 2.615 | 145 | 11.724 | 10.265 |
| 2024 | 2023-12-18 | 34803 | 85.840 | 2.141 | 109 | 13.761 | 10.471 |
| 2024 | 2023-12-25 | 25503 | 89.421 | 0 | 0 |  | 9.440 |
| 2024 | 2024-01-01 | 37176 | 92.573 | 3.231 | 183 | 9.836 | 11.616 |
| 2024 | 2024-01-08 | 36701 | 94.660 | 1.962 | 131 | 12.214 | 10.557 |
| 2024 | 2024-01-15 | 39248 | 92.214 | 4.859 | 396 | 7.828 | 10.195 |
| 2024 | 2024-01-22 | 38215 | 90.721 | 6.385 | 301 | 9.302 | 10.673 |
| 2024 | 2024-01-29 | 39159 | 91.966 | 5.590 | 418 | 9.569 | 9.676 |
| 2024 | 2024-02-05 | 38198 | 92.767 | 4.540 | 263 | 11.027 | 10.086 |
| 2024 | 2024-02-12 | 37589 | 92.900 | 4.616 | 226 | 9.292 | 9.814 |
| 2024 | 2024-02-19 | 37591 | 91.884 | 5.765 | 438 | 7.991 | 9.531 |
| 2024 | 2024-02-26 | 38693 | 92.766 | 5.453 | 353 | 13.314 | 10.474 |
| 2024 | 2024-03-04 | 35925 | 91.346 | 5.890 | 258 | 8.527 | 9.092 |
| 2024 | 2024-03-11 | 22427 | 50.756 | 57.872 | 2125 | 9.835 | 9.740 |
| 2024 | 2024-03-18 | 10670 | 51.125 | 57.676 | 971 | 10.711 | 12.034 |
| 2024 | 2024-03-25 | 3028 | 37.583 | 76.783 | 440 | 9.773 | 16.518 |
| 2024 | 2024-04-01 | 617 | 16.370 | 100 | 101 | 9.901 | 7.692 |
| 2024 | 2024-04-08 | 116 | 17.241 | 100 | 20 |  | 0 |
| 2025 | 2024-11-04 | 46934 | 98.281 | 0 | 0 |  | 10.587 |
| 2025 | 2024-11-11 | 42574 | 98.971 | 0.287 | 122 | 7.377 | 10.138 |
| 2025 | 2024-11-18 | 42576 | 98.748 | 0.550 | 229 | 4.367 | 10.570 |
| 2025 | 2024-11-25 | 41756 | 98.958 | 0.862 | 357 | 4.762 | 11.010 |
| 2025 | 2024-12-02 | 38717 | 98.990 | 0.579 | 223 | 3.587 | 10.417 |
| 2025 | 2024-12-09 | 26930 | 99.072 | 0.498 | 133 | 3.759 | 9.184 |
| 2025 | 2024-12-16 | 39427 | 99.219 | 0.619 | 240 | 4.167 | 9.329 |
| 2025 | 2024-12-23 | 15120 | 97.976 | 0 | 0 |  | 9.314 |
| 2025 | 2024-12-30 | 38317 | 98.922 | 0.394 | 150 | 2.667 | 9.312 |
| 2025 | 2025-01-06 | 38198 | 97.712 | 3.123 | 629 | 6.995 | 10.019 |
| 2025 | 2025-01-13 | 38598 | 98.225 | 2.117 | 462 | 9.740 | 10.299 |
| 2025 | 2025-01-20 | 39553 | 98.210 | 2.377 | 558 | 7.527 | 9.670 |
| 2025 | 2025-01-27 | 38807 | 97.841 | 3.533 | 852 | 4.695 | 9.870 |
| 2025 | 2025-02-03 | 38985 | 98.792 | 3.871 | 1335 | 3.745 | 10.253 |
| 2025 | 2025-02-10 | 38347 | 98.417 | 87.553 | 33005 | 90.829 | 9.798 |
| 2025 | 2025-02-17 | 38295 | 99.381 | 99.990 | 38058 | 94.217 | 8.208 |
| 2025 | 2025-02-24 | 40170 | 99.268 | 99.988 | 39876 | 94.556 | 9.254 |
| 2025 | 2025-03-03 | 36482 | 99.317 | 99.995 | 36233 | 94.373 | 8.307 |
| 2025 | 2025-03-10 | 21859 | 99.181 | 99.995 | 21680 | 94.211 | 7.949 |
| 2025 | 2025-03-17 | 9607 | 92.953 | 99.990 | 8929 | 94.635 | 6.717 |
| 2025 | 2025-03-24 | 2631 | 95.401 | 100 | 2510 | 94.343 | 9.198 |
| 2025 | 2025-03-31 | 2441 | 99.222 | 100 | 2422 | 94.509 | 5.700 |
| 2025 | 2025-04-07 | 122 | 100 | 100 | 122 | 86.885 | 6.667 |

### 3e. Charting accuracy against the play text (CBBD coordinates; distances folded to the near basket)

| season | check | n | misplaced_pct | flag |
|---|---|---|---|---|
| 2022 | text three charted inside 22.15 ft | 206705 | 1.829 |  |
| 2022 | text three charted inside 20.75 ft | 206705 | 0.127 |  |
| 2022 | text three charted inside 21.65 ft (0.5 ft slack) | 206705 | 0.311 |  |
| 2022 | text two-jumper charted outside 22.15 ft | 143502 | 0.388 |  |
| 2022 | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 143502 | 0.113 |  |
| 2022 | layup/tip charted > 6 ft from rim | 169238 | 11.281 |  |
| 2022 | dunk charted > 6 ft from rim | 18391 | 3.273 |  |
| 2022 | layup/dunk/tip charted > 10 ft from rim | 187629 | 0.789 |  |
| 2023 | text three charted inside 22.15 ft | 224841 | 1.619 |  |
| 2023 | text three charted inside 20.75 ft | 224841 | 0.143 |  |
| 2023 | text three charted inside 21.65 ft (0.5 ft slack) | 224841 | 0.274 |  |
| 2023 | text two-jumper charted outside 22.15 ft | 155040 | 0.342 |  |
| 2023 | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 155040 | 0.068 |  |
| 2023 | layup/tip charted > 6 ft from rim | 190734 | 11.197 |  |
| 2023 | dunk charted > 6 ft from rim | 20894 | 3.542 |  |
| 2023 | layup/dunk/tip charted > 10 ft from rim | 211628 | 0.823 |  |
| 2024 | text three charted inside 22.15 ft | 238613 | 1.810 |  |
| 2024 | text three charted inside 20.75 ft | 238613 | 0.133 |  |
| 2024 | text three charted inside 21.65 ft (0.5 ft slack) | 238613 | 0.280 |  |
| 2024 | text two-jumper charted outside 22.15 ft | 162667 | 0.210 |  |
| 2024 | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 162667 | 0.041 |  |
| 2024 | layup/tip charted > 6 ft from rim | 206984 | 11.019 |  |
| 2024 | dunk charted > 6 ft from rim | 21999 | 3.464 |  |
| 2024 | layup/dunk/tip charted > 10 ft from rim | 228983 | 0.738 |  |
| 2025 | text three charted inside 22.15 ft | 278323 | 1.858 |  |
| 2025 | text three charted inside 20.75 ft | 278323 | 0.138 |  |
| 2025 | text three charted inside 21.65 ft (0.5 ft slack) | 278323 | 0.287 |  |
| 2025 | text two-jumper charted outside 22.15 ft | 176693 | 0.159 |  |
| 2025 | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 176693 | 0.038 |  |
| 2025 | layup/tip charted > 6 ft from rim | 224716 | 10.316 |  |
| 2025 | dunk charted > 6 ft from rim | 24114 | 3.687 |  |
| 2025 | layup/dunk/tip charted > 10 ft from rim | 248830 | 0.667 |  |
| 2026 | text three charted inside 22.15 ft | 256220 | 1.661 |  |
| 2026 | text three charted inside 20.75 ft | 256220 | 0.028 |  |
| 2026 | text three charted inside 21.65 ft (0.5 ft slack) | 256220 | 0.157 |  |
| 2026 | text two-jumper charted outside 22.15 ft | 128658 | 0.175 |  |
| 2026 | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 128658 | 0.042 |  |
| 2026 | layup/tip charted > 6 ft from rim | 223372 | 9.541 |  |
| 2026 | dunk charted > 6 ft from rim | 26074 | 3.118 |  |
| 2026 | layup/dunk/tip charted > 10 ft from rim | 249446 | 0.552 |  |

Makes vs misses, and canned vs non-canned

| season | subset | check | n | misplaced_pct | flag |
|---|---|---|---|---|---|
| 2022 | makes | text three charted inside 22.15 ft | 69602 | 1.707 |  |
| 2022 | makes | text three charted inside 20.75 ft | 69602 | 0.029 |  |
| 2022 | makes | text three charted inside 21.65 ft (0.5 ft slack) | 69602 | 0.175 |  |
| 2022 | makes | text two-jumper charted outside 22.15 ft | 54509 | 0.143 |  |
| 2022 | makes | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 54509 | 0.013 |  |
| 2022 | makes | layup/tip charted > 6 ft from rim | 93471 | 7.457 |  |
| 2022 | makes | dunk charted > 6 ft from rim | 16144 | 2.843 |  |
| 2022 | makes | layup/dunk/tip charted > 10 ft from rim | 109615 | 0.359 |  |
| 2022 | misses | text three charted inside 22.15 ft | 137103 | 1.891 |  |
| 2022 | misses | text three charted inside 20.75 ft | 137103 | 0.177 |  |
| 2022 | misses | text three charted inside 21.65 ft (0.5 ft slack) | 137103 | 0.379 |  |
| 2022 | misses | text two-jumper charted outside 22.15 ft | 88993 | 0.538 |  |
| 2022 | misses | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 88993 | 0.174 |  |
| 2022 | misses | layup/tip charted > 6 ft from rim | 75767 | 15.998 |  |
| 2022 | misses | dunk charted > 6 ft from rim | 2247 | 6.364 |  |
| 2022 | misses | layup/dunk/tip charted > 10 ft from rim | 78014 | 1.395 |  |
| 2022 | canned pixels | text three charted inside 22.15 ft | 52 | 100 | UNDERPOWERED |
| 2022 | canned pixels | text three charted inside 20.75 ft | 52 | 100 | UNDERPOWERED |
| 2022 | canned pixels | text three charted inside 21.65 ft (0.5 ft slack) | 52 | 100 | UNDERPOWERED |
| 2022 | canned pixels | text two-jumper charted outside 22.15 ft | 3681 | 0 |  |
| 2022 | canned pixels | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 3681 | 0 |  |
| 2022 | canned pixels | layup/tip charted > 6 ft from rim | 46598 | 0 |  |
| 2022 | canned pixels | dunk charted > 6 ft from rim | 7747 | 0 |  |
| 2022 | canned pixels | layup/dunk/tip charted > 10 ft from rim | 54345 | 0 |  |
| 2022 | spike pixel | text three charted inside 22.15 ft | 30 | 100 | UNDERPOWERED |
| 2022 | spike pixel | text three charted inside 20.75 ft | 30 | 100 | UNDERPOWERED |
| 2022 | spike pixel | text three charted inside 21.65 ft (0.5 ft slack) | 30 | 100 | UNDERPOWERED |
| 2022 | spike pixel | text two-jumper charted outside 22.15 ft | 181 | 0 | UNDERPOWERED |
| 2022 | spike pixel | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 181 | 0 | UNDERPOWERED |
| 2022 | spike pixel | layup/tip charted > 6 ft from rim | 12127 | 0 |  |
| 2022 | spike pixel | dunk charted > 6 ft from rim | 2099 | 0 |  |
| 2022 | spike pixel | layup/dunk/tip charted > 10 ft from rim | 14226 | 0 |  |
| 2022 | non-canned | text three charted inside 22.15 ft | 206653 | 1.804 |  |
| 2022 | non-canned | text three charted inside 20.75 ft | 206653 | 0.102 |  |
| 2022 | non-canned | text three charted inside 21.65 ft (0.5 ft slack) | 206653 | 0.286 |  |
| 2022 | non-canned | text two-jumper charted outside 22.15 ft | 139821 | 0.398 |  |
| 2022 | non-canned | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 139821 | 0.116 |  |
| 2022 | non-canned | layup/tip charted > 6 ft from rim | 122640 | 15.567 |  |
| 2022 | non-canned | dunk charted > 6 ft from rim | 10644 | 5.656 |  |
| 2022 | non-canned | layup/dunk/tip charted > 10 ft from rim | 133284 | 1.111 |  |
| 2023 | makes | text three charted inside 22.15 ft | 76485 | 1.412 |  |
| 2023 | makes | text three charted inside 20.75 ft | 76485 | 0.024 |  |
| 2023 | makes | text three charted inside 21.65 ft (0.5 ft slack) | 76485 | 0.127 |  |
| 2023 | makes | text two-jumper charted outside 22.15 ft | 59735 | 0.127 |  |
| 2023 | makes | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 59735 | 0.010 |  |
| 2023 | makes | layup/tip charted > 6 ft from rim | 104888 | 7.330 |  |
| 2023 | makes | dunk charted > 6 ft from rim | 18336 | 3.049 |  |
| 2023 | makes | layup/dunk/tip charted > 10 ft from rim | 123224 | 0.373 |  |
| 2023 | misses | text three charted inside 22.15 ft | 148356 | 1.726 |  |
| 2023 | misses | text three charted inside 20.75 ft | 148356 | 0.205 |  |
| 2023 | misses | text three charted inside 21.65 ft (0.5 ft slack) | 148356 | 0.349 |  |
| 2023 | misses | text two-jumper charted outside 22.15 ft | 95305 | 0.476 |  |
| 2023 | misses | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 95305 | 0.105 |  |
| 2023 | misses | layup/tip charted > 6 ft from rim | 85846 | 15.922 |  |
| 2023 | misses | dunk charted > 6 ft from rim | 2558 | 7.076 |  |
| 2023 | misses | layup/dunk/tip charted > 10 ft from rim | 88404 | 1.449 |  |
| 2023 | canned pixels | text three charted inside 22.15 ft | 53 | 100 | UNDERPOWERED |
| 2023 | canned pixels | text three charted inside 20.75 ft | 53 | 100 | UNDERPOWERED |
| 2023 | canned pixels | text three charted inside 21.65 ft (0.5 ft slack) | 53 | 100 | UNDERPOWERED |
| 2023 | canned pixels | text two-jumper charted outside 22.15 ft | 3938 | 0 |  |
| 2023 | canned pixels | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 3938 | 0 |  |
| 2023 | canned pixels | layup/tip charted > 6 ft from rim | 50623 | 0 |  |
| 2023 | canned pixels | dunk charted > 6 ft from rim | 8042 | 0 |  |
| 2023 | canned pixels | layup/dunk/tip charted > 10 ft from rim | 58665 | 0 |  |
| 2023 | spike pixel | text three charted inside 22.15 ft | 29 | 100 | UNDERPOWERED |
| 2023 | spike pixel | text three charted inside 20.75 ft | 29 | 100 | UNDERPOWERED |
| 2023 | spike pixel | text three charted inside 21.65 ft (0.5 ft slack) | 29 | 100 | UNDERPOWERED |
| 2023 | spike pixel | text two-jumper charted outside 22.15 ft | 178 | 0 | UNDERPOWERED |
| 2023 | spike pixel | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 178 | 0 | UNDERPOWERED |
| 2023 | spike pixel | layup/tip charted > 6 ft from rim | 14442 | 0 |  |
| 2023 | spike pixel | dunk charted > 6 ft from rim | 2355 | 0 |  |
| 2023 | spike pixel | layup/dunk/tip charted > 10 ft from rim | 16797 | 0 |  |
| 2023 | non-canned | text three charted inside 22.15 ft | 224788 | 1.596 |  |
| 2023 | non-canned | text three charted inside 20.75 ft | 224788 | 0.120 |  |
| 2023 | non-canned | text three charted inside 21.65 ft (0.5 ft slack) | 224788 | 0.250 |  |
| 2023 | non-canned | text two-jumper charted outside 22.15 ft | 151102 | 0.351 |  |
| 2023 | non-canned | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 151102 | 0.070 |  |
| 2023 | non-canned | layup/tip charted > 6 ft from rim | 140111 | 15.242 |  |
| 2023 | non-canned | dunk charted > 6 ft from rim | 12852 | 5.758 |  |
| 2023 | non-canned | layup/dunk/tip charted > 10 ft from rim | 152963 | 1.138 |  |
| 2024 | makes | text three charted inside 22.15 ft | 80653 | 1.742 |  |
| 2024 | makes | text three charted inside 20.75 ft | 80653 | 0.040 |  |
| 2024 | makes | text three charted inside 21.65 ft (0.5 ft slack) | 80653 | 0.191 |  |
| 2024 | makes | text two-jumper charted outside 22.15 ft | 64023 | 0.116 |  |
| 2024 | makes | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 64023 | 0.006 |  |
| 2024 | makes | layup/tip charted > 6 ft from rim | 112748 | 7.085 |  |
| 2024 | makes | dunk charted > 6 ft from rim | 19305 | 2.880 |  |
| 2024 | makes | layup/dunk/tip charted > 10 ft from rim | 132053 | 0.292 |  |
| 2024 | misses | text three charted inside 22.15 ft | 157960 | 1.845 |  |
| 2024 | misses | text three charted inside 20.75 ft | 157960 | 0.181 |  |
| 2024 | misses | text three charted inside 21.65 ft (0.5 ft slack) | 157960 | 0.326 |  |
| 2024 | misses | text two-jumper charted outside 22.15 ft | 98644 | 0.272 |  |
| 2024 | misses | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 98644 | 0.064 |  |
| 2024 | misses | layup/tip charted > 6 ft from rim | 94236 | 15.725 |  |
| 2024 | misses | dunk charted > 6 ft from rim | 2694 | 7.647 |  |
| 2024 | misses | layup/dunk/tip charted > 10 ft from rim | 96930 | 1.345 |  |
| 2024 | canned pixels | text three charted inside 22.15 ft | 59 | 100 | UNDERPOWERED |
| 2024 | canned pixels | text three charted inside 20.75 ft | 59 | 100 | UNDERPOWERED |
| 2024 | canned pixels | text three charted inside 21.65 ft (0.5 ft slack) | 59 | 100 | UNDERPOWERED |
| 2024 | canned pixels | text two-jumper charted outside 22.15 ft | 5169 | 0 |  |
| 2024 | canned pixels | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 5169 | 0 |  |
| 2024 | canned pixels | layup/tip charted > 6 ft from rim | 60131 | 0 |  |
| 2024 | canned pixels | dunk charted > 6 ft from rim | 9086 | 0 |  |
| 2024 | canned pixels | layup/dunk/tip charted > 10 ft from rim | 69217 | 0 |  |
| 2024 | spike pixel | text three charted inside 22.15 ft | 28 | 100 | UNDERPOWERED |
| 2024 | spike pixel | text three charted inside 20.75 ft | 28 | 100 | UNDERPOWERED |
| 2024 | spike pixel | text three charted inside 21.65 ft (0.5 ft slack) | 28 | 100 | UNDERPOWERED |
| 2024 | spike pixel | text two-jumper charted outside 22.15 ft | 241 | 0 | UNDERPOWERED |
| 2024 | spike pixel | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 241 | 0 | UNDERPOWERED |
| 2024 | spike pixel | layup/tip charted > 6 ft from rim | 15823 | 0 |  |
| 2024 | spike pixel | dunk charted > 6 ft from rim | 2485 | 0 |  |
| 2024 | spike pixel | layup/dunk/tip charted > 10 ft from rim | 18308 | 0 |  |
| 2024 | non-canned | text three charted inside 22.15 ft | 238554 | 1.786 |  |
| 2024 | non-canned | text three charted inside 20.75 ft | 238554 | 0.109 |  |
| 2024 | non-canned | text three charted inside 21.65 ft (0.5 ft slack) | 238554 | 0.256 |  |
| 2024 | non-canned | text two-jumper charted outside 22.15 ft | 157498 | 0.217 |  |
| 2024 | non-canned | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 157498 | 0.043 |  |
| 2024 | non-canned | layup/tip charted > 6 ft from rim | 146853 | 15.530 |  |
| 2024 | non-canned | dunk charted > 6 ft from rim | 12913 | 5.901 |  |
| 2024 | non-canned | layup/dunk/tip charted > 10 ft from rim | 159766 | 1.058 |  |
| 2025 | makes | text three charted inside 22.15 ft | 94083 | 1.757 |  |
| 2025 | makes | text three charted inside 20.75 ft | 94083 | 0.048 |  |
| 2025 | makes | text three charted inside 21.65 ft (0.5 ft slack) | 94083 | 0.176 |  |
| 2025 | makes | text two-jumper charted outside 22.15 ft | 71841 | 0.074 |  |
| 2025 | makes | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 71841 | 0.010 |  |
| 2025 | makes | layup/tip charted > 6 ft from rim | 122434 | 6.573 |  |
| 2025 | makes | dunk charted > 6 ft from rim | 21166 | 3.132 |  |
| 2025 | makes | layup/dunk/tip charted > 10 ft from rim | 143600 | 0.257 |  |
| 2025 | misses | text three charted inside 22.15 ft | 184240 | 1.909 |  |
| 2025 | misses | text three charted inside 20.75 ft | 184240 | 0.185 |  |
| 2025 | misses | text three charted inside 21.65 ft (0.5 ft slack) | 184240 | 0.344 |  |
| 2025 | misses | text two-jumper charted outside 22.15 ft | 104852 | 0.217 |  |
| 2025 | misses | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 104852 | 0.057 |  |
| 2025 | misses | layup/tip charted > 6 ft from rim | 102282 | 14.796 |  |
| 2025 | misses | dunk charted > 6 ft from rim | 2948 | 7.666 |  |
| 2025 | misses | layup/dunk/tip charted > 10 ft from rim | 105230 | 1.227 |  |
| 2025 | canned pixels | text three charted inside 22.15 ft | 82 | 100 | UNDERPOWERED |
| 2025 | canned pixels | text three charted inside 20.75 ft | 82 | 100 | UNDERPOWERED |
| 2025 | canned pixels | text three charted inside 21.65 ft (0.5 ft slack) | 82 | 100 | UNDERPOWERED |
| 2025 | canned pixels | text two-jumper charted outside 22.15 ft | 14306 | 0 |  |
| 2025 | canned pixels | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 14306 | 0 |  |
| 2025 | canned pixels | layup/tip charted > 6 ft from rim | 70724 | 0 |  |
| 2025 | canned pixels | dunk charted > 6 ft from rim | 10963 | 0 |  |
| 2025 | canned pixels | layup/dunk/tip charted > 10 ft from rim | 81687 | 0 |  |
| 2025 | spike pixel | text three charted inside 22.15 ft | 40 | 100 | UNDERPOWERED |
| 2025 | spike pixel | text three charted inside 20.75 ft | 40 | 100 | UNDERPOWERED |
| 2025 | spike pixel | text three charted inside 21.65 ft (0.5 ft slack) | 40 | 100 | UNDERPOWERED |
| 2025 | spike pixel | text two-jumper charted outside 22.15 ft | 8671 | 0 |  |
| 2025 | spike pixel | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 8671 | 0 |  |
| 2025 | spike pixel | layup/tip charted > 6 ft from rim | 11048 | 0 |  |
| 2025 | spike pixel | dunk charted > 6 ft from rim | 1277 | 0 |  |
| 2025 | spike pixel | layup/dunk/tip charted > 10 ft from rim | 12325 | 0 |  |
| 2025 | non-canned | text three charted inside 22.15 ft | 278241 | 1.829 |  |
| 2025 | non-canned | text three charted inside 20.75 ft | 278241 | 0.109 |  |
| 2025 | non-canned | text three charted inside 21.65 ft (0.5 ft slack) | 278241 | 0.258 |  |
| 2025 | non-canned | text two-jumper charted outside 22.15 ft | 162387 | 0.173 |  |
| 2025 | non-canned | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 162387 | 0.041 |  |
| 2025 | non-canned | layup/tip charted > 6 ft from rim | 153992 | 15.053 |  |
| 2025 | non-canned | dunk charted > 6 ft from rim | 13151 | 6.760 |  |
| 2025 | non-canned | layup/dunk/tip charted > 10 ft from rim | 167143 | 0.993 |  |
| 2026 | canned pixels | text three charted inside 22.15 ft | 10 | 100 | UNDERPOWERED |
| 2026 | canned pixels | text three charted inside 20.75 ft | 10 | 100 | UNDERPOWERED |
| 2026 | canned pixels | text three charted inside 21.65 ft (0.5 ft slack) | 10 | 100 | UNDERPOWERED |
| 2026 | canned pixels | text two-jumper charted outside 22.15 ft | 2348 | 0 |  |
| 2026 | canned pixels | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 2348 | 0 |  |
| 2026 | canned pixels | layup/tip charted > 6 ft from rim | 75595 | 0 |  |
| 2026 | canned pixels | dunk charted > 6 ft from rim | 12220 | 0 |  |
| 2026 | canned pixels | layup/dunk/tip charted > 10 ft from rim | 87815 | 0 |  |
| 2026 | spike pixel | text three charted inside 22.15 ft | 4 | 100 | UNDERPOWERED |
| 2026 | spike pixel | text three charted inside 20.75 ft | 4 | 100 | UNDERPOWERED |
| 2026 | spike pixel | text three charted inside 21.65 ft (0.5 ft slack) | 4 | 100 | UNDERPOWERED |
| 2026 | spike pixel | text two-jumper charted outside 22.15 ft | 41 | 0 | UNDERPOWERED |
| 2026 | spike pixel | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 41 | 0 | UNDERPOWERED |
| 2026 | spike pixel | layup/tip charted > 6 ft from rim | 17399 | 0 |  |
| 2026 | spike pixel | dunk charted > 6 ft from rim | 2709 | 0 |  |
| 2026 | spike pixel | layup/dunk/tip charted > 10 ft from rim | 20108 | 0 |  |
| 2026 | non-canned | text three charted inside 22.15 ft | 256210 | 1.658 |  |
| 2026 | non-canned | text three charted inside 20.75 ft | 256210 | 0.024 |  |
| 2026 | non-canned | text three charted inside 21.65 ft (0.5 ft slack) | 256210 | 0.153 |  |
| 2026 | non-canned | text two-jumper charted outside 22.15 ft | 126310 | 0.178 |  |
| 2026 | non-canned | text two-jumper charted outside 23.65 ft (1.5 ft slack) | 126310 | 0.043 |  |
| 2026 | non-canned | layup/tip charted > 6 ft from rim | 147777 | 14.421 |  |
| 2026 | non-canned | dunk charted > 6 ft from rim | 13854 | 5.868 |  |
| 2026 | non-canned | layup/dunk/tip charted > 10 ft from rim | 161631 | 0.852 |  |

Paired test: is misplacement different for makes and misses within game (a = makes)?

| season | check | a | b | n_a | n_b | rate_a | rate_b | pooled_diff | games_paired | within_game_diff | boot_se | z | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | text three inside 22.15 ft | makes | misses | 69602 | 137103 | 0.017 | 0.019 | -0.002 | 5293 | -0.002 | 0.001 | -3.064 |  |
| 2022 | layup/dunk/tip > 6 ft | makes | misses | 109615 | 78014 | 0.068 | 0.157 | -0.089 | 5358 | -0.088 | 0.002 | -50.089 |  |
| 2022 | text two-jumper outside 22.15 ft | makes | misses | 54509 | 88993 | 0.001 | 0.005 | -0.004 | 5231 | -0.004 | 0.000 | -12.197 |  |
| 2023 | text three inside 22.15 ft | makes | misses | 76485 | 148356 | 0.014 | 0.017 | -0.003 | 5696 | -0.003 | 0.001 | -5.646 |  |
| 2023 | layup/dunk/tip > 6 ft | makes | misses | 123224 | 88404 | 0.067 | 0.157 | -0.090 | 5748 | -0.089 | 0.002 | -55.552 |  |
| 2023 | text two-jumper outside 22.15 ft | makes | misses | 59735 | 95305 | 0.001 | 0.005 | -0.003 | 5640 | -0.003 | 0.000 | -11.898 |  |
| 2024 | text three inside 22.15 ft | makes | misses | 80653 | 157960 | 0.017 | 0.018 | -0.001 | 5859 | -0.001 | 0.001 | -1.969 |  |
| 2024 | layup/dunk/tip > 6 ft | makes | misses | 132053 | 96930 | 0.065 | 0.155 | -0.090 | 5888 | -0.089 | 0.002 | -53.567 |  |
| 2024 | text two-jumper outside 22.15 ft | makes | misses | 64023 | 98644 | 0.001 | 0.003 | -0.002 | 5809 | -0.002 | 0.000 | -6.587 |  |
| 2025 | text three inside 22.15 ft | makes | misses | 94083 | 184240 | 0.018 | 0.019 | -0.002 | 6124 | -0.002 | 0.001 | -3.040 |  |
| 2025 | layup/dunk/tip > 6 ft | makes | misses | 143600 | 105230 | 0.061 | 0.146 | -0.085 | 6123 | -0.084 | 0.002 | -52.697 |  |
| 2025 | text two-jumper outside 22.15 ft | makes | misses | 71841 | 104852 | 0.001 | 0.002 | -0.001 | 6117 | -0.001 | 0.000 | -7.791 |  |

hoopR (ESPN) coordinates on hoopR's own located attempts

| season | hoopr_located | text three inside 22.15 ft (%) | text two-jumper outside 22.15 ft (%) | layup/dunk/tip > 6 ft (%) | n_three | n_rimtext |
|---|---|---|---|---|---|---|
| 2022 | 81360 | 0.137 | 0.417 | 0.159 | 29896 | 27735 |
| 2023 | 61885 | 0.209 | 0.331 | 0.164 | 22485 | 21291 |
| 2024 | 45491 | 0.123 | 0.351 | 0.080 | 17076 | 16173 |
| 2025 | 192467 | 5.594 | 0.201 | 9.029 | 74827 | 71835 |
| 2026 | 736281 | 2.956 | 0.290 | 9.878 | 291535 | 284751 |


## What this means for a shot-location stage (descriptive only; no recommendation to adopt anything)

- **Where the signal looks reliable.** Jumper and three placements agree with the play text on more than 98% of attempts in all four seasons, they are outcome-blind in whether they are located, and their misplacement differs between makes and misses by at most 0.5 points. Location-by-make-rate curves for those classes are measuring where shots are taken.
- **Where it is contaminated.** For rim attempts, distance from the basket is partly a function of the result: misses are charted more than 6 ft from the rim 2.3-2.4 times as often as makes, and makes are 1.3-2.1 points less likely to carry any coordinate. A make model that takes a located rim distance (or the per-attempt cell rate of the earlier round) as an input can therefore learn "charted far from the rim implies miss". A shot-location stage that samples locations from the located-shot distribution would reproduce the same coupling in the simulation, in the direction that flatters the make model offline. The evidence of this audit is that the outcome dependence is present in the ESPN values themselves in 2025-26 (hoopR and CBBD agree), not only in one vendor's processing.
- **Coverage is season-dependent and not random before 2025.** 2022-2024 coordinates exist for 80-88% of attempts, missing in runs inside games, with the gaps concentrated in power-conference, neutral-site, March and postseason games; 2025 is 98.6% and even. The 2025-26 season has a different gap pattern (11% of games entirely unlocated in CBBD, mostly low-tier games with no line), and hoopR fills it. Per-team rating features built on located shots rest on 1.0-1.7k located attempts per team-season at the 10th percentile.
- **Two measurement vintages.** For 2022-2024 the two available sources disagree on the same plays (about 9-10% within 1 ft), and the less-covered source (hoopR) is the one that agrees with the play text. For 2025-26 the sources agree and share the same text-inconsistency profile. There is no second-source confirmation of CBBD's 2022-2024 positions, which is where a stage trained on folds 1 and 2 would get most of its data.
- **One placeholder-like pixel.** A single pixel next to the basket centre holds about 3% of located attempts (8% of located rim attempts) in every season including the sealed one; its make rate is below that of its neighbours and its response to team shooting skill is about 0.3-0.45 times its neighbours', which is the only canned-pixel signature in the data. The other 14-18 pixels above 0.5% are the dense layup region and are not placeholders on any test here.
- **Stage mechanics already known.** The engine samples no location; a stage would need class-conditional location sampling that respects the missing-by-outcome pattern, and its own bottom-up gate. Nothing in this audit evaluates such a stage.
- **Not tested here.** Whether the rim outcome dependence is symmetric on the defence side, whether it varies by arena/broadcast crew, and whether the same asymmetry exists in a non-ESPN source; none is available on disk.
