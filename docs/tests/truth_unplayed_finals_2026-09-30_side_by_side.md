games graded: current 5710, corrected 5705

### G1 -- Possessions per game, mean and SD (overall, by month)

| quantity | current value | corrected value | target (cur / corr) | tolerance | status cur | status corr | changed |
|---|---|---|---|---|---|---|---|
| possessions/game mean | 69.866 vs 67.875 | 69.865 vs 67.875 | 67.875 / 67.875 | +/-1.0 | FAIL | FAIL | value |
| possessions/game SD | 5.552 vs 5.474 | 5.552 vs 5.474 | 5.474 / 5.474 | +/-0.75 | PASS | PASS |  |
| by month (mean and SD) | 0/5 powered months inside | 0/5 powered months inside | all inside / all inside | see per-month table | FAIL | FAIL |  |
| **gate overall** | | | | | **FAIL** | **FAIL** |  |

table `by breakdown`: 5/7 rows change (current then corrected)

| breakdown | group | n_games | sim_mean | actual_mean | ref_mean | sim_sd | actual_sd | ref_sd | d_mean | d_sd | status_mean | status_sd | truth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| season | all | 5710 | 69.8661 | 67.8753 | 67.8753 | 5.5520 | 5.4741 | 5.4741 | 1.9909 | 0.0779 | FAIL | PASS | current |
| season | all | 5705 | 69.8652 | 67.8753 | 67.8753 | 5.5523 | 5.4741 | 5.4741 | 1.9899 | 0.0782 | FAIL | PASS | corrected |
| month | 1 | 1423 | 69.6199 | 67.4339 | 67.4339 | 5.5594 | 5.4817 | 5.4817 | 2.1860 | 0.0777 | FAIL | PASS | current |
| month | 1 | 1422 | 69.6190 | 67.4339 | 67.4339 | 5.5597 | 5.4817 | 5.4817 | 2.1851 | 0.0780 | FAIL | PASS | corrected |
| month | 2 | 1367 | 69.3174 | 67.2344 | 67.2344 | 5.4575 | 5.1604 | 5.1604 | 2.0830 | 0.2971 | FAIL | PASS | current |
| month | 2 | 1365 | 69.3153 | 67.2344 | 67.2344 | 5.4582 | 5.1604 | 5.1604 | 2.0809 | 0.2977 | FAIL | PASS | corrected |
| month | 11 | 1222 | 70.9927 | 69.0697 | 69.0697 | 5.5243 | 5.5818 | 5.5818 | 1.9229 | -0.0575 | FAIL | PASS | current |
| month | 11 | 1221 | 70.9926 | 69.0697 | 69.0697 | 5.5245 | 5.5818 | 5.5818 | 1.9228 | -0.0573 | FAIL | PASS | corrected |
| month | 12 | 916 | 70.1059 | 68.2291 | 68.2291 | 5.5216 | 5.5994 | 5.5994 | 1.8768 | -0.0779 | FAIL | PASS | current |
| month | 12 | 915 | 70.1043 | 68.2291 | 68.2291 | 5.5218 | 5.5994 | 5.5994 | 1.8752 | -0.0776 | FAIL | PASS | corrected |

### G2 -- Points per possession, by offense tercile x defense tercile

| quantity | current value | corrected value | target (cur / corr) | tolerance | status cur | status corr | changed |
|---|---|---|---|---|---|---|---|
| PPP by offense x defense tercile (9 cells) | 2/9 powered cells inside +/-0.02 | 2/9 powered cells inside +/-0.02 | all inside / all inside | +/-0.02 per cell | FAIL | FAIL |  |
| **gate overall** | | | | | **FAIL** | **FAIL** |  |

table `by tercile cell`: 9/9 rows change (current then corrected)

| offense_tercile | defense_tercile | n_team_games | sim_ppp | actual_ppp | delta | status | truth |
|---|---|---|---|---|---|---|---|
| bottom_tercile | bottom_tercile | 1275 | 0.9606 | 0.9659 | -0.0053 | PASS | current |
| bottom_tercile | bottom_tercile | 1274 | 0.9619 | 0.9658 | -0.0039 | PASS | corrected |
| bottom_tercile | middle_tercile | 1247 | 0.9923 | 1.0139 | -0.0215 | FAIL | current |
| bottom_tercile | middle_tercile | 1241 | 0.9907 | 1.0119 | -0.0213 | FAIL | corrected |
| bottom_tercile | top_tercile | 1147 | 1.0227 | 1.0584 | -0.0357 | FAIL | current |
| bottom_tercile | top_tercile | 1145 | 1.0232 | 1.0602 | -0.0370 | FAIL | corrected |
| middle_tercile | bottom_tercile | 1254 | 1.0007 | 1.0189 | -0.0183 | PASS | current |
| middle_tercile | bottom_tercile | 1254 | 0.9999 | 1.0193 | -0.0194 | PASS | corrected |
| middle_tercile | middle_tercile | 1298 | 1.0365 | 1.0737 | -0.0372 | FAIL | current |
| middle_tercile | middle_tercile | 1296 | 1.0363 | 1.0741 | -0.0379 | FAIL | corrected |
| middle_tercile | top_tercile | 1262 | 1.0646 | 1.1189 | -0.0543 | FAIL | current |
| middle_tercile | top_tercile | 1243 | 1.0653 | 1.1182 | -0.0529 | FAIL | corrected |
| top_tercile | bottom_tercile | 1429 | 1.0464 | 1.0851 | -0.0387 | FAIL | current |
| top_tercile | bottom_tercile | 1430 | 1.0461 | 1.0852 | -0.0391 | FAIL | corrected |
| top_tercile | middle_tercile | 1252 | 1.0835 | 1.1346 | -0.0511 | FAIL | current |
| top_tercile | middle_tercile | 1258 | 1.0827 | 1.1322 | -0.0494 | FAIL | corrected |
| top_tercile | top_tercile | 1236 | 1.1179 | 1.1908 | -0.0729 | FAIL | current |
| top_tercile | top_tercile | 1259 | 1.1178 | 1.1913 | -0.0735 | FAIL | corrected |

### G3 -- Shot mix per possession: 3PA share, rim share, FTA/FGA (by team)

| quantity | current value | corrected value | target (cur / corr) | tolerance | status cur | status corr | changed |
|---|---|---|---|---|---|---|---|
| 3PA share & FTA/FGA by team | 0/0 powered team-metrics inside | 0/0 powered team-metrics inside | all inside / all inside | +/-1.5pp | NEEDS-INSTRUMENTATION | NEEDS-INSTRUMENTATION |  |
| rim share by team | 0/0 powered teams inside | 0/0 powered teams inside | all inside / all inside | +/-1.5pp (PROVISIONAL truth) | NEEDS-INSTRUMENTATION | NEEDS-INSTRUMENTATION |  |
| three_pa_share (season, pooled, PROVISIONAL) | 0.3868 vs 0.3906 | 0.3868 vs 0.3906 | 0.3906 / 0.3906 | +/-1.5pp | PASS | PASS |  |
| fta_per_fga (season, pooled, PROVISIONAL) | 0.3172 vs 0.3295 | 0.3172 vs 0.3295 | 0.3295 / 0.3295 | +/-1.5pp | PASS | PASS |  |
| rim_share (season, pooled, PROVISIONAL) | 0.3715 vs 0.3733 | 0.3715 vs 0.3733 | 0.3733 / 0.3733 | +/-1.5pp | PASS | PASS |  |
| **gate overall** | | | | | **NEEDS-INSTRUMENTATION** | **NEEDS-INSTRUMENTATION** |  |

table `by team-metric`: 30/1092 rows change (current then corrected)

| metric | team_id | n_games | sim | actual | delta_pp | status | truth |
|---|---|---|---|---|---|---|---|
| three_pa_share | 93 | 32 | 0.3946 | 0.4012 | -0.6597 | UNDERPOWERED | current |
| three_pa_share | 93 | 32 | 0.3951 | 0.4012 | -0.6053 | UNDERPOWERED | corrected |
| three_pa_share | 155 | 30 | 0.4049 | 0.3979 | 0.6994 | UNDERPOWERED | current |
| three_pa_share | 155 | 30 | 0.4054 | 0.3979 | 0.7520 | UNDERPOWERED | corrected |
| three_pa_share | 2031 | 30 | 0.3342 | 0.2972 | 3.6940 | UNDERPOWERED | current |
| three_pa_share | 2031 | 30 | 0.3328 | 0.2972 | 3.5564 | UNDERPOWERED | corrected |
| three_pa_share | 2197 | 28 | 0.3194 | 0.2996 | 1.9815 | UNDERPOWERED | current |
| three_pa_share | 2197 | 28 | 0.3188 | 0.2996 | 1.9206 | UNDERPOWERED | corrected |
| three_pa_share | 2261 | 30 | 0.3659 | 0.3680 | -0.2148 | UNDERPOWERED | current |
| three_pa_share | 2261 | 30 | 0.3659 | 0.3680 | -0.2122 | UNDERPOWERED | corrected |
| three_pa_share | 2351 | 31 | 0.4078 | 0.4024 | 0.5396 | UNDERPOWERED | current |
| three_pa_share | 2351 | 31 | 0.4088 | 0.4024 | 0.6424 | UNDERPOWERED | corrected |
| three_pa_share | 2460 | 32 | 0.3817 | 0.3636 | 1.8063 | UNDERPOWERED | current |
| three_pa_share | 2460 | 32 | 0.3821 | 0.3636 | 1.8546 | UNDERPOWERED | corrected |
| three_pa_share | 2492 | 34 | 0.3496 | 0.3471 | 0.2474 | UNDERPOWERED | current |
| three_pa_share | 2492 | 34 | 0.3498 | 0.3471 | 0.2701 | UNDERPOWERED | corrected |
| three_pa_share | 2619 | 31 | 0.3867 | 0.3916 | -0.4907 | UNDERPOWERED | current |
| three_pa_share | 2619 | 31 | 0.3872 | 0.3916 | -0.4411 | UNDERPOWERED | corrected |
| three_pa_share | 2636 | 29 | 0.4070 | 0.4410 | -3.4031 | UNDERPOWERED | current |
| three_pa_share | 2636 | 29 | 0.4060 | 0.4410 | -3.5025 | UNDERPOWERED | corrected |
| fta_per_fga | 93 | 32 | 0.3520 | 0.4169 | -6.4832 | UNDERPOWERED | current |
| fta_per_fga | 93 | 32 | 0.3514 | 0.4169 | -6.5456 | UNDERPOWERED | corrected |
| fta_per_fga | 155 | 30 | 0.3216 | 0.3425 | -2.0941 | UNDERPOWERED | current |
| fta_per_fga | 155 | 30 | 0.3231 | 0.3425 | -1.9379 | UNDERPOWERED | corrected |
| fta_per_fga | 2031 | 30 | 0.3355 | 0.2947 | 4.0808 | UNDERPOWERED | current |
| fta_per_fga | 2031 | 30 | 0.3351 | 0.2947 | 4.0409 | UNDERPOWERED | corrected |
| fta_per_fga | 2197 | 28 | 0.3351 | 0.3349 | 0.0212 | UNDERPOWERED | current |
| fta_per_fga | 2197 | 28 | 0.3342 | 0.3349 | -0.0707 | UNDERPOWERED | corrected |
| fta_per_fga | 2261 | 30 | 0.3022 | 0.3096 | -0.7370 | UNDERPOWERED | current |
| fta_per_fga | 2261 | 30 | 0.3004 | 0.3096 | -0.9174 | UNDERPOWERED | corrected |
| fta_per_fga | 2351 | 31 | 0.2972 | 0.2957 | 0.1505 | UNDERPOWERED | current |
| fta_per_fga | 2351 | 31 | 0.2971 | 0.2957 | 0.1475 | UNDERPOWERED | corrected |
| fta_per_fga | 2460 | 32 | 0.3291 | 0.3535 | -2.4365 | UNDERPOWERED | current |
| fta_per_fga | 2460 | 32 | 0.3276 | 0.3535 | -2.5848 | UNDERPOWERED | corrected |
| fta_per_fga | 2492 | 34 | 0.3373 | 0.3233 | 1.4022 | UNDERPOWERED | current |
| fta_per_fga | 2492 | 34 | 0.3371 | 0.3233 | 1.3819 | UNDERPOWERED | corrected |
| fta_per_fga | 2619 | 31 | 0.2981 | 0.3213 | -2.3217 | UNDERPOWERED | current |
| fta_per_fga | 2619 | 31 | 0.2970 | 0.3213 | -2.4259 | UNDERPOWERED | corrected |
| fta_per_fga | 2636 | 29 | 0.3151 | 0.2945 | 2.0533 | UNDERPOWERED | current |
| fta_per_fga | 2636 | 29 | 0.3116 | 0.2945 | 1.7021 | UNDERPOWERED | corrected |
| rim_share | 93 | 31 | 0.4146 | 0.4411 | -2.6525 | UNDERPOWERED | current |
| rim_share | 93 | 31 | 0.4136 | 0.4411 | -2.7468 | UNDERPOWERED | corrected |
| rim_share | 155 | 30 | 0.3721 | 0.4362 | -6.4107 | UNDERPOWERED | current |
| rim_share | 155 | 30 | 0.3729 | 0.4362 | -6.3249 | UNDERPOWERED | corrected |
| rim_share | 2031 | 29 | 0.3510 | 0.3269 | 2.4050 | UNDERPOWERED | current |
| rim_share | 2031 | 29 | 0.3513 | 0.3269 | 2.4430 | UNDERPOWERED | corrected |
| rim_share | 2197 | 24 | 0.3653 | 0.4026 | -3.7302 | UNDERPOWERED | current |
| rim_share | 2197 | 24 | 0.3665 | 0.4026 | -3.6070 | UNDERPOWERED | corrected |
| rim_share | 2261 | 28 | 0.3331 | 0.3114 | 2.1722 | UNDERPOWERED | current |
| rim_share | 2261 | 28 | 0.3332 | 0.3114 | 2.1762 | UNDERPOWERED | corrected |
| rim_share | 2351 | 31 | 0.3230 | 0.3137 | 0.9331 | UNDERPOWERED | current |
| rim_share | 2351 | 31 | 0.3223 | 0.3137 | 0.8588 | UNDERPOWERED | corrected |
| rim_share | 2460 | 32 | 0.3944 | 0.4158 | -2.1430 | UNDERPOWERED | current |
| rim_share | 2460 | 32 | 0.3934 | 0.4158 | -2.2479 | UNDERPOWERED | corrected |
| rim_share | 2492 | 34 | 0.3745 | 0.3611 | 1.3352 | UNDERPOWERED | current |
| rim_share | 2492 | 34 | 0.3732 | 0.3611 | 1.2040 | UNDERPOWERED | corrected |
| rim_share | 2619 | 29 | 0.3336 | 0.3149 | 1.8663 | UNDERPOWERED | current |
| rim_share | 2619 | 29 | 0.3331 | 0.3149 | 1.8164 | UNDERPOWERED | corrected |
| rim_share | 2636 | 29 | 0.3474 | 0.3268 | 2.0560 | UNDERPOWERED | current |
| rim_share | 2636 | 29 | 0.3472 | 0.3268 | 2.0406 | UNDERPOWERED | corrected |

### G4 -- Four factors, offense and defense (by team, by tier)

| quantity | current value | corrected value | target (cur / corr) | tolerance | status cur | status corr | changed |
|---|---|---|---|---|---|---|---|
| eFG% (offense/defense) | 0/0 powered teams inside | 0/0 powered teams inside | all inside / all inside | +/-1.0pp (PROVISIONAL truth) | NEEDS-INSTRUMENTATION | NEEDS-INSTRUMENTATION |  |
| tov_pct by team | 0/0 powered teams inside | 0/0 powered teams inside | all inside / all inside | +/-1.0pp | NEEDS-INSTRUMENTATION | NEEDS-INSTRUMENTATION |  |
| oreb_pct by team | 0/0 powered teams inside | 0/0 powered teams inside | all inside / all inside | +/-1.0pp | NEEDS-INSTRUMENTATION | NEEDS-INSTRUMENTATION |  |
| ft_rate by team | 0/0 powered teams inside | 0/0 powered teams inside | all inside / all inside | +/-0.015 | NEEDS-INSTRUMENTATION | NEEDS-INSTRUMENTATION |  |
| tov_pct (season, pooled, PROVISIONAL) | 0.1756 vs 0.1739 | 0.1756 vs 0.1739 | 0.1739 / 0.1739 | +/-1.0pp | PASS | PASS |  |
| oreb_pct (season, pooled, PROVISIONAL) | 0.2829 vs 0.2984 | 0.2829 vs 0.2984 | 0.2984 / 0.2984 | +/-1.0pp | FAIL | FAIL |  |
| ft_rate (season, pooled, PROVISIONAL) | 0.3172 vs 0.3295 | 0.3172 vs 0.3295 | 0.3295 / 0.3295 | +/-0.015 | PASS | PASS |  |
| efg_pct (season, pooled, PROVISIONAL) | 0.4975 vs 0.5086 | 0.4975 vs 0.5086 | 0.5086 / 0.5086 | +/-1.0pp | FAIL | FAIL |  |
| **gate overall** | | | | | **FAIL** | **FAIL** |  |

table `by team-metric`: 40/1456 rows change (current then corrected)

| metric | team_id | n_games | sim | actual | delta | status | truth |
|---|---|---|---|---|---|---|---|
| tov_pct | 93 | 32 | 0.1802 | 0.1720 | 0.0083 | UNDERPOWERED | current |
| tov_pct | 93 | 32 | 0.1806 | 0.1720 | 0.0087 | UNDERPOWERED | corrected |
| tov_pct | 155 | 30 | 0.1764 | 0.1666 | 0.0098 | UNDERPOWERED | current |
| tov_pct | 155 | 30 | 0.1752 | 0.1666 | 0.0086 | UNDERPOWERED | corrected |
| tov_pct | 2031 | 30 | 0.1790 | 0.2015 | -0.0225 | UNDERPOWERED | current |
| tov_pct | 2031 | 30 | 0.1788 | 0.2015 | -0.0227 | UNDERPOWERED | corrected |
| tov_pct | 2197 | 28 | 0.1866 | 0.1875 | -0.0008 | UNDERPOWERED | current |
| tov_pct | 2197 | 28 | 0.1862 | 0.1875 | -0.0013 | UNDERPOWERED | corrected |
| tov_pct | 2261 | 30 | 0.1717 | 0.1610 | 0.0107 | UNDERPOWERED | current |
| tov_pct | 2261 | 30 | 0.1721 | 0.1610 | 0.0111 | UNDERPOWERED | corrected |
| tov_pct | 2351 | 31 | 0.1645 | 0.1533 | 0.0112 | UNDERPOWERED | current |
| tov_pct | 2351 | 31 | 0.1637 | 0.1533 | 0.0105 | UNDERPOWERED | corrected |
| tov_pct | 2460 | 32 | 0.1650 | 0.1620 | 0.0030 | UNDERPOWERED | current |
| tov_pct | 2460 | 32 | 0.1652 | 0.1620 | 0.0033 | UNDERPOWERED | corrected |
| tov_pct | 2492 | 34 | 0.1756 | 0.1623 | 0.0134 | UNDERPOWERED | current |
| tov_pct | 2492 | 34 | 0.1759 | 0.1623 | 0.0136 | UNDERPOWERED | corrected |
| tov_pct | 2619 | 31 | 0.1811 | 0.1837 | -0.0027 | UNDERPOWERED | current |
| tov_pct | 2619 | 31 | 0.1807 | 0.1837 | -0.0030 | UNDERPOWERED | corrected |
| tov_pct | 2636 | 29 | 0.1680 | 0.1554 | 0.0126 | UNDERPOWERED | current |
| tov_pct | 2636 | 29 | 0.1681 | 0.1554 | 0.0127 | UNDERPOWERED | corrected |
| oreb_pct | 93 | 32 | 0.2580 | 0.2897 | -0.0317 | UNDERPOWERED | current |
| oreb_pct | 93 | 32 | 0.2568 | 0.2897 | -0.0329 | UNDERPOWERED | corrected |
| oreb_pct | 155 | 30 | 0.3016 | 0.3337 | -0.0321 | UNDERPOWERED | current |
| oreb_pct | 155 | 30 | 0.3023 | 0.3337 | -0.0313 | UNDERPOWERED | corrected |
| oreb_pct | 2031 | 30 | 0.2770 | 0.3001 | -0.0231 | UNDERPOWERED | current |
| oreb_pct | 2031 | 30 | 0.2770 | 0.3001 | -0.0231 | UNDERPOWERED | corrected |
| oreb_pct | 2197 | 28 | 0.2692 | 0.3014 | -0.0322 | UNDERPOWERED | current |
| oreb_pct | 2197 | 28 | 0.2684 | 0.3014 | -0.0330 | UNDERPOWERED | corrected |
| oreb_pct | 2261 | 30 | 0.2933 | 0.3421 | -0.0488 | UNDERPOWERED | current |
| oreb_pct | 2261 | 30 | 0.2939 | 0.3421 | -0.0481 | UNDERPOWERED | corrected |
| oreb_pct | 2351 | 31 | 0.2416 | 0.2293 | 0.0124 | UNDERPOWERED | current |
| oreb_pct | 2351 | 31 | 0.2400 | 0.2293 | 0.0107 | UNDERPOWERED | corrected |
| oreb_pct | 2460 | 32 | 0.2336 | 0.2494 | -0.0158 | UNDERPOWERED | current |
| oreb_pct | 2460 | 32 | 0.2323 | 0.2494 | -0.0171 | UNDERPOWERED | corrected |
| oreb_pct | 2492 | 34 | 0.2511 | 0.2754 | -0.0243 | UNDERPOWERED | current |
| oreb_pct | 2492 | 34 | 0.2500 | 0.2754 | -0.0254 | UNDERPOWERED | corrected |
| oreb_pct | 2619 | 31 | 0.2586 | 0.2939 | -0.0353 | UNDERPOWERED | current |
| oreb_pct | 2619 | 31 | 0.2579 | 0.2939 | -0.0360 | UNDERPOWERED | corrected |
| oreb_pct | 2636 | 29 | 0.2571 | 0.2605 | -0.0035 | UNDERPOWERED | current |
| oreb_pct | 2636 | 29 | 0.2566 | 0.2605 | -0.0040 | UNDERPOWERED | corrected |
| ft_rate | 93 | 32 | 0.3520 | 0.4169 | -0.0648 | UNDERPOWERED | current |
| ft_rate | 93 | 32 | 0.3514 | 0.4169 | -0.0655 | UNDERPOWERED | corrected |
| ft_rate | 155 | 30 | 0.3216 | 0.3425 | -0.0209 | UNDERPOWERED | current |
| ft_rate | 155 | 30 | 0.3231 | 0.3425 | -0.0194 | UNDERPOWERED | corrected |
| ft_rate | 2031 | 30 | 0.3355 | 0.2947 | 0.0408 | UNDERPOWERED | current |
| ft_rate | 2031 | 30 | 0.3351 | 0.2947 | 0.0404 | UNDERPOWERED | corrected |
| ft_rate | 2197 | 28 | 0.3351 | 0.3349 | 0.0002 | UNDERPOWERED | current |
| ft_rate | 2197 | 28 | 0.3342 | 0.3349 | -0.0007 | UNDERPOWERED | corrected |
| ft_rate | 2261 | 30 | 0.3022 | 0.3096 | -0.0074 | UNDERPOWERED | current |
| ft_rate | 2261 | 30 | 0.3004 | 0.3096 | -0.0092 | UNDERPOWERED | corrected |
| ft_rate | 2351 | 31 | 0.2972 | 0.2957 | 0.0015 | UNDERPOWERED | current |
| ft_rate | 2351 | 31 | 0.2971 | 0.2957 | 0.0015 | UNDERPOWERED | corrected |
| ft_rate | 2460 | 32 | 0.3291 | 0.3535 | -0.0244 | UNDERPOWERED | current |
| ft_rate | 2460 | 32 | 0.3276 | 0.3535 | -0.0258 | UNDERPOWERED | corrected |
| ft_rate | 2492 | 34 | 0.3373 | 0.3233 | 0.0140 | UNDERPOWERED | current |
| ft_rate | 2492 | 34 | 0.3371 | 0.3233 | 0.0138 | UNDERPOWERED | corrected |
| ft_rate | 2619 | 31 | 0.2981 | 0.3213 | -0.0232 | UNDERPOWERED | current |
| ft_rate | 2619 | 31 | 0.2970 | 0.3213 | -0.0243 | UNDERPOWERED | corrected |
| ft_rate | 2636 | 29 | 0.3151 | 0.2945 | 0.0205 | UNDERPOWERED | current |
| ft_rate | 2636 | 29 | 0.3116 | 0.2945 | 0.0170 | UNDERPOWERED | corrected |
| efg_pct | 93 | 32 | 0.5103 | 0.4985 | 0.0118 | UNDERPOWERED | current |
| efg_pct | 93 | 32 | 0.5113 | 0.4985 | 0.0128 | UNDERPOWERED | corrected |
| efg_pct | 155 | 30 | 0.5004 | 0.4979 | 0.0025 | UNDERPOWERED | current |
| efg_pct | 155 | 30 | 0.5015 | 0.4979 | 0.0036 | UNDERPOWERED | corrected |
| efg_pct | 2031 | 30 | 0.4924 | 0.4970 | -0.0047 | UNDERPOWERED | current |
| efg_pct | 2031 | 30 | 0.4935 | 0.4970 | -0.0035 | UNDERPOWERED | corrected |
| efg_pct | 2197 | 28 | 0.4645 | 0.4428 | 0.0217 | UNDERPOWERED | current |
| efg_pct | 2197 | 28 | 0.4650 | 0.4428 | 0.0222 | UNDERPOWERED | corrected |
| efg_pct | 2261 | 30 | 0.4602 | 0.4782 | -0.0180 | UNDERPOWERED | current |
| efg_pct | 2261 | 30 | 0.4607 | 0.4782 | -0.0175 | UNDERPOWERED | corrected |
| efg_pct | 2351 | 31 | 0.4819 | 0.5019 | -0.0200 | UNDERPOWERED | current |
| efg_pct | 2351 | 31 | 0.4827 | 0.5019 | -0.0192 | UNDERPOWERED | corrected |
| efg_pct | 2460 | 32 | 0.5232 | 0.5522 | -0.0290 | UNDERPOWERED | current |
| efg_pct | 2460 | 32 | 0.5246 | 0.5522 | -0.0275 | UNDERPOWERED | corrected |
| efg_pct | 2492 | 34 | 0.4894 | 0.4968 | -0.0074 | UNDERPOWERED | current |
| efg_pct | 2492 | 34 | 0.4900 | 0.4968 | -0.0068 | UNDERPOWERED | corrected |
| efg_pct | 2619 | 31 | 0.4733 | 0.4792 | -0.0059 | UNDERPOWERED | current |
| efg_pct | 2619 | 31 | 0.4736 | 0.4792 | -0.0056 | UNDERPOWERED | corrected |
| efg_pct | 2636 | 29 | 0.4903 | 0.4983 | -0.0080 | UNDERPOWERED | current |
| efg_pct | 2636 | 29 | 0.4909 | 0.4983 | -0.0074 | UNDERPOWERED | corrected |

### G5 -- Dispersion: SD ratio, score correlation, PIT histogram

| quantity | current value | corrected value | target (cur / corr) | tolerance | status cur | status corr | changed |
|---|---|---|---|---|---|---|---|
| margin SD ratio | 1.0396 | 1.0391 | 1.0 / 1.0 | 0.95-1.05 | PASS | PASS | value |
| total SD ratio | 0.8983 | 0.9233 | 1.0 / 1.0 | 0.95-1.05 | FAIL | FAIL | value |
| home/away score correlation | 0.1170 vs 0.2532 | 0.1169 vs 0.2283 | 0.2532 / 0.2283 | +/-0.05 | FAIL | FAIL | value |
| PIT K-S p | 0.016 | 0.0144 | > 0.1 / > 0.1 | > 0.1 | FAIL | FAIL | value |
| **gate overall** | | | | | **FAIL** | **FAIL** |  |

table `SD ratio`: 2/2 rows change (current then corrected)

| quantity | mean_sim_SD | SD(actual - sim mean) | ratio | status | truth |
|---|---|---|---|---|---|
| margin | 12.2815 | 11.8140 | 1.0396 | PASS | current |
| margin | 12.2818 | 11.8192 | 1.0391 | PASS | corrected |
| total | 15.8917 | 17.6912 | 0.8983 | FAIL | current |
| total | 15.8920 | 17.2131 | 0.9233 | FAIL | corrected |

table `PIT decile histogram`: 10/10 rows change (current then corrected)

| decile | share | truth |
|---|---|---|
| 1 | 0.0856 | current |
| 1 | 0.0847 | corrected |
| 2 | 0.0968 | current |
| 2 | 0.0983 | corrected |
| 3 | 0.1007 | current |
| 3 | 0.1013 | corrected |
| 4 | 0.1047 | current |
| 4 | 0.1046 | corrected |
| 5 | 0.1093 | current |
| 5 | 0.1078 | corrected |
| 6 | 0.1156 | current |
| 6 | 0.1152 | corrected |
| 7 | 0.0986 | current |
| 7 | 0.0989 | corrected |
| 8 | 0.0989 | current |
| 8 | 0.1008 | corrected |
| 9 | 0.0968 | current |
| 9 | 0.0955 | corrected |
| 10 | 0.0928 | current |
| 10 | 0.0929 | corrected |

### G6 -- Home margin, non-neutral vs neutral, same games

| quantity | current value | corrected value | target (cur / corr) | tolerance | status cur | status corr | changed |
|---|---|---|---|---|---|---|---|
| home margin (non-neutral) | +5.691 vs +5.738 | +5.695 vs +5.743 | +5.738 / +5.743 | +/-1.0 | PASS | PASS | value |
| home margin (neutral) | +2.105 vs +3.288 | +2.105 vs +3.288 | +3.288 / +3.288 | +/-1.0 | FAIL | FAIL |  |
| **gate overall** | | | | | **FAIL** | **FAIL** |  |

table `by site`: 1/2 rows change (current then corrected)

| site | n | sim | actual | delta | status | truth |
|---|---|---|---|---|---|---|
| non-neutral | 4974 | 5.6915 | 5.7382 | -0.0467 | PASS | current |
| non-neutral | 4969 | 5.6947 | 5.7432 | -0.0485 | PASS | corrected |

### G7 -- Overtime rate; first-half vs second-half scoring share

| quantity | current value | corrected value | target (cur / corr) | tolerance | status cur | status corr | changed |
|---|---|---|---|---|---|---|---|
| OT rate | 0.0304 vs 0.0557 | 0.0304 vs 0.0557 | 0.0557 / 0.0557 | +/-1.0pp | FAIL | FAIL |  |
| first/second half scoring share | n/a (contract has no per-period sim score) | n/a (contract has no per-period sim score) | 1H 0.4731 / 2H 0.5269 / 1H 0.4731 / 2H 0.5269 | +/-1.0pp | NEEDS-INSTRUMENTATION | NEEDS-INSTRUMENTATION |  |
| **gate overall** | | | | | **FAIL** | **FAIL** |  |

### G8 -- Player layer: minutes, usage share, distribution tails

| quantity | current value | corrected value | target (cur / corr) | tolerance | status cur | status corr | changed |
|---|---|---|---|---|---|---|---|
| rotation minutes mean | 30.57 vs 29.83 | 30.57 vs 29.83 | 29.83 / 29.83 | +/-2.0 | PASS | PASS |  |
| rotation minutes SD ratio | 1.2258 | 1.2258 | 1.0 / 1.0 | 0.9-1.1 | FAIL | FAIL |  |
| top-1 FGA share, mean | 0.2575 vs 0.2496 | 0.2575 vs 0.2496 | 0.2496 / 0.2496 | report only (no pre-registered tolerance) | NEEDS-INSTRUMENTATION | NEEDS-INSTRUMENTATION |  |
| players used per team-game, mean | 8.79 vs 9.80 | 8.79 vs 9.80 | 9.80 / 9.80 | K-S p > 0.1 | FAIL | FAIL |  |
| **gate overall** | | | | | **FAIL** | **FAIL** |  |

### G9 -- Spread and total accuracy: MAE, signed bias, calibration slope

| quantity | current value | corrected value | target (cur / corr) | tolerance | status cur | status corr | changed |
|---|---|---|---|---|---|---|---|
| margin bias | -0.1932 | -0.1949 | 0 / 0 | +/-0.5 | PASS | PASS | value |
| total bias | -0.8617 | -0.9836 | 0 / 0 | +/-1.0 | PASS | PASS | value |
| calibration slope | 0.9096 | 0.9096 | 1.0 / 1.0 | 0.95-1.05 | FAIL | FAIL |  |
| bias by month | 4/10 scored cells outside tolerance | 5/10 scored cells outside tolerance | all inside / all inside | see per-breakdown table | FAIL | FAIL | value |
| bias by tier | 3/6 scored cells outside tolerance | 4/6 scored cells outside tolerance | all inside / all inside | see per-breakdown table | FAIL | FAIL | value |
| bias by pred_total_tercile | 3/6 scored cells outside tolerance | 3/6 scored cells outside tolerance | all inside / all inside | see per-breakdown table | FAIL | FAIL |  |
| **gate overall** | | | | | **FAIL** | **FAIL** |  |

table `win-probability calibration by decile`: 5/10 rows change (current then corrected)

| decile | n | pred | actual | delta | truth |
|---|---|---|---|---|---|
| 3 | 575 | 0.4679 | 0.4991 | 0.0312 | current |
| 3 | 574 | 0.4679 | 0.5000 | 0.0321 | corrected |
| 4 | 572 | 0.5433 | 0.5787 | 0.0354 | current |
| 4 | 570 | 0.5434 | 0.5807 | 0.0373 | corrected |
| 5 | 585 | 0.6111 | 0.6256 | 0.0145 | current |
| 5 | 584 | 0.6111 | 0.6267 | 0.0156 | corrected |
| 6 | 554 | 0.6782 | 0.7040 | 0.0258 | current |
| 6 | 553 | 0.6782 | 0.7052 | 0.0270 | corrected |
| 7 | 593 | 0.7413 | 0.7454 | 0.0041 | current |
| 7 | 593 | 0.7413 | 0.7437 | 0.0024 | corrected |

table `by month`: 4/6 rows change (current then corrected)

| month | n | margin_mae | margin_bias | total_mae | total_bias | slope | status_margin_bias | status_total_bias | truth |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1423 | 9.0233 | 0.9197 | 13.7490 | 0.2677 | 0.8972 | FAIL | PASS | current |
| 1 | 1422 | 9.0270 | 0.9177 | 13.6614 | 0.1706 | 0.8973 | FAIL | PASS | corrected |
| 2 | 1367 | 8.7312 | -0.1832 | 13.4521 | -0.2825 | 0.8884 | PASS | PASS | current |
| 2 | 1365 | 8.7461 | -0.1813 | 13.2666 | -0.4882 | 0.8881 | PASS | PASS | corrected |
| 11 | 1222 | 10.4111 | -1.0824 | 14.3864 | -2.9727 | 0.8640 | FAIL | FAIL | current |
| 11 | 1221 | 10.4147 | -1.0883 | 14.2854 | -3.0880 | 0.8639 | FAIL | FAIL | corrected |
| 12 | 916 | 9.1864 | -0.1693 | 12.8084 | -0.9670 | 0.9176 | PASS | PASS | current |
| 12 | 915 | 9.1945 | -0.1714 | 12.6753 | -1.1152 | 0.9175 | PASS | FAIL | corrected |

table `by tier`: 2/3 rows change (current then corrected)

| home_tier | n | margin_mae | margin_bias | total_mae | total_bias | slope | status_margin_bias | status_total_bias | truth |
|---|---|---|---|---|---|---|---|---|---|
| bottom_tercile | 1607 | 9.0770 | 1.0360 | 13.4181 | -0.9070 | 0.8043 | FAIL | PASS | current |
| bottom_tercile | 1605 | 9.0823 | 1.0312 | 13.2628 | -1.0802 | 0.8047 | FAIL | FAIL | corrected |
| middle_tercile | 1833 | 9.0501 | -0.0069 | 14.1112 | 0.1142 | 0.7665 | PASS | PASS | current |
| middle_tercile | 1830 | 9.0656 | -0.0063 | 13.9077 | -0.1123 | 0.7662 | PASS | PASS | corrected |

table `by pred_total_tercile`: 3/3 rows change (current then corrected)

| pred_total_tercile | n | margin_mae | margin_bias | total_mae | total_bias | slope | status_margin_bias | status_total_bias | truth |
|---|---|---|---|---|---|---|---|---|---|
| bottom_tercile | 1905 | 9.1650 | -0.6717 | 13.4460 | -1.3829 | 0.8992 | FAIL | FAIL | current |
| bottom_tercile | 1902 | 9.1795 | -0.6727 | 13.1798 | -1.6738 | 0.8993 | FAIL | FAIL | corrected |
| middle_tercile | 1902 | 9.1508 | -0.0113 | 13.3574 | -1.0507 | 0.9012 | PASS | FAIL | current |
| middle_tercile | 1901 | 9.1519 | -0.0150 | 13.3015 | -1.1129 | 0.9009 | PASS | FAIL | corrected |
| top_tercile | 1903 | 9.4399 | 0.1039 | 13.9614 | -0.1511 | 0.9291 | PASS | PASS | current |
| top_tercile | 1902 | 9.4440 | 0.1031 | 13.9557 | -0.1642 | 0.9292 | PASS | PASS | corrected |

