| gate | line | v2 inputs (po4b_R_s25) | v3 replay inputs | B-A | v2, seeds 1000-1024 | N-A (floor) | verdict |
|---|---|---|---|---:|---|---:|---|
| G1 | possessions/game mean | 70.019 vs 68.328 [FAIL] | 69.918 vs 68.328 [FAIL] | -0.1010 | 69.920 vs 68.328 | -0.0990 | **MOVED** |
| G1 | possessions/game SD | 5.609 vs 5.191 [PASS] | 5.609 vs 5.191 [PASS] | +0.0000 | 5.552 vs 5.191 | -0.0570 | identical |
| G1 | by month (mean and SD) | 0/0 powered months inside [NEEDS-INSTRUMENTATION] | 0/0 powered months inside [NEEDS-INSTRUMENTATION] | +0.0000 | 0/0 powered months inside | +0.0000 | identical |
| G2 | PPP by offense x defense tercile (9 cells) | 2/9 powered cells inside +/-0.02 [FAIL] | 3/9 powered cells inside +/-0.02 [FAIL] | +1.0000 | 2/9 powered cells inside +/-0.02 | +0.0000 | **MOVED** |
| G3 | 3PA share & FTA/FGA by team | 0/0 powered team-metrics inside [NEEDS-INSTRUMENTATION] | 0/0 powered team-metrics inside [NEEDS-INSTRUMENTATION] | +0.0000 | 0/0 powered team-metrics inside | +0.0000 | identical |
| G3 | rim share by team | 0/0 powered teams inside [NEEDS-INSTRUMENTATION] | 0/0 powered teams inside [NEEDS-INSTRUMENTATION] | +0.0000 | 0/0 powered teams inside | +0.0000 | identical |
| G3 | three_pa_share (season, pooled, PROVISIONAL) | 0.3894 vs 0.3906 [PASS] | 0.3897 vs 0.3906 [PASS] | +0.0003 | 0.3894 vs 0.3906 | +0.0000 | **MOVED** |
| G3 | fta_per_fga (season, pooled, PROVISIONAL) | 0.3195 vs 0.3295 [PASS] | 0.3185 vs 0.3295 [PASS] | -0.0010 | 0.3177 vs 0.3295 | -0.0018 | inside floor |
| G3 | rim_share (season, pooled, PROVISIONAL) | 0.3721 vs 0.3733 [PASS] | 0.3718 vs 0.3733 [PASS] | -0.0003 | 0.3714 vs 0.3733 | -0.0007 | inside floor |
| G4 | eFG% (offense/defense) | 0/0 powered teams inside [NEEDS-INSTRUMENTATION] | 0/0 powered teams inside [NEEDS-INSTRUMENTATION] | +0.0000 | 0/0 powered teams inside | +0.0000 | identical |
| G4 | tov_pct by team | 0/0 powered teams inside [NEEDS-INSTRUMENTATION] | 0/0 powered teams inside [NEEDS-INSTRUMENTATION] | +0.0000 | 0/0 powered teams inside | +0.0000 | identical |
| G4 | oreb_pct by team | 0/0 powered teams inside [NEEDS-INSTRUMENTATION] | 0/0 powered teams inside [NEEDS-INSTRUMENTATION] | +0.0000 | 0/0 powered teams inside | +0.0000 | identical |
| G4 | ft_rate by team | 0/0 powered teams inside [NEEDS-INSTRUMENTATION] | 0/0 powered teams inside [NEEDS-INSTRUMENTATION] | +0.0000 | 0/0 powered teams inside | +0.0000 | identical |
| G4 | tov_pct (season, pooled, PROVISIONAL) | 0.1762 vs 0.1739 [PASS] | 0.1762 vs 0.1739 [PASS] | +0.0000 | 0.1758 vs 0.1739 | -0.0004 | identical |
| G4 | oreb_pct (season, pooled, PROVISIONAL) | 0.2836 vs 0.2984 [FAIL] | 0.2841 vs 0.2984 [FAIL] | +0.0005 | 0.2834 vs 0.2984 | -0.0002 | **MOVED** |
| G4 | ft_rate (season, pooled, PROVISIONAL) | 0.3195 vs 0.3295 [PASS] | 0.3185 vs 0.3295 [PASS] | -0.0010 | 0.3177 vs 0.3295 | -0.0018 | inside floor |
| G4 | efg_pct (season, pooled, PROVISIONAL) | 0.4990 vs 0.5086 [PASS] | 0.5029 vs 0.5086 [PASS] | +0.0039 | 0.4984 vs 0.5086 | -0.0006 | **MOVED** |
| G5 | margin SD ratio | 0.9695 [PASS] | 0.9711 [PASS] | +0.0016 | 0.9560 | -0.0135 | inside floor |
| G5 | total SD ratio | 0.8355 [FAIL] | 0.8286 [FAIL] | -0.0069 | 0.8346 | -0.0009 | **MOVED** |
| G5 | home/away score correlation | 0.1063 vs 0.2374 [FAIL] | 0.1023 vs 0.2374 [FAIL] | -0.0040 | 0.0948 vs 0.2374 | -0.0115 | inside floor |
| G5 | PIT K-S p | 0.455 [PASS] | 0.29 [PASS] | -0.1650 | 0.455 | +0.0000 | **MOVED** |
| G5 | margin | 12.2264 [PASS] | 12.1434 [PASS] | -0.0830 | 12.1758 | -0.0506 | **MOVED** |
| G5 | total | 15.8421 [FAIL] | 15.7992 [FAIL] | -0.0429 | 15.8302 | -0.0119 | **MOVED** |
| G6 | home margin (non-neutral) | +5.989 vs +5.669 [PASS] | +6.225 vs +5.669 [PASS] | +0.2360 | +5.761 vs +5.669 | -0.2280 | **MOVED** |
| G6 | home margin (neutral) | +0.506 vs +2.151 [UNDERPOWERED] | +0.490 vs +2.151 [UNDERPOWERED] | -0.0160 | +0.343 vs +2.151 | -0.1630 | inside floor |
| G7 | OT rate | 0.0312 vs 0.0680 [FAIL] | 0.0311 vs 0.0680 [FAIL] | -0.0001 | 0.0274 vs 0.0680 | -0.0038 | inside floor |
| G7 | first/second half scoring share | n/a (contract has no per-period sim score) [NEEDS-INSTRUMENTATION] | n/a (contract has no per-period sim score) [NEEDS-INSTRUMENTATION] | -- | n/a (contract has no per-period sim score) | -- | -- |
| G8 | rotation minutes mean | 30.47 vs 29.82 [PASS] | 30.47 vs 29.75 [PASS] | +0.0000 | 30.47 vs 29.82 | +0.0000 | identical |
| G8 | rotation minutes SD ratio | 1.2304 [FAIL] | 1.2353 [FAIL] | +0.0049 | 1.2301 | -0.0003 | **MOVED** |
| G8 | top-1 FGA share, mean | 0.2574 vs 0.2476 [NEEDS-INSTRUMENTATION] | 0.2572 vs 0.2473 [NEEDS-INSTRUMENTATION] | -0.0002 | 0.2574 vs 0.2476 | +0.0000 | **MOVED** |
| G8 | players used per team-game, mean | 8.83 vs 9.83 [FAIL] | 8.82 vs 9.85 [FAIL] | -0.0100 | 8.81 vs 9.83 | -0.0200 | inside floor |
| G9 | margin bias | +0.1117 [PASS] | +0.3212 [PASS] | +0.2095 | -0.1092 | -0.2209 | inside floor |
| G9 | total bias | -0.9624 [PASS] | -0.2972 [PASS] | +0.6652 | -1.3158 | -0.3534 | **MOVED** |
| G9 | calibration slope | 0.8920 [FAIL] | 0.8858 [FAIL] | -0.0062 | 0.8535 | -0.0385 | inside floor |
| G9 | bias by month | 0/0 scored cells outside tolerance [NEEDS-INSTRUMENTATION] | 0/0 scored cells outside tolerance [NEEDS-INSTRUMENTATION] | +0.0000 | 0/0 scored cells outside tolerance | +0.0000 | identical |
| G9 | bias by tier | 0/0 scored cells outside tolerance [NEEDS-INSTRUMENTATION] | 0/0 scored cells outside tolerance [NEEDS-INSTRUMENTATION] | +0.0000 | 0/0 scored cells outside tolerance | +0.0000 | identical |
| G9 | bias by pred_total_tercile | 0/0 scored cells outside tolerance [NEEDS-INSTRUMENTATION] | 0/0 scored cells outside tolerance [NEEDS-INSTRUMENTATION] | +0.0000 | 0/0 scored cells outside tolerance | +0.0000 | identical |

Gate-level verdicts: G1 FAIL/FAIL/FAIL, G2 FAIL/FAIL/FAIL, G3 NEEDS-INSTRUMENTATION/NEEDS-INSTRUMENTATION/NEEDS-INSTRUMENTATION, G4 FAIL/FAIL/FAIL, G5 FAIL/FAIL/FAIL, G6 PASS/PASS/PASS, G7 FAIL/FAIL/FAIL, G8 FAIL/FAIL/FAIL, G9 FAIL/FAIL/FAIL
