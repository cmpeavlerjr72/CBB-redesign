# Gate noise band -- A=`results/engine_v0/F2_2025_s200_v1_clockv3c_A` (200 seeds) vs B=`results/engine_v0/F2_2025_s200_v1_clockv3c_B` (200 seeds)

B is a spec-identical retrain under a different seed offset (the noise floor), never a second candidate. |A-B| on a line is the seed-noise band for that metric at this seed count -- a FAIL smaller than its own band is read differently than one that is not (CLAUDE.md "backtests must be honest" / "bake-off before any choice").

| gate | quantity | A (headline) | B (noise floor) | \|A-B\| | A status |
|---|---|---|---|---|---|
| G1 | possessions/game mean | 69.877 vs 67.875 | 69.877 vs 67.875 | 0.0000 | FAIL |
| G1 | possessions/game SD | 4.571 vs 5.474 | 4.569 vs 5.474 | 0.0020 | FAIL |
| G1 | by month (mean and SD) | 0/5 powered months inside | 0/5 powered months inside | n/a | FAIL |
| G2 | PPP by offense x defense tercile (9 cells) | 2/9 powered cells inside +/-0.02 | 2/9 powered cells inside +/-0.02 | 0.0000 | FAIL |
| G3 | 3PA share & FTA/FGA by team | 0/0 powered team-metrics inside | 0/0 powered team-metrics inside | n/a | NEEDS-INSTRUMENTATION |
| G3 | rim share by team | 0/0 powered teams inside | 0/0 powered teams inside | n/a | NEEDS-INSTRUMENTATION |
| G3 | three_pa_share (season, pooled, PROVISIONAL) | 0.3869 vs 0.3906 | 0.3868 vs 0.3906 | 0.0001 | PASS |
| G3 | fta_per_fga (season, pooled, PROVISIONAL) | 0.3168 vs 0.3295 | 0.3169 vs 0.3295 | 0.0001 | PASS |
| G3 | rim_share (season, pooled, PROVISIONAL) | 0.3715 vs 0.3733 | 0.3715 vs 0.3733 | 0.0000 | PASS |
| G4 | eFG% (offense/defense) | 0/0 powered teams inside | 0/0 powered teams inside | n/a | NEEDS-INSTRUMENTATION |
| G4 | tov_pct by team | 0/0 powered teams inside | 0/0 powered teams inside | n/a | NEEDS-INSTRUMENTATION |
| G4 | oreb_pct by team | 0/0 powered teams inside | 0/0 powered teams inside | n/a | NEEDS-INSTRUMENTATION |
| G4 | ft_rate by team | 0/0 powered teams inside | 0/0 powered teams inside | n/a | NEEDS-INSTRUMENTATION |
| G4 | tov_pct (season, pooled, PROVISIONAL) | 0.1756 vs 0.1739 | 0.1755 vs 0.1739 | 0.0001 | PASS |
| G4 | oreb_pct (season, pooled, PROVISIONAL) | 0.2829 vs 0.2984 | 0.2830 vs 0.2984 | 0.0001 | FAIL |
| G4 | ft_rate (season, pooled, PROVISIONAL) | 0.3168 vs 0.3295 | 0.3169 vs 0.3295 | 0.0001 | PASS |
| G4 | efg_pct (season, pooled, PROVISIONAL) | 0.4974 vs 0.5086 | 0.4975 vs 0.5086 | 0.0001 | FAIL |
| G5 | margin SD ratio | 1.0373 | 1.0379 | 0.0006 | PASS |
| G5 | total SD ratio | 0.8030 | 0.8035 | 0.0005 | FAIL |
| G5 | home/away score correlation | 0.0273 vs 0.2532 | 0.0284 vs 0.2532 | 0.0011 | FAIL |
| G5 | PIT K-S p | 0.0168 | 0.0166 | 0.0002 | FAIL |
| G6 | home margin (non-neutral) | +5.681 vs +5.738 | +5.704 vs +5.738 | 0.0230 | PASS |
| G6 | home margin (neutral) | +2.108 vs +3.288 | +2.035 vs +3.288 | 0.0730 | FAIL |
| G7 | OT rate | 0.0303 vs 0.0557 | 0.0306 vs 0.0557 | 0.0003 | FAIL |
| G7 | first/second half scoring share | n/a (contract has no per-period sim score) | n/a (contract has no per-period sim score) | n/a | NEEDS-INSTRUMENTATION |
| G8 | rotation minutes mean | 30.57 vs 29.83 | 30.57 vs 29.83 | 0.0000 | PASS |
| G8 | rotation minutes SD ratio | 1.2259 | 1.2265 | 0.0006 | FAIL |
| G8 | top-1 FGA share, mean | 0.2574 vs 0.2496 | 0.2574 vs 0.2496 | 0.0000 | NEEDS-INSTRUMENTATION |
| G8 | players used per team-game, mean | 8.79 vs 9.80 | 8.79 vs 9.80 | 0.0000 | FAIL |
| G9 | margin bias | -0.2018 | -0.1909 | 0.0109 | PASS |
| G9 | total bias | -0.8517 | -0.8151 | 0.0366 | PASS |
| G9 | calibration slope | 0.9086 | 0.9095 | 0.0009 | FAIL |
| G9 | bias by month | 4/10 scored cells outside tolerance | 4/10 scored cells outside tolerance | n/a | FAIL |
| G9 | bias by tier | 3/6 scored cells outside tolerance | 3/6 scored cells outside tolerance | n/a | FAIL |
| G9 | bias by pred_total_tercile | 1/6 scored cells outside tolerance | 3/6 scored cells outside tolerance | n/a | FAIL |
