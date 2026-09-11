
## 7. Round-3 results (train 2024, test 2025) -- run 2026-09-10T22:58:58Z

Test universe: random subset of 12 of 4689 eligible games, numpy RandomState seed 2025 -- the same game set the graded round-2 run used, 1 seeds per arm; noise floor A 2 seeds x 8 games, noise floor B a spec-identical refit under a second seed. Rotation player-games per arm 174 (actual 175); every cell is far above the n < 300 UNDERPOWERED threshold unless flagged.

### 7.1 G8 cells

| cell | tol | ACTUAL | R2_hier_dirichlet | R5_hybrid | R8_keep_cell | R7_keep_logistic |
|---|---|---:|---:|---:|---:|---:|
| minutes mean (rotation players) | +/- 2.0 | 24.67 | 24.33  PASS | 26.15  PASS | 24.66  PASS | 26.63  PASS |
| minutes SD ratio, pooled | 0.9-1.1 | 1.000 | 1.110  FAIL | 1.097  PASS | 1.127  FAIL | 1.193  FAIL |
| minutes SD ratio, within-player | 0.9-1.1 | 1.000 | nan  FAIL | nan  FAIL | nan  FAIL | nan  FAIL |
| top-5 share of team minutes | +/- 2 pp | 0.7393 | 0.7519  PASS | 0.7914  FAIL | 0.7709  FAIL | 0.8187  FAIL |
| top-8 share of team minutes | +/- 2 pp | 0.9620 | 0.9494  PASS | 0.9781  PASS | 0.9723  PASS | 0.9911  FAIL |
| players with > 0 minutes | +/- 1.0 | 9.58 | 9.54  PASS | 8.42  FAIL | 8.79  PASS | 8.21  FAIL |

### 7.2 State-dependence cells (an arm missing ANY of these is ineligible)

| cell | ACTUAL | R2_hier_dirichlet | R5_hybrid | R8_keep_cell | R7_keep_logistic |
|---|---:|---:|---:|---:|---:|
| final 8:00 starters' share, \|margin\| <= 5 (n=222 poss) | 0.7441 | 0.6910 (-5.3 pp)  FAIL | 0.7099 (-3.4 pp)  FAIL | 0.7910 (+4.7 pp)  FAIL | 0.7523 (+0.8 pp)  PASS |
| final 8:00 starters' share, \|margin\| 6-15 (n=176 poss) | 0.7636 | 0.6875 (-7.6 pp)  FAIL | 0.6625 (-10.1 pp)  FAIL | 0.6648 (-9.9 pp)  FAIL | 0.6386 (-12.5 pp)  FAIL |
| final 8:00 starters' share, \|margin\| > 15 (n=268 poss) | 0.4761 | 0.4418 (-3.4 pp)  FAIL | 0.5590 (+8.3 pp)  FAIL | 0.4366 (-4.0 pp)  FAIL | 0.5425 (+6.6 pp)  FAIL |
| starters on floor while carrying >= 4 fouls (n=307) | 0.4560 | 0.4122 (-4.4 pp)  FAIL | 0.5061 (+5.0 pp)  FAIL | 0.4497 (-0.6 pp)  PASS | 0.5186 (+6.3 pp)  FAIL |
| (diagnostic) at exactly 4 fouls (n=250) | 0.5600 | 0.5270 | 0.7930 | 0.5763 | 0.8345 |

As-of starter set overlaps the real starting five on **4.33 of 5**; the ACTUAL sequence re-graded with the MODEL's starter set separates "wrong five" from "wrong rotation":

| cell | ACTUAL (own starters) | ACTUAL (as-of starter set) | R2_hier_dirichlet | R5_hybrid | R8_keep_cell | R7_keep_logistic |
|---|---:|---:|---:|---:|---:|---:|
| final 8:00, \|margin\| <= 5 | 0.7441 | 0.7360 | 0.6910 | 0.7099 | 0.7910 | 0.7523 |
| final 8:00, \|margin\| 6-15 | 0.7636 | 0.6875 | 0.6875 | 0.6625 | 0.6648 | 0.6386 |
| final 8:00, \|margin\| > 15 | 0.4761 | 0.4493 | 0.4418 | 0.5590 | 0.4366 | 0.5425 |
| >= 4 fouls | 0.4560 | 0.4897 | 0.4122 | 0.5061 | 0.4497 | 0.5186 |

### 7.3 Lineup concentration

| metric | ACTUAL | R2_hier_dirichlet | R5_hybrid | R8_keep_cell | R7_keep_logistic |
|---|---:|---:|---:|---:|---:|
| top-1 lineup share of possessions | 0.2685 | 0.2344 | 0.2980 | 0.3018 | 0.3618 |
| top-3 lineup share | 0.5537 | 0.4571 | 0.5885 | 0.5809 | 0.6551 |
| top-5 lineup share | 0.7083 | 0.6054 | 0.7380 | 0.7369 | 0.8062 |
| distinct lineups per team-game | 14.38 | 16.71 | 13.46 | 14.00 | 11.12 |
| K-S of per-player minutes (D) | -- | 0.1058 | 0.1228 | 0.0767 | 0.1649 |
| K-S of the top-1 lineup share distribution (D) | -- | 0.3750 | 0.1667 | 0.2083 | 0.2917 |
| substitution rate at a possession boundary | 0.1518 (train) | 0.1476 | 0.1470 | 0.1545 | 0.1440 |

### 7.4 Noise floor A (2 seeds x 8 games)

| metric | R2_hier_dirichlet | R5_hybrid | R8_keep_cell | R7_keep_logistic |
|---|---:|---:|---:|---:|
| minutes_mean | 0.28199 | 0.62543 | 0.54567 | 0.28921 |
| top5_share | 0.00123 | 0.01933 | 0.01012 | 0.00967 |
| n_nonzero_mean | 0.04419 | 0.13258 | 0.22097 | 0.22097 |
| lu_top1 | 0.00499 | 0.01955 | 0.00201 | 0.01258 |
| late_starter_share_b0 | 0.04687 | 0.01069 | 0.03618 | 0.03124 |
| late_starter_share_b1 | 0.02073 | 0.00122 | 0.06461 | 0.07193 |
| late_starter_share_b2 | 0.00295 | 0.05500 | 0.01866 | 0.05991 |
| foul_trouble_share | 0.13424 | 0.08372 | 0.02435 | 0.07054 |

### 7.5 Noise floor B -- spec-identical refit under a second seed

Both fits use the same specification; the second draws a different training-game sample (fit seed 101 vs 11), a different logistic `random_state`, and a different sim seed inside the knob grid (23 vs 7). Graded on the 8-game noise universe so the two fits are compared on identical games.

Refit knobs: {"r7_block": [0.5, 0.06], "r7_keep_scale": 0.0, "r7_keep_base": 0.02, "r8_block": [0.5, 0.06], "r8_keep_theta": 0.0}

| cell | ACTUAL | R7 seed 1 | R7 seed 2 | |delta| pp | R8 seed 1 | R8 seed 2 | |delta| pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| late_starter_share_b0 | 0.7151 | 0.7372 | 0.6826 | 5.5 | 0.7779 | 0.6826 | 9.5 |
| late_starter_share_b1 | 0.7845 | 0.6810 | 0.7517 | 7.1 | 0.7052 | 0.7121 | 0.7 |
| late_starter_share_b2 | 0.6486 | 0.5792 | 0.6750 | 9.6 | 0.4056 | 0.6736 | 26.8 |
| foul_trouble_share | 0.5126 | 0.5841 | 0.5104 | 7.4 | 0.4564 | 0.5104 | 5.4 |
| top5_share | 0.7480 | 0.8129 | 0.8017 | 1.1 | 0.7505 | 0.7973 | 4.7 |
| top8_share | 0.9717 | 0.9905 | 0.9845 | 0.6 | 0.9693 | 0.9844 | 1.5 |

### 7.6 Slope check -- team quintile of the as-of starter-minutes share

Cell = starters' share of on-floor slots in the final 8:00 at |margin| <= 5.

| quintile | team-games | prior starter share | ACTUAL | R2_hier_dirichlet | R5_hybrid | R8_keep_cell | R7_keep_logistic | close-late possessions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 5 | 0.5451 | 0.6586 | 0.6172 | 0.6138 | 0.7345 | 0.7724 | 58 UNDERPOWERED |
| Q2 | 5 | 0.6375 | nan | nan | nan | nan | nan | 0 UNDERPOWERED |
| Q3 | 4 | 0.6707 | 0.7951 | 0.6683 | 0.7171 | 0.8220 | 0.8146 | 82 UNDERPOWERED |
| Q4 | 5 | 0.6976 | 0.8000 | 0.7429 | 0.7429 | 0.7429 | 0.7429 | 7 UNDERPOWERED |
| Q5 | 5 | 0.7640 | 0.7493 | 0.7680 | 0.7733 | 0.8053 | 0.6693 | 75 UNDERPOWERED |

- `ACTUAL` slope vs the prior: **+nan**, Q5 - Q1 = +9.1 pp
- `R2_hier_dirichlet` slope vs the prior: **+nan**, Q5 - Q1 = +15.1 pp
- `R5_hybrid` slope vs the prior: **+nan**, Q5 - Q1 = +16.0 pp
- `R8_keep_cell` slope vs the prior: **+nan**, Q5 - Q1 = +7.1 pp
- `R7_keep_logistic` slope vs the prior: **+nan**, Q5 - Q1 = -10.3 pp

### 7.7 Fitted round-3 components

R7's on-floor propensity (logit scale; positive = more likely on the floor):

| feature | coefficient |
|---|---:|
| `fouls` | +0.3377 |
| `fouls_x_is_starter` | -0.2803 |
| `foul_out` | -3.3193 |
| `is_starter` | +0.7665 |
| `abs_margin` | +0.2983 |
| `abs_margin_x_is_starter` | -0.5840 |
| `late` | -0.5185 |
| `late_x_is_starter` | +0.5773 |
| `is_close_x_late` | -0.0525 |
| `is_close_x_late_x_is_starter` | -0.1695 |
| `abs_margin_x_late` | -0.1553 |
| `target_share` | +4.7153 |
| `target_share_x_is_close_x_late` | +0.1839 |
| `target_share_x_late` | +0.8768 |
| `period2` | +0.0253 |
| `sec_left_frac` | +0.3204 |
| _intercept_ | -3.1888 |

Fitted on 36,165 (team-game, possession, candidate) rows, base rate 0.3312.


R7 knobs: block (scale 1.0, p0 0.06), keep (scale 4.0, q0 0.02). R8 knobs: block (scale 4.0, p0 0.02), theta 0.0.

R8's fitted `s*` table (training-season starters' share of on-floor slots):

| time bucket | \|m\| <= 5 | \|m\| 6-15 | \|m\| > 15 |
|---|---:|---:|---:|
| 1st half | 0.7446 | 0.6570 | 0.6306 |
| 2nd half > 8:00 | 0.7471 | 0.7361 | 0.6654 |
| 2nd half 8:00-2:00 | 0.7484 | 0.7323 | 0.5715 |
| final 2:00 | 0.7559 | 0.7043 | 0.3148 |
| OT | 0.7567 | 0.7145 | -- |

R7 knob grid (state-cell squared error; every point evaluated):

| pass | point | late b0 | late b1 | late b2 | >= 4 fouls | sub rate | sq. err |
|---|---|---:|---:|---:|---:|---:|---:|
| pass1_keep | keep=(0.0, 0.02) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass1_keep | keep=(0.0, 0.05) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass1_keep | keep=(0.0, 0.15) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1143 | 0.047774 |
| pass1_keep | keep=(0.0, 0.35) | 0.7000 | 0.6113 | 0.6000 | 0.6095 | 0.1000 | 0.046240 |
| pass1_keep | keep=(0.5, 0.02) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass1_keep | keep=(0.5, 0.05) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1179 | 0.047774 |
| pass1_keep | keep=(0.5, 0.15) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1143 | 0.047774 |
| pass1_keep | keep=(0.5, 0.35) | 0.7000 | 0.6113 | 0.6000 | 0.6095 | 0.1036 | 0.046240 |
| pass1_keep | keep=(1.0, 0.02) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass1_keep | keep=(1.0, 0.05) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1143 | 0.047774 |
| pass1_keep | keep=(1.0, 0.15) | 0.7000 | 0.6208 | 0.6000 | 0.6237 | 0.1125 | 0.045763 |
| pass1_keep | keep=(1.0, 0.35) | 0.7000 | 0.6113 | 0.6000 | 0.6095 | 0.1071 | 0.046240 |
| pass1_keep | keep=(2.0, 0.02) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass1_keep | keep=(2.0, 0.05) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1125 | 0.047774 |
| pass1_keep | keep=(2.0, 0.15) | 0.8000 | 0.6679 | 0.6000 | 0.6344 | 0.1161 | 0.052679 |
| pass1_keep | keep=(2.0, 0.35) | 0.8000 | 0.6660 | 0.6000 | 0.8609 | 0.1232 | 0.125870 |
| pass1_keep | keep=(4.0, 0.02) | 0.7000 | 0.6132 | 0.6000 | 0.5699 | 0.1143 | 0.045681 |
| pass1_keep | keep=(4.0, 0.05) | 0.7000 | 0.6226 | 0.6000 | 0.6344 | 0.1196 | 0.046439 |
| pass1_keep | keep=(4.0, 0.15) | 0.6000 | 0.5830 | 0.6000 | 0.7778 | 0.1250 | 0.097468 |
| pass1_keep | keep=(4.0, 0.35) | 0.7000 | 0.6151 | 0.6000 | 0.7778 | 0.1286 | 0.081840 |
| pass1_keep | keep=(8.0, 0.02) | 0.6000 | 0.5830 | 0.6000 | 0.7778 | 0.1250 | 0.097468 |
| pass1_keep | keep=(8.0, 0.05) | 0.6000 | 0.6075 | 0.6000 | 0.7778 | 0.1250 | 0.092979 |
| pass1_keep | keep=(8.0, 0.15) | 0.8000 | 0.6453 | 0.6000 | 0.6456 | 0.1321 | 0.055247 |
| pass1_keep | keep=(8.0, 0.35) | 0.7000 | 0.6264 | 0.6000 | 0.6456 | 0.1375 | 0.047169 |
| pass1_block | block_scale=0.5,p0=0.005 | 0.7000 | 0.6113 | 0.6000 | 0.5608 | 0.1125 | 0.046341 |
| pass1_block | block_scale=0.5,p0=0.02 | 0.7000 | 0.6113 | 0.6000 | 0.5608 | 0.1125 | 0.046341 |
| pass1_block | block_scale=0.5,p0=0.06 | 0.7000 | 0.6113 | 0.6000 | 0.5608 | 0.1125 | 0.046341 |
| pass1_block | block_scale=1.0,p0=0.005 | 0.7000 | 0.6113 | 0.6000 | 0.5608 | 0.1125 | 0.046341 |
| pass1_block | block_scale=1.0,p0=0.02 | 0.7000 | 0.6113 | 0.6000 | 0.5608 | 0.1125 | 0.046341 |
| pass1_block | block_scale=1.0,p0=0.06 | 0.7000 | 0.6132 | 0.6000 | 0.5699 | 0.1143 | 0.045681 |
| pass1_block | block_scale=2.0,p0=0.005 | 0.7000 | 0.6132 | 0.6000 | 0.5699 | 0.1125 | 0.045681 |
| pass1_block | block_scale=2.0,p0=0.02 | 0.7000 | 0.6208 | 0.6000 | 0.6857 | 0.1125 | 0.054263 |
| pass1_block | block_scale=2.0,p0=0.06 | 0.7000 | 0.5981 | 0.6000 | 0.6857 | 0.1107 | 0.057766 |
| pass1_block | block_scale=4.0,p0=0.005 | 0.7000 | 0.6208 | 0.6000 | 0.6857 | 0.1143 | 0.054263 |
| pass1_block | block_scale=4.0,p0=0.02 | 0.7000 | 0.5981 | 0.6000 | 0.6333 | 0.1161 | 0.050085 |
| pass1_block | block_scale=4.0,p0=0.06 | 0.7000 | 0.5792 | 0.6000 | 0.7937 | 0.1179 | 0.094599 |
| pass2_keep | keep=(0.0, 0.02) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass2_keep | keep=(0.0, 0.05) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass2_keep | keep=(0.0, 0.15) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1143 | 0.047774 |
| pass2_keep | keep=(0.0, 0.35) | 0.7000 | 0.6113 | 0.6000 | 0.6095 | 0.1000 | 0.046240 |
| pass2_keep | keep=(0.5, 0.02) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass2_keep | keep=(0.5, 0.05) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1179 | 0.047774 |
| pass2_keep | keep=(0.5, 0.15) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1143 | 0.047774 |
| pass2_keep | keep=(0.5, 0.35) | 0.7000 | 0.6113 | 0.6000 | 0.6095 | 0.1036 | 0.046240 |
| pass2_keep | keep=(1.0, 0.02) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass2_keep | keep=(1.0, 0.05) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1143 | 0.047774 |
| pass2_keep | keep=(1.0, 0.15) | 0.7000 | 0.6208 | 0.6000 | 0.6237 | 0.1125 | 0.045763 |
| pass2_keep | keep=(1.0, 0.35) | 0.7000 | 0.6113 | 0.6000 | 0.6095 | 0.1071 | 0.046240 |
| pass2_keep | keep=(2.0, 0.02) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass2_keep | keep=(2.0, 0.05) | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1125 | 0.047774 |
| pass2_keep | keep=(2.0, 0.15) | 0.8000 | 0.6679 | 0.6000 | 0.6344 | 0.1161 | 0.052679 |
| pass2_keep | keep=(2.0, 0.35) | 0.8000 | 0.6660 | 0.6000 | 0.8609 | 0.1232 | 0.125870 |
| pass2_keep | keep=(4.0, 0.02) | 0.7000 | 0.6132 | 0.6000 | 0.5699 | 0.1143 | 0.045681 |
| pass2_keep | keep=(4.0, 0.05) | 0.7000 | 0.6226 | 0.6000 | 0.6344 | 0.1196 | 0.046439 |
| pass2_keep | keep=(4.0, 0.15) | 0.6000 | 0.5830 | 0.6000 | 0.7778 | 0.1250 | 0.097468 |
| pass2_keep | keep=(4.0, 0.35) | 0.7000 | 0.6151 | 0.6000 | 0.7778 | 0.1286 | 0.081840 |
| pass2_keep | keep=(8.0, 0.02) | 0.6000 | 0.5830 | 0.6000 | 0.7778 | 0.1250 | 0.097468 |
| pass2_keep | keep=(8.0, 0.05) | 0.6000 | 0.6075 | 0.6000 | 0.7778 | 0.1250 | 0.092979 |
| pass2_keep | keep=(8.0, 0.15) | 0.8000 | 0.6453 | 0.6000 | 0.6456 | 0.1321 | 0.055247 |
| pass2_keep | keep=(8.0, 0.35) | 0.7000 | 0.6264 | 0.6000 | 0.6456 | 0.1375 | 0.047169 |
| pass2_block | block_scale=0.5,p0=0.005 | 0.7000 | 0.6113 | 0.6000 | 0.5608 | 0.1125 | 0.046341 |
| pass2_block | block_scale=0.5,p0=0.02 | 0.7000 | 0.6113 | 0.6000 | 0.5608 | 0.1125 | 0.046341 |
| pass2_block | block_scale=0.5,p0=0.06 | 0.7000 | 0.6113 | 0.6000 | 0.5608 | 0.1125 | 0.046341 |
| pass2_block | block_scale=1.0,p0=0.005 | 0.7000 | 0.6113 | 0.6000 | 0.5608 | 0.1125 | 0.046341 |
| pass2_block | block_scale=1.0,p0=0.02 | 0.7000 | 0.6113 | 0.6000 | 0.5608 | 0.1125 | 0.046341 |
| pass2_block | block_scale=1.0,p0=0.06 | 0.7000 | 0.6132 | 0.6000 | 0.5699 | 0.1143 | 0.045681 |
| pass2_block | block_scale=2.0,p0=0.005 | 0.7000 | 0.6132 | 0.6000 | 0.5699 | 0.1125 | 0.045681 |
| pass2_block | block_scale=2.0,p0=0.02 | 0.7000 | 0.6208 | 0.6000 | 0.6857 | 0.1125 | 0.054263 |
| pass2_block | block_scale=2.0,p0=0.06 | 0.7000 | 0.5981 | 0.6000 | 0.6857 | 0.1107 | 0.057766 |
| pass2_block | block_scale=4.0,p0=0.005 | 0.7000 | 0.6208 | 0.6000 | 0.6857 | 0.1143 | 0.054263 |
| pass2_block | block_scale=4.0,p0=0.02 | 0.7000 | 0.5981 | 0.6000 | 0.6333 | 0.1161 | 0.050085 |
| pass2_block | block_scale=4.0,p0=0.06 | 0.7000 | 0.5792 | 0.6000 | 0.7937 | 0.1179 | 0.094599 |

R8 knob grid (state-cell squared error; every point evaluated):

| pass | point | late b0 | late b1 | late b2 | >= 4 fouls | sub rate | sq. err |
|---|---|---:|---:|---:|---:|---:|---:|
| pass1_keep | keep=0.0 | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass1_keep | keep=0.1 | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass1_keep | keep=0.2 | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass1_keep | keep=0.3 | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass1_keep | keep=0.4 | 0.8000 | 0.6321 | 0.6000 | 0.6344 | 0.1161 | 0.055317 |
| pass1_keep | keep=0.5 | 0.8000 | 0.6189 | 0.6000 | 0.6095 | 0.1179 | 0.055157 |
| pass1_keep | keep=0.6 | 0.8000 | 0.6189 | 0.6000 | 0.6095 | 0.1179 | 0.055157 |
| pass1_keep | keep=0.7 | 0.8000 | 0.6189 | 0.6000 | 0.6095 | 0.1179 | 0.055157 |
| pass1_keep | keep=0.8 | 0.8000 | 0.6189 | 0.6000 | 0.6095 | 0.1179 | 0.055157 |
| pass1_keep | keep=0.9 | 0.8000 | 0.6132 | 0.6000 | 0.3833 | 0.1107 | 0.096572 |
| pass1_keep | keep=1.0 | 0.8000 | 0.6094 | 0.6000 | 0.3737 | 0.1179 | 0.101150 |
| pass1_block | block_scale=0.5,p0=0.005 | 0.7000 | 0.6113 | 0.6000 | 0.5405 | 0.1196 | 0.047781 |
| pass1_block | block_scale=0.5,p0=0.02 | 0.7000 | 0.6113 | 0.6000 | 0.5405 | 0.1196 | 0.047781 |
| pass1_block | block_scale=0.5,p0=0.06 | 0.7000 | 0.6113 | 0.6000 | 0.5405 | 0.1196 | 0.047781 |
| pass1_block | block_scale=1.0,p0=0.005 | 0.7000 | 0.6113 | 0.6000 | 0.5405 | 0.1196 | 0.047781 |
| pass1_block | block_scale=1.0,p0=0.02 | 0.7000 | 0.6113 | 0.6000 | 0.5405 | 0.1196 | 0.047781 |
| pass1_block | block_scale=1.0,p0=0.06 | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass1_block | block_scale=2.0,p0=0.005 | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1196 | 0.047774 |
| pass1_block | block_scale=2.0,p0=0.02 | 0.7000 | 0.6208 | 0.6000 | 0.6857 | 0.1196 | 0.054263 |
| pass1_block | block_scale=2.0,p0=0.06 | 0.7000 | 0.6208 | 0.6000 | 0.6857 | 0.1179 | 0.054263 |
| pass1_block | block_scale=4.0,p0=0.005 | 0.7000 | 0.6208 | 0.6000 | 0.6857 | 0.1214 | 0.054263 |
| pass1_block | block_scale=4.0,p0=0.02 | 0.7000 | 0.6208 | 0.6000 | 0.6333 | 0.1250 | 0.046582 |
| pass1_block | block_scale=4.0,p0=0.06 | 0.7000 | 0.5830 | 0.6000 | 0.7937 | 0.1321 | 0.093802 |
| pass2_keep | keep=0.0 | 0.7000 | 0.6208 | 0.6000 | 0.6333 | 0.1250 | 0.046582 |
| pass2_keep | keep=0.1 | 0.7000 | 0.6208 | 0.6000 | 0.6333 | 0.1250 | 0.046582 |
| pass2_keep | keep=0.2 | 0.7000 | 0.6208 | 0.6000 | 0.6333 | 0.1250 | 0.046582 |
| pass2_keep | keep=0.3 | 0.7000 | 0.6208 | 0.6000 | 0.6333 | 0.1250 | 0.046582 |
| pass2_keep | keep=0.4 | 0.8000 | 0.6396 | 0.6000 | 0.6333 | 0.1214 | 0.054446 |
| pass2_keep | keep=0.5 | 0.8000 | 0.6736 | 0.6000 | 0.6333 | 0.1250 | 0.052395 |
| pass2_keep | keep=0.6 | 0.8000 | 0.6736 | 0.6000 | 0.6333 | 0.1250 | 0.052395 |
| pass2_keep | keep=0.7 | 0.8000 | 0.6736 | 0.6000 | 0.6333 | 0.1250 | 0.052395 |
| pass2_keep | keep=0.8 | 0.8000 | 0.6736 | 0.6000 | 0.6333 | 0.1250 | 0.052395 |
| pass2_keep | keep=0.9 | 0.8000 | 0.6868 | 0.6000 | 0.6235 | 0.1232 | 0.051393 |
| pass2_keep | keep=1.0 | 0.8000 | 0.7283 | 0.6000 | 0.8421 | 0.1304 | 0.117207 |
| pass2_block | block_scale=0.5,p0=0.005 | 0.7000 | 0.6113 | 0.6000 | 0.5405 | 0.1196 | 0.047781 |
| pass2_block | block_scale=0.5,p0=0.02 | 0.7000 | 0.6113 | 0.6000 | 0.5405 | 0.1196 | 0.047781 |
| pass2_block | block_scale=0.5,p0=0.06 | 0.7000 | 0.6113 | 0.6000 | 0.5405 | 0.1196 | 0.047781 |
| pass2_block | block_scale=1.0,p0=0.005 | 0.7000 | 0.6113 | 0.6000 | 0.5405 | 0.1196 | 0.047781 |
| pass2_block | block_scale=1.0,p0=0.02 | 0.7000 | 0.6113 | 0.6000 | 0.5405 | 0.1196 | 0.047781 |
| pass2_block | block_scale=1.0,p0=0.06 | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1214 | 0.047774 |
| pass2_block | block_scale=2.0,p0=0.005 | 0.7000 | 0.6132 | 0.6000 | 0.5376 | 0.1196 | 0.047774 |
| pass2_block | block_scale=2.0,p0=0.02 | 0.7000 | 0.6208 | 0.6000 | 0.6857 | 0.1196 | 0.054263 |
| pass2_block | block_scale=2.0,p0=0.06 | 0.7000 | 0.6208 | 0.6000 | 0.6857 | 0.1179 | 0.054263 |
| pass2_block | block_scale=4.0,p0=0.005 | 0.7000 | 0.6208 | 0.6000 | 0.6857 | 0.1214 | 0.054263 |
| pass2_block | block_scale=4.0,p0=0.02 | 0.7000 | 0.6208 | 0.6000 | 0.6333 | 0.1250 | 0.046582 |
| pass2_block | block_scale=4.0,p0=0.06 | 0.7000 | 0.5830 | 0.6000 | 0.7937 | 0.1321 | 0.093802 |

### 7.8 Decision

| arm | G8 cells | state cells | total | eligible (all 4 state cells) | lineup K-S D | simplicity |
|---|---:|---:|---:|---|---:|---:|
| R8_keep_cell | 3/6 | 1/4 | 4 | NO | 0.2083 | 3 |
| R2_hier_dirichlet | 4/6 | 0/4 | 4 | NO | 0.3750 | 1 |
| R5_hybrid | 3/6 | 0/4 | 3 | NO | 0.1667 | 2 |
| R7_keep_logistic | 1/6 | 1/4 | 2 | NO | 0.2917 | 4 |

**No arm adopted.** no arm has every state-dependence cell inside +/- 3 pp. Cell-by-cell misses:

| arm | cell | sim | actual | miss |
|---|---|---:|---:|---:|
| R2_hier_dirichlet | late_starter_share_b0 | 0.6910 | 0.7441 | -5.3 pp |
| R2_hier_dirichlet | late_starter_share_b1 | 0.6875 | 0.7636 | -7.6 pp |
| R2_hier_dirichlet | late_starter_share_b2 | 0.4418 | 0.4761 | -3.4 pp |
| R2_hier_dirichlet | foul_trouble_share | 0.4122 | 0.4560 | -4.4 pp |
| R5_hybrid | late_starter_share_b0 | 0.7099 | 0.7441 | -3.4 pp |
| R5_hybrid | late_starter_share_b1 | 0.6625 | 0.7636 | -10.1 pp |
| R5_hybrid | late_starter_share_b2 | 0.5590 | 0.4761 | +8.3 pp |
| R5_hybrid | foul_trouble_share | 0.5061 | 0.4560 | +5.0 pp |
| R8_keep_cell | late_starter_share_b0 | 0.7910 | 0.7441 | +4.7 pp |
| R8_keep_cell | late_starter_share_b1 | 0.6648 | 0.7636 | -9.9 pp |
| R8_keep_cell | late_starter_share_b2 | 0.4366 | 0.4761 | -4.0 pp |
| R7_keep_logistic | late_starter_share_b1 | 0.6386 | 0.7636 | -12.5 pp |
| R7_keep_logistic | late_starter_share_b2 | 0.5425 | 0.4761 | +6.6 pp |
| R7_keep_logistic | foul_trouble_share | 0.5186 | 0.4560 | +6.3 pp |

