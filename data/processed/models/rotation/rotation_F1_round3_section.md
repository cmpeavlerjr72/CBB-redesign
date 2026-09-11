
## 7. Round-3 results (train 2024, test 2025) -- run 2026-09-10T23:45:11Z

Test universe: random subset of 1600 of 4689 eligible games, numpy RandomState seed 2025 -- the same game set the graded round-2 run used, 5 seeds per arm; noise floor A 20 seeds x 150 games, noise floor B a spec-identical refit under a second seed. Rotation player-games per arm 115,702 (actual 23,914); every cell is far above the n < 300 UNDERPOWERED threshold unless flagged.

### 7.1 G8 cells

| cell | tol | ACTUAL | R2_hier_dirichlet | R5_hybrid | R8_keep_cell | R7_keep_logistic |
|---|---|---:|---:|---:|---:|---:|
| minutes mean (rotation players) | +/- 2.0 | 24.55 | 25.55  PASS | 27.06  FAIL | 27.17  FAIL | 26.28  PASS |
| minutes SD ratio, pooled | 0.9-1.1 | 1.000 | 1.064  PASS | 1.039  PASS | 1.069  PASS | 1.022  PASS |
| minutes SD ratio, within-player | 0.9-1.1 | 1.000 | 1.354  FAIL | 1.294  FAIL | 1.330  FAIL | 1.253  FAIL |
| top-5 share of team minutes | +/- 2 pp | 0.7472 | 0.7683  FAIL | 0.7965  FAIL | 0.8036  FAIL | 0.7828  FAIL |
| top-8 share of team minutes | +/- 2 pp | 0.9560 | 0.9656  PASS | 0.9847  FAIL | 0.9863  FAIL | 0.9814  FAIL |
| players with > 0 minutes | +/- 1.0 | 9.64 | 9.04  PASS | 8.30  FAIL | 8.27  FAIL | 8.51  FAIL |

### 7.2 State-dependence cells (an arm missing ANY of these is ineligible)

| cell | ACTUAL | R2_hier_dirichlet | R5_hybrid | R8_keep_cell | R7_keep_logistic |
|---|---:|---:|---:|---:|---:|
| final 8:00 starters' share, \|margin\| <= 5 (n=29,660 poss) | 0.7491 | 0.7281 (-2.1 pp)  PASS | 0.6942 (-5.5 pp)  FAIL | 0.6982 (-5.1 pp)  FAIL | 0.7074 (-4.2 pp)  FAIL |
| final 8:00 starters' share, \|margin\| 6-15 (n=36,804 poss) | 0.7240 | 0.6943 (-3.0 pp)  PASS | 0.6566 (-6.7 pp)  FAIL | 0.6599 (-6.4 pp)  FAIL | 0.6749 (-4.9 pp)  FAIL |
| final 8:00 starters' share, \|margin\| > 15 (n=23,360 poss) | 0.5223 | 0.4750 (-4.7 pp)  FAIL | 0.5911 (+6.9 pp)  FAIL | 0.5956 (+7.3 pp)  FAIL | 0.5994 (+7.7 pp)  FAIL |
| starters on floor while carrying >= 4 fouls (n=58,652) | 0.4613 | 0.4648 (+0.4 pp)  PASS | 0.4964 (+3.5 pp)  FAIL | 0.4997 (+3.8 pp)  FAIL | 0.5125 (+5.1 pp)  FAIL |
| (diagnostic) at exactly 4 fouls (n=51,745) | 0.5166 | 0.5777 | 0.8199 | 0.8336 | 0.7951 |

As-of starter set overlaps the real starting five on **4.58 of 5**; the ACTUAL sequence re-graded with the MODEL's starter set separates "wrong five" from "wrong rotation":

| cell | ACTUAL (own starters) | ACTUAL (as-of starter set) | R2_hier_dirichlet | R5_hybrid | R8_keep_cell | R7_keep_logistic |
|---|---:|---:|---:|---:|---:|---:|
| final 8:00, \|margin\| <= 5 | 0.7491 | 0.7215 | 0.7281 | 0.6942 | 0.6982 | 0.7074 |
| final 8:00, \|margin\| 6-15 | 0.7240 | 0.6964 | 0.6943 | 0.6566 | 0.6599 | 0.6749 |
| final 8:00, \|margin\| > 15 | 0.5223 | 0.5048 | 0.4750 | 0.5911 | 0.5956 | 0.5994 |
| >= 4 fouls | 0.4613 | 0.4636 | 0.4648 | 0.4964 | 0.4997 | 0.5125 |

### 7.3 Lineup concentration

| metric | ACTUAL | R2_hier_dirichlet | R5_hybrid | R8_keep_cell | R7_keep_logistic |
|---|---:|---:|---:|---:|---:|
| top-1 lineup share of possessions | 0.2940 | 0.2369 | 0.3159 | 0.3256 | 0.2888 |
| top-3 lineup share | 0.5426 | 0.4957 | 0.6085 | 0.6230 | 0.5726 |
| top-5 lineup share | 0.6894 | 0.6555 | 0.7593 | 0.7732 | 0.7280 |
| distinct lineups per team-game | 14.84 | 15.01 | 12.90 | 12.42 | 14.05 |
| K-S of per-player minutes (D) | -- | 0.0883 | 0.1194 | 0.1349 | 0.0782 |
| K-S of the top-1 lineup share distribution (D) | -- | 0.1942 | 0.1047 | 0.1334 | 0.0302 |
| substitution rate at a possession boundary | 0.1518 (train) | 0.1402 | 0.1576 | 0.1547 | 0.1691 |

### 7.4 Noise floor A (20 seeds x 150 games)

| metric | R2_hier_dirichlet | R5_hybrid | R8_keep_cell | R7_keep_logistic |
|---|---:|---:|---:|---:|
| minutes_mean | 0.15436 | 0.21802 | 0.21758 | 0.17947 |
| top5_share | 0.00311 | 0.00320 | 0.00335 | 0.00293 |
| n_nonzero_mean | 0.05792 | 0.06813 | 0.06875 | 0.06448 |
| lu_top1 | 0.00563 | 0.00650 | 0.00678 | 0.00518 |
| late_starter_share_b0 | 0.00882 | 0.01054 | 0.01054 | 0.01102 |
| late_starter_share_b1 | 0.01059 | 0.01112 | 0.01207 | 0.00882 |
| late_starter_share_b2 | 0.01356 | 0.01383 | 0.01517 | 0.01211 |
| foul_trouble_share | 0.02138 | 0.02013 | 0.01826 | 0.02427 |

### 7.5 Noise floor B -- spec-identical refit under a second seed

Both fits use the same specification; the second draws a different training-game sample (fit seed 101 vs 11), a different logistic `random_state`, and a different sim seed inside the knob grid (23 vs 7). Graded on the 150-game noise universe so the two fits are compared on identical games.

Refit knobs: {"r7_block": [2.0, 0.02], "r7_keep_scale": 2.0, "r7_keep_base": 0.05, "r8_block": [4.0, 0.02], "r8_keep_theta": 0.3}

| cell | ACTUAL | R7 seed 1 | R7 seed 2 | |delta| pp | R8 seed 1 | R8 seed 2 | |delta| pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| late_starter_share_b0 | 0.7436 | 0.7038 | 0.6890 | 1.5 | 0.6940 | 0.7547 | 6.1 |
| late_starter_share_b1 | 0.7186 | 0.6767 | 0.6631 | 1.4 | 0.6615 | 0.6842 | 2.3 |
| late_starter_share_b2 | 0.5518 | 0.6120 | 0.5792 | 3.3 | 0.6030 | 0.5674 | 3.6 |
| foul_trouble_share | 0.4362 | 0.5164 | 0.5202 | 0.4 | 0.5066 | 0.5485 | 4.2 |
| top5_share | 0.7450 | 0.7838 | 0.7959 | 1.2 | 0.8053 | 0.7736 | 3.2 |
| top8_share | 0.9568 | 0.9824 | 0.9853 | 0.3 | 0.9873 | 0.9800 | 0.7 |

### 7.6 Slope check -- team quintile of the as-of starter-minutes share

Cell = starters' share of on-floor slots in the final 8:00 at |margin| <= 5.

| quintile | team-games | prior starter share | ACTUAL | R2_hier_dirichlet | R5_hybrid | R8_keep_cell | R7_keep_logistic | close-late possessions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 640 | 0.5572 | 0.6770 | 0.6506 | 0.5995 | 0.6046 | 0.6054 | 5,600 |
| Q2 | 640 | 0.6342 | 0.7226 | 0.7129 | 0.6780 | 0.6833 | 0.6873 | 6,238 |
| Q3 | 640 | 0.6698 | 0.7471 | 0.7303 | 0.7073 | 0.7113 | 0.7177 | 6,138 |
| Q4 | 640 | 0.7044 | 0.7763 | 0.7586 | 0.7274 | 0.7302 | 0.7446 | 5,898 |
| Q5 | 640 | 0.7685 | 0.8219 | 0.7859 | 0.7554 | 0.7582 | 0.7788 | 5,786 |

- `ACTUAL` slope vs the prior: **+0.692**, Q5 - Q1 = +14.5 pp
- `R2_hier_dirichlet` slope vs the prior: **+0.645**, Q5 - Q1 = +13.5 pp
- `R5_hybrid` slope vs the prior: **+0.742**, Q5 - Q1 = +15.6 pp
- `R8_keep_cell` slope vs the prior: **+0.728**, Q5 - Q1 = +15.4 pp
- `R7_keep_logistic` slope vs the prior: **+0.826**, Q5 - Q1 = +17.3 pp

### 7.7 Fitted round-3 components

R7's on-floor propensity (logit scale; positive = more likely on the floor):

| feature | coefficient |
|---|---:|
| `fouls` | +0.4434 |
| `fouls_x_is_starter` | -0.5909 |
| `foul_out` | -4.6622 |
| `is_starter` | +1.0886 |
| `abs_margin` | +0.2534 |
| `abs_margin_x_is_starter` | -0.5351 |
| `late` | +0.1434 |
| `late_x_is_starter` | +0.6912 |
| `is_close_x_late` | -0.6290 |
| `is_close_x_late_x_is_starter` | -0.0931 |
| `abs_margin_x_late` | -0.0313 |
| `target_share` | +4.8826 |
| `target_share_x_is_close_x_late` | +1.4707 |
| `target_share_x_late` | -0.9167 |
| `period2` | +0.0043 |
| `sec_left_frac` | +0.1255 |
| _intercept_ | -3.2368 |

Fitted on 1,081,245 (team-game, possession, candidate) rows, base rate 0.3322.


R7 knobs: block (scale 2.0, p0 0.02), keep (scale 1.0, q0 0.02). R8 knobs: block (scale 1.0, p0 0.06), theta 0.1.

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
| pass1_keep | keep=(0.0, 0.02) | 0.6950 | 0.6546 | 0.5791 | 0.4510 | 0.1729 | 0.007059 |
| pass1_keep | keep=(0.0, 0.05) | 0.6905 | 0.6537 | 0.5776 | 0.4588 | 0.1697 | 0.007416 |
| pass1_keep | keep=(0.0, 0.15) | 0.6714 | 0.6392 | 0.5940 | 0.4778 | 0.1526 | 0.013722 |
| pass1_keep | keep=(0.0, 0.35) | 0.6476 | 0.6507 | 0.5776 | 0.4990 | 0.1128 | 0.014642 |
| pass1_keep | keep=(0.5, 0.02) | 0.6950 | 0.6567 | 0.5770 | 0.4549 | 0.1732 | 0.006701 |
| pass1_keep | keep=(0.5, 0.05) | 0.6888 | 0.6485 | 0.5770 | 0.4612 | 0.1705 | 0.008171 |
| pass1_keep | keep=(0.5, 0.15) | 0.6656 | 0.6376 | 0.5973 | 0.4733 | 0.1566 | 0.014604 |
| pass1_keep | keep=(0.5, 0.35) | 0.6452 | 0.6472 | 0.5666 | 0.5256 | 0.1252 | 0.018203 |
| pass1_keep | keep=(1.0, 0.02) | 0.6950 | 0.6558 | 0.5615 | 0.4541 | 0.1745 | 0.005289 |
| pass1_keep | keep=(1.0, 0.05) | 0.6852 | 0.6491 | 0.5719 | 0.4560 | 0.1700 | 0.007509 |
| pass1_keep | keep=(1.0, 0.15) | 0.6603 | 0.6308 | 0.5782 | 0.4570 | 0.1587 | 0.012759 |
| pass1_keep | keep=(1.0, 0.35) | 0.6498 | 0.6397 | 0.5331 | 0.5292 | 0.1379 | 0.017352 |
| pass1_keep | keep=(2.0, 0.02) | 0.6921 | 0.6489 | 0.5525 | 0.4521 | 0.1766 | 0.005505 |
| pass1_keep | keep=(2.0, 0.05) | 0.6834 | 0.6451 | 0.5651 | 0.4579 | 0.1706 | 0.007540 |
| pass1_keep | keep=(2.0, 0.15) | 0.6636 | 0.6349 | 0.5030 | 0.4439 | 0.1568 | 0.008543 |
| pass1_keep | keep=(2.0, 0.35) | 0.6405 | 0.6141 | 0.4719 | 0.4949 | 0.1499 | 0.020167 |
| pass1_keep | keep=(4.0, 0.02) | 0.6563 | 0.6327 | 0.5203 | 0.4367 | 0.1695 | 0.009282 |
| pass1_keep | keep=(4.0, 0.05) | 0.6645 | 0.6235 | 0.5021 | 0.4286 | 0.1708 | 0.010451 |
| pass1_keep | keep=(4.0, 0.15) | 0.6305 | 0.5870 | 0.4278 | 0.4301 | 0.1624 | 0.030914 |
| pass1_keep | keep=(4.0, 0.35) | 0.6207 | 0.5724 | 0.4024 | 0.4610 | 0.1671 | 0.042284 |
| pass1_keep | keep=(8.0, 0.02) | 0.6492 | 0.6035 | 0.4254 | 0.4504 | 0.1640 | 0.024924 |
| pass1_keep | keep=(8.0, 0.05) | 0.6296 | 0.5747 | 0.4027 | 0.4415 | 0.1693 | 0.039522 |
| pass1_keep | keep=(8.0, 0.15) | 0.6071 | 0.5392 | 0.3349 | 0.4455 | 0.1705 | 0.075412 |
| pass1_keep | keep=(8.0, 0.35) | 0.5895 | 0.4940 | 0.3221 | 0.4384 | 0.1913 | 0.101900 |
| pass1_block | block_scale=0.5,p0=0.005 | 0.6892 | 0.6624 | 0.5716 | 0.4420 | 0.1806 | 0.005490 |
| pass1_block | block_scale=0.5,p0=0.02 | 0.6792 | 0.6550 | 0.5803 | 0.4401 | 0.1795 | 0.007761 |
| pass1_block | block_scale=0.5,p0=0.06 | 0.6845 | 0.6516 | 0.5618 | 0.4422 | 0.1687 | 0.006007 |
| pass1_block | block_scale=1.0,p0=0.005 | 0.6892 | 0.6623 | 0.5716 | 0.4399 | 0.1812 | 0.005495 |
| pass1_block | block_scale=1.0,p0=0.02 | 0.6910 | 0.6573 | 0.5851 | 0.4569 | 0.1806 | 0.007810 |
| pass1_block | block_scale=1.0,p0=0.06 | 0.6950 | 0.6558 | 0.5615 | 0.4541 | 0.1745 | 0.005289 |
| pass1_block | block_scale=2.0,p0=0.005 | 0.7003 | 0.6658 | 0.5785 | 0.4488 | 0.1823 | 0.005645 |
| pass1_block | block_scale=2.0,p0=0.02 | 0.6986 | 0.6607 | 0.5654 | 0.4542 | 0.1806 | 0.004991 |
| pass1_block | block_scale=2.0,p0=0.06 | 0.7175 | 0.6327 | 0.5457 | 0.5096 | 0.1847 | 0.012000 |
| pass1_block | block_scale=4.0,p0=0.005 | 0.7184 | 0.6759 | 0.5597 | 0.5248 | 0.1910 | 0.010312 |
| pass1_block | block_scale=4.0,p0=0.02 | 0.7341 | 0.6504 | 0.4997 | 0.5095 | 0.1937 | 0.009851 |
| pass1_block | block_scale=4.0,p0=0.06 | 0.7522 | 0.5665 | 0.4161 | 0.5502 | 0.1988 | 0.046416 |
| pass2_keep | keep=(0.0, 0.02) | 0.6979 | 0.6593 | 0.5821 | 0.4496 | 0.1785 | 0.006788 |
| pass2_keep | keep=(0.0, 0.05) | 0.6934 | 0.6575 | 0.5806 | 0.4569 | 0.1752 | 0.007150 |
| pass2_keep | keep=(0.0, 0.15) | 0.6712 | 0.6427 | 0.6113 | 0.4946 | 0.1601 | 0.017675 |
| pass2_keep | keep=(0.0, 0.35) | 0.6461 | 0.6553 | 0.5928 | 0.5020 | 0.1168 | 0.016619 |
| pass2_keep | keep=(0.5, 0.02) | 0.6994 | 0.6627 | 0.5809 | 0.4565 | 0.1793 | 0.006452 |
| pass2_keep | keep=(0.5, 0.05) | 0.6903 | 0.6521 | 0.5797 | 0.4599 | 0.1769 | 0.007901 |
| pass2_keep | keep=(0.5, 0.15) | 0.6652 | 0.6361 | 0.6134 | 0.4856 | 0.1625 | 0.018607 |
| pass2_keep | keep=(0.5, 0.35) | 0.6305 | 0.6448 | 0.5699 | 0.5239 | 0.1329 | 0.020682 |
| pass2_keep | keep=(1.0, 0.02) | 0.6986 | 0.6607 | 0.5654 | 0.4542 | 0.1806 | 0.004991 |
| pass2_keep | keep=(1.0, 0.05) | 0.6881 | 0.6545 | 0.5758 | 0.4550 | 0.1749 | 0.007093 |
| pass2_keep | keep=(1.0, 0.15) | 0.6594 | 0.6308 | 0.5922 | 0.4740 | 0.1639 | 0.015572 |
| pass2_keep | keep=(1.0, 0.35) | 0.6385 | 0.6364 | 0.5412 | 0.5170 | 0.1450 | 0.017568 |
| pass2_keep | keep=(2.0, 0.02) | 0.6948 | 0.6534 | 0.5573 | 0.4462 | 0.1827 | 0.005066 |
| pass2_keep | keep=(2.0, 0.05) | 0.6808 | 0.6467 | 0.5746 | 0.4559 | 0.1749 | 0.008342 |
| pass2_keep | keep=(2.0, 0.15) | 0.6521 | 0.6288 | 0.5107 | 0.4617 | 0.1623 | 0.011057 |
| pass2_keep | keep=(2.0, 0.35) | 0.6303 | 0.6094 | 0.4845 | 0.5255 | 0.1546 | 0.025990 |
| pass2_keep | keep=(4.0, 0.02) | 0.6623 | 0.6322 | 0.5293 | 0.4362 | 0.1744 | 0.008803 |
| pass2_keep | keep=(4.0, 0.05) | 0.6650 | 0.6226 | 0.5140 | 0.4355 | 0.1768 | 0.010179 |
| pass2_keep | keep=(4.0, 0.15) | 0.6385 | 0.6024 | 0.4313 | 0.4664 | 0.1686 | 0.026136 |
| pass2_keep | keep=(4.0, 0.35) | 0.5991 | 0.5411 | 0.3809 | 0.4995 | 0.1682 | 0.065154 |
| pass2_keep | keep=(8.0, 0.02) | 0.6590 | 0.5946 | 0.4230 | 0.4606 | 0.1687 | 0.026633 |
| pass2_keep | keep=(8.0, 0.05) | 0.6452 | 0.5682 | 0.3982 | 0.4769 | 0.1756 | 0.041614 |
| pass2_keep | keep=(8.0, 0.15) | 0.5956 | 0.5134 | 0.3209 | 0.4896 | 0.1747 | 0.095507 |
| pass2_keep | keep=(8.0, 0.35) | 0.5835 | 0.4510 | 0.3078 | 0.4448 | 0.1940 | 0.129966 |
| pass2_block | block_scale=0.5,p0=0.005 | 0.6892 | 0.6624 | 0.5716 | 0.4420 | 0.1806 | 0.005490 |
| pass2_block | block_scale=0.5,p0=0.02 | 0.6792 | 0.6550 | 0.5803 | 0.4401 | 0.1795 | 0.007761 |
| pass2_block | block_scale=0.5,p0=0.06 | 0.6845 | 0.6516 | 0.5618 | 0.4422 | 0.1687 | 0.006007 |
| pass2_block | block_scale=1.0,p0=0.005 | 0.6892 | 0.6623 | 0.5716 | 0.4399 | 0.1812 | 0.005495 |
| pass2_block | block_scale=1.0,p0=0.02 | 0.6910 | 0.6573 | 0.5851 | 0.4569 | 0.1806 | 0.007810 |
| pass2_block | block_scale=1.0,p0=0.06 | 0.6950 | 0.6558 | 0.5615 | 0.4541 | 0.1745 | 0.005289 |
| pass2_block | block_scale=2.0,p0=0.005 | 0.7003 | 0.6658 | 0.5785 | 0.4488 | 0.1823 | 0.005645 |
| pass2_block | block_scale=2.0,p0=0.02 | 0.6986 | 0.6607 | 0.5654 | 0.4542 | 0.1806 | 0.004991 |
| pass2_block | block_scale=2.0,p0=0.06 | 0.7175 | 0.6327 | 0.5457 | 0.5096 | 0.1847 | 0.012000 |
| pass2_block | block_scale=4.0,p0=0.005 | 0.7184 | 0.6759 | 0.5597 | 0.5248 | 0.1910 | 0.010312 |
| pass2_block | block_scale=4.0,p0=0.02 | 0.7341 | 0.6504 | 0.4997 | 0.5095 | 0.1937 | 0.009851 |
| pass2_block | block_scale=4.0,p0=0.06 | 0.7522 | 0.5665 | 0.4161 | 0.5502 | 0.1988 | 0.046416 |

R8 knob grid (state-cell squared error; every point evaluated):

| pass | point | late b0 | late b1 | late b2 | >= 4 fouls | sub rate | sq. err |
|---|---|---:|---:|---:|---:|---:|---:|
| pass1_keep | keep=0.0 | 0.7028 | 0.6567 | 0.5794 | 0.4581 | 0.1745 | 0.006903 |
| pass1_keep | keep=0.1 | 0.7028 | 0.6586 | 0.5779 | 0.4593 | 0.1721 | 0.006571 |
| pass1_keep | keep=0.2 | 0.7057 | 0.6521 | 0.5913 | 0.4660 | 0.1681 | 0.009299 |
| pass1_keep | keep=0.3 | 0.7137 | 0.6548 | 0.5863 | 0.4759 | 0.1623 | 0.008920 |
| pass1_keep | keep=0.4 | 0.7137 | 0.6632 | 0.5887 | 0.4847 | 0.1536 | 0.009081 |
| pass1_keep | keep=0.5 | 0.7157 | 0.6640 | 0.5940 | 0.4806 | 0.1505 | 0.009409 |
| pass1_keep | keep=0.6 | 0.7206 | 0.6721 | 0.5910 | 0.4909 | 0.1466 | 0.009325 |
| pass1_keep | keep=0.7 | 0.7277 | 0.6770 | 0.5946 | 0.4955 | 0.1426 | 0.010161 |
| pass1_keep | keep=0.8 | 0.7417 | 0.6755 | 0.5913 | 0.4968 | 0.1327 | 0.010611 |
| pass1_keep | keep=0.9 | 0.7393 | 0.6680 | 0.5916 | 0.4825 | 0.1279 | 0.009647 |
| pass1_keep | keep=1.0 | 0.7475 | 0.6715 | 0.5940 | 0.4818 | 0.1205 | 0.010164 |
| pass1_block | block_scale=0.5,p0=0.005 | 0.6934 | 0.6637 | 0.5887 | 0.4458 | 0.1780 | 0.007247 |
| pass1_block | block_scale=0.5,p0=0.02 | 0.6834 | 0.6558 | 0.5964 | 0.4435 | 0.1767 | 0.009608 |
| pass1_block | block_scale=0.5,p0=0.06 | 0.6921 | 0.6524 | 0.5779 | 0.4474 | 0.1657 | 0.007204 |
| pass1_block | block_scale=1.0,p0=0.005 | 0.6934 | 0.6635 | 0.5887 | 0.4439 | 0.1785 | 0.007238 |
| pass1_block | block_scale=1.0,p0=0.02 | 0.6959 | 0.6591 | 0.6012 | 0.4654 | 0.1788 | 0.010159 |
| pass1_block | block_scale=1.0,p0=0.06 | 0.7028 | 0.6586 | 0.5779 | 0.4593 | 0.1721 | 0.006571 |
| pass1_block | block_scale=2.0,p0=0.005 | 0.7055 | 0.6685 | 0.5955 | 0.4507 | 0.1800 | 0.007602 |
| pass1_block | block_scale=2.0,p0=0.02 | 0.7037 | 0.6632 | 0.5833 | 0.4623 | 0.1788 | 0.006861 |
| pass1_block | block_scale=2.0,p0=0.06 | 0.7253 | 0.6348 | 0.5663 | 0.5228 | 0.1814 | 0.015317 |
| pass1_block | block_scale=4.0,p0=0.005 | 0.7235 | 0.6756 | 0.5839 | 0.5345 | 0.1900 | 0.014658 |
| pass1_block | block_scale=4.0,p0=0.02 | 0.7424 | 0.6583 | 0.5164 | 0.5139 | 0.1923 | 0.009589 |
| pass1_block | block_scale=4.0,p0=0.06 | 0.7615 | 0.5658 | 0.4188 | 0.5715 | 0.1954 | 0.052111 |
| pass2_keep | keep=0.0 | 0.7028 | 0.6567 | 0.5794 | 0.4581 | 0.1745 | 0.006903 |
| pass2_keep | keep=0.1 | 0.7028 | 0.6586 | 0.5779 | 0.4593 | 0.1721 | 0.006571 |
| pass2_keep | keep=0.2 | 0.7057 | 0.6521 | 0.5913 | 0.4660 | 0.1681 | 0.009299 |
| pass2_keep | keep=0.3 | 0.7137 | 0.6548 | 0.5863 | 0.4759 | 0.1623 | 0.008920 |
| pass2_keep | keep=0.4 | 0.7137 | 0.6632 | 0.5887 | 0.4847 | 0.1536 | 0.009081 |
| pass2_keep | keep=0.5 | 0.7157 | 0.6640 | 0.5940 | 0.4806 | 0.1505 | 0.009409 |
| pass2_keep | keep=0.6 | 0.7206 | 0.6721 | 0.5910 | 0.4909 | 0.1466 | 0.009325 |
| pass2_keep | keep=0.7 | 0.7277 | 0.6770 | 0.5946 | 0.4955 | 0.1426 | 0.010161 |
| pass2_keep | keep=0.8 | 0.7417 | 0.6755 | 0.5913 | 0.4968 | 0.1327 | 0.010611 |
| pass2_keep | keep=0.9 | 0.7393 | 0.6680 | 0.5916 | 0.4825 | 0.1279 | 0.009647 |
| pass2_keep | keep=1.0 | 0.7475 | 0.6715 | 0.5940 | 0.4818 | 0.1205 | 0.010164 |
| pass2_block | block_scale=0.5,p0=0.005 | 0.6934 | 0.6637 | 0.5887 | 0.4458 | 0.1780 | 0.007247 |
| pass2_block | block_scale=0.5,p0=0.02 | 0.6834 | 0.6558 | 0.5964 | 0.4435 | 0.1767 | 0.009608 |
| pass2_block | block_scale=0.5,p0=0.06 | 0.6921 | 0.6524 | 0.5779 | 0.4474 | 0.1657 | 0.007204 |
| pass2_block | block_scale=1.0,p0=0.005 | 0.6934 | 0.6635 | 0.5887 | 0.4439 | 0.1785 | 0.007238 |
| pass2_block | block_scale=1.0,p0=0.02 | 0.6959 | 0.6591 | 0.6012 | 0.4654 | 0.1788 | 0.010159 |
| pass2_block | block_scale=1.0,p0=0.06 | 0.7028 | 0.6586 | 0.5779 | 0.4593 | 0.1721 | 0.006571 |
| pass2_block | block_scale=2.0,p0=0.005 | 0.7055 | 0.6685 | 0.5955 | 0.4507 | 0.1800 | 0.007602 |
| pass2_block | block_scale=2.0,p0=0.02 | 0.7037 | 0.6632 | 0.5833 | 0.4623 | 0.1788 | 0.006861 |
| pass2_block | block_scale=2.0,p0=0.06 | 0.7253 | 0.6348 | 0.5663 | 0.5228 | 0.1814 | 0.015317 |
| pass2_block | block_scale=4.0,p0=0.005 | 0.7235 | 0.6756 | 0.5839 | 0.5345 | 0.1900 | 0.014658 |
| pass2_block | block_scale=4.0,p0=0.02 | 0.7424 | 0.6583 | 0.5164 | 0.5139 | 0.1923 | 0.009589 |
| pass2_block | block_scale=4.0,p0=0.06 | 0.7615 | 0.5658 | 0.4188 | 0.5715 | 0.1954 | 0.052111 |

### 7.8 Decision

| arm | G8 cells | state cells | total | eligible (all 4 state cells) | lineup K-S D | simplicity |
|---|---:|---:|---:|---|---:|---:|
| R2_hier_dirichlet | 4/6 | 3/4 | 7 | NO | 0.1942 | 1 |
| R7_keep_logistic | 2/6 | 0/4 | 2 | NO | 0.0302 | 4 |
| R5_hybrid | 1/6 | 0/4 | 1 | NO | 0.1047 | 2 |
| R8_keep_cell | 1/6 | 0/4 | 1 | NO | 0.1334 | 3 |

**No arm adopted.** no arm has every state-dependence cell inside +/- 3 pp. Cell-by-cell misses:

| arm | cell | sim | actual | miss |
|---|---|---:|---:|---:|
| R2_hier_dirichlet | late_starter_share_b2 | 0.4750 | 0.5223 | -4.7 pp |
| R5_hybrid | late_starter_share_b0 | 0.6942 | 0.7491 | -5.5 pp |
| R5_hybrid | late_starter_share_b1 | 0.6566 | 0.7240 | -6.7 pp |
| R5_hybrid | late_starter_share_b2 | 0.5911 | 0.5223 | +6.9 pp |
| R5_hybrid | foul_trouble_share | 0.4964 | 0.4613 | +3.5 pp |
| R8_keep_cell | late_starter_share_b0 | 0.6982 | 0.7491 | -5.1 pp |
| R8_keep_cell | late_starter_share_b1 | 0.6599 | 0.7240 | -6.4 pp |
| R8_keep_cell | late_starter_share_b2 | 0.5956 | 0.5223 | +7.3 pp |
| R8_keep_cell | foul_trouble_share | 0.4997 | 0.4613 | +3.8 pp |
| R7_keep_logistic | late_starter_share_b0 | 0.7074 | 0.7491 | -4.2 pp |
| R7_keep_logistic | late_starter_share_b1 | 0.6749 | 0.7240 | -4.9 pp |
| R7_keep_logistic | late_starter_share_b2 | 0.5994 | 0.5223 | +7.7 pp |
| R7_keep_logistic | foul_trouble_share | 0.5125 | 0.4613 | +5.1 pp |

