# Rotation model (L4) -- experiments

Append-only. Section 1 is the pre-registration, written and committed BEFORE any
arm was run (CLAUDE.md standing rule "bake-off before any choice").

---

## 1. Pre-registration (PM-authored, 2026-09-10)

Purpose: the possession engine needs, at every possession, the five players on the floor for each team, produced by a stochastic rotation process that reproduces real minutes distributions, real lineup concentration, and real state dependence (foul trouble, blowouts, end-game). This bake-off chooses the rotation model.
Data: lineups exist only from 2023-24 (L13). Folds: F1 train 2024 test 2025 (selection); robustness = within-2025 walk-forward (train games before Jan 15 2025, test after). 2026 sealed (seal.assert_not_sealed).
Pregame inputs (all as-of, strictly before game date): each rostered player's season-to-date minutes per game (shift(1) expanding mean with shrinkage toward a role prior fitted on the training season), starter frequency, availability (did_not_play history; the last game's DNP), fouls per minute, and the team's as-of rotation depth (players with >= 10 min per game). Game-state inputs during the game: period, seconds remaining, score diff, each player's current fouls, minutes played so far vs target, time since last substitution.
Arms: (R1) minutes-share Dirichlet + scheduler: draw the ten-plus players' game minutes from a Dirichlet over as-of minute shares with a fitted concentration (and the "extend_profile" tail fix), then a deterministic scheduler that places stints to honour five-on-floor and period boundaries; (R2) hierarchical Dirichlet (starters family vs bench family, then within-family) + the same scheduler; (R3) stint hazard model: at each possession boundary, per-player logistic hazards of exiting (on floor) and entering (on bench) conditioned on the game-state inputs and the pregame target, fitted on 2024 substitution events; (R4) empirical stint-sequence resampling from the team's own last k games with state-conditioned splicing for blowouts (document k and the splice rule). Every arm must also produce garbage-time behaviour from data, not a rule: report how each does it.
Metrics on F1 test (2025 games with complete on-floor data): G8: per-player minutes mean and SD (rotation players, >= 10 min as-of) sim vs actual, K-S of per-player minutes distributions, share of team minutes by top-5 and top-8, number of players with > 0 minutes; lineup concentration: share of possessions played by the top-1, top-3, top-5 five-man lineups, sim vs actual; state dependence: starters' share of minutes in the final 8 minutes by margin band (<= 5, 6-15, > 15) and starters' share when in foul trouble (>= 4 fouls) sim vs actual; the CFB "too narrow / too short" pair: SD ratio of per-player minutes and the count of nonzero-minute players. Noise floor: seed-varied runs (50 seeds) per arm. All cells n < 300 player-games are UNDERPOWERED.
Decision rules: winner = the arm with the most G8 and state-dependence cells inside tolerance (minutes mean +/- 2.0; SD ratio 0.9-1.1; top-5 and top-8 shares +/- 2 pp; state-dependence shares +/- 3 pp), ties broken by lineup-concentration K-S, then by simplicity (R1 < R2 < R4 < R3). An arm that cannot reproduce blowout and foul-trouble behaviour within tolerance is not eligible regardless of G8. If no arm is eligible, adopt nothing and report the diagnosis.

---


## 2. F1 results (train 2024, test 2025) -- run 2026-09-10T18:27:50Z

Config: 1200 test games (of 4689 eligible, 5320 with complete on-floor data), 5 seeds per arm for the comparison and 50 seeds x 150 games for the noise floor. Rotation player-games per arm: 86,687 (actual 17,915) -- every cell below is far above the n < 300 UNDERPOWERED threshold unless flagged.

### 2.1 Player crosswalk coverage of the modelling path

| as-of feature rows | CBBD id -> ESPN id | joined to a hoopR box row | hoopR `did_not_play` agrees with zero on-floor minutes |
|---:|---:|---:|---:|
| 136,388 | 99.977% | 99.758% | 99.02% |

### 2.2 G8 and the CFB narrow/short pair

| metric | ACTUAL | R1_dirichlet | R2_hier_dirichlet | R3_stint_hazard | R4_stint_resample | tol |
|---|---:|---:|---:|---:|---:|---|
| minutes mean (rotation players) | 24.57 | 25.64  PASS | 25.57  PASS | 26.43  PASS | 26.15  PASS | +/- 2.0 |
| minutes SD, pooled | 9.72 | 9.89 | 10.30 | 9.52 | 9.31 | ratio 0.9-1.1 |
| SD ratio, pooled | 1.000 | 1.017  PASS | 1.059  PASS | 0.978  PASS | 0.958  PASS | 0.9-1.1 |
| minutes SD, within-player | 6.06 | 7.86 | 8.20 | 6.77 | 6.37 | ratio 0.9-1.1 |
| SD ratio, within-player | 1.000 | 1.296  FAIL | 1.353  FAIL | 1.117  FAIL | 1.051  PASS | 0.9-1.1 |
| top-5 share of team minutes | 0.7484 | 0.7620  PASS | 0.7690  FAIL | 0.7791  FAIL | 0.7701  FAIL | +/- 2 pp |
| top-8 share of team minutes | 0.9563 | 0.9647  PASS | 0.9660  PASS | 0.9821  FAIL | 0.9770  FAIL | +/- 2 pp |
| players with > 0 minutes | 9.62 | 9.02  PASS | 9.03  PASS | 8.57  FAIL | 8.64  PASS | +/- 1.0 |
| K-S of per-player minutes (D / p) | -- | 0.0800 / 5.37e-83 | 0.0893 / 1.43e-103 | 0.1006 / 7.73e-131 | 0.0592 / 1.10e-45 | report |

### 2.3 Lineup concentration

| metric | ACTUAL | R1_dirichlet | R2_hier_dirichlet | R3_stint_hazard | R4_stint_resample |
|---|---:|---:|---:|---:|---:|
| top-1 lineup share of possessions | 0.2961 | 0.2306 | 0.2378 | 0.2232 | 0.2762 |
| top-3 lineup share | 0.5450 | 0.4862 | 0.4974 | 0.4790 | 0.5449 |
| top-5 lineup share | 0.6921 | 0.6457 | 0.6574 | 0.6420 | 0.6993 |
| distinct lineups per team-game | 14.75 | 15.35 | 14.94 | 17.44 | 15.07 |
| K-S of the top-1 lineup share distribution (D) | -- | 0.2298 | 0.1917 | 0.2530 | 0.0355 |
| substitution rate at a possession boundary | 0.1518 (train) | 0.1429 | 0.1398 | 0.1848 | 0.1722 |

### 2.4 State dependence

| metric | ACTUAL | R1_dirichlet | R2_hier_dirichlet | R3_stint_hazard | R4_stint_resample | tol |
|---|---:|---:|---:|---:|---:|---|
| starters' share of on-floor slots, final 8:00, \|margin\| <= 5 (n=22,402 poss) | 0.7513 | 0.7235  PASS | 0.7275  PASS | 0.6797  FAIL | 0.6948  FAIL | +/- 3 pp |
| starters' share of on-floor slots, final 8:00, \|margin\| 6-15 (n=27,998 poss) | 0.7230 | 0.6903  FAIL | 0.6938  PASS | 0.6532  FAIL | 0.6817  FAIL | +/- 3 pp |
| starters' share of on-floor slots, final 8:00, \|margin\| > 15 (n=17,012 poss) | 0.5237 | 0.4632  FAIL | 0.4722  FAIL | 0.5634  FAIL | 0.6050  FAIL | +/- 3 pp |
| starters on floor while carrying >= 4 fouls (n=43,369) | 0.4596 | 0.4612  PASS | 0.4666  PASS | 0.4819  PASS | 0.4890  PASS | +/- 3 pp |
| (diagnostic) starters on floor at exactly 4 fouls (n=38,464) | 0.5109 | 0.5763 | 0.5794 | 0.8005 | 0.8099 | report |

**Diagnostic decomposition.** The rows above count, on each side, the five that side actually started. The model's as-of starter set overlaps the real starting five on **4.57 of 5** players, so part of any gap is picking the wrong fifth man rather than rotating him wrongly. Re-grading the **actual** on-floor sequence with the **model's** as-of starter set isolates that:

| metric | ACTUAL (own starters) | ACTUAL (as-of starter set) | R1_dirichlet | R2_hier_dirichlet | R3_stint_hazard | R4_stint_resample |
|---|---:|---:|---:|---:|---:|---:|
| final 8:00 starters' share, \|margin\| <= 5 | 0.7513 | 0.7256 | 0.7235 | 0.7275 | 0.6797 | 0.6948 |
| final 8:00 starters' share, \|margin\| 6-15 | 0.7230 | 0.6947 | 0.6903 | 0.6938 | 0.6532 | 0.6817 |
| final 8:00 starters' share, \|margin\| > 15 | 0.5237 | 0.5080 | 0.4632 | 0.4722 | 0.5634 | 0.6050 |
| starters on floor while carrying >= 4 fouls | 0.4596 | 0.4630 | 0.4612 | 0.4666 | 0.4819 | 0.4890 |


### 2.5 Noise floor (seed-varied runs)

SD across 50 seeds on 150 games:

| metric | R1_dirichlet | R2_hier_dirichlet | R3_stint_hazard | R4_stint_resample |
|---|---:|---:|---:|---:|
| minutes_mean | 0.15040 | 0.16960 | 0.15055 | 0.13197 |
| minutes_sd_pooled | 0.09033 | 0.09557 | 0.11900 | 0.08691 |
| top5_share | 0.00274 | 0.00307 | 0.00288 | 0.00231 |
| top8_share | 0.00160 | 0.00202 | 0.00140 | 0.00132 |
| n_nonzero_mean | 0.06708 | 0.07942 | 0.06940 | 0.04840 |
| lu_top1 | 0.00516 | 0.00566 | 0.00507 | 0.00493 |
| late_starter_share_b2 | 0.01356 | 0.01615 | 0.01466 | 0.01080 |
| foul_trouble_share | 0.01900 | 0.01806 | 0.01755 | 0.01734 |

### 2.6 Robustness -- within-2025 walk-forward (train games before 2025-01-15, test after; 400 games)

| metric | ACTUAL | R1_dirichlet | R2_hier_dirichlet | R3_stint_hazard | R4_stint_resample |
|---|---:|---:|---:|---:|---:|
| minutes mean | 24.72 | 25.19 | 25.14 | 25.86 | 25.95 |
| SD ratio pooled | 9.66 | 9.98 | 10.32 | 9.39 | 9.59 |
| top-5 share | 0.7484 | 0.7522 | 0.7569 | 0.7622 | 0.7681 |
| players > 0 min | 9.64 | 9.45 | 9.46 | 8.93 | 8.84 |
| late starter share, |m| > 15 | 0.5246 | 0.4647 | 0.4690 | 0.5445 | 0.6118 |
| foul-trouble share | 0.4391 | 0.4504 | 0.4496 | 0.4636 | 0.4737 |

### 2.7 Decision

| arm | G8 cells passed | state cells passed | total | eligible (blowout + foul trouble) | lineup-concentration K-S D | simplicity |
|---|---:|---:|---:|---|---:|---:|
| R2_hier_dirichlet | 4/6 | 3/4 | 7 | NO | 0.1917 | 2 |
| R1_dirichlet | 5/6 | 2/4 | 7 | NO | 0.2298 | 1 |
| R4_stint_resample | 4/6 | 1/4 | 5 | NO | 0.0355 | 3 |
| R3_stint_hazard | 2/6 | 1/4 | 3 | NO | 0.2530 | 4 |

**No arm adopted.** no arm reproduces both blowout and foul-trouble behaviour inside tolerance. Per the pre-registration, we adopt nothing and report the diagnosis.


### 2.8 Fitted parameters

| parameter | value | fitted how |
|---|---:|---|
| `k0` (shrinkage weight) | 0.00 | grid, min next-game minutes MAE on train |
| `w_dnp` | 0.431 | realised/predicted minutes of last-game-DNP players |
| `tail_ratio` | 0.695 | median share ratio of successive tail ranks |
| `n_profile` | 15 | joint scheduler grid (nonzero-count target) |
| `alpha` (R1) | 52.8 | method of moments on share residual variance |
| `alpha_family` (R2) | 28.0 | same, family level |
| `alpha_starters` (R2) | 45.2 | same, within starters |
| `alpha_bench` (R2) | 8.0 | same, within bench |
| `ema_horizon` (s) | 360 | joint grid on train (sub rate, distinct lineups, top-5 share) |
| `swap_threshold` | 0.48 | same joint grid |
| `lam_deficit` | 0.50 | same joint grid |
| `foul_rate_scale` | 1.065 | match train team fouls per game |
| `fpm_league` (fouls per on-floor minute) | 0.0830 | train pooled |
| `fpm_prior_min` | 150 | grid, min next-game foul MAE |
| `w_avail_dnp` | 0.387 | realised play rate of last-game-DNP players |
| `min_share` | 0.02306 | mean share of the smallest nonzero-minutes player in a team-game |
| `start_alpha` (starter-predictor EWMA decay) | 0.50 | grid, max overlap with the real starting five on train |
| `p_play` by as-of rank 1-12 | 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.89, 0.84, 0.73, 0.56, 0.42, 0.28 | train P(records any minutes) |


---

## 3. Round 2 pre-registration (PM-authored, 2026-09-10)

Round 1 diagnosis: the minutes-share arms (R1, R2) reproduce minutes but assemble lineups that are too diffuse and pull starters too early in blowouts; the stint-sequence arms (R3, R4) reproduce which fives play together but keep starters on the floor in foul trouble (81% vs 42% actual). Round 2 keeps R2 as the incumbent reference and adds two arms that separate "who plays together" from "when the coach deviates". Data, folds, metrics, tolerances, noise floor and decision rules are unchanged from round 1, except that the test set must be the FULL set of 2025 games with complete on-floor data (or a random subset of at least 1,500 games if compute forces it, with the subset seed recorded), not 240.
R5 hybrid: R4's own-recent-games stint-sequence resampling (k fitted on 2024 from {3, 5, 8, all-season}) provides the baseline lineup sequence; two state-conditioned OVERRIDE hazards, fitted on 2024 substitution events, modify it at possession boundaries: (a) foul-trouble exit: per-player logistic hazard on (fouls so far x period x seconds remaining x is_starter), and the corresponding re-entry hazard; (b) margin overrides: starters-exit hazard on (|margin| x seconds remaining x is_starter) for blowouts and a starters-re-enter / bench-exit hazard on (|margin| <= 5 x seconds remaining <= 480). The overrides must be fitted, with their coefficients reported; no hand-set thresholds.
R6 stint hazard, respecified: R3 with fouls-so-far, margin and seconds remaining entered as explicit interactions with is_starter and period (round 1's flat feature entry could not represent foul trouble), tree-parameter search on 2024 only if a tree variant is used.
Gates unchanged: G8 cells (minutes mean +/- 2.0, SD ratio 0.9-1.1, top-5/top-8 share +/- 2 pp), lineup concentration K-S, state-dependence cells (starters' share in the final 8 min by margin band <= 5 / 6-15 / > 15, starters' share under foul trouble, each +/- 3 pp). An arm that misses any state-dependence cell is ineligible regardless of G8. Winner = most cells inside tolerance among eligible arms, ties by lineup K-S then simplicity (R2 < R5 < R6). If no arm is eligible, adopt nothing and report which cell fails and by how much.

---

## 4. Round-2 results (train 2024, test 2025) -- run 2026-09-10T19:58:43Z

Test universe: random subset of 1600 of 4689 eligible games, numpy RandomState seed 2025 (subset seed 2025), 5 seeds per arm; noise floor 50 seeds x 150 games. Rotation player-games per arm 115,702 (actual 23,914); every cell is far above the n < 300 UNDERPOWERED threshold.

### 4.1 G8 cells

| cell | tol | ACTUAL | R2_hier_dirichlet | R5_hybrid | R6_stint_hazard_v2 |
|---|---|---:|---:|---:|---:|
| minutes mean (rotation players) | +/- 2.0 | 24.55 | 25.55  PASS | 26.22  PASS | 26.40  PASS |
| minutes SD ratio, pooled | 0.9-1.1 | 1.000 | 1.064  PASS | 0.999  PASS | 0.980  PASS |
| minutes SD ratio, within-player | 0.9-1.1 | 1.000 | 1.354  FAIL | 1.235  FAIL | 1.131  FAIL |
| top-5 share of team minutes | +/- 2 pp | 0.7472 | 0.7683  FAIL | 0.7759  FAIL | 0.7783  FAIL |
| top-8 share of team minutes | +/- 2 pp | 0.9560 | 0.9656  PASS | 0.9788  FAIL | 0.9817  FAIL |
| players with > 0 minutes | +/- 1.0 | 9.64 | 9.04  PASS | 8.57  FAIL | 8.58  FAIL |

### 4.2 State-dependence cells (an arm missing ANY of these is ineligible)

| cell | ACTUAL | R2_hier_dirichlet | R5_hybrid | R6_stint_hazard_v2 |
|---|---:|---:|---:|---:|
| final 8:00 starters' share, \|margin\| <= 5 (n=29,660 poss) | 0.7491 | 0.7281 (-2.1 pp)  PASS | 0.7063 (-4.3 pp)  FAIL | 0.6839 (-6.5 pp)  FAIL |
| final 8:00 starters' share, \|margin\| 6-15 (n=36,804 poss) | 0.7240 | 0.6943 (-3.0 pp)  PASS | 0.6555 (-6.9 pp)  FAIL | 0.6523 (-7.2 pp)  FAIL |
| final 8:00 starters' share, \|margin\| > 15 (n=23,360 poss) | 0.5223 | 0.4750 (-4.7 pp)  FAIL | 0.4937 (-2.9 pp)  PASS | 0.5422 (+2.0 pp)  PASS |
| starters on floor while carrying >= 4 fouls (n=58,652) | 0.4613 | 0.4648 (+0.4 pp)  PASS | 0.4680 (+0.7 pp)  PASS | 0.4731 (+1.2 pp)  PASS |
| (diagnostic) at exactly 4 fouls (n=51,745) | 0.5166 | 0.5777 | 0.8172 | 0.7982 |

The rows above count, on each side, the five that side actually started. The model's as-of starter set overlaps the real starting five on **4.58 of 5**; re-grading the ACTUAL on-floor sequence with the MODEL's starter set separates "wrong five" from "wrong rotation":

| cell | ACTUAL (own starters) | ACTUAL (as-of starter set) | R2_hier_dirichlet | R5_hybrid | R6_stint_hazard_v2 |
|---|---:|---:|---:|---:|---:|
| final 8:00, \|margin\| <= 5 | 0.7491 | 0.7215 | 0.7281 | 0.7063 | 0.6839 |
| final 8:00, \|margin\| 6-15 | 0.7240 | 0.6964 | 0.6943 | 0.6555 | 0.6523 |
| final 8:00, \|margin\| > 15 | 0.5223 | 0.5048 | 0.4750 | 0.4937 | 0.5422 |
| >= 4 fouls | 0.4613 | 0.4636 | 0.4648 | 0.4680 | 0.4731 |

### 4.3 Lineup concentration

| metric | ACTUAL | R2_hier_dirichlet | R5_hybrid | R6_stint_hazard_v2 |
|---|---:|---:|---:|---:|
| top-1 lineup share of possessions | 0.2940 | 0.2369 | 0.2876 | 0.2217 |
| top-3 lineup share | 0.5426 | 0.4957 | 0.5665 | 0.4748 |
| top-5 lineup share | 0.6894 | 0.6555 | 0.7206 | 0.6369 |
| distinct lineups per team-game | 14.84 | 15.01 | 14.34 | 17.68 |
| K-S of per-player minutes (D) | -- | 0.0883 | 0.0709 | 0.1019 |
| K-S of the top-1 lineup share distribution (D) | -- | 0.1942 | 0.0291 | 0.2645 |
| substitution rate at a possession boundary | 0.1518 (train) | 0.1402 | 0.1780 | 0.1871 |

### 4.4 Noise floor (50 seeds x 150 games)

| metric | R2_hier_dirichlet | R5_hybrid | R6_stint_hazard_v2 |
|---|---:|---:|---:|
| minutes_mean | 0.16426 | 0.14467 | 0.16226 |
| top5_share | 0.00304 | 0.00318 | 0.00264 |
| n_nonzero_mean | 0.07888 | 0.05331 | 0.07484 |
| lu_top1 | 0.00541 | 0.00659 | 0.00478 |
| late_starter_share_b0 | 0.00807 | 0.00977 | 0.00903 |
| late_starter_share_b1 | 0.00967 | 0.00752 | 0.00903 |
| late_starter_share_b2 | 0.01512 | 0.01438 | 0.01451 |
| foul_trouble_share | 0.02027 | 0.02083 | 0.01976 |

### 4.5 R5's fitted donor depth and override coefficients

Donor depth **k = 3** (last 3 games), chosen on the training season by lineup-concentration error:

| k | top-1 lineup share | top-3 | distinct lineups | rel. sq. error |
|---|---:|---:|---:|---:|
| 3 | 0.2645 | 0.5390 | 15.22 | 0.00216 |
| 5 | 0.2604 | 0.5118 | 16.11 | 0.01074 |
| 8 | 0.2690 | 0.5308 | 15.55 | 0.00220 |
| all | 0.2600 | 0.5133 | 16.18 | 0.01144 |

`override_scale` = **4.00**, `p0` (neutral-state block floor) = **0.020**, fitted on the TRAINING season against the four state-dependence cells the overrides exist to produce. Train targets: late band 0 0.7161, late band 1 0.7060, late band 2 0.5186, foul_trouble_share>= 4 fouls 0.4193.

| scale, p0 | late b0 | late b1 | late b2 | >= 4 fouls | sub rate | sq. err |
|---|---:|---:|---:|---:|---:|---:|
| scale=0.0,p0=0.005 | 0.7090 | 0.6774 | 0.6032 | 0.4793 | 0.1774 | 0.011640 |
| scale=0.5,p0=0.005 | 0.7090 | 0.6774 | 0.5998 | 0.4819 | 0.1785 | 0.011386 |
| scale=1.0,p0=0.005 | 0.7090 | 0.6762 | 0.5995 | 0.4819 | 0.1785 | 0.011413 |
| scale=2.0,p0=0.005 | 0.7090 | 0.6755 | 0.5993 | 0.4822 | 0.1773 | 0.011454 |
| scale=4.0,p0=0.005 | 0.7090 | 0.6748 | 0.5649 | 0.4793 | 0.1785 | 0.006770 |
| scale=8.0,p0=0.005 | 0.7241 | 0.6531 | 0.3926 | 0.4700 | 0.1957 | 0.021308 |
| scale=16.0,p0=0.005 | 0.8131 | 0.4511 | 0.2312 | 0.4113 | 0.2071 | 0.157024 |
| scale=32.0,p0=0.005 | 0.8683 | 0.2535 | 0.2572 | 0.4172 | 0.1924 | 0.296219 |
| scale=0.0,p0=0.02 | 0.7035 | 0.6717 | 0.6027 | 0.4907 | 0.1735 | 0.013520 |
| scale=0.5,p0=0.02 | 0.7035 | 0.6693 | 0.6027 | 0.4907 | 0.1747 | 0.013690 |
| scale=1.0,p0=0.02 | 0.7085 | 0.6680 | 0.6027 | 0.4909 | 0.1755 | 0.013717 |
| scale=2.0,p0=0.02 | 0.7161 | 0.6635 | 0.5782 | 0.4915 | 0.1730 | 0.010585 |
| scale=4.0,p0=0.02 | 0.7209 | 0.6530 | 0.5223 | 0.4812 | 0.1829 | 0.006677 |
| scale=8.0,p0=0.02 | 0.7530 | 0.5638 | 0.2666 | 0.4364 | 0.2058 | 0.085371 |
| scale=16.0,p0=0.02 | 0.8383 | 0.3599 | 0.2498 | 0.4293 | 0.2131 | 0.207083 |
| scale=32.0,p0=0.02 | 0.8746 | 0.2492 | 0.2624 | 0.4309 | 0.1954 | 0.299543 |
| scale=0.0,p0=0.06 | 0.6991 | 0.6673 | 0.6087 | 0.4977 | 0.1612 | 0.016063 |
| scale=0.5,p0=0.06 | 0.7007 | 0.6628 | 0.5886 | 0.4937 | 0.1641 | 0.012552 |
| scale=1.0,p0=0.06 | 0.7000 | 0.6490 | 0.5765 | 0.4870 | 0.1703 | 0.011454 |
| scale=2.0,p0=0.06 | 0.7129 | 0.6328 | 0.5507 | 0.4777 | 0.1759 | 0.009813 |
| scale=4.0,p0=0.06 | 0.7360 | 0.5878 | 0.3851 | 0.4852 | 0.1993 | 0.036518 |
| scale=8.0,p0=0.06 | 0.8034 | 0.4495 | 0.2339 | 0.4405 | 0.2084 | 0.154895 |
| scale=16.0,p0=0.06 | 0.8512 | 0.2955 | 0.2495 | 0.4054 | 0.2018 | 0.259326 |
| scale=32.0,p0=0.06 | 0.8807 | 0.2714 | 0.2730 | 0.4394 | 0.1990 | 0.276626 |

Fitted override coefficients (logit scale; positive = more likely to be benched / to come on):

| feature | exit (bench-him) | enter (bring-him-back) |
|---|---:|---:|
| `fouls` | +0.0000 | +0.0000 |
| `fouls_x_is_starter` | +0.0000 | +0.0000 |
| `fouls_x_period2` | +0.0000 | +0.0000 |
| `fouls_x_sec_left_frac` | +0.0000 | +0.0000 |
| `foul_out` | +0.0000 | +0.0000 |
| `is_starter` | -0.7086 | +1.9878 |
| `abs_margin` | -0.2617 | -0.1829 |
| `abs_margin_x_is_starter` | +0.2235 | -0.2977 |
| `abs_margin_x_late` | -0.0000 | +0.0809 |
| `abs_margin_x_sec_left_frac` | +0.5695 | +0.8891 |
| `is_close_x_late` | -0.0329 | -0.4278 |
| `is_close_x_late_x_is_starter` | -0.3927 | +0.2704 |
| `period2` | -0.1648 | +0.0280 |
| `sec_left_frac` | -0.6838 | -0.5228 |
| _intercept_ | -2.2781 | -4.2299 |

### 4.6 Decision

| arm | G8 cells | state cells | total | eligible (all 4 state cells) | lineup K-S D | simplicity |
|---|---:|---:|---:|---|---:|---:|
| R2_hier_dirichlet | 4/6 | 3/4 | 7 | NO | 0.1942 | 1 |
| R5_hybrid | 2/6 | 2/4 | 4 | NO | 0.0291 | 2 |
| R6_stint_hazard_v2 | 2/6 | 2/4 | 4 | NO | 0.2645 | 3 |

**No arm adopted.** no arm has every state-dependence cell inside +/- 3 pp. Cell-by-cell misses:

| arm | cell | sim | actual | miss |
|---|---|---:|---:|---:|
| R2_hier_dirichlet | late_starter_share_b2 | 0.4750 | 0.5223 | -4.7 pp |
| R5_hybrid | late_starter_share_b0 | 0.7063 | 0.7491 | -4.3 pp |
| R5_hybrid | late_starter_share_b1 | 0.6555 | 0.7240 | -6.9 pp |
| R6_stint_hazard_v2 | late_starter_share_b0 | 0.6839 | 0.7491 | -6.5 pp |
| R6_stint_hazard_v2 | late_starter_share_b1 | 0.6523 | 0.7240 | -7.2 pp |


---

## 5. Defect found after the round-2 run: the hazard design matrix carried no foul state (2026-09-10)

**What.** `cbb_sim.models.rotation.build_hazard_training` initialised each
team-game's foul vector to zero and never advanced it, so every logistic hazard
fitted in rounds 1 and 2 saw `fouls = 0` on every training row. sklearn returned
exactly `+0.0000` for all five foul coefficients of R5's override pair, which is
visible in section 4.5 above and is how the defect was caught. This is *not* a
leak and does not affect any gate on the actual side; it means the foul features
the round-2 pre-registration specifically asked for ("fouls so far x period x
seconds remaining x is_starter") were inert.

**Who is affected.** R3 (round 1) and R5, R6 (round 2), i.e. every arm with a
fitted hazard. **R1, R2 and R4 are unaffected** -- they use no hazard, and R2's
round-1 and round-2 numbers are identical to four decimals (minutes mean 25.57 /
25.55, top-5 0.7690 / 0.7683, late b0/b1/b2 0.7275-0.7281 / 0.6938-0.6943 /
0.4722-0.4750), which is the cross-check that the two runs are otherwise the same
experiment on different game subsets. Note also that the round-2 arms still
*passed* the `>= 4 fouls` cell with an inert foul term, because the `>= 4` state
pools fouled-out players (forced off the floor by the rules, share 0) with
four-foul players; the pre-registered diagnostic row "at exactly 4 fouls" exposes
it, at 0.8172 / 0.7982 against a real 0.5166. A cell can pass for the wrong
reason, and that is what the diagnostic row is for.

**Fix.** The real per-possession foul state is now read from the team-game's own
`PersonalFoul` events while fitting (legitimate there -- the label is that same
game's substitution) and never at simulation time, where fouls remain simulated
from the as-of rate. `since_change` was also reset properly on entry and exit.

**Corrected fit.** Re-fitting R5's overrides on 2024 with the corrected matrix
gives the signs the arm was specified to have, and the foul terms now do work:

| feature | exit (bench-him) | enter (bring-him-back) |
|---|---:|---:|
| `fouls` | +0.3454 | +0.0009 |
| `fouls_x_is_starter` | +0.1053 | -0.6248 |
| `fouls_x_period2` | -0.3287 | +0.5937 |
| `fouls_x_sec_left_frac` | +0.1169 | +0.5755 |
| `foul_out` | +0.3276 | -2.0976 |
| `is_starter` | -0.7963 | +2.1318 |
| `abs_margin` | -0.2361 | -0.0714 |
| `abs_margin_x_is_starter` | +0.1827 | -0.2281 |
| `abs_margin_x_late` | +0.0399 | +0.0544 |
| `abs_margin_x_sec_left_frac` | +0.5637 | +0.6413 |
| `is_close_x_late` | +0.1426 | -0.3341 |
| `is_close_x_late_x_is_starter` | -0.5479 | +0.5078 |
| `period2` | +0.0623 | -0.3083 |
| `sec_left_frac` | -0.2282 | -0.3439 |
| _intercept_ | -2.7412 | -4.3756 |

Read them as the arm's fitted coaching rules: an extra foul raises a player's
bench hazard (+0.345) and raises it further for a starter (+0.105), while cutting
a starter's re-entry hazard hard (-0.625); a fouled-out player's re-entry
collapses (-2.098); a wider margin raises a starter's bench hazard relative to a
bench player's (+0.183) and lowers his re-entry (-0.228); and close-and-late
specifically protects a starter from being benched (-0.548) and pulls him back on
(+0.508). With the corrected matrix the fitted override strength moves from
(scale 4.00, p0 0.020) to (scale 1.00, p0 0.060) -- a weaker, broader override,
because the foul terms now carry load the margin terms previously had to absorb.
On the training season the corrected arm lands at late b0/b1/b2 0.7089 / 0.6493 /
0.5800 and `>= 4 fouls` 0.4752 against targets 0.7161 / 0.7060 / 0.5186 / 0.4193.

**Artifact note.** The graded round-2 run's results JSON and CSV were
subsequently overwritten by a 30-game development smoke run and have been
renamed `rotation_F1_round2_SMOKE30_do_not_cite.{json,csv}`; **section 4 above is
the record of the graded round-2 run** (1,600 games, subset seed 2025, 5 seeds,
115,702 rotation player-games, 50-seed x 150-game noise floor), and re-running
`scripts/train_rotation_v2.py` with those arguments reproduces it. Round 1's
`rotation_F1_results.json` / `_table.csv` are intact and were verified complete
(4 arms x 12,000 team-game sims, 50-seed noise floor for every arm, 400-game
robustness fold).

**Status: OPEN.** The corrected full F1 re-run of R5 and R6 was started and was
cut off by the session's hard stop after R2 had reproduced its section-4 numbers
exactly; it must be completed before R5 or R6 is compared again. Section 4's R5
and R6 columns therefore stand as *inert-foul-term* results and must not be cited
as the arms' performance. R2's section-4 column, and the whole of section 2
except its R3 column, are unaffected. Artifact:
`data/processed/models/rotation/rotation_fit_round2_corrected_hazards.json`.

---

## 6. Round 3 pre-registration (PM-directed, worker-authored 2026-09-10)

Written and committed BEFORE any round-3 arm was run. Evidence it is built on:
`docs/tests/rotation_close_game_audit_2026-09-10.md` (also written before this
section, from the round-2 fits only).

### 6.1 What round 3 is, and why

Round 2 adopted nothing. The PM's direction for round 3, not to be reopened:
**R5 with a close-game keep-starters override, fitted from data exactly as the
donor depth and the block override were, and carried as a model component --
never as a post-hoc adjustment of engine output** (`CLAUDE.md`, "no hand tuning
on engine output").

The mechanism the direction names is the one the audit isolates. R5's existing
override is one-sided: it can only **block** a player onto the bench. The donor
sequence therefore sets a ceiling on the starters' share in every state and the
block can only push that share down, so an arm that is short of starters late in
a close game has no mechanism that could repair it. The keep override is the
missing half of the same object.

Two audit results qualify the round-2 record and are carried into this
pre-registration rather than discovered afterwards:

1. **The corrected R5 fails all three margin bands, not two.** Section 4's R5
   column was produced with section 5's inert foul terms. Re-run with the
   corrected matrix and its refitted knobs, R5's final-8:00 cells are 0.6948
   (-5.4 pp), 0.6572 (-6.7 pp) and **0.5868 (+6.5 pp)** against 0.7491 / 0.7240 /
   0.5223. The round-2 blowout PASS was an artifact of the defect. Round 3's
   baseline is an arm missing three cells, and the blowout miss is concentrated
   in the final 2:00 at |margin| > 15 (0.5482 vs 0.3405, +20.8 pp).
2. **The round-2 "starter time is spent too early" hypothesis (`model.md` section
   10) is refuted.** R5's close-band starter share is 5.3 pp LOW in the first half
   and 5.4 pp low in the final eight minutes; it is not a redistribution of a
   correct total. A within-game time-profile constraint would not have fixed it.

### 6.2 Folds -- a stated deviation from the standing fold rule

`CLAUDE.md` fixes fold 1 = train through 2022-23 / test 2023-24 and fold 2 =
train through 2023-24 / test 2024-25, with fold 2 selecting. **Those folds are
not constructible for this model.** CBBD carries no on-floor data at all before
2023-24 -- 0.0000 of possessions in both 2022 and 2023 (audit section 1, L13) --
so fold 1 has no training season and fold 2 is the only fold that exists. Round
3 therefore runs the model's established F1, **train 2024, test 2025**, which
*is* the standing fold 2, and reports the robustness fold rounds 1-2 used
(within-2025 walk-forward) only if an arm is otherwise adoptable. 2026 stays
sealed; `seal.assert_not_sealed` guards the trainer.

### 6.3 Arms

All four are graded by one blind grading path (`train_rotation_v1.build_row` /
`verdict` / `rotation.aggregate_stats`), the same code that graded rounds 1 and 2.

| arm | what it is | simplicity |
|---|---|---:|
| `R2_hier_dirichlet` | the incumbent reference, unchanged | 1 |
| `R5_hybrid` | round 2's donor + block hybrid with the **corrected** hazard matrix. This column closes section 5's OPEN item and is the baseline the keep arms must beat | 2 |
| `R8_keep_cell` | R5 + the **simpler threshold form** of the keep override | 3 |
| `R7_keep_logistic` | R5 + the **logistic form** of the keep override | 4 |

**R7 `keep_logistic`.** A logistic on-floor propensity `w_on` is fitted on the
training season over one row per (team-game, possession, candidate), label = "on
the floor at this possession", features (`rotation_v3.KEEP_FEATURES`): `fouls`,
`fouls x is_starter`, `foul_out`, `is_starter`, `|margin|`,
`|margin| x is_starter`, `late`, `late x is_starter`, `is_close x late`,
`is_close x late x is_starter`, `|margin| x late`, `target_share` (the team
prior), `target_share x is_close x late`, `target_share x late`, `period2`,
`sec_left_frac`. At simulation time the keep score is the propensity
**deviation** from a neutral state (early, tied, no fouls),

    z_i    = (x_i - x_i^neutral) . w_on
    p_i    = sigmoid(scale_k * z_i + logit(q0))
    keep_i = tol2_i < p_i,    tol2_i ~ U(0,1) drawn ONCE per player per game

so the override is inert in an ordinary state exactly as the block is, and the
per-game tolerance draw makes a keep persistent and self-clearing rather than a
per-possession coin flip. `is_starter`, `target_share`, `period2` and
`sec_left_frac` are the neutral carriers and cancel out of the deviation
(`KEEP_STATE_COLS`), the same convention `OVERRIDE_STATE_COLS` uses for the
block. Reading the training game's own foul events while fitting is legitimate
and is never done at simulation time, where fouls are simulated from the as-of
rate -- the rule established by section 5's fix.

**R8 `keep_cell`.** The simpler threshold form: no logistic, no player features
beyond the starter flag, one knob.

    p_i    = theta * s*[time_bucket, margin_bucket]   if i is a predicted starter
           = 0                                        otherwise
    keep_i = tol2_i < p_i

`s*` is the training season's own starters'-share-of-on-floor-slots table by
(time bucket x margin bucket) -- a pure data table, the same object
`TiltTables.state` is built from, and the reason R2 gets the close bands right.
It is league-average by construction; the pre-registered slope check (6.6) is
what will expose that if it matters.

**Shared displacement rule.** A kept player who is not in the donor-mapped five
replaces the worst as-of-ranked member of that five who is neither kept nor
blocked. No override may put a fouled-out or unavailable player on the floor and
no override may act on more than five players at once.

### 6.4 Fitting

Base fit = `rotation_fit_round2_corrected_hazards.json` (training season 2024
only), so R5's column is exactly section 5's corrected arm. Round 3 fits only
what it adds:

- `w_on` (R7) on 300 training games, `LogisticRegression(C=1.0, lbfgs)`;
- `s*` (R8) on the whole training season;
- the knobs, by **coordinate descent, two passes**, on 120 training team-games:
  keep knob(s) with the block held at its round-2 corrected value, then the
  block pair `(scale, p0)` at that keep, then both again. Objective: the sum of
  squared errors of the four state-dependence cells against the **training
  season's own** cells, the identical objective round 2's `fit_override_scale`
  used. Grids: keep scale {0, 0.5, 1, 2, 4, 8} x q0 {0.02, 0.05, 0.15, 0.35};
  theta {0.0, 0.1, ..., 1.0}; block scale {0.5, 1, 2, 4} x p0 {0.005, 0.02,
  0.06}. **Every grid point evaluated is reported**, as in section 4.5. There is
  no hand-set threshold anywhere in either arm.

### 6.5 Gates -- unchanged from round 2, none softened

Test universe: the **same** 1,600-game subset of 2025 the graded round-2 run
used (numpy RandomState seed 2025), 5 seeds per arm, so round-2 and round-3
columns are directly comparable.

- **G8 cells** (report, not veto): minutes mean +/- 2.0; minutes SD ratio pooled
  and within-player 0.9-1.1; top-5 and top-8 share of team minutes +/- 2 pp;
  players with > 0 minutes +/- 1.0.
- **State-dependence cells** (the veto): starters' share of on-floor slots in the
  final 8:00 at |margin| <= 5 / 6-15 / > 15, and starters' share while carrying
  >= 4 fouls, each +/- 3 pp. **An arm missing ANY of these four is ineligible
  regardless of G8.** The "at exactly 4 fouls" diagnostic row is reported
  alongside, because section 5 showed the >= 4 cell can pass for the wrong
  reason.
- **Lineup concentration**: top-1 / top-3 / top-5 five-man lineup share, distinct
  lineups per team-game, K-S D of the top-1 lineup share distribution, K-S D of
  per-player minutes, substitution rate at a possession boundary.
- **Per-player minutes**: mean, pooled SD, within-player SD and the K-S of the
  per-player minutes distribution, all through `pooled_and_within_sd` -- the
  round-1/2 code path, so per-player minutes error is reported in the same
  currency as the two earlier rounds rather than as a new statistic.
- Any cell with n < 300 player-games or possessions is labelled UNDERPOWERED.

### 6.6 Slope check (standing rule: matchup-specific, not league-average)

Team-games are bucketed into quintiles of the **pregame** team prior (the as-of
predicted share of team minutes going to the predicted starting five) and the
close-and-late cell (final 8:00, |margin| <= 5) is reported per quintile for
ACTUAL and every arm, with the fitted slope against the prior and Q5 - Q1.
Measured on the round-2 fits, actual is +0.692 (Q5 - Q1 +14.5 pp), R2 +0.632
(+13.3 pp), R5 +0.763 (+16.2 pp). An arm whose quintile profile is flat, or
whose slope sign disagrees with actual, is reported as not matchup-specific
whatever its pooled cells say.

### 6.7 Noise floor

Two, both reported:

- **A, seed-varied sim runs**: 20 seeds x 150 games per arm, the SD of every
  gate cell. (Round 2 used 50 x 150; 20 is a compute concession with four other
  workers on the machine and is stated as such. It widens the uncertainty on the
  noise estimate, it does not move any gate.)
- **B, spec-identical refit under a second seed**: the whole of 6.4 re-run with
  a different training-game sample (fit seed 101 vs 11), a different logistic
  `random_state`, and a different sim seed inside the knob grid (23 vs 7), then
  graded on the same 150-game universe as the first fit. A round-3 arm counts as
  beating the baseline only if its improvement on a state cell exceeds the
  refit-to-refit spread on that cell.

### 6.8 Decision rule

Winner = the eligible arm (all four state cells inside +/- 3 pp) with the most
cells inside tolerance across G8 + state; ties broken by lineup-concentration
K-S D, then by simplicity in the order R2 < R5 < R8 < R7. **Ties go to the
simpler model**, so R8 beats R7 on equal cells. An arm whose improvement over
R5 does not clear noise floor B on the cell it was built to fix is not adopted
on that cell. If no arm is eligible, **adopt nothing**, report which cell fails
and by how much, and name the diagnosis. No gate is relaxed to produce a winner.

### 6.9 Engine expressibility (a condition on adoption, checked and reported)

`RotationSampler.next_lineup` is the engine contract and
`engine/rotation_adapter.py` vectorises it. Whatever round 3 adopts must be
expressible in that decision-rule form; the two stated RNG divergences of
`docs/models/engine/model.md` section 4.5 (counter-based streams, one foul
uniform per roster slot) apply unchanged and any extra tolerance draw a keep
override needs becomes a third such stream. The adapter change required by each
candidate is reported with the results whether or not an arm is adopted.

---


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


### 7.9 Diagnosis -- why the keep override cannot reach the close-game cell

**The close-game cell is not reachable at any knob setting, and the grid proves
it rather than asserting it.** Over the whole 24-point R7 keep grid (7.7), the
largest close-band starters' share the arm produces on the training season is
**0.6994**, against a training target of 0.7110 and a test actual of 0.7491. The
grid is not a search that stopped early: raising the keep scale *lowers* b0
monotonically past scale 1 (0.6979 -> 0.6986 -> 0.6948 -> 0.6623 -> 0.6590 at
scale 0 / 1 / 2 / 4 / 8), and raising the base rate `q0` lowers it faster
(0.6979 -> 0.6934 -> 0.6712 -> 0.6461 at q0 0.02 / 0.05 / 0.15 / 0.35).

**The mechanism, read off the fitted coefficients.** The keep is applied to every
candidate, as pre-registered -- it is a fitted on-floor propensity, not a
starters-only rule. In the close-and-late state its deviation score separates a
starter from a bench player by only about **1.0 in log odds** (a starter with
target share 1.4 scores z = +0.89, a bench player with share 0.6 scores z =
-0.09, driven by `late_x_is_starter` +0.691 and `target_share_x_is_close_x_late`
+1.471 against `is_close_x_late` -0.629 and `target_share_x_late` -0.917).
A one-parameter scaling of a 1.0 log-odds separation cannot raise the starters'
share by 5 pp: at the fitted `q0 = 0.02` a starter is kept on 4.7% of
possessions, which moves the cell by about a point; raising `q0` far enough to
matter also keeps bench players (30% vs 14% at `q0 = 0.15`), and each kept bench
player displaces an un-kept starter, so the cell falls.

**In a blowout the same coefficients run the other way, and that is where the fit
spends the override.** At \|margin\| = 20 a starter scores z = -1.08 and a bench
player +0.04, so the keep pulls the bench on and pushes starters off -- which is
why the fitted setting (scale 1.0, q0 0.02) is the one that most reduces the
blowout cell (b2 0.5821 -> 0.5654 on train) and barely touches the close bands.
The coordinate descent behaved correctly: it minimised the pre-registered
four-cell objective, and the biggest error available to it was the blowout cell,
not the close one.

**The simpler form reaches the close cell and loses everything else.** R8's
`theta` grid *does* reach b0 = 0.7475 at `theta = 1.0`, above the training target
-- but with b1 stuck at 0.6715, b2 at 0.5940 and foul trouble at 0.4818, so the
joint objective selects `theta = 0.1` and R8 is R5 with a rounding error
(b0 +0.4 pp, b1 +0.3 pp on the test set). A per-cell keep rate cannot separate
"close and late" from "blown out and late" because `s*` is a level, not a
contrast, so every unit of theta that helps b0 hurts b2 by nearly as much.

**Neither keep arm clears noise floor B.** R7's improvement over R5 is +1.3 pp on
b0 and +1.8 pp on b1; the spec-identical refit under a second seed moves the same
cells by 1.5 pp and 1.4 pp. By the pre-registered rule (6.8) an improvement that
does not clear the refit-to-refit spread on the cell it was built to fix is not
adopted on that cell, and R7's b0 gain does not. R8 is worse than that: its
single knob is **unstable across refits** (`theta` 0.1 under fit seed 11 and 0.3
under fit seed 101) and its b0 moves 6.1 pp between the two fits, which is twice
the gate tolerance. R8 is not a usable component at any setting.

**What the donor family actually broke on, once the foul terms work.** The
binding failure of R5, R7 and R8 is no longer the close band; it is the blowout
band and foul trouble, both of which the corrected hazard matrix moved the wrong
way (b2 -2.9 pp under the defect, +6.9 pp corrected; `>= 4 fouls` +0.7 pp under
the defect, +3.5 pp corrected, with the "exactly 4 fouls" diagnostic at
0.82 against a real 0.52). The corrected foul terms take load off the margin
terms, the block stops emptying the bench when the game is decided, and the arm
now plays its starters through garbage time. Section 5's OPEN item is closed by
this run and its answer is that the corrected R5 is **worse**, not better, than
the column section 4 reported.

**What is left standing, and it is not new.** `R2_hier_dirichlet` reproduces its
round-1 and round-2 numbers to four decimals for the third time (late
0.7281 / 0.6943 / 0.4750, top-5 0.7683, minutes mean 25.55) and remains the only
arm in three rounds with three of four state cells inside tolerance, failing only
the blowout band at -4.7 pp. Nothing in round 3 displaces it and nothing in
round 3 is adopted.

**Two things round 3 did buy, recorded so the next round does not re-derive
them.** (1) R7 has the best lineup concentration of any arm across all three
rounds -- top-1 five-man lineup share 0.2888 against a real 0.2940, K-S D
**0.0302**, and the best per-player minutes K-S of any arm (0.0782 against R2's
0.0883). Whatever eventually wins on state dependence, this is the shape to
match. (2) Every arm's quintile slope against the pregame team prior is right
(actual +0.692; R2 +0.645, R5 +0.742, R8 +0.728, R7 +0.826), so the failures are
level failures and not responsiveness failures -- the standing slope check is
satisfied by all four arms and cannot be used to separate them.

**The decomposition the gate does not do, reported and not used to soften it.**
Re-grading the actual sequence with the model's as-of starter set (7.2) puts the
benchmark at 0.7215 / 0.6964 / 0.5048 / 0.4636. Against that benchmark R7 is
-1.4 pp on the close band and -2.2 pp on the moderate band -- both inside the 3
pp tolerance -- while its blowout miss grows to +9.5 pp. So roughly 2.8 pp of
R7's close-band miss is picking the wrong fifth starter, which needs availability
information the as-of feature set does not have, and the rest is real. **The gate
is scored as pre-registered, against the actual starting five, and R7 fails it.**

---

## 8. Round 3b pre-registration -- S1 scheme confirmation (PM-directed, worker-authored 2026-09-10)

Written BEFORE the round-3b run and after round 3 decided. Provenance, stated
plainly: the PM's note that **S1 is the standing default training scheme for
every sub-model** (L21, `docs/models/README.md`) arrived while round 3 was
already committed (`experiments.md` section 6, commit `4f6e61f`) and running, so
per that note round 3 was **not** changed mid-run; round 3b is the confirmation
it prescribes. Round 3's static columns stay the record of round 3.

### 8.1 What 3b asks

Rounds 1-3 fit once on 2024 and simulate all of 2025 from that fit ("static").
S1 refits at each month boundary of the test season on every prior season plus
the test season to date, strictly before the refit date, and simulates each game
with the parameter set whose window closed before its tipoff. 3b asks one
question: **does the scheme change any round-3 conclusion?**

Round 3 adopted nothing, so there is no winner to refit. 3b therefore runs the
three arms whose conclusions could move:

| arm | why it is in 3b |
|---|---|
| `R2_hier_dirichlet` | the incumbent and the only arm in three rounds with 3 of 4 state cells inside tolerance; if S1 moves its blowout band by more than the floor, the round-1/2/3 verdict changes |
| `R5_hybrid` | the corrected donor baseline the keep arms are measured against |
| `R7_keep_logistic` | round 3's best keep arm and the best lineup concentration of any arm in three rounds |

R8 is excluded and the exclusion is a result, not a convenience: its single knob
moved from `theta` 0.1 to 0.3 between two spec-identical refits and its close
band moved 6.1 pp (section 7.5), so an S1 column for R8 would be measuring knob
instability, not a scheme.

### 8.2 Windows, and the leak rule

Windows are the calendar months the 2024-25 season's games fall in. For window
starting at date `d`, training data = all of season 2024 plus every 2025 game
with `game_date < d`; test games are those with `game_date >= d` and before the
next window. The first window's training data is season 2024 alone, which **is**
the static fit, so the static fit is reused there rather than refitted -- an S1
run whose first window differed from the static fit would be measuring two things
at once. Every fitted object is therefore strictly pregame with respect to every
game it scores (`created_at < tipoff`), and the donor bank is unchanged because
`build_donor_bank` already uses only a team's own strictly earlier games.

### 8.3 What is refit per window

Everything `train_rotation_v1.fit_all` fits (role prior, `k0`, `w_dnp`, tail,
`min_share`, the starter-predictor decay, availability, `fpm`, the four Dirichlet
concentrations, both tilt tables, the foul-rate scale, the scheduler grid, the R3
hazards), plus R5's override hazards, plus everything round 3 adds (`s*`, the R7
on-floor propensity, and both knob pairs by the same two-pass coordinate descent
against that window's own state cells). The donor depth `k` is held at its
round-2 fitted value of 3 and that is stated as a deviation: it is fitted by
lineup-concentration error over a whole season and a one-month window cannot
re-fit it honestly.

### 8.4 Gates, floor and decision

Gates, tolerances, the eligibility veto and the grading code are **unchanged**
from sections 6.5 and 6.8 -- one blind path, `train_rotation_v1.build_row` /
`verdict`. Test universe identical (the same 1,600 games, subset seed 2025).
Seeds are 3 rather than 5, a compute concession stated here: round 3's noise
floor A puts the seed SD of the state cells at 0.009-0.024 on 150 games, so on
1,600 games at 3 seeds the Monte-Carlo standard error of every state cell is
under 0.15 pp, twenty times smaller than the 3 pp tolerance.

Floor: round 3's noise floor A, per arm and per cell. A cell counts as MOVED by
the scheme only if `|S1 - static|` exceeds that cell's floor.

Decision: **adopt S1 as the rotation model's training scheme unless a gate cell
regresses beyond the floor.** If a cell regresses beyond the floor, report which
and by how much and leave the scheme open for the PM. 3b cannot adopt a model --
round 3 adopted none -- it fixes the scheme round 4 is run under.

### 8.5 Persistence and naming

Each window's parameter set is written to
`data/processed/models/rotation/rotation_fit_v3_S1_{YYYYMM}.json`, where `YYYYMM`
is the window's **start** month, so the engine selects a game's parameter set by
taking the largest `YYYYMM` less than or equal to the game's month. The static
fit remains `rotation_fit_v3.json` and is byte-identical to the first window's
file. Nothing overwrites `rotation_fit.json`, which the engine currently reads.

---

## 9. Round-3b results -- S1 vs static (train 2024 + 2025-to-date, test 2025) -- run 2026-09-11T00:49:35Z

Test universe identical to rounds 2 and 3 (the same 1,600 games, subset seed
2025), 3 seeds, one blind grading path. Six S1 windows, `d(pp)` = S1 minus
static in the cell's own units x 100, `floor` = round 3's noise floor A for that
arm and cell. A cell is **MOVED** by the scheme only if `|d| > floor` (section
8.4).

Windows and the games each scored: 202411 147, 202412 297, 202501 438, 202502
444, 202503 267, 202504 7 (the last is UNDERPOWERED at 7 games and is reported
only because it exists). Window 202411 reuses the static fit by construction.
Refit cost 7.7-13.0 min per window on four threads; 61.4 min for the whole run.

### 9.1 Gate cells, S1 against static

| cell | ACTUAL | R2 static | R2 S1 | d(pp) | floor | R5 static | R5 S1 | d(pp) | floor | R7 static | R7 S1 | d(pp) | floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7491 | 0.7281 | 0.7254 | -0.27 | 0.88 | 0.6942 | 0.6871 | -0.70 | 1.05 | 0.7074 | 0.7065 | -0.09 | 1.10 |
| final 8:00, \|m\| 6-15 | 0.7240 | 0.6943 | 0.6914 | -0.29 | 1.06 | 0.6566 | 0.6551 | -0.15 | 1.11 | 0.6749 | 0.6687 | -0.61 | 0.88 |
| final 8:00, \|m\| > 15 | 0.5223 | 0.4750 | 0.4694 | -0.56 | 1.36 | 0.5911 | 0.5830 | -0.81 | 1.38 | 0.5994 | **0.5679** | **-3.15** | 1.21 |
| starters at >= 4 fouls | 0.4613 | 0.4648 | 0.4626 | -0.21 | 2.14 | 0.4964 | 0.4907 | -0.57 | 2.01 | 0.5125 | 0.5184 | +0.59 | 2.43 |
| top-5 share | 0.7472 | 0.7683 | **0.7630** | **-0.53** | 0.31 | 0.7965 | 0.7948 | -0.16 | 0.32 | 0.7828 | 0.7814 | -0.14 | 0.29 |
| top-8 share | 0.9560 | 0.9656 | **0.9626** | **-0.30** | 0.14 | 0.9847 | 0.9839 | -0.08 | 0.14 | 0.9814 | 0.9800 | -0.15 | 0.14 |
| players > 0 min | 9.644 | 9.039 | **9.129** | **+0.089** | 0.058 | 8.298 | 8.350 | +0.052 | 0.068 | 8.514 | **8.599** | **+0.084** | 0.064 |
| minutes mean | 24.551 | 25.553 | **25.364** | **-0.189** | 0.154 | 27.058 | 26.988 | -0.069 | 0.218 | 26.284 | **25.863** | **-0.421** | 0.179 |
| top-1 lineup share | 0.2940 | 0.2369 | **0.2286** | **-0.83** | 0.56 | 0.3159 | 0.3138 | -0.20 | 0.65 | 0.2888 | 0.2914 | +0.26 | 0.52 |
| K-S per-player minutes (D) | -- | 0.0883 | 0.0798 | -0.85 | -- | 0.1194 | 0.1159 | -0.36 | -- | 0.0782 | **0.0686** | -0.96 | -- |
| K-S top-1 lineup share (D) | -- | 0.1942 | 0.2273 | +3.31 | -- | 0.1047 | 0.0942 | -1.05 | -- | 0.0302 | 0.0288 | -0.15 | -- |
| substitution rate | 0.1518 | 0.1402 | 0.1437 | +0.35 | -- | 0.1576 | 0.1585 | +0.08 | -- | 0.1691 | 0.1682 | -0.09 | -- |

**Every cell that MOVED moved toward the actual, and every state cell that did
not move stayed inside its floor.** The moves are: R7's blowout band -3.15 pp
against a 1.21 pp floor (0.5994 -> 0.5679 against 0.5223, a real improvement that
still fails the 3 pp gate at +4.6 pp); R2's top-5 share, top-8 share, nonzero
count and minutes mean, all four toward the actual and all four beyond their
floors; R7's nonzero count and minutes mean likewise. Nothing regressed beyond a
floor.

### 9.2 Three verdict flips, all inside the floor, reported as flips anyway

| arm | cell | static | S1 | flip | \|d\| | floor |
|---|---|---:|---:|---|---:|---:|
| R2 | top-5 share | 0.7683 (+2.11 pp) | 0.7630 (+1.58 pp) | FAIL -> **PASS** | 0.53 | 0.31 |
| R2 | final 8:00, \|m\| 6-15 | 0.6943 (-2.97 pp) | 0.6914 (-3.26 pp) | PASS -> **FAIL** | 0.29 | 1.06 |
| R5 | starters at >= 4 fouls | 0.4964 (+3.51 pp) | 0.4907 (+2.94 pp) | FAIL -> **PASS** | 0.57 | 2.01 |

Two of these are cells sitting on the tolerance line, not scheme effects: R2's
6-15 band was at **-2.97 pp against a -3.00 pp tolerance** under the static fit,
so a 0.29 pp move -- a quarter of that cell's own seed noise -- flips it. Read
honestly this says R2's moderate band was never really inside tolerance, which
qualifies the "R2 is 3 of 4" headline of rounds 1-3: under S1 it is **2 of 4**,
and under either scheme the cell is indistinguishable from the boundary. R5's
foul-trouble flip is the same phenomenon in the other direction (+3.51 -> +2.94
against +3.00). Only R2's top-5 flip is backed by a move larger than the floor.
No arm becomes eligible: R2 still fails the blowout band (-5.29 pp under S1),
and R5 and R7 still fail three cells each.

### 9.3 The knobs are not identified across windows

| window | R7 block (scale, p0) | R7 keep (scale, q0) | R8 theta |
|---|---|---|---:|
| 202411 | (2.0, 0.02) | (1.0, 0.02) | 0.1 |
| 202412 | (4.0, 0.005) | (2.0, 0.05) | 0.1 |
| 202501 | (4.0, 0.02) | (0.5, 0.02) | 0.5 |
| 202502 | (2.0, 0.02) | (2.0, 0.02) | 0.1 |
| 202503 | (2.0, 0.02) | (0.0, 0.15) | 1.0 |
| 202504 | (2.0, 0.005) | (1.0, 0.05) | 0.3 |

The keep scale ranges over the whole grid from 0.0 to 2.0 and `q0` from 0.02 to
0.15 across six windows of the same season, and R8's single knob spans 0.1 to
1.0. This is the same instability noise floor B found between two spec-identical
static refits (section 7.5), now visible a second way and on a second axis. It is
a property of the override family, not of the scheme: the four-cell objective is
nearly flat in the knobs over most of the grid (section 7.7), so the argmin moves
with the sample. **A component whose fitted knob is not identified cannot be
shipped even if a future round makes its cells pass**, and that is the strongest
single result of round 3.

### 9.4 Decision

**S1 is adopted as the L4 rotation model's training scheme.** By the
pre-registered rule (8.4) no gate cell regressed beyond its floor, six cells
improved beyond their floors, and every improvement moved toward the actual. S1
does not make any arm eligible and 3b adopts no model -- round 3 adopted none and
3b cannot change that. What 3b fixes is the scheme round 4 is run under.

Practical consequences recorded for round 4 and for the engine:

1. Round 4 fits and grades under S1 by default, with the static column reported
   alongside as rounds 1-3 were.
2. Parameter sets are persisted per window as
   `rotation_fit_v3_S1_{YYYYMM}.json`, named by the window's **start** month; a
   game selects the largest `YYYYMM` at or before its own month. Six files exist
   for 2024-25. `rotation_fit.json`, which the engine currently reads, is
   untouched.
3. The engine's rotation adapter currently loads **one** fit for a whole run
   (`ad.rot_fit`). Running S1 in the engine needs the adapter to select a fit per
   game by month. Because every fitted object S1 varies (tilt tables, Dirichlet
   concentrations, `p_play`, `fpm`, the scheduler parameters) is a lookup table
   or a scalar, this is a gather, not a new decision rule -- the cheapest change
   in this whole round.
4. S1 costs 7.7-13.0 min per window to refit on four threads, so a full-season
   S1 fit of the rotation layer is about an hour. That is affordable and is
   recorded so the next pre-registration budgets it.

---

## 10. Round 4 pre-registration -- a new model family (PM-directed, worker-authored 2026-09-10)

Written and committed BEFORE any round-4 arm was run. Evidence it is built on:
`docs/tests/rotation_sub_hazard_audit_2026-09-10.md`, also written and committed
before this section.

### 10.1 Why round 4 changes family rather than knobs

Three rounds cannot produce two structural facts of a real rotation (L25):

* the second half opens at **0.90** starter share in every margin band over the
  H2 20:00-16:00 window (and **0.95-0.97** at the tip possession itself); no arm
  in three rounds exceeded 0.80 over the window and none was ever measured at
  the tip;
* starters are **kept** late in a close game (0.7491 in the final 8:00 at
  \|margin\| <= 5); the best arm reaches 0.7281 and the override family's
  reachable maximum over its whole 24-point knob grid is **0.6994**, with the
  knob running the wrong way past scale 1.

Rounds 1-3 were two families. (a) Dirichlet-share plus scheduler: draw each
player's minutes BUDGET, then place it. (b) Donor resampling plus overrides:
replay a real lineup sequence, then block or keep players on top of it. Both
decide *how much* a player plays and infer *when*; a coach decides, at a
stoppage, whom to take off and whom to bring on, and the minutes are the
consequence. Round 4 models that decision directly, as **per-player
discrete-time substitution hazards**.

The PM's direction for round 4, not to be reopened: change family; carry the two
structural facts as pre-registered gate cells; keep every round-3 gate unchanged.

### 10.2 The family, and the one decision rule every arm shares

At every possession boundary, for the five on the floor and for every eligible
bench candidate:

    p_out(i) = sigma(x_i . w_out)      i on the floor at the previous possession
    p_in(j)  = sigma(x_j . w_in)       j eligible and off the floor

Exits are independent Bernoulli draws. The five-on-the-floor constraint is then
imposed by taking exactly `n_exit` entrants from the bench in an
Efraimidis-Spirakis exponential race weighted by `p_in / (1 - p_in)` -- weighted
sampling without replacement, expressible as one uniform per roster slot plus an
argsort, which is why it is the same rule offline and in the engine adapter. A
player with five fouls leaves before anyone else and can never enter.

**Features (45, `rotation_v4.SUB_FEATURES`).** Player: `is_starter`, as-of
`share` of the team's five on-floor slots, `fouls`, `foul_out`,
`fouls x is_starter`, `state_min` (minutes in the current on/off state),
`half_min` (minutes played so far in this half), `half_min_dev` (`half_min`
minus `share x` elapsed half minutes). State: `sec_left_frac`, `is_ot`,
`team_fouls_frac`, and four `prev_end` dummies (`period_boundary`,
`dead_made_ft`, `dead_tov`, `dead_other`) which are the dead-ball opportunity.
Cells: eight time-cell dummies (the nine audit cells, `H1 20:00-10:00` the
reference) and two margin-band dummies. Interactions: `is_starter` with each
time-cell dummy, each margin-band dummy, `period_boundary`, `sec_left_frac`,
`is_close x late`, `is_blowout x late` and `abs_margin`; `share` with `late`,
`mb>15` and `period_boundary`; `fouls x late`; `abs_margin`.

**The design is deliberately saturated in (time cell x margin band x
is_starter), and that is declared here rather than discovered later.** The audit
shows a starter's exit hazard running 0.035 -> 0.020 -> 0.054 -> 0.027 -> 0.037
across the nine cells (non-monotone), a bench player's exit hazard jumping to
0.244 in H2 20:00-16:00, and the starter/bench ordering reversing sign in the
final two minutes of a blowout. A linear time term cannot represent that shape.
The gate reads OCCUPANCY in some of those cells, which is a different functional
of the process -- an equilibrium the hazards must produce under the
five-on-the-floor constraint, not a quantity fitted here -- and section 10.9's
reachability numbers are what make that claim falsifiable rather than rhetorical.

**Two features are measured in the audit and EXCLUDED, with the numbers.**
(a) A timeout indicator multiplies every hazard by 3-4x (starter exit 0.031 ->
0.136, starter entry 0.068 -> 0.243) and is excluded because the engine has no
timeout model, so a hazard conditioned on it could not be evaluated in
simulation -- the same class of exclusion as `is_transition` in `features.md`
section 3. (b) Prior-season minutes share has a real gradient (P5 - P1: -0.9 pp
on starter exit, +1.8 pp on starter entry) which is same-signed and about half
the size of the as-of share already in the design, and it is not expressible in
the engine without a new per-roster-slot input array, i.e. without rebuilding
`arrays_F2_2025.npz` while other workers read it.

### 10.3 Arms

| arm | what it is | simplicity |
|---|---|---:|
| `R2_hier_dirichlet` (static) | the incumbent under rounds 1-3's scheme | 1 |
| `R2_hier_dirichlet` (S1) | the incumbent under the adopted scheme; **the reference** | 1 |
| `H1_sub_hazard` | logistic hazards + a HARD reset to the predicted starting five at the first possession of period 2 | 2 |
| `H2_sub_hazard_noreset` | the same hazards, NO hard reset -- the reset must be EARNED by the `period_boundary` terms | 3 |
| `H3_sub_hazard_lgbm` | H1 with a LightGBM hazard (300 rounds, 31 leaves, `min_data_in_leaf` 200) in place of the logistic | 4 |

H1 and H2 differ in exactly one line of `run_sub_hazard`. H1 and H3 differ in
exactly one fitted object. That is deliberate: each arm isolates one question,
and H2 is the honest test of whether a fitted hazard produces a structural fact
without being told it.

The hard reset uses the model's own predicted starting five among the eligible,
at the first possession of period 2 only; overtime is left to the hazards.

### 10.4 Scheme and folds

**Scheme: S1 for every arm, with the static column reported alongside**, per
round 3b's decision (section 9.4). Windows are the calendar months of the
2024-25 season; a game uses the parameter set whose window closed before its
tipoff; the first window's training data is 2024 alone, which IS the static fit.

**Folds.** As in round 3 (section 6.2) and for the same reason: CBBD carries no
on-floor data at all before 2023-24 (0.0000 of possessions in 2022 and 2023,
L13), so `CLAUDE.md`'s fold 1 has no training season and fold 2 is the only fold
that exists. Round 4 runs **F1 = train 2024, test 2025**, which *is* the standing
fold 2. **This is the selection fold and the only one.** Within-2025
walk-forward is the robustness check and is reported only if an arm is otherwise
adoptable; it is not a selection metric. 2026 stays sealed
(`seal.assert_not_sealed` guards the trainer).

**Base fits are reused, not refitted.** `rotation_fit_v3.json` and the six
`rotation_fit_v3_S1_{YYYYMM}.json` written by round 3b are the base parameter
sets (role prior, shrinkage, availability, foul rate, the tilt tables and the
scheduler grid R2 needs). Round 4 fits ONLY the two hazards, per window, on that
window's own training data. Consequence, stated so it cannot be read as a
coincidence: the round-4 `R2 S1` column will be round 3b's `R2 S1` column, and
the round-4 `R2 static` column will be rounds 1-3's, so a difference between
rounds cannot be a difference in R2's fit. Nothing is written to any of those
files.

Hazard training rows come from `rotation_v4.build_sub_training` over the **as-of
candidate pool**, with the training game's own `PersonalFoul` events supplying
the foul state -- legitimate at fit time, where the label is that same game's
substitution, and never at simulation time (the rule established by section 5).
800 team-games are sampled per window (fit seed 11), giving roughly 550k out
rows and 1.09M in rows.

### 10.5 Test universe and grading path

The **same** 1,600-game subset of 2025 rounds 2, 3 and 3b used (numpy
RandomState seed 2025), 3 seeds per arm under S1 and 3 static, one blind grading
path: `train_rotation_v1.build_row` / `verdict` / `rotation.aggregate_stats`,
extended by `train_rotation_v4.extra_cells` for the two new cells and by
`train_rotation_v4.minutes_mae` for the primary metric. Sim and actual go
through the identical functions. Any cell with n < 300 player-games or
possessions is labelled UNDERPOWERED.

### 10.6 Gates -- every round-3 gate unchanged, plus two new cells

**G8 cells (report, not veto), unchanged:** minutes mean +/- 2.0; minutes SD
ratio pooled and within-player 0.9-1.1; top-5 and top-8 share of team minutes
+/- 2 pp; players with > 0 minutes +/- 1.0.

**State-dependence cells (the veto), the four from round 3:** starters' share of
on-floor slots in the final 8:00 at \|m\| <= 5 / 6-15 / > 15, and starters' share
while carrying >= 4 fouls, each **+/- 3 pp**. The "at exactly 4 fouls" diagnostic
is reported alongside, because section 5 showed the >= 4 cell can pass for the
wrong reason.

**Two NEW state cells, same +/- 3 pp tolerance, same veto status:**

| new cell | definition | why |
|---|---|---|
| `h2tip_starter_share_b{0,1,2}` | starters' share of the five at the **first possession of period 2**, by margin band | the audit's largest conditional probability (0.90 at a period boundary) and the fact three rounds could not produce; measured at the tip, not over the four-minute window, because the audit shows it is a point event that decays |
| `opentip_starter_share_close` | starters' share of on-floor slots over **H1 20:00-10:00 at \|m\| <= 5** | R2 is -17.0 pp here (round-3 audit section 7.3) and no gate saw it |

So the veto is **eight** state cells in total: the 4 from round 3, the 3
second-half-tip bands, and the opening tip. **An arm missing ANY of the eight is
ineligible regardless of G8 or of MAE.**

**Lineup concentration (report):** top-1 / top-3 / top-5 five-man lineup share,
distinct lineups per team-game, K-S D of the top-1 lineup share distribution,
K-S D of per-player minutes, substitution rate at a possession boundary.

**Decision 8 slope check (a condition on adoption):** team-games bucketed into
quintiles of the pregame as-of share of team minutes going to the predicted
starting five; the close-and-late cell reported per quintile for ACTUAL and every
arm, with the fitted slope and Q5 - Q1. An arm whose profile is flat, or whose
slope sign disagrees with actual, is reported as not matchup-specific whatever
its pooled cells say. The audit's whole-season reading is slope **+0.494**,
Q5 - Q1 **+14.0 pp**, monotone 4 of 4.

### 10.7 Primary metric

**Per-player minutes MAE**: outer join of simulated and actual minutes on
(game_id, team_id, pid) over the as-of rotation set (players with as-of
`mpg >= 10`), so a player the arm never plays and a player the arm invents both
count their full minutes as error; computed per seed and averaged over seeds.
Reported for every arm; used by the decision rule in 10.8.

### 10.8 Noise floor and the decision rule

**Floor A, seed-varied sim runs:** 20 seeds x 150 games per arm, the SD of every
gate cell and of the MAE (the round-3 configuration, so the two rounds' floors
are comparable).

**Floor B, spec-identical refit under a second seed:** the whole of 10.4 re-run
with a different training-game sample (fit seed 101 vs 11), a different logistic
`random_state` and a different sim seed (23 vs 7), graded on the same 150-game
universe. An arm counts as beating the reference on a cell only if its
improvement exceeds the refit-to-refit spread on that cell. This is run on the
**new family's reference arm (H1)**, because H1 is the object round 4 adds; R2's
own floor is round 3's noise floor A, already on record and unchanged by a run
that does not refit it.

**Decision rule.** Adopt the **simplest** arm that

1. passes **every** one of the eight state cells at +/- 3 pp, AND
2. beats `R2_S1` on per-player minutes MAE by more than the floor (the incumbent
   is exempt from this clause, being the reference), AND
3. satisfies the Decision 8 slope check, AND
4. passes the Decision 10 closed-loop check of 10.10.

Ties go to the simpler model in the order **R2 < H1 < H2 < H3**. An arm whose
improvement on the cell it was built to fix does not clear floor B is not
adopted on that cell. **If no arm is eligible, adopt nothing**, report which cell
fails and by how much, and name the diagnosis. No gate is relaxed to produce a
winner and no cell is dropped after seeing a result.

### 10.9 The L25 reachability check, computed BEFORE this section was written

L25: *"before fitting any knob, compute the fitted log-odds separation between
the classes the knob must move apart in the target state and report the target
cell's reachable range over the whole grid; if the target is outside it, the
family is wrong and no fitting will find it."*

The hazard family's analogue of "the whole grid" is its **saturated cell form**:
hazards indexed by (is_starter x time cell x margin band x period-boundary flag),
the most any per-player hazard conditioned on those variables can know. Fitted on
2024 and run forward on 400 random 2025 games under the five-on-the-floor
constraint with the same race selection rule
(`scripts/diag_rotation_sub_hazard.py --mode reach`; audit section 8):

| target cell | actual, same 400 games | family on paper | miss |
|---|---:|---:|---:|
| H2 tip, \|m\| <= 5 | 0.9732 | 0.9758 | +0.3 pp |
| H2 tip, \|m\| 6-15 | 0.9574 | 0.9670 | +1.0 pp |
| H2 tip, \|m\| > 15 | 0.9508 | 0.9525 | +0.2 pp |
| final 8:00, \|m\| <= 5 | 0.7388 | 0.7393 | +0.05 pp |
| final 8:00, \|m\| 6-15 | 0.7135 | 0.7031 | -1.0 pp |
| final 8:00, \|m\| > 15 | 0.5120 | 0.5201 | +0.8 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0.7722 | 0.7741 | +0.2 pp |

Against the override family's reachable maximum of 0.6994 on the close band
(section 7.9). The starter-vs-bench separation in the close-and-late state is
**0.98** in log odds on the exit risk set and **1.30** on the entry risk set --
two separations acting in opposite directions on two different risk sets, whose
equilibrium sets the cell, rather than R7's single 1.0 separation with one scale
knob.

**What the probe does not establish, and it is a real caveat:** it uses the real
starting five and the real participant pool. A bake-off arm's starter set
overlaps the real one on 4.58 of 5, and round 3 measured the cost -- the actual
sequence re-graded with the model's as-of starter set puts the close-late
benchmark at 0.7215, 2.8 pp below the 0.7491 the gate scores against. The family
is reachable on paper and the arms must still find 2-3 pp the as-of starter set
does not have. The gate is scored as pre-registered, against each side's own
starting five.

### 10.10 Decision 10: the closed-loop check

Every round-4 arm consumes `margin` and `fouls`, both of which the engine itself
produces, so Decision 10 applies in full. `R2` consumes them too, through
`TiltTables.state` and `TiltTables.foul`, and **has never had a closed-loop
gate** (L25), so one is run for it as well.

`ENGINE_ROTATION_FREEZE=1` is added to `engine/rotation_adapter.py` and
`engine/loop.py`. It holds, **for the rotation model only**, the margin at its
pregame value (0) and the personal-foul and team-foul counts at theirs (0), while
leaving the foul accrual, the foul-out eviction rule and the box-score foul
counters live -- a player with five fouls still leaves the floor, because that is
a rule of the game and not a model feature. Nothing else in the engine changes.

The check is a paired-stream run of **5 seeds over a fixed 500-game subset** of
the F2 2025 universe, live vs frozen, reporting margin SD ratio, home/away score
correlation, possessions per game and per-player minutes MAE. Sub-model flags are
pinned on every run so all arms share the same cascade regardless of when they
ran: `ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round2b_S_C_s1`, and both are
recorded in the results. A winner that moves margin SD ratio, home/away
correlation or possessions outside the G1/G2 tolerances between live and frozen
is not adopted.

### 10.11 Engine expressibility (a condition on adoption)

Every feature above is computable from state the possession loop already carries
at a boundary: `period`, `seconds_remaining`, `home_score_diff`, `prev_end` (the
six `PREV_END_LEVELS` codes, which are exactly the possessions table's
`start_reason`), `team_fouls`, the (2N, S) personal-foul block, and rotation
state the adapter owns. `rotation_v4.design()` is written over (M, S) arrays and
is called with M = 1 offline and M = 2N in the engine, so the offline sampler and
the adapter cannot drift apart. The winner ships behind `ENGINE_ROTATION=round4`
with S1 artifacts persisted per month and a manifest in the
`engine/manifest.py` format (`refit_date`, `path`, `max_train_date`). The two
stated RNG divergences of `docs/models/engine/model.md` section 4.5 apply
unchanged; the race draw is one uniform per roster slot and becomes a third such
stream. **A LightGBM hazard (H3) is a live model call in the sim loop, which
`CLAUDE.md` bans**; if H3 wins, the adapter cost is a lookup-table discretisation
of the booster and that is reported as a condition of adoption rather than
waived.

### 10.12 Disclosures

1. A **60-game, 1-seed development smoke run** (`--smoke`, artifacts named
   `rotation_F1_round4_SMOKE_*`) was executed before this section was written, to
   verify the code path end to end. It printed gate cells on 2025. Its numbers
   are not evidence, are not cited anywhere, and **no model specification was
   changed after seeing them** -- the 45-feature design was fixed from the 2024
   hazard tables and its faithfulness was checked on the 2024 training rows only
   (that calibration table is reported with the results).
2. The audit reports 2024 and 2025 hazard tables side by side. Feature choices
   were taken from the **2024** tables; the 2025 columns exist for the
   season-agreement check that round 3's audit section 7.5 also ran, and no
   fitted object sees 2025 data outside its own S1 window.
3. Base fits are reused from round 3b rather than refitted (10.4), which is a
   deviation from "refit everything per window" and is stated as one.

---

## 11. Round-4 results (train 2024, test 2025) -- run 2026-09-10T23:57Z

Test universe: the **same** 1,600-game subset of 2025 rounds 2, 3 and 3b used
(numpy RandomState seed 2025), 3 seeds per arm under S1 and 3 static, one blind
grading path. 9,600 simulated team-games and 68,859 rotation player-games per
arm against 3,200 actual team-games and 23,914 actual rotation player-games.
Every state cell is far above the n < 300 UNDERPOWERED threshold: final-8:00
slots 29,660 / 36,804 / 23,360, foul-trouble opportunities 58,652, second-half
tip slots 6,330 / 7,360 / 2,310, opening-ten-minutes slots 414,680.

Six S1 windows and the games each scored: 202411 147, 202412 297, 202501 438,
202502 444, 202503 267, 202504 7 (the last is UNDERPOWERED at 7 games and is
reported only because it exists). Window 202411's training data is 2024 alone,
which IS the static fit. Base fits reused from round 3b per section 10.4; only
the two hazards are fitted here, on 800 team-games per window (fit seed 11):
547,450 out rows at base rate 0.04275 and 1,091,977 in rows at base rate
0.02135 for the static fit.

### 11.1 Does the fitted hazard reproduce the hazards the audit measured? (training season only)

Pre-registered in 10.12 as the faithfulness check, computed on the **2024
training rows** and on nothing else. Actual / predicted, starter | bench:

| time cell | OUT \|m\|<=5 | OUT 6-15 | OUT >15 | IN \|m\|<=5 | IN 6-15 |
|---|---|---|---|---|---|
| H1 20:00-10:00 | 0.0312/0.0337 \| 0.0394/0.0421 | 0.0534/0.0441 \| 0.0484/0.0463 | 0.0432/0.0503 \| 0.0667/0.0431 | 0.0548/0.0581 \| 0.0112/0.0116 | 0.0659/0.0641 \| 0.0169/0.0130 |
| H1 10:00-00:00 | 0.0374/0.0344 \| 0.0696/0.0677 | 0.0367/0.0388 \| 0.0702/0.0708 | 0.0329/0.0440 \| 0.0687/0.0623 | 0.0751/0.0736 \| 0.0136/0.0120 | 0.0733/0.0737 \| 0.0138/0.0142 |
| H2 20:00-16:00 | 0.0169/0.0185 \| 0.1555/0.1634 | 0.0212/0.0202 \| 0.1556/0.1573 | 0.0229/0.0205 \| 0.1219/0.1114 | 0.1502/0.1567 \| 0.0081/0.0083 | 0.1523/0.1522 \| 0.0099/0.0093 |
| H2 16:00-12:00 | 0.0516/0.0481 \| 0.0398/0.0418 | 0.0506/0.0542 \| 0.0412/0.0422 | 0.0598/0.0566 \| 0.0404/0.0356 | 0.0570/0.0574 \| 0.0159/0.0138 | 0.0578/0.0602 \| 0.0160/0.0159 |
| H2 12:00-08:00 | 0.0342/0.0331 \| 0.0638/0.0639 | 0.0325/0.0353 \| 0.0619/0.0625 | 0.0442/0.0381 \| 0.0525/0.0527 | 0.0827/0.0787 \| 0.0092/0.0086 | 0.0746/0.0788 \| 0.0097/0.0097 |
| H2 08:00-04:00 | 0.0256/0.0223 \| 0.0701/0.0646 | 0.0258/0.0209 \| 0.0651/0.0659 | 0.0338/0.0479 \| 0.0534/0.0544 | 0.0826/0.0996 \| 0.0090/0.0087 | 0.0683/0.0674 \| 0.0101/0.0105 |
| H2 04:00-02:00 | 0.0204/0.0259 \| 0.0627/0.0541 | 0.0206/0.0259 \| 0.0619/0.0576 | 0.0769/0.0580 \| 0.0368/0.0450 | 0.0810/0.0755 \| 0.0068/0.0110 | 0.0587/0.0527 \| 0.0095/0.0142 |
| H2 02:00-00:00 | 0.0330/0.0363 \| 0.0738/0.0482 | 0.0341/0.0387 \| 0.0567/0.0591 | 0.0970/0.0704 \| 0.0277/0.0377 | 0.0856/0.0614 \| 0.0137/0.0141 | 0.0589/0.0563 \| 0.0145/0.0191 |

At a **period boundary**: OUT starter 0.0399 actual / 0.0383 predicted
(n = 2,859), OUT bench **0.7909 / 0.7856** (n = 1,521); IN starter
**0.7801 / 0.7707** (n = 1,501), IN bench 0.0200 / 0.0230 (n = 7,236).

**The fit is faithful.** The saturated design reproduces every cell it was given
to within about 1 pp, including the reset row the whole round turns on. So
anything the arms miss below is a property of the *sampler* -- of how hazards
become occupancy under the five-on-the-floor constraint -- and not of the fit.

### 11.2 G8 cells (report, not veto) -- S1

| cell | tol | ACTUAL | R2_hier_dirichlet | H1_sub_hazard | H2_sub_hazard_noreset | H3_sub_hazard_lgbm |
|---|---|---:|---:|---:|---:|---:|
| minutes mean (rotation players) | +/- 2.0 | 24.55 | 25.36  PASS | 25.99  PASS | 25.96  PASS | 25.75  PASS |
| minutes SD ratio, pooled | 0.9-1.1 | 1.0000 | 1.0548  PASS | 0.9117  PASS | 0.9074  PASS | 0.8562  FAIL |
| minutes SD ratio, within-player | 0.9-1.1 | 1.0000 | 1.3416  FAIL | **1.0792  PASS** | **1.0804  PASS** | **1.0219  PASS** |
| top-5 share of team minutes | +/- 2 pp | 0.7472 | 0.7630  PASS | 0.7583  PASS | 0.7566  PASS | 0.7457  PASS |
| top-8 share of team minutes | +/- 2 pp | 0.9560 | 0.9626  PASS | 0.9778  FAIL | 0.9776  FAIL | 0.9771  FAIL |
| players with > 0 minutes | +/- 1.0 | 9.64 | 9.13  PASS | 8.64  FAIL | 8.63  FAIL | 8.50  FAIL |

The within-player minutes SD ratio has **failed in every round for every arm**
since round 1 (1.296 / 1.353 / 1.117 / 1.051 in round 1, 1.354 / 1.235 / 1.131
in round 2, 1.354 / 1.294 / 1.330 / 1.253 in round 3). All three hazard arms
pass it. That is the CFB "too narrow / too wide" pair closing, and it is the
first time.

### 11.3 State cells (the veto) -- S1

| cell | ACTUAL | R2_hier_dirichlet | H1_sub_hazard | H2_sub_hazard_noreset | H3_sub_hazard_lgbm |
|---|---:|---:|---:|---:|---:|
| final 8:00 starters' share, \|m\| <= 5 | 0.7491 | 0.7254 (-2.4 pp)  PASS | 0.6692 (-8.0 pp)  FAIL | 0.6719 (-7.7 pp)  FAIL | 0.6797 (-6.9 pp)  FAIL |
| final 8:00 starters' share, \|m\| 6-15 | 0.7240 | 0.6914 (-3.3 pp)  FAIL | 0.6518 (-7.2 pp)  FAIL | 0.6489 (-7.5 pp)  FAIL | 0.6636 (-6.0 pp)  FAIL |
| final 8:00 starters' share, \|m\| > 15 | 0.5223 | 0.4694 (-5.3 pp)  FAIL | **0.5446 (+2.2 pp)  PASS** | **0.5471 (+2.5 pp)  PASS** | **0.5517 (+2.9 pp)  PASS** |
| starters on floor at >= 4 fouls | 0.4613 | 0.4626 (+0.1 pp)  PASS | 0.4344 (-2.7 pp)  PASS | 0.4341 (-2.7 pp)  PASS | 0.4005 (-6.1 pp)  FAIL |
| **H2 TIP starters' share, \|m\| <= 5** | 0.9678 | 0.7892 (-17.9 pp)  FAIL | **0.9390 (-2.9 pp)  PASS** | 0.8548 (-11.3 pp)  FAIL | **0.9394 (-2.8 pp)  PASS** |
| **H2 TIP starters' share, \|m\| 6-15** | 0.9611 | 0.7977 (-16.3 pp)  FAIL | **0.9374 (-2.4 pp)  PASS** | 0.8493 (-11.2 pp)  FAIL | **0.9418 (-1.9 pp)  PASS** |
| **H2 TIP starters' share, \|m\| > 15** | 0.9563 | 0.7338 (-22.3 pp)  FAIL | **0.9434 (-1.3 pp)  PASS** | 0.8483 (-10.8 pp)  FAIL | **0.9453 (-1.1 pp)  PASS** |
| **H1 20:00-10:00 starters' share, \|m\| <= 5** | 0.7822 | 0.5988 (-18.3 pp)  FAIL | 0.7434 (-3.9 pp)  FAIL | 0.7450 (-3.7 pp)  FAIL | **0.7581 (-2.4 pp)  PASS** |
| (diagnostic) at exactly 4 fouls | 0.5166 | 0.5722 | 0.6666 | 0.6579 | 0.5630 |

Each side is scored against **its own** starting five, the convention rounds 1-3
used. The model's as-of starter set overlaps the real one on **4.58 of 5**;
re-grading the ACTUAL sequence with the MODEL's as-of starter set gives the
benchmark 0.7215 / 0.6964 / 0.5048 / 0.4636 on the round-3 cells and
0.9024 / 0.8908 / 0.8909 / 0.7384 on the round-4 cells.

**Two readings that must not be overstated.** (1) An arm with a hard reset
passes the three tip cells close to by construction, so H1's and H3's passes
there are a statement about the mechanism, not independent evidence; the
informative rows are H2 at -10.8 to -11.3 pp (an *earned* reset) and R2 at -16.3
to -22.3 pp. (2) The tip cell is a single possession per team-game, so its
sampling noise is the largest of any state cell (floor A 0.8-2.4 pp).

### 11.4 Primary metric and lineup concentration -- S1

| metric | ACTUAL | R2_hier_dirichlet | H1_sub_hazard | H2_sub_hazard_noreset | H3_sub_hazard_lgbm |
|---|---:|---:|---:|---:|---:|
| **per-player minutes MAE (min)** | 0.0 | 9.7939 | **8.8189** | 8.8967 | **8.6739** |
| top-1 lineup share of possessions | 0.2940 | 0.2286 | 0.2054 | 0.1982 | 0.2196 |
| top-3 lineup share | 0.5426 | 0.4819 | 0.4427 | 0.4333 | 0.4492 |
| top-5 lineup share | 0.6894 | 0.6409 | 0.5992 | 0.5906 | 0.5953 |
| distinct lineups per team-game | 14.84 | 15.51 | 19.73 | 19.93 | 20.41 |
| K-S of per-player minutes (D) | -- | 0.0798 | 0.0796 | 0.0801 | 0.0813 |
| K-S of the top-1 lineup share (D) | -- | 0.2273 | 0.3373 | 0.3743 | 0.2653 |
| substitution rate at a possession boundary | 0.1518 (train) | 0.1437 | **0.2058** | 0.2052 | 0.2050 |

Every hazard arm beats R2 on the primary metric by **0.90 to 1.12 minutes**
against a floor-A seed SD of **0.0147**, i.e. by 61 to 76 noise floors. No arm
in three rounds had ever beaten R2 on per-player minutes at all. The price is
lineup concentration: the hazard arms run **19.7-20.4** distinct lineups per
team-game against a real 14.84 and R2's 15.51, and their top-1 five-man lineup
share falls to 0.198-0.220 against a real 0.294.

### 11.5 Static column (reported alongside, per 10.4)

| cell | ACTUAL | R2 | H1 | H2 | H3 |
|---|---:|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7491 | 0.7258  PASS | 0.6760  FAIL | 0.6743  FAIL | 0.6852  FAIL |
| final 8:00, \|m\| 6-15 | 0.7240 | 0.6936  FAIL | 0.6579  FAIL | 0.6556  FAIL | 0.6718  FAIL |
| final 8:00, \|m\| > 15 | 0.5223 | 0.4773  FAIL | 0.5583  FAIL | 0.5545  FAIL | 0.5585  FAIL |
| starters at >= 4 fouls | 0.4613 | 0.4650  PASS | 0.4414  PASS | 0.4377  PASS | 0.4024  FAIL |
| H2 tip, \|m\| <= 5 | 0.9678 | 0.7866  FAIL | 0.9353  FAIL | 0.8587  FAIL | 0.9389  PASS |
| H2 tip, \|m\| 6-15 | 0.9611 | 0.7969  FAIL | 0.9374  PASS | 0.8555  FAIL | 0.9385  PASS |
| H2 tip, \|m\| > 15 | 0.9563 | 0.7362  FAIL | 0.9429  PASS | 0.8481  FAIL | 0.9405  PASS |
| H1 20:00-10:00, \|m\| <= 5 | 0.7822 | 0.6120  FAIL | 0.7472  FAIL | 0.7479  FAIL | 0.7619  PASS |
| per-player minutes MAE | 0.0 | 9.8557 | 8.8863 | 8.9510 | 8.7622 |

S1 improves every arm's MAE and moves the blowout band toward the actual on all
three hazard arms (H1 0.5583 -> 0.5446, H2 0.5545 -> 0.5471, H3 0.5585 ->
0.5517), reproducing round 3b's finding that S1 moves cells toward truth.

### 11.6 Noise floor A (20 seeds x 150 games)

| metric | R2_hier_dirichlet | H1_sub_hazard | H2_sub_hazard_noreset | H3_sub_hazard_lgbm |
|---|---:|---:|---:|---:|
| minutes_mae | 0.01473 | 0.01304 | 0.01433 | 0.01356 |
| minutes_mean | 0.13394 | 0.15646 | 0.14532 | 0.15176 |
| top5_share | 0.00256 | 0.00300 | 0.00336 | 0.00271 |
| top8_share | 0.00114 | 0.00140 | 0.00170 | 0.00165 |
| n_nonzero_mean | 0.05849 | 0.05207 | 0.05597 | 0.04517 |
| lu_top1 | 0.00506 | 0.00493 | 0.00510 | 0.00460 |
| late_starter_share_b0 | 0.00646 | 0.01219 | 0.01137 | 0.01176 |
| late_starter_share_b1 | 0.00945 | 0.00641 | 0.00702 | 0.00942 |
| late_starter_share_b2 | 0.01302 | 0.01275 | 0.01465 | 0.01174 |
| foul_trouble_share | 0.01837 | 0.01815 | 0.02470 | 0.01748 |
| h2tip_starter_share_b0 | 0.01086 | 0.00795 | 0.01151 | 0.01103 |
| h2tip_starter_share_b1 | 0.00987 | 0.00896 | 0.01046 | 0.00660 |
| h2tip_starter_share_b2 | 0.02173 | 0.01447 | 0.02386 | 0.01780 |
| opentip_starter_share_close | 0.00637 | 0.00475 | 0.00570 | 0.00373 |

Caveat, stated: floor A's `minutes_mae` rows are computed against the whole
test universe's actual minutes while the sim side is 150 games, so their LEVEL
is not the level of 11.4; the seed-to-seed SD, which is what a floor is, is
unaffected, and on 1,600 games the Monte-Carlo SE is about a third of it. The
floor used in 11.8 is therefore conservative.

### 11.7 Noise floor B -- spec-identical refit under a second seed

Both fits use the same specification; the second draws a different
training-game sample (fit seed 101 vs 11), a different logistic `random_state`
and a different sim seed (23 vs 7). Graded on the same 150-game universe. H1 is
the arm floor B is run on, per 10.8.

| cell | ACTUAL (150 games) | H1 seed 1 | H1 seed 2 | \|delta\| pp |
|---|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7436 | 0.6662 | 0.6666 | 0.04 |
| final 8:00, \|m\| 6-15 | 0.7186 | 0.6602 | 0.6448 | 1.54 |
| final 8:00, \|m\| > 15 | 0.5518 | 0.5515 | 0.5308 | 2.07 |
| starters at >= 4 fouls | 0.4362 | 0.4746 | 0.4483 | 2.63 |
| H2 tip, \|m\| <= 5 | 0.9655 | 0.9328 | 0.9345 | 0.17 |
| H2 tip, \|m\| 6-15 | 0.9662 | 0.9095 | 0.9176 | 0.81 |
| H2 tip, \|m\| > 15 | 0.9611 | 0.9722 | 0.9667 | 0.56 |
| H1 20:00-10:00, \|m\| <= 5 | 0.7710 | 0.7373 | 0.7402 | 0.29 |

**The round-4 family is identified across refits.** Round 3's strongest single
result was that the override family's knobs were not (R8's single `theta` moved
0.1 -> 0.3 between two spec-identical refits and its close band 6.1 pp; R7's
keep scale spanned the whole grid across six S1 windows). Round 4 has no knobs
at all -- only fitted coefficients -- and the two refits agree to 0.04-0.81 pp
on six of eight cells. The two that move more, the blowout band (2.07 pp) and
foul trouble (2.63 pp), move less than round 3's R8 did on its own target cell.

### 11.8 Decision 8 slope check

Cell = starters' share of on-floor slots in the final 8:00 at \|margin\| <= 5,
by quintile of the pregame as-of predicted starter-minutes share.

| quintile | team-games | prior starter share | ACTUAL | R2 | H1 | H2 | H3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 640 | 0.5572 | 0.6770 | 0.6515 | 0.5927 | 0.5910 | 0.5992 |
| Q2 | 640 | 0.6342 | 0.7226 | 0.7080 | 0.6514 | 0.6536 | 0.6605 |
| Q3 | 640 | 0.6698 | 0.7471 | 0.7294 | 0.6768 | 0.6758 | 0.6807 |
| Q4 | 640 | 0.7044 | 0.7763 | 0.7530 | 0.6999 | 0.7072 | 0.7074 |
| Q5 | 640 | 0.7685 | 0.8219 | 0.7834 | 0.7229 | 0.7298 | 0.7490 |

- `ACTUAL` slope **+0.692**, Q5 - Q1 **+14.5 pp**
- `R2_hier_dirichlet` **+0.629**, +13.2 pp
- `H1_sub_hazard` **+0.628**, +13.0 pp
- `H2_sub_hazard_noreset` **+0.672**, +13.9 pp
- `H3_sub_hazard_lgbm` **+0.706**, +15.0 pp

Every arm is monotone in 4 of 4 steps with a slope inside [0.8, 1.2] of the
actual slope ratio (0.91 / 0.91 / 0.97 / 1.02). **Decision 8 is satisfied by all
four arms and cannot separate them**, exactly as in round 3: the close-band
failures are level failures, not responsiveness failures.

### 11.9 Decision 10: the closed-loop check

`ENGINE_ROTATION_FREEZE=1` holds the rotation model's margin at 0 and its
personal- and team-foul counts at 0 while leaving foul accrual, the foul-out
eviction rule and the box-score counters live. Paired-stream runs, 5 seeds over
the fixed 500-game subset of F2 2025 (the slate sorted by `game_id` ascending,
every 11th row, the first 500 -- the same subset the clock round-3c check uses),
with `ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round2b_S_C_s1`,
`ENGINE_FG3=decision8`, `ENGINE_CLOCK=reference` pinned and recorded in every
`run_meta.json`.

| arm | margin SD, live | frozen | ratio | home/away corr, live | frozen | delta | possessions, live | frozen | delta | per-player minutes MAE, live | frozen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R2 (incumbent) | 16.2915 | 16.3735 | **0.9950** | 0.0517 | 0.0543 | -0.0026 | 71.6704 | 71.6090 | +0.061 | 9.3323 | 9.5031 |
| H1 (round 4) | 16.4533 | 16.5148 | **0.9963** | 0.0437 | 0.0331 | +0.0106 | 71.6744 | 71.5800 | +0.094 | 8.7736 | 8.8395 |

**Both pass, and this is the first closed-loop gate the incumbent has ever had
(L25).** Freezing the rotation's engine-produced state moves margin SD by 0.5%,
the home/away correlation by 0.003-0.011 and possessions by 0.06-0.09 -- two
orders of magnitude smaller than the `score_diff` loop L23 found in `fg_make`
(margin SD 34.6 live vs 11.4 frozen). The rotation's consumption of `margin` and
`fouls` is **not** a feedback channel: the rotation changes who scores, not how
much, so it cannot inflate its own input.

The engine table also reproduces the offline MAE ordering independently: H1
8.774 against R2 9.332 on the engine's own player minutes, a 0.56-minute gap in
the same direction as the offline 0.98.

Engine cost of the round-4 family: 500 games x 1 seed in 123 s against R2's 93 s
on the same three workers, i.e. **+32%** on the rotation layer.

### 11.10 Decision

| arm | simplicity | state cells | eligible | G8 cells | minutes MAE | MAE vs R2 | floor | beats floor |
|---|---:|---:|---|---:|---:|---:|---:|---|
| R2_hier_dirichlet | 1 | 2/8 | NO | 5/6 | 9.7939 | -- | 0.0147 | -- |
| H1_sub_hazard | 2 | **5/8** | NO | 4/6 | **8.8189** | +0.9750 | 0.0147 | YES |
| H2_sub_hazard_noreset | 3 | 2/8 | NO | 4/6 | 8.8967 | +0.8971 | 0.0147 | YES |
| H3_sub_hazard_lgbm | 4 | **5/8** | NO | 3/6 | **8.6739** | +1.1200 | 0.0147 | YES |

**No arm adopted.** No arm has all eight state cells inside +/- 3 pp, so by the
pre-registered rule (10.8) nothing is adopted and the diagnosis is reported.
Cell-by-cell misses:

| arm | cell | sim | actual | miss |
|---|---|---:|---:|---:|
| R2 | final 8:00 \|m\| 6-15 | 0.6914 | 0.7240 | -3.3 pp |
| R2 | final 8:00 \|m\| > 15 | 0.4694 | 0.5223 | -5.3 pp |
| R2 | H2 tip (3 bands) | 0.7892 / 0.7977 / 0.7338 | 0.9678 / 0.9611 / 0.9563 | -17.9 / -16.3 / -22.3 pp |
| R2 | H1 20:00-10:00 \|m\| <= 5 | 0.5988 | 0.7822 | -18.3 pp |
| H1 | final 8:00 \|m\| <= 5 | 0.6692 | 0.7491 | -8.0 pp |
| H1 | final 8:00 \|m\| 6-15 | 0.6518 | 0.7240 | -7.2 pp |
| H1 | H1 20:00-10:00 \|m\| <= 5 | 0.7434 | 0.7822 | -3.9 pp |
| H2 | the same three, plus all three tip bands | 0.8548 / 0.8493 / 0.8483 | 0.9678 / 0.9611 / 0.9563 | -10.8 to -11.3 pp |
| H3 | final 8:00 \|m\| <= 5 / 6-15 | 0.6797 / 0.6636 | 0.7491 / 0.7240 | -6.9 / -6.0 pp |
| H3 | starters at >= 4 fouls | 0.4005 | 0.4613 | -6.1 pp |

### 11.11 Diagnosis

**1. The family does what the reachability check said it would, on the cells the
round was called for, and it is not a knob.** The two structural facts L25 named
are produced: the second-half tip goes from R2's 0.789 / 0.798 / 0.734 to
0.939 / 0.937 / 0.943 against 0.968 / 0.961 / 0.956, and the opening ten minutes
of a close game from R2's 0.599 to 0.743-0.758 against 0.782. The blowout band,
which R2 has failed in all four rounds (-4.7, -4.7, -4.7, -5.3 pp), passes on
all three hazard arms for the first time (+2.2 / +2.5 / +2.9 pp). The
within-player minutes SD ratio, which had failed for every arm in every round,
passes on all three (1.02-1.08 against 1.25-1.35). Per-player minutes MAE
improves by 61-76 noise floors. None of that came from a fitted knob: round 4
has no knobs, and floor B says the coefficients are identified.

**2. The reset must be GIVEN, not earned, and the reason is the sampler, not the
fit.** H2 is the pre-registered test of whether a fitted hazard can produce the
second-half reset without being told. It cannot: 0.849-0.855 against
0.956-0.968, a 11 pp miss, while H1 and H3 with the same hazards and a hard
reset land at 0.937-0.945. And section 11.1 shows the *fit* is right -- at a
period boundary the model predicts a bench player's exit at 0.786 against an
actual 0.791 and an off-floor starter's entry at 0.771 against 0.780. The gap is
the **sampler's independence assumption**: the real reset is a near-deterministic
*joint* swap of two or three players at one stoppage, and independent Bernoulli
exits capped at the bench size realise only about four fifths of it
(E[bench exits] = 1.7 of the ~2 bench players on the floor at the horn). A
per-player hazard with independent draws cannot represent a coordinated
substitution, however well each marginal rate is fitted. **That is the round's
sharpest structural finding and it generalises: any per-player hazard family
needs a joint substitution mechanism at dead balls, not five independent coins.**

**3. What broke is the cell the family did not previously fail, and the
mechanism is measurable.** H1's close-late band is 0.6692 against 0.7491, worse
than R2's 0.7254 and 7 pp below the 0.7393 the reachability probe reached.
The probe and the arms differ in exactly two ways, and the second is the cause:

  - the probe used the real starting five (the as-of set costs 2.8 pp, measured:
    the actual sequence re-graded with the model's starter set is 0.7215);
  - the probe's hazards were a saturated cell table applied to the real
    participant pool, while the arms substitute **43% too often** -- 0.2058
    against a real 0.1518 at a possession boundary and R2's 0.1437 -- and run
    **19.7 distinct lineups per team-game against 14.84**.

Over-substitution is the same independence defect as (2) seen from the other
side. Real substitutions arrive in bunches at dead balls; independent Bernoulli
draws on five players spread the same total exits across more boundaries, so the
number of boundaries with any change rises even though the per-player rate is
right. Each extra churn event resamples the floor toward the arm's unconditional
mix, which compresses state dependence: the arms' late-game spread across margin
bands is 0.669 -> 0.545 (12.4 pp) against an actual 0.749 -> 0.522 (22.7 pp).
The level miss and the compression are one phenomenon.

**4. The excluded timeout feature is the leading candidate cause, and the audit
said so before the run.** `experiments.md` 10.2 excluded a timeout indicator for
engine expressibility and recorded the expected cost: "the model's substitutions
are slightly more uniform in time than real ones". A timeout multiplies every
hazard by 3-4x (audit section 2.3), which is exactly the mechanism that makes
real substitutions bunch. The measured 43% excess in the boundary change rate is
that cost, now sized. Fixing it does not require a timeout model: it requires a
**joint** substitution draw -- one Bernoulli per (team, boundary) for "is there a
substitution wave here", then a size and a composition -- which is expressible in
the engine from `prev_end` alone and is the obvious round-5 arm.

**5. H3 is not worth its cost.** The LightGBM hazard buys the opening-tip cell
(-2.4 pp vs H1's -3.9) and 0.14 minutes of MAE, and loses the foul-trouble cell
(-6.1 pp, the only arm to fail it) and the pooled minutes SD ratio (0.856). It
is also not expressible in the sim loop without discretising the booster into a
lookup table (10.11). By the simplicity rule it would lose to H1 on equal cells;
it does not have equal cells.

**6. What round 4 leaves standing.** `R2_hier_dirichlet` is still the arm with
the most round-3 state cells, and it is now measurably the *worst* arm on the
two new cells (-16 to -22 pp at the tip, -18.3 pp over the opening ten minutes)
and on per-player minutes (+0.98 min). Neither family is adoptable. The honest
summary is that the two families fail in **orthogonal** ways -- R2 gets the
close-and-late level right and the within-game shape badly wrong; the hazard
family gets the shape right and the close-and-late level wrong -- and round 5
should be the hazard family with a joint dead-ball substitution mechanism, not a
third family.

### 11.12 Artifacts and flags

- Hazards: `rotation_v4_sub_static.json` and `rotation_v4_sub_S1_{YYYYMM}.json`
  for the five refit windows, with `rotation_v4_manifest.json` in the
  `engine/manifest.py` format (`model`, `scheme: S1`, `fold: F1`, `artifacts`
  with `refit_date` / `path` / `max_train_date`). A window trained on games with
  `game_date < m`, so `max_train_date` is the day before `m`; the manifest's two
  honest-backtest checks (`refit_date < tipoff`, `max_train_date < game_date`)
  both pass on the whole F2 2025 slate. Nothing overwrites `rotation_fit.json`,
  `rotation_fit_v3.json` or any `rotation_fit_v3_S1_*.json`.
- Engine flag: `ENGINE_ROTATION=round4` (plus `ENGINE_ROTATION_ARM=H1|H2`) is
  wired in `engine/rotation_adapter.py` and `engine/loop.py` and runs end to end
  on the 500-game subset. `engine/adapters.py` still refuses a non-`reference`
  value and is owned by another deliverable; `scripts/run_rot4_closed_loop.py`
  loads the adapters as `reference` and sets the mode in-worker, which is what a
  one-line relaxation of that guard would do. **Since no arm was adopted, the
  flag ships unused** and `ENGINE_ROTATION=reference` remains the default.
- `ENGINE_ROTATION_FREEZE=1` is now available for both rotation modes.
- Results: `rotation_F1_round4_results.json` / `_table.csv`,
  `results/engine_v0/rot4_{R2,H1}_{live,frozen}/`.

### 11.13 An engine defect found and fixed during this round

`engine/loop.py` called `push_lineups()` **before** the period/halftime block, so
at a period boundary the rotation was handed the previous possession's period and
clock: the five that took the floor for the first possession of the second half
were chosen with `period = 1, seconds_remaining = 0`, and the R2 adapter's own
period-boundary reshuffle fired one possession late. The offline samplers have
always used the possession's own state, so the engine and the bake-off disagreed
**exactly at the cell round 4 adds**. The call now sits after the period block.
Blast radius on aggregate engine metrics is about one possession in 137 and the
counter-based streams stay aligned (the number of draws per call is unchanged),
but it is a behaviour change for every engine run started after this commit and
any paired comparison that straddles it is invalid.

---

## 12. Round 5 pre-registration -- the joint substitution wave (PM-directed, worker-authored 2026-09-11)

Written and committed BEFORE any round-5 arm was run. Evidence it is built on:
`docs/tests/rotation_wave_audit_2026-09-11.md`, also written and committed
before this section, and round 4's own results (section 11).

### 12.1 Why round 5 changes the DRAW and not the family

Round 4 changed family and proved two things at once (L30):

* the fitted per-player hazards reproduce **every** marginal rate the audit
  measured -- at a period boundary a bench player's exit is 0.786 predicted
  against 0.791 actual, an off-floor starter's entry 0.771 against 0.780
  (section 11.1) -- so nothing is wrong with the fit;
* and the arms built on them cannot make a coordinated substitution. They
  substitute **43% too often** (0.2058 per boundary against 0.1518), run **19.7
  distinct lineups** per team-game against 14.84, lose **11 pp** at the
  second-half tip when the reset must be earned, and compress the late-game
  margin response to 12.4 pp against 22.7 (section 11.11).

The defect is the **independence assumption of the sampler**, not the family:
five independent Bernoulli coins spread the same total exits over more
boundaries, and each extra churn event resamples the floor toward the arm's
unconditional mix. Round 5 therefore keeps round 4's fitted hazards **byte for
byte** and changes only how a boundary is drawn:

1. one Bernoulli per (team, boundary) -- *is there a wave here* -- from a cell
   table over (`prev_end`, time cell, margin band, foul state);
2. a wave **size** from a categorical over the same cell;
3. a **composition** from round 4's own per-player exit/entry hazards;
4. optionally one shared dead-ball draw so the two teams' waves correlate, with
   every marginal preserved exactly.

The PM's direction for round 5, not to be reopened: the wave mechanism, the same
eight state cells as round 4, plus substitutions per boundary and distinct
lineups per game as gates.

### 12.2 The model

**The cell.** 324 = 6 `prev_end` x 9 time cells x 3 margin bands x 2 foul
states, where the foul state is 1 when any player on the floor carries >= 4
personal fouls. In 2024, 153 cells carry n >= 300 and hold **99.39%** of all
boundaries (audit 1.3). Every component is state the possession loop already
has at a boundary.

**The two tables.** `p_wave[cell]` and `p_size[cell, 1..5]`, each fitted by
**two-level shrinkage** -- cell -> (`prev_end` x time cell) -> `prev_end` ->
root -- with the shrinkage constant fixed at **k = 300**, the project's
UNDERPOWERED threshold. `k` is declared here and is NOT tuned: no grid, no
selection over it, and each parent is a marginalisation of the same counts, so
each level is the maximum-likelihood estimate of its own coarser model. Both
objects are lookup tables by construction, so the sim loop makes no model call
(`CLAUDE.md`).

**The composition.** Round 4's fitted logistic hazards, unchanged and not
refitted: `rotation_v4_sub_static.json` and `rotation_v4_sub_S1_{YYYYMM}.json`,
copied into each round-5 artifact so an artifact is self-contained. Given a
wave of size s, the s leavers come out of the five on the floor and the s
entrants off the eligible bench, by one of two knob-free rules per side:

* **rank** -- the s largest `p_out` leave, the s largest `p_in` enter;
* **draw** -- an Efraimidis-Spirakis race weighted by `p/(1-p)`, i.e. weighted
  sampling without replacement, which is round 4's own entry rule.

A player with five fouls leaves first and forces a wave; the wave size is then
at least the number of fouled-out players. That is a rule of the game, not a
fitted term.

**The hard second-half reset** (round 4's H1/H3 form: the model's own predicted
starting five among the eligible, at the first possession of period 2 only) is
part of **every** round-5 arm. Round 4's H2 landed 11 pp short with an earned
reset and the audit's probe lands 15-19 pp short with the wave draw and no hard
reset (audit 6, row "H2 tip (earned)"), so two families in two rounds have now
failed it from below and the question is **closed**, not re-opened.

**The coupling (W3 only).** One shared uniform per (game, boundary): with
probability `rho` both teams compare their own cell probability to the SAME
uniform, otherwise each uses its own. `rho` is one fitted scalar, by moment
matching on the training window's joint counts,
`P(both) = rho E[min(p_A,p_B)] + (1-rho) E[p_A p_B]`, over non-period
boundaries. Marginals are preserved exactly, because a uniform shared with the
other team is still a uniform. The audit measures the residual dependence the
scalar exists to carry: observed P(both) 0.0749 against 0.0296 under the fitted
cells, residual lift **2.53**, **rho = 0.393** on 2024 (audit 5).

**Deliberately EXCLUDED, with the number.** A timeout indicator: P(wave) is
0.4967 at a timeout against 0.1371 elsewhere and timeouts carry 13.3% of all
waves (audit 2). The engine has no timeout model, so a wave probability
conditioned on it could be fitted offline and could not be evaluated in
simulation -- the same class of exclusion as `is_transition` in `features.md`
section 3 and as round 4's own (10.2). Round 5 IS the test of whether the wave
draw makes it unnecessary.

### 12.3 Arms

| arm | composition (exit / entry) | coupling | simplicity | status |
|---|---|---|---:|---|
| `R2_hier_dirichlet` (S1) | -- | -- | 1 | reference (incumbent) |
| `H1_sub_hazard` (S1) | round 4's independent coins | -- | 2 | reference (round 4's best) |
| `W1_wave_rank` | rank / rank | no | 3 | candidate |
| `W2_wave_draw` | draw / draw | no | 4 | candidate |
| `W4_wave_rank_draw` | rank / draw | no | 5 | candidate |
| `W5_wave_draw_rank` | draw / rank | no | 6 | candidate |
| `W3_wave_coupled` | rank / rank | yes, `rho` | 7 | candidate |

W1 and W2 are the PM's two endpoints; W3 is the PM's coupling arm. **W4 and W5
are added by the worker on L25 grounds and the reason is recorded before the
fit**: the audit's reachability probe (audit 6) shows the target lies STRICTLY
INSIDE the composition grid, with the two uniform rules as its endpoints --
distinct lineups 10.85 (W1) < 15.37 actual < 17.54 (W2), close-and-late starter
share 0.6896 (W2) < 0.7388 actual < 0.7878 (W1). Pre-registering only the
endpoints would pre-register two arms the probe already says cannot pass, and
L25 exists to stop exactly that. The grid is exhaustive: two sides, two
knob-free rules, four members, all four run.

The PM's simplicity ordering `R2 < W1 < W2 < W3` is preserved; the two mixed
rules sit between the uniform rules and the coupled arm because each is less
simple than either uniform rule and simpler than adding a fitted scalar.
`H1` is graded as a reference and is **not adoptable in round 5**: it is
unchanged from round 4, where it failed 3 of the 8 state cells.

### 12.4 Scheme, folds, and what is refitted

**Scheme: S1 for every arm**, per round 3b (section 9.4) and round 4 (10.4).
Windows are the calendar months of the 2024-25 season; a game uses the parameter
set whose window closed before its tipoff; the first window's training data is
2024 alone. The static column is reported for the winner and for W1 only, not
for every arm, and that is a stated deviation from round 4 taken to keep the
round inside its compute budget.

**Folds.** F1 = train 2024, test 2025, which IS the standing fold 2; CBBD
carries no on-floor data before 2023-24 (L13) so no other fold exists. **This is
the selection fold and the only one.** Within-2025 walk-forward (the six S1
windows reported separately) is the robustness check and is reported only if an
arm is otherwise adoptable. 2026 stays sealed (`seal.assert_not_sealed` guards
the trainer).

**What round 5 fits: the two wave tables and `rho`, per window, and nothing
else.** Base fits (`rotation_fit_v3.json`, `rotation_fit_v3_S1_*.json`) and the
round-4 hazards (`rotation_v4_sub_*.json`) are REUSED, not refitted; nothing is
written to any of them. Consequence, stated so it cannot be read as a
coincidence: any difference between a round-5 arm and H1 is a difference in the
DRAW alone.

**The two reference columns are taken from round 4's results JSON**
(`rotation_F1_round4_results.json`), not re-simulated: the same 1,600-game
universe, the same subset seed 2025, the same sim seeds 0-2, the same base fits,
the same grading functions and the same hazards reproduce them by construction.
A **1-seed re-run of H1 is executed as a reproduction check** and its cells are
reported next to round 4's; if any cell moves by more than its floor-A SD the
reference columns are discarded and both arms are re-run in full. This is
declared in advance because it is a deviation from "one grading script scores
every arm in this run".

### 12.5 Test universe and grading path

The **same** 1,600-game subset of 2025 that rounds 2, 3, 3b and 4 used (numpy
RandomState seed 2025), 3 seeds per arm under S1, one blind grading path:
`train_rotation_v1.build_row` / `verdict` / `rotation.aggregate_stats`, extended
by `train_rotation_v4.extra_cells` for the two second-half-tip cells and
`train_rotation_v4.minutes_mae` for the primary metric, and by
`train_rotation_v5.wave_cells` for the two new gate cells. Sim and actual go
through the identical functions. Any cell with n < 300 player-games or
possessions is labelled UNDERPOWERED.

### 12.6 Gates

**Every round-4 gate unchanged.**

*G8 cells (report, not veto):* minutes mean +/- 2.0; minutes SD ratio pooled and
within-player 0.9-1.1; top-5 and top-8 share of team minutes +/- 2 pp; players
with > 0 minutes +/- 1.0.

*The eight state cells (the veto), each +/- 3 pp:* starters' share of on-floor
slots in the final 8:00 at \|m\| <= 5 / 6-15 / > 15; starters' share while
carrying >= 4 fouls; the second-half TIP starter share in each of the three
margin bands; and starters' share over H1 20:00-10:00 at \|m\| <= 5. **An arm
missing ANY of the eight is ineligible regardless of G8 or of MAE.** The "at
exactly 4 fouls" diagnostic is reported alongside.

**Two NEW cells, also veto, with their tolerances stated here.**

| new cell | definition | tolerance | why this number |
|---|---|---|---|
| `sub_rate_per_boundary` | fraction of possession boundaries at which the team's on-floor SET changes, `rotation.sim_change_rate`, computed identically on sim and on the actual sequence of the same universe | **+/- 0.015**, or 3x the floor-A seed SD if that is larger | 10% of the actual 0.1518. Round 4's arms sit at +0.054 (3.6 tolerances out) and R2 at -0.008 (inside), so the gate is neither vacuous nor unreachable |
| `distinct_lineups_per_game` | `aggregate_stats.n_lineups_mean`, distinct five-man lineups per team-game | **+/- 1.5**, or 3x the floor-A seed SD if that is larger | 10% of the actual 14.84. Round 4's arms sit at +4.9 to +5.6 and R2 at +0.67 |

Whichever of the two numbers is larger governs, and the governing one is named
in the results table. Both cells are computed for ACTUAL on the round-5 test
universe rather than quoted from the training season, so sim and actual are the
same universe and the same function.

**Lineup concentration (report):** top-1 / top-3 / top-5 five-man lineup share,
K-S D of the top-1 lineup share, K-S D of per-player minutes, mean wave size and
the wave-size histogram against the audit's.

**Decision 8 (a condition on adoption):** team-games bucketed into quintiles of
the pregame as-of share of team minutes going to the predicted starting five;
the close-and-late cell per quintile for ACTUAL and every arm, with slope and
Q5 - Q1. An arm whose profile is flat, or whose slope sign disagrees with
actual, is reported as not matchup-specific whatever its pooled cells say.

### 12.7 Primary metric

**Per-player minutes MAE**, unchanged from round 4 (10.7): outer join of
simulated and actual minutes on (game_id, team_id, pid) over the as-of rotation
set (as-of `mpg >= 10`), per seed then averaged over seeds.

### 12.8 Noise floor and the decision rule

**Floor A, seed-varied sim runs:** 20 seeds x 150 games per arm, the SD of every
gate cell, of the two new cells and of the MAE -- the round-3 and round-4
configuration, so the three rounds' floors are comparable.

**Floor B, spec-identical refit under a second seed:** the wave tables refitted
from a different training-game sample (fit seed 101 vs 11) and simulated under a
different sim seed (23 vs 7), graded on the same 150-game universe. Run on
**W1**, the new object round 5 adds. An arm counts as beating a reference on a
cell only if its improvement exceeds the refit-to-refit spread on that cell.

**Decision rule.** Adopt the **simplest** arm that

1. passes **every** one of the eight state cells at +/- 3 pp, AND
2. passes **both** new cells at the tolerances of 12.6, AND
3. beats `R2_S1` on per-player minutes MAE by more than the floor, AND
4. satisfies the Decision 8 slope check, AND
5. passes the Decision 10 checks of 12.9.

Ties go to the simpler model in the order
`R2 < H1 < W1 < W2 < W4 < W5 < W3`. An arm whose improvement on the cell it was
built to fix does not clear floor B is not adopted on that cell. **If no arm is
eligible, adopt nothing**, report which cell fails and by how much, and name the
diagnosis. No gate is relaxed to produce a winner and no cell is dropped after
seeing a result.

### 12.9 Decision 10: closed loop, BOTH the freeze and the refit-without

Every round-5 arm consumes `margin` and `fouls` twice over -- in the wave cell
and in round 4's composition hazards -- and the engine produces both. L31 fixed
the protocol: *"every Decision-10 gate runs both the frozen arm and the refit-
without-feature arm before any magnitude is quoted"*, because a freeze detects a
loop and only a refit sizes one.

Paired-stream runs of **5 seeds over the fixed 500-game subset** of the F2 2025
slate (sorted by `game_id` ascending, every 11th row, the first 500 -- the same
subset the clock round-3c and rotation round-4 checks use), reporting margin SD
ratio, home/away score correlation, possessions per game and per-player minutes
MAE. Sub-model flags pinned on every run and recorded in `run_meta.json`:
`ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round3_shooter_S_C_s1`,
`ENGINE_CLOCK=reference`.

| run | what it is |
|---|---|
| winner, live | the arm as served |
| winner, `ENGINE_ROTATION_FREEZE=1` | margin held at 0 and the personal/team-foul counts at 0 FOR THE ROTATION MODEL ONLY; foul accrual, the foul-out eviction and the box-score counters stay live |
| winner, **refit-without** | the wave tables refitted with the margin band and the foul state MARGINALISED OUT of the counts before the shrinkage -- the maximum-likelihood fit of the state-free cell model on the same rows, not an ablation of a fitted coefficient -- and round 4's hazards refitted on the same rows with every margin and foul column dropped |
| H1, live and frozen | round 4's arm, re-run so the two rounds' closed loops are the same comparison |

A winner that moves margin SD ratio, home/away correlation or possessions
outside the G1/G2 tolerances between live and frozen is not adopted, and no
magnitude for the loop is quoted from the freeze alone.

### 12.10 Engine expressibility (a condition on adoption)

`wave_cell()` is written over (M,) arrays and is called with M = 1 offline and
M = 2N in the engine, exactly as `rotation_v4.design()` is, so the offline
sampler and the adapter cannot drift apart. The wave draw is one uniform per
(team, boundary), the size an inverse-CDF lookup on a gathered (2N, 5) row, and
the composition an argsort over the 15 roster slots -- all vectorised, no model
call. W3 needs one extra uniform per (game, boundary) shared by the two team
rows, drawn from a game-keyed `StreamBook` family. The winner ships behind
`ENGINE_ROTATION=round5` (plus `ENGINE_ROTATION_ARM`) with S1 artifacts per
month and a manifest in the `engine/manifest.py` format under
`data/processed/models/rotation/round5/`. The two stated RNG divergences of
`docs/models/engine/model.md` section 4.5 apply unchanged, and the shared
coupling stream is a third.

### 12.11 Disclosures

1. The audit's reachability probe (audit section 6) was run before this section
   was written and its numbers are cited here, which is what L25 requires. It
   used the real starting fives, the real participant pools and the real foul
   sequence, so it is not a bake-off result and is never quoted as one.
2. W4 and W5 were added to the arm list after seeing that probe. The probe
   measures the FAMILY's frontier, not an arm's score on the gates, and no gate,
   tolerance or metric was changed after seeing it.
3. The reference columns for R2 and H1 are taken from round 4's results JSON
   rather than re-simulated (12.4), with a 1-seed H1 reproduction check.
4. The static column is reported for W1 and the winner only, not for every arm.
5. `k = 300` in the shrinkage and the `prev_end` x time-cell parent are fixed
   from the audit's cell-coverage table before any arm was fitted.
6. `scripts/diag_rotation_wave_v5.py` and `src/cbb_sim/models/rotation_v5.py`
   existed before this section was committed: the audit script is the evidence
   this section is built on, and the model module was needed to compute the
   `rho` of 12.2 from the audit's own counts. Neither had been run against a
   gate, a bake-off universe or a verdict.

---

## 13. Round-5 results (train 2024, test 2025) -- run 2026-09-11T01:52Z

Pre-registration section 12, committed **a14a569** before the run. Engine/model
code at `git rev-parse HEAD` = a14a569 plus the round-5 module and adapter added
after it (`src/cbb_sim/models/rotation_v5.py`,
`src/cbb_sim/engine/rotation_adapter.py` round-5 block); no fitted object of
rounds 1-4 was written to.

Test universe: the **same** 1,600-game subset of 2025 rounds 2, 3, 3b and 4 used
(numpy RandomState seed 2025), 3 seeds per wave arm under S1, one blind grading
path. Windows and games: 202411 147, 202412 297, 202501 438, 202502 444, 202503
267, 202504 7 (the last is UNDERPOWERED at 7 games and is reported only because
it exists). Base fits and round-4 hazards reused, never refitted; round 5 fits
only `p_wave`, `p_size` and `rho`, on 6,000 team-games per window (fit seed 11),
about 822k-826k boundaries per window.

**ACTUAL on this universe**, through the same functions as every arm:
substitutions per boundary **0.1509**, distinct lineups per team-game
**14.8356**.

### 13.1 The fitted wave tables

| window | team-games | boundaries | fitted wave rate | `rho` | hazards copied in |
|---|---:|---:|---:|---:|---|
| 202411 | 6,000 | 826,238 | 0.1516 | 0.3895 | `rotation_v4_sub_static.json` |
| 202412 | 6,000 | 826,812 | 0.1538 | 0.3905 | `rotation_v4_sub_S1_202412.json` |
| 202501 | 6,000 | 826,639 | 0.1538 | 0.3897 | `..._202501.json` |
| 202502 | 6,000 | 824,620 | 0.1548 | 0.3889 | `..._202502.json` |
| 202503 | 6,000 | 821,933 | 0.1548 | 0.3862 | `..._202503.json` |
| 202504 | 6,000 | 821,496 | 0.1526 | 0.3899 | `..._202504.json` |

`rho` is **0.386-0.391 across six independent windows**, against 0.393 measured
on all of 2024 in the audit. A fitted scalar that moves by half a point across
six refits is identified, which round 3's knobs never were (L25).

The realised simulated rate is 0.1568-0.1575 against a fitted 0.1516-0.1548:
**+0.3 pp, and it has an exact account** -- a foul-out forces a wave whatever
the draw says, and the hard second-half reset fires whatever the draw says.
Neither is a fitted term and both are rules of the game.

### 13.2 G8 cells (report, not veto) -- S1

| cell | tol | ACTUAL | R2 | H1 | W1 | W2 | W4 | W5 | W3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| minutes mean (rotation players) | +/- 2.0 | 24.55 | 25.36 P | 25.99 P | 27.17 **F** | 25.52 P | 25.97 P | 26.83 **F** | 27.19 **F** |
| minutes SD ratio, pooled | 0.9-1.1 | 1.0000 | 1.0548 P | 0.9117 P | 1.0181 P | 0.8843 **F** | 1.0156 P | 0.9047 P | 1.0171 P |
| minutes SD ratio, within-player | 0.9-1.1 | 1.0000 | 1.3416 **F** | 1.0792 P | 1.0867 P | 1.0956 P | 1.1153 **F** | 1.0609 P | 1.0806 P |
| top-5 share of team minutes | +/- 2 pp | 0.7472 | 0.7630 P | 0.7583 P | 0.8002 **F** | 0.7433 P | 0.7704 **F** | 0.7778 **F** | 0.8005 **F** |
| top-8 share of team minutes | +/- 2 pp | 0.9560 | 0.9626 P | 0.9778 **F** | 0.9946 **F** | 0.9727 **F** | 0.9766 **F** | 0.9931 **F** | 0.9948 **F** |
| players with > 0 minutes | +/- 1.0 | 9.64 | 9.13 P | 8.64 **F** | 8.08 **F** | 8.74 P | 8.80 P | 8.12 **F** | 8.08 **F** |
| **G8 passed** | | | **5/6** | 4/6 | 2/6 | **5/6** | 3/6 | 2/6 | 2/6 |

The within-player SD ratio -- which failed for every arm in rounds 1-3 and first
passed in round 4 -- passes on three of the five wave arms and misses W4 by
0.0153. The new failure is **concentration**: a RANKED entry rule (W1, W3, W5's
entry side) puts 0.80 of team minutes in five players against a real 0.747 and
uses 8.08 players against a real 9.64.

### 13.3 State cells (the veto) -- S1

ACTUAL | arm (delta in pp) and PASS/FAIL at +/- 3 pp:

| cell | ACTUAL | R2 | H1 | W1 | W2 | W4 | W5 | W3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7491 | 0.7254 (-2.4) P | 0.6692 (-8.0) F | **0.7281 (-2.1) P** | 0.6494 (-10.0) F | 0.7135 (-3.6) F | 0.6767 (-7.2) F | **0.7288 (-2.0) P** |
| final 8:00, \|m\| 6-15 | 0.7240 | 0.6914 (-3.3) F | 0.6518 (-7.2) F | **0.7149 (-0.9) P** | 0.6301 (-9.4) F | 0.6914 (-3.3) F | 0.6628 (-6.1) F | **0.7118 (-1.2) P** |
| final 8:00, \|m\| > 15 | 0.5223 | 0.4694 (-5.3) F | 0.5446 (+2.2) P | 0.5638 (+4.2) F | 0.5459 (+2.4) P | **0.5237 (+0.1) P** | 0.5852 (+6.3) F | 0.5680 (+4.6) F |
| starters at >= 4 fouls | 0.4613 | 0.4626 (+0.1) P | 0.4344 (-2.7) P | 0.4432 (-1.8) P | 0.4263 (-3.5) F | 0.4216 (-4.0) F | 0.4379 (-2.3) P | 0.4465 (-1.5) P |
| H2 TIP, \|m\| <= 5 | 0.9678 | 0.7892 (-17.9) F | 0.9390 (-2.9) P | 0.9367 (-3.1) F | 0.9377 (-3.0) F | **0.9402 (-2.8) P** | 0.9319 (-3.6) F | 0.9367 (-3.1) F |
| H2 TIP, \|m\| 6-15 | 0.9611 | 0.7977 (-16.3) F | 0.9374 (-2.4) P | 0.9400 (-2.1) P | 0.9374 (-2.4) P | 0.9423 (-1.9) P | 0.9351 (-2.6) P | 0.9393 (-2.2) P |
| H2 TIP, \|m\| > 15 | 0.9563 | 0.7338 (-22.3) F | 0.9434 (-1.3) P | 0.9453 (-1.1) P | 0.9391 (-1.7) P | 0.9431 (-1.3) P | 0.9391 (-1.7) P | 0.9439 (-1.2) P |
| H1 20:00-10:00, \|m\| <= 5 | 0.7822 | 0.5988 (-18.3) F | 0.7434 (-3.9) F | 0.7444 (-3.8) F | 0.7286 (-5.4) F | 0.7332 (-4.9) F | 0.7441 (-3.8) F | 0.7430 (-3.9) F |
| **state cells passed** | | **2/8** | **5/8** | **5/8** | 3/8 | 4/8 | 3/8 | **5/8** |
| (diagnostic) at exactly 4 fouls | 0.5166 | 0.5722 | 0.6666 | 0.6433 | 0.6614 | 0.6028 | 0.6888 | 0.6477 |

R2's and H1's columns are round 4's, per 12.4; the reproduction check is 13.7.

**The cell round 4 broke is fixed, and by the mechanism that was supposed to fix
it.** The close-and-late band goes from H1's 0.6692 (-8.0 pp) to W1's 0.7281
(-2.1 pp) and W3's 0.7288 (-2.0 pp) -- the first time any hazard-family arm has
passed it, and better than R2's own -2.4 pp. The \|m\| 6-15 band, which NO arm in
five rounds had ever passed, passes on W1 (-0.9 pp) and W3 (-1.2 pp).

**And the cell nobody has ever passed is still failed by everybody.** The
opening ten minutes of a close game is -3.8 to -5.4 pp on every round-5 arm,
-3.9 on H1 and -18.3 on R2. Round 4's H3 (-2.4 pp) remains the only arm ever to
pass it, and H3 is ineligible for other reasons (11.11 item 5).

### 13.4 The two new cells, the primary metric, and concentration -- S1

Tolerances, computed as pre-registered (the larger of the fixed number and 3x
the worst floor-A SD): `sub_rate_per_boundary` **+/- 0.015** governs (3x floor =
0.0061); `distinct_lineups_per_game` **+/- 1.5** governs (3x floor = 0.643).

| metric | ACTUAL | R2 | H1 | W1 | W2 | W4 | W5 | W3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **substitutions per boundary** | 0.1509 | 0.1437 (-0.007) P | 0.2058 (+0.055) **F** | 0.1570 P | 0.1571 P | 0.1570 P | 0.1575 P | 0.1568 P |
| **distinct lineups per team-game** | 14.836 | 15.514 (+0.68) P | 19.729 (+4.89) **F** | 11.515 (-3.32) **F** | 16.958 (+2.12) **F** | **14.474 (-0.36) P** | **14.409 (-0.43) P** | 11.504 (-3.33) **F** |
| per-player minutes MAE (min) | 0.0 | 9.7939 | **8.8189** | 9.1244 | 8.9654 | **8.9281** | 9.0348 | 9.1345 |
| MAE gain over R2 (floors) | -- | -- | +66 | +45 | +56 | +54 | +45 | +45 |
| top-1 lineup share | 0.2940 | 0.2286 | 0.2054 | 0.2875 | 0.2192 | 0.2489 | 0.2396 | 0.2866 |
| top-3 lineup share | 0.5426 | 0.4819 | 0.4427 | 0.6066 | 0.4743 | 0.5313 | 0.5192 | 0.6075 |
| top-5 lineup share | 0.6894 | 0.6409 | 0.5992 | 0.7840 | 0.6400 | 0.7021 | 0.6953 | 0.7855 |
| K-S D, per-player minutes | -- | 0.0798 | 0.0796 | 0.1152 | 0.0733 | **0.0747** | 0.1278 | 0.1146 |
| K-S D, top-1 lineup share | -- | 0.2273 | 0.3373 | **0.0843** | 0.2747 | 0.1307 | 0.1732 | **0.0844** |

**The joint draw fixes the over-substitution outright, on every arm.** Round 4's
arms sit at +0.055 per boundary (+36% over the actual on this universe); every
round-5 arm sits at **+0.006 (+4%)**, inside a tolerance derived from a floor of
0.0011-0.0020. That is the whole of L30's first symptom, gone, and it did not
cost the primary metric: every wave arm still beats R2 by 45-56 noise floors on
per-player minutes MAE, and W4 gives up only 0.11 minutes to H1.

**Distinct lineups separate the composition rules exactly as the on-paper probe
said they would** (13.9).

### 13.5 Noise floor A (20 seeds x 150 games, S1)

| metric | W1 | W2 | W4 | W5 | W3 |
|---|---:|---:|---:|---:|---:|
| minutes_mae | 0.01495 | 0.01348 | 0.01610 | 0.01693 | 0.00809 |
| sub_rate_per_boundary | 0.00196 | 0.00139 | 0.00110 | 0.00174 | 0.00203 |
| distinct_lineups_per_game | 0.1514 | 0.2144 | 0.1603 | 0.1796 | 0.1467 |
| late_starter_share_b0 | 0.01253 | 0.00907 | 0.00939 | 0.01063 | 0.01062 |
| late_starter_share_b1 | 0.01182 | 0.01036 | 0.00842 | 0.01302 | 0.00641 |
| late_starter_share_b2 | 0.01512 | 0.01402 | 0.01224 | 0.01158 | 0.00993 |
| foul_trouble_share | 0.01854 | 0.02011 | 0.01800 | 0.01857 | 0.01750 |
| h2tip_starter_share_b0 | 0.01047 | 0.01104 | 0.01121 | 0.01154 | 0.01027 |
| h2tip_starter_share_b1 | 0.00983 | 0.00909 | 0.00898 | 0.00730 | 0.00754 |
| h2tip_starter_share_b2 | 0.02232 | 0.01994 | 0.01446 | 0.02451 | 0.01655 |
| opentip_starter_share_close | 0.00559 | 0.00816 | 0.00753 | 0.00871 | 0.00613 |

R2's and H1's floors are round 4's (11.6) and are unchanged by a run that does
not refit them. The floor-A caveat of 11.6 applies unchanged: the `minutes_mae`
LEVEL is not the level of 13.4, only its seed-to-seed SD is a floor.

**Every state-cell miss above is larger than its floor.** W4's close-band miss
is -3.6 pp against a 0.94 pp floor (3.8 floors) and its \|m\| 6-15 miss -3.3 pp
against 0.84 pp (3.9 floors); W1's tip-b0 miss -3.1 pp against 1.05 pp. None of
the failures is a noise reading.

### 13.6 Noise floor B -- spec-identical wave refit under a second seed (W1)

A different training-game sample (fit seed 101 vs 11) and a different sim seed
(23 vs 7), graded on the same 150-game universe.

| cell | ACTUAL (150 games) | W1 seed 1 | W1 seed 2 | \|delta\| |
|---|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7436 | 0.7057 | 0.7289 | 2.31 pp |
| final 8:00, \|m\| 6-15 | 0.7186 | 0.6980 | 0.6973 | 0.08 pp |
| final 8:00, \|m\| > 15 | 0.5518 | 0.5555 | 0.5680 | 1.25 pp |
| starters at >= 4 fouls | 0.4362 | 0.4373 | 0.4415 | 0.42 pp |
| H2 tip, \|m\| <= 5 | 0.9655 | 0.9276 | 0.9414 | 1.38 pp |
| H2 tip, \|m\| 6-15 | 0.9662 | 0.9311 | 0.9486 | 1.76 pp |
| H2 tip, \|m\| > 15 | 0.9611 | 0.9389 | 0.9444 | 0.56 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0.7710 | 0.7498 | 0.7429 | 0.69 pp |
| substitutions per boundary | -- | 0.1544 | 0.1561 | 0.0017 |
| distinct lineups per team-game | -- | 11.073 | 11.353 | 0.280 |

**The round-5 objects are identified**, on the same reading as round 4's (0.04 -
2.63 pp): six of eight cells move under 1.4 pp and the two new cells move 1.1%
and 2.5% of their own level. The tables have no knobs -- they are counts with a
declared shrinkage constant -- and `rho` moves 0.386-0.391 across six windows
(13.1).

### 13.7 The H1 reproduction check (12.4)

A 1-seed re-run of H1 inside this round, against round 4's own 3-seed column:

| cell | round 4 (3 seeds) | round 5 (1 seed) | delta | floor A |
|---|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.6692 | 0.6679 | -0.0013 | 0.0122 |
| final 8:00, \|m\| 6-15 | 0.6518 | 0.6501 | -0.0017 | 0.0064 |
| final 8:00, \|m\| > 15 | 0.5446 | 0.5404 | -0.0042 | 0.0127 |
| starters at >= 4 fouls | 0.4344 | 0.4341 | -0.0003 | 0.0182 |
| H2 tip, \|m\| <= 5 | 0.9390 | 0.9392 | +0.0002 | 0.0080 |
| H2 tip, \|m\| 6-15 | 0.9374 | 0.9304 | -0.0070 | 0.0090 |
| H2 tip, \|m\| > 15 | 0.9434 | 0.9459 | +0.0025 | 0.0145 |
| H1 20:00-10:00, \|m\| <= 5 | 0.7434 | 0.7405 | -0.0029 | 0.0048 |
| substitutions per boundary | 0.2058 | 0.2061 | +0.0003 | -- |
| distinct lineups | 19.729 | 19.621 | -0.109 | -- |

**Every cell moves less than its floor-A SD, so the reference columns stand.**
(The largest move, the tip at \|m\| 6-15, is -0.70 pp against a 0.90 pp floor.)
The MAE moves +0.058 on one seed against a 0.013 floor, which is the expected
1-vs-3-seed Monte-Carlo difference and is why the check is pre-registered on the
state cells rather than on the MAE.

### 13.8 Decision 8 slope check

Cell = starters' share in the final 8:00 at \|margin\| <= 5, by quintile of the
pregame as-of predicted starter-minutes share (640 team-games per quintile).

| quintile | prior | ACTUAL | H1 | W1 | W2 | W4 | W5 | W3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.5572 | 0.6770 | 0.5920 | 0.6540 | 0.5762 | 0.6321 | 0.6114 | 0.6616 |
| Q2 | 0.6342 | 0.7226 | 0.6569 | 0.7113 | 0.6377 | 0.7019 | 0.6678 | 0.7182 |
| Q3 | 0.6698 | 0.7471 | 0.6724 | 0.7448 | 0.6593 | 0.7286 | 0.6791 | 0.7385 |
| Q4 | 0.7044 | 0.7763 | 0.6989 | 0.7568 | 0.6721 | 0.7444 | 0.7032 | 0.7555 |
| Q5 | 0.7685 | 0.8219 | 0.7169 | 0.7711 | 0.6991 | 0.7573 | 0.7200 | 0.7679 |
| **slope** | | **+0.692** | +0.598 | +0.570 | +0.578 | **+0.602** | +0.518 | +0.512 |
| slope ratio to actual | | 1.00 | 0.86 | 0.82 | 0.84 | **0.87** | **0.75** | **0.74** |
| Q5 - Q1 (pp) | | +14.5 | +12.5 | +11.7 | +12.3 | +12.5 | +10.9 | +10.6 |

Every arm is monotone in 4 of 4 steps. **W5 and W3 fall outside the [0.8, 1.2]
slope-ratio band** (0.75 and 0.74) and are reported as not matchup-specific
enough on this cell; W1, W2 and W4 are inside it. This is the first round in
which Decision 8 SEPARATES arms rather than passing all of them, and the arm it
penalises hardest is the coupled one -- a shared dead-ball draw pushes both
teams toward a common substitution rhythm and flattens the team-to-team spread.

### 13.9 The on-paper probe against the realised bake-off (an L25 postscript)

The audit's reachability probe (`rotation_wave_audit_2026-09-11.md` section 6)
was computed before the pre-registration, on 400 games, with the real starting
fives and the real foul sequence, and no hard reset. The bake-off ran 1,600
games x 3 seeds with as-of starters, simulated fouls and the hard reset. They
should not agree exactly, and the question is whether the probe RANKED the grid
correctly.

| quantity | probe W1 | run W1 | probe W2 | run W2 | probe W4 | run W4 | probe W5 | run W5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| substitutions per boundary | 0.1516 | 0.1570 | 0.1528 | 0.1571 | 0.1525 | 0.1570 | 0.1528 | 0.1575 |
| distinct lineups | 10.85 | 11.52 | 17.54 | 16.96 | 14.49 | 14.47 | 14.46 | 14.41 |
| final 8:00, \|m\| <= 5 | 0.7878 | 0.7281 | 0.6896 | 0.6494 | 0.7656 | 0.7135 | 0.7306 | 0.6767 |

**The probe got the ordering right on every cell and the LEVEL right on distinct
lineups to within 0.67, 0.58, 0.02 and 0.05.** It predicted, before anything was
fitted, that the two uniform composition rules would bracket the target and the
two mixed rules would land on it; both happened. It was uniformly optimistic on
the close-late band by 4.0-5.4 pp, and that offset has a known account: the
probe uses the real starting five, and round 3 measured the as-of starter set at
-2.8 pp on this exact cell, with the rest being the real foul sequence. **An
L25 probe is a ranking instrument and not a level instrument, and this round is
the first evidence of how good it is at the first job.**

### 13.10 Decision

| arm | simplicity | state cells | new cells | eligible | G8 | minutes MAE | vs R2 | floor | beats floor | D8 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| R2_hier_dirichlet | 1 | 2/8 | 2/2 | NO | 5/6 | 9.7939 | -- | 0.0147 | -- | pass |
| H1_sub_hazard | 2 | 5/8 | **0/2** | NO (and not adoptable) | 4/6 | **8.8189** | +0.975 | 0.0147 | YES | pass |
| W1_wave_rank | 3 | **5/8** | 1/2 | NO | 2/6 | 9.1244 | +0.670 | 0.0150 | YES | pass |
| W2_wave_draw | 4 | 3/8 | 1/2 | NO | 5/6 | 8.9654 | +0.828 | 0.0147 | YES | pass |
| W4_wave_rank_draw | 5 | 4/8 | **2/2** | NO | 3/6 | **8.9281** | +0.866 | 0.0161 | YES | pass |
| W5_wave_draw_rank | 6 | 3/8 | **2/2** | NO | 2/6 | 9.0348 | +0.759 | 0.0169 | YES | **FAIL** |
| W3_wave_coupled | 7 | **5/8** | 1/2 | NO | 2/6 | 9.1345 | +0.659 | 0.0147 | YES | **FAIL** |

**No arm adopted.** No arm passes all eight state cells AND both new cells, so
by the pre-registered rule (12.8) nothing is adopted. Cell-by-cell misses of the
two closest arms:

| arm | cell | sim | actual | miss | floors |
|---|---|---:|---:|---:|---:|
| W1 | final 8:00 \|m\| > 15 | 0.5638 | 0.5223 | +4.2 pp | 2.7 |
| W1 | H2 tip \|m\| <= 5 | 0.9367 | 0.9678 | -3.1 pp | 3.0 |
| W1 | H1 20:00-10:00 \|m\| <= 5 | 0.7444 | 0.7822 | -3.8 pp | 6.8 |
| W1 | distinct lineups | 11.52 | 14.84 | -3.32 | 21.9 |
| W4 | final 8:00 \|m\| <= 5 | 0.7135 | 0.7491 | -3.6 pp | 3.8 |
| W4 | final 8:00 \|m\| 6-15 | 0.6914 | 0.7240 | -3.3 pp | 3.9 |
| W4 | starters at >= 4 fouls | 0.4216 | 0.4613 | -4.0 pp | 2.2 |
| W4 | H1 20:00-10:00 \|m\| <= 5 | 0.7332 | 0.7822 | -4.9 pp | 6.5 |

### 13.11 Diagnosis

**1. L30's headline symptom is gone, and the joint draw is what removed it.**
Substitutions per boundary: 0.2058 under round 4 (+36% on this universe),
**0.1568-0.1575 under every round-5 arm (+4%)**, against a floor of 0.0011-0.0020.
The wave draw makes the number of boundaries carrying a change a fitted quantity
instead of an emergent one, and it costs nothing on the primary metric: every
wave arm still beats R2 by 45-56 floors on per-player minutes MAE. The excluded
timeout feature (audit section 2) is no longer needed to make substitutions
bunch -- that was the round's central question and the answer is yes.

**2. The close-and-late cell is fixed, for the first time in the hazard family,
and by the predicted mechanism.** W1 0.7281 and W3 0.7288 against 0.7491, both
inside +/- 3 pp and both better than R2's own -2.4 pp; the \|m\| 6-15 band, which
NO arm in five rounds had passed, passes at -0.9 and -1.2 pp. Round 4's H1 was
-8.0 and -7.2. The cause is exactly the one 11.11 item 3 named: fewer churn
events means fewer resamplings of the floor toward the arm's unconditional mix.

**3. The round's real finding is that the composition rule now trades two gate
families against each other, and no member of the knob-free grid is on both
sides of the trade.** Ranked entries make the floor sticky (close-late 0.728,
top-1 lineup share 0.288 against a real 0.294, K-S D of the top-1 lineup share
0.084 -- a third of R2's and a quarter of H1's) and concentrate the rotation too
far (8.08 players against 9.64, top-5 minutes 0.800 against 0.747, 11.5 distinct
lineups against 14.8). Drawn entries fix the breadth exactly (14.47 and 14.41
distinct lineups, K-S D of per-player minutes 0.0747, the best of any arm in five
rounds) and give the late stickiness back (close-late -3.6 pp). The two mixed
arms sit where the probe said they would. **This is a different failure from
round 4's**: round 4 could not make a coordinated substitution at all; round 5
makes coordinated substitutions and cannot choose WHO with the right amount of
randomness.

**4. The parameter that is missing is a per-player one, and it is nameable.**
A rank rule is a race at zero temperature and a draw rule is a race at
temperature one; the data says the truth is in between, and nothing in the
knob-free grid can express that. The honest options for round 6 are (a) one
fitted exponent `tau` on the entry weights, `w = (p/(1-p))^tau`, fitted on 2024
and frozen -- one parameter, identified if floor B says so, and NOT a post-hoc
knob because it is fitted against the training season rather than tuned against
the gate; or (b) the composition drawn from the hazards but conditioned on WHO
LEFT, which the audit already measures (a single swap takes a starter off 61% of
the time and puts a starter on 48%; at size 2 the 0/1/2-starters-out split is
0.25/0.37/0.37, audit section 4.1) and which is a joint object rather than a
temperature. (b) is the one this round's own logic points at: the wave fixed the
joint structure of WHETHER and HOW MANY, and the remaining defect is the joint
structure of WHO.

**5. Two cells are untouched by anything in five rounds and should be separated
from the rest.** The opening ten minutes of a close game is -3.8 to -5.4 pp on
every round-5 arm, -3.9 on H1, -18.3 on R2, and only round 4's ineligible H3 has
ever passed it. Starters at >= 4 fouls is inside tolerance on four arms but the
"exactly 4 fouls" diagnostic is 0.60-0.69 against a real 0.517 on every hazard
arm, i.e. the pooled cell passes while the mechanism underneath is wrong -- the
pattern section 5 warned about in round 2. Both are foul- and starter-identity
problems rather than substitution-timing problems.

**6. Decision 8 separated arms for the first time, and it penalised the
coupling.** W3's slope ratio is 0.74 and W5's 0.75, both outside [0.8, 1.2],
while W1/W2/W4 are 0.82-0.87. A shared dead-ball draw moves both benches on a
common rhythm and flattens the team-to-team spread the cell is supposed to
show. `rho = 0.39` is real in the data and identified across six windows, and it
still costs responsiveness when it is imposed on both teams' timing rather than
on their composition. That is worth recording against any future coupling arm.

### 13.12 Decision 10: closed loop, the freeze AND the refit-without (L31)

Paired-stream runs, **5 seeds over the fixed 500-game subset** of F2 2025 (the
slate sorted by `game_id` ascending, every 11th row, the first 500 -- the same
subset the clock round-3c and rotation round-4 checks use), with
`ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round3_shooter_S_C_s1`,
`ENGINE_CLOCK=reference` pinned and recorded in every `run_meta.json`. Runs:
`results/engine_v0/rot5_{W4,H1}_{live,frozen}` and `rot5_W4_nostate`. **W4 is
run as the arm closest to eligibility, not as a winner; there is no winner.**

| run | margin SD | home/away corr | possessions | total | per-player minutes MAE |
|---|---:|---:|---:|---:|---:|
| W4 live | 16.2567 | 0.0476 | 71.6112 | 149.065 | 8.7666 |
| W4 `ENGINE_ROTATION_FREEZE=1` | 16.3411 | 0.0608 | 71.6978 | 149.599 | 8.6987 |
| W4 **refit without margin and fouls** (live) | 16.3524 | 0.0732 | 71.7470 | 149.707 | 8.7147 |
| H1 live | 16.3535 | 0.0424 | 71.6522 | 149.286 | 8.7563 |
| H1 `ENGINE_ROTATION_FREEZE=1` | 16.2704 | 0.0789 | 71.7218 | 149.741 | 8.8537 |

| comparison | margin SD ratio | corr delta | possessions delta | verdict |
|---|---:|---:|---:|---|
| W4 live / frozen | **0.9948** | -0.0132 | -0.087 | PASS |
| H1 live / frozen | **1.0051** | -0.0365 | -0.070 | PASS |
| W4 live / refit-without | **0.9941** | -0.0256 | -0.136 | -- |

**Both arms pass, and the two instruments AGREE for the first time.** L31 was
written because clock round 3c's freeze read 4.47 possessions while the
refit-without read 0.42, a factor of ten: freezing `score_diff` at zero told the
clock model the game was tied late and stopped it producing intentional-foul
possessions. Here the freeze moves margin SD by 0.5% and possessions by 0.087,
and the refit-without moves them by 0.6% and 0.136 -- the same order of
magnitude, in the same direction. **The reason is worth recording: a rotation
handed a frozen margin of zero behaves like a rotation in a close game, which is
close to what it does anyway, so the frozen state is not the pathological state
it was for the clock.** A freeze and a refit agree when the frozen value sits
inside the model's normal operating range and diverge when it does not; neither
instrument is universally the conservative one, which is a sharpening of L31
rather than a contradiction of it.

The engine reproduces the offline reading independently: on its own player
minutes W4 is 8.767 and H1 8.756, against R2's 9.332 in round 4's run on the
same subset. Round 4's own H1 numbers (16.4533 live / 16.5148 frozen) are NOT
comparable to the H1 row above and are not quoted as such -- the pinned
`fg_make` moved from `round2b_S_C_s1` to `round3_shooter_S_C_s1` between the two
rounds, which is exactly why 12.4 pre-registered H1's re-run here.

### 13.13 Artifacts and flags

- Wave tables: `data/processed/models/rotation/round5/rotation_v5_wave_{YYYYMM}.json`
  (six windows) with `rotation_v5_manifest.json` in the `engine/manifest.py`
  format. The manifest's two honest-backtest checks (`refit_date < tipoff`,
  `max_train_date < game_date`) both pass on the whole F2 2025 slate. The
  directory is gitignored (`data/processed/models/*/round*/`) and HF-synced.
- Floor-B sibling: `rotation_v5_wave_seed2_*.json` +
  `rotation_v5_manifest_seed2.json`. L31 refit-without-state sibling:
  `rotation_v5_wave_nostate_*.json` + `rotation_v5_manifest_nostate.json`
  (`scripts/train_rotation_v5_nostate.py`; its `n_boundaries` field reads ~4.96M
  because the collapsed counts are broadcast back over the six margin x foul
  cells -- the real boundary count is the same 822k-826k as the live fit, and
  the probabilities are unaffected).
- Nothing was written to `rotation_fit.json`, `rotation_fit_v3*.json` or any
  `rotation_v4_sub_*.json`.
- Engine: `ENGINE_ROTATION=round5` plus `ENGINE_ROTATION_ARM=W1|W2|W3|W4|W5`,
  wired in `engine/rotation_adapter.py` (`Round5Batch`, `init_batch_round5`,
  `next_lineup_round5`, `load_round5`) and the rotation path of `engine/loop.py`
  (which also carries the shared `rotation_wave` stream W3 needs).
  `ENGINE_ROTATION_MANIFEST=nostate` selects the L31 schedule.
  `ENGINE_ROTATION_FREEZE=1` works for round 5 exactly as for round 4.
  **Since no arm was adopted the flag ships unused and
  `ENGINE_ROTATION=reference` remains the default.**
- Tests: `tests/test_rotation_v5.py` (10 cases: the cell function's M=1 vs M=2N
  equality, the fitted tables' normalisation, the collapse-state property, the
  adapter/offline composition parity, the shared stream's sharing and seed
  dependence, and the arm grid). `pytest tests/test_engine.py
  tests/test_rotation_v4.py -q` 28 passed after every edit.
- Results: `rotation_F1_round5_results.json` / `_table.csv`.
- Engine cost: 500 games x 1 seed on ONE worker in 715 s, against round 4's
  123 s on three workers (about 2x the round-4 rotation layer, which is the
  design-matrix evaluation now happening only at wave boundaries but with a
  wider per-row draw block).

### 13.14 Disclosure

A 60-game, 1-seed development smoke run (`--smoke`, artifacts
`rotation_F1_round5_SMOKE_*`) was executed before the full run to verify the
code path end to end. It printed gate cells on 2025. Its numbers are not
evidence, are not cited anywhere, and **no model specification, gate, tolerance
or arm was changed after seeing them** -- the arm list, the cell definition and
`k = 300` were fixed in section 12, committed at a14a569 before the smoke ran.

---

## 14. Round 6 pre-registration -- the composition conditioned on WHO LEFT (PM-directed, worker-authored 2026-09-11)

Written and committed BEFORE any round-6 object was fitted and before any arm
was run. Evidence it is built on: round 5's own results (section 13),
`docs/tests/rotation_wave_audit_2026-09-11.md` sections 4 and 6, and L33.

Numbering note: the PM's lane brief called this "section 12"; sections 12 and 13
are round 5's and `experiments.md` is append-only, so round 6 is section 14.

### 14.1 Why round 6 changes the ENTRY RULE and nothing else

Round 5 changed the draw and fixed the joint structure it modelled (L33):

* substitutions per boundary 0.2058 (round 4) -> **0.1568-0.1575** against a real
  0.1509, inside a tolerance derived from a 0.0011-0.0020 floor (13.4);
* the close-and-late band, which no hazard-family arm had ever passed, passes on
  W1 (-2.1 pp) and W3 (-2.0 pp), and the \|m\| 6-15 band, which NO arm in five
  rounds had passed, passes at -0.9 and -1.2 pp (13.3).

And it left exactly one thing open, which round 5 named before it was measured
and then measured (13.11 item 3): **WHO comes in.** The four knob-free
composition rules trade two gate families against each other and no member of
the grid is on both sides of the trade:

| | ranked entry (W1, W3, W5) | drawn entry (W2, W4) |
|---|---|---|
| close-and-late starters' share | 0.7281 / 0.7288 (PASS) | 0.6494 / 0.7135 (FAIL) |
| distinct lineups per team-game | 11.50-11.52 (FAIL, -3.3) | 16.96 / **14.47** |
| players with > 0 minutes | 8.08 against a real 9.64 | 8.74 / 8.80 |
| top-5 share of team minutes | 0.800 against a real 0.747 | 0.743 / 0.770 |

A rank rule is a race at zero temperature and a draw rule is a race at
temperature one. The data says the truth is between them, and **no knob-free
rule can express that** (13.11 item 3-4). Round 6 therefore keeps

* round 5's **wave tables byte for byte** (`p_wave`, `p_size`, per S1 window),
* round 4's **hazards byte for byte** (`rotation_v4_sub_*.json`),
* round 5's **rank exit rule** (W1/W3/W4's, the side the two round-5 readings
  agree on),
* the **hard second-half reset**, closed in round 4 and again in round 5,

and changes **only how the `size` entrants are chosen off the bench**.
Consequence, stated so it cannot be read as a coincidence later: any difference
between a round-6 arm and W4 is a difference in the **entry composition alone**.

### 14.2 The three candidate entry rules

Baseline (W4's, the incumbent): an Efraimidis-Spirakis race over the eligible
bench with weights `w_j = p_in_j / (1 - p_in_j)`, i.e. weighted sampling without
replacement at temperature 1, where `p_in` is round 4's fitted entry hazard.

**T1 `tau_entry` -- one fitted temperature (13.11 item 4a).**
`w_j = (p_in_j / (1 - p_in_j))^tau`. `tau = 1` is W4 and `tau -> inf` is W1, so
the single scalar spans the interior of the composition grid the probe
bracketed. **`tau` is fitted on the training window by maximum likelihood of the
observed entrant sets**, never against a gate cell: for every training wave of
size `s` with eligible bench pool `B`, the contribution is the
independent-choice (multinomial) approximation
`sum_{j in entrants} [ tau*log w_j - log sum_{b in B} w_b^tau ]`, maximised over
the declared grid `tau in {0.25, 0.50, ..., 5.00}` (20 points, step 0.25). The
grid, the estimator and the likelihood are fixed here. `tau` is reported per S1
window; a scalar that moves across windows is not identified and is reported as
such (the `rho` test of 13.1).

**K1 `cond_class` -- the entry CLASS COUNT conditioned on who left.**
The audit measures the marginals and not the joint: at size 1 a starter leaves
61% of the time and a starter enters 48% of the time; at size 2 the
starters-out split is 0.25 / 0.37 / 0.37 and the starters-in split 0.29 / 0.37 /
0.34 (audit 4.1). K1 fits the **joint**: `P(k_in | size, k_out)`, where `k_out`
is the number of the model's own predicted starters among the `size` leavers and
`k_in` the number among the entrants, as counts over the training window with
one-level shrinkage to `P(k_in | size)` at the project's `k = 300`. At
simulation time the leavers are drawn first (rank rule), `k_out` is read off
them, `k_in` is drawn from the fitted row (clipped to what the bench can
supply), and the `k_in` starters and `size - k_in` non-starters are then drawn
by the SAME temperature-1 race within their own class. Nothing else changes.

**A1 `tier_affinity` -- the entry WEIGHT conditioned on who left.**
Each candidate's entry log-weight gets a fitted additive term that depends on
the departing players' as-of tiers:
`log w_j = log(p_in_j/(1-p_in_j)) + mean_{l in leavers} logA[tier(l), tier(j)]`,
with **tiers from the as-of start rank** -- T0 = predicted starters (rank 1-5),
T1 = rank 6-7, T2 = rank 8-9, T3 = rank 10+ -- so `logA` is a 4x4 table. It is
fitted by **moment matching on the training window's own counts**, not by an
optimiser and not by a grid:
`logA[a, b] = log((obs[a, b] + k) / (exp[a, b] + k))` with `k = 300`, where
`obs[a, b]` counts observed (leaver of tier a, entrant of tier b) pairs and
`exp[a, b]` is the same count's expectation under the BASELINE (W4) entry
weights on the same rows, `exp[a, b] += s * sum_{j: tier(j)=b} w_j / sum_B w`.
A cell with no signal shrinks to `logA = 0`, i.e. to W4 exactly. `k = 300` is
the project's UNDERPOWERED threshold, declared here, not tuned, and identical to
round 5's.

All three are lookup objects (one scalar, one 15-row table, one 4x4 table);
none makes a model call in the sim loop, and none contains a margin, clock or
foul term (14.9).

### 14.3 Arms

| arm | entry rule | new fitted object | simplicity | status |
|---|---|---|---:|---|
| `R2_hier_dirichlet` (S1) | -- | -- | 1 | reference (incumbent, served) |
| `W1_wave_rank` (S1) | rank | -- | 3 | reference (round 5's best state cells) |
| `W4_wave_rank_draw` (S1) | temperature-1 race | -- | 5 | reference (round 5's only 2/2 arm with a passing D8 slope) |
| `T1_tau_entry` | race at fitted `tau` | 1 scalar | 8 | candidate |
| `K1_cond_class` | class count conditioned on `k_out`, then within-class race | 15-row table | 9 | candidate |
| `A1_tier_affinity` | race with a fitted 4x4 tier-pair log-affinity | 16-cell table | 10 | candidate |

The simplicity order `R2 < W1 < W4 < T1 < K1 < A1` is fixed here: one scalar is
simpler than a table over (size, k_out), which is simpler than a table over tier
pairs. **W1 and W4 are references and are not adoptable in round 6** -- they are
unchanged from round 5, where each failed at least one pre-registered cell.

### 14.4 Scheme, folds, and what is refitted

**Scheme: S1 for every arm**, per round 3b (9.4), round 4 (10.4) and round 5
(12.4). Windows are the calendar months of the 2024-25 season, a game uses the
parameter set whose window closed before its tipoff, and the first window trains
on 2024 alone. **No static column is run for any arm**; that is a stated
deviation from round 4, taken to keep the round inside its wall clock, and it
costs nothing that round 3b's scheme confirmation has not already settled.

**Folds.** F1 = train 2024, test 2025, which IS the standing fold 2; CBBD
carries no on-floor data before 2023-24 (L13), so no other fold exists. This is
the selection fold and the only one. 2026 stays sealed
(`seal.assert_not_sealed` guards the trainer).

**What round 6 fits: the three composition objects, per window, and nothing
else.** `rotation_fit_v3*.json`, `rotation_v4_sub_*.json` and
`round5/rotation_v5_wave_*.json` are REUSED and nothing is written to any of
them.

**The three reference columns (R2, W1, W4) are taken from the round-5 results
JSON** (`rotation_F1_round5_results.json`), not re-simulated: same 1,600-game
universe, same subset seed 2025, same sim seeds 0-2, same base fits, same
hazards, same wave tables, same grading functions. A **1-seed re-run of W4 is
executed inside this round as a reproduction check** and its cells are reported
next to round 5's; **if any state cell moves by more than its floor-A SD the
reference columns are discarded and the round is re-run in full.** Declared in
advance, exactly as 12.4 declared it for H1.

### 14.5 Test universe and grading path

The **same** 1,600-game subset of 2025 that rounds 2, 3, 3b, 4 and 5 used
(numpy RandomState seed 2025), **3 seeds per candidate arm** under S1, one blind
grading path: `train_rotation_v1.build_row` / `verdict` /
`rotation.aggregate_stats`, extended by `train_rotation_v4.extra_cells` and
`.minutes_mae` and by `train_rotation_v5.wave_cells`. Sim and actual go through
identical functions. Any cell with n < 300 player-games or possessions is
labelled UNDERPOWERED and is never read as signal or as absence of signal.

### 14.6 Gates -- every round-5 gate, unchanged, and nothing added or relaxed

*G8 cells (report, not veto):* minutes mean +/- 2.0; minutes SD ratio pooled and
within-player 0.9-1.1; top-5 and top-8 share of team minutes +/- 2 pp; players
with > 0 minutes +/- 1.0.

*The eight state cells (the veto), each +/- 3 pp:* starters' share of on-floor
slots in the final 8:00 at \|m\| <= 5 / 6-15 / > 15; starters' share while
carrying >= 4 fouls; the second-half TIP starter share in each of the three
margin bands; starters' share over H1 20:00-10:00 at \|m\| <= 5. **An arm
missing ANY of the eight is ineligible regardless of G8 or of MAE.**

*The two round-5 cells (also veto), at the round-5 tolerances:*
`sub_rate_per_boundary` +/- 0.015 or 3x the floor-A seed SD if larger;
`distinct_lineups_per_game` +/- 1.5 or 3x the floor-A seed SD if larger. The
governing number is named in the results table.

*Report only:* the "at exactly 4 fouls" diagnostic; top-1 / top-3 / top-5
five-man lineup share; K-S D of the top-1 lineup share and of per-player
minutes; mean wave size.

**One report-only diagnostic is ADDED, and it can only make the round harder to
pass, never easier.** Round 3 (5.4) measured that the model's as-of starter set
overlaps the real starting five on 4.57 of 5 and that re-grading the ACTUAL
on-floor sequence with the MODEL's starter set moves the late cells by -1.6 to
-2.8 pp. Round 6 computes that benchmark for **all eight** state cells and
reports it beside every arm, so a cell that no arm can reach because of starter
identification is visible as such. **It changes no tolerance and no verdict**:
every arm is still scored against each side's own real starting five, exactly as
in rounds 1-5.

### 14.7 Primary metric and the two responsiveness checks

**Primary metric: per-player minutes MAE**, unchanged from rounds 4 and 5: outer
join of simulated and actual minutes on (game_id, team_id, pid) over the as-of
rotation set (as-of `mpg >= 10`), per seed then averaged over seeds.

**Responsiveness check 1 (Decision 8, a condition on adoption), unchanged from
13.8:** team-games bucketed into quintiles of the pregame as-of share of team
minutes going to the predicted starting five; the close-and-late cell per
quintile for ACTUAL and every arm, with slope and Q5 - Q1. An arm whose slope
ratio to actual falls outside **[0.8, 1.2]**, or whose sign disagrees, is
reported as not matchup-specific and is NOT adoptable. (This is the band that
separated arms for the first time in round 5 and penalised W3 and W5.)

**Responsiveness check 2 (NEW this round, and a condition on adoption):
per-player minutes MAE by PLAYER quintile.** Players in the as-of rotation set
are bucketed into quintiles of their own pregame as-of minutes per game; the
primary metric is reported per quintile for every arm and every reference. The
round is about WHO comes in, so an arm that improves the pooled MAE by moving
minutes between the middle of the bench and the end of it, while getting the
starters' quintile no better than W4, has not fixed what the round exists to
fix. **The rule, fixed here:** an arm is adoptable only if it beats W4 beyond
the floor in the pooled MAE **and does not lose to W4 beyond the floor in any
single quintile**. Underpowered quintiles are labelled.

### 14.8 Noise floors and the decision rule

**Floor A, seed-varied sim runs:** 20 seeds x 150 games per candidate arm, the
SD of every G8 cell, every state cell, both round-5 cells and the MAE -- the
rounds 3/4/5 configuration, so four rounds' floors are comparable. R2's, W1's
and W4's floors are round 5's and are unchanged by a run that does not refit
them.

**Floor B, spec-identical refit under a second seed:** the round-6 composition
objects refitted from a different training-game sample (fit seed 101 vs 11) and
simulated under a different sim seed (23 vs 7), graded on the same 150-game
universe. Run on **A1**, the largest new object this round adds and the one most
at risk of being unidentified; run additionally on the arm the decision rule
selects, if the wall clock allows. An arm counts as beating a reference on a
cell only if its improvement exceeds the refit-to-refit spread on that cell.

**Decision rule.** Adopt the **simplest** arm that

1. passes **every** one of the eight state cells at +/- 3 pp, AND
2. passes **both** round-5 cells at the tolerances of 14.6, AND
3. beats `R2_hier_dirichlet` on per-player minutes MAE by more than the floor,
   AND
4. beats `W4_wave_rank_draw` on per-player minutes MAE by more than the floor and
   loses to it in no player quintile beyond the floor (14.7), AND
5. satisfies the Decision 8 slope check, AND
6. passes the Decision 10 checks of 14.9.

Ties go to the simpler model in the order `R2 < W1 < W4 < T1 < K1 < A1`. An arm
whose improvement on the cell it was built to fix does not clear floor B is not
adopted on that cell. **If no arm is eligible, adopt nothing**, report which cell
fails and by how much, name the diagnosis, and name the next structure. No gate
is relaxed to produce a winner and no cell is dropped after seeing a result.

### 14.9 Decision 10: the closed loop

**What round 6 adds is state-free by construction.** `tau`, `P(k_in|size,k_out)`
and `logA[tier, tier]` contain no margin, clock or foul term: they are fitted
over all training waves pooled, and the state enters a round-6 arm only through
objects round 5 already sized -- the wave cell (`margin band`, `foul state`) and
round 4's hazards. L31's refit-without-the-feature instrument was run on exactly
those objects in round 5 (13.12: W4 live/refit-without margin SD ratio 0.9941,
possessions -0.136) and **that number is quoted as the size of the loop rather
than re-derived**, because the objects are byte-identical. What round 6 runs is
the **freeze**, which is the instrument that detects whether the round-6 entry
rule opens a NEW channel.

Paired-stream runs over the fixed **500-game subset** of the F2 2025 slate
(sorted by `game_id` ascending, every 11th row, the first 500 -- the subset the
clock round-3c and the rotation round-4 and round-5 checks use), with
`ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round3_shooter_S_C_s1`,
`ENGINE_CLOCK=reference` pinned and recorded in every `run_meta.json`, reporting
margin SD ratio, home/away score correlation, possessions per game and
per-player minutes MAE:

| run | what it is |
|---|---|
| round-6 arm, live | the arm as it would be served |
| round-6 arm, `ENGINE_ROTATION_FREEZE=1` | margin held at 0 and the personal/team-foul counts at 0 FOR THE ROTATION MODEL ONLY; foul accrual, the foul-out eviction and the box-score counters stay live |

**Seed count, declared with its reason.** **5 seeds**, which is the seed count
rounds 4 and 5 ran and therefore the count that makes the three rounds one
comparison. A 25-seed repeat is run **only if the wall clock permits it**; the
lane's hard stop is 12:45 ET and a 25-seed pair costs about 100 minutes on the
6 workers this lane is capped at, so it is pre-registered as conditional and its
absence is reported, not hidden. The arm run is **the arm the decision rule
selects**, or, if no arm is eligible, **the arm closest to eligibility**, which
is stated to be a diagnostic and not a winner (round 5 did the same with W4).

An arm that moves margin SD ratio, home/away correlation or possessions outside
the G1/G2 tolerances between live and frozen is not adopted, and no magnitude
for the loop is quoted from the freeze alone (L31).

### 14.10 Engine expressibility (a condition on adoption)

Every rule is a vectorised array operation over the (2N, S) roster block:

* **T1** is one `**tau` on the weight array -- one line in
  `next_lineup_round5`'s entry block.
* **K1** needs `k_out = (leaving & is_starter).sum(axis=1)`, one gathered CDF row
  from a (5, 6, 6) table, one extra uniform per (team, boundary), and two
  `_pick_k` calls instead of one (starters and non-starters of the bench).
* **A1** needs the leavers' tier counts (a (2N, 4) bincount), one `(4, 4)`
  matmul, and a gather onto each slot's tier -- then the same single `_pick_k`.

All three ship behind **`ENGINE_ROTATION=round6`** plus
**`ENGINE_ROTATION_ARM=T1|K1|A1`**, wired in `engine/rotation_adapter.py` with
S1 artifacts per month and a manifest in the `engine/manifest.py` format under
`data/processed/models/rotation/round6/`. `ENGINE_ROTATION=reference` remains the
default and `engine/adapters.py` is not touched by this lane. The two stated RNG
divergences of `docs/models/engine/model.md` section 4.5 apply unchanged.

### 14.11 Disclosures

1. **The three round-6 arms draw the same uniforms whether or not they use
   them.** K1 needs one extra uniform per wave for `k_in`; T1 and A1 draw and
   discard it, so the three arms sit at identical stream positions and are
   paired. The cost, stated: a round-6 arm is NOT byte-aligned with round 5's
   W4, which is why W4's column comes from round 5's own JSON and is checked by
   the 1-seed reproduction re-run of 14.4 rather than by re-simulation under the
   round-6 code path.
2. `tau`'s likelihood is the independent-choice (multinomial) approximation to
   the sampling-without-replacement likelihood, declared in 14.2 before the fit.
   It is exact at `size = 1`, which is 68% of all waves (audit 3.1).
3. A1's `exp[a, b]` is computed under the baseline entry weights on the training
   rows themselves, so `logA = 0` is exactly W4 and the table measures a
   departure from it. Cells with no signal shrink to 0 at `k = 300`.
4. The reference columns for R2, W1 and W4 are taken from round 5's results JSON
   (14.4), with a 1-seed W4 reproduction check.
5. No static column is run (14.4); the 25-seed closed loop is conditional on the
   wall clock (14.9).
6. `k = 300` and the tier boundaries (5 / 7 / 9) are fixed here, before any fit,
   from the same cell-coverage reasoning round 5 used; neither is tuned.
7. The fitting sample is the same `--wave-team-games 6000` per window round 5
   used, drawn with the same fit seed, so the composition objects and the wave
   tables are fitted on the same rows.

---


## 15. Round-6 results (train 2024, test 2025) -- run 2026-09-11T14:32Z

Pre-registration section 14, committed **4380a08** before `rotation_v6.py`
existed and before anything was fitted. Evidence doc:
`docs/tests/rotation_composition_audit_2026-09-11.md`.

Test universe: the **same** 1,600-game subset of 2025 rounds 2, 3, 3b, 4 and 5
used (numpy RandomState seed 2025), 3 seeds per candidate arm under S1, one blind
grading path. Windows and games: 202411 147, 202412 297, 202501 438, 202502 444,
202503 267, 202504 7 (the last is UNDERPOWERED at 7 games and is reported only
because it exists). Round 5's wave tables, round 4's hazards and round 3b's base
fits are reused and never written to; round 6 fits only the three composition
objects, on 6,000 team-games per window (fit seed 11), 124,260-126,402 waves per
window of which 98.2% are usable (leaver and entrant counts equal and every
entrant inside the as-of pool).

**ACTUAL on this universe**, through the same functions as every arm:
substitutions per boundary **0.1509**, distinct lineups per team-game
**14.8356**, minutes mean 24.551, top-5 share 0.7472, top-8 0.9560, players with
> 0 minutes 9.6438.

### 15.1 The fitted objects

| window | team-games | waves | usable pairs | **tau** | `logA[T0,T2]` | `P(k_in=1 \| size 1, k_out=0)` |
|---|---:|---:|---:|---:|---:|---:|
| 202411 | 6,000 | 124,260 | 122,007 | **1.00** | +0.30 | 0.699 |
| 202412 | 6,000 | 124,935 | 122,309 | **1.00** | +0.31 | 0.700 |
| 202501 | 6,000 | 125,749 | 122,736 | **1.00** | +0.30 | 0.699 |
| 202502 | 6,000 | 126,293 | 123,682 | **1.00** | +0.30 | 0.701 |
| 202503 | 6,000 | 126,402 | 124,048 | **1.00** | +0.25 | 0.698 |
| 202504 | 6,000 | 124,828 | 122,740 | **1.00** | +0.29 | 0.700 |

**`tau` is 1.00 in every window, which IS round 5's W4**, and the likelihood is
not flat: `ll(0.5) - ll(1)` is -19,720 to -24,798 and `ll(5.0) - ll(1)` is
-320,464 to -349,743. **Round 5's option (a) -- "one fitted exponent `tau` on the
entry weights" (13.11 item 4) -- is refuted by its own likelihood**: there is no
temperature between the two knob-free endpoints that the training data prefers to
the one already in use. Stated honestly, `p_in` was itself fitted by logistic ML
on these rows, so the argmax at 1 is close to a property of that fit; the
curvature is not, and the curvature is what makes the option empty.

The two conditional objects are real. K1's table reproduces the measured joint on
the as-of pool -- `P(k_in = 1 | size 1, k_out = 0) = 0.699` against
`P(k_in = 1 | size 1, k_out = 1) = 0.319` -- and the descriptive measurement with
the game's own starters gives 0.749 / 0.316 on 2024 and 0.737 / 0.323 on 2025
(audit 1). A1's largest tier-pair lift is +0.30 (an odds ratio of 1.35) and every
row of `logA` suppresses the deep bench by -0.14 to -0.28.

### 15.2 G8 cells (report, not veto) -- S1

| cell | tol | ACTUAL | R2 | W1 | W4 | T1 | K1 | A1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minutes mean (rotation players) | +/- 2.0 | 24.551 | 25.364 P | 27.173 **F** | 25.968 P | 25.991 P | 25.636 P | 26.031 P |
| minutes SD ratio, pooled | 0.9-1.1 | 1.0000 | 1.0548 P | 1.0181 P | 1.0156 P | 1.0145 P | 0.9961 P | 1.0012 P |
| minutes SD ratio, within-player | 0.9-1.1 | 1.0000 | 1.3416 **F** | 1.0867 P | 1.1153 **F** | 1.1208 **F** | 1.1238 **F** | 1.1103 **F** |
| top-5 share of team minutes | +/- 2 pp | 0.7472 | 0.7630 P | 0.8002 **F** | 0.7704 **F** | 0.7707 **F** | 0.7597 P | 0.7701 **F** |
| top-8 share of team minutes | +/- 2 pp | 0.9560 | 0.9626 P | 0.9946 **F** | 0.9766 **F** | 0.9769 **F** | 0.9696 P | 0.9782 **F** |
| players with > 0 minutes | +/- 1.0 | 9.644 | 9.129 P | 8.083 **F** | 8.799 P | 8.794 P | 8.935 P | 8.754 P |
| **G8 passed** | | | **5/6** | 2/6 | 3/6 | 3/6 | **5/6** | 3/6 |

**K1 is the only arm outside the incumbent ever to carry 5 of 6**, and it does it
on the two cells the wave family has always missed: top-5 share 0.7597 (W4
0.7704) and top-8 0.9696 (W4 0.9766).

### 15.3 State cells (the veto) -- S1

ACTUAL, then each arm with its gap in pp and PASS/FAIL at +/- 3 pp:

| cell | ACTUAL | R2 | W1 | W4 | T1 | K1 | A1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7491 | 0.7254 (-2.4) P | **0.7281 (-2.1) P** | 0.7135 (-3.6) F | 0.7096 (-4.0) F | 0.7001 (-4.9) F | 0.7089 (-4.0) F |
| final 8:00, \|m\| 6-15 | 0.7240 | 0.6914 (-3.3) F | **0.7149 (-0.9) P** | 0.6914 (-3.3) F | 0.6929 (-3.1) F | 0.6890 (-3.5) F | 0.6922 (-3.2) F |
| final 8:00, \|m\| > 15 | 0.5223 | 0.4694 (-5.3) F | 0.5638 (+4.2) F | 0.5237 (+0.1) P | 0.5257 (+0.3) P | 0.5083 (-1.4) P | **0.5218 (-0.0) P** |
| starters at >= 4 fouls | 0.4613 | **0.4626 (+0.1) P** | 0.4432 (-1.8) P | 0.4216 (-4.0) F | 0.4286 (-3.3) F | 0.4212 (-4.0) F | 0.4295 (-3.2) F |
| H2 TIP, \|m\| <= 5 | 0.9678 | 0.7892 (-17.9) F | 0.9367 (-3.1) F | 0.9402 (-2.8) P | 0.9378 (-3.0) P | 0.9385 (-2.9) P | 0.9379 (-3.0) P |
| H2 TIP, \|m\| 6-15 | 0.9611 | 0.7977 (-16.3) F | 0.9400 (-2.1) P | 0.9423 (-1.9) P | 0.9401 (-2.1) P | 0.9410 (-2.0) P | 0.9413 (-2.0) P |
| H2 TIP, \|m\| > 15 | 0.9563 | 0.7338 (-22.3) F | 0.9453 (-1.1) P | 0.9431 (-1.3) P | 0.9465 (-1.0) P | 0.9456 (-1.1) P | 0.9470 (-0.9) P |
| H1 20:00-10:00, \|m\| <= 5 | 0.7822 | 0.5988 (-18.3) F | 0.7444 (-3.8) F | 0.7332 (-4.9) F | 0.7331 (-4.9) F | 0.7267 (-5.6) F | 0.7309 (-5.1) F |
| **state cells passed** | | **2/8** | **5/8** | 4/8 | 4/8 | 4/8 | 4/8 |
| (diagnostic) at exactly 4 fouls | 0.5166 | 0.5722 | 0.6433 | 0.6028 | 0.6143 | 0.5931 | 0.6109 |

R2's, W1's and W4's columns are round 5's, per 14.4; the reproduction check is
15.7. **No round-6 arm improves the veto count over W4**, and the three cells all
of them miss are the same three: both close-late bands, foul trouble and the
opening ten minutes. Section 15.10 splits those into what is reachable and what
is not.

### 15.4 The two round-5 cells, the primary metric, and concentration -- S1

Tolerances, computed as pre-registered (the larger of the fixed number and 3x the
worst floor-A SD): `sub_rate_per_boundary` **+/- 0.015** governs (3x floor =
0.0066); `distinct_lineups_per_game` **+/- 1.5** governs (3x floor = 0.656).

| metric | ACTUAL | R2 | W1 | W4 | T1 | K1 | A1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **substitutions per boundary** | 0.1509 | 0.1437 P | 0.1570 P | 0.1570 P | 0.1560 P | 0.1554 P | 0.1559 P |
| **distinct lineups per team-game** | 14.836 | 15.514 P | 11.515 **F** | 14.474 P | 14.442 P | **14.571 P** | 14.341 P |
| **per-player minutes MAE (min)** | 0.0 | 9.7939 | 9.1244 | 8.9281 | 8.9266 | **8.8622** | 8.9167 |
| MAE gain over R2 (floors) | -- | -- | +45 | +54 | +59 | **+63** | +60 |
| MAE gain over W4 (floors) | -- | -- | -12 | -- | +0.1 | **+4.1** | +0.7 |
| top-1 lineup share | 0.2940 | 0.2286 | 0.2875 | 0.2489 | 0.2489 | 0.2568 | 0.2498 |
| top-3 lineup share | 0.5426 | 0.4819 | 0.6066 | 0.5313 | 0.5312 | 0.5358 | 0.5333 |
| top-5 lineup share | 0.6894 | 0.6409 | 0.7840 | 0.7021 | 0.7028 | 0.7043 | 0.7052 |
| K-S D, per-player minutes | -- | 0.0798 | 0.1152 | 0.0747 | 0.0752 | **0.0589** | 0.0730 |
| K-S D, top-1 lineup share | -- | 0.2273 | **0.0843** | 0.1307 | 0.1297 | 0.0990 | 0.1273 |

**K1 sets the best per-player minutes of any arm in six rounds** (8.8622, +4.1
floors over W4 and +63 over R2) and **the best per-player minutes K-S D ever
measured** (0.0589 against W2's 0.0733 in round 5, the previous best), while
keeping the round-5 cells and improving the top-1 lineup K-S D from 0.1307 to
0.0990. **T1 is W4 to within the floor on every number here**, which is what
`tau = 1.00` requires and is a free extra seed-noise reading.

### 15.5 Noise floor A (20 seeds x 150 games, S1)

| metric | T1 | K1 | A1 |
|---|---:|---:|---:|
| minutes_mae | 0.01181 | 0.01348 | 0.01108 |
| sub_rate_per_boundary | 0.00219 | 0.00198 | 0.00204 |
| distinct_lineups_per_game | 0.2185 | 0.1842 | 0.2029 |
| late_starter_share_b0 | 0.00849 | 0.01087 | 0.00767 |
| late_starter_share_b1 | 0.00733 | 0.00961 | 0.00691 |
| late_starter_share_b2 | 0.01644 | 0.01764 | 0.01650 |
| foul_trouble_share | 0.02544 | 0.01863 | 0.02314 |
| h2tip_starter_share_b0 | 0.00728 | 0.00819 | 0.00690 |
| h2tip_starter_share_b1 | 0.00626 | 0.00669 | 0.00676 |
| h2tip_starter_share_b2 | 0.01704 | 0.01974 | 0.01884 |
| opentip_starter_share_close | 0.00818 | 0.00760 | 0.00836 |
| top5_share | 0.00216 | 0.00256 | 0.00238 |
| n_nonzero_mean | 0.05073 | 0.06463 | 0.05058 |

R2's, W1's and W4's floors are round 5's and are unchanged by a run that does not
refit them. **Every state-cell miss in 15.3 is larger than its floor**: K1's
close-band miss is -4.9 pp against a 1.09 pp floor (4.5 floors), its foul-trouble
miss -4.0 pp against 1.86 pp (2.2 floors) and its opening-ten-minutes miss
-5.6 pp against 0.76 pp (7.3 floors). The floor-A caveat of 11.6 applies
unchanged: the `minutes_mae` LEVEL is not the level of 15.4, only its
seed-to-seed SD is a floor.

### 15.6 Noise floor B -- spec-identical composition refit under a second seed (A1)

A different training-game sample (fit seed 101 against 11) and a different sim
seed (23 against 7), graded on the same 150-game universe, run in its own process
so the bake-off JSON was on disk before the second fit started.

| cell | ACTUAL (150 games) | A1 seed 1 | A1 seed 2 | \|delta\| |
|---|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7436 | 0.7026 | 0.6998 | 0.28 pp |
| final 8:00, \|m\| 6-15 | 0.7186 | 0.6888 | 0.6918 | 0.30 pp |
| final 8:00, \|m\| > 15 | 0.5518 | 0.5434 | 0.5216 | 2.17 pp |
| starters at >= 4 fouls | 0.4362 | 0.4293 | 0.4595 | 3.02 pp |
| H2 tip, \|m\| <= 5 | 0.9655 | 0.9241 | 0.9552 | 3.10 pp |
| H2 tip, \|m\| 6-15 | 0.9662 | 0.9324 | 0.9392 | 0.68 pp |
| H2 tip, \|m\| > 15 | 0.9611 | 0.9611 | 0.9611 | 0.00 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0.7710 | 0.7334 | 0.7456 | 1.22 pp |
| substitutions per boundary | -- | 0.1535 | 0.1566 | 0.0031 |
| distinct lineups per team-game | -- | 13.990 | 14.403 | 0.413 |

**The round-6 objects are identified, on the same reading as rounds 4 and 5.**
Under the refit `tau` is still **1.00 in all six windows**, `logA[T0]` moves
0.01-0.05 and `P(k_in = 1 | size 1, k_out = 0)` moves 0.001-0.011. The cell
spread (0.00-3.10 pp) is round 5's own (0.08-2.31 pp) to within its two thinnest
cells, and the objects have no knobs: they are counts with a declared shrinkage
constant and a likelihood argmax over a declared grid.

**Floor B was also run on K1**, which 14.8 allows for the arm the decision rule
would select and which this round's conclusion leans on. One `CompFit` artifact
carries all three objects, so the seed-101 refit already existed and this is a
grading run only; it writes a versioned sibling
(`rotation_F1_round6_floorB_K1.json`) and overwrites nothing.

| cell | ACTUAL (150 games) | K1 seed 1 | K1 seed 2 | \|delta\| |
|---|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7436 | 0.6993 | 0.7181 | 1.88 pp |
| final 8:00, \|m\| 6-15 | 0.7186 | 0.6721 | 0.6824 | 1.03 pp |
| final 8:00, \|m\| > 15 | 0.5518 | 0.5031 | 0.5072 | 0.41 pp |
| starters at >= 4 fouls | 0.4362 | 0.4154 | 0.4453 | 2.99 pp |
| H2 tip, \|m\| <= 5 | 0.9655 | 0.9224 | 0.9500 | 2.76 pp |
| H2 tip, \|m\| 6-15 | 0.9662 | 0.9270 | 0.9365 | 0.95 pp |
| H2 tip, \|m\| > 15 | 0.9611 | 0.9611 | 0.9333 | 2.78 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0.7710 | 0.7317 | 0.7282 | 0.35 pp |
| substitutions per boundary | -- | 0.1545 | 0.1585 | 0.0040 |
| distinct lineups per team-game | -- | 14.377 | 14.510 | 0.133 |

K1's refit-to-refit spread (0.35-2.99 pp) is A1's and round 5's W1's, and its two
new cells move 2.6% and 0.9% of their own level. **The K1 - W4 gap the decision
turns on is not a refit artefact in either direction**: the pooled MAE gain
(+0.066) and the Q2 loss (+0.060) are both smaller than this spread on the state
cells but are measured on 1,600 games x 3 seeds against a 0.016-minute seed floor,
which is the comparison 14.8 rule 4 names.

### 15.7 The W4 reproduction check (14.4)

A 1-seed re-run of W4 inside this round, against round 5's own 3-seed column:

| cell | round 5 (3 seeds) | round 6 (1 seed) | delta | floor A |
|---|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7135 | 0.7150 | +0.0015 | 0.0094 |
| final 8:00, \|m\| 6-15 | 0.6914 | 0.6891 | -0.0023 | 0.0084 |
| final 8:00, \|m\| > 15 | 0.5237 | 0.5229 | -0.0008 | 0.0122 |
| starters at >= 4 fouls | 0.4216 | 0.4247 | +0.0031 | 0.0180 |
| H2 tip, \|m\| <= 5 | 0.9402 | 0.9392 | -0.0011 | 0.0112 |
| H2 tip, \|m\| 6-15 | 0.9423 | 0.9357 | -0.0066 | 0.0090 |
| H2 tip, \|m\| > 15 | 0.9431 | 0.9485 | +0.0053 | 0.0145 |
| H1 20:00-10:00, \|m\| <= 5 | 0.7332 | 0.7340 | +0.0008 | 0.0075 |
| substitutions per boundary | 0.1570 | 0.1573 | +0.0003 | 0.0011 |
| distinct lineups | 14.474 | 14.439 | -0.0353 | 0.1603 |

**Every state cell moves less than its floor-A SD, so the reference columns
stand.** (The largest move, the tip at \|m\| 6-15, is -0.66 pp against a 0.90 pp
floor.) The MAE moves +0.047 on one seed against a 0.016 floor, which is the
expected 1-vs-3-seed Monte-Carlo difference and is why the check is pre-registered
on the state cells rather than on the MAE -- the same reading round 5's H1 check
gave.

### 15.8 The two responsiveness conditions

**Decision 8 (14.7 check 1).** Cell = starters' share in the final 8:00 at
\|margin\| <= 5, by quintile of the pregame as-of predicted starter-minutes share
(640 team-games per quintile).

| quintile | prior | ACTUAL | W1 | W4 | T1 | K1 | A1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.5572 | 0.6770 | 0.6540 | 0.6321 | 0.6303 | 0.6218 | 0.6291 |
| Q2 | 0.6342 | 0.7226 | 0.7113 | 0.7019 | 0.7002 | 0.6940 | 0.6978 |
| Q3 | 0.6698 | 0.7471 | 0.7448 | 0.7286 | 0.7198 | 0.7089 | 0.7209 |
| Q4 | 0.7044 | 0.7763 | 0.7568 | 0.7444 | 0.7369 | 0.7299 | 0.7367 |
| Q5 | 0.7685 | 0.8219 | 0.7711 | 0.7573 | 0.7577 | 0.7430 | 0.7570 |
| **slope** | | **+0.692** | +0.570 | +0.602 | +0.602 | +0.576 | +0.607 |
| slope ratio to actual | | 1.00 | 0.82 | 0.87 | 0.87 | 0.83 | 0.88 |
| Q5 - Q1 (pp) | | +14.5 | +11.7 | +12.5 | +12.7 | +12.1 | +12.8 |

Every arm is monotone in 4 of 4 steps and **every round-6 arm is inside the
[0.8, 1.2] band** (0.83-0.88). Decision 8 separates nothing this round; it
separated W3 and W5 in round 5 and both are absent here.

**The NEW per-player-quintile condition (14.7 check 2).** Per-player minutes MAE
by quintile of the player's own pregame as-of minutes per game (edges 15.97 /
20.89 / 25.33 / 29.71 mpg; 4,782-4,784 player-games per quintile, none
underpowered):

| quintile | W4 | T1 | K1 | A1 | K1 - W4 | floor |
|---|---:|---:|---:|---:|---:|---:|
| Q1 (lowest mpg) | 9.6681 | 9.4597 | **9.3108** | 9.4504 | **-0.357** | 0.016 |
| Q2 | **9.7294** | 9.8622 | 9.7891 | 9.8707 | **+0.060** | 0.016 |
| Q3 | 9.4074 | 9.3831 | **9.2575** | 9.3193 | -0.150 | 0.016 |
| Q4 | 8.6621 | 8.6099 | 8.6461 | 8.6389 | -0.016 | 0.016 |
| Q5 (highest mpg) | 7.3307 | 7.2410 | 7.2367 | **7.2271** | -0.094 | 0.016 |

**All three round-6 arms lose Q2 to W4 beyond the floor** (+0.060, +0.133,
+0.141), and that is the cell the pre-registered rule 4 vetoes. The pattern is
identical across three arms with different objects, so it is not seed noise: a
conditional entry rule buys its pooled gain at the bottom and the top of the
rotation and gives a little back on the **first men off the bench**, which is
exactly the class `k_in` moves.

### 15.9 Per-game and per-team evidence

**Per game** (per-player minutes MAE per game, averaged over seeds): W4 8.970
+/- 2.634 with a 90th percentile of 12.141; T1 8.926 +/- 2.211 / 11.427; **K1
8.868 +/- 2.239 / 11.288**; A1 8.917 +/- 2.223 / 11.352. K1's gain is in the
whole distribution, not in a few games: the worst decile improves by 0.85 minutes
per player.

**Per team** (starters' share in the final 8:00 at \|m\| <= 5, aggregated per
team; a team is powered when both its simulated and its actual denominators reach
300 on-floor slots -- **251 powered, 112 UNDERPOWERED and excluded**):

| | sim mean | sim SD | actual mean | actual SD | corr(sim, actual) | mean \|dev\| |
|---|---:|---:|---:|---:|---:|---:|
| W4 | 0.7161 | 0.0662 | 0.7512 | 0.0832 | 0.381 | 0.0727 |
| T1 | 0.7103 | 0.0493 | 0.7512 | 0.0832 | 0.432 | 0.0683 |
| K1 | 0.7004 | 0.0507 | 0.7512 | 0.0832 | 0.394 | 0.0757 |
| A1 | 0.7098 | 0.0481 | 0.7512 | 0.0832 | **0.433** | 0.0686 |

Every arm **under-disperses across teams** (SD 0.048-0.066 against a real 0.083)
while correlating 0.38-0.43 with the team's own actual share. This is the same
compression Decision 8 reads as a slope ratio of 0.83-0.88, seen at team
resolution.

### 15.10 The as-of starter benchmark (report only, 14.6)

The ACTUAL sequence of the same 1,600 games, re-graded with the MODEL's as-of
predicted starting five instead of the game's own. The model's five overlaps the
real five on **4.576 of 5**.

| cell | ACTUAL (own) | ACTUAL (as-of) | starter identification | best round-6 arm | arm vs BENCHMARK |
|---|---:|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7491 | 0.7215 | **-2.8 pp** | 0.7096 (T1) | -1.2 pp |
| final 8:00, \|m\| 6-15 | 0.7240 | 0.6964 | **-2.8 pp** | 0.6929 (T1) | -0.4 pp |
| final 8:00, \|m\| > 15 | 0.5223 | 0.5048 | -1.7 pp | 0.5218 (A1) | +1.7 pp |
| starters at >= 4 fouls | 0.4613 | 0.4636 | +0.2 pp | 0.4295 (A1) | **-3.4 pp** |
| H2 tip, \|m\| <= 5 | 0.9678 | 0.9024 | -6.5 pp | 0.9385 (K1) | +3.6 pp |
| H2 tip, \|m\| 6-15 | 0.9611 | 0.8908 | -7.0 pp | 0.9413 (A1) | +5.1 pp |
| H2 tip, \|m\| > 15 | 0.9563 | 0.8909 | -6.5 pp | 0.9470 (A1) | +5.6 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0.7822 | 0.7384 | **-4.4 pp** | 0.7331 (T1) | -0.5 pp |

**It changes no tolerance and no verdict** -- every arm above is scored against
each side's own real starting five, as in rounds 1-5 -- and it is the most useful
table in the round: the opening-ten-minutes cell that no arm in six rounds has
passed is **-4.4 pp unreachable for starter-identification reasons alone**, both
close-late bands are within 1.2 pp of the honest benchmark, and **foul trouble,
at +0.2 pp of benchmark, is the one veto cell whose failure is genuinely the
model's own**.

### 15.11 Decision

| arm | simplicity | state | round-5 cells | G8 | MAE | vs R2 | vs W4 | quintiles | D8 | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| R2_hier_dirichlet | 1 | 2/8 | 2/2 | 5/6 | 9.7939 | -- | -0.866 | ok | n/a | NO |
| W1_wave_rank | 3 | **5/8** | 1/2 | 2/6 | 9.1244 | +0.669 | -0.196 | ok | 0.82 | NO |
| W4_wave_rank_draw | 5 | 4/8 | 2/2 | 3/6 | 8.9281 | +0.866 | -- | ok | 0.87 | NO |
| T1_tau_entry | 8 | 4/8 | 2/2 | 3/6 | 8.9266 | +0.867 | +0.002 | **Q2 fails** | 0.87 | NO |
| K1_cond_class | 9 | 4/8 | 2/2 | **5/6** | **8.8622** | +0.932 | **+0.066** | **Q2 fails** | 0.83 | NO |
| A1_tier_affinity | 10 | 4/8 | 2/2 | 3/6 | 8.9167 | +0.877 | +0.011 | **Q2 fails** | 0.88 | NO |

**No arm adopted.** No arm passes all eight state cells, and the one arm that
clears every metric condition (K1: beats R2 by 63 floors, beats W4 by 4.1 floors,
5/6 G8, both round-5 cells, D8 inside the band) fails the pre-registered
per-player-quintile condition on Q2 and four of the eight state cells. Per 14.8
the rule adopts nothing; no gate was relaxed and no cell was dropped after seeing
a result. `ENGINE_ROTATION=reference` (R2) stays the served default and the
`round6` flag ships unused.

Cell-by-cell misses of the closest arm:

| arm | cell | sim | actual | miss | floors | of which starter identification |
|---|---|---:|---:|---:|---:|---:|
| K1 | final 8:00 \|m\| <= 5 | 0.7001 | 0.7491 | -4.9 pp | 4.5 | -2.8 pp |
| K1 | final 8:00 \|m\| 6-15 | 0.6890 | 0.7240 | -3.5 pp | 3.6 | -2.8 pp |
| K1 | starters at >= 4 fouls | 0.4212 | 0.4613 | -4.0 pp | 2.2 | **+0.2 pp (none)** |
| K1 | H1 20:00-10:00 \|m\| <= 5 | 0.7267 | 0.7822 | -5.6 pp | 7.3 | -4.4 pp |
| K1 | player quintile Q2 | 9.7891 | (W4 9.7294) | +0.060 | 3.7 | -- |

### 15.12 Diagnosis

**1. The temperature hypothesis is dead, and its own likelihood killed it.**
`tau = 1.00` in six independent windows, with the rank direction rejected by
320k-350k log units and the flatter direction by 20k-25k. T1 is therefore W4's
rule exactly, and its cells reproduce W4's inside the floor. Round 5's option (a)
is closed, not deferred.

**2. The conditional object is real, large and stable, and K1 over-represents
it.** The measured joint is a **class-reversing swap** -- at size 1 a bench player
leaving puts a starter on 0.749 (2025: 0.737) and a starter leaving puts one on
0.316 (0.323), n = 46k-89k per cell, and the same monotone pattern holds at sizes
2 and 3 (audit 1). In simulation, W4's unconditional race realises a spread of
20.4 pp against a real 34.0 pp; K1 realises 43.2 pp -- past the target, because
the table it applies was estimated on the training window's leaver mix and the
sim's leaver mix is different (item 3).

**3. The binding defect has moved to the EXIT side, and the number is 12-13 pp.**
Every arm since round 5 takes a starter off at **0.456-0.467 of single swaps
against a real 0.587** (audit 4), because round 6 pre-registered round 5's RANK
exit rule for all three arms. A conditional entry rule fed a wrong leaver mix
conditions correctly on a class that arrives with the wrong frequency; that is
why K1 moves the pooled minutes MAE to its best value in six rounds and moves no
state cell. **Round 7's object is the mirror of K1 on the exit side**:
`P(k_out | size, state)` drawn first, then the leavers raced within class.

**4. Most of what the remaining late and opening cells read is starter
identification, not rotation** (15.10): -2.8 pp of the close bands and -4.4 pp of
the opening ten minutes. Against the like-for-like benchmark the arms are -1.2 /
-0.4 / -0.5 pp. The gate is not relaxed on that account, and the PM now has the
number needed to decide whether the cell should be scored against the benchmark.

**5. Foul trouble is the only veto cell left that is the model's own**, at -3.2
to -4.2 pp against a benchmark that sits +0.2 pp from the actual, with the "at
exactly 4 fouls" diagnostic 0.59-0.61 against a real 0.517 on every hazard arm.
It is a foul-process and benching-response failure and nothing in the
substitution draw will fix it. Round 2's warning about a pooled cell passing over
a wrong mechanism now applies to the arms that FAIL the pooled cell as well.

**6. The composition needs exactly one state term, and it is measured** (audit
2): the final eight minutes of a decided game, where P(a starter enters | a bench
player leaves) is 0.5286 (2025: 0.5024) against 0.72-0.84 in the other eight
state cells. Round 6's objects are state-free by pre-registration and that was
the right call for a first cut; round 7 can add one cell with a measured size
rather than a guessed one.

### 15.13 Decision 10: the closed loop

Paired-stream runs over the fixed **500-game subset** of F2 2025 (the slate
sorted by `game_id` ascending, every 11th row, the first 500 -- the subset the
clock round-3c and the rotation round-4 and round-5 checks use), with
`ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round3_shooter_S_C_s1`,
`ENGINE_CLOCK=reference` pinned and recorded in every `run_meta.json`. Runs:
`results/engine_v0/rot6_K1_{live,frozen}` and `rot6_K1_{live,frozen}_s25`.
**K1 is run as the arm closest to eligibility, not as a winner; there is no
winner.**

| run | seeds | margin SD | home/away corr | possessions | total | per-player minutes MAE |
|---|---:|---:|---:|---:|---:|---:|
| K1 live | 5 | 16.5337 | 0.0361 | 71.7092 | 149.367 | 8.7390 |
| K1 `ENGINE_ROTATION_FREEZE=1` | 5 | 16.5367 | 0.0399 | 71.6818 | 149.516 | 8.6491 |
| K1 live | **25** | 16.1617 | 0.0665 | 71.7370 | 149.096 | 8.7613 |
| K1 `ENGINE_ROTATION_FREEZE=1` | **25** | 16.2695 | 0.0566 | 71.7533 | 149.265 | 8.6764 |

| comparison | margin SD ratio | corr delta | possessions delta | verdict |
|---|---:|---:|---:|---|
| K1 live / frozen, 5 seeds | **0.9998** | -0.0038 | +0.027 | PASS |
| K1 live / frozen, **25 seeds** | **0.9934** | +0.0099 | -0.016 | PASS |
| (round 5) W4 live / frozen, 5 seeds | 0.9948 | -0.0132 | -0.087 | PASS |

**The freeze passes at both seed counts, and the 25-seed pair was run** -- the
pre-registration made it conditional on the wall clock (14.9) and the clock
allowed it: each run is 500 games x 25 seeds in 1,580-1,600 s on 3 workers.
Freezing the margin and the foul counts for the rotation model only moves margin
SD by 0.02% at 5 seeds and 0.66% at 25, possessions by 0.027 and 0.016 against a
G1 tolerance of 1.0, and the home/away correlation by 0.004 and 0.010. The
5-seed and 25-seed LEVELS differ (16.53 against 16.16) because they are different
seed sets, not different models; each pair is internally paired and only the
within-pair ratio is read.

The engine reproduces the offline ordering independently: on its own player
minutes K1 is **8.739 (5 seeds) / 8.761 (25 seeds)** against round 5's W4 8.767
and round 4's R2 9.332 on the same 500-game subset with the same pinned
sub-models -- the same sign and about the same size as the offline 8.862 against
8.928 and 9.794.

**What round 6 does NOT re-derive, and why.** `tau`, `P(k_in | size, k_out)` and
`logA` carry **no margin, clock or foul term**: they are fitted over all training
waves pooled. The state reaches a round-6 arm only through objects round 5 already
sized with L31's refit-without instrument -- the wave cell and round 4's hazards,
byte-identical here -- so round 5's number (W4 live/refit-without margin SD ratio
0.9941, possessions -0.136, 13.12) is the size of that channel and is quoted
rather than recomputed. What round 6 runs is the **freeze**, the instrument that
detects whether the new entry rule opens a NEW channel. It does not.

### 15.14 Artifacts and flags

- Composition objects:
  `data/processed/models/rotation/round6/rotation_v6_comp_{YYYYMM}.json` (six
  windows) with `rotation_v6_manifest.json` in the `engine/manifest.py` format;
  the manifest's two honest-backtest checks (`refit_date < tipoff`,
  `max_train_date < game_date`) pass on the whole F2 2025 slate and
  `load_round6` additionally ASSERTS that the round-5 wave manifest and the
  round-6 composition manifest select the same segment for every game. The
  directory is gitignored (`data/processed/models/*/round*/`, 116 KB) and
  HF-synced.
- Floor-B sibling: `round6/rotation_v6_comp_seed2_*.json` +
  `rotation_v6_manifest_seed2.json`.
- Nothing was written to `rotation_fit.json`, `rotation_fit_v3*.json`, any
  `rotation_v4_sub_*.json` or any `round5/rotation_v5_wave_*.json`.
- Engine: `ENGINE_ROTATION=round6` plus `ENGINE_ROTATION_ARM=T1|K1|A1`, wired in
  `engine/rotation_adapter.py` (`Round6Batch`, `init_batch_round6`,
  `next_lineup_round6`, `load_round6`, `round6_rows`) and one default-off branch
  in the rotation path of `engine/loop.py`. `next_lineup_round5` is a COPY, not
  an edit, so a round-5 run started by another lane cannot change behaviour
  because round 6 exists. `ENGINE_ROTATION_FREEZE=1` works for round 6 exactly as
  for rounds 4 and 5. **Since no arm was adopted the flag ships unused and
  `ENGINE_ROTATION=reference` remains the default**; `engine/adapters.py` was not
  touched.
- Tests: `tests/test_rotation_v6.py` (11 cases: the tier map against the
  adapter's `digitize`, the declared tau grid, the `k_in` table's support and
  normalisation, `logA = 0` when observed equals expected, the tau argmax, the
  offline-vs-adapter entry parity for T1 at three temperatures and for A1's
  affinity, the K1 support clip, and the arm grid). `pytest tests/test_engine.py
  tests/test_rotation_v4.py tests/test_rotation_v5.py tests/test_rotation_v6.py
  -q` -> **49 passed**.
- Results: `rotation_F1_round6_{results.json,table.csv}` and
  `rotation_F1_round6_floorB.json`; audit cells in `comp_audit_2026-09-11.json`.
- Cost: the six composition fits 149-176 s each, run concurrently (6 workers);
  the seven grade/floor jobs 288-536 s each; the whole bake-off **17.5 min**.
  Each closed-loop run: 500 games x 5 seeds in 320 s on 3 workers.

### 15.15 Disclosures

1. A 60-game, 1-seed development smoke run (`--smoke`, artifacts
   `rotation_F1_round6_SMOKE_*`) was executed against the FINAL fitted objects to
   verify the code path end to end before the graded run was read. It printed
   gate cells on 2025. Its numbers are not evidence, are not cited anywhere, and
   **no model specification, gate, tolerance or arm was changed after seeing
   them** -- section 14 was committed at 4380a08 before `rotation_v6.py` existed.
   An earlier smoke process of the same script was killed by this worker (its own
   process) to remove a file race with the graded run; two orphaned workers of
   that same process were killed for the same reason. No process this worker did
   not start was signalled.
2. The reference columns for R2, W1 and W4 are taken from round 5's results JSON
   (14.4), with the 1-seed W4 reproduction check of 15.7.
3. No static column was run (14.4).
4. The 25-seed closed loop was pre-registered as conditional on the wall clock
   (14.9); see 15.13 for what was run.
5. `tau`'s likelihood is the independent-choice (multinomial) approximation to
   the sampling-without-replacement likelihood, declared in 14.2 before the fit;
   it is exact at size 1, which is 67% of all waves.
6. The three round-6 arms draw the `k_in` uniform whether or not they use it, so
   they are paired with each other and NOT byte-aligned with W4 (14.11).
7. `rotation_F1_round6_floorB.json`'s `minutes_mae` level (23.05) is the
   150-game-universe artifact the round-5 floor B carries for the same reason
   (11.6): the actual side is the full 1,600-game rotation set, so only the
   seed-to-seed and refit-to-refit DIFFERENCES are readable, never the level.

---

## 16. Round 7 pre-registration -- the EXIT side, `P(k_out | size, state)` (PM-directed, worker-authored 2026-09-11)

Written and committed BEFORE any round-7 object was fitted and before any arm
was run. Evidence it is built on: round 6's own results (section 15),
`docs/tests/rotation_composition_audit_2026-09-11.md` sections 2 and 4, and L33.

### 16.1 Why round 7 changes the EXIT RULE and nothing else

Round 6 moved the entry side and measured, on its own mechanism check, where the
error now lives (audit 4, 200 games, seed 0, the same as-of predicted starter set
on both sides so the rows are like for like):

| | P(starter in \| bench out) | P(starter in \| starter out) | spread | **share of single swaps that take a STARTER off** |
|---|---:|---:|---:|---:|
| **ACTUAL (as-of starters)** | 0.6848 | 0.3446 | 34.0 pp | **0.587** |
| W4 (round 5, unconditional race) | 0.5415 | 0.3373 | 20.4 pp | 0.464 |
| A1 (tier affinity) | 0.5683 | 0.3117 | 25.7 pp | 0.467 |
| K1 (conditional class count) | 0.6344 | 0.2026 | 43.2 pp | 0.456 |

**Every arm since round 5 removes a starter on 0.456-0.467 of single swaps
against a real 0.587**, a 12-13 pp miss, because rounds 5 and 6 both
pre-registered round 5's RANK exit rule. K1 conditions correctly on a class that
arrives with the wrong frequency: it sets the best per-player minutes MAE of any
arm in six rounds (8.8622, +63 floors over R2, +4.1 over W4) and the best
per-player minutes K-S D ever measured (0.0589), and it moves **no** state cell
(15.3, 15.12 item 3).

Round 7 therefore keeps

* round 5's **wave tables byte for byte** (`p_wave`, `p_size`, per S1 window),
* round 4's **hazards byte for byte** (`rotation_v4_sub_*.json`),
* round 3b's **base fits byte for byte**,
* round 6's **K1 entry rule byte for byte** (`P(k_in | size, k_out)` from
  `round6/rotation_v6_comp_*.json`, re-read, never re-fitted, never written to),
* the **hard second-half reset**,

and changes **only how the `size` leavers are chosen off the floor**.
Consequence, stated so it cannot be read as a coincidence later: any difference
between a round-7 arm and K1 is a difference in the **exit composition alone**.

### 16.2 The exit state, declared before it is fitted

The exit state is a **coarsening of round 5's wave cell with `prev_end`
dropped**: `exit_cell = (time_cell * 3 + margin_bucket) * 2 + foul_state`, 18
cells, with round 5's own `time_cell` (H1 / H2 20:00-08:00 / final 8:00),
`margin_bucket` (\|m\| <= 5 / 6-15 / > 15) and `foul_state` (1 when any player on
the floor carries >= 4 personal fouls). `prev_end` is dropped because the wave
DRAW already carries it and the exit CLASS has no measured dependence on it;
this is declared here, not after a fit.

Those 18 cells contain the eight round-6 gate contexts **plus the one new
measured cell round 6 named** (15.12 item 6, audit 2): the final eight minutes of
a decided game, `time_cell = 2, margin_bucket = 2`, where
P(a starter enters \| a bench player leaves) is **0.5286** (2025: 0.5024) against
0.72-0.84 in the other eight cells. The size of that cell is measured, not
guessed, and it is the reason the exit object carries a state term at all.

### 16.3 The three candidate exit rules

Baseline (K1's, and round 5's): the `size` leavers are the `size` on-floor
players with the largest fitted `p_out`, a rank rule at zero temperature, with a
foul-out override.

**X1 `exit_class` -- the exit CLASS COUNT, K1's mirror.**
`P(k_out | size, exit_cell)` is fitted as counts over the training window's own
waves, with **one-level shrinkage to `P(k_out | size)` at the project's
`k = 300`**, exactly as K1's table shrinks. At simulation time `k_out` is drawn
FIRST from the fitted row (clipped to what the floor can supply: at most `size`,
at most the number of on-floor starters, at least `size` minus the number of
on-floor non-starters), then the `k_out` starters and `size - k_out`
non-starters are chosen **by the SAME rank rule within their own class**, so the
within-class ordering is round 5's unchanged. `k_in` is then drawn from K1's
table conditioned on the realised `k_out`, byte-identical to round 6. Nothing
else changes. Foul-outs are forced out first and count toward `k_out`.

**X2 `exit_class_foul` -- the exit class conditioned on WHO is in foul trouble.**
Round 6 found foul trouble to be the one veto cell whose failure is the model's
own (15.10: the as-of-starter benchmark sits +0.2 pp from the actual there,
while the arms sit -3.2 to -4.2 pp). X2 replaces the binary `foul_state` axis of
16.2 with a three-level **foul class of the on-floor five**:
`0` = no on-floor player at >= 4 personal fouls, `1` = at least one at >= 4 but
none of them a predicted starter, `2` = at least one predicted STARTER at >= 4.
The table is `P(k_out | size, time_cell, margin_bucket, foul_class)` with
one-level shrinkage **to X1's own row** at `k = 300`, so a foul class with no
signal is X1 exactly. This is the identity of the departing player conditioned on
the foul state carried by his class, which is what 15.12 item 5 names.

**X3 `exit_class_prev` -- the exit count conditioned on the PREVIOUS wave.**
A joint over consecutive waves: `P(k_out | size, exit_cell, prev_class)` with
`prev_class` = `0` when this half has had no previous wave, `1` when the previous
wave of this half brought **0** predicted starters on, `2` when it brought **>= 1**
on, shrunk **to X1's own row** at `k = 300`. This is the mean-reversion channel a
rank rule cannot express (a starter who just came back on is not the man the rank
rule takes off next). **Conditional on support**: the fitted counts per
(size, exit_cell, prev_class) cell are reported in the results table and every
cell with n < 300 is labelled UNDERPOWERED and shrinks to X1 by construction. If
the whole `prev_class` axis is underpowered at size 1 the arm is reported as
unidentified and is not adopted on that ground.

All three are lookup tables (X1 (5, 18, 6), X2 (5, 9, 3, 6), X3 (5, 18, 3, 6));
none makes a model call in the sim loop, and each carries exactly the state terms
declared above and no others (16.10).

### 16.4 Arms

| arm | exit rule | new fitted object | simplicity | status |
|---|---|---|---:|---|
| `R2_hier_dirichlet` (S1) | -- | -- | 1 | reference (incumbent, SERVED) |
| `K1_cond_class` (S1) | rank | -- | 9 | reference (round 6's best arm; NOT served, NOT adoptable here) |
| `X1_exit_class` | class count drawn, then rank within class | (5, 18, 6) table | 11 | candidate |
| `X2_exit_class_foul` | X1 with a 3-level foul class | (5, 9, 3, 6) table | 12 | candidate |
| `X3_exit_class_prev` | X1 with the previous wave's entry class | (5, 18, 3, 6) table | 13 | candidate |

The simplicity order `R2 < K1 < X1 < X2 < X3` is fixed here. **K1 and R2 are
references and K1 is not adoptable in round 7** -- it is unchanged from round 6,
where it failed four state cells and the per-player-quintile condition.

### 16.5 Scheme, folds, and what is refitted

**Scheme: S1 for every arm**, per rounds 3b, 4, 5 and 6. Windows are the calendar
months of the 2024-25 season, a game uses the parameter set whose window closed
before its tipoff, and the first window trains on 2024 alone. No static column is
run, as in round 6.

**Folds.** F1 = train 2024, test 2025, which IS the standing fold 2; CBBD carries
no on-floor data before 2023-24 (L13), so no other fold exists. 2026 stays sealed
(`seal.assert_not_sealed` guards the trainer).

**What round 7 fits: the three exit objects, per window, and nothing else.**
`rotation_fit_v3*.json`, `rotation_v4_sub_*.json`, `round5/rotation_v5_wave_*.json`
and `round6/rotation_v6_comp_*.json` are REUSED and nothing is written to any of
them. The round-7 objects are fitted on the **same rows** as round 6's -- the
same `--wave-team-games 6000` per window, the same fit seed 11 -- so the exit and
entry tables cannot drift apart.

**The two reference columns (R2, K1) are taken from the round-6 results JSON**
(`rotation_F1_round6_results.json`), not re-simulated: same 1,600-game universe,
same subset seed 2025, same sim seeds 0-2, same base fits, hazards, wave tables
and grading functions. A **1-seed re-run of K1 is executed inside this round as a
reproduction check** and its cells are reported next to round 6's; **if any state
cell moves by more than its floor-A SD the reference columns are discarded and
the round is re-run in full.** Declared in advance, as 14.4 declared it for W4.

### 16.6 Test universe and grading path

The **same** 1,600-game subset of 2025 that rounds 2, 3, 3b, 4, 5 and 6 used
(numpy RandomState seed 2025), **3 seeds per candidate arm** under S1, **the
round-6 grading path unchanged**: `train_rotation_v1.build_row` / `verdict` /
`rotation.aggregate_stats`, extended by `train_rotation_v4.extra_cells` and
`.minutes_mae` and by `train_rotation_v5.wave_cells`. No gate cell is added, so
no grader line changes and the reference columns stay comparable byte for byte.
Any cell with n < 300 player-games or possessions is labelled UNDERPOWERED and is
never read as signal or as absence of signal.

### 16.7 Gates -- every round-6 gate, unchanged, and nothing added or relaxed

*G8 cells (report, not veto):* minutes mean +/- 2.0; minutes SD ratio pooled and
within-player 0.9-1.1; top-5 and top-8 share of team minutes +/- 2 pp; players
with > 0 minutes +/- 1.0.

*The eight state cells (the veto), each +/- 3 pp:* starters' share of on-floor
slots in the final 8:00 at \|m\| <= 5 / 6-15 / > 15; starters' share while
carrying >= 4 fouls; the second-half TIP starter share in each of the three
margin bands; starters' share over H1 20:00-10:00 at \|m\| <= 5. **An arm missing
ANY of the eight is ineligible regardless of G8 or of MAE.**

*The two round-5 cells (also veto), at the round-5/6 tolerances:*
`sub_rate_per_boundary` +/- 0.015 or 3x the floor-A seed SD if larger;
`distinct_lineups_per_game` +/- 1.5 or 3x the floor-A seed SD if larger. The
governing number is named in the results table.

*Report only:* the "at exactly 4 fouls" diagnostic; top-1 / top-3 / top-5
five-man lineup share; K-S D of the top-1 lineup share and of per-player minutes;
mean wave size; the as-of starter benchmark of 14.6 on all eight state cells.

**One report-only diagnostic is ADDED and it is the one the round exists to
move: the EXIT-SIDE STARTER SHARE.** The share of simulated swaps that take a
predicted starter off the floor, **overall and by swap size (1, 2, 3+)**, plus
the size-1 joint of audit 4, measured on ACTUAL and on every arm through one
function on 200 games at seed 0 with the as-of predicted starter set on both
sides. It changes no tolerance and no verdict; the actual side is 0.587 overall
and every arm since round 5 sits at 0.456-0.467.

### 16.8 Primary metric and the two responsiveness checks

**Primary metric: per-player minutes MAE**, unchanged from rounds 4, 5 and 6.

**Responsiveness check 1 (Decision 8), unchanged from 14.7:** team-games
bucketed into quintiles of the pregame as-of share of team minutes going to the
predicted starting five; the close-and-late cell per quintile for ACTUAL and
every arm, with slope and Q5 - Q1. An arm whose slope ratio to actual falls
outside **[0.8, 1.2]**, or whose sign disagrees, is not adoptable.

**Responsiveness check 2, per-player minutes MAE by PLAYER quintile, carried
forward from 14.7 and made STRICTLY HARDER.** Players in the as-of rotation set
are bucketed into quintiles of their own pregame as-of minutes per game. An arm
is adoptable only if it beats K1 beyond the floor in the pooled MAE **and loses
beyond the floor in no single quintile to EITHER reference it is measured
against -- K1 (this round's base) or W4 (round 6's base, whose Q2 column is the
one every round-6 arm lost)**. Carrying W4 forward is deliberate: round 6 vetoed
its own arms on that column and round 7 must not become adoptable by dropping
it. Both columns are read from the round-6 results JSON. Underpowered quintiles
are labelled.

### 16.9 Noise floors and the decision rule

**Floor A, seed-varied sim runs:** 20 seeds x 150 games per candidate arm, the SD
of every G8 cell, every state cell, both round-5 cells and the MAE -- the rounds
3/4/5/6 configuration, so five rounds' floors are comparable. R2's and K1's
floors are rounds 4's and 6's and are unchanged by a run that does not refit them.

**Floor B, spec-identical refit under a second seed:** the round-7 exit objects
refitted from a different training-game sample (fit seed 101 vs 11) and simulated
under a different sim seed (23 vs 7), graded on the same 150-game universe. Run
on **X3**, the largest new object this round adds and the one most at risk of
being unidentified, and additionally on the arm the decision rule selects if the
wall clock allows. An arm counts as beating a reference on a cell only if its
improvement exceeds the refit-to-refit spread on that cell.

**Decision rule.** Adopt the **simplest** arm that

1. passes **every** one of the eight state cells at +/- 3 pp, AND
2. passes **both** round-5 cells at the tolerances of 16.7, AND
3. beats `R2_hier_dirichlet` on per-player minutes MAE by more than the floor, AND
4. beats `K1_cond_class` on per-player minutes MAE by more than the floor and
   loses beyond the floor to neither K1 nor W4 in any player quintile (16.8), AND
5. satisfies the Decision 8 slope check, AND
6. passes the Decision 10 checks of 16.10.

Ties go to the simpler model in the order `R2 < K1 < X1 < X2 < X3`. An arm whose
improvement on the cell it was built to fix does not clear floor B is not adopted
on that cell. **If no arm is eligible, adopt nothing**, report which cell fails
and by how much, name the diagnosis, and name the next structure. No gate is
relaxed to produce a winner and no cell is dropped after seeing a result.
**The served default is not changed by this lane in any case**; the PM switches
it.

### 16.10 Decision 10: the closed loop

**What round 7 adds carries a state term BY DESIGN**, which is the difference
from round 6 and the reason the freeze is the right instrument: `P(k_out | size,
exit_cell)` contains a margin band, a time cell and a foul state. L31's
refit-without-the-feature instrument was run on the objects the state already
reached through in round 5 (13.12: W4 live/refit-without margin SD ratio 0.9941,
possessions -0.136) and that number is the size of THAT channel; what round 7
runs is the **freeze**, which is the instrument that detects whether the round-7
exit rule opens a NEW channel from the game state into the rotation.

Paired-stream runs over the fixed **500-game subset** of the F2 2025 slate
(sorted by `game_id` ascending, every 11th row, the first 500 -- the subset the
clock round-3c and the rotation round-4, 5 and 6 checks use), with
`ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round3_shooter_S_C_s1`,
`ENGINE_CLOCK=reference` pinned and recorded in every `run_meta.json`, reporting
margin SD ratio, home/away score correlation, possessions per game and per-player
minutes MAE:

| run | what it is |
|---|---|
| round-7 arm, live | the arm as it would be served |
| round-7 arm, `ENGINE_ROTATION_FREEZE=1` | margin held at 0 and the personal/team-foul counts at 0 FOR THE ROTATION MODEL ONLY; foul accrual, the foul-out eviction and the box-score counters stay live |

**Seed count, declared with its reason: 5 seeds**, the count rounds 4, 5 and 6
ran, so four rounds are one comparison. A 25-seed repeat is run **only if the
wall clock permits it**; the lane's hard stop is 12:45 ET, so it is pre-registered
as conditional and its absence is reported, not hidden. The arm run is **the arm
the decision rule selects**, or, if no arm is eligible, **the arm closest to
eligibility**, stated to be a diagnostic and not a winner.

An arm that moves margin SD ratio, home/away correlation or possessions outside
the G1/G2 tolerances between live and frozen is not adopted, and no magnitude for
the loop is quoted from the freeze alone (L31).

### 16.11 Engine expressibility (a condition on adoption)

Every rule is a vectorised array operation over the (2N, S) roster block:
`k_out` is one gathered CDF row from the exit table and one extra uniform per
(team, boundary); the within-class rank pick is two `argsort` calls on the
existing `p_out` vector instead of one; X2 needs a (2N,) foul-class bincount over
the on-floor five and X3 one (2N,) carried integer per half. All three ship
behind **`ENGINE_ROTATION=round7`** plus **`ENGINE_ROTATION_ARM=X1|X2|X3`**,
wired in `engine/rotation_adapter.py` with S1 artifacts per month and a manifest
in the `engine/manifest.py` format under
`data/processed/models/rotation/round7/`. `ENGINE_ROTATION=reference` remains the
default and `engine/adapters.py` is **not touched by this lane**. The two stated
RNG divergences of `docs/models/engine/model.md` section 4.5 apply unchanged.

### 16.12 Disclosures

1. **The three round-7 arms draw the same uniforms whether or not they use
   them.** Each draws one `k_out` uniform per wave before the `k_in` uniform, so
   the three arms sit at identical stream positions and are paired with each
   other. The cost, stated: a round-7 arm is **NOT** byte-aligned with round 6's
   K1, which is why K1's column comes from round 6's own JSON and is checked by
   the 1-seed reproduction re-run of 16.5.
2. `k = 300` is the project's UNDERPOWERED threshold, identical to rounds 5 and
   6, fixed here before any fit and never tuned. The foul class boundary (>= 4
   personal fouls) is round 5's own `foul_state` boundary, not a new constant.
3. The exit state drops `prev_end` from round 5's wave cell (16.2), declared
   before the fit.
4. No static column is run; the 25-seed closed loop is conditional on the wall
   clock (16.10).
5. The exit-side starter-share diagnostic (16.7) is measured on 200 games at seed
   0, the same configuration as audit 4, and is report-only.
6. X3 is conditional on table support (16.3) and its cell counts are published
   whether or not it is adopted.
7. Round 6's per-player-quintile condition is carried forward against BOTH K1 and
   W4 (16.8); this makes round 7 harder to pass than round 6, never easier.

---


## 17. Round-7 results (train 2024, test 2025) -- run 2026-09-11T15:34Z

Pre-registration section 16, committed **2aa29c2** before `rotation_v7.py`
existed and before anything was fitted. Evidence doc:
`docs/tests/rotation_exit_audit_2026-09-11.md`.

Test universe: the **same** 1,600-game subset of 2025 rounds 2, 3, 3b, 4, 5 and 6
used (numpy RandomState seed 2025), 3 seeds per candidate arm under S1, the
round-6 grading path unchanged. Windows and games: 202411 147, 202412 297,
202501 438, 202502 444, 202503 267, 202504 7 (UNDERPOWERED at 7 games, reported
only because it exists). Round 6's composition tables, round 5's wave tables,
round 4's hazards and round 3b's base fits are reused and never written to;
round 7 fits only the three exit objects, on 6,000 team-games per window (fit
seed 11), **124,260-126,402 waves per window of which 98.2% are usable -- the
same counts, window by window, that round 6 reported (15, first paragraph), so
the exit and entry tables are fitted on identical rows**, as 16.5 required.

**ACTUAL on this universe**, through the same functions as every arm:
substitutions per boundary **0.1509**, distinct lineups per team-game
**14.8356**, minutes mean 24.551, top-5 share 0.7472, top-8 0.9560, players with
> 0 minutes 9.6438, pooled minutes SD 9.6946, within-player 6.1196.

### 17.1 The fitted object: the exit class IS state-dependent, and the state is the one round 6 named

`P(k_out = 1 | size 1, cell)` per exit cell, window 202411 (the six windows agree
to 0.1-1.5 pp on every cell; the seed-101 refit agrees to 0.1-0.7 pp, 17.6):

| time cell | \|m\| <= 5 | 6-15 | > 15 | n (no foul trouble) |
|---|---:|---:|---:|---:|
| H1 | **0.6247** | 0.5781 | 0.5401 | 20,321 / 14,983 / 1,856 |
| H2 20:00-08:00 | 0.5961 | 0.6088 | 0.6381 | 8,762 / 10,542 / 4,451 |
| final 8:00 | **0.4851** | 0.5085 | 0.5748 | 5,199 / 5,721 / 2,749 |

and with a player on the floor at >= 4 personal fouls:

| time cell | \|m\| <= 5 | 6-15 | > 15 | n (foul trouble) |
|---|---:|---:|---:|---:|
| H1 | 0.5864 | 0.5864 | 0.5884 | **1 / 1 / 0** (structurally empty, shrunk to the parent) |
| H2 20:00-08:00 | 0.6100 | 0.5896 | 0.6217 | 173 / 264 / 193 (all UNDERPOWERED) |
| final 8:00 | **0.5792** | 0.6036 | 0.6143 | 3,147 / 2,913 / 868 |

Three measured facts, none of them guessed:

1. **The exit class carries a state term of 13.9 pp**: a single swap takes a
   starter off 0.6247 of the time in the first half and 0.4851 in the final eight
   minutes of a close game. The exit side has its own version of the entry side's
   decided-game cell (audit 2), and it runs the other way: **starters stop
   leaving** when the game is close and late.
2. **Foul trouble raises the starter's chance of being the man taken off by
   9.4 pp in the close-and-late cell** (0.4851 -> 0.5792, n = 5,199 / 3,147),
   which is the effect X2 was pre-registered to carry.
3. **Foul trouble in H1 does not exist at this threshold** (n = 1, 1, 0 rows in
   122,007): a player reaching 4 personal fouls before half-time is a
   once-a-window event, so three of X2's eighteen cells are structurally empty
   and shrink to X1 exactly. Declared, not discovered after the fact: 16.3 fixed
   the shrinkage and 16.7 fixed the UNDERPOWERED label.

X3's support, published as 16.3 required: **25 of the 54 (exit cell x prev class)
cells at size 1 are under 300 rows** and shrink to X1. The axis is half
unidentified, and X3's every cell below is X1's to within the floor, which is
what that means in practice.

### 17.2 G8 cells (report, not veto) -- S1

| cell | tol | ACTUAL | R2 | K1 | X1 | X2 | X3 |
|---|---|---:|---:|---:|---:|---:|---:|
| minutes mean (rotation players) | +/- 2.0 | 24.551 | 25.364 P | 25.636 P | 24.623 P | 24.631 P | 24.637 P |
| minutes SD ratio, pooled | 0.9-1.1 | 1.0000 | 1.0548 P | 0.9961 P | 0.8853 **F** | 0.8837 **F** | 0.8855 **F** |
| minutes SD ratio, within-player | 0.9-1.1 | 1.0000 | 1.3416 **F** | 1.1239 **F** | 1.1447 **F** | 1.1422 **F** | 1.1447 **F** |
| top-5 share of team minutes | +/- 2 pp | 0.7472 | 0.7630 P | 0.7597 P | 0.7232 **F** | 0.7230 **F** | 0.7232 **F** |
| top-8 share of team minutes | +/- 2 pp | 0.9560 | 0.9626 P | 0.9696 P | 0.9552 P | 0.9553 P | 0.9554 P |
| players with > 0 minutes | +/- 1.0 | 9.644 | 9.129 P | 8.935 P | 9.033 P | 9.031 P | 9.029 P |
| **G8 passed** | | | **5/6** | **5/6** | 3/6 | 3/6 | 3/6 |

The three round-7 arms miss top-5 share on the **other side** from every arm in
six rounds: 0.7232 against 0.7472, 2.4 pp LOW, where W4 was 2.3 pp high and K1
1.3 pp high. The pooled minutes SD falls to 0.885 of the real one. Both say the
same thing as 17.3 -- minutes are spread too evenly across the roster.

### 17.3 State cells (the veto) -- S1

ACTUAL, then each arm with its gap in pp and PASS/FAIL at +/- 3 pp:

| cell | ACTUAL | R2 | K1 | X1 | X2 | X3 |
|---|---:|---:|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7491 | 0.7254 (-2.4) P | 0.7001 (-4.9) F | **0.5394 (-21.0) F** | 0.5434 (-20.6) F | 0.5300 (-21.9) F |
| final 8:00, \|m\| 6-15 | 0.7240 | 0.6914 (-3.3) F | 0.6890 (-3.5) F | **0.5384 (-18.6) F** | 0.5431 (-18.1) F | 0.5278 (-19.6) F |
| final 8:00, \|m\| > 15 | 0.5223 | 0.4694 (-5.3) F | 0.5083 (-1.4) P | 0.5183 (-0.4) P | 0.5181 (-0.4) P | 0.5106 (-1.2) P |
| starters at >= 4 fouls | 0.4613 | 0.4626 (+0.1) P | 0.4212 (-4.0) F | 0.3823 (-7.9) F | 0.3752 (-8.6) F | 0.3832 (-7.8) F |
| H2 TIP, \|m\| <= 5 | 0.9678 | 0.7892 (-17.9) F | 0.9385 (-2.9) P | 0.9406 (-2.7) P | 0.9410 (-2.7) P | 0.9397 (-2.8) P |
| H2 TIP, \|m\| 6-15 | 0.9611 | 0.7977 (-16.3) F | 0.9410 (-2.0) P | 0.9424 (-1.9) P | 0.9420 (-1.9) P | 0.9424 (-1.9) P |
| H2 TIP, \|m\| > 15 | 0.9563 | 0.7338 (-22.3) F | 0.9456 (-1.1) P | 0.9449 (-1.1) P | 0.9452 (-1.1) P | 0.9430 (-1.3) P |
| H1 20:00-10:00, \|m\| <= 5 | 0.7822 | 0.5988 (-18.3) F | 0.7267 (-5.6) F | 0.6991 (-8.3) F | 0.6987 (-8.4) F | 0.6983 (-8.4) F |
| **state cells passed** | | **2/8** | **4/8** | **4/8** | **4/8** | **4/8** |
| (diagnostic) at exactly 4 fouls | 0.5166 | 0.5722 | 0.5931 | **0.5316** | **0.5186** | **0.5328** |

R2's and K1's columns are round 6's, per 16.5; the reproduction check is 17.7.
Every round-7 arm passes the same four cells K1 passes and **fails the two
close-and-late bands by 18-22 pp, six times K1's miss**. One cell moves the other
way and it is the one round 6 named as the model's own: the **"at exactly 4
fouls" diagnostic goes from 0.593 (K1) to 0.519-0.533 against a real 0.5166**,
the first time in seven rounds any arm has reproduced it; the graded
foul-trouble cell nevertheless gets worse (-7.8 to -8.6 pp), for the reason
17.10 gives.

### 17.4 The two round-5 cells, the primary metric, and concentration -- S1

Tolerances, computed as pre-registered: `sub_rate_per_boundary` **+/- 0.015**
governs (3x floor = 0.0063); `distinct_lineups_per_game` **+/- 1.5** governs
(3x floor = 0.719).

| metric | ACTUAL | R2 | K1 | X1 | X2 | X3 |
|---|---:|---:|---:|---:|---:|---:|
| **substitutions per boundary** | 0.1509 | 0.1437 P | 0.1554 P | 0.1557 P | 0.1558 P | 0.1559 P |
| **distinct lineups per team-game** | 14.836 | 15.514 P | **14.571 P** | 15.939 P | 15.960 P | 15.945 P |
| **per-player minutes MAE (min)** | 0.0 | 9.7939 | **8.8622** | 9.6159 | 9.5991 | 9.6397 |
| MAE gain over R2 (floors) | -- | -- | +63 | **+11.0** | +12.6 | +10.5 |
| MAE gain over K1 (floors) | -- | -- | -- | **-46.5** | -47.7 | -57.7 |
| top-1 lineup share | 0.2940 | 0.2286 | 0.2568 | 0.2345 | 0.2342 | 0.2331 |
| top-3 lineup share | 0.5426 | 0.4819 | 0.5358 | 0.4995 | 0.4988 | 0.4976 |
| top-5 lineup share | 0.6894 | 0.6409 | 0.7043 | 0.6663 | 0.6657 | 0.6645 |
| K-S D, per-player minutes | -- | 0.0798 | 0.0589 | **0.0441** | 0.0447 | 0.0443 |
| K-S D, top-1 lineup share | -- | 0.2273 | **0.0990** | 0.1993 | 0.2010 | 0.2040 |

Read those two K-S rows together with the MAE row, because they are the round's
signature. **The round-7 arms set the best per-player minutes DISTRIBUTION ever
measured (K-S D 0.0441 against K1's 0.0589) and lose 0.75 minutes of per-player
MAE**, 47-58 floors. A model that gets the marginal distribution of minutes right
and the assignment wrong is putting the right number of minutes on the wrong
players, which is exactly what 17.9 and 17.10 measure directly.

### 17.5 Noise floor A (20 seeds x 150 games, S1)

| metric | X1 | X2 | X3 |
|---|---:|---:|---:|
| minutes_mae | 0.01621 | 0.01544 | 0.01233 |
| sub_rate_per_boundary | 0.00202 | 0.00210 | 0.00155 |
| distinct_lineups_per_game | 0.2396 | 0.2381 | 0.1772 |
| late_starter_share_b0 | 0.01946 | 0.01930 | 0.02030 |
| late_starter_share_b1 | 0.01353 | 0.01367 | 0.01235 |
| late_starter_share_b2 | 0.02197 | 0.02158 | 0.02384 |
| foul_trouble_share | 0.02003 | 0.01763 | 0.01827 |
| h2tip_starter_share_b0 | 0.00793 | 0.00797 | 0.00741 |
| h2tip_starter_share_b1 | 0.00882 | 0.00821 | 0.00898 |
| h2tip_starter_share_b2 | 0.01570 | 0.01463 | 0.01884 |
| opentip_starter_share_close | 0.01460 | 0.01446 | 0.01294 |
| top5_share | 0.00263 | 0.00281 | 0.00266 |
| n_nonzero_mean | 0.06531 | 0.06809 | 0.06117 |

R2's and K1's floors are rounds 4's and 6's and are unchanged by a run that does
not refit them (R2 minutes_mae 0.01473, K1 0.01348). **Every miss in 17.3 is far
larger than its floor**: X1's close-band miss is -21.0 pp against a 1.95 pp floor
(10.8 floors) and its foul-trouble miss -7.9 pp against 2.00 pp (3.9 floors). The
floor-A caveat of 11.6 applies unchanged: only the seed-to-seed SD is a floor,
never the `minutes_mae` LEVEL of that 150-game run.

### 17.6 Noise floor B -- spec-identical exit refit under a second seed (X3)

A different training-game sample (fit seed 101 against 11) and a different sim
seed (23 against 7), graded on the same 150-game universe, run in its own process
so the bake-off JSON was on disk before the second fit started.

| cell | ACTUAL (150 games) | X3 seed 1 | X3 seed 2 | \|delta\| |
|---|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7436 | 0.5214 | 0.5551 | 3.37 pp |
| final 8:00, \|m\| 6-15 | 0.7186 | 0.5187 | 0.5385 | 1.98 pp |
| final 8:00, \|m\| > 15 | 0.5518 | 0.5246 | 0.4615 | 6.31 pp |
| starters at >= 4 fouls | 0.4362 | 0.3948 | 0.4261 | 3.13 pp |
| H2 tip, \|m\| <= 5 | 0.9655 | 0.9310 | 0.9603 | 2.93 pp |
| H2 tip, \|m\| 6-15 | 0.9662 | 0.9284 | 0.9338 | 0.54 pp |
| H2 tip, \|m\| > 15 | 0.9611 | 0.9611 | 0.9500 | 1.11 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0.7710 | 0.6939 | 0.6910 | 0.29 pp |
| substitutions per boundary | -- | 0.1515 | 0.1560 | 0.0044 |
| distinct lineups per team-game | -- | 15.390 | 15.977 | 0.587 |
| per-player minutes MAE | -- | 23.1128 | 23.0974 | 0.0154 |

**The round-7 objects are identified.** The fitted table moves 0.1-0.7 pp per
cell under the refit (`P(k_out = 1 | size 1)`: 0.6247 -> 0.6240 in the H1 close
cell, 0.4851 -> 0.4860 in the final-8:00 close cell), and the cell spread
(0.29-6.31 pp) is round 6's (0.00-3.10 pp) widened by the two thinnest cells of a
150-game universe. **The decision does not turn on a refit artefact in either
direction**: the close-band miss the round fails on is -21 pp, three times the
largest refit-to-refit move measured here and 11 floor-A SDs.

### 17.7 The K1 reproduction check (16.5)

A 1-seed re-run of K1 inside this round, against round 6's own 3-seed column:

| cell | round 6 (3 seeds) | round 7 (1 seed) | delta | floor A |
|---|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7001 | 0.7017 | +0.0015 | 0.0109 |
| final 8:00, \|m\| 6-15 | 0.6890 | 0.6861 | -0.0029 | 0.0096 |
| final 8:00, \|m\| > 15 | 0.5083 | 0.5083 | -0.0000 | 0.0176 |
| starters at >= 4 fouls | 0.4212 | 0.4196 | -0.0016 | 0.0186 |
| H2 tip, \|m\| <= 5 | 0.9385 | 0.9385 | +0.0001 | 0.0082 |
| H2 tip, \|m\| 6-15 | 0.9410 | 0.9370 | -0.0040 | 0.0067 |
| H2 tip, \|m\| > 15 | 0.9456 | 0.9498 | +0.0042 | 0.0197 |
| H1 20:00-10:00, \|m\| <= 5 | 0.7267 | 0.7301 | +0.0034 | 0.0076 |
| substitutions per boundary | 0.1554 | 0.1545 | -0.0008 | 0.0020 |
| distinct lineups | 14.571 | 14.467 | -0.1041 | 0.1842 |

**Every state cell moves less than its floor-A SD, so the reference columns
stand** (the largest move, the tip at \|m\| 6-15, is -0.40 pp against a 0.67 pp
floor). The MAE moves +0.056 on one seed against a 0.013 floor, the expected
1-vs-3-seed Monte-Carlo difference and the same reading rounds 5 and 6 gave.

### 17.8 The two responsiveness conditions

**Decision 8 (16.8 check 1).** Cell = starters' share in the final 8:00 at
\|margin\| <= 5, by quintile of the pregame as-of predicted starter-minutes share
(640 team-games per quintile).

| quintile | prior | ACTUAL | K1 | X1 | X2 | X3 |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 0.5572 | 0.6770 | 0.6218 | 0.4921 | 0.5010 | 0.4864 |
| Q2 | 0.6342 | 0.7226 | 0.6940 | 0.5280 | 0.5288 | 0.5193 |
| Q3 | 0.6698 | 0.7471 | 0.7089 | 0.5372 | 0.5403 | 0.5255 |
| Q4 | 0.7044 | 0.7763 | 0.7299 | 0.5563 | 0.5646 | 0.5510 |
| Q5 | 0.7685 | 0.8219 | 0.7430 | 0.5826 | 0.5819 | 0.5673 |
| **slope** | | **+0.692** | +0.576 | +0.426 | +0.395 | +0.390 |
| slope ratio to actual | | 1.00 | **0.83 PASS** | **0.62 FAIL** | **0.57 FAIL** | **0.56 FAIL** |
| Q5 - Q1 (pp) | | +14.5 | +12.1 | +9.1 | +8.1 | +8.1 |

Every arm is still monotone in 4 of 4 steps, but **all three round-7 arms fall
out of the [0.8, 1.2] band for the first time since round 5 vetoed W3 and W5**.
The exit draw flattens the response to the team's own rotation depth: a team
whose starters take 77% of the minutes is simulated 24 pp below its actual late
share, the same as a team at 56%.

**The per-player-quintile condition (16.8 check 2).** Per-player minutes MAE by
quintile of the player's own pregame as-of minutes per game (edges 15.97 / 20.89
/ 25.33 / 29.71 mpg; 4,782-4,784 player-games per quintile, none underpowered):

| quintile | K1 (r6) | W4 (r6) | X1 | X2 | X3 | X1 - K1 | X1 - W4 | floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 (lowest mpg) | 9.3108 | 9.6681 | 10.6053 | 10.5868 | 10.5998 | **+1.295** | +0.937 | 0.016 |
| Q2 | 9.7891 | 9.7294 | 10.4615 | 10.4331 | 10.4896 | **+0.672** | +0.732 | 0.016 |
| Q3 | 9.2575 | 9.4074 | 9.7867 | 9.7658 | 9.8029 | +0.529 | +0.379 | 0.016 |
| Q4 | 8.6461 | 8.6621 | 9.1081 | 9.1084 | 9.1732 | +0.462 | +0.446 | 0.016 |
| Q5 (highest mpg) | 7.2367 | 7.3307 | 8.0130 | 7.9973 | 8.0315 | +0.776 | +0.682 | 0.016 |

**All three arms lose all five quintiles to both references, by 24 to 81 floors.**
There is no quintile in which drawing the exit class pays.

### 17.9 Per-game and per-team evidence

**Per game** (per-player minutes MAE per game, averaged over seeds): K1 8.868
+/- 2.239 with a 90th percentile of 11.288; **X1 9.644 +/- 2.368 / 12.225**;
X2 9.628 +/- 2.379 / 12.226; X3 9.668 +/- 2.368 / 12.278. The loss is the whole
distribution, not a tail: the median game and the 90th-percentile game move by
about the same 0.8-0.9 minutes.

**Per team** (starters' share in the final 8:00 at \|m\| <= 5, aggregated per
team; a team is powered when both denominators reach 300 on-floor slots --
**251 powered, 112 UNDERPOWERED and excluded**):

| | sim mean | sim SD | actual mean | actual SD | corr(sim, actual) | mean \|dev\| |
|---|---:|---:|---:|---:|---:|---:|
| K1 (round 6) | 0.7004 | 0.0507 | 0.7512 | 0.0832 | **0.394** | 0.0757 |
| X1 | 0.5412 | 0.0599 | 0.7512 | 0.0832 | 0.209 | 0.2108 |
| X2 | 0.5451 | 0.0619 | 0.7512 | 0.0832 | 0.195 | 0.2067 |
| X3 | 0.5314 | 0.0614 | 0.7512 | 0.0832 | 0.198 | 0.2203 |

The cross-team SD rises slightly (0.051 -> 0.060 against a real 0.083) and the
correlation with the team's own actual share **halves**, 0.394 -> 0.195-0.209.
The extra dispersion is not matchup information; it is the drift of 17.10 varying
game to game.

### 17.10 The exit-side diagnostic (16.7): the arms hit the target and lose the cells

`scripts/diag_rotation_exit_v7.py`, 200 games, seed 0, the as-of predicted
starter set on BOTH sides, one function for every row. This is the object round 7
was built to move:

| | starter share of LEAVERS, overall | size 1 | size 2 | size 3+ | starter share of ENTRANTS | size-1 joint (bench out / starter out) | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ACTUAL (as-of starters)** | **0.5576** | **0.5867** | **0.5576** | **0.4931** | **0.4972** | 0.6848 / 0.3446 | 34.0 pp |
| K1 (round 6) | 0.4927 | 0.4564 | 0.5142 | 0.5496 | 0.4334 | 0.6344 / 0.2026 | 43.2 pp |
| **X1** | **0.5611** | 0.6013 | 0.5648 | 0.4616 | **0.4882** | 0.6762 / 0.3073 | 36.9 pp |
| X2 | 0.5591 | 0.5979 | 0.5609 | 0.4662 | 0.4874 | 0.6765 / 0.3050 | 37.1 pp |
| X3 | 0.5585 | 0.5938 | 0.5552 | 0.4805 | 0.4881 | 0.6917 / 0.3026 | 38.9 pp |

**The 12-13 pp exit-side defect round 6 named is closed.** X1 removes a starter
on 0.5611 of all departures against a real 0.5576 (K1: 0.4927) and on 0.6013 of
single swaps against a real 0.5867 (K1: 0.4564); the ENTRY side, which round 7
did not touch, comes along with it (0.4334 -> 0.4882 against 0.4972) because K1's
conditional table is finally fed the right `k_out` mix, and the size-1 joint
spread falls from K1's 43.2 pp to 36.9 pp against a real 34.0 pp. **Every
swap-level class marginal is now right to within 1.5 pp on both sides, and the
on-floor composition is 21 pp wrong.**

The two facts are reconciled by the time-since-reset gradient, which is the
mechanism reading of the round:

| cell | minutes since the last forced reset | X1 gap | K1 gap |
|---|---:|---:|---:|
| H2 tip, all three margin bands | 0 | -1.1 to -2.7 pp | -1.1 to -2.9 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0-10 | -8.3 pp | -5.6 pp |
| final 8:00, \|m\| <= 5 | 12+ | **-21.0 pp** | -4.9 pp |

X1 is correct at the reset and drifts monotonically away from it, six times
faster than K1 by the final eight minutes. A class-count draw conditioned on
(size, state) is a **level, not a rate**: it removes starters at the population
frequency no matter how many are on the floor at that moment, so when the floor
has drifted bench-heavy the same draw is far above proportional and nothing pulls
it back. Round 5's rank rule carried an implicit restoring force -- `p_out` rises
with time on the floor and `p_in` with the player's own minutes share, so a
starter who sits comes back -- and round 7 replaced that force with a frequency
that is only correct in aggregate. Round 6's own lesson (L36) in the mirror:
**condition on an input your model gets wrong and you inherit its error; replace
a self-correcting rule with a correct average and you lose the correction.**

### 17.11 The as-of starter benchmark (report only, 16.7)

The ACTUAL sequence of the same 1,600 games, re-graded with the MODEL's as-of
predicted starting five instead of the game's own; the model's five overlaps the
real five on **4.576 of 5**, unchanged from round 6. Starter identification costs
-2.8 pp on both close-and-late bands, -4.4 pp on the opening ten minutes, -6.5 to
-7.0 pp on the H2 tip cells and **+0.2 pp on foul trouble**. Against that
benchmark the round-7 arms are **-18.2 pp** on the close band (K1: -2.1) and
-3.9 pp on the opening cell (K1: -1.2). **It changes no tolerance and no
verdict**: every arm is scored against each side's own real starting five, as in
rounds 1-6.

### 17.12 Decision

| arm | simplicity | state | round-5 cells | G8 | MAE | vs R2 | vs K1 | quintiles | D8 | Dec-10 | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| R2_hier_dirichlet | 1 | 2/8 | 2/2 | 5/6 | 9.7939 | -- | -0.932 | ok | n/a | -- | NO (reference) |
| K1_cond_class | 9 | 4/8 | 2/2 | 5/6 | **8.8622** | +0.932 | -- | see 15.8 | 0.83 | PASS (15.13) | NO (reference) |
| X1_exit_class | 11 | 4/8 | 2/2 | 3/6 | 9.6159 | +0.178 | **-0.754** | **5/5 fail** | **0.62 F** | NOT RUN | NO |
| X2_exit_class_foul | 12 | 4/8 | 2/2 | 3/6 | 9.5991 | +0.195 | **-0.737** | **5/5 fail** | **0.57 F** | NOT RUN | NO |
| X3_exit_class_prev | 13 | 4/8 | 2/2 | 3/6 | 9.6397 | +0.154 | **-0.778** | **5/5 fail** | **0.56 F** | NOT RUN | NO |

**No arm adopted**, and no arm is close: each fails condition 1 (four of eight
state cells, two of them by 18-22 pp), condition 4 (loses 0.74-0.78 minutes of
MAE to K1, and every player quintile to both K1 and W4) and condition 5 (Decision
8 slope ratio 0.56-0.62 against a [0.8, 1.2] band). They pass conditions 2 and 3.
Per 16.9 the rule adopts nothing; no gate was relaxed and no cell was dropped
after seeing a result. `ENGINE_ROTATION=reference` (R2) stays the served default
and **this lane changed no default**; the `round7` flag ships nowhere, because
16.11's adapter was not wired (17.13).

Cell-by-cell misses of the closest arm:

| arm | cell | sim | actual | miss | floors | of which starter identification |
|---|---|---:|---:|---:|---:|---:|
| X1 | final 8:00 \|m\| <= 5 | 0.5394 | 0.7491 | -21.0 pp | 10.8 | -2.8 pp |
| X1 | final 8:00 \|m\| 6-15 | 0.5384 | 0.7240 | -18.6 pp | 13.7 | -2.8 pp |
| X1 | starters at >= 4 fouls | 0.3823 | 0.4613 | -7.9 pp | 3.9 | +0.2 pp (none) |
| X1 | H1 20:00-10:00 \|m\| <= 5 | 0.6991 | 0.7822 | -8.3 pp | 5.7 | -4.4 pp |
| X1 | player quintile Q1 | 10.6053 | (K1 9.3108) | +1.295 | 80 | -- |
| X1 | D8 slope ratio | 0.615 | (band 0.8-1.2) | -0.185 | -- | -- |

### 17.13 Decision 10: NOT RUN, and what that costs

16.10 pre-registered a 5-seed paired live/frozen closed loop on the arm the
decision rule selects or, failing that, the arm closest to eligibility. **It was
not run.** The reason is the wall clock and nothing else: serving a round-7 arm
in the engine needs a vectorised `next_lineup_round7` in
`engine/rotation_adapter.py` (round 6's equivalent is ~250 lines), its parity
tests, and two 500-game paired runs at about 11 minutes, and the lane's hard stop
is 12:45 ET. **The absence is reported, not hidden**, per 16.10.

What it costs is stated precisely: condition 6 of the decision rule is unmet for
every arm, so **no arm could have been adopted today even on a clean sheet of
offline cells**. It costs nothing on this round's actual outcome, because every
arm already fails conditions 1, 4 and 5 by 4 to 80 floors, and an arm that fails
the offline gates is not adoptable whatever the freeze says. If a later round
revives the exit family, the freeze must be run before adoption, and its cost --
the adapter plus two runs -- must be inside that round's budget from the start.

### 17.14 Diagnosis

**1. The exit-side defect is closed, and closing it was not enough.** The object
round 6 named (audit 4) is fixed to within 1.5 pp on both class marginals and at
every swap size (17.10), and the entry side improved for free. This is the
cleanest confirmation available that round 6's diagnosis of WHERE the error lived
was right, and it is also the round's refutation: **being right about the
marginal was not the same as being right about the mechanism.**

**2. A class-count draw is a level; the rule it replaced was a rate.** The
time-since-reset gradient (17.10) measures it: 0 pp of drift at the reset, -8.3
pp ten minutes in, -21.0 pp twelve minutes after the H2 reset. Nothing in
`P(k_out | size, state)` depends on how many starters are on the floor when it is
drawn, so a floor that drifts bench-heavy keeps shedding starters at the
population rate. The next object is therefore **`P(k_out | size, state,
n_starters_on_floor)`** -- the same table with the composition as a fourth axis,
which makes the draw proportional-by-construction and restores the correcting
force the rank rule had. It is fittable from the same rows (the on-floor count is
already in the training loop) and its support is ample: the size-1 cells here
carry 1,856-20,321 rows before the composition split.

**3. The right marginal distribution with the wrong assignment is a measurable
failure mode, and both instruments saw it.** Per-player minutes K-S D falls to
0.0441, the best ever measured, while per-player minutes MAE rises 0.75 and every
quintile loses (17.4, 17.8). A distributional metric alone would have called
round 7 a win. This is the multi-level-evidence rule earning its keep.

**4. The exit class DOES carry the state term the pre-registration guessed at,
and a second one it did not.** Starters leave on 0.6247 of H1 single swaps and
0.4851 in the final eight minutes of a close game (13.9 pp), and foul trouble
adds 9.4 pp on top in that same cell (17.1). Those numbers survive the refit
(17.6) and are worth keeping whatever family consumes them.

**5. Foul trouble: the benching response is now right and the graded cell is
worse.** "At exactly 4 fouls" goes 0.593 -> 0.519 against a real 0.5166, the
first arm in seven rounds to reproduce it, while the graded starters-at-4-fouls
cell falls to 0.375-0.383 against 0.4613. Both readings are consequences of the
same change: the model now takes the fouled starter off at the right rate, and it
does not put him back, because the drift of item 2 governs who is on the floor
when he would return. **The foul cell will not be readable until the drift is
fixed**, which reorders the queue: composition first, then the foul response.

**6. X2 and X3 add nothing over X1.** X2's foul class moves the close-late cell
by +0.4 pp and the MAE by -0.017 (inside the floor); X3's previous-wave axis is
underpowered in 25 of 54 size-1 cells and moves every cell inside the floor. The
simpler arm is the whole family, which is the tie-break 16.9 declared and, here,
the entire result of the two refinements.

### 17.15 Artifacts and flags

- Exit objects: `data/processed/models/rotation/round7/rotation_v7_exit_{YYYYMM}.json`
  (six windows) with `rotation_v7_manifest.json` in the `engine/manifest.py`
  format; each entry carries `refit_date`, `max_train_date`, the reused hazard,
  wave and composition artifact names, the published `P(k_out=1|size 1)` row and
  its cell counts, and X3's underpowered-cell count. The directory is gitignored
  (`data/processed/models/*/round*/`, 644 KB including the floor-B siblings) and
  HF-synced.
- Floor-B sibling: `round7/rotation_v7_exit_seed2_*.json` +
  `rotation_v7_manifest_seed2.json`.
- Nothing was written to `rotation_fit.json`, `rotation_fit_v3*.json`, any
  `rotation_v4_sub_*.json`, any `round5/rotation_v5_wave_*.json` or any
  `round6/rotation_v6_comp_*.json`.
- Engine: **not wired** (17.13). `ENGINE_ROTATION=round7` does not exist,
  `engine/adapters.py` and `engine/loop.py` were not touched by this lane, and
  `ENGINE_ROTATION=reference` remains the default.
- Tests: `tests/test_rotation_v7.py` (8 cases: the 18-cell exit state and its
  overtime mapping, the blowout cell's index, the three-level foul class, row
  normalisation and support, X2's and X3's exact shrinkage to X1, the declared
  constants, thin-vs-thick cell behaviour, the sampler's class clipping, and the
  arm grid). `pytest tests/test_rotation_v5.py tests/test_rotation_v6.py
  tests/test_rotation_v7.py -q` -> **29 passed**.
- Results: `rotation_F1_round7_{results.json,table.csv}`,
  `rotation_F1_round7_floorB.json`, `exit_audit_2026-09-11.json`.
- Cost: the six exit fits 54-70 s each, run concurrently (6 workers); the seven
  grade/floor jobs 218-467 s each; the whole bake-off **15.0 min**; the exit
  diagnostic 2.3 min; floor B 6.4 min.

### 17.16 Disclosures

1. A 60-game, 1-seed development smoke run (`--smoke`, artifacts
   `rotation_F1_round7_SMOKE_*`) was executed before the graded run to verify the
   code path end to end. It printed gate cells on 2025. Its numbers are not
   evidence, are not cited anywhere, and **no model specification, gate,
   tolerance or arm was changed after seeing them** -- section 16 was committed at
   2aa29c2 before `rotation_v7.py` existed. The smoke's own exit artifacts (400
   team-games per window) were overwritten by the graded run's 6,000-team-game
   fits in the same process, before any grading job read them.
2. The reference columns for R2 and K1 are taken from round 6's results JSON
   (16.5), with the 1-seed K1 reproduction check of 17.7.
3. No static column was run (16.5).
4. **Decision 10 was not run at all** (17.13), not merely the conditional 25-seed
   repeat. No closed-loop claim is made for any round-7 arm.
5. The three round-7 arms draw the `k_out` uniform before K1's `k_in` uniform, so
   they are paired with each other and NOT byte-aligned with round 6's K1
   (16.12 item 1).
6. X3's `prev_class` axis is underpowered in 25 of its 54 size-1 cells (17.1),
   which 16.3 pre-registered as the condition under which the arm is reported as
   unidentified. It is so reported, and it is not adopted on that ground as well
   as on the others.
7. `rotation_F1_round7_floorB.json`'s `minutes_mae` level (23.1) is the
   150-game-universe artifact rounds 5 and 6 carry for the same reason (11.6):
   the actual side is the full 1,600-game rotation set, so only the seed-to-seed
   and refit-to-refit DIFFERENCES are readable, never the level.
8. No process this worker did not start was signalled, and no existing data or
   results file was overwritten: every round-7 artifact is a new versioned
   sibling.

---


### 17.17 Post-round descriptive measurement: the support check for the next object

Measured AFTER the round's decision was read and written, on the ACTUAL
sequences of both seasons with each game's OWN starting five and its own
participant pool -- the descriptive convention of the round-6 composition audit
(`scripts/diag_rotation_exit_v7.py --by-composition`). **It is a measurement of
coaching behaviour, not a bake-off result; it changes no verdict, no tolerance
and no arm above.** It exists because 17.14 item 2 names an object and the
project does not pre-register an object whose support has not been measured.

`P(a starter is the man who leaves | single swap, starters on the floor)`:

| starters on the floor | 2024 | n | 2025 | n | proportional | 2025 / proportional |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3689 | 4,072 | 0.3697 | 5,981 | 0.20 | **1.85** |
| 2 | 0.4295 | 17,101 | 0.4374 | 23,159 | 0.40 | 1.09 |
| 3 | 0.5173 | 38,567 | 0.5209 | 48,014 | 0.60 | 0.87 |
| 4 | 0.6433 | 41,296 | 0.6562 | 48,261 | 0.80 | 0.82 |
| 5 | 1.0000 | 17,418 | 1.0000 | 19,996 | 1.00 | 1.00 |

Every cell is powered (4,072-48,261 leavers) and the two seasons agree to
0.8-1.3 pp on all five. The same split by time cell holds the shape at every
point of the game (2024: at three starters on the floor, 0.5396 in H1, 0.4487 in
H2 20:00-08:00, 0.4500 in the final 8:00; at four, 0.7405 / 0.6836 / 0.6537).

**The composition axis is worth 29 pp, twice the 13.9 pp time-and-margin term
round 7 did model**, and round 7's own table is a mixture over it: the 0.5576
marginal X1 reproduces is the average of 0.37 at one starter and 0.66 at four.

The arithmetic that closes 17.14 item 2, using the measured entry joint of the
round-6 audit (P(starter in \| bench out) 0.737, P(starter in \| starter out)
0.323 on 2025) and the exit rates above. Expected change in starters on the
floor per single swap:

| starters on the floor | REAL exit rate | E[change] | LEVEL form (X1) | E[change] |
|---:|---:|---:|---:|---:|
| 4 | 0.656 | **-0.191** | 0.558 | -0.071 |
| 3 | 0.521 | **+0.000** | 0.558 | -0.071 |
| 2 | 0.437 | **+0.119** | 0.558 | -0.071 |

**The real process has a fixed point at three starters between swaps and the
level form has none**: its drift is constant in the composition by construction,
so it walks down until the support clip binds. X1's measured drift is -0.073 per
leaver-slot (0.5611 out, 0.4882 in, 17.10) against the predicted -0.071, and its
late floor settles at 2.70 starters (0.5394, 17.3). The mechanism reading of
17.10 and 17.14 item 2 is therefore arithmetic, not interpretation.

The object for round 8 -- `P(k_out | size, state, n_starters_on_floor)` -- has
its support measured here and is fittable from the same rows. Its
pre-registration is the PM's, not this lane's.

---


## 18. Round 8 pre-registration -- the exit count as a RATE, `P(k_out | size, state, n_starters_on_floor)` (PM-directed, worker-authored 2026-09-11)

Written and committed BEFORE any round-8 object was fitted and before any arm
was run. Evidence it is built on: round 7's own results (section 17, commits
2aa29c2 / 2815cfe / 8599142), `docs/tests/rotation_exit_audit_2026-09-11.md`,
and the post-round support measurement 17.17.

### 18.1 Why round 8 adds ONE axis and nothing else

Round 7 closed the exit-side class marginal (17.10: starter share of leavers
0.5611 against a real 0.5576, K1 0.4927) and **lost 46-58 floors of per-player
minutes MAE and 18-22 pp on both close-and-late state cells**. 17.14 item 2 and
17.17 name the mechanism arithmetically and neither is an interpretation:

* the real exit rate runs **0.37 / 0.44 / 0.52 / 0.66 / 1.00** by the number of
  predicted starters on the floor (2025; 2024 agrees to 0.8-1.3 pp; 4,072-48,261
  leavers per cell, every cell powered), a **29 pp span**, against the 13.9 pp
  span of the time-and-margin term round 7 did model;
* with the round-6 entry joint (0.737 / 0.323) that rate gives a **fixed point at
  three starters on the floor** (E[change] -0.191 at four, +0.000 at three,
  +0.119 at two), while X1's level form has a constant drift of -0.071 per swap
  at every composition and therefore no fixed point at all;
* X1's measured drift is -0.073 per leaver-slot and its late floor settles at
  2.70 starters, which is the -21.0 pp close-band miss and the
  0 / -8.3 / -21.0 pp time-since-reset gradient of 17.10.

**A class count over `(size, state)` is a LEVEL; the object the sim needs is a
RATE.** Round 8 therefore keeps

* round 3b's **base fits**, round 4's **hazards**, round 5's **wave tables**,
  round 6's **K1 entry rule** and the **hard second-half reset** byte for byte,
* round 7's exit **mechanism** byte for byte -- draw `k_out` first, then round
  5's rank rule WITHIN class, then K1's `k_in` conditioned on the realised
  `k_out`, with the same uniforms in the same order,

and changes **only the conditioning set of the `k_out` table**, by one axis.
Consequence, stated so it cannot be read as a coincidence later: any difference
between a round-8 arm and X1 is a difference in **that one axis alone**.

### 18.2 The new axis, declared before it is fitted

`n_starters_on_floor` = the number of the model's own **predicted** starting five
among the five players on the floor at the substitution boundary, **before** the
swap, six levels `0..5`. It is the same quantity offline and in the sampler, it
is already computed in both loops (`is_st[on_idx].sum()`), and it is NOT a game
state: it is the model's own composition. No other axis is added, no axis is
dropped, and `prev_end` stays out of the exit cell (16.2).

### 18.3 The two candidate exit rules

**Y1 `exit_rate` -- X1 with the composition axis.**
`P(k_out | size, exit_cell, n_starters_on_floor)` fitted as counts over the same
training rows, with **one-level shrinkage to X1's own row**
`P(k_out | size, exit_cell)` at the project's `k = 300`, so a composition cell
with no signal is X1 exactly and the round is a strict extension. X1's parent
row is recomputed inside the round-8 object from the same counts and the same
fit seed, so it is X1's table by construction, not a re-fit of it. Table shape
(5, 18, 6, 6).

**Y2 `exit_rate_foul` -- Y1 with round 7's three-level foul class.**
`P(k_out | size, time_cell * margin_bucket, foul_class, n_starters_on_floor)`
with `foul_class` exactly X2's (`0` nobody on the floor at >= 4 PF, `1` somebody
but no predicted starter, `2` a predicted STARTER at >= 4 PF), shrunk **to Y1's
own row** at the matching `foul_state` (`0` for class 0, `1` for classes 1 and 2)
at `k = 300`, so a foul class with no signal is Y1 exactly. Table shape
(5, 9, 3, 6).

**Minimum cell size, declared here: `n = 300` fitted rows**, the project's
UNDERPOWERED threshold since round 5, unchanged and not tuned. Every
(size, cell, n_st) and (size, cell, foul_class, n_st) count is published in the
results table; **every cell under 300 rows is labelled UNDERPOWERED and shrinks
to its parent by construction.** Y2 is **conditional on support** in the sense
16.3 fixed for X3: if the foul-class axis is underpowered in more than half its
size-1 cells the arm is reported as unidentified and is not adopted on that
ground. Y1's support is the measured one of 17.17 (4,072-48,261 leavers per
composition cell before the 18-way state split) and is not conditional.

Both are lookup tables; neither makes a model call in the sim loop.

### 18.4 Arms

| arm | exit rule | new fitted object | simplicity | status |
|---|---|---|---:|---|
| `R2_hier_dirichlet` (S1) | -- | -- | 1 | reference (incumbent, SERVED) |
| `K1_cond_class` (S1) | rank | -- | 9 | reference (round 6's best; NOT adoptable here) |
| `X1_exit_class` (S1) | class LEVEL, then rank in class | -- | 11 | reference (round 7's best; NOT adoptable here) |
| `Y1_exit_rate` | class RATE in the composition, then rank in class | (5, 18, 6, 6) table | 14 | candidate |
| `Y2_exit_rate_foul` | Y1 with the 3-level foul class | (5, 9, 3, 6) table | 15 | candidate |

The simplicity order `R2 < K1 < X1 < Y1 < Y2` is fixed here. **K1, X1 and R2 are
references and none is adoptable in round 8** -- they are unchanged from the
rounds that produced them.

### 18.5 Scheme, folds, and what is refitted

**Scheme: S1 for every arm**, per rounds 3b-7. Windows are the calendar months of
the 2024-25 season; the first window trains on 2024 alone. No static column.

**Folds.** F1 = train 2024, test 2025, which IS the standing fold 2 (L13: CBBD
carries no on-floor data before 2023-24). 2026 stays sealed
(`seal.assert_not_sealed` guards the trainer).

**Round 8 fits the two exit objects per window and nothing else**, on the **same
rows** as rounds 6 and 7 (`--wave-team-games 6000`, fit seed 11), so the exit,
entry and wave tables cannot drift apart. `rotation_fit_v3*.json`,
`rotation_v4_sub_*.json`, `round5/rotation_v5_wave_*.json`,
`round6/rotation_v6_comp_*.json` and `round7/rotation_v7_exit_*.json` are REUSED
or recomputed-identically and **nothing is written to any of them**; the round-8
artifacts are new versioned siblings under `round8/`.

**The three reference columns (R2, K1, X1) are taken from the round-6 and
round-7 results JSONs**, not re-simulated: same 1,600-game universe, same subset
seed 2025, same sim seeds 0-2, same base fits, hazards, wave and composition
tables and grading functions. A **1-seed re-run of X1 is executed inside this
round as a reproduction check** and its cells are reported next to round 7's;
**if any state cell moves by more than its floor-A SD the reference columns are
discarded and the round is re-run in full.** Declared in advance, as 16.5
declared it for K1 and 14.4 for W4.

### 18.6 Test universe and grading path

The **same** 1,600-game subset of 2025 that rounds 2-7 used (numpy RandomState
seed 2025), **3 seeds per candidate arm** under S1, **the round-7 grading path
unchanged** (`train_rotation_v1.build_row` / `verdict` /
`rotation.aggregate_stats`, extended by `train_rotation_v4.extra_cells` /
`.minutes_mae` and `train_rotation_v5.wave_cells`, with round 6's
`quintile_mae`). No gate cell is added, so no grader line changes and the
reference columns stay comparable byte for byte. Any cell with n < 300
player-games, possessions or fitted rows is labelled UNDERPOWERED and is never
read as signal or as absence of signal.

### 18.7 Gates -- every round-7 gate, unchanged, nothing added or relaxed

*G8 cells (report, not veto):* minutes mean +/- 2.0; minutes SD ratio pooled and
within-player 0.9-1.1; top-5 and top-8 share of team minutes +/- 2 pp; players
with > 0 minutes +/- 1.0.

*The eight state cells (the veto), each +/- 3 pp:* starters' share of on-floor
slots in the final 8:00 at |m| <= 5 / 6-15 / > 15; starters' share while
carrying >= 4 fouls; the second-half TIP starter share in each of the three
margin bands; starters' share over H1 20:00-10:00 at |m| <= 5. **An arm missing
ANY of the eight is ineligible regardless of G8 or of MAE.**

*The two round-5 cells (also veto):* `sub_rate_per_boundary` +/- 0.015 or 3x the
floor-A seed SD if larger; `distinct_lineups_per_game` +/- 1.5 or 3x the floor-A
seed SD if larger. The governing number is named in the results table.

*Report only:* the "at exactly 4 fouls" diagnostic; top-1 / top-3 / top-5
five-man lineup share; K-S D of the top-1 lineup share and of per-player minutes;
mean wave size; the as-of starter benchmark of 14.6 on all eight state cells;
**the exit-side starter share of 16.7** (200 games, seed 0, the as-of predicted
starter set on both sides, one function for every row: ACTUAL 0.5576 overall,
K1 0.4927, X1 0.5611); **the time-since-reset gradient of 17.10** (the H2 tip at
0 minutes, H1 20:00-10:00 at 0-10, the final 8:00 close band at 12+), quoted at
twelve minutes as the round's headline drift number; and, added here because it
is the object the round exists to move, **the realised exit rate by
`n_starters_on_floor` in the SIM against the same rate on the ACTUAL sequences**
(17.17's table, measured through one function on both sides).

### 18.8 Primary metric and the two responsiveness checks

**Primary metric: per-player minutes MAE**, unchanged from rounds 4-7.

**Responsiveness check 1 (Decision 8), unchanged from 16.8:** team-games bucketed
into quintiles of the pregame as-of share of team minutes going to the predicted
starting five; the close-and-late cell per quintile for ACTUAL and every arm,
with slope and Q5 - Q1. An arm whose slope ratio to actual falls outside
**[0.8, 1.2]**, or whose sign disagrees, is not adoptable. (ACTUAL +0.692;
K1 0.83 PASS; X1 0.62 FAIL.)

**Responsiveness check 2, per-player minutes MAE by PLAYER quintile, carried
forward unchanged from 16.8.** Players in the as-of rotation set are bucketed
into quintiles of their own pregame as-of minutes per game. An arm is adoptable
only if it beats K1 beyond the floor in the pooled MAE **and loses beyond the
floor in no single quintile to EITHER reference it is measured against -- K1
(rounds 6-8's base) or W4 (round 5's, whose Q2 column vetoed round 6's arms)**.
Both columns are read from the round-6 results JSON, as in round 7. Carrying W4
forward is deliberate: round 8 must not become adoptable by dropping a column.
Underpowered quintiles are labelled.

### 18.9 Noise floors and the decision rule

**Floor A, seed-varied sim runs:** 20 seeds x 150 games per candidate arm, the SD
of every G8 cell, every state cell, both round-5 cells and the MAE -- the rounds
3/4/5/6/7 configuration, so six rounds' floors are comparable. R2's, K1's and
X1's floors are rounds 4's, 6's and 7's and are unchanged by a run that does not
refit them.

**Floor B, spec-identical refit under a second seed:** the round-8 exit objects
refitted from a different training-game sample (fit seed 101 vs 11) and simulated
under a different sim seed (23 vs 7), graded on the same 150-game universe, run
on the arm the decision rule selects or, failing that, on **Y1**, the arm the
round is built on. An arm counts as beating a reference on a cell only if its
improvement exceeds the refit-to-refit spread on that cell. **Floor B is
pre-registered as CONDITIONAL ON THE WALL CLOCK** -- the lane's hard stop is
12:45 ET -- and its absence, if it is absent, is reported and not hidden.

**Decision rule.** Adopt the **simplest** arm that

1. passes **every** one of the eight state cells at +/- 3 pp, AND
2. passes **both** round-5 cells at the tolerances of 18.7, AND
3. beats `R2_hier_dirichlet` on per-player minutes MAE by more than the floor, AND
4. beats `K1_cond_class` on per-player minutes MAE by more than the floor and
   loses beyond the floor to neither K1 nor W4 in any player quintile (18.8), AND
5. satisfies the Decision 8 slope check, AND
6. passes the Decision 10 freeze of 18.10.

Ties go to the simpler model in the order `R2 < K1 < X1 < Y1 < Y2`. An arm whose
improvement on the cell it was built to fix does not clear floor B is not adopted
on that cell. **If no arm is eligible, adopt nothing**, report which cell fails
and by how much, name the diagnosis, and name the next structure. No gate is
relaxed to produce a winner and no cell is dropped after seeing a result.
**The served default is not changed by this lane in any case**; the PM switches
it.

### 18.10 Decision 10: the closed loop

The round-8 exit rule carries a state term by design (the exit cell's time and
margin bands) exactly as round 7's did, so the freeze is the right instrument
and is required before any arm is served. Paired-stream runs over the fixed
**500-game subset** of the F2 2025 slate (sorted by `game_id` ascending, every
11th row, the first 500), with `ENGINE_EVENT=round2_s1`,
`ENGINE_FG_MAKE=round3_shooter_S_C_s1`, `ENGINE_CLOCK=reference` pinned and
recorded in every `run_meta.json`, reporting margin SD ratio, home/away score
correlation, possessions per game and per-player minutes MAE, live against
`ENGINE_ROTATION_FREEZE=1`, **5 seeds**, on the arm the decision rule selects.

**This requires a vectorised `next_lineup_round8` in
`engine/rotation_adapter.py` plus its parity tests and two 500-game paired runs
(~11 min), which round 7 could not fit inside its budget (17.13). It is
pre-registered here as REQUIRED BEFORE SERVING and CONDITIONAL ON THE WALL CLOCK
WITHIN THIS LANE.** If an arm passes conditions 1-5 offline and the freeze cannot
be run by the hard stop, the arm is reported as **PASSES OFFLINE, FREEZE
OUTSTANDING**, condition 6 is recorded as unmet, the arm is **not adopted**, and
the served default is left untouched. That is a reporting outcome, not a relaxed
gate.

An arm that moves margin SD ratio, home/away correlation or possessions outside
the G1/G2 tolerances between live and frozen is not adopted, and no magnitude for
the loop is quoted from the freeze alone (L31).

### 18.11 Engine expressibility (a condition on adoption)

The new axis is one integer the sampler already computes: `n_st` is
`is_st[on_idx].sum()`, a value in `0..5` gathered as a fourth index into the same
CDF table, so Y1 is one wider gather than X1 and adds no operation to the
(2N, S) roster block. Y2 adds X2's existing foul-class bincount. Both ship behind
**`ENGINE_ROTATION=round8`** plus **`ENGINE_ROTATION_ARM=Y1|Y2`**, wired in
`engine/rotation_adapter.py` with S1 artifacts per month and a manifest in the
`engine/manifest.py` format under `data/processed/models/rotation/round8/`.
`ENGINE_ROTATION=reference` remains the default and `engine/adapters.py` is **not
touched by this lane**.

### 18.12 Disclosures

1. **Y1 and Y2 draw the same uniforms as X1, in the same order** (round 5's
   coupling draw, the wave draw, the size draw, the `k_out` uniform, K1's `k_in`
   uniform, then the bench race vector), so the round-8 arms are byte-aligned
   with round 7's X1 and with each other. X1's column is nevertheless taken from
   round 7's own JSON and checked by the 1-seed reproduction re-run of 18.5.
2. `k = 300` is the project's UNDERPOWERED threshold, identical to rounds 5-7,
   fixed here before any fit and never tuned. The foul-class boundary (>= 4
   personal fouls) is round 5's own, not a new constant.
3. `n_starters_on_floor` is the model's own predicted five, not the game's; the
   as-of starter benchmark of 14.6 measures what that identification costs and
   is reported, as in every round since 4.
4. No static column is run; floor B and the Decision 10 freeze are both
   conditional on the wall clock (18.9, 18.10) and their absence is reported.
5. The support for the new axis was measured BEFORE this pre-registration, in
   17.17, on the ACTUAL sequences of both seasons, and is published there.
6. Y2 is conditional on table support (18.3) and its cell counts are published
   whether or not it is adopted.

---


## 19. Round-8 results (train 2024, test 2025) -- run 2026-09-11T16:14Z

Pre-registration section 18, committed **b6a18ec** before `rotation_v8.py`
existed and before anything was fitted. Evidence doc:
`docs/tests/rotation_exit_rate_2026-09-11.md`.

Test universe: the **same** 1,600-game subset of 2025 rounds 2-7 used (numpy
RandomState seed 2025), 3 seeds per candidate arm under S1, the round-7 grading
path unchanged. Windows and games: 202411 147, 202412 297, 202501 438, 202502
444, 202503 267, 202504 7 (UNDERPOWERED at 7 games, reported only because it
exists). Round 7's mechanism, round 6's composition tables, round 5's wave
tables, round 4's hazards and round 3b's base fits are reused and never written
to; round 8 fits only the two exit objects and their X1 parent, on 6,000
team-games per window (fit seed 11), **124,260-126,402 waves per window of which
98.2% are usable -- the same counts, window by window, that rounds 6 and 7
reported**, as 18.5 required.

**ACTUAL on this universe**, through the same functions as every arm:
substitutions per boundary **0.1509**, distinct lineups per team-game
**14.8356**, minutes mean 24.551, top-5 share 0.7472, top-8 0.9560, players with
> 0 minutes 9.6438, pooled minutes SD 9.6946, within-player 6.1196.

### 19.1 The fitted object: the composition axis is real, and the shrinkage swallows most of it

`P(k_out = 1 | size 1, n_starters_on_floor)`, averaged over the 18 exit cells,
window 202411 (the six windows agree to 0.1-1.1 pp on every level):

| starters on floor | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| **fitted Y1** | 0.5521 | **0.5202** | **0.4939** | 0.5348 | **0.6316** | 0.7553 |
| ACTUAL (17.17, 2025) | -- | **0.3697** | 0.4374 | 0.5209 | **0.6562** | 1.0000 |
| X1's level (the parent) | 0.5576 | 0.5576 | 0.5576 | 0.5576 | 0.5576 | 0.5576 |
| fitted rows, summed over the 18 cells | 379 | 2,764 | 12,646 | 28,762 | 28,011 | 9,582 |

The axis is identified where it is powered and **shrunk almost flat where it is
not**, and the arithmetic is the shrinkage rule's own: at one starter on the
floor the 2,764 rows split over 18 exit cells leave ~154 per cell, so at
`k = 300` the data carry weight 154 / (154 + 300) = 0.34 and the fitted rate is
0.34 x 0.37 + 0.66 x 0.5576 = **0.49**, which is the 0.5202 measured. At four
starters the 28,011 rows carry weight 0.84 and the fitted 0.6316 sits 2.5 pp
from the real 0.6562. **Y1 recovers the composition gradient in proportion to
each cell's own power, and the cells the round needed most are the thinnest
ones.**

**Support, published as 18.3 required.** Y1: **63 of the 108 size-1
(exit cell x n_st) cells carry fewer than 300 rows** and shrink to X1 (61-63
across the six windows). Y2: **118 of 162** size-1 cells are under 300 (117-118
across windows), i.e. **more than half the foul-class axis is unidentified**, the
condition 18.3 pre-registered for reporting the arm as unidentified. Y2 is so
reported. The composition MARGINAL is amply powered (379-28,762 rows per level,
only `n_st = 0` under 300 and that level is structurally forced to `k_out = 0`);
it is the 18-way state split that thins it, which is 19.7's diagnosis and the
next object.

### 19.2 G8 cells (report, not veto) -- S1

| cell | tol | ACTUAL | R2 | K1 | X1 | Y1 | Y2 |
|---|---|---:|---:|---:|---:|---:|---:|
| minutes mean (rotation players) | +/- 2.0 | 24.551 | 25.364 P | 25.636 P | 24.623 P | 24.947 P | 25.012 P |
| minutes SD ratio, pooled | 0.9-1.1 | 1.0000 | 1.0548 P | 0.9961 P | 0.8853 **F** | **0.9139 P** | 0.9221 P |
| minutes SD ratio, within-player | 0.9-1.1 | 1.0000 | 1.3416 **F** | 1.1239 **F** | 1.1447 **F** | 1.1380 **F** | 1.1407 **F** |
| top-5 share of team minutes | +/- 2 pp | 0.7472 | 0.7630 P | 0.7597 P | 0.7232 **F** | **0.7338 P** | 0.7361 P |
| top-8 share of team minutes | +/- 2 pp | 0.9560 | 0.9626 P | 0.9696 P | 0.9552 P | 0.9606 P | 0.9618 P |
| players with > 0 minutes | +/- 1.0 | 9.644 | 9.129 P | 8.935 P | 9.033 P | 8.995 P | 8.984 P |
| **G8 passed** | | | **5/6** | **5/6** | 3/6 | **5/6** | **5/6** |

**The two G8 cells round 7 broke are repaired by the one axis**: the pooled
minutes SD ratio goes 0.885 -> 0.914 (back inside the band) and the top-5 share
0.7232 -> 0.7338 against a real 0.7472. Minutes are no longer spread too evenly
across the roster, which is the first direct confirmation that the drift of
17.10 was the cause of both.

### 19.3 State cells (the veto) -- S1

ACTUAL, then each arm with its gap in pp and PASS/FAIL at +/- 3 pp:

| cell | ACTUAL | R2 | K1 | X1 | Y1 | Y2 |
|---|---:|---:|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7491 | 0.7254 (-2.4) P | 0.7001 (-4.9) F | 0.5394 (-21.0) F | **0.5737 (-17.5) F** | 0.5847 (-16.4) F |
| final 8:00, \|m\| 6-15 | 0.7240 | 0.6914 (-3.3) F | 0.6890 (-3.5) F | 0.5384 (-18.6) F | **0.5697 (-15.4) F** | 0.5801 (-14.4) F |
| final 8:00, \|m\| > 15 | 0.5223 | 0.4694 (-5.3) F | 0.5083 (-1.4) P | 0.5183 (-0.4) P | 0.5255 (+0.3) P | 0.5277 (+0.5) P |
| starters at >= 4 fouls | 0.4613 | 0.4626 (+0.1) P | 0.4212 (-4.0) F | 0.3823 (-7.9) F | 0.3925 (-6.9) F | 0.3930 (-6.8) F |
| H2 TIP, \|m\| <= 5 | 0.9678 | 0.7892 (-17.9) F | 0.9385 (-2.9) P | 0.9406 (-2.7) P | 0.9407 (-2.7) P | 0.9406 (-2.7) P |
| H2 TIP, \|m\| 6-15 | 0.9611 | 0.7977 (-16.3) F | 0.9410 (-2.0) P | 0.9424 (-1.9) P | 0.9414 (-2.0) P | 0.9410 (-2.0) P |
| H2 TIP, \|m\| > 15 | 0.9563 | 0.7338 (-22.3) F | 0.9456 (-1.1) P | 0.9449 (-1.1) P | 0.9453 (-1.1) P | 0.9443 (-1.2) P |
| H1 20:00-10:00, \|m\| <= 5 | 0.7822 | 0.5988 (-18.3) F | 0.7267 (-5.6) F | 0.6991 (-8.3) F | 0.7105 (-7.2) F | 0.7121 (-7.0) F |
| **state cells passed** | | **2/8** | **4/8** | **4/8** | **4/8** | **4/8** |

R2's, K1's and X1's columns are rounds 6's and 7's, per 18.5; the reproduction
check is 19.6. **Every arm passes the same four cells, and Y1 recovers 3.5 of
X1's 21.0 pp on the close band and 3.2 of 18.6 on the middle band** -- 2.0 and
2.4 floor-A SDs, real movement in the right direction and **one sixth of the
distance to K1, which is itself 4.9 pp short**.

### 19.4 The two round-5 cells, the primary metric, and concentration -- S1

Tolerances, computed as pre-registered: `sub_rate_per_boundary` **+/- 0.015**
governs (3x floor = 0.0053); `distinct_lineups_per_game` **+/- 1.5** governs
(3x floor = 0.572).

| metric | ACTUAL | R2 | K1 | X1 | Y1 | Y2 |
|---|---:|---:|---:|---:|---:|---:|
| **substitutions per boundary** | 0.1509 | 0.1437 P | 0.1554 P | 0.1557 P | 0.1559 P | 0.1558 P |
| **distinct lineups per team-game** | 14.836 | 15.514 P | **14.571 P** | 15.939 P | 15.899 P | 15.876 P |
| **per-player minutes MAE (min)** | 0.0 | 9.7939 | **8.8622** | 9.6159 | **9.3851** | **9.3564** |
| MAE gain over R2 (floors) | -- | -- | +63 | +11.0 | **+27.8** | +29.7 |
| MAE gain over X1 (floors) | -- | -- | +46.5 | -- | **+14.2** | +16.0 |
| MAE gain over K1 (floors) | -- | -- | -- | -46.5 | **-39.3** | -37.1 |
| top-1 lineup share | 0.2940 | 0.2286 | 0.2568 | 0.2345 | 0.2341 | 0.2342 |
| top-3 lineup share | 0.5426 | 0.4819 | 0.5358 | 0.4995 | 0.4996 | 0.4998 |
| top-5 lineup share | 0.6894 | 0.6409 | 0.7043 | 0.6663 | 0.6672 | 0.6675 |
| K-S D, per-player minutes | -- | 0.0798 | 0.0589 | 0.0441 | **0.0413** | 0.0414 |
| K-S D, top-1 lineup share | -- | 0.2273 | **0.0990** | 0.1993 | 0.1972 | 0.1989 |

**The one axis is worth +0.231 minutes of per-player MAE over X1, 14 floors, and
the arm is still 0.523 minutes -- 39 floors -- behind K1.** Round 7's signature
survives in weaker form: the best per-player minutes distribution ever measured
(K-S D 0.0413) alongside an assignment that is worse than a rank rule's.

### 19.5 Noise floor A (20 seeds x 150 games, S1)

| metric | Y1 | Y2 | (X1, round 7) |
|---|---:|---:|---:|
| minutes_mae | 0.01331 | 0.01348 | 0.01621 |
| sub_rate_per_boundary | 0.00178 | 0.00170 | 0.00202 |
| distinct_lineups_per_game | 0.1906 | 0.1699 | 0.2396 |
| late_starter_share_b0 | 0.01734 | 0.01679 | 0.01946 |
| late_starter_share_b1 | 0.01307 | 0.01177 | 0.01353 |
| late_starter_share_b2 | 0.01948 | 0.01767 | 0.02197 |
| foul_trouble_share | 0.02342 | 0.02436 | 0.02003 |
| h2tip_starter_share_b0 | 0.00976 | 0.01018 | 0.00793 |
| h2tip_starter_share_b1 | 0.00806 | 0.00814 | 0.00882 |
| h2tip_starter_share_b2 | 0.01545 | 0.01676 | 0.01570 |
| opentip_starter_share_close | 0.01199 | 0.01164 | 0.01460 |
| top5_share | 0.00293 | 0.00311 | 0.00263 |
| n_nonzero_mean | 0.06412 | 0.06654 | 0.06531 |

R2's, K1's and X1's floors are rounds 4's, 6's and 7's and are unchanged by a run
that does not refit them (R2 minutes_mae 0.01473, K1 0.01348, X1 0.01621).
**Every miss in 19.3 is far larger than its floor**: Y1's close-band miss is
-17.5 pp against a 1.73 pp floor (10.1 floors), the middle band -15.4 pp against
1.31 pp (11.8 floors) and the foul-trouble cell -6.9 pp against 2.34 pp (2.9
floors). The floor-A caveat of 11.6 applies unchanged: only the seed-to-seed SD
is a floor, never the `minutes_mae` LEVEL of that 150-game run.

### 19.6 The X1 reproduction check (18.5)

A 1-seed re-run of X1 inside this round, against round 7's own 3-seed column:

| cell | round 7 (3 seeds) | round 8 (1 seed) | delta | floor A |
|---|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.5394 | 0.5404 | +0.0010 | 0.0195 |
| final 8:00, \|m\| 6-15 | 0.5384 | 0.5346 | -0.0038 | 0.0135 |
| final 8:00, \|m\| > 15 | 0.5183 | 0.5263 | +0.0079 | 0.0220 |
| starters at >= 4 fouls | 0.3823 | 0.3792 | -0.0031 | 0.0200 |
| H2 tip, \|m\| <= 5 | 0.9406 | 0.9389 | -0.0017 | 0.0079 |
| H2 tip, \|m\| 6-15 | 0.9424 | 0.9359 | -0.0065 | 0.0088 |
| H2 tip, \|m\| > 15 | 0.9449 | 0.9459 | +0.0010 | 0.0157 |
| H1 20:00-10:00, \|m\| <= 5 | 0.6991 | 0.7002 | +0.0011 | 0.0146 |
| substitutions per boundary | 0.15573 | 0.15543 | -0.00030 | 0.00202 |
| distinct lineups | 15.939 | 15.887 | -0.0525 | 0.2396 |

**Every state cell moves less than its floor-A SD, so the reference columns
stand** (the largest move, the tip at \|m\| 6-15, is -0.65 pp against a 0.88 pp
floor). The MAE moves +0.035 on one seed against a 0.016 floor, the expected
1-vs-3-seed Monte-Carlo difference and the reading rounds 5, 6 and 7 gave.

### 19.7 The object the round exists to move: the exit rate BY COMPOSITION

`scripts/diag_rotation_exit_v8.py`, 200 games, seed 0, the as-of predicted
starter set on BOTH sides, one function for every row.
`P(a starter is the man who leaves | single swap, starters on the floor)`:

| starters on floor | ACTUAL | n | X1 | n | **Y1** | n | Y2 | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3115 | 244 UP | 0.5774 | 594 | **0.4612** | 451 | 0.4018 | 443 |
| 2 | 0.4369 | 982 | 0.5752 | 1,210 | **0.4447** | 1,158 | 0.4294 | 1,148 |
| 3 | 0.5115 | 1,775 | 0.5443 | 1,547 | **0.4844** | 1,734 | 0.4793 | 1,763 |
| 4 | 0.6761 | 1,689 | 0.5892 | 1,463 | **0.6286** | 1,578 | 0.6378 | 1,604 |
| 5 | 1.0000 | 544 | 1.0000 | 592 | 1.0000 | 597 | 1.0000 | 590 |
| **span, 1 -> 4** | **+36.5 pp** | | **+1.2 pp** | | **+16.7 pp** | | **+23.6 pp** | |

(UP = UNDERPOWERED at n < 300 and labelled, never read as signal.)

**X1's rate is flat in the composition -- 57.7 / 57.5 / 54.4 / 58.9 -- which is
the direct measurement of 17.14 item 2's "a level, not a rate", and it is the
first time that claim has been measured on the SIM side rather than inferred
from the arithmetic.** Y1's rate slopes the right way at every step and carries
**46% of the real span**; Y2 carries 65%, because its foul classes split the
thin low-composition cells differently. The residual is 19.1's attenuation and
nothing else: the axis that the data support (four starters, 28,011 rows) lands
2.5 pp from the truth, and the axis that the data do not (one starter, 154 rows
per cell) lands 15 pp away.

The simulated floor composition moves with it. Single swaps taken with 1 or 2
starters on the floor: **ACTUAL 1,226, X1 1,804, Y1 1,609, Y2 1,591** -- the
bench-heavy tail X1 walked into is a third smaller under Y1, and still 31% too
big.

**The exit-side starter share (16.7), so round 8 drops into 17.10's table:**

| | leavers, overall | size 1 | size 2 | size 3+ | entrants | size-1 joint (bench out / starter out) | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ACTUAL (as-of starters)** | **0.5576** | **0.5867** | **0.5576** | **0.4931** | **0.4972** | 0.6848 / 0.3446 | 34.0 pp |
| K1 (round 6) | 0.4927 | 0.4564 | 0.5142 | 0.5496 | 0.4334 | 0.6344 / 0.2026 | 43.2 pp |
| X1 (round 7) | 0.5611 | 0.6013 | 0.5648 | 0.4616 | 0.4882 | 0.6762 / 0.3073 | 36.9 pp |
| **Y1** | 0.5422 | 0.5636 | 0.5368 | 0.5000 | 0.4769 | 0.6891 / 0.2798 | 40.9 pp |
| Y2 | 0.5370 | 0.5576 | 0.5275 | 0.5042 | 0.4728 | 0.6887 / 0.2697 | 41.9 pp |

**The marginal X1 hit exactly is now 1.5 pp low** (Y1 0.5422 against 0.5576), and
that is the expected and correct consequence of making the rate conditional:
a marginal is only reproduced when the conditional rate AND the composition
distribution are both right, and Y1's floor still spends too long bench-heavy
(above). Trading 1.5 pp of a report-only marginal for 3.5 pp of a vetoed state
cell is the direction the pre-registration asked for.

**The time-since-reset gradient (17.10), the round's headline drift number:**

| cell | minutes since the last forced reset | K1 gap | X1 gap | **Y1 gap** |
|---|---:|---:|---:|---:|
| H2 tip, all three margin bands | 0 | -1.1 to -2.9 pp | -1.1 to -2.7 pp | -1.1 to -2.7 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0-10 | -5.6 pp | -8.3 pp | **-7.2 pp** |
| final 8:00, \|m\| <= 5 | 12+ | -4.9 pp | **-21.0 pp** | **-17.5 pp** |

**The gradient is flattened by 17% and not removed.** Y1 is still correct at the
reset and still drifts monotonically away from it, at 0.83 of X1's rate. One
composition axis, attenuated to 46% of its measured strength by a shrinkage
parent that is the level form itself, buys 17% of the drift: the proportions
agree, and 19.10 names what to do about it.

### 19.8 The two responsiveness conditions

**Decision 8 (18.8 check 1).** Cell = starters' share in the final 8:00 at
\|margin\| <= 5, by quintile of the pregame as-of predicted starter-minutes share
(640 team-games per quintile).

| quintile | prior | ACTUAL | K1 | X1 | Y1 | Y2 |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 0.5572 | 0.6770 | 0.6218 | 0.4921 | 0.5261 | 0.5357 |
| Q2 | 0.6342 | 0.7226 | 0.6940 | 0.5280 | 0.5607 | 0.5719 |
| Q3 | 0.6698 | 0.7471 | 0.7089 | 0.5372 | 0.5754 | 0.5910 |
| Q4 | 0.7044 | 0.7763 | 0.7299 | 0.5563 | 0.5923 | 0.6062 |
| Q5 | 0.7685 | 0.8219 | 0.7430 | 0.5826 | 0.6128 | 0.6172 |
| **slope** | | **+0.692** | +0.576 | +0.426 | **+0.416** | +0.399 |
| slope ratio to actual | | 1.00 | **0.83 PASS** | **0.62 FAIL** | **0.60 FAIL** | **0.58 FAIL** |
| Q5 - Q1 (pp) | | +14.5 | +12.1 | +9.1 | +8.7 | +8.2 |

Every arm is monotone in 4 of 4 steps. **Y1 lifts every quintile by 3.0-3.6 pp
and leaves the slope where X1 left it** (0.426 -> 0.416, inside the seed noise of
the cell). This is the round's sharpest negative result and it is not a surprise
once 19.1 is read: the composition axis is a WITHIN-GAME restoring force and
carries no information about which TEAM is playing, so it moves the level of
every quintile together and cannot move the response to the team's own rotation
depth. Whatever fixes Decision 8 is a different object from this one.

**The per-player-quintile condition (18.8 check 2).** Per-player minutes MAE by
quintile of the player's own pregame as-of minutes per game (4,782-4,784
player-games per quintile, none underpowered):

| quintile | K1 (r6) | W4 (r6) | X1 (r7) | **Y1** | Y2 | Y1 - K1 | Y1 - W4 | Y1 - X1 | floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 (lowest mpg) | 9.3108 | 9.6681 | 10.6053 | 10.2595 | 10.1673 | **+0.949** | +0.591 | **-0.346** | 0.013 |
| Q2 | 9.7891 | 9.7294 | 10.4615 | 10.2980 | 10.2687 | **+0.509** | +0.569 | **-0.164** | 0.013 |
| Q3 | 9.2575 | 9.4074 | 9.7867 | 9.6199 | 9.6141 | +0.362 | +0.213 | -0.167 | 0.013 |
| Q4 | 8.6461 | 8.6621 | 9.1081 | 8.9462 | 8.9625 | +0.300 | +0.284 | -0.162 | 0.013 |
| Q5 (highest mpg) | 7.2367 | 7.3307 | 8.0130 | 7.7025 | 7.6742 | +0.466 | +0.372 | **-0.311** | 0.013 |

**Y1 beats X1 in all five quintiles by 12-26 floors and loses all five to both
references by 16-71 floors.** The composition axis pays everywhere and does not
pay enough anywhere.

### 19.9 Per-game and per-team evidence

**Per team** (starters' share in the final 8:00 at \|m\| <= 5, aggregated per
team; a team is powered when both denominators reach 300 on-floor slots --
**251 powered, 112 UNDERPOWERED and excluded**):

| | sim mean | sim SD | actual mean | actual SD | corr(sim, actual) | mean \|dev\| |
|---|---:|---:|---:|---:|---:|---:|
| K1 (round 6) | 0.7004 | 0.0507 | 0.7512 | 0.0832 | **0.394** | 0.0757 |
| X1 (round 7) | 0.5412 | 0.0599 | 0.7512 | 0.0832 | 0.209 | 0.2108 |
| **Y1** | 0.5726 | 0.0548 | 0.7512 | 0.0832 | **0.211** | **0.1793** |
| Y2 | 0.5840 | 0.0555 | 0.7512 | 0.0832 | 0.212 | 0.1689 |

The mean absolute deviation falls 0.211 -> 0.179 and **the correlation with the
team's own actual share does not move at all** (0.209 -> 0.211 against K1's
0.394). Per team, exactly as per quintile in 19.8: the level improves and the
matchup responsiveness does not. The cross-team SD falls slightly (0.060 ->
0.055 against a real 0.083), so the extra dispersion round 7 had was the drift
varying game to game, as 17.9 said, and removing part of the drift removed part
of the dispersion without adding any information.

**Per-player minutes seed SD** (3 seeds, the same universe): Y1 0.0395, Y2
0.0255, X1 0.0379, K1 0.0737.

### 19.10 The as-of starter benchmark (report only, 18.7)

The ACTUAL sequence of the same 1,600 games, re-graded with the MODEL's as-of
predicted starting five instead of the game's own; the model's five overlaps the
real five on **4.576 of 5**, unchanged from rounds 6 and 7. Starter
identification costs -2.8 pp on the close-and-late band (0.7215 against 0.7491),
-2.8 pp on the middle band, -4.4 pp on the opening ten minutes and **+0.2 pp on
foul trouble**. Against that benchmark Y1 is **-14.8 pp** on the close band
(X1 -18.2, K1 -2.1) and -2.8 pp on the opening cell. **It changes no tolerance
and no verdict**: every arm is scored against each side's own real starting five,
as in rounds 1-7.

### 19.11 Decision

| arm | simplicity | state | round-5 cells | G8 | MAE | vs R2 | vs K1 | vs X1 | quintiles | D8 | Dec-10 | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| R2_hier_dirichlet | 1 | 2/8 | 2/2 | 5/6 | 9.7939 | -- | -0.932 | -0.178 | ok | n/a | -- | NO (reference) |
| K1_cond_class | 9 | 4/8 | 2/2 | 5/6 | **8.8622** | +0.932 | -- | +0.754 | see 15.8 | 0.83 | PASS (15.13) | NO (reference) |
| X1_exit_class | 11 | 4/8 | 2/2 | 3/6 | 9.6159 | +0.178 | -0.754 | -- | 5/5 fail | 0.62 F | NOT RUN | NO (reference) |
| Y1_exit_rate | 14 | 4/8 | 2/2 | **5/6** | **9.3851** | +0.409 | **-0.523** | **+0.231** | **5/5 fail** | **0.60 F** | NOT RUN | NO |
| Y2_exit_rate_foul | 15 | 4/8 | 2/2 | 5/6 | 9.3564 | +0.437 | -0.494 | +0.259 | **5/5 fail** | **0.58 F** | NOT RUN | NO |

**No arm adopted.** Y1 fails condition 1 (four of eight state cells, two of them
by 15-18 pp), condition 4 (loses 0.523 minutes of MAE to K1 and every player
quintile to both K1 and W4) and condition 5 (Decision 8 slope ratio 0.60 against
a [0.8, 1.2] band); condition 6 was not run. It passes conditions 2 and 3. Y2
fails the same three and is additionally **reported as UNIDENTIFIED** under 18.3,
118 of its 162 size-1 cells being under 300 rows. Per 18.9 the rule adopts
nothing; no gate was relaxed and no cell was dropped after seeing a result.
`ENGINE_ROTATION=reference` (R2) stays the served default and **this lane changed
no default**; the `round8` flag ships nowhere, because 18.11's adapter was not
wired (19.12).

Cell-by-cell misses of the closest arm:

| arm | cell | sim | actual | miss | floors | of which starter identification |
|---|---|---:|---:|---:|---:|---:|
| Y1 | final 8:00 \|m\| <= 5 | 0.5737 | 0.7491 | -17.5 pp | 10.1 | -2.8 pp |
| Y1 | final 8:00 \|m\| 6-15 | 0.5697 | 0.7240 | -15.4 pp | 11.8 | -2.8 pp |
| Y1 | starters at >= 4 fouls | 0.3925 | 0.4613 | -6.9 pp | 2.9 | +0.2 pp (none) |
| Y1 | H1 20:00-10:00 \|m\| <= 5 | 0.7105 | 0.7822 | -7.2 pp | 5.9 | -4.4 pp |
| Y1 | player quintile Q1 | 10.2595 | (K1 9.3108) | +0.949 | 71 | -- |
| Y1 | D8 slope ratio | 0.601 | (band 0.8-1.2) | -0.199 | -- | -- |

### 19.12 Floor B and Decision 10: NOT RUN, and what that costs

18.9 and 18.10 pre-registered both as **conditional on the wall clock**, with the
lane's hard stop at 12:45 ET. **Neither was run.** Floor B (a second exit refit
under fit seed 101 plus two 150-game grades, about 6 minutes by round 7's
measurement) and the Decision 10 freeze (a vectorised `next_lineup_round8` in
`engine/rotation_adapter.py`, its parity tests and two 500-game paired runs,
about 11 minutes plus the adapter) did not fit inside the lane after the
bake-off finished at 12:14 ET. **The absence is reported, not hidden.**

What it costs is stated precisely. Condition 6 of the decision rule is unmet for
every arm, so **no arm could have been adopted today even on a clean sheet of
offline cells**; it costs nothing on this round's actual outcome, because Y1
already fails conditions 1, 4 and 5 by 3 to 71 floors. The identification claim
of 19.1 rests on the **six-window agreement** of the fitted rate (0.1-1.1 pp per
level across six independent training windows, 19.1) and on round 7's own floor-B
result for the same object family under the same shrinkage constant (17.6: table
cells moved 0.1-0.7 pp under a fit-seed change), **not** on a round-8 floor-B
measurement, and it is labelled as such. If a later round revives this family the
freeze must be run before adoption and its cost must be inside that round's
budget from the start -- the second round running in which it has not been.

### 19.13 Diagnosis

**1. The mechanism reading of round 7 is confirmed on the sim side, directly.**
17.14 item 2 inferred "a level, not a rate" from arithmetic; 19.7 measures X1's
simulated exit rate at 0.577 / 0.575 / 0.544 / 0.589 by composition, flat to
within 4.5 pp across a real 36.5 pp span. The diagnosis was right.

**2. The fix works in the direction predicted, at the strength its support
allows, and that is not enough.** One axis repairs both broken G8 cells (19.2),
buys 14 floors of MAE (19.4), closes 3.5 pp of a 21.0 pp state-cell miss and 17%
of the drift gradient (19.7), improves every player quintile over X1 by 12-26
floors (19.8) and cuts the per-team mean deviation by 15% (19.9). It leaves the
arm 4/8 on the veto, 39 floors behind K1 and outside the Decision 8 band.

**3. The binding constraint is now the SHRINKAGE PARENT, and it is measurable,
not a judgement.** Y1 shrinks to X1's row -- the level form -- so at one starter
on the floor the fitted rate is 0.34 x (data 0.37) + 0.66 x (level 0.5576) =
0.49 against a real 0.37 (19.1), and the simulated rate lands at 0.46 against
0.31 (19.7). The composition MARGINAL `P(k_out | size, n_st)` is powered at
2,764-28,762 rows per level with a 29 pp span and no state split at all.
**The next object is therefore the same two axes with the hierarchy the other way
up: fit `P(k_out | size, n_st)` first and shrink the (size, state, n_st) cell to
THAT, instead of to `P(k_out | size, state)`.** A thin cell then falls back on
the composition rate, which is the strong term, rather than on the level, which
is the term that has no fixed point. It costs nothing new to fit -- the counts
are already in round 8's own artifacts -- and it is one line of `fit_exit8`.

**4. Decision 8 is a different problem and round 8 proves it.** The composition
axis lifted all five quintiles by 3.0-3.6 pp and moved the slope by -0.010
(19.8); the per-team correlation moved by +0.002 (19.9). A within-game restoring
force cannot carry between-team information by construction. Every arm since
round 5 that failed Decision 8 failed it for a reason the exit rule cannot
address, and the object that can is one whose table is indexed by something the
TEAM brings -- its own as-of starter-minutes share -- which no rotation arm in
eight rounds has carried.

**5. Foul trouble: the ordering of 17.14 item 5 still holds.** The graded
foul-trouble cell improves 0.3823 -> 0.3925 with the drift, is still -6.9 pp, and
Y2's three-level foul class adds +0.05 pp on it -- nothing. The cell will not be
readable until the drift is fixed, which is item 3.

**6. Y2 adds little and is unidentified.** Y2 beats Y1 by 0.029 minutes of MAE
(2.1 floors) and by 1.1 pp on the close band, and 118 of its 162 size-1 cells
carry under 300 rows, which 18.3 pre-registered as the condition for reporting it
unidentified. The simpler arm is the family, the same tie-break round 7 recorded
for X2 and X3.

### 19.14 Artifacts and flags

- Exit objects: `data/processed/models/rotation/round8/rotation_v8_exit_{YYYYMM}.json`
  (six windows) with `rotation_v8_manifest.json` in the `engine/manifest.py`
  format; each entry carries `refit_date`, `max_train_date`, the reused hazard,
  wave and composition artifact names, the published `P(k_out=1|size 1)` row by
  composition and by exit cell, the per-level row counts and the
  under-300 cell counts for Y1 and Y2. The directory is gitignored
  (`data/processed/models/*/round*/`) and HF-synced.
- Nothing was written to `rotation_fit.json`, `rotation_fit_v3*.json`, any
  `rotation_v4_sub_*.json`, any `round5/rotation_v5_wave_*.json`, any
  `round6/rotation_v6_comp_*.json` or any `round7/rotation_v7_exit_*.json`.
- Engine: **not wired** (19.12). `ENGINE_ROTATION=round8` does not exist,
  `engine/adapters.py` and `engine/loop.py` were not touched by this lane, and
  `ENGINE_ROTATION=reference` remains the default.
- Tests: `tests/test_rotation_v8.py` (8 cases: table shapes, row normalisation,
  support clipping to the wave size at every composition, Y1's exact shrinkage to
  X1 on an empty cell, Y2's exact shrinkage to Y1 at the matching foul state,
  the X1 parent reproducing `rotation_v7.fit_exit` on the same counts,
  identification on the composition axis with the level form shown to be the
  mixture, and the declared constants and arm grid) -> **8 passed**.
- Results: `rotation_F1_round8_{results.json,table.csv}`,
  `exit_audit_round8_2026-09-11.json`.
- Cost: the six exit fits 28-46 s each, run concurrently (6 workers); the five
  grade/floor jobs 99-232 s each; the whole bake-off **7.8 min**; the exit
  diagnostic 5.3 min, run concurrently with the bake-off's grade stage on the
  sixth worker.

### 19.15 Disclosures

1. **No smoke run was executed.** The code path was verified by
   `tests/test_rotation_v8.py` (8 cases, offline, on synthetic counts) and by the
   fit stage of the graded run itself; no model specification, gate, tolerance or
   arm was changed after any number was seen, and section 18 was committed at
   b6a18ec before `rotation_v8.py` existed.
2. The reference columns for R2 and K1 are taken from round 6's results JSON and
   X1's from round 7's (18.5), with the 1-seed X1 reproduction check of 19.6.
3. No static column was run (18.5).
4. **Floor B and Decision 10 were both NOT RUN** (19.12), on the wall clock. No
   closed-loop claim and no refit-to-refit claim is made for any round-8 arm; the
   identification evidence offered is the six-window agreement and round 7's
   floor B on the same family, both labelled.
5. Y1 and Y2 draw the same uniforms as X1 in the same order and are byte-aligned
   with it (18.12 item 1); X1's column is nevertheless taken from round 7's own
   JSON and checked by 19.6.
6. Y1's own support is published in 19.1: 63 of 108 size-1 (exit cell x n_st)
   cells carry under 300 rows and shrink to X1. 18.3 made only Y2 conditional on
   support, so Y1 is not reported as unidentified; its attenuation is reported as
   the round's binding constraint instead (19.13 item 3).
7. The exit-side and by-composition diagnostics are measured on 200 games at
   seed 0, the configuration of the round-6 and round-7 audits, and are
   report-only. The ACTUAL row at one starter on the floor carries 244 leavers
   and is labelled UNDERPOWERED; the 2024/2025 full-season measurement of the
   same quantity (17.17, 4,072-5,981 leavers) is 0.369/0.370 and is the powered
   version of that cell.
8. No process this worker did not start was signalled, and no existing data or
   results file was overwritten: every round-8 artifact is a new versioned
   sibling under `round8/` or a new `_round8_` stem.

---

### 19.16 Floor B -- run AFTER 19.12 was written and committed

**19.12 stands as written**: it was true when it was written, at 12:19 ET, and
this file is append-only. After committing section 19 the lane had 23 minutes of
wall clock left, which was enough for floor B (18.9) and not for the Decision 10
freeze (18.10, which needs an engine adapter this lane did not write). Floor B
was therefore run and this subsection supersedes 19.12 **on floor B only**;
**Decision 10 remains NOT RUN and condition 6 remains unmet for every arm.**

Floor B on `Y1_exit_rate`: a different training-game sample (fit seed 101 against
11) and a different sim seed (23 against 7), graded on the same 150-game
universe, run in its own process after the bake-off JSON was on disk, 2.1 min.

| cell | ACTUAL (150 games) | Y1 seed 1 | Y1 seed 2 | \|delta\| |
|---|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7436 | 0.5672 | 0.5889 | 2.17 pp |
| final 8:00, \|m\| 6-15 | 0.7186 | 0.5646 | 0.5539 | 1.08 pp |
| final 8:00, \|m\| > 15 | 0.5518 | 0.5299 | 0.4822 | 4.77 pp |
| starters at >= 4 fouls | 0.4362 | 0.4271 | 0.4228 | 0.43 pp |
| H2 tip, \|m\| <= 5 | 0.9655 | 0.9414 | 0.9621 | 2.07 pp |
| H2 tip, \|m\| 6-15 | 0.9662 | 0.9324 | 0.9432 | 1.08 pp |
| H2 tip, \|m\| > 15 | 0.9611 | 0.9556 | 0.9500 | 0.56 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0.7710 | 0.7058 | 0.7101 | 0.44 pp |
| substitutions per boundary | -- | 0.1500 | 0.1570 | 0.0071 |
| distinct lineups per team-game | -- | 15.233 | 16.067 | 0.833 |
| per-player minutes MAE | -- | 23.0877 | 23.0644 | 0.0232 |

**The round-8 objects are identified.** The fitted table moves **0.15-0.17 pp per
composition level** under the refit (window 202411,
`P(k_out = 1 | size 1, n_st)`: 0.5202 -> 0.5217 at one starter on the floor,
0.6316 -> 0.6299 at four; the six seed-101 windows agree with each other to
0.1-1.1 pp, as the seed-11 windows do) and the under-300 cell count moves
63 -> 65 of 108. The cell spread (0.43-4.77 pp) is round 7's (0.29-6.31 pp) on
the same thin universe.

**The decision does not turn on a refit artefact in either direction**: the
close-band miss Y1 fails on is -17.5 pp, **eight times** the largest
refit-to-refit move on that cell (2.17 pp) and 10 floor-A SDs, and the axis
attenuation 19.13 item 3 names is 15 pp at one starter against a 0.15 pp
refit move. `rotation_F1_round8_floorB_Y1.json`'s `minutes_mae` level (23.1) is
the 150-game-universe artifact rounds 5-7 carry for the same reason (11.6): only
the seed-to-seed and refit-to-refit DIFFERENCES are readable, never the level.

Artifacts: `round8/rotation_v8_exit_seed2_{YYYYMM}.json` +
`rotation_v8_manifest_seed2.json` (new versioned siblings; nothing was
overwritten), `rotation_F1_round8_floorB_Y1.json`.

**Amendment to 19.15 item 4:** floor B was run, as recorded here; Decision 10 was
not. The identification claim of 19.1 now rests on a measured refit-to-refit
spread as well as on the six-window agreement.

---

## 20. Round 9 pre-registration -- inverting the exit hierarchy onto the POWERED composition marginal (PM-directed, worker-authored 2026-09-11)

Written and committed BEFORE `rotation_v9.py` existed and before anything was
fitted. Evidence it is built on: round 8's own results (section 19, commits
b6a18ec..8e378b5), `docs/tests/rotation_exit_rate_2026-09-11.md`, and the
support measurement 19.1.

### 20.1 The one thing round 9 changes, and why

19.13 item 3 named it arithmetically and it is not an interpretation. Y1 fits
`P(k_out | size, exit_cell, n_st)` and shrinks it **to X1's row**, the LEVEL
`P(k_out | size, exit_cell)`. At one predicted starter on the floor the 2,764
training rows split over the 18 exit cells leave ~154 per cell, so at `k = 300`
the data carry weight 154 / (154 + 300) = 0.34 and the fitted rate is
0.34 x 0.37 + 0.66 x 0.5576 = **0.49 against a real 0.37** (19.1); the simulated
rate lands at 0.46 against 0.31 and the arm carries **46% of the real 36.5 pp
span** (19.7). Y1's gains are real -- both broken G8 cells repaired, +14 floors
of MAE over X1, 3.5 pp of a 21.0 pp close-band miss, the drift gradient
flattened 17% -- and they stop where the shrinkage parent stops.

The composition MARGINAL `P(k_out | size, n_st)` is **amply powered**: 379 /
2,764 / 12,646 / 28,762 / 28,011 / 9,582 rows at `n_st = 0..5` with a 29 pp span
(19.1), because it is pooled over the 18 exit cells and carries no state split at
all. **Round 9 therefore turns the hierarchy the other way up.** A thin
(state x composition) cell falls back on the composition rate -- the strong term,
which has a fixed point at three starters -- instead of on the level, which has
no fixed point and drifts -0.071 per swap (18.1).

Round 9 keeps round 3b's base fits, round 4's hazards, round 5's wave tables and
rank-within-class rule, round 6's K1 entry rule, the hard second-half reset and
rounds 7-8's exit MECHANISM byte for byte -- draw `k_out` first, then the rank
rule WITHIN class, then K1's `k_in` conditioned on the realised `k_out`, with the
same uniforms in the same order. **It changes only the shrinkage parent of the
`k_out` table** (and, in Z2, whether the interaction is used at all). Any
difference between a round-9 arm and Y1 is a difference in **that alone**.

### 20.2 The three-level hierarchy, declared before it is fitted

Per wave size `s`:

* **level 0**, `M0(s) = P(k_out | size)` -- the size row pooled over everything,
  the same object round 7's `fit_exit` and round 8's `fit_exit8` use as the root;
* **level 1**, `M1(s, n_st) = P(k_out | size, n_st)` -- **the composition
  marginal, pooled over all 18 exit cells**, counts shrunk to `M0(s)` at `k`;
* **level 2**, `Z(s, ce, n_st) = P(k_out | size, exit_cell, n_st)` -- the state
  cell's own counts shrunk to `M1(s, n_st)` at `k`.

Y1's hierarchy was `M0 -> P(k_out | size, ce) -> level 2`. Round 9's is
`M0 -> M1 -> level 2`. Same two axes, same counts, same rows, same mechanism;
the parent of the thin cell is the powered composition rate instead of the level.
No axis is added and none is dropped; `prev_end` stays out of the exit cell
(16.2). `n_st` is the quantity declared in 18.2, unchanged: the number of the
model's own PREDICTED starting five among the five on the floor BEFORE the swap,
`is_st[on_idx].sum()`, six levels `0..5`, identical offline and in the sampler.

### 20.3 The two candidate exit rules

**Z1 `exit_marg` -- the inverted hierarchy, continuous shrinkage.**
`Z(s, ce, n_st)` as 20.2 defines it, at the fitted `k` of 20.4. A state cell with
no signal is the composition marginal `M1(s, n_st)` exactly; a composition level
with no signal is `M0(s)` exactly. Table shape (5, 18, 6, 6).

**Z2 `exit_interact` -- the same, with the interaction used only where it is
powered.** `Z2(s, ce, n_st) = Z(s, ce, n_st)` where the cell carries
**>= 300 fitted rows**, and `= M1(s, n_st)` exactly otherwise. This is a hard
gate on the (state x composition) interaction in place of a continuous one, so
the thin cells contribute no partial state signal at all. Table shape
(5, 18, 6, 6).

**Minimum cell size, unchanged: `n = 300` fitted rows**, the project's
UNDERPOWERED threshold since round 5, not tuned here and identical to the
threshold Z2's gate uses -- they are the same constant, stated once. Every
(size, cell, n_st) count and every (size, n_st) marginal count is published in
the results table; every cell under 300 rows is labelled UNDERPOWERED.

Both are lookup tables; neither makes a model call in the sim loop. Support is
clipped to `0..size` and renormalised at every level, as in rounds 7 and 8.

### 20.4 The shrinkage constant is FITTED, not chosen by hand -- the method, declared here

Rounds 7 and 8 used the project's `k = 300` as a declared constant. Round 9's
result depends on how hard a thin cell is pulled onto its new parent, so `k` is
selected by a stated criterion on held-out TRAINING data and then frozen:

* **Data: the 2024 training season only** (the first S1 window's own training
  set), which precedes the entire F1 2025 test season, so no test row and no
  gate cell enters the choice.
* **Procedure: leave-one-fold-out over 5 disjoint folds of the 2024 training
  team-games** (`numpy.RandomState(11)` assigns folds). For each fold, the
  level-1 and level-2 tables are fitted from the other four folds' counts and
  scored on the left-out fold's counts.
* **Criterion: held-out multinomial log-likelihood per held-out row**, summed
  over every (size, exit_cell, n_st) cell, of the level-2 table Z1 defines.
* **Grid, declared here and not extended after any number is seen:
  `k in {30, 100, 300, 1000, 3000}`.** The argmax is taken; ties within 0.001
  nats per held-out row go to the LARGER `k` (more shrinkage, simpler model).
* The selected `k` is used for **both** arms, for **every** window, and for both
  shrinkage levels. It is published with the full grid of held-out scores.

This is a fit on training data against a likelihood, not a search against a gate
cell: no state cell, no MAE and no Decision 8 number is visible to it.

### 20.5 Arms

| arm | exit rule | shrinkage parent of the state cell | simplicity | status |
|---|---|---|---:|---|
| `R2_hier_dirichlet` (S1) | -- | -- | 1 | reference (incumbent, SERVED) |
| `K1_cond_class` (S1) | rank | -- | 9 | reference (round 6's best; NOT adoptable here) |
| `X1_exit_class` (S1) | class LEVEL | -- | 11 | reference (round 7's; NOT adoptable here) |
| `Y1_exit_rate` (S1) | class rate in the composition | the LEVEL `P(k_out given size, ce)` | 14 | reference (round 8's best; NOT adoptable here) |
| `Z1_exit_marg` | class rate in the composition | **the MARGINAL `P(k_out given size, n_st)`** | 16 | candidate |
| `Z2_exit_interact` | Z1 with the interaction gated at n >= 300 | marginal, or the cell where powered | 17 | candidate |

The simplicity order `R2 < K1 < X1 < Y1 < Z1 < Z2` is fixed here. **R2, K1, X1
and Y1 are references and none is adoptable in round 9.**

### 20.6 Scheme, folds, and what is refitted

**Scheme: S1 for every arm**, per rounds 3b-8. Windows are the calendar months of
the 2024-25 season; the first window trains on 2024 alone. No static column.

**Folds.** F1 = train 2024, test 2025, which IS the standing fold 2 (L13). 2026
stays sealed (`seal.assert_not_sealed` guards the trainer).

**Round 9 fits the two exit objects per window and nothing else**, on the **same
rows** as rounds 6, 7 and 8 (`--wave-team-games 6000`, fit seed 11).
`rotation_fit_v3*.json`, `rotation_v4_sub_*.json`, `round5/*`, `round6/*`,
`round7/*` and `round8/*` are REUSED and **nothing is written to any of them**;
the round-9 artifacts are new versioned siblings under `round9/`.

**The four reference columns (R2, K1, X1, Y1) are taken from the round-6, -7 and
-8 results JSONs**, not re-simulated: same 1,600-game universe, same subset seed
2025, same sim seeds 0-2, same base fits, hazards, wave and composition tables
and grading functions. A **1-seed re-run of Y1 is executed inside this round as a
reproduction check** and its cells are reported next to round 8's; **if any state
cell moves by more than its floor-A SD the reference columns are discarded and
the round is re-run in full**, as 18.5 declared for X1 and 16.5 for K1.

### 20.7 Test universe and grading path

The **same** 1,600-game subset of 2025 that rounds 2-8 used (numpy RandomState
seed 2025), **3 seeds per candidate arm** under S1, the **round-8 grading path
unchanged** (`train_rotation_v1.build_row` / `verdict` / `rotation.aggregate_stats`,
extended by `train_rotation_v4.extra_cells` / `.minutes_mae` and
`train_rotation_v5.wave_cells`, with round 6's `quintile_mae`). No gate cell is
added, so no grader line changes and the reference columns stay comparable byte
for byte. Any cell with n < 300 player-games, possessions or fitted rows is
labelled UNDERPOWERED and never read as signal or as absence of signal.

### 20.8 Gates -- every round-8 gate, unchanged, nothing added or relaxed

*G8 cells (report, not veto):* minutes mean +/- 2.0; minutes SD ratio pooled and
within-player 0.9-1.1; top-5 and top-8 share of team minutes +/- 2 pp; players
with > 0 minutes +/- 1.0.

*The eight state cells (the veto), each +/- 3 pp:* starters' share of on-floor
slots in the final 8:00 at |m| <= 5 / 6-15 / > 15; starters' share while carrying
>= 4 fouls; the second-half TIP starter share in each of the three margin bands;
starters' share over H1 20:00-10:00 at |m| <= 5. **An arm missing ANY of the
eight is ineligible regardless of G8 or of MAE.**

*The two round-5 cells (also veto):* `sub_rate_per_boundary` +/- 0.015 or 3x the
floor-A seed SD if larger; `distinct_lineups_per_game` +/- 1.5 or 3x the floor-A
seed SD if larger. The governing number is named in the results table.

*Report only:* the "at exactly 4 fouls" diagnostic; top-1 / top-3 / top-5
five-man lineup share; K-S D of the top-1 lineup share and of per-player minutes;
mean wave size; the as-of starter benchmark of 14.6 on all eight state cells;
**the exit-side starter share of 16.7** (200 games, seed 0, the as-of predicted
starter set on both sides, one function for every row: ACTUAL 0.5576 overall,
K1 0.4927, X1 0.5611, Y1 0.5422); **the twelve-minute time-since-reset gradient
of 17.10** (H2 tip at 0 minutes, H1 20:00-10:00 at 0-10, the final 8:00 close
band at 12+), quoted at twelve minutes as the round's headline drift number; and
**the realised exit rate by `n_starters_on_floor` in the SIM against the same
rate on the ACTUAL sequences** (19.7's table: ACTUAL span +36.5 pp, X1 +1.2,
Y1 +16.7, Y2 +23.6), which is the object the round exists to move.

### 20.9 Primary metric and the two responsiveness checks

**Primary metric: per-player minutes MAE**, unchanged from rounds 4-8.

**Responsiveness check 1 (Decision 8), unchanged from 18.8:** team-games bucketed
into quintiles of the pregame as-of share of team minutes going to the predicted
starting five; the close-and-late cell per quintile for ACTUAL and every arm,
with slope and Q5 - Q1. An arm whose slope ratio to actual falls outside
**[0.8, 1.2]**, or whose sign disagrees, is not adoptable. (ACTUAL +0.692;
K1 0.83 PASS; X1 0.62 FAIL; Y1 0.60 FAIL.) 19.13 item 4 predicts in advance that
a within-game restoring force cannot move this check; the prediction is recorded
here so the round can falsify it.

**Responsiveness check 2, per-player minutes MAE by PLAYER quintile, carried
forward unchanged from 18.8.** Players in the as-of rotation set are bucketed
into quintiles of their own pregame as-of minutes per game. An arm is adoptable
only if it beats K1 beyond the floor in the pooled MAE **and loses beyond the
floor in no single quintile to EITHER reference it is measured against -- K1
(rounds 6-9's base) or W4 (round 5's, whose Q2 column vetoed round 6's arms)**.
Both columns are read from the round-6 results JSON, as in rounds 7 and 8.
Carrying W4 forward is deliberate: round 9 must not become adoptable by dropping
a column. Underpowered quintiles are labelled.

### 20.10 Noise floors and the decision rule

**Floor A, seed-varied sim runs:** 20 seeds x 150 games per candidate arm, the SD
of every G8 cell, every state cell, both round-5 cells and the MAE -- the rounds
3/4/5/6/7/8 configuration, so seven rounds' floors are comparable. R2's, K1's,
X1's and Y1's floors are rounds 4's, 6's, 7's and 8's and are unchanged by a run
that does not refit them.

**Floor B, spec-identical refit under a second seed:** the round-9 exit objects
refitted from a different training-game sample (fit seed 101 vs 11) and simulated
under a different sim seed (23 vs 7), graded on the same 150-game universe, run
on the arm the decision rule selects or, failing that, on **Z1**, the arm the
round is built on. An arm counts as beating a reference on a cell only if its
improvement exceeds the refit-to-refit spread on that cell. **Floor B is
pre-registered as CONDITIONAL ON THE WALL CLOCK** -- the lane's hard stop is
13:05 ET -- and its absence, if it is absent, is reported and not hidden.

**Decision rule.** Adopt the **simplest** arm that

1. passes **every** one of the eight state cells at +/- 3 pp, AND
2. passes **both** round-5 cells at the tolerances of 20.8, AND
3. beats `R2_hier_dirichlet` on per-player minutes MAE by more than the floor, AND
4. beats `K1_cond_class` on per-player minutes MAE by more than the floor and
   loses beyond the floor to neither K1 nor W4 in any player quintile (20.9), AND
5. satisfies the Decision 8 slope check, AND
6. passes the Decision 10 freeze of 20.11.

Ties go to the simpler model in the order `R2 < K1 < X1 < Y1 < Z1 < Z2`. An arm
whose improvement on the cell it was built to fix does not clear floor B is not
adopted on that cell. **If no arm is eligible, adopt nothing**, report which cell
fails and by how much, name the diagnosis, and name the next structure. No gate
is relaxed to produce a winner and no cell is dropped after seeing a result.
**The served default is not changed by this lane in any case**; the PM switches
it.

### 20.11 Decision 10: the closed loop, and the adapter no lane has written

The round-9 exit rule carries a state term by design (the exit cell's time and
margin bands) exactly as rounds 7's and 8's did, so the freeze is the right
instrument and is required before any arm is served. Paired-stream runs over the
fixed **500-game subset** of the F2 2025 slate (sorted by `game_id` ascending,
every 11th row, the first 500), with `ENGINE_EVENT=round2_s1`,
`ENGINE_FG_MAKE=round3_shooter_S_C_s1`, `ENGINE_CLOCK=reference` pinned and
recorded in every `run_meta.json`, reporting margin SD ratio, home/away score
correlation, possessions per game and per-player minutes MAE, live against
`ENGINE_ROTATION_FREEZE=1`, **5 seeds**, on the arm the decision rule selects.

**Stated plainly, because it is now the third round running: the freeze cannot be
run for rounds 7, 8 or 9 because NO LANE HAS WRITTEN THE ENGINE ADAPTER.** It
needs a vectorised `next_lineup_round9` in `engine/rotation_adapter.py` behind a
new `ENGINE_ROTATION` value that is **default-off**, its parity tests, and two
500-game paired runs (~11 min). Round 7 could not fit it (17.13) and round 8 did
not (19.12). **Round 9 pre-registers writing that adapter, default-off and
untested against any gate, as a deliverable CONDITIONAL ON THE WALL CLOCK after
grading**, so that a later session can run the freeze without first writing code.
If an arm passes conditions 1-5 offline and the freeze cannot be run by the hard
stop, the arm is reported as **PASSES OFFLINE, FREEZE OUTSTANDING**, condition 6
is recorded as unmet, the arm is **not adopted**, and the served default is left
untouched. That is a reporting outcome, not a relaxed gate.

An arm that moves margin SD ratio, home/away correlation or possessions outside
the G1/G2 tolerances between live and frozen is not adopted, and no magnitude for
the loop is quoted from the freeze alone (L31).

### 20.12 Engine expressibility (a condition on adoption)

The round-9 tables have the SAME shape as Y1's, (5, 18, 6, 6), and are gathered
with the same four indices; only their fitted contents differ. Z2 adds no index
either -- its gate is applied at FIT time, so the sim-side gather is identical.
Round 9 is therefore expressible in the engine at exactly Y1's cost: one gather
into a (size, cell, n_st, k_out) CDF and no operation added to the (2N, S) roster
block, and the sim loop still makes no model call. Both ship behind
**`ENGINE_ROTATION=round9`** plus **`ENGINE_ROTATION_ARM=Z1|Z2`**, wired in
`engine/rotation_adapter.py` with S1 artifacts per month and a manifest in the
`engine/manifest.py` format under `data/processed/models/rotation/round9/`.
`ENGINE_ROTATION=reference` remains the default and `engine/adapters.py` is **not
touched by this lane**.

### 20.13 Disclosures

1. **Z1 and Z2 draw the same uniforms as X1 and Y1, in the same order** (round
   5's coupling draw, the wave draw, the size draw, the `k_out` uniform, K1's
   `k_in` uniform, then the bench race vector), so the round-9 arms are
   byte-aligned with rounds 7's and 8's arms and with each other. Y1's column is
   nevertheless taken from round 8's own JSON and checked by the 1-seed
   reproduction re-run of 20.6.
2. `k` is FITTED by 20.4's stated leave-one-fold-out criterion on 2024 training
   data over a grid declared before any fit; it is not hand-picked and it never
   sees a gate cell. The 300-row UNDERPOWERED threshold that Z2's gate uses is
   the project's own since round 5 and is not tuned.
3. `n_starters_on_floor` is the model's own predicted five, not the game's; the
   as-of starter benchmark of 14.6 measures what that identification costs and is
   reported, as in every round since 4.
4. No static column is run; floor B (20.10), the Decision 10 freeze and the
   adapter (20.11) are all conditional on the wall clock and their absence, if
   any, is reported.
5. The support for the inverted parent was measured BEFORE this pre-registration,
   in 19.1 and 19.7, and is published there.
6. 19.13 item 4's prediction -- that the Decision 8 slope will NOT move, because
   a within-game restoring force carries no between-team information -- is
   recorded here in advance of the result.

---


## 21. Round-9 results (train 2024, test 2025) -- run 2026-09-11T16:43Z

Pre-registration section 20, committed **3525996** before `rotation_v9.py`
existed and before anything was fitted; code at **354e4d8**. Evidence doc:
`docs/tests/rotation_exit_hierarchy_2026-09-11.md`.

Test universe: the **same** 1,600-game subset of 2025 rounds 2-8 used (numpy
RandomState seed 2025), 3 seeds per candidate arm under S1, the round-8 grading
path unchanged. Windows and games: 202411 147, 202412 297, 202501 438, 202502
444, 202503 267, 202504 7 (UNDERPOWERED at 7 games, reported only because it
exists). Round 8's counts and sampler, round 7's mechanism, round 6's
composition tables, round 5's wave tables, round 4's hazards and round 3b's base
fits are reused and never written to; round 9 fits only the two exit objects, on
6,000 team-games per window (fit seed 11), **124,260-126,402 waves per window of
which 98.2% are usable -- the same counts, window by window, that rounds 6, 7
and 8 reported**, as 20.6 required.

**ACTUAL on this universe**, through the same functions as every arm:
substitutions per boundary **0.1509**, distinct lineups per team-game
**14.8356**, minutes mean 24.551, top-5 share 0.7472, top-8 0.9560, players with
> 0 minutes 9.6438, pooled minutes SD 9.6946, within-player 6.1196.

### 21.1 The fitted shrinkage constant (20.4)

Leave-one-fold-out over 5 disjoint folds of the 2024 training season,
24,218-24,597 held-out exit rows per fold, held-out multinomial log-likelihood
per row, 37 s:

| `k` | 30 | 100 | 300 | 1000 | 3000 |
|---|---:|---:|---:|---:|---:|
| held-out nats/row | **-0.64950** | -0.65448 | -0.66471 | -0.68608 | -0.71814 |

**The argmax is `k = 30`, the smallest value on the declared grid**, and the
score is monotone decreasing in `k` across the whole grid, so the criterion is
not sitting on an interior ridge: with the MARGINAL as the parent the data want
**ten times less shrinkage** than rounds 7 and 8's declared `k = 300`. That is
the expected direction and it is the first quantitative consequence of 20.1 --
a thin cell pulled onto a parent that is nearly right can afford to be pulled
gently, where a thin cell pulled onto a parent that is wrong must be pulled hard
to hide the parent's error. The grid was declared in 20.4 before any fit; it was
not extended after the boundary value won, and `k = 30` is reported as the grid
argmax with that caveat named here rather than in a footnote. The same `k` is
used for both arms, every window and both shrinkage levels.

### 21.2 The fitted object: the inverted parent recovers the rate

`P(k_out = 1 | size 1, n_starters_on_floor)`, window 202411 (the six windows
agree to 0.1-1.1 pp on every level, as rounds 7's and 8's did):

| starters on floor | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| **level 1, the MARGINAL `M1`** | 0.0432 | **0.3227** | 0.4010 | 0.5001 | 0.6574 | 0.9987 |
| **fitted Z1** (mean over the 18 cells) | 0.0327 | **0.3076** | 0.3881 | 0.5032 | 0.6715 | 0.9996 |
| fitted Z2 (mean over the 18 cells) | 0.0432 | 0.3282 | 0.3934 | 0.4939 | 0.6553 | 0.9993 |
| *round 8's Y1, same cell* | *0.5521* | *0.5202* | *0.4939* | *0.5348* | *0.6316* | *0.7553* |
| ACTUAL (17.17, 2025) | -- | **0.3697** | 0.4374 | 0.5209 | 0.6562 | 1.0000 |
| fitted rows, summed over the 18 cells | 379 | 2,764 | 12,646 | 28,762 | 28,011 | 9,582 |

**The table Y1 could not fit is fitted.** At one starter on the floor Y1's
shrinkage to X1's level produced 0.5202 against a real 0.3697; Z1 produces
**0.3076**, and the residual is now an ordinary sampling gap in the marginal
itself (M1 0.3227 on 2,764 training rows) rather than 19 pp of a wrong parent.
At five starters Y1 read 0.7553 -- structurally impossible, since a single swap
from an all-starter floor must remove a starter -- and Z1 reads 0.9996.

**Support, published as 20.3 required.** 61-63 of the 108 size-1
(exit cell x n_st) cells carry fewer than 300 rows across the six windows --
**the same thin cells Y1 had**; what changed is only what they fall back on.
That is the round's whole content and it is stated as such.

### 21.3 G8 cells (report, not veto) -- S1

| cell | tol | ACTUAL | R2 | K1 | X1 | Y1 | Z1 | Z2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| minutes mean (rotation players) | +/- 2.0 | 24.551 | 25.364 P | 25.636 P | 24.623 P | 24.947 P | 25.076 P | 25.063 P |
| minutes SD ratio, pooled | 0.9-1.1 | 1.0000 | 1.0548 P | 0.9961 P | 0.8853 **F** | 0.9139 P | **0.9300 P** | 0.9281 P |
| minutes SD ratio, within-player | 0.9-1.1 | 1.0000 | 1.3416 **F** | 1.1239 **F** | 1.1447 **F** | 1.1380 **F** | 1.1390 **F** | 1.1370 **F** |
| top-5 share of team minutes | +/- 2 pp | 0.7472 | 0.7630 P | 0.7597 P | 0.7232 **F** | 0.7338 P | **0.7385 P** | 0.7380 P |
| top-8 share of team minutes | +/- 2 pp | 0.9560 | 0.9626 P | 0.9696 P | 0.9552 P | 0.9606 P | 0.9630 P | 0.9626 P |
| players with > 0 minutes | +/- 1.0 | 9.644 | 9.129 P | 8.935 P | 9.033 P | 8.995 P | 8.974 P | 8.977 P |
| **G8 passed** | | | **5/6** | **5/6** | 3/6 | **5/6** | **5/6** | **5/6** |

The two cells round 7 broke and round 8 repaired move further the right way
(pooled SD ratio 0.885 -> 0.914 -> **0.930**; top-5 share 0.7232 -> 0.7338 ->
**0.7385** against a real 0.7472). The within-player SD ratio fails for **every
arm in the bake-off including both references** and has since round 4; it is not
a round-9 regression.

### 21.4 State cells (the veto) -- S1

ACTUAL, then each arm with its gap in pp and PASS/FAIL at +/- 3 pp:

| cell | ACTUAL | R2 | K1 | X1 | Y1 | Z1 | Z2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7491 | 0.7254 (-2.4) P | 0.7001 (-4.9) F | 0.5394 (-21.0) F | 0.5737 (-17.5) F | **0.6019 (-14.7) F** | 0.5920 (-15.7) F |
| final 8:00, \|m\| 6-15 | 0.7240 | 0.6914 (-3.3) F | 0.6890 (-3.5) F | 0.5384 (-18.6) F | 0.5697 (-15.4) F | **0.5939 (-13.0) F** | 0.5888 (-13.5) F |
| final 8:00, \|m\| > 15 | 0.5223 | 0.4694 (-5.3) F | 0.5083 (-1.4) P | 0.5183 (-0.4) P | 0.5255 (+0.3) P | 0.5371 (+1.5) P | 0.5510 (+2.9) P |
| starters at >= 4 fouls | 0.4613 | 0.4626 (+0.1) P | 0.4212 (-4.0) F | 0.3823 (-7.9) F | 0.3925 (-6.9) F | **0.4018 (-5.9) F** | 0.4051 (-5.6) F |
| H2 TIP, \|m\| <= 5 | 0.9678 | 0.7892 (-17.9) F | 0.9385 (-2.9) P | 0.9406 (-2.7) P | 0.9407 (-2.7) P | 0.9392 (-2.9) P | 0.9397 (-2.8) P |
| H2 TIP, \|m\| 6-15 | 0.9611 | 0.7977 (-16.3) F | 0.9410 (-2.0) P | 0.9424 (-1.9) P | 0.9414 (-2.0) P | 0.9409 (-2.0) P | 0.9406 (-2.1) P |
| H2 TIP, \|m\| > 15 | 0.9563 | 0.7338 (-22.3) F | 0.9456 (-1.1) P | 0.9449 (-1.1) P | 0.9453 (-1.1) P | 0.9434 (-1.3) P | 0.9431 (-1.3) P |
| H1 20:00-10:00, \|m\| <= 5 | 0.7822 | 0.5988 (-18.3) F | 0.7267 (-5.6) F | 0.6991 (-8.3) F | 0.7105 (-7.2) F | 0.7122 (-7.0) F | 0.7122 (-7.0) F |
| **state cells passed** | | **2/8** | **4/8** | **4/8** | **4/8** | **4/8** | **4/8** |

R2's, K1's, X1's and Y1's columns are rounds 6's, 7's and 8's, per 20.6; the
reproduction check is 21.7. **Every arm passes the same four cells.** Z1 recovers
a further 2.8 pp of the close band and 2.4 pp of the middle band on top of
round 8's 3.5 and 3.2 -- 1.8 and 2.0 floor-A SDs each, real movement in the
right direction, and **6.3 of X1's 21.0 pp now closed against K1's own 4.9 pp
shortfall**. The count of passed cells does not move, because the cells that
fail, fail by 5.9-14.7 pp.

### 21.5 The two round-5 cells, the primary metric, and concentration -- S1

Tolerances, computed as pre-registered: `sub_rate_per_boundary` **+/- 0.015**
governs (3x floor = 0.0053); `distinct_lineups_per_game` **+/- 1.5** governs
(3x floor = 0.621).

| metric | ACTUAL | R2 | K1 | X1 | Y1 | Z1 | Z2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **substitutions per boundary** | 0.1509 | 0.1437 P | 0.1554 P | 0.1557 P | 0.1559 P | 0.1560 P | 0.1561 P |
| **distinct lineups per team-game** | 14.836 | 15.514 P | **14.571 P** | 15.939 P | 15.899 P | 15.854 P | 15.865 P |
| **per-player minutes MAE (min)** | 0.0 | 9.7939 | **8.8622** | 9.6159 | 9.3851 | **9.3132** | 9.3191 |
| MAE gain over R2 (floors) | -- | -- | +63 | +11.0 | +27.8 | **+32.6** | +32.2 |
| MAE gain over X1 (floors) | -- | -- | +46.5 | -- | +14.2 | **+18.7** | +18.3 |
| MAE gain over Y1 (floors) | -- | -- | +39.3 | -17.3 | -- | **+5.4** | +5.0 |
| MAE gain over K1 (floors) | -- | -- | -- | -46.5 | -39.3 | **-33.5** | -33.9 |
| top-1 lineup share | 0.2940 | 0.2286 | 0.2568 | 0.2345 | 0.2341 | 0.2345 | 0.2343 |
| top-3 lineup share | 0.5426 | 0.4819 | 0.5358 | 0.4995 | 0.4996 | 0.5007 | 0.5006 |
| top-5 lineup share | 0.6894 | 0.6409 | 0.7043 | 0.6663 | 0.6672 | 0.6683 | 0.6682 |
| K-S D, per-player minutes | -- | 0.0798 | 0.0589 | 0.0441 | **0.0413** | 0.0414 | 0.0414 |
| K-S D, top-1 lineup share | -- | 0.2273 | **0.0990** | 0.1993 | 0.1972 | 0.1981 | 0.1979 |
| "at exactly 4 fouls" (report) | 0.5166 | 0.5722 | 0.5931 | **0.5316** | 0.5473 | 0.5666 | 0.5714 |

**Inverting the parent is worth +0.072 minutes of per-player MAE over Y1 -- 5.4
floors, on top of round 8's 14 -- and the arm is still 0.451 minutes, 33.5
floors, behind K1.** Z2 and Z1 differ by 0.006 minutes, **0.46 of a floor**: the
hard 300-row gate and the continuous shrinkage at `k = 30` are the same model
within the noise, and 20.10's tie-break takes the simpler arm, Z1.

### 21.6 Noise floor A (20 seeds x 150 games, S1)

| metric | Z1 | Z2 | (Y1, round 8) | (X1, round 7) |
|---|---:|---:|---:|---:|
| minutes_mae | 0.01280 | 0.01210 | 0.01331 | 0.01621 |
| sub_rate_per_boundary | 0.00170 | 0.00176 | 0.00178 | 0.00202 |
| distinct_lineups_per_game | 0.1977 | 0.2071 | 0.1906 | 0.2396 |
| late_starter_share_b0 | 0.01534 | 0.01560 | 0.01734 | 0.01946 |
| late_starter_share_b1 | 0.01175 | 0.01147 | 0.01307 | 0.01353 |
| late_starter_share_b2 | 0.01916 | 0.01962 | 0.01948 | 0.02197 |
| foul_trouble_share | 0.02364 | 0.02303 | 0.02342 | 0.02003 |
| h2tip_starter_share_b0 | 0.01109 | 0.01063 | 0.00976 | 0.00793 |
| h2tip_starter_share_b1 | 0.00917 | 0.00930 | 0.00806 | 0.00882 |
| h2tip_starter_share_b2 | 0.01605 | 0.01606 | 0.01545 | 0.01570 |
| opentip_starter_share_close | 0.01095 | 0.01085 | 0.01199 | 0.01460 |
| top5_share | 0.00292 | 0.00312 | 0.00293 | 0.00263 |
| n_nonzero_mean | 0.06483 | 0.06648 | 0.06412 | 0.06531 |

R2's, K1's, X1's and Y1's floors are rounds 4's, 6's, 7's and 8's and are
unchanged by a run that does not refit them (R2 minutes_mae 0.01473, K1 0.01348,
X1 0.01621, Y1 0.01331). **Every miss in 21.4 is far larger than its floor**:
Z1's close-band miss is -14.7 pp against a 1.53 pp floor (**9.6 floors**), the
middle band -13.0 pp against 1.18 pp (11.1 floors), foul trouble -5.9 pp against
2.36 pp (2.5 floors) and the opening ten minutes -7.0 pp against 1.10 pp (6.4
floors). The floor-A caveat of 11.6 applies unchanged: only the seed-to-seed SD
is a floor, never the `minutes_mae` LEVEL of that 150-game run.

### 21.7 The Y1 reproduction check (20.6)

A 1-seed re-run of Y1 inside this round, against round 8's own 3-seed column:

| cell | round 8 (3 seeds) | round 9 (1 seed) | delta | floor A |
|---|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.5737 | 0.5687 | -0.0050 | 0.0173 |
| final 8:00, \|m\| 6-15 | 0.5697 | 0.5707 | +0.0010 | 0.0131 |
| final 8:00, \|m\| > 15 | 0.5255 | 0.5305 | +0.0050 | 0.0195 |
| starters at >= 4 fouls | 0.3925 | 0.3853 | -0.0071 | 0.0234 |
| H2 tip, \|m\| <= 5 | 0.9407 | 0.9379 | -0.0027 | 0.0098 |
| H2 tip, \|m\| 6-15 | 0.9414 | 0.9356 | -0.0058 | 0.0081 |
| H2 tip, \|m\| > 15 | 0.9453 | 0.9485 | +0.0032 | 0.0155 |
| H1 20:00-10:00, \|m\| <= 5 | 0.7105 | 0.7119 | +0.0014 | 0.0120 |
| substitutions per boundary | 0.15595 | 0.15538 | -0.00057 | 0.00178 |
| distinct lineups | 15.899 | 15.773 | -0.1263 | 0.1906 |

**Every state cell moves less than its floor-A SD, so the reference columns
stand** (the largest move, foul trouble, is -0.71 pp against a 2.34 pp floor).
The MAE moves +0.005 on one seed against a 0.013 floor.

### 21.8 The object the round exists to move: the exit rate BY COMPOSITION

`scripts/diag_rotation_exit_v9.py`, 200 games, seed 0, the as-of predicted
starter set on BOTH sides, one function for every row.
`P(a starter is the man who leaves | single swap, starters on the floor)`:

| starters on floor | ACTUAL | n | X1 (r7) | Y1 (r8) | n | **Z1** | n | Z2 | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3115 | 244 UP | 0.5774 | 0.4612 | 451 | **0.3049** | 387 | 0.3333 | 399 |
| 2 | 0.4369 | 982 | 0.5752 | 0.4447 | 1,158 | **0.3943** | 1,149 | 0.4009 | 1,120 |
| 3 | 0.5115 | 1,775 | 0.5443 | 0.4844 | 1,734 | **0.4818** | 1,818 | 0.4741 | 1,797 |
| 4 | 0.6761 | 1,689 | 0.5892 | 0.6286 | 1,578 | **0.6468** | 1,608 | 0.6384 | 1,615 |
| 5 | 1.0000 | 544 | 1.0000 | 1.0000 | 597 | 1.0000 | 575 | 1.0000 | 579 |
| **span, 1 -> 4** | **+36.5 pp** | | **+1.2 pp** | **+16.7 pp** | | **+34.2 pp** | | **+30.5 pp** | |
| **% of the real span** | 100% | | 3% | **46%** | | **94%** | | 84% |

(UP = UNDERPOWERED at n < 300 and labelled, never read as signal.)

**This is the round's result.** X1's simulated rate was flat (a level); Y1's
carried 46% of the span (a rate attenuated by a wrong parent); **Z1's carries
94%**, and every level is within 0.7-4.3 pp of the actual rate against Y1's
2.8-15.0 pp. The fitted-table arithmetic of 21.2 and the simulated rate agree,
which is the paired check rounds 7 and 8 established for this family.

The simulated floor composition moves with it. Single swaps taken with 1 or 2
starters on the floor: **ACTUAL 1,226, X1 1,804, Y1 1,609, Z1 1,536, Z2 1,519**
-- the bench-heavy tail is 15% smaller than X1's and still **25% too big**, so
the residual is no longer the conditional rate but the composition DISTRIBUTION
the rest of the mechanism produces.

**The exit-side starter share (16.7), so round 9 drops into 17.10's table:**

| | leavers, overall | size 1 | size 2 | size 3+ | entrants | size-1 joint (bench out / starter out) | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ACTUAL (as-of starters)** | **0.5576** | **0.5867** | **0.5576** | **0.4931** | **0.4972** | 0.6848 / 0.3446 | 34.0 pp |
| K1 (round 6) | 0.4927 | 0.4564 | 0.5142 | 0.5496 | 0.4334 | 0.6344 / 0.2026 | 43.2 pp |
| X1 (round 7) | 0.5611 | 0.6013 | 0.5648 | 0.4616 | 0.4882 | 0.6762 / 0.3073 | 36.9 pp |
| Y1 (round 8) | 0.5422 | 0.5636 | 0.5368 | 0.5000 | 0.4769 | 0.6891 / 0.2798 | 40.9 pp |
| **Z1** | 0.5311 | 0.5485 | 0.5232 | 0.5032 | 0.4694 | 0.6869 / 0.2593 | 42.8 pp |
| Z2 | 0.5297 | 0.5472 | 0.5215 | 0.5028 | 0.4675 | 0.6812 / 0.2579 | 42.3 pp |

The report-only marginal continues to drift down (X1 0.5611 -> Y1 0.5422 ->
Z1 0.5311 against 0.5576) **for the reason 19.7 gave and it is still the right
trade**: a marginal is reproduced only when the conditional rate AND the
composition distribution are both right, and Z1 now has the rate right and the
distribution still bench-heavy, so the marginal must come out low. Trading 1.1 pp
more of a report-only marginal for 2.8 pp of a vetoed state cell is the direction
20.1 asked for, and the arm that hit the marginal exactly (X1) is the arm that is
worst on every cell that matters.

**The time-since-reset gradient (17.10), the round's headline drift number:**

| cell | minutes since the last forced reset | K1 gap | X1 gap | Y1 gap | **Z1 gap** |
|---|---:|---:|---:|---:|---:|
| H2 tip, all three margin bands | 0 | -1.1 to -2.9 pp | -1.1 to -2.7 pp | -1.1 to -2.7 pp | -1.3 to -2.9 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0-10 | -5.6 pp | -8.3 pp | -7.2 pp | **-7.0 pp** |
| final 8:00, \|m\| <= 5 | 12+ | -4.9 pp | **-21.0 pp** | -17.5 pp | **-14.7 pp** |

**The gradient is flattened by 30% from X1 and 16% from Y1, and it is not
removed.** Z1 is still correct at the reset and still drifts monotonically away
from it, at 0.70 of X1's rate. Two thirds of the drift survive a conditional exit
rate that is now 94% right, which is 21.13 item 3.

### 21.9 The two responsiveness conditions

**Decision 8 (20.9 check 1).** Cell = starters' share in the final 8:00 at
\|margin\| <= 5, by quintile of the pregame as-of predicted starter-minutes
share (640 team-games per quintile).

| quintile | ACTUAL | K1 | X1 | Y1 | **Z1** | Z2 |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 0.6770 | 0.6218 | 0.4921 | 0.5261 | 0.5573 | 0.5448 |
| Q2 | 0.7226 | 0.6940 | 0.5280 | 0.5607 | 0.5913 | 0.5830 |
| Q3 | 0.7471 | 0.7089 | 0.5372 | 0.5754 | 0.6070 | 0.5958 |
| Q4 | 0.7763 | 0.7299 | 0.5563 | 0.5923 | 0.6236 | 0.6151 |
| Q5 | 0.8219 | 0.7430 | 0.5826 | 0.6128 | 0.6289 | 0.6199 |
| **slope** | **+0.692** | +0.576 | +0.426 | +0.416 | **+0.355** | +0.370 |
| slope ratio to actual | 1.00 | **0.83 PASS** | 0.62 FAIL | 0.60 FAIL | **0.51 FAIL** | 0.53 FAIL |
| Q5 - Q1 (pp) | +14.5 | +12.1 | +9.1 | +8.7 | **+7.2** | +7.5 |

Every arm is monotone in 4 of 4 steps. **20.13 item 6 recorded 19.13 item 4's
prediction in advance -- that this check would not move -- and the round
falsifies it in the WORSE direction: the slope ratio falls 0.60 -> 0.51.** The
mechanism is visible in the column: Z1 lifts Q1 by 3.1 pp and Q5 by only 1.6 pp,
because the composition axis is a restoring force and the quintile that spends
most time bench-heavy has most to restore. **A within-game restoring force does
not merely fail to carry between-team information -- fixing it COMPRESSES the
between-team response**, and any object that repairs the drift will do the same
unless something team-indexed is added at the same time. That is a stronger and
more useful statement than the one 19.13 item 4 made, and it is the round's
sharpest negative result.

**The per-player-quintile condition (20.9 check 2).** Per-player minutes MAE by
quintile of the player's own pregame as-of minutes per game (4,782-4,784
player-games per quintile, none underpowered):

| quintile | K1 (r6) | W4 (r6) | X1 (r7) | Y1 (r8) | **Z1** | Z2 | Z1 - K1 | Z1 - W4 | Z1 - Y1 | floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 (lowest mpg) | 9.3108 | 9.6681 | 10.6053 | 10.2595 | 10.0878 | 10.0770 | **+0.777** | +0.420 | **-0.172** | 0.013 |
| Q2 | 9.7891 | 9.7294 | 10.4615 | 10.2980 | 10.2116 | 10.2281 | **+0.423** | +0.482 | **-0.086** | 0.013 |
| Q3 | 9.2575 | 9.4074 | 9.7867 | 9.6199 | 9.6052 | 9.5921 | +0.348 | +0.198 | -0.015 | 0.013 |
| Q4 | 8.6461 | 8.6621 | 9.1081 | 8.9462 | 8.9734 | 8.9530 | +0.327 | +0.311 | **+0.027** | 0.013 |
| Q5 (highest mpg) | 7.2367 | 7.3307 | 8.0130 | 7.7025 | 7.5951 | 7.6538 | +0.358 | +0.264 | **-0.107** | 0.013 |

**Z1 loses all five quintiles to both K1 and W4** (24-58 floors) and beats Y1 in
four of five (Q1 by 12.9 floors, Q2 6.5, Q5 8.1, Q3 1.1 -- inside the floor) and
**loses Q4 by 2.0 floors**. The gain is concentrated exactly where the mechanism
predicts it: the deepest-bench quintile, whose minutes depend most on how often
the floor goes bench-heavy.

### 21.10 Per-game and per-team evidence

**Per team** (starters' share in the final 8:00 at \|m\| <= 5, aggregated per
team; a team is powered when both denominators reach 300 on-floor slots --
**251 powered, 112 UNDERPOWERED and excluded**):

| | sim mean | sim SD | actual mean | actual SD | corr(sim, actual) | mean \|dev\| |
|---|---:|---:|---:|---:|---:|---:|
| K1 (round 6) | 0.7004 | 0.0507 | 0.7512 | 0.0832 | **0.394** | 0.0757 |
| X1 (round 7) | 0.5412 | 0.0599 | 0.7512 | 0.0832 | 0.209 | 0.2108 |
| Y1 (round 8) | 0.5726 | 0.0548 | 0.7512 | 0.0832 | 0.211 | 0.1793 |
| **Z1** | 0.6018 | 0.0495 | 0.7512 | 0.0832 | **0.190** | **0.1518** |
| Z2 | 0.5927 | 0.0529 | 0.7512 | 0.0832 | 0.190 | 0.1605 |

The mean absolute deviation falls 0.211 -> 0.179 -> **0.152** while **the
correlation with the team's own actual share falls, 0.209 -> 0.211 -> 0.190**
against K1's 0.394, and the cross-team SD falls 0.060 -> 0.055 -> 0.050 against a
real 0.083. Per team exactly as per quintile in 21.9: **the level improves and
the matchup responsiveness gets slightly worse**, and the two measurements are
the same fact seen twice.

**Per-player minutes seed SD** (3 seeds, the same universe): Z1 0.0257,
Z2 0.0433, Y1 0.0395, X1 0.0379, K1 0.0737.

### 21.11 The as-of starter benchmark (report only, 20.8)

The ACTUAL sequence of the same 1,600 games, re-graded with the MODEL's as-of
predicted starting five instead of the game's own; the model's five overlaps the
real five on **4.576 of 5**, unchanged from rounds 6, 7 and 8. Starter
identification costs -2.8 pp on the close-and-late band (0.7215 against 0.7491),
-2.8 pp on the middle band, -4.4 pp on the opening ten minutes and **+0.2 pp on
foul trouble**. Against that benchmark Z1 is **-12.0 pp** on the close band
(Y1 -14.8, X1 -18.2, K1 -2.1) and -2.6 pp on the opening cell. **It changes no
tolerance and no verdict**: every arm is scored against each side's own real
starting five, as in rounds 1-8.

### 21.12 Decision

| arm | simplicity | state | round-5 cells | G8 | MAE | vs R2 | vs K1 | vs Y1 | quintiles | D8 | Dec-10 | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| R2_hier_dirichlet | 1 | 2/8 | 2/2 | 5/6 | 9.7939 | -- | -0.932 | -0.409 | ok | n/a | -- | NO (reference) |
| K1_cond_class | 9 | 4/8 | 2/2 | 5/6 | **8.8622** | +0.932 | -- | +0.523 | see 15.8 | 0.83 | PASS (15.13) | NO (reference) |
| X1_exit_class | 11 | 4/8 | 2/2 | 3/6 | 9.6159 | +0.178 | -0.754 | -0.231 | 5/5 fail | 0.62 F | NOT RUN | NO (reference) |
| Y1_exit_rate | 14 | 4/8 | 2/2 | 5/6 | 9.3851 | +0.409 | -0.523 | -- | 5/5 fail | 0.60 F | NOT RUN | NO (reference) |
| Z1_exit_marg | 16 | 4/8 | 2/2 | **5/6** | **9.3132** | +0.481 | **-0.451** | **+0.072** | **5/5 fail** | **0.51 F** | NOT RUN | NO |
| Z2_exit_interact | 17 | 4/8 | 2/2 | 5/6 | 9.3191 | +0.475 | -0.457 | +0.066 | **5/5 fail** | **0.53 F** | NOT RUN | NO |

**No arm adopted.** Z1 fails condition 1 (four of eight state cells, two of them
by 13-15 pp), condition 4 (loses 0.451 minutes of MAE to K1 and every player
quintile to both K1 and W4) and condition 5 (Decision 8 slope ratio 0.51 against
a [0.8, 1.2] band, **worse than Y1's 0.60**); condition 6 was not run. It passes
conditions 2 and 3. Z2 fails the same three and is inside Z1's floor on every
cell, so 20.10's tie-break takes the simpler arm in any case. Per 20.10 the rule
adopts nothing; no gate was relaxed and no cell was dropped after seeing a
result. `ENGINE_ROTATION=reference` (R2) stays the served default and **this lane
changed no default**; the `round9` flag ships nowhere, because 20.12's adapter
was not wired (21.14).

Cell-by-cell misses of the closest arm:

| arm | cell | sim | actual | miss | floors | of which starter identification |
|---|---|---:|---:|---:|---:|---:|
| Z1 | final 8:00 \|m\| <= 5 | 0.6019 | 0.7491 | -14.7 pp | 9.6 | -2.8 pp |
| Z1 | final 8:00 \|m\| 6-15 | 0.5939 | 0.7240 | -13.0 pp | 11.1 | -2.8 pp |
| Z1 | starters at >= 4 fouls | 0.4018 | 0.4613 | -5.9 pp | 2.5 | +0.2 pp (none) |
| Z1 | H1 20:00-10:00 \|m\| <= 5 | 0.7122 | 0.7822 | -7.0 pp | 6.4 | -4.4 pp |
| Z1 | player quintile Q1 | 10.0878 | (K1 9.3108) | +0.777 | 58 | -- |
| Z1 | D8 slope ratio | 0.513 | (band 0.8-1.2) | -0.287 | -- | -- |

### 21.13 Diagnosis

**1. The shrinkage parent WAS the binding constraint on the fitted object, and
inverting it removes that constraint almost completely.** 19.13 item 3 predicted
the arithmetic; 21.2 and 21.8 measure it. The fitted rate at one starter on the
floor goes 0.5202 -> 0.3076 against a real 0.3697, the simulated rate 0.4612 ->
0.3049 against 0.3115, and the simulated span 46% -> **94%** of the real 36.5 pp.
The diagnosis was right and the fix does exactly what it was designed to do.

**2. And the state cells move by a quarter of what the object moved.** The
conditional exit rate went from 46% right to 94% right; the close-band miss went
from -17.5 to -14.7 pp, a 16% improvement, and the drift gradient from 17.5 to
14.7 pp of a 21.0 pp starting point. **The exit rate is no longer the dominant
term in the drift.** Two rounds have now moved this object from flat to
essentially correct and the close-and-late cell has recovered 6.3 of X1's 21.0
pp, against K1's own residual of 4.9 pp on the same cell -- which means the
remaining 9.8 pp lives somewhere else in the mechanism.

**3. Where it lives is measurable and is named here, not guessed.** The
simulated composition distribution is still 25% too bench-heavy (21.8: 1,536
single swaps at 1-2 starters against 1,226) while the exit rate CONDITIONAL on
that composition is right to 0.7-4.3 pp. A correct conditional rate over a wrong
state distribution is exactly the signature of the WAVE side, not the exit side:
round 5's `P(wave)` and `P(size | cell)` decide how often and how big the swaps
are, they carry no composition axis at all, and every round since has conditioned
only on what happens WITHIN a wave once it has been triggered. **The next object
is `P(wave | cell, n_starters_on_floor)` and `P(size | cell, n_starters_on_floor)`
-- the same axis, one stage upstream** -- and its support should be measured
first on the actual sequences, as 17.17 was measured before round 8.

**4. Decision 8 is not merely untouched by this family; it is actively harmed by
it, and round 9 measures that for the first time.** 19.13 item 4 said a
within-game restoring force carries no between-team information. Round 9 shows
the stronger fact: the restoring force lifts the shallow-rotation quintile 3.1 pp
and the deep-rotation quintile 1.6 pp, so the slope ratio falls 0.60 -> 0.51 and
the per-team correlation 0.211 -> 0.190. **Every future repair to the drift will
push Decision 8 the wrong way** unless the same round adds an object indexed by
something the TEAM brings -- its own as-of starter-minutes share -- which no
rotation arm in nine rounds has carried. Decision 8 should be treated as a
blocking, separately-specified round, not as a gate a drift fix will eventually
clear.

**5. The fitted `k` is evidence in its own right.** With the level as parent,
rounds 7-8 used `k = 300` by declaration. With the marginal as parent the
held-out likelihood is monotone decreasing across the whole declared grid and
picks `k = 30` (21.1). A parent that is nearly right needs little pull; the
grid's left boundary winning is reported as such, and a later round reviving this
family should widen the grid downward and re-fit rather than inherit 30.

**6. Z2 adds nothing.** The hard 300-row gate and the continuous shrinkage differ
by 0.006 minutes of MAE (0.46 of a floor) and 1.0 pp on the close band, with the
same 4/8 and the same quintile failures. The simpler arm is the family, the same
tie-break rounds 7 and 8 recorded for X2/X3 and Y2.

**7. Foul trouble: the ordering of 17.14 item 5 still holds.** The graded cell
improves 0.3823 -> 0.3925 -> 0.4018 with the drift and is still -5.9 pp, while
the report-only "at exactly 4 fouls" diagnostic drifts the other way
(0.5316 -> 0.5473 -> 0.5666 against 0.5166). The cell will not be readable until
the drift is fixed, which is item 3.

### 21.14 Artifacts and flags

- Exit objects: `data/processed/models/rotation/round9/rotation_v9_exit_{YYYYMM}.json`
  (six windows) with `rotation_v9_manifest.json` in the `engine/manifest.py`
  format, and `rotation_v9_kselect.json` carrying the full leave-one-fold-out
  grid; each manifest entry carries `refit_date`, `max_train_date`, the fitted
  `k`, the reused hazard, wave and composition artifact names, the published
  `P(k_out=1|size 1)` rows for the marginal, Z1 and Z2 by composition, the
  per-level row counts and the under-300 cell counts. The directory is
  gitignored (`data/processed/models/*/round*/`) and HF-synced.
- Nothing was written to `rotation_fit.json`, `rotation_fit_v3*.json`, any
  `rotation_v4_sub_*.json`, any `round5/`, `round6/`, `round7/` or `round8/`
  artifact.
- Engine: **not wired** (21.15). `ENGINE_ROTATION=round9` does not exist,
  `engine/adapters.py` and `engine/loop.py` were not touched by this lane, and
  `ENGINE_ROTATION=reference` remains the default.
- Tests: `tests/test_rotation_v9.py` (9 cases: table shapes and normalisation,
  support clipping at every composition, an empty state cell equalling the
  MARGINAL exactly, Z2's 300-row gate on both sides, the marginal being pooled
  over exit cells and sloping in `n_st`, the hierarchy differing from round 8's
  on a thin cell in the direction 20.2 declares, `select_k` being a held-out
  likelihood over the declared grid with the stated tie-break, the sampler
  delegating to `run_wave8`, and the declared constants and arm grid)
  -> **9 passed**.
- Results: `rotation_F1_round9_{results.json,table.csv}`,
  `exit_audit_round9_2026-09-11.json`, `rotation_F1_round9_floorB_Z1.json`.
- Cost: the k fit 37 s (5 folds, 5 workers); the six exit fits 17-23 s each,
  run concurrently (5 workers); the five grade/floor jobs 54-138 s each; the
  whole bake-off **5.3 min**; the exit diagnostic 1.1 min, run on the sixth
  worker after the fit stage landed.

### 21.15 Decision 10: NOT RUN for the THIRD round running, and the adapter

20.11 pre-registered the freeze as conditional on the wall clock and
pre-registered writing the adapter as a conditional deliverable. **Neither was
done.** The adapter was scoped inside the lane and the scope is reported here so
the next session does not re-scope it: `next_lineup_round9` is
`next_lineup_round6`'s K1 path with the exit block replaced --

* widen the `rotation_sub` draw block from `2S + 4` to `2S + 5` and take the new
  scalar as the `k_out` uniform (a STATED RNG divergence from round 6, exactly as
  round 6 declared one from round 5: round-9 arms are paired with each other and
  not with round 6);
* `ce = V7.exit_cell(period, seconds_remaining, margin, foul_state)` and
  `n_st = (on & is_starter).sum(axis=1)`, both already computable from the
  round-6 batch;
* gather `row = rb.zexit[rb.seg, clip(size,1,5)-1, ce, clip(n_st,0,5)]`, clip the
  support to `[max(0, size - n_bench_on, forced_starters),
  min(size, n_starters_on, size - forced_bench)]`, renormalise, draw `k_out`;
* `leaving = _pick_k(on & is_st, key_out, k_out) | _pick_k(on & ~is_st, key_out,
  size - k_out)` with round 5's `key_out`, then round 6's K1 entry block
  unchanged, reading `k_out` off `leaving` as it already does;
* plus `Round9Batch`, `init_batch_round9`, `load_round9` / `round9_rows` over
  `round9/rotation_v9_manifest.json`, the `next_lineup` and `init_batch`
  dispatch, and parity tests against `rotation_v9.run_wave9` on a fixed seed.

That is 30-45 minutes of work including the parity tests and did not fit after
grading finished at 12:43 ET against a 13:10 ET hard stop. **Condition 6 is
unmet for every arm**, so no round-7, -8 or -9 arm may be served whatever it
grades offline. It costs nothing on this round's outcome -- Z1 already fails
conditions 1, 4 and 5 by 2 to 58 floors -- and it is reported, not hidden.

### 21.16 Floor B (20.10)

Run after the bake-off, on `Z1_exit_marg`: a different training-game sample (fit
seed 101 against 11) and a different sim seed (23 against 7), graded on the same
150-game universe, in its own process.

| cell | ACTUAL (150 games) | Z1 seed 1 | Z1 seed 2 | \|delta\| |
|---|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7436 | 0.5947 | 0.6234 | 2.87 pp |
| final 8:00, \|m\| 6-15 | 0.7186 | 0.5828 | 0.5883 | 0.55 pp |
| final 8:00, \|m\| > 15 | 0.5518 | 0.5290 | 0.4876 | 4.14 pp |
| starters at >= 4 fouls | 0.4362 | 0.4309 | 0.4193 | 1.16 pp |
| H2 tip, \|m\| <= 5 | 0.9655 | 0.9345 | 0.9638 | 2.93 pp |
| H2 tip, \|m\| 6-15 | 0.9662 | 0.9270 | 0.9432 | 1.62 pp |
| H2 tip, \|m\| > 15 | 0.9611 | 0.9500 | 0.9500 | 0.00 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0.7710 | 0.7093 | 0.7160 | 0.68 pp |
| substitutions per boundary | -- | 0.1502 | 0.1565 | 0.0063 |
| distinct lineups per team-game | -- | 15.197 | 15.977 | 0.780 |
| per-player minutes MAE | -- | 23.0852 | 23.0634 | 0.0218 |

**The round-9 objects are identified.** The fitted table moves **0.9 pp at one
starter on the floor and 0.4 pp at four** under the refit (window 202411,
`P(k_out = 1 | size 1, n_st)`: the marginal `M1` 0.3227 -> 0.3314 at one starter
and 0.6574 -> 0.6539 at four; Z1 0.3076 -> 0.3201 and 0.6715 -> 0.6702), and the
under-300 cell count moves 63 -> 65 of 108. The cell spread (0.00-4.14 pp) is
rounds 7's (0.29-6.31 pp) and 8's (0.43-4.77 pp) on the same thin universe.

**The decision does not turn on a refit artefact in either direction**: the
close-band miss Z1 fails on is -14.7 pp, **five times** the largest
refit-to-refit move on that cell (2.87 pp) and 9.6 floor-A SDs, and the axis
repair 21.13 item 1 claims is 15 pp at one starter against a 1.25 pp refit move.
`rotation_F1_round9_floorB_Z1.json`'s `minutes_mae` level (23.1) is the
150-game-universe artifact rounds 5-8 carry for the same reason (11.6): only the
seed-to-seed and refit-to-refit DIFFERENCES are readable, never the level.

Artifacts: `round9/rotation_v9_exit_seed2_{YYYYMM}.json` +
`rotation_v9_manifest_seed2.json` (new versioned siblings; nothing was
overwritten), `rotation_F1_round9_floorB_Z1.json`.

### 21.17 Disclosures

1. **A smoke run WAS executed** before the graded run (60 games, 1 seed,
   `--wave-team-games 400`, tag `F1-round9-SMOKE`), to validate the trainer's
   wiring end to end; it wrote `rotation_F1_round9_SMOKE_results.json` and
   round-9 exit artifacts at smoke size which the graded run then refitted and
   replaced. **No model specification, gate, tolerance, arm, grid or constant was
   changed after any number was seen**, and section 20 was committed at 3525996
   before `rotation_v9.py` existed. The smoke's own `k` selection (30, on 80
   team-games per fold) is not the selection used; the graded run refit it on the
   full 2024 training season (21.1).
2. The reference columns for R2 and K1 are round 6's, X1's is round 7's and Y1's
   is round 8's (20.6), with the 1-seed Y1 reproduction check of 21.7.
3. No static column was run (20.6).
4. `k = 30` is the argmax of a declared grid whose LEFT BOUNDARY won; the
   monotonicity of the held-out score across the whole grid is published in 21.1
   and the caveat is carried into 21.13 item 5. The grid was not extended after
   the result.
5. Z1 and Z2 draw the same uniforms as X1 and Y1 in the same order and are
   byte-aligned with them (20.13 item 1), because `run_wave9` delegates to
   `rotation_v8.run_wave8` through a table shim and changes no draw.
6. Z1's and Z2's support is published in 21.2: 61-63 of 108 size-1 cells carry
   under 300 rows -- the SAME cells Y1 had. 20.3 made neither arm conditional on
   support, since the round changes only what those cells fall back on.
7. The exit-side and by-composition diagnostics are measured on 200 games at
   seed 0, the configuration of the round-6, -7 and -8 audits, and are
   report-only. The ACTUAL row at one starter on the floor carries 244 leavers
   and is labelled UNDERPOWERED; 17.17's full-season measurement of the same
   quantity (4,072-5,981 leavers) is the powered version of that cell.
8. **Decision 10 was NOT RUN** (21.15). No closed-loop claim is made for any
   round-9 arm.
9. No process this worker did not start was signalled, and no existing data or
   results file was overwritten: every round-9 artifact is a new versioned
   sibling under `round9/` or a new `_round9_` stem.

---


## 22. The round-9 engine adapter and the Decision-10 freeze on K1 and Z1 (run 2026-09-11T17:04-17:16Z)

21.15 reported that the adapter had not been written for the THIRD round running
and that condition 6 was therefore unmet for every round-7, -8 and -9 arm. This
section closes that gap: the adapter is written to 21.15's own scope, the two
bit-identity checks it owes the served engine are run, it is checked against the
offline sampler, and the Decision-10 freeze is run on **K1** (the round-6 arm
closest to eligibility, re-run so the two arms are one comparison) and on **Z1**
(round 9's closest arm). **No default moved and no arm was adopted.** Z1 still
fails conditions 1, 4 and 5 offline by 2 to 58 floors (21.12); passing the freeze
changes none of that, and the freeze is reported here as the instrument it is,
not as a result in Z1's favour.

### 22.1 What was written

`ENGINE_ROTATION=round9`, default-off, in `src/cbb_sim/engine/rotation_adapter.py`:
`Round9Batch`, `init_batch_round9`, `next_lineup_round9`, `load_round9`,
`round9_rows`, `round9_arm`, and the `next_lineup` / `init_batch` dispatch
(`Round9Batch` is tested BEFORE `Round6Batch`, which it subclasses -- a test
asserts the order, because the reverse silently turns every round-9 run into a
round-6 run). `engine/loop.py` gains the one `load_round9` line round 6 has.
`ENGINE_ROTATION_ARM=Z1|Z2` selects the table. `adapters.py` `DEFAULTS` was not
touched; the runner keeps round 6's stated hack of loading the adapters as
`reference` and setting the mode in-worker.

`next_lineup_round9` is `next_lineup_round6`'s K1 path with the exit block
replaced, exactly as scoped:

* `ce = V7.exit_cell(period, seconds_remaining, margin, foul_state)` and
  `n_st = (on & is_starter).sum(axis=1)`, both off the round-6 batch;
* `row = zexit[seg, clip(size,1,5)-1, ce, min(n_st, N_ST-1)]`, support clipped to
  `[max(0, size - n_bench_on, forced_starters), min(size, n_starters_on,
  size - forced_bench)]` with round 8's `hi < lo` collapse, renormalised,
  `k_out` drawn;
* `leaving = _pick_k(on & is_st, key_out, k_out) | _pick_k(on & ~is_st, key_out,
  size - k_out)` with round 5's `key_out`;
* round 6's K1 entry block unchanged, reading `k_out` off `leaving` as it does.

**Stated RNG divergence**, declared before any number was read and exactly as
round 6 declared one from round 5 (14.11): the `rotation_sub` draw block is
`2S + 5` wide, not `2S + 4`, and the new scalar at `2S + 4` is the `k_out`
uniform. **Round-9 arms are paired with each other and NOT with round 6 or
round 5.** The K1 freeze pair below is internally paired on round 6's own
`2S + 4` block, so the two pairs are each internally valid and are not read
against one another.

A default-off audit hook (`ENGINE_ROT9_AUDIT=1`, `RA.ROUND9_AUDIT`) records every
wave the exit block resolves. It exists for 22.3 and costs one environment
lookup per possession when off.

### 22.2 The two bit-identity checks the served engine is owed

| check | method | result |
|---|---|---|
| served R2 path (`reference`) | 60 games x 5 seeds, `ENGINE_CLOCK=v5b_glat_pmean` pinned (the served clock, read out of `results/engine_v0/smoke60x5_default_v5b/run_meta.json`), `scripts/digest_engine_run.py --compare` | **PASS**, sha256 `492300a7...51a1` identical to `smoke60x5_default_v5b` |
| existing round-6 K1 path | a 30-game x 3-seed `round6`/`K1` chunk run under the HEAD tree and under the working tree, both frames hashed at 6 dp | **PASS**, `fd3ec157...f0ee` on both, 12,896 possessions |

The round-6 K1 path is checked a second time, at full scale and by accident of
design: `K1_live` and `K1_frozen` below reproduce 15.13's published K1 table to
every digit printed there (16.5337 / 16.5367, 0.0361 / 0.0399, 71.7092 /
71.6818, 8.7390 / 8.6491). A 500-game x 5-seed round-6 run is byte-for-byte what
it was before the round-9 code existed.

### 22.3 Parity against the offline sampler

The only thing that differs between the adapter and `rotation_v9.run_wave9` is
the exit draw, so that is what is checked, on real engine states rather than
synthetic ones: `ENGINE_ROT9_AUDIT=1` over **150 games x 3 seeds** (450
team-game sims, 64,896 possessions) recorded **21,506 resolved waves** with their
`(size, n_st_on, n_bn_on, forced_st, forced_bn, exit_cell, seg, u_x, k_out)`, and
`k_out` was recomputed from the same uniforms and the same fitted tables by the
OFFLINE scalar block in `rotation_v8.run_wave8` that `run_wave9` delegates to.

**Floor A for an identity check is 0**, and that is what is required here: the
two are the same sampler or they are not.

| n_starters on floor | waves | adapter starter share | offline starter share | \|delta\| |
|---:|---:|---:|---:|---:|
| 0 | 201 | 0.000000 | 0.000000 | 0.000000 (UNDERPOWERED, 201 waves) |
| 1 | 1,457 | 0.224374 | 0.224374 | 0.000000 |
| 2 | 4,588 | 0.350178 | 0.350178 | 0.000000 |
| 3 | 7,267 | 0.502056 | 0.502056 | 0.000000 |
| 4 | 5,728 | 0.728523 | 0.728523 | 0.000000 |
| 5 | 2,265 | 1.000000 | 1.000000 | 0.000000 |

**0 of 21,506 `k_out` draws differ.** The share slopes monotonically with
`n_starters` (0.224 -> 1.000), which is the responsiveness the exit side is
supposed to carry; that it slopes identically on both sides is the parity claim,
not a separate finding. Artifact:
`data/processed/models/rotation/round9_adapter_parity_2026-09-11.json`.

A 20,000-row randomised-state equality test over the same two blocks, including
the all-zero-row and degenerate-support branches, is in
`tests/test_rotation_round9.py`; `pytest tests/test_rotation_round9.py` 8 passed,
`pytest tests/test_engine.py` 20 passed.

### 22.4 The freeze (Decision 10)

Paired-stream runs on 14.9's fixed **500-game subset** of F2 2025 (sorted by
`game_id` ascending, every 11th row, the first 500), **5 seeds**, 3 workers per
arm, with 14.9's pinned sub-models (`ENGINE_EVENT=round2_s1`,
`ENGINE_FG_MAKE=round3_shooter_S_C_s1`, `ENGINE_CLOCK=reference`) recorded in
every `run_meta.json`. `ENGINE_ROTATION_FREEZE=1` holds margin at 0 and the
personal and team foul counts at 0 **for the rotation model only**; foul
accrual, the foul-out eviction and the box-score counters stay live. Runs:
`results/engine_v0/dec10_rot_20260911/{K1,Z1}_{live,frozen}`.

| run | seeds | margin SD | per-game margin SD | home/away corr | possessions | total | per-player minutes MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| K1 live | 5 | 16.5337 | 12.1434 | 0.0361 | 71.7092 | 149.367 | 8.7390 |
| K1 `FREEZE=1` | 5 | 16.5367 | 12.2718 | 0.0399 | 71.6818 | 149.516 | 8.6491 |
| Z1 live | 5 | 16.3696 | 11.9869 | 0.0404 | 71.7064 | 149.088 | 9.2922 |
| Z1 `FREEZE=1` | 5 | 16.2635 | 12.0815 | 0.0567 | 71.7038 | 149.162 | 9.3511 |

| comparison | margin SD ratio | corr delta | possessions delta | floor | verdict |
|---|---:|---:|---:|---|---|
| K1 live / frozen | **0.9998** | -0.0038 | +0.027 | SD ratio in [0.95, 1.05]; possessions within G1 mean +/- 1.0 | **PASS** |
| Z1 live / frozen | **1.0065** | -0.0163 | +0.003 | same | **PASS** |
| (15.13) K1, 5 seeds | 0.9998 | -0.0038 | +0.027 | -- | PASS, reproduced exactly |
| (15.13) K1, 25 seeds | 0.9934 | +0.0099 | -0.016 | -- | PASS (quoted, not re-run) |
| (13.12) W4, 5 seeds | 0.9948 | -0.0132 | -0.087 | -- | PASS (quoted) |

Z1's freeze moves margin SD by 0.65%, possessions by 0.003 against a G1
tolerance of 1.0, and the home/away correlation by 0.016 -- larger on the
correlation than K1's 0.004, smaller than round 5's W4 -0.0132 is from zero and
the same sign. 14.9 pins the margin-SD and possession floors numerically and
states the correlation floor only as "inside the G1/G2 tolerances"; **no numeric
correlation band is pre-registered**, so -0.0163 is reported against the
-0.0132 / -0.0038 / +0.0099 the three prior passing pairs produced rather than
against a threshold invented now.

**25 seeds were not run.** A 500-game x 25-seed pair costs about 22 minutes per
run on the 3 workers each arm was capped at, so four runs would have cost about
90 minutes against a 13:35 ET stop that the 5-seed set cleared at 13:16. Its
absence is reported, not hidden; 15.13's 25-seed K1 pair stands as the only
25-seed read on this family.

### 22.5 What the freeze does and does not license

**Condition 6 is now RUN for K1 and for Z1, and both pass.** It is not met for
X1, Y1, Z2 or any other round-7/8/9 arm, which were not run.

**Nothing is adopted.** Z1 still fails condition 1 (four of eight state cells),
condition 4 (-0.451 minutes of MAE to K1 offline, every player quintile to both
K1 and W4) and condition 5 (Decision 8 slope ratio 0.51 against a [0.8, 1.2]
band) by 2 to 58 floors (21.12). A closed-loop pass removes an obstacle; it
supplies no evidence for the arm. `ENGINE_ROTATION=reference` (R2) remains the
served default, `round9` ships default-off, and **this lane changed no default.**

The engine reproduces the offline ordering on its own player minutes, which is
the multi-level cross-check the freeze table is worth reading for: on the same
500 games with the same pinned sub-models, K1 8.7390 against Z1 9.2922, a
0.553-minute gap in the same direction and about the same size as the offline
8.8622 against 9.3132 (0.451). The engine and the offline sampler agree that
round 9's exit rule is WORSE than round 6's, independently.

### 22.6 Artifacts and disclosures

* Adapter: `src/cbb_sim/engine/rotation_adapter.py` (round-9 section),
  `src/cbb_sim/engine/loop.py` (one line), commit `3923f5f`.
* Runner: `scripts/run_rot9_closed_loop.py` (a copy of
  `scripts/run_rot6_closed_loop.py` with the `round9` arm added; the round-6
  runner was NOT edited, so a round-6 run another lane started cannot change
  because round 9 exists).
* Parity: `scripts/diag_rot9_adapter_parity.py`,
  `data/processed/models/rotation/round9_adapter_parity_2026-09-11.json`.
* Tests: `tests/test_rotation_round9.py` (8), `tests/test_engine.py` (20).
* Runs: `results/engine_v0/dec10_rot_20260911/` (four runs, 358.4-358.5k
  possessions each, 266-409 s) and `results/engine_v0/smoke60x5_rot9adapter`.
* Write-up: `docs/tests/rotation_adapter_freeze_2026-09-11.md`.
* **Disclosure.** The freeze pairs are 5 seeds, the minimum Decision 10 allows
  for an SD ratio and far below the 200 it asks for a final read. They detect a
  new state channel; they do not size one (L31), and no magnitude is quoted from
  them.
* **Disclosure.** The Z1 pair is internally paired on round 9's `2S + 5` stream
  and the K1 pair on round 6's `2S + 4` stream. The two pairs' LEVELS are
  therefore not comparable stream-for-stream; only the within-pair ratios are
  read as freeze evidence. The minutes-MAE comparison in 22.5 is across streams
  and is reported as a directional cross-check, not as a paired measurement.
* **Disclosure.** Round 9's exit tables were fitted on fold F1 (train 2024, test
  2025) and are served here through the same S1 manifest round 6 uses; the
  manifest's segment alignment against round 6 is asserted in `load_round9` and
  passed on all 6 windows.
* **Disclosure.** The bit-identity smoke ran one extra single-worker process
  alongside this lane's 6 engine workers for about 100 seconds.


## 23. Round 10 pre-registration -- the WAVE side, `P(wave | cell, n_st)` and `P(size | cell, n_st)` (PM-directed, worker-authored 2026-09-18)

Written and committed BEFORE `rotation_v10.py` existed and before anything was
fitted. Evidence it is built on: round 9's own results (section 21), the
Decision-10 freeze of section 22, and the support measurement 23.1 below, which
was run BEFORE this pre-registration exactly as 17.17 was run before round 8 and
19.1 before round 9.

### 23.1 The support measurement: the composition gap, decomposed

`scripts/diag_rotation_wave_v10.py`, **400 games of the 2025 test season, seed
0**, the as-of predicted starter set on BOTH sides, ONE function applied to the
ACTUAL on-floor sequence and to each arm's simulated sequence, the convention of
16.7, 17.17, 19.1 and 21.8. Artifact:
`data/processed/models/rotation/wave_support_round10_2026-09-18.json`.

**The kernel, stated once.** At a possession boundary the number of the model's
own predicted starters on the floor moves by `n_st -> n_st - k_out + k_in` and by
nothing else. FOUR objects decide it and they are the only four:

    A(n_st)                      P(a wave happens here | n_st)        ARRIVAL
    S(size | n_st)                                                    SIZE
    O(k_out | size, n_st)                                             EXIT
    I(k_in | size, k_out, n_st)                                       ENTRY

plus the hard second-half reset, counted at its realised per-boundary rate
(0.00746 on both sides). Round 9 repaired **O** and nothing else.

**Measured, 105,410 actual boundaries (1,655-32,129 per `n_st` level, none
underpowered):**

| `n_st` | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| actual boundaries | 1,655 | 4,524 | 16,402 | 32,129 | 31,018 | 20,682 |
| **A actual** | 0.0979 | 0.1746 | **0.1822** | 0.1674 | 0.1390 | **0.0784** |
| **A, Z1** | 0.1749 | 0.1615 | 0.1580 | 0.1577 | 0.1460 | **0.1235** |
| A, K1 | 0.1837 | 0.1497 | 0.1571 | 0.1535 | 0.1494 | 0.1286 |
| **mean size actual** | 1.654 | **1.696** | 1.518 | 1.417 | **1.328** | 1.449 |
| **mean size, Z1** | 1.474 | 1.408 | 1.411 | 1.397 | 1.412 | 1.418 |
| mean `k_out` actual | 0.000 | 0.379 | 0.583 | 0.754 | 0.990 | 1.449 |
| mean `k_out`, Z1 | 0.000 | 0.345 | 0.514 | 0.715 | 1.034 | 1.418 |
| mean `k_in` actual | 0.685 | 1.308 | 1.045 | 0.791 | 0.426 | 0.000 |
| mean `k_in`, Z1 | 1.263 | 1.113 | 0.943 | 0.700 | 0.368 | 0.000 |
| **drift actual** | +0.685 | +0.929 | +0.462 | +0.037 | -0.564 | -1.449 |
| **drift, Z1** | +1.263 | +0.768 | +0.429 | -0.015 | -0.665 | -1.418 |
| occupancy actual | 0.0164 | 0.0428 | 0.1540 | 0.3022 | 0.2917 | 0.1929 |
| occupancy, Z1 | 0.0082 | 0.0675 | 0.2032 | 0.3093 | 0.2812 | 0.1305 |

Four facts, in the order they matter:

1. **The per-wave DRIFT is already right.** The actual fixed point (drift = 0)
   sits at `n_st` 3.07 and Z1's at 2.97. Round 9 did its job: given that a wave
   happens and given its size, the net starter flow is correct to a tenth of a
   man at every level.
2. **The ARRIVAL rate is flat and it should be an inverted U.** Actual runs
   0.098 / 0.175 / 0.182 / 0.167 / 0.139 / **0.078**, a 10.4 pp fall from
   `n_st = 2` to `n_st = 5`. Z1 runs 0.175 / 0.162 / 0.158 / 0.158 / 0.146 /
   **0.124**, a 3.5 pp fall -- **33% of the real span**. At five starters on the
   floor Z1 breaks the lineup up **58% too often** (0.1235 against 0.0784), and
   at zero starters it churns **79% too often** (0.1749 against 0.0979). Both
   errors push mass out of the tails and into the bench-heavy middle.
3. **The SIZE is flat and it should slope.** Actual mean wave size falls
   1.696 -> 1.328 from `n_st = 1` to `n_st = 4`; Z1 sits at 1.408 -> 1.412, **1%
   of the span and the wrong sign**. Real basketball makes a BIGGER wave when the
   floor is bench-heavy -- a restoration -- and the sim makes the same 1.41-man
   wave everywhere.
4. **The ENTRY slope is slightly too steep, not too shallow.** Starters per
   entrant runs actual 0.771 / 0.688 / 0.558 / 0.321 at `n_st` 1-4 against Z1's
   0.790 / 0.668 / 0.501 / 0.261.

**The decomposition.** The four objects plus the reset define a Markov chain on
`n_st`. It is validated FIRST: the chain's stationary distribution understates
the observed mean `n_st` by 0.12-0.17 on all three sides in the same direction
(ACTUAL 3.2204 chain against 3.3888 observed, K1 3.2651/3.3803, Z1 3.0279/3.1793)
because it ignores the within-game correlation between the floor and availability
and foul-outs, but it reproduces **92% of the observed Z1-to-actual gap** (0.1925
against 0.2095), so DIFFERENCES may be read from it and LEVELS may not. That
caveat is stated here, before the table, and not in a footnote.

One component at a time is swapped between the simulated and the actual kernel,
in BOTH directions; the share of the gap each closes is published both ways and
neither is averaged away:

| component | share of the mean-`n_st` gap, in / out | share of the `P(n_st <= 2)` gap, in / out |
|---|---:|---:|
| **A arrival** | **+0.701 / +0.590** | **+0.485 / +0.355** |
| **S size** | **+0.415 / +0.427** | **+0.453 / +0.406** |
| **I entry** | +0.322 / +0.188 | **+0.486 / +0.331** |
| O exit | **-0.285 / -0.365** | -0.257 / -0.260 |
| reset | +0.000 / +0.000 | -0.000 / -0.000 |
| (sum) | 1.153 / 0.840 | 1.167 / 0.832 |

**Arrival is the largest single term, size is second, entry is third, and the
exit side now works AGAINST the gap** -- Z1's `O` is slightly over-corrected, so
importing the real exit kernel makes the floor MORE bench-heavy, not less. Round
9's own diagnosis (21.13 item 3) named the wave side from a count; this measures
it and orders it.

**By game state** (3 time cells x 3 margin bands, 3,010-31,490 actual boundaries
per cell, none underpowered): the arrival error is not uniform. In H1 the
simulated and actual rates agree to 0.3-0.6 pp; in **H2 20:00-08:00 at |m| 6-15**
Z1 is 0.1477 against 0.1309 (+12.8% relative) and in the **final 8:00** it is
0.1677 / 0.1646 / 0.1550 against 0.1515 / 0.1428 / 0.1432 (+10.7% / +15.3% /
+8.2%). The excess arrival is concentrated in the second half and late -- the
same place the four failing state cells live.

**Per team** (139 teams powered at >= 300 boundaries on both sides, **191
UNDERPOWERED and excluded**): mean `n_st` sim 3.199 against a real 3.405,
cross-team SD 0.239 against 0.396, corr 0.258 and mean |dev| 0.374 for Z1,
against K1's 3.380 / 0.319 / **0.522** / 0.281. And the arrival rate carries **no
team information at all**: the per-team correlation between the simulated and the
real `P(wave)` is **-0.038 for K1 and +0.136 for Z1**. That is 21.13 item 4 seen
on a second object, and it is recorded here before the round runs.

### 23.2 The one thing round 10 changes

Round 10 keeps round 3b's base fits, round 4's hazards, round 6's K1 entry rule,
round 9's Z1 exit rule, the hard second-half reset and every clip and uniform
BYTE FOR BYTE, and changes only **which table the wave Bernoulli and the
wave-size categorical are drawn from**:

    round 5   p_wave[cell]              p_size[cell, .]
    round 10  p_wave[cell, n_st]        p_size[cell, n_st, .]

`cell` is round 5's own `wave_cell` (prev_end x 9 time cells x 3 margin bands x
2 foul states, 324 cells), unchanged and not re-specified here. `n_st` is the
quantity 18.2 declared and rounds 8 and 9 used, unchanged: the number of the
model's own PREDICTED starting five among the five on the floor BEFORE the swap,
`is_st[on_idx].sum()`, six levels 0..5, identical offline and in the sampler.

**No uniform is added, removed or reordered.** `run_wave10` is `run_wave8`'s loop
with `n_st` computed before the wave draw instead of after it and the table
gathers widened by one index; the draw order stays round 5's coupling draw, the
wave draw, the size draw, the `k_out` uniform, K1's `k_in` uniform, the bench race
vector. **Round-10 arms are therefore byte-aligned with round 9's arms, with
round 8's and with each other**, and the claim is not asserted: 23.9 pre-registers
a bit-exact identity test of `run_wave10` fed with round 5's and round 6's tables
broadcast over the `n_st` axis against `run_wave9`'s own output.

### 23.3 The counts pass, and why a CONTROL arm is required

Round 5's wave counts (`build_wave_training`) carry no starter concept, so they
cannot be reused: the round-10 objects need a pass that knows the as-of predicted
five. `build_wave10_training` is round 8's `build_exit8_training` widened to count
EVERY boundary, not only the wave boundaries, on the SAME key list, the SAME
as-of candidate pool (`build_priors`), the SAME `--wave-team-games 6000` sample
and the SAME fit seed 11 that rounds 6, 7, 8 and 9 used. One pass produces all
three count objects: `ev[cell, n_st]` / `n[cell, n_st]`, `size[cell, n_st, s]`
and `kin[size, k_out, n_st, k_in]`.

Those rows are not byte-identical to round 5's, so a movement between Z1 and a
round-10 arm could in principle be the ROWS rather than the AXIS. **Arm V0 exists
to close that hole** (the L31 refit-without-the-feature pattern, the same
instrument `fit_wave(collapse_state=True)` provides): the round-5 objects are
refitted on the round-10 pass's own rows with the `n_st` axis marginalised out
BEFORE the shrinkage -- the maximum-likelihood fit of the axis-free model on the
same rows, not an ablation of a fitted number. Any difference between V0 and Z1
is the rows; any difference between V1/V2/V3 and V0 is the axis.

### 23.4 The shrinkage parent, and the two variants the round exists to compare

Both parents are powered here, which is new. There are 324 cells and six `n_st`
levels; the cell marginal `P0(cell)` is round 5's served, proven object and the
composition marginal `M(n_st)` carries 1,655-32,129 rows a season at the levels
23.1 measures. 21.13 item 5's lesson is that the parent of a thin cell must be the
nearly-right object; with two powered marginals the nearly-right object is the one
built from BOTH.

**The declared parent for V1, V2, V3 and V5 -- the PRODUCT parent, log-odds
additive in the two powered marginals:**

    Bernoulli:    logit Ppar(cell, n_st) = logit P0(cell) + logit M(n_st)
                                           - logit root
    categorical:  Ppar(cell, n_st, .) proportional to
                  P0size(cell, .) * Msize(n_st, .) / rootsize(.), renormalised

i.e. the no-interaction log-linear model on the two marginals, which nests each
marginal exactly when the other is flat. The (cell, n_st) cell's own counts are
then shrunk to it at the fitted `k` of 23.5, support clipped and renormalised as
in rounds 5 and 7-9.

**V4 is the same objects with the parent that rounds 5-8 used -- the cell LEVEL
`P0(cell)` alone.** It is the contrast 21.13 item 5 asks for and it is the only
difference between V4 and V3.

The third possible parent, the composition marginal `M(n_st)` ALONE, is **not run,
and the reason is stated before the result**: it would discard round 5's own
324-cell arrival structure in every thin cell, and the product parent nests it (it
is the product parent with `P0(cell) = root`). If V3 loses to V4, that choice is
wrong and the round says so.

Marginals `P0` and `M` are themselves shrunk: `P0` by round 5's own three-level
scheme (cell -> prev_end x time cell -> prev_end -> root) at the fitted `k`, `M`
one level to the root at the fitted `k`.

### 23.5 The shrinkage constants are FITTED, not chosen -- the method, declared here

21.13 item 5 recorded that round 9's grid argmax landed on the grid's left boundary
and that a later round reviving the family should widen the grid downward and
refit rather than inherit 30. Round 10 does exactly that, by round 9's own method
(20.4), on THREE objects independently:

* **Data: the 2024 training season only**, which precedes the whole F1 2025 test
  season. No test row and no gate cell enters the choice.
* **Procedure: leave-one-fold-out over 5 disjoint folds** of the 2024 training
  team-games (`numpy.RandomState(11)` assigns folds); each fold's table is fitted
  from the other four folds' counts and scored on the left-out fold's counts.
* **Criterion: held-out log-likelihood per held-out row** -- Bernoulli for the
  arrival object, multinomial for the size and entry objects -- of the level-2
  table the arm defines.
* **Grid, declared here and not extended after any number is seen:
  `k in {3, 10, 30, 100, 300, 1000, 3000}`.** The argmax is taken; ties within
  0.001 nats per held-out row go to the LARGER `k` (more shrinkage, simpler).
* `k_wave`, `k_size` and `k_kin` are selected independently, published with their
  full grids, used for every window and for every arm that carries that object.
* **V0's own `k`s are fitted by the same criterion on the same folds under the
  axis-free model**, so V0 is that model's maximum-likelihood fit and not V3's
  constants borrowed.

If any argmax lands on a grid boundary again it is reported as such, in the results
table, with the same caveat 21.1 carried.

### 23.6 Arms

| arm | arrival table | size table | entry table | exit rule | simplicity | status |
|---|---|---|---|---|---:|---|
| `R2_hier_dirichlet` | -- | -- | -- | -- | 1 | reference (incumbent, SERVED) |
| `K1_cond_class` | round 5 cell | round 5 cell | round 6 `(size, k_out)` | round 5 rank | 9 | reference (best MAE in nine rounds; NOT adoptable) |
| `Z1_exit_marg` | round 5 cell | round 5 cell | round 6 | round 9 Z1 | 16 | reference (round 9's closest; NOT adoptable) |
| `V0_wave_refit` | **refit, no axis** | **refit, no axis** | round 6 | round 9 Z1 | 17 | **CONTROL** (23.3; NOT adoptable) |
| `V1_wave_arrival` | **(cell, n_st)**, product parent | round 5 cell | round 6 | round 9 Z1 | 18 | candidate |
| `V2_wave_size` | round 5 cell | **(cell, n_st)**, product parent | round 6 | round 9 Z1 | 19 | candidate |
| `V3_wave_both` | **(cell, n_st)** | **(cell, n_st)** | round 6 | round 9 Z1 | 20 | candidate |
| `V4_wave_both_lvl` | (cell, n_st), **cell-LEVEL parent** | (cell, n_st), **cell-LEVEL parent** | round 6 | round 9 Z1 | 21 | candidate |
| `V5_wave_entry` | (cell, n_st) | (cell, n_st) | **`(size, k_out, n_st)`**, product parent | round 9 Z1 | 22 | candidate |

The simplicity order `R2 < K1 < Z1 < V0 < V1 < V2 < V3 < V4 < V5` is fixed HERE.
V4 carries the same number of fitted numbers as V3 and is declared the LESS simple
of the two deliberately: the product parent is the measured lesson of round 9, so
the level-parent variant is the deviation and carries the tie-break burden.

**Why no arm is built on K1's rank exit rule.** The wave fix applied to round 6's
exit rule would sit a repaired stage on top of a stage measured at 3% of the real
composition span (21.8), which the bottom-up standing rule forbids -- "never accept
a downstream stage that compensates for a known upstream bias". It is named here as
a possible later round, not run, and not run for a reason stated before the result
rather than after it.

**R2, K1, Z1 and V0 are references or controls and NONE is adoptable in round 10.**

### 23.7 Scheme, folds, and what is refitted

**Scheme: S1 for every arm**, per rounds 3b-9. Windows are the calendar months of
the 2024-25 season; the first window trains on 2024 alone. No static column.

**Folds.** F1 = train 2024, test 2025, which IS the standing fold 2 (L13). 2026
stays sealed (`seal.assert_not_sealed` guards the trainer). Fold 2 selects.

**Round 10 fits the three wave-side objects per window and nothing else.**
`rotation_fit_v3*.json`, `rotation_v4_sub_*.json`, `round5/*`, `round6/*`,
`round7/*`, `round8/*` and `round9/*` are REUSED and nothing is written to any of
them; the round-10 artifacts are new versioned siblings under `round10/`.

**The three reference columns (R2, K1, Z1) are taken from the round-6 and round-9
results JSONs**, not re-simulated: same 1,600-game universe, same subset seed 2025,
same sim seeds 0-2, same base fits, hazards, composition and exit tables and
grading functions. A **1-seed re-run of Z1 is executed inside this round as a
reproduction check** and its cells are reported next to round 9's; **if any state
cell moves by more than its floor-A SD the reference columns are discarded and the
round is re-run in full**, as 20.6 declared for Y1, 18.5 for X1 and 16.5 for K1.

### 23.8 Test universe and grading path

The **same** 1,600-game subset of 2025 rounds 2-9 used (numpy RandomState seed
2025), **3 seeds per candidate arm** under S1, the **round-9 grading path
unchanged** (`train_rotation_v1.build_row` / `verdict` / `rotation.aggregate_stats`,
extended by `train_rotation_v4.extra_cells` / `.minutes_mae` and
`train_rotation_v5.wave_cells`, with round 6's `quintile_mae`). No gate cell is
added, so no grader line changes and the reference columns stay comparable byte for
byte. Any cell with n < 300 player-games, possessions, boundaries or fitted rows is
labelled UNDERPOWERED and never read as signal or as absence of signal.

### 23.9 Gates -- every round-9 gate, unchanged, nothing added or relaxed

*G8 cells (report, not veto):* minutes mean +/- 2.0; minutes SD ratio pooled and
within-player 0.9-1.1; top-5 and top-8 share of team minutes +/- 2 pp; players with
> 0 minutes +/- 1.0.

*The eight state cells (the veto), each +/- 3 pp:* starters' share of on-floor slots
in the final 8:00 at |m| <= 5 / 6-15 / > 15; starters' share while carrying >= 4
fouls; the second-half TIP starter share in each of the three margin bands;
starters' share over H1 20:00-10:00 at |m| <= 5. **An arm missing ANY of the eight
is ineligible regardless of G8 or of MAE.**

*The two round-5 cells (also veto):* `sub_rate_per_boundary` +/- 0.015 or 3x the
floor-A seed SD if larger; `distinct_lineups_per_game` +/- 1.5 or 3x the floor-A
seed SD if larger. The governing number is named in the results table.

*Report only:* the "at exactly 4 fouls" diagnostic; top-1 / top-3 / top-5 five-man
lineup share; K-S D of the top-1 lineup share and of per-player minutes; mean wave
size; the as-of starter benchmark of 14.6 on all eight state cells; the exit-side
starter share of 16.7; the twelve-minute time-since-reset gradient of 17.10; the
realised exit rate by `n_starters_on_floor` (19.7 / 21.8's table), which must NOT
regress; and **the round's own objects, re-measured for every arm through
`scripts/diag_rotation_wave_v10.py` at 400 games and seed 0: `A(n_st)`, `S(n_st)`,
the floor occupancy of `n_st` with `P(n_st <= 2)` and mean `n_st` against the ACTUAL
0.2132 / 3.3888, the per-cell arrival table and the per-team arrival correlation.**

*A bit-exact identity test, floor 0:* `run_wave10` fed with round 5's `p_wave` and
`p_size` and round 6's `kin` broadcast over the `n_st` axis must reproduce
`run_wave9`'s Z1 lineup and foul arrays **element for element** on 30 games x 3
seeds. Two samplers are the same sampler or they are not. A failure stops the round.

### 23.10 Primary metric and the two responsiveness checks

**Primary metric: per-player minutes MAE**, unchanged from rounds 4-9.

**Responsiveness check 1 (Decision 8), unchanged from 20.9:** team-games bucketed
into quintiles of the pregame as-of share of team minutes going to the predicted
starting five; the close-and-late cell per quintile for ACTUAL and every arm, with
slope and Q5 - Q1. An arm whose slope ratio to actual falls outside **[0.8, 1.2]**,
or whose sign disagrees, is not adoptable. (ACTUAL +0.692; K1 0.83 PASS; Z1 0.51
FAIL.)

**A prediction is recorded here, in advance, so the round can falsify it** (the
20.13 item 6 pattern). 21.13 item 4 measured that a within-game restoring force does
not merely fail to carry between-team information -- it COMPRESSES the between-team
response. 23.1 measures that the arrival rate carries no team information either
(per-team corr -0.038 / +0.136). **The prediction is that V1-V5 move the Decision 8
slope ratio DOWN from Z1's 0.51, not up, and that the per-team correlation of the
close-band share falls from 0.190.** If they do, the round adopts nothing on
condition 5 and Decision 8 is confirmed as the blocking, separately-specified round
21.13 item 4 asked for.

**Responsiveness check 2, per-player minutes MAE by PLAYER quintile, carried forward
unchanged from 20.9.** Players in the as-of rotation set are bucketed into quintiles
of their own pregame as-of minutes per game. An arm is adoptable only if it beats K1
beyond the floor in the pooled MAE **and loses beyond the floor in no single
quintile to EITHER reference it is measured against -- K1 (rounds 6-9's base) or W4
(round 5's, whose Q2 column vetoed round 6's arms)**. Both columns are read from the
round-6 results JSON. Carrying W4 forward is deliberate: round 10 must not become
adoptable by dropping a column. Underpowered quintiles are labelled.

### 23.11 Noise floors and the decision rule

**Floor A, seed-varied sim runs:** 20 seeds x 150 games per candidate arm, the SD of
every G8 cell, every state cell, both round-5 cells and the MAE -- the rounds
3/4/5/6/7/8/9 configuration, so eight rounds' floors are comparable. R2's, K1's and
Z1's floors are rounds 4's, 6's and 9's and are unchanged by a run that does not
refit them.

**Floor B, spec-identical refit under a second seed:** the round-10 wave objects
refitted from a different training-game sample (fit seed 101 vs 11) and simulated
under a different sim seed (23 vs 7), graded on the same 150-game universe, run on
the arm the decision rule selects or, failing that, on **V3**, the arm the round is
built on. An arm counts as beating a reference on a cell only if its improvement
exceeds the refit-to-refit spread on that cell. **Floor B is pre-registered as
CONDITIONAL ON THE WALL CLOCK** and its absence, if it is absent, is reported and
not hidden.

**Decision rule.** Adopt the **simplest** arm that

1. passes **every** one of the eight state cells at +/- 3 pp, AND
2. passes **both** round-5 cells at the tolerances of 23.9, AND
3. beats `R2_hier_dirichlet` on per-player minutes MAE by more than the floor, AND
4. beats `K1_cond_class` on per-player minutes MAE by more than the floor and loses
   beyond the floor to neither K1 nor W4 in any player quintile (23.10), AND
5. satisfies the Decision 8 slope check, AND
6. passes the Decision 10 freeze of 23.12.

Ties go to the simpler model in the order of 23.6. An arm whose improvement on the
cell it was built to fix does not clear floor B is not adopted on that cell. **If no
arm is eligible, adopt nothing**, report which cell fails and by how much, name the
diagnosis, and name the next structure. No gate is relaxed to produce a winner and
no cell is dropped after seeing a result. **The served default is not changed by
this lane in any case**; the PM switches it.

### 23.12 Decision 10: the closed loop and the adapter

The round-10 wave objects carry a state term by design (round 5's own cell) exactly
as rounds 7-9's exit objects did, so the freeze is the right instrument and is
required before any arm is served. Paired-stream runs over the fixed **500-game
subset** of the F2 2025 slate (sorted by `game_id` ascending, every 11th row, the
first 500), with `ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round3_shooter_S_C_s1`,
`ENGINE_CLOCK=reference` pinned and recorded in every `run_meta.json`, reporting
margin SD ratio, home/away score correlation, possessions per game and per-player
minutes MAE, live against `ENGINE_ROTATION_FREEZE=1`, **5 seeds**, on the arm the
decision rule selects.

Section 22 wrote the `ENGINE_ROTATION=round9` adapter; round 10 extends it to
`ENGINE_ROTATION=round10` plus `ENGINE_ROTATION_ARM=V0|V1|V2|V3|V4|V5`,
**default-off**, with the `reference`, `round6` and `round9` paths required to stay
**bit-identical** (the two digest checks of 22.2, re-run). **The adapter and the
freeze are pre-registered as CONDITIONAL ON THE WALL CLOCK and as REQUIRED ONLY IF
AN ARM CLEARS CONDITIONS 1-5 OFFLINE.** If an arm passes 1-5 and the freeze cannot
be run, the arm is reported as **PASSES OFFLINE, FREEZE OUTSTANDING**, condition 6
is recorded as unmet, the arm is **not adopted**, and the served default is left
untouched. That is a reporting outcome, not a relaxed gate.

An arm that moves margin SD ratio, home/away correlation or possessions outside the
G1/G2 tolerances between live and frozen is not adopted, and no magnitude for the
loop is quoted from the freeze alone (L31).

### 23.13 Engine expressibility (a condition on adoption)

`p_wave[cell, n_st]` is (324, 6) and `p_size[cell, n_st, .]` is (324, 6, 5); V5's
entry table is (5, 6, 6, 6). `n_st` is `is_st[on_idx].sum()`, the integer rounds 8
and 9 already compute in the same block, so each object is ONE wider gather into a
table the adapter already holds, no operation is added to the (2N, S) roster block,
and the sim loop still makes no model call (`CLAUDE.md`). The only ordering change
is that `n_st` is computed before the wave draw instead of after it, which is a move
of an existing line.

### 23.14 Disclosures

1. **No uniform is added, removed or reordered** (23.2), so round-10 arms are
   byte-aligned with rounds 8's and 9's arms and with each other. The claim is
   tested bit-exactly, not asserted (23.9).
2. The counts pass is NEW (23.3) and its rows are not round 5's. **V0 is the control
   that makes every V1-V5 movement attributable to the axis rather than to the
   rows**, and it is a refit, not an ablation.
3. `k_wave`, `k_size` and `k_kin` are FITTED by 23.5's stated leave-one-fold-out
   criterion on 2024 training data over a grid declared before any fit; they never
   see a gate cell. The grid is widened downward because 21.13 item 5 said so, and
   it keeps 1000 and 3000 so the widening is not one-sided.
4. `n_st` is the model's own predicted five, not the game's; the as-of starter
   benchmark of 14.6 measures what that identification costs and is reported, as in
   every round since 4. 23.1's actual occupancy at `n_st` 0-1 (0.059) is partly that
   identification error and is not read as a rotation fact.
5. The support measurement of 23.1 was run BEFORE this pre-registration and its
   artifact is published with it. Its Markov chain reproduces differences and not
   levels, and that caveat governs every number taken from it.
6. Floor B (23.11), the Decision 10 freeze and the adapter (23.12) are conditional on
   the wall clock and their absence, if any, is reported.
7. 23.10 records in advance the prediction that every round-10 arm moves Decision 8
   the WRONG way, and the round is run so that the prediction can be falsified.

---
