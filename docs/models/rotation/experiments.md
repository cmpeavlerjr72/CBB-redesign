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
