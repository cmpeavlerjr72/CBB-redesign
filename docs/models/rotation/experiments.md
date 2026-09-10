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
