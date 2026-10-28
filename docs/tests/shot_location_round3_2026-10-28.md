# Shot-location round 3: replication of the Heuermann study plan (2026-10-28)

Facts only. Pre-registration: `docs/models/shot_location/experiments.md` "Round 3" (commit `34b6a66`, before any fit). Scripts: `scripts/build_shot_context_v1.py`, `scripts/train_shot_grade_v1.py`, `scripts/exp_shot_quality_rating_v1.py`, `scripts/exp_shot_quality_rating_tables_v1.py` (renders the tables below and applies the pass rule in code). Artifacts (gitignored): `data/processed/models/shot_location/round3/` (`build_report_*.json`, `grade_*.json`, `ep_*.parquet`, `rating_*.json`). Season id 2026 was never read. No adoption language: the PM decides. Bootstrap 500 resamples everywhere; every run took under 1 minute, so the 200-resample fallback was not used.

## 0. Verdicts under the registered rule (R3.7), computed in code

| arm | gain vs RAW_SHRUNK, fit 2023-24 / test 2024-25 | fit 2024-25 / test 2023-24 | verdict |
|---|---|---|---|
| QUALITY (L2 grade) | +1.49 floors | +1.42 floors | NOT CANDIDATE (needs > 2 floors in both directions) |
| CLASS (L0 grade) | -1.14 floors | +1.27 floors | NOT CANDIDATE |
| QUALITY minus CLASS (key secondary) | +3.60 floors | +0.81 floors | QUALITY ahead in both directions, beyond 2 floors in one |

Shrinking raw (RAW_SHRUNK) beats unshrunk RAW by +11.4 and +13.3 floors, i.e. nearly all of the partner's reported "quality beats raw" gain is available from shrinkage alone. The located-only sensitivity (section 6; not pre-registered) moves QUALITY to +1.49 / +2.17 floors, still below the rule in the first direction.

## 1. Facts about the data and the CBBD score convention

- CBBD `homeScore` / `awayScore` on a play row is the score AFTER the play. Check on de-duplicated files ordered by id: made FGA, total-score change from the previous row equals the shot value in 99.78% (2023-24) and 99.85% (2024-25) of cases; missed FGA, change is 0 in 99.96% / 99.90%; reading the next row instead matches 14.3% / 13.9%. The partner `margin - pts` is therefore correct on our files (build reports, `score_conv`).
- Date files overlap: 2,456,754 raw rows become 2,004,997 after de-duplicating on play id (2023-24); 2,657,828 become 2,190,197 (2024-25).
- Team name matches the games-file home or away team on 100% of FGA; games join 100%; `neutralSite` comes from the games file (about 10% of games neutral).
- D1 vs D1 (both games-file conferences present) and non-CBI: 657,189 FGA in 5,630 games (2023-24), 653,725 in 5,642 games (2024-25); 656,946 / 653,470 after the heave exclusion.
- Located share of D1 non-CBI FGA: 89.7% (2023-24), 98.7% (2024-25) (all FGA incl. non-D1: 87.5% / 98.6%).
- Fold sanity (located): threes inside 20.75 ft 0.129% / 0.121%; jumpers beyond 22.15 ft 0.216% / 0.154%.
- Heaves: located heaves excluded 243 / 255. Unlocated FGA with <= 3 s left (cannot be tested, stay in; upper bound on undetected heaves): 658 (2023-24), 62 (2024-25).
- Origin shares are near-identical across seasons (OPP_MAKE 39%, DRB 30-31%, ORB 13%, STEAL 8%, DEAD_TO 6.5%, UNKNOWN 0.6%).
- Class drift across seasons: TipShot is 0.11% of FGA in 2023-24 (732 shots in the scored set) and 1.33% in 2024-25 (8,678). A tip is a class in L0 and CLASS, so the class vocabulary is not stable across the two feeds.
- Grade level shift across seasons (mean per shot, scored season): points 1.0106 vs mean EP_L2 1.0209 (2023-24, grade from 2024-25); points 1.0176 vs EP_L2 1.0084 (2024-25, grade from 2023-24). Two-point jumpers: 0.7975 actual vs 0.8203 (L0) in 2023-24, 0.8194 vs 0.7980 in 2024-25. The ratings are league-relative, so a level shift cancels in the rating; it does not cancel in raw EP levels.

## 2. Part B: per-shot grade layers (spec R3.4)

HistGradientBoosting, partner hyperparameters, 656,946 / 653,470 test FGA, floors from 500 game-block resamples. "All" = every FGA (unlocated jumpers/threes carry NaN location); "Located" = located FGA only.

| direction | layer | log loss (all) | gain vs previous (all) | floor | floors (all) | floors (located only) |
|---|---|---|---|---|---|---|
| train 2023-24 / test 2024-25 | L0 | 0.65545 | - | - | - | - |
| | L1 (+location) | 0.65187 | +3.58e-3 | 1.20e-4 | **+29.9** | +29.5 |
| | L2 (+possession) | 0.65152 | +3.47e-4 | 1.17e-4 | **+2.97** | +2.63 |
| | D1 (L2 + home) | 0.65112 | +3.99e-4 | 2.96e-5 | +13.5 | +12.9 |
| | D2 (L2 + period, seconds left, margin) | 0.65032 | +1.20e-3 | 5.59e-5 | +21.5 | +21.4 |
| train 2024-25 / test 2023-24 | L0 | 0.65549 | - | - | - | - |
| | L1 (+location) | 0.65528 | +2.03e-4 | 1.51e-4 | **+1.35** | +21.9 |
| | L2 (+possession) | 0.65254 | +2.74e-3 | 8.63e-5 | **+31.8** | +28.1 |
| | D1 | 0.65218 | +3.67e-4 | 3.16e-5 | +11.6 | +6.9 |
| | D2 | 0.65166 | +8.80e-4 | 5.08e-5 | +17.3 | +14.7 |

Under the registered rule (> 2 floors in both directions, all-FGA metric): L1 FAILS (+29.9, +1.35); L2 passes (+2.97, +31.8; the first is marginal). The L1 failure in the second direction is the NaN shift: the model trained on 2024-25 (98.7% located) never saw a missing location, and 10.3% of 2023-24 test FGA are unlocated; the same layer on located rows is +21.9 floors. The L2 gain is very asymmetric across directions (+2.97 vs +31.8 floors; located-only +2.63 vs +28.1); no explanation was tested. D1 and D2 are diagnostics against L2: home and game state both move make rates beyond the floor in both directions (D2 gain is 2.4-3.0x the D1 gain).

Seed check of the grade models (random_state 1): L0 log loss 0.65545 / 0.65548 (seed 1) vs 0.65545 / 0.65549; L2 0.65181 vs 0.65152 (first direction), 0.65245 vs 0.65254 (second). Mean absolute EP difference between seeds per shot: L0 0.0009 / 0.0004, L2 0.0110 / 0.0107 (early stopping makes L2 vary by about 1.1 points per 100 shots between seeds).

## 3. Part C results (all FGA, as pre-registered)

### Chosen dials (FIT) and TEST MSE

| direction | arm | alpha | weights | K | FIT MSE | TEST MSE |
|---|---|---|---|---|---|---|
| fit 2024 / test 2025 | RAW | 0.5 | off | n/a | 0.026822 | 0.027036 |
| fit 2024 / test 2025 | RAW_SHRUNK | 0.75 | on | 500.0 | 0.025191 | 0.025497 |
| fit 2024 / test 2025 | QUALITY | 1.0 | on | 1000.0 | 0.025096 | 0.025411 |
| fit 2024 / test 2025 | CLASS | 1.0 | on | 250.0 | 0.025133 | 0.025560 |
| fit 2025 / test 2024 | RAW | 0.5 | off | n/a | 0.027036 | 0.026822 |
| fit 2025 / test 2024 | RAW_SHRUNK | 0.75 | on | 500.0 | 0.025497 | 0.025191 |
| fit 2025 / test 2024 | QUALITY | 1.0 | on | 1000.0 | 0.025411 | 0.025096 |
| fit 2025 / test 2024 | CLASS | 0.75 | on | 250.0 | 0.025557 | 0.025136 |

### Primary: TEST MSE gain vs RAW_SHRUNK (floors; positive = arm better)

| arm | fit 2024 / test 2025 | fit 2025 / test 2024 | seed-1 shift (floors) | pass rule |
|---|---|---|---|---|
| RAW | -1.54e-03 (floor 1.35e-04) = -11.42 | -1.63e-03 (floor 1.23e-04) = -13.25 | n/a | n/a (reference arm) |
| QUALITY | +8.67e-05 (floor 5.82e-05) = +1.49 | +9.49e-05 (floor 6.67e-05) = +1.42 | 0.02 / 0.01 | NOT CANDIDATE (>2 floors both dirs: False; seed-stable: True; responsiveness: False) |
| CLASS | -6.22e-05 (floor 5.47e-05) = -1.14 | +5.42e-05 (floor 4.26e-05) = +1.27 | 0.04 / 0.00 | NOT CANDIDATE (>2 floors both dirs: False; seed-stable: True; responsiveness: False) |

### Key secondary: QUALITY vs CLASS (positive = QUALITY better)

| direction | gain | floor | floors |
|---|---|---|---|
| fit 2024 / test 2025 | +1.49e-04 | 4.13e-05 | +3.60 |
| fit 2025 / test 2024 | +4.07e-05 | 5.05e-05 | +0.81 |

### Seed-1 retrain of the grade (dials frozen): gain vs RAW_SHRUNK in floors

| direction | arm | seed 0 | seed 1 |
|---|---|---|---|
| fit 2024 / test 2025 | QUALITY | +1.49 | +1.47 |
| fit 2024 / test 2025 | CLASS | -1.14 | -1.17 |
| fit 2025 / test 2024 | QUALITY | +1.42 | +1.43 |
| fit 2025 / test 2024 | CLASS | +1.27 | +1.28 |

### Gain vs RAW (unshrunk), floors

| arm | fit 2024 / test 2025 | fit 2025 / test 2024 |
|---|---|---|
| RAW_SHRUNK | +11.42 | +13.25 |
| QUALITY | +9.85 | +11.26 |
| CLASS | +9.55 | +11.98 |

### Segments: gain vs RAW_SHRUNK in floors (n team-games)

| segment | fit 2024 / test 2025 QUALITY | fit 2024 / test 2025 CLASS | fit 2024 / test 2025 Q-vs-C | fit 2025 / test 2024 QUALITY | fit 2025 / test 2024 CLASS | fit 2025 / test 2024 Q-vs-C |
|---|---|---|---|---|---|---|
| month_NovDec | +1.17 (n=2158) | -1.07 | +3.27 | +0.96 (n=2168) | +1.83 | -0.31 |
| month_JanApr | +0.99 (n=7116) | -0.60 | +2.19 | +1.05 (n=7082) | +0.04 | +1.31 |
| tier_power | +2.13 (n=2240) | +0.28 | +2.77 | +0.65 (n=2289) | +1.21 | -0.07 |
| tier_other | +0.56 (n=7034) | -1.47 | +2.67 | +1.28 (n=6961) | +0.77 | +1.03 |
| offence_only | -7.15 | -8.93 | - | -5.90 | -0.22 | - |
| defence_only | -7.15 | -9.63 | - | -3.92 | +5.32 | - |

### Responsiveness (TEST predicted-value quintiles; weighted slope of actual on predicted)

| direction | arm | slope [95%] | monotone | clause | q1..q5 actual pts/FGA | q1..q5 predicted |
|---|---|---|---|---|---|---|
| fit 2024 / test 2025 | RAW | 0.567 [0.534, 0.600] | True | False | 0.946 0.987 1.011 1.049 1.107 | 0.871 0.962 1.013 1.065 1.156 |
| fit 2024 / test 2025 | RAW_SHRUNK | 0.775 [0.732, 0.823] | True | False | 0.948 0.985 1.012 1.051 1.103 | 0.912 0.974 1.015 1.055 1.116 |
| fit 2024 / test 2025 | QUALITY | 0.804 [0.758, 0.851] | True | True | 0.948 0.988 1.012 1.046 1.105 | 0.915 0.976 1.015 1.053 1.112 |
| fit 2024 / test 2025 | CLASS | 0.767 [0.723, 0.816] | True | False | 0.950 0.986 1.011 1.048 1.105 | 0.912 0.975 1.015 1.055 1.117 |
| fit 2025 / test 2024 | RAW | 0.545 [0.513, 0.576] | True | False | 0.941 0.982 1.011 1.044 1.091 | 0.866 0.956 1.008 1.056 1.143 |
| fit 2025 / test 2024 | RAW_SHRUNK | 0.746 [0.701, 0.791] | True | False | 0.942 0.986 1.012 1.040 1.089 | 0.905 0.967 1.008 1.046 1.106 |
| fit 2025 / test 2024 | QUALITY | 0.761 [0.718, 0.806] | True | False | 0.941 0.987 1.015 1.034 1.092 | 0.907 0.968 1.008 1.046 1.104 |
| fit 2025 / test 2024 | CLASS | 0.768 [0.720, 0.814] | True | False | 0.943 0.989 1.010 1.038 1.090 | 0.909 0.969 1.008 1.046 1.103 |

### RAW alpha curve on TEST (RAW chosen weights)

- fit 2024 / test 2025: alpha 0.0: 0.027494, alpha 0.25: 0.027175, alpha 0.5: 0.027036, alpha 0.75: 0.027086, alpha 1.0: 0.027481
- fit 2025 / test 2024: alpha 0.0: 0.027221, alpha 0.25: 0.026938, alpha 0.5: 0.026822, alpha 0.75: 0.026882, alpha 1.0: 0.027262

### Stability (split-half, odd vs even games), per season

| season | side | measure | n teams | half r | full-length rel | FGA to rel 0.5 |
|---|---|---|---|---|---|---|
| 2024 | offence | raw | 362 | 0.624 | 0.769 | 547 |
| 2024 | offence | quality_L2 | 362 | 0.891 | 0.942 | 111 |
| 2024 | offence | making_L2 | 362 | 0.539 | 0.701 | 775 |
| 2024 | offence | quality_L0 | 362 | 0.850 | 0.919 | 160 |
| 2024 | offence | making_L0 | 362 | 0.569 | 0.725 | 688 |
| 2024 | defence | raw | 362 | 0.425 | 0.596 | 1228 |
| 2024 | defence | quality_L2 | 362 | 0.588 | 0.741 | 636 |
| 2024 | defence | making_L2 | 362 | 0.425 | 0.597 | 1226 |
| 2024 | defence | quality_L0 | 362 | 0.621 | 0.766 | 554 |
| 2024 | defence | making_L0 | 362 | 0.409 | 0.580 | 1313 |
| 2025 | offence | raw | 364 | 0.599 | 0.749 | 601 |
| 2025 | offence | quality_L2 | 364 | 0.852 | 0.920 | 157 |
| 2025 | offence | making_L2 | 364 | 0.554 | 0.713 | 724 |
| 2025 | offence | quality_L0 | 364 | 0.831 | 0.908 | 183 |
| 2025 | offence | making_L0 | 364 | 0.559 | 0.717 | 709 |
| 2025 | defence | raw | 364 | 0.510 | 0.676 | 862 |
| 2025 | defence | quality_L2 | 364 | 0.632 | 0.775 | 522 |
| 2025 | defence | making_L2 | 364 | 0.443 | 0.614 | 1130 |
| 2025 | defence | quality_L0 | 364 | 0.588 | 0.740 | 630 |
| 2025 | defence | making_L0 | 364 | 0.479 | 0.648 | 975 |

n team-games scored (TEST): fit 2024 / test 2025: 9274, fit 2025 / test 2024: 9250


Reading notes for section 3 (facts):
- "Seed-1 shift" is the absolute change in the arm's floors gain when the grade is retrained under random_state 1 with dials frozen; it is under 0.05 floors everywhere (nothing flagged SEED-UNSTABLE).
- All four arms have slope of actual on predicted below 1 (0.55-0.80). Only QUALITY in the first direction is inside the registered [0.8, 1.2] band (0.804, interval 0.758-0.851); every arm is monotone across quintiles.
- Offence-only and defence-only rows hold the other term at league. QUALITY's quality term is unshrunk by design, so its one-sided predictions are noisier than RAW_SHRUNK's (-3.2 to -7.2 floors); these rows say nothing about the two-sided gain.
- Segments with n < 2,000 would be labelled UNDERPOWERED; none in section 3 (smallest cell: Nov-Dec, 2,158 team-games); one in section 6 (Nov-Dec, fit 2025-26 test 2023-24 direction, 2,017). Segment floors are game-block bootstraps restricted to the segment; the Nov-Dec and power-conference cells are the smallest and their floors are widest.
- Stability above uses each season's cross-season EP, so the 2023-24 rows carry the NaN-shift issue of section 2 for L2 / L0 quality.

## 4. Deviations from the partner plan and from the spec

1. Our pbp is flattened (`shot_made`, `shot_range`, `shot_location_x/y`), not a `shotInfo` JSON; the partner reader was rewritten, the possession walker copied unchanged. The partner used a 2,944-game 2024-25 subset (345,820 FGA); we use the full season (5,642 D1 games).
2. Game-script weights: the partner CSV was unavailable; weights are `np.interp` on (0, 6, 12) -> (1.0, 0.56, 0.25), flat beyond 12.
3. Neutral site from the games file (hard error if missing); home flag from the games-file home team, not from `isHomeTeam` (they agree on 100%).
4. D1 vs D1 defined by non-null home and away conference in the games file (the partner used pbp conference columns; the two give the same 90.7% / 90.9% of games).
5. Grids matched at 70 combinations; RAW padded by repeating its 10 real combinations across the 7 K indices. RAW_SHRUNK is the added arm (fix 5).
6. The solver is a vectorised numpy re-implementation of the partner iteration (same order, 12 iterations). A class with no prior shots at all gets league 0 and zero weight; the first run produced NaN there and was corrected before any number was read (the run asserts a finite FIT grid and, per config, that QUALITY with K = 0 equals RAW and RAW_SHRUNK with K = 0 equals RAW to 1e-9).
7. The FIT-season grade is trained on the other (TEST) season, as the plan says; this lets TEST-season shot information into the FIT-season grade used for dial choice (no team identity is used).
8. Post-hoc, not pre-registered: section 6 re-runs part C on located FGA only, because of the NaN-shift finding in part B. It is a sensitivity; the registered result is section 3.
9. Not run: a QUALITY variant that also shrinks the quality term; rim x/y never enters (class only).
10. `shot_assisted` is not a feature; assisted rows are retained (partner wording ambiguous).
11. Tie rule never invoked (no arm exceeded the rule).

## 5. Pointers

Pass-rule inputs: sections 0 and 3. The RAW alpha curve (opponent-adjustment value) is in section 3: TEST MSE is lowest at alpha 0.5 for RAW in both directions and rises at 0 and at 1.

## 6. Sensitivity (post-hoc): Part C on located FGA only

Same scripts with `--located-only`: unlocated FGA are removed from the grade-evaluation shots, the team-game aggregates, the target and every arm (2023-24 loses 10.3% of FGA; 2024-25 1.3%). Dials re-chosen on FIT.

### Chosen dials (FIT) and TEST MSE

| direction | arm | alpha | weights | K | FIT MSE | TEST MSE |
|---|---|---|---|---|---|---|
| fit 2024 / test 2025 | RAW | 0.5 | off | n/a | 0.029293 | 0.027458 |
| fit 2024 / test 2025 | RAW_SHRUNK | 0.75 | on | 500.0 | 0.027421 | 0.025874 |
| fit 2024 / test 2025 | QUALITY | 1.0 | on | 1000.0 | 0.027288 | 0.025789 |
| fit 2024 / test 2025 | CLASS | 0.75 | on | 250.0 | 0.027360 | 0.025937 |
| fit 2025 / test 2024 | RAW | 0.5 | off | n/a | 0.027458 | 0.029293 |
| fit 2025 / test 2024 | RAW_SHRUNK | 0.75 | on | 500.0 | 0.025874 | 0.027421 |
| fit 2025 / test 2024 | QUALITY | 1.0 | on | 1000.0 | 0.025789 | 0.027288 |
| fit 2025 / test 2024 | CLASS | 0.75 | on | 250.0 | 0.025937 | 0.027360 |

### Primary: TEST MSE gain vs RAW_SHRUNK (floors; positive = arm better)

| arm | fit 2024 / test 2025 | fit 2025 / test 2024 | seed-1 shift (floors) | pass rule |
|---|---|---|---|---|
| RAW | -1.58e-03 (floor 1.29e-04) = -12.32 | -1.87e-03 (floor 1.49e-04) = -12.57 | n/a | n/a (reference arm) |
| QUALITY | +8.49e-05 (floor 5.70e-05) = +1.49 | +1.33e-04 (floor 6.10e-05) = +2.17 | 0.03 / 0.03 | NOT CANDIDATE (>2 floors both dirs: False; seed-stable: True; responsiveness: False) |
| CLASS | -6.36e-05 (floor 5.11e-05) = -1.24 | +6.09e-05 (floor 4.69e-05) = +1.30 | 0.04 / 0.00 | NOT CANDIDATE (>2 floors both dirs: False; seed-stable: True; responsiveness: False) |

### Key secondary: QUALITY vs CLASS (positive = QUALITY better)

| direction | gain | floor | floors |
|---|---|---|---|
| fit 2024 / test 2025 | +1.48e-04 | 4.28e-05 | +3.47 |
| fit 2025 / test 2024 | +7.16e-05 | 4.26e-05 | +1.68 |

### Seed-1 retrain of the grade (dials frozen): gain vs RAW_SHRUNK in floors

| direction | arm | seed 0 | seed 1 |
|---|---|---|---|
| fit 2024 / test 2025 | QUALITY | +1.49 | +1.46 |
| fit 2024 / test 2025 | CLASS | -1.24 | -1.28 |
| fit 2025 / test 2024 | QUALITY | +2.17 | +2.14 |
| fit 2025 / test 2024 | CLASS | +1.30 | +1.30 |

### Gain vs RAW (unshrunk), floors

| arm | fit 2024 / test 2025 | fit 2025 / test 2024 |
|---|---|---|
| RAW_SHRUNK | +12.32 | +12.57 |
| QUALITY | +10.69 | +11.34 |
| CLASS | +10.20 | +11.64 |

### Segments: gain vs RAW_SHRUNK in floors (n team-games)

| segment | fit 2024 / test 2025 QUALITY | fit 2024 / test 2025 CLASS | fit 2024 / test 2025 Q-vs-C | fit 2025 / test 2024 QUALITY | fit 2025 / test 2024 CLASS | fit 2025 / test 2024 Q-vs-C |
|---|---|---|---|---|---|---|
| month_NovDec | +1.17 (n=2154) | -0.29 | +1.87 | +2.07 (n=2017) | +1.87 | +0.91 |
| month_JanApr | +0.95 (n=7111) | -1.37 | +2.80 | +1.11 (n=6786) | +0.12 | +1.41 |
| tier_power | +2.06 (n=2238) | -0.51 | +3.22 | +1.49 (n=2172) | +1.07 | +0.90 |
| tier_other | +0.55 (n=7027) | -1.13 | +2.07 | +1.65 (n=6631) | +0.85 | +1.42 |
| offence_only | -6.97 | -0.21 | - | -3.20 | -0.44 | - |
| defence_only | -6.61 | +1.65 | - | -4.11 | +4.52 | - |


### Responsiveness (TEST predicted-value quintiles; weighted slope of actual on predicted)

| direction | arm | slope [95%] | monotone | clause | q1..q5 actual pts/FGA | q1..q5 predicted |
|---|---|---|---|---|---|---|
| fit 2024 / test 2025 | RAW | 0.563 [0.534, 0.594] | True | False | 0.940 0.983 1.005 1.045 1.101 | 0.864 0.956 1.007 1.059 1.151 |
| fit 2024 / test 2025 | RAW_SHRUNK | 0.773 [0.728, 0.819] | True | False | 0.944 0.980 1.007 1.045 1.098 | 0.906 0.968 1.009 1.049 1.111 |
| fit 2024 / test 2025 | QUALITY | 0.801 [0.755, 0.847] | True | True | 0.943 0.983 1.007 1.041 1.100 | 0.909 0.970 1.009 1.048 1.106 |
| fit 2024 / test 2025 | CLASS | 0.777 [0.729, 0.823] | True | False | 0.945 0.983 1.005 1.044 1.097 | 0.908 0.969 1.009 1.049 1.109 |
| fit 2025 / test 2024 | RAW | 0.534 [0.506, 0.567] | True | False | 0.931 0.979 1.002 1.036 1.090 | 0.853 0.948 1.001 1.051 1.143 |
| fit 2025 / test 2024 | RAW_SHRUNK | 0.748 [0.705, 0.794] | True | False | 0.935 0.980 1.008 1.031 1.085 | 0.898 0.960 1.001 1.040 1.100 |
| fit 2025 / test 2024 | QUALITY | 0.778 [0.734, 0.822] | True | False | 0.934 0.985 1.004 1.029 1.087 | 0.901 0.961 1.001 1.039 1.096 |
| fit 2025 / test 2024 | CLASS | 0.769 [0.725, 0.816] | True | False | 0.937 0.982 1.004 1.031 1.085 | 0.901 0.961 1.001 1.039 1.097 |

