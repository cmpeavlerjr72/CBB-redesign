# Venue charting asymmetry (PM ruling 17a, 2026-10-27)

Diagnostic only: no model adopted, no recommendation. Script `scripts/diag_venue_charting_v1.py`; full numbers in `docs/tests/venue_charting_asymmetry_2026-10-27.json`. Sonnet worker.

Data: CBBD located attempts 2022-2025 from `data/processed/models/shot_location/round1/design_loc_cols_v1.parquet` (round 1's bands and as-of profiles, unchanged), joined row-for-row to the fg_make design for `offense_is_home` and `neutral_site`. Neutral games dropped (1,933,306 of 2,239,678 rows remain). Venue = the home team of the game. A team-season enters a comparison with at least 40 located attempts of the class on each side. "Floor" for a team-season = SD of its home-minus-away difference over 40 permutations of the home/away labels across its own games (null: the site does not matter); the mean difference's floor is a bootstrap over team-seasons (400 reps). Under the null about 5% of team-seasons exceed 2 SE by chance.

## Answer

The same teams do chart differently at home and away, and a venue term explains most of the away-only defence-profile gain, but the diagnostics cannot say whether that venue term is the home crew's placement style or the home team's real defensive shaping of visitor shots. What the evidence does establish: (a) at home, a team's located jump2 attempts are 0.9 pp more short-two (floor 0.3 pp), its threes 1.2 pp more corner and 1.0 pp less top (floor 0.2 pp), its jumpers 0.2 ft closer (median) and its three-point p90 distance 0.3 ft shorter, and 10-16% of team-seasons differ beyond 2 SE on band shares (null 5%) and 10-31% on distance; (b) adding one ridge dummy per home venue (fit 2022-24, scored 2025) takes the defence-profile gain for AWAY offences from +5.9e-3 to +0.7e-3 per attempt on jump2 (89% absorbed) and from +6.3e-3 to +1.8e-3 on threes (71%), while the defence gain for HOME offences does not shrink (it rises: jump2 -1.2e-3 to +0.8e-3, three +1.5e-3 to +2.4e-3), and the jump2 defence coefficient falls from 0.80 to 0.36; (c) a venue term learned from the home team's own shots predicts what visitors do in that arena (gain +6.4e-3 jump2, +4.6e-3 three, SE 1.0e-3 and 0.7e-3) and venue coefficients from the away-only and home-only fits correlate 0.71 (jump2) and 0.62-0.80 (three), so the venue component is shared by both teams' shots as a crew effect would be, but it is also what a persistent real home-team defence style would produce; (d) the second-source check (hoopR vs CBBD, 2025 from 2025-02-10) cannot discriminate, because the two sources chart the same ESPN feed. Practical reading for the defence-allowed profile: its away-offence gain cannot be treated as matchup signal until a venue term is controlled; the home-offence gain is not affected by the venue term.

## Part 1. Same team, offence band shares, home minus away (non-neutral games, located attempts)

| class | band | team-seasons | mean diff (pp) | bootstrap SE (pp) | z | median abs team diff (pp) | teams beyond 2 SE (%) | null expectation (%) |
|---|---|---|---|---|---|---|---|---|
| jump2 | short2 | 1,339 | +0.92 | 0.31 | 3.0 | 7.4 | 16.1 (+9.3 / -6.8) | about 5 |
| three | corner3 | 1,379 | +1.19 | 0.18 | 6.6 | 4.2 | 14.9 (+8.3 / -6.6) | about 5 |
| three | wing3 | 1,379 | -0.17 | 0.19 | 0.9 | 4.5 | 13.6 | about 5 |
| three | top3 | 1,379 | -1.02 | 0.18 | 5.7 | 4.2 | 10.2 | about 5 |

By season (short2 / corner3 home minus away, pp, SE): 2022 +0.53 (0.69) / +1.67 (0.41); 2023 +0.97 (0.61) / +0.87 (0.43); 2024 +1.51 (0.57) / +1.21 (0.31); 2025 +0.65 (0.55) / +1.06 (0.32). The direction is stable in all four seasons; each single jump2 season is within about 2.5 SE. The mean differences are small beside the per-team spread (permutation SE per team 7.4 pp jump2, 4.7-5.1 pp three), so the finding is a modest mean shift plus extra heterogeneity across teams, not a large uniform offset. Opponent quality and tempo differ by site and are not controlled here.

## Part 2. Located share and distance from basket, home minus away

| class | stat | team-seasons | mean diff | bootstrap SE | z | teams beyond 2 SE (%) |
|---|---|---|---|---|---|---|
| rim | located share | 1,423 | +0.65 pp | 0.34 | 1.9 | 5.3 |
| jump2 | located share | 1,397 | +0.78 pp | 0.33 | 2.4 | 3.7 |
| three | located share | 1,425 | +0.83 pp | 0.33 | 2.5 | 2.5 |
| rim | distance median | 1,377 | -0.07 ft | 0.03 | 2.7 | 31.4 |
| rim | distance p90 | 1,377 | -0.45 ft | 0.03 | 12.9 | 25.5 |
| jump2 | distance median | 1,339 | -0.22 ft | 0.05 | 4.8 | 16.5 |
| jump2 | distance p90 | 1,339 | -0.16 ft | 0.03 | 5.1 | 10.2 |
| three | distance median | 1,379 | -0.03 ft | 0.02 | 1.8 | 26.6 |
| three | distance p90 | 1,379 | -0.31 ft | 0.03 | 11.6 | 24.3 |

Located share is close to site-neutral (+0.7 to +0.8 pp, about 2 SE, and the share of teams beyond 2 SE is at or below the null). Distance is not: shots at home are charted closer to the basket, most visibly in the tails (rim p90 0.45 ft, three p90 0.31 ft), and a quarter of team-seasons differ beyond 2 SE on rim and three-point p90. A three-point p90 that moves 0.3 ft with the site is difficult to explain by play (the line is fixed) and is the cleanest sign of a site-dependent placement offset.

## Part 3. Venue fixed effect, fold 2 (fit 2022-24, scored 2025)

Multinomial logit per class on located non-neutral rows (jump2: 321,160 train / 127,074 test; three: 468,518 / 213,227), 362 home venues in training (0.9% / 0.7% of test rows at an unseen venue). Arms: `off` = offence profile + offence-is-home; `off+def` adds the defence-allowed profile; `+venue` adds one ridge dummy (C = 1; profile columns scaled so the penalty barely touches them) per home venue. Gains are held-out log loss per attempt (arm A minus arm B, positive = B better), game-block bootstrap SE in brackets.

| class | rows | def profile given off (off to off+def) | def profile given off+venue | venue given off | venue given off+def |
|---|---|---|---|---|---|
| jump2 | all | +2.51e-3 (3.0e-4) | +0.72e-3 (1.2e-4) | +4.70e-3 (5.6e-4) | +2.91e-3 (4.9e-4) |
| jump2 | offence AWAY (65,670) | +5.94e-3 (4.3e-4) | +0.67e-3 (1.4e-4) | +9.08e-3 (8.3e-4) | +3.81e-3 (6.7e-4) |
| jump2 | offence HOME (61,404) | -1.15e-3 (4.0e-4) | +0.77e-3 (1.8e-4) | +0.03e-3 (6.9e-4) | +1.95e-3 (6.9e-4) |
| three | all | +3.85e-3 (2.4e-4) | +2.11e-3 (1.6e-4) | +3.44e-3 (5.2e-4) | +1.70e-3 (4.7e-4) |
| three | offence AWAY (105,486) | +6.27e-3 (3.6e-4) | +1.80e-3 (2.1e-4) | +6.70e-3 (6.2e-4) | +2.22e-3 (5.2e-4) |
| three | offence HOME (107,741) | +1.47e-3 (3.2e-4) | +2.41e-3 (2.4e-4) | +0.25e-3 (6.2e-4) | +1.19e-3 (6.0e-4) |

Held-out log loss, jump2: off 0.67402, off+def 0.67151, off+venue 0.66932, off+def+venue 0.66860. Three: 1.05618, 1.05233, 1.05273, 1.05063. The round-1 pattern (defence gain +5.6e-3 away, -0.9e-3 home on jump2) reproduces here (+5.9e-3, -1.2e-3).

Defence coefficients (log-odds per unit of the centred allowed profile): jump2 short2 0.80 without the venue term, 0.36 with it. Threes (own-band slope against the other classes): corner3 1.23 to -0.13, wing3 -0.44 to -0.13, top3 0.47 to 1.22. So the corner and the jump2 defence slopes collapse with a venue term in; the top-of-key slope does not. The venue coefficients have SD 0.28 (jump2) and 0.15 (three) in log-odds.

### Part 3b. Does the venue term behave like a crew that charts both teams?

A crew places every shot in its arena. The venue ridge term (offence profile in the model) was fit separately on AWAY-offence rows and on HOME-offence rows (2022-24) and applied to the other site's 2025 rows.

| class | corr of venue coefficients, away-fit vs home-fit | gain, fit on HOME rows scored on AWAY rows | gain, fit on AWAY rows scored on HOME rows |
|---|---|---|---|
| jump2 | 0.71 | +6.41e-3 (SE 1.02e-3) | -1.81e-3 (0.88e-3) |
| three | 0.74 / 0.80 / 0.62 (corner / wing / top) | +4.61e-3 (0.72e-3) | -1.14e-3 (0.81e-3) |

The two fits agree on which venues are extreme and the home-fit term transfers to visitors, which fits a venue-level component common to both teams. The transfer is one-directional: the away-fit term does not help home rows (slightly hurts), because a home-offence row's venue dummy is its own team's identity and the as-of offence profile, itself built from half home games, already carries that venue. This test does not identify the mechanism: a home team that persistently forces visitors toward short twos or off the corner would show the same venue coefficients with the same correlation.

## Part 4. Second charting of the same games: hoopR vs CBBD, 2025 from 2025-02-10

Expectation stated before the run. hoopR and CBBD report the same ESPN feed (coordinate audit: 91-95% within 1 ft from 2025-02-10), so they are not two independent charters. If the home-away difference is a placement style of the home crew, both sources should show it and agree with each other; band disagreement between the sources should not depend on the site. A source that disagreed more on home-charted attempts would indicate a pipeline-specific frame artifact, not crew style. A null here therefore cannot exonerate or convict the crew hypothesis.

Found (1,131 non-neutral games, 80,622 attempts located by both): band differs between the sources on 3.53% of away-offence and 3.62% of home-offence jump2 attempts, 4.55% and 4.71% of threes; the sources do not disagree more at home. Per-team home-minus-away band-share differences in the window (95 team-seasons jump2, 285 three; the window is 7 weeks, so every cell is underpowered): CBBD short2 +1.27 pp (SE 1.42), hoopR +1.79 pp (1.53); corner3 -0.07 / -0.33 pp (SE about 0.55 / 0.53); top3 +0.35 / +0.38 pp. Correlation across teams of the CBBD and hoopR differences: 0.87 (jump2), 0.92 / 0.90 / 0.90 (three: corner / top / wing); mean absolute difference between the sources 4.5 pp jump2, 2.3-3.4 pp three, which is the sampling noise of a 7-week window, not a source difference. The window's mean site differences are inside their SEs in both sources, so the 4-season effect of Part 1 neither replicates nor fails to replicate here.

## Limits

Site effects of real play (opponent quality, tempo, crowd, scheme) are not separated from crew style in Parts 1-3. The permutation null removes any site effect, real or charted. The venue dummies are the home team's identity, so the venue term absorbs a persistent home-team trait of any kind; Part 3 is therefore an upper bound on a crew effect. Parts 1-3 use located attempts only; located share is nearly site-balanced (Part 2), so selection into located status is not the driver. The 2022-24 missing-game bias (power, neutral and postseason games) applies to the training seasons.
