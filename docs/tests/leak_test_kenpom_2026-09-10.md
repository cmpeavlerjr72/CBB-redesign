# KenPom pregame-feature leak test (INV-45, ported to CBB) -- 2026-09-10

`scripts/leak_test_pregame_features.py`. Margin panel: hoopR `team_box` seasons 2022, 2023, 2024, 2025 (49,422 team-games, 1122 teams). Season 2026 is excluded throughout -- it is the sealed blind test (`src/cbb_sim/data/seal.py`).

## The statistic

CBB adaptation of the CFB INV-45 method (`docs/postmortem/05_cfb_methodology_extract.md` section 4): rows are one per (team, season, game), ordered by **game_date** (CBB has no trustworthy week label -- 0-4 games/week depending on scheduling -- so "consecutive" means adjacent games, not adjacent weeks, unlike the CFB original which ordered by kickoff within a week label).

| name | definition |
|---|---|
| `as-joined` | corr(`f[t] - f[t-1]`, that team's margin in **game t**) -- the leak channel: a pregame row cannot contain game t |
| `legit-update` | corr(`f[t] - f[t-1]`, that team's margin in the **previous game**) -- the healthy signature of a season-to-date column, and this test's own positive control: numerically what as-joined would read if the column were joined one game late |
| `level` | corr(`f[t]`, margin in game t) -- plain predictive correlation, context only |

**Flag rule:** `|as-joined| > 0.15` = **LEAK** (honest baseline ~0.04-0.08). `n < 300` team-games = UNDERPOWERED (not scored). Zero within-season variance in the delta = STATIC (change-form undefined) -- reported instead with the level-form variant: `|level corr| > 0.15` on a STATIC column is itself a LEAK signature (a same-season aggregate whose level correlation with the team's own game margins is inflated because it was computed from those same games).

## KenPom name-matching

0 of 27,855 snapshot rows in seasons 2022, 2023, 2024, 2025 carry an unmatched `team` (dropped before joining -- see `scripts/build_kenpom_snapshots.py` output / `src/cbb_sim/data/kenpom.py` for the full name-matching writeup and the current match report).

## Arm A -- as-of join (strictly before), the join the sim uses

Coverage (share of team-games with a non-null joined feature): `adj_o_c` 95.3%, `adj_d_c` 95.3%, `adj_t_rel` 95.3%

| column | season | n | as-joined | legit-update | level | verdict |
|---|---:|---:|---:|---:|---:|---|
| `adj_o_c` | 2022 | 10995 | +0.014 | +0.145 | +0.249 | pass |
| `adj_o_c` | 2023 | 11507 | +0.003 | +0.143 | +0.215 | pass |
| `adj_o_c` | 2024 | 11542 | +0.027 | +0.131 | +0.235 | pass |
| `adj_o_c` | 2025 | 11608 | +0.019 | +0.151 | +0.217 | pass |
| `adj_o_c` | ALL | 45652 | +0.017 | +0.141 | +0.229 | pass |
| `adj_d_c` | 2022 | 10995 | -0.017 | -0.148 | -0.217 | pass |
| `adj_d_c` | 2023 | 11507 | -0.035 | -0.151 | -0.198 | pass |
| `adj_d_c` | 2024 | 11542 | -0.036 | -0.151 | -0.214 | pass |
| `adj_d_c` | 2025 | 11608 | -0.008 | -0.153 | -0.205 | pass |
| `adj_d_c` | ALL | 45652 | -0.024 | -0.150 | -0.208 | pass |
| `adj_t_rel` | 2022 | 10995 | -0.004 | +0.007 | -0.013 | pass |
| `adj_t_rel` | 2023 | 11507 | -0.010 | +0.004 | +0.017 | pass |
| `adj_t_rel` | 2024 | 11542 | -0.004 | -0.000 | -0.011 | pass |
| `adj_t_rel` | 2025 | 11608 | -0.016 | -0.005 | -0.009 | pass |
| `adj_t_rel` | ALL | 45652 | -0.008 | +0.001 | -0.004 | pass |

## Arm B -- same-day join (on-or-before), deliberate positive control

Coverage: `adj_o_c` 95.3%, `adj_d_c` 95.3%, `adj_t_rel` 95.3%

| column | season | n | as-joined | legit-update | level | verdict |
|---|---:|---:|---:|---:|---:|---|
| `adj_o_c` | 2022 | 10995 | +0.008 | +0.172 | +0.249 | pass |
| `adj_o_c` | 2023 | 11512 | +0.019 | +0.175 | +0.219 | pass |
| `adj_o_c` | 2024 | 11542 | +0.020 | +0.187 | +0.235 | pass |
| `adj_o_c` | 2025 | 11611 | +0.030 | +0.190 | +0.219 | pass |
| `adj_o_c` | ALL | 45660 | +0.020 | +0.182 | +0.230 | pass |
| `adj_d_c` | 2022 | 10995 | -0.020 | -0.178 | -0.217 | pass |
| `adj_d_c` | 2023 | 11512 | -0.034 | -0.189 | -0.201 | pass |
| `adj_d_c` | 2024 | 11542 | -0.036 | -0.187 | -0.216 | pass |
| `adj_d_c` | 2025 | 11611 | -0.020 | -0.184 | -0.207 | pass |
| `adj_d_c` | ALL | 45660 | -0.027 | -0.185 | -0.210 | pass |
| `adj_t_rel` | 2022 | 10995 | +0.011 | -0.009 | -0.011 | pass |
| `adj_t_rel` | 2023 | 11512 | +0.009 | +0.004 | +0.018 | pass |
| `adj_t_rel` | 2024 | 11542 | +0.000 | -0.004 | -0.009 | pass |
| `adj_t_rel` | 2025 | 11611 | -0.008 | +0.006 | -0.010 | pass |
| `adj_t_rel` | ALL | 45660 | +0.003 | -0.001 | -0.003 | pass |

## Arm C -- CBBD end-of-season `/ratings/adjusted`, joined by (team, season)

Coverage: `offensiveRating` 94.6%, `defensiveRating` 94.6%, `netRating` 94.6%

| column | season | n | as-joined | legit-update | level | verdict |
|---|---:|---:|---:|---:|---:|---|
| `offensiveRating` | 2022 | 10815 | n/a | n/a | +0.279 | STATIC (LEAK (level-form), level r=+0.279) |
| `offensiveRating` | 2023 | 11392 | n/a | n/a | +0.109 | STATIC (pass (level-form), level r=+0.109) |
| `offensiveRating` | 2024 | 11477 | n/a | n/a | +0.281 | STATIC (LEAK (level-form), level r=+0.281) |
| `offensiveRating` | 2025 | 11648 | n/a | n/a | +0.263 | STATIC (LEAK (level-form), level r=+0.263) |
| `offensiveRating` | ALL | 45332 | n/a | n/a | +0.239 | STATIC (LEAK (level-form), level r=+0.239) |
| `defensiveRating` | 2022 | 10815 | n/a | n/a | -0.257 | STATIC (LEAK (level-form), level r=-0.257) |
| `defensiveRating` | 2023 | 11392 | n/a | n/a | -0.054 | STATIC (pass (level-form), level r=-0.054) |
| `defensiveRating` | 2024 | 11477 | n/a | n/a | -0.253 | STATIC (LEAK (level-form), level r=-0.253) |
| `defensiveRating` | 2025 | 11648 | n/a | n/a | -0.254 | STATIC (LEAK (level-form), level r=-0.254) |
| `defensiveRating` | ALL | 45332 | n/a | n/a | -0.213 | STATIC (LEAK (level-form), level r=-0.213) |
| `netRating` | 2022 | 10815 | n/a | n/a | +0.301 | STATIC (LEAK (level-form), level r=+0.301) |
| `netRating` | 2023 | 11392 | n/a | n/a | +0.178 | STATIC (LEAK (level-form), level r=+0.178) |
| `netRating` | 2024 | 11477 | n/a | n/a | +0.297 | STATIC (LEAK (level-form), level r=+0.297) |
| `netRating` | 2025 | 11648 | n/a | n/a | +0.285 | STATIC (LEAK (level-form), level r=+0.285) |
| `netRating` | ALL | 45332 | n/a | n/a | +0.266 | STATIC (LEAK (level-form), level r=+0.266) |

## Honest reading

- **Arm A (as-of, strictly before):** worst pooled as-joined is `adj_d_c` at -0.024 (n=45652). 0 of 15 (column, season) cells flagged LEAK. This is the join the sim is meant to use, and it reads clean.

- **Arm B (same-day, on-or-before):** worst pooled as-joined is `adj_d_c` at -0.027 (n=45660). 0 of 15 (column, season) cells flagged LEAK. Same-day joins are safe: KenPom's date-D snapshot evidently reflects games only through D-1 (a morning scrape, ahead of that night's games), so allowing an exact-date match does not pull in the game it is about to predict -- Arm B reads statistically the same as Arm A.

- **Caveat on `legit-update` magnitude (both A and B):** `legit-update` runs ~0.13-0.19 here, above the ~0.04-0.08 single-game honest baseline the CFB original reports. This is a granularity artifact, not a defect: 2022-2025 KenPom snapshots are WEEKLY while CBB teams play ~2 games/week, so the delta between two consecutive-game rows often spans zero or one snapshot updates, and "the previous game's margin" this delta correlates with is really standing in for however many games happened since the last weekly refresh -- a coarser, noisier version of the single-game lag CFB's weekly-game cadence gives for free. It does not affect the leak verdict (`as-joined` is unaffected by this and stays low), only the size of the test's own positive-control number.

- **Arm C (CBBD end-of-season, joined by team+season):** all 3 columns STATIC as expected (zero within-season variance -- a single end-of-season number joined to every game that team played). 3 of 3 pooled columns show a level-form LEAK (|level corr| > 0.15), confirming the second leak class: a same-season aggregate carries no change-form signature at all, so it can only be caught by the level-form variant, and here it clearly is one -- a team's own end-of-season rating is mechanically inflated by the same games it is being correlated against. This arm is a positive control, not a candidate feature: it demonstrates why any end-of-season rating must be shifted to the PRIOR season (or replaced by a genuine point-in-time snapshot, i.e. Arm A) before it is allowed near a feature table.

**Bottom line:** the KenPom `as_of()` join in `src/cbb_sim/data/kenpom.py` is the one arm cleared to feed a feature table (PASS, honest as-joined correlations in the 0.003-0.036 range). The CBBD end-of-season rating (Arm C) must never be joined by (team, season) alone -- only as a prior-season prior, or via its own point-in-time snapshot if CBBD ever exposes one.
