# fg_make: what `score_diff` really does to make probability -- 2026-09-10

Evidence for `docs/LEARNINGS.md` L23 and `ARCHITECTURE_DECISIONS.md` Decision 10,
written BEFORE the fg_make round-2 pre-registration it feeds
(`docs/models/fg_make/experiments.md` section 13).

Script: `scripts/diag_fg_make_state_confound.py`.
Numbers: `data/processed/models/fg_make/state_confound.json`.
Population: the round-1 bake-off's own universe -- D-I, non-truncated,
`pbp_complete`, possessions v2, seasons 2022-2025, **2,241,063 modelled
field-goal attempts**. 2026 is sealed and never read.

Cells with fewer than 2,000 attempts are labelled UNDERPOWERED and are never
read as signal or as absence of signal.

---

## 0. HEADLINE: `score_diff` in this model is a POST-OUTCOME feature

The round-1 `C_plus_state` bundle reads `score_diff` off the `homeScore` /
`awayScore` columns on the attempt's **own row**
(`cbb_sim.models.fg_make._season_events`, `np.where(off_home, hs - as_, as_ - hs)`).
In the CBBD/ESPN feed that column is the score **after** the play. A made three
already carries its own three points on the row that records it.

That puts `score_diff` in the same family as `blocked` and `and_one`, both of
which this model bans by name (`FG.BANNED_FEATURES`, `features.md` section 3),
and it was not caught because it is filed under "state" rather than under
"outcome".

Three independent reads, none of which uses the model:

**1. The feed text.** 2025 CBBD game 24, first rows: row 1 is
`Sean Craig made Three Point Jumper` and `homeScore` is already 3, from 0 on the
row before.

**2. The own-row score move, measured on the raw plays table** (all field-goal
rows, `playType` contains "Shot" and not "Free"/"Block"; the move is this row's
`homeScore`/`awayScore` minus the previous row's, from the shooting team's side):

| season | made FGA | delta == the shot's value | missed FGA | delta == 0 |
|---|---:|---:|---:|---:|
| 2022 | 344,838 | **94.47%** | 483,161 | 99.47% |
| 2023 | 364,760 | **94.35%** | 509,401 | 99.45% |
| 2024 | 372,616 | **93.64%** | 517,331 | 99.51% |
| 2025 | 386,214 | **93.62%** | 535,605 | 99.54% |

A made shot's own points are in its own row's score; a miss's are not. (The
~6% of makes whose delta is 0 are makes whose points arrived on an earlier row
at the same clock -- a dead-ball or timeout row. The shot's points are in the
score either way, so the correction below is right for both.)

**3. The effect collapses when the shot's own points are removed.**
`score_diff_pre = score_diff - (3 if a made three else 2 if a made two else 0)`.
Raw make rate by margin bucket, 2022-2025, every cell above 17,000 attempts:

**`FGA_rim`** -- span 20.12 pp POST-shot, **7.64 pp** pre-shot

| margin (shooter's side) | post-shot n | post-shot make% | pre-shot n | pre-shot make% |
|---|---:|---:|---:|---:|
| <= -20 | 29,597 | 50.46 | 34,509 | 57.51 |
| -19..-10 | 89,270 | 50.88 | 103,086 | 57.46 |
| -9..-4 | 135,625 | 51.97 | 152,131 | 57.18 |
| -3..+3 | 265,523 | 56.71 | 270,415 | 57.49 |
| +4..+9 | 152,570 | 64.41 | 133,742 | 59.40 |
| +10..+19 | 105,217 | 67.52 | 90,423 | 62.21 |
| >= +20 | 39,695 | 70.58 | 33,191 | 64.82 |

**`FGA_jump2`** -- span 16.21 pp POST-shot, **2.70 pp** pre-shot

| margin | post-shot n | post-shot make% | pre-shot n | pre-shot make% |
|---|---:|---:|---:|---:|
| <= -20 | 23,797 | 31.07 | 26,208 | 37.41 |
| -19..-10 | 65,143 | 32.61 | 71,246 | 38.38 |
| -9..-4 | 100,194 | 33.18 | 108,725 | 38.42 |
| -3..+3 | 201,599 | 37.55 | 203,686 | 38.19 |
| +4..+9 | 100,623 | 44.97 | 90,527 | 38.84 |
| +10..+19 | 61,276 | 46.57 | 54,625 | 40.06 |
| >= +20 | 19,926 | 47.28 | 17,541 | 40.11 |

**`FGA_3`** -- span 21.64 pp POST-shot, **3.56 pp** pre-shot

| margin | post-shot n | post-shot make% | pre-shot n | pre-shot make% |
|---|---:|---:|---:|---:|
| <= -20 | 33,295 | 23.25 | 37,438 | 31.74 |
| -19..-10 | 96,348 | 24.50 | 107,907 | 32.59 |
| -9..-4 | 147,724 | 25.95 | 163,581 | 33.13 |
| -3..+3 | 279,134 | 32.69 | 284,503 | 33.96 |
| +4..+9 | 152,584 | 42.12 | 135,093 | 34.63 |
| +10..+19 | 102,529 | 43.79 | 88,928 | 35.20 |
| >= +20 | 39,394 | 44.89 | 33,558 | 35.30 |

| class | span POST | span PRE | **manufactured by the leak** | share of the apparent effect |
|---|---:|---:|---:|---:|
| `FGA_rim` | 20.12 pp | 7.64 pp | **12.49 pp** | 62.0% |
| `FGA_jump2` | 16.21 pp | 2.70 pp | **13.51 pp** | 83.3% |
| `FGA_3` | 21.64 pp | 3.56 pp | **18.08 pp** | 83.5% |

The post-shot curve is also the wrong SHAPE: it has a step of 4.7-9.4 pp across
the tied bucket's boundaries, which is the arithmetic of a made shot pushing
itself into the next bucket, not a basketball effect.

**The engine feeds the RIGHT quantity.** `cbb_sim.engine.loop._state_block`
writes `st.off_score_diff()`, the live margin **before** the shot is resolved.
So this is a train/serve skew on top of a leak: the model learned "the margin
on this row is high **because this shot went in**" and the sim reads it as
"this team is ahead, so it shoots better". That is the mechanism behind L23's
margin SD 34.6 and home/away correlation -0.64, and it is ~3x larger than the
genuine state effect. Nothing about it is a tuning problem.

**Scope check on the rest of the cascade.** `possession_outcome` and `clock`
both take `score_diff` from the possessions table's `start_score_diff`
(possession start, strictly before any of the possession's own scoring) and are
CLEAN. `cbb_sim.models.attribution` line 534 builds `score_diff` from
`hs - as_` on the candidate's own row, the same construction as here, and
should be audited by that model's owner -- flagged, not touched.

---

## 1. (a) The pregame team-strength confound

Everything below uses the CORRECTED `score_diff_pre`.

Method: g-computation on nested logistic models. Every attempt is
counterfactually assigned to each margin bucket in turn and the mean predicted
make probability is reported; the SPAN of that curve is the model-implied
margin effect. Blocks are nested, so a span that collapses between two rows is
that block's confound share.

- **M0** margin bucket only
- **M1** + pregame own ratings of BOTH teams (`off_rating_off_c`,
  `off_rating_def_c`, `def_rating_off_c`, `def_rating_def_c`), site, season index
- **M2** + the two teams' as-of form on this class (`off_make_c`, `def_allow_c`)
- **M3** + the shooter block (as-of rate, attempts, prior season, position,
  exposure)

### `FGA_rim` (n = 817,497)

| model | <= -20 | -19..-10 | -9..-4 | -3..+3 | +4..+9 | +10..+19 | >= +20 | span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 57.51 | 57.46 | 57.18 | 57.49 | 59.40 | 62.21 | 64.82 | **7.64** |
| M1 | 60.92 | 59.22 | 58.02 | 57.46 | 58.47 | 60.30 | 61.26 | **3.80** |
| M2 | 61.01 | 59.25 | 58.05 | 57.45 | 58.47 | 60.27 | 61.14 | 3.69 |
| M3 | 61.59 | 59.33 | 58.01 | 57.31 | 58.39 | 60.22 | 62.01 | 4.70 |

### `FGA_jump2` (n = 572,558)

| model | <= -20 | -19..-10 | -9..-4 | -3..+3 | +4..+9 | +10..+19 | >= +20 | span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 37.41 | 38.38 | 38.42 | 38.19 | 38.84 | 40.06 | 40.11 | **2.70** |
| M1 | 39.66 | 39.52 | 38.93 | 38.10 | 38.19 | 38.71 | 37.63 | **2.03** |
| M2 | 39.68 | 39.52 | 38.93 | 38.10 | 38.18 | 38.70 | 37.66 | 2.02 |
| M3 | 40.22 | 39.60 | 38.89 | 38.00 | 38.12 | 38.69 | 38.38 | 2.22 |

### `FGA_3` (n = 851,008)

| model | <= -20 | -19..-10 | -9..-4 | -3..+3 | +4..+9 | +10..+19 | >= +20 | span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 31.75 | 32.60 | 33.13 | 33.95 | 34.64 | 35.21 | 35.32 | **3.56** |
| M1 | 33.28 | 33.39 | 33.52 | 33.94 | 34.18 | 34.25 | 33.54 | **0.97** |
| M2 | 33.30 | 33.40 | 33.53 | 33.94 | 34.18 | 34.24 | 33.51 | 0.94 |
| M3 | 34.60 | 33.59 | 33.46 | 33.58 | 34.08 | 34.32 | 35.11 | 1.65 |

### How much of the `score_diff` effect is team strength

| class | M0 span | M1 span | **removed by pregame ratings alone** | M2 span | M3 span |
|---|---:|---:|---:|---:|---:|
| `FGA_rim` | 7.64 | 3.80 | **50.3%** | 3.69 | 4.70 |
| `FGA_jump2` | 2.70 | 2.03 | **24.8%** | 2.02 | 2.22 |
| `FGA_3` | 3.56 | 0.97 | **72.9%** | 0.94 | 1.65 |

Read against section 0, the full accounting of the round-1 feature is:

| class | apparent effect (round 1's feature) | manufactured by the leak | pregame team strength | left for game state |
|---|---:|---:|---:|---:|
| `FGA_rim` | 20.12 pp | 12.49 (62.1%) | 3.84 (19.1%) | **3.80 (18.9%)** |
| `FGA_jump2` | 16.21 pp | 13.51 (83.3%) | 0.67 (4.1%) | **2.03 (12.5%)** |
| `FGA_3` | 21.64 pp | 18.08 (83.5%) | 2.60 (12.0%) | **0.97 (4.5%)** |

Three separate things follow.

1. **On threes -- the class where the loop did the most damage -- the whole
   thing is leak plus team strength.** 95.5% of the apparent effect is one or
   the other, and the residual curve is flat to within 0.97 pp.
2. **The M1 curve is U-shaped at the rim, not monotone.** A team down 20+
   makes rim shots at 60.9% adjusted, the same as a team up 20+ at 61.3%, and
   the MINIMUM is at the tied bucket. That is a blowout effect on both sides of
   the scoreboard, and no monotone function of the margin can represent it --
   which is the direct argument against a continuous margin term.
3. **M3 > M1 on every class.** Conditioning on WHO is shooting makes the margin
   effect bigger, not smaller: in a blowout the bench shoots, the shooter block
   already knows that, and once it is held fixed the remaining margin effect is
   the shot-quality part. This is suppression, not instability -- it is stable
   in sign and size across all four seasons (section 4).

### By minute bucket (classes pooled)

Game clock; `gsr` is seconds left in regulation, OT rows kept separate.

| minute bucket | n | M0 span | M1 span | M2 span | M3 span |
|---|---:|---:|---:|---:|---:|
| H1 20-10 | 563,325 | 4.77 | 3.61 | 3.51 | 4.47 |
| H1 10-0 | 557,442 | 6.79 | 1.16 | 1.17 | **0.98** |
| H2 20-10 | 561,118 | 7.16 | 1.49 | 1.46 | 1.80 |
| H2 10-5 | 267,199 | 6.42 | 1.86 | 1.83 | 2.86 |
| H2 5-2 | 153,944 | 5.51 | 2.97 | 2.97 | 3.07 |
| **H2 2-0** | 120,378 | 10.37 | 10.02 | 9.99 | **9.33** |
| **OT** | 17,657 | 16.20 | 14.88 | 14.85 | **14.34** |

This is the central result of the whole decomposition. **Through 38 minutes of
regulation the adjusted margin effect is 1-3 pp; in the last two minutes it is
9.3 pp and in overtime 14.3 pp**, and conditioning on pregame ratings barely
touches those two cells (10.37 -> 10.02, 16.20 -> 14.88). The state effect is
not "the leading team shoots better"; it is an END-GAME effect that happens to
correlate with the margin.

H1 20-10 is the one mid-game cell with a large M3 span (4.47). It is also the
cell where the margin is smallest and most noise-driven; the M0 -> M1 collapse
there is the smallest of any cell (4.77 -> 3.61), which is the signature of a
bucket boundary carrying early-game score noise rather than a state effect.

By minute x class (M3 spans; every cell above 2,000 attempts):

| minute bucket | `FGA_rim` | `FGA_jump2` | `FGA_3` |
|---|---:|---:|---:|
| H1 20-10 | 4.66 | 7.68 | 9.08 |
| H1 10-0 | 3.43 | 1.46 | 1.69 |
| H2 20-10 | 4.50 | 1.36 | 0.98 |
| H2 10-5 | 5.26 | 3.28 | 3.24 |
| H2 5-2 | 6.34 | 4.44 | 3.48 |
| H2 2-0 | 11.93 | 5.90 | 7.11 |
| OT | 13.44 | 21.81 | 5.05 |

### By offensive-team pregame rating quintile

| quintile | mean net rating (centred) | n | M0 span | M1 span | M3 span |
|---|---:|---:|---:|---:|---:|
| Q1 | -10.87 | 448,270 | 3.87 | 2.31 | 3.10 |
| Q2 | -3.83 | 448,177 | 4.18 | 1.65 | 2.44 |
| Q3 | +0.51 | 448,227 | 4.99 | 1.91 | 2.66 |
| Q4 | +4.92 | 448,203 | 4.58 | 1.73 | 2.99 |
| Q5 | +13.11 | 448,186 | 4.86 | 2.64 | 4.01 |

The residual effect is 1.65-2.64 pp in every quintile with no ordering in the
middle three: the surviving state effect is not a disguised team-quality term.
The mild U (Q1 and Q5 highest) is the blowout shape again -- the quintiles that
are in blowouts most often.

### By chance type

| chance type | n | M0 span | M1 span | M3 span |
|---|---:|---:|---:|---:|
| first (half-court) | 1,579,625 | 4.79 | 1.36 | 2.22 |
| transition, first chance | 363,563 | 8.53 | 5.14 | 6.09 |
| continuation (after OREB) | 297,875 | 3.59 | 2.88 | 3.62 |

Transition carries the largest residual, consistent with the blowout reading:
the extra rim efficiency in a lopsided game arrives in transition.

---

## 2. (b) Garbage time -- and it is NOT carried by the on-floor five

Lineup rows exist from 2024 (L13). On the **1,202,616** attempts of 2024-2025
that carry all ten on-floor ids, M4 adds the defensive five's as-of allowed
rates (`def5_rim_allow_c`, `def5_three_allow_c`, shrunk at the round-1 fitted
strength of 100 pseudo-attempts).

| class | M0 span | M1 span | M3 span | **M4 span (+ def five)** | M3 -> M4 |
|---|---:|---:|---:|---:|---:|
| `FGA_rim` | 8.11 | 4.11 | 4.88 | **4.78** | -0.10 |
| `FGA_jump2` | 3.64 | 2.19 | 2.72 | **2.73** | +0.01 |
| `FGA_3` | 3.87 | 1.16 | 1.29 | **1.28** | -0.01 |

**The on-floor five explain essentially none of it (<= 0.10 pp of a 4.9 pp
span).** The pre-registration's hypothesis -- that garbage time is a lineup-
quality effect that rotation should carry -- is measured and rejected. That
matches round 1 section 8's independent verdict that the defence is TEAM-LEVEL.

Average marginal effect of a garbage-time indicator, in pp on make probability,
across a 3 x 3 threshold grid (all nine cells above the 2,000-attempt floor):

| indicator | `FGA_rim` n / M3 / M4 | `FGA_jump2` n / M3 / M4 | `FGA_3` n / M3 / M4 |
|---|---|---|---|
| abs>=10 & gsr<=300 | 28,286 / +5.05 / +5.02 | 15,277 / +1.07 / +1.06 | 29,662 / -0.96 / -0.96 |
| abs>=10 & gsr<=480 | 44,615 / +4.54 / +4.51 | 24,985 / +1.70 / +1.69 | 45,296 / -0.30 / -0.30 |
| abs>=10 & gsr<=600 | 55,524 / +4.28 / +4.25 | 31,326 / +1.65 / +1.64 | 55,841 / -0.01 / -0.01 |
| **abs>=15 & gsr<=300** | 17,749 / +5.19 / +5.14 | 10,149 / +1.00 / +0.99 | 18,981 / -0.20 / -0.20 |
| **abs>=15 & gsr<=480** | 27,907 / **+4.79** / +4.74 | 16,297 / **+1.74** / +1.73 | 28,942 / **+0.53** / +0.53 |
| **abs>=15 & gsr<=600** | 34,531 / +4.57 / +4.53 | 20,112 / +1.76 / +1.75 | 35,456 / +0.63 / +0.63 |
| abs>=20 & gsr<=300 | 10,593 / +5.01 / +4.91 | 6,300 / +1.32 / +1.30 | 11,720 / +0.78 / +0.77 |
| abs>=20 & gsr<=480 | 16,618 / +4.87 / +4.78 | 10,034 / +2.29 / +2.28 | 17,641 / +1.18 / +1.17 |
| abs>=20 & gsr<=600 | 20,347 / +4.84 / +4.77 | 12,240 / +2.11 / +2.09 | 21,390 / +1.21 / +1.20 |

The effect is **+4.3 to +5.2 pp on rim shots in every one of the nine cells**,
+1.0 to +2.3 pp on two-point jumpers, and ~0 on threes. It is flat in the
thresholds, which is what makes an indicator the right parametrisation: the
answer does not depend on where the line is drawn, so drawing it is not tuning.

---

## 3. (c) The end-game tactical effect

### It lives mostly in the SHOT MIX

Every attempt, by minute bucket x margin. `mean_chance_elapsed_s` is the time
on the ball at release -- the "rushed" measure.

| minute | margin | n | 3PA share % | rim share % | elapsed s | make % |
|---|---|---:|---:|---:|---:|---:|
| H2 20-10 | trail 10+ | 125,404 | 35.85 | 37.67 | 14.66 | 43.63 |
| H2 20-10 | trail 4-9 | 99,420 | 35.69 | 38.35 | 14.83 | 44.86 |
| H2 20-10 | tied | 18,938 | 34.99 | 38.86 | 15.00 | 45.51 |
| H2 20-10 | lead 4-9 | 93,041 | 36.08 | 38.54 | 14.91 | 45.85 |
| H2 20-10 | lead 10+ | 110,757 | 37.73 | 39.13 | 14.24 | 47.69 |
| H2 5-2 | trail 10+ | 42,231 | 39.82 | 36.27 | 12.80 | 43.33 |
| H2 5-2 | trail 4-9 | 23,825 | 37.77 | 39.07 | 14.14 | 44.40 |
| H2 5-2 | tied | 4,504 | 34.24 | 39.30 | 16.94 | 44.56 |
| H2 5-2 | lead 4-9 | 21,313 | 33.35 | 39.92 | 18.31 | 45.57 |
| H2 5-2 | lead 10+ | 36,240 | 36.56 | 40.99 | 16.58 | 48.22 |
| **H2 2-0** | trail 10+ | 36,599 | 45.62 | 35.08 | 9.63 | 42.32 |
| **H2 2-0** | **trail 4-9** | 27,964 | **51.97** | 34.07 | **8.39** | **40.09** |
| **H2 2-0** | trail 1-3 | 12,663 | 41.68 | 37.16 | 12.54 | 38.54 |
| **H2 2-0** | tied | 4,213 | 32.85 | 35.56 | 16.49 | 37.27 |
| **H2 2-0** | **lead 1-3** | 7,676 | **28.40** | 41.43 | **20.19** | 43.33 |
| **H2 2-0** | lead 4-9 | 9,052 | 27.03 | 49.18 | 18.99 | 50.19 |
| **H2 2-0** | lead 10+ | 22,211 | 37.37 | 42.11 | 16.88 | 45.99 |

In the last two minutes, a team trailing by 4-9 takes **52.0%** of its attempts
from three at **8.4 s** on the ball; a team leading by 1-3 takes **28.4%** from
three at **20.2 s**. That 23.6 pp swing in 3PA share and 11.8 s swing in shot
clock is the end-game tactical effect, and **it belongs to the possession-
outcome model (shot mix) and the clock model (duration), not to fg_make.**

### The within-class residual that DOES belong to fg_make

Average marginal effect of an end-game indicator, in pp on make probability:

| indicator | `FGA_rim` n / M0 / M3 | `FGA_jump2` n / M0 / M3 | `FGA_3` n / M0 / M3 |
|---|---|---|---|
| gsr<=300 & trail 1-9 | 28,655 / +0.04 / +0.40 | 15,420 / +0.70 / +0.45 | 33,499 / -5.25 / **-5.17** |
| **gsr<=120 & trail 1-9** | 14,234 / +0.56 / +0.96 | 6,582 / -0.13 / -0.51 | 19,811 / -8.03 / **-7.78** |
| gsr<=120 & trail 1-6 | 9,652 / -0.91 / -0.78 | 4,713 / -0.75 / -1.33 | 12,721 / -8.19 / **-8.10** |
| gsr<=60 & trail 1-6 | 6,209 / -1.01 / -0.91 | 2,702 / -2.16 / -2.87 | 9,383 / -10.71 / **-10.57** |
| **gsr<=120 & lead 1-9** | 7,632 / +4.01 / +2.75 | 4,469 / -1.82 / **-2.99** | 4,627 / -2.34 / **-3.28** |
| gsr<=300 & lead 1-9 | 21,020 / +1.86 / +0.74 | 13,686 / -0.30 / -1.37 | 16,044 / -0.76 / -1.71 |
| gsr<=120 & trail 10+ | 12,839 / +3.30 / +6.49 | 7,062 / -0.35 / +1.69 | 16,698 / -4.83 / -2.25 |

**A trailing team's three-point make probability in the last two minutes is
7.8 pp below what every pregame feature predicts, and 10.6 pp below inside the
last minute.** Rim and jumper are within +/-1.3 pp over the same cells. The
mirror is smaller but real: a team leading by 1-9 inside two minutes is +2.8 pp
at the rim (uncontested finishes against a team that has stopped fouling into
the shot) and -3.0/-3.3 pp on jumpers and threes.

These are large, well-powered, and survive every pregame block -- they are the
"12 points of total" the L23 ablation lost when `score_diff` was deleted
outright, and they are why deletion is not the fix.

---

## 4. Per-season stability

M0 and M3 spans on the corrected margin, per season, per class:

| season | rim M0 / M3 | jump2 M0 / M3 | three M0 / M3 |
|---|---|---|---|
| 2022 (n 171,643 / 132,085 / 184,892) | 7.08 / 4.33 | 3.87 / 1.70 | 4.14 / 2.51 |
| 2023 (n 185,380 / 137,741 / 193,502) | 8.08 / 5.38 | 1.69 / 3.64 | 2.99 / 2.12 |
| 2024 (n 225,020 / 153,657 / 225,729) | 7.64 / 4.66 | 2.76 / 2.62 | 4.00 / 1.13 |
| 2025 (n 235,454 / 149,075 / 246,885) | 8.32 / 5.10 | 4.19 / 3.01 | 3.75 / 1.73 |

The rim result is the stable one (M0 7.1-8.3, M3 4.3-5.4 in all four seasons).
`FGA_jump2` is the noisiest (2023's M3 exceeds its M0), which is consistent
with its being the smallest and least state-sensitive class, and is a reason to
keep any jumper-side state term modest.

---

## 5. What this says for round 2

1. **`score_diff` as round 1 built it cannot be adopted at any weight.** It is
   post-outcome. The round-1 `C_plus_state` arms must be re-scored with the
   corrected column before any of their log-loss numbers mean anything.
2. **A continuous margin term is the wrong shape even when corrected.** The
   adjusted curve is U-shaped at the rim, flat to 0.97 pp on threes, and
   concentrated entirely in the last two minutes and overtime.
3. **Two indicators carry nearly all of the real effect**, both flat in their
   thresholds and both self-limiting in a simulation because they saturate:
   - garbage time, `|margin| >= 15 AND under 8:00 left in regulation`:
     +4.79 pp rim, +1.74 pp jumper, +0.53 pp three;
   - end-game trailing, `under 2:00 left AND behind by 1-9`:
     -7.78 pp three, +0.96 pp rim, -0.51 pp jumper;
   with an end-game LEADING mirror (`under 2:00, ahead by 1-9`) at +2.75 rim /
   -2.99 jumper / -3.28 three on thinner but still powered cells (4,469-7,632).
4. **The on-floor five are not the mechanism** and no lineup feature is needed.
5. **The shot-mix half of the end-game effect is not this model's** and must
   not be compensated for here (`CLAUDE.md`, bottom-up rule).

The round-2 arms, folds, metrics, closed-loop gate and decision rule built on
this evidence are pre-registered in `docs/models/fg_make/experiments.md`
section 13, committed before the round ran.
