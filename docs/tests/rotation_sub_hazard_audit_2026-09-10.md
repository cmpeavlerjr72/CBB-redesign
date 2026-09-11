# Rotation substitution-hazard audit (round-4 evidence) -- 2026-09-10

Worker: Opus. Inputs: `data/processed/possessions/possessions_{2024,2025}.parquet`
(on-floor sets, `start_reason`, `off_team_fouls`/`def_team_fouls`),
`data/raw/cbbd/pbp/plays_{season}.parquet` (`PersonalFoul`, the three timeout
play types), `data/raw/hoopr/player_box/player_box_{season-1}.parquet` via
`data/processed/player_crosswalk.parquet` (prior-season minutes share).
Scripts: `scripts/diag_rotation_sub_hazard.py` (`--mode sweep`, `--mode reach`).
JSON: `data/processed/models/rotation/sub_hazard_audit_2026-09-10.json`,
`.../sub_hazard_reachability_2026-09-10.json`.

Coverage: **8,502 team-games (2024)** and **10,640 team-games (2025)**, every
game with a complete on-floor set on every possession. Cells with fewer than 300
risk-set rows are marked `UP(n)` and are UNDERPOWERED -- neither signal nor
absence of signal. Every hazard carries a 95% Wilson score interval.

---

## 0. What this audit is for

L25 closed rounds 1-3 with a family verdict: *"The rotation family itself
(Dirichlet share plus scheduler, or donor-lineup resampling) cannot produce the
two structural facts of real rotations, starters reset at the second-half tip
and starters kept late in close games; round 4 changes family to per-player
substitution hazards, with those two facts as pre-registered gates."* It also
fixed the protocol: *"before fitting any knob, compute the fitted log-odds
separation between the classes the knob must move apart in the target state and
report the target cell's reachable range over the whole grid; if the target is
outside it, the family is wrong and no fitting will find it."*

This document is that computation for the new family, done **before** the
round-4 pre-registration (`docs/models/rotation/experiments.md` section 10) was
written, and before any arm was fitted.

### 0.1 What a hazard is here, and why it is measured this way

At every possession boundary k of a team-game:

* the **out** risk set is the five on the floor at possession k-1; the event is
  "not on the floor at possession k";
* the **in** risk set is every candidate not on the floor at k-1 and not fouled
  out; the event is "on the floor at k".

This is the same substitution event `rotation.build_hazard_training` derives --
CBBD carries **no** `Substitution` rows in 2024 and only ~42 per game in 2025
against ~145 in 2026 (`features.md` section 3), so substitutions are derived
from on-floor set transitions at possession-boundary resolution, which is also
the resolution the engine acts at.

The audit is **descriptive**: its candidate pool is every player who appears in
the team-game and its foul state is that game's own `PersonalFoul` events. Both
are contemporaneous with the game and are correct for a *measurement* of
coaching behaviour. The bake-off's own training path
(`rotation_v4.build_sub_training`) uses the as-of candidate pool and never reads
a simulated game's own fouls -- the rule established by `experiments.md`
section 5.

### 0.2 A defect found and fixed inside this audit

The first sweep bucketed the exit hazard by the player's **current stint
length** computed from on/off changes through possession k. The change at k *is*
the label at boundary k, so the feature was a copy of the answer: bucket "<1
min" came back at 0.244 and every longer bucket at exactly 0.0000. The stint is
now built from changes through k-1 only (`run_prev` in the sweep), the table in
section 5 is the corrected one, and nothing else in the audit used that column.
The model path was never affected -- `rotation_v4.build_sub_training` resets its
stint clock *after* building the features for the boundary, and its rows were
built after this fix.

---

## 1. The out and in hazards by time cell and margin band

Nine time cells, the profile `rotation_close_game_audit_2026-09-10.md` uses plus
overtime. Each cell shows **starter / bench**. `n` is the starter risk-set size;
bench risk sets are 2-4x larger on the in side and ~0.4x on the out side.

### 1.1 Season 2024 -- OUT hazard (on the floor at k-1, off at k)

| time cell | \|m\|<=5 | \|m\| 6-15 | \|m\|>15 | n (starter, by band) |
|---|---|---|---|---|
| H1 20:00-10:00 | 0.0324 / 0.0398 | 0.0553 / 0.0478 | 0.0591 / 0.0625 | 826,868 / 225,475 / 5,411 |
| H1 10:00-00:00 | 0.0367 / 0.0714 | 0.0369 / 0.0703 | 0.0396 / 0.0621 | 449,144 / 427,525 / 76,791 |
| **H2 20:00-16:00** | **0.0178 / 0.2449** | **0.0189 / 0.2290** | **0.0221 / 0.1825** | 205,024 / 241,879 / 87,717 |
| H2 16:00-12:00 | 0.0509 / 0.0406 | 0.0524 / 0.0394 | 0.0608 / 0.0348 | 130,129 / 159,727 / 74,499 |
| H2 12:00-08:00 | 0.0356 / 0.0644 | 0.0354 / 0.0656 | 0.0430 / 0.0557 | 123,930 / 153,185 / 80,914 |
| H2 08:00-04:00 | 0.0248 / 0.0699 | 0.0265 / 0.0695 | 0.0396 / 0.0489 | 133,480 / 167,871 / 95,271 |
| H2 04:00-02:00 | 0.0185 / 0.0688 | 0.0235 / 0.0666 | 0.0683 / 0.0360 | 67,069 / 85,709 / 40,725 |
| H2 02:00-00:00 | 0.0364 / 0.0895 | 0.0378 / 0.0568 | 0.0933 / 0.0244 | 92,271 / 105,980 / 25,589 |
| OT | 0.0218 / 0.0638 | 0.0328 / 0.0526 | UP(0) | 39,669 / 4,297 / 0 |

### 1.2 Season 2024 -- IN hazard (off the floor at k-1, on at k)

| time cell | \|m\|<=5 | \|m\| 6-15 | \|m\|>15 | n (starter, by band) |
|---|---|---|---|---|
| H1 20:00-10:00 | 0.0630 / 0.0287 | 0.0717 / 0.0434 | 0.0849 / 0.0318 | 218,872 / 124,830 / 4,509 |
| H1 10:00-00:00 | 0.0804 / 0.0386 | 0.0780 / 0.0348 | 0.0656 / 0.0301 | 219,961 / 213,635 / 43,569 |
| **H2 20:00-16:00** | **0.2446 / 0.0211** | **0.2296 / 0.0202** | **0.1851 / 0.0185** | 23,336 / 28,945 / 12,455 |
| H2 16:00-12:00 | 0.0606 / 0.0526 | 0.0594 / 0.0475 | 0.0503 / 0.0394 | 67,895 / 87,086 / 49,064 |
| H2 12:00-08:00 | 0.0877 / 0.0306 | 0.0866 / 0.0280 | 0.0705 / 0.0249 | 66,304 / 84,744 / 59,537 |
| H2 08:00-04:00 | 0.0852 / 0.0251 | 0.0827 / 0.0260 | 0.0539 / 0.0287 | 48,684 / 64,014 / 58,630 |
| H2 04:00-02:00 | 0.0840 / 0.0196 | 0.0766 / 0.0258 | 0.0239 / 0.0591 | 19,142 / 28,014 / 37,914 |
| H2 02:00-00:00 | 0.1102 / 0.0450 | 0.0678 / 0.0450 | 0.0071 / 0.0836 | 25,424 / 38,356 / 48,449 |
| OT | 0.0857 / 0.0294 | 0.0696 / 0.0464 | UP(0) | 9,027 / 1,121 / 0 |

### 1.3 Season 2025 -- the same two tables

| time cell | OUT \|m\|<=5 | OUT 6-15 | OUT >15 | IN \|m\|<=5 | IN 6-15 | IN >15 |
|---|---|---|---|---|---|---|
| H1 20:00-10:00 | 0.0350 / 0.0412 | 0.0586 / 0.0473 | 0.0642 / 0.0653 | 0.0644 / 0.0298 | 0.0702 / 0.0438 | 0.0819 / 0.0369 |
| H1 10:00-00:00 | 0.0387 / 0.0722 | 0.0400 / 0.0723 | 0.0424 / 0.0646 | 0.0799 / 0.0395 | 0.0793 / 0.0365 | 0.0683 / 0.0305 |
| H2 20:00-16:00 | 0.0196 / 0.2437 | 0.0212 / 0.2163 | 0.0231 / 0.1887 | 0.2447 / 0.0222 | 0.2176 / 0.0217 | 0.1872 / 0.0192 |
| H2 16:00-12:00 | 0.0540 / 0.0400 | 0.0551 / 0.0400 | 0.0662 / 0.0363 | 0.0607 / 0.0523 | 0.0596 / 0.0467 | 0.0512 / 0.0419 |
| H2 12:00-08:00 | 0.0371 / 0.0660 | 0.0385 / 0.0667 | 0.0447 / 0.0593 | 0.0868 / 0.0321 | 0.0861 / 0.0302 | 0.0714 / 0.0260 |
| H2 08:00-04:00 | 0.0267 / 0.0690 | 0.0294 / 0.0679 | 0.0421 / 0.0500 | 0.0845 / 0.0259 | 0.0845 / 0.0260 | 0.0549 / 0.0293 |
| H2 04:00-02:00 | 0.0206 / 0.0703 | 0.0247 / 0.0663 | 0.0698 / 0.0365 | 0.0865 / 0.0201 | 0.0798 / 0.0246 | 0.0252 / 0.0577 |
| H2 02:00-00:00 | 0.0374 / 0.0905 | 0.0376 / 0.0602 | 0.0928 / 0.0258 | 0.1105 / 0.0429 | 0.0667 / 0.0448 | 0.0076 / 0.0805 |
| OT | 0.0235 / 0.0554 | 0.0251 / 0.0456 | UP(0) | 0.0782 / 0.0277 | 0.0525 / 0.0364 | UP(0) |

**The two seasons agree cell for cell.** The largest starter-side difference over
the 54 (cell x band x direction) pairs with adequate exposure is 1.5 pp and the
median is 0.3 pp, which reproduces the finding of the round-3 audit section 7.5:
there is no season effect to model, and fitting on 2024 while gating on 2025 is
measuring one behaviour. Every feature choice in the round-4 design was taken
from the **2024** tables; the 2025 columns are reported for that agreement check
and for nothing else, and every model is fitted strictly on training data.

**Three shapes the tables show that a linear time term cannot represent.**

1. A starter's exit hazard is **non-monotone in time**: 0.035 (H1 20-10) ->
   0.039 (H1 10-0) -> **0.020** (H2 20-16) -> **0.054** (H2 16-12) -> 0.037 ->
   0.027 -> 0.021 -> 0.037 (H2 2-0), close band, 2025. The dip at the start of
   the second half and the spike four minutes later are the coach's first
   substitution wave, and they are 2.7x apart.
2. A bench player's exit hazard in H2 20:00-16:00 is **0.244**, against 0.041 in
   H1 20:00-10:00 -- a 6x jump in one cell.
3. The blowout reverses the starter/bench ordering entirely in the last two
   minutes: at \|m\|>15, H2 02:00-00:00, a starter's exit hazard is 0.093 and a
   bench player's is 0.026, and the in hazards are 0.008 (starter) against 0.081
   (bench). Garbage time is not a weaker version of the close game; it is the
   opposite sign.

---

## 2. The second-half reset, measured directly

### 2.1 By stoppage type (`start_reason`, which is exactly the engine's `prev_end`)

Season 2024; every hazard ± its Wilson interval half-width is under 0.2 pp
except `other`.

| `start_reason` | OUT starter | OUT bench | IN starter | IN bench | n (OUT starter) |
|---|---:|---:|---:|---:|---:|
| made_FG | 0.0265 | 0.0423 | 0.0523 | 0.0246 | 1,549,795 |
| DREB | 0.0276 | 0.0420 | 0.0526 | 0.0255 | 1,480,907 |
| TOV | 0.0423 | 0.0631 | 0.0802 | 0.0379 | 687,015 |
| made_FT | 0.0946 | 0.1392 | 0.1716 | 0.0902 | 361,214 |
| **period_start** | **0.0137** | **0.8992** | **0.9010** | **0.0193** | 30,184 |
| other | 0.0460 | 0.0687 | 0.0851 | 0.0435 | 17,034 |

Season 2025 reproduces it: `period_start` 0.0185 / 0.8877 / 0.8879 / 0.0235.

**This single row is the structural fact three rounds could not produce.** At a
period boundary a bench player who is on the floor leaves with probability
**0.90** and a starter who is off the floor comes on with probability **0.90**.
It is not a tendency; it is the largest conditional probability anywhere in the
rotation data, and it is available to the engine for free, because
`start_reason` and the engine's `prev_end` are the same six-level code
(`engine.state.PREV_END_LEVELS`).

### 2.2 The realised reset

| quantity | 2024 | 2025 |
|---|---:|---:|
| starter share of the five at the FIRST possession of period 2, \|m\|<=5 | 0.9707 | 0.9687 |
| same, \|m\| 6-15 | 0.9669 | 0.9608 |
| same, \|m\|>15 | 0.9575 | 0.9518 |
| P(starter on the floor at the H2 tip \| he was on at the last possession of H1) | 0.9888 | 0.9858 |
| P(starter on the floor at the H2 tip \| he was OFF at the last possession of H1) | 0.9258 | 0.9191 |
| P(bench player on the floor at the H2 tip \| he was on at the end of H1) | 0.0670 | 0.0732 |
| P(bench player on the floor at the H2 tip \| he was off at the end of H1) | 0.0165 | 0.0197 |
| starter share over the whole H2 20:00-16:00 window, \|m\|<=5 / 6-15 / >15 | 0.9073 / 0.9011 / 0.8795 | 0.9019 / 0.8933 / 0.8740 |

Risk sets for the transition rows: 25,400 / 14,596 / 14,596 / 27,914 (2024) and
32,241 / 18,499 / 18,499 / 34,701 (2025) player-boundaries.

The tip is 0.95-0.97 and the four-minute window that follows averages 0.87-0.91,
so the reset is a *point event* that decays, not a level. Three rounds of arms
produced 0.72-0.83 over the window and were never measured at the tip at all.

### 2.3 Timeouts -- observable, and deliberately NOT a model feature

| | OUT starter | OUT bench | IN starter | IN bench |
|---|---:|---:|---:|---:|
| no timeout at this stoppage (2024) | 0.0313 | 0.0567 | 0.0684 | 0.0291 |
| a timeout at this stoppage (2024) | 0.1364 | 0.1942 | 0.2433 | 0.1286 |
| no timeout (2025) | 0.0335 | 0.0573 | 0.0685 | 0.0299 |
| a timeout (2025) | 0.1438 | 0.1944 | 0.2407 | 0.1297 |

n (2024): 3.96M / 162k on the OUT-starter side. A timeout multiplies every
hazard by 3-4x and is the single strongest non-period-boundary stoppage signal
in the data.

**It is excluded from the round-4 design.** The engine has no timeout model, so
a hazard conditioned on timeouts could be fitted offline and could not be
evaluated in simulation -- the same class of exclusion as `is_transition` in
`features.md` section 3. What the engine *does* carry is `prev_end`, and
`made_FT` (0.095 / 0.139 / 0.172 / 0.090) already carries the part of the
timeout effect that arrives through free-throw stoppages. The cost of the
exclusion is that the model's substitutions are slightly more uniform in time
than real ones; it is recorded here rather than discovered later.

---

## 3. Fouls

Season 2025; the risk sets are 0.8M-3.3M rows except the five-foul row.

| own fouls | OUT starter | OUT bench | IN starter | IN bench | n (OUT starter) |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0331 | 0.0573 | 0.0833 | 0.0312 | 2,307,341 |
| 1 | 0.0397 | 0.0717 | 0.0799 | 0.0392 | 1,443,836 |
| 2 | 0.0424 | 0.0672 | 0.0665 | 0.0388 | 747,033 |
| 3 | 0.0439 | 0.0683 | 0.0633 | 0.0427 | 326,157 |
| 4 | **0.0695** | 0.0893 | **0.0627** | 0.0463 | 89,061 |
| 5 (fouled out) | 0.5539 | 0.6335 | -- | -- | 538 |

A starter's exit hazard doubles between 0 and 4 fouls and his entry hazard falls
by a quarter; the effect is monotone on both sides in both seasons. The
five-foul row is small (538 starter rows) because the eviction is usually
immediate; the engine enforces it as a rule, not a hazard.

Own team fouls are a much weaker and non-monotone signal (2025, OUT starter:
0.0322 / 0.0448 / 0.0391 / 0.0400 / 0.0467 across 0-2 / 3-5 / 6-8 / 9-11 / 12+,
risk sets 87k-2.1M). It stays in the design as one linear term and nothing more
is claimed for it.

---

## 4. Fatigue: stint length and minutes already played in the half

### 4.1 Current stint (out) / current rest (in), season 2025

| current state length | OUT starter | OUT bench | IN starter | IN bench |
|---|---:|---:|---:|---:|
| < 1 min | 0.0173 | 0.0357 | 0.0445 | 0.0202 |
| 1-2 min | 0.0199 | 0.0462 | 0.0726 | 0.0235 |
| 2-3 min | 0.0298 | 0.0651 | 0.0978 | 0.0339 |
| 3-4 min | 0.0438 | 0.0827 | 0.1080 | 0.0477 |
| 4-5 min | 0.0544 | 0.0938 | 0.1075 | 0.0573 |
| 5+ min | 0.0467 | 0.0805 | 0.0685 | 0.0327 |

Risk sets 0.16M-2.3M per cell. Monotone through 4-5 minutes on both sides and in
both seasons; the 5+ bucket pools genuine long shifts (a starter playing a whole
half) with long rests and turns over.

### 4.2 Minutes already played in the current half, season 2025

| half minutes so far | OUT starter | OUT bench | IN starter | IN bench |
|---|---:|---:|---:|---:|
| < 5 | 0.0256 | 0.0614 | 0.0782 | 0.0304 |
| 5-10 | 0.0538 | 0.0734 | 0.0724 | 0.0514 |
| 10-15 | 0.0395 | 0.0484 | 0.0801 | 0.0773 |
| 15-20 | 0.0249 | 0.0297 | 0.0940 | 0.1256 |
| 20-25 | 0.0133 | UP-ish 0.0287 (n=2,369) | 0.1630 | 0.1575 |
| 25+ | 0.0252 | 0.0394 (n=762) | 0.1451 | 0.1667 |

Also non-monotone: the exit hazard peaks in the 5-10 minute band (the first
substitution wave) and then falls away, because a player still on the floor
after fifteen minutes of a half is one the coach has decided not to rest.

---

## 5. Role: as-of minutes share, and prior-season minutes share

Quintiles of the as-of share of the team's five on-floor slots (strictly earlier
games only); season 2025, quintile edges 0.063 / 0.265 / 0.478 / 0.662.

| as-of share quintile | OUT starter | OUT bench | IN starter | IN bench |
|---|---:|---:|---:|---:|
| Q1 | 0.0426 | 0.0638 | 0.0663 | 0.0172 |
| Q2 | 0.0611 | 0.0786 | 0.0476 | 0.0260 |
| Q3 | 0.0546 | 0.0660 | 0.0584 | 0.0426 |
| Q4 | 0.0446 | 0.0518 | 0.0744 | 0.0596 |
| Q5 | 0.0294 | 0.0376 | 0.0940 | 0.0716 |

Monotone from Q2 up on every column: the higher a player's as-of share, the less
likely he leaves and the more likely he enters. Q1 is a small, odd population
(players with almost no prior minutes) and is reported as measured.

Prior-season minutes share (hoopR `player_box` of season-1 through the
crosswalk), same season, quintile edges 0.000 / 0.138 / 0.350 / 0.556, with a
sixth bucket for players with no prior season at all (a fifth of the rows):

| prior-season quintile | OUT starter | OUT bench | IN starter | IN bench |
|---|---:|---:|---:|---:|
| P1 | 0.0425 | 0.0696 | 0.0680 | 0.0274 |
| P2 | 0.0420 | 0.0647 | 0.0670 | 0.0263 |
| P3 | 0.0438 | 0.0671 | 0.0693 | 0.0388 |
| P4 | 0.0410 | 0.0599 | 0.0758 | 0.0455 |
| P5 | 0.0331 | 0.0550 | 0.0857 | 0.0465 |
| no prior season | 0.0390 | 0.0660 | 0.0719 | 0.0297 |

The gradient exists, runs the same direction as the as-of share, and is **about
half its size** (P5 - P1: -0.9 pp on the starter exit hazard and +1.8 pp on the
starter entry hazard, against -1.3 pp and +2.8 pp for the as-of share, and the
as-of share's own Q5 - Q2 span is larger still). Two reasons to leave it out of
the round-4 design, both recorded before the pre-registration:

1. It is a marginal, same-signed, smaller-magnitude proxy for a variable the
   design already carries; a bake-off arm would be testing a near-duplicate.
2. It is not expressible in the engine without a new per-roster-slot input
   array, which means rebuilding `data/processed/models/engine/arrays_F2_2025.npz`
   -- a file other workers are reading. Worker discipline forbids overwriting it
   and a versioned sibling would fork the engine's inputs for one feature worth
   under 2 pp of hazard.

If a later round wants it, this table is the reason to expect very little.

---

## 6. Occupancy -- the quantity the gate reads

Starter share of on-floor slots. This is the same construction the gate uses,
computed here over every complete team-game rather than the 1,600-game bake-off
subset.

| time cell | 2024 \|m\|<=5 / 6-15 / >15 | 2025 \|m\|<=5 / 6-15 / >15 |
|---|---|---|
| H1 20:00-10:00 | 0.7878 / 0.6347 / 0.5518 | 0.7791 / 0.6272 / 0.5422 |
| H1 10:00-00:00 | 0.6745 / 0.6692 / 0.6371 | 0.6665 / 0.6649 / 0.6297 |
| H2 20:00-16:00 | 0.9073 / 0.9011 / 0.8795 | 0.9019 / 0.8933 / 0.8740 |
| H2 16:00-12:00 | 0.6462 / 0.6355 / 0.5873 | 0.6378 / 0.6279 / 0.5694 |
| H2 12:00-08:00 | 0.6599 / 0.6527 / 0.5817 | 0.6577 / 0.6476 / 0.5688 |
| H2 08:00-04:00 | 0.7362 / 0.7255 / 0.6131 | 0.7291 / 0.7170 / 0.6041 |
| H2 04:00-02:00 | 0.7739 / 0.7461 / 0.4905 | 0.7658 / 0.7403 / 0.4909 |
| H2 02:00-00:00 | 0.7559 / 0.7043 / 0.3148 | 0.7549 / 0.7032 / 0.3220 |
| OT | 0.7567 / 0.7145 / UP | 0.7470 / 0.7157 / UP |

Starter and bench minutes, 2025: starter mean 28.32 (SD 7.92), bench 12.63
(SD 8.51), starters' share of team minutes 0.7016.

---

## 7. Responsiveness: team quintile of the as-of starter-minutes share

Team-games bucketed into quintiles of the pregame as-of share of team minutes
going to the five that side started; season 2025, all 10,640 team-games (this is
a slightly different prior from the round-3 slope check, which bucketed on the
*predicted* starting five over the 1,600-game subset, so the levels are not
directly comparable to section 7.6 there -- the shape is what matters).

| quintile | team-games | prior starter share | close-late cell (final 8:00, \|m\|<=5) | opening ten minutes, \|m\|<=5 | close-late possessions |
|---|---:|---:|---:|---:|---:|
| Q1 | 1,165 | 0.4495 | 0.6739 | 0.7293 | 19,124 |
| Q2 | 1,165 | 0.5941 | 0.7162 | 0.7546 | 19,087 |
| Q3 | 1,164 | 0.6365 | 0.7480 | 0.7714 | 19,512 |
| Q4 | 1,165 | 0.6727 | 0.7758 | 0.7815 | 19,366 |
| Q5 | 1,165 | 0.7284 | 0.8135 | 0.8251 | 18,915 |

Slope of the close-late cell against the prior: **+0.494**, Q5 - Q1 = **+14.0 pp**,
monotone in 4 of 4 steps. Any round-4 arm has to slope with this, per Decision 8
and the standing matchup-specific rule; a flat arm is not matchup-specific
whatever its pooled cell says.

---

## 8. The L25 reachability check -- before any arm is fitted

### 8.1 What is computed

L25 requires the family's reachable range for the target cells, computed from
the fitted objects, before fitting a knob. For a hazard family the analogue of
"the whole knob grid" is the **saturated cell form**: hazards indexed by
(is_starter x time cell x margin band x period-boundary flag), which is the most
any per-player hazard model conditioned on those variables can know. If the
occupancy that form produces **under the five-on-the-floor constraint** misses
the target cells, no member of the family reaches them and the family is wrong
on paper.

The probe (`scripts/diag_rotation_sub_hazard.py --mode reach`) fits that cell
table on **2024** and runs it forward on **400 randomly chosen 2025 games**
(numpy RandomState 2025) over the real possession scripts, with exits drawn as
independent Bernoulli and entrants taken in an Efraimidis-Spirakis exponential
race weighted by `p_in / (1 - p_in)` -- the same selection rule
`rotation_v4.run_sub_hazard` and the engine adapter use.

It is **not a bake-off arm**: it uses the real starting five and the real
participant pool, it has no as-of anything, and it is graded on nothing. It
answers one question, and it answers it before the pre-registration exists.

### 8.2 The answer

| cell | target (actual, same 400 games) | round-4 family, on paper | miss | best arm in rounds 1-3 | that family's reachable max |
|---|---:|---:|---:|---:|---:|
| starter share, H2 tip, \|m\|<=5 | 0.9732 | **0.9758** | +0.3 pp | not measured (R2 0.79 over the 4-min window) | -- |
| starter share, H2 tip, \|m\| 6-15 | 0.9574 | **0.9670** | +1.0 pp | -- | -- |
| starter share, H2 tip, \|m\|>15 | 0.9508 | **0.9525** | +0.2 pp | -- | -- |
| final 8:00, \|m\|<=5 | 0.7388 | **0.7393** | +0.05 pp | 0.7281 (R2) | **0.6994** (R5/R7 keep grid) |
| final 8:00, \|m\| 6-15 | 0.7135 | **0.7031** | -1.0 pp | 0.6943 (R2) | -- |
| final 8:00, \|m\|>15 | 0.5120 | **0.5201** | +0.8 pp | 0.5422 (R6) | -- |
| H1 20:00-10:00, \|m\|<=5 | 0.7722 | **0.7741** | +0.2 pp | 0.6120 (R2) | -- |

**The family reaches every target, all six inside 1.0 pp.** The contrast with
L25's finding is the point: the override family's maximum close-band share over
its entire 24-point knob grid was 0.6994 against a 0.7110 target, and raising the
knob *lowered* it. Here the close-band cell is hit to 0.05 pp with no knob at
all, because a hazard model does not spend a budget -- it reproduces the
conditional rate of the decision itself, and the occupancy is the equilibrium of
those rates.

### 8.3 The log-odds separation L25 asks for, for completeness

In the close-and-late state (H2 08:00-04:00, \|m\|<=5, 2025) the empirical
separation between a starter and a bench player is

* exit: 0.0267 against 0.0690 -- **0.98** in log odds;
* entry: 0.0845 against 0.0259 -- **1.30** in log odds.

R7's single fitted on-floor propensity had to do the whole job with a separation
of about **1.0** and one scale knob, and L25 showed why that fails: the knob
moves both classes together. The hazard family has *two* separations acting in
opposite directions on two different risk sets, and their equilibrium, not their
scale, sets the cell. That is the structural reason the numbers in 8.2 come out
where they do.

### 8.4 What the probe does NOT establish, stated plainly

The probe uses the **real** starting five and the **real** participant pool. A
bake-off arm has neither: its starter set overlaps the real one on 4.58 of 5
(rounds 2 and 3), and round 3's decomposition measured the cost -- re-grading the
actual sequence with the model's as-of starter set puts the close-late benchmark
at **0.7215**, 2.8 pp below the 0.7491 the gate scores against. So the family is
reachable on paper and the arms still have to find 2-3 pp that the as-of starter
set does not have. The gate is scored as pre-registered, against each side's own
starting five, and this paragraph is not a licence to soften it.

A second limitation: the probe's saturated cell table has 108 cells and the
round-4 arms are logistic or LightGBM models with 45 features. The design is
saturated in (time cell x margin band x is_starter) precisely so the arms can
represent what the probe represents, and whether the fit is faithful is checked
on the **training season** in `experiments.md` section 11, not assumed here.

---

## 9. What this audit hands the round-4 pre-registration

1. The second-half reset is a **conditional probability of 0.90** at a stoppage
   type the engine already codes (`prev_end == period_start`), not a level to be
   targeted. Both the hard-reset arm (H1) and the earned-reset arm (H2) are
   therefore testable, and H2 is the honest test of whether a fitted hazard can
   produce a structural fact without being told it.
2. The close-late keep is an **equilibrium of two hazards separated by 1.0-1.3
   in log odds on two different risk sets**, and the saturated form of the
   family lands on it to 0.05 pp.
3. The time profile of both hazards is **non-monotone** and the blowout profile
   is **sign-reversed**, so the design must be saturated in (time cell x margin
   band x is_starter). A linear time term cannot represent it, and this is the
   reason the round-4 design has 45 features rather than the 16-18 rounds 1-3
   used.
4. Timeouts (3-4x every hazard) and prior-season minutes share (about half the
   as-of share's gradient) are measured and **excluded**, for engine
   expressibility and for redundancy respectively.
5. The responsiveness slope to beat is **+0.494 with Q5 - Q1 = +14.0 pp**.
