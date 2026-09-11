# Rotation round 8 -- the exit count as a RATE: evidence

**Date** 2026-09-11. **Lane** rotation round 8.
**Pre-registration** `docs/models/rotation/experiments.md` section 18, commit
`b6a18ec`, written and committed before `rotation_v8.py` existed.
**Results** section 19. **Artifacts**
`data/processed/models/rotation/rotation_F1_round8_results.json`,
`rotation_F1_round8_table.csv`, `exit_audit_round8_2026-09-11.json`,
`round8/rotation_v8_exit_{YYYYMM}.json` + `rotation_v8_manifest.json`.

**Verdict: no arm adopted. The served default is unchanged
(`ENGINE_ROTATION=reference`, R2). Nothing in `engine/` was touched.**

---

## 0. What was tested and what was held fixed

One axis was added to round 7's exit table and nothing else:

```
X1 (round 7)   P(k_out | size, exit_cell)
Y1 (round 8)   P(k_out | size, exit_cell, n_starters_on_floor)   -> shrink to X1 at k = 300
Y2 (round 8)   P(k_out | size, time x margin, foul_class, n_st)  -> shrink to Y1 at k = 300
```

`n_starters_on_floor` is the number of the model's own predicted starting five
among the five on the floor **before** the swap, six levels 0-5 (18.2). Round
3b's base fits, round 4's hazards, round 5's wave tables and rank-within-class
rule, round 6's K1 entry rule, the hard second-half reset and round 7's whole
exit mechanism -- including the order of every uniform drawn -- are reused byte
for byte, so **any difference between a round-8 arm and X1 is a difference in
that one axis alone**. 1,600 games, 3 seeds, S1, the round-7 grader unchanged.

---

## 1. Overall

| | ACTUAL | R2 (served) | K1 (r6) | X1 (r7) | **Y1** | Y2 |
|---|---:|---:|---:|---:|---:|---:|
| per-player minutes MAE | 0 | 9.7939 | **8.8622** | 9.6159 | **9.3851** | 9.3564 |
| floors vs K1 | -- | -70 | -- | -46.5 | **-39.3** | -37.1 |
| floors vs X1 | -- | -13 | +46.5 | -- | **+14.2** | +16.0 |
| state cells passed (veto, +/- 3 pp) | -- | 2/8 | 4/8 | 4/8 | **4/8** | 4/8 |
| round-5 cells passed (veto) | -- | 2/2 | 2/2 | 2/2 | **2/2** | 2/2 |
| G8 cells passed (report) | -- | 5/6 | 5/6 | 3/6 | **5/6** | 5/6 |
| Decision 8 slope ratio (band 0.8-1.2) | 1.00 | n/a | 0.83 P | 0.62 F | **0.60 F** | 0.58 F |
| player quintiles lost to K1 and W4 | -- | 0 | -- | 5/5 | **5/5** | 5/5 |
| K-S D, per-player minutes | -- | 0.0798 | 0.0589 | 0.0441 | **0.0413** | 0.0414 |

The axis is worth **+0.231 minutes of MAE over X1 (14 floors)** and repairs both
G8 cells round 7 broke (pooled minutes SD ratio 0.885 -> 0.914, back inside the
band; top-5 share 0.7232 -> 0.7338 against a real 0.7472). It leaves the arm
**39 floors behind K1**, 4/8 on the veto and outside the Decision 8 band.

**Floor A** (20 seeds x 150 games): Y1 minutes_mae 0.01331, close-band
0.01734, middle band 0.01307, foul trouble 0.02342, opening cell 0.01199. Every
miss below is larger than its own floor by 3-12 SDs.

**Floor B WAS run** (section 6 below), after section 19 of `experiments.md` had
been written and committed saying it would not be; the correction is recorded in
19.16 rather than by editing 19.12, which stands as it was true when written.
**The Decision 10 freeze was NOT run** -- pre-registered as conditional on the
lane's 12:45 ET hard stop (18.10) and it needs an engine adapter this lane did
not write. Condition 6 of the decision rule is therefore unmet for every arm, and
no round-8 arm may be served.

---

## 2. Per state cell (the veto, +/- 3 pp)

| cell | ACTUAL | K1 | X1 | **Y1** | Y1 gap | floors | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| final 8:00, \|m\| <= 5 | 0.7491 | 0.7001 | 0.5394 | **0.5737** | -17.5 pp | 10.1 | FAIL |
| final 8:00, \|m\| 6-15 | 0.7240 | 0.6890 | 0.5384 | **0.5697** | -15.4 pp | 11.8 | FAIL |
| final 8:00, \|m\| > 15 | 0.5223 | 0.5083 | 0.5183 | 0.5255 | +0.3 pp | 0.2 | PASS |
| starters at >= 4 fouls | 0.4613 | 0.4212 | 0.3823 | 0.3925 | -6.9 pp | 2.9 | FAIL |
| H2 tip, \|m\| <= 5 | 0.9678 | 0.9385 | 0.9406 | 0.9407 | -2.7 pp | 2.8 | PASS |
| H2 tip, \|m\| 6-15 | 0.9611 | 0.9410 | 0.9424 | 0.9414 | -2.0 pp | 2.4 | PASS |
| H2 tip, \|m\| > 15 | 0.9563 | 0.9456 | 0.9449 | 0.9453 | -1.1 pp | 0.7 | PASS |
| H1 20:00-10:00, \|m\| <= 5 | 0.7822 | 0.7267 | 0.6991 | 0.7105 | -7.2 pp | 5.9 | FAIL |

Y1 recovers **3.5 pp of X1's 21.0 pp close-band miss and 3.2 pp of 18.6** -- 2.0
and 2.4 floor-A SDs, real movement, and one sixth of the distance to K1 (which is
itself 4.9 pp short). The four cells that pass are the same four every arm since
round 6 passes.

**The time-since-reset gradient, flattened by 17% and not removed:**

| minutes since the last forced reset | K1 gap | X1 gap | **Y1 gap** |
|---|---:|---:|---:|
| 0 (H2 tip, all three bands) | -1.1 to -2.9 pp | -1.1 to -2.7 pp | -1.1 to -2.7 pp |
| 0-10 (H1 20:00-10:00, close) | -5.6 pp | -8.3 pp | **-7.2 pp** |
| 12+ (final 8:00, close) | -4.9 pp | **-21.0 pp** | **-17.5 pp** |

Y1 is still correct at the reset and still drifts monotonically away from it, at
0.83 of X1's rate.

---

## 3. Per `n_starters_on_floor` cell -- the object the round exists to move

`scripts/diag_rotation_exit_v8.py`, 200 games, seed 0, the as-of predicted
starter set on BOTH sides, one function for every row.
`P(a starter is the man who leaves | single swap, starters on the floor)`:

| starters on floor | ACTUAL | n | X1 | n | **Y1** | n | Y2 | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3115 | 244 **UP** | 0.5774 | 594 | **0.4612** | 451 | 0.4018 | 443 |
| 2 | 0.4369 | 982 | 0.5752 | 1,210 | **0.4447** | 1,158 | 0.4294 | 1,148 |
| 3 | 0.5115 | 1,775 | 0.5443 | 1,547 | **0.4844** | 1,734 | 0.4793 | 1,763 |
| 4 | 0.6761 | 1,689 | 0.5892 | 1,463 | **0.6286** | 1,578 | 0.6378 | 1,604 |
| 5 | 1.0000 | 544 | 1.0000 | 592 | 1.0000 | 597 | 1.0000 | 590 |
| **span 1 -> 4** | **+36.5 pp** | | **+1.2 pp** | | **+16.7 pp** | | **+23.6 pp** | |

**UP** = UNDERPOWERED at n < 300, labelled and never read as signal; the powered
version of that cell is 17.17's full-season measurement, 0.3697 on 5,981 leavers
(2025) and 0.3689 on 4,072 (2024).

Two findings, both first-time measurements:

1. **X1's simulated rate is FLAT in the composition** -- 0.577 / 0.575 / 0.544 /
   0.589 across a real 36.5 pp span. Round 7 inferred "a level, not a rate" from
   arithmetic (17.14 item 2); this is the same claim measured on the sim side.
2. **Y1 slopes the right way at every step and carries 46% of the real span.**

**Why only 46%: the shrinkage parent, measured not judged.** Y1's parent is X1's
row -- the level. At one starter on the floor the 2,764 fitted rows split over 18
exit cells leave ~154 per cell, so at `k = 300` the data carry weight
154/(154+300) = 0.34 and the fitted rate is
`0.34 x 0.37 + 0.66 x 0.5576 = 0.49`, which is the 0.5202 in the artifact. At
four starters the 28,011 rows carry weight 0.84 and the fitted 0.6316 sits 2.5 pp
from the real 0.6562.

Fitted `P(k_out = 1 | size 1, n_st)` averaged over the 18 exit cells, window
202411 (six windows agree to 0.1-1.1 pp per level):

| n_st | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| fitted Y1 | 0.5521 | 0.5202 | 0.4939 | 0.5348 | 0.6316 | 0.7553 |
| real (17.17) | -- | 0.3697 | 0.4374 | 0.5209 | 0.6562 | 1.0000 |
| rows, summed over cells | 379 | 2,764 | 12,646 | 28,762 | 28,011 | 9,582 |

**Support, published as 18.3 required:** 63 of Y1's 108 size-1 (exit cell x n_st)
cells carry under 300 rows (61-63 across the six windows); **118 of Y2's 162**
do, so **Y2 is reported UNIDENTIFIED** under the condition 18.3 fixed.

**The simulated floor composition moves with the rate.** Single swaps taken with
1 or 2 starters on the floor: ACTUAL 1,226, X1 1,804, **Y1 1,609**, Y2 1,591 --
the bench-heavy tail X1 walked into is a third smaller and still 31% too big.

**The exit-side starter share (round 7's own diagnostic, same configuration):**

| | leavers overall | size 1 | size 2 | size 3+ | entrants | size-1 joint | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACTUAL | 0.5576 | 0.5867 | 0.5576 | 0.4931 | 0.4972 | 0.6848 / 0.3446 | 34.0 pp |
| K1 | 0.4927 | 0.4564 | 0.5142 | 0.5496 | 0.4334 | 0.6344 / 0.2026 | 43.2 pp |
| X1 | 0.5611 | 0.6013 | 0.5648 | 0.4616 | 0.4882 | 0.6762 / 0.3073 | 36.9 pp |
| **Y1** | 0.5422 | 0.5636 | 0.5368 | 0.5000 | 0.4769 | 0.6891 / 0.2798 | 40.9 pp |
| Y2 | 0.5370 | 0.5576 | 0.5275 | 0.5042 | 0.4728 | 0.6887 / 0.2697 | 41.9 pp |

The marginal X1 hit exactly is now 1.5 pp low, which is the correct consequence
of making the rate conditional: a marginal is reproduced only when the
conditional rate AND the composition distribution are both right, and Y1's floor
still spends too long bench-heavy.

---

## 4. Per team

Starters' share in the final 8:00 at \|m\| <= 5, aggregated per team; a team is
powered when both denominators reach 300 on-floor slots -- **251 powered, 112
UNDERPOWERED and excluded**.

| | sim mean | sim SD | actual mean | actual SD | corr(sim, actual) | mean \|dev\| |
|---|---:|---:|---:|---:|---:|---:|
| K1 (r6) | 0.7004 | 0.0507 | 0.7512 | 0.0832 | **0.394** | 0.0757 |
| X1 (r7) | 0.5412 | 0.0599 | 0.7512 | 0.0832 | 0.209 | 0.2108 |
| **Y1** | 0.5726 | 0.0548 | 0.7512 | 0.0832 | **0.211** | **0.1793** |
| Y2 | 0.5840 | 0.0555 | 0.7512 | 0.0832 | 0.212 | 0.1689 |

The mean absolute deviation falls 15% and **the correlation with the team's own
actual share does not move at all** (0.209 -> 0.211 against K1's 0.394). The
level improves; the matchup responsiveness does not.

---

## 5. Per player quintile

Per-player minutes MAE by quintile of the player's own pregame as-of minutes per
game; 4,782-4,784 player-games per quintile, **none underpowered**; floor 0.013.

| quintile | K1 (r6) | W4 (r6) | X1 (r7) | **Y1** | Y2 | Y1 - K1 | Y1 - W4 | Y1 - X1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 (lowest mpg) | 9.3108 | 9.6681 | 10.6053 | 10.2595 | 10.1673 | **+0.949** | +0.591 | **-0.346** |
| Q2 | 9.7891 | 9.7294 | 10.4615 | 10.2980 | 10.2687 | +0.509 | +0.569 | -0.164 |
| Q3 | 9.2575 | 9.4074 | 9.7867 | 9.6199 | 9.6141 | +0.362 | +0.213 | -0.167 |
| Q4 | 8.6461 | 8.6621 | 9.1081 | 8.9462 | 8.9625 | +0.300 | +0.284 | -0.162 |
| Q5 (highest mpg) | 7.2367 | 7.3307 | 8.0130 | 7.7025 | 7.6742 | +0.466 | +0.372 | **-0.311** |

**Y1 beats X1 in all five quintiles by 12-26 floors and loses all five to both
references by 16-71 floors.** The axis pays everywhere and pays enough nowhere.

**Decision 8, team-game quintiles of the pregame as-of starter-minutes share**
(640 team-games per quintile), close-and-late cell:

| quintile | ACTUAL | K1 | X1 | Y1 | Y2 |
|---|---:|---:|---:|---:|---:|
| Q1 | 0.6770 | 0.6218 | 0.4921 | 0.5261 | 0.5357 |
| Q2 | 0.7226 | 0.6940 | 0.5280 | 0.5607 | 0.5719 |
| Q3 | 0.7471 | 0.7089 | 0.5372 | 0.5754 | 0.5910 |
| Q4 | 0.7763 | 0.7299 | 0.5563 | 0.5923 | 0.6062 |
| Q5 | 0.8219 | 0.7430 | 0.5826 | 0.6128 | 0.6172 |
| slope | +0.692 | +0.576 | +0.426 | **+0.416** | +0.399 |
| ratio to actual | 1.00 | 0.83 P | 0.62 F | **0.60 F** | 0.58 F |

Monotone in 4 of 4 steps for every arm. **Y1 lifts every quintile by 3.0-3.6 pp
and leaves the slope where X1 left it.** A within-game restoring force carries no
between-team information by construction, so it moves every quintile's level
together and cannot move the response to the team's own rotation depth.

---

## 6. Reproduction check and what is and is not established

**X1 reproduction (18.5).** A 1-seed re-run of X1 inside round 8 against round
7's 3-seed column: every state cell moves less than its floor-A SD (largest,
the H2 tip at \|m\| 6-15, -0.65 pp against a 0.88 pp floor); MAE +0.035 on one
seed against a 0.016 floor, the expected 1-vs-3-seed difference. **The reference
columns stand.**

**Floor B (18.9): the round-8 objects are identified.** A different training-game
sample (fit seed 101 against 11) and a different sim seed (23 against 7), graded
on the same 150-game universe, run on Y1 in its own process after the bake-off
JSON was on disk.

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

The fitted table moves **0.15-0.17 pp per composition level** under the refit
(window 202411, `P(k_out = 1 | size 1, n_st)`: 0.5202 -> 0.5217 at one starter,
0.6316 -> 0.6299 at four) and the under-300 cell count moves 63 -> 65. The cell
spread (0.43-4.77 pp) is round 7's (0.29-6.31 pp) on the same thin universe.
**The decision does not turn on a refit artefact in either direction**: the
close-band miss Y1 fails on is -17.5 pp, eight times the largest refit-to-refit
move on that cell and 10 floor-A SDs. The `minutes_mae` LEVEL (23.1) is the
150-game-universe artifact rounds 5-7 carry for the same reason (11.6) and only
the differences are readable.

**Established.** The composition axis is real, is identified under a refit and
across six independent training windows, moves the sim's own exit rate in the
right direction at every step, repairs both broken G8 cells, buys 14 floors of
MAE and flattens 17% of the drift gradient.

**Not established.** No closed-loop freeze was run, so **no round-8 arm may be
served**, and none is proposed for serving.

---

## 7. What this hands the next round

1. **The binding constraint is the shrinkage parent.** Y1 falls back on the LEVEL
   in the 63 thin cells that matter most. The composition MARGINAL
   `P(k_out | size, n_st)` is powered at 2,764-28,762 rows per level with a 29 pp
   span and no state split. **Invert the hierarchy: fit `P(k_out | size, n_st)`
   first and shrink the (size, state, n_st) cell to THAT.** A thin cell then falls
   back on the strong term instead of on the term with no fixed point. The counts
   are already in round 8's artifacts; it is one line of `fit_exit8`.
2. **Decision 8 needs a different object.** Eight rounds of exit and entry rules
   have failed it; no rotation table yet carries anything the TEAM brings, such as
   its own as-of starter-minutes share.
3. **Foul trouble stays queued behind the drift** (round 7's ordering, unchanged):
   the graded cell improved 0.3823 -> 0.3925, is still -6.9 pp, and Y2's foul
   class added 0.05 pp.
4. **The Decision 10 freeze is now two rounds overdue** -- round 7 could not fit
   it and neither could round 8, because it needs a vectorised
   `next_lineup_round8` in `engine/rotation_adapter.py` that no lane has written.
   It must be inside the next round's budget from the start, adapter included, or
   this family can never be served however well it grades offline.
