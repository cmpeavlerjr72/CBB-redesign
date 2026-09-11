# Rotation round 9: inverting the exit hierarchy onto the powered composition marginal

2026-09-11. Evidence doc for `docs/models/rotation/experiments.md` sections 20
(pre-registration, commit 3525996) and 21 (results). Code 354e4d8. Every number
here is produced by the round-8 grading path unchanged, on the same 1,600-game
2025 subset rounds 2-8 used, 3 seeds per candidate arm under S1.

**Result in one line: the fitted object is repaired almost completely -- the
simulated exit rate by composition goes from 46% of the real span (round 8) to
94% -- the close-and-late state cell moves 2.8 pp of a 17.5 pp miss, per-player
minutes MAE gains 5.4 floors, and NO ARM IS ADOPTED, because the arm is still
33.5 floors behind K1, still 4/8 on the veto, and its Decision 8 slope ratio
falls from 0.60 to 0.51.**

---

## 1. What was changed, and only that

Round 8 fitted `P(k_out | size, exit_cell, n_st)` and shrank it to X1's row --
`P(k_out | size, exit_cell)`, the LEVEL. Round 9 keeps the counts, the sampler,
the mechanism, the uniform order, the wave tables, the hazards, the entry rule
and the hard second-half reset byte for byte (`rotation_v9.run_wave9` delegates
to `rotation_v8.run_wave8` through a table shim) and changes **only the
shrinkage parent**:

| level | round 8 (Y1) | round 9 (Z1) |
|---|---|---|
| 0 | `P(k_out \| size)` | `P(k_out \| size)` |
| 1 | `P(k_out \| size, exit_cell)` -- the LEVEL | **`P(k_out \| size, n_st)` -- the MARGINAL** |
| 2 | `P(k_out \| size, exit_cell, n_st)` | `P(k_out \| size, exit_cell, n_st)` |

Z2 is Z1 with the (state x composition) interaction used only where the cell
carries >= 300 rows and the marginal used exactly otherwise.

The shrinkage constant was **fitted**, not chosen: leave-one-fold-out
multinomial log-likelihood over 5 folds of the 2024 training season
(24,218-24,597 held-out rows each), grid declared in 20.4 before any fit.

| `k` | 30 | 100 | 300 | 1000 | 3000 |
|---|---:|---:|---:|---:|---:|
| held-out nats/row | **-0.64950** | -0.65448 | -0.66471 | -0.68608 | -0.71814 |

`k = 30` wins, and the score is monotone across the whole grid. The left
boundary winning is a caveat, carried in 21.13 item 5, not buried: with the
marginal as parent the data want ten times less shrinkage than rounds 7-8's
declared 300.

---

## 2. Level 1: the fitted table (offline, no simulation)

`P(k_out = 1 | size 1, starters on the floor)`, window 202411; the six windows
agree to 0.1-1.1 pp on every level.

| starters on floor | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| **marginal `M1`** | 0.0432 | **0.3227** | 0.4010 | 0.5001 | 0.6574 | 0.9987 |
| **Z1** | 0.0327 | **0.3076** | 0.3881 | 0.5032 | 0.6715 | 0.9996 |
| Z2 | 0.0432 | 0.3282 | 0.3934 | 0.4939 | 0.6553 | 0.9993 |
| *Y1 (round 8)* | *0.5521* | *0.5202* | *0.4939* | *0.5348* | *0.6316* | *0.7553* |
| ACTUAL (17.17) | -- | **0.3697** | 0.4374 | 0.5209 | 0.6562 | 1.0000 |
| fitted rows | 379 | 2,764 | 12,646 | 28,762 | 28,011 | 9,582 |

Y1's thin one-starter cell landed at 0.52 against a real 0.37 because 154 rows
per cell at `k = 300` gave the data weight 0.34 against a parent of 0.5576.
Z1's lands at 0.31: the same 154 rows, now weighted 0.84 at `k = 30`, against a
parent of 0.3227 that is itself fitted on 2,764 rows. **61-63 of the 108 size-1
cells are still under 300 rows** -- the round changed what they fall back on, not
how many there are.

Y1's five-starter cell read 0.7553, which is structurally impossible (a single
swap from an all-starter floor must remove a starter). Z1 reads 0.9996.

---

## 3. Level 2: the simulated exit rate by composition (the object the round exists to move)

`scripts/diag_rotation_exit_v9.py`, 200 games, seed 0, the as-of predicted
starter set on BOTH sides, one function for every row.
`P(a starter is the man who leaves | single swap, starters on the floor)`:

| starters on floor | ACTUAL | n | X1 (r7) | Y1 (r8) | **Z1** | n | Z2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3115 | 244 UP | 0.5774 | 0.4612 | **0.3049** | 387 | 0.3333 |
| 2 | 0.4369 | 982 | 0.5752 | 0.4447 | **0.3943** | 1,149 | 0.4009 |
| 3 | 0.5115 | 1,775 | 0.5443 | 0.4844 | **0.4818** | 1,818 | 0.4741 |
| 4 | 0.6761 | 1,689 | 0.5892 | 0.6286 | **0.6468** | 1,608 | 0.6384 |
| 5 | 1.0000 | 544 | 1.0000 | 1.0000 | 1.0000 | 575 | 1.0000 |
| **span 1 -> 4** | **+36.5 pp** | | +1.2 pp | +16.7 pp | **+34.2 pp** | | +30.5 pp |
| **% of real span** | 100% | | 3% | 46% | **94%** | | 84% |

UP = UNDERPOWERED (n < 300), labelled and never read as signal; the powered
full-season version of that row is 17.17's 0.369/0.370 on 4,072-5,981 leavers.

Every Z1 level is within 0.7-4.3 pp of actual against Y1's 2.8-15.0 pp. **The
offline table of section 2 and the simulated rate here agree**, which is the
paired offline/sim check this family has carried since round 7.

The composition DISTRIBUTION has not followed. Single swaps taken with 1 or 2
starters on the floor: **ACTUAL 1,226, X1 1,804, Y1 1,609, Z1 1,536, Z2 1,519**
-- still **25% too bench-heavy**. A correct conditional rate over a wrong state
distribution is the signature of the WAVE side, which carries no composition axis
at all; that is the next object (section 8).

---

## 4. Per state cell (the veto), 1,600 games, 3 seeds

| cell | ACTUAL | K1 | X1 | Y1 | **Z1** | Z1 miss | floor A | floors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7491 | 0.7001 | 0.5394 | 0.5737 | **0.6019** | -14.7 pp | 0.0153 | 9.6 F |
| final 8:00, \|m\| 6-15 | 0.7240 | 0.6890 | 0.5384 | 0.5697 | **0.5939** | -13.0 pp | 0.0118 | 11.1 F |
| final 8:00, \|m\| > 15 | 0.5223 | 0.5083 | 0.5183 | 0.5255 | 0.5371 | +1.5 pp | 0.0192 | 0.8 P |
| starters at >= 4 fouls | 0.4613 | 0.4212 | 0.3823 | 0.3925 | **0.4018** | -5.9 pp | 0.0236 | 2.5 F |
| H2 tip, \|m\| <= 5 | 0.9678 | 0.9385 | 0.9406 | 0.9407 | 0.9392 | -2.9 pp | 0.0111 | 2.6 P |
| H2 tip, \|m\| 6-15 | 0.9611 | 0.9410 | 0.9424 | 0.9414 | 0.9409 | -2.0 pp | 0.0092 | 2.2 P |
| H2 tip, \|m\| > 15 | 0.9563 | 0.9456 | 0.9449 | 0.9453 | 0.9434 | -1.3 pp | 0.0161 | 0.8 P |
| H1 20:00-10:00, \|m\| <= 5 | 0.7822 | 0.7267 | 0.6991 | 0.7105 | 0.7122 | -7.0 pp | 0.0110 | 6.4 F |
| **passed at +/- 3 pp** | | 4/8 | 4/8 | 4/8 | **4/8** | | | |

**The two close-and-late bands move 2.8 and 2.4 pp the right way (1.8 and 2.0
floors) and remain 9.6 and 11.1 floors short.** The count of passed cells does
not change, because what fails, fails by 5.9-14.7 pp.

The time-since-reset gradient, the round's headline drift number:

| minutes since the last forced reset | cell | K1 | X1 | Y1 | **Z1** |
|---:|---|---:|---:|---:|---:|
| 0 | H2 tip, three bands | -1.1 to -2.9 | -1.1 to -2.7 | -1.1 to -2.7 | -1.3 to -2.9 |
| 0-10 | H1 20:00-10:00, close | -5.6 | -8.3 | -7.2 | **-7.0** |
| 12+ | final 8:00, close | -4.9 | **-21.0** | -17.5 | **-14.7** |

Flattened **30% from X1 and 16% from Y1**, and not removed. An exit rate that is
94% right leaves two thirds of the drift standing.

---

## 5. Overall: the primary metric and the gates

| metric | ACTUAL | R2 | K1 | X1 | Y1 | **Z1** | Z2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| per-player minutes MAE | 0.0 | 9.7939 | **8.8622** | 9.6159 | 9.3851 | **9.3132** | 9.3191 |
| floors vs R2 / K1 / Y1 | -- | -- | -- | -- | -- | **+32.6 / -33.5 / +5.4** | +32.2 / -33.9 / +5.0 |
| substitutions per boundary (veto, +/- 0.015) | 0.1509 | 0.1437 P | 0.1554 P | 0.1557 P | 0.1559 P | **0.1560 P** | 0.1561 P |
| distinct lineups per team-game (veto, +/- 1.5) | 14.836 | 15.514 P | 14.571 P | 15.939 P | 15.899 P | **15.854 P** | 15.865 P |
| minutes SD ratio, pooled (G8) | 1.000 | 1.055 P | 0.996 P | 0.885 F | 0.914 P | **0.930 P** | 0.928 P |
| minutes SD ratio, within-player (G8) | 1.000 | 1.342 F | 1.124 F | 1.145 F | 1.138 F | 1.139 F | 1.137 F |
| top-5 share of team minutes (G8) | 0.7472 | 0.7630 P | 0.7597 P | 0.7232 F | 0.7338 P | **0.7385 P** | 0.7380 P |
| **G8 passed** | | 5/6 | 5/6 | 3/6 | 5/6 | **5/6** | 5/6 |
| K-S D, per-player minutes | -- | 0.0798 | 0.0589 | 0.0441 | **0.0413** | 0.0414 | 0.0414 |
| "at exactly 4 fouls" (report) | 0.5166 | 0.5722 | 0.5931 | **0.5316** | 0.5473 | 0.5666 | 0.5714 |

Z1 and Z2 differ by 0.006 minutes of MAE -- **0.46 of a floor**. The hard
300-row gate and continuous shrinkage at `k = 30` are the same model inside the
noise, so the tie-break takes the simpler arm.

---

## 6. Per team, 251 powered teams (112 UNDERPOWERED and excluded)

Starters' share in the final 8:00 at \|m\| <= 5, aggregated per team:

| | sim mean | sim SD | actual mean | actual SD | corr | mean \|dev\| |
|---|---:|---:|---:|---:|---:|---:|
| K1 (round 6) | 0.7004 | 0.0507 | 0.7512 | 0.0832 | **0.394** | 0.0757 |
| X1 (round 7) | 0.5412 | 0.0599 | 0.7512 | 0.0832 | 0.209 | 0.2108 |
| Y1 (round 8) | 0.5726 | 0.0548 | 0.7512 | 0.0832 | 0.211 | 0.1793 |
| **Z1** | 0.6018 | 0.0495 | 0.7512 | 0.0832 | **0.190** | **0.1518** |
| Z2 | 0.5927 | 0.0529 | 0.7512 | 0.0832 | 0.190 | 0.1605 |

**The level improves 15% and the correlation with the team's own actual share
goes DOWN, 0.211 -> 0.190.** Cross-team SD falls 0.055 -> 0.050 against a real
0.083. Fixing the within-game restoring force removes dispersion that was drift
noise and adds no team information.

---

## 7. Per player quintile, 4,782-4,784 player-games each (none underpowered)

Per-player minutes MAE by quintile of the player's own pregame as-of minutes per
game:

| quintile | K1 (r6) | W4 (r6) | X1 | Y1 | **Z1** | Z1 - K1 | Z1 - W4 | Z1 - Y1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 (lowest mpg) | 9.3108 | 9.6681 | 10.6053 | 10.2595 | 10.0878 | **+0.777** | +0.420 | **-0.172** |
| Q2 | 9.7891 | 9.7294 | 10.4615 | 10.2980 | 10.2116 | **+0.423** | +0.482 | **-0.086** |
| Q3 | 9.2575 | 9.4074 | 9.7867 | 9.6199 | 9.6052 | +0.348 | +0.198 | -0.015 |
| Q4 | 8.6461 | 8.6621 | 9.1081 | 8.9462 | 8.9734 | +0.327 | +0.311 | **+0.027** |
| Q5 (highest mpg) | 7.2367 | 7.3307 | 8.0130 | 7.7025 | 7.5951 | +0.358 | +0.264 | **-0.107** |

Floor 0.013. **All five quintiles lose to both K1 and W4 (24-58 floors)** -- the
pre-registered veto of 20.9 check 2. Against Y1, Z1 wins Q1 by 12.9 floors, Q2 by
6.5 and Q5 by 8.1, ties Q3 and **loses Q4 by 2.0 floors**. The gain is
concentrated in the deepest-bench quintile, which is where the mechanism predicts
it.

**Decision 8 (responsiveness by TEAM quintile), 640 team-games per quintile:**

| quintile | ACTUAL | K1 | X1 | Y1 | **Z1** |
|---|---:|---:|---:|---:|---:|
| Q1 (shallowest rotation) | 0.6770 | 0.6218 | 0.4921 | 0.5261 | 0.5573 |
| Q2 | 0.7226 | 0.6940 | 0.5280 | 0.5607 | 0.5913 |
| Q3 | 0.7471 | 0.7089 | 0.5372 | 0.5754 | 0.6070 |
| Q4 | 0.7763 | 0.7299 | 0.5563 | 0.5923 | 0.6236 |
| Q5 (deepest) | 0.8219 | 0.7430 | 0.5826 | 0.6128 | 0.6289 |
| **slope** | +0.692 | +0.576 | +0.426 | +0.416 | **+0.355** |
| slope ratio (band 0.8-1.2) | 1.00 | **0.83 P** | 0.62 F | 0.60 F | **0.51 F** |

20.13 item 6 recorded in advance 19.13 item 4's prediction that this check would
not move. **The round falsifies it in the worse direction.** Z1 lifts Q1 by
3.1 pp and Q5 by 1.6 pp, so the slope falls. A within-game restoring force does
not merely carry no between-team information -- **repairing it COMPRESSES the
between-team response.**

---

## 8. Noise floors, and what the decision does not turn on

**Floor A** (20 seeds x 150 games): Z1 `minutes_mae` 0.0128, close band 0.0153,
middle band 0.0118, foul trouble 0.0236, opening cell 0.0110. Every miss in
section 4 is 2.5-11.1 floors.

**Floor B** (fit seed 101 against 11, sim seed 23 against 7, same 150-game
universe, 2.3 min):

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

**The round-9 objects are identified.** The fitted marginal moves 0.9 pp at one
starter on the floor and 0.4 pp at four under the refit (202411: `M1` 0.3227 ->
0.3314, 0.6574 -> 0.6539; Z1 0.3076 -> 0.3201, 0.6715 -> 0.6702), and the
under-300 cell count 63 -> 65 of 108. **The decision does not turn on a refit
artefact**: the close-band miss Z1 fails on is -14.7 pp, five times the largest
refit-to-refit move on that cell (2.87 pp) and 9.6 floor-A SDs, and the axis
repair the round claims is 15 pp at one starter against a 1.25 pp refit move.
The `minutes_mae` LEVEL of 23.1 is the 150-game-universe artifact rounds 5-8
carry (11.6): only DIFFERENCES are readable there.

**Y1 reproduction check**: a 1-seed re-run of Y1 inside this round moves every
state cell less than its floor-A SD (largest, foul trouble, -0.71 pp against a
2.34 pp floor), so rounds 6-8's reference columns stand.

---

## 9. Verdict and the next object

**NO ARM ADOPTED.** Z1 passes conditions 2 and 3 of 20.10 and fails 1 (4/8 state
cells), 4 (33.5 floors behind K1, all five player quintiles lost to K1 and W4)
and 5 (Decision 8 ratio 0.51). Condition 6, the Decision 10 freeze, **was not
run for the third round in a row** because no lane has written the engine
adapter; its precise scope is written out in 21.15 so the next session can
implement it without re-scoping. `ENGINE_ROTATION=reference` stays the served
default and this lane changed no default.

Three things the round settles:

1. **The shrinkage parent was the binding constraint on the fitted object, and
   it is gone.** 46% -> 94% of the real composition span, every level within
   0.7-4.3 pp.
2. **The exit rate is no longer the dominant term in the drift.** Two rounds took
   this object from flat to essentially correct and recovered 6.3 of X1's 21.0 pp
   on the close band, against K1's own 4.9 pp residual. The remaining 9.8 pp is
   elsewhere.
3. **Where it is, is measurable: the WAVE side.** The conditional rate is right
   and the composition distribution is still 25% too bench-heavy, which is the
   signature of a wrong state distribution over a right conditional. Round 5's
   `P(wave | cell)` and `P(size | cell)` carry no composition axis at all. **The
   next object is `P(wave | cell, n_st)` and `P(size | cell, n_st)` -- the same
   axis, one stage upstream** -- with its support measured on the actual
   sequences first, as 17.17 was measured before round 8.

And one warning the round earns: **every future repair to the drift will push
Decision 8 further out of band** unless the same round adds an object indexed by
what the TEAM brings. Decision 8 needs its own pre-registered round; it is not a
gate a drift fix will eventually clear.
