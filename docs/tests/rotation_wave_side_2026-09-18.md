# Rotation round 10 -- the WAVE side, conditioned on starters on the floor

**Run 2026-09-18T18:23-18:44Z (bake-off, 20.4 min) and 18:53-19:02Z (floor B).**
Pre-registration `docs/models/rotation/experiments.md` section 23, committed
**83aa854** before `rotation_v10.py` existed; code **9834594**; results section 24.

**Verdict: NO ARM ADOPTED. No default changed. `ENGINE_ROTATION=reference` (R2)
remains served and no round-10 flag exists in the engine.**

The round moved the two objects it was built to move, by 94-97% and 68-79% of
their real spans, closed 64-82% of the floor-composition gap, and beat round 9's
Z1 by 12.5-14.3 noise floors of per-player minutes MAE -- and it still fails
three of the six pre-registered conditions, by the same margins round 9 failed
them and for the same reason. It also confirms, on a third object, the advance
prediction of 21.13 item 4.

---

## 1. What was measured first, and why round 10 exists

21.13 item 3 named the wave side from a count. 23.1 measured it. At a possession
boundary the number of the model's own predicted starters on the floor moves by
`n_st -> n_st - k_out + k_in` and four objects decide it -- arrival `A(n_st)`,
size `S(size | n_st)`, exit `O(k_out | size, n_st)` and entry
`I(k_in | size, k_out, n_st)` -- plus the hard second-half reset. Round 9
repaired `O` alone.

Measured on **105,410 actual boundaries** of 400 games (1,655-32,129 per `n_st`
level, none underpowered), the as-of predicted starter set on both sides, ONE
function for every row:

| | share of the mean-`n_st` gap (in / out) | share of the `P(n_st <= 2)` gap |
|---|---:|---:|
| **A arrival** | **+0.701 / +0.590** | +0.485 / +0.355 |
| **S size** | +0.415 / +0.427 | +0.453 / +0.406 |
| I entry | +0.322 / +0.188 | +0.486 / +0.331 |
| O exit | **-0.285 / -0.365** | -0.257 / -0.260 |
| reset | 0.000 | 0.000 |

The exit side now works **against** the gap: round 9 slightly over-corrected it.
The full decomposition, its Markov-chain validation (it reproduces 92% of the
observed gap in DIFFERENCES and biases every LEVEL by -0.12 to -0.17, so only
differences may be read from it), the per-cell and per-team breakdowns and the
191 underpowered teams that were excluded are published in 23.1 and in
`data/processed/models/rotation/wave_support_round10_2026-09-18.json`.

## 2. The sampler is the same sampler

23.9's bit-exact identity test, floor 0. `run_wave10` fed round 5's `p_wave` and
`p_size` and round 6's `kin` broadcast over the `n_st` axis, against
`run_wave9`'s Z1, on 30 games x 3 seeds:

**180 team-game sims, 25,224 possessions, 0 lineup mismatches, 0 foul
mismatches.** No uniform was added, removed or reordered, so every round-10 arm
is byte-aligned with rounds 8's and 9's arms and with each other. Artifact:
`round10_sampler_parity_2026-09-18.json`.

## 3. The control did its job

V0 refits round 5's axis-free objects on round 10's own rows. It is the only
thing standing between "the axis works" and "the counts pass changed something":

| | MAE | close band | `P(n_st <= 2)` | arrival span |
|---|---:|---:|---:|---:|
| Z1 (round 9) | 9.3132 | -14.7 pp | 0.2790 | 33% |
| **V0 control** | **9.3198** | **-14.7 pp** | 0.2730 | 38% |
| floor A (V0) | 0.0135 | 1.67 pp | -- | -- |

**V0 is Z1 to within half a noise floor on the MAE and to 0.0 pp on the close
band.** Every V1-V5 movement below is the AXIS, not the rows.

## 4. The two objects the round exists to move

`P(wave | n_st)`, 400 games, seed 0, the same function on the actual and the
simulated sequences:

| `n_st` | 0 | 1 | 2 | 3 | 4 | 5 | span 2->5 | % of real |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ACTUAL** | 0.0979 | 0.1746 | 0.1822 | 0.1674 | 0.1390 | **0.0784** | **-10.4 pp** | 100% |
| Z1 (r9) | 0.1749 | 0.1615 | 0.1580 | 0.1577 | 0.1459 | 0.1235 | -3.5 pp | 33% |
| V0 control | 0.1892 | 0.1547 | 0.1588 | 0.1540 | 0.1478 | 0.1193 | -4.0 pp | 38% |
| **V1** | 0.1309 | 0.1889 | 0.1876 | 0.1751 | 0.1437 | **0.0871** | -10.1 pp | **97%** |
| V2 | 0.1669 | 0.1612 | 0.1567 | 0.1583 | 0.1475 | 0.1231 | -3.4 pp | 32% |
| **V3** | 0.1236 | 0.1857 | 0.1845 | 0.1762 | 0.1430 | **0.0873** | -9.7 pp | **94%** |
| V4 (level parent) | 0.1377 | 0.1638 | 0.1677 | 0.1644 | 0.1466 | 0.1056 | -6.2 pp | 60% |
| **V5** | 0.1285 | 0.1872 | 0.1855 | 0.1770 | 0.1448 | **0.0853** | -10.0 pp | **96%** |

Mean wave size by `n_st`, span from `n_st` 1 to 4:

| | ACTUAL | Z1 | V0 | V1 | **V2** | **V3** | V4 | **V5** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| span | **-0.368** | +0.004 | -0.008 | -0.025 | **-0.267** | **-0.290** | -0.131 | -0.249 |
| % of real | 100% | -1% | 2% | 7% | **73%** | **79%** | 36% | 68% |

Each arm moves exactly the object it carries and leaves the other where it was,
which is the paired check rounds 7-9 established for this family.

**The exit rate does NOT regress** (23.9 required this):
`P(starter leaves | single swap, n_st)` at `n_st` 1/2/3/4 is ACTUAL
0.325/0.414/0.509/0.718, Z1 0.311/0.389/0.500/0.683, V3 0.315/0.391/0.525/0.671,
V5 0.324/0.394/0.521/0.679.

## 5. The floor composition

Observed occupancy of `n_st` over on-floor possessions, and the 21.8 count of
single swaps taken with 1-2 starters on the floor:

| | mean `n_st` | gap closed | `P(n_st <= 2)` | gap closed | single swaps at 1-2 st |
|---|---:|---:|---:|---:|---:|
| **ACTUAL** | **3.3888** | -- | **0.2132** | -- | **2,268** |
| Z1 (r9) | 3.1793 | -- | 0.2790 (+31%) | -- | 3,081 (+36%) |
| V0 control | 3.1949 | 7% | 0.2730 | 9% | 3,012 |
| V1 | 3.3026 | 59% | 0.2546 | 37% | 3,263 |
| V2 | 3.2395 | 29% | 0.2540 | 38% | 2,477 |
| **V3** | **3.3510** | **82%** | **0.2368 (+11%)** | **64%** | **2,694 (+19%)** |
| V4 | 3.2664 | 42% | 0.2550 | 37% | 2,785 |
| **V5** | **3.4003** | **106%** | **0.2062 (-3%)** | **111%** | **2,392 (+5%)** |

**Round 9's 25%-too-bench-heavy floor is 11% too bench-heavy under V3 and 3% too
STARTER-heavy under V5.** This is the round's positive result and it is the only
one.

## 6. The gates -- and the three that fail

| arm | simp | state | r5 cells | G8 | MAE | vs R2 | vs K1 | vs Z1 | floor A | quintiles | D8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| R2 (served) | 1 | 2/8 | 2/2 | 5/6 | 9.7939 | -- | -0.932 | -0.481 | 0.0147 | ok | n/a |
| K1 (r6) | 9 | 4/8 | 2/2 | 5/6 | **8.8622** | +0.932 | -- | +0.451 | 0.0135 | see 15.8 | **0.83 PASS** |
| Z1 (r9) | 16 | 4/8 | 2/2 | 5/6 | 9.3132 | +0.481 | -0.451 | -- | 0.0128 | 5/5 fail | 0.51 F |
| V0 control | 17 | 4/8 | 2/2 | 5/6 | 9.3198 | +0.474 | -0.458 | -0.007 | 0.0135 | 5/5 fail | 0.48 F |
| V1 | 18 | 4/8 | 2/2 | 5/6 | 9.1486 | +0.645 | -0.286 | **+0.165** | 0.0128 | 5/5 fail | 0.44 F |
| V2 | 19 | 4/8 | 2/2 | 5/6 | 9.2361 | +0.558 | -0.374 | **+0.077** | 0.0121 | 5/5 fail | 0.54 F |
| **V3** | 20 | 4/8 | 2/2 | 5/6 | **9.1000** | +0.694 | -0.238 | **+0.213** | 0.0170 | 5/5 fail | 0.44 F |
| V4 | 21 | 3/8 | 2/2 | 5/6 | 9.2012 | +0.593 | -0.339 | +0.112 | 0.0197 | 5/5 fail | 0.41 F |
| **V5** | 22 | 3/8 | 2/2 | 5/6 | **9.0629** | +0.731 | -0.201 | **+0.250** | 0.0176 | 5/5 fail | 0.42 F |

Every candidate beats Z1 on MAE beyond its own floor (V1 12.9 floors, V2 6.4,
V3 12.5, V4 5.7, V5 14.3) and **every candidate is still 11.4-30.9 floors behind
K1**, loses all five player quintiles to both K1 and W4, and misses the Decision
8 band. Three conditions of six fail for every arm.

State cells against actual, in pp:

| cell | act | K1 | Z1 | V1 | V2 | **V3** | V4 | **V5** | floor A (V3) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final 8:00 \|m\| <= 5 | 0.7491 | -4.9 | -14.7 | -12.9 | -13.5 | **-11.8** | -14.1 | **-10.5** | 2.04 |
| final 8:00 \|m\| 6-15 | 0.7240 | -3.5 | -13.0 | -11.7 | -11.7 | **-10.4** | -12.6 | -9.1 | 1.31 |
| final 8:00 \|m\| > 15 | 0.5223 | -1.4 | +1.5 | +2.5 | +2.9 | **+4.0 F** | +1.7 | **+6.1 F** | -- |
| starters at >= 4 fouls | 0.4613 | -4.0 | -5.9 | -5.1 | -5.4 | -4.4 | -5.2 | -4.2 | 2.33 |
| H2 tip, three bands | -- | -1.1/-2.0/-2.9 | -1.3/-2.0/-2.9 | | | -1.3/-2.2/-3.0 | | -1.4/-2.0/-3.1 | 1.2-1.9 |
| **H1 20:00-10:00 \|m\| <= 5** | 0.7822 | -5.6 | **-7.0** | **-2.5 P** | -6.6 | **-2.0 P** | -3.5 | **-1.5 P** | 1.06 |

The wave fix **passes a cell round 9 failed** -- the opening ten minutes of each
half, where the drift away from a reset lives -- and **loses the blowout band**,
where V3 and V5 now leave too MANY starters on the floor. No arm exceeds 4 of 8.

**The time-since-reset gradient (17.10), the headline drift number:**

| minutes since the last forced reset | cell | Z1 | **V3** | **V5** |
|---:|---|---:|---:|---:|
| 0 | H2 tip, all three bands | -1.3 to -2.9 pp | -1.3 to -3.0 pp | -1.4 to -3.1 pp |
| 0-10 | H1 20:00-10:00, \|m\| <= 5 | -7.0 pp | **-2.0 pp** | **-1.5 pp** |
| 12+ | final 8:00, \|m\| <= 5 | -14.7 pp | **-11.8 pp** | **-10.5 pp** |

**The drift over the first ten minutes after a reset is 71-79% gone. The
late-game residual falls by only 20-29%.** The gradient is no longer a gradient:
it is a step that happens somewhere between ten minutes and the final eight, and
that is a different defect from the one rounds 7-10 have been repairing.

## 7. Per player, per team, per game

Per-player minutes MAE by quintile of the player's own pregame as-of minutes per
game (4,782-4,784 player-games per quintile, none underpowered):

| quintile | K1 (r6) | W4 (r6) | Z1 (r9) | **V3** | **V5** | V3 - K1 | V3 - Z1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 (lowest mpg) | 9.3108 | 9.6681 | 10.0878 | 9.7122 | 9.7400 | **+0.401** | **-0.376** |
| Q2 | 9.7891 | 9.7294 | 10.2116 | 10.0072 | 9.9516 | **+0.218** | -0.204 |
| Q3 | 9.2575 | 9.4074 | 9.6052 | 9.4188 | 9.4091 | +0.161 | -0.186 |
| Q4 | 8.6461 | 8.6621 | 8.9734 | 8.8313 | 8.7633 | +0.185 | -0.142 |
| Q5 (highest mpg) | 7.2367 | 7.3307 | 7.5951 | 7.4499 | 7.3670 | +0.213 | -0.145 |

Every arm beats Z1 in every quintile and loses every quintile to both K1 and W4.
Per game: V3 9.1141 +/- 2.1812 (p90 11.6165), V5 9.0728 +/- 2.1702 (11.4858),
Z1 9.3328 (11.7221), K1 **8.8683** (11.2881).

Per team, starters' share in the final 8:00 at \|m\| <= 5 (**251 teams powered at
300 on-floor slots on both sides, 112 UNDERPOWERED and excluded**):

| | sim mean | sim SD | corr(sim, actual) | mean \|dev\| |
|---|---:|---:|---:|---:|
| actual | 0.7512 | 0.0832 | -- | -- |
| K1 (r6) | 0.7004 | 0.0507 | **0.394** | 0.0757 |
| Z1 (r9) | 0.6018 | 0.0495 | 0.190 | 0.1518 |
| V3 | 0.6316 | 0.0510 | **0.168** | 0.1271 |
| V5 | 0.6435 | 0.0484 | 0.227 | **0.1172** |

And on the round's own object, per team (139 powered at 300 boundaries, **191
UNDERPOWERED and excluded**): the correlation between a team's simulated and real
mean `n_st` falls from K1's 0.522 and Z1's 0.258 to **V3's 0.125 and V5's 0.089**,
while the mean absolute deviation falls 0.374 -> 0.358 -> 0.348. **The level
improves and the matchup responsiveness gets worse, for the third round running.**

## 8. Decision 8 -- the advance prediction, confirmed

23.10 recorded, before the round ran, that every round-10 arm would move the
Decision 8 slope ratio DOWN from Z1's 0.51 and the per-team correlation down from
0.190. Measured:

| | ACTUAL | K1 | Z1 | V0 | V1 | V2 | V3 | V4 | V5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.6770 | 0.6218 | 0.5573 | 0.5568 | 0.5857 | 0.5692 | 0.5955 | 0.5720 | 0.6093 |
| Q5 | 0.8219 | 0.7430 | 0.6289 | 0.6243 | 0.6485 | 0.6442 | 0.6589 | 0.6291 | 0.6691 |
| **slope** | +0.6919 | +0.5756 | +0.3551 | +0.3317 | +0.3072 | +0.3750 | +0.3017 | +0.2837 | +0.2926 |
| **ratio** | 1.00 | **0.83 P** | 0.51 F | **0.48 F** | **0.44 F** | 0.54 F | **0.44 F** | **0.41 F** | **0.42 F** |

**Five of six arms move it down, exactly as predicted; V2 (size alone) moves it up
by 0.029 and is the only partial falsification.** Every arm lifts Q1 more than Q5
(V3 +3.8 pp against +3.0 pp) because a restoring force helps most where there is
most to restore. Decision 8 has now been pushed the wrong way by three
consecutive families -- the exit rate, the exit hierarchy and the wave side -- and
it will be pushed the wrong way by the fourth unless that round adds an object
indexed by something the TEAM brings.

## 9. Floors

**Floor A**, 20 seeds x 150 games per candidate: `minutes_mae` V1 0.0128,
V2 0.0121, V3 0.0170, V4 0.0197, V5 0.0176, V0 0.0135 (Z1's is 0.0128, K1's
0.0135, R2's 0.0147). Close band 1.35-2.04 pp; H1 20:00-10:00 1.03-1.19 pp;
`sub_rate` 0.0013-0.0022; distinct lineups 0.161-0.255. The 11.6 caveat applies
unchanged: only the seed-to-seed SD is a floor, never the `minutes_mae` LEVEL of a
150-game run.

**Floor B**, a spec-identical refit under fit seed 101 and sim seed 23 on V3,
150 games (8.9 min):

| cell | seed 1 | seed 2 | \|delta\| | floor A |
|---|---:|---:|---:|---:|
| minutes MAE (150-game universe) | 23.0704 | 23.0521 | 0.0183 | 0.0170 |
| final 8:00 \|m\| <= 5 | 0.6422 | 0.6531 | **0.0109** | 0.0204 |
| final 8:00 \|m\| 6-15 | 0.6154 | 0.6057 | 0.0097 | 0.0131 |
| **final 8:00 \|m\| > 15** | 0.5818 | 0.5327 | **0.0491** | 0.0162 |
| H1 20:00-10:00 \|m\| <= 5 | 0.7557 | 0.7570 | 0.0013 | 0.0106 |
| starters at >= 4 fouls | 0.4380 | 0.4454 | 0.0074 | 0.0233 |
| distinct lineups | 15.180 | 15.620 | 0.4400 | 0.2337 |

V3's MAE gain over Z1 (0.2132, 12.5 floor-A units) is **11.6 refit-to-refit
spreads**;
its 2.9 pp close-band gain is **2.7 spreads**. Both clear floor B. **The blowout
band does NOT: its refit spread is 4.91 pp, larger than V3's whole 4.0 pp miss on
it, so that cell's failure cannot be resolved at this power and is labelled here
rather than argued either way.**

## 10. The fitted shrinkage constants, and what they say

Leave-one-fold-out over 5 folds of the 2024 training season, 23,156-23,538
held-out rows per fold, 113 s, over the grid declared in 23.5:

| object | 3 | 10 | 30 | 100 | 300 | 1000 | 3000 | selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arrival (Bernoulli, nats/row) | -0.38513 | -0.38496 | -0.38486 | **-0.38480** | -0.38484 | -0.38505 | -0.38543 | **3000** |
| size (multinomial) | -0.82201 | -0.81659 | -0.81320 | **-0.81137** | -0.81163 | -0.81388 | -0.81724 | **300** |
| entry (multinomial) | -0.61121 | **-0.61115** | -0.61118 | -0.61181 | -0.61433 | -0.62210 | -0.63971 | **100** |
| V0 arrival | -0.38711 | -0.38709 | **-0.38708** | -0.38708 | -0.38710 | -0.38721 | -0.38747 | 3000 |
| V0 size | -0.82128 | -0.82035 | **-0.81975** | -0.81984 | -0.82115 | -0.82378 | -0.82688 | 100 |

**The arrival object's held-out likelihood is FLAT: the whole grid spans 0.00063
nats per row, inside the pre-registered 0.001 tie band, so the declared tie rule
sent the constant to the grid's upper boundary, `k = 3000`.** That is not an
argmax and it is not reported as one. Its meaning is exact and it is the round's
second finding: **pooled over all 324 game-state cells, the (cell x `n_st`)
INTERACTION is worth nothing**, so V1/V3/V5's arrival table IS the product of the
two marginals, with the interaction shrunk to nothing. Size and entry do carry
interaction and their argmaxes are interior.

## 11. The parent, on a second object family

V4 is V3 with the shrinkage parent rounds 5-8 used -- the cell LEVEL alone --
and is otherwise identical:

| | arrival span | size span | close band | MAE | D8 | state |
|---|---:|---:|---:|---:|---:|---:|
| **V3 product parent** | **94%** | **79%** | **-11.8 pp** | **9.1000** | 0.436 | 4/8 |
| V4 cell-level parent | 60% | 36% | -14.1 pp | 9.2012 | 0.410 | 3/8 |

**V4 loses on every line.** Round 9's lesson -- that the parent of a thin cell must
be built from the powered marginals, not from a level -- is confirmed on a second
object family, prospectively, by a pre-registered contrast. That is the round's
most transferable result.

## 12. The measured next object

Not a guess. `P(wave | cell, n_st)` inside the cells where the gates fail,
400 games, actual against V3 and V5:

| cell | arm | `n_st`=1 | 2 | 3 | 4 | 5 |
|---|---|---:|---:|---:|---:|---:|
| **final 8:00 \|m\| <= 5** | ACTUAL | 0.2698 UP | **0.2351** | 0.1742 | 0.1471 | **0.0781** |
| | (n boundaries) | 63 UP | 536 | 2,382 | 2,963 | 1,254 |
| | V3 | 0.2027 | 0.1946 | 0.1800 | 0.1517 | **0.1070** |
| | V5 | 0.1786 | 0.1885 | 0.1926 | 0.1540 | 0.0899 |
| final 8:00 \|m\| 6-15 | ACTUAL | 0.1383 | 0.1680 | 0.1676 | 0.1379 | 0.0847 |
| | V3 | 0.1686 | 0.1909 | 0.1953 | 0.1651 | 0.1125 |
| H1 \|m\| <= 5 | ACTUAL | 0.1849 | 0.1780 | 0.1659 | 0.1293 | 0.0644 |
| | V3 | 0.1840 | 0.1705 | 0.1671 | 0.1259 | 0.0751 |

(UP = UNDERPOWERED at n < 300 and labelled, never read as signal.)

In H1 the product model is right to 0.1-1.1 pp at every level. **In the final 8:00
close band it is not: the real rate at two starters on the floor is 0.2351 against
a pooled 0.1822 -- a 1.29x lift -- and at five starters it is 0.0781 against a
pooled 0.0784, a lift of 1.00.** A coach late in a close game restores the
starters fast and then stops substituting entirely; the lift is a function of
`n_st`, which is exactly the interaction the pooled held-out likelihood of section
10 declared worthless and the tie rule then removed. Every arm sits at 0.090-0.107
at five starters against a real 0.078, which is 15-37% too many waves off the
lineup the close-and-late cell is graded on. **In the final 8:00 at \|m\| 6-15 the
error is a pure LEVEL: V3 is 2.3-3.1 pp high at every `n_st`.**

So the next object is named and sized: **the arrival rate in the final 8:00,
carried as a genuine (cell x `n_st`) interaction fitted where it is powered rather
than pooled over 324 cells** -- and, separately and blockingly, **Decision 8 as a
team-indexed round of its own.** A drift repair without a team index has now cost
Decision 8 three times running.

## 13. What was not run, and why

* **The Decision 10 freeze and the `ENGINE_ROTATION=round10` adapter were NOT
  run.** 23.12 pre-registered both as required ONLY IF an arm clears conditions
  1-5 offline. No arm does, so condition 6 is recorded as unmet by
  pre-registration and not by the wall clock. No engine file was touched by this
  lane; `ENGINE_ROTATION=round9` is still the newest flag and is still
  default-off.
* **The 16.7 exit-side marginal was not re-measured through
  `diag_rotation_exit_v9.py`**; it is recomputed from the round-10 accumulators
  as the leaver starter share (ACTUAL 0.5822, K1 0.5187, Z1 0.5698, V3 0.5602,
  V5 0.5719) and is reported as the report-only quantity it is. 21.8's caveat
  stands: a marginal is only interpretable when the rate and the distribution are
  both right.
* **Floor B was run on V3 only**, the arm 23.11 names when the decision rule
  selects nothing.
* One deviation from the pre-registration, stated: the counts pass **excludes the
  H1 -> H2 boundary**, because the sampler hard-resets there and runs no wave, so
  it is not a draw of this kernel; 23.1's measurement excluded it for the same
  reason. 23.3 did not fix the question either way. It makes V0 a refit of round
  5's model on round 10's rows rather than a copy of round 5's table, which is
  what a control is.

## 14. Verdict

**Adopt nothing. Change no default.** V3 and V5 are the best arms in ten rounds on
the object rounds 7-10 were built to fix and on per-player minutes MAE among
exit-corrected arms, and they are 14.0 and 11.4 noise floors behind K1, whose exit
rate carries 3% of the real composition span. The bake-off has now measured, twice
over, that **the mechanism that reproduces the floor and the mechanism that
reproduces per-player minutes are not the same mechanism**, and no round has yet
been specified that could be both.
