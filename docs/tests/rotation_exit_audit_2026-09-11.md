# Rotation EXIT audit (round-7 evidence) -- 2026-09-11

Worker: Opus. Inputs: `data/processed/possessions/possessions_{2024,2025}.parquet`
(on-floor sets), the round-3b base fits, round 4's fitted hazards
(`rotation_v4_sub_*.json`), round 5's wave tables
(`round5/rotation_v5_wave_*.json`) and round 6's composition tables
(`round6/rotation_v6_comp_*.json`) -- all four reused and never refitted.
Scripts: `scripts/train_rotation_v7.py`, `scripts/diag_rotation_exit_v7.py`.
JSON: `data/processed/models/rotation/rotation_F1_round7_results.json`,
`.../rotation_F1_round7_floorB.json`, `.../exit_audit_2026-09-11.json`.

Pre-registration: `docs/models/rotation/experiments.md` section 16, committed
**2aa29c2** before `rotation_v7.py` existed and before any object was fitted.
Results: section 17.

Coverage. Fitted tables: 6,000 team-games per S1 window, **124,260-126,402 waves
per window, 122,007-124,048 usable (98.2%)** -- the same counts, window by
window, that round 6 fitted on, so the exit and the entry tables come from
identical rows. Bake-off: the standing 1,600-game subset of 2025, 3 seeds per
arm, one blind grading path. Exit diagnostic: 200 games, seed 0, 400 team-games
per arm. Cells under 300 rows are labelled UNDERPOWERED and are never read as
signal or as absence of signal; the two places that bites are named in sections
2 and 6.

**One-line summary: round 7 closed the exit-side defect round 6 named -- the
class marginals are now right on both sides to within 1.5 pp -- and lost the
close-and-late state cells by 18-22 pp, because a class-count draw conditioned on
(size, state) is a level, not a rate, and the composition drifts with nothing to
pull it back.**

---

## 0. What this audit is for

Round 6 ended with a measurement, not a hypothesis (composition audit section 4):
every arm since round 5 takes a starter off on **0.456-0.467 of single swaps
against a real 0.587**, because rounds 5 and 6 both pre-registered round 5's rank
exit rule. K1, the best arm in six rounds on per-player minutes, was conditioning
correctly on a class that arrived 13 pp too rarely, and moved no state cell.

Round 7 built K1's mirror: `P(k_out | size, state)` drawn first, then the leavers
picked by the same rank rule within class (X1); the same table with a three-level
foul class (X2); the same table with the previous wave's entry class (X3). This
audit is the evidence for what those arms did, at every level the standing rule
requires.

---

## 1. The exit class is state-dependent, and the state term is large

`P(k_out = 1 | size 1, cell)` from the fitted tables, window 202411; the six S1
windows agree to 0.1-1.5 pp on every cell and the seed-101 refit to 0.1-0.7 pp.

| time cell | \|m\| <= 5 | 6-15 | > 15 | n (no foul trouble) |
|---|---:|---:|---:|---:|
| H1 | **0.6247** | 0.5781 | 0.5401 | 20,321 / 14,983 / 1,856 |
| H2 20:00-08:00 | 0.5961 | 0.6088 | 0.6381 | 8,762 / 10,542 / 4,451 |
| final 8:00 | **0.4851** | 0.5085 | 0.5748 | 5,199 / 5,721 / 2,749 |

| time cell (a player on the floor at >= 4 PF) | \|m\| <= 5 | 6-15 | > 15 | n |
|---|---:|---:|---:|---:|
| H1 | 0.5864 | 0.5864 | 0.5884 | **1 / 1 / 0 -- structurally empty** |
| H2 20:00-08:00 | 0.6100 | 0.5896 | 0.6217 | 173 / 264 / 193 -- **UNDERPOWERED** |
| final 8:00 | **0.5792** | 0.6036 | 0.6143 | 3,147 / 2,913 / 868 |

1. **13.9 pp of time-and-margin structure.** A single swap takes a starter off
   0.6247 of the time in the first half and 0.4851 in the final eight minutes of
   a close game: starters stop leaving when the game is close and late. This is
   the exit side's version of the entry side's decided-game cell (composition
   audit section 2) and it runs the other way.
2. **9.4 pp of foul structure in the cell that matters.** In the close-and-late
   cell, a player on the floor at >= 4 personal fouls raises the chance that the
   departing man is a starter from 0.4851 to 0.5792 (n = 5,199 / 3,147). This is
   exactly the effect X2 was pre-registered to carry, and it is real.
3. **Foul trouble does not exist in H1 at this threshold** (1, 1 and 0 rows in
   122,007): a player reaching his fourth foul before half-time is a
   once-a-window event. Three of X2's eighteen cells are therefore empty by
   construction and shrink to X1 exactly, which is what the declared `k = 300`
   shrinkage is for.
4. **X3's axis is half unidentified**: 25 of the 54 (exit cell x previous-wave
   class) cells at size 1 carry fewer than 300 rows. The pre-registration said an
   underpowered axis would be reported as unidentified rather than adopted, and
   it is.

---

## 2. The mechanism check: the arms hit the object they were built to move

`scripts/diag_rotation_exit_v7.py --sim-games 200`, seed 0, the as-of predicted
starter set on BOTH sides, one function for every row -- the round-6 table
extended to every swap size and to the entry side.

| | starter share of LEAVERS | size 1 | size 2 | size 3+ | starter share of ENTRANTS | size-1 joint: bench out / starter out | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ACTUAL (as-of starters)** | **0.5576** | **0.5867** | **0.5576** | **0.4931** | **0.4972** | 0.6848 / 0.3446 | 34.0 pp |
| K1 (round 6) | 0.4927 | 0.4564 | 0.5142 | 0.5496 | 0.4334 | 0.6344 / 0.2026 | 43.2 pp |
| **X1_exit_class** | **0.5611** | 0.6013 | 0.5648 | 0.4616 | **0.4882** | 0.6762 / 0.3073 | 36.9 pp |
| X2_exit_class_foul | 0.5591 | 0.5979 | 0.5609 | 0.4662 | 0.4874 | 0.6765 / 0.3050 | 37.1 pp |
| X3_exit_class_prev | 0.5585 | 0.5938 | 0.5552 | 0.4805 | 0.4881 | 0.6917 / 0.3026 | 38.9 pp |

n per size (X1 / ACTUAL): 5,546 / 5,282 at size 1, 2,091 / 1,980 at size 2,
746 / 738 at size 3+. No row here is underpowered.

Three readings:

1. **The 12-13 pp exit-side defect is closed.** 0.4927 -> 0.5611 overall against
   a real 0.5576, and 0.4564 -> 0.6013 at size 1 against 0.5867 (a 1.5 pp
   overshoot). Size 3+ goes the other way by 3.2 pp and is the only size the
   family gets worse.
2. **The entry side, which round 7 did not touch, improved for free**: K1's own
   conditional table is finally fed the right `k_out` mix, so the starter share
   of entrants goes 0.4334 -> 0.4882 against a real 0.4972 and the size-1 joint
   spread falls from K1's 43.2 pp (past the target) to 36.9 pp against 34.0. This
   is round 6's diagnosis confirmed by construction: the conditional was right
   and its input was wrong.
3. **Every swap-level class marginal is now right to within 1.5 pp on both sides,
   and the on-floor composition is 21 pp wrong** (section 3). That pairing is the
   finding of the round.

---

## 3. Per state cell: correct at the reset, 21 pp wrong twelve minutes later

The graded state cells, 1,600 games x 3 seeds, gap in pp against each side's own
real starting five:

| cell | ACTUAL | K1 | X1 | X2 | X3 | floor A (X1) |
|---|---:|---:|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7491 | -4.9 F | **-21.0 F** | -20.6 F | -21.9 F | 1.95 pp |
| final 8:00, \|m\| 6-15 | 0.7240 | -3.5 F | **-18.6 F** | -18.1 F | -19.6 F | 1.35 pp |
| final 8:00, \|m\| > 15 | 0.5223 | -1.4 P | -0.4 P | -0.4 P | -1.2 P | 2.20 pp |
| starters at >= 4 fouls | 0.4613 | -4.0 F | -7.9 F | -8.6 F | -7.8 F | 2.00 pp |
| H2 TIP, \|m\| <= 5 | 0.9678 | -2.9 P | -2.7 P | -2.7 P | -2.8 P | 0.79 pp |
| H2 TIP, \|m\| 6-15 | 0.9611 | -2.0 P | -1.9 P | -1.9 P | -1.9 P | 0.88 pp |
| H2 TIP, \|m\| > 15 | 0.9563 | -1.1 P | -1.1 P | -1.1 P | -1.3 P | 1.57 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0.7822 | -5.6 F | -8.3 F | -8.4 F | -8.4 F | 1.46 pp |
| **passed** | | **4/8** | **4/8** | **4/8** | **4/8** | |

Ordered by minutes since the last forced reset (the opening tip and the hard
second-half reset), the same numbers are a gradient:

| minutes since the last reset | cell | X1 gap | K1 gap |
|---:|---|---:|---:|
| 0 | H2 tip, all three bands | -1.1 to -2.7 pp | -1.1 to -2.9 pp |
| 0-10 | H1 20:00-10:00, \|m\| <= 5 | -8.3 pp | -5.6 pp |
| 12+ | final 8:00, \|m\| <= 5 | **-21.0 pp** | -4.9 pp |

**X1 is correct where the composition is forced and drifts monotonically away
from it, six times faster than K1 by the final eight minutes.** That is the
mechanism: `P(k_out | size, state)` does not depend on how many starters are on
the floor when it is drawn, so it removes starters at the population frequency
whatever the floor looks like. When the floor has drifted bench-heavy, the same
frequency is far above proportional and nothing pulls it back. Round 5's rank
rule carried an implicit restoring force -- `p_out` rises with time on the floor
and `p_in` with the player's own minutes share, so a starter who sits comes back
-- and round 7 replaced that force with an average that is only correct in
aggregate.

The decided-game cell (final 8:00, \|m\| > 15) is the one late cell every arm
passes, and it is passed for the right reason: it is the cell where the real
rotation IS bench-heavy (0.5223), so a model drifting toward the bench arrives at
the right answer.

---

## 4. Per player quintile: the family loses every quintile to both references

Per-player minutes MAE by quintile of the player's own pregame as-of minutes per
game (edges 15.97 / 20.89 / 25.33 / 29.71 mpg; 4,782-4,784 player-games per
quintile; none underpowered; floor 0.016).

| quintile | K1 (r6) | W4 (r6) | X1 | X2 | X3 | X1 - K1 | X1 - W4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 (lowest mpg) | 9.3108 | 9.6681 | 10.6053 | 10.5868 | 10.5998 | **+1.295** | +0.937 |
| Q2 | 9.7891 | 9.7294 | 10.4615 | 10.4331 | 10.4896 | +0.672 | +0.732 |
| Q3 | 9.2575 | 9.4074 | 9.7867 | 9.7658 | 9.8029 | +0.529 | +0.379 |
| Q4 | 8.6461 | 8.6621 | 9.1081 | 9.1084 | 9.1732 | +0.462 | +0.446 |
| Q5 (highest mpg) | 7.2367 | 7.3307 | 8.0130 | 7.9973 | 8.0315 | +0.776 | +0.682 |

Every cell loses, by 24 to 81 floors. Round 6's arms bought a pooled gain at the
ends of the rotation and gave a little back on the first men off the bench;
round 7's arms give back everywhere, and most at the two ends -- the deep bench
plays too much and the starters too little, in every game state after the drift
sets in.

The distributional instrument says the opposite, and that is worth recording:
**per-player minutes K-S D falls to 0.0441, the best ever measured** (K1 0.0589,
R2 0.0798), while the MAE rises 0.75 minutes. The round-7 arms put the right
NUMBER of minutes on the wrong PLAYERS. A distributional metric alone would have
called this round a win; the pre-registered pairing of a matching metric with a
distributional one is what caught it.

---

## 5. Per team: the compression becomes noise

Starters' share in the final 8:00 at \|m\| <= 5, aggregated per team; a team is
powered when both its simulated and its actual denominators reach 300 on-floor
slots (**251 powered, 112 UNDERPOWERED and excluded**).

| | sim mean | sim SD | actual mean | actual SD | corr(sim, actual) | mean \|dev\| |
|---|---:|---:|---:|---:|---:|---:|
| K1 (round 6) | 0.7004 | 0.0507 | 0.7512 | 0.0832 | **0.394** | 0.0757 |
| X1 | 0.5412 | 0.0599 | 0.7512 | 0.0832 | 0.209 | 0.2108 |
| X2 | 0.5451 | 0.0619 | 0.7512 | 0.0832 | 0.195 | 0.2067 |
| X3 | 0.5314 | 0.0614 | 0.7512 | 0.0832 | 0.198 | 0.2203 |

The cross-team SD rises (0.051 -> 0.060 against a real 0.083) while the
correlation with the team's own actual share halves. The added dispersion is not
matchup information: it is the drift of section 3 realising differently game to
game. The Decision 8 slope reads the same thing at quintile resolution: ACTUAL
+0.692, K1 +0.576 (ratio 0.83, inside the band), **X1 +0.426 (0.62), X2 +0.395
(0.57), X3 +0.390 (0.56), all outside the pre-registered [0.8, 1.2] band** -- a
team whose starters take 77% of its minutes is simulated 24 pp below its actual
late share, the same as a team at 56%.

---

## 6. Foul trouble: the benching response is right for the first time, the graded cell is worse

| | ACTUAL | R2 | W4 (r5) | K1 (r6) | X1 | X2 | X3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| "at exactly 4 fouls" (report-only) | 0.5166 | 0.5722 | 0.6028 | 0.5931 | **0.5316** | **0.5186** | **0.5328** |
| starters at >= 4 fouls (graded) | 0.4613 | 0.4626 | 0.4216 | 0.4212 | 0.3823 | 0.3752 | 0.3832 |
| as-of starter benchmark | 0.4636 (+0.2 pp of actual) | | | | | | |

**No arm in seven rounds had reproduced the "at exactly 4 fouls" diagnostic; all
three round-7 arms do** (0.519-0.533 against 0.5166, from K1's 0.593). The exit
table's foul term (section 1 item 2) is what does it: the fouled starter now
comes off at the rate the data says.

And the graded cell gets worse, because it asks a different question -- what
share of the on-floor slots held by players carrying >= 4 fouls belong to
starters -- and that is governed by who is on the floor at all, which section 3's
drift has already broken. **The foul cell is not readable until the drift is
fixed.** That reorders the queue round 6 left: composition first, then the foul
response. Note also that the benchmark sits +0.2 pp from the actual on this cell,
so unlike the close-and-late and opening cells, none of the miss is
starter-identification.

---

## 7. Identification and reproduction

**Floor B** (X3, fit seed 101 against 11, sim seed 23 against 7, same 150-game
universe): the fitted table moves 0.1-0.7 pp per cell
(`P(k_out = 1 | size 1)` 0.6247 -> 0.6240 in the H1 close cell, 0.4851 -> 0.4860
in the final-8:00 close cell) and the graded cells move 0.29-6.31 pp. The objects
are identified on the same reading as rounds 5 and 6, and the -21 pp miss the
round fails on is three times the largest refit-to-refit move and 11 floor-A SDs.

**The K1 reproduction check** (1 seed inside round 7 against round 6's 3-seed
column): every state cell moves less than its floor-A SD (largest -0.40 pp
against a 0.67 pp floor), the sub rate -0.0008 and distinct lineups -0.104, so
**the reference columns stand**. The MAE moves +0.056 against a 0.013 floor,
which is the expected 1-vs-3-seed Monte-Carlo difference and is why the check is
pre-registered on the state cells.

**Decision 10 was NOT RUN.** Wiring a vectorised `next_lineup_round7` into
`engine/rotation_adapter.py` plus two 500-game paired runs did not fit inside the
lane's 12:45 ET stop. The absence is reported rather than hidden; it costs
nothing on this round's outcome, since every arm already fails the offline state,
MAE, quintile and slope conditions by 4 to 80 floors.

---

## 8. The support check for the next object (measured after the decision)

`scripts/diag_rotation_exit_v7.py --by-composition`, run AFTER the round's
decision was read and written, on the ACTUAL sequences of both seasons with each
game's OWN starting five -- the descriptive convention of the round-6
composition audit. **It changes no verdict, no tolerance and no arm**; it exists
because section 9 names an object for round 8 and the project does not
pre-register an object whose support has not been measured.

`P(a starter is the man who leaves | single swap, starters on the floor)`:

| starters on the floor | 2024 | n | 2025 | n | proportional | 2025 / proportional |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3689 | 4,072 | 0.3697 | 5,981 | 0.20 | **1.85** |
| 2 | 0.4295 | 17,101 | 0.4374 | 23,159 | 0.40 | 1.09 |
| 3 | 0.5173 | 38,567 | 0.5209 | 48,014 | 0.60 | 0.87 |
| 4 | 0.6433 | 41,296 | 0.6562 | 48,261 | 0.80 | 0.82 |
| 5 | 1.0000 | 17,418 | 1.0000 | 19,996 | 1.00 | 1.00 |

Every cell is powered (4,072-48,261 leavers), the two seasons agree to 0.8-1.3 pp
on all five, and the shape survives the time split (2024, three starters on the
floor: 0.5396 H1, 0.4487 H2 20:00-08:00, 0.4500 final 8:00; four starters:
0.7405 / 0.6836 / 0.6537).

**The composition axis is worth 29 pp -- twice the 13.9 pp time-and-margin term
round 7 did model** -- and round 7's table is a mixture over it: the 0.5576
marginal X1 reproduces is the average of 0.37 at one starter on the floor and
0.66 at four.

With the round-6 audit's measured entry joint (P(starter in | bench out) 0.737,
P(starter in | starter out) 0.323 on 2025), the expected change in starters on
the floor per single swap is:

| starters on the floor | REAL exit rate | E[change] | LEVEL form (X1) | E[change] |
|---:|---:|---:|---:|---:|
| 4 | 0.656 | **-0.191** | 0.558 | -0.071 |
| 3 | 0.521 | **+0.000** | 0.558 | -0.071 |
| 2 | 0.437 | **+0.119** | 0.558 | -0.071 |

**The real process has a fixed point at three starters between swaps; the level
form has none** -- its drift is constant in the composition by construction, so
it walks down until the support clip binds. X1's measured drift is -0.073 per
leaver-slot (0.5611 out, 0.4882 in, section 2) against the predicted -0.071, and
its late floor settles at 2.70 starters (0.5394, section 3). **The mechanism of
section 3 is arithmetic, not interpretation.**

---

## 9. What this audit hands round 8

1. **The next object is `P(k_out | size, state, n_starters_on_floor)`** -- the
   same table with the current composition as a fourth axis, which makes the draw
   proportional-by-construction and restores the correcting force the rank rule
   had. Section 8 measures its support (4,072-48,261 leavers per composition
   cell, both seasons agreeing to 1.3 pp) and its size (29 pp). It is fittable
   from the same rows: the on-floor count is already in the training loop. The
   same argument applies to the entry table, which is also a level today.
2. **Keep the two measured state terms** (section 1): -13.9 pp from H1 to the
   close-and-late cell, +9.4 pp for foul trouble inside it. They survive the
   refit and they are not the reason round 7 failed.
3. **A composition rule must be graded on a matching metric and a distributional
   one together.** K-S D 0.0441 with MAE +0.75 is the signature of right
   marginals and wrong assignment, and only the pair detects it.
4. **The foul cell is downstream of the composition drift** (section 6) and
   should not be attacked until the composition is fixed.
5. **X2 and X3 add nothing over X1** (every cell inside the floor), so the family
   to carry forward is the single (size, state) table, with the composition axis
   added -- not the foul-class or previous-wave refinements.

---

## 10. Provenance

- Pre-registration `docs/models/rotation/experiments.md` section 16, commit
  2aa29c2, written before `rotation_v7.py` existed. Results section 17.
- Code: `src/cbb_sim/models/rotation_v7.py`, `scripts/train_rotation_v7.py`,
  `scripts/diag_rotation_exit_v7.py`, `tests/test_rotation_v7.py` (8 cases; 29
  passed with rounds 5 and 6).
- Artifacts: `data/processed/models/rotation/round7/rotation_v7_exit_{YYYYMM}.json`
  (+ `_seed2` siblings) with manifests; gitignored and HF-synced.
- Nothing was written to any round-3b, round-4, round-5 or round-6 artifact, no
  served default was changed, and no process this worker did not start was
  signalled.
