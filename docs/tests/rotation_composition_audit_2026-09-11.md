# Rotation COMPOSITION audit (round-6 evidence) -- 2026-09-11

Worker: Opus. Inputs: `data/processed/possessions/possessions_{2024,2025}.parquet`
(on-floor sets), the round-3b base fits, round 4's fitted hazards
(`rotation_v4_sub_*.json`) and round 5's wave tables
(`round5/rotation_v5_wave_*.json`), all three reused and never refitted.
Scripts: `scripts/diag_rotation_comp_v6.py` (`--seasons`, `--by-state`,
`--sim-check`), `scripts/train_rotation_v6.py`. JSON:
`data/processed/models/rotation/comp_audit_2026-09-11.json`,
`.../rotation_F1_round6_results.json`, `.../rotation_F1_round6_floorB.json`.

Pre-registration: `docs/models/rotation/experiments.md` section 14, committed
**4380a08** before any round-6 object was fitted. Results: section 15.

Coverage. Descriptive tables: **8,502 team-games / 177,666 waves (2024)** and
**10,640 team-games / 223,054 waves (2025)** -- every game with a complete
on-floor set on every possession, the universe
`rotation_wave_audit_2026-09-11.md` measured. Fitted tables: 6,000 team-games
per S1 window, 124k-126k waves each. Bake-off: the standing 1,600-game subset of
2025, 3 seeds per arm. Every rate carries a 95% Wilson interval in the JSON;
cells under 300 rows are labelled UNDERPOWERED and are never read as signal or
as absence of signal. **No cell reported below is underpowered**; the smallest
is n = 2,467.

---

## 0. What this audit is for

L33 closed round 5 with a verdict that named the next object: *"a joint draw
fixes the joint structure it models and exposes the next one: fixing WHETHER and
HOW MANY leaves WHO."* Round 5's own diagnosis (experiments.md 13.11 item 4) put
two candidates on the table -- (a) one fitted temperature on the entry weights,
(b) the composition conditioned on WHO LEFT -- and said (b) was the one its own
logic pointed at.

This document measures the object both candidates are about: the **joint of the
leaving class and the entering class**, which
`rotation_wave_audit_2026-09-11.md` section 4 measured only the **marginals** of
(a single swap takes a starter off 61% of the time and puts a starter on 48%).
It then reports what the round-6 arms do with it, at every level the standing
multi-level rule requires: overall, per game, per team, per player quintile.

---

## 1. The joint: a substitution REVERSES the class

`--seasons 2024 2025`, descriptive: each game's OWN starting five and its own
participant pool, at possession-boundary resolution. `k_out` is the number of
that game's starters among the leavers, `k_in` among the entrants.

**Size 1 (67% of all waves).**

| season | k_out | n | P(bench in) | **P(starter in)** |
|---|---:|---:|---:|---:|
| 2024 | 0 (a bench player left) | 46,340 | 0.2507 | **0.7493** |
| 2024 | 1 (a starter left) | 72,781 | 0.6842 | **0.3158** |
| 2025 | 0 | 57,385 | 0.2628 | **0.7372** |
| 2025 | 1 | 89,013 | 0.6773 | **0.3227** |

**The marginal 0.48 the wave audit reported is the average of a 0.75 / 0.32
split**, and the two seasons agree to 1.2 pp on both arms. The same pattern holds
at every size, monotonically:

| size | k_out | n (2024) | P(k_in = 0) | P(k_in = 1) | P(k_in = 2) | P(k_in = 3) |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 11,185 | 0.0314 | 0.1854 | **0.7832** | -- |
| 2 | 1 | 16,700 | 0.1372 | **0.5429** | 0.3199 | -- |
| 2 | 2 | 16,666 | **0.6186** | 0.3073 | 0.0740 | -- |
| 3 | 0 | 2,467 | 0.0118 | 0.0134 | 0.1265 | **0.8484** |
| 3 | 3 | 2,960 | **0.7172** | 0.2152 | 0.0676 | 0.0000 |

2025 reproduces every cell to under 2 pp. **A substitution is a class swap: the
players who come on are the counterpart class of the players who went off.**
That is a joint object; an unconditional race over the bench, which is round 5's
rule, has no term for it.

---

## 2. The joint by game state: stable except in a decided game

`--by-state`, size-1 swaps only, P(a starter enters).

| time | margin | k_out = 0 | n | k_out = 1 | n |
|---|---|---:|---:|---:|---:|
| H1 | \|m\| <= 5 | 0.7522 | 10,080 | 0.2936 | 19,209 |
| H1 | \|m\| 6-15 | 0.7427 | 8,784 | 0.3380 | 13,176 |
| H1 | \|m\| > 15 | 0.6916 | 1,349 | 0.3286 | 1,561 |
| H2 20:00-08:00 | \|m\| <= 5 | **0.8386** | 4,741 | 0.2988 | 7,963 |
| H2 20:00-08:00 | \|m\| 6-15 | 0.8172 | 5,677 | 0.3072 | 9,730 |
| H2 20:00-08:00 | \|m\| > 15 | 0.7469 | 2,406 | 0.3227 | 4,772 |
| final 8:00 | \|m\| <= 5 | 0.7461 | 5,408 | 0.3496 | 6,207 |
| final 8:00 | \|m\| 6-15 | 0.7201 | 5,551 | 0.3536 | 6,740 |
| final 8:00 | \|m\| > 15 | **0.5286** | 2,344 | 0.2688 | 3,423 |

2024 shown; 2025 agrees to 1-4 pp on every cell (JSON). Two readings:

1. **The class swap is a stable rule, not a state-dependent one, over seven of
   the nine cells** -- 0.72-0.84 when a bench player leaves, 0.27-0.35 when a
   starter does.
2. **The exception is the decided game**, where the final eight minutes at
   \|m\| > 15 drops to 0.5286 (2025: 0.5024) against 0.7461 close. Round 6's
   objects are state-free by pre-registration (14.9); this cell is the measured
   size of what a state term would add, and it is the round-7 candidate named in
   section 7.

---

## 3. The fitted objects (as-of pool, six S1 windows)

Fitted by `rotation_v6.build_comp_training` / `fit_comp` on the as-of candidate
pool and the as-of PREDICTED starters -- never the game's own five -- so the
object at fit time is the object at simulation time. 6,000 team-games per
window; 124,260-126,402 waves; 122,007-124,048 usable (leaver and entrant counts
equal and every entrant inside the as-of pool: 98.2%).

**T1's temperature is 1.00 in all six windows, and the grid is not flat.**

| window | tau (ML over the declared 20-point grid) | log-lik at tau = 1 | ll(0.5) - ll(1) | ll(2) - ll(1) | ll(5) - ll(1) |
|---|---:|---:|---:|---:|---:|
| 202411 | **1.00** | -272,443 | -24,378 | -47,003 | -336,369 |
| 202412 | **1.00** | -277,318 | -23,814 | -49,232 | -347,451 |
| 202501 | **1.00** | -282,844 | -23,038 | -49,722 | -349,743 |
| 202502 | **1.00** | -284,510 | -24,798 | -46,335 | -339,266 |
| 202503 | **1.00** | -279,262 | -19,720 | -43,819 | -320,464 |
| 202504 | **1.00** | -281,752 | -23,442 | -49,581 | -348,861 |

`tau = 1` IS round 5's W4 and `tau -> inf` is W1's rank rule. **The likelihood
rejects the rank direction by 320,000-350,000 log units and the flatter
direction by 20,000-25,000**, at every window, so there is no temperature between
the two knob-free endpoints that the data prefers to the one already in use.
Stated honestly: `p_in` was fitted by logistic maximum likelihood on these same
rows, so `tau = 1` being the argmax is close to a property of that fit; what is
NOT a tautology is the curvature, and the curvature says the round-5 diagnosis's
option (a) is empty.

**K1's table reproduces the descriptive joint on the as-of pool** (compare
section 1):

| size | k_out | P(k_in = 0) | P(k_in = 1) | P(k_in = 2) |
|---|---:|---:|---:|---:|
| 1 | 0 | 0.3009 | **0.6991** | -- |
| 1 | 1 | **0.6809** | 0.3191 | -- |
| 2 | 0 | 0.0538 | 0.2707 | **0.6755** |
| 2 | 2 | **0.5831** | 0.3372 | 0.0798 |

**A1's tier-pair log-affinity is real but small** (window 202501; rows are the
leaver's as-of tier, columns the entrant's; 0 = predicted starter, 1 = rank 6-7,
2 = rank 8-9, 3 = rank 10+):

| leaver \ entrant | T0 | T1 | T2 | T3 |
|---|---:|---:|---:|---:|
| T0 | -0.170 | +0.119 | **+0.300** | -0.192 |
| T1 | +0.041 | -0.084 | +0.076 | -0.272 |
| T2 | +0.050 | -0.064 | +0.085 | -0.277 |
| T3 | +0.027 | -0.003 | +0.062 | -0.260 |

The largest entry is an odds ratio of 1.35; every row suppresses the deep bench
(-0.19 to -0.28), which says the round-4 entry hazard over-weights tier 3.

**All three objects are identified.** Across the six windows `tau` never moves,
`logA[0, 2]` spans 0.250-0.313 and `P(k_in = 1 | size 1, k_out = 0)` spans
0.698-0.710. Under the floor-B refit (a different 6,000-team-game sample, fit
seed 101 against 11) `tau` is still 1.00 in all six windows, `logA[0]` moves
0.01-0.05 and the same `k_in` cell moves 0.001-0.011.

---

## 4. The mechanism check: do the arms REPRODUCE the joint?

`--sim-check`, 200 games, seed 0, the SAME as-of predicted starter set on the
simulated and on the actual side, so all four rows are like for like. Single
swaps only.

| | P(starter in \| bench out) | P(starter in \| starter out) | spread | n (bench out / starter out) | **share of swaps that take a STARTER off** |
|---|---:|---:|---:|---:|---:|
| **ACTUAL (as-of starters)** | **0.6848** | **0.3446** | **34.0 pp** | 2,183 / 3,099 | **0.587** |
| W4 (round 5, unconditional race) | 0.5415 | 0.3373 | 20.4 pp | 3,110 / 2,689 | 0.464 |
| A1 (tier affinity) | 0.5683 | 0.3117 | 25.7 pp | 2,951 / 2,586 | 0.467 |
| K1 (conditional class count) | 0.6344 | **0.2026** | **43.2 pp** | 2,992 / 2,512 | 0.456 |

Three things this settles, and the third is the round's finding.

1. **W4 under-represents the joint.** Its 20 pp spread against a real 34 pp is
   what an unconditional race can produce from the entry hazard's own features;
   the rest of the dependence is on who left, which it cannot see.
2. **K1 moves the object it was built to move, and overshoots on one arm.**
   Bench-out goes 0.542 -> 0.634 against 0.685, but starter-out goes 0.337 ->
   0.203 against 0.345, so the spread goes past the target rather than onto it.
3. **The binding error has moved to the EXIT side.** All three arms take a
   starter off at **0.456-0.467 of single swaps against a real 0.587** -- a
   12-13 pp miss -- because every round-6 arm inherits round 5's RANK exit rule
   by pre-registration. A conditional entry rule fed a wrong leaver mix
   propagates the error: K1 is conditioning correctly on a class that arrives
   with the wrong frequency. **This is measured, not inferred, and it is what
   round 7 should change.**

---

## 5. Multi-level bake-off evidence

Full tables in `experiments.md` section 15. What follows is the evidence at each
level the standing rule requires.

### 5.1 Overall

| | ACTUAL | R2 | W1 | W4 | T1 | K1 | A1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| state cells passed (of 8) | | 2 | 5 | 4 | 4 | 4 | 4 |
| round-5 cells passed (of 2) | | 2 | 1 | 2 | 2 | 2 | 2 |
| G8 cells passed (of 6) | | 5 | 2 | 3 | 3 | **5** | 3 |
| per-player minutes MAE | 0.0 | 9.7939 | 9.1244 | 8.9281 | 8.9266 | **8.8622** | 8.9167 |
| K-S D, per-player minutes | -- | 0.0798 | 0.1152 | 0.0747 | 0.0752 | **0.0589** | 0.0730 |
| K-S D, top-1 lineup share | -- | 0.2273 | 0.0843 | 0.1307 | 0.1297 | 0.0990 | 0.1273 |
| distinct lineups / team-game | 14.836 | 15.514 | 11.515 | 14.474 | 14.442 | 14.571 | 14.341 |
| substitutions per boundary | 0.1509 | 0.1437 | 0.1570 | 0.1570 | 0.1560 | 0.1554 | 0.1559 |

**K1 is the best arm on the primary metric in six rounds** (8.8622, beating W4 by
0.066 against a 0.016 floor and R2 by 0.93), has the best per-player minutes
distribution ever measured (K-S D 0.0589 against W2's 0.0733, the previous best),
and carries 5 of 6 G8 cells, which only the incumbent R2 has matched. It is
**not adopted**, for the reasons in 5.3 and in `experiments.md` 15.

### 5.2 Per game

Per-game per-player minutes MAE, averaged over seeds, over the 1,600-game
universe:

| | W4 | T1 | K1 | A1 |
|---|---:|---:|---:|---:|
| mean | 8.970 | 8.926 | **8.868** | 8.917 |
| SD across games | 2.634 | 2.211 | **2.239** | 2.223 |
| 90th percentile | 12.141 | 11.427 | **11.288** | 11.352 |

K1 is better than W4 at the mean AND in the tail: the worst decile of games
improves by 0.85 minutes per player. The improvement is not carried by a few
games.

### 5.3 Per player quintile (the pre-registered responsiveness condition, 14.7)

Players in the as-of rotation set, bucketed by their own pregame as-of minutes
per game (edges 15.97 / 20.89 / 25.33 / 29.71 mpg, 4,782-4,784 player-games per
quintile -- none underpowered):

| quintile | W4 | T1 | K1 | A1 | K1 - W4 |
|---|---:|---:|---:|---:|---:|
| Q1 (lowest mpg) | 9.6681 | 9.4597 | **9.3108** | 9.4504 | **-0.357** |
| Q2 | **9.7294** | 9.8622 | 9.7891 | 9.8707 | **+0.060** |
| Q3 | 9.4074 | 9.3831 | **9.2575** | 9.3193 | -0.150 |
| Q4 | 8.6621 | 8.6099 | 8.6461 | 8.6389 | -0.016 |
| Q5 (highest mpg) | 7.3307 | 7.2410 | **7.2367** | 7.2271 | -0.094 |

**Every round-6 arm loses Q2 to W4 beyond the floor** (0.016), and that single
cell is what the pre-registered rule 4 vetoes. The pattern is consistent across
all three arms and is therefore not seed noise: the conditional entry rules buy
their pooled gain at the bottom and the top of the rotation (Q1 -0.36, Q5 -0.09)
and give a little of it back on the **first men off the bench**, which is
precisely the class the `k_in` table moves.

### 5.4 Per team

Starters' share in the final 8:00 at \|m\| <= 5, aggregated per team over the
1,600 games; a team enters the powered set when both its simulated and its
actual denominators reach 300 on-floor slots.

| | teams powered | teams UNDERPOWERED | sim mean | sim SD | actual mean | actual SD | corr(sim, actual) | mean \|dev\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W4 | 251 | 112 | 0.7161 | 0.0662 | 0.7512 | 0.0832 | 0.381 | 0.0727 |
| T1 | 251 | 112 | 0.7103 | 0.0493 | 0.7512 | 0.0832 | 0.432 | 0.0683 |
| K1 | 251 | 112 | 0.7004 | 0.0507 | 0.7512 | 0.0832 | 0.394 | 0.0757 |
| A1 | 251 | 112 | 0.7098 | 0.0481 | 0.7512 | 0.0832 | **0.433** | 0.0686 |

**Every arm under-disperses across teams** (SD 0.048-0.066 against a real 0.083)
while correlating 0.38-0.43 with the team's own actual share. The 112
underpowered teams are labelled and excluded, not read as zero. This is the same
compression Decision 8's quintile slope reads (0.83-0.88 of the actual slope,
`experiments.md` 15.6) seen at team resolution instead of quintile resolution.

---

## 6. The as-of starter benchmark: which cells are reachable at all

Pre-registered as report-only (14.6). The ACTUAL on-floor sequence of the same
1,600 games, re-graded with the MODEL's as-of predicted starting five instead of
the game's own. The model's five overlaps the real five on **4.576 of 5**.

| cell | ACTUAL (own starters) | ACTUAL (as-of starters) | starter identification | best round-6 arm | arm vs BENCHMARK |
|---|---:|---:|---:|---:|---:|
| final 8:00, \|m\| <= 5 | 0.7491 | 0.7215 | **-2.8 pp** | 0.7096 (T1) | -1.2 pp |
| final 8:00, \|m\| 6-15 | 0.7240 | 0.6964 | **-2.8 pp** | 0.6929 (T1) | -0.4 pp |
| final 8:00, \|m\| > 15 | 0.5223 | 0.5048 | -1.7 pp | 0.5218 (A1) | +1.7 pp |
| starters at >= 4 fouls | 0.4613 | 0.4636 | +0.2 pp | 0.4295 (A1) | **-3.4 pp** |
| H2 tip, \|m\| <= 5 | 0.9678 | 0.9024 | **-6.5 pp** | 0.9385 (K1) | +3.6 pp |
| H2 tip, \|m\| 6-15 | 0.9611 | 0.8908 | -7.0 pp | 0.9413 (A1) | +5.1 pp |
| H2 tip, \|m\| > 15 | 0.9563 | 0.8909 | -6.5 pp | 0.9470 (A1) | +5.6 pp |
| H1 20:00-10:00, \|m\| <= 5 | 0.7822 | 0.7384 | **-4.4 pp** | 0.7331 (T1) | -0.5 pp |

This is the most useful single table in the round, and it changes no verdict:
every arm is still scored against each side's own real starting five, exactly as
in rounds 1-5.

1. **The opening-ten-minutes cell that no arm in six rounds has passed is
   -4.4 pp unreachable for starter-identification reasons alone.** Against the
   like-for-like benchmark the arms are -0.5 pp. The gate reads -4.9 pp and the
   tolerance is 3 pp, so the cell cannot be passed by any rotation rule until the
   as-of starter set improves -- which round 3 (5.4) already showed needs
   availability information (injury reports) the feature set does not have.
2. **Both close-and-late bands are within 1.2 pp of the benchmark**, so 2.8 of
   the 3.1-4.0 pp the gate reads is starter identification too.
3. **Foul trouble is NOT a starter-identification cell** (+0.2 pp) and the arms
   are -3.2 to -4.2 pp against it. **That failure is the model's own**, and it is
   the same one round 5's W4 carried (-4.0 pp). The "at exactly 4 fouls"
   diagnostic confirms the mechanism underneath is wrong in the same direction as
   in round 5: 0.59-0.61 against a real 0.517.
4. **The second-half tip is over-clean**: the arms put 3.6-5.6 pp MORE of their
   own predicted starters on the floor than the real sequence does, because the
   hard reset is deterministic while a real coach's second-half five is not.

---

## 7. What this audit hands round 7

1. **The exit rule is the binding defect, and the number is 12-13 pp** (section
   4): every arm since round 5 takes a starter off at 0.456-0.467 of single
   swaps against a real 0.587. Round 5 fixed WHETHER and HOW MANY, round 6 shows
   the entry side can be conditioned and that conditioning cannot pay while the
   leaver mix is wrong. The mirror of K1 -- `P(k_out | size, state)` drawn
   first, then the leavers raced within class -- is the object; it is the same
   counts pass, the same shrinkage constant, and the same engine primitives
   (`_pick_k` twice on the on-floor mask).
2. **The state term the composition needs is one cell, and it is measured**
   (section 2): the final eight minutes of a decided game, where P(starter in |
   bench out) is 0.53 against 0.75 everywhere else. Everything else is stable
   enough to stay state-free.
3. **Foul trouble is the only veto cell left that is the model's own** (section
   6): -3.2 to -4.2 pp against the like-for-like benchmark, with the "exactly 4
   fouls" share 0.59-0.61 against 0.517. It is a foul-process and
   benching-response question, not a composition question, and it will not be
   fixed by anything in the substitution draw.
4. **Two gate cells should be re-read, not relaxed.** The opening-ten-minutes
   cell and, in the other direction, the second-half tip are dominated by starter
   identification (-4.4 and -6.5 to -7.0 pp). No tolerance was changed in this
   round and none should be; the PM should decide whether the gate scores the
   sim against its own predicted five (as now) or against the benchmark, because
   those are different questions and only the second one is about rotation.
