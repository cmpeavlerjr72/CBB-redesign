# The late-game regime: the missing sub-model behind the OT shortfall (2026-09-11)

Lane: late-game regime. **Diagnosis and pre-registration only. Nothing is
fitted, no served default is changed, no arm is proposed as adopted.** Every
counterfactual below is arithmetic on measured distributions and is labelled as
such. A fix that raises the tie probability directly is banned by the standing
no-hand-tuning rule and nothing here proposes one.

Inputs:

    data/processed/possessions_v2/chances_2025.parquet   the ACTUAL 2024-25 pbp
    results/engine_v0/F2_2025_s200_v1_clockv3c_A          the 200-seed run A (reference)
    results/late_game/tap_s5                              NEW 200 games x 5 seeds, this lane
    results/late_game/actual_2025.txt                     full actual output
    results/late_game/compare_2025.txt                    full sim-vs-actual output
    results/late_game/actual_kernel.json                  the kernels, machine-readable

    scripts/diag_late_game_actual_v1.py    the actual measurement (new)
    scripts/diag_late_game_tap_v1.py       the per-possession tap (new, opt-in)
    scripts/diag_late_game_compare_v1.py   the comparison + attribution (new)

Carried in from `docs/tests/engine_v1_variance_ot_diag_2026-09-11.md` section 5:
a game goes to OT **iff** the regulation margin is 0, so G7's OT rate and the
regulation tie rate are the same number; the sim is at 0.539 of the actual and
that factors as `0.953` (margin 6% too wide) x `0.565` (missing pile-up at
zero); once tied the OT module is fine. This document is about the 0.565.

---

## 0. Definitions, and three caveats stated before any number

**The window.** The final 2:00 of REGULATION: `period == 2` and
`start_clock <= 120`. **The condition.** `|home margin at the 2:00 mark| <= 6`,
where the margin at the 2:00 mark is the score margin at the start of the first
possession with `start_clock <= 120` (actual: mean anchor 109.85 s, median
111.0; sim: mean 108.95 s, median 110.5 -- the two anchors agree, so the
comparison is on the same instant). **The reference window.** The rest of
regulation, same games, `period <= 2` and not in the window. Carrying the
reference window through every table is what separates a REGIME from a
SELECTION effect, and section 1.1 shows the separation is clean.

**Caveat 1 -- the universe.** The actual side is the pbp chances table: 5,593
games, and its pbp-accumulated regulation tie rate is **0.05096** against the
verified `game_finals_v2` rate of **0.0557** on 5,710 games. CLAUDE.md bans
grading against pbp-accumulated totals, and that ban is respected: the
**headline** tie rate is the verified 0.0557 and every gate line below is
stated against it. The window rates and the kernel can only be measured on pbp
(no other source carries possession state), so they are pbp quantities and the
(a)/(b) attribution is computed pbp-against-pbp on both sides so that the
universe cancels. Where a pbp number is used as a level, it is flagged.

**Caveat 2 -- `duration_s` is POST-OUTCOME.** The L5-flagged quantity: it runs
to the REBOUND on a miss and the MAKE on a make. Every duration line is
therefore a bound, and it is a bound on BOTH sides (the sim column is the
clock model's drawn duration, which is the pre-outcome quantity). The
DIRECTION of the sim-vs-actual gap survives that -- the sim's drawn duration is
already longer than the actual's post-outcome duration -- but the magnitude
does not, and no magnitude is claimed from it.

**Caveat 3 -- the tap is 1,000 simulations.** 200 games x 5 seeds. Every
per-cell kernel entry in section 2.2 sits on 30-57 simulations and is
**UNDERPOWERED**; it is shown so the shape is visible, never read as a level.
The two things read as levels are (i) the slate aggregates in section 1, which
sit on 2,993-43,867 possessions, and (ii) the (a)/(b) mixture in section 3,
which is corroborated independently by a construction that uses no sim kernel
at all (section 2.3). The tap's own tie rate is 0.0330 against run A's 0.0300
on 285,500 rows -- consistent, and run A's is the one to quote.

**Provenance of the sim side.** The tap pins run A's flags
(`ENGINE_CLOCK=v3c_srfloor_P3_s1`, `ENGINE_EVENT=round2_s1`,
`ENGINE_FG_MAKE=round4_B1`, ...), NOT the current served default
(`ENGINE_CLOCK=v5b_glat_pmean`, adopted at `e3ccce5`), so that it is comparable
with the gate read and the variance/OT diagnostic this lane follows from. The
tap wraps the clock and event adapters IN ITS OWN PROCESS ONLY, exactly as
`diag_pace_efficiency_poss_log_v1.py` established; `src/cbb_sim/` is unchanged.
**The default-clock re-read is an explicit unfinished item (section 6).**

---

## 1. The end-game table: ACTUAL vs SIM, final 2:00, `|margin at 2:00| <= 6`

### 1.1 Overall, and the regime-versus-selection separation

| | n | dur (s) | P(dur>25) | P(dur<=8) | FGA/poss | 3PA share | FTA/poss | TOV rate | bonus-FT rate | PPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ACT** rest of regulation | 312,742 | 15.78 | 0.1619 | 0.2634 | 0.7398 | 0.3830 | 0.2226 | 0.1492 | 0.0364 | 0.909 |
| **ACT** final 2:00 | 21,327 | 10.56 | 0.0987 | 0.5217 | 0.5626 | 0.4219 | 0.6002 | 0.1068 | **0.2775** | 0.973 |
| **SIM** rest of regulation | 43,867 | 17.74 | 0.2022 | 0.1845 | 0.7475 | 0.3933 | -- | 0.1568 | 0.0333 | -- |
| **SIM** final 2:00 | 2,993 | 13.35 | 0.1557 | 0.4013 | 0.5792 | 0.4123 | -- | 0.1322 | **0.2531** | -- |

Sim shares are the event model's own SERVED class probabilities on first
chances (the tap records the model's output, not a realisation, so they carry
no sampling noise beyond the state composition); sim durations are the clock
model's drawn durations. FTA/poss has no sim counterpart because the tap does
not record free-throw counts; `bonus-FT rate` is the comparable column.

**The regime is real and it is not selection.** In the reference window the
actual's trailing and leading possessions are indistinguishable -- 3PA share
0.3845 vs 0.3815, bonus-FT 0.0366 vs 0.0357, duration 15.74 vs 15.81, on
143,060 and 143,580 possessions. The same teams, in the same games, in the
final 2:00, split as far apart as any two cells in this project:

| final 2:00, by role at the 2:00 mark | n | dur | P(dur>25) | FGA/poss | 3PA share | FTA/poss | TOV | bonus-FT | PPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ACT** trailing offence | 10,503 | 9.60 | 0.0440 | 0.6914 | **0.4788** | 0.3776 | 0.0914 | **0.1564** | 0.917 |
| **ACT** leading offence | 9,217 | 11.45 | 0.1594 | 0.4126 | **0.3213** | 0.8630 | 0.1239 | **0.4216** | 1.042 |
| **ACT** tied | 1,607 | 11.70 | 0.1077 | 0.5812 | 0.3887 | 0.5476 | 0.1089 | 0.2421 | 0.938 |
| **SIM** trailing offence | 1,317 | -- | -- | 0.5879 | **0.4108** | -- | 0.1334 | **0.2421** | -- |
| **SIM** leading offence | 1,380 | -- | -- | 0.5684 | **0.4176** | -- | 0.1313 | **0.2660** | -- |

**This is the headline.** The actual's trailing-minus-leading gap is **-0.158**
on 3PA share and **+0.265** on the bonus-FT rate (the leading offence is the
one getting fouled). The sim's are **+0.007** (wrong sign) and **+0.024**. The
engine reproduces **0%** of the three-point role asymmetry and **9%** of the
fouling role asymmetry, while reproducing the CLOCK-conditioned averages almost
exactly (3PA share 0.4123 vs 0.4219, bonus-FT 0.2531 vs 0.2775). The defect is
not that the engine plays the last two minutes like the first two -- it plays
them like the last two minutes of an average game, and like the same last two
minutes for both teams.

### 1.2 By seconds remaining

| sec remaining | ACT dur | SIM dur | ACT 3PA sh | SIM 3PA sh | ACT bonus-FT | SIM bonus-FT | ACT FGA/p | SIM FGA/p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (90,120] | 16.20 | 18.13 | 0.3562 | 0.3598 | 0.1623 | 0.1426 | 0.6674 | 0.6794 |
| (60,90] | 15.88 | 17.21 | 0.3562 | 0.3658 | 0.1957 | 0.1860 | 0.6552 | 0.6408 |
| (30,60] | 11.39 | 10.74 | 0.3845 | 0.3917 | 0.2768 | 0.2714 | 0.5620 | 0.5603 |
| (10,30] | **6.56** | **10.60** | 0.4803 | 0.4599 | 0.3874 | 0.3485 | 0.4916 | 0.5016 |
| (0,10] | **2.89** | **10.35** | 0.6324 | 0.6183 | 0.3749 | 0.3280 | 0.4674 | 0.4968 |

**The clock-conditioned shot mix is right and the clock-conditioned DURATION is
not.** The event model's 3PA ramp (0.356 -> 0.632) and bonus-FT ramp (0.162 ->
0.375) track the actual to within 0.02-0.04 across all five buckets. The
duration **flatlines at ~10.4 s below 60 seconds remaining** where the actual
falls to 6.56 and then 2.89. The mechanism is named in the code: the served
`v3c_srfloor_P3` clock draws from empirical cells keyed on
`(prev_end, sr_bucket, period_group, bonus, tempo_tercile)` --
`src/cbb_sim/engine/clock_adapter_v3.py` `CELL_DIMS` -- and **`score_diff` is
not among them.** The clock model physically cannot shorten a trailing team's
possession or lengthen a leading team's, and cannot represent the actual's role
split at all:

| ACT, final 2:00 | (60,120] | (30,60] | (10,30] | (0,10] |
|---|---:|---:|---:|---:|
| trailing offence, duration | 13.52 | 10.38 | 7.77 | 3.53 |
| leading offence, duration | **18.38** | **12.14** | **4.88** | **2.15** |
| trailing offence, 3PA share | 0.3968 | 0.4452 | 0.5204 | 0.6609 |
| leading offence, 3PA share | 0.3138 | 0.2741 | 0.3177 | 0.5365 |
| trailing offence, bonus-FT | 0.1623 | 0.1507 | 0.1642 | 0.1566 |
| leading offence, bonus-FT | **0.2005** | **0.4307** | **0.6681** | **0.6502** |

The leading team holds the ball longer at 60-120 s (18.38 vs 13.52) and is off
it in 2.15 s inside 10 s because it is being fouled: **two thirds of a leading
team's possessions inside 30 seconds end in a bonus free-throw trip.** The
trailing team's bonus-FT rate is FLAT at 0.15-0.16 across the whole window --
it is not fouling anyone; it is being fouled at the league rate. The
intentional foul is a one-sided, clock-gated, role-conditioned behaviour and
nothing in the engine represents any of those three conditions.

### 1.3 By `|margin at 2:00|` bucket (ACTUAL)

| \|m@2:00\| | games | poss/game | dur | 3PA share | FTA/poss | bonus-FT | TOV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 174 | 9.24 | 11.70 | 0.3887 | 0.5476 | 0.2421 | 0.1089 |
| 1 | 340 | 9.34 | 11.57 | 0.4079 | 0.5534 | 0.2523 | 0.1052 |
| 2 | 362 | 9.54 | 11.31 | 0.4223 | 0.5472 | 0.2509 | 0.1138 |
| 3 | 331 | 10.12 | 10.63 | 0.4210 | 0.5955 | 0.2728 | 0.1110 |
| 4 | 319 | 10.60 | 10.03 | 0.4181 | 0.6246 | 0.2905 | 0.1109 |
| 5 | 290 | 10.80 | 9.78 | 0.4367 | 0.6450 | 0.3056 | 0.0961 |
| 6 | 285 | 11.34 | 9.42 | 0.4443 | 0.6648 | 0.3123 | 0.1012 |

Monotone on every line: the further behind, the more possessions are
manufactured (9.24 -> 11.34 per game, +23%), the shorter they are (11.70 ->
9.42 s) and the more fouling there is (0.242 -> 0.312). The sim produces
**8.855** possessions per game in the window against 9.24-11.34, and has no
slope in `|m@2:00|` to produce because its clock never sees the margin.

### 1.4 Per team quintile (the matchup-specific rule)

Offence teams split into quintiles by season PPG (>= 10 games), final-2:00
window, actual:

| quintile | n poss | 3PA share | FTA/poss | bonus-FT | dur | reference-window 3PA share |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (lowest PPG) | 4,186 | 0.4204 | 0.5397 | 0.2463 | 10.89 | 0.3793 |
| 2 | 4,272 | 0.4314 | 0.5789 | 0.2678 | 10.70 | 0.3761 |
| 3 | 4,295 | 0.4178 | 0.6042 | 0.2754 | 10.52 | 0.3836 |
| 4 | 4,336 | 0.4100 | 0.6179 | 0.2878 | 10.43 | 0.3802 |
| 5 (highest PPG) | 4,238 | 0.4302 | 0.6593 | 0.3096 | 10.25 | 0.3955 |

**The fouling channel slopes with team quality (0.246 -> 0.310, +26% across
the quintiles) and the three-point channel does not (flat at 0.41-0.43 against
a reference-window 0.376-0.396).** Read carefully: the PPG quintile is a proxy
for both quality and pace and this is a one-way cut, so the FTA/bonus-FT slope
is a responsiveness OBSERVATION to be reproduced, not an identified team
effect. What it does establish is that a league-average end-game regime is
already refuted by the data at the team level on the fouling channel, so a
candidate that conditions only on clock and margin is pre-registered as
insufficient on this cut (section 5, gate R3).

---

## 2. The kernel: where ties come from, and how far the engine is from making them

### 2.1 Where ties come from (ACTUAL)

| \|m@2:00\| | games | P(\|m@2:00\|) | P(reg tie \| m) | P(\|reg\|=1 \| m) | share of ALL ties |
|---:|---:|---:|---:|---:|---:|
| 0 | 174 | 0.0311 | 0.2012 | 0.0977 | 0.123 |
| 1 | 340 | 0.0608 | 0.1765 | 0.1559 | 0.211 |
| 2 | 362 | 0.0647 | 0.1685 | 0.1326 | 0.214 |
| 3 | 331 | 0.0592 | 0.1148 | 0.1088 | 0.133 |
| 4 | 319 | 0.0570 | 0.1223 | 0.0752 | 0.137 |
| 5 | 290 | 0.0519 | 0.0690 | 0.0586 | 0.070 |
| 6 | 285 | 0.0510 | 0.0597 | 0.0597 | 0.060 |
| 7 | 264 | 0.0472 | 0.0227 | 0.0303 | 0.021 |
| 8 | 257 | 0.0460 | 0.0195 | 0.0311 | 0.018 |
| 9 | 247 | 0.0442 | 0.0081 | 0.0122 | 0.007 |
| >= 10 | 2,624 | 0.4690 | 0.0004 | 0.0004 | 0.007 |

**93.4% of all regulation ties come from a margin of 6 or less at the 2:00
mark, and 68% from 4 or less.** A model that is wrong about the last two
minutes of a one-possession game is wrong about essentially the whole of G7.
Note also the trough at `|reg|=1` is already visible inside the kernel: from
`|m@2:00| = 0` the game is twice as likely to end tied (0.201) as to end at one
point (0.098). That is the fingerprint of a team that will take a two or a
three to tie rather than a one to trail by one, and it is the shape section 5
of the variance/OT diagnostic saw from the other end.

### 2.2 The kernel, three ways (per-cell entries UNDERPOWERED, shape only)

`P(land exactly on 0 at the end of regulation | |margin| at the 2:00 mark)`:

| \|m@2:00\| | **SIM** | **ACT** | **MID K0** | SIM/ACT | SIM/K0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0625 | 0.2012 | 0.1358 | 0.31 | 0.46 |
| 1 | 0.1930 | 0.1765 | 0.0932 | 1.09 | 2.07 |
| 2 | 0.1591 | 0.1685 | 0.1141 | 0.94 | 1.39 |
| 3 | 0.0392 | 0.1148 | 0.0834 | 0.34 | 0.47 |
| 4 | 0.0926 | 0.1223 | 0.0549 | 0.76 | 1.69 |
| 5 | 0.0851 | 0.0690 | 0.0334 | 1.23 | 2.55 |
| 6 | 0.0000 | 0.0597 | 0.0231 | 0.00 | 0.00 |
| 7 | 0.0213 | 0.0227 | 0.0089 | 0.94 | 2.38 |
| 8 | 0.0182 | 0.0195 | 0.0037 | 0.94 | 4.97 |

**`MID K0` is a regime-free kernel built from the actual data itself** and it is
the construction that makes this section trustworthy despite the tap's sample
size. It is the distribution of the two-minute margin CHANGE over FIRST-HALF
windows (`period == 1`, seven anchors from 240 s to 960 s, 39,124 rows) at the
same starting margin: real basketball, real scoring rates, and no late-game
strategy, which is exactly what a model with no end-game regime produces. Its
delta SD is **3.172** against the late window's **3.232** -- the two windows
move the margin by the SAME amount, so K0 is not a weaker kernel, it is a
differently-SHAPED one. Its excess kurtosis is **-0.096** against the late
window's **+0.196**.

**ACT/K0 is above 1 in every powered cell (1.38 to 5.32) and rises with the
margin.** That ratio IS the regime: at the same starting margin and the same
total movement, real late-game play lands on zero 1.4-2.6x more often than
mid-game play does, and 5x more often from eight points back, because the
trailing team manufactures extra possessions and targets the exact tie.

### 2.3 The construction that needs no sim at all

Mixing the ACTUAL `|m@2:00|` distribution against each kernel:

| | P(reg tie) |
|---|---:|
| ACT distribution x **ACT late kernel** | **0.05096** (= the observed pbp tie rate, a validation of the arithmetic) |
| ACT distribution x **MID K0 kernel** | **0.02892** |
| kernel factor (late / mid) | **1.762** |

**A model with the CORRECT margin-at-2:00 distribution and a regime-free
kernel produces a 0.0289 tie rate. Run A produces 0.0300 on 285,500 rows; this
tap produces 0.0330 on 1,000.** The engine's tie rate is, to within the
seed noise, exactly the number you get by playing the last two minutes like the
middle of the game. That single line is the strongest evidence in this document
and it uses no sim kernel, no Gaussian assumption and no counterfactual.

---

## 3. Attribution: (a) the upstream 2:00-margin distribution vs (b) the kernel

### 3.1 The 2:00-margin distribution (channel a)

| \|m@2:00\| | SIM P | ACT P | ratio |
|---:|---:|---:|---:|
| 0 | 0.0320 | 0.0311 | 1.03 |
| 1 | 0.0570 | 0.0608 | 0.94 |
| 2 | 0.0440 | 0.0647 | 0.68 |
| 3 | 0.0510 | 0.0592 | 0.86 |
| 4 | 0.0540 | 0.0570 | 0.95 |
| 5 | 0.0470 | 0.0519 | 0.91 |
| 6 | 0.0530 | 0.0510 | 1.04 |
| **<= 6 (total)** | **0.3380** | **0.3757** | **0.900** |
| 14 | 0.0440 | 0.0279 | 1.58 |
| 15 | 0.0340 | 0.0263 | 1.29 |

SD of the margin at the 2:00 mark: SIM 15.71. The sim is 10% short of the
actual's mass inside 6 points and long in the 13-15 band -- the same "6% too
wide, and no pile-up" signature the regulation margin carries, one window
earlier. This is real but it is small, and the per-cell entries at 1,000
simulations carry roughly +/-0.007 of binomial noise, so only the aggregate
`<= 6` line (0.900) is read.

### 3.2 The (a)/(b) split

Four cells on the same kernel support (`|m@2:00| <= 15`), all arithmetic on
measured distributions, not re-simulations:

| | P(reg tie) |
|---|---:|
| sim distribution x sim kernel | 0.03300 |
| **ACT** distribution x sim kernel -- **(a) alone** | 0.03783 |
| sim distribution x **ACT** kernel -- **(b) alone** | 0.04562 |
| ACT distribution x ACT kernel | 0.05096 |
| total gap | **0.01796** |

| channel | contribution | share of gap |
|---|---:|---:|
| **(a) the upstream margin-at-2:00 distribution** | +0.00483 | **+26.9%** |
| **(b) the end-game kernel -- the missing regime** | +0.01262 | **+70.3%** |
| interaction | +0.00051 | +2.8% |

**(b) carries 70% of the tie deficit, (a) 27%.** Two independent readings agree
with that split and neither uses the sim's own kernel:

1. Section 2.3: the regime-free kernel on the CORRECT 2:00 distribution gives
   0.0289 against the actual 0.0510 -- a kernel-only shortfall of 0.0221, i.e.
   **1.76x**, against the total shortfall of 0.0221+ from a tie rate of 0.0300.
   On that construction the kernel accounts for essentially the whole of it and
   (a) is the residual.
2. The variance/OT diagnostic's own factoring, one window later: `0.539 =
   0.953 x 0.565`. The scale factor (the margin being 6% too wide) is the same
   family of defect as (a) and carries 4.7 of the 46.1 shortfall points -- 10%;
   the shape factor carries 90%. The 27/70 split here is the same story
   measured at 2:00 instead of at 0:00, and the difference between 27% and 10%
   is that (a) as defined here also absorbs the sim's excess mass in the 13-15
   band, which the SD-only factoring does not.

**Read together: the missing end-game regime is the dominant channel, and
whoever fixes the upstream margin distribution will still be left with about
seven tenths of G7.** Conversely the regime lane must not be graded as if it
owned all of G7: a successful regime layer that leaves (a) untouched
predicts a tie rate near 0.0289 + 0.70 x (0.0510 - 0.0289) = **~0.044**, not
0.056. That number is pre-registered as the regime layer's own target in
section 5, and it is why the closed-loop gate is a two-sided band rather than
"hit 0.0557".

### 3.3 Within (b): which behaviour carries the most

This is a RANKING BY MEASURED DEFICIT with a stated mechanism, **not** a
counterfactual -- no single-channel re-simulation was run, and running one is
the pre-registered experiment (section 5, arm D). Ranked:

**1. Fouling / the free-throw channel.** Largest measured asymmetry deficit and
the most direct mechanism. The actual's leading-minus-trailing bonus-FT gap is
**+0.265**; the sim's is **+0.024**, i.e. **9% reproduced**. Inside 30 seconds
the actual leading offence ends **0.67** of its possessions at the line; the
sim has no cell that can produce that. The mechanism is arithmetic: free
throws are the only **one-point** increment in basketball, and section 4 shows
that of the 282 tie games whose last scoring possession in the window landed
the game on exactly zero, **34.0% did it from the free-throw line** on 25% of
window possessions -- the only over-represented channel -- and `P(exactly 1
point)` on that possession is **0.128**. A distribution cannot pile up on a
single integer without a one-point increment available at the right moment, and
the engine's late-game FT supply is role-blind.

**2. Duration and the hold (the number of end-game chances).** The served clock
carries **no `score_diff` at all** (`CELL_DIMS` = prev_end x sr_bucket x
period_group x bonus x tempo). Measured consequences: window duration 13.35 vs
10.56, P(dur <= 8) 0.401 vs 0.522, and the flatline at ~10.4 s inside 60
seconds against an actual 6.56 then 2.89. The engine plays **8.855**
possessions in the window against an actual 9.24 (level game) to 11.34 (six
points down): it gives the trailing team **up to 22% fewer chances to reach
zero**, and it gives the leading team no way to hold the ball and deny them.
Ranked second rather than first only because its effect on the pile-up is
indirect -- it scales the number of attempts rather than the granularity of the
increments -- and because its aggregate ramp (18.1 -> 10.4) is at least in the
right direction, where the fouling role split has the wrong sign.

**3. The three-point share.** The **smallest** defect of the three and the one
most at risk of being over-fixed. In aggregate the engine is already right
(0.4123 vs 0.4219) and its clock ramp tracks the actual across all five buckets
(0.360/0.366/0.392/0.460/0.618 against 0.356/0.356/0.385/0.480/0.632). Only the
role split is missing (sim +0.007 vs actual -0.158). And threes are the
channel LEAST able to land a game on zero: 24.8% of the tie-landing plays,
below their 24.6% share of window possessions, against free throws' 34.0% on
25%. A three ties a game only from exactly three down.

Turnovers are named and set aside: the sim's window TOV rate is 0.1322 against
0.1068, a real 2.5 pp excess that costs the trailing team possessions, but it
is a quarter of the size of the other three gaps and it is a rate the existing
possession-outcome model already owns with the right features.

---

## 4. Which play lands the game on zero (ACTUAL)

282 regulation-tie games whose last scoring possession inside the final 2:00
brought the margin to exactly 0:

| how the game was tied | share | n |
|---|---:|---:|
| a made two | 0.411 | 116 |
| **pure free throws** | **0.340** | **96** |
| a made three | 0.248 | 70 |

Points on that possession: mean 2.156; `P(1) = 0.128`, `P(2) = 0.589`,
`P(3) = 0.284`.

Share of ALL final-2:00 possessions by terminal event, for comparison:
bonus FT trip 0.249, FGA_3 0.246, FGA_rim 0.228, TOV 0.119, FGA_jump2 0.102,
shooting FT trip 0.026.

**The free-throw channel is the only one over-represented among tie-landing
plays relative to its share of possessions (0.340 against 0.249), and it is the
only source of the one-point increment that `P(1) = 0.128` records.** That is
the quantitative basis for ranking fouling first in section 3.3.

---

## 5. Verdict on ownership, and what is pre-registered

**This is a distinct REGIME LAYER, not a missing column in one sub-model.**
Three pieces of evidence, and the second is the one that decides it:

1. The served clock has no `score_diff` in its cell grid at all, so the
   duration half is structurally unreachable from the current clock arm.
2. **`possession_outcome`'s served arm ALREADY has `score_diff`,
   `seconds_remaining` and `in_bonus`** (`C_plus_state`, `STATE_FEATURES` in
   `src/cbb_sim/models/possession_outcome.py`), and it still delivers 9% of the
   fouling asymmetry and 0% of the three-point asymmetry while getting the
   clock-conditioned averages right. **The defect therefore cannot be diagnosed
   as a missing feature.** It is a conditional law that a single pooled fit
   averages away, because the regime is a small, sharply-bounded region of the
   state space (2.6% of regulation possessions) with a sign flip in it.
3. The gate is joint -- tie rate AND the margin density near zero AND kurtosis
   -- and no existing sub-model's `experiments.md` owns a gate of that shape.

Accordingly: the full pre-registration lives in a new
**`docs/models/late_game/experiments.md`** (section 1, PROPOSED), with
`model.md` and `features.md` alongside it per
`docs/models/DOCUMENTATION_STANDARD.md`, and short PROPOSED cross-reference
sections are appended to `docs/models/possession_outcome/experiments.md` and
`docs/models/clock/experiments.md` so neither lane can run a round that
collides with it unknowingly. **Nothing is adopted, nothing is run, and no
served default is changed by any of it.**

The five lines of the proposal, in full in `docs/models/late_game/experiments.md`:

- **Candidates.** A = reference (the served cascade, unchanged). B = state
  enrichment inside the existing sub-models (role, clock-gated foul-margin and
  hold indicators added to `possession_outcome`'s and the clock's feature sets,
  one pooled fit). C = a regime-conditioned REFIT (the same model classes,
  refit on the window only, served by a hard gate on `period == 2 and
  seconds_remaining <= 120`). D = a dedicated end-game model over an explicit
  `(role, sec_remaining, margin, bonus, possession arithmetic)` state.
- **Folds.** Fold 1 trains through 2022-23 / tests 2023-24; fold 2 trains
  through 2023-24 / tests 2024-25; **fold 2 selects**. 2025-26 stays sealed.
- **Primary metric.** Offline: multiclass log loss on window possessions only.
  Closed-loop primary: the regulation-margin density at `|margin| <= 2` --
  specifically `P(0)`, `P(1)` and their ratio -- with the tie rate and the
  regulation-margin excess kurtosis as the two named secondaries.
- **Noise floor.** A spec-identical retrain under a second seed offline; a
  seed-offset paired run at matched seed count for the closed loop (the 20-seed
  floor already on record puts G7 at 0.0009 and the margin SD ratio at 0.0060).
- **Decision rule.** Beat the floor on fold 2 on the primary; show the role
  split with the right SIGN on both the fouling and three-point cuts; slope with
  the team quintile on the fouling cut; ties go to the simpler arm (A > B > C >
  D). No arm ships without a paired-seed closed loop in which no gate
  regressed, per Decision 10.

---

## 6. What this document does NOT establish

1. **The sim side is run A's flag set, not the current served default.** The
   clock lane adopted `v5b_glat_pmean` at `e3ccce5` after run A. The duration
   lines in section 1.2 must be re-read against the default before any clock
   arm is pre-registered off them. That re-read is the first unfinished item.
2. **No single-channel counterfactual was simulated.** Section 3.3 is a ranking
   by measured deficit plus a stated mechanism. Which channel actually moves the
   tie rate most is arm D's own pre-registered measurement, not a claim here.
3. **The tap is 1,000 simulations.** Every per-cell kernel entry is
   underpowered and labelled. The (a)/(b) split inherits that noise; it is
   carried because two independent constructions agree with it, not because
   0.269/0.703 is a precise number. A 200-seed re-read is the second unfinished
   item.
4. **`duration_s` is post-outcome on the actual side** and pre-outcome on the
   sim side. The sign of the gap survives; the magnitude does not.
5. **The actual side is the pbp universe** (5,593 games, tie rate 0.05096)
   against the verified 5,710 / 0.0557. The attribution cancels the universe by
   construction; the headline levels are quoted from the verified source.
6. **The team-quintile cut is one-way.** PPG proxies for both quality and pace.
   It refutes a league-average regime; it does not identify a team effect.
7. **Nothing here says the regime layer is the whole of G7.** Section 3.2
   pre-registers ~0.044, not 0.0557, as what a regime-only fix predicts, and
   the residual belongs to whoever owns the margin-at-2:00 distribution.
