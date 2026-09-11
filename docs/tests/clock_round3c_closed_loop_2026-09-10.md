# Clock round 3c -- the Decision-10 closed-loop run

Pre-registration: `docs/models/clock/experiments.md` section 12, committed
**a08635f** before any round-3c code was written and before the first engine
process started. Harness commit **1450faf**. Runs executed 2026-09-10 21:50 to
2026-09-11 00:04 local.

**Verdict: NO ARM ADOPTED. 0 of 6 candidates pass G1 on either game set.**
The round nevertheless answers the question round 3b could not (section 11.2 of
`experiments.md`), and it produces a sharper diagnosis of what is left than any
previous round: the residual is a **uniform mean-possession-duration shortfall
of about 0.34 s**, not an end-of-period artefact and not the `score_diff` loop.

---

## 1. What was run

Seven engine arms over the pre-registered subset: the F2 2025 slate sorted by
`game_id` ascending, every 11th row, first 500 games (2024-11-04 to 2025-03-15,
159 clock-complete). Paired streams by construction -- the engine seeds on
(seed, game_id, family), so two arms on the same games and seeds consume aligned
streams and differ only where the clock model differs.

Every other sub-model was pinned by an explicit environment value on every run
and is recorded in each `run_meta.json`:
`ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round2b_S_C_s1`,
`ENGINE_FG3=decision8`, `ENGINE_ROTATION=reference`.

| id | `ENGINE_CLOCK` | model | state | scheme |
|---|---|---|---|---|
| I | `reference` | round-1 `lgbm_quantile` | round-1 `C_plus_score` | static (incumbent) |
| B3 | `v3c_srfloor_P3_s1` | `empirical_km3_srfloor` (cell-based) | P3 engine-safe | S1 monthly |
| B1 | `v3c_srfloor_P1_s1` | `empirical_km3_srfloor` (cell-based) | P1 margin live | S1 monthly |
| A3 | `v3c_gamma_P3_s1` | `gamma_aft` | P3 engine-safe | S1 monthly |
| A2 | `v3c_gamma_P2_s1` | `gamma_aft` | P2 margin deleted | S1 monthly |
| A1 | `v3c_gamma_P1_s1` | `gamma_aft` | P1 margin live | S1 monthly |
| F | `v3c_gamma_P1_s1` + `ENGINE_CLOCK_FREEZE=1` | `gamma_aft` | P1 with the margin held at 0 | S1 monthly |
| F3 | `v3c_gamma_P3_s1` + freeze | `gamma_aft` | P3 with the margin held at 0 | S1 monthly |

F3 is an **added diagnostic, not pre-registered and not a candidate**; it was run
at 5 seeds only and is reported so the freeze can be read on both
parametrisations. F is the pre-registered Decision-10 frozen arm; the
pre-registration already states that neither is adoptable, because freezing a
feature is an ablation, not a model.

`gamma_aft|P3` is served straight out of round 3b's `v3b_s1/` directory, so that
arm is byte-identical to the schedule round 3b scored. The other four schedules
were fitted by `scripts/train_clock_v3c_s1.py` into `v3c_s1/`; round 3b's
artifacts were never written.

---

## 2. The closed-loop table

Every arm row is a **delta against the same 500 games' own actuals**, never
against the season figure. Actual levels on the subset: possessions per
team-game **68.530** clock-complete / **68.328** all; margin SD 15.472;
home/away score correlation 0.237; PPP 1.0705; OT rate 6.80%.

### Screening read, 5 seeds

| id | arm | G1 cc mean | G1 cc SD | G1 all mean | G1 all SD | margin SD | corr(h,a) | total bias | PPP | OT% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I | incumbent | +2.662 | -0.327 | +3.394 | -0.159 | 16.418 | 0.045 | +3.007 | 1.0408 | 3.88 |
| B3 | srfloor P3 + S1 | **+1.072** | -0.603 | **+1.582** | -0.632 | 15.983 | 0.006 | -0.961 | 1.0397 | 2.64 |
| B1 | srfloor P1 + S1 | +1.022 | -0.512 | +1.620 | -0.570 | 16.131 | -0.000 | -0.749 | 1.0406 | 3.60 |
| A3 | gamma P3 + S1 | +1.403 | -0.191 | +1.863 | -0.045 | 16.330 | 0.055 | -0.043 | 1.0417 | 2.92 |
| A2 | gamma P2 + S1 | +1.379 | -0.048 | +1.851 | +0.072 | 16.220 | 0.070 | -0.070 | 1.0417 | 2.92 |
| A1 | gamma P1 + S1 | +1.889 | +0.266 | +2.219 | +0.220 | 15.883 | 0.110 | +0.744 | 1.0420 | 4.72 |
| F | gamma P1, margin FROZEN | -2.601 | -0.438 | -2.191 | -0.417 | 15.868 | 0.047 | -9.120 | 1.0369 | 3.60 |
| F3 | gamma P3, margin FROZEN | +0.366 | -0.231 | +0.818 | -0.151 | 16.159 | 0.050 | -2.398 | 1.0404 | 2.96 |

### Deciding read, 25 seeds

| id | arm | G1 cc mean | G1 cc SD | G1 all mean | G1 all SD | margin SD | corr(h,a) | total bias | PPP | OT% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I | incumbent | +2.697 | -0.058 | +3.411 | -0.090 | 16.255 | 0.069 | +2.830 | 1.0391 | 3.94 |
| B3 | srfloor P3 + S1 | **+1.161** | -0.493 | **+1.666** | -0.582 | 15.878 | 0.018 | -0.934 | 1.0385 | 2.76 |
| B1 | srfloor P1 + S1 | +1.243 | -0.285 | +1.741 | -0.417 | 15.806 | 0.033 | -0.737 | 1.0388 | 3.77 |
| A3 | gamma P3 + S1 | +1.469 | +0.133 | +1.921 | +0.068 | 16.074 | 0.075 | -0.136 | 1.0402 | 2.70 |
| A2 | gamma P2 + S1 | +1.458 | +0.146 | +1.872 | +0.105 | 16.070 | 0.079 | -0.262 | 1.0400 | 2.58 |
| A1 | gamma P1 + S1 | +1.876 | +0.363 | +2.245 | +0.294 | 15.895 | 0.109 | +0.549 | 1.0402 | 4.20 |
| F | gamma P1, margin FROZEN | -2.592 | -0.329 | -2.176 | -0.352 | 15.724 | 0.045 | -9.381 | 1.0347 | 2.89 |

### Noise floor -- the same arm re-run on seeds +1000

| floor, 25 seeds | G1 cc mean | G1 all mean | margin SD | corr(h,a) | total bias |
|---|---:|---:|---:|---:|---:|
| I incumbent | -0.089 | -0.037 | -0.072 | +0.008 | -0.261 |
| A3 gamma P3 + S1 | -0.047 | -0.055 | +0.158 | +0.001 | -0.128 |
| **THE FLOOR (larger of the two)** | **0.089** | **0.055** | **0.158** | **0.008** | **0.261** |

(5-seed floors: incumbent +0.135 / +0.091 / +0.004 / +0.028 / -0.310;
A3 +0.100 / +0.146 / +0.016 / +0.014 / -0.264.)

---

## 3. The decision

**Criterion 1 (pre-registration 12.6): G1 mean inside +/- 1.0 and SD inside
+/- 0.75 on BOTH the clock-complete set and all 500 games.**

| id | G1 cc mean | pass | G1 all mean | pass | G1 SD both | pass | criterion 1 |
|---|---:|---|---:|---|---|---|---|
| B3 | +1.161 | NO | +1.666 | NO | -0.493 / -0.582 | yes | **FAIL** |
| B1 | +1.243 | NO | +1.741 | NO | -0.285 / -0.417 | yes | **FAIL** |
| A3 | +1.469 | NO | +1.921 | NO | +0.133 / +0.068 | yes | **FAIL** |
| A2 | +1.458 | NO | +1.872 | NO | +0.146 / +0.105 | yes | **FAIL** |
| A1 | +1.876 | NO | +2.245 | NO | +0.363 / +0.294 | yes | **FAIL** |
| I | +2.697 | NO | +3.411 | NO | -0.058 / -0.090 | yes | **FAIL** |

**0 of 6. The SD half of G1 passes everywhere; the MEAN half fails everywhere.**
Dispersion has not been the blocking half of G1 since round 1 and still is not.

**Criterion 2: margin SD and home/away correlation must not move beyond the
floor relative to the incumbent.** Reported as pre-registered:

| id | margin SD vs I | floor 0.158 | corr vs I | floor 0.008 |
|---|---:|---|---:|---|
| B3 | -0.377 | outside | -0.051 | outside |
| B1 | -0.449 | outside | -0.036 | outside |
| A3 | -0.181 | outside | +0.006 | INSIDE |
| A2 | -0.185 | outside | +0.010 | outside |
| A1 | -0.360 | outside | +0.040 | outside |

**Criterion 2 as written is close to unsatisfiable and that is a
mis-specification in the pre-registration, recorded rather than repaired.** The
clock model changes the possession COUNT, and the count is a variance driver for
the score; a clock change that moved margin SD by less than 0.158 on 12,500 rows
would have to be a clock change that did nothing. The floor here measures
seed-to-seed noise in margin SD, which is small, not the size of a legitimate
clock effect. It changes no verdict -- criterion 1 already fails for every arm --
so **no gate is softened and nothing is adopted**; the next pre-registration
should state criterion 2 against the ACTUAL, not against the incumbent, or
restrict it to the frozen-versus-live contrast it was borrowed from.

**DECISION: adopt nothing.** `ENGINE_CLOCK` stays `reference` and
`provisional_clock` stays True. The best arm the project has produced inside the
engine is **`empirical_km3_srfloor | P3 | S1`** at +1.161 / +1.666, and the
pre-registered ordering would have selected exactly that arm had it passed:
cell-based before gamma, P3 before P1 on engine-safety.

---

## 4. The question round 3b could not answer: P1 vs P2 vs P3

Round 3b's negative result was that the offline chain feeds the model the real
score sequence, so it contains no loop and cannot rank the parametrisations
(offline the three sat within 0.227 possessions of each other on both base
arms). The engine ranks them, and it ranks them by about twice as much:

| contrast | offline (round 3b, G1-CC) | ENGINE (25 seeds, G1-CC) |
|---|---:|---:|
| gamma P1 -> P3 | -0.227 | **-0.407** |
| gamma P1 -> P2 | -0.038 | **-0.418** |
| gamma P2 -> P3 | -0.189 | +0.011 |
| srfloor P1 -> P3 | (not fitted offline) | -0.082 |

Readings, against a 0.089-possession floor:

1. **Removing the simulation's own margin from the clock is worth about 0.42
   possessions per team-game**, and only the engine can see it. Offline, deleting
   `score_diff` outright (P2) moved the count by 0.038 -- inside any floor.
2. **P2 and P3 are a TIE** (+1.458 vs +1.469, 0.011 apart against a 0.089 floor).
   The engine-safe end-game window buys nothing over deleting the margin
   outright, which means the late-game behaviour P3 was built to keep is not
   reaching the possession count. By the pre-registered simplicity order
   (P2 < P3) P2 would win that tie if the family were adoptable.
3. **On the cell-based arm the P1 -> P3 gap is 0.082, inside the floor** -- also a
   tie. The cell arm's score dimension is a 3-level state rather than a
   continuous coefficient, so it carries less loop to remove. That is consistent
   with L26's "prefer cell-based arms for the engine".

---

## 5. The frozen arm, and a correction to how Decision-10 ablations should be read

| | G1 cc mean | total bias | last-poss duration s | mean poss duration s |
|---|---:|---:|---:|---:|
| A1 gamma P1, margin live | +1.876 | +0.549 | 11.361 | 17.121 |
| F gamma P1, margin FROZEN at 0 | **-2.592** | **-9.381** | 15.344 | 18.210 |
| A2 gamma P2, REFIT without margin | +1.458 | -0.262 | 11.260 | 17.164 |
| A3 gamma P3, margin live | +1.469 | -0.136 | 11.601 | 17.153 |
| F3 gamma P3, margin FROZEN (5 seeds) | +0.366 | -2.398 | 12.663 | 17.429 |

Freezing the margin swings the count by **4.468 possessions** and the game total
by **9.9 points**; refitting the same model class without the margin swings it by
**0.418**. The two disagree by an order of magnitude, and the freeze lands 2.6
possessions and 9.4 points BELOW the actual.

**Why, stated as a claim rather than an observation:** a freeze feeds a model
trained WITH the feature a value it almost never saw in the states where it
matters. `score_diff = 0` with forty seconds left is a tie game, and the fitted
tie-game regime is "play normally" -- so the frozen model stops producing the
short intentional-foul possessions that really do end close games, its
last-possession duration goes to 15.3 s against an actual 11.9, and the count
collapses. A refit without the feature has no such problem: its remaining
coefficients absorb the average effect of the margin.

**Consequence for Decision 10:** the freeze ablation is a valid DETECTOR of a
feedback loop and an invalid MEASUREMENT of one. It was a fair measurement for
`fg_make` (L23) only because there the refit-without-the-feature arm agreed with
it; the two must both be run before a magnitude is quoted. This does not weaken
L23 -- the fg_make round-2 bake-off refit five state parametrisations and the
closed-loop gate agreed with the freeze -- but it does mean the clock's "+3.8
possessions of loop" figure quoted from the L23 ablation is an overstatement of
what a refit recovers. The honest engine figure is **0.42**.

---

## 6. End of half

Accumulated inside the clock adapter: a possession whose intended duration
reaches the time left is the period's last by construction, so the adapter
records the seconds left at its start (which is also the duration it consumed).
Truth is `clock_v3.actual_end_of_half_cc` + `clock.eoh_stats` on the subset's
926 clock-complete regulation halves -- the same two functions the offline gate
uses.

| id | share last poss starts < 35 s | gap | mean last-poss duration s | gap |
|---|---:|---:|---:|---:|
| I incumbent | 0.9968 | +0.1210 | 5.878 | **-6.012** |
| B3 srfloor P3 | 0.9847 | +0.1089 | 11.175 | -0.715 |
| B1 srfloor P1 | 0.9853 | +0.1095 | 11.242 | -0.648 |
| A3 gamma P3 | 0.9448 | **+0.0690** | 11.601 | **-0.289** |
| A2 gamma P2 | 0.9456 | +0.0698 | 11.260 | -0.630 |
| A1 gamma P1 | 0.9423 | +0.0665 | 11.361 | -0.530 |
| F gamma P1 frozen | 0.9100 | +0.0342 | 15.344 | +3.453 |
| **ACTUAL (926 cc halves)** | **0.8758** | -- | **11.890** | -- |

The incumbent's end-of-half duration gap is **-6.01 s**; every round-3 arm cuts
it to -0.29 to -0.72 s. That is the single largest improvement in the table and
it is the censoring fix (L20) arriving in the engine. The SHARE is still too
high on every arm (0.94-0.99 against 0.876): the sim almost always has a
possession in progress at the horn, reality sometimes does not.

---

## 7. Responsiveness, and the one place a cell arm clearly wins

Possessions per game by the game's own PREGAME tempo-prior quintile (100 games
per bucket). Team-level quintiles are UNDERPOWERED on a 500-game subset -- about
2.8 appearances per team -- and are not reported.

| id | slope ratio | sim span | actual span | monotone steps | Decision-8 band [0.8, 1.2] |
|---|---:|---:|---:|---:|---|
| I incumbent | 1.432 | +9.02 | +6.30 | 4/4 | **outside** |
| B3 srfloor P3 | **1.061** | +6.69 | +6.30 | 4/4 | inside |
| B1 srfloor P1 | **1.079** | +6.80 | +6.30 | 4/4 | inside |
| A3 gamma P3 | 1.376 | +8.67 | +6.30 | 4/4 | **outside** |
| A2 gamma P2 | 1.400 | +8.82 | +6.30 | 4/4 | **outside** |
| A1 gamma P1 | 1.388 | +8.75 | +6.30 | 4/4 | **outside** |

Every arm is monotone in 4 of 4 steps, so nothing here is flat -- the standing
matchup-specific rule is not violated by any arm. But the gamma family and the
incumbent **over-respond**: they spread the fast and slow quintiles 39-43% wider
than reality does, outside Decision 8's [0.8, 1.2] band. The two cell-based arms
are the only ones inside it. That is a second axis, independent of G1, on which
the cell-based arm is the better object, and it is new evidence -- the offline
round-1 responsiveness read had all five arms between 0.79 and 1.06 because the
offline chain replays the real sequence of previous-end types.

---

## 8. Per month (all 500 games, 25 seeds)

| id | 2024-11 (109) | 2024-12 (83)* | 2025-01 (132) | 2025-02 (124) | 2025-03 (52)* |
|---|---:|---:|---:|---:|---:|
| I incumbent | +3.52 | +3.90 | +3.07 | +3.48 | +3.10 |
| B3 srfloor P3 | +2.25 | +1.87 | +1.28 | +1.64 | +1.16 |
| B1 srfloor P1 | +2.29 | +1.78 | +1.37 | +1.82 | +1.30 |
| A3 gamma P3 | +2.13 | +2.37 | +1.58 | +1.90 | +1.69 |
| A2 gamma P2 | +2.03 | +2.29 | +1.58 | +1.82 | +1.73 |
| A1 gamma P1 | +2.38 | +2.67 | +1.97 | +2.21 | +2.07 |
| F gamma P1 frozen | -2.12 | -1.88 | -2.40 | -2.14 | -2.27 |

`*` fewer than 100 games: UNDERPOWERED, excluded from the decision and reported
only so the row is not silently missing. On the three powered months the
overshoot is largest in November and smallest in January for every arm -- a
1.0-possession seasonal swing on B3 (+2.25 -> +1.28). No month passes for any
arm, so the all-powered-months clause is not what is binding; the overall mean
is.

---

## 9. What the residual actually is

The candidates' **total bias is inside +/- 1.0 and that is NOT a pass.** It is
two errors cancelling, exactly the pattern the multi-level rule exists to catch:

| id | possessions | PPP | total bias |
|---|---:|---:|---:|
| B3 | +1.67 (+2.4%) | 1.0385 vs 1.0705 (-3.0%) | -0.934 |
| A3 | +1.92 (+2.8%) | 1.0402 (-2.8%) | -0.136 |
| I | +3.41 (+5.0%) | 1.0391 (-2.9%) | +2.830 |

PPP is -0.03 on every arm including the incumbent, so the per-possession
shortfall is NOT the clock's and is not moved by any clock arm. It belongs to the
scoring cascade and is its own defect.

The possession half has an exact arithmetic account. Actual mean regulation
possession duration on the subset is **17.555 s** on clock-complete games
(17.410 s over all games, which is biased low by feed truncation; 21,722 and
63,529 possessions).

| id | engine mean poss duration s | gap vs 17.555 | implied G1 cc | observed G1 cc |
|---|---:|---:|---:|---:|
| B3 | 17.216 | -0.339 (-1.93%) | +1.35 | +1.161 |
| A3 | 17.153 | -0.402 (-2.29%) | +1.61 | +1.469 |
| I | 16.835 | -0.720 (-4.10%) | +2.93 | +2.697 |

**The residual overshoot IS a uniform mean-duration shortfall of about a third of
a second per possession.** It is not concentrated at the horn: the end-of-half
last possession is now within 0.3-0.7 s (section 6), and at roughly two
period-ending possessions per game that accounts for about 0.02 possessions of
the 1.16. It is not the margin loop (section 4: 0.42, and removing it entirely
leaves +1.46). It is not binning error (the cell arms are lookup tables by
construction, L26). It is not a responsiveness failure (section 7: B3's slope
ratio is 1.061).

So round 4's target is a stated, measurable one: **the conditional mean of the
duration law is about 2% short across ordinary possessions**, and the candidates
that could produce that are (a) a missing possession class the segmentation
merges into ordinary possessions (dead-ball and administrative time the engine
never spends), (b) the OT gap -- sim OT rate 2.6-4.2% against an actual 6.8%,
which shortens simulated games and is a separate open defect, and (c) the
`unknown`-terminal rows kept at L5 whose actual mean duration is 6.48 s against a
model-implied 19.31 s (`model.md` section 4.5), which biases the count the OTHER
way and so cannot be the cause.

---

## 10. Reproduction

```
.venv/Scripts/python.exe scripts/train_clock_v3c_s1.py
.venv/Scripts/python.exe scripts/run_clk3c_closed_loop.py --arm <ENGINE_CLOCK> \
    [--freeze] --seeds {5|25} [--seed-offset 1000] --tag clock3c_<arm>_s{5|25}[_floor]
.venv/Scripts/python.exe scripts/grade_clk3c_closed_loop.py --season 2025 \
    --pattern "clock3c_*_s25*" --out data/processed/models/clock/v3c_closed_loop_s25.json
.venv/Scripts/python.exe scripts/diag_clk3c_tables.py --s5 ... --s25 ...
```

Results: `results/engine_v0/clock3c_*` (19 runs, gitignored).
Scored output: `data/processed/models/clock/v3c_closed_loop_{s5,s25}.json`,
`v3c_eoh_truth.json`, `v3c_duration_truth.json`.
Tests: `tests/test_clock_adapter_v3.py` 16 passed, `tests/test_engine.py` 15
passed.
