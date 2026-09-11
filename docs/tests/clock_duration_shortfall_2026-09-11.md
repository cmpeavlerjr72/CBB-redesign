# Clock round 4 diagnosis: where the served arm's mean-duration shortfall lives (2026-09-11)

Descriptive only. **This document fits no model and scores no arm.** It is the
evidence the round-4 pre-registration (`docs/models/clock/experiments.md`
section 14) is written against, and it was produced before that section existed,
on the same precedent as `docs/tests/clock_censoring_audit_2026-09-10.md`.

Scripts, all re-runnable, no number below is typed by hand:

- `scripts/diag_clock_v4_shortfall.py` -- the tiling identity, the train/serve
  quantity reconciliation, and the served arm's expected CONSUMED duration
  against the actual on the REAL 2025 possessions, cell by cell.
- `scripts/run_clk4_closed_loop.py --cc-only` with `ENGINE_CLOCK_DIAG=1` -- the
  engine run whose clock adapter accumulates possessions and consumed seconds
  per round-2 state cell (the engine writes no possession-level file, so the
  state distribution it VISITS is accumulated live or not at all).
- `scripts/diag_clk4_composition.py` -- the law/composition decomposition.

Artifacts: `data/processed/models/clock/v4_diag/` (cell CSVs,
`v4_diag_report.json`, `v4_composition_report.json`),
`results/engine_v0/clock4_REF_cc_diag/` (`clock_cells.parquet`, `games.parquet`,
`run_meta.json`). Engine run: `ENGINE_CLOCK=v3c_srfloor_P3_s1`,
`ENGINE_EVENT=round2_s1`, `ENGINE_FG_MAKE=round3_shooter_S_C_s1`,
`ENGINE_FG3=decision8`, `ENGINE_ROTATION=reference`, all passed as explicit
environment values; `loop.py` at commit `a0810d8`; 159 clock-complete games of
the pre-registered 500-game stride, 10 seeds, 220,340 simulated regulation
possessions against 21,722 real ones.

## 0. The claim under test, and the verdict

L31 recorded round 3c's account: *"the overshoot IS a uniform mean-duration
shortfall of about a third of a second on ordinary possessions"*.

**The shortfall is real and it is not uniform.** On the same games, the same
arm and the same subset:

| term | seconds | share of the gap |
|---|---:|---:|
| total: engine mean consumed duration minus actual | **-0.2368** | 100% |
| state COMPOSITION (the engine visits different states) | -0.1557 | 66% |
| composition x law interaction | -0.0580 | 24% |
| the model's own conditional LAW | **-0.0713** | 30% |
| cells the engine reaches that these 159 real games never did | +0.0482 | -20% |

(The four terms close the identity exactly; residual 3e-15.)

**Two thirds to nine tenths of the shortfall is the engine's `prev_end` mix, not
the clock model.** The engine starts 2.50 pp fewer possessions after a made
field goal and 2.10 pp more after a defensive rebound, and those two states'
durations differ by 7.5 s. That is an UPSTREAM defect. Compensating for it
inside the clock is precisely what the bottom-up standing rule forbids.

The remaining -0.071 s (offline, on the real state sequence over all 1,991
clock-complete 2025 games, -0.156 s) IS the clock's, and it splits again:
about 40% is the `srfloor` arm's own deliberate approximation in the 20-60 s
window, and about 60% is a season-level drift that a cell law pooled over four
seasons under-tracks.

At the closed-loop level: 69.706 possessions per team-game against an actual
68.530, **+1.176**, reproducing round 3c's +1.161 under a different fg_make
pin.

## 1. The tiling identity, and the train/serve quantity

The prime suspect the brief names -- "any mismatch between what the model was
trained to predict and what the loop subtracts" -- is **ruled out**, four ways.

| check | result |
|---|---|
| `duration_s == start_clock - end_clock` (2025, every possession) | TRUE, 0 of 768,834 rows differ |
| possession `start_clock` == previous possession's `end_clock` within a period | 69 gaps in 757,268 interior boundaries (0.009%), none in a clock-complete period |
| summed durations per clock-complete regulation period | mean **1199.58 s** of 1200; 2,807 of 3,982 periods sum to exactly 1200 |
| clock-complete regulation possessions in the design | 270,530 of 270,541; the 11 dropped are the `duration_s > 90` cap and move the mean by **-0.0036 s** |

So the data's possessions TILE the period: `start_clock` of a possession is by
construction the previous one's `end_clock` (`_GameMachine.prev_end_clock`), and
`duration_s = max(0, start_clock - end_clock)`. The engine tiles the same way:
`used = min(dur, left)` and `seconds_remaining -= used`, with the horn ending
the period. Mean consumed duration and possession count are therefore two
readings of ONE number, and the diagnosis is a duration diagnosis.

Consequences, each of which disposes of one hypothesis in the brief:

- **Dead-ball inbound time.** The game clock is stopped during a dead ball, so
  no seconds elapse and no possession absorbs them -- in the data OR in the
  engine. There is no missing inbound component to add: if there were, the
  clock-complete periods would not sum to 1200.
- **Timeouts and media stoppages inside a possession.** Same argument, same
  evidence.
- **Continuation chances after an OREB.** Both sides measure ONE duration per
  POSSESSION spanning every chance: `_emit` writes `start_clock` of the
  possession and `end_clock` of its LAST chance, and `loop.py` draws the clock
  once per possession at (a) and runs the whole OREB cascade inside it without
  consuming more. There is no "data measures from the rebound, engine measures
  from the shot" split to fix. (The 13.4% of possessions that do continue after
  an OREB average 25.54 s against the model's marginal 17.51 -- the model is
  marginal over an outcome it is forbidden to see, which is correct, not a
  defect.)
- **Second rounding.** CBBD's clock is integer seconds, `duration_s` is an
  integer difference, and the pmf support is the integer grid 0..90. No
  half-second is lost anywhere.
- **Shot-clock era.** 30 s in every season 2022-2026. The segment is
  DEGENERATE over this fold window; it is reported as degenerate rather than
  silently dropped.

## 2. The model's conditional law, on the real state sequence

The served S1 artifacts, routed by month, scored on the real possessions with
the quantity the loop consumes,
`E[min(T,R)] = sum_{t<R} t p(t) + R P(T >= R)`:

| game set | possessions | actual mean | model mean | gap | implied poss/team-game |
|---|---:|---:|---:|---:|---:|
| the 159 clock-complete subset games | 21,722 | 17.555 | 17.504 | **-0.051** (-0.29%) | +0.20 |
| all 1,991 clock-complete 2025 games | 270,530 | 17.653 | 17.497 | **-0.156** (-0.88%) | +0.61 |

Both are far short of the -0.24 to -0.34 s the engine produces. **The model's
law on the real state sequence is not the main term.**

### 2.1 By previous end type (all clock-complete 2025 games)

| prev_end | n | actual | model | gap s | share of poss | share of gap |
|---|---:|---:|---:|---:|---:|---:|
| DREB | 94,904 | 14.507 | 14.472 | -0.034 | 35.1% | 8% |
| TOV | 45,869 | 14.882 | 14.901 | +0.019 | 17.0% | -2% |
| made_FG | 99,695 | 21.787 | 21.526 | **-0.261** | 36.9% | 62% |
| made_FT | 24,681 | 18.588 | 18.100 | **-0.488** | 9.1% | 29% |
| other | 1,400 | 1.757 | 1.526 | -0.231 | 0.5% | 1% |
| period_start | 3,981 | 20.848 | 20.467 | -0.381 | 1.5% | 4% |

The law bias is **entirely on possessions that begin after a dead ball** (made
FG, made FT, period start) and is zero on live-ball starts. That is a shape the
brief anticipated ("a dead-ball inbound component") -- but section 1 shows there
is no missing quantity, so it is a LEVEL problem inside those cells, not a
missing term.

### 2.2 By fine clock bucket -- the `srfloor` approximation prices itself

| r2 bucket | n | actual | model | gap s | share of poss | share of gap |
|---|---:|---:|---:|---:|---:|---:|
| 300+ | 200,913 | 18.069 | 17.987 | -0.083 | 74.3% | 39% |
| 150-299 | 32,945 | 18.154 | 17.888 | -0.266 | 12.2% | 21% |
| 90-149 | 13,407 | 17.804 | 17.584 | -0.221 | 5.0% | 7% |
| 60-89 | 6,859 | 17.181 | 17.146 | -0.035 | 2.5% | 1% |
| 45-59 | 3,787 | 15.304 | 14.165 | **-1.138** | 1.4% | 10% |
| 30-44 | 4,133 | 16.441 | 13.655 | **-2.787** | 1.5% | 27% |
| 20-29 | 2,579 | 12.660 | 11.881 | -0.779 | 1.0% | 5% |
| 10-19 | 2,777 | 7.884 | 8.740 | +0.855 | 1.0% | -6% |
| 5-9 | 1,724 | 4.575 | 5.362 | +0.787 | 0.6% | -3% |
| 0-4 | 1,406 | 1.611 | 1.931 | +0.320 | 0.5% | -1% |

The 20-59 s band is 2.9% of possessions and 37.5% of the law gap: that band is
exactly where `empirical_km3_srfloor` floors the clock dimension at the 45-59 s
cell by design (`SR_FLOOR_BUCKET = 5`). The floor was adopted as the strongest
form of L20 and it is not free; round 4 prices it against an unfloored arm
rather than arguing about it.

### 2.3 The level problem: season drift the pooled fit under-tracks

Mean regulation possession duration on clock-complete games:

| season | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| cc mean duration (s) | 17.547 | 17.584 | 17.510 | **17.653** |

and WITHIN 2025, by month (clock-complete):

| month | Nov | Dec | Jan | Feb | Mar |
|---|---:|---:|---:|---:|---:|
| actual | 17.374 | 17.409 | 17.793 | 17.836 | 17.800 |
| model | 17.229 | 17.434 | 17.549 | 17.636 | 17.632 |
| gap | -0.146 | +0.025 | **-0.244** | **-0.201** | -0.168 |

Both effects are present in every previous-end cell (2025, clock-complete:
made_FG 21.45 in November to 22.03 in February; DREB 14.27 to 14.73), so this is
a level, not a mix. The S1 monthly refit trains on all prior seasons plus the
season to date, which by January is 13% of the training mass (279,523 of
2,155,511 rows), so a cell mean is anchored on four seasons that are 0.10-0.14 s
faster than the one being simulated. This is the clock's version of L11
("centred features carry no season level") and of L21 ("S1 is a calibration
fix"): S1 moves the fit toward the current season but a POOLED cell count cannot
move it far enough.

**It is not cell sparsity.** On the January artifact, the share of rows served
at the FULL five-dimension grid is 100.00% for DREB, made_FG and period_start,
99.90% for TOV and 99.05% for made_FT; only `other` (0.5% of rows) falls back
materially. The hierarchical fallback is not diluting the dead-ball cells.

### 2.4 The other cuts, for the record (all clock-complete 2025 games)

| cut | worst cell | gap s |
|---|---|---:|
| half | H2 | -0.250 (H1 -0.060) |
| minute bucket | 0-1m | -0.832 (6.1% of possessions); every other bucket within 0.28 |
| tempo quintile | Q1 (slowest) | -0.572; Q5 +0.033, no monotone slope |
| site | neutral -0.229, away -0.202, home -0.085 | site is NOT a dimension of the R2 cell grid |
| offence's foul situation | no bonus -0.314; bonus +0.015; double bonus +0.615 | |
| score state | leading -0.458, trailing -0.745, tied +0.196 | |

None is underpowered (every cell above has n >= 1,400). The home/away spread is
worth flagging under the standing "home/away is a first-class feature" rule: the
round-2 cell grid dropped `site_code`, and the home-away law gap here is 0.12 s.
It is not this round's target and is logged open.

## 3. The state composition the ENGINE visits

Accumulated live by the clock adapter over 220,340 simulated regulation
possessions, against the same 159 games' 21,722 real ones, on the same round-2
cell grid (`prev_end` x fine bucket x period x bonus x tempo tercile).

| prev_end | sim share | actual share | share gap (pp) | sim mean dur | actual mean dur | composition contribution (s) |
|---|---:|---:|---:|---:|---:|---:|
| DREB | 37.09% | 34.99% | **+2.10** | 14.432 | 14.306 | -0.068 |
| TOV | 17.23% | 16.95% | +0.28 | 14.866 | 14.803 | -0.008 |
| made_FG | 34.28% | 36.78% | **-2.50** | 21.427 | 21.892 | **-0.109** |
| made_FT | 9.57% | 9.27% | +0.30 | 18.371 | 18.152 | +0.002 |
| other | 0.38% | 0.53% | -0.15 | 2.000 | 1.138 | +0.025 |
| period_start | 1.44% | 1.46% | -0.02 | 20.304 | 20.374 | -0.001 |
| **sum** | | | | | | **-0.158** |

That sum is the whole composition term (-0.156 on the joint grid). **The state
composition defect is the `prev_end` mix and nothing else.** The other four cell
dimensions contribute: fine bucket -0.002 net, period -0.001, tempo tercile
+0.000, and the bonus dimension actually helps (+0.070; the engine is in the
bonus on 27.65% of possessions against 30.60% actual, and bonus possessions are
short).

Where the missing made-FG starts go is an upstream question this document does
not close, and the honest statement is the measurement: per game the engine
produces 47.8 made-FG-started possessions against 50.4 actual (-5.2%) while
making slightly MORE field goals per game (51.4 vs 50.8) over 2.3 more
possessions; PPP is 1.038 against 1.055 (-1.6%) and FGA per possession 0.849
against 0.842. A made field goal that is followed by a free-throw trip (and-one)
or that ends a period does not start a `made_FG` possession, and the split
between those channels is the event/fg_make cascade's to explain, not the
clock's.

## 4. Verdict

1. **The train/serve quantity is correct.** The model is trained on exactly the
   quantity `loop.py` subtracts. No inbound component, no stoppage time, no
   OREB-split and no rounding term is missing (section 1).
2. **The shortfall is not uniform.** 66% composition + 24% interaction vs 30%
   law, with +20% given back by cells the engine reaches and the real games did
   not (section 0).
3. **The dominant term is upstream.** The engine's `prev_end` mix is 2.50 pp
   short of made-FG starts and 2.10 pp long on DREB starts, worth -0.158 s of
   mean duration, which is +0.62 possessions per team-game on its own. Fixing
   this inside the clock would be a downstream stage compensating for a known
   upstream bias.
4. **The clock's own share is -0.071 s in the engine (-0.156 s offline at full
   power) and has two separable causes**: the `srfloor` floor in the 20-59 s
   band (37.5% of the offline gap, 2.9% of possessions), and a season-level
   drift the pooled cell fit under-tracks (the 150+ s buckets, 91% of
   possessions, -0.083 to -0.266 s; 2025 runs 0.10-0.14 s slower than the
   seasons the fit is anchored on and 0.42 s slower in February than in
   November).
5. **Round 4 attacks cause 4 only**, with the served arm as the reference and
   the composition term measured and reported on every arm so that an arm cannot
   pass by accidentally moving the mix. If round 4's best arm removes the whole
   law term, the closed-loop count still lands near +0.6 to +0.9 possessions,
   and the pre-registration says so in advance rather than treating a miss as a
   surprise.
