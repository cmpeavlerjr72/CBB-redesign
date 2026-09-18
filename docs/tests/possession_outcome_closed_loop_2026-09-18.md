# Possession-outcome round 5 -- the SHIP GATE: `G2` and `G3` in the engine, paired against the served model

2026-09-18. Pre-registration `docs/models/possession_outcome/experiments.md`
section 12 (commit `d57d351`); wiring, builder, runner, grader and tests
`6215ac8`; results section 13.

**Verdict: `G2` is NOT put forward for serving. Both halves of the
pre-registered decision rule fail, and neither failure is marginal.**
`G3`, the comparator, fails the same first half. `ENGINE_EVENT` is unchanged,
this lane changed no default and edited no ledger, and nothing was hand-tuned,
capped, scaled, clipped or blended at any point.

| | condition 1: no G1-G9 line regresses beyond floor | condition 2: the responsiveness slope does not fall beyond floor | verdict |
|---|---|---|---|
| **`G2`** (round-4b offline winner) | **FAIL** -- G5 total SD ratio 22.2 floors, G5 total SD 18.6 floors, G5 margin SD 2.6 floors, G5 margin SD ratio 1.5 floors and PASS -> FAIL, G3 rim share 2.3 floors | **FAIL** -- headline slope 0.8881 -> 0.8545, a fall of **5.17 floors**; `off_tov_c` falls **15.95 floors** | **NOT PUT FORWARD** |
| **`G3`** (comparator) | **FAIL** -- G5 total SD ratio 22.1 floors, G5 total SD 22.0 floors, G5 margin SD 3.2 floors, G5 home/away corr 1.2 floors | headline slope RISES 0.8881 -> 0.9044 (no fall); `off_tov_c` falls 12.2 floors under the scored reading but IMPROVES under the `\|slope-1\|` reading | comparator; also blocked by condition 1 |

---

## 1. What was wired

Round 4b's two shrinkage arms (`experiments.md` 8.1), behind the EXISTING
`round2_s1` code path rather than a new adapter:

    G2   off_{r}_g2 = w*raw_c + (1-w)*prior_c
    G3   off_{r}_g3 = w*raw_c + (1-w)*w2*prior_c

`ENGINE_EVENT=round4b_G2` / `round4b_G3`, **default-off**. The eight shrunk
style columns are READ from
`data/processed/models/possession_outcome/round4/design_v4.parquet`, the object
round 4b's grade scored; `k_r` is not re-estimated anywhere in the engine. The
round-2 trainer and the round-4 trainer both persist no booster, so
`scripts/build_engine_event_round4b.py` refits each arm's own spec
(`feature_set_v4(arm, pop)`, `lgbm` on `first`, `cascade` on `cont`, seed 0)
on the served S1 monthly schedule -- the same procedure
`build_engine_event_round2.py` applies to the served winners, IMPORTED from
that script rather than copied.

Four preconditions abort the build rather than being reported after the fact,
and all four passed for both arms:

| precondition | why | result |
|---|---|---|
| all 16 `TEAM_COLS` of `design_v4.parquet` identical to `round2/design.parquet` on the 10,890 season-2025 team-games | the arms must differ from the reference ONLY by the shrinkage | **max abs diff 0.0** on all 16 |
| team-block coverage counts equal to the reference's | the arm and the reference must see the same games | **530 uncovered / 529 filled as-of backward / 1 at the league mean**, identical |
| S1 refit dates and per-game segment assignment equal to the reference's | the arm must be scored by the same schedule | 6 dates `2024-11-01 .. 2025-04-01`, `seg_of_game` array-equal |
| per-refit training row counts and `max_train_date` equal to the reference's | the arm must see the same training set, and no game may be scored by a model that saw its own tipoff | equal on all 6 refits, both populations; `_assert_no_leak` passes per GAME |

Engine-side, the shrinkage is real and sizeable: the mean absolute change to a
team-game's eight style columns is **1.756** under `G2` and **1.644** under
`G3` (p10 0.80 / 0.74, p90 2.98 / 2.76), on columns whose own SD is about 4-5.

**Lookup, not live modelling.** The sim loop is unchanged: one batched predict
per (population, monthly refit) over the whole active batch, against a
precomputed `(5710, 2, 16)` team block. Thread counts pinned to 1 in every
worker.

## 2. The precondition: is the served path bit-identical?

Run BEFORE any arm artifact was fitted, as section 12.2 required.

| run | when | ENGINE_EVENT | simulated rows vs `smoke60x5_default_v5b` |
|---|---|---|---|
| `smoke60x5_default_v5b` | 2026-09-11, the PM's clock serving check at `e3ccce5` | `round2_s1` | reference |
| `smoke60x5_po4b_wiring` | 2026-09-18, after this lane's `adapters.py` change | `round2_s1` | **300 game rows EQUAL, 4,882 player rows EQUAL**, column for column |

**Every simulated value is identical and all nine `ENGINE_*` switches match.**
`scripts/digest_engine_run.py --compare` nonetheless reports a sha mismatch,
and this round reports that rather than waiving it. The diff is **exactly one
field, `meta.adapter_flags`, on exactly three leaves of 248**:

| leaf | 2026-09-11 | 2026-09-18 |
|---|---|---|
| `provisional_clock` | True | False |
| `sources.clock.adopted` | False | True |
| `sources.clock.note` | "round-5 closed-loop candidate..." | "`v5b_glat_pmean` (arm B1): ADOPTED 2026-09-11 13:45 EDT (PM)..." |

All three are the CLOCK lane's adoption strings, from that lane's own
**uncommitted** edit to `clock_adapter_v3.py` and `adapters.py` present in this
shared checkout; none is in the possession-outcome path and none is a computed
value. The bit-identity the precondition exists to establish -- that this
lane's wiring does not move the served simulation -- **holds, and holds across
a week and two other lanes' commits**, which is a stronger check than a
same-session A/B. The strict sha equality the pre-registration asked for does
not, for a reason outside this lane. Both facts are on the record.

Tests: `tests/test_event_adapter_round4b.py` **10 passed** (including a
block-level assertion that the eight non-style team columns are bit-identical
to the served block and the eight style ones are not, and an end-to-end
assertion that the arm's predictions are a proper simplex and are NOT the
reference's). `tests/test_engine.py` **20 passed**.

## 3. The runs

`scripts/run_po4b_closed_loop.py`, 500 games of the standing subset (F2 2025
sorted by `game_id`, every 11th row, first 500 -- the same sample clock rounds
3c/4/5/5b/5d used), 25 paired seeds, players kept, 8 workers.

| run | `ENGINE_EVENT` | seeds | rows | wall clock | poss/s |
|---|---|---|---:|---:|---:|
| `po4b_R_s25` | `round2_s1` (served) | 0-24 | 12,500 | 412 s | 4,250 |
| `po4b_G2_s25` | `round4b_G2` | 0-24 | 12,500 | 563 s | 3,109 |
| `po4b_G3_s25` | `round4b_G3` | 0-24 | 12,500 | 635 s | 2,755 |
| `po4b_R_s25_floor` | `round2_s1`, **seeds 1000-1024** | 1000-1024 | 12,500 | 603 s | 2,898 |

Pinned on all four and written into all four `run_meta.json`, and asserted
against `adapters.py`'s own defaults at startup: `ENGINE_CLOCK=v5b_glat_pmean`,
`ENGINE_FG_MAKE=round4_B1`, `ENGINE_FG3=decision8`, `ENGINE_REBOUND=s1_weekly`,
`ENGINE_FREE_THROW=s1_conf_aligned`, `ENGINE_ROTATION=reference`,
`ENGINE_ROTATION_SCHEME=s1`, `ENGINE_USAGE=reference`, inputs v2. **This is the
SERVED stack.** The clock lane's `clk5b_B1_s25` was NOT reused as the
reference, and section 12.3 said so before the run: that run pins the
superseded interim `round3_shooter_S_C_s1` fg_make, and a ship gate must price
an arm on the stack that is actually served. Every floor in this table is
measured here, on this stack.

`run_meta.json` records, per run, that the working tree was dirty with the
clock and rotation lanes' in-progress edits (`engine_dirty_src`). All four runs
were produced from the same tree, so the comparison is internally exact.

## 4. Gates G1-G9, priced in floors measured in this round

`scripts/eval_gates.py` at `docs/gates.yaml`'s tolerances over all four
directories, blind; `scripts/diag_pair_gate_reports.py` for the A/B/N pairing.
**floor = |N - R|** on that line. **dir** is whether the arm moves TOWARD or
AWAY from that line's own target. A line regresses when it moves AWAY by more
than one floor, or flips PASS -> FAIL.

| gate | line | target | R | N | floor | G2 | fl | dir | G3 | fl | dir |
|---|---|---:|---:|---:|---:|---:|---:|:--|---:|---:|:--|
| G1 | possessions/game mean | 68.328 | 70.019 | 69.920 | 0.099 | 69.982 | 0.37 | toward | 69.969 | 0.51 | toward |
| G1 | possessions/game SD | 5.191 | 5.609 | 5.552 | 0.057 | 5.539 | 1.23 | **toward** | 5.547 | 1.09 | **toward** |
| G3 | three_pa_share pooled | 0.3906 | 0.389413 | 0.389410 | 0.0000032 | 0.388942 | 147 | away | 0.389063 | 109 | away |
| G3 | fta_per_fga pooled | 0.3295 | 0.3195 | 0.3177 | 0.0018 | 0.3187 | 0.44 | away | 0.3187 | 0.44 | away |
| G3 | rim_share pooled | 0.3733 | 0.372052 | 0.371407 | 0.000644 | 0.370572 | **2.30** | away | 0.371367 | 1.06 | away |
| G4 | tov_pct pooled | 0.1739 | 0.1762 | 0.1758 | 0.0004 | 0.1759 | 0.75 | toward | 0.1754 | 2.00 | **toward** |
| G4 | oreb_pct pooled | 0.2984 | 0.2836 | 0.2834 | 0.0002 | 0.2840 | 2.00 | **toward** | 0.2840 | 2.00 | **toward** |
| G4 | ft_rate pooled | 0.3295 | 0.3195 | 0.3177 | 0.0018 | 0.3187 | 0.44 | away | 0.3187 | 0.44 | away |
| G4 | efg_pct pooled | 0.5086 | 0.4990 | 0.4984 | 0.0006 | 0.4986 | 0.67 | away **PASS->FAIL** | 0.4988 | 0.33 | away |
| **G5** | **margin SD ratio** | 1.000 | **0.9695** | 0.9560 | **0.0135** | **0.9495** | **1.48** | **away PASS->FAIL** | 0.9560 | 1.00 | away |
| **G5** | **total SD ratio** | 1.000 | **0.8355** | 0.8346 | **0.0009** | **0.8155** | **22.2** | **away** | **0.8156** | **22.1** | **away** |
| G5 | home/away score corr | 0.2374 | 0.1063 | 0.0948 | 0.0115 | 0.0966 | 0.84 | away | **0.0925** | **1.20** | **away** |
| G5 | PIT K-S p | -- | 0.455 | 0.455 | 0 | 0.455 | 0 | same | 0.455 | 0 | same |
| **G5** | **margin SD (points)** | 12.60 | **12.2264** | 12.1758 | **0.0506** | **12.0942** | **2.61** | **away PASS->FAIL** | **12.0653** | **3.18** | **away** |
| **G5** | **total SD (points)** | 18.96 | **15.8421** | 15.8302 | **0.0119** | **15.6209** | **18.6** | **away** | **15.5804** | **22.0** | **away** |
| G6 | home margin non-neutral | 5.669 | 5.989 | 5.761 | 0.228 | 5.854 | 0.59 | toward | 5.876 | 0.50 | toward |
| G6 | home margin neutral | 2.151 | 0.506 | 0.343 | 0.163 | 0.774 | 1.64 | toward | 0.614 | 0.66 | toward |
| G7 | OT rate | 0.0680 | 0.0312 | 0.0274 | 0.0038 | 0.0305 | 0.18 | away | 0.0282 | 0.79 | away |
| G8 | rotation minutes mean | 29.82 | 30.47 | 30.47 | <0.005 | 30.46 | -- | toward | 30.46 | -- | toward |
| G8 | rotation minutes SD ratio | 1.000 | 1.2304 | 1.2301 | 0.0003 | 1.2296 | 2.67 | **toward** | 1.2287 | 5.67 | **toward** |
| G8 | players used / team-game | 9.83 | 8.83 | 8.81 | 0.02 | 8.83 | 0 | same | 8.83 | 0 | same |
| G9 | margin bias | 0 | +0.1117 | -0.1092 | 0.2209 | +0.0194 | 0.42 | toward | +0.0219 | 0.41 | toward |
| G9 | total bias | 0 | -0.9624 | -1.3158 | 0.3534 | -1.0857 | 0.35 | away **PASS->FAIL** | -0.9992 | 0.10 | away |
| G9 | calibration slope | 1.000 | 0.8920 | 0.8535 | 0.0385 | 0.8736 | 0.48 | away | 0.8786 | 0.35 | away |

Lines reported and NOT scored, because the reference itself is
NEEDS-INSTRUMENTATION or UNDERPOWERED on them and section 12.4 says such a line
is never scored: G1 by month (0 powered months on a 500-game subset), G2's nine
PPP terciles (2/9 powered cells inside, identical in all four runs), the four
per-team G3/G4 breakdowns (0 powered teams), G7's half-share (the results
contract carries no per-period score), G8 top-1 FGA share, G9's three
by-breakdown counts, and G6's neutral-site cell. Both G6 cells and both
G8 count lines are reported above for completeness with their status marked.

**Gate-level statuses are identical in all four runs** -- G1 FAIL, G2 FAIL,
G3 NEEDS-INSTRUMENTATION, G4 FAIL, G5 FAIL, G6 PASS, G7 FAIL, G8 FAIL, G9 FAIL.
No arm changes an overall gate verdict; the ship question is entirely about
movement inside them, which is why the floors matter.

### 4.1 What condition 1 actually turns on

The dispersion gate. **Both arms compress the engine's between-game spread, and
the engine's spread is already far too small.** The total SD ratio is 0.8355 in
the reference against a target of 1.0 -- the single largest open defect in the
engine after the clock lane's work -- and both arms push it to 0.816, a move of
22 measured floors in the wrong direction. The same movement shows in points:
total SD 15.84 -> 15.62 (`G2`) / 15.58 (`G3`) against a realised 18.96, and
margin SD 12.23 -> 12.09 / 12.07.

This is not a surprise and it is not a wiring artefact: shrinking every team's
style profile toward a prior is, mechanically, a reduction in how different
teams are from one another, and a simulator fed less distinct teams produces
less distinct games. **The arm does exactly what it was designed to do, and the
thing it does is the wrong direction for the engine's worst-calibrated gate.**

`G2` additionally flips three lines PASS -> FAIL (G5 margin SD ratio, G5 margin
SD, G4 pooled eFG%, G9 total bias -- four, of which the last two are inside
floor and are flips by rounding at the tolerance edge). Under the
pre-registered wording a PASS -> FAIL flip is a regression whether or not it
clears the floor; the two inside-floor flips are labelled as such and the
verdict does not rest on them.

Against that, both arms produce real improvements, none of which condition 1
weighs against a regression: possessions/team-game SD moves 1.1-1.2 floors
toward the target, pooled OREB% and TOV% move toward theirs, rotation minutes
SD ratio moves 2.7-5.7 floors toward 1.0, and both arms cut the reference's
margin bias from +0.112 to about +0.02.

## 5. The PM's line: responsiveness, read closed-loop

`scripts/grade_po4b_closed_loop.py`. The 998 team-games of the subset that join
an actual (one game of the 500 has no actual team box and is excluded from
every arm alike) are bucketed into five quintiles of the OFFENCE team's own
as-of driver value, taken from the **served** round-2 team block, so all four
runs are cut on identical rows (quintile n = 197-200 per cell, all powered).
`slope_ratio = (sim Q5 - sim Q1) / (actual Q5 - actual Q1)`, which is
`responsiveness_by`'s span ratio transported to the closed loop.

| driver | class | actual span | R | N | floor | G2 | signed fall | floors | G3 | signed fall | floors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `off_3pa_c` | FGA_3 | +9.485 pp | 1.0307 | 1.0455 | 0.0148 | 1.0475 | -0.0168 | -1.14 (rises) | 1.0717 | -0.0410 | -2.77 (rises) |
| `off_rim_c` | FGA_rim | +7.826 pp | **0.8881** | 0.8946 | **0.0065** | **0.8545** | **+0.0336** | **5.17 FAIL** | 0.9044 | -0.0163 | -2.51 (rises) |
| `off_tov_c` | TOV | +2.813 pp | **1.0835** | 1.0720 | **0.0115** | **0.9001** | **+0.1834** | **15.95 FAIL** | **0.9428** | **+0.1407** | **12.23 FAIL** |
| **headline** (worst non-exempt) | -- | -- | **0.8881** | 0.8946 | **0.0065** | **0.8545** | **+0.0336** | **5.17 FAIL** | 0.9044 | -0.0163 | -2.51 (rises) |

No driver is exempt: the narrowest actual span is 2.813 pp, above Decision 8's
2.0 pp bar. Step agreement is 4/4, 4/4, 3/4 in every arm including the
reference, unchanged.

**`G2` fails the line, and it fails it the same way under both readings.** The
pre-registration fixed the scored reading as the SIGNED FALL before the run and
anticipated that a slope above 1.0 could make the two readings disagree. For
`G2` they do not: on `off_rim_c` the slope falls 0.8881 -> 0.8545 and
`|slope - 1|` worsens 0.1119 -> 0.1455; on `off_tov_c` the slope falls
1.0835 -> 0.9001 and `|slope - 1|` worsens 0.0835 -> 0.0999. Two of three
drivers, and the headline, move the wrong way by 5 to 16 floors.

**For `G3` the two readings do disagree, on one driver, and this is reported
before the verdict rather than chosen after it.** `off_tov_c` 1.0835 -> 0.9428
is a signed fall of 12.23 floors (the scored reading: FAIL) and a move from
0.0835 to 0.0572 away from 1.0 (the other reading: a 2.3-floor IMPROVEMENT).
The reference over-slopes that driver; `G3` brings it closer to 1.0 by
undershooting less than `G2` does. `G3` is a comparator and is not put forward
under any outcome, so nothing turns on which reading is preferred -- but the
PM should note that the one-sided wording of the line is written for slopes
below 1.0 and does not fit a driver the engine over-slopes. On the two drivers
that sit below 1.0 the readings agree and `G3` improves both.

The quintile detail on the driver that decides it, `off_rim_c` (rim FGA per
possession, percentage points):

| | Q1 | Q2 | Q3 | Q4 | Q5 | span |
|---|---:|---:|---:|---:|---:|---:|
| ACTUAL | 28.011 | 30.025 | 31.519 | 32.347 | 35.837 | +7.826 |
| R (served) | 28.261 | 30.073 | 31.518 | 32.511 | 35.212 | +6.951 |
| N (floor) | 28.072 | 30.166 | 31.585 | 32.686 | 35.074 | +7.002 |
| **G2** | 28.192 | 29.847 | 31.529 | 32.583 | **34.880** | **+6.688** |
| G3 | 28.132 | 29.795 | 31.692 | 32.628 | 35.211 | +7.079 |

`G2` flattens the top of the distribution: the rim-heaviest quintile drops from
35.21 to 34.88 against a realised 35.84. **This is the offline finding
reproducing in the engine.** Round 4b recorded `G2` with the round's worst
offline responsiveness slope, 0.9156 against the reference's 0.9473, and
section 11.5 point 2 flagged it as the thing the standing matchup-specific rule
says to look at. The closed loop confirms it at 5.17 measured floors.

## 6. Segment cells

Every cell is the same games for every arm; actuals are computed on those games
and never on a season-wide figure. `min_cell_n` = 300 team-games.

| cell | n team-games | n games | status |
|---|---:|---:|---|
| weeks 0-3 | 220 | 110 | **UNDERPOWERED** |
| weeks 4-7 | 156 | 78 | **UNDERPOWERED** |
| weeks 8+ | 622 | 311 | scored |
| conference | 640 | 320 | scored |
| non-conference | 358 | 179 | scored |
| ALL | 998 | 499 | scored |

Points bias (sim minus actual, per team-game) and points MAE:

| cell | R | N (floor run) | G2 | G3 | | R MAE | G2 MAE | G3 MAE |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| weeks 0-3 *(underpowered)* | -0.854 | -1.388 | -0.846 | -0.850 | | 9.266 | 9.321 | 9.245 |
| weeks 4-7 *(underpowered)* | -0.320 | -0.244 | -0.442 | -0.462 | | 9.095 | 9.335 | 9.062 |
| weeks 8+ | -0.612 | -0.731 | -0.684 | -0.607 | | 8.109 | 8.176 | 8.195 |
| conference | -0.666 | -0.787 | -0.733 | -0.645 | | 8.187 | 8.287 | 8.286 |
| non-conference | -0.538 | -0.822 | -0.590 | -0.624 | | 9.111 | 9.186 | 9.055 |
| ALL | -0.620 | -0.800 | -0.682 | -0.638 | | 8.518 | 8.610 | 8.562 |

**Read this table against its own floor and it says nothing either way.** The
seed-offset run moves the ALL-cell points bias by 0.180 and the non-conference
cell by 0.284; every arm-vs-reference movement in the table is smaller than the
floor movement in the same cell. The two cells round 4 was actually about --
weeks 0-3 and weeks 4-7 -- are UNDERPOWERED at 220 and 156 team-games on a
500-game subset, which section 12.4 predicted in advance. **This round cannot
confirm or refute the early-season improvement `G2` was selected for.** It was
never able to: the subset is a stride sample of the whole season, not an
early-season sample. That is a limitation of the standing 500-game subset, not
a finding about `G2`, and a round that wanted to price the early-season cell
closed-loop would need a different, week-stratified sample.

Possessions per team-game bias falls slightly in every cell under both arms
(ALL: +1.687 -> +1.649 / +1.636 against an actual excess of 0, floor 0.099),
and PPP is unmoved (1.0381 -> 1.0378 / 1.0386 against an actual 1.0732).

## 7. Multi-level evidence

**Per game.** Paired on (game, seed) with common random numbers. The per-game
total moves by SD **2.29 points** under `G2` and 2.32 under `G3`, against
**4.80 points** for the seed-offset reference run: an arm perturbs an
individual game about half as much as a seed change does. The mean movement is
-0.123 / -0.037 points. **So the arms' effect is a small systematic
compression, not a re-ordering of which games are high or low** -- consistent
with the dispersion finding in section 4.1 and inconsistent with any claim that
the arms sharpen game-level discrimination.

**Per team.** 347 teams appear in the subset; **157 have fewer than 3 subset
games and are UNDERPOWERED**, excluded from the correlations and never read.
On the 190 powered teams:

| | R | N (floor) | G2 | G3 |
|---|---:|---:|---:|---:|
| mean \|per-team points bias\| | 5.988 | 6.026 | **6.254** | **6.223** |
| SD of per-team points bias | 7.743 | 7.823 | **7.958** | **7.954** |

Both arms make the per-team points bias **worse**: the mean absolute bias moves
0.266 (`G2`) and 0.235 (`G3`) against a floor of 0.037, i.e. 7.1 and 6.3
floors. This is the per-team face of the same compression -- a shrunk team
profile predicts a team's own scoring less well.

Against the quantity the arm actually manipulates: `corr(shrinkage applied to
the team, |change in that team's bias|)` is **+0.107** (`G2`) and **+0.164**
(`G3`), against a floor correlation of +0.013 / +0.026 from the seed-offset
run. The effect is in the expected direction -- teams shrunk more move more --
but it is weak, and the by-quintile means are not monotone in either arm (the
most-shrunk quintile moves +0.150 / +0.221 points, the least-shrunk +0.076 /
-0.074, with the middle quintiles scattered). **The arm reaches the teams it is
supposed to reach, and what it does when it gets there is not systematically
in either direction.**

**Per possession type.** Realised terminal mix per possession, against the
event layer's own actuals on the same 998 team-games:

| run | 3PA | rim FGA | jump-2 FGA | TOV | FTA | OREB |
|---|---:|---:|---:|---:|---:|---:|
| R (served) | 0.33028 | 0.31507 | 0.20235 | 0.17483 | 0.27028 | 0.14842 |
| N (floor) | 0.33075 | 0.31499 | 0.20313 | 0.17441 | 0.26917 | 0.14866 |
| G2 | 0.33021 | 0.31416 | 0.20419 | 0.17449 | 0.26995 | 0.14891 |
| G3 | 0.33047 | 0.31496 | 0.20355 | 0.17403 | 0.27004 | 0.14888 |
| **ACTUAL** | **0.33741** | **0.31549** | **0.19940** | **0.17525** | **0.28256** | **0.15230** |

The engine's standing mix errors -- too few threes, too many two-point jumpers,
too few free-throw trips, too few offensive rebounds -- are **unchanged by
either arm at the level of this table**, and every arm-vs-reference movement is
within about one floor-run movement except the jump-2 share, which `G2` pushes
0.0018 further from the actual (floor 0.0008, 2.3 floors away) -- the same
direction as the pooled rim-share and three-point-share regressions in section
4.

**Player level is not read.** No arm here touches the player layer, G8's two
scored lines both move TOWARD their targets, and the round claims nothing
further about players.

## 8. Verdict against section 12.5

**Condition 1 fails for `G2`** on the G5 dispersion lines at 1.5 to 22 measured
floors, with a PASS -> FAIL flip on the margin SD ratio, plus the pooled rim
share at 2.3 floors and the pooled three-point share at a very large floor
multiple on a very small absolute move. **Condition 2 fails for `G2`** at 5.17
floors on the headline and 15.95 on `off_tov_c`, under both readings of the
line. Either failure alone is disqualifying under the rule; both fail, and the
rule is explicit that a failing line is not waived because the offline table is
good.

**`G2` is NOT put forward for serving. The served `round2_s1` remains the arm
this lane puts forward.**

**`G3`, the comparator, also fails condition 1** on the same dispersion lines
(22.1 floors on the total SD ratio) plus the home/away correlation at 1.2
floors. It passes condition 2 on the headline and on both below-1.0 drivers,
and it is the better arm of the two on every closed-loop line that separates
them -- but section 12.5 does not put a comparator forward, and its own
condition-1 failure is as large as `G2`'s. **Nothing here re-opens the offline
selection of 8.5 and nothing here asks the PM to.**

**What the round establishes beyond the verdict.** The offline selection and
the closed loop disagree about `G2`, and the disagreement is explainable rather
than mysterious: `G2` buys a segment (weeks 0-3 calibration, offline) by making
every team's style profile more like the league's, and the engine's largest
standing defect is that its games are already not different enough from one
another. **An early-season shrinkage arm that does not also preserve
between-team dispersion cannot ship into this engine**, and that is a
constraint on the next round's arms, not a property of `G2` alone. The
narrower, constructive reading the evidence supports: the reliability problem
round 4 identified is real, but it needs a form that shrinks a team's
*estimate* without shrinking the *spread of estimates* -- a variance-preserving
or hierarchical-with-rescaling shrinkage rather than a plain posterior mean.
`G3`'s partial rescue of two responsiveness slopes is the first evidence that
the two-level form moves in that direction.

**What this round cannot say.** It cannot price either arm on the early-season
cells it was selected for: they are underpowered on the standing subset and the
round says so with its own numbers. It cannot resolve Decision 9c (`A1`/`A2`
are still NOT RUN). It gives no fold-1 confirmation. It says nothing about the
sealed 2025-26 season or about market lines. And no default, ledger row or
adoption state was changed by this lane.

---

### Artifacts

| what | where |
|---|---|
| pre-registration | `docs/models/possession_outcome/experiments.md` section 12, commit `d57d351` |
| wiring, builder, runner, grader, tests | commit `6215ac8` |
| engine arm artifacts (46.9 MB each, gitignored) | `data/processed/models/engine/event_round4b_{G2,G3}_F2_2025/` |
| runs | `results/engine_v0/po4b_{R,G2,G3}_s25`, `results/engine_v0/po4b_R_s25_floor` |
| gate reports | `docs/tests/gates_po4b_*_2026-09-18.md` |
| paired gate tables | `results/engine_v0/po4b_grade/pair_{G2,G3}_vs_R.md` |
| responsiveness, segments, per-team, mix | `results/engine_v0/po4b_grade/{po4b_lines.json,responsiveness_lines.csv,segments.csv,per_team_response.csv,possession_mix.csv}` |
| bit-identity smoke | `results/engine_v0/smoke60x5_po4b_wiring` vs `smoke60x5_default_v5b` |
| build logs | `logs/po4b_build_G2.log`, `logs/po4b_build_G3.log`, `logs/po4b_runs.log` |

Wall clock, real: artifact builds 2026-09-18 18:00:50-18:56:34 EDT (3,344 s
each, two concurrent processes at 4 threads); the four closed-loop runs
18:58-19:34 EDT (2,213 s of engine time at 8 workers, run sequentially so the
8-worker cap held throughout); grading and diagnostics under 10 min. No process
this lane did not start was ever signalled.
