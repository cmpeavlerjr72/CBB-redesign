# Clock round 5b -- a mean-consistent per-game pace latent (2026-09-11)

Pre-registration: `docs/models/clock/experiments.md` section 21, committed
`58d5050` BEFORE the module change, the script and every fitted parameter.
Code: `ce2cd94`. Results and verdict: `experiments.md` section 22.
Round 5's evidence, which this document builds on and does not restate:
`docs/tests/clock_duration_dispersion_2026-09-11.md`.

**One-line answer: the mean cost is gone.** Round 5's arm A1 raised the
possession-count mean because it was specified mean-1 on the DURATION scale
while every gate reads the possession COUNT, which is the reciprocal. Moving the
latent's location to `m = +sigma^2/2` makes `E[1/A] = 1` exactly, costs one
character of algebra and nothing in dispersion, and removes the cost.

---

## 1. The defect, in one identity

    A = exp(sigma*z + m)          one draw per (seed, game), both offences
    E[A]   = exp(m + sigma^2/2)   what a DURATION-scale mean constraint fixes
    E[1/A] = exp(-m + sigma^2/2)  what the POSSESSION COUNT actually reads

`P = 1200/Dbar` on a clock-complete regulation game, and `Dbar` scales with `A`,
so `E[P]` scales with `E[1/A]`, not with `E[A]`.

| arm | `m` | `E[A]` | `E[1/A]` | implied `E[P]` inflation at `Pbar = 68` |
|---|---|---:|---:|---:|
| A1 `v5_glat_shared` (round 5) | `-sigma^2/2` | 1.000000 | **1.002235** | **+0.152 possessions** |
| B1 `v5b_glat_pmean` | `+sigma^2/2` | 1.002214 | **1.000000** | **0.000** |
| B2 `v5b_glat_joint` | fitted `log c - sigma^2/2` | 1.009734 | 0.992492 | -0.511 |

The +0.152 is the arithmetic section 20.3 wrote down and attributed to Jensen's
inequality on a convex map. It is that -- and it is also a free choice of
location, which is what round 5b tests. These are measured columns of
`data/processed/models/clock/v5b_bakeoff/v5b_bakeoff_F2.csv`, not assertions.

---

## 2. Offline, fold 2 (2025: 270,530 clock-complete regulation possessions,
1,991 games; needed per-team-game possession SD 4.2795)

One code path (`scripts/exp_clk5b_mean_consistent.py`) fits and scores every
arm; `clock_v3.score_arm_v3` is round 3's blind scorer, unedited.

| id | arm | sigma | P SD produced | **ratio** | CRPS_trunc | cens. loglik | PIT worst D | leak cells | `E[min(T,R)]` | mean gap | implied poss delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R | `v3c_srfloor_P3_s1` (served) | -- | 2.9430 | **0.6877** | 4.927457 | -3.51919 | 0.43062 | 20 | 17.4965 | -0.1561 | +0.6066 |
| A1 | `v5_glat_shared` (round 5) | 0.047248 | 4.2980 | **1.0043** | 4.927770 | -3.50994 | 0.43066 | 20 | 17.5041 | -0.1485 | +0.5768 |
| **B1** | **`v5b_glat_pmean`** | 0.047028 | **4.3194** | **1.0093** | **4.927595** | **-3.50698** | 0.43066 | 20 | **17.5275** | **-0.1251** | **+0.4853** |
| B2 | `v5b_glat_joint` | 0.046373 | 4.3189 | 1.0092 | 4.927663 | -3.51000 | 0.43060 | 20 | **17.6009** | **-0.0518** | **+0.1999** |
| B3 | `v5b_glat_pmean_tempo` | fn(tempo) | 4.3157 | 1.0085 | 4.927629 | -3.50716 | 0.42953 | 20 | 17.5269 | -0.1257 | +0.4876 |

**The R and A1 rows reproduce section 17.2 to four decimals** (2.9430 / 0.6877 /
17.4965 / -0.1561 / +0.607 and 4.2980 / 1.0043 / 17.5041 / -0.1485 / +0.577).
That is the byte-for-byte reuse check: the round-5b code path is round 5's with a
location argument whose default is round 5's own.

Fitted parameters, F2 train {2022, 2023, 2024} with F1 {2022, 2023} beside it:
`B1 sigma 0.047028 / 0.046742`, `m +0.001106 / +0.001092`;
`B2 sigma 0.046373 / 0.045924`, `c 1.009734 / 1.011575`;
`B3 b0 -0.0023313 / -0.0027448`, `b1 6.6792e-05 / 7.2634e-05`.
As in round 5, **F1 and F2 agree to the third decimal on every parameter**: the
dispersion and its location are properties of the sport, not of a season. The
exception is B2's `c`, which moves in the third decimal (1.0097 vs 1.0116) --
see section 6.

### Noise floor, re-measured this round

Game-block bootstrap refits of the F2 training window, seeds 20260911 /
20260912, graded through the same blind path: `B1 sigma 0.047271 / 0.046889`,
P SD produced **4.3301 / 4.3182** against B1's 4.3194. **Measured floor on the
primary = 0.0107 possessions.** Round 5's measured floor on the same metric was
0.0242 and the pre-registration carries it; the LARGER (0.0242) is used
everywhere below, which is the conservative choice.

### The three pre-registered predictions, checked

1. **`|P SD(B1) - P SD(A1)| <= 0.0242`** (location and dispersion separable):
   4.3194 - 4.2980 = **+0.0214, inside the floor**. CONFIRMED. The location
   change buys the mean without costing the primary.
2. **B1 improves round 4's mean gate rather than merely holding it**: predicted
   `+0.039 s`, measured **+0.0310 s** (17.4965 -> 17.5275, gap -0.1561 ->
   -0.1251). CONFIRMED, slightly under the prediction because `round(A*T)` and
   the horn truncation `min(T, R)` absorb part of it.
3. **B3 fails the responsiveness band at Q2**: see section 4. CONFIRMED.

No no-regression line moved beyond its floor: CRPS_trunc R -> B1 is **+0.000138
against a 0.00684 floor** (2% of it), the censored log-likelihood is BETTER for
every latent arm and best for B1 (-3.50698 against R's -3.51919), PIT worst-cell
D moves +0.00005 and the leaking-cell count is 20 for every arm including R.

---

## 3. Multi-level evidence I -- per `prev_end` cell (all six powered)

Mean-duration gap `E[min(T,R)] - actual`, seconds, F2 2025:

| `prev_end` | n | R | A1 | **B1** | B2 | B3 |
|---|---:|---:|---:|---:|---:|---:|
| DREB | 94,904 | -0.0343 | -0.0277 | **-0.0069** | +0.0619 | -0.0076 |
| TOV | 45,869 | +0.0193 | +0.0273 | **+0.0477** | +0.1127 | +0.0468 |
| made_FG | 99,695 | -0.2607 | -0.2520 | **-0.2247** | -0.1425 | -0.2250 |
| made_FT | 24,681 | -0.4881 | -0.4812 | **-0.4573** | -0.3837 | -0.4579 |
| period_start | 3,981 | -0.3805 | -0.3719 | **-0.3456** | -0.2635 | -0.3459 |
| other | 1,400 | -0.2308 | -0.2296 | **-0.2283** | -0.2230 | -0.2284 |

Every cell has at least 1,400 rows, so none is underpowered. The reading:

- **B1 moves every cell toward zero and none away from it.** The improvement is
  proportional to the cell's own mean duration (the location is multiplicative),
  which is why `made_FT` and `period_start` -- the long cells -- move most.
- **The cell RANKING is untouched.** L34's two structural cells, `made_FG`
  (-0.26) and `made_FT` (-0.49), remain the largest gaps under every arm. Round
  5b does not touch them and does not claim to: L34 assigns them to the upstream
  `prev_end` MIX and to the law's own level, and a location on a per-game latent
  cannot and does not reorder them.
- **B2 overshoots two cells.** Its fitted `c = 1.0097` sends DREB from -0.007 to
  **+0.062** and TOV from +0.048 to **+0.113**: a single global level parameter
  pays for the long cells' shortfall by overshooting the short ones. That is the
  visible cost of correcting a per-cell defect with a scalar, and it is the main
  reason B1, not B2, is the arm this lane puts forward.

---

## 4. Multi-level evidence II -- per pregame-tempo quintile (the responsiveness
check, and where every arm still fails)

`produced / needed` per-game possession SD, five powered quintiles of 398-399
games each. The needed column is a property of the games, identical for all arms:
Q1 3.8050, Q2 **3.6175**, Q3 3.9886, Q4 4.4199, Q5 5.0344.

| arm | Q1 slow | **Q2** | Q3 | Q4 | Q5 fast | worst |
|---|---:|---:|---:|---:|---:|---:|
| R | 0.711 | 0.793 | 0.732 | 0.684 | 0.631 | 0.369 |
| A1 | 1.035 | **1.168** | 1.067 | 0.997 | 0.919 | 0.168 |
| **B1** | 1.040 | **1.174** | 1.072 | 1.002 | 0.924 | 0.174 |
| B2 | 1.041 | **1.175** | 1.072 | 1.001 | 0.922 | 0.175 |
| B3 | **1.004** | **1.158** | 1.075 | 1.017 | **0.954** | 0.158 |

**Every arm that lands the primary sits outside the pre-registered
[0.85, 1.15] band at Q2, exactly as in round 5.** The band was carried into
section 21.7 verbatim and unsoftened, and it is not waived here.

The mechanism is visible in the needed column: **Q2's needed SD, 3.6175, is a
DIP below Q1's 3.8050**, while the produced side is monotone in tempo by
construction (a constant-CV latent produces an SD proportional to the game's own
pace). A non-monotone target cannot be matched by a monotone family. That is why
B3 -- `sigma^2` linear in pregame tempo, the only arm with a fitted dispersion
function -- closes both ENDS (Q1 1.035 -> 1.004, Q5 0.919 -> 0.954) and moves Q2
by only 0.010: **a monotone linear function of tempo cannot reproduce a
non-monotone dip, and section 21.1 item 4 predicted exactly this failure in
writing before the arm was fitted.**

This is a DISPERSION-FUNCTION question -- what state variable Q2's dip is a
function of -- and it is the target a round 6 would have. It is not a location
question, and round 5b neither addresses it nor pretends to.

---

## 5. Closed loop, paired, 500 games

500 games of the section 14.4 subset (159 clock-complete), **25 seeds**, paired
by construction through the `(seed, game_id, family)` streams, `ENGINE_EVENT=`
`round2_s1`, `ENGINE_FG_MAKE=round3_shooter_S_C_s1`, `ENGINE_FG3=decision8`,
`ENGINE_ROTATION=reference` on every run. The reference arm R is
`results/engine_v0/clock4_R_s25` and its seed-offset twin `clock4_R_s25_floor`
(offset 1000) -- the round-4 reference runs under the same pinning on the same
subset, REUSED rather than re-run, which is what made the 25-seed gate fit the
session. New arms in new directories: `clk5b_A1_s25`, `clk5b_B1_s25`,
`clk5b_B2_s25` (and 5-seed screens `clk5b_B1_s5`, `clk5b_B2_s5`). Nothing
existing was overwritten.

### 5.1 The floors, MEASURED at 25 seeds

Section 18.2 item 4 forbade reading `corr(P, eFG%)` as a finding until a
seed-offset floor existed for it. Grading the two R runs through
`grade_clk5_dispersion_loop.py` and `grade_clk3c_closed_loop.py` supplies it, and
supplies one for the within-game possession SD as well:

| line | R offset 0 | R offset 1000 | **measured floor** |
|---|---:|---:|---:|
| G1 possessions/team-game mean, cc | +1.156 | +0.999 | **0.157** |
| G1 possessions/team-game mean, all 500 | +1.704 | +1.630 | **0.074** |
| within-game possession SD, cc | 3.6992 | 3.6135 | **0.0857** |
| possession SD ratio, cc | 0.8041 | 0.7999 | **0.0042** |
| G5 total SD ratio | 0.7357 | 0.7522 | **0.0165** |
| G5 margin SD ratio | 0.9704 | 0.9819 | **0.0115** |
| **`corr(P, eFG%)`** | **-0.2105** | **-0.1874** | **0.0231** |
| `corr(home, away)` | +0.0032 | -0.0005 | **0.0037** |
| margin SD (points) | 15.941 | 16.022 | **0.081** |
| total bias | -0.931 | -1.308 | **0.377** |

Round 4's carried 25-seed floors were G1 cc 0.180, G1 all 0.074, margin SD
0.101, corr 0.013; the independent re-measurement agrees on G1 all to three
decimals, which is the cross-check on the floor itself.

### 5.2 The deciding table

| line | R | A1 | **B1** | B2 | floor | actual |
|---|---:|---:|---:|---:|---:|---:|
| **G1 poss mean, all 500** | +1.704 | **+1.813** | **+1.649** | **+1.135** | 0.074 | 0.000 |
| **G1 poss mean, cc** | +1.156 | +1.136 | **+0.989** | **+0.463** | 0.157 | 0.000 |
| within-game poss SD, cc | 3.6992 | 4.8202 | 4.8147 | 4.6821 | 0.0857 | 4.6652 |
| **possession SD ratio, cc** | 0.8041 | 1.0350 | **1.0320** | **1.0109** | 0.0042 | 1.000 |
| G5 total SD ratio | 0.7357 | 0.8256 | 0.8227 | 0.8158 | 0.0165 | 1.000 |
| `corr(home, away)` | +0.0032 | 0.1060 | 0.1005 | 0.0914 | 0.0037 | 0.2374 |
| `corr(P, eFG%)` | -0.2105 | -0.1242 | -0.1223 | -0.1248 | 0.0231 | ~0.000 |
| G5 margin SD ratio | 0.9704 | 0.9567 | **0.9563** | 0.9651 | 0.0115 | 1.000 |
| margin SD (points) | 15.941 | 15.867 | 15.881 | 15.847 | 0.081 | 15.472 |
| **total bias** | -0.931 | -0.696 | -1.061 | **-2.178** | 0.377 | 0.000 |
| PPP | 1.0380 | 1.0381 | 1.0379 | 1.0376 | -- | 1.0705 |
| responsiveness slope ratio | 1.047 | -- | 1.029 | -- | -- | 1.000 |
| G1 poss SD delta, cc | -0.477 | +0.445 | +0.423 | +0.329 | -- | 5.120 |

### 5.3 Is the mean cost gone? Yes.

| comparison | all 500 | clock-complete |
|---|---:|---:|
| **A1 - R** (round 5's arm) | **+0.109 = 1.5 floors WORSE** | -0.020, inside the floor |
| **B1 - R** (the count-scale location) | **-0.055, INSIDE the floor** | **-0.167, 1.1 floors BETTER** |
| **B1 - A1** | **-0.164 = 2.2 floors better** | -0.147, inside the floor |

Round 5 measured A1's cost at +0.230 on the all-500 set at 5 seeds and handed it
over as the reason not to adopt. At 25 paired seeds it is +0.109 -- the same
sign, about half the size, and still 1.5 measured floors. **B1 removes it
entirely and then some**, and it does so because the latent was specified to
preserve `E[P]` rather than `E[D]`, fitted walk-forward, with no correction
applied after the fact.

**And it is bought for nothing.** Every dispersion line is A1's to within about
one floor: possession SD ratio 1.0320 vs 1.0350 (0.7 floors), total SD ratio
0.8227 vs 0.8256 (0.2 floors), `corr(home, away)` 0.1005 vs 0.1060,
`corr(P, eFG%)` -0.1223 vs -0.1242 (0.1 floors). The offline prediction
`|P SD(B1) - P SD(A1)| <= 0.0242`, written before the arm was fitted, holds in
the engine as well as in the algebra.

### 5.4 What else moved, per line, against a measured floor

- **possession SD ratio 0.8041 -> 1.0320**, 54 floors, target 1.000. The primary
  defect of round 5 is closed and now slightly OVERSHOT (1.032, 7.6 floors past
  1.000) where the offline algebra predicted 1.009; the difference is the
  engine's own across-seed state composition, which section 17.1 said in advance
  would push the engine's produced side above the offline one. Reported, not
  adjusted.
- **G5 total SD ratio 0.7357 -> 0.8227**, +0.0870, 5.3 floors.
- **`corr(home, away)` +0.0032 -> +0.1005**, 26 floors, against an actual
  +0.2374: the line is 42% closed, not closed.
- **`corr(P, eFG%)` -0.2105 -> -0.1223**, 3.8 floors, 42% of the way to zero.
  The engine lane published the counterfactual "a pace latent alone moves the
  line -0.1976 -> **-0.124**" before any of this ran
  (`docs/tests/pace_efficiency_sign_2026-09-11.md`, `experiments.md` section 18).
  **Measured -0.1223.** Neither lane fitted against this line.
- **G5 margin SD ratio 0.9704 -> 0.9563**, -0.0141 against a 0.0115 floor: a
  **1.2-floor REGRESSION**, small, real, and not waived. The raw margin SD moves
  the other way (15.941 -> 15.881 against an actual 15.472, inside its own
  floor), so the two margin readings disagree in sign; the G5 ratio is the gate
  and it is the one recorded as failing.
- margin SD, total bias and PPP all inside their floors for B1.

### 5.5 A 5-seed within-game SD is biased low on BOTH sides of its own ratio

| line | 5 seeds | 25 seeds |
|---|---:|---:|
| R within-game possession SD, cc | 3.3796 (section 20.1) | **3.6992** |
| B1 within-game possession SD, cc | 4.5779 | **4.8147** |
| B1 possession SD ratio, cc | 0.8936 | **1.0320** |
| B1 G5 total SD ratio | 0.7337 | **0.8227** |

Round 5's headline for A1, `0.662 -> 0.907`, understates the move; the honest
25-seed read for B1 is `0.804 -> 1.032`. **No 5-seed number in section 20 should
be quoted as a level**, only as a direction. This is why the pre-registration put
the gate at 25 seeds and why the floors above were measured there.

### 5.6 Per game

`G1 poss SD delta, cc` is the per-game distribution check the standing rule
asks for: R produces a per-game possession spread **0.477 too NARROW** on the
clock-complete set; B1 produces one **0.423 too WIDE**; B2 **0.329 too wide**.
The served arm and the candidate miss the per-game spread by similar magnitudes
in opposite directions, and the candidate's miss is the smaller of the two. The
responsiveness slope across pregame-tempo quintiles is 1.047 for R and 1.029 for
B1 (monotone 4/4 on both), so the arm does not flatten the matchup response
while widening the spread.

---

## 6. What is a fitted parameter and what would have been a hand tune

The standing rule bans post-hoc multipliers, caps, clips, offsets, calibration
curves and blends on sim output (`CLAUDE.md`; `docs/SIM_GUARDRAILS.md` core
principle and section 5). Round 5b sits close enough to that line that the
distinction is written out rather than assumed.

**B1 is not a tune by any reading.** `m = +sigma^2/2` is the unique location
making `E[1/A] = 1`; it is an algebraic identity of the log-normal family with
ZERO free parameters beyond the `sigma` round 5 already fitted. Nothing is
estimated against an engine output, a gate value or a test row. The equivalent
statement of the same arm -- `P_game = P_mean * exp(eps)` with `E[exp(eps)] = 1`
and durations DERIVED -- makes it plain that the constraint is a property of the
model's parametrisation, not a correction to its output.

**B2 is a fitted parameter, and it is the one to watch.** Its `c = E[A]` is a
method-of-moments estimate on TRAINING seasons of the marginal mean duration --
the textbook joint fit of a scale-mixture model, the same class of estimator
round 4 used for the conditional mean, and computed without touching the test
fold or the engine. But three things are true of it and are reported rather than
buried:

1. its bootstrap spread is real -- `c` is 1.00816 / 1.01085 across the two
   game-block refits against a point estimate of 1.00973, i.e. **+/-0.0027, or
   about +/-0.18 possessions per team-game** on a line whose 25-seed floor is
   0.074;
2. it drifts between folds (F1 1.01158, F2 1.00973) where every other round-5
   and round-5b parameter agrees to the third decimal; and
3. section 3 shows it buys the long `prev_end` cells by overshooting the short
   ones.

A scalar that lifts every duration by 1% to make a mean land is exactly the
shape of thing the standing rule exists to make people justify. It is defensible
HERE because it is fitted on training rows against the actual durations of the
actual state mix -- not against engine output, where it would silently absorb
L34's upstream `prev_end` composition bias and hide a defect that belongs to the
event and fg_make lanes. **This lane puts B1 forward and does not put B2
forward**, and records B2's numbers so the PM can see what the joint refit is
worth and what it costs.

**What would have been banned, and was not done:** rescaling simulated
possessions after the run, capping `Dbar`, clipping the latent, or fitting `m`
or `c` against the closed loop's G1 line. The mean in every row above comes out
of the fitted law.

---

## 7. Files

- pre-registration: `docs/models/clock/experiments.md` section 21 (`58d5050`)
- results and verdict: `docs/models/clock/experiments.md` section 22
- fitter + blind grader: `scripts/exp_clk5b_mean_consistent.py`
- module: `src/cbb_sim/models/clock_v5.py` (`latent_values(..., m)`, `loc_for`)
- engine wiring: `src/cbb_sim/engine/clock_adapter_v3.py` (`V5_MODES`,
  `LatentClockAdapter.loc_kind`), `adapters.py` dispatch prefix only --
  **the default is untouched and stays `v3c_srfloor_P3_s1`**
- offline tables: `data/processed/models/clock/v5b_bakeoff/`
  (`v5b_bakeoff_F2.csv`, `v5b_responsiveness_F2.csv`, `v5b_prev_end_F2.csv`,
  `v5b_floor.csv`, `v5b_bakeoff_report.json`) -- gitignored, HF-synced
- closed loop: `results/engine_v0/clk5b_*` (new directories; nothing existing
  was overwritten), R at 25 seeds reused from `clock4_R_s25` and its
  seed-offset twin `clock4_R_s25_floor`
- closed-loop grades: `data/processed/models/clock/v5b_floor_R_dispersion.json`,
  `v5b_floor_R_g1.json`, `v5b_closed_loop_dispersion.json`, `v5b_closed_loop_g1.json`
