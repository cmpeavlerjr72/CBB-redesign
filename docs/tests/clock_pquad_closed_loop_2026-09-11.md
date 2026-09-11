# Clock round 5d -- the pace-quadratic latent (C4) in the engine, paired against the served B1

2026-09-11. Pre-registration `docs/models/clock/experiments.md` section 26
(commit `d940b41`); wiring and tests `e3b7754`; results section 27.

**Verdict: C4 `v5d_glat_pquad` does NOT supersede B1 `v5b_glat_pmean`.** Both
halves of the pre-registered decision rule fail by a little over one measured
floor. `ENGINE_CLOCK` is unchanged, `provisional_clock` is unchanged, this lane
changed no default, and nothing was hand-tuned, capped, scaled, clipped or
blended at any point.

---

## 1. What was wired, and what it is a function of

Round 5c's arm C4: the round-5b game latent with its dispersion made a function
of the game's pregame tempo.

    A = exp(sigma(t) * z + sigma(t)^2 / 2)
    sigma(t)^2 = 0.0020120712 + 5.3916331e-05 * (t - 68.81441)
                              + 2.6196426e-05 * (t - 68.81441)^2

`(b0, b1, b2, tbar)` are READ from
`data/processed/models/clock/v5c_bakeoff/v5c_params.json` key `F2`, the object
section 25's offline grade scored. Nothing is re-derived in the engine. `t` is
`tempo_prior_game`, a PREGAME feature already in the served round-3c design
frame's `TEAM_COLS`.

Four structural properties, each pinned by a test in
`tests/test_clock_adapter_v5d.py`:

| property | why it matters | check |
|---|---|---|
| `t` is game-level | one pace realisation per game, both teams scaled by it (CLAUDE.md) | max abs difference between a game's two team rows = **0.0** over all 5,710 games; the grader asserts it too |
| same stream, same ordinal `2**40` | a C4 run pairs with a B1 run game by game and seed by seed | both arms call `uniforms_at(keys, 2**40)` on the `clock` family; no new RNG family |
| location applied ROW-WISE at that row's own sigma | `E[1/A] = 1` holds per game, so the arm cannot move the possession count's expectation in its own favour | analytic identity asserted at 500 rows |
| `sigma^2` floor 1e-8 never active | the floor is a guard on a degenerate coefficient set, not a clip on output | `b0 + b1*t + b2*t^2 > 1e-6` over the full engine tempo range 58.6-79.5 |

Engine-side sigma over the slate: 0.0445 to 0.0702, mean 0.0477, against B1's
single scalar 0.0470. Mean `sigma^2` 0.002291 against B1's 0.002212.

## 2. The precondition: the served path is bit-identical

Run BEFORE any C4 number was read, as section 26.1 required.

| run | ENGINE_CLOCK | digest |
|---|---|---|
| `smoke60x5_default_v5b` (reference, pre-change, the PM's serving check at `e3ccce5`) | `v5b_glat_pmean` | `492300a7fd1e6dc388a3c47b822f61e04da6501720ef762bde0afe7ba15451a1` |
| `smoke60x5_v5d_wiring` (post-change, same command) | `v5b_glat_pmean` | `492300a7fd1e6dc388a3c47b822f61e04da6501720ef762bde0afe7ba15451a1` |

`scripts/digest_engine_run.py --compare` exits 0: **PASS -- bit-identical
digest**, over 300 game rows and 4,882 player rows with `adapter_flags` in the
hash. Tests: `test_clock_adapter_v5d.py` + `test_clock_adapter_v3.py` **44
passed**; `test_engine.py` **20 passed**.

This is what licenses reusing `results/engine_v0/clk5b_B1_s25` as the paired B1
reference instead of re-running it -- the same precedent section 22.3 used to
reuse `clock4_R_s25`. Same arm, same 500-game subset, same seeds `0..24`, same
`PINNED_SUBMODELS`, and `loop.py` (`b94b947`) and `clock_adapter_v3.py`
(`ce2cd94`) unchanged since it was produced.

## 3. The closed loop -- 25 paired seeds, 500 games, 159 clock-complete

`clk5d_C4_s25`: 555 s, 6 workers, 12,500 rows, 3,153 poss/s. Pinned on both
runs and written into both `run_meta.json`s: `ENGINE_EVENT=round2_s1`,
`ENGINE_FG_MAKE=round3_shooter_S_C_s1`, `ENGINE_FG3=decision8`,
`ENGINE_ROTATION=reference`.

| # | line | B1 | C4 | C4 - B1 | floor | floors | actual |
|---|---|---:|---:|---:|---:|---:|---:|
| L1 | G1 poss mean, all 500 | +1.649 | +1.667 | +0.018 | 0.074 | 0.24 | 0.000 |
| L2 | G1 poss mean, cc | +0.989 | +1.012 | +0.023 | 0.157 | 0.15 | 0.000 |
| L3 | possession SD ratio, cc | 1.03204 | **1.03040** | -0.00164 **toward 1** | 0.0042 | 0.39 | 1.000 |
| L4 | G5 total SD ratio | 0.82267 | 0.82235 | -0.00031 | 0.0165 | 0.02 | 1.000 |
| L5 | `corr(home, away)` | +0.10051 | +0.10028 | -0.00022 | 0.0037 | 0.06 | 0.2374 |
| L6 | `corr(P, eFG%)` | -0.12235 | -0.12371 | -0.00137 | 0.0231 | 0.06 | ~0.000 |
| L7 | G5 margin SD ratio | 0.95634 | 0.95586 | -0.00047 | 0.0115 | 0.04 | 1.000 |
| L8 | total bias | -1.061 | -1.075 | -0.014 | 0.377 | 0.04 | 0.000 |
| **L9** | **G9 calibration slope** | **1.029** | **1.041** | **+0.012** | **0.010** | **1.2 FAIL** | 1.000 |
| L10 | margin SD (points) | 15.881 | 15.922 | +0.041 | 0.081 | 0.51 | 15.472 |
| L11 | within-game poss SD, cc | 4.8147 | 4.8272 | +0.0125 | 0.0857 | 0.15 | 4.6652 |
| -- | PPP | 1.0379 | 1.0375 | -0.0004 | -- | -- | 1.0705 |
| -- | end-of-half share under 35 s | 0.9822 | 0.9830 | +0.0008 | -- | -- | 0.8758 |
| -- | mean last-possession duration (s) | 11.204 | 11.164 | -0.040 | -- | -- | 11.890 |

Floors are section 22.3's measured 25-seed seed-offset floors except L9, which
section 26.3 said would be re-measured here: `clock4_R_s25` slope ratio 1.047
against `clock4_R_s25_floor` 1.057, floor **0.010**. It is the tightest floor in
the table relative to the move it prices, and it is the line C4 fails.

**Per month** (all 500 games; cells under 100 games underpowered, marked `*`),
the possession-mean delta:

| arm | 2024-11 (109) | 2024-12* (83) | 2025-01 (132) | 2025-02 (124) | 2025-03* (52) |
|---|---:|---:|---:|---:|---:|
| B1 | +2.47 | +1.76 | +1.15 | +1.57 | +1.22 |
| C4 | +2.48 | +1.78 | +1.19 | +1.54 | +1.29 |

No month moves more than 0.07 possessions. C4 is not buying one part of the
calendar at another's expense.

## 4. L12 -- responsiveness by pregame-tempo quintile, read CLOSED-LOOP

`scripts/grade_clk5d_quintile_sd.py`. Ratio = engine mean per-game across-seed
possession SD over `SD(actual - sim per-game mean)`, both definitions taken
unchanged from `grade_clk5_dispersion_loop.score` and only cut by quintile.
Five powered cells of 99-100 games.

| arm | Q1 | Q2 | Q3 | Q4 | Q5 | ALL | spread | worst \|dev\| | outside [0.85, 1.15] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R (offset 0) | 0.778 | 0.786 | 0.702 | 0.836 | 0.692 | 0.755 | 0.1445 | 0.308 | all five |
| R (offset 1000) | 0.798 | 0.805 | 0.745 | 0.810 | 0.683 | 0.762 | 0.1274 | 0.317 | all five |
| **B1 (served)** | 0.986 | 0.966 | 0.944 | 1.050 | 0.919 | 0.970 | **0.1312** | 0.081 | **none** |
| **C4** | 0.986 | 0.944 | 0.904 | 1.057 | **0.999** | 0.976 | **0.1537** | 0.096 | **none** |
| per-quintile floor | 0.0206 | 0.0189 | 0.0438 | 0.0260 | 0.0087 | 0.0072 | 0.0171 | -- | -- |

Read cell by cell:

- **Q5, the fast end: 0.919 -> 0.999, +0.0803 against a 0.0087 floor = 9.2
  floors, landing on 1.000 to within 0.001.** The largest per-cell improvement
  any clock arm has produced in the engine, on a line nothing was fitted
  against, and exactly the direction section 25.3 predicted from the offline
  between-component table (C4 the only arm that raises the fast end rather than
  lowering everything).
- **Q3: 0.944 -> 0.904, -0.040 = 0.9 floors AWAY from 1.** **Q2: 0.966 ->
  0.944, -0.022 = 1.2 floors away.** Q1 and Q4 are unmoved (0.01 and 0.3
  floors).
- **Spread 0.1312 -> 0.1537, +0.0225 against a 0.0171 floor = 1.3 floors in the
  wrong direction.** Criterion 2 required it to narrow. It widened.

### The finding that is not about C4

**The offline defect round 5c exists to fix does not reproduce in the engine.**
Offline (section 25.3) B1's Q2 is **1.174**, outside the pre-registered
[0.85, 1.15] band, and that is one of the two failures blocking B1's adoption
(section 22.5). Closed-loop at 25 paired seeds, **B1's five quintiles span
0.919 to 1.050 -- every one inside the band -- and its worst deviation is 0.081
against 0.174 offline.**

The two readings measure different objects. The offline denominator is the
needed per-game duration-residual SD propagated through `P = 1200/Dbar`; the
closed-loop denominator is `SD(actual - sim per-game mean)` over whole simulated
games, which carries every other sub-model's variance and the engine's own state
composition as well. Neither is declared to supersede the other here. What is
recorded is that **they disagree about whether the served arm passes criterion
4**, and a PM adjudicating that criterion now has to say which reading it is
written against. That is a finding, not a licence to take the flattering one.

## 5. Verdict

| # | condition (section 26.4) | C4 |
|---|---|---|
| 1 | no line L1-L11 regresses beyond its measured floor | **FAIL** -- L9 1.2 floors. Nine of the other ten under 0.25 floors |
| 2 | L12 spread narrows beyond floor; no quintile worse outside the band | **FAIL** on the spread (1.3 floors). The band half is vacuous -- neither arm has a quintile outside it |

**C4 is REJECTED as a replacement for B1.** B1 `v5b_glat_pmean` stays the arm
this lane puts forward and stays the served provisional default. The flag
`ENGINE_CLOCK=v5d_glat_pquad` stays wired and default-off for reproduction.

What round 6 inherits, stated as a target rather than a hope: **a dispersion
function whose tempo derivative is large at the fast end and near zero through
the middle.** C4's fitted quadratic has its minimum at tempo 67.79, inside the
middle of the distribution, so it lifts Q5 by 9.2 floors and pushes Q2 and Q3
down on the way. A tempo-dependent dispersion function is not refuted by this
round; this particular quadratic is. And before round 6 fits anything, it should
fix which of the two responsiveness readings its criterion is written against.

## 6. Provenance

| item | value |
|---|---|
| pre-registration | `experiments.md` section 26, commit `d940b41` |
| wiring + tests + grader | commit `e3b7754` |
| offline fit read, never re-derived | `data/processed/models/clock/v5c_bakeoff/v5c_params.json` (commit `4107eb8`) |
| C4 run | `results/engine_v0/clk5d_C4_s25`, 25 seeds from 0, 6 workers, 555 s |
| B1 run (reused) | `results/engine_v0/clk5b_B1_s25`, 25 seeds from 0 |
| floor runs | `results/engine_v0/clock4_R_s25`, `clock4_R_s25_floor` |
| bit-identity smoke | `results/engine_v0/smoke60x5_v5d_wiring` vs `smoke60x5_default_v5b` |
| graders | `grade_clk3c_closed_loop.py`, `grade_clk5_dispersion_loop.py` (both unedited), new `grade_clk5d_quintile_sd.py` |
| graded outputs | `data/processed/models/clock/v5d_closed_loop.json`, `v5d_closed_loop_dispersion.json`, `v5d_quintile_sd.json`, `v5d_quintile_sd_floor.json`, `v5d_R_floor.json` |
