# Clock round 7: team pace responsiveness (lane H, 2026-09-30)

Pre-registration: `docs/models/clock/experiments.md` section 32 (commit `4a747d9`, before any fit). Offline result and closed-loop registration: section 33 (commit `759be01`, before any loop). Run log (EDT, from `Get-Date`):

| time | step |
|---|---|
| 22:24 | diagnosis |
| 22:27 | pre-registration committed |
| 22:29-22:32 | arm fits |
| 22:33-22:35 | offline grade |
| 22:36 | latent refit |
| 22:37 | flag-off parity |
| 22:39 | section 33 + wiring committed |
| 22:40 | box request `laneH_1` |
| 22:40-22:43 | L2 path check |
| 22:43-23:35 | local loops |
| 23:35 | grades |

**Nothing is adopted and no default changes.** The new mode `ENGINE_CLOCK=v5b_r7A2_glat_pmean` is default-off.

## 1. Diagnosis: why the slope is 0.38-0.40

Script `scripts/diag_clock_r7_pace_v1.py` -> `results/clock_r7/diag.json`. Data: 2025, the v4 L2 design and L2 schedule, 257,897 regulation possessions.

| question | answer (evidence) |
|---|---|
| Are team pace features in the clock? | Only one: `tempo_prior_game` = off_rel x def_rel x the as-of league mean. That is a **raw level**, which the modelling rules ban. |
| Encoding? | A **tercile**, and it is the last cell dimension; 99.9% of rows are served at full depth. Within a tercile the law is flat (e.g. 18.42/18.44/18.49 s across sub-thirds) while the actual spans about 1 s (19.18/18.50/18.18). |
| Offence vs defence? | **The law is symmetric; the truth is not.** OLS of duration on log rels: actual offence -28.9, defence -5.8; L2 offence -12.7, defence -12.9. By defence quintile the L2 model spreads 2.9x the actual span (slope 2.1). Split-half reliability of a team's own offensive duration is 0.70-0.77; of its defensive duration only 0.23-0.25. |
| Start types? | Not the cause. The compression is uniform: DREB 0.42, TOV 0.45, made_FG 0.38, made_FT 0.41, period_start 0.44. |
| Site? | Not the cause: neutral 0.41, offence away 0.42, offence home 0.39. |
| Month / feature noise? | Worst in November (0.30, rising to 0.56 by March). Corr(as-of off_rel, season offensive duration) is -0.62 in November vs -0.79 to -0.82 from January on. |
| Ceiling? | A leave-one-game-out team mean (an oracle, not usable) reaches slope 0.98. The spread is persistent and reachable. |

## 2. Offline (`scripts/grade_clock_r7_offline_v1.py` -> `results/clock_r7/offline_grade.json`)

**Arms.**
- **C0** = L2, refit (identity with L2: max abs diff 0.0).
- **A1** = no-tempo cells x a continuous symmetric AFT scale.
- **A2** = the A1 cells with offence/defence AFT scales by start type.
- **A3** = A2 + walk-forward partially pooled team offence/defence effects. Training estimates: carry c = 0.59/0.53 and kappa = 78/224 possessions (F2).

**Floors.** Reseed floor 0 by construction. Each floor is 2x the paired game-bootstrap SE.

| fold | arm | deviance (d vs C0, floor) | offence-q slope | defence-q slope | game-prior-q slope | implied poss / team-game | per-game count cal. slope |
|---|---|---|---:|---:|---:|---:|---:|
| F2 | C0 | 6.98976 | 0.401 | 2.106 | 0.826 | +0.527 | 0.913 |
| F2 | A1 | -0.0245 (0.0039) | 0.576 | 2.992 | 1.061 | +0.737 | 0.844 |
| F2 | **A2** | -0.0291 (0.0041) | **0.987** | **0.906** | 1.062 | +0.733 | 0.843 |
| F2 | A3 | -0.0311 (0.0044) | 1.019 | 0.850 | 1.070 | +0.657 | 0.796 |
| F1 | C0 | 6.99254 | 0.474 | 2.041 | 0.878 | +0.369 | 0.855 |
| F1 | A1 | -0.0338 (0.0052) | 0.684 | 2.931 | 1.179 | +0.201 | 0.769 |
| F1 | A2 | -0.0382 (0.0051) | 1.121 | 1.077 | 1.180 | +0.196 | 0.769 |
| F1 | A3 | -0.0384 (0.0051) | 1.131 | 1.137 | 1.178 | +0.147 | 0.711 |

**By segment (A2, F2), offence-quintile slope:**

| segment | slope |
|---|---|
| DREB | 0.99 |
| TOV | 1.00 |
| made_FG | 0.97 |
| made_FT | 0.96 |
| November | 0.71 |
| December | 0.87 |
| January | 1.02 |
| February | 1.17 |
| March | 1.31 |
| site | 0.96-1.03 |

**Start-type gap worsening:** at most +0.06 s.

**Per team (F2): RMS of the per-team mean residual, seconds.**

| arm | offence | defence |
|---|---:|---:|
| C0 | 0.995 | 0.716 |
| A1 | 0.905 | 0.775 |
| A2 | 0.797 | 0.653 |
| A3 | 0.633 | 0.578 |

Correlation of team-game predicted vs actual mean duration: C0 0.40, A2 0.54, A3 0.61. Per game, the count error is noise-dominated (mean |err| 6.75 for C0 vs 6.49-6.52).

**Reading.**
- Every arm wins both primaries by 6-7 deviance floors.
- A2 is the simplest arm within one floor of the best on the registered lines.
- A3's per-team gain is real and large, but it is not a registered line.
- **The count-gap veto fires on F2 for every arm:** +0.21 for A1/A2 and +0.13 for A3, against a limit of 0.10. On F1 the gap improves.
- The cause is in-season drift. The A2 gaps are Nov +0.05 and Jan-Mar -0.32 to -0.35 s, against C0's -0.14 to -0.24. The as-of league tempo mean inside C0's raw prior falls from about 70.0 (Nov) to about 68 (Mar) every season, so C0 was partly tracking the L34 drift through a banned raw level.
- **By the literal rule there is no winner.** A2 went to the loop as a Decision-11 read (section 33).

## 3. Closed loop, A2 vs L2 (`scripts/grade_clock_r7_loop_v1.py`, `CBB_TRUTH=verified_v1`)

**Wiring.**
- Flag-off parity 60x5: PASS, bit-identical (`0d4ddccc...029f`).
- L2 path: rerun bit-identical to lane B's `clk6_L2_s25` (1,500 rows, seeds 0-2).
- B1 latent sigma refit: 0.04326 (L2 0.04682).

**Full size: box `laneH_1`, 5,705 x 200, paired with `v3full_L2_s200_o0`.**
- **Floor:** the max of 2x bootstrap SE and the S0 seed draws. The local S0f1-f4 copies hold only 50 seeds each, so the floor is conservative.
- **Seed draws:** 4 draws, so Decision 12 is met.
- **G5:** read at full size.

**Local sample: 476 pc games, seeds 0-99 paired.**
- **Floor:** the R/Rf1/Rf2 and A2f1 draws (25 seeds each) plus the bootstrap.
- **G5 lines are UNDERPOWERED.**

| line | L2 full | A2 full | diff (floors) | sample diff (floors) |
|---|---:|---:|---:|---:|
| G1 count - v4 count | +0.797 | +0.970 | **+0.173 (3.7, worse)** | +0.166 (1.0) |
| pooled possession SD (v4 truth 5.57 full / 5.69 sample) | 5.492 | 5.839 | +0.346 (9.4); distance to truth 0.07 -> 0.27 | +0.349 (2.9) |
| **sim team pace slope, team quintile** | 0.672 | **1.017** | **+0.345 (15.3)** | +0.323 (4.9) |
| sim pace slope, game-prior quintile | 0.862 | 1.167 | +0.305 (12.1), overshoots 1 | +0.280 (3.7) |
| G5 total SD ratio | 0.917 | 0.909 | -0.008 (0.7) | underpowered |
| G5 total sim SD / resid SD | 15.69 / 17.11 | 15.47 / 17.02 | sim -0.22 (3.1) | underpowered |
| G5 margin SD ratio | 1.045 | 1.047 | +0.002 (0.1) | underpowered |
| G9 total bias | -2.01 | -1.52 | +0.48 (4.6, better) | +0.52 (1.3) |
| G9 total MAE | 13.44 | 13.36 | -0.07 (0.6) | +0.05 (0.1) |
| **G9 total slope (MC-corrected)** | 0.976 | **0.796** | **-0.179 (5.5, worse)** | -0.236 (2.3) |
| G9 margin bias | -0.117 | -0.105 | +0.01 (0.2) | +0.02 (0.1) |
| G9 margin slope (MC-corrected) | 0.935 | 0.936 | +0.0005 (0.1) | -0.001 (0.1) |
| OT rate (verified 0.056) | 0.0304 | 0.0303 | -0.0002 (0.3) | 0.0 |

**Team quintile sim possessions vs actual (full size).**
- Actual: 65.20 / 66.67 / 68.38 / 69.41 / 71.27.
- L2: 67.09 / 67.89 / 68.99 / 69.85 / 71.12.
- A2: 66.08 / 67.71 / 69.28 / 70.43 / 72.30.

**Total-slope decomposition (full size, per-game sim means).**

| quantity | L2 | A2 | actual |
|---|---:|---:|---:|
| SD of predicted game possessions | 2.68 | 3.53 | |
| corr with actual v4 count | 0.43 | 0.49 (better discrimination) | |
| count calibration slope | 0.92 | 0.79 (over-spread) | |
| corr(sim possessions, sim points per possession) | -0.03 | +0.15 | -0.09 |

The game level overshoots in the sim (1.17) more than offline (1.06). The likeliest reading is that other sub-models' start-type composition already carries part of the tempo signal, and the clock's new offence elasticity adds to it. This is not traced.

## 4. Verdict

| line | verdict |
|---|---|
| Offline primaries | PASS for A1, A2 and A3, both folds |
| Offline F2 count-gap veto | FIRES for all arms (exposed in-season drift) |
| Closed loop: team responsiveness | **PASS** (0.67 -> 1.02, 15 floors) |
| Closed loop: G1 count | **FAILS**, +0.17 worse (3.7 floors) |
| Closed loop: possession SD | **FAILS**, moves away from truth |
| Closed loop: G9 total slope | **FAILS**, 0.98 -> 0.80 (5.5 floors) |
| Closed loop: G9 margin slope | unchanged (0.936), so the E3 Stage-C over-spread is not affected |
| Closed loop: G5 margin, G5 total ratio, OT | inside floors |

**Status: REFUTED as a ship candidate in this form.** It does not reach VALIDATED-PENDING-SHIP-ACTION: the total-slope failure is a new over-spread, not an exposed compensation.

**Recommended next round:**
- A2/A3 with the elasticity re-estimated jointly with the sim's own start-type composition, or a game-level calibration of the offence elasticity on the closed loop's own data. This is a fitted sub-model change, not a post-hoc scale.
- Plus a calpart-style in-season level term (D1) for the drift exposed here.

## 5. Not run / caveats

- **Full-size A2 floor draw (seeds 1000-1199):** box tier 2 was queued and had not arrived.
- **First-half share:** not measured (needs taps).
- **A3:** not wired. It needs a per-(game, offence) side table in the engine.
- **The locally graded S0 seed draws hold 50 of 200 seeds each.**

**Artifacts.**
- Tracked: `data/processed/models/clock/r7_A2/F2/` and `r7_A2/v5b_bakeoff/`.
- Local only: `r7_{C0,A1,A3}`, `results/clock_r7/*`, and runs `results/engine_v0/laneH_*`.
