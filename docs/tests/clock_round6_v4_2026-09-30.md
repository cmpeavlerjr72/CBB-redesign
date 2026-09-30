# Clock round 6 on the event-layer v4 table: L2, L2a, D1 against R (2026-09-30)

Lane B. Pre-registration: `docs/models/clock/experiments.md` section 28 (commit
`33e9db5`), written before any of these arms existed. **Nothing is adopted and
no default is changed.** Every arm is a default-off `ENGINE_CLOCK` mode.

| arm | what it is | `ENGINE_CLOCK` |
|---|---|---|
| R | served `v5b_glat_pmean`: the v3c `srfloor_P3` S1 law (trained on the v1 table) plus the B1 latent | `v5b_glat_pmean` |
| **L2** | the same spec refitted on `possessions_v4` (phantoms removed; and-one successors labelled as the engine labels them) | `v5b_r6L2_glat_pmean` |
| **L2a** | refitted on `possessions_v4a` (phantoms removed; made-and-one successor kept at the old `made_FG` label) | `v5b_r6L2a_glat_pmean` |
| **D1** | L2's table with the round-4 `calpart` S1 refit (season-part pooling, L34) | `v5b_r6D1_glat_pmean` |

**Build.**
- **Tables.** `possessions_v4a` came from `scripts/build_possessions_v4.py
  --variant l2a`; its default path is again bit-identical to `possessions_v2`
  in 2022-2025.
- **Arm directories.** `scripts/exp_clk6_r6_arms_v1.py` writes each arm's
  artifacts under `data/processed/models/clock/r6_<arm>/`:
  - a censoring side table (the horn rule of `diag_clock_censoring_v1.py`,
    joined on v4's own `poss_index`);
  - the design (`clock.build_design` on the v4 table);
  - the S1 schedule;
  - the refitted B1 latent sigma.
- **Trainers.** The served trainers run unedited, with only their path
  constants redirected: `train_clock_v3c_s1.py` for L2/L2a,
  `train_clock_v4.py --only calpart` for D1, and
  `exp_clk5b_mean_consistent.py --fit-only` for the latent.
- **Refitted B1 sigma:** L2 0.04682, L2a 0.04688, D1 0.04696 (served 0.04703).

**Wiring.**
- `src/cbb_sim/engine/clock_adapter_v3.py`: new mode keys only (commits
  `eb6b83a` for L2/L2a, `37d1361` for D1). `git diff` showed only these hunks.
- **Flag-off parity:** two 60x5 smokes, both **PASS, bit-identical** to
  `docs/ops/parity_reference_windows_v6.json`.
- `tests/test_clock_adapter_v3.py`: 37 passed.

## 1. Offline (fold 2 selects; `scripts/grade_clock_r6_offline_v1.py` -> `results/clock_r6/offline_grade.json`)

The truth is the v4 table's 2025 rows: clock-complete games, regulation,
257,897 possessions in 1,916 games. The quantity is the one `loop.py`
subtracts, E[min(T,R)], with each arm's own S1 schedule routed by game date.

| arm | model - actual mean consumed (s) | implied possessions per team-game | made-FG cell gap (s) | made-FT cell gap (s) | team tempo span ratio |
|---|---:|---:|---:|---:|---:|
| R | -0.337 | **+1.298** | -0.658 | -0.672 | 0.387 |
| **L2** | -0.138 | **+0.527** | **-0.222** | -0.487 | 0.388 |
| L2a | -0.175 | +0.668 | -0.307 | -0.548 | 0.386 |
| **D1** | -0.133 | **+0.507** | -0.221 | -0.499 | 0.382 |

**Start-type composition of the truth:** DREB 35.3%, made_FG 35.1%, TOV
17.1%, made_FT 10.7%, period_start 1.5%, other 0.4%.

Possession contribution by start type (R -> L2):

| start type | R | L2 |
|---|---:|---:|
| made_FG | +0.890 | +0.296 |
| made_FT | +0.276 | +0.198 |
| DREB | +0.081 | +0.025 |
| TOV | +0.025 | -0.020 |
| period_start | +0.023 | +0.022 |

Refitting on v4 removes 0.59 of the made-FG cell's 0.89 possessions, which
were the phantoms and the and-one successors in its training cell.

**Season drift** (model - actual, seconds, by month of 2025):

| arm | Nov | Dec | Jan | Feb | Mar |
|---|---:|---:|---:|---:|---:|
| R | -0.32 | -0.17 | -0.44 | -0.37 | -0.34 |
| L2 | -0.12 | +0.03 | -0.24 | -0.16 | -0.14 |
| D1 | -0.17 | -0.03 | -0.19 | -0.13 | -0.11 |

April is underpowered. What remains is January-March, L34's in-season drift.
D1's season-part pooling trims it by 0.03-0.05 s.

**Per-team pace responsiveness is compressed in every arm.** Team tempo
quintile means span only 0.38-0.39 of the actual span (R quintiles 17.97 ->
16.92 s against actual 19.15 -> 16.43 s). The refits do not touch it; it is
the clock's tempo-tercile cell resolution and is open.

**Floors.**
- **Offline reseed floor: 0 by construction.** `empirical_km3_srfloor` is
  not in `clock_v3.STOCHASTIC_ARMS`, so a spec-identical refit under another
  seed is the identical object.
- **Fold 1 (test 2024): NOT RUN.** The served S1 trainer fits fold 2 only.
- D1 vs L2 (0.020 possessions) is therefore read in the closed loop.

## 2. Closed loop (`scripts/run_clk6_closed_loop_sample_v1.py`, graded by `scripts/grade_clock_r6_loop_v1.py` under `CBB_TRUTH=verified_v1`)

- **Inputs:** `--input-dir data/processed/models/engine_v3`,
  `--sample-file data/processed/truth/stride500_verified_v1_F2_2025.parquet`,
  25 seeds, ENGINE_EVENT `round2_s1`; only the clock pin differs.
- **Runs:** R, L2, L2a and D1 on seeds 0-24 (paired); **R floor runs Rf1
  (seeds 1000-1024) and Rf2 (2000-2024)**.
- **Truth:** the primary is scored on the 476 sample games with a complete
  v4 layer.
- **Bootstrap:** 1,000 paired game draws on every line (arm - R).
- **First-half share:** from halftime-score taps (`scripts/diag_g1g5_tap_v3.py`,
  10 seeds per arm). Their game rows are **bit-identical to the runner's**
  (5,000 of 5,000 rows, 0 mismatching cells, every arm).

| line | R | **L2** | L2a | **D1** | Rf1 | Rf2 | floor (max pairwise of R, Rf1, Rf2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **primary: count - v4 count** | **+1.793** | **+1.020** | +1.176 | **+1.007** | +1.799 | +1.824 | 0.031 |
| arm - R, paired bootstrap SE | | -0.773 (0.033) | -0.617 (0.031) | -0.786 (0.034) | +0.006 | +0.031 | |
| count - box estimator (old G1 line) | +2.159 | +1.386 | +1.544 | +1.372 | +2.180 | +2.206 | 0.047 |
| possession SD, pooled (v4 truth 5.690; box 5.532) | 5.553 | 5.457 | 5.470 | 5.472 | 5.586 | 5.592 | 0.039 |
| G5 margin SD ratio (sim SD / resid SD) | 1.010 (12.27/12.15) | 0.995 (12.20/12.26) | 1.004 (12.22/12.17) | 1.001 (12.14/12.13) | 1.001 | 1.017 | 0.015 |
| G5 total SD ratio (sim SD / resid SD), PROVISIONAL at 25 seeds | 0.912 (15.91/17.46) | 0.888 (15.62/17.59) | 0.895 (15.66/17.50) | 0.889 (15.57/17.53) | 0.919 | 0.914 | 0.007 |
| first-half share, regulation (actual v4 pbp 0.4737) | 0.4728 | 0.4740 | 0.4736 | 0.4736 | 0.4740 | -- | 0.0012 (R vs Rf1) |
| OT rate (verified finals 0.058) | 0.0313 | 0.0287 | 0.0289 | 0.0302 | 0.0288 | 0.0344 | 0.0056 |

**Reading under the section-28 rule.**
- **Primary.** Every refitted arm moves the count toward 0 by 20-25 floors
  (L2 -0.773, D1 -0.786, L2a -0.617; floor 0.031).
- **D1 vs L2:** 0.013, inside the floor. A tie, so it goes to the simpler L2.
- **L2 vs L2a:** L2 beats L2a by 0.156 (5 floors). **Relabelling to the
  floor's start types adds to the phantom fix**, as section 5.3 of the G1
  diagnostic predicted.

Vetoes:

| veto | result |
|---|---|
| G5 margin SD ratio | passes (every arm closer to 1) |
| First-half share | passes (every arm closer to the actual) |
| OT rate | passes against the max-pairwise floor. L2's -0.0026 is at the single-pair floors (0.0025, 0.0031) |
| **Possession SD** | **fires for all three.** The pooled SD falls 0.08-0.10 (bootstrap SE about 0.04) and moves away from both truths. The level drop takes spread with it; every arm stays inside G1's +/-0.75 SD tolerance. |
| **G5 total SD ratio** | **fires for all three, PROVISIONAL at 25 seeds.** The ratio falls 0.017-0.024 against a 0.007 floor (SE about 0.007). By component: the sim total SD drops by about 0.3, fewer possessions giving less total variance, while the residual SD moves +0.05 to +0.14. |

**By the pre-registered rule no arm passes: each fails the possession-SD
veto and the (provisional) G5 total SD veto.**

The G5 failure is the compensation the G1/G5 diagnostic predicted (section
2.4 there). R's total SD ratio was propped up by the excess possessions; the
real shortfall is the missing shared whistle latent (lane A's next round).
Fixing the count exposes it rather than causing it.

## 3. The count under the like-for-like definition

| scope | served stack R | best arm |
|---|---:|---:|
| this sample (476 games, 25 seeds) | +1.793 | L2 +1.020, D1 +1.007 |
| full 2025 slate (200 seeds, section 3 of the v4 doc) | +1.641 | -- |

What remains after L2 (+1.0):
- the offline law residual (+0.53: made-FT, made-FG and L34 drift);
- the upstream start-type composition (OREB / FT / make rates; G1
  diagnostic section 1.4);
- the overtime deficit, which offsets part of it (-0.3).

**Shipping note (section 28.4):** the count fix alone moves G9's total about
-2.2 points (L2's -0.77 possessions is about -1.6 points). It can ship only
together with the G4 fixes.

## 4. What did not run

1. **Fold 1** (no F1 S1 schedule in the served trainer).
2. **An offline reseed floor:** it is structurally 0 and is stated, not run.
3. **The first-half share at 25 seeds.** It was read at 10 seeds per arm, with
   one tap floor (R vs Rf1).
4. **A G5 read beyond 25 seeds.** The G5 lines are provisional.
5. **D1's own fold-1 half-life.** `calpart` does not use one; the round-4
   `v4_f1_halflife.json` was reused unchanged via `--skip-f1`.

Artifacts: `data/processed/models/clock/r6_{L2,L2a,D1}/` and
`data/processed/possessions_v4a/` (gitignored). Runs:
`results/engine_v0/clk6_{R,L2,L2a,D1,Rf1,Rf2}_s25`,
`results/clock_r6/tap_*`, and the grades `results/clock_r6/{offline_grade,loop_grade}.json`.
