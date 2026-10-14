# Clock round 10 full-size read (E1 vs served v3), 2026-10-14

Registration: `docs/models/clock/experiments.md` section 43. Closed-loop result: section 44. This read: section 45.
PM rulings: PROJECT_STATUS "PM RULINGS for the full-size read" 2-4 (2026-10-13), plus the PM ruling of 2026-10-14 on draw size (four control draws at the full 200 seeds; deciding lines read 200 vs 200; 50-seed-draw version reported beside it).
Opus engine worker. No served default changed. Freeze ON. No file under `src/` changed.

## 1. What ran

| item | value |
|---|---|
| fold / games / seeds | F2 (2024-25), all 5,710 games, 200 seeds per run |
| control | served v3 (`ENGINE_CLOCK` default `v5b_r8K2_glat_pmean`), `clk10full_ctrl_o0_s200`, seeds 0-199 |
| arm | `ENGINE_CLOCK=v5b_r8K2_envE1_glat_pmean`, `clk10full_E1_o0_s200`, seeds 0-199 (paired with the control) |
| floor draws | `clk10full_ctrl_o{1000,2000,3000,4000}_s200`, 200 seeds each, served v3 |
| only difference | `ENGINE_CLOCK` (run_meta `adapter_flags` checked by the grader) |
| engine SHA | `060157f3d4ca682c0e33961d06ad5db73144c337` on every run. `engine_dirty_src` lists only `scripts/*.sh`: `chmod +x` on the box changes the file mode. No file under `src/` is listed. |
| inputs | `data/processed/models/engine_v3`; HF pull of `engine_inputs model_artifacts team_rate_tables raw engine_inputs_v3`; grading-only files scp'd (possessions_v4otc/v4 2025, `results/diag_c4/f2all50_pregame.parquet`, 500 x 25 loop outputs for the environment check) |
| trajectories | all 200 seeds of every run (`--trajectory-seeds o+200`; side channel, game output unchanged); 157.0 M rows and 1.77 GB per run |
| truth | `CBB_TRUTH=verified_v1` |

### Reused round-9 fit (sha256, section 44 match)
All 14 files of `data/processed/models/clock/r9_env_E1/{F1,F2}` hash equal to `results/clock_env10/artifact_hash.json`, the section-44 check: 14 of 14 match.
- The six F2 pickles on the box hash the same: `adc44c57...`, `be355318...`, `7becf6a9...`, `2e45f257...`, `22dea78b...`, `b3884919...`.
- `v5b_bakeoff/v5b_bakeoff_report.json` (B1 latent sigma F2 0.046788, git-tracked) sha256 is `13f244d4303b3a4150ef6982979994a2157413c7f3ffdf9c5ea0c32a0e8476d9`.
- No coefficient was refitted.

## 2. Box, runtime, cost

| # | instance | type / market | AZ | launch (UTC) | end (UTC) | end reason | min | $/h (spot at launch) | approx cost |
|---|---|---|---|---|---|---|---:|---:|---:|
| 1 | i-0c95b12610d5d3ee8 | c7a.48xlarge spot | us-east-2a | 12:28:55 | 12:37:12 | spot reclaim during the HF pull (`Server.SpotInstanceTermination`); nothing run | 8.3 | 2.578 | $0.36 |
| 2 | i-028ce499a8b0c6720 | c7a.48xlarge spot | us-east-2b | 12:37:23 | 14:17:50 terminate call, 14:18:23 confirmed `terminated` | operator terminate | 101.0 | 2.702 | $4.55 |

- **Total about $4.9.** This is instance-minutes x the spot price at launch, not reconciled against the bill.
- Termination check with the CLI after the terminate call:
  - `describe-instances` for both ids returns `terminated`.
  - The filter `pending,running,stopping` returns nothing.
  - `describe-volumes status=available` returns nothing.
- Two STOPPED instances in the account (`kalshi-sniper` t3.small, `cfb-sweep-cloud10k_wk0c`) are not ours and were not touched.
- A `shutdown -h +240` safety net was set on each box.

Timeline on instance 2 (UTC):

| step | times | duration |
|---|---|---|
| pull | 12:39 to 12:45 | |
| build | to 12:55 | |
| parity | 12:55:12 to 12:55:46 | |
| sims | 12:55:54 to 13:32:12 | three pairs, two runs at a time at 94 workers each; each run took 711-726 s |
| post-processing | 13:32 to 13:39 | |
| 50-seed-draw grade | to 13:49 | about 10 min |
| deciding grade | to 14:16 | about 37 min; the six 200-seed trajectories are loaded sequentially, about 6 min each |
| HF upload and fetch | to 14:17 | |

Incident: the root disk (100 GB) filled during post-processing.
- Causes: the image bakes the 27 GB `data/processed` tree (`COPY .`), the 29 GB build cache, and 11 GB of trajectories.
- Effect: the six 50-seed extracts failed with ENOSPC. The 200-seed extracts and dispersions had already completed.
- Fix: `docker builder prune -af` freed 28 GB. The six 50-seed extracts and dispersions were then re-run with identical commands (`post50.sh`, scratch copy of the `post1` f50 half).
- Nothing else was affected.
- Next time: give the box 200 GB of root disk, or prune the build cache right after `docker build`.

### Parity and environment checks (before any number was read)
- **Parity v10 on the box:** `run_parity_smoke_v1.py` against `docs/ops/parity_reference_windows_v10.json` gave **PASS, bit-identical** (60 x 5, served defaults, `results/engine_v0/box1014_parity_v10`). It ran before any sim.
- **Grading environment on the box:** the unedited `grade_clock_env_loop_v1.py` on the 500 x 25 loop outputs reproduces the local `results/clock_env10/grade_s25.json`. That is 772 values with max abs diff 2.7e-15, the same verdict (CANDIDATE) and the same vetoes failing (none).

## 3. Grader change (PM ruling 2026-10-14), committed before any full-size arm output

- `060157f` adds `scripts/grade_clock_env_full_v2.py`, a versioned sibling. `grade_clock_env_full_v1.py` is unchanged.
- It parameterises v1's constants, with defaults equal to v1.
- In `load_tag` it cuts the draws' `games.parquet` and round-8 frame to their first `--draw-seeds` seeds. That cut is a no-op when the run holds exactly that many seeds.
- Grading logic is otherwise byte-identical.

The same commit adds `scripts/grade_clock_env_full_tables_v1.py` and `scripts/box_clk10_full_v1.sh`:
- The tables script reads the two grade JSONs and applies rulings R2-R4 on top of the literal verdict. It computes nothing from sim output.
- The box script handles parity, runs, post-processing and grading.

The two reads:

| read | trajectory lines | draws | floor |
|---|---|---|---|
| deciding | 200 seeds | 200-seed | draw SD unscaled (200 vs 200) |
| 50-seed-draw (v1 definition) | first 50 seeds of every run | first 50 seeds of each draw | 200-seed lines (`r8_*`, `*_200`) use the sqrt(50/200)-scaled draw SD, with the raw draw SD printed beside it, as v1 did |

Diff v1 -> v2 (`git diff --no-index -U0`):

```diff
--- a/scripts/grade_clock_env_full_v1.py
+++ b/scripts/grade_clock_env_full_v2.py
@@ -1 +1,15 @@
-"""grade_clock_env_full_v1.py -- clock round 10 (docs/models/clock/experiments.md section 43) FULL-SIZE paired read of the
+"""grade_clock_env_full_v2.py -- versioned sibling of grade_clock_env_full_v1.py (unchanged). PM ruling 2026-10-14 on draw
+ ... (docstring: the parameters below; defaults reproduce v1) ...
+grade_clock_env_full_v1.py -- clock round 10 (docs/models/clock/experiments.md section 43) FULL-SIZE paired read of the
@@ -46,0 +61,2 @@
+import argparse  # noqa: E402
+
@@ -58,2 +74,11 @@
-DRAWS = [f"clk10full_ctrl_o{o}_s50" for o in (1000, 2000, 3000, 4000)]
-S_DRAW = 50
+_AP = argparse.ArgumentParser()
+_AP.add_argument("--traj-seeds", type=int, default=50)
+_AP.add_argument("--draw-seeds", type=int, default=50)
+_AP.add_argument("--draw-run-seeds", type=int, default=None)
+_AP.add_argument("--disp-dir", default="results/clock_env10/full_dispersion")
+_AP.add_argument("--out", default="results/clock_env10/grade_full_s200.json")
+_A = _AP.parse_args()
+S_TRAJ = _A.traj_seeds
+GL.DISP_DIR = ROOT / _A.disp_dir
+DRAWS = [f"clk10full_ctrl_o{o}_s{_A.draw_run_seeds or _A.draw_seeds}" for o in (1000, 2000, 3000, 4000)]
+S_DRAW = _A.draw_seeds
@@ -66,0 +92,2 @@
+    if tag in DRAWS:
+        gm = gm[gm["seed"] < o + S_DRAW]
@@ -91 +118,2 @@
-    g8 = R8.per_game(R8.load_run(tag))
+    g8r = R8.load_run(tag)
+    g8 = R8.per_game(g8r[g8r["seed"] < o + S_DRAW] if tag in DRAWS else g8r)
@@ -185 +213 @@
-    out = ROOT / "results/clock_env10/grade_full_s200.json"
+    out = ROOT / _A.out
```

Post-processing per run, with the commands in `box_clk10_full_v1.sh post`:
- `diag_c4_extract_chunked_v1.py` on all 200 seeds, then `diag_margin_dispersion_v2.py --seeds 200`, written to `full_dispersion_s200/`.
- The same extract with `--seed-max o+50` (tag `<tag>_f50`), then `--seeds 50`, written to `full_dispersion_s50/`. This is exactly the committed v1 post-processing applied to the first 50 seeds.

Commands:
```
grade_clock_env_full_v2.py --traj-seeds 200 --draw-seeds 200 --draw-run-seeds 200 --disp-dir results/clock_env10/full_dispersion_s200 --out results/clock_env10/grade_full_s200_d200.json
grade_clock_env_full_v2.py --traj-seeds 50  --draw-seeds 50  --draw-run-seeds 200 --disp-dir results/clock_env10/full_dispersion_s50  --out results/clock_env10/grade_full_s200_d50.json
grade_clock_env_full_tables_v1.py --deciding ..._d200.json --d50 ..._d50.json --out results/clock_env10/tables_full.md
```

## 4. Result tables

The full tables, with every line, the draw SD (scaled and raw), the bootstrap SE, and both reads, are in `results/clock_env10/tables_full.md`. They are reproduced in condensed form below.

Floor = max(2 x SD of four control draws, 2 x paired game-bootstrap SE). The bootstrap uses 200 draws over 5,710 games, numpy seed 12345.

### 4.1 Deciding read (200 vs 200)

| line | actual | control | E1 | floor | result (literal s43) | 50-seed-draw read: control / E1 / floor / result |
|---|---:|---:|---:|---:|---|---|
| PRIMARY game possessions on env | 0.1561 | 0.0153 | 0.2066 | 0.0022 | toward by 40.7 floors: PASS (E1 overshoots the actual by 0.051) | 0.0115 / 0.2042 / 0.0045 / PASS (21.5) |
| PRIMARY game total on env | 0.6071 | 0.4098 | 0.6282 | 0.0033 | toward by 53.1 floors: PASS (overshoot 0.021) | 0.4088 / 0.6310 / 0.0070 / PASS (24.9) |
| cross-check possessions on env, 200 seeds from games.parquet (incl. OT) | 0.1561 | 0.0072 | 0.1033 | 0.0010 | toward (98.7) | same |
| cross-check total on env, 200 seeds | 0.5994 | 0.4064 | 0.6246 | 0.0033 | toward (50.8) | same |
| G1 possession mean | 67.875 | 68.748 | 68.728 | 0.023 | pass | pass |
| **G1 possession SD** | 5.474 | 5.629 | 5.650 | 0.014 | **FAIL** (away by 1.5 floors) | 5.632 / 5.659 / 0.020 / **FAIL** (1.4) |
| **half share H1** | 0.4671 | 0.4694 | 0.4696 | 0.0001 | **FAIL** (away by 0.0002 = 1.5 floors) | pass (0.8 floor) |
| half share H2 | 0.5200 | 0.5249 | 0.5247 | 0.0001 | pass | pass |
| G5 margin / total SD ratio | 1 | 1.0416 / 0.9454 | 1.0415 / 0.9474 | 0.0023 / 0.0028 | pass / pass | pass / pass |
| G9 margin bias | 0 | -0.254 | -0.236 | 0.022 | pass | pass |
| **G9 margin slope** | 1 | 0.9568 | 0.9486 | 0.0035 | **FAIL** (away by 2.3 floors) | 0.9598 / 0.9534 / 0.0070 / pass (0.9) |
| tempo elasticity ratio / team pace slope | 1 | 0.771 / 0.669 | 0.787 / 0.677 | 0.009 / 0.008 | pass / pass (both toward) | same |
| within-game N on OREB / FGM / TOV / FTA | -0.023 / 0.381 / 0.422 / 0.200 | -0.0074 / 0.3474 / 0.4720 / 0.2127 | -0.0076 / 0.3467 / 0.4711 / 0.2124 | 0.0013 / 0.0010 / 0.0009 / 0.0005 | all pass | all pass |
| OT (= regulation tie) rate | 0.0557 | 0.0351 | 0.0351 | 0.0005 | pass | pass |
| D1 home margin bias (non-neutral) | 0 | -0.114 | -0.099 | 0.030 | pass | pass |
| **D1 eFG home-away gap** | 0.0293 | 0.0292 | 0.0296 | 0.0002 | **FAIL** (away by 1.2 floors; 0.0004 eFG) | pass (0.6) |
| DISP margin SD ratio overall (regulation) | 1 | 1.0444 | 1.0440 | 0.0024 | pass (literal 0.009: pass) | pass |
| DISP margin ratio by abs-spread Q1 / Q2 / Q3 / Q4 / Q5 | 1 | 1.122 / 1.040 / 1.046 / 1.069 / 0.982 | 1.121 / 1.037 / 1.049 / 1.072 / 0.979 | 0.007 / 0.006 / 0.008 / 0.006 / 0.006 | all pass; literal 0.009 band: all pass (Q3 +0.0036, Q5 -0.0031 away) | all pass; literal 0.009: all pass |
| DISP pace SD ratio overall | 1, in [0.95,1.05] | 1.0112 | 1.0157 | 0.0045 | pass (away 0.0045 = 1.0 floor, not beyond) | pass |
| **DISP pace SD ratio abs-spread Q2** | 1 | 1.0292 | 1.0544 | 0.0128 | **FAIL** (away by 2.0 floors) | 1.0258 / 1.0530 / 0.0139 / **FAIL** (2.0) |
| DISP pace Q1 / Q3 / Q4 / Q5 | 1 | 1.009 / 1.026 / 1.044 / 0.933 | 1.013 / 1.023 / 1.049 / 0.928 | 0.011 / 0.010 / 0.014 / 0.010 | pass | pass |
| DISP regulation total SD ratio | 0.994 [0.974, 1.014] | 0.9936 | 0.9972 | band 0.020 | pass (both inside the interval) | pass |
| DISP pace share of margin variance | reported | 0.0018 | 0.0017 | | reported | 0.0019 / 0.0016 |
| H1 game possessions on env (registered: direction) | 0.0797 | 0.0075 | 0.1010 | 0.0007 | pass | pass |
| overall regulation margin SD ratio rise | | 1.0444 | 1.0440 | 0.0024 | pass | pass |

**Literal section-43 verdict, deciding read: NOT A CANDIDATE.** Both primaries clear. The vetoes failing are `G1_poss_sd`, `half1_share`, `G9_margin_slope`, `D1_efg_gap` and `disp_pace_q2`.

**50-seed-draw read: NOT A CANDIDATE.** The vetoes failing are `G1_poss_sd` and `disp_pace_q2`. Both reads agree on the verdict.

The two vetoes that fail in both reads are G1 possession SD and pace SD ratio at |spread| Q2. The other three fail only at the 200-seed floor:
- H1 share: floor 0.0001, effect 0.0002.
- G9 margin slope: floor 0.0035, effect 0.008.
- D1 eFG gap: floor 0.0002, effect 0.0004.

These three are effects too small to resolve at 50 seeds, not effects that appear only at 200.

### 4.2 PM rulings applied on top of the literal verdict

**R2, first-half magnitude: PASS in both reads.**
- Actual 0.0797, control 0.0075, E1 0.1010.
- E1 is closer by 0.051 (floor 0.0007).
- The overshoot beyond the actual is 0.021, less than the control's shortfall of 0.072.
- The 500 x 25 overshoot flag reproduces in sign: E1 is above the actual in the first half.

**R3, dispersion abs-spread Q3/Q5 with the 0.009 band at the full-size floor: PASS.**
- Q3 is away by 0.0036 (floor 0.0078); Q5 is away by 0.0031 (floor 0.0060).
- Both are inside the literal 0.009 band in both reads.
- The 500 x 25 literal-band fails (Q3 +0.047, Q5 -0.041) do not reproduce at full size.

**R4, possession gap (real - sim) by env quintile, 1,077-1,100 games per cell, not underpowered: PASS at the boundary.**

| env q | games | control gap | E1 gap | floor | further from 0 beyond floor |
|---|---:|---:|---:|---:|---|
| Q1 | 1,081 | -3.30 | -1.05 | 0.05 | no |
| Q2 | 1,082 | -2.41 | -1.54 | 0.04 | no |
| Q3 | 1,089 | -1.61 | -1.56 | 0.04 | no |
| Q4 | 1,077 | -1.33 | -2.13 | 0.02 | **yes** |
| Q5 | 1,100 | -0.27 | -2.44 | 0.05 | **yes** |

Two quintiles are worse, which is not more than two, so R4 passes. The 50-seed read gives the same two quintiles. The pattern is the 500 x 25 overshoot, confirmed at full size: E1 fixes the low-env end and over-pushes the high-env end. The sim already runs too many possessions in every quintile (every control gap is negative), and E1 adds more possessions where env is high.

### 4.3 Multi-level evidence (deciding read)

**Overall:** see the table above.

**Per game:**
- Mean per-game |possessions sim - real| (regulation, both teams): 7.202 -> 7.127. The improvement is 0.075, or 1.8 floors (floor 0.041). The 50-seed read gives 7.294 -> 7.214.
- Within-game possession SD (G1 SD) rises away from the actual, which is the failing veto.

**Per env quintile, total points (real - sim):**

| env q | control | E1 |
|---|---:|---:|
| Q1 | -2.47 | -0.01 |
| Q2 | -0.74 | +0.27 |
| Q3 | -0.14 | -0.05 |
| Q4 | +0.19 | -0.73 |
| Q5 | +1.98 | -0.57 |

- Floors are 0.06-0.14.
- The total-gap spread across quintiles shrinks from 4.45 to 1.0 points.
- Q4 is worse.

**Per home-rating quintile:**
- The possession gap moves by at most 0.4 (Q1 -2.17 -> -1.77; Q5 -0.98 -> -1.32).
- The total gap moves by at most 0.45.
- Floors are about 0.1.

**Per team** (offence-team quintile of x_off; 364 teams; regulation possessions per team-game):

| quintile | real | control | E1 | share of teams moving the right way |
|---|---:|---:|---:|---:|
| Q1 | 67.36 | 68.58 | 68.04 | 0.836 |
| Q2 | 67.50 | 68.45 | 68.24 | 0.740 |
| Q3 | 67.17 | 68.10 | 68.05 | 0.458 |
| Q4 | 67.73 | 68.43 | 68.51 | 0.452 |
| Q5 | 67.71 | 68.38 | 68.93 | 0.384 |
| all | | | | 0.574 |

- The share of teams whose |error| fell is 0.527.
- Across offence quintiles, Q1 to Q5: control is -0.20, flat to inverted; E1 is +0.89; real is +0.35.
- E1 turns the slope the right way but overshoots it.
- The high-offence teams move the wrong way.

**Per possession type** (start type, regulation seconds per possession; real / control / E1):

| start type | real | control | E1 |
|---|---:|---:|---:|
| period_start | 20.84 | 20.36 | 20.36 |
| DREB | 14.41 | 14.30 | 14.30 |
| TOV | 14.79 | 14.87 | 14.87 |
| made_FG | 22.07 | 21.96 | 21.96 |
| made_FT | 18.97 | 18.62 | 18.63 |
| other | 1.70 | 1.72 | 2.00 |

- E1 moves no mean start-type duration beyond 0.01 s, except `other` (+0.28 s, floor 0.016, away from real; 1,603 games).
- The env effect therefore moves possession counts through the per-game scale without moving any start-type mean.

**Seed check:** both primaries and the H1 line are resolved in both reads (2 x draw SD far below the control gap).

### 4.4 Reported lines (not selecting)

| line | actual | control | E1 | floor | note |
|---|---:|---:|---:|---:|---|
| possessions on gap | 0.047 | -0.004 | -0.016 | 0.005 | away |
| lead-change slope on abs(spread) | -0.117 | -0.094 | -0.096 | 0.001 | toward |
| time of decision on abs(spread) | -37.9 | -28.9 | -29.1 | 0.18 | toward |
| G9 total bias H1 / H2+OT | 0 | +0.31 / -0.36 | +0.34 / -0.39 | 0.025 / 0.032 | |
| MD4 H2 last-4 possessions | 15.53 | 16.74 | 16.72 | | |
| game-quintile pace slope | 1 | 0.853 | 0.861 | | |

## 5. Discrepancies between the 500 x 25 read (section 44) and this read

1. **Verdict.** 500 x 25: CANDIDATE, no veto failing. Full size: NOT A CANDIDATE in both reads.
   - Floors are 5-20 times narrower at full size. For example, the G1 SD floor is 0.11 at 500 x 25 and 0.014 here; the pace Q2 floor is 0.049 and 0.013.
   - The two robust failures, G1 SD and pace Q2, were in the same direction at 500 x 25 but inside the much wider floors:
     - G1 SD was -0.015 at 500 x 25, toward the actual; it is +0.021, away, at full size. The sign differs, because the sample differs.
     - Pace Q2 moved +0.005 at 500 x 25 and moves +0.025 here.
2. **Primaries.**
   - Both clear by more at full size (40.7 and 53.1 floors vs 11.5 and 9.6).
   - The possession slope overshoots the actual in both reads: E1 0.207 vs actual 0.156 here; 0.184 vs 0.124 at 500 x 25.
3. **Dispersion abs-spread Q3/Q5.** The literal 0.009 band failed at 500 x 25 and passes at full size.
4. **Env-quintile possession gap.** The overshoot at the high-env end was UNDERPOWERED at 500 x 25 and is now resolved: Q5 -0.27 -> -2.44, about 40 floors. Q1 now improves (-3.30 -> -1.05); the 500 x 25 read had Q1 overshooting positive.
5. **Sample level.** The control G5 margin ratio is 1.042 at full size vs 0.978 on the 500-game sample. The OT rate actual is 0.0557 vs 0.068. These are sample differences, not arm effects.

## 6. Outputs and sync

**Local** (gitignored): `results/clock_env10/`
- `full/*/{games.parquet, run_meta.json}`, about 21 MB per run
- `full_dispersion_s200/`, `full_dispersion_s50/`
- `grade_full_s200_d200.json`, `grade_full_s200_d50.json`
- `tables_full.{md,json}`
- `logs/`

**HF** (`mvpeav/cbb-sim-data`, under the `results/` key prefix, commit `07401e8f`):
- the same files, plus the extracts `results/diag_c4/clk10full_*` without `sim_late`

**Not synced:**
- the six 200-seed trajectories (10.6 GB)
- the `sim_late` extracts (3.0 GB)

These are regenerable bit-identically from SHA `060157f`, the seeds and the commands above. They were left on the terminated box.

## 7. Suite and parity (local, after the run)

- `pytest`: 847 passed, 1 skipped.
- `run_parity_smoke_v1.py` (v10): PASS, bit-identical (60 x 5).
- No file under `src/` changed.
