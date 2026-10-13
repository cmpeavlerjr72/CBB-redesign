# Clock round 10: closed-loop test of the scoring-environment AFT `E1`, results 2026-10-13

Pre-registration: `docs/models/clock/experiments.md` section 43 (PM Fable 5.1, committed `0ed3d18` before any E1 loop ran).
Results section: `experiments.md` section 44. **E1 is a CANDIDATE by the literal section-43 rule on the 500 x 25 closed loop.
NOT ADOPTED. No served default changed. Model freeze ON. The full-size 5,710 x 200 paired read is QUEUED, not run (PM time
limit).** Opus engine worker.

## 1. Outcome

| step | result |
|---|---|
| E1 artifacts are round 9's fit, unchanged | PASS: sha256 of all 14 files under `r9_env_E1/{F1,F2}` equal the HF `model_artifacts/clock/r9_env_E1` copies; every pickled refit's coefficient equals its manifest row (`scripts/diag_clock_env10_artifact_hash_v1.py` -> `results/clock_env10/artifact_hash.json`). No coefficient refitted |
| B1 game-latent sigma refit on the E1 law | unedited `exp_clk5b_mean_consistent.py --fit-only` via `scripts/exp_clk10_latent_env_v1.py` (sibling of `exp_clk8_latent_v2.py`; only schedule and output redirected): F2 **0.046788** (served K2 0.047101), F1 0.046290 (K2 0.046825); m +0.001095, c 1.002192; bootstrap refits 0.045600 / 0.049921. Report `data/processed/models/clock/r9_env_E1/v5b_bakeoff/v5b_bakeoff_report.json` |
| engine smoke of `ENGINE_CLOCK=v5b_r8K2_envE1_glat_pmean` | runs (30 games x 2 seeds); the adapter loads the r9_env_E1 latent (sigma 0.046788) |
| parity v10, flags off, clean `src/` | PASS bit-identical, 13:49 and 16:15 EDT (60 x 5) |
| control reproducibility | `clk10_ctrl_o0_s25` games.parquet is identical to this morning's `eg_ctrl_o0_s25` (same sample, seeds, served v3) |
| seed-count check | every deciding line RESOLVED at 25 seeds; no 50-seed rerun |
| decision (literal) | **CANDIDATE**: both primaries clear their floors (11.5 and 9.6 floors); no hard veto fails |
| full-size 5,710 x 200 read | **QUEUED, not run (PM time limit)**. It had been started before the PM's time-limit message (control run at 570 of 768 blocks); the worker stopped its own chain and control run, kept no output, and touched nothing else |

## 2. Setup

- Runs: `scripts/run_engine_window_v2.py` (via `scripts/chain_clk10_env_loop_v1.sh`), fold 2 / 2025, served stack v3, sample
  `data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet` (500 games, the endgame round's sample), 25 seeds,
  trajectories for every seed. Control `clk10_ctrl_o0_s25` (no ENGINE_* flag), floor draws `clk10_ctrl_o{1000,2000,3000,4000}_s25`,
  arm `clk10_E1_o0_s25` (`ENGINE_CLOCK=v5b_r8K2_envE1_glat_pmean` only; seeds 0-24 shared with the control). 125-135 s per run.
- Grader `scripts/grade_clock_env_loop_v1.py`, committed in `9d28d1a` BEFORE the arm ran and smoke-tested on control-only data.
  It imports the line machinery of `grade_endgame_v1` (G1, half shares, G5, G9, D1, D2 lines, tie rate) and of
  `grade_clock_r8_loop_v1` (tempo elasticity ratio, team pace slope, within-game possessions on OREB / FGM / TOV / FTA) unedited.
  Dispersion gates: `scripts/diag_margin_dispersion_v2.py` (v1's computations; only prefix, seed count and output dir are
  parametrised, and the 4c rows are saved) on each run's own `diag_c4_extract_v1` extract. Its values are the gate values;
  in-grader replicas (identical point values) supply the paired-bootstrap SE.
- Floor per line = max(2 x SD of the four draws, 2 x paired game-bootstrap SE); 200 draws, seed 12345. Truth: verified finals
  (`CBB_TRUTH=verified_v1`). Possession-level real lines use the 492 games whose pbp final equals the verified final.
- Grader conventions fixed in its docstring before the arm ran:
  - Dispersion baseline = the paired control on this sample (the 1.047 / 1.126 ... baselines are 5,710 x 50 values).
  - Margin band = max(0.009, floor); the literal 0.009 reading is reported alongside.
  - Total SD ratio "holds" means |E1 - control| <= max(0.020, floor).
  - The H1 env slope must move toward the actual (direction). A move within one floor is flagged.
  - D1 "within its seed SE" means no regression beyond the floor.
  - G9 total bias and the D3 period-end lines are reported, not selecting.

## 3. Closed-loop table (500 x 25, paired, four floor draws)

The "delta / floor" column for vetoes and reported lines = (|control - target| - |E1 - target|) / floor. Positive means E1 is closer to the target.

| line | actual / target | control | E1 | delta | floor | delta / floor | result |
|---|---:|---:|---:|---:|---:|---:|---|
| PRIMARY D2_poss_on_env | 0.1240 | -0.0124 | 0.1836 | 0.1960 | 0.0067 | toward by 11.5 floors | PASS |
| PRIMARY D2_total_on_env | 0.6676 | 0.4066 | 0.6368 | 0.2303 | 0.0240 | toward by 9.6 floors | PASS |
| veto G1_poss_mean | 68.3108 | 68.8826 | 68.8470 | -0.0356 | 0.0960 | +0.4 toward | pass |
| veto G1_poss_sd | 5.1951 | 5.6583 | 5.6431 | -0.0152 | 0.1144 | +0.1 toward | pass |
| veto half1_share | 0.4696 | 0.4693 | 0.4696 | 0.0003 | 0.0014 | +0.1 toward | pass |
| veto half2_share | 0.5179 | 0.5247 | 0.5250 | 0.0002 | 0.0009 | -0.3 toward | pass |
| veto G5_margin_sd_ratio | 1.0000 | 0.9778 | 0.9786 | 0.0009 | 0.0212 | +0.0 toward | pass |
| veto G5_total_sd_ratio | 1.0000 | 0.9248 | 0.9238 | -0.0010 | 0.0171 | -0.1 toward | pass |
| veto G9_margin_bias | 0.0000 | 0.0196 | -0.0407 | -0.0603 | 0.1978 | -0.1 toward | pass |
| veto G9_margin_slope | 1.0000 | 1.0015 | 0.9924 | -0.0090 | 0.0351 | -0.2 toward | pass |
| veto r8_elast_ratio | 1.0000 | 0.9180 | 0.9234 | 0.0054 | 0.0392 | +0.1 toward | pass |
| veto r8_slope_teamq | 1.0000 | 0.8098 | 0.8055 | -0.0043 | 0.0342 | -0.1 toward | pass |
| veto r8_N_on_oreb | -0.0739 | -0.0120 | -0.0104 | 0.0016 | 0.0169 | -0.1 toward | pass |
| veto r8_N_on_fgm | 0.3669 | 0.3550 | 0.3560 | 0.0010 | 0.0099 | +0.1 toward | pass |
| veto r8_N_on_tov | 0.4081 | 0.4709 | 0.4686 | -0.0023 | 0.0085 | +0.3 toward | pass |
| veto r8_N_on_fta | 0.1907 | 0.2134 | 0.2114 | -0.0020 | 0.0062 | +0.3 toward | pass |
| veto tie_rate | 0.0680 | 0.0362 | 0.0329 | -0.0033 | 0.0046 | -0.7 toward | pass |
| veto D1_home_margin_bias | 0.0000 | 0.2290 | 0.2498 | 0.0208 | 0.2082 | -0.1 toward | pass |
| veto D1_efg_gap | 0.0305 | 0.0312 | 0.0303 | -0.0010 | 0.0018 | +0.3 toward | pass |
| veto disp_regmargin | 1.0000 | 1.0031 | 1.0035 | 0.0004 | 0.0228 | -0.0 toward | pass; band 0.0228; literal 0.009 reading: pass |
| veto disp_regmargin_q1 | 1.0000 | 1.1716 | 1.1492 | -0.0224 | 0.0746 | +0.3 toward | pass; band 0.0746; literal 0.009 reading: pass |
| veto disp_regmargin_q2 | 1.0000 | 0.9698 | 0.9998 | 0.0300 | 0.0565 | +0.5 toward | pass; band 0.0565; literal 0.009 reading: pass |
| veto disp_regmargin_q3 | 1.0000 | 0.9966 | 1.0431 | 0.0465 | 0.0728 | -0.5 toward | pass; band 0.0728; literal 0.009 reading: FAIL |
| veto disp_regmargin_q4 | 1.0000 | 1.0457 | 1.0457 | 0.0001 | 0.0691 | -0.0 toward | pass; band 0.0691; literal 0.009 reading: pass |
| veto disp_regmargin_q5 | 1.0000 | 0.9682 | 0.9268 | -0.0414 | 0.0479 | -0.9 toward | pass; band 0.0479; literal 0.009 reading: FAIL |
| veto disp_pace | 1.0000 | 1.0218 | 1.0257 | 0.0039 | 0.0179 | -0.2 toward | pass; in [0.95,1.05]: True |
| veto disp_pace_q1 | 1.0000 | 0.9723 | 0.9791 | 0.0068 | 0.0376 | +0.2 toward | pass |
| veto disp_pace_q2 | 1.0000 | 1.0812 | 1.0861 | 0.0049 | 0.0488 | -0.1 toward | pass |
| veto disp_pace_q3 | 1.0000 | 1.0151 | 1.0056 | -0.0095 | 0.0463 | +0.2 toward | pass |
| veto disp_pace_q4 | 1.0000 | 0.9807 | 1.0025 | 0.0218 | 0.0624 | +0.3 toward | pass |
| veto disp_pace_q5 | 1.0000 | 1.0288 | 1.0348 | 0.0060 | 0.0453 | -0.1 toward | pass |
| veto disp_regtotal | - | 0.9480 | 0.9514 | 0.0033 | 0.0144 | +0.2 | pass; band 0.0200; arm in [0.974,1.014]: False, control: False |
| veto H1_poss_on_env | 0.0403 | -0.0104 | 0.0836 | 0.0940 | 0.0108 | +0.7 toward | pass; FLAG: toward but within one floor |
| veto REG_regmargin_sd_rise | - | 1.0031 | 1.0035 | 0.0004 | 0.0228 | +0.0 | pass |
| reported D2_poss_on_gap | 0.0852 | 0.0076 | 0.0010 | -0.0066 | 0.0240 | -0.3 toward | reported |
| reported D2_leadchg_on_spread | -0.0646 | -0.0979 | -0.0982 | -0.0003 | 0.0119 | -0.0 toward | reported |
| reported D2_tdec_on_spread | -32.9112 | -30.5242 | -30.8644 | -0.3401 | 1.5234 | 0.2 toward | reported |
| reported G9_total_bias | 0.0000 | -0.6930 | -0.6390 | 0.0541 | 0.2388 | 0.2 toward | reported |
| reported G9_total_bias_H1 | 0.0000 | -0.3624 | -0.2958 | 0.0666 | 0.2873 | 0.2 toward | reported |
| reported G9_total_bias_H2OT | 0.0000 | -0.3468 | -0.3721 | -0.0253 | 0.1968 | -0.1 toward | reported |
| reported MD4_h2_last4_poss | 15.4286 | 16.7499 | 16.7525 | 0.0026 | 0.0756 | -0.0 toward | reported |
| reported h1buz_ppp | 0.8333 | 0.9258 | 0.9429 | 0.0171 | 0.0374 | -0.5 toward | reported |
| reported r8_ot_rate | 0.0680 | 0.0363 | 0.0328 | -0.0035 | 0.0047 | -0.7 toward | reported |
| reported r8_slope_gameq | 1.0000 | 1.0434 | 1.0339 | -0.0095 | 0.0385 | 0.2 toward | reported |
| reported r8_count_cal_slope | 1.0000 | 0.7168 | 0.7414 | 0.0245 | 0.0585 | 0.4 toward | reported |
| reported r8_g1_count_minus_v4 | 0.0000 | 0.1827 | 0.1435 | -0.0392 | 0.0993 | 0.4 toward | reported |
| reported pergame_abs_poss_err | 0.0000 | 7.3804 | 7.3161 | -0.0643 | 0.1891 | 0.3 toward | reported |
| reported MD6_margin_sd_ratio | 1.0000 | 1.0087 | 1.0106 | 0.0018 | 0.0227 | -0.1 toward | reported |
| reported pace share of margin variance (4a) | - | 0.0019 | 0.0018 | -0.0002 | 0.0020 | - | reported |

Seed-count check: D2_poss_on_env 2xdraw SD 0.0062 vs |control - actual| 0.1364 -> resolved; D2_total_on_env 2xdraw SD 0.0207 vs |control - actual| 0.2610 -> resolved; H1_poss_on_env 2xdraw SD 0.0108 vs |control - actual| 0.0506 -> resolved

**env quintile (edges over all 5,710 fold-2 games)**: regulation game possessions (both teams) and final total; gap = real - sim; U = UNDERPOWERED (< 150 games)

| q | games | real poss | sim poss ctrl / E1 | poss gap ctrl / E1 (floor) | real total | sim total ctrl / E1 | total gap ctrl / E1 (floor) |
|---|---:|---:|---|---|---:|---|---|
| Q1 U | 88 | 135.60 | 137.37 / 134.96 | -1.77 / +0.65 (0.20) | 139.90 | 141.48 / 138.92 | -1.58 / +0.98 (0.42) |
| Q2 U | 95 | 134.89 | 136.92 / 136.13 | -2.02 / -1.23 (0.50) | 145.31 | 144.25 / 143.24 | +1.07 / +2.07 (0.48) |
| Q3 U | 93 | 136.73 | 137.47 / 137.42 | -0.74 / -0.69 (0.31) | 144.82 | 146.39 / 146.32 | -1.57 / -1.50 (0.46) |
| Q4 U | 102 | 136.63 | 137.14 / 137.93 | -0.52 / -1.30 (0.38) | 149.16 | 147.71 / 148.64 | +1.44 / +0.52 (0.73) |
| Q5 U | 97 | 136.06 | 136.15 / 138.23 | -0.09 / -2.17 (0.44) | 152.64 | 149.79 / 152.45 | +2.85 / +0.19 (0.82) |

**home-rating (home net) quintile**: regulation game possessions (both teams) and final total; gap = real - sim; U = UNDERPOWERED (< 150 games)

| q | games | real poss | sim poss ctrl / E1 | poss gap ctrl / E1 (floor) | real total | sim total ctrl / E1 | total gap ctrl / E1 (floor) |
|---|---:|---:|---|---|---:|---|---|
| Q1 U | 89 | 136.93 | 137.02 / 136.67 | -0.09 / +0.26 (0.40) | 147.17 | 144.15 / 143.69 | +3.03 / +3.48 (0.47) |
| Q2 U | 98 | 135.51 | 137.93 / 137.93 | -2.42 / -2.42 (0.35) | 144.53 | 148.08 / 148.30 | -3.55 / -3.77 (0.52) |
| Q3 U | 98 | 136.74 | 137.09 / 137.07 | -0.35 / -0.32 (0.42) | 146.26 | 145.61 / 145.53 | +0.65 / +0.74 (1.00) |
| Q4 U | 95 | 135.33 | 136.55 / 136.53 | -1.23 / -1.20 (0.43) | 145.97 | 145.58 / 145.63 | +0.39 / +0.34 (0.62) |
| Q5 U | 95 | 135.52 | 136.38 / 136.65 | -0.87 / -1.13 (0.32) | 148.85 | 146.54 / 147.03 | +2.32 / +1.82 (0.57) |

**Per team** (offence-team quintile of mean x_off = off_c(offence) + def_c(opponent); regulation possessions per team-game = per 40 min)

| q | teams | team-games | mean x_off | real | ctrl | E1 | right-sign share (E1 - ctrl vs real - ctrl) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 70 | 173 | -9.00 | 67.80 | 68.70 | 68.12 | 0.643 |
| Q2 | 69 | 208 | -2.79 | 68.24 | 68.65 | 68.49 | 0.478 |
| Q3 | 69 | 191 | -0.12 | 68.01 | 68.24 | 68.29 | 0.551 |
| Q4 | 69 | 199 | +3.33 | 68.18 | 68.94 | 69.03 | 0.507 |
| Q5 | 69 | 179 | +9.28 | 67.70 | 67.92 | 68.46 | 0.464 |
| ALL | 346 | 950 | | | | | 0.529 (|err| shrinks for 0.506) |

**Per possession type** (regulation seconds per possession by start type; real = clock design rows of the same games)

| start type | real rows | games | real s | ctrl s | E1 s | E1 - ctrl (floor) |
|---|---:|---:|---:|---:|---:|---|
| period_start | 886 | 444 | 20.458 | 20.298 | 20.337 | +0.039 (0.083) |
| DREB | 21146 | 444 | 14.215 | 14.259 | 14.261 | +0.002 (0.023) |
| TOV | 10420 | 444 | 14.674 | 14.846 | 14.839 | -0.007 (0.027) |
| made_FG | 21264 | 444 | 21.987 | 21.921 | 21.922 | +0.001 (0.044) |
| made_FT | 6459 | 444 | 18.747 | 18.634 | 18.618 | -0.016 (0.039) |
| other U | 232 | 149 | 1.379 | 1.709 | 1.924 | +0.215 (0.122) |

**Per game**: mean |sim mean regulation possessions - real| over pbp-matched games: control 7.380, E1 7.316 (delta -0.064, floor 0.189).

Notes on the table:
- Both primaries now reach or pass the actual. The game-possession slope on env goes from -0.012 to +0.184, against the
  sample's actual of +0.124 (full-size actual +0.152). It now overshoots by about half of the old shortfall. The game-total
  slope goes from 0.407 to 0.637, against an actual of 0.668.
- **First-half env slope: FLAG.** Control -0.010, E1 +0.084, actual +0.040. E1 overshoots, and the gain in |distance to actual|
  (0.007) is within one floor (0.011). It passes the registered direction test literally, but it is not a resolved move toward
  the actual.
- The |spread| Q3 and Q5 margin ratios move away from 1 by 0.040 and 0.041. That is inside their floors (0.073 and 0.048, the
  binding band) but outside the literal 0.009 band. On 500 games x 25 seeds the quintile noise is far larger (draw SD
  0.023-0.036), so the 0.009 reading only means something at full size.
- The total SD ratio is 0.948 (control) and 0.951 (E1) on this sample. Both are outside the full-size baseline interval
  [0.974, 1.014], and the ratio does not move (+0.003).
- Per env quintile (every cell UNDERPOWERED, 88-102 games): the total gap goes from -1.58 ... +2.85 (Q1 ... Q5) to
  +0.98 ... +0.19. The possession gap goes from -1.77 ... -0.09 to +0.65 ... -2.17, so E1 now overshoots at both ends.
- Per team: the right-sign share is 0.529 over 346 teams. Teams average 2.7 team-games each, so this is noisy.

## 4. Decision rule applied literally (section 43)

Rule: "E1 is a CANDIDATE if both primaries clear their floors and every hard veto passes."
- Primaries: +11.5 and +9.6 floors toward the actual.
- Hard vetoes (the 41.A.6 list, the four dispersion gates, the H1 env slope, the overall regulation margin SD ratio): none fails.

**Outcome: CANDIDATE.**

The next registered steps are parity v10 on a clean `src/` (done, PASS) and the full-size 5,710 x 200 paired read against served
v3 with four control draws. After that the PM rules on adoption, with an explicit freeze lift. The full-size read is QUEUED.

## 5. The queued full-size read (ready to run)

- Runs, via `scripts/chain_clk10_env_full_v2.sh` (drop its initial wait-for-PID):
  - served-v3 control, 5,710 x 200 (seeds 0-199);
  - E1, 5,710 x 200 (paired seeds);
  - four control draws, 5,710 x 50, at offsets 1000-4000;
  - trajectories for the first 50 seeds of each run.
- Post-processing: `scripts/chain_clk10_env_full_post_v1.sh`. It runs the chunked extract `diag_c4_extract_chunked_v1.py`
  (validated identical to v1 on the sample) and then dispersion v2 with `--seeds 50`.
- Grader: `scripts/grade_clock_env_full_v1.py`, committed in `f0161f1` and amended in `f9b7ea0`. The amendment was made before any
  full-size E1 output existed: it uses 50-seed draws and scales the draw SD by sqrt(50/200) for the lines read on 200 seeds.
- Measured local cost: about 120 game-seeds/s with 16 workers, i.e. **about 2.6 h per 200-seed run**, not ~1 h. The whole read is
  about 2 x 2.6 h + 4 x 0.65 h = 7.8 h plus post-processing. A box run would be faster.
- The stopped control run left a partial `results/trajectories/clk10full_ctrl_o0_s200` and no games.parquet. Rerun it from scratch.

## 6. Parity, suite, artifacts, not done

- Parity v10 (flags off, clean src): PASS bit-identical, twice.
- Suite: 847 passed, 1 skipped (full pytest, 14:07-15:05 EDT, after every src-free change of this round; no file under src/ changed this round). Rerun at 16:16 EDT on the final tree: 847 passed, 1 skipped (295 s).
- Artifacts: the round-9 E1 schedules are unchanged (49 MB each, gitignored, already on HF). The new latent report is a small
  tracked JSON. Loop outputs (`results/clock_env10/`, trajectories) are gitignored under `results/`. Nothing over 20 MB was added to git.
- Not done:
  - the full-size read (PM time limit);
  - E2 / E3 (section 43 tests E1 only).
- Nothing was adopted, no served default changed and the freeze is untouched. Not touched: results/daily, data/processed/ingest,
  the served inputs, PROJECT_STATUS.md, HANDOFF.md.
