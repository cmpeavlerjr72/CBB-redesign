# Foul round 8: corrected foul state + shared game-level whistle (possession-outcome round 8) -- 2026-09-30

Lane A. **NOTHING IS ADOPTED AND NO DEFAULT IS CHANGED.** The PM decides.
Pre-registration `docs/models/possession_outcome/experiments.md` section 22 (commit `8be4d89`,
before any fit), Amendment A section 23 (`110b3e3`, before any round-8 fit), results section 24.
Wall clock 12:36-13:25 EDT.

## 0. Verdict

| arm | accrual / trips / whistle | FTA/FGA (0.32955) | floors toward | FT-rate corr (target +0.207) | H1 / H2 (0.2367 / 0.4126) | V1 occ. max gap | V2 slope | V3 | V4 SD ratio | eligible |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| `CL0` served | const / PO / none | 0.31953 | 0 | +0.003 (SE 0.010) | 0.2540 / 0.3789 | 13.1 | 0.958 | -- | 0.821 | ref |
| `R8a` (= r7 `CL2a`) | A2 / PO / none | 0.31983 | +0.17 | +0.029 | 0.2428 / 0.3917 | 7.7 pass | 0.994 pass | G5 total -4.6 fl **FAIL** | 0.821 pass | no |
| `R8aS` | A2 / PO / shared | 0.31984 | +0.17 | +0.064 | 0.2429 / 0.3918 | 7.9 pass | 0.972 pass | G5 total -12.7 fl **FAIL** | 0.805 pass | no |
| `R8b` (= r7 `CL2`) | A2 / T2c / none | 0.33488 | +2.63 | +0.105 | 0.2506 / 0.4175 | 6.2 pass | 0.846 pass* | G5 total -13.9 fl **FAIL** | 0.786 pass* | no |
| `R8bS` | A2 / T2c / shared | 0.33489 | +2.63 | **+0.146** | 0.2508 / 0.4171 | 6.3 pass | 0.951 pass | G5 total -8.7 fl **FAIL** | 0.803 pass | no |
| `R8bU` control | A2 / T2c / unshared | 0.33445 | +2.88 | +0.114 | 0.2509 / 0.4163 | 6.4 pass | 0.929 pass | G5 total -7.6 fl **FAIL** | 0.808 pass | no |

\* Under round 8's noise bands (V2: drop <= max(0.05, 2 x paired bootstrap SE 0.090); V4: drop <= 2 x SE 0.024).
Round 7 read `CL2` as failing V2 and V4 on the strict thresholds.

**No arm is eligible: every arm still fails V3 on the G5 total SD ratio.** The home/away score
correlation, G1 mean and SD, the G5 margin ratio and the G9 bias are within the pre-registered
floor limits for every arm (worst: R8bS G5 margin ratio -1.1 floors, limit 2). Actual score correlation on these games 0.181; sim
0.097-0.112 (reference 0.106; floor 0.0115).

## 1. The corrected state and its verification (22.1, 23)

`scripts/build_foul_accrual_design_v3.py` -> `round6/foul_accrual_poss_v3.parquet` +
`rowless_trips_v3.parquet` (siblings; nothing served overwritten, no consumer switched).
Verifier `scripts/diag_foul_state_v3_verify_v1.py` -> `results/foul_joint/state_v3_verify.json`.

**Reconciliation with the box (per team-game, `fouls`):**

| season | team-games | box mean | round-6/7 replay log gap | **pbp counter gap (mean / MAE / exact)** | v3 (+ rowless) gap (mean / MAE / exact) |
|---|---:|---:|---:|---|---|
| 2022 | 10,564 | 16.77 | +1.46 | +0.07 / 0.17 / 86.7% | -0.02 / 0.21 / 83.3% |
| 2023 | 11,080 | 17.01 | +1.49 | +0.10 / 0.16 / 86.5% | -0.01 / 0.22 / 81.6% |
| 2024 | 11,102 | 17.01 | +1.63 | +0.12 / 0.15 / 86.8% | +0.03 / 0.19 / 83.8% |
| 2025 | 11,180 | 16.96 | +1.60 | +0.16 / 0.18 / 85.6% | +0.07 / 0.20 / 83.2% |

Raw `PersonalFoul` rows vs box, 2025: MAE 0.18, 85.6% exact.

- **The 1.60 gap I reported this morning was an artifact of my replay log, not a defect in the
  pbp state.** `_handle_fga` adds an and-one foul to the counter directly, so the replay, which wraps
  `_handle_foul`, never logged it. The segmenter's counter is verified against the box.
- **Round-7 finding 7 and E6 are retracted.** The "2.26% trip without a foul row" were and-ones
  (0.021/poss).
- True rowless trips number only 964-1,268 per season. Adding them makes box agreement worse, so
  v3 is **NOT VERIFIED** (pre-registered criterion).
- The corrected state used below is **v2**: the counter minus the possession's own pre-open fouls.
- One consequence for the round-7 decomposition tables: actual total and non-trip fouls per
  possession were understated by ~0.021. True values are ~0.245 and ~0.092; targets and features
  are unaffected.

**Bonus-state agreement by game minute (2025):** v3 matches the v2 state on 99.6-100% of
possessions and the labelled flag on 96.3-99.99%. The worst minutes are 30-34 (96.3%) and 15-19
(97.0%). There the label is in bonus 2-3.5 pp more often, from the possession's own pre-open foul.

**Rule check** (a missed single-FT trip that is not an and-one should be a one-and-one front end,
so the prior count should be >= 6; 25,684 first chances): label 85.6%, v2 79.1%, v3 79.5%. Even
v2/v3 leave about 20% of these trips below 6 fouls. The rule check cannot separate the count from
the trip-classification rules; reported, not resolved.

## 2. Train/serve skew trace (read-only)

| served model | state feature | how it is built in training | own pre-open foul included? | how the engine feeds it | skew |
|---|---|---|---|---|---|
| possession_outcome `first`/`cont` (`round2_s1`) | `in_bonus` | `off_in_bonus` at possession open (`possessions.py` 557/1122) | **yes**; `cont` keeps the open value | `st.in_bonus()` re-read every chance (`loop.py` ~405) | **yes** |
| clock `v5b_glat_pmean` (v3c base) | none (`in_bonus` is passed in, but no cell key uses it) | -- | n.a. | -- | no |
| fg_make `round4_B1` | `in_bonus` | `event_stream.in_bonus(fouls_opp_prior)` at the FGA row | n.a. (per event) | per chance, before silent fouls are added | **partial**: training counts earlier non-shooting fouls in the possession; the engine adds them at possession end |
| rebound `s1_weekly` | `in_bonus` | same event-level count at the miss row | n.a. | after trip and and-one fouls are added | **partial** (FGA misses) |
| free_throw `s1_conf_aligned` | `in_bonus` | trip foul class (one-and-one / double bonus) from the prior count | n.a. | `st.in_bonus()` after the trip's own foul is added (`loop.py` ~503/703) | **yes**: a 7th-foul shooting trip reads 1 at serve, 0 in training |
| late_game `clk_D` (off by default) | `bonus_code` | possessions-table `in_bonus` | **yes** | clock frame at possession start | **yes** |
| late_game `ev_BL3` / `ev_L0S0` (off by default) | `in_bonus`, `in_double_bonus` | round-2 design / possessions_v2 | **yes** | `_state_block` | **yes** |

**Corrected-state retrain for tonight's AWS session: ready, and no trainer edit is needed.**
- Table: `data/processed/models/possession_outcome/round8/in_bonus_overlay_v2state_v1.parquet`
  (2.2 MB, force-added in `08cd157` because `round*/` is gitignored; builder
  `scripts/build_po_in_bonus_overlay_v1.py`).
- Keys `game_id, poss_index, chance_number`; column `in_bonus` = (v2 open count + the defence's
  trip fouls on earlier chances) >= 6. It covers all 3,038,628 design rows, and changes 48,169
  rows (1.59%; 47,949 go 1 -> 0).
- Add to `train_possession_outcome_s1_par_v1.py`:
  `--feature-table <that path> --overlay-keys game_id,poss_index,chance_number --overlay-cols in_bonus`
  plus its own `--out-root` (`train_par_common_v1` asserts unique keys and full coverage).

## 3. Offline

- Under Amendment A, round 8's `A3` and `T3c` are round 7's `A2` and `T2c` (same v2 state, same
  spec, same fold-2 window).
- Their blind-graded results stand (section 21): `A2` +18.1 floors (fold 1 +17.5); `T2c` +8.9
  (fold 1 +7.9). The labelled-state twins lose: `A2lab` -10.6, `T2lab` -28.0.
- **Overshoot check:** the pre-open correction was already inside round-7 `CL2`, and it still
  overshoots the first half (H1 0.2506 vs 0.2367). The corrected count does not remove the
  overshoot, so it is not a label artifact.

**Whistle variance, FITTED on training seasons** (`scripts/exp_foul_whistle_var_v1.py`,
`results/foul_joint/whistle_var_v1.json`). Per team-game, `r = W/E - 1` with `W` = non-trip
events + FT trips against the team and `E` from the fold's own `A2` + `T2c`:

| fold | train c = cov(r_h, r_a) (SE) | test c (SE) | test within 2 SE | train var(r) - Poisson part |
|---|---:|---:|---|---:|
| F1 | 0.00376 (0.00097) | 0.00363 (0.00099) | yes | 0.00491 |
| F2 (served) | **0.00422 (0.00078)** | 0.00476 (0.00105) | yes | 0.00553 |

- Served `s^2 = ln(1 + c) = 0.004207`, so `s = 0.065`: a 6.5% SD multiplier on foul odds, the same
  for both teams in a game.
- The team-level extra-Poisson variance (0.0055) is almost all shared (0.0042).
- Sensitivity (regulation outside the final 2:00 only): c = 0.0066 / 0.0056 (F2 / F1).

## 4. Closed loop (500 x 25, the `po4b_R_s25` games and seeds)

**Parity (HEAD `6034ec0`, which includes other lanes' default-off engine hooks):** all bit-identical.
- `fj_PARITY8_s25` vs `po4b_R_s25`: games 28/28 and players 16/16 columns.
- v6 window digest: PASS.
- `R8a` vs round-7 `CL2a`: games and players. This also proves that turning the latent off
  reproduces the no-latent arm, and re-validates the reuse of `CL2` as `R8b` (smoke run: bit-identical).

**Shared vs unshared (the pre-registered contrast):**
- The shared latent raises the two-team FT-rate correlation:
  - `R8a` +0.029 -> `R8aS` +0.064 (+0.034, paired SE 0.011);
  - `R8b` +0.105 -> `R8bS` +0.146 (+0.041, SE 0.009).
- The unshared control at the same marginal variance stays near the no-latent arm:
  `R8bU` +0.114 (+0.010, SE 0.009).
- Within-game FT-rate SD (home / away) is about equal for shared and unshared: 0.1461/0.1377 vs
  0.1445/0.1369. So the control reproduces the marginal dispersion but not the correlation, as
  designed.
- `T2c` itself contributes most of the correlation (+0.003 -> +0.105), because it conditions on the
  foul differential between the two teams.
- **Target +0.207 is not reached** (best 0.146, 6.5 SE short). The fitted `s` is small, and it is not
  tuned toward the gate.

**V3 (G5 total SD ratio, target 1.0, reference 0.8835, floor 0.00105):**

| arm | G5 total ratio | floors vs reference |
|---|---:|---:|
| `R8a` | 0.8787 | -4.6 |
| `R8aS` | 0.8702 | -12.7 |
| `CL2` (= `R8b`) | 0.8689 | -13.9 |
| `R8bS` | 0.8743 | -8.7 |
| `R8bU` | 0.8756 | -7.6 |

- Within-game total-points SD: reference 15.84; `R8bS` 15.67 vs `R8bU` 15.76.
- **The shared whistle does not add total-points variance.** A whistle-driven FT trip mostly
  replaces a shot attempt, so the points covariance it creates is small. The total-variance gap is
  not this object's to close.
- The score correlation does not regress (`R8bS` 0.1082, +0.2 floors toward 0.181).

**Other lines:**
- Occupancy max gap: 6.2-7.9 pp for all arms, against 13.1 served.
- Team FT slope: `R8bS` restores it to 0.951, from `CL2`'s 0.846.
- Final 2:00 FTA/FGA: 0.828-0.861 (actual 0.937).
- OREB%: 0.2828-0.2835.

## 5. Reading (for the PM)

1. The foul-state label defect is real but narrow. It is the possession's own pre-open foul,
   worth 2-3.5 pp of bonus occupancy.
   - It is fixed in the arms, and an overlay for a possession-outcome retrain is ready.
   - The pbp counter itself is sound; this morning's box-gap alarm was mine.
2. A shared whistle is present in the data: the team-level extra-Poisson variance is almost
   entirely shared across the two teams.
   - Wired as fitted, it moves the correlation in the right direction and the unshared control
     does not.
   - Its fitted size reaches 0.146 of the 0.207 target.
3. The V3 blocker (G5 total SD) is not closed by the whistle. The round-7 foul arms narrow G5
   total because the served over-dispersed foul count carried points variance. The shared
   whistle recovers part of that (-13.9 -> -8.7 floors), but not all.
4. The first-half overshoot of `T2c` is not a label artifact. Its owner is still open: minutes
   10-19 are still over-occupied by 4-6 pp.

## 6. Caveats and what was not done

- The 500-game sample contains one unplayed game (the `verified_finals` opt-in exists; default
  unchanged).
- The served engine inputs carry same-game leaks that are being rebuilt tonight as v3 inputs.
  Paired deltas stand; absolute levels are provisional.
- The whistle expectation `E` is in-sample on the training seasons (the pre-registered estimator).
  Its test-season check passes on both folds.
- Not run:
  - a whistle on the split-attribution (`DO2`) arms;
  - the fit-window variance as a served alternative;
  - a v3-state refit (v3 failed verification).

## 7. Resume commands

    export PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe scripts/build_foul_accrual_design_v3.py          # v2 + v3 tables, 30 s
    .venv/Scripts/python.exe scripts/diag_foul_state_v3_verify_v1.py
    .venv/Scripts/python.exe scripts/train_foul_joint_v1.py --n-jobs 6         # round-7 fits (A2, T2c)
    .venv/Scripts/python.exe scripts/build_foul_joint_lut_v1.py
    .venv/Scripts/python.exe scripts/exp_foul_whistle_var_v1.py                # writes round7/whistle_var_v1.json
    .venv/Scripts/python.exe scripts/build_po_in_bonus_overlay_v1.py
    bash results/foul_joint/run_r8.sh                                          # parity + R8a, R8aS, R8bS, R8bU
    .venv/Scripts/python.exe scripts/grade_foul_joint_closed_loop_v2.py fj_R_s25 fj_R8a_s25 fj_R8aS_s25 \
        fj_CL2_s25 fj_R8bS_s25 fj_R8bU_s25

`round7/` holds the LUTs and `whistle_var_v1.json`. It is gitignored and is rebuilt by the
commands above.

## 8. Incidents

1. Another lane's commit `725e8f1` (Stage C team-rate draw) committed `loop.py` from the working
   tree while my three round-8 hunks were in it but uncommitted. Mine had been staged separately
   from HEAD.
   - From `725e8f1` until my `6034ec0`, HEAD's `loop.py` called `fj.init_whistle`, which the
     committed `foul_joint.py` did not yet define.
   - This breaks only runs with `ENGINE_FOUL_JOINT` set; the default path is unaffected.
   - It is resolved in `6034ec0`, with no content lost.
2. I retract my own round-7 finding 7 and E6 (section 1).
3. CRLF churn on `foul_joint.py` was fixed at source before commit (LF restored; `git diff --stat`
   checked).
