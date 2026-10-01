# G9 slope of served stack v2: the team-rate response round (lane A, day 2026-10-01)

Nothing is adopted, served or re-defaulted. No engine module was edited. Every arm is a sub-model retrain of fg_make; nothing was done to sim output.

- **Pre-registration:** `docs/models/fg_make/experiments.md` section 23, plus a pointer in `docs/models/aggregation/experiments.md` section 3. Commit 8442e22 (06:28 EDT), before any arm ran. The only training before that commit was the step-1 diagnostic, which is not an arm.
- **Ledger:** one row in `docs/models/change_ledger.md` section A.
- **Truth and data:** `CBB_TRUTH=verified_v1`. Fold 2 has 5,705 graded games, 5,380 of them lined (ESPN BET close). 2025-26 was not touched.
- **Compute:** 4 single-thread processes at most (trainer `--n-jobs 4`; the harness and grader ran one at a time, never alongside a trainer). No AWS.
- **Session log:** `results/g9_team_response_v1/session_log.txt`.

**Scripts:**
- `diag_g9_team_response_v1.py`: step 1.
- `train_fg_make_v4_par_g9_v1.py`: wrapper around the unchanged `train_fg_make_v4_par_v1.py`.
- `run_g9_fg_arms_v1.sh`: runs every arm.
- `diag_g9_harness_arm_v1.py`: arm harness with a train/serve parity gate.
- `run_g9_harness_arms_v1.sh`: runs the harness for every arm.
- `grade_g9_team_response_v1.py`: one blind grader for all arms.

**Outputs:** `results/g9_team_response_v1/`, mainly `attrib_v2_v1.json` and `grade_v1.json`. Artifacts are in `data/processed/models/fg_make/round_g9/`, which is untracked.

## 0. Verdicts per pre-registered line

| line | number | floor | verdict |
|---|---|---|---|
| Step 1: reproduce the attribution on served v2 | v2 G9 slope 0.948 (close 0.940). Sim-anchored k (realised beta): PO rates +0.035 (0.64), fg_make rates +0.020 (0.65), rebound +0.014 (0.67), ratings -0.019 (1.03), not_harnessed -0.013 | SE of beta: PO 0.08, FG 0.14 | REPRODUCED (last night: 0.57 / 0.57 on the harness alone) |
| Primary: fold-2 harness 1 - slope, arm minus served (0.0880) | aG3R +0.0007 / -0.0012 (seeds 0 / 1); aG3 +0.0060 / +0.0001; aG1 +0.0030 / +0.0105; aG1R +0.0085 / +0.0115; aTfix +0.0116 / +0.0043 | aR's own seed-1 retrain +0.0089, so the floor is >= 0.0089 for every arm; bootstrap SE 0.003-0.004 | NO WINNER. No arm is beyond 2 floors (-0.018 needed). aG1R, aTfix and aG1 point the wrong way, about 1 floor |
| Close lens of the primary | aG3R -0.0032 / -0.0066; aG3 -0.0028 / -0.0081 | aR seed spread 0.0081 | right direction for the two monotone arms, inside the floor |
| Co-primary: team-rate-part beta (design level, F2) | aR: rim 0.83, jump 0.73, three 0.35. aG3R: 0.87 / 0.99 / 1.11. aG3: 0.93 / 1.01 / 0.93 | seed spread 0.03-0.06 | MOVES TO 1 for aG3R and aG3 |
| Co-primary: team-game make slope (F2) | aR 0.83 / 0.79 / 0.73. aG3R 0.80 / 0.81 / 0.73. aG3 0.79 / 0.81 / 0.70 | about 0.01 | DOES NOT MOVE |
| Fold 1 (design level) | aG3R: team-game slope 0.80 -> 0.82, 0.65 -> 0.77, 0.73 -> 0.82; log loss better in all 3 classes | | same sign, but irrelevant without a fold-2 primary win |
| Guards | aG3R and aG3: F2 three log loss worse than aR by +0.000055 to +0.000089, against a guard of 2 x 0.000018. aG1 seed 0: jumper D8 FAIL. Parity: every arm at most 0.045% of shooter-dev rows off by more than 1e-4; team columns exact | | aG3R, aG3 and aG1 each break a guard |
| Responsiveness (offence-rate quintile slope, F2) | aR 0.95 / 1.16 / 1.31; aG3R 0.87 / 0.92 / 1.04; aG3 0.87 / 0.91 / 0.89 | | all inside [0.8, 1.25]; aG3R and aG3 slope with actuals better than served on three |

**Outcome: NO WINNER.** G1 and G3 (rebuilt as `aG1` and `aG3`), and both served-feature arms, are not shippable. No engine flag, closed-loop tap or box request was made, because the brief's step 4 requires an offline winner.

## 1. Step 1: attribution on served stack v2 (`attrib_v2_v1.json`)

**Input check.** The harness inputs are the served v2 arrays (equal) and the v3 event block (sha 228794fc), with the same PO, fg_make, rebound and free_throw artifacts. The five v2 loop changes sit in `not_harnessed`.

**Sim-anchored L1 decomposition** (X is the v2 200-seed mean margin):

| component | k, Y lens (SE) | beta, Y | k, close lens (SE) | beta, close |
|---|---:|---:|---:|---:|
| base (site, date, state) | +0.013 (0.004) | 0.77 | +0.012 (0.001) | 0.80 |
| PO team rates | +0.035 (0.008) | 0.64 | +0.054 (0.003) | 0.44 |
| fg_make team rates | +0.020 (0.008) | 0.65 | +0.026 (0.003) | 0.53 |
| rebound rates | +0.014 (0.005) | 0.67 | +0.023 (0.002) | 0.43 |
| ratings | -0.019 (0.024) | 1.03 | -0.050 (0.007) | 1.07 |
| shooter | +0.003 (0.005) | 0.90 | +0.006 (0.001) | 0.81 |
| not_harnessed (v2 loop changes, MC) | -0.013 (0.003) | 0.46 | -0.011 (0.001) | 0.52 |

**v2 against old S0, paired.**
- The +0.031 slope gain is not_harnessed -0.018 (SE 0.004) plus ratings -0.010 (SE 0.003).
- The sub-model rate components are unchanged (all within 0.002).
- So the v2 stack fixed none of the team-rate excess. Its loop-level changes (mainly the shared shooting latent and chance time) pull the sim's spread back against the harness: sim SD 9.31 against harness 9.64.

**Per possession type: sim team-game slope of realised on the v2 200-seed mean rate** (SE about 0.02-0.04):

| rate | TOV | FT trip | rim share | jump share | 3PA share | rim make | jump make | 3P make | OREB | FT% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 | 0.94 | 1.01 | 1.02 | 1.02 | 0.99 | 0.90 | 0.70 | 0.67 | 0.92 | 0.64 |
| S0 | 0.94 | 0.96 | 1.01 | 1.03 | 0.99 | 0.87 | 0.72 | 0.69 | 0.87 | 0.64 |

- **PO's rates are calibrated at team-game level.** Its margin component is still over-spread (beta 0.64, close 0.44). That is how a style difference turns into points, not a PO rate response.
- **A make-mix coupling test** (realised make residual on PO's predicted rates) found small effects with the wrong sign to explain it. Per 1 SD of predicted TOV, the make residual is -0.006 (three) and -0.004 (rim); per 1 SD of predicted 3PA share, +0.003. No style arm was registered.
- **The over-spread sub-models at their own team-game level are fg_make** (jumper and three, rim mildly) **and free_throw** (FT% 0.64, outside this brief).

**Responsiveness by team prior quintile** (as-of own-rating net, team view):
- Margin across quintiles has slope 0.99 for v2 and 0.96 for S0. Predicted against actual by quintile: -7.04 / -6.88, -2.93 / -2.83, 0.16 / 0.04, 2.91 / 2.51, 6.90 / 7.16.
- Slopes within a quintile are 0.93-0.99.
- So the team level is calibrated, and the excess is game-to-game movement within a team-season.

**Team-view k by team prior quintile** (Y lens):
- PO k is +0.009 / +0.025 / +0.018 / +0.019 / +0.039 for quintiles 0-4 (beta 0.88 / 0.61 / 0.64 / 0.71 / 0.38).
- fg_make k is -0.000 / +0.013 / +0.000 / +0.019 / +0.029.
- The excess concentrates on the strongest teams. Each cell has 2,282 team-games, and the per-cell SE is about 0.02, so the gradient alone is not significant.

**By month** (Y lens; each cell 765-1,422 games, SE of k about 0.015-0.035):

| month | slope | PO k (beta) | fg_make k (beta) |
|---|---:|---|---|
| Nov | 0.93 | +0.043 (0.35) | +0.042 (0.43) |
| Dec | 0.95 | +0.034 | +0.007 |
| Jan | 0.92 | +0.046 | +0.031 |
| Feb | 0.91 | -0.001 | -0.005 |
| Mar | 0.95 | +0.045 | +0.026 |

**By site.**
- Home/away (4,969 games): slope 0.961, PO +0.036, FG +0.015.
- Neutral (736 games): slope 0.846, PO +0.055, FG +0.079 (SE 0.03). The neutral cell is powered but noisy.

## 2. The arms (fold 2 selects, fold 1 confirms)

**Construction.**
- Every arm is the fg_make B1 spec on the S1 monthly schedule, built through the wrapper.
- The reference retrain `aR` seed 0 reproduces the served artifacts exactly: log loss 0.667252 / 0.666385 / 0.637738, and its harness is identical to `harness_S0` (max |ppp diff| 0).
- `aTfix` reproduces last night's Tfix (log loss and harness both exact).

**min_child_samples, chosen by team-season-grouped CV log loss:**
- aG3R: rim 200, jumper 12800, three 12800 (F2).
- aG3: 12800 for all three classes.

**Train/serve parity (checked by the harness before scoring each arm):**
- `off_make_c`, `def_allow_c` and the extra columns are exact.
- `shooter_shrunk_dev_c` has 0.03-0.045% of rows off by more than 1e-4, the same in every stack including the served reference.
- `aG1R`'s attempt counts have no design value for 530-541 of 11,420 (game, side, class) cells. These were served as 0 in the harness, which is a serving approximation, stated here.

**Primary and design-level table, F2.** Seeds 0 / 1. "brate" is the beta of the team-rate part. "tg" is the team-game make slope.

| arm | harness 1 - slope | d vs served (SE) | d close | brate rim / jump / three | tg rim / jump / three | log loss rim / jump / three (x1e-6 vs aR s0) |
|---|---|---|---|---|---|---|
| aR | 0.0880 / 0.0970 | 0 / +0.0089 (0.0026) | 0 / +0.0081 | 0.83 / 0.73 / 0.35 | 0.83 / 0.79 / 0.73 | 0 / 0 / 0; s1: -44 / -112 / -18 |
| aG3R | 0.0887 / 0.0869 | +0.0007 / -0.0012 (0.0033) | -0.0032 / -0.0066 | 0.87 / 0.99 / 1.11 | 0.80 / 0.81 / 0.73 | -34 / -178 / +67 |
| aG1R | 0.0965 / 0.0996 | +0.0085 / +0.0115 | +0.0111 / +0.0127 | 0.91 / 0.73 / 0.45 | 0.83 / 0.77 / 0.70 | -143 / +100 / +16 |
| aTfix | 0.0997 / 0.0923 | +0.0116 / +0.0043 | +0.0096 / +0.0026 | 0.91 / 0.85 / 0.28 | 0.80 / 0.77 / 0.71 | -221 / -336 / -2 |
| aG3 | 0.0940 / 0.0881 | +0.0060 / +0.0001 | -0.0028 / -0.0081 | 0.93 / 1.01 / 0.93 | 0.79 / 0.82 / 0.70 | -67 / -565 / +62 |
| aG1 | 0.0910 / 0.0986 | +0.0030 / +0.0105 | +0.0054 / +0.0114 | 0.94 / 0.84 / 0.36 | 0.82 / 0.73 / 0.71 | -370 / -17 / -14 |

**Fold 1, design level** (seed 0; seed 1 agrees within about 0.03):

| arm | tg rim / jump / three | brate rim / jump / three |
|---|---|---|
| aR | 0.80 / 0.65 / 0.74 | 0.85 / 0.42 / 0.33 |
| aG3R | 0.82 / 0.77 / 0.81 | 0.97 / 0.76 / 0.42 |
| aG3 | 0.81 / 0.79 / 0.80 | 0.97 / 0.87 / 0.66 |
| aG1R | 0.72 / 0.64 / 0.62 | |
| aTfix | 0.80 / 0.68 / 0.71 | |
| aG1 | 0.78 / 0.68 / 0.66 | |

**Segments for aR -> aG3R** (F2 team-game slope, three):
- By month: Nov 0.58 -> 0.56, Dec 0.76 -> 0.76, Jan 0.72 -> 0.70, Feb 0.78 -> 0.73, Mar 0.96 -> 1.05.
- The team-rate beta for three in November: 0.13 -> 0.71.
- By site, three: home/away 0.75 -> 0.73; neutral 0.53 -> 0.67. The neutral cells are 1,400-1,470 team-games and noisy.

## 3. What the round shows

**1. The constraint arms do what they target, but the over-spread moves.** With monotone constraints and heavier leaves (aG3R, aG3), fg_make's response to its team make-rate features becomes calibrated: the three beta goes from 0.35 to 1.1 on F2, and the November beta from 0.13 to 0.71. Yet the team-game make slope and the harness margin slope do not move. The excess moves into the rest of the prediction:
- The no-rate part's beta falls from 0.78-0.83 to 0.64-0.76.
- fg_make's share of the margin variance in the harness rises from 0.058 to 0.106-0.155, with margin beta still 0.4-0.5.

This is the same displacement last night's G2 showed: drop or constrain one team input and the learned team signal reappears through the correlated ones (ratings, site, the other side's rate). **fg_make's game-level over-spread is not owned by its team-rate features. It belongs to the team-level signal of the whole model**: at shot level fg_make passes D8 calibration while its team-game means are over-dispersed (slope 0.73-0.83 on F2, 0.65-0.80 on F1).

**2. The reliability and E3 arms make it worse or leave it unchanged.**
- Attempt counts (aG1R) and E3 posterior variances (aG1) improve rim log loss but steepen the margin.
- Shrunk E3 rates (aTfix) also steepen it, consistent with last night's G9-neutral read.

**3. The harness primary has a large retrain floor.** The served spec retrained under another seed moves the harness slope by 0.0089, which is about 10% of the miss. Any fg_make arm worth the name has to move it by about 0.02, which none of these do.

**4. PO's margin component is the largest single excess** (k 0.035; close-lens beta 0.44), but its own rates are calibrated at team-game level. It is not a PO rate-response defect, and no PO retrain is motivated by this evidence. The mechanism that values style differences in points remains unowned.

## 4. Not run

- Engine flag, closed-loop tap and box request: no offline winner, so step 4 did not trigger.
- PO retrain arms: not motivated (section 1).
- A fold-1 harness: no 2023-24 engine inputs exist.
- Fold 1 grades the design level only.
- The fold-1 `fit_m` m is chosen on 2023-24 for every arm alike (stated in the pre-registration).

## 5. Incidents

- A scratch `cat >` with no input hung a background shell for about 8 minutes (06:21-06:29). It was my own task, and I stopped it. No other lane was touched.
- The fold-1 E3 arms failed on their first launch: the F1 table has no 2024-25 rows. The wrapper now drops seasons after 2024 for F1 before the adapter runs. All six were rerun from 08:25.
- The runner's `rc=` log field reads the exit code of `date`, not of the trainer, so failures were found by grepping for Traceback.

## 6. Recommended next step

A two-stage fg_make, as a sub-model change and not a sim adjustment. Fit the team-game component of the make probability (offence and defence team effects together) as a partially pooled random effect whose shrinkage is estimated walk-forward on the training seasons. Let the GBM model only the within-game part: shooter, state and chance time. Pre-register it with the same harness primary and design-level team-game slope.

Separately:
- PO's margin over-spread needs a points-valuation diagnostic (realised PPP against predicted style at team-game level), not a PO retrain.
- free_throw's team-game FT% slope of 0.64 is the next-largest unowned own-level excess.

## 7. Round 2: two-stage fg_make

This round was tasked by the PM. Pre-registration: `fg_make/experiments.md` section 25, commit 476ca10 (08:55 EDT), committed before any arm ran.

**Run.** 08:56-09:30 EDT, local, 4 processes.
- Trainer: `scripts/train_fg_make_two_stage_v1.py`. Stage A is shared by the three arms of one fold and seed; each fold and seed takes about 7 min.
- Served-shape object: `scripts/fg_two_stage_model_v1.py`.
- Runner: `scripts/run_ts_round_v1.sh`.
- Grader: the same `grade_g9_team_response_v1.py` (`--arms arms_ts.json --out grade_ts_v1.json`).
- Artifacts: `data/processed/models/fg_make/round_ts/`. They are untracked and not uploaded, because there is no winner.

**Train/serve parity: EXACT.**
- The engine-side offsets and league rates were computed from the same as-of stats for all 5,710 games x 2 sides x 3 types, with 0 missing cells.
- They equal the design's per-shot values on every fold-2 test row (max |diff| 0).
- The shooter dev differs on 0.03-0.045% of rows, as in every stack.

**Fitted stage B** (F2; seeds 0 and 1 give the same values):
- tau_off / tau_def (logit scale): rim 0.13 / 0.10-0.13, jumper 0.08-0.10 / 0.08, three 0.06 / 0.04-0.06.
- TS3 prior-season carry rho: 0.75.
- Site terms: b_home +0.036 to +0.059, b_away -0.028 to -0.066.

### 7.1 Verdicts per pre-registered line

| line | aR (s0 / s1) | TS1 | TS2 | TS3 | verdict |
|---|---|---|---|---|---|
| Primary: harness 1 - slope (served 0.0880; floor 0.0089) | 0.0880 / 0.0970 | -0.216 / -0.219 | -0.238 / -0.241 | -0.155 / -0.157 | see 7.2: the slope OVERSHOOTS to 1.16-1.24 |
| Harness margin SD (served 9.65) | 9.65 / 9.72 | 6.96 | 6.84 | 7.38 | under-spread |
| Responsiveness: team-game make slope rim / jump / three (F2), toward 1.0 | 0.83 / 0.79 / 0.73 | 0.81 / 0.63 / 0.82 | 0.85 / 0.61 / 0.85 | 0.80 / 0.67 / 0.85 | jumper moves AWAY by 0.11-0.18 (limit 0.03): FAIL |
| Offence-quintile slope (F2) | 0.95 / 1.16 / 1.31 | 1.00 / 1.12 / 1.26 | 0.99 / 0.89 / 1.23 | 0.81 / 0.96 / 1.06 | pass |
| Guard: jumper log loss, at most aR + 0.25 x (stage A - aR) = 0.667048 | 0.666385 | 0.668218 / 0.668138 | 0.668221 / 0.668144 | 0.667890 / 0.667818 | FAIL, all arms |
| Guard: rim log loss (at most 0.667937) and three (at most 0.637818) | | 0.66787 / 0.63776-8 | 0.66783 / 0.63777-9 | 0.66738 / 0.63768-70 | pass |
| Guard: D8 calibration, both seeds | pass | jumper FAIL (3.8-3.9 pp); rim FAIL | jumper FAIL (3.7-4.0 pp) | jumper FAIL (3.6-3.8 pp) | FAIL, all arms |
| Guard: level, predicted - actual within 0.5 pp | | jumper -0.10, three +0.27 to +0.31 pp | | | pass |
| Fold 1 (design level), team-game slope | 0.80 / 0.65 / 0.74 | 0.75 / 0.59 / 0.90 | 0.79 / 0.59 / 0.90 | 0.77 / 0.63 / 0.87 | three moves toward 1; jumper and rim move away |

**Outcome: NO WINNER.** All three arms break the jumper log-loss guard, the jumper D8 guard and the jumper responsiveness line, on both seeds and both folds. No engine flag, tap, HF upload or box request was made. `d1001_A_1.md` was NOT written, because step 4 needs an offline winner.

### 7.2 The primary crossed 1.0 (disclosed, not used for the verdict)

The registered primary said "1 - slope, lower is better". It did not anticipate the slope crossing 1.0.
- Every two-stage arm drives the harness slope to 1.16-1.24, so 1 - slope is -0.15 to -0.24.
- That means the margin became UNDER-spread: SD 6.8-7.4 against 9.65.
- Read literally, that is a gain of 17-37 floors. Against the G9 target (slope 1.0) it is a larger miss than served: |1 - slope| is 0.155-0.241 against 0.088.

I do not call it a win. The verdict above rests on the guards, which fail independently. For the PM: future slope primaries should be written as |1 - slope|.

### 7.3 What it shows

1. **Ratings carry real make signal that residual history cannot replace.** This round took the four own-rating columns and the as-of team rates out of fg_make and gave it only a partially pooled effect of its own make residuals. That removes about 28% of the harness margin SD. So the served fg_make's game-level over-spread (sections 1-3) is not "too much team signal" in total. Team strength enters makes legitimately through the ratings, and the excess is a smaller over-response on top of that.
2. **Threes respond as hoped.** The pooled effect calibrates the three-point team-game slope (0.73 -> 0.82-0.85 on F2, 0.74 -> 0.87-0.90 on F1; team-effect beta 1.04-1.19), with shot-level log loss inside the guard. Rim is neutral (TS2: 0.83 -> 0.85).
3. **Jumpers break in stage A itself.** With no team inputs, the trees over-use the state and shooter features. At shot level the deciles are over-spread: the bottom decile predicts 0.30 against 0.34 realised, the top 0.47 against 0.44. The trees also cannot follow the league level from `lg_make_asof` used as a split variable (by month: Nov -1.4 pp, Mar +2.4 pp). This is a stage-A specification defect of this round (the league level should enter as an offset), not evidence about the pooled term.
4. **Variance composition: nothing is double-counted.**
   - Team-game residual variance above binomial (x1e-4, F2 rim / jump / three): aR 20.2 / 19.8 / 3.5; TS arms 21-23 / 22-24 / 3.2-3.5. The predictable term explains LESS than served, so it absorbs none of the zero-mean latents' variance.
   - Two-team shared residual covariance (G3's fitting basis, x1e-4): aR 10.6 / 14.5 / 1.4; TS arms 9.1-9.7 / 13.9-14.5 / 0.8-1.1. These are within about 10%, so G3's Sigma would need at most a small refit under such an arm.

### 7.4 Recommended next step (not run; needs its own pre-registration)

A two-stage arm `TS-R`:
- Stage B: the pooled residual effect (TS2) PLUS linear own-rating terms relative to the snapshot league mean (the offence's offensive rating, the defence's defensive rating). Fit the coefficients walk-forward on training seasons.
- Stage A: the league level enters as a logit OFFSET (`logit(lg_make_asof)`), not as a tree feature.
- Primary written as |1 - slope|.

The three-point result suggests the pooled term is the right tool for the noisy shot type, and the rating terms keep the strength signal the margin needs.

## 8. Round 3: TS-R (PM-tasked; the last round of the series)

Pre-registration: `fg_make/experiments.md` section 27, commit 5518059 (09:30 EDT), committed before any arm ran.

**Run.** 09:30-09:47 EDT, local, 4 processes.
- Trainer: `scripts/train_fg_make_two_stage_v2.py`. Served-shape object: `scripts/fg_two_stage_model_v2.py`. Runner: `scripts/run_tsr_round_v1.sh`.
- Grading: the same grader (`--arms arms_tsr.json --out grade_tsr_v1.json`), which now reports |1 - slope| and month-level bias.
- Artifacts: `data/processed/models/fg_make/round_tsr/`, untracked and not uploaded.

**Parity: EXACT** for every injected column (`fg_team_offset` built from the ENGINE's rating columns, `lg_make_asof`) and every served team column of `aL`.

**Fitted stage B for TSR** (F2, seeds 0 and 1 agree to 3 decimals):

| type | tau_off / tau_def | site home / away | off-rating | def-rating |
|---|---|---|---|---|
| rim | 0.10 / 0.10 | +0.046 / -0.062 | +0.0165 | +0.0105 |
| jumper | 0.08 / 0.06 | +0.041 / -0.022 | +0.0063 | +0.0122 |
| three | 0.04 / 0.00 | +0.028 / -0.023 | +0.0064 | +0.0088 |

### 8.1 Verdicts per pre-registered line

| line | served aR (s0 / s1) | aL (s0 / s1) | TSR (s0 / s1) | verdict |
|---|---|---|---|---|
| Primary: harness abs(1 - slope), F2 | 0.0880 / 0.0970 | 0.0867 / 0.0862 | **0.0038 / 0.0019** | TSR -0.084 / -0.086. Floor = max(0.0089, seed spread 0.0019, paired bootstrap SE 0.018-0.019) = 0.019, so **4.4 floors**. aL is inside the floor |
| Close lens abs(1 - slope) | 0.0957 / 0.1038 | 0.0968 / 0.0941 | 0.0140 / 0.0119 | TSR agrees |
| Harness margin SD (served 9.65) | 9.65 / 9.72 | 9.65 / 9.63 | 8.87 / 8.85 | the slope is NOT reached by collapsing spread (-8%; round 2 was -28%) |
| Team-game make slope rim / jump / three, F2 (decision line) | 0.83 / 0.79 / 0.73 | 0.82-0.83 / 0.82-0.83 / 0.73 | **0.85 / 0.97-0.98 / 0.89-0.90** | TSR: all three move toward 1. PASS |
| Offence-quintile slope (decision line) | 0.95 / 1.16 / 1.31 | 0.94-0.95 / 1.12-1.18 / 1.30 | 0.90 / 1.04 / 1.21 | PASS |
| Month-level calibration: every powered month within +-1.0 pp, every type (decision line) | jumper Mar +1.33 (served fails too) | jumper Mar +1.41 / +1.43 | **jumper Mar +1.52 / +1.50**; every other powered cell within +-0.6 pp; round 2's Nov drift is gone (jumper Nov +0.17) | **FAIL** (one cell, jumpers in March) |
| Guard: log loss rim (at most 0.667952) / jumper (at most 0.666723) / three | | fallback bounds: pass | 0.667293-0.667329 / 0.666265-0.666397 / 0.637525-0.637530 (three better than served by 210e-6) | PASS |
| Guard: D8 calibration, both seeds | | pass | pass | PASS |
| Guard: overall level within +-0.5 pp | | jumper +0.48 | rim +0.01, **jumper +0.52 / +0.51**, three +0.20 | **FAIL** (jumpers, by 0.01-0.02 pp) |
| Fold 1, design-level team-game slope | 0.80 / 0.65 / 0.74 | 0.80 / 0.61-0.63 / 0.69-0.74 | 0.85 / 0.78-0.79 / 0.84-0.85 | TSR: same direction in all three types. PASS |

**Outcome by the pre-registered rule: NO WINNER.**
- TSR wins the primary by 4.4 floors and passes every responsiveness, log-loss, D8, fold-1 and parity line.
- It breaks two jumper level lines. The month line fails in one cell: March, +1.5 pp, where the served model is itself +1.33 pp and `aL` +1.4 pp. The overall-level guard fails at +0.52 against 0.50.
- These lines were fixed before the run and are not replaced after it. So no flag, tap, HF upload or `d1001_A_1.md` was produced.
- `aL` (served trees plus the league offset only) is NO WINNER: its primary is inside the floor and it fails the same March cell.

### 8.2 Variance composition (no double counting)

Team-game residual variance above binomial (x1e-4, F2 rim / jump / three):

| | aR | TSR |
|---|---|---|
| F2 | 20.2 / 19.8 / 3.5 | 20.0 / 16.5 / 2.4 |
| F1 three | 0.56 | -0.4 |

- TSR's predictable term explains more of the jumper and three team-game variation, and what it leaves stays positive on F2.
- On F1 three the excess is -0.4e-4, i.e. zero within noise: three-point team-game variance is almost all binomial, and aL reads 0.12.
- Two-team shared residual covariance (G3's basis): TSR 11.4 / 13.9 / 1.3 against served 10.6 / 14.5 / 1.4. G3's Sigma basis is essentially unchanged.
- Lane B's random form latent sits in the remaining positive excess.

### 8.3 What the three rounds establish about fg_make's team signal

1. **The served fg_make's team-game over-spread is not owned by any one team input** (round 1).
   - Constraining the response to the team make-rate features (monotone, heavier leaves) calibrates that feature's beta. The excess then reappears through the correlated inputs (ratings, site), and the team-game slope and margin slope do not move.
   - Reliability features and E3 shrinkage make it worse.
2. **Ratings carry genuine make signal** (round 2). Removing them and keeping only a partially pooled residual history removes 28% of the margin spread and overshoots the slope to 1.16-1.24.
3. **The excess sits in how the trees learn the team-level effect, not in how much team information there is** (round 3).
   - The same information was given a parametric, walk-forward-fitted team stage: league-centred rating terms, a partially pooled residual effect, home/away/neutral, and the league level as a logit offset.
   - That stage calibrates the team-game make slopes (jumper 0.79 -> 0.98, three 0.73 -> 0.89, rim 0.83 -> 0.85 on F2; the same direction on F1).
   - It brings the deterministic harness margin slope from 0.912 to 0.996-0.998 with 8% less spread.
   - It costs no shot-level information: log loss is within guard, and threes are better.
   - The league-offset repair alone (`aL`) does nothing for the slope, so the gain is the two-stage team structure.
4. **What still blocks TSR is a jumper level defect, not the team stage.**
   - Jumpers are over-predicted by +0.5 pp overall and +1.5 pp in March.
   - The served model shows the same March cell (+1.33), and so does `aL` (+1.4). It sits in the within-game trees' league-level handling, shared by every arm of this series.
5. **Caution for any future closed loop.** The served sim's slope (0.948) sits above its harness slope (0.912), because the v2 loop changes pull spread back. A harness slope of about 1.0 could therefore land the sim slope slightly above 1.0. A full-size read would have to be read on |1 - slope| and margin SD, not slope alone.

**The series stops here, as instructed.** Whether the two jumper level breaks (0.02 pp beyond the level guard; one month cell the served model also fails) should disqualify TSR is a ruling for the PM.

If it is ruled through, step 4 needs about 1.5 h:
- the `ENGINE_FG_MAKE` value plus builder columns `fg_team_offset__k` and `lg_make_asof__k`;
- parity v9 on a clean `src/` tree;
- a local tap;
- an HF bulk key;
- `d1001_A_1.md`.

The box accepts requests until 14:30.
