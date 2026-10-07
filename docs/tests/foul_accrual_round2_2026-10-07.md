# Foul-accrual round 2: team foul priors, as-of level, and the calendar term on `A2` (2026-10-07)

Pre-registration: `docs/models/possession_outcome/experiments.md` section 34 (commit 0f83dee, pushed before any fit). Results: section 35 there.
Round 1: `docs/tests/foul_accrual_calendar_2026-10-07.md`. 2025-26 not read. Nothing adopted; no served default changed.

## Verdict

**No eligible arm. `A2` (served) stands.**
- `A2t` (team priors only, 3 params) fixes the standing responsiveness defect, offline and in the loop. It fails only O3's calendar-spread line, which a team term is not built to move.
- `A2tnc` (team + n + level + days buckets) passes every F2 gate. It fails F1 O3: the d0-14 H1 rel goes to +5.9% against the control's -0.6%. This is round 1's mechanism again.
- `A2tG` (the GBM ceiling) wins F2 by 8.8 floors but loses F1 by 8.4 floors. It is not servable here and not stable.
- Loop, descriptive, 50 paired seeds, full slates: no arm moves the d0-14 total by more than 0.2 points, so the 200-seed run was not triggered.
- The accrual channel is not the early-total lever. That was predicted in 34.0 and is confirmed here.

## 0. Pass-through (34.0): why +0.026 in-bonus share is worth only +0.15 points

Round-1 `A2dbk` vs served v3, F2 d0-14, per game, both teams:
- H1 in-bonus possessions +1.82 -> H1 bonus trips +0.28, H2 +0.15, shooting trips -0.18 -> **FTA +0.59 -> FTM +0.39**.
- Each bonus trip ends a possession that would otherwise have gone on to a shot: FGA -0.24 -> **field-goal points -0.24**.
- Net **+0.150** points (exact identity). That is **0.25 points per extra FTA**. In this round's arms the same ratio is 0.31-0.32 (F1 `A2tnc` d0-14 +0.61 FTA -> +0.19 pts; F2 d46+ -0.70 FTA -> -0.22 pts).
- So the whole d0-14 FTA/FGA gap (about 2.9 FTA/game) is worth about 0.7-0.9 points of the -5.2 point deficit.
- The early-FTA diag's "74% / 58%" is right in FTA units. Read as points it **overstated the stake**: about 0.55 / 0.45 points, not a large share of the early total.

## 1. Offline (one blind grader `scripts/grade_foul_team_v1.py`; `results/foul_r2/grade_v1.json`)

Primary = log loss of `y_nt`, test fit-window rows (F2 681,101; F1 683,043). Floor = max(control reseed, 2 x game-block bootstrap SE).
Rel = mean p / mean y - 1. Responsiveness = slope of the predicted quintile means on the realised ones, prior-season quintiles, about 70 teams per quintile.

| arm | F2 floors (delta) | F1 floors | F2 H1 rel d0-14 / d46+ | F1 H1 rel d0-14 / d46+ | resp def / off F2 | resp def / off F1 | gates failed |
|---|---:|---:|---|---|---|---|---|
| `A2` control (reseed 6.0e-6 / 6.1e-7) | -- | -- | -3.8% / +10.5% | -0.6% / +13.6% | -0.32 / -0.12 | -0.19 / -0.14 | (reference) |
| `A2t` | +4.56 (4.4e-4) | +3.57 | -3.9% / +10.5% | -0.6% / +14.0% | **0.41 / 0.31** | **0.45 / 0.56** | O3 spread (F2 0.144 vs 0.143; F1 0.146 vs 0.142) |
| `A2tn` | +6.32 (6.9e-4) | +3.56 | -8.5% / +1.7% | -0.7% / +14.0% | 0.39 / 0.29 | 0.45 / 0.54 | F2 O3 d0-14 (-8.5%); F1 O3 spread |
| `A2tnc` | **+6.52** (7.3e-4) | +4.20 | -2.9% / +1.4% | **+5.9%** / +12.4% | 0.39 / 0.30 | 0.45 / 0.54 | **F1 O3 d0-14** |
| `A2tG` ceiling (not eligible) | +8.76 | **-8.44** | +1.1% / +0.2% | +17.5% / +20.6% | 0.30 / 0.25 | 0.38 / 0.59 | F1 confirm, F1 O2, F1 O3 |

Fitted terms (seed 0):
- F2 `A2tnc`: D +0.052, O +0.048 (logit per unit x100 rate; the D SD is 1.9, so about +-10% per SD); D*u -0.010, O*u -0.026; Lv **+1.02**; buckets d0-7 +0.090, d8-14 +0.054.
- F1 `A2tnc`: D +0.055, O +0.053; Lv +0.002; buckets +0.102 / +0.074.
- The F1 level term is inert because the n0 grid chose 3.2e5 on 2023 alone (2023's level was close to 2022's), so `L` is the carried prior. F2 chose n0 = 0 (purely in-season as-of level).
- Seed-7 coefficients agree to the third decimal.

Mechanism:
- The team priors are a real, stable gain on both folds: +4.4e-4 / +4.1e-4. That is +0.55 in carried-floor units, about 3.4 x `A2dbk`.
- The level term only helps when the test season's level actually moves. 2025's fit-window rate is 0.0924 against 0.0990 train, and `A2tn` takes F2 d46+ from +10.5% to +1.7%. F1's 2024 level is 0.1020 against 0.1024 train, and nothing changes there.
- F1's +13.6% late over-prediction is therefore **not a league-level drift**. It sits in the defence count 6+ cell (A2 rel +55% F1 / +40% F2 vs +8% / +5% at 0-5): A2 over-predicts non-trip fouls inside the bonus. That is a separate A2 defect, worth its own item.
- The calendar buckets then over-lift F1's early rows, exactly as `A2dbk` did in round 1.

Segments:
- Site rel (home / away / neutral offence), F2: `A2t` +7.3 / +6.2 / +6.6% vs control +6.7 / +6.9 / +6.0%; `A2tnc` 0.0 / -0.9 / -0.7%. Site is in every arm through `STATE_T`.
- Leak check of the new priors (own pbp, reported): corr(change entering game g, own margin) is -0.020 (D) / +0.010 (O). The honest update (change after game g vs margin) is -0.078 / +0.080, inside the 0.04-0.08 band. Strictly-before is asserted in code. Engine slate features are identical to the training rows on every matched game (max |diff| 0.0).

## 2. Closed loop, DESCRIPTIVE (no eligible arm; s34.4 runs the best-F2 offset arm `A2tnc`; `A2t` is added as an extra descriptive read, not pre-registered)

Setup:
- Arm runs: seeds 0-49, full slates, home box at 8 workers, `scripts/run_foul_cal_tap_v1.py` with `ENGINE_FOUL_TEAM=coef_<arm>_F{1,2}`.
  F1 uses the `d1007_K2F1` overlay and static rotation.
- Control: round 1's served-v3 taps `fcal_F{1,2}_ctrl_s50` (same seeds and code path; no served default changed since).
- Flag-off parity v10: **PASS**, bit-identical.
- Engine offsets equal the offline offsets on every matched row (max |diff| 7e-16).
- Grader: `scripts/grade_foul_cal_closed_loop_v1.py` (`results/foul_r2/loop_F{1,2}_{arm}.json`). SE = paired game-bootstrap SE of the move.
- Actual = verified finals.

| line | F2 v3 | F2 `A2tnc` | F2 `A2t` | F1 v3 | F1 `A2tnc` | F1 `A2t` |
|---|---|---|---|---|---|---|
| total bias d0-14 (SE of move) | -5.16 | -5.17 (0.030) | -5.23 (0.033) | -4.93 | **-4.74** (0.032) | -4.91 (0.033) |
| total bias d15-45 | -0.13 | -0.32 (0.028) | -0.11 | -1.80 | -1.78 | -1.76 |
| total bias d46+ | +0.67 | +0.45 (0.015) | +0.69 (0.014) | -0.65 | **-0.69** (0.014) | -0.66 (0.014) |
| total bias all (G9) | -0.14 | **-0.33** (0.012) | -0.12 | -1.42 | -1.42 | -1.42 |
| H1 in-bonus d0-14 (actual 0.268 / 0.268) | 0.215 | 0.219 | 0.216 | 0.208 | 0.227 | 0.210 |
| H1 in-bonus d46+ (actual 0.182 / 0.187) | 0.206 | 0.187 | 0.207 | 0.209 | 0.208 | 0.211 |
| H2 in-bonus d46+ (actual 0.374 / 0.373) | 0.379 | **0.356** | 0.378 | 0.382 | 0.378 | 0.382 |
| H2 trips per in-bonus poss d46+ (actual 0.234 / 0.232) | 0.237 | 0.240 | 0.238 | 0.234 | 0.235 | 0.234 |
| G4 ft_rate all (actual 0.3295 / 0.3281) | 0.3307 | **0.3252** | 0.3311 | 0.3234 | 0.3238 | 0.3239 |
| G5 total SD / margin SD / corr | 0.963 / 1.031 / 0.170 | 0.963 / 1.028 / 0.170 | 0.962 / 1.030 / 0.170 | 0.952 / 1.024 / 0.189 | 0.950 / 1.024 / 0.188 | 0.951 / 1.025 / 0.189 |

Gate statuses (G1-G9) are identical to the control's in every run.

Loop lines (s32.4):
- **L1**: d0-14 fails in every run (0.04-0.05 short). d46+ passes only F2 `A2tnc` (0.005); F1 `A2tnc` is 0.0204, `A2t` 0.026 / 0.024.
- **L2**:
  - F2 `A2tnc` d0-14 does not move toward zero.
  - F2 `A2t` d0-14 moves away by 2.1 SE (descriptive).
  - **F1 `A2tnc` d46+ moves away by 2.9 SE: FAIL.**
  - F1 `A2t` passes.
- **L3**: no status change.
  - F2 `A2tnc` moves the G9 total bias from -0.14 to -0.33 (about 16 SE) and G4 ft_rate from +0.0012 to -0.0043 vs actual: **FAIL**.
  - The level term pulls H2 below actual: H2 in-bonus d46+ 0.356 vs 0.374, the compensation the diag warned about.
  - `A2t` passes L3 on both folds.

Per-team responsiveness in the loop (`scripts/grade_foul_team_loop_resp_v1.py`, `results/foul_r2/resp_F{1,2}.json`):
- Teams in prior-season quintiles; slope = sim quintile means on actual; offence = FTA/FGA earned, defence = FTA/FGA conceded; about 72 teams per quintile.

| fold | window | view | v3 | `A2t` | `A2tnc` |
|---|---|---|---:|---:|---:|
| F2 | all | offence drawn | 0.45 | **0.70** | 0.69 |
| F2 | all | defence committing | 0.49 | **0.65** | 0.64 |
| F2 | d0-14 | offence drawn | 0.05 | **0.25** | 0.20 |
| F2 | d0-14 | defence committing | 0.28 | **0.45** | 0.43 |
| F1 | all | offence drawn | 0.37 | **0.63** | 0.62 |
| F1 | all | defence committing | 0.51 | **0.67** | 0.67 |
| F1 | d0-14 | offence drawn | 0.27 | **0.52** | 0.45 |
| F1 | d0-14 | defence committing | 0.33 | **0.51** | 0.50 |

d0-14 cells hold teams with at least 60 FGA in the window. Quintile means are noisy (underpowered per cell); the slope direction is consistent on both folds and both views.

Seeds:
- 50 per arm. 200 seeds were not run: the largest d0-14 move is +0.19 points (F1 `A2tnc`), below the 1.0-point trigger.
- AWS: not used, $0.

## 3. Reading for the PM

1. **`A2t` is a standing-rule fix without a level or calendar effect.**
   - Team responsiveness rises from negative to +0.41 / +0.31 offline (F2), and to 0.65-0.70 for the all-season team FT rate in the loop. The served stack is at 0.45-0.49 there.
   - Totals, G4, G5 and G9 are unchanged.
   - Its only failed gate is O3's spread, which it was never meant to fix. Whether that pre-registered gate should bind a responsiveness fix is the PM's call; under the rule as written it is not eligible.
2. **The early-season FTA deficit is not recoverable through accrual in points.** Every arm moves d0-14 totals by at most 0.2 points of -5.
3. **A2 has a separate in-bonus defect.** Inside the bonus (defence count 6+), A2 over-predicts non-trip fouls by +40% / +55%. That, not a league level, carries F1's late over-prediction. It interacts with the H2 trips-per-in-bonus overshoot (diag Block C) and is the next accrual item if one is wanted.
4. **The level term works only when the league level actually moves (F2).** There it over-corrects H2. Not recommended without a fix for the 6+ cell.

Artifacts:
- Tracked: `data/processed/models/possession_outcome/round11team/coef_*.json`, `slate_feats_{2024,2025}.parquet`, `train_meta.json`.
- Local: the preds parquets (regenerable, `scripts/train_foul_team_v1.py`, about 2 minutes on 8 cores) and `results/foul_r2/`.
- Flag `ENGINE_FOUL_TEAM` (`src/cbb_sim/engine/foul_team.py`) is default off.
