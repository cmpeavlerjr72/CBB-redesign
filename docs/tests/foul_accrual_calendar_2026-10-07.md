# Foul-accrual calendar round: days-since-start term in the A2 accrual LUT (2026-10-07)

Pre-registration: `docs/models/possession_outcome/experiments.md` section 32 (commit 3337d13, pushed before any fit). Results: section 33 there.
Owner evidence: `docs/tests/early_fta_rate_diag_2026-10-07.md`. 2025-26 not read. Nothing adopted; no served default changed.

## Verdict

**No eligible arm. `A2` (served) stands.** `A2dbk` (A2 GBM + a days-bucket feature) is the only arm that beats the control on the primary.
It fails pre-registered offline gate O3 on both folds and O1 on F1, so the rule names no winner.
A DESCRIPTIVE paired loop of `A2dbk` was run anyway, on both folds, 50 seeds, full slates (precedent: PO section 31).
It closes about half the first-half (H1) bonus-state gap in d0-14, but is worth only +0.15 points of the d0-14 total deficit.
It fails loop line L1 on both folds (d0-14 is still more than 0.02 from actual) and L2 on F1 (the d46+ total bias moves away from zero by 4.4 paired SE).

**Main finding.** The calendar term in accrual is real but small:

| | F2 | F1 |
|---|---|---|
| d0-14 FTA/FGA gap closed by `A2dbk` | 22% (0.3204 -> 0.3260 vs 0.3454) | 17% (0.3092 -> 0.3138 vs 0.3365) |
| d0-14 total deficit | -5.16 | -4.93 |
| points recovered | +0.15 | +0.16 |

Most of the early total deficit lies outside the foul-accrual channel (TOV, FT make, pace and shot make; see `total_bias_decomp_2026-10-05.md`).

## 1. Offline (one blind grader, `scripts/grade_foul_cal_v1.py`; `results/foul_cal/grade_v1.json`)

Rows: test fit-window possessions on the true state (F2 681,101; F1 683,043). Primary = log loss of `y_nt`.
Floor = max(control reseed 0-vs-7 spread, 2 x game-block bootstrap SE of the paired delta).
The round-7 carried floor of 0.000804 was not used (s32.3); deltas in carried-floor units are listed for transparency.
"Rel" is mean p / mean y - 1.

| arm | F2 LL delta vs A2 (+ = better) | F2 floor | F2 floors | F1 floors | F2 H1 rel d0-14 / d46+ | F1 H1 rel d0-14 / d46+ |
|---|---:|---:|---:|---:|---|---|
| `A2` control | 0 | reseed 6.0e-6 | | | -3.8% / +10.5% | -0.6% / +13.6% |
| `A2dec` (b exp(-d/tau), b 0.12, tau 12 d) | -2.2e-5 | 2.0e-5 | -1.08 | -2.79 | +2.5% / +10.5% | +5.0% / +13.6% |
| `A2decH` (b_H1 0.16, b_H2 0.08, tau 12 d) | -2.2e-5 | 2.0e-5 | -1.09 | -2.47 | +4.9% / +10.5% | +7.2% / +13.7% |
| `A2dbk` (GBM + days bucket) | **+1.30e-4** | 3.7e-5 | **+3.52** | **+3.61** | +5.7% / +8.1% | +8.2% / +11.6% |

`A2dbk` is +0.16 in carried-floor units, so it would also have failed the old floor.

**Offline gates for `A2dbk`:**

| gate | F2 | F1 |
|---|---|---|
| O1 defence prior-quintile slope | -0.323 vs -0.323, pass | -0.199 vs -0.191 with reseed spread 0.001, **FAIL** |
| O2 d46+ log loss | +1.5e-4 / floor 3.6e-5, pass | +2.5e-4 / 3.8e-5, pass |
| O2 d46+ H1 rel | pass | pass |
| O3 d0-14 H1 rel toward zero | -3.8% -> +5.7%, **FAIL** | -0.6% -> +8.2%, **FAIL** |

Why the shape fit does not help the score:
- A2 over-predicts both test seasons late (+10.5% / +13.6% H1 d46+), a downward season trend in the non-trip rate.
- A train-fitted calendar term therefore pushes the early rows, which were near calibration, over.
- `A2dbk`'s log-loss gain comes from d46+ (bucket 4 learns a lower late level), not from the window.

Segments for `A2dbk` vs A2:
- Site: home / away / neutral offence rel, F2 +6.7 / +6.9 / +6.0% -> +6.5 / +6.7 / +5.2%. Site stays in every arm.
- Defence count 0-5 / 6+, F2: +5.0 / +40.3% -> +4.7 / +40.0%.

A2 binned-LUT identity: the seed-0 A2 refit reproduces the served `lut_acc_A2_F2` and the fold-1 `lut_acc_A2_F1` exactly (max |diff| 0.0).
Binning cost for `A2dbk`: model LL 0.29413 vs LUT LL 0.29465 (F2), against the served LUT's 0.29481.

**Standing-rule defect found, all arms (not fixed here).** A2 is not team-responsive.

| | defence prior quintiles Q1..Q5 |
|---|---|
| actual, F2 | 0.0876 .. 0.0980 |
| A2 prediction, F2 | 0.1010 .. 0.0971 |

The slope is negative (-0.32 F2, -0.19 F1) because A2 has no team term. Each quintile holds about 70 teams.

## 2. Closed loop, DESCRIPTIVE (`A2dbk` is not eligible)

**Setup.**
- Paired: served v3 vs served v3 + `ENGINE_FOUL_CAL=A2dbk`, seeds 0-49, every game of the slate, home box at 8 workers.
  - F2: 2025, `engine_v3`.
  - F1: 2024, `engine_v3_f1`, `d1007_K2F1` overlay, static rotation, table `lut_acc_A2dbk_F1`.
- Tap: `scripts/run_foul_cal_tap_v1.py`. Grader: `scripts/grade_foul_cal_closed_loop_v1.py` (`results/foul_cal/loop_F{1,2}.json`). Truth: verified finals.
- SE is the paired game-bootstrap SE of the move.
- Tap-control identity vs the box reads, seeds 0-49:
  - F1 `f1c_K2_full_s200_o0`: 281,750 / 281,750 game-seeds identical.
  - F2 `laneH_v3full_K2_s200_o0`: 285,499 / 285,500. One game-seed (401706923, seed 35) differs, a cross-platform numeric difference that does not involve the tap.
- Flag-off parity v10: PASS, bit-identical.

**Actual in-bonus share.** Actual in-bonus = `off_in_bonus_true` (the engine-definition state) over all first-half possessions of the verified games.
That is why the actual values run below the diag's (0.284 / 0.196), which were not on this definition.

| line | fold | d0-14 | d15-45 | d46+ | all |
|---|---|---|---|---|---|
| n games | F2 / F1 | 620 / 697 | 1,215 / 1,182 | 3,870 / 3,756 | 5,705 / 5,635 |
| total bias, v3 -> A2dbk (SE) | F2 | -5.16 -> -5.01 (0.03) | -0.13 -> -0.17 (0.01) | +0.67 -> +0.63 (0.01) | -0.14 -> -0.15 (0.007) |
| | F1 | -4.93 -> -4.77 (0.03) | -1.80 -> -1.80 (0.01) | **-0.65 -> -0.69 (0.009)** | -1.42 -> -1.43 (0.007) |
| H1 in-bonus share, v3 -> arm (actual) | F2 | 0.215 -> 0.241 (0.268) | 0.211 -> 0.212 (0.212) | 0.206 -> 0.198 (0.182) | |
| | F1 | 0.208 -> 0.233 (0.268) | 0.213 -> 0.213 (0.209) | 0.209 -> 0.203 (0.187) | |
| FTA/FGA (actual) | F2 | 0.3204 -> 0.3260 (0.3454) | 0.3251 -> 0.3251 (0.3272) | 0.3342 -> 0.3328 (0.3277) | 0.3307 -> 0.3304 (0.3295) |
| | F1 | 0.3092 -> 0.3138 (0.3365) | 0.3188 -> 0.3186 (0.3275) | 0.3274 -> 0.3263 (0.3268) | 0.3234 -> 0.3231 (0.3281) |
| H2 in-bonus share (actual) | F2 | 0.374 -> 0.384 (0.405) | 0.375 -> 0.374 (0.375) | 0.379 -> 0.378 (0.374) | |
| H2 bonus trips per in-bonus poss (actual) | F2 | 0.228 -> 0.228 (0.218) | 0.233 -> 0.233 (0.228) | 0.237 -> 0.238 (0.234) | |

Finer H1 buckets, in-bonus share v3 -> arm (actual):

| bucket | F2 | F1 |
|---|---|---|
| d0-7 | 0.215 -> 0.246 (0.279) | 0.206 -> 0.234 (0.292) |
| d8-14 | 0.215 -> 0.237 (0.256) | 0.211 -> 0.232 (0.243) |
| d15-30 | 0.213 -> 0.215 (0.216) | 0.210 -> 0.210 (0.204) |
| d31-45 | 0.208 -> 0.207 (0.206) | 0.216 -> 0.218 (0.216) |

**Gates (`eval_gates.py`, 50 seeds).** G5 lines have no measured floor at 50 seeds. Every move is under 0.003 and no status changes.

| gate | F2 v3 -> arm | F1 v3 -> arm |
|---|---|---|
| G5 total SD ratio | 0.9627 -> 0.9632 | 0.9515 -> 0.9521 |
| G5 margin SD ratio | 1.0313 -> 1.0317 | 1.0244 -> 1.0268 |
| G5 home/away corr (real 0.228 / 0.252) | 0.1701 -> 0.1693 | 0.1892 -> 0.1882 |
| G9 total bias | -0.137 -> -0.150 (1.9 SE away) | -1.423 -> -1.427 |
| G9 slope | 0.9275 -> 0.9276 | 0.9122 -> 0.9137 |
| G4 ft_rate pooled (real 0.3295 / 0.3281) | 0.3307 -> 0.3304 | 0.3234 -> 0.3231 |

Gate summaries are identical between arms on both folds.

**Loop lines (s32.4).**
- L1 (d0-14 and d46+ within 0.02 of actual):
  - d0-14 FAILS on both folds (gaps of 0.026 and 0.035 remain).
  - d46+ passes on both and moves toward actual (F2 0.024 -> 0.016; F1 0.022 -> 0.015).
- L2: d0-14 moves toward zero on both folds. d46+ moves toward zero on F2 but **away on F1, by 4.4 SE: FAIL**.
- L3: no PASS -> FAIL and no primary moves away by more than 2 SE.
- Per-team window FTA/P slope: NOT RUN (a reported line, not a gate).
- Underpowered: no bucket line here; window cells are 620 / 697 games x 50 seeds.

## 3. Reading for the PM

1. **The calendar is a real accrual effect, and a days-bucket GBM feature captures it.** Offline the gain is +3.5 floors on both folds. In the loop, `A2dbk` halves the d0-14 H1 bonus-state gap and moves d46+ toward actual.
2. **It is not the early-total lever.** It recovers +0.15 / +0.16 points of a -5 point d0-14 deficit, and 17-22% of the d0-14 FTA/FGA gap.
   The diag's "74% of the FTA gap" is in FTA units. FTA rate is only about 0.7-1.0 points of the d0-14 total (decomp Shapley). Closing it fully would still leave about 4 points.
3. **The remaining H1 gap and the unrecovered FTA have other candidates:**
   - the d0-7 H1 in-bonus share stays 0.03-0.06 short;
   - offensive fouls are absent from the sim's team-foul state (actual +33% early);
   - shooting-trip team response is weak early (diag section 3);
   - the H2 d0-14 in-bonus share is short (0.384 vs 0.405).
4. **Two A2 defects surfaced, worth their own round** (they interact with any calendar term):
   - a season-level drift: A2 over-predicts 2024 and 2025 late by 10-14%;
   - no team response: the prior-quintile slope is negative, against the standing matchup-specific rule.

   An as-of level term (the PO s30 T1 pattern) plus team foul priors would be the natural next arms. A calendar term could be re-entered on top of them.

Artifacts:
- Tracked: `data/processed/models/possession_outcome/round10cal/lut_acc_A2dbk_F{1,2}.npz`, `lut_report_v1.json`, `train_meta.json`.
- Local only (gitignored, regenerable by `scripts/train_foul_cal_v1.py`, about 75 s on 8 cores): the predictions and fits.

Flag `ENGINE_FOUL_CAL` (module `src/cbb_sim/engine/foul_cal.py`) is default off.
