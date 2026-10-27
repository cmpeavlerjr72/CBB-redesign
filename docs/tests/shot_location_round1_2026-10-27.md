# Shot-location stage round 1: evidence (2026-10-27)

Pre-registration: `docs/models/shot_location/experiments.md` section 1 (commit `ea73296`, before any fit). Scripts committed empty of output before the first fit (`5f44dcb`), grader committed before it ran (`ada9729`). Freeze ON. Nothing is adopted, wired or served. Results summary: `experiments.md` section 2.

## 0. Verdicts (section 1.6, read on F2; F1 reported, not gated)

| class | Read B | Read A make | Read C (carried arm B2) | 1.6 verdict, F2 | F1 (reported) |
|---|---|---|---|---|---|
| `FGA_jump2` | **MATCHUP** (B1 vs B0 +24.5 floors, B2 vs B1 +8.9 floors, all lines pass) | **SIGNAL** (A1 +5.77 floors, A2 +12.07 floors, gate held) | **no target improves**: eFG on jumpers -0.02 floors, OREB +0.41, and-one +0.84 | **NOT CANDIDATE** (clauses 2 and 3 fail) | NOT CANDIDATE (and-one -1.61 floors worse, nothing improves) |
| `FGA_3` | **MATCHUP** (+22.6, +16.8 floors, all lines pass) | **SIGNAL** (A1 +1.18 floors, A2 +2.36 floors, gate held) | 3P make rate **+1.17 floors** and OREB on three misses **+1.07 floors**, but and-one on made threes **-1.95 floors** (worse beyond floor), and responsiveness fails (make: off_corner3 slope -0.29, 2 steps; OREB: def_corner3 0.46, off_top3 2 steps, def_top3 0.55) | **NOT CANDIDATE** (clauses 2 and 3 fail) | NOT CANDIDATE (3P +1.67 floors, none worse, but responsiveness fails) |

- **Leak test: PASS** on both rating families. Every column and season is within the gate. Worst |corr| is 0.023 for the offence and 0.035 for the defence (table T4). Not REFUSED.
- **The registered finding clause does not fire.** That clause is "Read A SIGNAL with Read B FLAT". Read B is MATCHUP in both classes, so the fg_make and rebound ledgers get no upper-bound entry under that clause.
- **What the round shows in one line.** Location explains make, rebound and jumper and-one beyond class. The band mix of a matchup is predictable before tipoff, and the defence's allowed profile adds to the offence's own. But carrying the predicted mix through to the game aggregates moves team-game MAE by at most about one bootstrap SE. Nothing in Read C clears the registered rule.

Headline numbers (F2):

| read | quantity | jump2 | three |
|---|---|---|---|
| A make | LL gain A1 / A2 vs R (floor) | +9.4e-4 / +3.8e-3 (1.6e-4 / 3.2e-4) | +1.5e-4 / +3.1e-4 (1.3e-4) |
| A and-one (PARTIAL) | LL gain A1 / A2 (floor) | +4.3e-3 / +2.3e-3 (4.2e-4 / 3.8e-4) SIGNAL | +9e-6 / -8e-5 (2.1e-4) NO SIGNAL |
| A rebound | LL gain A1 / A2 (floor) | +6.9e-4 / +7.5e-4 (3.95e-4) SIGNAL | +4.4e-4 / +1.6e-3 (1.7e-4 / 2.1e-4) SIGNAL |
| B | LL B0 / B1 / B2 / B3 | 0.68986 / 0.67352 / 0.67134 / 0.67066 | 1.06004 / 1.05372 / 1.05027 / 1.04964 |
| B | team-game share MAE B0 -> B2 (pp) | short2 16.30 -> 14.37 | corner3 9.22 -> 8.56, wing3 9.80 -> 9.55, top3 10.09 -> 9.37 |
| C | team-game MAE R -> loc_B2 (floors) | eFG 0.12204 -> 0.12204 (-0.02); OREB 0.14713 -> 0.14708 (+0.41) | 3P 0.12448 -> 0.12434 (+1.17); OREB 0.10377 -> 0.10366 (+1.07); and-one 0.01129 -> 0.01133 (-1.95) |

## 1. Frame, bands, coverage

- **Frame.**
  - x is folded to the half court and the basket sits at (52.5, 250) in tenths of a foot, as in `build_pae_shot_rating_v1.py`.
  - d is the distance in feet from the basket centre.
  - theta = `atan2(y - 250, xf - 52.5)` in degrees, the `dist_angle` convention of that script and of the `da15` grid. 0 points straight out from the basket toward half court, +/-90 runs along the baseline, and |theta| > 90 is behind the basket.
  - Plot: `docs/tests/shot_location_round1_bands_2026-10-27.png` (2025, 30k located jump2 / three attempts, coloured by band, with the 30-degree rays, the 10 ft circle and the 4 ft corner limit drawn). Theta range by band in that sample: top3 -30.0 to +29.7, which confirms the cut. Wing3 and corner3 reach about +/-100 because corner threes sit near the baseline.
- **Bands.** Exactly section 1.2: short2 d <= 10; long2 d > 10; corner3 = |y - 250| >= 210 and d <= 24; wing3 = not corner and |theta| >= 30; top3 = not corner and |theta| < 30. The rim class is ONE band and no rim location enters any table.
- **Design join.** The fg_make design to pbp join matched 100.0% of 2,239,678 rows (made 100.0%, missed 100.0%). Located share: 79.6 / 84.5 / 89.3 / 98.7% for 2022-2025.
- **`cell_p`.** Re-derived with the PAE code (da15, prior-season baseline, 2022 on 2023). It reproduces the PAE table to 3e-8.
- **Rebound link.** 1,032,073 jump2 / three miss opportunities in the served rebound design.
  - 88.0% join to an fg design row. Within games present in both tables the join rate is 99.26%. The 11.3% that do not join are opportunities in games outside the fg design population (`pbp_complete`): 2022 81%, 2025 96%.
  - Of the joined rows, 88.9% are located. Unjoined rows carry no band and are never imputed.

Band shares by season and month, with the pooled training views: T1 and T2 below. The bands are stable across seasons. short2 is 44.1-44.4% in 2022-24 and 45.7% in 2025, the largest move. corner3 is 21.3-21.8%, wing3 44.1-44.8% and top3 33.9-34.1%.

## 2. Worker disclosures (every choice the spec left open)

1. **Read A population.** Located rows only, for train AND test and for every arm.
   - R is therefore the served B1 spec refit on the located population. It is not the served artifact.
   - The filter is applied inside the task builder, after `fit_m`, so `fit_m` sees the served full design. The rim class is not fitted: `FG.SHOT_CLASSES` is restricted in-process.
2. **"Decile gate held" in Read A.** The arm's worst decile gap must not exceed R's by more than max(|gap R - gap R_seed|, 0.10 pp). This is the PAE-round convention, used because R itself fails the absolute 2.0 pp gate in several cells (e.g. make jump2 F2 2.23 pp).
3. **And-one channel.**
   - Same wrapper with `y = and_one`, population = made located jump2 / three attempts.
   - Served B1 features and the class's frozen ladder params are reused unchanged.
   - `FG.score`'s Decision-8 drivers are the make drivers. They are reported in the JSON and not used.
4. **Rebound calendar is S0.** One static fit per fold. The served calendar is `S1_weekly`; S0 was chosen for budget, and every arm shares it.
   - The design is rebound round 3's `design_round3.parquet`. Its A0B0C0 reproduces the served S0 exactly, with the `C_plus_state` columns unchanged.
   - Arm A2 for rebound = raw d (ft) and theta (deg) of the miss.
5. **Read B design.**
   - Static fit on the fold's training seasons.
   - GLM = sklearn `LogisticRegression(C=1.0, lbfgs)`, multinomial for threes, on train-standardised features (the rebound `RidgeLogitArm` convention).
   - State columns are those carried on the fg_make design: `period`, `seconds_remaining`, `score_diff`, `in_bonus`, `is_transition_f`, `chance_number`. Site = `site_home`, `site_away`.
   - Jumpers use one profile column per side (`*_short2_c`), because long2 is exactly its negative in log-odds.
   - B3 = LightGBM with `FG.LGBM_BASE` plus the class's `lgbm_ladder_v2` frozen params (31 leaves, min_child 200). The objective is multiclass for threes and binary for jumpers.
6. **As-of profile recipe** (31.2 transposed to shares).
   - Share: p_b = (k (lg_b + rho prev_c_b) + S_b) / (k + N_class).
   - Feature: logit(p_b) - logit(lg_b), with p clipped to [1e-3, 1 - 1e-3].
   - `lg` is the as-of league share over games strictly before the date. At a season opener it is the previous season's final share; for 2022 it is the 2022 final share, which has no effect because S = N = prev_c = 0 gives exactly 0.
   - `prev_c` is the team's raw previous-season final share minus that season's league share.
   - One (k, rho) per side, shared across bands and classes. It was chosen on BOTH fold-1 training seasons, 2022 and 2023 (the PAE round used 2023 only). The same profiles serve both folds.
7. **Leak test.** Run on the four non-redundant columns per side (short2, corner3, wing3, top3). A family fails if any column-season row reads LEAK.
8. **Read B responsiveness.**
   - A class's offence (defence) line passes only if every band line of that side passes. For jumpers that is short2 only.
   - "Both responsiveness lines" in the MATCHUP clause are B2's offence and defence lines.
   - The [0.8, 1.2] band is read on the raw slope ratio; slope / sqrt(registered reliability) is reported only.
   - Not exercised: had B2 added on log loss but failed a line, the class would have read PROFILE.
9. **Read C construction.**
   - The location arm marginalises A1 (band one-hots); A2's fine cell cannot be marginalised over bands.
   - And-one marginalises over P(b | made) proportional to P_B(b) P_A1make(b). OREB marginalises over P(b | miss) proportional to P_B(b) (1 - P_A1make(b)).
   - The reference R is this round's located-population R, so R and A1 share a population. Both are applied to ALL test attempts, located or not.
   - Read C is reported for B0, B1 and B2 and graded on the carried arm, B2 (MATCHUP in both classes).
10. **Read C targets per class.**
    - jump2: eFG on jump2 attempts, and-one on made jump2, OREB on jump2 misses.
    - three: 3P make rate, and-one on made threes, OREB on three misses.
    - Combined eFG (jumpers + threes) is reported.
    - Clause 2 "improves" is read over {make target, OREB}; "worse" over all three targets.
11. **Read C metric.** MAE of the per-team-game rate, unweighted over team-games. Floor = game-resampled bootstrap SE of the per-team-game absolute-error difference.
    - Responsiveness lines are attempt-weighted quintiles of the class's profile columns (jump2: short2; three: corner3 / wing3 / top3).
    - They are read on the improved target(s), offence plus defence when MATCHUP, Decision-8 low-span exemption.
    - Located-only games = team-games with >= 95% of the class's attempts located.
12. **Segment definitions.**
    - Tier = the offence's conference: power = ACC, Big Ten, Big 12, SEC, Big East, Pac-12; mid = A-10, American, Mountain West, WCC, MVC; low = the rest. Taken from CBBD games via `offense_is_home`.
    - Located-share quartile = the game's located share of jump2 + three attempts, quartiles within season.
13. **Diagnostics.**
    - Diagnostic 1 reads the coordinate-audit master tables: CBBD `playType == JumpShot`, located; override = classified rim with the 2.27 ft override but not without it.
    - Diagnostic 2 maps hoopR to the folded frame by xf = 10 (47 - |hx|), y = 250 + 10 hy (the audit's pairing convention), on attempts both sources locate.
14. **Blinding caveat.** The wrapped trainers print their own log loss to their logs, as in every previous round. Those logs were seen before the blind grade. No arm, population or setting was changed after any log or grade output.
15. **Wall time.** Start 11:23 ET. Build 29 s; all 30 fits about 7 min of wall clock in parallel batches (12-16 threads); grader about 3 min; diagnostics 2 s. Results done by 11:45 ET, docs by about 12:15 ET.

## 3. Findings beyond the verdict (descriptive, not gated)

- **Read B's defence-profile gain is strongly site-asymmetric** (segment table under T8).
  - The B2-over-B1 gain is +5.6e-3 (15 SE) when the offence is AWAY and -8.8e-4 (2.6 SE worse) when it is HOME on jumpers.
  - On threes it is +6.1e-3 away against +1.5e-3 home.
  - When the offence is away, the defence is the home team, whose allowed profile was charted mostly in its own arena by the same crew. A venue-charting component in the "matchup" signal is therefore plausible and is not separated here.
  - This is a hypothesis for the PM, not a result.
- **Team-game band shares are over-dispersed relative to the multinomial.** z SD is 1.31 for short2 and 1.11-1.22 for threes under B2. There is a per-game latent in shot selection beyond the as-of profiles, which a location stage would have to carry as a game-level draw.
- **B3 beats B2 on F2** (+2.3 floors jump2, +2.6 threes) but not on F1 (-0.7, +0.2). This is recorded; the GLM is carried (section 1.4).
- **The B2alt sensitivity is inside the floor on jumpers.** Threes show +2.2 floors on F2 and +1.7 on F1, in favour of the second-best (k, rho). The profiles are not knife-edge on (k, rho).
- **Rim override, registered diagnostic 1. This goes to the possession-layer queue.**
  - The 2.27 ft override fires on 1.5-4.9% of located JumpShot makes against 0.5-1.4% of misses.
  - Within game the difference is +0.85 / +1.05 / +0.98 / +3.34 pp for 2022-2025, with SE 0.05-0.08 pp. That is far beyond any floor.
  - The override moves makes into the rim class preferentially, the same outcome-dependent rim placement the audit found, and 2025 is three times the earlier seasons.
- **Second source.** From 2025-02-10, hoopR and CBBD assign a different band on 3.5% of jumpers and 4.4% of threes. Team-game band shares correlate 0.90-0.94, MAE 1.5-3.2 pp.
- **Reliability on this round's table** is slightly below the registered numbers in every cell: offence 0.88 / 0.76 / 0.61 / 0.76, defence 0.67 / 0.71 / 0.46 / 0.71. Wing3 stays the weakest band.

## 4. Suite, parity, artifacts

- **Suite.** `pytest -q`: 879 passed, 1 skipped, 1 FAILED.
  - The failing test is `tests/test_shot_block_live_lut.py::test_attach_serves_live_table_and_missing_fails_loudly`: a live-LUT shape assertion, 5710 vs 6 games.
  - It is unrelated to this round. No file under `src/` or `tests/` changed, and the test reads a live table the scheduler maintains.
- **Parity v10.** `scripts/run_parity_smoke_v1.py`: **PASS, bit-identical** (60 x 5, served defaults, `results/engine_v0/parity_v10_shotloc_r1_20261027`). No `src` change.
- **Artifacts** (gitignored, HF `model_artifacts`) under `data/processed/models/shot_location/round1/`:
  - build: `design_loc_cols_v1.parquet`, `rebound_loc_cols_v1.parquet`, `team_band_profiles_v1.parquet`, `build_report_v1.json`, `diag_report_v1.json`;
  - runs: `readB/F{1,2}/`, `make/`, `andone/`, `rebound/` (one directory per arm, seed and fold, with preds, rows and counterfactual files).
  - Grade: `results/shot_location_v1/grade_v1.json`; logs in `results/shot_location_v1/`.
- **Reproduce:**

```
.venv/Scripts/python.exe scripts/build_shot_location_bands_v1.py --part build
.venv/Scripts/python.exe scripts/train_shot_location_v1.py --fold F1   (and F2)
.venv/Scripts/python.exe scripts/train_fg_make_v6_loc_v1.py --arm {R,A1,A2} --target make --fold {F1,F2} --seed {0,1 for R} --arms B1 --no-floor --no-leak --out-dir data/processed/models/shot_location/round1/make/<arm>_s<seed>_<fold>
.venv/Scripts/python.exe scripts/train_foul_draw_v1.py   (same options, out-dir .../andone/...)
.venv/Scripts/python.exe scripts/train_rebound_v2_loc_v1.py --arm {R,A1,A2} --fold {F1,F2} --seed {0,1} --out-dir .../rebound/<arm>_s<seed>_<fold>
.venv/Scripts/python.exe scripts/grade_shot_location_v1.py
.venv/Scripts/python.exe scripts/build_shot_location_bands_v1.py --part diag
```

## 5. NOT RUN

- The served `S1_weekly` calendar for the rebound arms; S0 was used (disclosure 4).
- Diagnostic 3's Sportradar 2015-18 reference appears as a quoted table, copied from `docs/tests/sportradar_inventory_2026-10-27.md` part 3. It was not recomputed.

## 6. Tables

Sportradar league band shares (older-era reference for diagnostic 3, from `sportradar_inventory_2026-10-27.md` section 4): corner3 18.1 / 18.2% in 2013-14 and 2014-15, then 21.5 / 21.5 / 22.6% in 2015-16 to 2017-18; wing3 47.9 -> 41.5%; top3 34.0-35.8%; short2 39.4 / 44.3 / 42.0 / 43.2% in 2014-15 to 2017-18. Against these, this round's CBBD 2022-25 located shares (T1) are corner3 21.3-21.8%, wing3 44.1-44.8%, top3 33.9-34.1% and short2 44.1-45.7%. Corner3 has sat at the post-2015 level ever since; short2 is about 1-2 points above Sportradar's 2015-18 level.

### T1. Band shares by season (located attempts, % within class)

| season | short2 | long2 | corner3 | wing3 | top3 | located jump2 | located three |
|---|---|---|---|---|---|---|---|
| 2022 | 44.12 | 55.88 | 21.37 | 44.74 | 33.89 | 104,229 | 148,329 |
| 2023 | 44.42 | 55.58 | 21.29 | 44.81 | 33.90 | 115,062 | 164,885 |
| 2024 | 44.22 | 55.78 | 21.73 | 44.24 | 34.03 | 136,515 | 202,620 |
| 2025 | 45.70 | 54.30 | 21.80 | 44.10 | 34.10 | 146,927 | 245,019 |
| pooled F1 train (2022-2023) | 44.28 | 55.72 | 21.33 | 44.77 | 33.89 | | |
| pooled F2 train (2022-2023-2024) | 44.26 | 55.74 | 21.49 | 44.56 | 33.95 | | |

### T2. Band drift by season and month (% within class)

| season | month | short2 | corner3 | wing3 | top3 |
|---|---|---|---|---|---|
| 2022 | 11 | 42.43 | 21.75 | 44.94 | 33.31 |
| 2022 | 12 | 43.82 | 22.19 | 44.29 | 33.51 |
| 2022 | 1 | 44.70 | 21.86 | 44.41 | 33.73 |
| 2022 | 2 | 45.13 | 20.61 | 45.27 | 34.13 |
| 2022 | 3 | 43.72 | 19.88 | 44.45 | 35.67 |
| 2022 | 4 | 62.50 | 25.00 | 33.33 | 41.67 |
| 2023 | 11 | 44.04 | 21.80 | 44.63 | 33.56 |
| 2023 | 12 | 44.81 | 21.96 | 44.78 | 33.26 |
| 2023 | 1 | 44.51 | 20.98 | 44.78 | 34.24 |
| 2023 | 2 | 44.48 | 20.97 | 45.18 | 33.85 |
| 2023 | 3 | 44.06 | 20.24 | 44.34 | 35.42 |
| 2023 | 4 | 50.00 | 16.67 | 33.33 | 50.00 |
| 2024 | 11 | 43.85 | 21.41 | 44.55 | 34.04 |
| 2024 | 12 | 44.31 | 22.84 | 44.18 | 32.98 |
| 2024 | 1 | 44.26 | 22.08 | 43.67 | 34.25 |
| 2024 | 2 | 44.49 | 21.18 | 44.50 | 34.32 |
| 2024 | 3 | 44.05 | 21.09 | 44.39 | 34.51 |
| 2024 | 4 | 39.39 | 7.14 | 45.24 | 47.62 |
| 2025 | 11 | 46.75 | 22.27 | 44.47 | 33.26 |
| 2025 | 12 | 46.41 | 22.38 | 44.29 | 33.33 |
| 2025 | 1 | 46.35 | 21.92 | 44.10 | 33.97 |
| 2025 | 2 | 44.72 | 21.41 | 44.15 | 34.44 |
| 2025 | 3 | 43.77 | 20.91 | 43.24 | 35.85 |
| 2025 | 4 | 41.96 | 15.72 | 43.26 | 41.02 |

### T3. As-of profile hyperparameters (fold-1 training seasons 2022-2023)

| side | chosen k | chosen rho | wMSE | second-best k | second-best rho | wMSE |
|---|---|---|---|---|---|---|
| off | 100 | 0.5 | 0.051929 | 100 | 0.75 | 0.051990 |
| def | 200 | 0.5 | 0.055455 | 400 | 0.5 | 0.055477 |

### T4. Leak test (change form vs verified own-game margin; gate |corr| <= 0.15)

| column | 2022 | 2023 | 2024 | 2025 | pooled | verdict |
|---|---|---|---|---|---|---|
| off_short2_c | +0.0216 | +0.0051 | +0.0217 | +0.0232 | +0.0183 | pass |
| off_corner3_c | +0.0197 | -0.0019 | +0.0064 | +0.0090 | +0.0082 | pass |
| off_wing3_c | -0.0099 | -0.0001 | +0.0030 | -0.0214 | -0.0075 | pass |
| off_top3_c | -0.0059 | +0.0032 | -0.0095 | +0.0110 | +0.0001 | pass |
| def_short2_c | -0.0283 | -0.0161 | -0.0135 | -0.0235 | -0.0202 | pass |
| def_corner3_c | -0.0240 | -0.0213 | -0.0205 | -0.0085 | -0.0180 | pass |
| def_wing3_c | -0.0117 | +0.0027 | -0.0056 | -0.0103 | -0.0065 | pass |
| def_top3_c | +0.0349 | +0.0166 | +0.0252 | +0.0195 | +0.0237 | pass |

Family verdicts: offence PASS, defence PASS; REFUSED = False.

### T5-make. Read A, make

| fold | class | n | LL(R) | abs LL(R)-LL(R_seed) | arm | LL gain | boot SE | floor | floors | worst decile gap R / arm (pp) | gate held | class verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F2 | FGA_jump2 | 146,927 | 0.665482 | 1.63e-04 | A1 | +9.42e-04 | 1.55e-04 | 1.63e-04 | +5.77 | 2.228 / 1.733 | True | SIGNAL |
| F2 | FGA_jump2 | 146,927 | 0.665482 | 1.63e-04 | A2 | +3.81e-03 | 3.16e-04 | 3.16e-04 | +12.07 | 2.228 / 1.718 | True |  |
| F2 | FGA_3 | 245,019 | 0.637680 | 1.30e-04 | A1 | +1.53e-04 | 7.38e-05 | 1.30e-04 | +1.18 | 2.024 / 1.980 | True | SIGNAL |
| F2 | FGA_3 | 245,019 | 0.637680 | 1.30e-04 | A2 | +3.06e-04 | 9.86e-05 | 1.30e-04 | +2.36 | 2.024 / 2.118 | True |  |
| F1 | FGA_jump2 | 136,515 | 0.665254 | 6.80e-05 | A1 | +5.78e-04 | 1.60e-04 | 1.60e-04 | +3.60 | 2.919 / 3.065 | False | SIGNAL |
| F1 | FGA_jump2 | 136,515 | 0.665254 | 6.80e-05 | A2 | +3.20e-03 | 3.08e-04 | 3.08e-04 | +10.40 | 2.919 / 2.532 | True |  |
| F1 | FGA_3 | 202,620 | 0.638162 | 1.73e-04 | A1 | +2.91e-04 | 9.43e-05 | 1.73e-04 | +1.68 | 2.030 / 2.417 | True | SIGNAL |
| F1 | FGA_3 | 202,620 | 0.638162 | 1.73e-04 | A2 | +5.16e-04 | 1.17e-04 | 1.73e-04 | +2.99 | 2.030 / 1.987 | True |  |

Responsiveness lines, F2: predicted / realised rate by band (n), and the slope ratio (monotone steps of 4) of the offence as-of band-share quintile line:

| class | arm | by band | off-share quintile line |
|---|---|---|---|
| FGA_jump2 | R | short2 0.3910 / 0.4097 (67,139); long2 0.3862 / 0.3697 (79,788) | short2 0.2789 (3) |
| FGA_jump2 | A1 | short2 0.4085 / 0.4097 (67,139); long2 0.3722 / 0.3697 (79,788) | short2 0.5468 (4) |
| FGA_jump2 | A2 | short2 0.4068 / 0.4097 (67,139); long2 0.3728 / 0.3697 (79,788) | short2 0.7429 (4) |
| FGA_3 | R | corner3 0.3388 / 0.3549 (53,413); wing3 0.3374 / 0.3355 (108,056); top3 0.3349 / 0.3292 (83,550) | corner3 0.9847 (2); wing3 0.3233 (3); top3 0.3911 (3) |
| FGA_3 | A1 | corner3 0.3563 / 0.3549 (53,413); wing3 0.3355 / 0.3355 (108,056); top3 0.3270 / 0.3292 (83,550) | corner3 -0.3285 (2); wing3 0.3705 (4); top3 0.1473 (3) |
| FGA_3 | A2 | corner3 0.3547 / 0.3549 (53,413); wing3 0.3351 / 0.3355 (108,056); top3 0.3258 / 0.3292 (83,550) | corner3 -1.0555 (2); wing3 0.4743 (4); top3 0.1366 (3) |

Segments, F2, the class's higher-scoring arm vs R (gain / SE; cells under the registered minimum are UNDERPOWERED):

| class | arm | segment | n | gain | SE | |
|---|---|---|---|---|---|---|
| FGA_jump2 | A2 | site=away | 65,670 | +3.90e-03 | 4.15e-04 |  |
| FGA_jump2 | A2 | site=home | 61,404 | +3.65e-03 | 4.60e-04 |  |
| FGA_jump2 | A2 | site=neutral | 19,853 | +4.00e-03 | 6.92e-04 |  |
| FGA_jump2 | A2 | month=Dec | 24,133 | +4.15e-03 | 7.71e-04 |  |
| FGA_jump2 | A2 | month=Feb | 32,822 | +2.54e-03 | 5.97e-04 |  |
| FGA_jump2 | A2 | month=Jan | 36,979 | +5.21e-03 | 6.12e-04 |  |
| FGA_jump2 | A2 | month=Mar+Apr | 21,561 | +1.29e-03 | 5.87e-04 |  |
| FGA_jump2 | A2 | month=Nov | 31,432 | +4.96e-03 | 7.21e-04 |  |
| FGA_jump2 | A2 | tier=low | 86,023 | +5.02e-03 | 4.28e-04 |  |
| FGA_jump2 | A2 | tier=mid | 24,877 | +2.54e-03 | 5.56e-04 |  |
| FGA_jump2 | A2 | tier=power | 36,027 | +1.81e-03 | 3.91e-04 |  |
| FGA_jump2 | A2 | locq=Q1 | 38,279 | +4.88e-03 | 6.45e-04 |  |
| FGA_jump2 | A2 | locq=Q2 | 35,567 | +3.39e-03 | 5.29e-04 |  |
| FGA_jump2 | A2 | locq=Q3 | 35,884 | +4.23e-03 | 7.07e-04 |  |
| FGA_jump2 | A2 | locq=Q4 | 37,197 | +2.71e-03 | 5.04e-04 |  |
| FGA_3 | A2 | site=away | 105,486 | +1.64e-04 | 1.42e-04 |  |
| FGA_3 | A2 | site=home | 107,741 | +4.95e-04 | 1.39e-04 |  |
| FGA_3 | A2 | site=neutral | 31,792 | +1.38e-04 | 2.26e-04 |  |
| FGA_3 | A2 | month=Dec | 40,577 | +3.05e-04 | 2.12e-04 |  |
| FGA_3 | A2 | month=Feb | 54,697 | +3.58e-04 | 1.95e-04 |  |
| FGA_3 | A2 | month=Jan | 61,205 | +4.45e-04 | 1.97e-04 |  |
| FGA_3 | A2 | month=Mar+Apr | 35,061 | +2.83e-04 | 2.35e-04 |  |
| FGA_3 | A2 | month=Nov | 53,479 | +1.10e-04 | 2.39e-04 |  |
| FGA_3 | A2 | tier=low | 142,576 | +2.37e-04 | 1.20e-04 |  |
| FGA_3 | A2 | tier=mid | 43,187 | +5.19e-04 | 2.17e-04 |  |
| FGA_3 | A2 | tier=power | 59,256 | +3.16e-04 | 2.00e-04 |  |
| FGA_3 | A2 | locq=Q1 | 59,744 | +6.37e-04 | 1.88e-04 |  |
| FGA_3 | A2 | locq=Q2 | 63,032 | +1.01e-04 | 1.92e-04 |  |
| FGA_3 | A2 | locq=Q3 | 61,308 | +3.65e-04 | 2.15e-04 |  |
| FGA_3 | A2 | locq=Q4 | 60,935 | +1.35e-04 | 1.69e-04 |  |

### T5-andone. Read A, and-one on made attempts (PARTIAL by construction)

| fold | class | n | LL(R) | abs LL(R)-LL(R_seed) | arm | LL gain | boot SE | floor | floors | worst decile gap R / arm (pp) | gate held | class verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F2 | FGA_jump2 | 57,006 | 0.159906 | 6.92e-05 | A1 | +4.26e-03 | 4.23e-04 | 4.23e-04 | +10.08 | 3.156 / 3.127 | True | SIGNAL |
| F2 | FGA_jump2 | 57,006 | 0.159906 | 6.92e-05 | A2 | +2.33e-03 | 3.75e-04 | 3.75e-04 | +6.23 | 3.156 / 3.044 | True |  |
| F2 | FGA_3 | 82,711 | 0.036357 | 2.09e-04 | A1 | +9.27e-06 | 1.30e-04 | 2.09e-04 | +0.04 | 1.015 / 0.975 | True | NO SIGNAL |
| F2 | FGA_3 | 82,711 | 0.036357 | 2.09e-04 | A2 | -8.02e-05 | 1.64e-04 | 2.09e-04 | -0.38 | 1.015 / 0.910 | True |  |
| F1 | FGA_jump2 | 52,588 | 0.165835 | 1.83e-04 | A1 | +4.10e-03 | 5.57e-04 | 5.57e-04 | +7.36 | 2.399 / 2.801 | True | SIGNAL |
| F1 | FGA_jump2 | 52,588 | 0.165835 | 1.83e-04 | A2 | +4.12e-03 | 4.97e-04 | 4.97e-04 | +8.29 | 2.399 / 1.982 | True |  |
| F1 | FGA_3 | 68,466 | 0.034819 | 1.09e-04 | A1 | -3.87e-05 | 1.64e-04 | 1.64e-04 | -0.24 | 0.884 / 0.809 | True | NO SIGNAL |
| F1 | FGA_3 | 68,466 | 0.034819 | 1.09e-04 | A2 | -9.55e-05 | 1.78e-04 | 1.78e-04 | -0.54 | 0.884 / 0.942 | True |  |

Responsiveness lines, F2: predicted / realised rate by band (n), and the slope ratio (monotone steps of 4) of the offence as-of band-share quintile line:

| class | arm | by band | off-share quintile line |
|---|---|---|---|
| FGA_jump2 | R | short2 0.0405 / 0.0562 (27,508); long2 0.0371 / 0.0203 (29,498) | short2 0.6138 (2) |
| FGA_jump2 | A1 | short2 0.0591 / 0.0562 (27,508); long2 0.0207 / 0.0203 (29,498) | short2 1.5121 (4) |
| FGA_jump2 | A2 | short2 0.0492 / 0.0562 (27,508); long2 0.0288 / 0.0203 (29,498) | short2 1.2388 (4) |
| FGA_3 | R | corner3 0.0048 / 0.0064 (18,954); wing3 0.0048 / 0.0052 (36,253); top3 0.0049 / 0.0057 (27,504) | corner3 -0.0954 (2); wing3 -0.0309 (1); top3 -0.019 (2) |
| FGA_3 | A1 | corner3 0.0046 / 0.0064 (18,954); wing3 0.0048 / 0.0052 (36,253); top3 0.0050 / 0.0057 (27,504) | corner3 -0.1143 (2); wing3 0.084 (2); top3 0.0223 (2) |
| FGA_3 | A2 | corner3 0.0048 / 0.0064 (18,954); wing3 0.0047 / 0.0052 (36,253); top3 0.0049 / 0.0057 (27,504) | corner3 -0.1658 (1); wing3 0.1472 (2); top3 0.0514 (3) |

Segments, F2, the class's higher-scoring arm vs R (gain / SE; cells under the registered minimum are UNDERPOWERED):

| class | arm | segment | n | gain | SE | |
|---|---|---|---|---|---|---|
| FGA_jump2 | A1 | site=away | 24,754 | +4.41e-03 | 7.18e-04 |  |
| FGA_jump2 | A1 | site=home | 24,612 | +3.24e-03 | 7.53e-04 |  |
| FGA_jump2 | A1 | site=neutral | 7,640 | +7.09e-03 | 1.16e-03 |  |
| FGA_jump2 | A1 | month=Dec | 9,458 | +6.02e-03 | 1.28e-03 |  |
| FGA_jump2 | A1 | month=Feb | 12,885 | +2.43e-03 | 9.12e-04 |  |
| FGA_jump2 | A1 | month=Jan | 14,493 | +3.57e-03 | 1.00e-03 |  |
| FGA_jump2 | A1 | month=Mar+Apr | 8,126 | +4.16e-03 | 1.10e-03 |  |
| FGA_jump2 | A1 | month=Nov | 12,044 | +5.76e-03 | 1.04e-03 |  |
| FGA_jump2 | A1 | tier=low | 33,295 | +4.07e-03 | 6.17e-04 |  |
| FGA_jump2 | A1 | tier=mid | 9,599 | +4.03e-03 | 1.00e-03 |  |
| FGA_jump2 | A1 | tier=power | 14,112 | +4.88e-03 | 9.77e-04 |  |
| FGA_jump2 | A1 | locq=Q1 | 14,803 | +4.04e-03 | 9.24e-04 |  |
| FGA_jump2 | A1 | locq=Q2 | 13,731 | +5.92e-03 | 1.07e-03 |  |
| FGA_jump2 | A1 | locq=Q3 | 14,045 | +4.28e-03 | 8.85e-04 |  |
| FGA_jump2 | A1 | locq=Q4 | 14,427 | +2.90e-03 | 8.40e-04 |  |
| FGA_3 | A1 | site=away | 34,858 | +2.52e-04 | 2.15e-04 |  |
| FGA_3 | A1 | site=home | 37,330 | -1.09e-04 | 1.98e-04 |  |
| FGA_3 | A1 | site=neutral | 10,523 | -3.77e-04 | 3.46e-04 |  |
| FGA_3 | A1 | month=Dec | 13,706 | +3.59e-04 | 3.08e-04 |  |
| FGA_3 | A1 | month=Feb | 18,772 | -3.36e-04 | 2.83e-04 |  |
| FGA_3 | A1 | month=Jan | 20,741 | +2.24e-04 | 2.61e-04 |  |
| FGA_3 | A1 | month=Mar+Apr | 11,860 | -2.04e-05 | 3.45e-04 |  |
| FGA_3 | A1 | month=Nov | 17,632 | -1.28e-04 | 2.97e-04 |  |
| FGA_3 | A1 | tier=low | 47,751 | -2.28e-04 | 1.54e-04 |  |
| FGA_3 | A1 | tier=mid | 14,642 | +3.91e-04 | 2.96e-04 |  |
| FGA_3 | A1 | tier=power | 20,318 | +2.92e-04 | 2.75e-04 |  |
| FGA_3 | A1 | locq=Q1 | 19,952 | -1.78e-04 | 2.92e-04 |  |
| FGA_3 | A1 | locq=Q2 | 21,440 | +7.76e-05 | 2.57e-04 |  |
| FGA_3 | A1 | locq=Q3 | 20,735 | -1.89e-04 | 2.56e-04 |  |
| FGA_3 | A1 | locq=Q4 | 20,584 | +3.19e-04 | 2.46e-04 |  |

### T5-rebound. Read A, rebound (3-class, S0 calendar)

| fold | class | n | LL(R) | abs LL(R)-LL(R_seed) | arm | LL gain | boot SE | floor | floors | worst decile gap R / arm (pp) | gate held | class verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F2 | FGA_jump2 | 89,239 | 0.647156 | 3.95e-04 | A1 | +6.94e-04 | 2.54e-04 | 3.95e-04 | +1.76 | 3.704 / 3.102 | True | SIGNAL |
| F2 | FGA_jump2 | 89,239 | 0.647156 | 3.95e-04 | A2 | +7.47e-04 | 2.93e-04 | 3.95e-04 | +1.89 | 3.704 / 3.142 | True |  |
| F2 | FGA_3 | 161,160 | 0.658438 | 2.20e-05 | A1 | +4.42e-04 | 1.72e-04 | 1.72e-04 | +2.56 | 3.398 / 3.400 | True | SIGNAL |
| F2 | FGA_3 | 161,160 | 0.658438 | 2.20e-05 | A2 | +1.63e-03 | 2.13e-04 | 2.13e-04 | +7.62 | 3.398 / 2.729 | True |  |
| F1 | FGA_jump2 | 83,277 | 0.617615 | 4.41e-05 | A1 | +4.02e-04 | 2.41e-04 | 2.41e-04 | +1.67 | 3.743 / 3.656 | True | SIGNAL |
| F1 | FGA_jump2 | 83,277 | 0.617615 | 4.41e-05 | A2 | +1.13e-03 | 3.41e-04 | 3.41e-04 | +3.30 | 3.743 / 3.078 | True |  |
| F1 | FGA_3 | 133,137 | 0.627096 | 8.55e-06 | A1 | +5.75e-04 | 1.85e-04 | 1.85e-04 | +3.11 | 4.089 / 4.272 | False | SIGNAL |
| F1 | FGA_3 | 133,137 | 0.627096 | 8.55e-06 | A2 | +1.04e-03 | 2.46e-04 | 2.46e-04 | +4.23 | 4.089 / 3.937 | True |  |

Responsiveness lines, F2: predicted / realised rate by band (n), and the slope ratio (monotone steps of 4) of the offence as-of band-share quintile line:

| class | arm | by band | off-share quintile line |
|---|---|---|---|
| FGA_jump2 | R | short2 0.2833 / 0.3065 (39,335); long2 0.2706 / 0.2680 (49,904) | short2 1.9749 (3) |
| FGA_jump2 | A1 | short2 0.2984 / 0.3065 (39,335); long2 0.2589 / 0.2680 (49,904) | short2 3.115 (3) |
| FGA_jump2 | A2 | short2 0.2982 / 0.3065 (39,335); long2 0.2590 / 0.2680 (49,904) | short2 3.3498 (3) |
| FGA_3 | R | corner3 0.2740 / 0.3007 (34,236); wing3 0.2723 / 0.2869 (71,284); top3 0.2713 / 0.2687 (55,640) | corner3 1.7695 (3); wing3 -20.5792 (3); top3 0.5013 (2) |
| FGA_3 | A1 | corner3 0.2873 / 0.3007 (34,236); wing3 0.2752 / 0.2869 (71,284); top3 0.2593 / 0.2687 (55,640) | corner3 2.6607 (3); wing3 -0.2761 (3); top3 1.3478 (2) |
| FGA_3 | A2 | corner3 0.2865 / 0.3007 (34,236); wing3 0.2747 / 0.2869 (71,284); top3 0.2604 / 0.2687 (55,640) | corner3 2.1065 (3); wing3 5.5298 (3); top3 1.2821 (2) |

Segments, F2, the class's higher-scoring arm vs R (gain / SE; cells under the registered minimum are UNDERPOWERED):

| class | arm | segment | n | gain | SE | |
|---|---|---|---|---|---|---|
| FGA_jump2 | A2 | site=away | 40,631 | +2.71e-04 | 4.34e-04 |  |
| FGA_jump2 | A2 | site=home | 36,506 | +1.23e-03 | 4.61e-04 |  |
| FGA_jump2 | A2 | site=neutral | 12,102 | +8.96e-04 | 9.85e-04 |  |
| FGA_jump2 | A2 | month=Dec | 14,568 | +2.18e-04 | 7.11e-04 |  |
| FGA_jump2 | A2 | month=Feb | 19,791 | -1.89e-04 | 7.20e-04 |  |
| FGA_jump2 | A2 | month=Jan | 22,304 | +1.25e-03 | 5.89e-04 |  |
| FGA_jump2 | A2 | month=Mar+Apr | 13,334 | +2.44e-03 | 9.42e-04 |  |
| FGA_jump2 | A2 | month=Nov | 19,242 | +3.55e-04 | 6.16e-04 |  |
| FGA_jump2 | A2 | tier=low | 52,375 | +1.02e-03 | 3.76e-04 |  |
| FGA_jump2 | A2 | tier=mid | 15,153 | +3.47e-04 | 8.19e-04 |  |
| FGA_jump2 | A2 | tier=power | 21,711 | +3.55e-04 | 6.52e-04 |  |
| FGA_jump2 | A2 | locq=Q1 | 23,307 | +2.16e-04 | 5.88e-04 |  |
| FGA_jump2 | A2 | locq=Q2 | 21,665 | +8.91e-04 | 6.63e-04 |  |
| FGA_jump2 | A2 | locq=Q3 | 21,666 | -3.73e-05 | 5.87e-04 |  |
| FGA_jump2 | A2 | locq=Q4 | 22,601 | +1.91e-03 | 6.15e-04 |  |
| FGA_3 | A2 | site=away | 70,132 | +1.88e-03 | 3.20e-04 |  |
| FGA_3 | A2 | site=home | 69,931 | +1.07e-03 | 3.16e-04 |  |
| FGA_3 | A2 | site=neutral | 21,097 | +2.61e-03 | 6.54e-04 |  |
| FGA_3 | A2 | month=Dec | 26,686 | +5.53e-04 | 5.07e-04 |  |
| FGA_3 | A2 | month=Feb | 35,686 | +2.14e-03 | 5.08e-04 |  |
| FGA_3 | A2 | month=Jan | 40,167 | +1.28e-03 | 4.09e-04 |  |
| FGA_3 | A2 | month=Mar+Apr | 23,060 | +3.54e-03 | 7.58e-04 |  |
| FGA_3 | A2 | month=Nov | 35,561 | +1.08e-03 | 4.53e-04 |  |
| FGA_3 | A2 | tier=low | 94,216 | +1.72e-03 | 3.02e-04 |  |
| FGA_3 | A2 | tier=mid | 28,319 | +1.72e-03 | 5.11e-04 |  |
| FGA_3 | A2 | tier=power | 38,625 | +1.34e-03 | 5.27e-04 |  |
| FGA_3 | A2 | locq=Q1 | 39,476 | +9.58e-04 | 4.48e-04 |  |
| FGA_3 | A2 | locq=Q2 | 41,310 | +1.71e-03 | 4.36e-04 |  |
| FGA_3 | A2 | locq=Q3 | 40,319 | +1.77e-03 | 5.18e-04 |  |
| FGA_3 | A2 | locq=Q4 | 40,055 | +2.06e-03 | 4.83e-04 |  |

### T6. Read B, log loss and comparisons (located attempts; floor = paired game-block bootstrap SE; seed floor 0)

| fold | class | n | LL B0 | LL B1 | LL B2 | LL B3 | LL B2alt | B1-B0 (floors) | B2-B1 (floors) | B3-B2 (floors) | B2alt-B2 (floors) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F2 | FGA_jump2 | 146,927 | 0.689855 | 0.673516 | 0.671340 | 0.670661 | 0.671339 | +1.63e-02 (+24.5) | +2.18e-03 (+8.9) | +6.79e-04 (+2.3) | +1.35e-06 (+0.0) | MATCHUP |
| F2 | FGA_3 | 245,019 | 1.060040 | 1.053721 | 1.050270 | 1.049637 | 1.050174 | +6.32e-03 (+22.6) | +3.45e-03 (+16.8) | +6.33e-04 (+2.6) | +9.58e-05 (+2.2) | MATCHUP |
| F1 | FGA_jump2 | 136,515 | 0.686451 | 0.672356 | 0.670342 | 0.670555 | 0.670599 | +1.41e-02 (+22.0) | +2.01e-03 (+8.0) | -2.13e-04 (-0.7) | -2.57e-04 (-2.7) | MATCHUP |
| F1 | FGA_3 | 202,620 | 1.059401 | 1.053604 | 1.051164 | 1.051116 | 1.051082 | +5.80e-03 (+20.3) | +2.44e-03 (+11.9) | +4.80e-05 (+0.2) | +8.23e-05 (+1.7) | MATCHUP |

### T7. Read B responsiveness (predicted vs realised band share by quintile of the as-of profile; slope ratio, monotone steps of 4, Decision-8 pass, slope / sqrt(registered reliability)). Defence lines and wing3 are WEAKER by registration.

| fold | class | arm | line | pred Q1..Q5 | actual Q1..Q5 | slope | steps | pass | slope/sqrt(rel) | label |
|---|---|---|---|---|---|---|---|---|---|---|
| F2 | FGA_jump2 | B0 | off_short2 | 0.443 0.443 0.443 0.443 0.443 | 0.340 0.409 0.455 0.500 0.581 | 0.0 | 0 | False | 0.0 |  |
| F2 | FGA_jump2 | B0 | def_short2 | 0.443 0.443 0.443 0.443 0.443 | 0.401 0.426 0.461 0.476 0.521 | 0.0 | 0 | False | 0.0 | WEAKER |
| F2 | FGA_jump2 | B1 | off_short2 | 0.326 0.395 0.436 0.484 0.569 | 0.340 0.409 0.455 0.500 0.581 | 1.0122 | 4 | True | 1.0729 |  |
| F2 | FGA_jump2 | B1 | def_short2 | 0.429 0.434 0.439 0.446 0.462 | 0.401 0.426 0.461 0.476 0.521 | 0.2683 | 4 | False | 0.323 | WEAKER |
| F2 | FGA_jump2 | B2 | off_short2 | 0.325 0.396 0.439 0.483 0.569 | 0.340 0.409 0.455 0.500 0.581 | 1.0132 | 4 | True | 1.074 |  |
| F2 | FGA_jump2 | B2 | def_short2 | 0.384 0.416 0.440 0.463 0.509 | 0.401 0.426 0.461 0.476 0.521 | 1.0504 | 4 | True | 1.2645 | WEAKER |
| F2 | FGA_jump2 | B3 | off_short2 | 0.332 0.392 0.433 0.483 0.574 | 0.340 0.409 0.455 0.500 0.581 | 1.0058 | 4 | True | 1.0661 |  |
| F2 | FGA_jump2 | B3 | def_short2 | 0.389 0.416 0.441 0.462 0.506 | 0.401 0.426 0.461 0.476 0.521 | 0.9755 | 4 | True | 1.1744 | WEAKER |
| F2 | FGA_3 | B0 | off_corner3 | 0.215 0.215 0.215 0.215 0.215 | 0.176 0.201 0.223 0.231 0.259 | 0.0 | 0 | False | 0.0 |  |
| F2 | FGA_3 | B0 | off_wing3 | 0.446 0.446 0.446 0.446 0.446 | 0.411 0.430 0.442 0.450 0.472 | 0.0 | 1 | False | 0.0 | WEAKER |
| F2 | FGA_3 | B0 | off_top3 | 0.339 0.339 0.339 0.339 0.339 | 0.297 0.326 0.337 0.358 0.388 | 0.0 | 0 | False | 0.0 |  |
| F2 | FGA_3 | B0 | def_corner3 | 0.215 0.215 0.215 0.215 0.215 | 0.179 0.205 0.213 0.228 0.265 | 0.0 | 0 | False | 0.0 | WEAKER |
| F2 | FGA_3 | B0 | def_wing3 | 0.446 0.446 0.446 0.446 0.446 | 0.418 0.433 0.441 0.451 0.462 | 0.0 | 1 | False | 0.0 | WEAKER |
| F2 | FGA_3 | B0 | def_top3 | 0.339 0.339 0.339 0.339 0.339 | 0.293 0.321 0.344 0.363 0.385 | 0.0 | 0 | False | 0.0 | WEAKER |
| F2 | FGA_3 | B1 | off_corner3 | 0.173 0.199 0.214 0.230 0.255 | 0.176 0.201 0.223 0.231 0.259 | 0.9915 | 4 | True | 1.1155 |  |
| F2 | FGA_3 | B1 | off_wing3 | 0.418 0.435 0.445 0.455 0.474 | 0.411 0.430 0.442 0.450 0.472 | 0.9184 | 4 | True | 1.122 | WEAKER |
| F2 | FGA_3 | B1 | off_top3 | 0.297 0.324 0.339 0.356 0.386 | 0.297 0.326 0.337 0.358 0.388 | 0.9666 | 4 | True | 1.0807 |  |
| F2 | FGA_3 | B1 | def_corner3 | 0.210 0.214 0.214 0.214 0.219 | 0.179 0.205 0.213 0.228 0.265 | 0.1061 | 3 | False | 0.1217 | WEAKER |
| F2 | FGA_3 | B1 | def_wing3 | 0.444 0.444 0.445 0.446 0.447 | 0.418 0.433 0.441 0.451 0.462 | 0.0743 | 4 | False | 0.103 | WEAKER |
| F2 | FGA_3 | B1 | def_top3 | 0.335 0.338 0.342 0.344 0.344 | 0.293 0.321 0.344 0.363 0.385 | 0.1065 | 4 | False | 0.123 | WEAKER |
| F2 | FGA_3 | B2 | off_corner3 | 0.171 0.199 0.215 0.230 0.256 | 0.176 0.201 0.223 0.231 0.259 | 1.0076 | 4 | True | 1.1336 |  |
| F2 | FGA_3 | B2 | off_wing3 | 0.417 0.434 0.444 0.455 0.475 | 0.411 0.430 0.442 0.450 0.472 | 0.9455 | 4 | True | 1.1551 | WEAKER |
| F2 | FGA_3 | B2 | off_top3 | 0.297 0.325 0.339 0.357 0.385 | 0.297 0.326 0.337 0.358 0.388 | 0.972 | 4 | True | 1.0867 |  |
| F2 | FGA_3 | B2 | def_corner3 | 0.177 0.201 0.212 0.225 0.256 | 0.179 0.205 0.213 0.228 0.265 | 0.9312 | 4 | True | 1.0682 | WEAKER |
| F2 | FGA_3 | B2 | def_wing3 | 0.425 0.436 0.444 0.453 0.467 | 0.418 0.433 0.441 0.451 0.462 | 0.9499 | 4 | True | 1.3173 | WEAKER |
| F2 | FGA_3 | B2 | def_top3 | 0.295 0.325 0.343 0.359 0.382 | 0.293 0.321 0.344 0.363 0.385 | 0.9392 | 4 | True | 1.0845 | WEAKER |
| F2 | FGA_3 | B3 | off_corner3 | 0.171 0.199 0.216 0.228 0.256 | 0.176 0.201 0.223 0.231 0.259 | 1.0192 | 4 | True | 1.1467 |  |
| F2 | FGA_3 | B3 | off_wing3 | 0.418 0.436 0.445 0.456 0.474 | 0.411 0.430 0.442 0.450 0.472 | 0.9123 | 4 | True | 1.1146 | WEAKER |
| F2 | FGA_3 | B3 | off_top3 | 0.297 0.326 0.340 0.356 0.383 | 0.297 0.326 0.337 0.358 0.388 | 0.9489 | 4 | True | 1.0609 |  |
| F2 | FGA_3 | B3 | def_corner3 | 0.176 0.202 0.212 0.223 0.256 | 0.179 0.205 0.213 0.228 0.265 | 0.9423 | 4 | True | 1.0809 | WEAKER |
| F2 | FGA_3 | B3 | def_wing3 | 0.426 0.436 0.447 0.453 0.467 | 0.418 0.433 0.441 0.451 0.462 | 0.9061 | 4 | True | 1.2565 | WEAKER |
| F2 | FGA_3 | B3 | def_top3 | 0.297 0.322 0.343 0.360 0.379 | 0.293 0.321 0.344 0.363 0.385 | 0.8811 | 4 | True | 1.0174 | WEAKER |
| F1 | FGA_jump2 | B0 | off_short2 | 0.443 0.443 0.443 0.443 0.443 | 0.338 0.403 0.432 0.478 0.561 | 0.0 | 0 | False | 0.0 |  |
| F1 | FGA_jump2 | B0 | def_short2 | 0.443 0.443 0.443 0.443 0.443 | 0.383 0.418 0.447 0.461 0.502 | 0.0 | 0 | False | 0.0 | WEAKER |
| F1 | FGA_jump2 | B1 | off_short2 | 0.329 0.399 0.442 0.484 0.560 | 0.338 0.403 0.432 0.478 0.561 | 1.0379 | 4 | True | 1.1002 |  |
| F1 | FGA_jump2 | B1 | def_short2 | 0.432 0.435 0.440 0.446 0.463 | 0.383 0.418 0.447 0.461 0.502 | 0.2624 | 4 | False | 0.3159 | WEAKER |
| F1 | FGA_jump2 | B2 | off_short2 | 0.329 0.401 0.440 0.482 0.559 | 0.338 0.403 0.432 0.478 0.561 | 1.0345 | 4 | True | 1.0966 |  |
| F1 | FGA_jump2 | B2 | def_short2 | 0.388 0.417 0.438 0.461 0.506 | 0.383 0.418 0.447 0.461 0.502 | 0.9975 | 4 | True | 1.2008 | WEAKER |
| F1 | FGA_jump2 | B3 | off_short2 | 0.341 0.396 0.435 0.481 0.561 | 0.338 0.403 0.432 0.478 0.561 | 0.9905 | 4 | True | 1.0499 |  |
| F1 | FGA_jump2 | B3 | def_short2 | 0.395 0.417 0.433 0.462 0.506 | 0.383 0.418 0.447 0.461 0.502 | 0.9401 | 4 | True | 1.1317 | WEAKER |
| F1 | FGA_3 | B0 | off_corner3 | 0.213 0.213 0.213 0.213 0.213 | 0.177 0.208 0.216 0.227 0.260 | 0.0 | 0 | False | 0.0 |  |
| F1 | FGA_3 | B0 | off_wing3 | 0.448 0.448 0.448 0.448 0.448 | 0.415 0.435 0.443 0.451 0.468 | 0.0 | 0 | False | 0.0 | WEAKER |
| F1 | FGA_3 | B0 | off_top3 | 0.339 0.339 0.339 0.339 0.339 | 0.298 0.324 0.339 0.359 0.382 | 0.0 | 0 | False | 0.0 |  |
| F1 | FGA_3 | B0 | def_corner3 | 0.213 0.213 0.213 0.213 0.213 | 0.185 0.206 0.212 0.229 0.254 | 0.0 | 0 | False | 0.0 | WEAKER |
| F1 | FGA_3 | B0 | def_wing3 | 0.448 0.448 0.448 0.448 0.448 | 0.421 0.434 0.446 0.448 0.464 | 0.0 | 0 | False | 0.0 | WEAKER |
| F1 | FGA_3 | B0 | def_top3 | 0.339 0.339 0.339 0.339 0.339 | 0.302 0.323 0.339 0.356 0.381 | 0.0 | 0 | False | 0.0 | WEAKER |
| F1 | FGA_3 | B1 | off_corner3 | 0.173 0.200 0.214 0.227 0.253 | 0.177 0.208 0.216 0.227 0.260 | 0.9651 | 4 | True | 1.0858 |  |
| F1 | FGA_3 | B1 | off_wing3 | 0.420 0.437 0.447 0.457 0.476 | 0.415 0.435 0.443 0.451 0.468 | 1.0617 | 4 | True | 1.2971 | WEAKER |
| F1 | FGA_3 | B1 | off_top3 | 0.295 0.322 0.339 0.356 0.385 | 0.298 0.324 0.339 0.359 0.382 | 1.071 | 4 | True | 1.1974 |  |
| F1 | FGA_3 | B1 | def_corner3 | 0.211 0.212 0.212 0.216 0.217 | 0.185 0.206 0.212 0.229 0.254 | 0.0797 | 4 | False | 0.0914 | WEAKER |
| F1 | FGA_3 | B1 | def_wing3 | 0.446 0.447 0.447 0.447 0.448 | 0.421 0.434 0.446 0.448 0.464 | 0.0464 | 3 | False | 0.0643 | WEAKER |
| F1 | FGA_3 | B1 | def_top3 | 0.333 0.339 0.340 0.341 0.343 | 0.302 0.323 0.339 0.356 0.381 | 0.1197 | 4 | False | 0.1382 | WEAKER |
| F1 | FGA_3 | B2 | off_corner3 | 0.173 0.200 0.213 0.225 0.253 | 0.177 0.208 0.216 0.227 0.260 | 0.9585 | 4 | True | 1.0784 |  |
| F1 | FGA_3 | B2 | off_wing3 | 0.420 0.437 0.446 0.457 0.477 | 0.415 0.435 0.443 0.451 0.468 | 1.0712 | 4 | True | 1.3087 | WEAKER |
| F1 | FGA_3 | B2 | off_top3 | 0.295 0.323 0.340 0.356 0.384 | 0.298 0.324 0.339 0.359 0.382 | 1.0672 | 4 | True | 1.1932 |  |
| F1 | FGA_3 | B2 | def_corner3 | 0.177 0.199 0.211 0.227 0.251 | 0.185 0.206 0.212 0.229 0.254 | 1.0725 | 4 | True | 1.2302 | WEAKER |
| F1 | FGA_3 | B2 | def_wing3 | 0.427 0.439 0.447 0.454 0.469 | 0.421 0.434 0.446 0.448 0.464 | 0.978 | 4 | True | 1.3562 | WEAKER |
| F1 | FGA_3 | B2 | def_top3 | 0.298 0.326 0.340 0.355 0.379 | 0.302 0.323 0.339 0.356 0.381 | 1.02 | 4 | True | 1.1778 | WEAKER |
| F1 | FGA_3 | B3 | off_corner3 | 0.173 0.198 0.216 0.224 0.251 | 0.177 0.208 0.216 0.227 0.260 | 0.9458 | 4 | True | 1.0641 |  |
| F1 | FGA_3 | B3 | off_wing3 | 0.422 0.437 0.449 0.460 0.475 | 0.415 0.435 0.443 0.451 0.468 | 0.9847 | 4 | True | 1.203 | WEAKER |
| F1 | FGA_3 | B3 | off_top3 | 0.297 0.322 0.340 0.356 0.381 | 0.298 0.324 0.339 0.359 0.382 | 1.0019 | 4 | True | 1.1202 |  |
| F1 | FGA_3 | B3 | def_corner3 | 0.175 0.201 0.211 0.226 0.248 | 0.185 0.206 0.212 0.229 0.254 | 1.0703 | 4 | True | 1.2277 | WEAKER |
| F1 | FGA_3 | B3 | def_wing3 | 0.429 0.441 0.448 0.455 0.469 | 0.421 0.434 0.446 0.448 0.464 | 0.9416 | 4 | True | 1.3058 | WEAKER |
| F1 | FGA_3 | B3 | def_top3 | 0.298 0.325 0.339 0.356 0.377 | 0.302 0.323 0.339 0.356 0.381 | 0.9929 | 4 | True | 1.1465 | WEAKER |

### T8. Read B team-game band shares (team-games with >= 5 located attempts in the class)

| fold | class | band | arm | team-games | MAE pp | resid SD pp | multinomial SD pp | z SD |
|---|---|---|---|---|---|---|---|---|
| F2 | FGA_jump2 | short2 | B0 | 10,358 | 16.298 | 20.038 | 14.468 | 1.460 |
| F2 | FGA_jump2 | short2 | B1 | 10,358 | 14.598 | 18.152 | 14.280 | 1.328 |
| F2 | FGA_jump2 | short2 | B2 | 10,358 | 14.370 | 17.924 | 14.248 | 1.307 |
| F2 | FGA_jump2 | short2 | B3 | 10,358 | 14.302 | 17.869 | 14.193 | 1.306 |
| F2 | FGA_3 | corner3 | B0 | 10,864 | 9.225 | 11.567 | 9.019 | 1.306 |
| F2 | FGA_3 | corner3 | B1 | 10,864 | 8.847 | 11.110 | 8.989 | 1.261 |
| F2 | FGA_3 | corner3 | B2 | 10,864 | 8.563 | 10.752 | 8.960 | 1.223 |
| F2 | FGA_3 | corner3 | B3 | 10,864 | 8.540 | 10.745 | 8.931 | 1.233 |
| F2 | FGA_3 | wing3 | B0 | 10,864 | 9.800 | 12.248 | 10.914 | 1.137 |
| F2 | FGA_3 | wing3 | B1 | 10,864 | 9.634 | 12.037 | 10.895 | 1.118 |
| F2 | FGA_3 | wing3 | B2 | 10,864 | 9.545 | 11.946 | 10.890 | 1.110 |
| F2 | FGA_3 | wing3 | B3 | 10,864 | 9.586 | 11.992 | 10.878 | 1.117 |
| F2 | FGA_3 | top3 | B0 | 10,864 | 10.085 | 12.707 | 10.398 | 1.240 |
| F2 | FGA_3 | top3 | B1 | 10,864 | 9.696 | 12.229 | 10.361 | 1.195 |
| F2 | FGA_3 | top3 | B2 | 10,864 | 9.372 | 11.841 | 10.346 | 1.156 |
| F2 | FGA_3 | top3 | B3 | 10,864 | 9.384 | 11.861 | 10.328 | 1.161 |
| F1 | FGA_jump2 | short2 | B0 | 9,105 | 15.739 | 19.369 | 14.127 | 1.449 |
| F1 | FGA_jump2 | short2 | B1 | 9,105 | 14.206 | 17.704 | 13.958 | 1.325 |
| F1 | FGA_jump2 | short2 | B2 | 9,105 | 14.000 | 17.478 | 13.930 | 1.305 |
| F1 | FGA_jump2 | short2 | B3 | 9,105 | 13.970 | 17.430 | 13.887 | 1.305 |
| F1 | FGA_3 | corner3 | B0 | 9,444 | 9.300 | 11.671 | 9.418 | 1.274 |
| F1 | FGA_3 | corner3 | B1 | 9,444 | 8.959 | 11.266 | 9.390 | 1.233 |
| F1 | FGA_3 | corner3 | B2 | 9,444 | 8.776 | 11.042 | 9.353 | 1.213 |
| F1 | FGA_3 | corner3 | B3 | 9,444 | 8.707 | 11.004 | 9.316 | 1.212 |
| F1 | FGA_3 | wing3 | B0 | 9,444 | 10.063 | 12.616 | 11.433 | 1.120 |
| F1 | FGA_3 | wing3 | B1 | 9,444 | 9.915 | 12.454 | 11.413 | 1.107 |
| F1 | FGA_3 | wing3 | B2 | 9,444 | 9.847 | 12.374 | 11.407 | 1.099 |
| F1 | FGA_3 | wing3 | B3 | 9,444 | 9.892 | 12.440 | 11.389 | 1.107 |
| F1 | FGA_3 | top3 | B0 | 9,444 | 10.165 | 12.807 | 10.883 | 1.198 |
| F1 | FGA_3 | top3 | B1 | 9,444 | 9.800 | 12.404 | 10.842 | 1.162 |
| F1 | FGA_3 | top3 | B2 | 9,444 | 9.594 | 12.117 | 10.835 | 1.134 |
| F1 | FGA_3 | top3 | B3 | 9,444 | 9.671 | 12.203 | 10.809 | 1.147 |

Read B segments, F2 (gain / SE; UNDERPOWERED under 5,000 attempts):

| class | comparison | segment | n | gain | SE | |
|---|---|---|---|---|---|---|
| FGA_jump2 | B2_vs_B1 | site=away | 65,670 | +5.60e-03 | 3.74e-04 |  |
| FGA_jump2 | B2_vs_B1 | site=home | 61,404 | -8.77e-04 | 3.41e-04 |  |
| FGA_jump2 | B2_vs_B1 | site=neutral | 19,853 | +2.87e-04 | 6.86e-04 |  |
| FGA_jump2 | B2_vs_B1 | month=Dec | 24,133 | +3.15e-03 | 5.51e-04 |  |
| FGA_jump2 | B2_vs_B1 | month=Feb | 32,822 | +1.51e-03 | 5.90e-04 |  |
| FGA_jump2 | B2_vs_B1 | month=Jan | 36,979 | +3.34e-03 | 5.20e-04 |  |
| FGA_jump2 | B2_vs_B1 | month=Mar+Apr | 21,561 | -1.08e-03 | 6.83e-04 |  |
| FGA_jump2 | B2_vs_B1 | month=Nov | 31,432 | +2.99e-03 | 5.25e-04 |  |
| FGA_jump2 | B2_vs_B1 | tier=low | 86,023 | +3.23e-03 | 3.48e-04 |  |
| FGA_jump2 | B2_vs_B1 | tier=mid | 24,877 | +8.98e-04 | 5.74e-04 |  |
| FGA_jump2 | B2_vs_B1 | tier=power | 36,027 | +5.33e-04 | 4.91e-04 |  |
| FGA_jump2 | B2_vs_B1 | locq=Q1 | 38,279 | +3.34e-03 | 5.48e-04 |  |
| FGA_jump2 | B2_vs_B1 | locq=Q2 | 35,567 | +2.76e-03 | 4.86e-04 |  |
| FGA_jump2 | B2_vs_B1 | locq=Q3 | 35,884 | +1.73e-03 | 4.99e-04 |  |
| FGA_jump2 | B2_vs_B1 | locq=Q4 | 37,197 | +8.57e-04 | 4.86e-04 |  |
| FGA_jump2 | B1_vs_B0 | site=away | 65,670 | +1.13e-02 | 9.82e-04 |  |
| FGA_jump2 | B1_vs_B0 | site=home | 61,404 | +2.37e-02 | 1.04e-03 |  |
| FGA_jump2 | B1_vs_B0 | site=neutral | 19,853 | +1.03e-02 | 1.91e-03 |  |
| FGA_jump2 | B1_vs_B0 | month=Dec | 24,133 | +1.89e-02 | 1.70e-03 |  |
| FGA_jump2 | B1_vs_B0 | month=Feb | 32,822 | +1.63e-02 | 1.49e-03 |  |
| FGA_jump2 | B1_vs_B0 | month=Jan | 36,979 | +1.90e-02 | 1.50e-03 |  |
| FGA_jump2 | B1_vs_B0 | month=Mar+Apr | 21,561 | +1.16e-02 | 1.75e-03 |  |
| FGA_jump2 | B1_vs_B0 | month=Nov | 31,432 | +1.45e-02 | 1.32e-03 |  |
| FGA_jump2 | B1_vs_B0 | tier=low | 86,023 | +1.67e-02 | 9.24e-04 |  |
| FGA_jump2 | B1_vs_B0 | tier=mid | 24,877 | +1.59e-02 | 1.74e-03 |  |
| FGA_jump2 | B1_vs_B0 | tier=power | 36,027 | +1.58e-02 | 1.06e-03 |  |
| FGA_jump2 | B1_vs_B0 | locq=Q1 | 38,279 | +1.82e-02 | 1.43e-03 |  |
| FGA_jump2 | B1_vs_B0 | locq=Q2 | 35,567 | +1.74e-02 | 1.38e-03 |  |
| FGA_jump2 | B1_vs_B0 | locq=Q3 | 35,884 | +1.54e-02 | 1.25e-03 |  |
| FGA_jump2 | B1_vs_B0 | locq=Q4 | 37,197 | +1.43e-02 | 1.37e-03 |  |
| FGA_3 | B2_vs_B1 | site=away | 105,486 | +6.09e-03 | 3.27e-04 |  |
| FGA_3 | B2_vs_B1 | site=home | 107,741 | +1.46e-03 | 2.69e-04 |  |
| FGA_3 | B2_vs_B1 | site=neutral | 31,792 | +1.44e-03 | 5.89e-04 |  |
| FGA_3 | B2_vs_B1 | month=Dec | 40,577 | +3.95e-03 | 5.02e-04 |  |
| FGA_3 | B2_vs_B1 | month=Feb | 54,697 | +4.00e-03 | 4.99e-04 |  |
| FGA_3 | B2_vs_B1 | month=Jan | 61,205 | +3.70e-03 | 4.23e-04 |  |
| FGA_3 | B2_vs_B1 | month=Mar+Apr | 35,061 | +3.25e-03 | 5.55e-04 |  |
| FGA_3 | B2_vs_B1 | month=Nov | 53,479 | +2.36e-03 | 4.31e-04 |  |
| FGA_3 | B2_vs_B1 | tier=low | 142,576 | +3.36e-03 | 2.46e-04 |  |
| FGA_3 | B2_vs_B1 | tier=mid | 43,187 | +3.74e-03 | 4.49e-04 |  |
| FGA_3 | B2_vs_B1 | tier=power | 59,256 | +3.45e-03 | 4.45e-04 |  |
| FGA_3 | B2_vs_B1 | locq=Q1 | 59,744 | +2.61e-03 | 3.82e-04 |  |
| FGA_3 | B2_vs_B1 | locq=Q2 | 63,032 | +3.90e-03 | 4.05e-04 |  |
| FGA_3 | B2_vs_B1 | locq=Q3 | 61,308 | +3.75e-03 | 3.93e-04 |  |
| FGA_3 | B2_vs_B1 | locq=Q4 | 60,935 | +3.51e-03 | 4.30e-04 |  |
| FGA_3 | B1_vs_B0 | site=away | 105,486 | +3.82e-03 | 3.99e-04 |  |
| FGA_3 | B1_vs_B0 | site=home | 107,741 | +9.03e-03 | 4.57e-04 |  |
| FGA_3 | B1_vs_B0 | site=neutral | 31,792 | +5.44e-03 | 7.14e-04 |  |
| FGA_3 | B1_vs_B0 | month=Dec | 40,577 | +6.23e-03 | 6.41e-04 |  |
| FGA_3 | B1_vs_B0 | month=Feb | 54,697 | +7.11e-03 | 6.65e-04 |  |
| FGA_3 | B1_vs_B0 | month=Jan | 61,205 | +6.97e-03 | 5.96e-04 |  |
| FGA_3 | B1_vs_B0 | month=Mar+Apr | 35,061 | +6.87e-03 | 8.34e-04 |  |
| FGA_3 | B1_vs_B0 | month=Nov | 53,479 | +4.47e-03 | 5.88e-04 |  |
| FGA_3 | B1_vs_B0 | tier=low | 142,576 | +5.39e-03 | 3.37e-04 |  |
| FGA_3 | B1_vs_B0 | tier=mid | 43,187 | +7.31e-03 | 6.74e-04 |  |
| FGA_3 | B1_vs_B0 | tier=power | 59,256 | +7.83e-03 | 6.91e-04 |  |
| FGA_3 | B1_vs_B0 | locq=Q1 | 59,744 | +5.35e-03 | 5.22e-04 |  |
| FGA_3 | B1_vs_B0 | locq=Q2 | 63,032 | +6.72e-03 | 6.01e-04 |  |
| FGA_3 | B1_vs_B0 | locq=Q3 | 61,308 | +6.53e-03 | 5.86e-04 |  |
| FGA_3 | B1_vs_B0 | locq=Q4 | 60,935 | +6.64e-03 | 6.48e-04 |  |

### T9. Read C, implied team-game rates vs realised (gain = MAE(R) - MAE(arm); floor = paired team-game bootstrap SE of the MAE difference)

| fold | target | scope | team-games | MAE R | arm | MAE arm | gain | SE | floors | bias R / arm | resid SD / pred SD, R | resid SD / pred SD, arm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F2 | efg_jump2 | all_games | 10,879 | 0.12204 | loc_B0 | 0.12218 | -1.45e-04 | 1.00e-04 | -1.44 | +0.00099 / +0.00144 | 0.1573 / 0.1506 | 0.1573 / 0.1506 |
| F2 | efg_jump2 | all_games | 10,879 | 0.12204 | loc_B1 | 0.12207 | -3.32e-05 | 1.00e-04 | -0.33 | +0.00099 / +0.00100 | 0.1573 / 0.1506 | 0.1572 / 0.1507 |
| F2 | efg_jump2 | all_games | 10,879 | 0.12204 | loc_B2 | 0.12204 | -2.17e-06 | 1.02e-04 | -0.02 | +0.00099 / +0.00080 | 0.1573 / 0.1506 | 0.1572 / 0.1507 |
| F2 | efg_jump2 | located_only_games | 10,133 | 0.12204 | loc_B0 | 0.12220 | -1.55e-04 | 1.10e-04 | -1.42 | -0.00138 / -0.00094 | 0.1573 / 0.1512 | 0.1574 / 0.1512 |
| F2 | efg_jump2 | located_only_games | 10,133 | 0.12204 | loc_B1 | 0.12209 | -4.77e-05 | 1.16e-04 | -0.41 | -0.00138 / -0.00136 | 0.1573 / 0.1512 | 0.1573 / 0.1513 |
| F2 | efg_jump2 | located_only_games | 10,133 | 0.12204 | loc_B2 | 0.12206 | -1.81e-05 | 1.17e-04 | -0.16 | -0.00138 / -0.00156 | 0.1573 / 0.1512 | 0.1573 / 0.1513 |
| F2 | andone_jump2 | all_games | 10,665 | 0.06794 | loc_B0 | 0.06765 | +2.88e-04 | 9.09e-05 | +3.17 | +0.00945 / +0.01056 | 0.1128 / 0.0992 | 0.1127 / 0.0982 |
| F2 | andone_jump2 | all_games | 10,665 | 0.06794 | loc_B1 | 0.06790 | +3.85e-05 | 8.98e-05 | +0.43 | +0.00945 / +0.00993 | 0.1128 / 0.0992 | 0.1127 / 0.0994 |
| F2 | andone_jump2 | all_games | 10,665 | 0.06794 | loc_B2 | 0.06786 | +7.59e-05 | 9.07e-05 | +0.84 | +0.00945 / +0.00994 | 0.1128 / 0.0992 | 0.1126 / 0.0993 |
| F2 | andone_jump2 | located_only_games | 9,874 | 0.06047 | loc_B0 | 0.05998 | +4.85e-04 | 8.70e-05 | +5.57 | -0.00159 / -0.00064 | 0.1012 / 0.0998 | 0.1010 / 0.0989 |
| F2 | andone_jump2 | located_only_games | 9,874 | 0.06047 | loc_B1 | 0.06032 | +1.47e-04 | 9.47e-05 | +1.55 | -0.00159 / -0.00124 | 0.1012 / 0.0998 | 0.1010 / 0.1000 |
| F2 | andone_jump2 | located_only_games | 9,874 | 0.06047 | loc_B2 | 0.06028 | +1.85e-04 | 9.62e-05 | +1.92 | -0.00159 / -0.00123 | 0.1012 / 0.0998 | 0.1010 / 0.1000 |
| F2 | oreb_jump2 | all_games | 10,828 | 0.14713 | loc_B0 | 0.14703 | +9.80e-05 | 1.16e-04 | +0.84 | +0.00455 / +0.00502 | 0.1887 / 0.1822 | 0.1886 / 0.1822 |
| F2 | oreb_jump2 | all_games | 10,828 | 0.14713 | loc_B1 | 0.14708 | +5.04e-05 | 1.17e-04 | +0.43 | +0.00455 / +0.00484 | 0.1887 / 0.1822 | 0.1886 / 0.1823 |
| F2 | oreb_jump2 | all_games | 10,828 | 0.14713 | loc_B2 | 0.14708 | +4.79e-05 | 1.18e-04 | +0.41 | +0.00455 / +0.00480 | 0.1887 / 0.1822 | 0.1887 / 0.1823 |
| F2 | oreb_jump2 | located_only_games | 10,567 | 0.14723 | loc_B0 | 0.14715 | +7.73e-05 | 1.29e-04 | +0.60 | +0.00292 / +0.00342 | 0.1888 / 0.1826 | 0.1887 / 0.1827 |
| F2 | oreb_jump2 | located_only_games | 10,567 | 0.14723 | loc_B1 | 0.14720 | +2.96e-05 | 1.29e-04 | +0.23 | +0.00292 / +0.00324 | 0.1888 / 0.1826 | 0.1888 / 0.1828 |
| F2 | oreb_jump2 | located_only_games | 10,567 | 0.14723 | loc_B2 | 0.14720 | +2.66e-05 | 1.30e-04 | +0.20 | +0.00292 / +0.00320 | 0.1888 / 0.1826 | 0.1888 / 0.1828 |
| F2 | three_make_rate | all_games | 10,890 | 0.12448 | loc_B0 | 0.12439 | +9.28e-05 | 1.16e-04 | +0.80 | +0.00040 / -0.00003 | 0.1572 / 0.1543 | 0.1572 / 0.1544 |
| F2 | three_make_rate | all_games | 10,890 | 0.12448 | loc_B1 | 0.12434 | +1.34e-04 | 1.17e-04 | +1.15 | +0.00040 / +0.00003 | 0.1572 / 0.1543 | 0.1572 / 0.1544 |
| F2 | three_make_rate | all_games | 10,890 | 0.12448 | loc_B2 | 0.12434 | +1.37e-04 | 1.17e-04 | +1.17 | +0.00040 / +0.00003 | 0.1572 / 0.1543 | 0.1571 / 0.1544 |
| F2 | three_make_rate | located_only_games | 10,702 | 0.12429 | loc_B0 | 0.12421 | +7.64e-05 | 1.14e-04 | +0.67 | +0.00045 / +0.00004 | 0.1571 / 0.1543 | 0.1571 / 0.1543 |
| F2 | three_make_rate | located_only_games | 10,702 | 0.12429 | loc_B1 | 0.12417 | +1.18e-04 | 1.13e-04 | +1.04 | +0.00045 / +0.00009 | 0.1571 / 0.1543 | 0.1570 / 0.1543 |
| F2 | three_make_rate | located_only_games | 10,702 | 0.12429 | loc_B2 | 0.12417 | +1.19e-04 | 1.13e-04 | +1.05 | +0.00045 / +0.00009 | 0.1571 / 0.1543 | 0.1570 / 0.1543 |
| F2 | andone_three | all_games | 10,870 | 0.01129 | loc_B0 | 0.01132 | -3.62e-05 | 2.01e-05 | -1.80 | +0.00220 / +0.00218 | 0.0339 / 0.0278 | 0.0339 / 0.0280 |
| F2 | andone_three | all_games | 10,870 | 0.01129 | loc_B1 | 0.01133 | -3.88e-05 | 2.02e-05 | -1.92 | +0.00220 / +0.00218 | 0.0339 / 0.0278 | 0.0339 / 0.0280 |
| F2 | andone_three | all_games | 10,870 | 0.01129 | loc_B2 | 0.01133 | -3.93e-05 | 2.02e-05 | -1.95 | +0.00220 / +0.00218 | 0.0339 / 0.0278 | 0.0339 / 0.0280 |
| F2 | andone_three | located_only_games | 10,636 | 0.00981 | loc_B0 | 0.00985 | -3.31e-05 | 2.07e-05 | -1.60 | +0.00063 / +0.00061 | 0.0296 / 0.0278 | 0.0296 / 0.0280 |
| F2 | andone_three | located_only_games | 10,636 | 0.00981 | loc_B1 | 0.00985 | -3.57e-05 | 2.07e-05 | -1.72 | +0.00063 / +0.00060 | 0.0296 / 0.0278 | 0.0296 / 0.0280 |
| F2 | andone_three | located_only_games | 10,636 | 0.00981 | loc_B2 | 0.00985 | -3.62e-05 | 2.07e-05 | -1.75 | +0.00063 / +0.00060 | 0.0296 / 0.0278 | 0.0296 / 0.0280 |
| F2 | oreb_three | all_games | 10,890 | 0.10377 | loc_B0 | 0.10371 | +6.63e-05 | 1.08e-04 | +0.61 | +0.00956 / +0.00975 | 0.1306 / 0.1207 | 0.1305 / 0.1207 |
| F2 | oreb_three | all_games | 10,890 | 0.10377 | loc_B1 | 0.10369 | +8.31e-05 | 1.07e-04 | +0.78 | +0.00956 / +0.00976 | 0.1306 / 0.1207 | 0.1305 / 0.1207 |
| F2 | oreb_three | all_games | 10,890 | 0.10377 | loc_B2 | 0.10366 | +1.15e-04 | 1.07e-04 | +1.07 | +0.00956 / +0.00973 | 0.1306 / 0.1207 | 0.1305 / 0.1207 |
| F2 | oreb_three | located_only_games | 10,689 | 0.10368 | loc_B0 | 0.10364 | +4.87e-05 | 1.11e-04 | +0.44 | +0.00925 / +0.00947 | 0.1305 / 0.1207 | 0.1305 / 0.1207 |
| F2 | oreb_three | located_only_games | 10,689 | 0.10368 | loc_B1 | 0.10362 | +6.68e-05 | 1.11e-04 | +0.60 | +0.00925 / +0.00948 | 0.1305 / 0.1207 | 0.1305 / 0.1207 |
| F2 | oreb_three | located_only_games | 10,689 | 0.10368 | loc_B2 | 0.10359 | +9.89e-05 | 1.12e-04 | +0.88 | +0.00925 / +0.00945 | 0.1305 / 0.1207 | 0.1304 / 0.1207 |
| F2 | efg_jump2_three | all_games | 10,890 | 0.08723 | loc_B0 | 0.08722 | +8.92e-06 | 7.63e-05 | +0.12 | +0.00226 / +0.00215 | 0.1093 / 0.1074 | 0.1094 / 0.1074 |
| F2 | efg_jump2_three | all_games | 10,890 | 0.08723 | loc_B1 | 0.08718 | +4.68e-05 | 7.62e-05 | +0.61 | +0.00226 / +0.00213 | 0.1093 / 0.1074 | 0.1093 / 0.1074 |
| F2 | efg_jump2_three | all_games | 10,890 | 0.08723 | loc_B2 | 0.08717 | +5.34e-05 | 7.73e-05 | +0.69 | +0.00226 / +0.00204 | 0.1093 / 0.1074 | 0.1093 / 0.1074 |
| F2 | efg_jump2_three | located_only_games | 10,730 | 0.08721 | loc_B0 | 0.08722 | -9.16e-06 | 8.31e-05 | -0.11 | +0.00210 / +0.00201 | 0.1093 / 0.1074 | 0.1093 / 0.1074 |
| F2 | efg_jump2_three | located_only_games | 10,730 | 0.08721 | loc_B1 | 0.08719 | +2.66e-05 | 8.55e-05 | +0.31 | +0.00210 / +0.00199 | 0.1093 / 0.1074 | 0.1093 / 0.1074 |
| F2 | efg_jump2_three | located_only_games | 10,730 | 0.08721 | loc_B2 | 0.08718 | +3.24e-05 | 8.68e-05 | +0.37 | +0.00210 / +0.00190 | 0.1093 / 0.1074 | 0.1093 / 0.1074 |
| F1 | efg_jump2 | all_games | 10,353 | 0.11502 | loc_B0 | 0.11511 | -9.65e-05 | 1.22e-04 | -0.79 | +0.00273 / +0.00281 | 0.1471 / 0.1409 | 0.1471 / 0.1410 |
| F1 | efg_jump2 | all_games | 10,353 | 0.11502 | loc_B1 | 0.11510 | -8.65e-05 | 1.30e-04 | -0.67 | +0.00273 / +0.00251 | 0.1471 / 0.1409 | 0.1471 / 0.1410 |
| F1 | efg_jump2 | all_games | 10,353 | 0.11502 | loc_B2 | 0.11509 | -7.13e-05 | 1.31e-04 | -0.55 | +0.00273 / +0.00241 | 0.1471 / 0.1409 | 0.1470 / 0.1410 |
| F1 | efg_jump2 | located_only_games (BIASED SUBSET) | 8,666 | 0.11522 | loc_B0 | 0.11528 | -6.48e-05 | 1.25e-04 | -0.52 | +0.00081 / +0.00106 | 0.1472 / 0.1419 | 0.1471 / 0.1419 |
| F1 | efg_jump2 | located_only_games (BIASED SUBSET) | 8,666 | 0.11522 | loc_B1 | 0.11527 | -4.69e-05 | 1.28e-04 | -0.37 | +0.00081 / +0.00074 | 0.1472 / 0.1419 | 0.1471 / 0.1420 |
| F1 | efg_jump2 | located_only_games (BIASED SUBSET) | 8,666 | 0.11522 | loc_B2 | 0.11526 | -4.12e-05 | 1.28e-04 | -0.32 | +0.00081 / +0.00063 | 0.1472 / 0.1419 | 0.1471 / 0.1420 |
| F1 | andone_jump2 | all_games | 10,201 | 0.06670 | loc_B0 | 0.06669 | +1.36e-05 | 9.35e-05 | +0.15 | +0.01174 / +0.01223 | 0.1087 / 0.0927 | 0.1087 / 0.0921 |
| F1 | andone_jump2 | all_games | 10,201 | 0.06670 | loc_B1 | 0.06691 | -2.11e-04 | 9.78e-05 | -2.15 | +0.01174 / +0.01172 | 0.1087 / 0.0927 | 0.1087 / 0.0930 |
| F1 | andone_jump2 | all_games | 10,201 | 0.06670 | loc_B2 | 0.06686 | -1.55e-04 | 9.63e-05 | -1.61 | +0.01174 / +0.01176 | 0.1087 / 0.0927 | 0.1087 / 0.0929 |
| F1 | andone_jump2 | located_only_games (BIASED SUBSET) | 8,450 | 0.05784 | loc_B0 | 0.05783 | +9.75e-06 | 9.95e-05 | +0.10 | +0.00041 / +0.00085 | 0.0949 / 0.0931 | 0.0950 / 0.0925 |
| F1 | andone_jump2 | located_only_games (BIASED SUBSET) | 8,450 | 0.05784 | loc_B1 | 0.05809 | -2.54e-04 | 1.02e-04 | -2.50 | +0.00041 / +0.00036 | 0.0949 / 0.0931 | 0.0950 / 0.0934 |
| F1 | andone_jump2 | located_only_games (BIASED SUBSET) | 8,450 | 0.05784 | loc_B2 | 0.05806 | -2.21e-04 | 1.01e-04 | -2.19 | +0.00041 / +0.00038 | 0.0949 / 0.0931 | 0.0950 / 0.0934 |
| F1 | oreb_jump2 | all_games | 10,336 | 0.13692 | loc_B0 | 0.13696 | -3.83e-05 | 1.44e-04 | -0.27 | +0.00444 / +0.00363 | 0.1761 / 0.1702 | 0.1761 / 0.1704 |
| F1 | oreb_jump2 | all_games | 10,336 | 0.13692 | loc_B1 | 0.13697 | -4.58e-05 | 1.47e-04 | -0.31 | +0.00444 / +0.00347 | 0.1761 / 0.1702 | 0.1761 / 0.1705 |
| F1 | oreb_jump2 | all_games | 10,336 | 0.13692 | loc_B2 | 0.13696 | -3.07e-05 | 1.48e-04 | -0.21 | +0.00444 / +0.00351 | 0.1761 / 0.1702 | 0.1761 / 0.1705 |
| F1 | oreb_jump2 | located_only_games (BIASED SUBSET) | 9,009 | 0.13679 | loc_B0 | 0.13676 | +3.14e-05 | 1.48e-04 | +0.21 | +0.00221 / +0.00148 | 0.1760 / 0.1714 | 0.1759 / 0.1716 |
| F1 | oreb_jump2 | located_only_games (BIASED SUBSET) | 9,009 | 0.13679 | loc_B1 | 0.13679 | +4.43e-06 | 1.48e-04 | +0.03 | +0.00221 / +0.00131 | 0.1760 / 0.1714 | 0.1759 / 0.1717 |
| F1 | oreb_jump2 | located_only_games (BIASED SUBSET) | 9,009 | 0.13679 | loc_B2 | 0.13677 | +2.00e-05 | 1.48e-04 | +0.14 | +0.00221 / +0.00133 | 0.1760 / 0.1714 | 0.1759 / 0.1717 |
| F1 | three_make_rate | all_games | 10,356 | 0.12554 | loc_B0 | 0.12532 | +2.24e-04 | 1.42e-04 | +1.58 | +0.00178 / +0.00136 | 0.1587 / 0.1577 | 0.1583 / 0.1577 |
| F1 | three_make_rate | all_games | 10,356 | 0.12554 | loc_B1 | 0.12527 | +2.73e-04 | 1.44e-04 | +1.90 | +0.00178 / +0.00140 | 0.1587 / 0.1577 | 0.1582 / 0.1577 |
| F1 | three_make_rate | all_games | 10,356 | 0.12554 | loc_B2 | 0.12530 | +2.38e-04 | 1.42e-04 | +1.67 | +0.00178 / +0.00145 | 0.1587 / 0.1577 | 0.1583 / 0.1577 |
| F1 | three_make_rate | located_only_games (BIASED SUBSET) | 9,063 | 0.12536 | loc_B0 | 0.12514 | +2.19e-04 | 1.45e-04 | +1.51 | +0.00099 / +0.00044 | 0.1584 / 0.1577 | 0.1580 / 0.1578 |
| F1 | three_make_rate | located_only_games (BIASED SUBSET) | 9,063 | 0.12536 | loc_B1 | 0.12509 | +2.62e-04 | 1.47e-04 | +1.79 | +0.00099 / +0.00045 | 0.1584 / 0.1577 | 0.1579 / 0.1578 |
| F1 | three_make_rate | located_only_games (BIASED SUBSET) | 9,063 | 0.12536 | loc_B2 | 0.12513 | +2.32e-04 | 1.48e-04 | +1.57 | +0.00099 / +0.00048 | 0.1584 / 0.1577 | 0.1580 / 0.1578 |
| F1 | andone_three | all_games | 10,338 | 0.00982 | loc_B0 | 0.00983 | -1.04e-05 | 2.04e-05 | -0.51 | +0.00266 / +0.00266 | 0.0334 / 0.0250 | 0.0334 / 0.0251 |
| F1 | andone_three | all_games | 10,338 | 0.00982 | loc_B1 | 0.00983 | -1.07e-05 | 2.03e-05 | -0.53 | +0.00266 / +0.00266 | 0.0334 / 0.0250 | 0.0334 / 0.0251 |
| F1 | andone_three | all_games | 10,338 | 0.00982 | loc_B2 | 0.00983 | -8.75e-06 | 2.03e-05 | -0.43 | +0.00266 / +0.00266 | 0.0334 / 0.0250 | 0.0334 / 0.0251 |
| F1 | andone_three | located_only_games (BIASED SUBSET) | 9,041 | 0.00843 | loc_B0 | 0.00843 | -3.79e-06 | 2.02e-05 | -0.19 | +0.00147 / +0.00147 | 0.0305 / 0.0247 | 0.0305 / 0.0247 |
| F1 | andone_three | located_only_games (BIASED SUBSET) | 9,041 | 0.00843 | loc_B1 | 0.00843 | -3.87e-06 | 2.02e-05 | -0.19 | +0.00147 / +0.00147 | 0.0305 / 0.0247 | 0.0305 / 0.0247 |
| F1 | andone_three | located_only_games (BIASED SUBSET) | 9,041 | 0.00843 | loc_B2 | 0.00843 | -1.43e-06 | 2.02e-05 | -0.07 | +0.00147 / +0.00148 | 0.0305 / 0.0247 | 0.0305 / 0.0247 |
| F1 | oreb_three | all_games | 10,356 | 0.10522 | loc_B0 | 0.10532 | -1.08e-04 | 1.28e-04 | -0.85 | +0.00362 / +0.00355 | 0.1332 / 0.1229 | 0.1333 / 0.1229 |
| F1 | oreb_three | all_games | 10,356 | 0.10522 | loc_B1 | 0.10533 | -1.13e-04 | 1.29e-04 | -0.88 | +0.00362 / +0.00361 | 0.1332 / 0.1229 | 0.1333 / 0.1229 |
| F1 | oreb_three | all_games | 10,356 | 0.10522 | loc_B2 | 0.10530 | -8.29e-05 | 1.29e-04 | -0.64 | +0.00362 / +0.00361 | 0.1332 / 0.1229 | 0.1332 / 0.1229 |
| F1 | oreb_three | located_only_games (BIASED SUBSET) | 9,035 | 0.10508 | loc_B0 | 0.10516 | -8.41e-05 | 1.34e-04 | -0.63 | +0.00285 / +0.00292 | 0.1329 / 0.1230 | 0.1330 / 0.1231 |
| F1 | oreb_three | located_only_games (BIASED SUBSET) | 9,035 | 0.10508 | loc_B1 | 0.10518 | -9.87e-05 | 1.35e-04 | -0.73 | +0.00285 / +0.00295 | 0.1329 / 0.1230 | 0.1330 / 0.1231 |
| F1 | oreb_three | located_only_games (BIASED SUBSET) | 9,035 | 0.10508 | loc_B2 | 0.10515 | -6.88e-05 | 1.36e-04 | -0.51 | +0.00285 / +0.00294 | 0.1329 / 0.1230 | 0.1330 / 0.1231 |
| F1 | efg_jump2_three | all_games | 10,356 | 0.08685 | loc_B0 | 0.08669 | +1.64e-04 | 1.01e-04 | +1.62 | +0.00408 / +0.00386 | 0.1086 / 0.1058 | 0.1083 / 0.1059 |
| F1 | efg_jump2_three | all_games | 10,356 | 0.08685 | loc_B1 | 0.08665 | +2.01e-04 | 1.04e-04 | +1.93 | +0.00408 / +0.00384 | 0.1086 / 0.1058 | 0.1082 / 0.1059 |
| F1 | efg_jump2_three | all_games | 10,356 | 0.08685 | loc_B2 | 0.08668 | +1.70e-04 | 1.04e-04 | +1.64 | +0.00408 / +0.00381 | 0.1086 / 0.1058 | 0.1083 / 0.1059 |
| F1 | efg_jump2_three | located_only_games (BIASED SUBSET) | 9,112 | 0.08718 | loc_B0 | 0.08701 | +1.62e-04 | 1.02e-04 | +1.59 | +0.00371 / +0.00349 | 0.1090 / 0.1061 | 0.1087 / 0.1061 |
| F1 | efg_jump2_three | located_only_games (BIASED SUBSET) | 9,112 | 0.08718 | loc_B1 | 0.08697 | +2.04e-04 | 1.02e-04 | +2.00 | +0.00371 / +0.00344 | 0.1090 / 0.1061 | 0.1086 / 0.1061 |
| F1 | efg_jump2_three | located_only_games (BIASED SUBSET) | 9,112 | 0.08718 | loc_B2 | 0.08700 | +1.74e-04 | 1.03e-04 | +1.69 | +0.00371 / +0.00340 | 0.1090 / 0.1061 | 0.1087 / 0.1061 |

### T10. Read C responsiveness, all games: implied vs realised team-game rate by quintile of the as-of profile -- slope ratio (monotone steps) [P/F], realised Q1-Q5 span

| fold | target | arm | off_short2 | off_corner3 | off_wing3 | off_top3 | def_short2 | def_corner3 | def_wing3 | def_top3 |
|---|---|---|---|---|---|---|---|---|---|---|
| F2 | efg_jump2 | R | 0.3459 (2) [F] +2.55pp | -0.3881 (2) [F] +0.40pp | 0.2906 (3) [P] -1.81pp | 0.8551 (3) [P] +0.48pp | 0.4883 (3) [F] +2.12pp | 1.8473 (3) [P] +0.34pp | 0.3942 (4) [P] +1.32pp | 0.876 (4) [P] -1.31pp |
| F2 | efg_jump2 | loc_B2 | 0.629 (3) [F] +2.55pp | -0.1112 (2) [F] +0.40pp | 0.2588 (3) [P] -1.81pp | 0.5881 (3) [P] +0.48pp | 0.6738 (4) [F] +2.12pp | 1.9505 (3) [P] +0.34pp | 0.3344 (4) [P] +1.32pp | 0.8602 (4) [P] -1.31pp |
| F2 | andone_jump2 | R | 0.4126 (2) [F] +1.53pp | -0.4452 (2) [F] +0.35pp | 1.5626 (2) [F] -0.17pp | -1.4929 (1) [F] -0.20pp | -0.0472 (2) [F] +0.84pp | -0.4283 (1) [F] +0.32pp | -0.5255 (1) [F] +0.21pp | -0.4539 (1) [F] -0.45pp |
| F2 | andone_jump2 | loc_B2 | 0.8924 (4) [P] +1.53pp | -0.1319 (3) [P] +0.35pp | 1.6865 (3) [P] -0.17pp | -1.2574 (0) [F] -0.20pp | 0.3885 (4) [P] +0.84pp | -0.3532 (1) [F] +0.32pp | -0.6297 (2) [F] +0.21pp | -0.4214 (0) [F] -0.45pp |
| F2 | oreb_jump2 | R | 1.2863 (3) [P] +0.89pp | 1.775 (3) [P] +0.17pp | 0.0176 (2) [F] +0.80pp | 0.4421 (2) [F] -0.72pp | 0.6555 (3) [P] -0.23pp | 0.8149 (4) [P] +1.67pp | 1.4911 (3) [P] +0.62pp | 0.7072 (4) [F] -2.61pp |
| F2 | oreb_jump2 | loc_B2 | 2.0158 (4) [P] +0.89pp | 1.8499 (3) [P] +0.17pp | -0.0005 (2) [F] +0.80pp | 0.4021 (2) [F] -0.72pp | -0.9046 (2) [F] -0.23pp | 0.8453 (4) [P] +1.67pp | 1.2934 (3) [P] +0.62pp | 0.702 (4) [F] -2.61pp |
| F2 | three_make_rate | R | 1.8161 (3) [P] +0.63pp | 0.8202 (2) [F] -0.28pp | 0.3351 (3) [P] -1.23pp | 0.405 (3) [P] +1.06pp | 0.8304 (3) [P] +1.05pp | 0.2966 (3) [P] +1.34pp | 0.6763 (3) [P] +0.57pp | 0.8606 (4) [P] -1.02pp |
| F2 | three_make_rate | loc_B2 | 1.8582 (3) [P] +0.63pp | -0.2875 (2) [F] -0.28pp | 0.3525 (4) [P] -1.23pp | 0.1539 (3) [P] +1.06pp | 0.8367 (3) [P] +1.05pp | 0.4755 (3) [P] +1.34pp | 0.7267 (3) [P] +0.57pp | 1.0741 (4) [P] -1.02pp |
| F2 | andone_three | R | 0.1298 (1) [F] +0.06pp | -0.0721 (2) [F] -0.09pp | -0.0186 (1) [F] -0.11pp | -0.0115 (2) [F] +0.15pp | 1.0012 (3) [P] -0.04pp | 0.4481 (1) [F] -0.03pp | 0.4179 (2) [F] -0.06pp | -0.4876 (1) [F] -0.06pp |
| F2 | andone_three | loc_B2 | 0.104 (2) [F] +0.06pp | -0.0811 (2) [F] -0.09pp | 0.0453 (2) [F] -0.11pp | 0.0181 (2) [F] +0.15pp | 0.6246 (3) [P] -0.04pp | 0.5363 (3) [P] -0.03pp | 0.3889 (1) [F] -0.06pp | -0.4592 (1) [F] -0.06pp |
| F2 | oreb_three | R | 1.0774 (3) [P] +0.66pp | 1.5892 (3) [P] +0.21pp | -1.6025 (3) [P] +0.09pp | 0.3098 (2) [F] -0.34pp | -1.6239 (1) [F] +0.13pp | 0.4075 (4) [F] +2.96pp | 1.0127 (3) [P] +0.96pp | 0.5223 (4) [F] -3.43pp |
| F2 | oreb_three | loc_B2 | 1.082 (3) [P] +0.66pp | 2.2759 (3) [P] +0.21pp | -0.2824 (3) [P] +0.09pp | 1.0018 (2) [F] -0.34pp | -0.831 (1) [F] +0.13pp | 0.4582 (4) [F] +2.96pp | 0.9392 (3) [P] +0.96pp | 0.5541 (4) [F] -3.43pp |
| F2 | efg_jump2_three | R | 0.8796 (4) [P] +2.64pp | 3.0465 (2) [F] -0.09pp | 0.2775 (3) [P] -1.37pp | 0.5215 (3) [P] +0.86pp | 0.6355 (3) [P] +1.44pp | 0.6761 (3) [P] +1.49pp | 0.5451 (3) [P] +0.92pp | 0.8952 (4) [P] -1.65pp |
| F2 | efg_jump2_three | loc_B2 | 0.989 (4) [P] +2.64pp | 0.4537 (2) [F] -0.09pp | 0.2712 (3) [P] -1.37pp | 0.2735 (3) [P] +0.86pp | 0.7421 (4) [P] +1.44pp | 0.7875 (3) [P] +1.49pp | 0.5322 (4) [P] +0.92pp | 0.9745 (4) [P] -1.65pp |
| F1 | efg_jump2 | R | 0.3957 (3) [P] +1.49pp | -0.8076 (1) [F] +0.19pp | 0.5547 (4) [P] -0.97pp | 0.6686 (3) [P] +0.78pp | 0.656 (4) [P] +1.65pp | -1.6018 (2) [F] -0.19pp | 0.0876 (2) [F] -0.28pp | -0.9615 (1) [F] +0.26pp |
| F1 | efg_jump2 | loc_B2 | 0.8886 (4) [P] +1.49pp | -0.2971 (1) [F] +0.19pp | 0.6252 (4) [P] -0.97pp | 0.5935 (2) [F] +0.78pp | 0.8068 (4) [P] +1.65pp | -1.3024 (2) [F] -0.19pp | -0.0269 (3) [P] -0.28pp | -0.7493 (1) [F] +0.26pp |
| F1 | andone_jump2 | R | 0.4767 (3) [P] +1.04pp | 0.881 (3) [P] +0.16pp | -1.1258 (0) [F] +0.36pp | -0.3455 (2) [F] -0.62pp | 0.0472 (3) [P] +1.02pp | 0.0468 (3) [P] +0.41pp | -1.1698 (2) [F] +0.09pp | -0.3231 (2) [F] -0.28pp |
| F1 | andone_jump2 | loc_B2 | 1.1577 (4) [P] +1.04pp | 1.5152 (3) [P] +0.16pp | -1.1306 (1) [F] +0.36pp | -0.2595 (2) [F] -0.62pp | 0.3574 (3) [P] +1.02pp | -0.016 (3) [P] +0.41pp | -0.8995 (2) [F] +0.09pp | -0.2872 (1) [F] -0.28pp |
| F1 | oreb_jump2 | R | 1.0136 (3) [P] +0.83pp | 0.5499 (2) [F] +0.87pp | -5.653 (2) [F] -0.08pp | 0.6061 (3) [P] -1.57pp | 0.4346 (3) [P] +1.50pp | 0.6552 (4) [F] +2.79pp | 1.3138 (4) [P] +0.86pp | 0.7122 (4) [F] -3.22pp |
| F1 | oreb_jump2 | loc_B2 | 1.7136 (3) [P] +0.83pp | 0.5315 (2) [F] +0.87pp | -7.5711 (2) [F] -0.08pp | 0.6419 (3) [P] -1.57pp | 0.5998 (4) [P] +1.50pp | 0.6397 (4) [F] +2.79pp | 1.3514 (4) [P] +0.86pp | 0.7217 (4) [F] -3.22pp |
| F1 | three_make_rate | R | 0.5058 (4) [F] +2.32pp | 0.4456 (3) [P] +0.46pp | 1.6356 (2) [F] -0.18pp | 1.3158 (2) [F] +0.28pp | 1.3189 (3) [P] +0.77pp | 3.982 (2) [F] +0.08pp | 0.0414 (2) [F] +0.34pp | 72.0256 (3) [P] -0.01pp |
| F1 | three_make_rate | loc_B2 | 0.4764 (4) [F] +2.32pp | 1.1053 (3) [P] +0.46pp | 1.4121 (2) [F] -0.18pp | 0.1805 (2) [F] +0.28pp | 1.2878 (3) [P] +0.77pp | 6.849 (3) [P] +0.08pp | 0.1248 (1) [F] +0.34pp | 123.3866 (3) [P] -0.01pp |
| F1 | andone_three | R | 7.1565 (2) [F] +0.01pp | 3.8655 (2) [F] +0.00pp | 0.9067 (4) [P] -0.06pp | 0.4288 (3) [P] +0.07pp | -1.7511 (1) [F] +0.03pp | 0.3169 (3) [P] -0.18pp | -0.3379 (1) [F] +0.08pp | 0.5184 (4) [P] +0.15pp |
| F1 | andone_three | loc_B2 | 6.6043 (2) [F] +0.01pp | 2.5029 (2) [F] +0.00pp | 0.6771 (4) [P] -0.06pp | 0.3592 (3) [P] +0.07pp | -2.0101 (1) [F] +0.03pp | 0.3138 (3) [P] -0.18pp | -0.3145 (1) [F] +0.08pp | 0.485 (4) [P] +0.15pp |
| F1 | oreb_three | R | 0.0159 (1) [F] -0.21pp | 14.4899 (3) [P] +0.05pp | 0.3267 (2) [F] +0.24pp | 8.2547 (2) [F] -0.08pp | 1.4948 (2) [F] +0.20pp | 0.5226 (4) [F] +3.16pp | 0.7344 (4) [P] +1.38pp | 0.5627 (4) [F] -3.63pp |
| F1 | oreb_three | loc_B2 | -0.262 (1) [F] -0.21pp | 16.7185 (3) [P] +0.05pp | 0.8647 (2) [F] +0.24pp | 10.6374 (3) [P] -0.08pp | 1.3424 (2) [F] +0.20pp | 0.5576 (4) [F] +3.16pp | 0.7385 (4) [P] +1.38pp | 0.5992 (4) [F] -3.63pp |
| F1 | efg_jump2_three | R | 0.6762 (4) [F] +3.04pp | 0.3698 (3) [P] +0.45pp | 0.8619 (2) [F] -0.72pp | 0.9278 (2) [F] +0.67pp | 0.921 (3) [P] +1.15pp | 1.7225 (4) [P] +0.49pp | 0.5392 (2) [F] +0.20pp | 2.0584 (3) [P] -0.40pp |
| F1 | efg_jump2_three | loc_B2 | 0.7614 (4) [F] +3.04pp | 0.8509 (3) [P] +0.45pp | 0.8681 (2) [F] -0.72pp | 0.614 (2) [F] +0.67pp | 0.9967 (3) [P] +1.15pp | 1.9722 (4) [P] +0.49pp | 0.6849 (2) [F] +0.20pp | 2.3905 (3) [P] -0.40pp |

- F2 oreb_jump2: rebound rows of the class in the test season 95,844; used (joined to an fg design row) 89,970
- F2 oreb_three: rebound rows of the class in the test season 166,407; used (joined to an fg design row) 162,259
- F1 oreb_jump2: rebound rows of the class in the test season 101,961; used (joined to an fg design row) 93,212
- F1 oreb_three: rebound rows of the class in the test season 158,635; used (joined to an fg design row) 148,195

### T11. Diagnostic 1: rim override (JumpShot within 2.27 ft -> FGA_rim) on located JumpShot attempts, makes vs misses

| season | located JumpShot | override % makes | override % misses | within-game diff pp | game-bootstrap SE pp | games with both outcomes |
|---|---|---|---|---|---|---|
| 2022 | 350,238 | 1.482 | 0.556 | +0.848 | 0.055 | 5,428 |
| 2023 | 380,032 | 1.667 | 0.539 | +1.053 | 0.057 | 5,801 |
| 2024 | 405,001 | 1.740 | 0.637 | +0.980 | 0.051 | 5,925 |
| 2025 | 457,788 | 4.947 | 1.430 | +3.340 | 0.076 | 6,124 |

### T12. Diagnostic 2: hoopR vs CBBD bands, 2025 from 2025-02-10 (118,417 jump2/three attempts, 114,346 located by both)

| class | pairs | band differs % | confusion (CBBD band: hoopR band counts) |
|---|---|---|---|
| jump2 | 43,025 | 3.479 | {"long2": {"long2": 23972, "short2": 1102}, "short2": {"long2": 395, "short2": 17556}} |
| three | 71,321 | 4.424 | {"corner3": {"corner3": 13683, "top3": 5, "wing3": 597}, "top3": {"corner3": 94, "top3": 24804, "wing3": 840}, "wing3": {"corner3": 1425, "top3": 194, "wing3": 29679}} |

| class | band | team-games (>= 5 attempts) | mean share CBBD | mean share hoopR | MAE pp | corr |
|---|---|---|---|---|---|---|
| jump2 | short2 | 3,065 | 0.4387 | 0.4219 | 3.148 | 0.9362 |
| jump2 | long2 | 3,065 | 0.5613 | 0.5781 | 3.148 | 0.9362 |
| three | corner3 | 3,179 | 0.2126 | 0.1999 | 2.459 | 0.9276 |
| three | wing3 | 3,179 | 0.4388 | 0.4415 | 3.241 | 0.8981 |
| three | top3 | 3,179 | 0.3486 | 0.3586 | 1.467 | 0.9144 |

### T13. Diagnostic 4: split-half reliability on this round's design table vs the registered numbers

| side | band | team-seasons | split-half r | Spearman-Brown (this round) | registered |
|---|---|---|---|---|---|
| offence | short2 | 1,310 | 0.790 | 0.882 | 0.89 |
| offence | corner3 | 1,410 | 0.616 | 0.762 | 0.79 |
| offence | wing3 | 1,410 | 0.437 | 0.608 | 0.67 |
| offence | top3 | 1,410 | 0.615 | 0.762 | 0.80 |
| defence | short2 | 1,358 | 0.503 | 0.669 | 0.69 |
| defence | corner3 | 1,424 | 0.550 | 0.710 | 0.76 |
| defence | wing3 | 1,424 | 0.301 | 0.462 | 0.52 |
| defence | top3 | 1,424 | 0.545 | 0.706 | 0.75 |
