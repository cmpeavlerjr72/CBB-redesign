# Season-drift anchor: cross-model experiments (append-only)

Owner lane: C (2026-09-30). Sub-models under test: `rebound`, `shot_block`,
free-throw technical rate (`ft_tech`). Control: `possession_outcome` (`po`,
`first` population). Offline only. NOTHING in this file adopts anything or
changes a served default; the PM decides from the results section.

Sources read before writing (not re-derived): `docs/models/rebound/experiments.md`
sections 9-11, `docs/models/shot_block/experiments.md`,
`docs/models/free_throw/experiments.md` sections 9-10,
`docs/tests/rebound_round_drift_block_carry_2026-09-18.md`, HANDOFF.md
"Cross-cutting findings" 1 and 2.

---

## 1. Pre-registration: round 1 (written 2026-09-30 ~11:05 EDT, COMMITTED BEFORE ANY FIT)

### 1.1 The question

League levels drift season over season (live OREB% 0.2824 -> 0.2868 -> 0.2916 ->
0.2992; rim block level -0.8 pp in every shot_block arm; technical trips
1,281 / 2,082 / 1,342 / 1,204 across 2022-2025, i.e. a spike and not a trend).
A pooled fit averages the drift away and misses the held-out level. The round
asks which ONE anchor design, applied identically to every sub-model, fixes the
held-out level without (a) failing when drift stops or reverses, (b) flattening
per-team responsiveness, or (c) compressing the spread of team predictions.

### 1.2 The anchor object (shared by every arm that uses one)

For a sub-model with target rate `r` (rebound: OREB / (OREB + DREB) on live
misses; shot_block: blocked share of missed FGA; ft_tech: verified technical
trips per team-chance exposure; po: the six class shares, one anchor per class),
per season `s` and game date `t`:

- `L_end(s-1)`: the realised league level of the previous COMPLETED season.
  For the first panel season (2022, train-only in both folds) there is no
  previous season in the panel; its day-0 value is the fold's pooled TRAIN level
  (a train-only constant; never touches a test row).
- `L_asof(s, t)`: the league's cumulative rate in season `s` over games
  strictly before date `t`; on day 0 (no prior games) it equals `L_end(s-1)`.
- `L_blend(s, t; n0) = (num_before + n0 * L_end(s-1)) / (den_before + n0)`:
  the prior season's end level carried forward and updated in season, with
  `n0` pseudo-units (rows / exposures) FITTED on the fold's TRAIN seasons that
  have a previous season (F1: 2023; F2: 2023 and 2024) by maximising the
  anchor-only likelihood of those seasons' rows (binomial; multinomial for po;
  Poisson for ft_tech), over the grid `n0 in {0, 1e1, 3e1, 1e2, ..., 1e8, inf}`
  (half-decades). `n0 = inf` is pure carry; `n0 = 0` is `L_asof`.
- `T(s)`: OLS of season level on season index over the fold's TRAIN seasons,
  extrapolated to `s` (F1 has only two train seasons: a two-point line).
- `Lbar`: the fold's pooled TRAIN level; every offset below is centred on it
  (a constant shift the model absorbs; it only keeps offsets near zero).

Every column is built from rows strictly before the row's own game date, or
from completed prior seasons, or from train-fold constants. 2025-26 (2026) is
SEALED: `assert_not_sealed` on every load.

### 1.3 Arms (identical in every sub-model)

| arm | design | how it enters the model | simplicity rank |
|---|---|---|---|
| `R` | pooled reference: the sub-model's reference model, unchanged | - | 0 |
| `C0` | prior-season level carry, NO in-season update (`L_end(s-1)`, i.e. `n0 = inf`) | link-scale OFFSET | 1 |
| `O` | target RELATIVE to the as-of league level (`L_asof`, day 0 = prior-season end) | link-scale OFFSET: the model predicts the deviation | 2 |
| `P` | prior-season level carried with an in-season update whose weight `n0` is FITTED on train folds (`L_blend`) | link-scale OFFSET | 3 |
| `W` | recency-weighted training: `0.5 ** (age_days / 365)` with age measured to the first test date (PO's S2 convention). Half-life FIXED at 365 d a priori, not tuned | sample weights | 4 |
| `F` | as-of league level as a FEATURE (`link(L_asof) - link(Lbar)`, day 0 = prior-season end; fixes rebound A4's day-0 = 0.0 defect) | added input column(s) | 5 |
| `T` | CAUTIONARY trend extrapolation (`T(s)`), rebound round 3's A5 generalised | link-scale OFFSET | 6 |

Link scale: logit for binary targets; for rebound's 3-class model the offset
is added to the OREB raw score only (DREB is the reference class for the live
split; DEAD untouched), exactly A5's mechanism; for po's 6-class multinomial
the offset vector is `log L_c(t) - log Lbar_c` per class; for ft_tech the
Poisson log link (on top of `log(exposure)`). `F` adds one column (po: six,
one per class). Offsets are applied identically at fit time (LightGBM
`init_score`; GLM offset) and at prediction time.

NOT run this round, reported NOT RUN with resume commands: the faster refit
cadence (`S1_weekly`, ~3 h per rebound cell) and any closed loop.

### 1.4 Reference models per sub-model (unchanged hyperparameters across arms)

- `rebound`: served `lgbm / C_plus_state` (16 features incl. `blocked_f` true
  flag), `RB.LgbmArm.PARAMS`, design `rebound/round3/design_round3.parquet`,
  `S0` static fit. Must reproduce round 3's `A0B0C0` (F2 log loss 0.645565).
- `shot_block`: `K2` (standardised L2 logistic, `C = 1`, feature list `KC`,
  22 inputs), the round-1 best arm. Refitted with an offset-capable L-BFGS
  logistic that minimises the same objective as sklearn's (`sum logloss +
  0.5 ||w||^2`, intercept unpenalised); `R` must reproduce sklearn `K2`'s F2
  log loss (0.263240) to 1e-5.
- `ft_tech`: no adopted arm exists. Reference = a Poisson GLM on team-game
  rows (target: verified technical trips committed by the team, round 1b's
  `technical_target_verified_trips_v1.parquet`; exposure: team-chances,
  `possessions_v2`), log link, offset `log(exposure)`, inputs: team in-season
  as-of observed/expected ratio (log, 5 pseudo-trips toward 1, expected at the
  in-season league as-of rate, 0 on day 0), team prior-season
  observed/expected ratio (log, 5 pseudo-trips), `site_home`, `site_away`. The
  5-trip constant is part of the shared feature builder, identical in every
  arm, fixed now and not tuned.
- `po` (control): `lgbm / C_plus_state` (G0), `first` population,
  `design_v4.parquet`, `S0` static fit.

### 1.5 One blind grader

`scripts/exp_season_drift_anchor_v1.py` fits every (sub-model, fold, arm,
seed) cell and writes predictions only; `scripts/grade_season_drift_anchor_v1.py`
scores every cell with the same functions (no arm-specific path). For every
cell it reports:

1. primary: fold-test log loss (rebound 3-class, shot_block binary, po
   6-class); ft_tech: Poisson deviance per team-game.
2. held-out LEVEL gap: predicted minus realised test-season level (pp;
   ft_tech also in relative %). po: per class, and the max |gap|.
3. NOVEMBER-DECEMBER level gap (the launch condition).
4. level by month, by sub-type (rebound miss type, shot_block shot type, po
   class), by site; per-game level MAE where games carry >= 10 rows.
5. DRIFT-STOPS / REVERSES robustness: rate x season cells (rebound 4 miss
   types, shot_block 3 shot types, ft_tech overall, po 6 classes; test seasons
   2024 and 2025) are classified from realised levels as TREND-CONTINUED if the
   change from the last train season to the test season has the sign of the
   train-season OLS slope AND at least half its magnitude; otherwise
   FLAT-OR-REVERSED. Every arm is scored on the FLAT-OR-REVERSED cells by the
   mean of `|relative gap_arm| - |relative gap_R|` (relative gap = gap / realised
   level), per sub-model and pooled over all four models and both folds.
6. per-team prior-quintile RESPONSIVENESS slope: teams bucketed by their
   previous season's realised rate (rebound: offence; shot_block: defence;
   ft_tech: offending team; po: offence, each class), span of mean predicted over
   span of mean realised, monotone steps; overall and Nov-Dec.
7. SPREAD: SD across teams of the team's mean prediction over SD across teams of
   the realised team rate; raw, and noise-corrected (realised variance minus
   the mean binomial/Poisson sampling variance); teams with >= 50 rows (ft_tech:
   all team-seasons).
8. the sub-model's own calibration gate: worst decile |gap| <= 2.0 pp
   (rebound: `RB.score`'s gate; shot_block: 2.0 pp; po: worst class decile gap,
   reported only); ft_tech: level within 10% relative (round 1b line 2).
9. paired game-cluster block bootstrap (200 reps, seed 12345) SE of the
   (arm - R) primary difference.

### 1.6 Noise floor

A spec-identical retrain of `R` under seed 1 on F2 for the stochastic models
(`rebound`, `po`: LightGBM with row subsampling). `shot_block` and `ft_tech`
are deterministic (L-BFGS / IRLS), so a reseed is identically zero; for them
the floor is the paired bootstrap. Operative floor per cell =
max(reseed |delta primary|, published floor where one exists (rebound
6.7e-05, po 8.04e-04), 2 x paired bootstrap SE of (arm - R)).

### 1.7 Decision rule (mechanical)

Per sub-model `m` in {rebound, shot_block, ft_tech}, an arm is ELIGIBLE iff,
on F2 (selection):

- E1 primary beats `R` by more than the operative floor;
- E2 |held-out level gap| < |`R`'s|;
- E3 |Nov-Dec level gap| <= |`R`'s| + tol (tol: 0.10 pp binary; 2% relative
  ft_tech);
- E4 drift-stops: the POOLED flat-or-reversed score (all models, both folds)
  <= +0.005 (half a percent of level). A design failing E4 is ineligible in
  EVERY sub-model ("works only while the trend continues");
- E5 team slope ratio (overall and Nov-Dec) >= `R`'s - 0.05, and noise-corrected
  SD ratio >= `R`'s - 0.05 (no spread compression);
- E6 does not newly fail the sub-model's own calibration gate;
- E7 FOLD 1 CONFIRMS: F1 primary better than `R`'s, and F1 |level gap| <=
  `R`'s + tol.

Selection within `m`: the eligible arm with the best F2 primary; any eligible
arm within one operative floor of it with a lower simplicity rank wins the tie.
If no arm is eligible, `m` selects `R` (no anchor).

Cross-model: a DESIGN wins the round iff it is the selection in at least TWO
of the three sub-models AND the po control does not degrade on F2: po primary
not worse than po `R` by more than the po floor, po max class |level gap| not
worse than `R`'s by more than 0.10 pp, and no po class slope ratio below `R`'s
- 0.05. Otherwise the round has NO winning design, stated as such. `T` is
eligible under the same rule but is labelled cautionary in every table; if it
wins, the result is reported as an extrapolation win.

### 1.8 What this round may not do

No post-hoc multiplier, cap, clip or blend on any model output: the anchor is
an input/offset of the model at fit time, and the same offset is applied at fit
and predict time. No served default changes. No engine file is touched. Stage 2
(`S1_weekly`) and closed loops are out of scope and reported NOT RUN.

### 1.9 Compute and order

Max 4 cores, LightGBM `n_jobs = 1`, parallelised over cells with joblib,
OMP/MKL/OPENBLAS pinned to 1. Priority: all F2 cells of the three sub-models,
the po F2 control, the F2 reseeds, then F1 cells (rebound, shot_block, ft_tech),
then po F1. Any cell not finished by 15:00 EDT is reported NOT RUN with its
exact resume command; an arm missing its F1 cell cannot pass E7.

---

## 2. Round 1 results (run 2026-09-30 11:02-12:10 EDT; `scripts/exp_season_drift_anchor_v1.py`, graded by `scripts/grade_season_drift_anchor_v1.py`; full report `docs/tests/season_drift_anchor_round_2026-09-30.md`)

All 58 pre-registered cells ran (7 arms x 4 models x 2 folds, plus the `R`
seed-1 reseeds for rebound and po on F2). Harness checks: rebound `R` = round 3
`A0B0C0` (0.645565, -1.137 pp); `T` = round 3 `A5` (0.645024, -0.337 pp);
shot_block `R` = sklearn `K2` to 1.5e-06.

**Mechanical decision (section 1.7): NO DESIGN WINS.** Selections: rebound `R`
(no eligible arm), shot_block `F` (1.24 floors, gain 7.2e-06), ft_tech `R` (no
eligible arm). No design is selected in two of three sub-models. NOTHING
ADOPTED.

- E4 (drift-stops, pooled over 10 flat/reversed cells) makes `C0` (+0.061),
  `T` (+0.092) and `W` (+0.008) ineligible everywhere. `T` fails where FT
  technicals reverse (F1 +114%); `C0` carries the 2023 technical spike (+78%).
- `F` (level as a feature) works in the linear models and breaks both trees
  (rebound -4.2 floors, team slope 0.67 -> 0.37; po -4.4 floors).
- `O` / `P` (target relative to the as-of league level; `P` with the fitted
  update weight, n0 ~ 1e4 units, a few days of play) pass E4 and improve level
  and likelihood in every model on both folds (`O` except shot_block F1):
  rebound -1.137 -> -0.084 / -0.115 pp (4.0 / 4.1 floors, F1 confirms); ft_tech
  +32.5% -> -10.1% / -8.5%; po control max class gap 1.54 -> 0.40 / 0.29 pp at
  ~1 published floor, calibration gate F -> P. They are blocked by (i) rebound
  E6: the top OREB decile (+2.20 / +2.07 pp vs the 2.0 gate), a shape defect
  `R` already had (+0.88 pp under a -1.02 pp level shift) that the level bias
  was hiding; (ii) ft_tech E5: the one-sided slope/spread line reads the removal
  of `R`'s +32% multiplicative level error as spread compression (level-
  normalised, the spread is unchanged: 0.68 -> 0.65-0.66). A supplementary
  symmetric check is reported beside the rule; it is not the rule.
- shot_block's rim miss is not a league-level drift (pooled -0.17 pp; rim
  -0.83, jump2 +0.29). An EXPLORATORY, not pre-registered, per-shot-type anchor
  (`scripts/exp_season_drift_anchor_cell_v1.py`) moves rim to +0.35 pp (`P`,
  F2) / -0.16 pp (F1) and passes the shot_block level gate on both folds. It
  decides nothing; it is the obvious next pre-registration.

Open for the PM: a ruling on E5's one-sidedness and on whether rebound's
unmasked top-decile shape defect blocks a level fix; then stage 2
(`S1_weekly`) for `R` / `O` / `P` (fitter v2 not yet written) and a closed loop.
