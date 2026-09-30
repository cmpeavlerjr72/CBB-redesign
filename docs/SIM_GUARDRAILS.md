# Sim Guardrails: CBB gates, debugging philosophy, and what last year taught us

Created 2026-09-10. Companion to `cfb-props-sim/docs/SIM_GUARDRAILS.md`, whose core principle applies verbatim here.

**Core principle (non-negotiable):** post-hoc multipliers, caps, clips, calibration curves, or blends applied to engine output defeat the purpose of having models. If a sub-model produces biased output, fix the sub-model. The gates below exist so we detect bias at the right level, and the first response to a failed gate is "which sub-model is producing the wrong distribution", never "what adjustment closes the gap".

---

## 1. The CBB-Monte cautionary inventory (2025-26)

Last year's engine did not fail through stacked multipliers the way CFB-Monte did. It failed through absences: no league centring on a drifting rating scale, no home court, no shared pace, no fitted dispersion, no refit, no player layer. The "improved" branch then started adding exactly the patches this document bans (asymmetric pace weights 0.6/0.4, a `net_diff * 0.01` fudge, a clip(50, 90), a hand-set shooting covariance), and the tournament fork added a 9-point post-hoc calibration curve on win probabilities. None of it shipped in the graded season, which is the only reason the scorecard is clean enough to learn from. Full detail: `docs/postmortem/01_engine_review.md` sections 5 and 8.

Measured cost of the absences over 5,229 games: total bias +9.4 points, home margin 3.1 points too low, margin distribution twice too wide, no residual signal after controlling for the closing line.

---

## 2. Reference targets

Every gate compares simulated quantities to real ones computed from the same games. Targets are not typed into this document; they are built per season by `scripts/build_gate_reference.py` into `data/reference/gate_targets_{season}.parquet` and read from there, because several move year over year. From the 2026-09-10 audit (`docs/tests/data_audit_hoopr_2026-09-10.md` section 6), D-I games:

| Season | Poss/game (SD) | Pts/team | 3PA share | FTA/FGA | Home margin, non-neutral (SD) | Neutral margin |
|---|---|---|---|---|---|---|
| 2021-22 | 68.6 (5.9) | 70.4 | 37.9% | 0.305 | +7.9 (17.4) | +1.1 |
| 2022-23 | 68.4 (5.9) | 71.1 | 37.6% | 0.318 | +8.4 (17.3) | +2.8 |
| 2023-24 | 68.9 (5.8) | 73.0 | 37.4% | 0.331 | +8.3 (17.6) | +2.6 |
| 2024-25 | 68.4 (5.8) | 73.2 | 39.2% | 0.332 | +8.9 (18.2) | +3.1 |
| 2025-26 | 68.7 (5.9) | 75.1 | 39.6% | 0.352 | +9.1 (18.7) | +2.8 |

Two things to read from this table. Scoring rose 4.7 points per team in four seasons, driven by free-throw rate and three-point volume, so any model trained on pooled seasons without season-aware features will under-shoot the current year (the CFB pace-drift lesson, in the other direction). And pace is roughly 3 possessions faster in November than in February, so pace targets are checked by month, not just by season. Raw home margin in this table is over ALL games and includes buy-game blowouts; restricted to D-I vs D-I games it is 5.0-5.7 points (2025: +5.74, SD 14.9; possessions 67.9, SD 5.5; total 145.5, SD 19.0; home/away score correlation 0.25; OT 5.6%). The gate tables in `data/reference/gate_targets_{season}.parquet` use the D-I-only definition (`docs/tests/gate_reference_2026-09-10.md`). The gate compares sim to real on the same games, so it is like-for-like, not an estimate of home-court advantage itself.

---

## 3. Gates

Every gate is evaluated on a held-out season, on D-I non-truncated games only, and at every level listed. Tolerances are provisional until the seed-noise study (week 6) replaces them with measured paired standard errors; until then a miss is a miss. A gate report is one markdown section per gate with target, sim, actual, delta, and a literal PASS / FAIL / NEEDS-INSTRUMENTATION status, written by `scripts/eval_gates.py`.

| Gate | Quantity | Levels | Provisional tolerance | Failure points at |
|---|---|---|---|---|
| G1 | Possessions per game, mean and SD | overall, by month, by tempo tercile | mean +/- 1.0; SD +/- 0.75 | pace model (L2) |
| G2 | Points per possession | by offense tercile x defense tercile | +/- 0.02 per cell | shot-make, FT, turnover models (L3) |
| G3 | Shot mix per possession: 3PA share, rim share, FTA/FGA | by team | +/- 1.5 pp | possession-outcome model (L3) |
| G4 | Four factors, offense and defense: eFG%, TOV%, OREB%, FT rate | by team, by tier | +/- 1.0 pp (eFG, TOV, OREB); +/- 0.015 (FT rate) | L3 sub-models individually |
| G5 | Dispersion: SD of (actual - sim mean) vs mean sim SD for margin and total; correlation of the two team scores; PIT histogram | overall, by spread band | SD ratio 0.95-1.05; corr +/- 0.05; PIT K-S p > 0.1 | shared pace draw, dispersion of L3 models |
| G6 | Home margin, non-neutral vs neutral, same games | overall, by conference tier | +/- 1.0 point | home feature wiring in every scoring-stage model |
| G7 | Overtime rate; first-half vs second-half scoring share | overall | OT +/- 1.0 pp; half share +/- 1.0 pp | end-of-period clock model, end-game behaviour (L5) |
| G8 | Player layer: minutes per player (mean, SD) by starter/bench; share of team FGA by top-1 and top-3; players with >0 minutes; per-player points/rebounds/assists distribution tails | by player role, by team | minutes mean +/- 2.0 for rotation players; SD ratio 0.9-1.1; K-S p > 0.1 on counts | rotation model, usage allocation, player rate shrinkage (L4) |
| G9 | Spread and total accuracy: MAE, signed bias, calibration slope; win-probability calibration by decile | overall, by month, by tier | margin bias +/- 0.5; total bias +/- 1.0; slope 0.95-1.05 | aggregate symptom; check G1-G8 first |
| G10 | Market scorecard where lines exist: MAE vs close, Brier vs de-vigged market, ROI by edge bucket, CLV sign agreement | by market, by season | report only; LEAK-SUSPECT if surprise corr > 0.15 and CLV agreement < 0.53 | a leak, not a model |

Meta-gates (always on):

- `created_at < tipoff` on every backtest row, enforced by the artifact writer.
- Every external feature passes `scripts/leak_test_pregame_features.py` (|change-form corr| <= 0.15; baseline 0.04-0.08) before entering a feature table. CBBD's season ratings endpoints are end-of-season snapshots and are banned as pregame features.
- Grading truth is verified against a second source (hoopR schedules vs CBBD games vs pbp-accumulated finals) before any gate is read. The audit found 1.2-1.6% of games per season with truncated pbp feeds; those are excluded from pbp-derived targets, never patched.
- The 2025-26 season is sealed (`src/cbb_sim/data/seal.py`) until selection on the earlier folds is complete.
- Minimum seed count for any ROI or gate read is set by the seed-noise study, not assumed.

---

## 4. Known risks to watch per upcoming sub-model

- **Event vocabulary drift.** hoopR pbp event types are not stable across seasons (Substitution only from 2024-25; Coach's Challenge and a bare "Shot" type only in 2025-26; a "Not Available" placeholder in 2021-23). Every pbp-derived feature must be built from an explicit mapping table with a test that fails on an unknown type.
- **Shot location.** Coordinates cover 6-26% of shots before 2024-25 and 100% in 2025-26. Rim-vs-jumper classification must come from type_text (DunkShot, LayUpShot, TipShot vs JumpShot), not coordinates, for backtest seasons.
- **Lineups.** hoopR pbp has no on-floor field. CBBD pbp carries all ten players per play for every season. Lineup features come from CBBD; the crosswalk between CBBD and ESPN player ids is a prerequisite and its match rate is a reported number.
- **Availability.** hoopR player_box `active` is a placeholder before 2025-26; `did_not_play` is the historical availability field.
- **Scoring trend.** Free-throw rate rose from 0.305 to 0.352 across the five seasons. Foul and FT models must carry season-aware features or be refit each preseason, and G4 FT rate is checked on the held-out season, not pooled.
- **Two-team correlation.** The shared pace draw induces positive score correlation; the reference correlation between home and away scores is in the gate table. Getting the total variance right while the correlation is wrong is the CFB INV-56 failure and is caught by G5's correlation check, not by the SD check.
- **Allocation under a correct total.** Team totals can be right while the favourite/underdog split is wrong (CFB INV-141). G6 and the G9 tier breakdown check the split independently of the total.
- **Rotation garbage time.** Bench minutes in blowouts affect props far more than game markets. G8 is broken out by final-margin band.

---

## 5. When gates fail

Do: trace the metric to the responsible sub-model; check that model's calibration on its own holdout; check the feature pipeline is delivering the right inputs at sim time; check the distribution shape, not just the mean; check whether the "bias" is actually the profitable side of a market inefficiency before chasing it to zero (CFB INV-98).

Don't: apply a multiplier; cap or clip a probability; add a conditional boost for hard cases; stack adjustments for different reasons onto one quantity; fit a calibration curve on the output; blend toward the market. If a model's architecture cannot represent the correct conditional distribution, rebuild the model.

---

## 6. Revision log

- 2026-09-10: created from the CBB-Monte postmortem and the hoopR/CBBD data audits. Tolerances provisional.
- 2026-09-30 (PM, evidence in `docs/tests/*_2026-09-30.md`): four amendments to how the gates are read. (a) **G1 truth is like for like**: the engine's possession count is graded against the pbp possession count with phantom possessions removed (event layer v4 definition), never against a box-score estimator; the +1.99 read of 09-18 is +1.61 under this definition (`g1_g5_possessions_corr_diagnostic_2026-09-30.md`). (b) **G4 TOV% vetoes are read on the turnover COUNT per team-game**, because the gate's box-possession denominator (FGA - OREB + TOV + 0.44 FTA) shrinks mechanically when OREB rises, so an OREB fix raises the rate with no change in turnovers (`shot_block_drawn_flag_2026-09-30.md`). (c) **Noise floors for closed-loop lines**: a single seed-offset draw understated the G5 total-SD-ratio seed variation about fivefold at 25 seeds (four served seed sets: SD 0.005 vs a 0.0009 "floor"). From now on every paired closed-loop veto line carries BOTH a multi-draw seed floor (at least four seed-offset reference runs) and a paired game bootstrap interval; a 500 x 25 loop cannot decide a G5 ratio line, only a 200-seed read with a multi-draw floor can, and every V3 veto read today at 25 seeds (foul rounds 7-8, drawn block flag) is PROVISIONAL until that read. (d) **The G5 total SD ratio mixes dispersion with accuracy** (mean within-game sim SD over the SD of actual minus sim-mean total); it is reported with its two components separately, so a fix that changes prediction accuracy is not misread as a dispersion change. (e) **Grading truth is verified finals only** (`truth_unplayed_finals_2026-09-30.md`): 54 unplayed games (0-0 and forfeit rows) sat inside the graded universe; corrected reference tables live in `data/reference/verified_v1/` and are selected with `CBB_TRUTH=verified_v1` until the loader default flips.
