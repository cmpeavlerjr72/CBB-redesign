# Early-season league-level anchor: cross-model experiments (DRAFT, NOT COMMITTED, NOT RUN)

Status: DRAFT pre-registration for PM review. Becomes `docs/models/early_anchor/experiments.md` (append-only)
when the PM commits it; NOTHING runs before that commit. Nothing here adopts anything or changes a served default.
Drafted 2026-10-08 by a Sonnet worker. The round is post-freeze (freeze 2026-10-10).

## 1. Why this round

The d0-14 total bias runs about -4.5 (F2) / -5.0 (F1) points (`docs/tests/early_total_points_decomp_2026-10-07.md`,
Shapley with possession replacement). The live path (A3 + R1) closes only +0.38 / +0.67
(`docs/tests/early_gap_live_path_2026-10-07.md`). Channels, F2 / F1 d0-14, in points: FT% -1.4 / -1.1; TOV -0.6 / -2.2;
FTA rate -0.7 / -0.9 (foul accrual round 2 had no eligible arm: `docs/tests/foul_accrual_round2_2026-10-07.md`);
possessions -0.9 / -0.5 (flipping to +0.8 / +0.6 at d46+); jump2 make -0.7 / -0.4. Every channel is a flat level offset
across prior-season team quintiles (slopes within 0.02 of zero), broad (60-79% of teams negative), and calendar-shaped
(small or opposite by d46+). One shared cause is plausible, so one round tests one shared early-season input across the
sub-models instead of five separate rounds.

Prior single-model art that this round must not repeat blind (all NOT ADOPTED):
- PO TOV `T1` (as-of league TOV level offset): fixed F1, failed E3 on F2 d15-45 (|gap| 0.11 -> 0.20 pp vs a 0.03 floor).
  PO experiments sections 30/31.
- FT `X1` (`days_since_start` feature): offline WIN (window gap -1.73 -> -1.20 pp) but G9 total bias regressed at full
  size. FT experiments sections 18/19.
- Foul accrual calendar term `A2dbk`: failed d0-14 calibration and F1 responsiveness. PO sections 32/33.
- Clock `M2D` (`days_since_start` in the pace model, round 8): not adopted; K2 was adopted without it.
- Season-drift anchor round 1 (`docs/models/season_drift/experiments.md`): the O/P/F/T designs for ACROSS-season drift.
  This round reuses its `L_blend` object; it does not re-ask the across-season question.

New here: ONE shared definition fed to every sub-model at once and judged JOINTLY in the closed loop. The earlier wins
moved one channel and were judged one channel at a time.

## 2. Sub-models in scope (reference = the served arm, hyperparameters unchanged)

| id | sub-model | served reference | channel | how the anchor enters |
|---|---|---|---|---|
| M1 | possession_outcome `first` | `lgbm / C_plus_state`, S1 monthly (PO section 30 `T0`) | TOV | input / link offset on the TOV raw score only |
| M2 | free_throw make | served `FT_FEATURES`, `S1_conf_aligned` (FT section 18 `X0`) | FT% | input column |
| M3 | clock pace | served K2 `v5b_r8K2_glat_pmean` | possessions | the `needs_days_since_start` path of `v5b_r8M2D` (clock sections 39/40) |
| M4 | fg_make jump2 | served structure and TS-R, jumper type (fg_make sections 29/30) | jump2 make | input column / offset on the jump2 logit |
| M5 (optional) | foul accrual `A2` (R9ao3) | served `A2` | FTA rate | see Q1; not in the primary decision unless the PM rules it in |

Each model is retrained offline on the fold's train seasons at its own served refit cadence. Rim make, 3 make and rim
share are 1-2 SE of game noise and stay out of scope.

## 3. Arms (the same definition in every sub-model; 5 cells including the control and the noise floor)

| arm | design | inputs added | simplicity rank |
|---|---|---|---|
| `Z0` | control: served model, no anchor | none | 0 |
| `Z0s1` | spec-identical retrain of `Z0` under seed 1 (noise floor; never selectable) | none | floor |
| `Z1` | calendar level: days since season start as a raw input | 1 column (`dss`) | 1 |
| `Z2` | shared early basis: one bounded column `early = exp(-dss / 14)` | 1 column | 2 |
| `Z3` | in-season league level, relative form: link-scale offset `link(L_blend(s,t)) - link(Lbar)` per sub-model | 1 offset per sub-model | 3 |

Why these three: `Z1` is the minimal calendar test (FT `X1`, clock `M2D` precedent). `Z2` encodes the observed shape
(strongest in week 0-2, gone by d46: the in-bonus share is 0.28-0.31 in week 0 vs 0.20 from d46) as ONE shared column, with
the decay constant FIXED a priori at 14 days to match the d0-14 window; it is not tuned. `Z3` tests the opposite mechanism,
a level learned in season (PO `T1` precedent). `Z3` has no calendar shape, so `Z2`-over-`Z3` separates "calendar" from
"lagging league level". A compound `Z2 + Z3` arm is held back (Q3). Ties go to the lower rank.

## 4. Feature definitions and leak safety

- `dss` = calendar days from the season's first D-I game date to the game date. The first date comes from the published
  schedule, not from games played so far, so no row looks ahead. It equals the engine's `days_since_start` team column
  (parity already verified: FT 5,593 / 5,593 F2 games match; clock 99.96% exact, 2 engine fills).
- `early = exp(-dss / 14)`: a deterministic function of `dss`, no fitted constant.
- `Z3` level `L_blend(s,t; n0) = (num_before + n0 * L_end(s-1)) / (den_before + n0)` over rows of season `s` with game
  date STRICTLY before `t`. `L_end(s-1)` = the previous completed season's end level; 2022 (train-only) uses the fold's
  pooled train level. `n0` comes from the grid {0, 1e1, 3e1, 1e2, ..., 1e8, inf}, chosen INSIDE the fold's train seasons
  that have a previous season (F1: 2023; F2: 2023, 2024) by anchor-only likelihood, per sub-model. Rates: M1 first-chance
  TOV share; M2 league FT make rate; M3 league possessions per team-game; M4 league jump2 make rate. Offsets are centred
  on the fold's train level `Lbar` (every level is relative to its own mean; raw levels are banned) and are applied
  identically at fit (LightGBM `init_score` / GLM offset) and at predict.
- Leak checks: `created_at < tipoff` asserted in code on every design row; a shift test rebuilds each anchor with the test
  season's future rows deleted and must reproduce it bit-for-bit. All levels are the league's own pbp / box quantities
  (not an external feed), so the KenPom change-form leak test does not apply.
- Home / away / neutral stays a first-class feature in every scoring-stage model; audit each arm's feature list.
- 2025-26 stays SEALED (`assert_not_sealed` on every load) and is not read in any step.
- Anchors are model INPUTS only. No multiplier, cap, clip, offset, curve or blend on sim output at any stage. Serving is
  through default-off input-side paths (existing `ENGINE_SEASON_ANCHOR` for PO, `ENGINE_FT_SCORE` for FT, clock mode keys);
  the off path stays bit-identical to the current parity reference.

## 5. Folds

Fold 1 trains through 2022-23 and tests 2023-24. Fold 2 trains through 2023-24 and tests 2024-25. Fold 2 selects; fold 1
confirms. 2025-26 sealed. Refit cadence is each sub-model's served cadence, identical across arms. Seed 0 for all arms
except `Z0s1`. Loop RNG is seeded on (seed, game_id, family) so paired arms share aligned streams.

## 6. Metrics

Offline, per sub-model, one blind grader for every arm and sub-model with no arm-specific path
(new `scripts/exp_early_anchor_v1.py` and `scripts/grade_early_anchor_v1.py`):
1. The sub-model's own primary (log loss, or Poisson deviance where that is its convention).
2. Window calibration: mean p - mean y by d0-14 / d15-45 / d46+ / all, and by month.
3. The sub-model's own gates (worst decile gap <= 2.0 pp; FT Decision 8 slope in [0.8, 1.2]; clock coupling gates;
   fg_make slope harness).
4. Responsiveness: predictions bucketed by team prior-season-rate quintile, slope = span of mean predicted over span of
   mean realised, overall and d0-45; per-team spread (SD of team mean prediction over noise-corrected realised SD). An arm
   that is flat at the mean fails.
5. Paired game-block bootstrap (200 reps, seed 12345) SE of every (arm - `Z0`) difference.

The round's PRIMARY metric is game-level d0-14 total bias and total MAE (sim mean total minus verified final), measured
in the closed loop of section 8.

## 7. Segment breakdowns (every table, both folds)

- Overall; d0-14, d15-45, d46+; by month; by site (home / away / neutral).
- Per-game: total bias, total MAE, margin MAE, margin bias, distribution of per-game total error (SD, 5th / 95th pct).
- Per-team (teams with >= 2 games in the window): share with negative d0-14 total gap, team mean and SD, slope on
  prior-season total, prior-quintile means. With 2-3 games per team, single-team values are labelled underpowered; shares
  and slopes are the usable evidence.
- Per-possession-type / per-channel: TOV rate, FT% and FTA rate, possessions per team-game, jump2 make, plus the Shapley
  channel table (`scripts/diag_early_total_points_decomp_v1.py`) rerun on each arm's sim taps. A total that improves by
  the wrong channel (for example a possession overshoot cancelling a TOV miss) is a FAIL under the multi-level evidence
  rule.
- Per-player (FT): FT% by group (anonymous slots / newcomers / returners) against bench, since 55-65% of the early FT%
  points sit in anonymous slots. The A3 + R1 live path is the reference.

## 8. Paired-seed closed-loop confirmation

Runs only for a design that passes offline eligibility (section 9) in at least 3 of M1-M4 on F2; else NOT RUN.
Mirrors `early_gap_live_path_2026-10-07.md`.
- Base is the live path A3 + R1 (the stack that ships Nov 2); the design's anchor is switched on in every sub-model where
  it was eligible together. `Z0` = live path, no anchor.
- 50 paired seeds, days 0-45, F1 and F2 (the `F{1,2}_srv_o0` game lists and seeds); reseed floor = `Z0` at seed offset 1000.
  Verified finals only.
- Read, with 95% game-bootstrap intervals: total bias and total MAE by d0-14 / d15-45 / d46+, margin MAE and bias, the
  channel table and Shapley table of section 7.
- A d0-14 claim needs the 50-seed move to exceed 3x the reseed move (seed-count check).
- A design that clears the loop gets a full-size 5,710 x 200 G1-G9 box request (`docs/ops/box_queue/`). The local sim is a
  screen, not adoption evidence. The PM decides adoption.

## 9. Noise floor and decision rule (mechanical)

Floor per line = max(|`Z0s1` - `Z0`|, 2 x paired bootstrap SE of (arm - `Z0`), the published floor where one exists:
PO multiclass LL 0.000804, PO TOV LL 0.000097 (F1) / 0.000063 (F2), FT LL 0.000147, FT gap 0.25 pp). Deterministic fits
(GLM, L-BFGS) have a zero reseed, so for them the bootstrap is the floor.

An arm is ELIGIBLE in sub-model `m` iff, on F2:
- E1 its own primary is not worse than `Z0` by more than the floor, and its d0-14 |gap| is smaller than `Z0`'s by more
  than the d0-14 floor;
- E2 d15-45 and d46+ |gap| are not worse than `Z0`'s by more than each bucket's floor (hard veto; the PO `T1` lesson);
- E3 the sub-model's own gates do not newly fail; quintile slope >= `Z0`'s - 0.05 (overall and d0-45) and noise-corrected
  spread ratio >= `Z0`'s - 0.05;
- E4 FOLD 1 CONFIRMS: F1 d0-14 |gap| not worse than `Z0`'s beyond the F1 floor; F1 primary, d15-45 and d46+ not worse
  beyond their floors;
- E5 the Dec-Mar level gap does not widen beyond the floor (an early term must not buy its gain by distorting the late
  season).

Within `m`, the eligible arm with the best F2 d0-14 |gap| wins; an eligible arm within one floor of it with a lower
simplicity rank wins the tie. No eligible arm: `m` keeps `Z0`.

ROUND WINNER: a design (applied identically where it is eligible) that is eligible in at least 3 of M1-M4 (M5 never counts
toward the 3) AND clears the closed loop on BOTH folds:
- C1 d0-14 total bias moves toward zero by more than 3x the reseed move, with the 95% CI excluding zero;
- C2 d15-45 and d46+ |total bias| not worse than `Z0` beyond the reseed move;
- C3 margin MAE and margin bias not worse beyond the reseed move (margin veto);
- C4 total MAE (all windows) not worse beyond the reseed move; per-team negative shares fall; quintile slopes not
  flattened;
- C5 the channel table shows the gain came from the intended channels (TOV, FT%, possessions, jump2), with no channel moving
  the wrong way by more than 0.3 pt in d15-45 or d46+ (the d46+ possession overshoot must not grow).

Otherwise the round has NO WINNER, stated as such. The PM-pending decision-layer policy (known-bias label on early totals,
no totals edge actioned before day 15) stays as the mitigation. A partial result (eligible in 1-2 models) is reported to
those models' own ledgers and is not a round win. Simplicity order: `Z1` < `Z2` < `Z3`; ties go to the simpler arm.

## 10. What this round may not do

No post-hoc multiplier, cap, clip, calibration curve or blend on sim output. No engine `src/` edit (default-off adapters
only). No served default change. No 2025-26 read. No paid data. No overwrite of a file another worker may read: outputs go
to versioned siblings under `results/early_anchor/` and `data/processed/models/early_anchor/` (gitignored and synced
through an `hf_sync_data.py` bulk key if over 20 MB). Status changes go to `docs/models/change_ledger.md` in the same
commit as the results section.

## 11. Compute and order

Max 4 cores, LightGBM `n_jobs = 1`, OMP / MKL / OPENBLAS pinned to 1. Order: F2 cells for M1, M2, M4, M3, then the `Z0s1`
reseeds, then F1 cells, then the closed loop for an eligible design. A cell unfinished at the wall-clock deadline is
reported NOT RUN with its exact resume command; an arm missing its F1 cell cannot pass E4.

## 12. Open design questions for the PM (rulings requested before commit)

Q1. Does foul accrual (M5) join? Round 2 had no eligible arm and round 3 (A2t + in-bonus over-prediction) is queued. Joining
    covers the FTA channel (-0.7 / -0.9) but couples this round to round 3. Draft default: optional, reported,
    excluded from the 3-of-4 count.
Q2. Is `exp(-dss/14)` acceptable as a single a-priori constant, or should two fixed values (7 and 28 days) run as extra
    arms? Draft default: one value, no tuning.
Q3. Add a compound `Z2 + Z3` arm (6 cells total, over the 3-5 arm budget)? Draft default: no.
Q4. Closed loop on the live path A3 + R1 (draft default, it is what ships Nov 2) or on the plain served stack?
Q5. F1's TOV step (0.198 vs 0.182 early) is a cross-season level shift, not an early-season effect, so only `Z3` can
    plausibly fix it. Draft default: no exception; E4 applies to it as written.
