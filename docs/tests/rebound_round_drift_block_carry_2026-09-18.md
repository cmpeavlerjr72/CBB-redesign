# Rebound round 3: season drift, the `blocked_f` feed, and prior-season carry (2026-09-18)

Lane: the rebound bake-off pre-registered as PROPOSED in
`docs/models/rebound/experiments.md` section 9 (commit 3b75c3c) and amended
against the PM's five conditions in section 10 (commit 9d0e55e, written and
pushed BEFORE the first fit). Evidence this round answers:
`docs/tests/g4_oreb_fta_diagnostic_2026-09-18.md`.

**NOTHING IS ADOPTED. No served default is changed. No engine file is edited.**
`data/processed/models/rebound/` is untouched; everything this round produced is
in its `round3/` sibling and in `data/processed/models/shot_block/`.

**What ran:** stage 1 (the `S0` static-calendar screen) for all 16 pre-registered
arms plus two combined arms, on **both folds**, plus the second-seed noise floor;
and the whole `shot_block` bake-off (`docs/models/shot_block/experiments.md`).
37 graded cells.

**What did NOT run, and why:** stage 2 (the served `S1_weekly` calendar, 23
refits per cell) and the paired closed loop. Measured cost on a machine four
other workers were using: **472-631 s per single fit**, so one `S1_weekly` cell
is ~3 h and the four-cell stage-2 confirmation is ~6 h wall. The session had a
hard 21:30 ET stop. Per section 10.5 those cells are reported **NOT RUN**, never
as a result, and **per section 10.5 the round's decision cannot be taken**: what
follows is a stage-1 screen and a recommendation, not a winner. Resume commands
are in section 7.

---

## 1. Headline

Two of the three G4 channels this round owns have a working candidate, and they
are **separable**: one arm fixes the level, a different arm fixes the slope.

| what | served reference | best arm | closes |
|---|---:|---:|---:|
| fold-2 model level error `L1` (true `blocked_f`) | -1.137 pp | **-0.337 pp** (`A5`) | 70% of the calibration channel |
| engine-feed level error `L2` | -1.877 pp | **-0.336 pp** (`A5` + the `B3` block draw) | 82% |
| per-team prior-quintile slope ratio | 0.685 offline / 0.676 engine | **1.122** (`A5+C1`) | the responsiveness defect, and then some |
| Nov-Dec slope ratio (the worst cell) | 0.595 offline / 0.561 engine | **1.197** (`A5+C1`) | — |

Projected engine OREB% gap, holding the two channels this round does not own
fixed at their measured values (sim state/mix **+0.234 pp**, grading source
**+0.082 pp**, G4 diagnostic section 1, so `engine gap = L2 + 0.316 pp` --
the identity reproduces the served -1.561 pp exactly, which is the check that
the projection is arithmetic and not a fit):

| stack | `L2` pp | projected engine gap pp | share of the -1.561 pp closed |
|---|---:|---:|---:|
| served (`A0B0C0`, `blocked_f = 0`) | -1.8769 | **-1.561** (measured) | 0% |
| served + `B3` block draw | -1.1370 | -0.821 | 47.4% |
| `A5`, `blocked_f = 0` | -1.0887 | -0.773 | 50.5% |
| **`A5` + `B3`** | -0.3356 | **-0.020** | **98.7%** |
| **`A5+C1` + `B3`** | -0.4422 | **-0.126** | **91.9%** |
| `A5+C3` + `B3` | -0.3381 | -0.022 | 98.6% |

This is a **projection from offline level readings**, not a sim result. The
closed loop is NOT RUN and nothing here may be read as a gate outcome.

---

## 2. Block B: two of the four pre-registered feeds are structurally invalid

The single most consequential finding of the round, and it is a refutation of
the pre-registration's own preferred arm.

Section 9.1 called `B2` -- feed the as-of measured block rate as "a continuous
value in [0,1], which is what the trained column's conditional expectation
means" -- "the cheapest arm that is not a false statement to the model, and the
one to beat". **It is a false statement to this model.** `blocked_f` is a 0/1
column, and LightGBM's bin boundary for a two-valued feature is at **0.0**, not
0.5: every one of the 180 splits on `blocked_f` in a test fit had threshold
exactly `0.0`. Any strictly positive value -- 0.26 as readily as 1.0 -- lands on
the "blocked" side. Measured directly, feeding 0.26 and feeding 1.0 give
**bit-identical** mean predictions.

All feeds below use the SAME trained reference model on the same fold-2 rows;
only the `blocked_f` column changes. Actual live OREB% 0.299232.

| feed | level pp | 3-class log loss | verdict |
|---|---:|---:|---|
| `B0_zero` (served status quo) | -1.8769 | 0.647713 | the defect |
| `B2_asof_cell` (as-of cell rate, continuous) | **+7.0912** | 0.664252 | **REFUTED -- structurally invalid** |
| `B3e_model` (block model's p-hat, continuous) | **+8.3292** | 0.667026 | **REFUTED -- same mechanism** |
| `B3_draw` (0/1 drawn from the block model) | **-1.1370** | 0.648339 | **closes the whole channel** |
| `B_true` (the true flag, unattainable) | -1.1370 | 0.645565 | the ceiling |

`B3_draw` matches `B_true` to **0.0000 pp**. The engine-feed channel is worth
-0.740 pp and a drawn indicator from the block sub-model recovers all of it.

`B1` (drop `blocked_f` and refit) also removes the -0.740 pp penalty by
construction (`L2 = L1 = -1.157 pp`) but costs log loss on both folds
(+0.00153 F2, +0.00167 F1) and throws away real information. It is the fallback
if no block model is wired, not the answer.

**`fg_make` is not double counted.** `fg_make` lists `blocked` in
`BANNED_FEATURES` and does not build it, so it prices blocked attempts inside
its miss population already. The block draw therefore sits AFTER the make/miss
draw and before the rebound draw, and changes no field-goal aggregate. Verified
in code, not assumed (`src/cbb_sim/models/fg_make.py`, `BANNED_FEATURES`).

---

## 3. The shot-block sub-model

Full pre-registration and results: `docs/models/shot_block/experiments.md`.
Target `P(blocked | the attempt MISSED, context)`, 1,404,428 missed FGA,
shooter joined at a measured 88.08% match rate with 100% `blocked` agreement.

**No arm cleared its own pre-registered rule.** Every fitted arm beats the
`K0` league-rate-by-shot-type baseline on fold-2 log loss by 30-50x the floor
(4.1e-05) and passes calibration and responsiveness, but every one fails the
level gate on **rim**: -0.78 to -0.83 pp against a 0.50 pp threshold. `K0`
itself misses rim by -0.53 pp. That is the **same season-drift defect** this
round found in the rebound model, in a second sub-model: the 2025 rim block rate
is above every training season's.

Fold-2 ranking (log loss): `K2` 0.263240 (ridge, the best), `K3_prior` 0.263305,
`K3_conf` 0.263480, `K3` 0.263508, `K0_oracle` 0.265223, `K0` 0.265250,
`K1` 0.267324, `K1_oa` 0.269724. The tree never beats the linear arm, so the
pre-registered "a tree must beat the best linear arm by more than the floor"
clause resolves to `K2`. Decision 9's arms both lose here: `K1_oa` (opponent
adjustment) is the **worst** arm of the eight and `K3_conf` (conference flag)
gains 2.8e-05, well inside the floor. `K3_prior` is the only arm whose
defence-prior quintile slope reaches 1.06; `K2`'s is 0.708 and `K0`'s is 0.111
(a flat predictor, as expected).

**The feed works even though the model failed its own gate.** `B3_draw` above
uses `K2`. Its pooled level error is -0.165 pp and its per-cell rates are close
enough that a drawn indicator reproduces the true-blocked rebound level exactly.
The rim level gate failing at 0.50 pp is a real defect worth its own round; it is
not what stands between the engine and the -0.740 pp channel.

---

## 4. Block A: the drift arms, and what PM condition (a) exposed

A tree **cannot extrapolate**, and this round measured it rather than arguing it:

- `A1` (`season_idx` as a feature) is not a trend arm. With `season_idx` in
  {0,1,2} in training and 3 at test, every split routes 2025 to the 2024 bin. It
  behaves as "use the most recent completed season's level": `L1` -0.829 pp,
  and it **loses** fold-2 log loss (-0.000251, i.e. worse than the reference).
- `A4`, the within-season as-of league anchor that CLAUDE.md's rule points at,
  **fixes the level and destroys the model**: `L1` **+0.306 pp** (it overshoots),
  but fold-2 log loss 0.649412 (the second-worst arm), calibration FAIL
  (2.585 pp) and the team slope ratio **collapses to 0.397** from 0.685. It is
  also out of support -- the round records
  `lg_oreb_asof_test_in_train_support = False` -- so the tree is reading its top
  bin for much of the test season. On fold 1 it is worse than the reference on
  every reading. **Refuted:** carrying the level in a league covariate buys the
  level by spending the matchup signal.
- `A2` (recency weights, 60/120/240-day half-lives) and `A3` (1- and 2-season
  rolling windows) all **lose** on fold-2 log loss (0.6462 to 0.6496 against the
  reference's 0.6456) while improving the level modestly (`L1` -0.83 to -1.02).
  Shorter half-lives help the level more and hurt the loss more, monotonically.
  Condition (a)(i) is therefore answered: recency weighting is not the fix.
- `A5`, the genuine **trend extrapolation** (an OLS of season league OREB% on
  season index fitted on the fold's completed TRAIN seasons and carried as a
  LightGBM `init_score` offset, because a feature cannot extrapolate), is the
  Block-A winner: fold-2 log loss 0.645024 (gain 8.1x floor), `L1` **-0.337 pp**,
  calibration PASS (1.921), responsiveness PASS.

**`A5` wins fold 1 too** (gain 0.000350, `L1` -0.075 pp), so amendment rule
10.6.9 -- an extrapolating arm must win both folds -- is satisfied, and it is
NOT a "wins by continuation" result. The honest caveat is the residual itself:
the fitted trend puts 2025 at **0.29606** against the realised **0.29923**, and
that 0.32 pp shortfall IS `A5`'s remaining level error. The arm is right about
the direction and systematically short on the magnitude. If the drift flattens
or reverses, `A5` becomes an over-prediction by the same mechanism.

---

## 5. Block C and Block D

**Block C (prior-season carry) is what fixes the responsiveness defect.** `C1`
(PO round 4's `G2`) lifts the per-team prior-quintile slope ratio from **0.685 to
1.107**, and by season segment from **0.595 / 0.779 / 0.723** to
**1.171 / 1.174 / 1.002** -- the Nov-Dec cell, the worst in the G4 diagnostic
(engine 0.561), becomes the best. `C2` (`G3`) is nearly identical (1.081).
`C3` (`G1`, league-mean shrinkage only, the control) does **not** move the slope
(0.714), which is the clean separation the control exists to give: it is the
**prior-season information**, not the shrinkage, that buys responsiveness.
`C1` and `C2` both beat the reference on fold-2 log loss beyond the floor (5.7x,
5.2x) and on fold 1, but both **fail the calibration gate** on fold 2 (2.281,
2.228 pp against 2.00) and both make the LEVEL slightly worse (`L1` -1.244,
-1.259 vs -1.137). Alone, neither is adoptable.

**Block D (Decision 9's mandatory arms) all fail here.** `D1` (opponent
adjustment, one_pass) fold-2 gain **-8.2e-05** (worse than the reference) and
fold-1 -0.000215; `D2` (iterative) +5.5e-05 = **0.8x floor**, inside it, and
-0.000199 on fold 1; both **reduce** the team slope (0.629 from 0.685). `D3`
(conference-game flag) gains 3.7x the floor on fold 2 but fails calibration
(2.147 pp) and gains only 2.0x on fold 1. **Decision 9 stays PENDING EVIDENCE**;
this sub-model gives it no support, and neither does `shot_block` (section 3).
`D4` (refit cadence) is cited from section 8 as declared in amendment 10.4 and is
unchanged: `S1_weekly` remains the served calendar.

**`C4` (roster-continuity weighting) is NOT RUN**, for the measured reason the
amendment required be checked first: `data/processed/roster_continuity_2027.parquet`
covers **season 2027 only** (367 rows). Coverage of the seasons `C4` needs
(2023, 2024, 2025) is **0/3**. The arm is not droppable-in-silence and it is not
droppable-by-assumption; it needs the continuity table built back to 2023.

---

## 6. The combined arm, and the mechanical decision

`A5+C1` is the stage-1 leader by the pre-registered rule, on fold 2:

| reading | reference `A0B0C0` | `A5+C1` |
|---|---:|---:|
| fold-2 log loss | 0.645565 | **0.644658** (gain 0.000907 = **13.5x** floor) |
| fold-1 log loss | 0.620446 | **0.619846** (gain 0.000600) |
| `L1` level | -1.137 pp | **-0.444 pp** |
| `L2` with `B3` draw | -1.137 pp | **-0.442 pp** |
| calibration gate F2 | PASS 1.958 | **PASS 1.674** |
| calibration gate F1 | PASS 1.629 | **FAIL 2.212** |
| responsiveness gate | PASS 4/4 | PASS 4/4 |
| team slope ratio (all / Nov-Dec / Jan / Feb-Apr) | 0.685 / 0.595 / 0.779 / 0.723 | **1.122 / 1.197 / 1.182 / 1.008** |
| per-game level error mean / MAE | -1.006 / 4.964 pp | **-0.320 / 4.861 pp** |
| first 4 weeks of conference play | -1.777 pp | **-1.045 pp** |

It is the only arm clearing all four applicable rules (beats the reference beyond
the floor; passes both gates on fold 2; does not worsen `L1`; does not reduce the
slope in any season segment). `A5+C3` is 0.000109 behind on the primary -- more
than the floor, so it is not a tie -- and has a better level (-0.339 pp) but no
slope improvement (0.717).

The floor: the second-seed retrain of the reference gave a log-loss spread of
**4e-06** and an `L1` spread of 0.039 pp. Per this model's standing convention
("the floor the decision rule uses is the larger", section 8.2) the operative
floor is the already-published 5-seed SD, **6.7e-05**. The game-clustered
block-bootstrap SE for this cell is 0.001412; `A5+C1`'s 0.000907 gain is **inside**
that wider reading, which is reported rather than smoothed over.

Full tables, every arm, both folds, every segment, are in section 8 below and in
`docs/models/rebound/experiments.md` section 11.

---

## 7. Verdict, caveats, and exactly what remains

**Recommended verdict: adopt nothing yet; fund stage 2.** The evidence is strong
and mechanistic, but section 10.5 says the decision is taken on stage-2 numbers
and stage 2 did not run. Specifically:

1. `A5+C1` is a **stage-1 leader, not a winner.** It must be re-run on the served
   `S1_weekly` calendar against the reference on the same calendar.
2. `A5` is an **extrapolation** and carries the risk named in section 4: it is
   systematically short (0.32 pp) and will invert if the league drift stops.
   The PM should weigh a shipped extrapolation against a shipped -1.137 pp bias.
3. `A5+C1` **fails calibration on fold 1** (2.212 pp). Fold 2 selects, so the rule
   does not bind, but two of three C-arms fail the same gate on fold 2 and this is
   a pattern, not a blip.
4. The `B3` block draw needs an **engine change** (a new `book.draw("shot_block", …)`
   family after the make/miss draw), which is the only way to feed a valid 0/1.
   It is not wired, DEFAULT-OFF or otherwise; no engine file was touched.
5. `shot_block` has **no adopted arm**; the rim level gate fails for every arm.
6. `C4` is **NOT RUN** for want of a continuity table before 2027.

Resume commands (each cell ~3 h on a loaded box):

    # stage 2, the served S1_weekly calendar, fold 2
    .venv/Scripts/python.exe scripts/train_rebound_v3_round3.py --stage 2 \
        --folds F2 --feeds --arms A0B0C0
    .venv/Scripts/python.exe scripts/train_rebound_v3_round3.py --stage 2 \
        --folds F2 --feeds --arms A0B0C0 --seed 1
    .venv/Scripts/python.exe scripts/train_rebound_v3_round3.py --stage 2 \
        --folds F2 --feeds --arms A5
    .venv/Scripts/python.exe scripts/train_rebound_v3_round3.py --stage 2 \
        --folds F2 --feeds --arms A5+C1
    # then
    .venv/Scripts/python.exe scripts/grade_rebound_round3_v1.py

The closed loop (500 games x 25 paired seeds, parity against
`docs/ops/parity_reference_windows_v6.json`) is **NOT RUN** and has no partial
state: no engine artifact was exported, no flag added, no `loop.py` line changed.
The served path is therefore bit-identical by construction, and the parity check
was not run because there was nothing to check.

---

## 8. Full tables (generated by `scripts/grade_rebound_round3_v1.py`)

**Stage 1 / F2 / `S0` (SELECTION)**

| arm | n_feat | n_fits | log_loss | brier | L1_pp | L2_B0_pp | L2_B3_pp | calib | gap_pp | resp | slope_q | mono | fit_s | why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A5+C1 | 16 | 1 | 0.644658 | 0.418606 | -0.4444 | -1.1966 | -0.4422 | PASS | 1.674 | PASS | 1.1219 | 4 | 472.2 | combined: A5 + C1 |
| A5+C3 | 16 | 1 | 0.644767 | 0.41868 | -0.3394 | -1.088 | -0.3381 | PASS | 1.67 | PASS | 0.717 | 4 | 471.3 | combined: A5 + C3 |
| A5 | 16 | 1 | 0.645024 | 0.418919 | -0.337 | -1.0887 | -0.3356 | PASS | 1.921 | PASS | 0.6895 | 4 | 295.6 | TREND EXTRAPOLATION: an OLS season-level offset fitted on the fo |
| C1 | 16 | 1 | 0.645182 | 0.41887 | -1.2439 | -1.9847 | -1.2431 | FAIL | 2.281 | PASS | 1.1072 | 4 | 531.5 | prior-season carry, PO round 4's G2 |
| C3 | 16 | 1 | 0.645196 | 0.418884 | -1.1633 | -1.904 | -1.1622 | PASS | 1.966 | PASS | 0.7138 | 4 | 315.4 | shrink toward the league mean only, G1 (the control) |
| C2 | 16 | 1 | 0.645218 | 0.418878 | -1.2592 | -2.0015 | -1.2584 | FAIL | 2.228 | PASS | 1.0814 | 4 | 504.9 | prior-season carry with the prior shrunk by its own reliability, |
| D3 | 17 | 1 | 0.645315 | 0.419058 | -1.1421 | -1.8841 | -1.1413 | FAIL | 2.147 | PASS | 0.6947 | 4 | 308.7 | Decision 9b: conference-game flag |
| D2 | 16 | 1 | 0.64551 | 0.419295 | -1.1309 | -1.8691 | -1.1289 | PASS | 1.974 | PASS | 0.6298 | 4 | 557.4 | Decision 9a: opponent adjustment, iterative |
| A0B0C0 | 16 | 1 | 0.645565 | 0.419188 | -1.137 | -1.8769 | -1.137 | PASS | 1.958 | PASS | 0.6851 | 4 | 631.4 | reference: the served lgbm / C_plus_state, unchanged |
| D1 | 16 | 1 | 0.645647 | 0.419245 | -1.1272 | -1.8671 | -1.1257 | PASS | 1.944 | PASS | 0.6294 | 4 | 590.8 | Decision 9a: opponent adjustment, one_pass |
| A1 | 17 | 1 | 0.645816 | 0.419003 | -0.8288 | -1.5806 | -0.8279 | PASS | 1.676 | PASS | 0.6728 | 4 | 529.2 | season index as a feature |
| A3_2s | 16 | 1 | 0.64618 | 0.419357 | -1.0167 | -1.7721 | -1.0167 | PASS | 1.984 | PASS | 0.6875 | 4 | 505.4 | rolling window: most recent 2 seasons of training rows |
| A2_h240 | 16 | 1 | 0.646239 | 0.419318 | -0.9962 | -1.7522 | -0.995 | FAIL | 2.181 | PASS | 0.6948 | 4 | 308.1 | exponential recency weights, 240-day half-life |
| B1 | 15 | 1 | 0.647099 | 0.420419 | -1.1573 | -1.1573 | -1.1573 | FAIL | 2.03 | PASS | 0.6867 | 4 | 503.4 | drop blocked_f and refit; marginalise over blocks |
| A2_h120 | 16 | 1 | 0.64768 | 0.419753 | -0.8878 | -1.6525 | -0.8857 | FAIL | 2.34 | PASS | 0.6984 | 4 | 519.4 | exponential recency weights, 120-day half-life |
| A2_h60 | 16 | 1 | 0.649226 | 0.420301 | -0.9445 | -1.7018 | -0.9412 | FAIL | 2.674 | PASS | 0.7179 | 4 | 507.4 | exponential recency weights, 60-day half-life |
| A4 | 17 | 1 | 0.649412 | 0.420191 | 0.3059 | -0.4405 | 0.3072 | FAIL | 2.585 | PASS | 0.397 | 4 | 531.5 | within-season as-of league OREB% anchor (PM condition a-ii) |
| A3_1s | 16 | 1 | 0.649568 | 0.420358 | -0.8344 | -1.591 | -0.8307 | FAIL | 2.851 | PASS | 0.6812 | 4 | 541.7 | rolling window: most recent 1 season of training rows |

**Stage 1 / F1 / `S0`**

| arm | n_feat | n_fits | log_loss | brier | L1_pp | L2_B0_pp | L2_B3_pp | calib | gap_pp | resp | slope_q | mono | fit_s | why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A5+C1 | 16 | 1 | 0.619846 | 0.407335 | -0.138 | -0.9017 | -0.143 | FAIL | 2.212 | PASS | 1.0652 | 4 | 456.4 | combined: A5 + C1 |
| A5+C3 | 16 | 1 | 0.619971 | 0.407472 | -0.0143 | -0.7739 | -0.0197 | FAIL | 2.196 | PASS | 0.664 | 4 | 252.9 | combined: A5 + C3 |
| C2 | 16 | 1 | 0.620077 | 0.40741 | -0.706 | -1.4626 | -0.7106 | PASS | 1.68 | PASS | 1.0114 | 4 | 273.8 | prior-season carry with the prior shrunk by its own reliability, |
| A5 | 16 | 1 | 0.620096 | 0.407541 | -0.0752 | -0.8407 | -0.0804 | PASS | 1.9 | PASS | 0.6124 | 4 | 301.7 | TREND EXTRAPOLATION: an OLS season-level offset fitted on the fo |
| C1 | 16 | 1 | 0.620181 | 0.407457 | -0.7204 | -1.4791 | -0.7246 | PASS | 1.761 | PASS | 1.0588 | 4 | 256.0 | prior-season carry, PO round 4's G2 |
| C3 | 16 | 1 | 0.62021 | 0.407515 | -0.6388 | -1.3934 | -0.6434 | PASS | 1.817 | PASS | 0.6456 | 4 | 289.5 | shrink toward the league mean only, G1 (the control) |
| A1 | 17 | 1 | 0.620267 | 0.407562 | -0.5179 | -1.2707 | -0.5226 | PASS | 1.568 | PASS | 0.6069 | 4 | 210.4 | season index as a feature |
| D3 | 17 | 1 | 0.620313 | 0.4076 | -0.6428 | -1.3975 | -0.6475 | PASS | 1.527 | PASS | 0.6195 | 4 | 287.3 | Decision 9b: conference-game flag |
| A0B0C0 | 16 | 1 | 0.620446 | 0.407653 | -0.6494 | -1.4088 | -0.6544 | PASS | 1.629 | PASS | 0.6093 | 4 | 335.9 | reference: the served lgbm / C_plus_state, unchanged |
| A3_2s | 16 | 1 | 0.620446 | 0.407653 | -0.6494 | -1.4088 | -0.6544 | PASS | 1.629 | PASS | 0.6093 | 4 | 255.0 | rolling window: most recent 2 seasons of training rows |
| D2 | 16 | 1 | 0.620645 | 0.407862 | -0.6355 | -1.391 | -0.6398 | PASS | 1.666 | PASS | 0.5835 | 4 | 286.0 | Decision 9a: opponent adjustment, iterative |
| D1 | 16 | 1 | 0.620661 | 0.407891 | -0.6479 | -1.4024 | -0.6511 | FAIL | 2.008 | PASS | 0.5757 | 4 | 346.3 | Decision 9a: opponent adjustment, one_pass |
| A2_h240 | 16 | 1 | 0.62087 | 0.407876 | -0.6332 | -1.4031 | -0.6395 | PASS | 1.764 | PASS | 0.6103 | 4 | 294.8 | exponential recency weights, 240-day half-life |
| A2_h120 | 16 | 1 | 0.621306 | 0.408194 | -0.6653 | -1.4456 | -0.6721 | PASS | 1.999 | PASS | 0.6131 | 4 | 276.5 | exponential recency weights, 120-day half-life |
| A2_h60 | 16 | 1 | 0.62195 | 0.408576 | -0.6634 | -1.4611 | -0.6706 | FAIL | 2.501 | PASS | 0.6079 | 4 | 260.7 | exponential recency weights, 60-day half-life |
| A4 | 17 | 1 | 0.62198 | 0.408835 | -0.7927 | -1.5705 | -0.7982 | PASS | 1.939 | PASS | 0.3134 | 4 | 274.8 | within-season as-of league OREB% anchor (PM condition a-ii) |
| A3_1s | 16 | 1 | 0.622116 | 0.408602 | -0.6175 | -1.4067 | -0.6258 | FAIL | 2.39 | PASS | 0.6067 | 4 | 232.4 | rolling window: most recent 1 season of training rows |
| B1 | 15 | 1 | 0.622119 | 0.409127 | -0.667 | -0.667 | -0.667 | PASS | 1.901 | PASS | 0.6131 | 4 | 206.8 | drop blocked_f and refit; marginalise over blocks |

### Noise floor

- stage 1, `A0B0C0` second-seed retrain on F2: log-loss spread **0.000004**, L1 spread 0.0393 pp. Operative floor = max(spread, published 5-seed SD 6.7e-05) = **0.000067**.
- game-clustered block-bootstrap SE for this cell (published, section 3): 0.001412.

### Decision rule applied mechanically, stage 1 (floor 0.000067)

| arm | F2 gain | x floor | F1 gain | rule1 (beats ref) | rule3 (gates) | rule4 (L1 not worse) | L1_pp | rule5 (slope not reduced) | slope_q |
|---|---|---|---|---|---|---|---|---|---|
| A5+C1 | 0.000907 | 13.5 | 0.0006 | True | True | True | -0.4444 | True | 1.1219 |
| A5+C3 | 0.000798 | 11.9 | 0.000475 | True | True | True | -0.3394 | False | 0.717 |
| A5 | 0.000541 | 8.1 | 0.00035 | True | True | True | -0.337 | False | 0.6895 |
| C1 | 0.000383 | 5.7 | 0.000265 | True | False | False | -1.2439 | True | 1.1072 |
| C3 | 0.000369 | 5.5 | 0.000236 | True | True | False | -1.1633 | False | 0.7138 |
| C2 | 0.000347 | 5.2 | 0.000369 | True | False | False | -1.2592 | True | 1.0814 |
| D3 | 0.00025 | 3.7 | 0.000133 | True | False | False | -1.1421 | False | 0.6947 |
| D2 | 5.5e-05 | 0.8 | -0.000199 | False | True | True | -1.1309 | False | 0.6298 |
| D1 | -8.2e-05 | -1.2 | -0.000215 | False | True | True | -1.1272 | False | 0.6294 |
| A1 | -0.000251 | -3.7 | 0.000179 | False | True | True | -0.8288 | False | 0.6728 |
| A3_2s | -0.000615 | -9.2 | 0.0 | False | True | True | -1.0167 | False | 0.6875 |
| A2_h240 | -0.000674 | -10.1 | -0.000424 | False | False | True | -0.9962 | True | 0.6948 |
| B1 | -0.001534 | -22.9 | -0.001673 | False | False | False | -1.1573 | False | 0.6867 |
| A2_h120 | -0.002115 | -31.6 | -0.00086 | False | False | True | -0.8878 | True | 0.6984 |
| A2_h60 | -0.003661 | -54.6 | -0.001504 | False | False | True | -0.9445 | True | 0.7179 |
| A4 | -0.003847 | -57.4 | -0.001534 | False | False | True | 0.3059 | False | 0.397 |
| A3_1s | -0.004003 | -59.7 | -0.00167 | False | False | True | -0.8344 | False | 0.6812 |

Eligible: ['A5+C1']. **Stage-1 leader by the rule: `A5+C1`** (ties inside one floor broken by simplicity).

### Multi-level evidence (stage-2 cells if present, else stage 1)


by miss type (level pp)

| arm | rim | jump2 | three | ft |
|---|---|---|---|---|
| A5+C1 | 0.3268 | -0.6047 | -0.7892 | -0.4932 |
| A5+C3 | 0.4305 | -0.5026 | -0.6783 | -0.4057 |
| A5 | 0.4192 | -0.4986 | -0.6667 | -0.4133 |
| C1 | -0.6159 | -1.4019 | -1.58 | -0.9398 |
| C3 | -0.5574 | -1.3279 | -1.4818 | -0.8641 |
| C2 | -0.6271 | -1.4164 | -1.6071 | -0.9115 |
| D3 | -0.5386 | -1.3113 | -1.4555 | -0.8477 |
| D2 | -0.5229 | -1.3212 | -1.4336 | -0.8419 |
| A0B0C0 | -0.5114 | -1.3169 | -1.4487 | -0.8835 |
| D1 | -0.5153 | -1.3095 | -1.428 | -0.8805 |
| A1 | -0.2426 | -0.9855 | -1.1249 | -0.6046 |
| A3_2s | -0.4434 | -1.1727 | -1.3061 | -0.7904 |
| A2_h240 | -0.4456 | -1.1589 | -1.2853 | -0.6896 |
| B1 | -0.6001 | -1.2907 | -1.4594 | -0.8877 |
| A2_h120 | -0.3568 | -1.0617 | -1.1752 | -0.5024 |
| A2_h60 | -0.3873 | -1.1268 | -1.2268 | -0.6342 |
| A4 | 0.1082 | -0.0737 | 0.5383 | 0.7972 |
| A3_1s | -0.3577 | -1.0393 | -1.1002 | -0.3143 |

by month (level pp)

| arm | 11 | 12 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| A5+C1 | -0.8577 | -0.6042 | -0.9641 | 0.1966 | 0.1959 | 1.242 |
| A5+C3 | -0.6152 | -0.532 | -0.8671 | 0.2319 | 0.2626 | 1.2284 |
| A5 | -0.6117 | -0.5381 | -0.8702 | 0.2637 | 0.2227 | 1.6678 |
| C1 | -1.6623 | -1.4162 | -1.7763 | -0.6001 | -0.5639 | 0.5141 |
| C3 | -1.422 | -1.3981 | -1.6903 | -0.5805 | -0.5656 | 0.6847 |
| C2 | -1.6874 | -1.3957 | -1.7802 | -0.6201 | -0.6119 | 0.1781 |
| D3 | -1.1614 | -1.0127 | -1.869 | -0.7456 | -0.6677 | 1.0753 |
| D2 | -1.3912 | -1.3331 | -1.6853 | -0.5344 | -0.5468 | 0.8974 |
| A0B0C0 | -1.427 | -1.3096 | -1.6932 | -0.5451 | -0.5285 | 0.8139 |
| D1 | -1.3836 | -1.3782 | -1.6666 | -0.5129 | -0.5391 | 0.4346 |
| A1 | -1.0448 | -1.0296 | -1.3725 | -0.2623 | -0.2782 | 0.975 |
| A3_2s | -1.3293 | -1.2558 | -1.5168 | -0.413 | -0.421 | 1.094 |
| A2_h240 | -1.2978 | -1.2255 | -1.5274 | -0.3755 | -0.4006 | 1.0478 |
| B1 | -1.4364 | -1.3231 | -1.7071 | -0.581 | -0.5602 | 0.8668 |
| A2_h120 | -1.2024 | -1.1216 | -1.4107 | -0.2694 | -0.2714 | 0.9037 |
| A2_h60 | -1.2115 | -1.2067 | -1.4809 | -0.3327 | -0.3344 | 0.9069 |
| A4 | -0.098 | 0.3876 | 0.2646 | 1.0153 | -0.3359 | 1.1713 |
| A3_1s | -1.1694 | -1.1202 | -1.3006 | -0.216 | -0.2367 | 1.2982 |

by site (level pp)

| arm | home | away | neutral |
|---|---|---|---|
| A5+C1 | -0.5505 | -0.2838 | -0.6434 |
| A5+C3 | -0.4288 | -0.1999 | -0.5216 |
| A5 | -0.4282 | -0.1996 | -0.5059 |
| C1 | -1.37 | -1.0683 | -1.428 |
| C3 | -1.2502 | -1.0274 | -1.3416 |
| C2 | -1.3683 | -1.0971 | -1.4532 |
| D3 | -1.2399 | -1.0176 | -1.2444 |
| D2 | -1.2176 | -1.0051 | -1.2749 |
| A0B0C0 | -1.2422 | -0.9905 | -1.2905 |
| D1 | -1.2269 | -0.9796 | -1.3023 |
| A1 | -1.029 | -0.5993 | -0.9515 |
| A3_2s | -1.1833 | -0.8177 | -1.1462 |
| A2_h240 | -1.1867 | -0.7725 | -1.1314 |
| B1 | -1.2687 | -1.0151 | -1.2755 |
| A2_h120 | -1.1521 | -0.6016 | -0.9914 |
| A2_h60 | -1.2501 | -0.6026 | -1.1024 |
| A4 | 0.4788 | 0.4209 | -0.6674 |
| A3_1s | -1.0785 | -0.547 | -1.0096 |

by conference game (level pp)

| arm | conf | nonconf |
|---|---|---|
| A5+C1 | -0.2938 | -0.7032 |
| A5+C3 | -0.2343 | -0.52 |
| A5 | -0.2241 | -0.5309 |
| C1 | -1.0942 | -1.501 |
| C3 | -1.0548 | -1.3498 |
| C2 | -1.1087 | -1.5177 |
| D3 | -1.2234 | -1.0022 |
| D2 | -1.0295 | -1.3051 |
| A0B0C0 | -1.0287 | -1.3233 |
| D1 | -1.0105 | -1.3277 |
| A1 | -0.7345 | -0.9909 |
| A3_2s | -0.8843 | -1.2442 |
| A2_h240 | -0.8725 | -1.2089 |
| B1 | -1.058 | -1.3279 |
| A2_h120 | -0.7633 | -1.1016 |
| A2_h60 | -0.8222 | -1.1548 |
| A4 | 0.4429 | 0.0705 |
| A3_1s | -0.6915 | -1.08 |

by period (level pp)

| arm | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| A5+C1 | -0.3149 | -0.579 | -0.1256 | -1.474 | UNDERPOWERED |
| A5+C3 | -0.2072 | -0.4765 | 0.0244 | -1.6542 | UNDERPOWERED |
| A5 | -0.218 | -0.4631 | 0.1832 | -1.5625 | UNDERPOWERED |
| C1 | -1.0868 | -1.4061 | -0.9505 | -2.3768 | UNDERPOWERED |
| C3 | -1.0069 | -1.3256 | -0.7792 | -2.3987 | UNDERPOWERED |
| C2 | -1.124 | -1.3988 | -0.9525 | -2.4985 | UNDERPOWERED |
| D3 | -1.0076 | -1.2843 | -0.6481 | -2.2109 | UNDERPOWERED |
| D2 | -0.961 | -1.3066 | -0.8178 | -2.1348 | UNDERPOWERED |
| A0B0C0 | -0.9825 | -1.2996 | -0.6119 | -2.3653 | UNDERPOWERED |
| D1 | -0.9783 | -1.2819 | -0.769 | -2.2117 | UNDERPOWERED |
| A1 | -0.7116 | -0.9553 | -0.2125 | -1.8515 | UNDERPOWERED |
| A3_2s | -0.8747 | -1.1647 | -0.6748 | -1.9737 | UNDERPOWERED |
| A2_h240 | -0.8478 | -1.1446 | -1.0908 | -1.981 | UNDERPOWERED |
| B1 | -1.0105 | -1.3115 | -0.7147 | -2.1638 | UNDERPOWERED |
| A2_h120 | -0.7723 | -0.999 | -1.212 | -1.9355 | UNDERPOWERED |
| A2_h60 | -0.8344 | -1.0462 | -1.3894 | -2.7861 | UNDERPOWERED |
| A4 | 0.6578 | -0.0643 | 1.1554 | -0.4948 | UNDERPOWERED |
| A3_1s | -0.7328 | -0.9318 | -1.1227 | -1.902 | UNDERPOWERED |

per-team prior-quintile slope ratio, by season segment (the G4 responsiveness defect; served engine 0.561 / 0.799 / 0.738)

| arm | all | Nov-Dec | Jan | Feb-Apr | gap_pp_by_q |
|---|---|---|---|---|---|
| A5+C1 | 1.1219 | 1.1973 | 1.1821 | 1.0078 | [-0.744, -0.807, -0.25, -0.336, 0.055] |
| A5+C3 | 0.717 | 0.5186 | 0.8506 | 0.8529 | [0.725, -0.031, -0.253, -0.799, -1.131] |
| A5 | 0.6895 | 0.5905 | 0.797 | 0.7295 | [0.902, -0.028, -0.286, -0.898, -1.134] |
| C1 | 1.1072 | 1.1712 | 1.1736 | 1.0015 | [-1.472, -1.6, -1.089, -1.139, -0.769] |
| C3 | 0.7138 | 0.5176 | 0.8356 | 0.8517 | [-0.105, -0.861, -1.067, -1.599, -1.981] |
| C2 | 1.0814 | 1.1251 | 1.1566 | 0.9949 | [-1.412, -1.522, -1.094, -1.237, -0.879] |
| D3 | 0.6947 | 0.5473 | 0.842 | 0.7648 | [0.044, -0.854, -1.08, -1.654, -1.957] |
| D2 | 0.6298 | 0.5024 | 0.7475 | 0.6942 | [0.295, -0.869, -1.021, -1.671, -2.132] |
| A0B0C0 | 0.6851 | 0.5946 | 0.7792 | 0.7231 | [0.114, -0.823, -1.073, -1.717, -1.951] |
| D1 | 0.6294 | 0.5165 | 0.7635 | 0.669 | [0.278, -0.826, -0.979, -1.691, -2.151] |
| A1 | 0.6728 | 0.5741 | 0.7841 | 0.71 | [0.47, -0.501, -0.787, -1.412, -1.675] |
| A3_2s | 0.6875 | 0.5846 | 0.7962 | 0.7273 | [0.212, -0.694, -0.972, -1.559, -1.837] |
| A2_h240 | 0.6948 | 0.5969 | 0.8015 | 0.7343 | [0.214, -0.622, -1.017, -1.544, -1.787] |
| B1 | 0.6867 | 0.5865 | 0.7806 | 0.7353 | [0.095, -0.858, -1.126, -1.725, -1.959] |
| A2_h120 | 0.6984 | 0.6025 | 0.7958 | 0.7431 | [0.308, -0.533, -0.883, -1.441, -1.669] |
| A2_h60 | 0.7179 | 0.6199 | 0.829 | 0.7543 | [0.202, -0.617, -0.923, -1.51, -1.648] |
| A4 | 0.397 | 0.3966 | 0.4622 | 0.3731 | [2.576, 0.832, 0.506, -0.465, -1.377] |
| A3_1s | 0.6812 | 0.59 | 0.7706 | 0.7223 | [0.42, -0.423, -0.89, -1.377, -1.67] |

per-game level error (pp) and the first four weeks of conference play

| arm | n_games | mean | median | sd | mae | P(pred<act) | conf4_n | conf4_level_pp |
|---|---|---|---|---|---|---|---|---|
| A5+C1 | 5593 | -0.3197 | -0.1063 | 6.0405 | 4.8614 | 0.5074 | 82748 | -1.0447 |
| A5+C3 | 5593 | -0.2092 | -0.0432 | 6.0618 | 4.8755 | 0.5038 | 82748 | -0.9666 |
| A5 | 5593 | -0.2066 | 0.0534 | 6.1228 | 4.9151 | 0.4969 | 82748 | -0.9745 |
| C1 | 5593 | -1.1177 | -0.9722 | 6.0329 | 4.9152 | 0.5594 | 82748 | -1.8496 |
| C3 | 5593 | -1.0324 | -0.8863 | 6.0687 | 4.9365 | 0.5561 | 82748 | -1.7896 |
| C2 | 5593 | -1.1318 | -0.9731 | 6.0299 | 4.9103 | 0.5632 | 82748 | -1.8477 |
| D3 | 5593 | -1.0152 | -0.8798 | 6.1047 | 4.9533 | 0.5518 | 82748 | -1.8908 |
| D2 | 5593 | -0.9934 | -0.7964 | 6.1367 | 4.9824 | 0.5507 | 82748 | -1.8194 |
| A0B0C0 | 5593 | -1.0057 | -0.7879 | 6.127 | 4.9645 | 0.5476 | 82748 | -1.7769 |
| D1 | 5593 | -0.9865 | -0.7954 | 6.1368 | 4.9694 | 0.5541 | 82748 | -1.8075 |
| A1 | 5593 | -0.6969 | -0.4589 | 6.104 | 4.9217 | 0.5308 | 82748 | -1.4568 |
| A3_2s | 5593 | -0.8858 | -0.6599 | 6.1362 | 4.956 | 0.5448 | 82748 | -1.6306 |
| A2_h240 | 5593 | -0.8648 | -0.6535 | 6.125 | 4.9487 | 0.5434 | 82748 | -1.6225 |
| B1 | 5593 | -1.0242 | -0.8247 | 6.1665 | 5.0012 | 0.5502 | 82748 | -1.7829 |
| A2_h120 | 5593 | -0.7574 | -0.4925 | 6.1444 | 4.9551 | 0.5314 | 82748 | -1.5355 |
| A2_h60 | 5593 | -0.8144 | -0.6343 | 6.177 | 4.9826 | 0.5357 | 82748 | -1.6008 |
| A4 | 5593 | 0.4554 | 0.6558 | 6.4259 | 5.1861 | 0.4645 | 82748 | 0.1494 |
| A3_1s | 5593 | -0.7032 | -0.4923 | 6.2195 | 5.0111 | 0.5287 | 82748 | -1.4352 |

### Block B: the `blocked_f` engine feed (all feeds, same trained model)


stage 1, `A0B0C0` on F2 (actual live OREB 0.299232)

| feed | level_pp | log_loss |
|---|---|---|
| B0_zero | -1.8769 | 0.647713 |
| B2_asof_cell | 7.0912 | 0.664252 |
| B3e_model | 8.3292 | 0.667026 |
| B3_draw | -1.137 | 0.648339 |
| B_true | -1.137 | 0.645565 |
