# Home/away/neutral site terms: audit, fg_make excess trace and offline bake-offs (Lane G, overnight 2026-09-30)

OFFLINE ONLY. No engine code was edited, no closed loop was run, no served default changed, and nothing is adopted. 2025-26 stayed sealed (`assert_not_sealed` in every script).

**Pre-registrations (committed before the arms ran):**

- e5dd38c: `free_throw/experiments.md` s11, `clock/experiments.md` s30, `possession_outcome/experiments.md` s25.
- ed640e4: `fg_make/experiments.md` s21, written after the S0 diagnosis and before any repair arm.

**Grader.** One blind grader, `scripts/grade_home_site_v1.py`, is used for every arm.

**Inputs:**

- Engine inputs are not used: nothing here is simulated.
- Truth is the hoopR box and the event-layer design rows. The sim column is the served v5b 200-seed fold-2 read already decomposed in `docs/tests/g9_g6_margin_slope_home_diagnostic_2026-09-30.md` section 3.

## 0. Verdicts

| model | line | result | outcome |
|---|---|---|---|
| free_throw | `FT1` cat / `FT2` signed vs served `FT0`, primary `G_site` (FT% FE-adjusted HCA gap) | F2: `FT0` G 0.0026; FT1 +0.0023 (worse), floor 0.0083; FT2 +0.0022 (worse), floor 0.0083. F1: FT1 -0.0019, floor 0.0033; FT2 -0.0015, floor 0.0039. The folds disagree in sign; nothing is beyond the floor. | **REFUTED** (tie, served stays) |
| clock | `C1` site cell dimension / `C2` signed time scale vs served base `C0`, primary `G_site` (seconds per possession) | F2: C1 +0.278 s (worse), floor 0.018. F1: C1 +0.128 (worse), floor 0.131. C2 is identical to C0 (degenerate, see 4.2). | **REFUTED** |
| foul accrual | `A1` cat / `A2` signed vs served constant `A0`, primary `G_site` (silent-foul rate per possession) | Realised HCA is -0.00006 ± 0.00054 (F2) and +0.00019 ± 0.00054 (F1): no site effect exists to model. F2 dG +0.0003, floor 0.0005; F1 -0.0001, floor 0.0004. | **REFUTED** |
| fg_make | `G1` +conf flag / `G2` +site x rating gap / `G4` FE-identified site offset vs served `S0`=B1, primary `G_pts` (summed per-class FE HCA gap, pts/game) | F2: S0 0.402. G1 -0.110 (floor 0.113, not beyond). G2 -0.038 (floor 0.073). **G4 -0.347, floor 0.331: beats by 1.05 floors, every F2 guard holds.** F1: G4 -0.176, floor 0.377, same sign, not beyond the floor. | **G4 = offline winner awaiting a paired closed loop** (marginal; F1 underpowered). G1 and G2 are REFUTED. |

## 1. Method

**Site.** Site is the offence's site: +1 home, -1 away, 0 neutral. Each game contributes one row per side, so a single fit covers the offence and defence perspectives together.

**HCA.** `rate = mu + off_FE + def_FE + b_home[home] + b_away[away]` on (game, offence) aggregates weighted by the model's denominator, fitted separately on the realised rate and on the predicted rate. HCA = `b_home - b_away`.

**Points.** Points = HCA x `P_ref x dPPP/drate`, with P_ref = 67.875. The points per unit rate are:

| rate | points per unit |
|---|---:|
| TOV/poss | -69.1 |
| FTA/poss | 18.4 |
| FT% | 19.1 |
| OREB% | 35.4 |
| rim share | 22.9 |
| 3PA share | 13.3 |
| rim FG% | 43.2 |
| jumper FG% | 27.4 |
| 3P% | 68.1 |

**Offline predictions** are held-out S1 predictions at REAL states:

- possession_outcome uses the served `round2_s1` artifacts on every fold-2 chance (`--part po_realstate`).
- fg_make, free_throw and clock use spec-identical refits of the served spec.

**The reference-state harness overstates site effects.** The g9ws `preds_v1` F0 harness predicts at fixed reference states (score 0, period 1, 900 s). There it overstates possession_outcome's TOV site effect by +0.44 pts per game against the same artifacts at real states. Its rebound number (+0.29) is therefore NOT used as evidence.

## 2. Audit: every served sub-model, fold 2 (2024-25), points per game per team side

The sim column is the exact channel decomposition of section 3 of the diagnostic. It closes to **3.41 - 3.06 = +0.35** with zero residual by construction.

The offline column is each model's own site error at real states. **Residual = sim - offline**: the part of the sim's HCA error that no model's site term owns (simulated state, cascade, derived channels, linearisation).

| sub-model (served) | site feature, encoding | realised HCA (target units; pts) | offline predicted HCA | offline gap pts | sim gap pts (exact) | residual (sim - offline) |
|---|---|---|---|---:|---:|---:|
| possession_outcome `round2_s1` first (lgbm) + cont (cascade): TOV | `site_home`, `site_away` one-hots, offence view (neutral reference) | -0.95 pp per chance (SE 0.09); 0.72 pts | -0.89 pp (first -0.90, cont -0.73) | -0.05 | -0.22 | -0.17 |
| possession_outcome: FT-trip class | same | +1.07 pp per chance (0.09); 0.48 pts | +1.16 pp (first +1.14; cont +1.21 vs real +0.71, underpowered: SE 0.23) | +0.04 | -0.09 | -0.13 |
| possession_outcome: shot mix | same | rim share +0.10 pp (0.19), 3PA share +0.35 pp (0.16) | +0.34 / +0.46 pp | +0.07 | -0.09 | -0.16 |
| fg_make `round4_B1` (3 classes) | same one-hots | rim +1.62 pp (0.24), jumper +1.20 (0.30), three +0.85 (0.22); 1.60 pts | +1.98 / +1.55 / +1.07 pp | **+0.40** (rim +0.16, jumper +0.10, three +0.15) | **+0.68** | +0.28 (of which about +0.16 is the margin-channel vs rate-level linearisation: rate-level sim gap is +0.52) |
| rebound `s1_weekly` | same one-hots | OREB% +0.87 pp (0.17); 0.31 pts | +0.86 pp at real states (served `s1_weekly`, 23 refits, `--part rb_realstate`); the reference-state harness had +1.68 pp | -0.00 | +0.04 | +0.04 |
| derived rebound chances (misses) | no model | - | - | - | -0.12 | -0.12 |
| free_throw `s1_conf_aligned` | **none** | FT% +0.58 pp (0.24); 0.11 pts | +0.83 pp WITHOUT any site feature (state and shooter features carry it) | +0.05 | +0.03 | -0.02 |
| clock `v5b_glat_pmean` | carried in `TEAM_COLS`, **not consumed** | home-offence duration -0.03 s per possession (SE 0.03; F1 -0.14, SE 0.03) | -0.08 s (tempo features) | < 0.01 (bound: 0.05 s x 70 possessions = 3.5 s per team-game) | +0.12 (pace +0.05, parity +0.07) | +0.12 |
| foul accrual (served scalar 0.123346) | **none** (site LUT default-off) | silent-foul rate -0.006 pp per possession (0.054): **no site effect** | 0 | 0.00 | (inside FT trips) | - |
| usage U1 | none | not measured (no team-level scoring rate) | - | - | - | - |
| rotation (R2 S1, reference) | none | not measured | - | - | - | - |
| late_game | not served (`ENGINE_LATE_GAME=off`); its model carries `site_neutral` | - | - | - | - | - |
| **total** | | **3.06 (close 3.07)** | | **+0.51** | **+0.35** | **-0.16** |

**Closure.** Offline gaps (+0.51) plus residual (-0.16) equal the sim's +0.35 exactly. The residual is stated per channel above. The rebound offline cell was run after the first draft of this table: real-state gap -0.001 pts (realised +0.86 pp, SE 0.17; predicted +0.86 pp), so its carried 0 was exact to the stated precision.

**What the audit shows:**

1. **fg_make is the only model whose own site term is materially wrong:** +0.40 pts offline (+0.50 on F1), the same sign as the sim's +0.68.
2. **possession_outcome's and rebound's site terms are right at real states** (PO -0.05 to +0.07 per channel; rebound -0.00). The sim's -0.40 on the PO channels is therefore NOT a site-feature defect. It is produced by the simulated state and cascade, and its owner is unidentified tonight. Candidates: in-sim score/bonus/transition state, which differ by site in the sim.
3. **free_throw has no site feature, yet already over-predicts home FT% by +0.25 pp (+0.05 pts) through state and shooter features.** Adding a site term does not help (section 4.1).
4. **Clock and foul accrual have no realised site effect worth modelling.** Clock: -0.03 ± 0.03 s per possession, inconsistent across folds. Silent fouls: zero.

The rule gap for these three models is closed by evidence (refuted arms), not by a feature.

**Per-possession-type (first vs continuation chances), possession_outcome.** First chances are calibrated by site:

| target | realised HCA | predicted HCA |
|---|---:|---:|
| TOV | -0.96 pp | -0.90 pp |
| trip | +1.12 pp | +1.14 pp |

The continuation model over-predicts the trip and rim-share site effects:

| target | realised HCA | predicted HCA |
|---|---:|---:|
| trip | +0.71 pp (SE 0.23) | +1.21 pp |
| rim share | +0.20 pp (0.42) | +1.11 pp |

Continuations are 13% of chances, so these cells are underpowered. They are worth about +0.03 pts each.

## 3. fg_make's +0.68: why

Source: `results/home_site/fgdiag_v1.{json,log}`, S0 = served B1 refit, reproducing round-4 log loss to six decimals.

**1. Only the site columns carry home advantage.** With them zeroed, the strength-adjusted predicted HCA is about 0 in every class:

| class | site columns zeroed | full prediction | realised |
|---|---:|---:|---:|
| rim | -0.0005 | +0.0198 | +0.0162 |

The site columns alone are worth +0.88 / +0.45 / +0.69 pts (rim, jumper, three) of predicted HCA, against realised +0.70 / +0.33 / +0.58.

**2. The site term is larger where the home side is stronger.** It is a tree site effect conditional on noisy as-of features. The rim make-probability site term (F2) by segment:

| segment | site term |
|---|---:|
| conference | 0.019 |
| non-conference | 0.027 |
| home-rating-gap quintile 1 | 0.018 |
| home-rating-gap quintile 5 | 0.027 |
| November | 0.028 |
| February | 0.016 |

F1 repeats the conference split (0.019 vs 0.029).

Home teams are stronger than their opponents. The mean offence-minus-defence rating gap on home rows is:

| season | all home rows | non-conference home rows |
|---|---:|---:|
| 2022 | +1.2 | +2.6 |
| 2025 | +3.4 | +9.1 |

**3. The realised strength-adjusted HCA is smaller than the site term** and shows no monotone season trend (rim 0.021 / 0.018 / 0.013 / 0.016 over 2022-2025). Train-season drift is not the cause.

**4. Raw calibration by site splits by conference.** Rim, F2, predicted minus realised, home minus away:

| segment | value | SE |
|---|---:|---:|
| conference | +0.55 pp | 0.27 |
| non-conference | -1.09 pp | 0.38 |

The pooled site term is a compromise between a strength-confounded non-conference effect and a smaller conference effect.

**5. Not shot-quality mix and not a leaky feature.** The FE-adjusted excess is present in all three classes. The per-class FE uses the same rows for realised and predicted, so the mix is held. The only site-correlated inputs are the as-of team rates and ratings, which are pregame and leak-tested.

**6. By month and season type.** The early-season under-prediction of the home side (Nov rim -1.41 pp) and the January over-prediction (+1.38 pp) are the same compromise moving with the share of buy games. Postseason cells are **underpowered** (n under 1,700).

**Mechanism.** The tree's site coefficient is not the FE-identified home effect. It also absorbs the part of the home side's strength edge that the as-of features under-capture: omitted-variable bias, largest in buy games.

This is the section-3 cancellation seen from inside one model. fg_make's site term over-produces home advantage by about +0.4-0.7 pts, while the sim's team part under-predicts the home side's strength (-0.34 vs the close). The owner of the second half is the as-of team strength features (team_rate_estimator / drift-anchor territory), not fg_make.

## 4. Bake-offs (offline, folds 1 and 2)

The shared definitions are in `free_throw/experiments.md` s11.

**Primary.** `G_site` = |predicted HCA - realised HCA| (FE-adjusted).

**Floor.** The floor is max(seed-refit |dG|, 2 x paired game-bootstrap SE). The bootstrap component dominates: it is about 2 SE of the realised single-season HCA. This is a power limit of the primary metric (one season per fold).

### 4.1 free_throw

Arms ran on the S1 monthly calendar (pre-registered cost deviation). `results/home_site/ft/grade_v1.json`.

| fold | arm | pred HCA | realised | G_site | dG (floor) | log loss (vs FT0) | shooter-quintile slope ratio | raw home - away residual | per-team mean abs home-away residual |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| F2 | FT0 (served) | +0.83 pp | +0.58 (0.24) | 0.0026 | - | 0.575237 | 0.962 | +0.55 pp | 0.0375 |
| F2 | FT1 cat | +0.09 | | 0.0049 | +0.0023 (0.0083) | -0.000094 (boot SE 0.000086; seed floor 0.000004) | 0.955 | -0.20 | 0.0371 |
| F2 | FT2 signed | +0.10 | | 0.0048 | +0.0022 (0.0083) | -0.000026 | 0.960 | -0.18 | 0.0372 |
| F1 | FT0 | +0.83 | +0.63 (0.24) | 0.0020 | - | 0.578044 | 0.982 | +0.58 | 0.0344 |
| F1 | FT1 | +0.63 | | 0.0001 | -0.0019 (0.0033) | -0.000027 | 0.982 | +0.37 | 0.0342 |
| F1 | FT2 | +0.59 | | 0.0005 | -0.0015 (0.0039) | -0.000005 | 0.984 | +0.31 | 0.0342 |

**Reading:**

- The site term learns a NEGATIVE conditional home effect: home FT% is lower than the state and shooter features predict.
- It over-corrects on F2 and is right on F1. Neither fold moves beyond the floor.
- **REFUTED.** Ties go to the served spec.
- The rule gap is closed by evidence. The served FT model's predicted site difference (+0.83 pp) already exceeds the realised (+0.58 to +0.63 pp), by about 0.05 pts.
- FT1's log-loss gain on F2 (-0.000094) is beyond the seed floor (0.000004), but it is about 1 bootstrap SE and reverses its site gap. It is not a win under the registered rule.

### 4.2 clock

`results/home_site/clock/grade_v1.json`. Realised HCA is in seconds per possession for the home offence.

| fold | arm | pred | realised | G_site | dG (floor) | CRPS_trunc (dCRPS) | tempo-quintile slope ratio |
|---|---|---:|---:|---:|---|---|---:|
| F2 | C0 (served base) | -0.078 | -0.026 (0.031) | 0.052 | - | 4.899455 | 0.765 |
| F2 | C1 site cell dimension | -0.356 | | 0.330 | +0.278 (0.018) | -0.0003 (carried floor 0.00684) | 0.765 |
| F2 | C2 signed scale | = C0 | | 0.052 | 0 | 0 | 0.765 |
| F1 | C0 | -0.063 | -0.139 (0.033) | 0.076 | - | 4.858358 | 0.784 |
| F1 | C1 | -0.343 | | 0.204 | +0.128 (0.131) | -0.0024 | 0.784 |
| F1 | C2 | = C0 | | 0.076 | 0 | 0 | 0.784 |

**C1.** The site cell learns a -0.35 s home-offence effect, which is strength and score-state confounded. It worsens `G_site` on both folds, and its CRPS gain is inside the carried floor.

**C2 is degenerate.** Its censored log-likelihood is flat for |beta| <= 0.005, because `round(T*exp(beta))` is the identity on the 0-90 s integer grid for |beta| below about 0.0056. The argmax tie then returned -0.005 and the arm equals C0. This is a design flaw of the pre-registered arm (integer rounding) and is recorded, not repaired post hoc. A mass-splitting version would be a new pre-registration, and the realised effect it would chase is -0.03 ± 0.03 s, so it is not recommended.

**REFUTED.**

### 4.3 foul accrual

`results/home_site/foul/grade_v1.json`. The target is a silent defensive foul in the possession.

**Realised HCA:**

| fold | realised HCA | train-season rates (home / away / neutral offence) |
|---|---|---|
| F2 | -0.00006 ± 0.00054 | 0.0771 / 0.0768 / 0.0789 |
| F1 | +0.00019 ± 0.00054 | |

The A2 logit slope is b = +0.002.

**Results.** Both site arms move `G_site` by at most 0.0003, inside a floor of about 0.0005. Log loss is unchanged (±0.000003). Per-team mean absolute home-away residual: 0.0080 for A0 vs 0.0080 for A1.

**REFUTED: there is no site effect in silent-foul accrual.** It follows that the sim's -0.09 FT-trip shortfall is not the accrual scalar's missing site term. The PO trip class reproduces the realised trip HCA at real states (section 2), so the shortfall is in the simulated state/cascade residual.

### 4.4 fg_make

`results/home_site/fg/grade_v1.json`, `fgarms_v1.{json,log}`.

`G_pts` sums the per-class absolute FE gap in points. Its floor is max(seed component 0.036 on F2 / 0.025 on F1, 2 x the combined bootstrap SE).

| fold | arm | G_pts | dG_pts (floor) | beats | rim / jumper / three G_site |
|---|---|---:|---|---|---|
| F2 | S0 (served B1) | 0.402 | - | - | 0.0036 / 0.0035 / 0.0022 |
| F2 | G1 +conf flag | 0.292 | -0.110 (0.113) | no | 0.0019 / 0.0036 / 0.0017 |
| F2 | G2 +site x gap | 0.364 | -0.038 (0.073) | no | 0.0029 / 0.0044 / 0.0017 |
| F2 | **G4 FE-identified offset** | **0.055** | **-0.347 (0.331)** | **yes (1.05 floors)** | 0.0001 / 0.0007 / 0.0004 |
| F1 | S0 | 0.506 | - | - | 0.0076 / 0.0027 / 0.0015 |
| F1 | G1 | 0.425 | -0.080 (0.088) | no | |
| F1 | G2 | 0.458 | -0.047 (0.091) | no | |
| F1 | G4 | 0.330 | -0.176 (0.377) | no (same sign) | 0.0046 / 0.0014 / 0.0014 |

**G4 guards (F2, all hold).**

| guard | rim | jumper | three |
|---|---|---|---|
| log loss vs S0 | -0.000004 | +0.000063 (S0 seed floor 0.000112) | -0.000060 |
| offence-quintile responsiveness ratio, G4 vs S0 | 1.019 vs 1.033 | 0.871 vs 0.878 | 0.745 vs 0.741 |
| raw site calibration, G4 vs S0 | -0.37 pp vs 0.00 (2 SE ≈ 0.44 pp) | -0.40 pp vs 0.00 (2 SE ≈ 0.55 pp) | 0.00 vs +0.18 pp |

- Responsiveness is not flattened.
- Raw site calibration WORSENS in rim and jumper, within the 2-SE guard. This is the expected exposure of the strength under-capture that S0's site term was compensating.
- F1 does not reverse the primary. On F1, however, the jumper log loss is worse by +0.000214 against a seed floor of 0.000132. The guard is registered on F2, but this is recorded as a caveat.

**Multi-level evidence for G4 (summed FE gap in points by segment).**

| fold | S0 conference | S0 non-conference | G4 conference | G4 non-conference |
|---|---:|---:|---:|---:|
| F2 | +0.49 | +0.27 | +0.17 | -0.33 |
| F1 | +0.38 | +0.72 | +0.05 | -0.05 |

- **Per game:** S0's site term varies from game to game (rim SD 0.46 pts, rising with the mismatch). G4's is about constant (SD 0.03): one home effect, not a strength proxy.
- **Per team:** the mean absolute home-away residual is unchanged (rim 0.0400 vs 0.0398).
- **G4 fitted offsets** (FE on the refit's training rows, rim, b_home - b_away): 0.0167 F2 and 0.0167-0.0190 F1. They are stable across the six refits.

**G1 and G2.** G1 improves log loss beyond the seed floor in rim (-0.000136) and three (-0.000080); jumper (-0.000104) is inside its 0.000112 floor. It does not improve `G_pts` beyond its floor. It moves the excess from conference to non-conference games: F2 conference +0.21 vs non-conference +0.69. G2 adds per-game site variance (rim SD 0.75) without reducing the gap.

## 5. What a paired closed loop for fg_make G4 needs (artifacts exported tonight; engine flag and loop NOT built)

1. **Artifacts: DONE.** `scripts/train_fg_make_v4_site.py --arms G4 --folds F2 --seeds 0 --export` wrote the fold-2 G4 S1 schedule to `data/processed/models/fg_make/round4_site/G4/`: 3 classes x 6 refits plus `manifest_FGA_*.json` in the round-4 format, with `max_train_date` and an `offset` block per artifact. The directory is 25 MB and gitignored (`round*/`); sync it with `scripts/hf_sync_data.py` before an AWS run.
   - Each artifact's model is `cbb_sim.models.fg_make_site_offset.SiteOffsetModel`, a new module that edits no existing file. Its `predict_proba` takes the declared features (B1 without site, then `site_home`, `site_away`) and adds the logit offset.
   - Export parity with the graded predictions is exact: max abs 0, asserted per refit. The re-run reproduced the graded G4 predictions bit for bit.
2. **Engine: one registry line, not written.** Add `("round4site_", "round4_site", "scripts/train_fg_make_v4_site.py")` to `_FG_DATED_ROUNDS` and a `"round4_site"` entry to `_FG_ROUND_NOTE` in `src/cbb_sim/engine/adapters.py`. `ENGINE_FG_MAKE=round4site_G4` then loads through the existing `_load_dated` path; no predict code changes.
   - Smoke test (`round4_B1` stays the default, so the off path is untouched by construction): with the registry patched in-process only, `FgMakeAdapter.load(inp_v3, "F2", "decision8", "round4site_G4")` loads all 18 artifacts, passes the manifest leak checks and predicts.
   - The parity reference must still be run when the line is added.
3. **Run.** A paired closed loop on v3 inputs with `CBB_TRUTH=verified_v1` under Decision 12 floors. Read the per-channel site table (`scripts/diag_g9_g6_margin_v1.py --part extra` on the new run) plus G6 and G9.
4. **Expected result, stated in advance.**
   - The fg site channels fall by about 0.35-0.5 pts.
   - G6 non-neutral (+5.69 vs 5.74) then FAILS low by about 0.4. This exposes the team-part compression (-0.34) that the excess was cancelling.
   - Under Decision 11 that is VALIDATED-PENDING-SHIP-ACTION, and the ship action is the strength-feature fix (team_rate_estimator / drift anchor), not a reversal of G4.

## 6. NOT RUN, limitations, incidents

**NOT RUN:**

- **Usage and rotation realised site effects.** They have no team-level scoring rate. Their only route to points is the shooter weighting, which the g9ws Pw freeze priced at k = 0.0002.
- The site x rating-gap (`int`) arms for free_throw, clock and foul accrual: no rating in those models (registered as NOT ENTERED).
- **FT on the served conf-aligned calendar** (registered cost deviation).
- **The sim's -0.40 on PO channels is not traced.** It is a sim-state/cascade residual, not a site feature; its owner is unknown.

**Power.** Every per-class and per-fold `G_site` floor is about 2 SE of a single season's realised HCA. G4's F2 win is 1.05 floors and its F1 effect is 0.47 floors. Treat it as a candidate for the closed loop, not as an established improvement.

**Incidents.** None caused. Core cap 2 held, with two 1-thread chains and graders run only in a free slot. Only my own PIDs were started. Other lanes' modified files were never staged.

## 7. Session log (wall clock, EDT)

| time | event |
|---|---|
| 20:41 | start (rules read) |
| 20:44 | audit part run (24 s) |
| 20:49-20:57 | fg S0 F2+F1 (2 threads; 247 s + 271 s) |
| 20:57 | foul arms (seconds) + fgdiag |
| 20:58 | FT/clock chain launched (1 thread) |
| 21:04 | pre-registrations e5dd38c (pushed) |
| 21:04 | fg arms chain launched (1 thread) after ed640e4 |
| 21:13 | FT seed 0 done (about 150 s per cell) |
| 21:19 | FT seed 1 done |
| 21:29 | clock done (29 s per C0/C1 cell, about 230 s per C2) |
| 21:38 | FT and clock graded |
| 21:45 | PO real-state audit (7 min) |
| 21:45 | fg arms done (about 8 min per F2 cell, about 5 min per F1 cell) |
| 21:58 | fg S0 seed 1 done |
| 22:19 | fg graded |
| 22:23 | results commit f191f1e |
| 22:23 | rebound real-state audit (1 min) |
| 22:24-22:30 | G4 export run (323 s, 1 thread), determinism and parity verified, adapter smoke test |

## 8. Proposed text for PM-owned docs (not edited)

**LEARNINGS:** "A tree's site coefficient is not the home effect. Conditional on noisy as-of strength features it absorbs the home side's unmeasured strength edge (buy games), so fg_make's site term over-produced HCA by +0.4-0.5 pts offline while the sim's team part under-predicted home strength. Identify site against team fixed effects (an FE-identified offset), and fix strength in the strength features."

**PROJECT_STATUS:**

- "Site rule audit closed for free_throw, clock and foul accrual: arms refuted, no realised effect to model or already over-produced by state features."
- "fg_make G4 offline winner (1.05 floors F2), needs engine flag and closed loop, expected to expose the G6 non-neutral cancellation."
