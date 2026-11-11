# Heuermann jqmk2 quality-gap check on OUR served sub-models (diagnostic, 2026-11-11)

Report only. No engine, model, table or existing-doc edit. Script: `scripts/diag_heuermann_jqmk2_qualitygap_v1.py`;
scratch output `results/heuermann_jqmk2_qualitygap/{tables.json,tables.md,run.log}` (gitignored).
Question: given our served sub-models' honest walk-forward per-event predictions, does the pre-game OVERALL quality gap
still predict real outcomes (Heuermann sim_v2 jqmk2: 3P g +.31, 2P g +.12 in mature games; make layers short 2.4 / 1.8
pts of favourite margin at close 20+)? Real events only, no simulation. 2025-26 and 2026-27 not read.

## Plain summary (10 lines)

1. Our as-of quality state q (own ridge ratings, off_c - def_c, per possession) is well scaled out of sample: real PPP margin on q difference slope 1.07 (F1) / 1.06 (F2), corr 0.59-0.60.
2. Test 1: given served fg_make, the gap loading g is ~0 in every family and season: 3P F2 -0.019 (SE 0.044), F1 +0.066 (0.047); 2P F2 +0.026 (0.025), F1 -0.017 (0.026). No channel meets "loading present".
3. Power: Heuermann's 3P +0.31 sits ~7 SE and his 2P +0.12 ~4 SE from our F2 estimates, so his magnitudes are excluded on our model, not just unconfirmed.
4. Own/opponent split: nothing replicates across seasons (F1 3P own +0.150 z 2.6, own-defence +0.30 z 2.3; F2 both ~0 or reversed).
5. Test 2: 99.4-99.8% of fold team-games are "mature" (our data starts 2022), so the low-data cell (26 / 68 team-games) is UNDERPOWERED; by season stage nothing is consistent (2P early F2 +0.084 z 2.2, F1 -0.034).
6. Test 3: actual/served ratio is flat across gap quintiles in both seasons and both families (0.994-1.011); no upward slope.
7. Test 4: sorted by OUR q-implied spread, the make-layer margin is ~0 in every bucket (pooled 20+ total -0.24 [-1.56, +1.17]).
8. Sorted by the CLOSE, the 20+ bucket is short: pooled total -1.52 [-2.47, -0.57] (3P -0.64 n.s., 2P -0.88 [-1.55, -0.20]); F1 3P -1.80, F2 2P -1.09. This is MATERIAL by the rule but the channel flips between seasons, and in those games our q implies a gap of only +18.7 points. The deficit goes with what the market knows beyond our q, not with our q gap.
9. Test 5: TOV, shooting-foul trips, bonus trips and 3PA share all have g within 1.2 SE in F2; ORB is fold-1 only (g -0.126, z -2.7, opposite sign to his finding), so it cannot meet the rule.
10. Test 6: the shrinkage mechanism is not supported: 3P gap x log prior attempts is +0.070 (z 2.2, F2) against the hypothesis's sign, and 2P is -0.047 (z 2.7, F2) / -0.028 (z 1.6, F1). Nothing passes the bar.

**Verdict:** the per-channel over-shrinkage defect Heuermann found in his sim is NOT present in our served fg_make, PO or
FT channels given our q. The one MATERIAL reading (close 20+ make-layer shortfall) is NOT carried by our quality gap. Its
owner is the information gap between our as-of team ratings (the `off_rating_*` / `def_rating_*` inputs, which come from
the own ratings and the as-of make rates `off_make_c` / `def_allow_c`) and the market in extreme mismatches. The make
sub-model's shrinkage toward the league mean is not the owner. Anything further needs its own pre-registered round, and
any test that uses the close as a covariate is a yardstick read, not a feature.

## Inputs and provenance

| input | file | notes |
|---|---|---|
| fg_make served predictions | `data/processed/models/fg_make/round_pae/R_s0_F{1,2}/preds_F{1,2}_s0.parquet` | PAE round control arm `R` = served B1 replica (fg_make experiments.md s31.3), S1 monthly, seed 0. 603,849 (F1) + 630,904 (F2) shots |
| PO `first` served predictions | `data/processed/models/fold1_v1/po/pred_first_F1.npy`, `data/processed/models/full_retrain_v1/identity/po_served/pred_first_F2.npy` | served lgbm/C_plus_state S1. Rows = round-2 design `first` rows of the test season in design order (same convention and shape assert as `grade_po_tovlevel_v1.py`). Mean pred vs realised printed in the log; F1 TOV level +0.54 pp (known, PO s30) |
| rebound served predictions | `data/processed/models/fold1_v1/rb/cuts/s2_F1_A0B0C0_seed0/*.pkl` | served spec, S1_weekly, F1 ONLY. Rows = `design_round3` season-2024 rows, segment [refit_k, refit_k+1); n_scored asserted per cut |
| quality q | `data/processed/ratings/own_ratings_{2024,2025}.parquet` | q = (off_c - def_c)/100, points per possession relative to the snapshot league mean; `as_of_date` = game date = fitted strictly before it (`own_ratings.join_as_of`). No market data. Gap = (q_off - q_def)/2, the Heuermann scale |
| prior games | `data/processed/games_universe.parquet`, seasons 2022-2025 | count strictly before the game date (all seasons in our data) and `n_games` from the ratings table (this season) |
| PPP truth for q validation | `data/processed/truth/game_finals_v2.parquet` (verified finals) / possessions counted from the PO round-2 design | |
| closes (yardstick only) | `data/processed/lines/lines_close_v2_verified.parquet`, median close over providers | `lines_open_close_v1.parquet` (named in the brief) holds 2026-27 only, 893 rows, so it cannot grade the folds. 4,806 (2024) / 5,150 (2025) games with a close and both teams' shots |

Fit: Poisson on team-game x family counts, log mu = log(sum p_served) + level + g x covariate. The level is a nuisance
parameter. SEs are cluster-robust by game. This is the same form as Heuermann's d50i fit. Bootstrap: 1,000 game resamples.

## Quality state validation (out of sample)

| fold | games | slope of real PPP margin on q_home - q_away | corr | home intercept (PPP) |
|---|---|---|---|---|
| F1 2023-24 | 5,178 | 1.071 (0.021) | 0.587 | +0.046 |
| F2 2024-25 | 5,445 | 1.056 (0.019) | 0.596 | +0.049 |

## Test 1: make residual loading, g on (q_off - q_def)/2 (SE in brackets; cluster by game)

| family | season | team-games | g | own q/2 | opp q/2 | reading |
|---|---|---|---|---|---|---|
| 3P | F1 2024 | 10,356 | +0.066 (0.047) z 1.4 | +0.150 (0.057) z 2.6 | +0.016 (0.057) | |
| 3P | F2 2025 | 10,890 | -0.019 (0.044) z -0.4 | -0.039 (0.051) | -0.002 (0.052) | not present |
| 3P | pooled | 21,246 | +0.020 (0.032) | +0.043 (0.038) | +0.002 (0.038) | |
| 2P | F1 2024 | 10,356 | -0.017 (0.026) z -0.7 | +0.016 (0.032) | +0.051 (0.032) | |
| 2P | F2 2025 | 10,890 | +0.026 (0.025) z 1.0 | +0.027 (0.029) | -0.024 (0.030) | not present |
| 2P | pooled | 21,246 | +0.005 (0.018) | +0.020 (0.022) | +0.010 (0.022) | |
| rim | F1 / F2 / pooled | | -0.009 (0.030) / -0.007 (0.029) / -0.008 (0.021) | | | not present |
| jump2 | F1 / F2 / pooled | | -0.011 (0.056) / +0.105 (0.051) z 2.0 / +0.047 (0.038) | F2 own +0.152 (0.060) z 2.5 | | not present |

Four-way split (own offence, own defence, opp offence, opp defence; each q component / 2). The only |z| >= 2 cells are
3P F1 own-defence +0.299 (0.129) and 2P F2 opp-defence -0.153 (0.070). Neither repeats in the other season. Full rows are
in `results/.../run.log`.

## Test 2: maturity and season stage (g on the gap)

| family | season | mature (both >= 20 prior games in our data) | low-data | early tercile | mid | late |
|---|---|---|---|---|---|---|
| 3P | F1 | +0.067 (0.047), n 10,330 | -0.51 (0.92), n 26 UNDERPOWERED | +0.034 (0.073) | +0.051 (0.089) | +0.136 (0.087) |
| 3P | F2 | -0.018 (0.044), n 10,822 | -0.18 (0.73), n 68 UNDERPOWERED | -0.064 (0.066) | -0.003 (0.082) | +0.046 (0.085) |
| 2P | F1 | -0.018 (0.026) | +0.30 (0.60) UNDERPOWERED | -0.034 (0.041) | -0.003 (0.048) | -0.004 (0.050) |
| 2P | F2 | +0.026 (0.025) | -0.09 (0.38) UNDERPOWERED | +0.084 (0.038) z 2.2 | -0.007 (0.047) | -0.042 (0.049) |

Terciles are cut on min(prior games this season) of the two teams: early 0-9, mid 9-20, late 20-39. Each tercile has
3,452 (F1) or 3,630 (F2) team-games. The low-data split cannot be tested on our data because the panel starts in 2022.

## Test 3: responsiveness by quality-gap quintile (team-games; actual / served-predicted make rate)

| season | family | Q1 (gap <= -0.049) | Q2 | Q3 | Q4 | Q5 (gap >= +0.049) |
|---|---|---|---|---|---|---|
| F1 | 3P act / pred / ratio | .3159 / .3172 / 0.996 | .3303 / .3289 / 1.004 | .3364 / .3365 / 1.000 | .3453 / .3449 / 1.001 | .3620 / .3579 / 1.011 |
| F1 | 2P | .4633 / .4636 / 0.999 | .4884 / .4870 / 1.003 | .5000 / .5023 / 0.996 | .5139 / .5161 / 0.996 | .5480 / .5490 / 0.998 |
| F2 | 3P | .3166 / .3161 / 1.002 | .3318 / .3307 / 1.003 | .3361 / .3366 / 0.999 | .3424 / .3444 / 0.994 | .3609 / .3591 / 1.005 |
| F2 | 2P | .4658 / .4679 / 0.995 | .4950 / .4949 / 1.000 | .5061 / .5064 / 1.000 | .5242 / .5211 / 1.006 | .5551 / .5552 / 1.000 |

The ratio is flat. The served model already reproduces the 4.6 pp (3P) and 8.9 pp (2P) span across gap quintiles.

## Test 4: make-layer margin, EXPECTED minus REAL, favourite minus underdog (points per game, 95% game-bootstrap CI)

Sign convention: negative = favourites made more than the served model expects. This matches Heuermann's "replay - real".
`q-spr` = our own q gap in the favourite's direction x 70 possessions.

By closing-spread bucket (yardstick only):

| bucket | season | games | q-spr | 3P | 2P | total |
|---|---|---|---|---|---|---|
| 0-3 | F1 / F2 | 1,012 / 1,175 | +0.6 | -0.16 [-0.74,+0.44] / -0.41 [-0.93,+0.08] | +0.22 / -0.62 [-1.12,-0.14] | +0.07 / -1.04 [-1.73,-0.36] |
| 3-6 | F1 / F2 | 1,319 / 1,323 | +3.1 | +0.33 / -0.13 | +0.82 [+0.35,+1.26] / -0.11 | +1.15 [+0.43,+1.88] / -0.24 |
| 6-10 | F1 / F2 | 1,109 / 1,187 | +6.3 | -0.17 / -0.43 | -0.35 / -0.37 | -0.52 / -0.80 [-1.48,-0.08] |
| 10-15 | F1 / F2 | 737 / 745 | +9.7 | -0.03 / -0.22 | -0.63 [-1.23,-0.02] / +0.07 | -0.67 / -0.15 |
| 15-20 | F1 / F2 | 342 / 334 | +13.0 | -0.79 [-1.78,+0.17] / +0.64 [-0.42,+1.70] | +0.08 / -0.67 [-1.68,+0.24] | -0.71 [-2.01,+0.68] / -0.03 [-1.49,+1.31] |
| 20+ | F1 | 287 | +18.3 | **-1.80 [-2.84,-0.82]** | -0.59 [-1.54,+0.35] | **-2.39 [-3.79,-1.02]** |
| 20+ | F2 | 386 | +19.1 | +0.22 [-0.82,+1.23] | **-1.09 [-1.94,-0.26]** | -0.87 [-2.18,+0.52] |
| 20+ | pooled | 673 | +18.7 | -0.64 [-1.40,+0.12] | **-0.88 [-1.55,-0.20]** | **-1.52 [-2.47,-0.57]** |

By our own q-implied spread bucket (no market data):

| bucket | pooled games | 3P | 2P | total |
|---|---|---|---|---|
| 0-3 | 3,081 | +0.21 [-0.12,+0.53] | +0.06 | +0.27 [-0.20,+0.70] |
| 3-6 | 2,657 | +0.17 | +0.12 | +0.29 [-0.16,+0.80] |
| 6-10 | 2,439 | -0.19 | -0.17 | -0.35 [-0.86,+0.14] |
| 10-15 | 1,574 | -0.08 | +0.04 | -0.04 [-0.65,+0.56] |
| 15-20 | 561 | -0.08 | +0.10 | +0.03 [-1.14,+1.15] |
| 20+ | 311 (F1 131 UNDERPOWERED, F2 180) | +0.03 [-1.03,+1.12] | -0.27 [-1.18,+0.59] | -0.24 [-1.56,+1.17] |

By q-gap quintile (pooled), total: Q1 +0.54 [-0.01,+1.08], Q2 +0.02, Q3 +0.26, Q4 -0.45 [-0.99,+0.06], Q5 -0.08 [-0.63,+0.44].

Reading. By the rule, the close-bucket shortfall is MATERIAL: the pooled 20+ total and 2P, F1 3P and total, and F2 2P
each have a CI that excludes 0 with |value| >= 0.5. It is not a stable per-family effect: 3P carries it in F1 and 2P in F2.
The close-bucket pattern is also not monotone (F2 0-3 is -1.04, F1 3-6 is +1.15), so some of this is bucket noise across 12
cells. Under our own quality sort the shortfall disappears. In the close-20+ games our q sees only +18.7 points of gap, so
the residual follows what the market knows about the mismatch beyond our as-of ratings, not a make loading on our q.

## Test 5: other channels (offset design, g on the gap)

| channel (served prediction) | F1 2024 | F2 2025 | pooled | reading |
|---|---|---|---|---|
| TOV per first chance (PO `first`) | -0.018 (0.052) | +0.045 (0.049) | +0.015 (0.036) | not present |
| shooting-foul FT trip per first chance | -0.026 (0.073) | -0.028 (0.068) | -0.027 (0.050) | not present |
| bonus FT trip (non-shooting foul) per first chance | +0.007 (0.084) | +0.033 (0.077) | +0.020 (0.057) | not present |
| 3PA share of FGA (P3 / (Prim+Pj2+P3)) | +0.024 (0.037) | -0.039 (0.032) | -0.010 (0.024) | not present |
| OREB rate on FGA misses (rebound S1_weekly) | -0.126 (0.046) z -2.7; split own -0.097, opp +0.157 (z 2.9) | NOT RUN | | fold 1 only: cannot meet the rule; sign opposite to Heuermann's |

## Test 6: mechanism (does the gap loading fade as the shooting team's own-season sample grows?)

Covariates: the gap, L_c (centred log(1 + the team's prior same-season attempts of that family)), and gap x L_c.

| family | season | gap | gap x L_c | L mean (sd) |
|---|---|---|---|---|
| 3P | F1 | +0.060 (0.048) | -0.029 (0.032) | 5.31 (1.31) |
| 3P | F2 | -0.003 (0.044) | +0.070 (0.031) z +2.2 | 5.40 (1.30) |
| 3P | pooled | +0.025 (0.032) | +0.026 (0.023) | |
| 2P | F1 | -0.022 (0.027) | -0.028 (0.017) z -1.6 | 5.82 (1.38) |
| 2P | F2 | +0.014 (0.026) | -0.047 (0.017) z -2.7 | 5.82 (1.36) |
| 2P | pooled | -0.003 (0.018) | -0.039 (0.012) z -3.2 | |

Under the over-shrinkage hypothesis, the loading is largest where the sample is small and fades as it grows, so the
interaction should be negative. 3P F2 has the wrong sign. 2P has the right sign in both seasons, but F2 |z| is 2.7, which
is below the bar of 3. The 2P main effect is ~0, so any small-sample 2P loading is a few hundredths early in the season.
That is far below Heuermann's +0.12. Not established.

## Could not be run, and why

- No earlier out-of-sample season. Served walk-forward predictions exist only for F1 (2023-24) and F2 (2024-25), so "present" is judged on those two.
- ORB on fold 2. The served rebound F2 S1_weekly cut predictions are not on disk: `identity/rb_served` has 2 of ~23 cuts, and the stage-B box run's F2 `R` cuts were not synced. Producing them would mean refitting about 23 weekly LightGBM models, which is a retrain and out of scope for a report-only diagnostic.
- Heuermann's low-data class. Our panel starts in 2022, so only 26 / 68 fold team-games fall below 20 prior games. The cell is UNDERPOWERED and the season-stage terciles stand in for it.
- PO channels cover `first` chances only. `cont` predictions were not used: rows are ordered by the cont design and the cascade arm. TOV and FT-trip rates are therefore per first chance, not per possession. Shooting fouls on made shots (and-ones) are not in the `first` trip class.
- Closes come from `lines_close_v2_verified.parquet`, not `lines_open_close_v1.parquet`, which has no fold-season rows.
- The fg_make predictions are the PAE-round `R` control (the served B1 spec retrained by the PAE wrapper, seed 0), not the engine's own artifact read back. The engine applies shared shooting G3 and the shot block on top. Neither is a pre-game quality term, so the base make probability is the sub-model under test.
