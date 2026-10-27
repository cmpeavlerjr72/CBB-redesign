# Rim override make bias (PM ruling 17c, 2026-10-27)

Diagnostic only: no retrain, no engine change, no recommendation. Script `scripts/diag_rim_override_bias_v1.py`; full numbers in `docs/tests/rim_override_make_bias_2026-10-27.json`. Sonnet worker.

Data: coordinate-audit master tables `results/coordinate_audit/master_{2022..2025}.parquet` (`cls0` = v1 label, no override; `cls` = v2 label with the 2.27 ft override). Part 4 joins the master 2025 rows to the fold-2 fg_make test rows on (CBBD game, period, seconds remaining, shooter, class, made) plus ordinal: 100.00% of the 235,348 rim and 148,773 jump2 test rows matched. Served predictions = `data/processed/models/fg_make/round_pae/R_s0_F2/preds_F2_s0.parquet`, the reference run of the PAE round, which is the served `ENGINE_FG_MAKE=round4_B1` recipe with the g9 wrapper (default in `adapters.py`); no model was fitted. "Overridden" = a two-point `JumpShot` (v1 jump2) whose located release point is within 2.27 ft of a basket. Rows without a coordinate are never overridden.

## Answer

The override is a relabelling of 3,098 / 3,591 / 4,194 / 12,425 located two-point jump shots (2022-2025) from jump2 to rim, and it moves the class make rates by +0.01 / +0.06 / +0.04 / +0.39 pp on rim (SE 0.014-0.023) and -0.38 / -0.49 / -0.47 / -1.88 pp on jump2 (SE 0.024-0.044); 2022-24 is small and 2025 is three to four times larger because about 70% of 2025 overridden rows sit on the single 0.39 ft pixel the audit flagged (5-6% in 2022-24). Overridden rows make 59% / 63% / 61% / 67% against 38-41% for all located v1-jump2 rows, and in every distance band they make 2-11 pp LESS than native located rim attempts in the same band (2025: 0-1 ft 68.7% vs 70.9%, 1-2 ft 60.7% vs 70.2%, 2-2.27 ft 60.3% vs 66.5%), so the rows the override moves behave like rim shots, not like extra-easy makes. The make-skew of the override is real and measurable on two-point jumpers alone (override rate of makes minus misses within game 1.63 / 2.12 / 1.82 / 6.63 pp, SE 0.10-0.16; the round-1 figures 0.85 / 1.05 / 0.98 / 3.34 pp included three-point JumpShots, which cannot be overridden, in the denominator), but it does not scale with charting quality: located-share quartiles of 2022-24 all sit at 1.5-2.5 pp whether a game is 22% or 100% located, and in 2025 the fully located quartile has the smallest gap (3.9 pp against 6.7-8.9 pp). A make-skewed override is also what true shot physics would produce (shots released within 2 ft go in about 60-70% of the time), so this table cannot separate outcome-dependent placement from that; the audit's independent finding (rim misses are charted farther from the basket than makes) says part of it is placement. What the served fg_make models see: rim class overall is calibrated (level +0.03 pp, SE 0.12; worst decile +1.13 pp; log loss 0.6673) and jump2 overall is calibrated (level -0.02 pp, SE 0.14; worst decile -1.48 pp). Within the rim class the overridden rows are under-predicted by 3.0 pp (SE 0.5; predicted 63.1% vs actual 66.1%, n = 10,485 or 4.5% of rim rows, so about +0.13 pp on the class mean), native located rim rows within 2.27 ft are under-predicted by 9.5 pp (SE 0.2), and native located rim overall is -0.6 pp; in the jump2 class the located rows just outside the override radius are under-predicted by 12.0 pp (2.27-3.5 ft) and 5.1 pp (3.5-5 ft), and rows with no coordinate make 59% against a predicted 39%. The overridden rows are therefore not the worst-calibrated part of either model, and the distance-dependent errors are the location-blind class model failing to see where the shot was. Engine version, stated plainly: the served engine consumes v2 (override ON, 2.27 ft) wherever the model reads a shot-class label, because the callers name `v2` explicitly; `DEFAULT_POSSESSION_VERSION = "v1"` is only the default for callers that name no version and does not gate the override in the served path.

## Part 1. Make rate under v1 vs v2 labels, all attempts (located and unlocated)

| season | class | n v1 | n v2 | make v1 | make v2 | shift (pp) | game-bootstrap SE (pp) |
|---|---|---|---|---|---|---|---|
| 2022 | rim | 235,777 | 238,875 | 0.5903 | 0.5903 | +0.005 | 0.014 |
| 2022 | jump2 | 180,264 | 177,166 | 0.3779 | 0.3741 | -0.378 | 0.024 |
| 2023 | rim | 251,561 | 255,152 | 0.5888 | 0.5894 | +0.062 | 0.014 |
| 2023 | jump2 | 185,150 | 181,559 | 0.3844 | 0.3795 | -0.492 | 0.025 |
| 2024 | rim | 262,998 | 267,192 | 0.5820 | 0.5824 | +0.040 | 0.015 |
| 2024 | jump2 | 190,310 | 186,116 | 0.3972 | 0.3925 | -0.473 | 0.025 |
| 2025 | rim | 254,328 | 266,753 | 0.5822 | 0.5861 | +0.388 | 0.023 |
| 2025 | jump2 | 181,668 | 169,243 | 0.4093 | 0.3905 | -1.881 | 0.044 |

The rim shift is near zero in 2022-24 because the moved rows (make share about 0.6) are close to the class mean; the jump2 class loses its highest-make rows, hence the negative shift. The 2025 jump2 make rate under v1 labels (40.9%) is inflated by 1.9 pp relative to v2 (39.1%): the repair removes the putback mistag from jump2.

## Part 2. Make share of overridden rows vs reference rows (located)

| season | overridden n | overridden make share (SE) | all located v1-jump2 make share | at spike pixel 0.39 ft: share of overridden / n / make share | not at spike: make share |
|---|---|---|---|---|---|
| 2022 | 3,098 | 0.594 (0.010) | 0.380 | 5.8% / 181 / 0.503 | 0.600 |
| 2023 | 3,591 | 0.633 (0.010) | 0.385 | 5.0% / 178 / 0.567 | 0.636 |
| 2024 | 4,194 | 0.607 (0.009) | 0.395 | 6.1% / 254 / 0.642 | 0.605 |
| 2025 | 12,425 | 0.666 (0.005) | 0.407 | 69.8% / 8,676 / 0.688 | 0.614 |

By distance band (overridden make share, SE; native located rim attempts in the same band, make share):

| season | 0-1 ft | 1-2 ft | 2-2.27 ft |
|---|---|---|---|
| 2022 | 0.610 (0.026), n 502; native 0.716 | 0.615 (0.013), n 1,748; native 0.718 | 0.541 (0.018), n 848; native 0.674 |
| 2023 | 0.659 (0.020), n 596; native 0.712 | 0.649 (0.012), n 2,090; native 0.710 | 0.578 (0.017), n 905; native 0.670 |
| 2024 | 0.645 (0.018), n 780; native 0.705 | 0.609 (0.012), n 2,336; native 0.702 | 0.576 (0.016), n 1,078; native 0.662 |
| 2025 | 0.687 (0.005), n 9,106; native 0.709 | 0.607 (0.011), n 2,298; native 0.702 | 0.603 (0.015), n 1,021; native 0.665 |

Reference rows just outside the radius (located, not overridden, make share, 2022 / 2023 / 2024 / 2025): 2.27-3.5 ft 0.513 / 0.515 / 0.540 / 0.531; 3.5-5 ft 0.429 / 0.437 / 0.456 / 0.456. Make share decays smoothly with distance (0.71 native rim, 0.60-0.69 overridden, 0.51-0.54 at 2.27-3.5 ft, 0.43-0.46 at 3.5-5 ft, 0.38-0.41 for all jumpers), with no step at 2.27 ft beyond the 2-11 pp offset between overridden and native rim attempts. The spike pixel in 2022-24 is 5-6% of overridden rows (n 178-254, underpowered for a make-rate statement); in 2025 it is 70% of them and they make 68.8%.

## Part 3. Does the override's make skew vary with located share of the game?

Two-point JumpShot rows only (v1 jump2, located). Games split into quartiles of the share of their attempts that carry a coordinate (within season). Override % of makes / % of misses; within-game difference of the two (pp) with game-bootstrap SE.

| season | all games: % of makes / % of misses / within-game diff (SE) | Q1 (least located) | Q2 | Q3 | Q4 (fully located) |
|---|---|---|---|---|---|
| 2022 | 3.37 / 1.41 / 1.63 (0.11) | 5.01 / 2.43 / 2.27 (0.47); located share 0.22 | 3.18 / 1.33 / 1.61 (0.19) | 3.29 / 1.41 / 1.53 (0.19) | 3.29 / 1.29 / 1.63 (0.19) |
| 2023 | 3.80 / 1.38 / 2.12 (0.11) | 3.59 / 1.53 / 1.77 (0.26); 0.39 | 4.09 / 1.23 / 2.46 (0.19) | 3.78 / 1.33 / 2.19 (0.22) | 3.62 / 1.53 / 1.86 (0.19) |
| 2024 | 3.88 / 1.64 / 1.82 (0.10) | 4.51 / 1.82 / 2.16 (0.28); 0.52 | 4.55 / 1.98 / 2.07 (0.20) | 3.40 / 1.44 / 1.58 (0.19) | 3.31 / 1.40 / 1.63 (0.18) |
| 2025 | 11.32 / 3.91 / 6.63 (0.16) | 12.02 / 4.36 / 6.93 (0.32); 0.96 | 11.19 / 3.68 / 6.66 (0.30) | 14.46 / 4.85 / 8.91 (0.34) | 7.39 / 2.73 / 3.94 (0.28) |

No monotone gradient with charting coverage in any season. In 2022-24 the least-located quartile (22-52% located) shows the same or a slightly larger gap than the fully located one; in 2025 the quartiles are all more than 95% located and the fully located quartile has the lowest override rate and the smallest gap. The 2025 quartile differences reflect game-level variation in how many putbacks the feed mislabels (a pure game-level effect), not a coverage gradient. Overridden make share by quartile is 0.57-0.68 in all cells.

## Part 4. Served fg_make (round4_B1 recipe), fold 2 (2025), decile calibration by group

Level = actual minus predicted make rate (pp, game-block bootstrap SE); worst = largest decile gap (pp); shape = largest decile gap after removing the level.

| class | group | n | predicted | actual | level (SE) | worst decile | shape | log loss |
|---|---|---|---|---|---|---|---|---|
| rim | all | 235,348 | 0.5840 | 0.5844 | +0.03 (0.12) | +1.13 | 1.09 | 0.6673 |
| rim | overridden (v1 jump2) | 10,485 | 0.6309 | 0.6612 | +3.03 (0.51) | +5.94 | 2.91 | 0.6363 |
| rim | overridden at spike pixel | 7,454 | 0.6546 | 0.6859 | +3.14 (0.58) | +7.29 | 4.15 | 0.6223 |
| rim | overridden not at spike | 3,031 | 0.5728 | 0.6005 | +2.77 (1.01) | +12.42 | 9.65 | 0.6708 |
| rim | native, located | 220,069 | 0.5816 | 0.5755 | -0.61 (0.12) | -1.25 | 1.21 | 0.6705 |
| rim | native, located, dist <= 2.27 ft | 62,626 | 0.5988 | 0.6939 | +9.51 (0.23) | +12.54 | 3.03 | 0.6278 |
| rim | no coordinate (unlocated) | 4,794 | 0.5948 | 0.8227 | +22.79 (1.02) | +29.92 | 8.52 | 0.5873 |
| jump2 | all | 148,773 | 0.3907 | 0.3905 | -0.02 (0.14) | -1.48 | 1.46 | 0.6664 |
| jump2 | located, 2.27-3.5 ft | 6,253 | 0.3995 | 0.5199 | +12.04 (0.70) | +16.84 | 4.80 | 0.7142 |
| jump2 | located, 3.5-5 ft | 9,377 | 0.3982 | 0.4489 | +5.07 (0.57) | +7.02 | 1.95 | 0.6892 |
| jump2 | located, > 5 ft | 131,053 | 0.3898 | 0.3770 | -1.28 (0.14) | -3.63 | 2.36 | 0.6611 |
| jump2 | no coordinate (unlocated) | 1,846 | 0.3875 | 0.5937 | +20.63 (1.95) | +33.03 | 12.40 | 0.7564 |

Unlocated rows in this fold are 2.0% of rim and 1.2% of jump2 test rows (the rim decile gap there, +30 pp, rests on 4,794 rows). The jump2 unlocated rows make 59% against 39% overall, and are plausibly the same putbacks the override can only repair when a coordinate exists; they are not repaired by the override and remain in the jump2 class. The 12 pp under-prediction of located jump2 at 2.27-3.5 ft is a distance effect that the location-blind class model cannot carry, with or without the override.

## Part 5. Which version does the engine consume?

Read from code and artifact metadata, not re-run:

- `src/cbb_sim/pbp/possessions.py`: `DEFAULT_POSSESSION_VERSION = "v1"`. This constant only binds callers that name no version (for example the round-1 possession_outcome trainer and the cached round-1 `data/processed/models/possession_outcome/design.parquet`).
- `src/cbb_sim/pbp/events.py`: `RIM_OVERRIDE_MAX_FT = 2.27`; `classify_frame` applies it by default, and v1 outputs on disk predate it.
- fg_make: `src/cbb_sim/models/fg_make.py` has `DEFAULT_VERSION = "v2"`; `data/processed/models/fg_make/design_v2_shotshooter.parquet` carries `possessions_version: v2, rim_override_max_ft: 2.27`; `scripts/build_engine_inputs.py` builds the fg design with `version="v2"` (lines 243 and 591). The served `round4_B1` models were trained and are served on those v2 labels.
- possession_outcome (event stage): `scripts/build_engine_event_round2.py` records the served round-2 team block with `possessions_version: v2, style_source: first_chance`.
- Residual: `build_engine_inputs.py` also reads the cached round-1 possession_outcome design for its base team arrays (line 540). The round-2 block is what the event stage is served; this report did not trace whether any v1-labelled rate column reaches a served model unchanged.

## Limits

The make/miss override gap cannot be split into true shot physics and outcome-dependent placement from these tables; both predict a gap, and the audit's rim-placement finding says some of it is placement. No retrain was done, so the effect of the override on a model fitted without it is not measured. Part 4 groups are defined after the fact from coordinates the models never see, and the served models are location-blind class models; calibration by distance is therefore expected to be uneven.
