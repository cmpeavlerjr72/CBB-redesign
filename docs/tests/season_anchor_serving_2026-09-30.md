# Season-drift anchor O served at predict time: `ENGINE_SEASON_ANCHOR` (lane M, 2026-09-30)

DEFAULT OFF. Nothing selected or adopted. This closes the gap in `docs/ops/engine_inputs_v3_tag_path_2026-09-30.md` section 4: an anchored model (LightGBM fitted with an init_score offset) could not be served, because LightGBM's `predict` never adds the offset back.

## 1. What was built

| file | what |
|---|---|
| `src/cbb_sim/engine/season_anchor_serving.py` (new) | the mode. `ENGINE_SEASON_ANCHOR=off` (default) or a path to an offsets `.npz`. `attach()` selects each engine game's offset row by game_id and binds it to the adapters; `predict_with_offset()` is the raw-score + offset softmax (`season_anchor.predict_proba_with_offset`'s formula). Rows whose offset is exactly zero use the model's own `predict_proba`, so zero offsets are bit-identical to the unanchored path. |
| `src/cbb_sim/engine/adapters.py` (10 small hunks) | `EventAdapter.anchor_marks / anchor_first (G, 6)`; `ReboundAdapter.anchor_marks / anchor_oreb (G,)`; in `predict`, the `first` branch (all six classes) and the rebound S1 branch (OREB column only, as `train_rebound_v3_round3.predict_arm`) call `predict_with_offset` when the offset is set; `Adapters.load` reads the switch. With it off, no flag is added and nothing is imported unless an artifact is marked anchored, which is then REFUSED. `cont` (cascade) takes no offset, as the trainer fits it. |
| `scripts/build_engine_anchor_offsets_v1.py` (new) | writes `<input-dir>/anchor_offsets_<fold>_<season>.npz` (+ `.json`) next to a tagged input dir (`build_engine_inputs_v3_tag_v1.py` is not edited). `po_first (G, 6)` and/or `rb_oreb (G,)`, per game, from `season_anchor.anchor_O` on the trainers' OWN rows (PO: round-2 design `first` rows, class shares; rebound: `RB.fold_slices` rows, live OREB share), each engine game entered as a zero-weight row on its date. It is not the E3 table's `L`. `--zeros` is for proof (b) only. |
| `scripts/train_possession_outcome_s1_par_anchor_artifacts_v1.py` (new) | lane C's anchored PO trainer, with the same arguments, PLUS the engine artifact dir. It runs the round-2 builder unedited over memoised pool fits, as lane J's trainer does. The `first` joblibs are `PO.LgbmArm` around the anchored booster and are marked `"anchor": {...}`, including `Lbar` and the day-0 priors. |
| `scripts/train_rebound_v3_par_anchor_artifacts_v1.py` (new) | lane C's anchor O combined with lane G's artifact worker and writer (G's wrapper refused offset arms). The segments are marked anchored. The checkpoint also keeps each scored row's index, game_id and `_a5_off`. |
| `scripts/diag_season_anchor_serving_proofs_v1.py`, `tests/test_season_anchor_serving.py` (new) | proofs (c) and (d); unit tests: offline formula, zero-offset bit identity, the guards, and realignment by game_id. |

Guards: an artifact marked anchored is refused when the mode is off (demonstrated below). Non-zero offsets on unmarked artifacts are refused; this is what happens on the box if the overlay is not mounted (demonstrated). A marked family with no offsets array is refused, and so is a game with no offset row.

## 2. Proofs

| # | claim | result |
|---|---|---|
| a | mode off is bit-identical to `docs/ops/parity_reference_windows_v6.json` | `run_engine.py --seeds 5 --max-games 60 --workers 2` at `936191f` (the adapters edit): digest `0d4ddccc64d7...`, **PASS, bit-identical** |
| b | mode on with all-zero offsets is bit-identical to mode off | `ENGINE_SEASON_ANCHOR=results/lanem/anchor_offsets_F2_2025_ZEROS_enginev2.npz` (po_first and rb_oreb zeros, served artifacts), same run: games (300 x 28) and players (4,882 x 16) are `DataFrame.equals` to the mode-off run. The digest differs ONLY in the two flag fields that record the switch (`flags.ENGINE_SEASON_ANCHOR`, `meta.adapter_flags`); no game field differs |
| c1 | PO: lane C's full-size anchored smoke `first` model (one refit 2024-11-01, E3_v2) served through `EventAdapter.predict` + the offsets file equals the pkl's own offline predictions | 8 games spread 2024-11-04 .. 2025-04-07, 1,056 possessions (design state and team values of those possessions): **max abs diff 0.0**. The same model without its offset differs by up to 0.0371; the mean absolute shift from the offset is 0.43 / 0.75 / 1.97 / 1.14 / 0.34 / 0.18 pp (TOV / rim / jump2 / 3 / FT shooting / FT bonus) |
| c2 | PO: complete anchored artifact set (the new wrapper, TEST hooks: every 60th game, 15 trees) through the REAL `EventAdapter.load` + `attach`, vs the model's offline formula with the trainer's own offsets, per S1 segment | 12,649 possessions, 91 games, all 5 segments: **max abs diff 0.0**. With the mode off, the same load is refused |
| c3 | rebound: anchored smoke artifact (new wrapper, one weekly refit 2024-11-04, full size, E3_v4, n_jobs 1) through the REAL `ReboundAdapter._load_dated` + `attach`, vs the worker's own offline `p` | 8 games, 613 misses: **max abs diff 0.0**. Without the offset the difference is up to 0.0271, and the mean OREB shift from the offset is +1.40 pp. The trainer's `_a5_off` equals the design rebuild exactly |
| d | fold-2 offsets equal `season_anchor.py`'s level for every date (its own definition, not E3's `L`) | po_first: 151 engine dates / 5,710 games. (d1) `anchor_O` on design rows only: max abs 0.0. (d2) the module's serving form `asof_level_live` on every engine date: 0.0. (d3) the trainer's own fitted offsets (`add_anchor_columns`), 742,025 rows: 0.0. rb_oreb: 151 dates, 0.0 / 0.0 / 0.0 over 392,538 trainer rows. Day-0 levels: rebound 0.29155 (= the 2024 end level; `Lbar` 0.28704); PO class shares 0.1536 / 0.2569 / 0.1940 / 0.2841 / 0.0608 / 0.0505. Offset ranges: po [-0.150, +0.181], rb [+0.022, +0.110] logit |

Consumption: `build_engine_inputs_v3_tag_v1.py --po-artifacts <tiny anchored PO dir> --rb-artifacts <anchored rebound smoke dir>` built `results/lanem/engine_v3_lanemTOtiny` unchanged. The offsets builder then wrote its npz there. `run_engine_live.py --overlay-dir ... --max-games 6 --seeds 2` ran with the mode on: `run_meta.adapter_flags` shows `ENGINE_SEASON_ANCHOR` and `sources.season_anchor` (both families, 6 games, rows realigned by game_id). With the mode off, the same run is refused. `run_engine.py` on the tag without the overlay (served artifacts) and with the offsets set is refused: "non-zero po_first offsets on ... artifacts that are not marked anchored".

Engine tests: `tests/test_engine.py`, `tests/test_smoke.py`, `tests/test_season_anchor.py` and `tests/test_season_anchor_serving.py` all pass: 28 passed and 1 skipped after the first commit, 26 passed after the realignment change.

## 3. Box

Commands: `docs/models/team_rate_estimator/experiments.md` section 7d (retrain both TO arms, build tag `S1TO`, write offsets, sim with the switch passed as `-e` in `BOX_DOCKER_ARGS`, grade). `box_run_v2.sh` does not forward `ENGINE_SEASON_ANCHOR` (lane G's file, not edited), which is why the switch goes through `BOX_DOCKER_ARGS`.

## 4. Caveats

- Zero-offset rows take `predict_proba`. The softmax of raw + 0 equals it only to about 1e-16, so this is what makes (b) exact. On the real fold-2 offsets every game row is non-zero, so every row takes the offset path: the smallest single-class |offset| is 9.5e-5 for PO, and the smallest rebound offset is 0.022.
- The PO wrapper's full-size run is untested locally; only its tiny test-hook run was exercised end to end. Its fits are lane C's functions and its artifact writing is lane J's path.
- Artifacts under `results/lanem/` are smoke and test-hook models, not Stage B results.
