# Possession-version trace: does any v1-labelled column reach a served model? (2026-10-27)

Read-only trace (ruling 24). Nothing in the engine inputs, models or builders was changed. Worker: Sonnet, 2026-10-27.
Reminder from `docs/tests/possessions_build_v2_2026-09-10.md`: v1 and v2 differ ONLY in event labels (v2 adds the rim-location override, 2.27 ft, which turns some `FGA_jump2` rows into `FGA_rim`, plus per-chance attempt counts). Possession and chance boundaries, durations and lineups are identical.

## Answer

- **The served possession_outcome model (`round2_s1`) reads only v2-derived data.** Its team block is `event_team_block_v3_F2_2025.npz` (adapters.py:132-148, 309), built by `LF.po_team_block_r2`, which hard-codes `possessions_dir("v2")`, `chances_*.parquet`, `style_source="first_chance"` (live/features.py:176-203). The plan is resolved against the block's own `idx["team_cols"]`, not `inp.team_names` (adapters.py:316-318). Training is the same: `scripts/build_engine_event_round2.py` / `round2/design.parquet` on v2.
- **A v1-derived set of columns does exist in the inputs, but no served model reads it.** `team_static[:, :, 0:8]` (`off_` and `opp_def_` x `3pa, rim, tov, ftr`) is built from v1 possessions with `style_source="all_chances"`. Backtest: `scripts/build_engine_inputs.py:540-548` reads the round-1 `data/processed/models/possession_outcome/design.parquet`, written 2026-09-10 11:52, before `possessions_v2` existed (13:21). Live and replay: `scripts/build_engine_inputs_live.py:232-234`, `LF.po_team_block` with `poss_dir=PO.DEFAULT_POSS_DIR`, i.e. v1 (`live/features.py:139-152`). No served consumer reads these eight columns:
  - a grep of `src/` for `(3pa|rim|tov|ftr)_c` finds them only in `models/possession_outcome.py` (design builder), `team_rate_adapter.py` (optional, unserved table path), `live/features.py` and `pbp/`;
  - the feature lists stored in the served rebound S1 artifacts (`rebound/s1_confirm/S1_weekly/F2/seg_*.joblib`, 16 features) and fg_make round-4 B1 artifacts (`fg_make/round4/B1/*.joblib`, 16 features) contain none of the eight (checked by loading them); `FT_FEATURES` (free_throw.py:418-426) and the clock and shot_block name lists (clock_adapter_v3.py:119-123, shot_block.py:67-69) contain none;
  - the only adapter that resolves a possession_outcome plan against `inp.team_names` is `reference` mode (adapters.py:262-272, `reference_not_adopted_*.pkl`), which is not served.
  These eight columns feed nothing served. If they were read they would differ from the v2 block: `all_chances` vs `first_chance` source (documented as mechanically biased, possession_outcome.py:116-128) and the 2025 rim/jump2 relabel (v2 moves 6.52 percent of 2025 jump2 rows, 1.7-2.2 percent in 2022-24).
- **`DEFAULT_POSSESSION_VERSION = "v1"` (pbp/possessions.py:313) binds only callers that name no version.** The served-path callers that matter name one: `po_team_block_r2` (v2), `FG.build_design(version="v2")` (build_engine_inputs.py:243, 591), and `events_v2*.parquet` carry attrs `possessions_version=v2, rim_override_max_ft=2.27`. The unversioned or v1 callers are in the table below.

## Per served engine input array

| Array / column group | Producer (file:line) | Possession version | Reaches a served model? | v1 vs v2 differ for it? |
|---|---|---|---|---|
| `event_team_block` (16 cols) | `po_team_block_r2`, features.py:176-203; build_engine_inputs_live.py:241-249; replay build_engine_inputs_v3_replay_evlayer_v1.py:55 | v2 chances, first_chance | yes, `round2_s1` first and cont (adapters.py:309) | n/a (is v2) |
| `team_static` 0-7 (PO style x8) | build_engine_inputs.py:540-548 (round-1 design); live features.py:139-165 | v1, all_chances | no served reader | yes (source and labels), but unread |
| `team_static` rating cols | own ratings (`ratings_for_slate`, features.py) | none (box-score ratings) | yes (rebound, fg_make, clock, shot_block) | not possession-derived |
| `team_static` `site_*`, `season_idx`, `days_since_start` | calendar | none | yes | no |
| `team_static` tempo (`off_/def_tempo_rel`, `tempo_prior_game`) | live: ratings join (features.py:221-246); backtest: clock round-1 `design.parquet` (build_engine_inputs.py:550-573, written 09-10 12:56 on v1 possessions) | v1 for backtest, ratings for live | yes (clock K2) | no: built from possession counts and durations, identical across versions (follows from the v2 doc's boundary statement; not re-derived) |
| `team_static` rebound (`off_oreb_c`, `opp_def_dreb_c`) | `RB.team_rebound_form` over `rebound/events_v1.parquet` (build_engine_inputs_live.py:53, 250-252) | events attrs `possessions_version=v1, rim_override_max_ft=0.0` | yes | no: rebound.py:80-90 states no rebound outcome, opportunity or team rate depends on the version; only `miss_type` labels do |
| `team_static` fg_make (`off_make_c__*`, `def_allow_c__*`) and slot shooter columns | `FG.build_design(version="v2")`, `events_v2*.parquet` (build_engine_inputs.py:243, 589-593) | v2 (override 2.27 ft) | yes | n/a |
| slot FT columns | `free_throw/attempts_v1_era.parquet` (build_engine_inputs.py:754) | free-throw attempts; no possession label column | yes | no |
| rotation priors / roster arrays | `ROT.load_team_possessions` -> `data/processed/possessions/possessions_{season}.parquet` (rotation.py:104, features.py:341) | v1 possessions table | yes (rotation S1) | no: minutes are summed `duration_s` and lineups, boundary quantities identical across versions |
| `rules.silent_foul_per_possession` | build_engine_inputs.py:916-920, distinct `(game_id, poss_index)` of the round-1 design | v1 | yes | no: a possession count |

## Where v1 labels do reach a served model (training, not an input array)

Found while tracing. The served rebound model (`ENGINE_REBOUND=s1_weekly`, `C_plus_state`) is trained on `rebound.build_design(..., version="v1", events=events_v1)` (`scripts/train_rebound_v2_s1.py:214-217`; events attrs `rim_override_max_ft=0.0`). Its `miss_type` feature (rim vs jump2) is therefore v1-labelled in training, while at serve time the sim supplies the miss type from its own shot class (loop.py:285, 622, 677), the v2 vocabulary of fg_make and the event model. Size of the mismatch: for the F2 backtest (trained on 2022-2024) the override would move 1.7-2.2 percent of jump2 rows to rim; the 2025 mistag (6.52 percent) enters only when a refit includes 2025, i.e. the 2027 live refits. Not quantified (no v2 rebound events exist on disk to compare), so this is a candidate for a labelled-version sensitivity check, not a finding of bias.

## Not done

- Own-ratings `tempo_rel` internals not re-derived; taken from `live/features.py` as coming from the ratings table.
- No numeric v1-vs-v2 comparison for the eight unread `team_static` columns (nothing reads them).
- The `trdraw` builder (`build_engine_inputs_trdraw_v1.py:70-71`) and `team_rate_adapter` can overwrite those eight columns from a team-rate table; that path is not served and was not traced further.
