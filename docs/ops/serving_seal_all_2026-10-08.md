# Serving-path seal exemption for all daily stages + real 4-seed sim (ops worker, 2026-10-08)

`seal_lift_approved` stays false; `data/overrides/ratings_day1_choices.json` untouched. No model or default change.
Scope of every exemption: serving season 2027 live, reading season 2026 as the PRIOR season. Explicit argument at the serving call site only (`serving=True` / `assert_not_sealed_serving`); no env var, no global; the `[seal] EXEMPTION USED` line is kept.

## 1. Static enumeration of gates reachable from chain_daily_v3 live stages
| file:line (before change) | stage | kind | class | action |
|---|---|---|---|---|
| chain_day1_2027_v1.py:40 `check_choices` (`seal_lift_approved` required) | ratings | flag check | serving-only | not required when season == 2027; still required for season 2026 |
| chain_day1_2027_v1.py:67 `with V2.unsealed()` | ratings | CBB_UNSEAL env | serving-only | removed |
| build_own_ratings_asof_v1.py:109 `assert_not_sealed` | ratings (manifest_uniform) | assert | SHARED (diag parity, rehearsal) | `serving` arg (default False) -> `assert_not_sealed_serving`; chain passes `serving=(season==2027)` |
| build_own_ratings_asof_C_v1.py:46 `assert_not_sealed` | ratings (arm_C) | assert | SHARED | same `serving` arg |
| chain_day1_2027_v1.py:102 `with V2.unsealed()` | inputs | CBB_UNSEAL env | serving-only | removed (build_live already uses the exemption) |
| build_engine_inputs_live.py:180 `assert_not_sealed_serving` | inputs, sim | assert | serving-only | already exempt (prior change) |
| chain_daily_v3.py:120 `V2.unsealed()` | sim | CBB_UNSEAL env | serving-only | removed |
| chain_daily_v2.py:313 `unsealed()` in `build_live_inputs` | inputs (v2 live branch, not used by v3 day-1) | CBB_UNSEAL env | serving-only | removed |
| build_engine_inputs_day1prior_v1.py:94 `assert_not_sealed([S-1])` in `tables()` | inputs (A3 seed_fn, v2 branch) | assert | SHARED (experiments, `cmd_serve`) | `serving` arg (default False) through `make_seed_fn`; `chain_daily_v2.day1_player_prior_seed` passes True |
| run_daily_sim_v1.py:137, run_engine_live.py:177 (season == 2026 + CBB_UNSEAL) | sim | env check | serving-only | not triggered for 2027 (target season 2026 only); unchanged |
| chain_daily_v2.py:67/:183 `seal_lift_approved` + `unsealed()` in v2 `stage_ratings` | ratings, pre-2027 v2 branch | flag + env | legacy, unreachable for 2027 from v3 (v3 takes the day-1 branch) | unchanged |
| build_ao_team_prior_v2.py:72, build_shot_block_prior_v1.py:57, ops_seal_week_v1.py (`seal_lift_approved`, `unseal`) | seal-week one-off builders | assert / flag / env | one-off | unchanged, stay gated; already ran |
| src/cbb_sim/eval/reference.py:58,116; models/*.py fold_slices; all train_/exp_/run_*closed_loop scripts | training, eval, experiments | assert | training | untouched, still raise |

Count: 9 serving-path gates exempted or already exempt (5 serving-only code sites removed/relaxed, 1 already exempt, 3 shared via explicit arg: asof v1, asof C, day1prior tables); 3 gates left alone because not reachable from the 2027 v3 chain or not triggered (run_daily_sim, run_engine_live, v2 stage_ratings); 3 one-off builders kept gated. Everything in training/eval stays gated.

## 2. Tests
`tests/test_serving_seal_exemption.py` now 6 tests: added the daily day-1 ratings stage for 2027 running while sealed (flag false, no CBB_UNSEAL, exemption line printed, env untouched) and a training-style `asof_ratings(2027, ...)` and `day1prior.tables(2027)` call still raising. `tests/test_day1_2027.py` updated: 2027 does not need the flag; a 2026 target still does. Full suite: 759 passed, 1 skipped, 0 failed (the two prior stale failures were already fixed upstream). Parity smoke vs parity_reference_windows_v10.json: PASS, bit-identical.

## 3. Real 4-seed sim, slate 2026-11-02 (`chain_daily_v3.py --seeds 4 --slate-date 2026-11-02 --no-probe-hoopr`, rc 0, sim 67 s)
Output `results/daily/sim/2026-11-02/s4_o0/` (gitignored); chain log `data/processed/ingest/chain_v3/2026-10-08.json`. Stages ratings, inputs, sim, publish all ok.
| item | value |
|---|---|
| games simulated / failed | 118 / 0 (472 game-seed rows) |
| mean total (game-mean of 4 seeds) | 139.1; SD across games 10.3; mean within-game SD 13.3 |
| mean abs margin | 15.3 (of 4-seed mean margin), 17.2 (per sim) |
| NaN / inf | 0 / 0 |
| highest totals | Florida State v Florida A&M 164.8; McNeese v UAB 160.8; Cornell v Pepperdine 158.8 |
| lowest totals | Minnesota v North Dakota 116.5; Temple v Central Connecticut 117.0; Washington State v Davidson 118.3 |
| non-anonymous player priors | 0% of slots. All 236 team-games are rotation fallback, 0 candidates per team-game |
| 38 unmapped non-D1 games | skipped, not errored: listed under `unmapped_non_d1` in `skipped.json`, counted as `slate_games_unmapped_non_d1` in the census |
| tips | 91 of 118 are placeholder midnight-ET tips |

Caveat: 4 seeds and 0% player priors make these totals a plumbing check, not a calibration read. Total SD across games (10.3) is plausible; the 4-seed within-game SD is noisy.

## 4. PM decision needed
The sim stage (`run_daily_sim_v1.py:167`) and the day-1 `inputs` stage call `build_live` with no `seed_fn`, so the A3 day-1 player prior (selected 2026-10-05, used only in `chain_daily_v2.build_live_inputs`) is never applied in the daily sim: every player slot is anonymous. Wiring `seed_fn` into `run_daily_sim_v1` would need `serving=True` tables (now supported) and a paired-seed check before it changes output. Not done here.
