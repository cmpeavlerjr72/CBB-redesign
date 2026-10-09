# A3+R1 day-1 player prior wired into the served daily path (ops worker, 2026-10-09)

Wiring only. No model, parameter or default chosen; the A3 arm (selected 2026-10-05) and the R1 fallback (selected 2026-10-05) are applied through the one definition the chain already had (`chain_daily_v2.day1_player_prior_seed`: `make_seed_fn("A3", roster_2027, fallback="R1", serving=True)`). Seal flag untouched (false); the serving exemption covers the S-1 reads, as before.

## 1. What was wrong
The served chain is `chain_daily_v3.py`. For a 2027 slate its `inputs` stage is `chain_day1_2027_v1.stage_inputs` (census build) and its `sim` stage is `run_daily_sim_v1.run_sim_stage`, which builds its own inputs. Neither passed `seed_fn` to `build_live`. Only `chain_daily_v2.build_live_inputs` did, and v3 never calls it. So every 2027 slot was anonymous.

There was a second gap, found while doing this: `data/processed/player_crosswalk.parquet` has no 2027 rows. That left `roster_espn` all -1, and the engine's player-output filter (`loop.py`: keep `espn > 0`) dropped every player. Even with names, the daily sim would have written no player rows.

## 2. Changes
| file | change |
|---|---|
| `scripts/run_daily_sim_v1.py` | `day1_prior_applies(season, replay)`: true for season >= 2027 and not replay. `day1_prior_seed(season)` hard-stops (RuntimeError naming each missing table) via `V2.day1_player_prior_missing`, then returns `V2.day1_player_prior_seed`. `run_sim_stage(..., day1_prior="auto")` passes `seed_fn` to `build_live` and calls `D1P.post`. `cfg["day1_prior"]="A3+R1"` is added to the config hash only when the prior applies, so replay hashes are unchanged. `run_meta.json` gets a `day1_prior` block (team-games seeded, slots, fallback teams, anon slot share). The stage return carries `anon_slot_share`, `d1p_team_games` and `fallback_roster_line`. CLI flag `--no-day1-prior` (paired checks only). |
| `scripts/chain_day1_2027_v1.py` | `stage_inputs` (census) uses the same rule and seed. Blocked (named tables) when a source is missing. Census reports `day1_prior`, `d1p_team_games`, `anon_slot_share` and the fallback line. |
| `scripts/build_engine_inputs_live.py` | `espn_map_uncrosswalked(cw, season)` is used ONLY when the crosswalk has no rows for the season. The map comes from the season-S roster file `espn_player_id` (wins) plus the latest crosswalk season < S (for R1 players). It only labels which player rows are emitted; no simulated quantity reads `roster_espn` (grep: `loop.py:937` is the only engine read). Seasons in the crosswalk (2024-26) never reach it. |
| `scripts/build_engine_inputs_day1prior_v1.py` | `build(..., serving=False)` passthrough to `make_seed_fn` (default False; experiments unchanged). Used by the parity check below. |
| `scripts/diag_a3_seed_wiring_v1.py` | new: daily vs harness check (below). |
| `tests/test_a3_seed_wiring.py` | new, 7 tests: rule; sim passes seed for 2027 and none for replay / opt-out; hard stop on missing sources; uses the chain's definition; census stage passes the seed; ESPN-map fallback. |

## 3. Acceptance

### (a) Real 4-seed 2026-11-02 sim
`ops_seal_week_v1.py --execute` refuses while `seal_lift_approved` is false (by design; I did not set it). I ran the exact command its `sim_4seed` stage runs under `--execute`: `chain_daily_v3.py --seeds 4 --slate-date 2026-11-02 --no-probe-hoopr`. I added `--root results/a3_wiring_chain_2026-10-09` so the 10-08 anonymous `results/daily/.../s4_o0` was not overwritten. rc 0, 2026-10-09 12:13Z wall clock. All stages ok; sim 72 s.

| | 10-08 (before) | 10-09 (after) |
|---|---|---|
| games / failed / NaN | 118 / 0 / 0 | 118 / 0 / 0 |
| team-games seeded | 0 of 236 | 236 of 236 |
| anon slot share (all 3,540 slots) | 100% | 23.7% |
| team-games with every slot anonymous | 236 | 0 |
| mean total (4-seed game means), SD across games | 139.1, 10.3 | 140.3, 10.3 |
| mean abs margin | 15.3 | 15.8 |
| fallback line | - | 36 teams on fallback roster (R1) |

Paired, same clock and seeds (diag below): the A3+R1 vs anonymous total is +1.21 pts, mean |per-game diff| 3.1. At 4 seeds this is a plumbing read, not a calibration read. The direction matches `early_gap_live_path` (seeded games +0.8 / +1.3).

Per roster source (slate teams; daily arm; anonymous minutes = 1 - named minutes / team minutes, seed mean):
| roster source | teams | anon slot share | anon minutes share | named slots per team |
|---|---|---|---|---|
| ESPN 2027 roster | 200 | 25.6% | 4.8% | 11.2 |
| R1 fallback (no 2027 roster) | 36 | 13.1% | 1.1% | 13.0 |
| none | 0 | - | - | - |
| all | 236 | 23.7% | 4.3% | 11.4 |

Every slate team has either a 2027 ESPN roster or an R1 roster, so no team stays at 100%. Against the 10-05 sizing doc's analog of about 7-11% anonymous MINUTES on rostered teams, this is lower (4.8%). Anonymous slots sit in the tail (slot index below), where the rotation share is small. R1 teams name more slots because R1 has no roster cut except final-year players.

Per team: anon slot share quantiles min 0, p10 0, p25 0.13, median 0.20, p75 0.33, p90 0.47, max 0.80. The most anonymous teams are all ESPN-roster teams with few returners or transfers by S-1 minutes:

| team | anon slot share | anon minutes share |
|---|---|---|
| Idaho State | 0.80 | 0.57 |
| Siena | 0.67 | 0.27 |
| Marist | 0.67 | 0.33 |
| Central Michigan | 0.67 | 0.37 |
| Eastern Washington | 0.60 | 0.24 |
| Utah State | 0.60 | 0.30 |
| Rider | 0.60 | 0.25 |
| Pittsburgh | 0.60 | 0.26 |

Full table: `results/a3_seed_wiring/2026-11-02/per_team.csv` (gitignored).

Per slot type (3,540 slots): returner 2,012; transfer-in 221; R1 fallback 469; anonymous 838 (767 on ESPN-roster teams, 71 on R1 teams).

Per slot index, anonymous share: slots 0-2: 0%; slots 3-4: 0.4%; slot 5: 1.7%; slot 6: 3.8%; slot 7: 7.2%; slot 8: 13.6%; slot 9: 21%; slot 10: 31%; slot 11: 47%; slot 12: 63%; slot 13: 77%; slot 14: 89%.

ESPN id coverage of named slots: 100% (2,702 / 2,702), by slot type returner / transfer / R1: 100% each. The map uses 5,651 rows from the 2026 crosswalk season and 4,072 from `roster_2027.parquet`.

### (b) Daily path vs live-path harness, same seeds and game set
Script: `scripts/diag_a3_seed_wiring_v1.py`, clock 2026-10-09T16:00Z, seeds 0-3, the 118 evening-pass games.
- The harness arm is the harness's own build call (`build_engine_inputs_day1prior_v1.build(..., "A3", anon=False, fallback="R1", strict_finish=False)`) with `serving=True` and the 2027 ratings dir.
- Both arms then go through the same adapter, shot-block attach and simulate.

| comparison | games | player rows | per-player mean rows | max abs diff minutes mean | max abs diff points mean | mismatched slots | games frame |
|---|---|---|---|---|---|---|---|
| daily vs harness | 118 | 8,487 vs 8,487 (0 unmatched) | 2,567 (0 one-sided) | 0.0 | 0.0 | 0 of 3,540 | identical |
| daily vs harness with the harness's LUT convention | 118 | 8,487 vs 8,487 | 2,567 | 4.18 | 7.0 (mean 0.047) | 0 | total mean 140.269 vs 140.261; mean per-game abs total diff 0.05 |

The daily path seeds players exactly as the harness builder does: identical slots, inputs and outputs, bit for bit.

**Flag for the PM (not changed):** `build_engine_inputs_d1p_live_early_v1.py` also zeroes the shot-block LUT `shooter`/`known` on seeded sides. That is the bake-off convention (the A3 bake-off held window shooter state anonymous). The served live LUT builder (`build_shot_block_lut_live_v1.build_table`) sets `known = 1` for any named slot; `shooter` is 0 on opening day either way, since there are no 2027 events.
- Served behaviour (known = 1 for named players) is the training semantics of the shot-block model, so I left it.
- Under the harness convention the slate total moves 0.008 pts and per-game totals 0.05 pts on average; per-player differences are RNG-path divergence at 4 seeds.
- The A3 selection evidence was therefore produced with known = 0 on seeded slots. Serving uses known = 1. The effect measured here is negligible at game level.

### (c) Tests and parity
- Full suite: 766 passed, 1 skipped, 0 failed (was 759; +7 new).
- Parity smoke vs `parity_reference_windows_v10.json`: PASS, bit-identical (60 games x 5 seeds).

### (d) Seal
- `tests/test_day1_2027.py` + `tests/test_serving_seal_exemption.py`: 20 passed. The latter includes `test_training_style_asof_ratings_and_day1_tables_still_raise` and `test_generic_loader_and_trainer_style_reads_stay_sealed`.
- The seal flag is still false. The new wiring reads S-1 only through `tables(serving=True)` → `assert_not_sealed_serving`.

## 4. Other gaps seen (not fixed here; PM)
1. **Sim cache ignores the clock.** `run_sim_stage`'s config hash has no build date. Every evening pass before 11-02 targets slate 11-02 with the same config, so after the first run each pass returns the cached sim (`cached: true`): no re-sim with re-pulled rosters or ratings. The 11-01 evening pass would serve the first run's numbers. A fix (date in run_id or hash) is a chain design change.
2. **The chain writes no player output.** `chain_daily_v3.stage_sim` calls `run_sim_stage` without `players=True`, so no `players.parquet` and no props are produced daily. The wiring above makes them correct when turned on.
3. **Injuries never reach the sim.** `run_sim_stage` does not pass `availability` (injuries) to `build_live`; only the unused v2 `build_live_inputs` does.
4. **Tips.** The tips table now has 50 real tips of 173 on 11-02. Of the 118 mapped games, 91 are still placeholders.
