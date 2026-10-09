# Daily sim: inputs-hash cache key, player output, shot-block `known` alignment (ops worker, 2026-10-09)

Wiring only. No model, parameter, default or probability was chosen or changed. The three PM rulings of 2026-10-09 (PROJECT_STATUS top block) are implemented as written. Model freeze respected; the seal flag is untouched.

## Step 1. Sim cache key = hash of the actual inputs
**Problem.** `run_sim_stage` cached on its config hash only (slate date, season, seeds, tips mode, ...). Every scheduled pass before 2026-11-02 therefore returned the first sim (`cached: true`) whatever had changed underneath.

**Change.** New `src/cbb_sim/live/inputs_key.py`. For a LIVE run (`pass_name` set or not replay) the stage computes `inputs_hash` after the pass filter and before any build. It hashes, by content:
- the games this run will simulate (ids, tip times, tip source, placeholder flag, team ids, neutral),
- the tip-times table rows for those games,
- the season-S roster file and the S-1 roster file (the R1 fallback feed),
- the ratings snapshot `own_ratings_<S>.parquet` of the ratings dir,
- `games_universe.parquet`,
- the served stack: every `src/cbb_sim/**/*.py` file, the sim / build scripts (`run_engine_live`, `build_engine_inputs_live`, `build_shot_block_lut_live_v1`, `build_engine_inputs_day1prior_v1`, `run_daily_sim_v1`, `chain_daily_v2`), `data/overrides/ao_team_table.json`, and the `ENGINE_*` environment,
- seeds and seed offset, and the Out set of the injury feed (step 5).
Columns ending `_at` (pull and build time stamps) are dropped before hashing, because every refresh rewrites them without changing data. A cache hit needs the config hash AND the inputs hash to match. Otherwise the stage prints `cache MISS ... (changed: [...])`, deletes the stale `_DONE.json` / `games.parquet` / `players.parquet`, re-runs in the same run dir, and records `supersedes` (previous hashes, which components changed) in `run_meta.json`. A `_DONE.json` without an inputs key (every sim written before today) counts as a miss. Replay runs (`replay=True`, no pass) keep the old config-only cache and their config hashes are unchanged.

**Tests.** `tests/test_sim_cache_key.py` (8): identical inputs hash equal even when pull stamps differ; a one-row tip change, a roster change, seed count, slate, extra file and `ENGINE_*` env each change the hash; the stage reuses only on an identical key, re-runs and clears the stale `_DONE` on a changed key or a key-less old `_DONE`.

**Confirmation** (4 seeds, separate roots, 118 games on 2026-11-02, evening pass):
| run | evidence |
|---|---|
| chain run 1 (`--seeds 4 --root results/ops_cache_check`) | sim 72.6 s, `cached: false`, publish new id |
| chain run 2, minutes later, nothing changed | sim 0.5 s, publish `cached: true`, same publish id (inputs byte-identical apart from stamps) |
| `run_sim_stage` on a sibling tip table (`tip_times_2027_sibA.parquet`, copy of the served table), separate root | miss, 75.6 s, inputs hash `487f41dd85f4`, equal to the chain run's hash |
| same call again | hit (`cached: True`, no run) |
| sibling B: ONE row changed (game 401925953 tip 23:00Z to 00:00Z next day, +60 min) | `cache MISS ... (changed: ['slate', 'tip_times'])`, re-run 69.3 s, new hash `9186eb657239`; `run_meta.supersedes` names the old hash and the two components |
| sibling B again | hit |
The served `tip_times_2027.parquet` was not modified for the check (`tip_table=` is a new optional argument of `load_slate` / `run_sim_stage`, default the served path). Scripts and harness: `results/ops_cache_check_sib/tipcheck.py` (gitignored).

## Step 2. Player output
`chain_daily_v3.stage_sim` now passes `players=True`. Config hash changes accordingly, so the old 10-09 sims were superseded, not refused.
- 4-seed chain run: 118 games, 8,487 player rows (118 games x about 72 rows), 2,567 distinct player-game means, anonymous slot share 0.2367.
- Served 200-seed on-demand pass (scheduler task, 09:38 ET): 118 games, 200 seeds, **422,860 player rows**, anonymous slot share **0.2367** (23.7%, unchanged from `a3_seed_wiring_2026-10-09.md`), `players.parquet` written next to `games.parquet`. Runtime 720 s.

## Step 3. Shot-block `known` aligned to the live-path harness
**Ruling.** The harness zeroes the shot-block `shooter` and `known` arrays on seeded sides; the served path kept `known = 1`. Align the served path to the harness.

**Change.** `build_shot_block_lut_live_v1.zero_seeded_sides` and `attach(..., seeded_sides=)`. The A3 `seed_fn` now records the (game row, side) pairs it filled (`seed_fn.seeded`; no behavioural change). `run_sim_stage` passes them, so `shooter` and `known` are zero on exactly the seeded sides. Sides with an in-season rotation prior are untouched. Replay and experiments pass nothing, so their LUTs are unchanged.

**Evidence** (`scripts/diag_a3_seed_wiring_v1.py`, 118 games, seeds 0-3, clock 2026-10-09T16:00Z, ratings of 2026-11-02):
| comparison | result |
|---|---|
| player rows daily vs harness-with-LUT-zero | 8,487 = 8,487, 0 rows on one side only |
| slots (named ids) | 0 mismatches |
| max abs diff, per-player minutes mean | **0.0** |
| max abs diff, per-player points mean | **0.0** (row level: minutes 0.0, points 0) |
| games frame, daily vs harness-with-LUT-zero (472 rows, 29 shared columns) | **identical** (`DataFrame.equals` True) |
| daily vs harness with the LUT as built (known = 1) | now differs (max player minutes mean 4.18, points 7.0): this is the old parity, expected to break by the ruling |
| anonymous slot share | 0.2367 daily = harness |
Before the change the daily path matched the unzeroed harness and differed from the zeroed one by the 0.05 pts/game the PM noted.

**Parity v10:** `run_parity_smoke_v1.py --ref docs/ops/parity_reference_windows_v10.json`: PASS, bit-identical (60 games x 5 seeds), run after step 3 and again after step 5. The reference was not touched.

**Tests:** `tests/test_shot_block_seeded_known.py` (4): zero only seeded sides; no-op without seeds; `attach` plumbing; the seed records its sides.

## Suite
After step 3: 787 passed, 1 skipped, 0 failed. After step 5: 798 passed, 1 skipped, 0 failed (full run, 4.5 min).

## Commits
`79af602` (steps 1-2), `d727cbb` (step 3), `9633a57` (steps 4-5 code).
