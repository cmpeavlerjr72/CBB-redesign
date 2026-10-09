# CBB trajectory emission: design (written before coding), 2026-10-09

Goal: write a per-possession trajectory per simulated (game, seed) so a game can be judged on how it reached
its final score. Mirrors the CFB `SIM_TRAJ` side-channel (see `trajectory_cfb_read_2026-10-09.md`). Model freeze
is on: instrumentation only, no sampling-path change.

## Capture point
`src/cbb_sim/engine/loop.py::simulate_chunk` advances every active simulation by exactly one possession per outer
step, so the natural row is one possession. Two read-only hooks in the step loop, both guarded by `if rec is not None`:
1. OPEN, right after `act`/`off` are computed (before the clock draw): snapshot period, seconds_remaining,
   home/away points, team fouls (defence, offence), bonus flags and the box counters of the active rows.
2. CLOSE, right after the possession bookkeeping (`poss_count`, `seconds_remaining -= used`) and BEFORE block (f)
   advances the period: diff the box counters (FGA/FGM per class, FTA/FTM, TOV, OREB), read `used`, `end_code`,
   `chance`, and the post-possession `pts`.
A third one-line hook in the chance loop copies the existing `shooter` array into a per-step buffer
(`last_shooter` = slot of the possession's final chance). No hook calls `book.draw`, an adapter, or mutates state.
The recorder only copies and subtracts arrays the engine already computed. Cost when off: one `is None` test per
step (not per row). Because capture is per outer step and not per game, no restructure of the vectorised loop is needed.

## Switch
- engine kwarg `simulate_chunk(..., trajectory_writer=W)` (W is a `TrajectoryWriter` file sink), and/or
- env `CBB_TRAJECTORY=1` (+ optional `CBB_TRAJECTORY_SEEDS=N`, record only seeds < N) which builds an in-memory
  recorder and attaches the frame as `ChunkResult.trajectory`.
- `run_engine.py --trajectory-seeds N` writes `results/trajectories/<run_tag>/trajectory.parquet` (parent process
  collects worker frames). `chain_daily_v3.py --trajectory-seeds N` passes it to the daily sim (default 0 = off).
Default OFF everywhere; ChunkResult gains an optional field defaulting to None, game/player frames untouched.

## Schema (one row per possession per game-seed)
| column | dtype | meaning |
|---|---|---|
| game_id, seed | int64, int32 | keys |
| poss_idx | int16 | 0-based possession ordinal in this game-seed (outer step; an OREB chain is ONE row) |
| period | int8 | 1, 2, 3+ = OT (as GameState) |
| clock_start, clock_end | int16 | seconds remaining WITHIN the period at open / after the possession |
| game_clock_sec_remaining | int16 | at open; regulation: seconds left in regulation; OT: seconds left in the OT period (GameState `usage_sec_remaining`) |
| off_side | int8 | 0 home has the ball, 1 away |
| off_team_id | int64 | offence team id |
| duration_s | int16 | `used`, the seconds the possession consumed |
| end_type | int8/str | engine `prev_end` code: DREB, TOV, made_FG, made_FT, other (OREB chains end in one of these) |
| outcome | category | derived at flush from the counts: TOV, FG3_MAKE, FG2_MAKE, FT_TRIP, MISS (incl. dead ball/DREB), AND1 variants kept via counts |
| n_chances, n_oreb | int8 | chance-chain length |
| fga2, fgm2, fga3, fgm3, fta, ftm, tov | int8 | counts inside this possession |
| points | int8 | offence points scored this possession (= 2*fgm2+3*fgm3+ftm) |
| home_score, away_score, margin | int16 | AFTER the possession; margin = home - away |
| shooter_slot, shooter_id | int8, int64 | last chance's actor slot and ESPN athlete id (-1 if unknown) |
| off_team_fouls, def_team_fouls | int8 | at open (carry from halftime reset; NCAA OT carries) |
| off_in_bonus, off_in_dbl_bonus | int8 | at open |
| bonus_prior_fouls, dbl_bonus_prior_fouls | int8 | rule-era flags from GameState |

Not captured (the engine has no cheap per-possession array for it): individual player foul counts, on-floor five
(available in `st.on_floor` but 10 ids per row; left out to hold size, can be added behind a second flag),
sub-possession timing of each chance.

## Size
Estimate: ~135 possessions per game-seed (regulation) x ~34 columns of narrow ints, zstd parquet with dictionary/RLE
encoding. Measured numbers are appended to this doc after the implementation (section "Measured size"). If the
projection for 118 games x 200 seeds is above ~1.5 GB the default for the chain flag stays a seed subset
(`--trajectory-seeds 20`).

## Proofs required (step 3)
(a) parity v10 bit-identical flag off; (b) 4-seed replay on vs off: max abs diff 0.0 in games and players;
(c) per-trajectory points replay reproduces the final score for every game-seed; (d) unit tests of the writer.

## Implemented (2026-10-09): what shipped vs the design
- `src/cbb_sim/engine/trajectory.py` (recorder, `TrajectoryWriter`), three guarded hooks in `loop.simulate_chunk`
  (open / shooter note / close), `ChunkResult.trajectory` (None unless on). No restructure of the vectorised loop was needed.
- `CBB_TRAJECTORY=1` (+ `CBB_TRAJECTORY_SEEDS=N`) or `simulate_chunk(..., trajectory_writer=TrajectoryWriter(path, seed_limit=N))`.
- `scripts/run_engine.py --trajectory-seeds N` and `scripts/run_engine_window_v2.py --trajectory-seeds N` (a game-id list runner;
  v1 untouched) write `results/trajectories/<tag>/trajectory.parquet`. `run_engine`'s `_run_block` still returns its 5-tuple when off.
- Schema as above; `outcome` is derived from the counts at flush (TOV, FT_TRIP, FG3_MAKE, FG2_MAKE, MISS in that priority order;
  an and-one is FT_TRIP with fgm>0, so use the count columns when the distinction matters).

## Measured size (394 games x 50 seeds, `replay394_s50`)
2,689,821 possession rows for 19,700 game-seeds (136.5 rows per game-seed), 31.4 MB zstd parquet = **1,592 bytes per
game-seed**. Projection for 118 games x 200 seeds (23,600 game-seeds): **37.6 MB** (3.2 MB for a 20-seed subset of 118 games).
The full 200 seeds is cheap, so the seed subset is a convenience, not a necessity; the chain flag default stays 0 (off).
Write cost: the 394 x 50 run took 209 s on 10 workers with the recorder on.

## How to switch it on
- Daily chain (live or replay): `chain_daily_v3.py ... --trajectory-seeds 20` writes
  `results/trajectories/daily_<slate_date>_<run_id>/trajectory.parquet` for seeds 0-19 and a `trajectory` block in the sim
  `run_meta.json` and in the sim stage result. It is NOT part of the sim config / inputs hash, so it never changes a cache key.
  If the sim is already cached for that slate and run id, the stage reports `NOT WRITTEN: cached run`; rerun with `--force`
  (run_daily_sim_v1) or after removing `_DONE.json`.
- Single run: `run_engine.py --fold F2 --season 2025 --seeds 200 --trajectory-seeds 200`.
- Dry check done: replay chain 2025-02-11, 8 seeds, trajectory off vs `--trajectory-seeds 4`: sim `games.parquet` identical
  (296 rows, max abs diff 0.0), trajectory file 20,389 rows / 37 games / seeds 0-3 / 283 kB.
- Consumers: `scripts/diag_trajectory_onoff_v1.py` (on/off + replay proofs), `scripts/diag_trajectory_vs_pbp_v1.py` (sim vs pbp).
- Live-path dry run (`chain_daily_v3.py --dry-run --dry-run-sim --trajectory-seeds 20 --seeds 20`, 2026-11-02 slate, 118 games):
  sim stage OK, `results/trajectories/daily_2026-11-02_s20_o0/trajectory.parquet` = 320,716 rows, 2,360 game-seeds, 3.71 MB
  (1,572 bytes per game-seed), so a 200-seed live slate is about 37 MB.
