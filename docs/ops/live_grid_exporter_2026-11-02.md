# Live-odds exporter (game level), 2026-11-02

Builder worker steps B0 + B2 of `docs/ops/live_trajectories_cbb_design_2026-11-02.md`. Spec and bake-off: `docs/models/live_grid/experiments.md`.
Code: `src/cbb_sim/live_grid.py` (tables, lookup = the shipped client rule), `scripts/export_live_grid_v1.py` (live.json), wiring in
`scripts/export_site_day_v1.py` (one block in `build_day`), grader `scripts/grade_live_grid_v1.py`, tests `tests/test_live_grid.py` and three tests appended to
`tests/test_export_site_day_v1.py`.

## What it does
`export_site_day_v1.build_day`: for each game, if `results/trajectories/daily_<day>_<game_run>/trajectory.parquet` exists (a sibling of `results/daily`, the
path `run_daily_sim_v1.py` already writes and the B1 parallel writer must keep), read that game's rows (parquet `game_id` filter), verify the trajectory's
terminal scores equal the run's `games.parquet` `home_pts/away_pts` for every kept seed (a mismatch refuses the game), build
`days/<day>/games/<slug>/live.json` beside `summary.json`, and add `live_path` to that game's entry in `index.json` (additive; the football index has no such key,
the site finds the file by convention, so the key is a convenience). No trajectory file for the game's run: nothing is imported, written or changed
(byte-identical bundle, proven: 18 files, HEAD vs working tree, frozen clock). A live-file failure (gate, mismatch, read error) never blocks the bundle: it is
recorded in `rep["gaps"]["live_failed"]` and printed. `upload()` deletes `days/<day>/**` first, so the live files are written into the stage BEFORE the one
upload, which is already how `build_day` then `upload` are ordered (no separate additive commit).

## Contract (what `live.json` carries; client = monte_site 6d2de56 `atStateBb`)
CFB keys, same meaning: `convention, game_id, slug, teamA, teamB, n_sims, updated, source_tag, orientation{engine_home, schedule_home, flipped, traj_vs_games}`,
`grid{time_bin_s=120, n_time=20, margin_bin=2, margin_min=-28, margin_max=28, n_margin=29, min_n, margin_rule, p_home_win, mean_total, mean_margin, n}`
(each seed once per cell it entered; null below `min_n`), `uncond{p_home_win, mean_total, mean_margin}`, `remaining{time_bin_s=120, n_time=20, margin_edges=[-11,-6,-2,3,7,12],
n_margin=7, min_n, rule, n[t][7], margin_sf[t][7], total_sf[t][..], time_margin_sf[t], time_total_sf[t]}`. Curve = `{k0, v}`, v per-mille P(remaining >= k0+j), 1000 below k0, 0 past the end.

CBB additions (all optional to the CFB parser): `clock{regulation_s:2400, period_s:1200, n_periods:2, ot_period_s:300}` (selects the basketball pricer),
`remaining.widened[t][7]`, `remaining.late{start_s:2160, time_bin_s:30, n_time:8, margin_edges:[-16,-10,-6,-3,-1,2,4,7,11,17], n[j][11], widened, margin_sf[j][11], total_sf[j][11]}`
(late curves keyed by LATE margin bin, total_sf included), `remaining.end{n_tied, tied_margin_sf, tied_total_sf}` (OT remainder of seeds tied at the horn; null when fewer than 3),
`live_arm`, `gates`, `note`. Arms with the total axis (`B_*`, `C_*`, behind `--arm`) also write `remaining.total_edges[t]` (4 integer edges; `total >= ceil(edge)` is exact for integer totals),
`remaining.total_n[t][5]` and total_sf keyed by total-so-far bin. The default `D_row` has NO total axis (bake-off 9.5): `total_edges` is absent and `total_sf[t]` is keyed by
margin bin, which is what the client reads when `total_edges` is absent. Thin cells (n < min_n; min_n = round(200 x seeds / 2500), 200 at 2,500 seeds) are `null` in `*_row` arms (client falls back to
`time_margin_sf` / `time_total_sf`, or to the coarse block for late cells); `*_pool` arms never write null (neighbour pooled, `widened` says how far).
Pinned in `test_client_contract_pinned_basketball_reading`, `test_default_arm_is_the_bakeoff_winner_and_client_compatible` and, when the football checkout is present, against
`cfb-props-sim/.../weeks/week10/games/kansasstate_oklahomastate/live.json`.

## Sizes and time (20 real games x 2,500 seeds, `live_grid_scale20_s2500`; D_row)
Per game 89,656 B mean (max 106,330): `grid` 12 KB, `remaining` 72 KB (total_sf 27 KB, margin_sf 20 KB, late 12 KB, time rows 10 KB). A 118-game day is about 10.6 MB;
`D_pool` is about 124 KB per game, `A_row` 79 KB. Build 0.16 s per game including the parquet read, about 19 s for a 118-game day (single process). The trajectory
file for a 118 x 2,500 run is about 470 MB (design doc); the per-game `game_id` filter relies on contiguous games per row group (B1 must write whole games per row group).

## Mismatches / flags
* The football index carries no `live_path`; added here as an additive CBB key.
* The PM contract lists total-so-far bins; the bake-off dropped them (no M2 gain at 100 or 2,000 seeds). The client reads both shapes. Reversible with `DEFAULT_ARM = "C_row"`.
* Client fixed sizes: none conflict (it reads `margin_edges`, `total_edges`, `late.margin_edges` from the file). One geometry difference found and aligned: the client has no kickoff row and
  keeps the 20 coarse bins to 2400 under the late block, so the grader's lookup was changed to that geometry (amendment 8b).
* A game with fewer than 3 seeds tied at the horn has `end.tied_*` null: a level game at the horn is then not priced (client returns null). A day-level league OT prior would fix it (not built).
* `tests/test_daily_chain_v3.py::test_sim_stage_trajectory_parallel_shards_are_bit_identical` fails here with `missing own ratings data/processed/ratings/own_ratings_2025.parquet` (B1 worker's test, an absent data file; not touched).

## What B1 must deliver for the exporter to run on a real day
1. `results/trajectories/daily_<date>_<run_id>/trajectory.parquet` for EVERY game of the run, with all seeds the run used (2,500), written by the parallel sim: columns `game_id, seed, poss_idx, period, clock_start,
   home_score, away_score` at minimum (all current schema columns are fine), `game_id` = the global ids in `games.parquet`, whole games per row group.
2. The terminal row of each (game, seed) equals `games.parquet` `home_pts/away_pts` (OT included): the exporter refuses a game otherwise.
3. The file in place BEFORE `export_site_day_v1.build_day` runs, under the run id that appears in `results/daily/sim/<day>/<run_id>` (the exporter uses each game's own newest run: `daily_<day>_<run_id>`, `_morning` suffixes included).
4. A real-day regrade: `scripts/grade_live_grid_v1.py --tags daily_<date>_<run> --n-train 2000 --n-test 500` (seed split) before the owner relies on live value, and a timing run at 118 games x 2,500 seeds.

## ESPN capture (B0)
Not yet taken: the 2026-11-02 slate tips from 6:00 PM ET (scoreboard is STATUS_SCHEDULED for all 18 games at 09:45 ET). Script ready:
`python scripts/diag_live_grid_espn_capture_v1.py --date 20261102 --calls 10` (one call a minute, waits for the first live game). Pre-game facts captured
(`results/diag/live_grid_2026-11-02/espn_capture/`): `status = {clock: 0.0, displayClock: "0:00", period: 0, type: {id "1", name STATUS_SCHEDULED, state "pre", completed false}}`;
`competition.format = {"regulation": {"periods": 2}}`; no `situation` key on the competition (also absent on completed games); finals are `STATUS_FINAL` (id 3, state post), period 2, clock 0.0.
Live `status.type`, period/clock during play, halftime representation and `situation` during play: to be recorded from the capture.
