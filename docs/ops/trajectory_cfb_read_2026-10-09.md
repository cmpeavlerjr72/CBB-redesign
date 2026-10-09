# CFB trajectory emission (SIM_TRAJ): read-only summary, 2026-10-09

Source: `cfb-props-sim/docs/plans/live_lines_trajectories.md`, `scripts/run_full_game_sim_fast.py`
(`_TRAJ`, `TRAJ_COLUMNS`, `traj_on`, `_traj_note`), `scripts/build_sweep_dist_v10.py`,
`scripts/exp_traj_gate.py`. Not modified.

**Schema.** One row per snap opportunity (play), long format keyed `(game_id, seed, play_index)`.
Columns: `play_index, period, clock (sec left in quarter), home_score, away_score, possession
(home/away), field_pos (yards to goal), down, distance, home_to, away_to`. Checkpoint is PRE-snap,
taken at the top of the play loop after the timeout block and before any draw. Row count is an exact
function of the play log (plays - timeouts - two-point tries + 1). One TERMINAL row is appended after
OT; it carries the final score and raw end state, so it is the label and must be filtered out of any
state kernel.

**Capture point.** A module-global list `_TRAJ` (None when off) appended to by `_traj_note(state)` in
`simulate_drive`; initialised at `simulate_game` entry; attached as `result["trajectory"]` only when
on. It reads existing GameState attributes, makes no RNG call, adds no key to play/drive dicts.
Switch: env `SIM_TRAJ` (default "off"), read once per game; the sweep runner `--traj` sets it in the
parent and inside each worker. Cloud: `entrypoint.sh --traj`.

**File format and size.** Third parquet next to the games/players pair:
`data/processed/sim/sweep_dist_<tag>_traj.parquet`, resume-safe per week. About 148 rows per
game-seed; about 695 B per game-seed at scale (dictionary on possession, RLE on period/down, delta on
scores); 31,392 game-seeds held 4.75M plays. A chunked-resume duplication defect was found and fixed
(one trajectory per pair, re-simulated pair replaces).

**Gate.** `exp_traj_gate.py`: games/players parquets byte-identical SIM_TRAJ off vs on; terminal row
score equals the games parquet.

**Consumers (what they answer).**
- inv108 (score-state mechanism): sim trajectories vs real pbp, drive-points by score state and
  favourite status; where the full-game margin is lost (rush/pass rate by lead, drive efficiency).
- inv120 (total-points decomposition): where the sim's missing points live (drives, possessions,
  points per drive, red zone) reconciled to real pbp.
- inv141b (blowout allocation): drive points by channel for games with big spreads, sim drives
  segmented on possession change in the trajectory vs real drives.
- exp_traj_gate / qslope_replay / live-line grid export: conditioning corpus (live state to final
  score) for in-game fair lines.
Common frame: same game_ids on both sides, regulation only, drive boundaries from score deltas.
