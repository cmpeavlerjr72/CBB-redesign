# Sim-stage parallel profile, 2026-11-02 (ops worker)

Box: Intel Core Ultra 7 265T, 20 cores / 20 logical (NO hyperthreading: 8 P-cores + 12 E-cores), 65 GB RAM. Other workers' python
jobs (pytest, bots) were running during the timings, so every parallel number is a slight over-estimate of the per-worker cost.
Harness and raw log: `results/diag/sim_parallel_2026-11-02/{bench.py,bench_log.jsonl,micro.py,inp.pkl}` (gitignored).
Sample: first 18 / 36 rows of the 2026-11-02 morning slate (inputs rebuilt with the same call as the pass), 500 seeds.

## Timing (seconds per game-seed per worker = busy time inside RL.simulate / game-seeds; single-core reference 0.031-0.035)

| workers | games x seeds | wall s | s per game-seed per worker | slowdown vs 1 | total game-seeds/s |
|---|---|---|---|---|---|
| 1 | 18 x 500 | 279 | 0.031 | 1.0x | 32 |
| 4 | 36 x 500 | 209 | 0.046 | 1.5x | 87 |
| 6 | 36 x 500 | 243 | 0.079 | 2.5x | 76 (noisy: other jobs on box) |
| 9 | 36 x 500 | 146 | 0.071 | 2.3x | 127 |
| 12 | 36 x 500 | 159 | 0.102 | 3.3x | 118 |
| 18 (as scheduled) | 36 x 500 | 161 | 0.155 | 5.0x | 116 |
| 18, one game per task | 36 x 500 | 191 | 0.184 | 5.9x | 98 |
| 20, one game per task | 36 x 500 | 202 | 0.201 | 6.5x | 100 |
| 10 (new default), 10 games x 2500 | live pass, full stage | 203 | 0.078 | 2.5x | 123 |

Throughput saturates near 8-10 workers at about 120 game-seeds/s (3.8x of one core). Beyond that workers only slow each other.

Other measurements: startup per worker 0.3-0.6 s (unpickle + import; 1.6 s at 20 workers); RSS 575-590 MB per worker; thread env vars
(OMP/MKL/OPENBLAS = 1) confirmed inside every spawned worker; no per-task reload (one task per worker, adapters loaded once); tail
imbalance small (busiest worker 160 s vs idlest 147 s at 18 workers; CPU 84-97% busy until the last 10 s). Single-core rate does not
depend on batch size above ~1,000 simulations per call (0.033 at 2,000 vs 0.031 at 9,000; 0.046 at 500; 0.095 at 125).

## Cause

Memory-bandwidth / shared-cache contention in the vectorised NumPy loop, not hyperthreading (there is none), not BLAS oversubscription
(pins verified in the children), not pickling or startup, not tail imbalance.
Evidence: a pure NumPy micro-benchmark (`micro.py`) run on N spawned processes. Cache-resident arrays (32 KB) slow 1.1x / 1.5x / 2.3x at
4 / 8 / 18 processes (clock and P/E-core mix). Arrays streaming 64 MB slow 1.5x / 2.5x / 5.0x at 4 / 8 / 18, which matches the engine's
1.5x / 2.5-2.7x / 5.0x almost exactly. The engine loop works on large per-batch arrays and is bandwidth-bound; the 35 W T-series part
and 12 E-cores add a little on top. Batch size is not a lever: smaller batches are slower single-core (500 sims 0.046, 125 sims 0.095)
and no faster under load (18 workers: 250-sim batches 0.177, 500-1000 sims 0.155, 2,000 sims 0.144).

## What changed

* `default_jobs()` = `min(10, cpu_count - 2)` (was 18): same throughput (123 vs 116-135 game-seeds/s), 8 fewer processes (about 4.6 GB less RAM)
  and CPU left over for the other jobs on the box. `jobs` stays out of every hash; output unchanged.
* No further speedup is available from process layout. The remaining lever is the engine's memory traffic (float32 state, in-place
  ops, fewer temporaries), which changes the engine and is out of scope for an ops change; it would need its own bit-identity round.
* B1: `--trajectory-seeds N` now works under the parallel path (see below).

## Wall-clock projections (measured throughput 120-135 game-seeds/s)

116 games x 2,500 seeds = 290,000 game-seeds: 2,143 s measured this morning at 18 workers; about 2,300 s at 10 workers (extrapolated).
150-game Saturday: 375,000 game-seeds, about 2,800-3,100 s (47-52 min) plus about 2.5 min of other stages.
Roughly 4x the single core is the ceiling on this box without engine changes.

## B1: parallel trajectory writer

* Each worker opens its own `TrajectoryWriter` at `results/trajectories/daily_<date>_<run_id>/_shards/part_<k>.parquet` (whole games per worker,
  zstd level 3; `TrajectoryWriter` gained `compression_level`, default 9 unchanged). The parent merges shards in chunk order into
  `trajectory.parquet` (zstd 3), checks the row count, sums the (game, seed) pair counts, deletes the shards, fills the existing
  `traj_meta`. Game ids are the global ones (the worker gets the full input plus its row subset). The writer makes no RNG call.
* Test `test_sim_stage_trajectory_parallel_shards_are_bit_identical`: games / players parquets identical with the writer off, on at jobs=1,
  2, 3; trajectory frames identical (sorted on game_id, seed, poss_idx); row and pair counts equal; shards removed.
* Cost, live slate rows, 10 games x 2,500 seeds (all seeds recorded), 10 workers: writer off 214 s total (203 s sim), writer on 203 s
  total (191 s sim): no measurable cost (inside the noise of the other jobs on the box). Merge 2.0 s. Output 3,334,083 rows,
  25,000 game-seeds, 36.8 MB = 3.7 MB per game. Extrapolated to 118 games: 434 MB, merge about 25 s, sim stage +0%.
  Default stays OFF in the chain.

## Switch-on (owner, 2026-11-02) and end-to-end check

* `scripts/run_daily_chain_pass_v1.ps1`: new `[int]$TrajectorySeeds = 2500` parameter; when > 0 the wrapper appends `--trajectory-seeds <N>` to the
  `chain_daily_v3.py` argv. The registered tasks (`-Pass evening|morning -Seeds 2500`) need no change; `-TrajectorySeeds 0` switches it off.
* Dry run (live inputs, 115 games x 200 seeds, 10 workers, run id `diag_e2e`, root `results/diag/sim_parallel_2026-11-02/e2eroot`, nothing under
  `results/daily`): sim 177.6 s, trajectory `results/trajectories/daily_2026-11-02_diag_e2e/trajectory.parquet` (3,137,452 rows, 23,000 game-seeds,
  10 row groups, 115 games, no game split across groups, terminal scores equal `games.parquet` for all 23,000 pairs, 200 seeds per game).
  `export_live_grid_v1.py` wrote 115 `live.json` (7.06 MB, 3.2 s) to `results/diag/sim_parallel_2026-11-02/live_out/`.
* Cost at 2,500 seeds, writer on: no sim-stage cost measured (203 s vs 214 s for 10 games); merge ~2 s per 37 MB. 118 games: sim ~2,300 s, merge ~25 s;
  150-game Saturday ~3,000 s; both far inside the 5 h / 4 h task limits. Disk: ~434 MB per 118-game run (3.7 MB per game), ~0.9 GB per day at two
  passes, ~1.2 GB on a 150-game day; 350 GB free. Nothing deletes old trajectory dirs automatically (design doc: keep 3 days).
