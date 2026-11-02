# Live conditioning from simulated trajectories: CFB mechanism and CBB design, 2026-11-02

Owner statement (2026-11-02): "the CFB scoreboard does read [the simulated trajectories] in order to show live odds and projected
player stats as the games progress. I want that same thing to happen for CBB games."

Status: read-only study plus design. No code written. Sources read, all unmodified: `cfb-props-sim` (HEAD 7f1cf22),
`monte_site/cfb-sim-explorer` (not a git repo; paths relative to it), and this repo. Line numbers as of 2026-11-02.
Abbreviations: CPS = `C:\Users\devuser\cfb-props-sim`, SITE = `C:\Users\devuser\monte_site\cfb-sim-explorer`.

---------------------------------------------------------------------------------------------------------------------------------
## PART 1. The CFB mechanism, end to end

One-paragraph version. The weekly sweep (10,000 seeds per game, cloud box) runs with `--traj`, which records every snap of every
simulated game to `data/processed/sim/sweep_dist_<tag>_traj.parquet` (plus a per-play log for players).
`scripts/export_live_grid.py` reduces that corpus, per game, to two small static JSON files next to the game's other site files:
`live.json` (clock x score-margin conditional win probability / total / margin, survival curves of the points STILL TO COME per
2-minute bin x 7 margin bins, and a ball-position block) and `live_players.json` (each player's mean REMAINING production per 2-minute
bin x 7 margin bins). The browser fetches each file once per page life; on every ESPN poll it re-prices: current score plus the sim's
remaining-points distribution at the current (elapsed, margin[, ball]) gives live win/cover/over probabilities and a projected final;
current box line plus the sim's mean remaining production gives each player's projected final. There is NO per-seed matching, kernel,
or server-side conditioning endpoint in production. The conditioning is done at export time by binning; the client is a table lookup
with linear interpolation between adjacent bin midpoints. The plan's kernel endpoint (6.2/6.3) and GBT surrogate (6.5) are unbuilt
(`CPS/docs/plans/live_lines_trajectories.md:490-533`).

### 1a. PRODUCER

P1. Engine emits checkpoints (SIM_TRAJ). `CPS/scripts/run_full_game_sim_fast.py:600` `traj_on()`, `:608` `_traj_note(state)`; hook calls
at `:3696` (top of the snap loop) and `:4326` (terminal checkpoint); `_TRAJ` reset at `:4031`, set None at `:4328`. One row per pre-snap
state plus one TERMINAL row carrying the final score (plan `:148-182`). Env `SIM_TRAJ`, default off. Gate `scripts/exp_traj_gate.py`:
games/players parquets byte-identical on vs off (plan `:372-395`).

P2. Runner writes the corpus. `CPS/scripts/build_sweep_dist_v10.py`: `--traj` (`:314`), `run_one(..., traj, plays)` (`:231`, sets
`SIM_TRAJ` inside the worker at `:242`), idempotent resume-aware append keyed (game, seed) (`:141` `_pair_key`; plan `:234-268`).
Outputs: `sweep_dist_<tag>_traj.parquet` (`game_id, seed, play_index, period, clock, home_score, away_score, possession, field_pos, down,
distance, home_to, away_to`) and `sweep_dist_<tag>_plays.parquet` (per-play player log: `seed, quarter, secs, type, player, target, yards,
flags`; read at `export_live_grid.py:371-372`). Measured 695 B per game-seed; 10k seeds = 7 MB per game (plan `:417-446`).

P3. The exporter is `CPS/scripts/export_live_grid.py` (1155 lines). It is NOT `export_site_week.py`, which it only imports for slug and
team resolution (`:66-70`). Per game, inside `build()` (`:936-1072`):
1. Orientation. Engine home = the `is_home==1` row of the pregame feature table (`engine_home_team :163`); flip if it differs from the
   schedule home; second leg `verify_traj_vs_games :188` compares traj terminal scores to the games parquet on at least 100 discriminating
   seeds; any ambiguity aborts.
2. `build_grid :242`. For each seed and each (2-min time bin 0..29, margin bin) cell the seed ever entered (a seed counts at most once per
   cell): mean over the seeds' FINALS of home-win indicator, total, margin. Cells with n < `MIN_N=100` (`:89`) publish n and null stats.
   Margin axis = 29 bins of 2 points, half-away-from-zero (`margin_index :104`; the half-to-even phantom-cell bug is documented at
   `:108-128`). Regulation only; OT rows excluded; the terminal row is the label only.
3. `build_remaining :457`. State = score at the MIDPOINT of each time bin (`mid_states :430`: last checkpoint at or before it). For each
   (time bin, 7 margin bins from `P_MARGIN_EDGES=[-14,-7,0,1,8,15]` `:361`) and an all-seeds time row, survival curves of remaining margin
   and remaining total: `{k0, v}`, `v[j]` = per-mille P(remaining >= k0+j). Cells below `P_MIN_N=200` (`:363`) are null; the client then
   uses the time row. Gate: kickoff row reprices the pregame median total within 0.03.
4. `build_remaining_fp :607` (INV-162). The same curves per (time bin, margin bin, possession A/B, 5 field zones `FP_ZONE_EDGES
   [6,21,51,81]` `:540`), every snap in the bin, one weight per seed per cell, `FP_MIN_N=150` (`:542`), thin zones pooled with
   neighbours, packed with a 64-char alphabet (`fp_encode :552`, `_wcurve :575`). A game that fails its gates ships WITHOUT this block
   (`:1010-1020`) instead of blocking the week.
5. `build_players :751` (only when the `_plays` parquet exists). Per-play log -> events (`play_events :382`), reconciled against the box
   parquet per player (hard gate). Each seed is classified by its margin at each bin midpoint; `rem[t][m]` = mean production on plays
   snapped at or after that midpoint (OT included); cells under 200 seeds are null (client uses `rem_time`). Gates: count stats
   non-increasing in time; bin-0 remaining within 90-100% of pregame. Published only for players with >=1 pass attempt, >=0.5 rush
   attempts or >=0.5 targets per game.
6. `sanity_gate :907`: kickoff cell within 0.02 of the unconditional win probability and 1.0 pt of mean total; late-game win probability
   ordered +12 > 0 > -12.
7. Writes `live.json` and `live_players.json` to `CPS/data/processed/site_stage/cfb-sims-2026/2026/weeks/week<NN>/games/<slug>/`
   (refuses unless that week's `index.json` exists, `:970-975`).
8. `upload :1075`: ONE additive commit of those files to HF dataset `mvpeav/cfb-sims-2026` (no delete patterns), then an unauthenticated
   GET byte-compare of every file.

Who runs it and how often: once per weekly sweep from the cloud box entrypoint step 6b (`CPS/scripts/cloud/entrypoint.sh:530-538`,
failure-isolated), and again by the post-run watcher after every site-week export (`CPS/scripts/cloud/postrun_watch.py:21,176-187`),
because the site-week upload replaces `games/**` wholesale and deletes any `live.json` not staged on that machine. There is NO refresh
during a game day: the files are the pregame ensemble, by design.

File sizes (week10 sample `akron_ohio`; 62 games that week): `live.json` 443,970 B (`grid` 22 KB, `remaining` 129 KB, `remaining_fp`
336 KB), `live_players.json` 137,947 B (26 players x 4 stats), `n_sims` 10,000, `source_tag cloud10k_wk10a`. About 0.58 MB per game; the
ball-position block is 76% of `live.json`.

### 1b. LIVE SOURCE

* Server `SITE/server/liveScores.ts`. ESPN scoreboard per sport (basketball URL `.../basketball/mens-college-basketball/scoreboard?dates=
  &groups=&limit=` at `:364`), cached with `ttlFor(liveCount)`: 20 s with live games, 120 s otherwise (`:518-522`). SSE
  `GET /api/live?sport=&date=` (`:849-918`) sends `hello`, then a `tick` every 20 s (`:914`), heartbeat comment every 25 s (`:916`).
  `type Sport = "cfb" | "cbb" | "mlb"` (`:23`). The route is already sport-generic; CBB works today.
* Client mapper `SITE/src/pages/ScoreboardV2.tsx:186-257` `mapEspnToLiveGames(payload, sport)`: per event `state`, `period`
  (`comp.status.period`), `clockSec` (`comp.status.clock`, seconds left IN the period), scores, ids, names, and
  `situation: football && state === "in" ? parseSituation(comp) : undefined` (`:236`). Basketball never gets a situation.
* `SITE/src/lib/espnGame.ts:77-97` `parseSituation`: `yardLine` (absolute, home goal line = 0), `down`, `distance`, `possessionId`,
  `isRedZone`, `homeTimeouts`, `awayTimeouts`, `lastPlay`, `attackDir`, ESPN's own `homeWinPct`. Pricing uses only `yardLine`, `down`
  (1-4) and `possessionId`; timeouts and distance are display-only.
* So the pricing state is: score A/B, period, seconds left in period, and (football) ball side + yards to goal, one new state per 20 s tick.

### 1c. CONSUMER

* Fetch: `SITE/src/lib/liveGrid.ts:297-323` `fetchLiveGrid` GETs `<periodDir(ns)>/<weekId>/games/<slug>/live.json` through the dataset
  proxy (`dataUrl`), `cache: "no-store"`, every failure is a quiet null. `useLiveGrids :431-454` keeps a module-level `gridCache`
  (`:427`): one fetch per game per page life. `periodDir("cbb-2027")` is already `"days"` (`SITE/src/lib/cfbData.ts:61-63`).
  `useLivePlayers` (`SITE/src/lib/livePlayers.ts:53-81`) does the same for `live_players.json`.
* Which cards: `ScoreboardV2.tsx:3190-3196` builds `liveGridReqs` for every filtered card with `liveInProgress && jsonRow && weekId`. The
  drawer's own `useLiveGrid`/`useLivePlayers` (`:5170-5185`) fire only while the "live" panel is open.
* State to sim state: `SITE/src/lib/liveRecord.ts:33-39` `simStateOf`: `elapsed = (period-1)*900 + (900-clockSec)` clamped 0..3600;
  `period >= 5` is OT, priced as `elapsed=3599, ot=true`. The football constants 900/3600/5 are hard-coded. `ballFromSituation :45-56`
  turns `yardLine/down/possessionId` into `{side, ytg}` or undefined (kickoffs, PATs, breaks).
* Conditioning rule (`liveGrid.ts:204-243` `atState`): margin bin via `marginBin(rem.margin_edges, A-B)` (`:194`); with a known ball use
  the field-zone cell, else the possession-only cell, else the clock+score cell, else the all-seeds time row (`:215-225`); evaluate at the
  two time-bin midpoints around `elapsed` and blend linearly (`:226-242`); after the last midpoint fade the remaining-points distribution
  to "nothing more is scored" by 3600 s unless tied or OT (`:230-238`). All in the sim's A/B orientation (A = sim home); `espnHomeIsA`
  (name match, `ScoreboardV2.tsx:5200-5205`) and `flipToEspnHome` handle neutral-site swaps.
* Outputs: `pCoverA/pCoverB/pOver/pWinA` (`liveGrid.ts:250-287`): final = current score + remaining; push counts half.
  `expectedRemaining :276` feeds `projectedFinal` (`liveRecord.ts:22-29`) = current score + mean remaining per side, used for the slate's
  "Projected" ATS/Totals/ML record (`ScoreboardV2.tsx:4731-4790`). Card live lines (`:4193-4208`, passed as `live={liveLines}` at `:4408`)
  feed the "Live value vs Kalshi" mode (`:3612-3615`). The older chart path `lookupGrid :341` (grid block) draws the sim line in
  LiveGamecast (`LiveGamecast.tsx:886-928`), widening the margin bin +/-1 then +/-2 when a cell is null.
* Match-count thresholds live in the EXPORTER (100 / 150 / 200 seeds), not the client. The client has no n_eff test; a null cell falls back
  as above, and if everything is null the card shows nothing. There is no nearest-neighbour or per-seed weighting anywhere in production.
* Players: `LiveGamecast.tsx:701-760` `PlayerBox`. Each ESPN box player is joined to a sim player by `playerKey(name)` (lowercase letters
  only, suffixes dropped, `livePlayers.ts:124-131`); `remainingFor` (`livePlayers.ts:98-120`) picks the margin-bin row (or `rem_time`),
  interpolates between bin midpoints, blends from `pregame` at kickoff, fades to zero past the last midpoint; projected final = ESPN
  actual so far + that remaining. Sim regulars with no box line yet show `pregame`. In OT only actuals are shown. The join of player
  totals to "seeds whose trajectory passes through the state" happens at EXPORT time (each seed classified by its margin at the bin
  midpoint), not at request time.

### 1d. DATA CONTRACT (exact, from the `akron_ohio` week10 files)

`live.json`:
```
convention: str   "teamA = home, teamB = away, margin = home minus away; spread is in HOME perspective (negative = home favored)"
game_id: int   slug: str   teamA: str   teamB: str   n_sims: int   updated: ISO-8601 str   source_tag: str
orientation: {engine_home: str, schedule_home: str, flipped: bool, traj_vs_games: "direct"}
grid: {time_bin_s:int, n_time:int, margin_bin:int, margin_min:int, margin_max:int, n_margin:int, min_n:int, margin_rule:str,
       p_home_win:[n_time][n_margin] float(4dp)|null, mean_total:[..] float(2dp)|null, mean_margin:[..] float(2dp)|null, n:[..] int}
uncond: {p_home_win: float(4dp), mean_total: float(2dp), mean_margin: float(2dp)}
remaining: {time_bin_s, n_time, margin_edges:[int]*6, n_margin:7, min_n:200, rule:str, n:[n_time][7] int,
            margin_sf:[n_time][7] Curve|null, total_sf:[n_time][7] Curve|null,
            time_margin_sf:[n_time] Curve, time_total_sf:[n_time] Curve}
  Curve = {k0:int, v:[int]}   v[j] = per-mille P(remaining >= k0+j); 1000 below k0, 0 past the end
remaining_fp (optional): {time_bin_s,n_time,margin_edges,n_margin,zone_edges,n_zone,min_n,alph,rule,widened_cells,slope_gate,
            poss_n, poss_margin_sf, poss_total_sf, n, margin_sf, total_sf}   curves packed as [k0:int, code:str]
```
`parseLiveGrid` (`liveGrid.ts:94-130`) returns null (whole file ignored) unless `grid` (four matrices plus seven numeric fields) and
`uncond` (three numbers) are present; `remaining` and `remaining_fp` are optional and parsed leniently.

`live_players.json`:
```
convention, game_id:int, slug, teamA, teamB, n_sims:int, updated, source_tag, time_bin_s:int, n_time:int, margin_edges:[int]*6,
n_margin:7, min_n:200, rule:str, n:[n_time][7] int,
players:[{player:str, team:str, side:"A"|"B", stats:[str], pregame:[float]*len(stats),
          rem_time:[n_time][len(stats)] float, rem:[n_time][7] (null | [len(stats)] float)}]
```
Rounding: counts 2 dp, yardage 1 dp. `parseLivePlayers` (`livePlayers.ts:38-51`) needs `players`, `margin_edges`, numeric `time_bin_s`/`n_time`.

Other consumers of the same files: `CPS/scripts/live_shadow_log.py` (logs our conditional line vs ESPN state and Kalshi live mids every N
seconds using the exporter's own `grid_lookup :866`; ESPN fetched with `curl.exe` because python TLS is blocked on this Windows box) and
`CPS/scripts/xbot/live_watch.py`.

---------------------------------------------------------------------------------------------------------------------------------
## PART 2. CBB design

### 2.0 Where CBB stands today (verified)

* A trajectory exists: `src/cbb_sim/engine/trajectory.py` (`TrajectoryRecorder`, `TrajectoryWriter`), one row per possession,
  1,572-1,592 B per game-seed (136.5 rows). The 2026-11-02 slate dry run: 2,360 game-seeds, 3.71 MB (20 seeds).
* `scripts/run_daily_sim_v1.py:305-313`: a writer forces the single in-process `RL.simulate`; the 18-way `simulate_parallel` (`:167-188`)
  runs only when `tw is None`. So `--trajectory-seeds 2500` today means a serial sim of 118 x 2,500 game-seeds. Unacceptable for the chain.
* Per-seed player totals exist (`players.parquet`) but the trajectory has no per-possession player attribution beyond `shooter_slot/
  shooter_id`, so CFB-style "remaining production" cannot be built from it. This is the main structural difference from CFB, whose
  play log carries every pass, rush and target.
* Site: `/api/live?sport=cbb` and the ESPN MBB scoreboard work; `mapEspnToLiveGames` already gates football-only fields (`:236,:243`). The
  CBB sport config sets `panels: { liveGamecast: false, teamStats: false }` (`ScoreboardV2.tsx:1444`), so there is no live drawer and no
  player box for CBB. `docs/ops/cbb_site_contract.md` says today: "no live.json / live_players.json".
* A measured fact that shapes the design. Read-only check on `results/trajectories/replay394_s50/trajectory.parquet` (19,700 game-seeds),
  state at the end of period 1, demeaned within game: corr(total so far, remaining total) = +0.136, OLS slope +0.158 points per point
  (SD of first-half total within game 10.2, SD of remaining total 11.8); corr(margin so far, remaining margin) = -0.147, slope -0.143. OT
  occurs in 3.7% of game-seeds. The CFB design assumes remaining points depend on (clock, margin) only. In this engine the per-game pace
  latent (one pace realisation per game) makes the scoreboard TOTAL informative about the rest of the game: a first half 20 points above
  the sim's expectation shifts the remaining total by about +3. Ignoring it would bias every live total.

### 2.1 Live state for basketball

ESPN MBB scoreboard gives `status.period` (1, 2 = halves; 3+ = overtimes), `status.clock` (seconds left in the period; OT periods are
5:00) and scores. `situation` is probably absent (needs a live capture to confirm).

| field | definition | source |
|---|---|---|
| elapsed | regulation: `(period-1)*1200 + (1200 - clockSec)`, 0..2400; status HALFTIME = 1200 whatever period/clock ESPN reports (verify on a capture) | ESPN |
| scoreDiff | A - B in sim orientation (A = sim home; flip by name match as the site already does) | ESPN |
| scoreTotal | A + B | ESPN |
| ot | period >= 3 | ESPN |
| possession | NOT used in v1 | see below |
| timeouts / fouls / bonus | NOT used in v1 | not on the scoreboard feed |

Possession is excluded in v1: it flips every ~17 s, is worth about one point of margin (small against a 5-point bin), and the MBB
scoreboard does not reliably carry it. If a live capture shows a usable `situation.possession`, add a possession-split block later behind
its own gate. Do not build the field-zone block at all (no basketball analogue).

### 2.2 Conditioning rule (decision 1): slice the ensemble by binning, on three axes, with exact boundaries

Keep the CFB principle: the live line is a slice of the SAME graded ensemble; no live model, no hand adjustment (so the standing "no hand
tuning on engine output" rule is respected and the line inherits the pregame grade). Binning, not the plan's kernel, because the client
is a lookup and no kernel endpoint exists. Changes for basketball:

1. Time. 20 two-minute bins over 2,400 s (about 7 possessions per bin). Midpoint states as in CFB. Add a LATE block: 30-second bins for the
   last 4 minutes (8 bins, elapsed 2160-2400), because intentional fouling and the bonus change the remaining-points distribution inside
   two minutes in a way 2-minute linear interpolation cannot represent. The late block uses finer margin bins (about 11, 1-3 points wide near
   zero; edges fixed from the pilot cell-n table in step B0). Thin late cells fall back to the coarse block, not to the time row.
2. Margin. 7 coarse bins with edges scaled to basketball margin SD (about 11 vs CFB 17): `margin_edges = [-11,-6,-2,3,7,12]` (bins <=-12 |
   -11..-7 | -6..-3 | -2..2 | 3..6 | 7..11 | >=12 under `searchsorted(side="right")`), so the client's `marginBin` is unchanged. The 29-bin
   `grid` block stays for contract compatibility (`parseLiveGrid` requires it): 2-point bins over +/-28, same half-away-from-zero rule.
3. TOTAL axis (new, from the measured pace dependence). `remaining.total_sf` is keyed by (time bin, total-so-far bin) instead of (time bin,
   margin bin). Total-so-far bins are 5 quantile bins of the sim's own total-so-far distribution at that time bin, edges stored per time bin
   (`total_edges[t]`), so a bin means "this game is running hot or cold relative to the sim's expectation". Margin curves stay keyed by margin
   bin. Remaining margin is nearly independent of the total and vice versa, so two 1-D conditionings keep cell n high at 2,500 seeds where a
   3-D grid would be empty. Over/under and projected total use the total-keyed curves; cover, moneyline and projected margin use the
   margin-keyed curves. If the B0 pilot shows the dependence is not significant after binning, drop the axis and revert to the CFB shape.
4. End of game exact, not faded. Add a row at elapsed = 2400 whose curves are the true remainder (tied: OT points; not tied: zero). This
   replaces CFB's client-side fade to zero over the last 60 s (`liveGrid.ts:230-238`), a patch over a data gap that is the kind of thing the
   no-patch rule is written against.
5. Overtime. v1 prices OT as the tied end-of-regulation state, as CFB does (`liveRecord.ts:37`); acceptable because OT is 3.7% of games and the
   tied late cells carry OT points. v2 adds an OT block (5-minute periods, 1-minute bins) because every tied-late live game passes through it.
   Trajectory rows already carry `period >= 3`.
6. Thin cells. Pool neighbouring margin (or total) bins until n reaches the threshold; fall back to the all-seeds time row only as a last
   resort (CFB's zone pooling, `export_live_grid.py:640-690`, shows pooling is better). Record `widened` per cell so the client can mark a
   thin price.
7. Checkpoint convention. CBB rows are per possession. Checkpoint = possession OPEN state: score before (previous row's after-score; 0-0 at
   `poss_idx` 0), `clock_start`, `period`, `off_side`. Label = after-score of the last row per (game_id, seed), OT included. No terminal row,
   so nothing to filter; conditioning rows are `period <= 2`. Each seed counts once per cell, as in CFB.

Pre-registration (bake-off rule). Choosing between (A) CFB-copy coarse bins, (B) coarse + late block + total axis, (C) Gaussian kernel on
(elapsed, margin, total) is a comparison. Spec goes to `docs/models/live_conditioning/experiments.md` BEFORE any exporter code: fit each arm
on a random half of the 2,500 seeds of a pilot slate and score on the other half (then swap); metrics = held-out log-loss of home win and CRPS
of remaining total and margin at 12 checkpoint times including the last 4 minutes; segments = margin bucket and favourite strength; noise
floor = a different half-split. Ties go to A.

### 2.3 How many trajectory seeds (decision 2): all 2,500

Use `--trajectory-seeds 2500`, not a 20-seed subset. From the CFB choice:
* CFB used 10,000 seeds with exporter thresholds 100 (grid), 150 (ball cells), 200 (remaining curves, players), and even so a 13-point
  favourite's seven margin-bin counts at one time bin were `[732, 2153, 3455, 1365, 1771, 460, 64]` (week10 `live_players.json.n[10]`): the
  underdog-leads cell was already under threshold at 10k seeds. Early CFB coverage was 75.7% of cells until the rounding fix (plan header).
* At 2,500 seeds counts are 4x lower. A central basketball coarse cell (|margin| <= 2) holds about 20-25% of seeds = 500-600; the +/-12 tails
  about 10-15% = 250-375; the opposite tail of a heavy favourite drops to tens. With a threshold of 200 the coarse grid is populated for
  balanced games and masked in the tails of mismatches, which is why pooling replaces the time-row fallback. A 1-point late block is NOT
  populated at 2,500 seeds (P(margin = k) near zero is about 4%, ~100 seeds), hence 1-3 point late bins.
* The thesis (a live line sliced from the graded pregame ensemble inherits its grade) needs the SAME seeds as the pregame files. A 20-seed
  subset is 125x thinner than the CFB corpus: SE of about 11 points on a win probability. SE at n = 200 is 3.5 points; at n = 500, 2.2.
* Reserve option, not default: if the B0 pilot shows more than 15% of live-relevant cells masked, run a trajectory-only extension pass
  (`seed_offset` 2,500..4,999, same inputs and engine) and pool. Same distribution, not the same seeds, so `source_tag` must say so.

### 2.4 File sizes at 118 games per day

| artefact | per game | 118 games | notes |
|---|---|---|---|
| trajectory parquet, 2,500 seeds | 4.0 MB | **470 MB per run** (295,000 game-seeds x 1,592 B, measured) | two runs a day = 0.94 GB/day; keep 3 days, delete after export succeeds; under gitignored `results/` |
| `live.json` | about 180 KB unpacked: `grid` 14 KB, coarse `remaining` (7 margin curves + 5 total curves per time bin x 20 bins x 2 quantities, about 60-80 values each) about 110 KB, late block about 50 KB | about 21 MB; about 9 MB with CFB-style packed curves | CFB's 444 KB is 76% ball-zone block, which CBB lacks |
| `live_players.json` | about 100 KB: about 18 named players x 8 stats x (20 + 20 x 7) numbers (CFB sample: 26 players x 4 stats = 138 KB) | about 12 MB | |
| added to the day bundle | about 0.3 MB | about 33 MB (about 20 MB packed) | the day upload already replaces the folder |

### 2.5 Producer changes (CBB repo)

P-A. Trajectory under the parallel sim (`scripts/run_daily_sim_v1.py`). Remove the `tw is None` condition at `:309`. Each worker
(`_sim_rows_worker`) opens its own `TrajectoryWriter` at `results/trajectories/daily_<date>_<run_id>/_shards/part_<k>.parquet` (whole games
per worker, so no (game, seed) is split), applies the existing `seed_limit`, and returns the shard path with its games/players frames. The
parent checks shard row counts and pairs, concatenates row groups in chunk order into `trajectory.parquet` at zstd level 3 (level 9 is too slow
on the critical path), deletes the shards, and fills the existing `traj_meta`. Games/players concatenation is untouched. The worker's sliced
`inp` must give the recorder the GLOBAL game ids; test explicitly. The trajectory stays out of the config/inputs hash.
Per-game run id: `export_site_day_v1.build_day` takes each game from the newest run containing it (`:508-511`), so the live exporter must read
`daily_<date>_<game_run_id>/trajectory.parquet`. A game whose run has no trajectory ships without live files and the client degrades.

P-B. Player snapshots (needed for projected player stats; engine instrumentation, Opus). Add a fourth recorder hook that, at possession OPEN,
holds the box counters of the active simulations' 2 x 13 slots and, when the possession crosses a bin midpoint, commits that pre-possession
state for that bin (at most one crossing per possession: bins are 120 s, a possession up to ~35 s). Reduction happens in-worker at chunk end,
per game: `n[t][margin bin]` and `sum_remaining[t][margin bin][side][slot][stat]`, remaining = final box minus snapshot box. Output is a small
per-game `.npz` shard (about 120 KB), NOT a per-seed table (per-seed snapshots would be about 6 KB per game-seed, 1.8 GB per run). Stats:
`min, pts, reb, ast, fgm, fga, fg3m, fg3a, fta, fouls` (the engine has no steals or blocks). Slots map to ESPN athlete ids via
`inp.roster_espn[gi, side, slot]`; anonymous slots are dropped. Hard gates as in CFB: final box per player equals `players.parquet` totals for
the kept seeds exactly; count stats non-increasing in time; bin-0 remaining / pregame within [0.9, 1.0]. Also record each slot's own foul count
at the snapshot so the foul-conditioned arm can be fit; foul trouble is basketball state with no CFB analogue.
Player-conditioning bake-off (pre-registered in the same experiments.md): P1 margin-bin rows as CFB; P2 margin bin plus the player's own foul
count (0-1, 2, 3, 4+) at the bin; P3 time only. Metric: held-out-seed MAE and calibration of each player's remaining minutes and points,
segmented by minutes tier.

P-C. Exporter `scripts/export_live_day_v1.py` (new file; the existing exporter changes by one call site). Imports `make_slug`, `name_key` from
`export_site_day_v1.py` so slugs match. Per game: read that game's trajectory rows (row-group filter on `game_id`); verify last-row scores equal
`games.parquet` `home_pts/away_pts` for every kept seed (CFB's second leg; CBB home/away are explicit in the slate so no feature-table leg);
build `grid`, `remaining`, `uncond`, `orientation`; run the gates; write `live.json` and `live_players.json` INTO the staged day folder
`stage/2027/days/<day>/games/<slug>/`. It must run before `export_site_day_v1.upload`, which deletes `days/<day>/**`
(`export_site_day_v1.py:743-760`; the same hazard that made CFB's post-run watcher re-run its live exporter). Wire into
`chain_daily_v3.site_export` (`:203-220`), failure-isolated (a live-file failure never blocks the day bundle, like CFB's `remaining_fp` drop).
Add optional `live_path`, `live_players_path` to each day-index game entry (additive; absent = no live layer).

P-D. Contract edits (additive; `docs/ops/cbb_site_contract.md` gets a section). CFB keys verbatim plus these optional CBB keys, so the existing
parser still reads the file: `clock: {regulation_s: 2400, period_s: 1200, n_periods: 2, ot_period_s: 300}` (so the client stops hard-coding
900/3600/5); `remaining.total_edges` and total-keyed `remaining.total_sf` (the CFB-shaped margin-keyed `total_sf` may be kept beside it as
`total_sf_margin` for old clients); `remaining.late` block (own `time_bin_s`, `margin_edges`, curves); per-player `athlete_id` (int, ESPN id).

### 2.6 Server and client changes for sport=cbb (SITE)

Football-specific items needing a basketball branch, by priority:
1. `liveRecord.ts:37` `simStateOf`: constants 900, 3600, `period >= 5`. Take a geometry argument read from `live.json.clock` (fallback football
   constants) so CBB uses 1200 / 2400 / `period >= 3`; handle halftime status explicitly.
2. `liveGrid.ts`: `LiveGrid` and `parseLiveGrid` gain the optional `clock`, `remaining.late`, `remaining.total_edges`. `atState` (`:204-243`)
   must (a) use the late block when `elapsed >= regulation_s - 240`, (b) key total curves by total-so-far bin, (c) skip the fade-to-zero when
   the file has the exact end row, (d) never enter the ball branch (already guarded by `rem.fp` presence). `SimState.ot` means period >= 3.
3. `ScoreboardV2.tsx`: `liveGridReqs` (`:3190`) and the projected-record builder (`:4759`) call `simStateOf` with football-only
   `ballFromSituation`; for cbb pass no ball and the geometry. For cbb `weekId` is the day id; confirm `c.jsonRow.slug` is populated for CBB
   cards (the bundle's `games[].slug`) and that the dataset proxy serves `days/<day>/games/<slug>/live.json` and `live_players.json` (one
   request to verify). Add a 30-minute expiry to `gridCache` for cbb so a tab left open across the morning re-publish refreshes.
4. `parseSituation` / `ballFromSituation`: unchanged; never called for cbb (`:236`).
5. `LiveGamecast.tsx` hard-codes 3600, quarter separators `[2700,1800,900]` (`:339`) and ESPN's football summary/win-probability/drives. Not
   reused for CBB v1 (`panels.liveGamecast=false`). Player projections need a NEW basketball live-box component, not an enable flag.
6. New `LiveBoxCbb` component: ESPN MBB summary boxscore (minutes, pts, reb, ast, FG, 3PT, FT, PF per player), joined to `live_players.json` by
   ESPN athlete id first and `playerKey(name)` second; `projected = actual + remainingFor(...)` (`remainingFor` reusable as is; add the foul-state
   selection only if arm P2 wins). Needs a new fetch against the basketball ESPN summary path (`espnGame.ts:25-28` is hard-coded to football).
7. Server: no change to `/api/live`. The 20 s tick is adequate (about one possession per tick). If a capture shows an MBB `situation.possession`,
   extend `parseSituation` with a basketball branch (possession, timeouts, bonus) behind the sport flag; not in v1.

### 2.7 Refresh cadence

Producer: every pass of the daily chain (evening run for the next slate, morning re-run), as the last publish step, in the same upload as the
day bundle. The files are the pregame ensemble and are static during a game, exactly like CFB. A mid-game injury or lineup surprise is NOT
reflected; the card should say "pregame ensemble" (CFB does not). Games that tip before the morning pass keep the evening run's files (`build_day`
takes each game from the newest run containing it). Client: fetch once per game per page life (30-minute expiry for cbb), re-price on every
`/api/live` tick (20 s); the Kalshi board refresh stays 45 s.

### 2.8 Test plan

1. Bit-identity with the writer ON (the inertness contract).
   (a) `scripts/diag_trajectory_onoff_v1.py` stays green.
   (b) New: parallel sim (jobs 18) with `--trajectory-seeds 2500` vs parallel with 0: `games.parquet` and `players.parquet` `DataFrame.equals`
   after sorting, max abs diff 0.0.
   (c) parallel+writer vs single-process+writer: trajectory frames equal after sorting by `(game_id, seed, poss_idx)`.
   (d) Score replay: last row `home_score/away_score` equals `games.parquet` for every kept (game, seed).
   (e) P-B snapshot hook ON vs OFF: same game/player frames; sim-stage wall-clock delta at most +10% (measure and report).
   (f) Cached-run guard still reports `NOT WRITTEN`.
2. Synthetic 3-game live-file test (pytest, no engine). Fabricate trajectories with known generating processes: G1 balanced random walk
   (analytic P(win | t, m) from the normal approximation); G2 neutral site with sim-home != schedule-home, to exercise orientation; G3 heavy
   favourite, to force masked tail cells and pooling. Assert: the file validates against the CFB key tree (the test loads a real CFB `live.json`,
   compares key tree and dtypes, and allows only the documented CBB additions); kickoff cell equals uncond within 0.02; late monotonicity
   (+12 > 0 > -12); grid vs analytic within 3 SE where n >= 200; masked-cell count as expected; packed/unpacked curve round trip; the
   per-player reconciliation gate FAILS when one synthetic player's total is perturbed by 0.1; the exporter refuses an ambiguous orientation;
   live files are staged before the folder-replacing upload (ordering test with a fake HF client).
3. Client unit test on a basketball state (SITE, the repo's TS test runner, tiny fixture grid): `simStateOf(40, 35, period 2, clock 300, cbb
   geometry)` gives elapsed 2100, ot false; period 3 gives ot true priced at end of regulation; halftime gives 1200. `pCoverA/pOver/pWinA` equal
   hand-computed values from the fixture curves including push = half; linear interpolation between adjacent midpoints; late block selected at
   elapsed >= 2160; at elapsed 2400 with margin > 0, `pWinA` is exactly 1 (no fade); total-axis lookup uses the total-so-far bin; a thin cell
   falls back to the pooled neighbour; the football fixtures still pass (regression); `parseLiveGrid` accepts the CBB file and rejects one
   missing `grid`.
4. Self-consistency backtest before any display (held-out seeds, 2.2), then real games: the shadow logger (B8) logs our conditional line vs ESPN
   state and Kalshi live mids for every live game for at least two weeks BEFORE the owner relies on "live value". The CLV cross-check (does our
   number predict the market's next move) is the leak detector, per the profitability frame. Until then the live layer is unvalidated.
5. End-to-end smoke on a real day: stage, upload to a scratch dataset path, run the site's `fetchLiveGrid` against it.

### 2.9 Build order, estimates, repos (decision 3: ship game-level first, players second)

Hours are focused worker hours (code, tests, one review pass), not wall clock. CBB = CBB-clean-sheet; SITE = monte_site/cfb-sim-explorer.
Assignments follow the PM/worker split.

| step | what | repo | est. | worker |
|---|---|---|---|---|
| B0 | Pre-register `docs/models/live_conditioning/experiments.md`; pilot on the first available 2,500-seed trajectory (cell-n table, total-axis significance gate, late-block bin edges, masked-cell share); live ESPN MBB capture (period/clock/halftime/situation) | CBB | 3 | Sonnet |
| B1 | Parallel trajectory writer: per-worker shards, parent concat, meta; tests 1b-1d, 1f | CBB | 5 | Sonnet |
| B2 | `export_live_day_v1.py` game level: grid, remaining (coarse, late, total axis, exact end row), uncond, orientation, gates, packing, synthetic 3-game test, wiring into `site_export` before upload, contract doc | CBB | 9 | Opus |
| B3 | Client geometry + CBB branch in `liveGrid.ts` / `liveRecord.ts` / `ScoreboardV2.tsx`; unit tests (2.8 item 3); cache expiry; proxy allow-list check | SITE | 6 | Sonnet |
| B4 | Smoke on a real day; chain flag defaults, retention, runtime and disk report. Ships live odds, live value and projected record for CBB | CBB + SITE | 3 | Sonnet |
| B5 | Player snapshot hook + in-worker accumulators + `.npz` shards + on/off bit-identity and runtime gate | CBB (engine) | 9 | Opus |
| B6 | `live_players.json` builder + reconciliation gates + player bake-off P1/P2/P3 | CBB | 7 | Sonnet |
| B7 | Basketball live-box component, ESPN MBB summary fetch, athlete-id join, tests | SITE | 10 | Sonnet |
| B8 | CBB live shadow logger (ESPN via `curl.exe`, Kalshi KXNCAAMB live mids, daily parquet, scorecard) | CBB | 6 | Sonnet |
| B9 | OT block (v2) | CBB + SITE | 5 | Opus |

Critical path to live odds on the CBB scoreboard: B0-B4, about 26 h. Projected player stats: B5-B7, about 26 h more. B8 runs alongside B5-B7 and
gates trust. B9 optional. Total about 63 h. Suggested chaining (one brief per worker, per the token rule): B0+B1; B2; B3+B4; B5; B6; B7; B8.

### 2.10 What NOT to copy from the CFB design

1. The ball-position block (`remaining_fp`): 76% of the CFB file, no basketball meaning. Do not invent a possession/zone analogue without a verified feed.
2. The fade-to-zero in the last 60 s (`liveGrid.ts:230-238`) and the `ZERO` curve: a client-side patch over a data gap. Use an exact end-of-regulation row.
3. Margin-only conditioning. The CFB claim that remaining points do not depend on how the points so far were scored was checked here and is false for this engine
   (slope +0.158 per point of first-half total). Condition on total so far.
4. Uniform 2-minute bins to the end of the game and fallback to the all-seeds time row for thin cells. The last minutes of a basketball game are driven by fouling and bonus
   state; use the late block and neighbour pooling.
5. Football margin edges (`[-14,-7,0,1,8,15]`) and the separate tied bin. A basketball game cannot end tied (OT); a tie matters only through time remaining.
6. Tie-as-half in `fin_win`: dead code in CBB because finals include OT. Push-as-half at a betting line stays.
7. Name-only player join (`playerKey`): basketball has same-surname players and transfers; join by ESPN athlete id first.
8. Additive separate upload of live files: CBB's day publish deletes `days/<day>/**` before upload, so the live files must be staged BEFORE the one upload, never added in a
   separate commit that the next publish would delete.
9. Seed-count assumptions behind the thresholds: keep the absolute n thresholds (they are statistical) but expect many more masked cells at 2,500 seeds and pool.
10. The stat list (pass/rush/rec) is replaced; the engine has no steals or blocks, so the live box must show no projection for stats the sim does not produce.
11. Silent absence: the CFB card shows nothing when the grid is missing or null. For CBB, show "no sim conditional for this game" with the reason (run lacked trajectory,
    orientation check failed, thin cell) so a missing live line is not read as no edge.

### 2.11 Risks and open items for the PM

* The sim-time cost of the recorder at 2,500 seeds under 18 workers is unmeasured (the 394 x 50 run took 209 s on 10 workers with the recorder on; its recorder-off time was
  not recorded). Gate: sim-stage wall-clock increase at most 10% against the same slate without the recorder (B1).
* Disk: 0.94 GB per day of trajectory; retention is a B4 decision.
* ESPN MBB feed details (halftime period value, `situation` presence, OT clock) are unverified until the B0 live capture.
* Player conditioning needs engine work (B5) that CFB did not, because the CBB trajectory has no per-possession player events. Without B5 only game-level live odds ship.
* Model freeze is on. B5 is instrumentation only (no sampling-path change) and must pass the bit-identity gate (2.8 item 1e) before merge.
* All of this is design. None of it is validated against real in-game outcomes; treat the live layer as unvalidated until B8 has at least two weeks of logged live games.
