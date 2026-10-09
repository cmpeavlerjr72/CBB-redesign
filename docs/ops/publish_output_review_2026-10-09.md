# Publish output review, 2026-11-02 slate (2026-10-09, ops only)

Reviewed from today's real passes under `results/daily/` (evening `s200_o0` publish `efb6280495`, 118 games, 91 on placeholder tips;
morning `s200_o0_morning` publish `f4d200824a`, 27 games with real tips; 200 seeds each). No lines exist yet (0 spreads, 0 totals, 0 moneylines),
so every line, edge and EV column is empty.

## What is written

Publish stage (`scripts/run_daily_publish_v1.py`): `results/daily/publish/<slate>/<run_id>/<publish_id>/` with `slate.parquet` (full, 86 columns),
`slate.csv` (same), `slate.md` (human table), `lines.parquet`, `publish_meta.json`. Player output is NOT in the publish directory. It is the sim
stage's `results/daily/sim/<slate>/<run_id>/players.parquet`.

## Per game (what a user sees)

`slate.md` table: tip (UTC), game (`away @ home`, `(N)` neutral), sim margin (SD), sim total (SD), p(home), line, provider, ATS / O/U lean with
points of disagreement, ML edge. The parquet / csv also carry per game:
- margin: mean, SD, q05 / q25 / q50 / q75 / q95 (home minus away); total: mean, SD, same quantiles; home / away points means; possessions mean
- `p_home` (raw frequency, ties 0.5), `sim_ot_rate`, `n_seeds` (200)
- per-team shot-type means (3PA, rim 2PA, jump 2PA, FTA and the makes)
- `created_at` (sim build), `published_at`, `tipoff_utc`, `tip_source`, `tip_time_is_placeholder`, `pre_tip_basis` (`real_tip` or
  `placeholder_lower_bound`), `pre_tip_verified` (true only on real tips), `publish_id`, `run_id`
- line columns (provider, spread, total, ML, open, `line_fetched_at`, cover / over probabilities, EV at flat -110), all NaN today.

## Per player (sim `players.parquet`)

One row per player per seed (not summarised): `game_id, seed, athlete_id (ESPN), cbbd_id, team_id, minutes, pts, reb, ast, fga, fg3a, fta,
fgm2_rim, fgm2_jump, fgm3, fouls, tipoff_utc, created_at`. 422,860 rows = 118 games x 200 seeds x about 9 players per team. No names, no team
or opponent labels, no mean / quantile summary: a bettor must aggregate seeds and join names. Anonymous roster slots are DROPPED from the
file, not labelled. Consequence: a team's named minutes do not sum to 200 (mean 191.7; 26% of teams under 190; the thinnest is team 304 in game
401922020 at 82 minutes, 328 at 148). Player points do not sum to team points, and the file carries nothing that tells a reader the shortfall
is anonymous minutes. Anonymous slot share is 23.7% of slots but 4.3% of minutes (in `run_meta.json`, not in any user-facing file).

## Early-season totals label

PM decision 2026-10-08: early-season totals carry a known-bias label, no numeric adjustment. It was ABSENT from publish rows before today.
Added (label only, tested in `tests/test_grade_stage_preflight.py`): boolean column `early_season_totals_flag`, true when the game date is 0-14 days
after the season's first D-I game (2026-11-02), NA when the season start is unknown. A one-line note is added to `slate.md` when any game is
flagged, and `publish_meta.json` records `season_start` and the flagged game count. Verified on a scratch re-publish of the cached morning
sim (27 of 27 flagged; every sim number and `publish_id` identical). Every 11-02 through 11-16 game is flagged.

## Traceability

Before today a publish row carried only `run_id`, `publish_id` and `created_at`: no model / stack version, no inputs hash (those lived only in the
sim `run_meta.json`). Added: `inputs_hash`, `config_hash` and `engine_tag` columns (and the two hashes in `publish_meta.json`), read from the sim
`run_meta.json`; `publish_id` is unchanged. `inputs_hash` is the cache key (rosters, R1 feed, tips, ratings snapshot, served-stack code, ENGINE_*
env, seeds, injury Out set). Caveat: the existing 10-09 morning publication was built before the cache-key fix, so its sim has no `inputs_hash`
(column is empty on re-publish); the evening sim does (`8a464ff8...`). The already-written publications on disk are not rewritten.

## Confusing for a bettor

1. Two publications per slate exist (evening 118 games, morning 27 games). Different `publish_id`, different game counts; grading uses the earliest
   per game. Nothing in `slate.md` says which is current.
2. 91 of 118 evening rows are `pre_tip_verified = False` (placeholder tips); the `.md` table shows a tip time (e.g. 05:00Z) that is the placeholder, not a
   real tip, with no marker. Real tips are only 50 of 173 so far.
3. Almost the whole opening slate is labelled neutral in the morning set (27 of 27 `(N)`); 11% of the evening set. Correct per the feed, but it
   surprises people expecting home court; home advantage is off for those games.
4. Player file: no names, anonymous minutes silently missing, per-seed rows, not a prop sheet (see above).
5. `p_home` is a raw frequency over 200 seeds (steps of 0.005); no line means ATS / O/U / ML columns are blank, not zero.
6. The d0-14 totals label says "known bias" but no direction or size is printed; the size lives in the diagnostics docs, not the slate.
