# Live-slate inference path: design note (Lane G, 2026-09-30)

Scope: take (slate date D, as-of cutoff timestamp T, schedule source) and emit the engine's existing
input schema (`EngineInputs`: games, team_static, slot_static, roster/rotation arrays, usage_rate,
reb_rate, names/rules/meta) for games that have NOT been played, then run the engine on them and write
prediction rows that carry `created_at` and `tipoff` with `created_at < tipoff` enforced in code.
No existing file is edited (`build_engine_inputs.py`, `run_engine.py`, `src/cbb_sim/*` stay as they are).
The results section at the bottom is appended once, at the end of the lane.

## 1. Backtest-only assumptions, and the live replacement

| # | Where | Backtest-only assumption | Live replacement |
|---|---|---|---|
| 1 | builder step 1 | Universe = `REF.load_actual_games(season)`: completed D-I games with verified finals | Slate = schedule rows for date D (CBBD `/games` or `games_universe` with scores dropped for parity). Only id, date, tipoff, teams, neutral are read. No score column is ever loaded |
| 2 | steps 2a/2b | Team form is READ from cached round-1 designs (`possession_outcome/design.parquet`, `clock/design.parquet`) keyed by (game_id, team). A future game has no design row, because designs are built from that game's own chances/possessions | Recompute with the SAME functions (`PO.build_team_form`, `pace` ratings join) on source tables cut to `game_date < D`, with one zero-content STUB row per slate team-game. Every as-of statistic is `cumsum - value`, so a stub row's own features are the state after all earlier games and the stub adds nothing to itself |
| 3 | steps 2c/2d | `RB.build_design`, `FG.build_design` run on all seasons INCLUDING the slate season's future games, then filtered to the season | Same functions, `events=` cut to `game_date < D` plus stub events, `universe=` cut the same way. Restricted to seasons {s-1, s} (features only look back one season) |
| 4 | step 3 | Rotation: `player_game_minutes(load_team_possessions(season))` over the whole season; priors keyed (game_id, team) exist only for games with complete on-floor data (95%); the other 5% take an anonymous league-mean fallback profile | Same `build_asof_player_features` / `build_priors` on `tp` cut to `< D` plus a stub `pg` row (pid=-1, 0 minutes) per slate team-game. EVERY live game gets a real prior (no 5% fallback). Documented parity difference |
| 5 | steps 3-6 | Player joins are `merge_asof(backward)` from design rows; a player with no row for THIS game inherits an earlier row (stale by any number of games, and league-centred at the old date) | Exact keyed join: stub events put every candidate on the floor, so each candidate has a row for the stub game with the current as-of state. Live is fresher than backtest here; the difference is quantified in the parity table, not hidden |
| 6 | step 2b, 5, 6 | Fill constants computed over the FULL season table: `tempo_default = nanmedian(ck.tempo_prior_game)`, usage `fallback = nanmedian(u_rate)`, rebound `med`, FG `fillna(median)`. These include rows dated on/after game day | Live fills come from the cut tables (`date < D`) only. Any difference is reported as a LEAK FINDING in the parity table |
| 7 | fg_make / FT / usage builders | Opening-day league rate is back-filled from the NEXT date (`bfill`), documented forward-looking in the modules | Not reproducible live (no next date). On opening day the live value is NaN then the module's own median/0.70 fallback. Reported as a difference, opening day only |
| 8 | design `days_since_start` | `min(game_date)` over played chances of the season | Season start taken from the schedule (`min` over the schedule for the season), known pregame |
| 9 | own ratings join | `own_ratings_{season}.parquet` must contain the row `as_of_date == D` for every team | Read with `as_of_date` forced to D and assert every slate team has a row and that no row with `as_of_date > D` is used. Producing that row for 2027 has no entry point yet (gap 3): `own_ratings_2027` does not exist |
| 10 | adapters / artifacts | `Adapters.load(inp, fold, season)` reads season/fold-keyed dated-refit directories (`event_*_F2_2025`, clock/rebound/FT manifests). Nothing exists for 2027 | Live runner takes `--fold F2 --season 2025` for the parity test (artifacts present). For 2027 a refit schedule through 2025-26 is a separate modelling/ops item (listed in the results section) |
| 11 | `rules` | Rule constants (dead-ball share, and-one rates, foul accrual) derived from the FOLD's training seasons | Copied from the fold's `names_*_v2.json` with provenance recorded; recomputation for a new fold is remaining work |
| 12 | runner | `run_engine.py`: `backtest=True`, `sealed_touched=False`, reads inputs by `{fold}_{season}` tag, no tipoff or created_at on rows, season >= 2026 refused | `run_engine_live.py`: in-process, 1 core, imports `run_engine.engine_provenance`; output rows joined to slate tipoff and stamped `created_at`; `assert created_at < tipoff` per row, raising on any violation |
| 13 | cutoff | Nothing checks the wall clock | Guard: as-of timestamp T must be `< min(tipoff)` of the slate; every SOURCE game used must have `game_date < D` and `tipoff_utc + 4h < T` |

## 2. Leak-guard design

`created_at` is the build time in UTC, passed in (`--as-of`) for parity runs and `now()` for real runs;
it must satisfy `created_at <= as_of`-cutoff semantics: `as_of < tipoff` for every slate game, and every
source table row used has `game_date < D`. Both are asserted in `src/cbb_sim/live/guards.py` and again on the
output frame in the runner.

## 3. Files (all new)

- `src/cbb_sim/live/__init__.py`, `guards.py` (cutoff and created_at asserts), `stubs.py` (stub-row makers), `schedule.py` (CBBD and universe schedule loaders, crosswalk lookup)
- `scripts/build_engine_inputs_live.py`, `scripts/run_engine_live.py`, `scripts/diag_live_parity_v1.py`, `scripts/diag_live_day1_v1.py`
- `data/reference/team_crosswalk_v2.parquet` (adds West Florida) if the crosswalk row can be sourced from CBBD/ESPN ids already on disk
- `tests/test_live_guards.py`

## 4. Proposed diffs to shared files (NOT applied)

See the results section, if any turned out to be needed.
