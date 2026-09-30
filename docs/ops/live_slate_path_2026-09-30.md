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

## 4. Proposed diff to a shared file (NOT applied)

The served event adapter (`ENGINE_EVENT=round2_s1`, the default) does not read `team_static`. It reads a
(G, 2, 16) block persisted per game ROW of the backtest universe
(`event_round2_s1_{fold}_{season}/team_block.npz`) and raises if `len(block) != inp.n_games`. The live runner
works around it (private adapter directory plus `adapters.ENGINE_DIR` rebinding inside `run_engine_live.py`).
The clean fix is a two-line change plus one optional `EngineInputs` field:

```
--- a/src/cbb_sim/engine/inputs.py   (EngineInputs: new optional field, saved and loaded with the arrays)
+    event_team_block: np.ndarray | None = None   # (G, 2, 16) round-2 event block for THESE rows
--- a/src/cbb_sim/engine/adapters.py (EventAdapter._load_dated)
-        z = np.load(d / "team_block.npz")
-        team_block = z["team_block"]
+        team_block = getattr(inp, "event_team_block", None)
+        if team_block is None:
+            team_block = np.load(d / "team_block.npz")["team_block"]
```

## 5. Results (2026-09-30)

### 5.1 What exists

`src/cbb_sim/live/{guards,features,players}.py`, `scripts/build_engine_inputs_live.py`, `scripts/run_engine_live.py`,
`scripts/diag_live_parity_v1.py`, `scripts/diag_live_engine_parity_v1.py`, `scripts/diag_live_day1_v1.py`,
`scripts/build_team_crosswalk_v2_westflorida.py` (plus `data/reference/team_crosswalk_v2.parquet`), `tests/test_live_guards.py`
(6 tests). Every as-of statistic is produced by the existing module function (`PO.build_team_form`, `RB.team_rebound_form`,
`RB.player_rebound_rates`, `FG.build_design`, `FT.build_ft_design`, `usage.build_player_asof`, `ROT.build_asof_player_features`
and `build_priors`, `orat.join_as_of`); column lists and slot order are imported from `build_engine_inputs.py`. About 40 s per
slate on one core. Two glue fragments are re-typed rather than imported, because the originals live inside `main()` or a trainer
script: array placement, and the round-4 `sh_assisted_share` panel (both exercised by the parity test).
`build_live` refuses seasons that include sealed 2026 unless CBB_UNSEAL=1, so the prior-season carry for 2027 stays blocked until unsealing.

### 5.2 Train/serve parity (F2, season 2025; live path with as-of = first tip minus 30 min, slate treated as unplayed)

87 column groups compared per date (27 team_static, 16 round-2 event block, 8 roster/rotation arrays, 29 slot columns, 5 usage, 2 reb).
UNEXPLAINED cells: 0 on all four dates. Every differing cell is attributed to one cause below.

| date | games | cells compared | cells differing | causes of the differing cells |
|---|---:|---:|---:|---|
| 2024-11-04 (opening day) | 111 | 36,186 | 2,317 | BT_NO_PO_DESIGN_ROW 1,156; BT_TEAM_GAME_NOT_IN_R2_DESIGN 1,149; BT_CLOCK_FALLBACK 12 |
| 2024-11-12 | 56 | 54,076 | 6,725 | BT_STALE_ROW 6,336; BT_ROT_FALLBACK 32 (7 team-games); BT_FIRST_ROW_CLASS_DEPENDENCE 7; BT_CLOCK_FALLBACK 6 |
| 2025-01-15 | 47 | 58,198 | 12,507 | BT_STALE_ROW 12,139; BT_NO_PO_DESIGN_ROW 52; BT_ROT_FALLBACK 25; BT_TEAM_GAME_NOT_IN_R2_DESIGN 30; BT_FIRST_ROW_CLASS_DEPENDENCE 7 |
| 2025-03-04 | 44 | 57,220 | 12,976 | BT_STALE_ROW 12,960; BT_CLOCK_FALLBACK 12; BT_FIRST_ROW_CLASS_DEPENDENCE 4 |

Exactly equal on every date (0 differing cells): all 16 possession-outcome columns, ratings, site, season_idx, days_since_start,
rebound team form, fg_make team form (all six), and the whole round-2 event block for covered team-games. Causes:

- BT_STALE_ROW: the backtest builder joins player values with `merge_asof(backward)`; a player with no design or as-of row for THIS game
  (he did not take a shot of that class, or did not appear in the usage/rebound panel) inherits his latest EARLIER row, possibly from the
  previous season and at an old league-centring date. Proof, not assumption: for the fg per-class columns the backtest value equals the
  last earlier design row in 100% of stale cells (2,147 / 3,764 / 3,973 cells). Live joins exactly. Reach: `shooter_att_c__rim` differs in
  27% / 41% / 42% of real slots on the three mid-season dates (max 130 attempts in early November: last season's counts).
  The round-4 shared columns (`sh_share_*`, `sh_assisted_share`) exist in the backtest only on the RIM-frame rows of the wide slot
  source (the outer merge leaves them empty for other classes), so about 45% of slots carry stale or 0.0 shot-mix shares (used by the B3/B4 arms, not by served B1).
- BT_ROT_FALLBACK: the backtest gave the team-game the anonymous league-mean rotation because THE GAME'S OWN on-floor data were
  incomplete (1,144 of 11,420 team-games, 10.0%); live always builds a real prior.
- BT_NO_PO_DESIGN_ROW: 235 of 11,420 team-games (2.1%, about 117 games) have an all-zero possession-outcome block including
  `season_idx = 0` (2022) and `site_home = site_away = 0`, because the game has no possession table. Consumers of `team_static` (fg_make, clock adapter, reference event arm) see these.
- BT_TEAM_GAME_NOT_IN_R2_DESIGN and BT_CLOCK_FALLBACK: 530 team-games not in the round-2 design are back-filled from an earlier game or left at 0.0 (`build_engine_event_round2.py`); 546 clock team-games take the season-wide fallback constants.
- BT_FIRST_ROW_CLASS_DEPENDENCE: see leak finding 1.

### 5.3 LEAK FINDINGS (backtest inputs that use information from on or after game day)

1. `has_prior_season_fg` (slot column) is class-specific in the design but stored class-independent; the backtest value is that of whichever
   shot class the player took FIRST in the game being predicted. 18 of about 3,400 real slots differ across the three mid-season dates
   (7 / 7 / 4). Weak and binary, but game-day outcome information in a served array. Live uses "any class" (max over the player's class rows); the definition is the PM's to fix.
2. Fill constants computed over the WHOLE season table, including rows dated on or after game day: clock `tempo_prior_game`
   fallback 68.6997 (live as-of league mean on the same dates: 67.15 to 70.57, up to 1.9 possessions off), used for 2 to 80 team-games per date;
   usage `no_history_rate` per class (abs diff up to 0.0022 on rates of 0.03 to 0.05) used by 77 / 53 / 2 real slots; rebound player-rate medians and the fg `fillna(median)`. Small, but future-derived.
3. Freshness of a player's slot features in the backtest is conditional on his participation in the game (a row exists only if he shot or was on the floor). Live cannot see participation. Value-neutral, but it is a channel.
4. Opening-day `bfill` of the league rate (fg_make, free_throw, usage) is documented forward-looking. On 2024-11-04 no served array carries it (team form takes the 0.0 branch, no candidates), so no served column is affected.

None of these alters the four-date exact equality of the team-form statistics, which are strictly-before by construction.

### 5.4 Engine parity (2024-11-12, 56 games, seeds 0..2, default flags)

- Backtest arrays run in reversed game order: bit-identical to forward order (RNG keyed on (seed, game_id, family) is aligned).
- Live team/roster/rotation/event arrays plus backtest slot arrays ("hybrid"): the 50 games whose team, roster, rotation and event inputs are equal give BIT-IDENTICAL output (150 rows). So the live team_static, round-2 event block, roster and rotation arrays reproduce the backtest engine exactly.
- Full live inputs vs backtest arrays: no game has fully identical inputs (stale slot rows differ in all 56). Mean |delta pts| 2.6 / 2.4 (home/away) against a within-game seed SD of 8.6 / 8.3; mean shift live minus backtest +0.80 / +0.06; possessions |delta| 0.47 (SD 4.4). The slot-array effect alone (live vs hybrid) is 1.77 / 1.73 |delta pts|. Three seeds: noise-level, not an estimate of bias.
- `run_engine_live.py` without `--replay` on this past slate raises `LeakGuardError` (created_at now is at or after tipoff); season 2026 is refused without CBB_UNSEAL=1.

### 5.5 Day-1 (first real slate 2026-11-02, season-2027-only context; nothing invented, sealed 2026 not read)

156 CBBD games that date; 118 map to engine ids; the 38 unmapped all have a non-D-I side (exhibitions), including West Florida vs Southeastern Baptist College.
`team_crosswalk_v2.parquet` (West Florida, ESPN 2697 / CBBD 1073) recovers 29 season games (4,887 to 4,916 both-side-mapped, matching the audit).

| family | feature | state | what is missing | consumer |
|---|---|---|---|---|
| ratings | `own_ratings_2027`, off/def_rating_*, tempo_rel, league_tempo_mean | BREAKS | file absent, no daily entry point | possession_outcome, clock, rebound, fg_make |
| schedule | tipoff_utc | DEGENERATE | 105 of 156 games `startTimeTbd`, CBBD stamps midnight ET; `created_at < tipoff` is provable only for runs before midnight ET of game day unless real tips (hoopR/ESPN) are supplied | guard |
| possession_outcome | 8 team-form columns (and the round-2 event block) | DEGENERATE | 100% exactly 0.0; no prior-season carry exists in code | event adapter |
| rebound | off_oreb_c, opp_def_dreb_c; reb_rate | DEGENERATE | 100% 0.0; player medians NaN | rebound, attribution |
| fg_make | off_make_c/def_allow_c (6), shooter block (15), round-4 slot columns (10) | DEGENERATE | 0.0; 0 named candidates | fg_make |
| free_throw | shooter_ft_asof, shooter_fta_asof, prior_season_ft, has_prior_season_ft | DEGENERATE | 0 rows | free_throw |
| usage | usage_rate x5 | DEGENERATE | fallback NaN then 0.0 (no rows dated before D) | usage, attribution |
| rotation | roster, rot_share/srank/start/fpm/pavail | DEGENERATE | 0 of 236 team-games have a prior; all anonymous league-mean | rotation |
| roster | CBBD 2027 rosters, availability.csv | MISSING | 0 player rows for 1,535 teams; overrides empty | candidates, positions |
| ids | player_crosswalk season 2027 | MISSING | covers 2024-2026 | roster_espn |
| adapters | dated-refit artifacts for 2027 | MISSING | only F2_2025 keys exist (event, clock, rebound, FT, fg, rotation manifests) | all adapters |
| rules | bonus era table | BREAKS | seasons 2023-2026 only; 2027 absent | GameState |
| rules | dead_share, and_one, foul accrual | STALE | derived on 2022-2024, copied from the template | engine loop |
| lines | 2027 line rows | MISSING | none until about 1 week before tip | market scorecard |

### 5.6 Remaining work

1. Adapter artifacts for 2027 (dated refit schedule through 2025-26, rule constants, bonus era 2027): PM manifest plus 8 to 12 h.
2. Daily `own_ratings` entry point for a live date: 4 to 6 h.
3. Post-game ingestion feeding possessions_2027, fg/rebound/FT/usage event tables and player box (audit gap 2, the same tables these families read): 10 to 14 h.
4. Day-1 priors: a PM bake-off, then wiring the carry into the team-form, rotation, usage and shooter families (`prior_season` columns exist for shooters only): 8 to 12 h after pre-registration.
5. Real tip times for TBD games (hoopR/ESPN scoreboard): 2 to 3 h. Candidate rosters from CBBD rosters once populated: 4 to 6 h.
6. Backtest-input defects for a v3 builder (5.2, 5.3): 4 to 6 h plus a paired sim.
7. Adopt the shared-file diff in section 4: 1 h.

Resume commands:

```
.venv/Scripts/python.exe scripts/build_engine_inputs_live.py --slate-date D --as-of ISO --season S --fold F2 --schedule-source cbbd --schedule-path data/raw/preseason/2027_v2_20260930/games_2027.parquet --crosswalk data/reference/team_crosswalk_v2.parquet
.venv/Scripts/python.exe scripts/run_engine_live.py --tag LIVE_F2_S_D --season S --seeds N [--replay --created-at ISO]
.venv/Scripts/python.exe scripts/diag_live_parity_v1.py --dates 2024-11-12,2025-01-15,2025-03-04
.venv/Scripts/python.exe scripts/diag_live_engine_parity_v1.py --date 2024-11-12
.venv/Scripts/python.exe scripts/diag_live_day1_v1.py
```
