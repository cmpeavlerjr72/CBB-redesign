# Daily post-game ingestion (Lane K, 2026-09-30)

Closes audit gap 2 (`docs/ops/readiness_2026-27_2026-09-30.md`). Files, all new: `scripts/pull_daily_ingest_v1.py` (the command),
`scripts/chain_ingest_daily_v1.py` (chain step, one-line hook, `chain_daily.py` not edited), `scripts/diag_ingest_replay_v1.py`
(scratch-root builder and table-by-table comparison), `scripts/diag_ingest_live_builder_replay_v1.py` (runs the live inputs
builder against any data root), `tests/test_daily_ingest.py` (4 tests). No existing script or `src/cbb_sim` file was edited.

## 1. Source tables the live path reads, where they come from, how they refresh

The read set was traced by an open() audit hook on `build_engine_inputs_live.py` (slate 2025-01-15), not guessed.

| table the live path reads | source | freshness unit | derived from | existing builder | incremental today? |
|---|---|---|---|---|---|
| `data/raw/hoopr/schedules/mbb_schedule_{s}` (status, scores, tip time, conference ids) | hoopR-mbb-data raw GitHub parquet | whole-season file, re-published daily | source | `pull_hoopr.py` | no: re-downloads whole files (schedules 2.9 MB, pbp 63 MB, box 0.8 + 3.5 MB, shots 9.6 MB) |
| hoopR team_box, player_box, pbp, shots `_{s}` | same | same | source | `pull_hoopr.py` | no (same) |
| `data/raw/cbbd/games_{s}` (status, points, period points) | CBBD `/games` (startDateRange window) | per call | source | `pull_cbbd.py` | full season in 14-day windows (rewrites the file); library call is windowed, so yes via new code |
| `data/raw/cbbd/pbp/plays_{s}` (event layer input, also `players.py`) | CBBD `/plays/date?date=` (date bucket = UTC date of startDate, or one day earlier for midnight-placeholder tips; a game appears in bucket D, D+1 or both, verified over all of 2025) | per date | source | `pull_cbbd_pbp.py` | yes per date file, but it walks the padded season range and re-concats; new code does D and D+1 only |
| `games_universe.parquet` | hoopR schedules + pbp + box presence + CBBD games + plays + lines | rows per game | hoopR + CBBD | `build_game_universe.py` | no: full rebuild; `build_universe([s])` is per season; new code runs it and keeps the verified games' rows |
| `possessions/possessions_{s}` (v1) and `possessions/chances` ; `possessions_v2/chances_{s}` | CBBD plays | rows per game | plays + universe | `build_possessions.py` (+ validation battery) | script: full season. Library `segment_season(universe=subset)`: yes per game, reproduces disk exactly. v1 = v2 classifier with the rim override OFF (`rim_override_max_ft=0`), and the v1 chance table has no `fga_*`/`fgm_*` columns: reproduced |
| `models/fg_make/events_v2_shotshooter`, `models/usage_v2/events_v2_shotshooter`, `models/rebound/events_v1`, `models/free_throw/{attempts,trips}_v1_era` | CBBD plays | rows per game | plays + universe (fg, usage: `pbp_complete` universe; rebound, FT: D-I non-truncated) | built and cached inside trainers (`train_fg_make_v3_shooter.py` etc.) | scripts: full. Library `FG.build_fg_events`, `U.build_usage_events` + `usable_events`, `RB.build_rebound_events`, `FT.build_trips_and_attempts` accept a universe: yes per game |
| `data/raw/hoopr/player_box` (rotation minutes, usage minutes) | hoopR | as above | | | |
| `truth/team_game_shots_v2`, `player_game_v2`, `game_finals_v2` (graders, not live inputs) | possessions_v2 + hoopR box + CBBD plays | rows per game | | `build_truth_tables(_v2).py` | scripts: full. Library functions run per season (module `SEASONS` patched), keyed to the ingested games |
| `ratings/own_ratings_{s}` | hoopR team box | one row per (as_of_date, team) | | `build_own_ratings.py` | Part 2 |
| `player_crosswalk`, `cbbd/rosters` | CBBD `/teams/roster` (3 calls for all seasons) | season | | `build_player_crosswalk.py` | not daily; must exist for 2027 (see section 6) |
| not read by the live path, not rebuilt daily | `possessions_v3`, `possessions_pbp`, `clock_censoring`, gate references, `lines_close`, trainer design caches | | | | listed so nobody assumes they advance |

### Latency (what is measured and what is not)

- hoopR: the season files are committed by a scheduled job. GitHub commit history of `mbb/{schedules,team_box,pbp,player_box}/parquet/*_2026.parquet`
  (last 100 commits each): 28 to 29 commits between 2026-01-05 and 2026-03-20, at 03:00 to 09:00 ET (peak 07:00 ET, 11 of 28), median gap between commits 15 to 19 h,
  p90 about 22 h. So the games of day D are normally in hoopR after roughly 08:00-09:00 ET on D+1, and occasionally a day later. Box, pbp and
  schedule files land in the same commit. The schedule has `PBP`, `team_box`, `player_box` availability flags per game.
- CBBD: no per-game availability timestamp exists in the API and the season is not running, so the `/games` and `/plays/date` lag is NOT measured. The
  pending list records `first_seen_at` and `last_attempt_at` per game and the manifest records the clock of each fetch, so the lag distribution is measured from day 1.
  Until then the working assumption is "next morning".

## 2. The command

```
.venv/Scripts/python.exe scripts/pull_daily_ingest_v1.py --dates 2026-11-02            # live sources, repo root
.venv/Scripts/python.exe scripts/pull_daily_ingest_v1.py --dates D1,D2 --root <scratch> --hoopr-mode local --hoopr-local-dir data/raw/hoopr
.venv/Scripts/python.exe scripts/chain_ingest_daily_v1.py --date 2026-11-03               # chain form: yesterday + retries
```

Per run: hoopR schedule rows for the dates (plus every pending game inside `--retry-days`, default 14) -> CBBD `/games` for the window ->
**verification** -> CBBD `/plays/date` for buckets D and D+1 (only if a verified game needs them) -> upserts by game id -> derived rebuild for the
verified games -> state tables -> manifest.

**Verified final** = hoopR says completed with both scores present and not 0-0, AND CBBD (joined on `sourceId == game_id`) says `final` with both
points present, AND home and away scores are equal. Everything else is **pending with a reason** (`not_final_either_source`, `awaiting_cbbd_game_row`,
`awaiting_cbbd_final`, `awaiting_hoopr_final`, `score_disagree`, `sides_flipped`, `awaiting_data` = verified but hoopR team box (2 rows) / player box / pbp or CBBD plays missing).
Games with a null hoopR conference id on a side are classed `out_of_scope_non_d1` (counted in the manifest, not retried). Pending games are retried
on every run (3 CBBD calls per pending date), are never appended to any table except the raw hoopR schedule row, and are never filled. A verified game still
missing a piece after `--patience-hours` (72, from tip) is ingested with the gap **listed** (`gaps` column in `finals_verified`, `gaps_listed` in the manifest).
Disagreeing finals (`score_disagree`, `sides_flipped`) stay pending until a human resolves them with a third source (the 2025 case 401722537 would sit there).

State (under `<root>/data/processed/ingest/`): `finals_verified_{season}.parquet` (both sources' scores, verified_at, gaps), `pending_{season}.parquet`
(reason, detail, first_seen_at, last_attempt_at, attempts), `manifests/ingest_{season}_{runid}.json` (rows added and replaced per table, pending list,
verification counts, hoopR sources with bytes/ETag/fetch time, every CBBD call with rows and `X-CallLimit-Remaining`, derived-build notes, warnings).
**Every appended row carries `ingested_at`** (UTC) in the raw and derived tables (the old rows of a table read NaT).

**Idempotence**: games already in `finals_verified` are skipped; an upsert leaves a game untouched when its incoming rows equal the stored ones (hash
of every column except `ingested_at`); writes are temp-file + `os.replace`. Re-running a day leaves all 22 checked files byte-identical (SHA-256, test in section 3).
**Safety**: refuses to write season <= 2026 into the repo root (`--allow-backfill` to override); all writes go under `--root`.

**D-I flag**: `games_universe.is_d1_game` needs a team with >= 5 conference-id games in the season, which is false for nearly every team for the first weeks. The command
therefore uses the existing rule UNION the prior season's D-I teams (changed 0 flags in the replay window; mid-January). Known gap: a team that is new to D-I gets
`is_d1_game = False` on its first four games; when it later crosses the threshold those games are DETECTED (`d1_flips_detected_not_repaired` in the manifest) but not backfilled.

## 3. Replay acceptance test (2025-01-13, 14, 15; following day 2025-01-16)

Scratch root `diag_ingest_replay_v1.py setup` = the on-disk tables with season-2025 rows cut to game_date < 2025-01-13 (other seasons and static inputs hard-linked, 2026 dropped), 407 MB, outside `data/`.
hoopR source = the on-disk season files (stand-in for what hoopR serves; the live mode downloads the same files). CBBD = the real API. One command per day, clock for the patience rule
set to that evening, then one retry run with the clock at 2025-01-21.

| day | verified | pending | out of scope | CBBD calls | wall |
|---|---:|---:|---:|---:|---:|
| 01-13 | 22 | 0 | 0 | 3 (`/games`, `/plays/date` D and D+1) | 69 s |
| 01-14 | 36 | 0 | 0 | 3 | 73 s |
| 01-15 | 46 | 1 (401714544, `awaiting_data`: no pbp in either source) | 1 (401730466, non-D-I) | 3 | 70 s |
| retry, clock 01-21 (72 h passed) | +1 (ingested with gap `hoopr_pbp,cbbd_plays`) | 0 | 1 | 3 | 72 s |
| idempotence reruns (3 dates, same clock) | 0 | 0 | 1 | 0 to 1 | 4 s |

Comparison of the final scratch tables with the on-disk tables cut to the end of 2025-01-15 (row hash of every column except `ingested_at`; `REPLAY_COMPARE`):

| table | rows on disk cut | rows replay | differences |
|---|---:|---:|---|
| hoopR schedules | 3,388 | 3,388 | none |
| hoopR team_box / player_box / pbp / shots | 6,756 / 114,220 / 1,064,140 / 506,127 | 6,754 / 114,183 / 1,063,837 / 505,983 | one game, 401730466 (non-D-I, `out_of_scope_non_d1` by design): 2 / 37 / 303 / 144 rows |
| CBBD games | 3,388 | 3,388 | none |
| CBBD plays | 1,305,605 | 1,304,999 | same game (606 rows) |
| games_universe | 21,874 | 21,873 | same game |
| possessions v1, chances v1, possessions v2, chances v2 | 379,192 / 438,218 (each) | identical | none |
| fg_events_v2_shotshooter, usage_events_v2_shotshooter, rebound_events_v1 | 1,925,129 / 1,213,109 / 1,348,472 | identical | none |
| ft_attempts_v1_era, ft_trips_v1_era | 701,446 / 389,710 | identical rows | `trip_id` only (4,241 / 2,315 cells in 104 games): a build-order counter; new ids are offset past the stored maximum to stay unique. All other columns equal |
| truth team_game_shots_v2, player_game_v2, game_finals_v2 | 38,958 / 614,957 / 19,533 | identical | none |

Before the retry (clock inside the patience window) the same comparison also shows game 401714544 missing from team_box/player_box/universe/truth: the pending behaviour, and after the retry it matches disk.
The added column `ingested_at` is the only other difference (by design, excluded from the hash).

**Live inputs builder, slate 2025-01-16** (62 games, as-of 10:00Z, `--schedule-source cbbd`): built from (a) the repo's on-disk root, (b) an independent root cut to < 2025-01-16 with no ingestion, (c) the replay root.
All three are bit-identical: 12 arrays, 85,188 cells, games table, names/rules json and the event block (results recorded in the JSON below). The run reads no dynamic table outside its root (files opened outside the root: `names_F2_2025_v2.json`, `m_fitted.json`, `player_crosswalk.parquet`, the schedule file, `team_crosswalk.parquet`: all static or the slate).
(A first run lacked `data/raw/cbbd/rosters` in the scratch roots and differed in `usage_rate` and `pos_G/F/C`: position groups come from `load_positions()` reading those roster files. That was a scratch-setup omission, fixed, not a builder leak.)

Files: `docs/ops/daily_ingestion_replay_compare_2026-09-30.json`.

## 4. API budget (CBBD)

Quota read from `X-CallLimit-Remaining`: 28,288 at the 2026-09-30 10:54 audit, 28,255 after the final replay run; my total for today was about 45 calls (two replays, debugging included). Steady state per ingested game date: **3 calls**
(`/games` window, `/plays/date` D, `/plays/date` D+1); each pending date retried adds 3. Catch-up of N dates in one invocation: 1 + (N+1). A full season of about 155 game days is about 470 calls plus retries, under 2% of a 30,000 monthly quota.
`--cbbd-max-calls` (default 12) stops a runaway run; the client stops at `X-CallLimit-Remaining` < 20,000 (existing `pull_cbbd_pbp.http_get` floor).

## 5. Known limits and what was not done

- hoopR revisions after a game is verified are not re-ingested (verified games are frozen; a correction needs `--dates` plus deleting the game from `finals_verified`).
- `possessions_v3`, `possessions_pbp`, gate references, `lines_close`, trainer design caches do not advance daily (not read by the live path).
- `games_universe_v2.parquet` is not advanced.
- The truth-table step uses the player crosswalk only for seasons the crosswalk has (2024-2026 today), see section 6.
- The replay uses the on-disk hoopR files as the hoopR source (the live download path was exercised for HTTP only through the commit-history probe, not end to end: the season is not running, and the 2027 files contain only the schedule).
- Only 2025 (season 2025 = fold 2) was replayed; the repo-root guard means nothing was written to `data/raw` or `data/processed`.
- Tests: `tests/test_daily_ingest.py` (verification classes incl. flipped/disagree/not final/non-D-I, idempotent upsert, season mapping), 4 pass.

## 6. Needed on day 1 of 2026-27, beyond this lane

1. `cbbd/rosters` and `player_crosswalk` for 2027 (CBBD rosters are still empty for 2027): positions, truth `player_game` mapping and the usage position prior read them.
2. Event-table files are all-season files (2022-2025, FT through 2026); 2027 rows are appended to them, the repo-root guard must be lifted deliberately (`--allow-backfill` is for seasons <= 2026 only; season 2027 needs nothing).
3. hoopR `mbb_schedule_2027` has 1,629 games; CBBD has 5,286. Ingestion runs from hoopR's schedule rows, so any CBBD-only game is never a candidate (audit gap 5).
4. Add the step to the chain: `import chain_ingest_daily_v1 as CI; STEPS.insert(1, CI.STEP)`.

## 7. Follow-up (2026-09-30 afternoon): chain v2, dry run on the live 2027 files, flip backfill, box cross-check

New: `scripts/chain_daily_v2.py` (`chain_daily.py` untouched; its schedule, lines, injuries, overrides steps are reused by import), `tests/test_chain_daily_v2.py`.
Stage order follows the readiness stage table with ingestion ahead of the as-of stages: schedule, lines, ingest, ratings, kenpom (off, audit gap 10), injuries, overrides, inputs, sim, publish, grade, bias_clv.
Each stage is exception-isolated, idempotent and logged; live runs write `data/processed/ingest/chain_v2/<date>.json`. The inputs stage asserts `created_at < tipoff` on the slate (`guards.assert_created_before_tipoff`) before anything else.
`CBB_UNSEAL=1` is set only inside `unsealed()`, reached only when `data/overrides/ratings_day1_choices.json` holds all six decisions (implemented options in the script); the dry run never reaches it and reads no 2026 outcome.

Dry run, `chain_daily_v2.py --dry-run`, run date 2026-09-30, slate 2026-11-02 (first real slate), 12 s wall:

| stage | result | expected | verdict |
|---|---|---|---|
| schedule | ok, 0 games today/yesterday both sources | ok | PASS |
| lines | ok, 0 rows | ok | PASS |
| ingest | ok: LIVE hoopR download of `mbb_schedule_2027` (133,247 bytes, 1,629 rows, ETag recorded), 0 schedule rows for 2026-09-29, 0 verified finals, pending list 0, 0 CBBD calls. HEAD probe of the 2027 season files: schedules 200, team_box / player_box / pbp / shots **404** (not created until the first game day) | zero verified, empty pending | PASS |
| ratings | BLOCKED, lists the six missing decisions (seal lift, team source, prior weight policy, new-team prior, fixed-term prior, early D-I rule); 2026 chain not read | stop at the choices | PASS |
| kenpom | skipped (PM decision) | skipped | PASS |
| injuries, overrides | ok (0 rows) | ok | PASS |
| inputs | guard PASS (created 17:54Z < first tip 2026-11-02 05:00Z); 118 mapped / 38 unmapped games; day-1 census reproduces **14 of 14** of lane G's breakage rows (own_ratings, tipoff, possession_outcome, rebound, fg_make, free_throw, usage, rotation, roster, ids, adapters, bonus era, rule constants, lines); 19 rows: 8 DEGENERATE, 6 MISSING, 3 BREAKS, 1 partial, 1 STALE | day-1 list | PASS (blocked, as expected) |
| sim, publish, grade, bias_clv | not_built | not built | FAIL (known gaps) |

Finding and fix: hoopR returns 404 for team_box / player_box / pbp / shots 2027 until the first game day, which would have raised in the ingest on the first real day. `HooprSource` now treats a 404 on those datasets as an empty frame (recorded in the manifest), so the day's games go to the pending list as `awaiting_data` and are retried.

**D-I flip backfill** (implemented): games stored earlier with `is_d1_game = False` whose teams now qualify are rebuilt (universe row, possessions, events, truth) in the next run that has candidates; listed as `d1_flips_backfilled`. Test on the replay root: one ingested game (401725665) had its flag set False and all its derived rows deleted, then 2025-01-16 was ingested; the game's rows came back and all tables equal disk (same two known differences: the non-D-I games and `trip_id`).
**CBBD box cross-check** (implemented): `/games/teams`, one call per run, compared with hoopR team_box per game and side. Hard fields (points, fgm, tpm, ftm) must agree or the game goes pending as `box_disagree`; soft fields (fga, tpa, fta, oreb, dreb, tov) are reported. A missing CBBD team box is `awaiting_data` until the patience window. Replay 2025-01-12..17: 448 team rows, 0 hard mismatches, soft mismatching rows fga 1, tpa 2, oreb 1, dreb 4, tov 1; the 2025-01-16 ingest compared 62 games, 0 hard, 0 soft.

CBBD calls in this follow-up: 3 probes of `/games/teams`, 4 for the 2025-01-16 replay ingest (`/games`, 2 x `/plays/date`, `/games/teams`), 4 for the chain dry run = 11. Steady state is now **4 calls per ingested game date** (3 + `/games/teams`), 3 per pending date retried; 0 calls on a day with nothing to do. Quota about 28,246 remaining. Day-1 list for the PM is section 4 of `docs/ops/own_ratings_daily_2026-09-30.md` plus the `ratings_day1_choices.json` keys above.
