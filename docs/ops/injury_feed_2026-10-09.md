# Free injury feed wired into the daily chain (ops worker, 2026-10-09)

Wiring only. No model, parameter or probability weighting. Status Out is applied; Questionable / Doubtful / Day-To-Day / Probable / Suspension are recorded and NOT applied in this version.

## 1. What existed
- `data/processed/injuries/player_out_2026-10-09.csv` (untracked) was written by the chain's old `injuries_parse` stage, i.e. `scripts/pull_injuries_player_out_v1.py` (commit `de00bab`, the 2027 roster / injuries worker). Schema: `athlete_id,date,status,note,espn_team_id,cbbd_team_id,cbbd_player_id,raw_status,out,source,created_at`, header only, 0 rows (the ESPN payload is empty in the preseason). The schema is usable, but:
  - nothing in the served chain read it: `chain_daily_v2.load_availability` feeds only the v2 inputs path, which `chain_daily_v3` (the served chain) never calls; the v3 sim and inputs stages passed no `availability`;
  - on the day-1 path (no in-season rotation priors) `build_live(availability=...)` would not have removed anyone anyway, because the A3 seed fills the slots after the priors;
  - its parser read the ESPN athlete id from `athlete.id`. In the ESPN injuries payload (checked on the NBA endpoint, same ESPN family, because the college payload is empty) the athlete object has NO `id`; the id is only in the player-card link. A non-empty college payload would have raised "schema drift" and, since a crashed stage stops a live chain, halted every stage after it.
- `build_live(availability=...)` and `apply_availability` (drop a pid from a team-game's in-season rotation prior) already existed and are reused.

## 2. Source and terms check (what was checked, 2026-10-09)
| source | finding | verdict |
|---|---|---|
| CBBD API (`api.collegebasketballdata.com`, key from `cfb-props-sim/.env`) | probed `/injuries`, `/players/injuries`, `/teams/injuries`, `/injury`, `/player/status`, `/players/status`, `/availability`, `/transactions`, `/news`: all 404 (the working routes `/games/players`, `/teams/roster`, `/lines`, `/draft/picks` return 200). No OpenAPI document is served at `/docs/json`, `/docs/openapi.json`, `/swagger.json`, `/json`. | no injury endpoint |
| hoopR-mbb-data | box scores and schedules only; no injury or status table | none |
| ESPN `site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/injuries` | HTTP 200, `{"status":"success","season":{"year":2027,"name":"Preseason"},"injuries":[]}`. Same route on the NBA returns 29 teams with statuses `Out`, `Day-To-Day` (shape verified there). College coverage during the season is unverified until the first injuries appear. | **used** |
| `sports.core.api.espn.com` | its `/robots.txt` returned 403 and no route was probed | not used |
| sports-reference, barttorvik, masseyratings, PrizePicks, Underdog, DraftKings | banned by CLAUDE.md | not touched |
Terms: this is ESPN's public JSON API that the project already uses for rosters, schedules and odds snapshots, and the one the task brief names as allowed. `robots.txt` of `site.api.espn.com` cannot be fetched (the CDN answers Access Denied to `/robots.txt`), so no robots rule can be read for that host. `www.espn.com/robots.txt` disallows named AI-training crawlers (GPTBot, anthropic-ai, CCBot and others) for the website. The pull is one request per pass with an identifying User-Agent, public injury-status JSON, used for a private research model; it is not a crawl and not for model training. If the PM reads the www disallow as covering the API host, the stage is a no-op by deleting one line of `chain_daily_v3` (`injuries_feed`) and the sim then applies only manual `availability.csv` rows. I recorded this rather than decide it.
**Coverage caveat:** ESPN college injury coverage is sparse; zero rows means "nothing reported", never "everyone healthy".

## 3. What was built
`scripts/pull_injuries_v1.py`
- one league-wide call; parses `athlete.id`, else the player-card link `/id/<n>/`, else the athlete `$ref`; non-empty payload with no parsed id = hard error;
- writes `data/processed/injuries/injuries_<date>.parquet` (latest pull of the date) with `season, team_id (ESPN), cbbd_team_id, espn_player_id, cbbd_player_id, player_name, status, status_norm, detail, source, pulled_at, snapshot_date`, and one manifest per pull `data/processed/injuries/manifests/injuries_<date>_<HHMMSS>.json` (counts by status, sha1, source URL, unmapped Out rows);
- `out_players(day, now)`: the Out set (`status_norm` in {`out`, `out for season`}) plus manual `status == out` rows of `data/overrides/availability.csv` (stamped with the file's mtime, so a row cannot predate its edit). Leak guard: any row pulled after the sim clock raises `LeakGuardError`.

Chain wiring (`chain_daily_v3`):
- stage `injuries_feed` replaces `injuries_parse` (same position, before overrides and inputs);
- `run_daily_sim_v1.injuries_for(now, season, replay)` loads the Out set for live runs (replay and past seasons: none, so their inputs are bit-identical);
- the A3 day-1 seed gets `out_pids`: `make_seed_fn(..., out_pids=)` removes the pid from the seeded candidate list before slots are assigned, exactly as a player missing from the roster file is handled; the others keep their own S-1 minutes order and move up, and `build_live` renormalises the rotation shares over the slots that remain. In-season team-games use `build_live(availability=...)` (`apply_availability`, existing);
- both the `inputs` census stage and the `sim` stage apply the same rule;
- the Out set is part of the sim cache key; `run_meta.json` carries an `injuries` block (feed date, Out rows, mapped / unmapped to CBBD ids, players removed by the seed, team-games affected, distinct players applied).
A pid that cannot be matched to a CBBD player id (not on any 2026 or 2027 roster file) is reported as `out_unmapped`, not applied.

## 4. Tests (`tests/test_injury_feed.py`, 11)
Parse by `athlete.id` and by player-card link; schema-drift hard error; empty payload is 0 rows; id mapping to CBBD team and player ids; parquet and manifest written; **only Out / Out For Season applied** (Questionable, Doubtful, Suspension recorded, not applied); manual rows merge; late-pulled row raises; no feed file means nobody out; replay never reads the feed; and the end-to-end test on a real 2026-11-02 game: the home side's top named player is made Out through the seed. Assertions: the pid is absent from `roster_cbbd`, `out_removed == {pid: 1}`, everyone else keeps order and moves up one slot, the rotation shares of the side still sum to 1 (slot allocation re-closes), the other side is untouched, and after a real 2-seed engine run the player has no output rows (0 minutes) while the side's named players still carry over 100 minutes per seed.

## 5. Chain run with the feed (evening pass, on demand via the scheduler task, 09:59:44 ET)
- rc 0 (`end exit=0`, 10:13:29), stage `injuries_feed` ok in 0.8 s.
- **Injuries pulled: 0** (ESPN payload `entries_raw: 0`, status success, season 2027 Preseason). Count by status: none. Out rows 0, unmapped 0.
- **Players applied as Out on the 11-02 slate: 0** (`run_meta.injuries.players_applied_distinct = 0`, 0 seed removals, 0 prior drops). Nothing was fabricated to make this non-zero; the Out path is exercised by the test above.
- Sim stage 794 s; cache MISS because the stack and the Out component of the key changed (`supersedes.components_changed = ['injuries_out', 'stack']`).
- Artifacts: `data/processed/injuries/injuries_2026-10-09.parquet` (0 rows, sha1 `80e8ff7`), `data/processed/injuries/manifests/injuries_2026-10-09_135944.json`.

## 6. Left for later / for the PM
- Doubtful / Questionable are recorded only. A probabilistic availability weight would be a model decision (bake-off), not made here.
- Status vocabulary of the college payload is unverified until injuries appear; unknown statuses are stored raw and not applied. First real college entries should be eyeballed once (`status` counts in the manifest).
- The old `pull_injuries_player_out_v1.py` is no longer called by the chain; it is kept because `ops_seal_week_v1` and `tests/test_rosters_injuries_2027.py` still reference it. The untracked `player_out_2026-10-09.csv` is its last output and has no consumer.
- The www.espn.com robots note in section 2 is a PM call.
