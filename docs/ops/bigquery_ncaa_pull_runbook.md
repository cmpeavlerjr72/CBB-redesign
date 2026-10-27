# BigQuery ncaa_basketball pull runbook (2026-10-27)

Script: `scripts/pull_bigquery_ncaa_v1.py`. Source: `bigquery-public-data.ncaa_basketball` (Sportradar play-by-play with shot coordinates). Facts and open questions: `docs/research/bigquery_ncaa_basketball_2026-10-27.md`.

## Why this is allowed

The dataset carries a Sportradar notice (secondary source, wording UNVERIFIED): data is "for internal research and testing purposes" and "may not be exploited in any way without express approval from Sportradar". User ruling 2026-10-27: our use is private internal research, within that notice. Consequences: the pulled data stays in gitignored `data/raw/bigquery_ncaa/`; it is not committed, not redistributed, not synced to the public HF dataset, and not used to build a product. Derived findings (bias measurements, coverage counts) may be reported; raw vendor rows may not be published.

## User steps (two)

1. Create a BigQuery sandbox project at console.cloud.google.com (no billing account or card needed) and note the project id.
2. Run once: `.venv/Scripts/python.exe scripts/pull_bigquery_ncaa_v1.py --project <PROJECT_ID> --auth`
   A browser opens for Google sign-in (read-only BigQuery scope). Credentials are cached under `~/.config/pydata/`. The command prints the signed-in account.
   The full auth URL is printed on its own line before the browser opens. If the page shows "Missing required parameter: redirect_uri" (Error 400: invalid_request), the browser copy of the URL was cut at the first `&`: paste the printed URL into the browser instead. The sign-in uses a local webserver on localhost to receive the code, so the browser must be on this machine (not a remote desktop or another device).

## Worker steps

```
set PYTHONIOENCODING=utf-8
.venv/Scripts/python.exe scripts/pull_bigquery_ncaa_v1.py --project P --list
.venv/Scripts/python.exe scripts/pull_bigquery_ncaa_v1.py --project P --schema mbb_pbp_sr
.venv/Scripts/python.exe scripts/pull_bigquery_ncaa_v1.py --project P --pull mbb_pbp_sr --dry-run
.venv/Scripts/python.exe scripts/pull_bigquery_ncaa_v1.py --project P --pull mbb_pbp_sr
```

- `--list`: every table with row count, GB, last-modified (table metadata, free). Resolves the fact sheet's open items 3 and 4 (coverage end, size).
- `--schema`: resolves open item 2 (made flag, clock, team, player columns). Record the schema in a sibling doc.
- `--pull --dry-run`: per-partition and total estimated bytes processed. Nothing runs.
- `--pull`: refuses if the estimate exceeds `--max-gb` (default 50). Writes `data/raw/bigquery_ncaa/<table>/<table>_<season>.parquet` (hash chunks `chunkNN` by `game_id` if no season column), Storage API read with REST fallback, columns untransformed (`event_coord_x`, `event_coord_y` as-is). Each file is appended to `data/raw/bigquery_ncaa/manifest.json` (path, bytes, sha256, downloaded_at, season, dataset, status, query, bytes_processed). Bytes processed are printed per file and in total.
- Also check the other tables in `--list` for men's box score / game tables worth pulling (same command).
- Verify coordinate null rate, units and court frame before use; the fact sheet marks them UNVERIFIED. Sportradar changed action-area coordinates on 2025-12-01 (UNVERIFIED), so check for a vintage break.

## Sandbox limits (fact sheet section 4-5)

- 1 TiB processed queries per month (track via the printed bytes; sum them).
- 10 GiB lifetime storage; tables expire after 60 days. We only read public tables and write locally, so storage limits should not bind.
- No billing account or credit card required.
- Dry-run estimates use the pre-execution bound; the actual bytes are printed after each query.


## Headless box: use a service-account key (added 2026-10-27 after the browser flow failed twice)

This machine is reached over Remote SSH, so a browser opened "here" shows nothing, and Google now
BLOCKS the shared desktop OAuth client that `pydata-google-auth` ships ("This app is blocked. This
app tried to access sensitive info in your Google Account"). The browser route is therefore dead
unless you register your own OAuth client. The supported route is a service-account key:

1. Console (https://console.cloud.google.com), project `graphic-centaur-381400`: IAM & Admin ->
   Service Accounts -> Create service account. Name `cbb-bq-reader`. Grant role **BigQuery User**
   (lets it run query jobs in this project; the public dataset is readable by everyone).
2. Open the account -> Keys -> Add key -> Create new key -> JSON. The file downloads to the laptop.
3. Put it on this server as `C:\Users\devuser\.config\pydata\bigquery_sa_key.json` (drag the file
   into that folder in the VS Code Explorer, or `scp` it). Never put it inside the repo.
4. Every command then works with no `--auth` at all:
   `.venv/Scripts/python.exe scripts/pull_bigquery_ncaa_v1.py --project graphic-centaur-381400 --list`
   (`--sa-key PATH` overrides the default location.)

The `--auth --port 8085` browser flow remains in the script for a machine with a browser; on
Remote SSH it also needs the port forwarded in the VS Code Ports panel.
