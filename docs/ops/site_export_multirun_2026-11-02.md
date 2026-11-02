# Site export: multi-run union + verified finals (2026-11-02)

Problem: `export_site_day_v1.py` built a day from ONE run (the newest) and the Hub upload replaces the day folder. The 12:00Z morning
pass skips games already tipped (401920912, 401921422), so its run (`s2500_o0_morning`, 116 games) deleted them from the site.

Change (scripts/export_site_day_v1.py):
- Default (no `--run-id`): the day is the UNION of games over every finished run (`_DONE.json` plus games/players/slate parquet;
  smoke runs without players.parquet are ignored), each game from the NEWEST run containing it. Game order = tipoff, game_id.
- `source_tag` is per game in index.json game entries (new key) and summary.json; the day-level `source_tag`/engine = newest run.
  An explicit `--run-id` still exports that run alone.
- `finalA/finalB`: grade ledger first; else `data/processed/ingest/finals_verified_2027.parquet` rows where hoopR and CBBD are both
  final and agree on both scores. No row, no final (nothing is fabricated).
- Caveat: a game that a newer run dropped on purpose (cancelled) would survive from an older run. Old test runs of the day
  (s200_*) hold the same 118 games, so they add nothing today.

Result 2026-11-02: Hub day has 118 games (116 from s2500_o0_morning, 2 from s2500_o0). Finals for the two tipped games arrive once
the 12:00Z+ ingest writes them to finals_verified_2027.parquet and the next publish pass re-exports.
Tests: tests/test_export_site_day_v1.py (multi-run fixture, verified finals, ledger precedence).
