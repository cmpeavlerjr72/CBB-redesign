# Data not in git

These stay on the collector machine because they are too big for GitHub (100 MB file limit):

- `data/collector.sqlite` (~9.7 GB): leaderboard snapshots, handle metrics, fills, social feed, 5-min tape aggregates, market metadata
- `data/tape/YYYY-MM-DD.jsonl.gz` (~120-180 MB/day): raw public trade prints
- `samples/inference/KXNFLGAME-26OCT25KCSEA-KC.trades.json` (74 MB)
- `logs/`: collector logs

Ask for an export or a share link if you need any of it.
