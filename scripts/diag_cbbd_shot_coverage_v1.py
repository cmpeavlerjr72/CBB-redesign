#!/usr/bin/env python
"""Sample N games per season (2013-2021) from CBBD /games and /plays/game/{id} and
report the share of shooting plays (non free-throw) carrying shot location x/y.
Reuses scripts/pull_cbbd.py client helpers (key, rate limits). ~ (1 + N) calls/season."""
import json, random, sys
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pull_cbbd as P

N = 20
OUT = Path(__file__).resolve().parent.parent / "docs" / "tests" / "cbbd_shot_coverage_2026-10-27.json"


def shot_loc(p):
    si = p.get("shotInfo") or {}
    loc = si.get("location") or {}
    return loc.get("x"), loc.get("y")


def main():
    s = requests.Session(); s.headers["Authorization"] = "Bearer " + P.load_api_key()
    tr = P.CallTracker(max_calls=400)
    res = []
    for season in range(2013, 2022):
        games = P.http_get(s, tr, "/games", {"season": season})
        games = [g for g in games if g.get("status") == "final" or g.get("homePoints") is not None]
        random.Random(season).shuffle(games)
        n_shots = n_loc = n_games_loc = n_ok = 0
        keys = None
        for g in games[:N]:
            plays = P.http_get(s, tr, f"/plays/game/{g['id']}", {})
            if plays and keys is None:
                keys = sorted(plays[0].keys())
            n_ok += 1
            def is_ft(p):
                z = ((p.get("playType") or "") + (p.get("playText") or "") + str((p.get("shotInfo") or {}).get("range"))).lower().replace(" ", "").replace("_", "")
                return "freethrow" in z
            sh = [p for p in plays if p.get("shootingPlay") and not is_ft(p)]
            loc = [p for p in sh if None not in shot_loc(p)]
            n_shots += len(sh); n_loc += len(loc); n_games_loc += bool(loc)
        row = {"season": season, "games_sampled": n_ok, "shot_plays": n_shots, "located": n_loc,
               "located_pct": round(100 * n_loc / n_shots, 2) if n_shots else None, "games_with_any_located": n_games_loc, "play_keys": keys}
        res.append(row); print(json.dumps({k: v for k, v in row.items() if k != "play_keys"}), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"calls": tr.count, "rows": res}, indent=1))


if __name__ == "__main__":
    main()
