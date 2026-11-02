"""
diag_live_grid_espn_capture_v1.py -- B0 ESPN capture for the CBB live layer: one scoreboard call per minute, 10 calls, while at least one
men's college basketball game is in progress. Saves raw JSON to results/diag/live_grid_<date>/espn_capture/ and prints the fields the
client reads (status.type.name/state, period, clock, displayClock, detail, situation presence, halftime representation).
Low volume by design (one request a minute, curl.exe because python TLS is blocked on this box). Waits (polling every 60 s, max
--wait-min) for the first live game; those waiting polls are saved only when a game is live.

    python scripts/diag_live_grid_espn_capture_v1.py --date 20261102 --calls 10
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={d}&limit=400"


def fetch(d):
    r = subprocess.run(["curl.exe", "-s", "--max-time", "30", URL.format(d=d)], capture_output=True, text=True)
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--wait-min", type=int, default=600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"results/diag/live_grid_{a.date[:4]}-{a.date[4:6]}-{a.date[6:]}/espn_capture")
    out.mkdir(parents=True, exist_ok=True)
    saved, waited = 0, 0
    seen = {}
    while saved < a.calls and waited <= a.wait_min:
        d = fetch(a.date)
        live = [e for e in d.get("events", []) if e["status"]["type"]["state"] == "in"]
        if live:
            (out / f"scoreboard_{time.strftime('%H%M%S')}.json").write_text(json.dumps(d), encoding="utf-8")
            saved += 1
            for e in live:
                c = e["competitions"][0]
                st = e["status"]
                key = (st["type"]["name"], st["period"])
                seen.setdefault(key, (e["shortName"], st["clock"], st["displayClock"], st["type"]["detail"], st["type"]["shortDetail"],
                                      "situation" in c, c.get("format")))
            print(f"[{saved}/{a.calls}] live games: {len(live)}; statuses seen so far: {sorted(seen)}", flush=True)
        else:
            waited += 1
        if saved < a.calls:
            time.sleep(60)
    print(json.dumps({f"{k[0]}|period{k[1]}": v for k, v in seen.items()}, indent=1))


if __name__ == "__main__":
    main()
