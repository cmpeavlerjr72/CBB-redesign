#!/usr/bin/env python
"""
pull_preseason_refresh_v2.py -- re-run scripts/pull_preseason.py into a versioned sibling
directory (data/raw/preseason/2027_v2_20260930/) so the 2026-09-10 pull is never overwritten,
then add (a) a hoopR 2027 schedule re-download to a sibling file and (b) a small CBBD /lines
probe for season 2027 (row/provider counts only) and the 2026 provider census.
Lane E ops readiness audit, 2026-09-30. No model code touched.
"""
import argparse, importlib.util, json, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd, requests

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pp", ROOT / "scripts" / "pull_preseason.py")
pp = importlib.util.module_from_spec(spec); sys.modules["pp"] = pp; spec.loader.exec_module(pp)
_ap = argparse.ArgumentParser(description="versioned preseason refresh; default tag keeps the 2026-09-30 sibling (never overwrite another tag's dir)")
_ap.add_argument("--tag", default="20260930", help="output dir data/raw/preseason/2027_v2_<tag>; pass the pull date for a new sibling")
_ap.add_argument("--dry-run", action="store_true", help="print the output dir and exit without any call")
_args = _ap.parse_args()
OUT = ROOT / "data" / "raw" / "preseason" / f"2027_v2_{_args.tag.replace('-', '')}"
if _args.dry_run:
    print(f"dry-run: would write {OUT} (exists={OUT.exists()}); no calls made")
    sys.exit(0)
OUT.mkdir(parents=True, exist_ok=True)
pp.OUT_DIR = OUT
sys.argv = ["x", "--max-calls", "80"]
rc = pp.main()

# hoopR schedule sibling
try:
    url = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/schedules/parquet/mbb_schedule_2027.parquet"
    r = requests.get(url, timeout=60, allow_redirects=True)
    if r.status_code == 200:
        (OUT / "hoopr_mbb_schedule_2027.parquet").write_bytes(r.content)
    print("hoopr", r.status_code, len(r.content))
except Exception as e:
    print("hoopr fail", e)

# lines probe (counts only)
key = pp.pull_cbbd.load_api_key()
s = requests.Session(); s.headers.update({"Authorization": f"Bearer {key}", "accept": "application/json"})
tr = pp.pull_cbbd.CallTracker(max_calls=10)
probe = {}
for label, params in {
    "season2027_nov": {"season": 2027, "startDateRange": "2026-11-01T00:00:00Z", "endDateRange": "2026-11-15T00:00:00Z"},
    "season2027_all": {"season": 2027},
}.items():
    try:
        d = pp.pull_cbbd.http_get(s, tr, "/lines", params)
        provs = {}
        for g in d:
            for l in g.get("lines") or []:
                provs[l.get("provider")] = provs.get(l.get("provider"), 0) + 1
        probe[label] = {"games": len(d), "providers": provs}
    except Exception as e:
        probe[label] = {"error": str(e)[:200]}
try:
    pr = pp.pull_cbbd.http_get(s, tr, "/lines/providers", {})
    probe["providers_endpoint"] = pr
except Exception as e:
    probe["providers_endpoint"] = {"error": str(e)[:200]}
probe["calls"] = tr.count; probe["remaining"] = tr.remaining
probe["at"] = datetime.now(timezone.utc).isoformat()
(OUT / "lines_probe_2027.json").write_text(json.dumps(probe, indent=2, default=str))
print(json.dumps(probe, default=str)[:1500])
sys.exit(rc)
