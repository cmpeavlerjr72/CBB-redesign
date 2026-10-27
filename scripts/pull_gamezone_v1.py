"""Pull the frozen GameZone play-by-play copy (JackLich10/gamezoneR-data) into data/raw/gamezone/.

Private internal research data only (user ruling 2026-10-27): never sync to the public HF dataset
mvpeav/cbb-sim-data-public, never copy into rfq-collab.

Downloads the 16 pbp_<YYYYYY>.rds files (2008-09 .. 2023-24) to data/raw/gamezone/rds/, converts each to
data/raw/gamezone/parquet/gamezone_pbp_<season>.parquet (season = ending year; columns as-is) with
pyreadr==0.5.8 (no R), and writes data/raw/gamezone/manifest.json (url, bytes, sha256, downloaded_at).
Downloaded files are untrusted data: nothing is executed from them; run with `python -I`.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/JackLich10/gamezoneR-data/main/data/play_by_play/rds/"
OUT = ROOT / "data/raw/gamezone"


def main() -> int:
    import pyreadr
    (OUT / "rds").mkdir(parents=True, exist_ok=True)
    (OUT / "parquet").mkdir(parents=True, exist_ok=True)
    man_p = OUT / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    for y in range(2009, 2025):
        tag = f"{y-1}{str(y)[2:]}"
        name = f"pbp_{tag}.rds"
        dest = OUT / "rds" / name
        url = BASE + name
        rel = f"rds/{name}"
        if not dest.exists():
            with urllib.request.urlopen(url, timeout=120) as r:
                dest.write_bytes(r.read())
            when = datetime.now(timezone.utc).isoformat(timespec="seconds")
        else:
            when = man.get(rel, {}).get("downloaded_at", "")
        b = dest.read_bytes()
        ent = {"url": url, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "downloaded_at": when,
               "season": y, "path": rel, "status": "downloaded"}
        pq = OUT / "parquet" / f"gamezone_pbp_{y}.parquet"
        try:
            res = pyreadr.read_r(str(dest))
            df = next(iter(res.values()))
            df.to_parquet(pq, index=False)
            ent["parquet_rows"] = len(df)
            ent["parquet_status"] = "ok"
        except Exception as e:  # report, do not install R
            ent["parquet_status"] = f"FAILED: {type(e).__name__}: {e}"
        man[rel] = ent
        print(y, ent["bytes"], ent.get("parquet_rows"), ent["parquet_status"], flush=True)
    man_p.write_text(json.dumps(man, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
