#!/usr/bin/env python
"""Download pre-2022 hoopR-mbb-data pbp parquet files into data/raw/hoopr_hist/pbp/
and count shot-coordinate coverage per season.  Run from project root.
Reads of the downloaded files should use `python -I`.  FGA = shooting_play true and
neither type_text nor text mentions Free Throw.  Located = both coords non-null and
not both zero... reported strictly: non-null and non-zero for each coordinate."""
import hashlib, json, sys, datetime as dt
from pathlib import Path
import requests
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "hoopr_hist"
PBP = OUT / "pbp"
API = "https://api.github.com/repos/sportsdataverse/hoopR-mbb-data/contents/mbb/pbp/parquet"
RAW = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/pbp/parquet/"


def main():
    PBP.mkdir(parents=True, exist_ok=True)
    listing = requests.get(API, timeout=60).json()
    files = sorted((x["name"], x["size"]) for x in listing if x["name"].endswith(".parquet"))
    man_path = OUT / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    rows = []
    for name, size in files:
        season = int(name.split("_")[-1].split(".")[0])
        if season >= 2022:
            continue
        dest = PBP / name
        key = f"pbp/{name}"
        if not dest.exists():
            r = requests.get(RAW + name, timeout=600)
            r.raise_for_status()
            dest.write_bytes(r.content)
            man[key] = {"bytes": dest.stat().st_size, "dataset": "pbp", "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "path": key, "season": season, "sha256": hashlib.sha256(r.content).hexdigest(), "status": "downloaded", "url": RAW + name}
        names = pq.read_schema(dest).names
        row = {"season": season, "bytes": dest.stat().st_size, "cols_present": [c for c in ("coordinate_x", "coordinate_y", "coordinate_x_raw", "coordinate_y_raw") if c in names]}
        want = [c for c in ("game_id", "type_text", "text", "shooting_play", "coordinate_x", "coordinate_y", "coordinate_x_raw", "coordinate_y_raw") if c in names]
        df = pq.read_table(dest, columns=want).to_pandas()
        row["rows"] = len(df); row["games"] = int(df["game_id"].nunique())
        if "shooting_play" in df:
            sp = df["shooting_play"].fillna(False).astype(bool)
            ft = df["type_text"].fillna("").str.contains("Free Throw", case=False) | df["text"].fillna("").str.contains("Free Throw", case=False)
            fga = df[sp & ~ft]
        else:
            fga = df.iloc[0:0]
        row["fga"] = len(fga)
        row["fga_games"] = int(fga["game_id"].nunique()) if len(fga) else 0
        for a, b in (("coordinate_x", "coordinate_y"), ("coordinate_x_raw", "coordinate_y_raw")):
            if a in df and b in df and len(fga):
                x = fga[a]; y = fga[b]
                ok = x.notna() & y.notna() & (x != 0) & (y != 0)
                row[a + "_located_pct"] = round(100 * ok.mean(), 2)
                g = ok.groupby(fga["game_id"]).any()
                row[a + "_games_with_any"] = int(g.sum())
            else:
                row[a + "_located_pct"] = None; row[a + "_games_with_any"] = None
        rows.append(row); print(json.dumps(row), flush=True)
    man_path.write_text(json.dumps(man, indent=2, sort_keys=True))
    (OUT / "coverage.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
