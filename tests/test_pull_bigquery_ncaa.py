import importlib.util
import json
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "pbq", Path(__file__).resolve().parent.parent / "scripts" / "pull_bigquery_ncaa_v1.py")
pbq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pbq)


def frame():
    return pd.DataFrame({"game_id": [f"g{i}" for i in range(60)],
                         "season": [2014 + i % 3 for i in range(60)],
                         "event_coord_x": range(60), "event_coord_y": range(60)})


def test_pick_season_column():
    assert pbq.pick_season_column(["game_id", "Season"]) == "Season"
    assert pbq.pick_season_column(["game_id"]) is None


def test_partition_by_season_lossless():
    parts = dict(pbq.partition_frame(frame(), "season"))
    assert set(parts) == {"2014", "2015", "2016"}
    assert sum(len(p) for p in parts.values()) == 60


def test_partition_hash_chunks_deterministic_and_lossless():
    df = frame().drop(columns="season")
    a = {k: list(v.game_id) for k, v in pbq.partition_frame(df, None, 5)}
    b = {k: list(v.game_id) for k, v in pbq.partition_frame(df, None, 5)}
    assert a == b and sum(len(v) for v in a.values()) == 60
    assert all(k.startswith("chunk") for k in a)


def test_manifest_append(tmp_path):
    base = tmp_path / "bq"
    p = base / "t" / "t_2014.parquet"
    p.parent.mkdir(parents=True)
    frame().to_parquet(p)
    mp = base / "manifest.json"
    pbq.append_manifest(mp, base, p, "t", "2014", "SELECT 1", 123)
    pbq.append_manifest(mp, base, p, "t", "2014", "SELECT 1", 123)
    m = json.loads(mp.read_text())
    e = m["t/t_2014.parquet"]
    assert len(m) == 1
    assert e["bytes"] == p.stat().st_size and len(e["sha256"]) == 64
    assert e["bytes_processed"] == 123 and e["query"] == "SELECT 1"
    assert {"path", "downloaded_at", "dataset", "season", "status"} <= set(e)
