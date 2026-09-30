#!/usr/bin/env python
"""
diag_ingest_replay_v1.py -- Lane K acceptance test for scripts/pull_daily_ingest_v1.py (2026-09-30).

    # 1. build the scratch root: on-disk tables cut to game_date < D0 (season 2025); unchanged files are hard links
    .venv/Scripts/python.exe scripts/diag_ingest_replay_v1.py setup   --root <scratch> --d0 2025-01-13
    # 2. ingest three days with the real command (see the design note), then
    # 3. compare, table by table, scratch vs on-disk cut to the end of the last ingested day
    .venv/Scripts/python.exe scripts/diag_ingest_replay_v1.py compare --root <scratch> --d0 2025-01-13 --d1 2025-01-15

Season 2025-26 (universe season 2026) is sealed and is neither copied nor read. Nothing is written under
data/raw or data/processed of the repo. The comparison hashes every row (minus `ingested_at`), so a row-order
change is not a difference but a changed value, a missing row or an extra row is; every difference is listed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
S = 2025
INGEST_COL = "ingested_at"
HOOPR = ["schedules", "team_box", "player_box", "pbp", "shots"]
STEM = {"schedules": "mbb_schedule", "team_box": "team_box", "player_box": "player_box", "pbp": "play_by_play", "shots": "shots"}
PROC = "data/processed"

# (label, relative path, key column, kind of id: 'hoopr' game_id or 'cbbd' id, scope: 'season2025' = single-season file)
TABLES = [
    *[(f"hoopr_{d}", f"data/raw/hoopr/{d}/{STEM[d]}_{S}.parquet", "game_id", "hoopr", True) for d in HOOPR],
    ("cbbd_games", f"data/raw/cbbd/games_{S}.parquet", "id", "cbbd", True),
    ("cbbd_plays", f"data/raw/cbbd/pbp/plays_{S}.parquet", "gameId", "cbbd", True),
    ("games_universe", f"{PROC}/games_universe.parquet", "game_id", "hoopr", False),
    ("possessions_v1", f"{PROC}/possessions/possessions_{S}.parquet", "game_id", "hoopr", True),
    ("chances_v1", f"{PROC}/possessions/chances_{S}.parquet", "game_id", "hoopr", True),
    ("possessions_v2", f"{PROC}/possessions_v2/possessions_{S}.parquet", "game_id", "hoopr", True),
    ("chances_v2", f"{PROC}/possessions_v2/chances_{S}.parquet", "game_id", "hoopr", True),
    ("fg_events_v2_shotshooter", f"{PROC}/models/fg_make/events_v2_shotshooter.parquet", "game_id", "hoopr", False),
    ("usage_events_v2_shotshooter", f"{PROC}/models/usage_v2/events_v2_shotshooter.parquet", "game_id", "hoopr", False),
    ("rebound_events_v1", f"{PROC}/models/rebound/events_v1.parquet", "game_id", "hoopr", False),
    ("ft_attempts_v1_era", f"{PROC}/models/free_throw/attempts_v1_era.parquet", "game_id", "hoopr", False),
    ("ft_trips_v1_era", f"{PROC}/models/free_throw/trips_v1_era.parquet", "game_id", "hoopr", False),
    ("truth_team_game_shots_v2", f"{PROC}/truth/team_game_shots_v2.parquet", "game_id", "hoopr", False),
    ("truth_player_game_v2", f"{PROC}/truth/player_game_v2.parquet", "game_id", "hoopr", False),
    ("truth_game_finals_v2", f"{PROC}/truth/game_finals_v2.parquet", "game_id", "hoopr", False),
]
LINK_ALWAYS = [   # static / other-season inputs the builders read (hard links, never written)
    "data/raw/cbbd/lines_2025.parquet", "data/raw/cbbd/lines_2024.parquet",
    "data/raw/cbbd/games_2024.parquet", "data/raw/cbbd/pbp/plays_2024.parquet",
    "data/raw/hoopr/schedules/mbb_schedule_2024.parquet", "data/raw/hoopr/team_box/team_box_2024.parquet",
    "data/raw/hoopr/player_box/player_box_2024.parquet", "data/raw/hoopr/pbp/play_by_play_2024.parquet",
    "data/raw/hoopr/shots/shots_2024.parquet",
    f"{PROC}/possessions/possessions_2024.parquet", f"{PROC}/possessions/chances_2024.parquet",
    f"{PROC}/possessions_v2/possessions_2024.parquet", f"{PROC}/possessions_v2/chances_2024.parquet",
    f"{PROC}/possessions/build_report.json", f"{PROC}/possessions_v2/build_report.json",
    f"{PROC}/ratings/own_ratings_2025.parquet", f"{PROC}/ratings/own_ratings_2024.parquet",
    f"{PROC}/player_crosswalk.parquet",
    f"{PROC}/models/rotation/rotation_fit.json", f"{PROC}/models/usage_v2/usage_params_v2.json",
    f"{PROC}/models/fg_make/round4/m_fitted.json", f"{PROC}/truth/diag_finals_resolution_2025.json",
]


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def et_date(start_iso: pd.Series) -> pd.Series:
    d = pd.to_datetime(start_iso, utc=True)
    return d.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()


def date_maps() -> tuple[dict, dict, set]:
    """game_id -> date (hoopR schedule 2025), cbbd id -> ET date (cbbd games 2025), 2026 hoopR ids (sealed, dropped)."""
    s = pd.read_parquet(REPO / f"data/raw/hoopr/schedules/mbb_schedule_{S}.parquet", columns=["game_id", "game_date"])
    h = dict(zip(s["game_id"].astype("int64"), pd.to_datetime(s["game_date"])))
    g = pd.read_parquet(REPO / f"data/raw/cbbd/games_{S}.parquet", columns=["id", "startDate"])
    c = dict(zip(g["id"].astype("int64"), et_date(g["startDate"])))
    s26 = pd.read_parquet(REPO / "data/raw/hoopr/schedules/mbb_schedule_2026.parquet", columns=["game_id"])
    return h, c, set(s26["game_id"].astype("int64"))


def keep_mask(df: pd.DataFrame, key: str, kind: str, maps, last: pd.Timestamp | None, first: pd.Timestamp | None) -> pd.Series:
    """Rows of a table that belong to 2025 games dated in [first, last] (None = open) OR to other non-sealed seasons."""
    h, c, s26 = maps
    k = pd.to_numeric(df[key], errors="coerce")
    ok = k.notna()
    k = k.fillna(-1).astype("int64")
    m = h if kind == "hoopr" else c
    d = k.map(m)
    is25 = d.notna()
    sealed = k.isin(s26) if kind == "hoopr" else pd.Series(False, index=df.index)
    other = ~is25 & ~sealed & ok
    inside = is25.copy()
    if first is not None:
        inside &= d >= first
    if last is not None:
        inside &= d <= last
    return (inside | other).fillna(False)


def cmd_setup(a) -> int:
    root = Path(a.root)
    if root.resolve() == REPO or (REPO / "data") in root.resolve().parents and root.resolve().parts[-1] in ("raw", "processed"):
        raise SystemExit("scratch root must be a separate directory")
    d0 = pd.Timestamp(a.d0)
    maps = date_maps()
    for label, rel, key, kind, single in TABLES:
        df = pd.read_parquet(REPO / rel)
        m = keep_mask(df, key, kind, maps, d0 - pd.Timedelta(days=1), None)
        cut = df[m].reset_index(drop=True)
        if label == "games_universe":                   # drop sealed-season rows
            cut = cut[cut["season"] != 2026].reset_index(drop=True)
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cut.to_parquet(dst, index=False)
        print(f"{label:30s} {len(df):>10,} -> {len(cut):>10,} rows")
    for rel in LINK_ALWAYS + [str(f.relative_to(REPO)).replace("\\", "/") for f in sorted((REPO / "data/raw/cbbd/rosters").glob("roster_*.parquet"))]:
        link(REPO / rel, root / rel)
    (root / "SETUP.json").write_text(json.dumps({"d0": a.d0, "season": S}), encoding="utf-8")
    return 0


def _hash_rows(df: pd.DataFrame, cols: list) -> pd.Series:
    d = df[cols].copy()
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].map(lambda v: "<NA>" if v is None or (isinstance(v, float) and v != v) or v is pd.NA else v if isinstance(v, (str, int, float, bool)) else str(np.asarray(v).tolist()) if hasattr(v, "__len__") else str(v))
    return pd.util.hash_pandas_object(d, index=False)


def compare_table(label, exp: pd.DataFrame, got: pd.DataFrame, key: str) -> dict:
    cols = [c for c in exp.columns if c != INGEST_COL]
    extra_cols = [c for c in got.columns if c not in exp.columns and c != INGEST_COL]
    miss_cols = [c for c in exp.columns if c not in got.columns and c != INGEST_COL]
    out = {"table": label, "rows_expected": int(len(exp)), "rows_got": int(len(got)),
           "extra_columns_in_got": extra_cols, "columns_missing_in_got": miss_cols,
           "ingested_at_rows": int(got[INGEST_COL].notna().sum()) if INGEST_COL in got.columns else 0}
    cc = [c for c in cols if c in got.columns]
    dt = {c: (str(exp[c].dtype), str(got[c].dtype)) for c in cc if str(exp[c].dtype) != str(got[c].dtype)}
    out["dtype_differences"] = dt
    ek = pd.to_numeric(exp[key], errors="coerce").astype("Int64")
    gk = pd.to_numeric(got[key], errors="coerce").astype("Int64")
    eh, gh = _hash_rows(exp, cc), _hash_rows(got, cc)
    e = pd.DataFrame({"k": ek.to_numpy(), "h": eh.to_numpy()})
    g = pd.DataFrame({"k": gk.to_numpy(), "h": gh.to_numpy()})
    eg = e.groupby("k")["h"].agg(["size", lambda x: int(np.bitwise_xor.reduce(x.to_numpy(dtype="uint64")) if len(x) else 0),
                                   lambda x: int(x.to_numpy(dtype="uint64").sum(dtype="uint64"))])
    gg = g.groupby("k")["h"].agg(["size", lambda x: int(np.bitwise_xor.reduce(x.to_numpy(dtype="uint64")) if len(x) else 0),
                                   lambda x: int(x.to_numpy(dtype="uint64").sum(dtype="uint64"))])
    eg.columns = gg.columns = ["n", "x", "s"]
    j = eg.join(gg, how="outer", lsuffix="_e", rsuffix="_g")
    only_e = j[j["n_g"].isna()].index.tolist()
    only_g = j[j["n_e"].isna()].index.tolist()
    both = j.dropna(subset=["n_e", "n_g"])
    rowdiff = both[both["n_e"] != both["n_g"]].index.tolist()
    valdiff = both[(both["n_e"] == both["n_g"]) & ((both["x_e"] != both["x_g"]) | (both["s_e"] != both["s_g"]))].index.tolist()
    out.update(games_expected=int(len(eg)), games_got=int(len(gg)), games_missing_in_got=[int(x) for x in only_e],
               games_extra_in_got=[int(x) for x in only_g], games_row_count_differs=[int(x) for x in rowdiff],
               games_value_differs=[int(x) for x in valdiff])
    # column-level attribution for the value-differing games
    coldiff: dict = {}
    if valdiff:
        vd = set(valdiff)
        ex = exp[ek.isin(vd).to_numpy()].copy(); go = got[gk.isin(vd).to_numpy()].copy()
        ex["_k"], go["_k"] = ek[ek.isin(vd)].to_numpy(), gk[gk.isin(vd)].to_numpy()
        ex = ex.sort_values("_k", kind="stable").reset_index(drop=True); go = go.sort_values("_k", kind="stable").reset_index(drop=True)
        if len(ex) == len(go):
            for c in cc:
                a_, b_ = ex[c], go[c]
                try:
                    if pd.api.types.is_float_dtype(a_) or pd.api.types.is_float_dtype(b_):
                        ne = ~np.isclose(a_.astype(float), b_.astype(float), equal_nan=True, rtol=0, atol=0)
                    else:
                        ne = (a_.astype(str).to_numpy() != b_.astype(str).to_numpy())
                except Exception:
                    ne = (a_.astype(str).to_numpy() != b_.astype(str).to_numpy())
                if int(ne.sum()):
                    coldiff[c] = int(ne.sum())
        else:
            coldiff["_note"] = "row order/count differs inside value-differing games; column attribution skipped"
    out["value_diff_columns_cells"] = coldiff
    out["identical"] = not (only_e or only_g or rowdiff or valdiff or miss_cols)
    return out


def cmd_compare(a) -> int:
    root = Path(a.root)
    d0, d1 = pd.Timestamp(a.d0), pd.Timestamp(a.d1)
    maps = date_maps()
    results = []
    for label, rel, key, kind, single in TABLES:
        exp = pd.read_parquet(REPO / rel)
        exp = exp[keep_mask(exp, key, kind, maps, d1, None)].reset_index(drop=True)
        if label == "games_universe":
            exp = exp[exp["season"] != 2026].reset_index(drop=True)
        got = pd.read_parquet(root / rel)
        r = compare_table(label, exp, got, key)
        results.append(r)
        print(f"{label:30s} exp {r['rows_expected']:>9,} got {r['rows_got']:>9,} identical={r['identical']} "
              f"missing={len(r['games_missing_in_got'])} extra={len(r['games_extra_in_got'])} "
              f"rowdiff={len(r['games_row_count_differs'])} valdiff={len(r['games_value_differs'])} "
              f"extra_cols={r['extra_columns_in_got']} dtype_diff={list(r['dtype_differences'])[:4]}", flush=True)
    Path(a.out or (root / "REPLAY_COMPARE.json")).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("setup", "compare"):
        p = sub.add_parser(n)
        p.add_argument("--root", required=True)
        p.add_argument("--d0", required=True)
        if n == "compare":
            p.add_argument("--d1", required=True)
            p.add_argument("--out", default=None)
    a = ap.parse_args()
    return cmd_setup(a) if a.cmd == "setup" else cmd_compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
