"""Inputs fingerprint for the daily sim cache (ops, 2026-10-09).

The daily sim stage used to cache on its CONFIG only (slate date, seeds, ...), so a later pass with new rosters, tips, ratings or
injuries silently re-served the first sim of the day. `inputs_hash` fingerprints every file the served build reads that can change
between passes; the stage re-runs when it differs and may reuse only when it is identical.

Content, not bytes, for parquet: columns that are pull / build time stamps (`*_at`) are dropped before hashing, because every refresh
rewrites them without changing the data. Everything else is hashed as-is. A missing file hashes as the literal "missing" (so its
appearance changes the key). No model, parameter or default is touched here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]

#: code that defines the served stack for the daily sim: a change here is a new stack, so it forces a re-run
STACK_SRC_GLOBS = ("src/cbb_sim/**/*.py",)
STACK_SCRIPTS = ("run_engine_live.py", "build_engine_inputs_live.py", "build_shot_block_lut_live_v1.py",
                 "build_engine_inputs_day1prior_v1.py", "run_daily_sim_v1.py", "chain_daily_v2.py")
STACK_FILES = ("data/overrides/ao_team_table.json",)


def _h(*parts: bytes) -> str:
    m = hashlib.sha1()
    for p in parts:
        m.update(len(p).to_bytes(8, "little"))
        m.update(p)
    return m.hexdigest()


def frame_digest(df: pd.DataFrame) -> str:
    """Order-insensitive content hash of a frame, ignoring `*_at` stamp columns."""
    df = df.drop(columns=[c for c in df.columns if str(c).endswith("_at")])
    df = df.reindex(sorted(df.columns), axis=1)
    if not len(df):
        return _h(json.dumps(list(df.columns)).encode(), b"empty")
    # stringify: hash_pandas_object is stable per value but dtype-sensitive for nullable types; text is canonical enough here
    row = pd.util.hash_pandas_object(df.astype(str), index=False).to_numpy()
    row = row.copy()
    row.sort()
    return _h(json.dumps(list(df.columns)).encode(), row.tobytes())


def file_digest(path) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    if p.suffix == ".parquet":
        return frame_digest(pd.read_parquet(p))
    if p.suffix == ".csv":
        return frame_digest(pd.read_csv(p))
    return _h(p.read_bytes())


def stack_version() -> str:
    files = []
    for g in STACK_SRC_GLOBS:
        files += sorted(REPO.glob(g))
    files += [REPO / "scripts" / s for s in STACK_SCRIPTS]
    files += [REPO / s for s in STACK_FILES]
    parts = [str(f.relative_to(REPO)).encode() + b"=" + _h(f.read_bytes() if f.exists() else b"missing").encode() for f in files]
    env = sorted((k, v) for k, v in os.environ.items() if k.startswith("ENGINE_"))
    return _h(*parts, json.dumps(env).encode())


def inputs_hash(season: int, slate: pd.DataFrame, seeds: int, seed_offset: int = 0, ratings_dir: str | None = None,
                extra_files: dict | None = None, repo: Path = REPO, tip_table: str | None = None,
                extra_digests: dict | None = None) -> tuple[str, dict]:
    """(hash, components). `slate` = the frame of games this run will simulate (ids, tips, tip source / placeholder flag)."""
    season = int(season)
    rd = Path(ratings_dir) if ratings_dir else None
    slate_cols = [c for c in ("game_id", "tipoff_utc", "tip_source", "tip_time_is_placeholder", "home_team_id", "away_team_id", "neutral")
                  if c in slate.columns]
    tip_tbl = Path(tip_table) if tip_table else repo / f"data/processed/ingest/tip_times_{season}.parquet"
    tips_d = "missing"
    if tip_tbl.exists():
        tt = pd.read_parquet(tip_tbl)
        tips_d = frame_digest(tt[tt["game_id"].isin(slate["game_id"])] if "game_id" in tt.columns and len(slate) else tt)
    comp = {
        "season": season, "seeds": int(seeds), "seed_offset": int(seed_offset),
        "slate": frame_digest(slate[slate_cols].astype(str)) if len(slate) else "empty",
        "tip_times": tips_d,
        "roster_season": file_digest(repo / f"data/raw/cbbd/rosters/roster_{season}.parquet"),
        "roster_prev_R1": file_digest(repo / f"data/raw/cbbd/rosters/roster_{season - 1}.parquet"),
        "ratings": file_digest(rd / f"own_ratings_{season}.parquet") if rd else "stored_batch_ratings",
        "universe": file_digest(repo / "data/processed/games_universe.parquet"),
        "stack": stack_version(),
    }
    for k, f in (extra_files or {}).items():
        comp[k] = file_digest(f)
    comp.update({k: str(v) for k, v in (extra_digests or {}).items()})
    return _h(json.dumps(comp, sort_keys=True).encode()), comp
