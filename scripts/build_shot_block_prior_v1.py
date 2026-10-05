#!/usr/bin/env python
"""
build_shot_block_prior_v1.py -- the K2_Ocell per-season anchor prior (`prior_<S-1>_end`), generalised by served season.

How the existing season input is built (`build_engine_shot_block_lut_v1.py`): for each missed-FG type t in rim / jump2 / three,
`SA.anchor_O(...)` takes the season-S day-0 level = the PRIOR season's realised end-of-season block rate of type t
(`Lend[S-1]`, = blocked / missed attempts over the whole prior season; stored as meta `K2_Ocell_<t>.prior_2024_end` for S = 2025);
`Lbar` is the pooled TRAIN-season level, a fitted constant of the F2 model (read from the backtest meta, never recomputed).
Served anchor = link(as-of level) - link(Lbar), the as-of level falling back to this prior on the first day.

    .venv/Scripts/python.exe scripts/build_shot_block_prior_v1.py --season 2027      # needs 2026 (2025-26): SEALED -> SealedSeasonError
    .venv/Scripts/python.exe scripts/build_shot_block_prior_v1.py --validate         # S=2025 from season 2024 vs stored meta (exact)

Output (`--out`, default data/processed/models/engine/shot_block_prior_<season>_v1.parquet), one row per type:
    season (served), src_season, miss_type, prior_end_level, n_miss, n_blocked, Lbar (from the F2 meta)
Rows are resolved first-chance-agnostic missed FGA (rebound outcome in OREB/DREB/DEAD), the rebound round-3 design population that
`train_shot_block_v1.build_design` takes its rows from. Events for the source season: `rebound/events_v1.parquet` when it holds the
season, else rebuilt from pbp with `RB.build_rebound_events([src], version="v1")` (the design builder's own call).

SEAL: the source season passes `assert_not_sealed` BEFORE any read (so --season 2027 stops unless CBB_UNSEAL=1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

ENGINE_DIR = ROOT / "data/processed/models/engine"
META = ENGINE_DIR / "shot_block_lut_F2_2025.meta.json"
EVENTS = ROOT / "data/processed/models/rebound/events_v1.parquet"
TYPES = ("rim", "jump2", "three")


def source_events(src: int, source: str = "auto") -> pd.DataFrame:
    if source in ("auto", "stored"):
        e = pd.read_parquet(EVENTS, columns=["season", "miss_type", "blocked", "outcome"])
        e = e[e["season"] == src]
        if len(e) or source == "stored":
            return e
    from cbb_sim.models import event_stream as ES
    from cbb_sim.models import rebound as RB
    ev = RB.build_rebound_events([src], universe=ES.load_universe(), version="v1")
    return ev[["season", "miss_type", "blocked", "outcome"]]


def build_prior(season: int, ev_source: str = "auto") -> pd.DataFrame:
    src = int(season) - 1
    assert_not_sealed(src, context=f"shot_block prior {season}: needs season {src} blocked-miss events")
    from cbb_sim.models import rebound as RB
    e = source_events(src, ev_source)
    e = e[e["outcome"].isin(RB.CLASSES) & e["miss_type"].isin(TYPES)]
    meta = json.loads(META.read_text(encoding="utf-8"))
    rows = []
    for t in TYPES:
        x = e[e["miss_type"] == t]
        n, b = float(len(x)), float(x["blocked"].astype("float64").sum())
        rows.append({"season": int(season), "src_season": src, "miss_type": t, "prior_end_level": b / n, "n_miss": int(n),
                     "n_blocked": int(b), "Lbar": float(meta[f"K2_Ocell_{t}"]["Lbar"])})
    return pd.DataFrame(rows)


def validate(ev_source: str = "auto") -> dict:
    """Season 2025 inputs from season 2024 vs the stored meta `prior_2024_end` (what the served 2025 table was built with)."""
    meta = json.loads(META.read_text(encoding="utf-8"))
    new = build_prior(2025, ev_source)
    rep = {}
    for r in new.itertuples():
        old = float(meta[f"K2_Ocell_{r.miss_type}"]["prior_2024_end"])
        rep[r.miss_type] = {"new": r.prior_end_level, "stored": old, "abs_diff": abs(r.prior_end_level - old), "bit_identical": r.prior_end_level == old}
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, help="served season; reads season-1 events only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--events-source", choices=("auto", "stored", "rebuild"), default="auto")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args(argv)
    if a.validate:
        rep = validate("stored" if a.events_source == "auto" else a.events_source)
        print(json.dumps(rep, indent=1))
        return 0 if all(r["abs_diff"] < 1e-12 for r in rep.values()) else 1
    if a.season is None:
        ap.error("--season or --validate required")
    t = build_prior(a.season, a.events_source)
    out = Path(a.out) if a.out else ENGINE_DIR / f"shot_block_prior_{a.season}_v1.parquet"
    t.to_parquet(out, index=False)
    print(f"wrote {out}\n{t.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
