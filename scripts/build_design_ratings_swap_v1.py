#!/usr/bin/env python
"""build_design_ratings_swap_v1.py -- re-derive a design's four own-rating columns from another ratings dir.

For the full-retrain chain (ratings C, `docs/models/own_ratings/experiments.md` `ratings_C_v1`). fg_make and
rebound build their designs with `orat.load_ratings(seasons, out_dir=ratings_dir)` +
`orat.join_as_of(..., team_col=off/def, date_col="game_date")`, renamed to
`off_rating_off_c, off_rating_def_c, def_rating_off_c, def_rating_def_c`, missing -> 0.0 (the league mean on
a centred scale). Rebuilding those multi-million-row designs from raw pbp only to change four columns
that are a pure function of (season, team, game_date) is unnecessary: this tool recomputes exactly those
four columns with the same two library calls and writes a NEW design file. Every other column is copied.

PROOF FIRST (`--identity`): run with the SERVED ratings dir; the output must equal the input design
column for column (DataFrame.equals). The chain refuses to run a swap whose identity proof is absent.

    .venv/Scripts/python.exe scripts/build_design_ratings_swap_v1.py --design data/processed/models/fg_make/design_v2_shotshooter.parquet \
        --ratings-dir data/processed/ratings --out <scratch>/fg_identity.parquet --identity
    .venv/Scripts/python.exe scripts/build_design_ratings_swap_v1.py --design <same> --ratings-dir data/processed/ratings_C_v1 \
        --out <root>/fg_design/design_v2_shotshooter_C.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RCOLS = ("off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c")


def swap(design, ratings_dir: str, off_col: str, def_col: str):
    import numpy as np
    import pandas as pd
    from cbb_sim.ratings import own_ratings as orat

    key = design[["season", "game_date", off_col, def_col]].drop_duplicates().reset_index(drop=True)
    ratings = orat.load_ratings(sorted(int(s) for s in key["season"].unique()), out_dir=ratings_dir)
    k = orat.join_as_of(key, ratings, team_col=off_col, date_col="game_date", suffix="__offteam",
                        cols=("off_c", "def_c"))
    k = orat.join_as_of(k, ratings, team_col=def_col, date_col="game_date", suffix="__defteam",
                        cols=("off_c", "def_c"))
    k = k.rename(columns={"off_c__offteam": "off_rating_off_c", "def_c__offteam": "off_rating_def_c",
                          "off_c__defteam": "def_rating_off_c", "def_c__defteam": "def_rating_def_c"})
    out = design.drop(columns=list(RCOLS)).merge(
        k[["season", "game_date", off_col, def_col, *RCOLS]],
        on=["season", "game_date", off_col, def_col], how="left", validate="many_to_one")
    n_missing = {c: int(out[c].isna().sum()) for c in RCOLS}
    for c in RCOLS:
        out[c] = out[c].fillna(0.0).astype(design[c].dtype)
    out = out[list(design.columns)]
    out.index = design.index
    return out, {"n_missing_filled_0": n_missing, "n_keys": int(len(k))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--ratings-dir", required=True)
    ap.add_argument("--off-col", default="off_team_id")
    ap.add_argument("--def-col", default="def_team_id")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--identity", action="store_true")
    a = ap.parse_args()
    if a.out.exists() or a.out.resolve() == a.design.resolve():
        raise SystemExit(f"{a.out} exists or is the input: never overwrite")
    import pandas as pd
    t0 = time.time()
    d = pd.read_parquet(a.design)
    new, rep = swap(d, a.ratings_dir, a.off_col, a.def_col)
    rep.update({"design": str(a.design), "ratings_dir": str(a.ratings_dir), "rows": int(len(new))})
    chg = {c: float((new[c].to_numpy() != d[c].to_numpy()).mean()) for c in RCOLS}
    rep["share_rows_changed"] = chg
    rep["max_abs_change"] = {c: float((new[c].astype("float64") - d[c].astype("float64")).abs().max()) for c in RCOLS}
    if a.identity:
        rep["identity_equal"] = bool(new.equals(d))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if not a.identity:
        tmp = a.out.with_suffix(".parquet.tmp")
        new.to_parquet(tmp, index=False)
        tmp.replace(a.out)
    rep["seconds"] = round(time.time() - t0, 1)
    rp = a.out.parent / (a.out.stem + "_swap_report.json")
    rp.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0 if rep.get("identity_equal", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
