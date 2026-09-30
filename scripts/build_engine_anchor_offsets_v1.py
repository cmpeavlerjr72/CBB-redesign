"""build_engine_anchor_offsets_v1.py -- per-game season-drift anchor `O` offsets for the engine (lane M, 2026-09-30).

Writes `<input-dir>/anchor_offsets_<fold>_<season>.npz` (+ `.json`) next to a tagged engine-input directory
(e.g. `engine_v3_<tag>` from `build_engine_inputs_v3_tag_v1.py`, which is not edited). The engine reads it only under
`ENGINE_SEASON_ANCHOR=<that .npz>` (`src/cbb_sim/engine/season_anchor_serving.py`); nothing is computed in the sim loop.

    game_ids  (G,)    the input dir's own game order
    po_first  (G, 6)  possession_outcome `first`: log L_c(t) - log Lbar_c per class (families contains po)
    rb_oreb   (G,)    rebound: logit L(t) - logit Lbar, L = live OREB share    (families contains rb)

The level is `cbb_sim.season_anchor.anchor_O` ITSELF, on the SAME rows the anchored trainer uses (its own definition,
not the E3 table's `L`):
    po  `train_possession_outcome_s1_par_anchor_v1.add_anchor_columns`: the round-2 design's `first` rows of the fold's
        train seasons + the test season, class shares, kind multi;
    rb  `train_rebound_v3_par_anchor_v1.install_anchor_O`: `RB.fold_slices(design, fold)` train + test rows, OREB share
        of live misses, kind binary, train seasons = the train slice's seasons.
Each engine game is added as a zero-weight row (num = 0, den = 0) on its own date, so its level is exactly the
module's as-of level for that date (strictly earlier dates of the test season; day 0 = previous season's end level),
and no design total moves. The anchor depends only on (season, date, y) of those rows, so the team-rate table does not
enter (the trainers apply it without dropping rows under `--team-rate-missing raise`).

    .venv/Scripts/python.exe scripts/build_engine_anchor_offsets_v1.py \
        --input-dir data/processed/models/engine_v3_TO --families po,rb
    # proof (b) only: --zeros writes all-zero arrays to anchor_offsets_<fold>_<season>_ZEROS.npz
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim import season_anchor as SA  # noqa: E402
from cbb_sim.engine.inputs import resolve_tag  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402

PO_DESIGN = ROOT / "data/processed/models/possession_outcome/round2/design.parquet"
RB_DESIGN = ROOT / "data/processed/models/rebound/round3/design_round3.parquet"
FOLD_TRAIN = {"F1": [2022, 2023], "F2": [2022, 2023, 2024]}
FOLD_TEST = {"F1": 2024, "F2": 2025}


def engine_games(input_dir: Path, fold: str, season: int) -> pd.DataFrame:
    disk_tag, _ = resolve_tag(input_dir, f"{fold}_{season}")
    g = pd.read_parquet(input_dir / f"games_{disk_tag}.parquet")
    date = pd.to_datetime(g["game_date"] if "game_date" in g else g["date"]).dt.normalize()
    return pd.DataFrame({"game_id": g["game_id"].to_numpy().astype(np.int64), "date": date.to_numpy()})


def _with_engine_rows(season, date, num, den, games, test_season):
    G = len(games)
    s = np.concatenate([np.asarray(season).astype("int64"), np.full(G, test_season, dtype="int64")])
    d = np.concatenate([pd.to_datetime(date).values.astype("datetime64[ns]"),
                        games["date"].to_numpy().astype("datetime64[ns]")])
    n = np.vstack([num, np.zeros((G, num.shape[1]))])
    e = np.concatenate([den, np.zeros(G)])
    return s, d, n, e


def po_levels(design: Path, games: pd.DataFrame, fold: str, season: int):
    d = pd.read_parquet(design, columns=["season", "game_date", "population", "y"])
    m = (d["population"] == "first") & d["season"].isin([*FOLD_TRAIN[fold], season])
    sub = d.loc[m]
    num, den = SA.po_inputs(sub, len(PO.CLASSES))
    s, dd, n, e = _with_engine_rows(sub["season"].to_numpy(), sub["game_date"].to_numpy(), num, den, games, season)
    a = SA.anchor_O(s, dd, n, e, FOLD_TRAIN[fold], "multi")
    G = len(games)
    return a.offset()[-G:], a.levels[-G:], a.meta, int(len(sub))


def rb_levels(design: Path, games: pd.DataFrame, fold: str, season: int):
    d = pd.read_parquet(design)
    tr, te = RB.fold_slices(d, fold)
    if sorted(int(x) for x in te["season"].unique()) != [season]:
        raise SystemExit(f"rebound fold {fold} test slice is not season {season}")
    both = pd.concat([tr[["season", "game_date", "y"]], te[["season", "game_date", "y"]]], ignore_index=True)
    num, den = SA.rebound_inputs(both, RB.CLASS_INDEX["OREB"], RB.CLASS_INDEX["DEAD"])
    train_seasons = sorted(int(x) for x in tr["season"].unique())
    s, dd, n, e = _with_engine_rows(both["season"].to_numpy(), both["game_date"].to_numpy(), num, den, games, season)
    a = SA.anchor_O(s, dd, n, e, train_seasons, "binary")
    G = len(games)
    return a.offset()[-G:, 0], a.levels[-G:, 0], a.meta, int(len(both))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--fold", default="F2", choices=sorted(FOLD_TRAIN))
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--families", default="po,rb", help="comma list of po,rb")
    ap.add_argument("--po-design", type=Path, default=PO_DESIGN)
    ap.add_argument("--rb-design", type=Path, default=RB_DESIGN)
    ap.add_argument("--zeros", action="store_true", help="PROOF ONLY: all-zero offsets (bit-identity to mode off)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    season = a.season or FOLD_TEST[a.fold]
    fams = [f for f in a.families.split(",") if f]
    if not fams or set(fams) - {"po", "rb"}:
        raise SystemExit("--families must be a comma list of po,rb")
    out = a.out or a.input_dir / f"anchor_offsets_{a.fold}_{season}{'_ZEROS' if a.zeros else ''}.npz"
    if out.exists() or out.with_suffix(".json").exists():
        raise SystemExit(f"{out} exists: never overwrite (write a versioned sibling with --out)")
    games = engine_games(a.input_dir, a.fold, season)
    arrays = {"game_ids": games["game_id"].to_numpy()}
    meta = {"builder": "build_engine_anchor_offsets_v1", "created_at": dt.datetime.now().astimezone().isoformat(),
            "fold": a.fold, "season": season, "families": fams, "zeros": bool(a.zeros),
            "input_dir": str(a.input_dir), "n_games": int(len(games))}
    try:
        meta["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                     cwd=ROOT, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        meta["git"] = "unknown"
    dates = games["date"].dt.date.astype(str).to_numpy()
    if "po" in fams:
        off, lev, m, nrows = po_levels(a.po_design, games, a.fold, season)
        arrays["po_first"] = np.zeros_like(off) if a.zeros else off
        u = pd.DataFrame(lev, columns=list(PO.CLASSES)).assign(date=dates).drop_duplicates("date").sort_values("date")
        meta["po"] = {"design": str(a.po_design), "rows": nrows, "kind": "multi", "classes": list(PO.CLASSES),
                      "Lbar": m["Lbar"], "prior_by_season": {str(k): v for k, v in m["prior_by_season"].items()},
                      "level_by_date": {r["date"]: [float(r[c]) for c in PO.CLASSES] for _, r in u.iterrows()}}
    if "rb" in fams:
        off, lev, m, nrows = rb_levels(a.rb_design, games, a.fold, season)
        arrays["rb_oreb"] = np.zeros_like(off) if a.zeros else off
        u = pd.DataFrame({"date": dates, "L": lev}).drop_duplicates("date").sort_values("date")
        meta["rb"] = {"design": str(a.rb_design), "rows": nrows, "kind": "binary", "target": "OREB share of live misses",
                      "Lbar": m["Lbar"], "prior_by_season": {str(k): v for k, v in m["prior_by_season"].items()},
                      "level_by_date": dict(zip(u["date"], u["L"].astype(float)))}
    np.savez(out, **arrays)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("fold", "season", "families", "zeros", "n_games")}))
    for f in fams:
        v = arrays["po_first" if f == "po" else "rb_oreb"]
        print(f"  {f}: shape {v.shape}, offset range [{v.min():+.5f}, {v.max():+.5f}]")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
