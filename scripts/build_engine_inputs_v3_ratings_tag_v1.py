"""
build_engine_inputs_v3_ratings_tag_v1.py -- a tagged engine-inputs set whose OWN-RATINGS cells come from a sibling ratings
directory (Lane N, 2026-09-30). Versioned sibling of `build_engine_inputs_v3_tag_v1.py` (not edited; its overlay
builder is imported), which has no ratings parameter.

    .venv/Scripts/python.exe scripts/build_engine_inputs_v3_ratings_tag_v1.py --tag N_C --ratings-dir data/processed/ratings_C_v1

The live path (`cbb_sim.live.features.rating_site_block`) puts the as-of ridge ratings of the slate date into four team
columns, identical in `team_static` and in the round-2 event block (columns 8-11):
    off_rating_off_c = off_c(offence team)   off_rating_def_c = def_c(offence team)
    def_rating_off_c = off_c(defence team)   def_rating_def_c = def_c(defence team)       (missing -> 0.0)
with side 0 = home team on offence, side 1 = away. Tempo columns are NOT touched (arm C changes the efficiency prior only;
its tempo ratings equal R's, checked here).
PARITY FIRST: the same recipe applied to the SERVED ratings (`data/processed/ratings`) must reproduce the base's eight
channels exactly (float32) or the build stops. Output `data/processed/models/engine_v3_<tag>/` (never an existing dir),
with the serving overlay (served artifacts, THIS tag's event block) from the imported builder.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import build_engine_inputs_v3_tag_v1 as TB  # noqa: E402

TAG = TB.TAG
COLS = ["off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c"]


def rating_cells(games: pd.DataFrame, rdir: Path) -> np.ndarray:
    """(G, 2, 4) float32 in COLS order."""
    season = int(games["season"].iloc[0])
    r = pd.read_parquet(rdir / f"own_ratings_{season}.parquet", columns=["as_of_date", "team_id", "off_c", "def_c", "tempo_rel"])
    r["as_of_date"] = pd.to_datetime(r["as_of_date"])
    r = r.set_index(["as_of_date", "team_id"])
    d = pd.to_datetime(games["game_date"]).dt.normalize()
    h = r.reindex(pd.MultiIndex.from_arrays([d, games["home_team_id"].astype("int64")]))
    a = r.reindex(pd.MultiIndex.from_arrays([d, games["away_team_id"].astype("int64")]))
    ho, hd = h["off_c"].to_numpy(), h["def_c"].to_numpy()
    ao, ad = a["off_c"].to_numpy(), a["def_c"].to_numpy()
    out = np.zeros((len(games), 2, 4), dtype=np.float32)
    for side, (oo, od, do, dd) in enumerate(((ho, hd, ao, ad), (ao, ad, ho, hd))):
        for k, v in enumerate((oo, od, do, dd)):
            out[:, side, k] = pd.Series(v).astype("float32").fillna(0.0).to_numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ratings-dir", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path, default=TB.BASE_DEFAULT)
    ap.add_argument("--served-ratings-dir", type=Path, default=ROOT / "data/processed/ratings")
    a = ap.parse_args()
    base = a.base_dir
    out = base.parent / f"engine_v3_{a.tag}"
    if out.exists() or out.resolve() == base.resolve():
        raise SystemExit(f"{out} exists or is the base: never overwrite")
    z = dict(np.load(base / f"arrays_{TAG}.npz"))
    names = json.loads((base / f"names_{TAG}.json").read_text(encoding="utf-8"))
    games = pd.read_parquet(base / f"games_{TAG}.parquet")
    eb = np.load(base / f"event_block_{TAG}.npz")["team_block"]
    tn = names["team_names"]
    j = [tn[c] for c in COLS]
    assert j == [8, 9, 10, 11]
    # ---- parity: the recipe on the served ratings reproduces the base ------------------------------------------
    served = rating_cells(games, a.served_ratings_dir)
    par = {"team_static_max_abs": float(np.abs(served - z["team_static"][:, :, j]).max()),
           "event_block_max_abs": float(np.abs(served - eb[:, :, j]).max())}
    print("parity vs base:", par, flush=True)
    if par["team_static_max_abs"] != 0.0 or par["event_block_max_abs"] != 0.0:
        raise SystemExit("PARITY FAILED: the recipe does not reproduce the base's rating cells")
    # tempo unchanged between served and sibling ratings (arm C touches efficiency only)
    s = int(games["season"].iloc[0])
    t1 = pd.read_parquet(a.served_ratings_dir / f"own_ratings_{s}.parquet", columns=["as_of_date", "team_id", "tempo_rel", "league_tempo_mean"])
    t2 = pd.read_parquet(a.ratings_dir / f"own_ratings_{s}.parquet", columns=["as_of_date", "team_id", "tempo_rel", "league_tempo_mean"])
    mt = t1.merge(t2, on=["as_of_date", "team_id"])
    tempo_diff = float(max(np.abs(mt["tempo_rel_x"] - mt["tempo_rel_y"]).max(), np.abs(mt["league_tempo_mean_x"] - mt["league_tempo_mean_y"]).max()))
    assert len(mt) == len(t1) == len(t2) and tempo_diff < 1e-9, tempo_diff
    # ---- substitute ---------------------------------------------------------------------------------------------
    new = rating_cells(games, a.ratings_dir)
    ts = z["team_static"].copy(); eb2 = eb.copy()
    ts[:, :, j] = new; eb2[:, :, j] = new
    out.mkdir(parents=True)
    arrs = dict(z); arrs["team_static"] = ts
    np.savez_compressed(out / f"arrays_{TAG}.npz", **arrs)
    np.savez_compressed(out / f"event_block_{TAG}.npz", team_block=eb2)
    for f in (f"games_{TAG}.parquet", f"names_{TAG}.json"):
        shutil.copyfile(base / f, out / f)
    ov = TB.build_overlay(out, eb2, TB.SERVED_EVENT, None, None, None)
    changed_ts = [c for c, k in tn.items() if not np.array_equal(ts[:, :, k], z["team_static"][:, :, k])]
    changed_eb = [i for i in range(eb.shape[2]) if not np.array_equal(eb2[:, :, i], eb[:, :, i])]
    d = new - served
    rep = {"tag": a.tag, "base": str(base), "ratings_dir": str(a.ratings_dir), "parity_vs_base": par,
           "tempo_max_abs_diff_served_vs_sibling": tempo_diff,
           "changed_team_static": changed_ts, "changed_event_block_cols": changed_eb,
           "unchanged_arrays": [k for k in z if k != "team_static"],
           "delta_abs_mean_by_col": {c: float(np.abs(d[:, :, k]).mean()) for k, c in enumerate(COLS)},
           "base_sha256": {f: TB.sha(base / f) for f in (f"arrays_{TAG}.npz", f"event_block_{TAG}.npz")},
           "outputs_sha256": {f: TB.sha(out / f) for f in (f"arrays_{TAG}.npz", f"games_{TAG}.parquet", f"names_{TAG}.json", f"event_block_{TAG}.npz")},
           "overlay": ov}
    (out / "builder_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("changed_team_static", "changed_event_block_cols", "delta_abs_mean_by_col")}, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
