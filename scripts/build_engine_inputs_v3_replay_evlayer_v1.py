"""build_engine_inputs_v3_replay_evlayer_v1.py -- engine inputs v3 with the EVENT-LAYER version as a parameter (lane D).

Versioned sibling of the v3 replay (`build_engine_inputs_v3_replay.py`, lane G; not edited). The only input of the v3
set that reads the possession layer AND is served to a retrained model is the possession_outcome round-2 event team
block (`event_block_F2_2025.npz`, the 16 columns `round2_s1` reads), built per slate by
`live/features.py::po_team_block_r2`, which hard-codes possessions **v2**. This replays exactly that block date by date
with the same slate, the same as-of cutoff (30 min before the first tip) and the same live functions
(`BL.load_slate_from_universe`, `LF.build_ctx`, `LF.po_team_block_r2`, `LF.rating_site_block`, the builder's merge and
no-history fill), with `--event-layer v2|v4` choosing the chance tables. Everything else is copied from `engine_v3`
unchanged: the other blocks either do not read the possession layer or read v1 through their own pinned paths
(`po_team_block` for team_static, the rotation priors), which this sibling does not change (stated scope).

  shard     --event-layer V --shard k --n-shards n --work W    per-date blocks -> W/dates/<D>.npz (resumable)
  assemble  --event-layer V --work W --out DIR [--ratings-dir R]  engine_v3 copy with the replayed event block,
                                                                  plus census.json (cells differing from engine_v3)
PROOF: `--event-layer v2` must reproduce engine_v3's event block bit for bit on every date (assemble asserts it when
V = v2 and writes nothing else).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
BASE = ROOT / "data/processed/models/engine_v3"
TAG = "F2_2025"


def dates_of(games):
    return sorted(pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d").unique())


def block_for_date(D: str, ids: list, layer: str, ratings_dir: str | None):
    import build_engine_inputs_live as BL
    from cbb_sim.live import features as LF
    from cbb_sim.pbp.possessions import possessions_dir
    slate = BL.load_slate_from_universe(D, 2025, only_ids=ids)
    first_tip = pd.to_datetime(slate["tipoff_utc"], utc=True).min()
    as_of = first_tip - pd.Timedelta(minutes=30)
    u = pd.read_parquet(BL.UNIVERSE)
    ctx = LF.build_ctx(slate, as_of, 2025, u, season_start="2024-11-04", strict_finish=False,
                       **({"ratings_dir": str(ratings_dir)} if ratings_dir else {}))
    tg = ctx.team_games()
    rs = LF.rating_site_block(ctx)
    po2 = LF.po_team_block_r2(ctx, poss_dir=ROOT / possessions_dir(layer)).merge(
        rs.drop(columns=["opp_id"]), on=["game_id", "team_id"])
    r2cols = list(BL.B.TEAM_COLS[:16])
    f2 = po2.set_index(["game_id", "team_id"])
    idx2 = pd.MultiIndex.from_frame(tg[["game_id", "team_id"]])
    out = {"game_id": tg["game_id"].to_numpy(dtype="int64"), "is_home": tg["is_home"].to_numpy(dtype=bool),
           "cols": np.array(r2cols)}
    vals = np.zeros((len(tg), 16), dtype=np.float32)
    for j, c in enumerate(r2cols):
        v = f2[c].reindex(idx2).to_numpy(dtype="float64")
        vals[:, j] = np.where(np.isfinite(v), v, 0.0).astype(np.float32)
    out["vals"] = vals
    return out


def shard(a):
    games = pd.read_parquet(BASE / f"games_{TAG}.parquet")
    dd = Path(a.work) / "dates"
    dd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds = dates_of(games)
    if a.max_dates:
        ds = ds[: a.max_dates]
    for i, D in enumerate(ds):
        if i % a.n_shards != a.shard or (dd / f"{D}.npz").exists():
            continue
        ids = games.loc[pd.to_datetime(games["game_date"]) == pd.Timestamp(D), "game_id"].tolist()
        t = time.time()
        o = block_for_date(D, ids, a.event_layer, a.ratings_dir)
        np.savez_compressed(dd / f"{D}.tmp.npz", **o)
        os.replace(dd / f"{D}.tmp.npz", dd / f"{D}.npz")
        print(f"[{time.time() - t0:7.1f}s] {D} {len(ids)} games {time.time() - t:5.1f}s", flush=True)
    return 0


def assemble(a):
    games = pd.read_parquet(BASE / f"games_{TAG}.parquet")
    eb0 = np.load(BASE / f"event_block_{TAG}.npz")["team_block"]
    pos = {int(g): i for i, g in enumerate(games["game_id"])}
    eb = np.zeros_like(eb0)
    filled = np.zeros(eb0.shape[:2], dtype=bool)
    dd = Path(a.work) / "dates"
    missing = [D for D in dates_of(games) if not (dd / f"{D}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} dates not replayed yet, e.g. {missing[:3]}")
    for D in dates_of(games):
        z = np.load(dd / f"{D}.npz")
        rows = np.array([pos[int(g)] for g in z["game_id"]])
        side = np.where(z["is_home"], 0, 1)
        eb[rows, side] = z["vals"]
        filled[rows, side] = True
    assert filled.all(), f"{int((~filled).sum())} team-games unfilled"
    cols = [str(c) for c in np.load(dd / f"{dates_of(games)[0]}.npz")["cols"]]
    diff = eb != eb0
    census = {"event_layer": a.event_layer, "team_games": int(eb0.shape[0] * 2),
              "cells_differing_by_col": {c: int(diff[:, :, j].sum()) for j, c in enumerate(cols)},
              "max_abs_by_col": {c: float(np.abs(eb[:, :, j].astype("f8") - eb0[:, :, j].astype("f8")).max())
                                 for j, c in enumerate(cols)},
              "mean_abs_by_col": {c: float(np.abs(eb[:, :, j].astype("f8") - eb0[:, :, j].astype("f8")).mean())
                                  for j, c in enumerate(cols)},
              "other_blocks": "copied from engine_v3 unchanged (arrays_F2_2025.npz incl. team_static, slots, "
                              "rotation priors; games; names): 0 cells differ by construction"}
    print(json.dumps(census, indent=1))
    if a.event_layer == "v2":
        ok = bool(np.array_equal(eb, eb0))
        census["proof_v2_bit_identical_to_engine_v3"] = ok
        (Path(a.work) / "census.json").write_text(json.dumps(census, indent=1))
        print("PROOF v2 == engine_v3 event block:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: never overwrite")
    out.mkdir(parents=True)
    for f in (f"games_{TAG}.parquet", f"names_{TAG}.json", f"arrays_{TAG}.npz"):
        shutil.copyfile(BASE / f, out / f)
    np.savez_compressed(out / f"event_block_{TAG}.npz", team_block=eb)
    (out / "census.json").write_text(json.dumps(census, indent=1))
    (out / "builder_report.json").write_text(json.dumps({"builder": "build_engine_inputs_v3_replay_evlayer_v1.py",
                                                         "event_layer": a.event_layer, "base": str(BASE),
                                                         "census": census}, indent=1))
    print("wrote", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["shard", "assemble"])
    ap.add_argument("--event-layer", choices=["v2", "v4"], required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-dates", type=int, default=0, help="timing slice only")
    ap.add_argument("--ratings-dir", default=None, help="the replay's own ratings (default: stored batch ratings)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    return shard(a) if a.step == "shard" else assemble(a)


if __name__ == "__main__":
    raise SystemExit(main())
