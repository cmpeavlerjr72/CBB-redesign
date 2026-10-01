"""build_engine_inputs_v3_replay_fold_v1.py -- engine inputs v3 (live replay per date) for ANY fold, fold 1 first.

Versioned sibling of `build_engine_inputs_v3_replay.py` (fold 2 only, NOT edited; lane D 2026-10-01). Same recipe:
for each game date of the fold's v2 backtest universe, run the live builder with the as-of cutoff 30 min before
the date's first tip (`created_at` = that cutoff), then stitch the per-date arrays back into v2 GAME ORDER and write
`{games,arrays,names}_{fold}_{season}.*` plus `event_block_{fold}_{season}.npz` (the round-2 event team block).

Differences from the fold-2 script, all parameters, no recipe change:
  * `--fold/--season`; the v2 universe and the rule-constant template are read from `--v2-dir`
    (fold 1: `data/processed/models/engine_f1`, built by `build_engine_inputs.py --fold F1 --season 2024
    --out-dir data/processed/models/engine_f1 --version v1` then `--version v2`);
  * `season_start` = the first game date of the fold's universe (the fold-2 script hard-codes 2024-11-04);
  * per-date staging goes to `--stage-dir` and per-date reports to `--report-dir` (new dirs; the fold-2 staging
    `engine_live/` is never written);
  * the per-date report is a plain cell census vs v2 (cells differing per array), not the fold-2 cause classifier
    (which is keyed to the fold-2 backtest's own artifacts).

    .venv/Scripts/python.exe scripts/build_engine_inputs_v3_replay_fold_v1.py shard --shard 0 --n-shards 4
    .venv/Scripts/python.exe scripts/build_engine_inputs_v3_replay_fold_v1.py assemble
Defaults are fold 1 (F1 / 2024). Output default `data/processed/models/engine_v3_f1` (HF key proposal:
`engine_inputs_v3_f1` -> that dir).
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_engine_inputs_live as BL  # noqa: E402


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shard", "assemble"])
    ap.add_argument("--fold", default="F1")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--v2-dir", default="data/processed/models/engine_f1")
    ap.add_argument("--stage-dir", default="data/processed/models/engine_live_f1")
    ap.add_argument("--report-dir", default="results/engine_v3_replay_f1/dates")
    ap.add_argument("--out", default="data/processed/models/engine_v3_f1")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-dates", type=int, default=0, help="timing slice: only the first N dates of this shard")
    return ap.parse_args()


def load_v2(a):
    d = ROOT / a.v2_dir
    tag = f"{a.fold}_{a.season}_v2"
    gb = pd.read_parquet(d / f"games_{tag}.parquet")
    zb = dict(np.load(d / f"arrays_{tag}.npz"))
    nb = json.loads((d / f"names_{tag}.json").read_text(encoding="utf-8"))
    return gb, zb, nb


def dates_of(gb):
    return sorted(pd.to_datetime(gb["game_date"]).dt.strftime("%Y-%m-%d").unique())


def run_date(a, D, gb, zb, season_start):
    t0 = time.time()
    sel = gb[pd.to_datetime(gb["game_date"]) == pd.Timestamp(D)]
    ids = sel["game_id"].tolist()
    slate = BL.load_slate_from_universe(D, a.season, only_ids=ids)
    first_tip = pd.to_datetime(slate["tipoff_utc"], utc=True).min()
    as_of = first_tip - pd.Timedelta(minutes=30)
    inp, diag = BL.build_live(slate, as_of, a.season, a.fold, created_at=as_of, season_start=season_start, t0=t0,
                              strict_finish=False)
    tag = f"LIVE_{a.fold}_{a.season}_{D}"
    sd = ROOT / a.stage_dir
    inp.save(sd, tag)
    np.savez_compressed(sd / f"event_block_{tag}.npz", team_block=inp.event_block,
                        cols=np.array(list(BL.B.TEAM_COLS[:16])))
    pos = {int(g): i for i, g in enumerate(gb["game_id"])}
    bi = np.array([pos[int(g)] for g in inp.games["game_id"]])
    z = np.load(sd / f"arrays_{tag}.npz")
    census = {}
    for k in z.files:
        if k in zb and zb[k].shape[1:] == z[k].shape[1:]:
            b = zb[k][bi]
            census[k] = {"cells": int(b.size), "n_diff": int((~np.isclose(z[k], b, equal_nan=True)).sum())}
    return {"date": D, "n_games": int(len(ids)), "n_built": int(len(inp.games)), "as_of": str(as_of),
            "seconds": round(time.time() - t0, 1), "census_vs_v2": census}


def shard(a):
    BL.ENGINE_DIR = ROOT / a.v2_dir          # the fold's rule-constant template (names_{fold}_{season}_v2.json)
    gb, zb, nb = load_v2(a)
    season_start = pd.to_datetime(gb["game_date"]).min().strftime("%Y-%m-%d")
    rd = ROOT / a.report_dir
    rd.mkdir(parents=True, exist_ok=True)
    (ROOT / a.stage_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mine = [D for i, D in enumerate(dates_of(gb)) if i % a.n_shards == a.shard]
    if a.max_dates:
        mine = mine[:a.max_dates]
    for D in mine:
        if (rd / f"{D}.json").exists():
            continue
        try:
            r = run_date(a, D, gb, zb, season_start)
            (rd / f"{D}.json").write_text(json.dumps(r, default=str), encoding="utf-8")
            print(f"[{time.time()-t0:7.0f}s] {D} games {r['n_games']} built {r['n_built']} ({r['seconds']} s)", flush=True)
        except Exception:
            (rd / f"{D}.err").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{time.time()-t0:7.0f}s] {D} FAILED", flush=True)


def assemble(a):
    gb, zb, nb = load_v2(a)
    sd = ROOT / a.stage_dir
    pos = {int(g): i for i, g in enumerate(gb["game_id"])}
    arrs = {k: np.zeros_like(v) for k, v in zb.items()}
    ev = np.zeros((len(gb), 2, 16), dtype=np.float32)
    filled = np.zeros(len(gb), bool)
    created = [None] * len(gb)
    for D in dates_of(gb):
        tag = f"LIVE_{a.fold}_{a.season}_{D}"
        g = pd.read_parquet(sd / f"games_{tag}.parquet")
        z = np.load(sd / f"arrays_{tag}.npz")
        e = np.load(sd / f"event_block_{tag}.npz")["team_block"]
        idx = np.array([pos[int(x)] for x in g["game_id"]])
        for k in arrs:
            arrs[k][idx] = z[k]
        ev[idx] = e
        filled[idx] = True
        for j, i in enumerate(idx):
            created[i] = g["created_at"].iloc[j]
    assert filled.all(), f"{int((~filled).sum())} games unfilled"
    names = dict(nb)
    meta = dict(nb["meta"])
    meta.update(inputs_version="v3-replay", fold=a.fold, season=a.season,
                note="built by replaying the live path per date (build_engine_inputs_v3_replay_fold_v1.py)")
    names["meta"] = meta
    games = gb.copy()
    games["created_at"] = pd.to_datetime(created, utc=True)
    assert (games["created_at"] < pd.to_datetime(games["tipoff_utc"], utc=True)).all() if "tipoff_utc" in games else True
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.fold}_{a.season}"
    games.to_parquet(out / f"games_{tag}.parquet", index=False)
    np.savez_compressed(out / f"arrays_{tag}.npz", **arrs)
    (out / f"names_{tag}.json").write_text(json.dumps(names, indent=1, default=str), encoding="utf-8")
    np.savez_compressed(out / f"event_block_{tag}.npz", team_block=ev)
    print("wrote", out, len(games), "games")


if __name__ == "__main__":
    a = args()
    shard(a) if a.cmd == "shard" else assemble(a)
