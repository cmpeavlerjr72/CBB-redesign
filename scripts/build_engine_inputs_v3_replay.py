"""
build_engine_inputs_v3_replay.py -- fold-2 engine inputs v3 by REPLAYING the live path date by date.

    .venv/Scripts/python.exe scripts/build_engine_inputs_v3_replay.py shard --shard 0 --n-shards 2
    .venv/Scripts/python.exe scripts/build_engine_inputs_v3_replay.py assemble

`shard`: for each fold-2 game date (dates of `games_F2_2025_v2.parquet`), run the live builder with the
as-of cutoff 30 min before the date's first tip (`created_at` = that cutoff, replay) through
`diag_live_parity_v1.run_date`, which also classifies every cell that differs from v2 by cause. Per-date
staging inputs go to `data/processed/models/engine_live/LIVE_F2_2025_{D}.*`, per-date reports to
`results/engine_v3_replay/dates/{D}.json`.
`assemble`: stitch the per-date arrays into full-universe arrays in v2 GAME ORDER and write the versioned
sibling `data/processed/models/engine_v3/{games,arrays,names}_F2_2025.*` plus `event_block_F2_2025.npz`
(the round-2 event team block for the same rows). v2 files are never touched.
hf_sync_data.py bulk key to register (not synced): `engine_inputs_v3` -> `data/processed/models/engine_v3`.
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import diag_live_parity_v1 as P
ED = ROOT / "data/processed/models/engine"; LD = ROOT / "data/processed/models/engine_live"
OUT = ROOT / "data/processed/models/engine_v3"; RES = ROOT / "results/engine_v3_replay/dates"


def dates_of(gb):
    return sorted(pd.to_datetime(gb["game_date"]).dt.strftime("%Y-%m-%d").unique())


def shard(k, n):
    RES.mkdir(parents=True, exist_ok=True)
    gb = pd.read_parquet(ED / "games_F2_2025_v2.parquet")
    zb = dict(np.load(ED / "arrays_F2_2025_v2.npz"))
    nb = json.loads((ED / "names_F2_2025_v2.json").read_text(encoding="utf-8"))
    t0 = time.time()
    for i, D in enumerate(dates_of(gb)):
        if i % n != k or (RES / f"{D}.json").exists():
            continue
        try:
            r = P.run_date(D, zb, gb, nb, lambda m: None)
            (RES / f"{D}.json").write_text(json.dumps(r, default=str), encoding="utf-8")
            print(f"[{time.time()-t0:7.0f}s] {D} games {r['n_games']} unexplained {len(r['unexplained'])}", flush=True)
        except Exception:
            (RES / f"{D}.err").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{time.time()-t0:7.0f}s] {D} FAILED", flush=True)


def assemble():
    gb = pd.read_parquet(ED / "games_F2_2025_v2.parquet")
    zb = dict(np.load(ED / "arrays_F2_2025_v2.npz"))
    nb = json.loads((ED / "names_F2_2025_v2.json").read_text(encoding="utf-8"))
    pos = {int(g): i for i, g in enumerate(gb["game_id"])}
    arrs = {k: np.zeros_like(v) for k, v in zb.items()}
    ev = np.zeros((len(gb), 2, 16), dtype=np.float32)
    filled = np.zeros(len(gb), bool); created = [None] * len(gb)
    rules = None
    for D in dates_of(gb):
        tag = f"LIVE_F2_2025_{D}"
        g = pd.read_parquet(LD / f"games_{tag}.parquet"); z = np.load(LD / f"arrays_{tag}.npz")
        e = np.load(LD / f"event_block_{tag}.npz")["team_block"]
        idx = np.array([pos[int(x)] for x in g["game_id"]])
        for k in arrs:
            arrs[k][idx] = z[k]
        ev[idx] = e; filled[idx] = True
        for j, i in enumerate(idx):
            created[i] = g["created_at"].iloc[j]
    assert filled.all(), f"{int((~filled).sum())} games unfilled"
    names = dict(nb); meta = dict(nb["meta"])
    meta.update(inputs_version="v3-replay", note="built by replaying the live path per date; see docs/tests/engine_inputs_v3_replay_2026-09-30.md")
    names["meta"] = meta
    games = gb.copy(); games["created_at"] = pd.to_datetime(created, utc=True)
    OUT.mkdir(parents=True, exist_ok=True)
    games.to_parquet(OUT / "games_F2_2025.parquet", index=False)
    np.savez_compressed(OUT / "arrays_F2_2025.npz", **arrs)
    (OUT / "names_F2_2025.json").write_text(json.dumps(names, indent=1, default=str), encoding="utf-8")
    np.savez_compressed(OUT / "event_block_F2_2025.npz", team_block=ev)
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["shard", "assemble"])
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n-shards", type=int, default=2)
    a = ap.parse_args()
    shard(a.shard, a.n_shards) if a.cmd == "shard" else assemble()
