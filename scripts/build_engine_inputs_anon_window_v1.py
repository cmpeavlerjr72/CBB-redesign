#!/usr/bin/env python
"""
build_engine_inputs_anon_window_v1.py -- player-layer day-1 sizing (Phase 1, 2026-10-05): a sibling of the served v3 inputs in
which every game of a season's OPENING WINDOW (first 14 days) carries the ANONYMOUS day-1 player state, i.e. exactly what
`build_live` produces for a team with no earlier appearance this season: league-mean rotation fallback, no named slots, no
shooter blocks, usage / rebound no-history fallbacks. Games outside the window are byte-identical to the served v3 inputs.

How: per window date, `build_live` is re-run with `LF.rotation_priors` returning {} (so the candidate pool is empty, as on
opening day of 2026-27); everything else (team blocks, as-of cutoff 30 min before first tip, fold template) is the v3 replay
recipe. A served-parity check rebuilds 2 window dates UNPATCHED and requires equality with the base v3 arrays (proves the
base is what this tree's live path builds). The shot-block table (slot-keyed) is copied from the served one with the window
games' `shooter`/`known` set to the anonymous values (0, 0: what `build_table` gives for negative slot ids) and registered in
`meta["shot_block_lut"]`.

    python scripts/build_engine_inputs_anon_window_v1.py build --fold F2      # then --fold F1
    python scripts/build_engine_inputs_anon_window_v1.py assemble --fold F2
Outputs: data/processed/models/engine_v3_anonwin{,_f1}/ (+ window_ids_<fold>.parquet). Nothing served is written.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import build_engine_inputs_live as BL  # noqa: E402
from cbb_sim.live import features as LF  # noqa: E402

M = ROOT / "data/processed/models"
CFG = {
    "F2": dict(season=2025, base=M / "engine_v3", template=M / "engine", out=M / "engine_v3_anonwin",
               lut=M / "engine/shot_block_K2_Ocell_F2_2025.npz"),
    "F1": dict(season=2024, base=M / "engine_v3_f1", template=M / "engine_f1", out=M / "engine_v3_anonwin_f1",
               lut=M / "fold1_v1/loop/shot_block_K2_Ocell_F1_2024.npz"),
}
WINDOW_DAYS = 14
PLAYER_ARRAYS = ("roster_cbbd", "roster_espn", "roster_valid", "rot_share", "rot_srank", "rot_start", "rot_fpm",
                 "rot_pavail", "slot_static", "usage_rate", "reb_rate")


def window(fold: str):
    c = CFG[fold]
    g = pd.read_parquet(c["base"] / f"games_{fold}_{c['season']}.parquet")
    g["game_date"] = pd.to_datetime(g["game_date"])
    d0 = g["game_date"].min()
    w = g[g["game_date"] < d0 + pd.Timedelta(days=WINDOW_DAYS)]
    return g, w, d0.strftime("%Y-%m-%d")


def stage(fold: str) -> Path:
    return CFG[fold]["out"] / "_stage"


def build_date(fold: str, D: str, ids, season_start: str, anon: bool):
    c = CFG[fold]
    BL.ENGINE_DIR = c["template"]
    slate = BL.load_slate_from_universe(D, c["season"], only_ids=ids)
    as_of = pd.to_datetime(slate["tipoff_utc"], utc=True).min() - pd.Timedelta(minutes=30)
    orig = LF.rotation_priors
    if anon:
        LF.rotation_priors = lambda ctx, fit, min_prior_games=1: {}
    try:
        inp, diag = BL.build_live(slate, as_of, c["season"], fold, created_at=as_of, season_start=season_start,
                                  strict_finish=False)
    finally:
        LF.rotation_priors = orig
    return inp, diag


def cmd_build(fold: str) -> None:
    c = CFG[fold]
    g, w, s0 = window(fold)
    sd = stage(fold)
    sd.mkdir(parents=True, exist_ok=True)
    zb = dict(np.load(c["base"] / f"arrays_{fold}_{c['season']}.npz"))
    pos = {int(x): i for i, x in enumerate(g["game_id"])}
    dates = sorted(w["game_date"].dt.strftime("%Y-%m-%d").unique())
    parity = {}
    t0 = time.time()
    for k, D in enumerate(dates):
        ids = w.loc[w["game_date"] == pd.Timestamp(D), "game_id"].tolist()
        tag = f"ANON_{fold}_{D}"
        if not (sd / f"games_{tag}.parquet").exists():
            inp, diag = build_date(fold, D, ids, s0, anon=True)
            inp.save(sd, tag)
            print(f"[{time.time()-t0:6.0f}s] {D} anon built ({len(inp.games)} games, fallback team-games "
                  f"{diag['rotation_fallback_team_games']})", flush=True)
        if k in (0, len(dates) // 2):          # served-parity check: unpatched rebuild == base v3
            inp, _ = build_date(fold, D, ids, s0, anon=False)
            idx = np.array([pos[int(x)] for x in inp.games["game_id"]])
            z = {a: getattr(inp, a) for a in zb if hasattr(inp, a)}
            parity[D] = {a: bool(np.array_equal(np.asarray(v), zb[a][idx], equal_nan=True)
                                 if np.asarray(v).dtype.kind == "f" else np.array_equal(np.asarray(v), zb[a][idx]))
                         for a, v in z.items()}
            print(f"[{time.time()-t0:6.0f}s] {D} served parity: {parity[D]}", flush=True)
    (sd / "parity.json").write_text(json.dumps(parity, indent=1), encoding="utf-8")


def cmd_assemble(fold: str) -> None:
    c = CFG[fold]
    tag = f"{fold}_{c['season']}"
    g, w, _ = window(fold)
    sd, out = stage(fold), c["out"]
    zb = dict(np.load(c["base"] / f"arrays_{tag}.npz"))
    arrs = {k: v.copy() for k, v in zb.items()}
    pos = {int(x): i for i, x in enumerate(g["game_id"])}
    filled = np.zeros(len(g), bool)
    same = {}
    for D in sorted(w["game_date"].dt.strftime("%Y-%m-%d").unique()):
        gg = pd.read_parquet(sd / f"games_ANON_{fold}_{D}.parquet")
        z = np.load(sd / f"arrays_ANON_{fold}_{D}.npz")
        idx = np.array([pos[int(x)] for x in gg["game_id"]])
        for k in zb:
            if k in PLAYER_ARRAYS:
                arrs[k][idx] = z[k]
            else:
                ok = np.array_equal(z[k], zb[k][idx], equal_nan=True) if z[k].dtype.kind == "f" else np.array_equal(z[k], zb[k][idx])
                same[k] = same.get(k, True) and bool(ok)
        filled[idx] = True
    win = g["game_id"].isin(w["game_id"]).to_numpy()
    assert (filled == win).all(), "window games not all rebuilt"
    assert (arrs["roster_cbbd"][win] < 0).all(), "a window slot is still named"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"arrays_{tag}.npz", **arrs)
    shutil.copy2(c["base"] / f"games_{tag}.parquet", out / f"games_{tag}.parquet")
    shutil.copy2(c["base"] / f"event_block_{tag}.npz", out / f"event_block_{tag}.npz")
    lut = dict(np.load(c["lut"]))
    assert np.array_equal(lut["game_id"], g["game_id"].to_numpy())
    lut["shooter"] = lut["shooter"].copy(); lut["known"] = lut["known"].copy()
    lut["shooter"][win] = 0; lut["known"][win] = 0
    lp = out / f"shot_block_K2_Ocell_anonwin_{tag}.npz"
    np.savez_compressed(lp, **lut)
    names = json.loads((c["base"] / f"names_{tag}.json").read_text(encoding="utf-8"))
    rel = lp.relative_to(ROOT).as_posix()
    names.setdefault("meta", {})["shot_block_lut"] = {"K2_Ocell": rel, "K2_Ocell_v3in": rel}
    names["meta"]["anon_window"] = {"builder": "scripts/build_engine_inputs_anon_window_v1.py", "days": WINDOW_DAYS,
                                    "n_games": int(win.sum()), "base": str(c["base"].relative_to(ROOT).as_posix()),
                                    "base_lut": str(c["lut"].relative_to(ROOT).as_posix())}
    (out / f"names_{tag}.json").write_text(json.dumps(names, indent=1, default=str), encoding="utf-8")
    w[["game_id", "game_date"]].to_parquet(out / f"window_ids_{fold}.parquet", index=False)
    rep = {"non_player_arrays_equal_base_on_window": same, "n_window_games": int(win.sum()),
           "parity": json.loads((sd / "parity.json").read_text(encoding="utf-8"))}
    (out / "assemble_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "assemble"])
    ap.add_argument("--fold", choices=list(CFG), default="F2")
    a = ap.parse_args()
    cmd_build(a.fold) if a.cmd == "build" else cmd_assemble(a.fold)
