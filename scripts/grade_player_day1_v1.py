#!/usr/bin/env python
"""
grade_player_day1_v1.py -- Phase 1 game-level table for the player-layer day-1 sizing (2026-10-05). Blind to arm identity
except by the run tags it is handed; one function scores every run.

Per run (verified truth, `gates.build_grading_frame`): margin MAE, total MAE, total bias, margin bias, home-win Brier, mean
sim total SD / realised |total residual| ratio. Per pair (arm vs reference, same seeds): paired delta of each metric with a
game-bootstrap 95% interval, and the mean per-game |shift| of the sim mean margin / total. The reseed pair (reference at
another seed offset) gives the noise floor for both. Segments: games where the reference had NO named slot for either team
(true day-1 games: arms must be identical there) vs games with named slots.

    python scripts/grade_player_day1_v1.py --season 2025 --input-dir data/processed/models/engine_v3 \
        --ref <run dir> --reseed <run dir> --arms <run dir>[,<run dir>] --out results/player_day1/phase1_F2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import gates as G  # noqa: E402


def frame(run: str, season: int) -> pd.DataFrame:
    s, _ = G.build_grading_frame(pd.read_parquet(Path(run) / "games.parquet"), season)
    return s.set_index("game_id").sort_index()


def metrics(s: pd.DataFrame) -> dict:
    em = s["sim_margin_mean"] - s["margin"]
    et = s["sim_total_mean"] - s["total"]
    hw = (s["margin"] > 0).astype(float)
    return {"n": int(len(s)), "margin_mae": float(em.abs().mean()), "total_mae": float(et.abs().mean()),
            "total_bias": float(et.mean()), "margin_bias": float(em.mean()),
            "brier": float(((s["p_home"] - hw) ** 2).mean()),
            "total_sd_ratio": float(s["sim_total_sd"].mean() / et.std()),
            "margin_sd_ratio": float(s["sim_margin_sd"].mean() / em.std())}


def pair(a: pd.DataFrame, r: pd.DataFrame, B: int = 2000, seed: int = 0) -> dict:
    ids = a.index.intersection(r.index)
    a, r = a.loc[ids], r.loc[ids]
    out = {"n": int(len(ids)),
           "mean_abs_shift_margin": float((a["sim_margin_mean"] - r["sim_margin_mean"]).abs().mean()),
           "mean_abs_shift_total": float((a["sim_total_mean"] - r["sim_total_mean"]).abs().mean()),
           "mean_shift_total": float((a["sim_total_mean"] - r["sim_total_mean"]).mean())}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    for k in ("margin_mae", "total_mae", "total_bias", "brier"):
        if k == "margin_mae":
            fa, fr = (a["sim_margin_mean"] - a["margin"]).abs().to_numpy(), (r["sim_margin_mean"] - r["margin"]).abs().to_numpy()
        elif k == "total_mae":
            fa, fr = (a["sim_total_mean"] - a["total"]).abs().to_numpy(), (r["sim_total_mean"] - r["total"]).abs().to_numpy()
        elif k == "total_bias":
            fa, fr = (a["sim_total_mean"] - a["total"]).to_numpy(), (r["sim_total_mean"] - r["total"]).to_numpy()
        else:
            hw = (a["margin"] > 0).astype(float).to_numpy()
            fa, fr = (a["p_home"].to_numpy() - hw) ** 2, (r["p_home"].to_numpy() - hw) ** 2
        d = fa - fr
        bs = d[idx].mean(axis=1)
        out[f"d_{k}"] = float(d.mean())
        out[f"d_{k}_ci95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--input-dir", required=True, help="reference inputs (to find games with named slots)")
    ap.add_argument("--fold", default=None)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--reseed", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fold = a.fold or ("F2" if a.season == 2025 else "F1")
    gi = pd.read_parquet(Path(a.input_dir) / f"games_{fold}_{a.season}.parquet")
    z = np.load(Path(a.input_dir) / f"arrays_{fold}_{a.season}.npz")
    named = pd.Series((z["roster_cbbd"] > 0).any(axis=(1, 2)), index=gi["game_id"].astype("int64"))
    runs = {"ref": a.ref, "reseed": a.reseed, **{Path(x).name: x for x in a.arms.split(",") if x}}
    fr = {k: frame(v, a.season) for k, v in runs.items()}
    res = {"runs": runs, "metrics": {}, "pairs": {}}
    for seg in ("all", "named", "day1"):
        for k, s in fr.items():
            m = named.reindex(s.index).fillna(False).to_numpy()
            ss = s if seg == "all" else (s[m] if seg == "named" else s[~m])
            res["metrics"][f"{k}|{seg}"] = metrics(ss) if len(ss) else None
        for k in [x for x in fr if x != "ref"]:
            s, r = fr[k], fr["ref"]
            m = named.reindex(r.index).fillna(False).to_numpy()
            rr = r if seg == "all" else (r[m] if seg == "named" else r[~m])
            res["pairs"][f"{k}_vs_ref|{seg}"] = pair(s, rr) if len(rr) else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
