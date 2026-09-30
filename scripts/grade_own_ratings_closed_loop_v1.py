"""grade_own_ratings_closed_loop_v1.py -- Lane N (2026-09-30): grade the chained-C vs R paired closed loop
(docs/models/own_ratings/experiments.md section 3.3-3.4).

    CBB_TRUTH=verified_v1 .venv/Scripts/python.exe scripts/grade_own_ratings_closed_loop_v1.py

1. eval_gates.py on every run (reports under results/engine_v0/laneN_grade/), then every headline line of R, C and the
   floor runs side by side (parsed with diag_pair_gate_reports.parse, never recomputed): delta C - R, floor (a) = max over
   floor runs |f - R|, VETO when C is farther from the target than R by more than the floor.
2. Primaries from game rows (verified finals game_finals_v2): seed-mean margin MAE, bias and G9 slope, full sample and
   weeks 0-7 (game_date < 2024-12-30); floor (a) as above and floor (b) = paired game bootstrap (1,000 draws; a game's
   25 paired seeds stay together because the statistic uses the per-game seed mean).
Writes results/engine_v0/laneN_grade/closed_loop_v1.json and .md.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))
import diag_pair_gate_reports as P  # noqa: E402

RES = ROOT / "results/engine_v0"
OUT = RES / "laneN_grade"
REF, ARM = "laneN_R_s25_o0", "laneN_C_s25_o0"
FLOORS = [f"laneN_R_s25_o{o}" for o in (1000, 2000, 3000, 4000)]
WEEK8 = pd.Timestamp("2024-12-30")
N_BOOT = 1000


def per_game(tag: str) -> pd.DataFrame:
    g = pd.read_parquet(RES / tag / "games.parquet", columns=["game_id", "seed", "home_pts", "away_pts"])
    g["m"] = g["home_pts"] - g["away_pts"]
    return g.groupby("game_id")["m"].mean().rename("sim").reset_index()


def stats(d: pd.DataFrame) -> dict:
    e = d["sim"] - d["margin"]
    return {"n": int(len(d)), "mae": float(e.abs().mean()), "bias": float(e.mean()),
            "slope": float(np.polyfit(d["sim"], d["margin"], 1)[0])}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [REF, ARM] + [f for f in FLOORS if (RES / f / "games.parquet").exists()]
    floors = [f for f in runs if f in FLOORS]
    env = dict(os.environ, CBB_TRUTH="verified_v1", PYTHONIOENCODING="utf-8")
    for t in runs:
        md = OUT / f"{t}.md"
        if not md.exists():
            subprocess.run([sys.executable, "scripts/eval_gates.py", "--results", str(RES / t), "--season", "2025",
                            "--out", str(md)], check=True, env=env, stdout=subprocess.DEVNULL)
    rep = {t: P.parse(OUT / f"{t}.md") for t in runs}
    lines = []
    for key, r in rep[REF]["rows"].items():
        c = rep[ARM]["rows"].get(key)
        if c is None:
            continue
        rv, cv, tv = P.first_num(r["value"]), P.first_num(c["value"]), P.first_num(r["target"])
        fl = [P.first_num(rep[f]["rows"][key]["value"]) for f in floors if key in rep[f]["rows"]]
        row = {"gate": key[0], "line": key[1], "target": r["target"], "R": r["value"], "C": c["value"],
               "status_R": r["status"], "status_C": c["status"]}
        if rv is not None and cv is not None:
            fa = max((abs(x - rv) for x in fl if x is not None), default=np.nan)
            row.update({"delta": cv - rv, "floor_a": fa})
            if tv is not None:
                away = abs(cv - tv) - abs(rv - tv)
                row["away_from_target"] = away
                row["veto"] = bool(np.isfinite(fa) and away > 0 and abs(cv - rv) > fa)
        lines.append(row)
    L = pd.DataFrame(lines)

    fin = pd.read_parquet("data/processed/truth/game_finals_v2.parquet", columns=["game_id", "home_score", "away_score"])
    fin["margin"] = fin["home_score"] - fin["away_score"]
    gm = pd.read_parquet("data/processed/models/engine_v3/games_F2_2025.parquet", columns=["game_id", "game_date"])
    pg = {t: per_game(t).merge(fin[["game_id", "margin"]], on="game_id").merge(gm, on="game_id") for t in runs}
    prim = {}
    rng = np.random.default_rng(20260930)
    for cell, sel in (("weeks 0-7", lambda d: d[d["game_date"] < WEEK8]), ("all", lambda d: d)):
        a, b = sel(pg[REF]).set_index("game_id"), sel(pg[ARM]).set_index("game_id")
        b = b.loc[a.index]
        sa, sb = stats(a), stats(b)
        fl = {f: stats(sel(pg[f]).set_index("game_id").loc[a.index]) for f in floors}
        n = len(a)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        bt = {k: [] for k in ("mae", "bias", "slope")}
        for i in idx:
            x, y = a.iloc[i], b.iloc[i]
            s1, s2 = stats(x), stats(y)
            for k in bt:
                bt[k].append(s2[k] - s1[k])
        prim[cell] = {"n": n, "R": sa, "C": sb,
                      "delta": {k: sb[k] - sa[k] for k in bt},
                      "boot95": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bt.items()},
                      "floor_a": {k: float(max(abs(fl[f][k] - sa[k]) for f in floors)) if floors else None for k in bt},
                      "floor_runs": {f: {k: fl[f][k] - sa[k] for k in bt} for f in floors}}
    res = {"ref": REF, "arm": ARM, "floors": floors, "primary": prim, "lines": L.to_dict("records"),
           "vetoes": L[L.get("veto", False) == True][["gate", "line", "R", "C", "delta", "floor_a"]].to_dict("records") if "veto" in L else []}  # noqa: E712
    (OUT / "closed_loop_v1.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 40)
    txt = [json.dumps(prim, indent=1), L.to_string()]
    (OUT / "closed_loop_v1.md").write_text("\n\n".join(txt), encoding="utf-8")
    print("\n\n".join(txt))
    print("VETOES:", res["vetoes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
