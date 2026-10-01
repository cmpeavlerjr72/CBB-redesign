"""diag_late_game_r4_owner_v1.py -- late-game ROUND 4: who owns the end-of-period possession value?

Reported lines (the pre-registration's motivating evidence), one code path per side. Possessions that START
at <= 35 s in period 1 or 2, by start bucket (and role in period 2), sim (round-4 tap) vs actual
(`possessions_v4` 2025, same 500 games and full season):
  points per possession, and its parts: no-shot rate (end_period), TOV rate, FGA mix per possession,
  make rate by type, FTA per possession. A shift-share splits the PPP gap into: no-shot, TOV, shot mix,
  make, FT (sequential, documented order).
Shots, first chances: make rate by time left at the shot (actual: chance end clock; sim: possession-start
clock minus the fed `chance_elapsed_s`), sim = mean fg_make probability, actual = realised.

    .venv/Scripts/python.exe scripts/diag_late_game_r4_owner_v1.py --run lg4_R9_s25 --out results/late_game/round4/owner_R9.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/engine_v0"
SAMPLE = ROOT / "data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet"
SB = [(-1, 3), (3, 6), (6, 10), (10, 20), (20, 35)]
TL = [(-99, 1), (1, 3), (3, 6), (6, 10), (10, 35)]
MIN_CELL = 200


def act_frames():
    p = pd.read_parquet(ROOT / "data/processed/possessions_v4/possessions_2025.parquet")
    p = p[(p["period"] <= 2) & (p["start_clock"] <= 35)]
    p = pd.DataFrame({"game_id": p["game_id"], "per": p["period"], "sec": p["start_clock"],
                      "role": np.sign(p["start_score_diff"]), "pts": p["points"],
                      "noshot": (p["terminal_event"] == "end_period").astype(int),
                      "tov": (p["terminal_event"] == "TOV").astype(int),
                      "fga3": p["fga_3"], "fgm3": p["fgm_3"], "fga2": p["fga_rim"] + p["fga_jump2"],
                      "fgm2": p["fgm_rim"] + p["fgm_jump2"], "fta": p["fta"], "ftm": p["ftm"]})
    c = pd.read_parquet(ROOT / "data/processed/possessions_v4/chances_2025.parquet")
    c = c[(c["period"] <= 2) & (c["chance_number"] == 1) & (c["start_clock"] <= 35)]
    rows = []
    for cls, a, m in (("rim", "fga_rim", "fgm_rim"), ("jump2", "fga_jump2", "fgm_jump2"), ("three", "fga_3", "fgm_3")):
        x = c[c[a] > 0]
        rows.append(pd.DataFrame({"game_id": x["game_id"], "per": x["period"], "tl": x["end_clock"],
                                  "cls": cls, "made": x[m] / x[a], "w": x[a]}))
    return p, pd.concat(rows)


def sim_frames(tag):
    t = pd.read_parquet(RES / tag / "tap_poss.parquet")
    t = t[t["sec"] <= 35]
    sgn = np.where(t["off"] == 0, 1, -1)
    p = pd.DataFrame({"game_id": t["game_id"], "per": t["per"], "sec": t["sec"],
                      "role": np.sign((t["hp"] - t["ap"]) * sgn), "pts": t["d_pts_off"], "noshot": 0,
                      "tov": t["d_tov"], "fga3": t["d_fga3"], "fgm3": t["d_fgm3"],
                      "fga2": t["d_fga2_rim"] + t["d_fga2_jump"], "fgm2": t["d_fgm2_rim"] + t["d_fgm2_jump"],
                      "fta": t["d_fta"], "ftm": t["d_ftm"]})
    # a round-4 no-shot possession leaves no box event at all (every served possession leaves one)
    p["noshot"] = ((t["d_fga3"] + t["d_fga2_rim"] + t["d_fga2_jump"] + t["d_fta"] + t["d_tov"]) == 0).astype(int).to_numpy()
    s = pd.read_parquet(RES / tag / "tap_shots.parquet")
    s = s[s["chance"] == 1]
    sh = pd.DataFrame({"per": s["per"], "tl": s["sec"] - s["elapsed"],
                       "cls": s["cls"].map({0: "rim", 1: "jump2", 2: "three"}), "made": s["p"], "w": 1.0})
    return p, sh


def poss_table(p: pd.DataFrame) -> dict:
    out = {}
    for per in (1, 2):
        for roles, rn in (((-1, 0, 1), "all"), ((0,), "tied"), ((-1,), "trail"), ((1,), "lead")):
            if per == 1 and rn != "all":
                continue
            for lo, hi in SB:
                x = p[(p["per"] == per) & p["role"].isin(roles) & (p["sec"] > lo) & (p["sec"] <= hi)]
                n = len(x)
                if not n:
                    continue
                out[f"P{per}|{rn}|({lo},{hi}]"] = {
                    "n": n, "label": "UNDERPOWERED" if n < MIN_CELL else "",
                    "ppp": x["pts"].mean(), "noshot": x["noshot"].mean(), "tov": x["tov"].mean(),
                    "fga3": x["fga3"].mean(), "fga2": x["fga2"].mean(),
                    "p3": x["fgm3"].sum() / max(x["fga3"].sum(), 1), "p2": x["fgm2"].sum() / max(x["fga2"].sum(), 1),
                    "fta": x["fta"].mean(), "ftp": x["ftm"].sum() / max(x["fta"].sum(), 1)}
    return out


def shift_share(s: dict, a: dict) -> dict:
    """PPP = 3*fga3*p3 + 2*fga2*p2 + fta*ftp (identity up to and-one/technicals; residual reported).
    Sequential substitution sim -> actual in the order: volume of shots (fga3, fga2, which absorb the
    no-shot and TOV rates), make (p3, p2), FT (fta, ftp)."""
    f = lambda d: 3 * d["fga3"] * d["p3"] + 2 * d["fga2"] * d["p2"] + d["fta"] * d["ftp"]  # noqa: E731
    cur = dict(s)
    parts = {}
    for name, keys in (("shot_volume_and_mix", ("fga3", "fga2")), ("make", ("p3", "p2")), ("ft", ("fta", "ftp"))):
        before = f(cur)
        for k in keys:
            cur[k] = a[k]
        parts[name] = f(cur) - before
    parts["identity_sim"] = f(s)
    parts["identity_act"] = f(a)
    parts["ppp_gap"] = s["ppp"] - a["ppp"]
    return parts


def shot_table(sh: pd.DataFrame) -> dict:
    out = {}
    for per in (1, 2):
        for cls in ("rim", "jump2", "three"):
            for lo, hi in TL:
                x = sh[(sh["per"] == per) & (sh["cls"] == cls) & (sh["tl"] > lo) & (sh["tl"] <= hi)]
                w = x["w"].sum()
                out[f"P{per}|{cls}|tl({lo},{hi}]"] = {"n": float(w), "label": "UNDERPOWERED" if w < MIN_CELL else "",
                                                     "make": float((x["made"] * x["w"]).sum() / w) if w else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = set(pd.read_parquet(SAMPLE)["game_id"].astype("int64"))
    pa, sa = act_frames()
    ps, ss = sim_frames(a.run)
    T = {"act_season": poss_table(pa), "act_sample": poss_table(pa[pa["game_id"].isin(ids)]), "sim": poss_table(ps)}
    S = {"act_season": shot_table(sa), "sim": shot_table(ss)}
    share = {k: shift_share(T["sim"][k], T["act_season"][k]) for k in T["sim"] if k in T["act_season"]}
    out = {"run": a.run, "poss": T, "shots": S, "shift_share_vs_season": share}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("possession cells: sim | season actual   (ppp noshot tov fga3 fga2 p3 p2 fta)  ; PPP gap split")
    for k in T["sim"]:
        s, x = T["sim"][k], T["act_season"].get(k)
        if x is None:
            continue
        f = lambda d: f"{d['ppp']:.3f} {d['noshot']:.3f} {d['tov']:.3f} {d['fga3']:.3f} {d['fga2']:.3f} {d['p3']:.3f} {d['p2']:.3f} {d['fta']:.3f}"  # noqa: E731
        sh = share[k]
        print(f" {k:22s} n {s['n']:6d}/{x['n']:5d}{'*' if x['label'] else ' '} | {f(s)} | {f(x)} | gap {sh['ppp_gap']:+.3f}"
              f" = vol {sh['shot_volume_and_mix']:+.3f} make {sh['make']:+.3f} ft {sh['ft']:+.3f}")
    print("\nfirst-chance make by time left at the shot: sim (mean p) | actual")
    for k in S["sim"]:
        s, x = S["sim"][k], S["act_season"][k]
        print(f" {k:24s} {s['make'] if s['make'] is not None else float('nan'):.3f} (n {s['n']:7.0f}) | "
              f"{x['make'] if x['make'] is not None else float('nan'):.3f} (n {x['n']:5.0f}){' UNDERPOWERED' if x['label'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
