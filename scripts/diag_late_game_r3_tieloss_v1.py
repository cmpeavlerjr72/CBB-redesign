"""diag_late_game_r3_tieloss_v1.py -- WHERE the regulation ties are lost (late_game experiments.md 7.6).

Reported lines only; no decision line lives here. One code path for every sim run and for the actual:
both sides are reduced to the same possession frame (period 2, start <= 180 s) and every quantity below
is computed by the same function on both.

  1. m(T) = home margin at the start of the first period-2 possession starting at <= T seconds
     (end-of-regulation margin if none), T = 120, 60, 30, 10. Distribution of |m(T)|.
  2. Kernel P(regulation tie | |m(T)| = k) and a shift-share of the tie-rate gap at each T into
     ARRIVAL (distribution of |m(T)|) and CONVERSION (kernel), both substitution orders averaged.
  3. Behaviour cells in the final 2:00 by offence role band x clock bucket.
  4. Tied final possessions (start tied, <= 35 s).

Actual: `possessions_v4` 2025 for states, verified finals (`reference.load_actual_games`) for OT.

    .venv/Scripts/python.exe scripts/diag_late_game_r3_tieloss_v1.py --runs lg3_R9_s25 lg3_R0_s25 \
        --out results/late_game/round3/tieloss.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CBB_TRUTH", "verified_v1")

from cbb_sim.eval import reference as REF                             # noqa: E402

RES = ROOT / "results/engine_v0"
POSS = ROOT / "data/processed/possessions_v4/possessions_2025.parquet"
SAMPLE = ROOT / "data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet"
TS = (120, 60, 30, 10)
KMAX = 8
MIN_CELL = 200
BANDS = (("trail_4_6", -6, -4), ("trail_1_3", -3, -1), ("tied", 0, 0), ("lead_1_3", 1, 3),
         ("lead_4_6", 4, 6))
BUCKETS = (("(60,120]", 60, 120), ("(30,60]", 30, 60), ("(10,30]", 10, 30), ("(0,10]", 0, 10))


def lab(n: int) -> str:
    return "UNDERPOWERED" if n < MIN_CELL else ""


# ---------------------------------------------------------------- frames
def actual_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(POSS)
    d = d[(d["period"] == 2) & (d["start_clock"] <= 180)].copy()
    sgn = np.where(d["offense_is_home"].to_numpy(), 1, -1)
    p = pd.DataFrame({
        "game_id": d["game_id"].to_numpy(), "seed": -1,
        "sec": d["start_clock"].to_numpy(), "off_margin": d["start_score_diff"].to_numpy(),
        "home_margin": d["start_score_diff"].to_numpy() * sgn, "sgn": sgn,
        "used": d["duration_s"].to_numpy(), "pts": d["points"].to_numpy(),
        "fta": d["fta"].to_numpy(), "ftm": d["ftm"].to_numpy(), "fga3": d["fga_3"].to_numpy(),
        "fga2": (d["fga_rim"] + d["fga_jump2"]).to_numpy(),
        "tov": (d["terminal_event"] == "TOV").to_numpy().astype(int),
        "oreb": d["oreb_count"].to_numpy(), "bon": d["off_in_bonus"].to_numpy().astype(int),
        "pts_def": d["tech_points_def"].to_numpy(),
    })
    act = REF.load_actual_games(2025)[["game_id", "n_periods"]]
    g = act.assign(ot=act["n_periods"] > 2, seed=-1)[["game_id", "seed", "ot"]]
    p = p[p["game_id"].isin(g["game_id"])]
    return p, g


def sim_frame(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.read_parquet(RES / tag / "tap_poss.parquet")
    if "per" in t:                       # round-4 tap logs both halves; this diagnostic reads period 2
        t = t[t["per"] == 2].reset_index(drop=True)
    sgn = np.where(t["off"].to_numpy() == 0, 1, -1)
    hm = (t["hp"] - t["ap"]).to_numpy()
    p = pd.DataFrame({
        "game_id": t["game_id"].to_numpy(), "seed": t["seed"].to_numpy(),
        "sec": t["sec"].to_numpy(), "off_margin": hm * sgn, "home_margin": hm, "sgn": sgn,
        "used": t["used"].to_numpy(), "pts": t["d_pts_off"].to_numpy(),
        "fta": t["d_fta"].to_numpy(), "ftm": t["d_ftm"].to_numpy(), "fga3": t["d_fga3"].to_numpy(),
        "fga2": (t["d_fga2_rim"] + t["d_fga2_jump"]).to_numpy(), "tov": t["d_tov"].to_numpy(),
        "oreb": t["d_oreb"].to_numpy(), "bon": t["bon"].to_numpy(), "pts_def": t["d_pts_def"].to_numpy(),
    })
    g = pd.read_parquet(RES / tag / "games.parquet", columns=["game_id", "seed", "n_periods"])
    g = g.assign(ot=g["n_periods"] > 2)[["game_id", "seed", "ot"]]
    return p, g


# ---------------------------------------------------------------- per-sim anchors
def anchors(p: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, seed): ot and |m(T)| for every T."""
    p = p.sort_values(["game_id", "seed", "sec"], ascending=[True, True, False])
    last = p.groupby(["game_id", "seed"]).tail(1)
    end_m = (last["home_margin"] + last["sgn"] * (last["pts"] - last["pts_def"])).to_numpy()
    out = last[["game_id", "seed"]].copy()
    out["m_end"] = end_m
    out = out.merge(g, on=["game_id", "seed"], how="inner")
    out.loc[out["ot"], "m_end"] = 0
    for T in TS:
        f = p[p["sec"] <= T].groupby(["game_id", "seed"]).head(1)[["game_id", "seed", "home_margin"]]
        out = out.merge(f.rename(columns={"home_margin": f"m{T}"}), on=["game_id", "seed"], how="left")
        out[f"m{T}"] = out[f"m{T}"].fillna(out["m_end"]).abs().astype(int)
    out["end_abs_reg"] = out["m_end"].abs()
    return out


def dist_kernel(a: pd.DataFrame, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.minimum(a[f"m{T}"].to_numpy(), KMAX)
    ot = a["ot"].to_numpy().astype(float)
    n = np.array([(k == j).sum() for j in range(KMAX + 1)], dtype=float)
    pk = n / n.sum()
    ker = np.array([ot[k == j].mean() if n[j] else 0.0 for j in range(KMAX + 1)])
    return pk, ker, n


def shift_share(sim: pd.DataFrame, act: pd.DataFrame, T: int) -> dict:
    ps, ks, ns = dist_kernel(sim, T)
    pa, ka, na = dist_kernel(act, T)
    tot = float((ps * ks).sum() - (pa * ka).sum())
    arr = 0.5 * (((ps - pa) * ka).sum() + ((ps - pa) * ks).sum())
    conv = 0.5 * ((pa * (ks - ka)).sum() + (ps * (ks - ka)).sum())
    return {"T": T, "sim_tie": float((ps * ks).sum()), "act_tie": float((pa * ka).sum()),
            "gap": tot, "arrival": float(arr), "conversion": float(conv),
            "arrival_share": float(arr / tot) if tot else None,
            "sim_dist": ps.round(4).tolist(), "act_dist": pa.round(4).tolist(),
            "sim_kernel": ks.round(4).tolist(), "act_kernel": ka.round(4).tolist(),
            "sim_n": ns.astype(int).tolist(), "act_n": na.astype(int).tolist(),
            "act_kernel_labels": [lab(int(x)) for x in na]}


# ---------------------------------------------------------------- behaviour cells
def behaviour(p: pd.DataFrame, a: pd.DataFrame) -> dict:
    w = p[p["sec"] <= 120].copy()
    out = {}
    for bn, lo, hi in BANDS:
        for kn, blo, bhi in BUCKETS:
            m = (w["off_margin"] >= lo) & (w["off_margin"] <= hi) & (w["sec"] > blo) & (w["sec"] <= bhi)
            x = w[m]
            n = len(x)
            fga = (x["fga3"] + x["fga2"]).sum()
            fonly = ((x["fta"] > 0) & (x["fga3"] + x["fga2"] == 0) & (x["tov"] == 0)).mean() if n else None
            out[f"{bn}|{kn}"] = {
                "n": int(n), "label": lab(n),
                "used_mean": float(x["used"].mean()) if n else None,
                "ft_only_share": float(fonly) if n else None,
                "three_share_fga": float(x["fga3"].sum() / fga) if fga else None,
                "ppp": float(x["pts"].mean()) if n else None,
                "ft_pct": float(x["ftm"].sum() / x["fta"].sum()) if x["fta"].sum() else None,
                "fta_per_poss": float(x["fta"].mean()) if n else None,
                "tov_rate": float(x["tov"].mean()) if n else None,
                "oreb_per_poss": float(x["oreb"].mean()) if n else None,
                "bonus_share": float(x["bon"].mean()) if n else None,
            }
    # tied final possessions
    t = p[(p["off_margin"] == 0) & (p["sec"] <= 35)].merge(a[["game_id", "seed", "ot"]],
                                                           on=["game_id", "seed"])
    n = len(t)
    tied = {"n": int(n), "label": lab(n)}
    if n:
        tied.update({
            "used_mean": float(t["used"].mean()),
            "share_to_horn": float((t["used"] >= t["sec"]).mean()),
            "share_used_ge_sec_minus_3": float((t["used"] >= t["sec"] - 3).mean()),
            "pts_dist": {str(k): float((np.minimum(t["pts"], 3) == k).mean()) for k in range(4)},
            "share_game_ot": float(t["ot"].mean()),
            "ft_only_share": float(((t["fta"] > 0) & (t["fga3"] + t["fga2"] == 0) & (t["tov"] == 0)).mean()),
            "three_share_fga": float(t["fga3"].sum() / max((t["fga3"] + t["fga2"]).sum(), 1)),
            "tov_rate": float(t["tov"].mean()),
        })
    # possessions per sim inside the window, overall
    return {"cells": out, "tied_final": tied,
            "final2_poss_per_sim": float(len(w) / len(a)),
            "final2_fta_per_sim": float(w["fta"].sum() / len(a))}


def transitions(a: pd.DataFrame) -> dict:
    """Convergence vs divergence: from |m(60)| = k, P(|m(10)| smaller / same / larger) and P(tie)."""
    out = {}
    for k in range(0, 7):
        x = a[a["m60"] == k]
        n = len(x)
        out[str(k)] = {"n": int(n), "label": lab(n),
                       "p_closer_at_10": float((x["m10"] < k).mean()) if n else None,
                       "p_same_at_10": float((x["m10"] == k).mean()) if n else None,
                       "p_wider_at_10": float((x["m10"] > k).mean()) if n else None,
                       "p_tie": float(x["ot"].mean()) if n else None}
    return out


def summarise(p, g) -> dict:
    a = anchors(p, g)
    return {"n_sims": int(len(a)), "ot_rate": float(a["ot"].mean()),
            "p_abs1_end": float((a["end_abs_reg"] == 1).mean()),
            "_a": a, "behaviour": behaviour(p, a), "transitions": transitions(a)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pa, ga = actual_frame()
    ids = set(pd.read_parquet(SAMPLE)["game_id"].astype("int64"))
    act = {"actual_season": summarise(pa, ga),
           "actual_sample": summarise(pa[pa["game_id"].isin(ids)], ga[ga["game_id"].isin(ids)])}
    runs = {t: summarise(*sim_frame(t)) for t in args.runs}
    out: dict = {"spec": "docs/models/late_game/experiments.md section 7.6", "actual": {}, "runs": {}}
    for k, v in act.items():
        out["actual"][k] = {kk: vv for kk, vv in v.items() if kk != "_a"}
    for t, v in runs.items():
        r = {kk: vv for kk, vv in v.items() if kk != "_a"}
        r["shift_share_vs_season"] = [shift_share(v["_a"], act["actual_season"]["_a"], T) for T in TS]
        r["shift_share_vs_sample"] = [shift_share(v["_a"], act["actual_sample"]["_a"], T) for T in TS]
        out["runs"][t] = r
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    # ---- console summary
    print("OT rate:", {k: round(v["ot_rate"], 4) for k, v in act.items()},
          {t: round(v["ot_rate"], 4) for t, v in runs.items()})
    for t, r in out["runs"].items():
        print(f"\n== {t}: shift-share vs season actual")
        for s in r["shift_share_vs_season"]:
            print(f"  T={s['T']:3d}: sim {s['sim_tie']:.4f} act {s['act_tie']:.4f} gap {s['gap']:+.4f}  "
                  f"arrival {s['arrival']:+.4f}  conversion {s['conversion']:+.4f}")
            print(f"     P(|m|=k) sim {s['sim_dist'][:7]}\n              act {s['act_dist'][:7]}")
            print(f"     kernel   sim {s['sim_kernel'][:7]}\n              act {s['act_kernel'][:7]}  n_act {s['act_n'][:7]}")
    print("\nbehaviour (final 2:00): used / ft_only / 3PA share / ppp / ft% / n")
    srcs = {**{k: out["actual"][k] for k in out["actual"]}, **out["runs"]}
    for cell in out["actual"]["actual_season"]["behaviour"]["cells"]:
        line = f"  {cell:22s}"
        for k, v in srcs.items():
            c = v["behaviour"]["cells"][cell]
            f = lambda x, d=3: "  -  " if x is None else f"{x:.{d}f}"  # noqa: E731
            line += (f" | {k[:10]} {f(c['used_mean'], 1)} {f(c['ft_only_share'])} {f(c['three_share_fga'])}"
                     f" {f(c['ppp'])} {f(c['ft_pct'])} {c['n']}")
        print(line)
    print("\ntied final possessions:")
    for k, v in srcs.items():
        print(f"  {k:16s} {v['behaviour']['tied_final']}")
    print("\ntransitions from |m(60)|=k:")
    for k, v in srcs.items():
        print(f"  {k:16s} " + " ".join(
            f"{kk}:{(x['p_closer_at_10'] or 0):.2f}/{(x['p_wider_at_10'] or 0):.2f}/{(x['p_tie'] or 0):.3f}"
            for kk, x in v["transitions"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
