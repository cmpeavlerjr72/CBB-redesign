"""diag_late_game_r6_onepoint_v1.py -- late-game ROUND 6, part 1: which final sequences produce one-point finishes?

Reported lines (no decision line). One code path per side: the sim's round-4 possession log (period 2,
possessions starting <= 180 s) and `possessions_v4` 2025 reduced to the same frame; regulation outcome from
the sim's games and from verified finals.

  1. P(|regulation margin| = 1) decomposed by the LAST scoring possession of regulation: its type (three, two,
     FT-only 2+ made, FT-only 1 of 2, FT-only 1 of 1, other), the scorer's role at that possession's start
     (leader / trailer / tied), and its start bucket. Sim minus actual per cell sums to the P(1) gap exactly.
  2. |margin| distribution at 0:30 / 0:10 / 0:05 (first possession starting at <= T; end margin if none).
  3. Final-30 s behaviour: trailing-by-3 offence (three share of FGA, FT-only share = fouled), trailing-by-2
     (two vs three), leading offence FT trips made 1 of 2, leading offence TOV rate; in all games and in games
     that end |m| = 1.

    .venv/Scripts/python.exe scripts/diag_late_game_r6_onepoint_v1.py --run lg4_R9_s25 --out results/late_game/round6/onepoint_R9.json
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
BUCK = ((-1, 10, "le10"), (10, 30, "10_30"), (30, 120, "30_120"), (120, 180, "120_180"))


def frame_act():
    d = pd.read_parquet(ROOT / "data/processed/possessions_v4/possessions_2025.parquet")
    d = d[(d["period"] == 2) & (d["start_clock"] <= 180)]
    sgn = np.where(d["offense_is_home"], 1, -1)
    p = pd.DataFrame({"game_id": d["game_id"].to_numpy(), "seed": -1, "sec": d["start_clock"].to_numpy(),
                      "om": d["start_score_diff"].to_numpy(), "hm": d["start_score_diff"].to_numpy() * sgn,
                      "pts": d["points"].to_numpy(), "fgm3": d["fgm_3"].to_numpy(),
                      "fgm2": (d["fgm_rim"] + d["fgm_jump2"]).to_numpy(), "fga3": d["fga_3"].to_numpy(),
                      "fga2": (d["fga_rim"] + d["fga_jump2"]).to_numpy(), "fta": d["fta"].to_numpy(),
                      "ftm": d["ftm"].to_numpy(), "tov": (d["terminal_event"] == "TOV").astype(int).to_numpy(),
                      "used": d["duration_s"].to_numpy()})
    a = REF.load_actual_games(2025)
    g = pd.DataFrame({"game_id": a["game_id"], "seed": -1, "ot": a["n_periods"] > 2,
                      "m_reg": np.where(a["n_periods"] > 2, 0, a["home_score"] - a["away_score"])})
    return p[p["game_id"].isin(g["game_id"])], g


def frame_sim(tag):
    t = pd.read_parquet(RES / tag / "tap_poss.parquet")
    if "per" in t:
        t = t[t["per"] == 2]
    sgn = np.where(t["off"] == 0, 1, -1)
    hm = (t["hp"] - t["ap"]).to_numpy()
    p = pd.DataFrame({"game_id": t["game_id"].to_numpy(), "seed": t["seed"].to_numpy(), "sec": t["sec"].to_numpy(),
                      "om": hm * sgn, "hm": hm, "pts": t["d_pts_off"].to_numpy(), "fgm3": t["d_fgm3"].to_numpy(),
                      "fgm2": (t["d_fgm2_rim"] + t["d_fgm2_jump"]).to_numpy(), "fga3": t["d_fga3"].to_numpy(),
                      "fga2": (t["d_fga2_rim"] + t["d_fga2_jump"]).to_numpy(), "fta": t["d_fta"].to_numpy(),
                      "ftm": t["d_ftm"].to_numpy(), "tov": t["d_tov"].to_numpy(), "used": t["used"].to_numpy()})
    g = pd.read_parquet(RES / tag / "games.parquet", columns=["game_id", "seed", "home_pts", "away_pts", "n_periods"])
    g = pd.DataFrame({"game_id": g["game_id"], "seed": g["seed"], "ot": g["n_periods"] > 2,
                      "m_reg": np.where(g["n_periods"] > 2, 0, g["home_pts"] - g["away_pts"])})
    return p, g


def stype(r) -> str:
    if r["fgm3"] > 0:
        return "three"
    if r["fgm2"] > 0:
        return "two"
    if r["ftm"] >= 2:
        return "ft_2plus"
    if r["ftm"] == 1 and r["fta"] >= 2:
        return "ft_1of2"
    if r["ftm"] == 1:
        return "ft_1of1"
    return "other"


def role(om):
    return np.where(om > 0, "leader", np.where(om < 0, "trailer", "tied"))


def bucket(sec):
    out = np.full(len(sec), "gt180", dtype=object)
    for lo, hi, n in BUCK:
        out[(sec > lo) & (sec <= hi)] = n
    return out


def last_score(p: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    s = p[p["pts"] > 0].sort_values(["game_id", "seed", "sec"], ascending=[True, True, False])
    last = s.groupby(["game_id", "seed"]).tail(1).copy()
    last["type"] = [stype(r) for r in last[["fgm3", "fgm2", "ftm", "fta"]].to_dict("records")]
    last["who"] = role(last["om"].to_numpy())
    last["bkt"] = bucket(last["sec"].to_numpy())
    x = g.merge(last[["game_id", "seed", "type", "who", "bkt"]], on=["game_id", "seed"], how="left")
    x[["type", "who", "bkt"]] = x[["type", "who", "bkt"]].fillna("none_in_final_180s")
    x["one"] = (~x["ot"]) & (x["m_reg"].abs() == 1)
    return x


def margins(p, g):
    p = p.sort_values(["game_id", "seed", "sec"], ascending=[True, True, False])
    out = {}
    for T in (30, 10, 5):
        f = p[p["sec"] <= T].groupby(["game_id", "seed"]).head(1)[["game_id", "seed", "hm"]]
        x = g.merge(f, on=["game_id", "seed"], how="left")
        m = x["hm"].fillna(x["m_reg"]).abs().clip(0, 8)
        out[f"T{T}"] = {str(k): float((m == k).mean()) for k in range(9)}
    return out


def behaviour(p, g):
    one = g[(~g["ot"]) & (g["m_reg"].abs() == 1)][["game_id", "seed"]].assign(one=True)
    p = p.merge(one, on=["game_id", "seed"], how="left").fillna({"one": False})
    out = {}
    for scope, q in (("all_games", p), ("one_point_games", p[p["one"]])):
        for nm, lo, hi in (("le10", -1, 10), ("10_30", 10, 30)):
            w = q[(q["sec"] > lo) & (q["sec"] <= hi)]
            def cell(m):
                x = w[m]
                fga = x["fga3"].sum() + x["fga2"].sum()
                ft2 = x[(x["fta"] >= 2) & (x["fga3"] + x["fga2"] == 0)]
                return {"n": int(len(x)), "three_share_fga": float(x["fga3"].sum() / fga) if fga else None,
                        "ft_only": float(((x["fta"] > 0) & (x["fga3"] + x["fga2"] == 0) & (x["tov"] == 0)).mean()) if len(x) else None,
                        "tov": float(x["tov"].mean()) if len(x) else None,
                        "ft_1of2_share": float(((ft2["fta"] == 2) & (ft2["ftm"] == 1)).mean()) if len(ft2) else None,
                        "ppp": float(x["pts"].mean()) if len(x) else None}
            out[f"{scope}|{nm}|trail3"] = cell(w["om"] == -3)
            out[f"{scope}|{nm}|trail2"] = cell(w["om"] == -2)
            out[f"{scope}|{nm}|trail1"] = cell(w["om"] == -1)
            out[f"{scope}|{nm}|lead1_3"] = cell((w["om"] >= 1) & (w["om"] <= 3))
            out[f"{scope}|{nm}|lead4_6"] = cell((w["om"] >= 4) & (w["om"] <= 6))
    return out


def decomp(xs, xa):
    keys = ["type", "who", "bkt"]
    cs = xs[xs["one"]].groupby(keys).size() / len(xs)
    ca = xa[xa["one"]].groupby(keys).size() / len(xa)
    j = pd.concat([cs.rename("sim"), ca.rename("act")], axis=1).fillna(0.0)
    j["gap"] = j["sim"] - j["act"]
    return j.sort_values("gap", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pa, ga = frame_act()
    ps, gs = frame_sim(a.run)
    xa, xs = last_score(pa, ga), last_score(ps, gs)
    j = decomp(xs, xa)
    by = {}
    for k in ("type", "who", "bkt"):
        s = xs[xs["one"]].groupby(k).size() / len(xs)
        t = xa[xa["one"]].groupby(k).size() / len(xa)
        by[k] = pd.concat([s.rename("sim"), t.rename("act")], axis=1).fillna(0).assign(gap=lambda d: d.sim - d.act).round(5).to_dict("index")
    # conditional: P(one-point | last score type x who), and the share of games in each cell
    cond = {}
    for nm, x in (("sim", xs), ("act", xa)):
        gg = x.groupby(["type", "who"])["one"].agg(["size", "mean"])
        gg["share"] = gg["size"] / len(x)
        cond[nm] = {f"{i[0]}|{i[1]}": {"share": float(r["share"]), "p_one": float(r["mean"]), "n": int(r["size"])}
                    for i, r in gg.iterrows()}
    out = {"run": a.run, "P1_sim": float(xs["one"].mean()), "P1_act": float(xa["one"].mean()),
           "decomp_type_who_bkt": {"|".join(i): r for i, r in j.round(5).to_dict("index").items()},
           "by": by, "conditional": cond, "margins_sim": margins(ps, gs), "margins_act": margins(pa, ga),
           "behaviour_sim": behaviour(ps, gs), "behaviour_act": behaviour(pa, ga)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"P1 sim {out['P1_sim']:.4f} act {out['P1_act']:.4f} gap {out['P1_sim'] - out['P1_act']:+.4f}")
    for k, v in by.items():
        print(f"-- by {k}:")
        for c, r in v.items():
            print(f"   {c:20s} sim {r['sim']:.4f} act {r['act']:.4f} gap {r['gap']:+.4f}")
    print("-- top cells (type|who|bucket):")
    for c, r in list(out["decomp_type_who_bkt"].items())[:12]:
        print(f"   {c:36s} sim {r['sim']:.4f} act {r['act']:.4f} gap {r['gap']:+.4f}")
    print("-- conditional P(one-point | last score type|who): share / p_one")
    for c in sorted(set(cond["sim"]) | set(cond["act"])):
        s, t = cond["sim"].get(c, {}), cond["act"].get(c, {})
        print(f"   {c:26s} sim {s.get('share', 0):.3f}/{s.get('p_one', 0):.3f}  act {t.get('share', 0):.3f}/{t.get('p_one', 0):.3f} (n {t.get('n', 0)})")
    for T in ("T30", "T10", "T5"):
        print(f"-- |m| at {T}: sim", [round(out['margins_sim'][T][str(k)], 3) for k in range(7)],
              " act", [round(out['margins_act'][T][str(k)], 3) for k in range(7)])
    print("-- behaviour (n, 3PA share, FT-only, TOV, FT 1-of-2, PPP): sim | act")
    for c in out["behaviour_act"]:
        s, t = out["behaviour_sim"][c], out["behaviour_act"][c]
        f = lambda d: " ".join("  -  " if d[k] is None else f"{d[k]:.3f}" for k in ("three_share_fga", "ft_only", "tov", "ft_1of2_share", "ppp"))  # noqa: E731
        print(f"   {c:34s} {s['n']:6d} {f(s)} | {t['n']:5d} {f(t)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
