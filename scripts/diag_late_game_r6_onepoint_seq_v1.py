"""diag_late_game_r6_onepoint_seq_v1.py -- round 6 part 1b: one-point finishes by the LAST scoring sequence
"margin before -> points scored" (e.g. trailer down 3 scores 2), sim vs actual, and the time left at it.
Same frames as diag_late_game_r6_onepoint_v1 (imported). Reported only.

    .venv/Scripts/python.exe scripts/diag_late_game_r6_onepoint_seq_v1.py --run lg4_R9_s25 --out results/late_game/round6/onepoint_seq_R9.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import diag_late_game_r6_onepoint_v1 as OP                            # noqa: E402


def seq(p, g):
    s = p[p["pts"] > 0].sort_values(["game_id", "seed", "sec"], ascending=[True, True, False])
    last = s.groupby(["game_id", "seed"]).tail(1)[["game_id", "seed", "om", "pts", "sec"]]
    x = g.merge(last, on=["game_id", "seed"], how="left")
    x["one"] = (~x["ot"]) & (x["m_reg"].abs() == 1)
    x["seq"] = x["om"].clip(-6, 6).fillna(99).astype(int).astype(str) + "->+" + x["pts"].fillna(0).clip(0, 4).astype(int).astype(str)
    x["tb"] = pd.cut(x["sec"], [-1, 5, 10, 30, 2000], labels=["le5", "5_10", "10_30", "gt30"]).astype(str)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    xs, xa = seq(*OP.frame_sim(a.run)), seq(*OP.frame_act())
    out = {}
    for k in (["seq"], ["seq", "tb"]):
        s = xs[xs["one"]].groupby(k).size() / len(xs)
        t = xa[xa["one"]].groupby(k).size() / len(xa)
        j = pd.concat([s.rename("sim"), t.rename("act")], axis=1).fillna(0)
        j["gap"] = j["sim"] - j["act"]
        j = j.sort_values("gap", ascending=False)
        out["|".join(k)] = {"|".join(map(str, i)) if isinstance(i, tuple) else str(i): r for i, r in j.round(5).to_dict("index").items()}
    # how often each pre-score state is the last scoring state at all, and P(one | state)
    for nm, x in (("sim", xs), ("act", xa)):
        gg = x.groupby("seq")["one"].agg(["size", "mean"])
        out[f"cond_{nm}"] = {i: {"share": float(r["size"] / len(x)), "p_one": float(r["mean"])} for i, r in gg.iterrows()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("one-point finishes by last scoring sequence (offence margin before -> points):")
    for c, r in list(out["seq"].items())[:10]:
        cs, ca = out["cond_sim"].get(c, {}), out["cond_act"].get(c, {})
        print(f"  {c:8s} sim {r['sim']:.4f} act {r['act']:.4f} gap {r['gap']:+.4f} | state share sim {cs.get('share', 0):.3f} act {ca.get('share', 0):.3f}")
    print("by sequence x time-left bucket (top):")
    for c, r in list(out["seq|tb"].items())[:10]:
        print(f"  {c:16s} sim {r['sim']:.4f} act {r['act']:.4f} gap {r['gap']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
