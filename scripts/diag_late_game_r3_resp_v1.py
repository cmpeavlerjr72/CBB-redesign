"""diag_late_game_r3_resp_v1.py -- responsiveness of the OT rate (late-game round 3, reported line).

Matchup-specific rule: bucket games by a PREGAME sim quantity (|mean simulated margin|, the engine's own
expected closeness) and check that the simulated OT rate slopes with the actual one. Read on the full-size
served-v2 run (5,710 x 200) and, per arm, on the round-3 500 x 25 runs (sample; quintile cells there are
UNDERPOWERED on the actual side and labelled so).

    .venv/Scripts/python.exe scripts/diag_late_game_r3_resp_v1.py --full v3full_COMB9GCTKD_s200_o0 \
        --runs lg3_R9_s25 lg3_Dt9_s25 --out results/late_game/round3/resp.json
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


def table(tag: str, act: pd.DataFrame, q: int = 5) -> dict:
    g = pd.read_parquet(RES / tag / "games.parquet", columns=["game_id", "home_pts", "away_pts", "n_periods"])
    g["m"] = g["home_pts"] - g["away_pts"]
    g["ot"] = g["n_periods"] > 2
    pg = g.groupby("game_id").agg(mm=("m", "mean"), sim_ot=("ot", "mean"), n=("ot", "size"))
    pg = pg.join(act, how="inner")
    pg["close"] = pg["mm"].abs()
    pg["q"] = pd.qcut(pg["close"].rank(method="first"), q, labels=False)
    rows = []
    for k, x in pg.groupby("q"):
        n_ot = int(x["went_ot"].sum())
        rows.append({"quintile_closest_first": int(k), "n_games": int(len(x)),
                     "abs_mean_sim_margin": float(x["close"].mean()),
                     "sim_ot": float(x["sim_ot"].mean()), "act_ot": float(x["went_ot"].mean()),
                     "act_ot_se": float(np.sqrt(x["went_ot"].mean() * (1 - x["went_ot"].mean()) / len(x))),
                     "act_n_ot": n_ot, "label": "UNDERPOWERED" if n_ot < 30 else ""})
    s = np.array([r["sim_ot"] for r in rows])
    a = np.array([r["act_ot"] for r in rows])
    slope_ratio = float((s[0] - s[-1]) / (a[0] - a[-1])) if a[0] != a[-1] else None
    return {"tag": tag, "rows": rows, "sim_span_q1_minus_q5": float(s[0] - s[-1]),
            "act_span_q1_minus_q5": float(a[0] - a[-1]), "span_ratio": slope_ratio,
            "sim_over_act_by_q": (s / np.where(a > 0, a, np.nan)).round(3).tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=None)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    act = REF.load_actual_games(2025).set_index("game_id")[["went_ot"]].astype(bool)
    out = {"spec": "late_game experiments.md 7.6 (reported)", "tables": []}
    for t in ([a.full] if a.full else []) + a.runs:
        out["tables"].append(table(t, act))
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for t in out["tables"]:
        print(f"== {t['tag']}: span sim {t['sim_span_q1_minus_q5']:+.4f} act {t['act_span_q1_minus_q5']:+.4f} "
              f"ratio {t['span_ratio']}")
        for r in t["rows"]:
            print(f"   q{r['quintile_closest_first']} n={r['n_games']} |m|={r['abs_mean_sim_margin']:.1f} "
                  f"sim {r['sim_ot']:.4f} act {r['act_ot']:.4f} (se {r['act_ot_se']:.4f}, {r['act_n_ot']} OT) {r['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
