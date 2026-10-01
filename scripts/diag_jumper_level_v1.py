"""diag_jumper_level_v1.py -- lane A day round 4 (fg_make/experiments.md s29): owner diagnosis of the jumper level
drift (Nov under / Mar over, overall +0.5 pp in TS-R). DIAGNOSTIC ONLY.

T1 league level: per season x month, jumper (and rim, three) realised league make rate against the attempt-weighted
   served as-of league rate (`lg_make_asof`) and the round-4 alternatives (roll28, seasonal) -- the as-of LAG /
   within-season seasonality hypothesis. Training seasons and 2024-25.
T2 tree re-learning: on 2024-25 test rows, monthly mean of (stage-A prediction - offset-only prediction) for TSR
   (p_zero vs lg_make_asof) and for aL (p vs lg_make_asof) -- what the trees add by month on top of the league level.
T3 within-type mix: the design has no shot location, so long-two vs short-mid-range mix is UNTESTABLE; proxies by
   month: transition share, putback share (chance_number > 1), shot-clock-late share (chance_elapsed_s >= 25).

    .venv/Scripts/python.exe scripts/diag_jumper_level_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import train_fg_make_two_stage_v1 as V1  # noqa: E402
import train_fg_make_two_stage_v3 as V3  # noqa: E402

OUT = ROOT / "results/g9_team_response_v1"


def main() -> int:
    d = pd.read_parquet(V1.DESIGN, columns=["season", "game_date", "shot_class", "y", "lg_make_asof",
                                            "is_transition_f", "chance_number", "chance_elapsed_s"])
    for mode in ("roll28", "seasonal"):
        v, _ = V3.league_level(d, mode, [2022, 2023, 2024])
        d[f"lg_{mode}"] = v
    d["mon"] = pd.to_datetime(d["game_date"]).dt.month
    res = {"T1": {}, "T3": {}}
    for (cls, s, m), g in d.groupby(["shot_class", "season", "mon"]):
        if len(g) < 2000:
            continue
        res["T1"][f"{cls}|{s}|{m}"] = {"n": int(len(g)), "actual": float(g["y"].mean()),
                                        "asof": float(g["lg_make_asof"].mean()), "roll28": float(g["lg_roll28"].mean()),
                                        "seasonal": float(g["lg_seasonal"].mean())}
        if cls == "FGA_jump2":
            res["T3"][f"{s}|{m}"] = {"transition": float(g["is_transition_f"].mean()),
                                     "putback": float((g["chance_number"] > 1).mean()),
                                     "late_clock": float((g["chance_elapsed_s"] >= 25).mean())}
    res["T2"] = {}
    for name, path, pcol in (("TSR_stageA", "round_tsr/TSR_s0_F2/preds_F2_s0.parquet", "p_zero"),
                             ("aL", "round_tsr/aL_s0_F2/preds_F2_s0.parquet", "p"),
                             ("served_aR", "round_g9/aR_s0_F2/preds_F2_s0.parquet", "p")):
        p = pd.read_parquet(ROOT / "data/processed/models/fg_make" / path)
        lg = d.loc[d["season"] == 2025, ["game_date", "shot_class", "lg_make_asof"]].drop_duplicates(
            ["game_date", "shot_class"])
        p["game_date"] = pd.to_datetime(p["game_date"])
        lg["game_date"] = pd.to_datetime(lg["game_date"])
        p = p.drop(columns=[c for c in ("lg_make_asof",) if c in p]).merge(lg, on=["game_date", "shot_class"],
                                                                          how="left")
        p["mon"] = p["game_date"].dt.month
        for (cls, m), g in p.groupby(["shot_class", "mon"]):
            if len(g) < 2000:
                continue
            res["T2"][f"{name}|{cls}|{m}"] = {"bias_pred": float((g[pcol] - g["y"]).mean()),
                                               "bias_offset_only": float((g["lg_make_asof"] - g["y"]).mean()),
                                               "tree_part": float((g[pcol] - g["lg_make_asof"]).mean())}
    (OUT / "diag_jumper_level_v1.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    t1 = pd.DataFrame(res["T1"]).T
    t1["asof_minus_act_pp"] = (t1["asof"] - t1["actual"]) * 100
    t1["roll_minus_act_pp"] = (t1["roll28"] - t1["actual"]) * 100
    t1["seas_minus_act_pp"] = (t1["seasonal"] - t1["actual"]) * 100
    print(t1[[c for c in t1 if c.endswith("_pp")]].round(2).to_string())
    print(pd.DataFrame(res["T2"]).T.mul(100).round(2).to_string())
    print(pd.DataFrame(res["T3"]).T.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
