#!/usr/bin/env python
"""
grade_shot_block_closed_loop_v1.py -- the drawn-block-flag paired closed loop,
graded blind (`docs/models/shot_block/experiments.md` section 5.4). Every arm
goes through the same functions; the arm list is the CLI, the FIRST tag is the
served reference, floors are |po4b_R_s25_floor - po4b_R_s25| per line.

Gate lines are computed with `cbb_sim.eval.gates` (the functions
`scripts/eval_gates.py` calls, with `docs/gates.yaml`), parsed from their own
check rows, so the numbers are the gate report's own; nothing is re-derived.

    .venv/Scripts/python.exe scripts/grade_shot_block_closed_loop_v1.py po4b_R_s25 sb_K2O_s25 sb_K2_s25
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.eval import contract as C  # noqa: E402
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval.cli import load_tolerances  # noqa: E402

R = ROOT / "results/engine_v0"
FLOOR_A, FLOOR_B = "po4b_R_s25_floor", "po4b_R_s25"
NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
#: (gate, quantity regex) -> short name; the veto and primary lines of section 5.4
LINES = {
    ("G1", r"possessions/game mean"): "G1_poss_mean",
    ("G1", r"possessions/game SD"): "G1_poss_sd",
    ("G4", r"oreb_pct \(season, pooled"): "G4_oreb",
    ("G4", r"efg_pct \(season, pooled"): "G4_efg",
    ("G4", r"tov_pct \(season, pooled"): "G4_tov",
    ("G4", r"ft_rate \(season, pooled"): "G4_ft_rate",
    ("G5", r"margin SD ratio"): "G5_margin_ratio",
    ("G5", r"total SD ratio"): "G5_total_ratio",
    ("G5", r"home/away score corr"): "G5_corr",
    ("G9", r"margin bias"): "G9_margin_bias",
    ("G9", r"total bias"): "G9_total_bias",
    ("G9", r"calibration slope"): "G9_slope",
}
VETO = ["G1_poss_mean", "G1_poss_sd", "G5_margin_ratio", "G5_total_ratio", "G5_corr",
        "G9_margin_bias", "G9_slope", "G4_efg", "G4_tov", "G4_ft_rate"]


def gate_lines(tag: str) -> dict:
    tol = load_tolerances(ROOT / "docs/gates.yaml")
    mcn = int(tol.get("min_cell_n", G.DEFAULT_MIN_CELL_N))
    eng = C.load_engine_results(str(R / tag))
    summ, raw = G.build_grading_frame(eng.games, 2025)
    td = str(ROOT / "data/processed/truth")
    res = [G.gate_g1(summ, raw, 2025, tol, mcn),
           G.gate_g4(summ, raw, 2025, tol, eng.box_available, mcn, truth_dir=td),
           G.gate_g5(summ, raw, 2025, tol), G.gate_g9(summ, tol, mcn)]
    out = {}
    for gr in res:
        for c in gr.checks:
            for (gate, pat), name in LINES.items():
                if gr.gate == gate and re.search(pat, c.quantity) and name not in out:
                    v, t = NUM.findall(str(c.value)), NUM.findall(str(c.target))
                    if v:
                        out[name] = {"value": float(v[0]), "target": float(t[0]) if t else None,
                                     "status": c.status, "quantity": c.quantity}
    return out


def team_oreb(tag: str, gmap: pd.DataFrame, box: pd.DataFrame, prior: pd.Series) -> dict:
    g = pd.read_parquet(R / tag / "games.parquet").merge(gmap, on="game_id")
    rows = []
    for s, o in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({"team": g[f"{s}_team_id"], "oreb": g[f"{s}_oreb"],
                                  "opp_dreb": g[f"{o}_dreb"], "dreb": g[f"{s}_dreb"],
                                  "opp_oreb": g[f"{o}_oreb"]}))
    t = pd.concat(rows).groupby("team").sum()
    sim_off = t["oreb"] / (t["oreb"] + t["opp_dreb"])
    sim_def = t["opp_oreb"] / (t["opp_oreb"] + t["dreb"])
    b = box[box["game_id"].isin(g["game_id"].unique())]
    opp = b[["game_id", "team_id", "offensive_rebounds", "defensive_rebounds"]].merge(
        b[["game_id", "team_id", "offensive_rebounds", "defensive_rebounds"]], on="game_id",
        suffixes=("", "_o"))
    opp = opp[opp["team_id"] != opp["team_id_o"]].groupby("team_id").sum(numeric_only=True)
    act_off = opp["offensive_rebounds"] / (opp["offensive_rebounds"] + opp["defensive_rebounds_o"])
    act_def = opp["offensive_rebounds_o"] / (opp["offensive_rebounds_o"] + opp["defensive_rebounds"])
    d = pd.DataFrame({"sim_off": sim_off, "act_off": act_off, "sim_def": sim_def,
                      "act_def": act_def}).dropna()
    q = d.join(prior.rename("prior"), how="inner")
    q["q"] = pd.qcut(q["prior"].rank(method="first"), 5, labels=False)
    qq = q.groupby("q")[["sim_off", "act_off"]].mean()
    return {"n_teams": int(len(d)),
            "off_mean_gap_pp": float(100 * (d["sim_off"] - d["act_off"]).mean()),
            "off_mae_pp": float(100 * (d["sim_off"] - d["act_off"]).abs().mean()),
            "def_mean_gap_pp": float(100 * (d["sim_def"] - d["act_def"]).mean()),
            "def_mae_pp": float(100 * (d["sim_def"] - d["act_def"]).abs().mean()),
            "off_slope_ratio": float((qq["sim_off"].iloc[-1] - qq["sim_off"].iloc[0])
                                     / (qq["act_off"].iloc[-1] - qq["act_off"].iloc[0])),
            "off_pred_by_q": [round(float(x), 4) for x in qq["sim_off"]],
            "off_act_by_q": [round(float(x), 4) for x in qq["act_off"]]}


def mechanism_sim(tag: str) -> dict:
    d = json.loads((R / tag / "run_meta.json").read_text())["diagnostics"]
    out = {}
    for t in ("rim", "jump2", "three"):
        if f"sb_{t}_blk_n" not in d:
            return {}
        b, u = d[f"sb_{t}_blk_n"], d[f"sb_{t}_unblk_n"]
        ob, db_ = d[f"sb_{t}_blk_oreb"], d[f"sb_{t}_blk_dreb"]
        ou, du = d[f"sb_{t}_unblk_oreb"], d[f"sb_{t}_unblk_dreb"]
        out[t] = {"blocked_share": b / max(b + u, 1), "oreb_blk": ob / max(ob + db_, 1),
                  "oreb_unblk": ou / max(ou + du, 1), "n": b + u}
    tb = sum(d[f"sb_{t}_blk_n"] for t in ("rim", "jump2", "three"))
    tn = sum(out[t]["n"] for t in out)
    out["all_fga"] = {"blocked_share": tb / tn}
    return out


def mechanism_actual(game_ids) -> dict:
    d = pd.read_parquet(ROOT / "data/processed/models/rebound/round3/design_round3.parquet",
                        columns=["game_id", "miss_type", "blocked", "y"])
    d = d[d["game_id"].isin(game_ids) & d["miss_type"].isin(["rim", "jump2", "three"])]
    from cbb_sim.models import rebound as RB
    out = {}
    for t, x in d.groupby("miss_type"):
        live = x[x["y"] != RB.CLASS_INDEX["DEAD"]]
        b = live["blocked"].astype(bool)
        o = live["y"] == RB.CLASS_INDEX["OREB"]
        out[t] = {"blocked_share": float(x["blocked"].mean()), "oreb_blk": float(o[b].mean()),
                  "oreb_unblk": float(o[~b].mean()), "n": int(len(x))}
    out["all_fga"] = {"blocked_share": float(d["blocked"].mean())}
    return out


def main() -> None:
    tags = sys.argv[1:]
    ref = tags[0]
    gl = {t: gate_lines(t) for t in dict.fromkeys([FLOOR_A, FLOOR_B, *tags])}
    floors = {k: abs(gl[FLOOR_A][k]["value"] - gl[FLOOR_B][k]["value"]) for k in gl[FLOOR_B]}
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(ROOT / "data/processed/models/engine", "F2_2025")
    gmap = inp.games[["game_id", "home_team_id", "away_team_id"]]
    box = pd.read_parquet(ROOT / "data/raw/hoopr/team_box/team_box_2025.parquet")
    b24 = pd.read_parquet(ROOT / "data/raw/hoopr/team_box/team_box_2024.parquet")
    o24 = b24[["game_id", "team_id", "offensive_rebounds", "defensive_rebounds"]]
    o24 = o24.merge(o24, on="game_id", suffixes=("", "_o"))
    o24 = o24[o24["team_id"] != o24["team_id_o"]].groupby("team_id").sum(numeric_only=True)
    prior = o24["offensive_rebounds"] / (o24["offensive_rebounds"] + o24["defensive_rebounds_o"])
    team = {t: team_oreb(t, gmap, box, prior) for t in dict.fromkeys([FLOOR_A, FLOOR_B, *tags])}
    tfl = {k: abs(team[FLOOR_A][k] - team[FLOOR_B][k]) for k in team[FLOOR_B]
           if isinstance(team[FLOOR_B][k], float)}
    gids = pd.read_parquet(R / ref / "games.parquet", columns=["game_id"])["game_id"].unique()
    out = {"floors": floors, "team_floors": tfl, "mech_actual": mechanism_actual(gids), "arms": {}}
    R0 = gl[ref]
    for t in tags:
        L = gl[t]
        rows = {}
        for k, v in L.items():
            fl = floors.get(k) or np.nan
            tgt = v["target"] if v["target"] is not None else (1.0 if "ratio" in k or "slope" in k else 0.0)
            d0, d1 = abs(R0[k]["value"] - tgt), abs(v["value"] - tgt)
            rows[k] = {"value": v["value"], "target": tgt, "status": v["status"],
                       "delta": v["value"] - R0[k]["value"], "floor": fl,
                       "floors_toward": (d0 - d1) / fl if fl and fl > 0 else None,
                       "flip": R0[k]["status"] != v["status"]}
        tm = team[t]
        tm_rows = {k: {"value": tm[k], "delta": tm[k] - team[ref][k],
                       "floors": (tm[k] - team[ref][k]) / tfl[k] if tfl.get(k) else None}
                   for k in tm if isinstance(tm[k], float)}
        veto = {k: bool(rows[k]["floors_toward"] is not None and rows[k]["floors_toward"] < -1.0)
                or (rows[k]["flip"] and rows[k]["status"] == "FAIL") for k in VETO if k in rows}
        slope_fall = bool(tm_rows["off_slope_ratio"]["floors"] is not None
                          and tm_rows["off_slope_ratio"]["floors"] < -1.0)
        prim = rows["G4_oreb"]["floors_toward"]
        out["arms"][t] = {"lines": rows, "team": tm_rows, "team_raw": tm, "vetoes": veto,
                          "team_slope_falls": slope_fall,
                          "primary_pass": bool(prim is not None and prim > 1.0),
                          "put_forward": bool(t != ref and prim is not None and prim > 1.0
                                              and not any(veto.values()) and not slope_fall),
                          "mech_sim": mechanism_sim(t)}
    Path(ROOT / "results/shot_block_round2").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/shot_block_round2/closed_loop_grade_v1.json").write_text(
        json.dumps(out, indent=1, default=float))
    print("floors:", {k: round(v, 5) for k, v in floors.items()})
    print("team floors:", {k: round(v, 4) for k, v in tfl.items()})
    print("actual mechanism:", json.dumps(out["mech_actual"], default=float))
    for t, a in out["arms"].items():
        print(f"\n== {t}  put_forward={a['put_forward']}  primary_pass={a['primary_pass']}  "
              f"vetoes={[k for k, v in a['vetoes'].items() if v]}  team_slope_falls={a['team_slope_falls']}")
        for k, v in a["lines"].items():
            ft = v["floors_toward"]
            print(f"   {k:16s} {v['value']:.4f} (tgt {v['target']:.4f}, {v['status']}) delta {v['delta']:+.4f} "
                  f"floor {v['floor']:.4f} toward {'' if ft is None else f'{ft:+.2f}'}")
        for k, v in a["team"].items():
            print(f"   team {k:16s} {v['value']:.4f} delta {v['delta']:+.4f} floors "
                  f"{'' if v['floors'] is None else f'{v['floors']:+.2f}'}")
        if a["mech_sim"]:
            print("   mechanism sim:", json.dumps(a["mech_sim"], default=float))


if __name__ == "__main__":
    main()
