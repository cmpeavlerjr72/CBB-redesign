#!/usr/bin/env python
"""
grade_foul_joint_v1.py -- ONE blind grading pass over every round-7 offline arm
(possession_outcome experiments.md section 20.2).

Arm-agnostic: every prediction column named `<target>__<arm>` in
round7/preds_{poss,trip}_{F1,F2}_seed{0,7}.parquet is scored the same way. The
grader knows the pre-registered BLOCK sums (A, D/O, T) and the applied floor
rule, nothing about which arm is expected to win.

    .venv/Scripts/python.exe scripts/grade_foul_joint_v1.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data/processed/models/possession_outcome/round7")
APPLIED_FLOOR_MIN = 0.000804          # round 4b, section 10 / 17
MIN_TEAM_ROWS = 300
UNDERPOWERED = 2000


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def arms_of(df, y):
    return [c.split("__", 1)[1] for c in df.columns if c.startswith(f"{y}__")]


def minute_of(df):
    if "game_minute" in df:
        return df["game_minute"].to_numpy()
    per, sc = df["period"].to_numpy(), df["start_clock"].to_numpy()
    return np.where(per == 1, (1200 - sc) / 60.0,
                    np.where(per == 2, (2400 - sc) / 60.0, (2400 + (per - 2) * 300 - sc) / 60.0))


def segments(df):
    m = minute_of(df)
    seg = {
        "half": df["half"].astype(str).to_numpy(),
        "def_fouls_true": pd.cut(df["def_f_t"], [-1, 3, 5, 6, 8, 9, 99],
                                 labels=["0-3", "4-5", "6", "7-8", "9", "10+"]).astype(str).to_numpy(),
        "minute": pd.cut(m, [0, 5, 10, 15, 20, 25, 30, 35, 38, 40.01, 99], right=False,
                         labels=["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
                                 "35-37", "38-40", "OT"]).astype(str),
        "site": np.where(df["site_home"] > 0, "home", np.where(df["site_away"] > 0, "away", "neutral")),
        "month": pd.to_datetime(df["game_date"]).dt.month.astype(str).to_numpy(),
    }
    if "is_conf_game" in df and df["is_conf_game"].notna().all():
        seg["conf"] = np.where(df["is_conf_game"] > 0, "conf", "nonconf")
    return seg


def team_stats(df, y, arm, team_col):
    g = df.groupby(team_col).agg(n=(y, "size"), act=(y, "mean"), pred=(f"{y}__{arm}", "mean"))
    g = g[g["n"] >= MIN_TEAM_ROWS]
    if len(g) < 20:
        return {"n_teams": int(len(g)), "status": "UNDERPOWERED"}
    return {"n_teams": int(len(g)), "sd_ratio": float(g["pred"].std() / g["act"].std()),
            "corr": float(np.corrcoef(g["pred"], g["act"])[0, 1]),
            "sd_actual_pp": float(100 * g["act"].std()), "sd_pred_pp": float(100 * g["pred"].std())}


def quintile_slope(df, prior, y, arm, team_col):
    """Teams bucketed by their PRIOR-season actual rate of the same target."""
    g = df.groupby(team_col).agg(n=(y, "size"), act=(y, "mean"), pred=(f"{y}__{arm}", "mean"))
    g = g[g["n"] >= MIN_TEAM_ROWS].join(prior.rename("prior"), how="inner")
    if len(g) < 50:
        return {"status": "UNDERPOWERED", "n_teams": int(len(g))}
    g["q"] = pd.qcut(g["prior"], 5, labels=False)
    q = g.groupby("q")[["act", "pred"]].mean()
    span_a, span_p = q["act"].iloc[-1] - q["act"].iloc[0], q["pred"].iloc[-1] - q["pred"].iloc[0]
    return {"n_teams": int(len(g)), "slope_ratio": float(span_p / span_a) if span_a else None,
            "monotone_pred": int((np.diff(q["pred"].to_numpy()) > 0).sum())}


BLOCKS = {  # block -> (file kind, [(target, team column)], arm-name map: block arm -> per-target arm)
    "A": ("poss", [("y_nt", "defense_team_id")]),
    "DO": ("poss", [("y_dnt", "defense_team_id"), ("y_off", "offense_team_id")]),
    "T": ("trip", [("y_bonus", "offense_team_id"), ("y_shoot", "offense_team_id")]),
}
DO_ARMS = {"DO0": {"y_dnt": "D0", "y_off": "O0"}, "DO2": {"y_dnt": "D2", "y_off": "O2"}}


def block_arms(block, df):
    if block == "DO":
        return DO_ARMS
    tg = BLOCKS[block][1]
    names = set(arms_of(df, tg[0][0]))
    for y, _ in tg[1:]:
        names &= set(arms_of(df, y))
    return {a: {y: a for y, _ in tg} for a in sorted(names)}


def main() -> None:
    out: dict = {"applied_floor_min": APPLIED_FLOOR_MIN, "blocks": {}}
    for block, (kind, targets) in BLOCKS.items():
        res = {}
        for fold in ("F1", "F2"):
            dfs = {s: pd.read_parquet(D / f"preds_{kind}_{fold}_seed{s}.parquet") for s in (0, 7)}
            prim = {s: d[d["in_fit_window"] == 1] for s, d in dfs.items()}
            arms = block_arms(block, dfs[0])
            fr = {}
            for arm, per_t in arms.items():
                tot = {s: sum(ll(prim[s][y], prim[s][f"{y}__{per_t[y]}"]) for y, _ in targets)
                       for s in (0, 7)}
                floor_m = abs(tot[0] - tot[7])
                r = {"logloss": tot[0], "logloss_seed7": tot[7], "floor_measured": floor_m,
                     "floor_applied": max(floor_m, APPLIED_FLOOR_MIN), "per_target": {}}
                for y, team_col in targets:
                    a = per_t[y]
                    d0 = prim[0]
                    t = {"logloss": ll(d0[y], d0[f"{y}__{a}"]),
                         "calib_gap_pp": float(100 * (d0[f"{y}__{a}"].mean() - d0[y].mean())),
                         "mean_actual": float(d0[y].mean()), "n": int(len(d0))}
                    seg = segments(d0)
                    t["segments"] = {}
                    for sname, sv in seg.items():
                        g = pd.DataFrame({"s": sv, "y": d0[y].to_numpy(), "p": d0[f"{y}__{a}"].to_numpy()}
                                         ).groupby("s").agg(n=("y", "size"), act=("y", "mean"), pred=("p", "mean"))
                        t["segments"][sname] = {
                            str(k): {"n": int(v.n), "gap_pp": round(100 * (v.pred - v.act), 3),
                                     "underpowered": bool(v.n < UNDERPOWERED)}
                            for k, v in g.iterrows()}
                    t["team"] = team_stats(d0, y, a, team_col)
                    if fold == "F2":
                        prior_df = pd.read_parquet(D / f"preds_{kind}_F1_seed0.parquet")
                        prior = prior_df[prior_df["in_fit_window"] == 1].groupby(team_col)[y].mean()
                        t["quintile_slope"] = quintile_slope(d0, prior, y, a, team_col)
                    else:
                        t["quintile_slope"] = {"status": "NOT COMPUTED -- prior season 2023 is not in the preds"}
                    r["per_target"][y] = t
                fr[arm] = r
            res[fold] = fr
        out["blocks"][block] = res
    # verdicts vs the block reference, in applied floors (pre-registered order)
    ref = {"A": "A0", "DO": "DO0", "T": "T0"}
    order = {"A": ["A0", "A1", "A2lab", "A2", "A2_D9a", "A2_D9b"], "DO": ["DO0", "DO2"],
             "T": ["T0", "T1c", "T2lab", "T2c"]}
    ver = {}
    for block, res in out["blocks"].items():
        ver[block] = {}
        for fold in ("F1", "F2"):
            r0 = res[fold][ref[block]]["logloss"]
            ver[block][fold] = {
                a: {"delta_vs_ref": res[fold][a]["logloss"] - r0,
                    "floors_vs_ref": (r0 - res[fold][a]["logloss"]) / res[fold][a]["floor_applied"]}
                for a in order[block] if a in res[fold]}
        # winner on F2: best arm that beats every SIMPLER arm by > 1 applied floor
        f2 = res["F2"]
        cand = [a for a in order[block] if a in f2]
        win = cand[0]
        for a in cand[1:]:
            if (f2[win]["logloss"] - f2[a]["logloss"]) > f2[a]["floor_applied"]:
                win = a
        f1 = res["F1"]
        confirmed = (win == ref[block]) or (f1[win]["logloss"] < f1[ref[block]]["logloss"])
        ver[block]["F2_winner"] = win
        ver[block]["F1_confirms"] = bool(confirmed)
    out["verdicts"] = ver
    (D / "grade_round7.json").write_text(json.dumps(out, indent=1, default=float))
    for block in ver:
        print(f"\n== block {block}: F2 winner {ver[block]['F2_winner']}  F1 confirms {ver[block]['F1_confirms']}")
        for fold in ("F2", "F1"):
            for a, v in ver[block][fold].items():
                r = out["blocks"][block][fold][a]
                pt = r["per_target"]
                team = "; ".join(f"{y}: sd_ratio {t['team'].get('sd_ratio', float('nan')):.3f} "
                                 f"slope {t.get('quintile_slope', {}).get('slope_ratio', float('nan')) if isinstance(t.get('quintile_slope', {}).get('slope_ratio'), float) else 'NA'} "
                                 f"cal {t['calib_gap_pp']:+.3f}pp" for y, t in pt.items())
                print(f"  {fold} {a:8s} ll {r['logloss']:.7f}  d {v['delta_vs_ref']:+.7f}  "
                      f"floors {v['floors_vs_ref']:+8.2f}  floor_meas {r['floor_measured']:.2e}  {team}")
    print(f"\nwrote {D / 'grade_round7.json'}")


if __name__ == "__main__":
    main()
