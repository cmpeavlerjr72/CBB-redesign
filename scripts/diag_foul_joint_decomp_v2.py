"""diag_foul_joint_decomp_v2.py -- v1 with the ACTUAL state on the ENGINE definition
(def_team_fouls_true = count before the possession's own pre-open fouls, from
foul_accrual_poss_v2). v1 docstring follows.

step 1 of the joint foul round: where does
the engine under-produce FT trips and team fouls, served vs actual.

Reads the per-possession tap written by `run_foul_joint_tap_v1.py` and builds
the SAME per-possession quantities for the actual 2025 season from
  data/processed/possessions_v2/chances_2025.parquet      (trips, FTA, FGA)
  data/processed/models/possession_outcome/round6/foul_accrual_poss_v1.parquet
                                                           (replayed foul counts)
Units are per POSSESSION, so the engine's +2 possessions/game cancel.

Identities (each closes by construction):
  total fouls / poss = trip fouls / poss + non-trip fouls / poss
  trip fouls = shooting-foul trips + bonus trips + and-ones
  FTA/FGA(bucket) = FTA/poss / FGA/poss

    diag_foul_joint_decomp_v1.py OUT_JSON TAG [TAG ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results/engine_v0")
BUCKETS = [0, 5, 10, 15, 20, 25, 30, 35, 38, 40.001]
BLAB = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-37", "38-40"]
FBUCK = [-1, 3, 5, 6, 8, 9, 99]
FLAB = ["0-3", "4-5", "6 (bonus 1-1)", "7-8", "9 (dbl)", "10+"]


def actual(sub_game_ids=None) -> pd.DataFrame:
    c = pd.read_parquet("data/processed/possessions_v2/chances_2025.parquet")
    c["fga"] = c["fga_rim"] + c["fga_jump2"] + c["fga_3"]
    c["sh"] = (c["terminal_event"] == "FT_trip_shooting").astype(int)
    c["bo"] = (c["terminal_event"] == "FT_trip_bonus").astype(int)
    c["ao"] = c["and_one"].astype(int)
    k = ["game_id", "period", "poss_index"]
    p = c.groupby(k).agg(fta=("fta", "sum"), fga=("fga", "sum"), sh=("sh", "sum"),
                         bo=("bo", "sum"), ao=("ao", "sum"),
                         n_chances=("fta", "size")).reset_index()
    d = pd.read_parquet("data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet",
                        columns=["game_id", "season", "period", "poss_index", "start_clock",
                                 "start_score_diff", "def_team_fouls_true", "off_team_fouls_true",
                                 "def_silent", "def_trip", "off_silent", "off_trip",
                                 "off_in_bonus", "offense_is_home", "neutral_site",
                                 "offense_team_id", "defense_team_id"])
    d = d[d["season"] == 2025]
    m = d.merge(p, on=k, how="inner", validate="one_to_one")
    if sub_game_ids is not None:
        m = m[m["game_id"].isin(sub_game_ids)]
    out = pd.DataFrame({
        "game_id": m["game_id"], "period": m["period"], "sec": m["start_clock"],
        "def_fouls": m["def_team_fouls_true"], "off_fouls": m["off_team_fouls_true"],
        "in_bonus": (m["def_team_fouls_true"] >= 6).astype(int),
        "n_shoot_trip": m["sh"], "n_bonus_trip": m["bo"], "n_and_one": m["ao"],
        "fta": m["fta"], "fga": m["fga"], "n_chances": m["n_chances"],
        "total_fouls": m["def_silent"] + m["def_trip"] + m["off_silent"] + m["off_trip"],
        "is_home_off": m["offense_is_home"].astype(int), "neutral": m["neutral_site"].astype(int),
    })
    out["trip_fouls"] = out["n_shoot_trip"] + out["n_bonus_trip"] + out["n_and_one"]
    out["nontrip_fouls"] = out["total_fouls"] - out["trip_fouls"]
    return out


def sim(tag: str) -> pd.DataFrame:
    t = pd.read_parquet(R / tag / "poss_tap.parquet")
    t["in_bonus"] = (t["def_fouls"] >= t["bonus_thr"]).astype(int)
    t["n_and_one"] = t["trip_fouls"] - t["n_shoot_trip"] - t["n_bonus_trip"]
    t["nontrip_fouls"] = t["silent"]
    t["total_fouls"] = t["trip_fouls"] + t["silent"] + (t["off_foul"] if "off_foul" in t else 0)
    t["nontrip_fouls"] = t["total_fouls"] - t["trip_fouls"]
    return t


def add_keys(d: pd.DataFrame) -> pd.DataFrame:
    reg = d["period"] <= 2
    gm = np.where(reg, (d["period"] - 1) * 20 + (1200 - d["sec"]) / 60.0, np.nan)
    d = d.assign(minute_b=pd.cut(gm, BUCKETS, right=False, labels=BLAB),
                 half=np.where(d["period"] >= 3, "OT", np.where(d["period"] == 1, "H1", "H2")),
                 final2=(d["period"] == 2) & (d["sec"] <= 120),
                 foul_b=pd.cut(d["def_fouls"], FBUCK, labels=FLAB))
    return d


COLS = ["in_bonus", "total_fouls", "trip_fouls", "nontrip_fouls", "n_shoot_trip",
        "n_bonus_trip", "n_and_one", "fta", "fga"]


def table(d: pd.DataFrame, by: str) -> pd.DataFrame:
    g = d.groupby(by, observed=True)
    t = g[COLS].mean()
    t["fta_fga"] = g["fta"].sum() / g["fga"].sum()
    t["n"] = g.size()
    return t


def main() -> None:
    out_json = Path(sys.argv[1])
    tags = sys.argv[2:]
    sub_ids = pd.read_parquet(R / tags[0] / "poss_tap.parquet", columns=["game_id"])["game_id"].unique()
    A_all = add_keys(actual())
    A_sub = add_keys(actual(sub_ids))
    res: dict = {"actual_all_n_poss": int(len(A_all)), "actual_sub_n_poss": int(len(A_sub))}
    frames = {"ACT_all": A_all, "ACT_sub": A_sub}
    for t in tags:
        frames[t] = add_keys(sim(t))
    pd.set_option("display.width", 250)
    for by in ("minute_b", "half", "final2", "foul_b"):
        res[by] = {}
        for name, d in frames.items():
            tb = table(d, by)
            res[by][name] = json.loads(tb.reset_index().astype({by: str}).to_json(orient="records"))
            print(f"\n=== {by} :: {name} ===")
            print(tb.round(4).to_string())
    # overall
    res["overall"] = {}
    for name, d in frames.items():
        o = d[COLS].mean().to_dict()
        o["fta_fga"] = float(d["fta"].sum() / d["fga"].sum())
        o["n"] = int(len(d))
        res["overall"][name] = o
    print("\n=== overall ===")
    print(pd.DataFrame(res["overall"]).round(5).to_string())
    # conditional trip rate by (half, foul bucket): P(bonus trip)/poss, P(shoot trip)/poss
    res["half_x_foul"] = {}
    for name, d in frames.items():
        tb = d.groupby(["half", "foul_b"], observed=True)[
            ["n_bonus_trip", "n_shoot_trip", "n_and_one", "nontrip_fouls", "total_fouls"]].mean()
        tb["n"] = d.groupby(["half", "foul_b"], observed=True).size()
        res["half_x_foul"][name] = json.loads(tb.reset_index().astype({"foul_b": str}).to_json(orient="records"))
        print(f"\n=== half x def-foul bucket :: {name} ===")
        print(tb.round(4).to_string())
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
