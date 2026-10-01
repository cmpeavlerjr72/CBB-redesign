"""grade_clock_r7_offline_v1.py -- clock round 7 OFFLINE grade (experiments.md section 32).

Lane H, 2026-09-30. One grader for every arm (C0, A1, A2, A3), both folds.
Truth: test-season rows of the L2 v4 design, clock-complete games, regulation,
L2's horn censoring. Each arm's own S1 schedule is routed by game date.

Per arm and fold:
  deviance  -2 mean log-lik (log p(y) uncensored, log P(T >= y) censored)
  e         E[min(T, R)] (the quantity loop.py subtracts)
  responsiveness slopes by OFFENCE / DEFENCE team quintile (season mean of the
            as-of off/def_tempo_rel), game-prior quintile; by start type, month, site
  per-game count calibration: 2400/model-mean vs 2400/actual-mean, slope and bias
  start-type and month gaps
Floor: 2 x paired game-block bootstrap SE (200 draws) of arm - C0 per line.
Also: C0 fold-2 identity check against L2's served-round-6 pickles.
Writes results/clock_r7/offline_grade.json and per-row results/clock_r7/rows_<fold>.parquet.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.models import clock as CK  # noqa: E402
from cbb_sim.models import clock_r7  # noqa: E402,F401  (unpickling)
from cbb_sim.models import clock_v3 as c3  # noqa: E402

CKD = ROOT / "data/processed/models/clock"
L2 = CKD / "r6_L2"
OUT = ROOT / "results/clock_r7"
ARMS = ("C0", "A1", "A2", "A3")
CHUNK = 20000
NBOOT = 200
COLS = ["season", "game_id", "period", "poss_index", "offense_team_id", "defense_team_id",
        "offense_is_home", "game_date", "duration_s", "censored", "start_reason", "prev_end",
        "seconds_remaining", "score_diff", "off_tempo_rel", "def_tempo_rel", "tempo_prior_game",
        "site_home", "site_away", "in_bonus", "is_ot", *CK.PREV_END_DUMMIES]


def load_schedule(path: Path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for m in doc["months"]:
        with open(CKD / m["model_file"], "rb") as f:
            out.append((pd.Timestamp(m["refit_date"]), pickle.load(f)))
    return sorted(out, key=lambda r: r[0])


def score(sched, d):
    grid = np.arange(CK.DURATION_CAP + 1, dtype=np.float64)
    cuts = np.array([np.datetime64(r[0], "ns") for r in sched])
    seg = np.searchsorted(cuts, pd.to_datetime(d["game_date"]).to_numpy(), side="right") - 1
    e = np.empty(len(d))
    ll = np.empty(len(d))
    y = np.clip(d["duration_s"].to_numpy().astype(np.int64), 0, CK.DURATION_CAP)
    cen = d["censored"].to_numpy(dtype=bool)
    for k in np.unique(seg):
        rows = np.flatnonzero(seg == k)
        arm = sched[int(k)][1]
        for lo in range(0, len(rows), CHUNK):
            r = rows[lo:lo + CHUNK]
            blk = d.iloc[r].reset_index(drop=True)
            p = np.asarray(arm.pmf(blk), dtype=np.float64)
            R = blk["seconds_remaining"].to_numpy(dtype=np.float64)[:, None]
            e[r] = (p * np.minimum(grid[None, :], R)).sum(axis=1)
            yy = y[r]
            pu = p[np.arange(len(r)), yy]
            S = 1.0 - np.cumsum(p, axis=1)[np.arange(len(r)), yy] + pu
            ll[r] = np.log(np.clip(np.where(cen[r], S, pu), 1e-12, None))
    return e, ll


def qslope(model, actual, q, w=None):
    t = pd.DataFrame({"m": model, "a": actual, "q": q}).groupby("q").mean()
    if len(t) < 2:
        return np.nan, t
    return float(np.polyfit(t["a"], t["m"], 1)[0]), t


def game_cal(d, col):
    g = d.groupby("game_id").agg(m=(col, "mean"), a=("duration_s", "mean"))
    pm, pa = 2400.0 / g["m"], 2400.0 / g["a"]
    return float(np.polyfit(pm, pa, 1)[0]), float((pm - pa).mean())


def main():
    univ = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet")
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    design = pd.read_parquet(L2 / "design_v2.parquet", columns=[c for c in COLS if c != "censored"] + ["censored"])
    design, _ = c3.attach_horn_censoring(design, [2022, 2023, 2024, 2025], censor_dir=L2 / "clock_censoring")
    c3.set_flag_inplace(design, "horn")
    rep = {"floor_rule": "2 x paired game-block bootstrap SE (200 draws); reseed floor 0 by construction",
           "folds": {}}
    for fold, season in (("F2", 2025), ("F1", 2024)):
        d = design[(design["season"] == season) & (design["period"] <= 2.0)
                   & design["game_id"].isin(cc)].reset_index(drop=True)
        side = pd.read_parquet(CKD / f"r7_A3/{fold}/team_re_side.parquet",
                               columns=["game_id", "offense_team_id", "re_o", "re_d"])
        d = d.merge(side, on=["game_id", "offense_team_id"], how="left", validate="m:1")
        d[["re_o", "re_d"]] = d[["re_o", "re_d"]].fillna(0.0)
        d["month"] = pd.to_datetime(d["game_date"]).dt.month
        d["site"] = np.where(d["site_home"] + d["site_away"] == 0, "neutral",
                             np.where(d["site_home"] > 0, "off_home", "off_away"))
        d["oq"] = d["offense_team_id"].map(pd.qcut(d.groupby("offense_team_id")["off_tempo_rel"].mean(), 5, labels=False))
        d["dq"] = d["defense_team_id"].map(pd.qcut(d.groupby("defense_team_id")["def_tempo_rel"].mean(), 5, labels=False))
        d["gq"] = pd.qcut(d["tempo_prior_game"].rank(method="first"), 5, labels=False)
        fr = {"n_rows": int(len(d)), "n_games": int(d["game_id"].nunique()),
              "actual_mean": float(d["duration_s"].mean()), "arms": {}}
        for arm in ARMS:
            e, ll = score(load_schedule(CKD / f"r7_{arm}/{fold}/manifest.json"), d)
            d[f"e_{arm}"] = e
            d[f"ll_{arm}"] = ll
            a = {}
            a["deviance"] = float(-2 * ll.mean())
            a["gap_s"] = float(e.mean() - d["duration_s"].mean())
            a["implied_poss_per_team_game"] = float(1200 / e.mean() - 1200 / d["duration_s"].mean())
            for q in ("oq", "dq", "gq"):
                s, t = qslope(e, d["duration_s"], d[q])
                a[f"slope_{q}"] = s
                a[f"quint_{q}"] = {"model": t["m"].round(3).tolist(), "actual": t["a"].round(3).tolist()}
            for by in ("start_reason", "month", "site"):
                a[f"slope_oq_by_{by}"] = {str(k): qslope(g[f"e_{arm}"], g["duration_s"], g["oq"])[0]
                                          for k, g in d.groupby(by) if len(g) > 3000}
                a[f"slope_dq_by_{by}"] = {str(k): qslope(g[f"e_{arm}"], g["duration_s"], g["dq"])[0]
                                          for k, g in d.groupby(by) if len(g) > 3000}
            st = d.groupby("start_reason").agg(m=(f"e_{arm}", "mean"), a=("duration_s", "mean"))
            a["gap_by_start"] = (st["m"] - st["a"]).round(4).to_dict()
            mo = d.groupby("month").agg(m=(f"e_{arm}", "mean"), a=("duration_s", "mean"))
            a["gap_by_month"] = {str(k): round(v, 4) for k, v in (mo["m"] - mo["a"]).items()}
            a["game_cal_slope"], a["game_cal_bias"] = game_cal(d, f"e_{arm}")
            gp = d.groupby("game_id").agg(m=(f"e_{arm}", "mean"), a=("duration_s", "mean"), gq=("gq", "median"))
            gp["pm"], gp["pa"] = 2400 / gp["m"], 2400 / gp["a"]
            a["game_count_by_gq"] = gp.groupby("gq")[["pm", "pa"]].mean().round(3).to_dict("list")
            fr["arms"][arm] = a
            print(fold, arm, round(a["deviance"], 5), round(a["slope_oq"], 3), round(a["slope_dq"], 3),
                  round(a["slope_gq"], 3), round(a["implied_poss_per_team_game"], 3),
                  round(a["game_cal_slope"], 3), flush=True)
        # paired game-block bootstrap vs C0
        rng = np.random.default_rng(20260930)
        games = d["game_id"].unique()
        gi = pd.factorize(d["game_id"])[0]
        idx_by_game = np.split(np.argsort(gi, kind="stable"), np.cumsum(np.bincount(gi))[:-1])
        boots = {arm: {"ddev": [], "dslope_oq": [], "dslope_dq": [], "dslope_gq": [], "dcal": []}
                 for arm in ARMS[1:]}
        y = d["duration_s"].to_numpy(dtype=float)
        for _ in range(NBOOT):
            pick = rng.integers(0, len(games), len(games))
            rows = np.concatenate([idx_by_game[i] for i in pick])
            sub = d.iloc[rows]
            base = {}
            for arm in ARMS:
                dev = -2 * sub[f"ll_{arm}"].mean()
                so = qslope(sub[f"e_{arm}"].to_numpy(), y[rows], sub["oq"].to_numpy())[0]
                sd = qslope(sub[f"e_{arm}"].to_numpy(), y[rows], sub["dq"].to_numpy())[0]
                sg = qslope(sub[f"e_{arm}"].to_numpy(), y[rows], sub["gq"].to_numpy())[0]
                g = sub.assign(gg=np.repeat(np.arange(len(pick)), [len(idx_by_game[i]) for i in pick]))
                gm = g.groupby("gg").agg(m=(f"e_{arm}", "mean"), a=("duration_s", "mean"))
                cal = float(np.polyfit(2400 / gm["m"], 2400 / gm["a"], 1)[0])
                if arm == "C0":
                    base = {"dev": dev, "so": so, "sd": sd, "sg": sg, "cal": cal}
                else:
                    b = boots[arm]
                    b["ddev"].append(dev - base["dev"])
                    b["dslope_oq"].append(so - base["so"])
                    b["dslope_dq"].append(sd - base["sd"])
                    b["dslope_gq"].append(sg - base["sg"])
                    b["dcal"].append(cal - base["cal"])
        c0 = fr["arms"]["C0"]
        for arm in ARMS[1:]:
            a = fr["arms"][arm]
            b = boots[arm]
            a["vs_C0"] = {
                "ddev": a["deviance"] - c0["deviance"], "ddev_floor": 2 * float(np.std(b["ddev"])),
                "dslope_oq": a["slope_oq"] - c0["slope_oq"], "dslope_oq_floor": 2 * float(np.std(b["dslope_oq"])),
                "dslope_dq": a["slope_dq"] - c0["slope_dq"], "dslope_dq_floor": 2 * float(np.std(b["dslope_dq"])),
                "dslope_gq": a["slope_gq"] - c0["slope_gq"], "dslope_gq_floor": 2 * float(np.std(b["dslope_gq"])),
                "dcal": a["game_cal_slope"] - c0["game_cal_slope"], "dcal_floor": 2 * float(np.std(b["dcal"])),
                "d_abs_count_gap": abs(a["implied_poss_per_team_game"]) - abs(c0["implied_poss_per_team_game"]),
                "max_start_gap_worsening_s": float(max(abs(a["gap_by_start"][k]) - abs(c0["gap_by_start"][k])
                                                       for k in c0["gap_by_start"] if k != "other")),
            }
            print(fold, arm, "vs C0", json.dumps({k: round(v, 4) for k, v in a["vs_C0"].items()}), flush=True)
        if fold == "F2":
            # identity: C0 vs L2's own schedule on the first 20k rows
            from cbb_sim.engine.clock_adapter_v3 import V3C_MODES
            l2 = load_schedule(CKD / V3C_MODES["v3c_r6L2_srfloor_P3_s1"]["manifest"])
            e_l2, ll_l2 = score(l2, d.iloc[:20000].reset_index(drop=True))
            fr["C0_vs_L2_identity_max_abs_e_diff"] = float(np.abs(e_l2 - d["e_C0"].to_numpy()[:20000]).max())
            fr["C0_vs_L2_identity_max_abs_ll_diff"] = float(np.abs(ll_l2 - d["ll_C0"].to_numpy()[:20000]).max())
            print("identity", fr["C0_vs_L2_identity_max_abs_e_diff"], fr["C0_vs_L2_identity_max_abs_ll_diff"])
        keep = ["game_id", "offense_team_id", "defense_team_id", "start_reason", "month", "site", "oq", "dq", "gq",
                "duration_s", "censored", *[f"e_{a}" for a in ARMS], *[f"ll_{a}" for a in ARMS]]
        d[keep].to_parquet(OUT / f"rows_{fold}.parquet", index=False)
        rep["folds"][fold] = fr
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "offline_grade.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
