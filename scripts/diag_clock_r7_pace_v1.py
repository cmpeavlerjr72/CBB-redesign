"""diag_clock_r7_pace_v1.py -- clock round 7, step 1: why is team pace responsiveness 0.38?

Lane H, overnight 2026-09-30. Read-only diagnostic on the event-layer v4 clock
design (`data/processed/models/clock/r6_L2/design_v2.parquet`, the L2 design)
and the L2 S1 schedule. For the test seasons 2024 (fold 1 test, L2 served only
for 2025 so 2024 uses the feature-level checks only) and 2025 (fold 2 test):

  1. Model E[min(T,R)] per possession under L2 (routed by game date).
  2. Responsiveness by OFFENCE team quintile (team's season mean of the as-of
     off_tempo_rel) and by DEFENCE team quintile (def_tempo_rel), overall and
     by start type, month, site. Slope = OLS of the 5 model means on the 5
     actual means; span ratio = model span / actual span.
  3. Where does the compression come from:
     a. the encoding: tempo enters only as a tercile of the GAME prior
        (off_rel * def_rel * league mean), the last cell dimension;
     b. offence/defence asymmetry: actual offensive-possession seconds per
        team vs defensive-possession seconds per team (who controls duration);
     c. feature noise: correlation of the as-of off_tempo_rel with the team's
        own end-of-season realised duration, by month;
     d. residual regression: (actual - model) on log(off_tempo_rel),
        log(def_tempo_rel) by start type. Non-zero coefficients = signal the
        law leaves unspent.
     e. the ceiling: a leave-one-game-out team mean (oracle, not usable) to
        size how much of the team spread is persistent.
Writes results/clock_r7/diag.json and prints the tables.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.engine.clock_adapter_v3 import CK_DIR as _CK, V3C_MODES  # noqa: E402
from cbb_sim.models import clock as CK  # noqa: E402

CKD = ROOT / _CK
OUT = ROOT / "results/clock_r7/diag.json"
L2 = "v3c_r6L2_srfloor_P3_s1"
CHUNK = 20000
COLS = ["season", "game_id", "period", "offense_team_id", "defense_team_id", "offense_is_home",
        "game_date", "duration_s", "censored", "start_reason", "prev_end", "seconds_remaining",
        "score_diff", "off_tempo_rel", "def_tempo_rel", "tempo_prior_game", "site_home",
        "site_away", "in_bonus", "is_ot"]


def load_schedule(mode):
    doc = json.loads((CKD / V3C_MODES[mode]["manifest"]).read_text(encoding="utf-8"))
    out = []
    for m in doc["months"]:
        with open(CKD / m["model_file"], "rb") as f:
            out.append({"refit_date": pd.Timestamp(m["refit_date"]), "arm": pickle.load(f)})
    return sorted(out, key=lambda r: r["refit_date"])


def e_consumed(arm, df):
    grid = np.arange(CK.DURATION_CAP + 1, dtype=np.float64)
    out = np.empty(len(df))
    for lo in range(0, len(df), CHUNK):
        blk = df.iloc[lo:lo + CHUNK].reset_index(drop=True)
        pmf = np.asarray(arm.pmf(blk), dtype=np.float64)
        r = blk["seconds_remaining"].to_numpy(dtype=np.float64)[:, None]
        out[lo:lo + len(blk)] = (pmf * np.minimum(grid[None, :], r)).sum(axis=1)
    return out


def resp(d, qcol, by=None):
    """Quintile table and slope/span of model on actual."""
    def one(x):
        t = x.groupby(qcol).agg(n=("e", "size"), model=("e", "mean"), actual=("duration_s", "mean"))
        if len(t) < 5:
            return {"n": int(len(x)), "slope": None}
        span_a = float(t["actual"].iloc[-1] - t["actual"].iloc[0])
        span_m = float(t["model"].iloc[-1] - t["model"].iloc[0])
        return {"n": int(len(x)), "slope": float(np.polyfit(t["actual"], t["model"], 1)[0]),
                "span_ratio": span_m / span_a if span_a else None,
                "actual_q": [round(v, 3) for v in t["actual"]],
                "model_q": [round(v, 3) for v in t["model"]]}
    if by is None:
        return one(d)
    return {str(k): one(g) for k, g in d.groupby(by)}


def main():
    univ = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet")
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    d = pd.read_parquet(CKD / "r6_L2/design_v2.parquet", columns=COLS)
    d = d[(d["period"] <= 2.0) & d["game_id"].isin(cc)].reset_index(drop=True)
    rep = {}

    # ---- c. feature noise, both test seasons: as-of rel vs realised -----------
    noise = {}
    for s in (2024, 2025):
        x = d[d["season"] == s]
        real_o = x.groupby("offense_team_id")["duration_s"].mean()
        real_d = x.groupby("defense_team_id")["duration_s"].mean()
        x = x.assign(month=pd.to_datetime(x["game_date"]).dt.month)
        g = x.groupby(["offense_team_id", "month"]).agg(o=("off_tempo_rel", "mean"))
        g = g.reset_index()
        g["real_o"] = g["offense_team_id"].map(real_o)
        g["real_d"] = g["offense_team_id"].map(real_d)
        noise[str(s)] = {str(m): {"corr_asof_offrel_vs_real_off_dur": float(np.corrcoef(gg["o"], gg["real_o"])[0, 1]),
                                  "corr_asof_offrel_vs_real_def_dur": float(np.corrcoef(gg["o"], gg["real_d"])[0, 1]),
                                  "n_teams": int(len(gg))}
                         for m, gg in g.groupby("month") if len(gg) > 50}
        # offence / defence: who controls duration
        both = pd.DataFrame({"off": real_o, "def": real_d}).dropna()
        noise[str(s)]["sd_team_off_dur"] = float(both["off"].std())
        noise[str(s)]["sd_team_def_dur"] = float(both["def"].std())
        noise[str(s)]["corr_team_off_def_dur"] = float(both.corr().iloc[0, 1])
        # split-half reliability of team off / def duration (odd vs even games)
        gnum = x.groupby("offense_team_id")["game_id"].rank(method="dense")
        x = x.assign(half=(gnum % 2).astype(int))
        ho = x.groupby(["offense_team_id", "half"])["duration_s"].mean().unstack()
        gnd = x.groupby("defense_team_id")["game_id"].rank(method="dense")
        x = x.assign(halfd=(gnd % 2).astype(int))
        hd = x.groupby(["defense_team_id", "halfd"])["duration_s"].mean().unstack()
        noise[str(s)]["splithalf_r_off"] = float(ho.corr().iloc[0, 1])
        noise[str(s)]["splithalf_r_def"] = float(hd.corr().iloc[0, 1])
    rep["feature_noise_and_asymmetry"] = noise

    # ---- 1-2. L2 model on 2025 ------------------------------------------------
    t = d[d["season"] == 2025].reset_index(drop=True)
    gd = pd.to_datetime(t["game_date"]).to_numpy()
    sched = load_schedule(L2)
    cuts = np.array([np.datetime64(r["refit_date"], "ns") for r in sched])
    seg = np.searchsorted(cuts, gd, side="right") - 1
    e = np.empty(len(t))
    for k in np.unique(seg):
        r = np.flatnonzero(seg == k)
        e[r] = e_consumed(sched[int(k)]["arm"], t.iloc[r])
    t["e"] = e
    t["month"] = pd.to_datetime(t["game_date"]).dt.month
    t["site"] = np.where(t["site_home"] + t["site_away"] == 0, "neutral",
                         np.where(t["offense_is_home"].astype(bool), "off_home", "off_away"))
    t["oq"] = t["offense_team_id"].map(pd.qcut(t.groupby("offense_team_id")["off_tempo_rel"].mean(), 5, labels=False))
    t["dq"] = t["defense_team_id"].map(pd.qcut(t.groupby("defense_team_id")["def_tempo_rel"].mean(), 5, labels=False))
    # game-prior quintile (what the clock actually sees)
    t["gq"] = pd.qcut(t["tempo_prior_game"].rank(method="first"), 5, labels=False)
    rep["n_rows_2025"] = int(len(t))
    rep["overall_gap_s"] = float(t["e"].mean() - t["duration_s"].mean())
    rep["resp_offence_q"] = resp(t, "oq")
    rep["resp_defence_q"] = resp(t, "dq")
    rep["resp_game_prior_q"] = resp(t, "gq")
    rep["resp_offence_q_by_start"] = resp(t, "oq", "start_reason")
    rep["resp_defence_q_by_start"] = resp(t, "dq", "start_reason")
    rep["resp_offence_q_by_month"] = resp(t, "oq", "month")
    rep["resp_offence_q_by_site"] = resp(t, "oq", "site")
    rep["resp_defence_q_by_site"] = resp(t, "dq", "site")

    # ---- a. the tercile encoding: how many distinct tempo levels per cell ----
    edges = None
    a0 = sched[-1]["arm"]
    inner = getattr(a0, "inner", a0)
    edges = getattr(inner, "tempo_edges", None)
    rep["tempo_edges_last_refit"] = list(edges) if edges is not None else None
    lv = inner.level_of(__import__("cbb_sim.models.clock_v3", fromlist=["add_p_state"]).add_p_state(t.iloc[:50000].copy()))
    rep["share_rows_served_at_full_depth_incl_tempo"] = float((lv == len(inner.dims)).mean())
    rep["dims"] = list(inner.dims)
    # within-tercile: model is flat by construction; the actual still slopes?
    t["tq3"] = np.searchsorted(np.asarray(edges), t["tempo_prior_game"].to_numpy(), side="right")
    within = {}
    for k, g in t.groupby("tq3"):
        g = g.assign(sub=pd.qcut(g["tempo_prior_game"].rank(method="first"), 3, labels=False))
        tt = g.groupby("sub").agg(model=("e", "mean"), actual=("duration_s", "mean"))
        within[str(k)] = {"model": [round(v, 3) for v in tt["model"]], "actual": [round(v, 3) for v in tt["actual"]]}
    rep["within_tercile_subthirds"] = within

    # ---- d. residual regression on log rels by start type --------------------
    rr = {}
    for st, g in [("ALL", t)] + list(t.groupby("start_reason")):
        if len(g) < 5000:
            continue
        X = np.column_stack([np.ones(len(g)), np.log(g["off_tempo_rel"].clip(0.7, 1.3)),
                             np.log(g["def_tempo_rel"].clip(0.7, 1.3))])
        y = (g["duration_s"] - g["e"]).to_numpy()
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        ya = g["duration_s"].to_numpy()
        ba, *_ = np.linalg.lstsq(X, ya, rcond=None)
        ym = g["e"].to_numpy()
        bm, *_ = np.linalg.lstsq(X, ym, rcond=None)
        rr[st] = {"n": int(len(g)), "resid_b_logoff": float(b[1]), "resid_b_logdef": float(b[2]),
                  "actual_b_logoff": float(ba[1]), "actual_b_logdef": float(ba[2]),
                  "model_b_logoff": float(bm[1]), "model_b_logdef": float(bm[2])}
    rep["resid_regression_2025"] = rr

    # ---- e. ceiling: leave-one-game-out team means (oracle) -----------------
    def logo(col):
        s = t.groupby(col)["duration_s"].agg(["sum", "count"])
        gsum = t.groupby([col, "game_id"])["duration_s"].agg(["sum", "count"])
        k = pd.MultiIndex.from_arrays([t[col], t["game_id"]])
        tot = s.loc[t[col]]
        gg = gsum.loc[k]
        return (tot["sum"].to_numpy() - gg["sum"].to_numpy()) / np.maximum(tot["count"].to_numpy() - gg["count"].to_numpy(), 1)
    t["oracle_o"] = logo("offense_team_id")
    t["oracle_d"] = logo("defense_team_id")
    lg = t["duration_s"].mean()
    t["oracle"] = t["oracle_o"] + t["oracle_d"] - lg
    tmp = t.assign(e=t["oracle"])
    rep["oracle_logo_resp_offence_q"] = resp(tmp, "oq")
    rep["oracle_logo_resp_defence_q"] = resp(tmp, "dq")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    for k in ("overall_gap_s", "resp_offence_q", "resp_defence_q", "resp_game_prior_q",
              "tempo_edges_last_refit", "share_rows_served_at_full_depth_incl_tempo",
              "within_tercile_subthirds", "oracle_logo_resp_offence_q", "oracle_logo_resp_defence_q"):
        print(k, json.dumps(rep[k], default=float))
    for k in ("resp_offence_q_by_start", "resp_defence_q_by_start", "resp_offence_q_by_month",
              "resp_offence_q_by_site", "resp_defence_q_by_site"):
        print(k)
        for kk, v in rep[k].items():
            print("  ", kk, v.get("n"), None if v.get("slope") is None else round(v["slope"], 3),
                  None if v.get("span_ratio") is None else round(v["span_ratio"], 3))
    print("resid_regression", json.dumps(rr, indent=0))
    print("noise", json.dumps(noise, indent=0))


if __name__ == "__main__":
    main()
