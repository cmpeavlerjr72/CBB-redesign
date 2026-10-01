"""diag_pace_eff_mech_v1.py -- the pace x efficiency channel's mechanism (lane B, 2026-10-01). DIAGNOSTIC ONLY.

(1) How game possessions respond to the game's outcome counts, real vs engine:
    regression of N (box estimator, game) on the game totals of OREB, FGM, TOV, FTA,
    sim: within-game deviations pooled (200 seeds); actual: residuals vs the sim's per-game mean.
(2) Real possession duration by start type x number of chances x terminal outcome
    (pbp possessions table, 2023-2025) -- the conditioning the served clock does not have
    (it draws one duration per possession BEFORE the chance cascade, marginal over chains).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "v3full_COMB9GCTKD_s200_o0"


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b[1:]


def main():
    rep = {}
    g = pd.read_parquet(ROOT / "results/engine_v0" / RUN / "games.parquet")
    summary, raw = G.build_grading_frame(g, 2025)
    gids = summary["game_id"].to_numpy()
    raw = raw.sort_values(["game_id", "seed"]).reset_index(drop=True)
    tot = {}
    for nm, cols in {"OREB": ["oreb"], "FGM": ["fgm2_rim", "fgm2_jump", "fgm3"], "TOV": ["tov"], "FTA": ["fta"],
                     "FGA": ["fga2_rim", "fga2_jump", "fga3"]}.items():
        tot[nm] = sum(raw[f"{s}_{c}"].to_numpy(float) for s in ("home", "away") for c in cols)
    N = 0.5 * (tot["FGA"] - tot["OREB"] + tot["TOV"] + 0.44 * tot["FTA"])
    gid = raw["game_id"].to_numpy()
    keys = ["OREB", "FGM", "TOV", "FTA"]
    D = pd.DataFrame({k: tot[k] for k in keys})
    D["N"] = N
    M = D.groupby(gid).mean()
    Wd = D - M.loc[gid].to_numpy()
    b_sim = ols(Wd[keys].to_numpy(), Wd["N"].to_numpy())
    # actual
    tb = R.load_actual_team_box(2025)
    tb = tb[tb["game_id"].isin(gids)]
    a = tb.groupby("game_id").agg(fga=("fga", "sum"), fgm=("fgm", "sum"), oreb=("oreb", "sum"),
                                  tov=("tov", "sum"), fta=("fta", "sum"), n=("team_id", "size"))
    a = a[a["n"] == 2]
    A = pd.DataFrame({"OREB": a["oreb"], "FGM": a["fgm"], "TOV": a["tov"], "FTA": a["fta"]}, index=a.index)
    A["N"] = 0.5 * (a["fga"] - a["oreb"] + a["tov"] + 0.44 * a["fta"])
    common = A.index.intersection(M.index)
    Ra = A.loc[common] - M.loc[common]
    b_act = ols(Ra[keys].to_numpy(), Ra["N"].to_numpy())
    rng = np.random.default_rng(5)
    bb = []
    for _ in range(200):
        ix = rng.integers(0, len(Ra), len(Ra))
        bb.append(ols(Ra[keys].to_numpy()[ix], Ra["N"].to_numpy()[ix]))
    rep["N_on_counts"] = {"keys": keys, "sim_within": b_sim.tolist(), "actual_resid": b_act.tolist(),
                          "actual_SE": np.std(bb, axis=0).tolist(), "n_games": int(len(common)),
                          "sim_corr_N_OREB": float(np.corrcoef(Wd["N"], Wd["OREB"])[0, 1]),
                          "act_corr_N_OREB": float(np.corrcoef(Ra["N"], Ra["OREB"])[0, 1]),
                          "sim_corr_N_FGM": float(np.corrcoef(Wd["N"], Wd["FGM"])[0, 1]),
                          "act_corr_N_FGM": float(np.corrcoef(Ra["N"], Ra["FGM"])[0, 1])}
    # (2) real durations by start x chances x terminal
    rows = []
    for s in (2023, 2024, 2025):
        c = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{s}.parquet",
                            columns=["game_id", "period", "poss_index", "chance_number", "terminal_event",
                                     "duration_s", "start_reason", "fgm_rim", "fgm_jump2", "fgm_3", "points"])
        c["made_fg"] = (c["fgm_rim"] + c["fgm_jump2"] + c["fgm_3"]) > 0
        p = c.groupby(["game_id", "period", "poss_index"]).agg(
            start=("start_reason", "first"), k=("chance_number", "max"), dur=("duration_s", "sum"),
            last=("terminal_event", "last"), made=("made_fg", "last"), pts=("points", "sum")).reset_index()
        p["season"] = s
        rows.append(p)
    P = pd.concat(rows, ignore_index=True)
    P = P[P["start"].isin(["made_FG", "DREB", "TOV", "made_FT"])]
    P["kk"] = np.minimum(P["k"], 3)
    P["end"] = np.where(P["last"] == "TOV", "TOV",
                        np.where(P["made"], "made_FG", np.where(P["last"].str.startswith("FT"), "FT", "missFG_or_other")))
    t1 = P.groupby(["start", "kk"])["dur"].agg(["mean", "size"]).reset_index()
    t2 = P[P["kk"] == 1].groupby(["start", "end"])["dur"].agg(["mean", "size"]).reset_index()
    rep["dur_by_start_chances"] = t1.to_dict("records")
    rep["dur_by_start_end_k1"] = t2.to_dict("records")
    m1 = P[P["kk"] == 1].groupby("start")["dur"].mean()
    m2 = P[P["kk"] == 2].groupby("start")["dur"].mean()
    rep["extra_seconds_per_extra_chance"] = (m2 - m1).to_dict()
    out = ROOT / "results/g5_channels/pace_eff_mech_v1.json"
    out.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep["N_on_counts"], indent=1))
    print(t1.to_string())
    print(t2.to_string())
    print(rep["extra_seconds_per_extra_chance"])


if __name__ == "__main__":
    main()
