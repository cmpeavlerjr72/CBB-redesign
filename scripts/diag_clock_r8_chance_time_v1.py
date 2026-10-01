"""diag_clock_r8_chance_time_v1.py -- clock round 8 amendment (lane H, 2026-10-01): verify lane B's
clock mechanism on real possessions and measure what an outcome-conditioned possession time needs.
READ-ONLY.

  1. sum identity: possession duration == sum of its chances' durations (v4 tables)
  2. durations by start type x number of chances; one-chance possessions by how they ended
     (made FG / missed FG -> DREB / TOV / FT trip)
  3. continuation (chance >= 2) durations: mean, quantiles by chance index and start group
  4. game-level: regression of team-game possessions on the game's OREB / FGM / TOV / FTA (actual,
     residualised on team-season fixed effects is lane B's; here plain within-season OLS with the
     game's tempo sum and month FE), seasons 2023-2025
  5. the KD feed: is chance-1 elapsed | whole-possession duration (KD's table) the same law as
     chance-1 elapsed | chance-1 duration? Compared on the fg_make design rows' real values.
Writes results/clock_r8/diag_chance_time.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "data/processed/possessions_v4"
OUT = ROOT / "results/clock_r8"


def main():
    rep = {}
    seasons = [2023, 2024, 2025]
    allp = []
    for s in seasons:
        p = pd.read_parquet(P4 / f"possessions_{s}.parquet",
                            columns=["game_id", "period", "poss_index", "offense_team_id", "start_reason", "duration_s",
                                     "n_chances", "oreb_count", "terminal_event", "points", "start_clock",
                                     "fgm_rim", "fgm_jump2", "fgm_3", "fga_rim", "fga_jump2", "fga_3", "fta"])
        c = pd.read_parquet(P4 / f"chances_{s}.parquet",
                            columns=["game_id", "period", "poss_index", "chance_number", "duration_s", "terminal_event",
                                     "points", "start_clock"])
        p["season"] = s
        cs = c.groupby(["game_id", "period", "poss_index"]).agg(sum_ch=("duration_s", "sum"), n_ch=("chance_number", "max"))
        c1 = c[c["chance_number"] == 1].set_index(["game_id", "period", "poss_index"])[["duration_s", "terminal_event", "points"]]
        c1.columns = ["d1", "term1", "pts1"]
        p = p.join(cs, on=["game_id", "period", "poss_index"]).join(c1, on=["game_id", "period", "poss_index"])
        if s == 2025:
            rep["sum_identity"] = {"share_equal": float((p["sum_ch"] == p["duration_s"]).mean()),
                                   "share_nch_equal": float((p["n_ch"] == p["n_chances"]).mean()),
                                   "mean_abs_diff": float((p["sum_ch"] - p["duration_s"]).abs().mean()),
                                   "n": int(len(p))}
            cc = c[c["chance_number"] >= 2].copy()
            cc = cc.merge(p[["game_id", "period", "poss_index", "start_reason"]], on=["game_id", "period", "poss_index"])
            cc["grp"] = np.where(cc["chance_number"] >= 3, "3+", "2")
            cc = cc[cc["start_clock"] >= 60]
            q = cc.groupby("grp")["duration_s"].describe(percentiles=[.1, .25, .5, .75, .9])
            rep["continuation_2025"] = q.round(2).to_dict("index")
            rep["continuation_by_start_2025"] = cc.groupby(["grp", "start_reason"])["duration_s"].mean().round(2).unstack().to_dict()
        allp.append(p)
    p = pd.concat(allp, ignore_index=True)
    r = p[(p["period"] <= 2) & (p["start_clock"] >= 60)]
    rep["dur_by_start_x_chances"] = r[r["n_chances"] <= 4].groupby(["start_reason", "n_chances"])["duration_s"].mean().round(2).unstack().to_dict("index")
    one = r[r["n_chances"] == 1].copy()
    fga = one["terminal_event"].str.startswith("FGA")
    made = (one["fgm_rim"] + one["fgm_jump2"] + one["fgm_3"]) > 0
    one["end"] = np.select([fga & made, fga & ~made, one["terminal_event"] == "TOV",
                            one["terminal_event"].str.startswith("FT")], ["made_FG", "missed_FG", "TOV", "FT_trip"], "other")
    rep["one_chance_by_end"] = one.groupby(["start_reason", "end"])["duration_s"].mean().round(2).unstack().to_dict("index")
    # first-chance duration of multi-chance possessions vs one-chance
    rep["d1_by_nchances"] = r[r["n_chances"] <= 4].groupby("n_chances")["d1"].mean().round(2).to_dict()
    rep["share_rows_by_nchances"] = (r["n_chances"].clip(upper=4).value_counts(normalize=True).sort_index().round(4).to_dict())
    # game-level regressions (actual), per season: team-game possessions on game totals
    gm = p[p["period"] <= 2].groupby(["season", "game_id"]).agg(
        n=("poss_index", "size"), oreb=("oreb_count", "sum"),
        fgm=("fgm_rim", "sum"), fgm2=("fgm_jump2", "sum"), fgm3=("fgm_3", "sum"),
        fta=("fta", "sum"), tov=("terminal_event", lambda x: int((x == "TOV").sum())))
    gm["fgm"] = gm["fgm"] + gm["fgm2"] + gm["fgm3"]
    gm["N"] = gm["n"] / 2.0
    out = {}
    for s, g in gm.groupby(level=0):
        X = g[["oreb", "fgm", "tov", "fta"]].to_numpy(float)
        X = np.column_stack([np.ones(len(g)), X])
        b, *_ = np.linalg.lstsq(X, g["N"].to_numpy(float), rcond=None)
        out[str(s)] = dict(zip(["const", "oreb", "fgm", "tov", "fta"], np.round(b, 4).tolist()))
    rep["actual_N_on_totals_raw_ols"] = out
    # 5. KD consistency: chance-1 elapsed conditioned on whole duration vs on chance-1 duration
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.argv = [sys.argv[0], "unused"]
    from exp_chance_time_offline_v1 import real_rows  # noqa: E402
    from build_chance_time_lut_v1 import dur_bin  # noqa: E402
    kd = {}
    for s in (2024, 2025):
        m = real_rows(s)
        m = m[(m["chance_number"] == 1) & m["duration_s"].notna()].copy()
        c = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{s}.parquet",
                            columns=["game_id", "period", "poss_index", "chance_number", "duration_s"])
        c1 = c[c["chance_number"] == 1].rename(columns={"duration_s": "d1"})[["game_id", "period", "poss_index", "d1"]]
        m = m.merge(c1, on=["game_id", "period", "poss_index"], how="left")
        m = m[m["d1"].notna()]
        m["bw"] = dur_bin(m["duration_s"].to_numpy().astype(np.int64))
        m["b1"] = dur_bin(m["d1"].to_numpy().astype(np.int64))
        same = (m["bw"] == m["b1"]).mean()
        e = m["chance_elapsed_s"].to_numpy(float)
        # mean elapsed by bin under each conditioning, and how far apart they are for the rows whose bins differ
        mw = m.groupby("bw")["chance_elapsed_s"].mean()
        m1 = m.groupby("b1")["chance_elapsed_s"].mean()
        diffrows = m[m["bw"] != m["b1"]]
        kd[str(s)] = {"n_rows": int(len(m)), "share_same_bin": float(same),
                      "share_elapsed_le_d1": float((e <= m["d1"].to_numpy() + 0.5).mean()),
                      "mean_elapsed": float(e.mean()),
                      "rows_diff_bin_mean_elapsed": float(diffrows["chance_elapsed_s"].mean()),
                      "rows_diff_bin_kd_whole_table_mean": float(diffrows["bw"].map(mw).mean()),
                      "rows_diff_bin_kd_d1_table_mean": float(diffrows["b1"].map(m1).mean()),
                      "rows_diff_bin_mean_d1": float(diffrows["d1"].mean()),
                      "rows_diff_bin_mean_whole": float(diffrows["duration_s"].mean())}
    rep["KD_consistency"] = kd
    (OUT / "diag_chance_time.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
