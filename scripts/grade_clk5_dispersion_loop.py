"""grade_clk5_dispersion_loop.py -- the DISPERSION lines of a round-5 closed loop.

`scripts/grade_clk3c_closed_loop.py` already scores the round-3c/4 table (G1
mean and SD, margin SD, corr(home, away), PPP, responsiveness) and is used
unchanged for round 5. It does NOT compute the two lines round 5 exists to
move, because rounds 3c and 4 were about the mean:

  * the WITHIN-game (across-seed) SD of possessions per team-game against
    `SD(actual - sim per-game mean)` -- the quantity
    `docs/tests/engine_v1_variance_ot_diag_2026-09-11.md` section 2 measured at
    3.743 produced against 4.972 needed; and
  * the same ratio for the total and the margin, which is `eval.gates.gate_g5`'s
    own definition (mean within-game sim SD over SD of the residual) rather
    than a pooled SD comparison -- the two are different numbers and section
    3.1 of that diagnostic says so explicitly; and

  * the within-game (across-seed) `corr(possessions, eFG%)`, which the ENGINE
    lane's proposed section 18 asks this round to carry beside the dispersion
    lines (`docs/tests/pace_efficiency_sign_2026-09-11.md`). Section 19 of
    `experiments.md` records what this run can and cannot say about it.

One path, both arms, identity read from `run_meta.json` and used only to label.
Nothing here adjusts anything.

    .venv/Scripts/python.exe scripts/grade_clk5_dispersion_loop.py \
        --pattern "clk5_[AR]*"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.eval import reference as REF  # noqa: E402

RESULTS = Path("results/engine_v0")
UNIVERSE_V2 = Path("data/processed/games_universe_v2.parquet")


def score(d: Path, act: pd.DataFrame, cc: set[int]) -> dict:
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    raw = pd.read_parquet(d / "games.parquet")
    raw["poss"] = raw["possessions"].astype(float)
    raw["total"] = raw["total"].astype(float)
    raw["margin"] = raw["margin"].astype(float)
    g = raw.groupby("game_id")
    per = pd.DataFrame({
        "n_seeds": g.size(),
        "poss_mean": g["poss"].mean(), "poss_sd": g["poss"].std(ddof=1),
        "total_mean": g["total"].mean(), "total_sd": g["total"].std(ddof=1),
        "margin_mean": g["margin"].mean(), "margin_sd": g["margin"].std(ddof=1),
    }).reset_index().merge(act, on="game_id", how="inner")

    out = {"tag": d.name, "arm": meta.get("arm"),
           "n_seeds": int(meta.get("n_seeds", 0)),
           "engine_clock": meta.get("adapter_flags", {}).get("ENGINE_CLOCK"),
           "loop_commit": meta.get("loop_commit"),
           "n_games": int(len(per))}
    for lab, sub in (("all", per), ("cc", per[per["game_id"].isin(cc)])):
        out[f"n_{lab}"] = int(len(sub))
        for q, a in (("poss", "act_poss"), ("total", "act_total"),
                     ("margin", "act_margin")):
            within = float(sub[f"{q}_sd"].mean())
            resid = float(np.std(sub[a].to_numpy() - sub[f"{q}_mean"].to_numpy(),
                                 ddof=1))
            out[f"{lab}_{q}_within_sd"] = within
            out[f"{lab}_{q}_resid_sd"] = resid
            out[f"{lab}_{q}_sd_ratio"] = within / resid if resid else float("nan")
            out[f"{lab}_{q}_mean"] = float(sub[f"{q}_mean"].mean())
            out[f"{lab}_{q}_actual_mean"] = float(sub[a].mean())
            out[f"{lab}_{q}_pooled_sd"] = float(
                raw[raw["game_id"].isin(sub["game_id"])][q if q != "poss"
                                                         else "poss"].std(ddof=1))
            out[f"{lab}_{q}_actual_pooled_sd"] = float(sub[a].std(ddof=1))
    # Within-game corr(P, eFG%), both teams pooled, exactly the object
    # `docs/tests/engine_v1_variance_ot_diag_2026-09-11.md` section 4 measured:
    # demean both quantities BY GAME across seeds, then correlate the residuals,
    # so only within-game (across-seed) covariation enters.
    fgm = (raw["home_fgm2_rim"] + raw["home_fgm2_jump"] + raw["home_fgm3"]
           + raw["away_fgm2_rim"] + raw["away_fgm2_jump"] + raw["away_fgm3"]).astype(float)
    fgm3 = (raw["home_fgm3"] + raw["away_fgm3"]).astype(float)
    fga = (raw["home_fga3"] + raw["home_fga2_rim"] + raw["home_fga2_jump"]
           + raw["away_fga3"] + raw["away_fga2_rim"] + raw["away_fga2_jump"]).astype(float)
    tmp = pd.DataFrame({"game_id": raw["game_id"].to_numpy(),
                        "P": raw["poss"].to_numpy(),
                        "efg": ((fgm + 0.5 * fgm3) / fga).to_numpy()})
    gg = tmp.groupby("game_id")
    pr = tmp["P"].to_numpy() - gg["P"].transform("mean").to_numpy()
    er = tmp["efg"].to_numpy() - gg["efg"].transform("mean").to_numpy()
    ok = np.isfinite(pr) & np.isfinite(er) & (np.abs(pr) + np.abs(er) > 0)
    out["corr_P_eFG_within"] = (float(np.corrcoef(pr[ok], er[ok])[0, 1])
                                if ok.sum() > 10 else float("nan"))
    out["corr_P_eFG_within_n"] = int(ok.sum())
    out["var_P_within"] = float(np.var(pr[ok], ddof=1))
    out["efg_mean"] = float(tmp["efg"].mean())
    out["corr_home_away"] = float(np.corrcoef(
        raw["home_pts"].astype(float), raw["away_pts"].astype(float))[0, 1])
    ph = act.set_index("game_id").loc[per["game_id"], ["home_score", "away_score"]]
    out["corr_home_away_actual"] = float(np.corrcoef(
        ph["home_score"].astype(float), ph["away_score"].astype(float))[0, 1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--pattern", default="clk5_[AR]*")
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--out", default="data/processed/models/clock/v5_closed_loop_dispersion.json")
    a = ap.parse_args()

    dirs = sorted(p for p in Path(a.results_dir).glob(a.pattern)
                  if (p / "games.parquet").exists())
    if not dirs:
        raise SystemExit(f"no results under {a.results_dir}/{a.pattern}")
    ids = pd.read_parquet(dirs[0] / "games.parquet")["game_id"].unique()
    act = REF.load_actual_games(a.season)[
        ["game_id", "margin", "total", "home_score", "away_score"]].rename(
        columns={"margin": "act_margin", "total": "act_total"})
    poss = REF.load_actual_possessions(a.season).rename(
        columns={"game_poss": "act_poss"})
    act = act.merge(poss, on="game_id", how="left")
    act = act[act["game_id"].isin(ids)].reset_index(drop=True)
    u = pd.read_parquet(UNIVERSE_V2)[["game_id", "clock_complete_reg"]]
    cc = set(u.loc[u["clock_complete_reg"].fillna(False), "game_id"].astype(int))

    rows = [score(d, act, cc) for d in dirs]
    seen = {r["n_games"] for r in rows}
    assert len(seen) == 1, f"arms were scored on different game sets: {seen}"
    Path(a.out).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    cols = ["tag", "arm", "n_seeds",
            "cc_poss_mean", "cc_poss_actual_mean",
            "cc_poss_within_sd", "cc_poss_resid_sd", "cc_poss_sd_ratio",
            "all_poss_mean", "all_poss_actual_mean",
            "all_poss_within_sd", "all_poss_resid_sd", "all_poss_sd_ratio",
            "all_total_sd_ratio", "all_margin_sd_ratio",
            "corr_P_eFG_within", "var_P_within", "efg_mean",
            "corr_home_away", "corr_home_away_actual"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(pd.DataFrame(rows)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
