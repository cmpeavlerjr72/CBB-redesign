#!/usr/bin/env python
"""
grade_foul_accrual_v1.py -- ONE blind grading pass over every round-6 arm.

Pre-registration: `docs/models/possession_outcome/experiments.md` section 13 as
amended by section 15. The script does not know which column is the reference;
it scores every probability column in the prediction file through the same code
path and the caller reads the table.

    .venv/Scripts/python.exe scripts/grade_foul_accrual_v1.py --target y_def
    .venv/Scripts/python.exe scripts/grade_foul_accrual_v1.py --target y_bonus

PRIMARY (13.4 / 15.3): log loss on fold-2 test rows inside the FIT WINDOW
(`period <= 2 and seconds_remaining > 120`). Fold 1 is reported for
confirmation. The noise floor is |seed A - seed B| on the same primary.

ELIGIBILITY LINES (15.6):
  * the rate-by-game-minute curve that drives bonus occupancy -- max absolute
    per-bucket gap between mean predicted and realised rate, in pp. Offline this
    is the DRIVER of the occupancy curve, not the occupancy curve itself, which
    only a closed loop can produce; the results section says so.
  * the team prior-season-quintile slope, powered cells only, power stated.
  * the SPREAD of team-level predicted rates against the realised SD (15.5).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_DIR = Path("data/processed/models/possession_outcome/round6")
EPS = 1e-12
MIN_CELL = 300


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def seg_table(df: pd.DataFrame, arm: str, y: str, key: str) -> list[dict]:
    rows = []
    for k, g in df.groupby(key, observed=True):
        n = len(g)
        rows.append({"cell": str(k), "n": int(n),
                     "actual": round(float(g[y].mean()) * 100, 4),
                     "pred": round(float(g[arm].mean()) * 100, 4),
                     "gap_pp": round(float(g[arm].mean() - g[y].mean()) * 100, 4),
                     "underpowered": bool(n < MIN_CELL)})
    return rows


def team_spread(df: pd.DataFrame, arm: str, y: str) -> dict:
    """SD of team-level predicted vs realised rate (15.5), defence side."""
    g = df.groupby("defense_team_id").agg(n=(y, "size"), act=(y, "mean"), pred=(arm, "mean"))
    g = g[g["n"] >= MIN_CELL]
    return {"n_teams_powered": int(len(g)),
            "sd_pred": round(float(g["pred"].std()) * 100, 5),
            "sd_actual": round(float(g["act"].std()) * 100, 5),
            "sd_ratio": round(float(g["pred"].std() / max(g["act"].std(), 1e-12)), 4),
            "corr": round(float(g["pred"].corr(g["act"])), 4)}


def quintile_slope(df: pd.DataFrame, arm: str, y: str, prior: pd.DataFrame) -> dict:
    g = df.groupby("defense_team_id").agg(n=(y, "size"), act=(y, "mean"), pred=(arm, "mean"))
    g = g.merge(prior, left_index=True, right_on="team_id", how="inner")
    g = g[g["n"] >= MIN_CELL]
    if len(g) < 25:
        return {"status": "UNDERPOWERED", "n_teams": int(len(g))}
    g["q"] = pd.qcut(g["prior_rate"], 5, labels=False, duplicates="drop")
    q = g.groupby("q").agg(teams=("act", "size"), act=("act", "mean"), pred=("pred", "mean"))
    span_a = float(q["act"].max() - q["act"].min())
    span_p = float(q["pred"].max() - q["pred"].min())
    mono = int(np.sum(np.diff(q["pred"].to_numpy()) > 0))
    return {"status": "scored", "n_teams": int(len(g)),
            "teams_per_q": q["teams"].tolist(),
            "slope_ratio": round(span_p / max(span_a, 1e-12), 4),
            "monotone_steps": f"{mono}/{len(q) - 1}",
            "q_actual_pp": [round(v * 100, 3) for v in q["act"].tolist()],
            "q_pred_pp": [round(v * 100, 3) for v in q["pred"].tolist()],
            "gap_q1_pp": round(float(q["pred"].iloc[0] - q["act"].iloc[0]) * 100, 3),
            "gap_q5_pp": round(float(q["pred"].iloc[-1] - q["act"].iloc[-1]) * 100, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="y_def")
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 7])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    y = a.target
    res: dict = {"target": y, "seeds": a.seeds, "folds": {}}

    # prior-season team defensive silent-foul rate, for the quintile cut
    design = pd.read_parquet(ROUND_DIR / "foul_accrual_poss_v1.parquet",
                             columns=["season", "defense_team_id", "def_silent",
                                      "offense_team_id", "off_silent", "terminal_event",
                                      "off_in_bonus"])
    for fold, prior_season in (("F1", 2023), ("F2", 2024)):
        f = ROUND_DIR / f"preds_{y}_{fold}_seed{a.seeds[0]}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        arms = [c for c in d.columns if c.startswith(("F", "T", "H", "D")) and c != "F0_ref"
                and d[c].dtype.kind == "f" and c not in
                ("game_minute",)]
        arms = [c for c in d.columns if c in
                ("F0", "F0_served", "F1", "F2", "F2m", "F3", "F3b", "F3c", "F3d", "F5",
                 "H1", "D9a", "D9b", "D9c", "T0", "T1", "T2", "T3")]
        fit = d[d["in_fit_window"] == 1].copy()
        held = d[d["in_fit_window"] == 0].copy()
        fit["minute_bucket"] = pd.cut(fit["game_minute"],
                                      [0, 5, 10, 15, 20, 25, 30, 35, 38, 45],
                                      right=False)
        fit["site"] = np.where(fit["site_home"] > 0, "home",
                               np.where(fit["site_away"] > 0, "away", "neutral"))
        fit["conf"] = np.where(fit["is_conf_game"] > 0, "conference", "non-conference")
        fit["season_seg"] = pd.cut(fit["days_since_start"], [-1, 45, 75, 400],
                                   labels=["Nov-Dec", "Jan", "Feb-Apr"])
        fit["fouls_bucket"] = pd.cut(fit["def_team_fouls"], [-1, 3, 6, 9, 99],
                                     labels=["0-3", "4-6", "7-9", "10+"])
        ps = design[design["season"] == prior_season]
        prior = ps.groupby("defense_team_id")["def_silent"].agg(["mean", "size"]).reset_index()
        prior = prior[prior["size"] >= 300].rename(
            columns={"defense_team_id": "team_id", "mean": "prior_rate"})[["team_id", "prior_rate"]]

        # noise floor: the same spec under the second seed
        f2 = ROUND_DIR / f"preds_{y}_{fold}_seed{a.seeds[1]}.parquet"
        floor = {}
        if f2.exists():
            d2 = pd.read_parquet(f2)
            fit2 = d2[d2["in_fit_window"] == 1]
            for arm in arms:
                if arm in fit2.columns:
                    floor[arm] = round(abs(log_loss(fit[y].to_numpy(), fit[arm].to_numpy())
                                           - log_loss(fit2[y].to_numpy(), fit2[arm].to_numpy())), 8)

        out = {}
        for arm in arms:
            yy = fit[y].to_numpy()
            pp = fit[arm].to_numpy()
            mb = fit.groupby("minute_bucket", observed=True).apply(
                lambda g, c=arm: (g[c].mean() - g[y].mean()) * 100, include_groups=False)
            out[arm] = {
                "primary_logloss": round(log_loss(yy, pp), 8),
                "floor": floor.get(arm),
                "mean_pred_pp": round(float(pp.mean()) * 100, 4),
                "mean_actual_pp": round(float(yy.mean()) * 100, 4),
                "calib_gap_pp": round(float(pp.mean() - yy.mean()) * 100, 4),
                "minute_curve_max_abs_gap_pp": round(float(np.abs(mb).max()), 4),
                "minute_curve_pp": {str(k): round(float(v), 3) for k, v in mb.items()},
                "held_out_final2_logloss": round(
                    log_loss(held[y].to_numpy(), held[arm].to_numpy()), 8) if len(held) else None,
                "held_out_calib_gap_pp": round(
                    float(held[arm].mean() - held[y].mean()) * 100, 4) if len(held) else None,
                "team_spread": team_spread(fit, arm, y),
                "prior_quintile": quintile_slope(fit, arm, y, prior),
                "by_site": seg_table(fit, arm, y, "site"),
                "by_conf": seg_table(fit, arm, y, "conf"),
                "by_season_seg": seg_table(fit, arm, y, "season_seg"),
                "by_def_fouls": seg_table(fit, arm, y, "fouls_bucket"),
                "by_period": seg_table(fit, arm, y, "period"),
            }
        res["folds"][fold] = {"n_fit": int(len(fit)), "n_held_out": int(len(held)),
                              "arms": out}

    path = Path(a.out) if a.out else ROUND_DIR / f"grade_{y}.json"
    path.write_text(json.dumps(res, indent=1))
    # console table
    for fold, fr in res["folds"].items():
        print(f"\n=== {fold} (n_fit {fr['n_fit']:,}) target {y} ===")
        print(f"{'arm':<11}{'logloss':>12}{'floor':>11}{'calib pp':>10}"
              f"{'minmax pp':>11}{'sd ratio':>10}{'slope':>8}")
        base = fr["arms"].get("F0", {}).get("primary_logloss") or \
            fr["arms"].get("T0", {}).get("primary_logloss")
        for arm, m in fr["arms"].items():
            sl = m["prior_quintile"].get("slope_ratio", float("nan"))
            print(f"{arm:<11}{m['primary_logloss']:>12.7f}{(m['floor'] or 0):>11.7f}"
                  f"{m['calib_gap_pp']:>10.3f}{m['minute_curve_max_abs_gap_pp']:>11.3f}"
                  f"{m['team_spread']['sd_ratio']:>10.3f}{sl:>8.3f}")
        if base:
            print(f"(reference log loss {base:.7f})")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
