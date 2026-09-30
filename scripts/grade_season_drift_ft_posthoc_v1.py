#!/usr/bin/env python
"""
grade_season_drift_ft_posthoc_v1.py -- POST-HOC re-score of the season-drift
round-1 ft_tech arms under the PM ruling (season_drift experiments.md section
3.2): E5's one-sided absolute spread line is replaced by the LEVEL-NORMALISED
spread (ratio of coefficients of variation: team-prediction CV over the
noise-corrected realised team-rate CV) plus the team slope. Every other
registered line (E1-E4, E6, E7) is read unchanged from `grade_v1.json`.

The team slope is reported under three readings because the registered line is
itself level-confounded in a log-link model: (a) raw span ratio, one-sided
(>= R - 0.05, the registered form); (b) level-normalised, one-sided; (c)
level-normalised, symmetric (|x - 1| <= |x_R - 1| + 0.05). Overall and Nov-Dec.

    .venv/Scripts/python.exe scripts/grade_season_drift_ft_posthoc_v1.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G = json.loads((ROOT / "results/season_drift/round1/grade_v1.json").read_text())
TOL = 0.05


def main() -> int:
    rows = []
    for arm in ["C0", "O", "P", "W", "F", "T"]:
        ch = dict(G["decision"]["ft_tech"]["rows"][arm]["checks"])
        ch.pop("E5_slope_spread")
        c = G["cards"][f"ft_tech|F2|{arm}|0"]["classes"][0]
        r = G["cards"]["ft_tech|F2|R|0"]["classes"][0]
        cv_ok = c["team"]["sd_ratio_nc_levelnorm"] >= r["team"]["sd_ratio_nc_levelnorm"] - TOL
        rd = {}
        for key in ("team", "team_novdec"):
            a, b = c[key], r[key]
            rd[key] = {
                "a_raw_one_sided": a["slope_ratio"] >= b["slope_ratio"] - TOL,
                "b_norm_one_sided": a["slope_levelnorm"] >= b["slope_levelnorm"] - TOL,
                "c_norm_symmetric": abs(a["slope_levelnorm"] - 1) <= abs(b["slope_levelnorm"] - 1) + TOL}
        others = all(ch.values())
        out = {"arm": arm, "other_lines_pass": others, "failed_other": [k for k, v in ch.items() if not v],
               "cv_ratio": c["team"]["sd_ratio_nc_levelnorm"], "cv_ratio_R": r["team"]["sd_ratio_nc_levelnorm"],
               "cv_ok": cv_ok}
        for rdg in ("a_raw_one_sided", "b_norm_one_sided", "c_norm_symmetric"):
            out[f"eligible_{rdg}"] = bool(others and cv_ok and rd["team"][rdg] and rd["team_novdec"][rdg])
        rows.append(out)
        print(json.dumps(out))
    (ROOT / "results/season_drift/round1/ft_posthoc_v1.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
