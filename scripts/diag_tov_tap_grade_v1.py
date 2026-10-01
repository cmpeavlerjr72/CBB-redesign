"""diag_tov_tap_grade_v1.py -- possession_outcome's turnover rate by chance in the SIM (diag_tov_tap_v1.py) vs REAL
(diag_tov_real_v1.py), for served v2 and RBTO (lane I, 2026-10-01).

Per tap dir: expected TOV share per chance (sum of the event adapter's P(TOV) / chances) by chance 1 / 2 / 3+ and chance-1
transition; chances per possession (chances / chance-1 rows); box TOV% (TOV / (FGA - OREB + TOV + 0.44 FTA)) from the
games rows; offence team quintile of the served `off_tov_c`. Then a closed TOV%-per-possession decomposition, sim minus
real: TOV per possession = sum_c (chances_c per possession) x (TOV rate_c), split into a mix term (chances) and a rate term.

Usage: diag_tov_tap_grade_v1.py <real_json> <out_json> <tap_dir> [<tap_dir> ...]
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REAL, OUT, TAPS = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), sys.argv[3:]
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402

inp = EngineInputs.load(str(ROOT / "data/processed/models/engine_v3"), "F2_2025")
offtov = inp.team_static[:, :, inp.team_names["off_tov_c"]]
r25 = REAL["2025"]
real_rate = {1: r25["chance"]["1"]["tov_rate"], 2: r25["chance"]["2"]["tov_rate"], 3: r25["chance"]["3"]["tov_rate"]}
real_n = {c: r25["chance"][str(c)]["n"] for c in (1, 2, 3)}
real_cpp = {c: real_n[c] / real_n[1] for c in (1, 2, 3)}
res = {"real_2025": {"rate": real_rate, "chances_per_possession": real_cpp,
                     "tov_per_possession": sum(real_cpp[c] * real_rate[c] for c in (1, 2, 3))}, "taps": {}}
for td in TAPS:
    t = pd.concat([pd.read_parquet(f) for f in glob.glob(str(Path(td) / "tov_*.parquet"))])
    g = pd.concat([pd.read_parquet(f) for f in glob.glob(str(Path(td) / "games_*.parquet"))])
    rate = {c: float(t.loc[t["chance"] == c, "p"].sum() / t.loc[t["chance"] == c, "n"].sum()) for c in (1, 2, 3)}
    n = {c: float(t.loc[t["chance"] == c, "n"].sum()) for c in (1, 2, 3)}
    cpp = {c: n[c] / n[1] for c in (1, 2, 3)}
    tr = t[t["chance"] == 1].groupby("trans")[["p", "n"]].sum()
    fga = sum(g[f"{s}_{k}"].sum() for s in ("home", "away") for k in ("fga3", "fga2_rim", "fga2_jump"))
    tov, oreb = g["home_tov"].sum() + g["away_tov"].sum(), g["home_oreb"].sum() + g["away_oreb"].sum()
    fta = g["home_fta"].sum() + g["away_fta"].sum()
    q = pd.qcut(pd.Series(offtov[t["gidx"].to_numpy(), t["off"].to_numpy()]).rank(method="first"), 5, labels=False).to_numpy()
    byq = {}
    for k in range(5):
        for lab, m in (("first", t["chance"].to_numpy() == 1), ("cont", t["chance"].to_numpy() >= 2)):
            mm = m & (q == k)
            byq.setdefault(lab, []).append(round(float(t["p"].to_numpy()[mm].sum() / t["n"].to_numpy()[mm].sum()), 4))
    tpp = sum(cpp[c] * rate[c] for c in (1, 2, 3))
    # LMDI-free two-term split vs real (midpoint weights)
    mix = sum((cpp[c] - real_cpp[c]) * (rate[c] + real_rate[c]) / 2 for c in (1, 2, 3))
    rt = sum((rate[c] - real_rate[c]) * (cpp[c] + real_cpp[c]) / 2 for c in (1, 2, 3))
    res["taps"][td] = {"rate": rate, "chances_per_possession": cpp, "chance1_transition_rate": {int(k): float(v["p"] / v["n"]) for k, v in tr.iterrows()},
                       "chance1_transition_share": float(tr.loc[1, "n"] / tr["n"].sum()) if 1 in tr.index else None,
                       "tov_per_possession_expected": tpp, "split_vs_real": {"chances_mix": mix, "rate": rt,
                                                                            "rate_by_chance": {c: (rate[c] - real_rate[c]) * (cpp[c] + real_cpp[c]) / 2 for c in (1, 2, 3)}},
                       "box_tov_pct": float(tov / (fga - oreb + tov + 0.44 * fta)), "box_oreb_share": float(oreb / (oreb + g["home_dreb"].sum() + g["away_dreb"].sum())),
                       "team_quintile_off_tov_c": byq}
res["real_2025"]["chance1_transition_share"] = r25["chance1_transition"]["1"]["n"] / r25["chance"]["1"]["n"]
res["real_2025"]["chance1_transition_rate"] = {k: v["tov_rate"] for k, v in r25["chance1_transition"].items()}
res["real_2025"]["team_quintile_first"] = [round(r25["team_tov_prior_quintile_first"][str(k)]["tov_rate"], 4) for k in range(5)]
res["real_2025"]["team_quintile_cont"] = [round(r25["team_tov_prior_quintile_cont"][str(k)]["tov_rate"], 4) for k in range(5)]
OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
