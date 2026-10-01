"""diag_clock_r8_chance_resid_v1.py -- POST-HOC line for the round-8 amendment (experiments.md
section 39). Lane H, 2026-10-01.

Section 38's offline line 1 compares the model's implied-count slopes on the game's event COUNTS
with the actual count's slopes. That comparison is confounded: a game that is faster for any
unmodelled reason has more possessions and therefore more OREB, FGM, TOV and FTA, so the actual
slope carries a positive reverse-causal term the model-implied count (built from the realised
outcomes only) cannot have. This script reads the confound-free version: per game, the RESIDUAL
(actual mean possession time minus the arm's mean implied time, seconds) regressed on the game's
per-possession RATES (OREB, FGM, TOV, FTA per possession). Target 0: a clock that prices the
realised outcomes leaves no residual dependence on them. Same games, arms, folds and bootstrap as
`grade_clock_r8_chance_offline_v1.py` (its scoring functions, imported unchanged).
Writes results/clock_r8/chance_resid_posthoc.json
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import grade_clock_r8_chance_offline_v1 as GK  # noqa: E402
from cbb_sim.models import clock as CK  # noqa: E402

RATES = ["oreb", "fgm", "tov", "fta"]


def resid_lines(g, arms, w):
    keep = w > 0
    ww = w[keep]
    n = g.loc[keep, "n"].to_numpy(float)
    X = np.column_stack([g.loc[keep, c].to_numpy(float) / n for c in RATES])
    out = {}
    for a in arms:
        r = (g.loc[keep, "dur"].to_numpy() - g.loc[keep, f"t_{a}"].to_numpy()) / n
        b = GK.wls(r, X, ww)
        out[a] = dict(zip([f"resid_on_{c}_rate" for c in RATES], map(float, b)))
    return out


def main():
    univ = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet")
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    design = GK.TK.T.load_design()
    first, _ = GK.TK.chance_tables([2024, 2025])
    design = design[design["season"].isin([2024, 2025])]
    design = GK.TK.build_d1_design(design, first)
    rep = {}
    for fold, season in (("F2", 2025), ("F1", 2024)):
        arms = [a for a in GK.ARMS if (GK.CKD / GK.ARMS[a] / fold / "manifest.json").exists()]
        d = design[(design["season"] == season) & (design["period"] <= 2.0) & design["game_id"].isin(cc)].reset_index(drop=True)
        p4 = pd.read_parquet(GK.TK.P4 / f"possessions_{season}.parquet",
                             columns=["game_id", "period", "poss_index", "duration_s", "oreb_count", "fgm_rim",
                                      "fgm_jump2", "fgm_3", "fta", "terminal_event"])
        p4 = p4.rename(columns={"period": "_p", "poss_index": "_i", "duration_s": "dur_poss", "terminal_event": "term_poss"})
        p4["_p"] = p4["_p"].astype("int64"); p4["_i"] = p4["_i"].astype("int64")
        d["_p"] = d["period"].astype("int64"); d["_i"] = d["poss_index"].astype("int64")
        d = d.merge(p4, on=["game_id", "_p", "_i"], how="left", validate="1:1")
        R = d["seconds_remaining"].to_numpy(float)
        for a in arms:
            if a in ("C0", "M2D"):
                e, _ = GK.G.score(GK.G.load_schedule(GK.CKD / GK.ARMS[a] / fold / "manifest.json"), d)
                d[f"t_{a}"] = e
            else:
                e_m, e_c, _, _, cont = GK.score_k(GK.G.load_schedule(GK.CKD / GK.ARMS[a] / fold / "manifest.json"), d)
                d[f"t_{a}"] = np.minimum((e_m if a == "K1" else e_c) + cont, R)
        d["fgm"] = d["fgm_rim"] + d["fgm_jump2"] + d["fgm_3"]
        d["tov"] = (d["term_poss"] == "TOV").astype(float)
        g = d.groupby("game_id").agg(n=("dur_poss", "size"), dur=("dur_poss", "sum"), oreb=("oreb_count", "sum"),
                                     fgm=("fgm", "sum"), tov=("tov", "sum"), fta=("fta", "sum"),
                                     **{f"t_{a}": (f"t_{a}", "sum") for a in arms})
        L = resid_lines(g, arms, np.ones(len(g)))
        rng = np.random.default_rng(20261001)
        boots = {a: [] for a in arms}
        for _ in range(GK.NBOOT):
            w = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g)).astype(float)
            Lb = resid_lines(g, arms, w)
            for a in arms:
                boots[a].append([Lb[a][f"resid_on_{c}_rate"] for c in RATES])
        for a in arms:
            B = np.array(boots[a]); B0 = np.array(boots["C0"])
            L[a]["se"] = dict(zip(RATES, B.std(axis=0).round(4).tolist()))
            if a != "C0":
                L[a]["d_abs_vs_C0"] = dict(zip(RATES, (np.abs([L[a][f"resid_on_{c}_rate"] for c in RATES])
                                                       - np.abs([L["C0"][f"resid_on_{c}_rate"] for c in RATES])).round(4).tolist()))
                L[a]["d_abs_vs_C0_floor"] = dict(zip(RATES, (2 * (np.abs(B) - np.abs(B0)).std(axis=0)).round(4).tolist()))
            print(fold, a, json.dumps(L[a]), flush=True)
        rep[fold] = L
    (GK.OUT / "chance_resid_posthoc.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
