"""grade_clock_r8_chance_offline_v1.py -- clock round 8 amendment OFFLINE grade (experiments.md
section 38). Lane H, 2026-10-01. One grader, blind over C0 (= L2), M2D, K1, K2, K2M, both folds.

Truth: test-season possessions of the L2 v4 design (clock-complete games, regulation, horn
censoring), joined to chances_v4 (d1, chance-1 end class, chance count) and possessions_v4 (OREB,
FGM, TOV, FTA). Per possession, each arm's implied time GIVEN the realised chances:
  C0, M2D          E[min(T, R)]                              (marginal possession law)
  K1               min(E[min(T1, R)] + sum of continuation means, R)
  K2, K2M          min(E[min(T1 | end1, R)] + sum of continuation means, R)
Lines per arm: per-game implied possessions (1200 / mean time) regressed on the game's OREB, FGM,
TOV, FTA (vs the actual count's slopes on the same games); per-game count calibration slope; count
gap; offence-quintile slope; game elasticity ratio; first-chance deviance (K arms).
Floors: 2 x paired game-bootstrap SE (200 draws) of arm - C0.
Writes results/clock_r8/chance_offline_grade.json
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
from cbb_sim.models import clock as CK  # noqa: E402
from cbb_sim.models import clock_r7  # noqa: E402,F401
from cbb_sim.models import clock_r8  # noqa: E402
from cbb_sim.models import clock_v3 as c3  # noqa: E402
import grade_clock_r8_offline_v1 as G  # noqa: E402
import train_clock_r8_chance_v1 as TK  # noqa: E402

CKD = ROOT / "data/processed/models/clock"
OUT = ROOT / "results/clock_r8"
ARMS = {"C0": "r7_C0", "M2D": "r8_M2D", "K1": "r8_K1", "K2": "r8_K2", "K2M": "r8_K2M"}
K_ARMS = ("K1", "K2", "K2M")
NBOOT = 200
GRID = np.arange(CK.DURATION_CAP + 1, dtype=np.float64)


def score_k(sched, d):
    """Per row: e1 marginal, e1 conditional, log-lik of d1 under each, continuation means."""
    cuts = np.array([np.datetime64(r[0], "ns") for r in sched])
    seg = np.searchsorted(cuts, pd.to_datetime(d["game_date"]).to_numpy(), side="right") - 1
    n = len(d)
    e_m, e_c, ll_m, ll_c, cont = (np.empty(n) for _ in range(5))
    y = np.clip(d["duration_s"].to_numpy().astype(np.int64), 0, CK.DURATION_CAP)
    cen = d["censored"].to_numpy(dtype=bool)
    nch = d["n_ch"].to_numpy().astype(np.int64)
    for k in np.unique(seg):
        rows = np.flatnonzero(seg == k)
        arm = sched[int(k)][1]
        mc = arm.mean_cont()
        cont[rows] = np.where(nch[rows] >= 2, mc[0], 0.0) + np.maximum(nch[rows] - 2, 0) * mc[1]
        for lo in range(0, len(rows), G.CHUNK):
            r = rows[lo:lo + G.CHUNK]
            blk = d.iloc[r].reset_index(drop=True)
            R = blk["seconds_remaining"].to_numpy(dtype=np.float64)[:, None]
            for fn, e_out, ll_out in ((arm.pmf, e_m, ll_m), (arm.pmf_end, e_c, ll_c)):
                p = np.asarray(fn(blk), dtype=np.float64)
                e_out[r] = (p * np.minimum(GRID[None, :], R)).sum(axis=1)
                pu = p[np.arange(len(r)), y[r]]
                S = 1.0 - np.cumsum(p, axis=1)[np.arange(len(r)), y[r]] + pu
                ll_out[r] = np.log(np.clip(np.where(cen[r], S, pu), 1e-12, None))
    return e_m, e_c, ll_m, ll_c, cont


def wls(Y, X, w):
    A = np.column_stack([np.ones(len(Y)), X])
    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(A * sw[:, None], Y * sw, rcond=None)
    return b[1:]


def lines(g: pd.DataFrame, arms, w):
    """g: per-game frame (sums). Returns per-arm dict."""
    keep = w > 0
    ww = w[keep]
    X = g.loc[keep, ["oreb", "fgm", "tov", "fta"]].to_numpy(float)
    pa = 1200.0 * g.loc[keep, "n"].to_numpy() / g.loc[keep, "dur"].to_numpy()
    ba = wls(pa, X, ww)
    out = {"actual": {"N_on_oreb": float(ba[0]), "N_on_fgm": float(ba[1]), "N_on_tov": float(ba[2]),
                      "N_on_fta": float(ba[3])}}
    n_all = float(ww @ g.loc[keep, "n"].to_numpy())
    dur_all = float(ww @ g.loc[keep, "dur"].to_numpy())
    for a in arms:
        pm = 1200.0 * g.loc[keep, "n"].to_numpy() / g.loc[keep, f"t_{a}"].to_numpy()
        b = wls(pm, X, ww)
        cal = wls(pa, pm[:, None], ww)[0]
        gap = 1200.0 * n_all / float(ww @ g.loc[keep, f"t_{a}"].to_numpy()) - 1200.0 * n_all / dur_all
        out[a] = {"N_on_oreb": float(b[0]), "N_on_fgm": float(b[1]), "N_on_tov": float(b[2]), "N_on_fta": float(b[3]),
                  "game_cal_slope": float(cal), "count_gap": float(gap),
                  "abs_oreb_err": float(abs(b[0] - ba[0])), "abs_fgm_err": float(abs(b[1] - ba[1]))}
    return out


def main():
    univ = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet")
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    design = TK.T.load_design()
    first, _ = TK.chance_tables([2023, 2024, 2025])
    design = design[design["season"].isin([2024, 2025])]
    design = TK.build_d1_design(design, first)
    rep = {"folds": {}}
    for fold, season in (("F2", 2025), ("F1", 2024)):
        arms = [a for a in ARMS if (CKD / ARMS[a] / fold / "manifest.json").exists()]
        d = design[(design["season"] == season) & (design["period"] <= 2.0) & design["game_id"].isin(cc)].reset_index(drop=True)
        # whole-possession truth and counts from possessions_v4
        p4 = pd.read_parquet(TK.P4 / f"possessions_{season}.parquet",
                             columns=["game_id", "period", "poss_index", "duration_s", "oreb_count", "fgm_rim",
                                      "fgm_jump2", "fgm_3", "fta", "terminal_event"])
        p4["period"] = p4["period"].astype("int64"); p4["poss_index"] = p4["poss_index"].astype("int64")
        d["_p"] = d["period"].astype("int64"); d["_i"] = d["poss_index"].astype("int64")
        d = d.merge(p4.rename(columns={"period": "_p", "poss_index": "_i", "duration_s": "dur_poss", "terminal_event": "term_poss"}),
                    on=["game_id", "_p", "_i"], how="left", validate="1:1")
        d["R"] = d["seconds_remaining"].to_numpy(float)
        rows = {}
        # marginal possession laws: score on the WHOLE-possession frame (truth irrelevant to e)
        for a in ("C0", "M2D"):
            if a not in arms:
                continue
            dd = d.copy()
            dd["duration_s"] = np.clip(dd["dur_poss"].to_numpy(dtype=np.int64), 0, CK.DURATION_CAP)
            dd["censored"] = dd["censored_poss"]
            e, ll = G.score(G.load_schedule(CKD / ARMS[a] / fold / "manifest.json"), dd)
            rows[a] = e
        res = {}
        for a in K_ARMS:
            if a not in arms:
                continue
            e_m, e_c, ll_m, ll_c, cont = score_k(G.load_schedule(CKD / ARMS[a] / fold / "manifest.json"), d)
            e1 = e_m if a == "K1" else e_c
            rows[a] = np.minimum(e1 + cont, d["R"].to_numpy())
            res[a] = {"d1_dev_marginal": float(-2 * ll_m.mean()), "d1_dev_conditional": float(-2 * ll_c.mean()),
                      "d1_gap_marginal_s": float(e_m.mean() - d["duration_s"].mean()),
                      "d1_gap_conditional_s": float(e_c.mean() - d["duration_s"].mean()),
                      "cont_mean_implied_s": float(cont.mean())}
            print(fold, a, json.dumps({k: round(v, 4) for k, v in res[a].items()}), flush=True)
        for a, t in rows.items():
            d[f"t_{a}"] = t
        ar = list(rows)
        d["fgm"] = d["fgm_rim"] + d["fgm_jump2"] + d["fgm_3"]
        d["tov"] = (d["term_poss"] == "TOV").astype(float)
        g = d.groupby("game_id").agg(n=("dur_poss", "size"), dur=("dur_poss", "sum"), oreb=("oreb_count", "sum"),
                                     fgm=("fgm", "sum"), tov=("tov", "sum"), fta=("fta", "sum"),
                                     **{f"t_{a}": (f"t_{a}", "sum") for a in ar})
        # team-slope / elasticity lines via the section-36 machinery (implied time as e)
        d["oq"] = d["offense_team_id"].map(pd.qcut(d.groupby("offense_team_id")["off_tempo_rel"].mean(), 5, labels=False))
        L = lines(g, ar, np.ones(len(g)))
        for a in ar:
            t = d.groupby("oq").agg(m=(f"t_{a}", "mean"), a=("dur_poss", "mean"))
            L[a]["slope_oq"] = float(np.polyfit(t["a"], t["m"], 1)[0])
            dd = d.assign(duration_s=d["dur_poss"])
            ge = clock_r8.game_elasticity(dd, d[f"t_{a}"].to_numpy())
            L[a]["elast_ratio"] = ge["e_law"] / ge["e_act"]
            if a in res:
                L[a].update(res[a])
        rng = np.random.default_rng(20261001)
        keys = ["N_on_oreb", "N_on_fgm", "abs_oreb_err", "abs_fgm_err", "game_cal_slope", "count_gap"]
        boots = {a: {k: [] for k in keys} for a in ar}
        for _ in range(NBOOT):
            w = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g)).astype(float)
            Lb = lines(g, ar, w)
            for a in ar:
                for k in keys:
                    boots[a][k].append(Lb[a][k])
        for a in ar:
            if a == "C0":
                continue
            L[a]["vs_C0"] = {}
            for k in keys:
                diff = np.array(boots[a][k]) - np.array(boots["C0"][k])
                L[a]["vs_C0"][f"d_{k}"] = L[a][k] - L["C0"][k]
                L[a]["vs_C0"][f"d_{k}_floor"] = 2 * float(np.std(diff))
            L[a]["vs_C0"]["d_abs_count_gap"] = abs(L[a]["count_gap"]) - abs(L["C0"]["count_gap"])
        for a in ar:
            print(fold, a, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in L[a].items()}), flush=True)
        print(fold, "actual", L["actual"], flush=True)
        rep["folds"][fold] = {"n_games": int(len(g)), "n_rows": int(len(d)), "lines": L}
    (OUT / "chance_offline_grade.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
