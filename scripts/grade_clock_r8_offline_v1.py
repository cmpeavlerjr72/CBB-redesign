"""grade_clock_r8_offline_v1.py -- clock round 8 OFFLINE grade (experiments.md section 36).

Lane H, 2026-10-01. ONE grader, blind over every arm (C0 = r7_C0 = L2, A2 = r7_A2, M1, M2,
M2D, G2D), both folds. Truth: test-season rows of the L2 v4 design, clock-complete games,
regulation, L2's horn censoring; each arm's own S1 schedule routed by game date.

Per arm and fold:
  deviance                 -2 mean log-lik (primary)
  slope_oq / slope_dq      offence / defence team quintile slope of E[min(T,R)] on actual
  elast_ratio              game-level elasticity of log(1200/mean e) on X (month FE) / actual's
  slope_gq                 game-prior quintile slope
  game_cal_slope           actual count on model count across games (target 1)
  between_sd_ratio         SD(model count) / (corr * SD(actual count))
  count_gap                implied possessions per team-game, model - actual
  gap_by_month / gap_by_start (seconds)
  segments                 month, site, conference tier: slope_oq, elast_ratio, count gap
Floors: 2 x paired game-block bootstrap SE (200 draws), arm - C0, and arm - A2 for the ratio.
Writes results/clock_r8/offline_grade.json and results/clock_r8/rows_<fold>.parquet.
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
from cbb_sim.models import clock_r8  # noqa: E402,F401  (unpickling)
from cbb_sim.models import clock_v3 as c3  # noqa: E402

CKD = ROOT / "data/processed/models/clock"
L2 = CKD / "r6_L2"
OUT = ROOT / "results/clock_r8"
ARMS = {"C0": "r7_C0", "A2": "r7_A2", "M1": "r8_M1", "M2": "r8_M2", "M2D": "r8_M2D", "G2D": "r8_G2D"}
CHUNK = 20000
NBOOT = 200
POWER = {"ACC", "Big Ten", "Big 12", "SEC", "Big East"}
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


def wls_slope(y, x, w, Z=None):
    """weighted OLS coefficient on x (with intercept and optional extra columns Z)."""
    cols = [np.ones(len(x)), x] + ([] if Z is None else [Z])
    A = np.column_stack(cols)
    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return float(b[1])


class GameAgg:
    """Per-game sufficient statistics so every line can be bootstrapped by game weights."""

    def __init__(self, d: pd.DataFrame, arms: list[str]):
        gi, games = pd.factorize(d["game_id"])
        self.n_games = len(games)
        self.gi = gi
        self.n = np.bincount(gi).astype(float)
        self.dur = np.bincount(gi, weights=d["duration_s"].to_numpy(dtype=float))
        first = d.groupby(gi).first()
        self.X = (np.log(first["off_tempo_rel"].clip(0.5, 2.0)) + np.log(first["def_tempo_rel"].clip(0.5, 2.0))).to_numpy()
        mo = pd.to_datetime(first["game_date"]).dt.month.astype(int)
        self.M = pd.get_dummies(mo, drop_first=True).to_numpy(dtype=float)
        self.gq = first["gq_game"].to_numpy()
        self.e = {a: np.bincount(gi, weights=d[f"e_{a}"].to_numpy()) for a in arms}
        self.ll = {a: np.bincount(gi, weights=d[f"ll_{a}"].to_numpy()) for a in arms}
        # per game x quintile sums for team slopes
        self.q = {}
        for q in ("oq", "dq"):
            qq = d[q].to_numpy().astype(int)
            S = np.zeros((self.n_games, 5)); N = np.zeros((self.n_games, 5)); A = np.zeros((self.n_games, 5))
            np.add.at(N, (gi, qq), 1.0)
            np.add.at(A, (gi, qq), d["duration_s"].to_numpy(dtype=float))
            Se = {}
            for a in arms:
                E = np.zeros((self.n_games, 5))
                np.add.at(E, (gi, qq), d[f"e_{a}"].to_numpy())
                Se[a] = E
            self.q[q] = (N, A, Se)

    def lines(self, a: str, w: np.ndarray) -> dict:
        n = w @ self.n
        out = {"deviance": float(-2 * (w @ self.ll[a]) / n)}
        out["count_gap"] = float(1200 * n / (w @ self.e[a]) - 1200 * n / (w @ self.dur))
        for q in ("oq", "dq"):
            N, A, Se = self.q[q]
            nn = w @ N
            m = (w @ Se[a]) / nn
            act = (w @ A) / nn
            out[f"slope_{q}"] = float(np.polyfit(act, m, 1)[0])
        pm = 1200 * self.n / self.e[a]
        pa = 1200 * self.n / self.dur
        keep = w > 0
        ww = w[keep]
        out["elast_law"] = wls_slope(np.log(pm[keep]), self.X[keep], ww, self.M[keep])
        out["elast_act"] = wls_slope(np.log(pa[keep]), self.X[keep], ww, self.M[keep])
        out["elast_ratio"] = out["elast_law"] / out["elast_act"]
        out["game_cal_slope"] = wls_slope(pa[keep], pm[keep], ww)
        mp = np.average(pm[keep], weights=ww); ma = np.average(pa[keep], weights=ww)
        vp = np.average((pm[keep] - mp) ** 2, weights=ww); va = np.average((pa[keep] - ma) ** 2, weights=ww)
        cv = np.average((pm[keep] - mp) * (pa[keep] - ma), weights=ww)
        corr = cv / np.sqrt(vp * va)
        out["between_sd_ratio"] = float(np.sqrt(vp) / (corr * np.sqrt(va)))
        gq = self.gq[keep]
        gm = pd.DataFrame({"m": pm[keep], "a": pa[keep], "w": ww, "q": gq})
        t = gm.groupby("q").apply(lambda g: pd.Series({"m": np.average(g["m"], weights=g["w"]),
                                                      "a": np.average(g["a"], weights=g["w"])}),
                                  include_groups=False)
        out["slope_gq_count"] = float(np.polyfit(t["a"], t["m"], 1)[0])
        return out


def main():
    univ = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet")
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    design = pd.read_parquet(L2 / "design_v2.parquet", columns=[c for c in COLS if c != "censored"] + ["censored"])
    design, _ = c3.attach_horn_censoring(design, [2022, 2023, 2024, 2025], censor_dir=L2 / "clock_censoring")
    c3.set_flag_inplace(design, "horn")
    design["days_since_start"] = clock_r8.season_day(design)
    # conference tier
    tiers = []
    for s in (2024, 2025):
        cg = pd.read_parquet(ROOT / f"data/raw/cbbd/games_{s}.parquet", columns=["id", "homeConference", "awayConference"])
        tiers.append(cg.rename(columns={"id": "cbbd_game_id"}))
    cg = pd.concat(tiers, ignore_index=True)
    cg["npow"] = cg["homeConference"].isin(POWER).astype(int) + cg["awayConference"].isin(POWER).astype(int)
    u2 = univ[["game_id", "cbbd_game_id"]].merge(cg[["cbbd_game_id", "npow"]], on="cbbd_game_id", how="left")
    tier_of = u2.drop_duplicates("game_id").set_index("game_id")["npow"]
    rep = {"floor_rule": "2 x paired game-block bootstrap SE (200 draws); reseed floor 0 by construction (proved separately)",
           "folds": {}}
    arms = [a for a in ARMS if (CKD / ARMS[a] / "F2" / "manifest.json").exists()]
    for fold, season in (("F2", 2025), ("F1", 2024)):
        fa = [a for a in arms if (CKD / ARMS[a] / fold / "manifest.json").exists()]
        d = design[(design["season"] == season) & (design["period"] <= 2.0)
                   & design["game_id"].isin(cc)].reset_index(drop=True)
        d["month"] = pd.to_datetime(d["game_date"]).dt.month
        d["site"] = np.where(d["site_home"] + d["site_away"] == 0, "neutral",
                             np.where(d["site_home"] > 0, "off_home", "off_away"))
        d["tier"] = d["game_id"].map(tier_of).map({2: "both_power", 1: "one_power", 0: "neither_power"}).fillna("unknown")
        d["oq"] = d["offense_team_id"].map(pd.qcut(d.groupby("offense_team_id")["off_tempo_rel"].mean(), 5, labels=False))
        d["dq"] = d["defense_team_id"].map(pd.qcut(d.groupby("defense_team_id")["def_tempo_rel"].mean(), 5, labels=False))
        gp = d.groupby("game_id")["tempo_prior_game"].first()
        d["gq_game"] = d["game_id"].map(pd.qcut(gp.rank(method="first"), 5, labels=False))
        fr = {"n_rows": int(len(d)), "n_games": int(d["game_id"].nunique()), "arms": {}}
        for arm in fa:
            e, ll = score(load_schedule(CKD / ARMS[arm] / fold / "manifest.json"), d)
            d[f"e_{arm}"] = e
            d[f"ll_{arm}"] = ll
            print(fold, arm, "scored", flush=True)
        G = GameAgg(d, fa)
        w1 = np.ones(G.n_games)
        for arm in fa:
            a = G.lines(arm, w1)
            st = d.groupby("start_reason").agg(m=(f"e_{arm}", "mean"), a=("duration_s", "mean"))
            a["gap_by_start"] = (st["m"] - st["a"]).round(4).to_dict()
            mo = d.groupby("month").agg(m=(f"e_{arm}", "mean"), a=("duration_s", "mean"), n=("duration_s", "size"))
            a["gap_by_month"] = {str(k): round(v, 4) for k, v in (mo["m"] - mo["a"]).items()}
            a["n_by_month"] = mo["n"].astype(int).to_dict()
            segs = {}
            for by in ("month", "site", "tier"):
                for k, g in d.groupby(by):
                    ng = g["game_id"].nunique()
                    s = {"n_rows": int(len(g)), "n_games": int(ng)}
                    s["count_gap"] = float(1200 / g[f"e_{arm}"].mean() - 1200 / g["duration_s"].mean())
                    if len(g) >= 3000:
                        t = g.groupby("oq").agg(m=(f"e_{arm}", "mean"), a=("duration_s", "mean"))
                        s["slope_oq"] = float(np.polyfit(t["a"], t["m"], 1)[0]) if len(t) > 1 else None
                    else:
                        s["slope_oq"] = "UNDERPOWERED"
                    if by != "site" and ng >= 150:
                        ge = clock_r8.game_elasticity(g, g[f"e_{arm}"].to_numpy())
                        s["elast_ratio"] = ge["e_law"] / ge["e_act"]
                        s["elast_law"], s["elast_act"] = ge["e_law"], ge["e_act"]
                    elif by != "site":
                        s["elast_ratio"] = "UNDERPOWERED"
                    segs[f"{by}={k}"] = s
            a["segments"] = segs
            fr["arms"][arm] = a
            print(fold, arm, json.dumps({k: round(v, 4) for k, v in a.items() if isinstance(v, float)}), flush=True)
        # paired game bootstrap
        rng = np.random.default_rng(20261001)
        keys = ["deviance", "slope_oq", "slope_dq", "elast_ratio", "game_cal_slope", "between_sd_ratio",
                "count_gap", "slope_gq_count"]
        boots = {arm: {k: [] for k in keys} for arm in fa}
        for _ in range(NBOOT):
            w = np.bincount(rng.integers(0, G.n_games, G.n_games), minlength=G.n_games).astype(float)
            L = {arm: G.lines(arm, w) for arm in fa}
            for arm in fa:
                for k in keys:
                    boots[arm][k].append(L[arm][k])
        c0 = fr["arms"]["C0"]
        for arm in fa:
            if arm == "C0":
                continue
            a = fr["arms"][arm]
            vs = {}
            for k in keys:
                diff = np.array(boots[arm][k]) - np.array(boots["C0"][k])
                vs[f"d_{k}"] = a[k] - c0[k]
                vs[f"d_{k}_floor"] = 2 * float(np.std(diff))
            # |ratio - 1| vs A2 and vs C0
            for ref in ("A2", "C0"):
                if ref in fa and ref != arm:
                    da = np.abs(np.array(boots[arm]["elast_ratio"]) - 1) - np.abs(np.array(boots[ref]["elast_ratio"]) - 1)
                    vs[f"d_absratio_vs_{ref}"] = abs(a["elast_ratio"] - 1) - abs(fr["arms"][ref]["elast_ratio"] - 1)
                    vs[f"d_absratio_vs_{ref}_floor"] = 2 * float(np.std(da))
            da = np.abs(np.array(boots[arm]["slope_oq"]) - 1) - np.abs(np.array(boots["C0"]["slope_oq"]) - 1)
            vs["d_absslope_oq_vs_C0"] = abs(a["slope_oq"] - 1) - abs(c0["slope_oq"] - 1)
            vs["d_absslope_oq_vs_C0_floor"] = 2 * float(np.std(da))
            vs["d_abs_count_gap"] = abs(a["count_gap"]) - abs(c0["count_gap"])
            vs["max_start_gap_worsening_s"] = float(max(abs(a["gap_by_start"][k]) - abs(c0["gap_by_start"][k])
                                                       for k in c0["gap_by_start"] if k != "other"))
            vs["max_month_gap_worsening_s_NovMar"] = float(max(abs(a["gap_by_month"][k]) - abs(c0["gap_by_month"][k])
                                                              for k in ("11", "12", "1", "2", "3") if k in c0["gap_by_month"]))
            a["vs_C0"] = vs
            print(fold, arm, "vs C0", json.dumps({k: round(v, 4) for k, v in vs.items()}), flush=True)
        keep = ["game_id", "offense_team_id", "defense_team_id", "start_reason", "month", "site", "tier", "oq", "dq",
                "gq_game", "duration_s", "censored", *[f"e_{a}" for a in fa], *[f"ll_{a}" for a in fa]]
        d[keep].to_parquet(OUT / f"rows_{fold}.parquet", index=False)
        rep["folds"][fold] = fr
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "offline_grade.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
