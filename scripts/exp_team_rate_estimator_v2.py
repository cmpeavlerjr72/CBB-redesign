"""exp_team_rate_estimator_v1.py -- Stage A of the team-rate estimator bake-off
(docs/models/team_rate_estimator/experiments.md section 1, committed 005c7dd before any fit).

Builds the per-team-game box counts (2022-2025; 2026 SEALED and never loaded), the league as-of
level, the prior-season carry inputs (c_prev, roster continuity, coach change), fits every arm's
free parameters on the fold's TRAIN seasons that have a prior season (F1: 2023; F2: 2023+2024) by
next-game likelihood, and writes the as-of estimates for the fold's TEST season (F1: 2024; F2: 2025)
for the blind grader `scripts/grade_team_rate_estimator_v1.py`.

Writes only under results/team_rate_estimator/ (versioned *_v2 files; v1 untouched). Touches no served table.

Usage:
    .venv/Scripts/python.exe scripts/exp_team_rate_estimator_v1.py [--arms E0,E1,E2,E2c,E4,E3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import reference as R  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

OUT = Path("results/team_rate_estimator")
SEASONS = (2022, 2023, 2024, 2025)
FOLDS = {"F1": {"train": (2023,), "test": 2024}, "F2": {"train": (2023, 2024), "test": 2025}}
RATES = {  # name: (num col, den col, family)
    "tov": ("tov", "P", "binom"), "ftr": ("fta", "fga", "pois"), "share3": ("fga3", "fga", "binom"),
    "share_rim": ("fga_rim", "fga", "binom"), "make_rim": ("fgm_rim", "fga_rim", "binom"),
    "make_jump": ("fgm_jump", "fga_jump", "binom"), "make3": ("fgm3", "fga3", "binom"),
    "oreb": ("oreb", "reb_ch", "binom"),
}
EPS = 1e-6


# --------------------------------------------------------------------------- data
def team_games() -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        assert_not_sealed(s, context="team_rate_estimator stage A")
        b = R.load_actual_team_box(s)
        sh = R.load_team_shot_truth(s)
        b = b.merge(sh[["game_id", "team_id", "ev_fga_rim", "ev_fgm_rim", "ev_fga_jump2", "ev_fgm_jump2"]],
                    on=["game_id", "team_id"], how="left")
        fga2 = (b["fga"] - b["tpa"]).astype(float); fgm2 = (b["fgm"] - b["tpm"]).astype(float)
        eva = b["ev_fga_rim"] + b["ev_fga_jump2"]; evm = b["ev_fgm_rim"] + b["ev_fgm_jump2"]
        lsa = float(b["ev_fga_rim"].sum() / eva.sum()); lsm = float(b["ev_fgm_rim"].sum() / evm.sum())
        sa = np.where(eva > 0, b["ev_fga_rim"] / eva.where(eva > 0), lsa)
        sm = np.where(evm > 0, b["ev_fgm_rim"] / evm.where(evm > 0), lsm)
        t = pd.DataFrame({
            "season": s, "game_id": b["game_id"], "team_id": b["team_id"], "opp_id": b["opp_team_id"],
            "game_date": pd.to_datetime(b["game_date"]),
            "fga": b["fga"].astype(float), "fga3": b["tpa"].astype(float), "fgm3": b["tpm"].astype(float),
            "fta": b["fta"].astype(float), "tov": b["tov"].astype(float), "oreb": b["oreb"].astype(float),
            "reb_ch": (b["oreb"] + b["opp_dreb"]).astype(float),
        })
        t["fga_rim"] = fga2 * sa
        t["fga_jump"] = fga2 - t["fga_rim"]
        t["fgm_rim"] = np.minimum(fgm2 * sm, t["fga_rim"])
        t["fgm_jump"] = fgm2 - t["fgm_rim"]
        t["P"] = t["fga"] - t["oreb"] + t["tov"] + 0.44 * t["fta"]
        rows.append(t.dropna(subset=["reb_ch"]))
    return pd.concat(rows, ignore_index=True)


def continuity() -> pd.DataFrame:
    """Returning share of prior-season minutes, roster = athletes listed (incl. DNP) in the team's
    first 3 games of season s. Coach change from data/reference/coaches.parquet."""
    pb = {}
    for s in SEASONS:
        assert_not_sealed(s, context="team_rate_estimator continuity")
        p = pd.read_parquet(f"data/raw/hoopr/player_box/player_box_{s}.parquet",
                            columns=["game_id", "game_date", "athlete_id", "team_id", "minutes"])
        p["game_date"] = pd.to_datetime(p["game_date"])
        pb[s] = p
    out = []
    coaches = pd.read_parquet("data/reference/coaches.parquet")
    for s in SEASONS[1:]:
        cur = pb[s]
        first = cur[["team_id", "game_id", "game_date"]].drop_duplicates().sort_values(["team_id", "game_date", "game_id"])
        first["k"] = first.groupby("team_id").cumcount()
        g3 = first[first["k"] < 3][["team_id", "game_id"]]
        roster = cur.merge(g3, on=["team_id", "game_id"])[["team_id", "athlete_id"]].drop_duplicates()
        prev = pb[s - 1].groupby(["team_id", "athlete_id"], as_index=False)["minutes"].sum()
        tot = prev.groupby("team_id")["minutes"].sum().rename("tot")
        ret = prev.merge(roster, on=["team_id", "athlete_id"]).groupby("team_id")["minutes"].sum().rename("ret")
        c = pd.concat([tot, ret], axis=1).fillna(0.0)
        c["cont"] = np.where(c["tot"] >= 2000, c["ret"] / c["tot"].where(c["tot"] > 0), np.nan)  # full D-I seasons only
        c = c.reset_index()
        cs = coaches[coaches["season"] == s][["espn_team_id", "head_coach", "interim"]]
        cp = coaches[coaches["season"] == s - 1][["espn_team_id", "head_coach"]].rename(columns={"head_coach": "hc_prev"})
        cc = cs.merge(cp, on="espn_team_id", how="left")
        cc["coach_change"] = ((cc["head_coach"] != cc["hc_prev"]) & cc["hc_prev"].notna()) | cc["interim"].fillna(False).astype(bool)
        c = c.merge(cc[["espn_team_id", "coach_change"]].rename(columns={"espn_team_id": "team_id"}), on="team_id", how="left")
        c["coach_change"] = c["coach_change"].fillna(False).astype(float)
        c["season"] = s
        out.append(c[["season", "team_id", "cont", "coach_change", "tot"]])
    return pd.concat(out, ignore_index=True)


def build_panel(tg: pd.DataFrame):
    """Per rate-side: padded arrays over (team-season, game index)."""
    opp = tg.rename(columns={c: f"o_{c}" for c in tg.columns if c not in ("season", "game_id")})
    both = tg.merge(opp, left_on=["season", "game_id", "opp_id"], right_on=["season", "game_id", "o_team_id"])
    both = both.sort_values(["season", "team_id", "game_date", "game_id"]).reset_index(drop=True)
    both["j"] = both.groupby(["season", "team_id"]).cumcount()
    first_date = both.groupby("season")["game_date"].transform("min")
    both["week"] = ((both["game_date"] - first_date).dt.days // 7).astype(int)
    return both


def league_asof(tg: pd.DataFrame, num: str, den: str) -> tuple[pd.DataFrame, dict]:
    """L(s, date) over team-games strictly before date; day 0 -> prior season final (2022: its first week)."""
    day = tg.groupby(["season", "game_date"], as_index=False)[[num, den]].sum().sort_values(["season", "game_date"])
    day["cn"] = day.groupby("season")[num].cumsum() - day[num]
    day["cd"] = day.groupby("season")[den].cumsum() - day[den]
    fin = tg.groupby("season")[[num, den]].sum()
    final = (fin[num] / fin[den]).to_dict()
    wk1 = tg[tg["game_date"] < tg.groupby("season")["game_date"].transform("min") + pd.Timedelta(days=7)]
    w1 = wk1.groupby("season")[[num, den]].sum()
    day0 = {s: final.get(s - 1, float(w1.loc[s, num] / w1.loc[s, den])) for s in fin.index}
    day["L"] = np.where(day["cd"] > 0, day["cn"] / day["cd"].where(day["cd"] > 0), day["season"].map(day0))
    return day[["season", "game_date", "L"]], final


# --------------------------------------------------------------------------- estimators
class Panel:
    def __init__(self, df: pd.DataFrame, num: str, den: str, fam: str, carry: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        key = df[["season", "team_id"]].drop_duplicates().reset_index(drop=True)
        key["ts"] = np.arange(len(key))
        self.key = key
        d = df.merge(key, on=["season", "team_id"])
        self.ts = d["ts"].to_numpy(); self.j = d["j"].to_numpy()
        T, J = len(key), int(d["j"].max()) + 1
        self.T, self.J = T, J
        self.num = np.zeros((T, J)); self.den = np.zeros((T, J)); self.L = np.full((T, J), np.nan)
        self.mask = np.zeros((T, J), bool)
        self.num[self.ts, self.j] = d[num].to_numpy(); self.den[self.ts, self.j] = d[den].to_numpy()
        self.L[self.ts, self.j] = d["L"].to_numpy(); self.mask[self.ts, self.j] = True
        ck = key.merge(carry, on=["season", "team_id"], how="left")
        self.cprev = ck["c_prev"].fillna(0.0).to_numpy()
        self.cont = ck["cont"].to_numpy()
        self.coach = ck["coach_change"].fillna(0.0).to_numpy()
        self.fam = fam
        self.season = key["season"].to_numpy()
        self.adj = None
        self.opp = np.full((T, J), -1, dtype=np.int64); self.date = np.zeros((T, J), dtype="datetime64[ns]")
        self.gid = np.full((T, J), -1, dtype=np.int64)
        self.opp[self.ts, self.j] = d["opp_id"].to_numpy(); self.gid[self.ts, self.j] = d["game_id"].to_numpy()

    def cum_before(self):
        N = np.cumsum(self.num, axis=1) - self.num
        D = np.cumsum(self.den, axis=1) - self.den
        return N, D


def _rho(P: Panel, th_carry, cont_mean):
    """Carry weight: scalar rho (E2/E4/E3) or continuity/coach-dependent (E2c/E4c/E3c)."""
    if len(th_carry) == 1:
        return np.full(P.T, th_carry[0])
    cont = np.nan_to_num(P.cont - cont_mean, nan=0.0)
    return th_carry[0] + th_carry[1] * cont + th_carry[2] * P.coach


def estimate(P: Panel, arm: str, th: np.ndarray, cont_mean: float = 0.0, with_var: bool = False):
    """Returns c (T, J) centred estimate entering each game; with_var also returns v, the estimation variance
    (section 3.2): E0 sigma^2/D (nan at D=0); E1/E2/E2c sigma^2/(D+k); E4/E4c sigma^2/(Dw+k); E3/E3c the
    filter's posterior variance. sigma^2 = L(1-L) (binomial) or L (Poisson) at the row's league level.
    E9 variants use P.adj (den x opponent's as-of opposite-side estimate) to opponent-adjust observations."""
    L = np.nan_to_num(P.L, nan=0.5)
    sig2 = L if P.fam == "pois" else L * (1 - L)
    adj = P.adj if P.adj is not None else np.zeros_like(P.num)
    base = arm.replace("+opp", "")
    if base in ("E0", "E1", "E2", "E2c"):
        N, D = P.cum_before()
        A = np.cumsum(adj, axis=1) - adj
        if base == "E0":
            c = np.where(D > 0, (N - L * D - A) / np.where(D > 0, D, 1), 0.0)
            v = np.where(D > 0, sig2 / np.where(D > 0, D, 1), np.nan)
            return (c, v) if with_var else c
        k = np.exp(th[0])
        if base == "E1":
            prior = 0.0
        else:
            prior = (_rho(P, th[1:], cont_mean) * P.cprev)[:, None]
        c = (N - L * D - A + k * prior) / (D + k)
        v = sig2 / (D + k)
        return (c, v) if with_var else c
    if base in ("E4", "E4c"):
        lam = 1 / (1 + np.exp(-th[0])); k = np.exp(th[1]); rho = _rho(P, th[2:], cont_mean)
        c = np.zeros((P.T, P.J)); v = np.zeros((P.T, P.J))
        Nw = np.zeros(P.T); Dw = np.zeros(P.T); Aw = np.zeros(P.T)
        for j in range(P.J):
            c[:, j] = (Nw - L[:, j] * Dw - Aw + k * rho * P.cprev) / (Dw + k)
            v[:, j] = sig2[:, j] / (Dw + k)
            Nw = lam * Nw + P.num[:, j]; Dw = lam * Dw + P.den[:, j]; Aw = lam * Aw + adj[:, j]
        return (c, v) if with_var else c
    if base in ("E3", "E3c"):
        q = np.exp(th[0]); P0 = np.exp(th[1]); rho = _rho(P, th[2:], cont_mean)
        cc = rho * P.cprev.copy(); V = np.full(P.T, P0)
        c = np.zeros((P.T, P.J)); v = np.zeros((P.T, P.J))
        for j in range(P.J):
            c[:, j] = cc; v[:, j] = V
            den = P.den[:, j]; Lj = np.clip(L[:, j], 0.01, 0.99)
            has = P.mask[:, j] & (den > 0)
            dd = np.where(den > 0, den, 1)
            z = np.where(has, (P.num[:, j] - adj[:, j]) / dd - Lj, 0.0)
            var_obs = (Lj if P.fam == "pois" else Lj * (1 - Lj)) / dd
            Kg = np.where(has, V / (V + var_obs), 0.0)
            cc = cc + Kg * (z - cc)
            V = (1 - Kg) * V + q
        return (c, v) if with_var else c
    raise KeyError(arm)


def deviance(num, den, p, fam):
    p = np.clip(p, EPS, 1 - EPS) if fam == "binom" else np.clip(p, EPS, None)
    if fam == "binom":
        y = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
        a = np.where(num > 0, num * np.log(np.maximum(y, 1e-300) / p), 0.0)
        b = np.where(den - num > 0, (den - num) * np.log(np.maximum(1 - y, 1e-300) / (1 - p)), 0.0)
        return 2 * (a + b)
    mu = den * p
    a = np.where(num > 0, num * np.log(np.maximum(num, 1e-300) / mu), 0.0)
    return 2 * (a - (num - mu))


INIT = {"E1": [4.0], "E2": [4.0, 0.5], "E2c": [4.0, 0.5, 0.0, 0.0], "E4": [3.0, 4.0, 0.5], "E3": [-9.0, -6.0, 0.5],
        "E4c": [3.0, 4.0, 0.5, 0.0, 0.0], "E3c": [-9.0, -6.0, 0.5, 0.0, 0.0]}


def fit(P: Panel, arm: str, train_mask_ts: np.ndarray, cont_mean: float, warm=None) -> tuple[np.ndarray, float]:
    base = arm.replace("+opp", "")
    if base == "E0":
        return np.array([]), float("nan")
    m = P.mask & train_mask_ts[:, None] & (P.den > 0)
    L = np.nan_to_num(P.L, nan=0.5)
    if warm is not None:
        init = np.array(warm, float)
    else:
        init = np.array(INIT[base], float)
    if warm is None and base in ("E1", "E2", "E2c", "E4", "E4c"):  # k init ~ 3 games of denominator
        kidx = 1 if base.startswith("E4") else 0
        init[kidx] = np.log(3 * np.nanmean(P.den[m]) + 1e-6)
    if warm is None and base.startswith("E3"):
        base = (np.nanmean(L[m]) if P.fam == "pois" else np.nanmean(L[m] * (1 - L[m])))
        init[0] = np.log(base / np.nanmean(P.den[m]) * 0.01)
        init[1] = np.log(base / np.nanmean(P.den[m]) * 0.5)

    def obj(th):
        c = estimate(P, arm, th, cont_mean)
        d = deviance(P.num[m], P.den[m], L[m] + c[m], P.fam)
        v = float(d.sum())
        return v if np.isfinite(v) else 1e18

    r = minimize(obj, init, method="Nelder-Mead", options={"maxiter": 600 * len(init), "xatol": 1e-4, "fatol": 1e-3})
    return r.x, float(r.fun)


def side_frame(panel, num, den, side):
    cols = ["season", "game_id", "team_id", "opp_id", "game_date", "j", "week"]
    if side == "off":
        return panel[cols + [num, den]].copy()
    return panel[cols + [f"o_{num}", f"o_{den}"]].rename(columns={f"o_{num}": num, f"o_{den}": den}).copy()


def rows_out(P, te, c, v, num, den, fold, rate, side, arm, fam):
    sel = te[P.ts]
    dd = P.df.loc[sel, ["season", "game_id", "team_id", "opp_id", "j", "week", num, den, "L"]].rename(
        columns={num: "num", den: "den"}).copy()
    dd["c"] = c[P.ts[sel], P.j[sel]]
    dd["v"] = v[P.ts[sel], P.j[sel]]
    dd["c_lag1"] = np.where(P.j[sel] > 0, c[P.ts[sel], np.maximum(P.j[sel] - 1, 0)], np.nan)
    dd["c_prev"] = P.cprev[P.ts[sel]]
    dd["fold"] = fold; dd["rate"] = rate; dd["side"] = side; dd["arm"] = arm; dd["fam"] = fam
    return dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="E0,E1,E2,E2c,E4,E4c,E3,E3c")
    ap.add_argument("--e9", default=None, help="winner arm to opponent-adjust (writes estimates_e9_v2)")
    a = ap.parse_args()
    arms = a.arms.split(",") if a.e9 is None else [a.e9]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tg = team_games()
    cont = continuity()
    cont.to_parquet(OUT / "continuity_v2.parquet", index=False)
    panel = build_panel(tg)
    est_rows, params = [], {}
    for rate, (num, den, fam) in RATES.items():
        lg, final = league_asof(tg, num, den)
        Ps = {}
        for side in ("off", "def"):
            d = side_frame(panel, num, den, side).merge(lg, on=["season", "game_date"], how="left")
            fs = d.groupby(["season", "team_id"])[[num, den]].sum()
            fs["c"] = fs[num] / fs[den] - fs.index.get_level_values("season").map(final).to_numpy()
            fs = fs.reset_index()[["season", "team_id", "c"]]
            fs["season"] = fs["season"] + 1
            carry = fs.rename(columns={"c": "c_prev"}).merge(cont[["season", "team_id", "cont", "coach_change"]],
                                                           on=["season", "team_id"], how="outer")
            Ps[side] = Panel(d, num, den, fam, carry)
        for fold, fd in FOLDS.items():
            fitted = {}
            for side in ("off", "def"):
                P = Ps[side]
                tr = np.isin(P.season, fd["train"])
                cm = float(np.nanmean(P.cont[tr])) if np.isfinite(P.cont[tr]).any() else 0.0
                te = P.season == fd["test"]
                for arm in arms:
                    warm = None
                    if arm in ("E4c", "E3c") and (side, arm[:2]) in fitted:
                        warm = list(fitted[(side, arm[:2])]) + [0.0, 0.0]
                    th, fval = fit(P, arm, tr, cm, warm)
                    fitted[(side, arm)] = th
                    c, v = estimate(P, arm, th, cm, with_var=True)
                    params[f"{fold}|{rate}|{side}|{arm}"] = {"theta": th.tolist(), "train_dev": fval, "cont_mean": cm}
                    fitted[(side, arm, "c")] = c; fitted[(side, "cm")] = cm
                    if a.e9 is None:
                        est_rows.append(rows_out(P, te, c, v, num, den, fold, rate, side, arm, fam))
            if a.e9 is not None:
                w = a.e9
                # opponent's opposite-side estimate at the same game: team o's row in game g on the other panel
                for side, other in (("off", "def"), ("def", "off")):
                    Po = Ps[other]; co = fitted[(other, w, "c")]
                    lut = pd.Series(co[Po.ts, Po.j], index=pd.MultiIndex.from_arrays([Po.df["team_id"].to_numpy(), Po.df["game_id"].to_numpy()]))
                    P = Ps[side]
                    look = lut.reindex(pd.MultiIndex.from_arrays([P.df["opp_id"].to_numpy(), P.df["game_id"].to_numpy()])).fillna(0.0).to_numpy()
                    adj = np.zeros_like(P.num); adj[P.ts, P.j] = P.den[P.ts, P.j] * look
                    P.adj = adj
                for side in ("off", "def"):
                    P = Ps[side]
                    tr = np.isin(P.season, fd["train"]); te = P.season == fd["test"]; cm = fitted[(side, "cm")]
                    th, fval = fit(P, w + "+opp", tr, cm, list(fitted[(side, w)]))
                    c, v = estimate(P, w + "+opp", th, cm, with_var=True)
                    params[f"{fold}|{rate}|{side}|{w}+opp"] = {"theta": th.tolist(), "train_dev": fval, "cont_mean": cm}
                    est_rows.append(rows_out(P, te, c, v, num, den, fold, rate, side, w + "+opp", fam))
                    saved = P.adj
                    P.adj = None
                    c0, v0 = estimate(P, w, fitted[(side, w)], cm, with_var=True)
                    est_rows.append(rows_out(P, te, c0, v0, num, den, fold, rate, side, w, fam))
            print(f"{rate} {fold} done ({time.time() - t0:.0f}s)", flush=True)
    est = pd.concat(est_rows, ignore_index=True)
    tag = "estimates_v2" if a.e9 is None else "estimates_e9_v2"
    est.to_parquet(OUT / f"{tag}.parquet", index=False)
    json.dump(params, open(OUT / (f"params_v2.json" if a.e9 is None else "params_e9_v2.json"), "w"), indent=1)
    print("wrote", OUT / f"{tag}.parquet", est.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
