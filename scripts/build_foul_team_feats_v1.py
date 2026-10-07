#!/usr/bin/env python
"""build_foul_team_feats_v1.py -- as-of team foul priors and the as-of league y_nt level for foul-accrual round 2
(possession_outcome experiments.md section 34.1, pre-registered in commit 0f83dee BEFORE any fit).

ONE function (`team_priors_asof`) serves both the training rows and the engine slates, so the two are identical by
construction. Everything is strictly before the query date (same season):
  defence committing rate  = (def_silent + def_trip) per defensive possession, x100
  offence drawn rate       = (opposing def_silent + def_trip) per offensive possession, x100
  in-season part           = team as-of rate - league as-of rate (pooled possessions, same dates)
  prior part               = team's previous-season final rate - that season's league rate (2022: 0)
  D / O = (P_in * r_in + K * r_prior) / (P_in + K),  K = 5 x panel mean possessions per team-game
  u = 1 / (1 + n_games_before)
League level: daily fit-window y_nt counts (`league_daily_y`); L(s,t) and Lv are fold-specific (n0, Lbar) and are
formed by the trainer (`level_asof`).

    .venv/Scripts/python.exe scripts/build_foul_team_feats_v1.py      # leak / sanity report only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ACC = ROOT / "data/processed/models/possession_outcome/round6/foul_accrual_poss_v2.parquet"
PRIOR_GAMES = 5.0


def team_games(acc: pd.DataFrame | None = None) -> pd.DataFrame:
    if acc is None:
        acc = pd.read_parquet(ACC, columns=["season", "game_id", "game_date", "offense_team_id", "defense_team_id",
                                            "def_silent", "def_trip"])
    a = acc.assign(f=acc["def_silent"] + acc["def_trip"])
    dfn = a.groupby(["season", "game_id", "game_date", "defense_team_id"]).agg(
        fc=("f", "sum"), pd_=("f", "size")).reset_index().rename(columns={"defense_team_id": "team_id"})
    off = a.groupby(["season", "game_id", "game_date", "offense_team_id"]).agg(
        fd=("f", "sum"), po=("f", "size")).reset_index().rename(columns={"offense_team_id": "team_id"})
    tg = dfn.merge(off, on=["season", "game_id", "game_date", "team_id"], how="outer").fillna(0.0)
    tg["game_date"] = pd.to_datetime(tg["game_date"]).dt.normalize()
    return tg.sort_values(["season", "team_id", "game_date", "game_id"]).reset_index(drop=True)


def _cum_before(daily: pd.DataFrame, keys: list[str], cols: list[str], q: pd.DataFrame) -> pd.DataFrame:
    """For each query row (keys + game_date), sums of `cols` over daily rows with date strictly before."""
    daily = daily.sort_values("game_date").copy()
    g = daily.groupby(keys, sort=False) if keys else None
    for c in cols:
        daily[f"cum_{c}"] = g[c].cumsum() if keys else daily[c].cumsum()
    qq = q.reset_index().rename(columns={"index": "_qi"}).sort_values("game_date")
    m = pd.merge_asof(qq, daily[keys + ["game_date"] + [f"cum_{c}" for c in cols]], on="game_date", by=keys or None,
                      allow_exact_matches=False, direction="backward")
    m = m.sort_values("_qi").set_index("_qi")
    out = pd.DataFrame(index=q.index)
    for c in cols:
        out[c] = m[f"cum_{c}"].fillna(0.0).to_numpy()
    return out


def team_priors_asof(tg: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """q: columns season, team_id, game_date (normalised). Returns D (committing), O (drawn), n, u, aligned with q."""
    q = q.copy()
    q["game_date"] = pd.to_datetime(q["game_date"]).dt.normalize()
    k = PRIOR_GAMES * float(pd.concat([tg["pd_"], tg["po"]]).mean())
    tday = tg.groupby(["season", "team_id", "game_date"], as_index=False)[["fc", "pd_", "fd", "po"]].sum()
    tday["ng"] = tg.groupby(["season", "team_id", "game_date"]).size().to_numpy()
    lday = tg.groupby(["season", "game_date"], as_index=False)[["fc", "pd_", "fd", "po"]].sum()
    T = _cum_before(tday, ["season", "team_id"], ["fc", "pd_", "fd", "po", "ng"], q[["season", "team_id", "game_date"]])
    Lg = _cum_before(lday, ["season"], ["fc", "pd_", "fd", "po"], q[["season", "game_date"]])
    with np.errstate(invalid="ignore", divide="ignore"):
        rin_c = np.where(T["pd_"] > 0, 100 * (T["fc"] / T["pd_"] - Lg["fc"] / Lg["pd_"]), 0.0)
        rin_d = np.where(T["po"] > 0, 100 * (T["fd"] / T["po"] - Lg["fd"] / Lg["po"]), 0.0)
    # previous-season final, centred on that season's league mean
    fin = tg.groupby(["season", "team_id"])[["fc", "pd_", "fd", "po"]].sum().reset_index()
    lfin = tg.groupby("season")[["fc", "pd_", "fd", "po"]].sum()
    fin["pc"] = 100 * (fin["fc"] / fin["pd_"] - (lfin["fc"] / lfin["pd_"]).reindex(fin["season"]).to_numpy())
    fin["pdr"] = 100 * (fin["fd"] / fin["po"] - (lfin["fd"] / lfin["po"]).reindex(fin["season"]).to_numpy())
    fin["season"] = fin["season"] + 1
    pr = q[["season", "team_id"]].merge(fin[["season", "team_id", "pc", "pdr"]], on=["season", "team_id"], how="left")
    pc = pr["pc"].fillna(0.0).to_numpy()
    pdr = pr["pdr"].fillna(0.0).to_numpy()
    out = pd.DataFrame(index=q.index)
    out["D"] = (T["pd_"].to_numpy() * rin_c + k * pc) / (T["pd_"].to_numpy() + k)
    out["O"] = (T["po"].to_numpy() * rin_d + k * pdr) / (T["po"].to_numpy() + k)
    out["n"] = T["ng"].to_numpy()
    out["u"] = 1.0 / (1.0 + out["n"])
    for c in out:
        out[c] = np.nan_to_num(out[c].to_numpy(dtype="float64"), nan=0.0)
    out.attrs["K"] = k
    return out


def league_daily_y(d: pd.DataFrame) -> pd.DataFrame:
    """daily fit-window y_nt sums per (season, game_date) from the possession design."""
    w = d[d["in_fit_window"] == 1]
    g = w.assign(game_date=pd.to_datetime(w["game_date"]).dt.normalize()).groupby(
        ["season", "game_date"])["y_nt"].agg(["sum", "size"]).reset_index()
    return g.rename(columns={"sum": "y", "size": "m"})


def level_asof(daily: pd.DataFrame, q: pd.DataFrame, n0: float, prior_by_season: dict) -> np.ndarray:
    """L(s,t) = (N_y,before + n0 * prior) / (N_before + n0); n0 = inf -> prior; no rows before with n0 = 0 -> prior."""
    q = q.copy()
    q["game_date"] = pd.to_datetime(q["game_date"]).dt.normalize()
    C = _cum_before(daily, ["season"], ["y", "m"], q[["season", "game_date"]])
    pri = q["season"].map(prior_by_season).to_numpy(dtype=float)
    if np.isinf(n0):
        return pri
    with np.errstate(invalid="ignore", divide="ignore"):
        L = (C["y"].to_numpy() + n0 * pri) / (C["m"].to_numpy() + n0)
    return np.where(np.isfinite(L), L, pri)


def main() -> None:
    """Sanity + leak report: strictly-before check and change-form correlations."""
    acc = pd.read_parquet(ACC, columns=["season", "game_id", "game_date", "offense_team_id", "defense_team_id",
                                        "def_silent", "def_trip", "start_score_diff", "offense_is_home"])
    tg = team_games(acc)
    q = tg[["season", "team_id", "game_date", "game_id"]].copy()
    f = team_priors_asof(tg, q)
    tg = pd.concat([tg, f], axis=1)
    tg["own_fc_rate"] = 100 * tg["fc"] / tg["pd_"].replace(0, np.nan)
    tg["own_fd_rate"] = 100 * tg["fd"] / tg["po"].replace(0, np.nan)
    # final margin per (game, team) from the box-free pbp: last start_score_diff is not final; use team points proxy absent ->
    # margin from hoopR team box (verified finals)
    from cbb_sim.eval import reference as R
    rows = []
    for s in sorted(tg["season"].unique()):
        a = R.load_actual_games(int(s))[["game_id", "home_team_id", "away_team_id", "home_score", "away_score"]]
        h = a.assign(team_id=a.home_team_id, margin=a.home_score - a.away_score)
        w = a.assign(team_id=a.away_team_id, margin=a.away_score - a.home_score)
        rows.append(pd.concat([h, w])[["game_id", "team_id", "margin"]])
    mg = pd.concat(rows)
    tg = tg.merge(mg, on=["game_id", "team_id"], how="left")
    g = tg.groupby(["season", "team_id"], sort=False)
    rep = {"K_poss": f.attrs["K"], "n_team_games": int(len(tg))}
    for c in ("D", "O"):
        # change entering the NEXT game = the update that includes this game's result (honest: expected nonzero);
        # change entering THIS game must not know this game's result (leak check)
        tg[f"d_in_{c}"] = tg[c] - g[c].shift(1)
        tg[f"d_next_{c}"] = g[c].shift(-1) - tg[c]
        own = "own_fc_rate" if c == "D" else "own_fd_rate"
        ok = tg[[f"d_in_{c}", f"d_next_{c}", "margin", own]].dropna()
        rep[c] = {"corr_change_in_vs_own_margin": float(ok[f"d_in_{c}"].corr(ok["margin"])),
                  "corr_change_in_vs_own_rate": float(ok[f"d_in_{c}"].corr(ok[own])),
                  "corr_change_next_vs_own_rate (honest update, expected > 0)": float(ok[f"d_next_{c}"].corr(ok[own])),
                  "corr_change_next_vs_own_margin": float(ok[f"d_next_{c}"].corr(ok["margin"])),
                  "sd": float(tg[c].std()), "share_exact_zero": float((tg[c] == 0).mean())}
    # first game of each team-season: in-season part must be 0 (strictly before)
    first = g.cumcount() == 0
    rep["first_game_n_max"] = float(tg.loc[first, "n"].max())
    print(json.dumps(rep, indent=1))
    out = ROOT / "results/foul_r2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "feat_leak_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
