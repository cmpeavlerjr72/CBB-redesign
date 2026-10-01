"""train_usage_ft_late_v1.py -- state-aware FT-trip allocation: WHO is fouled when the leading team has the ball late
(lane I, 2026-10-01). Pre-registration: `docs/models/usage/experiments.md` section "Late-game FT-trip allocation" (lane I,
committed before this ran).

Served rule (U0): P(player i of the five draws the FT trip) = r_i / sum r, r_i the shrunk as-of FT-trip rate
(usage_params_v1 FT_trip: position prior, m = 200), exactly the engine's `UsageAdapter` (state-blind).
Arms multiply r_i by exp(beta * z_i) inside the late leading window only (offence ahead before the event, period >= 2,
usage sec_remaining <= 120), with z_i the player's shrunk FT ability, built from the four engine FT slot columns:
    z = (fta_asof * shooter_ft_asof + 30 * prior_season_ft) / (fta_asof + 30)     (league-centred; 30 = FT EB m)
    UL1: one beta (<= 120 s);  UL2: beta_60 (<= 60 s) and beta_120 (60-120 s).
Outside the window every arm equals U0 exactly. Fitted by maximum conditional likelihood (scipy, deterministic).

Data: `usage/events_v2.parquet` FT_trip rows with the five known and the drawer in it (lineup data exist for 2024 and 2025
only). score_diff is rebuilt PRE-outcome (the previous usage event's post-event score, as train_usage_v3 does).
Folds: F2 = train 2024, test 2025 (selects). F1 (train <= 2023) CANNOT be built (no lineups before 2023-24): the stated
substitute confirmation is train 2024 Nov-Jan, test 2024 Feb-Apr.
Floor: max(2 x paired game-bootstrap SE of the test log-loss difference, |LL(beta refit on a seed-1 bootstrap resample of
the training games) - LL(beta)|) -- the spec-identical retrain of a deterministic fit.

    .venv/Scripts/python.exe scripts/train_usage_ft_late_v1.py --out results/laneI_1001/usage_ft_late_v1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402

M_FT = 30.0
WIN = 120


def load_events():
    e = pd.read_parquet("data/processed/models/usage/events_v2.parquet")
    e = e.sort_values(["game_id", "period", "sec_in_period"], ascending=[True, True, False], kind="stable")
    hd = np.where(e["offense_is_home"], e["score_diff"], -e["score_diff"]).astype(np.int32)   # home-relative, POST event
    prev = pd.Series(hd, index=e.index).groupby(e["game_id"]).shift(1).fillna(0).to_numpy()
    e["sd_pre"] = np.where(e["offense_is_home"], prev, -prev)
    f = e[(e["event_class"] == "FT_trip") & e["five_ok"] & e["in_five"]].copy()
    return f


def ft_skill_table():
    ES.load_universe()
    att = pd.read_parquet("data/processed/models/free_throw/attempts_v1_era.parquet")
    att = att[att["season"].isin([2023, 2024, 2025])]
    d = FT.build_ft_design(att)
    pg = d.groupby(["season", "shooter_id", "game_date"], as_index=False).agg(fta=("y", "size"), ftm=("y", "sum"))
    pg = pg.sort_values(["season", "shooter_id", "game_date"])
    pg["cum_a"] = pg.groupby(["season", "shooter_id"])["fta"].cumsum()
    pg["cum_m"] = pg.groupby(["season", "shooter_id"])["ftm"].cumsum()
    prev = d.groupby(["season", "shooter_id"])["y"].agg(["sum", "size"]).reset_index()
    prev["season"] += 1
    prev["prev_rate"] = prev["sum"] / prev["size"]
    lg = d.groupby("game_date")["lg_ft_asof"].first().sort_index()
    return pg, prev[["season", "shooter_id", "prev_rate"]], lg


def attach_z(f, pg, prev, lg):
    long = []
    for k in range(1, 6):
        x = f[["season", "game_date"]].copy()
        x["pid"] = f[f"alt_{k}"].to_numpy()
        x["slot"] = k - 1
        x["row"] = np.arange(len(f))
        long.append(x)
    L = pd.concat(long, ignore_index=True)
    L["game_date"] = pd.to_datetime(L["game_date"])
    pgs = pg.rename(columns={"shooter_id": "pid"})[["season", "pid", "game_date", "cum_a", "cum_m"]].copy()
    pgs["game_date"] = pd.to_datetime(pgs["game_date"])
    L = L.sort_values("game_date", kind="stable")
    L = pd.merge_asof(L, pgs.sort_values("game_date"), on="game_date", by=["season", "pid"], direction="backward",
                      allow_exact_matches=False)
    L = L.merge(prev.rename(columns={"shooter_id": "pid"}), on=["season", "pid"], how="left")
    li = lg.index.searchsorted(L["game_date"].to_numpy(), side="right") - 1
    L["lg"] = lg.to_numpy()[np.clip(li, 0, len(lg) - 1)]
    a = L["cum_a"].fillna(0).to_numpy()
    own_c = np.where(a > 0, L["cum_m"].fillna(0).to_numpy() / np.maximum(a, 1) - L["lg"].to_numpy(), 0.0)
    prior_c = np.where(L["prev_rate"].notna(), L["prev_rate"].to_numpy() - L["lg"].to_numpy(), 0.0)
    L["z"] = (a * own_c + M_FT * prior_c) / (a + M_FT)
    Z = np.zeros((len(f), 5))
    Z[L["row"].to_numpy(), L["slot"].to_numpy()] = L["z"].to_numpy()
    return Z


def attach_r(f):
    up = json.loads(Path("data/processed/models/usage/usage_params_v1.json").read_text())["per_class"]["FT_trip"]
    m = float(up["shrink_m"])
    a = pd.read_parquet("data/processed/models/usage/asof_v2.parquet",
                        columns=["season", "player_id", "game_id", "exposure_asof", "ev_FT_trip", "pos_ev_FT_trip", "pos_exposure"])
    a["prior"] = a["pos_ev_FT_trip"] / a["pos_exposure"]
    a["prior"] = a["prior"].fillna(a["prior"].median())
    a["u"] = (m * a["prior"] + a["ev_FT_trip"]) / (m + a["exposure_asof"])
    key = a.set_index(["game_id", "player_id"])["u"]
    key = key[~key.index.duplicated()]
    R = np.zeros((len(f), 5))
    for k in range(5):
        idx = pd.MultiIndex.from_arrays([f["game_id"].to_numpy(), f[f"alt_{k + 1}"].to_numpy()])
        R[:, k] = key.reindex(idx).to_numpy()
    fill = np.nanmedian(R)
    R = np.where(np.isfinite(R) & (R > 0), R, fill)
    return R


def windows(f):
    lead = f["sd_pre"].to_numpy() > 0
    late = (f["period"].to_numpy() >= 2) & (f["sec_remaining"].to_numpy() <= WIN)
    s60 = f["sec_remaining"].to_numpy() <= 60
    return {"w60": lead & late & s60, "w120": lead & late & ~s60}


def ll_rows(beta, arm, R, Z, W, y):
    logit = np.log(R)
    if arm == "UL1":
        logit = logit + beta[0] * Z * (W["w60"] | W["w120"])[:, None]
    elif arm == "UL2":
        logit = logit + beta[0] * Z * W["w60"][:, None] + beta[1] * Z * W["w120"][:, None]
    logit = logit - logit.max(axis=1, keepdims=True)
    lp = logit - np.log(np.exp(logit).sum(axis=1, keepdims=True))
    return lp[np.arange(len(y)), y], np.exp(lp)


def fit(arm, R, Z, W, y):
    if arm == "U0":
        return np.zeros(0)
    k = 1 if arm == "UL1" else 2
    res = minimize(lambda b: -ll_rows(b, arm, R, Z, W, y)[0].sum(), np.zeros(k), method="L-BFGS-B")
    return res.x


def sub(R, Z, W, y, m):
    return R[m], Z[m], {k: v[m] for k, v in W.items()}, y[m]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    f = load_events().reset_index(drop=True)
    pg, prev, lg = ft_skill_table()
    Z = attach_z(f, pg, prev, lg)
    R = attach_r(f)
    W = windows(f)
    y = (f["y"].to_numpy().astype(int) - 1) if f["y"].max() == 5 else f["y"].to_numpy().astype(int)
    gd = pd.to_datetime(f["game_date"])
    folds = {"F2": (f["season"].to_numpy() == 2024, f["season"].to_numpy() == 2025),
             "F1sub": ((f["season"].to_numpy() == 2024) & (gd < "2024-02-01").to_numpy(),
                       (f["season"].to_numpy() == 2024) & (gd >= "2024-02-01").to_numpy())}
    win = W["w60"] | W["w120"]
    res = {"n_ft_trips": int(len(f)), "window_share": float(win.mean()), "y_coding": int(f["y"].min()),
           "folds": {}}
    rng = np.random.default_rng(1)
    for fold, (tr, te) in folds.items():
        out = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "n_test_window": int((te & win).sum()), "arms": {}}
        games_tr = f.loc[tr, "game_id"].unique()
        boot_games = set(rng.choice(games_tr, len(games_tr), replace=True))
        base = None
        for arm in ("U0", "UL1", "UL2"):
            b = fit(arm, *sub(R, Z, W, y, tr))
            lr, P = ll_rows(b, arm, *sub(R, Z, W, y, te))
            ww = win[te]
            r = {"beta": b.tolist(), "ll_window": float(lr[ww].mean()), "ll_all": float(lr.mean())}
            zt = Z[te]
            for lab, m in (("lead_le60", W["w60"][te]), ("lead_60_120", W["w120"][te]), ("outside", ~ww)):
                r[f"z_pred_{lab}"] = float((P[m] * zt[m]).sum(axis=1).mean())
                r[f"z_actual_{lab}"] = float(zt[m][np.arange(m.sum()), y[te][m]].mean())
            if arm != "U0":
                trb = tr & f["game_id"].isin(boot_games).to_numpy()
                bb = fit(arm, *sub(R, Z, W, y, trb))
                lrb, _ = ll_rows(bb, arm, *sub(R, Z, W, y, te))
                r["reseed_beta"] = bb.tolist()
                r["reseed_spread"] = float(abs(lrb[ww].mean() - lr[ww].mean()))
                dlt = (lr - base)[ww]
                gid = f.loc[te, "game_id"].to_numpy()[ww]
                g = pd.DataFrame({"g": gid, "d": dlt}).groupby("g")["d"].agg(["sum", "size"])
                bs = []
                rr = np.random.default_rng(20261001)
                for _ in range(400):
                    s = g.iloc[rr.integers(0, len(g), len(g))]
                    bs.append(s["sum"].sum() / s["size"].sum())
                r["d_ll_window"] = float(dlt.mean())
                r["boot_se"] = float(np.std(bs))
                r["floor"] = max(2 * r["boot_se"], r["reseed_spread"])
                r["floors"] = r["d_ll_window"] / r["floor"]
            else:
                base = lr
            out["arms"][arm] = r
            print(fold, arm, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        res["folds"][fold] = out
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
