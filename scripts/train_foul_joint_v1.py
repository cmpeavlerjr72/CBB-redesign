#!/usr/bin/env python
"""
train_foul_joint_v1.py -- possession-outcome round 7 (joint foul accrual + FT-trip
production), OFFLINE arms. Pre-registration: `docs/models/possession_outcome/
experiments.md` section 20 (commit aa0ee91, pushed BEFORE this script ran).

    .venv/Scripts/python.exe scripts/train_foul_joint_v1.py --n-jobs 6

Blocks (20.2): A (non-trip accrual, engine attribution), D/O (split attribution:
defence non-trip + offensive fouls to the offence), T (FT-trip classes as a logit
offset on the PO proxy `T0`). Every arm is scored on TEST rows fed the TRUE
(engine-definition) foul state. Seeds 0 and 7 for every stochastic arm.

Writes data/processed/models/possession_outcome/round7/preds_{block}_{fold}_seed{s}.parquet
and the fitted fold-2 objects needed to export engine LUTs (`round7/fits_F2.joblib`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from train_foul_accrual_v1 import build_features  # noqa: E402

R6 = ROOT / "data/processed/models/possession_outcome/round6"
OUT = ROOT / "data/processed/models/possession_outcome/round7"
FOLDS = {"F1": {"train": [2022, 2023], "test": [2024]},
         "F2": {"train": [2022, 2023, 2024], "test": [2025]}}
SEASONS = [2022, 2023, 2024, 2025]
BONUS, DBONUS = 6, 9
MAXF = 10
SEEDS = (0, 7)

GBM_KW = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
              min_child_samples=500, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, n_jobs=1, verbose=-1)

# true (engine-definition) state and its labelled twin, same column order
STATE_T = ["period", "sec_rem", "margin", "abs_margin", "game_seconds", "def_f_t", "off_f_t",
           "offb_t", "defb_t", "offdb_t", "site_home", "site_away", "is_ot"]
STATE_L = ["period", "sec_rem", "margin", "abs_margin", "game_seconds", "def_f_l", "off_f_l",
           "offb_l", "defb_l", "offdb_l", "site_home", "site_away", "is_ot"]


# ---------------------------------------------------------------------------
# possession design (blocks A, D, O)
# ---------------------------------------------------------------------------
def possession_design() -> pd.DataFrame:
    d = pd.read_parquet(R6 / "foul_accrual_poss_v2.parquet")
    trips = []
    for s in SEASONS:
        c = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{s}.parquet",
                            columns=["game_id", "period", "poss_index", "terminal_event", "and_one"])
        c["trip"] = (c["terminal_event"].isin(["FT_trip_shooting", "FT_trip_bonus"]).astype(int)
                     + c["and_one"].astype(int))
        c["tov"] = (c["terminal_event"] == "TOV").astype(int)
        p = c.groupby(["game_id", "period", "poss_index"]).agg(
            trips=("trip", "sum"), ended_tov=("tov", "max")).reset_index()
        p["season"] = s
        trips.append(p)
    tp = pd.concat(trips, ignore_index=True)
    d = d.merge(tp, on=["season", "game_id", "period", "poss_index"], how="inner",
                validate="one_to_one")
    d = build_features(d)
    tot = d["def_silent"] + d["def_trip"] + d["off_silent"] + d["off_trip"]
    d["y_nt"] = ((tot - d["trips"]) >= 1).astype("int8")
    d["y_dnt"] = ((d["def_silent"] + d["def_trip"] - d["trips"]).clip(lower=0) >= 1).astype("int8")
    d["y_off"] = ((d["off_silent"] + d["off_trip"]) >= 1).astype("int8")
    d["def_f_t"] = d["def_team_fouls_true"].astype("float32")
    d["off_f_t"] = d["off_team_fouls_true"].astype("float32")
    d["def_f_l"] = d["def_team_fouls"].astype("float32")
    d["off_f_l"] = d["off_team_fouls"].astype("float32")
    for sfx in ("t", "l"):
        d[f"offb_{sfx}"] = (d[f"def_f_{sfx}"] >= BONUS).astype("float32")
        d[f"defb_{sfx}"] = (d[f"off_f_{sfx}"] >= BONUS).astype("float32")
        d[f"offdb_{sfx}"] = (d[f"def_f_{sfx}"] >= DBONUS).astype("float32")
    d["ended_tov"] = d["ended_tov"].astype("float32")
    d["dcell"] = d["def_f_t"].clip(0, MAXF).astype(int)
    return d


def _cells(tr_keys: np.ndarray, tr_y: np.ndarray, te_keys: np.ndarray, prior: float,
           k: float = 200.0) -> np.ndarray:
    """round-6 `_cells`: Laplace-smoothed cell means, k=200 toward the train rate."""
    s = pd.DataFrame({"k": tr_keys, "y": tr_y}).groupby("k")["y"].agg(["sum", "size"])
    p = (s["sum"] + k * prior) / (s["size"] + k)
    out = pd.Series(te_keys).map(p).to_numpy(dtype=float)
    return np.where(np.isfinite(out), out, prior)


def fit_gbm(Xtr, ytr, seed):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(random_state=seed, **GBM_KW)
    m.fit(Xtr, ytr)
    return m


def gbm_task(Xtr, ytr, Xte, seed):
    m = fit_gbm(Xtr, ytr, seed)
    return m.predict_proba(Xte)[:, 1], m


def run_possession_blocks(d: pd.DataFrame, n_jobs: int) -> dict:
    from joblib import Parallel, delayed
    fits = {}
    tasks, keys = [], []
    specs = {  # arm -> (target, train features, test features)
        "A2": ("y_nt", STATE_T, STATE_T),
        "A2lab": ("y_nt", STATE_L, STATE_T),       # labelled fit, true query
        "A2_D9a": ("y_nt", STATE_T + ["def_foul_oadj", "off_drawn_oadj"],
                   STATE_T + ["def_foul_oadj", "off_drawn_oadj"]),
        "A2_D9b": ("y_nt", STATE_T + ["is_conf_game"], STATE_T + ["is_conf_game"]),
        "D2": ("y_dnt", STATE_T, STATE_T),
        "O2": ("y_off", STATE_T + ["ended_tov"], STATE_T + ["ended_tov"]),
    }
    folds = {}
    for fold, sp in FOLDS.items():
        assert_not_sealed(sp["train"], context=f"{fold} train")
        assert_not_sealed(sp["test"], context=f"{fold} test")
        tr = d[d["season"].isin(sp["train"]) & (d["in_fit_window"] == 1)]
        te = d[d["season"].isin(sp["test"])]
        folds[fold] = (tr, te)
        for arm, (y, ftr, fte) in specs.items():
            if "is_conf_game" in ftr and d["is_conf_game"].isna().any():
                continue
            for s in SEEDS:
                tasks.append(delayed(gbm_task)(tr[ftr].to_numpy("float32"), tr[y].to_numpy(),
                                               te[fte].to_numpy("float32"), s))
                keys.append((fold, arm, s))
    print(f"  possession blocks: {len(tasks)} GBM fits on {n_jobs} workers", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(tasks)
    print(f"  GBM fits done in {time.time() - t0:.0f}s", flush=True)
    preds = {}
    for (fold, arm, s), (p, m) in zip(keys, res):
        preds[(fold, arm, s)] = p
        if fold == "F2" and s == 0 and arm in ("A2", "D2", "O2"):
            fits[arm] = m
    for fold, (tr, te) in folds.items():
        for s in SEEDS:
            out = te[["game_id", "season", "period", "poss_index", "offense_team_id",
                      "defense_team_id", "in_fit_window", "game_minute", "site_home",
                      "site_away", "is_conf_game", "game_date", "def_f_t", "off_f_t",
                      "y_nt", "y_dnt", "y_off"]].copy()
            out["half"] = np.where(te["period"] <= 1, 1, np.where(te["period"] == 2, 2, 3))
            for y in ("y_nt", "y_dnt", "y_off"):
                prior = float(tr[y].mean())
                out[f"{y}__C0"] = prior
            out["y_nt__A0"] = out.pop("y_nt__C0")
            out["y_dnt__D0"] = out.pop("y_dnt__C0")
            out["y_off__O0"] = out.pop("y_off__C0")
            out["y_nt__A_served"] = 0.123346
            key_tr = (np.where(tr["period"] <= 1, 0, 1) * 100 + tr["dcell"]).to_numpy()
            key_te = (np.where(te["period"] <= 1, 0, 1) * 100 + te["dcell"]).to_numpy()
            out["y_nt__A1"] = _cells(key_tr, tr["y_nt"].to_numpy(), key_te, float(tr["y_nt"].mean()))
            for arm, (y, _, _) in specs.items():
                if (fold, arm, s) in preds:
                    out[f"{y}__{arm}"] = preds[(fold, arm, s)]
            out.to_parquet(OUT / f"preds_poss_{fold}_seed{s}.parquet", index=False)
            if fold == "F2" and s == 0:
                fits["A1_table"] = {
                    "keys": sorted(set(key_tr.tolist())),
                    "p": _cells(key_tr, tr["y_nt"].to_numpy(),
                                np.array(sorted(set(key_tr.tolist()))), float(tr["y_nt"].mean())).tolist(),
                    "prior": float(tr["y_nt"].mean())}
    return fits


# ---------------------------------------------------------------------------
# chance design (block T)
# ---------------------------------------------------------------------------
def chance_design() -> pd.DataFrame:
    from train_foul_bonus_cond_v1 import anti_join_technicals
    from cbb_sim.models import possession_outcome as PO
    ch = pd.read_parquet(ROOT / "data/processed/models/possession_outcome/round2/design.parquet")
    ch, rep = anti_join_technicals(ch)
    print(f"  technical anti-join: dropped {rep.get('n_chances_dropped')}", flush=True)
    acc = pd.read_parquet(R6 / "foul_accrual_poss_v2.parquet",
                          columns=["game_id", "season", "poss_index", "def_team_fouls",
                                   "off_team_fouls", "def_team_fouls_true", "off_team_fouls_true"])
    ch = ch.merge(acc, on=["game_id", "season", "poss_index"], how="inner")
    ch = ch.sort_values(["season", "game_id", "poss_index", "chance_number"]).reset_index(drop=True)
    trip = (ch["terminal_event"].isin(["FT_trip_shooting", "FT_trip_bonus"]).astype(int)
            + ch["and_one"].astype(int))
    prev_trips = trip.groupby([ch["season"], ch["game_id"], ch["poss_index"]]).cumsum() - trip
    # live count at the chance: open count + the defence's trip fouls on earlier chances
    ch["def_f_t"] = (ch["def_team_fouls_true"] + prev_trips).astype("float32")
    ch["def_f_l"] = (ch["def_team_fouls"] + prev_trips).astype("float32")
    ch["off_f_t"] = ch["off_team_fouls_true"].astype("float32")
    ch["off_f_l"] = ch["off_team_fouls"].astype("float32")
    ch["inb_t"] = (ch["def_f_t"] >= BONUS).astype("float32")
    ch["inb_l"] = ch["in_bonus"].astype("float32")
    ch["is_cont"] = (ch["population"] == "cont").astype("float32")
    ch["y_bonus"] = (ch["terminal_event"] == "FT_trip_bonus").astype("int8")
    ch["y_shoot"] = (ch["terminal_event"] == "FT_trip_shooting").astype("int8")
    ch["in_fit_window"] = ((ch["period"] <= 2) & (ch["start_clock"] > 120)).astype("int8")
    ch["half"] = np.where(ch["period"] <= 1, 0, 1)
    base = [c for c in PO.feature_set("C_plus_state", "cont") + ["is_cont"] if c in ch.columns]
    return ch, base


def diff_bucket(dt, ot):
    x = dt - ot
    return np.where(x <= -2, 0, np.where(x >= 2, 2, 1))


def t_cells(ch, which: str, arm: str) -> np.ndarray:
    dcol, ocol, bcol = (("def_f_t", "off_f_t", "inb_t") if which == "t"
                        else ("def_f_l", "off_f_l", "inb_l"))
    h = ch["half"].to_numpy()
    if arm == "T1c":
        return h * 10 + ch[bcol].to_numpy().astype(int)
    dc = np.clip(ch[dcol].to_numpy(), 0, MAXF).astype(int)
    db = diff_bucket(ch[dcol].to_numpy(), ch[ocol].to_numpy())
    return h * 1000 + dc * 10 + db


def fit_offsets(keys: np.ndarray, y: np.ndarray, off: np.ndarray, lam: float = 20.0,
                iters: int = 200) -> dict:
    """Independent per-cell logit offsets delta_k solving sum(y - s(o + d)) = lam*d
    (one-hot cells, disjoint, so the cells decouple; the L2 pseudo-count lam=20 is
    fixed before the run and not tuned)."""
    df = pd.DataFrame({"k": keys, "y": y, "o": off})
    out = {}
    for k, g in df.groupby("k"):
        yy, oo = g["y"].to_numpy(float), g["o"].to_numpy(float)
        dlt = 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(oo + dlt, -35, 35)))
            grad = (yy - p).sum() - lam * dlt
            hess = (p * (1 - p)).sum() + lam
            step = float(np.clip(grad / hess, -1.0, 1.0))   # damped Newton (solver guard)
            dlt += step
            if abs(step) < 1e-9:
                break
        out[int(k)] = float(dlt)
    return out


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def t0_task(Xtr, ytr, Xq_list):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=1.0, max_iter=300)
    m.fit(sc.transform(Xtr), ytr)
    return [m.predict_proba(sc.transform(X))[:, 1] for X in Xq_list], (sc, m)


def run_trip_block(n_jobs: int) -> dict:
    from joblib import Parallel, delayed
    ch, base = chance_design()
    fits = {"T_base_features": base}
    # base features with the labelled vs the true in_bonus
    feats_l = base
    tasks, keys, parts = [], [], {}
    for fold, sp in FOLDS.items():
        assert_not_sealed(sp["train"], context=f"{fold} train")
        assert_not_sealed(sp["test"], context=f"{fold} test")
        tr = ch[ch["season"].isin(sp["train"]) & (ch["in_fit_window"] == 1)]
        te = ch[ch["season"].isin(sp["test"])]
        parts[fold] = (tr, te)
        Xtr_l = tr[feats_l].to_numpy("float64")
        Xtr_t = tr[feats_l].assign(in_bonus=tr["inb_t"])[feats_l].to_numpy("float64")
        Xte_t = te[feats_l].assign(in_bonus=te["inb_t"])[feats_l].to_numpy("float64")
        for y in ("y_bonus", "y_shoot"):
            tasks.append(delayed(t0_task)(Xtr_l, tr[y].to_numpy(), [Xtr_l, Xtr_t, Xte_t]))
            keys.append((fold, y))
    print(f"  trip block: {len(tasks)} T0 logistic fits", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=min(n_jobs, len(tasks)), backend="loky")(tasks)
    print(f"  T0 fits done in {time.time() - t0:.0f}s", flush=True)
    P = {k: r for k, r in zip(keys, res)}
    for fold, (tr, te) in parts.items():
        out = te[["game_id", "season", "period", "poss_index", "chance_number",
                  "offense_team_id", "defense_team_id", "in_fit_window", "start_clock",
                  "site_home", "site_away", "game_date", "def_f_t", "off_f_t", "inb_t",
                  "is_cont", "y_bonus", "y_shoot"]].copy()
        out["half"] = np.where(te["period"] <= 1, 1, np.where(te["period"] == 2, 2, 3))
        for y in ("y_bonus", "y_shoot"):
            (p_tr_l, p_tr_t, p_te_t), model = P[(fold, y)]
            out[f"{y}__T0"] = p_te_t
            ytr = tr[y].to_numpy()
            for arm, which, base_tr in (("T1c", "t", p_tr_t), ("T2c", "t", p_tr_t),
                                        ("T2lab", "l", p_tr_l)):
                k_tr = t_cells(tr, which, arm if arm != "T2lab" else "T2c")
                k_te = t_cells(te, "t", arm if arm != "T2lab" else "T2c")
                dl = fit_offsets(k_tr, ytr, logit(base_tr))
                d_te = pd.Series(k_te).map(dl).fillna(0.0).to_numpy()
                out[f"{y}__{arm}"] = 1.0 / (1.0 + np.exp(-(logit(p_te_t) + d_te)))
                if fold == "F2":
                    fits[f"delta_{arm}_{y}"] = dl
        for s in SEEDS:      # every T arm is deterministic: both seed files are identical
            out.to_parquet(OUT / f"preds_trip_{fold}_seed{s}.parquet", index=False)
    return fits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--blocks", default="poss,trip")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fits = {}
    if "poss" in a.blocks:
        d = possession_design()
        print(f"possession design {len(d):,} rows; means y_nt {d['y_nt'].mean():.5f} "
              f"y_dnt {d['y_dnt'].mean():.5f} y_off {d['y_off'].mean():.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        fits.update(run_possession_blocks(d, a.n_jobs))
    if "trip" in a.blocks:
        fits.update(run_trip_block(a.n_jobs))
    import joblib
    joblib.dump(fits, OUT / f"fits_F2_{a.blocks.replace(',', '_')}.joblib")
    meta = {"pre_registration": "experiments.md section 20 (commit aa0ee91)",
            "seconds": round(time.time() - t0, 1), "blocks": a.blocks, "seeds": SEEDS,
            "gbm": GBM_KW}
    (OUT / f"train_meta_{a.blocks.replace(',', '_')}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
