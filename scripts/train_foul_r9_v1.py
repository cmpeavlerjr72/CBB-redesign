#!/usr/bin/env python
"""
train_foul_r9_v1.py -- foul round 9 (possession-outcome experiments.md section 26, lane C,
2026-09-30), OFFLINE arms. Written before the pre-registration commit; run only after it.

    .venv/Scripts/python.exe scripts/train_foul_r9_v1.py --n-jobs 3

Block AO -- the and-one rate given a made field goal (the served engine draws it as a
per-shot-class constant `rules.and_one_rate_given_made`, never bake-offed). Rows: every
made-FG chance of `possessions_v2` chances, seasons 2022-2025, with the CORRECTED
(engine-definition) foul state of `round6/foul_accrual_poss_v2.parquet`; the live count at
a chance adds the defence's trip fouls on earlier chances of the possession (round 7 rule).
  AO0  per-class constant (the served definition, refit on these rows)
  AO1  cells: class x period index (H1, H2, OT) x clock bucket (sec_rem cuts 120/300/600/900),
       Laplace k=200 toward the class rate
  AO2  GBM (round-7 GBM_KW, n_jobs=1) on class one-hot + round-7 STATE_T (true state)
Block T -- FT-trip classes as logit offsets on the round-7 proxy T0 (exactly round 7's
machinery, imported):
  T2c  (reference = the R8b trip term) half x live count x differential
  T3t  T2c cells x clock bucket (same cuts)
  T3s  half x clock bucket x rule state (pre-bonus / one-and-one / double) x differential
Seeds 0 and 7 for the stochastic arm (AO2); cells arms are deterministic (both seed files
identical). Every arm is scored on TEST rows fed the TRUE state.

Writes data/processed/models/possession_outcome/round9/{preds_ao,preds_trip}_{fold}_seed{s}.parquet
and fits_F2.joblib (fold-2 TRAIN fits only, for the engine LUTs).
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
import train_foul_joint_v1 as R7  # noqa: E402

OUT = ROOT / "data/processed/models/possession_outcome/round9"
R6 = ROOT / "data/processed/models/possession_outcome/round6"
FOLDS = R7.FOLDS
SEASONS = [2022, 2023, 2024, 2025]
SEEDS = (0, 7)
CLOCK_CUTS = np.array([120, 300, 600, 900], dtype=np.float64)
CLASSES = ["rim", "jump2", "3"]
MAXF = 10


def pidx(period):
    period = np.asarray(period)
    return np.where(period <= 1, 0, np.where(period == 2, 1, 2))


def cidx(sec):
    return np.searchsorted(CLOCK_CUTS, np.asarray(sec, dtype=np.float64), side="right")


# ---------------------------------------------------------------------------
# block AO
# ---------------------------------------------------------------------------
def ao_design() -> pd.DataFrame:
    acc = pd.read_parquet(R6 / "foul_accrual_poss_v2.parquet",
                          columns=["game_id", "season", "period", "poss_index", "def_team_fouls_true",
                                   "off_team_fouls_true", "neutral_site", "offense_is_home",
                                   "offense_team_id", "defense_team_id", "game_date"])
    parts = []
    for s in SEASONS:
        c = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{s}.parquet")
        c = c.sort_values(["game_id", "period", "poss_index", "chance_number"])
        trip = (c["terminal_event"].isin(["FT_trip_shooting", "FT_trip_bonus"]).astype(int)
                + c["and_one"].astype(int))
        c["prev_trips"] = trip.groupby([c["game_id"], c["period"], c["poss_index"]]).cumsum() - trip
        made = (c["fgm_rim"] + c["fgm_jump2"] + c["fgm_3"]) > 0
        c = c[made].copy()
        c["season"] = s
        parts.append(c)
    c = pd.concat(parts, ignore_index=True)
    c = c.merge(acc, on=["game_id", "season", "period", "poss_index"], how="inner",
                suffixes=("", "_acc"))
    d = pd.DataFrame({
        "game_id": c["game_id"], "season": c["season"], "period": c["period"].astype("float32"),
        "poss_index": c["poss_index"], "chance_number": c["chance_number"],
        "offense_team_id": c["offense_team_id"], "defense_team_id": c["defense_team_id"],
        "sec_rem": c["start_clock"].astype("float32"),
        "margin": c["start_score_diff"].astype("float32"),
    })
    d["cls"] = np.where(c["fgm_rim"] > 0, 0, np.where(c["fgm_jump2"] > 0, 1, 2))
    d["abs_margin"] = d["margin"].abs()
    d["game_seconds"] = np.where(d["period"] <= 2, (d["period"] - 1) * 1200 + (1200 - d["sec_rem"]),
                                 2400 + (d["period"] - 3) * 300 + (300 - d["sec_rem"])).astype("float32")
    d["def_f_t"] = (c["def_team_fouls_true"] + c["prev_trips"]).astype("float32").to_numpy()
    d["off_f_t"] = c["off_team_fouls_true"].astype("float32").to_numpy()
    d["offb_t"] = (d["def_f_t"] >= R7.BONUS).astype("float32")
    d["defb_t"] = (d["off_f_t"] >= R7.BONUS).astype("float32")
    d["offdb_t"] = (d["def_f_t"] >= R7.DBONUS).astype("float32")
    neu = c["neutral_site"].astype(bool).to_numpy()
    home = c["offense_is_home"].astype(bool).to_numpy()
    d["site_home"] = ((~neu) & home).astype("float32")
    d["site_away"] = ((~neu) & ~home).astype("float32")
    d["is_ot"] = (d["period"] >= 3).astype("float32")
    for k, nm in enumerate(CLASSES):
        d[f"cls_{nm}"] = (d["cls"] == k).astype("float32")
    d["y_ao"] = c["and_one"].astype("int8").to_numpy()
    d["half"] = pidx(d["period"]) + 1
    return d


AO_FEATS = ["cls_rim", "cls_jump2", "cls_3"] + R7.STATE_T


def team_prior_table(d: pd.DataFrame, k: float = 200.0) -> pd.DataFrame:
    """PRIOR-season team and-one rates, centred on that season's own league mean
    (CLAUDE.md modeling rule): offence = and-ones drawn per made FG, defence = and-ones
    conceded per made FG allowed, each shrunk toward the league mean with k=200 made FGs.
    Row (season s, team) holds season s-1's rates; teams without a prior season get 0."""
    rows = []
    for s in sorted(d["season"].unique()):
        x = d[d["season"] == s]
        lg = float(x["y_ao"].mean())
        for side, col in (("off", "offense_team_id"), ("def", "defense_team_id")):
            g = x.groupby(col)["y_ao"].agg(["sum", "size"])
            sh = (g["sum"] + k * lg) / (g["size"] + k) - lg
            rows.append(pd.DataFrame({"season": s + 1, "team_id": g.index, "side": side, "c": sh.values}))
    t = pd.concat(rows).pivot_table(index=["season", "team_id"], columns="side", values="c").reset_index()
    return t.rename(columns={"off": "ao_off_prior_c", "def": "ao_def_prior_c"}).fillna(0.0)


def add_team_prior(d: pd.DataFrame, tab: pd.DataFrame) -> pd.DataFrame:
    o = tab[["season", "team_id", "ao_off_prior_c"]].rename(columns={"team_id": "offense_team_id"})
    q = tab[["season", "team_id", "ao_def_prior_c"]].rename(columns={"team_id": "defense_team_id"})
    d = d.merge(o, on=["season", "offense_team_id"], how="left").merge(q, on=["season", "defense_team_id"],
                                                                        how="left")
    d[["ao_off_prior_c", "ao_def_prior_c"]] = d[["ao_off_prior_c", "ao_def_prior_c"]].fillna(0.0)
    return d


def fit_team_terms(base_p: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """logit p = logit(base) + X b, b by Newton (no intercept: the base carries the level)."""
    o = R7.logit(base_p)
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(o + X @ b)))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def ao_cells_key(d):
    return d["cls"].to_numpy() * 100 + pidx(d["period"]) * 10 + cidx(d["sec_rem"])


def run_ao(n_jobs: int) -> dict:
    from joblib import Parallel, delayed
    d = ao_design()
    print(f"AO design {len(d):,} made-FG chances; and-one rate {d['y_ao'].mean():.5f}", flush=True)
    tp = team_prior_table(d)
    tp.to_parquet(OUT / "ao_team_prior_v1.parquet", index=False)   # season s holds s-1 rates
    d = add_team_prior(d, tp)
    fits, tasks, keys, parts = {}, [], [], {}
    for fold, sp in FOLDS.items():
        assert_not_sealed(sp["train"], context=f"{fold} train")
        assert_not_sealed(sp["test"], context=f"{fold} test")
        tr = d[d["season"].isin(sp["train"])]
        te = d[d["season"].isin(sp["test"])]
        parts[fold] = (tr, te)
        for s in SEEDS:
            tasks.append(delayed(R7.gbm_task)(tr[AO_FEATS].to_numpy("float32"), tr["y_ao"].to_numpy(),
                                              te[AO_FEATS].to_numpy("float32"), s))
            keys.append((fold, s))
    t0 = time.time()
    res = Parallel(n_jobs=min(n_jobs, len(tasks)), backend="loky")(tasks)
    print(f"  AO GBM fits {time.time() - t0:.0f}s", flush=True)
    P = dict(zip(keys, res))
    for fold, (tr, te) in parts.items():
        cls_rate = tr.groupby("cls")["y_ao"].mean()
        base = te["cls"].map(cls_rate).to_numpy()
        ktr, kte = ao_cells_key(tr), ao_cells_key(te)
        # AO1: per-cell Laplace toward the CLASS rate
        s_ = pd.DataFrame({"k": ktr, "y": tr["y_ao"].to_numpy(), "c": tr["cls"].to_numpy()})
        agg = s_.groupby("k").agg(sm=("y", "sum"), n=("y", "size"), c=("c", "first"))
        pr = agg["c"].map(cls_rate)
        tab = (agg["sm"] + 200 * pr) / (agg["n"] + 200)
        ao1 = pd.Series(kte).map(tab).to_numpy(dtype=float)
        ao1 = np.where(np.isfinite(ao1), ao1, base)
        # AO3: AO1 + prior-season team terms (offence drawn, defence conceded)
        ao1_tr = pd.Series(ktr).map(tab).to_numpy(dtype=float)
        tc = ["ao_off_prior_c", "ao_def_prior_c"]
        b = fit_team_terms(ao1_tr, tr[tc].to_numpy(float), tr["y_ao"].to_numpy(float))
        ao3 = 1.0 / (1.0 + np.exp(-(R7.logit(ao1) + te[tc].to_numpy(float) @ b)))
        print(f"  {fold} AO3 team terms b = {b.round(3).tolist()}", flush=True)
        for s in SEEDS:
            out = te[["game_id", "season", "period", "poss_index", "chance_number", "offense_team_id",
                      "defense_team_id", "sec_rem", "cls", "half", "def_f_t", "site_home", "site_away",
                      "y_ao"]].copy()
            out["y_ao__AO0"] = base
            out["y_ao__AO1"] = ao1
            out["y_ao__AO2"] = P[(fold, s)][0]
            out["y_ao__AO3"] = ao3
            out["ao_off_prior_c"] = te["ao_off_prior_c"].to_numpy()
            out["ao_def_prior_c"] = te["ao_def_prior_c"].to_numpy()
            out.to_parquet(OUT / f"preds_ao_{fold}_seed{s}.parquet", index=False)
        if fold == "F2":
            fits["AO0"] = cls_rate.to_dict()
            fits["AO1"] = tab.to_dict()
            fits["AO2"] = P[("F2", 0)][1]
            fits["AO3_b"] = b.tolist()
    return fits


# ---------------------------------------------------------------------------
# block T
# ---------------------------------------------------------------------------
def t_keys(ch: pd.DataFrame, arm: str) -> np.ndarray:
    h = ch["half"].to_numpy()
    dt = ch["def_f_t"].to_numpy()
    ot = ch["off_f_t"].to_numpy()
    db = R7.diff_bucket(dt, ot)
    ci = cidx(ch["start_clock"].to_numpy())
    if arm == "T2c":
        return R7.t_cells(ch, "t", "T2c")
    dc = np.clip(dt, 0, MAXF).astype(int)
    if arm == "T3t":
        return h * 100000 + ci * 10000 + dc * 10 + db
    if arm == "T3s":
        stt = np.where(dt >= R7.DBONUS, 2, np.where(dt >= R7.BONUS, 1, 0))
        return h * 1000 + ci * 100 + stt * 10 + db
    raise KeyError(arm)


def run_trip(n_jobs: int) -> dict:
    from joblib import Parallel, delayed
    ch, base = R7.chance_design()
    fits = {"T_base_features": base}
    tasks, keys, parts = [], [], {}
    for fold, sp in FOLDS.items():
        assert_not_sealed(sp["train"], context=f"{fold} train")
        assert_not_sealed(sp["test"], context=f"{fold} test")
        tr = ch[ch["season"].isin(sp["train"]) & (ch["in_fit_window"] == 1)]
        te = ch[ch["season"].isin(sp["test"])]
        parts[fold] = (tr, te)
        Xtr_l = tr[base].to_numpy("float64")
        Xtr_t = tr[base].assign(in_bonus=tr["inb_t"])[base].to_numpy("float64")
        Xte_t = te[base].assign(in_bonus=te["inb_t"])[base].to_numpy("float64")
        for y in ("y_bonus", "y_shoot"):
            tasks.append(delayed(R7.t0_task)(Xtr_l, tr[y].to_numpy(), [Xtr_t, Xte_t]))
            keys.append((fold, y))
    t0 = time.time()
    res = Parallel(n_jobs=min(n_jobs, len(tasks)), backend="loky")(tasks)
    print(f"  T0 fits {time.time() - t0:.0f}s", flush=True)
    P = dict(zip(keys, res))
    for fold, (tr, te) in parts.items():
        out = te[["game_id", "season", "period", "poss_index", "chance_number", "offense_team_id",
                  "defense_team_id", "in_fit_window", "start_clock", "site_home", "site_away",
                  "def_f_t", "off_f_t", "inb_t", "is_cont", "y_bonus", "y_shoot"]].copy()
        out["half"] = np.where(te["period"] <= 1, 1, np.where(te["period"] == 2, 2, 3))
        for y in ("y_bonus", "y_shoot"):
            (p_tr_t, p_te_t), _ = P[(fold, y)]
            out[f"{y}__T0"] = p_te_t
            for arm in ("T2c", "T3t", "T3s"):
                dl = R7.fit_offsets(t_keys(tr, arm), tr[y].to_numpy(), R7.logit(p_tr_t))
                d_te = pd.Series(t_keys(te, arm)).map(dl).fillna(0.0).to_numpy()
                out[f"{y}__{arm}"] = 1.0 / (1.0 + np.exp(-(R7.logit(p_te_t) + d_te)))
                if fold == "F2":
                    fits[f"delta_{arm}_{y}"] = dl
        for s in SEEDS:
            out.to_parquet(OUT / f"preds_trip_{fold}_seed{s}.parquet", index=False)
    return fits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--blocks", default="ao,trip")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fits = {}
    if "ao" in a.blocks:
        fits.update(run_ao(a.n_jobs))
    if "trip" in a.blocks:
        fits.update(run_trip(a.n_jobs))
    import joblib
    joblib.dump(fits, OUT / f"fits_F2_{a.blocks.replace(',', '_')}.joblib")
    meta = {"pre_registration": "possession_outcome experiments.md section 26",
            "seconds": round(time.time() - t0, 1), "blocks": a.blocks, "seeds": SEEDS, "gbm": R7.GBM_KW}
    (OUT / f"train_meta_{a.blocks.replace(',', '_')}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
