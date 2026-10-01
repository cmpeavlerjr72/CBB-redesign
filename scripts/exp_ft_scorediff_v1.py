#!/usr/bin/env python
"""exp_ft_scorediff_v1.py -- free_throw round 14: the score-margin feature as a within-game response (lane B, 2026-10-01).

Spec: docs/models/free_throw/experiments.md section 14 (committed BEFORE the `offline` stage ran).

    .venv/Scripts/python.exe scripts/exp_ft_scorediff_v1.py offline           # arms x folds x seeds, static fits
    .venv/Scripts/python.exe scripts/exp_ft_scorediff_v1.py build <ARM>       # S1_conf_aligned F2 serving artifacts

Arms: FT0 = served FT_FEATURES; FTn = FT_FEATURES minus score_diff; FTnE = FTn + gt_flag, eg_trail, eg_lead
(fg_make's own end-game / garbage-time indicators, the same arithmetic `loop._state_block` serves).
No trainer is edited: the served trainer `train_free_throw_v2_s1.py` is imported and its module feature
constant is swapped in this process only; artifacts go under data/processed/models/free_throw/s1_scorediff/.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_v] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import train_free_throw_v2_s1 as T  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402

FT = T.FT
BASE = tuple(FT.FT_FEATURES)
ARMS = {
    "FT0": BASE,
    "FTn": tuple(f for f in BASE if f != "score_diff"),
    "FTnE": tuple(f for f in BASE if f != "score_diff") + ("gt_flag", "eg_trail", "eg_lead"),
}
TEAM_BLOCK = ("off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c", "site_home", "site_away")
ARMS["FTp"] = ARMS["FTn"] + TEAM_BLOCK                                  # round 15 (section 15.2)
ARMS["FTpE"] = ARMS["FTn"] + TEAM_BLOCK + ("gt_flag", "eg_trail", "eg_lead")
SIMPLICITY.update({"FTp": 1, "FTpE": 2})
SIMPLICITY = {"FT0": 0, "FTn": 1, "FTnE": 2}
OUT = ROOT / "results/ft_scorediff"
OUT.mkdir(parents=True, exist_ok=True)
ART = ROOT / "data/processed/models/free_throw/s1_scorediff"
LL_GUARD = 0.0010
CAL_GUARD_PP = 1.0
NB = 200
_orig_dm = FT.design_matrix


def add_indicators(d: pd.DataFrame) -> pd.DataFrame:
    sd = d["score_diff"].to_numpy(dtype="float64")
    per = d["period"].to_numpy(dtype="float64")
    gsr = FG.regulation_seconds_remaining(per, d["seconds_remaining"].to_numpy(dtype="float64"))
    reg = per <= 2.0
    d = d.copy()
    d["gt_flag"] = (reg & (np.abs(sd) >= FG.R2_GT_MARGIN) & (gsr <= FG.R2_GT_SECONDS)).astype("float32")
    d["eg_trail"] = (reg & (sd <= -FG.R2_EG_LO) & (sd >= -FG.R2_EG_HI) & (gsr <= FG.R2_EG_SECONDS)).astype("float32")
    d["eg_lead"] = (reg & (sd >= FG.R2_EG_LO) & (sd <= FG.R2_EG_HI) & (gsr <= FG.R2_EG_SECONDS)).astype("float32")
    return d


def use_arm(arm: str) -> None:
    feats = ARMS[arm]
    FT.FT_FEATURES = feats
    FT.design_matrix = lambda d, features=None: _orig_dm(d, tuple(features) if features is not None else feats)


def design(team_block: bool = False) -> pd.DataFrame:
    att = pd.read_parquet(T.OUT_DIR / "attempts_v1_era.parquet")
    att = att[att["season"].isin(T.SEASONS)]
    d = add_indicators(FT.build_ft_design(att))
    if team_block:
        tb = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet",
                             columns=["game_id", "off_team_id", *TEAM_BLOCK])
        tb = tb.drop_duplicates(["game_id", "off_team_id"]).rename(columns={"off_team_id": "team_id"})
        n0 = len(d)
        d = d.merge(tb, on=["game_id", "team_id"], how="inner")
        print(f"team block joined: {len(d)} of {n0} attempts kept", flush=True)
    return d


def within_slope(te: pd.DataFrame, r: np.ndarray, w: np.ndarray | None = None) -> float:
    key = te["game_id"].astype(str).to_numpy() + "_" + te["team_id"].astype(str).to_numpy()
    d = te["score_diff"].to_numpy(dtype="float64")
    dc = d - pd.Series(d).groupby(key).transform("mean").to_numpy()
    if w is None:
        return float((dc * r).sum() / (dc * dc).sum())
    return float((w * dc * r).sum() / (w * dc * dc).sum())


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def offline() -> dict:
    t0 = time.time()
    d = design()
    rep = {"created_at": pd.Timestamp.now("UTC").isoformat(), "folds": {}}
    for fold in ("F1", "F2"):
        tr, te = FT.fold_slices(d, fold)
        te = te.reset_index(drop=True)
        y = te["y"].to_numpy().astype(float)
        preds = {}
        for arm in ARMS:
            use_arm(arm)
            for seed in (0, 1):
                m = FT.LgbmArm(seed=seed).fit(FT.design_matrix(tr), tr["y"].to_numpy())
                pp = m.predict_proba(FT.design_matrix(te))
                preds[(arm, seed)] = pp[:, -1] if pp.ndim == 2 else pp
                print(f"[{time.time()-t0:6.0f}s] {fold} {arm} seed{seed} ll {logloss(y, preds[(arm, seed)]):.5f}", flush=True)
        # game bootstrap weights
        gid = te["game_id"].to_numpy()
        ug, inv = np.unique(gid, return_inverse=True)
        rng = np.random.default_rng(14)
        Wg = rng.poisson(1.0, size=(NB, len(ug))).astype(float)
        res = {}
        s_ref = abs(within_slope(te, y - preds[("FT0", 0)]))
        s_ref1 = abs(within_slope(te, y - preds[("FT0", 1)]))
        bs_ref = np.array([abs(within_slope(te, y - preds[("FT0", 0)], Wg[b][inv])) for b in range(NB)])
        for arm in ARMS:
            p0, p1 = preds[(arm, 0)], preds[(arm, 1)]
            s = abs(within_slope(te, y - p0))
            s1 = abs(within_slope(te, y - p1))
            gain = s_ref - s
            bs = np.array([abs(within_slope(te, y - p0, Wg[b][inv])) for b in range(NB)])
            floor = max(float(np.std(bs_ref - bs)), abs(gain - (s_ref1 - s1)))
            # guards
            ll, ll_ref = logloss(y, p0), logloss(y, preds[("FT0", 0)])
            bins = pd.cut(te["score_diff"], [-200, -15, -8, -4, -1, 0, 3, 7, 14, 200])
            cal = pd.DataFrame({"b": bins, "g": y - p0}).groupby("b", observed=True)["g"].mean() * 100
            resp = FT.responsiveness(te, np.column_stack([1 - p0, p0]))
            resp_ok, resp_worst = FT.PM.responsiveness_verdict(resp, FT.RESPONSIVENESS_MIN_STEPS)
            res[arm] = {"within_slope_signed": within_slope(te, y - p0), "abs_slope": s,
                        "gain_vs_FT0": gain, "floor": floor, "floors": gain / floor if floor > 0 else 0.0,
                        "seed1_abs_slope": s1, "logloss": ll, "ll_worse_vs_FT0": ll - ll_ref,
                        "margin_bin_cal_pp": {str(k): float(v) for k, v in cal.round(3).items()}, "max_bin_cal_pp": float(cal.abs().max()),
                        "responsiveness_pass": bool(resp_ok), "responsiveness_worst": resp_worst}
        # real within-game slope of y on margin (the target the model's within response should carry)
        res["_real_within_slope_of_y"] = within_slope(te, y - y.mean())
        rep["folds"][fold] = {"n_test": int(len(te)), "arms": res}
    f1, f2 = rep["folds"]["F1"]["arms"], rep["folds"]["F2"]["arms"]
    elig = []
    for a in ("FTn", "FTnE"):
        ok = (f2[a]["gain_vs_FT0"] > 2 * f2[a]["floor"] and f1[a]["gain_vs_FT0"] > 0
              and f2[a]["ll_worse_vs_FT0"] <= LL_GUARD and f2[a]["max_bin_cal_pp"] <= CAL_GUARD_PP
              and f2[a]["responsiveness_pass"])
        if ok:
            elig.append(a)
    if not elig:
        winner = "FT0"
    else:
        best = max(elig, key=lambda a: f2[a]["gain_vs_FT0"])
        tied = [a for a in elig if f2[best]["gain_vs_FT0"] - f2[a]["gain_vs_FT0"] <= f2[best]["floor"]]
        winner = min(tied, key=lambda a: SIMPLICITY[a])
    rep["eligible"], rep["winner"] = elig, winner
    return rep


def offline15() -> dict:
    """Section 15.2: log loss primary among no-in-game-response arms; guards vs FT0."""
    t0 = time.time()
    d = design(team_block=True)
    rep = {"created_at": pd.Timestamp.now("UTC").isoformat(), "n_design": int(len(d)), "folds": {}}
    for fold in ("F1", "F2"):
        tr, te = FT.fold_slices(d, fold)
        te = te.reset_index(drop=True)
        y = te["y"].to_numpy().astype(float)
        gid = te["game_id"].to_numpy()
        ug, inv = np.unique(gid, return_inverse=True)
        Wg = np.random.default_rng(15).poisson(1.0, size=(NB, len(ug))).astype(float)[:, inv]
        preds = {}
        for arm in ("FT0", "FTp", "FTpE"):
            use_arm(arm)
            for seed in (0, 1):
                m = FT.LgbmArm(seed=seed).fit(FT.design_matrix(tr), tr["y"].to_numpy())
                preds[(arm, seed)] = m.predict_proba(FT.design_matrix(te))[:, -1]
                print(f"[{time.time()-t0:6.0f}s] {fold} {arm} seed{seed} ll {logloss(y, preds[(arm, seed)]):.5f}", flush=True)
        def rowll(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p))
        ref0, ref1 = rowll(preds[("FT0", 0)]), rowll(preds[("FT0", 1)])
        res = {}
        for arm in ("FT0", "FTp", "FTpE"):
            l0, l1 = rowll(preds[(arm, 0)]), rowll(preds[(arm, 1)])
            dif = l0 - ref0
            bs = (Wg * dif[None]).sum(1) / Wg.sum(1)
            floor = max(float(bs.std()), abs(dif.mean() - (l1 - ref1).mean()))
            p0 = preds[(arm, 0)]
            bins = pd.cut(te["score_diff"], [-200, -15, -8, -4, -1, 0, 3, 7, 14, 200])
            cal = pd.DataFrame({"b": bins, "g": y - p0}).groupby("b", observed=True)["g"].mean() * 100
            resp = FT.responsiveness(te, np.column_stack([1 - p0, p0]))
            ok, worst = FT.PM.responsiveness_verdict(resp, FT.RESPONSIVENESS_MIN_STEPS)
            res[arm] = {"logloss": float(l0.mean()), "ll_vs_FT0": float(dif.mean()), "floor": floor,
                        "floors": float(dif.mean() / floor) if floor > 0 else 0.0,
                        "seed1_logloss": float(l1.mean()),
                        "margin_bin_cal_pp": {str(k): float(v) for k, v in cal.round(3).items()},
                        "max_bin_cal_pp": float(cal.abs().max()), "responsiveness_pass": bool(ok),
                        "within_slope_signed": within_slope(te, y - p0)}
        rep["folds"][fold] = {"n_test": int(len(te)), "arms": res}
    f1, f2 = rep["folds"]["F1"]["arms"], rep["folds"]["F2"]["arms"]
    elig = [a for a in ("FTp", "FTpE") if f2[a]["ll_vs_FT0"] <= LL_GUARD and f1[a]["ll_vs_FT0"] <= LL_GUARD
            and f2[a]["max_bin_cal_pp"] <= CAL_GUARD_PP and f2[a]["responsiveness_pass"]]
    if not elig:
        winner = "FT0"
    else:
        best = min(elig, key=lambda a: f2[a]["logloss"])
        tied = [a for a in elig if f2[a]["logloss"] - f2[best]["logloss"] <= f2[best]["floor"]]
        winner = min(tied, key=lambda a: SIMPLICITY[a])
    rep["eligible"], rep["winner"] = elig, winner
    return rep


def _resp_pass(r) -> bool:
    try:
        return all(bool(v.get("pass", True)) for v in r.values() if isinstance(v, dict))
    except Exception:  # noqa: BLE001
        return True


def build(arm: str) -> None:
    use_arm(arm)
    T.S1_DIR = ART / arm
    T.ES.load_universe()
    d = design(team_block=arm in ("FTp", "FTpE"))
    conf_all = T.CF.build_conference_flags(T.SEASONS)
    firsts = T.CF.first_conference_game_dates(conf_all)
    tr, te = FT.fold_slices(d, "F2")
    row = T.run_cell("S1_conf_aligned", "F2", tr, te, conf_all, firsts, 2025, seed=0)
    row = {k: v for k, v in row.items() if k not in ("conf4", "clean_trip")}
    (ART / arm / "build_report.json").write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
    print(json.dumps(row, default=str)[:2000])


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "offline":
        r = offline()
        (OUT / "offline_v1.json").write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
        for f in ("F1", "F2"):
            for a, v in r["folds"][f]["arms"].items():
                if a.startswith("_"):
                    print(f, a, v)
                    continue
                print(f, a, {k: (round(x, 5) if isinstance(x, float) else x) for k, x in v.items()
                             if k not in ("margin_bin_cal_pp",)})
        print("eligible", r["eligible"], "winner", r["winner"])
    elif st == "offline15":
        r = offline15()
        (OUT / "offline15_v1.json").write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
        for f in ("F1", "F2"):
            for a, v in r["folds"][f]["arms"].items():
                print(f, a, {k: (round(x, 5) if isinstance(x, float) else x) for k, x in v.items()
                             if k != "margin_bin_cal_pp"})
                print("   cal", v["margin_bin_cal_pp"])
        print("eligible", r["eligible"], "winner", r["winner"])
    elif st == "build":
        build(sys.argv[2])
