#!/usr/bin/env python
"""
train_fg_make_v4_site.py -- Lane G (overnight 2026-09-30): fg_make home-site
repair bake-off around the SERVED round-4 B1 spec.

    .venv/Scripts/python.exe scripts/train_fg_make_v4_site.py --arms S0 --folds F2,F1 --seeds 0
    .venv/Scripts/python.exe scripts/train_fg_make_v4_site.py --arms S0,G1,G2 --folds F2,F1 --seeds 0,1

Pre-registration: `docs/models/fg_make/experiments.md` (Lane G home-site
section), committed BEFORE any non-reference arm ran. S0 is the served B1 spec
reproduced exactly (features, frozen ladder params, S1 monthly schedule); it is
run first because the step-2 diagnosis needs its held-out predictions.

Sibling of `train_fg_make_v4_shooter_block.py`; that file is READ (imported for
nothing) and never modified. Nothing is exported to an engine directory:
predictions only, under results/home_site/fg/.

Thread cap: LightGBM n_jobs is forced to CBB_NJOBS (default 2) by wrapping
`FG.LgbmArm.fit`, because the shared class hard-codes n_jobs=-1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

NJ = int(os.environ.get("CBB_NJOBS", "2"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(NJ)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

FG_DIR = Path("data/processed/models/fg_make")
DESIGN = FG_DIR / "design_v2_shotshooter.parquet"
EXTRA_CACHE = FG_DIR / "design_v4_extra_v2.parquet"
OUT = Path("results/home_site/fg")


def _fit(self, X, y):
    import lightgbm as lgb
    self.clf_ = lgb.LGBMClassifier(random_state=self.seed, n_jobs=NJ, **self.params)
    self.clf_.fit(X, y)
    return self


FG.LgbmArm.fit = _fit

B1 = list(FG.TEAM_FEATURES) + list(FG.R2_SAFE_STATE) + ["shooter_shrunk_dev_c"]
NO_SITE = [f for f in B1 if f not in ("site_home", "site_away")]

#: arm -> (feature list, description). Arms beyond S0 are defined by the
#: pre-registration; the registry is the single source of truth for the run.
ARMS: dict[str, tuple[list[str], str]] = {
    "S0": (B1, "served round-4 B1 (site one-hots, neutral = reference)"),
    # pre-registered in fg_make/experiments.md section 21 (Lane G), committed
    # BEFORE any of these ran
    "G1": (B1 + ["conf_game"], "B1 + conference-game flag (site may differ conf vs non-conf)"),
    "G2": (B1 + ["site_x_gap"], "B1 + site_signed x rating_gap"),
    "G4": (NO_SITE, "B1 without site columns + FE-identified site logit OFFSET (init_score)"),
}
OFFSET_ARMS = frozenset({"G4"})


def fe_site_offsets(rows: pd.DataFrame) -> tuple[float, float, float]:
    """Team-strength-adjusted site effects on the TRAINING rows of one refit:
    (game, offence) make rates ~ team-season offence FE + team-season defence
    FE + b_home[home] + b_away[away], weighted by attempts (neutral = the
    intercept's reference). Returns (b_home, b_away, pbar) in probability units."""
    import scipy.sparse as sp
    g = rows.groupby(["season", "game_id", "off_team_id"], sort=False).agg(
        def_team_id=("def_team_id", "first"), sh=("site_home", "first"),
        sa=("site_away", "first"), y=("y", "sum"), w=("y", "size")).reset_index()
    ot = g["season"].astype("int64") * 10_000_000 + g["off_team_id"].astype("int64")
    dt = g["season"].astype("int64") * 10_000_000 + g["def_team_id"].astype("int64")
    teams = np.unique(np.concatenate([ot, dt]))
    ti = pd.Series(np.arange(len(teams)), index=teams)
    n, T = len(g), len(teams)
    oi, di = ti[ot].to_numpy(), ti[dt].to_numpy()
    r = np.concatenate([np.arange(n)] * 5)
    c = np.concatenate([np.zeros(n, int), np.ones(n, int), np.full(n, 2), 3 + oi, 3 + T + di])
    v = np.concatenate([np.ones(n), g["sh"].to_numpy(float), g["sa"].to_numpy(float),
                        np.ones(n), np.ones(n)])
    X = sp.csr_matrix((v, (r, c)), shape=(n, 3 + 2 * T))
    w = g["w"].to_numpy(float)
    rate = g["y"].to_numpy(float) / w
    XtW = X.T @ sp.diags(w)
    A = (XtW @ X).toarray()
    A[np.arange(3, A.shape[0]), np.arange(3, A.shape[0])] += 1e-6 * float(w.mean())
    beta = np.linalg.solve(A, XtW @ rate)
    return float(beta[1]), float(beta[2]), float(rows["y"].mean())


def _offset(d: pd.DataFrame, bh: float, ba: float, pbar: float, flip: bool = False) -> np.ndarray:
    sh = d["site_home"].to_numpy(float)
    sa = d["site_away"].to_numpy(float)
    if flip:
        sh, sa = sa, sh
    return (bh * sh + ba * sa) / (pbar * (1 - pbar))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def add_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Derived columns used by the diagnosis and by registered repair arms.
    All are pregame (as-of ratings, schedule flags), never outcomes."""
    sh = d["site_home"].to_numpy(dtype="float32")
    sa = d["site_away"].to_numpy(dtype="float32")
    d["site_signed"] = (sh - sa).astype("float32")
    off_net = d["off_rating_off_c"].to_numpy("float64") - d["off_rating_def_c"].to_numpy("float64")
    def_net = d["def_rating_off_c"].to_numpy("float64") - d["def_rating_def_c"].to_numpy("float64")
    d["rating_gap"] = (off_net - def_net).astype("float32")
    d["site_x_gap"] = (d["site_signed"] * d["rating_gap"]).astype("float32")
    return d


def attach_conf(d: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for s in sorted(d["season"].unique()):
        g = pd.read_parquet(f"data/raw/cbbd/games_{int(s)}.parquet",
                            columns=["id", "conferenceGame", "seasonType"])
        frames.append(g)
    g = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    m = d[["cbbd_game_id"]].merge(g.rename(columns={"id": "cbbd_game_id"}),
                                  on="cbbd_game_id", how="left")
    d["conf_game"] = m["conferenceGame"].fillna(False).astype(bool).to_numpy()
    d["season_type"] = m["seasonType"].fillna("unknown").to_numpy()
    return d


EXPORT_DIR = FG_DIR / "round4_site"


def fit_s1(tr_all, te_all, feats, params, seed, offset_arm=False, log=None, export_arm=None):
    out = {}
    for c in FG.SHOT_CLASSES:
        tr, te = FG.class_slice(tr_all, c), FG.class_slice(te_all, c)
        te_dates = pd.to_datetime(te["game_date"])
        cuts = PO.month_boundaries(te_dates)
        cols = list(dict.fromkeys([*feats, "y"] + (["season", "game_id", "off_team_id", "def_team_id",
                                                      "site_home", "site_away"] if offset_arm else [])))
        p = np.zeros(len(te)); p0 = np.zeros(len(te)); pf = np.zeros(len(te))
        for k, cut in enumerate(cuts):
            nxt = cuts[k + 1] if k + 1 < len(cuts) else None
            seg = ((te_dates >= cut) if nxt is None
                   else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
            if not seg.any():
                continue
            before = (te_dates < cut).to_numpy()
            prior = te.loc[before, cols]
            rows = tr[cols] if not len(prior) else pd.concat([tr[cols], prior], ignore_index=True)
            ts = te.loc[seg]
            if offset_arm:
                import lightgbm as lgb
                bh, ba, pbar = fe_site_offsets(rows)
                if log is not None:
                    log.append({"class": c, "refit": str(pd.Timestamp(cut).date()),
                                "b_home": bh, "b_away": ba, "pbar": pbar})
                clf = lgb.LGBMClassifier(random_state=seed, n_jobs=NJ,
                                         **{**FG.LGBM_BASE, **params[c]})
                clf.fit(FG.design_matrix(rows, feats), rows["y"].to_numpy(),
                        init_score=_offset(rows, bh, ba, pbar))
                raw = clf.predict(FG.design_matrix(ts, feats), raw_score=True)
                p[seg] = _sig(raw + _offset(ts, bh, ba, pbar))
                p0[seg] = _sig(raw)
                pf[seg] = _sig(raw + _offset(ts, bh, ba, pbar, flip=True))
                if export_arm is not None:
                    _export(export_arm, c, cut, clf, feats, bh, ba, pbar, rows, tr, te_dates, before,
                            ts, p[seg])
                continue
            mdl = FG.LgbmArm(seed=seed, params=params[c]).fit(
                FG.design_matrix(rows, feats), rows["y"].to_numpy())
            p[seg] = mdl.predict_proba(FG.design_matrix(ts, feats))[:, 1]
            # counterfactuals: every site column set to neutral, and home<->away flipped
            tn = ts.copy()
            for f in ("site_home", "site_away", "site_signed", "site_x_gap"):
                if f in tn:
                    tn[f] = 0.0
            p0[seg] = mdl.predict_proba(FG.design_matrix(tn, feats))[:, 1]
            tf = ts.copy()
            if "site_home" in tf:
                tf["site_home"], tf["site_away"] = ts["site_away"].to_numpy(), ts["site_home"].to_numpy()
            for f in ("site_signed", "site_x_gap"):
                if f in tf:
                    tf[f] = -ts[f].to_numpy()
            pf[seg] = mdl.predict_proba(FG.design_matrix(tf, feats))[:, 1]
        if (p == 0).any():
            raise AssertionError("S1 left rows unscored")
        out[c] = (te.index.to_numpy(), p, p0, pf)
    return out


def _export(arm, c, cut, clf, feats, bh, ba, pbar, rows, tr, te_dates, before, ts, p_seg):
    """One dated G4 artifact in the round-4 joblib/manifest format, wrapped so
    `predict_proba` adds the site offset from the two trailing site columns.
    Written to a NEW directory (round4_site/<arm>/); parity with the fit-path
    prediction is asserted before anything is written."""
    import joblib
    from cbb_sim.models.fg_make_site_offset import SiteOffsetModel
    d = EXPORT_DIR / arm
    d.mkdir(parents=True, exist_ok=True)
    ext = list(feats) + ["site_home", "site_away"]
    w = SiteOffsetModel(clf, len(feats), bh, ba, pbar)
    chk = w.predict_proba(FG.design_matrix(ts, ext))[:, 1]
    gap = float(np.max(np.abs(chk - p_seg)))
    if gap > 1e-9:
        raise AssertionError(f"export parity failed: {gap}")
    tr_dates = pd.to_datetime(tr["game_date"])
    mx = tr_dates.max() if not before.any() else max(tr_dates.max(), te_dates[before].max())
    fname = f"{c}_{pd.Timestamp(cut).date()}.joblib"
    joblib.dump({"arm": "lgbm", "round4_arm": arm, "scheme": "S1", "shooter_key": "shot_shooter_id",
                 "feature_set": f"R4_B1_site_{arm}", "features": ext, "model": w, "fold": "F2",
                 "shot_class": c, "adopted": False, "refit_date": str(pd.Timestamp(cut).date()),
                 "max_train_date": str(pd.Timestamp(mx).date()), "possessions_version": "v2",
                 "servable": True, "offset": {"b_home": bh, "b_away": ba, "pbar": pbar},
                 "note": "Lane G home-site round arm G4 (fg_make experiments.md s21-22); NOT ADOPTED"},
                d / fname)
    man_p = d / f"manifest_{c}.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {
        "model": "fg_make", "scheme": "S1", "fold": "F2", "season": 2025, "key": c, "arm": arm,
        "feature_set": f"R4_B1_site_{arm}", "features": ext, "shooter_key": "shot_shooter_id",
        "servable": True, "adopted": False, "artifacts": []}
    man["artifacts"] = [x for x in man["artifacts"] if x["refit_date"] != str(pd.Timestamp(cut).date())]
    man["artifacts"].append({"refit_date": str(pd.Timestamp(cut).date()), "path": fname,
                             "max_train_date": str(pd.Timestamp(mx).date()), "n_train": int(len(rows)),
                             "parity_max_abs": gap})
    man_p.write_text(json.dumps(man, indent=2), encoding="utf-8")


KEEP = ["game_id", "cbbd_game_id", "season", "game_date", "off_team_id", "def_team_id",
        "offense_is_home", "neutral_site", "site_home", "site_away", "shot_class", "y",
        "rating_gap", "conf_game", "season_type", "off_make_c", "def_allow_c",
        "shooter_shrunk_dev_c", "n_prior_off", "n_prior_def"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="S0")
    ap.add_argument("--folds", default="F2")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--export", action="store_true",
                    help="write the offset arm's dated artifacts to round4_site/<arm>/ (F2 only)")
    a = ap.parse_args()
    t0 = time.time()
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([2022, 2023, 2024, 2025], context="fg_make site round")
    OUT.mkdir(parents=True, exist_ok=True)
    ladder = json.loads((FG_DIR / "lgbm_ladder_v2.json").read_text(encoding="utf-8"))
    params = {c: dict(v) for c, v in ladder["frozen_params"].items()}
    design = pd.read_parquet(DESIGN)
    extra = pd.read_parquet(EXTRA_CACHE, columns=["shooter_shrunk_dev_c"])
    design["shooter_shrunk_dev_c"] = extra["shooter_shrunk_dev_c"].to_numpy()
    del extra
    design = add_columns(attach_conf(design))
    print(f"[{time.time() - t0:6.0f}s] design {design.shape}; n_jobs={NJ}", flush=True)
    for fold in a.folds.split(","):
        tr_all, te_all = FG.fold_slices(design, fold)
        for arm in a.arms.split(","):
            feats, desc = ARMS[arm]
            for seed in (int(s) for s in a.seeds.split(",")):
                t1 = time.time()
                offs = []
                res = fit_s1(tr_all, te_all, feats, params, seed,
                             offset_arm=arm in OFFSET_ARMS, log=offs,
                             export_arm=(arm if (a.export and fold == "F2" and seed == 0
                                                 and arm in OFFSET_ARMS) else None))
                if offs:
                    (OUT / f"offsets_{arm}_{fold}_s{seed}.json").write_text(json.dumps(offs, indent=1))
                frames = []
                for c, (idx, p, p0, pf) in res.items():
                    f = te_all.loc[idx, KEEP].copy()
                    f["p"] = p; f["p_neutral"] = p0; f["p_flip"] = pf
                    frames.append(f)
                f = pd.concat(frames, ignore_index=True)
                path = OUT / f"preds_{arm}_{fold}_s{seed}.parquet"
                f.to_parquet(path, index=False)
                ll = {c: float(-np.mean(np.where(g["y"] == 1, np.log(np.clip(g["p"], 1e-9, 1)),
                                                 np.log(np.clip(1 - g["p"], 1e-9, 1)))))
                      for c, g in f.groupby("shot_class")}
                print(f"[{time.time() - t0:6.0f}s] {arm} {fold} s{seed}: {time.time() - t1:.0f}s "
                      f"ll={json.dumps({k: round(v, 6) for k, v in ll.items()})} -> {path}",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
