"""train_fg_make_two_stage_v2.py -- lane A day 2026-10-01, round 3 (docs/models/fg_make/experiments.md section 27):
TS-R, and the reference arm aL.

Versioned sibling of train_fg_make_two_stage_v1.py (imported for its stage-B helpers, not edited).

Stage A: LightGBM, served B1 params, S1 monthly schedule, fitted with init_score = logit(lg_make_asof) (the league
    level enters as a logit OFFSET, never as a split variable). Features:
      TSR: R2_SAFE_STATE + shooter_shrunk_dev_c                      (within-game only)
      aL : the served B1 feature list (team rates, ratings, site, season_idx, state, shooter dev)  -- reference arm:
           the served trees with only the league-offset change; no stage B.
Stage B (TSR only), per shot type, fitted on the fold's TRAINING seasons only:
    pass 1: logit offset x'b with x = (site_home, site_away, off_rating_off_c, def_rating_def_c) [ratings are
            league-centred], b by Newton on training rows (tau = 0);
    residual stats S, I as of (strictly before the game date) relative to p1 = sigmoid(stage A + x'b);
    pass 2: grid over tau_off, tau_def (TS2), b refitted by Newton at every grid point with offset u + v;
    final team term = x'b + u_off + v_def.
Outputs (per arm, as v1): B1/ manifests + TwoStageModelR joblibs (features = stage-A features + lg_make_asof +
fg_team_offset), preds_<fold>_s<seed>.parquet, offsets_<fold>_s<seed>.parquet (every fold-2 engine game x side x
type, computed from the ENGINE's rating columns), stageB_<fold>_s<seed>.json.
"""
from __future__ import annotations

import train_par_common_v1 as C

C.pin_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402
import train_fg_make_two_stage_v1 as V1  # noqa: E402
import train_fg_make_v4_shooter_block as R4M  # noqa: E402
from fg_two_stage_model_v2 import TwoStageModelR  # noqa: E402

FA_TSR = [*FG.R2_SAFE_STATE, "shooter_shrunk_dev_c"]
FA_AL = list(R4M.COMMON) + R4M.SHOOTER_BLOCK["B1"]
XB = ["site_home", "site_away", "off_rating_off_c", "def_rating_def_c"]
TAU_GRID = V1.TAU_GRID
logit, sig, EPS = V1.logit, V1.sig, V1.EPS


def _fit(X, y, Xte, params, seed, init, init_te):
    C.pin_threads()
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(random_state=seed, n_jobs=1, **{**FG.LGBM_BASE, **params})
    clf.fit(X, y, init_score=init)
    return {"clf": clf, "p": sig(clf.predict(Xte, raw_score=True) + init_te)}


def stage_a(design, fold, seed, params, n_jobs, ckpt, feats, oof, t0):
    from sklearn.model_selection import GroupKFold
    tr_all, te_all = FG.fold_slices(design, fold)
    tasks, plan = {}, {}
    for c in FG.SHOT_CLASSES:
        tr, te = FG.class_slice(tr_all, c), FG.class_slice(te_all, c)
        prm = params[c]
        if oof:
            X = FG.design_matrix(tr, feats); y = tr["y"].to_numpy(); off = logit(tr["lg_make_asof"].to_numpy())
            grp = (tr["season"].astype(str) + "_" + tr["off_team_id"].astype(str)).to_numpy()
            folds = list(GroupKFold(n_splits=3).split(X, y, grp))
            for f, (a, b) in enumerate(folds):
                tasks[f"oof_{c}_{f}"] = dict(X=X[a], y=y[a], Xte=X[b], params=prm, seed=seed, init=off[a], init_te=off[b])
        else:
            folds = []
        te_dates = pd.to_datetime(te["game_date"]); tr_dates = pd.to_datetime(tr["game_date"])
        cuts = PO.month_boundaries(te_dates)
        segs = []
        cols = [*feats, "y", "lg_make_asof"]
        for k, cut in enumerate(cuts):
            nxt = cuts[k + 1] if k + 1 < len(cuts) else None
            seg = ((te_dates >= cut) if nxt is None else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
            if not seg.any():
                continue
            before = (te_dates < cut).to_numpy()
            rows = pd.concat([tr[cols], te.loc[before, cols]], ignore_index=True) if before.any() else tr[cols]
            max_train = tr_dates.max() if not before.any() else max(tr_dates.max(), te_dates[before].max())
            key = f"s1_{c}_{cut.date()}"
            tasks[key] = dict(X=FG.design_matrix(rows, feats), y=rows["y"].to_numpy(),
                              Xte=FG.design_matrix(te.loc[seg], feats), params=prm, seed=seed,
                              init=logit(rows["lg_make_asof"].to_numpy()),
                              init_te=logit(te.loc[seg, "lg_make_asof"].to_numpy()))
            segs.append({"key": key, "seg": seg, "cut": cut, "max_train": str(pd.Timestamp(max_train).date()),
                         "n_train": len(rows)})
        plan[c] = {"tr_idx": tr.index.to_numpy(), "te_idx": te.index.to_numpy(), "folds": folds, "segs": segs}
    V1.log(f"stage A: {len(tasks)} fits", t0)
    res = C.run_tasks(tasks, _fit, n_jobs, ckpt)
    pA = pd.Series(np.nan, index=design.index)
    for c, pl in plan.items():
        for f, (a, b) in enumerate(pl["folds"]):
            pA.loc[pl["tr_idx"][b]] = res[f"oof_{c}_{f}"]["p"]
        for s in pl["segs"]:
            pA.loc[pl["te_idx"][s["seg"]]] = res[s["key"]]["p"]
    return pA, plan, res


def newton(y, base, X, iters=8):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sig(base + X @ b)
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        b = b + np.linalg.solve(H, g)
    p = np.clip(sig(base + X @ b), EPS, 1 - EPS)
    return b, float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def export(out, c, plan, res, arm, feats, fold, bst, test_season):
    arts = []
    for s in plan[c]["segs"]:
        fname = f"{c}_{s['cut'].date()}.joblib"
        joblib.dump({"arm": "lgbm", "scheme": "S1", "two_stage_arm": arm,
                     "features": [*feats, "lg_make_asof", "fg_team_offset"],
                     "model": TwoStageModelR(res[s["key"]]["clf"]), "fold": fold, "shot_class": c, "adopted": False,
                     "refit_date": str(s["cut"].date()), "max_train_date": s["max_train"], "stage_b": bst,
                     "note": "fg_make round TS-R (experiments.md s27), lane A 2026-10-01"}, out / "B1" / fname)
        arts.append({"refit_date": str(s["cut"].date()), "path": fname, "max_train_date": s["max_train"],
                     "n_train": s["n_train"]})
    (out / "B1" / f"manifest_{c}.json").write_text(json.dumps({
        "model": "fg_make", "scheme": "S1", "fold": fold, "season": test_season, "key": c, "arm": arm,
        "feature_set": f"two_stage_{arm}", "features": [*feats, "lg_make_asof", "fg_team_offset"],
        "shooter_key": "shot_shooter_id", "servable": False, "artifacts": arts}, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=["TSR", "aL"], required=True)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out-root", type=Path, default=ROOT / "data/processed/models/fg_make/round_tsr")
    a = ap.parse_args()
    t0 = time.time()
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([2022, 2023, 2024, 2025], context="fg_make TS-R")
    design = pd.read_parquet(V1.DESIGN)
    design["shooter_shrunk_dev_c"] = pd.read_parquet(V1.EXTRA, columns=["shooter_shrunk_dev_c"])[
        "shooter_shrunk_dev_c"].to_numpy()
    last = 2024 if a.fold == "F1" else 2025
    design = design[design["season"] <= last].reset_index(drop=True)
    params = {c: {k: x for k, x in dict(v).items() if k != "n_jobs"}
              for c, v in json.loads(V1.LADDER.read_text(encoding="utf-8"))["frozen_params"].items()}
    feats = FA_TSR if a.arm == "TSR" else FA_AL
    a.out_root.mkdir(parents=True, exist_ok=True)
    pA, plan, res = stage_a(design, a.fold, a.seed, params, a.n_jobs, a.out_root / f"_stageA_{a.arm}_s{a.seed}_{a.fold}",
                            feats, a.arm == "TSR", t0)
    used = pA.notna().to_numpy()
    d = design.loc[used].copy()
    d["pA"] = pA[used].to_numpy()
    test_season = max(FG.FOLDS[a.fold]["test"])
    out = a.out_root / f"{a.arm}_s{a.seed}_{a.fold}"
    (out / "B1").mkdir(parents=True, exist_ok=True)
    games = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet") if a.fold == "F2" else None
    ts_eng = None
    if games is not None:
        z = np.load(ROOT / "data/processed/models/engine_v3/arrays_F2_2025.npz")["team_static"]
        nm = json.loads((ROOT / "data/processed/models/engine_v3/names_F2_2025.json").read_text())["team_names"]
        ts_eng = {k: z[:, :, nm[k]].astype(np.float64) for k in ("off_rating_off_c", "def_rating_def_c")}
    fitted, pred_rows, off_rows = {}, [], []
    for c in FG.SHOT_CLASSES:
        key = V1.KEY[c]
        dc = d[d["shot_class"] == c].copy()
        te_mask = (dc["season"] == test_season).to_numpy()
        if a.arm == "aL":
            bst = {"b": [0.0] * 4, "tau_off": 0.0, "tau_def": 0.0}
            off_all = np.zeros(len(dc))
        else:
            tr = dc[~te_mask]
            Xtr = tr[XB].to_numpy(np.float64)
            y = tr["y"].to_numpy(float)
            b1, _ = newton(y, logit(tr["pA"].to_numpy()), Xtr)
            dc["p1"] = sig(logit(dc["pA"].to_numpy()) + dc[XB].to_numpy(np.float64) @ b1)
            dd = dc.assign(pA=dc["p1"])                         # residuals beyond stage A + site + ratings
            stats = {}
            for side, tcol in (("off", "off_team_id"), ("def", "def_team_id")):
                byd, fin = V1.team_game_stats(dd, side)
                stats[side] = byd
                keys = dc[["season", tcol, "shot_class", "game_date"]].rename(columns={tcol: "team_id"})
                S, I = V1.asof_lookup(byd, keys)
                dc[f"cS_{side}"] = S; dc[f"cI_{side}"] = I
            tr = dc[~te_mask]
            base = logit(tr["pA"].to_numpy()); Xtr = tr[XB].to_numpy(np.float64)
            grid = []
            for to in TAU_GRID:
                for td in TAU_GRID:
                    u = V1.post_mean(tr["cS_off"].to_numpy(), tr["cI_off"].to_numpy(), to, 0.0)
                    v = V1.post_mean(tr["cS_def"].to_numpy(), tr["cI_def"].to_numpy(), td, 0.0)
                    b, ll = newton(y, base + u + v, Xtr)
                    grid.append({"tau_off": to, "tau_def": td, "b": b.tolist(), "ll": ll})
            bst = min(grid, key=lambda r: r["ll"])
            bst["b_pass1"] = b1.tolist()
            fitted[c] = {"grid": grid}
            off_all = (dc[XB].to_numpy(np.float64) @ np.array(bst["b"])
                       + V1.post_mean(dc["cS_off"].to_numpy(), dc["cI_off"].to_numpy(), bst["tau_off"], 0.0)
                       + V1.post_mean(dc["cS_def"].to_numpy(), dc["cI_def"].to_numpy(), bst["tau_def"], 0.0))
            if games is not None:
                gd = pd.to_datetime(games["game_date"]).to_numpy()
                neu = games["neutral"].to_numpy().astype(bool)
                for side, own, opp in ((0, "home_team_id", "away_team_id"), (1, "away_team_id", "home_team_id")):
                    kf = pd.DataFrame({"season": test_season, "team_id": games[own].to_numpy(), "shot_class": c,
                                       "game_date": gd})
                    kd = kf.assign(team_id=games[opp].to_numpy())
                    So, Io = V1.asof_lookup(stats["off"], kf)
                    Sd, Id = V1.asof_lookup(stats["def"], kd)
                    sh = np.where(neu, 0.0, 1.0 if side == 0 else 0.0)
                    sa = np.where(neu, 0.0, 0.0 if side == 0 else 1.0)
                    xg = np.column_stack([sh, sa, ts_eng["off_rating_off_c"][:, side], ts_eng["def_rating_def_c"][:, side]])
                    val = (xg @ np.array(bst["b"]) + V1.post_mean(So, Io, bst["tau_off"], 0.0)
                           + V1.post_mean(Sd, Id, bst["tau_def"], 0.0))
                    off_rows.append(pd.DataFrame({"game_id": games["game_id"].to_numpy(), "side": side, "key": key,
                                                  "fg_team_offset": val}))
        te = dc[te_mask]
        ote = off_all[te_mask]
        p = sig(logit(te["pA"].to_numpy()) + ote)
        if games is not None:
            lg = te.assign(_d=pd.to_datetime(te["game_date"]).astype("datetime64[ns]")).groupby(
                "_d")["lg_make_asof"].first().sort_index()
            lgv = np.array([lg.asof(x) for x in pd.to_datetime(games["game_date"]).astype("datetime64[ns]")])
            for side in (0, 1):
                if a.arm == "aL":
                    off_rows.append(pd.DataFrame({"game_id": games["game_id"].to_numpy(), "side": side, "key": key,
                                                  "fg_team_offset": 0.0, "lg_make_asof": lgv}))
                else:
                    off_rows[-2 + side]["lg_make_asof"] = lgv
        fr = pd.DataFrame({"game_id": te["game_id"].to_numpy(), "season": te["season"].to_numpy(),
                           "game_date": te["game_date"].to_numpy(), "off_team_id": te["off_team_id"].to_numpy(),
                           "def_team_id": te["def_team_id"].to_numpy(), "shooter_id": te["shooter_id"].to_numpy(),
                           "shot_class": c, "y": te["y"].to_numpy(), "p": p, "p_zero": te["pA"].to_numpy(),
                           "off_make_c": te["off_make_c"].to_numpy(), "def_allow_c": te["def_allow_c"].to_numpy(),
                           "shooter_shrunk_dev_c": te["shooter_shrunk_dev_c"].to_numpy(),
                           "lg_make_asof": te["lg_make_asof"].to_numpy(), "fg_team_offset": ote})
        pred_rows.append(fr)
        sc = FG.score(te, np.column_stack([1 - p, p]))
        scA = FG.score(te, np.column_stack([1 - te["pA"].to_numpy(), te["pA"].to_numpy()]))
        fitted.setdefault(c, {})
        fitted[c].update({"best": bst, "stageA_log_loss": scA["log_loss"],
                          "score": {k: sc[k] for k in ("n", "log_loss", "calib_pass", "calib_worst_gap_pp",
                                                       "resp_pass_decision8", "pred_make_rate", "actual_make_rate")}})
        print(f"  {a.arm} {c}: tau {bst['tau_off']}/{bst['tau_def']} b {np.round(bst['b'], 4).tolist()} "
              f"ll {sc['log_loss']:.6f} (stage A {scA['log_loss']:.6f}) calib {sc['calib_worst_gap_pp']:.3f}pp "
              f"D8 {'PASS' if sc['resp_pass_decision8'] else 'FAIL'}", flush=True)
        export(out, c, plan, res, a.arm, feats, a.fold, bst, test_season)
    pd.concat(pred_rows, ignore_index=True).to_parquet(out / f"preds_{a.fold}_s{a.seed}.parquet", index=False)
    if off_rows:
        pd.concat(off_rows, ignore_index=True).to_parquet(out / f"offsets_{a.fold}_s{a.seed}.parquet", index=False)
    (out / f"stageB_{a.fold}_s{a.seed}.json").write_text(json.dumps(fitted, indent=1, default=float), encoding="utf-8")
    V1.log(f"{a.arm} written to {out}", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
