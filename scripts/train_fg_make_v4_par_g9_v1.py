"""train_fg_make_v4_par_g9_v1.py -- lane A day 2026-10-01: fg_make team-response arms for the G9 slope round
(docs/models/aggregation/experiments.md section 3).

Wraps `train_fg_make_v4_par_v1.py` (B1 spec, S1 monthly schedule) WITHOUT editing it or anything it imports.
What the wrapper changes, all by explicit option and all recorded in <out-dir>/g9_wrapper.json:

  --fold F1|F2          training/test fold of the S1 schedule (the wrapped trainer hard-codes F2). `fit_m`
                        is the wrapped trainer's own (m chosen on F1's TEST season 2024, as served); for fold-1
                        runs that is one scalar per class seen from the test season, identical for every arm.
  --extra-features      design columns appended to the team block (e.g. off_att_prior,def_att_prior;
                        off_make_v,def_allow_v = the E3 posterior variances, built by --e3-variance).
  --monotone            LightGBM monotone constraint +1 on off_make_c and def_allow_c.
  --min-child N|cv      min_child_samples per class; `cv` = chosen per class by team-season-grouped 3-fold CV
                        log loss on the fold's TRAINING seasons over --cv-grid (B1 params otherwise).
  --rawfix              with --team-rate-table: re-derive off_make_raw / def_allow_raw = c + lg_make_asof
                        (the 09-30 train/serve skew fix, = train_fg_make_v4_par_rawfix_v1.py).
  --e3-variance         with --team-rate-table: add off_make_v / def_allow_v from that table
                        (make_{rim,jump,3}_{off,def}_v of the offence / defence team).
Always: per test row, the out-of-sample make probability and the same model's probability with the two team
make-rate features (off_make_c, def_allow_c) set to 0 are written to <out-dir>/preds_<fold>_s<seed>.parquet
for the design-level team-game metrics (grade_g9_team_response_v1.py).

    .venv/Scripts/python.exe scripts/train_fg_make_v4_par_g9_v1.py --fold F2 --monotone --min-child cv \
        --arms B1 --no-floor --seed 0 --n-jobs 4 --out-dir data/processed/models/fg_make/round_g9/G3R_s0_F2
"""
from __future__ import annotations

import train_par_common_v1 as C

C.pin_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cbb_sim.team_rate_adapter as TRA  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402
import train_fg_make_v4_shooter_block as R4M  # noqa: E402
import train_fg_make_v4_par_v1 as P  # noqa: E402

TEAM_RATE = ("off_make_c", "def_allow_c")
E3_PREFIX = {"FGA_rim": "make_rim", "FGA_jump2": "make_jump", "FGA_3": "make3"}

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--fold", choices=["F1", "F2"], default="F2")
ap.add_argument("--extra-features", default="")
ap.add_argument("--monotone", action="store_true")
ap.add_argument("--min-child", default="")
ap.add_argument("--cv-grid", default="200,800,3200,12800")
ap.add_argument("--rawfix", action="store_true")
ap.add_argument("--e3-variance", action="store_true")
G9, REST = ap.parse_known_args()
EXTRA = [c for c in G9.extra_features.split(",") if c]
_STASH: dict = {}
_CTX: dict = {}


# --------------------------------------------------------------------------- team-rate adapter patch
_orig_apply = TRA.apply


def _apply_g9(frame, table_path, submodel, *a, **k):
    k["fold"] = G9.fold
    if G9.fold == "F1":   # the F1 table has no 2024-25 rows; fold 1 never uses them
        frame = frame[frame["season"] <= 2024].reset_index(drop=True)
    out = _orig_apply(frame, table_path, submodel, *a, **k)
    if submodel != "fg_make":
        return out
    if G9.rawfix:
        for c, raw in (("off_make_c", "off_make_raw"), ("def_allow_c", "def_allow_raw")):
            out[raw] = out[c].to_numpy(dtype="float64") + out["lg_make_asof"].to_numpy(dtype="float64")
        print("g9 rawfix: off_make_raw / def_allow_raw re-derived as c + lg_make_asof", flush=True)
    if G9.e3_variance:
        t = pd.read_parquet(table_path)
        t = t[t["fold"] == G9.fold]
        out["off_make_v"] = np.nan
        out["def_allow_v"] = np.nan
        for cls, pre in E3_PREFIX.items():
            m = (out["shot_class"] == cls).to_numpy()
            o = t[["game_id", "team_id", f"{pre}_off_v"]].rename(columns={"team_id": "off_team_id"})
            dd = t[["game_id", "team_id", f"{pre}_def_v"]].rename(columns={"team_id": "def_team_id"})
            sub = out.loc[m, ["game_id", "off_team_id", "def_team_id"]].reset_index()
            sub = sub.merge(o, on=["game_id", "off_team_id"], how="left").merge(dd, on=["game_id", "def_team_id"],
                                                                                  how="left")
            out.loc[sub["index"].to_numpy(), "off_make_v"] = sub[f"{pre}_off_v"].to_numpy()
            out.loc[sub["index"].to_numpy(), "def_allow_v"] = sub[f"{pre}_def_v"].to_numpy()
        miss = int(out[["off_make_v", "def_allow_v"]].isna().any(axis=1).sum())
        if miss:
            raise SystemExit(f"e3 variance: {miss} design rows without a table key")
        print("g9 e3 variance features joined (0 unmatched)", flush=True)
    return out


TRA.apply = _apply_g9


# --------------------------------------------------------------------------- tasks with fold / params
def _worker(X, y, Xte, params, seed, zero_idx):
    C.pin_threads()
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(random_state=seed, n_jobs=1, **{**FG.LGBM_BASE, **params})
    clf.fit(X, y)
    mdl = FG.LgbmArm(seed=seed, params=params)
    mdl.clf_ = clf
    Xz = Xte.copy()
    Xz[:, zero_idx] = 0.0
    return {"model": mdl, "p": mdl.predict_proba(Xte), "p_zero": mdl.predict_proba(Xz)}


def _class_params(base: dict, cls: str, feats: list[str]) -> dict:
    p = dict(base)
    mc = _CTX.get("min_child", {}).get(cls)
    if mc is not None:
        p["min_child_samples"] = int(mc)
    if G9.monotone:
        p["monotone_constraints"] = [1 if f in TEAM_RATE else 0 for f in feats]
    return p


def build_tasks(design, arm, params, seed, cuts_hook=None):
    from cbb_sim.models import possession_outcome as PO
    tr_all, te_all = FG.fold_slices(design, G9.fold)
    tasks, plan = {}, {}
    for c in FG.SHOT_CLASSES:
        feats = R4M.COMMON + R4M.SHOOTER_BLOCK[arm]
        zero_idx = [feats.index(f) for f in TEAM_RATE]
        tr, te = FG.class_slice(tr_all, c), FG.class_slice(te_all, c)
        te_dates = pd.to_datetime(te["game_date"])
        tr_dates = pd.to_datetime(tr["game_date"])
        cuts = PO.month_boundaries(te_dates)
        cols = [*feats, "y"]
        plan[c] = {"te": te, "n": len(te), "segs": []}
        prm = _class_params(params[c], c, feats)
        for k, cut in enumerate(cuts):
            nxt = cuts[k + 1] if k + 1 < len(cuts) else None
            seg = ((te_dates >= cut) if nxt is None else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
            if not seg.any():
                continue
            before = (te_dates < cut).to_numpy()
            prior = te.loc[before, cols]
            rows = tr[cols] if not len(prior) else pd.concat([tr[cols], prior], ignore_index=True)
            max_train = (tr_dates.max() if not before.any() else max(tr_dates.max(), te_dates[before].max()))
            key = f"{arm}_s{seed}_{c}_{cut.date()}"
            tasks[key] = dict(X=FG.design_matrix(rows, feats), y=rows["y"].to_numpy(),
                              Xte=FG.design_matrix(te.loc[seg], feats), params=prm, seed=seed, zero_idx=zero_idx)
            plan[c]["segs"].append({"key": key, "seg": seg, "cut": cut, "n_train": len(rows),
                                    "n_prior": len(prior), "n_scored": int(seg.sum()),
                                    "max_train": str(pd.Timestamp(max_train).date())})
    _CTX["plans"] = plan
    return tasks, plan


_orig_run_tasks = C.run_tasks


def _run_tasks(tasks, worker, n_jobs, ckpt_dir, *a, **k):
    res = _orig_run_tasks(tasks, _worker, n_jobs, ckpt_dir, *a, **k)
    _CTX["results"] = res
    return res


def _cv_min_child(design, params) -> dict:
    """team-season-grouped 3-fold CV log loss on the fold's TRAINING seasons, per class, over the grid."""
    from sklearn.model_selection import GroupKFold
    tr_all, _ = FG.fold_slices(design, G9.fold)
    grid = [int(x) for x in G9.cv_grid.split(",")]
    tasks, meta = {}, {}
    for c in FG.SHOT_CLASSES:
        feats = R4M.COMMON + R4M.SHOOTER_BLOCK["B1"]
        tr = FG.class_slice(tr_all, c)
        X = FG.design_matrix(tr, feats)
        y = tr["y"].to_numpy()
        grp = (tr["season"].astype(str) + "_" + tr["off_team_id"].astype(str)).to_numpy()
        for f, (i_tr, i_te) in enumerate(GroupKFold(n_splits=3).split(X, y, grp)):
            for mc in grid:
                prm = _class_params(params[c], c, feats)
                prm["min_child_samples"] = mc
                key = f"cv_{c}_{mc}_f{f}"
                tasks[key] = dict(X=X[i_tr], y=y[i_tr], Xte=X[i_te], params=prm, seed=0,
                                  zero_idx=[feats.index(t) for t in TEAM_RATE])
                meta[key] = (c, mc, f, y[i_te])
    res = _orig_run_tasks(tasks, _worker, _CTX["n_jobs"], _CTX["out_dir"] / "cv_cuts")
    tab = {}
    for key, (c, mc, f, yte) in meta.items():
        p = np.clip(res[key]["p"][:, FG.CLASS_INDEX["MAKE"]], 1e-6, 1 - 1e-6)
        ll = float(-np.mean(yte * np.log(p) + (1 - yte) * np.log(1 - p)))
        tab.setdefault(c, {}).setdefault(mc, []).append((ll, len(yte)))
    out = {}
    for c, d in tab.items():
        rows = {mc: float(sum(l * n for l, n in v) / sum(n for _, n in v)) for mc, v in d.items()}
        out[c] = {"grid_log_loss": rows, "chosen": int(min(rows, key=rows.get))}
    return out


_orig_score = FG.score


def _score(te, p):
    cls = str(te["shot_class"].iloc[0])
    _STASH[cls] = (te, p)
    return _orig_score(te, p)


def main() -> int:
    sys.argv = [sys.argv[0], *REST]
    a_ns = argparse.Namespace()
    pa = argparse.ArgumentParser(add_help=False)
    pa.add_argument("--out-dir", type=Path, required=True)
    pa.add_argument("--n-jobs", type=int, default=4)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--team-rate-table", type=Path, default=None)
    known, _ = pa.parse_known_args(REST, namespace=a_ns)
    out_dir = known.out_dir / known.team_rate_table.stem if known.team_rate_table else known.out_dir
    _CTX.update({"n_jobs": known.n_jobs, "out_dir": out_dir})
    if EXTRA:
        R4M.COMMON = list(R4M.COMMON) + EXTRA
    P.build_tasks = build_tasks
    C.run_tasks = _run_tasks
    FG.score = _score
    # min_child: fixed or CV (CV needs the design as the trainer builds it: hook the first build_tasks call)
    if G9.min_child and G9.min_child != "cv":
        _CTX["min_child"] = {c: int(G9.min_child) for c in FG.SHOT_CLASSES}
    if G9.min_child == "cv":
        inner = build_tasks

        def build_tasks_cv(design, arm, params, seed, cuts_hook=None):
            if "min_child" not in _CTX:
                cv = _cv_min_child(design, params)
                _CTX["min_child"] = {c: v["chosen"] for c, v in cv.items()}
                _CTX["cv"] = cv
                print("g9 min_child CV:", json.dumps(cv), flush=True)
            return inner(design, arm, params, seed, cuts_hook)
        P.build_tasks = build_tasks_cv
    rc = P.main()
    if rc:
        return rc
    rows = []
    for cls, (te, p) in _STASH.items():
        plan = _CTX["plans"][cls]
        pz = np.zeros(len(te))
        for s in plan["segs"]:
            pz[s["seg"]] = _CTX["results"][s["key"]]["p_zero"][:, FG.CLASS_INDEX["MAKE"]]
        fr = pd.DataFrame({"game_id": te["game_id"].to_numpy(), "season": te["season"].to_numpy(),
                           "game_date": te["game_date"].to_numpy(), "off_team_id": te["off_team_id"].to_numpy(),
                           "def_team_id": te["def_team_id"].to_numpy(), "shooter_id": te["shooter_id"].to_numpy(),
                           "shot_class": cls, "y": te["y"].to_numpy(), "p": p[:, FG.CLASS_INDEX["MAKE"]],
                           "p_zero": pz})
        for col in ["off_make_c", "def_allow_c", "shooter_shrunk_dev_c", *EXTRA]:
            fr[col] = te[col].to_numpy()
        rows.append(fr)
    pd.concat(rows, ignore_index=True).to_parquet(out_dir / f"preds_{G9.fold}_s{known.seed}.parquet", index=False)
    (out_dir / "g9_wrapper.json").write_text(json.dumps({
        "fold": G9.fold, "extra_features": EXTRA, "monotone": G9.monotone, "min_child": _CTX.get("min_child"),
        "cv": _CTX.get("cv"), "rawfix": G9.rawfix, "e3_variance": G9.e3_variance, "argv_wrapped": REST,
        "note": "artifact metadata 'fold' is the wrapped trainer's literal F2; trust this file's fold"},
        indent=2, default=str), encoding="utf-8")
    print("g9 wrapper: preds + g9_wrapper.json written to", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
