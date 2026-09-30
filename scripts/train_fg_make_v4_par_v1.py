"""Versioned parallel sibling of `train_fg_make_v4_shooter_block.py` (lane J, 2026-09-30).

EXECUTION-ONLY change: the serial `fit_s1` loop (class x monthly refit cut) becomes one
process-parallel task per (arm, seed, class, cut) through `train_par_common_v1.run_tasks`,
every fit single-threaded (LightGBM does not multi-thread in the cloud image, and locally
threads are slower; `FG.LgbmArm.fit` hard-codes n_jobs=-1, so it is replaced here by an
n_jobs=1 fit). Row selection, `FG.design_matrix`, params, seeds, artifact payloads, file
names, `FG.score`, the offline decision and the noise floor are the wrapped trainer's own
(imported unmodified). The wrapped trainer is NOT edited and nothing existing is overwritten:
every output goes under `--out-dir`.

Box usage (arm B1 + its seed-1 noise floor = 3 classes x ~6 cuts x 2 = ~36 fits at once):
    python scripts/train_fg_make_v4_par_v1.py --mode run --arms B1 --n-jobs 40 \
        --out-dir data/processed/models/fg_make/round5_par/ \
        --feature-table data/processed/team_rate_features_<arm>_v1.parquet \
        --extra-cache data/processed/models/fg_make/round5_par/design_v4_extra_v3.parquet
Resume after a spot reclaim: the SAME command (finished tasks are read from <out-dir>/cuts/).

OVERLAY (`--feature-table`): replaces `off_make_c`, `def_allow_c` (the two TEAM_FEATURES that are
team rates) by the sibling table's values, joined on `--overlay-keys` (default
game_id,off_team_id,shot_class). `off_make_raw` / `def_allow_raw` are re-derived as
`c + lg_make_asof` unless the table carries them. DERIVED columns that depend on the team rate:
`off_make_raw` -> `fit_m` and `shooter_shrunk_dev_c` (and the round-4 extra cache holding it,
plus the five_shrunk_dev_* spacing columns' shrink target is from EVENTS, not the design, so
those do NOT move). Therefore an overlay REQUIRES a fresh `--extra-cache` path that does not
exist (the wrapper refuses an existing cache with an overlay): build_extra reruns on the
overlaid design (needs `--events`, data for ES.load_universe). Not recomputed by the overlay:
`off_att_prior`/`def_att_prior`, `lg_make_asof`, the shooter columns, the rating columns.
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

import train_fg_make_v4_shooter_block as R4M  # noqa: E402

from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

OVERLAY_COLS = ["off_make_c", "def_allow_c"]


# ---------------------------------------------------------------------------
# worker (runs in a separate process): one fit + predict for one (class, cut)
# ---------------------------------------------------------------------------
def _fit_worker(X, y, Xte, params, seed):
    C.pin_threads()
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(random_state=seed, n_jobs=1, **{**FG.LGBM_BASE, **params})
    clf.fit(X, y)
    mdl = FG.LgbmArm(seed=seed, params=params)
    mdl.clf_ = clf
    return {"model": mdl, "p": mdl.predict_proba(Xte)}


def build_tasks(design, arm, params, seed, cuts_hook=None):
    """One task per (class, cut). Row selection is `fit_s1`'s, verbatim."""
    tr_all, te_all = FG.fold_slices(design, "F2")
    tasks, plan = {}, {}
    for c in FG.SHOT_CLASSES:
        feats = R4M.COMMON + R4M.SHOOTER_BLOCK[arm]
        tr, te = FG.class_slice(tr_all, c), FG.class_slice(te_all, c)
        te_dates = pd.to_datetime(te["game_date"])
        tr_dates = pd.to_datetime(tr["game_date"])
        cuts = PO.month_boundaries(te_dates)
        if cuts_hook:
            cuts = cuts_hook(cuts)
        cols = [*feats, "y"]
        plan[c] = {"te": te, "n": len(te), "segs": []}
        for k, cut in enumerate(cuts):
            nxt = cuts[k + 1] if k + 1 < len(cuts) else None
            seg = ((te_dates >= cut) if nxt is None
                   else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
            if not seg.any():
                continue
            before = (te_dates < cut).to_numpy()
            prior = te.loc[before, cols]
            rows = tr[cols] if not len(prior) else pd.concat([tr[cols], prior],
                                                             ignore_index=True)
            max_train = (tr_dates.max() if not before.any()
                         else max(tr_dates.max(), te_dates[before].max()))
            key = f"{arm}_s{seed}_{c}_{cut.date()}"
            tasks[key] = dict(X=FG.design_matrix(rows, feats), y=rows["y"].to_numpy(),
                              Xte=FG.design_matrix(te.loc[seg], feats),
                              params=params[c], seed=seed)
            plan[c]["segs"].append({"key": key, "seg": seg, "cut": cut, "n_train": len(rows),
                                    "n_prior": len(prior), "n_scored": int(seg.sum()),
                                    "max_train": str(pd.Timestamp(max_train).date())})
    return tasks, plan


def assemble(arm, seed, plan, results, out_dir, export, m, feats):
    """`fit_s1`'s post-processing: predictions into place, score, export artifacts."""
    scores, segs = {}, {}
    if export:
        out_dir.mkdir(parents=True, exist_ok=True)
    for c in FG.SHOT_CLASSES:
        te = plan[c]["te"]
        p_s1 = np.zeros((plan[c]["n"], len(FG.CLASSES)), dtype="float64")
        segments = []
        for s in plan[c]["segs"]:
            r = results[s["key"]]
            p_s1[s["seg"]] = r["p"]
            info = {"refit_date": str(s["cut"].date()), "n_train": int(s["n_train"]),
                    "n_train_from_test_season": int(s["n_prior"]),
                    "n_scored": s["n_scored"], "max_train_date": s["max_train"]}
            if export:
                fname = f"{c}_{s['cut'].date()}.joblib"
                joblib.dump({"arm": "lgbm", "round4_arm": arm, "scheme": "S1",
                             "shooter_key": R4M.SHOOTER_KEY, "feature_set": R4M.LABEL[arm],
                             "features": feats[c], "model": r["model"], "fold": "F2",
                             "shot_class": c, "adopted": False,
                             "refit_date": info["refit_date"],
                             "max_train_date": info["max_train_date"],
                             "possessions_version": "v2",
                             "shrinkage_m": m[c]["m"] if arm != "B0" else None,
                             "servable": arm not in R4M.UNSERVABLE,
                             "note": "fg_make round 4 (par_v1 wrapper), experiments.md "
                                     f"section 19: shooter-block arm {arm} "
                                     f"({R4M.LABEL[arm]})"}, out_dir / fname)
                info["path"] = fname
            segments.append(info)
        if (p_s1.sum(axis=1) == 0).any():
            raise AssertionError("S1 left test rows unscored; month partition is not a cover")
        scores[c] = FG.score(te, p_s1)
        segs[c] = segments
        if export:
            (out_dir / f"manifest_{c}.json").write_text(json.dumps({
                "model": "fg_make", "scheme": "S1", "fold": "F2", "season": 2025, "key": c,
                "arm": arm, "feature_set": R4M.LABEL[arm], "features": feats[c],
                "shooter_key": R4M.SHOOTER_KEY,
                "shrinkage_m": m[c]["m"] if arm != "B0" else None,
                "servable": arm not in R4M.UNSERVABLE,
                "artifacts": [{k: v for k, v in s.items()
                               if k in ("refit_date", "path", "max_train_date", "n_train")}
                              for s in segments]}, indent=2), encoding="utf-8")
    return scores, segs


def load_design(a, t0):
    design = pd.read_parquet(a.design)
    rep = None
    if a.feature_table:
        keys = a.overlay_keys.split(",")
        tab_cols = pd.read_parquet(a.feature_table).columns
        cols = [c for c in OVERLAY_COLS if c in tab_cols]
        if not cols:
            raise SystemExit(f"{a.feature_table} carries none of {OVERLAY_COLS}")
        design, rep = C.overlay_columns(design, a.feature_table, keys, cols,
                                        strict=not a.allow_unmatched)
        # raw = centred + league as-of, unless the table supplies the raw column itself
        for c, raw in (("off_make_c", "off_make_raw"), ("def_allow_c", "def_allow_raw")):
            if c in cols and raw not in tab_cols:
                design[raw] = (design[c].to_numpy(dtype="float64")
                               + design["lg_make_asof"].to_numpy(dtype="float64"))
                rep.setdefault("raw_rederived", []).append(raw)
        R4M.log(f"overlay applied: {rep}", t0)
    return design, rep


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "identity", "overlay-identity"], default="run")
    ap.add_argument("--arms", default="B1")
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--design", type=Path, default=R4M.DESIGN)
    ap.add_argument("--events", type=Path, default=R4M.EVENTS)
    ap.add_argument("--extra-cache", type=Path, default=R4M.EXTRA_CACHE)
    ap.add_argument("--ladder-version", default="v2")
    ap.add_argument("--team-rate-missing", choices=["raise", "keep_served"], default="raise",
                    help="adapter policy for design rows with no table key (PM decision; default raise)")
    ap.add_argument("--feature-table", type=Path, default=None)
    ap.add_argument("--overlay-keys", default="game_id,off_team_id,shot_class")
    ap.add_argument("--team-rate-table", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-unmatched", action="store_true")
    ap.add_argument("--no-leak", action="store_true")
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--identity-n-est", type=int, default=20)
    ap.add_argument("--identity-train-rows", type=int, default=20000)
    a = ap.parse_args()
    t0 = time.time()
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(R4M.SEASONS, context="fg_make round 4 par_v1")

    if a.mode == "overlay-identity":
        d = pd.read_parquet(a.design)
        keys = a.overlay_keys.split(",")
        tab = d[[*keys, *OVERLAY_COLS]].drop_duplicates(keys)
        tp = a.out_dir / "identity_table.parquet"
        a.out_dir.mkdir(parents=True, exist_ok=True)
        tab.to_parquet(tp, index=False)
        d2, rep = C.overlay_columns(d, tp, keys, OVERLAY_COLS)
        ok = all(d[c].equals(d2[c]) for c in d.columns)
        print("overlay identity:", ok, {k: v for k, v in rep.items() if k != "table"})
        return 0 if ok else 1

    if a.team_rate_table and a.feature_table:
        raise SystemExit("--team-rate-table and --feature-table are mutually exclusive")
    trt_meta = None
    if a.team_rate_table:
        a.out_dir = a.out_dir / a.team_rate_table.stem
        trt_meta = {"path": str(a.team_rate_table), "stem": a.team_rate_table.stem,
                    "sha256": C.sha256_file(a.team_rate_table)}
    C.assert_fresh_out_dir(a.out_dir)
    C.stamp_owner(a.out_dir, "train_fg_make_v4_par_v1", vars(a))
    ladder = json.loads((R4M.FG_DIR / f"lgbm_ladder_{a.ladder_version}.json"
                         ).read_text(encoding="utf-8"))
    params = {c: {k: x for k, x in dict(v).items() if k != "n_jobs"} for c, v in ladder["frozen_params"].items()}

    if a.mode == "identity":
        return identity(a, params, t0)

    if (a.feature_table or a.team_rate_table) and a.extra_cache.exists():
        raise SystemExit(f"--feature-table with an existing --extra-cache {a.extra_cache}: "
                         "the cache holds shooter_shrunk_dev_c built from the OLD team rate. "
                         "Give a fresh cache path.")
    design, ov = load_design(a, t0)
    if a.team_rate_table:
        try:
            from cbb_sim.team_rate_adapter import apply as _trt_apply
        except ImportError as e:
            raise SystemExit(f"--team-rate-table needs cbb_sim.team_rate_adapter, not "
                             f"importable yet: {e}")
        design = _trt_apply(design, a.team_rate_table, "fg_make", fold="F2",
                            missing=a.team_rate_missing)
    m = R4M.fit_m(design)
    (a.out_dir / "m_fitted.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    if a.extra_cache.exists():
        extra = pd.read_parquet(a.extra_cache)
    else:
        extra = R4M.build_extra(design, pd.read_parquet(a.events),
                                {c: m[c]["m"] for c in FG.SHOT_CLASSES}, t0)
        a.extra_cache.parent.mkdir(parents=True, exist_ok=True)
        extra.to_parquet(a.extra_cache, index=False)
    for c in extra.columns:
        design[c] = extra[c].to_numpy()

    slot = design[["shooter_id", "game_date", "shot_class", "class_key",
                   *R4M.SLOT_EXPORT_PER_CLASS, *R4M.SLOT_EXPORT_SHARED]].copy()
    slot = slot.drop_duplicates(subset=["shooter_id", "game_date", "class_key"])
    wide = None
    for key in ("rim", "jump2", "three"):
        s = slot[slot["class_key"] == key][
            ["shooter_id", "game_date", *R4M.SLOT_EXPORT_PER_CLASS, *R4M.SLOT_EXPORT_SHARED]]
        s = s.rename(columns={c: f"{c}__{key}" for c in R4M.SLOT_EXPORT_PER_CLASS})
        wide = s if wide is None else wide.merge(
            s.drop(columns=list(R4M.SLOT_EXPORT_SHARED)), on=["shooter_id", "game_date"],
            how="outer")
    wide.to_parquet(a.out_dir / "slot_source_v2.parquet", index=False)

    worst = {}
    if not a.no_leak:
        lk = R4M.leak(design)
        (a.out_dir / "leak_test.json").write_text(json.dumps(lk, indent=2, default=str),
                                                  encoding="utf-8")
        for c in FG.SHOT_CLASSES:
            for r in lk[c]:
                worst[r["column"]] = max(worst.get(r["column"], 0.0),
                                         abs(float(r["corr_asjoined"]))
                                         if np.isfinite(r["corr_asjoined"]) else 0.0)
        print("leak worst |corr|:", json.dumps({k: round(v, 4) for k, v in worst.items()}))

    arms = [x for x in a.arms.split(",") if x]
    jobs = [(arm, a.seed, True) for arm in arms]
    if R4M.FLOOR_ARM in arms and not a.no_floor:
        jobs.append((R4M.FLOOR_ARM, a.seed + 1, False))
    all_tasks, plans = {}, {}
    for arm, seed, _ in jobs:
        t, p = build_tasks(design, arm, params, seed)
        all_tasks.update(t)
        plans[(arm, seed)] = p
    R4M.log(f"{len(all_tasks)} fit tasks across {len(jobs)} arm/seed jobs", t0)
    results = C.run_tasks(all_tasks, _fit_worker, a.n_jobs, a.out_dir / "cuts")
    missing = [k for k in all_tasks if k not in results]
    if missing:
        print(f"INCOMPLETE: {len(missing)} tasks missing; rerun the same command")
        return 3

    report = {"created_at": pd.Timestamp.now("UTC").isoformat(), "wrapper": "par_v1",
              "overlay": ov, "team_rate_table": trt_meta, "seed": a.seed, "arms": {}, "leak_worst_abs_corr": worst,
              "m_fitted": {c: m[c]["m"] for c in FG.SHOT_CLASSES}}
    all_scores, sc1 = {}, None
    for arm, seed, export in jobs:
        feats = {c: R4M.COMMON + R4M.SHOOTER_BLOCK[arm] for c in FG.SHOT_CLASSES}
        sc, sg = assemble(arm, seed, plans[(arm, seed)], results, a.out_dir / arm, export,
                          m, feats)
        if not export:
            sc1 = sc
            continue
        all_scores[arm] = sc
        report["arms"][arm] = {
            "label": R4M.LABEL[arm], "by_class": {c: {k: sc[c][k] for k in (
                "n", "log_loss", "brier", "calib_pass", "calib_worst_gap_pp",
                "resp_pass_decision8", "pred_make_rate", "actual_make_rate")}
                for c in FG.SHOT_CLASSES}, "segments": sg}
        for c in FG.SHOT_CLASSES:
            print(f"  {arm} {c}: ll {sc[c]['log_loss']:.6f} calib {sc[c]['calib_worst_gap_pp']:.3f}pp"
                  f" D8 {'PASS' if sc[c]['resp_pass_decision8'] else 'FAIL'}")
    if sc1 is not None:
        report["noise_floor"] = {"arm": R4M.FLOOR_ARM, "method": "second-seed refit of S1",
                                 "by_class": {c: round(abs(
                                     all_scores[R4M.FLOOR_ARM][c]["log_loss"]
                                     - sc1[c]["log_loss"]), 8) for c in FG.SHOT_CLASSES},
                                 "seed1_log_loss": {c: sc1[c]["log_loss"]
                                                    for c in FG.SHOT_CLASSES}}
    report["runtime_s"] = round(time.time() - t0, 1)
    (a.out_dir / "run_report.json").write_text(json.dumps(report, indent=2, default=str),
                                               encoding="utf-8")
    print(f"wrote {a.out_dir}/run_report.json ({report['runtime_s']}s)")
    return 0


# ---------------------------------------------------------------------------
# identity check: original serial R4M.fit_s1 vs the parallel path, class FGA_3,
# first 2 cuts, arm B1, downsampled train, cut n_estimators (TEST ONLY)
# ---------------------------------------------------------------------------
def identity(a, params, t0) -> int:
    design = pd.read_parquet(a.design)
    extra = pd.read_parquet(a.extra_cache)
    for c in extra.columns:
        design[c] = extra[c].to_numpy()
    cls = "FGA_3"
    design = design[design["shot_class"] == cls]
    tr_seasons = FG.FOLDS["F2"]["train"]
    trm = design["season"].isin(tr_seasons)
    keep = design[trm].sample(a.identity_train_rows, random_state=0).index
    design = design[~trm | design.index.isin(keep)]
    P = {c: {**v, "n_estimators": a.identity_n_est} for c, v in params.items()}
    hook = lambda cuts: cuts[:2]                                          # noqa: E731
    orig_mb = PO.month_boundaries
    PO.month_boundaries = lambda d: hook(orig_mb(d))
    orig_fit = FG.LgbmArm.fit
    m = {c: {"m": 100.0} for c in FG.SHOT_CLASSES}

    def fit_capped(self, X, y):                      # reference path: capped at 2 threads
        import lightgbm as lgb
        self.clf_ = lgb.LGBMClassifier(random_state=self.seed, n_jobs=2, **self.params)
        self.clf_.fit(X, y)
        return self
    FG.LgbmArm.fit = fit_capped
    orig_classes = FG.SHOT_CLASSES
    FG.SHOT_CLASSES = (cls,)
    R4M.FG.SHOT_CLASSES = (cls,)
    try:
        # serial reference through the ORIGINAL fit_s1 (exports off)
        sc_ref, seg_ref = R4M.fit_s1(design, "B1", P, 0, False, m)
        FG.LgbmArm.fit = orig_fit
        tasks, plan = build_tasks(design, "B1", P, 0)
        res = C.run_tasks(tasks, _fit_worker, 2, a.out_dir / "cuts_identity")
        feats = {cls: R4M.COMMON + R4M.SHOOTER_BLOCK["B1"]}
        sc_par, seg_par = assemble("B1", 0, plan, res, a.out_dir / "B1_identity", False, m,
                                   feats)
    finally:
        PO.month_boundaries = orig_mb
        FG.LgbmArm.fit = orig_fit
        FG.SHOT_CLASSES = orig_classes
        R4M.FG.SHOT_CLASSES = orig_classes
    same_ll = sc_ref[cls]["log_loss"] == sc_par[cls]["log_loss"]
    strip = lambda s: [{k: v for k, v in x.items() if k != "fit_s"} for x in s]  # noqa: E731
    same_seg = strip(seg_ref[cls]) == strip(seg_par[cls])
    print(f"identity {cls} first 2 cuts: log_loss ref {sc_ref[cls]['log_loss']!r} "
          f"par {sc_par[cls]['log_loss']!r} equal={same_ll}; segments equal={same_seg}")
    return 0 if (same_ll and same_seg) else 1


if __name__ == "__main__":
    raise SystemExit(main())
