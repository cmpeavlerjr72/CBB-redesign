"""possession_outcome round-2 S1 retrain, parallel ACROSS (population, refit date). Lane J 2026-09-30.

Versioned sibling of `scripts/build_engine_event_round2.py` (which is NOT edited and runs its
12 fits serially) and of `PO.fit_predict_scheme` (the offline S1 grader). One run of this
script does, for one fold:

  1. optionally overlays a sibling team-rate feature table on the round-2 design
     (`--feature-table`, same column names; possession_outcome's `x_<a>__<b>` interaction
     columns are recomputed from the overlaid inputs -- see `train_par_common_v1.overlay_columns`);
  2. fits every (population, monthly refit date) S1 model as its own single-threaded worker
     process (joblib/loky, `--n-jobs`), each with a per-task checkpoint (spot-reclaim resume),
     and predicts that refit's scored segment of the test season;
  3. scores the assembled S1 predictions per population with the round-1 grader `R1.score`
     (log loss, calibration, responsiveness quintiles: the same numbers the bake-off reports)
     and writes `pred_<pop>_<fold>.npy` + `par_v1_report.json`;
  4. for a fold whose engine games file exists (F2/2025) it ALSO produces the engine artifact
     directory by calling `build_engine_event_round2.build()` UNMODIFIED, with `PO.fit_arm`
     answered from the models of step 2 (a memo shim keyed on the exact fit inputs; a miss is a
     hard error, never a silent serial fit). So the artifacts are byte-for-byte what the serial
     builder would write, by construction, and `--mode identity` proves it on a tiny case.

WHAT THIS DOES NOT DO: it does not choose anything. Arms and feature sets are read from the
round-2 `verdict.json` (`--verdict`) exactly as the builder does; no bake-off decision is taken.

Paths are parameters:
  --design          default the round-2 design.parquet
  --feature-table   sibling table (parquet/csv) with the design's own column names, keyed on
                    `--overlay-keys`. The design is per-chance; the team-rate columns are per
                    (game, offense team), so the default keys are game_id,offense_team_id
  --engine-dir      where games_<fold>_<season>.parquet lives (default data/processed/models/engine)
  --out-root        NEW directory that receives the event artifact dir, the overlaid design copy,
                    predictions and the report (never an existing artifact directory)

Box usage (192 vCPU; 12 fits per fold, one task each):
    python scripts/train_possession_outcome_s1_par_v1.py --mode run --fold F2 --season 2025 \\
        --out-root data/processed/models/engine_s1_teamrate_v1 --feature-table <table> --n-jobs 24
    python scripts/train_possession_outcome_s1_par_v1.py --mode run --fold F1 --season 2024 \\
        --out-root data/processed/models/engine_s1_teamrate_v1_f1 --feature-table <table> --n-jobs 24
Resume after a spot reclaim: rerun the SAME command (finished fits are not recomputed).
Identity proof (tiny, local):
    python scripts/train_possession_outcome_s1_par_v1.py --mode identity
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
import train_par_common_v1 as C  # noqa: E402

C.pin_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_engine_event_round2 as B  # noqa: E402
import train_possession_outcome_v1 as R1  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402


def _t(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def fit_key(arm: str, seed: int, rows: pd.DataFrame, feats: list[str]) -> str:
    """Identity of one fit's inputs: enough to make the memo shim exact and a mismatch loud."""
    x = rows[feats].to_numpy(dtype="float64")
    return (f"{arm}|{seed}|{len(rows)}|{int(rows['y'].sum())}|"
            f"{np.nansum(x):.6e}|{np.nansum(x[:, 0] * np.arange(len(x)) % 977):.6e}")


def _worker(arm: str, seed: int, feats: list[str], rows: pd.DataFrame, te_seg: pd.DataFrame,
            meta: dict, n_estimators: int | None = None):
    """One (population, refit date) fit, single-threaded, in its own process.

    `n_estimators` is a TEST HOOK carried in the payload because a loky worker is a fresh
    interpreter: a module-level monkeypatch in the parent does not reach it."""
    C.pin_threads()
    # single-threaded fit by design (a fresh interpreter has the un-patched n_jobs=-1 default)
    PO.LgbmArm.PARAMS = dict(PO.LgbmArm.PARAMS, n_jobs=1)
    if n_estimators:
        PO.LgbmArm.PARAMS = dict(PO.LgbmArm.PARAMS, n_estimators=int(n_estimators))
    t = time.time()
    model = PO.fit_arm(arm, rows, feats, seed=seed)
    for b in (getattr(model, "clf_", None),):
        try:
            b.set_params(n_jobs=1)          # exactly what the builder does before dumping
        except Exception:                                                # noqa: BLE001
            pass
    pred = PO.predict_arm(arm, model, te_seg, feats).astype("float64") if len(te_seg) else None
    return {"model": model, "pred": pred, "key": fit_key(arm, seed, rows, feats),
            "fit_s": round(time.time() - t, 1), "meta": meta}


def build_tasks(design: pd.DataFrame, fold: str, season: int, win: dict, seed: int,
                max_cuts: int | None = None, n_estimators: int | None = None):
    """Same partition as `build_engine_event_round2.build` / `PO.fit_predict_scheme`."""
    train_seasons = B.FOLD_TRAIN[fold]
    d = design.copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    cuts = PO.month_boundaries(d.loc[d["season"] == season, "game_date"])
    if max_cuts:
        cuts = cuts[:max_cuts]
    tasks, layout = {}, {}
    for pop in ("first", "cont"):
        arm = win[pop]["arm"]
        feats = PO.feature_set(win[pop]["feature_set"], pop)
        tr = d[(d["season"].isin(train_seasons)) & (d["population"] == pop)]
        te = d[(d["season"] == season) & (d["population"] == pop)].reset_index(drop=True)
        te_dates = te["game_date"]
        fit_cols = [*feats, "y", "season"]
        for k, cut in enumerate(cuts):
            nxt = cuts[k + 1] if k + 1 < len(cuts) else None
            seg = ((te_dates >= cut) if nxt is None
                   else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
            before = (te_dates < cut).to_numpy()
            prior = te.loc[before, fit_cols]
            rows = tr[fit_cols] if not len(prior) else pd.concat(
                [tr[fit_cols], prior], ignore_index=True)
            key = f"{pop}_{cut.date()}"
            tasks[key] = dict(arm=arm, seed=seed, feats=feats, rows=rows,
                              te_seg=te.loc[seg, feats], n_estimators=n_estimators,
                              meta={"pop": pop, "cut": str(cut.date()),
                                    "n_scored": int(seg.sum())})
            layout[key] = seg
        layout[f"_te_{pop}"] = te
    return tasks, layout, cuts


def score_populations(results: dict, layout: dict, win: dict, cuts: list, fold: str):
    out, preds = {}, {}
    for pop in ("first", "cont"):
        te = layout[f"_te_{pop}"]
        p = np.zeros((len(te), len(PO.CLASSES)), dtype="float64")
        for cut in cuts:
            key = f"{pop}_{cut.date()}"
            seg = layout[key]
            if seg.any():
                p[seg] = results[key]["pred"]
        if (p.sum(axis=1) == 0).any():
            raise AssertionError("S1 left test rows unscored; the month partition is not a cover")
        s = R1.score(te, p)
        preds[pop] = p
        out[pop] = {"arm": win[pop]["arm"], "fold": fold, "n_test": int(len(te)),
                    "log_loss": round(float(s["log_loss"]), 6),
                    "calibration_pass": bool(s["calibration_pass"]),
                    "worst_gated_gap_pp": round(float(s["worst_gated_gap_pp"]), 3),
                    "responsiveness_pass": bool(s["responsiveness_pass"]),
                    "brier": {k: round(float(v), 6) for k, v in s["brier"].items()},
                    "fit_s_per_refit": {k.split("_", 1)[1]: results[k]["fit_s"]
                                        for k in results if k.startswith(pop)}}
    return out, preds


def run(a) -> dict:
    t0 = time.time()
    out_root = Path(a.out_root)
    if a.team_rate_table:
        out_root = out_root / Path(a.team_rate_table).stem   # artifacts name the table
    if a.team_rate_table and a.feature_table:
        raise SystemExit("--team-rate-table and --feature-table are mutually exclusive")
    C.assert_fresh_out_dir(out_root)
    C.stamp_owner(out_root, "train_possession_outcome_s1_par_v1", vars(a))
    for s in [*B.FOLD_TRAIN[a.fold], a.season]:
        assert_not_sealed(s)
    if a.season != B.FOLD_TEST[a.fold]:
        raise SystemExit(f"fold {a.fold} tests season {B.FOLD_TEST[a.fold]}, not {a.season}")

    B.R2_VERDICT = Path(a.verdict)
    win = B.winners_from_verdict()
    _t(f"winners: first={win['first']['arm']}/{win['first']['feature_set']}, "
       f"cont={win['cont']['arm']}/{win['cont']['feature_set']}", t0)

    design = pd.read_parquet(a.design)
    overlay_rep = None
    design_path = Path(a.design)
    if a.team_rate_table:
        try:
            from cbb_sim.team_rate_adapter import apply as team_rate_apply
        except ImportError as e:
            raise SystemExit("src/cbb_sim/team_rate_adapter.py is not on this checkout yet "
                             f"({e}); pull the commit that adds it") from e
        design = team_rate_apply(design, a.team_rate_table, "possession_outcome", fold=a.fold,
                                   missing=a.team_rate_missing)
        overlay_rep = {"team_rate_table": str(a.team_rate_table),
                       "table_sha256": C.sha256_file(Path(a.team_rate_table)),
                       "stem": Path(a.team_rate_table).stem}
        design_path = out_root / "design_overlay.parquet"
        design.to_parquet(design_path, index=False)
        _t(f"team-rate adapter applied: {overlay_rep['stem']}", t0)
    if a.feature_table:
        design, overlay_rep = C.overlay_columns(
            design, Path(a.feature_table), keys=a.overlay_keys.split(","),
            cols=a.overlay_cols.split(",") if a.overlay_cols else None)
        design_path = out_root / "design_overlay.parquet"
        design.to_parquet(design_path, index=False)
        _t(f"overlay applied: {overlay_rep['columns_overlaid']} "
           f"(+{len(overlay_rep['interactions_recomputed'])} interactions)", t0)
    if a.sample_games:
        gid = np.sort(design["game_id"].unique())[:: a.sample_games]
        design = design[design["game_id"].isin(gid)].reset_index(drop=True)
        design_path = out_root / "design_sample.parquet"
        design.to_parquet(design_path, index=False)
        _t(f"TEST HOOK: design cut to every {a.sample_games}th game ({len(design):,} rows)", t0)
    if a.test_n_estimators:
        PO.LgbmArm.PARAMS = dict(PO.LgbmArm.PARAMS, n_estimators=int(a.test_n_estimators))
        _t(f"TEST HOOK: n_estimators -> {a.test_n_estimators} (never on the box)", t0)

    tasks, layout, cuts = build_tasks(design, a.fold, a.season, win, a.seed, a.max_cuts,
                                      int(a.test_n_estimators) or None)
    _t(f"{len(tasks)} fit tasks over {len(cuts)} refit dates x 2 populations", t0)
    results = C.run_tasks(tasks, _worker, a.n_jobs, out_root / "cuts",
                          stop_at=pd.Timestamp(a.stop_at) if a.stop_at else None,
                          log=lambda m: _t(m, t0))
    if len(results) < len(tasks):
        _t(f"INCOMPLETE {len(results)}/{len(tasks)}; rerun the same command to resume", t0)
        return {"complete": False}
    del tasks

    scores, preds = score_populations(results, layout, win, cuts, a.fold)
    for pop, p in preds.items():
        np.save(out_root / f"pred_{pop}_{a.fold}.npy", p.astype("float32"))
    for pop, s in scores.items():
        _t(f"{a.fold} {pop}: logloss {s['log_loss']} cal={'P' if s['calibration_pass'] else 'F'}"
           f"({s['worst_gated_gap_pp']}pp) resp={'P' if s['responsiveness_pass'] else 'F'}", t0)

    # ---- engine artifact directory through the UNMODIFIED builder ----------------------
    built = None
    games_src = Path(a.engine_dir) / f"games_{a.fold}_{a.season}.parquet"
    if not a.no_artifacts and games_src.exists() and not a.max_cuts:
        games_dst = out_root / games_src.name
        if not games_dst.exists():
            shutil.copyfile(games_src, games_dst)
        memo = {r["key"]: r["model"] for r in results.values()}
        real_fit = PO.fit_arm

        def _memo_fit(arm, tr, features, seed=0, sample_weight=None):
            k = fit_key(arm, seed, tr, features)
            if sample_weight is not None or k not in memo:
                raise AssertionError(f"builder asked for a fit the pool did not run ({k}); "
                                     "refusing to fit serially")
            return memo[k]

        B.ENGINE_DIR = out_root
        B.R2_DESIGN = design_path
        PO.fit_arm = _memo_fit
        try:
            built = B.build(a.fold, a.season, a.seed)
        finally:
            PO.fit_arm = real_fit
        _t(f"engine artifact dir: {built}", t0)
    elif not a.no_artifacts:
        _t(f"no engine artifact dir written (no {games_src.name} or max-cuts hook)", t0)

    rep = {"fold": a.fold, "season": a.season, "seed": a.seed, "scheme": "S1",
           "n_jobs": a.n_jobs, "overlay": overlay_rep, "design": str(design_path),
           "scores": scores, "artifact_dir": str(built) if built else None,
           "wall_s": round(time.time() - t0, 1), "test_hooks": {
               "sample_games": a.sample_games, "test_n_estimators": a.test_n_estimators,
               "max_cuts": a.max_cuts}}
    (out_root / "par_v1_report.json").write_text(json.dumps(rep, indent=1, default=str),
                                                 encoding="utf-8")
    return {"complete": True, "report": rep, "results": results, "built": built}


def identity(a) -> int:
    """Serial `build_engine_event_round2.build` vs this wrapper (n_jobs=2, real loky), same tiny
    design (every Nth game, reduced trees -- TEST ONLY). Artifacts must be bit-identical."""
    import tempfile
    scratch = Path(a.scratch or tempfile.mkdtemp(prefix="po_par_identity_"))
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"identity scratch: {scratch}", flush=True)
    full = pd.read_parquet(a.design)
    gid = np.sort(full["game_id"].unique())[:: a.sample_games or 60]
    small = full[full["game_id"].isin(gid)].reset_index(drop=True)
    small_p = scratch / "design_small.parquet"
    small.to_parquet(small_p, index=False)
    n_est = int(a.test_n_estimators or 15)
    PO.LgbmArm.PARAMS = dict(PO.LgbmArm.PARAMS, n_estimators=n_est, n_jobs=1)
    print(f"tiny design {len(small):,} rows; n_estimators={n_est} (test only)", flush=True)

    # (a) the ORIGINAL serial builder, pointed at scratch
    sdir = scratch / "serial"
    sdir.mkdir(exist_ok=True)
    shutil.copyfile(Path(a.engine_dir) / f"games_{a.fold}_{a.season}.parquet",
                    sdir / f"games_{a.fold}_{a.season}.parquet")
    B.ENGINE_DIR, B.R2_DESIGN, B.R2_VERDICT = sdir, small_p, Path(a.verdict)
    t = time.time()
    ser = B.build(a.fold, a.season, a.seed)
    t_ser = time.time() - t

    # (b) the wrapper, real 2-process loky
    ns = argparse.Namespace(**{**vars(a), "design": str(small_p), "out_root": str(scratch / "par"),
                               "n_jobs": 2, "feature_table": None, "team_rate_table": "", "sample_games": 0,
                               "test_n_estimators": n_est, "max_cuts": None, "stop_at": "",
                               "no_artifacts": False})
    t = time.time()
    r = run(ns)
    t_par = time.time() - t
    par = Path(r["built"])

    ok = True
    for f in sorted(p.name for p in ser.iterdir()):
        if f.endswith(".joblib"):
            import joblib
            ma, mb = joblib.load(ser / f)["model"], joblib.load(par / f)["model"]
            if hasattr(ma, "clf_") and hasattr(mb, "clf_"):
                same = ma.clf_.booster_.model_to_string() == mb.clf_.booster_.model_to_string()
            else:                                    # cascade arm: identical predictions
                same = _same_pickled_predictions(ser / f, par / f, small)
        elif f == "team_block.npz":
            a1, b1 = np.load(ser / f), np.load(par / f)
            same = all(np.array_equal(a1[k], b1[k]) for k in a1.files)
        else:                                                       # index.json
            ia, ib = json.loads((ser / f).read_text()), json.loads((par / f).read_text())
            for x in (ia, ib):
                for pop in x["populations"].values():
                    for s in pop["segments"]:
                        s.pop("size_mb", None)
            same = ia == ib
        ok &= bool(same)
        print(f"  {f:34s} {'IDENTICAL' if same else 'DIFFERENT'}", flush=True)
    names_par = sorted(p.name for p in par.iterdir())
    ok &= names_par == sorted(p.name for p in ser.iterdir())
    print(f"serial builder {t_ser:.1f}s; wrapper (2 procs, incl. scoring) {t_par:.1f}s; "
          f"scores: {json.dumps({k: v['log_loss'] for k, v in r['report']['scores'].items()})}",
          flush=True)
    print(f"IDENTITY (serial builder vs par wrapper, all artifacts): {'PASS' if ok else 'FAIL'}",
          flush=True)
    return 0 if ok else 1


def _same_pickled_predictions(pa: Path, pb: Path, design: pd.DataFrame) -> bool:
    import joblib
    A, Bm = joblib.load(pa), joblib.load(pb)
    x = design.head(2000)
    feats = A["features"]
    return bool(np.array_equal(PO.predict_arm(A["arm"], A["model"], x, feats),
                               PO.predict_arm(Bm["arm"], Bm["model"], x, feats)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "identity"], required=True)
    ap.add_argument("--fold", default="F2", choices=sorted(B.FOLD_TRAIN))
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--design", default=str(B.R2_DESIGN))
    ap.add_argument("--verdict", default=str(B.R2_VERDICT))
    ap.add_argument("--engine-dir", default=str(B.ENGINE_DIR))
    ap.add_argument("--out-root", default="")
    ap.add_argument("--team-rate-table", default="",
                    help="team_rate_features_*.parquet; applied through cbb_sim.team_rate_adapter."
                         "apply(frame, path, 'possession_outcome'); artifacts go to <out-root>/<stem>/."
                         " Absent = served features, output identical to the serial builder")
    ap.add_argument("--team-rate-missing", choices=["raise", "keep_served"], default="raise",
                    help="adapter policy for design rows with no table key (PM decision; default raise)")
    ap.add_argument("--feature-table", default="")
    ap.add_argument("--overlay-keys", default="game_id,offense_team_id")
    ap.add_argument("--overlay-cols", default="",
                    help="default: every non-key column of the table that exists in the design")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--stop-at", default="", help="ISO timestamp; local time")
    ap.add_argument("--no-artifacts", action="store_true")
    # test hooks (identity / smoke only)
    ap.add_argument("--sample-games", type=int, default=0)
    ap.add_argument("--test-n-estimators", type=int, default=0)
    ap.add_argument("--max-cuts", type=int, default=0)
    ap.add_argument("--scratch", default="")
    a = ap.parse_args()
    if a.mode == "identity":
        return identity(a)
    if not a.out_root:
        raise SystemExit("--out-root is required (a NEW directory)")
    r = run(a)
    return 0 if r.get("complete") else 3


if __name__ == "__main__":
    raise SystemExit(main())
