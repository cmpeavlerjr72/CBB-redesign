"""Box-parallel wrapper around `train_rebound_v3_round3.py` (lane J, 2026-09-30).

`train_rebound_v3_round3.py` is UNTOUCHED. This wrapper reimplements only the S1_weekly
refit loop (stage 2, ~23 cuts, ~3 h per cell serial) so every cut's fit+predict is one
process-pool task (`n_jobs=1` LightGBM per fit, joblib/loky ACROSS cuts), with a per-cut
checkpoint in `<out-dir>/cuts/<tag>/` (spot-reclaim resume: rerun the identical command).
Row selection, `cols`, seed, `R3.fit_arm`, `R3.predict_arm`, `R3.predict_with_feed`,
`R3.grade` and the cell JSON are R3's own.

Parameters instead of hard-coded paths: `--design`, `--feature-table` (+ `--overlay-keys`,
`--overlay-cols`), `--out-dir`.

!! OVERLAY CAVEAT (read before using --feature-table) !!
Only `--overlay-cols` (default off_oreb_c, opp_def_dreb_c) are replaced. The design's
DERIVED columns off_oreb_g1/g2/g3, off_oreb_oa1_c/oa2_c, opp_def_dreb_g1/g2/g3,
opp_def_dreb_oa1_c/oa2_c, off_w/def_w/off_priorc/... are NOT recomputed. Arms C1, C2, C3,
D1, D2 (and combos with them, e.g. A5+C1) read those columns and are INVALID on an overlaid
table. Valid on an overlaid table: A0B0C0, A1, A2_*, A3_*, A4, A5, B1, D3. The run JSON
records this in `par_v1.overlay_invalid_arms`; the wrapper refuses an invalid arm when an
overlay is given.

`--feeds` calls `R3.block_feed`, which imports `train_shot_block_v1`, builds the shot_block
design (needs that trainer's inputs, as `train_shot_block_v1.build_design` reads them) and
fits one K2 model per fold in the main process. Run without --feeds if those inputs are not
on the box.

Box usage (one process per cell, ~24 cut tasks each):
  python scripts/train_rebound_v3_par_v1.py --stage 2 --folds F2 --arms A0B0C0 --n-jobs 24 \
      --out-dir data/processed/models/rebound/round3_par_<tag> [--feeds] \
      [--feature-table PATH --overlay-keys game_id,off_team_id]
  (rerun the identical command after a reclaim; finished cuts are skipped)
  Several arms in one process run sequentially, all cuts of a cell in the pool.

Identity proof (tiny): --mode identity  (see run_identity).
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(_ROOT / "scripts"), str(_ROOT / "src"), os.environ.get("PYTHONPATH", "")])
import train_par_common_v1 as C  # noqa: E402
C.pin_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import train_rebound_v3_round3 as R3  # noqa: E402
from cbb_sim.features import conference as CF  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402

OVERLAY_INVALID_ARMS = {"C1", "C2", "C3", "D1", "D2"}


def _worker(spec, feats, seed, pool, sub, feed_sub, refit, n_scored, params_override):
    """One refit cut, in a separate process."""
    C.pin_threads()
    if params_override:
        R3.PARAMS = {**R3.PARAMS, **params_override}
    R3.PARAMS = {**R3.PARAMS, "n_jobs": 1}
    clf, m = R3.fit_arm(spec, pool, feats, seed)
    p = R3.predict_arm(spec, clf, sub, feats)
    pf = {k: R3.predict_with_feed(spec, clf, sub, feats, v) for k, v in feed_sub.items()}
    return {"p": p, "pf": pf, "seg": {"refit": refit, "n_scored": n_scored, **m}}


def fit_predict_scheme_par(spec, tr, te, feats, cuts, seed, feeds, n_jobs, ckpt_dir,
                           stop_at=None, params_override=None):
    """Same contract as `R3.fit_predict_scheme`; returns None if stopped before complete."""
    if cuts is None:
        return R3.fit_predict_scheme(spec, tr, te, feats, cuts, seed, feeds)
    n = len(te)
    te_dates = pd.to_datetime(te["game_date"])
    cols = list(dict.fromkeys(feats + ["y", "season", "game_date", "_a5_off"]))
    cols = [c for c in cols if c in te.columns]
    tasks, sels = {}, {}
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        sel = ((te_dates >= cut) if nxt is None
               else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if not sel.any():
            continue
        before = (te_dates < cut).to_numpy()
        prior = te.loc[before, cols]
        pool = tr[cols] if not len(prior) else pd.concat([tr[cols], prior], ignore_index=True)
        key = f"{k:03d}_{pd.Timestamp(cut).date()}"
        sels[key] = sel
        feed_sub = {kk: (None if v is None else np.asarray(v)[sel]) for kk, v in feeds.items()}
        tasks[key] = dict(spec=spec, feats=feats, seed=seed, pool=pool, sub=te.loc[sel],
                          feed_sub=feed_sub, refit=str(pd.Timestamp(cut).date()),
                          n_scored=int(sel.sum()), params_override=params_override)
    done = C.run_tasks(tasks, _worker, n_jobs, ckpt_dir, stop_at=stop_at)
    if set(done) != set(tasks):
        return None
    p_true = np.zeros((n, 3), dtype="float64")
    p_feed = {k: np.zeros((n, 3), dtype="float64") for k in feeds}
    segs = []
    for key in sorted(tasks):
        r = done[key]
        p_true[sels[key]] = r["p"]
        for kk in feeds:
            p_feed[kk][sels[key]] = r["pf"][kk]
        segs.append(r["seg"])
    if (p_true.sum(axis=1) == 0).any():
        raise AssertionError("refit calendar is not a cover of the test slice")
    return p_true, p_feed, segs


def load_design(path, feature_table, keys, cols):
    d = pd.read_parquet(path)
    rep = None
    if feature_table:
        d, rep = C.overlay_columns(d, Path(feature_table), keys, cols,
                                   recompute_interactions=False)
    return d, rep


def _limit(te, cuts, max_cuts):
    if max_cuts and max_cuts < len(cuts):
        te = te[pd.to_datetime(te["game_date"]) < cuts[max_cuts]]
        cuts = cuts[:max_cuts]
    return te, cuts


def run_identity(a):
    """(a) R3 serial loop vs (b) wrapper via real loky (n_jobs=2), 2 cuts, downsampled
    train, reduced n_estimators (test-only patch); plus overlay identity."""
    t0 = time.time()
    d = pd.read_parquet(R3.DESIGN)
    ovl_cols = ["off_oreb_c", "opp_def_dreb_c"]
    tab = Path(a.out_dir) / "identity_table.parquet"
    keys = ["game_id", "off_team_id"]
    nun = d.groupby(keys)[ovl_cols].nunique(dropna=False).max().max()
    d[keys + ovl_cols].drop_duplicates(keys).to_parquet(tab, index=False)
    d2, rep = C.overlay_columns(d, tab, keys, ovl_cols, recompute_interactions=False)
    same = all(np.array_equal(d[c].to_numpy(), d2[c].to_numpy(), equal_nan=True)
               if d[c].dtype.kind in "fc" else d[c].equals(d2[c]) for c in d.columns)
    print(f"OVERLAY IDENTITY: design unchanged={same} (max distinct values per key {nun}, "
          f"unmatched {rep['n_unmatched_rows']}) [{time.time()-t0:.0f}s]", flush=True)
    del d2
    over = {"n_estimators": a.test_n_estimators}
    R3.PARAMS = {**R3.PARAMS, **over, "n_jobs": 1}
    tr0, te0 = RB.fold_slices(d, "F2")
    tr, te, _ = R3.add_fold_columns(tr0, te0)
    tr = tr.iloc[::a.test_train_stride]
    spec = R3.arm_spec("A0B0C0")
    feats = R3.features_for(spec)
    cuts = R3.s1_weekly_cuts(pd.to_datetime(te["game_date"]))
    te, cuts = _limit(te, cuts, a.max_cuts or 3)
    feeds = {"B0_zero": np.zeros(len(te)),
             "B2_asof_cell": te["blk_asof_cell"].to_numpy(dtype="float64"),
             "B_true": te["blocked_f"].to_numpy(dtype="float64")}
    print(f"identity: train {len(tr)} rows (stride {a.test_train_stride}), test {len(te)}, "
          f"cuts {[str(pd.Timestamp(c).date()) for c in cuts]}, n_estimators "
          f"{a.test_n_estimators}", flush=True)
    t1 = time.time()
    ps, pfs, ss = R3.fit_predict_scheme(spec, tr, te, feats, cuts, 0, feeds)
    ts = time.time() - t1
    ck = Path(a.out_dir) / "identity_cuts"
    if ck.exists():
        for f in ck.glob("*"):
            f.unlink()
    t1 = time.time()
    pp, pfp, sp = fit_predict_scheme_par(spec, tr, te, feats, cuts, 0, feeds, 2, ck,
                                         params_override=over)
    tp = time.time() - t1
    exact = np.array_equal(ps, pp)
    feeds_ok = all(np.array_equal(pfs[k], pfp[k]) for k in feeds)
    seg_ok = ss == sp
    print(f"  serial {ts:.1f}s  parallel(2 procs) {tp:.1f}s")
    print(f"  p_true bit-identical: {exact}  max abs diff {np.max(np.abs(ps-pp)):.3e}")
    print(f"  feeds bit-identical: {feeds_ok}  segments identical: {seg_ok}")
    ok = exact and feeds_ok and seg_ok and same
    print(f"IDENTITY CHECK: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "identity"], default="run")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--arms", default="")
    ap.add_argument("--folds", default="F2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feeds", action="store_true")
    ap.add_argument("--design", default=str(R3.DESIGN))
    ap.add_argument("--team-rate-missing", choices=["raise", "keep_served"], default="raise",
                    help="adapter policy for design rows with no table key (PM decision; default raise)")
    ap.add_argument("--feature-table", default="")
    ap.add_argument("--overlay-keys", default="game_id,off_team_id")
    ap.add_argument("--overlay-cols", default="off_oreb_c,opp_def_dreb_c")
    ap.add_argument("--team-rate-table", default="",
                    help="optional; applied via cbb_sim.team_rate_adapter.apply(d, path, "
                         "'rebound'); exclusive with --feature-table; outputs go under "
                         "<out-dir>/<table stem>/")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--stop-at", default="")
    ap.add_argument("--max-cuts", type=int, default=0, help="TEST HOOK")
    ap.add_argument("--test-n-estimators", type=int, default=20, help="identity only")
    ap.add_argument("--test-train-stride", type=int, default=100, help="identity only")
    a = ap.parse_args()

    out = Path(a.out_dir).resolve()
    if out == Path(R3.OUT_DIR).resolve():
        raise SystemExit("--out-dir must differ from round3's own OUT_DIR (never overwrite)")
    C.assert_fresh_out_dir(out)
    C.stamp_owner(out, "train_rebound_v3_par_v1", vars(a))
    R3.DESIGN = Path(a.design)
    R3.OUT_DIR = out
    if a.team_rate_table and a.feature_table:
        raise SystemExit("--team-rate-table and --feature-table are mutually exclusive")
    trt_stem = Path(a.team_rate_table).stem if a.team_rate_table else ""
    base = out / trt_stem if trt_stem else out
    R3.CELLS = base / "cells"
    R3.CELLS.mkdir(parents=True, exist_ok=True)
    if a.mode == "identity":
        raise SystemExit(0 if run_identity(a) else 1)

    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed(R3.SEASONS, context="rebound round 3 par")
    t0 = time.time()
    keys = [k for k in a.overlay_keys.split(",") if k]
    ocols = [k for k in a.overlay_cols.split(",") if k]
    d, rep = load_design(R3.DESIGN, a.feature_table, keys, ocols)
    trt = None
    if a.team_rate_table:
        try:
            from cbb_sim.team_rate_adapter import apply as _apply_trt
        except ImportError as e:
            raise SystemExit(f"--team-rate-table needs cbb_sim.team_rate_adapter, not "
                             f"importable ({e}); it is committed by another worker")
        if "," in a.folds:
            raise SystemExit("--team-rate-table applies one fold's rows: pass a single --folds value")
        d = _apply_trt(d, a.team_rate_table, "rebound", fold=a.folds,
                     missing=a.team_rate_missing)
        trt = {"path": str(a.team_rate_table), "stem": trt_stem,
               "sha256": C.sha256_file(Path(a.team_rate_table))}
    conf_all = CF.build_conference_flags(R3.SEASONS)
    firsts = CF.first_conference_game_dates(conf_all)
    arms = [x for x in (a.arms.split(",") if a.arms else list(R3.ARMS)) if x]
    if rep or trt:
        bad = [x for x in arms if set(x.split(R3.COMBO_SEP)) & OVERLAY_INVALID_ARMS]
        if bad:
            raise SystemExit(f"arms {bad} read derived columns the overlay does not "
                             "recompute; invalid on an overlaid table")
    stop_at = None
    if a.stop_at:
        hh, mm = a.stop_at.split(":")
        stop_at = pd.Timestamp.now().normalize() + pd.Timedelta(hours=int(hh), minutes=int(mm))
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        git = "unknown"
    print(f"[{time.time()-t0:.0f}s] design {d.shape}; stage {a.stage}; arms {arms}; "
          f"folds {a.folds}; n_jobs {a.n_jobs}", flush=True)
    for fold in a.folds.split(","):
        tr0, te0 = RB.fold_slices(d, fold)
        tr, te, fmeta = R3.add_fold_columns(tr0, te0)
        cuts = None if a.stage == 1 else R3.s1_weekly_cuts(pd.to_datetime(te["game_date"]))
        if cuts is not None:
            te, cuts = _limit(te, cuts, a.max_cuts)
        feeds = {"B0_zero": np.zeros(len(te)),
                 "B2_asof_cell": te["blk_asof_cell"].to_numpy(dtype="float64")}
        if a.feeds:
            feeds["B3e_model"] = R3.block_feed(te, fold, None)
            rng = np.random.default_rng(20260918)
            feeds["B3_draw"] = (rng.random(len(te)) < feeds["B3e_model"]).astype("float64")
        feeds["B_true"] = te["blocked_f"].to_numpy(dtype="float64")
        for arm in arms:
            tag = f"s{a.stage}_{fold}_{arm}_seed{a.seed}"
            out_p = R3.CELLS / f"{tag}.json"
            if out_p.exists():
                print(f"SKIP {tag} (exists)", flush=True)
                continue
            spec = R3.arm_spec(arm)
            feats = R3.features_for(spec)
            t1 = time.time()
            res = fit_predict_scheme_par(spec, tr, te, feats, cuts, a.seed, feeds, a.n_jobs,
                                         base / "cuts" / tag, stop_at=stop_at)
            if res is None:
                print(f"{tag}: NOT COMPLETE (stop-at); rerun the same command to resume",
                      flush=True)
                continue
            p_true, p_feed, segs = res
            g = R3.grade(te, p_true, p_feed, firsts)
            g.update({"arm": arm, "fold": fold, "stage": a.stage, "seed": a.seed,
                      "scheme": "S0" if a.stage == 1 else "S1_weekly",
                      "features": feats, "n_features": len(feats),
                      "why": spec.get("why", ""), "n_fits": len(segs),
                      "fit_seconds": round(time.time() - t1, 1), "fold_meta": fmeta,
                      "created_at": pd.Timestamp.now("UTC").isoformat(),
                      "par_v1": {"n_jobs": a.n_jobs, "git": git, "design": str(R3.DESIGN),
                                 "overlay": rep, "team_rate_table": trt, "max_cuts_test_hook": a.max_cuts,
                                 "overlay_invalid_arms": sorted(OVERLAY_INVALID_ARMS)}})
            out_p.write_text(json.dumps(g, indent=1, default=str), encoding="utf-8")
            print(f"[{time.time()-t0:.0f}s] {tag}: ll={g['log_loss']} {g['fit_seconds']}s",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
