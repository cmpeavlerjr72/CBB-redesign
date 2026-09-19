"""Versioned sibling of `train_possession_outcome_v4.py`: EXECUTION-SCHEME-ONLY change
for the two alignment cells (`A1`, `A2`, Decision 9c) that failed to run three times
(local wall-clock twice, AWS once -- `docs/models/possession_outcome/experiments.md`
section 14). Pre-registration: section 16 of that file, committed BEFORE this run.

What changes vs the serial trainer, and nothing else
-----------------------------------------------------
* `R3.fit_predict_walkforward`'s per-refit-date loop is replaced by
  `fit_predict_walkforward_joblib`, which computes the same segments (same
  strictly-before rule, same boundaries) in the main process and dispatches each
  cut's fit+predict to a separate WORKER PROCESS via `joblib.Parallel(backend="loky")`,
  capped at 6 concurrent workers.
* Every worker is pinned to a single thread and `PO.LgbmArm.PARAMS["n_jobs"]` is
  patched to 1 at runtime (a monkeypatch from THIS script, `src/cbb_sim/` is never
  edited on disk) so a fit never competes with the process-level parallelism.
* Every other input -- the arm, the feature set, the scheme's refit calendar, the
  seed, `LgbmArm.PARAMS` values -- is untouched, imported from `train_possession_outcome_v3`
  / `train_possession_outcome_v4` / `cbb_sim.models.possession_outcome` unmodified.
* Checkpointing is PER REFIT DATE: each cut's prediction segment + fit metadata is
  written to `<OUT_DIR>/cuts/<cell_key>/` as soon as it is computed, so a killed run
  resumes by re-reading that directory and only dispatching the still-missing dates.

Usage
-----
Identity check (serial vs parallel must match exactly on 2 real refit dates, fold F1,
chosen only for speed -- the code path is fold/scheme-agnostic):
    .venv/Scripts/python.exe scripts/train_possession_outcome_v4_par.py --mode identity

Run a cell (A1 or A2) to completion or until --stop-at:
    .venv/Scripts/python.exe scripts/train_possession_outcome_v4_par.py --mode run \
        --stage 7 --stop-at 21:05 --n-jobs 6
    .venv/Scripts/python.exe scripts/train_possession_outcome_v4_par.py --mode run \
        --stage 8 --stop-at 21:05 --n-jobs 6

Merge whatever cells are checkpointed into a report through v4's own grading/decision
code (round4 + round4b + this round's checkpoint), read-only on round4/round4b:
    .venv/Scripts/python.exe scripts/train_possession_outcome_v4_par.py --mode merge
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import train_possession_outcome_v1 as R1  # noqa: E402
import train_possession_outcome_v3 as R3  # noqa: E402
import train_possession_outcome_v4 as R4  # noqa: E402

from cbb_sim.features import conference as CF  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

def _safe_dirname(key: str) -> str:
    """`cell_key` uses `|` as a separator, which Windows rejects in path components."""
    return key.replace("|", "_")


R4_DIR = _ROOT / "data/processed/models/possession_outcome/round4"
R4B_DIR = _ROOT / "data/processed/models/possession_outcome/round4b"
OUT_DIR = _ROOT / "data/processed/models/possession_outcome/round4_a1a2"
CUTS_DIR = OUT_DIR / "cuts"
SEASONS = R4.SEASONS


# ---------------------------------------------------------------------------
# 0. Worker-side pinning: process-local, never touches src/cbb_sim/ on disk
# ---------------------------------------------------------------------------
def _pin_single_thread() -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
        os.environ[v] = "1"
    if PO.LgbmArm.PARAMS.get("n_jobs") != 1:
        PO.LgbmArm.PARAMS = dict(PO.LgbmArm.PARAMS, n_jobs=1)


def _fit_cut_worker(key: str, arm: str, seed: int, features: list[str],
                     tr_fc: pd.DataFrame, prior_test: pd.DataFrame, te_pred: pd.DataFrame,
                     refit_date: str, max_train_date: str, n_scored: int,
                     n_train_from_test: int):
    """Runs in a SEPARATE PROCESS (joblib/loky). One refit date, start to finish."""
    _pin_single_thread()
    fit_rows = tr_fc if not len(prior_test) else pd.concat([tr_fc, prior_test], ignore_index=True)
    model = PO.fit_arm(arm, fit_rows, features, seed=seed)
    pred = PO.predict_arm(arm, model, te_pred, features)
    meta = {"refit_date": refit_date, "n_train": int(len(fit_rows)),
            "n_train_from_test_season": n_train_from_test,
            "n_scored": n_scored, "max_train_date": max_train_date}
    return key, pred.astype("float64"), meta


# ---------------------------------------------------------------------------
# 1. Orchestrator: prep in the main process (cheap), fit in workers (expensive),
#    checkpoint PER REFIT DATE so a kill loses at most one in-flight fit per worker.
# ---------------------------------------------------------------------------
def fit_predict_walkforward_joblib(arm: str, tr: pd.DataFrame, te: pd.DataFrame,
                                    features: list[str], cuts: list, seed: int,
                                    n_jobs: int, ckpt_dir: Path, cell_key: str,
                                    stop_at: pd.Timestamp | None = None,
                                    log: bool = True) -> tuple[np.ndarray | None, dict | None, dict]:
    """Same contract as `R3.fit_predict_walkforward` (returns `(p, meta)`), plus a third
    element, `status`, describing what ran. If `stop_at` is hit before every cut for this
    cell is fit, `(None, None, status)` is returned and the cell is NOT graded -- the
    per-date checkpoint on disk is untouched and complete for whatever finished."""
    assert n_jobs <= 6, "compute cap: at most 6 worker processes at any time"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    idx_path = ckpt_dir / "done.json"
    done: dict = json.loads(idx_path.read_text()) if idx_path.exists() else {}

    te = te.reset_index(drop=True)
    te_dates = pd.to_datetime(te["game_date"])
    tr_dates = pd.to_datetime(tr["game_date"])
    fit_cols = [*features, "y", "season"]
    tr_fc = tr[fit_cols]
    te_fc = te[fit_cols]
    te_feat = te[features]

    prepped: dict[str, np.ndarray] = {}
    tasks = []
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        seg = ((te_dates >= cut) if nxt is None else
               ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
        if not seg.any():
            continue
        key = str(k)
        prepped[key] = seg
        if key in done:
            continue
        before = (te_dates < cut).to_numpy()
        prior_test = te_fc.loc[before]
        max_train = (tr_dates.max() if not before.any()
                     else max(tr_dates.max(), te_dates[before].max()))
        tasks.append(dict(key=key, prior_test=prior_test, te_pred=te_feat.loc[seg],
                           refit_date=str(pd.Timestamp(cut).date()),
                           max_train_date=str(pd.Timestamp(max_train).date()),
                           n_scored=int(seg.sum()), n_train_from_test=int(len(prior_test))))

    if log:
        print(f"[{cell_key}] {len(prepped)} refit dates total, {len(done)} already "
              f"checkpointed, {len(tasks)} to run now, n_jobs={n_jobs}", flush=True)

    stopped_early = False
    if tasks:
        from joblib import Parallel, delayed
        gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(_fit_cut_worker)(t["key"], arm, seed, features, tr_fc, t["prior_test"],
                                      t["te_pred"], t["refit_date"], t["max_train_date"],
                                      t["n_scored"], t["n_train_from_test"])
            for t in tasks)
        n_done_now = 0
        t_wave0 = time.time()
        for key, pred, meta in gen:
            np.save(ckpt_dir / f"pred_{key}.npy", pred)
            done[key] = meta
            tmp = idx_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(done, indent=1, default=str))
            os.replace(tmp, idx_path)
            n_done_now += 1
            elapsed = time.time() - t_wave0
            rate = elapsed / n_done_now
            remaining = len(tasks) - n_done_now
            if log:
                print(f"  [{cell_key}] cut {meta['refit_date']} done "
                      f"({len(done)}/{len(prepped)})  n_train={meta['n_train']}  "
                      f"n_scored={meta['n_scored']}  [{elapsed/n_done_now:.0f}s/date avg, "
                      f"{remaining} left, ~{remaining*rate/n_jobs/60:.0f} min more at "
                      f"{n_jobs}-way]", flush=True)
            if stop_at is not None and pd.Timestamp.now() >= stop_at:
                stopped_early = True
                print(f"  [{cell_key}] STOP-AT {stop_at} reached with {len(done)}/"
                      f"{len(prepped)} dates checkpointed; leaving remaining in-flight "
                      f"workers to finish their current fit, then exiting.", flush=True)
                break

    status = {"n_prepped": len(prepped), "n_done": len(done),
              "complete": set(prepped) <= set(done), "stopped_early": stopped_early}
    if not status["complete"]:
        return None, None, status

    p = np.zeros((len(te), len(PO.CLASSES)), dtype="float64")
    segments = []
    for k in range(len(cuts)):
        key = str(k)
        if key not in prepped:
            continue
        p[prepped[key]] = np.load(ckpt_dir / f"pred_{key}.npy")
        segments.append(done[key])
    if (p.sum(axis=1) == 0).any():
        raise AssertionError("the refit calendar is not a cover of the test slice")
    meta = {"n_fits": len(segments), "segments": segments,
            "earliest_train_date": str(tr_dates.min().date()) if len(tr_dates) else None}
    return p, meta, status


# ---------------------------------------------------------------------------
# 2. Identity check: serial R3 loop vs the joblib path, 2 real refit dates, fold F1
#    (smaller, faster -- the function under test is fold/scheme-agnostic)
# ---------------------------------------------------------------------------
def run_identity_check(design: pd.DataFrame, conf: pd.DataFrame, sample_n: int = 4_000) -> bool:
    """Proves the REFACTOR (serial for-loop vs joblib-across-dates dispatch) preserves
    exact fit inputs/outputs on 2 real refit dates. Deliberately run on a random
    downsample of the real fold-F1 `first` rows (same real design columns, same real
    `PO.fit_arm`/`PO.predict_arm` code, same `LgbmArm.PARAMS`, same seed) rather than the
    full ~1.9M-row slice: this is a code-path-parity proof, not a scored cell, and full
    size would cost 10+ minutes per fit under the single-thread pin this round requires.
    Row count cannot change WHETHER two orchestrations agree on the same input, so this is
    not a weaker check -- it is a faster one. A1/A2 themselves run on the FULL slice."""
    _pin_single_thread()
    # DEVIATION (time-boxed session, documented in results): a real single-threaded fit
    # at the pre-registered n_estimators=400 on this loaded machine did not finish a
    # 40,000-row slice inside 5.5 minutes (killed, no result). This check exists to prove
    # ARCHITECTURE parity (does the joblib dispatch reconstruct the same segments/inputs as
    # the serial loop?), which does not depend on tree count, so `n_estimators` is cut to
    # 20 for THIS CHECK ONLY via the same runtime monkeypatch mechanism used for `n_jobs`.
    # A1/A2 themselves run with the full pre-registered `n_estimators=400`.
    PO.LgbmArm.PARAMS = dict(PO.LgbmArm.PARAMS, n_estimators=20)
    rng = np.random.RandomState(0)
    tr_full, te = PO.fold_slices(design, "F1", "first")
    tr = tr_full if len(tr_full) <= sample_n else tr_full.sample(sample_n, random_state=rng)
    feats = R4.feature_set_v4("G0", "first")
    all_cuts = R3.refit_dates("S1_monthly", te, conf)
    cuts = all_cuts[:2]
    print(f"identity check: fold F1, first, G0, S1_monthly, first 2 of "
          f"{len(all_cuts)} cuts: {[str(pd.Timestamp(c).date()) for c in cuts]}, "
          f"tr downsampled {len(tr_full)} -> {len(tr)} rows, n_estimators cut 400->20, "
          f"for speed", flush=True)

    t0 = time.time()
    p_serial, meta_serial = R3.fit_predict_walkforward("lgbm", tr, te, feats, cuts, seed=0)
    print(f"  serial path:   {time.time()-t0:.1f}s, n_fits={meta_serial['n_fits']}", flush=True)

    ck = CUTS_DIR / "_identity_check"
    if ck.exists():
        for f_ in ck.glob("*"):
            f_.unlink()
    t0 = time.time()
    p_par, meta_par, status = fit_predict_walkforward_joblib(
        "lgbm", tr, te, feats, cuts, seed=0, n_jobs=2, ckpt_dir=ck, cell_key="identity_check")
    print(f"  parallel path: {time.time()-t0:.1f}s, n_fits={meta_par['n_fits'] if meta_par else None}, "
          f"status={status}", flush=True)

    assert p_par is not None, "identity check did not complete"
    exact = np.array_equal(p_serial, p_par)
    max_abs_diff = float(np.max(np.abs(p_serial - p_par)))
    same_n_fits = meta_serial["n_fits"] == meta_par["n_fits"]
    same_segments = meta_serial["segments"] == meta_par["segments"]
    print(f"  predictions bit-identical (np.array_equal): {exact}", flush=True)
    print(f"  max abs diff: {max_abs_diff:.3e}", flush=True)
    print(f"  same n_fits: {same_n_fits}  same segment metadata: {same_segments}", flush=True)
    ok = exact and same_n_fits and same_segments
    print(f"IDENTITY CHECK: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# ---------------------------------------------------------------------------
# 3. Run one pre-registered cell (stage 7 = A1, stage 8 = A2) to completion or --stop-at
# ---------------------------------------------------------------------------
def run_cell(stage: int, design: pd.DataFrame, conf: pd.DataFrame, firsts: pd.DataFrame,
             n_jobs: int, stop_at: pd.Timestamp | None) -> dict:
    cells = [c for c in R4.planned_cells() if c["stage"] == stage]
    assert len(cells) == 1, f"stage {stage} does not name exactly one cell"
    c = cells[0]
    key = R4.cell_key(c)
    feats = R4.feature_set_v4(c["feature_arm"], c["population"])
    tr, te = R4.PO.fold_slices(design, c["fold"], c["population"])
    cuts = R3.refit_dates(c["scheme"], te, conf)
    ckpt_dir = CUTS_DIR / _safe_dirname(key)
    t0 = time.time()
    p, meta, status = fit_predict_walkforward_joblib(
        c["arm"], tr, te, feats, cuts, seed=c.get("seed", 0), n_jobs=n_jobs,
        ckpt_dir=ckpt_dir, cell_key=key, stop_at=stop_at)
    dt = time.time() - t0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cell_ckpt_path = OUT_DIR / "ckpt_a1a2.json"
    cell_ckpt = {"grid": {}, "detail": {}, "meta": {}, "extra": {}}
    if cell_ckpt_path.exists():
        cell_ckpt = json.loads(cell_ckpt_path.read_text())
        for kk in ("grid", "detail", "meta", "extra"):
            cell_ckpt.setdefault(kk, {})
    if not status["complete"]:
        print(f"stage {stage} ({key}): NOT COMPLETE this session -- "
              f"{status['n_done']}/{status['n_prepped']} refit dates checkpointed. "
              f"Resume with the exact same command; already-fit dates are not "
              f"recomputed.", flush=True)
        return {"stage": stage, "key": key, "status": status, "graded": False}

    R4.grade.firsts = firsts
    row, s, extra = R4.grade(c, te, p, meta, dt)
    cell_ckpt["grid"][key] = row
    cell_ckpt["detail"][key] = s
    cell_ckpt["meta"][key] = meta
    cell_ckpt["extra"][key] = extra
    tmp = cell_ckpt_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cell_ckpt, indent=1, default=str))
    os.replace(tmp, cell_ckpt_path)
    print(f"stage {stage} ({key}) GRADED: log_loss={row['log_loss']:.6f}  "
          f"wk03_gap_pp={row['wk03_gap_pp']}  nonconf_gap_pp={row['nonconf_gap_pp']}  "
          f"n_fits={row['n_fits']}  fit_seconds={row['fit_seconds']:.0f}", flush=True)
    return {"stage": stage, "key": key, "status": status, "graded": True, "row": row}


# ---------------------------------------------------------------------------
# 4. Merge into a report via v4's own (unmodified) grid/verdict/render path
# ---------------------------------------------------------------------------
def merge_report() -> None:
    import shutil
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "design_v4.parquet"
    meta_cache = OUT_DIR / "design_v4.meta.json"
    if not cache.exists():
        os.link(R4_DIR / "design_v4.parquet", cache)
        shutil.copyfile(R4_DIR / "design_v4.meta.json", meta_cache)
    # round4b's own directory is the authoritative, self-sufficient superset (section 11.1:
    # it holds its own copies of round4's ckpt_a/b/b2 PLUS its freshly-computed ckpt_c_g3 /
    # ckpt_d_floor). Pulling from round4b ALONE, exactly as `run_po_r4b_merge.py` did from
    # round4b's own glob, avoids re-deriving which of two same-named files is authoritative.
    ckpts = []
    for p_ in sorted(R4B_DIR.glob("ckpt_*.json")):
        if p_.name.startswith("ckpt_render"):
            continue
        dest = OUT_DIR / f"round4b_{p_.name}"
        shutil.copyfile(p_, dest)
        ckpts.append(dest.name)
    a1a2 = OUT_DIR / "ckpt_a1a2.json"
    if a1a2.exists():
        ckpts.append(a1a2.name)
    ckpts = sorted(set(ckpts))
    print("merging checkpoints:", ckpts, flush=True)
    old_argv = sys.argv
    old_out = R4.OUT_DIR
    try:
        R4.OUT_DIR = OUT_DIR
        sys.argv = ["train_possession_outcome_v4.py", "--render-only", "--stages", "0",
                    "--ckpt", "ckpt_render_a1a2.json", "--merge", ",".join(ckpts)]
        R4.main()
    finally:
        R4.OUT_DIR = old_out
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["identity", "run", "merge"], required=True)
    ap.add_argument("--stage", type=int, default=None, help="7 = A1, 8 = A2")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--stop-at", default="", help="ET wall clock HH:MM")
    a = ap.parse_args()
    assert a.n_jobs <= 6, "compute cap: at most 6 worker processes at any time"

    if a.mode == "merge":
        merge_report()
        return

    design, _dmeta = R4.build_design_v4(R4_DIR / "design_v4.parquet")
    conf = CF.build_conference_flags(SEASONS)
    firsts = CF.first_conference_game_dates(conf)
    firsts = firsts[firsts["season"] == PO.FOLDS[PO.SELECTION_FOLD]["test"][0]][
        ["team_id", "first_conf_date"]]

    if a.mode == "identity":
        ok = run_identity_check(design, conf)
        sys.exit(0 if ok else 1)

    stop_at = None
    if a.stop_at:
        hh, mm = a.stop_at.split(":")
        now = pd.Timestamp.now()
        stop_at = now.normalize() + pd.Timedelta(hours=int(hh), minutes=int(mm))
    assert a.stage is not None, "--mode run requires --stage 7 (A1) or 8 (A2)"
    run_cell(a.stage, design, conf, firsts, a.n_jobs, stop_at)


if __name__ == "__main__":
    main()
