"""Shared helpers for the box-parallel trainer wrappers (`train_*_par_v1.py`), lane J 2026-09-30.

Why this exists (docs/ops/aws_launch_chain.md section 17): `lightgbm==4.7.0` does not
multi-thread in the container image, and locally LightGBM threads are actively harmful
(300k x 50 rows: n_jobs=1 40 s vs n_jobs=20 218 s). What works is `n_jobs=1` per fit and
joblib process parallelism ACROSS refit dates / cells. Every wrapper therefore:

  * pins every thread-count env var to 1 BEFORE numpy / lightgbm import (call `pin_threads()`
    at the very top of the wrapper, before any other import);
  * dispatches one task per (population/class, refit date) through `run_tasks`, with a
    per-task checkpoint on disk, so a reclaimed spot instance loses at most the fits in flight;
  * takes the design / feature-table / output paths as parameters (`--design`,
    `--feature-table`, `--out-dir`) so the new team-rate feature table or an engine-inputs
    version is a command-line argument, never a hard-coded path;
  * NEVER writes into an existing artifact directory (`assert_fresh_out_dir`).

Nothing here changes what a model computes: fit inputs, seeds and hyper-parameters are the
wrapped trainer's own.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import subprocess
import time
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_THREAD_LIMIT")


def pin_threads() -> None:
    """Hard-set (not setdefault) every thread env var to 1. Call before importing numpy.

    `setdefault` was the 09-18 defect: the image's baked `OMP_NUM_THREADS=1` silently beat
    the trainer's own `CBB_THREADS` contract. Here 1 is the design, and it is forced."""
    for v in _THREAD_VARS:
        os.environ[v] = "1"
    # the trainers read CBB_THREADS for their n_jobs; keep them consistent with the pin
    os.environ["CBB_THREADS"] = "1"


def assert_fresh_out_dir(out_dir: Path, allow_resume: bool = True) -> None:
    """Refuse to write into a directory that holds a DIFFERENT run's artifacts.

    A resumed run of the same wrapper is allowed (it owns `.owner.json`); a directory that
    exists, is non-empty and has no owner stamp is somebody else's data and is refused."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not (out_dir / ".owner.json").exists():
        raise SystemExit(f"refusing to write into {out_dir}: it exists, is non-empty and is "
                         "not owned by a par_v1 wrapper. Pick a new --out-dir (never overwrite).")
    out_dir.mkdir(parents=True, exist_ok=True)


def stamp_owner(out_dir: Path, wrapper: str, args: dict) -> None:
    p = Path(out_dir) / ".owner.json"
    rec = {"wrapper": wrapper, "args": {k: str(v) for k, v in args.items()},
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        rec["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=20).stdout.strip()
    except Exception:                                                     # noqa: BLE001
        rec["git"] = "unknown"
    if not p.exists():
        p.write_text(json.dumps(rec, indent=1), encoding="utf-8")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Feature-table overlay: replace named columns of a design with a sibling table's values
# ---------------------------------------------------------------------------
_X_COL = re.compile(r"^x_(?P<a>.+?)__(?P<b>.+)$")


def overlay_columns(design, table_path: Path, keys: list[str], cols: list[str] | None = None,
                    recompute_interactions: bool = True, strict: bool = True):
    """Return `(design_with_overlaid_columns, report)`.

    `table_path` is a parquet (or csv) sibling feature table that carries the SAME column
    names as the design (the team-rate lane's contract: "a sibling table under the existing
    column names"). It is joined on `keys` (must be unique in the table). `cols` defaults to
    every non-key column of the table that also exists in the design.

    Interaction columns named `x_<a>__<b>` (possession_outcome's design) are the product of
    two team columns; when either input is overlaid they are recomputed as the float32
    product, because leaving them at the old values would be a silent train/serve skew.
    `strict=True` refuses any design row with no table match (a row the estimator does not
    cover must be an explicit decision, not a silent fall-back to the old value)."""
    import numpy as np
    import pandas as pd

    tp = Path(table_path)
    tab = pd.read_parquet(tp) if tp.suffix == ".parquet" else pd.read_csv(tp)
    miss_k = [k for k in keys if k not in tab.columns]
    if miss_k:
        raise SystemExit(f"feature table {tp} lacks key columns {miss_k}")
    if tab.duplicated(keys).any():
        raise SystemExit(f"feature table {tp} has duplicate keys {keys}")
    if cols is None:
        cols = [c for c in tab.columns if c not in keys and c in design.columns]
    bad = [c for c in cols if c not in tab.columns or c not in design.columns]
    if bad:
        raise SystemExit(f"overlay columns not in both design and table: {bad}")
    d = design
    left = d[keys].copy()
    got = left.merge(tab[[*keys, *cols]], on=keys, how="left", validate="many_to_one")
    n_unmatched = int(got[cols].isna().all(axis=1).sum()) if len(cols) else 0
    if strict and n_unmatched:
        raise SystemExit(f"{n_unmatched} design rows have no match in {tp} on {keys} "
                         "(strict overlay). Decide the fill rule explicitly.")
    d = d.copy()
    for c in cols:
        new = got[c].to_numpy()
        old = d[c].to_numpy()
        fill = np.isfinite(new.astype("float64"))
        merged = np.where(fill, new, old)
        d[c] = pd.Series(merged, index=d.index).astype(d[c].dtype)
    recomputed = []
    if recompute_interactions:
        for c in d.columns:
            m = _X_COL.match(c)
            if m and (m["a"] in cols or m["b"] in cols) and m["a"] in d and m["b"] in d:
                d[c] = (d[m["a"]].to_numpy(dtype="float32")
                        * d[m["b"]].to_numpy(dtype="float32")).astype(d[c].dtype)
                recomputed.append(c)
    report = {"table": str(tp), "table_sha256": sha256_file(tp), "keys": keys,
              "columns_overlaid": cols, "interactions_recomputed": recomputed,
              "n_design_rows": int(len(d)), "n_unmatched_rows": n_unmatched}
    return d, report


# ---------------------------------------------------------------------------
# Checkpointed task pool
# ---------------------------------------------------------------------------
def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


def run_tasks(tasks: dict, worker, n_jobs: int, ckpt_dir: Path, stop_at=None,
              log=print, max_jobs_cap: int | None = None) -> dict:
    """Run `worker(**payload)` for every `{key: payload}` in `tasks` and return
    `{key: result}`.

    * `n_jobs` process workers (joblib/loky). `n_jobs<=1` runs in-process (used by the
      identity tests, so a test never spawns more processes than it names).
    * Every finished task is pickled to `ckpt_dir/<key>.pkl` immediately; a rerun with the
      same `ckpt_dir` skips finished keys (spot-reclaim resume).
    * `stop_at` (a pandas Timestamp) stops DISPATCH of new results being awaited once
      reached; in-flight fits finish and are checkpointed.
    * `max_jobs_cap` is an optional hard ceiling (local safety); the box passes none."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if max_jobs_cap is not None and n_jobs > max_jobs_cap:
        raise SystemExit(f"n_jobs={n_jobs} exceeds this run's cap {max_jobs_cap}")
    done: dict = {}
    todo = {}
    for k, payload in tasks.items():
        f = ckpt_dir / f"{_safe(k)}.pkl"
        if f.exists():
            try:
                with open(f, "rb") as fh:
                    done[k] = pickle.load(fh)
                continue
            except Exception:                                             # noqa: BLE001
                f.unlink()   # a torn checkpoint (reclaim mid-write) is recomputed, not trusted
        todo[k] = payload
    log(f"[tasks] {len(tasks)} total, {len(done)} checkpointed, {len(todo)} to run, "
        f"n_jobs={n_jobs}")
    if not todo:
        return done

    def _save(k, res):
        f = ckpt_dir / f"{_safe(k)}.pkl"
        tmp = f.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(res, fh, protocol=4)
        os.replace(tmp, f)
        done[k] = res

    t0 = time.time()
    if n_jobs <= 1:
        for k, payload in todo.items():
            _save(k, worker(**payload))
            log(f"[tasks] {k} done ({len(done)}/{len(tasks)}, {time.time() - t0:.0f}s)")
            if stop_at is not None and time.time() >= stop_at.timestamp():
                log("[tasks] STOP-AT reached")
                break
        return done

    from joblib import Parallel, delayed
    keys = list(todo)
    gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
        delayed(_keyed)(worker, k, todo[k]) for k in keys)
    for k, res in gen:
        _save(k, res)
        log(f"[tasks] {k} done ({len(done)}/{len(tasks)}, {time.time() - t0:.0f}s)")
        if stop_at is not None and time.time() >= stop_at.timestamp():
            log("[tasks] STOP-AT reached; in-flight fits finish and are checkpointed "
                "on the next resume")
            break
    return done


def _keyed(worker, key, payload):
    pin_threads()
    return key, worker(**payload)
