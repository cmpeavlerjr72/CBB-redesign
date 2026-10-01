#!/usr/bin/env python
"""build_engine_inputs_chain_v1.py -- engine inputs for a full-retrain artifact set, by the v3 TAG path.

Composition only; nothing here is new modelling. Three existing, proven recipes in order:
  1. ratings (optional): the four own-rating channels of `engine_v3` re-derived from `--ratings-dir` with lane N's
     `build_engine_inputs_v3_ratings_tag_v1.rating_cells` (imported, not edited), with lane N's parity check first
     (the recipe on the served ratings must reproduce the base's channels exactly, 0.0) -> `<out>/base/`;
  2. team-rate table + serving overlay: lane G's `build_engine_inputs_v3_tag_v1.py` run UNEDITED as a subprocess with
     `--base-dir <out>/base` (the ratings-substituted base, or `engine_v3` itself) and the retrained PO / fg_make /
     rebound artifact dirs -> `<out>/inputs/` (arrays, event block, overlay/, builder_report.json);
  3. `overrides.json` for `scripts/run_engine_overlay_v1.py`: an engine scratch dir (the overlay's event dir plus the
     three static engine joblibs), and the fg_make / rebound overlay paths, the retrained clock root
     (`clock_adapter_v3.CK_DIR`, `V5_PARAMS`) and the retrained rotation manifest. Paths are repo-relative POSIX
     so the same file works on Windows and in the Linux box image.

With every option at its default the inputs equal `engine_v3` byte for byte and overrides.json serves the served
artifacts (the chain's DEFAULTS proof).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
MODELS = ROOT / "data/processed/models"
TAG = "F2_2025"
SERVED_RATINGS = ROOT / "data/processed/ratings"


def rel(p: Path) -> str:
    return Path(os.path.relpath(Path(p).resolve(), ROOT)).as_posix()


def ratings_base(base: Path, out: Path, ratings_dir: Path) -> dict:
    import numpy as np
    import pandas as pd
    import build_engine_inputs_v3_ratings_tag_v1 as RT
    out.mkdir(parents=True)
    z = dict(np.load(base / f"arrays_{TAG}.npz"))
    names = json.loads((base / f"names_{TAG}.json").read_text(encoding="utf-8"))
    games = pd.read_parquet(base / f"games_{TAG}.parquet")
    eb = np.load(base / f"event_block_{TAG}.npz")["team_block"]
    tn = names["team_names"]
    j = [tn[c] for c in RT.COLS]
    served = RT.rating_cells(games, SERVED_RATINGS)
    par = {"team_static_max_abs": float(np.abs(served - z["team_static"][:, :, j]).max()),
           "event_block_max_abs": float(np.abs(served - eb[:, :, j]).max())}
    if par["team_static_max_abs"] != 0.0 or par["event_block_max_abs"] != 0.0:
        raise SystemExit(f"PARITY FAILED (ratings recipe vs base): {par}")
    new = RT.rating_cells(games, ratings_dir)
    ts = z["team_static"].copy(); eb2 = eb.copy()
    ts[:, :, j] = new; eb2[:, :, j] = new
    arrs = dict(z); arrs["team_static"] = ts
    np.savez_compressed(out / f"arrays_{TAG}.npz", **arrs)
    np.savez_compressed(out / f"event_block_{TAG}.npz", team_block=eb2)
    for f in (f"games_{TAG}.parquet", f"names_{TAG}.json"):
        shutil.copyfile(base / f, out / f)
    return {"ratings_dir": str(ratings_dir), "parity_vs_base": par,
            "delta_abs_mean": {c: float(np.abs(new[:, :, k] - served[:, :, k]).mean()) for k, c in enumerate(RT.COLS)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="NEW dir: gets base/, inputs/, overrides.json")
    ap.add_argument("--base-dir", type=Path, default=MODELS / "engine_v3")
    ap.add_argument("--ratings-dir", type=Path, default=SERVED_RATINGS)
    ap.add_argument("--team-rate-table", type=Path, default=None)
    ap.add_argument("--team-rate-missing", choices=["raise", "keep_served"], default="raise")
    ap.add_argument("--po-artifacts", type=Path, default=None)
    ap.add_argument("--fg-artifacts", type=Path, default=None)
    ap.add_argument("--fg-m", type=Path, default=None)
    ap.add_argument("--rb-artifacts", type=Path, default=None)
    ap.add_argument("--clock-root", type=Path, default=None)
    ap.add_argument("--rotation-dir", type=Path, default=None)
    ap.add_argument("--tag", default="chain")
    a = ap.parse_args()
    out = a.out.resolve()
    if out.exists():
        raise SystemExit(f"{out} exists: never overwrite (the chain resumes per stage, not inside a stage)")
    out.mkdir(parents=True)
    rep: dict = {"base": rel(a.base_dir)}
    if a.ratings_dir.resolve() != SERVED_RATINGS.resolve():
        base = out / "base"
        rep["ratings"] = ratings_base(a.base_dir, base, a.ratings_dir)
    else:
        base = a.base_dir
        rep["ratings"] = "served"
    cmd = [sys.executable, str(ROOT / "scripts/build_engine_inputs_v3_tag_v1.py"), "--tag", a.tag,
           "--base-dir", str(base), "--out-dir", str(out / "inputs")]
    if a.team_rate_table:
        cmd += ["--team-rate-table", str(a.team_rate_table), "--team-rate-missing", a.team_rate_missing]
    if a.po_artifacts:
        cmd += ["--po-artifacts", str(a.po_artifacts)]
    if a.fg_artifacts:
        cmd += ["--fg-artifacts", str(a.fg_artifacts)]
    if a.fg_m:
        cmd += ["--fg-m", str(a.fg_m)]
    if a.rb_artifacts:
        cmd += ["--rb-artifacts", str(a.rb_artifacts)]
    rep["tag_builder_cmd"] = [rel(Path(c)) if os.path.isabs(c) else c for c in cmd[1:]]
    subprocess.run(cmd, check=True)
    inputs = out / "inputs"
    ov_models = inputs / "overlay/data/processed/models"
    # engine scratch dir: the overlay's event dir + the three static engine joblibs the adapters may open
    eng = out / "engine_scratch"
    shutil.copytree(ov_models / "engine" / f"event_round2_s1_{TAG}", eng / f"event_round2_s1_{TAG}")
    for name in (f"fg_make_FGA_3_decision8_F2.joblib", f"free_throw_F2.joblib", f"rebound_F2.joblib"):
        shutil.copy2(MODELS / "engine" / name, eng / name)
    over = {"adapters.ENGINE_DIR": rel(eng)}
    if a.fg_artifacts:
        over["adapters.FG_DIR"] = rel(ov_models / "fg_make")
    if a.rb_artifacts:
        over["adapters.RB_S1_MANIFEST"] = rel(ov_models / "rebound/s1_confirm/S1_weekly/F2/manifest.json")
    if a.clock_root:
        cr = a.clock_root.resolve()
        for need in ("v3c_s1/manifest_srfloor_P3.json", "v5b_bakeoff/v5b_bakeoff_report.json",
                     "v5_bakeoff/v5_bakeoff_report.json"):
            if not (cr / need).exists():
                raise SystemExit(f"clock root {cr} lacks {need}")
        over["clock_adapter_v3.CK_DIR"] = rel(cr)
        over["adapters.CK_DIR"] = rel(cr)
        over["clock_adapter_v3.V5_PARAMS"] = rel(cr / "v5_bakeoff/v5_bakeoff_report.json")
    if a.rotation_dir:
        mp = a.rotation_dir.resolve() / "rotation_r2_s1_F2_2025.json"
        if not mp.exists():
            raise SystemExit(f"{mp} missing (train_rotation_v3b_s1_poss_v1.py writes it)")
        over["rotation_adapter.R2_S1_MANIFEST"] = rel(mp)
    (inputs / "overrides.json").write_text(json.dumps(over, indent=1), encoding="utf-8")
    rep["overrides"] = over
    rep["tag_builder_report"] = json.loads((inputs / "builder_report.json").read_text(encoding="utf-8"))
    (out / "inputs_chain_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"overrides": over, "ratings": rep["ratings"] if isinstance(rep["ratings"], str)
                      else rep["ratings"]["delta_abs_mean"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
