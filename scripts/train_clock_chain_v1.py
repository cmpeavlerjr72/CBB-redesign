"""train_clock_chain_v1.py -- the SERVED clock (`v5b_glat_pmean` = v3c srfloor_P3 S1 + v5b latent sigma) refit
under a chosen possessions version and ratings dir, into a NEW root that mirrors `data/processed/models/clock/`.

Versioned sibling of `scripts/exp_clk6_r6_arms_v1.py` (lane B, NOT edited), generalised for the full-retrain
chain: lane B's wrapper hard-codes the arm roots under the served clock dir and the served ratings; this one
takes `--root`, `--poss-version` and `--ratings-dir`. As in lane B's wrapper, every served trainer is run
UNEDITED with only its module-level path constants re-pointed:

  step censor : horn-censoring side table from the chosen possessions (lane B's recipe); for `--poss-version v1`
                the served table `data/processed/clock_censoring/` IS that table and is used directly
  step design : `clock.build_design(SEASONS, poss_dir=..., ratings_dir=...)` -> <root>/design_v2.parquet
                (the call `train_clock_v2.py` makes, with the two inputs as parameters)
  step s1     : `train_clock_v3c_s1.py --only srfloor_P3` with OUT=<root>, S1_DIR=<root>/v3c_s1,
                DESIGN_V2=<root>/design_v2.parquet -> manifests whose `model_file` is relative to <root>,
                exactly the layout the engine resolves against CK_DIR
  step latent : `exp_clk5b_mean_consistent.py --fit-only` with its schedule and design read from <root>
                (`r5.CK_DIR`, `r5.DESIGN`) -> <root>/v5b_bakeoff/v5b_bakeoff_report.json
  step serve  : copies `v5_bakeoff/v5_bakeoff_report.json` (the adapter checks it exists; never reads values)
                so <root> can be served by rebinding CK_DIR (engine default ENGINE_CLOCK unchanged)

Identity (`--poss-version v1 --ratings-dir data/processed/ratings --identity`): the design, the six S1 pmfs and
B1_sigma must equal the served ones.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
CK = ROOT / "data/processed/models/clock"
SEASONS = [2022, 2023, 2024, 2025]
SERVED_CENSOR = ROOT / "data/processed/clock_censoring"


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def censor_dir_for(root: Path, poss_version: str) -> Path:
    return SERVED_CENSOR if poss_version == "v1" else root / "clock_censoring"


def step_censor(root: Path, poss_version: str) -> dict:
    import pandas as pd
    from cbb_sim.pbp.possessions import possessions_dir
    cdir = censor_dir_for(root, poss_version)
    if cdir == SERVED_CENSOR:
        return {"censor_dir": str(cdir), "note": "served table (v1 possessions)"}
    cdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in SEASONS:
        q = pd.read_parquet(possessions_dir(poss_version) / f"possessions_{s}.parquet")
        side = q[["game_id", "period", "poss_index", "start_clock", "end_clock", "duration_s"]].copy()
        side["censored_horn"] = (q["end_clock"] <= 0).to_numpy()
        side["flag_end_period"] = (q["terminal_event"] == "end_period").to_numpy()
        side.to_parquet(cdir / f"censoring_v1_{s}.parquet", index=False)
        out[s] = int(len(side))
    return {"censor_dir": str(cdir), "rows": out}


def step_design(root: Path, poss_version: str, ratings_dir: str) -> dict:
    from cbb_sim.models import clock as ck
    from cbb_sim.pbp.possessions import possessions_dir
    design, _diag = ck.build_design(SEASONS, poss_dir=possessions_dir(poss_version), ratings_dir=ratings_dir)
    design.to_parquet(root / "design_v2.parquet", index=False)
    return {"rows": int(len(design))}


def step_s1(root: Path, poss_version: str) -> int:
    from cbb_sim.models import clock_v3 as c3
    c3.attach_horn_censoring = functools.partial(c3.attach_horn_censoring,
                                                 censor_dir=censor_dir_for(root, poss_version))
    m = load("train_clock_v3c_s1_chain", "train_clock_v3c_s1.py")
    m.OUT, m.S1_DIR, m.DESIGN_V2 = root, root / "v3c_s1", root / "design_v2.parquet"
    argv = sys.argv
    sys.argv = [argv[0], "--only", "srfloor_P3"]
    try:
        return m.main()
    finally:
        sys.argv = argv


def step_latent(root: Path, poss_version: str) -> None:
    from cbb_sim.models import clock_v3 as c3
    c3.attach_horn_censoring = functools.partial(c3.attach_horn_censoring,
                                                 censor_dir=censor_dir_for(root, poss_version))
    m = load("exp_clk5b_chain", "exp_clk5b_mean_consistent.py")
    m.r5.DESIGN = root / "design_v2.parquet"
    m.r5.CK_DIR = root
    argv = sys.argv
    sys.argv = [argv[0], "--out", str(root / "v5b_bakeoff"), "--fit-only"]
    try:
        m.main()
    finally:
        sys.argv = argv


def step_serve(root: Path) -> dict:
    src = CK / "v5_bakeoff" / "v5_bakeoff_report.json"
    dst = root / "v5_bakeoff" / "v5_bakeoff_report.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return {"copied": str(dst)}


def identity(root: Path) -> dict:
    """design, the six srfloor_P3 pmfs on a fixed design sample, and B1_sigma vs the served clock."""
    import pickle
    import numpy as np
    import pandas as pd
    out = {}
    new_d = pd.read_parquet(root / "design_v2.parquet")
    old_d = pd.read_parquet(CK / "design_v2.parquet")
    out["design_equal"] = bool(new_d.equals(old_d))
    if not out["design_equal"] and list(new_d.columns) == list(old_d.columns) and len(new_d) == len(old_d):
        out["design_cols_differing"] = [c for c in old_d.columns if not new_d[c].equals(old_d[c])]
    mn = json.loads((root / "v3c_s1" / "manifest_srfloor_P3.json").read_text(encoding="utf-8"))
    mo = json.loads((CK / "v3c_s1" / "manifest_srfloor_P3.json").read_text(encoding="utf-8"))
    from cbb_sim.models import clock_v3  # noqa: F401  (the pickles need the extended module)
    samp = old_d[old_d["season"] == 2025].sample(20000, random_state=7).reset_index(drop=True)
    d = np.column_stack  # noqa: F841
    res = []
    for a, b in zip(mn["months"], mo["months"]):
        with open(root / a["model_file"], "rb") as f:
            an = pickle.load(f)
        with open(CK / b["model_file"], "rb") as f:
            ao = pickle.load(f)
        try:
            pn, po = np.asarray(an.pmf(samp)), np.asarray(ao.pmf(samp))
            res.append({"refit": a["refit_date"], "max_abs_pmf_diff": float(np.abs(pn - po).max()),
                        "bytes_equal": (root / a["model_file"]).read_bytes() == (CK / b["model_file"]).read_bytes()})
        except Exception as e:                                          # noqa: BLE001
            res.append({"refit": a["refit_date"], "pmf_error": f"{type(e).__name__}: {e}",
                        "bytes_equal": (root / a["model_file"]).read_bytes() == (CK / b["model_file"]).read_bytes()})
    out["s1_months"] = res
    rn = json.loads((root / "v5b_bakeoff" / "v5b_bakeoff_report.json").read_text(encoding="utf-8"))
    ro = json.loads((CK / "v5b_bakeoff" / "v5b_bakeoff_report.json").read_text(encoding="utf-8"))
    out["B1_sigma_new"] = rn["params"]["F2"]["B1_sigma"]
    out["B1_sigma_served"] = ro["params"]["F2"]["B1_sigma"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="NEW dir; becomes the served-layout clock dir")
    ap.add_argument("--poss-version", default="v4", choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--ratings-dir", default="data/processed/ratings")
    ap.add_argument("--steps", default="censor,design,s1,latent,serve")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = a.root.resolve()
    if root == CK.resolve() or CK.resolve() in root.parents and root.name in ("v3c_s1", "v5b_bakeoff"):
        raise SystemExit("refusing to write into the served clock dir")
    steps = [s for s in a.steps.split(",") if s]
    print(f"clock chain: possessions {a.poss_version}, ratings {a.ratings_dir} -> {root}; steps {steps}", flush=True)
    if a.dry_run:
        return 0
    root.mkdir(parents=True, exist_ok=True)
    rep_p = root / "clock_chain_report.json"
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {"steps": {}}
    rep.update({"poss_version": a.poss_version, "ratings_dir": a.ratings_dir})
    for s in steps:
        t = time.time()
        if s == "censor":
            r = step_censor(root, a.poss_version)
        elif s == "design":
            r = step_design(root, a.poss_version, a.ratings_dir)
        elif s == "s1":
            r = {"rc": step_s1(root, a.poss_version)}
        elif s == "latent":
            step_latent(root, a.poss_version)
            r = {}
        elif s == "serve":
            r = step_serve(root)
        else:
            raise SystemExit(f"unknown step {s}")
        r["seconds"] = round(time.time() - t, 1)
        rep["steps"][s] = r
        rep_p.write_text(json.dumps(rep, indent=1, default=str))
        print(f"[{s}] {r}", flush=True)
    if a.identity:
        rep["identity"] = identity(root)
        rep_p.write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps(rep["identity"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
