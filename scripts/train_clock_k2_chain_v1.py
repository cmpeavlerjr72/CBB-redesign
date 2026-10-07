"""train_clock_k2_chain_v1.py -- the SERVED clock since 2026-10-07 (`v5b_r8K2_glat_pmean`, clock round 8 arm K2,
outcome-conditioned possession time; docs/tests/adoption_clock_K2_2026-10-07.md) refit on a chain's clock design,
into a NEW root that mirrors the `data/processed/models/clock/` layout the K2 mode resolves.

Versioned sibling of `train_clock_chain_v1.py` (the L2 chain clock, unchanged) for the K2 arm. Same rule: the
served K2 recipe is run UNEDITED with only its module-level path constants re-pointed:

  step train  : `train_clock_r8_chance_v1.py --arm K2 --fold F2` with
                  train_clock_r8_tempo_v1.L2 = <design-root>   (its `design_v2.parquet` + `clock_censoring/`,
                                                                 i.e. the chain's clock stage root)
                  train_clock_r8_chance_v1.P4 = possessions_<poss-version> (the chance tables: d1, end class, continuation)
                  train_clock_r8_chance_v1.CKD = <root>
                -> <root>/r8_K2/F2/manifest.json + six S1 pickles (model_file relative to <root>)
  step latent : the recipe of `exp_clk8_latent_v2.py --arm K2` (the UNEDITED `exp_clk5b_mean_consistent.py
                --fit-only` with the design read as d1, schedule read from <root>) with
                  r5.DESIGN = <design-root>/design_v2.parquet, r5.CK_DIR = <root>
                -> <root>/r8_K2/v5b_bakeoff/v5b_bakeoff_report.json
  step serve  : copies `v5_bakeoff/v5_bakeoff_report.json` (the adapter checks it exists; never reads values)
                so <root> can be served by rebinding clock_adapter_v3.CK_DIR / V5_PARAMS with
                ENGINE_CLOCK=v5b_r8K2_glat_pmean (the engine default)

Identity (`--design-root data/processed/models/clock/r6_L2 --poss-version v4 --identity`): the six S1 manifests and
first-chance pmfs and B1_sigma must equal the served `clock/r8_K2` ones (checked 2026-10-07: equal, pmf max diff 0.0,
B1_sigma equal; and the engine served from the refit root through run_engine_overlay_v1 is bit-identical to parity v10,
60 games x 5 seeds, games and players).

    .venv/Scripts/python.exe scripts/train_clock_k2_chain_v1.py --design-root <chain>/clock/root --root <chain>/clock_k2/root \
        --poss-version v4otc
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
CK = ROOT / "data/processed/models/clock"
ARM = "K2"
FOLD = "F2"


class _PdShim:
    """exp_clk8_latent_v2's shim: the design read through it takes d1 / end class and the engine days_since_start."""
    def __init__(self, pd, design_path, fix):
        self._pd, self._path, self._fix = pd, Path(design_path), fix

    def __getattr__(self, k):
        return getattr(self._pd, k)

    def read_parquet(self, path, *args, **kw):
        df = self._pd.read_parquet(path, *args, **kw)
        if Path(path) == self._path and "game_date" in df.columns and "season" in df.columns:
            df = self._fix(df)
        return df


def _modules(design_root: Path, root: Path, poss_version: str):
    from cbb_sim.pbp.possessions import possessions_dir
    import train_clock_r8_tempo_v1 as T
    import train_clock_r8_chance_v1 as TK
    T.L2 = design_root
    TK.P4 = (ROOT / possessions_dir(poss_version)).resolve()
    TK.CKD = root
    return T, TK


def step_train(design_root: Path, root: Path, poss_version: str) -> int:
    _T, TK = _modules(design_root, root, poss_version)
    argv = sys.argv
    sys.argv = [argv[0], "--arm", ARM, "--fold", FOLD]
    try:
        return TK.main()
    finally:
        sys.argv = argv


def step_latent(design_root: Path, root: Path, poss_version: str) -> None:
    import numpy as np
    import pandas as pd
    from cbb_sim.models import clock_r8
    _T, TK = _modules(design_root, root, poss_version)
    first, _ = TK.chance_tables(TK.T.ALL_SEASONS)
    f = first.reset_index()
    f["_p"] = f["period"].astype("int64")
    f["_i"] = f["poss_index"].astype("int64")
    f = f[["game_id", "_p", "_i", "d1", "end1_code"]]

    def fix(df):
        df["days_since_start"] = clock_r8.season_day(df)
        k = df[["game_id", "period", "poss_index"]].copy()
        k["_p"] = k["period"].astype("int64")
        k["_i"] = k["poss_index"].astype("int64")
        m = k[["game_id", "_p", "_i"]].merge(f, on=["game_id", "_p", "_i"], how="left")
        d1 = m["d1"].to_numpy()
        ok = np.isfinite(d1)
        df = df.loc[ok].copy()
        df["duration_s"] = np.clip(d1[ok].astype("int64"), 0, 90)
        df["end1_code"] = m["end1_code"].to_numpy()[ok].astype("int64")
        return df

    spec = importlib.util.spec_from_file_location("exp_clk5b_k2chain", ROOT / "scripts" / "exp_clk5b_mean_consistent.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["exp_clk5b_k2chain"] = m
    spec.loader.exec_module(m)
    m.r5.DESIGN = design_root / "design_v2.parquet"
    m.r5.CK_DIR = root
    m.pd = _PdShim(pd, m.r5.DESIGN, fix)
    served_load = m.r5.load_schedule
    mode = f"v3c_r8{ARM}_P3_s1"
    m.r5.load_schedule = lambda _mode: served_load(mode)
    argv = sys.argv
    sys.argv = [argv[0], "--out", str(root / f"r8_{ARM}" / "v5b_bakeoff"), "--fit-only"]
    try:
        m.main()
    finally:
        sys.argv = argv


def step_serve(root: Path) -> dict:
    dst = root / "v5_bakeoff" / "v5_bakeoff_report.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CK / "v5_bakeoff" / "v5_bakeoff_report.json", dst)
    return {"copied": str(dst)}


def identity(root: Path, design_root: Path, poss_version: str) -> dict:
    """manifest months (minus fit_seconds), the six S1 first-chance pmfs on a fixed 2025 d1 design sample, and the
    latent B1_sigma vs the served clock/r8_K2. Pickle BYTES are reported but not required: a refit of the served
    recipe is pmf-identical yet pickles to different bytes (memo layout), as the 2026-10-07 check found."""
    import pickle
    import numpy as np
    T, TK = _modules(design_root, root, poss_version)
    mn = json.loads((root / f"r8_{ARM}" / FOLD / "manifest.json").read_text(encoding="utf-8"))
    mo = json.loads((CK / f"r8_{ARM}" / FOLD / "manifest.json").read_text(encoding="utf-8"))
    strip = lambda m: {k: v for k, v in m.items() if k != "fit_seconds"}  # noqa: E731
    d = T.load_design()
    d = d[d["season"] == 2025]
    first, _ = TK.chance_tables([2025])
    samp = TK.build_d1_design(d, first).sample(20000, random_state=3).reset_index(drop=True)
    res = []
    for a, b in zip(mn["months"], mo["months"]):
        with open(root / a["model_file"], "rb") as f:
            an = pickle.load(f)
        with open(CK / b["model_file"], "rb") as f:
            ao = pickle.load(f)
        res.append({"refit": a["refit_date"], "manifest_equal": strip(a) == strip(b),
                    "max_abs_pmf_diff": float(np.abs(np.asarray(an.pmf(samp)) - np.asarray(ao.pmf(samp))).max()),
                    "bytes_equal": (root / a["model_file"]).read_bytes() == (CK / b["model_file"]).read_bytes()})
    rn = json.loads((root / f"r8_{ARM}" / "v5b_bakeoff" / "v5b_bakeoff_report.json").read_text(encoding="utf-8"))
    ro = json.loads((CK / f"r8_{ARM}" / "v5b_bakeoff" / "v5b_bakeoff_report.json").read_text(encoding="utf-8"))
    out = {"n_months": [len(mn["months"]), len(mo["months"])], "s1_months": res,
           "B1_sigma_new": rn["params"]["F2"]["B1_sigma"], "B1_sigma_served": ro["params"]["F2"]["B1_sigma"]}
    out["identical"] = bool(len(mn["months"]) == len(mo["months"])
                            and all(r["manifest_equal"] and r["max_abs_pmf_diff"] == 0.0 for r in res)
                            and out["B1_sigma_new"] == out["B1_sigma_served"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design-root", type=Path, required=True,
                    help="dir holding design_v2.parquet + clock_censoring/ (the chain's clock stage root; served: clock/r6_L2)")
    ap.add_argument("--root", type=Path, required=True, help="NEW dir; becomes the K2 served-layout clock dir")
    ap.add_argument("--poss-version", default="v4", choices=["v4", "v4otc"])
    ap.add_argument("--steps", default="train,latent,serve")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root, droot = a.root.resolve(), a.design_root.resolve()
    if root == CK.resolve() or (root.parent == CK.resolve() and root.name.startswith(("r8_", "r6_", "v3c", "v5"))):
        raise SystemExit("refusing to write into the served clock dir")
    for need in ("design_v2.parquet", "clock_censoring"):
        if not (droot / need).exists() and not a.dry_run:
            raise SystemExit(f"--design-root {droot} lacks {need} (run the chain's clock stage first)")
    steps = [s for s in a.steps.split(",") if s]
    print(f"clock K2 chain: design {droot}, possessions {a.poss_version} -> {root}; steps {steps}", flush=True)
    if a.dry_run:
        return 0
    root.mkdir(parents=True, exist_ok=True)
    rep_p = root / "clock_k2_chain_report.json"
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {"steps": {}}
    rep.update({"poss_version": a.poss_version, "design_root": str(droot), "arm": ARM, "fold": FOLD})
    for s in steps:
        t = time.time()
        if s == "train":
            r = {"rc": step_train(droot, root, a.poss_version)}
        elif s == "latent":
            step_latent(droot, root, a.poss_version)
            r = {}
        elif s == "serve":
            r = step_serve(root)
        else:
            raise SystemExit(f"unknown step {s}")
        r["seconds"] = round(time.time() - t, 1)
        rep["steps"][s] = r
        rep_p.write_text(json.dumps(rep, indent=1, default=str))
        print(f"[{s}] {r}", flush=True)
    for need in (f"r8_{ARM}/{FOLD}/manifest.json", f"r8_{ARM}/v5b_bakeoff/v5b_bakeoff_report.json",
                 "v5_bakeoff/v5_bakeoff_report.json"):
        if not (root / need).exists():
            raise SystemExit(f"K2 chain root {root} lacks {need}")
    if a.identity:
        rep["identity"] = identity(root, droot, a.poss_version)
        rep_p.write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps(rep["identity"], indent=1, default=str))
        if not rep["identity"]["identical"]:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
