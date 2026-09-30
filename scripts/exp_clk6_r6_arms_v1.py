"""exp_clk6_r6_arms_v1.py -- clock round 6 (experiments.md section 28) arms L2 / L2a:
refit the SERVED clock specification on the event-layer v4 training table.

Lane B, 2026-09-30.  WRITTEN, NOT RUN (PM: resume commands only today).
Every served trainer is imported and run UNEDITED; only its module-level path
constants are pointed at an arm root, so nothing served is overwritten:

    data/processed/models/clock/r6_<arm>/design_v2.parquet     (step design)
    data/processed/models/clock/r6_<arm>/v3c_s1/...            (step s1: srfloor_P3 schedule)
    data/processed/models/clock/r6_<arm>/v5b_bakeoff/...       (step latent: v5b sigma refit)

Arms:
  L2   possessions_v4   (phantoms removed, and-one labels follow the floor)
  L2a  possessions_v4a  (phantoms removed, made-and-one label kept at made_FG);
       build it first: scripts/build_possessions_v4.py --variant l2a --seasons 2022 2023 2024 2025
D1 (season-part pooling of the S1 refit) needs a trainer option that does not
exist yet; it is not wired here.

Usage:
    .venv/Scripts/python.exe scripts/exp_clk6_r6_arms_v1.py --arm L2 --step design
    .venv/Scripts/python.exe scripts/exp_clk6_r6_arms_v1.py --arm L2 --step s1
    .venv/Scripts/python.exe scripts/exp_clk6_r6_arms_v1.py --arm L2 --step latent
    .venv/Scripts/python.exe scripts/exp_clk6_r6_arms_v1.py --arm L2 --step design --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CK = ROOT / "data/processed/models/clock"
POSS = {"L2": ROOT / "data/processed/possessions_v4", "L2a": ROOT / "data/processed/possessions_v4a"}
SEASONS = [2022, 2023, 2024, 2025]


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(POSS), required=True)
    ap.add_argument("--step", choices=["design", "s1", "latent"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = CK / f"r6_{a.arm}"
    print(f"arm {a.arm}: possessions {POSS[a.arm]} -> {root}", flush=True)
    if a.dry_run:
        return 0
    root.mkdir(parents=True, exist_ok=True)
    if a.step == "design":
        from cbb_sim.models import clock as ck
        design, diag = ck.build_design(SEASONS, poss_dir=POSS[a.arm])
        design.to_parquet(root / "design_v2.parquet", index=False)
        print(f"design {len(design):,} rows", flush=True)
    elif a.step == "s1":
        m = load("train_clock_v3c_s1_r6", "train_clock_v3c_s1.py")
        # OUT stays the clock dir so every manifest's `model_file` is relative
        # to CK_DIR, which is what the engine's ArtifactManifest resolves against
        m.OUT, m.S1_DIR, m.DESIGN_V2 = CK, root / "v3c_s1", root / "design_v2.parquet"
        sys.argv = [sys.argv[0], "--only", "srfloor_P3"]
        return m.main()
    else:
        m = load("exp_clk5b_r6", "exp_clk5b_mean_consistent.py")
        m.r5.DESIGN = root / "design_v2.parquet"
        served_load = m.r5.load_schedule
        mode = f"v3c_r6{a.arm}_srfloor_P3_s1"

        def load_schedule(_mode: str):
            # the v5b fit reads the served S1 schedule by name; point it at this arm's
            return served_load(mode)

        m.r5.load_schedule = load_schedule
        sys.argv = [sys.argv[0], "--out", str((root / "v5b_bakeoff").relative_to(ROOT)), "--fit-only"]
        m.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
