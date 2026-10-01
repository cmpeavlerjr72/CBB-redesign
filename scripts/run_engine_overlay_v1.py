"""run_engine_overlay_v1.py -- run an UNEDITED engine runner with a full-retrain artifact set served in-process.

Versioned sibling of lane N's `run_po4b_closed_loop_sample_overlay_v1.py` (not edited), generalised from three artifact
families to every family the full retrain touches. No Docker bind mounts (they are what made the box path fragile):
the engine's fixed artifact paths are module attributes, so they are rebound in the parent AND in every worker (the
worker initializer is wrapped; the overrides travel in an env var, so this holds under spawn and fork).

    overrides.json (written by scripts/build_engine_inputs_chain_v1.py next to the tagged input dir):
      {"adapters.ENGINE_DIR": "<scratch engine dir>", "adapters.FG_DIR": "...", "adapters.RB_S1_MANIFEST": "...",
       "adapters.CK_DIR": "...", "clock_adapter_v3.CK_DIR": "...", "clock_adapter_v3.V5_PARAMS": "...",
       "rotation_adapter.R2_S1_MANIFEST": "..."}
    keys absent = the served path stays. An empty overrides file serves the served stack.

    # sample closed loop (local smoke, box sample reads):
    python scripts/run_engine_overlay_v1.py --overrides <inputs>/overrides.json --runner sample -- \
        --sample-file data/processed/truth/stride500_verified_v1_F2_2025.parquet --arm round2_s1 \
        --input-dir <inputs> --seeds 25 --seed-offset 0 --workers 3 --tag <tag> --results-dir results/engine_v0
    # full-size (box): the same with --runner full -- <every scripts/run_engine.py argument>

After the run it stamps the overrides and every family's loaded source into the run's `run_meta.json`, and it
FAILS if any overridden family's source path in the adapters' own provenance does not point inside the overrides
(the "silently served the served artifact" trap of docs/ops/engine_inputs_v3_tag_path_2026-09-30.md section 1).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
ENV = "CBB_FULL_RETRAIN_OVERRIDES"
MODS = {"adapters": "cbb_sim.engine.adapters", "clock_adapter_v3": "cbb_sim.engine.clock_adapter_v3",
        "rotation_adapter": "cbb_sim.engine.rotation_adapter"}


def apply_overrides() -> dict:
    ov = json.loads(os.environ.get(ENV, "{}"))
    for k, v in ov.items():
        mod, attr = k.split(".", 1)
        m = importlib.import_module(MODS[mod])
        if not hasattr(m, attr):
            raise SystemExit(f"override {k}: {MODS[mod]} has no attribute {attr}")
        setattr(m, attr, Path(v))
    return ov


# Captured at IMPORT time, so a spawned worker (which re-imports this file as __mp_main__ and unpickles the
# initializer by name) finds the original initializers too.
import run_po4b_closed_loop as _R  # noqa: E402
import run_engine as _RE  # noqa: E402

_ORIG = {"sample": _R._init_worker, "full": _RE._init_worker}


def _stamp(ov: dict) -> None:
    """One line per worker: proof that the WORKER (not only the parent) serves the overrides."""
    from cbb_sim.engine import adapters as AD
    from cbb_sim.engine import rotation_adapter as RA
    print(f"[worker {os.getpid()}] overrides applied ({len(ov)}): ENGINE_DIR={AD.ENGINE_DIR} FG_DIR={AD.FG_DIR} "
          f"RB={AD.RB_S1_MANIFEST} ROT={RA.R2_S1_MANIFEST}", flush=True)


def _init_sample(*args):
    _stamp(apply_overrides())
    return _ORIG["sample"](*args)


def _init_full(*args):
    _stamp(apply_overrides())
    return _ORIG["full"](*args)


def check_sources(meta: dict, ov: dict) -> list[str]:
    """Every overridden family must report a source under its override."""
    src = (meta.get("adapter_flags") or {}).get("sources") or {}
    bad = []
    want = {"event": "adapters.ENGINE_DIR", "fg_make": "adapters.FG_DIR", "rebound": "adapters.RB_S1_MANIFEST",
            "clock": "clock_adapter_v3.CK_DIR", "rotation": "rotation_adapter.R2_S1_MANIFEST"}
    blob = json.dumps(src, default=str).replace("\\\\", "/").replace("\\", "/")
    for fam, key in want.items():
        if key not in ov:
            continue
        root = Path(ov[key])
        root = root.parent if root.suffix == ".json" else root
        tail = root.as_posix().split("/data/processed/models/")[-1] if "/data/processed/models/" in root.as_posix() else root.name
        if tail not in blob:
            bad.append(f"{fam}: override {root} not found in the run's recorded sources")
    return bad


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: run_engine_overlay_v1.py --overrides F --runner sample|full -- <runner args>")
    k = argv.index("--")
    own, rest = argv[:k], argv[k + 1:]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--overrides", type=Path, required=True)
    ap.add_argument("--runner", choices=["sample", "full"], required=True)
    a = ap.parse_args(own)
    ov = json.loads(a.overrides.read_text(encoding="utf-8")) if a.overrides.exists() else {}
    ov = {k: str((ROOT / v).resolve()) if not Path(v).is_absolute() else v for k, v in ov.items()}
    os.environ[ENV] = json.dumps(ov)
    apply_overrides()
    if a.runner == "sample":
        _R._init_worker = _init_sample
        import run_po4b_closed_loop_sample_v1 as S
        sys.argv = [sys.argv[0], *rest]
        rc = S.main()
    else:
        _RE._init_worker = _init_full
        sys.argv = [sys.argv[0], *rest]
        rc = _RE.main()
    tag = rest[rest.index("--tag") + 1]
    rd = Path(rest[rest.index("--results-dir") + 1]) if "--results-dir" in rest else ROOT / "results/engine_v0"
    mp = rd / tag / "run_meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
        meta["full_retrain_overrides"] = ov
        meta["full_retrain_overrides_file"] = str(a.overrides)
        bad = check_sources(meta, ov)
        meta["full_retrain_source_check"] = bad or "PASS"
        mp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        if bad:
            print("SOURCE CHECK FAILED:", *bad, sep="\n  ")
            return 5
        print("source check: PASS")
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
