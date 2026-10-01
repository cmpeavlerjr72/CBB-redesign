"""run_engine_overlay_v2.py -- run an UNEDITED engine runner with module attributes re-pointed in-process, for reads
on another FOLD's artifacts (lane D, 2026-10-01; fold-1 confirmation, docs/models/engine/experiments.md section 2).

Versioned sibling of `run_engine_overlay_v1.py` (NOT edited). v1 rebinds path attributes of three modules; a fold-1
read also needs the four adopted loop-level modules (their tables are module-level paths / stems keyed "F2") and the
clock's fitted-parameter fold key. v2 widens the overlay, nothing else:

  overrides.json:  {"<module>.<attr>": <value>, ...}
    value a string             -> a Path (as v1)
    value {"str": s}           -> the plain string s (e.g. clock_adapter_v3.PARAMS_FOLD = "F1")
    value {"path_str": s}      -> the string of a repo-resolved path (chance_time.LUT3 is formatted with .format(fold=))
    value {"items": {k: v}}    -> dict item updates on an existing dict attribute (JSON lists become tuples)
  modules: adapters, clock_adapter_v3, rotation_adapter, shot_block, foul_joint, foul_r9, shared_shooting, chance_time

The overrides travel in an env var and are applied in the parent AND in every worker (the worker initializer is
wrapped, as in v1). After the run the overrides are stamped into run_meta.json, and every overridden artifact family
must report a source under its override (v1's check) or the run FAILS.

    python scripts/run_engine_overlay_v2.py --overrides <f1>/overrides_V2.json --runner full -- \
        --fold F1 --season 2024 --input-dir <f1 inputs> --seeds 25 --workers 4 --tag <tag> --results-dir results/engine_v0
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
ENV = "CBB_OVERLAY_V2"
MODS = {"adapters": "cbb_sim.engine.adapters", "clock_adapter_v3": "cbb_sim.engine.clock_adapter_v3",
        "rotation_adapter": "cbb_sim.engine.rotation_adapter", "shot_block": "cbb_sim.engine.shot_block",
        "foul_joint": "cbb_sim.engine.foul_joint", "foul_r9": "cbb_sim.engine.foul_r9",
        "shared_shooting": "cbb_sim.engine.shared_shooting", "chance_time": "cbb_sim.engine.chance_time"}


def _resolve(v: str) -> str:
    return str((ROOT / v).resolve()) if not Path(v).is_absolute() else v


def apply_overrides() -> dict:
    ov = json.loads(os.environ.get(ENV, "{}"))
    for k, v in ov.items():
        mod, attr = k.split(".", 1)
        m = importlib.import_module(MODS[mod])
        if not hasattr(m, attr):
            raise SystemExit(f"override {k}: {MODS[mod]} has no attribute {attr}")
        if isinstance(v, str):
            setattr(m, attr, Path(_resolve(v)))
        elif "str" in v:
            setattr(m, attr, v["str"])
        elif "path_str" in v:
            setattr(m, attr, _resolve(v["path_str"]))
        elif "items" in v:
            d = getattr(m, attr)
            for kk, vv in v["items"].items():
                d[kk] = tuple(vv) if isinstance(vv, list) else vv
        else:
            raise SystemExit(f"override {k}: unknown value form {v!r}")
    return ov


import run_po4b_closed_loop as _R  # noqa: E402
import run_engine as _RE  # noqa: E402

_ORIG = {"sample": _R._init_worker, "full": _RE._init_worker}


def _stamp(ov: dict) -> None:
    from cbb_sim.engine import adapters as AD
    print(f"[worker {os.getpid()}] overlay v2 applied ({len(ov)}): ENGINE_DIR={AD.ENGINE_DIR} FG_DIR={AD.FG_DIR}",
          flush=True)


def _init_sample(*args):
    _stamp(apply_overrides())
    return _ORIG["sample"](*args)


def _init_full(*args):
    _stamp(apply_overrides())
    return _ORIG["full"](*args)


def check_sources(meta: dict, ov: dict) -> list[str]:
    src = (meta.get("adapter_flags") or {}).get("sources") or {}
    blob = json.dumps(src, default=str).replace("\\\\", "/").replace("\\", "/")
    want = {"event": "adapters.ENGINE_DIR", "fg_make": "adapters.FG_DIR", "rebound": "adapters.RB_S1_MANIFEST",
            "free_throw": "adapters.FT_S1_MANIFEST", "clock": "clock_adapter_v3.CK_DIR",
            "rotation": "rotation_adapter.R2_S1_MANIFEST"}
    bad = []
    for fam, key in want.items():
        if key not in ov:
            continue
        root = Path(_resolve(ov[key]))
        root = root.parent if root.suffix == ".json" else root
        tail = root.as_posix().split("/data/processed/models/")[-1]
        if tail not in blob:
            bad.append(f"{fam}: override {root} not found in the run's recorded sources")
    return bad


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: run_engine_overlay_v2.py --overrides F --runner sample|full -- <runner args>")
    k = argv.index("--")
    own, rest = argv[:k], argv[k + 1:]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--overrides", type=Path, required=True)
    ap.add_argument("--runner", choices=["sample", "full"], required=True)
    a = ap.parse_args(own)
    ov = json.loads(a.overrides.read_text(encoding="utf-8"))
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
        meta["overlay_v2_overrides"] = ov
        meta["overlay_v2_overrides_file"] = str(a.overrides)
        bad = check_sources(meta, ov)
        meta["overlay_v2_source_check"] = bad or "PASS"
        mp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        if bad:
            print("SOURCE CHECK FAILED:", *bad, sep="\n  ")
            return 5
        print("source check: PASS")
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
