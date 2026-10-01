"""train_rotation_v3b_s1_poss_v1.py -- the served rotation S1 schedule (R2 + scheduler, round 3b) refit on a chosen
possessions version, into a NEW directory.

Versioned sibling of `scripts/train_rotation_v3b_s1.py` (NOT edited). That trainer reads possessions through
`train_rotation_v1.load_season -> rotation.load_team_possessions(season)` whose `poss_dir` default is bound to
`data/processed/possessions` (v1) at definition time, and writes into the served `data/processed/models/rotation/`.
This wrapper changes exactly two things, both in THIS process only:
  * `rotation.load_team_possessions` -> the same function with `poss_dir=<possessions_dir(--poss-version)>`;
  * the trainer's `OUT_DIR` -> `--out-dir` (a NEW directory; the static first-window fit
    `rotation_fit_v3.json` is copied in from the served dir first, because the trainer reads it from OUT_DIR).
and then calls the trainer's own `main()` with its own CLI (pass-through after `--`).
It also writes `rotation_r2_s1_F2_2025.json`, the engine manifest, with paths relative to its own directory.

SCOPE (stated, not hidden): the FIRST window reuses the static `rotation_fit_v3.json` exactly as the served
schedule does, and that static fit (plus the engine-input priors baked from `rotation_fit.json` by
`build_engine_inputs.py`) stays on the v1 possession table. Refitting the static chain
(`train_rotation_v1.py` -> `train_rotation_v3.py`) on v4 is NOT done here (rotation is PARKED; open item).

Identity: `--poss-version v1 --max-windows 2` must reproduce the served `rotation_fit_v3_S1_<second month>.json`.

    .venv/Scripts/python.exe scripts/train_rotation_v3b_s1_poss_v1.py --poss-version v4 --out-dir <root>/rotation
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
SERVED = ROOT / "data/processed/models/rotation"
SERVED_MANIFEST = ROOT / "data/processed/models/engine/rotation_r2_s1_F2_2025.json"


def write_manifest(out_dir: Path, partial: bool = False) -> Path:
    """The served engine manifest, re-pointed at this directory's fits (same entries, same dates).
    `partial` (smoke / --max-windows only): keep the entries whose fit exists; later games use the last one."""
    obj = json.loads(SERVED_MANIFEST.read_text(encoding="utf-8"))
    arts = obj["artifacts"]
    keep = []
    for e in arts:
        name = Path(e["path"]).name
        if not (out_dir / name).exists():
            if partial:
                continue
            raise SystemExit(f"{name} missing in {out_dir}; the S1 run did not finish")
        e["path"] = name
        keep.append(e)
    obj["artifacts"] = keep
    if partial:
        obj["note_partial"] = "SMOKE / --max-windows: not every S1 window was fitted"
    obj["note_full_retrain"] = "train_rotation_v3b_s1_poss_v1.py; paths relative to this file"
    p = out_dir / SERVED_MANIFEST.name
    p.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    return p


def main() -> int:
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        k = argv.index("--")
        argv, rest = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--poss-version", default="v4", choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-windows", type=int, default=0, help="identity / smoke only: first N S1 windows")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--only-window", default="",
                    help="YYYYMM: fit ONLY this S1 window (plus the static first window, a copy), then stop; "
                         "lets the chain run the windows in parallel. Implies --fits-only")
    ap.add_argument("--fits-only", action="store_true",
                    help="stop as soon as the last needed window fit is written (skip the trainer's evaluation sims)")
    a = ap.parse_args(argv)
    out = a.out_dir.resolve()
    if out == SERVED.resolve():
        raise SystemExit("refusing to write into the served rotation dir")
    out.mkdir(parents=True, exist_ok=True)
    if a.manifest_only:
        print(write_manifest(out))
        return 0
    shutil.copyfile(SERVED / "rotation_fit_v3.json", out / "rotation_fit_v3.json")

    from cbb_sim.models import rotation as R
    from cbb_sim.pbp.possessions import possessions_dir
    R.load_team_possessions = functools.partial(R.load_team_possessions,
                                                poss_dir=ROOT / possessions_dir(a.poss_version))
    import train_rotation_v3b_s1 as T
    T.OUT_DIR = out
    orig = T.month_windows
    want: set[str] = set()

    def _windows(d):
        ms = orig(d)
        if a.max_windows:
            ms = ms[: a.max_windows]
        if a.only_window:
            ms = [m for i, m in enumerate(ms) if i == 0 or m.strftime("%Y%m") == a.only_window]
            if len(ms) < 2:
                raise SystemExit(f"--only-window {a.only_window} is not an S1 window of this run")
        want.update(m.strftime("%Y%m") for m in ms)
        return ms

    T.month_windows = _windows

    class _FitsDone(Exception):
        pass

    if a.fits_only or a.only_window:
        written: set[str] = set()
        orig_to_json = R.RotationFit.to_json

        def _to_json(self, path, *args, **kw):
            r = orig_to_json(self, path, *args, **kw)
            name = Path(path).name
            if name.startswith("rotation_fit_v3_S1_"):
                written.add(name[len("rotation_fit_v3_S1_"):-len(".json")])
                if want and want <= written:
                    raise _FitsDone()
            return r

        R.RotationFit.to_json = _to_json
    t0 = time.time()
    sys.argv = [sys.argv[0], *rest]
    try:
        T.main()
    except _FitsDone:
        print(f"fits written for windows {sorted(want)}; evaluation sims skipped (--fits-only)", flush=True)
    rep = {"poss_version": a.poss_version, "out_dir": str(out), "seconds": round(time.time() - t0, 1),
           "max_windows": a.max_windows, "trainer_args": rest}
    if not a.only_window:
        rep["manifest"] = str(write_manifest(out, partial=bool(a.max_windows)))
    if a.identity:
        res = {}
        for f in sorted(out.glob("rotation_fit_v3_S1_*.json")):
            res[f.name] = (f.read_bytes() == (SERVED / f.name).read_bytes())
            if not res[f.name]:
                x, y = json.loads(f.read_text()), json.loads((SERVED / f.name).read_text())
                res[f.name + "_json_equal"] = (x == y)
        rep["identity"] = res
    (out / "rotation_chain_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
