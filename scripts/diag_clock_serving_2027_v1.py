#!/usr/bin/env python
"""diag_clock_serving_2027_v1.py -- seal-week check that the SERVED clock (adapters.CLOCK_DEFAULT, or ENGINE_CLOCK)
resolves a complete artifact set for a 2026-27 slate (served stack v3, 2026-10-07: `v5b_r8K2_glat_pmean`).

What 2026-27 serving uses (the same mechanism as every other dated-refit family, scripts/build_season_2027_artifacts_v1.py):
the clock mode's manifest path does not depend on the season, and the engine's selection rule (cbb_sim.engine.manifest:
latest refit_date strictly before tipoff, plus max_train_date < game_date) picks the LAST fold-2 refit (2025-04-01,
trained through 2025-03) for every 2026-27 game. That is a CARRY-FORWARD, NOT A REFIT through 2025-26; a 2026-27 refit
reads the sealed season and is a PM decision (retrain-set dimension), not this check.

Checks, read-only (no sealed data, no engine run): the mode is known; the S1 manifest, every pickle it names, the latent
params file and the V5 params file exist; the manifest selects an artifact for a game on --slate-date with both
honest-backtest checks passing. --pull first pulls a missing gitignored set from the HF `model_artifacts` key.

    .venv/Scripts/python.exe scripts/diag_clock_serving_2027_v1.py --slate-date 2026-11-02
    .venv/Scripts/python.exe scripts/diag_clock_serving_2027_v1.py --slate-date 2026-11-02 --pull
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
PY = str(REPO / ".venv/Scripts/python.exe") if (REPO / ".venv/Scripts/python.exe").exists() else sys.executable


def resolve(mode: str) -> dict:
    from cbb_sim.engine import clock_adapter_v3 as C3
    if mode not in C3.V5_MODES:
        raise SystemExit(f"FAIL: served clock {mode!r} is not a V5_MODES latent mode")
    spec = C3.V5_MODES[mode]
    base = C3.V3C_MODES[spec["base_mode"]]
    ck = Path(C3.CK_DIR)
    ck = ck if ck.is_absolute() else REPO / ck
    v5 = Path(C3.V5_PARAMS)
    return {"ck": ck, "manifest": ck / base["manifest"],
            "params": ck / spec.get("params_file", "v5_bakeoff/v5_bakeoff_report.json"),
            "v5_params": v5 if v5.is_absolute() else REPO / v5}


def check(mode: str, slate: str) -> tuple[list[str], list[str]]:
    import json
    import pandas as pd
    from cbb_sim.engine import clock_adapter_v3 as C3
    from cbb_sim.engine.manifest import ArtifactManifest
    r = resolve(mode)
    bad, info = [], []
    for k in ("manifest", "params", "v5_params"):
        if not r[k].exists():
            bad.append(f"missing {k}: {r[k].relative_to(REPO) if REPO in r[k].parents else r[k]}")
    if r["manifest"].exists():
        doc = json.loads(r["manifest"].read_text(encoding="utf-8"))
        miss = [m["model_file"] for m in doc.get("months", []) if not (r["ck"] / m["model_file"]).exists()]
        if miss:
            bad.append(f"manifest names {len(miss)} missing pickle(s): {miss[:3]}")
        else:
            g = pd.DataFrame({"game_id": [0], "game_date": [pd.Timestamp(slate)]})
            man = ArtifactManifest.from_obj(C3._manifest_obj(r["manifest"]), r["ck"], g)
            e = man.entries[int(man.seg_of_game[0])]
            info.append(f"{mode}: {len(man.entries)} refits; a {slate} game is served by refit {e.refit_date.date()} "
                        f"(max_train_date {e.max_train_date.date()}; fold {doc.get('fold')} season {doc.get('season')}) "
                        f"-- CARRIED FORWARD, not refit through 2025-26")
    return bad, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slate-date", default="2026-11-02")
    ap.add_argument("--mode", default=None, help="default: ENGINE_CLOCK, else adapters.CLOCK_DEFAULT")
    ap.add_argument("--pull", action="store_true", help="pull a missing set from HF model_artifacts first")
    a = ap.parse_args(argv)
    from cbb_sim.engine import adapters as AD
    mode = a.mode or os.environ.get("ENGINE_CLOCK") or AD.CLOCK_DEFAULT
    bad, info = check(mode, a.slate_date)
    if bad and a.pull:
        r = resolve(mode)
        sub = Path(os.path.relpath(r["manifest"].parent.parent, REPO / "data/processed/models")).as_posix()
        cmd = [PY, "scripts/hf_sync_data.py", "pull", "--dirs", "model_artifacts", "--only", f"{sub}/**"]
        print("pulling: " + " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc != 0:
            print(f"FAIL: pull exit {rc}")
            return rc
        bad, info = check(mode, a.slate_date)
    for x in info:
        print(x)
    for x in bad:
        print("FAIL: " + x)
    print(f"clock serving {mode}: {'OK' if not bad else 'FAIL'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
