"""build_engine_inputs_v3_combine_v1.py -- combine two tagged engine-input dirs that touch DISJOINT arrays (lane I, 2026-10-01).

`--team-from` (e.g. engine_v3_I_RBTO: team_static rebound columns, overlay/, anchor offsets) is copied whole; its
`slot_static` is then replaced by `--slot-from`'s (e.g. engine_v3_I_A1: anonymous-slot FT block). Asserted: against the
base `engine_v3`, the team dir differs only in team_static and the slot dir only in slot_static, so the union is exact.

    .venv/Scripts/python.exe scripts/build_engine_inputs_v3_combine_v1.py --team-from data/processed/models/engine_v3_I_RBTO \
        --slot-from data/processed/models/engine_v3_I_A1 --out data/processed/models/engine_v3_I_RBTOA1
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

TAG = "F2_2025"
ap = argparse.ArgumentParser()
ap.add_argument("--team-from", type=Path, required=True)
ap.add_argument("--slot-from", type=Path, required=True)
ap.add_argument("--base", type=Path, default=Path("data/processed/models/engine_v3"))
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
if a.out.exists():
    raise SystemExit(f"{a.out} exists: never overwrite")
z0, zt, zs = (dict(np.load(p / f"arrays_{TAG}.npz")) for p in (a.base, a.team_from, a.slot_from))
for k in z0:
    if k != "team_static":
        assert np.array_equal(z0[k], zt[k]), f"team dir differs in {k}"
    if k != "slot_static":
        assert np.array_equal(z0[k], zs[k]), f"slot dir differs in {k}"
shutil.copytree(a.team_from, a.out)
z = dict(zt)
z["slot_static"] = zs["slot_static"]
(a.out / f"arrays_{TAG}.npz").unlink()
np.savez_compressed(a.out / f"arrays_{TAG}.npz", **z)
rep = {"team_from": str(a.team_from), "slot_from": str(a.slot_from), "base": str(a.base),
       "team_static_changed_cols": int((~np.all(zt["team_static"] == z0["team_static"], axis=(0, 1))).sum()),
       "slot_static_changed_cells": int((zs["slot_static"] != z0["slot_static"]).sum())}
(a.out / "combine_report.json").write_text(json.dumps(rep, indent=1))
print(rep)
