"""build_fg_two_stage_serving_v1.py -- lane A 2026-10-01: make a two-stage fg_make arm servable by the engine.

1. Re-exports the arm's dated joblibs with the model object as `cbb_sim.engine.fg_two_stage.TwoStageModelR` (same
   fitted trees; the trainer pickled the scripts-side twin) into data/processed/models/fg_make/round_tsr_served/<name>/
   with the manifests copied, so `ENGINE_FG_MAKE=g9ts_<name>` resolves through the engine's dated fg loader.
2. Builds the versioned input-dir sibling data/processed/models/engine_v3_<name>_laneA/: engine_v3's files with
   team_static extended by `fg_team_offset__{rim,jump2,three}` and `lg_make_asof__{rim,jump2,three}` (from the arm's
   offsets_F2_s<seed>.parquet, computed pre-game by the trainer from games before tip). engine_v3 is not modified.

    .venv/Scripts/python.exe scripts/build_fg_two_stage_serving_v1.py --arm-dir data/processed/models/fg_make/round_tsr/TSR_s0_F2 --name TSR
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cbb_sim.engine.fg_two_stage import TwoStageModelR  # noqa: E402

KEYS = ("rim", "jump2", "three")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", type=Path, default=ROOT / "data/processed/models/engine_v3")
    a = ap.parse_args()
    src_b1 = a.arm_dir / "B1"
    dst = ROOT / "data/processed/models/fg_make/round_tsr_served" / a.name
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for m in sorted(src_b1.glob("manifest_*.json")):
        shutil.copy2(m, dst / m.name)
        for art in json.loads(m.read_text(encoding="utf-8"))["artifacts"]:
            w = joblib.load(src_b1 / art["path"])
            w["model"] = TwoStageModelR(w["model"].clf_)
            joblib.dump(w, dst / art["path"])
            n += 1
    # input-dir sibling
    out = ROOT / f"data/processed/models/engine_v3_{a.name}_laneA"
    out.mkdir(parents=True, exist_ok=True)
    for f in a.base.iterdir():
        if f.is_file() and not f.name.startswith(("arrays_", "names_")):
            shutil.copy2(f, out / f.name)
    z = dict(np.load(a.base / "arrays_F2_2025.npz"))
    names = json.loads((a.base / "names_F2_2025.json").read_text(encoding="utf-8"))
    games = pd.read_parquet(a.base / "games_F2_2025.parquet")
    gpos = pd.Series(np.arange(len(games)), index=games["game_id"].to_numpy())
    off = pd.read_parquet(sorted(a.arm_dir.glob("offsets_F2_s*.parquet"))[0])
    ts = z["team_static"]
    add, tn = [], dict(names["team_names"])
    for col in ("fg_team_offset", "lg_make_asof"):
        for k in KEYS:
            arr = np.full((ts.shape[0], 2), np.nan)
            o = off[off["key"] == k]
            arr[gpos.reindex(o["game_id"]).to_numpy(), o["side"].to_numpy()] = o[col].to_numpy(float)
            if np.isnan(arr).any():
                raise SystemExit(f"{col}__{k}: {int(np.isnan(arr).sum())} cells missing")
            tn[f"{col}__{k}"] = ts.shape[2] + len(add)
            add.append(arr)
    z["team_static"] = np.concatenate([ts, np.stack(add, axis=2).astype(ts.dtype)], axis=2)
    names["team_names"] = tn
    np.savez(out / "arrays_F2_2025.npz", **z)
    (out / "names_F2_2025.json").write_text(json.dumps(names), encoding="utf-8")
    (out / "builder_report.json").write_text(json.dumps({
        "builder": "build_fg_two_stage_serving_v1.py", "arm_dir": str(a.arm_dir), "base": str(a.base),
        "added_team_columns": [k for k in tn if k not in names["team_names"] or k.startswith(("fg_team", "lg_make"))],
        "n_artifacts_reexported": n}, indent=1), encoding="utf-8")
    print(f"re-exported {n} artifacts to {dst}; input dir {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
