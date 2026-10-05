#!/usr/bin/env python
"""
build_shot_block_lut_inputs_v1.py -- the K2_Ocell shot-block table for ANY fold-2 inputs directory (lane F, 2026-10-01).

The shot-block table is slot-keyed (`shooter`, `known` follow the inputs' roster slots), so an inputs sibling with other roster
ids needs its own table. Same model, anchors and team arrays as the served table (`build_shot_block_lut_live_v1.build_table`, full-season
events, per-date as-of); only the slot-keyed arrays follow the new roster. Writes `<inputs-dir>/shot_block_K2_Ocell_<name>_F2_2025.npz`
and registers it in the inputs' names json as `meta["shot_block_lut"]` for the arms `K2_Ocell` and `K2_Ocell_v3in`, which the engine
already reads (`shot_block.ShotBlock`: live table named in inp.meta wins over the tag lookup). No engine change.

    python scripts/build_shot_block_lut_inputs_v1.py --inputs-dir data/processed/models/engine_v3_seed --name seed
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import build_shot_block_lut_live_v1 as SBL
from cbb_sim.engine.inputs import EngineInputs

ap = argparse.ArgumentParser()
ap.add_argument("--inputs-dir", type=Path, required=True)
ap.add_argument("--name", required=True)
a = ap.parse_args()
d = a.inputs_dir if a.inputs_dir.is_absolute() else ROOT / a.inputs_dir
inp = EngineInputs.load(d, "F2_2025")
tab = SBL.build_table(inp, "K2_Ocell", as_of=None)
out = d / f"shot_block_K2_Ocell_{a.name}_F2_2025.npz"
np.savez_compressed(out, roster_cbbd=inp.roster_cbbd, **tab)
rel = out.relative_to(ROOT).as_posix()
nj = d / "names_F2_2025.json"
names = json.loads(nj.read_text(encoding="utf-8"))
names.setdefault("meta", {})["shot_block_lut"] = {"K2_Ocell": rel, "K2_Ocell_v3in": rel}
nj.write_text(json.dumps(names, indent=1, default=str), encoding="utf-8")
v3in = np.load(ROOT / "data/processed/models/engine/shot_block_K2_Ocell_v3in_F2_2025.npz")
same_roster = np.array_equal(inp.roster_cbbd, np.load(ROOT / "data/processed/models/engine_v3/arrays_F2_2025.npz")["roster_cbbd"])
chg = (inp.roster_cbbd != np.load(ROOT / "data/processed/models/engine_v3/arrays_F2_2025.npz")["roster_cbbd"]).any(axis=(1, 2))
rep = {"table": rel, "roster_equals_v3": bool(same_roster), "games_with_changed_roster": int(chg.sum())}
for k in ("team", "anchor", "shooter", "known"):
    x, y = np.asarray(tab[k]), np.asarray(v3in[k])
    rep[f"{k}_unchanged_games_equal_v3in"] = bool(np.array_equal(x[~chg], y[~chg]))
    rep[f"{k}_changed_games_max_abs_diff"] = float(np.abs(x[chg].astype(float) - y[chg].astype(float)).max()) if chg.any() else 0.0
print(json.dumps(rep, indent=1))
(ROOT / "results/shot_block_inputs_table_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
