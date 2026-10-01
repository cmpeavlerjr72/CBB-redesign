"""build_engine_inputs_v3_ftbio_v1.py -- tagged engine inputs that carry the free-throw bio columns of arm N1 / N3
(free_throw experiments.md section 13.2), lane I 2026-10-01.

Copies `engine_v3` (read, never written) into `data/processed/models/engine_v3_<tag>/` and APPENDS two slot columns,
`height_c` and `d1_years`, built exactly as `train_free_throw_v3_newcomer.add_bio` builds them (CBBD 2025 season roster,
by the slot's CBBD player id; unknown or anonymous slot = NaN). No existing column changes (asserted), so every other
sub-model's plan reads the same values. `pos_G/F/C` are NOT rebuilt (the engine already carries them from the fg_make
design); their agreement with the trainer's roster mapping is reported (train/serve parity).

    .venv/Scripts/python.exe scripts/build_engine_inputs_v3_ftbio_v1.py --tag I_N3 [--base-dir data/processed/models/engine_v3]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAG = "F2_2025"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base-dir", type=Path, default=ROOT / "data/processed/models/engine_v3")
    a = ap.parse_args()
    base = a.base_dir
    out = ROOT / "data/processed/models" / f"engine_v3_{a.tag}"
    if out.exists():
        raise SystemExit(f"{out} exists: never overwrite")
    z = dict(np.load(base / f"arrays_{TAG}.npz"))
    names = json.loads((base / f"names_{TAG}.json").read_text(encoding="utf-8"))
    sn = dict(names["slot_names"])
    ros = z["roster_cbbd"]
    r = pd.read_parquet(ROOT / "data/raw/cbbd/rosters/roster_2025.parquet",
                        columns=["season", "cbbd_player_id", "position", "height", "start_season"])
    r = r.dropna(subset=["cbbd_player_id"])
    r["cbbd_player_id"] = r["cbbd_player_id"].astype("int64")
    r = r.drop_duplicates(["season", "cbbd_player_id"])
    r["height_c"] = r["height"] - r["height"].mean()
    r["d1_years"] = np.clip(r["season"] - r["start_season"], 0, 4)
    r.loc[r["start_season"].isna(), "d1_years"] = np.nan
    p = r["position"].fillna("").str.upper().str[0]
    for k in ("G", "F", "C"):
        r[f"tpos_{k}"] = (p == k).astype("float32")
    r = r.set_index("cbbd_player_id")
    flat = ros.reshape(-1)
    new = {}
    for c in ("height_c", "d1_years"):
        v = pd.Series(flat).map(r[c]).to_numpy(dtype="float64").copy()
        v[flat <= 0] = np.nan
        new[c] = v.astype(np.float32).reshape(ros.shape)
    sl = z["slot_static"]
    add = np.stack([new["height_c"], new["d1_years"]], axis=-1)
    sl2 = np.concatenate([sl, add], axis=-1)
    assert np.array_equal(sl2[..., :sl.shape[-1]], sl)
    for c in ("height_c", "d1_years"):
        if c in sn:
            raise SystemExit(f"slot column {c} already exists")
        sn[c] = len(sn)
    names2 = dict(names); names2["slot_names"] = sn
    out.mkdir(parents=True)
    for f in (f"games_{TAG}.parquet", f"event_block_{TAG}.npz"):
        shutil.copy2(base / f, out / f)
    (out / f"names_{TAG}.json").write_text(json.dumps(names2, indent=1), encoding="utf-8")
    z2 = dict(z); z2["slot_static"] = sl2
    np.savez_compressed(out / f"arrays_{TAG}.npz", **z2)
    named = flat > 0
    rep = {"base": str(base), "out": str(out), "added_slot_columns": ["height_c", "d1_years"],
           "named_slots": int(named.sum()),
           "height_found_share_named": float(np.isfinite(new["height_c"].reshape(-1)[named]).mean()),
           "d1_found_share_named": float(np.isfinite(new["d1_years"].reshape(-1)[named]).mean())}
    for k in ("G", "F", "C"):
        tv = pd.Series(flat[named]).map(r[f"tpos_{k}"]).fillna(0).to_numpy()
        ev = sl[..., sn[f"pos_{k}"]].reshape(-1)[named]
        rep[f"pos_{k}_agreement_named"] = float(np.mean(np.isclose(tv, ev)))
    (out / "builder_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
