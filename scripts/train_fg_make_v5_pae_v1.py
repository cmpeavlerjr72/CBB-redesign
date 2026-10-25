"""train_fg_make_v5_pae_v1.py -- fg_make PAE round (experiments.md section 31): the served B1 structure
(the lane-A reference `aR`, sections 23-30) with the shot-location PAE team ratings as extra features.

Versioned wrapper of `train_fg_make_v4_par_g9_v1.py` (which wraps `train_fg_make_v4_par_v1.py`); NOTHING
existing is edited. What this wrapper adds, and nothing else:

  --arm R | P1 | P2   R  = the served B1 spec, unchanged (control; run under --seed 1 it is R_seed, the floor)
                      P1 = R + pae_off_c (shooting team's offence PAE) + pae_def_c (defending team's PAE allowed)
                      P2 = P1 + cell_p (the attempt's own prior-season cell make rate where located, else the
                           prior-season class mean) + loc_missing (1 when the attempt has no coordinate)
  --pae-cols PATH     the row-aligned table written by `build_pae_shot_rating_v1.py --part build`
                      (default data/processed/models/fg_make/pae_v1/design_pae_cols_v1.parquet). It is attached
                      to the design BY POSITION after asserting game_id / shooter_id / shot_class / made are equal
                      row for row, so a misaligned table raises instead of joining silently.

Every other option is passed through to the g9 wrapper (`--fold`, `--seed`, `--out-dir`, `--n-jobs`, `--no-floor`,
`--no-leak`, `--arms B1`). The PAE columns are attached for EVERY arm (R does not read them), so the design frame is
byte-identical across arms except for the feature list.

    .venv/Scripts/python.exe scripts/train_fg_make_v5_pae_v1.py --arm P1 --fold F2 --seed 0 --arms B1 \
        --no-floor --no-leak --n-jobs 8 --out-dir data/processed/models/fg_make/round_pae/P1_s0_F2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

ARM_FEATURES = {"R": [], "P1": ["pae_off_c", "pae_def_c"],
                "P2": ["pae_off_c", "pae_def_c", "cell_p", "loc_missing"]}
PAE_COLS = ROOT / "data/processed/models/fg_make/pae_v1/design_pae_cols_v1.parquet"
ATTACH = ["pae_off_c", "pae_def_c", "pae_off_v", "pae_def_v", "pae_off_n", "pae_def_n", "cell_p",
          "loc_missing", "located"]

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--arm", choices=list(ARM_FEATURES), required=True)
_ap.add_argument("--pae-cols", type=Path, default=PAE_COLS)
OWN, REST = _ap.parse_known_args()
if any(x.startswith("--extra-features") for x in REST):
    raise SystemExit("--extra-features is set by --arm in this wrapper")
_extra = ARM_FEATURES[OWN.arm]
sys.argv = [sys.argv[0], *REST, *([f"--extra-features={','.join(_extra)}"] if _extra else [])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import train_fg_make_v4_par_g9_v1 as G9  # noqa: E402  (parses sys.argv at import, by design)
import train_fg_make_v4_par_v1 as P  # noqa: E402

_orig_load_design = P.load_design


def _load_design_pae(a, t0):
    design, rep = _orig_load_design(a, t0)
    cols = pd.read_parquet(OWN.pae_cols)
    if len(cols) != len(design):
        raise SystemExit(f"pae cols {len(cols)} rows vs design {len(design)}")
    for k in ("game_id", "shooter_id", "shot_class", "made"):
        if not (cols[k].to_numpy() == design[k].to_numpy()).all():
            raise SystemExit(f"pae cols misaligned with the design on {k}")
    for c in ATTACH:
        design[c] = cols[c].to_numpy(dtype="float32") if c != "located" else cols[c].to_numpy()
    for c in _extra:
        if not np.isfinite(design[c].to_numpy(dtype="float64")).all():
            raise SystemExit(f"feature {c} has non-finite values")
    print(f"pae wrapper: arm {OWN.arm}, extra features {_extra}, cols {OWN.pae_cols}", flush=True)
    return design, rep


P.load_design = _load_design_pae

if __name__ == "__main__":
    raise SystemExit(G9.main())
