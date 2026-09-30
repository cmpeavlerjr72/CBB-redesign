"""ops_build_tfs_table_v1.py -- operator 2026-09-30. Builds the ONE feature table for arm Tfs
(T + corrected in_bonus), because train_possession_outcome_s1_par_v1.py refuses --team-rate-table together
with --feature-table (mutually exclusive) and the trainer is not edited.

Tfs = arm T (E3 v4 through cbb_sim.team_rate_adapter.apply, missing='raise') plus the corrected `in_bonus`
overlay. This script applies the adapter to the round-2 design exactly as the trainer does, takes the columns
the adapter changed (the 8 PO style columns), adds the overlay's `in_bonus`, and writes a sibling table keyed
game_id,poss_index,chance_number. The trainer's own --feature-table overlay then recomputes the x_ interaction
columns (float32 product) from the overlaid style columns, the same rule the adapter uses.

Verification printed: (a) the overlay applied to the ORIGINAL design reproduces the adapter's frame on every
overlaid and interaction column (bit-equal), (b) in_bonus equals the round-8 overlay.

    python scripts/ops_build_tfs_table_v1.py --table data/processed/team_rate_features_E3_v4.parquet \
        --overlay data/processed/models/possession_outcome/round8/in_bonus_overlay_v2state_v1.parquet \
        --out data/processed/models/possession_outcome/round_stageb/tfs_feature_table_v1.parquet
Run inside the image via scripts/box_run.sh.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import train_par_common_v1 as C  # noqa: E402
from cbb_sim.team_rate_adapter import apply  # noqa: E402

KEYS = ["game_id", "poss_index", "chance_number"]

ap = argparse.ArgumentParser()
ap.add_argument("--design", default="data/processed/models/possession_outcome/round2/design.parquet")
ap.add_argument("--table", required=True)
ap.add_argument("--overlay", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fold", default="F2")
a = ap.parse_args()

design = pd.read_parquet(a.design)
adapted = apply(design.copy(), a.table, "possession_outcome", fold=a.fold, missing="raise")
changed = [c for c in design.columns
           if c not in KEYS and not c.startswith("x_")
           and not design[c].equals(adapted[c])]
print("style columns changed by the adapter:", changed)
ov = pd.read_parquet(a.overlay)
assert not ov.duplicated(KEYS).any() and not design.duplicated(KEYS).any()
tab = adapted[KEYS + changed].merge(ov[KEYS + ["in_bonus"]], on=KEYS, how="left", validate="one_to_one")
assert len(tab) == len(design) and tab["in_bonus"].notna().all()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
tab.to_parquet(a.out, index=False)

# verification
d2, rep = C.overlay_columns(design, Path(a.out), keys=KEYS, cols=[*changed, "in_bonus"])
for c in changed + list(rep["interactions_recomputed"]):
    assert d2[c].equals(adapted[c]), c
d3, _ = C.overlay_columns(design, Path(a.overlay), keys=KEYS, cols=["in_bonus"])
assert d2["in_bonus"].equals(d3["in_bonus"])
print("VERIFIED: overlay == adapter on", len(changed), "style +", len(rep["interactions_recomputed"]),
      "interaction columns; in_bonus == round-8 overlay;", rep["n_unmatched_rows"], "unmatched; rows", len(tab))
