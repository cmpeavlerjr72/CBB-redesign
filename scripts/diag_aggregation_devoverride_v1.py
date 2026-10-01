"""diag_aggregation_devoverride_v1.py -- lane A 2026-09-30, addendum F2: write the training design's
fold-2 shooter_shrunk_dev_c per (game, shooter, class) for the T design (E3 overlay + local E3 cache)."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import train_fg_make_v4_shooter_block as R4M  # noqa: E402
from cbb_sim.team_rate_adapter import apply as trt_apply  # noqa: E402
d = trt_apply(pd.read_parquet(R4M.DESIGN), ROOT / "data/processed/team_rate_features_E3_v4.parquet", "fg_make", fold="F2", missing="raise")
extra = pd.read_parquet(ROOT / "data/processed/models/fg_make/round_aggfix/design_v4_extra_E3_local_g0.parquet")
d["shooter_shrunk_dev_c"] = extra["shooter_shrunk_dev_c"].to_numpy()
d = d[d["season"] == 2025]
o = d.groupby(["game_id", "shooter_id", "shot_class"], as_index=False)["shooter_shrunk_dev_c"].first()
o.to_parquet(ROOT / "results/aggregation_v1/devoverride_T_v1.parquet", index=False)
print(len(o))
