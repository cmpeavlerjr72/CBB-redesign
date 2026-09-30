#!/usr/bin/env python
"""Lane H: which data/reference/gate_targets_{season}.parquet game-level rows are contaminated by
unplayed / forfeit games (seasons 2022-2025 only; 2026 sealed, not computed). Compares the stored
season/all rows with a recompute from the D-I non-truncated universe as built (must match) and
with the unverified-final games removed (`reference.unverified_final_game_ids`)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import reference as R
u = pd.read_parquet(ROOT / "data/processed/games_universe.parquet")
rows = []
for s in (2022, 2023, 2024, 2025):
    ref = R.load_gate_targets(s)
    base = u[(u.season == s) & u.is_d1_game & ~u.pbp_truncated]
    bad = R.unverified_final_game_ids(s)
    clean = base[~base.game_id.isin(bad)]
    def stats(g):
        m = g.home_score - g.away_score; t = g.home_score + g.away_score
        nn = g[~g.neutral_site]; n = g[g.neutral_site]
        return {"n_games": len(g), "total_points_mean": t.mean(), "total_points_sd": t.std(), "margin_sd": m.std(),
                "home_away_score_corr": g.home_score.corr(g.away_score),
                "home_margin_mean_nonneutral": (nn.home_score - nn.away_score).mean(),
                "home_margin_sd_nonneutral": (nn.home_score - nn.away_score).std(),
                "home_margin_mean_neutral": (n.home_score - n.away_score).mean()}
    a, b = stats(base), stats(clean)
    for k in a:
        stored = np.nan
        if k != "n_games":
            m = ref[(ref.breakdown == "season") & (ref.metric == k)]
            stored = float(m.value.iloc[0]) if len(m) else np.nan
        rows.append({"season": s, "metric": k, "stored": stored, "recomputed_as_built": a[k], "unverified_removed": b[k],
                     "n_unverified_in_base": len(base) - len(clean)})
    # month rows affected
    bm = set(pd.to_datetime(base[base.game_id.isin(bad)].game_date).dt.month)
    rows.append({"season": s, "metric": "months_touched", "stored": np.nan, "recomputed_as_built": len(bm), "unverified_removed": np.nan, "n_unverified_in_base": len(base) - len(clean)})
out = pd.DataFrame(rows)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
out.to_csv(ROOT / "data/processed/truth/truth_gate_targets_check_v1.csv", index=False)
