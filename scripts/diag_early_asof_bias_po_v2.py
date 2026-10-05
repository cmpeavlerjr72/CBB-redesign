"""Phase 1c (diagnostic, nothing fitted): downstream effect of SHRUNK as-of team features on the possession_outcome
`first` early-season class level, reusing existing retrains (no new fit).

Paired, same rows (fold 2, test 2025, `design_combined` of the 2026-10-01 full retrain):
  FR = served S1 lgbm retrained on the served expanding-mean features (`full_retrain_v1/FR_box_v1`)
  FT = the same retrain on team_rate_estimator E3 v4 features (shrunk, calibrated; `full_retrain_v1/FT_box_v2`)
Lines: TOV and FT-trip (shooting + bonus) level gap (mean p - realised, pp) and multiclass log loss, by days bucket
and by as-of sample cell; paired (FT - FR) with a game-block bootstrap SE. Fold 1 has no FT retrain (not readable).
Output: results/early_asof_bias/phase1c_po_FRvFT_v1.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "data/processed/models/full_retrain_v1"
ARMS = {"FR": (B / "FR_box_v1/po_train/out/design_combined.parquet", B / "FR_box_v1/po_train/out/pred_first_F2.npy"),
        "FT": (B / "FT_box_v2/po_train/out/team_rate_features_E3_v4/design_combined.parquet",
               B / "FT_box_v2/po_train/out/team_rate_features_E3_v4/pred_first_F2.npy")}
OUT = ROOT / "results/early_asof_bias"
RNG = np.random.default_rng(11)


def main():
    cols = ["season", "population", "y", "days_since_start", "game_id", "offense_team_id", "defense_team_id",
            "n_prior_off"]
    te = pd.read_parquet(ARMS["FR"][0], columns=cols)
    te = te[(te["season"] == 2025) & (te["population"] == "first")].reset_index(drop=True)
    t2 = pd.read_parquet(ARMS["FT"][0], columns=["season", "population", "y", "game_id"])
    t2 = t2[(t2["season"] == 2025) & (t2["population"] == "first")]
    assert (t2["game_id"].to_numpy() == te["game_id"].to_numpy()).all() and (t2["y"].to_numpy() == te["y"].to_numpy()).all()
    P = {a: np.load(p).astype("float64") for a, (_, p) in ARMS.items()}
    for a in P:
        assert P[a].shape == (len(te), 6)
    npo = te.groupby(["game_id", "offense_team_id"])["n_prior_off"].first()
    te["n_def"] = npo.reindex(pd.MultiIndex.from_arrays([te["game_id"], te["defense_team_id"]])).fillna(0).to_numpy()
    nmin = np.fmin(te["n_prior_off"], te["n_def"]); nmax = np.fmax(te["n_prior_off"], te["n_def"])
    bucket = np.where(te["days_since_start"] <= 14, "d0-14", np.where(te["days_since_start"] <= 45, "d15-45", "d46+"))
    cell = np.where(nmax == 0, "both_n0", np.where(nmin == 0, "one_n0", np.where(nmin <= 3, "n1-3", "n4+")))
    y = te["y"].to_numpy()
    tgt = {"TOV": ((y == 0).astype(float), {a: P[a][:, 0] for a in P}),
           "FT_trip": (np.isin(y, [4, 5]).astype(float), {a: P[a][:, 4] + P[a][:, 5] for a in P}),
           "LL": (None, {a: -np.log(np.clip(P[a][np.arange(len(y)), y], 1e-12, 1)) for a in P})}
    g = te["game_id"].to_numpy()
    rows = []
    labels = [("bucket", bucket), ("bucket|cell", np.char.add(np.char.add(bucket.astype(str), "|"), cell.astype(str))),
              ("all", np.full(len(te), "all"))]
    for gname, lab in labels:
        for v in sorted(set(lab)):
            m = lab == v
            if m.sum() < 2000:
                continue
            ug, inv = np.unique(g[m], return_inverse=True)
            n = np.bincount(inv)
            for t, (yy, pp) in tgt.items():
                s = {a: np.bincount(inv, pp[a][m]) for a in pp}
                sy = np.bincount(inv, yy[m]) if yy is not None else np.zeros(len(ug))
                scale = 100.0 if yy is not None else 1.0
                est = {a: scale * (s[a].sum() - sy.sum()) / n.sum() for a in s}
                diffs = []
                for _ in range(300):
                    k = RNG.integers(0, len(ug), len(ug))
                    diffs.append(scale * (s["FT"][k].sum() - s["FR"][k].sum()) / n[k].sum())
                rows.append(dict(group=gname, cell=v, line=t, n=int(m.sum()), n_games=len(ug),
                                 FR=est["FR"], FT=est["FT"], d_FT_FR=est["FT"] - est["FR"], se_d=float(np.std(diffs))))
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "phase1c_po_FRvFT_v1.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print(r.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
