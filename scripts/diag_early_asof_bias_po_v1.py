"""Phase 1b (diagnostic, nothing fitted): does the served possession_outcome `first` model's early-season level gap
live in team-games with THIN as-of features?

Served predictions T0 (F1 `fold1_v1/po`, F2 `full_retrain_v1/identity/po_served`, rows = round-2 design `first` rows
of the test season). Class level gap = mean p - realised share (pp) for TOV and FT trips (shooting + bonus), by
days bucket x as-of sample cell. Rows where BOTH teams have zero prior games carry team features of exactly 0.0
(the league mean, i.e. perfectly shrunk): if the window gap is the same there as in thin-sample rows, the gap is not
produced by thin-feature noise. Also by quintile of feature extremity inside d0-14 with n >= 1.
Output: results/early_asof_bias/phase1b_po_v1.csv. 2025-26 not loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "data/processed/models/possession_outcome/round2/design.parquet"
PRED = {"F1": (2024, ROOT / "data/processed/models/fold1_v1/po/pred_first_F1.npy"),
        "F2": (2025, ROOT / "data/processed/models/full_retrain_v1/identity/po_served/pred_first_F2.npy")}
OUT = ROOT / "results/early_asof_bias"
RNG = np.random.default_rng(7)


def gap(y, p, g, reps=200):
    est = 100 * (p.mean() - y.mean())
    ug, inv = np.unique(g, return_inverse=True)
    sp = np.bincount(inv, p); sy = np.bincount(inv, y); n = np.bincount(inv)
    bs = []
    for _ in range(reps):
        k = RNG.integers(0, len(ug), len(ug))
        bs.append(100 * (sp[k].sum() - sy[k].sum()) / n[k].sum())
    return est, float(np.std(bs))


def main():
    cols = ["season", "population", "y", "days_since_start", "game_id", "offense_team_id", "defense_team_id",
            "n_prior_off", "off_tov_c", "opp_def_tov_c", "off_ftr_c", "opp_def_ftr_c"]
    d = pd.read_parquet(DESIGN, columns=cols)
    rows = []
    for fold, (season, pp) in PRED.items():
        assert season <= 2025
        te = d[(d["season"] == season) & (d["population"] == "first")].reset_index(drop=True)
        P = np.load(pp).astype("float64")
        assert P.shape == (len(te), 6)
        npo = te.groupby(["game_id", "offense_team_id"])["n_prior_off"].first()
        idx = pd.MultiIndex.from_arrays([te["game_id"], te["defense_team_id"]])
        te["n_prior_def"] = npo.reindex(idx).to_numpy()
        te["nmin"] = np.fmin(te["n_prior_off"], te["n_prior_def"].fillna(0))
        te["nmax"] = np.fmax(te["n_prior_off"], te["n_prior_def"].fillna(0))
        b = np.where(te["days_since_start"] <= 14, "d0-14", np.where(te["days_since_start"] <= 45, "d15-45", "d46+"))
        cell = np.where(te["nmax"] == 0, "both_n0", np.where(te["nmin"] == 0, "one_n0",
                        np.where(te["nmin"] <= 3, "n1-3", "n4+")))
        te["bucket"], te["cell"] = b, cell
        targets = {"TOV": ((te["y"] == 0).to_numpy(float), P[:, 0], ("off_tov_c", "opp_def_tov_c")),
                   "FT_trip": (te["y"].isin([4, 5]).to_numpy(float), P[:, 4] + P[:, 5], ("off_ftr_c", "opp_def_ftr_c"))}
        for tname, (y, p, fc) in targets.items():
            ext = te[fc[0]].abs() + te[fc[1]].abs()
            groups = [("bucket", te["bucket"]), ("bucket|cell", te["bucket"] + "|" + te["cell"])]
            m14 = (te["bucket"] == "d0-14") & (te["nmin"] >= 1)
            q = pd.Series("na", index=te.index)
            q[m14] = "d0-14|n>=1|extQ" + (pd.qcut(ext[m14], 5, labels=False) + 1).astype(str)
            groups.append(("extremity", q))
            # signed: does the gap follow the feature sign (over-reaction to noisy features)?
            sq = pd.Series("na", index=te.index)
            sgn = te[fc[0]] + te[fc[1]]
            sq[m14] = "d0-14|n>=1|signQ" + (pd.qcut(sgn[m14], 5, labels=False) + 1).astype(str)
            groups.append(("signed", sq))
            for gname, lab in groups:
                for v in sorted(set(lab) - {"na"}):
                    m = (lab == v).to_numpy()
                    e, se = gap(y[m], p[m], te["game_id"].to_numpy()[m])
                    rows.append(dict(fold=fold, target=tname, group=gname, cell=v, n=int(m.sum()),
                                     n_games=int(te.loc[m, "game_id"].nunique()), pred=100 * p[m].mean(),
                                     actual=100 * y[m].mean(), gap_pp=e, se_pp=se))
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "phase1b_po_v1.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 400)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
