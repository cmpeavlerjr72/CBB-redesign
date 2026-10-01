"""exp_chance_time_offline_v1.py -- chance_time round 1 OFFLINE feed-parity bake-off (chance_time/experiments.md s1.3).

Lane I, 2026-09-30. One blind grader for every arm: each arm only supplies a FEED function
(real state -> fed chance_elapsed_s / is_transition_f); the same served round4_B1 S1 artifacts score
every row. Real rows: fg_make design held-out fold-test rows joined to their possession.

Usage: exp_chance_time_offline_v1.py <out_json>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
from cbb_sim.engine.adapters import Adapters  # noqa: E402
from cbb_sim.engine.inputs import EngineInputs  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402
from build_chance_time_lut_v1 import dur_bin, CLASSES  # noqa: E402

OUT = Path(sys.argv[1])
PV = np.array([2.0, 2.0, 3.0])
TRANS_MAX = 8.0
MAXE = 60.0


def real_rows(season: int) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet")
    d = d[d["season"] == season].copy()
    ex = pd.read_parquet(ROOT / "data/processed/models/fg_make/design_v4_extra_v2.parquet",
                         columns=["shooter_shrunk_dev_c"])
    d["shooter_shrunk_dev_c"] = ex.loc[d.index, "shooter_shrunk_dev_c"].to_numpy()
    d["row"] = np.arange(len(d))
    c = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{season}.parquet",
                        columns=["game_id", "period", "poss_index", "chance_number", "offense_team_id",
                                 "start_clock"])
    p = pd.read_parquet(ROOT / f"data/processed/possessions_v2/possessions_{season}.parquet",
                        columns=["game_id", "period", "poss_index", "duration_s", "start_reason"])
    d["start_clock"] = (d["seconds_remaining"] + d["chance_elapsed_s"]).round().astype("int64")
    c["start_clock"] = c["start_clock"].astype("int64")
    m = d.merge(c.rename(columns={"offense_team_id": "off_team_id"}),
                on=["game_id", "period", "off_team_id", "chance_number", "start_clock"], how="left")
    m = m.drop_duplicates("row")
    m = m.merge(p, on=["game_id", "period", "poss_index"], how="left")
    m = m.sort_values("row").reset_index(drop=True)
    return m


def feed(arm: str, m: pd.DataFrame, lut: dict, ce_med: dict, rng: np.random.Generator):
    """Return (elapsed, transition) fed for every row, from the row's REAL state."""
    ch = m["chance_number"].to_numpy()
    D = m["duration_s"].to_numpy(np.float64)
    start_tr = m["start_reason"].isin(["DREB", "TOV"]).to_numpy()
    k = m["shot_class"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    first = ch == 1
    el = np.empty(len(m))
    # chance >= 2
    if arm == "R":
        el[~first] = np.array([ce_med.get(min(max(int(j), 1), 3), 3.0) for j in ch[~first]])
    else:
        b = np.where(ch[~first] >= 3, 1, 0)
        u = rng.random((~first).sum())
        qi = np.minimum((u * (lut["cont_q"].shape[2] - 1)).round().astype(int), lut["cont_q"].shape[2] - 1)
        el[~first] = lut["cont_q"][b, k[~first], qi]
    # chance 1
    if arm in ("R", "C2"):
        e1 = np.minimum(D[first], MAXE)
        tr1 = (D[first] <= TRANS_MAX) & start_tr[first]
    else:
        g = np.where(start_tr[first], 0, 1)
        db = np.clip(dur_bin(D[first].astype(np.int64)), 0, lut["r_q"].shape[1] - 1)
        u = rng.random(first.sum())
        qi = np.minimum((u * (lut["r_q"].shape[2] - 1)).round().astype(int), lut["r_q"].shape[2] - 1)
        r = lut["r_q"][g, db, qi]
        e1r = np.round(r * D[first])
        e1 = np.minimum(e1r, MAXE)
        tr1 = (e1r <= TRANS_MAX) & start_tr[first]
    el[first] = e1
    tr = np.zeros(len(m))
    tr[first] = tr1
    return el, tr


def main():
    inp = EngineInputs.load(str(ROOT / "data/processed/models/engine_v3"), "F2_2025")
    ad = Adapters.load(inp, "F2", 2025)
    pos = {int(g): i for i, g in enumerate(inp.games["game_id"].to_numpy())}
    ce_med = {int(k): float(v) for k, v in inp.rules["chance_elapsed_median_by_chance"].items()}
    lut2 = dict(np.load(ROOT / "data/processed/models/chance_time/F2/lut_v1.npz"))
    m = real_rows(2025)
    ok = m["duration_s"].notna() & m["game_id"].map(pos).notna()
    res = {"n_rows": int(len(m)), "matched_share": float(ok.mean())}
    m = m[ok].reset_index(drop=True)
    gidx = m["game_id"].map(pos).astype(np.int64).to_numpy()
    feats = {c: json.loads((Path(ad.fg.source[c]["path"]) / f"manifest_{c}.json").read_text())["features"]
             for c in CLASSES}
    games_n = m["game_id"].nunique()

    def score(el, tr):
        p = np.empty(len(m))
        for c in CLASSES:
            sel = (m["shot_class"] == c).to_numpy()
            X = np.asarray(FG.design_matrix(m[sel], feats[c]), dtype=np.float64)
            f = feats[c]
            if el is not None:
                X[:, f.index("chance_elapsed_s")] = el[sel]
                X[:, f.index("is_transition_f")] = tr[sel]
            segs = ad.fg.manifests[c].segments(gidx[sel])
            out = np.empty(sel.sum())
            for s in np.unique(segs):
                r = np.flatnonzero(segs == s)
                out[r] = ad.fg.models_by_seg[c][s].predict_proba(np.ascontiguousarray(X[r], dtype=np.float32))[:, 1]
            p[sel] = out
        return p

    y = m["y"].to_numpy(np.float64)
    kk = m["shot_class"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    first = (m["chance_number"] == 1).to_numpy()
    p_true = score(None, None)
    A = np.array([(kk == i).sum() / games_n for i in range(3)])

    def G(p, idx=None):
        sel = np.ones(len(m), bool) if idx is None else idx
        g = 0.0
        for i in range(3):
            s = sel & (kk == i)
            g += PV[i] * A[i] * abs(p[s].mean() - p_true[s].mean())
        return g

    def ll(p, s):
        return float(-np.mean(y[s] * np.log(np.clip(p[s], 1e-9, 1)) + (1 - y[s]) * np.log(np.clip(1 - p[s], 1e-9, 1))))

    preds = {}
    arms = {}
    for arm in ("R", "C2", "C12"):
        for seed in (0, 1):
            if arm == "R" and seed == 1:
                continue
            el, tr = feed(arm, m, lut2, ce_med, np.random.default_rng(1000 + seed))
            p = score(el, tr)
            preds[(arm, seed)] = (p, el, tr)
        p, el, tr = preds[(arm, 0)]
        cls = {}
        for i, c in enumerate(CLASSES):
            s = kk == i
            cls[c] = {"mean_p_arm": float(p[s].mean()), "mean_p_true": float(p_true[s].mean()),
                      "mean_y": float(y[s].mean()), "gap_pp": 100 * float(p[s].mean() - p_true[s].mean()),
                      "gap_pp_chance1": 100 * float(p[s & first].mean() - p_true[s & first].mean()),
                      "gap_pp_chance2plus": 100 * float(p[s & ~first].mean() - p_true[s & ~first].mean()),
                      "ll_arm_minus_true": ll(p, s) - ll(p_true, s),
                      "fed_trans_share_chance1": float(tr[s & first].mean()),
                      "real_trans_share_chance1": float(m.loc[s & first, "is_transition_f"].mean()),
                      "fed_elapsed_q_c1": np.quantile(el[s & first], [.1, .25, .5, .75, .9]).tolist(),
                      "real_elapsed_q_c1": np.quantile(m.loc[s & first, "chance_elapsed_s"], [.1, .25, .5, .75, .9]).tolist(),
                      "fed_elapsed_q_c2p": np.quantile(el[s & ~first], [.1, .25, .5, .75, .9]).tolist(),
                      "real_elapsed_q_c2p": np.quantile(m.loc[s & ~first, "chance_elapsed_s"], [.1, .25, .5, .75, .9]).tolist()}
            # responsiveness by off_make_c quintile
            q = pd.qcut(m.loc[s, "off_make_c"].rank(method="first"), 5, labels=False).to_numpy()
            mp = [p[s][q == j].mean() for j in range(5)]
            my = [y[s][q == j].mean() for j in range(5)]
            cls[c]["resp_slope"] = float((mp[4] - mp[0]) / (my[4] - my[0]))
        # month / site cuts of the points gap (signed, sim-style: arm minus true, pts/game)
        cuts = {}
        mon = pd.to_datetime(m["game_date"]).dt.month.to_numpy()
        site = np.where(m["neutral_site"], "neutral", np.where(m["offense_is_home"], "home", "away"))
        for nm, vals in (("month", mon), ("site", site)):
            cuts[nm] = {}
            for v in np.unique(vals):
                sel = vals == v
                gpts = sum(PV[i] * A[i] * (p[sel & (kk == i)].mean() - p_true[sel & (kk == i)].mean()) for i in range(3))
                cuts[nm][str(v)] = {"n": int(sel.sum()), "signed_pts": float(gpts)}
        arms[arm] = {"G_feed": G(p), "classes": cls, "cuts": cuts,
                     "signed_pts": float(sum(PV[i] * A[i] * (p[kk == i].mean() - p_true[kk == i].mean()) for i in range(3)))}
        print(arm, "G_feed", round(arms[arm]["G_feed"], 4), "signed", round(arms[arm]["signed_pts"], 4),
              {c: (round(v["gap_pp"], 3), round(v["ll_arm_minus_true"], 6), round(v["resp_slope"], 3)) for c, v in cls.items()},
              flush=True)
    # floors: reseed and paired game bootstrap of G(arm) - G(R)
    gids = m["game_id"].to_numpy()
    ug, inv = np.unique(gids, return_inverse=True)
    rngb = np.random.default_rng(12345)
    for arm in ("C2", "C12"):
        reseed = abs(G(preds[(arm, 1)][0]) - G(preds[(arm, 0)][0]))
        diffs = []
        pa, pr = preds[(arm, 0)][0], preds[("R", 0)][0]
        for _ in range(200):
            w = np.bincount(rngb.integers(0, len(ug), len(ug)), minlength=len(ug))[inv].astype(float)
            def Gw(p):
                g = 0.0
                for i in range(3):
                    s = kk == i
                    ww = w[s]
                    g += PV[i] * A[i] * abs((p[s] * ww).sum() / ww.sum() - (p_true[s] * ww).sum() / ww.sum())
                return g
            diffs.append(Gw(pa) - Gw(pr))
        se = float(np.std(diffs))
        floor = max(reseed, 2 * se)
        arms[arm]["floor"] = {"reseed": reseed, "boot_se": se, "floor": floor,
                              "dG_vs_R": arms[arm]["G_feed"] - arms["R"]["G_feed"],
                              "beats_R_by_floors": (arms["R"]["G_feed"] - arms[arm]["G_feed"]) / floor}
        print(arm, arms[arm]["floor"], flush=True)
    res["arms"] = arms
    res["true_ll"] = {c: ll(p_true, kk == i) for i, c in enumerate(CLASSES)}
    res["attempts_per_game"] = A.tolist()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
