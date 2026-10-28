"""Round 3 Part B: per-shot grade layers L0/L1/L2 (+D1, +D2 diagnostics) and the EP table for Part C.

Usage: python scripts/train_shot_grade_v1.py TRAIN_SEASON TEST_SEASON   (2024/2025 only)
Spec: docs/models/shot_location/experiments.md R3.4. Model class and hyperparameters as the partner b_model.py.
Writes data/processed/models/shot_location/round3/grade_{train}_to_{test}.json and ep_{test}.parquet
(per-FGA EP of the TEST season from the model trained on the TRAIN season; seed 0 and seed 1 retrains of L0 and L2).
"""
import sys, json, pathlib, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/models/shot_location/round3"
JUMP = ("jumper2", "three")
L2 = ["shot_class", "dist_j", "angle_j", "corner3", "origin", "poss_sec", "after_timeout"]
LAYERS = {
    "L0": ["shot_class"],
    "L1": ["shot_class", "dist_j", "angle_j", "corner3"],
    "L2": L2,
    "D1": L2 + ["home"],
    "D2": L2 + ["period", "game_sec_left", "margin"],
}
BASE_FOR = {"D1": "L2", "D2": "L2", "L1": "L0", "L2": "L1"}
CATS = ["shot_class", "origin"]
NBOOT = 500


def prep(d):
    d = d[~d.heave].copy()
    j = d.shot_class.isin(JUMP)
    d["dist_j"] = np.where(j, d.dist_ft, np.nan)
    d["angle_j"] = np.where(j, d.angle_deg, np.nan)
    d["located"] = d.x.notna()
    d["corner3"] = d.corner3.fillna(False).astype(int)
    for c in ("after_timeout", "home"):
        d[c] = d[c].fillna(False).astype(int)
    for c in CATS:
        d[c] = d[c].astype("category")
    d["yb"] = d.made.astype(int)
    return d.reset_index(drop=True)


def fit_predict(tr, te, cols, seed=0):
    X_tr, X_te = tr[cols].copy(), te[cols].copy()
    for c in CATS:
        if c in cols:
            cats = tr[c].cat.categories
            X_tr[c] = X_tr[c].cat.set_categories(cats).cat.codes
            X_te[c] = pd.Categorical(X_te[c], categories=cats).codes
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                       categorical_features=[cols.index(c) for c in CATS if c in cols],
                                       early_stopping=True, validation_fraction=0.1, random_state=seed)
    m.fit(X_tr, tr.yb)
    return np.clip(m.predict_proba(X_te)[:, 1], 1e-6, 1 - 1e-6)


def floor_gain(gi, n_games, boots, base_ll, ll, mask=None):
    b, l = (base_ll, ll) if mask is None else (base_ll[mask], ll[mask])
    g = gi if mask is None else gi[mask]
    gain_g = np.bincount(g, b - l, minlength=n_games)
    n_g = np.bincount(g, minlength=n_games)
    gain = gain_g.sum() / n_g.sum()
    sd = np.std([(w @ gain_g) / (w @ n_g) for w in boots if (w @ n_g) > 0])
    return float(gain), float(sd)


def main(train_s, test_s):
    assert train_s in (2024, 2025) and test_s in (2024, 2025) and train_s != test_s
    tr = prep(pd.read_parquet(OUT / f"shot_context_{train_s}.parquet"))
    te = prep(pd.read_parquet(OUT / f"shot_context_{test_s}.parquet"))
    print(f"train {train_s}: FGA {len(tr):,}   test {test_s}: FGA {len(te):,} ({te.gameId.nunique():,} games)", flush=True)
    ug, gi = np.unique(te.gameId.values, return_inverse=True)
    rng = np.random.default_rng(0)
    boots = [np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(NBOOT)]
    loc = te.located.values
    lls, res, ep = {}, {}, te[["gameId", "id", "team", "opp_team", "date", "home", "neutral", "shot_class", "value", "pts",
                               "abs_script_index", "team_conf", "opp_conf", "located"]].copy()
    for name, cols in LAYERS.items():
        t0 = time.time()
        p = fit_predict(tr, te, cols)
        ll = -(te.yb * np.log(p) + (1 - te.yb) * np.log(1 - p)).values
        lls[name] = ll
        r = {"logloss": float(ll.mean()), "logloss_located": float(ll[loc].mean()), "secs": round(time.time() - t0)}
        if name in BASE_FOR:
            b = lls[BASE_FOR[name]]
            gain, sd = floor_gain(gi, len(ug), boots, b, ll)
            gl, sdl = floor_gain(gi, len(ug), boots, b, ll, loc)
            r.update(base=BASE_FOR[name], gain=gain, floor=sd, gain_floors=gain / sd,
                     gain_located=gl, floor_located=sdl, gain_located_floors=gl / sdl)
        res[name] = r
        if name in ("L0", "L2"):
            ep["ep_" + name] = p * te.value.values
            ep["p_" + name] = p
        print(name, json.dumps(r), flush=True)
    # seed-1 retrains (noise-floor check for the grade models)
    for name in ("L0", "L2"):
        p = fit_predict(tr, te, LAYERS[name], seed=1)
        ll = -(te.yb * np.log(p) + (1 - te.yb) * np.log(1 - p)).values
        ep["ep_" + name + "_s1"] = p * te.value.values
        res[name + "_seed1"] = {"logloss": float(ll.mean()), "mean_abs_ep_diff_vs_seed0": float(np.abs(p * te.value.values - ep["ep_" + name]).mean())}
        print(name + "_seed1", json.dumps(res[name + "_seed1"]), flush=True)
    # segment: located-only share by class for the record
    res["n_fga_test"] = int(len(te)); res["n_fga_train"] = int(len(tr)); res["located_share_test"] = float(loc.mean())
    res["boot"] = NBOOT
    (OUT / f"grade_{train_s}_to_{test_s}.json").write_text(json.dumps(res, indent=1))
    ep.to_parquet(OUT / f"ep_{test_s}.parquet", index=False)
    print("wrote", OUT / f"ep_{test_s}.parquet")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
