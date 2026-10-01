"""train_fg_make_two_stage_v1.py -- lane A day 2026-10-01, second round (docs/models/fg_make/experiments.md
section 25): a TWO-STAGE fg_make.

Stage A (trees, within-game only): LightGBM, the served B1 params, the S1 monthly schedule, on
    R2_SAFE_STATE (period, seconds_remaining, in_bonus, chance_number, chance_elapsed_s, is_transition_f)
    + shooter_shrunk_dev_c (the served shooter block) + lg_make_asof (the league's as-of make rate on this
    shot type: a league snapshot, no team input). No team rate, no rating, no site, no season index.
Stage B (the PREDICTABLE team-game effect, logit scale, per shot type), added as an offset:
    eta = logit(pA) + b_home*site_home + b_away*site_away + u_off + v_def
    u_off = (S_off + mu_off / tau_off^2) / (I_off + 1 / tau_off^2)     (one-step posterior mean of a random
    v_def = (S_def + mu_def / tau_def^2) / (I_def + 1 / tau_def^2)      team effect, linearised at 0)
    S = sum(y - pA), I = sum(pA (1 - pA)) over the team's EARLIER games of the same season (as-of, strictly
    before the game date) as offence / as defence on this shot type; pA there is out of sample (S1 schedule in
    the test season, team-season-grouped 3-fold OOF in the training seasons). mu = rho * last season's final
    effect (arm TS3 only; 0 otherwise). tau, rho, b_home, b_away are fitted ON THE FOLD'S TRAINING SEASONS
    ONLY (grid for tau / rho, Newton for the site terms) and frozen for the test season.
    Every team input is relative to the league: stage A carries the league level, so S and I measure the
    team against its own snapshot's league.

Arms: TS1 (one tau per shot type, offence and defence pooled), TS2 (tau_off, tau_def separately), TS3 (TS2 +
prior-season carry rho). Stage A is shared by the arms of one (fold, seed); the reseed is stage A's seed.

Outputs under <out-root>/<arm>_s<seed>_<fold>/:
    B1/manifest_FGA_*.json + per-cut joblibs holding a `TwoStageModel` (features = stage A features +
        `fg_team_offset`, the LAST column, added to the logit) -- loadable by the engine's dated fg loader;
    preds_<fold>_s<seed>.parquet (per test shot: p, p_zero = pA, and every input) for the grader;
    offsets_<fold>_s<seed>.parquet (per engine game x side x type: fg_team_offset, lg_make_asof) for serving;
    stageB_<fold>_s<seed>.json (fitted tau / rho / site terms with the training-season grid).

    .venv/Scripts/python.exe scripts/train_fg_make_two_stage_v1.py --fold F2 --seed 0 --n-jobs 4 \
        --out-root data/processed/models/fg_make/round_ts
"""
from __future__ import annotations

import train_par_common_v1 as C

C.pin_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402
from fg_two_stage_model_v1 import TwoStageModel  # noqa: E402

DESIGN = ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet"
EXTRA = ROOT / "data/processed/models/fg_make/design_v4_extra_v2.parquet"
LADDER = ROOT / "data/processed/models/fg_make/lgbm_ladder_v2.json"
FA = [*FG.R2_SAFE_STATE, "shooter_shrunk_dev_c", "lg_make_asof"]
KEY = {"FGA_rim": "rim", "FGA_jump2": "jump2", "FGA_3": "three"}
TAU_GRID = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25)
RHO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ARMS = ("TS1", "TS2", "TS3")
EPS = 1e-6


def log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _fit(X, y, Xte, params, seed):
    C.pin_threads()
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(random_state=seed, n_jobs=1, **{**FG.LGBM_BASE, **params})
    clf.fit(X, y)
    return {"clf": clf, "p": clf.predict_proba(Xte)[:, 1]}


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- stage A
def stage_a(design, fold, seed, params, n_jobs, ckpt, t0):
    tr_all, te_all = FG.fold_slices(design, fold)
    tasks, plan = {}, {}
    from sklearn.model_selection import GroupKFold
    for c in FG.SHOT_CLASSES:
        tr, te = FG.class_slice(tr_all, c), FG.class_slice(te_all, c)
        prm = params[c]
        # (1) team-season-grouped 3-fold OOF on the training seasons
        X = FG.design_matrix(tr, FA)
        y = tr["y"].to_numpy()
        grp = (tr["season"].astype(str) + "_" + tr["off_team_id"].astype(str)).to_numpy()
        folds = list(GroupKFold(n_splits=3).split(X, y, grp))
        for f, (a, b) in enumerate(folds):
            tasks[f"oof_{c}_{f}"] = dict(X=X[a], y=y[a], Xte=X[b], params=prm, seed=seed)
        # (2) the S1 schedule on the test season
        te_dates = pd.to_datetime(te["game_date"])
        tr_dates = pd.to_datetime(tr["game_date"])
        cuts = PO.month_boundaries(te_dates)
        segs = []
        cols = [*FA, "y"]
        for k, cut in enumerate(cuts):
            nxt = cuts[k + 1] if k + 1 < len(cuts) else None
            seg = ((te_dates >= cut) if nxt is None else ((te_dates >= cut) & (te_dates < nxt))).to_numpy()
            if not seg.any():
                continue
            before = (te_dates < cut).to_numpy()
            rows = pd.concat([tr[cols], te.loc[before, cols]], ignore_index=True) if before.any() else tr[cols]
            max_train = tr_dates.max() if not before.any() else max(tr_dates.max(), te_dates[before].max())
            key = f"s1_{c}_{cut.date()}"
            tasks[key] = dict(X=FG.design_matrix(rows, FA), y=rows["y"].to_numpy(),
                              Xte=FG.design_matrix(te.loc[seg], FA), params=prm, seed=seed)
            segs.append({"key": key, "seg": seg, "cut": cut, "max_train": str(pd.Timestamp(max_train).date()),
                         "n_train": len(rows)})
        plan[c] = {"tr_idx": tr.index.to_numpy(), "te_idx": te.index.to_numpy(), "folds": folds, "segs": segs}
    log(f"stage A: {len(tasks)} fits", t0)
    res = C.run_tasks(tasks, _fit, n_jobs, ckpt)
    pA = pd.Series(np.nan, index=design.index)
    for c, pl in plan.items():
        tri = pl["tr_idx"]
        for f, (a, b) in enumerate(pl["folds"]):
            pA.loc[tri[b]] = res[f"oof_{c}_{f}"]["p"]
        tei = pl["te_idx"]
        for s in pl["segs"]:
            pA.loc[tei[s["seg"]]] = res[s["key"]]["p"]
    return pA, plan, res


# --------------------------------------------------------------------------- stage B pieces
def team_game_stats(d: pd.DataFrame, side: str) -> pd.DataFrame:
    """per (season, team, game, class) S, I and the AS-OF cumulative before the game date (same season)."""
    tcol = "off_team_id" if side == "off" else "def_team_id"
    g = d.assign(r=d["y"] - d["pA"], i=d["pA"] * (1 - d["pA"])).groupby(
        ["season", tcol, "shot_class", "game_id", "game_date"], as_index=False).agg(S=("r", "sum"), I=("i", "sum"))
    g = g.rename(columns={tcol: "team_id"})
    # by date (doubleheaders: same-date games are excluded from each other's as-of)
    byd = g.groupby(["season", "team_id", "shot_class", "game_date"], as_index=False)[["S", "I"]].sum()
    byd = byd.sort_values(["season", "team_id", "shot_class", "game_date"])
    byd[["cS", "cI"]] = byd.groupby(["season", "team_id", "shot_class"])[["S", "I"]].cumsum() - byd[["S", "I"]].values
    fin = byd.groupby(["season", "team_id", "shot_class"], as_index=False)[["S", "I"]].sum()
    return byd, fin


def asof_lookup(byd: pd.DataFrame, keys: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """keys: season, team_id, shot_class, game_date -> cumulative S, I strictly before game_date."""
    out_S = np.zeros(len(keys)); out_I = np.zeros(len(keys))
    k2 = keys.reset_index(drop=True).copy()
    k2["_r"] = np.arange(len(k2))
    b = byd.copy()
    for f in (k2, b):
        f["game_date"] = pd.to_datetime(f["game_date"]).astype("datetime64[ns]")
        f["season"] = f["season"].astype("int64")
        f["team_id"] = f["team_id"].astype("int64")
        f["shot_class"] = f["shot_class"].astype(str)
    b["tS"] = b["cS"] + b["S"]; b["tI"] = b["cI"] + b["I"]       # totals THROUGH that date
    m = pd.merge_asof(k2.sort_values("game_date"), b.sort_values("game_date")[["season", "team_id", "shot_class",
                                                                              "game_date", "tS", "tI"]],
                      on="game_date", by=["season", "team_id", "shot_class"], allow_exact_matches=False,
                      direction="backward")
    m = m.sort_values("_r")
    out_S = m["tS"].fillna(0.0).to_numpy(); out_I = m["tI"].fillna(0.0).to_numpy()
    return out_S, out_I


def prior_lookup(fin: pd.DataFrame, keys: pd.DataFrame, tau: float) -> np.ndarray:
    f = fin.copy()
    f["prev"] = f["S"] / (f["I"] + (1 / tau ** 2 if tau > 0 else np.inf)) if tau > 0 else 0.0
    f["season"] = f["season"].astype("int64") + 1
    f["team_id"] = f["team_id"].astype("int64")
    kk = keys[["season", "team_id", "shot_class"]].reset_index(drop=True).astype(
        {"season": "int64", "team_id": "int64"})
    m = kk.merge(f[["season", "team_id", "shot_class", "prev"]],
                                                         on=["season", "team_id", "shot_class"], how="left")
    return m["prev"].fillna(0.0).to_numpy()


def post_mean(S, I, tau, mu):
    if tau <= 0:
        return np.zeros_like(S)
    return (S + mu / tau ** 2) / (I + 1 / tau ** 2)


def fit_site(y, base, sh, sa, iters=6):
    b = np.zeros(2)
    X = np.column_stack([sh, sa])
    for _ in range(iters):
        p = sig(base + X @ b)
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        b = b + np.linalg.solve(H, g)
    p = np.clip(sig(base + X @ b), EPS, 1 - EPS)
    return b, float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_stage_b(tr: pd.DataFrame, arm: str) -> dict:
    """grid over tau (and rho) on the TRAINING rows; site terms by Newton at each grid point."""
    y = tr["y"].to_numpy(float); base = logit(tr["pA"].to_numpy())
    sh, sa = tr["site_home"].to_numpy(float), tr["site_away"].to_numpy(float)
    grid = []
    if arm == "TS1":
        cands = [(t, t, 0.0) for t in TAU_GRID]
    elif arm == "TS2":
        cands = [(to, td, 0.0) for to in TAU_GRID for td in TAU_GRID]
    else:
        cands = [(to, td, r) for to in TAU_GRID if to > 0 for td in TAU_GRID if td > 0 for r in RHO_GRID]
    for to, td, rho in cands:
        mo = rho * tr["prev_off_" + f"{to:.3f}"].to_numpy() if rho else 0.0
        md = rho * tr["prev_def_" + f"{td:.3f}"].to_numpy() if rho else 0.0
        u = post_mean(tr["cS_off"].to_numpy(), tr["cI_off"].to_numpy(), to, mo)
        v = post_mean(tr["cS_def"].to_numpy(), tr["cI_def"].to_numpy(), td, md)
        b, ll = fit_site(y, base + u + v, sh, sa)
        grid.append({"tau_off": to, "tau_def": td, "rho": rho, "b_home": float(b[0]), "b_away": float(b[1]), "ll": ll})
    best = min(grid, key=lambda r: r["ll"])
    return {"best": best, "grid": grid}


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out-root", type=Path, default=ROOT / "data/processed/models/fg_make/round_ts")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    t0 = time.time()
    from cbb_sim.data.seal import assert_not_sealed
    assert_not_sealed([2022, 2023, 2024, 2025], context="fg_make two-stage")
    design = pd.read_parquet(DESIGN)
    extra = pd.read_parquet(EXTRA, columns=["shooter_shrunk_dev_c"])
    design["shooter_shrunk_dev_c"] = extra["shooter_shrunk_dev_c"].to_numpy()
    last = 2024 if a.fold == "F1" else 2025
    design = design[design["season"] <= last].reset_index(drop=True)
    params = {c: {k: x for k, x in dict(v).items() if k != "n_jobs"}
              for c, v in json.loads(LADDER.read_text(encoding="utf-8"))["frozen_params"].items()}
    a.out_root.mkdir(parents=True, exist_ok=True)
    ckpt = a.out_root / f"_stageA_s{a.seed}_{a.fold}"
    pA, plan, res = stage_a(design, a.fold, a.seed, params, a.n_jobs, ckpt, t0)
    used = pA.notna().to_numpy()
    d = design.loc[used].copy()
    d["pA"] = pA[used].to_numpy()
    test_season = max(FG.FOLDS[a.fold]["test"])
    # as-of team stats
    stats = {}
    for side in ("off", "def"):
        byd, fin = team_game_stats(d, side)
        stats[side] = (byd, fin)
        tcol = "off_team_id" if side == "off" else "def_team_id"
        keys = d[["season", tcol, "shot_class", "game_date"]].rename(columns={tcol: "team_id"})
        S, I = asof_lookup(byd, keys)
        d[f"cS_{side}"] = S
        d[f"cI_{side}"] = I
        for t in TAU_GRID:
            if t > 0:
                d[f"prev_{side}_{t:.3f}"] = prior_lookup(fin, keys, t)
    log("stage B inputs built", t0)
    # engine games (fold 2 only): every game x side x type, offsets from the same stats
    games = None
    if a.fold == "F2":
        games = pd.read_parquet(ROOT / "data/processed/models/engine_v3/games_F2_2025.parquet")
    arms = [x for x in a.arms.split(",") if x]
    for arm in arms:
        out = a.out_root / f"{arm}_s{a.seed}_{a.fold}"
        (out / "B1").mkdir(parents=True, exist_ok=True)
        fitted, pred_rows, off_rows = {}, [], []
        for c in FG.SHOT_CLASSES:
            dc = d[d["shot_class"] == c]
            tr = dc[dc["season"] < test_season]
            te = dc[dc["season"] == test_season]
            fb = fit_stage_b(tr, arm)
            bst = fb["best"]
            fitted[c] = {"best": bst, "grid": fb["grid"], "n_train_rows": int(len(tr))}
            to, td, rho = bst["tau_off"], bst["tau_def"], bst["rho"]

            def offset(fr):
                mo = rho * fr[f"prev_off_{to:.3f}"].to_numpy() if rho else 0.0
                md = rho * fr[f"prev_def_{td:.3f}"].to_numpy() if rho else 0.0
                return (bst["b_home"] * fr["site_home"].to_numpy() + bst["b_away"] * fr["site_away"].to_numpy()
                        + post_mean(fr["cS_off"].to_numpy(), fr["cI_off"].to_numpy(), to, mo)
                        + post_mean(fr["cS_def"].to_numpy(), fr["cI_def"].to_numpy(), td, md))
            off_te = offset(te)
            p = sig(logit(te["pA"].to_numpy()) + off_te)
            fr = pd.DataFrame({"game_id": te["game_id"].to_numpy(), "season": te["season"].to_numpy(),
                               "game_date": te["game_date"].to_numpy(), "off_team_id": te["off_team_id"].to_numpy(),
                               "def_team_id": te["def_team_id"].to_numpy(), "shooter_id": te["shooter_id"].to_numpy(),
                               "shot_class": c, "y": te["y"].to_numpy(), "p": p, "p_zero": te["pA"].to_numpy(),
                               "off_make_c": te["off_make_c"].to_numpy(), "def_allow_c": te["def_allow_c"].to_numpy(),
                               "shooter_shrunk_dev_c": te["shooter_shrunk_dev_c"].to_numpy(),
                               "lg_make_asof": te["lg_make_asof"].to_numpy(), "fg_team_offset": off_te})
            pred_rows.append(fr)
            sc = FG.score(te, np.column_stack([1 - p, p]))
            fitted[c]["score"] = {k: sc[k] for k in ("n", "log_loss", "calib_pass", "calib_worst_gap_pp",
                                                     "resp_pass_decision8", "pred_make_rate", "actual_make_rate")}
            scA = FG.score(te, np.column_stack([1 - te["pA"].to_numpy(), te["pA"].to_numpy()]))
            fitted[c]["stageA_log_loss"] = scA["log_loss"]
            print(f"  {arm} {c}: tau_off {to} tau_def {td} rho {rho} site {bst['b_home']:+.3f}/{bst['b_away']:+.3f} "
                  f"ll {sc['log_loss']:.6f} (stage A {scA['log_loss']:.6f}) calib {sc['calib_worst_gap_pp']:.3f}pp "
                  f"D8 {'PASS' if sc['resp_pass_decision8'] else 'FAIL'}", flush=True)
            # engine offsets for every fold-2 game x side (as-of from the same stats)
            if games is not None:
                lg = te.assign(_d=pd.to_datetime(te["game_date"]).astype("datetime64[ns]")).groupby(
                    "_d")["lg_make_asof"].first().sort_index()
                for side, own, opp, sh_, sa_ in ((0, "home_team_id", "away_team_id", 1.0, 0.0),
                                                 (1, "away_team_id", "home_team_id", 0.0, 1.0)):
                    kf = pd.DataFrame({"season": test_season, "team_id": games[own].to_numpy(), "shot_class": c,
                                       "game_date": pd.to_datetime(games["game_date"]).to_numpy()})
                    kd = kf.assign(team_id=games[opp].to_numpy())
                    So, Io = asof_lookup(stats["off"][0], kf)
                    Sd, Id = asof_lookup(stats["def"][0], kd)
                    mo = rho * prior_lookup(stats["off"][1], kf, to) if rho else 0.0
                    md = rho * prior_lookup(stats["def"][1], kd, td) if rho else 0.0
                    neu = games["neutral"].to_numpy().astype(bool)
                    site = np.where(neu, 0.0, bst["b_home"] * sh_ + bst["b_away"] * sa_)
                    val = site + post_mean(So, Io, to, mo) + post_mean(Sd, Id, td, md)
                    lgv = pd.Series(pd.to_datetime(games["game_date"]).to_numpy()).map(
                        lambda x: lg.asof(x) if len(lg) else np.nan)
                    off_rows.append(pd.DataFrame({"game_id": games["game_id"].to_numpy(), "side": side, "key": KEY[c],
                                                  "fg_team_offset": val, "lg_make_asof": lgv.to_numpy()}))
            # export: one TwoStageModel per S1 cut
            arts = []
            for s in plan[c]["segs"]:
                fname = f"{c}_{s['cut'].date()}.joblib"
                joblib.dump({"arm": "lgbm", "scheme": "S1", "two_stage_arm": arm, "features": [*FA, "fg_team_offset"],
                             "model": TwoStageModel(res[s["key"]]["clf"]), "fold": a.fold, "shot_class": c,
                             "adopted": False, "refit_date": str(s["cut"].date()), "max_train_date": s["max_train"],
                             "stage_b": bst, "note": "fg_make two-stage (experiments.md s25), lane A 2026-10-01"},
                            out / "B1" / fname)
                arts.append({"refit_date": str(s["cut"].date()), "path": fname, "max_train_date": s["max_train"],
                             "n_train": s["n_train"]})
            (out / "B1" / f"manifest_{c}.json").write_text(json.dumps({
                "model": "fg_make", "scheme": "S1", "fold": a.fold, "season": test_season, "key": c, "arm": arm,
                "feature_set": f"two_stage_{arm}", "features": [*FA, "fg_team_offset"],
                "shooter_key": "shot_shooter_id", "servable": False, "artifacts": arts}, indent=2), encoding="utf-8")
        pd.concat(pred_rows, ignore_index=True).to_parquet(out / f"preds_{a.fold}_s{a.seed}.parquet", index=False)
        if off_rows:
            pd.concat(off_rows, ignore_index=True).to_parquet(out / f"offsets_{a.fold}_s{a.seed}.parquet", index=False)
        (out / f"stageB_{a.fold}_s{a.seed}.json").write_text(json.dumps(fitted, indent=1, default=float),
                                                             encoding="utf-8")
        log(f"{arm} written to {out}", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
