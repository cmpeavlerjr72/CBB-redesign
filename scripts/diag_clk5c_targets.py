"""
diag_clk5c_targets.py -- round 5c DIAGNOSIS ONLY, part 2: does a dispersion
FUNCTION have a real target, and is it a TEAM property?

It FITS NOTHING and SELECTS NOTHING. Companion to `diag_clk5c_q2.py`. Round 5b
section 22.4 left one question: "what state variable Q2's dip is a function
of". Before a round 5c is run, three things have to be measured:

  (T1) is the per-game latent variance a TEAM property at all? -- the
       between-team variance of the per-game latent estimate against a label
       PERMUTATION null (arm C1, a hierarchical per-team sigma, needs this);
  (T2) does a team's own AS-OF possession-count SD (a measured, walk-forward,
       pregame quantity -- not a knob) predict the per-game latent? (arm C2);
  (T3) how much of the per-quintile responsiveness miss would either close,
       read as the between-component ratio by bucket of the candidate feature.

Estimator, round 5b's B3 estimator UNCHANGED (`fit_b3`):
    s2_hat(g) = (rc_g^2 - V_iid(g)) / (3 V_iid(g) + mbar_g^2),  rc = rbar - mean(rbar)
an unbiased-in-expectation but very noisy per-game estimate; every reading below
is therefore an AVERAGE of it over many games, with its own bootstrap or
permutation noise attached.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, fn: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fn)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r5 = _load("exp_clk5", "exp_clk5_dispersion_bakeoff.py")
r5b = _load("exp_clk5b", "exp_clk5b_mean_consistent.py")
log = r5.log
OUT = ROOT / "data" / "processed" / "models" / "clock" / "v5c_diag"
MIN_PRIOR = 5


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    univ = pd.read_parquet(r5.UNIV_V2)
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    design = pd.read_parquet(r5.DESIGN)
    design = design[(design["period"] <= 2.0) & (design["game_id"].isin(cc))].copy()
    sched = r5.load_schedule("v3c_srfloor_P3_s1")

    te = design[design["season"] == 2025].sort_values(
        ["game_id", "period", "poss_index"], kind="stable").reset_index(drop=True)
    cuts = np.array([np.datetime64(r["refit_date"], "ns") for r in sched])
    sg = np.searchsorted(cuts, pd.to_datetime(te["game_date"]).to_numpy(),
                         side="right") - 1
    e = np.empty(len(te)); v = np.empty(len(te))
    for k in np.unique(sg):
        r = np.flatnonzero(sg == k)
        a1, a2 = r5b.node_moments_loc(sched[int(k)]["arm"], te.iloc[r],
                                      np.zeros(len(r)), np.zeros(len(r)),
                                      np.array([0.0]))
        e[r] = a1[:, 0]; v[r] = a2[:, 0] - a1[:, 0] ** 2
    te["e_min"] = e; te["v_min"] = v
    gt, _ = r5.summarise(te)

    rc = gt["rbar"].to_numpy() - gt["rbar"].mean()
    V = gt["V_iid"].to_numpy(); mb = gt["mbar"].to_numpy()
    gt["s2_hat"] = (rc ** 2 - V) / (3.0 * V + mb ** 2)

    # ---- per-team possession counts and the game's two teams ---------------
    tp = (te.groupby(["game_id", "offense_team_id"]).size()
            .rename("poss").reset_index())
    tp = tp.merge(te.groupby("game_id")["game_date"].first().reset_index(),
                  on="game_id")
    tp = tp.merge(gt[["game_id", "s2_hat", "tempo", "M"]], on="game_id")
    tp = tp.sort_values(["offense_team_id", "game_date"], kind="stable")

    # ---- (T1) is the latent a TEAM property? permutation null --------------
    g = tp.groupby("offense_team_id")
    tm = pd.DataFrame({"n": g.size(), "mean_s2": g["s2_hat"].mean()}).reset_index()
    tm = tm[tm["n"] >= 10]
    obs = float(np.var(tm["mean_s2"].to_numpy(), ddof=1))
    vals = tp["s2_hat"].to_numpy()
    keys = tp["offense_team_id"].to_numpy()
    keep = np.isin(keys, tm["offence_team_id"].to_numpy()
                   if "offence_team_id" in tm else tm["offense_team_id"].to_numpy())
    vals_k, keys_k = vals[keep], keys[keep]
    rng = np.random.default_rng(20260911)
    null = np.empty(400)
    for b in range(400):
        p = rng.permutation(vals_k)
        d = pd.DataFrame({"k": keys_k, "v": p}).groupby("k")["v"].mean()
        null[b] = float(np.var(d.to_numpy(), ddof=1))
    t1 = {"n_teams": int(len(tm)), "obs_between_team_var": obs,
          "perm_null_mean": float(null.mean()), "perm_null_sd": float(null.std(ddof=1)),
          "z": float((obs - null.mean()) / null.std(ddof=1)),
          "p_one_sided": float((null >= obs).mean())}
    log(f"T1 between-team var of s2_hat: obs {obs:.6g} null "
        f"{null.mean():.6g} +/- {null.std(ddof=1):.2g}  z {t1['z']:.2f}  "
        f"p {t1['p_one_sided']:.3f}")

    # ---- (T2) as-of possession-count SD, walk-forward, prior games only ----
    def asof(s: pd.Series) -> pd.Series:
        return s.shift(1).expanding(MIN_PRIOR).std(ddof=0)

    tp["asof_poss_sd"] = g["poss"].transform(asof)
    tp["asof_poss_mean"] = g["poss"].transform(
        lambda s: s.shift(1).expanding(MIN_PRIOR).mean())
    gm = tp.groupby("game_id").agg(
        asof_sd=("asof_poss_sd", "mean"), asof_n=("asof_poss_sd", "count"),
        asof_mean=("asof_poss_mean", "mean"))
    gm = gm[gm["asof_n"] == 2]
    gg = gt.merge(gm.reset_index(), on="game_id", how="inner")
    r_s2 = float(np.corrcoef(gg["asof_sd"], gg["s2_hat"])[0, 1])
    r_tempo = float(np.corrcoef(gg["asof_sd"], gg["tempo"])[0, 1])
    log(f"T2 n={len(gg)} games with both teams as-of; corr(asof_sd, s2_hat) "
        f"{r_s2:+.4f}; corr(asof_sd, tempo) {r_tempo:+.4f}")

    # ---- (T3) needed vs produced-between by bucket of each candidate -------
    rows = []
    for feat in ("asof_sd", "tempo", "asof_mean"):
        gg["b"] = pd.qcut(gg[feat], 5, labels=[f"B{i}" for i in range(1, 6)],
                          duplicates="drop").astype(str)
        for b, sub in gg.groupby("b"):
            rb = sub["rbar"].to_numpy()
            needed = float(np.var(rb, ddof=0))
            floor = float(sub["V_iid"].mean())
            rng2 = np.random.default_rng(20260911)
            bs = np.array([
                (lambda i: np.var(rb[i], ddof=0) - sub["V_iid"].to_numpy()[i].mean())(
                    rng2.integers(0, len(sub), len(sub))) for _ in range(600)])
            rows.append({"feature": feat, "bucket": b, "n_games": int(len(sub)),
                         "feat_mean": float(sub[feat].mean()),
                         "needed_var": needed, "iid_floor": floor,
                         "needed_tau2": needed - floor,
                         "needed_tau2_se": float(bs.std(ddof=1)),
                         "mean_s2_hat": float(sub["s2_hat"].mean()),
                         "mbar2": float(np.mean(sub["mbar"].to_numpy() ** 2))})
    t3 = pd.DataFrame(rows)
    t3["implied_cv_pct"] = 100.0 * np.sqrt(np.clip(t3["needed_tau2"], 0, None)
                                           / t3["mbar2"])
    t3.to_csv(OUT / "v5c_target_buckets.csv", index=False)
    pd.DataFrame([t1]).to_csv(OUT / "v5c_team_permutation.csv", index=False)
    gg[["game_id", "tempo", "asof_sd", "asof_mean", "s2_hat", "rbar",
        "V_iid", "mbar", "M"]].to_csv(OUT / "v5c_game_features.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(t3.to_string(index=False))
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
