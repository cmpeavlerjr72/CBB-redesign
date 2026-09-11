"""
diag_clk5c_q2.py -- round 5c DIAGNOSIS ONLY: the tempo-quintile-2 responsiveness band.

It FITS NOTHING and SELECTS NOTHING. It re-reads round 5b's already-fitted
parameters from `data/processed/models/clock/v5b_bakeoff/v5b_bakeoff_report.json`
and reuses round 5's / round 5b's code BYTE FOR BYTE (`exp_clk5_dispersion_bakeoff`
for the design loader, the S1 schedule, `per_unit`, `var_sum`, `summarise`;
`exp_clk5b_mean_consistent` for `node_moments_loc` and `sigma_m_rows`).

The question, from section 22.4: every arm that lands the primary sits at
1.157-1.178 in tempo quintile Q2 against a pre-registered [0.85, 1.15] band.
Is that excess

  (i)  an over-spread of the BETWEEN-game pace level at Q2 (the latent's sigma
       is too wide for those games),
  (ii) a WITHIN-game issue (the conditional law is wrong at Q2), or
  (iii) a BUCKETING ARTEFACT / noise (Q2's NEEDED SD is itself a noisy or
        unstable statistic)?

Method. The grade path's per-game identity is used unchanged:

    Var_arm(Dbar) over a set of games = mean_g[(eH_g + vF_g) / M_g^2]
        eH_g / M_g^2   the WITHIN (iid conditional) contribution
        vF_g / M_g^2   the BETWEEN (latent) contribution
    needed over the same set   = Var_g(rbar)
        mean_g(V_iid_g)             the iid sampling floor (within)
        Var_g(rbar) - mean_g(V_iid) the implied latent variance tau^2 (between)
    ratio = sqrt(produced / needed)      -- the bridge constant cancels

so every per-quintile number is an average or a variance over the games of that
quintile, and a GAME-BLOCK BOOTSTRAP over those games gives the noise on each.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, fn: str):
    p = ROOT / "scripts" / fn
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r5 = _load("exp_clk5", "exp_clk5_dispersion_bakeoff.py")
r5b = _load("exp_clk5b", "exp_clk5b_mean_consistent.py")
log = r5.log

OUT = ROOT / "data" / "processed" / "models" / "clock" / "v5c_diag"
NBOOT = 2000
BOOT_SEED = 20260911


def per_game_components(spec, te, seg, sched, z, w):
    """Per-game (eH, vF, M) for one arm, through round 5b's own node moments."""
    sigma, m = r5b.sigma_m_rows(spec, te)
    n = len(te)
    m1 = np.empty((n, len(z))); m2 = np.empty((n, len(z)))
    for k in np.unique(seg):
        r = np.flatnonzero(seg == k)
        a, b = r5b.node_moments_loc(sched[int(k)]["arm"], te.iloc[r],
                                    sigma[r], m[r], z)
        m1[r] = a; m2[r] = b
    u = r5.per_unit(te, m1, m2, ["game_id"])
    eH, vF = r5.var_sum(u, w)
    M = u["n"].to_numpy(dtype=np.float64)
    return pd.DataFrame({"game_id": u["game_id"].to_numpy(),
                         "within_g": eH / M ** 2, "between_g": vF / M ** 2,
                         "var_dbar_g": (eH + vF) / M ** 2})


COLS = ("rbar", "V_iid", "within_g", "between_g")


def _arrs(sub: pd.DataFrame) -> tuple:
    return tuple(sub[c].to_numpy(dtype=np.float64) for c in COLS)


def _stats(a: tuple) -> dict:
    """Every quantity the diagnosis reads, on one set of games."""
    rbar, V_iid, within, between = a
    needed = float(np.var(rbar, ddof=0))
    floor = float(V_iid.mean())
    prod_w = float(within.mean())
    prod_b = float(between.mean())
    prod = prod_w + prod_b
    return {"needed": needed, "needed_floor": floor,
            "needed_tau2": needed - floor,
            "produced": prod, "produced_within": prod_w,
            "produced_between": prod_b,
            "ratio": float(np.sqrt(prod / needed)) if needed > 0 else np.nan,
            "within_ratio": prod_w / floor if floor > 0 else np.nan,
            "between_ratio": (prod_b / (needed - floor)
                              if needed - floor > 0 else np.nan)}


def boot(a: tuple, nboot: int, seed: int) -> dict:
    """Game-block bootstrap: games drawn with replacement, everything a game
    carries moving with it. Returns SE and the 2.5/97.5 percentiles per key."""
    rng = np.random.default_rng(seed)
    n = len(a[0])
    arr = {k: np.empty(nboot) for k in
           ("ratio", "needed", "produced", "needed_tau2", "produced_between",
            "within_ratio", "between_ratio")}
    idx_all = rng.integers(0, n, size=(nboot, n))
    for b in range(nboot):
        i = idx_all[b]
        s = _stats(tuple(x[i] for x in a))
        for k in arr:
            arr[k][b] = s[k]
    out = {}
    for k, v in arr.items():
        v = v[np.isfinite(v)]
        out[f"{k}_se"] = float(np.std(v, ddof=1))
        out[f"{k}_lo"] = float(np.percentile(v, 2.5))
        out[f"{k}_hi"] = float(np.percentile(v, 97.5))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from cbb_sim.models import clock_v5 as c5
    z, w = c5.gh_nodes(9)

    univ = pd.read_parquet(r5.UNIV_V2)
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    log("loading design")
    design = pd.read_parquet(r5.DESIGN)
    design = design[(design["period"] <= 2.0) & (design["game_id"].isin(cc))].copy()
    sched = r5.load_schedule("v3c_srfloor_P3_s1")

    # ---- model-free replication check: does Q2's dip exist in other seasons?
    rep_rows = []
    for season, d in design.groupby("season"):
        g = d.groupby("game_id").agg(M=("game_id", "size"),
                                     tempo=("tempo_prior_game", "first"))
        g = g.dropna()
        if len(g) < 500:
            continue
        g["P"] = g["M"] / 2.0
        g["tq"] = pd.qcut(g["tempo"], 5,
                          labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
        rng = np.random.default_rng(BOOT_SEED)
        for q, sub in g.groupby("tq"):
            p = sub["P"].to_numpy()
            bs = np.array([np.std(p[rng.integers(0, len(p), len(p))], ddof=0)
                           for _ in range(500)])
            rep_rows.append({"season": int(season), "tempo_quintile": q,
                             "n_games": int(len(sub)),
                             "raw_poss_sd": float(np.std(p, ddof=0)),
                             "raw_poss_sd_se": float(np.std(bs, ddof=1)),
                             "mean_tempo": float(sub["tempo"].mean()),
                             "mean_poss": float(p.mean())})
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(OUT / "v5c_raw_poss_sd_by_season.csv", index=False)
    log("model-free per-season raw possession SD by tempo quintile:")
    print(rep.pivot(index="season", columns="tempo_quintile",
                    values="raw_poss_sd").to_string())

    # ---- the F2 test slice, exactly as round 5b built it -------------------
    te = design[design["season"] == 2025].copy()
    te = te.sort_values(["game_id", "period", "poss_index"],
                        kind="stable").reset_index(drop=True)
    cuts = np.array([np.datetime64(r["refit_date"], "ns") for r in sched])
    sg = np.searchsorted(cuts, pd.to_datetime(te["game_date"]).to_numpy(),
                         side="right") - 1
    assert (sg >= 0).all()
    e = np.empty(len(te)); v = np.empty(len(te))
    for k in np.unique(sg):
        r = np.flatnonzero(sg == k)
        a1, a2 = r5b.node_moments_loc(sched[int(k)]["arm"], te.iloc[r],
                                      np.zeros(len(r)), np.zeros(len(r)),
                                      np.array([0.0]))
        e[r] = a1[:, 0]; v[r] = a2[:, 0] - a1[:, 0] ** 2
    te["_seg"] = sg
    te["e_min"] = e; te["v_min"] = v
    te["resid"] = te["duration_s"].to_numpy(dtype=np.float64) - e
    seg = te["_seg"].to_numpy()
    gt, ot = r5.summarise(te)
    gt["tq"] = pd.qcut(gt["tempo"], 5,
                       labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
    gt["tdec"] = pd.qcut(gt["tempo"], 10,
                         labels=[f"D{i}" for i in range(1, 11)]).astype(str)
    gt["mq"] = pd.qcut(gt["mbar"], 5,
                       labels=[f"M{i}" for i in range(1, 6)]).astype(str)
    kbridge = float(gt["P"].mean()) / float(te["duration_s"].mean())
    log(f"test {len(te)} poss / {len(gt)} games, bridge k {kbridge:.4f}")

    p2 = json.loads((ROOT / "data/processed/models/clock/v5b_bakeoff/"
                     "v5b_bakeoff_report.json").read_text())["params"]["F2"]
    arms = [
        ("R", {"kind": "none"}),
        ("A1", {"kind": "b", "loc_kind": "minus_half", "sigma": p2["A1_sigma"],
                "m": -0.5 * p2["A1_sigma2"]}),
        ("B1", {"kind": "b", "loc_kind": "plus_half", "sigma": p2["B1_sigma"],
                "m": p2["B1_m"]}),
        ("B3", {"kind": "b3", "beta0": p2["B3_beta0"], "beta1": p2["B3_beta1"]}),
    ]

    rows, dec_rows, mq_rows = [], [], []
    for name, spec in arms:
        comp = per_game_components(spec, te, seg, sched, z, w)
        g = gt.merge(comp, on="game_id", how="inner")
        assert len(g) == len(gt)
        for label, col in (("ALL", None), ("tq", "tq")):
            groups = [("ALL", g)] if col is None else list(g.groupby(col))
            for q, sub in groups:
                aa = _arrs(sub)
                s = _stats(aa)
                b = boot(aa, NBOOT, BOOT_SEED + 3)
                rows.append({"arm": name, "bucket": q, "n_games": int(len(sub)),
                             "mean_tempo": float(sub["tempo"].mean()),
                             "P_sd_needed": kbridge * np.sqrt(s["needed"]),
                             "P_sd_produced": kbridge * np.sqrt(s["produced"]),
                             **s, **b})
        for q, sub in g.groupby("tdec"):
            aa = _arrs(sub)
            s = _stats(aa)
            b = boot(aa, 400, BOOT_SEED + 7)
            dec_rows.append({"arm": name, "decile": q, "n_games": int(len(sub)),
                             "mean_tempo": float(sub["tempo"].mean()),
                             **s, "ratio_se": b["ratio_se"]})
        for q, sub in g.groupby("mq"):
            aa = _arrs(sub)
            s = _stats(aa)
            b = boot(aa, 400, BOOT_SEED + 11)
            mq_rows.append({"arm": name, "mbar_quintile": q,
                            "n_games": int(len(sub)), **s,
                            "ratio_se": b["ratio_se"]})
        log(f"{name} done")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "v5c_quintile_ratios.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(OUT / "v5c_decile_ratios.csv", index=False)
    pd.DataFrame(mq_rows).to_csv(OUT / "v5c_mbar_quintile_ratios.csv", index=False)

    with pd.option_context("display.width", 260, "display.max_columns", 60):
        print("\n=== produced/needed ratio with game-block bootstrap noise ===")
        print(res[["arm", "bucket", "n_games", "P_sd_needed", "P_sd_produced",
                   "ratio", "ratio_se", "ratio_lo", "ratio_hi"]]
              .to_string(index=False))
        print("\n=== the decomposition (variance units, s^2) ===")
        print(res[["arm", "bucket", "needed", "needed_floor", "needed_tau2",
                   "produced_within", "produced_between", "within_ratio",
                   "between_ratio", "needed_tau2_se", "produced_between_se"]]
              .to_string(index=False))
        print("\n=== tempo DECILES (bucketing check) ===")
        print(pd.DataFrame(dec_rows).pivot(index="arm", columns="decile",
                                           values="ratio").to_string())
        print("\n=== bucketed by the MODEL's own mean duration mbar ===")
        print(pd.DataFrame(mq_rows).pivot(index="arm", columns="mbar_quintile",
                                          values="ratio").to_string())
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
