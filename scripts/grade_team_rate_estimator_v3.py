"""grade_team_rate_estimator_v3.py -- the ONE blind grader for team-rate estimator ROUND 2
(docs/models/team_rate_estimator/experiments.md section 5). Arm labels are opaque keys.

  --q1   variance: per rate-side x games band (g1, g2-3, g4-6, g7+, pooled): calibration slope, variance
         ratio [var(c)+mean(v)]/split-half true var, z SD with z=(y-p)/sqrt(v+phi s^2); arm verdicts per 5.2;
         paired deviance vs V0.
  --q2   prior mean: paired P1-P0 deviance on game 1 and games 2-3; game-1 slope; Pd diagnostic for rim off.
  --q3   two-sided matchup deviance, final vs final+opp, paired.
  --guard FILE ARM   the band guard line for one arm.
Paired test everywhere: team-block bootstrap (1,000 draws) of the per-team summed difference.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

OUT = Path("results/team_rate_estimator")
EPS = 1e-6
N_BOOT = 1000
BANDS = ([-1, 0, 2, 5, 1000], ["g1", "g2-3", "g4-6", "g7+"])


def deviance(num, den, p, fam):
    num = np.asarray(num, float); den = np.asarray(den, float); p = np.asarray(p, float)
    b = np.asarray(fam) == "binom"
    pb = np.clip(p, EPS, 1 - EPS)
    y = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    a1 = np.where(num > 0, num * np.log(np.maximum(y, 1e-300) / pb), 0.0)
    a2 = np.where(den - num > 0, (den - num) * np.log(np.maximum(1 - y, 1e-300) / (1 - pb)), 0.0)
    mu = den * np.clip(p, EPS, None)
    c1 = np.where(num > 0, num * np.log(np.maximum(num, 1e-300) / np.maximum(mu, 1e-300)), 0.0)
    return np.where(b, 2 * (a1 + a2), 2 * (c1 - (num - mu)))


def prep(path):
    e = pd.read_parquet(path)
    e = e[e["den"] > 0].copy()
    e["p"] = e["L"] + e["c"]
    e["y"] = e["num"] / e["den"]
    e["rs"] = e["rate"] + "|" + e["side"]
    e["dev"] = deviance(e["num"], e["den"], e["p"], e["fam"].to_numpy())
    pb = np.clip(e["p"], EPS, 1 - EPS)
    e["s2"] = np.where(e["fam"] == "binom", pb * (1 - pb) / e["den"], np.clip(e["p"], EPS, None) / e["den"])
    e["band"] = pd.cut(e["j"], BANDS[0], labels=BANDS[1]).astype(str)
    return e


def true_var(e):
    out = {}
    base = e[e["arm"] == e["arm"].iloc[0]]
    for (fold, rs), d in base.groupby(["fold", "rs"]):
        lg = float(d["num"].sum() / d["den"].sum())
        hh = d.assign(h=d["j"] % 2).groupby(["team_id", "h"])[["num", "den"]].sum()
        hr = (hh["num"] / hh["den"] - lg).unstack("h").dropna()
        out[(fold, rs)] = float(max(np.cov(hr[0], hr[1])[0, 1], 1e-12))
    return out


def wls(y, x, w):
    sw = np.sqrt(w); Z = np.column_stack([np.ones(len(y)), x]) * sw[:, None]
    b, *_ = np.linalg.lstsq(Z, y * sw, rcond=None)
    r = y * sw - Z @ b; s2 = float(r @ r / max(len(y) - 2, 1))
    try:
        return float(b[1]), float(np.sqrt(s2 * np.linalg.inv(Z.T @ Z)[1, 1]))
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def band_table(e, tv):
    rows = []
    for (fold, rs, arm), d in e.groupby(["fold", "rs", "arm"]):
        for band, db in [("pooled", d)] + list(d.groupby("band")):
            sl, se = wls(db["y"].to_numpy(), db["p"].to_numpy(), db["den"].to_numpy()) if db["c"].std() > 1e-9 else (np.nan, np.nan)
            z = (db["y"] - db["p"]) / np.sqrt(db["v"] + db["phi"] * db["s2"])
            rows.append({"fold": fold, "rs": rs, "arm": arm, "band": band, "n": len(db), "slope": sl, "slope_se": se,
                         "underpowered": (not np.isfinite(se)) or se > 0.10,
                         "var_ratio": float((db["c"].var() + db["v"].mean()) / tv[(fold, rs)]), "z_sd": float(z.std())})
    return pd.DataFrame(rows)


def inband(x):
    return (x >= 0.9) & (x <= 1.1)


def paired(e, a0, a1, mask=None):
    """Per fold: team-block bootstrap of sum(dev_a0 - dev_a1) (positive = a1 better)."""
    d = e if mask is None else e[mask]
    pv = d.pivot_table(index=["fold", "rs", "game_id", "team_id"], columns="arm", values="dev")[[a0, a1]].dropna()
    pv["g"] = pv[a0] - pv[a1]
    rng = np.random.default_rng(20260930); out = {}
    for fold, dd in pv.groupby(level="fold"):
        t = dd.groupby(level="team_id")["g"].sum()
        b = rng.multinomial(len(t), np.full(len(t), 1 / len(t)), size=N_BOOT) @ t.to_numpy()
        out[fold] = {"diff": float(t.sum()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
                     "pct_of_a0": 100 * float(t.sum() / dd[a0].sum())}
    return out


def q1():
    e = prep(OUT / "estimates_q1_v3.parquet"); tv = true_var(e)
    B = band_table(e, tv); B.to_json(OUT / "grade_q1_bands_v3.json", orient="records", indent=1)
    F2 = B[B["fold"] == "F2"]
    med = F2.groupby(["arm", "band"]).agg(slope=("slope", "median"), var_ratio=("var_ratio", "median"),
                                          z_sd=("z_sd", "median")).reset_index()
    verdict = []
    for arm, d in med.groupby("arm"):
        bands = d[d["band"] != "pooled"]
        ok_v = bool(inband(bands["var_ratio"]).all()); ok_z = bool(inband(bands["z_sd"]).all())
        cells = F2[(F2["arm"] == arm) & (F2["band"] != "pooled")]
        n_in = int(inband(cells["var_ratio"]).sum() + inband(cells["z_sd"]).sum())
        pdv = paired(e, "V0", arm) if arm != "V0" else {"F2": {"diff": 0, "lo": 0, "hi": 0, "pct_of_a0": 0}, "F1": {"diff": 0, "lo": 0, "hi": 0, "pct_of_a0": 0}}
        verdict.append({"arm": arm, "var_all_bands": ok_v, "z_all_bands": ok_z, "passes": ok_v and ok_z,
                        "cells_in_band": f"{n_in}/{2 * len(cells)}",
                        "dev_vs_V0_F2": pdv["F2"], "dev_vs_V0_F1": pdv["F1"],
                        "dev_guard_fail": pdv["F2"]["hi"] < 0})
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(med.pivot_table(index="band", columns="arm", values=["var_ratio", "z_sd", "slope"]).round(3).to_string())
    for v in verdict:
        print(v)
    json.dump(verdict, open(OUT / "grade_q1_verdict_v3.json", "w"), indent=1, default=float)
    phi = e.groupby(["fold", "arm", "rs"])["phi"].first().unstack("arm")
    print(phi.loc["F2"].round(3).to_string())


def q2():
    e = prep(OUT / "estimates_q2_v3.parquet"); tv = true_var(e)
    res = {"g1": paired(e, "P0", "P1", e["j"] == 0), "g2-3": paired(e, "P0", "P1", e["j"].isin([1, 2])),
           "all": paired(e, "P0", "P1")}
    B = band_table(e, tv); B.to_json(OUT / "grade_q2_bands_v3.json", orient="records", indent=1)
    g1 = B[(B["band"] == "g1") & (B["fold"] == "F2")].groupby("arm")["slope"].median()
    res["g1_slope_median_F2"] = g1.to_dict()
    rim = B[(B["rs"] == "make_rim|off")][["fold", "arm", "band", "slope", "slope_se", "var_ratio", "z_sd"]]
    res["rim_off"] = rim.to_dict(orient="records")
    json.dump(res, open(OUT / "grade_q2_v3.json", "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("g1", "g2-3", "all", "g1_slope_median_F2")}, indent=1))
    print(rim.round(3).to_string())
    per = B[(B["band"] == "pooled") & (B["fold"] == "F2")].pivot_table(index="rs", columns="arm", values="slope")
    print(per.round(3).to_string())


def q3():
    e = prep(OUT / "estimates_q3_v3.parquet")
    off = e[e["side"] == "off"]
    dfn = e[e["side"] == "def"][["fold", "rate", "arm", "game_id", "team_id", "c"]].rename(columns={"team_id": "opp_id", "c": "c_opp"})
    m = off.merge(dfn, on=["fold", "rate", "arm", "game_id", "opp_id"])
    m["p"] = m["L"] + m["c"] + m["c_opp"]
    m["dev"] = deviance(m["num"], m["den"], m["p"], m["fam"].to_numpy())
    res = {"two_sided": paired(m, "final", "final+opp"), "one_sided": paired(e, "final", "final+opp")}
    json.dump(res, open(OUT / "grade_q3_v3.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1))


def guard(path, arm):
    e = prep(path); e = e[e["arm"] == arm]; tv = true_var(e)
    B = band_table(e, tv)
    B.to_json(OUT / f"grade_guard_{arm}_v3.json", orient="records", indent=1)
    for fold in ("F2", "F1"):
        g = B[B["fold"] == fold].groupby("band").agg(slope=("slope", "median"), n_under=("underpowered", "sum"),
                                                     var_ratio=("var_ratio", "median"), z_sd=("z_sd", "median"),
                                                     z_in=("z_sd", lambda s: int(inband(s).sum())),
                                                     v_in=("var_ratio", lambda s: int(inband(s).sum())))
        print(fold); print(g.round(3).to_string())


if __name__ == "__main__":
    if "--q1" in sys.argv:
        q1()
    elif "--q2" in sys.argv:
        q2()
    elif "--q3" in sys.argv:
        q3()
    elif "--guard" in sys.argv:
        i = sys.argv.index("--guard"); guard(sys.argv[i + 1], sys.argv[i + 2])
