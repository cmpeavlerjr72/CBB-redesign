"""grade_team_rate_estimator_v2.py -- the ONE blind grader (v2: section 3 guards added; v1 metrics unchanged) for the team-rate estimator Stage A
(docs/models/team_rate_estimator/experiments.md section 1). Scores every arm in
results/team_rate_estimator/estimates_v1.parquet identically; arm labels are opaque keys.

Metrics (per fold x arm x rate-side, and pooled over the 16 rate-sides):
  * next-game predictive deviance (binomial / Poisson) and the paired gain vs the reference arm E0;
  * team-block bootstrap (teams resampled with replacement, 1,000 draws) 95% interval of the gain;
  * weeks bands 0-3 / 4-7 / 8-15 / 16+;
  * day-0 cells: a team's game 1, games 2-3, games 4-6: gain and team slope (realised on predicted, WLS);
  * G-A1 level-SD ratio: SD over teams of the mean as-of estimate / SD over teams of the realised
    centred season rate (teams with >= 10 scored games);
  * G-A2 lag-1 movement ratio b_move / b_level from (y - L) ~ c_lag1 + (c - c_lag1), WLS by denominator.

Usage:
    .venv/Scripts/python.exe scripts/grade_team_rate_estimator_v1.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

IN = Path("results/team_rate_estimator/estimates_v2.parquet")
OUT = Path("results/team_rate_estimator")
REF = "E0"
TAG_JSON = "grade_v2.json"
BANDS = ([-1, 0, 2, 5, 1000], ["g1", "g2-3", "g4-6", "g7+"])
EPS = 1e-6
N_BOOT = 1000


def deviance(num, den, p, fam):
    num = np.asarray(num, float); den = np.asarray(den, float); p = np.asarray(p, float)
    out = np.zeros(len(num))
    b = fam == "binom"
    pb = np.clip(p, EPS, 1 - EPS)
    y = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    a1 = np.where(num > 0, num * np.log(np.maximum(y, 1e-300) / pb), 0.0)
    a2 = np.where(den - num > 0, (den - num) * np.log(np.maximum(1 - y, 1e-300) / (1 - pb)), 0.0)
    out[b] = (2 * (a1 + a2))[b]
    pp = np.clip(p, EPS, None); mu = den * pp
    c1 = np.where(num > 0, num * np.log(np.maximum(num, 1e-300) / np.maximum(mu, 1e-300)), 0.0)
    out[~b] = (2 * (c1 - (num - mu)))[~b]
    return out


def wls_slope(y, X, w):
    sw = np.sqrt(w)
    Z = np.column_stack([np.ones(len(y)), X]) * sw[:, None]
    b, *_ = np.linalg.lstsq(Z, y * sw, rcond=None)
    return b


def main():
    e = pd.read_parquet(IN)
    e = e[e["den"] > 0].copy()
    e["p"] = e["L"] + e["c"]
    e["dev"] = deviance(e["num"], e["den"], e["p"], e["fam"].to_numpy())
    e["y"] = e["num"] / e["den"]
    e["rs"] = e["rate"] + "|" + e["side"]
    arms = sorted(e["arm"].unique(), key=lambda a: (a != REF, a))
    key = ["fold", "rs", "season", "game_id", "team_id"]
    ref = e[e["arm"] == REF].set_index(key)["dev"].rename("dev_ref")
    e = e.join(ref, on=key)
    e["gain"] = e["dev_ref"] - e["dev"]
    rng = np.random.default_rng(20260930)
    res = {"pooled": [], "per_rs": [], "weeks": [], "day0": [], "guards": []}
    for fold, ef in e.groupby("fold"):
        teams = np.sort(ef["team_id"].unique())
        tix = {t: i for i, t in enumerate(teams)}
        W = rng.multinomial(len(teams), np.full(len(teams), 1 / len(teams)), size=N_BOOT).astype(float)
        for arm in arms:
            ea = ef[ef["arm"] == arm]
            tot_ref = float(ea["dev_ref"].sum())
            per_team = ea.groupby(["team_id", "rs"])["gain"].sum().unstack("rs").fillna(0.0)
            M = np.zeros((len(teams), per_team.shape[1]))
            M[[tix[t] for t in per_team.index]] = per_team.to_numpy()
            boots = W @ M
            pooled = boots.sum(axis=1)
            g = float(M.sum())
            res["pooled"].append({"fold": fold, "arm": arm, "gain": g, "gain_pct": 100 * g / tot_ref,
                                  "lo": float(np.percentile(pooled, 2.5)), "hi": float(np.percentile(pooled, 97.5))})
            for i, rs in enumerate(per_team.columns):
                gi = float(M[:, i].sum()); dr = float(ea.loc[ea["rs"] == rs, "dev_ref"].sum())
                res["per_rs"].append({"fold": fold, "arm": arm, "rs": rs, "gain": gi, "gain_pct": 100 * gi / dr,
                                      "lo": float(np.percentile(boots[:, i], 2.5)), "hi": float(np.percentile(boots[:, i], 97.5))})
            wb = pd.cut(ea["week"], [-1, 3, 7, 15, 100], labels=["0-3", "4-7", "8-15", "16+"])
            for w, d in ea.groupby(wb, observed=True):
                res["weeks"].append({"fold": fold, "arm": arm, "weeks": str(w), "n": len(d),
                                     "gain": float(d["gain"].sum()), "gain_pct": 100 * float(d["gain"].sum() / d["dev_ref"].sum())})
            cell = pd.cut(ea["j"], [-1, 0, 2, 5], labels=["g1", "g2-3", "g4-6"])
            for (cl, rs), d in ea.groupby([cell, "rs"], observed=True):
                sd_c = float(d["c"].std())
                slope = float(wls_slope(d["y"].to_numpy(), d["p"].to_numpy(), d["den"].to_numpy())[1]) if sd_c > 1e-9 else np.nan
                res["day0"].append({"fold": fold, "arm": arm, "cell": str(cl), "rs": rs, "n": len(d),
                                    "gain": float(d["gain"].sum()), "gain_pct": 100 * float(d["gain"].sum() / d["dev_ref"].sum()),
                                    "slope": slope, "sd_pred": sd_c})
            for rs, d in ea.groupby("rs"):
                tm = d.groupby("team_id").agg(c=("c", "mean"), num=("num", "sum"), den=("den", "sum"), n=("c", "size"))
                tm = tm[tm["n"] >= 10]
                lg = float(d["num"].sum() / d["den"].sum())
                real = tm["num"] / tm["den"] - lg
                ratio = float(tm["c"].std() / real.std())
                dl = d[d["c_lag1"].notna()]
                b = wls_slope((dl["y"] - dl["L"]).to_numpy(), np.column_stack([dl["c_lag1"], dl["c"] - dl["c_lag1"]]),
                              dl["den"].to_numpy())
                res["guards"].append({"fold": fold, "arm": arm, "rs": rs, "level_sd_ratio": ratio,
                                      "b_level": float(b[1]), "b_move": float(b[2]),
                                      "move_ratio": float(b[2] / b[1]) if b[1] != 0 else np.nan})
    json.dump(res, open(OUT / TAG_JSON, "w"), indent=1, default=float)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30)
    P = pd.DataFrame(res["pooled"]); print(P.round(2).to_string())
    Gd = pd.DataFrame(res["guards"])
    summ = Gd.groupby(["fold", "arm"]).agg(level_ratio_med=("level_sd_ratio", "median"), move_ratio_med=("move_ratio", "median")).reset_index()
    print(summ.round(3).to_string())
    print(pd.DataFrame(res["weeks"]).pivot_table(index=["fold", "arm"], columns="weeks", values="gain_pct").round(3).to_string())
    D = pd.DataFrame(res["day0"])
    print(D.groupby(["fold", "arm", "cell"]).agg(gain=("gain", "sum"), slope_med=("slope", "median")).round(3).unstack("cell").to_string())


def wls_se(y, x, w):
    sw = np.sqrt(w)
    Z = np.column_stack([np.ones(len(y)), x]) * sw[:, None]
    b, *_ = np.linalg.lstsq(Z, y * sw, rcond=None)
    r = y * sw - Z @ b
    s2 = float(r @ r / max(len(y) - 2, 1))
    try:
        cov = s2 * np.linalg.inv(Z.T @ Z)
        return float(b[1]), float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def section3(e_path=None, out_json="grade_s3_v2.json"):
    """Section 3 guards. G-A1a calibration slope (WLS of y on p) per rate-side x games band, UNDERPOWERED if
    slope SE > 0.10 or predictions have no spread. G-A1b (i) [var(c) + mean(v)] / split-half true between-team
    variance; (ii) SD of z = (y - p)/sqrt(v + s^2), s^2 the next-game sampling variance at p.
    Arm-level verdict per check: median over rate-sides (non-underpowered for G-A1a) within [0.90, 1.10]."""
    e = pd.read_parquet(e_path or IN)
    e = e[e["den"] > 0].copy()
    e["p"] = e["L"] + e["c"]
    e["y"] = e["num"] / e["den"]
    e["rs"] = e["rate"] + "|" + e["side"]
    pb = np.clip(e["p"], EPS, 1 - EPS)
    e["s2"] = np.where(e["fam"] == "binom", pb * (1 - pb) / e["den"], np.clip(e["p"], EPS, None) / e["den"])
    e["band"] = pd.cut(e["j"], BANDS[0], labels=BANDS[1]).astype(str)
    rows, calib = [], []
    for (fold, rs), d0 in e.groupby(["fold", "rs"]):
        ref = d0[d0["arm"] == d0["arm"].iloc[0]]
        lg = float(ref["num"].sum() / ref["den"].sum())
        hh = ref.assign(h=ref["j"] % 2).groupby(["team_id", "h"])[["num", "den"]].sum()
        hr = (hh["num"] / hh["den"] - lg).unstack("h").dropna()
        true_var = float(max(np.cov(hr[0], hr[1])[0, 1], 1e-12))
        for arm, d in d0.groupby("arm"):
            for band, db in [("pooled", d)] + list(d.groupby("band")):
                sd_c = float(db["c"].std())
                if sd_c > 1e-9:
                    b, se = wls_se(db["y"].to_numpy(), db["p"].to_numpy(), db["den"].to_numpy())
                else:
                    b, se = np.nan, np.nan
                under = (not np.isfinite(se)) or se > 0.10
                dv = db[db["v"].notna()]
                z = (dv["y"] - dv["p"]) / np.sqrt(dv["v"] + dv["s2"])
                z0 = (db["y"] - db["p"]) / np.sqrt(db["s2"])
                rows.append({"fold": fold, "rs": rs, "arm": arm, "band": band, "n": len(db), "n_v": len(dv),
                             "slope": b, "slope_se": se, "underpowered": bool(under),
                             "var_ratio": float((dv["c"].var() + dv["v"].mean()) / true_var) if len(dv) > 2 else np.nan,
                             "sd_c_over_true": float(sd_c / np.sqrt(true_var)),
                             "z_sd": float(z.std()) if len(dv) > 2 else np.nan, "z_sd_no_v": float(z0.std()),
                             "true_sd": float(np.sqrt(true_var))})
    R_ = pd.DataFrame(rows)
    R_.to_json(OUT / out_json, orient="records", indent=1)
    pool = R_[R_["band"] == "pooled"]
    summ = []
    for (fold, arm), d in pool.groupby(["fold", "arm"]):
        ok = d[~d["underpowered"]]
        summ.append({"fold": fold, "arm": arm,
                     "A1a_median_slope": float(ok["slope"].median()) if len(ok) else np.nan,
                     "A1a_in_band": f"{int(((ok['slope'] >= 0.9) & (ok['slope'] <= 1.1)).sum())}/{len(ok)}",
                     "A1b_i_median": float(d["var_ratio"].median()),
                     "A1b_i_in_band": f"{int(((d['var_ratio'] >= 0.9) & (d['var_ratio'] <= 1.1)).sum())}/{len(d)}",
                     "A1b_ii_median": float(d["z_sd"].median()),
                     "A1b_ii_in_band": f"{int(((d['z_sd'] >= 0.9) & (d['z_sd'] <= 1.1)).sum())}/{len(d)}",
                     "z_sd_no_v_median": float(d["z_sd_no_v"].median())})
    S = pd.DataFrame(summ)
    for c_, lo, hi in (("A1a_median_slope", .9, 1.1), ("A1b_i_median", .9, 1.1), ("A1b_ii_median", .9, 1.1)):
        S[c_ + "_pass"] = (S[c_] >= lo) & (S[c_] <= hi)
    S.to_json(OUT / out_json.replace(".json", "_summary.json"), orient="records", indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
    print(S.round(3).to_string())
    bands = R_[R_["fold"] == "F2"].groupby(["arm", "band"]).agg(slope_med=("slope", "median"),
                                                                  n_under=("underpowered", "sum"),
                                                                  var_ratio_med=("var_ratio", "median"),
                                                                  z_sd_med=("z_sd", "median")).round(3)
    print(bands.to_string())
    return R_, S



def e9_two_sided(path="results/team_rate_estimator/estimates_e9_v2.parquet"):
    """POST-HOC SECONDARY (not the registered primary): E9 scored as the sub-models consume the features,
    p = L + c_side(team) + c_opposite(opponent), both from the same arm, on the offence-side rows (the def-side
    rows are the same games seen from the other team). Deviance gain of E3+opp vs E3, team-block bootstrap."""
    e = pd.read_parquet(path)
    off = e[e["side"] == "off"]; dfn = e[e["side"] == "def"][["fold", "rate", "arm", "game_id", "team_id", "c"]]
    dfn = dfn.rename(columns={"team_id": "opp_id", "c": "c_opp"})
    m = off.merge(dfn, on=["fold", "rate", "arm", "game_id", "opp_id"], how="inner")
    m = m[m["den"] > 0].copy()
    m["p"] = m["L"] + m["c"] + m["c_opp"]
    m["dev"] = deviance(m["num"], m["den"], m["p"], m["fam"].to_numpy())
    piv = m.pivot_table(index=["fold", "rate", "game_id", "team_id"], columns="arm", values="dev").dropna()
    arms = list(piv.columns); base = [a for a in arms if "+opp" not in a][0]; adj = [a for a in arms if "+opp" in a][0]
    piv["gain"] = piv[base] - piv[adj]
    rng = np.random.default_rng(7)
    out = []
    for fold, d in piv.groupby(level="fold"):
        t = d.groupby(level="team_id")["gain"].sum()
        W = rng.multinomial(len(t), np.full(len(t), 1 / len(t)), size=N_BOOT)
        b = W @ t.to_numpy()
        out.append({"fold": fold, "gain": float(t.sum()), "gain_pct": 100 * float(t.sum() / d[base].sum()),
                    "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))})
    print("POSTHOC two-sided E9 vs base:", out)
    json.dump(out, open(OUT / "grade_e9_two_sided_v2.json", "w"), indent=1)



def paired(a1, a2):
    """POST-HOC INFORMATION (not in the decision rule): paired deviance gain of a2 over a1, team-block bootstrap."""
    e = pd.read_parquet(IN, columns=["fold", "rate", "side", "arm", "game_id", "team_id", "num", "den", "L", "c", "fam"])
    e = e[(e["den"] > 0) & e["arm"].isin([a1, a2])].copy()
    e["dev"] = deviance(e["num"], e["den"], e["L"] + e["c"], e["fam"].to_numpy())
    pv = e.pivot_table(index=["fold", "rate", "side", "game_id", "team_id"], columns="arm", values="dev").dropna()
    pv["g"] = pv[a1] - pv[a2]
    rng = np.random.default_rng(11); out = []
    for fold, d in pv.groupby(level="fold"):
        t = d.groupby(level="team_id")["g"].sum()
        b = rng.multinomial(len(t), np.full(len(t), 1 / len(t)), size=N_BOOT) @ t.to_numpy()
        out.append({"fold": fold, "gain": float(t.sum()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))})
    print(f"POSTHOC paired {a2} over {a1}:", out)
    json.dump(out, open(OUT / f"grade_paired_{a2}_over_{a1}_v2.json", "w"), indent=1)


def posthoc():
    """POST-HOC, NOT PRE-REGISTERED (added after section-1 G-A1 was read): the registered G-A1 compares
    against RAW realised season rates, whose SD carries sampling noise, and against E0, whose ratio > 1 is
    itself inflated by few-game noise. Two noise-aware variants, reported for the PM, never used to select:
      (a) SD over teams of the estimate entering games j >= 20 (estimates built on >= 20 games), mean per team,
          / split-half true SD (sqrt cov(odd-game rate, even-game rate), centred on the league);
      (b) SD of the team's LAST as-of estimate / the same true SD."""
    e = pd.read_parquet(IN)
    e = e[e["den"] > 0].copy()
    e["rs"] = e["rate"] + "|" + e["side"]
    rows = []
    for (fold, rs), d0 in e.groupby(["fold", "rs"]):
        ref = d0[d0["arm"] == REF]
        lg = float(ref["num"].sum() / ref["den"].sum())
        ref = ref.assign(h=(ref["j"] % 2))
        hh = ref.groupby(["team_id", "h"])[["num", "den"]].sum()
        hr = (hh["num"] / hh["den"] - lg).unstack("h").dropna()
        true_sd = float(np.sqrt(max(np.cov(hr[0], hr[1])[0, 1], 0)))
        for arm, d in d0.groupby("arm"):
            late = d[d["j"] >= 20].groupby("team_id")["c"].mean()
            last = d.sort_values("j").groupby("team_id")["c"].last()
            rows.append({"fold": fold, "rs": rs, "arm": arm, "true_sd": true_sd,
                         "ratio_late": float(late.std() / true_sd), "ratio_last": float(last.std() / true_sd)})
    P = pd.DataFrame(rows)
    P.to_json(OUT / "grade_posthoc_v2.json", orient="records", indent=1)
    print("POSTHOC (not pre-registered)")
    print(P.groupby(["fold", "arm"])[["ratio_late", "ratio_last"]].median().round(3).to_string())
    print(P[P["fold"] == "F2"].pivot_table(index="rs", columns="arm", values="ratio_late").round(3).to_string())


if __name__ == "__main__":
    import sys
    if "--posthoc" in sys.argv:
        posthoc()
    elif "--pair" in sys.argv:
        i = sys.argv.index("--pair"); paired(sys.argv[i + 1], sys.argv[i + 2])
    elif "--e9two" in sys.argv:
        e9_two_sided()
    elif "--s3" in sys.argv:
        section3()
    elif "--e9" in sys.argv:
        IN = Path("results/team_rate_estimator/estimates_e9_v2.parquet")
        REF = sys.argv[sys.argv.index("--e9") + 1]
        TAG_JSON = "grade_e9_v2.json"
        main()
        section3(IN, "grade_e9_s3_v2.json")
    else:
        main()
        section3()
