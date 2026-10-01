"""
grade_g9_team_response_v1.py -- lane A day 2026-10-01: ONE grading script for every arm of
docs/models/fg_make/experiments.md section 23. Arms are read from a JSON map {label: {"F1": [s0_dir, s1_dir],
"F2": [...], "harness": [name_s0, name_s1]}}; the labels are replaced by blind codes for the computation and
restored only when the table is written.

Design level (both folds), from <dir>/preds_<fold>_s<seed>.parquet:
  log loss per class; attempt-weighted team-game slope of realised on predicted make rate per class; beta of the
  team-rate part (p - p_zero) in the joint WLS of realised on [p_zero, p - p_zero]; segments by month, offence
  prior quintile (as-of off_make_c and own-rating net), site; responsiveness (quintile slope).
Harness (fold 2): 1 - slope(Y on X_h) and the close lens from results/g9_team_response_v1/harness_<name>.parquet,
  paired against harness_S0 with a 200-rep game bootstrap.

    .venv/Scripts/python.exe scripts/grade_g9_team_response_v1.py --arms results/g9_team_response_v1/arms.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CBB_TRUTH", "verified_v1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "g9_team_response_v1"
CLS = ("FGA_rim", "FGA_jump2", "FGA_3")


def wslope(y, x, w):
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))


def wls(y, X, w):
    sw = np.sqrt(w)
    Z = np.column_stack([np.ones(len(y)), X]) * sw[:, None]
    return np.linalg.lstsq(Z, y * sw, rcond=None)[0]


def team_games(pr: pd.DataFrame) -> pd.DataFrame:
    pr = pr.assign(_bv=pr["p"] * (1 - pr["p"]))
    g = pr.groupby(["game_id", "off_team_id", "shot_class"]).agg(
        n=("y", "size"), y=("y", "mean"), p=("p", "mean"), pz=("p_zero", "mean"), bv=("_bv", "sum"),
        oc=("off_make_c", "first"), date=("game_date", "first")).reset_index()
    return g


def design_metrics(pr: pd.DataFrame, rating_q: pd.DataFrame | None) -> dict:
    out = {}
    pr = pr.copy()
    p = np.clip(pr["p"].to_numpy(), 1e-6, 1 - 1e-6)
    pr["ll"] = -(pr["y"] * np.log(p) + (1 - pr["y"]) * np.log(1 - p))
    tg = team_games(pr)
    tg["mon"] = pd.to_datetime(tg["date"]).dt.month
    for c in CLS:
        s = tg[tg["shot_class"] == c]
        w = s["n"].to_numpy(float)
        e = {"log_loss": float(pr.loc[pr["shot_class"] == c, "ll"].mean()), "n_team_games": int(len(s)),
             "tg_slope": wslope(s["y"].to_numpy(), s["p"].to_numpy(), w)}
        b = wls(s["y"].to_numpy(), np.column_stack([s["pz"], s["p"] - s["pz"]]), w)
        e["b_norate"], e["b_rate"] = float(b[1]), float(b[2])
        e["sd_rate_part"] = float(np.sqrt(np.cov(s["p"] - s["pz"], aweights=w)))
        # variance composition (round 2, s25): team-game residual variance vs its binomial expectation, and
        # the two-team shared residual covariance (G3's fitting basis)
        res_ = s["y"] - s["p"]
        e["resid_var"] = float(np.mean(res_ ** 2))
        e["binom_var"] = float(np.mean(s["bv"] / s["n"] ** 2))
        e["excess_var"] = e["resid_var"] - e["binom_var"]
        pr2 = s.assign(res=res_).sort_values(["game_id", "off_team_id"])
        cnt = pr2.groupby("game_id")["res"].transform("size")
        pr2 = pr2[cnt == 2]
        a_ = pr2.groupby("game_id")["res"].first().to_numpy(); b_ = pr2.groupby("game_id")["res"].last().to_numpy()
        e["shared_cov"] = float(np.mean((a_ - a_.mean()) * (b_ - b_.mean())))
        e["by_month"] = {}
        for m, sm in s.groupby("mon"):
            if len(sm) < 300:
                e["by_month"][int(m)] = {"n": int(len(sm)), "underpowered": True}
                continue
            wm = sm["n"].to_numpy(float)
            bm = wls(sm["y"].to_numpy(), np.column_stack([sm["pz"], sm["p"] - sm["pz"]]), wm)
            e["by_month"][int(m)] = {"n": int(len(sm)), "tg_slope": wslope(sm["y"].to_numpy(), sm["p"].to_numpy(), wm),
                                     "b_rate": float(bm[2])}
        # responsiveness: offence as-of make-rate quintile
        q = pd.qcut(s["oc"].rank(method="first"), 5, labels=False)
        qs = s.assign(q=q).groupby("q").apply(lambda z: pd.Series({
            "pred": np.average(z["p"], weights=z["n"]), "act": np.average(z["y"], weights=z["n"]), "n": len(z)}),
            include_groups=False)
        e["quintile_offrate"] = {int(k): v.to_dict() for k, v in qs.iterrows()}
        e["quintile_offrate_slope"] = float(np.polyfit(qs["pred"], qs["act"], 1)[0])
        if rating_q is not None:
            sr = s.merge(rating_q, left_on=["game_id", "off_team_id"], right_on=["game_id", "team_id"], how="inner")
            qr = sr.groupby("q").apply(lambda z: pd.Series({
                "pred": np.average(z["p"], weights=z["n"]), "act": np.average(z["y"], weights=z["n"]), "n": len(z)}),
                include_groups=False)
            e["quintile_rating"] = {int(k): v.to_dict() for k, v in qr.iterrows()}
            e["quintile_rating_slope"] = float(np.polyfit(qr["pred"], qr["act"], 1)[0])
            e["by_site"] = {}
            for site, ss in sr.groupby("site"):
                e["by_site"][site] = {"n": int(len(ss)), "tg_slope": wslope(ss["y"].to_numpy(), ss["p"].to_numpy(),
                                                                             ss["n"].to_numpy(float))}
        out[c] = e
    return out


def rating_quintiles() -> pd.DataFrame:
    """fold-2 only: offence team's as-of own-rating net quintile and site, from the served engine inputs."""
    import diag_aggregation_overspread_v1 as A
    d = A.base_frame()
    rows = [pd.DataFrame({"game_id": d["game_id"], "team_id": d["home_team_id"], "q": d["q_h"], "site": d["site"]}),
            pd.DataFrame({"game_id": d["game_id"], "team_id": d["away_team_id"], "q": d["q_a"], "site": d["site"]})]
    return pd.concat(rows, ignore_index=True)


def harness_metrics(names: dict, nb: int = 200) -> dict:
    import diag_aggregation_overspread_v1 as A
    d = A.base_frame()
    Y = d["margin"].to_numpy(float); C = d["close"].to_numpy(float); lined = d["close"].notna().to_numpy()
    X = {}
    for code, path in names.items():
        h = pd.read_parquet(path)
        h = h[h["arm"] == "FULL"]
        w = h.pivot_table(index="game_id", columns="side", values="ppp")
        X[code] = (67.875 * (w[0] - w[1])).reindex(d["game_id"]).to_numpy(float)

    def stats(idx):
        o = {}
        li = idx[lined[idx]]
        for k, x in X.items():
            o[f"oms_{k}"] = 1 - A.slope(Y[idx], x[idx])
            o[f"omsC_{k}"] = 1 - A.slope(C[li], x[li])
            o[f"sd_{k}"] = float(np.std(x[idx], ddof=1))
        for k in X:
            if k != "SERVED":
                o[f"d_{k}"] = o[f"oms_{k}"] - o["oms_SERVED"]
                o[f"dC_{k}"] = o[f"omsC_{k}"] - o["omsC_SERVED"]
        return o
    est = stats(np.arange(len(Y)))
    rng = np.random.default_rng(20261001)
    reps = [stats(rng.integers(0, len(Y), len(Y))) for _ in range(nb)]
    se = pd.DataFrame(reps).std(ddof=1).to_dict()
    return {"est": est, "se": se}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=Path, required=True)
    a = ap.parse_args()
    arms = json.loads(a.arms.read_text(encoding="utf-8"))
    labels = sorted(arms)
    rng = np.random.default_rng(len(labels) * 7919)
    codes = {lab: f"arm{c:02d}" for lab, c in zip(labels, rng.permutation(len(labels)))}
    rq = rating_quintiles()
    res = {"design": {}, "harness": None}
    for lab in labels:
        code = codes[lab]
        for fold in ("F1", "F2"):
            for si, dd in enumerate(arms[lab].get(fold, [])):
                pp = Path(dd) / f"preds_{fold}_s{si}.parquet"
                if not pp.exists():
                    res["design"][f"{code}|{fold}|s{si}"] = {"missing": str(pp)}
                    continue
                pr = pd.read_parquet(pp)
                res["design"][f"{code}|{fold}|s{si}"] = design_metrics(pr, rq if fold == "F2" else None)
    hn = {"SERVED": OUT.parent / "aggregation_v1" / "harness_S0.parquet"}
    for lab in labels:
        for si, nm in enumerate(arms[lab].get("harness", [])):
            p = OUT / f"harness_{nm}.parquet"
            if p.exists():
                hn[f"{codes[lab]}_s{si}"] = p
    res["harness"] = harness_metrics(hn)
    # unblind only now
    inv = {v: k for k, v in codes.items()}

    def unb(s: str) -> str:
        for c, lab in inv.items():
            s = s.replace(c, lab)
        return s
    res = json.loads(unb(json.dumps(res, default=float)))
    res["codes"] = codes
    (OUT / "grade_v1.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("wrote", OUT / "grade_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
