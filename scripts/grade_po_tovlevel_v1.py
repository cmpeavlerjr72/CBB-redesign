"""grade_po_tovlevel_v1.py -- the ONE blind grader of PO section 30 (TOV level round). PO worker, 2026-10-05.

Every arm goes through the same functions. Inputs: `pred_first_<fold>.npy` of each arm (rows = the round-2 design's
`first` rows of the test season in design order) and the trainers' reports (R1.score gates). Lines per fold:
TOV-binary log loss (primary), multiclass log loss, TOV gap (mean p - realised, pp) by days bucket / month / site,
prior-season team-TOV quintile slope (overall, d0-45), team spread ratio, per-game TOV level MAE; paired game-block
bootstrap (200 reps, seed 12345) SE of every (arm - T0) difference; floors from the T1 seed-1 refit; the section-30
decision rule. Writes results/po_tovlevel/grade_v1.{json,md}.

    .venv/Scripts/python.exe scripts/grade_po_tovlevel_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

DESIGN = ROOT / "data/processed/models/possession_outcome/round2/design.parquet"
TL = ROOT / "data/processed/models/possession_outcome/tovlevel"
FOLDS = {"F1": 2024, "F2": 2025}
ARMS = {
    "F1": {"T0": ROOT / "data/processed/models/fold1_v1/po", "T1": TL / "T1_F1_s0", "T2": TL / "T2_F1_s0",
           "T1_seed1": TL / "T1_F1_s1"},
    "F2": {"T0": ROOT / "data/processed/models/full_retrain_v1/identity/po_served", "T1": TL / "T1_F2_s0",
           "T2": TL / "T2_F2_s0", "T1_seed1": TL / "T1_F2_s1"},
}
BUCKETS = [("d0-14", -1, 14), ("d15-45", 14, 45), ("d46+", 45, 10_000)]
PUB_FLOOR_MULTI = 0.000804
OUT = ROOT / "results/po_tovlevel"


def load_test(season: int, prior: pd.Series) -> pd.DataFrame:
    assert_not_sealed(season)
    d = pd.read_parquet(DESIGN, columns=["season", "population", "y", "days_since_start", "game_id",
                                         "offense_team_id", "site_home", "site_away", "game_date"])
    te = d[(d["season"] == season) & (d["population"] == "first")].reset_index(drop=True)
    te["tov"] = (te["y"] == 0).astype("float64")
    te["bucket"] = pd.cut(te["days_since_start"], [b[1] for b in BUCKETS] + [BUCKETS[-1][2]],
                          labels=[b[0] for b in BUCKETS]).astype(str)
    te["month"] = pd.to_datetime(te["game_date"]).dt.strftime("%Y-%m")
    te["site"] = np.where(te["site_home"] == 1, "home", np.where(te["site_away"] == 1, "away", "neutral"))
    te["prior_tov"] = te["offense_team_id"].map(prior)
    return te


def prior_team_tov(season: int) -> pd.Series:
    d = pd.read_parquet(DESIGN, columns=["season", "population", "y", "offense_team_id"])
    p = d[(d["season"] == season - 1) & (d["population"] == "first")]
    return p.groupby("offense_team_id")["y"].apply(lambda s: float((s == 0).mean()))


def per_game(te: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    p0 = np.clip(P[:, 0], 1e-12, 1 - 1e-12)
    py = np.clip(P[np.arange(len(P)), te["y"].to_numpy().astype(int)], 1e-12, None)
    t = te["tov"].to_numpy()
    g = pd.DataFrame({"game_id": te["game_id"], "n": 1.0, "t": t, "p": P[:, 0],
                      "ll_tov": -(t * np.log(p0) + (1 - t) * np.log(1 - p0)), "ll_multi": -np.log(py)})
    return g.groupby("game_id", sort=True).sum()


def stat_lines(te: pd.DataFrame, P: np.ndarray) -> dict:
    out = {}
    pg = per_game(te, P)
    out["ll_tov"] = pg["ll_tov"].sum() / pg["n"].sum()
    out["ll_multi"] = pg["ll_multi"].sum() / pg["n"].sum()
    out["gap_all"] = 100 * (pg["p"].sum() - pg["t"].sum()) / pg["n"].sum()
    x = te.assign(p=P[:, 0])
    for k, col in (("bucket", "bucket"), ("month", "month"), ("site", "site")):
        g = x.groupby(col)[["p", "tov"]].mean()
        for idx, r in g.iterrows():
            out[f"gap_{k}_{idx}" if k != "bucket" else f"gap_{idx}"] = 100 * (r["p"] - r["tov"])
    out["pg_level_mae_pp"] = 100 * float(np.mean(np.abs((pg["p"] - pg["t"]) / pg["n"])))
    # responsiveness: prior-season team TOV quintile
    for lab, m in (("all", np.ones(len(x), bool)), ("d0-45", (x["days_since_start"] <= 45).to_numpy())):
        y = x[m & x["prior_tov"].notna().to_numpy()]
        teams = y.groupby("offense_team_id")["prior_tov"].first()
        q = pd.qcut(teams, 5, labels=False)
        y = y.assign(q=y["offense_team_id"].map(q))
        g = y.groupby("q")[["p", "tov"]].mean()
        out[f"slope_{lab}"] = float((g["p"].max() - g["p"].min()) / (g["tov"].max() - g["tov"].min()))
        out[f"quintiles_{lab}"] = {int(i): [round(float(r["p"]), 5), round(float(r["tov"]), 5)] for i, r in g.iterrows()}
    tm = x.groupby("offense_team_id").agg(n=("tov", "size"), p=("p", "mean"), t=("tov", "mean"))
    tm = tm[tm["n"] >= 300]
    noise = float(np.mean(tm["t"] * (1 - tm["t"]) / tm["n"]))
    out["spread_ratio"] = float(tm["p"].std() / np.sqrt(max(tm["t"].var() - noise, 1e-12)))
    return out


def boot_diffs(te, P_arm, P_ref, reps=200, seed=12345) -> dict:
    """Paired game-block bootstrap SE of (arm - ref) for log losses and bucket gaps."""
    a, r = per_game(te, P_arm), per_game(te, P_ref)
    bk = te.groupby("game_id", sort=True)["bucket"].first().reindex(a.index).to_numpy()
    rng = np.random.default_rng(seed)
    G = len(a)
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(reps)]).astype("float64")
    n = W @ a["n"].to_numpy()
    res = {"ll_tov": (W @ (a["ll_tov"] - r["ll_tov"]).to_numpy()) / n,
           "ll_multi": (W @ (a["ll_multi"] - r["ll_multi"]).to_numpy()) / n,
           "gap_all": 100 * (W @ (a["p"] - r["p"]).to_numpy()) / n}
    for b, _, _ in BUCKETS:
        m = (bk == b).astype("float64")
        res[f"gap_{b}"] = 100 * (W @ ((a["p"] - r["p"]).to_numpy() * m)) / (W @ (a["n"].to_numpy() * m))
    return {k: float(np.std(v, ddof=1)) for k, v in res.items()}


def gates(arm_dir: Path) -> dict:
    for name in ("par_anchor_artifacts_v1_report.json", "par_v1_report.json"):
        p = arm_dir / name
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))["scores"]["first"]
            return {k: s[k] for k in ("log_loss", "calibration_pass", "worst_gated_gap_pp", "responsiveness_pass")}
    return {}


def main() -> int:
    res = {}
    for fold, season in FOLDS.items():
        te = load_test(season, prior_team_tov(season))
        preds = {}
        for arm, d in ARMS[fold].items():
            p = d / f"pred_first_{fold}.npy"
            if p.exists():
                P = np.load(p).astype("float64")
                assert P.shape == (len(te), 6), (arm, P.shape)
                preds[arm] = P
        fr = {"n_rows": int(len(te)), "n_games": int(te["game_id"].nunique()), "arms": {}}
        for arm, P in preds.items():
            fr["arms"][arm] = {"lines": stat_lines(te, P), "gates": gates(ARMS[fold][arm])}
            if arm != "T0":
                fr["arms"][arm]["boot_se_vs_T0"] = boot_diffs(te, P, preds["T0"])
        if "T1_seed1" in preds and "T1" in preds:
            L1, L0 = fr["arms"]["T1_seed1"]["lines"], fr["arms"]["T1"]["lines"]
            fr["reseed"] = {k: abs(L1[k] - L0[k]) for k in ("ll_tov", "ll_multi", "gap_all", *[f"gap_{b}" for b, _, _ in BUCKETS])}
        res[fold] = fr
        print(fold, {a: round(v["lines"]["ll_tov"], 6) for a, v in fr["arms"].items()}, flush=True)

    # floors and decision (section 30)
    keys = ["ll_tov", "ll_multi", "gap_all", *[f"gap_{b}" for b, _, _ in BUCKETS]]
    dec = {}
    for fold in FOLDS:
        fr = res[fold]
        for arm in ("T1", "T2"):
            if arm not in fr["arms"]:
                continue
            se = fr["arms"][arm]["boot_se_vs_T0"]
            fl = {k: max(fr.get("reseed", {}).get(k, 0.0), 2 * se[k]) for k in keys}
            fl["ll_multi"] = max(fl["ll_multi"], PUB_FLOOR_MULTI)
            fr["arms"][arm]["floor"] = fl
    for arm in ("T1", "T2"):
        if any(arm not in res[f]["arms"] for f in FOLDS):
            dec[arm] = {"eligible": False, "why": "NOT RUN"}
            continue
        A2, R2 = res["F2"]["arms"][arm], res["F2"]["arms"]["T0"]
        A1, R1 = res["F1"]["arms"][arm], res["F1"]["arms"]["T0"]
        f2, f1 = A2["floor"], A1["floor"]
        chk = {
            "E1_F2_ll_tov_noninferior": A2["lines"]["ll_tov"] - R2["lines"]["ll_tov"] <= f2["ll_tov"],
            "E2_F2_ll_multi_noninferior": A2["lines"]["ll_multi"] - R2["lines"]["ll_multi"] <= f2["ll_multi"],
            "E3_F2_bucket_gaps": all(abs(A2["lines"][k]) - abs(R2["lines"][k]) <= f2[k]
                                     for k in ["gap_all", *[f"gap_{b}" for b, _, _ in BUCKETS]]),
            "E4_gates_slope": (A2["gates"].get("calibration_pass", False) or not R2["gates"].get("calibration_pass", True))
                              and (A2["gates"].get("responsiveness_pass", False) or not R2["gates"].get("responsiveness_pass", True))
                              and all(res[f]["arms"][arm]["lines"][f"slope_{s}"] >= res[f]["arms"]["T0"]["lines"][f"slope_{s}"] - 0.05
                                      for f in FOLDS for s in ("all", "d0-45")),
            "E5_F1_confirms": (R1["lines"]["ll_tov"] - A1["lines"]["ll_tov"] > f1["ll_tov"])
                              and (abs(R1["lines"]["gap_all"]) - abs(A1["lines"]["gap_all"]) > f1["gap_all"]),
        }
        dec[arm] = {"checks": chk, "eligible": all(chk.values()), "F2_ll_tov": A2["lines"]["ll_tov"]}
    elig = [a for a in ("T1", "T2") if dec[a]["eligible"]]
    winner = "T0"
    if elig:
        best = min(elig, key=lambda a: dec[a]["F2_ll_tov"])
        winner = best
        if best == "T2" and "T1" in elig and dec["T1"]["F2_ll_tov"] - dec["T2"]["F2_ll_tov"] <= res["F2"]["arms"]["T2"]["floor"]["ll_tov"]:
            winner = "T1"
    res["decision"] = {"by_arm": dec, "winner": winner}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grade_v1.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")

    # markdown
    L = ["| fold | arm | TOV LL (d vs T0) | multi LL (d vs T0) | gap d0-14 | d15-45 | d46+ | all | slope all / d0-45 | spread | pg MAE pp | calib worst pp |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for fold in FOLDS:
        fr = res[fold]
        r0 = fr["arms"]["T0"]["lines"]
        for arm, v in fr["arms"].items():
            x = v["lines"]
            dl = "" if arm == "T0" else f" ({x['ll_tov'] - r0['ll_tov']:+.6f})"
            dm = "" if arm == "T0" else f" ({x['ll_multi'] - r0['ll_multi']:+.6f})"
            L.append(f"| {fold} | {arm} | {x['ll_tov']:.6f}{dl} | {x['ll_multi']:.6f}{dm} | {x['gap_d0-14']:+.2f} | "
                     f"{x['gap_d15-45']:+.2f} | {x['gap_d46+']:+.2f} | {x['gap_all']:+.2f} | {x['slope_all']:.3f} / "
                     f"{x['slope_d0-45']:.3f} | {x['spread_ratio']:.3f} | {x['pg_level_mae_pp']:.2f} | "
                     f"{v['gates'].get('worst_gated_gap_pp', float('nan'))} |")
        for arm in ("T1", "T2"):
            if arm in fr["arms"] and "floor" in fr["arms"][arm]:
                fl = fr["arms"][arm]["floor"]
                L.append(f"| {fold} | floor {arm} | {fl['ll_tov']:.6f} | {fl['ll_multi']:.6f} | {fl['gap_d0-14']:.2f} | "
                         f"{fl['gap_d15-45']:.2f} | {fl['gap_d46+']:.2f} | {fl['gap_all']:.2f} | | | | |")
    L.append("")
    L.append(f"Decision: winner **{winner}**. " + "; ".join(
        f"{a}: " + ", ".join(f"{k.split('_')[0]}={'P' if ok else 'F'}" for k, ok in v.get("checks", {}).items())
        for a, v in dec.items()))
    (OUT / "grade_v1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
