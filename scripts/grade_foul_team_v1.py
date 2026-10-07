#!/usr/bin/env python
"""
grade_foul_team_v1.py -- ONE grader for foul-accrual round 2 (possession_outcome experiments.md section 34,
commit 0f83dee; built from grade_foul_cal_v1 with the s34.3 gates). Reads round11team/preds_{F1,F2}_seed{0,7}.parquet,
scores every `p__<arm>` column the same way, applies the pre-registered rule (34.3) mechanically.

    .venv/Scripts/python.exe scripts/grade_foul_team_v1.py
Writes results/foul_r2/grade_v1.json and prints the tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

D = ROOT / "data/processed/models/possession_outcome/round11team"
OUT = ROOT / "results/foul_r2"
ORDER = ["A2t", "A2tn", "A2tnc"]               # complexity order, simplest first (34.2)
CEILING = "A2tG"
CONTROL = "A2"
BKT_EDGES = np.array([7, 14, 30, 45])
BKT_LAB = ["d0-7", "d8-14", "d15-30", "d31-45", "d46+"]
N_BOOT, BOOT_SEED = 200, 20261007
UNDER = 2000
CARRIED = 0.000804


def rowll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_se(delta: np.ndarray, games: np.ndarray) -> float:
    """game-block bootstrap SE of the mean per-row delta."""
    codes, inv = np.unique(games, return_inverse=True)
    s = np.bincount(inv, weights=delta)
    n = np.bincount(inv).astype(float)
    rng = np.random.default_rng(BOOT_SEED)
    k = len(codes)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
        out[b] = (w * s).sum() / (w * n).sum()
    return float(out.std(ddof=1))


def team_prior() -> pd.DataFrame:
    """prior-season (s-1) defence non-trip rate per possession, centred on that season's league mean."""
    import train_foul_joint_v1 as FJ
    d = FJ.possession_design()
    d = d[d["in_fit_window"] == 1]
    rows = []
    for side, col in (("def", "defense_team_id"), ("off", "offense_team_id")):
        t = d.groupby(["season", col])["y_nt"].agg(["mean", "size"]).reset_index()
        t = t[t["size"] >= 300]
        t["c"] = t["mean"] - t.groupby("season")["mean"].transform("mean")
        t["season"] = t["season"] + 1                      # season s holds s-1 rates
        rows.append(t.rename(columns={col: "team_id", "c": f"prior_{side}"})[["season", "team_id", f"prior_{side}"]])
    return rows[0].merge(rows[1], on=["season", "team_id"], how="outer")


def quint_slope(df: pd.DataFrame, team_col: str, prior: pd.DataFrame, pcol: str, arms) -> dict:
    g = df.groupby(["season", team_col]).agg(y=("y_nt", "mean"), n=("y_nt", "size"),
                                             **{a: (f"p__{a}", "mean") for a in arms}).reset_index()
    g = g[g["n"] >= 300].merge(prior.rename(columns={"team_id": team_col}), on=["season", team_col], how="inner")
    g = g.dropna(subset=[pcol])
    g["q"] = pd.qcut(g[pcol], 5, labels=False)
    q = g.groupby("q")[["y"] + list(arms)].mean()
    out = {"n_teams": int(len(g)), "actual_q": q["y"].round(5).tolist()}
    for a in arms:
        out[a] = {"q": q[a].round(5).tolist(), "slope": float(np.polyfit(q["y"], q[a], 1)[0])}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = team_prior()
    res: dict = {"pre_registration": "experiments.md s34 (0f83dee)", "folds": {}}
    for fold in ("F2", "F1"):
        P = {s: pd.read_parquet(D / f"preds_{fold}_seed{s}.parquet") for s in (0, 7)}
        arms = [c[3:] for c in P[0].columns if c.startswith("p__")]
        fr = {}
        for s in (0, 7):
            df = P[s]
            df["bkt"] = np.searchsorted(BKT_EDGES, df["dss"].to_numpy(), side="left")
            df["half"] = np.where(df["period"] <= 1, 1, 2)
            for a in arms:
                df[f"ll__{a}"] = rowll(df["y_nt"].to_numpy(), df[f"p__{a}"].to_numpy())
        df0 = P[0]
        w = df0[df0["in_fit_window"] == 1]
        w7 = P[7][P[7]["in_fit_window"] == 1]
        ctrl_floor = abs(w[f"ll__{CONTROL}"].mean() - w7[f"ll__{CONTROL}"].mean())
        fr["n_rows_primary"] = int(len(w))
        fr["control_reseed_floor"] = float(ctrl_floor)
        fr["arms"] = {}
        late = w[w["bkt"] == 4]
        late7 = P[7][(P[7]["in_fit_window"] == 1) & (P[7]["bkt"] == 4)]
        late_reseed = abs(late[f"ll__{CONTROL}"].mean() - late7[f"ll__{CONTROL}"].mean())
        for a in arms:
            r = {"ll": float(w[f"ll__{a}"].mean()), "ll_seed7": float(w7[f"ll__{a}"].mean()),
                 "ll_all_rows": float(df0[f"ll__{a}"].mean()),
                 "reseed_spread": float(abs(w[f"ll__{a}"].mean() - w7[f"ll__{a}"].mean()))}
            if a != CONTROL:
                dl = (w[f"ll__{CONTROL}"] - w[f"ll__{a}"]).to_numpy()        # positive = arm better
                se = boot_se(dl, w["game_id"].to_numpy())
                r["delta_vs_control"] = float(dl.mean())
                r["boot_se"] = se
                r["floor"] = float(max(ctrl_floor, 2 * se))
                r["floors"] = r["delta_vs_control"] / r["floor"]
                r["in_carried_floor_units"] = r["delta_vs_control"] / CARRIED
                dll = (late[f"ll__{CONTROL}"] - late[f"ll__{a}"]).to_numpy()
                sel = boot_se(dll, late["game_id"].to_numpy())
                r["d46_delta"] = float(dll.mean())
                r["d46_floor"] = float(max(late_reseed, 2 * sel))
            # calibration by half x bucket (mean p / mean y - 1) and ll by bucket
            cal = {}
            for (h, b), g in w.groupby(["half", "bkt"]):
                cal[f"H{h}_{BKT_LAB[b]}"] = {"n": int(len(g)), "rel": float(g[f"p__{a}"].mean() / g["y_nt"].mean() - 1),
                                            "underpowered": bool(len(g) < UNDER)}
            r["cal_half_bkt"] = cal
            for nm, sel_ in (("H1_d0-14", (w["half"] == 1) & (w["bkt"] <= 1)),
                             ("H1_d46+", (w["half"] == 1) & (w["bkt"] == 4)),
                             ("H2_d0-14", (w["half"] == 2) & (w["bkt"] <= 1)),
                             ("H2_d46+", (w["half"] == 2) & (w["bkt"] == 4))):
                g = w[sel_]
                r[f"rel_{nm}"] = float(g[f"p__{a}"].mean() / g["y_nt"].mean() - 1)
            site = np.where(w["site_home"] > 0, "home", np.where(w["site_away"] > 0, "away", "neutral"))
            r["rel_site"] = {s_: float(w.loc[site == s_, f"p__{a}"].mean() / w.loc[site == s_, "y_nt"].mean() - 1)
                             for s_ in ("home", "away", "neutral")}
            cnt = np.where(w["def_f_t"] >= 6, "6+", "0-5")
            r["rel_count"] = {c: float(w.loc[cnt == c, f"p__{a}"].mean() / w.loc[cnt == c, "y_nt"].mean() - 1)
                              for c in ("0-5", "6+")}
            fr["arms"][a] = r
        fr["resp_def"] = quint_slope(w, "defense_team_id", prior, "prior_def", arms)
        fr["resp_off"] = quint_slope(w, "offense_team_id", prior, "prior_off", arms)
        fr["resp_def_control_seed7"] = quint_slope(w7, "defense_team_id", prior, "prior_def",
                                                   [CONTROL])[CONTROL]["slope"]
        fr["resp_off_control_seed7"] = quint_slope(w7, "offense_team_id", prior, "prior_off",
                                                   [CONTROL])[CONTROL]["slope"]
        res["folds"][fold] = fr
    # ---- decision (34.3), mechanical ----
    f2, f1 = res["folds"]["F2"], res["folds"]["F1"]
    gates = {}
    for a in ORDER + [CEILING]:
        gates[a] = {}
        for fold, fr in (("F2", f2), ("F1", f1)):
            c, x = fr["arms"][CONTROL], fr["arms"][a]
            g = {}
            for side in ("def", "off"):
                spread = abs(fr[f"resp_{side}"][CONTROL]["slope"] - fr[f"resp_{side}_control_seed7"])
                s_ = fr[f"resp_{side}"][a]["slope"]
                g[f"O1_resp_{side}"] = bool(s_ > 0 and s_ > fr[f"resp_{side}"][CONTROL]["slope"] + spread)
            g["O2_d46_ll"] = bool(x["d46_delta"] >= -x["d46_floor"])
            g["O2_d46_H1rel"] = bool(abs(x["rel_H1_d46+"]) <= abs(c["rel_H1_d46+"]) + 0.01)
            g["O3_d014_H1rel"] = bool(abs(x["rel_H1_d0-14"]) <= abs(c["rel_H1_d0-14"]) + 0.01)
            g["O3_H1_spread"] = bool(abs(x["rel_H1_d46+"] - x["rel_H1_d0-14"]) < abs(c["rel_H1_d46+"] - c["rel_H1_d0-14"]))
            gates[a][fold] = g
        gates[a]["beats_F2"] = bool(f2["arms"][a]["delta_vs_control"] > f2["arms"][a]["floor"])
        gates[a]["F1_confirms"] = bool(f1["arms"][a]["delta_vs_control"] > 0)
        gates[a]["eligible"] = bool(gates[a]["beats_F2"] and gates[a]["F1_confirms"]
                                    and all(all(gates[a][f].values()) for f in ("F2", "F1")))
    elig = [a for a in ORDER if gates[a]["eligible"]]
    dec = {"gates": gates, "eligible": elig}
    if elig:
        best = max(elig, key=lambda a: f2["arms"][a]["delta_vs_control"])
        win = next(a for a in ORDER if a in elig and
                   f2["arms"][best]["delta_vs_control"] - f2["arms"][a]["delta_vs_control"] <= f2["arms"][a]["floor"])
        dec.update({"best": best, "winner": win})
    else:
        dec["winner"] = None
    bo = max(ORDER, key=lambda a: f2["arms"][a]["delta_vs_control"])
    dec["best_offset_F2"] = bo
    dec["ceiling_beats_best_offset_by_floor"] = {
        f: bool(fr["arms"][CEILING]["delta_vs_control"] - fr["arms"][bo]["delta_vs_control"] > fr["arms"][CEILING]["floor"])
        for f, fr in (("F2", f2), ("F1", f1))}
    res["decision"] = dec
    (OUT / "grade_v1.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for fold in ("F2", "F1"):
        fr = res["folds"][fold]
        print(f"\n== {fold}  rows {fr['n_rows_primary']:,}  control reseed floor {fr['control_reseed_floor']:.2e}")
        for a, r in fr["arms"].items():
            extra = ""
            if a != CONTROL:
                extra = (f" d {r['delta_vs_control']:+.2e} floor {r['floor']:.2e} ({r['floors']:+.2f} fl, "
                         f"{r['in_carried_floor_units']:+.3f} carried) d46 {r['d46_delta']:+.2e}/{r['d46_floor']:.1e}")
            print(f"  {a:7s} ll {r['ll']:.6f} s7 {r['ll_seed7']:.6f}{extra}")
            print(f"          H1 rel d0-14 {r['rel_H1_d0-14']:+.3f} d46+ {r['rel_H1_d46+']:+.3f} | H2 d0-14 "
                  f"{r['rel_H2_d0-14']:+.3f} d46+ {r['rel_H2_d46+']:+.3f} | resp def {fr['resp_def'][a]['slope']:.3f}"
                  f" off {fr['resp_off'][a]['slope']:.3f}")
    print("\nDECISION", json.dumps(dec, default=str))


if __name__ == "__main__":
    main()
