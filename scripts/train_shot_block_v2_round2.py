#!/usr/bin/env python
"""
train_shot_block_v2_round2.py -- shot_block round 2 (POST-HOC on folds 1-2):
the per-shot-type level anchor. Pre-registration: `docs/models/shot_block/
experiments.md` section 3 (commit 9411040, before this run).

Arms: K0 (round-1 reference), K2 (round-1 leader), K2_Ocell, K2_Pcell. Gates
scored by round 1's own grader (`train_shot_block_v1.grade`); Nov-Dec, team
slope and SD ratio by the season-drift grader (`grade_season_drift_anchor_v1`).

    .venv/Scripts/python.exe scripts/train_shot_block_v2_round2.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import exp_season_drift_anchor_v1 as X  # noqa: E402
import grade_season_drift_anchor_v1 as G  # noqa: E402
import train_shot_block_v1 as SB  # noqa: E402
from cbb_sim import season_anchor as SA  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402

OUT = ROOT / "results/shot_block_round2"
ARMS = ["K0", "K2", "K2_Ocell", "K2_Pcell"]
SIMPLE = {a: i for i, a in enumerate(ARMS)}
PUB_FLOOR = 4.1e-05


def cell_levels(F: dict, spec: dict, arm: str):
    df, sub = F["df"], F["sub"]
    n = len(df)
    lev, lbar, meta = np.zeros((n, 1)), np.zeros((n, 1)), {}
    for s in np.unique(sub):
        m = sub == s
        if arm == "K2_Ocell":
            a = SA.anchor_O(df.loc[m, "season"].to_numpy(), df.loc[m, "game_date"].to_numpy(),
                            F["num"][m], F["den"][m], spec["train"], "binary")
            lev[m], lb = a.levels, a.Lbar
            meta[s] = {"prior": a.meta["prior_by_season"]}
        else:
            anc, mt = X.build_anchor(df.loc[m, "season"].to_numpy(), df.loc[m, "game_date"].to_numpy(),
                                     F["num"][m], F["den"][m], spec["train"], "binary")
            lev[m], lb = anc["levels"]["P"], anc["Lbar"]
            meta[s] = {"n0": mt["n0_fitted"], "prior": mt["prior_for_test"]}
        lbar[m] = lb[None, :]
        meta[s]["Lbar"] = float(lb[0])
    return (SA.link(lev, "binary") - SA.link(lbar, "binary"))[:, 0], meta


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    assert_not_sealed([2022, 2023, 2024, 2025], context="shot_block round 2")
    full, _ = SB.build_design()
    full["shooter_blocked_c"] = SB.shooter_block_rates(full, 50.0)
    res = {}
    for fold in ("F2", "F1"):
        spec = X.FOLDS[fold]
        F = X.fold_data("shot_block", fold)
        df, tr, te = F["df"], F["tr"], F["te"]
        te_full = full[full["season"].isin(spec["test"])].reset_index(drop=True)
        chk = ["game_id", "y", "miss_type"]
        assert (te_full[chk].to_numpy() == df.loc[te, chk].reset_index(drop=True).to_numpy()).all(), \
            "row alignment between round-1 fold data and the fresh design failed"
        fr = pd.read_parquet(X.FRAMES / f"shot_block_{fold}.parquet")
        Xm = df[SB.KC].to_numpy(dtype="float32")
        preds, metas = {}, {}
        tr_df = df.loc[tr]
        k0 = tr_df[tr_df["season"] == tr_df["season"].max()].groupby("miss_type")["y"].mean()
        preds["K0"] = df.loc[te, "miss_type"].map(k0).to_numpy(dtype="float64")
        for arm in ARMS[1:]:
            off = None
            if arm != "K2":
                off, metas[arm] = cell_levels(F, spec, arm)
            m = X.fit_glm(Xm[tr], F["num"][tr, 0], None if off is None else off[tr], None, "binomial")
            preds[arm] = X.predict_glm(m, Xm[te], None if off is None else off[te])
            np.save(OUT / f"{fold}_{arm}.npy", preds[arm])
        cards, gl = {}, {}
        for arm, p in preds.items():
            g = SB.grade(te_full, np.clip(p, 1e-9, 1 - 1e-9))
            c, ex = G.score_cell("shot_block", fold, fr, p[:, None])
            gl[arm] = ex["game_loss"]
            c0 = c["classes"][0]
            cards[arm] = {
                "log_loss": g["log_loss"], "level_pp": g["level_pp"],
                "by_type_pp": {k: v.get("level_pp") for k, v in g["by_shot_type"].items()},
                "calib_gap_pp": g["calib_worst_decile_gap_pp"], "calib_pass": g["calib_pass"],
                "level_pass": g["level_pass"], "resp_pass": g["resp_pass"],
                "def_slope": g["slope_def_prior"].get("slope_ratio"),
                "def_mono": g["slope_def_prior"].get("monotone_steps"),
                "novdec_pp": c0["novdec"], "team_slope": c0["team"].get("slope_ratio"),
                "team_slope_novdec": c0["team_novdec"].get("slope_ratio"),
                "sd_nc": c0["team"].get("sd_ratio_nc"), "anchor": metas.get(arm)}
        for arm in ARMS:
            se = 0.0 if arm in ("K0", "K2") else G.paired_boot_se(gl[arm], gl["K2"])
            cards[arm]["paired_se_vs_K2"] = se
            cards[arm]["floor"] = max(PUB_FLOOR, 2 * se)
            cards[arm]["gain_vs_K0"] = round(cards["K0"]["log_loss"] - cards[arm]["log_loss"], 7)
            cards[arm]["gain_vs_K2"] = round(cards["K2"]["log_loss"] - cards[arm]["log_loss"], 7)
        res[fold] = cards
        for arm in ARMS:
            print(fold, arm, json.dumps({k: v for k, v in cards[arm].items() if k != "anchor"}), flush=True)
    # decision (section 3.3)
    elig = []
    for arm in ARMS[1:]:
        c2, c1 = res["F2"][arm], res["F1"][arm]
        ok = (c2["gain_vs_K0"] > c2["floor"] and c2["calib_pass"] and c2["resp_pass"] and c2["level_pass"]
              and np.sign(c1["gain_vs_K0"]) == np.sign(c2["gain_vs_K0"]) and c1["level_pass"])
        res["F2"][arm]["eligible"] = bool(ok)
        if ok:
            elig.append(arm)
    sel = None
    if elig:
        best = min(elig, key=lambda a: res["F2"][a]["log_loss"])
        tie = [a for a in elig if res["F2"][a]["log_loss"] - res["F2"][best]["log_loss"] <= res["F2"][best]["floor"]]
        sel = min(tie, key=lambda a: SIMPLE[a])
    res["eligible"], res["selected"] = elig, sel
    res["created_at"] = pd.Timestamp.now("UTC").isoformat()
    (OUT / "round2_v1.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("eligible", elig, "selected", sel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
