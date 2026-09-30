#!/usr/bin/env python
"""
exp_season_drift_anchor_cell_v1.py -- EXPLORATORY, NOT PRE-REGISTERED, NOT
SELECTABLE. Added after round 1's shot_block cells showed the pooled block
level is only -0.17 pp off while rim is -0.83 pp: the drift is per shot type,
which a scalar anchor cannot move. This runs the round-1 anchors `O` and `P`
with the anchor built PER SHOT TYPE (the model's own level-gate cells) instead
of one scalar, for shot_block only, and scores them with the round-1 grader's
own functions. It informs the next pre-registration; it decides nothing.

    .venv/Scripts/python.exe scripts/exp_season_drift_anchor_cell_v1.py
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

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import exp_season_drift_anchor_v1 as X  # noqa: E402
import grade_season_drift_anchor_v1 as G  # noqa: E402

OUT = _ROOT / "results/season_drift/round1_exploratory"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    A = X.load_shot_block()
    res = {}
    for fold in ("F2", "F1"):
        F = X.fold_data("shot_block", fold)
        df, tr, te = F["df"], F["tr"], F["te"]
        spec = X.FOLDS[fold]
        sub = F["sub"]
        n = len(df)
        lev = {a: np.zeros((n, 1)) for a in ("O", "P")}
        lbar = np.zeros((n, 1))
        metas = {}
        for s in np.unique(sub):
            m = sub == s
            anc, meta = X.build_anchor(df.loc[m, "season"].to_numpy(), df.loc[m, "game_date"].to_numpy(),
                                       F["num"][m], F["den"][m], spec["train"], "binary")
            for a in ("O", "P"):
                lev[a][m] = anc["levels"][a]
            lbar[m] = anc["Lbar"][None, :]
            metas[s] = {"n0": meta["n0_fitted"], "prior_for_test": meta["prior_for_test"],
                        "Lbar": meta["Lbar"]}
        fr = pd.read_parquet(X.FRAMES / f"shot_block_{fold}.parquet")
        pR = np.load(X.PREDS / f"shot_block_{fold}_R_s0.npy")
        cR, exR = G.score_cell("shot_block", fold, fr, pR)
        X_ = df[A["feats"]].to_numpy(dtype="float32")
        for a in ("O", "P"):
            off = (X._link(lev[a], "binary") - X._link(lbar, "binary"))[:, 0]
            mdl = X.fit_glm(X_[tr], F["num"][tr, 0], off[tr], None, "binomial")
            p = X.predict_glm(mdl, X_[te], off[te])[:, None]
            np.save(OUT / f"shot_block_{fold}_{a}cell_s0.npy", p)
            c, ex = G.score_cell("shot_block", fold, fr, p)
            c["boot_se"] = G.paired_boot_se(ex["game_loss"], exR["game_loss"])
            c0 = c["classes"][0]
            res[f"{fold}|{a}cell"] = {
                "primary": c["primary"], "gain_vs_R": round(cR["primary"] - c["primary"], 7),
                "boot_se": c["boot_se"], "level_pp": c0["level"], "novdec_pp": c0["novdec"],
                "by_type_pp": {k: v["level"] for k, v in c0["by_sub"].items()},
                "by_type_rel": {k: v["rel"] for k, v in c0["by_sub"].items()},
                "slope": c0["team"].get("slope_ratio"), "slope_novdec": c0["team_novdec"].get("slope_ratio"),
                "sd_nc": c0["team"].get("sd_ratio_nc"), "calib": c["calib"], "anchor": metas,
                "converged": mdl["converged"]}
            print(fold, a, json.dumps(res[f"{fold}|{a}cell"], default=str), flush=True)
        c0 = cR["classes"][0]
        res[f"{fold}|R"] = {"primary": cR["primary"], "level_pp": c0["level"],
                            "by_type_pp": {k: v["level"] for k, v in c0["by_sub"].items()},
                            "by_type_rel": {k: v["rel"] for k, v in c0["by_sub"].items()}}
    (OUT / "shot_block_cell_anchor_v1.json").write_text(json.dumps(res, indent=1, default=str),
                                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
