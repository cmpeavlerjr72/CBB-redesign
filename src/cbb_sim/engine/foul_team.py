"""Foul-accrual round 2: team foul priors / league level / days-bucket logit offsets on the accrual table
(possession_outcome experiments.md section 34, 2026-10-07).

DEFAULT OFF. `ENGINE_FOUL_TEAM` unset / "off" -> `load()` returns None and `loop.py` calls `FoulJoint.accrual`
exactly as before (no extra argument, no extra computation, no RNG change).

`ENGINE_FOUL_TEAM=<stem>` (e.g. `coef_A2tnc_F2`; `_F1` stems for fold-1 reads with the overlay that loads
`lut_acc_A2_F1`) reads `round11team/<stem>.json` (fold TRAIN fit, seed 0) and `round11team/slate_feats_<season>.parquet`
(as-of team priors D / O / u, league level Lv, days bucket per game, built by `scripts/train_foul_team_v1.py` through
`build_foul_team_feats_v1.team_priors_asof`). Per game and defence side it precomputes
    delta = beta . [1, D_def, O_off, D_def u_def, O_off u_off, Lv, 1{bkt=0..3}]   (the arm's own columns)
and the defence accrual probability becomes expit(logit(lut[ix]) + delta). The possession's single `foul_accrual`
uniform is unchanged. Nothing here is fitted to sim output.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

LUT_DIR = Path("data/processed/models/possession_outcome/round11team")
ENV = "ENGINE_FOUL_TEAM"


class FoulTeam:
    def __init__(self, stem: str, inp, fj):
        cj = json.loads((LUT_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        seasons = pd.unique(inp.games["season"])
        if len(seasons) != 1:
            raise ValueError(f"{ENV} expects a one-season slate, got {seasons}")
        sl = pd.read_parquet(LUT_DIR / f"slate_feats_{int(seasons[0])}.parquet").set_index("game_id")
        gids = inp.games["game_id"].to_numpy()
        miss = np.setdiff1d(gids, sl.index.to_numpy())
        if len(miss):
            raise ValueError(f"{ENV}: {len(miss)} slate games have no features")
        sl = sl.loc[gids]
        self.stem, self.fj = stem, fj
        beta = np.asarray(cj["beta"], dtype=np.float64)
        n = len(gids)
        self.delta = np.zeros((n, 2), dtype=np.float64)
        for ds, (d_sfx, o_sfx) in enumerate((("h", "a"), ("a", "h"))):        # defence side 0 = home
            D = sl[f"D_{d_sfx}"].to_numpy(np.float64)
            O = sl[f"O_{o_sfx}"].to_numpy(np.float64)
            cols = {"one": np.ones(n), "D": D, "O": O, "Du": D * sl[f"u_{d_sfx}"].to_numpy(np.float64),
                    "Ou": O * sl[f"u_{o_sfx}"].to_numpy(np.float64), "Lv": sl["Lv"].to_numpy(np.float64)}
            bk = sl["dss_bkt"].to_numpy()
            for b in range(4):
                cols[f"b{b}"] = (bk == b).astype(np.float64)
            X = np.column_stack([cols[c] for c in cj["cols"]])
            self.delta[:, ds] = X @ beta

    def p_def(self, ix: tuple, gidx: np.ndarray, def_side: np.ndarray) -> np.ndarray:
        p = self.fj.acc["lut"][ix]
        z = np.log(p / (1.0 - p)) + self.delta[gidx, np.asarray(def_side, dtype=np.int64)]
        return 1.0 / (1.0 + np.exp(-z))


def load(inp, fj) -> FoulTeam | None:
    stem = os.environ.get(ENV, "off") or "off"
    if stem == "off":
        return None
    if fj is None:
        raise ValueError(f"{ENV} needs the foul joint (ENGINE_FOUL_JOINT must not be 'reference')")
    if (os.environ.get("ENGINE_FOUL_CAL", "off") or "off") != "off":
        raise ValueError(f"{ENV} cannot be combined with ENGINE_FOUL_CAL")
    return FoulTeam(stem, inp, fj)
