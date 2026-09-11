"""
usage_tree_adapter.py -- Decision-10 closed-loop gate arms for the usage U5
LightGBM tree. NOT ADOPTED, NOT SERVED.

`ENGINE_USAGE=reference` (the default, `adapters.UsageAdapter`) is completely
unaffected by this module: it is loaded only when `ENGINE_USAGE` names one of
the three arms below, all read by `scripts/run_usage_tree_closed_loop.py`
alone. Evidence and verdict: `docs/tests/usage_decision10_gate_2026-09-11.md`;
pre-registration: `docs/models/usage/experiments.md` section 11.

Arms (`ENGINE_USAGE`):
  tree_v3          round-3 (corrected `score_diff`) LightGBM tree, LIVE
                    engine state read off `GameState` every step.
  tree_v3_freeze   the SAME boosters, state features FROZEN at their pregame
                    value for the whole game (`score_diff=0`,
                    `sec_remaining=2400`, `period=1`, `chance_number=1`) --
                    the Decision-10 freeze ablation (L31/L33).
  tree_v3_nostate  a tree refit with NO state features at all
                    (`scripts/train_usage_v3_save_tree.py`'s `_nostate`
                    artifact), live otherwise -- the Decision-10
                    refit-without arm, which SIZES a loop a freeze can only
                    DETECT (L31).

WHY THE PER-ALTERNATIVE STATIC FEATURES ARE BUILT HERE, NOT IN
`build_engine_inputs.py`: the tree's non-state features (`prior_rate`,
`exposure_asof`, `minutes_asof`, `minutes_per_game_asof`, `games_asof`,
`position_code`, `is_transfer`) are all PREGAME quantities, exactly like
`EngineInputs.usage_rate` (which already supplies `rate`/`share` for every
arm here, unchanged by the round-3 fix -- round 3 changes `score_diff` only,
never the shrinkage prior/m `usage_rate` is built from). Building them from
`inp.roster_cbbd` + `inp.games` the same way `usage_rate` itself is built
(`asof_backward`, `scripts/build_engine_inputs.py` section 5) keeps this
module a read-only, additive side-table over an EXISTING engine-inputs
artifact -- nothing under `data/processed/models/engine/` is touched, and no
other lane's default run is affected.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cbb_sim.engine.inputs import EngineInputs
from cbb_sim.models import usage as U

ROOT = Path(__file__).resolve().parents[3]
V2_ASOF = ROOT / "data/processed/models/usage_v2/asof_v2_shotshooter.parquet"
V3_TREE_DIR = ROOT / "data/processed/models/usage_v3/lgbm_tree"

#: pregame value of each state feature, in the units `usage.build_usage_events`
#: writes them: 0 margin, 40:00 left in regulation, period 1, first chance.
PREGAME_STATE = {"score_diff": 0.0, "sec_remaining": 2400.0, "period": 1.0,
                 "chance_number": 1.0}


def _asof_join(inp: EngineInputs, asof: pd.DataFrame) -> pd.DataFrame:
    """(row, side, slot) -> every non-state tree feature, for every named
    roster slot of `inp` -- the same backward-as-of join
    `scripts/build_engine_inputs.py` uses for `usage_rate`, reproduced here so
    this module needs no import-time dependency on that script."""
    G, _, S = inp.roster_cbbd.shape
    flat = pd.DataFrame({
        "row": np.repeat(np.arange(G), 2 * S),
        "side": np.tile(np.repeat([0, 1], S), G),
        "slot": np.tile(np.arange(S), 2 * G),
        "pid": inp.roster_cbbd.reshape(-1),
        "game_date": np.repeat(pd.to_datetime(inp.games["game_date"]).to_numpy(), 2 * S),
    })
    real = flat[(flat["pid"] > 0) & inp.roster_valid.reshape(-1)].copy()

    cols = ["exposure_asof", "minutes_asof", "minutes_per_game_asof", "games_asof",
            "position_group", "prev_team_id", "has_prior_season",
            *[f"prev_rate_{c}" for c in U.EVENT_CLASSES]]
    s = asof.rename(columns={"player_id": "pid"})[["pid", "game_date", *cols]].copy()
    s["pid"] = s["pid"].astype("int64")
    s["game_date"] = pd.to_datetime(s["game_date"])
    s = s.sort_values("game_date", kind="stable").drop_duplicates(
        subset=["pid", "game_date"], keep="last")

    real = real.sort_values("game_date", kind="stable").reset_index(drop=True)
    s = s.reset_index(drop=True)
    got = pd.merge_asof(real, s, on="game_date", by="pid",
                        direction="backward", allow_exact_matches=True)

    home_tid = inp.games["home_team_id"].to_numpy()
    away_tid = inp.games["away_team_id"].to_numpy()
    team_id = np.where(got["side"].to_numpy() == 0,
                       home_tid[got["row"].to_numpy()], away_tid[got["row"].to_numpy()])
    got["team_id_this_game"] = team_id
    got["is_transfer"] = ((got["has_prior_season"].fillna(0) > 0)
                          & got["prev_team_id"].notna()
                          & (got["prev_team_id"] != got["team_id_this_game"])).astype("float64")
    pos_idx = {p: i for i, p in enumerate(U.POSITION_LEVELS)}
    got["position_code"] = got["position_group"].map(pos_idx).fillna(
        pos_idx["UNK"]).astype("float64")
    return got


@dataclass
class UsageTreeAdapter:
    classes: tuple[str, ...]
    class_index: dict[str, int]
    boosters: dict            # event_class -> {"model", "features", ...}
    exposure_asof: np.ndarray        # (G, 2, S)
    minutes_asof: np.ndarray         # (G, 2, S)
    minutes_per_game_asof: np.ndarray  # (G, 2, S)
    games_asof: np.ndarray           # (G, 2, S)
    position_code: np.ndarray        # (G, 2, S)
    is_transfer: np.ndarray          # (G, 2, S)
    prior_rate: np.ndarray           # (G, 2, S, n_classes)
    freeze: bool
    source: dict
    provisional: bool = True

    @classmethod
    def load(cls, inp: EngineInputs, mode: str) -> UsageTreeAdapter:
        if mode not in ("tree_v3", "tree_v3_freeze", "tree_v3_nostate"):
            raise ValueError(f"unknown ENGINE_USAGE mode {mode!r}")
        suffix = "nostate" if mode == "tree_v3_nostate" else "corrected"
        boosters = {}
        for ec in U.EVENT_CLASSES:
            p = V3_TREE_DIR / f"{ec}_{suffix}.joblib"
            boosters[ec] = joblib.load(p)

        asof = pd.read_parquet(V2_ASOF)
        got = _asof_join(inp, asof)
        G, _, S = inp.roster_cbbd.shape
        n_cls = len(U.EVENT_CLASSES)

        def scatter(col: str, dtype="float32") -> np.ndarray:
            out = np.zeros((G, 2, S), dtype=dtype)
            v = got[col].to_numpy(dtype="float64")
            v = np.where(np.isfinite(v), v, 0.0)
            out[got["row"].to_numpy(), got["side"].to_numpy(), got["slot"].to_numpy()] = v
            return out

        prior_rate = np.zeros((G, 2, S, n_cls), dtype="float32")
        for k, ec in enumerate(U.EVENT_CLASSES):
            prior_rate[:, :, :, k] = scatter(f"prev_rate_{ec}")

        return cls(
            classes=U.EVENT_CLASSES,
            class_index={c: i for i, c in enumerate(U.EVENT_CLASSES)},
            boosters=boosters,
            exposure_asof=scatter("exposure_asof"),
            minutes_asof=scatter("minutes_asof"),
            minutes_per_game_asof=scatter("minutes_per_game_asof"),
            games_asof=scatter("games_asof"),
            position_code=scatter("position_code"),
            is_transfer=scatter("is_transfer"),
            prior_rate=prior_rate,
            freeze=(mode == "tree_v3_freeze"),
            source={"mode": mode, "boosters": {k: str(V3_TREE_DIR / f"{k}_{suffix}.joblib")
                                               for k in U.EVENT_CLASSES},
                    "why": "Decision-10 closed-loop gate arm, NOT ADOPTED, NOT SERVED"},
        )

    def probs(self, rate_five: np.ndarray, *, inp: EngineInputs, gidx: np.ndarray,
             side: np.ndarray, five: np.ndarray, usage_class: np.ndarray,
             score_diff: np.ndarray, sec_remaining: np.ndarray,
             period: np.ndarray, chance_number: np.ndarray) -> np.ndarray:
        """(k, 5) choice probabilities for every row of this batched call.

        `rate_five` is `inp.usage_rate` already gathered for these rows'
        on-floor five and usage class -- exactly U1's `rate`/`share` inputs,
        UNCHANGED by the round-3 fix (only `score_diff` moved). Rows are
        grouped by `usage_class` and scored one booster call per class, the
        same "one predict per disjoint row set" pattern `fg_make`/`rebound`
        use elsewhere in this loop."""
        k = rate_five.shape[0]
        out = np.empty((k, 5), dtype="float64")
        rate = rate_five.astype("float64")
        share = U.normalise(rate)
        rank = np.argsort(np.argsort(-share, axis=1, kind="stable"), axis=1).astype("float64")

        if self.freeze:
            score_diff = np.full(k, PREGAME_STATE["score_diff"])
            sec_remaining = np.full(k, PREGAME_STATE["sec_remaining"])
            period = np.full(k, PREGAME_STATE["period"])
            chance_number = np.full(k, PREGAME_STATE["chance_number"])

        g5 = gidx[:, None]
        s5 = side[:, None]
        exposure = self.exposure_asof[g5, s5, five]
        minutes = self.minutes_asof[g5, s5, five]
        minutes_pg = self.minutes_per_game_asof[g5, s5, five]
        games = self.games_asof[g5, s5, five]
        pos = self.position_code[g5, s5, five]
        transfer = self.is_transfer[g5, s5, five]

        for ci, ec in enumerate(self.classes):
            m = usage_class == ci
            if not m.any():
                continue
            boost = self.boosters[ec]
            feats: dict[str, np.ndarray] = {
                "share": share[m], "rate": rate[m],
                "prior_rate": self.prior_rate[g5[m], s5[m], five[m], ci],
                "exposure_asof": exposure[m], "minutes_asof": minutes[m],
                "minutes_per_game_asof": minutes_pg[m], "games_asof": games[m],
                "position_code": pos[m], "is_transfer": transfer[m], "usage_rank": rank[m],
                "score_diff": np.repeat(score_diff[m][:, None], 5, axis=1),
                "sec_remaining": np.repeat(sec_remaining[m][:, None], 5, axis=1),
                "period": np.repeat(period[m][:, None], 5, axis=1),
                "chance_number": np.repeat(chance_number[m][:, None], 5, axis=1),
            }
            names = boost["features"]
            nk = m.sum()
            X = np.empty((nk * 5, len(names)), dtype="float32")
            for j, name in enumerate(names):
                X[:, j] = feats[name].reshape(-1)
            init = np.log(np.maximum(share[m], 1e-9)).reshape(-1)
            p = boost["model"].predict_proba(X, init=init)
            out[m] = p
        return out
