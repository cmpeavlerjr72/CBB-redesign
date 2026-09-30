"""Tests for the drawn block flag (`cbb_sim.engine.shot_block`, shot_block section 5)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbb_sim.engine import rng
from cbb_sim.engine import shot_block as SBK

ROOT = Path(__file__).resolve().parents[1]
LUT = ROOT / "data/processed/models/engine/shot_block_K2_Ocell_F2_2025.npz"


def test_default_off():
    assert SBK.load(None, None) is None
    assert SBK.load("reference", None) is None
    with pytest.raises(KeyError):
        SBK.load("nope", None)


def test_family_is_appended_and_independent():
    assert rng.FAMILIES[-1] == "shot_block"
    seeds, gids = np.array([0, 1, 2]), np.array([401700000, 401700001, 401700002])
    a = rng.StreamBook(seeds, gids)
    b = rng.StreamBook(seeds, gids, families=tuple(f for f in rng.FAMILIES if f != "shot_block"))
    a.draw("shot_block")
    for fam in ("clock", "event", "rebound"):
        assert np.array_equal(a.draw(fam), b.draw(fam))


@pytest.mark.skipif(not LUT.exists(), reason="shot_block engine LUT not built")
def test_engine_probability_equals_the_fitted_model_on_real_rows():
    """The engine path (LUT gather + live state) reproduces the offline K2_Ocell
    probability on 2025 missed-FGA rows whose shooter is on the engine roster."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import exp_season_drift_anchor_v1 as X
    import train_shot_block_v1 as SB
    from cbb_sim import season_anchor as SA
    from cbb_sim.engine.inputs import EngineInputs
    inp = EngineInputs.load(ROOT / "data/processed/models/engine", "F2_2025")
    sb = SBK.load("K2_Ocell", inp)
    full, _ = SB.build_design()
    full["shooter_blocked_c"] = SB.shooter_block_rates(full, 50.0)
    F = X.fold_data("shot_block", "F2")
    df, tr, te = F["df"], F["tr"], F["te"]
    off = np.zeros(len(df))
    for t in ("rim", "jump2", "three"):
        m = F["sub"] == t
        off[m] = SA.anchor_O(df.loc[m, "season"].to_numpy(), df.loc[m, "game_date"].to_numpy(),
                             F["num"][m], F["den"][m], [2022, 2023, 2024], "binary").offset()[:, 0]
    mdl = {"mu": sb.mu, "sd": sb.sd, "theta": sb.coef, "family": "binomial"}
    Xm = df[SB.KC].to_numpy(dtype="float32")
    p_off = X.predict_glm(mdl, Xm[te], off[te])
    t25 = full[full["season"] == 2025].reset_index(drop=True)
    g = inp.games
    gpos = pd.Series(np.arange(len(g)), index=g["game_id"].to_numpy())
    keep = t25["game_id"].isin(gpos.index).to_numpy() & (t25["shooter_id"].to_numpy() >= 0)
    rows = np.flatnonzero(keep)[:: 50]
    x = t25.iloc[rows]
    gi = gpos.loc[x["game_id"]].to_numpy()
    side = np.where(x["off_team_id"].to_numpy() == g["home_team_id"].to_numpy()[gi], 0, 1)
    hit = inp.roster_cbbd[gi, side] == x["shooter_id"].to_numpy()[:, None]
    ok = hit.any(1)
    gi, side, slot, x, rows = gi[ok], side[ok], hit.argmax(1)[ok], x[ok], rows[ok]
    tix = x["miss_type"].map(SBK.TYPE_INDEX).to_numpy()
    p_eng = sb.prob(inp.team_static, gi, side, slot, tix, x["period"].to_numpy(float),
                    x["seconds_remaining"].to_numpy(float), x["score_diff"].to_numpy(float),
                    x["in_bonus"].to_numpy(float))
    # engine team_static carries the same ratings/site columns as the design; the
    # shooter_known flag is 1 for a roster player (the design's join flag can be 0)
    same = x["shooter_known"].to_numpy() == 1
    assert len(p_eng) > 500
    d = np.abs(p_eng - p_off[rows])[same]
    assert np.median(d) < 1e-6, np.median(d)
