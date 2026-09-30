"""The as-of entry point's stub device: ratings at D must not depend on anything dated D or later."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_own_ratings_asof_v1 as E  # noqa: E402
from cbb_sim.ratings import own_ratings as orat  # noqa: E402


def _tg(seed=0):
    rng = np.random.default_rng(seed)
    teams = [1, 2, 3, 4, 5, 6]
    rows, gid = [], 100
    for d in pd.date_range("2025-01-01", periods=6):
        perm = rng.permutation(teams)
        for h, a in zip(perm[0::2], perm[1::2]):
            gid += 1
            poss = 68 + rng.normal(0, 3)
            for team, opp, home in ((h, a, True), (a, h, False)):
                rows.append(dict(game_id=gid, season=2025, game_date=d, team_id=int(team), opp_team_id=int(opp),
                                 home_team_id=int(h), away_team_id=int(a), neutral_site=False, site_home=float(home),
                                 site_away=float(not home), off_eff=100 + rng.normal(0, 10), game_poss=poss))
    return pd.DataFrame(rows)


def test_stub_pairs_cover_all_teams():
    s = E.stub_games(np.array([1, 2, 3, 4, 5]), pd.Timestamp("2025-01-07"), 2025)
    assert set(s["team_id"]) == {1, 2, 3, 4, 5} and (s["game_id"] < 0).all()


def test_ratings_at_D_ignore_D_and_later():
    tg = _tg()
    D = pd.Timestamp("2025-01-05")
    hp = dict(l=5.0, w=0.8)
    full = orat.fit_season(tg, 2025, hp["l"], hp["l"], None, None, 0.0, 0.0)
    a = orat.run_to_frame(full)
    a = a[a["as_of_date"] == D].sort_values("team_id").reset_index(drop=True)
    cur = tg[tg["game_date"] < D]
    stubs = E.stub_games(np.array(sorted(tg["team_id"].unique())), D, 2025)
    panel = pd.concat([cur[[c for c in cur.columns if c in stubs.columns]], stubs[cur.columns.intersection(stubs.columns)]],
                      ignore_index=True)
    # the panel carries only rows dated before D plus stubs; the full-data fit at D must match it
    b = orat.run_to_frame(orat.fit_season(panel, 2025, hp["l"], hp["l"], None, None, 0.0, 0.0))
    b = b[b["as_of_date"] == D].sort_values("team_id").reset_index(drop=True)
    for c in ("off_c", "def_c", "tempo_rel", "n_games", "league_off_mean", "league_tempo_mean"):
        assert np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), atol=1e-9, rtol=0), c
