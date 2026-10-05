"""TOV level monitor: formula, bucketing, report-only table on synthetic games."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]
from cbb_sim.live import tov_monitor as TM  # noqa: E402


def _games(n=40, sim_rate=0.17, act_rate=0.16, seed=0):
    rng = np.random.default_rng(seed)
    poss = rng.uniform(120, 150, n)
    return pd.DataFrame({"game_id": np.arange(n), "days": rng.integers(0, 90, n), "sim_poss": poss, "sim_tov": poss * sim_rate,
                         "act_poss": poss, "act_tov": poss * act_rate, "sim_engine_poss": poss / 2})


def test_buckets():
    b = TM.bucket_of(pd.Series([0, 6, 7, 13, 14, 27, 28, 55, 56, 200]))
    assert list(b) == ["d00-06", "d00-06", "d07-13", "d07-13", "d14-27", "d14-27", "d28-55", "d28-55", "d56+", "d56+"]


def test_table_diff_and_underpowered_label():
    d = _games()
    d["bucket"] = TM.bucket_of(d["days"])
    t = TM.tov_level_table(d, n_boot=50)
    s = t[t["bucket"] == "season-to-date"].iloc[0]
    assert abs(s["diff"] - 0.01) < 1e-9 and s["n_games"] == 40 and s["note"] == "UNDERPOWERED"
    assert len(t) == len(TM.BUCKETS) + 1


def test_actual_formula_and_two_row_filter():
    tb = pd.DataFrame({"game_id": [1, 1, 2], "turnovers": [10, 12, 5], "field_goals_attempted": [60, 58, 50],
                       "offensive_rebounds": [10, 8, 5], "free_throws_attempted": [20, 20, 10]})
    a = TM.actual_game_tov(tb, [1, 2])
    assert list(a["game_id"]) == [1]                       # game 2 has one team row: dropped, not filled
    assert a["act_tov"].iloc[0] == 22
    assert abs(a["act_poss"].iloc[0] - ((60 - 10 + 10 + 9.5) + (58 - 8 + 12 + 9.5))) < 1e-9


def test_report_block_skipped_and_table():
    assert "skipped" in TM.report_block({"skipped": "x"}, "2024-11-04")[2]
    d = _games(); d["bucket"] = TM.bucket_of(d["days"])
    lines = TM.report_block({"table": TM.tov_level_table(d, 20), "n_missing_sim": 0, "n_missing_actual": 0}, "2024-11-04")
    assert any("season-to-date" in l for l in lines) and "report only" in lines[0]
