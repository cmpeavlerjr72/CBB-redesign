"""Free injury feed (ops 2026-10-09, docs/ops/injury_feed_2026-10-09.md): parse, schema safety, Out-only application, leak guard, and the
end-to-end effect on a real 11-02 game: the Out player's minutes go to 0 and the team's slot allocation re-closes."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import pull_injuries_v1 as IJ  # noqa: E402
import run_daily_sim_v1 as SIM  # noqa: E402
from cbb_sim.live import guards as G  # noqa: E402

NOW = "2026-11-01T01:00:00Z"
DAY = date(2026, 10, 31)          # ET date of NOW


def _inj(status, aid=None, link_id=None, name="A Player"):
    ath = {"displayName": name, "team": {"id": "99"}}
    if aid is not None:
        ath["id"] = str(aid)
    if link_id is not None:
        ath["links"] = [{"rel": ["playercard", "desktop", "athlete"], "href": f"https://www.espn.com/mens-college-basketball/player/_/id/{link_id}/a-player"}]
    return {"status": status, "athlete": ath, "details": {"type": "Knee", "returnDate": "2026-11-20"}, "shortComment": "sat out"}


def _payload(*injs):
    return {"status": "success", "season": {"year": 2027}, "injuries": [{"id": "150", "displayName": "Duke", "injuries": list(injs)}]}


def _rosters(tmp_path):
    d = tmp_path / "rosters"
    d.mkdir()
    pd.DataFrame({"espn_player_id": [1001, 1002, 1003], "cbbd_player_id": [501, 502, 503]}).to_parquet(d / "roster_2027.parquet")
    pd.DataFrame({"source_id": ["2001"], "cbbd_player_id": [601]}).to_parquet(d / "roster_2026.parquet")
    x = tmp_path / "xw.parquet"
    pd.DataFrame({"espn_team_id": [150], "cbbd_team_id": [7]}).to_parquet(x)
    return d, x


def test_parse_reads_id_from_athlete_or_player_card_link():
    df, info = IJ.parse_league(_payload(_inj("Out", aid=1001), _inj("Questionable", link_id=1002), _inj("Day-To-Day", aid=1003)), NOW, str(DAY))
    assert sorted(df["espn_player_id"]) == [1001, 1002, 1003] and info["entries_raw"] == 3
    assert set(df["status_norm"]) == {"out", "questionable", "day-to-day"}
    assert {"season", "team_id", "espn_player_id", "player_name", "status", "detail", "source", "pulled_at"} <= set(df.columns)


def test_nonempty_payload_without_athlete_ids_is_a_hard_error():
    with pytest.raises(RuntimeError, match="schema drift"):
        IJ.parse_league(_payload(_inj("Out")), NOW, str(DAY))


def test_empty_payload_is_zero_rows_not_an_error():
    df, info = IJ.parse_league({"status": "success", "season": {"year": 2027}, "injuries": []}, NOW, str(DAY))
    assert len(df) == 0 and info["entries_raw"] == 0


def test_attach_ids_maps_team_and_player(tmp_path):
    d, x = _rosters(tmp_path)
    df, _ = IJ.parse_league(_payload(_inj("Out", aid=1001), _inj("Out", aid=2001), _inj("Out", aid=9)), NOW, str(DAY))
    a = IJ.attach_ids(df, 2027, d, x).set_index("espn_player_id")
    assert a.loc[1001, "cbbd_player_id"] == 501 and a.loc[2001, "cbbd_player_id"] == 601 and pd.isna(a.loc[9, "cbbd_player_id"])
    assert (a["cbbd_team_id"] == 7).all()


def test_write_pull_parquet_and_manifest(tmp_path):
    d, x = _rosters(tmp_path)
    df, info = IJ.parse_league(_payload(_inj("Out", aid=1001), _inj("Questionable", aid=1002)), NOW, str(DAY))
    df = IJ.attach_ids(df, 2027, d, x)
    man = IJ.write_pull(df, info, DAY, tmp_path / "inj", pulled_at=NOW)
    back = pd.read_parquet(tmp_path / "inj" / f"injuries_{DAY}.parquet")
    assert len(back) == 2 and man["out_rows"] == 1 and man["by_status"] == {"Out": 1, "Questionable": 1}
    assert list((tmp_path / "inj" / "manifests").glob("injuries_*.json"))


def _feed(tmp_path, *injs, pulled_at=NOW):
    d, x = _rosters(tmp_path)
    df, info = IJ.parse_league(_payload(*injs), pulled_at, str(DAY))
    IJ.write_pull(IJ.attach_ids(df, 2027, d, x), info, DAY, tmp_path / "inj", pulled_at=pulled_at)
    return d


def test_only_out_is_applied(tmp_path):
    d = _feed(tmp_path, _inj("Out", aid=1001), _inj("Out For Season", aid=1002), _inj("Questionable", aid=1003), _inj("Doubtful", aid=2001),
              _inj("Suspension", aid=9))
    o = IJ.out_players(DAY, NOW, tmp_path / "inj", d, tmp_path / "none.csv")
    assert sorted(o["cbbd_player_id"].astype(int)) == [501, 502]


def test_manual_out_rows_merge(tmp_path):
    d = _feed(tmp_path, _inj("Out", aid=1001))
    ov = tmp_path / "availability.csv"
    pd.DataFrame({"athlete_id": [1003], "date": [str(DAY)], "status": ["out"], "note": ["x"]}).to_csv(ov, index=False)
    o = IJ.out_players(DAY, "2030-01-01T00:00:00Z", tmp_path / "inj", d, ov)
    assert sorted(o["cbbd_player_id"].astype(int)) == [501, 503] and set(o["source"]) == {"espn_league_injuries", "manual"}


def test_row_pulled_after_the_sim_clock_raises(tmp_path):
    d = _feed(tmp_path, _inj("Out", aid=1001), pulled_at="2026-11-01T03:00:00Z")
    with pytest.raises(G.LeakGuardError):
        IJ.out_players(DAY, NOW, tmp_path / "inj", d, tmp_path / "none.csv")


def test_no_feed_file_means_nobody_out(tmp_path):
    o = IJ.out_players(DAY, NOW, tmp_path / "empty", tmp_path, tmp_path / "none.csv")
    assert len(o) == 0


def test_replay_never_reads_the_feed():
    a, pids, meta, dig = SIM.injuries_for(NOW, 2025, replay=True)
    assert a is None and not pids and dig == "replay"


# ---- end to end on a real 11-02 game: Out player's minutes -> 0, slot allocation re-closes ---------------------------------------------

def _have_real_inputs():
    need = ["data/raw/cbbd/rosters/roster_2027.parquet", "data/processed/ratings_asof/2026-11-02/own_ratings_2027.parquet",
            "data/processed/ingest/tip_times_2027.parquet"]
    return all((REPO / n).exists() for n in need)


@pytest.mark.skipif(not _have_real_inputs(), reason="needs the 2027 serving inputs on disk")
def test_out_player_minutes_go_to_zero_and_slots_reclose(tmp_path, monkeypatch):
    import build_engine_inputs_live as BL
    import chain_daily_v3 as V3
    import run_engine_live as RL
    monkeypatch.setattr(SIM, "injuries_for", lambda now, season, replay: (None, frozenset(), {}, "none"))   # baseline: nobody out
    now = pd.Timestamp("2026-10-09T13:00:00Z")
    rd = str(REPO / "data/processed/ratings_asof/2026-11-02")

    def build(out_pids):
        slate = SIM.load_slate("2026-11-02", 2027, "cbbd", str(V3.SCHED_2027), str(V3.CROSSWALK), "table").iloc[:1]
        cols = ["game_id", "cbbd_game_id", "season", "game_date", "tipoff_utc", "home_team_id", "away_team_id", "neutral"]
        D1P, fn = SIM.day1_prior_seed(2027, frozenset(out_pids)) if out_pids else SIM.day1_prior_seed(2027)
        inp, _ = BL.build_live(slate[cols], now, 2027, "F2", created_at=now, season_start="2026-11-02", strict_finish=False, ratings_dir=rd, seed_fn=fn)
        D1P.post(inp, fn)
        return inp, fn

    base, _ = build([])
    ros0 = base.roster_cbbd[0, 0]
    named0 = ros0[ros0 > 0]
    victim = int(named0[0])                                     # the top-ranked named player of the home side
    inp, fn = build([victim])
    ros = inp.roster_cbbd[0, 0]
    assert victim not in ros                                    # removed from the available roster
    assert fn.out_removed == {victim: 1}
    after = [int(x) for x in ros[ros > 0]]
    assert [p for p in map(int, named0) if p != victim] == after[:len(named0) - 1]   # everyone else keeps their order and moves up one slot
    # slot allocation re-closes: the rotation shares over the remaining slots still sum to 1, and the other side is untouched
    assert float(inp.rot_share[0, 0].sum()) == pytest.approx(1.0, abs=1e-4)
    assert np.array_equal(inp.roster_cbbd[0, 1], base.roster_cbbd[0, 1])
    # the engine: the Out player has no rows (0 minutes); a teammate absorbs minutes, so the named minutes still fill the side
    import build_shot_block_lut_live_v1 as SBL
    adir = RL.prepare_adapter_dir(inp.event_block, "F2", 2027, tmp_path / "_adapter")
    SBL.attach(inp, tmp_path / "_adapter", as_of=now, seeded_sides=fn.seeded)
    _, pl, _ = RL.simulate(inp, "F2", 2027, np.arange(2, dtype=np.int64), keep_players=True, adapter_dir=adir)
    assert pl is not None and len(pl)
    assert victim not in set(pl["cbbd_id"].astype(int))
    side = pl[pl["team_id"] == int(inp.games["home_team_id"].iloc[0])]
    assert side.groupby("seed")["minutes"].sum().min() > 100.0  # the home side's named players still carry the minutes
