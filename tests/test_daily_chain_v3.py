"""Tests for the daily chain v3 stages (sim guard, publish, grade, bias / CLV monitor). Synthetic frames; no engine run."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cbb_sim.live import daily as D  # noqa: E402
from cbb_sim.live import guards as G  # noqa: E402
from cbb_sim.live import lines as L  # noqa: E402

NOW = pd.Timestamp("2025-02-11T14:00:00Z")


def make_slate(n=3):
    return pd.DataFrame({
        "game_id": np.arange(1, n + 1), "cbbd_game_id": np.arange(101, 101 + n), "game_date": pd.Timestamp("2025-02-11"),
        "tipoff_utc": [NOW + pd.Timedelta(hours=h) for h in (-1, 3, 5)][:n],       # game 1 has already tipped
        "home_team_id": np.arange(10, 10 + n), "away_team_id": np.arange(20, 20 + n), "neutral": False})


def make_games(ids, seeds=40, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in ids:
        for s in range(seeds):
            h = int(rng.normal(75 + shift, 8)); a = int(rng.normal(70, 8))
            rows.append({"game_id": g, "seed": s, "home_pts": h, "away_pts": a, "possessions": 68.0, "n_periods": 2,
                         "home_fga3": 22, "away_fga3": 20, "created_at": NOW - pd.Timedelta(minutes=5)})
    return pd.DataFrame(rows)


def make_lines(ids, spread=-4.5, total=145.5):
    return pd.DataFrame({"cbbd_game_id": [100 + i for i in ids], "provider": "Draft Kings", "spread": spread, "total": total,
                         "home_ml": -190, "away_ml": 160, "spread_open": spread + 1, "total_open": total - 1,
                         "line_fetched_at": NOW - pd.Timedelta(minutes=1), "line_kind": "live", "ml_is_close_proxy": False})


def test_tipped_games_are_refused_and_assertion_fires():
    slate = make_slate()
    ok, late = D.split_tipped(slate, NOW)
    assert list(late["game_id"]) == [1] and list(ok["game_id"]) == [2, 3]
    with pytest.raises(G.LeakGuardError):                         # the guard itself fires on the late game
        G.assert_created_before_tipoff(slate.assign(created_at=NOW))
    G.assert_created_before_tipoff(ok.assign(created_at=NOW))      # the survivors pass
    # a game without a tip time can never be proven pregame
    s2 = slate.copy(); s2["tipoff_utc"] = pd.NaT
    assert len(D.split_tipped(s2, NOW)[0]) == 0


def test_publish_is_raw_sim_summary_and_refuses_late():
    slate = make_slate()
    games = make_games([1, 2, 3])
    ln = make_lines([1, 2, 3])
    pub, refused = D.build_publish(games, slate, ln, NOW, "s40_o0")
    assert list(refused["game_id"]) == [1] and set(pub["game_id"]) == {2, 3}
    g2 = games[games.game_id == 2]
    m = (g2.home_pts - g2.away_pts)
    r = pub[pub.game_id == 2].iloc[0]
    assert r["sim_margin_mean"] == pytest.approx(m.mean())          # nothing adjusted
    assert r["sim_margin_q50"] == pytest.approx(np.median(m))
    assert r["p_home_cover"] == pytest.approx(((m - 4.5) > 0).mean())
    assert r["p_home_cover"] + r["p_away_cover"] + r["p_ats_push"] == pytest.approx(1.0)
    assert r["p_over"] == pytest.approx(((g2.home_pts + g2.away_pts) > 145.5).mean())
    assert 0 < r["market_p_home_devig"] < 1 and r["ml_vig"] > 0
    assert pd.Timestamp(r["line_fetched_at"]) <= pd.Timestamp(r["published_at"]) < pd.Timestamp(r["tipoff_utc"])
    assert r["created_at"] <= r["published_at"]
    # same lines -> same publish_id (idempotent); moved line -> a new one
    pub2, _ = D.build_publish(games, slate, ln, NOW, "s40_o0")
    assert pub2["publish_id"].iloc[0] == pub["publish_id"].iloc[0]
    pub3, _ = D.build_publish(games, slate, make_lines([1, 2, 3], spread=-5.5), NOW, "s40_o0")
    assert pub3["publish_id"].iloc[0] != pub["publish_id"].iloc[0]
    assert "| tip (UTC) |" in D.slate_markdown(pub, refused, "2025-02-11", "x", NOW)


def test_publish_without_lines_still_publishes():
    pub, _ = D.build_publish(make_games([2, 3]), make_slate(), None, NOW, "r")
    assert len(pub) == 2 and pub["spread"].isna().all() and pub["sim_margin_sd"].notna().all()


def test_publish_rejects_line_stamped_after_clock():
    ln = make_lines([2, 3]); ln["line_fetched_at"] = NOW + pd.Timedelta(hours=1)
    with pytest.raises(G.LeakGuardError):
        D.build_publish(make_games([2, 3]), make_slate(), ln, NOW, "r")


def finals(ids, margins):
    return pd.DataFrame({"game_id": ids, "home_score": [80 + m for m in margins], "away_score": [80] * len(ids),
                         "finals_source": "t"})


def test_grade_settles_at_stored_lines_and_keeps_pending(tmp_path):
    slate, games = make_slate(), make_games([2, 3], shift=6.0)       # sim home margin around +11 > market 4.5 -> home lean
    pub, _ = D.build_publish(games, slate, make_lines([2, 3]), NOW, "r")
    led = D.settle(pub, finals([2], [10]), NOW + pd.Timedelta(days=1))
    assert list(led["game_id"]) == [2]                                # game 3 has no verified final: pending, not graded
    r = led.iloc[0]
    assert r["ats_lean"] == "home" and r["ats_result"] == "win" and r["ats_pnl"] == 1.0   # 10 > 4.5
    assert r["ml_lean"] == "home" and r["ml_pnl"] == pytest.approx(100 / 190)             # real odds -190
    led2 = D.settle(pub, finals([2], [3]), NOW)
    assert led2.iloc[0]["ats_result"] == "loss" and led2.iloc[0]["ats_pnl"] == -1.1
    # ledger upsert is idempotent
    p = tmp_path / "ledger.parquet"
    _, n1 = D.ledger_upsert(p, led); _, n2 = D.ledger_upsert(p, led)
    assert n1 == 1 and n2 == 0 and len(pd.read_parquet(p)) == 1


def test_first_publication_is_earliest():
    slate, games = make_slate(), make_games([2, 3])
    a, _ = D.build_publish(games, slate, make_lines([2, 3]), NOW, "r")
    b, _ = D.build_publish(games, slate, make_lines([2, 3], spread=-6.5), NOW + pd.Timedelta(hours=1), "r")
    first = D.first_publication(pd.concat([b, a], ignore_index=True))
    assert (first["spread"] == -4.5).all()


def test_monitor_bias_clv_and_alarms_never_modify_input():
    rng = np.random.default_rng(1)
    n = 400
    led = pd.DataFrame({"game_date": pd.date_range("2025-02-01", periods=n, freq="h").normalize(),
                        "sim_margin_mean": rng.normal(0, 8, n)})
    led["margin"] = led["sim_margin_mean"] + rng.normal(0, 11, n) - 2.0      # actual lower than sim: bias +2
    led["margin_err"] = led["sim_margin_mean"] - led["margin"]
    led["total_err"] = rng.normal(0, 14, n)
    led["market_margin"] = led["sim_margin_mean"] - rng.normal(0, 2, n)
    led["ats_disagree_pts"] = led["sim_margin_mean"] - led["market_margin"]
    led["ats_cover_margin"] = led["margin"] - led["market_margin"]
    led["ats_lean"] = np.where(led["ats_disagree_pts"] > 0, "home", "away")
    led["total"] = 140.0; led["ou_lean"] = ""; led["ml_lean"] = ""; led["market_p_home_devig"] = np.nan
    led["cbbd_game_id"] = np.arange(n); led["provider"] = "X"; led["ml_is_close_proxy"] = False
    close = pd.DataFrame({"cbbd_game_id": np.arange(n), "provider": "X", "spread": -(led["market_margin"] + 0.5), "total": 140.0,
                          "home_ml": np.nan, "away_ml": np.nan})
    before = led.copy()
    m = D.monitor(led, close, {"g10_surprise_corr": 0.15, "g10_clv_agreement": 0.53})
    pd.testing.assert_frame_equal(led, before)                          # input untouched: it reports, never corrects
    rb = m["rolling_bias"]
    allm = rb[(rb.window == "all") & (rb.quantity == "margin")].iloc[0]
    assert allm["bias"] == pytest.approx(led["margin_err"].mean()) and allm["se"] == pytest.approx(led["margin_err"].std(ddof=1) / np.sqrt(n))
    assert any(a.startswith("BIAS margin") for a in m["alarms"])        # +2 over 400 games is z ~ 3.6
    # close moved +0.5 toward home on every game: home leans gain, away leans lose
    d = m["clv_frame"]
    assert np.allclose(d.loc[d.ats_lean == "home", "clv_ats_pts"], 0.5)
    assert np.allclose(d.loc[d.ats_lean == "away", "clv_ats_pts"], -0.5)
    assert "leaked" in D.RULE_TEXT


def test_leak_verdict_path_flags_edge_without_line_movement():
    from cbb_sim.eval import market as M
    tol = {"g10_surprise_corr": 0.15, "g10_clv_agreement": 0.53}
    assert M.leak_verdict(0.4, 0.5, tol) == "LEAK-SUSPECT"
    assert M.leak_verdict(0.4, 0.7, tol) == "PASS"


def test_lines_provider_preference_and_devig():
    raw = pd.DataFrame({"cbbd_game_id": [1, 1, 2], "provider": ["Bovada", "Draft Kings", "Bovada"], "spread": [-3.0, -3.5, -1.0]})
    out = L.pick_provider(raw.assign(total=1, home_ml=-110, away_ml=-110), L.LIVE_PROVIDERS)
    assert out.set_index("cbbd_game_id")["provider"].to_dict() == {1: "Draft Kings", 2: "Bovada"}
    p, vig = L.devig(np.array([-110.0]), np.array([-110.0]))
    assert p[0] == pytest.approx(0.5) and vig[0] == pytest.approx(0.0476, abs=1e-3)
    assert len(L.pick_provider(raw.assign(total=1, home_ml=1, away_ml=1), ("ESPN BET",))) == 0


@pytest.mark.skipif(not (REPO / "data/processed/games_universe.parquet").exists(), reason="needs the games universe")
def test_sim_stage_parallel_jobs_is_bit_identical_to_single_process(tmp_path):
    """The sim stage split across worker processes by game (jobs > 1) writes the SAME games / players rows, values and order as the
    single in-process call: every draw is keyed on (seed, game_id). jobs is not part of the config hash. Tiny replay slate (4 games,
    2 seeds), 2 workers so the chunk boundary does not line up with the engine's 20-game block."""
    import run_daily_sim_v1 as SIM
    kw = dict(seeds=2, now="2025-02-11T14:00:00Z", schedule_source="universe", replay=True, players=True, max_games=4)
    r1 = SIM.run_sim_stage("2025-02-11", 2025, root=tmp_path / "j1", jobs=1, **kw)
    r2 = SIM.run_sim_stage("2025-02-11", 2025, root=tmp_path / "j2", jobs=2, **kw)
    assert r1["n_games"] == r2["n_games"] == 4 and r1["n_rows"] == r2["n_rows"] == 8
    for name, keys in (("games.parquet", ["game_id", "seed"]), ("players.parquet", ["game_id", "seed", "athlete_id"])):
        a, b = pd.read_parquet(Path(r1["out"]) / name), pd.read_parquet(Path(r2["out"]) / name)
        assert list(a.columns) == list(b.columns) and len(a) == len(b) and len(a) > 0
        assert a.equals(b), f"{name}: raw row order / values differ"
        a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)
        for c in a.columns:
            x, y = a[c].to_numpy(), b[c].to_numpy()
            assert (np.array_equal(x, y, equal_nan=True) if a[c].dtype.kind in "fc" else a[c].equals(b[c])), f"{name}.{c} differs"
    d1, d2 = (json.loads((Path(r["out"]) / "_DONE.json").read_text(encoding="utf-8")) for r in (r1, r2))
    assert d1["config_hash"] == d2["config_hash"]
    m1, m2 = (json.loads((Path(r["out"]) / "run_meta.json").read_text(encoding="utf-8")) for r in (r1, r2))
    assert m1["adapter_flags"]["ENGINE_EVENT"] == m2["adapter_flags"]["ENGINE_EVENT"] and m1["seeds"] == m2["seeds"]


@pytest.mark.skipif(not (REPO / "data/processed/games_universe.parquet").exists(), reason="needs the games universe")
def test_sim_stage_strict_raises_on_tipped_game_before_any_engine_work(tmp_path):
    import run_daily_sim_v1 as SIM
    with pytest.raises(G.LeakGuardError):
        SIM.run_sim_stage("2025-01-15", 2025, now="2025-01-16T03:00:00Z", root=tmp_path, schedule_source="universe", strict=True, replay=True)


def test_chain_v3_prereqs_list_day1_gaps():
    import chain_daily_v3 as C3
    miss = C3.sim_prereqs(2027, "F2", "2026-11-02", None)
    assert any("ratings" in m for m in miss)
    # the 2027 adapter / names keys exist after scripts/build_season_2027_artifacts_v1.py (lane F2); before it they are listed
    have = (C3.ENGINE_DIR / "event_round2_s1_F2_2027").exists() and (C3.ENGINE_DIR / "names_F2_2027_v2.json").exists()
    assert have != any("adapter" in m for m in miss)


def test_chain_exit_code_is_nonzero_when_a_stage_crashes():
    import chain_daily_v2 as V2
    import chain_daily_v3 as C3
    res = []
    V2.run_stage("ok", lambda: {}, res)
    V2.run_stage("blocked", lambda: {"_status": "blocked"}, res)
    assert C3.exit_code(res) == 0

    def boom():
        raise OSError("disk")
    assert V2.run_stage("ratings", boom, res).status == "error"
    assert C3.exit_code(res) == 1


@pytest.mark.skipif(not (REPO / "data/processed/games_universe.parquet").exists(), reason="needs the games universe")
def test_sim_stage_trajectory_parallel_shards_are_bit_identical(tmp_path):
    """B1 (2026-11-02): --trajectory-seeds under the parallel path. Per-worker shards merged into trajectory.parquet; games / players
    identical with the writer off, on single-process, and on 2 / 3 workers; the trajectory rows are identical (sorted on game, seed,
    poss_idx; global game ids) and shards are gone."""
    import run_daily_sim_v1 as SIM
    import shutil
    kw = dict(seeds=3, now="2025-02-11T14:00:00Z", schedule_source="universe", replay=True, players=True, max_games=4)
    ra = SIM.run_sim_stage("2025-02-11", 2025, root=tmp_path / "off", jobs=1, **kw)
    rs = [SIM.run_sim_stage("2025-02-11", 2025, root=tmp_path / f"on{j}", jobs=j, trajectory_seeds=2, run_id=f"pytest_b1_t{j}", **kw) for j in (1, 2, 3)]
    for name in ("games.parquet", "players.parquet"):
        base = pd.read_parquet(Path(ra["out"]) / name)
        for r in rs:
            assert base.equals(pd.read_parquet(Path(r["out"]) / name)), name
    key = ["game_id", "seed", "poss_idx"]
    tr = [pd.read_parquet(r["trajectory"]["path"]).sort_values(key).reset_index(drop=True) for r in rs]
    assert len(tr[0]) > 0 and tr[0].equals(tr[1]) and tr[0].equals(tr[2])
    games = pd.read_parquet(Path(ra["out"]) / "games.parquet")
    assert set(tr[1].game_id) == set(games.game_id) and set(tr[1].seed) == {0, 1}
    assert {r["trajectory"]["rows"] for r in rs} == {len(tr[0])} and {r["trajectory"]["game_seeds"] for r in rs} == {8}
    assert not (Path(rs[1]["trajectory"]["path"]).parent / "_shards").exists()
    for r in rs:
        shutil.rmtree(Path(r["trajectory"]["path"]).parent, ignore_errors=True)
