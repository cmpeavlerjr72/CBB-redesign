"""Grade + bias/CLV stages end to end on fabricated 2026-27 finals (ops 2026-10-09, docs/ops/grade_stage_preflight_2026-10-09.md).

The grade and bias stages have never run on 2026-27 (always SKIPPED before the season). These tests fabricate a slate of 20 games on the
2026-11-02 opener, verified finals for it (one OT game, one placeholder-tip "tip time unknown" game), and run the real stage functions into
a tmp root. They also cover the two-source finals rule (one-point disagreement is never graded) and the early-season totals label.
All fabricated truth lives in tmp_path and is removed with it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO / "tests")]

import diag_daily_bias_clv_v1 as MON  # noqa: E402
import grade_daily_v1 as GR  # noqa: E402
from cbb_sim.live import daily as D  # noqa: E402

DAY = "2026-11-02"
PUBLISHED = pd.Timestamp("2026-10-09T13:00:00Z")
GRADE_NOW = pd.Timestamp("2026-11-03T14:00:00Z")
N = 20
OT_GAME, PLACEHOLDER_GAME = 5, 7          # game_ids in the fabricated slate


def fab_slate() -> pd.DataFrame:
    tips = [pd.Timestamp("2026-11-02T18:00:00Z") + pd.Timedelta(hours=i % 6) for i in range(N)]
    ph = np.zeros(N, dtype=bool)
    ph[PLACEHOLDER_GAME - 1] = True
    tips[PLACEHOLDER_GAME - 1] = pd.Timestamp("2026-11-02T05:00:00Z")       # 00:00 ET placeholder
    return pd.DataFrame({"game_id": np.arange(1, N + 1), "cbbd_game_id": np.arange(101, 101 + N), "season": 2027,
                         "game_date": pd.Timestamp(DAY), "tipoff_utc": tips, "home_team_id": np.arange(10, 10 + N),
                         "away_team_id": np.arange(40, 40 + N), "neutral": False, "tip_source": "cbbd",
                         "tip_time_is_placeholder": ph})


def fab_games(seeds: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for g in range(1, N + 1):
        for s in range(seeds):
            rows.append({"game_id": g, "seed": s, "home_pts": int(rng.normal(72, 9)), "away_pts": int(rng.normal(70, 9)),
                         "possessions": 67.0, "n_periods": 3 if (g == OT_GAME and s % 10 == 0) else 2,
                         "created_at": PUBLISHED - pd.Timedelta(minutes=5)})
    return pd.DataFrame(rows)


def fab_lines() -> pd.DataFrame:
    return pd.DataFrame({"cbbd_game_id": np.arange(101, 101 + N), "provider": "Draft Kings", "spread": -2.5, "total": 142.5,
                         "home_ml": -140, "away_ml": 120, "spread_open": -2.0, "total_open": 141.5,
                         "line_fetched_at": PUBLISHED - pd.Timedelta(minutes=1), "line_kind": "live", "ml_is_close_proxy": False})


@pytest.fixture()
def world(tmp_path, monkeypatch):
    root = tmp_path / "daily"
    monkeypatch.setattr(GR, "REPO", tmp_path)                      # no real tip_times_2027 -> placeholder row stays "unverified"
    monkeypatch.setattr(GR, "INGEST_DIR", tmp_path / "ingest")
    monkeypatch.setattr(GR, "season_start_of", lambda season: DAY)  # the opener
    (tmp_path / "ingest").mkdir()
    pub, refused = D.build_publish(fab_games(), fab_slate(), fab_lines(), PUBLISHED, "s40_o0", season_start=DAY,
                                   build_ids={"inputs_hash": "abc123", "config_hash": "cfg1", "engine_tag": "t"})
    assert len(pub) == N and not len(refused)
    out = D.pub_dir(root, DAY, "s40_o0") / pub["publish_id"].iloc[0]
    out.mkdir(parents=True)
    pub.to_parquet(out / "slate.parquet", index=False)
    return root, pub, tmp_path


def fab_finals(pub: pd.DataFrame, drop=()) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    h = rng.integers(55, 95, N)
    a = rng.integers(55, 95, N)
    h[OT_GAME - 1], a[OT_GAME - 1] = 84, 82                         # OT game
    f = pd.DataFrame({"game_id": np.arange(1, N + 1), "home_score": h, "away_score": a, "n_periods": 2,
                      "finals_source": "fabricated_test"})
    f.loc[f.game_id == OT_GAME, "n_periods"] = 3
    return f[~f.game_id.isin(drop)].reset_index(drop=True)


def test_grade_and_bias_end_to_end_on_fabricated_finals(world):
    root, pub, _ = world
    fin = fab_finals(pub)
    r = GR.run_grade_stage(DAY, 2027, GRADE_NOW, root, finals_frame=fin, truth_box=pd.DataFrame(), tov_monitor=False, n_boot=20)
    assert r["_status"] == "ok" and r["graded"] == N and r["pending"] == 0
    led = pd.read_parquet(root / "grade" / "ledger.parquet")
    assert len(led) == N
    led = led.set_index("game_id")
    # per-game rows are exactly sim minus verified actual
    assert (led["margin"] == led["home_score"] - led["away_score"]).all()
    assert np.allclose(led["margin_err"], led["sim_margin_mean"] - led["margin"])
    assert np.allclose(led["total_err"], led["sim_total_mean"] - led["total_pts"])
    assert led.at[OT_GAME, "n_periods"] == 3 and led.at[OT_GAME, "margin"] == 2
    # the placeholder-tip game is graded but stays flagged (no real tip known): never assumed verified
    assert led.at[PLACEHOLDER_GAME, "pre_tip_status"] == "placeholder_unverified"
    assert not bool(led.at[PLACEHOLDER_GAME, "pre_tip_verified"])
    assert (led.drop(index=PLACEHOLDER_GAME)["pre_tip_status"] == "real_tip").all()
    # ATS / O/U / ML columns settled at the stored lines
    assert set(led["ats_result"]) <= {"win", "loss", "push", "none"} and led["ats_pnl"].notna().all()
    assert led["ml_pnl"].notna().all()
    # label and trace columns reach the ledger rows
    assert led["early_season_totals_flag"].astype(bool).all()
    assert (led["inputs_hash"] == "abc123").all()
    summ = pd.read_json(root / "grade" / DAY / "summary.json", typ="series")
    assert summ["n_games"] == N and summ["n_early_season_totals_flag"] == N
    assert (root / "grade" / DAY / "report.md").exists() and (root / "grade" / DAY / "pending.parquet").exists()
    # idempotent re-grade
    r2 = GR.run_grade_stage(DAY, 2027, GRADE_NOW, root, finals_frame=fin, truth_box=pd.DataFrame(), tov_monitor=False, n_boot=20)
    assert r2["new_ledger_rows"] == 0 and r2["ledger_rows"] == N

    # ---- bias / CLV monitor on the same ledger, closing lines moved 1 point
    close = fab_lines().assign(spread=-3.5, total=143.5, line_kind="close")
    asof = GRADE_NOW + pd.Timedelta(hours=1)
    m = MON.run_monitor_stage(2027, asof, root, close_frame=close)
    assert m["_status"] in ("ok", "alarm") and m["n_ledger"] == N
    mdir = root / "monitor" / asof.strftime("%Y-%m-%d")
    assert (mdir / "report.md").exists() and (mdir / "monitor.json").exists()
    clv = pd.read_parquet(mdir / "clv.parquet")
    assert len(clv) == N and clv["early_season_totals_flag"].astype(bool).all()     # label reaches the CLV rows
    assert clv["clv_ou_pts"].notna().all()
    assert "UNDERPOWERED" in m["leak_verdict_ats"]                                  # 20 games: labelled, not read as signal


def test_missing_final_stays_pending(world):
    root, pub, _ = world
    r = GR.run_grade_stage(DAY, 2027, GRADE_NOW, root, finals_frame=fab_finals(pub, drop=(3, 4)), truth_box=pd.DataFrame(),
                           tov_monitor=False, n_boot=20)
    assert r["graded"] == N - 2 and r["pending"] == 2
    pend = pd.read_parquet(root / "grade" / DAY / "pending.parquet")
    assert set(pend["game_id"]) == {3, 4} and set(pend["reason"]) == {"no_verified_final"}


def test_one_point_source_disagreement_is_never_graded(world):
    """hoopR vs CBBD differ by one point on game 2: the ingest finals table must not grade it, and the pending reason names the disagreement."""
    root, pub, tmp = world
    fin = fab_finals(pub)
    ing = pd.DataFrame({"game_id": fin["game_id"], "hoopr_home": fin["home_score"], "hoopr_away": fin["away_score"],
                        "cbbd_home": fin["home_score"], "cbbd_away": fin["away_score"], "verified_at": GRADE_NOW - pd.Timedelta(hours=2)})
    ing.loc[ing.game_id == 2, "cbbd_away"] += 1                                    # one-point disagreement
    ing.loc[ing.game_id == 9, "cbbd_home"] = np.nan                                 # CBBD side missing: also not verified
    ing.to_parquet(tmp / "ingest" / "finals_verified_2027.parquet", index=False)
    f = GR.finals_ingest(2027)
    assert 2 not in set(f["game_id"]) and 9 not in set(f["game_id"]) and len(f) == N - 2
    r = GR.run_grade_stage(DAY, 2027, GRADE_NOW, root, finals="ingest", truth_box=pd.DataFrame(), tov_monitor=False, n_boot=20)
    assert r["graded"] == N - 2 and r["pending"] == 2
    led = pd.read_parquet(root / "grade" / "ledger.parquet")
    assert 2 not in set(led["game_id"]) and 9 not in set(led["game_id"])
    pend = pd.read_parquet(root / "grade" / DAY / "pending.parquet").set_index("game_id")
    assert pend.at[2, "reason"] == "finals_sources_disagree" and pend.at[9, "reason"] == "finals_sources_disagree"


def test_early_season_flag_boundaries():
    d = pd.Series(pd.to_datetime(["2026-11-02", "2026-11-16", "2026-11-17", "2026-10-31"]))
    out = D.early_season_flag(d, "2026-11-02")
    assert out.tolist() == [True, True, False, False]                              # d0..d14 inclusive; before start is not early
    assert D.early_season_flag(d, None).isna().all()                               # unknown start is NA, never silently False


def test_publish_rows_carry_label_and_trace_without_changing_numbers(world):
    _, pub, _ = world
    base, _ = D.build_publish(fab_games(), fab_slate(), fab_lines(), PUBLISHED, "s40_o0")      # old call signature still works
    assert base["early_season_totals_flag"].isna().all()
    assert pub["early_season_totals_flag"].all() and (pub["inputs_hash"] == "abc123").all()
    num = [c for c in pub.columns if c.startswith("sim_") or c in ("p_home", "ats_disagree_pts", "ou_disagree_pts")]
    pd.testing.assert_frame_equal(pub[num], base[num])                              # label/trace are additive only
    assert pub["publish_id"].iloc[0] == base["publish_id"].iloc[0]
    late = D.build_publish(fab_games(), fab_slate().assign(game_date=pd.Timestamp("2027-01-15")), fab_lines(), PUBLISHED, "r",
                           season_start=DAY)[0]
    assert not late["early_season_totals_flag"].any()


REAL = sorted((REPO / "results/daily/publish" / DAY).glob("s200_o0/*/slate.parquet"))


@pytest.mark.skipif(not REAL, reason="real 2026-11-02 served publication not on disk")
def test_real_served_slate_grades_with_fabricated_finals(tmp_path, monkeypatch):
    """Same run on 20 games of the real served 11-02 publication (sim numbers real, finals fabricated)."""
    real = pd.read_parquet(REAL[-1]).sort_values("game_id").head(20).reset_index(drop=True)
    root = tmp_path / "daily"
    out = D.pub_dir(root, DAY, str(real["run_id"].iloc[0])) / str(real["publish_id"].iloc[0])
    out.mkdir(parents=True)
    real.to_parquet(out / "slate.parquet", index=False)
    monkeypatch.setattr(GR, "REPO", tmp_path)
    rng = np.random.default_rng(5)
    fin = pd.DataFrame({"game_id": real["game_id"], "home_score": rng.integers(55, 95, 20), "away_score": rng.integers(55, 95, 20),
                        "finals_source": "fabricated_test"})
    r = GR.run_grade_stage(DAY, 2027, GRADE_NOW, root, finals_frame=fin, truth_box=pd.DataFrame(), tov_monitor=False, n_boot=20)
    assert r["_status"] == "ok" and r["graded"] == 20
    led = pd.read_parquet(root / "grade" / "ledger.parquet")
    assert "early_season_totals_flag" in led.columns and led["early_season_totals_flag"].astype(bool).all()   # backfilled if absent
    m = MON.run_monitor_stage(2027, GRADE_NOW + pd.Timedelta(hours=1), root, close="none")
    assert m["_status"] in ("ok", "alarm") and m["n_ledger"] == 20
