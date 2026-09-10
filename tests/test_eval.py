"""Tests for the engine-agnostic evaluation harness (`src/cbb_sim/eval/`).

Three things this task requires proof of:

1. The results-contract VALIDATOR actually refuses malformed artifacts (a
   games.parquet missing a required column, a live run whose created_at is
   not before tipoff, a run_meta.json claiming it touched the sealed
   season) -- and accepts the Control's legacy pre-contract shape via the
   documented shims, never silently.
2. A GATE actually distinguishes PASS from FAIL on synthetic data with a
   known-good and a known-bad cell (`gate_g6`, the smallest gate to set up
   without touching disk).
3. The LEAK-SUSPECT rule (`market.leak_verdict`) fires exactly on the
   documented (surprise_corr, clv_agreement) combination, matches the CLV
   coin-flip story, and reports NEEDS-INSTRUMENTATION rather than a silent
   PASS when there are no opening lines to check.

Plus one synthetic end-to-end run of `props.grade_props()` proving the
player-prop grader (for which no real lines exist yet, see
`scripts/grade_market_props.py`) is correct today, against a fabricated
lines fixture matching the documented schema.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cbb_sim.eval import contract as C
from cbb_sim.eval import gates as G
from cbb_sim.eval import market as M
from cbb_sim.eval import props as P

TOL = {
    "g10_surprise_corr": 0.15,
    "g10_clv_agreement": 0.53,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_run_meta(results_dir, **overrides):
    meta = {
        "engine_tag": "synthetic_engine",
        "created_at": "2026-09-10T00:00:00+00:00",
        "seeds": 10,
        "fold": "F2",
        "backtest": True,
        "sealed_touched": False,
    }
    meta.update(overrides)
    (results_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _minimal_games_df(n_games=3, n_seeds=10):
    rows = []
    for gid in range(n_games):
        for seed in range(n_seeds):
            rows.append({
                "game_id": gid, "seed": seed, "home_pts": 70 + seed % 5, "away_pts": 65 + seed % 3,
                "possessions": 68.0, "n_periods": 2,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. contract validator
# ---------------------------------------------------------------------------
def test_load_games_refuses_missing_required_column(tmp_path):
    df = _minimal_games_df().drop(columns=["possessions"])
    df.to_parquet(tmp_path / "games.parquet")
    with pytest.raises(C.ContractError, match="possessions"):
        C.load_games(tmp_path)


def test_load_games_legacy_n_ot_shim(tmp_path):
    df = _minimal_games_df().drop(columns=["n_periods"])
    df["n_ot"] = [0, 1] * (len(df) // 2)
    df.to_parquet(tmp_path / "games.parquet")
    with pytest.warns(UserWarning, match="legacy"):
        res = C.load_games(tmp_path)
    assert "n_periods" in res.frame.columns
    assert (res.frame["n_periods"] == 2 + df["n_ot"]).all()
    assert res.warnings  # non-fatal, but recorded


def test_load_players_missing_file_is_not_an_error(tmp_path):
    res = C.load_players(tmp_path)
    assert res.frame is None
    assert res.warnings == []


def test_load_players_refuses_incomplete_schema(tmp_path):
    bad = pd.DataFrame({"game_id": [1], "seed": [0], "athlete_id": [1]})  # missing minutes, pts, ...
    bad.to_parquet(tmp_path / "players.parquet")
    with pytest.raises(C.ContractError, match="minutes"):
        C.load_players(tmp_path)


def test_run_meta_legacy_defaults_are_recorded(tmp_path):
    meta = {"created_at": "2026-09-10T00:00:00+00:00", "seeds": 10, "fold": "F2", "backtest": True}
    (tmp_path / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.warns(UserWarning):
        res = C.load_run_meta(tmp_path)
    assert res.frame["engine_tag"] == tmp_path.name
    assert res.frame["sealed_touched"] is False


def test_run_meta_missing_required_key_raises(tmp_path):
    meta = {"created_at": "2026-09-10T00:00:00+00:00", "seeds": 10, "backtest": True}  # no fold
    (tmp_path / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(C.ContractError, match="fold"):
        C.load_run_meta(tmp_path)


def test_tipoff_safety_backtest_true_skips_check():
    games = pd.DataFrame({"created_at": ["2026-01-01"], "tipoff": ["2020-01-01"]})  # nonsensically "after", but backtest
    problems = C.validate_tipoff_safety(games, {"backtest": True})
    assert problems == []


def test_tipoff_safety_live_run_catches_violation():
    games = pd.DataFrame({
        "created_at": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        "tipoff": pd.to_datetime(["2026-01-02", "2025-12-31"]),  # second row violates created_at < tipoff
    })
    problems = C.validate_tipoff_safety(games, {"backtest": False})
    assert len(problems) == 1
    assert "1 row" in problems[0]


def test_tipoff_safety_live_run_passes_when_honest():
    games = pd.DataFrame({
        "created_at": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        "tipoff": pd.to_datetime(["2026-01-02", "2026-01-03"]),
    })
    assert C.validate_tipoff_safety(games, {"backtest": False}) == []


def test_load_engine_results_refuses_sealed_touched(tmp_path):
    _write_run_meta(tmp_path, sealed_touched=True)
    _minimal_games_df().to_parquet(tmp_path / "games.parquet")
    with pytest.raises(C.ContractError, match="sealed_touched"):
        C.load_engine_results(tmp_path)
    # deliberate override works
    engine = C.load_engine_results(tmp_path, allow_sealed=True)
    assert engine.run_meta["sealed_touched"] is True


def test_load_engine_results_happy_path(tmp_path):
    _write_run_meta(tmp_path)
    _minimal_games_df().to_parquet(tmp_path / "games.parquet")
    engine = C.load_engine_results(tmp_path)
    assert engine.engine_tag == "synthetic_engine"
    assert len(engine.games) == 30
    assert engine.has_players is False
    assert all(not ok for ok in engine.box_available.values())  # no optional box columns supplied


# ---------------------------------------------------------------------------
# 2. synthetic gate pass/fail (G6 -- smallest gate that needs no disk I/O)
# ---------------------------------------------------------------------------
def test_gate_g6_distinguishes_pass_from_fail():
    tol = {"g6_margin": 1.0}
    rows = []
    # non-neutral: sim tracks actual closely (within tolerance) -> PASS
    for i in range(20):
        rows.append({"neutral": 0.0, "sim_margin_mean": 6.0 + (i % 2) * 0.1, "margin": 5.8})
    # neutral: sim badly overshoots actual -> FAIL
    for _i in range(20):
        rows.append({"neutral": 1.0, "sim_margin_mean": 8.0, "margin": 2.5})
    summary = pd.DataFrame(rows)

    result = G.gate_g6(summary, tol, min_cell_n=1)
    by_site = {c.quantity: c.status for c in result.checks}
    assert by_site["home margin (non-neutral)"] == "PASS"
    assert by_site["home margin (neutral)"] == "FAIL"
    assert result.status == "FAIL"  # any FAIL check fails the whole gate


def test_gate_g6_underpowered_cells_are_labelled_not_scored():
    tol = {"g6_margin": 1.0}
    summary = pd.DataFrame([
        {"neutral": 0.0, "sim_margin_mean": 6.0, "margin": 5.9},
        {"neutral": 1.0, "sim_margin_mean": 2.0, "margin": 2.1},
    ])
    result = G.gate_g6(summary, tol, min_cell_n=300)  # n=1 per cell, far below the floor
    assert all(c.status == "UNDERPOWERED" for c in result.checks)
    # SIM_GUARDRAILS: underpowered is never presented as PASS or FAIL evidence.
    assert result.status == "NEEDS-INSTRUMENTATION"


# ---------------------------------------------------------------------------
# 3. the LEAK-SUSPECT rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "surprise_corr,clv_agreement,expected",
    [
        (0.20, 0.50, "LEAK-SUSPECT"),   # beats the close, but the edge is coin-flip on which way it moves
        (0.20, 0.60, "PASS"),           # beats the close AND correctly anticipates the move: real information
        (0.05, 0.50, "PASS"),           # never beat the close in the first place: nothing to suspect
        (0.20, float("nan"), "NEEDS-INSTRUMENTATION"),  # real surprise corr but no opens to check CLV against
        (0.15, 0.50, "PASS"),           # exactly at the gate is not ">" it
        (0.20, 0.53, "PASS"),           # exactly at the CLV gate is not "<" it
    ],
)
def test_leak_verdict(surprise_corr, clv_agreement, expected):
    assert M.leak_verdict(surprise_corr, clv_agreement, TOL) == expected


def test_gate_g10_end_to_end_flags_leak_suspect():
    """Construct a summary + lines fixture where the model's disagreement
    with the close is real (correlates with the actual surprise) but is a
    coin flip on which way the market later moved the line -- exactly the
    CLV-paradox story `docs/postmortem/05_cfb_methodology_extract.md`
    section 4 describes as how the original leak was actually caught."""
    rng = np.random.default_rng(20260910)
    n = 800
    market_margin = rng.normal(0, 8, n)
    noise = rng.normal(0, 6, n)
    actual_margin = market_margin + noise
    # the model's disagreement with the close IS correlated with the actual surprise (a real edge)...
    disagree = 0.6 * noise + rng.normal(0, 3, n)
    model_margin = market_margin + disagree
    # ...but the line's later movement (close - open) is INDEPENDENT random noise, and small relative
    # to the model's own disagreement, so the model's stance at the open cannot anticipate which way
    # the market later moves -- the coin-flip signature of a leaked (post-hoc-only) feature. (Algebraically
    # dis_open = disagree_close + move always, so CLV agreement -> 0.5 as move's scale -> 0 relative to
    # disagree_close's; move is kept small on purpose to isolate that limit rather than fight it.)
    move = rng.normal(0, 0.05, n)
    open_margin = market_margin - move

    home_score = np.round(70 + actual_margin / 2).astype(int)
    away_score = np.round(70 - actual_margin / 2).astype(int)
    summary = pd.DataFrame({
        "cbbd_game_id": np.arange(n), "margin": actual_margin, "total": home_score + away_score,
        "sim_margin_mean": model_margin, "sim_total_mean": 145.0,
        "p_home": 0.5 + np.clip(model_margin, -40, 40) / 200, "is_home_win": (actual_margin > 0).astype(float),
    })
    lines = pd.DataFrame({
        "cbbd_game_id": np.arange(n), "provider_used": "ESPN BET",
        "spread": -market_margin, "spreadOpen": -open_margin,
        "overUnder": 145.0, "overUnderOpen": 145.0,
        "homeMoneyline": -120.0, "awayMoneyline": 100.0,
    })
    tol = {**TOL}
    res = M.gate_g10(summary, season=2025, tol=tol, lines=lines, n_boot=200)
    assert res.surprise_corr > tol["g10_surprise_corr"]
    assert res.clv_agreement == pytest.approx(0.5, abs=0.08)
    assert res.leak_status == "LEAK-SUSPECT"
    assert res.notes and "DO NOT TRUST" in res.notes[0]


# ---------------------------------------------------------------------------
# 4. synthetic player-prop grading (no real props lines exist yet)
# ---------------------------------------------------------------------------
def test_props_synthetic_grading():
    # two games, two players each getting a "points" prop line
    seeds = np.arange(40)
    players_rows = []
    for game_id, athlete_id, base in ((1, 101, 14.0), (2, 102, 8.0)):
        rng = np.random.default_rng(athlete_id)
        pts = np.round(rng.normal(base, 4.0, len(seeds))).clip(min=0)
        for seed, p in zip(seeds, pts, strict=True):
            players_rows.append({
                "game_id": game_id, "seed": int(seed), "athlete_id": athlete_id, "team_id": athlete_id // 100,
                "minutes": 30.0, "pts": float(p), "reb": 5.0, "ast": 3.0, "fga": 10.0, "fg3a": 2.0, "fta": 3.0,
            })
    players = pd.DataFrame(players_rows)

    actual_games = pd.DataFrame({"game_id": [1, 2], "cbbd_game_id": [901, 902]})

    props_lines = pd.DataFrame([
        {"cbbd_game_id": 901, "athlete_id": 101, "stat": "pts", "provider": "ESPN BET",
         "snapshot": "close", "line": 13.5, "over_odds": -110.0, "under_odds": -110.0},
        {"cbbd_game_id": 902, "athlete_id": 102, "stat": "pts", "provider": "ESPN BET",
         "snapshot": "close", "line": 7.5, "over_odds": -115.0, "under_odds": -105.0},
    ])
    # schema is valid against the documented contract
    assert P.validate_props_lines(props_lines, strict=False) == []

    actual_stats = pd.DataFrame([
        {"cbbd_game_id": 901, "athlete_id": 101, "stat": "pts", "value": 16.0},   # went OVER 13.5
        {"cbbd_game_id": 902, "athlete_id": 102, "stat": "pts", "value": 6.0},    # went UNDER 7.5
    ])

    res = P.grade_props(players, props_lines, actual_games, actual_stats, TOL, settle="book", n_boot=200)
    assert res.n == 2
    assert res.settle_mode == "book"
    assert 0.0 <= res.push_rate <= 1.0
    assert np.isfinite(res.model_brier)
    assert np.isfinite(res.market_brier)
    assert len(res.by_stat) == 1  # only "pts" is present in this fixture
    assert res.by_stat.iloc[0]["stat"] == "pts"


def test_props_lines_schema_rejects_unknown_stat():
    bad = pd.DataFrame([{
        "cbbd_game_id": 1, "athlete_id": 1, "stat": "blocks",  # not in PROP_STATS
        "provider": "ESPN BET", "snapshot": "close", "line": 1.5, "over_odds": -110.0, "under_odds": -110.0,
    }])
    problems = P.validate_props_lines(bad, strict=False)
    assert any("blocks" in p for p in problems)
    with pytest.raises(P.PropsContractError):
        P.validate_props_lines(bad, strict=True)
