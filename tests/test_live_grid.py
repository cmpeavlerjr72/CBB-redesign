"""Tests for src/cbb_sim/live_grid.py and scripts/export_live_grid_v1.py (game-level live.json).

Synthetic 3-game trajectory fixture (no engine): G1 balanced random walk, G2 heavy favourite (forces thin tail cells), G3 with forced
overtime seeds. Pins the live.json contract against (a) the CFB key tree (when the football checkout is present) and (b) the
basketball keys the site client reads (monte_site 6d2de56 `atStateBb`)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import export_live_grid_v1 as XL  # noqa: E402
from cbb_sim import live_grid as LG  # noqa: E402

CFB_LIVE = Path(r"C:\Users\devuser\cfb-props-sim\data\processed\site_stage\cfb-sims-2026\2026\weeks\week10\games\kansasstate_oklahomastate\live.json")
CFB_TOP = {"convention", "game_id", "slug", "teamA", "teamB", "n_sims", "updated", "source_tag", "orientation", "grid", "remaining", "uncond"}
CFB_GRID = {"time_bin_s", "n_time", "margin_bin", "margin_min", "margin_max", "n_margin", "min_n", "margin_rule", "p_home_win",
            "mean_total", "mean_margin", "n"}
CFB_REMAINING = {"time_bin_s", "n_time", "margin_edges", "n_margin", "min_n", "rule", "n", "margin_sf", "total_sf", "time_margin_sf",
                 "time_total_sf"}
CFB_ORIENT = {"engine_home", "schedule_home", "flipped", "traj_vs_games"}
CBB_TOP_EXTRA = {"clock", "live_arm", "gates", "note"}


def make_traj(game_id: int, n_seeds: int, rate_home: float, rate_away: float, seed0: int, force_ot: int = 0) -> pd.DataFrame:
    """Random-walk game: 2 x 1200 s, ~66 possessions a half, Poisson points per possession; force_ot seeds end level and play OT."""
    rng = np.random.default_rng(game_id + seed0)
    rows = []
    for s in range(n_seeds):
        h = a = 0
        poss = 0
        for period in (1, 2):
            clock = 1200
            while clock > 0:
                side = poss % 2
                pts = int(rng.poisson(rate_home if side == 0 else rate_away) * 1.0)
                rows.append((game_id, seed0 + s, poss, period, clock, h + pts * (side == 0), a + pts * (side == 1)))
                h += pts * (side == 0)
                a += pts * (side == 1)
                clock -= int(rng.integers(12, 25))
                poss += 1
        if s < force_ot:                             # level at the horn + OT points
            diff = h - a
            if diff > 0:
                a += diff
            else:
                h += -diff
            rows.append((game_id, seed0 + s, poss, 2, 1, h, a))       # a no-op row so regulation ends level in the data
            clock = 300
            while clock > 0:
                side = poss % 2
                pts = int(rng.poisson(1.0))
                h += pts * (side == 0)
                a += pts * (side == 1)
                rows.append((game_id, seed0 + s, poss + 1, 3, clock, h, a))
                clock -= 25
                poss += 1
            if h == a:
                h += 1
                rows[-1] = (*rows[-1][:5], h, a)
    return pd.DataFrame(rows, columns=["game_id", "seed", "poss_idx", "period", "clock_start", "home_score", "away_score"])


@pytest.fixture(scope="module")
def traj() -> pd.DataFrame:
    return pd.concat([make_traj(1, 300, 0.95, 0.95, 0, force_ot=0), make_traj(2, 300, 1.25, 0.8, 0),
                      make_traj(3, 300, 0.95, 0.95, 0, force_ot=30)], ignore_index=True)


@pytest.fixture(scope="module")
def gs_all(traj):
    return {gid: LG.prepare_game(g) for gid, g in traj.groupby("game_id")}


def _game(gid):
    return {"game_id": gid, "slug": f"home{gid}_away{gid}", "teamA": f"Home{gid}", "teamB": f"Away{gid}"}


def _games_parquet(gs):
    h = ((gs.fm + gs.ft) // 2).astype(int)
    return pd.DataFrame({"game_id": gs.game_id, "seed": gs.seeds, "home_pts": h, "away_pts": gs.ft - h})


@pytest.fixture(scope="module")
def docs(gs_all):
    return {gid: XL.build_live(gs, _game(gid), "C_row", "t", "2026-11-02T00:00:00+00:00", _games_parquet(gs)) for gid, gs in gs_all.items()}


# ----------------------------------------------------------------------------------------------------------------------
def test_prepare_game_labels_and_regulation_state(gs_all):
    gs = gs_all[1]
    assert gs.S == 300 and gs.E[:, 0].max() == 0.0                       # every seed's first checkpoint is elapsed 0
    assert (gs.reg_t <= gs.ft).all() and (gs.ft - gs.reg_t >= 0).all()
    gs3 = gs_all[3]
    assert gs3.ot.sum() == 30 and (gs3.reg_m[gs3.ot] == 0).all()


def test_contract_cfb_key_tree_and_cbb_additions(docs):
    d = docs[1]
    assert set(d) == CFB_TOP | CBB_TOP_EXTRA
    assert set(d["grid"]) == CFB_GRID and set(d["orientation"]) == CFB_ORIENT
    assert set(d["uncond"]) == {"p_home_win", "mean_total", "mean_margin"}
    assert CFB_REMAINING <= set(d["remaining"])
    assert d["grid"]["n_time"] == 20 and len(d["grid"]["p_home_win"]) == 20 and len(d["grid"]["p_home_win"][0]) == 29
    assert d["grid"]["margin_min"] == -28 and d["grid"]["margin_max"] == 28 and d["grid"]["margin_bin"] == 2
    assert d["orientation"] == {"engine_home": "Home1", "schedule_home": "Home1", "flipped": False, "traj_vs_games": "direct"}
    json.dumps(d)                                                         # serialisable, no NaN


@pytest.mark.skipif(not CFB_LIVE.exists(), reason="football checkout not on this machine")
def test_key_tree_matches_the_live_cfb_reference_file(docs):
    ref = json.loads(CFB_LIVE.read_text())
    d = docs[1]
    assert CFB_TOP <= set(ref) and set(d) >= CFB_TOP
    assert set(ref["grid"]) == set(d["grid"]) and set(ref["uncond"]) == set(d["uncond"]) and set(ref["orientation"]) == set(d["orientation"])
    assert set(ref["remaining"]) <= set(d["remaining"])
    assert {type(v).__name__ for v in ref["grid"].values()} == {type(v).__name__ for v in d["grid"].values()}
    c = ref["remaining"]["margin_sf"][3][3]
    assert set(c) == {"k0", "v"} and set(d["remaining"]["time_margin_sf"][3]) == {"k0", "v"}


def test_client_contract_pinned_basketball_reading(docs):
    """What monte_site `atStateBb` / `parseLiveGrid` read (commit 6d2de56). A mismatch here = the client silently prices nothing."""
    d = docs[2]
    assert d["clock"] == {"regulation_s": 2400, "period_s": 1200, "n_periods": 2, "ot_period_s": 300}
    r = d["remaining"]
    assert r["time_bin_s"] == 120 and r["n_time"] == 20 and len(r["margin_edges"]) == 6 and r["n_margin"] == 7
    assert r["margin_edges"] == [-11, -6, -2, 3, 7, 12]
    assert len(r["total_edges"]) == 20 and all(len(e) == 4 and e == sorted(e) and all(isinstance(x, int) for x in e) for e in r["total_edges"])
    assert len(r["total_sf"]) == 20 and all(len(x) == 5 for x in r["total_sf"])             # keyed by total-so-far bin (5), not margin
    assert len(r["margin_sf"]) == 20 and all(len(x) == 7 for x in r["margin_sf"])
    assert len(r["time_margin_sf"]) == 20 and len(r["time_total_sf"]) == 20
    L = r["late"]
    assert (L["start_s"], L["time_bin_s"], L["n_time"]) == (2160, 30, 8)
    assert L["margin_edges"] == [-16, -10, -6, -3, -1, 2, 4, 7, 11, 17]
    assert len(L["margin_sf"]) == 8 and all(len(x) == 11 for x in L["margin_sf"]) and all(len(x) == 11 for x in L["total_sf"])
    assert set(r["end"]) >= {"tied_margin_sf", "tied_total_sf"}
    assert r["end"]["tied_margin_sf"] is None and r["end"]["n_tied"] < 3          # no tied seeds -> null, never invented
    e3 = docs[3]["remaining"]["end"]
    for c in [r["time_margin_sf"][0], r["time_total_sf"][5], e3["tied_margin_sf"], e3["tied_total_sf"]]:
        assert set(c) == {"k0", "v"} and c["v"][0] == 1000 and c["v"][-1] == 0 and all(a >= b for a, b in zip(c["v"], c["v"][1:]))
    # counts exposed so the client's thin-cell rule can keep applying
    assert len(r["n"]) == 20 and len(L["n"]) == 8 and len(r["total_n"]) == 20


def test_bins_sum_to_the_seed_count(docs, gs_all):
    for gid, d in docs.items():
        S = gs_all[gid].S
        r = d["remaining"]
        assert all(sum(row) == S for row in r["n"]) and all(sum(row) == S for row in r["total_n"])
        assert all(sum(row) == S for row in r["late"]["n"])
        assert d["n_sims"] == S


def test_thin_cells_are_null_in_row_mode_and_pooled_not_null_in_pool_mode(gs_all):
    gs = gs_all[2]
    row = XL.build_live(gs, _game(2), "C_row", "t", "u")
    pool = XL.build_live(gs, _game(2), "C_pool", "t", "u")
    thin_cells = [(t, k) for t, ns in enumerate(row["remaining"]["n"]) for k, n in enumerate(ns) if n < row["remaining"]["min_n"]]
    assert thin_cells, "heavy-favourite fixture must produce thin tail cells"
    assert all(row["remaining"]["margin_sf"][t][k] is None for t, k in thin_cells)
    assert all(pool["remaining"]["margin_sf"][t][k] is not None for t, k in thin_cells)
    assert any(w > 0 for ws in pool["remaining"]["widened"] for w in ws)


def test_curve_round_trip_mean_and_survival(gs_all):
    gs = gs_all[1]
    tab = LG.build_tables(gs, LG.ARMS["C_row"], LG.scaled_min_n(gs.S))
    r = tab.rows[7]
    pmf = r.rm_row
    c = XL.curve(pmf, LG.RM_OFF)
    v = np.array(c["v"]) / 1000.0
    mean_curve = c["k0"] + v[1:].sum()
    mean_pmf = (pmf * (np.arange(LG.RM_N) - LG.RM_OFF)).sum()
    assert abs(mean_curve - mean_pmf) < 0.06                              # per-mille rounding only
    k = np.arange(len(v)) + c["k0"]
    ref = np.array([pmf[(np.arange(LG.RM_N) - LG.RM_OFF) >= kk].sum() for kk in k])
    assert np.abs(ref - v).max() < 0.0006


def test_end_of_regulation_row_is_exact(gs_all, docs):
    gs = gs_all[3]
    tied = gs.reg_m == 0
    e = docs[3]["remaining"]["end"]
    assert e["n_tied"] == int(tied.sum()) >= 30
    ot_pts = (gs.ft[tied] - gs.reg_t[tied])
    c = e["tied_total_sf"]
    for j, kk in enumerate(range(c["k0"], c["k0"] + len(c["v"]))):
        assert abs(c["v"][j] / 1000 - (ot_pts >= kk).mean()) < 0.0006
    tab = LG.build_tables(gs, LG.ARMS["C_row"], LG.scaled_min_n(gs.S))
    rm, rt, _, _ = LG.price(tab, 2400.0, np.array([5, -5, 0]), np.array([150, 150, 150]))
    assert rm[0, LG.RM_OFF] == 1.0 and rm[1, LG.RM_OFF] == 1.0 and rt[0, 0] == 1.0     # decided at the horn: exactly nothing more
    assert abs((rt[2] * np.arange(LG.RT_N)).sum() - ot_pts.mean()) < 1e-9
    wp = LG.win_prob(rm, np.array([5, -5, 0]))
    assert wp[0] == LG.PMAX and wp[1] == LG.PMIN                                        # 1 / 0 at per-mille resolution, no fade


def test_g1_balanced_grid_and_win_probability_calibration(gs_all):
    """Late-game win probability by margin is ordered and (balanced game) near the normal-approximation sign structure."""
    gs = gs_all[1]
    tab = LG.build_tables(gs, LG.ARMS["C_row"], LG.scaled_min_n(gs.S))
    m = np.array([-12, -6, 0, 6, 12])
    rm, rt, _, _ = LG.price(tab, 2200.0, m, np.full(5, 100))
    p = LG.win_prob(rm, m)
    assert (np.diff(p) > 0).all() and p[2] > 0.3 and p[2] < 0.7 and p[4] > 0.9 and p[0] < 0.1
    # grid vs a naive recount (independent of the vectorised builder) on an n >= 50 cell
    g = XL.build_grid(gs, 3)
    tb, mb = 10, 14                                                                     # elapsed 1200-1319, tied bucket
    wins, tots, n = 0, 0, 0
    for i in range(gs.S):
        hit = False
        for j in range(gs.E.shape[1]):
            e = gs.E[i, j]
            if not np.isfinite(e):
                break
            if int(e // 120) == tb and int(XL.grid_bucket(np.array([gs.Mg[i, j]]))[0]) == mb:
                hit = True
        if hit:
            n += 1
            wins += int(gs.fm[i] > 0)
            tots += int(gs.ft[i])
    assert n >= 20 and g["n"][tb][mb] == n
    assert abs(g["p_home_win"][tb][mb] - wins / n) < 1e-4 and abs(g["mean_total"][tb][mb] - tots / n) < 0.006


def test_grid_bucket_rule_half_away_from_zero():
    b = XL.grid_bucket(np.array([0, 1, 2, 3, 4, -1, -2, -3, 27, 28, 29, 100, -28, -29, -100]))
    assert b.tolist() == [14, 15, 15, 16, 16, 13, 13, 12, 28, 28, 28, 28, 0, 0, 0]


def test_deterministic_output(gs_all):
    a = XL.dumps(XL.build_live(gs_all[1], _game(1), "C_row", "t", "u"))
    b = XL.dumps(XL.build_live(gs_all[1], _game(1), "C_row", "t", "u"))
    assert a == b


def test_orientation_mismatch_is_refused(gs_all):
    gs = gs_all[1]
    g = _games_parquet(gs)
    g = g.rename(columns={"home_pts": "away_pts", "away_pts": "home_pts"})
    with pytest.raises(ValueError, match="orientation or run mismatch"):
        XL.verify_vs_games(gs, g)
    with pytest.raises(ValueError, match="absent"):
        XL.verify_vs_games(gs, _games_parquet(gs).iloc[:10])


def test_arms_other_than_the_default_are_selectable_and_shape_correct(gs_all):
    d = XL.build_live(gs_all[1], _game(1), "A_row", "t", "u")
    assert "late" not in d["remaining"] and "total_edges" not in d["remaining"] and len(d["remaining"]["total_sf"][0]) == 7
    d = XL.build_live(gs_all[1], _game(1), "B_pool", "t", "u")
    assert "late" not in d["remaining"] and len(d["remaining"]["total_edges"][0]) == 4


def test_table_build_never_reads_test_seeds(gs_all):
    """Tables from a TRAIN subset are unchanged when the TEST seeds' data is scrambled (the grader's leak guard)."""
    gs = gs_all[1]
    tr = np.arange(0, 200)
    a = LG.build_tables(gs.take(tr), LG.ARMS["C_row"], 8)
    g2 = gs.take(np.arange(0, 300))
    g2.fm[200:] = 999
    g2.ft[200:] = 999
    b = LG.build_tables(g2.take(tr), LG.ARMS["C_row"], 8)
    assert all(np.array_equal(x.rm, y.rm) for x, y in zip(a.rows, b.rows))
    assert np.array_equal(a.end.ot_rt, b.end.ot_rt) if a.end.ot_rt is not None else True


def test_default_arm_is_the_bakeoff_winner_and_client_compatible(gs_all):
    """D_row (late block, margin-keyed totals): no total_edges, so the client's `else` branch reads total_sf[tb][margin bin]."""
    assert XL.DEFAULT_ARM == "D_row"
    d = XL.build_live(gs_all[1], _game(1), None or XL.DEFAULT_ARM, "t", "u")
    r = d["remaining"]
    assert "total_edges" not in r and len(r["total_sf"]) == 20 and all(len(x) == 7 for x in r["total_sf"])
    assert r["late"]["n_time"] == 8 and set(r["end"]) >= {"tied_margin_sf", "tied_total_sf"}
    assert d["clock"]["regulation_s"] == 2400 and d["live_arm"] == "D_row"
