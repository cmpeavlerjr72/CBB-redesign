"""Day-1 (season 2027) pieces from lane F2: tip-time refresh logic, evening / morning passes, carried-forward artifacts, ratings choices."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cbb_sim.live import tips as TP  # noqa: E402

ET = "America/New_York"


def _cbbd(rows):
    return pd.DataFrame([{"sourceId": g, "id": g + 1000, "startDate": t, "startTimeTbd": tbd} for g, t, tbd in rows])


def _hoopr(rows):
    return pd.DataFrame([{"game_id": g, "date": t, "time_valid": v} for g, t, v in rows])


def test_merge_tips_sources_and_placeholder():
    cb = TP.cbbd_tip_frame(_cbbd([(1, "2026-11-02T05:00:00Z", True),     # TBD placeholder, hoopR real
                                  (2, "2026-11-02T05:00:00Z", True),     # TBD, hoopR also placeholder
                                  (3, "2026-11-02T23:00:00Z", False),    # CBBD real only
                                  (4, "2026-11-03T00:00:00Z", False),    # both real, disagree 30 min
                                  (5, "2026-11-02T05:00:00Z", False)]))  # flagged not-TBD but exactly midnight ET: still placeholder
    ho = TP.hoopr_tip_frame(_hoopr([(1, "2026-11-02T19:00Z", True), (2, "2026-11-02T05:00Z", False),
                                    (4, "2026-11-03T00:30Z", True), (5, "2026-11-02T05:00Z", True)]))
    t = TP.merge_tips(cb, ho, "2026-10-01T00:00:00Z").set_index("game_id")
    assert t.loc[1, "tip_source"] == "hoopr_schedule" and not t.loc[1, "tip_time_is_placeholder"]
    assert t.loc[1, "tipoff_utc"] == pd.Timestamp("2026-11-02T19:00:00Z")
    assert t.loc[2, "tip_time_is_placeholder"] and t.loc[2, "tip_source"] == "cbbd_placeholder_midnight_et"
    assert t.loc[3, "tip_source"] == "cbbd" and not t.loc[3, "tip_time_is_placeholder"]
    assert t.loc[4, "tip_source"] == "cbbd+hoopr_min" and t.loc[4, "tipoff_utc"] == pd.Timestamp("2026-11-03T00:00:00Z")   # EARLIER wins
    assert t.loc[4, "tip_disagree_min"] == pytest.approx(30.0)
    assert t.loc[5, "tip_time_is_placeholder"]


def _slate():
    return pd.DataFrame({"game_id": [1, 2, 3],
                         "tipoff_utc": pd.to_datetime(["2026-11-02T19:00Z", "2026-11-02T05:00Z", "2026-11-02T23:00Z"], utc=True),
                         "tip_time_is_placeholder": [False, True, False]})


def test_evening_pass_keeps_placeholders_morning_pass_does_not():
    ev_clock = TP.default_clock("2026-11-02", "evening")
    ok, late = TP.select_for_pass(_slate(), ev_clock, "evening")
    assert len(ok) == 3 and len(late) == 0
    ok, late = TP.select_for_pass(_slate(), ev_clock, "morning")
    assert sorted(ok["game_id"]) == [1, 3] and late["game_id"].tolist() == [2]
    assert late["refuse_reason"].iloc[0].startswith("placeholder")


def test_morning_pass_real_and_future_only_and_guard_not_loosened():
    m_clock = TP.default_clock("2026-11-02", "morning")        # 09:00 ET = 14:00Z: the placeholder (05:00Z) is "in the past"
    ok, late = TP.select_for_pass(_slate(), m_clock, "evening")
    assert sorted(ok["game_id"]) == [1, 3] and late["game_id"].tolist() == [2]       # evening-style filter still refuses on the guard
    ok, late = TP.select_for_pass(_slate(), pd.Timestamp("2026-11-02T20:00Z"), "morning")
    assert ok["game_id"].tolist() == [3]                                                # game 1 tipped at 19:00Z
    from cbb_sim.live import guards as G
    with pytest.raises(G.LeakGuardError):                                                # a row at/after tip still raises
        G.assert_created_before_tipoff(_slate().assign(created_at=pd.Timestamp("2026-11-02T20:00Z")))


def test_no_tip_time_is_refused():
    s = _slate()
    s.loc[0, "tipoff_utc"] = pd.NaT
    ok, late = TP.select_for_pass(s, TP.default_clock("2026-11-02", "evening"), "evening")
    assert 1 in late["game_id"].tolist()


def test_default_clocks():
    assert TP.default_clock("2026-11-02", "evening") == pd.Timestamp("2026-11-02T01:00:00Z")    # 20:00 EST Nov 1 (DST ended Nov 1)
    assert TP.default_clock("2026-11-02", "morning") == pd.Timestamp("2026-11-02T14:00:00Z")


def test_carried_forward_artifacts_exist_and_are_marked():
    eng = REPO / "data/processed/models/engine"
    idx_p = eng / "event_round2_s1_F2_2027/index.json"
    if not idx_p.exists():
        pytest.skip("run scripts/build_season_2027_artifacts_v1.py")
    idx = json.loads(idx_p.read_text(encoding="utf-8"))
    assert idx["season"] == 2027 and "CARRIED FORWARD" in idx["carried_forward_from"]["note"]
    for pop in idx["populations"].values():
        for sg in pop["segments"]:
            assert (idx_p.parent / sg["file"]).resolve().exists()
    a = json.loads((eng / "names_F2_2025_v2.json").read_text(encoding="utf-8"))
    b = json.loads((eng / "names_F2_2027_v2.json").read_text(encoding="utf-8"))
    assert a["rules"] == b["rules"] and a["team_names"] == b["team_names"] and a["slot_names"] == b["slot_names"]
    assert b["meta"]["season"] == 2027 and "UNVERIFIED" in b["meta"]["rule_constants_status"]


def test_bonus_era_2027_row_explicit_and_old_rows_unchanged():
    from cbb_sim.engine import state as ST
    era = ST.load_bonus_era()
    if 2027 not in era:
        pytest.skip("run scripts/build_season_2027_artifacts_v1.py")
    assert era[2027] == era[2026] == (6, 9)
    assert all(era[s] == (6, 9) for s in (2022, 2023, 2024, 2025, 2026))
    raw = json.loads((REPO / "data/processed/models/free_throw/bonus_era.json").read_text(encoding="utf-8"))
    assert "CARRIED FORWARD" in raw["by_season"]["2027"]["derived_from"]


def test_choices_file_gates_on_seal_and_policy(tmp_path):
    import chain_day1_2027_v1 as DAY1
    good = {"seal_lift_approved": True, "teams_source": "cbbd", "prior_weight_policy": "manifest_uniform",
            "new_team_prior": "league_mean", "fixed_term_prior": "previous_final", "early_season_d1_rule": "provisional_union"}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(good))
    assert DAY1.check_choices(p, 2027)[1] == []
    p.write_text(json.dumps({**good, "prior_weight_policy": "arm_C_conference"}))
    assert DAY1.check_choices(p, 2027)[1] == []
    p.write_text(json.dumps({**good, "prior_weight_policy": "nonsense", "seal_lift_approved": False}))
    assert set(DAY1.check_choices(p, 2027)[1]) == {"prior_weight_policy", "seal_lift_approved"}
    assert DAY1.check_choices(p, 2025)[1] == ["prior_weight_policy"]                    # a pre-2026 rehearsal does not need the seal flag


def test_shipped_choices_file_defaults_are_served_and_seal_not_lifted():
    import chain_day1_2027_v1 as DAY1
    j = json.loads(DAY1.CHOICES_PATH.read_text(encoding="utf-8"))
    st = j["_status"]
    assert set(st) == set(DAY1.OPTIONS) and len(st) == 6
    assert all(v["status"] in ("SERVED DEFAULT", "PROPOSED") for v in st.values())
    assert j["prior_weight_policy"] == "manifest_uniform" and st["prior_weight_policy"]["status"] == "SERVED DEFAULT"
    assert j["seal_lift_approved"] is False


def test_stage_ratings_blocked_without_seal(tmp_path):
    import chain_day1_2027_v1 as DAY1
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seal_lift_approved": False}))
    ctx = SimpleNamespace(slate_date=pd.Timestamp("2026-11-02").date(), dry_run=True, state={})
    r = DAY1.stage_ratings(ctx, None, None, season=2027, choices_path=p)
    assert r["_status"] == "blocked" and "seal_lift_approved" in r["missing_or_invalid"]
    assert "ratings_dir" not in ctx.state


def test_sim_prereqs_after_artifacts_only_ratings_block_and_rosters_warn():
    import chain_daily_v3 as C3
    if not (REPO / "data/processed/models/engine/names_F2_2027_v2.json").exists():
        pytest.skip("artifacts not built")
    miss = C3.sim_prereqs(2027, "F2", "2026-11-02", None)
    # R9ao3 team-prior hard stop (seal-gated, added after this test was written) is an expected second blocker
    assert len(miss) == 2 and "own ratings" in miss[0] and "R9ao3" in miss[1]
    assert [m for m in C3.sim_prereqs(2027, "F2", "2026-11-02", "x") if "R9ao3" not in m] == []
    warn = C3.sim_warnings(2027)                       # empty rosters today; if CBBD fills them in this branch is vacuous
    if warn:
        assert "PLAYER layer" in warn[0]
        assert any("rosters" in m for m in C3.sim_prereqs(2027, "F2", "2026-11-02", "x", require_rosters=True))
    assert C3.sim_warnings(2026) == []


def test_missing_sealed_priors_2027_hard_stop_names_each_table(tmp_path):
    import chain_daily_v2 as C2
    miss = C2.missing_sealed_priors(2027, repo=tmp_path, r9_seasons=[2023, 2024, 2025, 2026])
    assert len(miss) == 2
    assert any("shot_block_prior_2027_v1.parquet" in m for m in miss) and any("R9ao3" in m for m in miss)
    assert C2.missing_sealed_priors(2025, repo=tmp_path) == []          # only 2027 carries the sealed-prior list


def test_missing_sealed_priors_clear_when_tables_present(tmp_path):
    import chain_daily_v2 as C2
    f = tmp_path / "data/processed/models/engine/shot_block_prior_2027_v1.parquet"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    assert C2.missing_sealed_priors(2027, repo=tmp_path, r9_seasons=[2026, 2027]) == []


def test_build_live_inputs_blocks_without_priors_never_builds(tmp_path):
    import chain_daily_v2 as C2
    slate = pd.DataFrame({"game_id": [1], "season": [2027], "game_date": [pd.Timestamp("2026-11-02")]})
    ctx = SimpleNamespace(slate_date=pd.Timestamp("2026-11-02").date(), dry_run=False, state={})

    class NoBuild:
        def build_live(self, *a, **k):
            raise AssertionError("must not build when a sealed prior table is missing")
    r = C2.build_live_inputs(ctx, NoBuild(), slate, {})
    assert r["_status"] == "blocked" and r["built"] is False and any("own ratings" in m for m in r["blocked_on"])
