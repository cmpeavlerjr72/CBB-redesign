"""Seal-week builders (R9ao3 team prior, shot-block prior) and the A3 day-1 seed wiring in the chain inputs stage."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from cbb_sim.data.seal import SealedSeasonError  # noqa: E402

AO = ROOT / "data/processed/models/possession_outcome/round9/ao_team_prior_v1.parquet"
SB_META = ROOT / "data/processed/models/engine/shot_block_lut_F2_2025.meta.json"


@pytest.mark.skipif(not AO.exists(), reason="served R9ao3 table not on disk")
def test_ao_prior_reproduces_stored_rows_exactly():
    import build_ao_team_prior_v2 as M
    rep = M.validate("stored")
    assert set(rep) == {2025, 2026}
    assert all(r["bit_identical"] for r in rep.values())


def test_ao_prior_2027_hits_the_seal_before_any_read(monkeypatch):
    monkeypatch.delenv("CBB_UNSEAL", raising=False)
    import build_ao_team_prior_v2 as M
    with pytest.raises(SealedSeasonError):
        M.build_season(2027)


def test_ao_prior_refuses_to_overwrite_v1():
    import build_ao_team_prior_v2 as M
    with pytest.raises(SystemExit):
        M.main(["--season", "2025", "--out", str(M.V1)])


@pytest.mark.skipif(not SB_META.exists(), reason="shot_block meta not on disk")
def test_shot_block_prior_reproduces_stored_2025_prior_exactly():
    import build_shot_block_prior_v1 as M
    rep = M.validate("stored")
    assert set(rep) == {"rim", "jump2", "three"} and all(r["bit_identical"] for r in rep.values())


def test_shot_block_prior_2027_hits_the_seal(monkeypatch):
    monkeypatch.delenv("CBB_UNSEAL", raising=False)
    import build_shot_block_prior_v1 as M
    with pytest.raises(SealedSeasonError):
        M.build_prior(2027)


def test_day1_missing_lists_both_sources(tmp_path):
    import chain_daily_v2 as C2
    assert len(C2.day1_player_prior_missing(2027, repo=tmp_path)) == 2
    (tmp_path / "data/raw/cbbd/rosters").mkdir(parents=True)
    (tmp_path / "data/raw/cbbd/rosters/roster_2027.parquet").write_bytes(b"x")
    assert len(C2.day1_player_prior_missing(2027, repo=tmp_path)) == 1


def _ctx():
    return SimpleNamespace(slate_date=pd.Timestamp("2026-11-02").date(), dry_run=False, state={"ratings_dir": "x"}, today=None)


SLATE = pd.DataFrame({"game_id": [1], "season": [2027], "game_date": [pd.Timestamp("2026-11-02")]})


def test_live_inputs_hard_stop_when_day1_table_missing(monkeypatch):
    import chain_daily_v2 as C2
    monkeypatch.setattr(C2, "missing_sealed_priors", lambda season: [])
    monkeypatch.setattr(C2, "day1_player_prior_missing", lambda season, repo=C2.REPO: ["A3 roster table missing"])

    class NoBuild:
        def build_live(self, *a, **k):
            raise AssertionError("must not build")
    r = C2.build_live_inputs(_ctx(), NoBuild(), SLATE, {})
    assert r["_status"] == "blocked" and any("A3" in m for m in r["blocked_on"])


def test_live_inputs_passes_the_a3_seed_fn_to_build_live(monkeypatch):
    import chain_daily_v2 as C2
    monkeypatch.setattr(C2, "missing_sealed_priors", lambda season: [])
    monkeypatch.setattr(C2, "day1_player_prior_missing", lambda season, repo=C2.REPO: [])
    monkeypatch.setattr(C2, "load_availability", lambda ctx: None)
    sentinel = object()
    monkeypatch.setattr(C2, "day1_player_prior_seed", lambda season, repo=C2.REPO: (SimpleNamespace(post=lambda i, f: None), sentinel))
    seen = {}

    class BL:
        def build_live(self, *a, **k):
            seen.update(k)
            raise StopIteration

    with pytest.raises(StopIteration):
        C2.build_live_inputs(_ctx(), BL(), SLATE, {})
    assert seen["seed_fn"] is sentinel


def test_build_live_default_seed_fn_is_none():
    import inspect

    import build_engine_inputs_live as BL
    assert inspect.signature(BL.build_live).parameters["seed_fn"].default is None
