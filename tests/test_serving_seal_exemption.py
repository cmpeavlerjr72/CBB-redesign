"""Serving-path seal exemption (2026-10-08): live 2027 serving may read sealed 2025-26 prior tables; training may not."""
import pytest

from cbb_sim.data import seal as S


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(S.UNSEAL_ENV_VAR, raising=False)


def test_serving_2027_reads_prior_season_while_sealed(capsys):
    S.assert_not_sealed_serving([2022, 2023, 2024, 2025, 2026, 2027], 2027, context="t")
    assert "EXEMPTION USED" in capsys.readouterr().out


def test_serving_exemption_not_for_other_seasons():
    with pytest.raises(S.SealedSeasonError):
        S.assert_not_sealed_serving([2025, 2026], 2026, context="t")


def test_generic_loader_and_trainer_style_reads_stay_sealed():
    with pytest.raises(S.SealedSeasonError):
        S.assert_not_sealed([2024, 2025, 2026], context="trainer")
    from cbb_sim.eval import reference
    with pytest.raises(S.SealedSeasonError):
        reference.assert_not_sealed(2026, context="eval")


def test_exemption_does_not_set_env():
    import os
    S.assert_not_sealed_serving([2026, 2027], 2027)
    assert os.environ.get(S.UNSEAL_ENV_VAR) is None
    with pytest.raises(S.SealedSeasonError):
        S.assert_not_sealed([2026])


def test_daily_ratings_stage_2027_runs_while_sealed(tmp_path, monkeypatch, capsys):
    """The daily day-1 ratings stage for 2027 runs with the flag false and no CBB_UNSEAL; a training-style call still raises."""
    import json, sys
    from pathlib import Path
    from types import SimpleNamespace
    import pandas as pd
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo / "scripts")]
    import chain_day1_2027_v1 as DAY1
    import build_own_ratings_asof_v1 as E
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seal_lift_approved": False, "teams_source": "cbbd", "prior_weight_policy": "manifest_uniform",
                             "new_team_prior": "league_mean", "fixed_term_prior": "previous_final", "early_season_d1_rule": "provisional_union"}))
    seen = {}

    def fake(season, as_of, root, src, chain_start, cache, serving=False):
        seen["serving"] = serving
        E.assert_not_sealed_serving(list(range(2026, season + 1)), season, context="t") if serving else E.assert_not_sealed([2026])
        return pd.DataFrame({"team_id": [1], "x": [0.0]}), {}
    monkeypatch.setattr(E, "asof_ratings", fake)
    ctx = SimpleNamespace(slate_date=pd.Timestamp("2026-11-02").date(), dry_run=True, state={})
    r = DAY1.stage_ratings(ctx, None, None, season=2027, choices_path=p, root=tmp_path)
    assert r.get("_status") != "blocked" and seen["serving"] is True
    assert "EXEMPTION USED" in capsys.readouterr().out
    import os
    assert os.environ.get(S.UNSEAL_ENV_VAR) is None


def test_training_style_asof_ratings_and_day1_tables_still_raise():
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo / "scripts")]
    import build_own_ratings_asof_v1 as E
    import build_engine_inputs_day1prior_v1 as D
    with pytest.raises(S.SealedSeasonError):
        E.asof_ratings(2027, "2026-11-02", repo, "tg", 2022, {})                 # serving defaults to False
    with pytest.raises(S.SealedSeasonError):
        D.tables(2027, True)                                                       # serving defaults to False
