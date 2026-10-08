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
