import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import chain_daily_v2 as C  # noqa: E402


def test_choices_missing_reported_and_seal_env_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CHOICES_PATH", tmp_path / "none.json")
    have, miss = C._missing_choices()
    assert set(miss) == set(C.DAY1_CHOICES)
    import os
    os.environ.pop("CBB_UNSEAL", None)
    with C.unsealed():
        assert os.environ["CBB_UNSEAL"] == "1"
    assert "CBB_UNSEAL" not in os.environ


def test_stage_isolation():
    res = []
    C.run_stage("boom", lambda: 1 / 0, res)
    C.run_stage("next", lambda: {"x": 1}, res)
    assert res[0].status == "error" and res[1].status == "ok"
