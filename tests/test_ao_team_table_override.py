import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cbb_sim.engine import foul_r9 as F


def _setup(tmp_path, monkeypatch, with_override, table_exists=True):
    lut = tmp_path / "lut"
    lut.mkdir()
    if table_exists:
        (lut / "ao_team_prior_v2.parquet").write_bytes(b"x")
    monkeypatch.setattr(F, "LUT_DIR", lut)
    fake_root = tmp_path / "repo"
    (fake_root / "data" / "overrides").mkdir(parents=True)
    if with_override:
        (fake_root / "data" / "overrides" / "ao_team_table.json").write_text(json.dumps({"team_table": "ao_team_prior_v2.parquet"}))
    fake_file = fake_root / "src" / "cbb_sim" / "engine" / "foul_r9.py"
    monkeypatch.setattr(F, "__file__", str(fake_file))


def test_override_present_uses_v2(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, True)
    assert F.served_team_table("ao_team_prior_v1.parquet") == "ao_team_prior_v2.parquet"


def test_override_absent_uses_npz_default(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, False)
    assert F.served_team_table("ao_team_prior_v1.parquet") == "ao_team_prior_v1.parquet"


def test_override_table_missing_in_lut_dir_falls_back(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, True, table_exists=False)
    assert F.served_team_table("ao_team_prior_v1.parquet") == "ao_team_prior_v1.parquet"


def test_repo_override_serves_v2_with_2027():
    assert F.served_team_table("ao_team_prior_v1.parquet") == "ao_team_prior_v2.parquet"
    assert 2027 in F.table_seasons("R9ao3")
