"""ops_seal_week_v1: refusal without the seal flag, static stage checks, no flag writes."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import ops_seal_week_v1 as O  # noqa: E402


def test_stage_order_and_names():
    names = [s.name for s in O.stages("2026-11-02", "2026-11-02", "2026-11-15")]
    # 2026-10-07 (served stack v3): the parity v10 smoke is stage 1, before the preflight tests
    assert names[0] == "parity_v10" and names[1] == "preflight" and names[-1] == "reseal"
    assert names.index("clock_serving") < names.index("chain_dry_run")
    for need in ("rosters", "a3_day1_priors", "ratings", "tip_times", "sim_4seed"):
        assert need in names
    assert names.index("rosters") < names.index("a3_day1_priors") < names.index("sim_4seed")


def test_every_script_stage_args_documented_in_source():
    for s in O.stages("2026-11-02", "2026-11-02", "2026-11-15"):
        if s.script:
            assert O.source_ok(s.script, s.tokens) is None, s.name


def test_seal_flag_reader(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seal_lift_approved": False}))
    assert O.seal_approved(p) is False
    p.write_text(json.dumps({"seal_lift_approved": "true"}))
    assert O.seal_approved(p) is False          # only the JSON boolean true counts
    p.write_text(json.dumps({"seal_lift_approved": True}))
    assert O.seal_approved(p) is True
    assert O.seal_approved(tmp_path / "missing.json") is False


def test_execute_refused_without_flag_and_flag_untouched(monkeypatch, tmp_path, capsys):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seal_lift_approved": False}))
    monkeypatch.setattr(O, "CHOICES", p)
    monkeypatch.setattr(O, "seal_approved", lambda: False)
    before = p.read_text()
    assert O.main(["--execute"]) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert p.read_text() == before


def test_the_script_never_writes_the_flag():
    src = (REPO / "scripts/ops_seal_week_v1.py").read_text(encoding="utf-8")
    assert "write_text" not in src and "json.dump(" not in src
