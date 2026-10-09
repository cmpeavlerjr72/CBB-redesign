"""chain_daily_v3 `lines_probe` stage: evening pass only, default on, never fatal."""
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import chain_daily_v3 as C  # noqa: E402


def _ctx():
    return SimpleNamespace(today=date(2026, 10, 9), slate_date=date(2026, 11, 2))


def test_morning_pass_skips(monkeypatch):
    monkeypatch.setattr(C, "_run_script", lambda args: (_ for _ in ()).throw(AssertionError("must not run")))
    r = C.stage_lines_probe(_ctx(), SimpleNamespace(pass_name="morning", dry_run=False, lines_probe=True))
    assert r["_status"] == "skipped"


def test_opt_out_skips(monkeypatch):
    monkeypatch.setattr(C, "_run_script", lambda args: (_ for _ in ()).throw(AssertionError("must not run")))
    assert C.stage_lines_probe(_ctx(), SimpleNamespace(pass_name="evening", dry_run=False, lines_probe=False))["_status"] == "skipped"


def test_evening_runs_with_slate_window_and_dry_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(C, "_run_script", lambda args: seen.setdefault("a", args) and {"rc": 0, "out": "ok"})
    r = C.stage_lines_probe(_ctx(), SimpleNamespace(pass_name="evening", dry_run=True, lines_probe=True))
    assert "_status" not in r and r["rc"] == 0                    # run_stage turns a missing _status into "ok"
    a = seen["a"]
    assert a[1:5] == ["--start", "2026-11-02", "--end", "2026-11-09"] and a[-1] == "--dry-run"


def test_failure_is_never_fatal(monkeypatch):
    monkeypatch.setattr(C, "_run_script", lambda args: {"rc": 3, "out": "boom"})
    r = C.stage_lines_probe(_ctx(), SimpleNamespace(pass_name="evening", dry_run=False, lines_probe=True))
    assert r["_status"] == "failed_nonfatal"
    def raiser(args): raise TimeoutError("slow")
    monkeypatch.setattr(C, "_run_script", raiser)
    r = C.stage_lines_probe(_ctx(), SimpleNamespace(pass_name="evening", dry_run=False, lines_probe=True))
    assert r["_status"] == "failed_nonfatal"
    assert C.exit_code([SimpleNamespace(status="failed_nonfatal")]) == 0
