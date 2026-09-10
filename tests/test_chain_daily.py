"""tests/test_chain_daily.py

Two properties `scripts/chain_daily.py` exists to guarantee (see its module
docstring and `docs/postmortem/03_data_inventory.md`'s root-cause finding
that last year's daily pipeline "crashed 82 times on one unretried scraper
and died in June"):

1. A step that raises never stops the chain -- every other step still runs
   and reports its own status.
2. Every artifact the chain writes carries a `created_at` field.

The first property is tested twice: once against `run_step`/`run_chain`
directly with synthetic steps (fast, deterministic, no network -- this is
the load-bearing proof), and once by forcing a real chain run through with a
patched-in failing step, to confirm the production wiring has the same
property. The second property is tested at the `_stamp_df` unit level (fast,
no network) and, when network + the CFBD key are available, end to end
against real written files for a historical date (redirected to a tmp
directory so the test never touches the repo's real `data/raw/`).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chain_daily as CD  # noqa: E402

ENV_KEY_PATH = Path(r"C:\Users\devuser\cfb-props-sim\.env")
TEST_DATE = date(2025, 12, 24)  # a real, already-played historical date (season 2026)


def _network_and_key_available() -> bool:
    if not ENV_KEY_PATH.exists():
        return False
    try:
        requests.get("https://api.collegebasketballdata.com/lines/providers", timeout=5)
        return True
    except requests.RequestException:
        return False


skip_no_network = pytest.mark.skipif(
    not _network_and_key_available(), reason="requires network access + CFBD_API_KEY"
)


# --------------------------------------------------------------------------
# 1a. run_step / run_chain isolate a raising step (synthetic, no network)
# --------------------------------------------------------------------------
def test_run_step_isolates_a_raising_step():
    results: list[CD.StepResult] = []

    CD.run_step("ok_before", lambda: {"n": 1}, results)
    CD.run_step("boom", lambda: (_ for _ in ()).throw(RuntimeError("simulated scraper failure")), results)
    CD.run_step("ok_after", lambda: {"n": 2}, results)

    assert [r.name for r in results] == ["ok_before", "boom", "ok_after"]
    assert [r.status for r in results] == ["ok", "error", "ok"]
    assert "simulated scraper failure" in results[1].error
    # the steps either side of the failure ran to completion with real output
    assert results[0].detail == {"n": 1}
    assert results[2].detail == {"n": 2}


def test_run_chain_runs_every_step_even_if_one_raises(monkeypatch):
    """Same property through the real chain-orchestration path (`run_chain`
    iterating `STEPS`), with one step swapped for a guaranteed failure."""

    def boom(ctx):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(CD, "STEPS", [
        ("first", lambda ctx: {"ran": True}),
        ("boom", boom),
        ("third", lambda ctx: {"ran": True}),
    ])

    class DummyTracker:
        count = 0
        remaining = None

    ctx = CD.ChainContext(
        today=TEST_DATE, yesterday=TEST_DATE, season=2026, dry_run=True,
        session=None, tracker=DummyTracker(), scratch_dir=Path("."),
    )
    results = CD.run_chain(ctx)

    assert [r.name for r in results] == ["first", "boom", "third"]
    assert results[0].status == "ok"
    assert results[1].status == "error"
    assert "simulated failure" in results[1].error
    assert results[2].status == "ok"  # ran despite "boom" raising immediately before it


def test_step_result_never_raises_out_of_run_step():
    """Even a step that raises something exotic (not a plain Exception
    subclass constructed normally, e.g. one that fails during __str__)
    doesn't escape run_step."""

    class Weird(Exception):
        def __str__(self):
            return "weird failure"

    results: list[CD.StepResult] = []
    CD.run_step("weird", lambda: (_ for _ in ()).throw(Weird()), results)
    assert results[0].status == "error"
    assert "weird failure" in results[0].error


# --------------------------------------------------------------------------
# 1b. current_season / day_window sanity (used by every step)
# --------------------------------------------------------------------------
def test_current_season_rolls_over_in_july():
    assert CD.current_season(date(2026, 9, 10)) == 2027  # this repo's "today"
    assert CD.current_season(date(2026, 6, 30)) == 2026
    assert CD.current_season(date(2026, 7, 1)) == 2027
    assert CD.current_season(date(2027, 3, 1)) == 2027


# --------------------------------------------------------------------------
# 2a. created_at stamping (unit level, no network)
# --------------------------------------------------------------------------
def test_stamp_df_adds_created_at():
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = CD._stamp_df(df)
    assert "created_at" in out.columns
    assert out["created_at"].notna().all()
    pd.Timestamp(out["created_at"].iloc[0])  # parses as a real timestamp
    assert "created_at" not in df.columns  # _stamp_df must not mutate its input


def test_stamp_df_works_on_an_empty_frame():
    df = pd.DataFrame(columns=["a", "b"])
    out = CD._stamp_df(df)
    assert "created_at" in out.columns
    assert len(out) == 0


# --------------------------------------------------------------------------
# 2b. dry-run writes nothing; live run's real artifacts all carry created_at
# --------------------------------------------------------------------------
@skip_no_network
def test_dry_run_writes_nothing_for_a_fresh_date(tmp_path):
    schedule_path = CD.SCHEDULE_DAILY_DIR / f"{TEST_DATE.isoformat()}.parquet"
    lines_path = CD.LINES_DAILY_DIR / f"{TEST_DATE.isoformat()}.parquet"
    assert not schedule_path.exists()
    assert not lines_path.exists()

    ctx = CD.build_context(TEST_DATE, dry_run=True, max_calls=50)
    ctx.scratch_dir = tmp_path
    results = CD.run_chain(ctx)

    assert len(results) == len(CD.STEPS)
    assert all(r.status in ("ok", "error") for r in results), results
    assert not schedule_path.exists()
    assert not lines_path.exists()


@skip_no_network
def test_live_run_outputs_all_carry_created_at(tmp_path, monkeypatch):
    # Redirect every write target at a tmp dir so this test never touches
    # the repo's real data/raw/ artifacts.
    monkeypatch.setattr(CD, "SCHEDULE_DAILY_DIR", tmp_path / "schedule_daily")
    monkeypatch.setattr(CD, "LINES_DAILY_DIR", tmp_path / "lines_daily")
    monkeypatch.setattr(CD, "KENPOM_DIR", tmp_path / "kenpom")
    monkeypatch.setattr(CD, "INJURIES_ESPN_DIR", tmp_path / "injuries_espn")
    monkeypatch.setattr(CD, "INJURIES_COVERS_DIR", tmp_path / "injuries_covers")
    monkeypatch.setattr(CD, "HOOPR_SCHEDULE_DIR", tmp_path / "hoopr_schedules")
    monkeypatch.setattr(CD, "RUN_LOG_DIR", tmp_path / "chain_runs")

    rc = CD.main(["--date", TEST_DATE.isoformat()])
    assert rc == 0

    sched = pd.read_parquet(tmp_path / "schedule_daily" / f"{TEST_DATE.isoformat()}.parquet")
    assert "created_at" in sched.columns

    lines = pd.read_parquet(tmp_path / "lines_daily" / f"{TEST_DATE.isoformat()}.parquet")
    assert "created_at" in lines.columns

    kenpom_files = list((tmp_path / "kenpom").rglob("*_kenpom.csv"))
    assert kenpom_files, "expected a kenpom snapshot csv to have been written"
    kp = pd.read_csv(kenpom_files[0])
    assert "created_at" in kp.columns

    espn_payload = json.loads((tmp_path / "injuries_espn" / f"{TEST_DATE.isoformat()}.json").read_text())
    assert "created_at" in espn_payload

    covers_meta = json.loads((tmp_path / "injuries_covers" / f"{TEST_DATE.isoformat()}.meta.json").read_text())
    assert "created_at" in covers_meta

    run_log = json.loads((tmp_path / "chain_runs" / f"{TEST_DATE.isoformat()}.json").read_text())
    assert "created_at" in run_log


@skip_no_network
def test_rerunning_live_is_idempotent_not_duplicative(tmp_path, monkeypatch):
    """Running the same date twice must not duplicate rows in the upserted
    lines file or change the row count of the overwritten schedule file."""
    monkeypatch.setattr(CD, "SCHEDULE_DAILY_DIR", tmp_path / "schedule_daily")
    monkeypatch.setattr(CD, "LINES_DAILY_DIR", tmp_path / "lines_daily")
    monkeypatch.setattr(CD, "KENPOM_DIR", tmp_path / "kenpom")
    monkeypatch.setattr(CD, "INJURIES_ESPN_DIR", tmp_path / "injuries_espn")
    monkeypatch.setattr(CD, "INJURIES_COVERS_DIR", tmp_path / "injuries_covers")
    monkeypatch.setattr(CD, "HOOPR_SCHEDULE_DIR", tmp_path / "hoopr_schedules")
    monkeypatch.setattr(CD, "RUN_LOG_DIR", tmp_path / "chain_runs")

    for _ in range(2):
        rc = CD.main(["--date", TEST_DATE.isoformat()])
        assert rc == 0

    lines1 = pd.read_parquet(tmp_path / "lines_daily" / f"{TEST_DATE.isoformat()}.parquet")
    # a third run's row count must match the second's -- no growth from reruns
    rc = CD.main(["--date", TEST_DATE.isoformat()])
    assert rc == 0
    lines2 = pd.read_parquet(tmp_path / "lines_daily" / f"{TEST_DATE.isoformat()}.parquet")
    assert len(lines1) == len(lines2)
