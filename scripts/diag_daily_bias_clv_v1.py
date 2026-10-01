#!/usr/bin/env python
"""
diag_daily_bias_clv_v1.py -- daily chain v3, stage BIAS / CLV MONITOR (2026-09-30, lane F).

Reads the grade ledger (`results/daily/grade/ledger.parquet`) and reports:
  * rolling margin and total bias (sim minus actual) with standard errors, trailing 14 d / 30 d / all;
  * per-stat bias (3PA, rim / jump 2PA, FTA and the matching makes) where the event-layer truth table has the game;
  * responsiveness: predicted-margin quintile vs actual margin (must slope);
  * closing-line value of each published lean: publish-time stored line vs the close (same provider): ATS and O/U in points,
    moneyline in de-vigged probability (not computable in replay, where the stored ML is the close);
  * surprise correlation and CLV agreement, and the leak verdict (cbb_sim.eval.market.leak_verdict, tolerances from docs/gates.yaml).

RULE (CLAUDE.md): an edge that beats the close but cannot predict line movement is presumed leaked.
It REPORTS and ALARMS. It never corrects sim output. Exit status of the stage is "alarm" when any alarm fires; nothing else changes.

Close sources: `live` = `data/raw/lines_daily/*.parquet` snapshot_type == close (the chain's lines step, fetched the morning after);
`replay_close` = `lines_{season}.parquet` (REPLAY); `none`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cbb_sim.live import daily as D  # noqa: E402
from cbb_sim.live import lines as L  # noqa: E402


def report_markdown(m: dict, asof, close_mode: str) -> str:
    o = [f"# Bias / CLV monitor, as of {asof}", "", f"> {m['rule']}", "",
         f"Ledger games: {m['n_ledger']}. Close source: {close_mode}. Reports and alarms only; no sim output is changed.", "",
         "## Alarms", ""]
    o += [f"- {a}" for a in m["alarms"]] or ["- none"]
    o += ["", "## Rolling bias (sim minus actual; SE = SD / sqrt(n); z = bias / SE)", "", D.fmt_df(m["rolling_bias"]),
          "## Per-stat bias per team-game (event-layer truth; sim minus actual)", "", D.fmt_df(m["stat_bias"]),
          "(rebounds, turnovers and assists have no per-game truth table in this stage: not monitored here.)", ""]
    if len(m["responsiveness"]):
        o += ["## Responsiveness (predicted margin quintile vs actual margin)", "", D.fmt_df(m["responsiveness"])]
    else:
        o += ["## Responsiveness", "", "UNDERPOWERED (< 25 graded games)", ""]
    o += ["## Closing-line value of published leans", ""]
    if m.get("clv"):
        rows = [{"market": k, **v} for k, v in m["clv"].items()]
        o += [D.fmt_df(pd.DataFrame(rows)),
              "Positive CLV = the line moved toward the lean. `clv_agreement` = share of moved lines that moved toward the lean (about 0.50 is coin-flip). "
              "Points for ATS / O/U, de-vigged probability for ML.", "",
              f"Surprise correlation (ATS disagreement at the publish line vs realised cover margin): {m['surprise_corr_ats']:.3f} (n={m['surprise_n']})",
              f"Leak verdict (ATS): **{m['leak_verdict_ats']}**", ""]
    else:
        o += [f"PENDING: {m['leak_verdict_ats']}", ""]
    return "\n".join(o)


def run_monitor_stage(season: int | None, asof=None, root: Path = D.DAILY_ROOT, close: str = "live", gates_yaml: Path | None = None,
                      close_frame: pd.DataFrame | None = None, lines_daily: Path | None = None) -> dict:
    root = Path(root)
    lp = root / "grade" / "ledger.parquet"
    if not lp.exists():
        return {"_status": "skipped", "why": "no grade ledger yet"}
    led = pd.read_parquet(lp)
    asof = D.utc(asof) if asof is not None else pd.Timestamp.now("UTC")
    led = led[pd.to_datetime(led["graded_at"], utc=True) <= asof] if len(led) else led
    tol = yaml.safe_load(Path(gates_yaml or REPO / "docs/gates.yaml").read_text(encoding="utf-8"))
    if close_frame is not None:
        cl = close_frame
    elif close == "live":
        cl = L.daily_snapshots(lines_daily or REPO / "data/raw/lines_daily", "close")
    elif close == "replay_close":
        cl = L.replay_lines(int(season), "close", asof, game_ids=led["cbbd_game_id"], providers=tuple(led["provider"].dropna().unique()) or L.REPLAY_PROVIDERS)
    else:
        cl = None
    m = D.monitor(led, cl, tol, asof=led["game_date"].max() if len(led) else None)
    odir = root / "monitor" / f"{asof.strftime('%Y-%m-%d')}"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "report.md").write_text(report_markdown(m, asof, close), encoding="utf-8")
    if "clv_frame" in m:
        m["clv_frame"].to_parquet(odir / "clv.parquet", index=False)
    summ = {"alarms": m["alarms"], "n_ledger": m["n_ledger"], "leak_verdict_ats": m["leak_verdict_ats"],
            "clv": m.get("clv"), "rolling_bias": m["rolling_bias"].to_dict("records"), "rule": m["rule"]}
    (odir / "monitor.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    return {"_status": "alarm" if m["alarms"] else "ok", "alarms": m["alarms"], "n_ledger": m["n_ledger"],
            "leak_verdict_ats": m["leak_verdict_ats"], "report": str(odir / "report.md")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--asof", default=None)
    ap.add_argument("--root", default=str(D.DAILY_ROOT))
    ap.add_argument("--close", choices=("live", "replay_close", "none"), default="live")
    a = ap.parse_args(argv)
    print(json.dumps(run_monitor_stage(a.season, a.asof, Path(a.root), a.close), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
