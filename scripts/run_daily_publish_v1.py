#!/usr/bin/env python
"""
run_daily_publish_v1.py -- daily chain v3, stage PUBLISH (2026-09-30, lane F).

Per game: the sim's margin / total distribution summaries (mean, SD, 5/25/50/75/95 quantiles), home win frequency, and market
probabilities against the lines available at run time (spread cover, over, moneyline; moneyline de-vigged for the probability
comparison only), each line stored with its provider and fetch timestamp. Raw sim frequencies only: no calibration, no adjustment.

    results/daily/publish/<slate_date>/<run_id>/<publish_id>/{slate.parquet, slate.csv, slate.md, lines.parquet, publish_meta.json}

`publish_id` hashes the run id and the stored lines, so re-running with identical lines is a no-op and a re-run with moved lines is a
NEW publication beside the first (grading and CLV use the earliest publication per game: the line that was actually available).
Lines come only through the CBBD API client (`pull_cbbd`) or a file that client wrote; nothing is scraped.

    --lines live          CBBD /lines now (providers Draft Kings, then Bovada, then consensus)
    --lines replay_open   REPLAY: spread / total from the OPEN columns of lines_{season}.parquet (moneyline = close, flagged)
    --lines replay_close  REPLAY: the close
    --lines none
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cbb_sim.live import daily as D  # noqa: E402
from cbb_sim.live import lines as L  # noqa: E402


def run_publish_stage(slate_date: str, run_id: str, now=None, root: Path = D.DAILY_ROOT, lines_mode: str = "live",
                      season: int | None = None, lines_frame: pd.DataFrame | None = None, session=None, tracker=None) -> dict:
    now = D.utc(now) if now is not None else pd.Timestamp.now("UTC")
    sd = D.sim_dir(root, slate_date, run_id)
    if not (sd / "_DONE.json").exists() or not (sd / "games.parquet").exists():
        return {"_status": "skipped", "why": f"no finished sim for {slate_date} {run_id}"}
    games = pd.read_parquet(sd / "games.parquet")
    slate = pd.read_parquet(sd / "slate.parquet")
    if lines_frame is not None:
        lines = lines_frame
    elif lines_mode == "live":
        lines = L.fetch_cbbd_lines(slate_date, now, session=session, tracker=tracker)
    elif lines_mode in ("replay_open", "replay_close"):
        lines = L.replay_lines(int(season), lines_mode.split("_")[1], now, game_ids=slate["cbbd_game_id"])
    else:
        lines = pd.DataFrame(columns=L.LINE_COLS)
    lines = lines[lines["cbbd_game_id"].isin(set(slate["cbbd_game_id"].astype("int64")))] if len(lines) else lines
    import grade_daily_v1 as GR
    meta_p = sd / "run_meta.json"
    rm = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    build_ids = {k: rm.get(k) for k in ("inputs_hash", "config_hash", "engine_tag")}
    try:
        season_start = GR.season_start_of(int(slate["season"].iloc[0] if "season" in slate.columns else season))
    except Exception:  # noqa: BLE001  (label unknown -> NA, never a crash and never a silent False)
        season_start = None
    pub, refused = D.build_publish(games, slate, lines, now, run_id, season_start=season_start, build_ids=build_ids)
    if not len(pub):
        return {"_status": "skipped", "why": "no unplayed game left at publish time", "refused": int(len(refused))}
    pid = pub["publish_id"].iloc[0]
    out = D.pub_dir(root, slate_date, run_id) / pid
    if (out / "slate.parquet").exists():
        return {"_status": "ok", "cached": True, "publish_id": pid, "out": str(out), "n_games": int(len(pub))}
    out.mkdir(parents=True, exist_ok=True)
    pub.to_parquet(out / "slate.parquet", index=False)
    pub.to_csv(out / "slate.csv", index=False)
    lines.to_parquet(out / "lines.parquet", index=False)
    (out / "slate.md").write_text(D.slate_markdown(pub, refused, slate_date, run_id, now), encoding="utf-8")
    meta = {"publish_id": pid, "published_at": str(now), "slate_date": slate_date, "run_id": run_id, "lines_mode": lines_mode,
            "n_games": int(len(pub)), "n_refused_tipped": int(len(refused)), "refused": refused["game_id"].astype(int).tolist(),
            "n_with_spread": int(pub["spread"].notna().sum()), "n_with_total": int(pub["total"].notna().sum()),
            "n_with_ml": int(pub["home_ml"].notna().sum()),
            "providers": pub["provider"].value_counts().to_dict(), "line_kinds": pub["line_kind"].dropna().unique().tolist(),
            "line_fetched_at": sorted({str(x) for x in pub["line_fetched_at"].dropna().unique()}),
            "spread_total_price_assumption": "-110 (CBBD carries no price for spreads / totals)",
            "inputs_hash": build_ids["inputs_hash"], "config_hash": build_ids["config_hash"], "season_start": str(season_start),
            "early_season_totals_flag_games": int(pub["early_season_totals_flag"].fillna(False).sum()),
            "adjustments_to_sim_output": "none"}
    (out / "publish_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return {"_status": "ok", "cached": False, "publish_id": pid, "out": str(out), "n_games": int(len(pub)),
            "n_refused_tipped": int(len(refused)), "n_with_spread": meta["n_with_spread"], "n_with_ml": meta["n_with_ml"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slate-date", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--now", default=None)
    ap.add_argument("--root", default=str(D.DAILY_ROOT))
    ap.add_argument("--lines", choices=("live", "replay_open", "replay_close", "none"), default="live")
    ap.add_argument("--season", type=int, default=None)
    a = ap.parse_args(argv)
    print(json.dumps(run_publish_stage(a.slate_date, a.run_id, a.now, Path(a.root), a.lines, a.season), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
