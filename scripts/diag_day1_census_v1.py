#!/usr/bin/env python
"""
diag_day1_census_v1.py -- tip-time census of a slate under several clocks and the unmapped-game list (lane F2, 2026-09-30).
Reads the tip-time table (pull_tip_times_v1.py) and the CBBD games dump; writes results/day1_census_<slate>.json and prints markdown tables.

    .venv/Scripts/python.exe scripts/diag_day1_census_v1.py --slate-date 2026-11-02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_daily_sim_v1 as SIM  # noqa: E402
from cbb_sim.live import tips as TP  # noqa: E402

GAMES = REPO / "data/raw/preseason/2027_v2_20260930/games_2027.parquet"
XW = REPO / "data/reference/team_crosswalk_v2.parquet"


def census(slate: pd.DataFrame, slate_date: str) -> list[dict]:
    d = pd.Timestamp(slate_date)
    et = "America/New_York"
    clocks = {
        "evening-before 20:00 ET (default)": TP.default_clock(slate_date, "evening"),
        "evening-before 23:59 ET": (d - pd.Timedelta(minutes=1)).tz_localize(et).tz_convert("UTC"),
        "game day 09:00 ET (the old 14:00Z morning clock)": TP.default_clock(slate_date, "morning"),
        "game day 12:00 ET": (d + pd.Timedelta(hours=12)).tz_localize(et).tz_convert("UTC"),
    }
    rows = []
    for name, now in clocks.items():
        for p in ("evening", "morning"):
            ok, late = TP.select_for_pass(slate, now, p)
            rows.append({"clock": name, "clock_utc": str(now), "pass": p, "games": len(slate), "simulate": len(ok), "refuse": len(late),
                         "refuse_tipped": int(late["refuse_reason"].str.startswith("tipoff").sum()) if len(late) else 0,
                         "refuse_placeholder": int(late["refuse_reason"].str.startswith("placeholder").sum()) if len(late) else 0,
                         "simulate_with_placeholder_tip": int(ok["tip_time_is_placeholder"].sum()) if len(ok) else 0})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate-date", required=True)
    ap.add_argument("--season", type=int, default=2027)
    a = ap.parse_args(argv)
    slate = SIM.load_slate(a.slate_date, a.season, "cbbd", str(GAMES), str(XW), "table")
    rows = census(slate, a.slate_date)
    # the OLD behaviour (no refresh table): CBBD tips + hoopR time_valid override, guard only
    old = SIM.load_slate(a.slate_date, a.season, "cbbd", str(GAMES), str(XW), "hoopr")
    from cbb_sim.live import daily as D
    old_rows = []
    for name, now in (("evening-before 20:00 ET", TP.default_clock(a.slate_date, "evening")), ("game day 09:00 ET", TP.default_clock(a.slate_date, "morning"))):
        ok, late = D.split_tipped(old, now)
        old_rows.append({"clock": name, "simulate": len(ok), "refuse": len(late)})
    un = slate.attrs.get("unmapped", [])
    g = pd.read_parquet(GAMES)
    cw = pd.read_parquet(XW)
    cb = "cbbd_team_id" if "cbbd_team_id" in cw.columns else "cbbd_id"
    known = set(cw[cb].astype("int64"))
    gi = g.set_index("id")
    unm = []
    for u in un:
        r = gi.loc[u["id"]]
        unm.append({"cbbd_game_id": int(u["id"]), "home": u["homeTeam"], "away": u["awayTeam"],
                    "home_in_crosswalk": int(r["homeTeamId"]) in known, "away_in_crosswalk": int(r["awayTeamId"]) in known,
                    "home_conf": r["homeConference"], "away_conf": r["awayConference"]})
    out = {"slate_date": a.slate_date, "tips": {"by_source": slate["tip_source"].value_counts().to_dict(),
                                                "placeholder": int(slate["tip_time_is_placeholder"].sum()), "real": int((~slate["tip_time_is_placeholder"]).sum())},
           "census_new": rows, "census_old_guard_only": old_rows, "unmapped": unm,
           "unmapped_both_sides_in_crosswalk": int(sum(x["home_in_crosswalk"] and x["away_in_crosswalk"] for x in unm))}
    p = REPO / f"results/day1_census_{a.slate_date}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("tips", "census_old_guard_only", "unmapped_both_sides_in_crosswalk")}, default=str))
    print(pd.DataFrame(rows).drop(columns="clock_utc").to_string(index=False))
    print(pd.DataFrame(unm).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
