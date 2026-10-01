#!/usr/bin/env python
"""
diag_daily_twopass_replay_v1.py -- evidence for the two-pass daily chain (lane F, 2026-10-01). No engine runs here.

  census   per date: old single clock (14:00Z of the slate date), evening pass (20:00 ET D-1), morning pass (09:00 ET D), on the universe
           slate (replay tips are real: 0 midnight-ET placeholders on all four dates), plus a SYNTHETIC placeholder arm on the opening day
           (a deterministic share of games has its tip replaced by midnight ET of the slate date and flagged, exactly as the live CBBD
           feed does; the true tip is the "refreshed" value): evening / morning before refresh / morning after refresh.
  verify   reads the sim outputs of the two-pass replay (`chain_daily_v3.py --replay-season S --slate-date D --replay-pass P`) and checks
           rows = games x seeds, created_at < tipoff on every row, pre_tip_basis columns, and that placeholder rows are never verified.
  guards   negative tests: a clock after the earliest possible tip with a placeholder row raises; a tipped game under --strict raises.

    .venv/Scripts/python.exe scripts/diag_daily_twopass_replay_v1.py census|verify|guards [--root results/daily_replay_twopass]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from cbb_sim.live import guards as G  # noqa: E402
from cbb_sim.live import tips as TP  # noqa: E402
import run_daily_sim_v1 as SIM  # noqa: E402

DATES = [("2024-11-04", 2025, "opening day 2024-25"), ("2024-11-05", 2025, "day 2 (opening-day sim blocked, see doc)"), ("2025-02-11", 2025, "v3 doc date"), ("2025-02-25", 2025, "v3 doc date"),
         ("2025-03-04", 2025, "v3 doc date")]
MASK_FRAC = 0.8


def mask(slate: pd.DataFrame, slate_date: str, frac: float = MASK_FRAC) -> pd.DataFrame:
    """Synthetic live-feed state: `frac` of games carry the midnight-ET placeholder (deterministic by game_id)."""
    s = slate.copy()
    h = s["game_id"].astype("int64").map(lambda x: int(hashlib.sha1(str(x).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    m = h < frac
    s["true_tip"] = s["tipoff_utc"]
    s.loc[m, "tipoff_utc"] = TP.earliest_possible_tip(slate_date)
    s["tip_time_is_placeholder"] = m.to_numpy()
    return s


def census() -> list[dict]:
    rows = []
    for d, season, why in DATES:
        sl = SIM.load_slate(d, season, "universe", None, None, None)
        old_clock = pd.Timestamp(f"{d}T14:00:00Z")
        r = {"date": d, "why": why, "games": len(sl), "placeholders_real_feed": int(sl["tip_time_is_placeholder"].sum()),
             "old_single_clock_14Z_simulate": int(len(TP.select_for_pass(sl, old_clock, "evening")[0]))}
        for p in ("evening", "morning"):
            ok, late = TP.select_for_pass(sl, TP.default_clock(d, p), p)
            r[f"{p}_simulate"], r[f"{p}_refuse"] = int(len(ok)), int(len(late))
        ms = mask(sl, d)
        ev = TP.default_clock(d, "evening")
        mo = TP.default_clock(d, "morning")
        a, _ = TP.select_for_pass(ms, ev, "evening")
        b, bl = TP.select_for_pass(ms, mo, "morning")
        c, _ = TP.select_for_pass(ms.assign(tipoff_utc=ms["true_tip"], tip_time_is_placeholder=False), mo, "morning")
        r["synthetic"] = {"masked_games": int(ms["tip_time_is_placeholder"].sum()), "evening_simulate": int(len(a)),
                          "morning_before_refresh_simulate": int(len(b)), "morning_before_refresh_refuse": int(len(bl)),
                          "morning_refuse_reasons": bl["refuse_reason"].value_counts().to_dict() if len(bl) else {},
                          "morning_after_refresh_simulate": int(len(c))}
        # the old single clock on a masked slate: tipped test only (a 14Z clock is after midnight ET placeholders)
        r["synthetic"]["old_single_clock_14Z_simulate_masked"] = int(len(TP.select_for_pass(ms, old_clock, "evening")[0]))
        rows.append(r)
    return rows


def verify(root: Path) -> list[dict]:
    out = []
    for d, season, why in DATES:
        for p in ("evening", "morning"):
            rid = "s4_o0" + ("_morning" if p == "morning" else "")
            f = root / "sim" / d / rid / "games.parquet"
            if not f.exists():
                out.append({"date": d, "pass": p, "status": "NOT RUN"})
                continue
            g = pd.read_parquet(f)
            meta = json.loads((root / "sim" / d / rid / "run_meta.json").read_text(encoding="utf-8"))
            sk = json.loads((root / "sim" / d / rid / "skipped.json").read_text(encoding="utf-8"))
            G.assert_created_before_tipoff(g)
            n_g = g["game_id"].nunique()
            out.append({"date": d, "pass": p, "games": int(n_g), "rows": int(len(g)), "seeds": int(len(meta["seeds"])),
                        "rows_eq_games_x_seeds": bool(len(g) == n_g * len(meta["seeds"])), "refused": len(sk["already_tipped"]),
                        "created_at": str(g["created_at"].iloc[0]), "min_tipoff": str(g["tipoff_utc"].min()),
                        "rows_created_before_tipoff": int((pd.to_datetime(g["created_at"], utc=True) < pd.to_datetime(g["tipoff_utc"], utc=True)).sum()),
                        "placeholder_rows": int(g["tip_time_is_placeholder"].sum()), "verified_rows": int(g["pre_tip_verified"].sum()),
                        "guard": "held"})
    return out


def guards() -> dict:
    res = {}
    d = "2024-11-04"
    sl = SIM.load_slate(d, 2025, "universe", None, None, None)
    ms = mask(sl, d)
    ev = TP.default_clock(d, "evening")
    ok, _ = TP.select_for_pass(ms, ev, "evening")
    games = ok[["game_id", "tipoff_utc"]].assign(created_at=ev, home_pts=0)
    st = TP.stamp_pre_tip_basis(games, ok, d)
    res["evening_masked_rows"] = int(len(st))
    res["evening_masked_placeholder_rows"] = int(st["tip_time_is_placeholder"].sum())
    res["evening_masked_verified_rows"] = int(st["pre_tip_verified"].sum())
    res["placeholder_rows_never_verified"] = bool(not st.loc[st["tip_time_is_placeholder"], "pre_tip_verified"].any())
    # clock after the earliest possible tip, placeholder row smuggled in -> must raise
    late = games.assign(created_at=TP.earliest_possible_tip(d) + pd.Timedelta(hours=1))
    try:
        TP.stamp_pre_tip_basis(late, ok, d)
        res["late_clock_placeholder_row"] = "NO RAISE (BUG)"
    except G.LeakGuardError as e:
        res["late_clock_placeholder_row"] = f"LeakGuardError: {e}"
    # the pass filter refuses placeholders in the morning and never calls them tipped
    b, bl = TP.select_for_pass(ms, TP.default_clock(d, "morning"), "morning")
    res["morning_refuse_reason_placeholders"] = bl.loc[bl["tip_time_is_placeholder"], "refuse_reason"].value_counts().to_dict()
    # the base guard still raises on a tipped game
    try:
        G.assert_created_before_tipoff(games.assign(created_at=games["tipoff_utc"].min() + pd.Timedelta(minutes=1)))
        res["tipped_row"] = "NO RAISE (BUG)"
    except G.LeakGuardError as e:
        res["tipped_row"] = f"LeakGuardError: {str(e)[:80]}"
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("census", "verify", "guards"))
    ap.add_argument("--root", default="results/daily_replay_twopass")
    a = ap.parse_args()
    r = {"census": census, "guards": guards}.get(a.what, lambda: verify(Path(a.root)))()
    print(json.dumps(r, indent=1, default=str))
    Path(REPO / f"results/twopass_{a.what}.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
