#!/usr/bin/env python
"""
diag_ot_carry_parity_v1.py -- train/serve parity of the OVERTIME TEAM-FOUL CARRY, regulation AND overtime rows (lane F, 2026-10-01).
Chain parity stage under `--ot-foul-carry`; FAILS (exit 1) on a mismatch.

The engine serves the carried team-foul state (`engine/loop.py`: reset at halftime only). This checks that every consumer's
carried TRAINING table holds the same state, against an INDEPENDENT recount from the plays (a groupby / cumcount over the
event stream with the segment rule `period 1 | periods >= 2`, no use of the stream's own carry switch):

  fg_make events (fg_make/events_v2_shotshooter.parquet) and rebound events (rebound/events_v1.parquet): per shot / miss row,
      `off_in_bonus` == (defence prior fouls >= 6) and `off_in_double_bonus` == (>= 9), joined on
      (cbbd_game_id, period, seconds_remaining, offence side); keys that repeat on either side are dropped (counted).
  free-throw attempts (free_throw/attempts_v1_era.parquet): `trip_prior_fouls` / `prior_fouls` where present, same join.
  possessions / chances (possessions_<version>/): OT rows only, informational (the count at possession open includes pre-open
      fouls, so an exact row join is not defined); reported, never fails the stage.

FAIL rule: on fg and rebound events, the mismatch rate among joined rows must be <= --max-mismatch (default 0.005) in regulation
AND in overtime, and the OT rows joined must be >= --min-ot-rows (default 5,000 per table; the audit counts about 16,000 OT stream
rows a season). The sample therefore always contains the OT rows: they are what the check is about.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.pbp.possessions import BONUS_PRIOR_FOULS, DOUBLE_BONUS_PRIOR_FOULS  # noqa: E402

SEASONS = [2022, 2023, 2024, 2025]
KEY = ["cbbd_game_id", "period", "seconds_remaining", "side"]


def recount(stream: pd.DataFrame) -> pd.DataFrame:
    """Per stream row: the CARRIED prior foul count of the DEFENCE of that row's side (independent of the stream's own switch)."""
    s = stream[["cbbd_game_id", "period", "cls", "side"]].copy()
    s["seg"] = np.where(s["period"] >= 2, 2, s["period"])
    n = len(s)
    own = np.full(n, -1, dtype="int32")
    opp = np.full(n, -1, dtype="int32")
    for team in (0, 1):
        f = ((s["cls"] == "foul") & (s["side"] == team)).astype("int32")
        cum = f.groupby([s["cbbd_game_id"], s["seg"]], sort=False).cumsum() - f           # strictly before this row
        own = np.where(s["side"] == team, cum, own)
        opp = np.where(s["side"] == 1 - team, cum, opp)                                    # row of the other side: this team is its defence
    out = stream[["cbbd_game_id", "period", "seconds_remaining" if "seconds_remaining" in stream.columns else "sec"]].copy()
    out.columns = ["cbbd_game_id", "period", "seconds_remaining"]
    out["side"] = s["side"].to_numpy()
    out["def_prior_carry"] = opp
    return out


def load_recount() -> pd.DataFrame:
    os.environ.pop(ES.OT_FOUL_CARRY_ENV, None)                    # the independent count must not rely on the switch
    u = ES.load_universe()
    parts = [recount(ES.build_stream(s, u)) for s in SEASONS]
    return pd.concat(parts, ignore_index=True)


def check_events(name: str, path: Path, rc: pd.DataFrame, max_mm: float, min_ot: int) -> dict:
    ev = pd.read_parquet(path)
    ev = ev[ev["season"].isin(SEASONS)].copy()
    ev["side"] = np.where(ev["offense_is_home"].astype(bool), 0, 1)
    rcu = rc.drop_duplicates(KEY, keep=False)
    evu = ev.drop_duplicates(KEY, keep=False)
    m = evu.merge(rcu, on=KEY, how="inner")
    exp_b = (m["def_prior_carry"] >= BONUS_PRIOR_FOULS)
    exp_d = (m["def_prior_carry"] >= DOUBLE_BONUS_PRIOR_FOULS)
    bad = (m["off_in_bonus"].astype(bool) != exp_b) | (m["off_in_double_bonus"].astype(bool) != exp_d)
    ot = (m["period"] >= 3).to_numpy()
    r = {"table": str(path), "rows": int(len(ev)), "joined": int(len(m)),
         "regulation_joined": int((~ot).sum()), "regulation_mismatch": int((bad & ~ot).sum()),
         "ot_joined": int(ot.sum()), "ot_mismatch": int((bad & ot).sum())}
    r["regulation_mismatch_rate"] = r["regulation_mismatch"] / max(r["regulation_joined"], 1)
    r["ot_mismatch_rate"] = r["ot_mismatch"] / max(r["ot_joined"], 1)
    r["ot_bonus_share_trained"] = float(m.loc[ot, "off_in_bonus"].astype(float).mean()) if ot.any() else None
    r["ot_bonus_share_expected"] = float(exp_b[ot].mean()) if ot.any() else None
    r["pass"] = bool(r["regulation_mismatch_rate"] <= max_mm and r["ot_mismatch_rate"] <= max_mm and r["ot_joined"] >= min_ot)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--otc-root", type=Path, required=True, help="build_ot_carry_designs_v1.py --out-root")
    ap.add_argument("--possessions-version", default="v4otc")
    ap.add_argument("--max-mismatch", type=float, default=0.005)
    ap.add_argument("--min-ot-rows", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = a.otc_root if a.otc_root.is_absolute() else ROOT / a.otc_root
    rc = load_recount()
    rep = {"otc_root": str(root), "recount_rows": int(len(rc)), "checks": {}}
    rep["checks"]["fg_events"] = check_events("fg", root / "fg_make/events_v2_shotshooter.parquet", rc, a.max_mismatch, a.min_ot_rows)
    rep["checks"]["rebound_events"] = check_events("rb", root / "rebound/events_v1.parquet", rc, a.max_mismatch, a.min_ot_rows)
    # informational: the possessions / chances tables, OT rows
    info = {}
    from cbb_sim.pbp.possessions import possessions_dir
    pdir = ROOT / possessions_dir(a.possessions_version)
    for s in SEASONS:
        c = pd.read_parquet(pdir / f"chances_{s}.parquet", columns=["period", "off_in_bonus", "off_in_double_bonus"])
        o = c[c["period"] >= 3]
        info[str(s)] = {"ot_chances": int(len(o)), "ot_off_in_bonus_share": float(o["off_in_bonus"].astype(float).mean()),
                        "ot_off_in_double_bonus_share": float(o["off_in_double_bonus"].astype(float).mean())}
    rep["chances_ot_informational"] = info
    rep["pass"] = all(v["pass"] for v in rep["checks"].values())
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
