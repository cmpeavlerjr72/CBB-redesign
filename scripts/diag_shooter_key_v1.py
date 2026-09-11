#!/usr/bin/env python
"""
diag_shooter_key_v1.py -- how often is CBBD `participant_1_id` the SHOOTER?

Context. The attribution worker found (2026-09-10, `HANDOFF.md` SHUTDOWN
section) that CBBD's `participants` array is not ordered shooter-first on a
two-participant row: on an ASSISTED made field goal `participant_1_id` is the
ASSISTER about half the time. `cbb_sim.models.event_stream.build_stream` sets
`player_id = participant_1_id` for every row, and `cbb_sim.models.usage` reads
the credited player of an FGA off that column, so the adopted usage bake-off's
shooter labels are contaminated on assisted makes.

This script measures the defect before anything is changed, at the levels
`CLAUDE.md` requires (overall, per season, per event type, per team), and
reports the coverage of the replacement column `shot_shooter_id` so the
drop-versus-impute decision is made on numbers rather than on assumption.

2026 is SEALED and is never read here.

Usage:
    .venv/Scripts/python.exe scripts/diag_shooter_key_v1.py
    .venv/Scripts/python.exe scripts/diag_shooter_key_v1.py --out docs/tests/x.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.pbp.events import PLAY_COLUMNS, classify_frame, load_plays  # noqa: E402

SEASONS = (2022, 2023, 2024, 2025)          # 2026 sealed
AUDIT_COLUMNS: tuple[str, ...] = (
    *PLAY_COLUMNS, "participant_1_id", "participant_2_id",
    "shot_assisted", "shot_assisted_by_id",
)

FGA = ("FGA_rim", "FGA_jump2", "FGA_3")
FT = ("FT_made", "FT_missed")

#: The event buckets of the audit. `usage` keys a player off the first five;
#: the rest are reported so the audit covers every populated participant row.
BUCKET_ORDER: tuple[str, ...] = (
    "FGA made, assisted", "FGA made, unassisted", "FGA made, assist flag missing",
    "FGA missed", "FTA", "TOV", "OREB", "DREB", "DeadBallReb", "foul",
    "steal", "block", "technical", "other",
)


def _bool(col: pd.Series) -> np.ndarray:
    s = col
    if s.dtype == object:
        s = s.map({True: True, False: False})
    return s.astype("boolean").fillna(False).to_numpy(dtype=bool)


def bucket_of(cls: np.ndarray, made: np.ndarray, assisted: np.ndarray,
              assist_known: np.ndarray) -> np.ndarray:
    """Event bucket per row (module docstring)."""
    b = np.full(len(cls), "other", dtype=object)
    is_fga = np.isin(cls, FGA)
    b[is_fga & ~made] = "FGA missed"
    b[is_fga & made & assist_known & assisted] = "FGA made, assisted"
    b[is_fga & made & assist_known & ~assisted] = "FGA made, unassisted"
    b[is_fga & made & ~assist_known] = "FGA made, assist flag missing"
    b[np.isin(cls, FT)] = "FTA"
    for c in ("TOV", "OREB", "DREB", "DeadBallReb", "foul", "steal", "block",
              "technical"):
        b[cls == c] = c
    return b


def season_frame(season: int, universe: pd.DataFrame | None) -> pd.DataFrame:
    """One season's de-duplicated rows with the three id columns and a bucket."""
    gids = None
    if universe is not None:
        u = universe[universe["season"] == season]
        gids = set(u["cbbd_game_id"].dropna().astype("int64").tolist())
    plays = load_plays(season, game_ids=gids, columns=AUDIT_COLUMNS)
    cls = classify_frame(plays, rim_override_max_ft=0.0).to_numpy(dtype=object)
    made = _bool(plays["shot_made"]) | _bool(plays["scoringPlay"])
    assisted = _bool(plays["shot_assisted"])
    assist_known = plays["shot_assisted"].notna().to_numpy()
    p1 = pd.to_numeric(plays["participant_1_id"], errors="coerce").to_numpy()
    p2 = pd.to_numeric(plays["participant_2_id"], errors="coerce").to_numpy()
    sh = pd.to_numeric(plays["shot_shooter_id"], errors="coerce").to_numpy()
    ast = pd.to_numeric(plays["shot_assisted_by_id"], errors="coerce").to_numpy()
    return pd.DataFrame({
        "season": season,
        "team_id": pd.to_numeric(plays["teamId"], errors="coerce").to_numpy(),
        "bucket": bucket_of(cls, made, assisted, assist_known),
        "p1": p1, "p2": p2, "shooter": sh, "assister": ast,
    })


def agg(d: pd.DataFrame) -> dict:
    n = len(d)
    p1 = d["p1"].to_numpy()
    p2 = d["p2"].to_numpy()
    sh = d["shooter"].to_numpy()
    ast = d["assister"].to_numpy()
    has_p1 = np.isfinite(p1)
    has_sh = np.isfinite(sh)
    both = has_p1 & has_sh
    agree = both & (p1 == sh)
    p1_is_ast = both & ~agree & np.isfinite(ast) & (p1 == ast)
    p2_is_sh = both & ~agree & np.isfinite(p2) & (p2 == sh)
    return {
        "n": int(n),
        "p1_present_pct": round(float(has_p1.mean() * 100), 4) if n else None,
        "shooter_present_pct": round(float(has_sh.mean() * 100), 4) if n else None,
        "shooter_missing_pct": round(float((~has_sh).mean() * 100), 4) if n else None,
        "both_present_n": int(both.sum()),
        "agree_pct": round(float(agree.sum() / both.sum() * 100), 4) if both.any() else None,
        "p1_is_assister_pct_of_disagree": round(
            float(p1_is_ast.sum() / (both.sum() - agree.sum()) * 100), 4)
        if (both.sum() - agree.sum()) else None,
        "p2_is_shooter_pct_of_disagree": round(
            float(p2_is_sh.sum() / (both.sum() - agree.sum()) * 100), 4)
        if (both.sum() - agree.sum()) else None,
    }


def per_team(d: pd.DataFrame) -> dict:
    """Distribution across teams of the agreement rate on shooting rows."""
    m = d[np.isfinite(d["shooter"].to_numpy()) & np.isfinite(d["p1"].to_numpy())]
    m = m[np.isfinite(m["team_id"].to_numpy())]
    if not len(m):
        return {}
    g = m.assign(ok=(m["p1"].to_numpy() == m["shooter"].to_numpy())).groupby("team_id")
    r = g["ok"].agg(["size", "mean"])
    r = r[r["size"] >= 200]
    if not len(r):
        return {}
    q = r["mean"] * 100
    return {"teams": int(len(r)),
            "min_pct": round(float(q.min()), 4),
            "p05_pct": round(float(q.quantile(0.05)), 4),
            "median_pct": round(float(q.median()), 4),
            "p95_pct": round(float(q.quantile(0.95)), 4),
            "max_pct": round(float(q.max()), 4),
            "sd_pp": round(float(q.std(ddof=1)), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/tests/shooter_key_audit_2026-09-10.md")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    universe = ES.load_universe(require_pbp_complete=True)
    out: dict = {"seasons": {}, "buckets": {}, "per_team": {}, "raw_all_rows": {}}

    frames = []
    for s in SEASONS:
        d = season_frame(s, universe)
        frames.append(d)
        print(f"{s}: {len(d):,} universe rows ({time.time() - t0:.0f}s)", flush=True)
    allrows = pd.concat(frames, ignore_index=True)

    shooting = allrows[allrows["bucket"].str.startswith("FGA")]
    for s in SEASONS:
        sd = allrows[allrows["season"] == s]
        out["seasons"][str(s)] = {
            "all_rows": agg(sd),
            "fga_rows": agg(sd[sd["bucket"].str.startswith("FGA")]),
            "made_fga_rows": agg(sd[sd["bucket"].str.startswith("FGA made")]),
        }
        out["per_team"][str(s)] = per_team(sd[sd["bucket"].str.startswith("FGA")])
    out["per_team"]["all"] = per_team(shooting)

    for b in BUCKET_ORDER:
        bd = allrows[allrows["bucket"] == b]
        if not len(bd):
            continue
        out["buckets"][b] = {"pooled": agg(bd),
                             "by_season": {str(s): agg(bd[bd["season"] == s])
                                           for s in SEASONS}}

    # the same numbers with NO universe filter, so the HANDOFF figures (which
    # were measured on raw 2025 made field goals) can be tied out
    for s in SEASONS:
        d = season_frame(s, None)
        out["raw_all_rows"][str(s)] = {
            "all_rows": agg(d),
            "made_fga_rows": agg(d[d["bucket"].str.startswith("FGA made")]),
            "made_fga_assisted": agg(d[d["bucket"] == "FGA made, assisted"]),
        }
        print(f"{s} raw: {len(d):,} rows ({time.time() - t0:.0f}s)", flush=True)

    jp = Path(args.json_out) if args.json_out else \
        ROOT / "data/processed/models/usage_v2/shooter_key_audit.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {jp} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
