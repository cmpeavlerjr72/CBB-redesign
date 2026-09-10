#!/usr/bin/env python3
"""
Walk the full cbb-sims-2026 season output tree, load summary.json + final.json
(+ clv.json when present) for every graded game, and build one master parquet
file for downstream analysis. Also loads boosted_sims / improved / pace-experiment
variants for side-by-side comparison on the overlapping date range.

Run with: python results_load_data.py
Outputs (into this scratchpad dir):
  master_games.parquet   (base model, all dates with a final score)
  boosted_games.parquet
  improved_games.parquet
  pace_games.parquet
  load_summary.json      (counts / diagnostics)
"""
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path(r"C:\Users\devuser\CBB-Monte-storage\out")
SCRATCH = Path(r"C:\Users\devuser\AppData\Local\Temp\claude\C--Users-devuser-CBB-clean-sheet\f90c786b-0ad5-48b7-b88a-f8ef6f7594ed\scratchpad")

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_json(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def to_num(x):
    try:
        if x is None or x == "":
            return None
        n = float(x)
        if math.isfinite(n):
            return n
    except Exception:
        pass
    return None


def extract_final_AB(F, teamA_name):
    """Same logic as compute_daily_results.py: returns (finalA, finalB) or None."""
    if not F:
        return None
    status = F.get("status")
    state = F.get("state")
    if not (status in (1, "1") or (isinstance(state, str) and state.lower() == "final")):
        return None
    home = (F.get("scores") or {}).get("home")
    away = (F.get("scores") or {}).get("away")
    if home is None or away is None:
        home = (F.get("game") or {}).get("home", {})
        away = (F.get("game") or {}).get("away", {})
        home = home.get("score") if isinstance(home, dict) else F.get("home_score", F.get("final_home"))
        away = away.get("score") if isinstance(away, dict) else F.get("away_score", F.get("final_away"))
    home = to_num(home)
    away = to_num(away)
    if home is None or away is None:
        return None
    matched = str((F.get("odds") or {}).get("matched_home_side", F.get("matched_home_side", "A"))).upper()
    if matched == "A":
        return home, away
    if matched == "B":
        return away, home
    home_name = str(
        (F.get("matched_with") or {}).get("home")
        or F.get("home_team")
        or ((F.get("game") or {}).get("home") if isinstance((F.get("game") or {}).get("home"), str)
            else ((F.get("game") or {}).get("home") or {}).get("name", ""))
    ).lower()
    if home_name and teamA_name and teamA_name.lower() in home_name:
        return home, away
    return away, home


def spread_for_A(closing_spread, matched_home_side):
    """Return the spread number applied to team A (A covers if actual_A-B + this > 0)."""
    if not closing_spread:
        return None
    home_line = to_num(closing_spread.get("home_line"))
    away_line = to_num(closing_spread.get("away_line"))
    if matched_home_side == "A":
        return home_line
    if matched_home_side == "B":
        if away_line is not None:
            return away_line
        if home_line is not None:
            return -home_line
    return None


def open_spread_for_A(open_odds, matched_home_side):
    if not open_odds:
        return None
    home_line = to_num(open_odds.get("home_spread"))
    away_line = to_num(open_odds.get("away_spread"))
    if matched_home_side == "A":
        return home_line
    if matched_home_side == "B":
        if away_line is not None:
            return away_line
        if home_line is not None:
            return -home_line
    return None


def ml_for_side(closing_ml_or_open, side, matched_home_side):
    if not closing_ml_or_open:
        return None
    home = to_num(closing_ml_or_open.get("home"))
    away = to_num(closing_ml_or_open.get("away"))
    if matched_home_side == "A":
        return home if side == "A" else away
    else:
        return away if side == "A" else home


def open_ml_for_side(open_odds, side, matched_home_side):
    if not open_odds:
        return None
    home = to_num(open_odds.get("home_ml"))
    away = to_num(open_odds.get("away_ml"))
    if matched_home_side == "A":
        return home if side == "A" else away
    else:
        return away if side == "A" else home


def american_to_prob(price):
    if price is None:
        return None
    price = float(price)
    if price < 0:
        return (-price) / ((-price) + 100.0)
    else:
        return 100.0 / (price + 100.0)


def load_variant(root: Path, kind: str):
    """
    kind == 'nested'  -> root/<date>/games/<game>/{summary.json,final.json,clv.json}
                          (used by cbb-sims-2026 and experiments/cbb-sims-pace-2026)
    kind == 'boosted' -> root/<date>/<game>.json  (single file per game, boosted_sims)
    kind == 'improved'-> root/<date>/improved_summaries.json (list of games)
    """
    rows = []
    if not root.exists():
        return pd.DataFrame(rows)

    if kind == "nested":
        for date_dir in sorted(root.iterdir()):
            if not date_dir.is_dir() or not DATE_RE.match(date_dir.name):
                continue
            games_dir = date_dir / "games"
            if not games_dir.is_dir():
                continue
            for gdir in sorted(games_dir.iterdir()):
                if not gdir.is_dir():
                    continue
                S = read_json(gdir / "summary.json")
                F = read_json(gdir / "final.json")
                if not S:
                    continue
                row = build_row_from_S_F(date_dir.name, gdir.name, S, F)
                if row:
                    C = read_json(gdir / "clv.json")
                    if C and isinstance(C.get("clv"), dict):
                        row["clv_json"] = json.dumps(C["clv"])
                    rows.append(row)
        return pd.DataFrame(rows)

    if kind == "boosted":
        for date_dir in sorted(root.iterdir()):
            if not date_dir.is_dir() or not DATE_RE.match(date_dir.name):
                continue
            for gf in sorted(date_dir.glob("*.json")):
                S = read_json(gf)
                if not S:
                    continue
                # boosted_sims files appear to be summary-shaped; final score may be embedded or absent
                F = S.get("final") if isinstance(S.get("final"), dict) else None
                gid = gf.stem
                row = build_row_from_S_F(date_dir.name, gid, S, F)
                if row:
                    rows.append(row)
        return pd.DataFrame(rows)

    if kind == "improved":
        for date_dir in sorted(root.iterdir()):
            if not date_dir.is_dir() or not DATE_RE.match(date_dir.name):
                continue
            fp = date_dir / "improved_summaries.json"
            data = read_json(fp)
            if not isinstance(data, list):
                continue
            for S in data:
                if not isinstance(S, dict):
                    continue
                gid = S.get("game_id") or f"{date_dir.name}__{S.get('A_slug')}__{S.get('B_slug')}"
                F = S.get("final") if isinstance(S.get("final"), dict) else None
                row = build_row_from_S_F(date_dir.name, gid, S, F)
                if row:
                    rows.append(row)
        return pd.DataFrame(rows)

    return pd.DataFrame(rows)


def build_row_from_S_F(date_str, game_id, S, F):
    teamA = str(S.get("A_kp_name") or S.get("A_name") or S.get("A_slug") or "A")
    teamB = str(S.get("B_kp_name") or S.get("B_name") or S.get("B_slug") or "B")
    A_slug = str(S.get("A_slug") or teamA).lower()
    B_slug = str(S.get("B_slug") or teamB).lower()

    ab = extract_final_AB(F, teamA) if F else None
    finalA = finalB = None
    if ab:
        finalA, finalB = ab

    odds = S.get("odds") or {}
    matched = str(odds.get("matched_home_side", "A")).upper()

    A_win_prob = to_num(S.get("A_win_prob"))
    # base/boosted/pace schema uses margin_p50/total_p50; "improved" schema uses
    # spread_mean/total_mean instead -- fall back accordingly.
    margin_p50 = to_num(S.get("margin_p50"))
    if margin_p50 is None:
        margin_p50 = to_num(S.get("spread_mean"))
    total_p50 = to_num(S.get("total_p50"))
    if total_p50 is None:
        total_p50 = to_num(S.get("total_mean"))

    open_total = to_num(odds.get("total"))
    open_spreadA = open_spread_for_A(odds, matched)
    open_mlA = open_ml_for_side(odds, "A", matched)
    open_mlB = open_ml_for_side(odds, "B", matched)

    close_spreadA = None
    close_total = None
    close_mlA = close_mlB = None
    if F:
        closing = F.get("closing") or {}
        close_spreadA = spread_for_A(closing.get("spread"), matched)
        close_total = to_num((closing.get("total") or {}).get("line"))
        close_mlA = ml_for_side(closing.get("moneyline"), "A", matched)
        close_mlB = ml_for_side(closing.get("moneyline"), "B", matched)

    row = {
        "date": date_str,
        "game_id": game_id,
        "teamA": teamA,
        "teamB": teamB,
        "A_slug": A_slug,
        "B_slug": B_slug,
        "join_key": date_str + "__" + "__".join(sorted([A_slug, B_slug])),
        "matched_home_side": matched,
        "finalA": finalA,
        "finalB": finalB,
        "has_final": finalA is not None and finalB is not None,
        "A_win_prob": A_win_prob,
        "margin_p50": margin_p50,
        "total_p50": total_p50,
        "open_total": open_total,
        "open_spreadA": open_spreadA,
        "open_mlA": open_mlA,
        "open_mlB": open_mlB,
        "close_spreadA": close_spreadA,
        "close_total": close_total,
        "close_mlA": close_mlA,
        "close_mlB": close_mlB,
        "created_at": S.get("created_at"),
        "start_utc": S.get("start_utc"),
    }
    return row


def main():
    print("Loading base model (cbb-sims-2026) ...", file=sys.stderr)
    base = load_variant(OUT / "cbb-sims-2026/2026/days", "nested")
    print(f"  base rows: {len(base)}", file=sys.stderr)
    base.to_parquet(SCRATCH / "master_games.parquet", index=False)

    print("Loading boosted_sims ...", file=sys.stderr)
    boosted = load_variant(OUT / "boosted_sims", "boosted")
    print(f"  boosted rows: {len(boosted)}", file=sys.stderr)
    boosted.to_parquet(SCRATCH / "boosted_games.parquet", index=False)

    print("Loading improved ...", file=sys.stderr)
    improved = load_variant(OUT / "improved/2026", "improved")
    print(f"  improved rows: {len(improved)}", file=sys.stderr)
    improved.to_parquet(SCRATCH / "improved_games.parquet", index=False)

    print("Loading pace experiment ...", file=sys.stderr)
    pace = load_variant(OUT / "experiments/cbb-sims-pace-2026/2026/days", "nested")
    print(f"  pace rows: {len(pace)}", file=sys.stderr)
    pace.to_parquet(SCRATCH / "pace_games.parquet", index=False)

    summary = {
        "base_rows": len(base),
        "base_rows_with_final": int(base["has_final"].sum()) if len(base) else 0,
        "base_date_min": str(base["date"].min()) if len(base) else None,
        "base_date_max": str(base["date"].max()) if len(base) else None,
        "boosted_rows": len(boosted),
        "boosted_rows_with_final": int(boosted["has_final"].sum()) if len(boosted) else 0,
        "improved_rows": len(improved),
        "improved_rows_with_final": int(improved["has_final"].sum()) if len(improved) else 0,
        "pace_rows": len(pace),
        "pace_rows_with_final": int(pace["has_final"].sum()) if len(pace) else 0,
    }
    with (SCRATCH / "load_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
