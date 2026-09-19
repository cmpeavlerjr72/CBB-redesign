#!/usr/bin/env python
"""
diag_technical_blast_radius_v1.py -- old vs new event-layer tables for the
bounded same-clock technical free-throw lookahead.

    .venv/Scripts/python.exe scripts/diag_technical_blast_radius_v1.py

Compares, per season 2022-2025 and for fold 2 (test season 2025) specifically:

  possessions  data/processed/possessions_v2  vs  data/processed/possessions_v3
               (v3 IS v2 plus the lookahead, so the diff is the fix alone)
  trips        models/free_throw/trips_v1_era.parquet
               vs  models/free_throw/trips_v1_era_techfix.parquet
  target       models/free_throw/technical_target_verified_trips_v1.parquet

Reads only; writes results/event_layer_technical/blast_radius.json (gitignored).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SEASONS = [2022, 2023, 2024, 2025]
OLD = Path("data/processed/possessions_v2")
NEW = Path("data/processed/possessions_v3")
FT_DIR = Path("data/processed/models/free_throw")
RESULTS = Path("results/event_layer_technical")

KEY_COLS = ["period", "offense_team_id", "terminal_event", "fta", "ftm", "points",
            "n_chances", "tech_points_off", "tech_points_def", "start_reason"]


def season_block(s: int) -> dict:
    a = pd.read_parquet(OLD / f"possessions_{s}.parquet")
    b = pd.read_parquet(NEW / f"possessions_{s}.parquet")
    out: dict = {"season": s}

    def agg(p: pd.DataFrame, tag: str) -> dict:
        fga = (p["fga_rim"] + p["fga_jump2"] + p["fga_3"]).sum()
        ng = p["game_id"].nunique()
        tov = int((p["terminal_event"] == "TOV").sum())
        d = {
            f"{tag}_n_poss": int(len(p)),
            f"{tag}_n_games": int(ng),
            f"{tag}_poss_per_game": round(len(p) / ng, 4),
            f"{tag}_poss_per_team_game": round(len(p) / (2 * ng), 4),
            f"{tag}_tov": tov,
            f"{tag}_tov_pct": round(100.0 * tov / len(p), 4),
            f"{tag}_fta": int(p["fta"].sum()),
            f"{tag}_fga": int(fga),
            f"{tag}_fta_per_fga": round(float(p["fta"].sum()) / float(fga), 5),
            f"{tag}_points": int(p["points"].sum()),
            f"{tag}_tech_points": int(p["tech_points_off"].sum() + p["tech_points_def"].sum()),
        }
        for t in ("FT_trip_shooting", "FT_trip_bonus"):
            m = p["terminal_event"] == t
            d[f"{tag}_n_{t}"] = int(m.sum())
            d[f"{tag}_fta_{t}"] = int(p.loc[m, "fta"].sum())
        d[f"{tag}_fta_other_terminals"] = int(
            p.loc[~p["terminal_event"].isin(["FT_trip_shooting", "FT_trip_bonus"]), "fta"].sum())
        d[f"{tag}_n_ambiguous"] = int(p["ft_trip_ambiguous"].sum())
        d[f"{tag}_and_one"] = int(p["and_one"].sum())
        return d

    out.update(agg(a, "old"))
    out.update(agg(b, "new"))

    # ---- rows changed -----------------------------------------------------
    ka = a.groupby("game_id")[KEY_COLS].apply(lambda x: tuple(map(tuple, x.to_numpy())))
    kb = b.groupby("game_id")[KEY_COLS].apply(lambda x: tuple(map(tuple, x.to_numpy())))
    common = ka.index.intersection(kb.index)
    diff_games = [g for g in common if ka[g] != kb[g]]
    out["n_games_total"] = int(len(common))
    out["n_games_changed"] = int(len(diff_games))
    out["pct_games_changed"] = round(100.0 * len(diff_games) / len(common), 3)
    chg = set(diff_games)
    out["n_poss_in_changed_games_old"] = int(a["game_id"].isin(chg).sum())
    out["pct_poss_in_changed_games"] = round(100.0 * a["game_id"].isin(chg).mean(), 3)

    # per-row diff, on the rows that still line up 1:1 in a changed game
    n_row_diff = 0
    for g in diff_games:
        xa, xb = ka[g], kb[g]
        if len(xa) == len(xb):
            n_row_diff += sum(1 for u, v in zip(xa, xb) if u != v)
        else:
            n_row_diff += max(len(xa), len(xb))
    out["n_possession_rows_changed"] = int(n_row_diff)
    out["pct_possession_rows_changed"] = round(100.0 * n_row_diff / len(a), 5)
    return out


def trips_block() -> dict:
    old = pd.read_parquet(FT_DIR / "trips_v1_era.parquet")
    new = pd.read_parquet(FT_DIR / "trips_v1_era_techfix.parquet")
    tgt = pd.read_parquet(FT_DIR / "technical_target_verified_trips_v1.parquet")
    out = {}
    for s in SEASONS:
        o, nw = old[old["season"] == s], new[new["season"] == s]
        t = tgt[tgt["season"] == s]
        row = {"n_trips_old": int(len(o)), "n_trips_new": int(len(nw))}
        for cls in sorted(set(o["foul_class"]) | set(nw["foul_class"])):
            mo, mn = o["foul_class"] == cls, nw["foul_class"] == cls
            row[f"trips_{cls}_old"] = int(mo.sum())
            row[f"trips_{cls}_new"] = int(mn.sum())
            row[f"fta_{cls}_old"] = int(o.loc[mo, "trip_len"].sum())
            row[f"fta_{cls}_new"] = int(nw.loc[mn, "trip_len"].sum())
        row["verified_target_trips"] = int(len(t))
        row["verified_target_fta"] = int(t["n_attempts_verified"].sum())
        tech_o = int((o["foul_class"] == "technical").sum())
        tech_n = int((nw["foul_class"] == "technical").sum())
        row["recovery_old_pct_of_target"] = round(100.0 * tech_o / max(len(t), 1), 2)
        row["recovery_new_pct_of_target"] = round(100.0 * tech_n / max(len(t), 1), 2)
        # moment-level match against the verified target
        kt = set(zip(t["game_id"], t["period"], t["clock"], t["beneficiary"]))
        def moments(d):
            x = d[d["foul_class"] == "technical"]
            return set(zip(x["game_id"], x["period"], x["seconds_remaining"], x["team_id"]))
        mo, mn = moments(o), moments(nw)
        row["target_moments"] = len(kt)
        row["old_matched"] = len(mo & kt)
        row["new_matched"] = len(mn & kt)
        row["new_missed_vs_target"] = len(kt - mn)
        row["new_extra_vs_target"] = len(mn - kt)
        out[str(s)] = row
    return out


def attempts_block() -> dict:
    o = pd.read_parquet(FT_DIR / "attempts_v1_era.parquet",
                        columns=["season", "game_id", "trip_id", "trip_pos", "foul_class", "made"])
    n = pd.read_parquet(FT_DIR / "attempts_v1_era_techfix.parquet",
                        columns=["season", "game_id", "trip_id", "trip_pos", "foul_class", "made"])
    out = {}
    for s in SEASONS:
        a, b = o[o["season"] == s].reset_index(drop=True), n[n["season"] == s].reset_index(drop=True)
        if len(a) != len(b):
            out[str(s)] = {"row_count_moved": True, "n_old": len(a), "n_new": len(b)}
            continue
        moved = (a["foul_class"].to_numpy() != b["foul_class"].to_numpy())
        out[str(s)] = {
            "n_attempts": int(len(a)),
            "n_attempts_reclassified": int(moved.sum()),
            "pct_attempts_reclassified": round(100.0 * moved.mean(), 4),
            "from_class": a.loc[moved, "foul_class"].value_counts().to_dict(),
            "to_class": b.loc[moved, "foul_class"].value_counts().to_dict(),
            "make_rate_moved_attempts": round(float(a.loc[moved, "made"].mean()), 4),
            "make_rate_ft2_universe_old": round(
                float(a.loc[a["foul_class"] != "technical", "made"].mean()), 4),
            "make_rate_ft2_universe_new": round(
                float(a.loc[b["foul_class"].to_numpy() != "technical", "made"].mean()), 4),
        }
    return out


def main() -> int:
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rep: dict = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "possessions": {}}
    for s in SEASONS:
        if not (NEW / f"possessions_{s}.parquet").exists():
            print(f"season {s}: v3 not built yet, skipped", flush=True)
            continue
        rep["possessions"][str(s)] = season_block(s)
        print(f"[{time.time()-t0:6.1f}s] possessions season {s} done", flush=True)
    if (FT_DIR / "trips_v1_era_techfix.parquet").exists():
        rep["trips"] = trips_block()
        rep["attempts"] = attempts_block()
        print(f"[{time.time()-t0:6.1f}s] trips/attempts done", flush=True)
    (RESULTS / "blast_radius.json").write_text(json.dumps(rep, indent=1, default=str))

    # ---------------- printed tables ----------------
    p = pd.DataFrame(rep["possessions"]).T
    if len(p):
        cols = ["old_poss_per_game", "new_poss_per_game", "old_tov", "new_tov",
                "old_fta", "new_fta", "old_tech_points", "new_tech_points",
                "old_fta_per_fga", "new_fta_per_fga", "old_tov_pct", "new_tov_pct",
                "n_games_changed", "pct_games_changed", "n_possession_rows_changed",
                "pct_possession_rows_changed", "old_points", "new_points"]
        print("\n=== POSSESSIONS: v2 (old) vs v3 (new) ===")
        print(p[[c for c in cols if c in p.columns]].to_string())
        print("\n=== FTA by trip class (possession terminals) ===")
        print(p[[c for c in p.columns if "FT_trip" in c or "fta_other" in c]].to_string())
    if "trips" in rep:
        print("\n=== TRIPS: trips_v1_era vs trips_v1_era_techfix, and the verified target ===")
        print(pd.DataFrame(rep["trips"]).T.to_string())
        print("\n=== ATTEMPTS reclassified (FT-2's training universe) ===")
        print(pd.DataFrame(rep["attempts"]).T.to_string())
    print(f"\nwrote {RESULTS / 'blast_radius.json'} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
