"""diag_clock_v4_shortfall.py -- where the served clock arm's -0.34 s lives.

Round 3c (L31, `docs/models/clock/experiments.md` section 13.5) left an exact
account of the possession overshoot: the served arm
`empirical_km3_srfloor|P3|S1` produces a mean regulation possession duration of
17.216 s against 17.555 s actual on clock-complete games -- a uniform -1.93%
shortfall that implies +1.35 possessions per team-game against an observed
+1.161. This script fits no model and scores no arm. It answers three
descriptive questions before round 4 pre-registers anything:

  1. **The tiling identity.** Do the data's possessions tile the period
     exactly (`start_clock` of a possession == `end_clock` of the previous one,
     durations summing to the period length), and does the engine consume
     exactly that quantity? If both tile, mean consumed duration and possession
     count are two readings of ONE number and the diagnosis is a duration
     diagnosis.

  2. **The train/serve quantity.** Is the quantity the model was trained to
     predict (`duration_s` on the possession row) the quantity `loop.py`
     subtracts from the clock (`min(intended_draw, seconds_remaining)`)? Any
     mismatch is the prime suspect. Reported as an identity check on the data
     plus a row-set reconciliation: every possession the data tiles a period
     with, against every row the design keeps.

  3. **Where the shortfall lives.** On the REAL 2025 possessions, the served
     S1 artifacts routed by month, the model's own expected CONSUMED duration
       E[min(T, R)] = sum_{t<R} t p(t) + R P(T >= R)
     against the actual consumed duration, cell by cell: previous end type,
     first chance vs continuation (`n_chances`), shot-clock era, period, minute
     bucket, clock bucket, team tempo quintile, home/away/neutral, the
     offence's foul situation, and the terminal event. A shortfall that is
     uniform across every cell is a different defect from one concentrated in
     a few.

The offline read is on the real state sequence, so it isolates the MODEL. The
engine's own cell table (which measures the state COMPOSITION the engine
visits) is written by `scripts/diag_clock_v4_engine_cells.py`.

Usage:
    .venv/Scripts/python.exe scripts/diag_clock_v4_shortfall.py \
        --season 2025 --mode v3c_srfloor_P3_s1 --out data/processed/models/clock/v4_diag
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import clock as CK  # noqa: E402

CK_DIR = ROOT / "data" / "processed" / "models" / "clock"
POSS_DIR = ROOT / "data" / "processed" / "possessions_v2"
UNIV_V2 = ROOT / "data" / "processed" / "games_universe_v2.parquet"
SLATE = ROOT / "data" / "processed" / "models" / "engine" / "games_F2_2025.parquet"
DESIGN = CK_DIR / "design_v2.parquet"

CHUNK = 40_000


# ---------------------------------------------------------------------------
# 1. the tiling identity, straight off the possession table
# ---------------------------------------------------------------------------
def tiling_check(poss: pd.DataFrame, cc_games: set[int]) -> dict:
    """Does the data tile each period exactly, and is `duration_s` that tile?"""
    p = poss.sort_values(["game_id", "period", "poss_index"], kind="stable")
    out: dict = {"n_rows": int(len(p))}

    d = (p["start_clock"] - p["end_clock"]).clip(lower=0)
    out["duration_equals_start_minus_end"] = bool((d == p["duration_s"]).all())
    out["n_duration_mismatch"] = int((d != p["duration_s"]).sum())

    # start_clock of row i+1 == end_clock of row i, within (game, period)
    g = p.groupby(["game_id", "period"], sort=False)
    prev_end_clock = g["end_clock"].shift(1)
    same = prev_end_clock.notna()
    out["n_interior_boundaries"] = int(same.sum())
    out["n_boundary_gaps"] = int((p.loc[same, "start_clock"] != prev_end_clock[same]).sum())
    out["boundary_gap_seconds_total"] = float(
        (prev_end_clock[same] - p.loc[same, "start_clock"]).abs().sum())

    # per regulation period on CLOCK-COMPLETE games: summed durations vs 1200
    reg = p[(p["period"] <= 2) & (p["game_id"].isin(cc_games))]
    per = reg.groupby(["game_id", "period"], as_index=False).agg(
        dur=("duration_s", "sum"), n=("duration_s", "size"),
        first_start=("start_clock", "max"), last_end=("end_clock", "min"))
    out["cc_periods"] = int(len(per))
    out["cc_period_seconds_mean"] = float(per["dur"].mean())
    out["cc_period_seconds_deficit_mean"] = float(1200.0 - per["dur"].mean())
    out["cc_periods_exactly_1200"] = int((per["dur"] == 1200).sum())
    out["cc_mean_possessions_per_period"] = float(per["n"].mean())
    out["cc_mean_duration_from_tiling"] = float(per["dur"].sum() / per["n"].sum())
    out["cc_mean_duration_row_mean"] = float(reg["duration_s"].mean())
    # the count a perfect duration law implies, and the count actually observed
    out["cc_implied_possessions_per_team_game"] = float(
        2 * 1200.0 / out["cc_mean_duration_from_tiling"] / 2.0)
    return out


def rowset_reconciliation(poss: pd.DataFrame, design: pd.DataFrame,
                          cc_games: set[int]) -> dict:
    """Every possession the data tiles a period with, against every design row.

    A possession the design DROPS still consumed clock in the real game, so its
    seconds are time the engine will re-tile with modelled possessions. This is
    the only channel by which the training row set can bias the count."""
    reg = poss[(poss["period"] <= 2) & (poss["game_id"].isin(cc_games))]
    key = ["game_id", "period", "poss_index"]
    dkey = design[design["period"] <= 2.0][key].copy()
    dkey["_in_design"] = True
    m = reg.merge(dkey, on=key, how="left")
    ind = m["_in_design"].fillna(False).astype(bool).to_numpy()
    kept, drop = m[ind], m[~ind]
    return {
        "cc_reg_possessions": int(len(m)),
        "in_design": int(len(kept)),
        "dropped": int(len(drop)),
        "dropped_pct": round(100.0 * len(drop) / max(len(m), 1), 5),
        "dropped_seconds_total": float(drop["duration_s"].sum()),
        "dropped_mean_duration": float(drop["duration_s"].mean()) if len(drop) else 0.0,
        "kept_mean_duration": float(kept["duration_s"].mean()),
        "all_mean_duration": float(m["duration_s"].mean()),
        "mean_shift_from_dropping": float(kept["duration_s"].mean() - m["duration_s"].mean()),
        "dropped_by_terminal": drop["terminal_event"].value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# 2. the served arm, routed by month, and its expected CONSUMED duration
# ---------------------------------------------------------------------------
def load_schedule(mode: str) -> list[dict]:
    from cbb_sim.engine.clock_adapter_v3 import V3C_MODES
    spec = V3C_MODES[mode]
    doc = json.loads((CK_DIR / spec["manifest"]).read_text(encoding="utf-8"))
    out = []
    for m in doc["months"]:
        with open(CK_DIR / m["model_file"], "rb") as f:
            out.append({"refit_date": pd.Timestamp(m["refit_date"]),
                        "arm": pickle.load(f), "month": m.get("month")})
    out.sort(key=lambda r: r["refit_date"])
    return out


def expected_consumed(arm, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(E[min(T,R)], E[T]) per row, from the arm's own predictive pmf.

    `R` is the seconds that were left at the possession's start, so E[min(T,R)]
    is exactly the quantity `loop.py` subtracts from the clock."""
    grid = np.arange(CK.DURATION_CAP + 1, dtype=np.float64)
    e_min = np.empty(len(df), dtype=np.float64)
    e_t = np.empty(len(df), dtype=np.float64)
    for lo in range(0, len(df), CHUNK):
        blk = df.iloc[lo:lo + CHUNK]
        pmf = np.asarray(arm.pmf(blk.reset_index(drop=True)), dtype=np.float64)
        r = blk["seconds_remaining"].to_numpy(dtype=np.float64)[:, None]
        e_t[lo:lo + len(blk)] = pmf @ grid
        e_min[lo:lo + len(blk)] = (pmf * np.minimum(grid[None, :], r)).sum(axis=1)
    return e_min, e_t


# ---------------------------------------------------------------------------
# 3. cells
# ---------------------------------------------------------------------------
MINUTE_EDGES = np.array([0, 60, 120, 300, 600, 900, 1201])
MINUTE_LABELS = ["0-1m", "1-2m", "2-5m", "5-10m", "10-15m", "15-20m"]


def add_cells(d: pd.DataFrame, poss: pd.DataFrame) -> pd.DataFrame:
    key = ["game_id", "period", "poss_index"]
    extra = poss[key + ["n_chances", "oreb_count", "terminal_event", "off_team_fouls",
                        "off_in_bonus", "off_in_double_bonus", "is_transition"]]
    d = d.merge(extra, on=key, how="left", suffixes=("", "_p"))
    d["chance_kind"] = np.where(d["n_chances"].fillna(1) > 1, "continuation(OREB)", "first only")
    d["minute_bucket"] = pd.cut(d["seconds_remaining"], bins=MINUTE_EDGES, right=False,
                                labels=MINUTE_LABELS).astype("str")
    d["period_cell"] = np.where(d["period"] <= 1.0, "H1",
                                np.where(d["period"] <= 2.0, "H2", "OT"))
    d["site"] = np.where(d["site_home"] > 0, "home",
                         np.where(d["site_away"] > 0, "away", "neutral"))
    d["foul_situation"] = np.where(d["off_in_double_bonus"].fillna(0) > 0, "double bonus",
                                   np.where(d["in_bonus"] > 0, "bonus", "no bonus"))
    d["shot_clock_era"] = "30s (2022-2026, no boundary in window)"
    d["terminal_cell"] = d["terminal_event_p"].fillna(d["terminal_event"])
    return d


CELLS = ["prev_end", "chance_kind", "period_cell", "minute_bucket", "r2_bucket",
         "tempo_quintile", "site", "foul_situation", "terminal_cell", "score_state",
         "shot_clock_era", "month"]


def cell_table(d: pd.DataFrame, by: str) -> pd.DataFrame:
    g = d.groupby(by, dropna=False)
    t = g.agg(n=("duration_s", "size"),
              actual_mean=("duration_s", "mean"),
              model_mean=("e_min", "mean"),
              model_intended_mean=("e_t", "mean"),
              actual_seconds=("duration_s", "sum"),
              model_seconds=("e_min", "sum")).reset_index()
    t["gap_s"] = t["model_mean"] - t["actual_mean"]
    t["gap_pct"] = 100.0 * t["gap_s"] / t["actual_mean"]
    t["share_of_possessions"] = t["n"] / t["n"].sum()
    t["share_of_total_gap"] = (t["model_seconds"] - t["actual_seconds"]) / (
        d["e_min"].sum() - d["duration_s"].sum())
    t["underpowered"] = t["n"] < 300
    return t.sort_values(by)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--mode", default="v3c_srfloor_P3_s1")
    ap.add_argument("--out", default="data/processed/models/clock/v4_diag")
    a = ap.parse_args()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    univ = pd.read_parquet(UNIV_V2)
    cc = set(univ.loc[univ["clock_complete_reg"].fillna(False), "game_id"].astype(int))
    slate = pd.read_parquet(SLATE).sort_values("game_id", kind="stable")
    subset = set(slate["game_id"].astype(int).to_numpy()[::11][:500])

    poss = pd.read_parquet(POSS_DIR / f"possessions_{a.season}.parquet")
    design = pd.read_parquet(DESIGN)
    design = design[design["season"] == a.season].copy()

    report: dict = {"season": a.season, "mode": a.mode,
                    "clock_complete_games_in_season": len(cc & set(poss["game_id"].astype(int))),
                    "subset_games": len(subset), "subset_clock_complete": len(subset & cc)}
    report["tiling"] = tiling_check(poss, cc)
    report["rowset"] = rowset_reconciliation(poss, design, cc)

    sched = load_schedule(a.mode)
    cuts = np.array([np.datetime64(r["refit_date"], "ns") for r in sched])

    for label, games in (("subset500", subset & cc), ("all_cc", cc)):
        d = design[design["game_id"].isin(games) & (design["period"] <= 2.0)].copy()
        if not len(d):
            continue
        d["season"] = a.season
        gd = pd.to_datetime(d["game_date"]).to_numpy()
        seg = np.searchsorted(cuts, gd, side="right") - 1
        assert (seg >= 0).all(), "a game tips before the first refit date"
        d["_seg"] = seg
        e_min = np.empty(len(d)); e_t = np.empty(len(d))
        for k in np.unique(seg):
            r = np.flatnonzero(seg == k)
            em, et = expected_consumed(sched[int(k)]["arm"], d.iloc[r])
            e_min[r] = em; e_t[r] = et
        d["e_min"] = e_min
        d["e_t"] = e_t

        tq = d.groupby("game_id")["tempo_prior_game"].first()
        qs = pd.qcut(tq, 5, labels=[f"Q{i}" for i in range(1, 6)])
        d["tempo_quintile"] = d["game_id"].map(qs).astype("str")
        d["r2_bucket"] = np.asarray(CK.R2_SR_LABELS, dtype=object)[
            CK.r2_bucket_id(d["seconds_remaining"].to_numpy())]
        d = add_cells(d, poss)

        tot = {
            "n_possessions": int(len(d)),
            "n_games": int(d["game_id"].nunique()),
            "actual_mean_duration": float(d["duration_s"].mean()),
            "model_mean_consumed": float(d["e_min"].mean()),
            "gap_s": float(d["e_min"].mean() - d["duration_s"].mean()),
            "gap_pct": float(100.0 * (d["e_min"].mean() - d["duration_s"].mean())
                             / d["duration_s"].mean()),
            "model_mean_intended": float(d["e_t"].mean()),
            "actual_seconds": float(d["duration_s"].sum()),
            "model_seconds": float(d["e_min"].sum()),
            "implied_poss_per_team_game_actual": float(
                d["duration_s"].sum() / d["duration_s"].mean() / d["game_id"].nunique() / 2),
            "implied_poss_per_team_game_model": float(
                d["duration_s"].sum() / d["e_min"].mean() / d["game_id"].nunique() / 2),
        }
        tot["implied_delta_possessions"] = (tot["implied_poss_per_team_game_model"]
                                            - tot["implied_poss_per_team_game_actual"])
        report[f"overall_{label}"] = tot

        for by in CELLS:
            t = cell_table(d, by)
            t.to_csv(out / f"v4_cells_{label}_{by}.csv", index=False)
        d[["game_id", "period", "poss_index", "duration_s", "e_min", "e_t",
           "seconds_remaining", "prev_end", "terminal_cell", "chance_kind"]].to_parquet(
            out / f"v4_rows_{label}.parquet", index=False)

    (out / "v4_diag_report.json").write_text(json.dumps(report, indent=2, default=str),
                                             encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("overall_")},
                     indent=2, default=str))
    for k, v in report.items():
        if k.startswith("overall_"):
            print(k, json.dumps(v, indent=2))


if __name__ == "__main__":
    main()
