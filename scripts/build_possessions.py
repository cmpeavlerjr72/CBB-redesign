#!/usr/bin/env python
"""
build_possessions.py -- build the L3 possession event layer from CBBD pbp.

    .venv/Scripts/python.exe scripts/build_possessions.py
    .venv/Scripts/python.exe scripts/build_possessions.py --seasons 2025
    .venv/Scripts/python.exe scripts/build_possessions.py --validate-only

Writes, per season:
    data/processed/possessions/possessions_{season}.parquet   one row / possession
    data/processed/possessions/chances_{season}.parquet       one row / chance
    data/processed/possessions/build_report.json              validation numbers

The segmentation rule itself lives in `src/cbb_sim/pbp/possessions.py` (module
docstring) and the event vocabulary in `src/cbb_sim/pbp/events.py`. This script
is the CLI plus the validation battery written up in
`docs/tests/possessions_build_2026-09-10.md`:

  1. possessions per game vs the box formula FGA - OREB + TOV + 0.44*FTA, per
     season: mean difference, SD, correlation, and the 20 worst games.
  2. points accumulated from possessions vs the schedule's final score:
     mismatch count per season.
  3. terminal-event shares per season.
  4. duration distribution by terminal event.

SEAL. Season 2026 IS built here -- a possession table is data, not a fit, the
same standing this project already gives the descriptive gate-reference
tables. Nothing in this script fits or selects anything. Every training and
selection path that reads these files calls `assert_not_sealed`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.pbp import possessions as poss_mod  # noqa: E402
from cbb_sim.pbp.events import classify_frame, load_plays, three_point_signal_disagreement  # noqa: E402
from cbb_sim.pbp.report import render_doc  # noqa: E402

DEFAULT_SEASONS = [2022, 2023, 2024, 2025, 2026]
DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")
DEFAULT_OUT_DIR = Path("data/processed/possessions")
DEFAULT_PBP_DIR = Path("data/raw/cbbd/pbp")


def load_universe(path: Path = DEFAULT_UNIVERSE) -> pd.DataFrame:
    u = pd.read_parquet(path)
    u = u[u["is_d1_game"] & ~u["pbp_truncated"] & u["cbbd_game_id"].notna()].copy()
    u["cbbd_game_id"] = u["cbbd_game_id"].astype("int64")
    return u


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def box_formula_possessions(poss: pd.DataFrame) -> pd.DataFrame:
    """Per (game, team) box-formula possessions from the possession table's own
    accumulated box line: FGA - OREB + TOV + 0.44*FTA.

    Using our own accumulated box line rather than hoopR team_box keeps the
    comparison a comparison of DEFINITIONS (event segmentation vs the box
    approximation) instead of a comparison of data sources."""
    p = poss.copy()
    p["fga"] = p["fga_rim"] + p["fga_jump2"] + p["fga_3"]
    p["tov"] = (p["terminal_event"] == "TOV").astype(int)
    g = p.groupby(["game_id", "offense_team_id"], as_index=False).agg(
        fga=("fga", "sum"), oreb=("oreb_count", "sum"), tov=("tov", "sum"),
        fta=("fta", "sum"), poss_seg=("poss_index", "count"),
        points=("points", "sum"), tech_off=("tech_points_off", "sum"),
    )
    g["poss_box"] = g["fga"] - g["oreb"] + g["tov"] + 0.44 * g["fta"]
    return g


def feed_points(season: int, universe: pd.DataFrame, pbp_dir: Path) -> pd.DataFrame:
    """Points implied by the raw CBBD event stream itself, per (game_id, team).

    This separates two different failures that a naive "possession points vs
    final score" check conflates:

      * FEED COMPLETENESS -- the CBBD event stream for a game simply does not
        contain every scoring play. `games_universe.pbp_truncated` is computed
        from the hoopR feed, not this one, so it does not catch these.
      * SEGMENTATION FIDELITY -- our state machine failing to attribute a
        scoring play that IS in the stream to a possession.

    The first is a data fact to be reported; only the second is a bug.
    """
    u = universe[universe["season"] == int(season)]
    meta = u.set_index("cbbd_game_id")[["game_id", "home_team_id", "away_team_id"]]
    plays = load_plays(season, pbp_dir=pbp_dir, game_ids=set(meta.index))
    cls = classify_frame(plays)
    made = plays["shot_made"]
    if made.dtype == object:
        made = made.map({True: True, False: False})
    made = made.astype("boolean").fillna(plays["scoringPlay"].astype("boolean")).fillna(False).to_numpy(dtype=bool)
    pv = np.where(cls.to_numpy() == "FGA_3", 3,
                  np.where(np.isin(cls.to_numpy(), ["FGA_rim", "FGA_jump2"]), 2,
                           np.where(cls.to_numpy() == "FT_made", 1, 0)))
    pv = np.where(np.isin(cls.to_numpy(), ["FGA_3", "FGA_rim", "FGA_jump2"]) & ~made, 0, pv)

    is_home = plays["isHomeTeam"]
    if is_home.dtype == object:
        is_home = is_home.map({True: True, False: False})
    is_home = is_home.astype("boolean")
    has_team = pd.to_numeric(plays["teamId"], errors="coerce").notna() & is_home.notna()
    side = np.where(is_home.fillna(False).to_numpy(), 0, 1)
    side = poss_mod._fix_flipped_sides(plays, side, has_team.to_numpy())

    df = pd.DataFrame({"cbbd_game_id": plays["gameId"].to_numpy(), "side": side, "pv": pv})
    df = df[has_team.to_numpy() & (df["pv"] > 0)]
    agg = df.groupby(["cbbd_game_id", "side"], as_index=False)["pv"].sum()
    agg = agg.merge(meta.reset_index(), on="cbbd_game_id", how="left")
    agg["offense_team_id"] = np.where(agg["side"] == 0, agg["home_team_id"], agg["away_team_id"])
    return agg[["game_id", "offense_team_id", "pv"]].rename(columns={"pv": "feed_points"})


def putback_label_drift(season: int, universe: pd.DataFrame, pbp_dir: Path) -> dict:
    """Share of each shot `playType` on the attempt that immediately follows an
    offensive rebound by the same team -- the putback.

    This exists because the possession build surfaced a discontinuity that is
    NOT a segmentation artefact: the terminal-event mix of continuation chances
    moves by 6.5 pp between 2024 and 2025 while the first-chance mix moves by
    less than 2 pp. Tracing it to the raw feed shows an upstream ESPN
    relabelling of putbacks, which is exactly the "event vocabulary drift"
    risk `docs/SIM_GUARDRAILS.md` section 4 names. Reported, not corrected."""
    u = universe[universe["season"] == int(season)]
    plays = load_plays(season, pbp_dir=pbp_dir, game_ids=set(u["cbbd_game_id"]),
                       columns=("gameId", "id", "playType", "teamId", "shot_range"))
    pt = plays["playType"].to_numpy()
    tm = pd.to_numeric(plays["teamId"], errors="coerce").to_numpy()
    g = plays["gameId"].to_numpy()
    same_game = np.concatenate([[False], g[1:] == g[:-1]])
    prev_oreb = np.concatenate([[False], pt[:-1] == "Offensive Rebound"]) & same_game
    same_team = np.concatenate([[False], tm[1:] == tm[:-1]]) & same_game
    shots = np.isin(pt, ["JumpShot", "LayUpShot", "DunkShot", "TipShot"])
    sel = prev_oreb & same_team & shots
    if sel.sum() == 0:
        return {}
    lab = pd.Series(pt[sel]).astype("string")
    rng = plays["shot_range"].astype("string").str.lower().to_numpy()[sel]
    lab = lab.where(lab != "JumpShot",
                    pd.Series(np.where(rng == "three_pointer", "JumpShot (3)", "JumpShot (2)")))
    share = (lab.value_counts(normalize=True) * 100).round(2).to_dict()
    return {"n_putback_attempts": int(sel.sum()), "shares_pct": share}


def validate_season(season: int, poss: pd.DataFrame, universe: pd.DataFrame,
                    feed: pd.DataFrame | None = None) -> dict:
    u = universe[universe["season"] == int(season)].set_index("game_id")
    g = box_formula_possessions(poss)

    # --- 1. possessions per game: segmentation vs box formula ---------------
    per_game = g.groupby("game_id").agg(poss_seg=("poss_seg", "mean"), poss_box=("poss_box", "mean"))
    per_game["diff"] = per_game["poss_seg"] - per_game["poss_box"]
    worst = per_game.reindex(per_game["diff"].abs().sort_values(ascending=False).index).head(20)
    worst_rows = []
    for gid, r in worst.iterrows():
        meta = u.loc[gid] if gid in u.index else None
        worst_rows.append({
            "game_id": int(gid),
            "poss_seg": round(float(r["poss_seg"]), 2),
            "poss_box": round(float(r["poss_box"]), 2),
            "diff": round(float(r["diff"]), 2),
            "matchup": (f"{meta['away_display_name']} @ {meta['home_display_name']}" if meta is not None else ""),
            "date": (str(meta["game_date"]) if meta is not None else ""),
            "n_periods": (float(meta["n_periods"]) if meta is not None else float("nan")),
        })

    # --- 2. points from possessions vs the schedule final score -------------
    pts = g.groupby(["game_id", "offense_team_id"])[["points", "tech_off"]].sum().reset_index()
    # technical FTs shot by the DEFENCE while a possession was open
    tech_def = poss.groupby(["game_id", "defense_team_id"])["tech_points_def"].sum().reset_index()
    tech_def = tech_def.rename(columns={"defense_team_id": "offense_team_id", "tech_points_def": "tech_def"})
    pts = pts.merge(tech_def, on=["game_id", "offense_team_id"], how="left").fillna({"tech_def": 0})
    pts["scored"] = pts["points"] + pts["tech_off"] + pts["tech_def"]

    truth = pd.concat([
        u.reset_index()[["game_id", "home_team_id", "home_score"]].rename(
            columns={"home_team_id": "offense_team_id", "home_score": "final"}),
        u.reset_index()[["game_id", "away_team_id", "away_score"]].rename(
            columns={"away_team_id": "offense_team_id", "away_score": "final"}),
    ], ignore_index=True)
    # Games absent from the CBBD pbp extract entirely (the audit counts
    # 37-157 such games per season) cannot be reconciled and are reported
    # separately rather than counted as a segmentation failure.
    built_games = set(poss["game_id"].unique())
    missing_games = sorted(set(u.index) - built_games)
    truth = truth[truth["game_id"].isin(built_games)]
    cmp_ = truth.merge(pts[["game_id", "offense_team_id", "scored"]], on=["game_id", "offense_team_id"], how="left")
    cmp_["scored"] = cmp_["scored"].fillna(0)
    cmp_["delta"] = cmp_["scored"] - cmp_["final"]
    bad_games = cmp_.loc[cmp_["delta"] != 0, "game_id"].nunique()

    feed_stats = {}
    if feed is not None:
        f = cmp_.merge(feed, on=["game_id", "offense_team_id"], how="left")
        f["feed_points"] = f["feed_points"].fillna(0)
        f["feed_vs_final"] = f["feed_points"] - f["final"]
        f["seg_vs_feed"] = f["scored"] - f["feed_points"]
        feed_stats = {
            "feed_vs_final_mismatch_rows": int((f["feed_vs_final"] != 0).sum()),
            "feed_vs_final_mismatch_games": int(f.loc[f["feed_vs_final"] != 0, "game_id"].nunique()),
            "feed_vs_final_mean_abs": round(float(f["feed_vs_final"].abs().mean()), 4),
            "seg_vs_feed_mismatch_rows": int((f["seg_vs_feed"] != 0).sum()),
            "seg_vs_feed_mismatch_games": int(f.loc[f["seg_vs_feed"] != 0, "game_id"].nunique()),
            "seg_vs_feed_mean_abs": round(float(f["seg_vs_feed"].abs().mean()), 5),
            "seg_vs_feed_hist": {str(k): int(v) for k, v in
                                 f["seg_vs_feed"].clip(-5, 5).value_counts().sort_index().items()},
        }

    # --- 3. terminal-event shares ------------------------------------------
    shares = (poss["terminal_event"].value_counts(normalize=True) * 100).round(3).to_dict()
    chance_note = {}

    # --- 4. duration by terminal event -------------------------------------
    dur = poss.groupby("terminal_event")["duration_s"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    dur_tbl = {k: {kk: round(float(vv), 2) for kk, vv in v.items()} for k, v in dur.to_dict("index").items()}

    return {
        "season": int(season),
        "n_games": int(poss["game_id"].nunique()),
        "n_possessions": int(len(poss)),
        "poss_per_game_seg_mean": round(float(per_game["poss_seg"].mean()), 3),
        "poss_per_game_seg_sd": round(float(per_game["poss_seg"].std()), 3),
        "poss_per_game_box_mean": round(float(per_game["poss_box"].mean()), 3),
        "poss_per_game_box_sd": round(float(per_game["poss_box"].std()), 3),
        "diff_mean": round(float(per_game["diff"].mean()), 4),
        "diff_sd": round(float(per_game["diff"].std()), 4),
        "diff_corr": round(float(per_game["poss_seg"].corr(per_game["poss_box"])), 5),
        "diff_mae": round(float(per_game["diff"].abs().mean()), 4),
        "worst_20": worst_rows,
        "n_universe_games": int(len(u)),
        "n_games_missing_from_cbbd": int(len(missing_games)),
        "score_mismatch_team_rows": int((cmp_["delta"] != 0).sum()),
        "score_mismatch_games": int(bad_games),
        **feed_stats,
        "score_rows": int(len(cmp_)),
        "score_mean_abs_delta": round(float(cmp_["delta"].abs().mean()), 4),
        "score_delta_hist": {str(k): int(v) for k, v in
                             cmp_["delta"].clip(-5, 5).value_counts().sort_index().items()},
        "terminal_shares_pct": shares,
        "chance_note": chance_note,
        "duration_by_terminal": dur_tbl,
        "mean_chances": round(float(poss["n_chances"].mean()), 4),
        "oreb_rate": round(float((poss["oreb_count"] > 0).mean()), 4),
        "ambiguous_ft_trips_pct": round(
            float(poss.loc[poss["terminal_event"].isin(["FT_trip_bonus", "FT_trip_shooting"]),
                           "ft_trip_ambiguous"].mean() * 100), 3),
        "transition_share_pct": round(float(poss["is_transition"].mean() * 100), 3),
        "bonus_share_pct": round(float(poss["off_in_bonus"].mean() * 100), 3),
        "and_one_per_game": round(float(poss["and_one"].sum() / poss["game_id"].nunique()), 4),
    }


def validate_chances(chances: pd.DataFrame) -> dict:
    first = chances[chances["chance_number"] == 1]
    cont = chances[chances["chance_number"] > 1]
    return {
        "n_chances": int(len(chances)),
        "n_first": int(len(first)),
        "n_continuation": int(len(cont)),
        "first_shares_pct": (first["terminal_event"].value_counts(normalize=True) * 100).round(3).to_dict(),
        "cont_shares_pct": (cont["terminal_event"].value_counts(normalize=True) * 100).round(3).to_dict()
        if len(cont) else {},
    }


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--pbp-dir", type=Path, default=DEFAULT_PBP_DIR)
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--write-doc", action="store_true",
                    help="render docs/tests/possessions_build_2026-09-10.md from build_report.json")
    ap.add_argument("--validate-only", action="store_true",
                    help="re-run the validation battery off existing parquet files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    universe = load_universe(args.universe)

    report: dict = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seasons": {}}
    rp = out_dir / "build_report.json"
    if rp.exists():
        with contextlib.suppress(json.JSONDecodeError):
            report = json.loads(rp.read_text())
    report.setdefault("seasons", {})

    for season in args.seasons:
        t0 = time.time()
        pp = out_dir / f"possessions_{season}.parquet"
        cp = out_dir / f"chances_{season}.parquet"
        if args.validate_only:
            poss = pd.read_parquet(pp)
            chances = pd.read_parquet(cp)
            diag = report["seasons"].get(str(season), {}).get("machine_diag", {})
            sig = report["seasons"].get(str(season), {}).get("three_point_signal", {})
        else:
            print(f"[{season}] segmenting ...", flush=True)
            poss, chances, diag = poss_mod.segment_season(season, universe, pbp_dir=args.pbp_dir)
            raw = load_plays(season, pbp_dir=args.pbp_dir)
            sig = three_point_signal_disagreement(raw)
            del raw
            poss.to_parquet(pp, index=False)
            chances.to_parquet(cp, index=False)
            print(f"[{season}] {len(poss):,} possessions, {len(chances):,} chances, "
                  f"{time.time() - t0:.1f}s -> {pp}", flush=True)

        feed = feed_points(season, universe, Path(args.pbp_dir))
        v = validate_season(season, poss, universe, feed=feed)
        v["putback_label_drift"] = putback_label_drift(season, universe, Path(args.pbp_dir))
        v["machine_diag"] = diag
        v["three_point_signal"] = sig
        v["chances"] = validate_chances(chances)
        report["seasons"][str(season)] = v
        print(f"[{season}] poss/game seg {v['poss_per_game_seg_mean']:.2f} vs box "
              f"{v['poss_per_game_box_mean']:.2f} (diff {v['diff_mean']:+.3f}, sd {v['diff_sd']:.3f}, "
              f"corr {v['diff_corr']:.4f}); score mismatch games {v['score_mismatch_games']} "
              f"(of which feed-incomplete {v.get('feed_vs_final_mismatch_games', 0)}; "
              f"segmentation-only {v.get('seg_vs_feed_mismatch_games', 0)})", flush=True)

    rp.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {rp}")
    if args.write_doc:
        print(f"wrote {render_doc(report)}")


if __name__ == "__main__":
    main()
