#!/usr/bin/env python
"""
build_possessions.py -- build the L3 possession event layer from CBBD pbp.

    .venv/Scripts/python.exe scripts/build_possessions.py
    .venv/Scripts/python.exe scripts/build_possessions.py --seasons 2025
    .venv/Scripts/python.exe scripts/build_possessions.py --validate-only

    .venv/Scripts/python.exe scripts/build_possessions.py --version v2
    .venv/Scripts/python.exe scripts/build_possessions.py --threshold-ladder

Writes, per season, into the directory the `--version` label resolves to
(`cbb_sim.pbp.possessions.POSSESSION_VERSIONS`):
    possessions_{season}.parquet   one row / possession
    chances_{season}.parquet       one row / chance
    build_report.json              validation numbers

VERSIONS. `--version` DEFAULTS TO `v2`, not to v1. v1
(`data/processed/possessions/`) is frozen: it is the table other workers hold
long-running reads on, and a bare re-run must never overwrite it. Building v1
again requires asking for it by name.

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

from cbb_sim.pbp import events as EV  # noqa: E402
from cbb_sim.pbp import possessions as poss_mod  # noqa: E402
from cbb_sim.pbp.events import classify_frame, load_plays, three_point_signal_disagreement  # noqa: E402
from cbb_sim.pbp.report import render_doc  # noqa: E402

DEFAULT_SEASONS = [2022, 2023, 2024, 2025, 2026]
DEFAULT_UNIVERSE = Path("data/processed/games_universe.parquet")
DEFAULT_VERSION = "v2"
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


def feed_completeness(season: int, universe_all: pd.DataFrame, pbp_dir: Path) -> dict:
    """Reconcile the two feed-completeness numbers that were in circulation,
    and report the `pbp_complete` share.

      A) EVENT SUM vs final -- `possessions_build_2026-09-10.md` section 2's
         19.5% for 2022. The classified scoring events do not add up to the
         box score.
      B) RUNNING SCORE vs final -- `shot_classification_diag_2026-09-10.md`
         section 8's 3.6% for 2022. The feed's own last `homeScore`/`awayScore`
         does not reach the box score.

    The candidate explanations are each measured rather than argued: points on
    rows with no usable team, `Not Available` rows flagged `scoringPlay`,
    field-goal rows flagged `scoringPlay` whose made flag says otherwise, and
    technical free throws (which are counted on BOTH sides of A, so they cannot
    open a gap -- `n_technical_ft_points` is reported to show their size).
    Whatever those four do not explain is the residual: scoring ROWS absent
    from the stream while the running-score column, carried forward on every
    later row, still reaches the final."""
    u = universe_all[(universe_all["season"] == int(season))
                     & universe_all["is_d1_game"] & universe_all["cbbd_game_id"].notna()].copy()
    u["cbbd_game_id"] = u["cbbd_game_id"].astype("int64")
    plays = load_plays(season, pbp_dir=pbp_dir, game_ids=set(u["cbbd_game_id"]))
    cls = classify_frame(plays).to_numpy()

    made = plays["shot_made"]
    if made.dtype == object:
        made = made.map({True: True, False: False})
    made = made.astype("boolean").fillna(plays["scoringPlay"].astype("boolean")).fillna(False).to_numpy(dtype=bool)
    pv = np.where(cls == "FGA_3", 3, np.where(np.isin(cls, ["FGA_rim", "FGA_jump2"]), 2,
                  np.where(cls == "FT_made", 1, 0)))
    pv = np.where(np.isin(cls, ["FGA_3", "FGA_rim", "FGA_jump2"]) & ~made, 0, pv)

    is_home = plays["isHomeTeam"]
    if is_home.dtype == object:
        is_home = is_home.map({True: True, False: False})
    is_home = is_home.astype("boolean")
    has_team = (pd.to_numeric(plays["teamId"], errors="coerce").notna() & is_home.notna()).to_numpy()
    side = np.where(is_home.fillna(False).to_numpy(), 0, 1)
    side = poss_mod._fix_flipped_sides(plays, side, has_team)

    # technical free throws: the FT rows whose preceding non-inert row is a
    # technical foul. Sized here only to show they are not the explanation.
    keep = ~pd.Series(cls).isin(list(poss_mod.INERT_CLASSES)).to_numpy()
    c_keep = cls[keep]
    prev_tech = np.concatenate([[False], c_keep[:-1] == "technical"])
    n_tech_ft_pts = int((prev_tech & (c_keep == "FT_made")).sum())

    scoring_flag = plays["scoringPlay"].fillna(False).astype(bool).to_numpy()
    g = plays["gameId"].to_numpy()
    df = pd.DataFrame({
        "cbbd_game_id": g,
        "home_pts": np.where(has_team & (side == 0), pv, 0),
        "away_pts": np.where(has_team & (side == 1), pv, 0),
        "pts_no_team": np.where(~has_team, pv, 0),
        "unknown_scoring_rows": ((cls == "unknown") & scoring_flag).astype("int64"),
        "fga_scoringplay_not_made": (np.isin(cls, ["FGA_3", "FGA_rim", "FGA_jump2"])
                                     & scoring_flag & ~made).astype("int64"),
        "rows": 1,
    }).groupby("cbbd_game_id", as_index=True).sum()
    last = pd.DataFrame({
        "cbbd_game_id": g,
        "hs": pd.to_numeric(plays["homeScore"], errors="coerce").ffill().fillna(0).to_numpy(),
        "as_": pd.to_numeric(plays["awayScore"], errors="coerce").ffill().fillna(0).to_numpy(),
    }).groupby("cbbd_game_id").last()

    m = u[["cbbd_game_id", "game_id", "home_score", "away_score", "pbp_truncated"]].join(
        df, on="cbbd_game_id").join(last, on="cbbd_game_id")
    has_rows = m["rows"].notna().to_numpy()
    for c in ("home_pts", "away_pts", "pts_no_team", "unknown_scoring_rows",
              "fga_scoringplay_not_made", "hs", "as_"):
        m[c] = m[c].fillna(0)
    ev_bad = ((m["home_pts"] != m["home_score"]) | (m["away_pts"] != m["away_score"])).to_numpy()
    run_bad = ((m["hs"] != m["home_score"]) | (m["as_"] != m["away_score"])).to_numpy()
    ev_vs_run = (m["hs"] + m["as_"] - m["home_pts"] - m["away_pts"]).to_numpy()
    n = len(m)
    complete = has_rows & ~ev_bad
    return {
        "n_d1_games": int(n),
        "n_no_cbbd_rows": int((~has_rows).sum()),
        "A_event_sum_vs_final_incomplete_pct": round(100.0 * float(ev_bad.mean()), 2),
        "B_running_score_vs_final_incomplete_pct": round(100.0 * float(run_bad.mean()), 2),
        "events_disagree_with_running_score_pct": round(100.0 * float((ev_vs_run != 0).mean()), 2),
        "mean_running_minus_events_pts": round(float(ev_vs_run.mean()), 3),
        "explained_pts_on_rows_with_no_team": int(m["pts_no_team"].sum()),
        "explained_unknown_rows_flagged_scoring": int(m["unknown_scoring_rows"].sum()),
        "explained_fga_scoringplay_but_not_made": int(m["fga_scoringplay_not_made"].sum()),
        "n_technical_ft_points": n_tech_ft_pts,
        "pbp_complete_games": int(complete.sum()),
        "pbp_complete_pct": round(100.0 * float(complete.mean()), 2),
        "pbp_complete_pct_of_not_truncated": round(
            100.0 * float(complete[~m["pbp_truncated"].to_numpy()].mean()), 2),
    }


def threshold_ladder(seasons: list[int], universe: pd.DataFrame, pbp_dir: Path) -> dict:
    """Derive the rim-override threshold from the data and measure every rung.

    Step 1 -- the distribution the cutoff comes out of: release-distance
    quantiles of the rows the FEED ITSELF calls a rim attempt (`DunkShot`,
    `LayUpShot`, `TipShot`), per season and pooled.
    Step 2 -- for each stated quantile of that distribution, re-segment every
    season and record the continuation-chance and first-chance rim / jump2
    shares.

    The selection rule was fixed before the numbers were looked at: 2025's
    continuation-chance rim and jump2 shares must land inside the
    2022-2024/2026 band, and no clean season may move by 0.5 pp or more.
    `pick` applies it mechanically."""
    per_season, dunk_pool, fam_pool, calib = {}, [], [], {}
    for s in seasons:
        p = load_plays(s, pbp_dir=pbp_dir)
        per_season[str(s)] = EV.rim_family_distance_quantiles(p)
        calib[str(s)] = EV.basket_calibration(p)
        d = EV.shot_distance_ft(p)
        pt = p["playType"].astype("string").fillna("").to_numpy(dtype=object)
        dunk_pool.append(d[(pt == "DunkShot") & np.isfinite(d)])
        fam_pool.append(d[np.isin(pt, list(EV.RIM_PLAY_TYPES)) & np.isfinite(d)])
        del p, d
    dunk = np.concatenate(dunk_pool)
    fam = np.concatenate(fam_pool)

    rungs: list[dict] = []
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        rungs.append({"label": f"DunkShot p{int(q * 100)}", "ft": round(float(np.quantile(dunk, q)), 3)})
    for q in (0.25, 0.50):
        rungs.append({"label": f"rim_family p{int(q * 100)}", "ft": round(float(np.quantile(fam, q)), 3)})
    merged: list[dict] = []
    for r in sorted(rungs, key=lambda r: r["ft"]):
        if merged and merged[-1]["ft"] == r["ft"]:
            merged[-1]["label"] += " / " + r["label"]
            continue
        merged.append(dict(r))

    orig = EV.classify_frame
    rows: list[dict] = []
    try:
        for t in [0.0] + [r["ft"] for r in merged]:
            poss_mod.classify_frame = (lambda pl, _t=t: orig(pl, rim_override_max_ft=_t))
            for s in seasons:
                _poss, ch, _diag = poss_mod.segment_season(s, universe, pbp_dir=pbp_dir)
                first = ch[ch["chance_number"] == 1]["terminal_event"].value_counts(normalize=True) * 100
                cont = ch[ch["chance_number"] > 1]["terminal_event"].value_counts(normalize=True) * 100
                row = {"threshold_ft": t, "season": int(s)}
                for k in ("FGA_rim", "FGA_jump2", "FGA_3"):
                    row[f"cont_{k}"] = round(float(cont.get(k, 0.0)), 3)
                    row[f"first_{k}"] = round(float(first.get(k, 0.0)), 3)
                rows.append(row)
                print(f"  ladder t={t:5.3f} {s}: cont rim {row['cont_FGA_rim']:.3f} "
                      f"jump2 {row['cont_FGA_jump2']:.3f}", flush=True)
    finally:
        poss_mod.classify_frame = orig

    return {"rungs": merged, "per_season_quantiles": per_season, "basket_calibration": calib,
            "pooled_n_dunk": int(len(dunk)), "pooled_n_rim_family": int(len(fam)),
            "effect": rows, "selection": pick_threshold(rows, merged)}


CLEAN_SEASONS = (2022, 2023, 2024, 2026)
CONTAMINATED_SEASON = 2025
CLEAN_SEASON_MAX_MOVE_PP = 0.5
GATED_CLASSES = ("FGA_rim", "FGA_jump2")


def pick_threshold(effect_rows: list[dict], rungs: list[dict]) -> dict:
    """Apply the pre-stated selection rule to the ladder, mechanically.

    THE RULE, fixed before any rung was measured, and transcribed here rather
    than restated:

      * HARD GATE -- "2022-2024 and 2026 must move by < 0.5 pp, otherwise the
        override is over-reaching". A rung is disqualified if ANY clean season's
        continuation-chance `FGA_rim` or `FGA_jump2` share moves 0.5 pp or more
        away from its v1 value. This is the false-positive test: those seasons
        are not mislabelled, so anything the override does to them is damage.
      * OBJECTIVE among the survivors -- "2025 returns to the 2022-2024/2026
        band". Minimise how far outside that band 2025 still sits (zero if it
        is inside), taking the worse of the two classes. The band is the
        four clean seasons' own min-max AT THAT RUNG, so it moves with the
        rung and is never a frozen target.
      * TIE-BREAK -- the SMALLER threshold. Two rungs that restore 2025 equally
        well are not equal: the narrower one touches fewer rows, and a repair
        should be the smallest one that works.

    `band_distance_pp` is reported for every rung, passing or not, so the shape
    of the trade-off is visible instead of just its argmax."""
    df = pd.DataFrame(effect_rows)
    base = df[df["threshold_ft"] == 0.0].set_index("season")
    out = []
    for r in rungs:
        t = r["ft"]
        sub = df[df["threshold_ft"] == t].set_index("season")
        moves = {int(s): {k: round(float(sub.loc[s, f"cont_{k}"] - base.loc[s, f"cont_{k}"]), 3)
                          for k in GATED_CLASSES} for s in sub.index}
        clean_move = max(abs(v[k]) for s, v in moves.items() if s in CLEAN_SEASONS
                         for k in GATED_CLASSES)
        band = {k: (min(float(sub.loc[s, f"cont_{k}"]) for s in CLEAN_SEASONS),
                    max(float(sub.loc[s, f"cont_{k}"]) for s in CLEAN_SEASONS))
                for k in GATED_CLASSES}
        dist = {}
        for k in GATED_CLASSES:
            v = float(sub.loc[CONTAMINATED_SEASON, f"cont_{k}"])
            lo, hi = band[k]
            dist[k] = round(max(0.0, lo - v, v - hi), 3)
        out.append({
            "threshold_ft": t, "quantile": r["label"],
            "cont_2025_rim": float(sub.loc[CONTAMINATED_SEASON, "cont_FGA_rim"]),
            "cont_2025_jump2": float(sub.loc[CONTAMINATED_SEASON, "cont_FGA_jump2"]),
            "clean_band_rim": [round(band["FGA_rim"][0], 3), round(band["FGA_rim"][1], 3)],
            "clean_band_jump2": [round(band["FGA_jump2"][0], 3), round(band["FGA_jump2"][1], 3)],
            "band_distance_pp": round(max(dist.values()), 3),
            "band_distance_by_class_pp": dist,
            "worst_clean_season_move_pp": round(clean_move, 3),
            "moves_pp": moves,
            "in_band": bool(max(dist.values()) == 0.0),
            "clean_move_ok": bool(clean_move < CLEAN_SEASON_MAX_MOVE_PP),
        })
    survivors = [r for r in out if r["clean_move_ok"]]
    chosen = min(survivors, key=lambda r: (r["band_distance_pp"], r["threshold_ft"])) if survivors else None
    return {
        "rungs": out,
        "chosen": chosen,
        "rule": ("hard gate: every clean season (2022-2024, 2026) moves < "
                 f"{CLEAN_SEASON_MAX_MOVE_PP} pp on cont FGA_rim and FGA_jump2. Among survivors, "
                 "minimise how far 2025 still sits outside the clean-season band; ties to the "
                 "smaller threshold."),
        "adopted_constant": EV.RIM_OVERRIDE_MAX_FT,
    }


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
    ap.add_argument("--version", default=DEFAULT_VERSION,
                    choices=sorted(poss_mod.POSSESSION_VERSIONS),
                    help="which possessions table to build (default v2; v1 is frozen)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="explicit output directory; overrides --version")
    ap.add_argument("--pbp-dir", type=Path, default=DEFAULT_PBP_DIR)
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--write-doc", action="store_true",
                    help="render the build/validation doc for this version from build_report.json")
    ap.add_argument("--validate-only", action="store_true",
                    help="re-run the validation battery off existing parquet files")
    ap.add_argument("--threshold-ladder", action="store_true",
                    help="derive the rim-override threshold from the data and measure every rung "
                         "(slow: it re-segments every season once per candidate cutoff)")
    args = ap.parse_args()

    out_dir = poss_mod.possessions_dir(args.version, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    universe = load_universe(args.universe)
    universe_all = pd.read_parquet(args.universe)

    report: dict = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seasons": {}}
    rp = out_dir / "build_report.json"
    if rp.exists():
        with contextlib.suppress(json.JSONDecodeError):
            report = json.loads(rp.read_text())
    report.setdefault("seasons", {})

    report["version"] = args.version
    report["rim_override"] = {
        "max_ft": EV.RIM_OVERRIDE_MAX_FT,
        "basket_xy": [list(b) for b in EV.BASKET_XY],
        "rim_play_types": list(EV.RIM_PLAY_TYPES),
    }
    if args.threshold_ladder:
        print("deriving the rim-override threshold from the data ...", flush=True)
        report["rim_override_threshold_ladder"] = threshold_ladder(
            list(args.seasons), universe, Path(args.pbp_dir))
        rp0 = out_dir / "build_report.json"
        rp0.write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote the ladder to {rp0}")

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
        v["feed_completeness"] = feed_completeness(season, universe_all, Path(args.pbp_dir))
        raw_cols = load_plays(season, pbp_dir=Path(args.pbp_dir),
                              game_ids=set(universe.loc[universe["season"] == season, "cbbd_game_id"]))
        v["rim_override_counts"] = EV.rim_override_counts(raw_cols)
        v["rim_family_quantiles"] = EV.rim_family_distance_quantiles(raw_cols)
        v["basket_calibration"] = EV.basket_calibration(raw_cols)
        del raw_cols
        report["seasons"][str(season)] = v
        print(f"[{season}] poss/game seg {v['poss_per_game_seg_mean']:.2f} vs box "
              f"{v['poss_per_game_box_mean']:.2f} (diff {v['diff_mean']:+.3f}, sd {v['diff_sd']:.3f}, "
              f"corr {v['diff_corr']:.4f}); score mismatch games {v['score_mismatch_games']} "
              f"(of which feed-incomplete {v.get('feed_vs_final_mismatch_games', 0)}; "
              f"segmentation-only {v.get('seg_vs_feed_mismatch_games', 0)})", flush=True)

    rp.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {rp}")
    if args.write_doc:
        print(f"wrote {render_doc(report, version=args.version)}")


if __name__ == "__main__":
    main()
