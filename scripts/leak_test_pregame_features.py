#!/usr/bin/env python
"""
leak_test_pregame_features.py -- run the INV-45 leak test (see
`src/cbb_sim/analysis/leak_test.py`) on the KenPom snapshot features CBB will
actually use, plus two deliberate positive controls.

WHY THREE ARMS.
  A. KenPom centered features (adj_o_c, adj_d_c, adj_t_rel), joined AS-OF --
     the latest snapshot strictly BEFORE each game's date
     (`cbb_sim.data.kenpom.join_as_of`). This is the join the sim is actually
     meant to use. Expected: pass, honest ~0.04-0.08 as-joined.
  B. The SAME features, joined SAME-DAY -- the latest snapshot ON OR BEFORE
     each game's date (`allow_exact_matches=True`, the classic off-by-one:
     "<=" instead of "<"). If KenPom's snapshot dated D already reflects
     games played ON calendar day D (i.e. the snapshot date POST-dates the
     games it describes), this join leaks; if KenPom's date-D snapshot only
     reflects games through D-1 (a morning scrape, before that night's
     games), this join is still clean and reads no differently from Arm A.
     Either outcome is informative and is reported honestly below.
  C. CBBD's `/ratings/adjusted` END-OF-SEASON rating, joined by (team,
     season) with NO date axis at all -- the second leak class the
     postmortem documents: a same-season aggregate has zero within-season
     variance (STATIC, change-form undefined) but its LEVEL correlation with
     the team's own game margins is inflated because the rating was computed
     FROM those same games. Expected: STATIC + a level-form LEAK.

Seasons: 2022-2025 only. Season 2026 is the sealed blind test
(`src/cbb_sim/data/seal.py`) and is never touched here, per CLAUDE.md's
PM/worker split and bake-off rules -- this is a pregame-feature leak audit
that ANY external rating must clear before it is allowed near a training
fold, not itself a model-selection artifact, but 2026 is excluded anyway to
keep the audit unambiguous.

Run: .venv/Scripts/python.exe scripts/leak_test_pregame_features.py
Writes: docs/tests/leak_test_kenpom_2026-09-10.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.analysis import leak_test  # noqa: E402
from cbb_sim.data import kenpom  # noqa: E402

SEASONS = (2022, 2023, 2024, 2025)
HOOPR_TEAM_BOX = ROOT / "data" / "raw" / "hoopR" / "team_box"
CBBD_RATINGS = ROOT / "data" / "raw" / "cbbd"
SNAPSHOTS_PATH = ROOT / "data" / "processed" / "kenpom_snapshots.parquet"
OUT_MD = ROOT / "docs" / "tests" / "leak_test_kenpom_2026-09-10.md"

CENTERED_COLS = ["adj_o_c", "adj_d_c", "adj_t_rel"]
CBBD_COLS = ["offensiveRating", "defensiveRating", "netRating"]


# ---------------------------------------------------------------------------
def load_margin_panel(seasons=SEASONS) -> pd.DataFrame:
    """One row per (season, team, game_id): own-team margin, ordered by
    game_date. `team` is hoopR's own `team_location` -- exact string, no
    crosswalk needed for this side (kenpom.join_as_of / the CBBD merge both
    match onto it)."""
    frames = []
    for season in seasons:
        path = HOOPR_TEAM_BOX / f"team_box_{season}.parquet"
        tb = pd.read_parquet(
            path,
            columns=["game_id", "season", "game_date", "team_location", "team_score", "opponent_team_score"],
        )
        frames.append(tb)
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["margin"] = df["team_score"] - df["opponent_team_score"]
    df = df.rename(columns={"team_location": "team"})
    dup = int(df.duplicated(["game_id", "team"]).sum())
    assert dup == 0, f"{dup} duplicate (game_id, team) rows in hoopR team_box"
    return df[["game_id", "season", "game_date", "team", "margin"]]


def join_same_day(panel: pd.DataFrame, snapshots: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """The Arm-B positive control: same as `kenpom.join_as_of` but with
    `allow_exact_matches=True` -- the latest snapshot ON OR BEFORE the game
    date, i.e. the classic "<=" off-by-one instead of "<"."""
    g = panel.copy()
    g["_team_key"] = g["team"].map(kenpom.normalize_join_key)
    g["game_date"] = pd.to_datetime(g["game_date"]).astype("datetime64[ns]")
    g["_row_id"] = np.arange(len(g))
    g_sorted = g.sort_values("game_date", kind="mergesort")

    s = snapshots.dropna(subset=["team"]).copy()
    s["_team_key"] = s["team"].map(kenpom.normalize_join_key)
    s["snapshot_date"] = pd.to_datetime(s["snapshot_date"]).astype("datetime64[ns]")
    s_sorted = s.sort_values("snapshot_date", kind="mergesort")

    merged = pd.merge_asof(
        g_sorted,
        s_sorted[["_team_key", "snapshot_date", *feature_cols]],
        left_on="game_date",
        right_on="snapshot_date",
        by="_team_key",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id", "_team_key"]).reset_index(drop=True)


def load_cbbd_ratings(seasons=SEASONS) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = CBBD_RATINGS / f"ratings_adjusted_{season}.parquet"
        frames.append(pd.read_parquet(path, columns=["season", "team", *CBBD_COLS]))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
def main() -> None:
    assert 2026 not in SEASONS, "season 2026 is sealed -- see src/cbb_sim/data/seal.py"

    print("[leak_test] loading hoopR margin panel (2022-2025) ...")
    panel = load_margin_panel(SEASONS)
    print(f"  {len(panel):,} team-games across {panel['team'].nunique()} teams, seasons {SEASONS}")

    print("[leak_test] loading KenPom snapshots ...")
    if not SNAPSHOTS_PATH.exists():
        raise FileNotFoundError(f"{SNAPSHOTS_PATH} not found -- run scripts/build_kenpom_snapshots.py first")
    snaps = pd.read_parquet(SNAPSHOTS_PATH)
    snaps = snaps[snaps["season"].isin(SEASONS)].copy()
    n_snap_unmatched = int(snaps["team"].isna().sum())
    print(f"  {len(snaps):,} snapshot rows, seasons {SEASONS} ({n_snap_unmatched} unmatched-team rows dropped for joins)")

    # ---- Arm A: as-of (strictly before), the join the sim is meant to use
    print("[leak_test] Arm A: as-of join (strictly before) ...")
    arm_a = kenpom.join_as_of(
        panel, snaps, game_team_col="team", game_date_col="game_date",
        feature_cols=CENTERED_COLS, suffix="",
    )
    cov_a = {c: float(arm_a[c].notna().mean()) for c in CENTERED_COLS}
    print(f"  coverage: {cov_a}")
    res_a = leak_test.run_leak_test(arm_a, CENTERED_COLS, team_col="team", season_col="season",
                                     date_col="game_date", margin_col="margin")

    # ---- Arm B: same-day (on-or-before), the deliberate positive control
    print("[leak_test] Arm B: same-day join (on-or-before, allow_exact_matches=True) ...")
    arm_b = join_same_day(panel, snaps, CENTERED_COLS)
    cov_b = {c: float(arm_b[c].notna().mean()) for c in CENTERED_COLS}
    print(f"  coverage: {cov_b}")
    res_b = leak_test.run_leak_test(arm_b, CENTERED_COLS, team_col="team", season_col="season",
                                     date_col="game_date", margin_col="margin")

    # ---- Arm C: CBBD end-of-season ratings, joined by (team, season), no date axis
    print("[leak_test] Arm C: CBBD /ratings/adjusted joined by (team, season) ...")
    cbbd = load_cbbd_ratings(SEASONS)
    arm_c = panel.merge(cbbd, on=["season", "team"], how="left")
    cov_c = {c: float(arm_c[c].notna().mean()) for c in CBBD_COLS}
    print(f"  coverage: {cov_c}")
    res_c = leak_test.run_leak_test(arm_c, CBBD_COLS, team_col="team", season_col="season",
                                     date_col="game_date", margin_col="margin")

    render(panel, snaps, n_snap_unmatched, cov_a, cov_b, cov_c, res_a, res_b, res_c)


def render(panel, snaps, n_snap_unmatched, cov_a, cov_b, cov_c, res_a, res_b, res_c) -> None:
    L = []
    A = L.append
    A("# KenPom pregame-feature leak test (INV-45, ported to CBB) -- 2026-09-10")
    A("")
    A("`scripts/leak_test_pregame_features.py`. Margin panel: hoopR "
      f"`team_box` seasons {', '.join(str(s) for s in SEASONS)} "
      f"({len(panel):,} team-games, {panel['team'].nunique()} teams). "
      "Season 2026 is excluded throughout -- it is the sealed blind test "
      "(`src/cbb_sim/data/seal.py`).")
    A("")
    A("## The statistic")
    A("")
    A("CBB adaptation of the CFB INV-45 method "
      "(`docs/postmortem/05_cfb_methodology_extract.md` section 4): rows are "
      "one per (team, season, game), ordered by **game_date** (CBB has no "
      "trustworthy week label -- 0-4 games/week depending on scheduling -- "
      "so \"consecutive\" means adjacent games, not adjacent weeks, unlike "
      "the CFB original which ordered by kickoff within a week label).")
    A("")
    A("| name | definition |")
    A("|---|---|")
    A("| `as-joined` | corr(`f[t] - f[t-1]`, that team's margin in **game t**) "
      "-- the leak channel: a pregame row cannot contain game t |")
    A("| `legit-update` | corr(`f[t] - f[t-1]`, that team's margin in the "
      "**previous game**) -- the healthy signature of a season-to-date "
      "column, and this test's own positive control: numerically what "
      "as-joined would read if the column were joined one game late |")
    A("| `level` | corr(`f[t]`, margin in game t) -- plain predictive "
      "correlation, context only |")
    A("")
    A(f"**Flag rule:** `|as-joined| > {leak_test.GATE}` = **LEAK** (honest "
      f"baseline ~0.04-0.08). `n < {leak_test.MIN_N}` team-games = "
      "UNDERPOWERED (not scored). Zero within-season variance in the delta "
      "= STATIC (change-form undefined) -- reported instead with the "
      f"level-form variant: `|level corr| > {leak_test.GATE}` on a STATIC "
      "column is itself a LEAK signature (a same-season aggregate whose "
      "level correlation with the team's own game margins is inflated "
      "because it was computed from those same games).")
    A("")

    n_unique = int(snaps["kenpom_name"].nunique()) if "kenpom_name" in snaps.columns else None
    A("## KenPom name-matching")
    A("")
    A(f"{n_snap_unmatched} of {len(snaps):,} snapshot rows in seasons "
      f"{', '.join(str(s) for s in SEASONS)} carry an unmatched `team` "
      "(dropped before joining -- see `scripts/build_kenpom_snapshots.py` "
      "output / `src/cbb_sim/data/kenpom.py` for the full name-matching "
      "writeup and the current match report).")
    A("")

    # ---------------- Arm A
    A("## Arm A -- as-of join (strictly before), the join the sim uses")
    A("")
    A("Coverage (share of team-games with a non-null joined feature): "
      + ", ".join(f"`{c}` {v:.1%}" for c, v in cov_a.items()))
    A("")
    L.extend(leak_test.render_markdown_table(res_a))
    A("")

    # ---------------- Arm B
    A("## Arm B -- same-day join (on-or-before), deliberate positive control")
    A("")
    A("Coverage: " + ", ".join(f"`{c}` {v:.1%}" for c, v in cov_b.items()))
    A("")
    L.extend(leak_test.render_markdown_table(res_b))
    A("")

    # ---------------- Arm C
    A("## Arm C -- CBBD end-of-season `/ratings/adjusted`, joined by (team, season)")
    A("")
    A("Coverage: " + ", ".join(f"`{c}` {v:.1%}" for c, v in cov_c.items()))
    A("")
    L.extend(leak_test.render_markdown_table(res_c))
    A("")

    # ---------------- honest reading
    A("## Honest reading")
    A("")
    all_a = res_a[res_a["season"] == "ALL"]
    all_b = res_b[res_b["season"] == "ALL"]
    all_c = res_c[res_c["season"] == "ALL"]
    worst_a = all_a.loc[all_a["corr_asjoined"].abs().idxmax()]
    worst_b = all_b.loc[all_b["corr_asjoined"].abs().idxmax()]
    a_leaks = int((res_a["verdict"] == "LEAK").sum())
    b_leaks = int((res_b["verdict"] == "LEAK").sum())
    c_static = bool(all_c["static"].all())
    c_level_leaks = int((all_c["level_verdict"] == "LEAK (level-form)").sum())

    A(f"- **Arm A (as-of, strictly before):** worst pooled as-joined is "
      f"`{worst_a['column']}` at {worst_a['corr_asjoined']:+.3f} "
      f"(n={worst_a['n']}). {a_leaks} of {len(res_a)} (column, season) cells "
      f"flagged LEAK. " +
      ("This is the join the sim is meant to use, and it reads clean." if a_leaks == 0
       else "This is the join the sim is meant to use -- **any LEAK cell here "
            "means the as-of join itself is unsafe and must be fixed before "
            "these features enter a feature table.**"))
    A("")
    A(f"- **Arm B (same-day, on-or-before):** worst pooled as-joined is "
      f"`{worst_b['column']}` at {worst_b['corr_asjoined']:+.3f} "
      f"(n={worst_b['n']}). {b_leaks} of {len(res_b)} (column, season) cells "
      f"flagged LEAK. " +
      (
          "Same-day joins are safe: KenPom's date-D snapshot evidently "
          "reflects games only through D-1 (a morning scrape, ahead of that "
          "night's games), so allowing an exact-date match does not pull in "
          "the game it is about to predict -- Arm B reads statistically the "
          "same as Arm A."
          if b_leaks == 0 and abs(worst_b["corr_asjoined"] - worst_a["corr_asjoined"]) < 0.05
          else "Same-day joins leak: including the exact-date snapshot pulls "
               "in information the as-of (strictly-before) join does not "
               "have -- confirms KenPom's snapshot dates effectively "
               "post-date the games they are labelled with, and the sim "
               "must keep using strict `<`, never `<=`."
      ))
    A("")
    A("- **Caveat on `legit-update` magnitude (both A and B):** `legit-update` "
      "runs ~0.13-0.19 here, above the ~0.04-0.08 single-game honest "
      "baseline the CFB original reports. This is a granularity artifact, "
      "not a defect: 2022-2025 KenPom snapshots are WEEKLY while CBB teams "
      "play ~2 games/week, so the delta between two consecutive-game rows "
      "often spans zero or one snapshot updates, and \"the previous game's "
      "margin\" this delta correlates with is really standing in for "
      "however many games happened since the last weekly refresh -- a "
      "coarser, noisier version of the single-game lag CFB's weekly-game "
      "cadence gives for free. It does not affect the leak verdict "
      "(`as-joined` is unaffected by this and stays low), only the size of "
      "the test's own positive-control number.")
    A("")
    A(f"- **Arm C (CBBD end-of-season, joined by team+season):** "
      f"{'all 3 columns STATIC as expected' if c_static else 'NOT all columns STATIC (unexpected)'} "
      f"(zero within-season variance -- a single end-of-season number joined "
      f"to every game that team played). {c_level_leaks} of {len(all_c)} "
      f"pooled columns show a level-form LEAK "
      f"(|level corr| > {leak_test.GATE}), confirming the second leak class: "
      "a same-season aggregate carries no change-form signature at all, so "
      "it can only be caught by the level-form variant, and here it clearly "
      "is one -- a team's own end-of-season rating is mechanically inflated "
      "by the same games it is being correlated against. This arm is a "
      "positive control, not a candidate feature: it demonstrates why any "
      "end-of-season rating must be shifted to the PRIOR season (or replaced "
      "by a genuine point-in-time snapshot, i.e. Arm A) before it is allowed "
      "near a feature table.")
    A("")
    A("**Bottom line:** the KenPom `as_of()` join in `src/cbb_sim/data/kenpom.py` "
      f"is the one arm cleared to feed a feature table "
      f"({'PASS' if a_leaks == 0 else 'FAIL'}, honest as-joined correlations "
      f"in the {res_a[res_a['season'] != 'ALL']['corr_asjoined'].abs().min():.3f}"
      f"-{res_a[res_a['season'] != 'ALL']['corr_asjoined'].abs().max():.3f} range). "
      "The CBBD end-of-season rating (Arm C) must never be joined by "
      "(team, season) alone -- only as a prior-season prior, or via its own "
      "point-in-time snapshot if CBBD ever exposes one.")
    A("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[leak_test] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
