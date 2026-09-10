"""
leak_test.py -- the INV-45 leak-quantification statistic, ported to CBB.

WHY. CLAUDE.md's "Standing rule: backtests must be honest": "Every external
feature (KenPom, any rating feed) passes the leak test before entering a
feature table: change-form correlation with own-week margin, |corr| <= 0.15,
honest baseline 0.04-0.08." This module is that test. Method and gate are
lifted verbatim from `cfb-props-sim/scripts/leak_test_ratings.py` and
`leak_test_pregame_features.py` (see
docs/postmortem/05_cfb_methodology_extract.md section 4 for the full writeup,
including the historical numbers this port is checked against: 0.707 / 0.839
leaked -> 0.076 clean for two CFB rating feeds).

CBB ADAPTATION. CFB re-keyed one row per (team, season) ordered by KICKOFF
within a WEEK label (needed because CFBD mislabels "Week 0" games as week 1).
CBB has no week label worth trusting at all -- teams play 0-4 games in a
calendar week depending on scheduling -- so the CBB unit is one row per
(team, season) ordered by GAME_DATE directly, and "consecutive games" means
adjacent rows in that ordering, not adjacent week numbers. Everything else
(the three statistics, the gate, the STATIC / UNDERPOWERED labels) carries
over unchanged.

THE STATISTIC (unchanged from CFB). For a feature column f already joined
onto a (team, season, game_date, margin) panel:

    d_t = f[t] - f[t-1]     between CONSECUTIVE games of the same team within
                            a season (bye/off weeks are skipped, not
                            interpolated -- there is no fixed schedule grid
                            to interpolate against in CBB anyway)

    as-joined     = corr(d_t, margin_t)        -- the leak channel. A truly
                    pregame column cannot contain game t, so this can only be
                    schedule autocorrelation (honest ~0.04-0.08).
    legit-update  = corr(d_t, margin_{t-1})    -- the healthy signature of a
                    season-to-date column updating on the game it just saw,
                    and this test's own positive control: it is numerically
                    what as-joined would read if the column were joined one
                    game late.
    level         = corr(f_t, margin_t)        -- plain predictive
                    correlation, context only for a genuine pregame column;
                    carries no leak information there.

FLAG RULE: |as-joined| > GATE (0.15) => LEAK. n < MIN_N (300) team-games =>
UNDERPOWERED (not scored pass/fail). A column with zero within-season
variance in d_t (every delta identically 0, e.g. a single end-of-season
value joined to every game that team played) => STATIC -- the change-form
test is UNDEFINED for it, not "safe": a same-season aggregate joined pregame
has no week-over-week delta for the change-form test to see at all, so it can
silently return PASS while carrying a real leak. For STATIC columns this
module also reports the LEVEL-FORM verdict (`level_verdict`): flagging
|level corr| > GATE, because a same-season aggregate's level correlation with
that team's own individual-game margins is inflated exactly to the extent the
aggregate was computed FROM those same games (the CFB postmortem's
documented example moved from +0.535 leaked to +0.005 clean after shifting
the join to the prior season's value instead of the current one).
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

GATE = 0.15   # |as-joined corr| above this = LEAK (standing INV-45 gate)
MIN_N = 300   # team-games below this = UNDERPOWERED, not scored pass/fail


def corr_and_n(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    """Pearson corr over the jointly-non-null rows, and that row count.

    A constant `x` (e.g. a STATIC column's all-zero delta) makes pandas fall
    through to `np.corrcoef` on a zero-stddev input, which correctly returns
    NaN but emits a RuntimeWarning about it -- expected here (that is
    precisely how the STATIC case is detected upstream), so it is
    suppressed rather than left to spam every STATIC column's report."""
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 3:
        return float("nan"), n
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        c = float(x[m].corr(y[m]))
    return c, n


def build_delta_panel(
    panel: pd.DataFrame,
    feature_col: str,
    team_col: str = "team",
    season_col: str = "season",
    date_col: str = "game_date",
    margin_col: str = "margin",
) -> pd.DataFrame:
    """Sort one row per (team, season, game) by game_date and attach:
    `_delta` = f[t] - f[t-1] between consecutive games of the same team
    within a season, and `_prev_margin` = that team's margin in the PREVIOUS
    game. Ties on game_date (a rare same-day doubleheader) are broken by
    `game_id` when present, for a deterministic ordering."""
    sort_cols = [team_col, season_col, date_col]
    if "game_id" in panel.columns:
        sort_cols.append("game_id")
    p = panel.sort_values(sort_cols, kind="mergesort").reset_index(drop=True).copy()
    g = p.groupby([team_col, season_col])
    p["_delta"] = g[feature_col].diff()
    p["_prev_margin"] = g[margin_col].shift(1)
    return p


def leak_test_column(
    panel: pd.DataFrame,
    feature_col: str,
    team_col: str = "team",
    season_col: str = "season",
    date_col: str = "game_date",
    margin_col: str = "margin",
    min_n: int = MIN_N,
    gate: float = GATE,
) -> dict:
    """One INV-45 leak-test result row for `feature_col` over `panel`
    (already restricted to whatever season/segment the caller wants)."""
    p = build_delta_panel(panel, feature_col, team_col, season_col, date_col, margin_col)

    n_delta = int(p["_delta"].notna().sum())
    n_nonzero = int((p["_delta"].abs() > 1e-12).sum())
    static = n_delta > 0 and n_nonzero == 0

    c_as, n_as = corr_and_n(p["_delta"], p[margin_col])
    c_up, n_up = corr_and_n(p["_delta"], p["_prev_margin"])
    c_lv, n_lv = corr_and_n(p[feature_col], p[margin_col])

    if static:
        verdict = "STATIC"
    elif n_as < min_n:
        verdict = "UNDERPOWERED"
    elif not np.isfinite(c_as):
        verdict = "n/a"
    elif abs(c_as) > gate:
        verdict = "LEAK"
    else:
        verdict = "pass"

    if not np.isfinite(c_lv):
        level_verdict = "n/a"
    elif n_lv < min_n:
        level_verdict = "UNDERPOWERED"
    elif abs(c_lv) > gate:
        level_verdict = "LEAK (level-form)"
    else:
        level_verdict = "pass (level-form)"

    return {
        "column": feature_col,
        "n": n_as,
        "n_update": n_up,
        "n_level": n_lv,
        "corr_asjoined": c_as,
        "corr_update": c_up,
        "corr_level": c_lv,
        "static": static,
        "n_nonzero_deltas": n_nonzero,
        "verdict": verdict,
        "level_verdict": level_verdict,
    }


def run_leak_test(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    team_col: str = "team",
    season_col: str = "season",
    date_col: str = "game_date",
    margin_col: str = "margin",
    min_n: int = MIN_N,
    gate: float = GATE,
    seasons: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """`leak_test_column()` for every column in `feature_cols`, once per
    season plus a pooled "ALL" row. One row per (column, season) in the
    returned frame."""
    season_values = list(seasons) if seasons is not None else sorted(panel[season_col].dropna().unique())
    rows = []
    for col in feature_cols:
        for season in list(season_values) + [None]:
            sub = panel if season is None else panel[panel[season_col] == season]
            r = leak_test_column(sub, col, team_col, season_col, date_col, margin_col, min_n, gate)
            r["season"] = season if season is not None else "ALL"
            rows.append(r)
    cols = ["column", "season", "n", "corr_asjoined", "corr_update", "corr_level",
            "static", "n_nonzero_deltas", "verdict", "level_verdict", "n_update", "n_level"]
    return pd.DataFrame(rows)[cols]


def _fmt(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:+.3f}"


def render_markdown_table(results: pd.DataFrame) -> list[str]:
    """Standard INV-45 markdown table: column | season | n | as-joined |
    legit-update | level | verdict."""
    lines = [
        "| column | season | n | as-joined | legit-update | level | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in results.iterrows():
        verdict = r["verdict"]
        if r["static"]:
            verdict = f"STATIC ({r['level_verdict']}, level r={_fmt(r['corr_level'])})"
        elif verdict == "LEAK":
            verdict = "**LEAK**"
        lines.append(
            f"| `{r['column']}` | {r['season']} | {r['n']} | "
            f"{_fmt(r['corr_asjoined'])} | {_fmt(r['corr_update'])} | "
            f"{_fmt(r['corr_level'])} | {verdict} |"
        )
    return lines
