"""
variance.py -- sequential (nested) fixed-effects R^2 decomposition, ported
from `cfb-props-sim/src/cfb_props_sim/analysis/variance.py` verbatim in
method (see `docs/postmortem/05_cfb_methodology_extract.md` section 5 for the
full writeup of why this exists and what it drove in the CFB build).

WHY. The question it answers: how much of a team- or player-driven metric's
game-to-game variance is attributable to head coach (scheme), to team
identity beyond the coach, to a specific player's identity beyond coach+team,
to the opponent, and to irreducible noise -- used directly to decide what a
rating/prior table should be keyed on (coach, team-season, player, or
league). CFB found "team-beyond-coach ~= 0%" on every tendency metric it
measured (PROE, pace, 4th-down go-rate, explosive rate) -- a coach's tendency
IS the team's tendency there, because scheme moves with the coach and a
stable staff runs the same system year to year. The postmortem explicitly
warns not to assume that result carries over to CBB: the transfer-portal era
can turn over 4-5 of 5 starters under an unchanged head coach, so
"team-beyond-coach" is plausibly a real, non-trivial term in basketball.
`scripts/exp_variance_decomp_v1.py` is the CBB re-run that tests this
directly rather than assuming it.

THE METHOD (unchanged from CFB, sport-agnostic by construction -- nothing in
`decompose()` is football- or basketball-specific). It computes the total
sum-of-squares around the (optionally weighted) grand mean of `y`; reports a
**one-way R^2** for each grouping alone (that grouping's weighted
between-group SS over total SS -- "how much would this explain if it were
the only signal available"); then computes a **sequential R^2**: residualize
`y` by the first grouping's weighted group-mean, record the explained
variance as a fraction of total SS, then residualize what's left by the next
grouping (within the first), record its incremental share, and so on down
the `groups` list; whatever's left after every grouping is `residual_R2`.
The sequential number is the one that answers "does this grouping add
anything once we already know the earlier ones" -- the one-way number
cannot, because groupings that are correlated with each other (e.g. a coach
who mostly coaches one team) will each show inflated one-way R^2 for
variance that's really shared.

CBB ADAPTATION. Purely a call-site change, not a method change: CFB's chain
was coach -> team-beyond-coach -> QB-or-RB-beyond-both -> residual. CBB's
Study 1 (team-game metrics) chain is coach_id -> (team_id, season)
beyond-coach -> opponent-coach beyond both -> residual (opponent takes the
third slot instead of a second player, since a team-game row has no
individual-player identity of its own); CBB's Study 2 (player-game metrics)
chain is coach_id -> (team_id, season) beyond-coach -> athlete_id beyond both
-> residual, which is CFB's QB/RB slot exactly. Missing-group values are
filled with the literal string "UNKNOWN" (e.g. a team-season with no matched
head coach in `data/reference/coaches.parquet`) so they form their own group
rather than NaN-ing out the residualization -- see
`docs/tests/variance_decomp_2026-09-10.md` for how many rows that touches
per study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decompose(df: pd.DataFrame, y: str, groups: list[str], weight_col: str | None = None) -> dict:
    """Return a dict with per-grouping R^2 (one-way) and sequential-nested R^2s.

    Missing-group values are filled with 'UNKNOWN' so they don't NaN out
    residualization. If weight_col is given, the variance calculation is
    weighted (more possessions / more minutes = more signal in that row).

    Output keys, for each g in groups:
      - f"{g}_R2"            one-way R^2 (group alone, ignoring order)
      - f"{g}_n_groups"      number of distinct group values
      - f"{g}_sequential_R2" incremental share of total SS explained by g,
                              after residualizing on every earlier grouping
    Plus "residual_R2": whatever's left after every grouping in order.
    """
    # reset_index(drop=True) is a correctness fix over the CFB original, not a
    # behavior change: the sequential loop below indexes the positional
    # `residual` numpy array via `s.index` (a groupby sub-frame's pandas
    # index labels). That is only safe if df's index is a clean 0..n-1
    # range. The CFB call sites always happened to pass such a frame; a
    # caller that passes a frame with a non-contiguous index (e.g. built via
    # an upstream .dropna()/boolean mask, as CBB's real team-game/player-game
    # panels are) would silently raise an IndexError or, worse, misalign
    # rows without one. Resetting here makes decompose() safe for any input.
    df = df.copy().dropna(subset=[y]).reset_index(drop=True)
    for g in groups:
        df[g] = df[g].fillna("UNKNOWN").astype(str)

    y_arr = df[y].to_numpy(dtype=float)
    w = df[weight_col].to_numpy(dtype=float) if weight_col else np.ones_like(y_arr)
    y_mean = np.average(y_arr, weights=w)
    total_ss = float(np.sum(w * (y_arr - y_mean) ** 2))

    out: dict = {}
    # One-way R^2 -- group-mean (weighted) for each grouping separately.
    for g in groups:
        gm = (
            df.groupby(g)
            .apply(lambda s: np.average(s[y], weights=s[weight_col] if weight_col else None), include_groups=False)
            .to_dict()
        )
        pred = df[g].map(gm).to_numpy(dtype=float)
        ss_between = float(np.sum(w * (pred - y_mean) ** 2))
        out[f"{g}_R2"] = ss_between / total_ss if total_ss > 0 else float("nan")
        out[f"{g}_n_groups"] = int(df[g].nunique())

    # Sequential: residualize after each level, in the order `groups` was given.
    residual = y_arr - y_mean
    for g in groups:
        gm = (
            df.groupby(g)
            .apply(lambda s: np.average(residual[s.index], weights=s[weight_col] if weight_col else None), include_groups=False)
            .to_dict()
        )
        step_pred = df[g].map(gm).to_numpy(dtype=float)
        explained = float(np.sum(w * step_pred**2))
        out[f"{g}_sequential_R2"] = explained / total_ss if total_ss > 0 else float("nan")
        residual = residual - step_pred
    out["residual_R2"] = float(np.sum(w * residual**2) / total_ss) if total_ss > 0 else float("nan")
    return out


def format_decomp_table(decomp: dict, groups: list[str], title: str = "") -> str:
    """Markdown table of the decomposition results: Grouping | # groups |
    One-way R^2 | Sequential R^2, plus a residual row."""
    lines: list[str] = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    lines.append("| Grouping | # groups | One-way R^2 | Sequential R^2 |")
    lines.append("|---|---:|---:|---:|")
    for g in groups:
        lines.append(
            f"| {g} | {decomp[f'{g}_n_groups']} | "
            f"{decomp[f'{g}_R2']:.3f} | {decomp[f'{g}_sequential_R2']:.3f} |"
        )
    lines.append(f"| residual | -- | -- | {decomp['residual_R2']:.3f} |")
    return "\n".join(lines)
