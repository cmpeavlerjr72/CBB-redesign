"""Honest-backtest guards for the live path, enforced in code (CLAUDE.md).

`created_at < tipoff` on every output row; every source row used has
`game_date < slate date`; the as-of cutoff precedes the first tip of the slate.
Every function raises `LeakGuardError`; none returns a boolean to be ignored.
"""

from __future__ import annotations

import pandas as pd

#: a source game counts as finished this long after its tip (hours)
FINISH_MARGIN_H = 4.0


class LeakGuardError(RuntimeError):
    pass


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def assert_cutoff_before_slate(as_of, slate: pd.DataFrame) -> pd.Timestamp:
    """The as-of cutoff must precede EVERY tip of the slate."""
    t = _utc(as_of)
    first = pd.to_datetime(slate["tipoff_utc"], utc=True).min()
    if pd.isna(first):
        raise LeakGuardError("slate has no tipoff_utc; created_at < tipoff cannot be proven")
    if not t < first:
        raise LeakGuardError(f"as-of cutoff {t} is not before the first tip {first}")
    return t


def assert_sources_before(frame: pd.DataFrame, slate_date, name: str,
                          date_col: str = "game_date") -> None:
    """Every source row used for features must be dated strictly before the slate date."""
    if frame is None or not len(frame):
        return
    d = pd.to_datetime(frame[date_col]).max()
    if not d < pd.Timestamp(slate_date):
        raise LeakGuardError(f"{name}: source row dated {d.date()} is not before slate date "
                             f"{pd.Timestamp(slate_date).date()}")


def assert_sources_finished(universe_prior: pd.DataFrame, as_of) -> None:
    """Every source game must have finished (tip + FINISH_MARGIN_H) before the cutoff."""
    if "tipoff_utc" not in universe_prior.columns or not len(universe_prior):
        return
    t = _utc(as_of)
    last = pd.to_datetime(universe_prior["tipoff_utc"], utc=True).max()
    if pd.notna(last) and not last + pd.Timedelta(hours=FINISH_MARGIN_H) <= t:
        raise LeakGuardError(f"source game tipped {last} is not finished before cutoff {t}")


def assert_created_before_tipoff(df: pd.DataFrame, created_col: str = "created_at",
                                 tipoff_col: str = "tipoff_utc") -> None:
    """Every row carries created_at and tipoff and created_at < tipoff, or raise."""
    for c in (created_col, tipoff_col):
        if c not in df.columns:
            raise LeakGuardError(f"output is missing required column {c!r}")
    ca = pd.to_datetime(df[created_col], utc=True)
    tp = pd.to_datetime(df[tipoff_col], utc=True)
    if ca.isna().any() or tp.isna().any():
        raise LeakGuardError("created_at / tipoff_utc contain nulls")
    bad = ~(ca < tp)
    if bad.any():
        raise LeakGuardError(f"{int(bad.sum())} row(s) have created_at >= tipoff; first: "
                             f"{df.loc[bad].iloc[0].to_dict()}")
