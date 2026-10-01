"""
Real tip times for the daily chain (lane F2, 2026-09-30). Pure logic; the fetching lives in `scripts/pull_tip_times_v1.py`.

Problem: for most opening-week games CBBD carries `startTimeTbd = True` and stamps midnight ET as `startDate`, and hoopR stamps
05:00Z (`time_valid = False`). A midnight placeholder is NOT a tip time. This module:

  * builds one row per game with `tipoff_utc`, `tip_source`, `tip_time_is_placeholder` (never trusted as real) and `tip_fetched_at`;
  * prefers a real time (hoopR `time_valid`, CBBD not-TBD); when both are real and disagree the EARLIER one is used (the guard
    `created_at < tipoff` can then only get stricter, never looser);
  * treats ANY tip at exactly 00:00 America/New_York as a placeholder even if a feed says it is valid;
  * `select_for_pass` splits a slate for the two chain passes:
        evening  the default: run the evening before the slate; every game whose tip (real or placeholder) is after the clock;
                 a placeholder is later than any evening-before clock, so it can only err toward refusal after midnight ET
        morning  the same-day re-publish: ONLY games with a real tip that is still in the future.
The guard itself (`guards.assert_created_before_tipoff`) is untouched and still runs on every output row.
"""

from __future__ import annotations

import pandas as pd

ET = "America/New_York"


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def is_midnight_et(tip: pd.Series) -> pd.Series:
    t = pd.to_datetime(tip, utc=True, errors="coerce").dt.tz_convert(ET)
    return (t.dt.hour == 0) & (t.dt.minute == 0) & t.notna()


def hoopr_tip_frame(sched: pd.DataFrame) -> pd.DataFrame:
    """hoopR schedule -> (game_id, hoopr_tip, hoopr_valid). `date` is the UTC start, valid only when `time_valid`."""
    ok = sched["time_valid"].fillna(False).astype(bool) if "time_valid" in sched.columns else pd.Series(False, index=sched.index)
    return pd.DataFrame({"game_id": pd.to_numeric(sched["game_id"], errors="coerce").astype("Int64"),
                         "hoopr_tip": pd.to_datetime(sched["date"], utc=True, errors="coerce"),
                         "hoopr_valid": ok.to_numpy()}).dropna(subset=["game_id"]).drop_duplicates("game_id", keep="last")


def cbbd_tip_frame(games: pd.DataFrame) -> pd.DataFrame:
    """CBBD /games -> (game_id = sourceId, cbbd_game_id, cbbd_tip, cbbd_tbd)."""
    gid = games["sourceId"] if "sourceId" in games.columns else games["id"]
    return pd.DataFrame({"game_id": pd.to_numeric(gid, errors="coerce").astype("Int64"),
                         "cbbd_game_id": pd.to_numeric(games["id"], errors="coerce").astype("Int64"),
                         "cbbd_tip": pd.to_datetime(games["startDate"], utc=True, errors="coerce"),
                         "cbbd_tbd": games["startTimeTbd"].fillna(True).astype(bool)}).dropna(subset=["game_id"]).drop_duplicates("game_id", keep="last")


def merge_tips(cbbd: pd.DataFrame, hoopr: pd.DataFrame | None, fetched_at) -> pd.DataFrame:
    """One row per CBBD game (the universe of the chain). Columns: game_id, cbbd_game_id, tipoff_utc, tip_source,
    tip_time_is_placeholder, tip_disagree_min (real vs real, minutes), tip_fetched_at."""
    c = cbbd.copy()
    c["cbbd_real"] = ~c["cbbd_tbd"] & ~is_midnight_et(c["cbbd_tip"]) & c["cbbd_tip"].notna()
    if hoopr is not None and len(hoopr):
        h = hoopr.copy()
        h["hoopr_real"] = h["hoopr_valid"] & ~is_midnight_et(h["hoopr_tip"]) & h["hoopr_tip"].notna()
        c = c.merge(h[["game_id", "hoopr_tip", "hoopr_real"]], on="game_id", how="left")
        c["hoopr_real"] = c["hoopr_real"].fillna(False).astype(bool)
    else:
        c["hoopr_tip"], c["hoopr_real"] = pd.NaT, False
    both = c["cbbd_real"] & c["hoopr_real"]
    tip = pd.Series(c["cbbd_tip"].to_numpy(), index=c.index)
    src = pd.Series("cbbd_placeholder_midnight_et", index=c.index, dtype=object)
    only_h = c["hoopr_real"] & ~c["cbbd_real"]
    only_c = c["cbbd_real"] & ~c["hoopr_real"]
    tip[only_h] = c.loc[only_h, "hoopr_tip"]
    src[only_h] = "hoopr_schedule"
    src[only_c] = "cbbd"
    earlier = c[["cbbd_tip", "hoopr_tip"]].min(axis=1)
    tip[both] = earlier[both]
    src[both] = "cbbd+hoopr_min"
    c["tipoff_utc"] = pd.to_datetime(tip, utc=True)
    c["tip_source"] = src
    c["tip_time_is_placeholder"] = ~(c["cbbd_real"] | c["hoopr_real"])
    c["tip_disagree_min"] = ((c["cbbd_tip"] - c["hoopr_tip"]).abs().dt.total_seconds() / 60).where(both)
    c["tip_fetched_at"] = _utc(fetched_at)
    return c[["game_id", "cbbd_game_id", "tipoff_utc", "tip_source", "tip_time_is_placeholder", "tip_disagree_min", "tip_fetched_at"]]


def select_for_pass(slate: pd.DataFrame, now, pass_name: str = "evening") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(simulate, refuse). `slate` needs tipoff_utc and tip_time_is_placeholder. Refuse rows carry `refuse_reason`."""
    if pass_name not in ("evening", "morning"):
        raise ValueError(pass_name)
    now = _utc(now)
    tip = pd.to_datetime(slate["tipoff_utc"], utc=True)
    ph = slate["tip_time_is_placeholder"].fillna(True).astype(bool) if "tip_time_is_placeholder" in slate.columns \
        else pd.Series(True, index=slate.index)
    tipped = tip.isna() | ~(now < tip)
    reason = pd.Series("", index=slate.index, dtype=object)
    reason[tipped] = "tipoff <= clock (or no tip time)"
    if pass_name == "morning":
        reason[ph] = "placeholder tip time (not real); morning pass needs a real tip"     # priority: a placeholder is never judged "tipped"
    refuse = reason != ""
    return slate[~refuse].reset_index(drop=True), slate[refuse].assign(refuse_reason=reason[refuse]).reset_index(drop=True)


def default_clock(slate_date, pass_name: str = "evening") -> pd.Timestamp:
    """Documented default run clocks: evening = 20:00 ET the day before the slate; morning = 09:00 ET on the slate date."""
    d = pd.Timestamp(slate_date)
    if pass_name == "evening":
        return (d - pd.Timedelta(days=1) + pd.Timedelta(hours=20)).tz_localize(ET).tz_convert("UTC")
    return (d + pd.Timedelta(hours=9)).tz_localize(ET).tz_convert("UTC")
