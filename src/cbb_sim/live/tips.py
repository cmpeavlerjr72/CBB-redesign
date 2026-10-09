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
    tip = effective_tip(slate)                      # a placeholder proves only "not before 00:00 ET of the game date"
    ph = slate["tip_time_is_placeholder"].fillna(True).astype(bool) if "tip_time_is_placeholder" in slate.columns \
        else pd.Series(True, index=slate.index)
    tipped = tip.isna() | ~(now < tip)
    reason = pd.Series("", index=slate.index, dtype=object)
    reason[tipped] = late_reasons(slate[tipped])
    if pass_name == "morning":
        reason[ph] = "placeholder tip time (not real); morning pass needs a real tip"     # priority: a placeholder is never judged "tipped"
    refuse = reason != ""
    return slate[~refuse].reset_index(drop=True), slate[refuse].assign(refuse_reason=reason[refuse]).reset_index(drop=True)


#: refuse reason for a placeholder tip the clock has passed (2026-10-09, docs/ops/tip_guard_2026-10-09.md): the game has NOT been shown to
#: have tipped; its tip time is unknown, so created_at < tipoff cannot be proven. Refused (never simulated) and labelled as such.
TIP_UNKNOWN = "tip time unknown: placeholder 00:00 ET already passed, real tip not yet published; cannot prove pre-tip"
TIPPED = "tipoff <= clock (or no tip time)"


def effective_tip(slate: pd.DataFrame) -> pd.Series:
    """The latest clock at which a row can still be PROVEN pre-tip: the tip itself when real; for a placeholder (feed flag or midnight ET)
    the earlier of the placeholder and 00:00 ET of the game date (`game_date` when present, else the placeholder's own ET date)."""
    tip = pd.to_datetime(slate["tipoff_utc"], utc=True, errors="coerce")
    if not len(slate):
        return tip
    ph = flag_placeholders(slate)["tip_time_is_placeholder"].to_numpy()
    base = slate["game_date"] if "game_date" in slate.columns else tip.dt.tz_convert(ET).dt.tz_localize(None)
    bound = earliest_possible_tips(pd.Series(base, index=slate.index))
    return tip.where(~ph | tip.isna() | (tip <= bound), bound)


def late_reasons(late: pd.DataFrame) -> pd.Series:
    """Refuse reason per row of a slate subset the clock has passed: TIP_UNKNOWN for a placeholder (feed flag or midnight ET), else TIPPED."""
    if not len(late):
        return pd.Series([], index=late.index, dtype=object)
    ph = flag_placeholders(late)["tip_time_is_placeholder"].to_numpy()
    tip = pd.to_datetime(late["tipoff_utc"], utc=True, errors="coerce")
    return pd.Series([TIP_UNKNOWN if p and pd.notna(t) else TIPPED for p, t in zip(ph, tip)], index=late.index, dtype=object)


def default_clock(slate_date, pass_name: str = "evening") -> pd.Timestamp:
    """Documented default run clocks: evening = 20:00 ET the day before the slate; morning = 09:00 ET on the slate date."""
    d = pd.Timestamp(slate_date)
    if pass_name == "evening":
        return (d - pd.Timedelta(days=1) + pd.Timedelta(hours=20)).tz_localize(ET).tz_convert("UTC")
    return (d + pd.Timedelta(hours=9)).tz_localize(ET).tz_convert("UTC")


# ---------------------------------------------------------------------------------------------- pre-tip basis (lane F, 2026-10-01)
def flag_placeholders(slate: pd.DataFrame) -> pd.DataFrame:
    """Return `slate` with an explicit `tip_time_is_placeholder`: kept where a feed already set it, and ALSO true for any tip at exactly
    00:00 America/New_York or with no tip. Used by the universe (replay) path, which has no feed flag."""
    s = slate.copy()
    base = s["tip_time_is_placeholder"].fillna(True).astype(bool) if "tip_time_is_placeholder" in s.columns else pd.Series(False, index=s.index)
    tip = pd.to_datetime(s["tipoff_utc"], utc=True, errors="coerce")
    s["tip_time_is_placeholder"] = (base | is_midnight_et(tip) | tip.isna()).to_numpy()
    return s


def earliest_possible_tip(slate_date) -> pd.Timestamp:
    """00:00 America/New_York of the slate date, in UTC: no real tip of that slate date can be earlier. It is the only thing a placeholder tip proves."""
    return pd.Timestamp(slate_date).normalize().tz_localize(ET).tz_convert("UTC")


def stamp_pre_tip_basis(games: pd.DataFrame, slate: pd.DataFrame, slate_date) -> pd.DataFrame:
    """Add `tip_time_is_placeholder`, `tip_source`, `pre_tip_basis` and `pre_tip_verified` to every output row.

      real tip         pre_tip_basis 'real_tip',               pre_tip_verified True  (created_at < a tip a feed reports as real)
      placeholder tip  pre_tip_basis 'placeholder_lower_bound', pre_tip_verified False (created_at < placeholder is NOT evidence of
                       a pre-tip row; it is accepted only because created_at is before the earliest possible tip of the slate date,
                       which is asserted here, and the row stays unverified until a real tip arrives)

    Raises LeakGuardError for a placeholder row whose created_at is not before `earliest_possible_tip(slate_date)`."""
    from cbb_sim.live.guards import LeakGuardError
    s = flag_placeholders(slate)[["game_id", "tip_time_is_placeholder"] + (["tip_source"] if "tip_source" in slate.columns else [])]
    out = games.merge(s.drop_duplicates("game_id"), on="game_id", how="left")
    ph = out["tip_time_is_placeholder"].fillna(True).astype(bool)
    out["tip_time_is_placeholder"] = ph
    out["pre_tip_basis"] = ph.map({True: "placeholder_lower_bound", False: "real_tip"})
    out["pre_tip_verified"] = ~ph
    bound = earliest_possible_tip(slate_date)
    bad = ph & ~(pd.to_datetime(out["created_at"], utc=True) < bound)
    if bad.any():
        raise LeakGuardError(f"{int(bad.sum())} row(s) rest on a placeholder tip time with created_at >= {bound} (earliest possible tip of {slate_date})")
    return out


def earliest_possible_tips(game_date: pd.Series) -> pd.Series:
    """Vector form of `earliest_possible_tip`: 00:00 America/New_York of each row's game date, in UTC."""
    d = pd.to_datetime(game_date)
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert(ET).dt.tz_localize(None)
    return d.dt.normalize().dt.tz_localize(ET).dt.tz_convert("UTC")


def stamp_pre_tip_rows(rows: pd.DataFrame, created_col: str, date_col: str = "game_date") -> pd.DataFrame:
    """In-place-style variant of `stamp_pre_tip_basis` for a frame that already carries `tipoff_utc` (and, if known, the feed's
    `tip_time_is_placeholder`) per row, e.g. the publish rows. Same rule: a placeholder row is accepted only when `created_col` is before the
    earliest possible tip of its game date, and it is flagged `pre_tip_verified` False; a real-tip row is verified by created < tip."""
    from cbb_sim.live.guards import LeakGuardError
    out = flag_placeholders(rows)
    ph = out["tip_time_is_placeholder"].astype(bool)
    out["pre_tip_basis"] = ph.map({True: "placeholder_lower_bound", False: "real_tip"})
    out["pre_tip_verified"] = ~ph
    bad = ph & ~(pd.to_datetime(out[created_col], utc=True) < earliest_possible_tips(out[date_col]))
    if bad.any():
        raise LeakGuardError(f"{int(bad.sum())} row(s) rest on a placeholder tip time with {created_col} at or after 00:00 ET of the game date")
    return out


PRE_TIP_STATUS = ("real_tip", "placeholder_reverified", "placeholder_unverified", "violated")


def reverify_pre_tip(rows: pd.DataFrame, tip_table: pd.DataFrame | None, created_col: str = "published_at") -> pd.DataFrame:
    """Grade-time re-check (2026-10-09). Rows certified only by a placeholder lower bound are checked against the latest REAL tip
    (`tip_table`: game_id, tipoff_utc, tip_time_is_placeholder; the tips stage's latest table) once one exists:
      real_tip                the row's own tip was real at build time (already verified by created < tip)
      placeholder_reverified  a real tip is now known and created < real tip
      placeholder_unverified  still no real tip: the row stays flagged (pre-tip status unknown, not assumed)
      violated                a real tip is known and created >= real tip: NOT pre-tip; the caller must not grade it
    Adds `pre_tip_status` and `real_tipoff_utc`."""
    out = flag_placeholders(rows)
    ph = out["tip_time_is_placeholder"].astype(bool)
    real = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    if tip_table is not None and len(tip_table):
        t = tip_table[~tip_table["tip_time_is_placeholder"].fillna(True).astype(bool)]
        t = t.assign(_t=pd.to_datetime(t["tipoff_utc"], utc=True))
        t = t[~is_midnight_et(t["_t"])].drop_duplicates("game_id", keep="last").set_index("game_id")["_t"]
        real = out["game_id"].map(t)
        real = pd.to_datetime(real, utc=True)
    created = pd.to_datetime(out[created_col], utc=True)
    status = pd.Series("real_tip", index=out.index, dtype=object)
    status[ph & real.isna()] = "placeholder_unverified"
    status[ph & real.notna() & (created < real)] = "placeholder_reverified"
    status[ph & real.notna() & ~(created < real)] = "violated"
    out["pre_tip_status"] = status
    out["real_tipoff_utc"] = real.where(ph, pd.to_datetime(out["tipoff_utc"], utc=True))
    return out
