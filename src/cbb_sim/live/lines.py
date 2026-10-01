"""Line sources for the daily chain (publish / grade / CLV). Read-only on the sim; lines come ONLY from the CBBD API client
already in the repo (`scripts/pull_cbbd.py`) or from files that client wrote. Nothing here scrapes a book.

Normalised line frame (one row per game x provider):
    cbbd_game_id, provider, spread (home perspective, negative = home favoured), total, home_ml, away_ml,
    spread_open, total_open, line_fetched_at (UTC), line_kind

`line_kind`:
    "live"          fetched from CBBD at run time; `spread` is the current line, `line_fetched_at` the fetch clock.
    "replay_close"  REPLAY of a past season: the CLOSING line from `lines_{season}.parquet` (no timestamp exists; the fake clock is stamped).
    "replay_open"   REPLAY: spread / total from the OPEN columns, moneylines from the close (the file has no open moneyline,
                    so `ml_is_close_proxy` is True and moneyline CLV is not computable).
CBBD gives no price for spreads and totals; -110 is assumed everywhere (as the eval harness does).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: ESPN BET ended: DraftKings first, Bovada second; `consensus` only as a labelled last resort.
LIVE_PROVIDERS: tuple[str, ...] = ("Draft Kings", "Bovada", "consensus")
REPLAY_PROVIDERS: tuple[str, ...] = ("ESPN BET",)
NUM = ("spread", "total", "home_ml", "away_ml", "spread_open", "total_open")
LINE_COLS = ["cbbd_game_id", "provider", *NUM, "line_fetched_at", "line_kind", "ml_is_close_proxy"]


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def normalise(raw: pd.DataFrame, fetched_at, kind: str) -> pd.DataFrame:
    """CBBD `/lines` flat frame (`pull_cbbd.flatten_lines`) or a `lines_{season}.parquet` slice -> the normalised frame."""
    if raw is None or not len(raw):
        return pd.DataFrame(columns=LINE_COLS)
    d = raw.dropna(subset=["provider"]).copy()
    out = pd.DataFrame({
        "cbbd_game_id": pd.to_numeric(d["gameId"], errors="coerce").astype("Int64"),
        "provider": d["provider"].astype(str),
        "spread": pd.to_numeric(d["spread"], errors="coerce"),
        "total": pd.to_numeric(d["overUnder"], errors="coerce"),
        "home_ml": pd.to_numeric(d["homeMoneyline"], errors="coerce"),
        "away_ml": pd.to_numeric(d["awayMoneyline"], errors="coerce"),
        "spread_open": pd.to_numeric(d["spreadOpen"], errors="coerce"),
        "total_open": pd.to_numeric(d["overUnderOpen"], errors="coerce"),
    })
    out["line_fetched_at"] = _utc(fetched_at)
    out["line_kind"] = kind
    out["ml_is_close_proxy"] = False
    return out.dropna(subset=["cbbd_game_id"]).reset_index(drop=True)


def pick_provider(lines: pd.DataFrame, preference: tuple[str, ...]) -> pd.DataFrame:
    """One row per game: the first provider in `preference` that has a spread OR a total OR a full moneyline pair."""
    if not len(lines):
        return lines
    d = lines[lines["provider"].isin(preference)].copy()
    d["_rank"] = d["provider"].map({p: i for i, p in enumerate(preference)})
    d = d.sort_values(["cbbd_game_id", "_rank"], kind="mergesort").drop_duplicates("cbbd_game_id", keep="first")
    return d.drop(columns="_rank").reset_index(drop=True)


def replay_lines(season: int, kind: str, now, game_ids=None, lines_dir: str | Path = "data/raw/cbbd",
                 providers: tuple[str, ...] = REPLAY_PROVIDERS) -> pd.DataFrame:
    """Replay line source from `lines_{season}.parquet`. `kind` is "close" or "open"; see the module docstring."""
    if kind not in ("open", "close"):
        raise ValueError(kind)
    raw = pd.read_parquet(Path(lines_dir) / f"lines_{int(season)}.parquet")
    raw = raw[raw["provider"].isin(providers)]
    if game_ids is not None:
        raw = raw[pd.to_numeric(raw["gameId"], errors="coerce").isin(set(int(g) for g in game_ids))]
    out = normalise(raw, now, f"replay_{kind}")
    if kind == "open":
        out["spread"], out["total"] = out["spread_open"], out["total_open"]
        out["ml_is_close_proxy"] = True
    return pick_provider(out, providers)


def fetch_cbbd_lines(slate_date, now, providers: tuple[str, ...] = LIVE_PROVIDERS, max_calls: int = 6,
                     session=None, tracker=None) -> pd.DataFrame:
    """LIVE: one CBBD `/lines` call over the slate date's UTC window plus the next 30 h (ET evening games fall on the next UTC date),
    through the repo's own client (`pull_cbbd.http_get`, which enforces the quota floor). Stamped with the fetch clock."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import pull_cbbd  # noqa: WPS433
    import requests
    d = pd.Timestamp(slate_date)
    start = pd.Timestamp(d.year, d.month, d.day, tz="UTC")
    end = start + pd.Timedelta(hours=54)
    season = d.year + 1 if d.month >= 5 else d.year
    if session is None:
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {pull_cbbd.load_api_key()}", "accept": "application/json"})
    tracker = tracker or pull_cbbd.CallTracker(max_calls=max_calls)
    games = pull_cbbd.http_get(session, tracker, "/lines", {"season": season, "startDateRange": pull_cbbd.iso(start.to_pydatetime()),
                                                         "endDateRange": pull_cbbd.iso(end.to_pydatetime())})
    return pick_provider(normalise(pull_cbbd.flatten_lines(games), now, "live"), providers)


def daily_snapshots(lines_daily_dir: str | Path, snapshot_type: str) -> pd.DataFrame:
    """LIVE closing lines: `chain_daily.step_lines` writes `data/raw/lines_daily/<date>.parquet` with a `close` snapshot of
    yesterday's games fetched the morning after (the last line the book showed). Latest fetch per (game, provider) wins."""
    frames = [pd.read_parquet(p) for p in sorted(Path(lines_daily_dir).glob("*.parquet"))]
    if not frames:
        return pd.DataFrame(columns=LINE_COLS)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["snapshot_type"] == snapshot_type].sort_values("fetched_at")
    raw = raw.drop_duplicates(["gameId", "provider"], keep="last")
    out = normalise(raw, pd.Timestamp.now("UTC"), "live_close" if snapshot_type == "close" else "live_open")
    ts = pd.to_datetime(raw["fetched_at"], utc=True)
    out["line_fetched_at"] = ts.reindex(raw.index).to_numpy() if len(out) == len(raw) else out["line_fetched_at"]
    return out


def devig(home_ml, away_ml) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided moneyline de-vig -> (p_home, vig). For the probability comparison ONLY; bets settle at the real odds."""
    def p(ml):
        ml = np.asarray(ml, dtype="float64")
        return np.where(ml > 0, 100.0 / (ml + 100.0), -ml / (-ml + 100.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ph, pa = p(home_ml), p(away_ml)
    return ph / (ph + pa), ph + pa - 1.0
