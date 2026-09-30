#!/usr/bin/env python
"""
pull_daily_ingest_v1.py -- daily post-game ingestion (Lane K, 2026-09-30).

    .venv/Scripts/python.exe scripts/pull_daily_ingest_v1.py --dates 2026-11-02 [--root .]
    .venv/Scripts/python.exe scripts/pull_daily_ingest_v1.py --dates 2025-01-13,2025-01-14 \
        --root <scratch root> --hoopr-mode local --hoopr-local-dir data/raw/hoopr

Design note: docs/ops/daily_ingestion_2026-09-30.md.

For each game date (US/Eastern) it
  1. reads the hoopR schedule for the date and CBBD /games for the date window,
  2. VERIFIES finals across the two sources: a game is a "verified final" only when BOTH sources say
     final and BOTH give the same home and away score; anything else (not final, one source
     missing, score disagreement, flipped sides, box/pbp not yet published) goes to the PENDING list,
     which is retried on every later run and never dropped or filled,
  3. upserts (by game id) the season's raw tables: hoopR schedules / team_box / player_box / pbp / shots,
     CBBD games and CBBD plays (flattened exactly as pull_cbbd_pbp.py does),
  4. rebuilds the derived tables for the verified games only, with the existing library builders
     restricted to those games: games_universe rows, possessions + chances (v1 and v2 directories),
     fg_make / rebound / free-throw-attempt / usage event tables,
  5. writes the state tables (finals_verified_{season}, pending_{season}) and a manifest.

Every appended row carries `ingested_at` (UTC). A game already complete in finals_verified is skipped
(zero writes), so running a date twice changes nothing. Nothing outside `--root` is written; refuses to
touch the repo root for seasons <= 2026 (sealed / back seasons) unless --allow-backfill.

Only NEW files; the existing pull_/build_ scripts are imported for helpers and never edited.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

HOOPR_DATASETS = {          # key -> (github dir, stem)
    "schedules": ("schedules", "mbb_schedule"),
    "team_box": ("team_box", "team_box"),
    "player_box": ("player_box", "player_box"),
    "pbp": ("pbp", "play_by_play"),
    "shots": ("shots", "shots"),
}
HOOPR_URL = ("https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/"
             "{dirname}/parquet/{stem}_{season}.parquet")
INGEST_COL = "ingested_at"
PATIENCE_HOURS = 72          # after tip + this, a verified final with a missing box/pbp piece is ingested with a listed gap


# --------------------------------------------------------------------------- helpers
def utcnow() -> pd.Timestamp:
    return pd.Timestamp.now("UTC")


def season_of(d: date) -> int:
    return d.year + 1 if d.month >= 8 else d.year


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _align(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Cast `new` columns to the dtype of `old` where the columns match (concat would otherwise drift)."""
    for c in new.columns:
        if c in old.columns and new[c].dtype != old[c].dtype:
            with contextlib.suppress(Exception):
                new[c] = new[c].astype(old[c].dtype)
    return new


def _row_hash(df: pd.DataFrame) -> pd.Series:
    d = df.drop(columns=[INGEST_COL], errors="ignore").copy()
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].map(lambda v: "<NA>" if v is None or (isinstance(v, float) and v != v) or v is pd.NA else v if isinstance(v, (str, int, float, bool)) else str(np.asarray(v).tolist()) if hasattr(v, "__len__") else str(v))
    return pd.util.hash_pandas_object(d, index=False)


def _game_sigs(df: pd.DataFrame, key: str, cols: list) -> dict:
    h = _row_hash(df[cols]).to_numpy(dtype="uint64")
    k = pd.to_numeric(df[key], errors="coerce").fillna(-1).astype("int64").to_numpy()
    g = pd.DataFrame({"k": k, "h": h}).groupby("k")["h"].agg(["size", lambda x: int(x.to_numpy().sum(dtype="uint64")),
                                                               lambda x: int(np.bitwise_xor.reduce(x.to_numpy()))])
    return {int(i): tuple(r) for i, r in zip(g.index, g.itertuples(index=False))}


def upsert(path: Path, new: pd.DataFrame, key: str, ts: pd.Timestamp, ledger: dict, table: str,
           drop_cols: tuple = ()) -> None:
    """Replace-by-key then append `new` (which gets `ingested_at`). Records rows added / replaced."""
    if new is None or not len(new):
        ledger.setdefault(table, {"rows_added": 0, "rows_replaced": 0, "games": 0})
        return
    new = new.drop(columns=[c for c in drop_cols if c in new.columns]).copy()
    new[INGEST_COL] = ts
    replaced = 0
    if path.exists():
        old = pd.read_parquet(path)
        if INGEST_COL not in old.columns:
            old[INGEST_COL] = pd.Series(pd.NaT, index=old.index, dtype=new[INGEST_COL].dtype)
        # idempotence: a game whose incoming rows equal the stored rows (ignoring ingested_at) is left untouched
        common = [c for c in new.columns if c in old.columns and c != INGEST_COL]
        if len(old) and common:
            kn0 = set(pd.to_numeric(new[key], errors="coerce").dropna().astype("int64").tolist())
            oldsub = old[pd.to_numeric(old[key], errors="coerce").isin(kn0)]
            if len(oldsub):
                so = _game_sigs(_align(oldsub.copy(), new), key, common)
                sn = _game_sigs(new, key, common)
                same = {g for g in sn if g in so and so[g] == sn[g]}
                if same:
                    new = new[~pd.to_numeric(new[key], errors="coerce").isin(same)]
                    if not len(new):
                        ledger.setdefault(table, {"rows_added": 0, "rows_replaced": 0, "games": 0, "path": str(path)})
                        return
        kn = pd.to_numeric(new[key], errors="coerce").dropna().astype("int64")
        drop = pd.to_numeric(old[key], errors="coerce").isin(set(kn.tolist())).to_numpy()
        replaced = int(drop.sum())
        old = old[~drop]
        new = _align(new, old)
        for c in old.columns:                       # keep the old column set and order
            if c not in new.columns:
                new[c] = pd.Series(pd.NA, index=new.index, dtype=object) if old[c].dtype == object else np.nan
        new = new[[c for c in old.columns] + [c for c in new.columns if c not in old.columns]]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new.reset_index(drop=True)
    atomic_write_parquet(out, path)
    e = ledger.setdefault(table, {"rows_added": 0, "rows_replaced": 0, "games": 0, "path": str(path)})
    e["rows_added"] += int(len(new))
    e["rows_replaced"] += replaced
    e["games"] += int(pd.to_numeric(new[key], errors="coerce").nunique())


# --------------------------------------------------------------------------- sources
class HooprSource:
    """hoopR-mbb-data season files. mode=live downloads the raw GitHub parquet (a full season file per
    dataset, there is no per-date file); mode=local reads a directory laid out like data/raw/hoopr
    (used by the replay, where the on-disk files stand in for what hoopR serves)."""

    def __init__(self, mode: str, local_dir: Path | None, cache_dir: Path):
        self.mode, self.local_dir, self.cache_dir = mode, local_dir, cache_dir
        self.meta: dict = {}
        self._frames: dict = {}

    def load(self, dataset: str, season: int) -> pd.DataFrame:
        k = (dataset, season)
        if k in self._frames:
            return self._frames[k]
        dirname, stem = HOOPR_DATASETS[dataset]
        if self.mode == "local":
            p = Path(self.local_dir) / dirname / f"{stem}_{season}.parquet"
            df = pd.read_parquet(p)
            self.meta[f"{dataset}_{season}"] = {"source": f"local:{p}", "bytes": p.stat().st_size,
                                                "fetched_at": str(utcnow())}
        else:
            import requests
            url = HOOPR_URL.format(dirname=dirname, stem=stem, season=season)
            r = requests.get(url, timeout=300, stream=True,
                             headers={"User-Agent": "cbb-clean-sheet-daily-ingest/1.0"})
            r.raise_for_status()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            p = self.cache_dir / f"{dataset}_{season}.parquet"
            n = 0
            with open(p, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    n += len(chunk)
            df = pd.read_parquet(p)
            self.meta[f"{dataset}_{season}"] = {
                "source": url, "bytes": n, "etag": r.headers.get("ETag"),
                "http_date": r.headers.get("Date"), "last_modified": r.headers.get("Last-Modified"),
                "fetched_at": str(utcnow())}
        self.meta[f"{dataset}_{season}"]["rows"] = int(len(df))
        self._frames[k] = df
        return df


class CbbdSource:
    """CBBD API. Every call is counted; the quota header is recorded."""

    def __init__(self, max_calls: int, cache_dir: Path, reuse_cache: bool):
        import pull_cbbd_pbp as PP
        self.PP = PP
        self.session = None
        self.tracker = PP.CallTracker()
        self.calls: list = []
        self.max_calls = max_calls
        self.cache_dir, self.reuse_cache = cache_dir, reuse_cache
        self.remaining_first: int | None = None

    def _sess(self):
        if self.session is None:
            import requests
            from pull_cbbd import load_api_key
            self.session = requests.Session()
            self.session.headers.update({"Authorization": f"Bearer {load_api_key()}", "Accept": "application/json"})
        return self.session

    def get(self, endpoint: str, params: dict):
        ck = self.cache_dir / (endpoint.strip("/").replace("/", "_") + "_" +
                               "_".join(f"{k}-{str(v).replace(':', '')}" for k, v in sorted(params.items())) + ".json")
        if self.reuse_cache and ck.exists():
            self.calls.append({"endpoint": endpoint, "params": params, "cached": True})
            return json.loads(ck.read_text(encoding="utf-8"))
        if len(self.calls) - sum(1 for c in self.calls if c.get("cached")) >= self.max_calls:
            raise RuntimeError(f"CBBD call budget for this run exhausted ({self.max_calls})")
        t = time.time()
        data = self.PP.http_get(self._sess(), self.tracker, endpoint, params)
        if self.remaining_first is None:
            self.remaining_first = self.tracker.remaining
        self.calls.append({"endpoint": endpoint, "params": params, "rows": len(data),
                           "remaining_after": self.tracker.remaining, "at": str(utcnow()),
                           "seconds": round(time.time() - t, 1)})
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ck.write_text(json.dumps(data), encoding="utf-8")
        return data

    def games(self, season: int, d0: date, d1: date) -> pd.DataFrame:
        """Games whose startDate is in [d0 - 1 day, d1 + 2 days] UTC (covers ET dates d0..d1), windows <= 14 days."""
        rows: dict = {}
        cur = d0 - timedelta(days=1)
        end = d1 + timedelta(days=2)
        while cur <= end:
            w1 = min(cur + timedelta(days=13), end)
            data = self.get("/games", {"season": season, "startDateRange": f"{cur}T00:00:00Z",
                                       "endDateRange": f"{w1}T23:59:59Z"})
            for g in data:
                rows[g["id"]] = g
            cur = w1 + timedelta(days=1)
        return pd.DataFrame(list(rows.values()))

    def plays(self, bucket: date, home_away: dict) -> pd.DataFrame:
        data = self.get("/plays/date", {"date": str(bucket)})
        return self.PP.flatten_plays(data, home_away, str(bucket))


# --------------------------------------------------------------------------- verification
def et_date(start_iso: pd.Series) -> pd.Series:
    d = pd.to_datetime(start_iso, utc=True)
    return d.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()


def verify_finals(hs: pd.DataFrame, cg: pd.DataFrame) -> pd.DataFrame:
    """One row per hoopR schedule row in `hs` with status + reason. See module docstring."""
    cgi = cg.copy() if len(cg) else pd.DataFrame(columns=["id", "sourceId", "status", "homePoints", "awayPoints"])
    cgi["src"] = pd.to_numeric(cgi["sourceId"], errors="coerce")
    cgi = cgi.dropna(subset=["src"]).drop_duplicates("src")
    cgi.index = cgi["src"].astype("int64")
    out = []
    for r in hs.itertuples(index=False):
        gid = int(r.game_id)
        c = cgi.loc[gid] if gid in cgi.index else None
        in_scope = pd.notna(r.home_conference_id) and pd.notna(r.away_conference_id)
        h_final = bool(r.status_type_completed) and pd.notna(r.home_score) and pd.notna(r.away_score) \
            and (float(r.home_score) + float(r.away_score)) > 0
        rec = {"game_id": gid, "cbbd_game_id": (int(c["id"]) if c is not None else pd.NA),
               "game_date": pd.Timestamp(r.game_date), "home_team_id": int(r.home_id), "away_team_id": int(r.away_id),
               "hoopr_status": str(r.status_type_name), "hoopr_home": r.home_score, "hoopr_away": r.away_score,
               "cbbd_status": (str(c["status"]) if c is not None else None),
               "cbbd_home": (c["homePoints"] if c is not None else np.nan),
               "cbbd_away": (c["awayPoints"] if c is not None else np.nan),
               "tipoff_utc": pd.Timestamp(r.game_date_time).tz_convert("UTC") if pd.notna(r.game_date_time) else pd.NaT}
        if not in_scope:
            rec.update(state="out_of_scope_non_d1", reason="hoopR conference_id null on a side", detail="")
        elif c is None:
            rec.update(state="pending", reason="awaiting_cbbd_game_row" if h_final else "not_final_hoopr_and_no_cbbd",
                       detail="no CBBD game with sourceId == hoopR game_id")
        else:
            c_final = (str(c["status"]).lower() == "final") and pd.notna(c["homePoints"]) and pd.notna(c["awayPoints"])
            if h_final and c_final:
                if float(r.home_score) == float(c["homePoints"]) and float(r.away_score) == float(c["awayPoints"]):
                    rec.update(state="final_verified", reason="", detail="")
                elif float(r.home_score) == float(c["awayPoints"]) and float(r.away_score) == float(c["homePoints"]):
                    rec.update(state="pending", reason="sides_flipped", detail="hoopR and CBBD have home/away scores swapped")
                else:
                    rec.update(state="pending", reason="score_disagree",
                               detail=f"hoopR {r.home_score}-{r.away_score} vs CBBD {c['homePoints']}-{c['awayPoints']}")
            elif h_final and not c_final:
                rec.update(state="pending", reason="awaiting_cbbd_final", detail=f"cbbd status {c['status']}")
            elif c_final and not h_final:
                rec.update(state="pending", reason="awaiting_hoopr_final", detail=f"hoopR status {r.status_type_name}")
            else:
                rec.update(state="pending", reason="not_final_either_source",
                           detail=f"hoopR {r.status_type_name}; cbbd {c['status']}")
        out.append(rec)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- derived builds
@contextlib.contextmanager
def v1_classify(off: bool):
    from cbb_sim.pbp import events as EV
    from cbb_sim.pbp import possessions as PM
    orig = PM.classify_frame
    if off:
        PM.classify_frame = lambda pl: EV.classify_frame(pl, rim_override_max_ft=0.0)
    try:
        yield
    finally:
        PM.classify_frame = orig


def build_universe_rows(root: Path, season: int, ids: set, ledger: dict, notes: dict) -> pd.DataFrame:
    """Rows of the games universe for the verified games, by the EXISTING builder run on the root's raw tables,
    with the D-I flag made provisional-safe for a partial season (see docs note section 4)."""
    from cbb_sim.data import universe as UNI
    full, _ = UNI.build_universe([season], hoopr_dir=root / "data/raw/hoopr", cbbd_dir=root / "data/raw/cbbd")
    sch = UNI.load_schedules(root / "data/raw/hoopr", [season])
    strict = UNI.compute_d1_team_seasons(sch)
    carry = set()
    up = root / "data/processed/games_universe.parquet"
    if up.exists():
        u_prev = pd.read_parquet(up, columns=["season", "home_team_id", "away_team_id", "is_d1_game"])
        u_prev = u_prev[(u_prev["season"] == season - 1) & u_prev["is_d1_game"]]
        carry = {(season, int(t)) for t in pd.concat([u_prev["home_team_id"], u_prev["away_team_id"]]).unique()}
    d1 = strict | carry
    ids_ = pd.to_numeric(full["game_id"]).astype("int64")
    rows = full[ids_.isin(ids)].copy()
    new_d1 = np.array([(season, int(h)) in d1 and (season, int(a)) in d1
                       for h, a in zip(rows["home_team_id"], rows["away_team_id"])])
    strict_d1 = rows["is_d1_game"].to_numpy()
    notes["d1_rule"] = "strict(>=5 conf games through the ingested date) UNION prior-season D-I teams"
    notes["d1_flag_changed_by_carry"] = int((new_d1 != strict_d1).sum())
    rows["is_d1_game"] = new_d1
    # DETECTION ONLY: games ingested earlier as non-D-I (stored flag False) whose teams now qualify (a team crossing the
    # >= 5 conference-game threshold, e.g. a new D-I member). Their derived rows were never built; listed, not repaired.
    flips: list = []
    if up.exists():
        cur = pd.read_parquet(up, columns=["game_id", "season", "is_d1_game"])
        cur = cur[(cur["season"] == season) & ~cur["is_d1_game"]]
        now_d1 = np.array([(season, int(h)) in d1 and (season, int(a)) in d1
                           for h, a in zip(full["home_team_id"], full["away_team_id"])])
        flips = sorted(set(cur["game_id"].astype("int64")) & set(pd.to_numeric(full.loc[now_d1, "game_id"]).astype("int64")))
    notes["d1_flips_detected_not_repaired"] = [int(x) for x in flips]
    return rows.reset_index(drop=True)


def rebuild_derived(root: Path, season: int, ids: set, ts: pd.Timestamp, ledger: dict, notes: dict) -> None:
    from cbb_sim.models import event_stream as ES
    from cbb_sim.models import fg_make as FG
    from cbb_sim.models import free_throw as FT
    from cbb_sim.models import rebound as RB
    from cbb_sim.models import usage as U
    from cbb_sim.pbp import possessions as PM

    t0 = time.time()
    urows = build_universe_rows(root, season, ids, ledger, notes)
    upsert(root / "data/processed/games_universe.parquet", urows, "game_id", ts, ledger, "games_universe")
    have = set(pd.read_parquet(root / f"data/raw/cbbd/pbp/plays_{season}.parquet", columns=["gameId"])["gameId"].astype("int64"))
    n_no_plays = int((~urows["cbbd_game_id"].astype("float64").isin({float(x) for x in have})).sum())
    notes["verified_games_without_cbbd_plays_not_built"] = n_no_plays
    u = urows[urows["is_d1_game"] & ~urows["pbp_truncated"]
              & urows["cbbd_game_id"].astype("float64").isin({float(x) for x in have})].copy()
    u["game_date"] = pd.to_datetime(u["game_date"])
    u_pc = u[u["pbp_complete"]].copy()
    ucb = u[u["cbbd_game_id"].notna()].copy()
    ucb["cbbd_game_id"] = ucb["cbbd_game_id"].astype("int64")
    notes["derived_universe"] = {"verified_games": int(len(urows)), "d1_not_truncated": int(len(u)),
                                 "pbp_complete": int(len(u_pc))}
    pbp_dir = root / "data/raw/cbbd/pbp"

    for ver, sub, is_v1 in (("v1", "possessions", True), ("v2", "possessions_v2", False)):
        if not len(ucb):
            break
        with v1_classify(is_v1):
            poss, ch, diag = PM.segment_season(season, ucb, pbp_dir=pbp_dir)
        if is_v1:                                    # the frozen v1 chance table predates the per-class shot columns
            ch = ch.drop(columns=[c for c in ch.columns if c.startswith(("fga_", "fgm_"))])
        upsert(root / f"data/processed/{sub}/possessions_{season}.parquet", poss, "game_id", ts, ledger, f"possessions_{ver}")
        upsert(root / f"data/processed/{sub}/chances_{season}.parquet", ch, "game_id", ts, ledger, f"chances_{ver}")
        notes[f"machine_diag_{ver}"] = {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in diag.items()}

    mdir = root / "data/processed/models"
    if len(u_pc):
        ev = FG.build_fg_events([season], universe=u_pc, version="v2", shooter_key="shot_shooter_id")
        upsert(mdir / "fg_make/events_v2_shotshooter.parquet", ev, "game_id", ts, ledger, "fg_events_v2_shotshooter")
        rim = ES.rim_override_for_version("v2")
        raw = U.build_usage_events(season, u_pc, rim_override_max_ft=rim, shooter_key="shot_shooter_id")
        upsert(mdir / "usage_v2/events_v2_shotshooter.parquet", U.usable_events(raw), "game_id", ts, ledger,
               "usage_events_v2_shotshooter")
    if len(u):
        rb = RB.build_rebound_events([season], universe=u, version="v1")
        upsert(mdir / "rebound/events_v1.parquet", rb, "game_id", ts, ledger, "rebound_events_v1")
        trips, att = FT.build_trips_and_attempts([season], universe=u, version="v1")
        # trip_id is a build-order counter over the whole table: offset the new ids so they stay unique
        base = 0
        for f in ("trips_v1_era.parquet", "attempts_v1_era.parquet"):
            p = mdir / "free_throw" / f
            if p.exists():
                base = max(base, int(pd.read_parquet(p, columns=["trip_id"])["trip_id"].max()) + 1)
        if len(att):
            trips = trips.copy(); att = att.copy()
            trips["trip_id"] = trips["trip_id"] + base
            att["trip_id"] = att["trip_id"] + base
        notes["ft_trip_id_offset"] = int(base)
        upsert(mdir / "free_throw/trips_v1_era.parquet", trips, "game_id", ts, ledger, "ft_trips_v1_era")
        upsert(mdir / "free_throw/attempts_v1_era.parquet", att, "game_id", ts, ledger, "ft_attempts_v1_era")
    notes["derived_seconds"] = round(time.time() - t0, 1)


def rebuild_truth(root: Path, season: int, ids: set, ts: pd.Timestamp, ledger: dict, notes: dict) -> None:
    """Grading-truth tables (team_game_shots_v2, player_game_v2, game_finals_v2) by the existing v2 builders, run on the
    root's tables for the season and keyed down to the ingested games. They are graders' inputs, not live-path inputs,
    so a failure here is reported in the manifest and does not block the state write."""
    import build_truth_tables as T1
    import build_truth_tables_v2 as T2
    T1.SEASONS = T2.SEASONS = (season,)
    cw = root / "data/processed/player_crosswalk.parquet"
    if cw.exists():
        cws = pd.read_parquet(cw, columns=["season"])["season"].unique()
        if season in set(int(x) for x in cws):
            T1.CROSSWALK_SEASONS = T2.CROSSWALK_SEASONS = tuple(sorted(set(T1.CROSSWALK_SEASONS) | {season}))
    uni = T1.load_universe()
    tgs, _ = T2.build_team_game_shots_v2(uni)
    pg, _ = T2.build_player_game_v2(uni)
    gf1, _ = T1.build_game_finals(uni)
    gf, _ = T2.apply_finals_resolutions(gf1)
    for name, df in (("team_game_shots_v2", tgs), ("player_game_v2", pg), ("game_finals_v2", gf)):
        upsert(root / f"data/processed/truth/{name}.parquet", df[df["game_id"].isin(ids)], "game_id", ts, ledger, f"truth_{name}")


# --------------------------------------------------------------------------- state tables
def read_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", required=True, help="comma-separated ET game dates YYYY-MM-DD")
    ap.add_argument("--root", default=str(REPO), help="data root (contains data/raw, data/processed)")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--hoopr-mode", choices=("live", "local"), default="live")
    ap.add_argument("--hoopr-local-dir", default=None)
    ap.add_argument("--cbbd-max-calls", type=int, default=12)
    ap.add_argument("--reuse-cache", action="store_true", help="reuse cached CBBD responses under <root>/data/raw/_ingest_cache")
    ap.add_argument("--now", default=None, help="clock for the patience rule (ISO UTC); default wall clock")
    ap.add_argument("--retry-days", type=int, default=14, help="pending games older than this are listed but not retried")
    ap.add_argument("--patience-hours", type=int, default=PATIENCE_HOURS)
    ap.add_argument("--allow-backfill", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    t_start = time.time()
    root = Path(a.root).resolve()
    dates = sorted({date.fromisoformat(x.strip()) for x in a.dates.split(",")})
    season = a.season or season_of(dates[0])
    if root == REPO and season <= 2026 and not a.allow_backfill:
        raise SystemExit(f"refusing to write season {season} into the repo root (sealed / back season); use a scratch --root")
    local_dir = Path(a.hoopr_local_dir).resolve() if a.hoopr_local_dir else None
    os.chdir(root)                                  # library builders resolve data/... relative to cwd
    now = pd.Timestamp(a.now, tz="UTC") if a.now else utcnow()
    ts = utcnow()
    run_id = f"{ts.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
    ing = root / "data/processed/ingest"
    ing.mkdir(parents=True, exist_ok=True)
    fv_path, pend_path = ing / f"finals_verified_{season}.parquet", ing / f"pending_{season}.parquet"
    fv, pend = read_or_empty(fv_path), read_or_empty(pend_path)
    done = set(fv["game_id"].astype("int64")) if len(fv) else set()

    cache = root / "data/raw/_ingest_cache"
    hoopr = HooprSource(a.hoopr_mode, local_dir, cache / "hoopr")
    cbbd = CbbdSource(a.cbbd_max_calls, cache / "cbbd", a.reuse_cache)
    manifest: dict = {"run_id": run_id, "started_at": str(ts), "clock_for_patience": str(now), "root": str(root),
                      "season": season, "dates": [str(d) for d in dates], "dry_run": a.dry_run, "warnings": []}

    # ---- 1. candidates: the requested dates plus pending games still inside the retry window
    hs = hoopr.load("schedules", season)
    hs = hs.copy()
    hs["gd"] = pd.to_datetime(hs["game_date"])
    want_dates = {pd.Timestamp(d) for d in dates}
    retry_ids: set = set()
    if len(pend):
        cut = pd.Timestamp(now.tz_localize(None).normalize()) - pd.Timedelta(days=a.retry_days)
        pr = pend[(pd.to_datetime(pend["game_date"]) >= cut)]
        retry_ids = set(pr["game_id"].astype("int64"))
        want_dates |= {pd.Timestamp(x) for x in pr["game_date"]}
    cand = hs[hs["gd"].isin(want_dates) & ~hs["game_id"].astype("int64").isin(done)].copy()
    cand_all = hs[hs["gd"].isin(want_dates)]
    manifest["n_schedule_rows_requested_dates"] = int(len(cand_all))
    manifest["n_skipped_already_complete"] = int(len(cand_all) - len(cand))
    if not len(cand):
        manifest.update(finished_at=str(utcnow()), note="nothing to do: every game on the requested dates is already complete",
                        cbbd_calls=0, rows_added={}, hoopr_sources=hoopr.meta)
        _write_manifest(ing, manifest, a.dry_run)
        print(json.dumps({"run_id": run_id, "nothing_to_do": True}))
        return 0

    if not (cand["home_conference_id"].notna() & cand["away_conference_id"].notna()).any():
        manifest.update(finished_at=str(utcnow()), note="nothing to do: only out-of-scope (non-D-I) games remain on the requested dates",
                        n_out_of_scope_non_d1=int(len(cand)), cbbd_calls=0, rows_added={}, hoopr_sources=hoopr.meta)
        _write_manifest(ing, manifest, a.dry_run)
        print(json.dumps({"run_id": run_id, "nothing_to_do": True, "out_of_scope": int(len(cand))}))
        return 0

    # ---- 2. CBBD games for the window, finals verification
    lo, hi = min(want_dates).date(), max(want_dates).date()
    cg = cbbd.games(season, lo, hi)
    if not len(cg):
        manifest["warnings"].append("CBBD /games returned no rows for the window")
        cg = pd.DataFrame(columns=["id", "sourceId", "status", "homePoints", "awayPoints", "startDate", "homeTeam", "awayTeam"])
    cg["et_date"] = et_date(cg["startDate"]) if len(cg) else pd.Series(dtype="datetime64[ns]")
    ver = verify_finals(cand, cg)
    manifest["verification_counts"] = ver["state"].value_counts().to_dict()

    # ---- 3. data presence for the verified finals (hoopR box/pbp; CBBD plays)
    vids = set(ver.loc[ver["state"] == "final_verified", "game_id"])
    tb = hoopr.load("team_box", season); pb = hoopr.load("player_box", season); pbp = hoopr.load("pbp", season)
    n_tb = tb[tb["game_id"].isin(vids)].groupby("game_id").size()
    n_pb = pb[pb["game_id"].isin(vids)].groupby("game_id").size()
    n_pbp = pbp[pbp["game_id"].isin(vids)].groupby("game_id").size()
    cb_id = dict(zip(ver["game_id"], ver["cbbd_game_id"]))
    home_away = {int(r.id): (r.homeTeam, r.awayTeam) for r in cg.itertuples(index=False)} if len(cg) else {}
    play_frames = []
    need_plays = {g: int(cb_id[g]) for g in vids}
    if need_plays:
        buckets = sorted({(pd.Timestamp(d) + pd.Timedelta(days=k)).date() for d in ver.loc[ver["game_id"].isin(vids), "game_date"] for k in (0, 1)})
        for b in buckets:
            pf = cbbd.plays(b, home_away)
            if len(pf):
                play_frames.append(pf[pf["gameId"].isin(set(need_plays.values()))])
    plays = pd.concat(play_frames, ignore_index=True) if play_frames else pd.DataFrame()
    n_cplays = plays.groupby("gameId").size() if len(plays) else pd.Series(dtype="int64")

    gaps, complete, awaiting = {}, [], []
    tip = ver.set_index("game_id")["tipoff_utc"]
    for g in sorted(vids):
        miss = []
        if n_tb.get(g, 0) != 2: miss.append("hoopr_team_box")
        if n_pb.get(g, 0) < 10: miss.append("hoopr_player_box")
        if n_pbp.get(g, 0) < 1: miss.append("hoopr_pbp")
        if n_cplays.get(need_plays[g], 0) < 1: miss.append("cbbd_plays")
        if not miss:
            complete.append(g)
        elif pd.notna(tip.get(g)) and now > tip[g] + pd.Timedelta(hours=a.patience_hours):
            gaps[g] = miss
            complete.append(g)              # ingested with a LISTED gap, never filled
        else:
            awaiting.append((g, miss))
    for g, miss in awaiting:
        i = ver.index[ver["game_id"] == g][0]
        ver.loc[i, ["state", "reason", "detail"]] = ["pending", "awaiting_data", ",".join(miss)]
    ids = set(complete)
    manifest["n_complete"] = len(ids); manifest["n_awaiting_data"] = len(awaiting); manifest["gaps_listed"] = {str(k): v for k, v in gaps.items()}

    # ---- 4. raw appends
    ledger: dict = {}
    notes: dict = {}
    if not a.dry_run:
        # hoopR schedule rows for EVERY game on the requested dates (schedule truth; includes postponed etc.)
        upsert(root / f"data/raw/hoopr/schedules/mbb_schedule_{season}.parquet",
               cand.drop(columns=["gd"]), "game_id", ts, ledger, "hoopr_schedules")
        for ds, frame in (("team_box", tb), ("player_box", pb), ("pbp", pbp), ("shots", None)):
            fr = frame if frame is not None else hoopr.load(ds, season)
            upsert(root / f"data/raw/hoopr/{ds}/{HOOPR_DATASETS[ds][1]}_{season}.parquet",
                   fr[fr["game_id"].isin(ids)], "game_id", ts, ledger, f"hoopr_{ds}")
        # CBBD games rows: every CBBD game whose ET date is one of the requested dates
        cgd = cg[cg["et_date"].isin(want_dates)].copy()
        cgd = cgd.drop(columns=["et_date"])
        upsert(root / f"data/raw/cbbd/games_{season}.parquet", cgd, "id", ts, ledger, "cbbd_games")
        if len(plays):
            keep = plays[plays["gameId"].isin({need_plays[g] for g in ids})]
            keep = keep.sort_values(["query_date", "gameId", "id"], kind="stable")
            upsert(root / f"data/raw/cbbd/pbp/plays_{season}.parquet", keep, "gameId", ts, ledger, "cbbd_plays")
        # ---- 5. derived
        derived_ok = True
        if ids:
            try:
                rebuild_derived(root, season, ids, ts, ledger, notes)
                try:
                    rebuild_truth(root, season, ids, ts, ledger, notes)
                except Exception as exc:
                    import traceback
                    manifest["warnings"].append(f"truth tables not rebuilt: {exc!r} :: {traceback.format_exc()[-600:]}")
            except Exception as exc:                      # raw tables are in; state is NOT written, so the next run redoes these games
                import traceback
                derived_ok = False
                manifest["errors"] = [f"rebuild_derived failed: {exc!r}", traceback.format_exc()]
                ids = set()
        # ---- 6. state
        v_new = ver[ver["game_id"].isin(ids)].copy()
        v_new["gaps"] = v_new["game_id"].map(lambda g: ",".join(gaps.get(g, [])))
        v_new["verified_at"] = ts
        v_new = v_new.drop(columns=["state", "reason", "detail"])
        fv2 = pd.concat([fv, v_new.assign(**{INGEST_COL: ts})], ignore_index=True) if len(fv) else v_new.assign(**{INGEST_COL: ts})
        if len(v_new):
            atomic_write_parquet(fv2, fv_path)
        p_new = ver[ver["state"] == "pending"].copy()
        p_new["last_attempt_at"] = ts
        prev = pend.set_index("game_id") if len(pend) else None
        p_new["first_seen_at"] = p_new["game_id"].map(lambda g: prev.loc[g, "first_seen_at"] if prev is not None and g in prev.index else ts)
        p_new["attempts"] = p_new["game_id"].map(lambda g: int(prev.loc[g, "attempts"]) + 1 if prev is not None and g in prev.index else 1)
        p_new = p_new.drop(columns=["state"])
        keep_old = pend[~pend["game_id"].isin(set(ver["game_id"]))] if len(pend) else pend
        p_all = pd.concat([keep_old, p_new], ignore_index=True) if len(keep_old) else p_new
        if len(p_new) or len(keep_old) != len(pend):
            atomic_write_parquet(p_all, pend_path)
        manifest["pending_after"] = int(len(p_all))
    manifest.update(rows_added=ledger, derived_notes=notes, hoopr_sources=hoopr.meta,
                    cbbd_calls_total=len([c for c in cbbd.calls if not c.get("cached")]), cbbd_calls=cbbd.calls,
                    cbbd_quota_remaining_last=cbbd.tracker.remaining,
                    pending_list=ver[ver["state"] == "pending"][["game_id", "cbbd_game_id", "game_date", "reason", "detail"]]
                    .astype(str).to_dict("records"),
                    n_out_of_scope_non_d1=int((ver["state"] == "out_of_scope_non_d1").sum()),
                    finished_at=str(utcnow()), seconds=round(time.time() - t_start, 1))
    _write_manifest(ing, manifest, a.dry_run)
    if manifest.get("errors"):
        print("ERROR: derived rebuild failed; see manifest", manifest["errors"][0])
        return 1
    print(json.dumps({"run_id": run_id, "complete": len(ids), "pending": int((ver["state"] == "pending").sum()),
                      "out_of_scope": manifest["n_out_of_scope_non_d1"],
                      "cbbd_calls": manifest["cbbd_calls_total"], "seconds": manifest["seconds"]}))
    return 0


def _write_manifest(ing: Path, manifest: dict, dry: bool) -> None:
    if dry:
        return
    mdir = ing / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"ingest_{manifest['season']}_{manifest['run_id']}.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
