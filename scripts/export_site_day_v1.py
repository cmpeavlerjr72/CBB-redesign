#!/usr/bin/env python
"""
export_site_day_v1.py -- export one slate day of the daily chain into the site's per-day HF bundle (`mvpeav/cbb-sims-2027`).

CONTRACT (docs: the CBB site dataset contract, 2026-10-29). The basketball dataset mirrors the football dataset
`mvpeav/cfb-sims-2026` (cfb-props-sim/scripts/export_site_week.py) KEY FOR KEY, with DAYS in place of WEEKS, so the site's
football scoreboard reads either. No live.json / live_players.json (the site's live layer is `/api/live?sport=cbb&date=`).

    2027/season_index.json                                   # {sport, season, convention, layout, days: [...]}  MERGED across days
    2027/days/<YYYY-MM-DD>/index.json                        # the day: games list with paths relative to the season prefix
    2027/days/<YYYY-MM-DD>/lines.json                        # betting-record strip input (football week lines.json shape)
    2027/days/<YYYY-MM-DD>/games/<slug>/summary.json         # headline numbers + odds (+ finals once the grade ledger has them)
    2027/days/<YYYY-MM-DD>/games/<slug>/compact.json         # score quantiles (1..99) + histograms + per-seed series
    2027/days/<YYYY-MM-DD>/games/<slug>/players.json         # per-player stat summaries (mean, p10..p90)
    2027/days/<YYYY-MM-DD>/games/<slug>/players_dist.json    # per-player integer PMFs (mass == nsims for every stat)
    2027/days/<YYYY-MM-DD>/games/<slug>/seeds.json           # seed-aligned joint series (A_pts, B_pts, player prop stats)

ORIENTATION CONVENTION (stamped into every file, verbatim from football):
    teamA = home, teamB = away, margin = home minus away; spread is in HOME perspective (negative = home favored)

Sources (all under results/daily/ by default):
    sim      results/daily/sim/<date>/<run_id>/{games.parquet, players.parquet, slate.parquet, run_meta.json, _DONE.json}
    publish  results/daily/publish/<date>/<run_id>/<publish_id>/slate.parquet  (sim means + stored lines: spread, total,
             spread_open, total_open, provider)  -- the newest publish_id of the run
    finals   results/daily/grade/ledger.parquet  (game_id, home_score, away_score) once the grade stage has graded the slate;
             else data/processed/ingest/finals_verified_2027.parquet (hoopR and CBBD final scores that agree), so finals reach
             the site before the next day's grade stage
    ids      data/reference/team_crosswalk_v2.parquet  espn_team_id (the chain's team_id IS the ESPN id)
    names    the preseason roster (`roster_players_2027.parquet`, cbb_sim.live.preseason.preseason_dir) when it carries
             players, else the ESPN roster pull data/raw/cbbd/rosters/roster_2027.parquet (espn_player_id -> name, position)

Slug = football `make_slug(home, away)`: lowercase, non-[a-z0-9] stripped, `<home>_<away>`. `game_id` = the chain's game_id.
`run_id` default = ALL finished runs of the day (2026-11-02 multi-run rule): the day bundle is the UNION of games across runs,
each game taken from the NEWEST run that contains it (the morning pass skips already-tipped games, the evening run still has
them); a game's `source_tag` (index.json + summary.json) is the run it came from, the day-level `source_tag` is the newest run.
An explicit `run_id` exports that single run only. Players with mean minutes < 1 are dropped.
A stat that is zero for every player on the day (2026-10-29: `ast`, so `pra` with it) is dropped from the day's stat_keys and
from every file rather than published as a column of zeros (printed as `stat keys dropped`). A placeholder tip
(`tip_time_is_placeholder`) publishes `time_utc` null / `time_et` "TBD"; a real verified tip is published as the source gives it.

Usage:
    .venv/Scripts/python.exe scripts/export_site_day_v1.py --date 2026-11-02
    .venv/Scripts/python.exe scripts/export_site_day_v1.py --date 2026-11-02 --run-id s2500_o0 --upload

Re-runs are idempotent: the day directory is rebuilt from scratch and, on --upload, replaces the day on the Hub while
season_index.json is MERGED (local stage + Hub copy) so other days survive. The chain calls `export_day()` as the last step of
its publish stage (chain_daily_v3.stage_publish); a failure there never fails the chain.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SPORT = "cbb"
SEASON = 2027
DEFAULT_REPO = f"mvpeav/cbb-sims-{SEASON}"
DEFAULT_STAGE_ROOT = REPO / "data" / "processed" / "site_stage" / f"cbb-sims-{SEASON}"
DAILY_ROOT = REPO / "results" / "daily"
CROSSWALK = REPO / "data" / "reference" / "team_crosswalk_v2.parquet"
VERIFIED_FINALS = REPO / "data" / "processed" / "ingest" / "finals_verified_2027.parquet"
ESPN_ROSTER = REPO / "data" / "raw" / "cbbd" / "rosters" / f"roster_{SEASON}.parquet"
CFB_ENV = Path(r"C:\Users\devuser\cfb-props-sim\.env")
ET = ZoneInfo("America/New_York")

CONVENTION = ("teamA = home, teamB = away, margin = home minus away; "
              "spread is in HOME perspective (negative = home favored)")

# players.parquet column (or derived column) -> site stat name. `fgm` = fgm2_rim + fgm2_jump + fgm3; `ftm` is exact from the
# points identity (pts - 2*fgm2 - 3*fgm3; checked 0 <= ftm <= fta on the 2026-11-02 run); `pra` = pts + reb + ast; `min` = minutes.
PLAYER_STATS = {
    "minutes": "min", "pts": "pts", "reb": "reb", "ast": "ast", "fgm": "fgm", "fga": "fga", "fgm3": "fg3m", "fg3a": "fg3a",
    "ftm": "ftm", "fta": "fta", "fouls": "fouls", "pra": "pra",
}
STAT_KEYS = list(PLAYER_STATS.values())
SEED_STATS = ["pts", "reb", "ast", "fg3m", "pra"]          # seeds.json carries the prop-relevant stats only
# A stat the player layer does not produce yet (2026-10-29: `ast` is 0 for every player on the slate) is DROPPED from the day's
# stat_keys rather than published as a column of zeros; `pra` goes with `ast` (it would silently be pts + reb).
DERIVED_FROM = {"pra": ("pts", "reb", "ast")}
PCTS = [10, 25, 50, 75, 90]
MIN_MEAN_MINUTES = 1.0
GAME_FILES = ("summary.json", "compact.json", "players.json", "players_dist.json", "seeds.json")

ASCII_ALNUM = set("abcdefghijklmnopqrstuvwxyz0123456789")


# ------------------------------------------------------------------------------------------------ naming / json
def name_key(s: str) -> str:
    """The site's exact join normalizer (`s.toLowerCase().replace(/[^a-z0-9]+/g,"")`), ASCII-only on purpose."""
    return "".join(ch for ch in str(s).lower() if ch in ASCII_ALNUM)


def make_slug(home: str, away: str) -> str:
    return f"{name_key(home)}_{name_key(away)}"


def jnum(x):
    """JSON-safe number: numpy -> python, NaN -> None, floats rounded to 4 places."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if np.isnan(f) else round(f, 4)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def day_label(day: str) -> str:
    """'Mon Nov 2' for 2026-11-02 (no zero padding, portable on Windows)."""
    d = datetime.strptime(day, "%Y-%m-%d")
    return f"{d.strftime('%a %b')} {d.day}"


# ------------------------------------------------------------------------------------------------ sources
def newest_run_id(root: Path, day: str) -> str:
    """The newest FINISHED run of the day (`_DONE.json` mtime)."""
    base = root / "sim" / day
    runs = [p for p in base.glob("*") if p.is_dir() and (p / "_DONE.json").exists()]
    if not runs:
        raise FileNotFoundError(f"no finished sim run under {base}")
    return max(runs, key=lambda p: (p / "_DONE.json").stat().st_mtime).name


def finished_runs(root: Path, day: str) -> list[str]:
    """Every finished run of the day that carries the three sim files, NEWEST first (`_DONE.json` mtime)."""
    base = root / "sim" / day
    runs = [p for p in base.glob("*") if p.is_dir() and (p / "_DONE.json").exists()
            and all((p / f).exists() for f in ("games.parquet", "players.parquet", "slate.parquet"))]
    runs.sort(key=lambda p: ((p / "_DONE.json").stat().st_mtime, p.name), reverse=True)
    return [p.name for p in runs]


def newest_publish_dir(root: Path, day: str, run_id: str) -> Path | None:
    """The newest publish of the run (publish_meta.json `published_at`, mtime as the tie-break); None when never published."""
    base = root / "publish" / day / run_id
    if not base.exists():
        return None
    cands = []
    for p in base.glob("*"):
        if not p.is_dir() or not (p / "slate.parquet").exists():
            continue
        when = ""
        mp = p / "publish_meta.json"
        if mp.exists():
            try:
                when = str(json.loads(mp.read_text(encoding="utf-8")).get("published_at") or "")
            except Exception:  # noqa: BLE001 -- provenance is best effort
                when = ""
        cands.append((when, p.stat().st_mtime, p))
    return max(cands)[2] if cands else None


def load_run(root: Path, day: str, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    sd = root / "sim" / day / run_id
    for f in ("games.parquet", "players.parquet", "slate.parquet"):
        if not (sd / f).exists():
            raise FileNotFoundError(f"missing sim file: {sd / f}")
    games = pd.read_parquet(sd / "games.parquet")
    players = pd.read_parquet(sd / "players.parquet")
    slate = pd.read_parquet(sd / "slate.parquet")
    meta = json.loads((sd / "run_meta.json").read_text(encoding="utf-8")) if (sd / "run_meta.json").exists() else {}
    return games, players, slate, meta


def load_espn_ids(crosswalk: Path = CROSSWALK) -> set[int]:
    """The chain's team ids are ESPN ids already; the crosswalk tells us which ones the site can build a logo for."""
    if not crosswalk.exists():
        return set()
    return set(pd.read_parquet(crosswalk, columns=["espn_team_id"])["espn_team_id"].dropna().astype(int))


def load_roster(preseason_roster: Path | None = None, espn_roster: Path = ESPN_ROSTER) -> pd.DataFrame:
    """athlete_id (ESPN player id) -> name, position. Preseason roster first when it carries players, else the ESPN pull."""
    frames = []
    cands = [preseason_roster] if preseason_roster else []
    if preseason_roster is None:
        try:
            from cbb_sim.live.preseason import preseason_dir
            cands.append(preseason_dir() / f"roster_players_{SEASON}.parquet")
        except Exception:  # noqa: BLE001 -- the override file may be absent in a test checkout
            pass
    cands.append(espn_roster)
    for p in cands:
        if p is None or not Path(p).exists():
            continue
        r = pd.read_parquet(p)
        idcol = next((c for c in ("espn_player_id", "athlete_id", "source_id") if c in r.columns), None)
        if idcol is None or "name" not in r.columns or len(r) == 0:
            continue
        ids = pd.to_numeric(r[idcol], errors="coerce")
        out = pd.DataFrame({"athlete_id": ids, "name": r["name"].astype(str),
                            "position": r["position"].astype(str) if "position" in r.columns else ""})
        out = out.dropna(subset=["athlete_id"])
        out["athlete_id"] = out["athlete_id"].astype("int64")
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["athlete_id", "name", "position"]).set_index("athlete_id")
    allr = pd.concat(frames, ignore_index=True).drop_duplicates("athlete_id", keep="first")
    allr["position"] = allr["position"].fillna("").replace({"nan": "", "None": "", "<NA>": ""})
    return allr.set_index("athlete_id")


def load_verified_finals(path: Path | None, game_ids: list[int]) -> dict[int, tuple[int, int]]:
    """game_id -> (home, away) from the ingest's verified finals: both sources final and agreeing on both scores."""
    if path is None or not Path(path).exists():
        return {}
    v = pd.read_parquet(path)
    need = {"game_id", "hoopr_status", "hoopr_home", "hoopr_away", "cbbd_status", "cbbd_home", "cbbd_away"}
    if not need <= set(v.columns):
        return {}
    v = v[v["game_id"].isin(game_ids)]
    v = v[v["hoopr_status"].astype(str).str.upper().str.contains("FINAL") & v["cbbd_status"].astype(str).str.lower().eq("final")]
    v = v.dropna(subset=["hoopr_home", "hoopr_away", "cbbd_home", "cbbd_away"])
    v = v[(v["hoopr_home"].astype(float) == v["cbbd_home"].astype(float))
          & (v["hoopr_away"].astype(float) == v["cbbd_away"].astype(float))]
    v = v.drop_duplicates("game_id", keep="last")
    return {int(r.game_id): (int(r.hoopr_home), int(r.hoopr_away)) for r in v.itertuples()}


def load_finals(root: Path, game_ids: list[int], verified_path: Path | None = None) -> dict[int, tuple[int, int]]:
    """game_id -> (home_score, away_score): the grade ledger first, then the ingest's verified finals for games the ledger
    does not have (the ledger is empty until the next day's grade stage). Never fabricated: no row, no final."""
    out: dict[int, tuple[int, int]] = {}
    p = root / "grade" / "ledger.parquet"
    if p.exists():
        led = pd.read_parquet(p)
        if {"game_id", "home_score", "away_score"} <= set(led.columns):
            led = led[led["game_id"].isin(game_ids)].dropna(subset=["home_score", "away_score"])
            led = led.drop_duplicates("game_id", keep="last")
            out = {int(r.game_id): (int(r.home_score), int(r.away_score)) for r in led.itertuples()}
    for gid, fin in load_verified_finals(verified_path, [g for g in game_ids if g not in out]).items():
        out[gid] = fin
    return out


# ------------------------------------------------------------------------------------------------ odds
def odds_block(prow: pd.Series | None, lines: pd.DataFrame | None, cbbd_game_id: int | None) -> dict:
    """Football keys. current = the stored line at publish time (publish slate `spread` / `total`), open = `*_open` when present
    else the same line; provider_count = providers in the publish's lines.parquet for the game, else 1 / 0 from the slate."""
    empty = {"spread_open": None, "over_under_open": None, "spread_current": None, "over_under_current": None,
             "open_date": None, "provider_count": 0}
    if prow is None:
        return empty

    def g(col):
        v = prow.get(col, np.nan) if col in prow.index else np.nan
        return None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else float(v)

    spread, total = g("spread"), g("total")
    spread_open, total_open = g("spread_open"), g("total_open")
    n_prov = 0
    if lines is not None and len(lines) and cbbd_game_id is not None and "cbbd_game_id" in lines.columns:
        n_prov = int(lines[lines["cbbd_game_id"] == cbbd_game_id]["provider"].nunique())
    if n_prov == 0:
        prov = prow.get("provider", None) if "provider" in prow.index else None
        n_prov = 0 if prov is None or pd.isna(prov) else 1
    open_date = None
    if "line_fetched_at" in prow.index and not pd.isna(prow["line_fetched_at"]):
        open_date = pd.Timestamp(prow["line_fetched_at"]).strftime("%Y-%m-%d")
    return {
        "spread_open": jnum(spread_open if spread_open is not None else spread),
        "over_under_open": jnum(total_open if total_open is not None else total),
        "spread_current": jnum(spread),
        "over_under_current": jnum(total),
        "open_date": open_date if (spread is not None or total is not None) else None,
        "provider_count": n_prov if (spread is not None or total is not None) else 0,
    }


# ------------------------------------------------------------------------------------------------ per-game builders
def lines_entry(gid: int, home: str, away: str, prow, odds: dict, home_pts: np.ndarray, away_pts: np.ndarray) -> dict | None:
    """One game of the day's lines.json (the football week `lines.json` shape, parsed by the site's weekLines.ts): home-POV
    spread/total with open/close, american ML, and the sim's own side probabilities at each line, computed from the same seed
    series the compact file carries (push excluded from a side, so a pair can sum below 1; ML = A_win_prob, ties split).
    None when the game carries no stored line at all."""
    if prow is None or (odds["spread_current"] is None and odds["over_under_current"] is None):
        return None

    def g(col):
        v = prow.get(col, np.nan) if col in prow.index else np.nan
        return None if v is None or pd.isna(v) else float(v)

    n_prov = int(odds["provider_count"]) or 1
    sp_c, sp_o = g("spread"), g("spread_open")
    to_c, to_o = g("total"), g("total_open")
    sp_o = sp_c if sp_o is None else sp_o
    to_o = to_c if to_o is None else to_o
    hm, am = g("home_ml"), g("away_ml")
    margin, total = home_pts - away_pts, home_pts + away_pts
    n = len(margin)

    def spread_p(line):
        return None if line is None else {"home": jnum(float((margin + line > 0).sum() / n)),
                                          "away": jnum(float((margin + line < 0).sum() / n))}

    def total_p(line):
        return None if line is None else {"over": jnum(float((total > line).sum() / n)),
                                          "under": jnum(float((total < line).sum() / n))}

    p_home = float((margin > 0).mean() + 0.5 * (margin == 0).mean())
    return {
        "game_id": gid, "teamA": home, "teamB": away,
        "spread": {"open": jnum(sp_o), "close": jnum(sp_c), "n_open": n_prov if sp_o is not None else 0,
                   "n_close": n_prov if sp_c is not None else 0},
        "total": {"open": jnum(to_o), "close": jnum(to_c), "n_open": n_prov if to_o is not None else 0,
                  "n_close": n_prov if to_c is not None else 0},
        # the stored ML is the publish-time price (ml_is_close_proxy): open and close are the same snapshot
        "ml": {"home_open": jnum(hm), "away_open": jnum(am), "home_close": jnum(hm), "away_close": jnum(am),
               "n_open": n_prov if hm is not None else 0, "n_close": n_prov if hm is not None else 0},
        "sim": {"n": int(n), "spread": {"open": spread_p(sp_o), "close": spread_p(sp_c)},
                "total": {"open": total_p(to_o), "close": total_p(to_c)},
                "ml": {"home": jnum(p_home), "away": jnum(1 - p_home)}},
    }


def build_compact(home_pts: np.ndarray, away_pts: np.ndarray) -> dict:
    """Verbatim football `build_compact`: 1..99 quantiles, 2-point histograms, per-seed integer series."""
    margin = home_pts - away_pts          # home minus away
    total = home_pts + away_pts
    levels = list(range(1, 100))

    def q(a):
        return [jnum(v) for v in np.percentile(a, levels)]

    def hist(a, width=2):
        lo = np.floor(a.min() / width) * width
        hi = np.ceil((a.max() + 1e-9) / width) * width
        if hi <= lo:
            hi = lo + width
        edges = np.arange(lo, hi + width, width)
        counts, _ = np.histogram(a, bins=edges)
        return [{"start": float(edges[i]), "end": float(edges[i + 1]), "count": int(counts[i])}
                for i in range(len(counts)) if counts[i] > 0]

    return {
        "convention": CONVENTION,
        "n_sims": int(len(home_pts)),
        "quantile_levels": levels,
        "quantiles": {"margin": q(margin), "total": q(total), "teamA_pts": q(home_pts), "teamB_pts": q(away_pts)},
        "hist": {"margin": hist(margin), "total": hist(total)},
        # CBB-scoreboard compatible raw series. NOTE `spreads` follows the page's B-minus-A sign, the opposite of `margin`.
        "A_pts": [int(v) for v in home_pts],
        "B_pts": [int(v) for v in away_pts],
        "totals": [int(v) for v in total],
        "spreads": [int(v) for v in -margin],
    }


def derive_player_stats(pl: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns (fgm, ftm, pra) the site stat map needs."""
    pl = pl.copy()
    pl["fgm"] = pl["fgm2_rim"].astype("int64") + pl["fgm2_jump"].astype("int64") + pl["fgm3"].astype("int64")
    pl["ftm"] = pl["pts"].astype("int64") - 2 * (pl["fgm2_rim"].astype("int64") + pl["fgm2_jump"].astype("int64")) \
        - 3 * pl["fgm3"].astype("int64")
    pl["pra"] = pl["pts"].astype("int64") + pl["reb"].astype("int64") + pl["ast"].astype("int64")
    return pl


def active_stats(players_day: pd.DataFrame) -> tuple[dict, list[str]]:
    """(src -> site stat map, dropped site keys) for the day: a stat whose value is zero for EVERY player row of the day is
    dropped, and a derived stat is dropped with any of its inputs (`pra` with `ast`)."""
    pl = derive_player_stats(players_day) if len(players_day) else players_day
    dropped = []
    for src, dst in PLAYER_STATS.items():
        if len(pl) == 0 or src not in pl.columns or float(np.abs(pl[src].to_numpy(dtype=float)).sum()) == 0.0:
            dropped.append(dst)
    for dst, parts in DERIVED_FROM.items():
        if dst not in dropped and any(p in dropped for p in parts):
            dropped.append(dst)
    stats = {src: dst for src, dst in PLAYER_STATS.items() if dst not in dropped}
    return stats, dropped


def build_players(pl_game: pd.DataFrame, seeds: np.ndarray, home: str, away: str, home_id: int, away_id: int,
                  roster: pd.DataFrame, stats: dict | None = None) -> tuple[list[dict], list[dict], dict, dict]:
    """Return (summaries, distributions, series, gaps) built from ONE integer per-seed array per player-stat, so the PMFs and
    the seed-aligned series are lossless encodings of exactly the sample the quantiles were taken over (football rule).

    Minutes are the only non-integral stat; they are rounded once here (a box score prints whole minutes) and every file reads
    from that array. DNP seeds are zero-filled so every stat's PMF mass == nsims. Players averaging < 1 minute are dropped.
    `stats` = the day's src -> site key map (active_stats); default = every stat.
    """
    stats = dict(PLAYER_STATS) if stats is None else stats
    seed_stats = [s for s in SEED_STATS if s in stats.values()]
    n = len(seeds)
    if not np.all(np.diff(seeds) > 0):
        raise ValueError("seed vector is not strictly increasing -- seed alignment between games and players cannot be trusted")
    seed_index = pd.Index(seeds, name="seed")
    pl_game = derive_player_stats(pl_game)
    team_name = {int(home_id): home, int(away_id): away}
    gaps = {"unnamed": [], "unknown_team": []}
    summaries, dists, series = [], [], {}
    rows = []
    for (team_id, aid), grp in pl_game.groupby(["team_id", "athlete_id"], sort=False):
        per_seed = grp.groupby("seed")[sorted(set(stats) | {"minutes"})].sum().reindex(seed_index, fill_value=0.0)
        if float(per_seed["minutes"].mean()) < MIN_MEAN_MINUTES:
            continue
        team = team_name.get(int(team_id))
        if team is None:
            gaps["unknown_team"].append(int(aid))
            continue
        if int(aid) in roster.index:
            name = str(roster.at[int(aid), "name"])
            role = str(roster.at[int(aid), "position"] or "")
        else:
            name, role = f"Player {int(aid)}", ""
            gaps["unnamed"].append(int(aid))
        rows.append((team, name, role, int(aid), per_seed, int((grp.groupby("seed").size() > 0).sum())))

    # seeds.json keys players by NAME ALONE (the site builds "<player>|<stat>" from its picker). Two men with one name on
    # opposite sides get the team appended (football rule); same name on one side gets the jersey-less id appended.
    names_teams = {}
    for team, name, *_ in rows:
        names_teams.setdefault(name, set()).add(team)
    seen: dict[tuple[str, str], int] = {}
    for team, name, role, aid, per_seed, used in rows:
        player = f"{name} ({team})" if len(names_teams[name]) > 1 else name
        seen[(team, player)] = seen.get((team, player), 0) + 1
        if seen[(team, player)] > 1:
            player = f"{player} #{aid}"
        head = {"player": player, "team": team, "role": role}
        entry = {**head, "n_sims": n, "sims_used": used}
        stats_pmf = {}
        for src, dst in stats.items():
            v = np.rint(per_seed[src].to_numpy(dtype=float)).astype(np.int64)
            pcts = np.percentile(v, PCTS)
            entry[dst] = {"mean": round(float(v.mean()), 2),
                          **{f"p{p}": round(float(pcts[i]), 2) for i, p in enumerate(PCTS)}}
            vals, counts = np.unique(v, return_counts=True)
            stats_pmf[dst] = {"pmf": {str(int(a)): int(b) for a, b in zip(vals, counts)}}
            if dst in seed_stats and v.any():            # all-zero columns dropped, as football does
                series[f"{player}|{dst}"] = v
        summaries.append(entry)
        dists.append({**head, "stats": stats_pmf})

    # home team first, then by minutes -- deterministic file bytes
    weight = {(e["team"], e["player"]): (0 if e["team"] == home else 1, -e.get("min", {}).get("mean", 0.0), e["player"])
              for e in summaries}
    key = lambda e: weight[(e["team"], e["player"])]          # noqa: E731
    summaries.sort(key=key)
    dists.sort(key=key)
    names = [e["player"] for e in summaries]
    dupes = sorted({x for x in names if names.count(x) > 1})
    if dupes:
        raise ValueError(f"name collision inside one game -- seeds.json keys would be ambiguous for {dupes}")
    return summaries, dists, series, gaps


def engine_block(meta: dict, run_id: str, day: str) -> dict:
    """Provenance of the run behind this day (football keys): tag, model, git, seeds, rulebook, note."""
    seeds = meta.get("seeds")
    n_seeds = len(seeds) if isinstance(seeds, list) else (int(seeds) if isinstance(seeds, (int, float)) else None)
    return {
        "tag": str(meta.get("engine_tag") or f"daily/{day}/{run_id}"),
        "model": None,
        "git": None,
        "seeds": n_seeds,
        "rulebook": None,
        "note": ("no decision rulebook has been measured on this engine; labels are the plain net-edge bar and the regime "
                 "is shown for information only, graded by the regime scorecard after settlement"),
    }


# ------------------------------------------------------------------------------------------------ main build
def build_day(day: str, run_id: str | None = None, root: Path = DAILY_ROOT, stage_root: Path = DEFAULT_STAGE_ROOT,
              hub_repo: str | None = None, token: str | None = None, roster: pd.DataFrame | None = None,
              espn_ids: set[int] | None = None, publish_dir: Path | None = None,
              verified_finals: Path | None = None) -> dict:
    """Build `<stage_root>/2027/days/<day>/` from the run and merge the day into `<stage_root>/2027/season_index.json`.

    Returns a report dict: run_id, publish_id, day_dir, n_games, games (per-game slug / counts / sizes), gaps.
    `hub_repo` (+ token) only makes the season-index merge read the Hub copy first; nothing is uploaded here."""
    root, stage_root = Path(root), Path(stage_root)
    run_ids = [run_id] if run_id else finished_runs(root, day)
    if not run_ids:
        raise FileNotFoundError(f"no finished sim run under {root / 'sim' / day}")
    run_id = run_ids[0]                       # day-level source_tag = the newest run
    # newest run first; a game is taken from the first (newest) run that contains it
    units: dict[int, dict] = {}
    run_meta: dict[str, dict] = {}
    for rid in run_ids:
        r_games, r_players, r_slate, r_meta = load_run(root, day, rid)
        run_meta[rid] = r_meta
        pdir = publish_dir if (publish_dir is not None and rid == run_id) else newest_publish_dir(root, day, rid)
        r_pub = pd.read_parquet(pdir / "slate.parquet").set_index("game_id") if pdir else None
        r_lines = pd.read_parquet(pdir / "lines.parquet") if pdir and (pdir / "lines.parquet").exists() else None
        for srow in r_slate.itertuples(index=False):
            gid = int(srow.game_id)
            if gid in units:
                continue
            units[gid] = {"run": rid, "srow": srow, "games": r_games[r_games["game_id"] == gid], "pub": r_pub, "lines": r_lines,
                          "players": r_players[r_players["game_id"] == gid], "publish_id": pdir.name if pdir else None}
    ordered = sorted(units.values(), key=lambda u: (pd.Timestamp(u["srow"].tipoff_utc), int(u["srow"].game_id)))
    from_runs = sorted({u["run"] for u in ordered})
    meta = run_meta[run_id]
    publish_id = next((u["publish_id"] for u in ordered if u["run"] == run_id), None)
    players = pd.concat([u["players"] for u in ordered], ignore_index=True) if ordered else pd.DataFrame()
    game_ids = [int(u["srow"].game_id) for u in ordered]
    roster = load_roster() if roster is None else roster
    espn_ids = load_espn_ids() if espn_ids is None else espn_ids
    finals = load_finals(root, game_ids, VERIFIED_FINALS if verified_finals is None else verified_finals)
    stats, dropped_stats = active_stats(players)
    stat_keys = sorted(stats.values())
    seed_stats = [s for s in SEED_STATS if s in stats.values()]
    if dropped_stats:
        print(f"[export] stat keys dropped for {day} (zero for every player on the day): {dropped_stats}")
    if len(from_runs) > 1:
        print(f"[export] union of {len(run_ids)} run(s): " +
              ", ".join(f"{r}={sum(1 for u in ordered if u['run'] == r)}" for r in from_runs))

    season_dir = stage_root / str(SEASON)
    day_dir = season_dir / "days" / day
    if day_dir.exists():
        shutil.rmtree(day_dir)                # rebuild clean: no stale slugs
    (day_dir / "games").mkdir(parents=True, exist_ok=True)
    updated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    lines_games: dict[str, dict] = {}
    index_games, report, gaps = [], [], {"missing_espn_id": [], "unnamed": [], "unknown_team": [], "missing_lines": [],
                                         "no_players": [], "missing_sim_rows": [], "placeholder_tip": [],
                                         "dropped_stats": list(dropped_stats)}
    for unit in ordered:
        srow, gg, pub, lines, game_run = unit["srow"], unit["games"].sort_values("seed"), unit["pub"], unit["lines"], unit["run"]
        gid = int(srow.game_id)
        home, away = str(srow.home_name), str(srow.away_name)
        home_id, away_id = int(srow.home_team_id), int(srow.away_team_id)
        if gg.empty:
            gaps["missing_sim_rows"].append(gid)
            continue
        slug = make_slug(home, away)
        seeds = gg["seed"].to_numpy()
        home_pts = gg["home_pts"].to_numpy(dtype=float)
        away_pts = gg["away_pts"].to_numpy(dtype=float)
        margin, total = home_pts - away_pts, home_pts + away_pts
        n_sims = int(len(seeds))
        tip_utc = pd.Timestamp(srow.tipoff_utc)
        tip_utc = tip_utc.tz_localize("UTC") if tip_utc.tzinfo is None else tip_utc.tz_convert("UTC")
        tip_et = tip_utc.tz_convert(ET)
        # A placeholder tip (date-only feed row, chain `tip_time_is_placeholder`) is not a time: the card shows TBD, not a
        # wrong hour. A real, verified tip is published as the source gives it.
        placeholder = bool(getattr(srow, "tip_time_is_placeholder", False))
        time_utc = None if placeholder else tip_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        time_et = "TBD" if placeholder else tip_et.strftime("%H:%M")
        date_str = day if placeholder else tip_et.strftime("%Y-%m-%d")
        if placeholder:
            gaps["placeholder_tip"].append(gid)
        a_espn = home_id if home_id in espn_ids else None
        b_espn = away_id if away_id in espn_ids else None
        for tid, nm in ((home_id, home), (away_id, away)):
            if tid not in espn_ids:
                gaps["missing_espn_id"].append({"team_id": tid, "team": nm})

        prow = pub.loc[gid] if (pub is not None and gid in pub.index) else None
        odds = odds_block(prow, lines, int(srow.cbbd_game_id) if hasattr(srow, "cbbd_game_id") else None)
        if odds["spread_current"] is None and odds["over_under_current"] is None:
            gaps["missing_lines"].append(gid)
        fin = finals.get(gid)
        summary = {
            "convention": CONVENTION,
            "sport": SPORT, "season": SEASON, "day": day,
            "game_id": gid, "slug": slug,
            "teamA": home, "teamB": away,
            "A_espn_id": a_espn, "B_espn_id": b_espn,
            "neutral_site": bool(getattr(srow, "neutral", False)),
            "date": date_str,
            "time_utc": time_utc,
            "time_et": time_et,
            "A_win_prob": jnum(float((margin > 0).mean() + 0.5 * (margin == 0).mean())),
            "median_margin": jnum(float(np.median(margin))),
            "median_total": jnum(float(np.median(total))),
            "p25_margin": jnum(float(np.percentile(margin, 25))),
            "p75_margin": jnum(float(np.percentile(margin, 75))),
            "p25_total": jnum(float(np.percentile(total, 25))),
            "p75_total": jnum(float(np.percentile(total, 75))),
            "mean_margin": jnum(float(margin.mean())),
            "mean_total": jnum(float(total.mean())),
            "mean_A_pts": jnum(float(home_pts.mean())),
            "mean_B_pts": jnum(float(away_pts.mean())),
            "median_A_pts": jnum(float(np.median(home_pts))),
            "median_B_pts": jnum(float(np.median(away_pts))),
            "nsims": n_sims,
            "updated": updated,
            "finalA": fin[0] if fin else None, "finalB": fin[1] if fin else None,
            "odds": odds,
            "source_tag": game_run,
        }
        compact = build_compact(home_pts, away_pts)
        compact.update(game_id=gid, slug=slug, teamA=home, teamB=away, updated=updated)

        pg = unit["players"]
        psum, dists, series, pgaps = build_players(pg, seeds, home, away, home_id, away_id, roster, stats)
        gaps["unnamed"].extend(pgaps["unnamed"])
        gaps["unknown_team"].extend(pgaps["unknown_team"])
        if not psum:
            gaps["no_players"].append(gid)
        players_doc = {
            "convention": CONVENTION, "game_id": gid, "slug": slug, "teamA": home, "teamB": away,
            "n_sims": n_sims, "updated": updated, "stat_keys": stat_keys,
            "percentiles": [f"p{p}" for p in PCTS], "players": psum,
        }
        dist_doc = {
            "convention": CONVENTION, "game_id": gid, "slug": slug, "teamA": home, "teamB": away,
            "nsims": n_sims, "updated": updated, "stat_keys": stat_keys,
            # sparse {stringified int value -> seed count}; every stat's mass sums to nsims (DNP seeds zero-filled)
            "players": dists,
        }
        # Seed-aligned joint block: A_pts[i], B_pts[i] and every player column index i are THE SAME simulated game.
        seeds_doc = {
            "convention": CONVENTION, "game_id": gid, "slug": slug, "teamA": home, "teamB": away,
            "nsims": n_sims, "updated": updated,
            "seed_order": "ascending seed; all arrays share this index",
            "stat_keys": seed_stats,
            "A_pts": [int(v) for v in home_pts],
            "B_pts": [int(v) for v in away_pts],
            "players": {k: [int(x) for x in v] for k, v in series.items()},
        }
        gdir = day_dir / "games" / slug
        gdir.mkdir(parents=True, exist_ok=True)
        for name, doc in (("summary.json", summary), ("compact.json", compact), ("players.json", players_doc),
                          ("players_dist.json", dist_doc), ("seeds.json", seeds_doc)):
            (gdir / name).write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")

        rel = f"days/{day}/games/{slug}"
        index_games.append({
            "game_id": gid, "slug": slug, "teamA": home, "teamB": away,
            "A_espn_id": a_espn, "B_espn_id": b_espn,
            "date": summary["date"], "time_utc": summary["time_utc"], "n_sims": n_sims, "source_tag": game_run,
            "summary_path": f"{rel}/summary.json", "compact_path": f"{rel}/compact.json",
            "players_path": f"{rel}/players.json", "players_dist_path": f"{rel}/players_dist.json",
            "seeds_path": f"{rel}/seeds.json",
        })
        le = lines_entry(gid, home, away, prow, odds, home_pts, away_pts)
        if le is not None:
            lines_games[slug] = le
        report.append({"slug": slug, "game_id": gid, "n_sims": n_sims, "players": len(psum), "seed_cols": len(series),
                       "providers": odds["provider_count"], "sizes": {f: (gdir / f).stat().st_size for f in GAME_FILES}})

    # lines.json: the site's betting-record strip (Value / +Dogs / By edge / PnL) reads <day>/lines.json (weekLines.ts). `week` is
    # the YYYYMMDD integer the site's feedPeriod uses for days.
    (day_dir / "lines.json").write_text(json.dumps({
        "season": SEASON, "week": int(day.replace("-", "")), "generated_at": updated,
        "convention": ("spread/total home-POV, negative = home favored; ML american sign; the stored line at publish time "
                       "(one snapshot: open = the slate's opener where present, close = the publish-time line); sim "
                       "probabilities are exact over the day's seed series"),
        "spread_total_price": -110, "sim_tag": run_id, "games": lines_games}, separators=(",", ":")), encoding="utf-8")
    has_open = any(r["providers"] > 0 for r in report)
    day_index = {
        "sport": SPORT, "season": SEASON, "day": day, "label": day_label(day),
        "folder_original": day, "legacy_key": day,
        "convention": CONVENTION,
        "n_games": len(index_games), "has_open": has_open,
        "updated": updated, "source_tag": run_id,
        "engine": engine_block(meta, run_id, day),
        "games": index_games,
    }
    (day_dir / "index.json").write_text(json.dumps(day_index, indent=1), encoding="utf-8")
    entry = {"day": day, "id": day, "dir": f"days/{day}", "label": day_index["label"], "legacy_key": day,
             "n_games": len(index_games), "has_open": has_open, "updated": updated, "engine": day_index["engine"]}
    write_season_index(season_dir, entry, hub_repo, token)
    return {"run_id": run_id, "runs": from_runs, "publish_id": publish_id, "day_dir": str(day_dir), "n_games": len(index_games),
            "games": report, "gaps": gaps, "updated": updated}


def hub_season_index(hub_repo: str, token: str | None) -> dict | None:
    """The Hub's current season_index.json (None when the repo / file does not exist yet)."""
    url = f"https://huggingface.co/datasets/{hub_repo}/resolve/main/{SEASON}/season_index.json"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code not in (401, 403, 404):
            print(f"[warn] hub season_index fetch: {e}")
    except Exception as e:                                # noqa: BLE001
        print(f"[warn] hub season_index unreachable ({e}); local merge only")
    return None


def write_season_index(season_dir: Path, entry: dict, hub_repo: str | None, token: str | None = None) -> dict:
    """Merge this day into season_index.json -- local stage first, then the Hub copy if we are about to upload, so other
    days are never clobbered. The Hub is authoritative for the days this run is not touching."""
    days: dict[str, dict] = {}
    local = season_dir / "season_index.json"
    if local.exists():
        for d in json.loads(local.read_text(encoding="utf-8")).get("days", []):
            days[str(d["day"])] = d
    if hub_repo:
        hub = hub_season_index(hub_repo, token)
        if hub:
            for d in hub.get("days", []):
                if str(d["day"]) != str(entry["day"]):
                    days[str(d["day"])] = d
            print(f"[export] merged {len(days)} day(s) from the Hub index")
    days[str(entry["day"])] = entry
    doc = {
        "sport": SPORT, "season": SEASON,
        "convention": CONVENTION,
        "layout": {
            "season_index": f"{SEASON}/season_index.json",
            "day_index": f"{SEASON}/days/<date>/index.json",
            "game_summary": f"{SEASON}/days/<date>/games/<slug>/summary.json",
            "game_compact": f"{SEASON}/days/<date>/games/<slug>/compact.json",
            "game_players": f"{SEASON}/days/<date>/games/<slug>/players.json",
        },
        "days": [days[k] for k in sorted(days)],
    }
    season_dir.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


# ------------------------------------------------------------------------------------------------ upload
def resolve_token() -> str | None:
    """HF_TOKEN from the environment, else this repo's .env, else cfb-props-sim/.env. Never printed."""
    tok = os.environ.get("HF_TOKEN", "").strip()
    for env_file in (REPO / ".env", CFB_ENV):
        if tok:
            break
        if env_file.exists():
            env = dict(re.findall(r"^\s*([A-Z_]+)\s*=\s*(.+?)\s*$", env_file.read_text(encoding="utf-8"), re.M))
            tok = env.get("HF_TOKEN", "").strip().strip('"').strip("'")
    return tok or None


def upload(stage_root: Path, repo_id: str, day: str, token: str | None, private: bool = False) -> dict:
    """Push the day folder + season_index.json; the day's folder on the Hub is REPLACED (delete pattern), nothing else touched.
    Returns {ok, files, missing, url, checks}."""
    from huggingface_hub import HfApi, create_repo

    token = token or resolve_token()
    if not token:
        raise RuntimeError("No HF_TOKEN (environment, .env, or cfb-props-sim/.env)")
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private, token=token)
    api = HfApi(token=token)
    stage_root = Path(stage_root)
    day_prefix = f"{SEASON}/days/{day}"
    allow = [f"{day_prefix}/**", f"{SEASON}/season_index.json"]
    files = [p for p in stage_root.rglob("*") if p.is_file()
             and not ({".cache", ".git"} & set(p.relative_to(stage_root).parts))]
    local = {str(p.relative_to(stage_root)).replace("\\", "/") for p in files}
    local = {f for f in local if f.startswith(day_prefix + "/") or f == f"{SEASON}/season_index.json"}
    size = sum((stage_root / f).stat().st_size for f in local)
    print(f"[upload] {len(local)} files / {size / 1e6:.2f}MB -> {repo_id} ({day_prefix})")
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(stage_root),
                      allow_patterns=allow, delete_patterns=[f"{day_prefix}/**"],
                      commit_message=f"site export: {SEASON} {day}")
    hub = {f for f in api.list_repo_files(repo_id, repo_type="dataset") if not f.startswith(".")}
    missing = sorted(local - hub)
    print(f"hub {len(hub)} files | local {len(local)} | missing {len(missing)}")
    for m in missing[:20]:
        print("  !!", m)
    base = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    checks = [f"{base}/{SEASON}/season_index.json", f"{base}/{day_prefix}/index.json"]
    idx = json.loads((stage_root / day_prefix / "index.json").read_text(encoding="utf-8"))
    if idx["games"]:
        for k in ("summary_path", "players_path", "players_dist_path", "seeds_path"):
            checks.append(f"{base}/{SEASON}/{idx['games'][0][k]}")
    bad = []
    print("=== hub verification (GET) ===")
    for u in checks:
        req = urllib.request.Request(u, headers={"Authorization": f"Bearer {token}"} if private else {})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
                json.loads(body)
                print(f"  {r.status}  {len(body):8d}B  {u}")
        except Exception as e:                                # noqa: BLE001
            bad.append(u)
            print(f"  FAIL {e}  {u}")
    return {"ok": not missing and not bad, "files": len(local), "bytes": size, "missing": missing, "failed_checks": bad,
            "url": f"https://huggingface.co/datasets/{repo_id}", "day_index": f"{day_prefix}/index.json"}


# ------------------------------------------------------------------------------------------------ entry points
def export_day(day: str, run_id: str | None = None, root: Path = DAILY_ROOT, stage_root: Path = DEFAULT_STAGE_ROOT,
               repo: str = DEFAULT_REPO, do_upload: bool = False, private: bool = False, quiet: bool = False) -> dict:
    """Build (and optionally upload) one day. The chain's entry point: returns the build report plus an `upload` block."""
    token = resolve_token() if do_upload else None
    rep = build_day(day, run_id, root, stage_root, hub_repo=repo if do_upload else None, token=token)
    if not quiet:
        print_report(rep)
    rep["upload"] = upload(stage_root, repo, day, token, private) if do_upload else None
    return rep


def print_report(rep: dict) -> None:
    print(f"[export] {rep['n_games']} games from run(s) {rep.get('runs') or rep['run_id']} (publish {rep['publish_id']}) -> {rep['day_dir']}")
    print(f"{'slug':40s} {'n':>5s} {'plyr':>5s} {'cols':>5s} {'prov':>4s} {'sum':>6s} {'cmp':>7s} {'ply':>7s} {'dist':>7s} {'seeds':>8s}")
    for g in rep["games"]:
        s = g["sizes"]
        print(f"{g['slug']:40s} {g['n_sims']:5d} {g['players']:5d} {g['seed_cols']:5d} {g['providers']:4d} "
              f"{s['summary.json']:6d} {s['compact.json']:7d} {s['players.json']:7d} {s['players_dist.json']:7d} {s['seeds.json']:8d}")
    gaps = rep["gaps"]
    tot = {f: sum(g["sizes"][f] for g in rep["games"]) for f in GAME_FILES}
    print(f"totals: {', '.join(f'{k} {v / 1e6:.2f}MB' for k, v in tot.items())}")
    print(f"gaps: missing_espn_id={len(gaps['missing_espn_id'])} unnamed athletes={len(set(gaps['unnamed']))} "
          f"unknown_team={len(gaps['unknown_team'])} missing_lines={len(gaps['missing_lines'])}/{rep['n_games']} "
          f"no_players={len(gaps['no_players'])} missing_sim_rows={len(gaps['missing_sim_rows'])} "
          f"placeholder_tip={len(gaps['placeholder_tip'])} dropped_stats={gaps['dropped_stats']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="slate date YYYY-MM-DD")
    ap.add_argument("--run-id", default=None, help="e.g. s2500_o0; default = the newest finished run of the day")
    ap.add_argument("--root", default=str(DAILY_ROOT), help="daily results root (sim/ publish/ grade/)")
    ap.add_argument("--stage-root", default=str(DEFAULT_STAGE_ROOT))
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--private", action="store_true", help="the site fetches unauthenticated -- keep the repo PUBLIC like cfb-sims-2026")
    a = ap.parse_args(argv)
    rep = export_day(a.date, a.run_id, Path(a.root), Path(a.stage_root), a.repo, a.upload, a.private)
    up = rep.get("upload")
    if up is not None:
        print(f"[upload] {'OK' if up['ok'] else 'INCOMPLETE'} {up['url']}  day index {up['day_index']}")
        return 0 if up["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
