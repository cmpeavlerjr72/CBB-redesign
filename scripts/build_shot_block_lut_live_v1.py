#!/usr/bin/env python
"""
build_shot_block_lut_live_v1.py -- per-slate lookup table for the served drawn block flag
(`ENGINE_SHOT_BLOCK=K2_Ocell`, adopted 2026-10-01) on LIVE engine inputs.

The backtest table `data/processed/models/engine/shot_block_<arm>_F2_2025.npz` is built by
`build_engine_shot_block_lut_v1.py` for the 5,710-game backtest slate only. A live slate (daily chain,
fold-2 replay) has its own game rows and roster slots, so the engine had no table to load
(`shot_block_K2_Ocell_.npz` FileNotFoundError). This builds the same arrays for ANY `EngineInputs`:

  * team / shooter / known / anchor: the v1 builder's own as-of functions (`asof_by_date`,
    `league_before`, imported, not copied). Every value uses events of the slate season
    strictly BEFORE the game date, and with `as_of` also strictly before the as_of date.
  * coef / mu / sd / features: read from the backtest table, not refitted. The served model
    is the F2 fit, so train and serve use one set of coefficients.
  * anchor: link(as-of league level) - link(Lbar), with Lbar and the season prior read from
    the v1 builder's meta json (exact float round trip).

Season-generic: the day-0 anchor prior is read from `engine/shot_block_prior_<season>_v1.parquet`
(build_shot_block_prior_v1.py; Lbar stays the F2 meta constant). A missing prior file raises, naming the file; never a fallback.

The table is written to a private directory (the sim stage's `_adapter`), never under
data/processed/models/engine. `inputs.meta["shot_block_lut"]` points the engine at it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_engine_shot_block_lut_v1 as B1  # noqa: E402
from cbb_sim import season_anchor as SA  # noqa: E402

ENGINE_DIR = ROOT / "data/processed/models/engine"
_EV: dict = {}


def load_prior(season: int) -> dict:
    """Per-type season-day-0 anchor prior `engine/shot_block_prior_<season>_v1.parquet` (built by build_shot_block_prior_v1.py).
    Hard stop naming the file when missing; no fallback to another season or to zero."""
    path = ENGINE_DIR / f"shot_block_prior_{season}_v1.parquet"
    if not path.exists():
        raise RuntimeError(
            f"shot_block live LUT: season {season} anchor prior file missing: {path}. Build it with "
            f"`scripts/build_shot_block_prior_v1.py --season {season}` (reads season {season - 1} blocked-miss events"
            + ("; for 2027 that is the SEALED 2025-26 season, BLOCKED until the user lifts the seal, audit window Oct 10-17" if season >= 2027 else "")
            + "). There is no silent fallback to zero or to another season.")
    t = pd.read_parquet(path)
    return {r.miss_type: float(r.prior_end_level) for r in t.itertuples()}


def season_events(season: int) -> pd.DataFrame:
    if season not in _EV:
        import train_shot_block_v1 as SB
        full, _ = SB.build_design()
        full["game_date"] = pd.to_datetime(full["game_date"]).dt.normalize()
        ev = full[full["season"] == season].copy()
        ev["blk"] = ev["blocked"].astype("float64")
        ev["n"] = 1.0
        _EV[season] = ev
    return _EV[season]


def build_table(inp, arm: str = "K2_Ocell", as_of=None, ev: pd.DataFrame | None = None) -> dict:
    g = inp.games.reset_index(drop=True)
    seasons = sorted(set(int(s) for s in g["season"]))
    if len(seasons) != 1:
        raise RuntimeError(f"shot_block live LUT: one season per slate, got {seasons}")
    prior = load_prior(seasons[0])      # hard stop (names the file) before any other read
    if ev is None:
        ev = season_events(seasons[0])
    if as_of is not None:
        t = pd.Timestamp(as_of)
        cut = (t.tz_convert(None) if t.tzinfo else t).normalize()
        ev = ev[ev["game_date"] < cut]
    G, S = len(g), inp.n_slots
    gdate = pd.to_datetime(g["game_date"]).dt.normalize().to_numpy()

    team = np.zeros((G, 2, 2), dtype="float32")
    lg = B1.league_before(gdate, ev, "blk", "n")
    for side, tcol in ((0, "home_team_id"), (1, "away_team_id")):
        keys = pd.DataFrame({"t": g[tcol].to_numpy(), "game_date": gdate})
        dn, dd = B1.asof_by_date(keys, ev.rename(columns={"def_team_id": "t"}), "t", "blk", "n")
        on, od = B1.asof_by_date(keys, ev.rename(columns={"off_team_id": "t"}), "t", "blk", "n")
        with np.errstate(invalid="ignore", divide="ignore"):
            dv, ov = dn / np.where(dd > 0, dd, np.nan), on / np.where(od > 0, od, np.nan)
        team[:, side, 0] = np.where(np.isnan(dv) | np.isnan(lg), 0.0, dv - lg)
        team[:, side, 1] = np.where(np.isnan(ov) | np.isnan(lg), 0.0, ov - lg)

    known = (inp.roster_cbbd >= 0).astype("uint8")
    evs = ev[ev["shooter_id"] >= 0]
    rid = inp.roster_cbbd.reshape(-1)
    keys = pd.DataFrame({"t": rid, "game_date": np.repeat(gdate, 2 * S)})
    sn, sd_ = B1.asof_by_date(keys, evs.rename(columns={"shooter_id": "t"}), "t", "blk", "n")
    lg_s = B1.league_before(gdate, evs, "blk", "n")
    lgr = np.repeat(np.where(np.isnan(lg_s), 0.0, lg_s), 2 * S)
    val = (sn + B1.K_SHOOTER * lgr) / (sd_ + B1.K_SHOOTER) - lgr
    shooter = np.where(rid >= 0, val, 0.0).reshape(G, 2, S).astype("float32")

    ref = np.load(ENGINE_DIR / f"shot_block_{arm}_F2_2025.npz")
    meta = json.loads((ENGINE_DIR / "shot_block_lut_F2_2025.meta.json").read_text(encoding="utf-8"))
    anchor = np.zeros((G, 3), dtype="float32")
    if arm == "K2_Ocell":
        for ti, t in enumerate(B1.TYPES):
            m = meta[f"{arm}_{t}"]
            L = B1.league_before(gdate, ev[ev["miss_type"] == t], "blk", "n")
            L = np.where(np.isnan(L), prior[t], L)
            anchor[:, ti] = (SA.link(L, "binary") - SA.link(np.float64(m["Lbar"]), "binary")).astype("float32")
    return {"game_id": g["game_id"].to_numpy(), "team": team, "shooter": shooter, "known": known,
            "anchor": anchor, "coef": ref["coef"], "mu": ref["mu"], "sd": ref["sd"], "features": ref["features"]}


def zero_seeded_sides(tab: dict, seeded_sides) -> dict:
    """PM ruling 2026-10-09 (docs/ops/): a slot filled by the A3+R1 day-1 seed has no known block rate, so the shooter term and the `known`
    flag are zeroed on every seeded (game row, side), exactly as the live-path harness does (`diag_a3_seed_wiring_v1.harness_arm`). Wiring, not a
    model change: the harness produced the selection evidence. Sides with an in-season prior (not seeded) are untouched."""
    for i, side in seeded_sides:
        tab["shooter"][i, side, :] = 0
        tab["known"][i, side, :] = 0
    return tab


def attach(inp, dest: Path, as_of=None, arm: str | None = None, seeded_sides=None) -> Path | None:
    """Build the served arm's table for `inp`, write it under `dest`, point `inp.meta` at it.
    `seeded_sides` (default None = unchanged): iterable of (game row, side) filled by the day-1 seed; see `zero_seeded_sides`.
    No-op (None) when the shot block is switched off (`ENGINE_SHOT_BLOCK=reference`)."""
    import os

    from cbb_sim.engine import shot_block as SBK
    arm = arm or os.environ.get("ENGINE_SHOT_BLOCK", SBK.DEFAULT)
    if not arm or arm == "reference":
        return None
    tab = build_table(inp, arm, as_of=as_of)
    if seeded_sides:
        zero_seeded_sides(tab, seeded_sides)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"shot_block_{arm}_live.npz"
    np.savez_compressed(path, roster_cbbd=inp.roster_cbbd, **tab)   # roster kept for audits
    lut = dict(inp.meta.get("shot_block_lut") or {})
    lut[arm] = str(path)
    inp.meta["shot_block_lut"] = lut
    return path
