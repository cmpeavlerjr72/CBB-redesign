#!/usr/bin/env python
"""
run_daily_sim_v1.py -- daily chain v3, stage SIM (2026-09-30, lane F).

For a slate date: load the slate, REFUSE every game already tipped by the (injectable) clock, build live inputs for the rest through
the live path (`build_engine_inputs_live.build_live`), run the served engine at a configurable seed count, stamp every output row
with `created_at` and `tipoff_utc`, assert `created_at < tipoff` per row, and write the results contract under

    results/daily/sim/<slate_date>/<run_id>/{games.parquet, [players.parquet], run_meta.json, slate.parquet, skipped.json, _DONE.json}

run_id defaults to `s{seeds}_o{seed_offset}`. Idempotent: a finished run with the same config hash is returned untouched; a run id
that exists with a different config is refused unless --force. No engine sampling code and no sim output is edited here.

    .venv/Scripts/python.exe scripts/run_daily_sim_v1.py --slate-date 2025-01-15 --season 2025 --seeds 20 --schedule-source universe \
        --now 2025-01-15T14:00:00Z --replay
    .venv/Scripts/python.exe scripts/run_daily_sim_v1.py --slate-date 2026-11-02 --season 2027 --schedule-source cbbd \
        --schedule-path data/raw/preseason/2027_v2_20260930/games_2027.parquet --crosswalk data/reference/team_crosswalk_v2.parquet

Season 2026 stays sealed (CBB_UNSEAL must be set by the caller; the chain never sets it for this stage).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cbb_sim.live import daily as D  # noqa: E402
from cbb_sim.live import guards as G  # noqa: E402

UNIVERSE = REPO / "data/processed/games_universe.parquet"
HOOPR_SCHED = REPO / "data/raw/hoopr/schedules"


def hoopr_tips(season: int, sched_dir: Path = HOOPR_SCHED) -> pd.DataFrame:
    """Real tip times from the hoopR schedule (`date`, valid when `time_valid`): fixes CBBD's midnight-ET placeholder for TBD games."""
    p = Path(sched_dir) / f"mbb_schedule_{int(season)}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["game_id", "tipoff_utc"])
    h = pd.read_parquet(p)
    ok = h["time_valid"].fillna(False).astype(bool) if "time_valid" in h.columns else pd.Series(True, index=h.index)
    t = pd.to_datetime(h.loc[ok, "date"], utc=True, errors="coerce")
    return pd.DataFrame({"game_id": h.loc[ok, "game_id"].astype("int64"), "tipoff_utc": t}).dropna()


def load_slate(slate_date: str, season: int, source: str, schedule_path: str | None, crosswalk: str | None,
               tips: str | None) -> pd.DataFrame:
    import build_engine_inputs_live as BL
    if source == "universe":
        slate = BL.load_slate_from_universe(slate_date, season)
        names = pd.read_parquet(UNIVERSE, columns=["game_id", "home_display_name", "away_display_name"]).rename(
            columns={"home_display_name": "home_name", "away_display_name": "away_name"})
        slate = slate.merge(names, on="game_id", how="left")
        slate["tip_source"] = "universe"
        from cbb_sim.live import tips as TP
        return TP.flag_placeholders(slate)      # explicit placeholder flag (midnight ET or no tip); replay games rows are unchanged
    slate = BL.load_slate_from_cbbd(schedule_path, slate_date, crosswalk)
    unmapped = slate.attrs.get("unmapped", [])
    g = pd.read_parquet(schedule_path, columns=["id", "homeTeam", "awayTeam", "startTimeTbd"])
    g = g.rename(columns={"id": "cbbd_game_id", "homeTeam": "home_name", "awayTeam": "away_name"})
    slate = slate.merge(g, on="cbbd_game_id", how="left")
    slate["tip_source"] = np.where(slate["startTimeTbd"].fillna(False), "cbbd_placeholder_midnight_et", "cbbd")
    if tips == "hoopr":
        ht = hoopr_tips(season).rename(columns={"tipoff_utc": "_hoopr_tip"})
        slate = slate.merge(ht, on="game_id", how="left")
        has = slate["_hoopr_tip"].notna()
        slate.loc[has, "tipoff_utc"] = slate.loc[has, "_hoopr_tip"]
        slate.loc[has, "tip_source"] = "hoopr_schedule"
        slate = slate.drop(columns="_hoopr_tip")
    if tips == "table":                       # tip-time refresh table (pull_tip_times_v1): source + placeholder flag per game
        tt = pd.read_parquet(REPO / f"data/processed/ingest/tip_times_{int(season)}.parquet",
                             columns=["game_id", "tipoff_utc", "tip_source", "tip_time_is_placeholder"])
        tt = tt.rename(columns={"tipoff_utc": "_t", "tip_source": "_s", "tip_time_is_placeholder": "_p"}).drop_duplicates("game_id", keep="last")
        slate = slate.merge(tt, on="game_id", how="left")
        has = slate["_t"].notna()                 # a game missing from the table keeps CBBD's own time, flagged by startTimeTbd
        slate.loc[has, "tipoff_utc"] = slate.loc[has, "_t"]
        slate.loc[has, "tip_source"] = slate.loc[has, "_s"]
        slate["tip_time_is_placeholder"] = np.where(has, slate["_p"].fillna(True).astype(bool), slate["startTimeTbd"].fillna(True).astype(bool))
        slate = slate.drop(columns=["_t", "_s", "_p"])
    else:
        slate["tip_time_is_placeholder"] = slate["startTimeTbd"].fillna(True).astype(bool) & (slate["tip_source"] != "hoopr_schedule")
    slate["tipoff_utc"] = pd.to_datetime(slate["tipoff_utc"], utc=True)
    slate.attrs["unmapped"] = unmapped
    return slate.drop(columns=["startTimeTbd"])


def season_start_of(season: int, source: str, schedule_path: str | None):
    if source == "universe":
        u = pd.read_parquet(UNIVERSE, columns=["season", "game_date", "is_d1_game"])
        return str(pd.to_datetime(u.loc[(u["season"] == season) & u["is_d1_game"], "game_date"]).min().date())
    g = pd.read_parquet(schedule_path, columns=["startDate"])
    d = pd.to_datetime(g["startDate"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return str(d.min().date())


def day1_prior_applies(season: int, replay: bool, day1_prior: str | None = "auto") -> bool:
    """The selected A3 day-1 player prior + R1 roster fallback (A3 selected 2026-10-05, R1 selected 2026-10-05) applies to LIVE
    serving seasons (>= 2027) only. Replay / past seasons never take it (their inputs stay bit-identical). `day1_prior=None` turns it
    off explicitly (tests, paired checks); "auto" = the served rule above."""
    if day1_prior is None:
        return False
    return int(season) >= 2027 and not replay


def day1_prior_seed(season: int):
    """(module, seed_fn) for the served A3+R1 seed, through the ONE definition the chain uses (`chain_daily_v2.day1_player_prior_seed`,
    serving=True tables). Hard stop (RuntimeError naming each table) when a source table is missing; never a silent anonymous build."""
    import chain_daily_v2 as V2
    miss = V2.day1_player_prior_missing(int(season))
    if miss:
        raise RuntimeError("A3 day-1 player prior sources missing: " + "; ".join(miss))
    return V2.day1_player_prior_seed(int(season))


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_sim_stage(slate_date: str, season: int, fold: str = "F2", seeds: int = 20, seed_offset: int = 0, now=None,
                  root: Path = D.DAILY_ROOT, run_id: str | None = None, schedule_source: str = "universe",
                  schedule_path: str | None = None, crosswalk: str | None = None, tips: str | None = None,
                  strict: bool = False, force: bool = False, replay: bool = False, players: bool = False,
                  max_games: int = 0, pass_name: str | None = None, ratings_dir: str | None = None,
                  day1_prior: str | None = "auto") -> dict:
    t0 = time.time()
    now = D.utc(now) if now is not None else pd.Timestamp.now("UTC")
    run_id = run_id or D.default_run_id(seeds, seed_offset) + (f"_{pass_name}" if pass_name == "morning" else "")
    out = D.sim_dir(root, slate_date, run_id)
    cfg = {"slate_date": slate_date, "season": season, "fold": fold, "seeds": seeds, "seed_offset": seed_offset,
           "schedule_source": schedule_source, "schedule_path": schedule_path, "tips": tips, "players": players,
           "max_games": max_games, "replay": replay}
    if pass_name:                                  # absent for the replay path, so its config hashes are unchanged
        cfg["pass"] = pass_name
    if ratings_dir:
        cfg["ratings_dir"] = str(ratings_dir)
    use_d1p = day1_prior_applies(season, replay, day1_prior)
    if use_d1p:                                    # absent for replay / past seasons, so their config hashes are unchanged
        cfg["day1_prior"] = "A3+R1"
    h = config_hash(cfg)
    done = out / "_DONE.json"
    if done.exists() and not force:
        prev = json.loads(done.read_text(encoding="utf-8"))
        if prev.get("config_hash") != h:
            raise SystemExit(f"{out} exists with a different config ({prev.get('config_hash')} vs {h}); use another --run-id or --force")
        return {"_status": "ok", "cached": True, "out": str(out), "n_rows": prev.get("n_rows"), "n_games": prev.get("n_games")}
    if int(season) == 2026 and os.environ.get("CBB_UNSEAL") != "1":
        raise SystemExit("season 2026 is SEALED; the daily chain never sets CBB_UNSEAL for the sim stage")

    slate = load_slate(slate_date, season, schedule_source, schedule_path, crosswalk, tips)
    unmapped = slate.attrs.get("unmapped", [])
    if max_games:
        slate = slate.iloc[:max_games].reset_index(drop=True)
    if strict:                                     # the assertion itself: raises LeakGuardError on any tipped game
        G.assert_created_before_tipoff(slate.assign(created_at=now))
    if pass_name:                                   # evening / morning pass (tip-time refresh table); guard rules unchanged
        from cbb_sim.live import tips as TP
        ok, late = TP.select_for_pass(slate, now, pass_name)
    else:
        ok, late = D.split_tipped(slate, now)
        late = late.assign(refuse_reason="tipoff <= created_at (or no tip time): refused, not simulated")
    out.mkdir(parents=True, exist_ok=True)
    skipped = {"clock": str(now), "pass": pass_name, "already_tipped": [
        {"game_id": int(r.game_id), "tipoff_utc": str(r.tipoff_utc), "tip_source": r.tip_source,
         "reason": r.refuse_reason} for r in late.itertuples()],
        "unmapped_non_d1": unmapped}
    (out / "skipped.json").write_text(json.dumps(skipped, indent=2, default=str), encoding="utf-8")
    for r in skipped["already_tipped"]:
        print(f"[sim] SKIP game {r['game_id']}: tipoff {r['tipoff_utc']} not after clock {now} ({r['tip_source']})", flush=True)
    if not len(ok):
        done.write_text(json.dumps({"config_hash": h, "n_rows": 0, "n_games": 0, "note": "every game already tipped"}), encoding="utf-8")
        return {"_status": "skipped", "why": "every game already tipped or no slate", "out": str(out), "skipped": len(late)}

    import build_engine_inputs_live as BL
    import run_engine_live as RL
    slate_cols = ["game_id", "cbbd_game_id", "season", "game_date", "tipoff_utc", "home_team_id", "away_team_id", "neutral"]
    D1P, seed_fn = day1_prior_seed(season) if use_d1p else (None, None)
    inp, diag = BL.build_live(ok[slate_cols], now, season, fold, created_at=now,
                              season_start=season_start_of(season, schedule_source, schedule_path), t0=t0,
                              strict_finish=not replay, ratings_dir=ratings_dir, seed_fn=seed_fn)
    if seed_fn is not None:                         # A2-only share rewrite; a no-op for A3 (kept so the call matches build_live_inputs)
        D1P.post(inp, seed_fn)
        import chain_daily_v2 as V2
        diag["d1p_fallback_line"] = V2.fallback_line(diag.get("d1p_fallback_teams"))
    named = (inp.roster_cbbd > 0)
    d1p_meta = {"day1_prior": "A3+R1" if use_d1p else None, "d1p_team_games": diag.get("d1p_team_games"),
                "d1p_slots": diag.get("d1p_slots"), "d1p_fallback_teams": [int(t) for t in (diag.get("d1p_fallback_teams") or [])],
                "d1p_fallback_line": diag.get("d1p_fallback_line"), "anon_slot_share": round(float(1.0 - named.mean()), 4),
                "team_games_all_anonymous": int((~named.any(axis=2)).sum())}
    import run_engine as RE
    prov = RE.engine_provenance()
    adir = RL.prepare_adapter_dir(inp.event_block, fold, season, out / "_adapter")
    # served shot_block K2_Ocell (adopted 2026-10-01) needs a per-slate table; as-of the clock `now`.
    import build_shot_block_lut_live_v1 as SBL
    SBL.attach(inp, out / "_adapter", as_of=now)
    seed_arr = np.arange(seed_offset, seed_offset + seeds, dtype=np.int64)
    games, pl, ad = RL.simulate(inp, fold, season, seed_arr, keep_players=players, adapter_dir=adir)
    games = RL.stamp_rows(games, inp, now, per_game=False)         # created_at + tipoff_utc, asserts created_at < tipoff
    G.assert_created_before_tipoff(games)
    if pass_name:                                   # two-pass chain: placeholder tips never certify a row as pre-tip (tips.stamp_pre_tip_basis)
        from cbb_sim.live import tips as TP
        games = TP.stamp_pre_tip_basis(games, ok, slate_date)
    games.to_parquet(out / "games.parquet", index=False)
    if players and pl is not None:
        RL.stamp_rows(pl, inp, now, per_game=False).to_parquet(out / "players.parquet", index=False)
    ok.to_parquet(out / "slate.parquet", index=False)
    meta = {"engine_tag": f"daily/{slate_date}/{run_id}", "created_at": str(now), "seeds": seed_arr.tolist(), "fold": fold,
            "backtest": False, "sealed_touched": False, "live": True, "replay": bool(replay), "season": season,
            "slate_date": slate_date, "n_games": int(inp.n_games), "n_rows": int(len(games)), "n_skipped_tipped": int(len(late)),
            "runtime_s": round(time.time() - t0, 1), "created_at_before_tipoff_asserted": True, "config_hash": h,
            "build_diag": {k: v for k, v in diag.items() if not isinstance(v, (dict, list))}, "day1_prior": d1p_meta, **prov, "adapter_flags": ad.flags,
            "engine_env": {k: v for k, v in os.environ.items() if k.startswith("ENGINE_")}}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    done.write_text(json.dumps({"config_hash": h, "n_rows": int(len(games)), "n_games": int(inp.n_games),
                                "finished_at": str(pd.Timestamp.now("UTC"))}), encoding="utf-8")
    return {"_status": "ok", "cached": False, "out": str(out), "n_games": int(inp.n_games), "n_rows": int(len(games)),
            "skipped_tipped": int(len(late)), "runtime_s": round(time.time() - t0, 1),
            "day1_prior": d1p_meta["day1_prior"], "anon_slot_share": d1p_meta["anon_slot_share"],
            "d1p_team_games": d1p_meta["d1p_team_games"], "fallback_roster_line": d1p_meta["d1p_fallback_line"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slate-date", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--now", default=None, help="injected clock (ISO UTC); default wall clock")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--root", default=str(D.DAILY_ROOT))
    ap.add_argument("--schedule-source", choices=("universe", "cbbd"), default="universe")
    ap.add_argument("--schedule-path", default=None)
    ap.add_argument("--crosswalk", default=None)
    ap.add_argument("--tips", choices=("hoopr", "table"), default=None,
                    help="hoopr: override tip times from the hoopR schedule where time_valid; table: data/processed/ingest/tip_times_{season}.parquet")
    ap.add_argument("--pass", dest="pass_name", choices=("evening", "morning"), default=None,
                    help="evening: every game tipping after the clock; morning: only games with a REAL tip still in the future")
    ap.add_argument("--strict", action="store_true", help="raise LeakGuardError on any already-tipped game instead of skipping it")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--replay", action="store_true", help="past slate with a pretend clock (strict_finish off, recorded)")
    ap.add_argument("--players", action="store_true")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--ratings-dir", default=None, help="dir holding own_ratings_{season}.parquet with the as-of row (default: the stored batch ratings)")
    ap.add_argument("--no-day1-prior", action="store_true",
                    help="season >= 2027 only: build WITHOUT the served A3+R1 day-1 player prior (paired checks; not for serving)")
    a = ap.parse_args(argv)
    r = run_sim_stage(a.slate_date, a.season, a.fold, a.seeds, a.seed_offset, a.now, Path(a.root), a.run_id, a.schedule_source,
                      a.schedule_path, a.crosswalk, a.tips, a.strict, a.force, a.replay, a.players, a.max_games, a.pass_name, a.ratings_dir,
                      day1_prior=None if a.no_day1_prior else "auto")
    print(json.dumps(r, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
