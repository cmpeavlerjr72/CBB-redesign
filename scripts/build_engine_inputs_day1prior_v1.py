#!/usr/bin/env python
"""
build_engine_inputs_day1prior_v1.py -- player-layer DAY-1 PRIORS through the default-off `build_live(..., seed_fn=...)` hook
(2026-10-05; pre-registration docs/models/player_day1/experiments.md section 1). Nothing served changes: `build_live` is called
with `seed_fn=None` everywhere else.

A seed acts ONLY on a team-game whose 15 slots are all anonymous (no in-season rotation prior = opening day). It names players in
slots 0..k-1; the live builder's own as-of joins then give each named player his fg / FT shooter blocks, usage and rebound rates
(which already read season S-1 with each served sub-model's own shrinkage). Arms:
  A0   none (anonymous)
  A1n  players with S-1 minutes for this team, by S-1 minutes (no current roster; departures included); league role-profile shares
  A1   A1n restricted to this team's season-S roster (true returners); league role-profile shares
  A2   A1 + turnover-aware shares: returner i keeps his S-1 fraction p_i of the team's minutes; the departed mass 1 - sum(p) is spread
       over the anonymous slots by the league role profile (no free parameter)
  A3   A1 + transfers in (on this team's S roster, S-1 minutes for another team), by S-1 minutes, after the returners

SOURCES (HARD STOP, no fallback, if missing or empty): S-1 on-floor possessions table (`rotation.load_team_possessions(S-1)`) and,
for A1 / A2 / A3, the CBBD roster `data/raw/cbbd/rosters/roster_{S}.parquet` (`build_player_crosswalk.pull_rosters([S])`; one
`/teams/roster?season=S` call). For 2026-27 (S = 2027) S-1 = 2025-26 is SEALED until the seal lift: `assert_not_sealed` runs first.

    # bake-off (opening window, fold 2 then fold 1): per-date builds + assemble into data/processed/models/engine_v3_d1p_<arm>{,_f1}/
    python scripts/build_engine_inputs_day1prior_v1.py window --fold F2 --arm A1
    # serving (one command, 2026-27): a slate's inputs with the seed applied to every team-game that has no in-season prior
    python scripts/build_engine_inputs_day1prior_v1.py serve --arm A1 --slate-date 2026-11-02 --as-of <ISO> --season 2027 \
        --schedule-source cbbd --schedule-path <games_2027.parquet>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import build_engine_inputs_anon_window_v1 as AW  # noqa: E402
import build_engine_inputs_live as BL  # noqa: E402
from cbb_sim.live import features as LF  # noqa: E402

ARMS = ("A0", "A1n", "A1", "A2", "A3")
FALLBACKS = (None, "R1", "R2")      # default-off roster-less-team fallback (experiments.md section 4)
ROSTER_DIR = ROOT / "data/raw/cbbd/rosters"


class SourceMissing(RuntimeError):
    """A day-1 prior source table is missing or empty. Never caught here: the build stops."""


_C: dict = {}


def prev_minutes(season: int, source: str) -> pd.DataFrame:
    """(team_id, pid, game minutes) rows of season S-1. `onfloor` = the rotation on-floor table (the registered source; exists from
    2023-24 on). `box` = hoopR player box, ESPN athlete ids mapped to CBBD ids through the CBBD rosters' `source_id` (seasons S-1, S);
    fold-1 deviation, experiments.md section 2."""
    from cbb_sim.models import rotation as ROT
    prev = season - 1
    if source == "onfloor":
        pp = Path(ROT.DEFAULT_POSS_DIR) / f"possessions_{prev}.parquet"
        if not pp.exists():
            raise SourceMissing(f"{pp} missing: day-1 priors need the S-1 on-floor possessions table")
        return ROT.player_game_minutes(ROT.load_team_possessions(prev))[["team_id", "pid", "minutes"]]
    bp = ROOT / f"data/raw/hoopr/player_box/player_box_{prev}.parquet"
    if not bp.exists():
        raise SourceMissing(f"{bp} missing")
    b = pd.read_parquet(bp, columns=["athlete_id", "team_id", "minutes", "did_not_play"]).dropna(subset=["athlete_id", "minutes"])
    b = b[(b["minutes"] > 0) & ~b["did_not_play"].fillna(False).astype(bool)]
    m = {}
    for s_ in (prev, season):
        rp = ROSTER_DIR / f"roster_{s_}.parquet"
        if rp.exists():
            r = pd.read_parquet(rp, columns=["source_id", "cbbd_player_id"]).dropna()
            m.update(dict(zip(pd.to_numeric(r["source_id"], errors="coerce"), r["cbbd_player_id"].astype("int64"))))
    b["pid"] = b["athlete_id"].astype("int64").map(m)
    print(f"box minutes {prev}: {b['pid'].notna().mean():.4f} of player-game rows mapped to CBBD ids", flush=True)
    b = b.dropna(subset=["pid"])
    return pd.DataFrame({"team_id": b["team_id"].astype("int64"), "pid": b["pid"].astype("int64"), "minutes": b["minutes"].astype(float)})


def tables(season: int, need_roster: bool, roster_path: str | None = None, minutes_source: str = "onfloor",
           need_prev_roster: bool = False, serving: bool = False) -> dict:
    key = (season, need_roster, roster_path, minutes_source, need_prev_roster, serving)
    if key in _C:
        return _C[key]
    from cbb_sim.data.seal import assert_not_sealed, assert_not_sealed_serving
    if serving:                       # explicit argument from the live serving call site only (daily chain); never defaulted on
        assert_not_sealed_serving([season - 1, season], season, context="day-1 player priors (reads season S-1 minutes)")
    else:
        assert_not_sealed([season - 1], context="day-1 player priors (reads season S-1 minutes)")
    pgm = prev_minutes(season, minutes_source)
    if not len(pgm):
        raise SourceMissing(f"no S-1 player minutes ({minutes_source})")
    tot = pgm.groupby(["team_id", "pid"])["minutes"].sum().reset_index()
    tot = tot[tot["minutes"] > 0]
    team_min = tot.groupby("team_id")["minutes"].sum().to_dict()
    by_team = {int(t): [(int(p), float(m)) for p, m in g.sort_values("minutes", ascending=False)[["pid", "minutes"]].to_numpy()]
               for t, g in tot.groupby("team_id")}
    any_min = tot.groupby("pid")["minutes"].sum().to_dict()
    out = {"by_team": by_team, "team_min": team_min, "any_min": {int(k): float(v) for k, v in any_min.items()}, "roster": None}
    if need_roster:
        rp = Path(roster_path) if roster_path else ROSTER_DIR / f"roster_{season}.parquet"
        if not rp.exists():
            raise SourceMissing(f"{rp} missing: run build_player_crosswalk.pull_rosters([{season}]) (CBBD /teams/roster)")
        ros = pd.read_parquet(rp, columns=["team_source_id", "cbbd_player_id"]).dropna()
        if not len(ros):
            raise SourceMissing(f"{rp} is empty (CBBD has not populated season {season} rosters)")
        ros["team_id"] = pd.to_numeric(ros["team_source_id"], errors="coerce")
        ros = ros.dropna(subset=["team_id"])
        out["roster"] = {int(t): set(g["cbbd_player_id"].astype("int64").tolist()) for t, g in ros.groupby("team_id")}
        out["pid_team"] = {}                            # pid -> teams whose season-S roster lists him (R2 outgoing-transfer test)
        for t, g in ros.groupby("team_id"):
            for p_ in g["cbbd_player_id"].astype("int64").tolist():
                out["pid_team"].setdefault(int(p_), set()).add(int(t))
    if need_prev_roster:
        # R1 final-year proxy from the S-1 roster file: S-1 - start_season >= 3 (4th or later D-I season). Hard stop if missing.
        pp = ROSTER_DIR / f"roster_{season - 1}.parquet"
        if not pp.exists():
            raise SourceMissing(f"{pp} missing: fallback R1/R2 needs the S-1 roster (start_season)")
        pr = pd.read_parquet(pp, columns=["cbbd_player_id", "start_season"]).dropna(subset=["cbbd_player_id"])
        pr["cbbd_player_id"] = pr["cbbd_player_id"].astype("int64")
        pr = pr.groupby("cbbd_player_id")["start_season"].min()
        out["final_year"] = {int(p_) for p_, st in pr.items() if pd.notna(st) and (season - 1) - int(st) >= 3}
    _C[key] = out
    return out


def fallback_seeds(fb: str, T: dict, team: int) -> tuple[list[int], list[float]]:
    """Roster-less team (no season-S roster): S-1 players of this team by S-1 minutes, minus final-year players (R1) and, for R2,
    minus those listed on ANOTHER team's season-S roster (observed outgoing transfers). No season-S information about this team."""
    prev = T["by_team"].get(team, [])
    keep = [(p, m) for p, m in prev if p not in T["final_year"]]
    if fb == "R2":
        keep = [(p, m) for p, m in keep if not (T["pid_team"].get(p, set()) - {team})]
    return [p for p, _ in keep], [m for _, m in keep]


def seeds_for(arm: str, T: dict, team: int, withheld: frozenset = frozenset(), fallback: str | None = None,
              used: set | None = None) -> tuple[list[int], list[float]]:
    """(player ids in slot order, their S-1 minutes at THIS team (0 for transfers)). A0 -> empty.
    `withheld` = teams whose season-S roster is treated as missing (experiment only). `fallback` (default None = off) = R1/R2 for a
    team with no season-S roster (absent from the roster file or withheld); such teams are added to `used` for the diag."""
    if arm == "A0":
        return [], []
    prev = T["by_team"].get(team, [])
    if arm == "A1n":
        return [p for p, _ in prev], [m for _, m in prev]
    if team in withheld or team not in T["roster"]:
        if fallback:
            if used is not None:
                used.add(team)
            return fallback_seeds(fallback, T, team)
        if team in withheld:
            return [], []
    ros = T["roster"].get(team, set())
    ret = [(p, m) for p, m in prev if p in ros]
    pids, mins = [p for p, _ in ret], [m for _, m in ret]
    if arm == "A3":
        prevset = {p for p, _ in prev}
        tr = sorted((p for p in ros if p not in prevset and p in T["any_min"]), key=lambda p: -T["any_min"][p])
        pids += tr
        mins += [0.0] * len(tr)
    return pids, mins


def make_seed_fn(arm: str, roster_path: str | None = None, minutes_source: str = "onfloor", fallback: str | None = None,
                 withheld: frozenset = frozenset(), serving: bool = False):
    if arm not in ARMS:
        raise KeyError(arm)
    if fallback not in FALLBACKS:
        raise KeyError(fallback)
    used: set = set()

    def seed_fn(ctx, games, tg, roster_cbbd, roster_valid, S, diag) -> dict:
        T = tables(int(ctx.season), arm in ("A1", "A2", "A3"), roster_path, minutes_source, need_prev_roster=bool(fallback), serving=serving)
        gpos = {int(g): i for i, g in enumerate(games["game_id"])}
        seed_fn.shares = {}
        n_tg = n_slots = 0
        for g, t, h in zip(tg["game_id"], tg["team_id"], tg["is_home"]):
            i, side = gpos[int(g)], (0 if bool(h) else 1)
            if (roster_cbbd[i, side] > 0).any():          # has an in-season prior: untouched
                continue
            pids, mins = seeds_for(arm, T, int(t), withheld, fallback, used)
            k = min(len(pids), S)
            if not k:
                continue
            roster_cbbd[i, side, :k] = pids[:k]
            n_tg += 1; n_slots += k
            if arm == "A2":
                tm = T["team_min"].get(int(t), 0.0)
                seed_fn.shares[(i, side)] = np.array(mins[:k]) / tm if tm > 0 else None
        diag.update({"d1p_arm": arm, "d1p_team_games": n_tg, "d1p_slots": n_slots, "d1p_fallback": fallback,
                     "d1p_fallback_teams": sorted(used)})      # which teams used the roster-less fallback (empty when off)
        return {(int(g), int(t)): [int(x) for x in roster_cbbd[gpos[int(g)], 0 if bool(h) else 1] if x > 0]
                for g, t, h in zip(tg["game_id"], tg["team_id"], tg["is_home"])}

    seed_fn.shares = {}
    return seed_fn


def post(inp, seed_fn) -> None:
    """A2 only: turnover-aware shares on the seeded rows (the row's current share vector is the league role profile)."""
    for (i, side), p in getattr(seed_fn, "shares", {}).items():
        if p is None:
            continue
        role = inp.rot_share[i, side].astype(np.float64)
        k = len(p)
        rest = role[k:]
        new = np.concatenate([p, (1.0 - p.sum()) * rest / rest.sum() if rest.sum() > 0 else np.zeros(len(rest))])
        inp.rot_share[i, side] = (new / new.sum()).astype(np.float32)


def build(slate, as_of, season, fold, arm, season_start=None, roster_path=None, anon=False, minutes_source="onfloor",
          fallback=None, withheld=frozenset(), **kw):
    """`build_live` with the arm's seed (and, for the bake-off window, the in-season rotation prior suppressed)."""
    fn = make_seed_fn(arm, roster_path, minutes_source, fallback, withheld)
    orig = LF.rotation_priors
    if anon:
        LF.rotation_priors = lambda ctx, fit, min_prior_games=1: {}
    try:
        inp, diag = BL.build_live(slate, as_of, season, fold, created_at=as_of, season_start=season_start, seed_fn=fn, **kw)
    finally:
        LF.rotation_priors = orig
    post(inp, fn)
    return inp, diag


def out_dir(fold: str, arm: str, tag: str = "") -> Path:
    return ROOT / "data/processed/models" / f"engine_v3_d1p_{arm}{tag}{'' if fold == 'F2' else '_f1'}"


def treated_teams(fold: str, frac: float = 0.2, seed: int = 20261005) -> frozenset:
    """Section 4 experiment: seeded random `frac` of the teams playing in the fold's opening window (sorted ids)."""
    _, w, _ = AW.window(fold)
    ids = np.array(sorted(set(w["home_team_id"].astype("int64")) | set(w["away_team_id"].astype("int64"))))
    pick = np.random.default_rng(seed).choice(ids, size=int(round(frac * len(ids))), replace=False)
    return frozenset(int(x) for x in pick)


def cmd_window(fold: str, arm: str, fallback: str | None = None, treated: frozenset = frozenset(), vtag: str = "") -> None:
    c = AW.CFG[fold]
    tag = f"{fold}_{c['season']}"
    g, w, s0 = AW.window(fold)
    od = out_dir(fold, arm, vtag)
    sd = od / "_stage"
    sd.mkdir(parents=True, exist_ok=True)
    BL.ENGINE_DIR = c["template"]
    t0 = time.time()
    diags = {}
    for D in sorted(w["game_date"].dt.strftime("%Y-%m-%d").unique()):
        st = f"D1P_{arm}{vtag}_{fold}_{D}"
        if (sd / f"games_{st}.parquet").exists():
            continue
        ids = w.loc[w["game_date"] == pd.Timestamp(D), "game_id"].tolist()
        slate = BL.load_slate_from_universe(D, c["season"], only_ids=ids)
        as_of = pd.to_datetime(slate["tipoff_utc"], utc=True).min() - pd.Timedelta(minutes=30)
        inp, diag = build(slate, as_of, c["season"], fold, arm, season_start=s0, anon=True, strict_finish=False,
                          minutes_source="onfloor" if fold == "F2" else "box", fallback=fallback, withheld=treated)
        inp.save(sd, st)
        diags[D] = {k: diag.get(k) for k in ("n_games", "rotation_fallback_team_games", "d1p_team_games", "d1p_slots",
                                             "d1p_fallback_teams")}
        print(f"[{time.time()-t0:6.0f}s] {D} {arm}: {diags[D]}", flush=True)
    # assemble on the anonymous-window base (same LUT: window shooter/known held anonymous for every arm, spec section 1)
    base = c["out"]
    zb = dict(np.load(base / f"arrays_{tag}.npz"))
    pos = {int(x): i for i, x in enumerate(g["game_id"])}
    arrs = {k: v.copy() for k, v in zb.items()}
    same = {}
    for D in sorted(w["game_date"].dt.strftime("%Y-%m-%d").unique()):
        st = f"D1P_{arm}{vtag}_{fold}_{D}"
        gg = pd.read_parquet(sd / f"games_{st}.parquet")
        z = np.load(sd / f"arrays_{st}.npz")
        idx = np.array([pos[int(x)] for x in gg["game_id"]])
        for k in zb:
            if k in AW.PLAYER_ARRAYS:
                arrs[k][idx] = z[k]
            else:
                ok = np.array_equal(z[k], zb[k][idx], equal_nan=True) if z[k].dtype.kind == "f" else np.array_equal(z[k], zb[k][idx])
                same[k] = same.get(k, True) and bool(ok)
    np.savez_compressed(od / f"arrays_{tag}.npz", **arrs)
    for f in (f"games_{tag}.parquet", f"event_block_{tag}.npz", f"names_{tag}.json", f"window_ids_{fold}.parquet"):
        shutil.copy2(base / f, od / f)
    win = g["game_id"].isin(w["game_id"]).to_numpy()
    rc = arrs["roster_cbbd"][win]
    rep = {"arm": arm, "fallback": fallback, "treated_teams": sorted(treated),
           "fallback_teams_used": sorted({t for d in diags.values() for t in (d.get("d1p_fallback_teams") or [])}),
           "non_player_arrays_equal_anon_base": same, "window_games": int(win.sum()),
           "named_slots_per_team_game": float((rc > 0).sum(axis=2).mean()),
           "team_games_with_any_named": float((rc > 0).any(axis=2).mean())}
    (od / "assemble_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep), flush=True)


def cmd_serve(a) -> None:
    if a.schedule_source == "universe":
        slate = BL.load_slate_from_universe(a.slate_date, a.season)
    else:
        slate = BL.load_slate_from_cbbd(a.schedule_path, a.slate_date, a.crosswalk)
    tables(a.season, a.arm in ("A1", "A2", "A3"), a.roster, a.minutes_source, need_prev_roster=bool(a.fallback))  # hard stop BEFORE any build work
    inp, diag = build(slate, a.as_of, a.season, a.fold, a.arm, season_start=a.season_start, roster_path=a.roster,
                      minutes_source=a.minutes_source, ratings_dir=a.ratings_dir, fallback=a.fallback)
    tag = a.tag or f"D1P_{a.arm}_{a.fold}_{a.season}_{a.slate_date}"
    inp.save(a.out_dir, tag)
    np.savez_compressed(Path(a.out_dir) / f"event_block_{tag}.npz", team_block=inp.event_block, cols=np.array(list(BL.BL_TEAM16)))
    (Path(a.out_dir) / f"build_diag_{tag}.json").write_text(json.dumps(diag, indent=2, default=str), encoding="utf-8")
    print(f"wrote {a.out_dir}/(games|arrays|names)_{tag}.* ; seeded team-games {diag.get('d1p_team_games')}")
    ft = diag.get("d1p_fallback_teams") or []
    print(f"{len(ft)} teams on fallback roster" + (": " + ", ".join(map(str, ft)) if ft else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["window", "serve"])
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--slate-date"); ap.add_argument("--as-of"); ap.add_argument("--season", type=int)
    ap.add_argument("--schedule-source", choices=("universe", "cbbd"), default="cbbd")
    ap.add_argument("--schedule-path"); ap.add_argument("--crosswalk", default="data/reference/team_crosswalk.parquet")
    ap.add_argument("--season-start"); ap.add_argument("--roster", default=None)
    ap.add_argument("--ratings-dir", default=None)
    ap.add_argument("--fallback", choices=("R1", "R2"), default=None,
                    help="default OFF; roster-less teams get S-1 roster minus final-year (R1) / minus outgoing transfers (R2)")
    ap.add_argument("--treated-frac", type=float, default=0.0,
                    help="window only (experiment section 4): withhold this fraction of teams' season-S rosters")
    ap.add_argument("--minutes-source", choices=("onfloor", "box"), default="onfloor")
    ap.add_argument("--out-dir", default="data/processed/models/engine_live"); ap.add_argument("--tag")
    a = ap.parse_args()
    if a.cmd == "window":
        tr = treated_teams(a.fold, a.treated_frac) if a.treated_frac > 0 else frozenset()
        cmd_window(a.fold, a.arm, a.fallback, tr, f"_{a.tag}" if a.tag else "")
    else:
        cmd_serve(a)
