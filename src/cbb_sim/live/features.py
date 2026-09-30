"""As-of feature families for a slate of UNPLAYED games (Lane G).

ONE FEATURE DEFINITION. Nothing here re-implements an as-of statistic. Each
family calls the same function the backtest designs are built with, on source
tables cut to `game_date < D`, plus zero-content STUB rows for the slate's
team-games (and, for player families, one stub row per candidate player). Every
as-of statistic in this codebase is `cumsum() - value` inside (season, entity)
ordered by (game_date, game_id), i.e. the state after all EARLIER games, so a
stub row's own features are exactly the pregame state and the stub contributes
nothing to itself. Stubs are only correct for ONE slate date per call (a stub
on D would otherwise enter a later date's league totals); `build_ctx` enforces
that the slate is a single date.

Glue that IS duplicated from the backtest builder (placement of values into
arrays, site/season/day columns) is limited to lines that carry no statistic and
is exercised by the parity test (`scripts/diag_live_parity_v1.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.live import guards as G
from cbb_sim.models import possession_outcome as PO
from cbb_sim.models import rebound as RB
from cbb_sim.ratings import own_ratings as orat

SLATE_COLS = ["game_id", "cbbd_game_id", "season", "game_date", "tipoff_utc",
              "home_team_id", "away_team_id", "neutral"]


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    slate: pd.DataFrame            # one row per game to predict; NO result columns
    slate_date: pd.Timestamp
    as_of: pd.Timestamp
    season: int
    seasons: list[int]             # {season-1, season}
    universe_prior: pd.DataFrame   # completed D-I non-truncated games, date < D
    season_start: pd.Timestamp
    ratings_dir: str = "data/processed/ratings"
    notes: dict = field(default_factory=dict)

    @property
    def prior_ids(self) -> set:
        return set(self.universe_prior["game_id"].tolist())

    def team_games(self) -> pd.DataFrame:
        """(game_id, team_id, opp_id, is_home) both sides of every slate game."""
        s = self.slate
        h = pd.DataFrame({"game_id": s["game_id"].to_numpy(), "team_id": s["home_team_id"].to_numpy(),
                          "opp_id": s["away_team_id"].to_numpy(), "is_home": True})
        a = pd.DataFrame({"game_id": s["game_id"].to_numpy(), "team_id": s["away_team_id"].to_numpy(),
                          "opp_id": s["home_team_id"].to_numpy(), "is_home": False})
        return pd.concat([h, a], ignore_index=True)


def build_ctx(slate: pd.DataFrame, as_of, season: int, universe: pd.DataFrame,
              season_start=None, ratings_dir: str = "data/processed/ratings",
              prior_season: bool = True, strict_finish: bool = True) -> Ctx:
    """`slate`: SLATE_COLS only. `universe`: the games universe (it may contain
    results; only rows dated strictly before the slate date are ever kept, and
    those are COMPLETED games, which is what the backtest also reads)."""
    for c in SLATE_COLS:
        if c not in slate.columns:
            raise KeyError(f"slate is missing {c!r}")
    leaked = [c for c in slate.columns if c in ("home_score", "away_score", "final_margin")]
    if leaked:
        raise G.LeakGuardError(f"slate carries result columns {leaked}; drop them before the live path")
    slate = slate[SLATE_COLS].copy().reset_index(drop=True)
    slate["game_date"] = pd.to_datetime(slate["game_date"])
    dates = slate["game_date"].unique()
    if len(dates) != 1:
        raise ValueError(f"the live path takes ONE slate date per call (stub rows would enter a later "
                         f"date's league totals); got {sorted(str(d)[:10] for d in dates)}")
    D = pd.Timestamp(dates[0])
    T = G.assert_cutoff_before_slate(as_of, slate)
    seasons = [season - 1, season] if prior_season else [season]
    u = universe.copy()
    u["game_date"] = pd.to_datetime(u["game_date"])
    u = u[u["is_d1_game"] & ~u["pbp_truncated"] & u["season"].isin(seasons) & (u["game_date"] < D)]
    G.assert_sources_before(u, D, "universe_prior")
    n_unfinished = G.assert_sources_finished(u, T, strict=strict_finish)
    if season_start is None:
        season_start = slate["game_date"].min()
    return Ctx(slate=slate, slate_date=D, as_of=T, season=int(season), seasons=seasons,
               universe_prior=u.reset_index(drop=True), season_start=pd.Timestamp(season_start),
               ratings_dir=ratings_dir, notes={"n_source_unfinished_at_cutoff": n_unfinished})


def stub_universe(ctx: Ctx) -> pd.DataFrame:
    """Universe rows for the slate games (id, date, season only)."""
    return pd.DataFrame({"game_id": ctx.slate["game_id"].to_numpy(),
                         "game_date": ctx.slate["game_date"].to_numpy(),
                         "season": ctx.season, "is_d1_game": True, "pbp_truncated": False})


def universe_with_stubs(ctx: Ctx, complete_only: bool = False) -> pd.DataFrame:
    u = ctx.universe_prior
    if complete_only:
        u = u[u["pbp_complete"]]
    cols = ["game_id", "game_date", "season"]
    return pd.concat([u[cols], stub_universe(ctx)[cols]], ignore_index=True)


# ---------------------------------------------------------------------------
# ratings (one as-of row per team, forced to the slate date)
# ---------------------------------------------------------------------------
def ratings_for_slate(ctx: Ctx) -> pd.DataFrame:
    """The ridge ratings rows with as_of_date == D ONLY. `as_of_date == D` means
    'fitted on games strictly before D' (own_ratings.join_as_of docstring). A
    slate team with no row raises: nothing is imputed."""
    p = Path(ctx.ratings_dir) / f"own_ratings_{ctx.season}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing own ratings: {p} (no own_ratings for season {ctx.season}: "
                                "there is no daily entry point yet, docs/ops/readiness gap 3)")
    r = pd.read_parquet(p)
    r["as_of_date"] = pd.to_datetime(r["as_of_date"])
    r = r[r["as_of_date"] == ctx.slate_date]
    have = set(r["team_id"].tolist())
    need = set(ctx.slate["home_team_id"]) | set(ctx.slate["away_team_id"])
    miss = sorted(need - have)
    if miss:
        raise KeyError(f"own_ratings_{ctx.season} has no as_of_date == {ctx.slate_date.date()} row for "
                       f"{len(miss)} slate team(s): {miss[:10]}")
    return r


# ---------------------------------------------------------------------------
# family 1: possession-outcome team form (16 team columns)
# ---------------------------------------------------------------------------
def po_team_block(ctx: Ctx, poss_dir=PO.DEFAULT_POSS_DIR) -> pd.DataFrame:
    """One row per (game_id, team_id): the sixteen PO base columns for the team
    on OFFENCE against the opponent (opp_def_* is the opponent's def form)."""
    tg = ctx.team_games()
    stub = pd.DataFrame({
        "season": ctx.season, "game_id": tg["game_id"].to_numpy(),
        "offense_team_id": tg["team_id"].to_numpy(), "defense_team_id": tg["opp_id"].to_numpy(),
        "poss_index": 0, "terminal_event": "stub", "fga_rim": 0, "fga_jump2": 0, "fga_3": 0,
        "fta": 0, "points": 0})
    frames = {}
    cols = ["season", "game_id", "offense_team_id", "defense_team_id", "poss_index",
            "terminal_event", "fga_rim", "fga_jump2", "fga_3", "fta", "points"]
    for s in ctx.seasons:
        p = Path(poss_dir) / f"possessions_{s}.parquet"
        if p.exists():
            f = pd.read_parquet(p, columns=cols)
            f = f[f["game_id"].isin(ctx.prior_ids)]
        else:
            f = stub.iloc[0:0].copy()                  # typed empty frame (no history)
        if s == ctx.season:
            f = pd.concat([f, stub], ignore_index=True)
        frames[s] = f
    uni = universe_with_stubs(ctx)
    form = PO.build_team_form(frames, uni, style_source="all_chances")
    form = form[form["game_id"].isin(ctx.slate["game_id"])].drop_duplicates(["game_id", "team_id"])
    rate = list(PO.RATE_DEFS)
    off = form.set_index(["game_id", "team_id"])
    out = tg.copy()
    idx = pd.MultiIndex.from_frame(out[["game_id", "team_id"]])
    oppidx = pd.MultiIndex.from_frame(out[["game_id", "opp_id"]])
    for r in rate:
        out[f"off_{r}_c"] = off[f"off_{r}_c"].reindex(idx).to_numpy()
        out[f"opp_def_{r}_c"] = off[f"def_{r}_c"].reindex(oppidx).to_numpy()
    out["n_prior_off"] = off["n_prior_off"].reindex(idx).to_numpy()
    return out


def po_team_block_r2(ctx: Ctx, poss_dir=None) -> pd.DataFrame:
    """The ROUND-2 event team block (what the served `round2_s1` event adapter reads):
    possessions v2 CHANCE tables, `first_chance` style source, pbp_complete universe
    (`build_engine_event_round2.py`, `round2/design.parquet`). Same shape as `po_team_block`."""
    from cbb_sim.pbp.possessions import possessions_dir
    d = possessions_dir("v2", poss_dir)
    tg = ctx.team_games()
    stub = pd.DataFrame({
        "season": ctx.season, "game_id": tg["game_id"].to_numpy(),
        "offense_team_id": tg["team_id"].to_numpy(), "defense_team_id": tg["opp_id"].to_numpy(),
        "chance_number": 1, "terminal_event": "stub", "fga_rim": 0, "fga_jump2": 0, "fga_3": 0,
        "fta": 0, "points": 0})
    cols = ["season", "game_id", "offense_team_id", "defense_team_id", "chance_number",
            "terminal_event", "fga_rim", "fga_jump2", "fga_3", "fta", "points"]
    ids = set(ctx.universe_prior.loc[ctx.universe_prior["pbp_complete"], "game_id"].tolist())
    frames = {}
    for s_ in ctx.seasons:
        p = Path(d) / f"chances_{s_}.parquet"
        f = pd.read_parquet(p, columns=cols) if p.exists() else stub.iloc[0:0].copy()
        f = f[f["game_id"].isin(ids)]
        if s_ == ctx.season:
            f = pd.concat([f, stub], ignore_index=True)
        frames[s_] = f
    form = PO.build_team_form(frames, universe_with_stubs(ctx, complete_only=True),
                              style_source="first_chance")
    form = form[form["game_id"].isin(ctx.slate["game_id"])].drop_duplicates(["game_id", "team_id"])
    off = form.set_index(["game_id", "team_id"])
    out = tg.copy()
    idx = pd.MultiIndex.from_frame(out[["game_id", "team_id"]])
    oppidx = pd.MultiIndex.from_frame(out[["game_id", "opp_id"]])
    for r in PO.RATE_DEFS:
        out[f"off_{r}_c"] = off[f"off_{r}_c"].reindex(idx).to_numpy()
        out[f"opp_def_{r}_c"] = off[f"def_{r}_c"].reindex(oppidx).to_numpy()
    return out


def rating_site_block(ctx: Ctx) -> pd.DataFrame:
    """rating, site and calendar columns per (game_id, team_id) offence view."""
    tg = ctx.team_games()
    r = ratings_for_slate(ctx)
    s = ctx.slate.set_index("game_id")
    g = tg.copy()
    g["game_date"] = ctx.slate_date
    g["season"] = ctx.season
    g = orat.join_as_of(g, r, "team_id", suffix="__o",
                        cols=("off_c", "def_c", "tempo_rel", "league_tempo_mean"))
    g = orat.join_as_of(g, r, "opp_id", suffix="__d", cols=("off_c", "def_c", "tempo_rel"))
    out = pd.DataFrame({
        "game_id": g["game_id"], "team_id": g["team_id"], "opp_id": g["opp_id"],
        "off_rating_off_c": g["off_c__o"].astype("float32").fillna(0.0),
        "off_rating_def_c": g["def_c__o"].astype("float32").fillna(0.0),
        "def_rating_off_c": g["off_c__d"].astype("float32").fillna(0.0),
        "def_rating_def_c": g["def_c__d"].astype("float32").fillna(0.0),
    })
    neutral = s.loc[g["game_id"], "neutral"].to_numpy().astype(bool)
    is_home = g["is_home"].to_numpy()
    out["site_home"] = ((~neutral) & is_home).astype("float32")
    out["site_away"] = ((~neutral) & (~is_home)).astype("float32")
    out["season_idx"] = np.float32(ctx.season - 2022)
    out["days_since_start"] = np.float32((ctx.slate_date - ctx.season_start).days)
    # clock tempo prior (Decision 7). Product of the two teams' as-of tempo_rel
    # and the as-of league mean, exactly as clock.build_design; NaN -> its own
    # declared fallbacks (1.0 / the as-of league mean of the SAME date).
    tr_o = g["tempo_rel__o"].to_numpy(dtype="float64")
    tr_d = g["tempo_rel__d"].to_numpy(dtype="float64")
    lgm = g["league_tempo_mean__o"].to_numpy(dtype="float64")
    out["off_tempo_rel"] = np.where(np.isfinite(tr_o), tr_o, 1.0).astype("float32")
    out["def_tempo_rel"] = np.where(np.isfinite(tr_d), tr_d, 1.0).astype("float32")
    prior = (tr_o * tr_d * lgm).astype("float32")
    lg_mean = float(np.nanmedian(lgm)) if np.isfinite(lgm).any() else np.nan
    out["tempo_prior_game"] = np.where(np.isfinite(prior), prior, np.float32(lg_mean)).astype("float32")
    return out


# ---------------------------------------------------------------------------
# family 2: rebound team form and per-player rates
# ---------------------------------------------------------------------------
def _template(df: pd.DataFrame) -> dict:
    """A real row used to give stub rows every column with a sane dtype. Taken from
    the UNCUT table (stashed in attrs by `cut`) so an empty cut (season 2027 day 1)
    still has a template."""
    return df.attrs.get("tpl") or df.iloc[-1].to_dict()


def cut(full: pd.DataFrame, mask) -> pd.DataFrame:
    tpl = full.iloc[-1].to_dict()
    out = full[mask].reset_index(drop=True)
    out.attrs["tpl"] = tpl
    return out


def rebound_events_cut(ctx: Ctx, path: str) -> pd.DataFrame:
    full = pd.read_parquet(path)
    ev = cut(full, full["season"].isin(ctx.seasons) & full["game_id"].isin(ctx.prior_ids))
    G.assert_sources_before(ev, ctx.slate_date, "rebound events")
    return ev


def rebound_team_block(ctx: Ctx, ev: pd.DataFrame) -> pd.DataFrame:
    """(game_id, team_id) -> off_oreb_c, opp_def_dreb_c, with the design's own
    fillna(0.0)."""
    tg = ctx.team_games()
    tpl = _template(ev)
    rows = []
    for r in tg.itertuples(index=False):
        d = dict(tpl)
        d.update(game_id=int(r.game_id), season=ctx.season, game_date=ctx.slate_date,
                 off_team_id=int(r.team_id), def_team_id=int(r.opp_id), outcome="OREB",
                 chance_index=0)
        rows.append(d)
    stub = pd.DataFrame(rows, columns=list(ev.columns)).astype(ev.dtypes.to_dict())
    allev = pd.concat([ev, stub], ignore_index=True)
    form = RB.team_rebound_form(allev, universe_with_stubs(ctx), first_chance_only=True)
    form = form[form["game_id"].isin(ctx.slate["game_id"])].drop_duplicates(["game_id", "team_id"])
    f = form.set_index(["game_id", "team_id"])
    out = tg.copy()
    idx = pd.MultiIndex.from_frame(out[["game_id", "team_id"]])
    oppidx = pd.MultiIndex.from_frame(out[["game_id", "opp_id"]])
    out["off_oreb_c"] = f["off_oreb_c"].reindex(idx).to_numpy().astype("float32")
    out["opp_def_dreb_c"] = f["def_dreb_c"].reindex(oppidx).to_numpy().astype("float32")
    out[["off_oreb_c", "opp_def_dreb_c"]] = out[["off_oreb_c", "opp_def_dreb_c"]].fillna(0.0)
    return out


def rebound_player_rates(ctx: Ctx, ev: pd.DataFrame, cand: dict, prior_opps: int = 50):
    """`cand`: {(game_id, team_id): [pid, ...]} candidate players. Returns
    (frame keyed (game_id, pid) with oreb_rate, dreb_rate, medians dict). Stub
    events put every candidate on the floor so each has a row for the stub game."""
    tpl = _template(ev)
    rows = []
    for gm in ctx.slate.itertuples(index=False):
        h = [p for p in cand.get((int(gm.game_id), int(gm.home_team_id)), []) if p > 0]
        a = [p for p in cand.get((int(gm.game_id), int(gm.away_team_id)), []) if p > 0]
        n = max((len(h) + 4) // 5, (len(a) + 4) // 5)
        for k in range(n):
            d = dict(tpl)
            d.update(game_id=int(gm.game_id), season=ctx.season, game_date=ctx.slate_date,
                     off_team_id=int(gm.home_team_id), def_team_id=int(gm.away_team_id),
                     offense_is_home=True, outcome="OREB", chance_index=0, rebounder_id=np.nan)
            for i in range(5):
                hp = h[5 * k + i] if 5 * k + i < len(h) else -1
                ap = a[5 * k + i] if 5 * k + i < len(a) else -1
                d[f"home_on_{i + 1}"] = float(hp)
                d[f"away_on_{i + 1}"] = float(ap)
            rows.append(d)
    stub = pd.DataFrame(rows, columns=list(ev.columns)).astype(ev.dtypes.to_dict())
    allev = pd.concat([ev, stub], ignore_index=True)
    prr = RB.player_rebound_rates(allev, prior_opps=prior_opps)
    prr["game_date"] = pd.to_datetime(prr["game_date"])
    past = prr[prr["game_date"] < ctx.slate_date]
    med = {c: float(np.nanmedian(past[c].to_numpy())) if len(past) else float("nan")
           for c in ("oreb_rate", "dreb_rate")}
    cur = prr[prr["game_id"].isin(ctx.slate["game_id"])].rename(columns={"player_id": "pid"})
    return cur[["game_id", "pid", "oreb_rate", "dreb_rate"]].drop_duplicates(["game_id", "pid"]), med


# ---------------------------------------------------------------------------
# family 3: rotation priors (roster, minute shares, start order, fouls/min)
# ---------------------------------------------------------------------------
def rotation_priors(ctx: Ctx, fit, min_prior_games: int = 1):
    """`{(game_id, team_id): TeamPrior}` for every slate team-game that has any
    earlier appearance. Same functions as the backtest builder, on the season's
    on-floor table cut to `game_date < D` plus a stub `pg` row (pid=-1, zero
    minutes) per slate team-game so the team-game exists in the ordering."""
    from cbb_sim.models import rotation as ROT
    p = Path(ROT.DEFAULT_POSS_DIR) / f"possessions_{ctx.season}.parquet"
    if p.exists():
        tp = ROT.load_team_possessions(ctx.season)
        tp = tp[pd.to_datetime(tp["game_date"]) < ctx.slate_date]
        G.assert_sources_before(tp, ctx.slate_date, "rotation on-floor table")
        pgm = ROT.player_game_minutes(tp)
        try:
            gu = ctx.universe_prior[ctx.universe_prior["season"] == ctx.season][["game_id", "cbbd_game_id"]]
            fouls = ROT.player_game_fouls(ctx.season).merge(gu, on="cbbd_game_id", how="inner")
            fouls = fouls.merge(pgm[["game_id", "team_id", "pid"]].drop_duplicates(),
                                on=["game_id", "pid"], how="inner")
        except Exception:                                                # noqa: BLE001
            fouls = None
    else:
        pgm = pd.DataFrame({"game_id": pd.Series(dtype="int64"), "season": pd.Series(dtype="int64"),
                            "game_date": pd.Series(dtype="datetime64[ns]"), "team_id": pd.Series(dtype="int64"),
                            "pid": pd.Series(dtype="int64"), "duration_s": pd.Series(dtype="float64"),
                            "minutes": pd.Series(dtype="float64"), "is_starter": pd.Series(dtype="bool"),
                            "minutes_rank": pd.Series(dtype="int64")})
        fouls = None
    tg = ctx.team_games()
    stub = pd.DataFrame({"game_id": tg["game_id"].to_numpy(), "season": ctx.season,
                         "game_date": ctx.slate_date, "team_id": tg["team_id"].to_numpy(),
                         "pid": -1, "duration_s": 0.0, "minutes": 0.0, "is_starter": False,
                         "minutes_rank": 99})
    pg = pd.concat([pgm, stub], ignore_index=True)
    feats = ROT.build_asof_player_features(pg, fouls=fouls)
    return ROT.build_priors(feats, fit, min_prior_games=min_prior_games)


def candidates_from_priors(priors: dict, n_slots: int = 15) -> dict:
    return {k: [int(x) for x in v.pids[: min(v.n, n_slots)] if x > 0] for k, v in priors.items()}


# ---------------------------------------------------------------------------
# family 4: fg_make design (team form + shooter block) via the module's own build_design
# ---------------------------------------------------------------------------
def fg_events_cut(ctx: Ctx, path: str) -> pd.DataFrame:
    full = pd.read_parquet(path)
    ev = cut(full, full["season"].isin(ctx.seasons) & full["game_id"].isin(ctx.prior_ids))
    G.assert_sources_before(ev, ctx.slate_date, "fg events")
    return ev


def fg_design_live(ctx: Ctx, ev: pd.DataFrame, cand: dict):
    """`FG.build_design` on events + stub events. One stub event per (game,
    side, candidate, class). Returns (design rows of the slate games, the
    stub-inclusive events, which the round-4 columns need)."""
    from cbb_sim.models import fg_make as FG
    tpl = _template(ev)
    cls_map = {"FGA_rim": "rim", "FGA_jump2": "jump2", "FGA_3": "three"}
    rows = []
    for gm in ctx.slate.itertuples(index=False):
        for is_home, tid, oid in ((True, gm.home_team_id, gm.away_team_id),
                                  (False, gm.away_team_id, gm.home_team_id)):
            for pid in cand.get((int(gm.game_id), int(tid)), []):
                for sc, ck in cls_map.items():
                    d = dict(tpl)
                    d.update(game_id=int(gm.game_id), cbbd_game_id=int(gm.cbbd_game_id),
                             season=ctx.season, game_date=ctx.slate_date,
                             neutral_site=bool(gm.neutral), period=1, seconds_remaining=1200,
                             score_diff=0, off_team_id=int(tid), def_team_id=int(oid),
                             offense_is_home=is_home, shooter_id=float(pid), shot_class=sc,
                             class_key=ck, made=False, blocked=False, and_one=False,
                             chance_number=1)
                    for k in range(1, 6):
                        d[f"home_on_{k}"] = np.nan
                        d[f"away_on_{k}"] = np.nan
                    rows.append(d)
    if not rows:
        return pd.DataFrame(), ev
    stub = pd.DataFrame(rows, columns=list(ev.columns)).astype(ev.dtypes.to_dict())
    allev = pd.concat([ev, stub], ignore_index=True)
    allev.attrs.update(ev.attrs)
    design = FG.build_design(ctx.seasons, universe=universe_with_stubs(ctx, complete_only=True),
                             version="v2", events=allev, with_ratings=False)
    return design[design["game_id"].isin(ctx.slate["game_id"])].reset_index(drop=True), allev
