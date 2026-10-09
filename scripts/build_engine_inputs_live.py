"""
build_engine_inputs_live.py -- engine inputs for a slate of games that have NOT been played.

    .venv/Scripts/python.exe scripts/build_engine_inputs_live.py \
        --slate-date 2024-11-12 --as-of 2024-11-12T12:00:00Z --season 2025 --fold F2 \
        --schedule-source universe --out-dir data/processed/models/engine_live

Design: docs/ops/live_slate_path_2026-09-30.md. ONE feature definition: every as-of
statistic is computed by the module function the backtest designs call (see
`cbb_sim.live.features`), on source tables cut to `game_date < D` plus stub rows for
the slate. Column lists, slot order and array layout are IMPORTED from
`build_engine_inputs.py`, never copied. Output schema == `EngineInputs` (v2 layout:
the fg_make shooter block keyed on shot_shooter_id, round-4 slot columns, usage_v2).

Nothing here reads a result: the slate frame is rejected if it carries score columns,
every source table is cut to `game_date < D`, `as_of < first tip` is asserted, and
`created_at` is stamped on the games table and on every downstream prediction row.

Fold/artifact keying: `--fold` selects the rule constants (`names_{fold}_{season}_v2.json`
template) and, in `run_engine_live.py`, the dated model artifacts. Nothing exists for
season 2027 yet (docs/ops/live_slate_path section 5).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_engine_inputs as B  # noqa: E402  (constants and layout only; main() is not run)
from cbb_sim.engine.inputs import N_SLOTS, EngineInputs  # noqa: E402
from cbb_sim.live import features as LF  # noqa: E402
from cbb_sim.live import guards as G  # noqa: E402
from cbb_sim.live import players as LP  # noqa: E402
from cbb_sim.models import rotation as ROT  # noqa: E402

ENGINE_DIR = ROOT / "data/processed/models/engine"
FG_EVENTS = ROOT / "data/processed/models/fg_make/events_v2_shotshooter.parquet"
RB_EVENTS = ROOT / "data/processed/models/rebound/events_v1.parquet"
FT_ATTEMPTS = ROOT / "data/processed/models/free_throw/attempts_v1_era.parquet"
USAGE_EVENTS_V2 = ROOT / "data/processed/models/usage_v2/events_v2_shotshooter.parquet"
R4_M = ROOT / "data/processed/models/fg_make/round4/m_fitted.json"
UNIVERSE = ROOT / "data/processed/games_universe.parquet"
CROSSWALK = ROOT / "data/processed/player_crosswalk.parquet"


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# schedule sources
# ---------------------------------------------------------------------------
def load_slate_from_universe(slate_date: str, season: int, only_ids=None) -> pd.DataFrame:
    """Parity/dry-run source: the games universe, RESULT COLUMNS DROPPED. Used to
    'pretend the games are unplayed'."""
    u = pd.read_parquet(UNIVERSE)
    u["game_date"] = pd.to_datetime(u["game_date"])
    u = u[(u["season"] == season) & (u["game_date"] == pd.Timestamp(slate_date))
          & u["is_d1_game"] & ~u["pbp_truncated"]]
    if only_ids is not None:
        u = u[u["game_id"].isin(set(only_ids))]
    return pd.DataFrame({
        "game_id": u["game_id"].to_numpy(), "cbbd_game_id": u["cbbd_game_id"].astype("int64").to_numpy(),
        "season": season, "game_date": u["game_date"].to_numpy(), "tipoff_utc": u["tipoff_utc"].to_numpy(),
        "home_team_id": u["home_team_id"].to_numpy(), "away_team_id": u["away_team_id"].to_numpy(),
        "neutral": u["neutral_site"].to_numpy().astype(bool)}).reset_index(drop=True)


def load_slate_from_cbbd(path: str, slate_date: str, crosswalk_path: str) -> pd.DataFrame:
    """Live source: a CBBD `/games` dump (json list or parquet). CBBD team ids are mapped to
    the engine's ESPN team ids through the team crosswalk; games with an unmapped side are
    RETURNED IN `attrs['unmapped']` and never silently dropped."""
    p = Path(path)
    g = pd.read_parquet(p) if p.suffix == ".parquet" else pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
    g["_d"] = pd.to_datetime(g["startDate"], utc=True)
    g["game_date"] = g["_d"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    g = g[g["game_date"] == pd.Timestamp(slate_date)].copy()
    cw = pd.read_parquet(crosswalk_path)
    cb_col = "cbbd_team_id" if "cbbd_team_id" in cw.columns else "cbbd_id"
    es_col = "espn_team_id" if "espn_team_id" in cw.columns else "espn_id"
    m = dict(zip(cw[cb_col].astype("int64"), cw[es_col].astype("int64")))
    hm = g["homeTeamId"].astype("int64").map(m)
    am = g["awayTeamId"].astype("int64").map(m)
    ok = hm.notna() & am.notna()
    out = pd.DataFrame({
        "game_id": g.loc[ok, "sourceId"].astype("int64") if "sourceId" in g else g.loc[ok, "id"].astype("int64"),
        "cbbd_game_id": g.loc[ok, "id"].astype("int64"), "season": g.loc[ok, "season"].astype(int),
        "game_date": g.loc[ok, "game_date"], "tipoff_utc": g.loc[ok, "_d"],
        "home_team_id": hm[ok].astype("int64"), "away_team_id": am[ok].astype("int64"),
        "neutral": g.loc[ok, "neutralSite"].astype(bool)}).reset_index(drop=True)
    out.attrs["unmapped"] = g.loc[~ok, ["id", "homeTeam", "awayTeam"]].to_dict("records")
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def apply_availability(priors: dict, availability, created_at, diag: dict) -> dict:
    """Drop listed-out players from each team-game's rotation prior. Returns a NEW dict (input untouched); nothing is hand-redistributed:
    the remaining players keep their own as-of share / fouls-per-minute / availability, `build_live` renormalises shares, and start order is
    re-ranked 1..k by the existing start_order (so the next-ranked players start). A team-game left with no real player falls back to the
    league-mean profile (the same path as a team with no prior). Pids that match no slate candidate list are reported in diag, never dropped
    silently. Leak guard: rows stamped after `created_at` raise."""
    import dataclasses
    from cbb_sim.live import guards as _G
    df = pd.read_csv(availability) if isinstance(availability, (str, Path)) else pd.DataFrame(availability)
    if "cbbd_player_id" not in df.columns:
        raise ValueError("availability needs a cbbd_player_id column")
    if "created_at" in df.columns and len(df):
        ts = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        late = ts > created_at
        if late.any():
            raise _G.LeakGuardError(f"{int(late.sum())} availability rows stamped after created_at {created_at}")
    ids = pd.to_numeric(df["cbbd_player_id"], errors="coerce")
    out_ids = set(int(x) for x in ids.dropna().astype("int64") if x > 0)
    seen: set = set()
    new: dict = {}
    n_tg = n_drop = n_fb = 0
    for key, pr in priors.items():
        hit = np.isin(pr.pids, list(out_ids)) if out_ids else np.zeros(len(pr.pids), dtype=bool)
        if not hit.any():
            new[key] = pr
            continue
        seen.update(int(x) for x in pr.pids[hit])
        n_tg += 1
        n_drop += int(hit.sum())
        keep = ~hit
        if not (pr.pids[keep] > 0).any():
            n_fb += 1                                   # nobody real left: omit, the fallback profile path takes over
            continue
        k = int(keep.sum())
        order = np.argsort(pr.srank[keep], kind="stable")
        srank = np.empty(k, dtype=pr.srank.dtype)
        srank[order] = np.arange(1, k + 1)
        so = np.argsort(srank, kind="stable").astype(pr.start_order.dtype)
        new[key] = dataclasses.replace(pr, pids=pr.pids[keep], share=pr.share[keep], rank=np.arange(1, k + 1, dtype=pr.rank.dtype),
                                       srank=srank, start_order=so, fpm=pr.fpm[keep], p_avail=pr.p_avail[keep])
    diag["availability"] = {"rows": int(len(df)), "rows_without_cbbd_id": int(ids.isna().sum()), "out_ids": len(out_ids),
                            "team_games_affected": n_tg, "players_dropped": n_drop, "team_games_fallback_after_drop": n_fb,
                            "unknown_pids": sorted(out_ids - seen)}
    return new


def espn_map_uncrosswalked(cw: pd.DataFrame, season: int) -> tuple[dict, str]:
    """{cbbd_player_id: espn_athlete_id} for a season the player crosswalk does not cover yet (2027 serving, 2026-10-09,
    docs/ops/a3_seed_wiring_2026-10-09.md). Only labels which player rows the engine emits (loop.py keeps espn > 0); no simulated
    quantity reads it. Season-S roster file (`espn_player_id`, from the ESPN puller) first, then the latest crosswalk season < S
    (R1 fallback players are S-1 players). Seasons IN the crosswalk never reach this function."""
    out: dict = {}
    src = []
    prev = cw.loc[(cw["season"] < int(season)) & cw["espn_athlete_id"].notna(), "season"]
    if len(prev):
        ps = int(prev.max())
        g = cw[(cw["season"] == ps) & cw["espn_athlete_id"].notna()]
        out.update(zip(g["cbbd_player_id"].astype("int64"), g["espn_athlete_id"].astype("int64")))
        src.append(f"crosswalk season {ps}: {len(g)}")
    rp = ROOT / f"data/raw/cbbd/rosters/roster_{int(season)}.parquet"
    if rp.exists():
        r = pd.read_parquet(rp)
        col = "espn_player_id" if "espn_player_id" in r.columns else "source_id"
        r = r[["cbbd_player_id", col]].dropna()
        r = r[pd.to_numeric(r[col], errors="coerce").notna()]
        out.update(zip(r["cbbd_player_id"].astype("int64"), pd.to_numeric(r[col]).astype("int64")))     # season-S roster wins
        src.append(f"{rp.name} {col}: {len(r)}")
    return out, "; ".join(src) or "none"


def build_live(slate: pd.DataFrame, as_of, season: int, fold: str, created_at=None,
               season_start=None, template_tag: str | None = None, families: str = "all",
               t0: float | None = None, strict_finish: bool = True,
               ratings_dir: str | None = None, seed_fn=None, availability=None) -> tuple[EngineInputs, dict]:
    """`seed_fn` (DEFAULT None = unchanged; lane F, 2026-10-01, docs/tests/early_season_roster_slots_2026-10-01.md): an as-of-safe
    candidate-seeding hook, called as seed_fn(ctx, games, tg, roster_cbbd, roster_valid, S, diag) after the rotation roster arrays are built;
    it fills anonymous tail slots in place and returns the candidate dict. Used only by scripts/build_engine_inputs_seeded_v1.py.

    `availability` (DEFAULT None = bit-identical to before; 2026-10-05, docs/ops/readiness_gaps_2026-10-05.md): a DataFrame (or csv path) in the
    `player_out_for` format with a `cbbd_player_id` column (the set of players OUT for this slate, supplied by the caller). Each listed pid is removed from
    the matching team-game's rotation prior (see `apply_availability`) before the shares are normalised and the candidate list is formed."""
    t0 = t0 or time.time()
    template_tag = template_tag or f"{fold}_{season}_v2"
    names_t = json.loads((ENGINE_DIR / f"names_{template_tag}.json").read_text(encoding="utf-8"))
    team_names = {k: int(v) for k, v in names_t["team_names"].items()}
    slot_names = {k: int(v) for k, v in names_t["slot_names"].items()}
    rules = dict(names_t["rules"])
    u = pd.read_parquet(UNIVERSE)
    ctx = LF.build_ctx(slate, as_of, season, u, season_start=season_start, strict_finish=strict_finish,
                       **({"ratings_dir": str(ratings_dir)} if ratings_dir else {}))   # None = the stored batch ratings (unchanged)
    from cbb_sim.data.seal import assert_not_sealed_serving
    assert_not_sealed_serving(ctx.seasons, season, context="live inputs (prior-season carry reads season-1 tables)")
    created_at = pd.Timestamp(created_at) if created_at is not None else pd.Timestamp.now("UTC")
    created_at = created_at.tz_localize("UTC") if created_at.tzinfo is None else created_at.tz_convert("UTC")
    if not created_at < pd.to_datetime(ctx.slate["tipoff_utc"], utc=True).min():
        raise G.LeakGuardError(f"created_at {created_at} is not before the first tip of the slate")
    games = ctx.slate.copy()
    Gn, S = len(games), N_SLOTS
    gpos = {int(g): i for i, g in enumerate(games["game_id"])}
    diag: dict = {"n_games": Gn, "slate_date": str(ctx.slate_date.date()), "as_of": str(ctx.as_of),
                  "created_at": str(created_at), "n_universe_prior": len(ctx.universe_prior),
                  "n_source_unfinished_at_cutoff": ctx.notes.get("n_source_unfinished_at_cutoff", 0)}
    log(f"ctx: {Gn} games on {ctx.slate_date.date()}, {len(ctx.universe_prior)} prior games in "
        f"seasons {ctx.seasons}", t0)

    # ---- team block -------------------------------------------------------
    team_static = np.zeros((Gn, 2, len(B.TEAM_COLS)), dtype=np.float32)
    tg = ctx.team_games()
    row_of = np.array([gpos[int(g)] for g in tg["game_id"]])
    side_of = np.where(tg["is_home"].to_numpy(), 0, 1)

    def put_team(frame: pd.DataFrame, cols) -> None:
        f = frame.set_index(["game_id", "team_id"])
        idx = pd.MultiIndex.from_frame(tg[["game_id", "team_id"]])
        for c in cols:
            v = f[c].reindex(idx).to_numpy(dtype="float64")
            ok = np.isfinite(v)
            team_static[row_of[ok], side_of[ok], team_names[c]] = v[ok].astype(np.float32)

    po = LF.po_team_block(ctx)
    po_cols = [f"{p}_{r}_c" for p in ("off", "opp_def") for r in ("3pa", "rim", "tov", "ftr")]
    put_team(po, po_cols)
    rs = LF.rating_site_block(ctx)
    put_team(rs, ["off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c",
                  "site_home", "site_away", "season_idx", "days_since_start",
                  "off_tempo_rel", "def_tempo_rel", "tempo_prior_game"])
    # round-2 event team block (16 cols): the block the served event adapter reads
    r2cols = list(B.TEAM_COLS[:16])
    po2 = LF.po_team_block_r2(ctx).merge(
        rs.drop(columns=["opp_id"]), on=["game_id", "team_id"])
    event_block = np.zeros((Gn, 2, 16), dtype=np.float32)
    f2 = po2.set_index(["game_id", "team_id"])
    idx2 = pd.MultiIndex.from_frame(tg[["game_id", "team_id"]])
    for j, c in enumerate(r2cols):
        v = f2[c].reindex(idx2).to_numpy(dtype="float64")
        v = np.where(np.isfinite(v), v, 0.0)          # the backtest block's own no-history state
        event_block[row_of, side_of, j] = v.astype(np.float32)
    rb_ev = LF.rebound_events_cut(ctx, str(RB_EVENTS))
    rbt = LF.rebound_team_block(ctx, rb_ev)
    put_team(rbt, ["off_oreb_c", "opp_def_dreb_c"])
    log("team blocks: possession outcome, ratings/site/tempo, rebound", t0)

    # ---- rotation priors (roster) ----------------------------------------
    fit = ROT.RotationFit.from_json(B.ROT_FIT)
    priors = LF.rotation_priors(ctx, fit, min_prior_games=1)
    if availability is not None:
        priors = apply_availability(priors, availability, created_at, diag)
    n_have = sum((int(g), int(t)) in priors for g in games["game_id"] for t in tg.loc[tg["game_id"] == g, "team_id"])
    log(f"rotation priors: {n_have} of {2 * Gn} slate team-games have a prior", t0)
    roster_cbbd = np.zeros((Gn, 2, S), dtype=np.int64)
    roster_valid = np.zeros((Gn, 2, S), dtype=bool)
    rot_share = np.zeros((Gn, 2, S), dtype=np.float32)
    rot_srank = np.zeros((Gn, 2, S), dtype=np.int16)
    rot_start = np.zeros((Gn, 2, S), dtype=np.int16)
    rot_fpm = np.full((Gn, 2, S), float(fit.fpm_league), dtype=np.float32)
    rot_pavail = np.zeros((Gn, 2, S), dtype=np.float32)
    role = np.asarray(fit.role_prior, dtype=np.float64)[:S]
    if len(role) < S:
        role = np.concatenate([role, np.full(S - len(role), role[-1] * fit.tail_ratio)])
    fb_share = (role / role.sum()).astype(np.float32)
    p_play = np.asarray(fit.p_play, dtype=np.float64) if fit.p_play else np.full(S, 0.85)
    fb_pavail = p_play[np.clip(np.arange(1, S + 1), 1, len(p_play)) - 1].astype(np.float32)
    n_fallback = 0
    for side, tcol in ((0, "home_team_id"), (1, "away_team_id")):
        for i in range(Gn):
            pr = priors.get((int(games["game_id"].iloc[i]), int(games[tcol].iloc[i])))
            if pr is None:
                n_fallback += 1
                roster_cbbd[i, side] = -np.arange(1, S + 1)
                roster_valid[i, side] = True
                rot_share[i, side] = fb_share
                rot_srank[i, side] = np.arange(1, S + 1)
                rot_start[i, side] = np.arange(S)
                rot_pavail[i, side] = fb_pavail
                continue
            k = min(pr.n, S)
            roster_cbbd[i, side, :k] = pr.pids[:k]
            roster_valid[i, side, :k] = True
            sh = pr.share[:k]
            rot_share[i, side, :k] = (sh / sh.sum()).astype(np.float32)
            rot_srank[i, side, :k] = pr.srank[:k]
            rot_fpm[i, side, :k] = pr.fpm[:k]
            rot_pavail[i, side, :k] = pr.p_avail[:k]
            order = np.argsort(pr.srank[:k], kind="stable")
            pos = np.empty(k, dtype=np.int16)
            pos[order] = np.arange(k, dtype=np.int16)
            rot_start[i, side, :k] = pos
            if k < S:
                rot_srank[i, side, k:] = np.arange(k + 1, S + 1)
                rot_start[i, side, k:] = np.arange(k, S)
    diag["rotation_fallback_team_games"] = n_fallback
    cand = LF.candidates_from_priors(priors, S)
    if seed_fn is not None:
        cand = seed_fn(ctx, games, tg, roster_cbbd, roster_valid, S, diag)
    diag["candidates_per_team_game_mean"] = float(np.mean([len(v) for v in cand.values()])) if cand else 0.0

    # ---- fg_make design (team form + shooter block + round 4) ------------
    fg_ev = LF.fg_events_cut(ctx, str(FG_EVENTS))
    cand_fg = {}
    for gm in ctx.slate.itertuples(index=False):
        for tid in (gm.home_team_id, gm.away_team_id):
            cand_fg[(int(gm.game_id), int(tid))] = cand.get((int(gm.game_id), int(tid)), []) + [-1]
    design, fg_ev_all = LF.fg_design_live(ctx, fg_ev, cand_fg)
    log(f"fg_make live design: {len(design)} stub rows", t0)
    for cls, key in B.SHOT_KEY.items():
        sl = design[design["shot_class"] == cls].drop_duplicates(["game_id", "off_team_id"])
        t = sl.rename(columns={"off_team_id": "team_id"})
        f = t.set_index(["game_id", "team_id"])
        idx = pd.MultiIndex.from_frame(tg[["game_id", "team_id"]])
        for src, dst in (("off_make_c", f"off_make_c__{key}"), ("def_allow_c", f"def_allow_c__{key}")):
            v = f[src].reindex(idx).to_numpy(dtype="float64")
            ok = np.isfinite(v)
            team_static[row_of[ok], side_of[ok], team_names[dst]] = v[ok].astype(np.float32)

    # ---- slot block ------------------------------------------------------
    slot_static = np.zeros((Gn, 2, S, len(slot_names)), dtype=np.float32)
    flat = pd.DataFrame({
        "row": np.repeat(np.arange(Gn), 2 * S), "side": np.tile(np.repeat([0, 1], S), Gn),
        "slot": np.tile(np.arange(S), 2 * Gn), "pid": roster_cbbd.reshape(-1),
        "game_id": np.repeat(games["game_id"].to_numpy(), 2 * S)})
    real = flat[flat["pid"] > 0].copy()

    def joined(frame: pd.DataFrame, cols) -> pd.DataFrame:
        f = frame[["game_id", "pid", *cols]].drop_duplicates(["game_id", "pid"])
        f["pid"] = f["pid"].astype("int64")
        return real.merge(f, on=["game_id", "pid"], how="left")

    def put_slot(col: str, got: pd.DataFrame, value_col: str) -> None:
        j = slot_names[col]
        v = got[value_col].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        slot_static[got["row"].to_numpy()[ok], got["side"].to_numpy()[ok],
                    got["slot"].to_numpy()[ok], j] = v[ok].astype(np.float32)

    dd = design.rename(columns={"shooter_id": "pid"})
    for cls, key in B.SHOT_KEY.items():
        got = joined(dd[dd["shot_class"] == cls], list(B.V2_PER_CLASS))
        for c in B.V2_PER_CLASS:
            put_slot(f"{c}__{key}", got, c)
    # has_prior_season is CLASS-SPECIFIC in the design (prior-season attempts in the row's own
    # class), yet the engine slot column is class-independent and the backtest value is whichever
    # class row came first in the player's OWN game (a game-day dependence, see parity report).
    # Live takes "any prior-season attempts in any class" (max over the player's class rows).
    dd["has_prior_season"] = dd.groupby(["game_id", "pid"])["has_prior_season"].transform("max")
    got = joined(dd, ["has_prior_season", "pos_G", "pos_F", "pos_C", "shooter_games_asof", "shooter_fga_asof"])
    put_slot("has_prior_season_fg", got, "has_prior_season")
    for c in ("pos_G", "pos_F", "pos_C", "shooter_games_asof", "shooter_fga_asof"):
        put_slot(c, got, c)
    log("fg_make shooter block joined", t0)

    ft = LP.ft_design_live(ctx, str(FT_ATTEMPTS), cand)
    if len(ft):
        got = joined(ft.rename(columns={"shooter_id": "pid"}),
                     ["shooter_ft_asof", "shooter_fta_asof", "prior_season_ft", "has_prior_season"])
        put_slot("shooter_ft_asof", got, "shooter_ft_asof")
        put_slot("shooter_fta_asof", got, "shooter_fta_asof")
        put_slot("prior_season_ft", got, "prior_season_ft")
        put_slot("has_prior_season_ft", got, "has_prior_season")
    log("free_throw shooter block joined", t0)

    if families in ("all", "r4"):
        m = json.loads(R4_M.read_text(encoding="utf-8"))
        r4 = LP.r4_slot_live(ctx, design, fg_ev_all, cand, m)
        r4 = r4.rename(columns={"shooter_id": "pid"})
        for c in B.V2_R4_PER_CLASS:
            for cls, key in B.SHOT_KEY.items():
                got = joined(r4[r4["class_key"] == key], [c])
                put_slot(f"{c}__{key}", got, c)
        got = joined(r4, list(B.V2_R4_SHARED))
        for c in B.V2_R4_SHARED:
            put_slot(c, got, c)
        log("round-4 slot columns joined", t0)

    # ---- usage -----------------------------------------------------------
    up = json.loads(B.USAGE_V2_PARAMS.read_text(encoding="utf-8"))["per_class"]
    asof = LP.usage_asof_live(ctx, str(USAGE_EVENTS_V2), cand)
    rates = LP.usage_rates(asof, ctx.slate_date, up, B.USAGE_CLASSES)
    usage_rate = np.zeros((Gn, 2, S, len(B.USAGE_CLASSES)), dtype=np.float32)
    for k, cls in enumerate(B.USAGE_CLASSES):
        d, fallback = rates[cls]
        got = joined(d, ["u_rate"])
        v = got["u_rate"].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        usage_rate[got["row"].to_numpy()[ok], got["side"].to_numpy()[ok],
                   got["slot"].to_numpy()[ok], k] = v[ok].astype(np.float32)
        blank = usage_rate[:, :, :, k] == 0.0
        usage_rate[:, :, :, k][blank] = np.float32(fallback) if np.isfinite(fallback) else 0.0
        rules[f"usage_prior_{cls}"] = {"prior_kind": up[cls]["prior_kind"], "m": float(up[cls]["shrink_m"]),
                                       "no_history_rate": round(fallback, 6) if np.isfinite(fallback) else None}
    log("usage rates joined", t0)

    # ---- per-player rebound rates ---------------------------------------
    reb_rate = np.zeros((Gn, 2, S, 2), dtype=np.float32)
    prr, med = LF.rebound_player_rates(ctx, rb_ev, cand, prior_opps=50)
    got = joined(prr.rename(columns={}), ["oreb_rate", "dreb_rate"])
    for k, c in enumerate(("oreb_rate", "dreb_rate")):
        v = got[c].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        reb_rate[got["row"].to_numpy()[ok], got["side"].to_numpy()[ok],
                 got["slot"].to_numpy()[ok], k] = v[ok].astype(np.float32)
        blank = reb_rate[:, :, :, k] == 0.0
        reb_rate[:, :, :, k][blank] = np.float32(med[c]) if np.isfinite(med[c]) else 0.0
    log("player rebound rates joined", t0)

    # ---- ESPN athlete ids ------------------------------------------------
    roster_espn = np.full((Gn, 2, S), -1, dtype=np.int64)
    try:
        from cbb_sim.data import player_ids as PID
        cw = pd.read_parquet(CROSSWALK)
        mp = PID.cbbd_to_espn_map(cw, season)
        if not len(mp):                                             # season not in the crosswalk (2027 serving): roster-file ids
            mp, diag["espn_map_source"] = espn_map_uncrosswalked(cw, season)
        lut = pd.Series(mp) if not isinstance(mp, pd.Series) else mp
        mapped = pd.Series(roster_cbbd.reshape(-1)).map(lut).to_numpy()
        roster_espn = np.where(pd.isna(mapped), -1, mapped).astype(np.int64).reshape(Gn, 2, S)
    except Exception as exc:                                        # noqa: BLE001
        diag["espn_crosswalk_error"] = str(exc)

    games_out = games.copy()
    games_out["neutral"] = games_out["neutral"].astype("float64")
    games_out["created_at"] = created_at
    meta = {"fold": fold, "season": season, "inputs_version": "v2-live", "live": True,
            "slate_date": str(ctx.slate_date.date()), "as_of": str(ctx.as_of), "created_at": str(created_at),
            "template_names": template_tag, "n_games": Gn, "n_slots": S,
            "rules_source": f"copied from names_{template_tag}.json (fold-trained constants); "
                            "usage no_history_rate recomputed from rows dated < D",
            "rotation_fallback_team_games": n_fallback}
    inp = EngineInputs(
        games=games_out, team_static=team_static, team_names=team_names,
        slot_static=slot_static, slot_names=slot_names, roster_cbbd=roster_cbbd,
        roster_espn=roster_espn, roster_valid=roster_valid, rot_share=rot_share, rot_srank=rot_srank,
        rot_start=rot_start, rot_fpm=rot_fpm, rot_pavail=rot_pavail, usage_rate=usage_rate,
        usage_classes=B.USAGE_CLASSES, reb_rate=reb_rate, rules=rules, meta=meta)
    inp.event_block = event_block          # (G, 2, 16) round-2 event team block
    return inp, diag


BL_TEAM16 = B.TEAM_COLS[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate-date", required=True)
    ap.add_argument("--as-of", required=True, help="cutoff timestamp (ISO, UTC); must precede the first tip")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--fold", default="F2")
    ap.add_argument("--schedule-source", choices=("universe", "cbbd"), default="universe")
    ap.add_argument("--schedule-path", default=None)
    ap.add_argument("--crosswalk", default="data/reference/team_crosswalk.parquet")
    ap.add_argument("--season-start", default=None)
    ap.add_argument("--out-dir", default="data/processed/models/engine_live")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--created-at", default=None)
    ap.add_argument("--availability", default=None, help="default off: player_out csv (cbbd_player_id column) of players OUT; see build_live")
    args = ap.parse_args()
    t0 = time.time()
    if args.schedule_source == "universe":
        slate = load_slate_from_universe(args.slate_date, args.season)
    else:
        slate = load_slate_from_cbbd(args.schedule_path, args.slate_date, args.crosswalk)
        print("unmapped games:", slate.attrs.get("unmapped"))
    inp, diag = build_live(slate, args.as_of, args.season, args.fold, created_at=args.created_at,
                           season_start=args.season_start, t0=t0, availability=args.availability)
    tag = args.tag or f"LIVE_{args.fold}_{args.season}_{args.slate_date}"
    inp.save(args.out_dir, tag)
    np.savez_compressed(Path(args.out_dir) / f"event_block_{tag}.npz", team_block=inp.event_block,
                        cols=np.array(list(BL_TEAM16)))
    (Path(args.out_dir) / f"build_diag_{tag}.json").write_text(json.dumps(diag, indent=2, default=str),
                                                               encoding="utf-8")
    log(f"wrote {args.out_dir}/(games|arrays|names)_{tag}.*", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
