"""
build_engine_inputs.py -- the possession engine's one-time, per-season prep.

    .venv/Scripts/python.exe scripts/build_engine_inputs.py --fold F2 --season 2025

CLAUDE.md: "Sim loop uses lookup tables and vectorized NumPy, never live model
calls." This script is where every quantity that is CONSTANT within a simulated
game is computed once and written to
`data/processed/models/engine/{games,arrays,names}_{fold}_{season}.*`, so the
loop only indexes.

What it does, in order:

 1. UNIVERSE. The games to simulate are exactly `eval.reference.load_actual_games`'s
    set (D-I, non-truncated, completed), so the engine and the Control are
    graded on the same rows.

 2. TEAM-LEVEL AS-OF FORM. Read out of the CACHED designs the sub-model
    trainers already wrote, deduplicated to one row per (game, offence team):
      * possession_outcome round-1 `design.parquet` -> the 14 base features,
        `season_idx`, `days_since_start`;
      * clock round-1 `design.parquet` -> `off_tempo_rel`, `def_tempo_rel`,
        `tempo_prior_game` (Decision 7: the L2 winner enters as a PRIOR
        FEATURE, never as a possession sampler);
      * rebound / fg_make designs, rebuilt here from their cached event tables.
    Every one of these columns is a strictly-before expanding statistic, so
    reading it for the game it belongs to is pregame, not a leak.

 3. PLAYER-LEVEL AS-OF FORM, per roster slot, from the same designs plus
    `usage/asof_v2.parquet` and `rebound.player_rebound_rates`. A player who
    took no shot of a given class in the game he is being prepared for has no
    design row for it; his as-of value is then taken from his most recent
    EARLIER row by `merge_asof(direction="backward")` -- at most one of his own
    games stale, never forward-looking -- and falls back to the model's own
    no-prior state (a centred rate of exactly 0.0 = the league mean) when he
    has no earlier row at all.

 4. ROTATION PRIORS. `rotation.build_priors` over `build_asof_player_features`,
    flattened to fixed-width (game, side, slot) arrays.

 5. ADAPTER FITS. The rebound and free-throw bake-offs BOTH selected a winner
    and NEITHER trainer persists a model (`train_rebound_v1.py`'s docstring
    promises `winner.joblib` and never writes it). The engine therefore refits
    each winner's own spec, from its own module, on its own fold-2 training
    seasons, and records the fit here rather than inventing a substitute.

 6. RULE CONSTANTS DERIVED FROM DATA, never typed in: the dead-ball share by
    miss type, the three-attempt share of shooting-foul trips, the and-one rate
    per made shot class, and the non-shooting-foul accrual rate that the L3
    class vocabulary cannot express (see `docs/models/engine/model.md` section
    5.3 -- it is a measured gap, flagged `provisional_foul_accrual`).

Nothing under `src/cbb_sim/models/` or `docs/models/<submodel>/` is written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.engine.inputs import N_SLOTS, EngineInputs  # noqa: E402
from cbb_sim.eval import reference as REF  # noqa: E402
from cbb_sim.models import event_stream as ES  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import free_throw as FT  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402
from cbb_sim.models import rotation as ROT  # noqa: E402

OUT_DIR = Path("data/processed/models/engine")
PO_DESIGN = Path("data/processed/models/possession_outcome/design.parquet")
CLOCK_DESIGN = Path("data/processed/models/clock/design.parquet")
USAGE_ASOF = Path("data/processed/models/usage/asof_v2.parquet")
USAGE_EVENTS = Path("data/processed/models/usage/events_v2.parquet")
USAGE_PARAMS = Path("data/processed/models/usage/usage_params_v1.json")
ROT_FIT = Path("data/processed/models/rotation/rotation_fit.json")
CROSSWALK = Path("data/processed/player_crosswalk.parquet")

FOLD_TRAIN = {"F1": [2022, 2023], "F2": [2022, 2023, 2024]}
FOLD_TEST = {"F1": [2024], "F2": [2025]}

USAGE_CLASSES = ("FGA_rim", "FGA_jump2", "FGA_3", "TOV", "FT_trip")
SHOT_KEY = {"FGA_rim": "rim", "FGA_jump2": "jump2", "FGA_3": "three"}

#: Team-level static columns the engine carries, in a fixed order.
TEAM_COLS: tuple[str, ...] = (
    # possession_outcome base
    "off_3pa_c", "off_rim_c", "off_tov_c", "off_ftr_c",
    "opp_def_3pa_c", "opp_def_rim_c", "opp_def_tov_c", "opp_def_ftr_c",
    "off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c",
    "site_home", "site_away", "season_idx", "days_since_start",
    # clock (Decision 7: the pace winner is a prior feature)
    "off_tempo_rel", "def_tempo_rel", "tempo_prior_game",
    # rebound
    "off_oreb_c", "opp_def_dreb_c",
    # fg_make, per shot class
    "off_make_c__rim", "def_allow_c__rim",
    "off_make_c__jump2", "def_allow_c__jump2",
    "off_make_c__three", "def_allow_c__three",
)

#: Per-roster-slot static columns.
SLOT_COLS: tuple[str, ...] = (
    "shooter_make_c__rim", "shooter_att_c__rim", "prior_season_make_c__rim",
    "shooter_make_c__jump2", "shooter_att_c__jump2", "prior_season_make_c__jump2",
    "shooter_make_c__three", "shooter_att_c__three", "prior_season_make_c__three",
    "has_prior_season_fg", "pos_G", "pos_F", "pos_C",
    "shooter_games_asof", "shooter_fga_asof",
    "shooter_ft_asof", "shooter_fta_asof", "prior_season_ft", "has_prior_season_ft",
)


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def asof_backward(target: pd.DataFrame, source: pd.DataFrame, by: list[str],
                  on: str, cols: list[str]) -> pd.DataFrame:
    """`merge_asof(direction="backward")` of `cols` from `source` onto `target`.

    Both frames must carry `by` and `on`. Exact date matches are allowed: a
    player's as-of row FOR THE GAME BEING PREPARED is the correct value (every
    source column is itself a strictly-before expanding statistic); an earlier
    row is at most one of his own games stale. Nothing forward-looking can be
    selected."""
    t = target.copy()
    s = source[by + [on] + cols].copy()
    # one datetime resolution on both sides: the caches come from parquet with
    # mixed [s] / [ms] / [ns] units and merge_asof refuses to mix them.
    t[on] = pd.to_datetime(t[on]).astype("datetime64[ns]")
    s[on] = pd.to_datetime(s[on]).astype("datetime64[ns]")
    t = t.sort_values(on, kind="stable").reset_index(drop=True)
    s = s.sort_values(on, kind="stable").reset_index(drop=True)
    return pd.merge_asof(t, s, on=on, by=by,
                         direction="backward", allow_exact_matches=True)


def dedupe_team_rows(df: pd.DataFrame, game_col: str, team_col: str,
                     cols: list[str]) -> pd.DataFrame:
    d = df[[game_col, team_col] + cols].drop_duplicates(subset=[game_col, team_col])
    return d.rename(columns={game_col: "game_id", team_col: "team_id"})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="F2", choices=sorted(FOLD_TRAIN))
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--max-games", type=int, default=0, help="debug: cap the universe")
    args = ap.parse_args()
    t0 = time.time()
    season = int(args.season)
    train_seasons = FOLD_TRAIN[args.fold]
    assert_not_sealed([season], context=f"engine inputs {args.fold}/{season}")
    assert_not_sealed(train_seasons, context=f"engine inputs {args.fold} train")
    all_seasons = sorted(set(train_seasons + [season]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.fold}_{season}"
    meta: dict = {"fold": args.fold, "season": season, "train_seasons": train_seasons}
    rules: dict = {}

    # ---- 1. universe -----------------------------------------------------
    games = REF.load_actual_games(season)
    if args.max_games:
        games = games.sort_values("game_date").head(args.max_games).reset_index(drop=True)
    games = games[["game_id", "season", "game_date", "tipoff_utc", "home_team_id",
                   "away_team_id", "neutral"]].reset_index(drop=True)
    games["game_date"] = pd.to_datetime(games["game_date"])
    G = len(games)
    gpos = pd.Series(np.arange(G), index=games["game_id"].to_numpy())
    log(f"universe: {G} games in season {season}", t0)

    team_static = np.zeros((G, 2, len(TEAM_COLS)), dtype=np.float32)
    team_names = {c: i for i, c in enumerate(TEAM_COLS)}
    filled = {c: 0 for c in TEAM_COLS}

    def put_team(col: str, frame: pd.DataFrame, value_col: str) -> None:
        """Scatter one (game_id, team_id) -> value frame into `team_static`."""
        j = team_names[col]
        f = frame.dropna(subset=["game_id", "team_id"])
        gi = f["game_id"].map(gpos)
        ok = gi.notna().to_numpy()
        gi = gi[ok].to_numpy().astype(np.int64)
        tid = f.loc[ok, "team_id"].to_numpy()
        val = f.loc[ok, value_col].to_numpy(dtype=np.float64)
        home = games["home_team_id"].to_numpy()[gi]
        side = np.where(tid == home, 0, 1).astype(np.int64)
        good = np.isfinite(val)
        team_static[gi[good], side[good], j] = val[good].astype(np.float32)
        filled[col] += int(good.sum())

    # ---- 2a. possession_outcome team form (cached round-1 design) ---------
    po_cols = list(TEAM_COLS[:16])
    po = pd.read_parquet(PO_DESIGN, columns=["game_id", "season", "offense_team_id"] + po_cols)
    po = po[po["season"] == season]
    po = dedupe_team_rows(po, "game_id", "offense_team_id", po_cols)
    for c in po_cols:
        put_team(c, po, c)
    log(f"possession_outcome form: {len(po)} team-games", t0)
    del po

    # ---- 2b. clock team form (cached round-1 design) ----------------------
    ck_cols = ["off_tempo_rel", "def_tempo_rel", "tempo_prior_game"]
    ck = pd.read_parquet(CLOCK_DESIGN, columns=["game_id", "season", "offense_team_id"] + ck_cols)
    ck = ck[ck["season"] == season]
    ck = dedupe_team_rows(ck, "game_id", "offense_team_id", ck_cols)
    for c in ck_cols:
        put_team(c, ck, c)
    # the clock design covers fewer team-games than the universe (its own
    # CBBD-complete restriction). A team-game it does not cover takes the
    # module's OWN declared NaN fallback -- tempo_rel 1.0, tempo_prior_game the
    # as-of league tempo mean -- never a zero, which on this scale is not a
    # missing value but a 0-possession game.
    tempo_default = float(np.nanmedian(ck["tempo_prior_game"].to_numpy()))
    for c, default in (("off_tempo_rel", 1.0), ("def_tempo_rel", 1.0),
                       ("tempo_prior_game", tempo_default)):
        j = team_names[c]
        blank = team_static[:, :, j] == 0.0
        team_static[:, :, j][blank] = np.float32(default)
    rules["clock_tempo_fallback"] = {"off_tempo_rel": 1.0, "def_tempo_rel": 1.0,
                                     "tempo_prior_game": round(tempo_default, 6),
                                     "n_team_games_filled": int(
                                         2 * G - len(ck.drop_duplicates(["game_id", "team_id"])))}
    log(f"clock form: {len(ck)} team-games (fallback tempo_prior_game={tempo_default:.3f})", t0)
    del ck

    # ---- 2c. rebound design (rebuilt; also the training slice) ------------
    rb_events = pd.read_parquet(Path("data/processed/models/rebound/events_v1.parquet"))
    rb_universe = ES.load_universe(require_pbp_complete=False)
    rb_design = RB.build_design(all_seasons, universe=rb_universe, events=rb_events)
    log(f"rebound design: {len(rb_design):,} rows", t0)
    rb_team = dedupe_team_rows(rb_design[rb_design["season"] == season],
                               "game_id", "off_team_id", ["off_oreb_c", "opp_def_dreb_c"])
    put_team("off_oreb_c", rb_team, "off_oreb_c")
    put_team("opp_def_dreb_c", rb_team, "opp_def_dreb_c")
    rules["dead_share"] = RB.deterministic_dead_share(
        rb_design[rb_design["season"].isin(train_seasons)])
    log(f"dead-ball share by miss type (train fold): {rules['dead_share']}", t0)

    # ---- 2d. fg_make design (rebuilt; also the training slice) ------------
    fg_events = pd.read_parquet(Path("data/processed/models/fg_make/events_v2.parquet"))
    fg_universe = ES.load_universe(require_pbp_complete=True)
    fg_design = FG.build_design(all_seasons, universe=fg_universe, version="v2", events=fg_events)
    log(f"fg_make design: {len(fg_design):,} rows", t0)
    fg_season = fg_design[fg_design["season"] == season]
    for cls, key in SHOT_KEY.items():
        sl = fg_season[fg_season["shot_class"] == cls]
        t = dedupe_team_rows(sl, "game_id", "off_team_id", ["off_make_c", "def_allow_c"])
        put_team(f"off_make_c__{key}", t, "off_make_c")
        put_team(f"def_allow_c__{key}", t, "def_allow_c")

    # team columns whose game had no fg design row (pbp-incomplete games) fall
    # back to that team's most recent EARLIER covered game
    for cls, key in SHOT_KEY.items():
        sl = fg_design[fg_design["shot_class"] == cls]
        src = (sl[["off_team_id", "game_date", "off_make_c", "def_allow_c"]]
               .rename(columns={"off_team_id": "team_id"})
               .drop_duplicates(subset=["team_id", "game_date"]))
        src["game_date"] = pd.to_datetime(src["game_date"])
        for side in (0, 1):
            j_o = team_names[f"off_make_c__{key}"]
            j_d = team_names[f"def_allow_c__{key}"]
            miss = (team_static[:, side, j_o] == 0.0) & (team_static[:, side, j_d] == 0.0)
            if not miss.any():
                continue
            tgt = pd.DataFrame({
                "row": np.flatnonzero(miss),
                "team_id": games["home_team_id" if side == 0 else "away_team_id"].to_numpy()[miss],
                "game_date": games["game_date"].to_numpy()[miss],
            })
            got = asof_backward(tgt, src, ["team_id"], "game_date",
                                ["off_make_c", "def_allow_c"])
            r = got["row"].to_numpy()
            for j, c in ((j_o, "off_make_c"), (j_d, "def_allow_c")):
                v = got[c].to_numpy(dtype=np.float64)
                ok = np.isfinite(v)
                team_static[r[ok], side, j] = v[ok].astype(np.float32)
    log("fg_make team form scattered", t0)

    # ---- 3. rotation priors ---------------------------------------------
    fit = ROT.RotationFit.from_json(ROT_FIT)
    tp = ROT.load_team_possessions(season)
    pgm = ROT.player_game_minutes(tp)
    try:
        # the same three-step join `scripts/train_rotation_v1.load_season` uses:
        # CBBD game id -> ESPN game id -> the fouling player's team
        gu = pd.read_parquet(Path("data/processed/games_universe.parquet"),
                             columns=["game_id", "cbbd_game_id", "season"])
        gu = gu[gu["season"] == season][["game_id", "cbbd_game_id"]]
        fouls = ROT.player_game_fouls(season).merge(gu, on="cbbd_game_id", how="inner")
        fouls = fouls.merge(pgm[["game_id", "team_id", "pid"]].drop_duplicates(),
                            on=["game_id", "pid"], how="inner")
        print(f"  rotation fouls: {len(fouls):,} personal fouls joined to team-games")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  rotation fouls unavailable ({exc}); fpm falls back to the fitted league rate")
        fouls = None
    feats = ROT.build_asof_player_features(pgm, fouls=fouls)
    priors = ROT.build_priors(feats, fit, min_prior_games=1)
    log(f"rotation priors: {len(priors)} team-games (min_prior_games=1)", t0)
    del tp, pgm, feats

    S = N_SLOTS
    roster_cbbd = np.zeros((G, 2, S), dtype=np.int64)
    roster_valid = np.zeros((G, 2, S), dtype=bool)
    rot_share = np.zeros((G, 2, S), dtype=np.float32)
    rot_srank = np.zeros((G, 2, S), dtype=np.int16)
    rot_start = np.zeros((G, 2, S), dtype=np.int16)
    rot_fpm = np.full((G, 2, S), float(fit.fpm_league), dtype=np.float32)
    rot_pavail = np.zeros((G, 2, S), dtype=np.float32)

    # league-mean fallback profile: the FITTED role prior by as-of rank and the
    # fitted P(plays | rank). No team identity, no look-ahead, anonymous ids.
    role = np.asarray(fit.role_prior, dtype=np.float64)[:S]
    if len(role) < S:
        role = np.concatenate([role, np.full(S - len(role), role[-1] * fit.tail_ratio)])
    fb_share = (role / role.sum()).astype(np.float32)
    p_play = np.asarray(fit.p_play, dtype=np.float64) if fit.p_play else np.full(S, 0.85)
    fb_pavail = p_play[np.clip(np.arange(1, S + 1), 1, len(p_play)) - 1].astype(np.float32)

    n_fallback = 0
    for side, tcol in ((0, "home_team_id"), (1, "away_team_id")):
        tids = games[tcol].to_numpy()
        gids = games["game_id"].to_numpy()
        for i in range(G):
            pr = priors.get((int(gids[i]), int(tids[i])))
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
    meta["rotation_fallback_team_games"] = int(n_fallback)
    meta["rotation_fallback_pct"] = round(100.0 * n_fallback / max(2 * G, 1), 3)
    log(f"rotation flattened; {n_fallback} team-games ({meta['rotation_fallback_pct']}%) "
        "used the fitted league-mean fallback profile", t0)
    del priors

    # ---- 4. per-slot player form ----------------------------------------
    slot_static = np.zeros((G, 2, S, len(SLOT_COLS)), dtype=np.float32)
    slot_names = {c: i for i, c in enumerate(SLOT_COLS)}

    flat = pd.DataFrame({
        "row": np.repeat(np.arange(G), 2 * S),
        "side": np.tile(np.repeat([0, 1], S), G),
        "slot": np.tile(np.arange(S), 2 * G),
        "pid": roster_cbbd.reshape(-1),
        "game_date": np.repeat(games["game_date"].to_numpy(), 2 * S),
    })
    real = flat[flat["pid"] > 0].copy()
    log(f"per-slot join: {len(real):,} named roster slots of {len(flat):,}", t0)

    def put_slot(col: str, frame: pd.DataFrame, value_col: str) -> None:
        j = slot_names[col]
        v = frame[value_col].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        slot_static[frame["row"].to_numpy()[ok], frame["side"].to_numpy()[ok],
                    frame["slot"].to_numpy()[ok], j] = v[ok].astype(np.float32)

    # fg_make shooter block, per class then class-independent
    for cls, key in SHOT_KEY.items():
        sl = fg_design[fg_design["shot_class"] == cls]
        src = (sl[["shooter_id", "game_date", "shooter_make_c", "shooter_att_c",
                   "prior_season_make_c"]]
               .rename(columns={"shooter_id": "pid"})
               .drop_duplicates(subset=["pid", "game_date"]))
        src["game_date"] = pd.to_datetime(src["game_date"])
        src["pid"] = src["pid"].astype("int64")
        got = asof_backward(real, src, ["pid"], "game_date",
                            ["shooter_make_c", "shooter_att_c", "prior_season_make_c"])
        put_slot(f"shooter_make_c__{key}", got, "shooter_make_c")
        put_slot(f"shooter_att_c__{key}", got, "shooter_att_c")
        put_slot(f"prior_season_make_c__{key}", got, "prior_season_make_c")
    src = (fg_design[["shooter_id", "game_date", "has_prior_season", "pos_G", "pos_F",
                      "pos_C", "shooter_games_asof", "shooter_fga_asof"]]
           .rename(columns={"shooter_id": "pid"})
           .drop_duplicates(subset=["pid", "game_date"]))
    src["game_date"] = pd.to_datetime(src["game_date"])
    src["pid"] = src["pid"].astype("int64")
    got = asof_backward(real, src, ["pid"], "game_date",
                        ["has_prior_season", "pos_G", "pos_F", "pos_C",
                         "shooter_games_asof", "shooter_fga_asof"])
    put_slot("has_prior_season_fg", got, "has_prior_season")
    for c in ("pos_G", "pos_F", "pos_C", "shooter_games_asof", "shooter_fga_asof"):
        put_slot(c, got, c)
    log("fg_make shooter block joined", t0)

    # free_throw shooter block
    ft_att = pd.read_parquet(Path("data/processed/models/free_throw/attempts_v1_era.parquet"))
    ft_att = ft_att[ft_att["season"].isin(all_seasons)]
    ft_design = FT.build_ft_design(ft_att)
    log(f"free_throw design: {len(ft_design):,} rows", t0)
    src = (ft_design[["shooter_id", "game_date", "shooter_ft_asof", "shooter_fta_asof",
                      "prior_season_ft", "has_prior_season"]]
           .rename(columns={"shooter_id": "pid"})
           .drop_duplicates(subset=["pid", "game_date"]))
    src["game_date"] = pd.to_datetime(src["game_date"])
    src["pid"] = src["pid"].astype("int64")
    got = asof_backward(real, src, ["pid"], "game_date",
                        ["shooter_ft_asof", "shooter_fta_asof", "prior_season_ft",
                         "has_prior_season"])
    put_slot("shooter_ft_asof", got, "shooter_ft_asof")
    put_slot("shooter_fta_asof", got, "shooter_fta_asof")
    put_slot("prior_season_ft", got, "prior_season_ft")
    put_slot("has_prior_season_ft", got, "has_prior_season")
    log("free_throw shooter block joined", t0)

    # ---- 5. usage shrunk rates ------------------------------------------
    up = json.loads(USAGE_PARAMS.read_text(encoding="utf-8"))["per_class"]
    asof = pd.read_parquet(USAGE_ASOF)
    asof = asof[asof["season"].isin(all_seasons)]
    asof["game_date"] = pd.to_datetime(asof["game_date"])
    usage_rate = np.zeros((G, 2, S, len(USAGE_CLASSES)), dtype=np.float32)
    for k, cls in enumerate(USAGE_CLASSES):
        prior_kind = up[cls]["prior_kind"]
        m = float(up[cls]["shrink_m"])
        pri_col = {"league": f"lg_rate_{cls}", "position": f"pos_rate_{cls}",
                   "prior_season": f"prev_rate_{cls}"}[prior_kind]
        d = asof[["player_id", "game_date", "exposure_asof", f"ev_{cls}", pri_col]].copy()
        d["prior"] = d[pri_col].astype("float64")
        d["prior"] = d["prior"].fillna(d["prior"].median())
        d["u_rate"] = ((m * d["prior"] + d[f"ev_{cls}"].astype("float64"))
                       / (m + d["exposure_asof"].astype("float64")))
        d = d.rename(columns={"player_id": "pid"})[["pid", "game_date", "u_rate"]]
        d["pid"] = d["pid"].astype("int64")
        d = d.drop_duplicates(subset=["pid", "game_date"])
        got = asof_backward(real, d, ["pid"], "game_date", ["u_rate"])
        v = got["u_rate"].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        usage_rate[got["row"].to_numpy()[ok], got["side"].to_numpy()[ok],
                   got["slot"].to_numpy()[ok], k] = v[ok].astype(np.float32)
        # a slot with no as-of row at all sits at the class prior, which is the
        # model's own no-history state, not a zero
        fallback = float(np.nanmedian(d["u_rate"].to_numpy()))
        blank = usage_rate[:, :, :, k] == 0.0
        usage_rate[:, :, :, k][blank] = fallback
        rules[f"usage_prior_{cls}"] = {"prior_kind": prior_kind, "m": m,
                                       "no_history_rate": round(fallback, 6)}
    log("usage shrunk rates built", t0)

    # ---- 6. per-player rebound rates (attribution only, L17) -------------
    reb_rate = np.zeros((G, 2, S, 2), dtype=np.float32)
    prr = RB.player_rebound_rates(rb_events[rb_events["season"].isin(all_seasons)],
                                  prior_opps=50)
    prr["game_date"] = pd.to_datetime(prr["game_date"])
    src = (prr[["player_id", "game_date", "oreb_rate", "dreb_rate"]]
           .rename(columns={"player_id": "pid"}).drop_duplicates(subset=["pid", "game_date"]))
    src["pid"] = src["pid"].astype("int64")
    got = asof_backward(real, src, ["pid"], "game_date", ["oreb_rate", "dreb_rate"])
    for k, c in enumerate(("oreb_rate", "dreb_rate")):
        v = got[c].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        reb_rate[got["row"].to_numpy()[ok], got["side"].to_numpy()[ok],
                 got["slot"].to_numpy()[ok], k] = v[ok].astype(np.float32)
        med = float(np.nanmedian(src[c].to_numpy()))
        blank = reb_rate[:, :, :, k] == 0.0
        reb_rate[:, :, :, k][blank] = med
    log("player rebound rates joined", t0)
    del prr, src, got

    # ---- 7. ESPN athlete ids for players.parquet -------------------------
    roster_espn = np.full((G, 2, S), -1, dtype=np.int64)
    try:
        from cbb_sim.data import player_ids as PID
        cw = pd.read_parquet(CROSSWALK)
        m = PID.cbbd_to_espn_map(cw, season)
        lut = pd.Series(m) if not isinstance(m, pd.Series) else m
        flat_pid = roster_cbbd.reshape(-1)
        mapped = pd.Series(flat_pid).map(lut).to_numpy()
        mapped = np.where(pd.isna(mapped), -1, mapped).astype(np.int64)
        roster_espn = mapped.reshape(G, 2, S)
        meta["espn_id_coverage_pct"] = round(
            100.0 * float((roster_espn[roster_cbbd > 0] > 0).mean()), 2)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  ESPN crosswalk unavailable ({exc}); players.parquet carries CBBD ids "
              "and records that in run_meta")
        meta["espn_id_coverage_pct"] = 0.0
    log(f"ESPN id coverage {meta.get('espn_id_coverage_pct')}%", t0)

    # ---- 8. adapter fits the sub-model trainers never persisted ----------
    import joblib

    rb_tr, _ = RB.fold_slices(rb_design, args.fold)
    rb_feats = RB.feature_set("C_plus_state")
    rb_model = RB.fit_arm("lgbm", rb_tr, rb_feats, seed=0)
    joblib.dump({"arm": "lgbm", "feature_set": "C_plus_state", "fold": args.fold,
                 "features": rb_feats, "model": rb_model, "classes": RB.CLASSES,
                 "why": "rebound bake-off winner lgbm/C_plus_state; train_rebound_v1.py "
                        "persists no model, so the engine refits the winner's own spec"},
                out_dir / f"rebound_{args.fold}.joblib")
    log("rebound lgbm/C_plus_state refit", t0)

    ft_tr, _ = FT.fold_slices(ft_design, args.fold)
    Xtr = FT.design_matrix(ft_tr)
    ytr = ft_tr["y"].to_numpy()
    ft_model = FT.LgbmArm(seed=0).fit(Xtr, ytr)
    joblib.dump({"arm": "lgbm", "fold": args.fold, "features": list(FT.FT_FEATURES),
                 "model": ft_model, "classes": FT.CLASSES,
                 "why": "free-throw bake-off winner lgbm; train_free_throw_v1.py persists "
                        "no model, so the engine refits the winner's own spec"},
                out_dir / f"free_throw_{args.fold}.joblib")
    log("free_throw lgbm refit", t0)

    # Decision 8: the corrected responsiveness gate selects LightGBM for FGA_3,
    # but `winner_FGA_3.joblib` predates the decision and still holds the flat
    # team_baseline. Fit the Decision-8 model here and let the engine choose by
    # flag; the on-disk artifact is never touched.
    fg_tr, _ = FG.fold_slices(fg_design, args.fold)
    fg3 = FG.fit_arm("lgbm", FG.class_slice(fg_tr, "FGA_3"), "C_plus_state", seed=0,
                     params={"num_leaves": 31, "min_child_samples": 200})
    joblib.dump({"arm": "lgbm", "feature_set": "C_plus_state", "fold": args.fold,
                 "shot_class": "FGA_3", "features": FG.feature_set("C_plus_state"),
                 "model": fg3,
                 "why": "ARCHITECTURE_DECISIONS Decision 8 selects LightGBM for FGA_3; "
                        "winner_FGA_3.joblib predates that decision and was not re-exported"},
                out_dir / f"fg_make_FGA_3_decision8_{args.fold}.joblib")
    log("fg_make FGA_3 Decision-8 lgbm fit", t0)

    # ---- 9. rule constants derived from the training fold ----------------
    tr_mask = fg_design["season"].isin(train_seasons)
    and_one = {}
    for cls in FG.SHOT_CLASSES:
        s = fg_design[tr_mask & (fg_design["shot_class"] == cls) & fg_design["made"]]
        and_one[cls] = round(float(s["and_one"].mean()), 6) if len(s) else 0.0
    rules["and_one_rate_given_made"] = and_one

    trips = pd.read_parquet(Path("data/processed/models/free_throw/trips_v1_era.parquet"))
    trips = trips[trips["season"].isin(train_seasons)]
    sh = trips[trips["foul_class"] == "shooting"]
    rules["shooting_trip_three_attempt_share"] = round(
        float((sh["trip_len"] == 3).mean()), 6) if len(sh) else 0.0
    db = trips[trips["foul_class"] == "double_bonus"]
    rules["double_bonus_three_attempt_share"] = round(
        float((db["trip_len"] == 3).mean()), 6) if len(db) else 0.0
    rules["technical_trip_rate_per_team_game"] = round(float(
        len(trips[trips["foul_class"] == "technical"])
        / max(trips["game_id"].nunique() * 2, 1)), 6)

    # The non-shooting foul that awards no attempt is invisible to the L3 class
    # vocabulary but drives the bonus. Measure it: team fouls per defensive
    # possession that produce NO free-throw trip, on the training fold.
    try:
        fl = []
        for s in train_seasons:
            f = ROT.player_game_fouls(int(s))
            f["season"] = s
            fl.append(f)
        fouls_tr = pd.concat(fl, ignore_index=True)
        n_fouls = len(fouls_tr)
        n_trip_fouls = int(len(trips[trips["foul_class"] != "technical"]))
        po_tr = pd.read_parquet(PO_DESIGN, columns=["game_id", "season", "poss_index"])
        po_tr = po_tr[po_tr["season"].isin(train_seasons)]
        n_poss = int(po_tr.drop_duplicates(["game_id", "poss_index"]).shape[0])
        silent = max(n_fouls - n_trip_fouls, 0) / max(n_poss, 1)
        rules["silent_foul_per_possession"] = round(float(silent), 6)
        rules["silent_foul_source"] = {
            "n_personal_fouls": int(n_fouls), "n_trip_fouls": n_trip_fouls,
            "n_possessions": n_poss,
            "note": "personal fouls that award no free-throw trip, per possession; "
                    "the L3 class vocabulary has no class for them and the bonus "
                    "cannot arrive on time without them (provisional_foul_accrual)"}
    except Exception as exc:                                    # noqa: BLE001
        rules["silent_foul_per_possession"] = 0.0
        rules["silent_foul_source"] = {"error": str(exc)}
    log(f"rule constants: {json.dumps({k: v for k, v in rules.items() if not isinstance(v, dict)})}",
        t0)

    # median chance-elapsed time by chance number, for the fg_make state block
    ce = fg_design[tr_mask].groupby(
        fg_design.loc[tr_mask, "chance_number"].clip(1, 3))["chance_elapsed_s"].median()
    rules["chance_elapsed_median_by_chance"] = {int(k): round(float(v), 3)
                                                for k, v in ce.items()}

    meta["team_columns_filled"] = filled
    meta["built_at"] = pd.Timestamp.utcnow().isoformat()
    meta["n_games"] = G
    meta["n_slots"] = S

    inp = EngineInputs(
        games=games, team_static=team_static, team_names=team_names,
        slot_static=slot_static, slot_names=slot_names,
        roster_cbbd=roster_cbbd, roster_espn=roster_espn, roster_valid=roster_valid,
        rot_share=rot_share, rot_srank=rot_srank, rot_start=rot_start,
        rot_fpm=rot_fpm, rot_pavail=rot_pavail,
        usage_rate=usage_rate, usage_classes=USAGE_CLASSES, reb_rate=reb_rate,
        rules=rules, meta=meta,
    )
    inp.save(out_dir, tag)
    log(f"wrote {out_dir}/(games|arrays|names)_{tag}.*", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
