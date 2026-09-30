"""Player-level as-of families for the live path (free throw, usage, fg round-4 slot columns).

Same rule as `features.py`: call the module's own function on source tables cut
to `game_date < D` plus one stub row per candidate player, read the stub game's
row. See that module's docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.live import guards as G
from cbb_sim.live.features import Ctx, _template, cut


def _team_sides(gm):
    return ((True, gm.home_team_id, gm.away_team_id), (False, gm.away_team_id, gm.home_team_id))


# ---------------------------------------------------------------------------
# free-throw shooter block
# ---------------------------------------------------------------------------
def ft_design_live(ctx: Ctx, attempts_path: str, cand: dict) -> pd.DataFrame:
    from cbb_sim.models import free_throw as FT
    full = pd.read_parquet(attempts_path)
    a = cut(full, full["season"].isin(ctx.seasons) & full["game_id"].isin(ctx.prior_ids))
    G.assert_sources_before(a, ctx.slate_date, "free-throw attempts")
    tpl = _template(a)
    rows = []
    for gm in ctx.slate.itertuples(index=False):
        for is_home, tid, oid in _team_sides(gm):
            for pid in cand.get((int(gm.game_id), int(tid)), []):
                d = dict(tpl)
                d.update(season=ctx.season, game_id=int(gm.game_id), cbbd_game_id=int(gm.cbbd_game_id),
                         game_date=ctx.slate_date, neutral_site=bool(gm.neutral),
                         shooter_is_home=is_home, team_id=int(tid), opp_id=int(oid),
                         shooter_id=float(pid), period=1, seconds_remaining=1200, score_diff=0,
                         made=False, foul_class="shooting")
                rows.append(d)
    if not rows:
        return pd.DataFrame()
    stub = pd.DataFrame(rows, columns=list(a.columns)).astype(a.dtypes.to_dict())
    design = FT.build_ft_design(pd.concat([a, stub], ignore_index=True))
    return design[design["game_id"].isin(ctx.slate["game_id"])].reset_index(drop=True)


# ---------------------------------------------------------------------------
# usage as-of panel (v2) and the fitted shrunk rates
# ---------------------------------------------------------------------------
def usage_asof_live(ctx: Ctx, events_path: str, cand: dict) -> pd.DataFrame:
    from cbb_sim.models import usage as U
    full = pd.read_parquet(events_path)
    ev = cut(full, full["season"].isin(ctx.seasons) & full["game_id"].isin(ctx.prior_ids))
    G.assert_sources_before(ev, ctx.slate_date, "usage events")
    tpl = _template(ev)
    rows = []
    for gm in ctx.slate.itertuples(index=False):
        for is_home, tid, oid in _team_sides(gm):
            c = cand.get((int(gm.game_id), int(tid)), [])
            for k in range(0, len(c), 5):
                chunk = c[k:k + 5] + [-1] * (5 - len(c[k:k + 5]))
                d = dict(tpl)
                d.update(game_id=int(gm.game_id), cbbd_game_id=int(gm.cbbd_game_id), season=ctx.season,
                         game_date=ctx.slate_date, team_id=int(tid), opp_id=int(oid),
                         offense_is_home=is_home, neutral_site=bool(gm.neutral),
                         event_class="TOV", player_id=np.nan, y=0, five_ok=True, in_five=True)
                for j in range(5):
                    d[f"alt_{j + 1}"] = int(chunk[j])
                rows.append(d)
    stub = (pd.DataFrame(rows, columns=list(ev.columns)).astype(ev.dtypes.to_dict())
            if rows else ev.iloc[0:0])
    allev = pd.concat([ev, stub], ignore_index=True)
    per_season = {int(s): allev[allev["season"] == s] for s in ctx.seasons if (allev["season"] == s).any()}
    if not per_season:
        return pd.DataFrame(columns=["season", "player_id", "game_id", "game_date"])   # no history, no candidates
    minutes = U.load_minutes(ctx.seasons)
    minutes = minutes[minutes["game_id"].isin(ctx.prior_ids)]
    asof = U.build_player_asof(per_season, minutes=minutes)
    asof["game_date"] = pd.to_datetime(asof["game_date"])
    return asof


def usage_rates(asof: pd.DataFrame, slate_date, params: dict, classes) -> dict:
    """{class: (frame(game_id, pid, u_rate), fallback)}. Same formula as
    build_engine_inputs.build_v2 step 3; the fill medians come from rows dated
    strictly before D only (the backtest builder takes them over the whole
    season table: leak finding, quantified by the parity script)."""
    out = {}
    if not len(asof):
        return {cls: (pd.DataFrame(columns=["game_id", "pid", "u_rate"]), float("nan")) for cls in classes}
    past = (asof["game_date"] < slate_date).to_numpy()
    for cls in classes:
        prior_kind = params[cls]["prior_kind"]
        m = float(params[cls]["shrink_m"])
        pri_col = {"league": f"lg_rate_{cls}", "position": f"pos_rate_{cls}",
                   "prior_season": f"prev_rate_{cls}"}[prior_kind]
        d = asof[["player_id", "game_id", "game_date", "exposure_asof", f"ev_{cls}", pri_col]].copy()
        d["prior"] = d[pri_col].astype("float64")
        d["prior"] = d["prior"].fillna(d.loc[past, "prior"].median())
        d["u_rate"] = ((m * d["prior"] + d[f"ev_{cls}"].astype("float64"))
                       / (m + d["exposure_asof"].astype("float64")))
        fallback = float(np.nanmedian(d.loc[past, "u_rate"].to_numpy())) if past.any() else float("nan")
        d = d.rename(columns={"player_id": "pid"})
        out[cls] = (d[~past][["game_id", "pid", "u_rate"]].drop_duplicates(["game_id", "pid"]), fallback)
    return out


# ---------------------------------------------------------------------------
# fg_make round-4 slot columns
# ---------------------------------------------------------------------------
def r4_slot_live(ctx: Ctx, design: pd.DataFrame, fg_events_all: pd.DataFrame, cand: dict,
                 m: dict) -> pd.DataFrame:
    """Per (game_id, shooter_id, class_key) round-4 columns for the slate, from
    the live design rows. `m` is round4/m_fitted.json."""
    from cbb_sim.models import fg_make as FG
    from cbb_sim.models import prob_metrics as PM
    from cbb_sim.models.event_stream import DEFAULT_PBP_DIR as PBP_DIR
    d = design
    cls = d["shot_class"].to_numpy(dtype=object)
    mvec = np.array([float(m[c]["m"]) for c in cls], dtype="float64")
    team = d["off_make_raw"].to_numpy(dtype="float64")
    mk = d["shooter_mk_c"].to_numpy(dtype="float64")
    att = d["shooter_att_c"].to_numpy(dtype="float64")
    out = d[["game_id", "shooter_id", "class_key", "prior_season_att_c"]].copy()
    out["shooter_shrunk_dev_c"] = (((mvec * team + mk) / (mvec + att)) - team).astype("float32")
    sf = FG.shooter_form(fg_events_all)
    sf = sf[["shooter_id", "game_id", "att_rim", "att_jump2", "att_three", "att_all"]].copy()
    tot = sf["att_all"].to_numpy(dtype="float64")
    for k in ("rim", "jump2", "three"):
        sf[f"sh_share_{k}"] = np.where(tot > 0, sf[f"att_{k}"].to_numpy(dtype="float64")
                                       / np.maximum(tot, 1e-9), 0.0)
    out = out.merge(sf[["shooter_id", "game_id", "sh_share_rim", "sh_share_jump2", "sh_share_three"]]
                    .drop_duplicates(["shooter_id", "game_id"]), on=["shooter_id", "game_id"], how="left")
    # as-of assisted share: panel over every ATTEMPT (the round-4 missingness rule),
    # played games < D from pbp, plus one stub row per candidate for the slate game
    frames = []
    for s in ctx.seasons:
        p = Path(PBP_DIR) / f"plays_{s}.parquet"
        if not p.exists():
            continue
        pl = pd.read_parquet(p, columns=["gameId", "id", "shot_shooter_id", "shot_assisted", "shot_made"])
        pl = pl.drop_duplicates(subset=["gameId", "id"], keep="first")
        sh = pd.to_numeric(pl["shot_shooter_id"], errors="coerce")
        made = pl["shot_made"]
        if made.dtype == object:
            made = made.map({True: True, False: False})
        made = made.astype("boolean").fillna(False).to_numpy()
        asst = pl["shot_assisted"]
        if asst.dtype == object:
            asst = asst.map({True: True, False: False})
        asst = asst.astype("boolean").fillna(False).to_numpy()
        keep = np.isfinite(sh.to_numpy())
        frames.append(pd.DataFrame({
            "season": s, "cbbd_game_id": pl.loc[keep, "gameId"].to_numpy(),
            "shooter_id": sh[keep].to_numpy().astype("int64"), "fga": 1.0,
            "mk": made[keep].astype("float64"), "mk_asst": (made[keep] & asst[keep]).astype("float64")}))
    cols = ["season", "cbbd_game_id", "shooter_id", "fga", "mk", "mk_asst"]
    a = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    a = a.groupby(["season", "cbbd_game_id", "shooter_id"], as_index=False)[["fga", "mk", "mk_asst"]].sum()
    dates = (fg_events_all[["season", "cbbd_game_id", "game_date"]]
             .drop_duplicates(subset=["season", "cbbd_game_id"]))
    dates["game_date"] = pd.to_datetime(dates["game_date"])
    a = a.merge(dates, on=["season", "cbbd_game_id"], how="inner")
    a = a[a["game_date"] < ctx.slate_date].copy()
    a["_game_id"] = -1
    stub = []
    for gm in ctx.slate.itertuples(index=False):
        for tid in (gm.home_team_id, gm.away_team_id):
            for pid in cand.get((int(gm.game_id), int(tid)), []):
                stub.append({"season": ctx.season, "cbbd_game_id": int(gm.cbbd_game_id),
                             "shooter_id": int(pid), "fga": 0.0, "mk": 0.0, "mk_asst": 0.0,
                             "game_date": ctx.slate_date, "_game_id": int(gm.game_id)})
    a = pd.concat([a, pd.DataFrame(stub)], ignore_index=True)
    a = a.sort_values(["season", "shooter_id", "game_date", "cbbd_game_id"], kind="stable").reset_index(drop=True)
    asof = PM.expanding_asof(a, ["season", "shooter_id"], ["fga", "mk", "mk_asst"])
    res = pd.DataFrame({"shooter_id": a["shooter_id"], "_game_id": a["_game_id"],
                        "sh_assisted_share": np.where(asof["mk"].to_numpy() > 0,
                                                      asof["mk_asst"] / asof["mk"].clip(lower=1), 0.0)})
    res = res[res["_game_id"] > 0].rename(columns={"_game_id": "game_id"})
    out = out.merge(res, on=["shooter_id", "game_id"], how="left")
    return out
