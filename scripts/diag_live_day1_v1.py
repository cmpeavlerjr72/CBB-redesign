"""
diag_live_day1_v1.py -- point the live path at the first REAL 2026-27 slate and list exactly
what breaks or is degenerate. Nothing is invented, filled or borrowed to make it run.

    .venv/Scripts/python.exe scripts/diag_live_day1_v1.py [--date 2026-11-02]

SEAL. 2025-26 (season 2026) is sealed for modelling, so this script builds the context with
`prior_season=False` (source tables cut to season 2027 only). That reproduces exactly the
'no history' state of day 1 and computes nothing against 2026 results. What the prior-season
carry WOULD read is listed from the code, not run.

Each family is attempted inside try/except; the row records the exception or, when the block
runs, how degenerate its output is. Writes results/live_day1_2026-09-30/day1.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "LIGHTGBM_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_engine_inputs_live as BL  # noqa: E402
from cbb_sim.live import features as LF  # noqa: E402
from cbb_sim.live import players as LP  # noqa: E402
from cbb_sim.models import rotation as ROT  # noqa: E402

PRE = ROOT / "data/raw/preseason/2027_v2_20260930"
OUT = ROOT / "results/live_day1_2026-09-30"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-11-02")
    ap.add_argument("--crosswalk", default="data/reference/team_crosswalk.parquet")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {"date": args.date, "rows": [], "created_at": pd.Timestamp.now("UTC").isoformat()}

    def row(family, feature, status, detail, consumer):
        rep["rows"].append({"family": family, "feature": feature, "status": status, "detail": detail,
                            "consumer": consumer})
        print(f"[{status:10}] {family:22} {feature:34} {detail[:110]}", flush=True)

    # ---- schedule --------------------------------------------------------
    raw = pd.read_parquet(PRE / "games_2027.parquet")
    d_all = raw[pd.to_datetime(raw["startDate"], utc=True).dt.tz_convert("America/New_York")
                .dt.strftime("%Y-%m-%d") == args.date]
    slate = BL.load_slate_from_cbbd(str(PRE / "games_2027.parquet"), args.date, args.crosswalk)
    unm = slate.attrs.get("unmapped", [])
    cw_v2 = BL.load_slate_from_cbbd(str(PRE / "games_2027.parquet"), args.date, "data/reference/team_crosswalk_v2.parquet")
    allg = raw[(raw["homeTeam"] == "West Florida") | (raw["awayTeam"] == "West Florida")]
    rep["west_florida"] = {"games_in_cbbd_2027": int(len(allg)),
                           "unmapped_on_date_with_crosswalk_v1": [u for u in unm if "West Florida" in (u["homeTeam"], u["awayTeam"])],
                           "mapped_games_on_date_v1": int(len(slate)), "mapped_games_on_date_v2": int(len(cw_v2))}
    cwv1 = pd.read_parquet(args.crosswalk)
    cwv2 = pd.read_parquet("data/reference/team_crosswalk_v2.parquet")
    d1 = set(cwv1["cbbd_team_id"].astype("int64")); d2 = set(cwv2["cbbd_team_id"].astype("int64"))
    both = lambda ids: raw["homeTeamId"].astype("int64").isin(ids) & raw["awayTeamId"].astype("int64").isin(ids)  # noqa: E731
    rep["west_florida"].update({
        "season_games_both_sides_mapped_v1": int(both(d1).sum()), "season_games_both_sides_mapped_v2": int(both(d2).sum()),
        "wf_games_with_a_mapped_opponent_v1": int((both(d1 | {1073}) & ~both(d1)).sum()),
        "wf_games_recovered_by_v2": int(both(d2).sum() - both(d1).sum()),
        "season_games_total": int(len(raw))})
    rep["schedule"] = {"cbbd_games_that_date": int(len(d_all)), "mapped_to_engine_ids": int(len(slate)),
                       "unmapped": unm, "start_time_tbd": int(d_all["startTimeTbd"].sum())}
    row("schedule", "team crosswalk (CBBD -> ESPN ids)", "BREAKS" if unm else "ok",
        f"{len(unm)} of {len(d_all)} games have an unmapped side: {[(u['homeTeam'], u['awayTeam']) for u in unm][:6]}",
        "every engine array is keyed on ESPN team ids")
    row("schedule", "tipoff_utc", "DEGENERATE" if d_all["startTimeTbd"].all() else "partial",
        f"{int(d_all['startTimeTbd'].sum())} of {len(d_all)} games have startTimeTbd=True; CBBD stamps "
        f"{d_all['startDate'].iloc[0]} (midnight ET placeholder), so created_at < tipoff can only be proven "
        "for a run made BEFORE midnight ET of game day unless real tip times are supplied (hoopR/ESPN)",
        "created_at < tipoff guard")
    rep["hoopr_games_that_date"] = int(pd.read_parquet(PRE / "hoopr_mbb_schedule_2027.parquet")
                                       .assign(d=lambda x: x["game_date"].astype(str)).query("d == @args.date").shape[0])

    # ---- context, season 2027 only (sealed 2026 not read) ------------------
    as_of = pd.Timestamp(f"{args.date}T00:00:00Z") - pd.Timedelta(hours=6)
    slate2 = slate.copy()
    universe = pd.read_parquet(BL.UNIVERSE)
    ctx = LF.build_ctx(slate2, as_of, 2027, universe, season_start=slate2["game_date"].min(),
                       prior_season=False)
    rep["ctx"] = {"n_games": int(len(slate2)), "universe_prior_rows": int(len(ctx.universe_prior))}

    def attempt(family, feature, fn, consumer, describe):
        try:
            out = fn()
            row(family, feature, *describe(out), consumer)
            return out
        except Exception as exc:                                     # noqa: BLE001
            row(family, feature, "BREAKS", f"{type(exc).__name__}: {str(exc)[:220]}", consumer)
            rep.setdefault("tracebacks", {})[f"{family}:{feature}"] = traceback.format_exc()[-1500:]
            return None

    ratings_p = ROOT / "data/processed/ratings/own_ratings_2027.parquet"
    row("ratings", "own_ratings_2027.parquet", "MISSING" if not ratings_p.exists() else "ok",
        "file absent; no daily/2027 entry point (build_all_seasons is a batch)" if not ratings_p.exists() else "",
        "possession_outcome, clock (tempo_rel, league_tempo_mean), rebound, fg_make: off/def_rating_* and tempo columns")
    attempt("ratings", "rating_site_block", lambda: LF.rating_site_block(ctx),
            "possession_outcome, clock, rebound, fg_make (rating/site/tempo/season/days columns)",
            lambda o: ("ok", "ran"))

    def nz(frame, cols):
        z = {c: float((frame[c] == 0).mean()) for c in cols}
        return z

    po = attempt("possession_outcome", "off/opp_def x {3pa,rim,tov,ftr}_c (8 cols)", lambda: LF.po_team_block(ctx),
                 "possession_outcome (event mix), engine event adapter",
                 lambda o: ("DEGENERATE", f"runs; share of team-games exactly 0.0 = "
                            f"{np.mean([(o[c] == 0).mean() for c in [f'{p}_{r}_c' for p in ('off', 'opp_def') for r in ('3pa', 'rim', 'tov', 'ftr')]]):.2f} "
                            f"(league-mean, no history; no prior-season carry exists in the code)"))
    rb = LF.rebound_events_cut(ctx, str(BL.RB_EVENTS))
    attempt("rebound", "off_oreb_c, opp_def_dreb_c", lambda: LF.rebound_team_block(ctx, rb),
            "rebound adapter",
            lambda o: ("DEGENERATE", f"runs; share exactly 0.0 = {float(((o['off_oreb_c'] == 0) & (o['opp_def_dreb_c'] == 0)).mean()):.2f}"))

    fit = ROT.RotationFit.from_json(BL.B.ROT_FIT)
    pri = attempt("rotation", "roster / rot_share / rot_srank / rot_start / rot_fpm / rot_pavail",
                  lambda: LF.rotation_priors(ctx, fit), "rotation adapter, attribution, usage slot join",
                  lambda o: ("DEGENERATE", f"{len(o)} of {2 * len(slate2)} team-games have a prior; the rest take the "
                             "anonymous league-mean profile (negative pids, no named players)"))
    cand = LF.candidates_from_priors(pri or {}, 15)
    rep["n_named_candidates"] = int(sum(len(v) for v in cand.values()))
    fg_ev = LF.fg_events_cut(ctx, str(BL.FG_EVENTS))
    cand_fg = {(int(g.game_id), int(t)): [-1] for g in slate2.itertuples(index=False)
               for t in (g.home_team_id, g.away_team_id)}
    fgd = attempt("fg_make", "off_make_c / def_allow_c per class (6 cols)", lambda: LF.fg_design_live(ctx, fg_ev, cand_fg),
                  "fg_make adapter (round-4 B1 etc.)",
                  lambda o: ("DEGENERATE", f"runs; {len(o[0])} stub rows; team form centred on a league rate that "
                             f"does not exist on day 1 -> {float((o[0]['off_make_c'] == 0).mean()) if len(o[0]) else float('nan'):.2f} "
                             "share exactly 0.0"))
    row("fg_make", "shooter_make_c / shooter_att_c / shooter_games_asof / shooter_fga_asof", "DEGENERATE",
        f"{rep['n_named_candidates']} named candidates (roster comes from last-season-independent on-floor history "
        "of season 2027 only), so the slot block is all zeros; prior_season_* need season-2026 tables joined at "
        "2027 (the code does this via `shooter_form` season+1 totals, but season 2027 has no events table)",
        "fg_make (shooter block), usage attribution")
    attempt("free_throw", "shooter_ft_asof / shooter_fta_asof / prior_season_ft / has_prior_season_ft",
            lambda: LP.ft_design_live(ctx, str(BL.FT_ATTEMPTS), cand),
            "free_throw adapter",
            lambda o: ("DEGENERATE", f"runs, {len(o)} design rows: no named candidates and no 2027 attempts; "
                       "prior_season_ft would need the 2026 attempts joined at season+1 (not read: sealed)"))
    attempt("usage", "usage_rate (5 classes)", lambda: LP.usage_asof_live(ctx, str(BL.USAGE_EVENTS_V2), cand),
            "usage adapter, attribution",
            lambda o: ("DEGENERATE", f"runs, {len(o)} player rows; every slot falls to the class no-history "
                       "median, which live takes from rows dated < D (none in season 2027: NaN -> 0.0)"))
    prr = attempt("attribution/rebound", "reb_rate (oreb_rate, dreb_rate)",
                  lambda: LF.rebound_player_rates(ctx, rb, cand, prior_opps=50), "attribution (L17)",
                  lambda o: ("DEGENERATE", f"runs, {len(o[0])} player rows; medians {o[1]}"))
    row("roster", "CBBD /teams/roster season 2027", "MISSING",
        f"{len(pd.read_parquet(PRE / 'roster_players_2027.parquet'))} player rows for "
        f"{len(pd.read_parquet(PRE / 'roster_teams_2027.parquet'))} teams (populates weeks before tip)",
        "positions (pos_G/F/C), rotation candidates for transfers/freshmen")
    row("availability", "data/overrides/availability.csv", "MISSING", "header only (0 rows)", "roster / injury override layer")
    cwp = pd.read_parquet(ROOT / "data/processed/player_crosswalk.parquet")
    row("ids", "player_crosswalk season 2027", "MISSING" if not (cwp["season"] == 2027).any() else "ok",
        f"seasons covered: {sorted(cwp['season'].unique().tolist())}", "roster_espn / players.parquet")

    # ---- artifacts and rule tables keyed on season/fold ----------------------
    eng = ROOT / "data/processed/models/engine"
    arts = sorted(p.name for p in eng.glob("event_*_*_*")) + sorted(p.name for p in eng.glob("*_F2_2025*.json"))[:6]
    row("adapters", "dated-refit artifacts for season 2027", "MISSING",
        f"only fold/season keys F2_2025 exist: {arts[:5]}...; Adapters.load(inp, fold, 2027) has no directory to "
        "read; a refit schedule through 2025-26 is required (manifest.py selects by date)",
        "event, clock, rebound, free_throw, fg_make adapters")
    try:
        from cbb_sim.engine import state as ST
        era = ST.load_bonus_era()
        row("rules", "bonus era table has season 2027", "ok" if 2027 in era else "BREAKS",
            f"seasons in table: {sorted(era)[-4:]}" if not isinstance(era, dict) else f"seasons: {sorted(era)[-4:]}",
            "engine GameState (rule-era flags)")
    except Exception as exc:                                        # noqa: BLE001
        row("rules", "bonus era table", "UNKNOWN", f"{type(exc).__name__}: {exc}", "engine GameState")
    row("rules", "rule constants (dead_share, and_one, foul accrual)", "STALE",
        "derived from fold-2 train seasons 2022-2024; copied from the template names file; recomputation for a "
        "2023-2026 fit is not wired", "engine loop")
    row("lines", "2027 line rows", "MISSING", "0 provider rows until ~1 week before tip (readiness audit section 1)",
        "market scorecard, CLV")
    (OUT / "day1.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("wrote", OUT / "day1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
