#!/usr/bin/env python
"""
diag_a3_seed_wiring_v1.py -- daily-path vs live-path-harness check for the A3+R1 day-1 player prior wiring (ops, 2026-10-09;
docs/ops/a3_seed_wiring_2026-10-09.md). Wiring check only: no model, parameter or default is chosen here.

Arms on ONE slate, ONE clock, ONE seed set (paired RNG streams keyed on (seed, game_id, family)):
  daily         run_daily_sim_v1.run_sim_stage (the served daily sim stage, A3+R1 applied by the season rule), players kept
  anon          the same stage with day1_prior=None (the pre-wiring state: every slot anonymous)
  harness       build_engine_inputs_day1prior_v1.build(..., 'A3', anon=False, fallback='R1') = the build call of the live-path harness
                (build_engine_inputs_d1p_live_early_v1.py), serving=True for 2027; then the same adapter / shot-block attach / simulate
  harness_lut0  harness inputs with the harness's bake-off LUT convention (shot-block shooter/known zeroed on seeded sides)

    python scripts/diag_a3_seed_wiring_v1.py --slate-date 2026-11-02 --seeds 4 --now 2026-10-09T16:00:00Z
Output: results/a3_seed_wiring/<slate>/ (gitignored) + summary.json.
"""
from __future__ import annotations

import argparse
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
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
import run_daily_sim_v1 as SIM  # noqa: E402
import chain_daily_v3 as V3  # noqa: E402
import chain_daily_v2 as V2  # noqa: E402
import build_engine_inputs_day1prior_v1 as DP  # noqa: E402
from cbb_sim.live import tips as TP  # noqa: E402

COLS = ["game_id", "cbbd_game_id", "season", "game_date", "tipoff_utc", "home_team_id", "away_team_id", "neutral"]


def harness_arm(slate_date, season, now, rd, seeds, out: Path, lut_zero: bool):
    import run_engine_live as RL
    import build_shot_block_lut_live_v1 as SBL
    slate = SIM.load_slate(slate_date, season, "cbbd", str(V3.SCHED_2027), str(V3.CROSSWALK), "table")
    ok, _ = TP.select_for_pass(slate, now, "evening")
    inp, diag = DP.build(ok[COLS], now, season, "F2", "A3", season_start=SIM.season_start_of(season, "cbbd", str(V3.SCHED_2027)),
                         anon=False, strict_finish=False, minutes_source="onfloor", fallback="R1", serving=True,
                         roster_path=str(REPO / f"data/raw/cbbd/rosters/roster_{season}.parquet"), ratings_dir=rd)
    out.mkdir(parents=True, exist_ok=True)
    adir = RL.prepare_adapter_dir(inp.event_block, "F2", season, out / "_adapter")
    lp = SBL.attach(inp, out / "_adapter", as_of=now)
    if lut_zero:
        z = dict(np.load(lp))
        side = (inp.roster_cbbd > 0).any(axis=2)          # opening day: every named side is a seeded side
        for r, s in zip(*np.nonzero(side)):
            z["shooter"][r, s, :] = 0
            z["known"][r, s, :] = 0
        np.savez_compressed(lp, **z)
    games, pl, _ = RL.simulate(inp, "F2", season, np.asarray(seeds, dtype=np.int64), keep_players=True, adapter_dir=adir)
    games = RL.stamp_rows(games, inp, now)
    pl = RL.stamp_rows(pl, inp, now)
    games.to_parquet(out / "games.parquet", index=False)
    pl.to_parquet(out / "players.parquet", index=False)
    np.save(out / "roster_cbbd.npy", inp.roster_cbbd)
    return games, pl, inp.roster_cbbd, diag


def daily_arm(slate_date, season, now, rd, nseeds, root: Path, d1p):
    r = SIM.run_sim_stage(slate_date, season, "F2", nseeds, 0, now, root, None, "cbbd", str(V3.SCHED_2027), str(V3.CROSSWALK), "table",
                          strict=False, force=True, players=True, pass_name="evening", ratings_dir=rd, day1_prior=d1p)
    d = Path(r["out"])
    ros = np.load(d / "_adapter/shot_block_K2_Ocell_live.npz")["roster_cbbd"]
    pp = d / "players.parquet"                      # absent when no slot is named (anonymous arm): no player rows
    return pd.read_parquet(d / "games.parquet"), (pd.read_parquet(pp) if pp.exists() else pd.DataFrame()), ros, r


def team_minutes(gm: pd.DataFrame) -> pd.Series:
    ot = (gm["n_periods"] - 2).clip(lower=0)           # two halves + OT periods
    return 200.0 + 25.0 * ot


def cmp_players(p1: pd.DataFrame, p2: pd.DataFrame) -> dict:
    key = ["game_id", "seed", "athlete_id", "cbbd_id", "team_id"]
    m = p1.merge(p2, on=key, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m[m["_merge"] == "both"]
    a1 = p1.groupby(["game_id", "team_id", "cbbd_id"])[["minutes", "pts"]].mean()
    a2 = p2.groupby(["game_id", "team_id", "cbbd_id"])[["minutes", "pts"]].mean()
    j = a1.join(a2, lsuffix="_a", rsuffix="_b", how="outer")
    return {"player_rows_a": int(len(p1)), "player_rows_b": int(len(p2)), "rows_only_one_side": int((m["_merge"] != "both").sum()),
            "player_mean_rows": int(len(j)), "player_mean_rows_missing_one_side": int(j.isna().any(axis=1).sum()),
            "max_abs_diff_minutes_mean": float(np.nanmax(np.abs(j["minutes_a"] - j["minutes_b"]))) if len(j) else None,
            "max_abs_diff_pts_mean": float(np.nanmax(np.abs(j["pts_a"] - j["pts_b"]))) if len(j) else None,
            "mean_abs_diff_pts_mean": float(np.nanmean(np.abs(j["pts_a"] - j["pts_b"]))) if len(j) else None,
            "max_abs_diff_row_minutes": float(np.abs(both["minutes_a"] - both["minutes_b"]).max()) if len(both) else None,
            "max_abs_diff_row_pts": int(np.abs(both["pts_a"].astype(int) - both["pts_b"].astype(int)).max()) if len(both) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate-date", default="2026-11-02")
    ap.add_argument("--season", type=int, default=2027)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--now", default="2026-10-09T16:00:00Z")
    ap.add_argument("--ratings-dir", default=None)
    a = ap.parse_args()
    now = pd.Timestamp(a.now)
    rd = a.ratings_dir or str(REPO / f"data/processed/ratings_asof/{a.slate_date}")
    base = REPO / "results/a3_seed_wiring" / a.slate_date
    t0 = time.time()
    seeds = list(range(a.seeds))
    gd, pdy, rosd, rdaily = daily_arm(a.slate_date, a.season, now, rd, a.seeds, base / "daily", "auto")
    print(f"[{time.time()-t0:5.0f}s] daily done {rdaily}", flush=True)
    ga, pa, rosa, _ = daily_arm(a.slate_date, a.season, now, rd, a.seeds, base / "anon", None)
    print(f"[{time.time()-t0:5.0f}s] anon done", flush=True)
    gh, ph, rosh, dh = harness_arm(a.slate_date, a.season, now, rd, seeds, base / "harness", False)
    print(f"[{time.time()-t0:5.0f}s] harness done", flush=True)
    gz, pz, _, _ = harness_arm(a.slate_date, a.season, now, rd, seeds, base / "harness_lut0", True)
    print(f"[{time.time()-t0:5.0f}s] harness_lut0 done", flush=True)

    # ---- slot types (daily roster) --------------------------------------------------------------------------------------
    T = DP.tables(a.season, True, str(REPO / f"data/raw/cbbd/rosters/roster_{a.season}.parquet"), "onfloor",
                  need_prev_roster=True, serving=True)
    fb = set(int(t) for t in (dh.get("d1p_fallback_teams") or []))
    slate = pd.read_parquet(Path(rdaily["out"]) / "slate.parquet")
    rows = []
    for i, g in enumerate(slate.itertuples()):
        for side, (tid, nm, opp) in enumerate(((g.home_team_id, g.home_name, g.away_name), (g.away_team_id, g.away_name, g.home_name))):
            prev = {p for p, _ in T["by_team"].get(int(tid), [])}
            src = "espn_2027" if int(tid) in T["roster"] else ("R1" if int(tid) in fb else "none")
            for s in range(rosd.shape[2]):
                pid = int(rosd[i, side, s])
                if pid <= 0:
                    typ = "anonymous"
                elif int(tid) in fb:
                    typ = "R1_fallback"
                elif pid in prev:
                    typ = "returner"
                else:
                    typ = "transfer_in"
                rows.append(dict(game_id=int(g.game_id), team_id=int(tid), team=nm, opp=opp, side=side, slot=s, pid=pid, slot_type=typ,
                                 roster_source=src, harness_pid=int(rosh[i, side, s]), anon_pid=int(rosa[i, side, s])))
    S = pd.DataFrame(rows)
    S.to_parquet(base / "slots.parquet", index=False)

    # ---- minutes: named minutes from players (named = cbbd_id > 0), team minutes from the games frame -----------------
    tm = gd[["game_id", "seed"]].assign(team_min=team_minutes(gd).to_numpy())
    nm_d = pdy[pdy["cbbd_id"] > 0].groupby(["game_id", "seed", "team_id"])["minutes"].sum().rename("named").reset_index()
    tg = S[["game_id", "team_id"]].drop_duplicates()
    allk = tg.merge(pd.DataFrame({"seed": seeds}), how="cross").merge(tm, on=["game_id", "seed"])
    nm_d = allk.merge(nm_d, on=["game_id", "seed", "team_id"], how="left").fillna({"named": 0.0})

    gcols = [c for c in gd.columns if c in gh.columns and c not in ("created_at", "tip_time_is_placeholder", "tip_source",
                                                                     "pre_tip_basis", "pre_tip_verified")]
    srt = ["game_id", "seed"]
    games_equal = gd[gcols].sort_values(srt).reset_index(drop=True).equals(gh[gcols].sort_values(srt).reset_index(drop=True))

    def tot(g):
        return (g["home_pts"] + g["away_pts"]).groupby(g["game_id"]).mean()
    arms = {"daily": gd, "anon": ga, "harness": gh, "harness_lut0": gz}
    summ = {"slate_date": a.slate_date, "clock": str(now), "seeds": seeds, "n_games": int(gd["game_id"].nunique()),
            "daily_stage_return": rdaily,
            "slot_mismatch_daily_vs_harness": int((S["pid"] != S["harness_pid"]).sum()),
            "games_frame_equal_daily_vs_harness": bool(games_equal),
            "daily_vs_harness": cmp_players(pdy, ph), "daily_vs_harness_lut0": cmp_players(pdy, pz),
            "game_total_mean": {k: round(float(tot(g).mean()), 3) for k, g in arms.items()},
            "game_margin_mean": {k: round(float((g["home_pts"] - g["away_pts"]).mean()), 3) for k, g in arms.items()},
            "per_game_total_absdiff_mean": {"daily_vs_anon": float((tot(gd) - tot(ga)).abs().mean()),
                                            "harness_lut0_vs_harness": float((tot(gz) - tot(gh)).abs().mean())},
            "anon_slot_share": {"daily": float((S["pid"] <= 0).mean()), "anon_arm": float((S["anon_pid"] <= 0).mean()),
                                "harness": float((S["harness_pid"] <= 0).mean())},
            "anon_minutes_share_daily": float(1 - nm_d["named"].sum() / nm_d["team_min"].sum()),
            "anon_player_rows_anon_arm": int(len(pa)),
            "slot_type_counts": S["slot_type"].value_counts().to_dict(),
            "slot_type_by_roster_source": S.groupby(["roster_source", "slot_type"]).size().unstack(fill_value=0).to_dict(orient="index"),
            "anon_slot_share_by_roster_source": S.groupby("roster_source")["pid"].apply(lambda x: float((x <= 0).mean())).to_dict(),
            "teams_by_roster_source": S.drop_duplicates(["team_id"]).groupby("roster_source").size().to_dict(),
            "fallback_line": V2.fallback_line(dh.get("d1p_fallback_teams")),
            "harness_d1p_team_games": dh.get("d1p_team_games"), "harness_d1p_slots": dh.get("d1p_slots")}
    # per team (daily)
    per = S.groupby(["team_id", "team", "roster_source"]).agg(
        anon_slot_share=("pid", lambda x: float((x <= 0).mean())), named_slots=("pid", lambda x: int((x > 0).sum())),
        returner=("slot_type", lambda x: int((x == "returner").sum())), transfer_in=("slot_type", lambda x: int((x == "transfer_in").sum())),
        r1=("slot_type", lambda x: int((x == "R1_fallback").sum()))).reset_index()
    tmn = nm_d.groupby("team_id").apply(lambda x: 1 - x["named"].sum() / x["team_min"].sum()).rename("anon_minutes_share").reset_index()
    per = per.merge(tmn, on="team_id", how="left")
    per.to_csv(base / "per_team.csv", index=False)
    summ["per_team_anon_slot_share_quantiles"] = per["anon_slot_share"].quantile([0, .1, .25, .5, .75, .9, 1]).round(4).to_dict()
    summ["per_team_by_source"] = per.groupby("roster_source")[["anon_slot_share", "anon_minutes_share", "named_slots"]].mean().round(4).to_dict(orient="index")
    summ["teams_100pct_anon"] = per.loc[per["anon_slot_share"] >= 1.0, ["team_id", "team", "roster_source"]].to_dict(orient="records")
    (base / "summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=str)[:8000])


if __name__ == "__main__":
    main()
