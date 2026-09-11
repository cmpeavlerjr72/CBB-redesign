"""
diag_engine_multilevel.py -- the multi-level evidence CLAUDE.md requires next
to any engine gate read.

`CLAUDE.md`, standing rule "multi-level evidence": *"Every finding and every fix
shows overall, per-game, per-team, per-possession-type, and per-player (when
relevant) evidence. Aggregate-only proof is insufficient."* And the standing
rule "matchup-specific, not league-average": *"predictions bucketed by team or
player prior quintile must slope with actuals, not sit flat at the mean."*

`scripts/eval_gates.py` answers the pre-registered G1-G9 questions. This script
answers the three the gate report does not, on the same results file:

  1. OVERALL and PER-GAME: possessions, PPP, total, margin against the season
     truth, with the per-game MAE and bias, not only the slate mean. This is
     the offsetting-error check -- a total that lands while possessions run
     high and PPP runs low is a FAIL that a total-only read would call a PASS.
  2. PER-TEAM QUINTILE: teams bucketed by their own actual season quality, then
     predicted vs actual per bucket, with the slope ratio. Flat is a failure
     even when the mean is right.
  3. PER-POSSESSION-TYPE: the shot/turnover/foul mix per possession, which is
     what the L3 sub-models own, separated from the possession COUNT, which is
     what the clock model owns. The point of separating them is that the two
     have different owners and a defect in one must not be reported as a
     defect in the other.

Nothing here adjusts anything; it only reports.

Usage:
    .venv/Scripts/python.exe scripts/diag_engine_multilevel.py \
        --results results/engine_v0/<tag> --season 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.eval import reference as REF  # noqa: E402


def per_game(sim: pd.DataFrame) -> pd.DataFrame:
    """Collapse the seeds: one row per game, the engine's own prediction."""
    sim = sim.copy()
    sim["margin"] = sim["home_pts"] - sim["away_pts"]
    sim["total"] = sim["home_pts"] + sim["away_pts"]
    sim["fga"] = sim[["home_fga3", "away_fga3", "home_fga2_rim", "away_fga2_rim",
                      "home_fga2_jump", "away_fga2_jump"]].sum(axis=1)
    sim["fga3"] = sim[["home_fga3", "away_fga3"]].sum(axis=1)
    sim["fga_rim"] = sim[["home_fga2_rim", "away_fga2_rim"]].sum(axis=1)
    sim["fta"] = sim[["home_fta", "away_fta"]].sum(axis=1)
    sim["tov"] = sim[["home_tov", "away_tov"]].sum(axis=1)
    sim["oreb"] = sim[["home_oreb", "away_oreb"]].sum(axis=1)
    sim["dreb"] = sim[["home_dreb", "away_dreb"]].sum(axis=1)
    sim["win"] = (sim["margin"] > 0).astype(float)
    agg = {c: "mean" for c in ("margin", "total", "possessions", "n_periods", "win",
                               "fga", "fga3", "fga_rim", "fta", "tov", "oreb", "dreb")}
    out = sim.groupby("game_id").agg(agg)
    out["n_seeds"] = sim.groupby("game_id").size()
    out["ot_rate"] = sim.assign(ot=(sim["n_periods"] > 2).astype(float)) \
                        .groupby("game_id")["ot"].mean()
    return out.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--label", default="PROVISIONAL")
    a = ap.parse_args()

    res = Path(a.results)
    sim = pd.read_parquet(res / "games.parquet")
    if not len(sim):
        raise SystemExit(f"{res}/games.parquet is empty -- the run wrote no complete seed")
    meta = json.loads((res / "run_meta.json").read_text(encoding="utf-8"))
    pg = per_game(sim)

    # `load_actual_games` carries margin/total/n_periods under those exact
    # names, and `per_game` produces the same three for the sim, so the merge
    # is explicit rather than suffix-guessed: a silently-swapped sim/actual
    # column is the one mistake this whole report exists to avoid.
    act = REF.load_actual_games(a.season)[
        ["game_id", "home_team_id", "away_team_id", "neutral",
         "margin", "total", "n_periods", "went_ot"]].rename(columns={
            "margin": "act_margin", "total": "act_total",
            "n_periods": "act_n_periods", "went_ot": "act_went_ot"})
    ap_ = REF.load_actual_possessions(a.season).rename(columns={"game_poss": "act_poss"})
    m = pg.merge(act, on="game_id", how="inner").merge(ap_, on="game_id", how="left")
    acol = {"margin": "act_margin", "total": "act_total", "poss": "act_poss"}

    print("=" * 78)
    print(f"MULTI-LEVEL BREAKDOWN [{a.label}]  {res}")
    print(f"  seeds={meta.get('n_seeds')}  games={m.shape[0]}  "
          f"flags={ {k: v for k, v in meta.get('adapter_flags', {}).items() if k != 'sources'} }")
    print("=" * 78)

    # ---- 1. overall + per game -------------------------------------------
    print("\n1. OVERALL AND PER-GAME (engine vs verified finals)")
    print(f"{'quantity':<26}{'engine':>10}{'actual':>10}{'delta':>10}{'per-game MAE':>14}")
    rows = []
    for name, s_col, a_col in (("margin", "margin", acol["margin"]),
                               ("total", "total", acol["total"])):
        s, av = m[s_col].to_numpy(float), m[a_col].to_numpy(float)
        ok = np.isfinite(s) & np.isfinite(av)
        rows.append((name, s[ok].mean(), av[ok].mean(), s[ok].mean() - av[ok].mean(),
                     np.abs(s[ok] - av[ok]).mean()))
    if "poss" in acol:
        s = m["possessions"].to_numpy(float)
        av = m[acol["poss"]].to_numpy(float)
        ok = np.isfinite(s) & np.isfinite(av)
        rows.append(("possessions/game", s[ok].mean(), av[ok].mean(),
                     s[ok].mean() - av[ok].mean(), np.abs(s[ok] - av[ok]).mean()))
        ppp_s = m.loc[ok, "total"].to_numpy() / (2 * s[ok])
        ppp_a = m.loc[ok, acol["total"]].to_numpy() / (2 * av[ok])
        rows.append(("points per possession", ppp_s.mean(), ppp_a.mean(),
                     ppp_s.mean() - ppp_a.mean(), np.abs(ppp_s - ppp_a).mean()))
    for n_, s_, a_, d_, e_ in rows:
        print(f"{n_:<26}{s_:>10.4f}{a_:>10.4f}{d_:>+10.4f}{e_:>14.4f}")
    print(f"{'SD of margin':<26}{m['margin'].std():>10.4f}"
          f"{m[acol['margin']].std():>10.4f}"
          f"{m['margin'].std() - m[acol['margin']].std():>+10.4f}{'--':>14}")
    print(f"{'SD of total':<26}{m['total'].std():>10.4f}"
          f"{m[acol['total']].std():>10.4f}"
          f"{m['total'].std() - m[acol['total']].std():>+10.4f}{'--':>14}")
    print(f"{'OT rate':<26}{m['ot_rate'].mean():>10.4f}"
          f"{m['act_went_ot'].mean():>10.4f}"
          f"{m['ot_rate'].mean() - m['act_went_ot'].mean():>+10.4f}{'--':>14}")

    # ---- 2. per-team quintile --------------------------------------------
    print("\n2. PER-TEAM QUINTILE (the matchup-specific rule; flat = FAIL)")
    long = pd.concat([
        pd.DataFrame({"team_id": m["home_team_id"], "sm": m["margin"],
                      "am": m[acol["margin"]]}),
        pd.DataFrame({"team_id": m["away_team_id"], "sm": -m["margin"],
                      "am": -m[acol["margin"]]}),
    ], ignore_index=True)
    tm = long.groupby("team_id").agg(sim_margin=("sm", "mean"), act_margin=("am", "mean"),
                                     n=("sm", "size"))
    tm = tm[tm["n"] >= 5]
    tm["q"] = pd.qcut(tm["act_margin"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    g_ = tm.groupby("q", observed=True)
    qt = pd.DataFrame({"teams": g_.size(), "engine": g_["sim_margin"].mean(),
                       "actual": g_["act_margin"].mean()}).sort_index()
    print(f"{'quintile':<10}{'teams':>7}{'engine avg margin':>20}{'actual avg margin':>20}")
    for q, r in qt.iterrows():
        print(f"{int(q):<10}{int(r['teams']):>7}{r['engine']:>20.3f}{r['actual']:>20.3f}")
    sp_e = qt["engine"].iloc[-1] - qt["engine"].iloc[0]
    sp_a = qt["actual"].iloc[-1] - qt["actual"].iloc[0]
    steps = int(np.sum(np.sign(np.diff(qt["engine"])) == np.sign(np.diff(qt["actual"]))))
    print(f"  span engine {sp_e:+.3f} vs actual {sp_a:+.3f}; SLOPE RATIO "
          f"{sp_e / sp_a if sp_a else float('nan'):.4f}; monotone in {steps}/4 steps "
          f"(gate: >= 3, and slope ratio far from 0)")

    # ---- 3. per-possession type ------------------------------------------
    print("\n3. PER-POSSESSION-TYPE MIX (owned by the L3 sub-models, not the clock)")
    tb = None
    try:
        tb = REF.load_actual_team_box(a.season)
    except Exception as e:                                        # noqa: BLE001
        print(f"  (actual team box unavailable: {e})")
    poss2 = 2 * m["possessions"]
    mix = {
        "3PA share of FGA": (m["fga3"] / m["fga"]).mean(),
        "rim share of FGA": (m["fga_rim"] / m["fga"]).mean(),
        "FTA / FGA": (m["fta"] / m["fga"]).mean(),
        "TOV per possession": (m["tov"] / poss2).mean(),
        "FGA per possession": (m["fga"] / poss2).mean(),
        "OREB%": (m["oreb"] / (m["oreb"] + m["dreb"])).mean(),
    }
    act_mix = {}
    if tb is not None and len(tb):
        tb = tb[tb["game_id"].isin(m["game_id"])]
        fga, tpa, fta = tb["fga"].sum(), tb["tpa"].sum(), tb["fta"].sum()
        act_mix["3PA share of FGA"] = float(tpa / fga)
        act_mix["FTA / FGA"] = float(fta / fga)
        act_mix["OREB%"] = float(tb["oreb"].sum() / (tb["oreb"].sum() + tb["dreb"].sum()))
        act_mix["TOV per possession"] = float(tb["tov"].sum() / tb["poss_team"].sum())
        act_mix["FGA per possession"] = float(fga / tb["poss_team"].sum())
    print(f"{'quantity':<26}{'engine':>10}{'actual':>10}{'delta':>10}")
    for k, v in mix.items():
        av = act_mix.get(k)
        d = f"{v - av:+10.4f}" if av is not None else f"{'--':>10}"
        avs = f"{av:>10.4f}" if av is not None else f"{'--':>10}"
        print(f"{k:<26}{v:>10.4f}{avs}{d}")
    print("\nRead 1 and 3 together: if every rate in 3 lands and possessions/game in 1")
    print("does not, the defect is the possession COUNT (the clock model), not the mix.")
    print(f"\n[{a.label}] -- clock, rotation and usage are unadopted reference arms and the")
    print("seed count is whatever this run holds; see docs/tests/engine_seed_count_*.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
