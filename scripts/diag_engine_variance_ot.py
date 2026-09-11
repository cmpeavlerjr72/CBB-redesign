"""diag_engine_variance_ot.py -- variance and overtime diagnostic for an
engine_v0 results directory.

WHAT THIS ANSWERS, AND WHY IT IS SEPARATE FROM `eval_gates.py`.
`eval_gates.py` reports that G1's possession SD is short, that G5's total SD
ratio is 0.79 while the margin SD ratio is 1.03, and that G7's OT rate is half
the actual. Those are three symptoms. This script asks which VARIANCE COMPONENT
is missing, because a score is a product

    team points  =  possessions  x  points-per-possession

and a total and a margin are the sum and difference of two such products that
share one possession count. A shortfall in Var(total) can come from Var(P),
from Var(PPP), or from the covariance between the two teams' PPP, and those
three have three different owners (the clock model, the L3 event/shot models,
and nothing at all -- the engine has no sub-model that couples the two teams'
efficiencies).

Nothing here adjusts anything; it only reports. No fixes, per the lane brief.

The three questions, stated as they were pre-registered:

  Q1  Is the possession SD shortfall a BETWEEN-GAME pace-level variance
      shortfall (the one-pace-realisation draw is too tight across games) or a
      WITHIN-GAME duration variance shortfall (the draw is right across games
      but each game's realisations are too similar)?
      Method: the sim has many seeds per game, so its possession variance
      splits exactly into between-game and within-game parts. The actual has
      ONE realisation per game, so its two parts are not separately observed --
      but they ARE identified against the sim's own per-game mean by regressing
      the actual on the prediction. A slope > 1 means the sim's between-game
      spread is shrunk; a residual SD larger than the sim's within-game SD
      means the within-game draw is too tight. Both are reported; neither is
      assumed.

  Q2  Is the home/away score correlation near zero because possessions barely
      vary between games (the shared pace channel is too weak) or because the
      two teams' PPP are uncorrelated / anti-correlated in the sim?
      Method: exact covariance decomposition of Cov(home_pts, away_pts) into
      the shared-possession channel and the PPP-covariance channel, for sim
      and for actual, plus two counterfactuals -- the Var(P) that would be
      needed to reach the actual correlation with the sim's PPP covariance
      held at its measured value, and the PPP correlation that would be needed
      at the sim's measured Var(P).

  Q3  Is the OT shortfall a TIE-RATE shortfall (a regulation-margin
      distribution question) or an OT-HANDLING defect (the engine mishandles a
      game once it is tied)?
      Method: a game goes to OT iff the regulation margin is 0, so the
      regulation margin is RECOVERABLE from what the contract already writes --
      it is 0 when n_periods > 2 and the final margin otherwise, on both the
      sim and the actual side. The regulation-margin density near zero is
      therefore directly comparable. OT handling is then graded separately on
      the conditional distribution of n_periods GIVEN OT, which no tie-rate
      defect can touch.

Usage:
    .venv/Scripts/python.exe scripts/diag_engine_variance_ot.py \
        --results results/engine_v0/<tag> --season 2025 \
        --out results/engine_v0/<tag>/diag_variance_ot.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.eval import reference as REF  # noqa: E402

MIN_CELL = 30  # below this a per-team / per-month cell is labelled UNDERPOWERED

OUT: list[str] = []


def say(s: str = "") -> None:
    OUT.append(s)
    print(s)


def _v(x: np.ndarray) -> float:
    """Population variance (ddof=0): every quantity here is a full enumeration
    of the rows that exist, not a sample from a wider pool."""
    return float(np.var(np.asarray(x, dtype=np.float64)))


def _c(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def product_decomp(p: np.ndarray, r: np.ndarray, label: str) -> dict:
    """Var(P * R) split into the three delta-method channels plus the exact
    remainder. P is the possession count, R the per-possession rate (sum or
    difference of the two teams' PPP).

        Var(PR) ~= E[R]^2 Var(P) + E[P]^2 Var(R) + 2 E[P] E[R] Cov(P,R) + rem

    `rem` is the exact residual (it collects the fourth-moment terms), reported
    rather than dropped so the three channels cannot silently fail to add up.
    """
    p = np.asarray(p, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    ep, er = p.mean(), r.mean()
    vp, vr, cpr = _v(p), _v(r), _c(p, r)
    total = _v(p * r)
    a = er * er * vp
    b = ep * ep * vr
    c = 2.0 * ep * er * cpr
    return {"label": label, "n": len(p), "E[P]": ep, "E[R]": er,
            "Var(P)": vp, "Var(R)": vr, "Cov(P,R)": cpr,
            "Var(PR)": total, "SD(PR)": float(np.sqrt(total)),
            "chan_P": a, "chan_R": b, "chan_cov": c, "rem": total - (a + b + c)}


def decomp_rows(d: dict) -> str:
    t = d["Var(PR)"]
    return (f"  {d['label']:<22s} n={d['n']:>7d}  Var={t:10.3f}  SD={d['SD(PR)']:8.4f}\n"
            f"      E[P]={d['E[P]']:8.4f}  Var(P)={d['Var(P)']:9.4f}   "
            f"E[R]={d['E[R]']:8.5f}  Var(R)={d['Var(R)']:9.6f}  Cov(P,R)={d['Cov(P,R)']:9.5f}\n"
            f"      channel P   {d['chan_P']:10.3f} ({100*d['chan_P']/t:6.1f}%)   "
            f"channel R   {d['chan_R']:10.3f} ({100*d['chan_R']/t:6.1f}%)   "
            f"channel cov {d['chan_cov']:10.3f} ({100*d['chan_cov']/t:6.1f}%)   "
            f"rem {d['rem']:8.3f}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seed", type=int, default=0,
                    help="keep only seeds < this (subset a larger run; 0 = all)")
    args = ap.parse_args()

    res = Path(args.results)
    sim = pd.read_parquet(res / "games.parquet")
    if args.max_seed:
        sim = sim[sim["seed"] < args.max_seed].reset_index(drop=True)
    n_seeds = sim["seed"].nunique()

    ag = REF.load_actual_games(args.season)
    fin = REF.load_game_finals_truth(args.season)
    if fin is not None:
        # CLAUDE.md: grade against VERIFIED finals, never pbp-accumulated totals.
        f = fin[["game_id", "home_score", "away_score", "n_periods"]].rename(
            columns={"home_score": "h_act", "away_score": "a_act",
                     "n_periods": "np_act"})
        ag = ag.drop(columns=["home_score", "away_score", "n_periods"]).merge(
            f, on="game_id", how="inner")
        truth_src = "game_finals_v2 (verified, two-source)"
    else:
        ag = ag.rename(columns={"home_score": "h_act", "away_score": "a_act",
                                "n_periods": "np_act"})
        truth_src = "universe schedule (NO verified finals table found)"
    ag["m_act"] = ag["h_act"] - ag["a_act"]
    ag["t_act"] = ag["h_act"] + ag["a_act"]

    poss = REF.load_actual_possessions(args.season)
    ag = ag.merge(poss, on="game_id", how="left")

    tb = REF.load_actual_team_box(args.season)

    sim = sim[sim["game_id"].isin(set(ag["game_id"]))].reset_index(drop=True)
    sim["margin"] = sim["home_pts"].astype(float) - sim["away_pts"].astype(float)
    sim["total"] = sim["home_pts"].astype(float) + sim["away_pts"].astype(float)
    sim["P"] = sim["possessions"].astype(float)
    sim["ppp_h"] = sim["home_pts"] / sim["P"]
    sim["ppp_a"] = sim["away_pts"] / sim["P"]
    sim["ppp_sum"] = sim["ppp_h"] + sim["ppp_a"]
    sim["ppp_dif"] = sim["ppp_h"] - sim["ppp_a"]
    sim["ot"] = (sim["n_periods"] > 2).astype(float)
    # A game goes to OT iff regulation ended level, so the regulation margin is
    # 0 on every OT row and the final margin on every other row -- on BOTH
    # sides. This is what makes the tie rate and the OT rate the same number
    # and lets the regulation-margin density be compared directly.
    sim["reg_margin"] = np.where(sim["n_periods"] > 2, 0.0, sim["margin"])

    say("=" * 100)
    say(f"ENGINE VARIANCE / OVERTIME DIAGNOSTIC -- {res}")
    say(f"season {args.season}   seeds={n_seeds}   sim rows={len(sim):,}   "
        f"games={sim['game_id'].nunique():,}")
    say(f"truth: {truth_src}; possessions truth = box estimate "
        f"(FGA - OREB + TOV + 0.44*FTA, averaged over both teams)")
    say("population variance (ddof=0) throughout; every table is a full enumeration")
    say("=" * 100)

    have_p = ag["game_poss"].notna()
    agp = ag[have_p].reset_index(drop=True)
    act_ppp_sum = agp["t_act"] / agp["game_poss"]
    act_ppp_dif = agp["m_act"] / agp["game_poss"]

    # =====================================================================
    say("")
    say("## A. TOTAL and MARGIN variance, decomposed into possessions x PPP")
    say("")
    say("Var(P*R) = E[R]^2 Var(P) + E[P]^2 Var(R) + 2 E[P] E[R] Cov(P,R) + remainder")
    say("R = ppp_home + ppp_away for the total, ppp_home - ppp_away for the margin.")
    say("")
    dt_sim = product_decomp(sim["P"].to_numpy(), sim["ppp_sum"].to_numpy(), "TOTAL sim")
    dt_act = product_decomp(agp["game_poss"].to_numpy(), act_ppp_sum.to_numpy(), "TOTAL actual")
    dm_sim = product_decomp(sim["P"].to_numpy(), sim["ppp_dif"].to_numpy(), "MARGIN sim")
    dm_act = product_decomp(agp["game_poss"].to_numpy(), act_ppp_dif.to_numpy(), "MARGIN actual")
    for d in (dt_sim, dt_act, dm_sim, dm_act):
        say(decomp_rows(d))
        say("")

    say(f"  TOTAL  SD ratio sim/actual = {dt_sim['SD(PR)'] / dt_act['SD(PR)']:.4f}")
    say(f"  MARGIN SD ratio sim/actual = {dm_sim['SD(PR)'] / dm_act['SD(PR)']:.4f}")
    say("")
    say("  channel-by-channel ratio sim/actual (the diagnosis, not the symptom):")
    for nm, ds, da in (("TOTAL", dt_sim, dt_act), ("MARGIN", dm_sim, dm_act)):
        say(f"    {nm:<7s} Var(P) {ds['Var(P)']/da['Var(P)']:7.4f}   "
            f"Var(R) {ds['Var(R)']/da['Var(R)']:7.4f}   "
            f"chanP {ds['chan_P']/da['chan_P']:7.4f}   "
            f"chanR {ds['chan_R']/da['chan_R']:7.4f}")
    say("")

    # The algebraic identity that ties the two gates together.
    vh_s, va_s = _v(sim["ppp_h"]), _v(sim["ppp_a"])
    ch_s = _c(sim["ppp_h"], sim["ppp_a"])
    act_h_ppp = agp["h_act"] / agp["game_poss"]
    act_a_ppp = agp["a_act"] / agp["game_poss"]
    vh_a, va_a = _v(act_h_ppp), _v(act_a_ppp)
    ch_a = _c(act_h_ppp, act_a_ppp)
    say("  Var(ppp_sum) = Var(h) + Var(a) + 2Cov ;  Var(ppp_dif) = Var(h) + Var(a) - 2Cov")
    say(f"    sim     Var(ppp_h)={vh_s:.6f}  Var(ppp_a)={va_s:.6f}  "
        f"Cov={ch_s:+.6f}  corr={ch_s/np.sqrt(vh_s*va_s):+.4f}")
    say(f"    actual  Var(ppp_h)={vh_a:.6f}  Var(ppp_a)={va_a:.6f}  "
        f"Cov={ch_a:+.6f}  corr={ch_a/np.sqrt(vh_a*va_a):+.4f}")
    say("")

    # =====================================================================
    say("")
    say("## B. Q1 -- possession variance: between-game vs within-game")
    say("")
    gm = sim.groupby("game_id").agg(P_mean=("P", "mean"), P_var=("P", "var"),
                                    n=("P", "size"))
    between = _v(gm["P_mean"].to_numpy())
    within = float(np.nanmean(gm["P_var"].to_numpy()))  # unbiased within-game var
    tot = _v(sim["P"].to_numpy())
    say(f"  sim possessions           mean {sim['P'].mean():8.4f}   "
        f"SD(pooled) {np.sqrt(tot):7.4f}   Var {tot:8.4f}")
    say(f"    between games (Var of the per-game mean)   {between:8.4f}   "
        f"SD {np.sqrt(between):7.4f}   ({100*between/tot:5.1f}% of pooled)")
    say(f"    within  game  (mean across-seed Var)       {within:8.4f}   "
        f"SD {np.sqrt(within):7.4f}   ({100*within/tot:5.1f}% of pooled)")
    va = _v(agp["game_poss"].to_numpy())
    say(f"  actual possessions        mean {agp['game_poss'].mean():8.4f}   "
        f"SD {np.sqrt(va):7.4f}   Var {va:8.4f}")
    say("")

    j = gm.reset_index().merge(agp[["game_id", "game_poss", "month"]], on="game_id", how="inner")
    x = j["P_mean"].to_numpy()
    y = j["game_poss"].to_numpy()
    vx, cxy = _v(x), _c(x, y)
    slope = cxy / vx
    r = cxy / np.sqrt(vx * _v(y))
    resid = y - (y.mean() + slope * (x - x.mean()))
    say("  Calibration of the per-game pace PREDICTION (regress actual on the sim's")
    say("  per-game mean; slope > 1 means the sim's between-game spread is shrunk):")
    say(f"    n games {len(j):,}   slope {slope:.4f}   corr {r:.4f}   "
        f"Var(pred) {vx:.4f}   Var(actual) {_v(y):.4f}")
    say(f"    SD of the actual around the prediction (irreducible within-game) "
        f"{np.sqrt(_v(resid)):.4f}")
    say(f"    sim's own within-game SD                                          "
        f"{np.sqrt(within):.4f}")
    say(f"    -> within-game variance the sim is MISSING: "
        f"{_v(resid) - within:+.4f}  "
        f"(SD {np.sqrt(max(_v(resid),0)):.4f} needed vs {np.sqrt(within):.4f} produced)")
    say(f"    -> between-game variance the sim is MISSING (Var to reach slope 1): "
        f"{cxy - vx:+.4f}  (Var(pred) {vx:.4f} -> {cxy:.4f} needed)")
    say("")
    need_between = cxy          # Var(pred) that makes the slope 1
    need_within = _v(resid)
    say(f"  Reconstruction: a perfectly calibrated engine would show pooled "
        f"possession Var = {need_between + need_within:.4f} "
        f"(SD {np.sqrt(need_between + need_within):.4f}); "
        f"actual Var {va:.4f} (SD {np.sqrt(va):.4f}); sim {tot:.4f} (SD {np.sqrt(tot):.4f}).")
    say(f"  Share of the MISSING pooled possession variance "
        f"({va - tot:+.4f}) attributable to")
    miss = va - tot
    if abs(miss) > 1e-9:
        say(f"    between-game (pace level):  {100*(need_between - between)/miss:6.1f}%")
        say(f"    within-game  (duration)  :  {100*(need_within - within)/miss:6.1f}%")
        say(f"    (the two need not sum to 100%: the actual's own split is not "
            f"separately observed; the reconstruction is the identified one)")
    say("")

    # =====================================================================
    say("")
    say("## C. Q2 -- home/away score covariance: which channel is missing")
    say("")
    cov_s = _c(sim["home_pts"].astype(float), sim["away_pts"].astype(float))
    corr_s = cov_s / np.sqrt(_v(sim["home_pts"].astype(float)) * _v(sim["away_pts"].astype(float)))
    cov_a = _c(agp["h_act"].astype(float), agp["a_act"].astype(float))
    corr_a = cov_a / np.sqrt(_v(agp["h_act"].astype(float)) * _v(agp["a_act"].astype(float)))
    say(f"  corr(home_pts, away_pts)   sim {corr_s:+.4f}   actual {corr_a:+.4f}   "
        f"(gate: actual {corr_a:+.4f} +/- 0.05)")
    say(f"  cov (home_pts, away_pts)   sim {cov_s:+9.4f}   actual {cov_a:+9.4f}")
    say("")
    say("  Cov(P*h, P*a) ~= E[h]E[a] Var(P) + E[P]^2 Cov(h,a) + E[P](E[a]Cov(P,h)+E[h]Cov(P,a))")
    for nm, P, h, a in (("sim", sim["P"].to_numpy(), sim["ppp_h"].to_numpy(), sim["ppp_a"].to_numpy()),
                        ("actual", agp["game_poss"].to_numpy(),
                         act_h_ppp.to_numpy(), act_a_ppp.to_numpy())):
        eP, eh, ea = P.mean(), h.mean(), a.mean()
        k_pace = eh * ea * _v(P)
        k_ppp = eP * eP * _c(h, a)
        k_x = eP * (ea * _c(P, h) + eh * _c(P, a))
        exact = _c(P * h, P * a)
        if nm == "actual":
            act_pace_channel = k_pace
        say(f"    {nm:<7s} exact {exact:+9.4f} = pace {k_pace:+8.4f} + "
            f"ppp-cov {k_ppp:+8.4f} + cross {k_x:+8.4f} + rem {exact-k_pace-k_ppp-k_x:+7.4f}")
    say("")

    # counterfactuals
    eP, eh, ea = sim["P"].mean(), sim["ppp_h"].mean(), sim["ppp_a"].mean()
    sd_h = np.sqrt(_v(sim["home_pts"].astype(float)))
    sd_a = np.sqrt(_v(sim["away_pts"].astype(float)))
    need_cov = corr_a * sd_h * sd_a
    gap = need_cov - cov_s
    say(f"  To reach the actual correlation {corr_a:+.4f} the sim needs "
        f"Cov(home,away) = {need_cov:+.4f}; it has {cov_s:+.4f}; gap {gap:+.4f}.")
    say(f"    (a) pace channel only, PPP covariance held at its measured value:")
    vP_sim = _v(sim["P"].to_numpy())
    say(f"        Var(P) would have to rise by {gap/(eh*ea):+.3f} to "
        f"{vP_sim + gap/(eh*ea):.3f}  (SD {np.sqrt(vP_sim):.3f} -> "
        f"{np.sqrt(max(vP_sim + gap/(eh*ea), 0)):.3f})")
    say(f"        -- actual possession Var is only {va:.3f} (SD {np.sqrt(va):.3f}), so "
        f"{'THE PACE CHANNEL ALONE CANNOT CLOSE IT' if _v(sim['P'])+gap/(eh*ea) > va else 'the pace channel alone could close it'}.")
    say(f"    (b) PPP-covariance channel only, Var(P) held at its measured value:")
    dc = gap / (eP * eP)
    say(f"        Cov(ppp_h, ppp_a) would have to rise by {dc:+.6f} to "
        f"{ch_s + dc:+.6f}  (corr {ch_s/np.sqrt(vh_s*va_s):+.4f} -> "
        f"{(ch_s+dc)/np.sqrt(vh_s*va_s):+.4f})")
    say(f"        -- actual PPP corr is {ch_a/np.sqrt(vh_a*va_a):+.4f}.")
    say("")

    # between/within split of the home-away correlation in the sim
    g2 = sim.groupby("game_id").agg(h=("home_pts", "mean"), a=("away_pts", "mean"))
    corr_between = _c(g2["h"].to_numpy(), g2["a"].to_numpy()) / np.sqrt(_v(g2["h"]) * _v(g2["a"]))
    dev = sim.merge(g2.rename(columns={"h": "hm", "a": "am"}), on="game_id")
    dh = dev["home_pts"] - dev["hm"]
    da = dev["away_pts"] - dev["am"]
    corr_within = _c(dh.to_numpy(), da.to_numpy()) / np.sqrt(_v(dh) * _v(da))
    say(f"  sim corr(home,away) BETWEEN games (per-game means) {corr_between:+.4f}   "
        f"WITHIN game (seed deviations) {corr_within:+.4f}")
    say(f"  sim corr(ppp_h, ppp_a) WITHIN game "
        f"{_c((sim['ppp_h'] - sim.groupby('game_id')['ppp_h'].transform('mean')).to_numpy(), (sim['ppp_a'] - sim.groupby('game_id')['ppp_a'].transform('mean')).to_numpy()) / np.sqrt(_v(sim['ppp_h'] - sim.groupby('game_id')['ppp_h'].transform('mean')) * _v(sim['ppp_a'] - sim.groupby('game_id')['ppp_a'].transform('mean'))):+.4f}")
    say("  (BETWEEN is what team-strength matchup produces; WITHIN is what the")
    say("   engine's own draw produces inside one fixed matchup. The actual has")
    say("   exactly one realisation per game, so its pooled correlation contains")
    say("   both -- which is why the sim's pooled number must be compared to the")
    say("   actual's pooled number and the split read as mechanism, not as a gate.)")
    say("")

    # =====================================================================
    say("")
    say("## D. Per-team evidence (matchup-specific rule)")
    say("")
    long = pd.concat([
        sim[["game_id", "seed", "P"]].assign(pts=sim["home_pts"].astype(float), side="home")
           .merge(ag[["game_id", "home_team_id"]].rename(columns={"home_team_id": "team_id"}), on="game_id"),
        sim[["game_id", "seed", "P"]].assign(pts=sim["away_pts"].astype(float), side="away")
           .merge(ag[["game_id", "away_team_id"]].rename(columns={"away_team_id": "team_id"}), on="game_id"),
    ], ignore_index=True)
    long["ppp"] = long["pts"] / long["P"]
    simt = long.groupby("team_id").agg(n_gs=("game_id", "nunique"),
                                       P_sd=("P", "std"), ppp_sd=("ppp", "std"),
                                       pts_sd=("pts", "std"), pts_mean=("pts", "mean"))
    actt = tb.groupby("team_id").agg(n_gs=("game_id", "nunique"),
                                     P_sd=("game_poss", "std"), ppp_sd=("ppp", "std"),
                                     pts_sd=("team_score", "std"),
                                     pts_mean=("team_score", "mean"))
    jt = simt.join(actt, lsuffix="_sim", rsuffix="_act", how="inner")
    jt = jt[jt["n_gs_act"] >= MIN_CELL // 3]
    say(f"  teams with >= {MIN_CELL // 3} actual games: {len(jt)}")
    say(f"  per-team SD ratios (sim/actual), median over teams:")
    for c in ("P_sd", "ppp_sd", "pts_sd"):
        rr = jt[f"{c}_sim"] / jt[f"{c}_act"]
        say(f"    {c:<8s} median {rr.median():.4f}   q1 {rr.quantile(.25):.4f}   "
            f"q3 {rr.quantile(.75):.4f}   frac<1 {float((rr < 1).mean()):.3f}")
    say("")
    jt["q"] = pd.qcut(jt["pts_mean_act"], 5, labels=[1, 2, 3, 4, 5])
    say("  by team scoring quintile (actual points/game):")
    say("   q | teams | sim pts SD | act pts SD | ratio | sim P SD | act P SD | ratio | "
        "sim ppp SD | act ppp SD | ratio")
    for q, sub in jt.groupby("q", observed=True):
        tag = "" if len(sub) >= MIN_CELL else "  UNDERPOWERED"
        say(f"   {q} | {len(sub):5d} | {sub['pts_sd_sim'].mean():10.3f} | "
            f"{sub['pts_sd_act'].mean():10.3f} | "
            f"{sub['pts_sd_sim'].mean()/sub['pts_sd_act'].mean():5.3f} | "
            f"{sub['P_sd_sim'].mean():8.3f} | {sub['P_sd_act'].mean():8.3f} | "
            f"{sub['P_sd_sim'].mean()/sub['P_sd_act'].mean():5.3f} | "
            f"{sub['ppp_sd_sim'].mean():10.4f} | {sub['ppp_sd_act'].mean():10.4f} | "
            f"{sub['ppp_sd_sim'].mean()/sub['ppp_sd_act'].mean():5.3f}{tag}")
    say("")

    # =====================================================================
    say("")
    say("## E. Q3 -- ties at the end of regulation, and OT handling")
    say("")
    ag["reg_margin"] = np.where(ag["np_act"] > 2, 0.0, ag["m_act"].astype(float))
    ot_sim = float(sim["ot"].mean())
    ot_act = float((ag["np_act"] > 2).mean())
    say(f"  OT rate (== regulation tie rate)   sim {ot_sim:.4f}   actual {ot_act:.4f}   "
        f"delta {ot_sim - ot_act:+.4f}   ratio {ot_sim/ot_act:.4f}")
    say(f"  sim rows {len(sim):,} ({int(sim['ot'].sum()):,} OT)   "
        f"actual games {len(ag):,} ({int((ag['np_act'] > 2).sum()):,} OT)")
    say("")
    say("  Regulation-margin distribution (the tie rate IS the density at 0):")
    say("    |reg margin| | sim P(==k) | act P(==k) | ratio | sim P(<=k) | act P(<=k)")
    sm = np.abs(sim["reg_margin"].to_numpy())
    am = np.abs(ag["reg_margin"].to_numpy())
    for k in range(0, 11):
        ps = float((sm == k).mean())
        pa = float((am == k).mean())
        cs = float((sm <= k).mean())
        ca = float((am <= k).mean())
        rr = ps / pa if pa > 0 else float("nan")
        say(f"    {k:>12d} | {ps:10.5f} | {pa:10.5f} | {rr:5.3f} | {cs:10.5f} | {ca:10.5f}")
    say("")
    sd_reg_s = float(np.std(sim["reg_margin"]))
    sd_reg_a = float(np.std(ag["reg_margin"]))
    sd_fin_s = float(np.std(sim["margin"]))
    sd_fin_a = float(np.std(ag["m_act"].astype(float)))
    say(f"  regulation-margin SD   sim {sd_reg_s:7.4f}   actual {sd_reg_a:7.4f}   "
        f"ratio {sd_reg_s/sd_reg_a:.4f}")
    say(f"  final-margin      SD   sim {sd_fin_s:7.4f}   actual {sd_fin_a:7.4f}   "
        f"ratio {sd_fin_s/sd_fin_a:.4f}")
    say(f"  excess kurtosis of the regulation margin   sim "
        f"{float(pd.Series(sim['reg_margin']).kurtosis()):+.4f}   "
        f"actual {float(pd.Series(ag['reg_margin']).kurtosis()):+.4f}")
    say("")
    say("  A Gaussian control. If the regulation margin were normal with the SD")
    say("  each side actually has, the implied P(|m| <= 0.5) would be:")
    for nm, s in (("sim", sd_reg_s), ("actual", sd_reg_a)):
        # crude: density at 0 of N(mean, sd) times a unit-wide bin
        mu = float(np.mean(sim["reg_margin"])) if nm == "sim" else float(np.mean(ag["reg_margin"]))
        dens = np.exp(-0.5 * (mu / s) ** 2) / (s * np.sqrt(2 * np.pi))
        obs = ot_sim if nm == "sim" else ot_act
        say(f"    {nm:<7s} SD {s:7.4f}  normal P(tie) {dens:.5f}   observed {obs:.5f}   "
            f"observed/normal {obs/dens:.4f}")
    say("  (the observed/normal ratio is the EXCESS mass at zero that late-game")
    say("   dynamics produce. A sim whose ratio is materially below the actual's")
    say("   is missing late-game clustering, not OT handling.)")
    say("")
    say("  Per-game tie rate (sim has many seeds per game, so P(tie|game) is observed):")
    tg = sim.groupby("game_id").agg(tie=("ot", "mean"), sd=("margin", "std"),
                                    mm=("margin", "mean"))
    tgj = tg.join(ag.set_index("game_id")[["np_act", "m_act"]], how="inner")
    tgj["act_ot"] = (tgj["np_act"] > 2).astype(float)
    tgj["q"] = pd.qcut(tgj["mm"].abs(), 5, labels=[1, 2, 3, 4, 5])
    say("   |pred margin| quintile | games | mean sim P(tie) | actual OT rate | ratio")
    for q, sub in tgj.groupby("q", observed=True):
        tag = "" if len(sub) >= MIN_CELL else "  UNDERPOWERED"
        ra = float(sub["act_ot"].mean())
        rs = float(sub["tie"].mean())
        say(f"   {q:>21} | {len(sub):5d} | {rs:15.5f} | {ra:14.5f} | "
            f"{(rs/ra if ra else float('nan')):5.3f}{tag}")
    say("")
    say("  Within-game margin SD (the engine's own late-game dispersion at a fixed matchup):")
    say(f"    mean over games {float(tg['sd'].mean()):.4f}   "
        f"between-game SD of the per-game mean margin {float(np.std(tg['mm'])):.4f}")
    resid_m = tgj["m_act"].astype(float) - tgj["mm"]
    say(f"    SD of the actual margin around the sim's per-game mean {float(np.std(resid_m)):.4f}")
    say(f"    -> within-game margin variance missing: "
        f"{_v(resid_m) - float(np.mean(tg['sd']**2)):+.4f}")
    say("")
    say("  OT HANDLING, conditional on a tie (no tie-rate defect can touch this):")
    dist_s = sim.loc[sim["ot"] > 0, "n_periods"].value_counts(normalize=True).sort_index()
    dist_a = ag.loc[ag["np_act"] > 2, "np_act"].value_counts(normalize=True).sort_index()
    say("    n_periods | sim P(.|OT) | act P(.|OT) | sim n | act n")
    for k in sorted(set(dist_s.index) | set(dist_a.index)):
        ns = int((sim.loc[sim["ot"] > 0, "n_periods"] == k).sum())
        na = int((ag.loc[ag["np_act"] > 2, "np_act"] == k).sum())
        tag = "  UNDERPOWERED" if na < MIN_CELL else ""
        say(f"    {k:>9d} | {float(dist_s.get(k, 0.0)):11.5f} | "
            f"{float(dist_a.get(k, 0.0)):11.5f} | {ns:6d} | {na:5d}{tag}")
    otm_s = sim.loc[sim["ot"] > 0]
    otm_a = ag.loc[ag["np_act"] > 2]
    say(f"    points scored in OT games   sim total mean {float(otm_s['total'].mean()):.3f}   "
        f"actual {float(otm_a['t_act'].mean()):.3f}")
    say(f"    |final margin| in OT games  sim {float(otm_s['margin'].abs().mean()):.3f}   "
        f"actual {float(otm_a['m_act'].abs().mean()):.3f}")
    say(f"    home win rate in OT games   sim {float((otm_s['margin'] > 0).mean()):.4f}   "
        f"actual {float((otm_a['m_act'] > 0).mean()):.4f}")
    say("")

    # =====================================================================
    say("")
    say("## G. The G5 SD ratio on its OWN definition, decomposed")
    say("")
    say("  `cbb_sim.eval.gates.gate_g5` does NOT compare pooled SDs. It compares")
    say("    mean over games of the WITHIN-GAME (across-seed) sim SD")
    say("    against SD(actual - sim per-game mean), the engine's own residual.")
    say("  So G5 asks whether the engine's per-game dispersion is the right size")
    say("  for its own error. Section A's pooled ratios answer a different")
    say("  question and the two must not be quoted interchangeably.")
    say("")
    gg = sim.groupby("game_id")
    wg = pd.DataFrame({
        "P_mean": gg["P"].mean(), "P_var": gg["P"].var(),
        "t_mean": gg["total"].mean(), "t_var": gg["total"].var(),
        "m_mean": gg["margin"].mean(), "m_var": gg["margin"].var(),
        "rs_mean": gg["ppp_sum"].mean(), "rs_var": gg["ppp_sum"].var(),
        "rd_mean": gg["ppp_dif"].mean(), "rd_var": gg["ppp_dif"].var(),
        "h_var": gg["ppp_h"].var(), "a_var": gg["ppp_a"].var(),
    })
    # within-game covariances, one pass
    dev = sim.join(wg[["P_mean", "rs_mean", "rd_mean"]], on="game_id")
    dev["dP"] = dev["P"] - dev["P_mean"]
    dev["dRs"] = dev["ppp_sum"] - dev["rs_mean"]
    dev["dRd"] = dev["ppp_dif"] - dev["rd_mean"]
    dev["dh"] = dev["ppp_h"] - dev.groupby("game_id")["ppp_h"].transform("mean")
    dev["da"] = dev["ppp_a"] - dev.groupby("game_id")["ppp_a"].transform("mean")
    k = n_seeds / max(n_seeds - 1, 1)
    wg["cov_P_Rs"] = dev.groupby("game_id").apply(
        lambda d: float(np.mean(d["dP"] * d["dRs"])) * k, include_groups=False)
    wg["cov_P_Rd"] = dev.groupby("game_id").apply(
        lambda d: float(np.mean(d["dP"] * d["dRd"])) * k, include_groups=False)
    wg["cov_h_a"] = dev.groupby("game_id").apply(
        lambda d: float(np.mean(d["dh"] * d["da"])) * k, include_groups=False)

    wj = wg.join(ag.set_index("game_id")[["t_act", "m_act"]], how="inner")
    wj = wj.join(agp.set_index("game_id")[["game_poss"]], how="left")
    resid_t = float((wj["t_act"] - wj["t_mean"]).std())
    resid_m = float((wj["m_act"] - wj["m_mean"]).std())
    rp = wj["game_poss"].notna()
    resid_p = float((wj.loc[rp, "game_poss"] - wj.loc[rp, "P_mean"]).std())
    sd_t = float(np.sqrt(wj["t_var"]).mean())
    sd_m = float(np.sqrt(wj["m_var"]).mean())
    sd_p = float(np.sqrt(wj["P_var"]).mean())
    say(f"    quantity     mean within-game sim SD   SD(actual - sim mean)   G5 ratio")
    say(f"    margin       {sd_m:22.4f}   {resid_m:19.4f}   {sd_m/resid_m:8.4f}")
    say(f"    total        {sd_t:22.4f}   {resid_t:19.4f}   {sd_t/resid_t:8.4f}")
    say(f"    possessions  {sd_p:22.4f}   {resid_p:19.4f}   {sd_p/resid_p:8.4f}   "
        f"(same definition, not a pre-registered G5 line)")
    say("")
    say("  Within-game variance of the TOTAL, split into its three channels")
    say("  (mean over games; total = P x (ppp_h + ppp_a)):")
    eP = float(wj["P_mean"].mean())
    eRs = float(wj["rs_mean"].mean())
    eRd = float(wj["rd_mean"].mean())
    chA = float((wj["rs_mean"] ** 2 * wj["P_var"]).mean())
    chB = float((wj["P_mean"] ** 2 * wj["rs_var"]).mean())
    chC = float((2 * wj["P_mean"] * wj["rs_mean"] * wj["cov_P_Rs"]).mean())
    have = float(wj["t_var"].mean())
    need = resid_t ** 2
    say(f"    have  mean Var_g(total) {have:10.3f}   need {need:10.3f}   "
        f"gap {need - have:+10.3f}  (ratio of SDs {np.sqrt(have/need):.4f})")
    say(f"      channel P   (E[Rs]^2 Var_g(P))      {chA:10.3f}  ({100*chA/have:5.1f}%)")
    say(f"      channel R   (E[P]^2  Var_g(Rs))     {chB:10.3f}  ({100*chB/have:5.1f}%)")
    say(f"      channel cov (2 E[P]E[Rs] Cov_g)     {chC:10.3f}  ({100*chC/have:5.1f}%)")
    say(f"      remainder                            {have-chA-chB-chC:10.3f}")
    say("")
    say("  Three counterfactuals, each applied ALONE to the measured run (arithmetic")
    say("  on the decomposition, NOT a re-simulation and NOT a proposed fix):")
    need_Pvar = resid_p ** 2
    d1 = float((wj["rs_mean"] ** 2 * (need_Pvar - wj["P_var"])).mean())
    say(f"    (i)  possession draw widened to its own residual "
        f"(Var_g(P) {float(wj['P_var'].mean()):.3f} -> {need_Pvar:.3f}):")
    say(f"         Var_g(total) {have:.3f} -> {have+d1:.3f}, closes "
        f"{100*d1/(need-have):5.1f}% of the gap; G5 total ratio "
        f"{np.sqrt(have/need):.4f} -> {np.sqrt((have+d1)/need):.4f}")
    d2 = float((wj["P_mean"] ** 2 * (-2 * wj["cov_h_a"])).mean())
    say(f"    (ii) within-game Cov(ppp_h, ppp_a) set to ZERO "
        f"(mean {float(wj['cov_h_a'].mean()):+.6f} -> 0):")
    say(f"         Var_g(total) {have:.3f} -> {have+d2:.3f}, closes "
        f"{100*d2/(need-have):5.1f}% of the gap; G5 total ratio "
        f"{np.sqrt(have/need):.4f} -> {np.sqrt((have+d2)/need):.4f}")
    d3 = -chC
    say(f"    (iii) within-game Cov(P, ppp_sum) set to ZERO "
        f"(mean {float(wj['cov_P_Rs'].mean()):+.5f} -> 0):")
    say(f"         Var_g(total) {have:.3f} -> {have+d3:.3f}, closes "
        f"{100*d3/(need-have):5.1f}% of the gap; G5 total ratio "
        f"{np.sqrt(have/need):.4f} -> {np.sqrt((have+d3)/need):.4f}")
    say(f"    (i)+(ii)+(iii) together: Var_g(total) -> {have+d1+d2+d3:.3f}, G5 ratio "
        f"{np.sqrt((have+d1+d2+d3)/need):.4f}")
    say("")
    say("  The same three channels for the MARGIN (which PASSES G5), so the pass")
    say("  is read for whether it stands on separately-right parts or on cancellation:")
    mA = float((wj["rd_mean"] ** 2 * wj["P_var"]).mean())
    mB = float((wj["P_mean"] ** 2 * wj["rd_var"]).mean())
    mC = float((2 * wj["P_mean"] * wj["rd_mean"] * wj["cov_P_Rd"]).mean())
    haveM = float(wj["m_var"].mean())
    needM = resid_m ** 2
    say(f"    have mean Var_g(margin) {haveM:10.3f}   need {needM:10.3f}   "
        f"ratio of SDs {np.sqrt(haveM/needM):.4f}")
    say(f"      channel P {mA:8.3f} ({100*mA/haveM:5.1f}%)   "
        f"channel R {mB:9.3f} ({100*mB/haveM:5.1f}%)   "
        f"channel cov {mC:8.3f} ({100*mC/haveM:5.1f}%)")
    say(f"    -> the possession channel carries {100*mA/haveM:.1f}% of the margin's")
    say(f"       within-game variance and {100*chA/have:.1f}% of the total's, because")
    say(f"       E[ppp_dif]={eRd:.4f} against E[ppp_sum]={eRs:.4f}. The margin gate is")
    say(f"       therefore BLIND to the possession draw; only the total gate sees it.")
    say("")
    say("  The same counterfactual (i) carried to G5's home/away CORRELATION.")
    say("  Assumption, stated because it is an assumption: the missing within-game")
    say("  possession variance enters as extra SHARED pace noise, independent of")
    say("  both teams' PPP -- i.e. exactly what 'one pace realisation per game,")
    say("  both teams scaled by it' says it should be. Numerator and denominator")
    say("  both move; neither is held fixed to flatter the number.")
    dP_extra = need_Pvar - float(wj["P_var"].mean())
    eh_, ea_ = float(sim["ppp_h"].mean()), float(sim["ppp_a"].mean())
    vh_pts = _v(sim["home_pts"].astype(float)) + eh_ ** 2 * dP_extra
    va_pts = _v(sim["away_pts"].astype(float)) + ea_ ** 2 * dP_extra
    cov_new = cov_s + eh_ * ea_ * dP_extra
    say(f"    extra shared pace variance {dP_extra:+.3f}  "
        f"(within-game Var(P) {float(wj['P_var'].mean()):.3f} -> {need_Pvar:.3f})")
    say(f"    Cov(home,away) {cov_s:+.4f} -> {cov_new:+.4f}   "
        f"(actual {cov_a:+.4f})")
    say(f"    corr(home,away) {corr_s:+.4f} -> {cov_new/np.sqrt(vh_pts*va_pts):+.4f}   "
        f"(actual {corr_a:+.4f}, gate +/- 0.05)")
    say(f"    pooled pace channel E[h]E[a]Var(P) {eh_*ea_*_v(sim['P']):+.3f} -> "
        f"{eh_*ea_*(_v(sim['P'].to_numpy())+dP_extra):+.3f}   "
        f"(actual pace channel {act_pace_channel:+.3f})")
    say("")
    say(f"  Within-game Cov(ppp_h, ppp_a): mean {float(wj['cov_h_a'].mean()):+.6f}, "
        f"share of games negative {float((wg['cov_h_a'] < 0).mean()):.3f}")
    say(f"  Within-game corr(ppp_h, ppp_a): mean over games "
        f"{float((wg['cov_h_a'] / np.sqrt(wg['h_var'] * wg['a_var'])).mean()):+.4f}")
    say("")

    # =====================================================================
    say("")
    say("## H. Which event channel carries the within-game pace/efficiency link")
    say("")
    say("  Section G channel (iii) says a within-game Cov(P, ppp_sum) of -0.155")
    say("  cancels 22% of the engine's own within-game total variance. This names")
    say("  the channel instead of guessing at it: within each game, across seeds,")
    say("  the correlation of the possession count with each rate the L3 sub-models")
    say("  own -- and the same correlation in the actual, where it is a between-game")
    say("  correlation because there is one realisation per game.")
    say("")
    s2 = sim.copy()
    s2["fga"] = (s2["home_fga3"] + s2["away_fga3"] + s2["home_fga2_rim"]
                 + s2["away_fga2_rim"] + s2["home_fga2_jump"] + s2["away_fga2_jump"])
    s2["fgm"] = (s2["home_fgm3"] + s2["away_fgm3"] + s2["home_fgm2_rim"]
                 + s2["away_fgm2_rim"] + s2["home_fgm2_jump"] + s2["away_fgm2_jump"])
    s2["tpm"] = s2["home_fgm3"] + s2["away_fgm3"]
    s2["fta_"] = s2["home_fta"] + s2["away_fta"]
    s2["tov_"] = s2["home_tov"] + s2["away_tov"]
    s2["oreb_"] = s2["home_oreb"] + s2["away_oreb"]
    s2["dreb_"] = s2["home_dreb"] + s2["away_dreb"]
    s2["r_tov"] = s2["tov_"] / (2 * s2["P"])
    s2["r_efg"] = (s2["fgm"] + 0.5 * s2["tpm"]) / s2["fga"]
    s2["r_ftr"] = s2["fta_"] / s2["fga"]
    s2["r_oreb"] = s2["oreb_"] / (s2["oreb_"] + s2["dreb_"])
    s2["r_fgapp"] = s2["fga"] / (2 * s2["P"])
    cols = [("TOV per poss", "r_tov"), ("eFG%", "r_efg"), ("FTA/FGA", "r_ftr"),
            ("OREB%", "r_oreb"), ("FGA per poss", "r_fgapp"), ("PPP sum", "ppp_sum")]
    say("    rate            | sim WITHIN-game corr with P | sim POOLED corr | actual corr")
    tbg = tb.groupby("game_id").agg(
        fga=("fga", "sum"), fgm=("fgm", "sum"), tpm=("tpm", "sum"),
        fta=("fta", "sum"), tov=("tov", "sum"), oreb=("oreb", "sum"),
        dreb=("dreb", "sum"), pts=("team_score", "sum")).reset_index()
    tbg = tbg.merge(agp[["game_id", "game_poss"]], on="game_id", how="inner")
    tbg["r_tov"] = tbg["tov"] / (2 * tbg["game_poss"])
    tbg["r_efg"] = (tbg["fgm"] + 0.5 * tbg["tpm"]) / tbg["fga"]
    tbg["r_ftr"] = tbg["fta"] / tbg["fga"]
    tbg["r_oreb"] = tbg["oreb"] / (tbg["oreb"] + tbg["dreb"])
    tbg["r_fgapp"] = tbg["fga"] / (2 * tbg["game_poss"])
    tbg["ppp_sum"] = tbg["pts"] / tbg["game_poss"]
    dP2 = s2["P"] - s2.groupby("game_id")["P"].transform("mean")
    for nm, c in cols:
        dc = s2[c] - s2.groupby("game_id")[c].transform("mean")
        cw = _c(dP2.to_numpy(), dc.to_numpy()) / np.sqrt(_v(dP2) * _v(dc))
        cp = _c(s2["P"].to_numpy(), s2[c].to_numpy()) / np.sqrt(_v(s2["P"]) * _v(s2[c]))
        ok = tbg[c].notna() & np.isfinite(tbg[c])
        ca_ = (_c(tbg.loc[ok, "game_poss"].to_numpy(), tbg.loc[ok, c].to_numpy())
               / np.sqrt(_v(tbg.loc[ok, "game_poss"]) * _v(tbg.loc[ok, c])))
        say(f"    {nm:<15s} | {cw:+27.4f} | {cp:+15.4f} | {ca_:+11.4f}")
    say("")
    say("  (the sim's WITHIN-game column is the one channel (iii) prices. The")
    say("   actual column is between-game and shares the box possession estimate's")
    say("   own algebra -- TOV and FGA sit in its numerator -- so it is a bound on")
    say("   the real dependence, not a clean measurement of it. Both are reported.)")
    say("")

    # =====================================================================
    say("")
    say("## F. Per month (the level check)")
    say("")
    mm = sim.merge(ag[["game_id", "month"]], on="game_id", how="left")
    say("   month | n games | sim P SD | act P SD | ratio | sim tot SD | act tot SD | ratio | "
        "sim OT | act OT")
    for m, sub in mm.groupby("month"):
        asub = ag[ag["month"] == m]
        apsub = agp[agp["month"] == m]
        tag = "" if len(asub) >= MIN_CELL else "  UNDERPOWERED"
        sp = float(np.std(sub["P"]))
        apv = float(np.std(apsub["game_poss"])) if len(apsub) else float("nan")
        st = float(np.std(sub["total"]))
        at = float(np.std(asub["t_act"].astype(float)))
        say(f"   {int(m):5d} | {len(asub):7d} | {sp:8.3f} | {apv:8.3f} | {sp/apv:5.3f} | "
            f"{st:10.3f} | {at:10.3f} | {st/at:5.3f} | "
            f"{float(sub['ot'].mean()):6.4f} | {float((asub['np_act'] > 2).mean()):6.4f}{tag}")
    say("")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
