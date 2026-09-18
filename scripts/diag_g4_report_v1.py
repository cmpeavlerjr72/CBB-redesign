"""diag_g4_report_v1.py -- channel decomposition for gate G4's OREB% and
FTA/FGA misses on fold 2 (2025), plus the multi-level cuts.

DIAGNOSTIC ONLY.  Consumes `diag_g4_tap_v1.py`'s instrumented sim output, the
served 75-seed run, the hoopR box (the grader's ACTUAL), and the pbp event
layer.  Fits nothing, changes no default, writes no engine file.

Every channel table closes: the channel contributions are defined so that they
sum to the measured gap by construction, and the residual is printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEASON = 2025
MISS = ("rim", "jump2", "three", "ft")
PO_CLASSES = ("TOV", "FGA_rim", "FGA_jump2", "FGA_3", "FT_trip_shooting", "FT_trip_bonus")
OUT: dict = {}
pd.set_option("display.width", 200)


def _p(*a):
    print(*a, flush=True)


def _hdr(t):
    _p("\n" + "=" * 78)
    _p(t)
    _p("=" * 78)


def sim_team_frame(g: pd.DataFrame) -> pd.DataFrame:
    cols = ("fga3", "fga2_rim", "fga2_jump", "fta", "tov", "oreb", "dreb",
            "fgm2_rim", "fgm2_jump", "fgm3", "ftm")
    out = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = g[["game_id", "seed", f"{opp}_dreb"] + [f"{side}_{c}" for c in cols]].copy()
        d.columns = ["game_id", "seed", "opp_dreb"] + list(cols)
        d["side"] = side
        out.append(d)
    s = pd.concat(out, ignore_index=True)
    s["fga"] = s.fga3 + s.fga2_rim + s.fga2_jump
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap", default="results/g4_diag/tap")
    ap.add_argument("--run", default="results/engine_v0/F2_2025_s200_v5b_A")
    ap.add_argument("--offline", default="results/g4_diag/g4_offline_truth_actual_sim.json")
    ap.add_argument("--reb-offline", default="results/g4_diag/g4_offline_reb.json")
    ap.add_argument("--out", default="results/g4_diag")
    a = ap.parse_args()
    tapd, outd = Path(a.tap), Path(a.out)
    off = json.loads(Path(a.offline).read_text(encoding="utf-8"))
    rebo = (json.loads(Path(a.reb_offline).read_text(encoding="utf-8"))
            if Path(a.reb_offline).exists() else None)
    if rebo is not None and "reb" in rebo:
        rebo = rebo["reb"]

    from cbb_sim.eval import reference as ref_mod

    # ---------------------------------------------------------------- load
    _hdr("0. inputs")
    tg = pd.concat([pd.read_parquet(p) for p in sorted(tapd.glob("games_*.parquet"))],
                   ignore_index=True)
    ch = pd.concat([pd.read_parquet(p) for p in sorted(tapd.glob("chances_*.parquet"))],
                   ignore_index=True)
    tp = pd.concat([pd.read_parquet(p) for p in sorted(tapd.glob("trips_*.parquet"))],
                   ignore_index=True)
    rb = pd.concat([pd.read_parquet(p) for p in sorted(tapd.glob("rebs_*.parquet"))],
                   ignore_index=True)
    _p(f"tap: sims {len(tg)}  games {tg.game_id.nunique()}  seeds {sorted(tg.seed.unique())}")
    _p(f"     chances {len(ch)}  trips {len(tp)}  rebound opps {len(rb)}")
    full = pd.read_parquet(Path(a.run) / "games.parquet")
    box = ref_mod.load_actual_team_box(SEASON)

    # parity: the tap must reproduce the served run bit for bit
    ov = full.merge(tg[["game_id", "seed"]], on=["game_id", "seed"], how="inner")
    cmp_cols = [c for c in tg.columns if c in full.columns and c not in ("game_id", "seed")]
    l = tg.sort_values(["game_id", "seed"]).reset_index(drop=True)
    r = ov.sort_values(["game_id", "seed"]).reset_index(drop=True)
    identical = bool(len(l) == len(r) and all((l[c].values == r[c].values).all() for c in cmp_cols))
    _p(f"tap vs served run on the same (game, seed): n={len(r)}  BIT-IDENTICAL={identical}")
    OUT["tap_parity_bit_identical"] = identical
    OUT["tap_parity_n"] = int(len(r))

    sfull = sim_team_frame(full)
    stap = sim_team_frame(tg)
    sim_oreb_full = float(sfull.oreb.sum() / (sfull.oreb + sfull.opp_dreb).sum())
    sim_ftr_full = float(sfull.fta.sum() / sfull.fga.sum())
    sim_oreb_tap = float(stap.oreb.sum() / (stap.oreb + stap.opp_dreb).sum())
    sim_ftr_tap = float(stap.fta.sum() / stap.fga.sum())
    box_oreb = float(box.oreb.sum() / (box.oreb + box.opp_dreb).sum())
    box_ftr = float(box.fta.sum() / box.fga.sum())
    _p(f"\nOREB%   sim(full run) {sim_oreb_full:.5f}  sim(tap subset) {sim_oreb_tap:.5f}  "
       f"ACTUAL box {box_oreb:.5f}")
    _p(f"FTA/FGA sim(full run) {sim_ftr_full:.5f}  sim(tap subset) {sim_ftr_tap:.5f}  "
       f"ACTUAL box {box_ftr:.5f}")
    _p(f"subset representativeness: OREB% {abs(sim_oreb_tap - sim_oreb_full) * 100:.3f} pp, "
       f"FT rate {abs(sim_ftr_tap - sim_ftr_full) * 100:.3f} pp from the full run")
    OUT["levels"] = {"sim_oreb_full": sim_oreb_full, "sim_oreb_tap": sim_oreb_tap,
                     "sim_ft_rate_full": sim_ftr_full, "sim_ft_rate_tap": sim_ftr_tap,
                     "box_oreb": box_oreb, "box_ft_rate": box_ftr}

    # ================================================================== OREB
    _hdr("1. OREB%: the channel chain (each link closes into the next)")
    pbp_oreb = off["truth"]["pbp_oreb_pct"]
    live = rb[rb.outcome < 2]                    # 0 OREB, 1 DREB, 2 DEAD
    sim_o = float((live.outcome == 0).mean())
    sim_p = float(live.p_oreb.mean())
    _p(f"tap realised OREB% {sim_o:.5f}  vs tap mean model p(OREB) {sim_p:.5f} "
       f"(Monte-Carlo consistency: {(sim_o - sim_p) * 100:+.3f} pp)")

    B = rebo["model_true_oreb"] if rebo else float("nan")
    C = rebo["model_blk0_oreb"] if rebo else float("nan")
    chain = [
        ("ACTUAL, hoopR box (what G4 grades against)", box_oreb, None),
        ("ACTUAL, pbp event layer (the model's own target)", pbp_oreb,
         "grading-source channel"),
        ("served rebound arm, real fold-2 rows, TRUE blocked_f", B,
         "fold-2 model calibration"),
        ("served rebound arm, real fold-2 rows, blocked_f=0 (engine feed)", C,
         "blocked_f=0 feed"),
        ("SIM realised (full 75-seed run)", sim_oreb_full,
         "sim state/mix distribution"),
    ]
    rows = []
    prev = None
    for name, v, chan in chain:
        d = None if prev is None else (v - prev) * 100
        rows.append({"level": name, "value": v, "channel": chan or "-",
                     "delta_pp": d})
        prev = v
    t = pd.DataFrame(rows)
    _p(t.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    tot = (sim_oreb_full - box_oreb) * 100
    _p(f"\ntotal OREB% gap (sim - box) = {tot:+.3f} pp")
    sh = t.dropna(subset=["delta_pp"]).copy()
    sh["share_pct"] = sh.delta_pp / tot * 100
    _p(sh[["channel", "delta_pp", "share_pct"]].to_string(index=False,
                                                          float_format=lambda v: f"{v:+.3f}"))
    _p(f"closure: channels sum to {sh.delta_pp.sum():+.3f} pp vs measured {tot:+.3f} pp "
       f"(residual {sh.delta_pp.sum() - tot:+.4f} pp)")
    OUT["oreb_chain"] = t.to_dict("records")
    OUT["oreb_total_gap_pp"] = tot

    # sub-split of the last link: miss MIX vs within-type rate
    _hdr("1b. OREB%: inside the sim state/mix channel -- miss MIX vs within-type rate")
    sim_mix = live.groupby("miss_type").size() / len(live)
    sim_rate = live.groupby("miss_type").apply(lambda d: float((d.outcome == 0).mean()),
                                               include_groups=False)
    sim_pred = live.groupby("miss_type").p_oreb.mean()
    act_mix = pd.Series(off["actual"]["actual_mix"]).reindex(MISS)
    ref_rate = (pd.Series({r["miss_type"]: r["model_blk0"] for r in rebo["by_miss_type"]})
                if rebo else pd.Series(dtype=float)).reindex(MISS)
    sim_mix.index = [MISS[int(i)] for i in sim_mix.index]
    sim_rate.index = [MISS[int(i)] for i in sim_rate.index]
    sim_pred.index = [MISS[int(i)] for i in sim_pred.index]
    sim_mix, sim_rate, sim_pred = (sim_mix.reindex(MISS), sim_rate.reindex(MISS),
                                   sim_pred.reindex(MISS))
    cmp_t = pd.DataFrame({"sim_mix": sim_mix, "act_mix": act_mix,
                          "sim_p_oreb": sim_pred, "realrows_p_oreb_blk0": ref_rate,
                          "actual_rate": pd.Series(off["actual"]["actual_rate"]).reindex(MISS)})
    _p(cmp_t.to_string(float_format=lambda v: f"{v:.5f}"))
    wbar = (sim_mix + act_mix) / 2
    rbar = (sim_pred + ref_rate) / 2
    mix_ch = float(((sim_mix - act_mix) * rbar).sum()) * 100
    rate_ch = float((wbar * (sim_pred - ref_rate)).sum()) * 100
    meas = (sim_p - C) * 100 if rebo else float("nan")
    _p(f"\nmix channel      {mix_ch:+.3f} pp")
    _p(f"within-type rate {rate_ch:+.3f} pp")
    _p(f"sum {mix_ch + rate_ch:+.3f} pp vs measured (tap mean p - real-rows blk0 p) {meas:+.3f} pp")
    OUT["oreb_mix_vs_rate"] = {"mix_pp": mix_ch, "rate_pp": rate_ch, "measured_pp": meas,
                               "table": cmp_t.reset_index().to_dict("records")}

    # ================================================================ FT rate
    _hdr("2. FTA/FGA: the channel chain")
    # sim FTA composition from the tap
    tp = tp.copy()
    tp["kind"] = np.where(tp.cls == 4, "shooting",
                          np.where(tp.cls == 5, "bonus", "and_one"))
    tp.loc[(tp.n_att == 1) & (tp.one_and_one == 0) & (tp.cls < 4), "kind"] = "and_one"
    comp = tp.groupby("kind").agg(n_trips=("fta", "size"), fta=("fta", "sum"),
                                  fta_per_trip=("fta", "mean"))
    tap_fga = float(stap.fga.sum())
    tap_fta = float(stap.fta.sum())
    comp["per_fga"] = comp.fta / tap_fga
    comp["share"] = comp.fta / tap_fta
    _p(f"tap FTA total {tap_fta:.0f} (trip-tap sum {tp.fta.sum():.0f}); FGA {tap_fga:.0f}")
    _p(comp.to_string(float_format=lambda v: f"{v:.5f}"))
    OUT["sim_fta_composition"] = comp.reset_index().to_dict("records")

    # actual composition on the pbp event layer, same three kinds
    ac = pd.read_parquet(ROOT / f"data/processed/possessions_v2/chances_{SEASON}.parquet")
    ac["fga"] = ac.fga_rim + ac.fga_jump2 + ac.fga_3
    act_fga_pbp = float(ac.fga.sum())
    a_ao = float(ac.loc[ac.and_one, "fta"].sum())
    te = ac.terminal_event.astype(str)
    a_sh = float(ac.loc[te == "FT_trip_shooting", "fta"].sum())
    a_bo = float(ac.loc[te == "FT_trip_bonus", "fta"].sum())
    a_ot = float(ac.fta.sum()) - a_ao - a_sh - a_bo
    # box side, same team-games, so the technical / feed channel is explicit
    box_ftr_matched = off["truth"]["box_ft_rate"]
    pbp_ftr_matched = off["truth"]["pbp_ft_rate"]
    tech_pp = (pbp_ftr_matched - box_ftr_matched) * 100

    sim_ao = float(comp.loc["and_one", "per_fga"]) if "and_one" in comp.index else 0.0
    sim_sh = float(comp.loc["shooting", "per_fga"]) if "shooting" in comp.index else 0.0
    sim_bo = float(comp.loc["bonus", "per_fga"]) if "bonus" in comp.index else 0.0
    act_ao, act_sh, act_bo = a_ao / act_fga_pbp, a_sh / act_fga_pbp, a_bo / act_fga_pbp
    act_ot = a_ot / act_fga_pbp

    rows = [
        ("technical FTs + feed gap (engine has no technical-FT rule)", tech_pp),
        ("and-one FTA / FGA", (sim_ao - act_ao) * 100),
        ("shooting-foul trip FTA / FGA", (sim_sh - act_sh) * 100),
        ("bonus trip FTA / FGA", (sim_bo - act_bo) * 100),
        ("other event-layer FTA / FGA", (0.0 - act_ot) * 100),
    ]
    tot_ft = (sim_ftr_full - box_ftr) * 100
    # the tap subset carries its own small offset from the full run; state it
    subset_pp = (sim_ftr_full - sim_ftr_tap) * 100
    rows.append(("tap-subset vs full-run offset (measurement, not a defect)", subset_pp))
    t2 = pd.DataFrame(rows, columns=["channel", "delta_pp"])
    t2["share_pct"] = t2.delta_pp / tot_ft * 100
    _p("\nchannel table (each entry = sim FTA of that kind per FGA minus actual's):")
    _p(t2.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    _p(f"\ntotal FTA/FGA gap (sim full run - box) = {tot_ft:+.3f} pp")
    _p(f"closure: channels sum to {t2.delta_pp.sum():+.3f} pp "
       f"(unexplained residual {tot_ft - t2.delta_pp.sum():+.3f} pp)")
    OUT["ft_channels"] = t2.to_dict("records")
    OUT["ft_total_gap_pp"] = tot_ft

    # trip RATE vs FTA-PER-TRIP inside each kind
    _hdr("2b. FTA/FGA: trip rate vs attempts-per-trip, and the trip-size mix")
    sim_pos = int((ch.is_first == 1).sum())
    act_pos = int(pd.read_parquet(
        ROOT / f"data/processed/possessions_v2/possessions_{SEASON}.parquet",
        columns=["game_id"]).shape[0])
    _p(f"sim possessions (tap) {sim_pos}   actual possessions (pbp) {act_pos}")
    rows = []
    for kind, cls_id, te_name in (("and_one", None, None),
                                  ("shooting", 4, "FT_trip_shooting"),
                                  ("bonus", 5, "FT_trip_bonus")):
        if kind == "and_one":
            s_n = int((tp.kind == "and_one").sum())
            a_n = int(ac.and_one.sum())
            s_f, a_f = float(tp.loc[tp.kind == "and_one", "fta"].sum()), a_ao
        else:
            s_n = int((ch.cls == cls_id).sum())
            a_n = int((te == te_name).sum())
            s_f = float(tp.loc[tp.kind == kind, "fta"].sum())
            a_f = a_sh if kind == "shooting" else a_bo
        rows.append({"kind": kind,
                     "sim_trips_per_100poss": s_n / sim_pos * 100,
                     "act_trips_per_100poss": a_n / act_pos * 100,
                     "sim_fta_per_trip": s_f / max(s_n, 1),
                     "act_fta_per_trip": a_f / max(a_n, 1)})
    t3 = pd.DataFrame(rows)
    t3["trip_rate_ratio"] = t3.sim_trips_per_100poss / t3.act_trips_per_100poss
    t3["fta_per_trip_ratio"] = t3.sim_fta_per_trip / t3.act_fta_per_trip
    _p(t3.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    OUT["ft_trip_rate_vs_size"] = t3.to_dict("records")

    _p("\nsim trip-size mix (n_att, one_and_one) by kind:")
    _p(tp.groupby(["kind", "n_att", "one_and_one"]).agg(
        n=("fta", "size"), fta_per_trip=("fta", "mean")).to_string())
    _p("\nactual FTA-per-trip distribution by terminal event:")
    trip_a = ac[te.isin(["FT_trip_shooting", "FT_trip_bonus"])]
    _p(pd.crosstab(trip_a.terminal_event.astype(str), trip_a.fta,
                   normalize="index").to_string(float_format=lambda v: f"{v:.4f}"))

    # -------------------------------------------------- bonus-state occupancy
    _hdr("2c. FTA/FGA: the bonus STATE -- occupancy vs conditional trip rate")
    ch["late"] = (ch.period == 2) & (ch.sec <= 120)
    tp["late"] = (tp.period == 2) & (tp.sec <= 120)
    ac["late"] = (ac.period == 2) & (ac.start_clock <= 120)
    s_occ = float(ch.bonus.mean())
    a_occ = float(ac.off_in_bonus.mean())
    s_p1 = float((ch.loc[ch.bonus == 1, "cls"] == 5).mean())
    s_p0 = float((ch.loc[ch.bonus == 0, "cls"] == 5).mean())
    a_te = ac.terminal_event.astype(str)
    a_p1 = float((a_te[ac.off_in_bonus] == "FT_trip_bonus").mean())
    a_p0 = float((a_te[~ac.off_in_bonus] == "FT_trip_bonus").mean())
    s_rate, a_rate = s_occ * s_p1 + (1 - s_occ) * s_p0, a_occ * a_p1 + (1 - a_occ) * a_p0
    pbar1, pbar0 = (s_p1 + a_p1) / 2, (s_p0 + a_p0) / 2
    occ_ch = (s_occ - a_occ) * (pbar1 - pbar0)
    cond_ch = ((s_occ + a_occ) / 2) * (s_p1 - a_p1) + (1 - (s_occ + a_occ) / 2) * (s_p0 - a_p0)
    _p(f"P(offence in bonus)       sim {s_occ:.5f}  act {a_occ:.5f}  "
       f"({(s_occ - a_occ) * 100:+.3f} pp)")
    _p(f"P(bonus trip | in bonus)  sim {s_p1:.5f}  act {a_p1:.5f}")
    _p(f"P(bonus trip | not)       sim {s_p0:.5f}  act {a_p0:.5f}")
    _p(f"bonus trips per chance    sim {s_rate:.5f}  act {a_rate:.5f}  "
       f"({(s_rate - a_rate) / a_rate * 100:+.2f}%)")
    _p(f"  split: bonus-state OCCUPANCY {occ_ch / (s_rate - a_rate) * 100:+.1f}%   "
       f"CONDITIONAL trip rate {cond_ch / (s_rate - a_rate) * 100:+.1f}%  "
       f"(residual {(s_rate - a_rate - occ_ch - cond_ch):.2e})")
    s_db = float((tp.loc[tp.kind == "bonus", "dbonus"] > 0).mean())
    a_db = float(ac.loc[a_te == "FT_trip_bonus", "off_in_double_bonus"].mean())
    _p(f"share of BONUS trips taken in double bonus: sim {s_db:.5f}  act {a_db:.5f}")
    OUT["double_bonus_trip_share"] = {"sim": s_db, "act": a_db}
    OUT["bonus_state"] = {"sim_occupancy": s_occ, "act_occupancy": a_occ,
                          "sim_p_trip_given_bonus": s_p1, "act_p_trip_given_bonus": a_p1,
                          "sim_p_trip_given_not": s_p0, "act_p_trip_given_not": a_p0,
                          "occupancy_share": float(occ_ch / (s_rate - a_rate)),
                          "conditional_share": float(cond_ch / (s_rate - a_rate))}

    _hdr("2d. FTA/FGA: exact 4-term split by clock window (final 2:00 vs rest)")
    ch["is_fga"] = ch.cls.isin([1, 2, 3])
    C_s, C_a = float(len(ch)), float(len(ac))
    f_s = {w: float(tp.loc[tp.late == w, "fta"].sum()) / C_s for w in (True, False)}
    g_s = {w: float(ch.loc[ch.late == w, "is_fga"].sum()) / C_s for w in (True, False)}
    f_a = {w: float(ac.loc[ac.late == w, "fta"].sum()) / C_a for w in (True, False)}
    g_a = {w: float(ac.loc[ac.late == w, "fga"].sum()) / C_a for w in (True, False)}
    G_s, G_a = g_s[True] + g_s[False], g_a[True] + g_a[False]
    F_a = f_a[True] + f_a[False]
    rows = []
    for w, lab in ((True, "final 2:00 (reg)"), (False, "rest of game")):
        rows.append({"window": lab, "term": "FTA", "pp": (f_s[w] - f_a[w]) / G_s * 100})
        rows.append({"window": lab, "term": "FGA (denominator)",
                     "pp": -F_a * (g_s[w] - g_a[w]) / (G_s * G_a) * 100})
    t4 = pd.DataFrame(rows)
    ev_gap = (f_s[True] + f_s[False]) / G_s * 100 - F_a / G_a * 100
    t4["share_of_event_gap_pct"] = t4.pp / ev_gap * 100
    t4["share_of_box_gap_pct"] = t4.pp / tot_ft * 100
    _p(f"sim chance share in the window {float(ch.late.mean()):.5f}  "
       f"act {float(ac.late.mean()):.5f}")
    _p(f"local FTA/FGA in the window: sim "
       f"{f_s[True] / g_s[True]:.5f}  act {f_a[True] / g_a[True]:.5f}")
    _p(t4.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    _p(f"\nevent-layer gap (sim tap - pbp) {ev_gap:+.3f} pp; 4 terms sum "
       f"{t4.pp.sum():+.3f} pp (residual {t4.pp.sum() - ev_gap:+.4f})")
    lateshare = t4[t4.window == "final 2:00 (reg)"].pp.sum()
    _p(f"FINAL 2:00 total: {lateshare:+.3f} pp = {lateshare / tot_ft * 100:.1f}% of the "
       f"{tot_ft:+.3f} pp box gap")
    OUT["ft_window_split"] = {"table": t4.to_dict("records"),
                              "late_total_pp": float(lateshare),
                              "late_share_of_box_gap_pct": float(lateshare / tot_ft * 100)}

    _hdr("2e. bonus-state OCCUPANCY and FT supply by game minute")

    def minute_bucket0(period, sec):
        per = np.asarray(period)
        s = np.asarray(sec)
        m = np.where(per <= 2, (per - 1) * 20 + (20 - np.ceil(s / 60.0)), 40.0)
        return np.clip(m, 0, 40).astype(int)

    BINS = [-1, 4, 9, 14, 19, 24, 29, 34, 37, 40]
    ch["mb0"] = pd.cut(minute_bucket0(ch.period, ch.sec), BINS)
    ac["mb0"] = pd.cut(minute_bucket0(ac.period, ac.start_clock), BINS)
    so = ch.groupby("mb0", observed=True).agg(n_sim=("bonus", "size"),
                                              sim_bonus=("bonus", "mean"))
    ao_ = ac.groupby("mb0", observed=True).agg(n_act=("off_in_bonus", "size"),
                                               act_bonus=("off_in_bonus", "mean"))
    jb = so.join(ao_)
    jb["d_pp"] = (jb.sim_bonus - jb.act_bonus) * 100
    jb["sim_bonus_trip"] = ch.groupby("mb0", observed=True).cls.apply(lambda x: float((x == 5).mean()))
    jb["act_bonus_trip"] = ac.groupby("mb0", observed=True).terminal_event.apply(
        lambda x: float((x.astype(str) == "FT_trip_bonus").mean()))
    _p("P(offence in bonus) and bonus-trip rate per chance, by game minute:")
    _p(jb.to_string(float_format=lambda v: f"{v:.4f}"))
    OUT["bonus_by_minute"] = jb.reset_index().astype(str).to_dict("records")

    _p("\nactual season drift in the graded quantities (box, all D-I team-games):")
    drift = []
    for yr in (2022, 2023, 2024, 2025):
        b = ref_mod.load_actual_team_box(yr)
        drift.append({"season": yr, "n_team_games": len(b),
                      "oreb_pct": float(b.oreb.sum() / (b.oreb + b.opp_dreb).sum()),
                      "ft_rate": float(b.fta.sum() / b.fga.sum())})
    dft = pd.DataFrame(drift)
    _p(dft.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    OUT["box_season_drift"] = drift

    _p("\nbonus trips per chance, in vs out of the window:")
    bw = pd.DataFrame({
        "window": ["final 2:00", "rest"],
        "sim": [float((ch.loc[ch.late, "cls"] == 5).mean()),
                float((ch.loc[~ch.late, "cls"] == 5).mean())],
        "act": [float((a_te[ac.late] == "FT_trip_bonus").mean()),
                float((a_te[~ac.late] == "FT_trip_bonus").mean())],
        "sim_chances": [int(ch.late.sum()), int((~ch.late).sum())],
        "act_chances": [int(ac.late.sum()), int((~ac.late).sum())]})
    bw["ratio"] = bw.sim / bw.act
    _p(bw.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    OUT["bonus_trips_by_window"] = bw.to_dict("records")

    # ---------------------------------------------------- multi-level cuts
    _hdr("3. multi-level evidence")
    uni = pd.read_parquet(ROOT / "data/processed/games_universe.parquet")
    u = uni[uni.season == SEASON][["game_id", "game_date", "neutral_site",
                                   "home_team_id", "away_team_id"]].copy()
    u["month"] = pd.to_datetime(u.game_date).dt.month
    try:
        from cbb_sim.features import conference as CF
        conf = CF.build_conference_flags([SEASON])[["game_id", "is_conf_game"]]
    except Exception as e:                                        # noqa: BLE001
        _p(f"conference flags unavailable: {e}")
        conf = pd.DataFrame({"game_id": [], "is_conf_game": []})

    sf = sfull.copy()
    sf["oreb_den"] = sf.oreb + sf.opp_dreb
    sf = sf.merge(u, on="game_id", how="inner")
    sf["site"] = np.where(sf.neutral_site, "neutral", sf.side)
    sf = sf.merge(conf, on="game_id", how="left")
    sf["team_id"] = np.where(sf.side == "home", sf.home_team_id, sf.away_team_id)

    bx = box[["game_id", "team_id", "opp_team_id", "fga", "fta", "oreb", "dreb",
              "opp_dreb", "tov"]].copy()
    bx["oreb_den"] = bx.oreb + bx.opp_dreb
    bx = bx.merge(u, on="game_id", how="inner")
    bx["site"] = np.where(bx.neutral_site, "neutral",
                          np.where(bx.team_id == bx.home_team_id, "home", "away"))
    bx = bx.merge(conf, on="game_id", how="left")

    def pooled(d):
        return pd.Series({"n": len(d),
                          "oreb_pct": d.oreb.sum() / d.oreb_den.sum(),
                          "ft_rate": d.fta.sum() / d.fga.sum()})

    for label, key in (("site (home/away/neutral)", "site"),
                       ("conference game", "is_conf_game"),
                       ("month", "month")):
        s = sf.groupby(key).apply(pooled, include_groups=False)
        b = bx.groupby(key).apply(pooled, include_groups=False)
        j = s.join(b, lsuffix="_sim", rsuffix="_act")
        j["d_oreb_pp"] = (j.oreb_pct_sim - j.oreb_pct_act) * 100
        j["d_ftr_pp"] = (j.ft_rate_sim - j.ft_rate_act) * 100
        j["underpowered"] = j.n_act < 300
        _p(f"\n--- {label} ---")
        _p(j[["n_act", "oreb_pct_sim", "oreb_pct_act", "d_oreb_pp",
              "ft_rate_sim", "ft_rate_act", "d_ftr_pp", "underpowered"]].to_string(
            float_format=lambda v: f"{v:.4f}"))
        OUT.setdefault("cuts", {})[key] = j.reset_index().to_dict("records")

    # per-team, bucketed by PRIOR-season (2024) quintile of the same metric
    _hdr("3b. per-team, by PRIOR-season (2024) quintile -- does the gap slope?")
    prior = ref_mod.load_actual_team_box(2024)
    prior["oreb_den"] = prior.oreb + prior.opp_dreb
    pt = prior.groupby("team_id").agg(n=("game_id", "size"), oreb=("oreb", "sum"),
                                      den=("oreb_den", "sum"), fta=("fta", "sum"),
                                      fga=("fga", "sum"))
    pt = pt[pt.n >= 10]
    pt["prior_oreb"] = pt.oreb / pt.den
    pt["prior_ftr"] = pt.fta / pt.fga
    simt = sf.groupby("team_id").agg(n=("game_id", "size"), oreb=("oreb", "sum"),
                                     den=("oreb_den", "sum"), fta=("fta", "sum"),
                                     fga=("fga", "sum"))
    actt = bx.groupby("team_id").agg(n=("game_id", "size"), oreb=("oreb", "sum"),
                                     den=("oreb_den", "sum"), fta=("fta", "sum"),
                                     fga=("fga", "sum"))
    j = pt[["prior_oreb", "prior_ftr"]].join(
        (simt.oreb / simt.den).rename("sim_oreb")).join(
        (actt.oreb / actt.den).rename("act_oreb")).join(
        (simt.fta / simt.fga).rename("sim_ftr")).join(
        (actt.fta / actt.fga).rename("act_ftr")).join(
        actt.n.rename("n_games_2025")).dropna()
    for metric, pcol, scol, acol in (("OREB%", "prior_oreb", "sim_oreb", "act_oreb"),
                                     ("FTA/FGA", "prior_ftr", "sim_ftr", "act_ftr")):
        j["q"] = pd.qcut(j[pcol], 5, labels=False)
        q = j.groupby("q").agg(teams=(pcol, "size"), n_games=("n_games_2025", "sum"),
                               prior=(pcol, "mean"), sim=(scol, "mean"),
                               act=(acol, "mean"))
        q["delta_pp"] = (q.sim - q.act) * 100
        span_s = q.sim.iloc[-1] - q.sim.iloc[0]
        span_a = q.act.iloc[-1] - q.act.iloc[0]
        mono = int(sum(np.diff(q.sim.to_numpy()) > 0))
        _p(f"\n--- {metric} by 2024 prior quintile ---")
        _p(q.to_string(float_format=lambda v: f"{v:.5f}"))
        _p(f"sim span {span_s:.5f}  actual span {span_a:.5f}  "
           f"slope ratio {span_s / span_a:.4f}  monotone {mono}/4  "
           f"gap slope Q5-Q1 {(q.delta_pp.iloc[-1] - q.delta_pp.iloc[0]):+.3f} pp")
        OUT.setdefault("quintiles", {})[metric] = {
            "table": q.reset_index().to_dict("records"),
            "slope_ratio": float(span_s / span_a), "monotone": mono,
            "gap_slope_q5_q1_pp": float(q.delta_pp.iloc[-1] - q.delta_pp.iloc[0])}

    # per-game distribution
    _hdr("3c. per-game")
    sg = sf.groupby("game_id").apply(pooled, include_groups=False)
    bg = bx.groupby("game_id").apply(pooled, include_groups=False)
    jg = sg.join(bg, lsuffix="_sim", rsuffix="_act").dropna()
    for m in ("oreb_pct", "ft_rate"):
        d = (jg[f"{m}_sim"] - jg[f"{m}_act"]) * 100
        _p(f"{m}: per-game delta mean {d.mean():+.3f} pp  median {d.median():+.3f}  "
           f"SD {d.std():.3f}  P(sim<act) {float((d < 0).mean()):.3f}  MAE {d.abs().mean():.3f}")
        OUT.setdefault("per_game", {})[m] = {"mean_pp": float(d.mean()),
                                             "median_pp": float(d.median()),
                                             "sd_pp": float(d.std()),
                                             "p_sim_below": float((d < 0).mean())}

    # by period / game-minute, from the tap
    _hdr("3d. by period and game-minute (tap vs pbp)")
    rb["late"] = (rb.period == 2) & (rb.sec <= 120)
    ev = pd.read_parquet(ROOT / "data/processed/models/rebound/events_v1.parquet",
                         columns=["season", "period", "seconds_remaining", "miss_type",
                                  "outcome", "blocked"])
    ev = ev[(ev.season == SEASON) & ev.outcome.isin(["OREB", "DREB"])].copy()
    ev["o"] = (ev.outcome == "OREB").astype(float)

    def minute_bucket(period, sec):
        per = np.asarray(period)
        s = np.asarray(sec)
        m = np.where(per <= 2, (per - 1) * 20 + (20 - np.ceil(s / 60.0)), 40.0)
        return np.clip(m, 0, 40).astype(int)

    rl = rb[rb.outcome < 2].copy()
    rl["mb"] = pd.cut(minute_bucket(rl.period, rl.sec), [-1, 4, 9, 14, 19, 24, 29, 34, 37, 40])
    ev["mb"] = pd.cut(minute_bucket(ev.period, ev.seconds_remaining),
                      [-1, 4, 9, 14, 19, 24, 29, 34, 37, 40])
    s = rl.groupby("mb", observed=True).agg(n_sim=("outcome", "size"),
                                            sim=("outcome", lambda x: float((x == 0).mean())))
    b = ev.groupby("mb", observed=True).agg(n_act=("o", "size"), act=("o", "mean"))
    jm = s.join(b)
    jm["d_pp"] = (jm.sim - jm.act) * 100
    jm["underpowered"] = jm.n_sim < 300
    _p("OREB% by game minute:")
    _p(jm.to_string(float_format=lambda v: f"{v:.4f}"))
    OUT["oreb_by_minute"] = jm.reset_index().astype(str).to_dict("records")

    tp["mb"] = pd.cut(minute_bucket(tp.period, tp.sec), [-1, 4, 9, 14, 19, 24, 29, 34, 37, 40])
    ch["mb"] = pd.cut(minute_bucket(ch.period, ch.sec), [-1, 4, 9, 14, 19, 24, 29, 34, 37, 40])
    ac["mb"] = pd.cut(minute_bucket(ac.period, ac.start_clock),
                      [-1, 4, 9, 14, 19, 24, 29, 34, 37, 40])
    sfta = tp.groupby("mb", observed=True).fta.sum()
    sfga = ch.groupby("mb", observed=True).is_fga.sum()
    afta = ac.groupby("mb", observed=True).fta.sum()
    afga = ac.groupby("mb", observed=True).fga.sum()
    jf = pd.DataFrame({"sim_ft_rate": sfta / sfga, "act_ft_rate": afta / afga,
                       "n_sim_fga": sfga, "n_act_fga": afga})
    jf["d_pp"] = (jf.sim_ft_rate - jf.act_ft_rate) * 100
    jf["gap_share_pct"] = (sfta / sfga.sum() - afta / afga.sum()) * 100 / tot_ft * 100
    _p("\nFTA/FGA by game minute (gap_share_pct = that minute-bucket's share of the "
       "event-layer gap, using each side's own total FGA as the denominator):")
    _p(jf.to_string(float_format=lambda v: f"{v:.4f}"))
    OUT["ft_by_minute"] = jf.reset_index().astype(str).to_dict("records")

    _p("\nby period:")
    sp = rl.groupby("period").agg(n=("outcome", "size"),
                                  sim=("outcome", lambda x: float((x == 0).mean())))
    bp = ev.groupby("period").agg(n=("o", "size"), act=("o", "mean"))
    _p(sp.join(bp, lsuffix="_sim", rsuffix="_act").to_string(float_format=lambda v: f"{v:.4f}"))

    p = outd / "g4_report.json"
    p.write_text(json.dumps(OUT, indent=1, default=str), encoding="utf-8")
    _p(f"\nwrote {p}")


if __name__ == "__main__":
    main()
