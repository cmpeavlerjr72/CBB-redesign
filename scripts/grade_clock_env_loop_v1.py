"""grade_clock_env_loop_v1.py -- the ONE closed-loop grader of clock round 10 (docs/models/clock/experiments.md
section 43: closed-loop test of the scoring-environment AFT `E1`, registered 0ed3d18 before any E1 loop ran).

Opus engine worker, 2026-10-13. Written and COMMITTED BEFORE the E1 closed-loop run. Adopts nothing.

Runs (all `scripts/run_engine_window_v2.py`, served stack v3, fold 2 / 2025, the verified 500-game stride sample
`data/processed/truth/stride500_verified_minswap_v1_F2_2025.parquet`, trajectories for every seed, games under
`results/clock_env10/loop/<tag>/`, trajectories under `results/trajectories/<tag>/`):
  control  `clk10_ctrl_o0_s<S>`            no ENGINE_* flag (served v3, clock K2), seeds 0..S-1
  draws    `clk10_ctrl_o{1000..4000}_s<S>` the same at four disjoint seed offsets (the section-43 floor draws)
  arm      `clk10_E1_o0_s<S>`              ENGINE_CLOCK=v5b_r8K2_envE1_glat_pmean only, seeds 0..S-1 (paired)
Dispersion summaries: `scripts/diag_margin_dispersion_v2.py` (v1's computations, parametrised) on each run's own
extract (`diag_c4_extract_v1.py --traj-file`), `results/clock_env10/dispersion/<tag>/summary.json`.

Line machinery is IMPORTED, not re-written: `grade_endgame_v1` (LoopLines: G1, half shares, G5, G9, D1, D2 env/gap/total,
lead changes, tie/OT rate, MD lines; per-game frames from trajectories) and `grade_clock_r8_loop_v1` (tempo elasticity
ratio, team pace quintile slope, within-game possessions on OREB / FGM / TOV / FTA vs the actual residual regression).
Added here (section 43 names them; none existed as a loop line): first-half game-possession slope on env; time-of-decision
slope on |spread|; per-quintile tables; per start type sim duration; per-team table; per-game |possessions sim - real|;
paired-bootstrap replicas of the dispersion-gate ratios (for their floors only; the gate VALUES are the script's).

Floor per line (section 43) = max(2 x SD of the four control draws (ddof 1), 2 x paired game-bootstrap SE of
(arm - control)); bootstrap 200 draws over sample games, numpy seed 12345 (`grade_endgame_v1.boot_weights`).
"Regression" of a veto line = |arm - target| - |control - target| > floor.

Decision (section 43, literal): E1 is a CANDIDATE iff
  (P1) sim game-possession slope on env moves toward the actual by more than its floor, AND
  (P2) game-total slope on env moves toward the actual by more than its floor, AND every hard veto passes:
  V-list (41.A.6): G1 possession mean and SD; first- and second-half points share; G5 margin and total SD ratios (target 1);
       G9 margin bias (0) and MC-corrected slope (1); tempo elasticity ratio (1) and team pace slope (1);
       OREB / FGM / TOV / FTA within-game possession slopes (target = actual residual slope); OT (= regulation tie) rate
       (actual); D1 home-margin bias on non-neutral games (target 0; "within its seed SE" read as not regressing beyond
       the floor, whose draw part IS the seed noise) and the eFG home-minus-away gap (actual).
       G9 total bias is REPORTED as priced exposure by channel (H1, H2+OT), per Decision 11, never a disqualifier.
       D3 period-end lines are REPORTED (H2 possessions ending in the last 4:00 = MD4; H1 buzzer PPP).
  DISP (41.0 item 1, values from diag_margin_dispersion_v2 on each run; baseline = the paired control on this sample,
       because the 5,710 x 50 baselines 1.047 / 1.126 ... are full-size values and the sample's own control differs by
       game sampling): (D-1) regulation-margin SD ratio overall and by |spread| quintile must not move further from 1.0
       than the control by more than its band, band = max(0.009, floor) (the 0.009 seed band is the full-size value; the
       literal 0.009-only reading is reported alongside); (D-2) pace SD ratio overall inside [0.95, 1.05] AND not further
       from 1.0 than the control by more than its floor overall or in any quintile; (D-3) regulation total SD ratio must
       HOLD: |arm - control| <= max(0.020, floor) (0.020 = half-width of the full-size baseline CI [0.974, 1.014]);
       absolute [0.974, 1.014] reported; (D-4) pace share of margin variance REPORTED.
  H1: the first-half game-possession slope on env must move toward the actual (|arm - act| < |control - act|);
       the improvement in floors is reported, and "toward but within one floor" is flagged for the PM.
  REG: the overall regulation-margin SD ratio (dispersion 4b) may not RISE beyond its floor (arm - control > floor fails).
Reported, not selecting: game-possession slope on gap; per-quintile total gap (env and home-rating quintiles); lead-change
and time-of-decision slopes on |spread|.
Seed-count check (section 43): a deciding line (P1, P2, H1) is RESOLVED at S seeds when 2 x draw SD < |control - actual|.
If any is unresolved, every run is repeated at 50 seeds and the 50-seed read is the one used (relabelled s50).
Cells under 150 games are labelled UNDERPOWERED (multi-level tables).

    set CBB_TRUTH=verified_v1
    .venv/Scripts/python.exe scripts/grade_clock_env_loop_v1.py --seeds 25
writes results/clock_env10/grade_s<S>.json and results/clock_env10/tables_s<S>.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CBB_TRUTH", "verified_v1")
if os.environ["CBB_TRUTH"] != "verified_v1":
    raise SystemExit("set CBB_TRUTH=verified_v1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import diag_c4_lib_v1 as L  # noqa: E402
import grade_clock_r8_loop_v1 as R8  # noqa: E402
import grade_endgame_v1 as EG  # noqa: E402

OUT = ROOT / "results/clock_env10"
LOOP_DIR = OUT / "loop"
DISP_DIR = OUT / "dispersion"
TRAJ_DIR = ROOT / "results/trajectories"
SAMPLE = EG.SAMPLE
EG.LOOP_DIR = LOOP_DIR
R8.EV = LOOP_DIR
N_BOOT = EG.N_BOOT
UNDER = 150
START_LEVELS = ("period_start", "DREB", "TOV", "made_FG", "made_FT", "other")


# ---------------------------------------------------------------------------------------------------------------------
# per-game extras (sim side from the raw trajectory; real side from the possession-table std frame)
# ---------------------------------------------------------------------------------------------------------------------
def sim_extras(t: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    t = t.sort_values(["game_id", "seed", "poss_idx"], kind="stable").reset_index(drop=True)
    gid, seed, per = t["game_id"].to_numpy(np.int64), t["seed"].to_numpy(np.int64), t["period"].to_numpy(np.int64)
    key = gid * 1000 + seed
    kp = key * 10 + per
    first = np.r_[True, kp[1:] != kp[:-1]]
    end = t["end_type"].to_numpy(np.int64)
    start = np.where(first, 0, np.r_[0, end[:-1]])          # engine: prev_end within the period, period_start at open
    reg = per <= 2
    dur = t["duration_s"].to_numpy(float)
    f = pd.DataFrame({"game_id": gid})
    for j, lv in enumerate(START_LEVELS):
        m = reg & (start == j)
        f[f"st_{lv}_n"] = m.astype(float)
        f[f"st_{lv}_d"] = np.where(m, dur, 0.0)
    f["h1_poss"] = (per == 1).astype(float)
    out = f.groupby("game_id").sum()
    d = L.std_from_sim(t)
    gs = L.game_seed(d)
    r = d[d["period"] <= 2]
    regtot = r.groupby(["game_id", "seed"])["pts"].sum().rename("rt").reset_index()
    out["tdec_sum"] = gs.groupby("game_id")["t_decision"].sum()
    out["regtot_sum"] = regtot.groupby("game_id")["rt"].sum()
    out["regtot_sq"] = (regtot["rt"] ** 2).groupby(regtot["game_id"]).sum()
    return out.reindex(ids)


def real_extras(rstd: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    gs = L.game_seed(rstd.assign(seed=0)).set_index("game_id")
    out = pd.DataFrame(index=ids)
    out["h1_poss"] = rstd[rstd["period"] == 1].groupby("game_id").size().reindex(ids)
    out["tdec"] = gs["t_decision"].reindex(ids)
    out["regtot"] = rstd[rstd["period"] <= 2].groupby("game_id")["pts"].sum().reindex(ids)
    return out


def real_start_types(ids: np.ndarray) -> pd.DataFrame:
    """Real regulation possession duration by start type on the sample games: the clock design (r6_L2 design_v2,
    the table every clock law is fitted and graded on)."""
    d = pd.read_parquet(ROOT / "data/processed/models/clock/r6_L2/design_v2.parquet",
                        columns=["game_id", "season", "period", "prev_end", "duration_s"])
    d = d[(d["season"] == 2025) & (d["period"] <= 2) & d["game_id"].isin(set(ids.tolist()))]
    return d


# ---------------------------------------------------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------------------------------------------------
class Lines:
    def __init__(self, ids, real, pgi):
        self.ids = ids
        self.LL = EG.LoopLines(ids, real, pgi)
        self.LL.r = EG.add_seed_level(self.LL.r, real["std"].assign(seed=0), ids)
        self.rx = real_extras(real["std"], ids)
        self.pgi = pgi
        self.T, self.tg = R8.truth_frame()
        self.T = self.T.loc[[g for g in ids if g in self.T.index]]
        pc = ((self.T["nt"] == 2) & self.T["pbp_complete"].fillna(False).astype(bool) & self.T["cnt_v4"].notna()
              & self.T["X"].notna())
        self.T = self.T[pc]
        # quintile edges over EVERY fold-2 game (the D2 / dispersion population), applied to the sample
        allpg = pd.read_parquet(ROOT / "results/diag_c4/f2all50_pregame.parquet")
        s = int(allpg["_rating_sign"].dropna().iloc[0])
        env_all = (allpg["home_off_c"] + allpg["away_off_c"] - s * (allpg["home_def_c"] + allpg["away_def_c"])).to_numpy(float)
        self.env_edges = np.nanquantile(env_all, [0.2, 0.4, 0.6, 0.8])
        self.home_edges = np.nanquantile(allpg["home_net"].to_numpy(float), [0.2, 0.4, 0.6, 0.8])
        self.envq = np.where(np.isfinite(self.LL.env), np.searchsorted(self.env_edges, self.LL.env, side="right"), -1)
        hn = pgi["home_net"].to_numpy(float)
        self.homeq = np.where(np.isfinite(hn), np.searchsorted(self.home_edges, hn, side="right"), -1)
        # |spread| quintile for the dispersion replicas: as diag_margin_dispersion (qcut over sample games with a close)
        ap = self.LL.absp
        self.dq = pd.qcut(pd.Series(ap), 5, labels=False).to_numpy()
        self.sign = s

    def lines(self, pg, gm, sx, g8, w) -> dict:
        LL, r, ok = self.LL, self.LL.r, self.LL.ok
        out = LL.lines(pg, gm, w)
        S = pg["n_seeds"].to_numpy(float)
        Sc = float(np.nanmax(S))
        # ---- round-8 lines (own truth frame T, the bootstrap weights carried over by game)
        wT = pd.Series(w, index=self.ids).reindex(self.T.index).fillna(0).to_numpy()
        r8 = R8.lines(g8.loc[self.T.index], self.T, self.tg, wT)
        for k in ("elast_ratio", "elast_sim", "elast_act", "slope_teamq", "slope_gameq", "count_cal_slope",
                  "g1_count_minus_v4", "poss_sd_ratio_within_over_resid", "ot_rate"):
            out[f"r8_{k}"] = r8[k]
        for c in ("oreb", "fgm", "tov", "fta"):
            out[f"r8_N_on_{c}"] = r8[f"sim_N_on_{c}"]
            out[f"act_r8_N_on_{c}"] = r8[f"act_N_on_{c}"]

        def wm(x, m):
            m = m & np.isfinite(x)
            return float((w[m] * x[m]).sum() / max(w[m].sum(), 1e-12))
        okr = ok & np.isfinite(LL.env)
        # ---- H1 game-possession slope on env
        sh1 = sx["h1_poss"].to_numpy(float) / S
        rh1 = self.rx["h1_poss"].to_numpy(float)
        out["H1_poss_on_env"] = LL._wslope(LL.env, sh1, w, okr)
        out["act_H1_poss_on_env"] = LL._wslope(LL.env, rh1, w, okr)
        # ---- time of decision on |spread| (reported)
        oks = ok & np.isfinite(LL.absp)
        out["D2_tdec_on_spread"] = LL._wslope(LL.absp, sx["tdec_sum"].to_numpy(float) / S, w, oks)
        out["act_D2_tdec_on_spread"] = LL._wslope(LL.absp, self.rx["tdec"].to_numpy(float), w, oks)
        # ---- per-game |possessions sim - real| (regulation, both teams)
        sp = (pg["poss_reg_h"].to_numpy() + pg["poss_reg_a"].to_numpy()) / S
        rp = r["poss_reg_h"].to_numpy() + r["poss_reg_a"].to_numpy()
        out["pergame_abs_poss_err"] = wm(np.abs(sp - rp), ok)
        # ---- quintile tables (env and home rating): possessions and totals, sim and real, gap = real - sim
        tmu = pg["total"].to_numpy() / S
        for nm, q in (("envq", self.envq), ("homeq", self.homeq)):
            for k in range(5):
                m = ok & (q == k)
                out[f"{nm}{k + 1}_poss_sim"] = wm(sp, m)
                out[f"{nm}{k + 1}_poss_gap"] = wm(rp - sp, m)
                mt = (q == k) & np.isfinite(LL.ytot)
                out[f"{nm}{k + 1}_total_sim"] = wm(tmu, mt)
                out[f"{nm}{k + 1}_total_gap"] = wm(LL.ytot - tmu, mt)
        # ---- start type: sim regulation duration per possession
        for lv in START_LEVELS:
            n = sx[f"st_{lv}_n"].to_numpy(float)
            out[f"st_{lv}_sec"] = float((w * sx[f"st_{lv}_d"].to_numpy(float)).sum() / max((w * n).sum(), 1e-12))
        # ---- dispersion-gate replicas (paired floors): regulation margin, pace, regulation total
        okd = ok & np.isfinite(LL.absp)

        def disp(sum_, sq_, real_, mask, div=1.0):
            mu = sum_ / S
            s2 = (sq_ - S * mu ** 2) / (S - 1) / div ** 2
            rr = (real_ - mu) / div
            m = mask & np.isfinite(rr)
            sv = (w[m] * s2[m]).sum() / w[m].sum()
            rv = (w[m] * rr[m] ** 2).sum() / w[m].sum() - sv / Sc
            return float(np.sqrt(sv / max(rv, 1e-12)))
        rm_sum, rm_sq = pg["reg_margin"].to_numpy(float), pg["reg_margin_sq"].to_numpy(float)
        rreg = r["reg_margin"].to_numpy(float)
        ps, pq = pg["poss_reg"].to_numpy(float), pg["poss_sq"].to_numpy(float)
        rpos = r["poss_reg"].to_numpy(float)
        ts, tq = sx["regtot_sum"].to_numpy(float), sx["regtot_sq"].to_numpy(float)
        rtot = self.rx["regtot"].to_numpy(float)
        out["rep_regmargin_ratio"] = disp(rm_sum, rm_sq, rreg, ok)
        out["rep_pace_ratio"] = disp(ps, pq, rpos, ok, 2.0)
        out["rep_regtotal_ratio"] = disp(ts, tq, rtot, ok)
        for k in range(5):
            m = okd & (self.dq == k)
            out[f"rep_regmargin_q{k + 1}"] = disp(rm_sum, rm_sq, rreg, m)
            out[f"rep_pace_q{k + 1}"] = disp(ps, pq, rpos, m, 2.0)
        return out


def disp_values(tag: str) -> dict:
    s = json.loads((DISP_DIR / tag / "summary.json").read_text(encoding="utf-8"))
    c, q = s["components"], s["part4c"]
    out = {"disp_regmargin": c["regulation margin"]["ratio"], "disp_pace": c["regulation possessions per team (pace)"]["ratio"],
           "disp_regtotal": c["regulation total points"]["ratio"], "disp_finalmargin": c["final margin (OT incl.)"]["ratio"],
           "disp_pace_share": s["share_pace_eff"]["pace (possession-count latent) x expected PPP gap"]}
    for k in range(1, 6):
        out[f"disp_regmargin_q{k}"] = q[f"spread_Q{k}"]["reg_margin"]["ratio"]
        out[f"disp_pace_q{k}"] = q[f"spread_Q{k}"]["pace"]["ratio"]
        out[f"disp_n_q{k}"] = q[f"spread_Q{k}"]["pace"]["n"]
    out["disp_n_all"] = c["regulation margin"]["n"]
    return out


#: dispersion value key -> replica key (the replica supplies only the paired-bootstrap SE of the difference)
DISP_REP = {"disp_regmargin": "rep_regmargin_ratio", "disp_pace": "rep_pace_ratio", "disp_regtotal": "rep_regtotal_ratio",
            **{f"disp_regmargin_q{k}": f"rep_regmargin_q{k}" for k in range(1, 6)},
            **{f"disp_pace_q{k}": f"rep_pace_q{k}" for k in range(1, 6)}}

VETO = [("G1_poss_mean", "act_G1_poss_mean"), ("G1_poss_sd", "act_G1_poss_sd"),
        ("half1_share", "act_half1_share"), ("half2_share", "act_half2_share"),
        ("G5_margin_sd_ratio", 1.0), ("G5_total_sd_ratio", 1.0), ("G9_margin_bias", 0.0), ("G9_margin_slope", 1.0),
        ("r8_elast_ratio", 1.0), ("r8_slope_teamq", 1.0),
        ("r8_N_on_oreb", "act_r8_N_on_oreb"), ("r8_N_on_fgm", "act_r8_N_on_fgm"), ("r8_N_on_tov", "act_r8_N_on_tov"),
        ("r8_N_on_fta", "act_r8_N_on_fta"), ("tie_rate", "act_tie_rate"),
        ("D1_home_margin_bias", 0.0), ("D1_efg_gap", "act_D1_efg_gap")]
PRIMARY = [("D2_poss_on_env", "act_D2_poss_on_env"), ("D2_total_on_env", "act_D2_total_on_env")]
REPORTED = [("D2_poss_on_gap", "act_D2_poss_on_gap"), ("D2_leadchg_on_spread", "act_D2_leadchg_on_spread"),
            ("D2_tdec_on_spread", "act_D2_tdec_on_spread"), ("G9_total_bias", 0.0), ("G9_total_bias_H1", 0.0),
            ("G9_total_bias_H2OT", 0.0), ("MD4_h2_last4_poss", "act_MD4_h2_last4_poss"), ("h1buz_ppp", "act_h1buz_ppp"),
            ("r8_ot_rate", "act_tie_rate"), ("r8_slope_gameq", 1.0), ("r8_count_cal_slope", 1.0),
            ("r8_g1_count_minus_v4", 0.0), ("pergame_abs_poss_err", 0.0), ("MD6_margin_sd_ratio", 1.0)]


def tgt(t, act):
    return act[t] if isinstance(t, str) else t


def toward(row, tv):
    e0, e1 = abs(row["ctrl"] - tv), abs(row["arm"] - tv)
    return {"target": tv, "ctrl": row["ctrl"], "arm": row["arm"], "floor": row["floor"], "improve": e0 - e1,
            "floors": (e0 - e1) / row["floor"] if row["floor"] else None,
            "toward_by_floor": bool(e0 - e1 > row["floor"]), "away_by_floor": bool(e1 - e0 > row["floor"])}


def judge(rows: dict, act: dict) -> dict:
    v = {"primary": {k: toward(rows[k], tgt(t, act)) for k, t in PRIMARY}}
    v["primary_moves"] = all(x["toward_by_floor"] for x in v["primary"].values())
    vet = {}
    for k, t in VETO:
        x = toward(rows[k], tgt(t, act))
        vet[k] = {**x, "pass": not x["away_by_floor"]}
    # dispersion gates
    d = {}
    for k in ["disp_regmargin"] + [f"disp_regmargin_q{i}" for i in range(1, 6)]:
        x = toward(rows[k], 1.0)
        band = max(0.009, rows[k]["floor"])
        away = -x["improve"]
        d[k] = {**x, "band": band, "pass": bool(away <= band), "pass_literal_0009": bool(away <= 0.009)}
    for k in ["disp_pace"] + [f"disp_pace_q{i}" for i in range(1, 6)]:
        x = toward(rows[k], 1.0)
        d[k] = {**x, "pass": not x["away_by_floor"]}
    d["disp_pace"]["in_095_105"] = bool(0.95 <= rows["disp_pace"]["arm"] <= 1.05)
    d["disp_pace"]["pass"] = bool(d["disp_pace"]["pass"] and d["disp_pace"]["in_095_105"])
    r = rows["disp_regtotal"]
    band = max(0.020, r["floor"])
    d["disp_regtotal"] = {"ctrl": r["ctrl"], "arm": r["arm"], "floor": r["floor"], "band": band,
                          "delta": r["arm"] - r["ctrl"], "pass": bool(abs(r["arm"] - r["ctrl"]) <= band),
                          "arm_in_0974_1014": bool(0.974 <= r["arm"] <= 1.014),
                          "ctrl_in_0974_1014": bool(0.974 <= r["ctrl"] <= 1.014)}
    d["disp_pace_share"] = {"ctrl": rows["disp_pace_share"]["ctrl"], "arm": rows["disp_pace_share"]["arm"],
                            "reported": True}
    vet.update({k: x for k, x in d.items() if "pass" in x})
    # H1 and the overall regulation-margin SD ratio rise
    h1 = toward(rows["H1_poss_on_env"], act["act_H1_poss_on_env"])
    vet["H1_poss_on_env"] = {**h1, "pass": bool(h1["improve"] > 0),
                             "flag_within_one_floor": bool(0 < h1["improve"] <= h1["floor"])}
    rr = rows["disp_regmargin"]
    vet["REG_regmargin_sd_rise"] = {"ctrl": rr["ctrl"], "arm": rr["arm"], "floor": rr["floor"],
                                    "delta": rr["arm"] - rr["ctrl"], "pass": bool(rr["arm"] - rr["ctrl"] <= rr["floor"])}
    v["vetoes"] = vet
    v["vetoes_failing"] = [k for k, x in vet.items() if not x["pass"]]
    v["reported"] = {k: {**toward(rows[k], tgt(t, act))} for k, t in REPORTED}
    v["candidate"] = bool(v["primary_moves"] and not v["vetoes_failing"])
    return v


def seed_check(ctrl: dict, act: dict, draw_sd: dict) -> dict:
    out = {}
    for k, t in [*PRIMARY, ("H1_poss_on_env", "act_H1_poss_on_env")]:
        gap = abs(ctrl[k] - act[t])
        out[k] = {"floor_draws": 2 * draw_sd[k], "gap": gap, "resolved": bool(2 * draw_sd[k] < gap)}
    out["all_resolved"] = all(v["resolved"] for v in out.values() if isinstance(v, dict))
    return out


# ---------------------------------------------------------------------------------------------------------------------
def team_table(ids, pgi, real, sims: dict) -> list[dict]:
    """Per offence team: team-level mean x_off = off_c(offence) + def_c(opponent) (D2 sign) over its sample team-games;
    team quintiles; regulation possessions per team-game (= per 40 min) sim vs real; right-sign share of arm - control
    against real - control."""
    s = int(pgi["_rating_sign"].dropna().iloc[0])
    rows = []
    for side, o, dpre in ((0, "home", "away"), (1, "away", "home")):
        rows.append(pd.DataFrame({"game_id": ids, "off_side": side, "team": pgi[f"{o}_team_id"].to_numpy(),
                                  "x_off": (pgi[f"{o}_off_c"] - s * pgi[f"{dpre}_def_c"]).to_numpy(float)}))
    tg = pd.concat(rows, ignore_index=True)
    rtg = L.team_game(real["std"].assign(seed=0)).set_index(["game_id", "off_side"])["poss"]
    tg["real"] = rtg.reindex(pd.MultiIndex.from_arrays([tg["game_id"], tg["off_side"]])).to_numpy(float)
    okg = pd.Series(real["pbp_ok"].reindex(ids).fillna(False).to_numpy(), index=ids)
    tg = tg[tg["game_id"].map(okg).astype(bool) & np.isfinite(tg["x_off"]) & np.isfinite(tg["real"])]
    for nm, stg in sims.items():
        sm = stg.groupby(["game_id", "off_side"])["poss"].mean()
        tg[nm] = sm.reindex(pd.MultiIndex.from_arrays([tg["game_id"], tg["off_side"]])).to_numpy(float)
    team = tg.groupby("team").agg(x_off=("x_off", "mean"), n=("x_off", "size"), real=("real", "mean"),
                                  **{nm: (nm, "mean") for nm in sims})
    team["q"] = pd.qcut(team["x_off"], 5, labels=False)
    out = []
    for q in range(5):
        t = team[team["q"] == q]
        tgq = tg[tg["team"].isin(t.index)]
        r = {"quintile": q + 1, "n_teams": int(len(t)), "n_team_games": int(len(tgq)),
             "x_off_mean": float(t["x_off"].mean()), "real": float(tgq["real"].mean())}
        for nm in sims:
            r[nm] = float(tgq[nm].mean())
        if "arm" in sims and "ctrl" in sims:
            sgn = np.sign(t["arm"] - t["ctrl"]) == np.sign(t["real"] - t["ctrl"])
            r["right_sign_share"] = float(sgn.mean())
        r["underpowered"] = bool(len(tgq) < UNDER)
        out.append(r)
    if "arm" in sims and "ctrl" in sims:
        sgn = np.sign(team["arm"] - team["ctrl"]) == np.sign(team["real"] - team["ctrl"])
        out.append({"quintile": "ALL", "n_teams": int(len(team)), "n_team_games": int(len(tg)),
                    "right_sign_share": float(sgn.mean()),
                    "abs_err_down_share": float(((team["arm"] - team["real"]).abs() < (team["ctrl"] - team["real"]).abs()).mean())})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--arm", default="E1")
    a = ap.parse_args()
    S = a.seeds
    control = f"clk10_ctrl_o0_s{S}"
    draws = [f"clk10_ctrl_o{o}_s{S}" for o in (1000, 2000, 3000, 4000)]
    arm = f"clk10_{a.arm}_o0_s{S}"
    tags = [control, *draws, arm]
    ids = np.sort(pd.read_parquet(SAMPLE)["game_id"].astype("int64").to_numpy())
    real = EG._real_side(ids)
    pgi = EG._pregame(ids)
    LN = Lines(ids, real, pgi)
    frames, info, stgs = {}, {}, {}
    for t in tags:
        pg, gm, inf = EG.run_frame(t, ids)
        tr = pd.read_parquet(TRAJ_DIR / t / "trajectory.parquet")
        tr = tr[tr["game_id"].isin(set(ids.tolist()))]
        pg = EG.add_seed_level(pg, L.std_from_sim(tr), ids)
        sx = sim_extras(tr, ids)
        stgs[t] = L.team_game(L.std_from_sim(tr))
        del tr
        g8 = R8.per_game(R8.load_run(t))
        frames[t] = (pg, gm, sx, g8)
        info[t] = inf
        if inf["n_seeds"] != S:
            raise SystemExit(f"{t}: {inf['n_seeds']} seeds, expected {S}")
        print(f"loaded {t}: {inf['n_seeds']} seeds, flags {inf['flags']}", flush=True)
    if info[arm]["seeds"] != info[control]["seeds"]:
        raise SystemExit("arm and control seeds differ (pairing broken)")
    G = len(ids)
    one = np.ones(G)
    val = {t: {**LN.lines(*frames[t], one), **disp_values(t)} for t in tags}
    act = {k: v for k, v in val[control].items() if k.startswith("act_")}
    names = [k for k in val[control] if not k.startswith("act_")]
    draw_sd = {k: float(np.nanstd([val[t][k] for t in draws], ddof=1)) for k in names}
    wb = EG.boot_weights(G)
    diffs = {k: [] for k in names}
    for b in range(N_BOOT):
        la, lc = LN.lines(*frames[arm], wb[b]), LN.lines(*frames[control], wb[b])
        for k in names:
            src = DISP_REP.get(k, k)
            if src in la:
                diffs[k].append(la[src] - lc[src])
    rows = {}
    for k in names:
        se = float(np.nanstd(diffs[k], ddof=1)) if diffs[k] else float("nan")
        fl = float(np.nanmax([2 * draw_sd[k], 2 * se]))
        rows[k] = {"arm": val[arm][k], "ctrl": val[control][k], "delta": val[arm][k] - val[control][k],
                   "draw_sd": draw_sd[k], "boot_se": se, "floor": fl}
    res = {"spec": "docs/models/clock/experiments.md section 43", "seeds": S, "control": control, "draws": draws,
           "arm": arm, "n_games": int(G), "n_pbp_ok": int(LN.LL.ok.sum()), "n_r8_games": int(len(LN.T)),
           "flags": {t: info[t]["flags"] for t in tags}, "actual": act, "rows": rows,
           "draw_lines": {t: val[t] for t in draws}}
    res["seed_check"] = seed_check(val[control], act, draw_sd)
    res["verdict"] = judge(rows, act)
    # multi-level tables
    rs = real_start_types(ids)
    st_real = rs.groupby("prev_end")["duration_s"].agg(["mean", "size"])
    st_games = rs.groupby("prev_end")["game_id"].nunique()
    res["start_type"] = {lv: {"real_sec": float(st_real["mean"].get(lv, np.nan)), "real_rows": int(st_real["size"].get(lv, 0)),
                              "games": int(st_games.get(lv, 0)),
                              "ctrl_sec": rows[f"st_{lv}_sec"]["ctrl"], "arm_sec": rows[f"st_{lv}_sec"]["arm"],
                              "floor": rows[f"st_{lv}_sec"]["floor"]} for lv in START_LEVELS}
    res["quintiles"] = {}
    for nm, q in (("envq", LN.envq), ("homeq", LN.homeq)):
        tab = []
        for k in range(5):
            m = LN.LL.ok & (q == k)
            rp = (LN.LL.r["poss_reg_h"] + LN.LL.r["poss_reg_a"]).to_numpy(float)
            n_t = int(((q == k) & np.isfinite(LN.LL.ytot)).sum())
            tab.append({"q": k + 1, "n_games_poss": int(m.sum()), "n_games_total": n_t,
                        "underpowered": bool(m.sum() < UNDER),
                        "real_poss": float(np.nanmean(rp[m])),
                        "real_total": float(np.nanmean(LN.LL.ytot[(q == k) & np.isfinite(LN.LL.ytot)])),
                        **{f"{c}_{x}": rows[f"{nm}{k + 1}_{c}"][x] for c in ("poss_sim", "poss_gap", "total_sim", "total_gap")
                           for x in ("ctrl", "arm", "floor")}})
        res["quintiles"][nm] = tab
    res["team_table"] = team_table(ids, pgi, real, {"ctrl": stgs[control], "arm": stgs[arm]})
    res["written_at"] = datetime.now(UTC).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"grade_s{S}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    v = res["verdict"]
    print("CANDIDATE" if v["candidate"] else "NOT A CANDIDATE", "| primary_moves", v["primary_moves"],
          "| vetoes failing", v["vetoes_failing"])
    print("seed check:", json.dumps(res["seed_check"]))
    for k, x in v["primary"].items():
        print(f"  {k}: act {x['target']:.4f} ctrl {x['ctrl']:.4f} arm {x['arm']:.4f} floor {x['floor']:.4f} floors {x['floors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
