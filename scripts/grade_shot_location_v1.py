"""grade_shot_location_v1.py -- the ONE blind grader for shot-location round 1 (experiments.md section 1): Reads A, B, C.

Blinding: within each read and fold the arms found on disk are shuffled under a fixed RNG into codes X1..Xn; every
metric, floor and comparison is computed per code; the code -> label map is applied only when the final tables are
written (`reveal`). The decision rule (section 1.6) is applied after the reveal, on F2; F1 is reported.

Read A (attempt level, located rows; make and and-one: S1-monthly fg_make wrapper runs; rebound: S0 wrapper runs):
  log loss; decile battery (worst gap, level, shape) via FG.score / RB.score; floor = max(|LL(R) - LL(R_seed)|, paired
  game-block bootstrap SE of LL(arm) - LL(R), 200 reps); "decile gate held" = the arm's worst decile gap is not worse
  than R's by more than max(|gap(R) - gap(R_seed)|, 0.10 pp) (the fg_make PAE-round convention, because R itself does
  not clear the absolute gate on every cell); responsiveness lines: predicted vs realised rate by band, and by quintile
  of the offence's as-of band share; segments site / month / offence tier / located-share-of-game quartile, cells
  under 5,000 attempts (make, foul) or 2,000 opportunities (rebound) UNDERPOWERED.
Read B (located attempts): multinomial log loss per class; B1 vs B0, B2 vs B1, B3 vs B2, B2alt vs B2, paired
  game-block bootstrap SE (the seed floor is zero: deterministic fits); team-game realised vs predicted band shares
  (MAE; SD of the z-residual against the multinomial SD); responsiveness by quintile of off_band_c and def_band_c
  (Decision-8 battery, low-span exemption, monotone 4 of 4), slope also shown divided by sqrt(registered reliability).
Read C (team-game, ALL attempts): implied = sum_b P_B(b | row) x P_A1(outcome | row, b) against the class-level pair
  R; and-one marginalises over P(b | made) and OREB over P(b | miss) (Bayes with the A1 make model); MAE vs realised,
  paired team-game bootstrap SE of the MAE difference, dispersion, responsiveness by quintile of the offence's and the
  defence's as-of band profile; all games and located-only games (>= 95% of the class's attempts located) side by side.

    .venv/Scripts/python.exe scripts/grade_shot_location_v1.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.models import prob_metrics as PM  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402

R1 = ROOT / "data/processed/models/shot_location/round1"
DESIGN = ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet"
REB_DESIGN = ROOT / "data/processed/models/rebound/round3/design_round3.parquet"
N_BOOT, BOOT_SEED = 200, 12345
CALIB_MIN_FLOOR_PP = 0.10
BANDS = {"FGA_jump2": ("short2", "long2"), "FGA_3": ("corner3", "wing3", "top3")}
PROF = {"FGA_jump2": ("short2",), "FGA_3": ("corner3", "wing3", "top3")}
MT = {"FGA_jump2": "jump2", "FGA_3": "three"}
REL = {("off", "short2"): .89, ("off", "corner3"): .79, ("off", "wing3"): .67, ("off", "top3"): .80,
       ("def", "short2"): .69, ("def", "corner3"): .76, ("def", "wing3"): .52, ("def", "top3"): .75}
POWER = {"ACC", "Big Ten", "Big 12", "SEC", "Big East", "Pac-12"}
MID = {"A-10", "American", "Mountain West", "WCC", "MVC"}
MONTHS = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar+Apr", 4: "Mar+Apr"}
FOLDS = ("F1", "F2")
TEST = {"F1": 2024, "F2": 2025}


# ============================================================================ helpers
def boot_mean(games, d, n_rep=N_BOOT, seed=BOOT_SEED):
    g = pd.DataFrame({"g": games, "d": d}).groupby("g")["d"].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    reps = np.empty(n_rep)
    for r in range(n_rep):
        k = rng.integers(0, len(s), len(s))
        reps[r] = s[k].sum() / n[k].sum()
    return float(reps.std(ddof=1))


def ll_bin(y, p):
    return -np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-12, 1.0))


def ll_multi(y, P):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))


def d8(span_act, steps, slope):
    r = {"span_actual": span_act, "pred_monotone_steps": steps, "slope_ratio": slope}
    return FG.decision8_verdict({"x": r}, reading="low_span_exempt")["by_driver"]["x"]


def rate_resp(driver, pred, act, w=None, n_q=5):
    """Quintile line for continuous rates (team-game or attempt): mean predicted vs realised per quintile of the
    driver's RANK; slope ratio = predicted span / realised span; monotone steps against the realised direction."""
    v = FG._rank(np.asarray(driver, dtype="float64"))
    edges = np.quantile(v, np.linspace(0, 1, n_q + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    q = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_q - 1)
    w = np.ones(len(v)) if w is None else np.asarray(w, dtype="float64")
    pa = np.array([np.average(pred[q == b], weights=w[q == b]) for b in range(n_q)])
    aa = np.array([np.average(act[q == b], weights=w[q == b]) for b in range(n_q)])
    sp, sa = float(pa[-1] - pa[0]), float(aa[-1] - aa[0])
    ref = np.sign(sa) if sa else np.sign(sp)
    steps = int(np.sum(np.sign(np.diff(pa)) == ref))
    slope = round(sp / sa, 4) if sa else None
    v8 = d8(sa, steps, slope)
    return {"pred": [round(x, 5) for x in pa], "actual": [round(x, 5) for x in aa], "span_pred": round(sp, 5),
            "span_actual": round(sa, 5), "slope_ratio": slope, "steps": steps, "d8": v8, "pass": bool(v8["pass"])}


def codes_for(labels, rng):
    perm = rng.permutation(len(labels))
    return {lab: f"X{perm[i] + 1}" for i, lab in enumerate(labels)}


def seg_table(te: pd.DataFrame, d: np.ndarray, games: np.ndarray, powered: int) -> dict:
    out = {}
    for seg in ("site", "month", "tier", "locq"):
        for v in sorted(te[seg].astype(str).unique()):
            m = (te[seg].astype(str) == v).to_numpy()
            out[f"{seg}={v}"] = {"n": int(m.sum()), "gain": float(-d[m].mean()), "boot_se": boot_mean(games[m], d[m]),
                                 "powered": bool(m.sum() >= powered)}
    return out


# ============================================================================ context
def load_context() -> pd.DataFrame:
    des = pd.read_parquet(DESIGN, columns=["game_id", "cbbd_game_id", "season", "game_date", "shot_class", "y", "made",
                                           "and_one", "shooter_id", "offense_is_home", "neutral_site", "off_team_id",
                                           "def_team_id", "chance_number", "shooter_make_c", "shooter_att_c",
                                           "def_allow_c", "def_att_prior"])
    loc = pd.read_parquet(R1 / "design_loc_cols_v1.parquet")
    assert (loc["game_id"].to_numpy() == des["game_id"].to_numpy()).all()
    for c in loc.columns:
        if c not in des.columns:
            des[c] = loc[c].to_numpy()
    des["site"] = np.where(des["neutral_site"].astype(bool), "neutral",
                           np.where(des["offense_is_home"].astype(bool), "home", "away"))
    des["month"] = pd.to_datetime(des["game_date"]).dt.month.map(MONTHS)
    confs = []
    for s in sorted(des["season"].unique()):
        g = pd.read_parquet(ROOT / f"data/raw/cbbd/games_{s}.parquet", columns=["id", "homeConference", "awayConference"])
        confs.append(g.rename(columns={"id": "cbbd_game_id"}))
    cg = pd.concat(confs, ignore_index=True).drop_duplicates("cbbd_game_id")
    m = des[["cbbd_game_id"]].merge(cg, on="cbbd_game_id", how="left")
    oc = np.where(des["offense_is_home"].astype(bool), m["homeConference"], m["awayConference"])
    des["tier"] = np.where(pd.Series(oc).isin(POWER), "power", np.where(pd.Series(oc).isin(MID), "mid", "low"))
    # located share of the game's jumper + three attempts, quartile within season
    jt = des["shot_class"].isin(list(BANDS))
    ls = des[jt].groupby("game_id")["located"].mean()
    des["game_loc_share"] = des["game_id"].map(ls).fillna(0.0)
    des["locq"] = "Q?"
    for s in des["season"].unique():
        ms = (des["season"] == s).to_numpy()
        gs = ls[ls.index.isin(des.loc[ms, "game_id"].unique())]
        r = gs.rank(method="first")
        q = pd.qcut(r, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        des.loc[ms, "locq"] = des.loc[ms, "game_id"].map(q).astype(str).to_numpy()
    return des


# ============================================================================ Read A
def readA_fg(chan: str, ctx: pd.DataFrame, rng) -> dict:
    """chan = make | andone."""
    out = {}
    for fold in FOLDS:
        runs = {"R": ("R", 0), "R_seed": ("R", 1), "A1": ("A1", 0), "A2": ("A2", 0)}
        found = {lab: R1 / chan / f"{a}_s{s}_{fold}" for lab, (a, s) in runs.items()
                 if (R1 / chan / f"{a}_s{s}_{fold}" / f"preds_{fold}_s{s}.parquet").exists()}
        code = codes_for(list(found), rng)
        per = {}
        rows_ref = None
        for lab, dpath in found.items():
            s = runs[lab][1]
            pr = pd.read_parquet(dpath / f"preds_{fold}_s{s}.parquet")
            rw = pd.read_parquet(dpath / f"rows_{fold}_s{s}.parquet")
            assert len(pr) == len(rw)
            if rows_ref is None:
                rows_ref = rw
            assert (rw["fg_row"].to_numpy() == rows_ref["fg_row"].to_numpy()).all()
            per[code[lab]] = pr
        res = {"code_of": code, "by_class": {}}
        for cls in BANDS:
            m = (rows_ref["shot_class"] == cls).to_numpy()
            fr = rows_ref.loc[m, "fg_row"].to_numpy()
            te = ctx.iloc[fr].reset_index(drop=True)
            te["y"] = (te["and_one"] if chan == "andone" else te["made"]).astype("int64").to_numpy()
            y = te["y"].to_numpy()
            g = te["game_id"].to_numpy()
            cm = {}
            for c, pr in per.items():
                pc = pr.loc[m]
                assert (pc["game_id"].to_numpy() == te["game_id"].to_numpy()).all()
                assert (pc["y"].to_numpy() == y).all(), (chan, fold, cls)
                p = pc["p"].to_numpy(dtype="float64")
                sc = FG.score(te, np.column_stack([1 - p, p]))
                cal = sc["calibration"]["MAKE"]
                lines = {"by_band": {}, "by_off_share": {}}
                for b in BANDS[cls]:
                    mb = (te["band"] == b).to_numpy()
                    lines["by_band"][b] = {"n": int(mb.sum()), "pred": float(p[mb].mean()), "actual": float(y[mb].mean())}
                for b in PROF[cls]:
                    lines["by_off_share"][b] = rate_resp(te[f"off_{b}_c"].to_numpy(), p, y.astype(float))
                cm[c] = {"p": p, "llr": ll_bin(y, p), "log_loss": sc["log_loss"], "calib_worst_gap_pp": sc["calib_worst_gap_pp"],
                         "calib_pass": sc["calib_pass"], "calib_level_pp": cal["level_shift_pp"],
                         "calib_shape_pp": cal["max_abs_gap_pp_after_level_shift"], "lines": lines,
                         "pred_rate": float(p.mean()), "actual_rate": float(y.mean())}
            res["by_class"][cls] = {"te": te, "per": cm, "n": int(len(te))}
        out[fold] = res
    return out


def readA_reb(ctx: pd.DataFrame, rng) -> dict:
    out = {}
    rd = pd.read_parquet(REB_DESIGN, columns=["game_id", "season", "game_date", "offense_is_home", "neutral_site",
                                              "off_oreb_c", "opp_def_dreb_c", "miss_type", "y"])
    for fold in FOLDS:
        runs = {"R": ("R", 0), "R_seed": ("R", 1), "A1": ("A1", 0), "A2": ("A2", 0)}
        found = {lab: R1 / "rebound" / f"{a}_s{s}_{fold}" / f"preds_{fold}_s{s}.parquet" for lab, (a, s) in runs.items()}
        found = {k: v for k, v in found.items() if v.exists()}
        code = codes_for(list(found), rng)
        per = {code[lab]: pd.read_parquet(f) for lab, f in found.items()}
        ref = next(iter(per.values()))
        for pr in per.values():
            assert (pr["row"].to_numpy() == ref["row"].to_numpy()).all()
        res = {"code_of": code, "by_class": {}}
        for cls in BANDS:
            m = (ref["miss_type"] == MT[cls]).to_numpy()
            te = rd.iloc[ref.loc[m, "row"].to_numpy()].reset_index(drop=True)
            fr = ref.loc[m, "fg_row"].to_numpy()
            for c in ("band", "tier", "locq", "month", *[f"off_{b}_c" for b in PROF[cls]]):
                te[c] = ctx[c].to_numpy()[fr]
            te["site"] = np.where(te["neutral_site"].astype(bool), "neutral",
                                  np.where(te["offense_is_home"].astype(bool), "home", "away"))
            y = te["y"].to_numpy()
            cm = {}
            for c, pr in per.items():
                P = pr.loc[m, ["p_OREB", "p_DREB", "p_DEAD"]].to_numpy(dtype="float64")
                sc = RB.score(te, P)
                po = P[:, 0]
                yo = (y == 0).astype(float)
                lines = {"by_band": {}, "by_off_share": {}}
                for b in BANDS[cls]:
                    mb = (te["band"] == b).to_numpy()
                    lines["by_band"][b] = {"n": int(mb.sum()), "pred": float(po[mb].mean()), "actual": float(yo[mb].mean())}
                for b in PROF[cls]:
                    lines["by_off_share"][b] = rate_resp(te[f"off_{b}_c"].to_numpy(), po, yo)
                cm[c] = {"llr": ll_multi(y, P), "log_loss": sc["log_loss"], "calib_worst_gap_pp": sc["calib_worst_gap_pp"],
                         "calib_pass": sc["calib_pass"], "lines": lines, "pred_rate": float(po.mean()),
                         "actual_rate": float(yo.mean())}
            res["by_class"][cls] = {"te": te, "per": cm, "n": int(len(te))}
        out[fold] = res
    return out


def decideA(F: dict, powered: int) -> dict:
    co = F["code_of"]
    out = {}
    for cls, C in F["by_class"].items():
        per, te = C["per"], C["te"]
        R, RS = per[co["R"]], per[co["R_seed"]]
        seed_diff = abs(R["log_loss"] - RS["log_loss"])
        gfl = max(abs(R["calib_worst_gap_pp"] - RS["calib_worst_gap_pp"]), CALIB_MIN_FLOOR_PP)
        g = te["game_id"].to_numpy()
        arms = {}
        for lab in ("A1", "A2"):
            A = per[co[lab]]
            d = A["llr"] - R["llr"]
            se = boot_mean(g, d)
            floor = max(seed_diff, se)
            gain = float(-d.mean())
            held = A["calib_worst_gap_pp"] <= R["calib_worst_gap_pp"] + gfl
            arms[lab] = {"gain": gain, "boot_se": se, "floor": floor, "gain_in_floors": gain / floor if floor else None,
                         "calib_gate_held": bool(held), "beats_floor": gain > floor,
                         "segments": seg_table(te, d, g, powered)}
        sig = any(a["beats_floor"] and a["calib_gate_held"] for a in arms.values())
        out[cls] = {"seed_diff": seed_diff, "calib_floor_pp": gfl, "arms": arms, "verdict": "SIGNAL" if sig else "NO SIGNAL"}
    return out


# ============================================================================ Read B
def readB(ctx: pd.DataFrame, rng) -> dict:
    out = {}
    for fold in FOLDS:
        pr = pd.read_parquet(R1 / "readB" / fold / "preds.parquet")
        arms = sorted({c.split("__")[0] for c in pr.columns if "__p_" in c})
        code = codes_for(arms, rng)
        res = {"code_of": code, "by_class": {}}
        for cls, bands in BANDS.items():
            P_ = pr[(pr["shot_class"] == cls) & pr["located"]].reset_index(drop=True)
            te = ctx.iloc[P_["fg_row"].to_numpy()].reset_index(drop=True)
            y = te["band"].map({b: i for i, b in enumerate(bands)}).astype("int64").to_numpy()
            g = te["game_id"].to_numpy()
            per = {}
            for arm in arms:
                P = P_[[f"{arm}__p_{b}" for b in bands]].to_numpy(dtype="float64")
                P = P / P.sum(axis=1, keepdims=True)
                llr = ll_multi(y, P)
                # team-game shares
                tg = pd.DataFrame({"g": g, "t": te["off_team_id"].to_numpy()})
                tgs = {}
                for i, b in enumerate(bands):
                    tg[f"a_{b}"] = (y == i).astype(float)
                    tg[f"p_{b}"] = P[:, i]
                    tg[f"v_{b}"] = P[:, i] * (1 - P[:, i])
                agg = tg.groupby(["g", "t"]).agg(**{"n": ("g", "size")},
                                                 **{f"a_{b}": (f"a_{b}", "mean") for b in bands},
                                                 **{f"p_{b}": (f"p_{b}", "mean") for b in bands},
                                                 **{f"v_{b}": (f"v_{b}", "sum") for b in bands})
                agg = agg[agg["n"] >= 5]
                for b in bands:
                    r = agg[f"a_{b}"] - agg[f"p_{b}"]
                    z = r / np.sqrt(agg[f"v_{b}"] / agg["n"] ** 2)
                    tgs[b] = {"team_games": int(len(agg)), "mae_pp": round(100 * float(r.abs().mean()), 4),
                              "resid_sd_pp": round(100 * float(r.std()), 4),
                              "implied_sd_pp": round(100 * float(np.sqrt((agg[f"v_{b}"] / agg["n"] ** 2).mean())), 4),
                              "z_sd": round(float(z.std()), 4)}
                lines = {}
                for side in ("off", "def"):
                    for i, b in enumerate(PROF[cls]):
                        bi = bands.index(b)
                        r_ = PM.quintile_responsiveness(FG._rank(te[f"{side}_{b}_c"].to_numpy()), y, P, bi, n_q=5)
                        v8 = d8(r_["span_actual"], r_["pred_monotone_steps"], r_["slope_ratio"])
                        r_["d8"] = v8
                        r_["pass"] = bool(v8["pass"])
                        r_["reliability"] = REL[(side, b)]
                        r_["slope_over_sqrt_rel"] = (round(r_["slope_ratio"] / np.sqrt(REL[(side, b)]), 4)
                                                     if r_["slope_ratio"] is not None else None)
                        r_["weaker_label"] = bool(side == "def" or b == "wing3")
                        lines[f"{side}_{b}"] = r_
                per[code[arm]] = {"llr": llr, "log_loss": float(llr.mean()), "team_game": tgs, "lines": lines,
                                  "off_lines_pass": all(v["pass"] for k, v in lines.items() if k.startswith("off_")),
                                  "def_lines_pass": all(v["pass"] for k, v in lines.items() if k.startswith("def_")),
                                  "pred_share": P.mean(axis=0).round(5).tolist(),
                                  "actual_share": np.bincount(y, minlength=len(bands)).astype(float).__truediv__(len(y)).round(5).tolist()}
            res["by_class"][cls] = {"te": te, "per": per, "n": int(len(te))}
        out[fold] = res
    return out


def decideB(F: dict) -> dict:
    co = F["code_of"]
    out = {}
    for cls, C in F["by_class"].items():
        per, te = C["per"], C["te"]
        g = te["game_id"].to_numpy()
        cmp_ = {}
        for a, b in (("B1", "B0"), ("B2", "B1"), ("B3", "B2"), ("B2alt", "B2"), ("B2", "B0")):
            d = per[co[a]]["llr"] - per[co[b]]["llr"]
            se = boot_mean(g, d)
            gain = float(-d.mean())
            cmp_[f"{a}_vs_{b}"] = {"gain": gain, "boot_se": se, "floor": se, "gain_in_floors": gain / se if se else None,
                                   "beats_floor": gain > se,
                                   "segments": seg_table(te, d, g, 5000) if (a, b) in (("B1", "B0"), ("B2", "B1")) else None}
        b1, b2 = per[co["B1"]], per[co["B2"]]
        matchup = cmp_["B2_vs_B1"]["beats_floor"] and b2["off_lines_pass"] and b2["def_lines_pass"]
        profile = cmp_["B1_vs_B0"]["beats_floor"] and b1["off_lines_pass"]
        verdict = "MATCHUP" if matchup else ("PROFILE" if profile else "FLAT")
        carried = {"MATCHUP": "B2", "PROFILE": "B1", "FLAT": "B1"}[verdict]
        out[cls] = {"comparisons": cmp_, "verdict": verdict, "carried_arm": carried,
                    "b3_beats_b2": cmp_["B3_vs_B2"]["beats_floor"]}
    return out


# ============================================================================ Read C
def readC(ctx: pd.DataFrame, decB: dict, rng) -> dict:
    out = {}
    rd = pd.read_parquet(REB_DESIGN, columns=["game_id", "off_team_id", "miss_type", "y", "season"])
    for fold in FOLDS:
        bp = pd.read_parquet(R1 / "readB" / fold / "preds.parquet")
        s = TEST[fold]
        mk = {lab: pd.read_parquet(R1 / "make" / f"{lab}_s0_{fold}" / f"cf_{fold}_s0.parquet") for lab in ("R", "A1")}
        ao = {lab: pd.read_parquet(R1 / "andone" / f"{lab}_s0_{fold}" / f"cf_{fold}_s0.parquet") for lab in ("R", "A1")}
        rb = {lab: pd.read_parquet(R1 / "rebound" / f"{lab}_s0_{fold}" / f"cf_{fold}_s0.parquet") for lab in ("R", "A1")}
        # implied arms: served pair R, and loc_<Bk> for B0 / B1 / B2 (the carried one is marked after the reveal)
        labels = ["R", "loc_B0", "loc_B1", "loc_B2"]
        code = codes_for(labels, rng)
        res = {"code_of": code, "by_target": {}}
        for cls, bands in BANDS.items():
            B = bp[bp["shot_class"] == cls].set_index("fg_row")
            Mr = mk["R"][mk["R"]["shot_class"] == cls].set_index("fg_row")
            Ma = mk["A1"][mk["A1"]["shot_class"] == cls].set_index("fg_row")
            rows = Mr.index.to_numpy()
            assert set(rows) == set(B.index)
            te = ctx.iloc[rows].reset_index(drop=True)
            Ma = Ma.loc[rows]
            Bx = B.loc[rows]
            val = 1.5 if cls == "FGA_3" else 1.0
            pm = {"R": Mr.loc[rows, "p"].to_numpy()}
            pmb = np.column_stack([Ma[f"p_{b}"].to_numpy() for b in bands])
            pb = {}
            for arm in ("B0", "B1", "B2"):
                PB = np.column_stack([Bx[f"{arm}__p_{b}"].to_numpy() for b in bands])
                PB = PB / PB.sum(axis=1, keepdims=True)
                pb[arm] = PB
                pm[f"loc_{arm}"] = (PB * pmb).sum(axis=1)
            make_frame = pd.DataFrame({"g": te["game_id"].to_numpy(), "t": te["off_team_id"].to_numpy(),
                                       "d": te["def_team_id"].to_numpy(), "act": te["made"].astype(float).to_numpy() * val,
                                       "located": te["located"].to_numpy().astype(bool)})
            for lab in labels:
                make_frame[lab] = pm[lab] * val
                make_frame["v_" + lab] = pm[lab] * (1 - pm[lab]) * val ** 2
            tname = "efg_jump2" if cls == "FGA_jump2" else "three_make_rate"
            res["by_target"][tname] = make_frame
            # and-one on made rows: P(b | made) proportional to P_B(b) P_A1make(b)
            Ar = ao["R"][ao["R"]["shot_class"] == cls].set_index("fg_row")
            Aa = ao["A1"][ao["A1"]["shot_class"] == cls].set_index("fg_row")
            mrows = Ar.index.to_numpy()
            pos = pd.Series(np.arange(len(rows)), index=rows).loc[mrows].to_numpy()
            ta = ctx.iloc[mrows].reset_index(drop=True)
            fr = pd.DataFrame({"g": ta["game_id"].to_numpy(), "t": ta["off_team_id"].to_numpy(),
                               "d": ta["def_team_id"].to_numpy(), "act": ta["and_one"].astype(float).to_numpy(),
                               "located": ta["located"].to_numpy().astype(bool)})
            pab = np.column_stack([Aa.loc[mrows, f"p_{b}"].to_numpy() for b in bands])
            fr["R"] = Ar.loc[mrows, "p"].to_numpy()
            for arm in ("B0", "B1", "B2"):
                w = pb[arm][pos] * pmb[pos]
                w = w / w.sum(axis=1, keepdims=True)
                fr[f"loc_{arm}"] = (w * pab).sum(axis=1)
            for lab in labels:
                fr["v_" + lab] = fr[lab] * (1 - fr[lab])
            res["by_target"]["andone_" + MT[cls]] = fr
            # OREB on misses: P(b | miss) proportional to P_B(b) (1 - P_A1make(b)); rows with an fg row only
            Rr = rb["R"][rb["R"]["miss_type"] == MT[cls]].set_index("row")
            Ra = rb["A1"][rb["A1"]["miss_type"] == MT[cls]].set_index("row")
            ok = Rr["fg_row"].to_numpy() >= 0
            rr = Rr.index.to_numpy()[ok]
            frow = Rr["fg_row"].to_numpy()[ok]
            pos = pd.Series(np.arange(len(rows)), index=rows).reindex(frow).to_numpy()
            ok2 = np.isfinite(pos)
            rr, frow, pos = rr[ok2], frow[ok2], pos[ok2].astype(int)
            tr_ = rd.iloc[rr]
            tf = ctx.iloc[frow]
            fo = pd.DataFrame({"g": tr_["game_id"].to_numpy(), "t": tr_["off_team_id"].to_numpy(),
                               "d": tf["def_team_id"].to_numpy(), "act": (tr_["y"].to_numpy() == 0).astype(float),
                               "located": tf["located"].to_numpy().astype(bool)})
            pob = np.column_stack([Ra.loc[rr, f"p_OREB_{b}"].to_numpy() for b in bands])
            fo["R"] = Rr.loc[rr, "p_OREB"].to_numpy()
            for arm in ("B0", "B1", "B2"):
                w = pb[arm][pos] * (1 - pmb[pos])
                w = w / w.sum(axis=1, keepdims=True)
                fo[f"loc_{arm}"] = (w * pob).sum(axis=1)
            for lab in labels:
                fo["v_" + lab] = fo[lab] * (1 - fo[lab])
            fo.attrs["n_rebound_rows_total"] = int(len(Rr))
            fo.attrs["n_rows_used"] = int(len(fo))
            res["by_target"]["oreb_" + MT[cls]] = fo
        # combined eFG on jumpers + threes
        res["by_target"]["efg_jump2_three"] = pd.concat([res["by_target"]["efg_jump2"],
                                                         res["by_target"]["three_make_rate"]], ignore_index=True)
        # team-game aggregation and metrics, per code
        prof = ctx[["game_id", "off_team_id", "def_team_id"] + [f"{s_}_{b}_c" for s_ in ("off", "def")
                                                                 for b in ("short2", "corner3", "wing3", "top3")]]
        po = prof.drop_duplicates(["game_id", "off_team_id"]).set_index(["game_id", "off_team_id"])
        metrics = {}
        for tname, fr in res["by_target"].items():
            agg = fr.groupby(["g", "t"]).agg(n=("act", "size"), act=("act", "mean"), loc_share=("located", "mean"),
                                             **{lab: (lab, "mean") for lab in labels},
                                             **{"v_" + lab: ("v_" + lab, "sum") for lab in labels})
            agg = agg.reset_index()
            pf = po.reindex(list(zip(agg["g"], agg["t"])))
            for c in pf.columns:
                if c.endswith("_c"):
                    agg[c] = pf[c].to_numpy()
            tm = {}
            for scope in ("all_games", "located_only_games"):
                A = agg if scope == "all_games" else agg[agg["loc_share"] >= 0.95]
                per = {}
                for lab in labels:
                    r = A["act"] - A[lab]
                    psd = np.sqrt(A["v_" + lab] / A["n"] ** 2)
                    per[code[lab]] = {"mae": float(r.abs().mean()), "absr": r.abs().to_numpy(),
                                      "bias": float(r.mean()), "resid_sd": float(r.std()),
                                      "pred_sd_rms": float(np.sqrt((psd ** 2).mean())),
                                      "lines": {}}
                    if scope == "all_games":
                        for side in ("off", "def"):
                            for b in ("short2", "corner3", "wing3", "top3"):
                                col = f"{side}_{b}_c"
                                okc = A[col].notna().to_numpy()
                                per[code[lab]]["lines"][col] = rate_resp(A.loc[okc, col].to_numpy(),
                                                                          A.loc[okc, lab].to_numpy(),
                                                                          A.loc[okc, "act"].to_numpy(),
                                                                          w=A.loc[okc, "n"].to_numpy())
                tm[scope] = {"team_games": int(len(A)), "g": A["g"].to_numpy(), "per": per}
            tm["attrs"] = dict(fr.attrs)
            metrics[tname] = tm
        res["metrics"] = metrics
        del res["by_target"]
        out[fold] = res
    return out


def decideC(F: dict) -> dict:
    co = F["code_of"]
    out = {}
    for tname, tm in F["metrics"].items():
        out[tname] = {}
        for scope in ("all_games", "located_only_games"):
            S = tm[scope]
            R = S["per"][co["R"]]
            rr = {}
            for lab in ("loc_B0", "loc_B1", "loc_B2"):
                A = S["per"][co[lab]]
                d = A["absr"] - R["absr"]
                se = boot_mean(S["g"], d)
                gain = float(-d.mean())
                rr[lab] = {"mae_R": R["mae"], "mae_arm": A["mae"], "gain": gain, "boot_se": se,
                           "gain_in_floors": gain / se if se else None, "improves": gain > se, "worse": -gain > se}
            out[tname][scope] = rr
    return out


# ============================================================================ reveal + decision
def strip_arrays(o):
    if isinstance(o, dict):
        return {k: strip_arrays(v) for k, v in o.items() if k not in ("p", "llr", "absr", "te", "g")}
    if isinstance(o, list):
        return [strip_arrays(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def reveal(per: dict, code_of: dict) -> dict:
    lab = {v: k for k, v in code_of.items()}
    return {lab[c]: v for c, v in per.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/shot_location_v1/grade_v1.json")
    a = ap.parse_args()
    rng = np.random.default_rng(20261027)
    ctx = load_context()
    print("context loaded", flush=True)
    build = json.loads((R1 / "build_report_v1.json").read_text(encoding="utf-8"))
    refused = bool(build["leak_test"]["refused"])
    rep = {"created_at": pd.Timestamp.now("UTC").isoformat(), "leak_refused": refused, "folds": {}}
    A_make = readA_fg("make", ctx, rng)
    print("Read A make graded", flush=True)
    A_and = readA_fg("andone", ctx, rng)
    print("Read A and-one graded", flush=True)
    A_reb = readA_reb(ctx, rng)
    print("Read A rebound graded", flush=True)
    B = readB(ctx, rng)
    print("Read B graded", flush=True)
    for fold in FOLDS:
        fr: dict = {}
        dB = decideB(B[fold])
        dA = {"make": decideA(A_make[fold], 5000), "andone": decideA(A_and[fold], 5000),
              "rebound": decideA(A_reb[fold], 2000)}
        fr["readA"] = {}
        for chan, F in (("make", A_make[fold]), ("andone", A_and[fold]), ("rebound", A_reb[fold])):
            fr["readA"][chan] = {"decision": dA[chan], "metrics": {
                cls: {"n": C["n"], "arms": reveal(strip_arrays(C["per"]), F["code_of"])} for cls, C in F["by_class"].items()}}
        fr["readB"] = {"decision": dB, "metrics": {cls: {"n": C["n"], "arms": reveal(strip_arrays(C["per"]), B[fold]["code_of"])}
                                                   for cls, C in B[fold]["by_class"].items()}}
        rep["folds"][fold] = fr
        print(f"{fold}: decisions A/B done", flush=True)
    C = readC(ctx, None, rng)
    print("Read C computed", flush=True)
    for fold in FOLDS:
        dC = decideC(C[fold])
        mets = {t: {sc: {"team_games": tm[sc]["team_games"],
                         "arms": reveal(strip_arrays(tm[sc]["per"]), C[fold]["code_of"])}
                    for sc in ("all_games", "located_only_games")} | {"attrs": tm["attrs"]}
                for t, tm in C[fold]["metrics"].items()}
        rep["folds"][fold]["readC"] = {"decision": dC, "metrics": mets}
    # ---- section 1.6 per class, read on F2 (F1 reported with the same mechanics, not gated)
    tmap = {"FGA_jump2": {"make": "efg_jump2", "oreb": "oreb_jump2", "andone": "andone_jump2"},
            "FGA_3": {"make": "three_make_rate", "oreb": "oreb_three", "andone": "andone_three"}}
    for fold in FOLDS:
        fr = rep["folds"][fold]
        dec = {}
        for cls in BANDS:
            bv = fr["readB"]["decision"][cls]
            carried = bv["carried_arm"]
            lab = f"loc_{carried}"
            dc = {k: fr["readC"]["decision"][t]["all_games"][lab] for k, t in tmap[cls].items()}
            improved = [k for k in ("make", "oreb") if dc[k]["improves"]]
            worse = [k for k, v in dc.items() if v["worse"]]
            c2 = bool(improved) and not [w for w in worse if w not in improved]
            lines_ok = True
            line_detail = {}
            for k in improved:
                L = fr["readC"]["metrics"][tmap[cls][k]]["all_games"]["arms"][lab]["lines"]
                for b in PROF[cls]:
                    sides = ("off", "def") if bv["verdict"] == "MATCHUP" else ("off",)
                    for sd in sides:
                        v = L[f"{sd}_{b}_c"]
                        line_detail[f"{k}:{sd}_{b}"] = {"slope": v["slope_ratio"], "steps": v["steps"], "pass": v["pass"]}
                        lines_ok &= v["pass"]
            c3 = bool(improved) and lines_ok
            c1 = bv["verdict"] in ("MATCHUP", "PROFILE")
            c4 = fr["readA"]["make"]["decision"][cls]["verdict"] == "SIGNAL"
            checks = {"1_readB_predictable": c1, "2_readC_improves_none_worse": c2, "3_readC_responsiveness": c3,
                      "4_readA_make_signal": c4}
            if refused:
                verdict = "REFUSED (leak)"
            elif all(checks.values()):
                verdict = "CANDIDATE (offline)"
            elif fr["readA"]["make"]["decision"][cls]["verdict"] == "SIGNAL" and bv["verdict"] == "FLAT":
                verdict = "NOT CANDIDATE -- FINDING (Read A SIGNAL with Read B FLAT)"
            else:
                verdict = "NOT CANDIDATE"
            dec[cls] = {"checks": checks, "readB": bv["verdict"], "carried": carried, "readC_improved": improved,
                        "readC_worse": worse, "readC_lines": line_detail, "verdict": verdict}
        fr["section_1_6"] = dec
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(strip_arrays(rep), indent=1, default=str), encoding="utf-8")
    for fold in FOLDS:
        print(f"== {fold}" + (" (decision fold)" if fold == "F2" else " (reported, not gated)"))
        for cls, d in rep["folds"][fold]["section_1_6"].items():
            print(f"  {cls}: {d['verdict']}  checks={d['checks']}  B={d['readB']} C+={d['readC_improved']} C-={d['readC_worse']}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
