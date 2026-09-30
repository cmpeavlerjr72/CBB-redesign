"""diag_g1_possessions_v1.py -- closed decomposition of the G1 possessions/game miss.

Lane B, 2026-09-30.  DIAGNOSTIC ONLY.  Reads the served 200-seed run, the
per-possession tap written by `diag_g1g5_tap_v1.py` (full slate, 6 seeds,
bit-identity with the served run checked HERE), the 2025 pbp possession layer
and the hoopR box.  Fits nothing, changes nothing, writes
`results/g1g5_diag/g1_report.json`.

All quantities are PER TEAM-GAME (the contract's `possessions` is the mean of
the two sides' counts; the grader's `game_poss` is the mean of the two sides'
box estimates FGA - OREB + TOV + 0.44 FTA).

Chain (each step is a difference of measured levels, so it closes exactly):

  gap = sim200(all graded) - est(all graded)
      = [cnt - est]_pc                 grading source: box estimator vs pbp count
      + subset(all graded -> pbp-complete)
      + [sim200 - simtap]_pc           tap seed sample vs the 200-seed run
      + [simtap - cnt]_pc = OT + regulation
        regulation_pc = regulation_cc + subset(pc -> clock-complete)
        regulation_cc = tiling slack + composition + law + interaction  (by start type)

Tiling: in the engine every regulation half consumes exactly 1200 s
(`used = min(dur, left)`), so pooled N_reg per team-game = 1200 / d_s exactly.
In the pbp layer N_reg = S_a / d_a with S_a = summed durations / (2 G).
  N_s - N_a = (1200 - S_a) / d_s + S_a (d_a - d_s) / (d_s d_a)
and d_s - d_a = sum_k (s_sk - s_ak) d_ak + sum_k s_ak (d_sk - d_ak)
              + sum_k (s_sk - s_ak)(d_sk - d_ak)            (k = start type)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402

RUN = ROOT / "results/engine_v0/F2_2025_s200_v5b_A_full/games.parquet"
TAP = ROOT / "results/g1g5_diag/tap_full"
TAP_SEEDS = (20, 22, 24)
OUT = ROOT / "results/g1g5_diag/g1_report.json"
PREV = {0: "period_start", 1: "DREB", 2: "TOV", 3: "made_FG", 4: "made_FT", 5: "other"}
TYPES = ["period_start", "DREB", "TOV", "made_FG", "made_FT", "other"]
ENDS = ["DREB", "TOV", "made_FG", "made_FT", "other"]

rep: dict = {}

# ------------------------------------------------------------------ served run + grader
g200 = pd.read_parquet(RUN)
summary, raw = G.build_grading_frame(g200, 2025)
graded = summary["game_id"].to_numpy()
sim200 = raw.groupby("game_id")["possessions"].mean()
est = summary.set_index("game_id")["game_poss"]
rep["gate"] = {"sim": float(raw["possessions"].mean()), "act_est": float(est.mean()),
               "gap": float(raw["possessions"].mean() - est.mean()), "n_games": int(len(summary))}

# ------------------------------------------------------------------ tap + bit identity
tg = pd.concat([pd.read_parquet(TAP / f"games_{s}.parquet") for s in TAP_SEEDS], ignore_index=True)
tp = pd.concat([pd.read_parquet(TAP / f"poss_{s}.parquet") for s in TAP_SEEDS], ignore_index=True)
mm = tg.merge(g200, on=["game_id", "seed"], suffixes=("_t", "_a"))
cols = [c for c in g200.columns if c not in ("game_id", "seed")]
mismatch = {c: int((mm[c + "_t"] != mm[c + "_a"]).sum()) for c in cols}
rep["bit_identity"] = {"tap_rows": int(len(tg)), "matched_rows": int(len(mm)),
                       "columns_checked": len(cols), "mismatching_cells": int(sum(mismatch.values()))}
gid_of = tg.drop_duplicates("gidx").set_index("gidx")["game_id"]
tp["game_id"] = gid_of.loc[tp["gidx"].astype(int)].to_numpy()
tp["type"] = tp["prev_end"].astype(int).map(PREV)
# record order within one simulation is chronological (one row per loop step)
tp["ord"] = np.arange(len(tp))
tp = tp.sort_values(["seed", "gidx", "ord"]).reset_index(drop=True)
n_sims_per_game = len(TAP_SEEDS) * 2
# tiling check: regulation seconds consumed per sim
reg = tp[tp["period"] <= 2]
secs = reg.groupby(["seed", "game_id"])["used"].sum()
rep["tiling_sim"] = {"mean_reg_seconds": float(secs.mean()), "share_exactly_2400": float((secs == 2400).mean())}

# ------------------------------------------------------------------ actual pbp layer
p = pd.read_parquet(ROOT / "data/processed/possessions_v2/possessions_2025.parquet")
p = p.sort_values(["game_id", "period", "poss_index"])
u = pd.read_parquet(ROOT / "data/processed/games_universe_v2.parquet").set_index("game_id")
cnt_all = p.groupby(["game_id", "offense_team_id"]).size().groupby("game_id").mean()
cnt_reg = p[p["period"] <= 2].groupby(["game_id", "offense_team_id"]).size().groupby("game_id").mean()
n_teams = p.groupby("game_id")["offense_team_id"].nunique()
pc_ids = [gid for gid in graded
          if gid in cnt_all.index and bool(u.at[gid, "pbp_complete"]) and n_teams.get(gid, 0) == 2]
cc_ids = [gid for gid in pc_ids if bool(u.at[gid, "clock_complete_reg"])]
pc_ids, cc_ids = np.array(pc_ids), np.array(cc_ids)
rep["subsets"] = {"graded": int(len(graded)), "pbp_complete": int(len(pc_ids)), "clock_complete_reg": int(len(cc_ids))}

# box estimator components (hoopR), per team-game
tb = R.load_actual_team_box(2025)
tb = tb[tb["game_id"].isin(pc_ids)]
box = tb.groupby("game_id")[["fga", "oreb", "tov", "fta"]].mean()
two = tb.groupby("game_id").size()
box = box[two.reindex(box.index) == 2]
pc_ids = pc_ids[np.isin(pc_ids, box.index)]
cc_ids = cc_ids[np.isin(cc_ids, box.index)]
rep["subsets"].update({"pbp_complete_with_box": int(len(pc_ids)), "clock_complete_with_box": int(len(cc_ids))})

# ------------------------------------------------------------------ 1. grading source, exact by terminal type
pp = p[p["game_id"].isin(pc_ids)]
term = pp.groupby(["game_id", "terminal_event"]).size().unstack(fill_value=0) / 2.0
t_tov = term.get("TOV", 0)
t_fga = term[["FGA_rim", "FGA_jump2", "FGA_3"]].sum(axis=1)
t_ft = term[["FT_trip_shooting", "FT_trip_bonus"]].sum(axis=1)
t_unk = term.get("unknown", 0)
t_end = term.get("end_period", 0)
b = box.loc[term.index]
src = {
    "TOV: terminal TOV possessions - box TOV": float((t_tov - b["tov"]).mean()),
    "FG: terminal FGA possessions - (box FGA - box OREB)": float((t_fga - (b["fga"] - b["oreb"])).mean()),
    "FT: terminal FT-trip possessions - 0.44 x box FTA": float((t_ft - 0.44 * b["fta"]).mean()),
    "eventless: 'unknown' terminal (mid-period)": float(t_unk.mean()),
    "eventless: 'end_period' terminal (horn)": float(t_end.mean()),
}
cnt_pc = float(cnt_all.loc[pc_ids].mean())
est_pc = float(est.loc[pc_ids].mean())
rep["grading_source"] = {"cnt_pc": cnt_pc, "est_pc": est_pc, "diff": cnt_pc - est_pc,
                         "terms": src, "closure": cnt_pc - est_pc - sum(src.values())}
# the same identity inside the engine: count - estimator on the sim's own box
for s in ("home", "away"):
    raw[s + "_est"] = (raw[s + "_fga3"] + raw[s + "_fga2_rim"] + raw[s + "_fga2_jump"]
                       - raw[s + "_oreb"] + raw[s + "_tov"] + 0.44 * raw[s + "_fta"])
raw["est"] = 0.5 * (raw["home_est"] + raw["away_est"])
rep["sim_count_minus_own_estimator"] = float((raw["possessions"] - raw["est"]).mean())
rep["like_for_like"] = {
    "count_vs_count_pc": float(sim200.loc[pc_ids].mean() - cnt_pc),
    "estimator_vs_estimator_all": float(raw["est"].mean() - est.mean()),
}

# ------------------------------------------------------------------ 2-3. subset and seed-sample terms
gap = rep["gate"]["gap"]
sim_pc200 = float(sim200.loc[pc_ids].mean())
subset_pc = gap - (sim_pc200 - est_pc)
simtap_game = tg.groupby("game_id")["possessions"].mean()
simtap_pc = float(simtap_game.loc[pc_ids].mean())
seed_term = sim_pc200 - simtap_pc

# ------------------------------------------------------------------ 4. OT vs regulation on pc
tp_pc = tp[tp["game_id"].isin(pc_ids)]
n_sims_pc = len(pc_ids) * n_sims_per_game
sim_ot_pp = float((tp_pc["period"] >= 3).sum() / 2.0 / n_sims_pc)
sim_reg_pc = float((tp_pc["period"] <= 2).sum() / 2.0 / n_sims_pc)
act_ot_pp = float((pp["period"] >= 3).sum() / 2.0 / len(pc_ids))
act_reg_pc = float(cnt_reg.loc[pc_ids].mean())
ot_games_sim = tg[tg["game_id"].isin(pc_ids)]["n_periods"].gt(2).mean()
ot_games_act = float(u.loc[pc_ids, "n_periods"].gt(2).mean())
ot_periods_sim = float((tg[tg["game_id"].isin(pc_ids)]["n_periods"] - 2).mean())
ot_periods_act = float((u.loc[pc_ids, "n_periods"] - 2).mean())
ot = {"sim_ot_poss_per_team_game": sim_ot_pp, "act_ot_poss_per_team_game": act_ot_pp,
      "term": sim_ot_pp - act_ot_pp,
      "sim_ot_game_rate": float(ot_games_sim), "act_ot_game_rate": ot_games_act,
      "sim_ot_periods_per_game": ot_periods_sim, "act_ot_periods_per_game": ot_periods_act,
      "sim_poss_per_ot_period_per_team": sim_ot_pp / max(ot_periods_sim, 1e-9),
      "act_poss_per_ot_period_per_team": act_ot_pp / max(ot_periods_act, 1e-9)}
# split OT term: frequency x size
ot["freq_part"] = (ot_periods_sim - ot_periods_act) * ot["act_poss_per_ot_period_per_team"]
ot["size_part"] = ot_periods_sim * (ot["sim_poss_per_ot_period_per_team"] - ot["act_poss_per_ot_period_per_team"])

# ------------------------------------------------------------------ 5. regulation on clock-complete: tiling decomposition
tp_cc = tp[tp["game_id"].isin(cc_ids) & (tp["period"] <= 2)]
pa_cc = p[p["game_id"].isin(cc_ids) & (p["period"] <= 2)]
d_s = float(tp_cc["used"].mean())
d_a = float(pa_cc["duration_s"].mean())
N_s = float(len(tp_cc) / 2.0 / (len(cc_ids) * n_sims_per_game))
N_a = float(len(pa_cc) / 2.0 / len(cc_ids))
S_a = float(pa_cc["duration_s"].sum() / 2.0 / len(cc_ids))
sh_s = tp_cc["type"].value_counts(normalize=True).reindex(TYPES, fill_value=0)
sh_a = pa_cc["start_reason"].value_counts(normalize=True).reindex(TYPES, fill_value=0)
du_s = tp_cc.groupby("type")["used"].mean().reindex(TYPES)
du_a = pa_cc.groupby("start_reason")["duration_s"].mean().reindex(TYPES)
comp_s = ((sh_s - sh_a) * du_a).fillna(0)
law_s = (sh_a * (du_s - du_a)).fillna(0)
int_s = ((sh_s - sh_a) * (du_s - du_a)).fillna(0)
K = -S_a / (d_s * d_a)                 # possessions per second of mean-duration gap
slack = (1200.0 - S_a) / d_s
by_type = pd.DataFrame({"sim_share": sh_s, "act_share": sh_a, "sim_dur": du_s, "act_dur": du_a,
                        "comp_poss": comp_s * K, "law_poss": law_s * K, "inter_poss": int_s * K})
reg_cc = {"N_s": N_s, "N_a": N_a, "diff": N_s - N_a, "d_s": d_s, "d_a": d_a, "S_a": S_a,
          "tiling_slack": slack, "composition": float(comp_s.sum() * K), "law": float(law_s.sum() * K),
          "interaction": float(int_s.sum() * K),
          "closure": (N_s - N_a) - slack - float((comp_s.sum() + law_s.sum() + int_s.sum()) * K),
          "by_type": by_type.reset_index(names="type").to_dict("records")}
reg_pc_term = sim_reg_pc - act_reg_pc
subset_cc = reg_pc_term - (N_s - N_a)

# law term split: horn (period-final) possessions vs the rest, per start type
tp_cc = tp_cc.copy()
tp_cc["final"] = (tp_cc["used"] >= tp_cc["sec"])       # the possession the horn ends
pa_cc = pa_cc.copy()
pa_cc["final"] = ~pa_cc.duplicated(["game_id", "period"], keep="last")
hz = {}
for lab, fs, fa in (("period_final", tp_cc["final"], pa_cc["final"]),
                    ("interior", ~tp_cc["final"], ~pa_cc["final"])):
    hz[lab] = {"sim_share": float(fs.mean()), "act_share": float(fa.mean()),
               "sim_dur": float(tp_cc.loc[fs, "used"].mean()), "act_dur": float(pa_cc.loc[fa, "duration_s"].mean())}
rep["horn_vs_interior_cc"] = hz

# ---- and-one start-type convention (ARITHMETIC re-labelling of the same split)
# The pbp layer (the clock's training table) labels the successor of EVERY
# and-one possession `made_FG` (measured: 100%), FT made or missed.  The engine
# overwrites the and-one's end code with `made_FT` (FT made) or the rebound
# outcome (FT missed).  Re-label the engine's and-one successors to the training
# convention and recompute the split: the TOTAL is unchanged (same possessions,
# same durations), only composition vs law moves.
g4t = pd.concat([pd.read_parquet(ROOT / f"results/g4_diag/tap/trips_{s}.parquet") for s in (0, 4, 7)])
g4c = pd.concat([pd.read_parquet(ROOT / f"results/g4_diag/tap/chances_{s}.parquet") for s in (0, 4, 7)])
a_s = float(g4t["cls"].isin([1.0, 2.0, 3.0]).sum() / (g4c["is_first"] == 1).sum())   # and-one trips / possession
g4r = pd.concat([pd.read_parquet(ROOT / f"results/g4_diag/tap/rebs_{s}.parquet") for s in (0, 4, 7)])
o_ft = float(g4r.loc[g4r["miss_type"] == 3, "p_oreb"].mean())
q_ft = float(((raw["home_ftm"] + raw["away_ftm"]).sum()) / ((raw["home_fta"] + raw["away_fta"]).sum()))
m1 = a_s * q_ft                         # and-one successors the engine labels made_FT
m2 = a_s * (1 - q_ft) * (1 - o_ft)      # ... labels DREB (dead-ball share ignored, ~0)
sh_r = sh_s.copy()
du_r = du_s.copy()
du_r["made_FG"] = (sh_s["made_FG"] * du_s["made_FG"] + m1 * du_s["made_FT"] + m2 * du_s["DREB"]) / (sh_s["made_FG"] + m1 + m2)
sh_r["made_FG"] += m1 + m2
sh_r["made_FT"] -= m1
sh_r["DREB"] -= m2
comp_r = float(((sh_r - sh_a) * du_a).fillna(0).sum() * K)
law_r = float((sh_a * (du_r - du_a)).fillna(0).sum() * K)
int_r = float(((sh_r - sh_a) * (du_r - du_a)).fillna(0).sum() * K)
by_r = pd.DataFrame({"sim_share": sh_r, "act_share": sh_a, "sim_dur": du_r, "act_dur": du_a,
                     "comp_poss": ((sh_r - sh_a) * du_a).fillna(0) * K,
                     "law_poss": (sh_a * (du_r - du_a)).fillna(0) * K})
# feed counterfactual: if the engine fed `made_FG` to the clock after an and-one
# (the training convention), those possessions would draw the made_FG law
d_feed = d_s + m1 * (du_s["made_FG"] - du_s["made_FT"]) + m2 * (du_s["made_FG"] - du_s["DREB"])
rep["and_one_convention"] = {
    "sim_and_one_per_poss": a_s, "sim_ft_pct": q_ft, "sim_p_oreb_ft_miss": o_ft,
    "moved_made_FT_to_made_FG": m1, "moved_DREB_to_made_FG": m2,
    "relabelled": {"composition": comp_r, "law": law_r, "interaction": int_r,
                   "sum": comp_r + law_r + int_r,
                   "by_type": by_r.reset_index(names="type").to_dict("records")},
    "feed_counterfactual_dN": 1200.0 / d_feed - 1200.0 / d_s,
    "act_next_poss_dur_after_and_one": float(
        pa_cc.assign(nd=pa_cc.groupby(["game_id", "period"])["duration_s"].shift(-1))
        .loc[pa_cc["and_one"], "nd"].mean()),
}

# per-half and per-minute counts (possessions STARTED per team-game), cc games
def minute_bucket(period, sec):
    gm = np.where(period == 1, (1200 - sec) / 60.0, 20 + (1200 - sec) / 60.0)
    return np.minimum(gm.astype(int) // 5, 7)


tp_cc["mb"] = minute_bucket(tp_cc["period"].to_numpy(), tp_cc["sec"].to_numpy())
pa_cc["mb"] = minute_bucket(pa_cc["period"].to_numpy(), pa_cc["start_clock"].to_numpy())
den_s = len(cc_ids) * n_sims_per_game * 2.0
den_a = len(cc_ids) * 2.0
half = []
for h in (1, 2):
    xs, xa = tp_cc[tp_cc["period"] == h], pa_cc[pa_cc["period"] == h]
    half.append({"half": h, "sim": len(xs) / den_s, "act": len(xa) / den_a, "diff": len(xs) / den_s - len(xa) / den_a,
                 "sim_dur": float(xs["used"].mean()), "act_dur": float(xa["duration_s"].mean())})
mins = []
for k in range(8):
    xs, xa = tp_cc[tp_cc["mb"] == k], pa_cc[pa_cc["mb"] == k]
    mins.append({"game_min": f"{5 * k}-{5 * k + 5}", "sim": len(xs) / den_s, "act": len(xa) / den_a,
                 "diff": len(xs) / den_s - len(xa) / den_a,
                 "sim_dur": float(xs["used"].mean()), "act_dur": float(xa["duration_s"].mean())})
rep["per_half_cc"] = half
rep["per_5min_cc"] = mins

# ------------------------------------------------------------------ 6. end-type mix and the upstream counterfactual (ARITHMETIC)
# end type of a possession = the start type of the next possession in the same
# period (the period-final possession's end is overwritten by period_start on
# both sides, so both sides use interior possessions only).
def interior_end_mix_sim(df):
    df = df[["seed", "game_id", "period", "type"]]
    nxt = df["type"].shift(-1)
    same = (df["seed"].shift(-1) == df["seed"]) & (df["game_id"].shift(-1) == df["game_id"]) & \
           (df["period"].shift(-1) == df["period"])
    return nxt[same].value_counts(normalize=True).reindex(ENDS, fill_value=0)


def interior_end_mix_act(df):
    nxt = df["start_reason"].shift(-1)
    same = (df["game_id"].shift(-1) == df["game_id"]) & (df["period"].shift(-1) == df["period"])
    return nxt[same].value_counts(normalize=True).reindex(ENDS, fill_value=0)


tp_all_cc = tp[tp["game_id"].isin(cc_ids) & (tp["period"] <= 2)]
pi_s = interior_end_mix_sim(tp_all_cc)
pi_a = interior_end_mix_act(p[p["game_id"].isin(cc_ids) & (p["period"] <= 2)])
# continuation (OREB) per chance: sim from its own box OREB, actual from pbp oreb_count
tgc = tg[tg["game_id"].isin(cc_ids)]
oreb_s = float((tgc["home_oreb"] + tgc["away_oreb"]).sum())
poss_s = float((tgc["possessions"] * 2).sum())
oreb_a = float(p[p["game_id"].isin(cc_ids)]["oreb_count"].sum())
poss_a = float(len(p[p["game_id"].isin(cc_ids)]))
c_s = oreb_s / (oreb_s + poss_s)
c_a = oreb_a / (oreb_a + poss_a)


def chance_vec(pi, c):
    e = pi * (1 - c)                         # per-chance terminal probabilities
    return e, c


def mix_from(e, c):
    return e / (1 - c)


e_s, _ = chance_vec(pi_s, c_s)
e_a, _ = chance_vec(pi_a, c_a)
o_s = c_s / (c_s + e_s["DREB"])
o_a = c_a / (c_a + e_a["DREB"])


def cf(e, c, oreb_to=None, ft_to=None, tov_to=None, fg_to=None):
    e = e.copy()
    if fg_to is not None:
        # make rate: a miss that went to a live rebound becomes a make; the
        # live-rebound pool (c + DREB) shrinks by the same amount, OREB share kept
        dlt = fg_to - e["made_FG"]
        r = c + e["DREB"]
        o = c / r
        r2 = r - dlt
        e["made_FG"] = fg_to
        c, e["DREB"] = r2 * o, r2 * (1 - o)
    if ft_to is not None:
        rest = 1.0 - e["made_FT"]
        scale = (1.0 - ft_to) / rest
        c = c * scale
        e = e * scale
        e["made_FT"] = ft_to
    if tov_to is not None:
        rest = 1.0 - e["TOV"]
        scale = (1.0 - tov_to) / rest
        c = c * scale
        e = e * scale
        e["TOV"] = tov_to
    if oreb_to is not None:
        r = c + e["DREB"]
        c = r * oreb_to
        e["DREB"] = r * (1 - oreb_to)
    return e, c


# start-type shares implied by an interior end mix (period_start share held fixed)
def N_from_mix(pi, dur, ps_share):
    sh = pd.Series({"period_start": ps_share})
    for k in ENDS:
        sh[k] = (1 - ps_share) * pi[k]
    d = float((sh * dur.reindex(sh.index)).sum())
    return 1200.0 / d, d


ps = float(sh_s["period_start"])
N0, d0 = N_from_mix(pi_s, du_s, ps)
cf_rows = {}
for lab, kw in (("OREB to actual", dict(oreb_to=o_a)),
                ("FT-trip ends to actual", dict(ft_to=float(e_a["made_FT"]))),
                ("OREB + FT to actual", dict(oreb_to=o_a, ft_to=float(e_a["made_FT"]))),
                ("TOV to actual", dict(tov_to=float(e_a["TOV"]))),
                ("made-FG per chance to actual", dict(fg_to=float(e_a["made_FG"]))),
                ("OREB + FT + TOV to actual", dict(oreb_to=o_a, ft_to=float(e_a["made_FT"]),
                                                   tov_to=float(e_a["TOV"]))),
                ("OREB + FT + TOV + made-FG to actual", dict(oreb_to=o_a, ft_to=float(e_a["made_FT"]),
                                                             tov_to=float(e_a["TOV"]),
                                                             fg_to=float(e_a["made_FG"]))),
                ("full actual chance vector", None)):
    if kw is None:
        e2, c2 = e_a, c_a
    else:
        e2, c2 = cf(e_s, c_s, **kw)
    pi2 = mix_from(e2, c2)
    N2, d2 = N_from_mix(pi2, du_s, ps)
    cf_rows[lab] = {"dN_per_team_game": N2 - N0, "mean_dur": d2,
                    "mix": {k: float(pi2[k]) for k in ENDS},
                    "chances_per_poss": float(1 / (1 - c2))}
# the same counterfactuals with the engine's and-one successors re-labelled to
# the training convention (section and_one_convention): the labelling is the
# one the ACTUAL mix uses, so this is the like-for-like attribution
pi_r = pi_s.copy()
pi_r["made_FG"] += (m1 + m2) / (1 - ps)
pi_r["made_FT"] -= m1 / (1 - ps)
pi_r["DREB"] -= m2 / (1 - ps)
e_r, _ = chance_vec(pi_r, c_s)
N0r, _ = N_from_mix(pi_r, du_r, ps)
cf_rel = {}
for lab, kw in (("OREB to actual", dict(oreb_to=o_a)),
                ("FT-trip ends to actual", dict(ft_to=float(e_a["made_FT"]))),
                ("OREB + FT to actual", dict(oreb_to=o_a, ft_to=float(e_a["made_FT"]))),
                ("TOV to actual", dict(tov_to=float(e_a["TOV"]))),
                ("made-FG per chance to actual", dict(fg_to=float(e_a["made_FG"]))),
                ("full actual chance vector", None)):
    e2, c2 = (e_a, c_a) if kw is None else cf(e_r, c_s, **kw)
    N2, _ = N_from_mix(mix_from(e2, c2), du_r, ps)
    cf_rel[lab] = {"dN_per_team_game": N2 - N0r, "chances_per_poss": float(1 / (1 - c2))}
rep["end_mix_relabelled"] = {"sim_interior_end_mix": pi_r.to_dict(), "per_chance_sim": e_r.to_dict(),
                             "N_model": N0r, "counterfactuals": cf_rel}
rep["end_mix"] = {"sim_interior_end_mix": pi_s.to_dict(), "act_interior_end_mix": pi_a.to_dict(),
                  "c_sim": c_s, "c_act": c_a, "oreb_share_sim": float(o_s), "oreb_share_act": float(o_a),
                  "per_chance_sim": e_s.to_dict(), "per_chance_act": e_a.to_dict(),
                  "N_model_sim": N0, "N_measured_sim_cc": N_s, "counterfactuals": cf_rows}

# ------------------------------------------------------------------ 7. compensation: points = 2 N PPP
# On the pbp-complete games (count-vs-count, the like-for-like possession
# definition), sim from the 200-seed run, actual points from the verified
# finals, actual FTM from the hoopR box, chances = possessions + OREB (sim box;
# pbp `oreb_count` for the actual).
rp = raw[raw["game_id"].isin(pc_ids)]
pts_s = float(((rp["home_pts"] + rp["away_pts"]) / 2).mean())
ftm_s = float(((rp["home_ftm"] + rp["away_ftm"]) / 2).mean())
Nsc = float(rp["possessions"].mean())
orb_s = float(((rp["home_oreb"] + rp["away_oreb"]) / 2).mean())
fin_pc = summary.set_index("game_id").loc[pc_ids]
pts_a = float(((fin_pc["home_score"] + fin_pc["away_score"]) / 2).mean())
tbc = R.load_actual_team_box(2025)
ftm_a = float(tbc[tbc["game_id"].isin(pc_ids)].groupby("game_id")["ftm"].mean().loc[pc_ids].mean())
Nac = float(cnt_all.loc[pc_ids].mean())
orb_a = float(pp.groupby("game_id")["oreb_count"].sum().loc[pc_ids].mean() / 2.0)
ch_s, ch_a = (Nsc + orb_s) / Nsc, (Nac + orb_a) / Nac
ppp_s, ppp_a = pts_s / Nsc, pts_a / Nac
comp = {"n_games": int(len(pc_ids)), "pts_per_team_sim": pts_s, "pts_per_team_act": pts_a,
        "N_sim": Nsc, "N_act": Nac, "ppp_sim": ppp_s, "ppp_act": ppp_a,
        "total_gap": 2 * (pts_s - pts_a),
        "from_possessions": 2 * (Nsc - Nac) * ppp_a,
        "from_ppp": 2 * Nsc * (ppp_s - ppp_a),
        "chances_per_poss_sim": ch_s, "chances_per_poss_act": ch_a,
        "pts_per_chance_sim": ppp_s / ch_s, "pts_per_chance_act": ppp_a / ch_a,
        "ln_ppp_gap": float(np.log(ppp_s / ppp_a)),
        "ln_chances_per_poss_gap": float(np.log(ch_s / ch_a)),
        "ln_pts_per_chance_gap": float(np.log((ppp_s / ch_s) / (ppp_a / ch_a)))}
chn_s, chn_a = Nsc * ch_s, Nac * ch_a
comp["ft_pts_per_chance_sim"], comp["ft_pts_per_chance_act"] = ftm_s / chn_s, ftm_a / chn_a
comp["fg_pts_per_chance_sim"] = (pts_s - ftm_s) / chn_s
comp["fg_pts_per_chance_act"] = (pts_a - ftm_a) / chn_a
# exact split of ln(pts/chance): ln(1 + dFT/x_a + dFG/x_a) vs its two linear parts
xa_ = ppp_a / ch_a
comp["ln_pts_per_chance_FT_part"] = float((ftm_s / chn_s - ftm_a / chn_a) / xa_)
comp["ln_pts_per_chance_FG_part"] = float(((pts_s - ftm_s) / chn_s - (pts_a - ftm_a) / chn_a) / xa_)
comp["ln_pts_per_chance_nonlinear"] = comp["ln_pts_per_chance_gap"] - comp["ln_pts_per_chance_FT_part"]     - comp["ln_pts_per_chance_FG_part"]
# the points each channel moves, allocating `from_ppp` by the exact log shares
L_ = comp["ln_ppp_gap"]
for k in ("ln_chances_per_poss_gap", "ln_pts_per_chance_FT_part", "ln_pts_per_chance_FG_part",
          "ln_pts_per_chance_nonlinear"):
    comp["points_" + k] = comp["from_ppp"] * comp[k] / L_
# the same identity on ALL graded games with the grader's own definitions (G9)
comp["g9_total_bias_all_graded"] = float((summary["sim_total_mean"] - summary["total"]).mean())
ok00 = ~((summary["home_score"] == 0) & (summary["away_score"] == 0))
comp["g9_total_bias_excluding_0_0_finals"] = float((summary.loc[ok00, "sim_total_mean"] - summary.loc[ok00, "total"]).mean())
comp["n_0_0_finals_in_graded_set"] = int((~ok00).sum())
rep["compensation_pc"] = comp

# pace-quintile cut of the regulation tiling decomposition (as-of game pace =
# the 200-seed sim mean, which uses as-of inputs only)
qg = pd.qcut(sim200.loc[cc_ids], 5, labels=False)
qrows = []
for qq in range(5):
    ids = qg.index[qg == qq].to_numpy()
    xs = tp_cc[tp_cc["game_id"].isin(ids)]
    xa = pa_cc[pa_cc["game_id"].isin(ids)]
    dS, dA = float(xs["used"].mean()), float(xa["duration_s"].mean())
    SA = float(xa["duration_s"].sum() / 2.0 / len(ids))
    ss = xs["type"].value_counts(normalize=True).reindex(TYPES, fill_value=0)
    sa = xa["start_reason"].value_counts(normalize=True).reindex(TYPES, fill_value=0)
    ds_ = xs.groupby("type")["used"].mean().reindex(TYPES)
    da_ = xa.groupby("start_reason")["duration_s"].mean().reindex(TYPES)
    KK = -SA / (dS * dA)
    Ns = len(xs) / 2.0 / (len(ids) * n_sims_per_game)
    Na = len(xa) / 2.0 / len(ids)
    qrows.append({"quintile": qq + 1, "games": int(len(ids)), "N_sim": Ns, "N_act": Na, "diff": Ns - Na,
                  "slack": (1200 - SA) / dS,
                  "composition": float(((ss - sa) * da_).fillna(0).sum() * KK),
                  "law": float((sa * (ds_ - da_)).fillna(0).sum() * KK),
                  "interaction": float(((ss - sa) * (ds_ - da_)).fillna(0).sum() * KK)})
rep["regulation_cc_by_pace_quintile"] = qrows

# ------------------------------------------------------------------ closure of the whole chain
chain = {
    "grading source (pbp count - box estimator)": cnt_pc - est_pc,
    "subset: all graded -> pbp-complete": subset_pc,
    "seed sample: 200-seed run - 6-seed tap": seed_term,
    "overtime possessions": ot["term"],
    "subset: pbp-complete -> clock-complete (regulation)": subset_cc,
    "regulation: tiling slack (pbp periods not summing to 1200 s)": slack,
    "regulation: start-type COMPOSITION": reg_cc["composition"],
    "regulation: duration LAW within start type": reg_cc["law"],
    "regulation: composition x law interaction": reg_cc["interaction"],
}
rep["chain"] = chain
rep["chain_closure"] = gap - sum(chain.values()) - reg_cc["closure"]
rep["regulation_cc"] = reg_cc
rep["ot"] = ot
rep["levels"] = {"sim_pc200": sim_pc200, "simtap_pc": simtap_pc, "sim_reg_pc": sim_reg_pc,
                 "act_reg_pc": act_reg_pc}

# ------------------------------------------------------------------ multi-level cuts (count vs count, pc games)
fin = summary.set_index("game_id").loc[pc_ids]
d_game = sim200.loc[pc_ids] - cnt_all.loc[pc_ids]
cuts = {"per_game": {"mean": float(d_game.mean()), "median": float(d_game.median()),
                     "sd": float(d_game.std()), "share_sim_gt_act": float((d_game > 0).mean())}}
cuts["site"] = {("neutral" if k else "home/away"): float(v) for k, v in d_game.groupby(fin["neutral"].astype(bool)).mean().items()}
cuts["month"] = {int(k): {"n": int((fin["month"] == k).sum()), "diff": float(v)}
                 for k, v in d_game.groupby(fin["month"]).mean().items()}
# per-team: team's as-of tempo quintile (sim mean possessions) -> slope check
teams = pd.concat([fin[["home_team_id"]].rename(columns={"home_team_id": "t"}),
                   fin[["away_team_id"]].rename(columns={"away_team_id": "t"})])
teams["d"] = pd.concat([d_game, d_game]).to_numpy()
teams["sim"] = pd.concat([sim200.loc[pc_ids], sim200.loc[pc_ids]]).to_numpy()
teams["act"] = pd.concat([cnt_all.loc[pc_ids], cnt_all.loc[pc_ids]]).to_numpy()
per_team = teams.groupby("t").agg(n=("d", "size"), d=("d", "mean"), sim=("sim", "mean"), act=("act", "mean"))
per_team = per_team[per_team["n"] >= 10]
q = pd.qcut(per_team["sim"], 5, labels=False).rename("quintile")
cuts["team_quintile_by_sim_pace"] = per_team.groupby(q).agg(
    teams=("d", "size"), sim=("sim", "mean"), act=("act", "mean"), diff=("d", "mean")).reset_index().to_dict("records")
cuts["per_team"] = {"n_teams": int(len(per_team)), "median_diff": float(per_team["d"].median()),
                    "share_teams_sim_gt_act": float((per_team["d"] > 0).mean()),
                    "sd_diff": float(per_team["d"].std())}
rep["cuts_count_vs_count"] = cuts

OUT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: rep[k] for k in ("gate", "bit_identity", "tiling_sim", "subsets", "grading_source",
                                      "sim_count_minus_own_estimator", "like_for_like", "chain",
                                      "chain_closure", "ot")}, indent=1, default=float))
print(json.dumps({"reg": {k: v for k, v in reg_cc.items() if k != "by_type"}}, indent=1, default=float))
print(pd.DataFrame(reg_cc["by_type"]).to_string())
print(json.dumps({"end_mix": rep["end_mix"], "comp": comp, "half": half, "min": mins, "horn": hz,
                  "cuts": cuts}, indent=1, default=float))
