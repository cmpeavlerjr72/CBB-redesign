"""diag_g5_corr_v1.py -- G5 home/away score correlation and total-SD decomposition.

Lane B, 2026-09-30.  DIAGNOSTIC ONLY: reads the served 200-seed run and the
2025 verified finals / hoopR team box; fits nothing that is served, changes no
default, writes only `results/g1g5_diag/g5_report.json`.

Objects (all population moments, ddof=0, so every identity closes exactly):

  S  sim:    per (game, seed) rows of `F2_2025_s200_v5b_A_full`.
  m  as-of expectation per game and side = the sim's own 200-seed mean.  The
     engine consumes as-of inputs only (created_at < tipoff), so `m` is an
     as-of team-strength / matchup adjustment that uses no same-game info.
  r  actual residual = actual - m (per game, per side, per stat).
  w  sim within-game deviation = sim row - m.

G5's two lines, restated exactly:
  corr_sim  = pooled corr(home_pts, away_pts) over all (game, seed) rows
            = [Cov(m_h, m_a) + E_g Cov_w(h, a)] / D_sim
  corr_act  = [Cov(m_h, m_a) + Cov(m_h, r_a) + Cov(r_h, m_a) + Cov(r_h, r_a)] / D_act
  total SD ratio = mean_g SD_w(T) / SD(r_T)   (ddof as the gate: ddof=1 inside)

Pace/efficiency split on the log scale (exact):  pts_i = N * e_i, N = the
game's possessions (box estimator, mean of the two sides, SAME definition on
both sides), e_i = pts_i / N.
  Cov(ln h, ln a) = Var(ln N) + Cov(ln N, ln e_h) + Cov(ln N, ln e_a) + Cov(ln e_h, ln e_a)
  Var(ln T)       = Var(ln N) + Var(ln E) + 2 Cov(ln N, ln E),  E = e_h + e_a

Shared components beyond pace: home-vs-away residual correlation of each
team-game rate (actual r vs sim w), and an exact attribution of
Cov(ln e_h, ln e_a) to four factors via a per-team-game regression of ln e on
the four factors' own deviations (fitted separately on each side; the
explained part, the cross terms with the regression residual and the
residual-residual term are all printed, so it closes).
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
OUT = ROOT / "results/g1g5_diag/g5_report.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260930)

FACTORS = ["efg", "tovp", "orebp", "ftr"]
RATES = ["efg", "tovp", "orebp", "ftr", "fg3_share", "ft_pct", "fg3_pct", "fg2_pct", "fga_pp", "ln_e"]


def pcov(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def pcorr(x, y):
    return pcov(x, y) / np.sqrt(pcov(x, x) * pcov(y, y))


def side_stats(pts, fga, fgm, fg3a, fg3m, fta, ftm, oreb, dreb_opp, tov, n):
    """Per team-game rates. `n` is the GAME possession count (shared)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "pts": pts, "e": pts / n, "ln_e": np.log(pts / n),
            "efg": (fgm + 0.5 * fg3m) / fga, "tovp": tov / n,
            "orebp": oreb / (oreb + dreb_opp), "ftr": fta / fga,
            "fg3_share": fg3a / fga, "ft_pct": np.where(fta > 0, ftm / np.maximum(fta, 1), np.nan),
            "fg3_pct": np.where(fg3a > 0, fg3m / np.maximum(fg3a, 1), np.nan),
            "fg2_pct": (fgm - fg3m) / np.maximum(fga - fg3a, 1), "fga_pp": fga / n,
        }


# ------------------------------------------------------------------ sim
g = pd.read_parquet(RUN)
summary, raw = G.build_grading_frame(g, 2025)
gids = summary["game_id"].to_numpy()
raw = raw[raw["game_id"].isin(gids)].copy()
corr_gate_sim = float(np.corrcoef(raw["home_pts"].astype(float), raw["away_pts"].astype(float))[0, 1])
corr_gate_act = float(np.corrcoef(summary["home_score"].astype(float), summary["away_score"].astype(float))[0, 1])
ratio_gate = float(summary["sim_total_sd"].mean() / (summary["total"] - summary["sim_total_mean"]).std())

S = {}
for s, o in (("h", "home"), ("a", "away")):
    oo = "away" if o == "home" else "home"
    fga = (raw[f"{o}_fga3"] + raw[f"{o}_fga2_rim"] + raw[f"{o}_fga2_jump"]).to_numpy(float)
    fgm = (raw[f"{o}_fgm3"] + raw[f"{o}_fgm2_rim"] + raw[f"{o}_fgm2_jump"]).to_numpy(float)
    S[s] = dict(pts=raw[f"{o}_pts"].to_numpy(float), fga=fga, fgm=fgm,
                fg3a=raw[f"{o}_fga3"].to_numpy(float), fg3m=raw[f"{o}_fgm3"].to_numpy(float),
                fta=raw[f"{o}_fta"].to_numpy(float), ftm=raw[f"{o}_ftm"].to_numpy(float),
                oreb=raw[f"{o}_oreb"].to_numpy(float), dreb_opp=raw[f"{oo}_dreb"].to_numpy(float),
                tov=raw[f"{o}_tov"].to_numpy(float))
    S[s]["est"] = S[s]["fga"] - S[s]["oreb"] + S[s]["tov"] + 0.44 * S[s]["fta"]
N_sim = 0.5 * (S["h"]["est"] + S["a"]["est"])
sim = pd.DataFrame({"game_id": raw["game_id"].to_numpy(), "N": N_sim, "lnN": np.log(N_sim)})
for s in ("h", "a"):
    d = S[s]
    st = side_stats(d["pts"], d["fga"], d["fgm"], d["fg3a"], d["fg3m"], d["fta"], d["ftm"],
                    d["oreb"], d["dreb_opp"], d["tov"], N_sim)
    for k, v in st.items():
        sim[f"{k}_{s}"] = v
sim["T"] = sim["pts_h"] + sim["pts_a"]
sim["lnT"] = np.log(sim["T"])
sim["lnE"] = np.log(sim["e_h"] + sim["e_a"])
sim["ln_h"] = np.log(sim["pts_h"])
sim["ln_a"] = np.log(sim["pts_a"])

COLS = ["pts_h", "pts_a", "T", "N", "lnN", "lnT", "lnE", "ln_h", "ln_a"] + \
    [f"{k}_{s}" for k in RATES + ["e"] for s in ("h", "a")]
COLS = list(dict.fromkeys(COLS))
M = sim.groupby("game_id")[COLS].mean()          # as-of expectation per game
W = sim[["game_id"] + COLS].copy()
W[COLS] = W[COLS].to_numpy() - M.loc[W["game_id"], COLS].to_numpy()   # within deviations

# ------------------------------------------------------------------ actual
tb = R.load_actual_team_box(2025)
tb = tb[tb["game_id"].isin(gids)]
h = tb[tb["team_id"] == tb["home_team_id"]].set_index("game_id")
a = tb[tb["team_id"] == tb["away_team_id"]].set_index("game_id")
fin = summary.set_index("game_id")
common = fin.index.intersection(h.index).intersection(a.index)
fin, h, a = fin.loc[common], h.loc[common], a.loc[common]
N_act = 0.5 * (h["poss_team"] + a["poss_team"]).to_numpy(float)
act = pd.DataFrame(index=common)
act["N"] = N_act
act["lnN"] = np.log(N_act)
for s, t, pts, opp in (("h", h, fin["home_score"], a), ("a", a, fin["away_score"], h)):
    st = side_stats(pts.to_numpy(float), t["fga"].to_numpy(float), t["fgm"].to_numpy(float),
                    t["tpa"].to_numpy(float), t["tpm"].to_numpy(float), t["fta"].to_numpy(float),
                    t["ftm"].to_numpy(float), t["oreb"].to_numpy(float), opp["dreb"].to_numpy(float),
                    t["tov"].to_numpy(float), N_act)
    for k, v in st.items():
        act[f"{k}_{s}"] = v
act["T"] = act["pts_h"] + act["pts_a"]
act["lnT"] = np.log(act["T"])
act["lnE"] = np.log(act["e_h"] + act["e_a"])
act["ln_h"] = np.log(act["pts_h"])
act["ln_a"] = np.log(act["pts_a"])
Mc = M.loc[common]
Rr = act[COLS] - Mc[COLS]                        # actual residual vs as-of expectation
month = fin["month"]
Wg = W[W["game_id"].isin(common)]
n_games = int(len(common))

# ------------------------------------------------------------------ (A) exact G5 corr decomposition
mh, ma = Mc["pts_h"].to_numpy(), Mc["pts_a"].to_numpy()
rh, ra = Rr["pts_h"].to_numpy(), Rr["pts_a"].to_numpy()
wh, wa = Wg["pts_h"].to_numpy(), Wg["pts_a"].to_numpy()
# sim: pooled over rows of `common` games only (equal seeds per game -> exact split)
cov_mm = pcov(mh, ma)
cov_w = float(np.mean(wh * wa))                   # E_g Cov_w (deviations already game-centred)
var_sh = pcov(mh, mh) + float(np.mean(wh * wh))
var_sa = pcov(ma, ma) + float(np.mean(wa * wa))
D_s = np.sqrt(var_sh * var_sa)
xh = act["pts_h"].to_numpy()
xa = act["pts_a"].to_numpy()
D_a = np.sqrt(pcov(xh, xh) * pcov(xa, xa))
terms_act = {"between": cov_mm, "cross_m_r": pcov(mh, ra) + pcov(rh, ma), "resid": pcov(rh, ra)}
terms_sim = {"between": cov_mm, "cross_m_r": 0.0, "resid": cov_w}
corr_act = sum(terms_act.values()) / D_a
corr_sim = sum(terms_sim.values()) / D_s
A = {"corr_act": corr_act, "corr_sim": corr_sim, "gap": corr_act - corr_sim,
     "D_act": D_a, "D_sim": D_s,
     "gate_corr_sim": corr_gate_sim, "gate_corr_act": corr_gate_act, "gate_total_sd_ratio": ratio_gate,
     "n_games": n_games, "terms": {}}
for k in terms_act:
    A["terms"][k] = {"act_cov": terms_act[k], "sim_cov": terms_sim[k],
                     "act_corr_units": terms_act[k] / D_a, "sim_corr_units": terms_sim[k] / D_s,
                     "gap_corr_units": terms_act[k] / D_a - terms_sim[k] / D_s}
# split the 'between' gap into a denominator effect (same covariance, different D)
A["var_parts"] = {"act_var_h": pcov(xh, xh), "act_var_a": pcov(xa, xa),
                  "sim_var_h_between": pcov(mh, mh), "sim_var_h_within": float(np.mean(wh * wh)),
                  "sim_var_a_between": pcov(ma, ma), "sim_var_a_within": float(np.mean(wa * wa)),
                  "act_var_h_resid": pcov(rh, rh), "act_var_a_resid": pcov(ra, ra)}

# ------------------------------------------------------------------ (B) residual/within: variance of total and home-away cov
def moments(df_or_arr_getter, centred):
    """centred=True: deviations are already game-centred (sim within) -> E[x*y];
    False: residuals (actual) -> population covariance."""
    f = (lambda x, y: float(np.mean(x * y))) if centred else pcov
    get = df_or_arr_getter
    out = {}
    out["var_T"] = f(get("T"), get("T"))
    out["var_h"] = f(get("pts_h"), get("pts_h"))
    out["var_a"] = f(get("pts_a"), get("pts_a"))
    out["cov_ha"] = f(get("pts_h"), get("pts_a"))
    out["corr_ha"] = out["cov_ha"] / np.sqrt(out["var_h"] * out["var_a"])
    # log scale, exact
    out["log_cov_ha"] = f(get("ln_h"), get("ln_a"))
    out["log_var_h"] = f(get("ln_h"), get("ln_h"))
    out["log_var_a"] = f(get("ln_a"), get("ln_a"))
    out["log_corr_ha"] = out["log_cov_ha"] / np.sqrt(out["log_var_h"] * out["log_var_a"])
    out["log_parts_cov_ha"] = {
        "var_lnN": f(get("lnN"), get("lnN")),
        "cov_lnN_lneh": f(get("lnN"), get("ln_e_h")),
        "cov_lnN_lnea": f(get("lnN"), get("ln_e_a")),
        "cov_lneh_lnea": f(get("ln_e_h"), get("ln_e_a")),
    }
    out["log_parts_cov_ha_sum"] = sum(out["log_parts_cov_ha"].values())
    out["log_var_T"] = f(get("lnT"), get("lnT"))
    out["log_parts_var_T"] = {"var_lnN": f(get("lnN"), get("lnN")), "var_lnE": f(get("lnE"), get("lnE")),
                              "2cov_lnN_lnE": 2 * f(get("lnN"), get("lnE"))}
    out["log_parts_var_T_sum"] = sum(out["log_parts_var_T"].values())
    out["var_lneh"] = f(get("ln_e_h"), get("ln_e_h"))
    out["var_lnea"] = f(get("ln_e_a"), get("ln_e_a"))
    return out


Bact = moments(lambda c: Rr[c].to_numpy(), centred=False)
Bsim = moments(lambda c: Wg[c].to_numpy(), centred=True)
# SD(T) ratio in the gate's own convention, for reference
B = {"act_resid": Bact, "sim_within": Bsim,
     "total_sd_ratio_pop": float(np.sqrt(Bsim["var_T"] / Bact["var_T"]))}

# ------------------------------------------------------------------ (C) shared components, per rate
def shared_table(Rdf, Wdf, cols, boot=200):
    rows = []
    idx = np.arange(len(Rdf))
    for k in cols:
        x = Rdf[f"{k}_h"].to_numpy()
        y = Rdf[f"{k}_a"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        ca = pcorr(x[ok], y[ok])
        bs = []
        for _ in range(boot):
            b = RNG.choice(idx[ok], size=ok.sum(), replace=True)
            bs.append(pcorr(x[b], y[b]))
        xs = Wdf[f"{k}_h"].to_numpy()
        ys = Wdf[f"{k}_a"].to_numpy()
        oks = np.isfinite(xs) & np.isfinite(ys)
        cs = float(np.mean(xs[oks] * ys[oks]) / np.sqrt(np.mean(xs[oks] ** 2) * np.mean(ys[oks] ** 2)))
        rows.append({"rate": k, "act_resid_corr": ca, "act_se_boot": float(np.std(bs)),
                     "sim_within_corr": cs, "gap": ca - cs, "n_act": int(ok.sum())})
    return rows


C_all = shared_table(Rr, Wg, RATES + ["e"])
C_lnN = []
for k in RATES:
    for s in ("h", "a"):
        x = Rr[f"{k}_{s}"].to_numpy()
        ok = np.isfinite(x)
        xs = Wg[f"{k}_{s}"].to_numpy()
        oks = np.isfinite(xs)
        C_lnN.append({"rate": f"{k}_{s}",
                      "act_corr_with_lnN": pcorr(x[ok], Rr["lnN"].to_numpy()[ok]),
                      "sim_corr_with_lnN": float(np.mean(xs[oks] * Wg["lnN"].to_numpy()[oks]) /
                                                 np.sqrt(np.mean(xs[oks] ** 2) * np.mean(Wg["lnN"].to_numpy()[oks] ** 2)))})

# month-centred robustness: remove any month-level common residual (league drift)
Rm = Rr.copy()
Rm[COLS] = Rr[COLS] - Rr[COLS].groupby(month.to_numpy()).transform("mean")
C_month = shared_table(Rm, Wg, RATES + ["e"], boot=0)
Bact_m = moments(lambda c: Rm[c].to_numpy(), centred=False)

# ------------------------------------------------------------------ (D) factor attribution of Cov(ln e_h, ln e_a)
def factor_attr(df, centred):
    """ln e_i = sum_j b_j z_ij + u_i, OLS on pooled team-games of this side's
    deviations. Returns the exact split of Cov(ln e_h, ln e_a)."""
    f = (lambda x, y: float(np.mean(x * y))) if centred else pcov
    Zh = np.column_stack([df[f"{k}_h"].to_numpy() for k in FACTORS])
    Za = np.column_stack([df[f"{k}_a"].to_numpy() for k in FACTORS])
    yh, ya = df["ln_e_h"].to_numpy(), df["ln_e_a"].to_numpy()
    ok = np.isfinite(Zh).all(1) & np.isfinite(Za).all(1) & np.isfinite(yh) & np.isfinite(ya)
    Zh, Za, yh, ya = Zh[ok], Za[ok], yh[ok], ya[ok]
    Z = np.vstack([Zh, Za])
    y = np.concatenate([yh, ya])
    Zc = Z - (0 if centred else Z.mean(0))
    yc = y - (0 if centred else y.mean())
    b, *_ = np.linalg.lstsq(Zc, yc, rcond=None)
    uh = yh - (0 if centred else yh.mean()) - (Zh - (0 if centred else Zh.mean(0))) @ b
    ua = ya - (0 if centred else ya.mean()) - (Za - (0 if centred else Za.mean(0))) @ b
    r2 = 1 - np.mean(np.concatenate([uh, ua]) ** 2) / np.mean(yc ** 2)
    total = f(yh, ya)
    contrib = {}
    for j, kj in enumerate(FACTORS):
        v = 0.0
        for k, kk in enumerate(FACTORS):
            v += b[j] * b[k] * 0.5 * (f(Zh[:, j], Za[:, k]) + f(Zh[:, k], Za[:, j]))
        contrib[kj] = v
    diag = {kj: b[j] ** 2 * f(Zh[:, j], Za[:, j]) for j, kj in enumerate(FACTORS)}
    cross_u = sum(b[j] * (f(Zh[:, j], ua) + f(uh, Za[:, j])) for j in range(len(FACTORS)))
    uu = f(uh, ua)
    return {"beta": dict(zip(FACTORS, map(float, b))), "r2": float(r2), "total": total,
            "contrib": contrib, "diag_only": diag, "factor_x_resid": float(cross_u),
            "resid_resid": float(uu), "closure": total - sum(contrib.values()) - cross_u - uu,
            "n": int(ok.sum())}


Dact = factor_attr(Rr, centred=False)
Dsim = factor_attr(Wg, centred=True)
Dact_m = factor_attr(Rm, centred=False)

# ------------------------------------------------------------------ (E) segments: residual/within corr of pts and of key rates
def seg_rows(mask_act, mask_sim, label):
    out = {"segment": label, "n_games": int(mask_act.sum())}
    for k in ["pts", "ln_e", "ftr", "tovp", "efg", "orebp"]:
        x, y = Rr.loc[mask_act, f"{k}_h"].to_numpy(), Rr.loc[mask_act, f"{k}_a"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        out[f"act_{k}"] = pcorr(x[ok], y[ok])
        xs, ys = Wg.loc[mask_sim, f"{k}_h"].to_numpy(), Wg.loc[mask_sim, f"{k}_a"].to_numpy()
        oks = np.isfinite(xs) & np.isfinite(ys)
        out[f"sim_{k}"] = float(np.mean(xs[oks] * ys[oks]) / np.sqrt(np.mean(xs[oks] ** 2) * np.mean(ys[oks] ** 2)))
    out["se_approx"] = float(1 / np.sqrt(max(out["n_games"] - 3, 1)))
    out["underpowered"] = bool(out["se_approx"] > 0.05)
    return out


seg = []
gid_sim = Wg["game_id"].to_numpy()
neutral = fin["neutral"].astype(bool)
seg.append(seg_rows(~neutral.to_numpy(), ~np.isin(gid_sim, common[neutral.to_numpy()]), "site: home/away"))
seg.append(seg_rows(neutral.to_numpy(), np.isin(gid_sim, common[neutral.to_numpy()]), "site: neutral"))
for mo in sorted(month.unique()):
    mk = (month == mo).to_numpy()
    seg.append(seg_rows(mk, np.isin(gid_sim, common[mk]), f"month {int(mo)}"))
# expected-pace tercile and expected-FTR (whistle) tercile, as-of
for col, lab in (("N", "as-of pace"), ("ftr_h", "as-of home FTR")):
    q = pd.qcut(Mc[col], 3, labels=False).to_numpy()
    for t in range(3):
        mk = q == t
        seg.append(seg_rows(mk, np.isin(gid_sim, common[mk]), f"{lab} tercile {t + 1}"))
# per-team: residual corr of the team's own games (home or away) -- median and
# share of teams whose actual exceeds its sim, labelled underpowered (n~30/team)
team_rows = []
tid_h = fin["home_team_id"].to_numpy()
tid_a = fin["away_team_id"].to_numpy()
for t in np.unique(np.concatenate([tid_h, tid_a])):
    mk = (tid_h == t) | (tid_a == t)
    if mk.sum() < 20:
        continue
    ca = pcorr(Rr.loc[mk, "pts_h"].to_numpy(), Rr.loc[mk, "pts_a"].to_numpy())
    ms = np.isin(gid_sim, common[mk])
    xs, ys = Wg.loc[ms, "pts_h"].to_numpy(), Wg.loc[ms, "pts_a"].to_numpy()
    cs = float(np.mean(xs * ys) / np.sqrt(np.mean(xs ** 2) * np.mean(ys ** 2)))
    team_rows.append((int(t), int(mk.sum()), ca, cs))
tr = pd.DataFrame(team_rows, columns=["team", "n", "act", "sim"])
TEAM = {"n_teams": int(len(tr)), "median_n": float(tr["n"].median()),
        "act_median": float(tr["act"].median()), "sim_median": float(tr["sim"].median()),
        "share_act_gt_sim": float((tr["act"] > tr["sim"]).mean()),
        "sim_sd_across_teams": float(tr["sim"].std()), "act_sd_across_teams": float(tr["act"].std()),
        "per_team_se_approx": float(1 / np.sqrt(tr["n"].median() - 3))}

# ------------------------------------------------------------------ (F) exact points-scale split of the residual covariance
# Per row, with per-game as-of means m (sim 200-seed means):
#   dpts_i = m_e,i dN + m_N de_FG,i + m_N de_FT,i + xi_i      (xi = exact remainder)
#   e_FG = FG points / N = vol * val   (vol = FGA/N, val = FG points/FGA)
#   e_FT = FTM / N       = fta_pp * ft_pct
# and the same one level down for e_FG and e_FT.  Cov of the sums = sum of
# pairwise covariances, so every table closes exactly.
def build_parts(df, Mg, gid, centred):
    """df: per-row actual or sim values (pts_h, N, fgpts_h, ftm_h, fga_h, fta_h, ... ),
    Mg: per-game means aligned to rows."""
    f = (lambda x, y: float(np.mean(x * y))) if centred else pcov
    out = {}
    parts = {}
    for s in ("h", "a"):
        dN = df["N"] - Mg["N"]
        me = Mg[f"e_{s}"]
        mN = Mg["N"]
        eFG = df[f"fgpts_{s}"] / df["N"]
        eFT = df[f"ftm_{s}"] / df["N"]
        dFG = eFG - Mg[f"efgpp_{s}"]
        dFT = eFT - Mg[f"eftpp_{s}"]
        dpts = df[f"pts_{s}"] - Mg[f"pts_{s}"]
        P = me * dN
        B = mN * dFG
        C = mN * dFT
        xi = dpts - P - B - C
        # FG one level down: e_FG = vol * val
        vol = df[f"fga_{s}"] / df["N"]
        val = df[f"fgpts_{s}"] / df[f"fga_{s}"]
        dvol, dval = vol - Mg[f"vol_{s}"], val - Mg[f"val_{s}"]
        Bv = mN * Mg[f"val_{s}"] * dvol
        Bs = mN * Mg[f"vol_{s}"] * dval
        Bx = B - Bv - Bs
        # FT one level down: e_FT = fta_pp * ft_pct (ft_pct := FTM/FTA, 0 when FTA=0)
        ftapp = df[f"fta_{s}"] / df["N"]
        ftp = np.where(df[f"fta_{s}"] > 0, df[f"ftm_{s}"] / np.maximum(df[f"fta_{s}"], 1), 0.0)
        dfa, dfp = ftapp - Mg[f"ftapp_{s}"], ftp - Mg[f"ftp_{s}"]
        Cw = mN * Mg[f"ftp_{s}"] * dfa
        Cp = mN * Mg[f"ftapp_{s}"] * dfp
        Cx = C - Cw - Cp
        parts[s] = {k: np.asarray(v, float) for k, v in dict(
            pace=P, fg_vol=Bv, fg_val=Bs, fg_x=Bx, ft_whistle=Cw, ft_pct=Cp, ft_x=Cx, rem=xi, dpts=dpts).items()}
    names = ["pace", "fg_vol", "fg_val", "fg_x", "ft_whistle", "ft_pct", "ft_x", "rem"]
    M2 = {}
    for x in names:
        for y in names:
            M2[(x, y)] = f(parts["h"][x], parts["a"][y])
    total = f(parts["h"]["dpts"], parts["a"]["dpts"])

    def grp(xs, ys):
        v = 0.0
        for x in xs:
            for y in ys:
                v += M2[(x, y)]
                if set(xs) != set(ys):
                    v += M2[(y, x)]
        return v

    FG = ["fg_vol", "fg_val", "fg_x"]
    FT = ["ft_whistle", "ft_pct", "ft_x"]
    out["total"] = total
    out["pace x pace"] = M2[("pace", "pace")]
    out["pace x efficiency (both directions)"] = grp(["pace"], FG + FT)
    out["FG x FG"] = sum(M2[(x, y)] for x in FG for y in FG)
    out["FT x FT"] = sum(M2[(x, y)] for x in FT for y in FT)
    out["FG x FT (both directions)"] = sum(M2[(x, y)] + M2[(y, x)] for x in FG for y in FT)
    out["remainder (second-order, any pair with xi)"] = sum(
        M2[(x, "rem")] + M2[("rem", x)] for x in names if x != "rem") + M2[("rem", "rem")]
    out["closure"] = total - sum(v for k, v in out.items() if k not in ("total",))
    # one level down
    out["detail"] = {
        "FG: volume x volume (shots per possession: TOV, OREB)": M2[("fg_vol", "fg_vol")],
        "FG: value x value (shooting environment)": M2[("fg_val", "fg_val")],
        "FG: volume x value (both directions)": M2[("fg_vol", "fg_val")] + M2[("fg_val", "fg_vol")],
        "FG: terms with the FG product remainder": sum(M2[(x, "fg_x")] + M2[("fg_x", x)] for x in FG if x != "fg_x") + M2[("fg_x", "fg_x")],
        "FT: whistle x whistle (FTA per possession)": M2[("ft_whistle", "ft_whistle")],
        "FT: FT% x FT%": M2[("ft_pct", "ft_pct")],
        "FT: whistle x FT% (both directions)": M2[("ft_whistle", "ft_pct")] + M2[("ft_pct", "ft_whistle")],
        "FT: terms with the FT product remainder": sum(M2[(x, "ft_x")] + M2[("ft_x", x)] for x in FT if x != "ft_x") + M2[("ft_x", "ft_x")],
    }
    for s in ("h", "a"):
        out[f"var_pts_{s}"] = f(parts[s]["dpts"], parts[s]["dpts"])
    return out


def add_cols(df, fgpts_h, fgpts_a, ftm_h, ftm_a, fga_h, fga_a, fta_h, fta_a):
    df = df.copy()
    for s, fp, fm, fa, ft in (("h", fgpts_h, ftm_h, fga_h, fta_h), ("a", fgpts_a, ftm_a, fga_a, fta_a)):
        df[f"fgpts_{s}"], df[f"ftm_{s}"], df[f"fga_{s}"], df[f"fta_{s}"] = fp, fm, fa, ft
        df[f"efgpp_{s}"] = fp / df["N"]
        df[f"eftpp_{s}"] = fm / df["N"]
        df[f"vol_{s}"] = fa / df["N"]
        df[f"val_{s}"] = fp / fa
        df[f"ftapp_{s}"] = ft / df["N"]
        df[f"ftp_{s}"] = np.where(ft > 0, fm / np.maximum(ft, 1), 0.0)
    return df


SX = sim[["game_id", "N", "pts_h", "pts_a", "e_h", "e_a"]].copy()
SX = add_cols(SX, S["h"]["pts"] - S["h"]["ftm"], S["a"]["pts"] - S["a"]["ftm"], S["h"]["ftm"], S["a"]["ftm"],
              S["h"]["fga"], S["a"]["fga"], S["h"]["fta"], S["a"]["fta"])
SX = SX[SX["game_id"].isin(common)].reset_index(drop=True)
MCOLS = [c for c in SX.columns if c != "game_id"]
MG = SX.groupby("game_id")[MCOLS].mean()
MGrow = MG.loc[SX["game_id"]].reset_index(drop=True)
AX = act[["N", "pts_h", "pts_a", "e_h", "e_a"]].copy()
AX = add_cols(AX, (act["pts_h"] - h["ftm"].to_numpy()).to_numpy(), (act["pts_a"] - a["ftm"].to_numpy()).to_numpy(),
              h["ftm"].to_numpy(float), a["ftm"].to_numpy(float), h["fga"].to_numpy(float),
              a["fga"].to_numpy(float), h["fta"].to_numpy(float), a["fta"].to_numpy(float))
MGa = MG.loc[common]
F_act = build_parts(AX.reset_index(drop=True), MGa.reset_index(drop=True), None, centred=False)
F_sim = build_parts(SX, MGrow, None, centred=True)
F = {"act": F_act, "sim": F_sim, "D_act": D_a, "D_sim": D_s}
# bootstrap SE of the actual terms (games resampled)
bs = {k: [] for k in F_act if k not in ("detail", "closure")}
bsd = {k: [] for k in F_act["detail"]}
AXr, MGr = AX.reset_index(drop=True), MGa.reset_index(drop=True)
for _ in range(100):
    ii = RNG.integers(0, len(AXr), len(AXr))
    fb = build_parts(AXr.iloc[ii].reset_index(drop=True), MGr.iloc[ii].reset_index(drop=True), None, centred=False)
    for k in bs:
        bs[k].append(fb[k])
    for k in bsd:
        bsd[k].append(fb["detail"][k])
F["act_se"] = {k: float(np.std(v)) for k, v in bs.items()}
F["act_detail_se"] = {k: float(np.std(v)) for k, v in bsd.items()}

rep = {"F_points_scale_split": F, "A_gate_corr_decomp": A, "B_moments": B, "B_moments_act_month_centred": Bact_m,
       "C_shared_rates": C_all, "C_shared_rates_month_centred": C_month, "C_rate_vs_pace": C_lnN,
       "D_factor_attr_act": Dact, "D_factor_attr_sim": Dsim, "D_factor_attr_act_month_centred": Dact_m,
       "E_segments": seg, "E_per_team": TEAM,
       "means": {"act_N": float(act["N"].mean()), "sim_N": float(sim["N"].mean()),
                 "act_T": float(act["T"].mean()), "sim_T": float(sim["T"].mean())}}
OUT.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
print(json.dumps({"A": A, "B_total_sd_ratio_pop": B["total_sd_ratio_pop"]}, indent=1, default=float))
