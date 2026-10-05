"""diag_g5_variance_decomp_v1.py -- G5 total-variance decomposition on served stack v2.

Variance worker, 2026-10-05. DIAGNOSTIC ONLY: reads a finished served-v2 sim run
and the season's verified finals / hoopR team box; fits nothing that is served,
changes no default, writes `results/g5_vdecomp/<tag>.json` (gitignored).

Objects (population moments, ddof=0, identities close exactly):
  m  per-game as-of expectation = the sim's own seed mean (no same-game info).
  w  sim within-game deviation  = sim row - m.
  r  realised residual          = actual - m.

Points-scale split of the total's deviation dT (same definitions both sides):
  N   = game possessions (box estimator FGA - OREB + TOV + 0.44 FTA, mean of sides)
  e_i = pts_i / N
  P   = (m_e_h + m_e_a) * dN                (a) possessions
  Q_i = m_N * de_i                          per-team PPP channel
  rem = dT - P - Q_h - Q_a                  nonlinearity (dN * de)
  Var(dT) = Var(P)                          (a) possessions
          + 4 Cov(Q_h, Q_a)                 (b) shared PPP (game-level efficiency)
          + Var(Q_h) + Var(Q_a) - 2Cov      (c) team-specific PPP
          + 2 Cov(P, Q_h + Q_a)             (d) pace x efficiency covariance
          + Var(rem) + 2 Cov(rem, P + Q)    (e) nonlinearity remainder
Realised only: the part of each component that is a PERSISTENT team effect
(mean-model error repeated across a team's games, which a within-game variance
function cannot and should not carry) is estimated by method of moments from
pairs of games sharing a team: sigma2_team = mean over same-team pairs of x_i x_j
(x = the component's per-game contribution, month-centred); a game has two
teams, so the persistent share is 2 * sigma2_team (variance channels only).

Usage: diag_g5_variance_decomp_v1.py --run <dir> --season 2025 --tag F2
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
from cbb_sim.eval import gates as G  # noqa: E402
from cbb_sim.eval import reference as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--season", type=int, required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--boot", type=int, default=200)
args = ap.parse_args()
OUT = ROOT / "results/g5_vdecomp" / f"{args.tag}.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261005)


def pcov(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def side(pts, fga, fgm, fg3a, fg3m, fta, ftm, oreb, dreb_opp, tov, n):
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "pts": pts, "e": pts / n,
            "efg": (fgm + 0.5 * fg3m) / fga, "tovp": tov / n,
            "orebp": oreb / (oreb + dreb_opp), "ftr": fta / fga, "ftapp": fta / n,
            "fg3_share": fg3a / fga,
            "ft_pct": np.where(fta > 0, ftm / np.maximum(fta, 1), np.nan),
            "fg3_pct": np.where(fg3a > 0, fg3m / np.maximum(fg3a, 1), np.nan),
            "fg2_pct": (fgm - fg3m) / np.maximum(fga - fg3a, 1), "fgapp": fga / n,
        }


RATES = ["e", "efg", "fg2_pct", "fg3_pct", "fg3_share", "ft_pct", "ftr", "ftapp", "tovp", "orebp", "fgapp"]

# ------------------------------------------------------------------ sim
g = pd.read_parquet(Path(args.run) / "games.parquet")
summary, raw = G.build_grading_frame(g, args.season)
summary = summary[summary["home_score"].fillna(0) + summary["away_score"].fillna(0) > 0]
gids = summary["game_id"].to_numpy()
raw = raw[raw["game_id"].isin(gids)].copy()
gate_ratio = float(summary["sim_total_sd"].mean() / (summary["total"] - summary["sim_total_mean"]).std())
gate_corr_sim = float(np.corrcoef(raw["home_pts"].astype(float), raw["away_pts"].astype(float))[0, 1])
gate_corr_act = float(np.corrcoef(summary["home_score"].astype(float), summary["away_score"].astype(float))[0, 1])

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
sim = pd.DataFrame({"game_id": raw["game_id"].to_numpy(), "N": N_sim, "Neng": raw["possessions"].to_numpy(float)})
for s in ("h", "a"):
    d = S[s]
    for k, v in side(d["pts"], d["fga"], d["fgm"], d["fg3a"], d["fg3m"], d["fta"], d["ftm"],
                     d["oreb"], d["dreb_opp"], d["tov"], N_sim).items():
        sim[f"{k}_{s}"] = v
sim["T"] = sim["pts_h"] + sim["pts_a"]
COLS = ["T", "N"] + [f"{k}_{s}" for k in ["pts"] + RATES for s in ("h", "a")]
M = sim.groupby("game_id")[COLS].mean()

# ------------------------------------------------------------------ actual
tb = R.load_actual_team_box(args.season)
tb = tb[tb["game_id"].isin(gids)]
h = tb[tb["team_id"] == tb["home_team_id"]].drop_duplicates("game_id").set_index("game_id")
a = tb[tb["team_id"] == tb["away_team_id"]].drop_duplicates("game_id").set_index("game_id")
fin = summary.set_index("game_id")
common = fin.index.intersection(h.index).intersection(a.index)
fin, h, a = fin.loc[common], h.loc[common], a.loc[common]
N_act = 0.5 * ((h["fga"] - h["oreb"] + h["tov"] + 0.44 * h["fta"]) +
               (a["fga"] - a["oreb"] + a["tov"] + 0.44 * a["fta"])).to_numpy(float)
act = pd.DataFrame(index=common)
act["N"] = N_act
for s, t, pts, opp in (("h", h, fin["home_score"], a), ("a", a, fin["away_score"], h)):
    for k, v in side(pts.to_numpy(float), t["fga"].to_numpy(float), t["fgm"].to_numpy(float),
                     t["tpa"].to_numpy(float), t["tpm"].to_numpy(float), t["fta"].to_numpy(float),
                     t["ftm"].to_numpy(float), t["oreb"].to_numpy(float), opp["dreb"].to_numpy(float),
                     t["tov"].to_numpy(float), N_act).items():
        act[f"{k}_{s}"] = v
act["T"] = act["pts_h"] + act["pts_a"]
Mc = M.loc[common]
Rr = act[COLS] - Mc[COLS]
sim = sim[sim["game_id"].isin(common)].reset_index(drop=True)
Mrow = M.loc[sim["game_id"]].reset_index(drop=True)
W = sim[COLS] - Mrow[COLS]
month = fin["month"].to_numpy()
neutral = fin["neutral"].astype(bool).to_numpy()
hid, aid = fin["home_team_id"].to_numpy(), fin["away_team_id"].to_numpy()
gid_sim = sim["game_id"].to_numpy()
n_games = len(common)
pos_of = pd.Series(np.arange(n_games), index=common)
gpos_sim = pos_of.loc[gid_sim].to_numpy()

# ------------------------------------------------------------------ components per row
def comps(D, Mg):
    """D: deviations (T, N, e_h, e_a ...); Mg: per-row game means."""
    dN = D["N"].to_numpy(float)
    P = (Mg["e_h"].to_numpy() + Mg["e_a"].to_numpy()) * dN
    Qh = Mg["N"].to_numpy() * D["e_h"].to_numpy(float)
    Qa = Mg["N"].to_numpy() * D["e_a"].to_numpy(float)
    dT = D["T"].to_numpy(float)
    rem = dT - P - Qh - Qa
    return dict(dT=dT, P=P, Qh=Qh, Qa=Qa, rem=rem, dN=dN)


def decomp(C, centred, mask=None):
    f = (lambda x, y: float(np.mean(x * y))) if centred else pcov
    C = {k: v if mask is None else v[mask] for k, v in C.items()}
    Q = C["Qh"] + C["Qa"]
    cov_q = f(C["Qh"], C["Qa"])
    out = {
        "var_T": f(C["dT"], C["dT"]),
        "a_poss": f(C["P"], C["P"]),
        "b_shared_ppp": 4 * cov_q,
        "c_team_ppp": f(C["Qh"], C["Qh"]) + f(C["Qa"], C["Qa"]) - 2 * cov_q,
        "d_pace_x_eff": 2 * f(C["P"], Q),
        "e_nonlin": f(C["rem"], C["rem"]) + 2 * f(C["rem"], C["P"] + Q),
        "cov_Qh_Qa": cov_q,
        "corr_Qh_Qa": cov_q / np.sqrt(f(C["Qh"], C["Qh"]) * f(C["Qa"], C["Qa"])),
        "var_dN": f(C["dN"], C["dN"]),
        "n": int(len(C["dT"])),
    }
    out["closure"] = out["var_T"] - sum(out[k] for k in ("a_poss", "b_shared_ppp", "c_team_ppp", "d_pace_x_eff", "e_nonlin"))
    out["sd_T"] = float(np.sqrt(out["var_T"]))
    return out


Ca = comps(Rr, Mc)
Cs = comps(W, Mrow)
KEYS = ["var_T", "a_poss", "b_shared_ppp", "c_team_ppp", "d_pace_x_eff", "e_nonlin"]


def boot_se(C, nb):
    if nb <= 0:
        return {}
    vals = {k: [] for k in KEYS}
    idx = np.arange(len(C["dT"]))
    for _ in range(nb):
        b = RNG.choice(idx, size=len(idx), replace=True)
        d = decomp({k: v[b] for k, v in C.items()}, centred=False)
        for k in KEYS:
            vals[k].append(d[k])
    return {k: float(np.std(v)) for k, v in vals.items()}


res = {"tag": args.tag, "run": args.run, "season": args.season, "n_games": int(n_games),
       "gate_total_sd_ratio": gate_ratio, "gate_corr_sim": gate_corr_sim, "gate_corr_act": gate_corr_act}
res["overall"] = {"act": decomp(Ca, False), "sim": decomp(Cs, True), "act_se": boot_se(Ca, args.boot)}

# month-centred realised (league drift removed)
Cam = {k: v - pd.Series(v).groupby(month).transform("mean").to_numpy() for k, v in Ca.items()}
res["overall"]["act_month_centred"] = decomp(Cam, False)

# ------------------------------------------------------------------ persistent team part of realised components
def team_sigma2(x):
    """mean over same-team game pairs of x_i * x_j (x month-centred, per game)."""
    x = np.asarray(x, float)
    num, npairs = 0.0, 0
    teams = np.unique(np.concatenate([hid, aid]))
    for t in teams:
        v = x[(hid == t) | (aid == t)]
        if len(v) < 2:
            continue
        s, ss = v.sum(), (v * v).sum()
        num += 0.5 * (s * s - ss)
        npairs += len(v) * (len(v) - 1) // 2
    return num / npairs


Qa_sum = Cam["Qh"] + Cam["Qa"]
pers = {
    "sigma2_team_T": team_sigma2(Cam["dT"]),
    "sigma2_team_P": team_sigma2(Cam["P"]),
    "sigma2_team_Qsum": team_sigma2(Qa_sum),
}
# shared-PPP persistence: per game product Qh*Qa is the shared estimator; its team-level repeat:
pers["note"] = "persistent share of Var = 2*sigma2_team (two teams per game); pairs within team, month-centred"
pers["persistent_share_var_T"] = 2 * pers["sigma2_team_T"] / res["overall"]["act_month_centred"]["var_T"]
res["persistent_team"] = pers

# ------------------------------------------------------------------ segments
def seg(mask_g, label):
    ms = mask_g[gpos_sim]
    d = {"segment": label, "n_games": int(mask_g.sum()),
         "act": decomp(Ca, False, mask_g), "sim": decomp(Cs, True, ms)}
    d["ratio_sd_T"] = float(np.sqrt(d["sim"]["var_T"] / d["act"]["var_T"]))
    d["underpowered"] = bool(mask_g.sum() < 400)
    return d


segs = [seg(~neutral, "site: home/away"), seg(neutral, "site: neutral")]
Nq = pd.qcut(Mc["N"].to_numpy(), 3, labels=False)
for t in range(3):
    segs.append(seg(Nq == t, f"as-of pace tercile {t + 1}"))
Tq = pd.qcut(Mc["T"].to_numpy(), 5, labels=False)
for t in range(5):
    segs.append(seg(Tq == t, f"as-of total quintile {t + 1}"))
mis = np.abs(Mc["pts_h"].to_numpy() - Mc["pts_a"].to_numpy())
Mq = pd.qcut(mis, 5, labels=False)
for t in range(5):
    segs.append(seg(Mq == t, f"as-of |margin| quintile {t + 1}"))
# team-prior quintile: team's season-mean as-of net (expected own pts - opp pts) -> games where HOME team in quintile
net_h = pd.Series(Mc["pts_h"].to_numpy() - Mc["pts_a"].to_numpy())
tn = pd.concat([pd.DataFrame({"t": hid, "v": net_h}), pd.DataFrame({"t": aid, "v": -net_h})]).groupby("t")["v"].mean()
tq = pd.qcut(tn, 5, labels=False)
avgq = (tq.loc[hid].to_numpy() + tq.loc[aid].to_numpy()) / 2.0
for lo, hi, lab in ((0, 1, "both-team prior avg quintile <=1"), (1.5, 2.5, "prior avg quintile 1.5-2.5"),
                    (3, 4, "prior avg quintile >=3")):
    segs.append(seg((avgq >= lo) & (avgq <= hi), f"team prior: {lab}"))
for m in sorted(np.unique(month)):
    segs.append(seg(month == m, f"month {int(m)}"))
res["segments"] = segs

# ------------------------------------------------------------------ per team: share of teams where sim var_T < act var_T
rows = []
for t in np.unique(np.concatenate([hid, aid])):
    mk = (hid == t) | (aid == t)
    if mk.sum() < 20:
        continue
    da, ds = decomp(Ca, False, mk), decomp(Cs, True, mk[gpos_sim])
    rows.append((int(t), int(mk.sum()), da["var_T"], ds["var_T"], da["b_shared_ppp"], ds["b_shared_ppp"]))
tr = pd.DataFrame(rows, columns=["team", "n", "act_vT", "sim_vT", "act_b", "sim_b"])
res["per_team"] = {"n_teams": int(len(tr)), "median_games": float(tr["n"].median()),
                   "median_ratio_sd": float(np.sqrt(tr["sim_vT"] / tr["act_vT"]).median()),
                   "share_sim_lt_act": float((tr["sim_vT"] < tr["act_vT"]).mean()),
                   "share_sim_b_lt_act_b": float((tr["sim_b"] < tr["act_b"]).mean()),
                   "label": "per-team cells ~30 games: underpowered individually; read the shares only"}

# ------------------------------------------------------------------ game-level efficiency latent: per-rate home/away residual corr
lat = []
for k in RATES + ["N"]:
    if k == "N":
        continue
    x, y = Rr[f"{k}_h"].to_numpy(float), Rr[f"{k}_a"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    xm = x - pd.Series(x).groupby(month).transform("mean").to_numpy()
    ym = y - pd.Series(y).groupby(month).transform("mean").to_numpy()
    ca = pcov(xm[ok], ym[ok]) / np.sqrt(pcov(xm[ok], xm[ok]) * pcov(ym[ok], ym[ok]))
    bs = []
    idx = np.where(ok)[0]
    for _ in range(100):
        b = RNG.choice(idx, size=len(idx), replace=True)
        bs.append(pcov(xm[b], ym[b]) / np.sqrt(pcov(xm[b], xm[b]) * pcov(ym[b], ym[b])))
    xs, ys = W[f"{k}_h"].to_numpy(float), W[f"{k}_a"].to_numpy(float)
    oks = np.isfinite(xs) & np.isfinite(ys)
    cs = float(np.mean(xs[oks] * ys[oks]) / np.sqrt(np.mean(xs[oks] ** 2) * np.mean(ys[oks] ** 2)))
    lat.append({"rate": k, "act_resid_corr_mc": float(ca), "se": float(np.std(bs)), "sim_within_corr": cs,
                "act_cov_mc": float(pcov(xm[ok], ym[ok])),
                "sim_cov": float(np.mean(xs[oks] * ys[oks]))})
res["rate_latent"] = lat
# corr of each rate residual with dN (pace x efficiency mechanism)
pl = []
for k in ["e", "efg", "ftapp", "tovp", "orebp", "fgapp", "fg3_share"]:
    for s in ("h", "a"):
        x = Rr[f"{k}_{s}"].to_numpy(float)
        ok = np.isfinite(x)
        xs = W[f"{k}_{s}"].to_numpy(float)
        oks = np.isfinite(xs)
        n_a, n_s = Rr["N"].to_numpy(float), W["N"].to_numpy(float)
        pl.append({"rate": f"{k}_{s}",
                   "act_corr_dN": pcov(x[ok], n_a[ok]) / np.sqrt(pcov(x[ok], x[ok]) * pcov(n_a[ok], n_a[ok])),
                   "sim_corr_dN": float(np.mean(xs[oks] * n_s[oks]) / np.sqrt(np.mean(xs[oks] ** 2) * np.mean(n_s[oks] ** 2)))})
res["pace_coupling"] = pl
# engine possession count vs box estimator in sim (definition check)
res["sim_N_box_vs_engine"] = {"mean_box": float(sim["N"].mean()), "mean_engine": float(sim["Neng"].mean()),
                              "within_sd_box": float(np.sqrt(np.mean(W["N"] ** 2))),
                              "within_sd_engine": float(np.sqrt(np.mean((sim["Neng"] - sim.groupby("game_id")["Neng"].transform("mean")) ** 2))),
                              "act_resid_sd_box": float(Rr["N"].std(ddof=0))}

OUT.write_text(json.dumps(res, indent=1, default=float))
o = res["overall"]
print(f"[{args.tag}] games {n_games} gate ratio {gate_ratio:.4f} corr sim {gate_corr_sim:.3f} act {gate_corr_act:.3f}")
for k in KEYS + ["cov_Qh_Qa", "corr_Qh_Qa", "var_dN", "closure"]:
    print(f"  {k:14s} act {o['act'][k]:9.3f}  actMC {o['act_month_centred'][k]:9.3f}  sim {o['sim'][k]:9.3f}  se {o['act_se'].get(k, float('nan')):.3f}")
print(" persistent", {k: v for k, v in pers.items() if k != 'note'})
print(" per_team", res["per_team"])
for d in segs:
    print(f"  {d['segment']:40s} n={d['n_games']:5d} ratioSD={d['ratio_sd_T']:.3f} "
          + " ".join(f"{k[:6]}={d['act'][k]:.1f}/{d['sim'][k]:.1f}" for k in KEYS))
for r_ in lat:
    print(f"  latent {r_['rate']:10s} act {r_['act_resid_corr_mc']:+.3f} (se {r_['se']:.3f}) sim {r_['sim_within_corr']:+.3f}")
for r_ in pl:
    print(f"  dN-coupling {r_['rate']:12s} act {r_['act_corr_dN']:+.3f} sim {r_['sim_corr_dN']:+.3f}")
print(" N defs", res["sim_N_box_vs_engine"])
