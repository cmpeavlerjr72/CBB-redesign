"""diag_g5_variance_parts_v1.py -- sub-model attribution of the G5 total-variance gap.

Variance worker, 2026-10-05. DIAGNOSTIC ONLY (companion to diag_g5_variance_decomp_v1.py).
Same objects (m = sim seed mean per game, w = sim - m, r = actual - m), box possession
estimator on both sides. Splits each team's PPP deviation one level down, exactly:

  e_i = FGpts_i/N + FTM_i/N
  FGpts/N = vol * val    (vol = FGA/N  [owner: PO mix -- TOV / FT-trip / OREB extra FGA],
                          val = FGpts/FGA [owner: fg_make + shot mix])
  FTM/N   = fta_pp * ft_pct  (fta_pp = FTA/N [owner: foul / FT-trip], ft_pct [owner: free_throw])
  Q_i^k = m_N * (first-order term k) ; x = exact remainder.

Reports, actual vs sim:
  (1) the shared-PPP term 4 Cov(Q_h, Q_a) split into the 4x4 cross-team blocks
      (vol, val, fta, ftp, x) -> which channel carries the shared game-level component;
  (2) the pace x efficiency term 2 Cov(P, Q_h + Q_a) split by channel;
  (3) the same decomposition on regulation-only games (n_periods == 2) on both sides
      and the OT mixture share of Var(dT) (G7 OT rate 0.031 vs 0.056 is a known defect);
  (4) a team-block bootstrap SE on the realised persistent-team share.
Writes results/g5_vdecomp/<tag>_parts.json (gitignored).
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
args = ap.parse_args()
OUT = ROOT / "results/g5_vdecomp" / f"{args.tag}_parts.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261005)


def pcov(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(np.mean((x - x.mean()) * (y - y.mean())))


g = pd.read_parquet(Path(args.run) / "games.parquet")
summary, raw = G.build_grading_frame(g, args.season)
summary = summary[summary["home_score"].fillna(0) + summary["away_score"].fillna(0) > 0]
gids = summary["game_id"].to_numpy()
raw = raw[raw["game_id"].isin(gids)].reset_index(drop=True)


def frame_sim(raw):
    d = pd.DataFrame({"game_id": raw["game_id"].to_numpy(), "nper": raw["n_periods"].to_numpy(int)})
    est = {}
    for s, o in (("h", "home"), ("a", "away")):
        fga = (raw[f"{o}_fga3"] + raw[f"{o}_fga2_rim"] + raw[f"{o}_fga2_jump"]).to_numpy(float)
        fgpts = (2 * raw[f"{o}_fgm2_rim"] + 2 * raw[f"{o}_fgm2_jump"] + 3 * raw[f"{o}_fgm3"]).to_numpy(float)
        d[f"fga_{s}"], d[f"fgpts_{s}"] = fga, fgpts
        d[f"fta_{s}"], d[f"ftm_{s}"] = raw[f"{o}_fta"].to_numpy(float), raw[f"{o}_ftm"].to_numpy(float)
        d[f"pts_{s}"] = raw[f"{o}_pts"].to_numpy(float)
        est[s] = fga - raw[f"{o}_oreb"].to_numpy(float) + raw[f"{o}_tov"].to_numpy(float) + 0.44 * d[f"fta_{s}"]
    d["N"] = 0.5 * (est["h"] + est["a"])
    return d


def frame_act(summary, season):
    tb = R.load_actual_team_box(season)
    tb = tb[tb["game_id"].isin(summary["game_id"])]
    h = tb[tb["team_id"] == tb["home_team_id"]].drop_duplicates("game_id").set_index("game_id")
    a = tb[tb["team_id"] == tb["away_team_id"]].drop_duplicates("game_id").set_index("game_id")
    fin = summary.set_index("game_id")
    common = fin.index.intersection(h.index).intersection(a.index)
    fin, h, a = fin.loc[common], h.loc[common], a.loc[common]
    d = pd.DataFrame({"game_id": common.to_numpy(), "nper": fin["n_periods"].fillna(2).to_numpy(int)})
    est = {}
    for s, t, pts in (("h", h, fin["home_score"]), ("a", a, fin["away_score"])):
        d[f"fga_{s}"] = t["fga"].to_numpy(float)
        d[f"fgpts_{s}"] = (2 * (t["fgm"] - t["tpm"]) + 3 * t["tpm"]).to_numpy(float)
        d[f"fta_{s}"], d[f"ftm_{s}"] = t["fta"].to_numpy(float), t["ftm"].to_numpy(float)
        d[f"pts_{s}"] = pts.to_numpy(float)
        est[s] = (t["fga"] - t["oreb"] + t["tov"] + 0.44 * t["fta"]).to_numpy(float)
    d["N"] = 0.5 * (est["h"] + est["a"])
    return d, fin


def derived(d):
    for s in ("h", "a"):
        d[f"e_{s}"] = d[f"pts_{s}"] / d["N"]
        d[f"vol_{s}"] = d[f"fga_{s}"] / d["N"]
        d[f"val_{s}"] = d[f"fgpts_{s}"] / d[f"fga_{s}"]
        d[f"fta_pp_{s}"] = d[f"fta_{s}"] / d["N"]
        d[f"ftp_{s}"] = np.where(d[f"fta_{s}"] > 0, d[f"ftm_{s}"] / np.maximum(d[f"fta_{s}"], 1), 0.0)
    d["T"] = d["pts_h"] + d["pts_a"]
    return d


VARS = ["T", "N"] + [f"{k}_{s}" for k in ("pts", "e", "vol", "val", "fta_pp", "ftp") for s in ("h", "a")]
S = derived(frame_sim(raw))
A, fin = derived(frame_act(summary, args.season)[0]), frame_act(summary, args.season)[1]
S = S[S["game_id"].isin(A["game_id"])].reset_index(drop=True)
M = S.groupby("game_id")[VARS].mean()
MA = M.loc[A["game_id"]].reset_index(drop=True)
MS = M.loc[S["game_id"]].reset_index(drop=True)
CH = ["vol", "val", "fta", "ftp", "x"]


def parts(D, Mg):
    dN = (D["N"] - Mg["N"]).to_numpy(float)
    P = (Mg["e_h"] + Mg["e_a"]).to_numpy() * dN
    mN = Mg["N"].to_numpy()
    out = {"P": P, "dT": (D["T"] - Mg["T"]).to_numpy(float)}
    for s in ("h", "a"):
        de = (D[f"e_{s}"] - Mg[f"e_{s}"]).to_numpy(float)
        dvol = (D[f"vol_{s}"] - Mg[f"vol_{s}"]).to_numpy(float)
        dval = (D[f"val_{s}"] - Mg[f"val_{s}"]).to_numpy(float)
        dfa = (D[f"fta_pp_{s}"] - Mg[f"fta_pp_{s}"]).to_numpy(float)
        dfp = (D[f"ftp_{s}"] - Mg[f"ftp_{s}"]).to_numpy(float)
        q = {"vol": mN * Mg[f"val_{s}"].to_numpy() * dvol, "val": mN * Mg[f"vol_{s}"].to_numpy() * dval,
             "fta": mN * Mg[f"ftp_{s}"].to_numpy() * dfa, "ftp": mN * Mg[f"fta_pp_{s}"].to_numpy() * dfp}
        Q = mN * de
        q["x"] = Q - sum(q.values())
        out[f"Q_{s}"] = Q
        for k, v in q.items():
            out[f"{k}_{s}"] = v
    return out


def summarise(C, centred, mask=None):
    f = (lambda x, y: float(np.mean(x * y))) if centred else pcov
    C = {k: v if mask is None else v[mask] for k, v in C.items()}
    Qs = C["Q_h"] + C["Q_a"]
    shared = {}
    for k1 in CH:
        for k2 in CH:
            # symmetric cross-team block in total-variance units: entry (k1,k2) = 2*(Cov(h_k1,a_k2) + Cov(h_k2,a_k1)),
            # so diag = 4 Cov(h_k, a_k) and the (k1,k2)+(k2,k1) pair = 4 (Cov12 + Cov21); all entries sum to 4 Cov(Q_h, Q_a)
            shared[f"{k1}x{k2}"] = 2 * (f(C[f"{k1}_h"], C[f"{k2}_a"]) + f(C[f"{k2}_h"], C[f"{k1}_a"]))
    sh_total = 4 * f(C["Q_h"], C["Q_a"])
    # collapse to own-channel diagonal + off-diagonal
    shared_diag = {k: 4 * f(C[f"{k}_h"], C[f"{k}_a"]) for k in CH}
    shared_off = sh_total - sum(shared_diag.values())
    pace = {k: 2 * f(C["P"], C[f"{k}_h"] + C[f"{k}_a"]) for k in CH}
    return {"var_T": f(C["dT"], C["dT"]), "var_P": f(C["P"], C["P"]), "shared_blocks": shared,
            "shared_total": sh_total, "shared_own_channel": shared_diag, "shared_cross_channel": shared_off,
            "pace_x_eff_total": 2 * f(C["P"], Qs), "pace_x_eff_by_channel": pace,
            "team_var_by_channel": {k: f(C[f"{k}_h"], C[f"{k}_h"]) + f(C[f"{k}_a"], C[f"{k}_a"]) for k in CH},
            "n": int(len(C["dT"]))}


CA, CS = parts(A, MA), parts(S, MS)
res = {"tag": args.tag, "overall": {"act": summarise(CA, False), "sim": summarise(CS, True)}}

# regulation-only on both sides (selection on n_periods == 2)
regA = A["nper"].to_numpy() == 2
regS = S["nper"].to_numpy() == 2
res["regulation_only"] = {"act": summarise(CA, False, regA), "sim": summarise(CS, True, regS),
                          "ot_rate_act": float(1 - regA.mean()), "ot_rate_sim": float(1 - regS.mean())}
# OT mixture share: Var(dT) = E[Var|OT] + Var(E[.|OT]) ; report the OT-induced part
for lab, C, ot, centred in (("act", CA, ~regA, False), ("sim", CS, ~regS, True)):
    dT = C["dT"]
    p = ot.mean()
    mu1, mu0 = dT[ot].mean(), dT[~ot].mean()
    res["regulation_only"][f"ot_between_var_{lab}"] = float(p * (1 - p) * (mu1 - mu0) ** 2)
    res["regulation_only"][f"ot_mean_shift_{lab}"] = float(mu1 - mu0)
    res["regulation_only"][f"ot_within_var_{lab}"] = float(dT[ot].var())

# persistent-team share of realised Var(dT), team-block bootstrap
hid, aid = fin.loc[A["game_id"], "home_team_id"].to_numpy(), fin.loc[A["game_id"], "away_team_id"].to_numpy()
month = fin.loc[A["game_id"], "month"].to_numpy()
x = CA["dT"] - pd.Series(CA["dT"]).groupby(month).transform("mean").to_numpy()
teams = np.unique(np.concatenate([hid, aid]))
tsum, tss, tn = {}, {}, {}
for t in teams:
    v = x[(hid == t) | (aid == t)]
    tsum[t], tss[t], tn[t] = v.sum(), (v * v).sum(), len(v)


def s2(ts):
    num = sum(0.5 * (tsum[t] ** 2 - tss[t]) for t in ts)
    den = sum(tn[t] * (tn[t] - 1) / 2 for t in ts)
    return num / den


point = 2 * s2(teams)
bs = [2 * s2(RNG.choice(teams, size=len(teams), replace=True)) for _ in range(300)]
res["persistent_team_T"] = {"var_pts2": float(point), "se_team_boot": float(np.std(bs)),
                            "share_of_var_T": float(point / x.var())}

# persistent-team part of the realised possessions term P, all games and regulation-only
def pers_of(v, mask):
    xm = v - pd.Series(v).groupby(month).transform("mean").to_numpy()
    xm = np.where(mask, xm, 0.0)
    num = den = 0.0
    for t in teams:
        sel = ((hid == t) | (aid == t)) & mask
        u = xm[sel]
        if len(u) < 2:
            continue
        num += 0.5 * (u.sum() ** 2 - (u * u).sum())
        den += len(u) * (len(u) - 1) / 2
    return 2 * num / den


res["persistent_team_P"] = {"all": float(pers_of(CA["P"], np.ones(len(regA), bool))),
                            "regulation_only": float(pers_of(CA["P"], regA)),
                            "Qsum_all": float(pers_of(CA["Q_h"] + CA["Q_a"], np.ones(len(regA), bool)))}
OUT.write_text(json.dumps(res, indent=1, default=float))
print("persistent P/Q", res["persistent_team_P"])


def show(lbl, d):
    print(f"== {lbl}: n={d['n']} var_T {d['var_T']:.1f} var_P {d['var_P']:.1f} shared {d['shared_total']:.1f} "
          f"cross-ch {d['shared_cross_channel']:.1f} pace_x_eff {d['pace_x_eff_total']:.1f}")
    print("   shared own-ch ", {k: round(v, 1) for k, v in d["shared_own_channel"].items()})
    print("   pace x eff ch ", {k: round(v, 1) for k, v in d["pace_x_eff_by_channel"].items()})
    bl = {k: v for k, v in d["shared_blocks"].items() if k.split("x")[0] <= k.split("x")[1]}
    bl = {k: (v if k.split("x")[0] == k.split("x")[1] else 2 * v) for k, v in bl.items()}
    print("   shared blocks ", {k: round(v, 1) for k, v in sorted(bl.items(), key=lambda kv: -abs(kv[1]))[:15]})
    print("   team var ch   ", {k: round(v, 1) for k, v in d["team_var_by_channel"].items()})


for k in ("act", "sim"):
    show(f"{args.tag} overall {k}", res["overall"][k])
for k in ("act", "sim"):
    show(f"{args.tag} regulation-only {k}", res["regulation_only"][k])
print({k: v for k, v in res["regulation_only"].items() if not isinstance(v, dict)})
print("persistent", res["persistent_team_T"])
