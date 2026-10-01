"""diag_g5_channels_v1.py -- G5 ownership table by channel on a sim run (lane B, 2026-10-01).

DIAGNOSTIC ONLY. Fits nothing that is served, changes no default.

Exact per-team-game points identity (every term linear, so covariances close):

  pts_i = e0 N                                  pace       (N = game possessions, box estimator, mean of sides)
        + e0 (N_i - N)                          imbalance  (own-side estimator minus game N)
        + v0 (OREB_i - o0 N_i)                  oreb       (offensive rebounds beyond the expected per-possession rate)
        - v0 (TOV_i  - t0 N_i)                  tov
        + (f0 - 0.44 v0)(FTA_i - a0 N_i)        fta        (foul / FTA volume)
        + 3 A3 p0_3 + v2 A2 - v0 FGA_i         mix3       (three-point share at expected make rates)
        + 2 (A_rim p0_rim + A_jump p0_jump) - v2 A2   mixRJ (rim / jumper split of the twos)
        + 2 (FGM_rim  - FGA_rim  p0_rim)        rim make
        + 2 (FGM_jump - FGA_jump p0_jump)       jump make
        + 3 (FGM_3    - FGA_3    p0_3)          three make
        + (FTM - FTA f0)                        ft make
        + data remainder (actual only: final points minus the box identity)

with FGA_i = N_i + OREB_i - TOV_i - 0.44 FTA_i (the box estimator, exact) and the
base constants (e0, v0, o0, t0, a0, f0, p0_k, mix0) = the sim's own per-game,
per-side 200-seed ratio-of-means (as-of: the engine reads as-of inputs only).
Base rows: `v0 = sum_k pts_k A0_k p0_k / FGA0`, `e0 = v0 (1 + o0 - t0 - 0.44 a0) + f0 a0`.

Sim: within-game deviations from the per-game mean, pooled (equal seeds per game).
Actual: residual = actual channel - the sim's per-game mean channel.

G5 correlation chain (as diag_g5_corr_v1 2.1) with the residual-covariance term split
by channel pair (h_k, a_l): act_kl / D_act - sim_kl / D_sim. Closes exactly.
Total variance: Var(T) = sum_kl Cov(C_k, C_l), C_k = c_hk + c_ak, closes exactly.

Actual rim / jumper split: box 2PA / 2PM split by the pbp (fg_make design rows)
rim share of the game-side's 2PA and of its 2PM. Games without design rows on
both sides are dropped (count printed).

Per season (real data only, 2023 / 2024 / 2025): the same channels with league-season
base constants, each channel residualised on offence team-season + defence
team-season + home/neutral fixed effects (sparse least squares); between-team
covariance of the residual channel vectors. DESCRIPTIVE: it makes a 2025-only
artefact visible; it is not the sim comparison.
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

CH = ["pace", "imbal", "oreb", "tov", "fta", "mix3", "mixRJ", "rim", "jump", "three", "ft", "rem"]
KT = ["rim", "jump", "three"]
PTS = {"rim": 2.0, "jump": 2.0, "three": 3.0}
NB = 200


# ------------------------------------------------------------------ counts
def sim_counts(raw: pd.DataFrame) -> dict:
    out = {}
    for s, o in (("h", "home"), ("a", "away")):
        d = {
            "pts": raw[f"{o}_pts"].to_numpy(float),
            "A_rim": raw[f"{o}_fga2_rim"].to_numpy(float), "A_jump": raw[f"{o}_fga2_jump"].to_numpy(float),
            "A_three": raw[f"{o}_fga3"].to_numpy(float),
            "M_rim": raw[f"{o}_fgm2_rim"].to_numpy(float), "M_jump": raw[f"{o}_fgm2_jump"].to_numpy(float),
            "M_three": raw[f"{o}_fgm3"].to_numpy(float),
            "fta": raw[f"{o}_fta"].to_numpy(float), "ftm": raw[f"{o}_ftm"].to_numpy(float),
            "oreb": raw[f"{o}_oreb"].to_numpy(float), "tov": raw[f"{o}_tov"].to_numpy(float),
        }
        out[s] = d
    return out


def actual_counts(season: int, gids: np.ndarray | None, finals: pd.DataFrame | None) -> tuple[pd.Index, dict]:
    tb = R.load_actual_team_box(season)
    if gids is not None:
        tb = tb[tb["game_id"].isin(gids)]
    pr = pd.read_parquet(ROOT / "results/shared_shooting/preds_v1.parquet",
                         columns=["game_id", "season", "off_team_id", "class_key", "y"])
    pr = pr[pr["season"] == season]
    two = pr[pr["class_key"].isin(["rim", "jump2"])]
    agg = two.assign(rim=(two["class_key"] == "rim").astype(float),
                     rim_m=((two["class_key"] == "rim") & (two["y"] == 1)).astype(float),
                     m=(two["y"] == 1).astype(float)) \
        .groupby(["game_id", "off_team_id"])[["rim", "rim_m", "m"]].agg(["sum", "count"])
    sh = pd.DataFrame({"rim_a": agg[("rim", "sum")], "n2": agg[("rim", "count")],
                       "rim_m": agg[("rim_m", "sum")], "m2": agg[("m", "sum")]})
    sh["share_a"] = sh["rim_a"] / sh["n2"].clip(lower=1)
    sh["share_m"] = np.where(sh["m2"] > 0, sh["rim_m"] / sh["m2"].clip(lower=1), sh["share_a"])
    sh = sh.reset_index()
    tb = tb.merge(sh[["game_id", "off_team_id", "share_a", "share_m"]],
                  left_on=["game_id", "team_id"], right_on=["game_id", "off_team_id"], how="inner")
    h = tb[tb["team_id"] == tb["home_team_id"]].set_index("game_id")
    a = tb[tb["team_id"] == tb["away_team_id"]].set_index("game_id")
    common = h.index.intersection(a.index)
    if finals is not None:
        common = common.intersection(finals.index)
    h, a = h.loc[common], a.loc[common]
    out = {}
    for s, t, ptscol in (("h", h, "home_score"), ("a", a, "away_score")):
        fg2a = (t["fga"] - t["tpa"]).to_numpy(float)
        fg2m = (t["fgm"] - t["tpm"]).to_numpy(float)
        pts = (finals.loc[common, ptscol] if finals is not None else t[ptscol]).to_numpy(float)
        out[s] = {
            "pts": pts,
            "A_rim": fg2a * t["share_a"].to_numpy(float), "A_jump": fg2a * (1 - t["share_a"].to_numpy(float)),
            "A_three": t["tpa"].to_numpy(float),
            "M_rim": fg2m * t["share_m"].to_numpy(float), "M_jump": fg2m * (1 - t["share_m"].to_numpy(float)),
            "M_three": t["tpm"].to_numpy(float),
            "fta": t["fta"].to_numpy(float), "ftm": t["ftm"].to_numpy(float),
            "oreb": t["oreb"].to_numpy(float), "tov": t["tov"].to_numpy(float),
        }
    return common, out


def add_n(c: dict) -> np.ndarray:
    for s in ("h", "a"):
        d = c[s]
        d["fga"] = d["A_rim"] + d["A_jump"] + d["A_three"]
        d["Ni"] = d["fga"] - d["oreb"] + d["tov"] + 0.44 * d["fta"]
    return 0.5 * (c["h"]["Ni"] + c["a"]["Ni"])


# ------------------------------------------------------------------ base constants and channels
def base_from_means(mean: dict) -> dict:
    """Ratio-of-means base constants from mean counts (arrays per game)."""
    b = {}
    for k in KT:
        b[f"p0_{k}"] = mean[f"M_{k}"] / np.maximum(mean[f"A_{k}"], 1e-9)
    b["f0"] = mean["ftm"] / np.maximum(mean["fta"], 1e-9)
    fga0 = mean["A_rim"] + mean["A_jump"] + mean["A_three"]
    b["v0"] = sum(PTS[k] * mean[f"A_{k}"] * b[f"p0_{k}"] for k in KT) / np.maximum(fga0, 1e-9)
    a20 = mean["A_rim"] + mean["A_jump"]
    b["v2"] = 2.0 * (mean["A_rim"] * b["p0_rim"] + mean["A_jump"] * b["p0_jump"]) / np.maximum(a20, 1e-9)
    Ni0 = np.maximum(mean["Ni"], 1e-9)
    b["o0"] = mean["oreb"] / Ni0
    b["t0"] = mean["tov"] / Ni0
    b["a0"] = mean["fta"] / Ni0
    b["e0"] = b["v0"] * (1 + b["o0"] - b["t0"] - 0.44 * b["a0"]) + b["f0"] * b["a0"]
    return b


def channels(d: dict, N: np.ndarray, b: dict) -> np.ndarray:
    """(n, 11) channel matrix; columns CH; sums to pts exactly (rem closes data error)."""
    Ni = d["Ni"]
    c = np.zeros((len(N), len(CH)))
    c[:, 0] = b["e0"] * N
    c[:, 1] = b["e0"] * (Ni - N)
    c[:, 2] = b["v0"] * (d["oreb"] - b["o0"] * Ni)
    c[:, 3] = -b["v0"] * (d["tov"] - b["t0"] * Ni)
    c[:, 4] = (b["f0"] - 0.44 * b["v0"]) * (d["fta"] - b["a0"] * Ni)
    a2 = d["A_rim"] + d["A_jump"]
    c[:, 5] = 3.0 * d["A_three"] * b["p0_three"] + b["v2"] * a2 - b["v0"] * d["fga"]
    c[:, 6] = 2.0 * (d["A_rim"] * b["p0_rim"] + d["A_jump"] * b["p0_jump"]) - b["v2"] * a2
    for j, k in enumerate(KT):
        c[:, 7 + j] = PTS[k] * (d[f"M_{k}"] - d[f"A_{k}"] * b[f"p0_{k}"])
    c[:, 10] = d["ftm"] - d["fta"] * b["f0"]
    c[:, 11] = d["pts"] - c[:, :11].sum(1)
    return c


# ------------------------------------------------------------------ moments
def pooled_within(X: np.ndarray, gid: np.ndarray) -> np.ndarray:
    """mean over rows of x x^T after removing each game's mean (equal rows per game)."""
    df = pd.DataFrame(X)
    Xc = X - df.groupby(gid).transform("mean").to_numpy()
    return Xc.T @ Xc / len(Xc)


def wcov(X: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    if w is None:
        w = np.ones(len(X))
    w = w / w.sum()
    mu = w @ X
    Xc = X - mu
    return (Xc * w[:, None]).T @ Xc


GROUPS = {
    "pace x pace": [("pace", "pace")],
    "pace x makes": [("pace", b) for b in ("rim", "jump", "three", "ft")],
    "pace x OREB": [("pace", "oreb")],
    "pace x TOV": [("pace", "tov")],
    "pace x FTA": [("pace", "fta")],
    "pace x shot mix": [("pace", "mix3"), ("pace", "mixRJ")],
    "pace x imbalance": [("pace", "imbal")],
    "two-point block": [(a, b) for a in ("rim", "jump", "mixRJ") for b in ("rim", "jump", "mixRJ")],
    "three make": [("three", "three")],
    "FT make": [("ft", "ft")],
    "FTA (whistle)": [("fta", "fta")],
    "OREB": [("oreb", "oreb")],
    "TOV": [("tov", "tov")],
    "3PA share": [("mix3", "mix3")],
    "imbalance": [("imbal", "imbal")],
    "makes cross-type (three, FT with each other and with twos)": [
        (a, b) for a in ("three", "ft") for b in ("rim", "jump", "mixRJ", "three", "ft") if a != b],
    "OREB x makes / shot mix": [("oreb", b) for b in ("rim", "jump", "three", "ft", "mixRJ", "mix3")],
    "TOV x non-pace": [("tov", b) for b in ("oreb", "fta", "mix3", "mixRJ", "rim", "jump", "three", "ft")],
    "FTA x non-pace": [("fta", b) for b in ("oreb", "mix3", "mixRJ", "rim", "jump", "three", "ft")],
    "imbalance x non-pace": [("imbal", b) for b in ("oreb", "tov", "fta", "mix3", "mixRJ", "rim", "jump", "three", "ft")],
}


def grouped(Ca: np.ndarray, Cs: np.ndarray, Da: float, Ds: float, nch: int) -> dict:
    """Corr units (between-team block) and total-variance pts^2 by GROUPS; 'other cross
    terms' is every remaining ordered pair, so the groups close to the residual gap."""
    names = CH[:nch]
    ix = {c: i for i, c in enumerate(names)}
    H, A = slice(0, nch), slice(nch, 2 * nch)
    Ba, Bs = Ca[H, A], Cs[H, A]
    Ta = Ca[H, H] + Ca[A, A] + Ca[H, A] + Ca[A, H]
    Ts = Cs[H, H] + Cs[A, A] + Cs[H, A] + Cs[A, H]
    used = set()
    out = {}
    def take(pairs):
        v = np.zeros(4)
        for p, q in pairs:
            for x, y in {(p, q), (q, p)}:
                if (x, y) in used:
                    continue
                used.add((x, y))
                i, j = ix[x], ix[y]
                v += [Ba[i, j], Bs[i, j], Ta[i, j], Ts[i, j]]
        return v
    for g, pairs in GROUPS.items():
        v = take(pairs)
        out[g] = {"act": v[0] / Da, "sim": v[1] / Ds, "gap": v[0] / Da - v[1] / Ds,
                  "tot_act": v[2], "tot_sim": v[3], "tot_gap": v[2] - v[3]}
    rest = [(x, y) for x in names for y in names if (x, y) not in used]
    v = take(rest)
    out["other cross terms"] = {"act": v[0] / Da, "sim": v[1] / Ds, "gap": v[0] / Da - v[1] / Ds,
                                "tot_act": v[2], "tot_sim": v[3], "tot_gap": v[2] - v[3]}
    return out


def group_tables(Ca: np.ndarray, Cs: np.ndarray, Da: float, Ds: float, nch: int) -> dict:
    """Ca, Cs: (2nch x 2nch) covariance of [h channels, a channels]. Corr units of the
    between-team block, and pts^2 of the total-variance decomposition, by group."""
    H = slice(0, nch)
    A = slice(nch, 2 * nch)
    Ba, Bs = Ca[H, A], Cs[H, A]                    # Cov(h_k, a_l)
    # total: C_k = h_k + a_k
    Ta = Ca[H, H] + Ca[A, A] + Ca[H, A] + Ca[A, H]
    Ts = Cs[H, H] + Cs[A, A] + Cs[H, A] + Cs[A, H]
    out = {"corr_units": {}, "total_pts2": {}, "cov_pts2": {}}
    names = CH[:nch]
    pace = names.index("pace")
    for k, nm in enumerate(names):
        out["corr_units"][f"{nm} x {nm}"] = {"act": Ba[k, k] / Da, "sim": Bs[k, k] / Ds,
                                             "gap": Ba[k, k] / Da - Bs[k, k] / Ds}
        out["cov_pts2"][f"{nm} x {nm}"] = {"act": Ba[k, k], "sim": Bs[k, k], "gap": Ba[k, k] - Bs[k, k]}
        out["total_pts2"][f"{nm} x {nm}"] = {"act": Ta[k, k], "sim": Ts[k, k], "gap": Ta[k, k] - Ts[k, k]}
    # pace x efficiency: pace with every non-pace channel, both directions
    mask = np.zeros((nch, nch), bool)
    mask[pace, :] = True
    mask[:, pace] = True
    mask[pace, pace] = False
    for lab, Mx in (("pace x efficiency", mask),):
        out["corr_units"][lab] = {"act": Ba[Mx].sum() / Da, "sim": Bs[Mx].sum() / Ds,
                                  "gap": Ba[Mx].sum() / Da - Bs[Mx].sum() / Ds}
        out["cov_pts2"][lab] = {"act": Ba[Mx].sum(), "sim": Bs[Mx].sum(), "gap": Ba[Mx].sum() - Bs[Mx].sum()}
        out["total_pts2"][lab] = {"act": Ta[Mx].sum(), "sim": Ts[Mx].sum(), "gap": Ta[Mx].sum() - Ts[Mx].sum()}
    # remaining cross terms (k != l, neither pace)
    cross = ~np.eye(nch, dtype=bool) & ~mask
    out["corr_units"]["other cross terms"] = {"act": Ba[cross].sum() / Da, "sim": Bs[cross].sum() / Ds,
                                              "gap": Ba[cross].sum() / Da - Bs[cross].sum() / Ds}
    out["cov_pts2"]["other cross terms"] = {"act": Ba[cross].sum(), "sim": Bs[cross].sum(),
                                            "gap": Ba[cross].sum() - Bs[cross].sum()}
    out["total_pts2"]["other cross terms"] = {"act": Ta[cross].sum(), "sim": Ts[cross].sum(),
                                              "gap": Ta[cross].sum() - Ts[cross].sum()}
    # the five largest cross pairs (symmetric sum) by |corr gap|
    pairs = []
    for k in range(nch):
        for l in range(k + 1, nch):
            if k == pace or l == pace:
                continue
            ga = (Ba[k, l] + Ba[l, k]) / Da - (Bs[k, l] + Bs[l, k]) / Ds
            gt = 2 * (Ta[k, l] - Ts[k, l])
            pairs.append((names[k] + " x " + names[l], ga, gt))
    pairs.sort(key=lambda t: -abs(t[1]))
    out["top_cross"] = [{"pair": p, "corr_gap": g, "total_gap_pts2": t} for p, g, t in pairs[:8]]
    pairs_pe = []
    for l in range(nch):
        if l == pace:
            continue
        ga = (Ba[pace, l] + Ba[l, pace]) / Da - (Bs[pace, l] + Bs[l, pace]) / Ds
        gt = 2 * (Ta[pace, l] - Ts[pace, l])
        pairs_pe.append({"pair": "pace x " + names[l], "corr_gap": ga, "total_gap_pts2": gt,
                         "act_cov": Ba[pace, l] + Ba[l, pace], "sim_cov": Bs[pace, l] + Bs[l, pace]})
    out["pace_x_parts"] = pairs_pe
    out["sum_check"] = {"corr_resid_gap": Ba.sum() / Da - Bs.sum() / Ds, "total_gap": Ta.sum() - Ts.sum(),
                        "var_T_act": Ta.sum(), "var_T_sim": Ts.sum(),
                        "cov_ha_act": Ba.sum(), "cov_ha_sim": Bs.sum()}
    return out


# ------------------------------------------------------------------ sim vs actual on one run
def run_read(run: Path, season: int, segments: bool = True, boot: int = NB) -> dict:
    g = pd.read_parquet(run / "games.parquet")
    summary, raw = G.build_grading_frame(g, season)
    fin = summary.set_index("game_id")
    gids_all = fin.index.to_numpy()
    common, ac = actual_counts(season, gids_all, fin)
    raw = raw[raw["game_id"].isin(common)].sort_values(["game_id", "seed"]).reset_index(drop=True)
    sc = sim_counts(raw)
    Ns = add_n(sc)
    gid = raw["game_id"].to_numpy()
    # per-game sim means of every count, and base constants
    keys = ["pts", "A_rim", "A_jump", "A_three", "M_rim", "M_jump", "M_three", "fta", "ftm", "oreb", "tov",
            "fga", "Ni"]
    base, Xs, mean_ch = {}, [], {}
    gorder = pd.Index(pd.unique(gid))
    for s in ("h", "a"):
        dfm = pd.DataFrame({k: sc[s][k] for k in keys}).groupby(gid).mean().loc[gorder]
        base[s] = base_from_means({k: dfm[k].to_numpy() for k in keys})
        rowpos = gorder.get_indexer(gid)
        brow = {k: v[rowpos] for k, v in base[s].items()}
        Xs.append(channels(sc[s], Ns, brow))
    Xs = np.hstack(Xs)                                       # (rows, 22)
    Ms = pd.DataFrame(Xs).groupby(gid).mean().loc[gorder].to_numpy()    # per-game mean channels
    Cs = pooled_within(Xs, gid)
    # actual channels in game order
    ac2 = {s: {k: pd.Series(v, index=common).loc[gorder].to_numpy() for k, v in ac[s].items()} for s in ac}
    Na = add_n(ac2)
    Xa = np.hstack([channels(ac2[s], Na, base[s]) for s in ("h", "a")])
    Ra = Xa - Ms                                            # actual residual channels
    nch = len(CH)
    # denominators (gate corr chain): sim pooled variance of pts, actual variance of pts
    mh, ma = Ms[:, :nch].sum(1), Ms[:, nch:].sum(1)
    xh, xa = Xa[:, :nch].sum(1), Xa[:, nch:].sum(1)
    var_sh = np.var(mh) + Cs[:nch, :nch].sum()
    var_sa = np.var(ma) + Cs[nch:, nch:].sum()
    Ds = np.sqrt(var_sh * var_sa)
    Da = np.sqrt(np.var(xh) * np.var(xa))
    rh, ra = xh - mh, xa - ma
    Ca = wcov(Ra)
    cov_mm = np.cov(mh, ma, bias=True)[0, 1]
    chain = {
        "corr_act": (cov_mm + np.cov(mh, ra, bias=True)[0, 1] + np.cov(rh, ma, bias=True)[0, 1]
                     + np.cov(rh, ra, bias=True)[0, 1]) / Da,
        "corr_sim": (cov_mm + Cs[:nch, nch:].sum()) / Ds,
        "between_gap": cov_mm / Da - cov_mm / Ds,
        "cross_m_r": (np.cov(mh, ra, bias=True)[0, 1] + np.cov(rh, ma, bias=True)[0, 1]) / Da,
        "resid_gap": Ca[:nch, nch:].sum() / Da - Cs[:nch, nch:].sum() / Ds,
        "gate_corr_sim": float(np.corrcoef(raw["home_pts"], raw["away_pts"])[0, 1]),
        "gate_corr_act": float(np.corrcoef(xh, xa)[0, 1]),
        "n_games": int(len(gorder)), "n_graded": int(len(fin)),
        "D_act": Da, "D_sim": Ds,
        "rem_abs_mean_act": float(np.abs(Xa[:, nch - 1]).mean() + np.abs(Xa[:, 2 * nch - 1]).mean()) / 2,
        "rem_abs_max_sim": float(np.abs(Xs[:, [nch - 1, 2 * nch - 1]]).max()),
    }
    chain["gap"] = chain["corr_act"] - chain["corr_sim"]
    tabs = group_tables(Ca, Cs, Da, Ds, nch)
    out = {"run": str(run), "season": season, "chain": chain, "tables": tabs,
           "var_total": {"act": float(np.var(rh + ra)), "sim": float(Cs.sum()),
                         "ratio_sqrt": float(np.sqrt(Cs.sum() / np.var(rh + ra)))}}
    # bootstrap SE (actual side only; the sim side is 200 seeds) of every group's actual value
    rng = np.random.default_rng(20261001)
    bs = {lab: [] for lab in tabs["corr_units"]}
    bst = {lab: [] for lab in tabs["total_pts2"]}
    for _ in range(boot):
        w = rng.poisson(1.0, len(Ra)).astype(float)
        Cb = wcov(Ra, w)
        tb_ = group_tables(Cb, Cs, Da, Ds, nch)
        for lab in bs:
            bs[lab].append(tb_["corr_units"][lab]["act"])
        for lab in bst:
            bst[lab].append(tb_["total_pts2"][lab]["act"])
    out["boot_se"] = {"corr_units": {k: float(np.std(v)) for k, v in bs.items()},
                      "total_pts2": {k: float(np.std(v)) for k, v in bst.items()}}
    out["grouped"] = grouped(Ca, Cs, Da, Ds, nch)
    gb = {k: [] for k in out["grouped"]}
    gbt = {k: [] for k in out["grouped"]}
    rng2 = np.random.default_rng(20261002)
    for _ in range(boot):
        w = rng2.poisson(1.0, len(Ra)).astype(float)
        gg = grouped(wcov(Ra, w), Cs, Da, Ds, nch)
        for k in gb:
            gb[k].append(gg[k]["act"])
            gbt[k].append(gg[k]["tot_act"])
    out["grouped_se"] = {k: {"corr": float(np.std(gb[k])) if gb[k] else None,
                             "tot": float(np.std(gbt[k])) if gbt[k] else None} for k in gb}
    if segments:
        out["segments"] = segment_reads(gorder, Ra, Xs, gid, Ms, Xa, fin, season, nch)
    out["Ca"] = Ca.tolist()
    out["Cs"] = Cs.tolist()
    # keep arrays for callers
    out["_arrays"] = (gorder, Ra, Xs, gid, Ms, Xa, Cs, Ca)
    return out


def segment_reads(gorder, Ra, Xs, gid, Ms, Xa, fin, season, nch, boot=100):
    from cbb_sim.features import conference as CF
    conf = CF.build_conference_flags([season]).set_index("game_id")["is_conf_game"]
    neutral = fin.loc[gorder, "neutral"].astype(bool).to_numpy() if "neutral" in fin.columns else \
        fin.loc[gorder, "neutral_site"].astype(bool).to_numpy()
    pm = Ms[:, :nch].sum(1) - Ms[:, nch:].sum(1)
    tier = pd.qcut(np.abs(pm), 3, labels=["close", "mid", "wide"]).astype(str)
    cf = conf.reindex(gorder).fillna(False).astype(bool).to_numpy()
    month = pd.to_datetime(fin.loc[gorder, "game_date"]).dt.month.to_numpy() if "game_date" in fin.columns \
        else fin.loc[gorder, "month"].to_numpy()
    segs = {"conference": cf, "non-conference": ~cf, "home/away site": ~neutral, "neutral": neutral}
    for t in ("close", "mid", "wide"):
        segs[f"pred |margin| {t}"] = tier == t
    pos = pd.Index(gorder).get_indexer(gid)
    rng = np.random.default_rng(7)
    res = {}
    for lab, m in segs.items():
        rows = m[pos]
        Cs_ = pooled_within(Xs[rows], gid[rows])
        R_ = Ra[m]
        Ca_ = wcov(R_)
        xh, xa = Xa[m, :nch].sum(1), Xa[m, nch:].sum(1)
        mh, ma = Ms[m, :nch].sum(1), Ms[m, nch:].sum(1)
        Da = np.sqrt(np.var(xh) * np.var(xa))
        Ds = np.sqrt((np.var(mh) + Cs_[:nch, :nch].sum()) * (np.var(ma) + Cs_[nch:, nch:].sum()))
        t = group_tables(Ca_, Cs_, Da, Ds, nch)
        sim_corr = float(np.corrcoef(Xs[rows, :nch].sum(1), Xs[rows, nch:].sum(1))[0, 1])
        act_corr = float(np.corrcoef(xh, xa)[0, 1])
        bsv = {k: [] for k in t["corr_units"]}
        for _ in range(boot):
            w = rng.poisson(1.0, int(m.sum())).astype(float)
            tb_ = group_tables(wcov(R_, w), Cs_, Da, Ds, nch)
            for k in bsv:
                bsv[k].append(tb_["corr_units"][k]["act"])
        gq = grouped(Ca_, Cs_, Da, Ds, nch)
        gqb = {k: [] for k in gq}
        for _ in range(boot):
            w = rng.poisson(1.0, int(m.sum())).astype(float)
            g2 = grouped(wcov(R_, w), Cs_, Da, Ds, nch)
            for k in gqb:
                gqb[k].append(g2[k]["act"])
        res[lab] = {"grouped": {k: v["gap"] for k, v in gq.items()},
                    "grouped_se": {k: float(np.std(v)) for k, v in gqb.items()},
                    "grouped_tot": {k: v["tot_gap"] for k, v in gq.items()},
                    "chain_cross_m_r": float((np.cov(mh, xa - ma, bias=True)[0, 1]
                                              + np.cov(xh - mh, ma, bias=True)[0, 1]) / Da),
                    "n": int(m.sum()), "corr_act": act_corr, "corr_sim": sim_corr,
                    "total_sd_ratio_sqrt": float(np.sqrt(Cs_.sum() / np.var(R_[:, :nch].sum(1) + R_[:, nch:].sum(1)))),
                    "corr_units": {k: v["gap"] for k, v in t["corr_units"].items()},
                    "corr_se": {k: float(np.std(v)) for k, v in bsv.items()},
                    "total_pts2": {k: v["gap"] for k, v in t["total_pts2"].items()}}
    return res


# ------------------------------------------------------------------ per-season, real data only
def season_descriptive(season: int, boot: int = NB) -> dict:
    fin = R.load_actual_games(season).set_index("game_id")
    common, ac = actual_counts(season, fin.index.to_numpy(), fin)
    Na = add_n(ac)
    keys = ["pts", "A_rim", "A_jump", "A_three", "M_rim", "M_jump", "M_three", "fta", "ftm", "oreb", "tov",
            "fga", "Ni"]
    mean = {k: np.full(len(common), 0.5 * (ac["h"][k].mean() + ac["a"][k].mean())) for k in keys}
    b = base_from_means(mean)
    X = np.hstack([channels(ac[s], Na, b) for s in ("h", "a")])
    nch = len(CH)
    # FE residualisation on stacked team-games: off team, def team, home / away / neutral
    from scipy.sparse import csr_matrix, hstack as shs
    from scipy.sparse.linalg import lsqr
    fr = fin.loc[common]
    ht, at = fr["home_team_id"].to_numpy(), fr["away_team_id"].to_numpy()
    neu = fr["neutral_site"].astype(bool).to_numpy() if "neutral_site" in fr.columns else fr["neutral"].astype(bool).to_numpy()
    teams = pd.Index(np.unique(np.r_[ht, at]))
    n = len(common)
    offt = np.r_[teams.get_indexer(ht), teams.get_indexer(at)]
    deft = np.r_[teams.get_indexer(at), teams.get_indexer(ht)]
    site = np.r_[np.where(neu, 0, 1), np.where(neu, 0, 2)]
    rows = np.arange(2 * n)
    T = len(teams)
    Dm = shs([csr_matrix((np.ones(2 * n), (rows, offt)), shape=(2 * n, T)),
              csr_matrix((np.ones(2 * n), (rows, deft)), shape=(2 * n, T)),
              csr_matrix((np.ones(2 * n), (rows, site)), shape=(2 * n, 3))]).tocsr()
    Y = np.r_[X[:, :nch], X[:, nch:]]
    Rr = np.zeros_like(Y)
    for k in range(nch):
        y = Y[:, k] - Y[:, k].mean()
        beta = lsqr(Dm, y, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
        Rr[:, k] = y - Dm @ beta
    Rg = np.hstack([Rr[:n], Rr[n:]])
    C = wcov(Rg)
    Da = np.sqrt(np.var(X[:, :nch].sum(1)) * np.var(X[:, nch:].sum(1)))
    zero = np.zeros_like(C)
    t = group_tables(C, zero, Da, 1.0, nch)
    rng = np.random.default_rng(11)
    bsv = {k: [] for k in t["cov_pts2"]}
    for _ in range(boot):
        w = rng.poisson(1.0, n).astype(float)
        tb_ = group_tables(wcov(Rg, w), zero, Da, 1.0, nch)
        for k in bsv:
            bsv[k].append(tb_["cov_pts2"][k]["act"])
    gq = grouped(C, zero, 1.0, 1.0, nch)
    gqb = {k: [] for k in gq}
    for _ in range(boot):
        w = rng.poisson(1.0, n).astype(float)
        g2 = grouped(wcov(Rg, w), zero, 1.0, 1.0, nch)
        for k in gqb:
            gqb[k].append(g2[k]["act"])
    return {"grouped_cov_pts2": {k: v["act"] for k, v in gq.items()},
            "grouped_cov_se": {k: float(np.std(v)) for k, v in gqb.items()},
            "season": season, "n_games": n,
            "cov_pts2": {k: v["act"] for k, v in t["cov_pts2"].items()},
            "cov_se": {k: float(np.std(v)) for k, v in bsv.items()},
            "corr_units": {k: v["act"] for k, v in t["corr_units"].items()},
            "total_cov_ha": float(C[:nch, nch:].sum()), "D_act": Da,
            "resid_corr_pts": float(C[:nch, nch:].sum() / np.sqrt(C[:nch, :nch].sum() * C[nch:, nch:].sum())),
            "pace_x_parts": t["pace_x_parts"]}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["results/engine_v0/v3full_COMB9GCTKD_s200_o0"])
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seasons-desc", nargs="*", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--no-segments", action="store_true")
    ap.add_argument("--boot", type=int, default=NB)
    ap.add_argument("--out", default="results/g5_channels/channels_v1.json")
    a = ap.parse_args()
    out = {"runs": {}, "seasons": {}}
    for r in a.runs:
        print("run", r, flush=True)
        out["runs"][Path(r).name] = _clean(run_read(ROOT / r, a.season, segments=not a.no_segments,
                                                       boot=a.boot))
    for s in a.seasons_desc:
        print("season", s, flush=True)
        out["seasons"][str(s)] = _clean(season_descriptive(s))
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
