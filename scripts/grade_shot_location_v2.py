"""grade_shot_location_v2.py -- shot-location round 2: ONE blind grader (docs/models/shot_location/experiments.md
section 3; PRE-REGISTERED at 5f6d6b0). Block V is scored by `--block V`; block P by `--block P` (added to this file and
committed before block P is run). Arm codes (block P) stay hidden until the tables are written.

Block V (diagnostic, no arm): reads the inputs written by `diag_shot_selection_dispersion_v1.py` and writes
  results/shot_location_v2/grade_V.json  and  results/shot_location_v2/tables_V.md

Conventions (spec left open, disclosed in the evidence doc):
  * unit = team-game (game, offence); band part = (game, offence, class) with >= 5 located attempts in the class.
  * share residual r = (count - sum p) / n; implied SD = sqrt(sum p(1-p)) / n  (multinomial marginal, per indicator).
  * over-dispersion ratio = SD(r) / sqrt(mean(implied var of share)), unweighted over team-games; the z SD
    (SD of (count - sum p)/sqrt(sum p(1-p))) is reported beside it.
  * floor = max(parametric-bootstrap SD of the ratio under the model (200 replicates, outcomes redrawn from the
    model's own probabilities), game-resampled bootstrap SE of the realised ratio). Excess = ratio - mean(null ratio).
    BEYOND = excess > floor.
  * latent: first-order logit shift u ~ N(0, s2) applied to every chance of a game (binary indicator vs rest);
    d mean(p) / du = sum p(1-p) / n, so  s2 = (mean r_dm^2 - mean implied share var) / mean((sum p(1-p)/n)^2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

R2 = ROOT / "data/processed/models/shot_location/round2"
VIN = R2 / "V"
RES = ROOT / "results/shot_location_v2"
B_REP = 200
SEED = 20261027
CLS5 = ["TOV", "rim", "jump2", "3PA", "FT"]
BAND5 = ["short2", "long2", "corner3", "wing3", "top3"]
BAND_CLASS = {"short2": "jump2", "long2": "jump2", "corner3": "three", "wing3": "three", "top3": "three"}
MIN_N = 5


def log(m, t0):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def md(rows, cols=None, fmt=None):
    if not rows:
        return "(empty)\n"
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = "nan" if not np.isfinite(v) else (f"{v:.4g}" if abs(v) < 1e4 else f"{v:.0f}")
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ core statistics
def draw(P, rng):
    u = rng.random(len(P))
    return (u[:, None] > np.cumsum(P, axis=1)).sum(axis=1).clip(0, P.shape[1] - 1)


def agg(g, G, y, P):
    """Per-group counts, sum p, sum p(1-p), n for indicator columns. P is (n_rows, K) marginals (columns need not
    sum to 1: FT is the sum of two classes); y is the realised column index."""
    K = P.shape[1]
    n = np.bincount(g, minlength=G).astype(float)
    cnt = np.stack([np.bincount(g, weights=(y == k).astype(float), minlength=G) for k in range(K)], axis=1)
    mu = np.stack([np.bincount(g, weights=P[:, k], minlength=G) for k in range(K)], axis=1)
    var = np.stack([np.bincount(g, weights=P[:, k] * (1 - P[:, k]), minlength=G) for k in range(K)], axis=1)
    return n, cnt, mu, var


def resid_share(n, cnt, mu, var):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (cnt - mu) / n[:, None]
        vs = var / n[:, None] ** 2
        z = (cnt - mu) / np.sqrt(var)
        slope = var / n[:, None]
    return r, vs, z, slope


def stats(r, vs, z, slope, ok):
    """ratio, z SD, mean bias, latent variance for each column over the rows in ok (bool mask, per column)."""
    out = []
    for k in range(r.shape[1]):
        m = ok[:, k] if ok.ndim == 2 else ok
        rr, vv, zz, ss = r[m, k], vs[m, k], z[m, k], slope[m, k]
        rd = rr - rr.mean()
        s2 = (np.mean(rd ** 2) - np.mean(vv)) / np.mean(ss ** 2)
        out.append((np.std(rr) / np.sqrt(np.mean(vv)), float(np.nanstd(zz)), float(rr.mean()), s2, int(m.sum()),
                    np.sqrt(np.mean(vv)), np.std(rr)))
    return out


def game_boot(g_of_tg, r, vs, z, slope, ok, rng, B=B_REP):
    """Game-resampled bootstrap of the realised ratio and latent s2 (resample games, take all their team-games)."""
    games = pd.factorize(g_of_tg)[0]
    ng = games.max() + 1
    order = np.argsort(games, kind="stable")
    starts = np.searchsorted(games[order], np.arange(ng + 1))
    res = []
    for _ in range(B):
        pick = rng.integers(0, ng, ng)
        idx = np.concatenate([order[starts[p]:starts[p + 1]] for p in pick])
        res.append([(s[0], s[3]) for s in stats(r[idx], vs[idx], z[idx], slope[idx], ok[idx])])
    a = np.array(res)                      # B x K x 2
    return a[:, :, 0].std(axis=0, ddof=1), a[:, :, 1].std(axis=0, ddof=1)


# ------------------------------------------------------------------ data assembly
def load_V(fold):
    """Returns dict with team-game frame and row-level arrays for the class part (F2 only) and the band part."""
    d = {}
    mg = pd.read_parquet(VIN / "margins.parquet")
    if fold == "F2":
        ch = pd.read_parquet(VIN / "chance_probs_F2_2025.parquet")
        key = ch[["game_id", "offense_team_id"]]
        codes, uniq = pd.factorize(pd.MultiIndex.from_frame(key))
        tg = pd.DataFrame(list(uniq), columns=["game_id", "team_id"])
        d["chance_g"] = codes.astype(np.int64)
        pc = ch[[f"p_{c}" for c in ["TOV", "FGA_rim", "FGA_jump2", "FGA_3", "FT_trip_shooting", "FT_trip_bonus"]]
                ].to_numpy(dtype="float64")
        d["chance_P"] = np.column_stack([pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3], pc[:, 4] + pc[:, 5]])
        d["chance_y"] = np.minimum(ch["y"].to_numpy(), 4).astype(np.int64)
        d["chance_pop"] = ch["population"].to_numpy()
        cc = pd.read_parquet(VIN / "class_context_F2_2025.parquet").rename(columns={"offense_team_id": "team_id"})
        tg = tg.merge(cc, on=["game_id", "team_id"], how="left")
    else:
        bp = pd.read_parquet(VIN / f"band_probs_{fold}.parquet")
        key = bp[["game_id", "off_team_id"]].rename(columns={"off_team_id": "team_id"})
        codes, uniq = pd.factorize(pd.MultiIndex.from_frame(key))
        tg = pd.DataFrame(list(uniq), columns=["game_id", "team_id"])
    bp = pd.read_parquet(VIN / f"band_probs_{fold}.parquet")
    k2 = pd.MultiIndex.from_frame(bp[["game_id", "off_team_id"]].rename(columns={"off_team_id": "team_id"}))
    pos = pd.Index(pd.MultiIndex.from_frame(tg[["game_id", "team_id"]])).get_indexer(k2)
    keep = pos >= 0
    bp = bp[keep].copy()
    bp["g"] = pos[keep]
    d["band"] = bp
    tg = tg.merge(mg[["game_id", "team_id", "margin", "is_home", "neutral"]], on=["game_id", "team_id"], how="left")
    d["tg"] = tg
    return d


def band_arrays(d, G):
    """Per class (jump2 / three): (n, cnt, mu, var) over groups, using the attempt-level B2 probabilities."""
    out = {}
    bp = d["band"]
    for cls, shot, bands in (("jump2", "FGA_jump2", ("short2", "long2")),
                             ("three", "FGA_3", ("corner3", "wing3", "top3"))):
        b = bp[bp["shot_class"] == shot]
        P = b[[f"p_{x}" for x in bands]].to_numpy(dtype="float64")
        y = b["band"].map({x: i for i, x in enumerate(bands)}).to_numpy(dtype=np.int64)
        out[cls] = dict(g=b["g"].to_numpy(dtype=np.int64), P=P, y=y, bands=bands)
    return out


# ------------------------------------------------------------------ one replicate (observed or null)
def replicate(d, G, B, rng, null):
    """Returns dict of arrays on the common team-game index: class r/vs/z/slope (G,5) with n; band arrays."""
    res = {}
    if "chance_g" in d:
        y = draw(d["chance_P5full"], rng) if null else d["chance_y_full"]
        y5 = np.minimum(y, 4)
        n, cnt, mu, var = agg(d["chance_g"], G, y5, d["chance_P"])
        res["class"] = (n, *resid_share(n, cnt, mu, var))
    for cls, a in B.items():
        y = draw(a["P"], rng) if null else a["y"]
        n, cnt, mu, var = agg(a["g"], G, y, a["P"])
        res[cls] = (n, *resid_share(n, cnt, mu, var))
    return res


def assemble_cols(res):
    """Stack the nine residual-share columns (class: TOV rim jump2 3PA FT; band: short2 corner3 wing3 top3) + ok mask."""
    cols, oks, names = [], [], []
    if "class" in res:
        n, r, vs, z, sl = res["class"]
        for k, nm in enumerate(CLS5):
            cols.append((r[:, k], vs[:, k], z[:, k], sl[:, k])); oks.append(n >= 1); names.append(nm)
    for cls, bands in (("jump2", ("short2", "long2")), ("three", ("corner3", "wing3", "top3"))):
        n, r, vs, z, sl = res[cls]
        for k, nm in enumerate(bands):
            cols.append((r[:, k], vs[:, k], z[:, k], sl[:, k])); oks.append(n >= MIN_N); names.append(nm)
    R = np.column_stack([c[0] for c in cols]); V = np.column_stack([c[1] for c in cols])
    Z = np.column_stack([c[2] for c in cols]); S = np.column_stack([c[3] for c in cols])
    OK = np.column_stack(oks) & np.isfinite(R) & np.isfinite(V) & np.isfinite(Z)
    return names, R, V, Z, S, OK


def corr_mat(R, OK, cols):
    m = OK[:, cols].all(axis=1)
    C = np.corrcoef(R[m][:, cols].T)
    return C, int(m.sum())


def pc1_share(C):
    w = np.linalg.eigvalsh(C)
    return float(w[-1] / len(C))


def run_fold(fold, t0):
    d = load_V(fold)
    tg = d["tg"]
    G = len(tg)
    B = band_arrays(d, G)
    if fold == "F2":
        Pf = d["chance_P"]
        # null draws need the full 6-class probabilities for the class part: rebuild from the parquet columns
        ch = pd.read_parquet(VIN / "chance_probs_F2_2025.parquet")
        P6 = ch[[f"p_{c}" for c in ["TOV", "FGA_rim", "FGA_jump2", "FGA_3", "FT_trip_shooting", "FT_trip_bonus"]]
                ].to_numpy(dtype="float64")
        d["chance_P5full"] = P6
        d["chance_y_full"] = ch["y"].to_numpy().astype(np.int64)
    obs = replicate(d, G, B, None, null=False)
    names, R, V, Z, S, OK = assemble_cols(obs)
    st = stats(R, V, Z, S, OK)
    rng = np.random.default_rng([SEED, 1 if fold == "F2" else 2])
    # null replicates: ratios, latent s2, correlation matrices, pc1
    ncol = len(names)
    null_ratio = np.zeros((B_REP, ncol)); null_s2 = np.zeros((B_REP, ncol)); null_z = np.zeros((B_REP, ncol))
    sel = [i for i, n in enumerate(names) if n != "long2"]
    null_C = np.zeros((B_REP, len(sel), len(sel))); null_pc1 = np.zeros(B_REP)
    for b in range(B_REP):
        rep = replicate(d, G, B, rng, null=True)
        _, R_, V_, Z_, S_, OK_ = assemble_cols(rep)
        s_ = stats(R_, V_, Z_, S_, OK_)
        null_ratio[b] = [x[0] for x in s_]; null_s2[b] = [x[3] for x in s_]; null_z[b] = [x[1] for x in s_]
        C_, _ = corr_mat(R_, OK_, sel)
        null_C[b] = C_; null_pc1[b] = pc1_share(C_)
        if b % 50 == 0:
            log(f"{fold} null replicate {b}", t0)
    bse_ratio, bse_s2 = game_boot(tg["game_id"].to_numpy(), R, V, Z, S, OK, np.random.default_rng([SEED, 7]))
    rows = []
    for i, nm in enumerate(names):
        ratio, zsd, bias, s2, n_tg, sd_imp, sd_real = st[i]
        nr_mean, nr_sd = null_ratio[:, i].mean(), null_ratio[:, i].std(ddof=1)
        floor = max(nr_sd, bse_ratio[i])
        excess = ratio - nr_mean
        sigma = float(np.sqrt(max(s2, 0.0)))
        rows.append(dict(unit="class" if nm in CLS5 else "band", name=nm, team_games=n_tg,
                         mean_resid_pp=100 * bias, sd_implied_pp=100 * sd_imp, sd_realised_pp=100 * sd_real,
                         ratio=ratio, null_mean=nr_mean, null_sd=nr_sd, boot_se=bse_ratio[i], floor=floor,
                         excess_over_floor=excess / floor, beyond_floor=bool(excess > floor), z_sd=zsd,
                         null_z_sd=null_z[:, i].mean(),
                         latent_s2=s2, latent_sigma=sigma, sigma_boot_se_s2=bse_s2[i],
                         null_s2_sd=null_s2[:, i].std(ddof=1), latent_beyond=bool(s2 > max(null_s2[:, i].std(ddof=1),
                                                                                       bse_s2[i]))))
    out = dict(fold=fold, rows=rows, names=names)
    # ---- correlation matrix of residuals + PC1
    C, nC = corr_mat(R, OK, sel)
    nm_sel = [names[i] for i in sel]
    out["corr"] = dict(names=nm_sel, n=nC, obs=C.tolist(), null_mean=null_C.mean(axis=0).tolist(),
                       null_sd=null_C.std(axis=0, ddof=1).tolist(), pc1_obs=pc1_share(C),
                       pc1_null_mean=float(null_pc1.mean()), pc1_null_sd=float(null_pc1.std(ddof=1)))
    # ---- covariates
    mg = tg
    cov = {}
    have_class = "class" in obs
    nmg = {nm: i for i, nm in enumerate(names)}
    site = np.where(mg["neutral"].fillna(0).to_numpy() > 0.5, np.nan, mg["is_home"].astype(float).to_numpy())
    Nch = obs["class"][0] if have_class else None
    covs = {"margin": mg["margin"].to_numpy(float), "site_home": site}
    if have_class:
        covs["n_chances"] = Nch
        covs["opp_def_profile"] = None
    bp = d["band"]
    defcol = {"short2": "def_short2_c", "corner3": "def_corner3_c", "wing3": "def_wing3_c", "top3": "def_top3_c"}
    offcol = {"short2": "off_short2_c", "corner3": "off_corner3_c", "wing3": "off_wing3_c", "top3": "off_top3_c"}
    bprof = {}
    for col in (*defcol.values(), *offcol.values()):
        bprof[col] = np.full(G, np.nan)
        s_ = bp.groupby("g")[col].first()
        bprof[col][s_.index.to_numpy()] = s_.to_numpy()
    cl_def = {"TOV": "opp_def_tov_c", "rim": "opp_def_rim_c", "3PA": "opp_def_3pa_c"}
    rows_c = []
    for i, nm in enumerate(names):
        if nm == "long2":
            continue
        m = OK[:, i]
        for cn, cv in covs.items():
            if cv is None:
                continue
            ok = m & np.isfinite(cv)
            rows_c.append(dict(resid=nm, covariate=cn, n=int(ok.sum()), corr=float(np.corrcoef(R[ok, i], cv[ok])[0, 1])))
        prof = None
        if nm in defcol:
            prof = bprof[defcol[nm]]; pn = defcol[nm]
        elif nm in cl_def and have_class:
            prof = mg[cl_def[nm]].to_numpy(float); pn = cl_def[nm]
        if prof is not None:
            ok = m & np.isfinite(prof)
            rows_c.append(dict(resid=nm, covariate="opp_defence_profile:" + pn, n=int(ok.sum()),
                               corr=float(np.corrcoef(R[ok, i], prof[ok])[0, 1])))
    # null for the covariate correlations (sampling noise): SD from the replicates' own corr is not stored; use
    # the Fisher approximation 1/sqrt(n) as the stated floor for every covariate line.
    for r_ in rows_c:
        r_["floor_1_over_sqrt_n"] = 1 / np.sqrt(r_["n"])
        r_["beyond_floor"] = bool(abs(r_["corr"]) > 2 * r_["floor_1_over_sqrt_n"])
    out["covariates"] = rows_c
    # ---- V-B by offence quintile and V-C
    segs = []
    prof_for = {"3PA": "off_3pa_c", "rim": "off_rim_c", "TOV": "off_tov_c", "jump2": "off_3pa_c"}
    for nm, col in prof_for.items():
        if not have_class or nm not in nmg:
            continue
        i = nmg[nm]
        v = mg[col].to_numpy(float)
        q = pd.qcut(pd.Series(v), 5, labels=False, duplicates="drop").to_numpy()
        for qq in range(5):
            m = OK[:, i] & (q == qq)
            s_ = stats(R[m][:, [i]], V[m][:, [i]], Z[m][:, [i]], S[m][:, [i]], np.ones((m.sum(), 1), bool))[0]
            segs.append(dict(name=nm, offence_quintile=qq + 1, profile=col, team_games=int(m.sum()), ratio=s_[0],
                             z_sd=s_[1], latent_s2=s_[3], mean_resid_pp=100 * s_[2]))
    for nm, col in offcol.items():
        i = nmg[nm]
        v = bprof[col]
        qs = pd.qcut(pd.Series(v[np.isfinite(v)]), 5, labels=False, duplicates="drop")
        q = np.full(G, -1); q[np.isfinite(v)] = qs.to_numpy()
        for qq in range(5):
            m = OK[:, i] & (q == qq)
            s_ = stats(R[m][:, [i]], V[m][:, [i]], Z[m][:, [i]], S[m][:, [i]], np.ones((m.sum(), 1), bool))[0]
            segs.append(dict(name=nm, offence_quintile=qq + 1, profile=col, team_games=int(m.sum()), ratio=s_[0],
                             z_sd=s_[1], latent_s2=s_[3], mean_resid_pp=100 * s_[2]))
    out["by_offence_quintile"] = segs
    if have_class:
        # by population (first vs cont chances are one team-game unit; reported only as the class mix)
        vc = []
        i3 = nmg["3PA"]
        rel = {}
        for nm in names:
            j = nmg[nm]
            ok = OK[:, j]
            rel[nm] = max(0.0, 1 - float(np.mean(V[ok, j]) / np.var(R[ok, j])))
        for a in ("3PA", "rim"):
            for b in ("corner3", "wing3", "top3", "short2", "long2"):
                if b == "long2":
                    jb = nmg["short2"]; sign = -1.0
                else:
                    jb = nmg[b]; sign = 1.0
                ja = nmg[a]
                ok = OK[:, ja] & OK[:, jb]
                r_ = sign * float(np.corrcoef(R[ok, ja], R[ok, jb])[0, 1])
                # null
                nc = []
                # disattenuated
                rel_ab = np.sqrt(rel[a] * rel["short2" if b == "long2" else b])
                # game-bootstrap SE of the raw and disattenuated corr
                games = pd.factorize(tg["game_id"].to_numpy())[0]
                order = np.argsort(games, kind="stable"); ng = games.max() + 1
                starts = np.searchsorted(games[order], np.arange(ng + 1))
                rg = np.random.default_rng([SEED, 11]); bs = []
                for _ in range(B_REP):
                    pick = rg.integers(0, ng, ng)
                    idx = np.concatenate([order[starts[p]:starts[p + 1]] for p in pick])
                    okb = ok[idx]
                    xa, xb = R[idx][okb, ja], R[idx][okb, jb]
                    ra = max(0.0, 1 - float(np.mean(V[idx][okb, ja]) / np.var(xa)))
                    rb = max(0.0, 1 - float(np.mean(V[idx][okb, jb]) / np.var(xb)))
                    bs.append((np.corrcoef(xa, xb)[0, 1], np.corrcoef(xa, xb)[0, 1] / max(np.sqrt(ra * rb), 1e-9)))
                bs = np.array(bs)
                ia, ib = sel.index(ja), sel.index(jb)
                vc.append(dict(class_resid=a, band_resid=b, n=int(ok.sum()), corr=r_,
                               null_mean=sign * float(null_C[:, ia, ib].mean()),
                               null_sd=float(null_C[:, ia, ib].std(ddof=1)),
                               boot_se=float(bs[:, 0].std(ddof=1)),
                               reliability_class=rel[a], reliability_band=rel["short2" if b == "long2" else b],
                               disattenuated_corr=r_ / max(rel_ab, 1e-9) if rel_ab > 0 else float("nan"),
                               disattenuated_boot_se=float(bs[:, 1].std(ddof=1))))
        out["V_C"] = vc
    return out


# ============================================================================ BLOCK P
FOLDS = ("F1", "F2")
TEST = {"F1": 2024, "F2": 2025}
TRAIN = {"F1": [2022, 2023], "F2": [2022, 2023, 2024]}
SHOT_BANDS = {"FGA_jump2": ("short2", "long2"), "FGA_3": ("corner3", "wing3", "top3")}
PROF = {"FGA_jump2": ("short2",), "FGA_3": ("corner3", "wing3", "top3")}
N_BOOT = 200
CALIB_MIN_FLOOR_PP = 0.10
SLOPE_LO, SLOPE_HI = 0.8, 1.2
POWERED_PC = 2000
CROSSWALK = ROOT / "data/processed/player_crosswalk_v2.parquet"
BOX = ROOT / "data/raw/hoopr/player_box"


def ll_bin(y, p):
    return -np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-12, 1.0))


def ll_multi(y, P):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))


def boot_game_mean(games, d, n_rep=N_BOOT, seed=12345):
    g = pd.DataFrame({"g": games, "d": d}).groupby("g")["d"].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    reps = np.empty(n_rep)
    for r in range(n_rep):
        k = rng.integers(0, len(s), len(s))
        reps[r] = s[k].sum() / n[k].sum()
    return float(reps.std(ddof=1))


def boot_unit_mean(d, n_rep=N_BOOT, seed=54321):
    rng = np.random.default_rng(seed)
    n = len(d)
    return float(np.std([d[rng.integers(0, n, n)].mean() for _ in range(n_rep)], ddof=1))


def quint_line(driver, pred, act, w=None):
    """Round-1 `rate_resp` convention: attempt-weighted quintiles of the driver's rank; slope ratio = predicted span /
    realised span; monotone steps against the realised direction; Decision-8 low-span-exempt verdict beside the raw
    [0.8, 1.2] test."""
    from cbb_sim.models import fg_make as FG
    v = FG._rank(np.asarray(driver, dtype="float64"))
    edges = np.quantile(v, np.linspace(0, 1, 6))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    q = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, 4)
    w = np.ones(len(v)) if w is None else np.asarray(w, dtype="float64")
    pa = np.array([np.average(pred[q == b], weights=w[q == b]) for b in range(5)])
    aa = np.array([np.average(act[q == b], weights=w[q == b]) for b in range(5)])
    sp, sa = float(pa[-1] - pa[0]), float(aa[-1] - aa[0])
    ref = np.sign(sa) if sa else np.sign(sp)
    steps = int(np.sum(np.sign(np.diff(pa)) == ref))
    slope = round(sp / sa, 4) if sa else None
    d8 = FG.decision8_verdict({"x": {"span_actual": sa, "pred_monotone_steps": steps, "slope_ratio": slope}},
                              reading="low_span_exempt")["by_driver"]["x"]
    return {"pred": [round(float(x), 5) for x in pa], "actual": [round(float(x), 5) for x in aa],
            "span_pred": round(sp, 5), "span_actual": round(sa, 5), "slope_ratio": slope, "steps": steps,
            "raw_slope_in_0.8_1.2": bool(slope is not None and SLOPE_LO <= slope <= SLOPE_HI),
            "d8_pass": bool(d8["pass"])}


def load_ctx_P():
    import grade_shot_location_v1 as G1
    ctx = G1.load_context()
    c2 = pd.read_parquet(R2 / "design_props_cols_v1.parquet")
    for k in ("game_id", "shooter_id", "shot_class"):
        assert (c2[k].to_numpy() == ctx[k].to_numpy()).all(), k
    for c in c2.columns:
        if c.startswith(("sh_", "smk_")):
            ctx[c] = c2[c].to_numpy()
    return G1, ctx


# ------------------------------------------------------------------ P-A
def p_a(G1, ctx, rng):
    from cbb_sim.models import fg_make as FG
    out = {}
    for fold in FOLDS:
        runs = {"R": ("R", 0), "R_seed": ("R", 1), "P0": ("P0", 0), "P1": ("P1", 0)}
        code = G1.codes_for(list(runs), rng)
        per_raw = {}
        rows_ref = None
        for lab, (a, s) in runs.items():
            d = R2 / "make" / f"{a}_s{s}_{fold}"
            pr = pd.read_parquet(d / f"preds_{fold}_s{s}.parquet")
            rw = pd.read_parquet(d / f"rows_{fold}_s{s}.parquet")
            if rows_ref is None:
                rows_ref = rw
            assert (rw["fg_row"].to_numpy() == rows_ref["fg_row"].to_numpy()).all()
            per_raw[code[lab]] = pr
        res = {"code_of": code, "by_class": {}}
        for cls in SHOT_BANDS:
            m = (rows_ref["shot_class"] == cls).to_numpy()
            fr = rows_ref.loc[m, "fg_row"].to_numpy()
            te = ctx.iloc[fr].reset_index(drop=True)
            te["y"] = te["made"].astype("int64").to_numpy()
            y = te["y"].to_numpy()
            cm = {}
            for c, pr in per_raw.items():
                pc = pr.loc[m]
                assert (pc["y"].to_numpy() == y).all()
                p = pc["p"].to_numpy(dtype="float64")
                sc = FG.score(te, np.column_stack([1 - p, p]))
                lines = {}
                for b in SHOT_BANDS[cls]:
                    mb = (te["band"] == b).to_numpy()
                    lines[b] = quint_line(te.loc[mb, f"smk_{b}"].to_numpy(), p[mb], y[mb].astype(float))
                cm[c] = {"p": p, "llr": ll_bin(y, p), "log_loss": sc["log_loss"],
                         "calib_worst_gap_pp": sc["calib_worst_gap_pp"],
                         "calib_level_pp": sc["calibration"]["MAKE"]["level_shift_pp"],
                         "calib_shape_pp": sc["calibration"]["MAKE"]["max_abs_gap_pp_after_level_shift"],
                         "lines": lines, "pred_rate": float(p.mean()), "actual_rate": float(y.mean())}
            res["by_class"][cls] = {"te": te, "per": cm, "n": int(len(te))}
        out[fold] = res
    return out


def seg_rows(te, d, g, powered):
    segs = {}
    for seg in ("site", "month", "tier", "locq"):
        for v in sorted(te[seg].astype(str).unique()):
            mm = (te[seg].astype(str) == v).to_numpy()
            segs[f"{seg}={v}"] = {"n": int(mm.sum()), "gain": float(-d[mm].mean()),
                                 "boot_se": boot_game_mean(g[mm], d[mm]), "powered": bool(mm.sum() >= powered)}
    return segs


def decide_p_a(F, powered=5000):
    co = F["code_of"]
    out = {}
    for cls, C in F["by_class"].items():
        per, te = C["per"], C["te"]
        R, RS, P0, P1 = (per[co[k]] for k in ("R", "R_seed", "P0", "P1"))
        seed_diff = abs(R["log_loss"] - RS["log_loss"])
        gfl = max(abs(R["calib_worst_gap_pp"] - RS["calib_worst_gap_pp"]), CALIB_MIN_FLOOR_PP)
        g = te["game_id"].to_numpy()
        arms = {}
        for lab, A, ref in (("P1_vs_P0", P1, P0), ("P0_vs_R", P0, R)):
            d = A["llr"] - ref["llr"]
            se = boot_game_mean(g, d)
            floor = max(seed_diff, se)
            gain = float(-d.mean())
            held = A["calib_worst_gap_pp"] <= ref["calib_worst_gap_pp"] + gfl
            arms[lab] = {"gain": gain, "boot_se": se, "seed_diff": seed_diff, "floor": floor,
                         "gain_in_floors": gain / floor, "calib_gate_held": bool(held), "beats_floor": gain > floor,
                         "segments": seg_rows(te, d, g, powered)}
        out[cls] = {"seed_diff": seed_diff, "calib_floor_pp": gfl, "arms": arms,
                    "verdict": "SIGNAL" if (arms["P1_vs_P0"]["beats_floor"] and arms["P1_vs_P0"]["calib_gate_held"])
                    else "NO SIGNAL"}
    return out


# ------------------------------------------------------------------ P-B
def p_b(G1, ctx, rng):
    out = {}
    for fold in FOLDS:
        code = G1.codes_for(["B2", "BP"], rng)
        pr = {}
        for arm in ("B2", "BP"):
            pr[code[arm]] = pd.read_parquet(R2 / "readB" / f"{arm}_{fold}" / "preds.parquet")
        ref = next(iter(pr.values()))
        for p_ in pr.values():
            assert (p_["fg_row"].to_numpy() == ref["fg_row"].to_numpy()).all()
        res = {"code_of": code, "by_class": {}}
        for cls, bands in SHOT_BANDS.items():
            m = ((ref["shot_class"] == cls) & ref["located"]).to_numpy()
            fr = ref.loc[m, "fg_row"].to_numpy()
            te = ctx.iloc[fr].reset_index(drop=True)
            y = te["band"].map({b: i for i, b in enumerate(bands)}).to_numpy(dtype=np.int64)
            cm = {}
            for c, p_ in pr.items():
                arm = [k for k, v in code.items() if v == c][0]
                P = p_.loc[m, [f"{arm}__p_{b}" for b in bands]].to_numpy(dtype="float64")
                lines = {}
                for i, b in enumerate(bands):
                    if b not in PROF[cls]:
                        continue
                    lines[b] = quint_line(te[f"sh_{b}_c"].to_numpy(), P[:, i], (y == i).astype(float))
                cm[c] = {"P": P, "llr": ll_multi(y, P), "log_loss": float(ll_multi(y, P).mean()), "lines": lines}
            res["by_class"][cls] = {"te": te, "per": cm, "n": int(len(te)), "y": y}
        out[fold] = res
    return out


def decide_p_b(F, powered=5000):
    co = F["code_of"]
    out = {}
    for cls, C in F["by_class"].items():
        per, te = C["per"], C["te"]
        B2, BP = per[co["B2"]], per[co["BP"]]
        g = te["game_id"].to_numpy()
        d = BP["llr"] - B2["llr"]
        se = boot_game_mean(g, d)
        gain = float(-d.mean())
        out[cls] = {"gain": gain, "floor": se, "gain_in_floors": gain / se, "beats_floor": gain > se,
                    "lines_passing": {b: bool(BP["lines"][b]["raw_slope_in_0.8_1.2"]) for b in BP["lines"]},
                    "segments": seg_rows(te, d, g, powered)}
    return out


# ------------------------------------------------------------------ P-C
def p_c(ctx):
    """Player-game 3PM and FGM: implied over the player's ACTUAL attempts. (i) served R; (ii) P1 marginalised over BP."""
    cw = pd.read_parquet(CROSSWALK)[["season", "cbbd_player_id", "espn_athlete_id"]].dropna()
    cw = cw.rename(columns={"cbbd_player_id": "shooter_id"})
    res = {}
    for fold in FOLDS:
        season = TEST[fold]
        R_ = pd.read_parquet(R2 / "make" / f"R_s0_{fold}" / f"cf_{fold}_s0.parquet")
        P_ = pd.read_parquet(R2 / "make" / f"P1_s0_{fold}" / f"cf_{fold}_s0.parquet")
        assert (R_["fg_row"].to_numpy() == P_["fg_row"].to_numpy()).all()
        BP = pd.read_parquet(R2 / "readB" / f"BP_{fold}" / "preds.parquet").set_index("fg_row")
        BP = BP.loc[R_["fg_row"].to_numpy()]
        q_i = R_["p"].to_numpy(dtype="float64")
        q_ii = np.full(len(R_), np.nan)
        for cls, bands in SHOT_BANDS.items():
            m = (R_["shot_class"] == cls).to_numpy()
            acc = np.zeros(int(m.sum()))
            for b in bands:
                acc += BP.loc[m, f"BP__p_{b}"].to_numpy(dtype="float64") * P_.loc[m, f"p_{b}"].to_numpy(dtype="float64")
            q_ii[m] = acc
        assert np.isfinite(q_i).all() and np.isfinite(q_ii).all()
        att = pd.DataFrame({"fg_row": R_["fg_row"].to_numpy(), "cls": R_["shot_class"].to_numpy(),
                            "q_i": q_i, "q_ii": q_ii, "located": R_["located"].to_numpy()})
        att["game_id"] = ctx["game_id"].to_numpy()[att["fg_row"]]
        att["shooter_id"] = ctx["shooter_id"].to_numpy()[att["fg_row"]]
        att["made"] = ctx["made"].to_numpy()[att["fg_row"]].astype(float)
        rim_tr = ctx[(ctx["shot_class"] == "FGA_rim") & ctx["season"].isin(TRAIN[fold])]["made"].astype(float).mean()
        rim = ctx[(ctx["shot_class"] == "FGA_rim") & (ctx["season"] == season)]
        ra = pd.DataFrame({"fg_row": rim.index.to_numpy(), "cls": "FGA_rim", "q_i": rim_tr, "q_ii": rim_tr,
                           "located": True, "game_id": rim["game_id"].to_numpy(),
                           "shooter_id": rim["shooter_id"].to_numpy(), "made": rim["made"].astype(float).to_numpy()})
        A = pd.concat([att, ra], ignore_index=True)
        for q in ("q_i", "q_ii"):
            A["v_" + q] = A[q] * (1 - A[q])
        A["is3"] = (A["cls"] == "FGA_3").astype(float)
        A["unloc"] = (~A["located"].astype(bool) & (A["cls"] != "FGA_rim")).astype(float)
        for q in ("q_i", "q_ii", "v_q_i", "v_q_ii", "made"):
            A[q + "_3"] = A[q] * A["is3"]
        A["one"] = 1.0
        agg = A.groupby(["game_id", "shooter_id"]).agg(
            n_all=("one", "sum"), n3=("is3", "sum"), n_unloc=("unloc", "sum"),
            fgm_i=("q_i", "sum"), fgm_ii=("q_ii", "sum"), fv_i=("v_q_i", "sum"), fv_ii=("v_q_ii", "sum"),
            fgm_a=("made", "sum"), tp_i=("q_i_3", "sum"), tp_ii=("q_ii_3", "sum"), tv_i=("v_q_i_3", "sum"),
            tv_ii=("v_q_ii_3", "sum"), tp_a=("made_3", "sum")).reset_index()
        agg["season"] = season
        agg = agg.merge(cw, on=["season", "shooter_id"], how="left")
        box = pd.read_parquet(BOX / f"player_box_{season}.parquet",
                              columns=["game_id", "athlete_id", "field_goals_made", "field_goals_attempted",
                                       "three_point_field_goals_made", "three_point_field_goals_attempted"])
        box["game_id"] = box["game_id"].astype("int64")
        box = box.drop_duplicates(["game_id", "athlete_id"])
        agg = agg.merge(box.rename(columns={"athlete_id": "espn_athlete_id"}), on=["game_id", "espn_athlete_id"],
                        how="left")
        cnt = {"player_games_in_design": int(len(agg)),
               "no_athlete_map": int(agg["espn_athlete_id"].isna().sum()),
               "no_box_row": int(agg["field_goals_attempted"].isna().sum() - agg["espn_athlete_id"].isna().sum())}
        j = agg[agg["field_goals_attempted"].notna()].copy()
        cnt["joined"] = int(len(j))
        cnt["join_rate"] = cnt["joined"] / cnt["player_games_in_design"]
        j["match3"] = j["n3"] == j["three_point_field_goals_attempted"]
        j["matchF"] = j["n_all"] == j["field_goals_attempted"]
        cnt["3pa_count_mismatch_dropped_from_3PM"] = int(((j["n3"] >= 1) & ~j["match3"]).sum())
        cnt["fga_count_mismatch_dropped_from_FGM"] = int((~j["matchF"]).sum())
        mm = (j["n3"] >= 1) & j["match3"]
        cnt["3pm_design_equals_box_among_matched_attempts"] = float((j.loc[mm, "tp_a"] == j.loc[mm, "three_point_field_goals_made"]).mean())
        res[fold] = {"counts": cnt, "pg": j}
    return res


def pc_line(j, kind):
    if kind == "3pm":
        d = j[(j["n3"] >= 1) & j["match3"]].copy()
        qi, qii, vi, vii, n = (d[c].to_numpy() for c in ("tp_i", "tp_ii", "tv_i", "tv_ii", "n3"))
        a = d["three_point_field_goals_made"].to_numpy(dtype=float)
    else:
        d = j[j["matchF"] & (j["n_all"] >= 1)].copy()
        qi, qii, vi, vii, n = (d[c].to_numpy() for c in ("fgm_i", "fgm_ii", "fv_i", "fv_ii", "n_all"))
        a = d["field_goals_made"].to_numpy(dtype=float)
    loc_only = (d["n_unloc"] == 0).to_numpy()
    g = d["game_id"].to_numpy()
    ei, eii = np.abs(a - qi), np.abs(a - qii)
    dd = ei - eii

    def block(mask, label):
        if mask.sum() < 3:
            return None
        gain = float(dd[mask].mean())
        se_u = boot_unit_mean(dd[mask])
        se_g = boot_game_mean(g[mask], dd[mask])
        floor = max(se_u, se_g)
        return {"label": label, "n_player_games": int(mask.sum()), "underpowered": bool(mask.sum() < POWERED_PC),
                "mean_actual": float(a[mask].mean()), "mean_i": float(qi[mask].mean()), "mean_ii": float(qii[mask].mean()),
                "mae_i": float(ei[mask].mean()), "mae_ii": float(eii[mask].mean()), "gain": gain,
                "se_player_game": se_u, "se_game_block": se_g, "floor": floor, "gain_in_floors": gain / floor,
                "dispersion_ratio_i": float(np.std(a[mask] - qi[mask]) / np.sqrt(vi[mask].mean())),
                "dispersion_ratio_ii": float(np.std(a[mask] - qii[mask]) / np.sqrt(vii[mask].mean()))}

    out = {"overall": block(np.ones(len(d), bool), "all"),
           "located_only": block(loc_only, "all attempts located"), "tiers": []}
    tiers = ((("3PA 1-2", 1, 2), ("3PA 3-5", 3, 5), ("3PA 6+", 6, 99)) if kind == "3pm"
             else (("FGA 1-4", 1, 4), ("FGA 5-9", 5, 9), ("FGA 10+", 10, 99)))
    for lab, lo_, hi_ in tiers:
        out["tiers"].append(block((n >= lo_) & (n <= hi_), lab))
    delta = (qii - qi) / n
    out["profile_quintile"] = {
        "driver": "per-attempt shift (ii)-(i)",
        "arm_i": quint_line(delta, qi / n, a / n, w=n),
        "arm_ii": quint_line(delta, qii / n, a / n, w=n),
        "delta_sd": float(np.std(delta)), "delta_range": [float(delta.min()), float(delta.max())]}
    q = pd.qcut(pd.Series(delta).rank(method="first"), 5, labels=False).to_numpy()
    xs = np.array([np.average(delta[q == b], weights=n[q == b]) for b in range(5)])
    ys = np.array([np.average(a[q == b] / n[q == b] - qi[q == b] / n[q == b], weights=n[q == b]) for b in range(5)])
    out["profile_quintile"]["excess_vs_shift_slope"] = float(np.polyfit(xs, ys, 1)[0]) if np.ptp(xs) > 0 else None
    # POST-HOC DESCRIPTIVE (added after the first P-C read; NOT a decision input): player-game bootstrap of the
    # registered slope ratio of arm (ii) and of the excess-vs-shift slope
    rng_b = np.random.default_rng(99)
    sr, ex = [], []
    for _ in range(N_BOOT):
        ii = rng_b.integers(0, len(delta), len(delta))
        dl, nn, aa_, qi_, qii_ = delta[ii], n[ii], a[ii], qi[ii], qii[ii]
        qq = pd.qcut(pd.Series(dl).rank(method="first"), 5, labels=False).to_numpy()
        pa2 = np.array([np.average(qii_[qq == b] / nn[qq == b], weights=nn[qq == b]) for b in range(5)])
        ac2 = np.array([np.average(aa_[qq == b] / nn[qq == b], weights=nn[qq == b]) for b in range(5)])
        xs2 = np.array([np.average(dl[qq == b], weights=nn[qq == b]) for b in range(5)])
        ys2 = np.array([np.average(aa_[qq == b] / nn[qq == b] - qi_[qq == b] / nn[qq == b], weights=nn[qq == b])
                        for b in range(5)])
        sr.append((pa2[-1] - pa2[0]) / (ac2[-1] - ac2[0]) if ac2[-1] != ac2[0] else np.nan)
        ex.append(np.polyfit(xs2, ys2, 1)[0])
    out["profile_quintile"]["post_hoc_boot"] = {
        "slope_ratio_ii_p2.5_p50_p97.5": [float(x) for x in np.nanpercentile(sr, [2.5, 50, 97.5])],
        "excess_slope_p2.5_p50_p97.5": [float(x) for x in np.nanpercentile(ex, [2.5, 50, 97.5])]}
    out["profile_quintile"]["quintile_mean_shift"] = [float(x) for x in xs]
    out["profile_quintile"]["quintile_mean_actual_minus_i"] = [float(y_) for y_ in ys]
    return out


# ------------------------------------------------------------------ decision (literal 3.2)
def decide_overall(fold, pa, pb, pc3, rel):
    pcm = pc3["overall"]
    c1 = pcm["gain"] > pcm["floor"]
    tiers = [t for t in pc3["tiers"] if t]
    c2 = all(t["gain"] > -t["floor"] for t in tiers)
    resp = pc3["profile_quintile"]["arm_ii"]
    c3 = bool(resp["raw_slope_in_0.8_1.2"])
    pa_beats = any(v["verdict"] == "SIGNAL" for v in pa.values())
    pb_beats = any(v["beats_floor"] for v in pb.values())
    c4 = pa_beats or pb_beats
    return {"fold": fold, "c1_3PM_MAE_improves_beyond_floor": bool(c1), "c1_gain": pcm["gain"], "c1_floor": pcm["floor"],
            "c2_no_tier_worse_beyond_floor": bool(c2),
            "c3_profile_quintile_slope_in_0.8_1.2": c3, "c3_slope_ratio": resp["slope_ratio"],
            "c3_reliability_sqrt": rel, "c4_P_A_or_P_B_beats_floor": bool(c4), "P_A_beats": bool(pa_beats),
            "P_B_beats": bool(pb_beats),
            "verdict": "CANDIDATE (offline, props pathway)" if (c1 and c2 and c3 and c4) else "NOT CANDIDATE"}


def md_table_pc(o):
    rows = [o[k] for k in ("overall", "located_only") if o[k]] + [t for t in o["tiers"] if t]
    cols = ["label", "n_player_games", "underpowered", "mean_actual", "mean_i", "mean_ii", "mae_i", "mae_ii", "gain",
            "se_player_game", "se_game_block", "floor", "gain_in_floors", "dispersion_ratio_i", "dispersion_ratio_ii"]
    return md(rows, cols)


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in ("p", "llr", "te", "P")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_block_P(t0):
    rng = np.random.default_rng(20261027)
    build = json.loads((R2 / "build_report_v1.json").read_text(encoding="utf-8"))
    if build.get("refused"):
        raise SystemExit("leak test REFUSED the shooter features; block P does not run")
    G1, ctx = load_ctx_P()
    log("context loaded", t0)
    PA = p_a(G1, ctx, rng)
    log("P-A scored", t0)
    PB = p_b(G1, ctx, rng)
    log("P-B scored", t0)
    PCx = p_c(ctx)
    log("P-C inputs built", t0)
    # BLIND tables first (arm codes only); the reveal happens in the tables written after this one
    blind = []
    for fold in FOLDS:
        for nm, F in (("P-A", PA[fold]), ("P-B", PB[fold])):
            for cls, C in F["by_class"].items():
                row = dict(fold=fold, read=nm, cls=cls)
                row.update({f"LL_{c}": v["log_loss"] for c, v in sorted(C["per"].items())})
                blind.append(row)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "tables_P_blind.md").write_text("### blind log losses (arm codes hidden)\n" + md(blind), encoding="utf-8")
    log("blind tables written", t0)
    out = {"folds": {}}
    tx = []
    relrows = [r for r in build["reliability"]["rows"] if r["stratum"] == "powered(>=100)"]
    tx.append("### Reliability (split-half, Spearman-Brown); shooter-seasons >= 100 located attempts in the class "
              "(the 40-99 stratum is UNDERPOWERED and lives in build_report_v1.json)\n" +
              md(relrows, ["quantity", "class", "band", "shooter_seasons", "split_half_r", "spearman_brown"]))
    rel_c = {r["band"]: r["spearman_brown"] for r in relrows if r["quantity"] == "make_rate_contrast"}
    for fold in FOLDS:
        lab = "" if fold == "F2" else " (F1: located subset is a BIASED SUBSET)"
        dA = decide_p_a(PA[fold])
        dB = decide_p_b(PB[fold])
        ro = []
        for cls, v in dA.items():
            for a_, w in v["arms"].items():
                ro.append(dict(cls=cls, comparison=a_, gain=w["gain"], seed_diff=w["seed_diff"], boot_se=w["boot_se"],
                               floor=w["floor"], gain_in_floors=w["gain_in_floors"], gate_held=w["calib_gate_held"],
                               beats_floor=w["beats_floor"]))
        tx.append(f"### P-A {fold}: log loss gain (positive = first arm better); floor = max(|LL R - LL R_seed|, game-block bootstrap SE){lab}\n" + md(ro))
        co = PA[fold]["code_of"]
        rl, rs = [], []
        for cls, C in PA[fold]["by_class"].items():
            for arm in ("R", "R_seed", "P0", "P1"):
                m = C["per"][co[arm]]
                rl.append(dict(cls=cls, arm=arm, log_loss=m["log_loss"], worst_decile_gap_pp=m["calib_worst_gap_pp"],
                               level_pp=m["calib_level_pp"], shape_pp=m["calib_shape_pp"]))
            for arm in ("R", "P0", "P1"):
                for b, ln in C["per"][co[arm]]["lines"].items():
                    rs.append(dict(cls=cls, arm=arm, band=b, slope_ratio=ln["slope_ratio"], steps=ln["steps"],
                                   span_pred=ln["span_pred"], span_actual=ln["span_actual"],
                                   in_0_8_1_2=ln["raw_slope_in_0.8_1.2"], d8_pass=ln["d8_pass"],
                                   pred=ln["pred"], actual=ln["actual"]))
        tx.append(f"### P-A {fold}: decile battery as served (worst gap / level / shape, pp)\n" + md(rl))
        tx.append(f"### P-A {fold}: make rate by quintile of the shooter's within-band skill smk_<band> (responsiveness)\n" + md(rs))
        sg = []
        for cls, v in dA.items():
            for k, w in v["arms"]["P1_vs_P0"]["segments"].items():
                sg.append(dict(cls=cls, segment=k, n=w["n"], gain=w["gain"], boot_se=w["boot_se"],
                               in_floors=w["gain"] / w["boot_se"], underpowered=not w["powered"]))
        tx.append(f"### P-A {fold}: P1 vs P0 by segment (cells < 5,000 attempts UNDERPOWERED)\n" + md(sg))
        coB = PB[fold]["code_of"]
        pbr, rb = [], []
        for cls, v in dB.items():
            C = PB[fold]["by_class"][cls]
            pbr.append(dict(cls=cls, LL_B2=C["per"][coB["B2"]]["log_loss"], LL_BP=C["per"][coB["BP"]]["log_loss"],
                            gain=v["gain"], floor=v["floor"], gain_in_floors=v["gain_in_floors"],
                            beats_floor=v["beats_floor"]))
            for arm in ("B2", "BP"):
                for b, ln in C["per"][coB[arm]]["lines"].items():
                    rb.append(dict(cls=cls, arm=arm, band=b, slope_ratio=ln["slope_ratio"], steps=ln["steps"],
                                   in_0_8_1_2=ln["raw_slope_in_0.8_1.2"], d8_pass=ln["d8_pass"], pred=ln["pred"],
                                   actual=ln["actual"]))
        tx.append(f"### P-B {fold}: multinomial log loss, B2 vs BP (floor = paired game-block bootstrap SE; seed floor 0){lab}\n" + md(pbr))
        tx.append(f"### P-B {fold}: realised band share by quintile of the shooter's own as-of share (wing3 WEAKER)\n" + md(rb))
        sgb = []
        for cls, v in dB.items():
            for k, w in v["segments"].items():
                sgb.append(dict(cls=cls, segment=k, n=w["n"], gain=w["gain"], boot_se=w["boot_se"],
                                in_floors=w["gain"] / w["boot_se"], underpowered=not w["powered"]))
        tx.append(f"### P-B {fold}: BP vs B2 by segment\n" + md(sgb))
        cnt = PCx[fold]["counts"]
        tx.append(f"### P-C {fold}: player-game join counts{lab}\n```\n{json.dumps(cnt, indent=1)}\n```\n")
        pc3 = pc_line(PCx[fold]["pg"], "3pm")
        pcf = pc_line(PCx[fold]["pg"], "fgm")
        tx.append(f"### P-C {fold}: player-game 3PM, (i) served R vs (ii) P1 over the BP band mix (gain = MAE(i) - MAE(ii); floor = max(player-game, game-block bootstrap SE); dispersion ratio = SD(actual - implied) / RMS(implied Bernoulli SD))\n" + md_table_pc(pc3))
        tx.append(f"### P-C {fold}: player-game FGM (rim attempts held at the pooled training-seasons rim make rate in both arms)\n" + md_table_pc(pcf))
        for nm, o in (("3PM", pc3), ("FGM", pcf)):
            pq = o["profile_quintile"]
            tx.append(f"### P-C {fold} {nm}: responsiveness by quintile of the shooter's profile shift ({pq['driver']}; delta SD {pq['delta_sd']:.4g}, range {pq['delta_range'][0]:.3g}..{pq['delta_range'][1]:.3g}); slope of (actual - (i)) on shift = {pq['excess_vs_shift_slope']}; POST-HOC bootstrap 95% (not a decision input): slope ratio (ii) {np.round(pq['post_hoc_boot']['slope_ratio_ii_p2.5_p50_p97.5'], 3).tolist()}, excess slope {np.round(pq['post_hoc_boot']['excess_slope_p2.5_p50_p97.5'], 3).tolist()}\n" +
                      md([dict(arm="(i) R", **{k: v for k, v in pq["arm_i"].items() if k != "d8_pass"}),
                          dict(arm="(ii) P1 x BP", **{k: v for k, v in pq["arm_ii"].items() if k != "d8_pass"})]) +
                      f"\nquintile mean shift {np.round(pq['quintile_mean_shift'], 5).tolist()}; actual - (i) {np.round(pq['quintile_mean_actual_minus_i'], 5).tolist()}\n")
        rel = {"sqrt_rel_make_contrast_short2_long2": float(np.sqrt(max(rel_c.get("short2 - long2", 0), 0))),
               "sqrt_rel_make_contrast_corner3_noncorner3": float(np.sqrt(max(rel_c.get("corner3 - noncorner3", 0), 0)))}
        dec = decide_overall(fold, dA, dB, pc3, rel)
        tx.append(f"### Block-P decision {fold} (literal section 3.2)\n```\n{json.dumps(dec, indent=1)}\n```\n")
        out["folds"][fold] = {"P_A": strip(dA), "P_B": strip(dB), "P_C_counts": cnt, "P_C_3pm": pc3, "P_C_fgm": pcf,
                              "decision": dec, "code_of_P_A": PA[fold]["code_of"], "code_of_P_B": PB[fold]["code_of"]}
        log(f"{fold} P graded: {dec['verdict']}", t0)
    (RES / "grade_P.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (RES / "tables_P.md").write_text("\n".join(tx), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["V", "P"], required=True)
    a = ap.parse_args()
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    if a.block == "V":
        out = {"F2": run_fold("F2", t0), "F1": run_fold("F1", t0)}
        (RES / "grade_V.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
        tx = []
        for fold in ("F2", "F1"):
            o = out[fold]
            lab = "" if fold == "F2" else " (F1: located subset is BIASED SUBSET; band part only)"
            cols = ["unit", "name", "team_games", "mean_resid_pp", "sd_implied_pp", "sd_realised_pp", "ratio",
                    "null_mean", "floor", "excess_over_floor", "beyond_floor", "z_sd"]
            tx.append(f"### V-A/V-B {fold}: over-dispersion ratio per class and band{lab}\n" + md(o["rows"], cols))
            tx.append(f"### V-B latent {fold}: logit-normal random effect by method of moments\n" +
                      md(o["rows"], ["unit", "name", "latent_s2", "latent_sigma", "sigma_boot_se_s2", "null_s2_sd",
                                     "latent_beyond"]))
            c = o["corr"]
            cm = [dict(zip(["resid"] + c["names"], [nm] + [f"{c['obs'][i][j]:+.3f} ({c['null_mean'][i][j]:+.3f})"
                                                           for j in range(len(c['names']))]))
                  for i, nm in enumerate(c["names"])]
            tx.append(f"### V-A {fold}: within-team-game residual correlation, observed (null mean), n={c['n']}; PC1 "
                      f"share {c['pc1_obs']:.3f} (null {c['pc1_null_mean']:.3f} +/- {c['pc1_null_sd']:.3f})\n" + md(cm))
            tx.append(f"### V-A {fold}: residual vs covariates\n" + md(o["covariates"]))
            tx.append(f"### V-B {fold}: by offence quintile\n" + md(o["by_offence_quintile"]))
            if "V_C" in o:
                tx.append(f"### V-C {fold}\n" + md(o["V_C"]))
        (RES / "tables_V.md").write_text("\n".join(tx), encoding="utf-8")
        log("block V graded", t0)
    else:
        run_block_P(t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
