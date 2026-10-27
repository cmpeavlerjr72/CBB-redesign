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
        raise SystemExit("block P grader not yet written (committed before block P is run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
