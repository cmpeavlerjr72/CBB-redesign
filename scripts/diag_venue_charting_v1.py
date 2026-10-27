#!/usr/bin/env python
"""diag_venue_charting_v1.py -- venue charting asymmetry diagnostic (PM ruling 17a, 2026-10-27). DESCRIPTIVE.

Question: round 1 found the defence-allowed band profile's gain large when the offence is AWAY and ~0 at home.
Hypothesis: the home crew's placement style is learned as the home team's "defence allowed" profile.

Parts (CBBD located attempts 2022-2025, non-neutral games; frame, bands and design table from
build_shot_location_bands_v1 / design_loc_cols_v1.parquet):
  1. same team, offence band shares HOME vs AWAY, per team-season; mean difference per band with a paired
     bootstrap over team-seasons; share of team-seasons with |diff| > 2 SE where SE = SD of the diff under
     a within-team-season permutation of the home/away labels over games (the placebo also gives the expected
     share beyond 2 SE under "no site effect").
  2. same for located share and distance-from-basket (median, p90) per class.
  3. venue fixed effect: multinomial logit of the band on offence + defence profile (+ offence-is-home, and the
     defence profile split by site) with and without one dummy per HOME VENUE (home team); fit on 2022-24, scored
     on 2025 (fold 2); held-out log-loss gain of the venue term (game-block bootstrap SE) and the defence
     coefficient with / without the venue term.
  4. 2025 from 2025-02-10: both sources (CBBD, hoopR) on the same attempts: the home-away difference in each source,
     cross-source correlation of the per-team differences, and the band-disagreement rate by site of the offence.

    .venv/Scripts/python.exe scripts/diag_venue_charting_v1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
import build_shot_location_bands_v1 as BL  # noqa: E402

R1 = ROOT / "data/processed/models/shot_location/round1"
DESIGN = ROOT / "data/processed/models/fg_make/design_v2_shotshooter.parquet"
AUDIT = ROOT / "results/coordinate_audit"
SEASONS = [2022, 2023, 2024, 2025]
N_PERM, N_BOOT = 40, 400
MIN_N = 40            # located attempts per class per side for a team-season to enter
CLS = {"FGA_rim": "rim", "FGA_jump2": "jump2", "FGA_3": "three"}
BANDS = {"jump2": ("short2", "long2"), "three": ("corner3", "wing3", "top3")}
REP: dict = {}


def log(m, t0):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def load() -> pd.DataFrame:
    d = pd.read_parquet(R1 / "design_loc_cols_v1.parquet")
    s = pd.read_parquet(DESIGN, columns=["neutral_site", "offense_is_home"])
    assert len(s) == len(d)
    d["neutral"] = s["neutral_site"].to_numpy().astype(bool)
    d["off_home"] = s["offense_is_home"].to_numpy().astype(bool)
    d["cls"] = d["shot_class"].map(CLS)
    d["venue"] = np.where(d["off_home"], d["off_team_id"], d["def_team_id"])
    return d[~d["neutral"]].reset_index(drop=True)


def pstack(f, perm, k):
    r = []
    for p in perm:
        v = f(p)
        r.append(np.full(k, np.nan) if v is None else np.atleast_1d(v))
    return np.array(r)


# ---------------------------------------------------------------- parts 1-2: paired home/away with a permutation null
def team_stats(att: pd.DataFrame, stat: str, bands: tuple, rng) -> pd.DataFrame:
    """att: attempts of one class for one source (cols season, team, game, home(bool), band, d, located).
    Returns one row per (season, team, stat-column) with obs home-away difference and the permutation SD."""
    out = []
    for (s, t), g in att.groupby(["season", "team"], sort=False):
        gm, gi = np.unique(g["game"].to_numpy(), return_inverse=True)
        home_g = np.zeros(len(gm), bool)
        home_g[gi] = g["home"].to_numpy()
        if home_g.all() or (~home_g).all():
            continue
        perm = np.stack([rng.permutation(home_g) for _ in range(N_PERM)])      # (P, G)
        if stat == "share":
            L = g[g["band"] != ""]
            if len(L) == 0:
                continue
            li = np.searchsorted(gm, L["game"].to_numpy())
            C = np.zeros((len(gm), len(bands)))
            for j, b in enumerate(bands):
                np.add.at(C[:, j], li, (L["band"].to_numpy() == b).astype(float))
            n = C.sum(1)

            def diff(h):
                hs, as_ = (h[:, None] * C).sum(0), ((~h)[:, None] * C).sum(0)
                nh, na = hs.sum(), as_.sum()
                if nh < MIN_N or na < MIN_N:
                    return None
                return hs / nh - as_ / na
            o = diff(home_g)
            if o is None:
                continue
            ps = pstack(diff, perm, len(bands))
            for j, b in enumerate(bands):
                out.append((s, t, b, o[j], np.nanstd(ps[:, j], ddof=1), int(C[home_g].sum()), int(C[~home_g].sum())))
        elif stat == "located":
            C = np.zeros((len(gm), 2))
            np.add.at(C[:, 0], gi, g["located"].to_numpy().astype(float))
            np.add.at(C[:, 1], gi, 1.0)

            def diff(h):
                hn, an = C[h, 1].sum(), C[~h, 1].sum()
                if hn < MIN_N or an < MIN_N:
                    return None
                return C[h, 0].sum() / hn - C[~h, 0].sum() / an
            o = diff(home_g)
            if o is None:
                continue
            ps = pstack(diff, perm, 1)
            out.append((s, t, "located_share", o, np.nanstd(ps[:, 0], ddof=1), int(C[home_g, 1].sum()), int(C[~home_g, 1].sum())))
        else:                                                                    # distance quantiles, located only
            L = g[g["located"] & g["d"].notna()]
            if len(L) == 0:
                continue
            li = np.searchsorted(gm, L["game"].to_numpy())
            d = L["d"].to_numpy()

            def diff(h, q):
                m = h[li]
                if m.sum() < MIN_N or (~m).sum() < MIN_N:
                    return None
                return np.quantile(d[m], q) - np.quantile(d[~m], q)
            for q, nm in ((0.5, "dist_median"), (0.9, "dist_p90")):
                o = diff(home_g, q)
                if o is None:
                    continue
                ps = pstack(lambda h: diff(h, q), perm, 1)
                out.append((s, t, nm, o, np.nanstd(ps[:, 0], ddof=1), int(home_g[li].sum()), int((~home_g)[li].sum())))
    return pd.DataFrame(out, columns=["season", "team", "stat", "diff", "perm_se", "n_home", "n_away"])


def summarise(T: pd.DataFrame, rng) -> list[dict]:
    rows = []
    for (stat,), g in T.groupby(["stat"], sort=False):
        z = g["diff"] / g["perm_se"]
        d = g["diff"].to_numpy()
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
        r = {"stat": stat, "team_seasons": int(len(g)), "mean_diff_home_minus_away": float(d.mean()),
             "boot_se_paired": float(boots.std(ddof=1)), "boot_z": float(d.mean() / boots.std(ddof=1)),
             "median_abs_diff": float(np.median(np.abs(d))),
             "pct_teams_abs_z_gt_2": float(100 * (np.abs(z) > 2).mean()),
             "pct_teams_z_gt_2": float(100 * (z > 2).mean()), "pct_teams_z_lt_m2": float(100 * (z < -2).mean()),
             "mean_perm_se": float(g["perm_se"].mean())}
        rows.append(r)
    return rows


def part12(D: pd.DataFrame, t0) -> None:
    rng = np.random.default_rng(20261027)
    att = D.rename(columns={"off_team_id": "team", "game_id": "game", "off_home": "home"})
    att = att[["season", "team", "game", "home", "cls", "band", "d", "located"]]
    out1, out2 = {}, {}
    for c in ("jump2", "three"):
        a = att[att["cls"] == c]
        T = team_stats(a, "share", BANDS[c], rng)
        out1[c] = summarise(T, rng)
        out1[c + "__by_season"] = {int(s): summarise(g, rng) for s, g in T.groupby("season")}
        log(f"1 {c}: {out1[c]}", t0)
    for c in ("rim", "jump2", "three"):
        a = att[att["cls"] == c]
        T = pd.concat([team_stats(a, "located", (), rng), team_stats(a, "dist", (), rng)], ignore_index=True)
        out2[c] = summarise(T, rng)
        log(f"2 {c}: {out2[c]}", t0)
    REP["part1_band_shares"] = out1
    REP["part2_located_and_distance"] = out2


# ---------------------------------------------------------------- part 3: venue fixed effect
def fit_logit(X, y, C, K):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=2500, solver="lbfgs")
    m.fit(X, y)
    return m


def part3(D: pd.DataFrame, t0) -> None:
    """2x2 on fold 2 (fit 2022-24, score 2025): {offence profile + offence-is-home, + defence profile} x
    {no venue term, + one ridge dummy per HOME VENUE}. Reported: held-out log loss per arm overall and by the
    offence's site, the venue term's gain given the profiles, the defence profile's gain with and without the venue
    term (by site of the offence), and the defence coefficients with / without the venue term."""
    import scipy.sparse as sp
    from sklearn.preprocessing import OneHotEncoder
    L = D[(D["band"] != "") & D["cls"].isin(["jump2", "three"])]
    out = {}
    SC = 30.0       # profile columns are scaled by SC so the (shared) ridge penalty barely touches them
    for c, bands in BANDS.items():
        Q = L[L["cls"] == c]
        pf = {"jump2": ["short2"], "three": ["corner3", "wing3", "top3"]}[c]
        tr, te = Q[Q["season"] <= 2024], Q[Q["season"] == 2025]
        classes = list(bands) if c == "three" else ["long2", "short2"]
        enc = OneHotEncoder(handle_unknown="ignore").fit(tr[["venue"]])
        ytr = pd.Categorical(tr["band"], categories=classes).codes
        yte = pd.Categorical(te["band"], categories=classes).codes

        def X(q, with_def, with_venue):
            parts = [q[[f"off_{b}_c" for b in pf]].to_numpy() * SC, q["off_home"].to_numpy().astype(float)[:, None] * SC]
            if with_def:
                parts.append(q[[f"def_{b}_c" for b in pf]].to_numpy() * SC)
            M = sp.csr_matrix(np.hstack(parts))
            return sp.hstack([M, enc.transform(q[["venue"]])]).tocsr() if with_venue else M
        res = {"n_train": int(len(tr)), "n_test": int(len(te)), "venues_train": int(len(enc.categories_[0])),
               "venues_test_unseen_share": float((~te["venue"].isin(enc.categories_[0])).mean())}
        ll, coefs = {}, {}
        site = te["off_home"].to_numpy()
        gi = pd.factorize(te["game_id"])[0]
        for name, wd, wv in (("off", False, False), ("off+def", True, False), ("off+venue", False, True),
                             ("off+def+venue", True, True)):
            m = fit_logit(X(tr, wd, wv), ytr, 1.0, None)
            p = m.predict_proba(X(te, wd, wv))
            ll[name] = -np.log(np.clip(p[np.arange(len(yte)), yte], 1e-12, 1))
            if wd:
                k = len(pf) + 1
                coefs[name] = (m.coef_[:, k:k + len(pf)] * SC).round(4).tolist()
            if wv:
                nf = len(pf) + 1 + (len(pf) if wd else 0)
                coefs[name + "__venue_coef_sd"] = float(m.coef_[:, nf:].std())
            log(f"3 {c} {name}: test ll {ll[name].mean():.5f}  converged_iter={m.n_iter_}", t0)

        def gain(a, b, mask=None):
            d = ll[a] - ll[b]
            mk = np.ones(len(d), bool) if mask is None else mask
            dg = np.bincount(gi[mk], weights=d[mk], minlength=gi.max() + 1)
            ng = np.bincount(gi[mk], minlength=gi.max() + 1)
            keep = ng > 0
            dg, ng = dg[keep], ng[keep]
            rng = np.random.default_rng(7)
            bs = [dg[k].sum() / ng[k].sum() for k in (rng.integers(0, len(ng), len(ng)) for _ in range(N_BOOT))]
            return {"gain_per_attempt": float(d[mk].mean()), "boot_se": float(np.std(bs, ddof=1)), "n": int(mk.sum()),
                    "deviance": float(2 * d[mk].sum())}
        res["test_logloss"] = {k: float(v.mean()) for k, v in ll.items()}
        res["gains"] = {}
        for lab, mk in (("all", None), ("offence_away", ~site), ("offence_home", site)):
            res["gains"][lab] = {
                "def_profile_given_off (off -> off+def)": gain("off", "off+def", mk),
                "def_profile_given_off_and_venue (off+venue -> off+def+venue)": gain("off+venue", "off+def+venue", mk),
                "venue_given_off (off -> off+venue)": gain("off", "off+venue", mk),
                "venue_given_off_and_def (off+def -> off+def+venue)": gain("off+def", "off+def+venue", mk)}
        res["def_coefs_" + "_".join(pf)] = coefs
        out[c] = res
    REP["part3_venue_fe"] = out


def part3b(D: pd.DataFrame, t0) -> None:
    """Crew-style vs defence-shape discriminator. A crew places EVERY shot in its arena, so a venue effect learned
    on AWAY-offence rows (shots by visitors) should carry to HOME-offence rows at the same venue (the home team's
    own shots), and vice versa. Venue ridge coefficients (offence profile + offence-is-home in the model, fit on
    2022-24) are fit separately on away-offence rows and on home-offence rows; reported: correlation across venues,
    and the 2025 log-loss gain of each fit's venue term on the OTHER site's rows (venue term learned on one site's
    rows, applied to the other's)."""
    import scipy.sparse as sp
    from sklearn.preprocessing import OneHotEncoder
    L = D[(D["band"] != "") & D["cls"].isin(["jump2", "three"])]
    out = {}
    SC = 30.0
    for c, bands in BANDS.items():
        Q = L[L["cls"] == c]
        pf = {"jump2": ["short2"], "three": ["corner3", "wing3", "top3"]}[c]
        classes = list(bands) if c == "three" else ["long2", "short2"]
        tr, te = Q[Q["season"] <= 2024], Q[Q["season"] == 2025]
        enc = OneHotEncoder(handle_unknown="ignore").fit(tr[["venue"]])

        def X(q, venue):
            M = sp.csr_matrix(np.hstack([q[[f"off_{b}_c" for b in pf]].to_numpy() * SC]))
            return sp.hstack([M, enc.transform(q[["venue"]])]).tocsr() if venue else M
        fits, res = {}, {}
        for site in (False, True):                                    # offence_is_home value of the fitting rows
            a = tr[tr["off_home"] == site]
            y = pd.Categorical(a["band"], categories=classes).codes
            fits[site] = fit_logit(X(a, True), y, 1.0, None)
        k = len(pf)
        V = {st: fits[st].coef_[:, k:] for st in fits}                # (n_class_rows, n_venues)
        res["venue_coef_corr_away_fit_vs_home_fit"] = [float(np.corrcoef(V[False][i], V[True][i])[0, 1])
                                                       for i in range(V[False].shape[0])]
        res["venue_coef_sd_away_fit"] = float(V[False].std())
        res["venue_coef_sd_home_fit"] = float(V[True].std())
        # transfer: venue term learned on site F applied to rows of site T (profile coefs from the same fit)
        for fit_site, to_site in ((False, True), (True, False)):
            b = te[te["off_home"] == to_site]
            y = pd.Categorical(b["band"], categories=classes).codes
            m = fits[fit_site]
            p1 = m.predict_proba(X(b, True))
            # profile-only comparator: same fit's profile coefs, venue coefs zeroed
            m0 = fit_logit(X(tr[tr["off_home"] == fit_site], False),
                           pd.Categorical(tr[tr["off_home"] == fit_site]["band"], categories=classes).codes, 1.0, None)
            p0 = m0.predict_proba(X(b, False))
            ll1 = -np.log(np.clip(p1[np.arange(len(y)), y], 1e-12, 1))
            ll0 = -np.log(np.clip(p0[np.arange(len(y)), y], 1e-12, 1))
            gi = pd.factorize(b["game_id"])[0]
            dg, ng = np.bincount(gi, weights=ll0 - ll1), np.bincount(gi)
            rng = np.random.default_rng(7)
            bs = [dg[kk].sum() / ng[kk].sum() for kk in (rng.integers(0, len(ng), len(ng)) for _ in range(N_BOOT))]
            res[f"transfer_venue_fit_on_{'home' if fit_site else 'away'}_offence_rows_scored_on_{'home' if to_site else 'away'}_offence_rows"] = {
                "gain_per_attempt": float((ll0 - ll1).mean()), "boot_se": float(np.std(bs, ddof=1)), "n": int(len(y))}
            log(f"3b {c} fit {fit_site}->{to_site}: gain {(ll0 - ll1).mean():.2e} se {np.std(bs, ddof=1):.2e}", t0)
        log(f"3b {c} corr {res['venue_coef_corr_away_fit_vs_home_fit']}", t0)
        out[c] = res
    REP["part3b_crew_transfer"] = out


# ---------------------------------------------------------------- part 4: two sources, 2025 from 2025-02-10
def part4(t0) -> None:
    rng = np.random.default_rng(20261028)
    m = pd.read_parquet(AUDIT / "master_2025.parquet")
    m = m[(m["game_date"] >= "2025-02-10") & m["cls"].isin(["jump2", "three"])].copy()
    ctx = pd.read_parquet(DESIGN, columns=["cbbd_game_id", "neutral_site"]).drop_duplicates("cbbd_game_id")
    m = m.merge(ctx.rename(columns={"cbbd_game_id": "gameId"}), on="gameId", how="left")
    m = m[m["neutral_site"].fillna(True).astype(bool).eq(False)]
    cls = m["cls"].map({"jump2": "FGA_jump2", "three": "FGA_3"}).to_numpy()
    bc, _, _ = BL.assign_band(cls, m["xf"].to_numpy(float), m["y"].to_numpy(float), m["located"].to_numpy(bool))
    hxf = (47.0 - np.abs(m["hx"].to_numpy(float))) * 10.0
    hyy = 250.0 + m["hy"].to_numpy(float) * 10.0
    hl = m["h_located"].to_numpy(bool) & ~m["h_sentinel"].fillna(False).to_numpy(bool)
    bh, _, _ = BL.assign_band(cls, hxf, hyy, hl)
    m["band_c"], m["band_h"] = bc, bh
    both = m[(m["band_c"] != "") & (m["band_h"] != "")].copy()
    both["home"] = both["isHomeTeam"].astype(bool)
    res = {"attempts_both_located": int(len(both)), "games": int(both["gameId"].nunique())}
    dis = []
    for c in ("jump2", "three"):
        b = both[both["cls"] == c]
        for site, g in b.groupby("home"):
            dis.append({"class": c, "offence_site": "home" if site else "away", "pairs": int(len(g)),
                        "band_differs_pct": round(100 * float((g["band_c"] != g["band_h"]).mean()), 3)})
    res["disagreement_by_offence_site"] = dis
    per_src = {}
    for src, col in (("cbbd", "band_c"), ("hoopr", "band_h")):
        per_src[src] = {}
        for c in ("jump2", "three"):
            a = both[both["cls"] == c].assign(band=lambda x: x[col])
            a = a.rename(columns={"teamId": "team", "gameId": "game"})
            a["season"] = 2025
            a["d"] = np.nan
            a["located"] = True
            T = team_stats(a[["season", "team", "game", "home", "band", "d", "located"]], "share", BANDS[c], rng)
            per_src[src][c] = T
            res.setdefault("home_minus_away_" + src, {})[c] = summarise(T, rng)
    cross = []
    for c in ("jump2", "three"):
        j = per_src["cbbd"][c].merge(per_src["hoopr"][c], on=["season", "team", "stat"], suffixes=("_c", "_h"))
        for b, g in j.groupby("stat"):
            cross.append({"class": c, "band": b, "team_seasons": int(len(g)),
                          "corr_of_team_diffs": float(np.corrcoef(g["diff_c"], g["diff_h"])[0, 1]),
                          "mean_abs_diff_between_sources": float((g["diff_c"] - g["diff_h"]).abs().mean()),
                          "mean_diff_cbbd": float(g["diff_c"].mean()), "mean_diff_hoopr": float(g["diff_h"].mean())})
    res["cross_source"] = cross
    REP["part4_two_sources_2025"] = res
    log(f"4: {cross}", t0)


def main() -> int:
    t0 = time.time()
    assert_not_sealed(SEASONS, context="venue charting diag")
    D = load()
    log(f"loaded non-neutral {len(D):,} rows", t0)
    REP["created_at"] = pd.Timestamp.now("UTC").isoformat()
    REP["rows_non_neutral"] = int(len(D))
    part12(D, t0)
    part3(D, t0)
    part3b(D, t0)
    part4(t0)
    out = ROOT / "docs/tests/venue_charting_asymmetry_2026-10-27.json"
    out.write_text(json.dumps(REP, indent=2, default=float), encoding="utf-8")
    log(f"wrote {out}", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
