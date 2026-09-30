"""diag_g9_within_season_v1.py -- Lane F step 1 (2026-09-30): which inputs produce the sim's excess
within-season movement of mean margin (SD 3.56 vs the close's 2.28; 0.091 of the 0.098 close-referenced
slope miss, `docs/tests/g9_g6_margin_slope_home_diagnostic_2026-09-30.md` section 1.2)?

DIAGNOSTIC ONLY, OFFLINE, NO SIM. Loads the SERVED sub-model artifacts through the engine's own adapter
classes (read-only; nothing under src/ is edited) and re-predicts each (game, offence side) at a fixed
reference game state under counterfactual inputs:

  F0      served inputs, served per-game artifact (S1 schedules)
  Toff/Tdef freeze   offence-owned / defence-owned as-of TEAM features -> that team's fold-2 season mean
  Toff/Tdef lag{1,5} offence-owned / defence-owned team features taken from that team's k-th previous game
                     (backward-looking reference: no future information, so the movement test is unbiased)
  D       days_since_start -> season mean (season_idx is constant)
  A       every game scored by the FIRST refit (trained only before 2024-11-01; no leak)
  Pf      per-player slot features (shooter as-of) -> that player's season mean on this team
  Pw      per-player rotation share and usage rates -> that player's season mean on this team
  C1..C5  cumulative: T(both) -> +D -> +A -> +Pf -> +Pw

Team-level rates per (game, side): PO first-chance P(TOV), FT-trip prob, shot shares (averaged over two
reference states); fg_make P(make) per class and FT P(make) weighted over slots by rot_share x usage rate;
rebound P(OREB | miss) over a fixed miss-type mix.

Usage:
    .venv/Scripts/python.exe scripts/diag_g9_within_season_v1.py --part predict
    .venv/Scripts/python.exe scripts/diag_g9_within_season_v1.py --part analyse
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "LIGHTGBM_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = Path("results/g9ws_diag")
PRED = OUT / "preds_v1.parquet"
SHOT = ("rim", "jump2", "three")
FGCLS = {"rim": "FGA_rim", "jump2": "FGA_jump2", "three": "FGA_3"}
USAGE_IDX = {"rim": 0, "jump2": 1, "three": 2, "TOV": 3, "FT": 4}

VARIANTS = {
    "F0": {},
    "Toff_fz": {"Toff": "fz"}, "Tdef_fz": {"Tdef": "fz"},
    "Toff_l1": {"Toff": "l1"}, "Tdef_l1": {"Tdef": "l1"},
    "Toff_l5": {"Toff": "l5"}, "Tdef_l5": {"Tdef": "l5"},
    "D": {"D": True}, "A": {"A": True}, "Pf": {"Pf": True}, "Pw": {"Pw": True},
    "C1": {"Toff": "fz", "Tdef": "fz"},
    "C2": {"Toff": "fz", "Tdef": "fz", "D": True},
    "C3": {"Toff": "fz", "Tdef": "fz", "D": True, "A": True},
    "C4": {"Toff": "fz", "Tdef": "fz", "D": True, "A": True, "Pf": True},
    "C5": {"Toff": "fz", "Tdef": "fz", "D": True, "A": True, "Pf": True, "Pw": True},
}


def owner(col: str) -> str:
    if col.startswith("off_"):
        return "off"
    if col.startswith("opp_def_") or col.startswith("def_"):
        return "def"
    return "other"


# --------------------------------------------------------------------------- helpers on (G, 2, F) team blocks
def team_ids(games):
    return np.stack([games["home_team_id"].to_numpy(), games["away_team_id"].to_numpy()], axis=1)  # (G,2) offence team


def season_mean_block(block, cols_idx, own_team):
    """own_team (G,2): the team that OWNS these columns in row (g, s). Replace by that team's mean."""
    out = block.copy()
    flat_t = own_team.reshape(-1)
    for j in cols_idx:
        v = block[:, :, j].reshape(-1).astype(np.float64)
        s = pd.Series(v).groupby(flat_t).transform("mean").to_numpy()
        out[:, :, j] = s.reshape(own_team.shape)
    return out


def lag_block(block, cols_idx, own_team, order, k):
    """Take the owner's k-th previous row (by game order). Rows without one -> NaN."""
    out = block.copy().astype(np.float64)
    flat_t = own_team.reshape(-1)
    G = own_team.shape[0]
    gi = np.repeat(np.arange(G), 2); si = np.tile([0, 1], G)
    df = pd.DataFrame({"t": flat_t, "ord": order[gi], "g": gi, "s": si})
    df = df.sort_values(["t", "ord"])
    df["lag_g"] = df.groupby("t")["g"].shift(k); df["lag_s"] = df.groupby("t")["s"].shift(k)
    has = df["lag_g"].notna().to_numpy()
    g, s = df["g"].to_numpy(), df["s"].to_numpy()
    lg = df["lag_g"].fillna(0).astype(int).to_numpy(); ls = df["lag_s"].fillna(0).astype(int).to_numpy()
    for j in cols_idx:
        vals = np.where(has, block[lg, ls, j], np.nan)
        out[g, s, j] = vals
    return out


def apply_team(block, names, spec, own_off, own_def, order):
    b = block.astype(np.float64).copy()
    off_idx = [i for c, i in names.items() if owner(c) == "off"]
    def_idx = [i for c, i in names.items() if owner(c) == "def"]
    for key, idx, own in (("Toff", off_idx, own_off), ("Tdef", def_idx, own_def)):
        m = spec.get(key)
        if m == "fz":
            b = season_mean_block(b, idx, own)
        elif m in ("l1", "l5"):
            b = lag_block(b, idx, own, order, int(m[1]))
    if spec.get("D") and "days_since_start" in names:
        j = names["days_since_start"]; b[:, :, j] = float(np.nanmean(block[:, :, j]))
    return b


def player_mean(arr, roster, team_off):
    """arr (G,2,S,...) per slot; replace by the (team, player) season mean where player id >= 0."""
    G, _, S = roster.shape
    tid = np.broadcast_to(team_off[:, :, None], roster.shape).reshape(-1)
    pid = roster.reshape(-1)
    flat = arr.reshape(G * 2 * S, -1).astype(np.float64)
    key = pd.Series(tid.astype(np.int64) * 10_000_000_000 + np.where(pid >= 0, pid, -1 - np.arange(len(pid))))
    out = flat.copy()
    valid = pid >= 0
    for j in range(flat.shape[1]):
        m = pd.Series(flat[:, j]).groupby(key.to_numpy()).transform("mean").to_numpy()
        out[valid, j] = m[valid]
    return out.reshape(arr.shape)


# --------------------------------------------------------------------------- predict
def part_predict():
    from cbb_sim.engine.inputs import EngineInputs
    from cbb_sim.engine import adapters as AD
    OUT.mkdir(parents=True, exist_ok=True)
    inp = EngineInputs.load("data/processed/models/engine", "F2_2025")
    G = inp.n_games
    ev = AD.EventAdapter.load(inp, "round2_s1", "F2", 2025)
    fg = AD.FgMakeAdapter.load(inp, "F2", "decision8", "round4_B1")
    ft = AD.FreeThrowAdapter.load(inp, "F2", "s1_conf_aligned")
    rb = AD.ReboundAdapter.load(inp, "F2", "s1_weekly")
    games = inp.games.copy()
    order = pd.to_datetime(games["tipoff_utc"]).rank(method="first").to_numpy()
    toff = team_ids(games)                   # offence team in row (g, s)
    tdef = toff[:, ::-1]                     # defence team in row (g, s)
    idx = json.loads((AD.ENGINE_DIR / "event_round2_s1_F2_2025" / "index.json").read_text(encoding="utf-8"))
    r2_names = {c: i for i, c in enumerate(idx["team_cols"])}
    gidx = np.repeat(np.arange(G), 2); sidx = np.tile([0, 1], G)
    S = inp.n_slots
    SI = AD.STATE_INDEX

    def state(n, **kw):
        st = np.zeros((n, len(AD.STATE_COLS)))
        for k, v in kw.items():
            st[:, SI[k]] = v
        return st

    # segments
    seg_po = ev.manifests["first"].segments(gidx)
    seg_fg = {k: fg.manifests[FGCLS[k]].segments(gidx) for k in SHOT}
    seg_ft = ft.manifest.segments(gidx)
    seg_rb = rb.manifest.segments(gidx)
    json.dump({"po_refits": [str(e.refit_date) for e in ev.manifests["first"].entries],
               "fg_refits": [str(e.refit_date) for e in fg.manifests["FGA_rim"].entries],
               "ft_refits": [str(e.refit_date) for e in ft.manifest.entries],
               "rb_refits": [str(e.refit_date) for e in rb.manifest.entries]},
              open(OUT / "refits_v1.json", "w"), indent=1)

    def by_seg(models, seg, fixA, fn):
        out = None
        s = np.zeros_like(seg) if fixA else seg
        for k in np.unique(s):
            r = np.flatnonzero(s == k)
            v = fn(models[k], r)
            if out is None:
                out = np.full((len(seg),) + v.shape[1:], np.nan)
            out[r] = v
        return out

    rows = []
    cache_po, cache_rb = {}, {}
    for vname, spec in VARIANTS.items():
        print("variant", vname, flush=True)
        fixA = bool(spec.get("A"))
        team_key = (spec.get("Toff"), spec.get("Tdef"), bool(spec.get("D")), fixA)
        rec = {"game_idx": gidx, "side": sidx, "variant": np.full(len(gidx), vname)}
        # ---- possession outcome (first chance), two reference states
        if team_key not in cache_po:
            blk = apply_team(ev.team_block, r2_names, spec, toff, tdef, order)
            tb = blk[gidx, sidx]
            acc = 0
            for kw in ({"period": 1, "seconds_remaining": 900, "score_diff": 0, "in_bonus": 0, "is_transition": 0},
                       {"period": 2, "seconds_remaining": 600, "score_diff": 0, "in_bonus": 1, "is_transition": 0}):
                st = state(len(gidx), **kw)
                ok = np.isfinite(tb).all(axis=1)
                p = np.full((len(gidx), len(AD.PO.CLASSES)), np.nan)
                pp = by_seg(ev.models_first, seg_po[ok], fixA,
                            lambda m, r: m.predict_proba(np.ascontiguousarray(
                                AD._assemble(ev.plan_first, tb[ok][r], None, st[ok][r]), dtype=np.float32)))
                p[ok] = pp
                acc = acc + p / 2
            # rebound: miss-type mix
            blk2 = apply_team(inp.team_static, inp.team_names, spec, toff, tdef, order)
            tb2 = blk2[gidx, sidx]
            ok2 = np.isfinite(tb2).all(axis=1)
            orb = np.zeros(len(gidx))
            for mt, w in (("miss_rim", 0.35), ("miss_jump2", 0.35), ("miss_three", 0.30)):
                st = state(len(gidx), period=1, seconds_remaining=600, score_diff=0, in_bonus=0, **{mt: 1})
                pr = np.full((len(gidx), 3), np.nan)
                pr[ok2] = by_seg(rb.models_by_seg, seg_rb[ok2], fixA,
                                 lambda m, r: m.predict_proba(np.ascontiguousarray(
                                     AD._assemble(rb.plan, tb2[ok2][r], None, st[ok2][r]), dtype=np.float32)))
                orb = orb + w * pr[:, 0] / (pr[:, 0] + pr[:, 1])
            cache_po[team_key] = (acc, orb, blk2)
        acc, orb, blk2 = cache_po[team_key]
        C = list(AD.PO.CLASSES)
        rec["p_tov"] = acc[:, C.index("TOV")]
        rec["p_trip"] = acc[:, C.index("FT_trip_shooting")] + acc[:, C.index("FT_trip_bonus")]
        fga = acc[:, C.index("FGA_rim")] + acc[:, C.index("FGA_jump2")] + acc[:, C.index("FGA_3")]
        rec["s_rim"] = acc[:, C.index("FGA_rim")] / fga
        rec["s_jump2"] = acc[:, C.index("FGA_jump2")] / fga
        rec["s_three"] = acc[:, C.index("FGA_3")] / fga
        rec["p_oreb"] = orb
        # ---- slot-level models: fg_make x3, free throw
        slot = inp.slot_static
        if spec.get("Pf"):
            slot = player_mean(slot, inp.roster_cbbd, toff)
        share, usage = inp.rot_share, inp.usage_rate
        if spec.get("Pw"):
            share = player_mean(share[..., None], inp.roster_cbbd, toff)[..., 0]
            usage = player_mean(usage, inp.roster_cbbd, toff)
        tb2 = blk2[gidx, sidx]
        ok2 = np.isfinite(tb2).all(axis=1)
        valid = inp.roster_valid[gidx, sidx]                        # (n, S)
        for k in SHOT:
            num = np.zeros(len(gidx)); den = np.zeros(len(gidx))
            st = state(len(gidx), period=1, seconds_remaining=600, in_bonus=0, chance_number=1,
                       chance_elapsed_s=12, is_transition_f=0)
            for s_ in range(S):
                w = share[gidx, sidx, s_] * usage[gidx, sidx, s_, USAGE_IDX[k]] * valid[:, s_]
                use = ok2 & (w > 1e-4)
                if not use.any():
                    continue
                sl = slot[gidx, sidx, s_]
                p = by_seg(fg.models_by_seg[FGCLS[k]], seg_fg[k][use], fixA,
                           lambda m, r: m.predict_proba(np.ascontiguousarray(
                               AD._assemble(fg.plans[FGCLS[k]], tb2[use][r], sl[use][r], st[use][r]),
                               dtype=np.float32))[:, AD.FG.CLASS_INDEX["MAKE"]])
                num[use] += w[use] * p; den[use] += w[use]
            rec[f"p_make_{k}"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        num = np.zeros(len(gidx)); den = np.zeros(len(gidx))
        st = state(len(gidx), period=2, seconds_remaining=600, score_diff=0, in_bonus=1)
        for s_ in range(S):
            w = share[gidx, sidx, s_] * usage[gidx, sidx, s_, USAGE_IDX["FT"]] * valid[:, s_]
            use = w > 1e-4
            if not use.any():
                continue
            sl = slot[gidx, sidx, s_]
            p = by_seg(ft.models_by_seg, seg_ft[use], fixA,
                       lambda m, r: m.predict_proba(np.ascontiguousarray(
                           AD._assemble(ft.plan, np.nan_to_num(tb2[use][r]), sl[use][r], st[use][r]),
                           dtype=np.float32))[:, AD.FT.CLASS_INDEX["MAKE"]])
            num[use] += w[use] * p; den[use] += w[use]
        rec["p_ft"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        rows.append(pd.DataFrame(rec))
    out = pd.concat(rows, ignore_index=True)
    out["game_id"] = games["game_id"].to_numpy()[out["game_idx"].to_numpy()]
    out["seg_po"] = np.tile(seg_po, len(VARIANTS)); out["seg_rb"] = np.tile(seg_rb, len(VARIANTS))
    out["seg_ft"] = np.tile(seg_ft, len(VARIANTS)); out["seg_fg"] = np.tile(seg_fg["rim"], len(VARIANTS))
    out.to_parquet(PRED, index=False)
    print("wrote", PRED, out.shape)


# --------------------------------------------------------------------------- analyse
RATE_MAP = {  # offline column -> (actual per-side rate key in actual_channels_v1, weight key)
    "p_tov": ("t", "P"), "p_trip": ("rp", "P"), "s_rim": ("s_rim", "fga"), "s_jump2": ("s_jump", "fga"),
    "s_three": ("s_3", "fga"), "p_make_rim": ("p_rim", "n_fga_rim"), "p_make_jump2": ("p_jump", "n_fga_jump"),
    "p_make_three": ("p_3", "n_fga_3"), "p_oreb": ("rho", "chances"), "p_ft": ("f", "n_fta"),
}
SUBMODEL = {"p_tov": "possession_outcome TOV", "p_trip": "possession_outcome FT trip",
            "s_rim": "possession_outcome shot mix", "s_jump2": "possession_outcome shot mix",
            "s_three": "possession_outcome shot mix", "p_make_rim": "fg_make rim", "p_make_jump2": "fg_make jump2",
            "p_make_three": "fg_make three", "p_oreb": "rebound OREB", "p_ft": "free_throw FT%"}
CH_SIM = {"tov": ["tov"], "ft_trips": ["ft_trips"], "mix": ["mix_rim", "mix_jump", "mix_3"],
          "make_rim": ["make_rim"], "make_jump": ["make_jump"], "make_3": ["make_3"], "oreb": ["oreb"],
          "ft_pct": ["ft_pct"]}
CH_OWNER = {"tov": "possession_outcome", "ft_trips": "possession_outcome", "mix": "possession_outcome",
            "make_rim": "fg_make", "make_jump": "fg_make", "make_3": "fg_make", "oreb": "rebound",
            "ft_pct": "free_throw"}


def offline_channels(w: pd.DataFrame, ref: dict, P: float) -> dict:
    """w: wide frame indexed by game with columns (rate, side). Linear margin channels (points)."""
    Q = sum({"rim": 2, "jump": 2, "3": 3}[k] * ref[f"s_{k}"] * ref[f"p_{k}"] for k in ("rim", "jump", "3"))
    A = 1 + ref["rho"] * ref["m"] - ref["t"] - 0.44 * ref["rp"]
    d = lambda c: w[(c, 0)] - w[(c, 1)]  # noqa: E731
    return {
        "tov": P * (-Q) * d("p_tov"),
        "ft_trips": P * (ref["f"] - 0.44 * Q) * 2.0 * d("p_trip"),
        "mix": P * A * (2 * ref["p_rim"] * d("s_rim") + 2 * ref["p_jump"] * d("s_jump2") + 3 * ref["p_3"] * d("s_three")),
        "make_rim": P * A * 2 * ref["s_rim"] * d("p_make_rim"),
        "make_jump": P * A * 2 * ref["s_jump"] * d("p_make_jump2"),
        "make_3": P * A * 3 * ref["s_3"] * d("p_make_three"),
        "oreb": P * Q * ref["m"] * d("p_oreb"),
        "ft_pct": P * ref["rp"] * d("p_ft"),
    }


def wls(y, X, w):
    sw = np.sqrt(w)
    Z = np.column_stack([np.ones(len(y)), X]) * sw[:, None]
    b, *_ = np.linalg.lstsq(Z, y * sw, rcond=None)
    r = y * sw - Z @ b
    s2 = float(r @ r / (len(y) - Z.shape[1]))
    cov = s2 * np.linalg.inv(Z.T @ Z)
    return b, np.sqrt(np.diag(cov))


def part_analyse():
    import diag_g9_g6_margin_v1 as M1
    res = {}
    an = json.load(open("results/g9g6_diag/analysis_v1.json"))
    ref, P = an["ref_rates"], an["P_ref"]
    pr = pd.read_parquet(PRED)
    rates = list(RATE_MAP)
    frame, _ = M1.load_frame("A")
    d = frame.dropna(subset=["Y_tov", "close_margin"]).reset_index(drop=True)
    gids = d["game_id"].to_numpy()
    wide = {v: g.set_index(["game_id", "side"])[rates].unstack("side") for v, g in pr.groupby("variant")}
    ch = {v: pd.DataFrame(offline_channels(wide[v], ref, P)).reindex(gids) for v in wide}
    # ---- scale offline channels to the sim's own per-game channel means
    scale, corr = {}, {}
    for c, cols in CH_SIM.items():
        xs = d[[f"X_{k}" for k in cols]].sum(axis=1).to_numpy()
        xo = ch["F0"][c].to_numpy()
        m = np.isfinite(xo)
        scale[c] = M1.slope(xs[m], xo[m]); corr[c] = float(np.corrcoef(xs[m], xo[m])[0, 1])
    res["scale"] = scale; res["corr_sim_offline"] = corr
    # ---- family components (sim points), per channel
    x = d["sim_margin_mean"].to_numpy(float); c_ = d["close_margin"].to_numpy(float)
    vx = np.var(x, ddof=1)
    Xd = M1.fe_design(d)
    fams = ["Toff", "Tdef", "Tint", "D", "A", "Pf", "Pw", "frozen", "not_harnessed"]
    comp = {f: np.zeros(len(d)) for f in fams}
    compc = {}
    for c, cols in CH_SIM.items():
        s = scale[c]
        g = lambda v: np.nan_to_num(ch[v][c].to_numpy(), nan=0.0) * s  # noqa: E731
        f0, c1, c2, c3, c4, c5 = g("F0"), g("C1"), g("C2"), g("C3"), g("C4"), g("C5")
        mt = f0 - c1; mto = f0 - g("Toff_fz"); mtd = f0 - g("Tdef_fz")
        parts = {"Toff": mto, "Tdef": mtd, "Tint": mt - mto - mtd, "D": c1 - c2, "A": c2 - c3, "Pf": c3 - c4,
                 "Pw": c4 - c5, "frozen": c5,
                 "not_harnessed": d[[f"X_{k}" for k in cols]].sum(axis=1).to_numpy() - f0}
        for f, v in parts.items():
            comp[f] += v; compc[(c, f)] = v
    for cc in ("pace", "possdiff", "reb_chances"):
        comp["not_harnessed"] += d[f"X_{cc}"].to_numpy(); compc[(cc, "not_harnessed")] = d[f"X_{cc}"].to_numpy()
    assert np.allclose(sum(comp.values()), x), "family components must sum to sim_margin_mean"
    within = {f: M1.fe_fit(Xd, v)["resid"] for f, v in comp.items()}
    Cw = M1.fe_fit(Xd, c_)["resid"]; Xw = M1.fe_fit(Xd, x)["resid"]
    Z = np.column_stack([np.ones(len(d))] + [within[f] for f in fams])
    beta, *_ = np.linalg.lstsq(Z, Cw, rcond=None)
    rr = Cw - Z @ beta
    se = np.sqrt(np.diag(float(rr @ rr / (len(d) - Z.shape[1])) * np.linalg.inv(Z.T @ Z)))
    fam_rows = []
    for i, f in enumerate(fams):
        k = float((1 - beta[i + 1]) * np.cov(within[f], x)[0, 1] / vx)
        fam_rows.append({"family": f, "sd_within_pts": float(np.std(within[f], ddof=1)), "beta_close": float(beta[i + 1]),
                         "se": float(se[i + 1]), "k": k, "pts_at_1sd": k * float(np.sqrt(vx))})
    res["families"] = fam_rows
    res["k_within_total"] = float(np.cov(Xw - Cw, x)[0, 1] / vx)
    res["sd_Xw"] = float(np.std(Xw, ddof=1)); res["sd_Cw"] = float(np.std(Cw, ddof=1))
    bmap = {f: beta[i + 1] for i, f in enumerate(fams)}
    cells = []
    for (c, f), v in compc.items():
        wv = M1.fe_fit(Xd, v)["resid"]
        cells.append({"channel": c, "owner": CH_OWNER.get(c, "clock"), "family": f,
                      "sd_within_pts": float(np.std(wv, ddof=1)),
                      "k": float((1 - bmap[f]) * np.cov(wv, x)[0, 1] / vx)})
    res["cells"] = cells
    # ---- by weeks into season
    days = (pd.to_datetime(d["game_date"]) - pd.to_datetime(d["game_date"]).min()).dt.days.to_numpy()
    wk = days // 7
    bands = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 30)]
    wrows = []
    for lo, hi in bands:
        m = (wk >= lo) & (wk < hi)
        vxb = np.var(x[m], ddof=1)
        row = {"weeks": f"{lo}-{hi - 1}", "n": int(m.sum()), "sd_Xw": float(np.std(Xw[m], ddof=1)),
               "sd_Cw": float(np.std(Cw[m], ddof=1)), "k_within": float(np.cov((Xw - Cw)[m], x[m])[0, 1] / vxb)}
        for f in fams:
            row[f"sd_{f}"] = float(np.std(within[f][m], ddof=1))
            row[f"k_{f}"] = float((1 - bmap[f]) * np.cov(within[f][m], x[m])[0, 1] / vxb)
        wrows.append(row)
    res["weeks"] = wrows
    # ---- rate level: within-team SD and unbiased lag slopes (offline, per rate x side of ownership)
    Y = pd.read_parquet("results/g9g6_diag/actual_channels_v1.parquet").set_index("game_id")
    pw = {v: g.set_index(["game_id", "side"]) for v, g in pr.groupby("variant")}
    base = pw["F0"]
    act = []
    for s, pre in ((0, "h"), (1, "a")):
        a = pd.DataFrame(index=Y.index)
        for r, (k, wk_) in RATE_MAP.items():
            a[f"y_{r}"] = Y[f"{pre}_{k}"]
            if wk_ == "P":
                a[f"w_{r}"] = Y[f"{pre}_P"]
            elif wk_ == "fga":
                a[f"w_{r}"] = Y[f"{pre}_n_fga_rim"] + Y[f"{pre}_n_fga_jump"] + Y[f"{pre}_n_fga_3"]
            elif wk_ == "chances":
                a[f"w_{r}"] = Y[f"{pre}_n_oreb"] + Y[f"{pre}_n_oppdreb"]
            else:
                a[f"w_{r}"] = Y[f"{pre}_{wk_}"]
        a["side"] = s
        act.append(a.reset_index().set_index(["game_id", "side"]))
    act = pd.concat(act)
    games = pr[pr["variant"] == "F0"][["game_id", "side", "game_idx"]].set_index(["game_id", "side"])
    rrows = []
    for r in rates:
        x0 = base[r]
        dfr = pd.DataFrame({"x0": x0}).join(act[[f"y_{r}", f"w_{r}"]], how="inner").dropna()
        # within-team SD of F0 predictions: residual of offence-team + defence-team FE (additive, rate scale)
        for side_key in ("off", "def"):
            for lag in ("l1", "l5"):
                v = f"T{side_key}_{lag}"
                xl = pw[v][r].reindex(dfr.index)
                m = xl.notna() & (dfr[f"w_{r}"] > 0)
                if m.sum() < 200:
                    continue
                y = dfr.loc[m, f"y_{r}"].to_numpy(float); w = dfr.loc[m, f"w_{r}"].to_numpy(float)
                xlv = xl[m].to_numpy(float); mv = dfr.loc[m, "x0"].to_numpy(float) - xlv
                if float(np.std(mv)) < 1e-9:
                    rrows.append({"rate": r, "submodel": SUBMODEL[r], "side": side_key, "lag": lag, "n": int(m.sum()),
                                  "sd_move": 0.0, "b_level": np.nan, "b_move": np.nan, "se_move": np.nan, "ratio": np.nan})
                    continue
                b, seb = wls(y, np.column_stack([xlv, mv]), w)
                rrows.append({"rate": r, "submodel": SUBMODEL[r], "side": side_key, "lag": lag, "n": int(m.sum()),
                              "sd_move": float(np.std(mv)), "b_level": float(b[1]), "b_move": float(b[2]),
                              "se_move": float(seb[2]), "ratio": float(b[2] / b[1]) if b[1] != 0 else np.nan})
    res["lag_slopes"] = rrows
    # within-team SD of predictions by family (one-at-a-time freeze), rate units, per side of ownership
    sd_rows = []
    for r in rates:
        tmp = pd.DataFrame({v: pw[v][r] for v in ("F0", "Toff_fz", "Tdef_fz", "D", "A", "Pf", "Pw", "C5")})
        tmp = tmp.join(games)
        tmp["off_team"] = np.nan
        sd_rows.append({"rate": r, "submodel": SUBMODEL[r],
                        **{f"sd_move_{v}": float((tmp["F0"] - tmp[v]).std()) for v in ("Toff_fz", "Tdef_fz", "D", "A", "Pf", "Pw")},
                        "sd_F0": float(tmp["F0"].std()), "sd_C5": float(tmp["C5"].std())})
    res["rate_family_sd"] = sd_rows
    # ---- refit boundaries: artifact effect steps between a team's consecutive games (offence rows)
    f0 = pr[pr["variant"] == "F0"].copy(); fa = pr[pr["variant"] == "A"].set_index(["game_id", "side"])
    ginfo = frame.set_index("game_id")[["home_team_id", "away_team_id", "tipoff_utc"]]
    f0 = f0.join(ginfo, on="game_id")
    f0["team"] = np.where(f0["side"] == 0, f0["home_team_id"], f0["away_team_id"])
    f0 = f0.dropna(subset=["team"]).sort_values(["team", "tipoff_utc"])
    brow = []
    for r, segc in (("p_tov", "seg_po"), ("p_trip", "seg_po"), ("s_three", "seg_po"), ("p_make_rim", "seg_fg"),
                    ("p_make_jump2", "seg_fg"), ("p_make_three", "seg_fg"), ("p_oreb", "seg_rb"), ("p_ft", "seg_ft")):
        e = f0[r].to_numpy() - fa[r].reindex(pd.MultiIndex.from_arrays([f0["game_id"], f0["side"]])).to_numpy()
        t = f0["team"].to_numpy(); sg = f0[segc].to_numpy(); xv = f0[r].to_numpy()
        same = t[1:] == t[:-1]
        cross = same & (sg[1:] != sg[:-1]); within_ = same & (sg[1:] == sg[:-1])
        de = np.abs(np.diff(e)); dx = np.abs(np.diff(xv))
        brow.append({"rate": r, "segments": segc, "n_cross": int(cross.sum()), "n_within": int(within_.sum()),
                     "mean_abs_dEffect_cross": float(np.nanmean(de[cross])), "mean_abs_dEffect_within": float(np.nanmean(de[within_])),
                     "mean_abs_dx_cross": float(np.nanmean(dx[cross])), "mean_abs_dx_within": float(np.nanmean(dx[within_])),
                     "sd_effect_within_team": float(pd.Series(e).groupby(t).transform(lambda s: s - s.mean()).std())})
    res["refit_boundary"] = brow
    json.dump(res, open(OUT / "analyse_v1.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(json.dumps({k: res[k] for k in ("scale", "corr_sim_offline", "k_within_total", "sd_Xw", "sd_Cw")}, indent=1))
    print(pd.DataFrame(res["families"]).round(4).to_string())
    cdf = pd.DataFrame(res["cells"])
    print(cdf.pivot_table(index=["owner", "channel"], columns="family", values="k", aggfunc="sum").round(4).to_string())
    print(cdf.pivot_table(index=["owner", "channel"], columns="family", values="sd_within_pts", aggfunc="sum").round(3).to_string())
    print(pd.DataFrame(res["weeks"]).round(3).to_string())
    print(pd.DataFrame(res["lag_slopes"]).round(4).to_string())
    print(pd.DataFrame(res["rate_family_sd"]).round(4).to_string())
    print(pd.DataFrame(res["refit_boundary"]).round(5).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", choices=["predict", "analyse"])
    a = ap.parse_args()
    parts = a.part or ["predict", "analyse"]
    if "predict" in parts:
        part_predict()
    if "analyse" in parts:
        part_analyse()
