#!/usr/bin/env python
"""
diag_rotation_wave_v10.py -- the round-10 SUPPORT measurement, run BEFORE the
round-10 pre-registration, as 17.17 was run before round 8 and 19.1 before
round 9.

The question
------------
Round 9 made the exit rate CONDITIONAL on the composition essentially correct
(94% of the real 36.5 pp span, 21.8) and the simulated floor is STILL about 25%
too bench-heavy. A correct conditional rate over a wrong state distribution
means the error lives in the rest of the transition kernel. This script measures
which part.

The kernel, stated once
-----------------------
At a possession boundary the number of the model's own predicted starters on the
floor, `n_st` in 0..5, moves by `n_st -> n_st - k_out + k_in` and nothing else.
Four objects decide it, and they are the ONLY four:

    A(n_st)              P(a wave happens at this boundary | n_st)   ARRIVAL
    S(size | n_st)                                                   SIZE
    O(k_out | size, n_st)                                            EXIT
    I(k_in | size, k_out, n_st)                                      ENTRY

plus the hard second-half reset, which lands on the predicted five and is
counted here at its realised per-boundary rate.

All four are measured through ONE function applied to the ACTUAL on-floor
sequence and to each arm's SIMULATED sequence with the SAME as-of predicted
starter set on both sides, so every row is like for like (the convention of
16.7, 17.17, 19.1 and 21.8).

The decomposition
-----------------
The four objects plus the reset define a Markov chain on `n_st`. Its stationary
distribution is compared with the OBSERVED occupancy on both sides first (the
validation: if the chain does not reproduce what was observed, none of the rest
may be read). Then one component at a time is swapped between the simulated and
the actual kernel, in BOTH directions, and the share of the occupancy gap each
one closes is reported. Order dependence is real, so both directions are
published and neither is averaged away silently.

    .venv/Scripts/python.exe scripts/diag_rotation_wave_v10.py --sim-games 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.models import rotation as R  # noqa: E402
from cbb_sim.models.rotation import actual_foul_matrix  # noqa: E402
from cbb_sim.models.rotation_v5 import MAX_WAVE  # noqa: E402
from cbb_sim.models.rotation_v7 import (  # noqa: E402
    EXIT_TC,
    FOUL_TROUBLE,
    N_ETC,
    N_EXIT_CELL,
    N_MB,
    exit_cell,
)
from cbb_sim.models.rotation_v8 import N_ST  # noqa: E402

OUT = (ROOT / "data" / "processed" / "models" / "rotation"
       / "wave_support_round10_2026-09-18.json")
#: 23.1's own two arms. `--arms` points the SAME measurement at the round-10
#: arms for 23.9's report-only objects; the default is unchanged, so the support
#: measurement the pre-registration was written on is reproducible verbatim.
ARMS = ["K1_cond_class", "Z1_exit_marg"]
MW1 = MAX_WAVE + 1
UP = 300          # the project's UNDERPOWERED threshold since round 5


# ===========================================================================
# 1. the accumulator, and the ONE walk that fills it
# ===========================================================================
def _empty() -> dict:
    return {
        "n_bnd": np.zeros(N_ST),
        "n_wave": np.zeros(N_ST),
        "size_h": np.zeros((N_ST, MAX_WAVE)),
        "kout_h": np.zeros((MAX_WAVE, N_ST, MW1)),
        "kin_h": np.zeros((MAX_WAVE, MW1, N_ST, MW1)),
        "occ_poss": np.zeros(N_ST),
        "ce_bnd": np.zeros((N_EXIT_CELL, N_ST)),
        "ce_wave": np.zeros((N_EXIT_CELL, N_ST)),
        "ce_size": np.zeros((N_EXIT_CELL, N_ST)),
        "ce_kout": np.zeros((N_EXIT_CELL, N_ST)),
        "ce_kin": np.zeros((N_EXIT_CELL, N_ST)),
        "n_reset": 0.0,
        "n_bnd_all": 0.0,
        "n_tg": 0.0,
    }


def _add(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = a[k] + v
    return a


def _walk(lu: np.ndarray, on_fouls: np.ndarray, period: np.ndarray,
          clock: np.ndarray, margin: np.ndarray, starters: set) -> dict:
    """One team-game. `lu` is (npos, 5) of player ids, `on_fouls` (npos, 5) of the
    personal fouls of those same five men at the same index.

    The H1 -> H2 boundary is EXCLUDED: the sampler hard-resets there and runs no
    wave, so it is not a draw of the kernel. Every other boundary is included,
    exactly as `build_wave_training` counts them."""
    acc = _empty()
    npos = lu.shape[0]
    if npos < 20:
        return acc
    acc["n_tg"] = 1.0
    st_prev = np.array([sum(1 for p in row if int(p) in starters) for row in lu])
    np.add.at(acc["occ_poss"], np.minimum(st_prev, N_ST - 1), 1.0)
    prev = set(int(x) for x in lu[0])
    for k in range(1, npos):
        cur = set(int(x) for x in lu[k])
        reset = bool(period[k] == 2 and period[k - 1] == 1)
        acc["n_bnd_all"] += 1.0
        if reset:
            acc["n_reset"] += 1.0
            prev = cur
            continue
        n = int(min(st_prev[k - 1], N_ST - 1))
        fs = int((on_fouls[k - 1] >= FOUL_TROUBLE).any())
        ce = int(exit_cell(np.array([period[k]]), np.array([clock[k]]),
                           np.array([margin[k]]), np.array([fs]))[0])
        acc["n_bnd"][n] += 1.0
        acc["ce_bnd"][ce, n] += 1.0
        off, on_ = prev - cur, cur - prev
        s = len(off)
        if s and len(on_) == s:
            ko = sum(1 for p in off if p in starters)
            ki = sum(1 for p in on_ if p in starters)
            sc = int(min(s, MAX_WAVE))
            acc["n_wave"][n] += 1.0
            acc["size_h"][n, sc - 1] += 1.0
            acc["kout_h"][sc - 1, n, min(ko, MAX_WAVE)] += 1.0
            acc["kin_h"][sc - 1, min(ko, MAX_WAVE), n, min(ki, MAX_WAVE)] += 1.0
            acc["ce_wave"][ce, n] += 1.0
            acc["ce_size"][ce, n] += float(sc)
            acc["ce_kout"][ce, n] += float(ko)
            acc["ce_kin"][ce, n] += float(ki)
        prev = cur
    return acc


# ===========================================================================
# 2. the kernel and its stationary distribution
# ===========================================================================
def _norm(row: np.ndarray, lo: int, hi: int, fallback: np.ndarray | None = None
          ) -> np.ndarray:
    """Clip a (MW1,) count/probability row to the feasible support and normalise.
    The clip is the sampler's own: `run_wave8` zeroes outside [lo, hi]."""
    r = np.asarray(row, dtype=np.float64).copy()
    r[:lo] = 0.0
    r[hi + 1:] = 0.0
    t = r.sum()
    if t > 0:
        return r / t
    if fallback is not None:
        return _norm(fallback, lo, hi, None)
    out = np.zeros(MW1)
    out[lo] = 1.0
    return out


def components(acc: dict) -> dict:
    """`A`, `S`, `O`, `I` and the reset rate, as probability tables. Thin cells
    fall back on their own coarser marginal and every fallback is counted."""
    nb = acc["n_bnd"]
    A = np.divide(acc["n_wave"], np.maximum(nb, 1.0))
    S = acc["size_h"] / np.maximum(acc["size_h"].sum(axis=1, keepdims=True), 1.0)
    for n in range(N_ST):
        if acc["size_h"][n].sum() < 1.0:
            tot = acc["size_h"].sum(axis=0)
            S[n] = tot / max(tot.sum(), 1.0)
    O = np.zeros((MAX_WAVE, N_ST, MW1))
    fb_o = 0
    for s in range(MAX_WAVE):
        marg = acc["kout_h"][s].sum(axis=0)
        for n in range(N_ST):
            row = acc["kout_h"][s, n]
            if row.sum() < 1.0:
                row = marg
                fb_o += 1
            O[s, n] = row / max(row.sum(), 1e-12)
    I = np.zeros((MAX_WAVE, MW1, N_ST, MW1))
    fb_i = 0
    for s in range(MAX_WAVE):
        marg_s = acc["kin_h"][s].sum(axis=(0, 1))
        for ko in range(MW1):
            marg = acc["kin_h"][s, ko].sum(axis=0)
            for n in range(N_ST):
                row = acc["kin_h"][s, ko, n]
                if row.sum() < 1.0:
                    row = marg if marg.sum() >= 1.0 else marg_s
                    fb_i += 1
                I[s, ko, n] = row / max(row.sum(), 1e-12)
    r = float(acc["n_reset"] / max(acc["n_bnd_all"], 1.0))
    return {"A": A, "S": S, "O": O, "I": I, "reset": r,
            "n_fallback_O": fb_o, "n_fallback_I": fb_i}


def kernel(cp: dict) -> np.ndarray:
    """(N_ST, N_ST) transition matrix on `n_st`, with the sampler's own support
    clips and the hard reset onto the predicted five."""
    A, S, O, I, r = cp["A"], cp["S"], cp["O"], cp["I"], cp["reset"]
    T = np.zeros((N_ST, N_ST))
    for n in range(N_ST):
        T[n, n] += 1.0 - A[n]
        for s in range(1, MAX_WAVE + 1):
            ps = S[n, s - 1]
            if ps <= 0:
                continue
            lo_o, hi_o = max(0, s - (5 - n)), min(s, n)
            if hi_o < lo_o:
                continue
            orow = _norm(O[s - 1, n], lo_o, hi_o)
            for ko in range(lo_o, hi_o + 1):
                if orow[ko] <= 0:
                    continue
                lo_i, hi_i = 0, min(s, 5 - n)
                if hi_i < lo_i:
                    continue
                irow = _norm(I[s - 1, min(ko, MAX_WAVE), n], lo_i, hi_i)
                for ki in range(lo_i, hi_i + 1):
                    if irow[ki] <= 0:
                        continue
                    nn = int(np.clip(n - ko + ki, 0, N_ST - 1))
                    T[n, nn] += A[n] * ps * orow[ko] * irow[ki]
        T[n] /= max(T[n].sum(), 1e-12)
    # the hard reset: with probability r the next state is the predicted five
    Tr = (1.0 - r) * T
    Tr[:, 5] += r
    return Tr


def stationary(T: np.ndarray) -> np.ndarray:
    p = np.full(N_ST, 1.0 / N_ST)
    for _ in range(20000):
        q = p @ T
        if np.abs(q - p).sum() < 1e-14:
            p = q
            break
        p = q
    return p / p.sum()


def metrics(p: np.ndarray) -> dict:
    return {"mean_n_st": float((np.arange(N_ST) * p).sum()),
            "p_le2": float(p[:3].sum()),
            "p_le1": float(p[:2].sum()),
            "dist": [round(float(x), 5) for x in p]}


def decompose(cs: dict, ca: dict) -> dict:
    """One component at a time, in both directions."""
    base_s, base_a = kernel(cs), kernel(ca)
    ms, ma = metrics(stationary(base_s)), metrics(stationary(base_a))
    out = {"sim": ms, "actual": ma, "swaps": {}}
    for key in ("A", "S", "O", "I", "reset"):
        cin = dict(cs)
        cin[key] = ca[key]
        cout = dict(ca)
        cout[key] = cs[key]
        m_in = metrics(stationary(kernel(cin)))
        m_out = metrics(stationary(kernel(cout)))
        rec = {"sim_plus_actual": m_in, "actual_minus": m_out}
        for met in ("mean_n_st", "p_le2"):
            den = ma[met] - ms[met]
            rec[met + "_share_in"] = (float((m_in[met] - ms[met]) / den)
                                      if abs(den) > 1e-12 else None)
            rec[met + "_share_out"] = (float((ma[met] - m_out[met]) / den)
                                       if abs(den) > 1e-12 else None)
        out["swaps"][key] = rec
    return out


# ===========================================================================
# 3. the run
# ===========================================================================
def _on_fouls_from_matrix(lu: np.ndarray, fm: np.ndarray, idx_of: dict) -> np.ndarray:
    npos = lu.shape[0]
    out = np.zeros((npos, 5), dtype=np.int64)
    if fm is None or fm.size == 0:
        return out
    for k in range(npos):
        kk = min(k, fm.shape[0] - 1)
        for j, p in enumerate(lu[k]):
            i = idx_of.get(int(p))
            if i is not None:
                out[k, j] = fm[kk, i]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-games", type=int, default=400)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--arms", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    t0 = time.time()
    global ARMS, OUT
    if args.arms:
        ARMS = [a for a in args.arms.split(",") if a]
    if args.out:
        OUT = ROOT / "data" / "processed" / "models" / "rotation" / args.out

    import train_rotation_v5 as V5T
    import train_rotation_v10 as V10T
    import train_rotation_v9 as V9T
    make_arm = V10T.make_arm10
    args_d = {"test_games": args.sim_games, "seeds": 1, "noise_seeds": 3,
              "noise_games": 30, "wave_team_games": 6000, "min_prior_games": 3,
              "fit_seed": 11, "floor_fit_seed": 101, "floor_seed": 23,
              "floor_b_arm": "Z1_exit_marg", "workers": 1, "skip_floor_a": True,
              "skip_fit": True, "k_folds": 5, "mode": "bakeoff", "smoke": False,
              "tag": "wave10"}
    c = V9T.ensure_ctx(args_d)

    accs: dict[str, dict] = {}
    per_team: dict[str, dict] = {}

    # ---- the ACTUAL sequences, with the as-of predicted starter set ---------
    kset = {k for g in c["sel"] for k in c["keys_by_game"][g]}
    tps = c["test"]["tp"]
    tps = tps[[(int(g), int(t)) in kset for g, t in zip(tps["game_id"], tps["team_id"])]]
    fouls = c["test"]["fouls"]
    gset = {int(g) for g in c["sel"]}
    fl = fouls[fouls["game_id"].isin(gset)] if fouls is not None and len(fouls) else None
    evg = ({(int(g), int(t)): d for (g, t), d in
            fl.groupby(["game_id", "team_id"], sort=False)} if fl is not None else {})
    pr_all = c["priors_static"]
    tot = _empty()
    pt: dict[int, dict] = {}
    for key, g in tps.groupby(["game_id", "team_id"], sort=False):
        key = (int(key[0]), int(key[1]))
        if key not in pr_all or key not in c["scripts"]:
            continue
        pr, sc = pr_all[key], c["scripts"][key]
        st = set(int(x) for x in pr.pids[pr.starters()[:5]])
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        npos = min(len(lu), sc.n)
        lu = lu[:npos]
        idx_of = {int(p): i for i, p in enumerate(pr.pids)}
        fm = actual_foul_matrix(evg.get(key), sc.period[:npos], sc.start_clock[:npos],
                                idx_of)
        onf = _on_fouls_from_matrix(lu, fm, idx_of)
        a = _walk(lu, onf, sc.period[:npos], sc.start_clock[:npos], sc.margin[:npos], st)
        tot = _add(tot, a)
        pt[key[1]] = _add(pt.get(key[1], _empty()), a)
    accs["ACTUAL"] = tot
    per_team["ACTUAL"] = pt

    # ---- each arm's SIMULATED sequences, one seed, same games ---------------
    for name in ARMS:
        tot = _empty()
        pt = {}
        for tag, gl in c["games_by_window"].items():
            if not gl:
                continue
            pw = V5T.priors_for("S1", tag)
            arm = make_arm(name, tag, "")
            for gid in gl:
                keys = [k for k in c["keys_by_game"][gid] if k in pw]
                if len(keys) != 2:
                    continue
                rng = R.game_stream(args.sim_seed, gid)
                for key in keys:
                    pr, sc = pw[key], c["scripts"][key]
                    lu, fh = arm.simulate(pr, sc, rng)
                    st = set(int(x) for x in pr.pids[pr.starters()[:5]])
                    idx_of = {int(p): i for i, p in enumerate(pr.pids)}
                    onf = _on_fouls_from_matrix(lu, np.asarray(fh), idx_of)
                    a = _walk(lu, onf, sc.period, sc.start_clock, sc.margin, st)
                    tot = _add(tot, a)
                    pt[key[1]] = _add(pt.get(key[1], _empty()), a)
        accs[name] = tot
        per_team[name] = pt
        print(f"[{time.time() - t0:6.0f}s] simulated {name}")

    # ---- the tables --------------------------------------------------------
    res: dict = {"config": {"sim_games": args.sim_games, "sim_seed": args.sim_seed,
                            "n_team_games": {k: int(v["n_tg"]) for k, v in accs.items()}},
                 "by_n_st": {}, "decomposition": {}, "by_cell": {}, "per_team": {},
                 "occupancy": {}}
    comps = {k: components(v) for k, v in accs.items()}
    for k, acc in accs.items():
        cp = comps[k]
        nb, nw = acc["n_bnd"], acc["n_wave"]
        rows = []
        for n in range(N_ST):
            sz = acc["size_h"][n]
            wv = max(nw[n], 1.0)
            ko = float((acc["kout_h"][:, n, :]
                        * np.arange(MW1)[None, :]).sum() / wv)
            ki = float((acc["kin_h"][:, :, n, :]
                        * np.arange(MW1)[None, None, :]).sum() / wv)
            s1 = acc["kout_h"][0, n]
            rows.append({
                "n_st": n, "n_boundaries": int(nb[n]), "n_waves": int(nw[n]),
                "n_single_swaps": int(s1.sum()),
                "p_starter_leaves_size1": (round(float(s1[1] / s1.sum()), 4)
                                           if s1.sum() else None),
                "p_wave": round(float(nw[n] / nb[n]), 5) if nb[n] else None,
                "mean_size": round(float((sz * np.arange(1, MAX_WAVE + 1)).sum()
                                         / max(sz.sum(), 1.0)), 4) if sz.sum() else None,
                "p_size1": round(float(sz[0] / sz.sum()), 4) if sz.sum() else None,
                "mean_k_out": round(ko, 4) if nw[n] else None,
                "mean_k_in": round(ki, 4) if nw[n] else None,
                "mean_drift": round(ki - ko, 4) if nw[n] else None,
                "occupancy": round(float(acc["occ_poss"][n]
                                         / max(acc["occ_poss"].sum(), 1.0)), 5),
                "underpowered": bool(nb[n] < UP),
            })
        res["by_n_st"][k] = rows
        obs = acc["occ_poss"] / max(acc["occ_poss"].sum(), 1.0)
        pred = stationary(kernel(cp))
        res["occupancy"][k] = {
            "observed": [round(float(x), 5) for x in obs],
            "chain_stationary": [round(float(x), 5) for x in pred],
            "observed_mean": round(float((np.arange(N_ST) * obs).sum()), 4),
            "chain_mean": round(float((np.arange(N_ST) * pred).sum()), 4),
            "observed_p_le2": round(float(obs[:3].sum()), 4),
            "chain_p_le2": round(float(pred[:3].sum()), 4),
            "reset_rate": round(cp["reset"], 5),
            "n_fallback_O": cp["n_fallback_O"], "n_fallback_I": cp["n_fallback_I"],
        }

    for name in ARMS:
        res["decomposition"][name] = decompose(comps[name], comps["ACTUAL"])

    # ---- by game-state cell (coarse: 3 time cells x 3 margin bands) --------
    for k, acc in accs.items():
        rows = []
        for tc in range(N_ETC):
            for mb in range(N_MB):
                ces = [ce for ce in range(N_EXIT_CELL)
                       if (ce // 2) // N_MB == tc and (ce // 2) % N_MB == mb]
                nb = acc["ce_bnd"][ces].sum()
                nw = acc["ce_wave"][ces].sum()
                sz = acc["ce_size"][ces].sum()
                ko = acc["ce_kout"][ces].sum()
                ki = acc["ce_kin"][ces].sum()
                # composition-weighted: the mean n_st at the boundaries of the cell
                nst_mean = float((acc["ce_bnd"][ces] * np.arange(N_ST)[None, :]).sum()
                                 / max(nb, 1.0))
                rows.append({
                    "time_cell": ["H1", "H2 20:00-08:00", "final 8:00"][tc],
                    "margin_band": ["|m|<=5", "|m| 6-15", "|m|>15"][mb],
                    "n_boundaries": int(nb), "n_waves": int(nw),
                    "p_wave": round(float(nw / nb), 5) if nb else None,
                    "mean_size": round(float(sz / nw), 4) if nw else None,
                    "mean_k_out": round(float(ko / nw), 4) if nw else None,
                    "mean_k_in": round(float(ki / nw), 4) if nw else None,
                    "mean_drift": round(float((ki - ko) / nw), 4) if nw else None,
                    "mean_n_st_at_boundary": round(nst_mean, 4),
                    "underpowered": bool(nb < UP),
                    # the (cell x n_st) INTERACTION the product parent cannot carry
                    "p_wave_by_nst": [
                        (round(float(acc["ce_wave"][ces, j].sum()
                                     / acc["ce_bnd"][ces, j].sum()), 4)
                         if acc["ce_bnd"][ces, j].sum() else None)
                        for j in range(N_ST)],
                    "n_bnd_by_nst": [int(acc["ce_bnd"][ces, j].sum())
                                     for j in range(N_ST)],
                })
        res["by_cell"][k] = rows

    # ---- per team ----------------------------------------------------------
    act_pt = per_team["ACTUAL"]
    for name in ARMS:
        sim_pt = per_team[name]
        tids = [t for t in sim_pt
                if t in act_pt and sim_pt[t]["n_bnd"].sum() >= UP
                and act_pt[t]["n_bnd"].sum() >= UP]
        def _mn(d):
            o = d["occ_poss"]
            return float((np.arange(N_ST) * o).sum() / max(o.sum(), 1.0))
        def _pw(d):
            return float(d["n_wave"].sum() / max(d["n_bnd"].sum(), 1.0))
        sim_m = np.array([_mn(sim_pt[t]) for t in tids])
        act_m = np.array([_mn(act_pt[t]) for t in tids])
        sim_w = np.array([_pw(sim_pt[t]) for t in tids])
        act_w = np.array([_pw(act_pt[t]) for t in tids])
        res["per_team"][name] = {
            "n_teams_powered": len(tids),
            "n_teams_underpowered": len(sim_pt) - len(tids),
            "mean_n_st_sim": round(float(sim_m.mean()), 4) if len(sim_m) else None,
            "mean_n_st_act": round(float(act_m.mean()), 4) if len(act_m) else None,
            "sd_n_st_sim": round(float(sim_m.std(ddof=1)), 4) if len(sim_m) > 1 else None,
            "sd_n_st_act": round(float(act_m.std(ddof=1)), 4) if len(act_m) > 1 else None,
            "corr_n_st": (round(float(np.corrcoef(sim_m, act_m)[0, 1]), 4)
                          if len(sim_m) > 2 else None),
            "mean_abs_dev_n_st": (round(float(np.abs(sim_m - act_m).mean()), 4)
                                  if len(sim_m) else None),
            "p_wave_sim": round(float(sim_w.mean()), 5) if len(sim_w) else None,
            "p_wave_act": round(float(act_w.mean()), 5) if len(act_w) else None,
            "corr_p_wave": (round(float(np.corrcoef(sim_w, act_w)[0, 1]), 4)
                            if len(sim_w) > 2 else None),
        }

    res["seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # ---- print -------------------------------------------------------------
    print("\n=== P(wave | n_st), mean size, mean k_out, mean k_in, drift ===")
    for k in ["ACTUAL"] + ARMS:
        print(f"-- {k}")
        for r in res["by_n_st"][k]:
            print(f"   n_st {r['n_st']}  bnd {r['n_boundaries']:7d}  "
                  f"p_wave {str(r['p_wave']):8s}  size {str(r['mean_size']):7s}  "
                  f"k_out {str(r['mean_k_out']):7s}  k_in {str(r['mean_k_in']):7s}  "
                  f"drift {str(r['mean_drift']):8s}  occ {r['occupancy']:.4f}"
                  f"{'  UNDERPOWERED' if r['underpowered'] else ''}")
    print("\n=== occupancy: observed vs the chain's own stationary distribution ===")
    for k, v in res["occupancy"].items():
        print(f"{k:16s} obs mean {v['observed_mean']:.4f} chain {v['chain_mean']:.4f} "
              f" obs P(<=2) {v['observed_p_le2']:.4f} chain {v['chain_p_le2']:.4f} "
              f" reset {v['reset_rate']:.5f}")
    print("\n=== decomposition: share of the mean-n_st gap each component closes ===")
    for name in ARMS:
        d = res["decomposition"][name]
        print(f"-- {name}: sim {d['sim']['mean_n_st']:.4f} -> actual "
              f"{d['actual']['mean_n_st']:.4f}  (P<=2 {d['sim']['p_le2']:.4f} -> "
              f"{d['actual']['p_le2']:.4f})")
        for key, rec in d["swaps"].items():
            print(f"   {key:6s} mean share in {rec['mean_n_st_share_in']:+.3f} "
                  f"out {rec['mean_n_st_share_out']:+.3f}   "
                  f"P<=2 share in {rec['p_le2_share_in']:+.3f} "
                  f"out {rec['p_le2_share_out']:+.3f}")
    print(f"\nwritten -> {OUT}  ({res['seconds']}s)")


if __name__ == "__main__":
    main()
