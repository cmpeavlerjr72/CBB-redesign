#!/usr/bin/env python
"""
diag_rotation_wave_v5.py -- ROUND-5 EVIDENCE: the JOINT structure of a
substitution at a dead ball.

Why this exists
---------------
Round 4 changed family to per-player discrete-time substitution hazards and
proved two things at once (L30, `docs/models/rotation/experiments.md` section
11): the fitted hazards reproduce every marginal rate the audit measured (a
bench player's exit at a period boundary 0.786 predicted vs 0.791 actual), and
the arms built on them still cannot make a COORDINATED substitution. Five
independent Bernoulli coins spread the same total exits over more boundaries:
0.206 substitutions per boundary against a real 0.152, 19.7 distinct lineups per
team-game against 14.8, and an 11 pp shortfall at the second-half tip when the
reset has to be earned.

So the quantity round 5 needs is not another per-player rate. It is the JOINT
object: at a dead ball, does a team substitute AT ALL, how many players move
when it does, and who. This script measures exactly that, with confidence
intervals, and -- per L25 -- computes on paper what a wave model built from
these tables plus round 4's own per-player hazards would produce, BEFORE the
round-5 pre-registration is written and before any arm is fitted.

What a "wave" is here
---------------------
At every possession boundary k of a team-game (the resolution the engine acts
at, and the only resolution CBBD supports -- there are no Substitution rows in
2024 at all, features.md section 3):

    wave(k)  = the team's on-floor SET at possession k differs from k-1
    size(k)  = |on(k-1) \\ on(k)|, the number of players who leave, which under
               the five-on-the-floor constraint is also the number who enter

`prev_end` is the possession's own `start_reason`, i.e. the dead-ball type, and
is exactly `engine.state.PREV_END_LEVELS`.

Descriptive, not predictive
---------------------------
Candidate pool is every player who appears in the team-game and the foul state
is that game's own `PersonalFoul` events -- correct for a MEASUREMENT of
coaching behaviour, and never what the bake-off's own training path uses. The
as-of share used by the reachability probe IS as-of (strictly earlier games),
because it feeds round 4's fitted hazards, which were fitted on that column.

Usage
-----
    .venv/Scripts/python.exe scripts/diag_rotation_wave_v5.py --mode sweep
    .venv/Scripts/python.exe scripts/diag_rotation_wave_v5.py --mode reach
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
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.models import rotation as R  # noqa: E402
from cbb_sim.models import rotation_v4 as V4  # noqa: E402

import diag_rotation_sub_hazard as D  # noqa: E402

OUT_DIR = ROOT / "data" / "processed" / "models" / "rotation"
OUT_JSON = OUT_DIR / "wave_audit_2026-09-11.json"
REACH_JSON = OUT_DIR / "wave_reachability_2026-09-11.json"

UNDERPOWERED = 300

#: engine order -- `engine.state.PREV_END_LEVELS`
PE = list(V4.PREV_END_LEVELS)
N_PE, N_TC, N_MB, N_FS = len(PE), 9, 3, 2
N_CELL = N_PE * N_TC * N_MB * N_FS
MAX_WAVE = 5


def cell_index(pe: np.ndarray, tc: np.ndarray, mb: np.ndarray, fs: np.ndarray) -> np.ndarray:
    return ((pe * N_TC + tc) * N_MB + mb) * N_FS + fs


class Tab:
    """events / trials by integer key."""

    def __init__(self, size: int):
        self.ev = np.zeros(size, dtype="float64")
        self.n = np.zeros(size, dtype="float64")

    def add(self, key, y) -> None:
        k = np.asarray(key, dtype="int64")
        if k.size == 0:
            return
        self.ev += np.bincount(k, weights=np.asarray(y, dtype="float64"),
                               minlength=len(self.ev))
        self.n += np.bincount(k, minlength=len(self.n))

    def out(self) -> dict:
        return {"ev": self.ev.tolist(), "n": self.n.tolist()}


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def sweep(season: int, max_games: int = 0) -> dict:
    t0 = time.time()
    tp = R.load_team_possessions(season)
    side = D.load_side_state(season)
    tp = tp.merge(side, on=["game_id", "poss_index", "is_home"], how="left")
    tp["reason"] = tp["reason"].fillna("other")
    gids = np.sort(tp["game_id"].unique())
    if max_games:
        gids = gids[:max_games]
        tp = tp[tp["game_id"].isin(set(int(g) for g in gids))]

    gu = pd.read_parquet(ROOT / "data" / "processed" / "games_universe.parquet")
    gu = gu[gu["season"] == season][["game_id", "cbbd_game_id"]]
    cbbd_of = dict(zip(gu["game_id"].to_numpy().astype("int64"),
                       gu["cbbd_game_id"].to_numpy().astype("int64")))
    tos = D.load_timeouts(season)

    pg = R.player_game_minutes(tp)
    ev = R.player_game_fouls(season).merge(gu, on="cbbd_game_id", how="inner")
    ev = ev.merge(pg[["game_id", "team_id", "pid"]].drop_duplicates(),
                  on=["game_id", "pid"], how="inner")
    evg = {k: g for k, g in ev.groupby(["game_id", "team_id"], sort=False)}

    A = {
        "cell": Tab(N_CELL),                 # P(wave) on the fitting cell
        "pe": Tab(N_PE),
        "period": Tab(4),                    # 1, 2, OT, (unused)
        "tc": Tab(N_TC),
        "mb": Tab(N_MB),
        "fs": Tab(N_FS),                     # any on-floor player with >=4 fouls
        "tf": Tab(5),                        # own team fouls 0-2/3-5/6-8/9-11/12+
        "timeout": Tab(2),
        "pe_x_to": Tab(N_PE * 2),
    }
    # size histogram: [cell][size-1] and the marginals
    size_cell = np.zeros((N_CELL, MAX_WAVE), dtype="float64")
    size_pe = np.zeros((N_PE, MAX_WAVE), dtype="float64")
    size_tc = np.zeros((N_TC, MAX_WAVE), dtype="float64")
    size_to = np.zeros((2, MAX_WAVE), dtype="float64")
    # composition: [size-1][starters_out] and [size-1][starters_in]
    comp_out = np.zeros((MAX_WAVE, MAX_WAVE + 1), dtype="float64")
    comp_in = np.zeros((MAX_WAVE, MAX_WAVE + 1), dtype="float64")
    # leaver / entrant profile by starter flag: stint (min), fouls
    prof = {"leave_stint": Tab(2), "leave_fouls": Tab(2), "stay_stint": Tab(2),
            "stay_fouls": Tab(2), "enter_rest": Tab(2), "sit_rest": Tab(2)}
    # cross-team: 2x2 table of (home wave, away wave) at the same boundary,
    # overall and split period_start / not
    cross = np.zeros((2, 2, 2), dtype="float64")     # [is_period_start][h][a]
    cross_size = np.zeros((2, MAX_WAVE + 1, MAX_WAVE + 1), dtype="float64")
    # per-team-game totals for the two round-5 gate cells
    per_tg = []
    # wave rate by boundary ordinal within the game (first 3 after a period start)
    n_tg = 0
    n_bnd = 0
    by_game: dict[int, dict] = {}

    for (gid, tid), g in tp.groupby(["game_id", "team_id"], sort=False):
        gid, tid = int(gid), int(tid)
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        npos = lu.shape[0]
        if npos < 20:
            continue
        dur = g["duration_s"].to_numpy(dtype="float64")
        period = g["period"].to_numpy(dtype="int64")
        clock = g["start_clock"].to_numpy(dtype="int64")
        margin = g["margin"].to_numpy(dtype="int64")
        reason = g["reason"].to_numpy()
        tfouls = np.nan_to_num(g["own_team_fouls"].to_numpy(dtype="float64"))
        poss_idx = g["poss_index"].to_numpy(dtype="int64")
        is_home = bool(g["is_home"].to_numpy()[0])
        tc = D.time_cell9(period, clock)
        mb = R.margin_bucket(margin)
        pe = np.array([V4.PREV_END_CODE.get(r, V4.PREV_END_CODE["other"]) for r in reason],
                      dtype="int64")

        pids = np.unique(lu)
        nc = len(pids)
        on = np.zeros((npos, nc), dtype=bool)
        for s in range(5):
            on[np.arange(npos), np.searchsorted(pids, lu[:, s])] = True
        starters = np.zeros(nc, dtype=bool)
        starters[np.searchsorted(pids, np.unique(lu[0]))] = True

        fm = R.actual_foul_matrix(evg.get((gid, tid)), period, clock,
                                  {int(p): i for i, p in enumerate(pids)}).astype("int64")
        if fm.shape[0] < npos:
            fm = np.vstack([fm, np.repeat(fm[-1:], npos - fm.shape[0], axis=0)])

        # run length in the current on/off state at the START of k (changes up
        # to k-1 only -- `chg[k]` IS the label, using it would copy the answer)
        cum = np.concatenate([[0.0], np.cumsum(dur)])
        chg = np.empty_like(on)
        chg[0] = True
        chg[1:] = on[1:] != on[:-1]
        idx = np.arange(npos)[:, None]
        run_start = np.maximum.accumulate(np.where(chg, idx, -1), axis=0)
        run_prev = np.vstack([np.zeros((1, nc), dtype=run_start.dtype), run_start[:-1]])
        run_min = (cum[np.arange(npos)][:, None] - cum[np.clip(run_prev, 0, None)]) / 60.0

        to_flag = np.zeros(npos, dtype=bool)
        cg = cbbd_of.get(gid)
        if cg is not None and tos:
            for k in range(npos):
                if (cg, int(period[k]), int(clock[k])) in tos:
                    to_flag[k] = True

        # ---- the boundary arrays (k = 1..npos-1) -------------------------
        leave = on[:-1] & ~on[1:]                 # (npos-1, nc)
        enter = ~on[:-1] & on[1:]
        size = leave.sum(axis=1)
        wave = (size > 0).astype("float64")
        k = np.arange(1, npos)
        # "foul state": any player on the floor at k-1 carrying >= 4 fouls
        fs = ((fm[k] >= 4) & on[:-1]).any(axis=1).astype("int64")
        tfb = np.clip((tfouls[k] // 3).astype("int64"), 0, 4)
        ci = cell_index(pe[k], tc[k], mb[k], fs)

        A["cell"].add(ci, wave)
        A["pe"].add(pe[k], wave)
        A["period"].add(np.clip(period[k] - 1, 0, 2), wave)
        A["tc"].add(tc[k], wave)
        A["mb"].add(mb[k], wave)
        A["fs"].add(fs, wave)
        A["tf"].add(tfb, wave)
        A["timeout"].add(to_flag[k].astype("int64"), wave)
        A["pe_x_to"].add(pe[k] * 2 + to_flag[k].astype("int64"), wave)
        n_bnd += len(k)

        w = np.flatnonzero(size > 0)
        if len(w):
            sz = np.clip(size[w], 1, MAX_WAVE) - 1
            np.add.at(size_cell, (ci[w], sz), 1.0)
            np.add.at(size_pe, (pe[k][w], sz), 1.0)
            np.add.at(size_tc, (tc[k][w], sz), 1.0)
            np.add.at(size_to, (to_flag[k][w].astype("int64"), sz), 1.0)
            so = (leave[w] & starters[None, :]).sum(axis=1)
            si = (enter[w] & starters[None, :]).sum(axis=1)
            np.add.at(comp_out, (sz, np.clip(so, 0, MAX_WAVE)), 1.0)
            np.add.at(comp_in, (sz, np.clip(si, 0, MAX_WAVE)), 1.0)

        # leaver / stayer profile at boundaries where a wave happened
        if len(w):
            lw = leave[w]
            stay = on[:-1][w] & ~lw
            ew = enter[w]
            sit = ~on[:-1][w] & ~ew & (fm[k][w] < R.FOUL_OUT)
            rm = run_min[k][w]
            fmw = fm[k][w].astype("float64")
            stf = np.broadcast_to(starters[None, :], lw.shape)
            for tab, msk, val in (("leave_stint", lw, rm), ("leave_fouls", lw, fmw),
                                  ("stay_stint", stay, rm), ("stay_fouls", stay, fmw),
                                  ("enter_rest", ew, rm), ("sit_rest", sit, rm)):
                sel = np.flatnonzero(msk.ravel())
                if len(sel):
                    prof[tab].add(stf.ravel()[sel].astype("int64"), val.ravel()[sel])

        # per-team-game gate quantities
        lu_keys = [tuple(sorted(int(x) for x in row)) for row in lu]
        distinct = len(set(lu_keys))
        per_tg.append((gid, tid, float(wave.mean()), float(distinct),
                       float(size[size > 0].mean() if (size > 0).any() else 0.0),
                       int(npos)))

        by_game.setdefault(gid, {})[is_home] = {
            "poss": poss_idx[1:], "wave": wave, "size": size,
            "pe": pe[k], "to": to_flag[k],
        }
        n_tg += 1

    # ---- cross-team correlation ------------------------------------------
    for gid, sides in by_game.items():
        if True not in sides or False not in sides:
            continue
        h, a = sides[True], sides[False]
        common, ih, ia = np.intersect1d(h["poss"], a["poss"], return_indices=True)
        if not len(common):
            continue
        wh = h["wave"][ih].astype("int64")
        wa = a["wave"][ia].astype("int64")
        ps = (h["pe"][ih] == V4.PREV_END_CODE["period_start"]).astype("int64")
        np.add.at(cross, (ps, wh, wa), 1.0)
        np.add.at(cross_size, (ps, np.clip(h["size"][ih], 0, MAX_WAVE),
                               np.clip(a["size"][ia], 0, MAX_WAVE)), 1.0)

    ptg = pd.DataFrame(per_tg, columns=["game_id", "team_id", "sub_rate",
                                        "distinct_lineups", "mean_wave_size", "npos"])
    out = {
        "season": season,
        "team_games": n_tg,
        "boundaries": int(n_bnd),
        "seconds": round(time.time() - t0, 1),
        "acc": {k: v.out() for k, v in A.items()},
        "prof": {k: v.out() for k, v in prof.items()},
        "size_cell": size_cell.tolist(),
        "size_pe": size_pe.tolist(),
        "size_tc": size_tc.tolist(),
        "size_to": size_to.tolist(),
        "comp_out": comp_out.tolist(),
        "comp_in": comp_in.tolist(),
        "cross": cross.tolist(),
        "cross_size": cross_size.tolist(),
        "gate": {
            "sub_rate_per_boundary": float(A["cell"].ev.sum() / max(A["cell"].n.sum(), 1)),
            "distinct_lineups_mean": float(ptg["distinct_lineups"].mean()),
            "distinct_lineups_sd": float(ptg["distinct_lineups"].std(ddof=1)),
            "mean_wave_size": float(np.average(np.arange(1, MAX_WAVE + 1),
                                               weights=size_pe.sum(axis=0))),
            "subs_per_team_game": float((size_pe.sum(axis=0)
                                         * np.arange(1, MAX_WAVE + 1)).sum() / max(n_tg, 1)),
            "timeout_rows": int(len(tos)),
        },
    }
    return out


# ---------------------------------------------------------------------------
# the wave tables (the object round 5 fits; here built from the audit)
# ---------------------------------------------------------------------------
def build_wave_tables(acc_cell: dict, size_cell: np.ndarray, k_shrink: float = 300.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(p_wave[cell], p_size[cell, 5]) with two-level shrinkage.

    Parent 1 is (prev_end x time cell) collapsed over margin band and foul
    state; parent 2 is (prev_end). `k_shrink` is the UNDERPOWERED threshold
    (300), fixed, never tuned."""
    ev = np.asarray(acc_cell["ev"], dtype="float64")
    n = np.asarray(acc_cell["n"], dtype="float64")
    ev3 = ev.reshape(N_PE, N_TC, N_MB, N_FS)
    n3 = n.reshape(N_PE, N_TC, N_MB, N_FS)
    p_root = ev.sum() / max(n.sum(), 1.0)
    ev2 = ev3.sum(axis=(2, 3))
    n2 = n3.sum(axis=(2, 3))
    ev1 = ev3.sum(axis=(1, 2, 3))
    n1 = n3.sum(axis=(1, 2, 3))
    p1 = (ev1 + k_shrink * p_root) / (n1 + k_shrink)                 # (PE,)
    p2 = (ev2 + k_shrink * p1[:, None]) / (n2 + k_shrink)            # (PE, TC)
    p3 = (ev3 + k_shrink * p2[:, :, None, None]) / (n3 + k_shrink)   # full

    s3 = size_cell.reshape(N_PE, N_TC, N_MB, N_FS, MAX_WAVE)
    s_root = size_cell.sum(axis=0)
    s_root = s_root / max(s_root.sum(), 1.0)
    s2 = s3.sum(axis=(2, 3))
    s1 = s3.sum(axis=(1, 2, 3))
    q1 = (s1 + k_shrink * s_root[None, :]) / (s1.sum(axis=1, keepdims=True) + k_shrink)
    q2 = (s2 + k_shrink * q1[:, None, :]) / (s2.sum(axis=2, keepdims=True) + k_shrink)
    q3 = (s3 + k_shrink * q2[:, :, None, None, :]) / (
        s3.sum(axis=4, keepdims=True) + k_shrink)
    q3 = q3 / q3.sum(axis=4, keepdims=True)
    return p3.reshape(N_CELL), q3.reshape(N_CELL, MAX_WAVE)


# ---------------------------------------------------------------------------
# the L25 reachability probe -- on paper, BEFORE anything is fitted
# ---------------------------------------------------------------------------
def reach(train_season: int, test_season: int, n_games: int, seed: int = 7,
          audit: dict | None = None, comp: str = "rank_rank") -> dict:
    """Run the WAVE sampler -- audit wave probability and size, round-4 fitted
    per-player hazards as the composition rule -- on real 2025 game scripts with
    the REAL starting fives and REAL participant pools, and read the round-4
    gate cells plus the two round-5 cells.

    This is not a bake-off arm: no as-of starter set, no fitted wave model of
    its own, graded against nothing. It answers one question -- CAN the joint
    family reach the cells? -- before the round-5 pre-registration is written.
    """
    if audit is None:
        audit = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    tr = audit[str(train_season)]
    p_wave, p_size = build_wave_tables(tr["acc"]["cell"], np.array(tr["size_cell"]))

    sub = V4.SubHazardFit.from_json(OUT_DIR / "rotation_v4_sub_static.json")

    tp = R.load_team_possessions(test_season)
    side = D.load_side_state(test_season)
    tp = tp.merge(side, on=["game_id", "poss_index", "is_home"], how="left")
    tp["reason"] = tp["reason"].fillna("other")

    # as-of share (strictly earlier games), the column round-4's hazards were
    # fitted on. Built over the WHOLE season so the as-of panel is right, then
    # restricted to the probe's games.
    pg = R.player_game_minutes(tp)
    gu = pd.read_parquet(ROOT / "data" / "processed" / "games_universe.parquet")
    gu = gu[gu["season"] == test_season][["game_id", "cbbd_game_id"]]
    evf = R.player_game_fouls(test_season).merge(gu, on="cbbd_game_id", how="inner")
    evf = evf.merge(pg[["game_id", "team_id", "pid"]].drop_duplicates(),
                    on=["game_id", "pid"], how="inner")
    feats = R.build_asof_player_features(pg, fouls=evf)
    fs = feats[["game_id", "team_id", "pid", "mpg_asof_raw"]].copy()
    tt = fs.groupby(["game_id", "team_id"])["mpg_asof_raw"].transform("sum")
    fs["sh"] = np.where(tt > 0, fs["mpg_asof_raw"] / tt * 5.0, 0.0)
    share_asof = {(int(g), int(t), int(p)): float(s)
                  for g, t, p, s in zip(fs["game_id"], fs["team_id"], fs["pid"], fs["sh"])}
    evg = {k: g for k, g in evf.groupby(["game_id", "team_id"], sort=False)}

    gids = np.sort(tp["game_id"].unique())
    rs = np.random.RandomState(2025)
    gids = gids[rs.choice(len(gids), min(n_games, len(gids)), replace=False)]
    tp = tp[tp["game_id"].isin(set(int(g) for g in gids))]
    rng = np.random.default_rng(seed)

    def blank():
        return {"h2_tip": np.zeros((3, 2)), "late": np.zeros((3, 2)),
                "open_close": np.zeros(2), "foul4": np.zeros(2),
                "profile": np.zeros((9, 3, 2)), "subs": np.zeros(2),
                "distinct": [], "size_hist": np.zeros(MAX_WAVE)}

    simc, actc = blank(), blank()
    for (gid, tid), g in tp.groupby(["game_id", "team_id"], sort=False):
        gid, tid = int(gid), int(tid)
        lu = g[R.SLOTS].to_numpy(dtype="int64")
        npos = lu.shape[0]
        if npos < 20:
            continue
        dur = g["duration_s"].to_numpy(dtype="float64")
        period = g["period"].to_numpy(dtype="int64")
        clock = g["start_clock"].to_numpy(dtype="int64")
        margin = g["margin"].to_numpy(dtype="int64")
        tfouls = np.nan_to_num(g["own_team_fouls"].to_numpy(dtype="float64"))
        reason = g["reason"].to_numpy()
        tc = D.time_cell9(period, clock)
        mb = R.margin_bucket(margin)
        pe = np.array([V4.PREV_END_CODE.get(r, V4.PREV_END_CODE["other"]) for r in reason],
                      dtype="int64")
        pids = np.unique(lu)
        nc = len(pids)
        if nc < 6:
            continue
        st = np.zeros(nc, dtype=bool)
        st[np.searchsorted(pids, np.unique(lu[0]))] = True
        fm = R.actual_foul_matrix(evg.get((gid, tid)), period, clock,
                                  {int(p): i for i, p in enumerate(pids)}).astype("int64")
        if fm.shape[0] < npos:
            fm = np.vstack([fm, np.repeat(fm[-1:], npos - fm.shape[0], axis=0)])
        share = np.array([share_asof.get((gid, tid, int(p)), 0.0) for p in pids])
        if share.sum() <= 0:
            share = np.full(nc, 5.0 / nc)
        share = share / share.sum() * 5.0
        stf = st.astype(np.float64)

        onmask = st.copy()
        state_min = np.zeros(nc)
        half_min = np.zeros(nc)
        prev_half = 0
        sim_share = np.zeros(npos)
        sim_lu = []
        n_wave = 0
        for k in range(npos):
            cur_half = 0 if period[k] == 1 else 1
            if cur_half != prev_half:
                half_min[:] = 0.0
                prev_half = cur_half
            prev_on = onmask.copy()
            if k > 0:
                elig = fm[k] < R.FOUL_OUT
                on_idx = np.flatnonzero(onmask)
                bench = np.flatnonzero(~onmask & elig)
                if len(on_idx) == 5 and len(bench):
                    fsq = int(((fm[k] >= 4) & onmask).any())
                    ci = int(cell_index(np.array([pe[k]]), np.array([tc[k]]),
                                        np.array([mb[k]]), np.array([fsq]))[0])
                    forced = int((fm[k][on_idx] >= R.FOUL_OUT).sum())
                    if rng.random() < p_wave[ci] or forced:
                        sz = int(rng.choice(MAX_WAVE, p=p_size[ci]) + 1)
                        sz = max(sz, forced)
                        sz = int(min(sz, len(bench), 5))
                        X = V4.design(stf[None, :], share[None, :],
                                      fm[k].astype(np.float64)[None, :],
                                      state_min[None, :], half_min[None, :],
                                      np.array([period[k]]), np.array([clock[k]]),
                                      np.array([margin[k]]), np.array([pe[k]]),
                                      np.array([tfouls[k]]))[0]
                        po = sub.p_out(X[on_idx])
                        po = np.where(fm[k][on_idx] >= R.FOUL_OUT, 1e9, po)
                        pi = np.clip(sub.p_in(X[bench]), 1e-9, 1 - 1e-9)
                        ex_mode, en_mode = comp.split("_")
                        if ex_mode == "rank":
                            leaving = on_idx[np.argsort(-po, kind="stable")[:sz]]
                        else:
                            wo = np.clip(np.minimum(po, 1 - 1e-9), 1e-9, 1 - 1e-9)
                            wo = np.where(po >= 1e8, 1e12, wo / (1 - wo))
                            ko = -np.log(np.clip(rng.random(5), 1e-12, 1.0)) / wo
                            leaving = on_idx[np.argsort(ko, kind="stable")[:sz]]
                        if en_mode == "rank":
                            entering = bench[np.argsort(-pi, kind="stable")[:sz]]
                        else:
                            wi = pi / (1 - pi)
                            ki = -np.log(np.clip(rng.random(len(bench)), 1e-12, 1.0)) / wi
                            entering = bench[np.argsort(ki, kind="stable")[:sz]]
                        onmask[leaving] = False
                        onmask[entering] = True
                        n_wave += 1
                        simc["size_hist"][min(sz, MAX_WAVE) - 1] += 1
            cur = np.flatnonzero(onmask)
            if len(cur) != 5:
                sc = np.where(fm[k] < R.FOUL_OUT, share, -1e12)
                sc[cur] += 10.0
                cur = np.argsort(-sc)[:5]
                onmask[:] = False
                onmask[cur] = True
            sim_share[k] = float(onmask[st].sum())
            sim_lu.append(tuple(sorted(int(x) for x in np.flatnonzero(onmask))))
            state_min[onmask != prev_on] = 0.0
            state_min += dur[k] / 60.0
            half_min[onmask] += dur[k] / 60.0

        a_share = np.array([float(np.isin(pids[st], lu[k]).sum()) for k in range(npos)])
        a_on = np.zeros((npos, nc), dtype=bool)
        for s in range(5):
            a_on[np.arange(npos), np.searchsorted(pids, lu[:, s])] = True
        a_wave = int(((a_on[1:] != a_on[:-1]).any(axis=1)).sum())
        a_lu = [tuple(sorted(int(x) for x in row)) for row in
                np.argsort(~a_on, kind="stable")[:, :5]]

        for arr, sh, nw, lus in ((simc, sim_share, n_wave, sim_lu),
                                 (actc, a_share, a_wave, a_lu)):
            h2 = np.flatnonzero(period == 2)
            if len(h2):
                kk = int(h2[0])
                arr["h2_tip"][mb[kk], 0] += sh[kk]
                arr["h2_tip"][mb[kk], 1] += 5.0
            late = np.isin(tc, (5, 6, 7))
            for b in range(3):
                m = late & (mb == b)
                arr["late"][b, 0] += sh[m].sum()
                arr["late"][b, 1] += 5.0 * m.sum()
            oc = (tc == 0) & (mb == 0)
            arr["open_close"][0] += sh[oc].sum()
            arr["open_close"][1] += 5.0 * oc.sum()
            arr["subs"][0] += nw
            arr["subs"][1] += npos - 1
            arr["distinct"].append(len(set(lus)))
            for t in range(9):
                for b in range(3):
                    m = (tc == t) & (mb == b)
                    if m.any():
                        arr["profile"][t, b, 0] += sh[m].sum()
                        arr["profile"][t, b, 1] += 5.0 * m.sum()

    out = {"train_season": train_season, "test_season": test_season,
           "n_games": int(len(gids)), "composition": comp}
    for tag, d in (("sim", simc), ("actual", actc)):
        out[tag] = {
            "h2_tip": (d["h2_tip"][:, 0] / np.maximum(d["h2_tip"][:, 1], 1)).tolist(),
            "late": (d["late"][:, 0] / np.maximum(d["late"][:, 1], 1)).tolist(),
            "open_close": float(d["open_close"][0] / max(d["open_close"][1], 1)),
            "sub_rate": float(d["subs"][0] / max(d["subs"][1], 1)),
            "distinct_lineups": float(np.mean(d["distinct"])),
            "profile": (d["profile"][:, :, 0] / np.maximum(d["profile"][:, :, 1], 1)).tolist(),
        }
    out["sim_size_hist"] = (simc["size_hist"] / max(simc["size_hist"].sum(), 1)).tolist()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sweep", choices=["sweep", "reach"])
    ap.add_argument("--seasons", default="2024,2025")
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--reach-games", type=int, default=400)
    ap.add_argument("--comp", default="rank_rank,draw_draw,draw_rank,rank_draw",
                    help="composition rules to probe: <exit>_<entry>, rank|draw")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.mode == "reach":
        res = {}
        for comp in a.comp.split(","):
            r = reach(2024, 2025, a.reach_games, comp=comp)
            res[comp] = r
            print(comp,
                  "sub_rate sim %.4f act %.4f | distinct sim %.2f act %.2f"
                  % (r["sim"]["sub_rate"], r["actual"]["sub_rate"],
                     r["sim"]["distinct_lineups"], r["actual"]["distinct_lineups"]))
            print("  h2_tip sim", np.round(r["sim"]["h2_tip"], 4).tolist(),
                  "act", np.round(r["actual"]["h2_tip"], 4).tolist())
            print("  late   sim", np.round(r["sim"]["late"], 4).tolist(),
                  "act", np.round(r["actual"]["late"], 4).tolist())
            print("  open   sim %.4f act %.4f" % (r["sim"]["open_close"],
                                                  r["actual"]["open_close"]))
        REACH_JSON.write_text(json.dumps(res), encoding="utf-8")
        print("wrote", REACH_JSON)
        return 0
    res = {}
    for s in [int(x) for x in a.seasons.split(",")]:
        print(f"[wave sweep] season {s}", flush=True)
        res[str(s)] = sweep(s, a.games)
        print(f"  {res[str(s)]['team_games']} team-games, "
              f"{res[str(s)]['boundaries']} boundaries in {res[str(s)]['seconds']}s "
              f"| sub rate {res[str(s)]['gate']['sub_rate_per_boundary']:.4f} "
              f"| distinct {res[str(s)]['gate']['distinct_lineups_mean']:.2f}", flush=True)
    OUT_JSON.write_text(json.dumps(res), encoding="utf-8")
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
