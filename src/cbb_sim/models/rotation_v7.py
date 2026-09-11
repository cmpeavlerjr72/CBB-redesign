"""
rotation_v7.py -- ROUND 7 of the L4 rotation bake-off: the EXIT side,
`P(k_out | size, state)`, K1's mirror.

Why round 7 exists
------------------
Round 6 moved the entry side and measured where the error then lived
(`docs/tests/rotation_composition_audit_2026-09-11.md` section 4, 200 games,
seed 0, the same as-of predicted starter set on both sides):

    ACTUAL      P(starter in | bench out) 0.6848  | starter out 0.3446
    W4          0.5415 / 0.3373      share of single swaps taking a STARTER off 0.464
    A1          0.5683 / 0.3117                                              0.467
    K1          0.6344 / 0.2026                                              0.456
    ACTUAL                                                                   0.587

Every arm since round 5 removes a starter on 0.456-0.467 of single swaps
against a real 0.587, because rounds 5 and 6 both pre-registered round 5's RANK
exit rule. K1 conditions correctly on a class that ARRIVES WITH THE WRONG
FREQUENCY: best per-player minutes MAE in six rounds (8.8622) and no state cell
moved (experiments.md 15.12 item 3).

Round 7 keeps round 5's wave tables, round 4's hazards, round 3b's base fits,
round 6's K1 entry rule and the hard second-half reset BYTE FOR BYTE, and
changes only how the `size` leavers are chosen off the floor:

    X1  `exit_class`       P(k_out | size, exit_cell) drawn first, then the
                           leavers picked by round 5's RANK rule within class
    X2  `exit_class_foul`  X1 with the binary foul_state axis replaced by a
                           three-level foul class (nobody / a non-starter / a
                           STARTER at >= 4 personal fouls), shrunk to X1
    X3  `exit_class_prev`  X1 with the previous wave-of-this-half's entry class
                           as an extra axis, shrunk to X1

Pre-registration: `docs/models/rotation/experiments.md` section 16, written and
committed (2aa29c2) before any object here was fitted.

Engine expressibility
---------------------
All three are lookup tables ((5,18,6), (5,9,3,6), (5,18,3,6)) and all three are
array operations over the (2N, S) roster block, so the sim loop makes no model
call (`CLAUDE.md`). The state terms they carry are exactly the ones section 16.2
declares -- a 3-level time cell, a margin band, a foul state -- and no others.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.models.rotation import (
    FOUL_OUT,
    SLOTS,
    GameScript,
    RotationFit,
    TeamPrior,
    actual_foul_matrix,
    build_priors,
    build_scripts,
    draw_available,
    margin_bucket,
)
from cbb_sim.models.rotation_v4 import (
    SubHazardFit,
    design,
    time_cell,
    _side_arrays,
    _side_state_index,
)
from cbb_sim.models.rotation_v5 import MAX_WAVE, N_FS, N_MB, WaveFit, _WaveArm, wave_cell
from cbb_sim.models.rotation_v6 import CompFit, _race, tier_of

#: the declared 3-level exit time cell (16.2): a coarsening of round 4's nine
#: audit time cells into H1 / H2 20:00-08:00 / final 8:00. Overtime joins the
#: final-8:00 cell because an OT period never has more than 8:00 left, which is
#: the audit's own definition (`per >= 2 and clock <= 480`).
EXIT_TC = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
N_ETC = 3
N_EXIT_CELL = N_ETC * N_MB * N_FS          # 18
N_FCLASS = 3                                # none / non-starter / STARTER at >= 4
N_PCLASS = 3                                # no previous wave / prev k_in = 0 / >= 1
FOUL_TROUBLE = 4                            # round 5's own foul_state boundary

#: the project's UNDERPOWERED threshold, the constant rounds 5 and 6 declared.
K_SHRINK = 300.0


def exit_cell(period: np.ndarray, sec_left: np.ndarray, margin: np.ndarray,
              foul_state: np.ndarray) -> np.ndarray:
    """(M,) exit-state index, `(tc * 3 + mb) * 2 + fs`, 18 cells (16.2). The
    SAME function offline and in the engine adapter."""
    tc = EXIT_TC[time_cell(np.asarray(period, dtype=np.int64), np.asarray(sec_left))]
    mb = margin_bucket(np.asarray(margin))
    fs = np.asarray(foul_state, dtype=np.int64)
    return (tc * N_MB + mb) * N_FS + fs


def foul_class(fouls_on: np.ndarray, is_st_on: np.ndarray) -> int:
    """0 nobody on the floor at >= 4 PF, 1 somebody but no predicted starter,
    2 a predicted STARTER at >= 4 PF (16.3)."""
    trouble = fouls_on >= FOUL_TROUBLE
    if not trouble.any():
        return 0
    return 2 if bool((trouble & (is_st_on > 0.5)).any()) else 1


# ===========================================================================
# 1. The fitted object
# ===========================================================================
@dataclass
class ExitFit:
    """The three round-7 exit objects. One artifact carries all three so the
    arms are fitted on identical rows and cannot drift apart."""

    p_x1: list = field(default_factory=list)   # (MAX_WAVE, N_EXIT_CELL, MAX_WAVE+1)
    p_x2: list = field(default_factory=list)   # (MAX_WAVE, N_ETC*N_MB, N_FCLASS, MW+1)
    p_x3: list = field(default_factory=list)   # (MAX_WAVE, N_EXIT_CELL, N_PCLASS, MW+1)
    n_x1: list = field(default_factory=list)
    n_x2: list = field(default_factory=list)
    n_x3: list = field(default_factory=list)
    k_shrink: float = K_SHRINK
    n_waves: int = 0
    n_pairs: int = 0
    wave_source: str = ""
    comp_source: str = ""
    hazard_source: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def X1(self) -> np.ndarray:
        return np.asarray(self.p_x1, dtype=np.float64).reshape(
            MAX_WAVE, N_EXIT_CELL, MAX_WAVE + 1)

    @property
    def X2(self) -> np.ndarray:
        return np.asarray(self.p_x2, dtype=np.float64).reshape(
            MAX_WAVE, N_ETC * N_MB, N_FCLASS, MAX_WAVE + 1)

    @property
    def X3(self) -> np.ndarray:
        return np.asarray(self.p_x3, dtype=np.float64).reshape(
            MAX_WAVE, N_EXIT_CELL, N_PCLASS, MAX_WAVE + 1)

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "ExitFit":
        return ExitFit(**json.loads(Path(path).read_text(encoding="utf-8")))


# ===========================================================================
# 2. Training counts -- ONE pass, all three objects, round 6's own rows
# ===========================================================================
def build_exit_training(tp: pd.DataFrame, feats: pd.DataFrame, fit: RotationFit,
                        game_ids, fouls: pd.DataFrame | None = None,
                        max_team_games: int = 0, seed: int = 11) -> dict:
    """Counts for `P(k_out | size, state)` over the training window's own wave
    boundaries.

    The usable-wave filter is round 6's, byte for byte
    (`rotation_v6.build_comp_training`): equal leaver and entrant counts, five
    men on the floor, a non-empty as-of bench pool and every entrant inside it.
    Rows are built over the **as-of candidate pool** (`build_priors`), never the
    game's own participant list, so the pool at fit time is the pool at
    simulation time, and the sample (`max_team_games`, `seed`) is round 6's, so
    the exit and entry tables are fitted on the same rows (16.5).
    """
    gset = set(int(g) for g in game_ids)
    sel = tp[tp["game_id"].isin(gset)]
    scripts = build_scripts(sel)
    priors = build_priors(feats[feats["game_id"].isin(gset)], fit)
    actual = {(int(g), int(t)): d[SLOTS].to_numpy(dtype="int64")
              for (g, t), d in sel.groupby(["game_id", "team_id"], sort=False)}
    keys = [k for k in actual if k in priors and k in scripts]
    if max_team_games and len(keys) > max_team_games:
        rs = np.random.RandomState(seed)
        keys = [keys[i] for i in rs.choice(len(keys), max_team_games, replace=False)]

    evg = {}
    if fouls is not None and len(fouls):
        f = fouls[fouls["game_id"].isin(gset)]
        evg = {(int(g), int(t)): d for (g, t), d in f.groupby(["game_id", "team_id"],
                                                              sort=False)}

    c1 = np.zeros((MAX_WAVE, N_EXIT_CELL, MAX_WAVE + 1))
    c2 = np.zeros((MAX_WAVE, N_ETC * N_MB, N_FCLASS, MAX_WAVE + 1))
    c3 = np.zeros((MAX_WAVE, N_EXIT_CELL, N_PCLASS, MAX_WAVE + 1))
    n_waves = n_used = n_tg = 0
    for key in keys:
        lu = actual[key]
        prior, script = priors[key], scripts[key]
        idx_of = {int(p): i for i, p in enumerate(prior.pids)}
        n = prior.n
        npos = min(len(lu), script.n)
        if npos < 20:
            continue
        n_tg += 1
        fm = actual_foul_matrix(evg.get(key), script.period, script.start_clock, idx_of)

        on = np.zeros((npos, n), dtype=bool)
        for k in range(npos):
            for p in lu[k]:
                j = idx_of.get(int(p))
                if j is not None:
                    on[k, j] = True

        is_st = np.zeros(n, dtype=np.float64)
        is_st[prior.starters()] = 1.0
        prev_half, prev_class = 0, 0
        for k in range(1, npos):
            cur_half = 0 if script.period[k] == 1 else 1
            if cur_half != prev_half:
                prev_class = 0
                prev_half = cur_half
            prev_on = on[k - 1]
            leave = np.flatnonzero(prev_on & ~on[k])
            enter = np.flatnonzero(~prev_on & on[k])
            if not len(leave):
                continue
            n_waves += 1
            fouls_v = fm[k].astype("float64") if k < fm.shape[0] else np.zeros(n)
            pool = np.flatnonzero(~prev_on & (fouls_v < FOUL_OUT))
            if not (len(leave) == len(enter) and len(pool) and prev_on.sum() == 5
                    and set(enter) <= set(pool.tolist())):
                continue
            s = int(min(len(leave), MAX_WAVE))
            on_idx = np.flatnonzero(prev_on)
            fs = int((fouls_v[on_idx] >= FOUL_TROUBLE).any())
            ce = int(exit_cell(np.array([script.period[k]]),
                               np.array([script.start_clock[k]]),
                               np.array([script.margin[k]]), np.array([fs]))[0])
            tm = ce // N_FS
            fc = foul_class(fouls_v[on_idx], is_st[on_idx])
            k_out = int(min((is_st[leave] > 0.5).sum(), MAX_WAVE))
            k_in = int((is_st[enter] > 0.5).sum())
            c1[s - 1, ce, k_out] += 1.0
            c2[s - 1, tm, fc, k_out] += 1.0
            c3[s - 1, ce, prev_class, k_out] += 1.0
            prev_class = 1 if k_in == 0 else 2
            n_used += 1
    return {"c1": c1, "c2": c2, "c3": c3, "n_waves": n_waves, "n_used": n_used,
            "team_games": n_tg}


def _shrink(counts: np.ndarray, parent: np.ndarray, k: float, size: int) -> np.ndarray:
    """One-level shrinkage of a (k_out,) count row toward `parent` at `k`, with
    the support clipped to 0..size and renormalised."""
    row = (counts + k * parent) / (counts.sum() + k)
    row = row.copy()
    row[size + 1:] = 0.0
    z = row.sum()
    return row / z if z > 0 else np.eye(MAX_WAVE + 1)[0]


def fit_exit(counts: dict, k: float = K_SHRINK, wave_source: str = "",
             comp_source: str = "", hazard_source: str = "") -> ExitFit:
    """The three objects, from counts and a declared shrinkage constant. No
    optimiser, no grid search against any gate cell."""
    c1, c2, c3 = counts["c1"], counts["c2"], counts["c3"]
    p1 = np.zeros_like(c1)
    p2 = np.zeros_like(c2)
    p3 = np.zeros_like(c3)
    for s in range(MAX_WAVE):
        tot = c1[s].sum()
        marg = c1[s].sum(axis=0)
        marg = marg / tot if tot > 0 else np.eye(MAX_WAVE + 1)[0]
        for ce in range(N_EXIT_CELL):
            p1[s, ce] = _shrink(c1[s, ce], marg, k, s + 1)
        for tm in range(N_ETC * N_MB):
            for fc in range(N_FCLASS):
                # X2's parent is X1's own row for the same cell; foul class 0 is
                # foul_state 0 and classes 1 and 2 are foul_state 1 (16.3).
                fs = 0 if fc == 0 else 1
                p2[s, tm, fc] = _shrink(c2[s, tm, fc], p1[s, tm * N_FS + fs], k, s + 1)
        for ce in range(N_EXIT_CELL):
            for pc in range(N_PCLASS):
                p3[s, ce, pc] = _shrink(c3[s, ce, pc], p1[s, ce], k, s + 1)
    return ExitFit(p_x1=p1.tolist(), p_x2=p2.tolist(), p_x3=p3.tolist(),
                   n_x1=c1.sum(axis=2).tolist(), n_x2=c2.sum(axis=3).tolist(),
                   n_x3=c3.sum(axis=3).tolist(), k_shrink=float(k),
                   n_waves=int(counts["n_waves"]), n_pairs=int(counts["n_used"]),
                   wave_source=wave_source, comp_source=comp_source,
                   hazard_source=hazard_source,
                   notes={"team_games": int(counts["team_games"])})


# ===========================================================================
# 3. The sampler -- round 6's K1 with the EXIT rule replaced
# ===========================================================================
def run_wave7(prior: TeamPrior, script: GameScript, avail: np.ndarray, wf: WaveFit,
              cf: CompFit, xf: ExitFit, fit: RotationFit, rng: np.random.Generator,
              mode: str = "X1", hard_reset: bool = True,
              prev_end: np.ndarray | None = None,
              team_fouls: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """One team-game's on-floor sequence: round 5's wave draw, round 6's K1
    entry rule, and the round-7 EXIT rule named by `mode`.

    Every round-7 arm draws the same uniforms whether or not it uses them (the
    `k_out` uniform, then K1's `k_in` uniform, then one race vector over the
    bench), so the three arms sit at identical stream positions and are paired
    (16.12 item 1)."""
    sub = wf.hazards()
    pw, ps = wf.pw, wf.ps
    cum_size = np.cumsum(ps, axis=1)
    kin_tab = cf.kin
    x1, x2, x3 = xf.X1, xf.X2, xf.X3
    n = prior.n
    npos = script.n
    out = np.empty((npos, 5), dtype="int64")
    fouls_hist = np.zeros((npos, n), dtype="int8")
    fouls = np.zeros(n, dtype="int64")
    state_min = np.zeros(n)
    half_min = np.zeros(n)
    onmask = np.zeros(n, dtype=bool)
    order = [i for i in prior.start_order if avail[i]]
    start5 = np.asarray(order[:5] if len(order) >= 5 else list(np.flatnonzero(avail))[:5],
                        dtype="int64")
    onmask[start5] = True
    is_st = np.zeros(n, dtype=np.float64)
    is_st[prior.starters()] = 1.0
    share = prior.share / max(prior.share.sum(), 1e-9) * 5.0
    fscale = fit.foul_rate_scale
    if prev_end is None:
        from cbb_sim.models.rotation_v4 import PREV_END_CODE
        prev_end = np.where(np.concatenate([[True], np.diff(script.period) != 0]),
                            PREV_END_CODE["period_start"], PREV_END_CODE["DREB"])
    if team_fouls is None:
        team_fouls = np.zeros(npos)
    prev_half = 0
    prev_class = 0

    for k in range(npos):
        cur_half = 0 if script.period[k] == 1 else 1
        if cur_half != prev_half:
            half_min[:] = 0.0
            prev_class = 0
            prev_half = cur_half
        fouls_hist[k] = np.minimum(fouls, 127)
        eligible = avail & (fouls < FOUL_OUT)
        prev_onmask = onmask.copy()

        reset_now = (hard_reset and k > 0 and script.period[k] == 2
                     and script.period[k - 1] == 1)
        if reset_now and eligible.sum() >= 5:
            pick = np.asarray([i for i in prior.start_order if eligible[i]][:5],
                              dtype="int64")
            onmask[:] = False
            onmask[pick] = True
        elif k > 0:
            on_idx = np.flatnonzero(onmask)
            bench_idx = np.flatnonzero(~onmask & eligible)
            if len(on_idx) == 5 and len(bench_idx):
                fs = int(((fouls[on_idx] >= FOUL_TROUBLE).any()))
                ci = int(wave_cell(np.array([prev_end[k]]), np.array([script.period[k]]),
                                   np.array([script.start_clock[k]]),
                                   np.array([script.margin[k]]), np.array([fs]))[0])
                forced = int((fouls[on_idx] >= FOUL_OUT).sum())
                _u_c = rng.random()            # round 5's coupling draw, unused here
                u_w = rng.random()
                if u_w < pw[ci] or forced:
                    sz = int(np.searchsorted(cum_size[ci], rng.random()) + 1)
                    sz = int(min(max(sz, forced), len(bench_idx), 5))
                    u_x = rng.random()          # drawn by EVERY round-7 arm (16.12)
                    u_k = rng.random()          # round 6's k_in draw, unchanged
                    X = design(is_st[None, :], share[None, :],
                               fouls.astype(np.float64)[None, :], state_min[None, :],
                               half_min[None, :], np.array([script.period[k]]),
                               np.array([script.start_clock[k]]),
                               np.array([script.margin[k]]), np.array([prev_end[k]]),
                               np.array([team_fouls[k]]))[0]
                    po = sub.p_out(X[on_idx])
                    po = np.where(fouls[on_idx] >= FOUL_OUT, 1e9, po)
                    pi = np.clip(sub.p_in(X[bench_idx]), 1e-9, 1 - 1e-9)

                    # ---- the ROUND-7 exit rule ---------------------------
                    ce = int(exit_cell(np.array([script.period[k]]),
                                       np.array([script.start_clock[k]]),
                                       np.array([script.margin[k]]),
                                       np.array([fs]))[0])
                    st_o = np.flatnonzero(is_st[on_idx] > 0.5)
                    bn_o = np.flatnonzero(is_st[on_idx] <= 0.5)
                    forced_st = int((fouls[on_idx[st_o]] >= FOUL_OUT).sum()) if len(st_o) else 0
                    forced_bn = int((fouls[on_idx[bn_o]] >= FOUL_OUT).sum()) if len(bn_o) else 0
                    if mode == "X2":
                        fc = foul_class(fouls[on_idx].astype(np.float64), is_st[on_idx])
                        row = x2[sz - 1, ce // N_FS, fc].copy()
                    elif mode == "X3":
                        row = x3[sz - 1, ce, prev_class].copy()
                    else:
                        row = x1[sz - 1, ce].copy()
                    lo = max(0, sz - len(bn_o), forced_st)
                    hi = min(sz, len(st_o), sz - forced_bn)
                    if hi < lo:
                        hi = lo = int(min(max(lo, 0), sz))
                    row[:lo] = 0.0
                    row[hi + 1:] = 0.0
                    tot = row.sum()
                    if tot <= 0:
                        k_out = lo
                    else:
                        k_out = int(np.searchsorted(np.cumsum(row / tot), u_x))
                        k_out = int(min(max(k_out, lo), hi))
                    # within class, round 5's RANK rule, unchanged
                    take_so = st_o[np.argsort(-po[st_o], kind="stable")[:k_out]] \
                        if k_out else np.array([], dtype="int64")
                    take_bo = bn_o[np.argsort(-po[bn_o], kind="stable")[:sz - k_out]] \
                        if sz - k_out else np.array([], dtype="int64")
                    leaving = on_idx[np.concatenate([take_so, take_bo]).astype(int)]

                    # ---- round 6's K1 entry rule, byte for byte ----------
                    logw = np.log(pi) - np.log1p(-pi)
                    st_b = np.flatnonzero(is_st[bench_idx] > 0.5)
                    bn_b = np.flatnonzero(is_st[bench_idx] <= 0.5)
                    u_race = rng.random(len(bench_idx))
                    krow = kin_tab[sz - 1, min(k_out, MAX_WAVE)].copy()
                    klo = max(0, sz - len(bn_b))
                    khi = min(sz, len(st_b))
                    krow[:klo] = 0.0
                    krow[khi + 1:] = 0.0
                    ktot = krow.sum()
                    if ktot <= 0:
                        k_in = klo
                    else:
                        k_in = int(np.searchsorted(np.cumsum(krow / ktot), u_k))
                        k_in = int(min(max(k_in, klo), khi))
                    w = np.exp(logw - logw.max())
                    take_st = st_b[_race(w[st_b], u_race[st_b], k_in)] \
                        if k_in else np.array([], dtype="int64")
                    take_bn = bn_b[_race(w[bn_b], u_race[bn_b], sz - k_in)] \
                        if sz - k_in else np.array([], dtype="int64")
                    entering = bench_idx[np.concatenate([take_st, take_bn]).astype(int)]
                    onmask[leaving] = False
                    onmask[entering] = True
                    prev_class = 1 if k_in == 0 else 2

        cur = np.flatnonzero(onmask)
        if len(cur) != 5:
            score = np.where(eligible, share, -1e12)
            score[cur] += 10.0
            cur = np.argsort(-score)[:5]
            onmask[:] = False
            onmask[cur] = True
        bad = cur[fouls[cur] >= FOUL_OUT]
        if len(bad):
            pool = np.flatnonzero(~onmask & eligible)
            if len(pool):
                pool = pool[np.argsort(-share[pool])]
                for a, b in zip(bad, pool):
                    onmask[a] = False
                    onmask[b] = True
            cur = np.flatnonzero(onmask)
        out[k] = prior.pids[cur]

        state_min[onmask != prev_onmask] = 0.0
        d = script.dur[k]
        state_min += d / 60.0
        half_min[cur] += d / 60.0
        hit = rng.random(5) < prior.fpm[cur] * (d / 60.0) * fscale
        if hit.any():
            fouls[cur[hit]] += 1
    return out, fouls_hist


# ===========================================================================
# 4. Arms
# ===========================================================================
class _ExitArm(_WaveArm):
    mode = "X1"

    def __init__(self, fit: RotationFit, wave: WaveFit, comp: CompFit, exit_: ExitFit,
                 side_state: dict | None = None):
        super().__init__(fit, wave, side_state)
        self.comp = comp
        self.exit = exit_

    def simulate(self, prior: TeamPrior, script: GameScript,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        avail = draw_available(prior, rng)
        pe, tf = _side_arrays(self.side_state, (prior.game_id, prior.team_id),
                              script, script.n)
        return run_wave7(prior, script, avail, self.wave, self.comp, self.exit,
                         self.fit, rng, mode=self.mode, hard_reset=self.hard_reset,
                         prev_end=pe, team_fouls=tf)


class X1ExitClass(_ExitArm):
    """the exit class count conditioned on size and game state"""

    name = "X1_exit_class"
    simplicity_rank = 11
    mode = "X1"


class X2ExitClassFoul(_ExitArm):
    """X1 with a three-level foul class in place of the binary foul state"""

    name = "X2_exit_class_foul"
    simplicity_rank = 12
    mode = "X2"


class X3ExitClassPrev(_ExitArm):
    """X1 with the previous wave-of-this-half's entry class"""

    name = "X3_exit_class_prev"
    simplicity_rank = 13
    mode = "X3"


ARMS = {"X1_exit_class": X1ExitClass, "X2_exit_class_foul": X2ExitClassFoul,
        "X3_exit_class_prev": X3ExitClassPrev}
