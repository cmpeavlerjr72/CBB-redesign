"""
rotation_v8.py -- ROUND 8 of the L4 rotation bake-off: the exit count as a RATE,
`P(k_out | size, state, n_starters_on_floor)`.

Why round 8 exists
------------------
Round 7 closed the exit-side class marginal and lost the cells
(`experiments.md` 17.10, 17.14 item 2, 17.17):

    starter share of leavers   ACTUAL 0.5576   K1 0.4927   X1 0.5611
    final 8:00 |m| <= 5        ACTUAL 0.7491   K1 0.7001   X1 0.5394  (-21.0 pp)

because `P(k_out | size, exit_cell)` is a LEVEL: it removes starters at the
population frequency no matter how many are on the floor at that moment. The
real rate runs 0.37 / 0.44 / 0.52 / 0.66 / 1.00 by the number of starters on the
floor (17.17, 4,072-48,261 leavers per cell, both seasons), a 29 pp span with a
FIXED POINT at three starters; the level form has a constant drift of -0.071 per
swap and no fixed point, so the floor walks down until the support clip binds.

Round 8 keeps round 3b's base fits, round 4's hazards, round 5's wave tables and
rank-within-class rule, round 6's K1 entry rule, the hard second-half reset and
round 7's whole exit MECHANISM byte for byte, and changes only the conditioning
set of the `k_out` table, by ONE axis:

    Y1  `exit_rate`       P(k_out | size, exit_cell, n_starters_on_floor),
                          shrunk one level to X1's own row at k = 300
    Y2  `exit_rate_foul`  Y1 with round 7's three-level foul class in place of
                          the binary foul_state axis, shrunk to Y1's own row

Pre-registration: `docs/models/rotation/experiments.md` section 18, written and
committed (b6a18ec) before any object here was fitted.

Engine expressibility
---------------------
`n_st` is `is_st[on_idx].sum()`, an integer in 0..5 the sampler already computes,
gathered as a fourth index into the same CDF table: Y1 is one wider gather than
X1 and adds no operation to the (2N, S) roster block, so the sim loop still makes
no model call (`CLAUDE.md`).
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
)
from cbb_sim.models.rotation_v4 import design, _side_arrays
from cbb_sim.models.rotation_v5 import MAX_WAVE, N_FS, N_MB, WaveFit, _WaveArm, wave_cell
from cbb_sim.models.rotation_v6 import CompFit, _race
from cbb_sim.models.rotation_v7 import (
    FOUL_TROUBLE,
    K_SHRINK,
    N_ETC,
    N_EXIT_CELL,
    N_FCLASS,
    exit_cell,
    foul_class,
)

#: the declared composition axis (18.2): the number of the model's own PREDICTED
#: starting five among the five on the floor BEFORE the swap, six levels 0..5.
N_ST = 6


# ===========================================================================
# 1. The fitted object
# ===========================================================================
@dataclass
class ExitFit8:
    """The two round-8 exit objects plus their X1 parent. One artifact carries
    all three so the arms are fitted on identical rows and cannot drift apart."""

    p_x1: list = field(default_factory=list)   # (MAX_WAVE, N_EXIT_CELL, MW+1)
    p_y1: list = field(default_factory=list)   # (MAX_WAVE, N_EXIT_CELL, N_ST, MW+1)
    p_y2: list = field(default_factory=list)   # (MW, N_ETC*N_MB, N_FCLASS, N_ST, MW+1)
    n_x1: list = field(default_factory=list)
    n_y1: list = field(default_factory=list)
    n_y2: list = field(default_factory=list)
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
    def Y1(self) -> np.ndarray:
        return np.asarray(self.p_y1, dtype=np.float64).reshape(
            MAX_WAVE, N_EXIT_CELL, N_ST, MAX_WAVE + 1)

    @property
    def Y2(self) -> np.ndarray:
        return np.asarray(self.p_y2, dtype=np.float64).reshape(
            MAX_WAVE, N_ETC * N_MB, N_FCLASS, N_ST, MAX_WAVE + 1)

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "ExitFit8":
        return ExitFit8(**json.loads(Path(path).read_text(encoding="utf-8")))


# ===========================================================================
# 2. Training counts -- round 7's pass, byte for byte, with the n_st axis
# ===========================================================================
def build_exit8_training(tp: pd.DataFrame, feats: pd.DataFrame, fit: RotationFit,
                         game_ids, fouls: pd.DataFrame | None = None,
                         max_team_games: int = 0, seed: int = 11) -> dict:
    """Counts for `P(k_out | size, state, n_starters_on_floor)` over the training
    window's own wave boundaries.

    This is `rotation_v7.build_exit_training` with one extra index. The usable-
    wave filter is rounds 6's and 7's, byte for byte (equal leaver and entrant
    counts, five men on the floor, a non-empty as-of bench pool, every entrant
    inside it); rows are built over the as-of candidate pool (`build_priors`),
    never the game's own participant list; the sample (`max_team_games`, `seed`)
    is rounds 6's and 7's, so the exit, entry and wave tables are fitted on the
    same rows (18.5).
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
    cy1 = np.zeros((MAX_WAVE, N_EXIT_CELL, N_ST, MAX_WAVE + 1))
    cy2 = np.zeros((MAX_WAVE, N_ETC * N_MB, N_FCLASS, N_ST, MAX_WAVE + 1))
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
        for k in range(1, npos):
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
            nst = int(min((is_st[on_idx] > 0.5).sum(), N_ST - 1))
            k_out = int(min((is_st[leave] > 0.5).sum(), MAX_WAVE))
            c1[s - 1, ce, k_out] += 1.0
            cy1[s - 1, ce, nst, k_out] += 1.0
            cy2[s - 1, tm, fc, nst, k_out] += 1.0
            n_used += 1
    return {"c1": c1, "cy1": cy1, "cy2": cy2, "n_waves": n_waves, "n_used": n_used,
            "team_games": n_tg}


def _shrink(counts: np.ndarray, parent: np.ndarray, k: float, size: int) -> np.ndarray:
    """Round 7's `_shrink`, byte for byte: one-level shrinkage of a (k_out,)
    count row toward `parent` at `k`, support clipped to 0..size, renormalised."""
    row = (counts + k * parent) / (counts.sum() + k)
    row = row.copy()
    row[size + 1:] = 0.0
    z = row.sum()
    return row / z if z > 0 else np.eye(MAX_WAVE + 1)[0]


def fit_exit8(counts: dict, k: float = K_SHRINK, wave_source: str = "",
              comp_source: str = "", hazard_source: str = "") -> ExitFit8:
    """The two objects and their X1 parent, from counts and a declared shrinkage
    constant. No optimiser, no grid search against any gate cell."""
    c1, cy1, cy2 = counts["c1"], counts["cy1"], counts["cy2"]
    p1 = np.zeros_like(c1)
    py1 = np.zeros_like(cy1)
    py2 = np.zeros_like(cy2)
    for s in range(MAX_WAVE):
        tot = c1[s].sum()
        marg = c1[s].sum(axis=0)
        marg = marg / tot if tot > 0 else np.eye(MAX_WAVE + 1)[0]
        for ce in range(N_EXIT_CELL):
            # X1's own row, recomputed from the same counts with round 7's
            # constant: identical to round 7's table by construction (18.3).
            p1[s, ce] = _shrink(c1[s, ce], marg, k, s + 1)
            for ns in range(N_ST):
                py1[s, ce, ns] = _shrink(cy1[s, ce, ns], p1[s, ce], k, s + 1)
        for tm in range(N_ETC * N_MB):
            for fc in range(N_FCLASS):
                # Y2's parent is Y1's own row at the matching foul_state: class 0
                # is foul_state 0, classes 1 and 2 are foul_state 1 (18.3).
                fs = 0 if fc == 0 else 1
                for ns in range(N_ST):
                    py2[s, tm, fc, ns] = _shrink(cy2[s, tm, fc, ns],
                                                 py1[s, tm * N_FS + fs, ns], k, s + 1)
    return ExitFit8(p_x1=p1.tolist(), p_y1=py1.tolist(), p_y2=py2.tolist(),
                    n_x1=c1.sum(axis=2).tolist(), n_y1=cy1.sum(axis=3).tolist(),
                    n_y2=cy2.sum(axis=4).tolist(), k_shrink=float(k),
                    n_waves=int(counts["n_waves"]), n_pairs=int(counts["n_used"]),
                    wave_source=wave_source, comp_source=comp_source,
                    hazard_source=hazard_source,
                    notes={"team_games": int(counts["team_games"])})


# ===========================================================================
# 3. The sampler -- round 7's `run_wave7` with the composition index
# ===========================================================================
def run_wave8(prior: TeamPrior, script: GameScript, avail: np.ndarray, wf: WaveFit,
              cf: CompFit, xf: ExitFit8, fit: RotationFit, rng: np.random.Generator,
              mode: str = "Y1", hard_reset: bool = True,
              prev_end: np.ndarray | None = None,
              team_fouls: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """One team-game's on-floor sequence: round 5's wave draw, round 7's exit
    mechanism with the composition axis, round 6's K1 entry rule.

    The uniform order is round 7's, byte for byte (round 5's coupling draw, the
    wave draw, the size draw, the `k_out` uniform, K1's `k_in` uniform, the bench
    race vector), so the round-8 arms are byte-aligned with X1 and with each
    other (18.12 item 1)."""
    sub = wf.hazards()
    pw, ps = wf.pw, wf.ps
    cum_size = np.cumsum(ps, axis=1)
    kin_tab = cf.kin
    y1, y2 = xf.Y1, xf.Y2
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

    for k in range(npos):
        cur_half = 0 if script.period[k] == 1 else 1
        if cur_half != prev_half:
            half_min[:] = 0.0
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
                    u_x = rng.random()          # round 7's k_out draw, same position
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

                    # ---- the ROUND-8 exit rule: X1's mechanism, one axis more
                    ce = int(exit_cell(np.array([script.period[k]]),
                                       np.array([script.start_clock[k]]),
                                       np.array([script.margin[k]]),
                                       np.array([fs]))[0])
                    st_o = np.flatnonzero(is_st[on_idx] > 0.5)
                    bn_o = np.flatnonzero(is_st[on_idx] <= 0.5)
                    nst = int(min(len(st_o), N_ST - 1))
                    forced_st = int((fouls[on_idx[st_o]] >= FOUL_OUT).sum()) if len(st_o) else 0
                    forced_bn = int((fouls[on_idx[bn_o]] >= FOUL_OUT).sum()) if len(bn_o) else 0
                    if mode == "Y2":
                        fc = foul_class(fouls[on_idx].astype(np.float64), is_st[on_idx])
                        row = y2[sz - 1, ce // N_FS, fc, nst].copy()
                    else:
                        row = y1[sz - 1, ce, nst].copy()
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
class _ExitArm8(_WaveArm):
    mode = "Y1"

    def __init__(self, fit: RotationFit, wave: WaveFit, comp: CompFit, exit_: ExitFit8,
                 side_state: dict | None = None):
        super().__init__(fit, wave, side_state)
        self.comp = comp
        self.exit = exit_

    def simulate(self, prior: TeamPrior, script: GameScript,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        avail = draw_available(prior, rng)
        pe, tf = _side_arrays(self.side_state, (prior.game_id, prior.team_id),
                              script, script.n)
        return run_wave8(prior, script, avail, self.wave, self.comp, self.exit,
                         self.fit, rng, mode=self.mode, hard_reset=self.hard_reset,
                         prev_end=pe, team_fouls=tf)


class Y1ExitRate(_ExitArm8):
    """the exit class count conditioned on size, game state AND the composition"""

    name = "Y1_exit_rate"
    simplicity_rank = 14
    mode = "Y1"


class Y2ExitRateFoul(_ExitArm8):
    """Y1 with a three-level foul class in place of the binary foul state"""

    name = "Y2_exit_rate_foul"
    simplicity_rank = 15
    mode = "Y2"


ARMS = {"Y1_exit_rate": Y1ExitRate, "Y2_exit_rate_foul": Y2ExitRateFoul}
