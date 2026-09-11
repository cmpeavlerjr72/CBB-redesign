"""
rotation_v5.py -- ROUND 5 of the L4 rotation bake-off: the JOINT substitution
wave at a dead ball.

Why round 5 exists
------------------
Round 4 changed family to per-player discrete-time substitution hazards and
proved two things at once (L30, `docs/models/rotation/experiments.md` section
11):

  * the fitted hazards reproduce every marginal rate the audit measured -- at a
    period boundary a bench player's exit is 0.786 predicted against 0.791
    actual, an off-floor starter's entry 0.771 against 0.780;
  * and the arms built on them still cannot make a COORDINATED substitution.
    Five independent Bernoulli coins spread the same total exits over more
    boundaries: 0.206 substitutions per boundary against a real 0.152, 19.7
    distinct lineups per team-game against 14.8, an 11 pp shortfall at the
    second-half tip when the reset has to be earned, and a late-game margin
    response compressed to 12.4 pp against 22.7.

Calibrated marginals are necessary and not sufficient when the constraint is
joint. Round 5 keeps round 4's fitted hazards EXACTLY as they are and changes
only the draw:

    1. one Bernoulli per (team, boundary) -- "is there a wave here" -- from a
       cell table over (`prev_end`, time cell, margin band, foul state);
    2. a wave SIZE from a categorical over the same cell;
    3. a COMPOSITION from round 4's own per-player exit/entry hazards, either
       ranked (W1/W3) or drawn (W2);
    4. optionally one shared dead-ball draw so the two teams' waves correlate
       (W3), with the marginals preserved exactly.

Evidence: `docs/tests/rotation_wave_audit_2026-09-11.md`. Pre-registration:
`docs/models/rotation/experiments.md` section 12, written and committed before
any arm was fitted.

Engine expressibility
---------------------
The cell is (`prev_end`, time cell, margin band, foul state): every component is
state the possession loop already carries at a boundary. `p_wave` and `p_size`
are LOOKUP TABLES by construction, so the sim loop makes no model call; the
composition step is round 4's own linear score, already vectorised in
`engine/rotation_adapter.py`. `wave_cell()` is written over (M,) arrays and is
called with M = 1 offline and M = 2N in the engine, so the offline sampler and
the adapter cannot drift apart.
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
    RotationArm,
    RotationFit,
    TeamPrior,
    actual_foul_matrix,
    build_priors,
    build_scripts,
    draw_available,
    margin_bucket,
)
from cbb_sim.models.rotation_v4 import (
    PREV_END_CODE,
    PREV_END_LEVELS,
    SubHazardFit,
    design,
    time_cell,
    _side_arrays,
    _side_state_index,
)

N_PE = len(PREV_END_LEVELS)
N_TC = 9
N_MB = 3
N_FS = 2
N_CELL = N_PE * N_TC * N_MB * N_FS
MAX_WAVE = 5

#: the shrinkage constant, fixed at the project's UNDERPOWERED threshold and
#: never tuned: a cell with fewer than 300 boundaries is pulled most of the way
#: to its parent. Declared in the pre-registration, not fitted.
K_SHRINK = 300.0


def wave_cell(prev_end: np.ndarray, period: np.ndarray, sec_left: np.ndarray,
              margin: np.ndarray, foul_state: np.ndarray) -> np.ndarray:
    """(M,) cell index. The SAME function offline and in the engine adapter.

    `foul_state` is 1 when any player on the floor carries >= 4 personal fouls,
    which is the team-level foul signal the audit measures; `margin` is this
    team's own signed margin."""
    pe = np.asarray(prev_end, dtype=np.int64)
    tc = time_cell(np.asarray(period, dtype=np.int64), np.asarray(sec_left))
    mb = margin_bucket(np.asarray(margin))
    fs = np.asarray(foul_state, dtype=np.int64)
    return ((pe * N_TC + tc) * N_MB + mb) * N_FS + fs


# ===========================================================================
# 1. The fitted object
# ===========================================================================
@dataclass
class WaveFit:
    """The wave Bernoulli, the size categorical, the coupling scalar, and a
    COPY of round 4's hazard coefficients so an artifact is self-contained."""

    p_wave: list = field(default_factory=list)       # (N_CELL,)
    p_size: list = field(default_factory=list)       # (N_CELL, MAX_WAVE)
    rho: float = 0.0
    k_shrink: float = K_SHRINK
    n_boundaries: int = 0
    n_waves: int = 0
    out_coef: list = field(default_factory=list)
    out_intercept: float = 0.0
    in_coef: list = field(default_factory=list)
    in_intercept: float = 0.0
    hazard_source: str = ""
    notes: dict = field(default_factory=dict)

    # -- arrays ----------------------------------------------------------
    @property
    def pw(self) -> np.ndarray:
        return np.asarray(self.p_wave, dtype=np.float64)

    @property
    def ps(self) -> np.ndarray:
        return np.asarray(self.p_size, dtype=np.float64).reshape(N_CELL, MAX_WAVE)

    def hazards(self) -> SubHazardFit:
        return SubHazardFit(kind="logistic", out_coef=list(self.out_coef),
                            out_intercept=float(self.out_intercept),
                            in_coef=list(self.in_coef),
                            in_intercept=float(self.in_intercept))

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "WaveFit":
        return WaveFit(**json.loads(Path(path).read_text(encoding="utf-8")))


# ===========================================================================
# 2. Training counts and the fit
# ===========================================================================
def build_wave_training(tp: pd.DataFrame, game_ids, side_state: pd.DataFrame | None = None,
                        fouls: pd.DataFrame | None = None, max_team_games: int = 0,
                        seed: int = 11) -> dict:
    """Boundary-level counts: P(wave) and the size histogram per cell, plus the
    cross-team joint counts the coupling scalar is fitted on.

    Rows are the team-game's own on-floor transitions, at possession-boundary
    resolution -- the only resolution CBBD supports (no `Substitution` rows in
    2024 at all) and the resolution the engine acts at."""
    gset = set(int(g) for g in game_ids)
    sel = tp[tp["game_id"].isin(gset)]
    keys = list(sel.groupby(["game_id", "team_id"], sort=False).groups.keys())
    if max_team_games and len(keys) > max_team_games:
        rs = np.random.RandomState(seed)
        keys = [keys[i] for i in rs.choice(len(keys), max_team_games, replace=False)]
    kset = set((int(a), int(b)) for a, b in keys)

    evg = {}
    if fouls is not None and len(fouls):
        f = fouls[fouls["game_id"].isin(gset)]
        evg = {(int(g), int(t)): d for (g, t), d in f.groupby(["game_id", "team_id"],
                                                              sort=False)}
    ss = _side_state_index(side_state, gset)

    ev = np.zeros(N_CELL)
    n = np.zeros(N_CELL)
    size = np.zeros((N_CELL, MAX_WAVE))
    per_game: dict[int, dict] = {}
    n_tg = 0
    for (gid, tid), g in sel.groupby(["game_id", "team_id"], sort=False):
        gid, tid = int(gid), int(tid)
        if (gid, tid) not in kset:
            continue
        lu = g[SLOTS].to_numpy(dtype="int64")
        npos = lu.shape[0]
        if npos < 20:
            continue
        period = g["period"].to_numpy(dtype="int64")
        clock = g["start_clock"].to_numpy(dtype="int64")
        margin = g["margin"].to_numpy(dtype="int64")
        pids = np.unique(lu)
        on = np.zeros((npos, len(pids)), dtype=bool)
        for s in range(5):
            on[np.arange(npos), np.searchsorted(pids, lu[:, s])] = True
        fm = actual_foul_matrix(evg.get((gid, tid)), period, clock,
                                {int(p): i for i, p in enumerate(pids)}).astype("int64")
        if fm.shape[0] < npos:
            fm = np.vstack([fm, np.repeat(fm[-1:], npos - fm.shape[0], axis=0)])
        pe, _tf = _side_arrays(ss, (gid, tid), _Script(period, clock), npos)

        k = np.arange(1, npos)
        leave = on[:-1] & ~on[1:]
        sz = leave.sum(axis=1)
        wv = (sz > 0).astype(np.float64)
        fs = ((fm[k] >= 4) & on[:-1]).any(axis=1).astype(np.int64)
        ci = wave_cell(pe[k], period[k], clock[k], margin[k], fs)
        ev += np.bincount(ci, weights=wv, minlength=N_CELL)
        n += np.bincount(ci, minlength=N_CELL)
        w = np.flatnonzero(sz > 0)
        if len(w):
            np.add.at(size, (ci[w], np.clip(sz[w], 1, MAX_WAVE) - 1), 1.0)
        per_game.setdefault(gid, {})[bool(g["is_home"].to_numpy()[0])] = (
            g["poss_index"].to_numpy(dtype="int64")[1:], wv, ci)
        n_tg += 1

    # cross-team joint, excluding period boundaries (the hard reset owns those)
    joint = {"both": 0.0, "n": 0.0, "pa_pb": 0.0, "min_ab": 0.0, "pairs": []}
    for gid, sides in per_game.items():
        if True not in sides or False not in sides:
            continue
        (ph, wh, ch), (pa, wa, ca) = sides[True], sides[False]
        common, ih, ia = np.intersect1d(ph, pa, return_indices=True)
        if not len(common):
            continue
        keep = (np.asarray(ch)[ih] // (N_TC * N_MB * N_FS)) != PREV_END_CODE["period_start"]
        joint["both"] += float((wh[ih][keep] * wa[ia][keep]).sum())
        joint["n"] += float(keep.sum())
        joint["pairs"].append((np.asarray(ch)[ih][keep], np.asarray(ca)[ia][keep]))
    return {"ev": ev, "n": n, "size": size, "joint": joint, "team_games": n_tg}


class _Script:
    """the two fields `_side_arrays` reads"""

    def __init__(self, period, start_clock):
        self.period = period
        self.start_clock = start_clock
        self.n = len(period)


def _shrink_wave(ev: np.ndarray, n: np.ndarray, k: float = K_SHRINK) -> np.ndarray:
    """Two-level shrinkage: cell -> (prev_end x time cell) -> prev_end -> root.

    The parents are marginalisations of the same counts, so each level is the
    maximum-likelihood estimate of its own coarser model; `k` is fixed at the
    UNDERPOWERED threshold and is never tuned."""
    ev3 = ev.reshape(N_PE, N_TC, N_MB, N_FS)
    n3 = n.reshape(N_PE, N_TC, N_MB, N_FS)
    root = ev.sum() / max(n.sum(), 1.0)
    e1, m1 = ev3.sum(axis=(1, 2, 3)), n3.sum(axis=(1, 2, 3))
    p1 = (e1 + k * root) / (m1 + k)
    e2, m2 = ev3.sum(axis=(2, 3)), n3.sum(axis=(2, 3))
    p2 = (e2 + k * p1[:, None]) / (m2 + k)
    p3 = (ev3 + k * p2[:, :, None, None]) / (n3 + k)
    return p3.reshape(N_CELL)


def _shrink_size(size: np.ndarray, k: float = K_SHRINK) -> np.ndarray:
    s3 = size.reshape(N_PE, N_TC, N_MB, N_FS, MAX_WAVE)
    root = size.sum(axis=0)
    root = root / max(root.sum(), 1.0)
    s1 = s3.sum(axis=(1, 2, 3))
    q1 = (s1 + k * root[None, :]) / (s1.sum(axis=1, keepdims=True) + k)
    s2 = s3.sum(axis=(2, 3))
    q2 = (s2 + k * q1[:, None, :]) / (s2.sum(axis=2, keepdims=True) + k)
    q3 = (s3 + k * q2[:, :, None, None, :]) / (s3.sum(axis=4, keepdims=True) + k)
    q3 = q3 / q3.sum(axis=4, keepdims=True)
    return q3.reshape(N_CELL, MAX_WAVE)


def fit_wave(counts: dict, hazards: SubHazardFit, hazard_source: str = "",
             k: float = K_SHRINK, collapse_state: bool = False) -> WaveFit:
    """Fit the wave tables and the coupling scalar.

    `collapse_state=True` is the L31 refit-WITHOUT-the-feature arm: the margin
    band and the foul state are marginalised out of the counts BEFORE the
    shrinkage, which is the maximum-likelihood fit of the state-free cell model
    on the same rows -- not an ablation of a fitted coefficient."""
    ev, n, size = counts["ev"].copy(), counts["n"].copy(), counts["size"].copy()
    if collapse_state:
        ev3, n3 = ev.reshape(N_PE, N_TC, N_MB, N_FS), n.reshape(N_PE, N_TC, N_MB, N_FS)
        ev = np.broadcast_to(ev3.sum(axis=(2, 3))[:, :, None, None],
                             ev3.shape).reshape(N_CELL).copy()
        n = np.broadcast_to(n3.sum(axis=(2, 3))[:, :, None, None],
                            n3.shape).reshape(N_CELL).copy()
        s3 = size.reshape(N_PE, N_TC, N_MB, N_FS, MAX_WAVE)
        size = np.broadcast_to(s3.sum(axis=(2, 3))[:, :, None, None, :],
                               s3.shape).reshape(N_CELL, MAX_WAVE).copy()
    p_wave = _shrink_wave(ev, n, k)
    p_size = _shrink_size(size, k)

    # coupling: P(both) = rho * E[min(pA, pB)] + (1 - rho) * E[pA pB]
    j = counts["joint"]
    rho = 0.0
    if j["n"] > 0 and j["pairs"]:
        ca = np.concatenate([a for a, _ in j["pairs"]])
        cb = np.concatenate([b for _, b in j["pairs"]])
        pa, pb = p_wave[ca], p_wave[cb]
        e_ind = float(np.mean(pa * pb))
        e_min = float(np.mean(np.minimum(pa, pb)))
        obs = j["both"] / j["n"]
        if e_min > e_ind + 1e-9:
            rho = float(np.clip((obs - e_ind) / (e_min - e_ind), 0.0, 1.0))
    return WaveFit(p_wave=p_wave.tolist(), p_size=p_size.tolist(), rho=rho,
                   k_shrink=float(k), n_boundaries=int(n.sum()),
                   n_waves=int(ev.sum()),
                   out_coef=list(hazards.out_coef), out_intercept=hazards.out_intercept,
                   in_coef=list(hazards.in_coef), in_intercept=hazards.in_intercept,
                   hazard_source=hazard_source,
                   notes={"collapse_state": bool(collapse_state),
                          "joint_n": float(j["n"]),
                          "joint_both": float(j["both"])})


# ===========================================================================
# 3. The sampler
# ===========================================================================
def run_wave(prior: TeamPrior, script: GameScript, avail: np.ndarray, wf: WaveFit,
             fit: RotationFit, rng: np.random.Generator, hard_reset: bool = True,
             draw_exit: bool = False, draw_entry: bool = False,
             prev_end: np.ndarray | None = None,
             team_fouls: np.ndarray | None = None,
             shared: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One team-game's on-floor sequence under the joint wave draw.

    `shared` is (npos, 2) of uniforms shared by the two teams of the game and is
    used only by W3: column 0 decides whether this boundary is coupled, column 1
    is the common uniform. Marginals are preserved exactly, because a uniform
    shared with the other team is still a uniform."""
    sub = wf.hazards()
    pw, ps = wf.pw, wf.ps
    cum_size = np.cumsum(ps, axis=1)
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
                fs = int(((fouls[on_idx] >= 4).any()))
                ci = int(wave_cell(np.array([prev_end[k]]), np.array([script.period[k]]),
                                   np.array([script.start_clock[k]]),
                                   np.array([script.margin[k]]), np.array([fs]))[0])
                forced = int((fouls[on_idx] >= FOUL_OUT).sum())
                u_c = rng.random()
                u_w = rng.random()
                if shared is not None and u_c < wf.rho:
                    u_w = float(shared[k, 1])
                if u_w < pw[ci] or forced:
                    sz = int(np.searchsorted(cum_size[ci], rng.random()) + 1)
                    sz = int(min(max(sz, forced), len(bench_idx), 5))
                    X = design(is_st[None, :], share[None, :],
                               fouls.astype(np.float64)[None, :], state_min[None, :],
                               half_min[None, :], np.array([script.period[k]]),
                               np.array([script.start_clock[k]]),
                               np.array([script.margin[k]]), np.array([prev_end[k]]),
                               np.array([team_fouls[k]]))[0]
                    po = sub.p_out(X[on_idx])
                    po = np.where(fouls[on_idx] >= FOUL_OUT, 1e9, po)
                    pi = np.clip(sub.p_in(X[bench_idx]), 1e-9, 1 - 1e-9)
                    if draw_exit:
                        wo = np.clip(np.minimum(po, 1 - 1e-9), 1e-9, 1 - 1e-9)
                        wo = np.where(po >= 1e8, 1e12, wo / (1.0 - wo))
                        ko = -np.log(np.clip(rng.random(len(on_idx)), 1e-12, 1.0)) / wo
                        leaving = on_idx[np.argsort(ko, kind="stable")[:sz]]
                    else:
                        leaving = on_idx[np.argsort(-po, kind="stable")[:sz]]
                    if draw_entry:
                        wi = pi / (1.0 - pi)
                        ki = -np.log(np.clip(rng.random(len(bench_idx)), 1e-12, 1.0)) / wi
                        entering = bench_idx[np.argsort(ki, kind="stable")[:sz]]
                    else:
                        entering = bench_idx[np.argsort(-pi, kind="stable")[:sz]]
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
class _WaveArm(RotationArm):
    hard_reset = True
    draw_exit = False
    draw_entry = False
    coupled = False

    def __init__(self, fit: RotationFit, wave: WaveFit, side_state: dict | None = None):
        super().__init__(fit)
        self.wave = wave
        self.side_state = side_state or {}
        self._shared: np.ndarray | None = None

    def set_shared(self, shared: np.ndarray | None) -> None:
        """The caller sets the game-level shared uniforms before BOTH teams of a
        game are simulated; only W3 reads them. Explicit rather than hidden,
        because the offline sampler runs the two teams sequentially while the
        engine runs them in one block."""
        self._shared = shared

    def draw_targets(self, prior, script, rng, avail):      # not used
        return prior.share * script.total_slot

    def simulate(self, prior: TeamPrior, script: GameScript,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        avail = draw_available(prior, rng)
        pe, tf = _side_arrays(self.side_state, (prior.game_id, prior.team_id),
                              script, script.n)
        sh = self._shared if self.coupled else None
        if sh is not None and len(sh) < script.n:
            sh = np.vstack([sh, np.zeros((script.n - len(sh), 2))])
        return run_wave(prior, script, avail, self.wave, self.fit, rng,
                        hard_reset=self.hard_reset, draw_exit=self.draw_exit,
                        draw_entry=self.draw_entry,
                        prev_end=pe, team_fouls=tf, shared=sh)


class W1WaveRank(_WaveArm):
    """rank the exits, rank the entries"""

    name = "W1_wave_rank"
    simplicity_rank = 3


class W2WaveDraw(_WaveArm):
    """draw both sides"""

    name = "W2_wave_draw"
    simplicity_rank = 4
    draw_exit = True
    draw_entry = True


class W4WaveRankDraw(_WaveArm):
    """rank the exits, draw the entries"""

    name = "W4_wave_rank_draw"
    simplicity_rank = 5
    draw_entry = True


class W5WaveDrawRank(_WaveArm):
    """draw the exits, rank the entries"""

    name = "W5_wave_draw_rank"
    simplicity_rank = 6
    draw_exit = True


class W3WaveCoupled(_WaveArm):
    """W1 plus the shared dead-ball draw"""

    name = "W3_wave_coupled"
    simplicity_rank = 7
    coupled = True


ARMS = {"W1_wave_rank": W1WaveRank, "W2_wave_draw": W2WaveDraw,
        "W4_wave_rank_draw": W4WaveRankDraw, "W5_wave_draw_rank": W5WaveDrawRank,
        "W3_wave_coupled": W3WaveCoupled}


def shared_stream(seed: int, game_id: int, npos: int) -> np.ndarray:
    """(npos, 2) uniforms shared by the two teams of one game-simulation.

    Keyed on (seed, game_id) and on nothing else, so the home and away calls of
    the same simulation see the same numbers and a different seed sees different
    ones. The engine draws the same two columns from a game-keyed StreamBook."""
    rng = np.random.default_rng([int(seed), int(game_id), 991])
    return rng.random((int(npos), 2))
