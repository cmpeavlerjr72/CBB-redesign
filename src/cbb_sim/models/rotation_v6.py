"""
rotation_v6.py -- ROUND 6 of the L4 rotation bake-off: the substitution
COMPOSITION conditioned on WHO LEFT.

Why round 6 exists
------------------
Round 5 changed the DRAW and fixed the joint structure it modelled (L33): the
wave Bernoulli plus the wave-size categorical removed round 4's 43% excess
substitution rate outright (0.2058 -> 0.1568-0.1575 per boundary against a real
0.1509) and made the close-and-late starters' share reachable by the hazard
family for the first time. It left exactly one thing open, and named it before
it was measured: **WHO comes in**.

The four knob-free composition rules of round 5 trade two gate families against
each other and no member of the grid is on both sides of the trade -- a ranked
entry makes the floor sticky (close-late 0.728 against 0.749, but 11.5 distinct
lineups against 14.8 and 8.08 players against 9.64) and a drawn entry gets the
breadth exactly right (14.47 lineups) and gives the late stickiness back
(close-late 0.714). A rank rule is a race at zero temperature and a draw rule a
race at temperature one; the truth is between them.

Round 6 keeps round 5's wave tables BYTE FOR BYTE, round 4's hazards BYTE FOR
BYTE, round 5's rank exit rule and the hard second-half reset, and changes only
how the `size` entrants are chosen off the bench:

    T1  `tau_entry`      w_j = (p_in/(1-p_in))**tau, one scalar fitted by
                         maximum likelihood of the observed entrant sets
    K1  `cond_class`     P(k_in | size, k_out) -- the number of predicted
                         STARTERS entering, conditioned on how many left --
                         then a temperature-1 race within each class
    A1  `tier_affinity`  log w_j += mean over leavers of logA[tier(leaver),
                         tier(j)], a fitted 4x4 tier-pair log-affinity

Pre-registration: `docs/models/rotation/experiments.md` section 14, written and
committed (4380a08) before any object here was fitted.

Engine expressibility
---------------------
All three are lookup objects (one scalar, one (5,6,6) table, one (4,4) table)
and all three are array operations over the (2N, S) roster block, so the sim
loop makes no model call (`CLAUDE.md`). None of them carries a margin, clock or
foul term: the state enters a round-6 arm only through objects round 5 already
sized under L31.
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
from cbb_sim.models.rotation_v4 import (
    SubHazardFit,
    design,
    _side_arrays,
    _side_state_index,
)
from cbb_sim.models.rotation_v5 import MAX_WAVE, WaveFit, _WaveArm, wave_cell

#: as-of start-rank tiers: predicted starters, 6-7, 8-9, 10+. Fixed in the
#: pre-registration (14.2), never tuned.
TIER_EDGES = (5, 7, 9)
N_TIER = 4

#: the declared tau grid (14.2): 20 points, step 0.25. tau = 1 is exactly W4.
TAU_GRID = np.round(np.arange(0.25, 5.0001, 0.25), 2)

#: the project's UNDERPOWERED threshold, the same constant round 5 declared.
K_SHRINK = 300.0


def tier_of(prior: TeamPrior) -> np.ndarray:
    """(n,) as-of tier per candidate, from the position in `start_order`."""
    t = np.full(prior.n, N_TIER - 1, dtype=np.int64)
    for pos, i in enumerate(np.asarray(prior.start_order, dtype=np.int64)):
        t[i] = (0 if pos < TIER_EDGES[0] else
                1 if pos < TIER_EDGES[1] else
                2 if pos < TIER_EDGES[2] else 3)
    return t


# ===========================================================================
# 1. The fitted object
# ===========================================================================
@dataclass
class CompFit:
    """The three round-6 composition objects. One artifact carries all three so
    the arms are fitted on identical rows and cannot drift apart."""

    tau: float = 1.0
    p_kin: list = field(default_factory=list)        # (MAX_WAVE, MAX_WAVE+1, MAX_WAVE+1)
    log_a: list = field(default_factory=list)        # (N_TIER, N_TIER)
    tau_grid: list = field(default_factory=list)
    tau_ll: list = field(default_factory=list)
    k_shrink: float = K_SHRINK
    n_waves: int = 0
    n_pairs: int = 0
    wave_source: str = ""
    hazard_source: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def kin(self) -> np.ndarray:
        return np.asarray(self.p_kin, dtype=np.float64).reshape(
            MAX_WAVE, MAX_WAVE + 1, MAX_WAVE + 1)

    @property
    def A(self) -> np.ndarray:
        return np.asarray(self.log_a, dtype=np.float64).reshape(N_TIER, N_TIER)

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "CompFit":
        return CompFit(**json.loads(Path(path).read_text(encoding="utf-8")))


# ===========================================================================
# 2. Training counts -- ONE pass, all three objects
# ===========================================================================
def build_comp_training(tp: pd.DataFrame, feats: pd.DataFrame, fit: RotationFit,
                        game_ids, hazards: SubHazardFit,
                        fouls: pd.DataFrame | None = None,
                        side_state: pd.DataFrame | None = None,
                        max_team_games: int = 0, seed: int = 11) -> dict:
    """Counts for `P(k_in | size, k_out)`, the tier-pair lift and the tau
    likelihood, over the training window's own wave boundaries.

    Rows are built over the **as-of candidate pool** (`build_priors`), never the
    game's own participant list, so the pool at fit time is the pool at
    simulation time. The training game's own `PersonalFoul` events supply the
    foul state -- legitimate at fit time, where the label is that same game's
    substitution, and never at simulation time (the rule of section 5).
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
    ss = _side_state_index(side_state, gset)

    kin = np.zeros((MAX_WAVE, MAX_WAVE + 1, MAX_WAVE + 1))
    obs = np.zeros((N_TIER, N_TIER))
    exp = np.zeros((N_TIER, N_TIER))
    ll = np.zeros(len(TAU_GRID))
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
        prev_end, team_fouls = _side_arrays(ss, key, script, npos)

        on = np.zeros((npos, n), dtype=bool)
        for k in range(npos):
            for p in lu[k]:
                j = idx_of.get(int(p))
                if j is not None:
                    on[k, j] = True

        share = prior.share / max(prior.share.sum(), 1e-9) * 5.0
        is_st = np.zeros(n, dtype=np.float64)
        is_st[prior.starters()] = 1.0
        tier = tier_of(prior)

        state_min = np.zeros(n)
        half_min = np.zeros(n)
        prev_half = 0
        for k in range(npos):
            cur_half = 0 if script.period[k] == 1 else 1
            if cur_half != prev_half:
                half_min[:] = 0.0
                prev_half = cur_half
            if k > 0:
                prev_on = on[k - 1]
                leave = np.flatnonzero(prev_on & ~on[k])
                enter = np.flatnonzero(~prev_on & on[k])
                if len(leave):
                    n_waves += 1
                fouls_v = fm[k].astype("float64") if k < fm.shape[0] else np.zeros(n)
                pool = np.flatnonzero(~prev_on & (fouls_v < FOUL_OUT))
                # only fully-mapped waves enter the joint counts: an entrant who
                # is not in the as-of candidate pool cannot be represented by any
                # arm, and counting him would bias the class table toward zero.
                if (len(leave) and len(leave) == len(enter) and len(pool)
                        and prev_on.sum() == 5 and set(enter) <= set(pool.tolist())):
                    s = int(min(len(leave), MAX_WAVE))
                    X = design(is_st[None, :], share[None, :], fouls_v[None, :],
                               state_min[None, :], half_min[None, :],
                               np.array([script.period[k]]),
                               np.array([script.start_clock[k]]),
                               np.array([script.margin[k]]), np.array([prev_end[k]]),
                               np.array([team_fouls[k]]))[0]
                    pi = np.clip(hazards.p_in(X[pool]), 1e-9, 1 - 1e-9)
                    logw = np.log(pi) - np.log1p(-pi)
                    k_out = int((is_st[leave] > 0.5).sum())
                    k_in = int((is_st[enter] > 0.5).sum())
                    kin[s - 1, min(k_out, MAX_WAVE), min(k_in, MAX_WAVE)] += 1.0
                    # tier-pair lift: observed pairs and their expectation under
                    # the BASELINE (tau = 1) entry weights on this same row
                    w = np.exp(logw - logw.max())
                    wshare = np.zeros(N_TIER)
                    np.add.at(wshare, tier[pool], w)
                    wshare = wshare / max(wshare.sum(), 1e-300)
                    for a in tier[leave]:
                        np.add.at(obs[a], tier[enter], 1.0)
                        exp[a] += s * wshare
                    # tau likelihood, the declared independent-choice form
                    ent_lw = logw[np.searchsorted(pool, enter)]
                    z = TAU_GRID[:, None] * logw[None, :]
                    zm = z.max(axis=1)
                    lse = zm + np.log(np.exp(z - zm[:, None]).sum(axis=1))
                    ll += TAU_GRID * ent_lw.sum() - s * lse
                    n_used += 1
                state_min[on[k] != prev_on] = 0.0
            d = script.dur[k] / 60.0
            state_min += d
            half_min[on[k]] += d
    return {"kin": kin, "obs": obs, "exp": exp, "ll": ll, "n_waves": n_waves,
            "n_used": n_used, "team_games": n_tg}


def fit_comp(counts: dict, k: float = K_SHRINK, wave_source: str = "",
             hazard_source: str = "") -> CompFit:
    """The three objects, from counts and a declared shrinkage constant. No
    optimiser, no grid search against any gate cell."""
    kin = counts["kin"].copy()
    out = np.zeros_like(kin)
    for s in range(MAX_WAVE):
        tot = kin[s].sum()
        marg = kin[s].sum(axis=0)
        marg = marg / tot if tot > 0 else np.eye(MAX_WAVE + 1)[0]
        for ko in range(MAX_WAVE + 1):
            row = kin[s, ko]
            out[s, ko] = (row + k * marg) / (row.sum() + k)
            out[s, ko, s + 2:] = 0.0
            z = out[s, ko].sum()
            out[s, ko] = out[s, ko] / z if z > 0 else np.eye(MAX_WAVE + 1)[0]
    log_a = np.log((counts["obs"] + k) / (counts["exp"] + k))
    ll = np.asarray(counts["ll"], dtype=np.float64)
    tau = float(TAU_GRID[int(np.argmax(ll))]) if np.isfinite(ll).any() else 1.0
    return CompFit(tau=tau, p_kin=out.tolist(), log_a=log_a.tolist(),
                   tau_grid=TAU_GRID.tolist(), tau_ll=ll.tolist(), k_shrink=float(k),
                   n_waves=int(counts["n_waves"]), n_pairs=int(counts["n_used"]),
                   wave_source=wave_source, hazard_source=hazard_source,
                   notes={"kin_counts": counts["kin"].sum(axis=2).tolist(),
                          "obs_pairs": counts["obs"].tolist(),
                          "exp_pairs": counts["exp"].tolist(),
                          "team_games": int(counts["team_games"])})


# ===========================================================================
# 3. The sampler -- round 5's wave draw with the entry rule replaced
# ===========================================================================
def _race(w: np.ndarray, u: np.ndarray, k: int) -> np.ndarray:
    """Efraimidis-Spirakis: the k smallest exponential keys, i.e. weighted
    sampling without replacement."""
    key = -np.log(np.clip(u, 1e-12, 1.0)) / np.maximum(w, 1e-300)
    return np.argsort(key, kind="stable")[:k]


def run_wave6(prior: TeamPrior, script: GameScript, avail: np.ndarray, wf: WaveFit,
              cf: CompFit, fit: RotationFit, rng: np.random.Generator,
              mode: str = "T1", hard_reset: bool = True,
              prev_end: np.ndarray | None = None,
              team_fouls: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """One team-game's on-floor sequence: round 5's wave draw, round 5's RANK
    exit rule, and the round-6 entry rule named by `mode`.

    Every round-6 arm draws the same uniforms whether or not it uses them (the
    `k_in` uniform, and one race vector over the bench), so the three arms sit
    at identical stream positions and are paired (14.11)."""
    sub = wf.hazards()
    pw, ps = wf.pw, wf.ps
    cum_size = np.cumsum(ps, axis=1)
    kin_tab, log_a, tau = cf.kin, cf.A, float(cf.tau)
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
    tier = tier_of(prior)
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
                fs = int(((fouls[on_idx] >= 4).any()))
                ci = int(wave_cell(np.array([prev_end[k]]), np.array([script.period[k]]),
                                   np.array([script.start_clock[k]]),
                                   np.array([script.margin[k]]), np.array([fs]))[0])
                forced = int((fouls[on_idx] >= FOUL_OUT).sum())
                _u_c = rng.random()            # round 5's coupling draw, unused here
                u_w = rng.random()
                if u_w < pw[ci] or forced:
                    sz = int(np.searchsorted(cum_size[ci], rng.random()) + 1)
                    sz = int(min(max(sz, forced), len(bench_idx), 5))
                    u_k = rng.random()          # drawn by EVERY arm (14.11)
                    X = design(is_st[None, :], share[None, :],
                               fouls.astype(np.float64)[None, :], state_min[None, :],
                               half_min[None, :], np.array([script.period[k]]),
                               np.array([script.start_clock[k]]),
                               np.array([script.margin[k]]), np.array([prev_end[k]]),
                               np.array([team_fouls[k]]))[0]
                    po = sub.p_out(X[on_idx])
                    po = np.where(fouls[on_idx] >= FOUL_OUT, 1e9, po)
                    pi = np.clip(sub.p_in(X[bench_idx]), 1e-9, 1 - 1e-9)
                    leaving = on_idx[np.argsort(-po, kind="stable")[:sz]]
                    u_race = rng.random(len(bench_idx))
                    logw = np.log(pi) - np.log1p(-pi)

                    if mode == "T1":
                        w = np.exp(tau * (logw - logw.max()))
                        entering = bench_idx[_race(w, u_race, sz)]
                    elif mode == "A1":
                        adj = log_a[tier[leaving]].mean(axis=0)     # (N_TIER,)
                        lw = logw + adj[tier[bench_idx]]
                        w = np.exp(lw - lw.max())
                        entering = bench_idx[_race(w, u_race, sz)]
                    else:                                           # K1
                        k_out = int((is_st[leaving] > 0.5).sum())
                        st_b = np.flatnonzero(is_st[bench_idx] > 0.5)
                        bn_b = np.flatnonzero(is_st[bench_idx] <= 0.5)
                        row = kin_tab[sz - 1, min(k_out, MAX_WAVE)].copy()
                        lo = max(0, sz - len(bn_b))
                        hi = min(sz, len(st_b))
                        row[:lo] = 0.0
                        row[hi + 1:] = 0.0
                        tot = row.sum()
                        if tot <= 0:
                            k_in = lo
                        else:
                            k_in = int(np.searchsorted(np.cumsum(row / tot), u_k))
                            k_in = int(min(max(k_in, lo), hi))
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
class _CompArm(_WaveArm):
    mode = "T1"

    def __init__(self, fit: RotationFit, wave: WaveFit, comp: CompFit,
                 side_state: dict | None = None):
        super().__init__(fit, wave, side_state)
        self.comp = comp

    def simulate(self, prior: TeamPrior, script: GameScript,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        avail = draw_available(prior, rng)
        pe, tf = _side_arrays(self.side_state, (prior.game_id, prior.team_id),
                              script, script.n)
        return run_wave6(prior, script, avail, self.wave, self.comp, self.fit, rng,
                         mode=self.mode, hard_reset=self.hard_reset,
                         prev_end=pe, team_fouls=tf)


class T1TauEntry(_CompArm):
    """one fitted temperature on the entry weights"""

    name = "T1_tau_entry"
    simplicity_rank = 8
    mode = "T1"


class K1CondClass(_CompArm):
    """the entry class count conditioned on how many starters left"""

    name = "K1_cond_class"
    simplicity_rank = 9
    mode = "K1"


class A1TierAffinity(_CompArm):
    """a fitted tier-pair log-affinity on the entry weights"""

    name = "A1_tier_affinity"
    simplicity_rank = 10
    mode = "A1"


ARMS = {"T1_tau_entry": T1TauEntry, "K1_cond_class": K1CondClass,
        "A1_tier_affinity": A1TierAffinity}
