"""
rotation_v10.py -- ROUND 10 of the L4 rotation bake-off: the WAVE side,
`P(wave | cell, n_st)` and `P(size | cell, n_st)`.

Why round 10 exists
-------------------
Round 9 made the exit rate CONDITIONAL on the composition essentially correct
(94% of the real 36.5 pp span, `experiments.md` 21.8) and the simulated floor is
still about 25% too bench-heavy. The support measurement 23.1 decomposes what is
left. At a possession boundary `n_st -> n_st - k_out + k_in` and four objects
decide it:

    A(n_st)                 P(a wave happens here | n_st)          ARRIVAL
    S(size | n_st)                                                 SIZE
    O(k_out | size, n_st)                                          EXIT
    I(k_in | size, k_out, n_st)                                    ENTRY

Round 9 repaired `O` alone. Measured on 105,410 actual boundaries:

    A actual   0.098 0.175 0.182 0.167 0.139 0.078   (a 10.4 pp inverted U)
    A  Z1      0.175 0.162 0.158 0.158 0.146 0.124   (3.5 pp -- 33% of it)
    size act   1.654 1.696 1.518 1.417 1.328 1.449
    size Z1    1.474 1.408 1.411 1.397 1.412 1.418   (flat, and the wrong sign)

and a validated Markov decomposition puts arrival at 0.59-0.70 of the mean-`n_st`
gap, size at 0.42-0.43, entry at 0.19-0.32 and the exit side at -0.29 to -0.37
(it now works AGAINST the gap).

Round 10 keeps round 3b's base fits, round 4's hazards, round 6's K1 entry rule,
round 9's Z1 exit rule and the hard second-half reset BYTE FOR BYTE and changes
only which table the wave Bernoulli and the wave-size categorical are drawn from:

    round 5   p_wave[cell]              p_size[cell, .]
    round 10  p_wave[cell, n_st]        p_size[cell, n_st, .]

Arms (23.6)
-----------
    V0  `wave_refit`       the axis-free objects REFITTED on round 10's own rows
                           -- the L31 control, so any V1-V5 movement is the AXIS
    V1  `wave_arrival`     arrival on (cell, n_st), product parent
    V2  `wave_size`        size on (cell, n_st), product parent
    V3  `wave_both`        both
    V4  `wave_both_lvl`    both, parent = the cell LEVEL (rounds 5-8's hierarchy)
    V5  `wave_entry`       V3 plus P(k_in | size, k_out, n_st), product parent

Pre-registration: `docs/models/rotation/experiments.md` section 23, written and
committed (83aa854) before this file existed.

No uniform is added, removed or reordered
-----------------------------------------
`run_wave10` is `run_wave8`'s loop with `n_st` computed BEFORE the wave draw
instead of after it and the table gathers widened by one index. Every arm is one
call of that one sampler with different tables, and feeding it round 5's and
round 6's tables broadcast over the `n_st` axis must reproduce `run_wave9`
element for element (the bit-exact identity test of 23.9).
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
from cbb_sim.models.rotation_v4 import design, _side_arrays, _side_state_index
from cbb_sim.models.rotation_v5 import (
    MAX_WAVE,
    N_CELL,
    WaveFit,
    _WaveArm,
    _shrink_size,
    _shrink_wave,
    wave_cell,
)
from cbb_sim.models.rotation_v6 import CompFit, _race
from cbb_sim.models.rotation_v7 import FOUL_TROUBLE, N_EXIT_CELL, exit_cell
from cbb_sim.models.rotation_v8 import N_ST
from cbb_sim.models.rotation_v9 import ExitFit9

MW1 = MAX_WAVE + 1
EPS = 1e-6

#: the declared grid the three shrinkage constants are fitted over (23.5),
#: widened DOWNWARD from round 9's because 21.13 item 5 said to, and keeping
#: 1000 and 3000 so the widening is not one-sided. Not extended after any number
#: is seen.
K_GRID = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)

#: ties within this many nats per held-out row go to the LARGER k (23.5).
K_TIE_NATS = 0.001


def _lg(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


# ===========================================================================
# 1. The fitted object
# ===========================================================================
@dataclass
class WaveFit10:
    """Every round-10 table plus the marginals they are built from. One artifact
    carries all of them so the arms are fitted on identical rows."""

    p_wave_v0: list = field(default_factory=list)      # (N_CELL,)        V0
    p_size_v0: list = field(default_factory=list)      # (N_CELL, MW)     V0
    p_wave_prod: list = field(default_factory=list)    # (N_CELL, N_ST)   V1/V3/V5
    p_size_prod: list = field(default_factory=list)    # (N_CELL,N_ST,MW) V2/V3/V5
    p_wave_lvlp: list = field(default_factory=list)    # (N_CELL, N_ST)   V4
    p_size_lvlp: list = field(default_factory=list)    # (N_CELL,N_ST,MW) V4
    p_kin_prod: list = field(default_factory=list)     # (MW,MW1,N_ST,MW1) V5
    p_wave_marg: list = field(default_factory=list)    # (N_ST,)   report
    p_size_marg: list = field(default_factory=list)    # (N_ST, MW) report
    n_bnd: list = field(default_factory=list)          # (N_CELL, N_ST)
    n_wave: list = field(default_factory=list)         # (N_CELL, N_ST)
    n_kin: list = field(default_factory=list)          # (MW, MW1, N_ST)
    k_wave: float = 0.0
    k_size: float = 0.0
    k_kin: float = 0.0
    k_wave_v0: float = 0.0
    k_size_v0: float = 0.0
    k_selected_by: str = ""
    k_grid_scores: dict = field(default_factory=dict)
    n_boundaries: int = 0
    n_waves: int = 0
    n_pairs: int = 0
    wave_source: str = ""
    comp_source: str = ""
    exit_source: str = ""
    hazard_source: str = ""
    notes: dict = field(default_factory=dict)

    def _a(self, name: str, shape) -> np.ndarray:
        return np.asarray(getattr(self, name), dtype=np.float64).reshape(shape)

    @property
    def W_V0(self) -> np.ndarray:
        return np.repeat(self._a("p_wave_v0", (N_CELL, 1)), N_ST, axis=1)

    @property
    def S_V0(self) -> np.ndarray:
        return np.repeat(self._a("p_size_v0", (N_CELL, 1, MAX_WAVE)), N_ST, axis=1)

    @property
    def W_PROD(self) -> np.ndarray:
        return self._a("p_wave_prod", (N_CELL, N_ST))

    @property
    def S_PROD(self) -> np.ndarray:
        return self._a("p_size_prod", (N_CELL, N_ST, MAX_WAVE))

    @property
    def W_LVLP(self) -> np.ndarray:
        return self._a("p_wave_lvlp", (N_CELL, N_ST))

    @property
    def S_LVLP(self) -> np.ndarray:
        return self._a("p_size_lvlp", (N_CELL, N_ST, MAX_WAVE))

    @property
    def KIN(self) -> np.ndarray:
        return self._a("p_kin_prod", (MAX_WAVE, MW1, N_ST, MW1))

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "WaveFit10":
        return WaveFit10(**json.loads(Path(path).read_text(encoding="utf-8")))


def broadcast_wave(wf: WaveFit) -> tuple:
    """Round 5's own tables presented on the round-10 `(cell, n_st)` shape. Used
    by V1/V2 for the object they do NOT change and by the identity test of 23.9."""
    return (np.repeat(wf.pw.reshape(N_CELL, 1), N_ST, axis=1),
            np.repeat(wf.ps.reshape(N_CELL, 1, MAX_WAVE), N_ST, axis=1))


def broadcast_kin(cf: CompFit) -> np.ndarray:
    """Round 6's `P(k_in | size, k_out)` on the round-10 shape."""
    return np.repeat(cf.kin.reshape(MAX_WAVE, MW1, 1, MW1), N_ST, axis=2)


# ===========================================================================
# 2. Training counts -- ONE pass over every boundary
# ===========================================================================
def build_wave10_training(tp: pd.DataFrame, feats: pd.DataFrame, fit: RotationFit,
                          game_ids, fouls: pd.DataFrame | None = None,
                          side_state: pd.DataFrame | None = None,
                          max_team_games: int = 0, seed: int = 11) -> dict:
    """Counts for the three round-10 objects over the training window's own
    boundaries (23.3).

    Round 8's `build_exit8_training` widened to count EVERY boundary, on the SAME
    key list, the SAME as-of candidate pool (`build_priors`), the SAME
    `max_team_games` sample and the SAME seed rounds 6-9 used, so the wave, exit
    and composition tables are fitted on the same team-games.

    Two stated conventions:

      * the H1 -> H2 boundary is EXCLUDED. The sampler hard-resets there and runs
        no wave, so it is not a draw of this kernel; 23.1's measurement excluded
        it for the same reason. This makes V0 a refit of round 5's model on round
        10's rows, which is what a control is, not a copy of round 5's table.
      * ARRIVAL and SIZE are counted on every wave (round 5's own convention, no
        usability filter); the ENTRY object uses rounds 6-9's usable-wave filter
        byte for byte, because that is the filter its parent was fitted under.
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

    ev = np.zeros((N_CELL, N_ST))
    nb = np.zeros((N_CELL, N_ST))
    size = np.zeros((N_CELL, N_ST, MAX_WAVE))
    kin = np.zeros((MAX_WAVE, MW1, N_ST, MW1))
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
        fm = actual_foul_matrix(evg.get(key), script.period[:npos],
                                script.start_clock[:npos], idx_of)
        on = np.zeros((npos, n), dtype=bool)
        for k in range(npos):
            for p in lu[k]:
                j = idx_of.get(int(p))
                if j is not None:
                    on[k, j] = True
        is_st = np.zeros(n, dtype=np.float64)
        is_st[prior.starters()] = 1.0
        pe, _tf = _side_arrays(ss, key, script, npos)

        for k in range(1, npos):
            if script.period[k] == 2 and script.period[k - 1] == 1:
                continue                       # the hard reset, not a kernel draw
            prev_on = on[k - 1]
            if prev_on.sum() != 5:
                continue
            on_idx = np.flatnonzero(prev_on)
            fouls_v = fm[k].astype("float64") if k < fm.shape[0] else np.zeros(n)
            fs = int((fouls_v[on_idx] >= FOUL_TROUBLE).any())
            ci = int(wave_cell(np.array([pe[k]]), np.array([script.period[k]]),
                               np.array([script.start_clock[k]]),
                               np.array([script.margin[k]]), np.array([fs]))[0])
            nst = int(min((is_st[on_idx] > 0.5).sum(), N_ST - 1))
            nb[ci, nst] += 1.0
            leave = np.flatnonzero(prev_on & ~on[k])
            if not len(leave):
                continue
            n_waves += 1
            s = int(min(len(leave), MAX_WAVE))
            ev[ci, nst] += 1.0
            size[ci, nst, s - 1] += 1.0
            enter = np.flatnonzero(~prev_on & on[k])
            pool = np.flatnonzero(~prev_on & (fouls_v < FOUL_OUT))
            if not (len(leave) == len(enter) and len(pool)
                    and set(enter.tolist()) <= set(pool.tolist())):
                continue
            k_out = int(min((is_st[leave] > 0.5).sum(), MAX_WAVE))
            k_in = int(min((is_st[enter] > 0.5).sum(), MAX_WAVE))
            kin[s - 1, k_out, nst, k_in] += 1.0
            n_used += 1
    return {"ev": ev, "nb": nb, "size": size, "kin": kin, "n_waves": n_waves,
            "n_used": n_used, "team_games": n_tg}


# ===========================================================================
# 3. The fit -- the marginals, the two parents, the three objects
# ===========================================================================
def _wave_levels(ev: np.ndarray, nb: np.ndarray, k: float) -> tuple:
    """`(root, P0(cell), M(n_st), product parent, level-2 product, level-2 cell)`
    for the arrival Bernoulli."""
    root = float(ev.sum() / max(nb.sum(), 1.0))
    p0 = _shrink_wave(ev.sum(axis=1), nb.sum(axis=1), k)                # (N_CELL,)
    e1, n1 = ev.sum(axis=0), nb.sum(axis=0)
    m1 = (e1 + k * root) / (n1 + k)                                     # (N_ST,)
    par_prod = 1.0 / (1.0 + np.exp(-(_lg(p0)[:, None] + _lg(m1)[None, :]
                                     - _lg(np.array([root]))[0])))
    par_lvl = np.repeat(p0[:, None], N_ST, axis=1)
    z_prod = (ev + k * par_prod) / (nb + k)
    z_lvl = (ev + k * par_lvl) / (nb + k)
    return root, p0, m1, par_prod, z_prod, z_lvl


def _size_levels(size: np.ndarray, k: float) -> tuple:
    """The same three levels for the wave-size categorical."""
    flat = size.sum(axis=1)                                   # (N_CELL, MW)
    root = flat.sum(axis=0)
    root = root / max(root.sum(), 1.0)
    p0 = _shrink_size(flat, k)                                # (N_CELL, MW)
    s1 = size.sum(axis=0)                                     # (N_ST, MW)
    m1 = (s1 + k * root[None, :]) / (s1.sum(axis=1, keepdims=True) + k)
    m1 = m1 / m1.sum(axis=1, keepdims=True)
    par_prod = (p0[:, None, :] * m1[None, :, :]
                / np.maximum(root[None, None, :], EPS))
    par_prod = par_prod / np.maximum(par_prod.sum(axis=2, keepdims=True), EPS)
    par_lvl = np.repeat(p0[:, None, :], N_ST, axis=1)
    tot = size.sum(axis=2, keepdims=True)
    z_prod = (size + k * par_prod) / (tot + k)
    z_prod = z_prod / z_prod.sum(axis=2, keepdims=True)
    z_lvl = (size + k * par_lvl) / (tot + k)
    z_lvl = z_lvl / z_lvl.sum(axis=2, keepdims=True)
    return root, p0, m1, z_prod, z_lvl


def _kin_levels(kin: np.ndarray, k: float) -> tuple:
    """`P(k_in | size, k_out, n_st)`: round 6's own object as one marginal, the
    composition marginal as the other, the product as the parent."""
    z = np.zeros_like(kin)
    k6 = np.zeros((MAX_WAVE, MW1, MW1))
    mk = np.zeros((MAX_WAVE, N_ST, MW1))
    for s in range(MAX_WAVE):
        tot = kin[s].sum()
        marg = kin[s].sum(axis=(0, 1))
        marg = marg / tot if tot > 0 else np.eye(MW1)[0]
        for ko in range(MW1):
            row = kin[s, ko].sum(axis=0)
            r = (row + k * marg) / (row.sum() + k)
            r = r.copy()
            r[s + 2:] = 0.0
            k6[s, ko] = r / r.sum() if r.sum() > 0 else np.eye(MW1)[0]
        for ns in range(N_ST):
            row = kin[s, :, ns].sum(axis=0)
            r = (row + k * marg) / (row.sum() + k)
            r = r.copy()
            r[s + 2:] = 0.0
            mk[s, ns] = r / r.sum() if r.sum() > 0 else np.eye(MW1)[0]
        for ko in range(MW1):
            for ns in range(N_ST):
                par = k6[s, ko] * mk[s, ns] / np.maximum(marg, EPS)
                par = par / par.sum() if par.sum() > 0 else k6[s, ko]
                r = (kin[s, ko, ns] + k * par) / (kin[s, ko, ns].sum() + k)
                r = r.copy()
                r[s + 2:] = 0.0
                z[s, ko, ns] = r / r.sum() if r.sum() > 0 else np.eye(MW1)[0]
    return k6, mk, z


# ---- the leave-one-fold-out selection of the three constants (23.5) --------
def _ll_wave(fit_c: dict, out_c: dict, k: float) -> tuple:
    _r, _p, _m, _pp, z, _zl = _wave_levels(fit_c["ev"], fit_c["nb"], k)
    e, n = out_c["ev"], out_c["nb"]
    p = np.clip(z, 1e-12, 1 - 1e-12)
    ll = float((e * np.log(p) + (n - e) * np.log(1.0 - p)).sum())
    return ll, float(n.sum())


def _ll_wave_v0(fit_c: dict, out_c: dict, k: float) -> tuple:
    """the axis-free model's own held-out score, for V0's constant."""
    p0 = _shrink_wave(fit_c["ev"].sum(axis=1), fit_c["nb"].sum(axis=1), k)
    e, n = out_c["ev"].sum(axis=1), out_c["nb"].sum(axis=1)
    p = np.clip(p0, 1e-12, 1 - 1e-12)
    ll = float((e * np.log(p) + (n - e) * np.log(1.0 - p)).sum())
    return ll, float(n.sum())


def _ll_size(fit_c: dict, out_c: dict, k: float) -> tuple:
    _r, _p, _m, z, _zl = _size_levels(fit_c["size"], k)
    c = out_c["size"]
    return float((c * np.log(np.maximum(z, 1e-12))).sum()), float(c.sum())


def _ll_size_v0(fit_c: dict, out_c: dict, k: float) -> tuple:
    p0 = _shrink_size(fit_c["size"].sum(axis=1), k)
    c = out_c["size"].sum(axis=1)
    return float((c * np.log(np.maximum(p0, 1e-12))).sum()), float(c.sum())


def _ll_kin(fit_c: dict, out_c: dict, k: float) -> tuple:
    _k6, _mk, z = _kin_levels(fit_c["kin"], k)
    c = out_c["kin"]
    return float((c * np.log(np.maximum(z, 1e-12))).sum()), float(c.sum())


_SCORERS = {"wave": _ll_wave, "size": _ll_size, "kin": _ll_kin,
            "wave_v0": _ll_wave_v0, "size_v0": _ll_size_v0}


def select_k(fold_counts: list, which: str, grid=K_GRID) -> dict:
    """Leave-one-fold-out selection on TRAINING data only (23.5). No gate cell,
    no test row and no MAE is visible here."""
    scorer = _SCORERS[which]
    keys = ("ev", "nb", "size", "kin")
    scores = {}
    for k in grid:
        tot_ll = tot_n = 0.0
        for i in range(len(fold_counts)):
            fit_c = {kk: sum(fc[kk] for j, fc in enumerate(fold_counts) if j != i)
                     for kk in keys}
            ll, n = scorer(fit_c, fold_counts[i], k)
            tot_ll += ll
            tot_n += n
        scores[str(k)] = float(tot_ll / tot_n) if tot_n else float("-inf")
    best = max(scores.values())
    near = [float(kk) for kk, v in scores.items() if best - v <= K_TIE_NATS]
    return {"object": which, "k": float(max(near)), "scores": scores,
            "n_folds": len(fold_counts), "grid": [float(g) for g in grid],
            "tie_nats": K_TIE_NATS, "at_grid_boundary":
                bool(float(max(near)) in (grid[0], grid[-1])),
            "criterion": "leave-one-fold-out held-out log-likelihood per held-out "
                         "row, 2024 training season only"}


def fit_wave10(counts: dict, k_wave: float, k_size: float, k_kin: float,
               k_wave_v0: float, k_size_v0: float, wave_source: str = "",
               comp_source: str = "", exit_source: str = "",
               hazard_source: str = "", k_selected_by: str = "",
               k_grid_scores: dict | None = None) -> WaveFit10:
    """Every round-10 table, from one counts pass and five fitted constants."""
    ev, nb, size, kin = counts["ev"], counts["nb"], counts["size"], counts["kin"]
    _root, _p0, m1, _pp, zw_prod, zw_lvl = _wave_levels(ev, nb, k_wave)
    p0_v0 = _shrink_wave(ev.sum(axis=1), nb.sum(axis=1), k_wave_v0)
    _rs, _p0s, m1s, zs_prod, zs_lvl = _size_levels(size, k_size)
    p0s_v0 = _shrink_size(size.sum(axis=1), k_size_v0)
    _k6, _mk, zk = _kin_levels(kin, k_kin)
    return WaveFit10(
        p_wave_v0=p0_v0.tolist(), p_size_v0=p0s_v0.tolist(),
        p_wave_prod=zw_prod.tolist(), p_size_prod=zs_prod.tolist(),
        p_wave_lvlp=zw_lvl.tolist(), p_size_lvlp=zs_lvl.tolist(),
        p_kin_prod=zk.tolist(), p_wave_marg=m1.tolist(), p_size_marg=m1s.tolist(),
        n_bnd=nb.tolist(), n_wave=ev.tolist(), n_kin=kin.sum(axis=3).tolist(),
        k_wave=float(k_wave), k_size=float(k_size), k_kin=float(k_kin),
        k_wave_v0=float(k_wave_v0), k_size_v0=float(k_size_v0),
        k_selected_by=k_selected_by, k_grid_scores=dict(k_grid_scores or {}),
        n_boundaries=int(nb.sum()), n_waves=int(counts["n_waves"]),
        n_pairs=int(counts["n_used"]), wave_source=wave_source,
        comp_source=comp_source, exit_source=exit_source,
        hazard_source=hazard_source,
        notes={"team_games": int(counts["team_games"])})


# ===========================================================================
# 4. The sampler -- run_wave8's loop, n_st computed before the wave draw
# ===========================================================================
def run_wave10(prior: TeamPrior, script: GameScript, avail: np.ndarray, wf: WaveFit,
               pw10: np.ndarray, ps10: np.ndarray, zexit: np.ndarray,
               kin10: np.ndarray, fit: RotationFit, rng: np.random.Generator,
               hard_reset: bool = True, prev_end: np.ndarray | None = None,
               team_fouls: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """One team-game's on-floor sequence.

    `rotation_v8.run_wave8`'s loop with `n_st` computed before the wave draw and
    four gathers widened by one index. The uniform order is round 5's coupling
    draw, the wave draw, the size draw, the `k_out` uniform, K1's `k_in` uniform,
    then the bench race vector -- unchanged, so a round-10 arm is byte-aligned
    with rounds 8's and 9's arms and with every other round-10 arm.
    """
    sub = wf.hazards()
    cum_size = np.cumsum(ps10, axis=2)
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
                # ---- the ROUND-10 axis, computed BEFORE the wave draw -------
                nst = int(min((is_st[on_idx] > 0.5).sum(), N_ST - 1))
                forced = int((fouls[on_idx] >= FOUL_OUT).sum())
                _u_c = rng.random()            # round 5's coupling draw, unused here
                u_w = rng.random()
                if u_w < pw10[ci, nst] or forced:
                    sz = int(np.searchsorted(cum_size[ci, nst], rng.random()) + 1)
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

                    # ---- round 9's Z1 exit rule, byte for byte --------------
                    ce = int(exit_cell(np.array([script.period[k]]),
                                       np.array([script.start_clock[k]]),
                                       np.array([script.margin[k]]),
                                       np.array([fs]))[0])
                    st_o = np.flatnonzero(is_st[on_idx] > 0.5)
                    bn_o = np.flatnonzero(is_st[on_idx] <= 0.5)
                    forced_st = int((fouls[on_idx[st_o]] >= FOUL_OUT).sum()) if len(st_o) else 0
                    forced_bn = int((fouls[on_idx[bn_o]] >= FOUL_OUT).sum()) if len(bn_o) else 0
                    row = zexit[sz - 1, ce, nst].copy()
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
                    take_so = st_o[np.argsort(-po[st_o], kind="stable")[:k_out]] \
                        if k_out else np.array([], dtype="int64")
                    take_bo = bn_o[np.argsort(-po[bn_o], kind="stable")[:sz - k_out]] \
                        if sz - k_out else np.array([], dtype="int64")
                    leaving = on_idx[np.concatenate([take_so, take_bo]).astype(int)]

                    # ---- round 6's K1 entry rule, one index wider ----------
                    logw = np.log(pi) - np.log1p(-pi)
                    st_b = np.flatnonzero(is_st[bench_idx] > 0.5)
                    bn_b = np.flatnonzero(is_st[bench_idx] <= 0.5)
                    u_race = rng.random(len(bench_idx))
                    krow = kin10[sz - 1, min(k_out, MAX_WAVE), nst].copy()
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
# 5. Arms -- one sampler, different tables
# ===========================================================================
class _WaveArm10(_WaveArm):
    """`tables()` is the ONLY thing that differs between round-10 arms."""

    arrival = "round5"          # round5 | v0 | prod | lvlp
    sizing = "round5"           # round5 | v0 | prod | lvlp
    entry = "round6"            # round6 | prod

    def __init__(self, fit: RotationFit, wave: WaveFit, comp: CompFit,
                 exit_: ExitFit9, wave10: WaveFit10 | None = None,
                 side_state: dict | None = None):
        super().__init__(fit, wave, side_state)
        self.comp = comp
        self.exit = exit_
        self.wave10 = wave10
        self._tab: tuple | None = None

    def tables(self) -> tuple:
        if self._tab is None:
            bw, bs = broadcast_wave(self.wave)
            w10 = self.wave10
            pw = {"round5": bw, "v0": None if w10 is None else w10.W_V0,
                  "prod": None if w10 is None else w10.W_PROD,
                  "lvlp": None if w10 is None else w10.W_LVLP}[self.arrival]
            ps = {"round5": bs, "v0": None if w10 is None else w10.S_V0,
                  "prod": None if w10 is None else w10.S_PROD,
                  "lvlp": None if w10 is None else w10.S_LVLP}[self.sizing]
            kin = (broadcast_kin(self.comp) if self.entry == "round6"
                   else w10.KIN)
            self._tab = (np.ascontiguousarray(pw), np.ascontiguousarray(ps),
                         self.exit.Z1, np.ascontiguousarray(kin))
        return self._tab

    def simulate(self, prior: TeamPrior, script: GameScript,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        avail = draw_available(prior, rng)
        pe, tf = _side_arrays(self.side_state, (prior.game_id, prior.team_id),
                              script, script.n)
        pw, ps, zx, kin = self.tables()
        return run_wave10(prior, script, avail, self.wave, pw, ps, zx, kin,
                          self.fit, rng, hard_reset=self.hard_reset,
                          prev_end=pe, team_fouls=tf)


class VREFZ1(_WaveArm10):
    """the identity control of 23.9: round 5's and round 6's own tables on the
    round-10 shape. Must reproduce `run_wave9`'s Z1 element for element."""

    name = "VREF_z1_broadcast"
    simplicity_rank = 16


class V0WaveRefit(_WaveArm10):
    """the axis-free objects refitted on round 10's own rows (the L31 control)"""

    name = "V0_wave_refit"
    simplicity_rank = 17
    arrival = "v0"
    sizing = "v0"


class V1WaveArrival(_WaveArm10):
    """P(wave | cell, n_st), product parent"""

    name = "V1_wave_arrival"
    simplicity_rank = 18
    arrival = "prod"


class V2WaveSize(_WaveArm10):
    """P(size | cell, n_st), product parent"""

    name = "V2_wave_size"
    simplicity_rank = 19
    sizing = "prod"


class V3WaveBoth(_WaveArm10):
    """both wave objects on (cell, n_st), product parent"""

    name = "V3_wave_both"
    simplicity_rank = 20
    arrival = "prod"
    sizing = "prod"


class V4WaveBothLvl(_WaveArm10):
    """V3 with the parent rounds 5-8 used -- the cell LEVEL alone"""

    name = "V4_wave_both_lvl"
    simplicity_rank = 21
    arrival = "lvlp"
    sizing = "lvlp"


class V5WaveEntry(_WaveArm10):
    """V3 plus P(k_in | size, k_out, n_st)"""

    name = "V5_wave_entry"
    simplicity_rank = 22
    arrival = "prod"
    sizing = "prod"
    entry = "prod"


ARMS = {"VREF_z1_broadcast": VREFZ1, "V0_wave_refit": V0WaveRefit,
        "V1_wave_arrival": V1WaveArrival, "V2_wave_size": V2WaveSize,
        "V3_wave_both": V3WaveBoth, "V4_wave_both_lvl": V4WaveBothLvl,
        "V5_wave_entry": V5WaveEntry}

__all__ = ["ARMS", "K_GRID", "K_TIE_NATS", "MW1", "N_ST", "WaveFit10",
           "broadcast_kin", "broadcast_wave", "build_wave10_training",
           "fit_wave10", "run_wave10", "select_k"]
