"""
live_grid.py -- conditional remaining-points tables for the CBB live-odds file (game level).

A table row is one time bin of the stored trajectory ensemble: for the seeds' midpoint states it holds the empirical
distribution of the REMAINING margin and REMAINING total, keyed by the current margin bin (and, in the total-axis
arms, by the current total-so-far quantile bin). The client / grader looks a state up, blends the two adjacent rows
linearly in elapsed time, and reads win probability and remaining-points distributions off the blended pmfs.

No fitted model, no adjustment: every number is a count over the run's own seeds. Pre-registered arm definitions:
docs/models/live_grid/experiments.md. Design: docs/ops/live_trajectories_cbb_design_2026-11-02.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REG_S = 2400
PERIOD_S = 1200
COARSE_TB = 120
LATE_START = 2160
LATE_TB = 30
COARSE_EDGES = (-11, -6, -2, 3, 7, 12)
FOOTBALL_EDGES = (-14, -7, 0, 1, 8, 15)
LATE_EDGES = (-16, -10, -6, -3, -1, 2, 4, 7, 11, 17)
N_TOTAL_BINS = 5
RM_OFF = 80
RM_N = 161          # remaining margin -80..80
RT_N = 260          # remaining total 0..259
PMIN, PMAX = 0.001, 0.999   # published per-mille resolution


def scaled_min_n(n_seeds: int, prod_min_n: int = 200, prod_seeds: int = 2500, floor: int = 3) -> int:
    """Production threshold 200 of 2,500 seeds, scaled by seed count (statistical, so it scales)."""
    return max(floor, int(round(prod_min_n * n_seeds / prod_seeds)))


# ---------------------------------------------------------------------------------------------------------------
# arm specification
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Arm:
    name: str
    edges: tuple = COARSE_EDGES
    total_axis: bool = False       # coarse rows: total curves keyed by total-so-far quantile bin
    late_block: bool = False       # 30 s rows for the last 240 s with LATE_EDGES
    pool: str = "row"              # 'row' (time-row fallback) | 'pool' (neighbour pooling) | 'pool2' (coarse then pool)
    state: bool = True             # False => R0 (no state conditioning)
    exact_end: bool = True         # False => A0 literal CFB fade to zero
    kickoff: bool = True


ARMS = {
    "R0": Arm("R0", state=False),
    "A0": Arm("A0", edges=FOOTBALL_EDGES, exact_end=False, kickoff=False),
    "A_row": Arm("A_row"),
    "A_pool": Arm("A_pool", pool="pool"),
    "B_row": Arm("B_row", total_axis=True),
    "B_pool": Arm("B_pool", total_axis=True, pool="pool"),
    "C_row": Arm("C_row", total_axis=True, late_block=True),
    "C_pool": Arm("C_pool", total_axis=True, late_block=True, pool="pool"),
    "C_pool2": Arm("C_pool2", total_axis=True, late_block=True, pool="pool2"),
}
ELIGIBLE = ("A_row", "A_pool", "B_row", "B_pool", "C_row", "C_pool", "C_pool2")   # simplicity order


def margin_bin(m, edges):
    return np.searchsorted(np.asarray(edges), m, side="right")


# ---------------------------------------------------------------------------------------------------------------
# per-game seed matrices
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class GameSeeds:
    game_id: int
    seeds: np.ndarray         # (S,) seed ids, sorted
    E: np.ndarray             # (S,P) elapsed at possession open (regulation rows), +inf padded
    Mg: np.ndarray            # (S,P) home-away margin BEFORE the possession
    Tt: np.ndarray            # (S,P) total BEFORE the possession
    fm: np.ndarray            # (S,) final margin incl. OT
    ft: np.ndarray            # (S,) final total incl. OT
    reg_m: np.ndarray         # (S,) margin at end of regulation
    reg_t: np.ndarray         # (S,) total at end of regulation
    ot: np.ndarray            # (S,) bool, game went to OT

    @property
    def S(self) -> int:
        return len(self.seeds)

    def take(self, idx) -> "GameSeeds":
        idx = np.asarray(idx)
        return GameSeeds(self.game_id, self.seeds[idx], self.E[idx], self.Mg[idx], self.Tt[idx], self.fm[idx], self.ft[idx],
                         self.reg_m[idx], self.reg_t[idx], self.ot[idx])

    def state_at(self, t: float):
        """(margin, total) of the last checkpoint at or before elapsed t, per seed."""
        idx = (self.E <= t).sum(axis=1) - 1
        idx = np.maximum(idx, 0)
        r = np.arange(self.S)
        return self.Mg[r, idx], self.Tt[r, idx]


def prepare_game(df: pd.DataFrame) -> GameSeeds:
    """df: one game's trajectory rows (all seeds, all periods). Needs seed, poss_idx, period, clock_start, home_score, away_score."""
    df = df.sort_values(["seed", "poss_idx"], kind="stable")
    seed = df["seed"].to_numpy()
    useed, start, counts = np.unique(seed, return_index=True, return_counts=True)
    hs = df["home_score"].to_numpy().astype(np.int32)
    as_ = df["away_score"].to_numpy().astype(np.int32)
    per = df["period"].to_numpy().astype(np.int32)
    cs = df["clock_start"].to_numpy().astype(np.int32)
    S = len(useed)
    # score BEFORE each row = previous row's after-score within the seed; 0 at the first row
    bm = np.empty(len(df), dtype=np.int32)
    bt = np.empty(len(df), dtype=np.int32)
    marg = hs - as_
    tot = hs + as_
    bm[1:] = marg[:-1]
    bt[1:] = tot[:-1]
    bm[start] = 0
    bt[start] = 0
    reg = per <= 2
    el = np.where(reg, (per - 1) * PERIOD_S + (PERIOD_S - cs), 0).astype(np.float64)
    pos = np.arange(len(df)) - np.repeat(start, counts)
    P = int(counts.max())
    sid = np.repeat(np.arange(S), counts)
    E = np.full((S, P), np.inf)
    Mg = np.zeros((S, P), dtype=np.int32)
    Tt = np.zeros((S, P), dtype=np.int32)
    sel = reg
    E[sid[sel], pos[sel]] = el[sel]
    Mg[sid[sel], pos[sel]] = bm[sel]
    Tt[sid[sel], pos[sel]] = bt[sel]
    end = start + counts - 1
    fm = marg[end]
    ft = tot[end]
    # regulation end = after-score of the last regulation row of the seed
    last_reg = np.zeros(S, dtype=np.int64)
    ridx = np.where(reg)[0]
    np.maximum.at(last_reg, sid[ridx], ridx)
    reg_m = marg[last_reg]
    reg_t = tot[last_reg]
    ot = per[end] >= 3
    return GameSeeds(int(df["game_id"].iloc[0]), useed.astype(np.int64), E, Mg, Tt, fm.astype(np.int32), ft.astype(np.int32),
                     reg_m.astype(np.int32), reg_t.astype(np.int32), ot)


# ---------------------------------------------------------------------------------------------------------------
# table rows
# ---------------------------------------------------------------------------------------------------------------
def _hist(idx2: np.ndarray, nb: int, L: int) -> np.ndarray:
    return np.bincount(idx2, minlength=nb * L).reshape(nb, L).astype(np.float64)


def _resolve(h: np.ndarray, n: np.ndarray, row_h: np.ndarray, min_n: int, pool: bool):
    """Return (cells (nb,L) counts, thin (nb,) bool, widened (nb,) int)."""
    nb = len(n)
    cells = h.copy()
    thin = n < min_n
    widened = np.zeros(nb, dtype=np.int32)
    for b in np.where(thin)[0]:
        if not pool:
            cells[b] = row_h
            continue
        lo = hi = b
        tot = n[b]
        acc = h[b].copy()
        mid = (nb - 1) / 2.0
        while tot < min_n and (lo > 0 or hi < nb - 1):
            nl = n[lo - 1] if lo > 0 else -1
            nr = n[hi + 1] if hi < nb - 1 else -1
            if nl > nr or (nl == nr and abs(lo - 1 - mid) < abs(hi + 1 - mid)):
                lo -= 1
                tot += nl
                acc += h[lo]
            else:
                hi += 1
                tot += nr
                acc += h[hi]
        cells[b] = acc if tot >= min_n else row_h
        widened[b] = hi - lo
    return cells, thin, widened


def _pmf(c: np.ndarray) -> np.ndarray:
    s = c.sum(axis=-1, keepdims=True)
    return c / np.where(s > 0, s, 1.0)


@dataclass
class Row:
    mid: float
    kind: str                       # 'kick' | 'coarse' | 'late'
    edges: tuple                    # margin edges of the margin-keyed partition
    n_m: np.ndarray                 # (nb,) seeds per margin bin
    rm: np.ndarray                  # (nb,RM_N) pmf of remaining margin per margin bin (resolved)
    rt_m: np.ndarray                # (nb,RT_N) pmf of remaining total per margin bin (resolved)
    thin_m: np.ndarray
    widened_m: np.ndarray
    rm_row: np.ndarray              # (RM_N,) all-seeds pmf
    rt_row: np.ndarray
    t_edges: np.ndarray | None = None      # (4,) total-so-far quantile edges (total axis)
    n_t: np.ndarray | None = None
    rt_t: np.ndarray | None = None         # (5,RT_N)
    thin_t: np.ndarray | None = None
    widened_t: np.ndarray | None = None
    coarse: "Row | None" = None            # pool2: coarse partition of the same late row
    min_n: int = 0
    n_all: int = 0

    def cell(self, m: np.ndarray, s: np.ndarray):
        """Vectorised lookup -> (rm pmf (n,RM_N), rt pmf (n,RT_N), thin (n,), widened (n,))."""
        k = margin_bin(m, self.edges)
        rm = self.rm[k]
        thin = self.thin_m[k].astype(np.float64)
        wid = self.widened_m[k].astype(np.float64)
        if self.coarse is not None:
            ft = self.thin_m[k]
            if ft.any():
                kc = margin_bin(m, self.coarse.edges)
                rm = np.where(ft[:, None], self.coarse.rm[kc], rm)
                rt = np.where(ft[:, None], self.coarse.rt_m[kc], self.rt_m[k])
                wid = np.where(ft, self.coarse.widened_m[kc], 0.0)
                return rm, rt, thin, wid
            return rm, self.rt_m[k], thin, wid
        if self.rt_t is not None:
            kt = np.searchsorted(self.t_edges, s, side="right")
            rt = self.rt_t[kt]
            thin = np.maximum(thin, self.thin_t[kt].astype(np.float64))
            wid = np.maximum(wid, self.widened_t[kt].astype(np.float64))
        else:
            rt = self.rt_m[k]
        return rm, rt, thin, wid


@dataclass
class EndRow:
    mid: float
    ot_rm: np.ndarray       # (RM_N,) pmf of OT margin given tied
    ot_rt: np.ndarray       # (RT_N,) pmf of OT points given tied
    n_ot: int
    pooled: bool
    kind: str = "end"
    zero_rm: np.ndarray = field(default_factory=lambda: _delta(RM_N, RM_OFF))
    zero_rt: np.ndarray = field(default_factory=lambda: _delta(RT_N, 0))

    def cell(self, m: np.ndarray, s: np.ndarray):
        tied = (m == 0)[:, None]
        return (np.where(tied, self.ot_rm[None, :], self.zero_rm[None, :]),
                np.where(tied, self.ot_rt[None, :], self.zero_rt[None, :]),
                np.zeros(len(m)), np.zeros(len(m)))


def _delta(L: int, i: int) -> np.ndarray:
    v = np.zeros(L)
    v[i] = 1.0
    return v


@dataclass
class Tables:
    arm: Arm
    rows: list
    mids: np.ndarray
    min_n: int
    n_seeds: int
    uncond_p_win: float
    uncond_total: float
    uncond_margin: float


def midstate(gs: GameSeeds, mid: float):
    m, s = gs.state_at(mid)
    remM = np.clip(gs.fm - m + RM_OFF, 0, RM_N - 1)
    remT = np.clip(gs.ft - s, 0, RT_N - 1)
    return m, s, remM.astype(np.int64), remT.astype(np.int64)


def _make_row(gs, mid, kind, edges, total_axis, pool, min_n, cache=None, coarse_edges=None):
    if cache is not None and mid in cache:
        m, s, remM, remT = cache[mid]
    else:
        m, s, remM, remT = midstate(gs, mid)
        if cache is not None:
            cache[mid] = (m, s, remM, remT)
    nb = len(edges) + 1
    k = margin_bin(m, edges)
    hm = _hist(k * RM_N + remM, nb, RM_N)
    ht = _hist(k * RT_N + remT, nb, RT_N)
    n = np.bincount(k, minlength=nb)
    rm_row = _hist(remM, 1, RM_N)[0]
    rt_row = _hist(remT, 1, RT_N)[0]
    do_pool = pool in ("pool", "pool2")
    cm, thin, wid = _resolve(hm, n, rm_row, min_n, do_pool)
    ct, _, _ = _resolve(ht, n, rt_row, min_n, do_pool)
    row = Row(mid, kind, tuple(edges), n, _pmf(cm), _pmf(ct), thin, wid, _pmf(rm_row), _pmf(rt_row), min_n=min_n, n_all=gs.S)
    if total_axis:
        qs = np.quantile(s, np.linspace(0, 1, N_TOTAL_BINS + 1)[1:-1])
        kt = np.searchsorted(qs, s, side="right")
        htt = _hist(kt * RT_N + remT, N_TOTAL_BINS, RT_N)
        nt = np.bincount(kt, minlength=N_TOTAL_BINS)
        ctt, thin_t, wid_t = _resolve(htt, nt, rt_row, min_n, do_pool)
        row.t_edges, row.n_t, row.rt_t, row.thin_t, row.widened_t = qs, nt, _pmf(ctt), thin_t, wid_t
    if coarse_edges is not None:        # pool2: coarse partition of the same row, neighbour pooled
        row.coarse = _make_row(gs, mid, "coarse", coarse_edges, False, "pool", min_n, cache)
    return row


def row_mids(arm: Arm):
    mids = [(i + 0.5) * COARSE_TB for i in range(REG_S // COARSE_TB)]
    if arm.late_block:
        mids = [x for x in mids if x < LATE_START] + [LATE_START + (i + 0.5) * LATE_TB for i in range((REG_S - LATE_START) // LATE_TB)]
    return mids


def build_tables(gs: GameSeeds, arm: Arm, min_n: int, league_ot=None, cache=None) -> Tables:
    """gs must contain TRAIN seeds only. league_ot = (rm_pmf, rt_pmf) fallback for tied-at-2400 OT remainder."""
    rows = []
    if arm.kickoff:
        rows.append(_make_row(gs, 0.0, "kick", (), False, "row", min_n, cache))
    mids = row_mids(arm)
    for mid in mids:
        late = arm.late_block and mid > LATE_START
        if not arm.state:
            r = _make_row(gs, mid, "coarse", (), False, "row", min_n, cache)
        elif late:
            r = _make_row(gs, mid, "late", LATE_EDGES, False, arm.pool, min_n, cache,
                          coarse_edges=COARSE_EDGES if arm.pool == "pool2" else None)
        else:
            r = _make_row(gs, mid, "coarse", arm.edges, arm.total_axis, arm.pool, min_n, cache)
        rows.append(r)
    if arm.exact_end:
        rows.append(make_end_row(gs, league_ot))
    return Tables(arm, rows, np.array([r.mid for r in rows]), min_n, gs.S,
                  float((gs.fm > 0).mean()), float(gs.ft.mean()), float(gs.fm.mean()))


def make_end_row(gs: GameSeeds, league_ot=None) -> EndRow:
    tied = gs.reg_m == 0
    n = int(tied.sum())
    if n >= 3:
        om = np.clip(gs.fm[tied] + RM_OFF, 0, RM_N - 1)
        ot_ = np.clip(gs.ft[tied] - gs.reg_t[tied], 0, RT_N - 1)
        return EndRow(float(REG_S), _pmf(np.bincount(om, minlength=RM_N).astype(float)),
                      _pmf(np.bincount(ot_, minlength=RT_N).astype(float)), n, False)
    if league_ot is None:
        raise ValueError("fewer than 3 tied-at-2400 seeds and no league OT prior supplied")
    return EndRow(float(REG_S), league_ot[0], league_ot[1], n, True)


def league_ot_prior(games: list[GameSeeds]):
    om, ot_ = [], []
    for gs in games:
        tied = gs.reg_m == 0
        om.append(np.clip(gs.fm[tied] + RM_OFF, 0, RM_N - 1))
        ot_.append(np.clip(gs.ft[tied] - gs.reg_t[tied], 0, RT_N - 1))
    om = np.concatenate(om)
    ot_ = np.concatenate(ot_)
    return (_pmf(np.bincount(om, minlength=RM_N).astype(float)), _pmf(np.bincount(ot_, minlength=RT_N).astype(float)))


# ---------------------------------------------------------------------------------------------------------------
# lookup (the client rule): blend the two adjacent rows linearly in elapsed time
# ---------------------------------------------------------------------------------------------------------------
def price(tab: Tables, t: float, m: np.ndarray, s: np.ndarray):
    """Vectorised over states at one elapsed t. Returns rm pmf (n,RM_N), rt pmf (n,RT_N), thin (n,), widened (n,)."""
    m = np.asarray(m)
    s = np.asarray(s)
    mids = tab.mids
    rows = tab.rows
    if t >= mids[-1]:
        j0 = j1 = len(rows) - 1
        w = 0.0
    elif t <= mids[0]:
        j0 = j1 = 0
        w = 0.0
    else:
        j1 = int(np.searchsorted(mids, t, side="right"))
        j0 = j1 - 1
        w = (t - mids[j0]) / (mids[j1] - mids[j0])
    a = rows[j0].cell(m, s)
    if j1 != j0 and w > 0:
        b = rows[j1].cell(m, s)
        out = tuple((1 - w) * x + w * y for x, y in zip(a, b))
    else:
        out = a
    if (not tab.arm.exact_end) and t > mids[-1]:      # A0: CFB fade to zero over the last 60 s unless tied
        f = min(1.0, (t - mids[-1]) / 60.0)
        tied = (m == 0)[:, None]
        zr, zt = _delta(RM_N, RM_OFF), _delta(RT_N, 0)
        rm = np.where(tied, out[0], (1 - f) * out[0] + f * zr[None, :])
        rt = np.where(tied, out[1], (1 - f) * out[1] + f * zt[None, :])
        out = (rm, rt, out[2], out[3])
    return out


def win_prob(rm: np.ndarray, m: np.ndarray) -> np.ndarray:
    """P(m + remaining margin > 0), floored/capped at per-mille resolution."""
    cdf = np.cumsum(rm, axis=1)
    idx = np.clip(-np.asarray(m) + RM_OFF, -1, RM_N - 1)        # P(rem <= -m) = cdf[idx]
    c = np.where(idx < 0, 0.0, cdf[np.arange(len(m)), np.maximum(idx, 0)])
    return np.clip(1.0 - c, PMIN, PMAX)


def crps_discrete(pmf: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    F = np.cumsum(pmf, axis=1)
    L = pmf.shape[1]
    heav = (np.arange(L)[None, :] >= np.asarray(y_idx)[:, None]).astype(np.float64)
    return ((F - heav) ** 2).sum(axis=1)
