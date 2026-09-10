"""
rotation_adapter.py -- `rotation.RotationSampler.next_lineup` vectorised over
N simulations and both teams at once.

WHY THIS FILE EXISTS, AND WHAT IT IS ALLOWED TO BE. `cbb_sim.models.rotation`
ships `RotationSampler.next_lineup(state)`, which is the engine contract, but it
holds ONE team of ONE game in Python objects and draws from a sequential
`np.random.Generator`. The F2 run is 5,710 games x 200 seeds x ~137 possessions
x 2 teams, about 3.1e8 calls: a Python-object sampler cannot be the engine's
rotation layer at that scale, and `rotation.py` is READ-ONLY for this
deliverable.

So this module re-expresses the SAME DECISION RULE over arrays of shape
(2N, n_slots) -- both teams of every simulation in one block. Line for line it
is `RotationSampler.next_lineup`:

  * credit the possession just played to the five who were on the floor;
  * decay the on-floor-rate EWMA by exp(-d / ema_horizon) and add the
    five back in;
  * draw each on-floor player's personal foul from `prior.fpm * d/60 *
    fit.foul_rate_scale`;
  * weight = targets * tilt.state[rank_bucket(srank), time_bucket, margin_bucket]
    * tilt.foul[min(fouls, 5), time_bucket], zeroed at five fouls;
  * utility = 5q - ema + lam_deficit * (path - played) / (5 * ema_horizon);
  * reshuffle to the five highest q at a period boundary;
  * then the sticky swap: j-th worst on the floor against j-th best on the
    bench, swap while `u[bench] - u[on] > swap_threshold`, stop at the first j
    that fails -- the `break` semantics reproduced as a cumulative mask.

TWO DELIBERATE DIVERGENCES FROM `rotation.py`, both stated rather than hidden:

  1. RNG. `rotation.game_stream` builds a sequential `PCG64` per game. The
     engine's rule (CLAUDE.md, and deliverable 2's stream discipline) is the
     counter-based (seed, game_id, family, ordinal) stream, so availability,
     the Dirichlet targets and the foul hazard are drawn from families
     "rotation" and "rotation_foul" through `engine.rng.StreamBook`. The
     DECISION RULE is identical; the realised draws are not the same numbers
     the offline sampler would produce for the same (seed, game_id). A run is
     still exactly reproducible and still differences game-by-game against a
     paired arm, which is what the RNG rule exists to give.
  2. The foul hazard draws one uniform per ROSTER SLOT and masks to the five on
     the floor, instead of five uniforms for the five. Same distribution per
     player, different stream positions.

Both are recorded in `docs/models/engine/model.md` section 4.5 and in
`run_meta.json` under `provisional_rotation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

from cbb_sim.models.rotation import (
    FOUL_OUT,
    MAX_CANDIDATES,
    RotationFit,
    margin_bucket,
    rank_bucket,
    time_bucket,
)

#: `RotationSampler`'s own basis: five slots for a 40-minute game. An overtime
#: game runs past it, exactly as the offline sampler does -- the target is a
#: long-run minute budget, not a hard cap.
EXPECTED_TOTAL_SECONDS = 2400.0

_NEG = -1e12


def _dirichlet_rows(p: np.ndarray, alpha: float, u: np.ndarray) -> np.ndarray:
    """Row-wise Dirichlet(alpha * p) by inverse-CDF Gamma, the construction
    `usage._gamma_ppf` / `rotation._dirichlet` use (`rng.gamma` there; the exact
    inverse CDF here so the draw is a function of a counter-based uniform).
    Rows whose mass is zero are returned unchanged."""
    a = np.maximum(np.asarray(p, dtype=np.float64) * float(alpha), 1e-6)
    g = special.gammaincinv(a, np.clip(u, 1e-12, 1.0 - 1e-12))
    s = g.sum(axis=1, keepdims=True)
    ok = s[:, 0] > 0
    out = np.array(p, dtype=np.float64, copy=True)
    out[ok] = g[ok] / s[ok]
    return out


def _apply_min_target(s: np.ndarray, avail: np.ndarray, min_share: float) -> np.ndarray:
    """`rotation.apply_min_target`, row-wise."""
    out = np.where(avail, np.maximum(s, float(min_share)), 0.0)
    tot = out.sum(axis=1, keepdims=True)
    bad = tot[:, 0] <= 0
    out[bad] = s[bad]
    tot[bad] = np.maximum(s[bad].sum(axis=1, keepdims=True), 1e-12)
    return out / np.where(tot > 0, tot, 1.0)


@dataclass
class RotationBatch:
    """Rotation state for 2N team-simulations (home rows first, then away)."""

    fit: RotationFit
    n: int                      # number of simulations (so 2n rows)
    srank: np.ndarray           # (2n, S) int16, start-rank order, 1-based
    fpm: np.ndarray             # (2n, S) float64
    avail: np.ndarray           # (2n, S) bool
    targets: np.ndarray         # (2n, S) float64, seconds of the 5-slot budget
    played: np.ndarray          # (2n, S) float64
    path: np.ndarray            # (2n, S) float64
    ema: np.ndarray             # (2n, S) float64
    onmask: np.ndarray          # (2n, S) bool
    q_prev: np.ndarray          # (2n, S) float64
    prev_on: np.ndarray         # (2n, S) bool
    prev_period: np.ndarray     # (2n,) int16
    rankb: np.ndarray           # (2n, S) int64, rank_bucket(srank)
    started: np.ndarray         # (2n,) bool, has a possession been credited yet
    diag: dict

    @property
    def n_slots(self) -> int:
        return self.srank.shape[1]


def init_batch(fit: RotationFit, share: np.ndarray, srank: np.ndarray,
               fpm: np.ndarray, pavail: np.ndarray, book, rows: np.ndarray) -> RotationBatch:
    """Open the rotation for 2N team-simulations.

    `share`/`srank`/`fpm`/`pavail` are (2n, S) gathers of the per-(game, side)
    prior. `book` is the engine `StreamBook`; `rows` maps each of the 2n rows to
    its simulation index so home and away share the simulation's stream (as the
    offline sampler shares one Generator between the two teams).
    """
    two_n, S = share.shape
    u_av = book.draw_block("rotation", rows, S)
    avail = u_av < pavail
    # every team must field five: if the availability draw leaves fewer, the
    # highest-share slots are forced available and the event is counted.
    short = avail.sum(axis=1) < 5
    diag = {"rotation_availability_forced": int(short.sum())}
    if short.any():
        order = np.argsort(-share[short], kind="stable")[:, :5]
        fix = np.zeros_like(avail[short])
        np.put_along_axis(fix, order, True, axis=1)
        avail[short] |= fix

    is_start = (srank <= 5) & avail
    is_bench = avail & ~is_start
    p = np.where(avail, share.astype(np.float64), 0.0)
    ps = p.sum(axis=1, keepdims=True)
    p = p / np.where(ps > 0, ps, 1.0)

    u_fam = book.draw_block("rotation", rows, 2)
    u_dir = book.draw_block("rotation", rows, S)

    p_start = np.where(is_start, p, 0.0).sum(axis=1)
    p_bench = np.where(is_bench, p, 0.0).sum(axis=1)
    hier = (p_start > 0) & (p_bench > 0)

    out = np.zeros((two_n, S), dtype=np.float64)
    if (~hier).any():
        out[~hier] = _dirichlet_rows(p[~hier], fit.alpha, u_dir[~hier])
    if hier.any():
        tot = (p_start + p_bench)[hier]
        fam_p = np.column_stack([p_start[hier] / tot, p_bench[hier] / tot])
        fam = _dirichlet_rows(fam_p, fit.alpha_family, u_fam[hier])
        ps_n = np.where(is_start[hier], p[hier], 0.0) / p_start[hier][:, None]
        pb_n = np.where(is_bench[hier], p[hier], 0.0) / p_bench[hier][:, None]
        ds = _dirichlet_rows(ps_n, fit.alpha_starters, u_dir[hier])
        db = _dirichlet_rows(pb_n, fit.alpha_bench, u_dir[hier])
        blk = np.where(is_start[hier], fam[:, :1] * ds, 0.0) \
            + np.where(is_bench[hier], fam[:, 1:2] * db, 0.0)
        out[hier] = blk
    out = _apply_min_target(out, avail, fit.min_share)
    targets = out * (5.0 * EXPECTED_TOTAL_SECONDS)

    # the opening five: the top of the start order among the available
    key = np.where(avail, srank.astype(np.float64), 1e9)
    pick = np.argsort(key, kind="stable")[:, :5]
    onmask = np.zeros((two_n, S), dtype=bool)
    np.put_along_axis(onmask, pick, True, axis=1)

    return RotationBatch(
        fit=fit, n=two_n // 2, srank=srank.astype(np.int16), fpm=fpm.astype(np.float64),
        avail=avail, targets=targets,
        played=np.zeros((two_n, S)), path=np.zeros((two_n, S)),
        ema=onmask.astype(np.float64), onmask=onmask,
        q_prev=np.zeros((two_n, S)), prev_on=onmask.copy(),
        prev_period=np.ones(two_n, dtype=np.int16),
        rankb=rank_bucket(srank.astype(np.int64)),
        started=np.zeros(two_n, dtype=bool), diag=diag,
    )


def next_lineup(rb: RotationBatch, period: np.ndarray, seconds_remaining: np.ndarray,
                home_score_diff: np.ndarray, last_duration: np.ndarray,
                fouls: np.ndarray, book, rows: np.ndarray,
                live: np.ndarray) -> np.ndarray:
    """Advance the rotation one possession. Returns the (2n, 5) on-floor slots.

    `period`, `seconds_remaining`, `home_score_diff` and `last_duration` are
    (2n,) -- the same simulation value repeated for home and away; `fouls` is
    the (2n, S) view of `GameState.player_fouls` and is UPDATED IN PLACE, so the
    foul-out rule the scheduler enforces and the foul-out the box score reports
    are the same number. `live` selects the rows whose game is still running.
    """
    S = rb.n_slots
    half = rb.n

    # ---- credit the possession just played ------------------------------
    credit = live & rb.started
    if credit.any():
        d = last_duration.astype(np.float64)
        rb.path += (5.0 * d)[:, None] * rb.q_prev * credit[:, None]
        rb.played += d[:, None] * rb.prev_on * credit[:, None]
        decay = np.exp(-d / max(rb.fit.ema_horizon, 1.0))
        dk = np.where(credit, decay, 1.0)[:, None]
        rb.ema = rb.ema * dk + (1.0 - dk) * rb.prev_on
        u_f = np.zeros((len(rb.avail), S))
        sel = np.flatnonzero(credit)
        u_f[sel] = book.draw_block("rotation_foul", rows[sel], S)
        hit = (u_f < rb.fpm * (d[:, None] / 60.0) * rb.fit.foul_rate_scale) \
            & rb.prev_on & credit[:, None]
        fouls += hit.astype(fouls.dtype)

    # ---- the weights, exactly the scheduler's -----------------------------
    margin = np.concatenate([home_score_diff[:half], -home_score_diff[half:]])
    tb = time_bucket(period.astype(np.int64), seconds_remaining.astype(np.int64))
    mb = margin_bucket(margin.astype(np.int64))
    tilt_state = rb.fit.tilt.state[rb.rankb, tb[:, None], mb[:, None]]
    fs = np.minimum(fouls, FOUL_OUT).astype(np.int64)
    tilt_foul = rb.fit.tilt.foul[fs, np.broadcast_to(tb[:, None], fs.shape)]
    w = rb.targets * tilt_state * tilt_foul
    out_of_fouls = fouls >= FOUL_OUT
    w = np.where(out_of_fouls, 0.0, w)
    tot = w.sum(axis=1, keepdims=True)
    dead = tot[:, 0] <= 0
    if dead.any():
        w[dead] = rb.avail[dead].astype(np.float64)
        tot[dead] = np.maximum(w[dead].sum(axis=1, keepdims=True), 1.0)
    q = w / np.where(tot > 0, tot, 1.0)

    u = (5.0 * q - rb.ema
         + rb.fit.lam_deficit * (rb.path - rb.played) / (5.0 * max(rb.fit.ema_horizon, 1.0)))
    u = np.where(rb.avail, u, _NEG)
    u = np.where(out_of_fouls, _NEG, u)

    eligible = rb.avail & ~out_of_fouls
    n_elig = eligible.sum(axis=1)

    # ---- period boundary: reshuffle to the five highest q -----------------
    newper = live & (period.astype(np.int16) != rb.prev_period) & (n_elig >= 5)
    if newper.any():
        key = np.where(eligible[newper], q[newper], -1.0)
        pick = np.argsort(-key, kind="stable")[:, :5]
        nm = np.zeros_like(rb.onmask[newper])
        np.put_along_axis(nm, pick, True, axis=1)
        rb.onmask[newper] = nm
        rb.ema[newper] = np.where(nm, np.maximum(rb.ema[newper], 0.5), rb.ema[newper])
    rb.prev_period = np.where(live, period.astype(np.int16), rb.prev_period)

    # ---- the sticky swap, `break` as a cumulative mask --------------------
    on_order = np.argsort(np.where(rb.onmask, u, np.inf), kind="stable")[:, :5]
    bench_ok = ~rb.onmask & eligible
    n_bench = bench_ok.sum(axis=1)
    bench_order = np.argsort(np.where(bench_ok, u, _NEG * 10), kind="stable")
    bench_order = bench_order[:, ::-1][:, :5]
    rows_idx = np.arange(len(u))
    go = live.copy()
    for j in range(5):
        if not go.any():
            break
        a = on_order[:, j]
        b = bench_order[:, j]
        ua = u[rows_idx, a]
        ub = u[rows_idx, b]
        keep_a = (fouls[rows_idx, a] < FOUL_OUT) & rb.avail[rows_idx, a]
        brk = keep_a & ~((ub - ua) > rb.fit.swap_threshold)
        do = go & (j < n_bench) & ~brk
        if do.any():
            sel = np.flatnonzero(do)
            rb.onmask[sel, a[sel]] = False
            rb.onmask[sel, b[sel]] = True
        go = do

    # ---- the five-on-the-floor invariant ----------------------------------
    cnt = rb.onmask.sum(axis=1)
    bad = live & (cnt != 5)
    if bad.any():
        rb.diag["rotation_five_repaired"] = rb.diag.get("rotation_five_repaired", 0) + int(bad.sum())
        pick = np.argsort(-u[bad], kind="stable")[:, :5]
        nm = np.zeros_like(rb.onmask[bad])
        np.put_along_axis(nm, pick, True, axis=1)
        rb.onmask[bad] = nm

    rb.q_prev = np.where(live[:, None], q, rb.q_prev)
    rb.prev_on = np.where(live[:, None], rb.onmask, rb.prev_on)
    rb.started |= live
    return np.argsort(~rb.onmask, kind="stable")[:, :5].astype(np.int16)


def assert_max_candidates(n_slots: int) -> None:
    if n_slots > MAX_CANDIDATES:
        raise ValueError(
            f"n_slots={n_slots} exceeds rotation.MAX_CANDIDATES={MAX_CANDIDATES}; the "
            "fitted p_play / role_prior tables are only defined to that rank")
