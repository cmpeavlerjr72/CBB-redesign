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

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import special

from cbb_sim.models import rotation_v4 as V4
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


# ===========================================================================
# ROTATION ROUND 3b: S1 as the TRAINING SCHEME, served per game
# ===========================================================================
#: Rotation round 3b adopted **S1** as the L4 rotation model's training scheme
#: (`docs/models/rotation/experiments.md` section 9.4): no gate cell regressed
#: beyond its floor, six improved beyond theirs, and every improvement moved
#: toward the actual. Its own practical note said what the engine needed:
#:
#:   "The engine's rotation adapter currently loads ONE fit for a whole run
#:    (`ad.rot_fit`). Running S1 in the engine needs the adapter to select a fit
#:    per game by month. Because every fitted object S1 varies (tilt tables,
#:    Dirichlet concentrations, `p_play`, `fpm`, the scheduler parameters) is a
#:    lookup table or a scalar, this is a gather, not a new decision rule."
#:
#: `FitSet` is that gather, and nothing else. The DECISION RULE in
#: `init_batch` / `next_lineup` is unchanged line for line; every place that
#: read one scalar off `rb.fit` now reads a per-ROW value that is that row's
#: game's fit, and every place that indexed one tilt table now indexes that
#: game's table. With a single fit the per-row arrays are constant and the
#: arithmetic is identical, which is what keeps `ENGINE_ROTATION_SCHEME=static`
#: bit-for-bit reproducible against every gate report before 2026-09-11.
#:
#: SCOPE, STATED: S1 reaches what `RotationBatch` consumes -- the Dirichlet
#: concentrations, `min_share`, the scheduler parameters and the two tilt
#: tables. It does NOT yet reach the as-of PRIOR construction (`k0`,
#: `role_prior`, `p_play`, the fpm shrinkage, `tail_ratio`, `w_dnp`), which
#: `scripts/build_engine_inputs.py` bakes into the input arrays once per run
#: from the static fit. That half is recorded in `run_meta.json` as
#: `rotation_s1_scope` rather than left for a reader to discover.

_FIT_SCALARS = ("alpha", "alpha_family", "alpha_starters", "alpha_bench",
                "min_share", "ema_horizon", "lam_deficit", "swap_threshold",
                "foul_rate_scale")


@dataclass
class FitSet:
    """K dated `RotationFit`s plus the per-row choice among them."""

    fits: tuple[RotationFit, ...]
    seg: np.ndarray                  # (2n,) int64, which fit each row's game uses
    scalars: dict                    # name -> (2n,) float64
    tilt_state: np.ndarray           # (K, R, T, M)
    tilt_foul: np.ndarray            # (K, F, T)
    provenance: dict

    @classmethod
    def build(cls, fits, seg: np.ndarray, provenance: dict | None = None) -> "FitSet":
        fits = tuple(fits)
        seg = np.asarray(seg, dtype=np.int64)
        scal = {k: np.array([float(getattr(f, k)) for f in fits],
                            dtype=np.float64)[seg] for k in _FIT_SCALARS}
        st = np.stack([np.asarray(f.tilt.state, dtype=np.float64) for f in fits])
        fo = np.stack([np.asarray(f.tilt.foul, dtype=np.float64) for f in fits])
        return cls(fits=fits, seg=seg, scalars=scal, tilt_state=st, tilt_foul=fo,
                   provenance=dict(provenance or {}))

    @classmethod
    def single(cls, fit: RotationFit, two_n: int) -> "FitSet":
        """A static fit is a schedule of length one, the same way a static
        artifact is a manifest of length one (`engine/manifest.py`)."""
        return cls.build([fit], np.zeros(int(two_n), dtype=np.int64),
                         {"scheme": "static", "n_fits": 1})

    def col(self, name: str) -> np.ndarray:
        return self.scalars[name]


def _dirichlet_rows(p: np.ndarray, alpha, u: np.ndarray) -> np.ndarray:
    """Row-wise Dirichlet(alpha * p) by inverse-CDF Gamma, the construction
    `usage._gamma_ppf` / `rotation._dirichlet` use (`rng.gamma` there; the exact
    inverse CDF here so the draw is a function of a counter-based uniform).
    Rows whose mass is zero are returned unchanged.

    `alpha` is a scalar or one value PER ROW (the S1 schedule); a scalar is the
    per-row case with every row equal, so the two paths are the same code."""
    al = np.asarray(alpha, dtype=np.float64)
    if al.ndim == 1:
        al = al[:, None]
    a = np.maximum(np.asarray(p, dtype=np.float64) * al, 1e-6)
    g = special.gammaincinv(a, np.clip(u, 1e-12, 1.0 - 1e-12))
    s = g.sum(axis=1, keepdims=True)
    ok = s[:, 0] > 0
    out = np.array(p, dtype=np.float64, copy=True)
    out[ok] = g[ok] / s[ok]
    return out


def _apply_min_target(s: np.ndarray, avail: np.ndarray, min_share) -> np.ndarray:
    """`rotation.apply_min_target`, row-wise. `min_share` is a scalar or one
    value per row (the S1 schedule)."""
    ms = np.asarray(min_share, dtype=np.float64)
    if ms.ndim == 1:
        ms = ms[:, None]
    out = np.where(avail, np.maximum(s, ms), 0.0)
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
    freeze: bool = False        # ENGINE_ROTATION_FREEZE (Decision 10)
    #: the S1 schedule: per-row scheduler scalars and per-row tilt tables.
    #: A static run carries `FitSet.single`, which is the same arithmetic.
    fitset: FitSet | None = None

    @property
    def n_slots(self) -> int:
        return self.srank.shape[1]


def init_batch(fit: RotationFit, share: np.ndarray, srank: np.ndarray,
               fpm: np.ndarray, pavail: np.ndarray, book, rows: np.ndarray,
               round4: dict | None = None, fitset: FitSet | None = None):
    """Open the rotation for 2N team-simulations.

    `share`/`srank`/`fpm`/`pavail` are (2n, S) gathers of the per-(game, side)
    prior. `book` is the engine `StreamBook`; `rows` maps each of the 2n rows to
    its simulation index so home and away share the simulation's stream (as the
    offline sampler shares one Generator between the two teams).

    `fitset` is the S1 schedule (one `RotationFit` per dated refit plus the
    per-row choice). `None` means the static fit, which is built here as a
    schedule of length one so there is exactly one code path.
    """
    if round4 is not None:
        return init_batch_round4(fit, share, srank, fpm, pavail, book, rows, round4)
    two_n, S = share.shape
    fs = fitset if fitset is not None else FitSet.single(fit, two_n)
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
        out[~hier] = _dirichlet_rows(p[~hier], fs.col("alpha")[~hier], u_dir[~hier])
    if hier.any():
        tot = (p_start + p_bench)[hier]
        fam_p = np.column_stack([p_start[hier] / tot, p_bench[hier] / tot])
        fam = _dirichlet_rows(fam_p, fs.col("alpha_family")[hier], u_fam[hier])
        ps_n = np.where(is_start[hier], p[hier], 0.0) / p_start[hier][:, None]
        pb_n = np.where(is_bench[hier], p[hier], 0.0) / p_bench[hier][:, None]
        ds = _dirichlet_rows(ps_n, fs.col("alpha_starters")[hier], u_dir[hier])
        db = _dirichlet_rows(pb_n, fs.col("alpha_bench")[hier], u_dir[hier])
        blk = np.where(is_start[hier], fam[:, :1] * ds, 0.0) \
            + np.where(is_bench[hier], fam[:, 1:2] * db, 0.0)
        out[hier] = blk
    out = _apply_min_target(out, avail, fs.col("min_share"))
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
        freeze=freeze_enabled(), fitset=fs,
    )


def next_lineup(rb, period: np.ndarray, seconds_remaining: np.ndarray,
                home_score_diff: np.ndarray, last_duration: np.ndarray,
                fouls: np.ndarray, book, rows: np.ndarray,
                live: np.ndarray, prev_end: np.ndarray | None = None,
                team_fouls: np.ndarray | None = None) -> np.ndarray:
    """Advance the rotation one possession. Returns the (2n, 5) on-floor slots.

    `period`, `seconds_remaining`, `home_score_diff` and `last_duration` are
    (2n,) -- the same simulation value repeated for home and away; `fouls` is
    the (2n, S) view of `GameState.player_fouls` and is UPDATED IN PLACE, so the
    foul-out rule the scheduler enforces and the foul-out the box score reports
    are the same number. `live` selects the rows whose game is still running.
    """
    if isinstance(rb, Round4Batch):
        if prev_end is None or team_fouls is None:
            raise ValueError("ENGINE_ROTATION=round4 needs `prev_end` and "
                             "`team_fouls`; the loop carries both")
        return next_lineup_round4(rb, period, seconds_remaining, home_score_diff,
                                  last_duration, fouls, book, rows, live,
                                  prev_end, team_fouls)
    S = rb.n_slots
    half = rb.n

    # ---- credit the possession just played ------------------------------
    credit = live & rb.started
    if credit.any():
        d = last_duration.astype(np.float64)
        rb.path += (5.0 * d)[:, None] * rb.q_prev * credit[:, None]
        rb.played += d[:, None] * rb.prev_on * credit[:, None]
        decay = np.exp(-d / np.maximum(_fs(rb).col("ema_horizon"), 1.0))
        dk = np.where(credit, decay, 1.0)[:, None]
        rb.ema = rb.ema * dk + (1.0 - dk) * rb.prev_on
        u_f = np.zeros((len(rb.avail), S))
        sel = np.flatnonzero(credit)
        u_f[sel] = book.draw_block("rotation_foul", rows[sel], S)
        hit = (u_f < rb.fpm * (d[:, None] / 60.0)
               * _fs(rb).col("foul_rate_scale")[:, None]) \
            & rb.prev_on & credit[:, None]
        fouls += hit.astype(fouls.dtype)

    # ---- the weights, exactly the scheduler's -----------------------------
    margin = np.concatenate([home_score_diff[:half], -home_score_diff[half:]])
    # Decision 10: with the freeze on, the ROTATION MODEL sees the pregame
    # margin and no fouls; the foul-out rule below still reads the real counts.
    f_model = fouls
    if getattr(rb, "freeze", False):
        margin = np.zeros_like(margin)
        f_model = np.zeros_like(fouls)
    tb = time_bucket(period.astype(np.int64), seconds_remaining.astype(np.int64))
    mb = margin_bucket(margin.astype(np.int64))
    # S1: the tilt tables are indexed by the row's own game's refit as well as
    # by (rank bucket, time bucket, margin bucket). With one fit `seg` is all
    # zeros and this is the single-table lookup it replaces.
    sch = _fs(rb)
    seg = sch.seg[:, None]
    tilt_state = sch.tilt_state[seg, rb.rankb, tb[:, None], mb[:, None]]
    fs = np.minimum(f_model, FOUL_OUT).astype(np.int64)
    tilt_foul = sch.tilt_foul[seg, fs, np.broadcast_to(tb[:, None], fs.shape)]
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
         + sch.col("lam_deficit")[:, None] * (rb.path - rb.played)
         / (5.0 * np.maximum(sch.col("ema_horizon"), 1.0)[:, None]))
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
        brk = keep_a & ~((ub - ua) > sch.col("swap_threshold"))
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


def _fs(rb) -> FitSet:
    """The row's schedule. A batch opened before S1 existed (or a hand-built
    one in a test) carries none, and a single fit IS a schedule of length one --
    the same rule `engine/manifest.py` applies to artifacts."""
    if rb.fitset is None:
        rb.fitset = FitSet.single(rb.fit, len(rb.srank))
    return rb.fitset


def assert_max_candidates(n_slots: int) -> None:
    if n_slots > MAX_CANDIDATES:
        raise ValueError(
            f"n_slots={n_slots} exceeds rotation.MAX_CANDIDATES={MAX_CANDIDATES}; the "
            "fitted p_play / role_prior tables are only defined to that rank")


# ===========================================================================
# Decision 10: the rotation state freeze
# ===========================================================================
#: `ENGINE_ROTATION_FREEZE=1` holds, FOR THE ROTATION MODEL ONLY, the margin at
#: its pregame value (0) and the personal- and team-foul counts at theirs (0).
#: Foul accrual, the foul-out eviction rule and the box-score foul counters stay
#: live -- a player with five fouls still leaves the floor, because that is a
#: rule of the game and not a model feature.
#:
#: WHY: Decision 10 and L23. `margin` and `fouls` are produced by the engine and
#: consumed by the rotation, which closes a feedback loop offline scoring cannot
#: see. A paired-stream run with the feature live vs frozen must keep margin SD
#: ratio, home/away score correlation and possessions per game inside the G1/G2
#: tolerances. The incumbent R2 consumes both through `TiltTables.state` and
#: `TiltTables.foul` and has never had this check run (L25).


def freeze_enabled() -> bool:
    return os.environ.get("ENGINE_ROTATION_FREEZE", "0") not in ("", "0", "false", "False")


def rotation_mode() -> str:
    return os.environ.get("ENGINE_ROTATION", "reference")


# ===========================================================================
# ENGINE_ROTATION=round4 -- the per-player substitution-hazard sampler
# ===========================================================================
#: `rotation_v4.run_sub_hazard` re-expressed over (2N, S) arrays. The DECISION
#: RULE is identical line for line:
#:
#:   * exits are independent Bernoulli draws on `p_out` for the five on the
#:     floor, with a five-foul player forced out;
#:   * entrants are the `n_exit` smallest keys of an Efraimidis-Spirakis
#:     exponential race, `key = -log(u) / w` with `w = p_in / (1 - p_in)`, over
#:     the eligible bench -- weighted sampling without replacement, one uniform
#:     per roster slot and an argsort;
#:   * `state_min` resets for every slot whose on/off state changed;
#:   * `half_min` resets when the half changes;
#:   * H1/H3 additionally reset the five to the predicted starters at the first
#:     possession of period 2; H2 does not.
#:
#: The feature block is `rotation_v4.design()` itself, called with M = 2N -- the
#: offline sampler calls the same function with M = 1, so the two cannot drift.
#:
#: TWO STATED DIVERGENCES beyond the two of `docs/models/engine/model.md`
#: section 4.5:
#:   3. the race draws one uniform per roster slot from family "rotation_sub";
#:      the offline sampler draws `len(bench)` sequential uniforms. Same rule,
#:      different stream positions.
#:   4. `team_fouls` carries over into overtime in the engine (NCAA rule,
#:      `state.py`) while the possessions table the hazards were fitted on
#:      resets it at every period. This affects 0.7% of possessions through one
#:      weak linear term and is recorded rather than silently reconciled.


@dataclass
class Round4Batch:
    """Rotation state for 2N team-simulations under the round-4 hazard family."""

    fit: RotationFit
    n: int
    srank: np.ndarray           # (2n, S) int16
    share: np.ndarray           # (2n, S) float64, as-of share x 5
    fpm: np.ndarray             # (2n, S) float64
    avail: np.ndarray           # (2n, S) bool
    is_starter: np.ndarray      # (2n, S) float64
    played: np.ndarray          # (2n, S) float64 seconds
    half_min: np.ndarray        # (2n, S) float64 minutes this half
    state_min: np.ndarray       # (2n, S) float64 minutes in the current state
    onmask: np.ndarray          # (2n, S) bool
    prev_on: np.ndarray         # (2n, S) bool
    prev_period: np.ndarray     # (2n,) int16
    prev_half: np.ndarray       # (2n,) int8
    w_out: np.ndarray           # (2n, F) float64
    b_out: np.ndarray           # (2n,) float64
    w_in: np.ndarray            # (2n, F) float64
    b_in: np.ndarray            # (2n,) float64
    hard_reset: bool
    started: np.ndarray         # (2n,) bool
    freeze: bool
    diag: dict

    @property
    def n_slots(self) -> int:
        return self.srank.shape[1]


def init_batch_round4(fit: RotationFit, share: np.ndarray, srank: np.ndarray,
                      fpm: np.ndarray, pavail: np.ndarray, book, rows: np.ndarray,
                      coefs: dict) -> Round4Batch:
    """Open the round-4 rotation for 2N team-simulations.

    `coefs` carries the per-row hazard coefficients already gathered from the S1
    manifest (`w_out`, `b_out`, `w_in`, `b_in`) plus `hard_reset`."""
    two_n, S = share.shape
    u_av = book.draw_block("rotation", rows, S)
    avail = u_av < pavail
    short = avail.sum(axis=1) < 5
    diag = {"rotation_availability_forced": int(short.sum())}
    if short.any():
        order = np.argsort(-share[short], kind="stable")[:, :5]
        fix = np.zeros_like(avail[short])
        np.put_along_axis(fix, order, True, axis=1)
        avail[short] |= fix

    # `share` is normalised over ALL candidates, not only the available ones,
    # because `rotation_v4.run_sub_hazard` normalises `prior.share` the same way
    # -- the feature is a role prior, not a budget over tonight's roster.
    sh = share.astype(np.float64)
    tot = sh.sum(axis=1, keepdims=True)
    sh = sh / np.where(tot > 0, tot, 1.0) * 5.0

    key = np.where(avail, srank.astype(np.float64), 1e9)
    pick = np.argsort(key, kind="stable")[:, :5]
    onmask = np.zeros((two_n, S), dtype=bool)
    np.put_along_axis(onmask, pick, True, axis=1)

    return Round4Batch(
        fit=fit, n=two_n // 2, srank=srank.astype(np.int16), share=sh,
        fpm=fpm.astype(np.float64), avail=avail,
        is_starter=(srank <= 5).astype(np.float64),
        played=np.zeros((two_n, S)), half_min=np.zeros((two_n, S)),
        state_min=np.zeros((two_n, S)), onmask=onmask, prev_on=onmask.copy(),
        prev_period=np.ones(two_n, dtype=np.int16),
        prev_half=np.zeros(two_n, dtype=np.int8),
        w_out=coefs["w_out"], b_out=coefs["b_out"],
        w_in=coefs["w_in"], b_in=coefs["b_in"],
        hard_reset=bool(coefs.get("hard_reset", True)),
        started=np.zeros(two_n, dtype=bool),
        freeze=freeze_enabled(), diag=diag)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def next_lineup_round4(rb: Round4Batch, period: np.ndarray,
                       seconds_remaining: np.ndarray, home_score_diff: np.ndarray,
                       last_duration: np.ndarray, fouls: np.ndarray, book,
                       rows: np.ndarray, live: np.ndarray, prev_end: np.ndarray,
                       team_fouls: np.ndarray) -> np.ndarray:
    S = rb.n_slots
    half = rb.n
    two_n = 2 * half

    # ---- credit the possession just played ------------------------------
    credit = live & rb.started
    if credit.any():
        d = last_duration.astype(np.float64)
        c = credit[:, None]
        rb.played += d[:, None] * rb.prev_on * c
        rb.half_min += (d[:, None] / 60.0) * rb.prev_on * c
        rb.state_min += (d[:, None] / 60.0) * c
        u_f = np.zeros((two_n, S))
        sel = np.flatnonzero(credit)
        u_f[sel] = book.draw_block("rotation_foul", rows[sel], S)
        hit = (u_f < rb.fpm * (d[:, None] / 60.0) * rb.fit.foul_rate_scale)             & rb.prev_on & c
        fouls += hit.astype(fouls.dtype)

    # ---- the half boundary resets minutes-in-half ------------------------
    cur_half = (period >= 2).astype(np.int8)
    newhalf = live & (cur_half != rb.prev_half)
    if newhalf.any():
        rb.half_min[newhalf] = 0.0
    rb.prev_half = np.where(live, cur_half, rb.prev_half)

    # ---- the state the rotation model sees -------------------------------
    margin = np.concatenate([home_score_diff[:half], -home_score_diff[half:]])
    f_model = fouls.astype(np.float64)
    tf_model = np.asarray(team_fouls, dtype=np.float64)
    if rb.freeze:                       # Decision 10
        margin = np.zeros_like(margin)
        f_model = np.zeros_like(f_model)
        tf_model = np.zeros_like(tf_model)

    X = V4.design(rb.is_starter, rb.share, f_model, rb.state_min, rb.half_min,
                  period, seconds_remaining, margin, prev_end, tf_model)

    out_of_fouls = fouls >= FOUL_OUT
    eligible = rb.avail & ~out_of_fouls
    n_elig = eligible.sum(axis=1)

    p_out = _sigmoid(np.einsum("msf,mf->ms", X, rb.w_out) + rb.b_out[:, None])
    p_in = _sigmoid(np.einsum("msf,mf->ms", X, rb.w_in) + rb.b_in[:, None])
    p_out = np.where(out_of_fouls, 1.0, p_out)

    u_sub = book.draw_block("rotation_sub", rows, 2 * S)
    u_exit, u_race = u_sub[:, :S], u_sub[:, S:]

    on = rb.onmask
    bench_ok = ~rb.onmask & eligible
    n_bench = bench_ok.sum(axis=1)

    exits = (u_exit < p_out) & on
    n_drawn = exits.sum(axis=1)
    n_exit = np.minimum(n_drawn, n_bench)

    # who leaves: the drawn exits, fouled-out first, capped at `n_exit`
    rank_key = np.where(exits, -out_of_fouls.astype(np.float64), np.inf)
    order_out = np.argsort(rank_key, kind="stable")
    rank_out = np.empty_like(order_out)
    np.put_along_axis(rank_out, order_out, np.arange(S)[None, :].repeat(two_n, 0), axis=1)
    leaving = exits & (rank_out < n_exit[:, None])

    # who enters: the `n_exit` smallest race keys over the eligible bench
    w = np.clip(p_in, 1e-9, 1.0 - 1e-9)
    w = w / (1.0 - w)
    key = np.where(bench_ok, -np.log(np.clip(u_race, 1e-12, 1.0)) / np.maximum(w, 1e-12),
                   np.inf)
    order_in = np.argsort(key, kind="stable")
    rank_in = np.empty_like(order_in)
    np.put_along_axis(rank_in, order_in, np.arange(S)[None, :].repeat(two_n, 0), axis=1)
    entering = bench_ok & (rank_in < n_exit[:, None])

    act = live & (on.sum(axis=1) == 5) & (n_bench > 0)
    if act.any():
        a = act[:, None]
        rb.onmask = np.where(a, (rb.onmask & ~leaving) | entering, rb.onmask)

    # ---- the hard second-half reset (H1 / H3 only) -----------------------
    if rb.hard_reset:
        newper = live & (period.astype(np.int16) == 2) & (rb.prev_period == 1)             & (n_elig >= 5)
        if newper.any():
            k = np.where(eligible[newper], rb.srank[newper].astype(np.float64), 1e9)
            pick = np.argsort(k, kind="stable")[:, :5]
            nm = np.zeros_like(rb.onmask[newper])
            np.put_along_axis(nm, pick, True, axis=1)
            rb.onmask[newper] = nm
    rb.prev_period = np.where(live, period.astype(np.int16), rb.prev_period)

    # ---- the five-on-the-floor invariant and the foul-out rule -----------
    cnt = rb.onmask.sum(axis=1)
    bad = live & (cnt != 5)
    if bad.any():
        rb.diag["rotation_five_repaired"] = rb.diag.get("rotation_five_repaired", 0)             + int(bad.sum())
        score = np.where(eligible[bad], rb.share[bad], _NEG)             + 10.0 * rb.onmask[bad].astype(np.float64)
        pick = np.argsort(-score, kind="stable")[:, :5]
        nm = np.zeros_like(rb.onmask[bad])
        np.put_along_axis(nm, pick, True, axis=1)
        rb.onmask[bad] = nm

    stuck = rb.onmask & out_of_fouls
    if stuck.any():
        for _ in range(2):
            rowsel = np.flatnonzero(stuck.any(axis=1) & live)
            if not len(rowsel):
                break
            for r in rowsel:
                bad_slots = np.flatnonzero(stuck[r])
                pool = np.flatnonzero(~rb.onmask[r] & eligible[r])
                pool = pool[np.argsort(-rb.share[r][pool], kind="stable")]
                for aa, bb in zip(bad_slots, pool):
                    rb.onmask[r, aa] = False
                    rb.onmask[r, bb] = True
            stuck = rb.onmask & out_of_fouls

    changed = rb.onmask != rb.prev_on
    rb.state_min = np.where(changed & live[:, None], 0.0, rb.state_min)
    rb.prev_on = np.where(live[:, None], rb.onmask, rb.prev_on)
    rb.started |= live
    return np.argsort(~rb.onmask, kind="stable")[:, :5].astype(np.int16)


# ===========================================================================
# R2 under S1: the manifest, gathered per simulation row
# ===========================================================================
R2_S1_MANIFEST = Path("data/processed/models/engine/rotation_r2_s1_F2_2025.json")


def rotation_scheme() -> str:
    """`ENGINE_ROTATION_SCHEME`: `s1` (round 3b's adopted scheme, the default
    since 2026-09-11) or `static` (the single `rotation_fit.json`, kept so every
    gate report before that date reproduces)."""
    return os.environ.get("ENGINE_ROTATION_SCHEME", "s1")


def load_r2_s1(games, manifest_path: Path | None = None) -> dict:
    """Load the R2 S1 schedule and return the fits plus the per-GAME choice.

    Selection goes through `engine.manifest.ArtifactManifest`, so the
    honest-backtest rule (`refit_date < tipoff` AND `max_train_date <
    game_date`) is enforced in the one place every sub-model passes through --
    not re-implemented here."""
    from cbb_sim.engine.manifest import ArtifactManifest

    p = Path(manifest_path or R2_S1_MANIFEST)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing; rotation round 3b adopted S1 as the scheme "
            "(experiments.md s9.4) and the engine serves it from this manifest. "
            "Set ENGINE_ROTATION_SCHEME=static to run the single rotation_fit.json.")
    man = ArtifactManifest.from_json(p, games)
    fits = tuple(RotationFit.from_json(e.path) for e in man.entries)
    for f in fits:
        if f.tilt is None:
            raise ValueError("an R2 S1 fit carries no tilt tables; R2 is the "
                             "hierarchical Dirichlet PLUS the fitted scheduler")
    return {"manifest": man, "fits": fits, "seg_of_game": man.seg_of_game,
            "provenance": man.provenance()}


def r2_s1_fitset(r2: dict, game_rows: np.ndarray) -> FitSet:
    """The (2N) `FitSet` for one chunk: home rows then away rows, each row
    carrying its own game's refit."""
    seg = r2["seg_of_game"][np.asarray(game_rows, dtype=np.int64)]
    return FitSet.build(r2["fits"], seg, r2["provenance"])


# ===========================================================================
# round-4 artifacts: the S1 manifest, gathered per simulation row
# ===========================================================================
ROUND4_DIR = Path("data/processed/models/rotation")
ROUND4_MANIFEST = ROUND4_DIR / "rotation_v4_manifest.json"


def load_round4(games, hard_reset: bool | None = None,
                manifest_path: Path | None = None) -> dict:
    """Load the round-4 S1 hazard schedule and return per-GAME coefficients.

    Selection is `engine.manifest.ArtifactManifest`, so the honest-backtest rule
    (`refit_date < tipoff` AND `max_train_date < game_date`) is enforced in the
    one place every sub-model passes through."""
    from cbb_sim.engine.manifest import ArtifactManifest
    from cbb_sim.models.rotation_v4 import SubHazardFit

    p = Path(manifest_path or ROUND4_MANIFEST)
    man = ArtifactManifest.from_json(p, games)
    fits = [SubHazardFit.from_json(e.path) for e in man.entries]
    for f in fits:
        if f.kind != "logistic":
            raise NotImplementedError(
                "ENGINE_ROTATION=round4 serves logistic hazards only; a LightGBM "
                "hazard would be a live model call in the sim loop, which "
                "CLAUDE.md bans. Discretise the booster into a lookup table first.")
    W_out = np.array([f.out_coef for f in fits], dtype=np.float64)
    B_out = np.array([f.out_intercept for f in fits], dtype=np.float64)
    W_in = np.array([f.in_coef for f in fits], dtype=np.float64)
    B_in = np.array([f.in_intercept for f in fits], dtype=np.float64)
    hr = hard_reset
    if hr is None:
        hr = os.environ.get("ENGINE_ROTATION_ARM", "H1") != "H2"
    return {"manifest": man, "seg_of_game": man.seg_of_game,
            "W_out": W_out, "B_out": B_out, "W_in": W_in, "B_in": B_in,
            "hard_reset": bool(hr),
            "provenance": man.provenance()}


def round4_rows(r4: dict, game_rows: np.ndarray) -> dict:
    """Gather the per-row (2N) coefficient block for one chunk."""
    seg = r4["seg_of_game"][np.asarray(game_rows, dtype=np.int64)]
    return {"w_out": r4["W_out"][seg], "b_out": r4["B_out"][seg],
            "w_in": r4["W_in"][seg], "b_in": r4["B_in"][seg],
            "hard_reset": r4["hard_reset"]}
