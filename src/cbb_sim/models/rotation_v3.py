"""
rotation_v3.py -- ROUND-3 rotation arms: R5's donor/override hybrid with a
FITTED close-game keep-starters override.

`cbb_sim.models.rotation` is not modified by this module (another worker is
reading it for engine v0); everything here imports from it and subclasses.

WHY A KEEP OVERRIDE. R5's override is one-sided: it can only *block* a player
onto the bench, never force one onto the floor. The donor sequence therefore
sets the ceiling on the starters' share in every state, and the block floor
`p0` can only push that share down. Round 2 measured the consequence -- the
final-8:00 starters' share is 4-7 pp low in the close and moderate bands while
the blowout band is inside tolerance -- and the block has no way to repair it,
because repairing it means putting a starter back on the floor that the donor
game did not have on. The keep override is the missing half of the same
mechanism, fitted the same way and reported the same way.

TWO FUNCTIONAL FORMS, both fitted, neither hand-set:

  R7 `keep_logistic`  a logistic on-floor propensity fitted on the training
                      season over (fouls, |margin|, time remaining, is_starter,
                      the team prior = the player's as-of target share) and their
                      interactions. The keep score is the propensity DEVIATION
                      from a neutral state (early, tied, no fouls),
                          z_i     = (x_i - x_i^neutral) . w_on
                          p_i     = sigmoid(scale_k * z_i + logit(q0))
                          keep_i  = tol2_i < p_i,  tol2_i ~ U(0,1) once per game
                      so the override is inert in an ordinary state exactly as
                      R5's block is, and `scale_k`, `q0` are fitted on the
                      training season against the four state-dependence cells.

  R8 `keep_cell`      the simpler threshold form: one fitted scalar `theta`
                      against the training season's own starters'-share-by-
                      (time bucket x margin bucket) table `s*`,
                          p_i     = theta * s*[tb, mb]  if i is a predicted starter
                                  = 0                   otherwise
                          keep_i  = tol2_i < p_i
                      No logistic, no player features beyond the starter flag,
                      one knob. It is league-average by construction, which the
                      pre-registered team-quintile slope check is there to
                      expose.

Both forms apply the same displacement rule: a kept player who is not in the
donor-mapped five replaces the worst-ranked member of that five who is not
himself kept and not himself blocked. Neither form can put a fouled-out or
unavailable player on the floor, and neither is allowed to override more than
five players.
"""

from __future__ import annotations

import numpy as np

from cbb_sim.models.rotation import (
    FOUL_OUT,
    N_MARGIN_BUCKETS,
    N_TIME_BUCKETS,
    OVERRIDE_STATE_COLS,
    RotationArm,
    RotationFit,
    _evict_fouled_out,
    _override_design,
    _reassign,
    _state_scalars,
    apply_min_target,
    draw_available,
)

# ---------------------------------------------------------------------------
# R7's keep design
# ---------------------------------------------------------------------------
#: the on-floor propensity feature list. `is_starter`, `target_share`,
#: `period2` and `sec_left_frac` are the NEUTRAL carriers -- present in an
#: ordinary state too -- and are excluded from the deviation, exactly as
#: `OVERRIDE_STATE_COLS` excludes them for the block.
KEEP_FEATURES = [
    "fouls",                            # 0
    "fouls_x_is_starter",               # 1
    "foul_out",                         # 2
    "is_starter",                       # 3   neutral
    "abs_margin",                       # 4
    "abs_margin_x_is_starter",          # 5
    "late",                             # 6
    "late_x_is_starter",                # 7
    "is_close_x_late",                  # 8
    "is_close_x_late_x_is_starter",     # 9
    "abs_margin_x_late",                # 10
    "target_share",                     # 11  neutral
    "target_share_x_is_close_x_late",   # 12
    "target_share_x_late",              # 13
    "period2",                          # 14  neutral
    "sec_left_frac",                    # 15  neutral
]
KEEP_STATE_COLS = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13], dtype="int64")


def keep_design(prior, script, targets, played, fouls, since_change, idx, k, rem):
    """Feature matrix for R7's on-floor propensity, `KEEP_FEATURES` order.

    `share` is the player's as-of target share of the team's slot-seconds -- the
    pre-registration's "team prior" term, and the only place a matchup enters the
    keep score."""
    idx = np.asarray(idx, dtype="int64")
    total, am, late, slf, p2, close = _state_scalars(script, k, rem)
    f = fouls[idx].astype("float64")
    st = np.zeros(prior.n, dtype="float64")
    st[prior.starters()] = 1.0
    st = st[idx]
    share = (targets[idx] / max(total, 1.0)) * 5.0     # ~1.0 for an even split
    m = len(idx)
    cl = close * late
    return np.column_stack([
        f, f * st, (f >= FOUL_OUT).astype("float64"),
        st,
        np.full(m, am), st * am,
        np.full(m, late), st * late,
        np.full(m, cl), st * cl,
        np.full(m, am * late),
        share, share * cl, share * late,
        np.full(m, p2), np.full(m, slf),
    ])


def build_keep_training(tp, feats, fit, game_ids, fouls=None):
    """Rows are (team-game, possession, candidate); the label is "on the floor at
    this possession".

    Reading the training game's own foul events here is legitimate and is never
    done at simulation time -- the same rule `build_hazard_training` states after
    the round-2 defect (`experiments.md` section 5).
    """
    from cbb_sim.models.rotation import (
        SLOTS, actual_foul_matrix, build_priors, build_scripts,
    )

    sel = tp[tp["game_id"].isin(game_ids)]
    scripts = build_scripts(sel)
    priors = build_priors(feats[feats["game_id"].isin(game_ids)], fit)
    actual = {(int(g), int(t)): d[SLOTS].to_numpy(dtype="int64")
              for (g, t), d in sel.groupby(["game_id", "team_id"], sort=False)}
    evg = {}
    if fouls is not None and len(fouls):
        f = fouls[fouls["game_id"].isin(game_ids)]
        evg = {(int(g), int(t)): d for (g, t), d in f.groupby(["game_id", "team_id"],
                                                             sort=False)}
    X, y = [], []
    for key, lu in actual.items():
        if key not in priors or key not in scripts:
            continue
        prior, script = priors[key], scripts[key]
        idx_of = {int(p): i for i, p in enumerate(prior.pids)}
        n = prior.n
        targets = prior.share * script.total_slot
        played = np.zeros(n)
        since = np.zeros(n)
        rem = script.remaining_slot()
        npos = min(len(lu), script.n)
        fm = actual_foul_matrix(evg.get(key), script.period, script.start_clock, idx_of)
        idx_all = np.arange(n, dtype="int64")
        for k in range(npos):
            cur = {idx_of.get(int(p)) for p in lu[k]}
            cur.discard(None)
            fouls_v = fm[k].astype("int64") if k < fm.shape[0] else np.zeros(n, dtype="int64")
            X.append(keep_design(prior, script, targets, played, fouls_v, since,
                                 idx_all, k, rem))
            y.append(np.array([1.0 if i in cur else 0.0 for i in range(n)]))
            d = script.dur[k]
            if cur:
                played[np.asarray(sorted(cur), dtype="int64")] += d
            since += d
    if not X:
        raise RuntimeError("no keep training rows produced")
    return np.vstack(X), np.concatenate(y)


def fit_keep_model(X, y, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", random_state=seed)
    m.fit(X, y)
    return {"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
            "n": int(len(y)), "base_rate": float(np.mean(y)),
            "features": list(KEEP_FEATURES)}


def fit_state_starter_table(tp) -> np.ndarray:
    """R8's `s*`: the training season's own starters' share of on-floor slots in
    each (time bucket x margin bucket) cell. A pure data table -- the same object
    `TiltTables.state` is built from, at the resolution R8's threshold needs."""
    from cbb_sim.models.rotation import SLOTS

    num = np.zeros((N_TIME_BUCKETS, N_MARGIN_BUCKETS), dtype="float64")
    den = np.zeros((N_TIME_BUCKETS, N_MARGIN_BUCKETS), dtype="float64")
    for _key, g in tp.groupby(["game_id", "team_id"], sort=False):
        lu = g[SLOTS].to_numpy(dtype="int64")
        starters = set(int(x) for x in lu[0])
        ids, inv = np.unique(lu, return_inverse=True)
        st = np.array([1.0 if int(i) in starters else 0.0 for i in ids])
        ss = st[inv.reshape(lu.shape)].sum(axis=1)
        tb = g["time_bucket"].to_numpy()
        mb = g["margin_bucket"].to_numpy()
        np.add.at(num, (tb, mb), ss)
        np.add.at(den, (tb, mb), 5.0)
    return np.where(den > 0, num / np.maximum(den, 1.0), np.nan)


# ---------------------------------------------------------------------------
# The shared donor + block + keep simulator
# ---------------------------------------------------------------------------
def run_resample_keep(prior, script, avail, fit, donors, rng, block=None, keep=None):
    """`rotation.run_resample` with a second, fitted override that can put a
    player back ON the floor.

    Line for line the donor splice, the block and the foul accounting are
    `rotation.run_resample`; the only addition is the `keep` block marked below.
    """
    bank = donors.get((prior.game_id, prior.team_id), [])
    n = prior.n
    npos = script.n
    out = np.empty((npos, 5), dtype="int64")
    fouls_hist = np.zeros((npos, n), dtype="int8")
    order_avail = np.array([i for i in np.argsort(prior.rank) if avail[i]], dtype="int64")
    if len(order_avail) < 5 or not bank:
        five = order_avail[:5] if len(order_avail) >= 5 else np.arange(min(5, n))
        out[:] = prior.pids[five]
        return out, fouls_hist

    from cbb_sim.models.rotation import _state_band

    band = _state_band(script.time_bucket, script.margin_bucket)
    rank_of_pid = {int(p): i for i, p in enumerate(prior.pids) if avail[i]}

    donor_rank: list[dict[int, int]] = []
    for d in bank:
        ids, cnt = np.unique(d["lineups"].ravel(), return_counts=True)
        ordering = ids[np.argsort(-cnt)]
        donor_rank.append({int(p): i for i, p in enumerate(ordering)})

    played = np.zeros(n, dtype="float64")
    fouls = np.zeros(n, dtype="int64")
    fscale = fit.foul_rate_scale
    # one "coach tolerance" draw per player per game for EACH override, so a
    # block and a keep are persistent and self-clearing rather than independent
    # coin flips every possession. BOTH are drawn whether or not the arm uses
    # them, so R5, R7 and R8 sit at identical stream positions and the bake-off
    # arms are paired (CLAUDE.md, "paired bake-off arms share aligned streams").
    # The cost is that this R5 is not bit-identical to `rotation.R5Hybrid` -- the
    # decision rule is the same and the unused draw changes only which numbers
    # come out, which is stated in `experiments.md` rather than hidden.
    tol = rng.random(n) if block is not None else None
    tol2 = rng.random(n)
    rem = script.remaining_slot()
    targets = prior.share * script.total_slot
    zeros = np.zeros(n, dtype="float64")
    is_starter = np.zeros(n, dtype=bool)
    is_starter[prior.starters()] = True
    st_f = is_starter.astype("float64")
    # the player's as-of target share of the team's slot-seconds, the keep
    # design's "team prior" column (~1.0 for an even split)
    share_v = (targets / max(script.total_slot, 1.0)) * 5.0
    idx_all = np.arange(n, dtype="int64")
    # worst-first displacement order among the candidates
    worst_first = np.argsort(-prior.rank)

    k = 0
    while k < npos:
        b = int(band[k])
        j = k
        while j < npos and int(band[j]) == b:
            j += 1
        w = np.array([float((d["band"] == b).sum()) for d in bank]) + 1.0
        if w.sum() <= len(bank):
            for alt in sorted(range(5), key=lambda x: abs(x - b)):
                w2 = np.array([float((d["band"] == alt).sum()) for d in bank]) + 1.0
                if w2.sum() > len(bank):
                    w, b = w2, alt
                    break
        di = int(rng.choice(len(bank), p=w / w.sum()))
        donor = bank[di]
        src = np.flatnonzero(donor["band"] == b)
        if len(src) == 0:
            src = np.arange(len(donor["band"]))
        rmap = donor_rank[di]

        blk = j - k
        pos = np.linspace(0, len(src) - 1e-9, blk).astype("int64")
        for t in range(blk):
            lu = donor["lineups"][src[pos[t]]]
            chosen: list[int] = []
            used: set[int] = set()
            for pid in lu:
                idx = rank_of_pid.get(int(pid))
                if idx is None or fouls[idx] >= FOUL_OUT:
                    r = rmap.get(int(pid), len(order_avail) - 1)
                    idx = int(order_avail[min(r, len(order_avail) - 1)])
                if idx in used or fouls[idx] >= FOUL_OUT:
                    for cand in order_avail:
                        if int(cand) not in used and fouls[int(cand)] < FOUL_OUT:
                            idx = int(cand)
                            break
                used.add(int(idx))
                chosen.append(int(idx))
            kk = k + t
            fouls_hist[kk] = np.minimum(fouls, 127)

            ov_out = np.zeros(n, dtype=bool)
            if block is not None:
                X = _override_design(prior, script, targets, played, fouls, zeros,
                                     idx_all, kk, rem)
                z = X[:, OVERRIDE_STATE_COLS] @ block.dw
                pb = 1.0 / (1.0 + np.exp(-(block.scale * z + block.block_logit)))
                ov_out = tol < pb
                ov_out[fouls >= FOUL_OUT] = True
                if (~ov_out & avail).sum() < 5:
                    for i2 in np.argsort(pb):
                        if avail[i2] and fouls[i2] < FOUL_OUT:
                            ov_out[i2] = False
                        if (~ov_out & avail).sum() >= 5:
                            break
                chosen = _reassign(chosen, ov_out, fouls, order_avail)

            # ---- the keep override (round 3) ------------------------------
            if keep is not None:
                pk = keep.p_keep(script, kk, rem, fouls, st_f, share_v, is_starter)
                kp = (tol2 < pk) & avail & (fouls < FOUL_OUT) & ~ov_out
                if kp.sum() > 5:                       # never override past five
                    drop = np.argsort(-pk)[5:]
                    kp[drop] = False
                if kp.any():
                    cur_l = list(chosen[:5])
                    on = set(cur_l)
                    missing = [int(i) for i in np.flatnonzero(kp) if int(i) not in on]
                    if missing:
                        # displace the worst-ranked current member who is not
                        # himself kept, worst first
                        victims = [c for c in worst_first
                                   if int(c) in on and not kp[int(c)]]
                        for i2 in missing:
                            if not victims:
                                break
                            v = int(victims.pop(0))
                            cur_l[cur_l.index(v)] = i2
                        chosen = cur_l

            cur = np.asarray(chosen[:5], dtype="int64")
            if (fouls[cur] >= FOUL_OUT).any():
                m = np.zeros(n, dtype=bool)
                m[cur] = True
                cur = _evict_fouled_out(m, fouls, avail, -prior.rank.astype("float64"))
                if len(cur) != 5:
                    cur = np.asarray(chosen[:5], dtype="int64")
            out[kk] = prior.pids[cur]
            d = script.dur[kk]
            played[cur] += d
            hit = rng.random(5) < prior.fpm[cur] * (d / 60.0) * fscale
            if hit.any():
                fouls[cur[hit]] += 1
        k = j
    return out, fouls_hist


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
class _Block:
    """R5's fitted block override, read off a `RotationFit`."""

    def __init__(self, fit: RotationFit, scale=None, p0=None):
        ex = np.asarray(fit.notes["r5_exit"]["coef"], dtype="float64")
        en = np.asarray(fit.notes["r5_enter"]["coef"], dtype="float64")
        self.dw = (ex - en)[OVERRIDE_STATE_COLS]
        self.scale = float(fit.notes.get("r5_override_scale", 1.0) if scale is None else scale)
        p = float(np.clip(fit.notes.get("r5_block_base", 0.02) if p0 is None else p0,
                          1e-6, 0.5))
        self.block_logit = float(np.log(p / (1.0 - p)))
        self.p0 = p


class _KeepLogistic:
    """R7's fitted keep override."""

    def __init__(self, fit: RotationFit, scale=None, q0=None):
        w = np.asarray(fit.notes["r7_keep"]["coef"], dtype="float64")
        self.w = w[KEEP_STATE_COLS]
        self.scale = float(fit.notes.get("r7_keep_scale", 1.0) if scale is None else scale)
        q = float(np.clip(fit.notes.get("r7_keep_base", 0.05) if q0 is None else q0,
                          1e-6, 0.9))
        self.keep_logit = float(np.log(q / (1.0 - q)))
        self.q0 = q

    def p_keep(self, script, k, rem, fouls, st_f, share_v, is_starter):
        """`sigmoid(scale * z + logit(q0))` with `z` the state deviation of the
        fitted on-floor propensity. Written out term by term rather than through
        `keep_design`, because this runs once per possession per player and the
        column-stack dominates the arm's cost; `test_rotation_v3.py` asserts the
        two agree to 1e-12."""
        _t, am, late, slf, p2, close = _state_scalars(script, k, rem)
        f = fouls.astype("float64")
        fo = (fouls >= FOUL_OUT).astype("float64")
        w = self.w                       # KEEP_STATE_COLS order
        cl = close * late
        const = w[3] * am + w[5] * late + w[7] * cl + w[9] * am * late
        z = (w[0] * f + w[1] * f * st_f + w[2] * fo
             + const
             + st_f * (w[4] * am + w[6] * late + w[8] * cl)
             + share_v * (w[10] * cl + w[11] * late))
        return 1.0 / (1.0 + np.exp(-(self.scale * z + self.keep_logit)))


class _KeepCell:
    """R8's fitted per-cell keep rate."""

    def __init__(self, fit: RotationFit, theta=None):
        self.table = np.asarray(fit.notes["r8_state_starter_table"], dtype="float64")
        self.theta = float(fit.notes.get("r8_keep_theta", 0.0) if theta is None else theta)

    def p_keep(self, script, k, rem, fouls, st_f, share_v, is_starter):
        s = self.table[int(script.time_bucket[k]), int(script.margin_bucket[k])]
        if not np.isfinite(s):
            s = float(np.nanmean(self.table))
        return st_f * (self.theta * s)


class _DonorArm(RotationArm):
    def __init__(self, fit: RotationFit, donors: dict):
        super().__init__(fit)
        self.donors = donors

    def draw_targets(self, prior, script, rng, avail):
        p = apply_min_target(prior.share, avail, self.fit.min_share)
        return p * script.total_slot


class R5HybridV3(_DonorArm):
    """R5 with the corrected hazard matrix -- the round-2 arm re-run, and the
    baseline the keep arms must beat. Identical decision rule to
    `rotation.R5Hybrid`; it lives here only so all four arms share one
    simulator."""
    name = "R5_hybrid"
    simplicity_rank = 2

    def __init__(self, fit, donors, block_scale=None, block_p0=None):
        super().__init__(fit, donors)
        self.block = _Block(fit, block_scale, block_p0)

    def simulate(self, prior, script, rng):
        avail = draw_available(prior, rng)
        return run_resample_keep(prior, script, avail, self.fit, self.donors, rng,
                                 block=self.block, keep=None)


class R7KeepLogistic(_DonorArm):
    """R5 + the fitted logistic close-game keep-starters override."""
    name = "R7_keep_logistic"
    simplicity_rank = 4

    def __init__(self, fit, donors, block_scale=None, block_p0=None,
                 keep_scale=None, keep_q0=None):
        super().__init__(fit, donors)
        self.block = _Block(fit, block_scale, block_p0)
        self.keep = _KeepLogistic(fit, keep_scale, keep_q0)

    def simulate(self, prior, script, rng):
        avail = draw_available(prior, rng)
        return run_resample_keep(prior, script, avail, self.fit, self.donors, rng,
                                 block=self.block, keep=self.keep)


class R8KeepCell(_DonorArm):
    """R5 + the simpler fitted per-cell keep threshold."""
    name = "R8_keep_cell"
    simplicity_rank = 3

    def __init__(self, fit, donors, block_scale=None, block_p0=None, theta=None):
        super().__init__(fit, donors)
        self.block = _Block(fit, block_scale, block_p0)
        self.keep = _KeepCell(fit, theta)

    def simulate(self, prior, script, rng):
        avail = draw_available(prior, rng)
        return run_resample_keep(prior, script, avail, self.fit, self.donors, rng,
                                 block=self.block, keep=self.keep)
