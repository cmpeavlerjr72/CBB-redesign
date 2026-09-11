"""
rotation_v4.py -- ROUND 4 of the L4 rotation bake-off: a new model family.

Why a new family
----------------
Three rounds could not produce two structural facts of a real rotation (L25):

  * the second half opens at **0.90** starter share in every margin band -- no
    arm in three rounds exceeded 0.80;
  * starters are **kept** late in a close game (0.749 in the final 8:00 at
    |margin| <= 5) -- the best arm reached 0.728 and the override family's
    reachable maximum over its whole knob grid was 0.699.

Rounds 1-3 were two families: (a) draw a per-player MINUTES BUDGET (Dirichlet
over as-of shares) and let a scheduler place it; (b) replay a DONOR lineup
sequence and override it. Both decide *how much* each player plays and then
infer *when*. A coach does the opposite: he decides, at a stoppage, whom to take
off and whom to bring on, and the minutes are the consequence. Round 4 models
that decision directly.

The family
----------
Per-player **discrete-time substitution hazards** at every possession boundary:

    p_out(i) = sigma(x_i . w_out)   for the five on the floor
    p_in(j)  = sigma(x_j . w_in)    for every eligible bench candidate

Exits are independent Bernoulli draws; the five-on-the-floor constraint is then
imposed by taking exactly `n_exit` entrants from the bench in an
Efraimidis-Spirakis exponential race weighted by `p_in / (1 - p_in)`, which is
weighted sampling without replacement and is the SAME rule the engine adapter
runs (`engine/rotation_adapter.py`, `ENGINE_ROTATION=round4`). The offline
sampler and the adapter therefore share a decision rule exactly, not
approximately; only the RNG stream differs (`docs/models/engine/model.md`
section 4.5).

The three arms differ in exactly one thing each:

  H1  logistic hazards + a HARD reset to the predicted starting five at the
      start of period 2.
  H2  the same hazards, NO hard reset: the second-half reset must be EARNED by
      the `period_boundary` terms of the design.
  H3  H1 with a LightGBM hazard in place of the logistic.

Engine expressibility
---------------------
Every feature in `SUB_FEATURES` is computable from state the possession loop
already carries at a boundary: `period`, `seconds_remaining`, `home_score_diff`,
`prev_end` (the five `PREV_END_LEVELS` codes, i.e. the dead-ball opportunity),
`team_fouls`, the (2N, S) personal-foul block, and rotation-batch state the
adapter owns (minutes this half, current stint/rest, as-of share, start rank).
`design()` is written over (M, S) arrays and is called with M = 1 offline and
M = 2N in the engine, so the two cannot drift apart.

Deliberately EXCLUDED, and why: a timeout indicator. Timeouts are observable in
CBBD pbp and the audit reports their effect, but the engine has no timeout
model, so a hazard conditioned on them could not be evaluated in simulation.
Same class of exclusion as `is_transition` in features.md section 3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.models.rotation import (
    FOUL_OUT,
    MAX_CANDIDATES,
    SLOTS,
    GameScript,
    RotationArm,
    RotationFit,
    TeamPrior,
    actual_foul_matrix,
    build_priors,
    build_scripts,
    draw_available,
)

#: `engine.state.PREV_END_LEVELS` order. The possession's `start_reason` is the
#: previous possession's end, which is what the engine's `prev_end` holds.
PREV_END_LEVELS = ("period_start", "DREB", "TOV", "made_FG", "made_FT", "other")
PREV_END_CODE = {v: i for i, v in enumerate(PREV_END_LEVELS)}

HALF_SECONDS = 1200.0
OT_SECONDS = 300.0

#: feature order. Anything added here must be added to `design()` in the same
#: position. The design is SATURATED in (time cell x margin band x is_starter)
#: because the audit (`docs/tests/rotation_sub_hazard_audit_2026-09-10.md`)
#: shows both hazards move sharply and NON-MONOTONICALLY across those cells --
#: a starter's exit hazard runs 0.035 (H1 20-10) -> 0.020 (H2 20-16) -> 0.054
#: (H2 16-12) -> 0.027 (H2 8-4) -> 0.037 (H2 2-0) in the close band, and a
#: bench player's exit hazard is 0.244 in H2 20-16 against 0.041 in H1 20-10.
#: A linear time term cannot represent that shape. The cells are the same nine
#: the audit tabulates; the gate reads OCCUPANCY in (some of) them, which is a
#: different functional of the process -- an equilibrium the hazards have to
#: produce under the five-on-the-floor constraint, not a quantity fitted here.
TIME_CELL_NAMES = ("H1_20_10", "H1_10_00", "H2_20_16", "H2_16_12", "H2_12_08",
                   "H2_08_04", "H2_04_02", "H2_02_00", "OT")
N_TIME_CELLS = len(TIME_CELL_NAMES)
#: time cells counted as "late" for the close-and-late gate cell
LATE_CELLS = (5, 6, 7)

SUB_FEATURES = (
    ["is_starter", "share", "fouls", "foul_out", "fouls_x_is_starter",
     "state_min", "half_min", "half_min_dev",
     "sec_left_frac", "is_ot", "team_fouls_frac",
     "period_boundary", "dead_made_ft", "dead_tov", "dead_other"]
    + [f"tc_{n}" for n in TIME_CELL_NAMES[1:]]
    + ["mb_6_15", "mb_gt15"]
    + [f"is_starter_x_tc_{n}" for n in TIME_CELL_NAMES[1:]]
    + ["is_starter_x_mb_6_15", "is_starter_x_mb_gt15",
       "is_starter_x_period_boundary", "is_starter_x_sec_left_frac",
       "is_starter_x_close_x_late", "is_starter_x_blowout_x_late",
       "share_x_late", "share_x_mb_gt15", "share_x_period_boundary",
       "fouls_x_late", "abs_margin", "is_starter_x_abs_margin"]
)
N_FEATURES = len(SUB_FEATURES)


def time_cell(period: np.ndarray, sec_left: np.ndarray) -> np.ndarray:
    """The nine audit time cells. 0-1 first half, 2-7 second half, 8 overtime."""
    p = np.asarray(period)
    c = np.asarray(sec_left)
    out = np.full(p.shape, 8, dtype=np.int64)
    h1 = p == 1
    out[h1 & (c > 600)] = 0
    out[h1 & (c <= 600)] = 1
    h2 = p == 2
    out[h2 & (c > 960)] = 2
    out[h2 & (c <= 960) & (c > 720)] = 3
    out[h2 & (c <= 720) & (c > 480)] = 4
    out[h2 & (c <= 480) & (c > 240)] = 5
    out[h2 & (c <= 240) & (c > 120)] = 6
    out[h2 & (c <= 120)] = 7
    return out


# ===========================================================================
# 1. The design matrix -- ONE implementation, (M, S) in and (M, S, F) out
# ===========================================================================
def design(is_starter: np.ndarray, share: np.ndarray, fouls: np.ndarray,
           state_min: np.ndarray, half_min: np.ndarray,
           period: np.ndarray, sec_left: np.ndarray, margin: np.ndarray,
           prev_end: np.ndarray, team_fouls: np.ndarray) -> np.ndarray:
    """(M, S, F) design block.

    Player arrays (`is_starter`, `share`, `fouls`, `state_min`, `half_min`) are
    (M, S); state arrays (`period`, `sec_left`, `margin`, `prev_end`,
    `team_fouls`) are (M,). `margin` is this team's own signed margin, `sec_left`
    is seconds remaining IN THE PERIOD, and `prev_end` is a code into
    `PREV_END_LEVELS`.
    """
    is_starter = np.asarray(is_starter, dtype=np.float64)
    M, S = is_starter.shape
    share = np.asarray(share, dtype=np.float64)
    fouls = np.asarray(fouls, dtype=np.float64)
    state_min = np.asarray(state_min, dtype=np.float64)
    half_min = np.asarray(half_min, dtype=np.float64)

    per = np.asarray(period, dtype=np.float64).reshape(M, 1)
    sl = np.asarray(sec_left, dtype=np.float64).reshape(M, 1)
    mg = np.asarray(margin, dtype=np.float64).reshape(M, 1)
    pe = np.asarray(prev_end, dtype=np.int64).reshape(M, 1)
    tf = np.asarray(team_fouls, dtype=np.float64).reshape(M, 1)

    tc = time_cell(per.astype(np.int64), sl)
    is_ot = (per >= 3).astype(np.float64)
    plen = np.where(per >= 3, OT_SECONDS, HALF_SECONDS)
    sec_frac = np.clip(sl / plen, 0.0, 1.0)
    elapsed_half = np.clip((plen - sl) / 60.0, 0.0, None)
    am = np.abs(mg) / 10.0
    is_close = (np.abs(mg) <= 5).astype(np.float64)
    mb1 = ((np.abs(mg) > 5) & (np.abs(mg) <= 15)).astype(np.float64)
    mb2 = (np.abs(mg) > 15).astype(np.float64)
    late = np.isin(tc, LATE_CELLS).astype(np.float64)
    pb = (pe == PREV_END_CODE["period_start"]).astype(np.float64)
    d_ft = (pe == PREV_END_CODE["made_FT"]).astype(np.float64)
    d_tov = (pe == PREV_END_CODE["TOV"]).astype(np.float64)
    d_oth = (pe == PREV_END_CODE["other"]).astype(np.float64)
    tcd = [(tc == j).astype(np.float64) for j in range(1, N_TIME_CELLS)]

    o = np.ones((M, S), dtype=np.float64)
    X = np.empty((M, S, N_FEATURES), dtype=np.float64)
    c = 0

    def put(v):
        nonlocal c
        X[:, :, c] = v
        c += 1

    put(is_starter)
    put(share)
    put(fouls)
    put((fouls >= FOUL_OUT).astype(np.float64))
    put(fouls * is_starter)
    put(state_min)
    put(half_min)
    put(half_min - share * elapsed_half)
    put(sec_frac * o)
    put(is_ot * o)
    put(tf / 5.0 * o)
    put(pb * o)
    put(d_ft * o)
    put(d_tov * o)
    put(d_oth * o)
    for t in tcd:
        put(t * o)
    put(mb1 * o)
    put(mb2 * o)
    for t in tcd:
        put(is_starter * t)
    put(is_starter * mb1)
    put(is_starter * mb2)
    put(is_starter * pb)
    put(is_starter * sec_frac)
    put(is_starter * is_close * late)
    put(is_starter * mb2 * late)
    put(share * late)
    put(share * mb2)
    put(share * pb)
    put(fouls * late)
    put(am * o)
    put(is_starter * am)
    assert c == N_FEATURES, (c, N_FEATURES)
    return X


# ===========================================================================
# 2. Fitted object
# ===========================================================================
@dataclass
class SubHazardFit:
    """The two hazards plus the provenance the manifest needs."""

    kind: str = "logistic"                     # "logistic" | "lgbm"
    out_coef: list[float] = field(default_factory=list)
    out_intercept: float = 0.0
    in_coef: list[float] = field(default_factory=list)
    in_intercept: float = 0.0
    out_model_path: str | None = None          # lgbm only
    in_model_path: str | None = None
    n_out: int = 0
    n_in: int = 0
    base_out: float = 0.0
    base_in: float = 0.0
    features: list[str] = field(default_factory=lambda: list(SUB_FEATURES))
    notes: dict = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "SubHazardFit":
        return SubHazardFit(**json.loads(Path(path).read_text(encoding="utf-8")))

    # -- scoring ---------------------------------------------------------
    def p_out(self, X: np.ndarray) -> np.ndarray:
        return _score(X, self.out_coef, self.out_intercept, self.kind,
                      self.notes.get("_out_booster"))

    def p_in(self, X: np.ndarray) -> np.ndarray:
        return _score(X, self.in_coef, self.in_intercept, self.kind,
                      self.notes.get("_in_booster"))


def _score(X: np.ndarray, coef, intercept: float, kind: str, booster) -> np.ndarray:
    """(..., F) -> (...) probability."""
    shp = X.shape[:-1]
    flat = X.reshape(-1, X.shape[-1])
    if kind == "lgbm" and booster is not None:
        p = booster.predict(flat)
        return np.asarray(p, dtype=np.float64).reshape(shp)
    z = flat @ np.asarray(coef, dtype=np.float64) + float(intercept)
    return (1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))).reshape(shp)


# ===========================================================================
# 3. Training rows
# ===========================================================================
def build_sub_training(tp: pd.DataFrame, feats: pd.DataFrame, fit: RotationFit,
                       game_ids, fouls: pd.DataFrame | None = None,
                       side_state: pd.DataFrame | None = None,
                       max_team_games: int = 0, seed: int = 11
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Risk-set rows for the two hazards, from on-floor set transitions.

    Rows are built over the **as-of candidate pool** (`build_priors`), never the
    game's own participant list, so the design matrix at fit time is the design
    matrix at simulation time. The training game's own `PersonalFoul` events
    supply the foul state -- legitimate at fit time, where the label is that same
    game's substitution, and never at simulation time (the rule established by
    `experiments.md` section 5).
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

    Xo, yo, Xi, yi = [], [], [], []
    for key in keys:
        lu = actual[key]
        prior, script = priors[key], scripts[key]
        idx_of = {int(p): i for i, p in enumerate(prior.pids)}
        n = prior.n
        npos = min(len(lu), script.n)
        if npos < 20:
            continue
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
                fouls_v = fm[k].astype("float64") if k < fm.shape[0] else np.zeros(n)
                out_idx = np.flatnonzero(prev_on)
                in_idx = np.flatnonzero(~prev_on & (fouls_v < FOUL_OUT))
                if len(out_idx) == 5 and len(in_idx):
                    X = design(is_st[None, :], share[None, :], fouls_v[None, :],
                               state_min[None, :], half_min[None, :],
                               np.array([script.period[k]]),
                               np.array([script.start_clock[k]]),
                               np.array([script.margin[k]]), np.array([prev_end[k]]),
                               np.array([team_fouls[k]]))[0]
                    Xo.append(X[out_idx])
                    yo.append((~on[k][out_idx]).astype(np.float64))
                    Xi.append(X[in_idx])
                    yi.append(on[k][in_idx].astype(np.float64))
                # `state_min` is time in the CURRENT on/off state: it resets for
                # anyone whose state just changed, exactly as the sampler does.
                state_min[on[k] != prev_on] = 0.0
            d = script.dur[k] / 60.0
            state_min += d
            half_min[on[k]] += d
    if not Xo:
        raise RuntimeError("no substitution-hazard training rows produced")
    return (np.vstack(Xo), np.concatenate(yo), np.vstack(Xi), np.concatenate(yi))


def _side_state_index(side_state: pd.DataFrame | None, gset: set) -> dict:
    if side_state is None or not len(side_state):
        return {}
    d = side_state[side_state["game_id"].isin(gset)]
    return {(int(g), int(t)): (dd["prev_end"].to_numpy(dtype="int64"),
                               dd["own_team_fouls"].to_numpy(dtype="float64"))
            for (g, t), dd in d.groupby(["game_id", "team_id"], sort=False)}


def _side_arrays(ss: dict, key, script: GameScript, npos: int):
    got = ss.get(key)
    if got is None:
        pe = np.where(np.concatenate([[True], np.diff(script.period) != 0]),
                      PREV_END_CODE["period_start"], PREV_END_CODE["DREB"])
        return pe[:npos], np.zeros(npos)
    pe, tf = got
    if len(pe) < npos:
        pe = np.concatenate([pe, np.full(npos - len(pe), PREV_END_CODE["other"])])
        tf = np.concatenate([tf, np.full(npos - len(tf), tf[-1] if len(tf) else 0.0)])
    return pe[:npos], tf[:npos]


def fit_sub_models(Xo, yo, Xi, yi, kind: str = "logistic", seed: int = 0,
                   n_estimators: int = 300, num_leaves: int = 31,
                   learning_rate: float = 0.05, threads: int = 4) -> SubHazardFit:
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        out = []
        for X, y in ((Xo, yo), (Xi, yi)):
            m = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs",
                                   random_state=seed)
            m.fit(X, y)
            out.append((m.coef_[0].tolist(), float(m.intercept_[0])))
        return SubHazardFit(kind="logistic",
                            out_coef=out[0][0], out_intercept=out[0][1],
                            in_coef=out[1][0], in_intercept=out[1][1],
                            n_out=int(len(yo)), n_in=int(len(yi)),
                            base_out=float(np.mean(yo)), base_in=float(np.mean(yi)))
    import lightgbm as lgb
    boosters = []
    for X, y in ((Xo, yo), (Xi, yi)):
        ds = lgb.Dataset(X, label=y, feature_name=list(SUB_FEATURES))
        params = {"objective": "binary", "learning_rate": learning_rate,
                  "num_leaves": num_leaves, "min_data_in_leaf": 200,
                  "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
                  "verbose": -1, "seed": seed, "num_threads": threads,
                  "deterministic": True, "force_row_wise": True}
        boosters.append(lgb.train(params, ds, num_boost_round=n_estimators))
    f = SubHazardFit(kind="lgbm", n_out=int(len(yo)), n_in=int(len(yi)),
                     base_out=float(np.mean(yo)), base_in=float(np.mean(yi)))
    f.notes["_out_booster"] = boosters[0]
    f.notes["_in_booster"] = boosters[1]
    return f


# ===========================================================================
# 4. The sampler
# ===========================================================================
def _race_pick(w: np.ndarray, u: np.ndarray, k: int) -> np.ndarray:
    """Efraimidis-Spirakis: the `k` smallest `-log(u)/w` are a weighted sample
    without replacement with weights `w`. Vectorises, and is the SAME rule the
    engine adapter runs."""
    key = -np.log(np.clip(u, 1e-12, 1.0)) / np.maximum(w, 1e-12)
    return np.argsort(key, kind="stable")[:k]


def run_sub_hazard(prior: TeamPrior, script: GameScript, avail: np.ndarray,
                   sub: SubHazardFit, fit: RotationFit, rng: np.random.Generator,
                   hard_reset: bool = True,
                   prev_end: np.ndarray | None = None,
                   team_fouls: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one team-game's on-floor sequence from the two hazards."""
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
                X = design(is_st[None, :], share[None, :], fouls.astype(np.float64)[None, :],
                           state_min[None, :], half_min[None, :],
                           np.array([script.period[k]]), np.array([script.start_clock[k]]),
                           np.array([script.margin[k]]), np.array([prev_end[k]]),
                           np.array([team_fouls[k]]))[0]
                po = sub.p_out(X[on_idx])
                po = np.where(fouls[on_idx] >= FOUL_OUT, 1.0, po)
                exits = rng.random(5) < po
                n_exit = int(min(exits.sum(), len(bench_idx)))
                if n_exit:
                    pi = np.clip(sub.p_in(X[bench_idx]), 1e-9, 1 - 1e-9)
                    w = pi / (1.0 - pi)
                    pick = _race_pick(w, rng.random(len(bench_idx)), n_exit)
                    cand_out = on_idx[exits]
                    cand_out = cand_out[np.argsort(
                        -(fouls[cand_out] >= FOUL_OUT).astype(int), kind="stable")]
                    leaving = cand_out[:n_exit]
                    onmask[leaving] = False
                    onmask[bench_idx[pick]] = True

        cur = np.flatnonzero(onmask)
        if len(cur) != 5:
            score = np.where(eligible, share, -1e12)
            score[cur] += 10.0
            cur = np.argsort(-score)[:5]
            onmask[:] = False
            onmask[cur] = True
        # a fouled-out player never stays on the floor while a substitute exists
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
# 5. Arms
# ===========================================================================
class _SubHazardArm(RotationArm):
    hard_reset = True

    def __init__(self, fit: RotationFit, sub: SubHazardFit,
                 side_state: dict | None = None):
        super().__init__(fit)
        self.sub = sub
        self.side_state = side_state or {}

    def draw_targets(self, prior, script, rng, avail):      # not used
        return prior.share * script.total_slot

    def simulate(self, prior: TeamPrior, script: GameScript,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        avail = draw_available(prior, rng)
        pe, tf = _side_arrays(self.side_state, (prior.game_id, prior.team_id),
                              script, script.n)
        return run_sub_hazard(prior, script, avail, self.sub, self.fit, rng,
                              hard_reset=self.hard_reset, prev_end=pe, team_fouls=tf)


class H1SubHazard(_SubHazardArm):
    name = "H1_sub_hazard"
    simplicity_rank = 2
    hard_reset = True


class H2SubHazardNoReset(_SubHazardArm):
    name = "H2_sub_hazard_noreset"
    simplicity_rank = 3
    hard_reset = False


class H3SubHazardLGBM(_SubHazardArm):
    name = "H3_sub_hazard_lgbm"
    simplicity_rank = 4
    hard_reset = True


ARMS = {
    "H1_sub_hazard": H1SubHazard,
    "H2_sub_hazard_noreset": H2SubHazardNoReset,
    "H3_sub_hazard_lgbm": H3SubHazardLGBM,
}


# ===========================================================================
# 6. Side state (prev_end / team fouls) -- the engine already has both
# ===========================================================================
def load_side_state(season: int, poss_dir: Path = Path("data/processed/possessions")
                    ) -> pd.DataFrame:
    """(game_id, team_id, poss_index) -> prev_end code and own team fouls.

    `rotation.load_team_possessions` carries neither and is not ours to change,
    so both are re-read from the same parquet and keyed the same way."""
    p = pd.read_parquet(Path(poss_dir) / f"possessions_{season}.parquet",
                        columns=["game_id", "poss_index", "offense_is_home", "start_reason",
                                 "off_team_fouls", "def_team_fouls"])
    gu = pd.read_parquet("data/processed/games_universe.parquet")
    gu = gu[gu["season"] == season][["game_id", "home_team_id", "away_team_id"]]
    p = p.merge(gu, on="game_id", how="inner")
    code = p["start_reason"].map(PREV_END_CODE).fillna(PREV_END_CODE["other"]).to_numpy()
    rows = []
    for is_home, col in ((True, "home_team_id"), (False, "away_team_id")):
        off = p["offense_is_home"].to_numpy() == is_home
        rows.append(pd.DataFrame({
            "game_id": p["game_id"].to_numpy(),
            "team_id": p[col].to_numpy(),
            "poss_index": p["poss_index"].to_numpy(),
            "prev_end": code,
            "own_team_fouls": np.where(off, p["off_team_fouls"].to_numpy(),
                                       p["def_team_fouls"].to_numpy()),
        }))
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["game_id", "team_id", "poss_index"]).reset_index(drop=True)


def side_state_index(side_state: pd.DataFrame, game_ids=None) -> dict:
    gset = set(int(g) for g in game_ids) if game_ids is not None else None
    d = side_state if gset is None else side_state[side_state["game_id"].isin(gset)]
    return _side_state_index(d, set(d["game_id"].unique().tolist()))
