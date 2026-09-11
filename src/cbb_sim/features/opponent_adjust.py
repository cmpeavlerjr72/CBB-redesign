"""
opponent_adjust.py -- strictly as-of opponent-strength adjustment of team RATE
features. Shared by every sub-model, which is why it lives here.

WHY (ARCHITECTURE_DECISIONS.md Decision 9a). A raw as-of team rate is the rate a
team produced against the opponents it happened to have played. In the first six
weeks of a CBB season those opponents are mostly non-conference and of wildly
different quality, so the raw rate carries the schedule as much as the team. The
existing rule ("every rating feature is expressed relative to its own snapshot's
league mean") removes the LEVEL drift but not the SCHEDULE. Decision 9 makes the
adjustment method a bake-off arm and never an assumption: the reference arm is
the raw-centred rate this module does not touch, and the two arms it implements
are `one_pass` and `iterative`. Adoption is pending evidence; a tie or a loss for
the adjusted arms is a legitimate result.

WHAT IS ADJUSTED, AND WHAT IS NOT. This module adjusts STYLE RATES (3PA per
possession, rim share of FGA, turnover rate, free-throw rate) computed from box
counters. It does NOT touch `cbb_sim.ratings.own_ratings`, whose `off_c`/`def_c`
are ALREADY opponent-adjusted: they come from a ridge on offence dummies AND
defence dummies fitted jointly on the same window, which is a regularised
simultaneous adjustment. Adjusting them again would double-count.

THE DECOMPOSITION. For a rate with numerator `n` and denominator `d`, a team i's
raw centred offensive deviation as of date t is

    off_dev_i(t) = scale * N_i(<t)/D_i(<t)  -  scale * N_league(<t)/D_league(<t)

and the mirrored quantity for what i ALLOWED is `def_dev_i(t)`. Let M(t) be the
matrix of denominator mass team i produced against team j in games strictly
before t. Then

    one_pass:    off_adj_i = off_dev_i  -  [rownorm(M) @ def_dev]_i
                 def_adj_i = def_dev_i  -  [rownorm(M^T) @ off_dev]_i

    iterative:   the same two equations alternated to convergence -- which is
                 ALTERNATING LEAST SQUARES on the weighted two-way model
                     minimise  sum_g  d_g * (r_g - o_i - a_j)^2
                 over the team-games g = (offence i, defence j) played strictly
                 before t, weighted by each game's denominator. The two update
                 formulas above ARE the coordinate-descent steps of that
                 objective, so `one_pass` is literally this solve truncated
                 after its first offence update.

TWO IMPLEMENTATION FACTS THAT ARE NOT COSMETIC, both found by test rather than
by argument.

1. The sweep must be GAUSS-SEIDEL (update the defence side with the offence
   values just computed), not Jacobi (update both from the previous sweep).
   Gauss-Seidel is coordinate descent on a convex quadratic and therefore
   converges monotonically; Jacobi is not, and on a schedule graph whose
   components are small it DIVERGES -- measured on a synthetic schedule, the
   Jacobi form left 6 of 24 early-season dates unconverged at 50 iterations and
   growing, which would have shipped a garbage feature for exactly the November
   games the adjustment exists to fix.

2. Re-centring must preserve the null space. Adding c to every offence effect in
   a connected component and subtracting c from every defence effect in it
   leaves every prediction o_i + a_j unchanged, so the split is unidentified
   inside each component and the iterates drift along that direction. The fix is
   to move the component's denominator-weighted mean OUT of the defence side and
   INTO the offence side (`a -= mu; o += mu`), which pins the split without
   touching the objective. Re-centring the two sides independently would change
   the fit. Identification within a component is the honest limit of any
   opponent adjustment -- two teams that share no chain of opponents cannot be
   compared -- and the component count is reported rather than hidden.

`M^T` is used for the defensive side rather than a second accumulator because
the denominator team i ALLOWED against j is by construction the denominator team
j PRODUCED against i.

THE OWN-RATE COMPONENT IS UNTOUCHED. `off_adj` is returned as
`off_dev - correction`, where `off_dev` is passed in by the caller -- exactly the
column the reference arm uses. So the adjusted arm differs from the reference
arm by the subtracted opponent term and by nothing else, which is what makes the
bake-off a test of the adjustment rather than of two different pipelines.

LEAK SAFETY. Every quantity at date t is accumulated over games with
`game_date < t` only. The date-snapshot loop below never sees the day's own
games. `tests/test_opponent_adjust.py::test_strictly_as_of_appending_future_games_changes_nothing`
is the standing check.

NO SHRINKAGE, NO CAPS. A team with no prior games has no opponents and gets a
correction of exactly 0.0, which on a league-centred scale IS the league mean.
A team with one prior game gets a noisy correction, and so does its raw rate;
adding a shrinkage constant here would be a hand-tuned knob on a feature
(CLAUDE.md, standing rule "no hand tuning"). The regularised quality signal is
`own_ratings`' job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

METHODS: tuple[str, ...] = ("none", "one_pass", "iterative")

#: Iterative arm: hard cap and tolerance, both fixed here rather than tuned per
#: call so every consumer of the module runs the same estimator.
#:
#: The cap is 200 because alternating least squares converges at a linear rate
#: set by how well connected the schedule graph is, and the November graph is
#: barely connected: on the synthetic schedule in the tests the sweep needs 22
#: iterations at 2 components, 42 at the date the graph first joins up, and more
#: than 50 on the two dates after that (it was at 3e-5 after 50, still falling).
#: Every one of those numbers is negligible as a FEATURE value -- 3e-5 of a
#: percentage point -- so the cap is not protecting the feature from a bad
#: value, it is protecting the run from an unbounded loop; dates that hit it are
#: counted in the returned meta and reported, never silently accepted.
MAX_ITER = 200
TOL = 1e-6


@dataclass
class AdjustResult:
    """`off_adj` / `def_adj`: one column per rate, indexed like the input rows.
    `meta`: per-season convergence and coverage, for the experiments doc."""
    off_adj: pd.DataFrame
    def_adj: pd.DataFrame
    meta: dict


def _snapshots(boxes: pd.DataFrame, teams: np.ndarray, dates: np.ndarray,
               num: str, den: str, scale: float) -> tuple[np.ndarray, ...]:
    """Per-(date, team) cumulative numerator/denominator over games STRICTLY
    BEFORE the date, for the offence side and the allowed side, plus the league
    rate on the same window.

    Returns (off_dev, def_dev, off_den, def_den, league_rate), each shaped
    (n_dates, n_teams) except league_rate which is (n_dates,)."""
    ti = pd.Index(teams)
    di = pd.Index(dates)
    n_t, n_d = len(teams), len(dates)
    r = ti.get_indexer(boxes["team_id"].to_numpy())
    o = ti.get_indexer(boxes["opp_id"].to_numpy())
    k = di.get_indexer(boxes["game_date"].to_numpy())
    nv = boxes[num].to_numpy(dtype="float64")
    dv = boxes[den].to_numpy(dtype="float64")

    def _acc(rows: np.ndarray, vals: np.ndarray) -> np.ndarray:
        a = np.zeros((n_d, n_t), dtype="float64")
        np.add.at(a, (k, rows), vals)
        # cumulative over dates, then shifted one date down => strictly before
        c = np.cumsum(a, axis=0)
        out = np.zeros_like(c)
        out[1:] = c[:-1]
        return out

    off_n, off_d = _acc(r, nv), _acc(r, dv)
    def_n, def_d = _acc(o, nv), _acc(o, dv)
    lg_n = off_n.sum(axis=1)
    lg_d = off_d.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(lg_d > 0, scale * lg_n / np.maximum(lg_d, 1e-9), np.nan)
        off_r = np.where(off_d > 0, scale * off_n / np.maximum(off_d, 1e-9), np.nan)
        def_r = np.where(def_d > 0, scale * def_n / np.maximum(def_d, 1e-9), np.nan)
    off_dev = np.nan_to_num(off_r - lg[:, None], nan=0.0)
    def_dev = np.nan_to_num(def_r - lg[:, None], nan=0.0)
    return off_dev, def_dev, off_d, def_d, lg


def _rownorm(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=1, keepdims=True)
    return np.divide(M, s, out=np.zeros_like(M), where=s > 0)


def _components(M: np.ndarray) -> tuple[int, np.ndarray]:
    """Connected components of the played-so-far schedule graph."""
    A = csr_matrix((M + M.T) > 0)
    n, lab = connected_components(A, directed=False)
    return int(n), lab.astype("int64")


def _component_mean(v: np.ndarray, w: np.ndarray, labels: np.ndarray,
                    n_comp: int) -> np.ndarray:
    """Each team's connected component's own weighted mean of `v`, broadcast
    back to team length."""
    tw = np.bincount(labels, weights=w, minlength=n_comp)
    sw = np.bincount(labels, weights=w * v, minlength=n_comp)
    mu = np.divide(sw, tw, out=np.zeros_like(sw), where=tw > 0)
    return mu[labels]


def _iterate(off_dev: np.ndarray, def_dev: np.ndarray, R: np.ndarray, C: np.ndarray,
             w_def: np.ndarray, labels: np.ndarray, n_comp: int,
             max_iter: int = MAX_ITER, tol: float = TOL
             ) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Alternating least squares for the two-way model at one date.

    Gauss-Seidel sweeps with null-space-preserving component re-centring; see
    the module docstring for why both details are load-bearing. Returns
    `(offence effects, defence effects, iterations, final max change)`."""
    o = off_dev.copy()
    a = def_dev.copy()
    it, delta = 0, float("inf")
    for it in range(1, max_iter + 1):
        o_new = off_dev - R @ a
        a_new = def_dev - C @ o_new
        mu = _component_mean(a_new, w_def, labels, n_comp)
        a_new = a_new - mu
        o_new = o_new + mu
        delta = float(max(np.abs(o_new - o).max(initial=0.0),
                          np.abs(a_new - a).max(initial=0.0)))
        o, a = o_new, a_new
        if delta < tol:
            break
    return o, a, it, delta


def adjust_rate(boxes: pd.DataFrame, num: str, den: str, scale: float,
                method: str, max_iter: int = MAX_ITER, tol: float = TOL
                ) -> tuple[pd.Series, pd.Series, dict]:
    """Opponent-strength CORRECTIONS for one rate, one season.

    `boxes` is one row per team-game with `team_id, opp_id, game_date, <num>,
    <den>`. Returns `(off_correction, def_correction, meta)` aligned to
    `boxes.index`: the caller subtracts these from its own raw centred
    deviations, so the own-rate component of the adjusted feature is bit-
    identical to the reference arm's."""
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; known: {METHODS}")
    zero = pd.Series(0.0, index=boxes.index, dtype="float64")
    if method == "none":
        return zero, zero.copy(), {"method": "none"}

    teams = np.sort(boxes["team_id"].unique())
    dates = np.sort(boxes["game_date"].unique())
    off_dev, def_dev, off_den, def_den, _lg = _snapshots(boxes, teams, dates, num, den, scale)

    ti = pd.Index(teams)
    di = pd.Index(dates)
    r = ti.get_indexer(boxes["team_id"].to_numpy())
    o = ti.get_indexer(boxes["opp_id"].to_numpy())
    k = di.get_indexer(boxes["game_date"].to_numpy())
    dv = boxes[den].to_numpy(dtype="float64")

    n_t, n_d = len(teams), len(dates)
    M = np.zeros((n_t, n_t), dtype="float64")
    off_corr = np.zeros((n_d, n_t), dtype="float64")
    def_corr = np.zeros((n_d, n_t), dtype="float64")
    iters = np.zeros(n_d, dtype="int32")
    deltas = np.zeros(n_d, dtype="float64")
    comps = np.zeros(n_d, dtype="int32")

    order = np.argsort(k, kind="stable")
    k_sorted = k[order]
    starts = np.searchsorted(k_sorted, np.arange(n_d), side="left")
    ends = np.searchsorted(k_sorted, np.arange(n_d), side="right")

    for kk in range(n_d):
        # M holds only games strictly before dates[kk] -- it is updated AFTER
        # the correction for this date is computed.
        R = _rownorm(M)
        C = _rownorm(M.T)
        if method == "one_pass":
            off_corr[kk] = R @ def_dev[kk]
            def_corr[kk] = C @ off_dev[kk]
            iters[kk] = 1
        else:
            n_comp, labels = _components(M)
            comps[kk] = n_comp
            oe, ae, it, dl = _iterate(off_dev[kk], def_dev[kk], R, C,
                                      def_den[kk], labels, n_comp, max_iter, tol)
            off_corr[kk] = off_dev[kk] - oe
            def_corr[kk] = def_dev[kk] - ae
            iters[kk], deltas[kk] = it, dl
        sl = order[starts[kk]:ends[kk]]
        if len(sl):
            np.add.at(M, (r[sl], o[sl]), dv[sl])

    meta = {"method": method, "n_teams": int(n_t), "n_dates": int(n_d),
            "max_iterations": int(iters.max()) if n_d else 0,
            "mean_iterations": float(iters.mean()) if n_d else 0.0,
            "n_dates_not_converged": int((deltas > tol).sum()) if method == "iterative" else 0,
            "worst_final_delta": float(deltas.max()) if n_d else 0.0,
            "max_components": int(comps.max()) if (n_d and method == "iterative") else 0,
            "final_components": int(comps[-1]) if (n_d and method == "iterative") else 0}
    return (pd.Series(off_corr[k, r], index=boxes.index),
            pd.Series(def_corr[k, r], index=boxes.index), meta)


def adjust_team_form(boxes: pd.DataFrame, rate_defs: dict, rate_scale: dict,
                     method: str, season_col: str = "season",
                     max_iter: int = MAX_ITER, tol: float = TOL) -> AdjustResult:
    """Opponent-strength corrections for every rate in `rate_defs`, per season.

    `boxes`: one row per team-game with `season, team_id, opp_id, game_date` and
    every numerator/denominator column named by `rate_defs`.
    `rate_defs`: `{name: (numerator_col, denominator_col)}`.
    `rate_scale`: `{name: scale}`.

    Seasons are adjusted INDEPENDENTLY -- a season's league mean, its schedule
    graph and its rule era are its own, and an adjustment that pooled seasons
    would put a 2022 opponent into a 2025 correction."""
    off = pd.DataFrame(index=boxes.index)
    dfn = pd.DataFrame(index=boxes.index)
    meta: dict = {}
    for name, (num, den) in rate_defs.items():
        oc = pd.Series(0.0, index=boxes.index, dtype="float64")
        dc = pd.Series(0.0, index=boxes.index, dtype="float64")
        meta[name] = {}
        for season, idx in boxes.groupby(season_col).groups.items():
            b = boxes.loc[idx]
            o, d, m = adjust_rate(b, num, den, float(rate_scale[name]), method,
                                  max_iter=max_iter, tol=tol)
            oc.loc[idx] = o.to_numpy()
            dc.loc[idx] = d.to_numpy()
            meta[name][int(season)] = m
        off[name] = oc.astype("float32")
        dfn[name] = dc.astype("float32")
    return AdjustResult(off_adj=off, def_adj=dfn, meta=meta)
