"""team_rate_estimator.py -- the E3 (local-level state-space) team-rate estimator.

This module is new, added by the team-rate estimator round 2 on 2026-09-30. Background is in
docs/models/team_rate_estimator/experiments.md, sections 1-6. It is shared by two callers:

  * the historical emission, `scripts/exp_team_rate_estimator_v3.py --part emit`;
  * the live-slate path, `estimate_asof`.

Both callers therefore run the same filter.

What is estimated. For every team, rate and side, the model is:

  * a latent, league-centred team effect c. It starts at c0 = rho * c_prev (the team's prior-season
    final centred rate) with variance P0;
  * each game played adds an observation z = num/den - L. Here L is the league's as-of level, and the
    observation variance is phi * sigma^2(L) / den, with sigma^2 = L(1-L) for binomial rates and L for
    Poisson rates;
  * c follows a random walk with process variance q per game.

The estimate entering a game is the filtered mean c and variance v. They use only games strictly before
that game. The prediction for the game is p = L + c.

Nothing in here reads data files, fits a parameter, or knows about seasons. The parameters come from
the round-2 fit (`results/team_rate_estimator/params_v3.json`), keyed "F2|<rate>|<side>".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: rate name -> (numerator, denominator, likelihood family). Sides: "off" = the team's own rate,
#: "def" = what opponents did against the team.
RATES: dict[str, tuple[str, str, str]] = {
    "tov": ("tov", "P", "binom"), "ftr": ("fta", "fga", "pois"), "share3": ("fga3", "fga", "binom"),
    "share_rim": ("fga_rim", "fga", "binom"), "make_rim": ("fgm_rim", "fga_rim", "binom"),
    "make_jump": ("fgm_jump", "fga_jump", "binom"), "make3": ("fgm3", "fga3", "binom"),
    "oreb": ("oreb", "reb_ch", "binom"),
}
SIDES = ("off", "def")


def sigma2(L: np.ndarray, fam: str) -> np.ndarray:
    Lc = np.clip(L, 0.01, 0.99)
    return Lc if fam == "pois" else Lc * (1 - Lc)


def kalman_filter(num: np.ndarray, den: np.ndarray, L: np.ndarray, mask: np.ndarray, fam: str,
                  q: float, P0, c0: np.ndarray, phi: float = 1.0,
                  adj: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Filter the (T, J) arrays, which hold teams by game index in date order.

    Returns (c, v), both (T, J). c[:, j] and v[:, j] are the state BEFORE game j, so they use only
    games 0..j-1. P0 is a scalar or a (T,) array. `adj` is den x the opponent's opposite-side
    estimate; it is used only by the opponent-adjusted (+opp) variant."""
    T, J = num.shape
    c = np.asarray(c0, float).copy()
    V = np.broadcast_to(np.asarray(P0, float), (T,)).copy()
    out_c = np.zeros((T, J)); out_v = np.zeros((T, J))
    for j in range(J):
        out_c[:, j] = c; out_v[:, j] = V
        d = den[:, j]
        has = mask[:, j] & (d > 0)
        dd = np.where(d > 0, d, 1.0)
        a = adj[:, j] if adj is not None else 0.0
        Lj = np.clip(np.nan_to_num(L[:, j], nan=0.5), 0.01, 0.99)
        z = np.where(has, (num[:, j] - a) / dd - Lj, 0.0)
        r = phi * sigma2(Lj, fam) / dd
        K = np.where(has, V / (V + r), 0.0)
        c = c + K * (z - c)
        V = (1 - K) * V + q
    return out_c, out_v


@dataclass
class RateParams:
    """Fitted parameters for one rate-side (see params_v3.json)."""
    q: float
    log_p0: tuple[float, float, float]      # a0, a1 (x (cont - cont_mean)), a2 (x coach_change)
    carry: tuple[float, float, float]       # r0, r1 (x (cont - cont_mean)), r2 (x coach_change)
    phi: float
    cont_mean: float

    def p0(self, cont: np.ndarray, coach: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(cont - self.cont_mean, nan=0.0)
        return np.exp(self.log_p0[0] + self.log_p0[1] * x + self.log_p0[2] * coach)

    def rho(self, cont: np.ndarray, coach: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(cont - self.cont_mean, nan=0.0)
        return self.carry[0] + self.carry[1] * x + self.carry[2] * coach


def estimate_asof(team_games: pd.DataFrame, as_of: pd.Timestamp, params: dict[str, RateParams],
                  league_level: dict[str, float], carry: pd.DataFrame) -> pd.DataFrame:
    """Live-slate estimate: one row per team and rate-side, as of `as_of`.

    Inputs:
      * team_games: one row per (game, team) in the CURRENT season. Columns are game_date, team_id and
        opp_id, plus the RATES numerator/denominator columns for the team (bare names) and for its
        opponent (prefixed `o_`).
      * league_level: rate -> the league level L on `as_of`, i.e. the as-of cumulative league rate over
        games before `as_of`; on day 0 it is the prior season's final level. This L centres the
        returned estimate.
      * Optional per-game columns `L_<rate>`: the league level that was in force on each past game's
        date. Past observations are centred on these, exactly as the historical emission does. If
        they are absent, every past game is centred on `league_level`, which is an approximation
        (about 0.003 max abs difference in c on a mid-January check).
      * carry: team_id, c_prev_<rate>_<side>, cont, coach_change.

    Only games with game_date < as_of are used, and this is asserted. Returns team_id, rate, side, c,
    v, L."""
    as_of = pd.Timestamp(as_of)
    hist = team_games[pd.to_datetime(team_games["game_date"]) < as_of].copy()
    assert (pd.to_datetime(hist["game_date"]) < as_of).all(), "as-of violation"
    hist = hist.sort_values(["team_id", "game_date"])
    teams = pd.Index(sorted(set(carry["team_id"]) | set(hist["team_id"])))
    rows = []
    for rate, (num, den, fam) in RATES.items():
        L = float(league_level[rate])
        for side in SIDES:
            n_col, d_col = (num, den) if side == "off" else (f"o_{num}", f"o_{den}")
            h = hist.assign(j=hist.groupby("team_id").cumcount())
            J = int(h["j"].max()) + 1 if len(h) else 0
            ti = teams.get_indexer(h["team_id"])
            N = np.zeros((len(teams), J + 1)); D = np.zeros((len(teams), J + 1))
            M = np.zeros((len(teams), J + 1), bool)
            Lg = np.full((len(teams), J + 1), L)
            if len(h) and f"L_{rate}" in h.columns:
                Lg[ti, h["j"]] = h[f"L_{rate}"].to_numpy(float)
            if len(h):
                N[ti, h["j"]] = h[n_col].to_numpy(float); D[ti, h["j"]] = h[d_col].to_numpy(float)
                M[ti, h["j"]] = True
            cr = carry.set_index("team_id").reindex(teams)
            prm = params[f"{rate}|{side}"]
            cont = cr["cont"].to_numpy(float); coach = cr["coach_change"].fillna(0).to_numpy(float)
            cprev = cr.get(f"c_prev_{rate}_{side}", pd.Series(0.0, index=teams)).fillna(0.0).to_numpy(float)
            c, v = kalman_filter(N, D, Lg, M, fam, prm.q, prm.p0(cont, coach),
                                 prm.rho(cont, coach) * cprev, prm.phi)
            last = M.sum(axis=1)             # the column after each team's last played game
            rows.append(pd.DataFrame({"team_id": teams, "rate": rate, "side": side,
                                      "c": c[np.arange(len(teams)), last], "v": v[np.arange(len(teams)), last],
                                      "L": L}))
    return pd.concat(rows, ignore_index=True)
