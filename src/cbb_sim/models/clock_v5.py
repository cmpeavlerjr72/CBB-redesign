"""clock_v5.py -- round 5's within-game DISPERSION arms.

Pre-registration: `docs/models/clock/experiments.md` section 16, committed at
`28b5f17` BEFORE this module existed. Rounds 1-4 settled the family
(cell-based), the censoring rule (L20/L26), the state parametrisation (P3) and
the refit scheme (S1); NONE of that is reopened here. Every arm below wraps the
round-4 reference artifact and changes ONE thing: the dependence structure
between the draws inside a game.

WHY A WRAPPER AND NOT A REFIT
-----------------------------
The measurement (`scripts/diag_clk5_dispersion.py`) puts 98.9% of the per-game
mean-duration variance gap on a missing within-game LEVEL and 1.1% on the
conditional law, whose SD ratio is 0.9977. There is nothing to refit in the
conditional law; what is missing is a random effect on top of it. So every
latent arm reads the SAME fitted pickle the engine serves today and adds a
scale mixture with `E[A] = 1`, which leaves every conditional mean untouched.

WHAT IS AND IS NOT A HAND TUNE
------------------------------
`A ~ LogNormal(-s2/2, s2)` with `s` estimated by method of moments on TRAINING
rows only is a model parameter of a random-effects model: it has a likelihood,
it is fitted walk-forward, and it changes the predictive DISTRIBUTION, not the
engine's output. A multiplier applied to a simulated duration to make the
possession count look right would be the banned pattern
(`docs/SIM_GUARDRAILS.md` section 5). The distinguishing test is that `E[A] = 1`
by construction, so no arm here can move the conditional mean, the possession
count mean, or round 4's mean gate in its own favour -- and the grader reports
that gate for every arm so the claim is checked rather than asserted.

THE MARGINAL, AND WHY THE SCORER CAN BE THE ROUND-3 ONE
-------------------------------------------------------
`LatentArm.pmf(df)` returns the MARGINAL predictive law of the intended
duration, `P(round(A*T) = t)`, integrated over `A` on a Gauss-Hermite grid. It
has the same `(n, 91)` shape and the same meaning as every arm since round 1,
so `clock_v3.score_arm_v3` scores it BLIND with no v5-specific code path. The
AR(1) arm is a copula: it preserves every marginal exactly, so its `pmf` is the
inner arm's and its CRPS / PIT are identical to the reference BY CONSTRUCTION,
which is stated in the pre-registration rather than discovered afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss

from cbb_sim.models import clock as ck

GRID = np.arange(ck.DURATION_CAP + 1, dtype=np.float64)
N_GRID = len(GRID)


def gh_nodes(k: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """(z, w) standard-normal Gauss-Hermite nodes, weights summing to 1."""
    z, w = hermegauss(k)
    return z, w / w.sum()


def latent_values(sigma: np.ndarray | float, z: np.ndarray,
                  m: np.ndarray | float | None = None) -> np.ndarray:
    """`A = exp(sigma*z + m)`.

    ROUND 5 (`m is None`, the DEFAULT): `m = -sigma^2/2`, so `E[A] = 1` exactly
    and the conditional MEAN duration is unchanged. Every round-5 number is
    reproduced bit for bit by this branch.

    ROUND 5b (`m` given, `experiments.md` section 21): the LOCATION of the
    latent is what round 5b varies. `E[A] = exp(m + sigma^2/2)` and
    `E[1/A] = exp(-m + sigma^2/2)`, and the quantity every gate reads is the
    possession COUNT `P = 1200/Dbar`, which is the RECIPROCAL of the duration
    mean. `m = +sigma^2/2` therefore makes `E[1/A] = 1`, i.e. it is the
    mean-preserving specification ON THE COUNT SCALE (arm B1); round 5's
    `m = -sigma^2/2` is mean-preserving on the DURATION scale and raises
    `E[P]` by `exp(sigma^2)` (section 20.3's Jensen term, section 21's
    arithmetic). `m` is an analytic identity of the family (B1) or a method-of-
    moments estimate on TRAINING rows (B2) -- never an adjustment to output."""
    s = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    if m is None:
        loc = -0.5 * s ** 2
    else:
        loc = np.broadcast_to(np.atleast_1d(np.asarray(m, dtype=np.float64)),
                              s.shape)
    return np.exp(s[:, None] * z[None, :] + loc[:, None])


def loc_for(sigma: np.ndarray | float, kind: str,
            log_c: float = 0.0) -> np.ndarray | float:
    """The round-5b latent location `m` for one arm (section 21.2).

    `minus_half` = round 5's A1 (E[A]=1); `plus_half` = B1 (E[1/A]=1);
    `log_c` = B2, whose `E[A] = c` is a fitted training moment."""
    s2 = np.asarray(sigma, dtype=np.float64) ** 2
    if kind == "minus_half":
        return -0.5 * s2
    if kind == "plus_half":
        return 0.5 * s2
    if kind == "log_c":
        return float(log_c) - 0.5 * s2
    raise ValueError(f"unknown latent location kind {kind!r}")


def transfer(a: float) -> np.ndarray:
    """The (91, 91) 0/1 matrix sending `p(t)` to `p'(t') = P(round(a*T) = t')`.

    Durations live on the integer grid the whole project uses, so the scaled
    draw is rounded back onto it. Rounding is the ONLY place a latent arm can
    move a mean, and the grader measures that movement (N2/N4) rather than
    assuming it away."""
    idx = np.clip(np.rint(a * GRID).astype(np.int64), 0, ck.DURATION_CAP)
    S = np.zeros((N_GRID, N_GRID), dtype=np.float64)
    S[np.arange(N_GRID), idx] = 1.0
    return S


@dataclass
class LatentArm:
    """The reference arm's law, scale-mixed by a per-(game, unit) latent.

    `sigma` is either a scalar (arms A1/A2/A3) or a callable `df -> sigma per
    row` (arm A6, whose dispersion is a fitted function of pregame tempo). The
    JOINT structure -- whether the latent is shared by both offences, drawn per
    offence, or correlated across them -- lives in the grader's variance
    algebra and in the engine adapter, not here: a single possession's MARGINAL
    law is the same under all three, which is exactly why the primary metric of
    round 5 is a per-game quantity and CRPS is only a no-regression line."""

    inner: object
    sigma: float | object
    k_nodes: int = 9
    sigma_buckets: int = 20
    #: round-5b latent LOCATION (section 21.2): "minus_half" reproduces round 5
    #: exactly, "plus_half" is B1, "log_c" is B2 with `log_c` fitted.
    loc_kind: str = "minus_half"
    log_c: float = 0.0

    def __post_init__(self) -> None:
        self._z, self._w = gh_nodes(self.k_nodes)

    def _sigma_rows(self, df: pd.DataFrame) -> np.ndarray:
        if callable(self.sigma):
            return np.asarray(self.sigma(df), dtype=np.float64)
        return np.full(len(df), float(self.sigma), dtype=np.float64)

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        p = np.asarray(self.inner.pmf(df), dtype=np.float64)
        s = self._sigma_rows(df)
        out = np.zeros_like(p)
        if np.ptp(s) < 1e-12:
            a = latent_values(float(s[0]), self._z,
                              loc_for(float(s[0]), self.loc_kind, self.log_c))[0]
            for j, wj in enumerate(self._w):
                out += wj * (p @ transfer(float(a[j])))
            return out
        # Arm A6 only: bucket the row sigma so the transfer matrix can still be
        # shared. The bucketing is a SCORING approximation and is reported; the
        # primary metric uses the exact per-row sigma (see the grader's m1/m2).
        edges = np.quantile(s, np.linspace(0, 1, self.sigma_buckets + 1))
        bid = np.clip(np.searchsorted(edges, s, side="right") - 1,
                      0, self.sigma_buckets - 1)
        for b in np.unique(bid):
            r = np.flatnonzero(bid == b)
            sb = float(s[r].mean())
            a = latent_values(sb, self._z,
                              loc_for(sb, self.loc_kind, self.log_c))[0]
            for j, wj in enumerate(self._w):
                out[r] += wj * (p[r] @ transfer(float(a[j])))
        return out


@dataclass
class Ar1CopulaArm:
    """Gaussian-copula AR(1) on the duration residual within (game, offence).

    Marginals are preserved exactly, so `pmf` is the inner arm's and the arm is
    identical to the reference on CRPS, the log score and PIT. It differs ONLY
    in the joint law across a game, which is what round 5 is about. Kept as a
    class so the grader treats it like every other arm."""

    inner: object
    rho: float

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.inner.pmf(df), dtype=np.float64)


def ar1_mean_inflation(rho: float, n: np.ndarray) -> np.ndarray:
    """Var(mean of n AR(1) terms) / (sigma^2 / n), exactly.

        1 + 2/n * sum_{k=1..n-1} (n-k) rho^k
      = (1+rho)/(1-rho) - 2 rho (1 - rho^n) / (n (1-rho)^2)
    """
    n = np.asarray(n, dtype=np.float64)
    if abs(rho) < 1e-12:
        return np.ones_like(n)
    return ((1 + rho) / (1 - rho)
            - 2 * rho * (1 - rho ** n) / (n * (1 - rho) ** 2))
