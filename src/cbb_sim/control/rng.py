"""
rng.py -- counter-based RNG keyed on (seed, game_id, family).

CLAUDE.md modelling rule: "RNG seeded on (seed, game_id, family). Paired
bake-off arms share aligned streams."

A per-(seed, game_id) `np.random.Generator` would satisfy the rule but cost a
SeedSequence spawn per game-sim (1.1M of them for a 200-seed season), so the
streams here are counter-based instead: a 64-bit key is mixed from
(seed, game_id, family) with splitmix64, and draw j of that stream is
splitmix64(key + j * PHI). Nothing about a game's stream depends on which other
games are in the run, which is exactly the property the rule exists to give:
arm A and arm B (and the noise-floor run at seed offset +1000) can be differenced
game by game without the streams drifting apart, and dropping a game from the
universe does not move any other game's draws.

Uniforms are turned into variates by inverse CDF (`scipy.special.ndtri`,
`scipy.stats.poisson/nbinom/binom.ppf`, `scipy.special.betaincinv`), which is
exact and vectorises over per-row parameters.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats

_M1 = np.uint64(0xFF51AFD7ED558CCD)
_M2 = np.uint64(0xC4CEB9FE1A85EC53)
_PHI = np.uint64(0x9E3779B97F4A7C15)
_S33 = np.uint64(33)
_S11 = np.uint64(11)
_TWO53 = np.float64(1.0 / (1 << 53))


_MASK64 = 0xFFFFFFFFFFFFFFFF


def _mix64(z: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser (murmur3 fmix64), vectorised over uint64.

    Unsigned 64-bit multiplication is meant to wrap here; numpy's overflow
    warning is the whole point of the algorithm, so it is silenced locally."""
    z = np.asarray(z, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> _S33)) * _M1
        z = (z ^ (z >> _S33)) * _M2
        return z ^ (z >> _S33)


def family_hash(family: str) -> np.uint64:
    """Stable 64-bit hash of a family label (FNV-1a; Python's hash() is salted
    per process and would not reproduce across runs)."""
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    for ch in family.encode("utf-8"):
        h = ((h ^ ch) * prime) & _MASK64
    return np.uint64(h)


def stream_keys(seed: int, game_ids: np.ndarray, family: str = "control") -> np.ndarray:
    """One 64-bit key per game for a given seed and family."""
    gid = np.asarray(game_ids, dtype=np.uint64)
    s = np.uint64((int(seed) * 0x9E3779B97F4A7C15) & _MASK64)
    with np.errstate(over="ignore"):
        base = gid * _PHI
    return _mix64(_mix64(base) ^ s ^ family_hash(family))


def uniforms(keys: np.ndarray, index: int) -> np.ndarray:
    """Draw `index` of each key's stream, as a float64 uniform on [0, 1)."""
    off = np.uint64((int(index) * 0x9E3779B97F4A7C15) & _MASK64)
    with np.errstate(over="ignore"):
        z = _mix64(np.asarray(keys, dtype=np.uint64) + off)
    u = (z >> _S11).astype(np.float64) * _TWO53
    # Guard the open interval so ndtri / ppf never see an exact 0 or 1.
    return np.clip(u, 1e-12, 1.0 - 1e-12)


# ---------------------------------------------------------------------------
# Inverse-CDF variate helpers (all vectorised over per-row parameters)
# ---------------------------------------------------------------------------
def normal(keys: np.ndarray, index: int, mean: np.ndarray, sd: float) -> np.ndarray:
    return mean + sd * special.ndtri(uniforms(keys, index))


def poisson(keys: np.ndarray, index: int, mu: np.ndarray) -> np.ndarray:
    return stats.poisson.ppf(uniforms(keys, index), np.maximum(mu, 1e-9))


def negbin(keys: np.ndarray, index: int, mu: np.ndarray, alpha: float) -> np.ndarray:
    """NB2 parameterisation: Var = mu + alpha * mu^2, i.e. size n = 1/alpha and
    success probability p = n / (n + mu)."""
    mu = np.maximum(mu, 1e-9)
    n = 1.0 / alpha
    p = n / (n + mu)
    return stats.nbinom.ppf(uniforms(keys, index), n, p)


def beta_binomial(
    keys: np.ndarray, index_beta: int, index_binom: int,
    trials: np.ndarray, p: np.ndarray, rho: float,
) -> np.ndarray:
    """Beta-Binomial as the explicit mixture: draw the per-game success
    probability from Beta(a, b) with mean p and intra-cluster correlation rho
    (a + b = (1 - rho) / rho), then draw Binomial(trials, that probability).
    rho == 0 collapses to a plain Binomial."""
    trials = np.asarray(trials, dtype=np.float64)
    if rho <= 0:
        pp = p
    else:
        m = (1.0 - rho) / rho
        a = np.clip(p * m, 1e-6, None)
        b = np.clip((1.0 - p) * m, 1e-6, None)
        pp = special.betaincinv(a, b, uniforms(keys, index_beta))
    out = np.zeros_like(trials)
    nz = trials > 0
    if nz.any():
        out[nz] = stats.binom.ppf(uniforms(keys, index_binom)[nz], trials[nz], np.clip(pp, 1e-9, 1 - 1e-9)[nz])
    return out
