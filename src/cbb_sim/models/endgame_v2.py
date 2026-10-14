"""endgame_v2.py -- clock experiments.md section 46 window laws (blocks A and B), NOT ADOPTED, default-off.

Shared by the trainer (`scripts/train_endgame_clock_v2.py`), the offline grader inputs and the engine adapter
(`engine/endgame_adapter.py`, `ENGINE_ENDGAME=CA1|CA2|CA3|FP1`, `ENGINE_FPS_MAKE=FP2`). Every law acts on the BASE
(served) law's own row prediction; nothing here touches sim output.

Block A (cell term ON the served K2 window law). Window: regulation, possession start <= 240 s. Cell = the nine
margin buckets (`endgame.roleband`: leader 1-3/4-7/8-12/13+, trailer the same, tied) x five time buckets
(`endgame.tbucket`) = 45 cells per half (period 2 selects, period 1 fitted and reported).
  CA1  RATIO form: p(d | row, cell) = p0(d | row) r_cell(bin(d)) / Z over 8 duration bins (`BIN_EDGES`), one ratio
       table for the marginal first-chance law and one for K2's end-class-conditional law (each fitted against the
       law it multiplies). r = observed / expected bin counts (expected = the served row law's mass; a horn-censored
       row's observation is spread over the bins at or above its censoring time in proportion to the served law),
       shrunk hierarchically with m pseudo-rows: bucket r_b = (n_b OE_b + m) / (n_b + m), cell
       r_c = (n_c OE_c + m r_b) / (n_c + m).
  CA2  CELL-AFT: the served pmf scaled by k = exp(beta_cell) (`clock_r7.scale_pmf`, continuous CDF); beta fitted by
       the round-8 M2 Poisson pseudo-likelihood with the K2 conditional-cell fixed effects on uncensored window rows,
       at bucket level and at cell level, then shrunk: beta_b' = n_b / (n_b + m) beta_b,
       beta_c' = (n_c beta_c + m beta_b') / (n_c + m) (n = uncensored rows).
  CA3  CA2 with beta smooth in margin per time bucket: log k = b_L max(m, 0)/12 + b_T max(-m, 0)/12, (b_L, b_T) per
       time bucket and half (10 per half), same estimator, no shrink.
Block B (final-possession conversion). FPS = regulation, possession start <= 30 s, offence trailing by 1-3 (the only
fitted cell; ruling 9), first chance.
  FP1  possession_outcome class probabilities on FPS rows: p propto p0 * r, r = (n O/E + m) / (n + m), m = 50, per half.
  FP2  FP1 plus an FPS intercept on the fg_make logit of first-chance shots in FPS: logit p = logit p0 + delta_h,
       delta_h = n / (n + m) * delta_mle (shrunk to 0, m = 50), per half and pooled over shot classes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cbb_sim.models.endgame import N_GRID, N_RB, N_TB, roleband, scale_pmf, tbucket

BIN_EDGES = np.array([0, 3, 6, 9, 13, 18, 24, 31, 91])        # 8 duration bins over 0..90 s
N_BIN = len(BIN_EDGES) - 1
BIN_OF = np.searchsorted(BIN_EDGES, np.arange(N_GRID), side="right") - 1
FPS_SEC = 30


def _half(period) -> np.ndarray:
    return np.clip(np.asarray(period).astype(np.int64) - 1, 0, 1)


def cell_index(period, sec, m) -> np.ndarray:
    """Flat (half, roleband, tbucket) index in [0, 90)."""
    return (_half(period) * N_RB + roleband(m)) * N_TB + tbucket(sec)


def bucket_index(period, sec) -> np.ndarray:
    return _half(period) * N_TB + tbucket(sec)


# ---------------------------------------------------------------------------------------------------------------
# CA1
# ---------------------------------------------------------------------------------------------------------------
def oe_counts(p0: np.ndarray, y: np.ndarray, cen: np.ndarray, key: np.ndarray, n_key: int) -> tuple:
    """(O, E, n) per key x bin. Censored rows spread their unit over bins >= y in proportion to p0."""
    p0 = np.asarray(p0, dtype=np.float64)
    y = np.clip(np.asarray(y).astype(np.int64), 0, N_GRID - 1)
    cen = np.asarray(cen, dtype=bool)
    Eb = np.zeros((len(p0), N_BIN))
    for b in range(N_BIN):
        Eb[:, b] = p0[:, BIN_EDGES[b]:BIN_EDGES[b + 1]].sum(1)
    Ob = np.zeros_like(Eb)
    unc = np.flatnonzero(~cen)
    Ob[unc, BIN_OF[y[unc]]] = 1.0
    c = np.flatnonzero(cen)
    if len(c):
        tail = p0[c] * (np.arange(N_GRID)[None, :] >= y[c][:, None])
        tot = tail.sum(1, keepdims=True)
        tail = np.where(tot > 0, tail / np.maximum(tot, 1e-300), 0.0)
        for b in range(N_BIN):
            Ob[c, b] = tail[:, BIN_EDGES[b]:BIN_EDGES[b + 1]].sum(1)
        # a censored row the served law gives no tail mass contributes its unit to the censoring bin
        z = tot[:, 0] <= 0
        Ob[c[z], BIN_OF[y[c[z]]]] = 1.0
    obs = np.zeros((n_key, N_BIN))
    E = np.zeros((n_key, N_BIN))
    np.add.at(obs, key, Ob)
    np.add.at(E, key, Eb)
    n = np.bincount(key, minlength=n_key).astype(float)
    return obs, E, n


def shrink_ratio(obs, E, n, parent, m: float) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        oe = np.where(E > 1e-12, obs / E, 1.0)
    return (n[:, None] * oe + m * parent) / (n[:, None] + m)


@dataclass
class CA1Law:
    rm: np.ndarray          # (2, 9, 5, 8) marginal ratio
    re: np.ndarray          # (2, 9, 5, 8) conditional ratio
    n: np.ndarray           # (2, 9, 5)
    m: float
    info: dict

    @staticmethod
    def _apply(r, p0, period, sec, m):
        h = _half(period)
        rr = r[h, roleband(m), tbucket(sec)][:, BIN_OF]
        q = np.asarray(p0, dtype=np.float64) * rr
        s = q.sum(1, keepdims=True)
        return np.where(s > 0, q / np.maximum(s, 1e-300), np.asarray(p0, dtype=np.float64))

    def apply(self, p0, period, sec, m, end1=None) -> np.ndarray:
        return self._apply(self.rm, p0, period, sec, m)

    def apply_end(self, p0, period, sec, m, end1=None) -> np.ndarray:
        return self._apply(self.re, p0, period, sec, m)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, rm=self.rm, re=self.re, n=self.n, m=np.float64(self.m),
                            info=np.array(json.dumps(self.info)))

    @classmethod
    def load(cls, path: Path) -> CA1Law:
        z = np.load(path, allow_pickle=False)
        return cls(rm=z["rm"], re=z["re"], n=z["n"], m=float(z["m"]), info=json.loads(str(z["info"])))


def fit_ca1(period, sec, m, y, cen, p0, p0e, m_pseudo: float) -> CA1Law:
    kc = cell_index(period, sec, m)
    kb = bucket_index(period, sec)
    par_b = np.array([(c // (N_RB * N_TB)) * N_TB + (c % N_TB) for c in range(2 * N_RB * N_TB)])
    out = {}
    for nm, p in (("rm", p0), ("re", p0e)):
        Ob, Eb, nb = oe_counts(p, y, cen, kb, 2 * N_TB)
        rb = shrink_ratio(Ob, Eb, nb, 1.0, m_pseudo)
        Oc, Ec, nc = oe_counts(p, y, cen, kc, 2 * N_RB * N_TB)
        out[nm] = shrink_ratio(Oc, Ec, nc, rb[par_b], m_pseudo).reshape(2, N_RB, N_TB, N_BIN)
        out["n"] = nc.reshape(2, N_RB, N_TB)
    return CA1Law(rm=out["rm"], re=out["re"], n=out["n"], m=float(m_pseudo),
                  info={"n_rows": int(len(kc)), "m": float(m_pseudo), "bins": BIN_EDGES.tolist()})


# ---------------------------------------------------------------------------------------------------------------
# CA2 / CA3 (AFT scale)
# ---------------------------------------------------------------------------------------------------------------
def poisson_fe(y: np.ndarray, X: np.ndarray, key: np.ndarray, max_iter: int = 100) -> tuple[np.ndarray, dict]:
    """`clock_r8.fit_poisson` (round-8 M2 estimator, cell effects profiled out) with a minimum-norm step, so a column
    the fixed effects absorb gets coefficient 0 instead of a singular solve. Deterministic."""
    _, inv = np.unique(key, return_inverse=True)
    b = np.zeros(X.shape[1])
    ysum = np.bincount(inv, weights=y)
    it = 0
    for it in range(max_iter):  # noqa: B007
        ee = np.exp(X @ b)
        mu_c = ysum / np.bincount(inv, weights=ee)
        mu = mu_c[inv] * ee
        g = X.T @ (y - mu)
        H = (X * mu[:, None]).T @ X
        wsum = np.bincount(inv, weights=mu)
        Xm = np.stack([np.bincount(inv, weights=mu * X[:, j]) for j in range(X.shape[1])], axis=1)
        inv_w = np.divide(1.0, wsum, out=np.zeros_like(wsum), where=wsum > 0)
        H = H - (Xm * inv_w[:, None]).T @ Xm
        step = np.linalg.lstsq(H, g, rcond=1e-10)[0]
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    return b, {"newton_iter": int(it + 1), "n_fit_rows": int(len(y))}


def ca3_features(period, sec, m) -> np.ndarray:
    """(n, 20): per half x time bucket, max(m,0)/12 and max(-m,0)/12."""
    kb = bucket_index(period, sec)
    mm = np.asarray(m, dtype=np.float64)
    X = np.zeros((len(kb), 2 * N_TB * 2))
    r = np.arange(len(kb))
    X[r, 2 * kb] = np.maximum(mm, 0) / 12.0
    X[r, 2 * kb + 1] = np.maximum(-mm, 0) / 12.0
    return X


@dataclass
class CAScaleLaw:
    """log k per row from a table: CA2 beta (2, 9, 5); CA3 coef (20,) on `ca3_features`."""

    arm: str
    beta: np.ndarray
    info: dict

    def log_k(self, period, sec, m) -> np.ndarray:
        if self.arm == "CA3":
            return ca3_features(period, sec, m) @ self.beta
        return self.beta.reshape(-1)[cell_index(period, sec, m)]

    def apply(self, p0, period, sec, m, end1=None) -> np.ndarray:
        return scale_pmf(p0, np.exp(self.log_k(period, sec, m)))

    apply_end = apply

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"law": self.arm, "beta": np.asarray(self.beta).tolist(), "info": self.info},
                                   indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CAScaleLaw:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(arm=d["law"], beta=np.asarray(d["beta"], dtype=np.float64), info=d.get("info", {}))


def fit_ca2(period, sec, m, y, fe_key, m_pseudo: float) -> CAScaleLaw:
    kb = bucket_index(period, sec)
    kc = cell_index(period, sec, m)
    nb, nc = 2 * N_TB, 2 * N_RB * N_TB
    Xb = np.zeros((len(y), nb))
    Xb[np.arange(len(y)), kb] = 1.0
    Xc = np.zeros((len(y), nc))
    Xc[np.arange(len(y)), kc] = 1.0
    yy = np.asarray(y, dtype=np.float64)
    bb, ib = poisson_fe(yy, Xb, fe_key)
    bc, ic = poisson_fe(yy, Xc, fe_key)
    n_b = np.bincount(kb, minlength=nb).astype(float)
    n_c = np.bincount(kc, minlength=nc).astype(float)
    bb_s = n_b / (n_b + m_pseudo) * bb
    par = np.array([(c // (N_RB * N_TB)) * N_TB + (c % N_TB) for c in range(nc)])
    bc_s = (n_c * bc + m_pseudo * bb_s[par]) / (n_c + m_pseudo)
    return CAScaleLaw(arm="CA2", beta=bc_s.reshape(2, N_RB, N_TB),
                      info={"m": float(m_pseudo), "beta_bucket_raw": bb.tolist(), "beta_bucket_shrunk": bb_s.tolist(),
                            "fit_bucket": ib, "fit_cell": ic, "n_cell": n_c.tolist()})


def fit_ca3(period, sec, m, y, fe_key) -> CAScaleLaw:
    b, info = poisson_fe(np.asarray(y, dtype=np.float64), ca3_features(period, sec, m), fe_key)
    return CAScaleLaw(arm="CA3", beta=b, info=info)


# ---------------------------------------------------------------------------------------------------------------
# Block B
# ---------------------------------------------------------------------------------------------------------------
def in_fps(period, sec, m) -> np.ndarray:
    mm = np.asarray(m)
    return (np.asarray(period) <= 2) & (np.asarray(sec) <= FPS_SEC) & (mm <= -1) & (mm >= -3)


@dataclass
class FPLaw:
    r: np.ndarray                  # (2, 6) class ratio per half on FPS rows
    delta: np.ndarray              # (2,) make-logit intercept per half (FP1: zeros)
    arm: str
    info: dict

    def apply(self, p0, period, sec, m) -> np.ndarray:
        q = np.asarray(p0, dtype=np.float64).copy()
        w = in_fps(period, sec, m)
        if w.any():
            qq = q[w] * self.r[_half(np.asarray(period)[w])]
            q[w] = qq / qq.sum(1, keepdims=True)
        return q

    def apply_make(self, p0, period, sec, m) -> np.ndarray:
        p = np.asarray(p0, dtype=np.float64).copy()
        w = in_fps(period, sec, m)
        if w.any() and np.any(self.delta != 0):
            pc = np.clip(p[w], 1e-9, 1 - 1e-9)
            z = np.log(pc / (1 - pc)) + self.delta[_half(np.asarray(period)[w])]
            p[w] = 1.0 / (1.0 + np.exp(-z))
        return p

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"law": self.arm, "r": self.r.tolist(), "delta": self.delta.tolist(),
                                    "info": self.info}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FPLaw:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(r=np.asarray(d["r"], dtype=np.float64), delta=np.asarray(d["delta"], dtype=np.float64),
                   arm=d["law"], info=d.get("info", {}))


def fit_fp_classes(period, y, p0, m_pseudo: float = 50.0) -> tuple[np.ndarray, dict]:
    h = _half(period)
    K = p0.shape[1]
    r = np.ones((2, K))
    info = {}
    for hh in (0, 1):
        s = h == hh
        n = float(s.sum())
        obs = np.bincount(np.asarray(y)[s].astype(np.int64), minlength=K).astype(float)
        E = np.asarray(p0, dtype=np.float64)[s].sum(0)
        oe = np.where(E > 0, obs / np.maximum(E, 1e-300), 1.0)
        r[hh] = (n * oe + m_pseudo) / (n + m_pseudo)
        info[f"period{hh + 1}"] = {"n": int(n), "O": obs.tolist(), "E": E.tolist()}
    return r, info


def fit_fp_make(period, made, p0, m_pseudo: float = 50.0) -> tuple[np.ndarray, dict]:
    from scipy.optimize import minimize_scalar
    h = _half(period)
    delta = np.zeros(2)
    info = {}
    for hh in (0, 1):
        s = h == hh
        y = np.asarray(made, dtype=np.float64)[s]
        pc = np.clip(np.asarray(p0, dtype=np.float64)[s], 1e-9, 1 - 1e-9)
        z0 = np.log(pc / (1 - pc))

        def nll(d, z0=z0, y=y):
            p = 1.0 / (1.0 + np.exp(-(z0 + d)))
            return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
        r = minimize_scalar(nll, bounds=(-3, 3), method="bounded", options={"xatol": 1e-8})
        n = float(s.sum())
        delta[hh] = n / (n + m_pseudo) * float(r.x)
        info[f"period{hh + 1}"] = {"n": int(n), "delta_mle": float(r.x), "delta_shrunk": float(delta[hh]),
                                   "mean_y": float(y.mean()) if n else None, "mean_p0": float(pc.mean()) if n else None}
    return delta, info


# ---------------------------------------------------------------------------------------------------------------
# CA1E (clock experiments.md s49, NOT ADOPTED, default off): within-cell env AFT scale x CA1 ratio on the de-scaled
# baseline. Step 1: log k = X c, X = [z_off] (+ [def_c] when fold 1 resolves it), fitted by the round-8 M2 Poisson
# pseudo-likelihood (`poisson_fe`) with fixed effects = window cell (half x 45) x K2 conditional finest cell, on
# uncensored window rows, so c is identified from within-cell variation only. Step 2: the CA1 ratio table fitted on
# durations (and censoring times) divided by k against the served row law. Served: p = scale_pmf(CA1(p0), k_row).
# No coefficient is shrunk or capped; m is the CA1 pseudo-count chosen blind on fold 1.
# ---------------------------------------------------------------------------------------------------------------
ENV_COLS = {"zoff": ("off_rating_off_c", "def_rating_def_c"), "defz": ("def_rating_def_c",)}


def env_design(df, cols: list[str]) -> np.ndarray:
    """(n, len(cols)) env design; z_off = off_c(offence) + def_c(defence), defz = def_c(defence). NaN -> 0."""
    out = np.zeros((len(df), len(cols)))
    for j, c in enumerate(cols):
        v = np.zeros(len(df))
        for src in ENV_COLS[c]:
            v = v + np.asarray(df[src], dtype=np.float64)
        out[:, j] = np.nan_to_num(v, nan=0.0)
    return out


def poisson_fe_cluster_se(y, X, key, cluster, b) -> np.ndarray:
    """Game-clustered sandwich SE of the profiled M2 Poisson coefficients at b."""
    _, inv = np.unique(key, return_inverse=True)
    y = np.asarray(y, dtype=np.float64)
    ee = np.exp(X @ b)
    mu = (np.bincount(inv, weights=y) / np.bincount(inv, weights=ee))[inv] * ee
    wsum = np.bincount(inv, weights=mu)
    xm = np.stack([np.bincount(inv, weights=mu * X[:, j]) for j in range(X.shape[1])], axis=1)
    xt = X - (xm / np.maximum(wsum, 1e-300)[:, None])[inv]
    H = (xt * mu[:, None]).T @ xt
    _, ci = np.unique(cluster, return_inverse=True)
    sc = np.stack([np.bincount(ci, weights=xt[:, j] * (y - mu)) for j in range(X.shape[1])], axis=1)
    Hi = np.linalg.pinv(H)
    V = Hi @ (sc.T @ sc) @ Hi
    return np.sqrt(np.diag(V))


@dataclass
class CA1ELaw:
    ca1: CA1Law
    cols: list
    coef: np.ndarray
    info: dict

    def log_k(self, df) -> np.ndarray:
        return env_design(df, self.cols) @ self.coef

    def apply(self, p0, period, sec, m, df=None, end1=None) -> np.ndarray:
        return scale_pmf(self.ca1.apply(p0, period, sec, m), np.exp(self.log_k(df)))

    def apply_end(self, p0, period, sec, m, df=None, end1=None) -> np.ndarray:
        return scale_pmf(self.ca1.apply_end(p0, period, sec, m), np.exp(self.log_k(df)))

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.ca1.save(d / "ca1e_ratio.npz")
        (d / "ca1e_env.json").write_text(json.dumps({"cols": list(self.cols), "coef": np.asarray(self.coef).tolist(),
                                                      "info": self.info}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> CA1ELaw:
        e = json.loads((Path(d) / "ca1e_env.json").read_text(encoding="utf-8"))
        return cls(ca1=CA1Law.load(Path(d) / "ca1e_ratio.npz"), cols=e["cols"],
                   coef=np.asarray(e["coef"], dtype=np.float64), info=e.get("info", {}))


def fit_env_within(y, X, fe_key) -> tuple[np.ndarray, dict]:
    return poisson_fe(np.asarray(y, dtype=np.float64), X, fe_key)


def descale(y, k) -> np.ndarray:
    """Durations (or censoring times) divided by k on the integer grid (rounded, clipped to 0..90)."""
    return np.clip(np.rint(np.asarray(y, dtype=np.float64) / np.asarray(k, dtype=np.float64)), 0, N_GRID - 1).astype(np.int64)
