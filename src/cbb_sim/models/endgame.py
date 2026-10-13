"""endgame.py -- late-game round 7 window laws (late_game/experiments.md section 15), NOT ADOPTED, default-off.

One module holds the fitted laws so that the offline grader and the engine serve the SAME code:
`train_endgame_{clock,event}_v1.py` fit and pickle-free-save them (npz / json), `engine/endgame_adapter.py` loads
them behind `ENGINE_ENDGAME`. Every law acts on the BASE (served) law's own row prediction; nothing here touches
sim output.

Window (s15.A.2): regulation, periods 1-2, possession start `seconds_remaining <= 240`, offence-perspective
`score_diff` at the start. Cells: role-band (leader 0-3/4-7/8-12/13+, trailer the same, tied) x time bucket
((180,240], (120,180], (60,120], (30,60], (0,30]).

GT2  smooth margin response on the served duration law, accelerated-failure-time scale on a continuous CDF
     (`clock_r7.scale_pmf`): log k = w(t) (b_L max(m,0)/12 + b_T max(-m,0)/12), w(t) = min(1, (240 - t)/120), two
     fitted parameters per half. Applied identically to the marginal first-chance law and to K2's
     end-class-conditional re-draw law.
GT1  cell law: first-chance duration pmf (Kaplan-Meier with the round-3 tail rule) by role-band x time bucket,
     hierarchically shrunk with m = 50 pseudo-rows: cell -> time bucket (per half) -> the served row law GT0:
         p_row = a_c p_c + (1 - a_c) (a_b p_b + (1 - a_b) p0_row),   a = n / (n + m).
     The conditional (re-draw) law is the same hierarchy with the chance-1 end class added to every level and the
     served conditional row law as the root.
EV1  possession_outcome class probabilities refit on window first chances by role-band x time bucket, shrunk to the
     served row law EV0 with m = 50 pseudo-rows, in the observed/expected form that keeps EV0's row-level effects
     (site main effect included, s15.A.2):  p_row,k  propto  p0_row,k * r_c,k,
         r_c,k = (n_c O_c,k / E_c,k + m) / (n_c + m)  (= (O + m pbar0) / (E + m pbar0)),
     O = observed class counts, E = summed EV0 probabilities, n = rows in the cell (training seasons only).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WIN_SEC = 240
M_PSEUDO = 50.0
N_RB, N_TB, N_END, N_GRID = 9, 5, 5, 91


def band(a) -> np.ndarray:
    a = np.abs(np.asarray(a))
    return np.select([a <= 3, a <= 7, a <= 12], [0, 1, 2], 3).astype(np.int64)


def roleband(m) -> np.ndarray:
    m = np.asarray(m)
    b = band(m)
    return np.where(m > 0, b, np.where(m < 0, 4 + b, 8)).astype(np.int64)


def tbucket(sec) -> np.ndarray:
    s = np.asarray(sec)
    return np.select([s > 180, s > 120, s > 60, s > 30], [0, 1, 2, 3], 4).astype(np.int64)


def in_window(period, sec, periods=(1, 2)) -> np.ndarray:
    return np.isin(np.asarray(period), periods) & (np.asarray(sec) <= WIN_SEC)


def scale_pmf(p0: np.ndarray, k: np.ndarray) -> np.ndarray:
    from cbb_sim.models.clock_r7 import scale_pmf as _s
    return _s(np.asarray(p0, dtype=np.float64), np.asarray(k, dtype=np.float64))


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class GT2Law:
    """b[h] = (b_L, b_T) for half h = 0 (period 1), 1 (period 2)."""

    b: np.ndarray                  # (2, 2)
    info: dict

    @staticmethod
    def log_k(b: np.ndarray, period, sec, m) -> np.ndarray:
        h = np.clip(np.asarray(period).astype(np.int64) - 1, 0, 1)
        t = np.asarray(sec, dtype=np.float64)
        mm = np.asarray(m, dtype=np.float64)
        w = np.clip((WIN_SEC - t) / 120.0, 0.0, 1.0)
        return w * (b[h, 0] * np.maximum(mm, 0) / 12.0 + b[h, 1] * np.maximum(-mm, 0) / 12.0)

    def apply(self, p0, period, sec, m, end1=None) -> np.ndarray:
        return scale_pmf(p0, np.exp(self.log_k(self.b, period, sec, m)))

    apply_end = apply

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"law": "GT2", "b": self.b.tolist(), "info": self.info}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> GT2Law:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(b=np.asarray(d["b"], dtype=np.float64), info=d.get("info", {}))


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class GT1Law:
    """Level pmfs and row counts. Marginal: bucket (2, 5, G), cell (2, 9, 5, G). Conditional: the same with a trailing
    end-class axis before G."""

    pb: np.ndarray
    nb: np.ndarray
    pc: np.ndarray
    nc: np.ndarray
    pbe: np.ndarray
    nbe: np.ndarray
    pce: np.ndarray
    nce: np.ndarray
    m: float
    info: dict

    def _mix(self, p0, h, rb, tb, pb, nb, pc, nc) -> np.ndarray:
        ab = (nb / (nb + self.m))[:, None]
        ac = (nc / (nc + self.m))[:, None]
        return ac * pc + (1.0 - ac) * (ab * pb + (1.0 - ab) * np.asarray(p0, dtype=np.float64))

    def apply(self, p0, period, sec, m, end1=None) -> np.ndarray:
        h = np.clip(np.asarray(period).astype(np.int64) - 1, 0, 1)
        rb, tb = roleband(m), tbucket(sec)
        return self._mix(p0, h, rb, tb, self.pb[h, tb], self.nb[h, tb], self.pc[h, rb, tb], self.nc[h, rb, tb])

    def apply_end(self, p0, period, sec, m, end1) -> np.ndarray:
        h = np.clip(np.asarray(period).astype(np.int64) - 1, 0, 1)
        rb, tb = roleband(m), tbucket(sec)
        e = np.clip(np.asarray(end1).astype(np.int64), 0, N_END - 1)
        return self._mix(p0, h, rb, tb, self.pbe[h, tb, e], self.nbe[h, tb, e], self.pce[h, rb, tb, e],
                         self.nce[h, rb, tb, e])

    def save(self, path: Path) -> None:
        np.savez_compressed(path, pb=self.pb, nb=self.nb, pc=self.pc, nc=self.nc, pbe=self.pbe, nbe=self.nbe,
                            pce=self.pce, nce=self.nce, m=np.float64(self.m),
                            info=np.array(json.dumps(self.info)))

    @classmethod
    def load(cls, path: Path) -> GT1Law:
        z = np.load(path, allow_pickle=False)
        return cls(pb=z["pb"], nb=z["nb"], pc=z["pc"], nc=z["nc"], pbe=z["pbe"], nbe=z["nbe"], pce=z["pce"],
                   nce=z["nce"], m=float(z["m"]), info=json.loads(str(z["info"])))


def fit_gt1(period, sec, m, end1, y, cen, p0, p0e, m_pseudo: float = M_PSEUDO) -> GT1Law:
    """Kaplan-Meier pmfs per level (clock_v3's tail rule: the unresolved survival goes to t > t* in proportion to the
    parent's pmf; the parent of a bucket is the mean served row law over the bucket's rows)."""
    from cbb_sim.models.clock_v3 import kaplan_meier_pmf_v3 as km
    h = np.clip(np.asarray(period).astype(np.int64) - 1, 0, 1)
    rb, tb = roleband(m), tbucket(sec)
    e = np.clip(np.asarray(end1).astype(np.int64), 0, N_END - 1)
    y = np.clip(np.asarray(y).astype(np.int64), 0, N_GRID - 1)
    cen = np.asarray(cen, dtype=bool)

    def level(key_b, nkb, key_c, nkc, parent_c_of_b, p0x):
        # bucket level: parent (tail rule only) = mean served row law over the bucket's rows
        mean0 = np.zeros((nkb, N_GRID))
        np.add.at(mean0, key_b, p0x)
        cnt = np.bincount(key_b, minlength=nkb).astype(float)
        mean0 = np.where(cnt[:, None] > 0, mean0 / np.maximum(cnt, 1)[:, None], 1.0 / N_GRID)
        pb, nb, _ = km(key_b, y, cen, nkb, mean0, np.arange(nkb))
        pb = pb / np.maximum(pb.sum(1, keepdims=True), 1e-300)
        pc, nc, _ = km(key_c, y, cen, nkc, pb, parent_c_of_b)
        pc = pc / np.maximum(pc.sum(1, keepdims=True), 1e-300)
        return pb, nb, pc, nc

    # marginal
    kb = h * N_TB + tb
    kc = (h * N_RB + rb) * N_TB + tb
    par = np.array([(c // (N_RB * N_TB)) * N_TB + (c % N_TB) for c in range(2 * N_RB * N_TB)])
    pb, nb, pc, nc = level(kb, 2 * N_TB, kc, 2 * N_RB * N_TB, par, p0)
    # conditional on the chance-1 end class
    kbe = kb * N_END + e
    kce = kc * N_END + e
    pare = np.array([par[c // N_END] * N_END + (c % N_END) for c in range(2 * N_RB * N_TB * N_END)])
    pbe, nbe, pce, nce = level(kbe, 2 * N_TB * N_END, kce, 2 * N_RB * N_TB * N_END, pare, p0e)
    return GT1Law(pb=pb.reshape(2, N_TB, N_GRID), nb=nb.reshape(2, N_TB),
                  pc=pc.reshape(2, N_RB, N_TB, N_GRID), nc=nc.reshape(2, N_RB, N_TB),
                  pbe=pbe.reshape(2, N_TB, N_END, N_GRID), nbe=nbe.reshape(2, N_TB, N_END),
                  pce=pce.reshape(2, N_RB, N_TB, N_END, N_GRID), nce=nce.reshape(2, N_RB, N_TB, N_END),
                  m=float(m_pseudo), info={"n_rows": int(len(y))})


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class EV1Law:
    r: np.ndarray                  # (2, 9, 5, 6) class ratio per half x role-band x bucket
    n: np.ndarray                  # (2, 9, 5)
    info: dict

    def apply(self, p0, period, sec, m) -> np.ndarray:
        h = np.clip(np.asarray(period).astype(np.int64) - 1, 0, 1)
        q = np.asarray(p0, dtype=np.float64) * self.r[h, roleband(m), tbucket(sec)]
        return q / q.sum(1, keepdims=True)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, r=self.r, n=self.n, info=np.array(json.dumps(self.info)))

    @classmethod
    def load(cls, path: Path) -> EV1Law:
        z = np.load(path, allow_pickle=False)
        return cls(r=z["r"], n=z["n"], info=json.loads(str(z["info"])))


def fit_ev1(period, sec, m, y, p0, m_pseudo: float = M_PSEUDO) -> EV1Law:
    h = np.clip(np.asarray(period).astype(np.int64) - 1, 0, 1)
    c = (h * N_RB + roleband(m)) * N_TB + tbucket(sec)
    K = p0.shape[1]
    nc = 2 * N_RB * N_TB
    obs = np.zeros((nc, K))
    E = np.zeros((nc, K))
    np.add.at(obs, (c, np.asarray(y).astype(np.int64)), 1.0)
    np.add.at(E, c, np.asarray(p0, dtype=np.float64))
    n = np.bincount(c, minlength=nc).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        oe = np.where(E > 0, obs / E, 1.0)
    r = (n[:, None] * oe + m_pseudo) / (n[:, None] + m_pseudo)
    return EV1Law(r=r.reshape(2, N_RB, N_TB, K), n=n.reshape(2, N_RB, N_TB),
                  info={"n_rows": int(len(c)), "m": float(m_pseudo)})
