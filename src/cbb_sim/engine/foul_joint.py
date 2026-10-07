"""Joint foul accrual + FT-trip production (possession-outcome round 7).

Pre-registration: `docs/models/possession_outcome/experiments.md` section 20.

DEFAULT OFF. `ENGINE_FOUL_JOINT` unset (or "reference") makes `load()` return
None and `loop.py` takes exactly the served path, RNG draws included.

An arm serves up to three lookup tables, all exported from FOLD-2 TRAIN fits
(`scripts/build_foul_joint_lut_v1.py`); nothing here is fitted to sim output:

* ``accrual``  P(non-trip foul charged to the DEFENCE this possession), indexed
  like round 6's table: (period idx, clock bucket, margin bucket, def fouls,
  off fouls, site), every count the ENGINE-definition state (fouls before the
  possession's own fouls).
* ``off``      optional, P(foul charged to the OFFENCE this possession), the
  explicit offensive-foul mechanism, same index plus `ended_tov` (0/1).
* ``trip``     optional, per-chance logit offsets for the two FT-trip classes,
  indexed (half, live defence fouls 0..10, foul-differential bucket); the
  served possession-outcome probabilities are shifted by them and the other four
  classes rescaled so each row still sums to one.

The defence and offence draws share the possession's single `foul_accrual`
uniform (u < p_def: defence foul; p_def <= u < p_def + p_off: offence foul), so
no RNG family is added and the served streams are untouched.

Round 8 (experiments.md s22/s23): optional per-game WHISTLE latent. One draw per
simulated game from its own stream family `foul_whistle` keyed on (seed, game_id);
multiplier A = exp(s z - s^2/2) (E[A] = 1, the clock v5b pace-latent rule) on the
ODDS of the fouling team's accrual draw and of both FT-trip classes. "shared": one
z for both teams; "unshared" (control): an independent z per team, home using the
shared column. s^2 is FITTED (`scripts/exp_foul_whistle_var_v1.py`, train seasons
only). An arm without a whistle takes exactly the round-7 code path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LUT_DIR = Path("data/processed/models/possession_outcome/round7")
#: SERVED DEFAULT, ADOPTED 2026-10-01 (Decision 11 set, PM under user delegation;
#: docs/tests/adoption_served_v2_2026-10-01.md): `loop.py` passes this when
#: `ENGINE_FOUL_JOINT` is unset. `reference` reproduces served-v1 (no tables).
DEFAULT = "R9ao3"

#: arm -> which tables it serves (file stems under LUT_DIR)
ARMS = {
    "CL1": {"accrual": "lut_acc_A1_F2"},
    "CL2a": {"accrual": "lut_acc_A2_F2"},
    "CL2": {"accrual": "lut_acc_A2_F2", "trip": "lut_trip_T2c_F2"},
    "CL3": {"accrual": "lut_acc_D2_F2", "off": "lut_off_O2_F2", "trip": "lut_trip_T2c_F2"},
    "CL4": {"accrual": "lut_acc_D2_F2", "off": "lut_off_O2_F2", "trip": "lut_trip_T2lab_F2"},
    # round 8: R8a == CL2a and R8b == CL2 (section 23), plus the whistle arms
    "R8a": {"accrual": "lut_acc_A2_F2"},
    "R8aS": {"accrual": "lut_acc_A2_F2", "whistle": "shared"},
    "R8b": {"accrual": "lut_acc_A2_F2", "trip": "lut_trip_T2c_F2"},
    "R8bS": {"accrual": "lut_acc_A2_F2", "trip": "lut_trip_T2c_F2", "whistle": "shared"},
    "R8bU": {"accrual": "lut_acc_A2_F2", "trip": "lut_trip_T2c_F2", "whistle": "unshared"},
}
WHISTLE_VAR = "whistle_var_v1.json"
WHISTLE_FAMILY = "foul_whistle"

#: set by an instrumentation tap (scripts/run_foul_joint_tap_v1.py) in ITS OWN
#: process only; called with (p_def, p_off) at every accrual draw.
TAP_HOOK = None

_CACHE: dict = {}


def _npz(stem: str) -> dict:
    if stem not in _CACHE:
        z = np.load(LUT_DIR / f"{stem}.npz")
        _CACHE[stem] = {k: z[k] for k in z.files}
    return _CACHE[stem]


def diff_bucket(def_f: np.ndarray, off_f: np.ndarray) -> np.ndarray:
    x = def_f.astype(np.int64) - off_f.astype(np.int64)
    return np.where(x <= -2, 0, np.where(x >= 2, 2, 1))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


class FoulJoint:
    def __init__(self, arm: str):
        cfg = ARMS[arm]
        self.arm = arm
        self.acc = _npz(cfg["accrual"])
        self.off = _npz(cfg["off"]) if "off" in cfg else None
        self.trip = _npz(cfg["trip"]) if "trip" in cfg else None
        self.whistle = cfg.get("whistle")
        self.lnA = None          # (n, 2) log-multipliers, home column 0, away column 1

    def init_whistle(self, seeds, game_ids) -> None:
        """Draw the per-game latent. No-op (no stream touched) for a no-whistle arm."""
        if self.whistle is None:
            return
        import json

        from scipy.special import ndtri

        from cbb_sim.engine.rng import StreamBook
        s2 = float(json.loads((LUT_DIR / WHISTLE_VAR).read_text())["served_s2_F2"])
        sd = np.sqrt(s2)
        n = len(seeds)
        u = StreamBook(seeds, game_ids, families=(WHISTLE_FAMILY,)).draw_block(
            WHISTLE_FAMILY, np.arange(n), 2)
        z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        la = sd * z - 0.5 * s2
        if self.whistle == "shared":
            la[:, 1] = la[:, 0]
        elif self.whistle != "unshared":
            raise KeyError(f"unknown whistle mode {self.whistle!r}")
        self.lnA = la
        self.s2 = s2
    @staticmethod
    def _idx(t: dict, period, sec_rem, margin, def_f, off_f, site):
        maxf = int(t["max_fouls"])
        pi = np.where(period <= 1, 0, np.where(period == 2, 1, 2))
        ci = np.searchsorted(t["clock_cuts"], sec_rem, side="right")
        mi = np.searchsorted(t["margin_cuts"], margin, side="right")
        return (pi, ci, mi, np.clip(def_f, 0, maxf).astype(np.int64),
                np.clip(off_f, 0, maxf).astype(np.int64), site)

    def accrual(self, u, period, sec_rem, margin, def_f, off_f, site, ended_tov,
                rows=None, def_side=None, cal=None):
        """(defence foul mask, offence foul mask) for this possession.

        `cal` = (foul_cal.FoulCal, game index per row) or None (default: the served path, unchanged);
        when given, the calendar arm's table replaces the defence accrual probability (PO s32)."""
        ix = self._idx(self.acc, period, sec_rem, margin, def_f, off_f, site)
        p_def = self.acc["lut"][ix] if cal is None else cal[0].p_def(ix, cal[1])
        if self.off is not None:
            p_off = self.off["lut"][ix + (ended_tov.astype(np.int64),)]
        else:
            p_off = np.zeros_like(p_def)
        if self.lnA is not None:
            ds = np.asarray(def_side, dtype=np.int64)
            p_def = 1.0 / (1.0 + np.exp(-(_logit(p_def) + self.lnA[rows, ds])))
            if self.off is not None:
                p_off = 1.0 / (1.0 + np.exp(-(_logit(p_off) + self.lnA[rows, 1 - ds])))
        if TAP_HOOK is not None:
            TAP_HOOK(p_def, p_off)
        sf = u < p_def
        of = (u >= p_def) & (u < p_def + p_off)
        return sf, of

    def adjust_trips(self, probs, period, def_f, off_f, c_bonus: int, c_shoot: int,
                     rows=None, def_side=None):
        """Shift the two FT-trip classes by the fitted cell offsets (and the game's
        whistle, if the arm has one); rescale the rest."""
        if self.trip is None and self.lnA is None:
            return probs
        if self.trip is not None:
            maxf = int(self.trip["max_fouls"])
            h = (period >= 2).astype(np.int64)
            dc = np.clip(def_f, 0, maxf).astype(np.int64)
            db = diff_bucket(def_f, off_f)
            d_b = self.trip["delta_bonus"][h, dc, db]
            d_s = self.trip["delta_shoot"][h, dc, db]
        else:
            d_b = d_s = 0.0
        if self.lnA is not None:
            la = self.lnA[rows, np.asarray(def_side, dtype=np.int64)]
            d_b = d_b + la
            d_s = d_s + la
        pb, ps = probs[:, c_bonus], probs[:, c_shoot]
        pb2 = 1.0 / (1.0 + np.exp(-(_logit(pb) + d_b)))
        ps2 = 1.0 / (1.0 + np.exp(-(_logit(ps) + d_s)))
        tot = pb2 + ps2
        over = tot > 0.999
        if over.any():
            pb2 = np.where(over, pb2 * 0.999 / tot, pb2)
            ps2 = np.where(over, ps2 * 0.999 / tot, ps2)
        rest_old = np.maximum(1.0 - pb - ps, 1e-12)
        rest_new = 1.0 - pb2 - ps2
        out = probs * (rest_new / rest_old)[:, None]
        out[:, c_bonus] = pb2
        out[:, c_shoot] = ps2
        return out


def load(arm: str | None):
    """None for the served path (unset / 'reference'), else the arm's tables."""
    if not arm or arm == "reference":
        return None
    if arm.startswith("R9"):     # foul round 9 (PO experiments.md s26), default off
        from cbb_sim.engine import foul_r9
        return foul_r9.load(arm)
    if arm not in ARMS:
        raise KeyError(f"unknown ENGINE_FOUL_JOINT={arm!r}; known: {sorted(ARMS)} or 'reference'")
    return FoulJoint(arm)
