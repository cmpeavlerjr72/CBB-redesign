"""State-aware FT-trip allocation in the late leading window. DEFAULT OFF.

Lane I, 2026-10-01. Pre-registration and results: `docs/models/usage/experiments.md` section 13.

    ENGINE_USAGE_FT_LATE = off (default) | UL2 | UL1

Unset / `off`: `adapters.Adapters.load` never imports this module and the served `UsageAdapter` is untouched.
With an arm: for FT_trip draws only, when the offence is ahead before the event (`score_diff > 0`), the period is >= 2
and the usage `sec_remaining` is <= 120, the served proportional probabilities p_i are multiplied by exp(beta z_i) and
renormalised, with z_i the slot's shrunk, league-centred FT ability from the four engine FT slot columns:
    z = (fta_asof * shooter_ft_asof + 30 * prior_season_ft) / (fta_asof + 30)
(beta_60 for sec_remaining <= 60, beta_120 for 60-120). Every other row is returned exactly as the served adapter returns
it. Betas are the fold-2 fits of `scripts/train_usage_ft_late_v1.py` (train 2024).
"""
from __future__ import annotations

import numpy as np

M_FT = 30.0
ARMS = {"UL2": (2.8752241514373416, 1.0313029314498414),
        "UL1": (2.3037969854814118, 2.3037969854814118)}


class UsageFtLate:
    def __init__(self, base, inp, arm: str):
        self.base = base
        self.arm = arm
        self.b60, self.b120 = ARMS[arm]
        self.ft_cls = base.class_index["FT_trip"]
        sn = inp.slot_names
        self.cols = (sn["shooter_ft_asof"], sn["shooter_fta_asof"], sn["prior_season_ft"])

    def __getattr__(self, k):
        return getattr(self.base, k)

    def probs(self, rate_five, **kw):
        p = self.base.probs(rate_five, **kw)
        uc = kw.get("usage_class")
        if uc is None:
            return p
        w = ((np.asarray(uc) == self.ft_cls) & (np.asarray(kw["score_diff"]) > 0)
             & (np.asarray(kw["period"]) >= 2) & (np.asarray(kw["sec_remaining"]) <= 120))
        if not w.any():
            return p
        inp, g, s, five = kw["inp"], np.asarray(kw["gidx"])[w], np.asarray(kw["side"])[w], np.asarray(kw["five"])[w]
        blk = inp.slot_static[g[:, None], s[:, None], five]                 # (k, 5, Fs)
        own, fta, pri = (blk[..., c].astype(np.float64) for c in self.cols)
        z = (fta * own + M_FT * pri) / (fta + M_FT)
        beta = np.where(np.asarray(kw["sec_remaining"])[w] <= 60, self.b60, self.b120)[:, None]
        q = p[w] * np.exp(beta * z)
        p = p.copy()
        p[w] = q / q.sum(axis=1, keepdims=True)
        return p


def wrap(inp, usage, arm: str):
    if arm not in ARMS:
        raise ValueError(f"ENGINE_USAGE_FT_LATE={arm!r}: expected off or one of {sorted(ARMS)}")
    return UsageFtLate(usage, inp, arm), {"arm": arm, "betas": ARMS[arm], "module": "cbb_sim.engine.usage_ft_late",
                                          "doc": "docs/models/usage/experiments.md section 13"}
