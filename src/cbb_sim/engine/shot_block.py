"""Drawn block flag for the rebound model (shot_block round 2 / section 5).

Pre-registration: `docs/models/shot_block/experiments.md` section 5.

DEFAULT OFF. `ENGINE_SHOT_BLOCK` unset (or "reference") makes `load()` return
None; `loop.py` then feeds `blocked_f = 0.0` exactly as before and never draws
from the `"shot_block"` RNG family, so the served path is bit-identical.

With an arm named, every missed FIELD GOAL (free-throw misses stay 0) gets
P(blocked | missed) from the arm's linear logit over the 17 `Kc` inputs of
`scripts/train_shot_block_v1.py`, assembled from precomputed lookups
(`scripts/build_engine_shot_block_lut_v1.py`: team and shooter as-of block
rates, the per-shot-type anchor) plus the live state, and a 0/1 flag is DRAWN
(`u < p`, one `"shot_block"` uniform per missed FGA). The flag must be drawn:
the rebound LightGBM splits `blocked_f` at exactly 0.0, so a probability fed
continuously reads as "blocked" (rebound round 3, section 2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LUT_DIR = Path("data/processed/models/engine")
ARMS = {"K2": "shot_block_K2", "K2_Ocell": "shot_block_K2_Ocell",
        # DIAGNOSTIC (section 7): K2_Ocell with both team block rates at the league level (0.0)
        "K2_Ocell_noteam": "shot_block_K2_Ocell_noteam",
        # DEFAULT OFF (lane F, 2026-10-01; docs/tests/live_vs_backtest_slot_skew_2026-10-01.md): K2_Ocell with the table built
        # against the v3 inputs' roster slots (the table the live chain already serves). Same model and anchors; differs only in
        # the shooter / known arrays of the 334 games whose slots the v2-built table left anonymous.
        "K2_Ocell_v3in": "shot_block_K2_Ocell_v3in"}
TYPE_INDEX = {"rim": 0, "jump2": 1, "three": 2}
#: SERVED DEFAULT, ADOPTED 2026-10-01 (Decision 11 set, PM under user delegation;
#: docs/tests/adoption_served_v2_2026-10-01.md): `loop.py` passes this when
#: `ENGINE_SHOT_BLOCK` is unset. `reference` reproduces served-v1 (no draw).
DEFAULT = "K2_Ocell"

_CACHE: dict = {}


class ShotBlock:
    def __init__(self, arm: str, inp):
        tag = str(inp.meta.get("inputs_tag_loaded", ""))
        slate = tag[:-3] if tag.endswith("_v2") else tag
        # Live slates (2026-10-01): `scripts/build_shot_block_lut_live_v1.attach` builds the
        # per-slate table and names it in inp.meta; the backtest path below is unchanged.
        live = (inp.meta.get("shot_block_lut") or {}).get(arm)
        path = Path(live) if live else LUT_DIR / f"{ARMS[arm]}_{slate}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path}: no shot_block table for this slate (tag {tag!r}). Live inputs need "
                "scripts/build_shot_block_lut_live_v1.attach(inp, ...) before the sim. "
                "ENGINE_SHOT_BLOCK=reference would serve an ungated mix; do not use it to get past this.")
        z = np.load(path)
        if not np.array_equal(z["game_id"], inp.games["game_id"].to_numpy()):
            raise ValueError(f"{path}: game order does not match the engine slate {tag}")
        self.arm = arm
        self.team = z["team"].astype(np.float64)          # (G, 2, [def_block_c, off_blocked_c])
        self.shooter = z["shooter"].astype(np.float64)    # (G, 2, S)
        self.known = z["known"].astype(np.float64)        # (G, 2, S)
        self.anchor = z["anchor"].astype(np.float64)      # (G, 3)
        self.coef = z["coef"].astype(np.float64)
        self.mu = z["mu"].astype(np.float64)
        self.sd = z["sd"].astype(np.float64)
        self.features = [str(f) for f in z["features"]]
        self._t = {k: inp.team_names[k] for k in
                   ("site_home", "site_away", "off_rating_off_c", "off_rating_def_c",
                    "def_rating_off_c", "def_rating_def_c")}

    def prob(self, team_static, gidx, off, slot, type_idx, period, sec, score_diff, in_bonus):
        """P(blocked | missed FGA) for a batch; every argument is per row."""
        n = len(gidx)
        dfn = 1 - off
        cols = {
            "miss_rim": (type_idx == 0).astype(np.float64),
            "miss_jump2": (type_idx == 1).astype(np.float64),
            "miss_three": (type_idx == 2).astype(np.float64),
            "def_block_c": self.team[gidx, dfn, 0],
            "off_blocked_c": self.team[gidx, off, 1],
            "shooter_blocked_c": self.shooter[gidx, off, slot],
            "shooter_known": self.known[gidx, off, slot],
            "period": period, "seconds_remaining": sec, "score_diff": score_diff,
            "in_bonus": in_bonus,
        }
        ts = team_static[gidx, off]
        for k, j in self._t.items():
            cols[k] = ts[:, j].astype(np.float64)
        X = np.empty((n, len(self.features)), dtype=np.float64)
        for j, f in enumerate(self.features):
            X[:, j] = cols[f]
        eta = ((X - self.mu) / self.sd) @ self.coef[1:] + self.coef[0]
        eta = eta + self.anchor[gidx, type_idx]
        return 1.0 / (1.0 + np.exp(-eta))


def load(arm: str | None, inp):
    """None for the served path (unset / 'reference'), else the arm's lookups."""
    if not arm or arm == "reference":
        return None
    if arm not in ARMS:
        raise KeyError(f"unknown ENGINE_SHOT_BLOCK={arm!r}; known: {sorted(ARMS)} or 'reference'")
    key = (arm, id(inp))
    if key not in _CACHE:
        _CACHE[key] = ShotBlock(arm, inp)
    return _CACHE[key]
