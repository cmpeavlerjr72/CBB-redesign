"""late_game_adapter.py -- late-game ROUND 2 window arms, DEFAULT-OFF.

Pre-registration: `docs/models/late_game/experiments.md` section 4.

`ENGINE_LATE_GAME` unset or `off` -> this module is never imported and the
served path is untouched. Otherwise the value is a `+`-joined list of

    clk_C2   window clock law = round 1's `C2_clk` (P3R cells, no 45 s floor)
    clk_D    window clock law = round 1's `D_clk` (role3 x bucket x bonus x prev_end)
    ev_BL3   window FIRST-chance event model = round 1's `B / L3_gates` lgbm (S0)
    ev_L0S0  window FIRST-chance event model = round 1's `A / L0_reference` lgbm (S0),
             the cadence CONTROL for ev_BL3

THE WINDOW: `period == 2 and seconds_remaining <= 120 and |score_diff| <= 6`
(offence perspective, live state). Overtime is OUTSIDE it (section 4.1).

BIT-IDENTITY OUTSIDE THE WINDOW. Both wrappers first call the SERVED adapter on
every row, with the same arguments, and overwrite only the window rows. The
served adapter therefore sees exactly the calls it sees in a default run (its
accumulators included), and every non-window row returns the served value.
Since RNG streams are counter-based per (seed, game_id, family), everything a
simulation does before its first window possession is bit-identical to the
reference -- which is what the first-half veto checks.

THE CLOCK LATENT. The served clock (`v5b_glat_pmean`) scales the cell draw by
one per-game latent `A`. The window draw uses the same duration uniform and is
scaled by the same `A` (the served adapter's own `_latent`, same keys, same
ordinal), then rounded and clipped to the duration grid exactly as the served
draw is. One pace realisation per game is preserved.

No multiplier, cap or blend is applied to any output: the window rows are drawn
from a different fitted law, which is the arm.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cbb_sim.models import clock as CK
from cbb_sim.models import late_game as LGM  # registers the P3R/LGD cell dims

R2_DIR = Path("data/processed/models/late_game/round2")
CLOCK_ARMS = ("clk_C2", "clk_D")
EVENT_ARMS = ("ev_BL3", "ev_L0S0")


def parse_mode(mode: str) -> tuple[str | None, str | None]:
    parts = [p.strip() for p in mode.split("+") if p.strip()]
    clk = [p for p in parts if p in CLOCK_ARMS]
    ev = [p for p in parts if p in EVENT_ARMS]
    bad = [p for p in parts if p not in CLOCK_ARMS + EVENT_ARMS]
    if bad or len(clk) > 1 or len(ev) > 1 or not parts:
        raise ValueError(
            f"ENGINE_LATE_GAME={mode!r}: expected '+'-joined members of "
            f"{CLOCK_ARMS + EVENT_ARMS}, at most one clock and one event arm")
    return (clk[0] if clk else None), (ev[0] if ev else None)


def _load(name: str):
    p = R2_DIR / f"{name}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run scripts/train_late_game_r2_v1.py")
    with open(p, "rb") as f:
        return pickle.load(f), str(p)


@dataclass
class LateGameClock:
    """The served clock outside the window; a round-1 window law inside it."""

    inner: object                    # the served clock adapter (v5b LatentClockAdapter)
    arm: object                      # the fitted window cell law
    name: str
    source: dict
    provisional: bool = True
    wants_game_index: bool = True
    wants_sim_keys: bool = True
    n_window: int = 0
    _state_idx: dict = field(default_factory=dict)

    def __getattr__(self, k):
        return getattr(self.__dict__["inner"], k)

    def pmf(self, team, state, gidx=None):
        return self.inner.pmf(team, state, gidx)

    def draw(self, team, state, u, gidx=None, keys=None):
        dur = np.asarray(self.inner.draw(team, state, u, gidx, keys), dtype=np.float64).copy()
        idx = self.inner.inner.state_idx
        w = LGM.in_window(state[:, idx["period"]], state[:, idx["seconds_remaining"]],
                          state[:, idx["score_diff"]])
        if not w.any():
            return dur
        r = np.flatnonzero(w)
        base = self.inner.inner                      # ClockAdapterV3: the frame builder
        df = base._frame(team[r], state[r], None if gidx is None else np.asarray(gidx)[r])
        t = CK.sample_from_pmf(self.arm.pmf(df), np.asarray(u)[r]).astype(np.float64)
        a = self.inner._latent(np.asarray(keys)[r], team[r])
        dur[r] = np.clip(np.rint(a * t), 0.0, float(CK.DURATION_CAP))
        self.n_window += len(r)
        return dur


@dataclass
class LateGameEvent:
    """The served event model everywhere; a round-1 lgbm on window FIRST chances."""

    inner: object                    # the served EventAdapter (round2_s1)
    model: object
    features: list
    name: str
    source: dict
    provisional: bool = True
    plan_l0: object = None
    j_home: int = -1
    j_away: int = -1

    def __getattr__(self, k):
        return getattr(self.__dict__["inner"], k)

    def predict(self, team, state, is_first, gidx=None, off=None):
        out = self.inner.predict(team, state, is_first, gidx, off)
        from cbb_sim.engine.adapters import STATE_INDEX as I
        from cbb_sim.engine.adapters import _assemble
        w = np.asarray(is_first, dtype=bool) & LGM.in_window(
            state[:, I["period"]], state[:, I["seconds_remaining"]], state[:, I["score_diff"]])
        if not w.any():
            return out
        r = np.flatnonzero(w)
        team_r2 = self.inner.team_block[np.asarray(gidx)[r], np.asarray(off)[r]]
        m0 = _assemble(self.plan_l0, team_r2, None, state[r])
        if len(self.features) > m0.shape[1]:
            ext = LGM.l3_extras(team_r2[:, self.j_home], team_r2[:, self.j_away],
                                state[r, I["score_diff"]], state[r, I["seconds_remaining"]],
                                state[r, I["in_double_bonus"]])
            m0 = np.hstack([m0, ext])
        out = out.copy()
        out[r] = self.model.predict_proba(np.ascontiguousarray(m0, dtype=np.float32))
        return out


def wrap(inp, event, clock, mode: str) -> tuple[object, object, dict]:
    """Return (event, clock, provenance) with the named window arms applied."""
    clk_name, ev_name = parse_mode(mode)
    src: dict = {"mode": mode, "window": "period==2 & seconds_remaining<=120 & |score_diff|<=6",
                 "overtime": "outside the window (served law)", "adopted": False,
                 "preregistration": "docs/models/late_game/experiments.md section 4"}
    if clk_name is not None:
        if not hasattr(clock, "_latent") or not hasattr(getattr(clock, "inner", None), "_frame"):
            raise ValueError("ENGINE_LATE_GAME clock arms wrap the served v5b latent clock only")
        arm, p = _load(clk_name)
        s = {"arm": clk_name, "path": p, "scheme": "S0", "fold": "F2",
             "train_seasons": [2022, 2023, 2024]}
        clock = LateGameClock(inner=clock, arm=arm, name=clk_name, source={**clock.source,
                              "late_game": s})
        src["clock"] = s
    if ev_name is not None:
        if getattr(event, "team_block", None) is None:
            raise ValueError("ENGINE_LATE_GAME event arms wrap the served round2_s1 event model only")
        obj, p = _load(ev_name)
        from cbb_sim.engine.adapters import STATE_INDEX
        from cbb_sim.engine.inputs import plan_features
        from cbb_sim.models import possession_outcome as PO
        feats = list(obj["features"])
        l0 = PO.feature_set("C_plus_state", "first")
        if feats[:len(l0)] != l0:
            raise ValueError(f"{ev_name}: bundle does not start with the served C_plus_state list")
        extra = feats[len(l0):]
        if extra and tuple(extra) != LGM.L3_EXTRAS:
            raise ValueError(f"{ev_name}: extras {extra} are not the L3_gates columns")
        import json
        from cbb_sim.engine.adapters import ENGINE_DIR
        idx = json.loads((ENGINE_DIR / "event_round2_s1_F2_2025" / "index.json")
                         .read_text(encoding="utf-8"))
        names = {c: i for i, c in enumerate(idx["team_cols"])}
        plan = plan_features(l0, names, {}, STATE_INDEX)
        try:
            obj["model"].clf_.set_params(n_jobs=1)
        except Exception:                                            # noqa: BLE001
            pass
        s = {"arm": ev_name, "path": p, "bundle": obj["bundle"], "scheme": "S0",
             "fold": "F2", "population": "first", "train_seasons": [2022, 2023, 2024]}
        event = LateGameEvent(inner=event, model=obj["model"], features=feats, name=ev_name,
                              source={**event.source, "late_game": s}, plan_l0=plan,
                              j_home=names["site_home"], j_away=names["site_away"])
        src["event"] = s
    return event, clock, src


def mode_from_env() -> str:
    return os.environ.get("ENGINE_LATE_GAME", "off")
