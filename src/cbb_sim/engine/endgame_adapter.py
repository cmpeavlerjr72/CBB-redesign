"""endgame_adapter.py -- late-game round 7 window arms (late_game/experiments.md section 15), DEFAULT-OFF.

`ENGINE_ENDGAME` unset or `off` -> this module is never imported and the served path is untouched (parity v10).
Otherwise a `+`-joined list of at most one clock arm and one event arm:

    GT1   block C cell law (role-band x time bucket KM, m = 50 hierarchy to the served law)
    GT2   block C smooth margin response (AFT scale on the served law, 2 parameters per half)
    EV1   block E class-probability ratio law on window FIRST chances (shrunk to the served law, m = 50)
    CA1   clock s46 block A RATIO cell law on the served K2 window law (`endgame_v2.CA1Law`, m = 100 chosen on
          fold 1). Added 2026-10-14 for the LABELLED DIAGNOSTIC loop on CA1 (PM ruling 10-14; block A was
          REFUTED offline, s47): never a selection read, never adopted, default off.

Block F is `ENGINE_LATE_FOUL=eg_LF1|eg_LF2` (`late_foul.py`), the section-29 count law refitted by this round.

SCOPE IN THE ENGINE: period 2 only, possession start <= 240 s (s15.A.2 selects on period 2; the PM's Q3 ruling keeps
the period-1 window REPORTED only), so the first half is bit-identical to the served stack. Overtime keeps the served
law.

SAME UNIFORMS, SAME LATENT. The served adapter is called first on every row with the same arguments (its
accumulators see exactly the calls of a default run). Window rows are then re-drawn from the arm's law at the SAME
duration uniform (clock: the marginal pre-cascade draw AND K2's end-class-conditional re-draw; the continuation draws
are untouched) and scaled by the same per-game latent A, rounded and clipped exactly as the served draw; the event
arm rescales the served class probabilities before the served foul-joint trip adjustment. No RNG stream is added.
No multiplier, cap, clip or blend on sim output: the window rows are drawn from a different fitted law (the arm).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cbb_sim.models import clock as CK
from cbb_sim.models import endgame as EG

ART = Path("data/processed/models/late_game/endgame_v1")
CLOCK_ARMS = ("GT1", "GT2", "CA1")
ART_V2 = Path("data/processed/models/late_game/endgame_clock_v2")
EVENT_ARMS = ("EV1",)
PERIODS = (2,)


def parse_mode(mode: str) -> tuple[str | None, str | None]:
    parts = [p.strip() for p in mode.split("+") if p.strip()]
    clk = [p for p in parts if p in CLOCK_ARMS]
    ev = [p for p in parts if p in EVENT_ARMS]
    bad = [p for p in parts if p not in CLOCK_ARMS + EVENT_ARMS]
    if bad or len(clk) > 1 or len(ev) > 1 or not parts:
        raise ValueError(f"ENGINE_ENDGAME={mode!r}: expected '+'-joined members of {CLOCK_ARMS + EVENT_ARMS}, "
                         "at most one clock and one event arm")
    return (clk[0] if clk else None), (ev[0] if ev else None)


def load_law(name: str, fold: str):
    d = ART / fold
    if name == "CA1":
        from cbb_sim.models import endgame_v2 as E2
        q = ART_V2 / fold / "ca1_law.npz"
        return E2.CA1Law.load(q), str(q)
    if name == "GT1":
        return EG.GT1Law.load(d / "gt1_law.npz"), str(d / "gt1_law.npz")
    if name == "GT2":
        return EG.GT2Law.load(d / "gt2_law.json"), str(d / "gt2_law.json")
    return EG.EV1Law.load(d / "ev1_law.npz"), str(d / "ev1_law.npz")


@dataclass
class EndgameClock:
    inner: object          # the served LatentClockAdapter (K2)
    law: object
    name: str
    source: dict
    provisional: bool = True
    n_window: int = 0
    n_window_end: int = 0

    def __getattr__(self, k):
        return getattr(self.__dict__["inner"], k)

    @property
    def cont_mode(self):
        return self.inner.cont_mode

    def _win(self, state) -> np.ndarray:
        idx = self.inner.inner.state_idx
        return EG.in_window(state[:, idx["period"]], state[:, idx["seconds_remaining"]], PERIODS)

    def _cols(self, state, r):
        idx = self.inner.inner.state_idx
        return (state[r, idx["period"]], state[r, idx["seconds_remaining"]], state[r, idx["score_diff"]])

    def _scale(self, t, keys, team):
        return np.clip(np.rint(self.inner._latent(np.asarray(keys), team) * t), 0.0, float(CK.DURATION_CAP))

    def pmf(self, team, state, gidx=None):
        return self.inner.pmf(team, state, gidx)

    def draw(self, team, state, u, gidx=None, keys=None):
        dur = np.asarray(self.inner.draw(team, state, u, gidx, keys), dtype=np.float64).copy()
        w = self._win(state)
        if not w.any():
            return dur
        r = np.flatnonzero(w)
        base = self.inner.inner
        g = None if gidx is None else np.asarray(gidx)[r]
        p0 = base._pmf_from_frame(base._frame(team[r], state[r], g), g)
        p = self.law.apply(p0, *self._cols(state, r))
        t = CK.sample_from_pmf(p, np.asarray(u)[r]).astype(np.float64)
        dur[r] = self._scale(t, np.asarray(keys)[r], team[r])
        self.n_window += len(r)
        return dur

    def redraw_end(self, team, state, u, gidx, keys, end1):
        out = np.asarray(self.inner.redraw_end(team, state, u, gidx, keys, end1), dtype=np.float64).copy()
        w = self._win(state)
        if not w.any():
            return out
        r = np.flatnonzero(w)
        base = self.inner.inner
        g = None if gidx is None else np.asarray(gidx)[r]
        df = base._frame(team[r], state[r], g)
        df["end1_code"] = np.asarray(end1, dtype=np.int64)[r]
        segs = base._route(g, len(df))
        p0 = np.empty((len(df), CK.DURATION_CAP + 1), dtype=np.float64)
        for k in np.unique(segs):
            rr = np.flatnonzero(segs == k)
            p0[rr] = base.arms[int(k)].pmf_end(df.iloc[rr].reset_index(drop=True))
        p = self.law.apply_end(p0, *self._cols(state, r), np.asarray(end1)[r])
        t = CK.sample_from_pmf(p, np.asarray(u)[r]).astype(np.float64)
        out[r] = self._scale(t, np.asarray(keys)[r], team[r])
        self.n_window_end += len(r)
        return out

    def draw_cont(self, team, gidx, grp, u, keys):
        return self.inner.draw_cont(team, gidx, grp, u, keys)


@dataclass
class EndgameEvent:
    inner: object
    law: object
    name: str
    source: dict
    provisional: bool = True
    n_window: int = 0

    def __getattr__(self, k):
        return getattr(self.__dict__["inner"], k)

    def predict(self, team, state, is_first, gidx=None, off=None, kidx=None):
        out = (self.inner.predict(team, state, is_first, gidx, off) if kidx is None
               else self.inner.predict(team, state, is_first, gidx, off, kidx=kidx))
        from cbb_sim.engine.adapters import STATE_INDEX as I
        w = np.asarray(is_first, dtype=bool) & EG.in_window(state[:, I["period"]], state[:, I["seconds_remaining"]],
                                                            PERIODS)
        if not w.any():
            return out
        r = np.flatnonzero(w)
        out = np.array(out, dtype=np.float64, copy=True)
        out[r] = self.law.apply(out[r], state[r, I["period"]], state[r, I["seconds_remaining"]],
                                state[r, I["score_diff"]])
        self.n_window += len(r)
        return out


def wrap(event, clock, mode: str, fold: str) -> tuple[object, object, dict]:
    clk_name, ev_name = parse_mode(mode)
    src: dict = {"mode": mode, "fold": fold, "adopted": False, "periods": list(PERIODS),
                 "window": "period 2, possession start <= 240 s (s15.A.2; period 1 reported offline only, Q3)",
                 "preregistration": "docs/models/late_game/experiments.md section 15"}
    if clk_name is not None:
        if not (hasattr(clock, "_latent") and hasattr(getattr(clock, "inner", None), "_frame")
                and getattr(clock, "cont_mode", None) == "K2"):
            raise ValueError("ENGINE_ENDGAME clock arms wrap the served K2 latent clock only")
        law, p = load_law(clk_name, fold)
        s = {"arm": clk_name, "path": p}
        clock = EndgameClock(inner=clock, law=law, name=clk_name, source={**clock.source, "endgame": s})
        src["clock"] = s
    if ev_name is not None:
        law, p = load_law(ev_name, fold)
        s = {"arm": ev_name, "path": p, "population": "first"}
        event = EndgameEvent(inner=event, law=law, name=ev_name, source={**event.source, "endgame": s})
        src["event"] = s
    return event, clock, src


def mode_from_env() -> str:
    return os.environ.get("ENGINE_ENDGAME", "off") or "off"
