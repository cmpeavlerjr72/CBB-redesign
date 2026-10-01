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
CLOCK_ARMS = ("clk_C2", "clk_D", "clk_Dt", "clk_Dtt", "clk_DtL", "clk_DtLL")
EVENT_ARMS = ("ev_BL3", "ev_L0S0")
# Round 3 (experiments.md sections 6.3 / 7.3): the SAME clk_D law, gated by the offence's
# live score_diff sign at possession start. None = every window row (round 2's clk_D).
#   clk_Dt  -> tied rows only; clk_Dtt -> tied and trailing rows. No artifact is refit.
# Round 4 (experiments.md section 9.3): clk_DtL = clk_D on tied rows + the LGL law (role refined to
# five bands) on LEADING rows; trailing rows keep the served law.
ROLE_GATE = {"clk_Dt": (0,), "clk_Dtt": (0, -1), "clk_DtL": (0,), "clk_DtLL": (0,)}
ARTIFACT_OF = {"clk_Dt": "clk_D", "clk_Dtt": "clk_D", "clk_DtL": "clk_D", "clk_DtLL": "clk_D"}
R4_DIR = Path("data/processed/models/late_game/round4")
LEAD_ARTIFACT = {"clk_DtL": R4_DIR / "clk_LGL.pkl", "clk_DtLL": R4_DIR / "clk_LGL.pkl"}
# Round 5 (experiments.md section 11): clk_DtLL serves LGL on trailing AND leading window rows.
LEAD_ROLES = {"clk_DtL": (1,), "clk_DtLL": (1, -1)}


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
    roles: tuple | None = None       # round 3: allowed sign(score_diff) values; None = all
    arm_lead: object = None          # round 4: law for LEADING window rows (clk_DtL); None = none
    lead_roles: tuple = (1,)         # round 5: sign(score_diff) values arm_lead serves

    def __getattr__(self, k):
        return getattr(self.__dict__["inner"], k)

    def pmf(self, team, state, gidx=None):
        return self.inner.pmf(team, state, gidx)

    def draw(self, team, state, u, gidx=None, keys=None):
        dur = np.asarray(self.inner.draw(team, state, u, gidx, keys), dtype=np.float64).copy()
        idx = self.inner.inner.state_idx
        win = LGM.in_window(state[:, idx["period"]], state[:, idx["seconds_remaining"]],
                            state[:, idx["score_diff"]])
        w = win.copy()
        if self.roles is not None:
            w &= np.isin(np.sign(state[:, idx["score_diff"]]), self.roles)
        self._redraw(self.arm, w, team, state, u, gidx, keys, dur)
        if self.arm_lead is not None:
            self._redraw(self.arm_lead, win & np.isin(np.sign(state[:, idx["score_diff"]]), self.lead_roles),
                         team, state, u, gidx, keys, dur)
        return dur

    def _redraw(self, arm, w, team, state, u, gidx, keys, dur):
        if not w.any():
            return
        r = np.flatnonzero(w)
        base = self.inner.inner                     # ClockAdapterV3: the frame builder
        df = base._frame(team[r], state[r], None if gidx is None else np.asarray(gidx)[r])
        t = CK.sample_from_pmf(arm.pmf(df), np.asarray(u)[r]).astype(np.float64)
        a = self.inner._latent(np.asarray(keys)[r], team[r])
        dur[r] = np.clip(np.rint(a * t), 0.0, float(CK.DURATION_CAP))
        self.n_window += len(r)


# ---------------------------------------------------------------------------
# Round 4 (experiments.md section 9): the end-of-period make law and the no-shot law.
# Both are fitted cell laws (data, not sim output), DEFAULT OFF, periods 1-2 only.
# ---------------------------------------------------------------------------
MAKE_ARMS = ("MK1", "MK2")
BUZZER_ARMS = ("BZ1", "BZ2", "BZ3")
_CLS = {"FGA_rim": 0, "FGA_jump2": 1, "FGA_3": 2}


@dataclass
class LateGameMake:
    """Served fg_make everywhere; on FIRST-chance shots with <= 1 s left AT THE SHOT
    (possession-start clock minus the fed chance_elapsed_s, rounded) in periods 1-2, the buzzer
    make law (`train_late_game_r4_make_v1.py`, fold 2 fit) replaces the probability."""

    inner: object
    lut: dict
    arm: str
    source: dict
    n_replaced: int = 0

    def __getattr__(self, k):
        return getattr(self.__dict__["inner"], k)

    def predict(self, shot_class, team, slot, xs, gidx=None, *a, **k):
        p = np.asarray(self.inner.predict(shot_class, team, slot, xs, gidx, *a, **k), dtype=np.float64)
        from cbb_sim.engine.adapters import STATE_INDEX as I
        per = xs[:, I["period"]]
        tl = np.rint(xs[:, I["seconds_remaining"]] - xs[:, I["chance_elapsed_s"]])
        w = (per <= 2) & (xs[:, I["chance_number"]] == 1) & (tl <= 1)
        if not w.any() or shot_class not in _CLS:
            return p
        p = p.copy()
        key = _CLS[shot_class] * 2 + (per[w].astype(np.int64) - 1)
        if self.arm == "MK2":
            key = key * 2 + np.clip(tl[w], 0, 1).astype(np.int64)
        p[w] = np.asarray(self.lut[self.arm])[key]
        self.n_replaced += int(w.sum())
        return p


def wrap_make(fg, mode: str):
    if mode not in MAKE_ARMS:
        raise ValueError(f"ENGINE_LG_MAKE={mode!r}: expected one of {MAKE_ARMS}")
    import json
    p = R4_DIR / "make_F2.json"
    lut = json.loads(p.read_text(encoding="utf-8"))
    src = {"arm": mode, "path": str(p), "fold": "F2", "train_seasons": [2022, 2023, 2024],
           "scope": "first-chance FGA, periods 1-2, rint(sec - chance_elapsed_s) <= 1",
           "preregistration": "docs/models/late_game/experiments.md section 9", "adopted": False}
    return LateGameMake(inner=fg, lut=lut, arm=mode, source={**getattr(fg, "source", {}), "late_game_make": src}), src


class Buzzer:
    """P(possession ends at the horn with no terminal event | start state), periods 1-2, start <= 35 s
    (`train_late_game_r4_buzzer_v1.py`, fold 2 fit). Own stream (seed, game_id, "lg_buzzer")."""

    ST = {"made_FG": 0, "made_FT": 0, "DREB": 1, "TOV": 2}
    SIZES = {"per": 2, "sb": 7, "role": 3, "st": 4}

    def __init__(self, lut: dict, arm: str, seeds, gids):
        from cbb_sim.engine import rng as RNG
        from cbb_sim.engine import state as S
        self.rate = np.asarray(lut["rate"])
        self.dims = lut["dims"]
        self.edges = np.asarray(lut["edges"])
        self.arm = arm
        self.book = RNG.StreamBook(seeds, gids, families=("lg_buzzer",))
        code = S.PREV_END_CODE
        self.st_of = np.full(max(code.values()) + 1, 3, dtype=np.int64)
        for name, v in self.ST.items():
            if name in code:
                self.st_of[code[name]] = v
        self.n_noshot = 0

    def draw(self, act, period, sec, off_sd, prev) -> np.ndarray:
        ns = np.zeros(len(act), dtype=bool)
        g = (period <= 2) & (sec <= 35)
        if not g.any():
            return ns
        r = np.flatnonzero(g)
        c = {"per": period[r].astype(np.int64) - 1,
             "sb": np.clip(np.searchsorted(self.edges, sec[r], side="left"), 0, len(self.edges) - 1),
             "role": (np.sign(off_sd[r]).astype(np.int64) + 1),
             "st": self.st_of[np.asarray(prev[r], dtype=np.int64)]}
        k = np.zeros(len(r), dtype=np.int64)
        for dm in self.dims:
            k = k * self.SIZES[dm] + c[dm]
        u = self.book.draw("lg_buzzer", act[r])
        ns[r] = u < self.rate[k]
        self.n_noshot += int(ns.sum())
        return ns


def load_buzzer(seeds, gids):
    """ENGINE_LG_BUZZER unset/off -> None (nothing imported or drawn)."""
    mode = os.environ.get("ENGINE_LG_BUZZER", "off") or "off"
    if mode == "off":
        return None
    if mode not in BUZZER_ARMS:
        raise ValueError(f"ENGINE_LG_BUZZER={mode!r}: expected one of {BUZZER_ARMS}")
    import json
    lut = json.loads((R4_DIR / f"buzzer_{mode}_F2.json").read_text(encoding="utf-8"))
    return Buzzer(lut, mode, seeds, gids)


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
        arm, p = _load(ARTIFACT_OF.get(clk_name, clk_name))
        s = {"arm": clk_name, "path": p, "scheme": "S0", "fold": "F2",
             "train_seasons": [2022, 2023, 2024]}
        if clk_name in ROLE_GATE:
            s["role_gate_sign_score_diff"] = list(ROLE_GATE[clk_name])
            s["preregistration"] = "docs/models/late_game/experiments.md sections 6-7"
        arm_lead = None
        if clk_name in LEAD_ARTIFACT:
            with open(LEAD_ARTIFACT[clk_name], "rb") as f:
                arm_lead = pickle.load(f)
            s["leading_rows_law"] = str(LEAD_ARTIFACT[clk_name])
            s["preregistration"] = "docs/models/late_game/experiments.md section 9"
        clock = LateGameClock(inner=clock, arm=arm, name=clk_name, source={**clock.source,
                              "late_game": s}, roles=ROLE_GATE.get(clk_name), arm_lead=arm_lead,
                              lead_roles=LEAD_ROLES.get(clk_name, (1,)))
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
