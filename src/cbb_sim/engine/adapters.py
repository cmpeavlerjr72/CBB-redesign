"""
adapters.py -- one batched, vectorised adapter per sub-model.

Every adapter obeys the same three rules.

 1. ONE PREDICT PER STEP FOR THE WHOLE BATCH. No adapter is ever called per
    game or per possession. `predict(rows, state)` takes every active
    simulation at once, assembles one contiguous matrix by indexing the static
    blocks that `build_engine_inputs.py` precomputed, and makes exactly one
    batched call into the fitted object (one per shot class for fg_make, which
    is three disjoint row sets, and one per quantile level for the clock arm,
    which is what that arm IS).

 2. THE SUB-MODEL MODULE IS READ, NEVER MODIFIED OR REIMPLEMENTED. Feature
    orders come from each module's own `feature_set`, class orders from its own
    `CLASSES`, probabilities from its own fitted object's `predict_proba` /
    `pmf`. Where a module's predict helper wants a DataFrame the adapter builds
    one around the already-assembled matrix rather than reimplementing the
    maths.

 3. EVERY ADAPTER DECLARES WHETHER IT IS PROVISIONAL, and the flag is written
    into `run_meta.json` (deliverable 4). An adapter standing on a
    `reference_not_adopted` artifact is provisional by definition, and nothing
    downstream is allowed to forget it.

Adoption state as of 2026-09-10 (each model's own verdict file is the source):

    possession_outcome  NO WINNER (round 1, 0 of 11 arms passed; round 2 still
                        running). `ENGINE_EVENT=reference` wires
                        `reference_not_adopted_first.pkl` (lgbm) and
                        `_cont.pkl` (cascade), which are the same arm CLASSES
                        the round-2 spec names. -> provisional_event=True
    clock               NO ARM ADOPTED (0 of 20 eligible). `ENGINE_CLOCK=reference`
                        wires the best-CRPS arm, `reference_not_adopted_lgbm_quantile.pkl`.
                        -> provisional_clock=True
    rotation            NO ARM ADOPTED (state-dependence veto, both rounds).
                        `ENGINE_ROTATION=reference` wires R2 hierarchical
                        Dirichlet + the fitted scheduler from
                        `rotation_fit.json`. -> provisional_rotation=True
    usage               WINNER = the LightGBM choice arm, whose booster is NOT
                        persisted anywhere. The engine runs `usage.draw_player`'s
                        own U1 proportional path with the fitted (prior, m) per
                        class. -> provisional_usage=True
    fg_make             ADOPTED per class. `FGA_3` is the one wrinkle: Decision 8
                        selects LightGBM, `winner_FGA_3.joblib` predates the
                        decision and still holds the flat team_baseline.
                        `ENGINE_FG3=decision8` (default) uses the Decision-8
                        refit; `artifact` uses the stale joblib. Not provisional
                        either way, but which one ran is recorded.
    rebound             ADOPTED (lgbm/C_plus_state) -- refit by the engine
                        because the trainer persists nothing.
    free_throw          ADOPTED (lgbm) -- same, plus the rule table, which is
                        deterministic and needs no fit.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cbb_sim.engine.inputs import EngineInputs, FeaturePlan, plan_features
from cbb_sim.models import clock as CK
from cbb_sim.models import fg_make as FG
from cbb_sim.models import free_throw as FT
from cbb_sim.models import possession_outcome as PO
from cbb_sim.models import rebound as RB
from cbb_sim.models import rotation as ROT

ENGINE_DIR = Path("data/processed/models/engine")
PO_DIR = Path("data/processed/models/possession_outcome")
CK_DIR = Path("data/processed/models/clock")
FG_DIR = Path("data/processed/models/fg_make")
ROT_FIT = Path("data/processed/models/rotation/rotation_fit.json")

#: The state block: every feature that is a function of the live simulation.
#: One matrix per step, shared by every adapter, so a state quantity is
#: computed once and cannot disagree between two sub-models.
STATE_COLS: tuple[str, ...] = (
    "period", "seconds_remaining", "score_diff", "in_bonus", "is_ot",
    "x_score_diff__seconds_remaining", "chance_number",
    "is_transition", "is_transition_f", "chance_elapsed_s",
    "prev_end_DREB", "prev_end_TOV", "prev_end_made_FG", "prev_end_made_FT", "prev_end_other",
    "miss_rim", "miss_jump2", "miss_three", "blocked_f",
)
STATE_INDEX = {c: i for i, c in enumerate(STATE_COLS)}

#: fg_make / free_throw column aliases: the engine's static blocks carry the
#: class-suffixed or model-suffixed name, the model asks for the bare one.
FG_TEAM_ALIAS = {"off_make_c": "off_make_c__{k}", "def_allow_c": "def_allow_c__{k}"}
FG_SLOT_ALIAS = {
    "shooter_make_c": "shooter_make_c__{k}",
    "shooter_att_c": "shooter_att_c__{k}",
    "prior_season_make_c": "prior_season_make_c__{k}",
    "has_prior_season": "has_prior_season_fg",
}
FT_SLOT_ALIAS = {"has_prior_season": "has_prior_season_ft"}


def _alias(names: dict[str, int], alias: dict[str, str], key: str = "") -> dict[str, int]:
    out = dict(names)
    for bare, templ in alias.items():
        real = templ.format(k=key)
        if real in names:
            out[bare] = names[real]
    return out


def _assemble(plan: FeaturePlan, team: np.ndarray, slot: np.ndarray | None,
              state: np.ndarray) -> np.ndarray:
    """One contiguous float64 design matrix in the model's declared order."""
    m = np.empty((len(team), plan.width), dtype=np.float64)
    if len(plan.team_dst):
        m[:, plan.team_dst] = team[:, plan.team_src]
    if len(plan.slot_dst):
        if slot is None:
            raise ValueError("plan needs a per-slot block but none was gathered")
        m[:, plan.slot_dst] = slot[:, plan.slot_src]
    if len(plan.state_dst):
        m[:, plan.state_dst] = state[:, plan.state_src]
    return m


# ===========================================================================
# possession_outcome -- the terminal-event mix
# ===========================================================================
@dataclass
class EventAdapter:
    plan_first: FeaturePlan
    plan_cont: FeaturePlan
    arm_first: str
    arm_cont: str
    model_first: object
    model_cont: object
    provisional: bool
    source: dict

    @classmethod
    def load(cls, inp: EngineInputs, mode: str = "reference") -> EventAdapter:
        if mode != "reference":
            raise NotImplementedError(
                "possession_outcome round 2 has written no winner artifact; "
                "ENGINE_EVENT=reference is the only wired mode")
        src = {}
        models, arms, plans = {}, {}, {}
        for pop in ("first", "cont"):
            p = PO_DIR / f"reference_not_adopted_{pop}.pkl"
            d = pickle.loads(p.read_bytes())
            models[pop] = d["model"]
            arms[pop] = d["arm"]
            plans[pop] = plan_features(d["features"], inp.team_names, {}, STATE_INDEX)
            src[pop] = {"path": str(p), "arm": d["arm"], "feature_set": d["feature_set"],
                        "fold": d["fold"], "adopted": bool(d["adopted"])}
        return cls(plans["first"], plans["cont"], arms["first"], arms["cont"],
                   models["first"], models["cont"], True, src)

    def predict(self, team: np.ndarray, state: np.ndarray, is_first: np.ndarray) -> np.ndarray:
        """(n, 6) in `PO.CLASSES` order. One batched predict per population."""
        out = np.empty((len(team), len(PO.CLASSES)), dtype=np.float64)
        for mask, plan, model in ((is_first, self.plan_first, self.model_first),
                                  (~is_first, self.plan_cont, self.model_cont)):
            if not mask.any():
                continue
            m = _assemble(plan, team[mask], None, state[mask])
            out[mask] = model.predict_proba(np.ascontiguousarray(m, dtype=np.float32))
        return out


# ===========================================================================
# clock -- possession duration
# ===========================================================================
@dataclass
class ClockAdapter:
    plan: FeaturePlan
    arm: object
    provisional: bool
    source: dict

    @classmethod
    def load(cls, inp: EngineInputs, mode: str = "reference") -> ClockAdapter:
        if mode == "reference":
            p = CK_DIR / "reference_not_adopted_lgbm_quantile.pkl"
        elif mode == "reference_empirical":
            p = CK_DIR / "reference_not_adopted_empirical.pkl"
        else:
            raise NotImplementedError(
                "the clock bake-off adopted nothing and round 2 has written no winner; "
                "ENGINE_CLOCK must be 'reference' (best CRPS, lgbm_quantile) or "
                "'reference_empirical'")
        d = pickle.loads(p.read_bytes())
        arm = d["model"]
        for b in getattr(arm, "boosters", []):
            try:
                b.set_params(n_jobs=1)
            except Exception:                                   # noqa: BLE001
                pass
        plan = plan_features(arm.features, inp.team_names, {}, STATE_INDEX)
        return cls(plan, arm, True,
                   {"path": str(p), "arm": d["arm"], "feature_set": d["feature_set"],
                    "fold": d["fold"], "adopted": bool(d["adopted"])})

    def pmf(self, team: np.ndarray, state: np.ndarray) -> np.ndarray:
        m = _assemble(self.plan, team, None, state)
        df = pd.DataFrame(m, columns=list(self.plan.features), copy=False)
        return self.arm.pmf(df)

    def draw(self, team: np.ndarray, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Inverse-CDF duration draw through the module's own sampler."""
        return CK.sample_from_pmf(self.pmf(team, state), u)


# ===========================================================================
# fg_make -- make probability per shot class, keyed on the shooter
# ===========================================================================
@dataclass
class FgMakeAdapter:
    plans: dict[str, FeaturePlan]
    arms: dict[str, str]
    models: dict[str, object]
    feature_sets: dict[str, str]
    source: dict
    provisional: bool

    @classmethod
    def load(cls, inp: EngineInputs, fold: str = "F2", fg3: str = "decision8") -> FgMakeAdapter:
        plans, arms, models, fsets, src = {}, {}, {}, {}, {}
        for cls_name, key in (("FGA_rim", "rim"), ("FGA_jump2", "jump2"), ("FGA_3", "three")):
            path = FG_DIR / f"winner_{cls_name}.joblib"
            w = joblib.load(path)
            arm, model, feats, fs = w.arm, w.models[cls_name], w.features, w.feature_set
            note = "adopted winner artifact"
            if cls_name == "FGA_3" and fg3 == "decision8":
                alt = ENGINE_DIR / f"fg_make_FGA_3_decision8_{fold}.joblib"
                if not alt.exists():
                    raise FileNotFoundError(
                        f"{alt} missing; run scripts/build_engine_inputs.py, or pass "
                        "ENGINE_FG3=artifact to run the stale team_baseline winner")
                a = joblib.load(alt)
                arm, model, feats, fs = a["arm"], a["model"], a["features"], a["feature_set"]
                note = ("ARCHITECTURE_DECISIONS Decision 8 selects LightGBM for FGA_3; "
                        "the joblib winner predates it and holds the flat team_baseline")
                path = alt
            for m in (getattr(model, "clf_", None),):
                try:
                    m.set_params(n_jobs=1)
                except Exception:                               # noqa: BLE001
                    pass
            arms[cls_name] = arm
            models[cls_name] = model
            fsets[cls_name] = fs
            plans[cls_name] = plan_features(
                feats, _alias(inp.team_names, FG_TEAM_ALIAS, key),
                _alias(inp.slot_names, FG_SLOT_ALIAS, key), STATE_INDEX)
            src[cls_name] = {"path": str(path), "arm": arm, "feature_set": fs, "note": note}
        return cls(plans, arms, models, fsets, src, False)

    def predict(self, cls_name: str, team: np.ndarray, slot: np.ndarray,
                state: np.ndarray) -> np.ndarray:
        """P(make) for one shot class, one batched predict."""
        plan = self.plans[cls_name]
        m = _assemble(plan, team, slot, state)
        model = self.models[cls_name]
        p = model.predict_proba(np.ascontiguousarray(m, dtype=np.float32))
        return p[:, FG.CLASS_INDEX["MAKE"]]


# ===========================================================================
# free_throw -- the rule table plus a shooter-keyed make probability
# ===========================================================================
@dataclass
class FreeThrowAdapter:
    plan: FeaturePlan
    model: object
    source: dict
    provisional: bool
    bonus_thresholds: dict

    @classmethod
    def load(cls, inp: EngineInputs, fold: str = "F2") -> FreeThrowAdapter:
        p = ENGINE_DIR / f"free_throw_{fold}.joblib"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run scripts/build_engine_inputs.py")
        d = joblib.load(p)
        model = d["model"]
        try:
            model.clf_.set_params(n_jobs=1)
        except Exception:                                       # noqa: BLE001
            pass
        plan = plan_features(FT.FT_FEATURES, inp.team_names,
                             _alias(inp.slot_names, FT_SLOT_ALIAS), STATE_INDEX)
        return cls(plan, model,
                   {"path": str(p), "arm": d["arm"], "why": d["why"]}, False,
                   FT.load_bonus_era())

    def predict(self, team: np.ndarray, slot: np.ndarray, state: np.ndarray) -> np.ndarray:
        m = _assemble(self.plan, team, slot, state)
        p = self.model.predict_proba(np.ascontiguousarray(m, dtype=np.float32))
        return p[:, FT.CLASS_INDEX["MAKE"]]


# ===========================================================================
# rebound -- team-level OREB/DREB with dead balls as a fixed share (L17)
# ===========================================================================
@dataclass
class ReboundAdapter:
    plan: FeaturePlan
    model: object
    dead_share: dict[str, float]
    source: dict
    provisional: bool

    @classmethod
    def load(cls, inp: EngineInputs, fold: str = "F2") -> ReboundAdapter:
        p = ENGINE_DIR / f"rebound_{fold}.joblib"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run scripts/build_engine_inputs.py")
        d = joblib.load(p)
        model = d["model"]
        try:
            model.clf_.set_params(n_jobs=1)
        except Exception:                                       # noqa: BLE001
            pass
        plan = plan_features(d["features"], inp.team_names, {}, STATE_INDEX)
        return cls(plan, model, dict(inp.rules["dead_share"]),
                   {"path": str(p), "arm": d["arm"], "why": d["why"]}, False)

    def predict(self, team: np.ndarray, state: np.ndarray) -> np.ndarray:
        """(n, 3) in `RB.CLASSES` order = (OREB, DREB, DEAD)."""
        m = _assemble(self.plan, team, None, state)
        return self.model.predict_proba(np.ascontiguousarray(m, dtype=np.float32))


# ===========================================================================
# usage -- who of the five on the floor (usage.draw_player's own U1 path)
# ===========================================================================
@dataclass
class UsageAdapter:
    classes: tuple[str, ...]
    class_index: dict[str, int]
    source: dict
    provisional: bool

    @classmethod
    def load(cls, inp: EngineInputs) -> UsageAdapter:
        return cls(inp.usage_classes, {c: i for i, c in enumerate(inp.usage_classes)},
                   {"path": "data/processed/models/usage/usage_params_v1.json",
                    "arm": "U1 proportional (usage.draw_player's own rule)",
                    "why": "the adopted LightGBM choice arm persists no booster anywhere; "
                           "the engine runs draw_player's proportional rule over the five "
                           "with the fitted (prior, m) per class",
                    "priors": {k: v for k, v in inp.rules.items()
                               if k.startswith("usage_prior_")}},
                   True)

    def probs(self, rate_five: np.ndarray) -> np.ndarray:
        """`usage.normalise` over the five on the floor, verbatim: a non-finite
        or all-zero row falls back to the uniform, never to slot 0."""
        return ROT_NORMALISE(rate_five)


def ROT_NORMALISE(r: np.ndarray) -> np.ndarray:          # noqa: N802
    """`cbb_sim.models.usage.normalise`, called through so the engine cannot
    drift from it."""
    from cbb_sim.models.usage import normalise
    return normalise(r)


# ===========================================================================
# the bundle
# ===========================================================================
@dataclass
class Adapters:
    event: EventAdapter
    clock: ClockAdapter
    fg: FgMakeAdapter
    ft: FreeThrowAdapter
    reb: ReboundAdapter
    usage: UsageAdapter
    rot_fit: ROT.RotationFit
    rotation_mode: str
    flags: dict

    @classmethod
    def load(cls, inp: EngineInputs, fold: str = "F2") -> Adapters:
        ev_mode = os.environ.get("ENGINE_EVENT", "reference")
        ck_mode = os.environ.get("ENGINE_CLOCK", "reference")
        rot_mode = os.environ.get("ENGINE_ROTATION", "reference")
        fg3 = os.environ.get("ENGINE_FG3", "decision8")
        if rot_mode != "reference":
            raise NotImplementedError(
                "the rotation bake-off adopted nothing; ENGINE_ROTATION=reference "
                "(R2 hierarchical Dirichlet + the fitted scheduler) is the only wired mode")
        event = EventAdapter.load(inp, ev_mode)
        clock = ClockAdapter.load(inp, ck_mode)
        fg = FgMakeAdapter.load(inp, fold, fg3)
        ft = FreeThrowAdapter.load(inp, fold)
        reb = ReboundAdapter.load(inp, fold)
        usage = UsageAdapter.load(inp)
        rot_fit = ROT.RotationFit.from_json(ROT_FIT)
        flags = {
            "ENGINE_EVENT": ev_mode, "ENGINE_CLOCK": ck_mode,
            "ENGINE_ROTATION": rot_mode, "ENGINE_FG3": fg3,
            "provisional_event": event.provisional,
            "provisional_clock": clock.provisional,
            "provisional_rotation": True,
            "provisional_usage": usage.provisional,
            "provisional_fg": fg.provisional,
            "provisional_rebound": reb.provisional,
            "provisional_free_throw": ft.provisional,
            "provisional_foul_accrual": True,
            "provisional_and_one": True,
            "sources": {
                "event": event.source, "clock": clock.source, "fg_make": fg.source,
                "free_throw": ft.source, "rebound": reb.source, "usage": usage.source,
                "rotation": {"path": str(ROT_FIT), "arm": "R2_hier_dirichlet + scheduler",
                             "adopted": False,
                             "why": "no rotation arm passed the pre-registered "
                                    "state-dependence gate in either round"},
            },
        }
        return cls(event, clock, fg, ft, reb, usage, rot_fit, rot_mode, flags)

    def describe(self) -> str:
        return json.dumps(self.flags, indent=1, default=str)
