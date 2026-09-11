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

    possession_outcome  ROUND 2 ADOPTED (2026-09-10): `first` -> lgbm + S1,
                        `cont` -> cascade + S1, both C_plus_state
                        (`round2/verdict.json`).
                        `ENGINE_EVENT=round2_s1` wires them and clears
                        provisional_event; because the round-2 trainer
                        persists no booster, the engine refits the winners'
                        own specs into its own directory
                        (`scripts/build_engine_event_round2.py`), the same
                        rule already applied to rebound and free_throw.
                        `ENGINE_EVENT=reference` (round 1's best-loss but
                        NOT-ADOPTED arms) stays wired and still sets
                        provisional_event=True.
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
    fg_make             ADOPTED per class. `FGA_3` was a wrinkle and is no
                        longer one: `winner_FGA_3.joblib` was re-exported at
                        15:52 on 2026-09-10, AFTER Decision 8 (15:47), and now
                        holds the lgbm / C_plus_state model. Verified
                        bit-identical to the engine's own Decision-8 refit
                        (max |diff| 0.0 over 4,000 probes, identical 25-feature
                        order, both 400 trees), so `ENGINE_FG3=decision8` and
                        `artifact` now agree. Which one ran is still recorded.
    rebound             ADOPTED (lgbm/C_plus_state) -- refit by the engine
                        because the trainer persists nothing.
    free_throw          ADOPTED (lgbm) -- same, plus the rule table, which is
                        deterministic and needs no fit.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cbb_sim.engine.inputs import EngineInputs, FeaturePlan, plan_features
from cbb_sim.engine.manifest import ArtifactManifest
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
    "x_score_diff__seconds_remaining", "chance_number", "chance_number_at_start",
    "is_transition", "is_transition_f", "chance_elapsed_s",
    "prev_end_DREB", "prev_end_TOV", "prev_end_made_FG", "prev_end_made_FT", "prev_end_other",
    "miss_rim", "miss_jump2", "miss_three", "blocked_f",
    # --- fg_make round 2 (experiments.md section 13). APPENDED, never
    # inserted: every existing plan indexes this tuple by position, so a new
    # column may only ever go on the end. `score_diff_pre` is the SAME live
    # value as `score_diff` on the simulated side -- the engine's margin is
    # already pre-shot -- and carries a different NAME because in TRAINING the
    # two are different quantities (the training `score_diff` is post-outcome,
    # `docs/tests/fg_make_state_confound_2026-09-10.md`).
    "score_diff_pre", "gt_flag", "eg_trail", "eg_lead",
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
    mode: str = "reference"
    #: `round2_s1` only. One fitted model per (population, monthly refit), the
    #: refit index per game, and the round-2 team block the round-2 arms were
    #: trained on.
    models_first: tuple = ()
    models_cont: tuple = ()
    team_block: np.ndarray | None = None
    #: one `ArtifactManifest` per population. The per-game artifact choice is
    #: made by `cbb_sim.engine.manifest`, not here, because S1 is the standing
    #: default for EVERY sub-model and the rule must live in one place.
    manifests: dict = field(default_factory=dict)

    @classmethod
    def load(cls, inp: EngineInputs, mode: str = "reference",
             fold: str = "F2", season: int = 2025) -> EventAdapter:
        if mode == "reference":
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
                       models["first"], models["cont"], True, src, mode="reference")

        if mode != "round2_s1":
            raise NotImplementedError(
                "ENGINE_EVENT must be 'reference' (round-1 best-loss arms, not adopted) "
                "or 'round2_s1' (the round-2 winners: lgbm+S1 first, cascade+S1 cont)")

        # ---- round 2, scheme S1 -------------------------------------------
        # S1 is a SCHEDULE, not a model: one refit per calendar month of the
        # test season, each fitted only on games strictly before its own refit
        # date, and each game scored by the most recent refit at or before its
        # own date. `scripts/build_engine_event_round2.py` replayed that
        # schedule and asserted, per GAME, that the selected model's last
        # training date precedes that game's tipoff. The engine only indexes.
        d = ENGINE_DIR / f"event_round2_s1_{fold}_{season}"
        if not d.exists():
            raise FileNotFoundError(
                f"{d} missing; run scripts/build_engine_event_round2.py --fold {fold} "
                f"--season {season}, or pass ENGINE_EVENT=reference")
        idx = json.loads((d / "index.json").read_text(encoding="utf-8"))
        z = np.load(d / "team_block.npz")
        team_block = z["team_block"]
        if len(team_block) != inp.n_games:
            raise ValueError(
                f"round-2 team block covers {len(team_block)} games but the engine inputs "
                f"carry {inp.n_games}; rebuild both from the same universe")
        # The round-2 arms are served the round-2 columns, so the plan resolves
        # against the block's OWN name map, not `inp.team_names`.
        r2_names = {c: i for i, c in enumerate(idx["team_cols"])}
        loaded, plans, arms, src, mans = {}, {}, {}, {}, {}
        for pop in ("first", "cont"):
            info = idx["populations"][pop]
            # The per-game artifact choice goes through the shared manifest
            # rule, which re-derives it from the dated entries and re-asserts
            # both safety checks at load; the builder's own precomputed index
            # is deliberately NOT trusted here.
            man = ArtifactManifest.from_obj(
                {"model": "possession_outcome", "scheme": "S1", "key": pop,
                 "fold": fold, "season": season,
                 "artifacts": [{"refit_date": sg["refit_date"], "path": sg["file"],
                                "max_train_date": sg["max_train_date"],
                                "n_train": sg.get("n_train")}
                               for sg in info["segments"]]},
                d, inp.games)
            ms = []
            for e in man.entries:
                m = joblib.load(e.path)["model"]
                for b in (getattr(m, "clf_", None),):
                    try:
                        b.set_params(n_jobs=1)
                    except Exception:                              # noqa: BLE001
                        pass
                ms.append(m)
            loaded[pop] = tuple(ms)
            mans[pop] = man
            arms[pop] = info["arm"]
            plans[pop] = plan_features(info["features"], r2_names, {}, STATE_INDEX)
            src[pop] = {
                "path": str(d), "arm": info["arm"], "feature_set": info["feature_set"],
                "adopted": True, **man.provenance(),
                "note": "round-2 winner; train_possession_outcome_v2.py persists no booster, so "
                        "the engine refits the winner's own spec (as for rebound/free_throw)",
            }
        src["team_block"] = idx["team_block_provenance"]
        return cls(plans["first"], plans["cont"], arms["first"], arms["cont"],
                   loaded["first"][-1], loaded["cont"][-1], False, src,
                   mode="round2_s1", models_first=loaded["first"],
                   models_cont=loaded["cont"], team_block=team_block, manifests=mans)

    def predict(self, team: np.ndarray, state: np.ndarray, is_first: np.ndarray,
                gidx: np.ndarray | None = None,
                off: np.ndarray | None = None) -> np.ndarray:
        """(n, 6) in `PO.CLASSES` order.

        `reference`: one batched predict per population.
        `round2_s1`: one batched predict per (population, monthly refit). A
        simulated game's date is fixed, so its refit index is a per-game
        constant looked up once -- the loop still makes no per-game call. A
        block of games that spans one month therefore costs exactly what
        `reference` costs; one that spans k months costs k batched predicts
        over disjoint row sets, which is the same total row count."""
        n = len(state)
        out = np.empty((n, len(PO.CLASSES)), dtype=np.float64)
        if self.mode == "reference":
            for mask, plan, model in ((is_first, self.plan_first, self.model_first),
                                      (~is_first, self.plan_cont, self.model_cont)):
                if not mask.any():
                    continue
                m = _assemble(plan, team[mask], None, state[mask])
                out[mask] = model.predict_proba(np.ascontiguousarray(m, dtype=np.float32))
            return out

        if gidx is None or off is None:
            raise ValueError("ENGINE_EVENT=round2_s1 needs (gidx, off) to select the "
                             "monthly refit and the round-2 team row")
        team_r2 = self.team_block[gidx, off]
        for pop, mask, plan, models in (
                ("first", is_first, self.plan_first, self.models_first),
                ("cont", ~is_first, self.plan_cont, self.models_cont)):
            if not mask.any():
                continue
            rows = np.flatnonzero(mask)
            s = self.manifests[pop].segments(gidx[rows])
            for k in np.unique(s):
                r = rows[s == k]
                m = _assemble(plan, team_r2[r], None, state[r])
                out[r] = models[k].predict_proba(np.ascontiguousarray(m, dtype=np.float32))
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
    mode: str = "reference"
    #: `reference_empirical` only. `EmpiricalArm._codes` reads two RAW columns
    #: that are not in the model's own declared feature list -- `prev_end` as a
    #: STRING level and `season` as an integer -- so a matrix built from
    #: `arm.features` alone cannot drive it. They are supplied here.
    season: int = 2025

    @classmethod
    def load(cls, inp: EngineInputs, mode: str = "reference",
             season: int = 2025) -> ClockAdapter:
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
                    "fold": d["fold"], "adopted": bool(d["adopted"]),
                    "note": ("binned lookup table: EmpiricalArm.level_pmfs over 7 nested "
                             "state dimensions" if mode == "reference_empirical"
                             else "live batched predict, 9 quantile boosters")},
                   mode=mode, season=int(season))

    def pmf(self, team: np.ndarray, state: np.ndarray) -> np.ndarray:
        m = _assemble(self.plan, team, None, state)
        df = pd.DataFrame(m, columns=list(self.plan.features), copy=False)
        if self.mode == "reference_empirical":
            # Rebuild the two raw columns the binned arm codes its cells from.
            # `prev_end` comes back from the five dummies the plan already
            # carries: `clock.PREV_END_LEVELS` and the engine's
            # `state.PREV_END_CODE` are built from the SAME tuple in the same
            # order, and an all-zero dummy row is `period_start`, which has no
            # dummy of its own. This is a decode of a value the engine already
            # holds, not a new modelling choice.
            d = np.column_stack([df[c].to_numpy() for c in CK.PREV_END_DUMMIES])
            code = np.where(d.any(axis=1), d.argmax(axis=1) + 1, 0)
            df = df.assign(
                prev_end=np.asarray(CK.PREV_END_LEVELS, dtype=object)[code],
                season=np.int64(self.season))
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

    #: `winner` (the adopted round-1 artifacts, the default and the historical
    #: behaviour), `round2_S_A` .. `round2_S_E` (fg_make round 2,
    #: `docs/models/fg_make/experiments.md` section 13) or `round2b_S_C_s1`
    #: (round 2b, section 15: the same arm as a monthly S1 schedule).
    mode: str = "winner"
    #: round 2b only: one `ArtifactManifest` per shot class, and the fitted
    #: model per (class, refit segment). The per-game artifact choice is made by
    #: `cbb_sim.engine.manifest`, never here, because S1 is the standing scheme
    #: for EVERY sub-model and that rule lives in one place.
    manifests: dict = field(default_factory=dict)
    models_by_seg: dict = field(default_factory=dict)

    @classmethod
    def load(cls, inp: EngineInputs, fold: str = "F2", fg3: str = "decision8",
             mode: str = "winner") -> FgMakeAdapter:
        if mode.startswith("round2b_"):
            return cls._load_round2b(inp, fold, mode)
        if mode != "winner":
            return cls._load_round2(inp, fold, mode)
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
        return cls(plans, arms, models, fsets, src, False, mode="winner")

    @classmethod
    def _load_round2(cls, inp: EngineInputs, fold: str, mode: str) -> FgMakeAdapter:
        """One fg_make ROUND-2 arm, from its own versioned directory.

        Nothing under `data/processed/models/engine/` and no
        `winner_FGA_*.joblib` is read or written here: a round-2 arm lives in
        `data/processed/models/fg_make/round2/<arm>/` and is selected only by
        `ENGINE_FG_MAKE`, whose default value keeps the old path byte for byte."""
        arm = mode.removeprefix("round2_")
        if arm not in FG.R2_ARMS:
            raise NotImplementedError(
                f"ENGINE_FG_MAKE must be 'winner' (the adopted round-1 artifacts) or one of "
                f"{['round2_' + a for a in FG.R2_ARMS]}; got {mode!r}")
        d = FG_DIR / "round2" / arm
        plans, arms_, models, fsets, src = {}, {}, {}, {}, {}
        for cls_name, key in (("FGA_rim", "rim"), ("FGA_jump2", "jump2"), ("FGA_3", "three")):
            p = d / f"fg_make_{cls_name}_{fold}.joblib"
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} missing; run scripts/train_fg_make_v2.py --export, or pass "
                    "ENGINE_FG_MAKE=winner")
            w = joblib.load(p)
            model = w["model"]
            try:
                model.clf_.set_params(n_jobs=1)
            except Exception:                                   # noqa: BLE001
                pass
            arms_[cls_name] = w["arm"]
            models[cls_name] = model
            fsets[cls_name] = w["feature_set"]
            plans[cls_name] = plan_features(
                w["features"], _alias(inp.team_names, FG_TEAM_ALIAS, key),
                _alias(inp.slot_names, FG_SLOT_ALIAS, key), STATE_INDEX)
            src[cls_name] = {"path": str(p), "arm": w["arm"],
                             "feature_set": w["feature_set"], "round2_arm": arm,
                             "note": w.get("note", "fg_make round 2 (experiments.md s13)"),
                             "adopted": bool(w.get("adopted", False))}
        # A round-2 arm is PROVISIONAL until the round-2 decision adopts it.
        provisional = not all(v.get("adopted") for v in src.values())
        return cls(plans, arms_, models, fsets, src, provisional, mode=mode)

    @classmethod
    def _load_round2b(cls, inp: EngineInputs, fold: str, mode: str) -> FgMakeAdapter:
        """fg_make round 2b: one arm served as a dated S1 schedule per class."""
        arm = mode.removeprefix("round2b_")
        d = FG_DIR / "round2b" / arm
        if not d.exists():
            raise FileNotFoundError(
                f"{d} missing; run scripts/train_fg_make_v2b_s1.py, or pass "
                "ENGINE_FG_MAKE=winner")
        plans, arms_, models, fsets, src, mans, by_seg = {}, {}, {}, {}, {}, {}, {}
        for cls_name, key in (("FGA_rim", "rim"), ("FGA_jump2", "jump2"), ("FGA_3", "three")):
            mpath = d / f"manifest_{cls_name}.json"
            if not mpath.exists():
                raise FileNotFoundError(f"{mpath} missing; run scripts/train_fg_make_v2b_s1.py")
            obj = json.loads(mpath.read_text(encoding="utf-8"))
            man = ArtifactManifest.from_obj(obj, d, inp.games)
            loaded = tuple(joblib.load(e.path) for e in man.entries)
            for w in loaded:
                try:
                    w["model"].clf_.set_params(n_jobs=1)
                except Exception:                               # noqa: BLE001
                    pass
            feats = obj["features"]
            arms_[cls_name] = "lgbm"
            models[cls_name] = loaded[-1]["model"]
            by_seg[cls_name] = tuple(w["model"] for w in loaded)
            fsets[cls_name] = obj["feature_set"]
            mans[cls_name] = man
            plans[cls_name] = plan_features(
                feats, _alias(inp.team_names, FG_TEAM_ALIAS, key),
                _alias(inp.slot_names, FG_SLOT_ALIAS, key), STATE_INDEX)
            src[cls_name] = {"path": str(d), "arm": "lgbm", "scheme": "S1",
                             "feature_set": obj["feature_set"], "round2_arm": obj.get("arm"),
                             "adopted": bool(loaded[-1].get("adopted", False)),
                             **man.provenance(),
                             "note": "fg_make round 2b (experiments.md s15): the round-2 "
                                     "winner under the standing S1 scheme"}
        provisional = not all(v.get("adopted") for v in src.values())
        return cls(plans, arms_, models, fsets, src, provisional, mode=mode,
                   manifests=mans, models_by_seg=by_seg)

    def predict(self, cls_name: str, team: np.ndarray, slot: np.ndarray,
                state: np.ndarray, gidx: np.ndarray | None = None) -> np.ndarray:
        """P(make) for one shot class.

        One batched predict, except under an S1 schedule, where it is one
        batched predict per refit segment over disjoint row sets -- the same
        total row count, and still no per-game model call."""
        plan = self.plans[cls_name]
        if self.manifests:
            if gidx is None:
                raise ValueError(
                    f"ENGINE_FG_MAKE={self.mode} serves a dated S1 schedule and needs the "
                    "per-row game index to select each game's refit")
            out = np.empty(len(team), dtype=np.float64)
            segs = self.manifests[cls_name].segments(gidx)
            models = self.models_by_seg[cls_name]
            for k in np.unique(segs):
                r = np.flatnonzero(segs == k)
                m = _assemble(plan, team[r], slot[r], state[r])
                out[r] = models[k].predict_proba(
                    np.ascontiguousarray(m, dtype=np.float32))[:, FG.CLASS_INDEX["MAKE"]]
            return out
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


def _load_clock(inp: EngineInputs, mode: str, season: int):
    """Round-3c clock arms live in `clock_adapter_v3`; everything else is
    unchanged. `docs/models/clock/experiments.md` section 12. The import is
    deferred so this module has no new import-time dependency."""
    if mode.startswith("v3c_"):
        from cbb_sim.engine.clock_adapter_v3 import ClockAdapterV3
        return ClockAdapterV3.load(inp, mode, season)
    return ClockAdapter.load(inp, mode, season)


def _scheme_flags(event, clock, fg, ft, reb, usage) -> dict:
    """`scheme_static_<model>` per sub-model.

    An adapter that carries `ArtifactManifest`s reports False when any of them
    is a dated schedule; an adapter with no manifests is static by definition,
    because a single fitted object is a manifest of length one."""
    out = {}
    for name, ad in (("possession_outcome", event), ("clock", clock), ("fg_make", fg),
                     ("free_throw", ft), ("rebound", reb), ("usage", usage)):
        mans = getattr(ad, "manifests", None) or {}
        out[f"scheme_static_{name}"] = (not mans) or all(m.is_static for m in mans.values())
    # rotation is served from a single fitted json, never a schedule
    out["scheme_static_rotation"] = True
    return out


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
    def load(cls, inp: EngineInputs, fold: str = "F2", season: int = 2025) -> Adapters:
        ev_mode = os.environ.get("ENGINE_EVENT", "reference")
        ck_mode = os.environ.get("ENGINE_CLOCK", "reference")
        rot_mode = os.environ.get("ENGINE_ROTATION", "reference")
        fg3 = os.environ.get("ENGINE_FG3", "decision8")
        fg_mode = os.environ.get("ENGINE_FG_MAKE", "round2b_S_C_s1")
        if rot_mode != "reference":
            raise NotImplementedError(
                "the rotation bake-off adopted nothing; ENGINE_ROTATION=reference "
                "(R2 hierarchical Dirichlet + the fitted scheduler) is the only wired mode")
        event = EventAdapter.load(inp, ev_mode, fold, season)
        clock = _load_clock(inp, ck_mode, season)
        fg = FgMakeAdapter.load(inp, fold, fg3, fg_mode)
        ft = FreeThrowAdapter.load(inp, fold)
        reb = ReboundAdapter.load(inp, fold)
        usage = UsageAdapter.load(inp)
        rot_fit = ROT.RotationFit.from_json(ROT_FIT)
        flags = {
            "ENGINE_EVENT": ev_mode, "ENGINE_CLOCK": ck_mode,
            "ENGINE_ROTATION": rot_mode, "ENGINE_FG3": fg3,
            "ENGINE_FG_MAKE": fg_mode,
            "provisional_event": event.provisional,
            "provisional_clock": clock.provisional,
            "provisional_rotation": True,
            "provisional_usage": usage.provisional,
            "provisional_fg": fg.provisional,
            "provisional_rebound": reb.provisional,
            "provisional_free_throw": ft.provisional,
            "provisional_foul_accrual": True,
            "provisional_and_one": True,
            # Per sub-model: is it served by ONE undated artifact, or by a
            # dated S1 schedule? S1 is the standing default for every
            # sub-model (possession_outcome round 2; L21), so a `True` here is
            # a statement that this model has NOT yet been re-fit under it --
            # and it is the only way a reader can tell a deliberately static
            # model from one silently serving a stale S1 artifact.
            **_scheme_flags(event, clock, fg, ft, reb, usage),
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
