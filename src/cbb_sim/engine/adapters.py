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
    clock               NO ARM ADOPTED (0 of 20 eligible offline, 0 of 6 in the
                        round-3c closed loop). The DEFAULT since 2026-09-11 is
                        `ENGINE_CLOCK=v3c_srfloor_P3_s1`, the best arm inside the
                        engine (L31), served per game off its S1 manifest;
                        `reference` still wires round 1's best-CRPS
                        `reference_not_adopted_lgbm_quantile.pkl`.
                        -> provisional_clock=True either way
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
RB_DIR = Path("data/processed/models/rebound")
FT_DIR = Path("data/processed/models/free_throw")

#: rebound and free_throw both confirmed S1 as their training scheme
#: (each model's `experiments.md` section 7). Their manifests are DATED
#: schedules of the winning arm's own spec, so the engine stops refitting a
#: single static object and starts selecting per game, exactly as
#: possession_outcome and fg_make already do.
RB_S1_MANIFEST = RB_DIR / "s1_confirm" / "S1_weekly" / "F2" / "manifest.json"
FT_S1_MANIFEST = FT_DIR / "s1_confirm" / "S1_conf_aligned" / "F2" / "manifest.json"

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
    # fg_make round 4 (experiments.md section 19). `_alias` only adds a key
    # when the real column EXISTS in the inputs' own `slot_names`, so these two
    # are a no-op against the stock engine inputs and every pre-round-4
    # `ENGINE_FG_MAKE` value keeps resolving exactly as before. They resolve
    # only against a slot block built by
    # `scripts/build_engine_inputs_shotshooter.py --with-round4`.
    "shooter_shrunk_dev_c": "shooter_shrunk_dev_c__{k}",
    "prior_season_att_c": "prior_season_att_c__{k}",
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
#: The shooter column every artifact built before fg_make round 3 was keyed on.
#: A manifest that does not declare `shooter_key` predates the round-3 fix, so
#: this is the honest value to report for it, not a guess.
ES_DEFAULT_SHOOTER_KEY = "participant_1_id"

#: `ENGINE_FG_MAKE` prefix -> (artifact round directory, the trainer that
#: writes it). Every round from 2b on serves a DATED S1 schedule of the same
#: shape, so `_load_dated` handles all of them and the round directory is never
#: hard-coded inside it. Order matters only in that no prefix may be a prefix of
#: another; `round2_` is NOT here (it is the static round-2 loader).
_FG_DATED_ROUNDS: tuple[tuple[str, str, str], ...] = (
    ("round2b_", "round2b", "scripts/train_fg_make_v2b_s1.py"),
    ("round3_shooter_", "round3_shooter", "scripts/train_fg_make_v3_shooter.py"),
    ("round4_", "round4", "scripts/train_fg_make_v4_shooter_block.py"),
)
_FG_ROUND_NOTE = {
    "round2b": ("fg_make round 2b (experiments.md s15): the round-2 winner under "
                "the standing S1 scheme; shooter keyed on participant_1_id (L28/L29)"),
    "round3_shooter": ("fg_make round 3 (experiments.md s18, L29): the round-2b arm "
                       "retrained with the shooter keyed on shot_shooter_id; the "
                       "INTERIM served model, Decision-8 shooter slope open on "
                       "FGA_jump2 and FGA_3"),
    "round4": ("fg_make round 4 (experiments.md s19): the shooter block re-baked "
               "from scratch on shot_shooter_id"),
}


@dataclass
class FgMakeAdapter:
    plans: dict[str, FeaturePlan]
    arms: dict[str, str]
    models: dict[str, object]
    feature_sets: dict[str, str]
    source: dict
    provisional: bool

    #: `winner` (the adopted round-1 artifacts, the historical behaviour),
    #: `round2_S_A` .. `round2_S_E` (fg_make round 2,
    #: `docs/models/fg_make/experiments.md` section 13), `round2b_S_C_s1`
    #: (round 2b, section 15: the same arm as a monthly S1 schedule),
    #: `round3_shooter_S_C_s1` (round 3, section 18 + L29: the same arm again,
    #: trained with the shooter keyed on `shot_shooter_id`; the interim served
    #: model and today's DEFAULT), or `round4_<arm>` (round 4's shooter-block
    #: bake-off, section 19).
    mode: str = "winner"
    #: Every DATED round (2b, 3, 4): one `ArtifactManifest` per shot class, and
    #: the fitted model per (class, refit segment). The per-game artifact choice
    #: is made by `cbb_sim.engine.manifest`, never here, because S1 is the
    #: standing scheme for EVERY sub-model and that rule lives in one place.
    manifests: dict = field(default_factory=dict)
    models_by_seg: dict = field(default_factory=dict)

    @classmethod
    def load(cls, inp: EngineInputs, fold: str = "F2", fg3: str = "decision8",
             mode: str = "winner") -> FgMakeAdapter:
        for prefix, round_dir, trainer in _FG_DATED_ROUNDS:
            if mode.startswith(prefix):
                return cls._load_dated(inp, fold, mode, prefix, round_dir, trainer)
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
    def _load_dated(cls, inp: EngineInputs, fold: str, mode: str, prefix: str,
                    round_dir: str, trainer: str) -> FgMakeAdapter:
        """One fg_make arm served as a DATED S1 schedule per shot class.

        Round 2b was the first such round and this used to be `_load_round2b`
        with `round2b` hard-coded in the path. Rounds 3 and 4 produce artifact
        sets of exactly the same shape in their own directories, so the round
        directory is now derived from the `ENGINE_FG_MAKE` prefix
        (`_FG_DATED_ROUNDS`) rather than typed in. Every previously valid
        `ENGINE_FG_MAKE` value still resolves to the same files."""
        arm = mode.removeprefix(prefix)
        d = FG_DIR / round_dir / arm
        if not d.exists():
            raise FileNotFoundError(
                f"{d} missing; run {trainer}, or pass ENGINE_FG_MAKE=winner")
        plans, arms_, models, fsets, src, mans, by_seg = {}, {}, {}, {}, {}, {}, {}
        for cls_name, key in (("FGA_rim", "rim"), ("FGA_jump2", "jump2"), ("FGA_3", "three")):
            mpath = d / f"manifest_{cls_name}.json"
            if not mpath.exists():
                raise FileNotFoundError(f"{mpath} missing; run {trainer}")
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
                             "shooter_key": obj.get("shooter_key", ES_DEFAULT_SHOOTER_KEY),
                             **man.provenance(),
                             "note": _FG_ROUND_NOTE[round_dir]}
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
    #: `s1_conf_aligned` (the adopted scheme, default since 2026-09-11) or
    #: `static` (the engine's own single refit).
    mode: str = "static"
    manifest: ArtifactManifest | None = None
    models_by_seg: tuple = ()

    @classmethod
    def load(cls, inp: EngineInputs, fold: str = "F2",
             mode: str | None = None) -> FreeThrowAdapter:
        mode = mode or os.environ.get("ENGINE_FREE_THROW", "s1_conf_aligned")
        if mode == "s1_conf_aligned":
            return cls._load_dated(inp, fold, mode, FT_S1_MANIFEST)
        if mode != "static":
            raise NotImplementedError(
                "ENGINE_FREE_THROW must be 's1_conf_aligned' (the adopted S1 schedule) "
                f"or 'static' (the engine's own single refit); got {mode!r}")
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
                   {"path": str(p), "arm": d["arm"], "why": d["why"],
                    "scheme": "static"}, False,
                   FT.load_bonus_era(), mode="static")

    @classmethod
    def _load_dated(cls, inp: EngineInputs, fold: str, mode: str,
                    mpath: Path) -> FreeThrowAdapter:
        if not mpath.exists():
            raise FileNotFoundError(
                f"{mpath} missing; the free-throw S1 confirmation writes it. Pass "
                "ENGINE_FREE_THROW=static to run the engine's own single refit.")
        obj = json.loads(mpath.read_text(encoding="utf-8"))
        man = ArtifactManifest.from_obj(obj, mpath.parent, inp.games)
        loaded = tuple(joblib.load(e.path) for e in man.entries)
        feats = loaded[0]["features"]
        for w in loaded:
            if list(w["features"]) != list(feats):
                raise ValueError("free_throw S1 artifacts disagree on feature order; "
                                 "one plan cannot serve them")
            try:
                w["model"].clf_.set_params(n_jobs=1)
            except Exception:                                   # noqa: BLE001
                pass
        plan = plan_features(feats, inp.team_names,
                             _alias(inp.slot_names, FT_SLOT_ALIAS), STATE_INDEX)
        return cls(plan, loaded[-1]["model"],
                   {"path": str(mpath.parent), "arm": loaded[-1].get("arm", "lgbm"),
                    "feature_set": obj.get("key"), "scheme": obj.get("scheme"),
                    "why": "free-throw bake-off winner lgbm, served as the dated S1 "
                           "schedule its own scheme confirmation adopted",
                    "note": loaded[-1].get("note", ""), **man.provenance()},
                   False, FT.load_bonus_era(), mode=mode, manifest=man,
                   models_by_seg=tuple(w["model"] for w in loaded))

    def predict(self, team: np.ndarray, slot: np.ndarray, state: np.ndarray,
                gidx: np.ndarray | None = None) -> np.ndarray:
        if self.manifest is not None:
            if gidx is None:
                raise ValueError(
                    f"ENGINE_FREE_THROW={self.mode} serves a dated S1 schedule and "
                    "needs the per-row game index to select each game's refit")
            out = np.empty(len(team), dtype=np.float64)
            segs = self.manifest.segments(gidx)
            for k in np.unique(segs):
                r = np.flatnonzero(segs == k)
                m = _assemble(self.plan, team[r], slot[r], state[r])
                out[r] = self.models_by_seg[k].predict_proba(
                    np.ascontiguousarray(m, dtype=np.float32))[:, FT.CLASS_INDEX["MAKE"]]
            return out
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
    #: `s1_weekly` (the adopted scheme, default since 2026-09-11) or `static`
    #: (the engine's own single refit, kept so earlier gate reports reproduce).
    mode: str = "static"
    manifest: ArtifactManifest | None = None
    models_by_seg: tuple = ()

    @classmethod
    def load(cls, inp: EngineInputs, fold: str = "F2",
             mode: str | None = None) -> ReboundAdapter:
        mode = mode or os.environ.get("ENGINE_REBOUND", "s1_weekly")
        if mode == "s1_weekly":
            return cls._load_dated(inp, fold, mode, RB_S1_MANIFEST)
        if mode != "static":
            raise NotImplementedError(
                "ENGINE_REBOUND must be 's1_weekly' (the adopted S1 schedule) or "
                f"'static' (the engine's own single refit); got {mode!r}")
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
                   {"path": str(p), "arm": d["arm"], "why": d["why"],
                    "scheme": "static"}, False, mode="static")

    @classmethod
    def _load_dated(cls, inp: EngineInputs, fold: str, mode: str,
                    mpath: Path) -> ReboundAdapter:
        if not mpath.exists():
            raise FileNotFoundError(
                f"{mpath} missing; the rebound S1 confirmation writes it. Pass "
                "ENGINE_REBOUND=static to run the engine's own single refit.")
        obj = json.loads(mpath.read_text(encoding="utf-8"))
        man = ArtifactManifest.from_obj(obj, mpath.parent, inp.games)
        loaded = tuple(joblib.load(e.path) for e in man.entries)
        feats = loaded[0]["features"]
        for w in loaded:
            if list(w["features"]) != list(feats):
                raise ValueError("rebound S1 artifacts disagree on feature order; "
                                 "one plan cannot serve them")
            try:
                w["model"].clf_.set_params(n_jobs=1)
            except Exception:                                   # noqa: BLE001
                pass
        plan = plan_features(feats, inp.team_names, {}, STATE_INDEX)
        return cls(plan, loaded[-1]["model"], dict(inp.rules["dead_share"]),
                   {"path": str(mpath.parent), "arm": loaded[-1].get("arm", "lgbm"),
                    "feature_set": obj.get("key"), "scheme": obj.get("scheme"),
                    "why": "rebound bake-off winner lgbm/C_plus_state, served as the "
                           "dated S1 schedule its own scheme confirmation adopted",
                    "note": loaded[-1].get("note", ""), **man.provenance()},
                   False, mode=mode, manifest=man,
                   models_by_seg=tuple(w["model"] for w in loaded))

    def predict(self, team: np.ndarray, state: np.ndarray,
                gidx: np.ndarray | None = None) -> np.ndarray:
        """(n, 3) in `RB.CLASSES` order = (OREB, DREB, DEAD).

        One batched predict, except under an S1 schedule, where it is one
        batched predict per refit segment over disjoint row sets -- the same
        total row count, and still no per-game model call."""
        if self.manifest is not None:
            if gidx is None:
                raise ValueError(
                    f"ENGINE_REBOUND={self.mode} serves a dated S1 schedule and needs "
                    "the per-row game index to select each game's refit")
            out = np.empty((len(team), len(RB.CLASSES)), dtype=np.float64)
            segs = self.manifest.segments(gidx)
            for k in np.unique(segs):
                r = np.flatnonzero(segs == k)
                m = _assemble(self.plan, team[r], None, state[r])
                out[r] = self.models_by_seg[k].predict_proba(
                    np.ascontiguousarray(m, dtype=np.float32))
            return out
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


def RA_SCHEME() -> str:                                       # noqa: N802
    from cbb_sim.engine.rotation_adapter import rotation_scheme
    return rotation_scheme()


def RA_MANIFEST() -> Path:                                    # noqa: N802
    from cbb_sim.engine.rotation_adapter import R2_S1_MANIFEST
    return R2_S1_MANIFEST


def _load_clock(inp: EngineInputs, mode: str, season: int):
    """Round-3c and round-4 clock arms live in `clock_adapter_v3`; everything
    else is unchanged. `docs/models/clock/experiments.md` sections 12 and 14.
    The import is deferred so this module has no new import-time dependency."""
    if mode.startswith(("v3c_", "v4_")):
        from cbb_sim.engine.clock_adapter_v3 import ClockAdapterV3
        return ClockAdapterV3.load(inp, mode, season)
    return ClockAdapter.load(inp, mode, season)


def _manifests_of(ad) -> dict:
    """Every `ArtifactManifest` an adapter is serving, keyed for run_meta.

    An adapter exposes either `manifests` (a dict keyed by population or shot
    class) or a single `manifest`; both are normalised here so the run_meta
    writer has one shape to walk."""
    mans = dict(getattr(ad, "manifests", None) or {})
    one = getattr(ad, "manifest", None)
    if one is not None:
        mans.setdefault(one.key or "-", one)
    return mans


def _scheme_flags(event, clock, fg, ft, reb, usage, rotation_manifest=None) -> dict:
    """`scheme_static_<model>` per sub-model.

    An adapter that carries `ArtifactManifest`s reports False when any of them
    is a dated schedule; an adapter with no manifests is static by definition,
    because a single fitted object is a manifest of length one."""
    out = {}
    for name, ad in (("possession_outcome", event), ("clock", clock), ("fg_make", fg),
                     ("free_throw", ft), ("rebound", reb), ("usage", usage)):
        mans = _manifests_of(ad)
        out[f"scheme_static_{name}"] = (not mans) or all(m.is_static for m in mans.values())
    out["scheme_static_rotation"] = (rotation_manifest is None
                                     or bool(rotation_manifest.is_static))
    return out


def _train_dates(event, clock, fg, ft, reb, usage, rotation_manifest=None) -> dict:
    """`max_train_date` PER ARTIFACT, per sub-model, for `run_meta.json`.

    WHY THIS IS A FIRST-CLASS RUN_META FIELD. `CLAUDE.md`: "every backtest row
    must satisfy `created_at < tipoff`, enforced in code". `manifest._select`
    already asserts `max_train_date < game_date` for every game at LOAD time,
    which is the binding check -- but until now only the event model's dates
    reached the results directory, so a grader could re-assert the property for
    one family and had to take the other six on trust. Every family that serves
    a dated schedule now writes, per artifact, its `refit_date` and
    `max_train_date`, plus the LATEST max_train_date over the artifacts this
    run actually used (`max_train_date_overall`) -- which is the single number
    `scripts/grade_market_games_v2.py` compares against the earliest tipoff in
    the slate.

    A STATIC model reports `max_train_date: null` and `scheme: "static"` rather
    than a fabricated date: the engine's own refits (`rebound_F2.joblib`,
    `free_throw_F2.joblib`, the Decision-8 FGA_3 model) were fitted on the
    fold's TRAIN SEASONS, which is stated in `fold_train_seasons`, and inventing
    a per-artifact date for them would be exactly the kind of manufactured
    provenance the honest-backtest rule exists to stop."""
    out: dict = {}
    fams = [("possession_outcome", event), ("clock", clock), ("fg_make", fg),
            ("free_throw", ft), ("rebound", reb), ("usage", usage)]
    if rotation_manifest is not None:
        fams.append(("rotation", type("_M", (), {"manifest": rotation_manifest})()))
    for name, ad in fams:
        mans = _manifests_of(ad)
        if not mans:
            out[name] = {"scheme": "static", "max_train_date": None,
                         "n_artifacts": 1,
                         "why": "a single fitted object; its training window is the "
                                "fold's train seasons, reported in fold_train_seasons"}
            continue
        keys: dict = {}
        latest = None
        for key, man in mans.items():
            ents = [{"refit_date": str(e.refit_date.date()),
                     "max_train_date": (None if e.max_train_date is None
                                        else str(e.max_train_date.date())),
                     "path": Path(e.path).name} for e in man.entries]
            for e in man.entries:
                if e.max_train_date is not None:
                    latest = e.max_train_date if latest is None else max(latest, e.max_train_date)
            keys[str(key)] = {"scheme": man.scheme, "is_static": man.is_static,
                              "n_artifacts": len(man.entries), "artifacts": ents}
        out[name] = {"scheme": next(iter(mans.values())).scheme,
                     "max_train_date": None if latest is None else str(latest.date()),
                     "n_artifacts": sum(len(m.entries) for m in mans.values()),
                     "keys": keys}
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
    #: `ENGINE_ROTATION_SCHEME=s1`: the round-3b S1 schedule of R2 fits plus the
    #: per-game choice, gathered per chunk by `rotation_adapter.r2_s1_fitset`.
    #: `None` under `static`, where `rot_fit` alone is the model.
    rot_s1: dict | None = None

    @classmethod
    def load(cls, inp: EngineInputs, fold: str = "F2", season: int = 2025) -> Adapters:
        # DEFAULT, 2026-09-11: the possession-outcome ROUND-2 WINNERS (lgbm+S1
        # on first chances, cascade+S1 on continuations), which round 2 ADOPTED
        # on 2026-09-10. They have been wired behind this flag since that
        # evening but the default was never moved, so every run since has
        # silently served round 1's best-loss NOT-ADOPTED arms unless the caller
        # remembered to export the variable. `reference` stays selectable and
        # still sets provisional_event=True.
        ev_mode = os.environ.get("ENGINE_EVENT", "round2_s1")
        # DEFAULT, 2026-09-11 (PM decision, L31 + change ledger): the best arm
        # the project has produced inside the engine,
        # `empirical_km3_srfloor|P3|S1`. It is NOT ADOPTED -- round 3c passed no
        # arm and `provisional_clock` stays True -- this is a serving choice of
        # the best available arm over an incumbent it beats on G1 (+1.16 vs
        # +2.70 possessions), end-of-half duration (-0.3 s vs -6.0 s) and
        # Decision-8 responsiveness (1.06 vs 1.43). `reference` stays selectable
        # for reproduction of every gate report before this date.
        ck_mode = os.environ.get("ENGINE_CLOCK", "v3c_srfloor_P3_s1")
        rot_mode = os.environ.get("ENGINE_ROTATION", "reference")
        fg3 = os.environ.get("ENGINE_FG3", "decision8")
        # DEFAULT, 2026-09-10 (PM decision recorded in L29 and the change
        # ledger): the corrected-label round-3 arm. Round 2b is keyed on
        # `participant_1_id`, which L28 proved is the ASSISTER on half of all
        # assisted made field goals, so it is ineligible to be SERVED on data
        # integrity -- the same precedence round 2 applied to the leaked S-A.
        # `round2b_S_C_s1` stays selectable for reproduction.
        # DEFAULT, 2026-09-11 (fg_make round 4, experiments.md s20.6/s20.7):
        # `B1 R4_B1_shrunk`, the round-4 WINNER. It is the only arm that passes
        # calibration and Decision 8 on all three shot classes, it moves the
        # shooter slope from 0.28-0.31 (no shooter block) to 0.84-1.04, and it
        # holds the best closed-loop margin SD (13.757) and home/away
        # correlation (+0.242) of any fg_make arm. s20.7 recorded exactly one
        # blocker on making it the default -- `shooter_shrunk_dev_c` did not
        # exist in the shared engine inputs -- and engine inputs v2 closes it.
        # `round3_shooter_S_C_s1` (the interim) and every earlier value stay
        # selectable and resolve to the same files as before.
        fg_mode = os.environ.get("ENGINE_FG_MAKE", "round4_B1")
        rb_mode = os.environ.get("ENGINE_REBOUND", "s1_weekly")
        ft_mode = os.environ.get("ENGINE_FREE_THROW", "s1_conf_aligned")
        rot_scheme = RA_SCHEME()
        if rot_mode != "reference":
            raise NotImplementedError(
                "the rotation bake-off adopted nothing; ENGINE_ROTATION=reference "
                "(R2 hierarchical Dirichlet + the fitted scheduler) is the only wired mode")
        event = EventAdapter.load(inp, ev_mode, fold, season)
        clock = _load_clock(inp, ck_mode, season)
        fg = FgMakeAdapter.load(inp, fold, fg3, fg_mode)
        ft = FreeThrowAdapter.load(inp, fold, ft_mode)
        reb = ReboundAdapter.load(inp, fold, rb_mode)
        usage = UsageAdapter.load(inp)
        # ROTATION, 2026-09-11: round 3b adopted S1 as the SCHEME (not an arm --
        # no rotation arm has ever passed the state-dependence gate, so
        # `provisional_rotation` stays True). The six dated R2 fits are selected
        # per game by `engine.manifest`; `rot_fit` stays the LAST fit so any
        # caller that still wants one object gets a real one rather than None.
        rot_s1 = None
        rot_man = None
        if rot_mode == "reference" and rot_scheme == "s1":
            from cbb_sim.engine import rotation_adapter as _RA
            rot_s1 = _RA.load_r2_s1(inp.games)
            rot_man = rot_s1["manifest"]
            rot_fit = rot_s1["fits"][-1]
        else:
            rot_fit = ROT.RotationFit.from_json(ROT_FIT)
        flags = {
            "ENGINE_EVENT": ev_mode, "ENGINE_CLOCK": ck_mode,
            "ENGINE_ROTATION": rot_mode, "ENGINE_FG3": fg3,
            "ENGINE_FG_MAKE": fg_mode, "ENGINE_REBOUND": rb_mode,
            "ENGINE_FREE_THROW": ft_mode, "ENGINE_ROTATION_SCHEME": rot_scheme,
            "ENGINE_INPUTS_VERSION": str(inp.meta.get("inputs_version_loaded", "v1")),
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
            **_scheme_flags(event, clock, fg, ft, reb, usage, rot_man),
            # Every family's artifact dates, so a grader can assert
            # `max_train_date < tipoff` on ALL of them, not only the event model.
            "max_train_date": _train_dates(event, clock, fg, ft, reb, usage, rot_man),
            "fold_train_seasons": {"F1": [2022, 2023],
                                   "F2": [2022, 2023, 2024]}.get(fold, []),
            "rotation_s1_scope": (
                "S1 reaches what RotationBatch consumes (Dirichlet concentrations, "
                "min_share, the scheduler parameters, both tilt tables). The as-of "
                "PRIOR construction (k0, role_prior, p_play, fpm shrinkage, "
                "tail_ratio, w_dnp) is baked into the input arrays by "
                "build_engine_inputs.py from the STATIC fit and is unchanged; closing "
                "that half is an inputs-v3 item."
                if rot_man is not None else "static: one fit for the whole run"),
            "sources": {
                "event": event.source, "clock": clock.source, "fg_make": fg.source,
                "free_throw": ft.source, "rebound": reb.source, "usage": usage.source,
                "rotation": ({"path": str(RA_MANIFEST()), "adopted": False,
                              "arm": "R2_hier_dirichlet + scheduler",
                              "scheme": "S1 (rotation round 3b, experiments.md s9.4)",
                              "why": "no rotation arm passed the pre-registered "
                                     "state-dependence gate in any round; S1 is the "
                                     "adopted SCHEME, not an adopted arm",
                              **rot_s1["provenance"]}
                             if rot_s1 is not None else
                             {"path": str(ROT_FIT), "arm": "R2_hier_dirichlet + scheduler",
                              "adopted": False, "scheme": "static",
                              "why": "no rotation arm passed the pre-registered "
                                     "state-dependence gate in either round"}),
            },
        }
        return cls(event, clock, fg, ft, reb, usage, rot_fit, rot_mode, flags,
                   rot_s1=rot_s1)

    def describe(self) -> str:
        return json.dumps(self.flags, indent=1, default=str)
