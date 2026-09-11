"""
clock_adapter_v3.py -- the round-3c clock arms, inside the engine.

Pre-registration: `docs/models/clock/experiments.md` section 12. This module
holds ALL of round 3c's engine-side code; `adapters.py` carries a few-line hook
and every pre-existing `ENGINE_CLOCK` value keeps its exact behaviour.

WHAT THIS EXISTS TO DECIDE
--------------------------
Round 3b proved a negative (experiments.md 11.2): the offline chain feeds the
clock model the REAL score sequence, so it contains no feedback loop, so it
cannot rank P1 (margin live), P2 (margin deleted) and P3 (margin only as coarse
end-game indicators). Decision 10 says a state feature the ENGINE produces is
decided by a paired-stream run inside the engine. This is the adapter that run
needs.

THREE RULES IT KEEPS
--------------------
1. **The fitted object is READ, never reimplemented.** Every arm exposes
   `pmf(df) -> (n, 91)`, the same interface `clock.chain_halves` and the
   offline scorer consume, and the durations come out of `clock.sample_from_pmf`
   -- the module's own inverse-CDF sampler -- driven by the engine's own
   uniforms. The engine therefore draws from EXACTLY the object round 3b
   scored, with no second implementation to drift.
2. **Intended duration in, horn truncation in the engine (L20).** The adapter
   returns the duration the offence INTENDED. `loop.py` already applies
   `used = min(dur, left)`; nothing here clips, caps or reshapes a draw. A
   possession whose intended duration reaches the time left IS the period's
   last possession, which is what the end-of-half accumulator below counts.
3. **No game is served by an artifact that has seen it.** Artifact choice goes
   through `cbb_sim.engine.manifest.ArtifactManifest`, which enforces both
   `refit_date <= game_date` and `max_train_date < game_date` at load. The rule
   is not re-implemented here.

HOW S1 IS SERVED (round 4, 2026-09-11)
--------------------------------------
Originally `loop.py` called `clock.draw(team, state, u)` with no game index --
the clock was the one sub-model whose call site carried no `gidx`, because every
clock arm before round 3c was static -- so a 6-month schedule could only be
honoured by PARTITIONING THE RUN: `ENGINE_CLOCK_SEGMENT=k` serves manifest entry
`k` and the runner dispatches that month's games to that run. That made the
schedule unservable as an engine DEFAULT (a mixed-month batch raised rather than
silently serving one month's fit to a whole season).

Round 4 makes the call site game-indexed, the same way `EventAdapter.predict`
and `FgMakeAdapter.predict` already route their S1 schedules: the adapter
declares `wants_game_index = True`, `loop.py` passes the active rows' `gidx`,
and `ArtifactManifest.segments(gidx)` selects each GAME's own artifact inside
one run -- one batched `pmf` per segment over disjoint row sets, the same total
row count and still no per-game model call. `ENGINE_CLOCK_SEGMENT=k` keeps
working unchanged and pins the whole run to entry `k`, which is what
`scripts/run_clk3c_closed_loop.py` and every round-3c result used; the two paths
read the same `seg_of_game`, so they cannot disagree. An adapter that is handed
no `gidx` and has more than one entry still REFUSES rather than serving a stale
artifact -- the failure mode the change ledger calls out.

THE FREEZE FLAG
---------------
`ENGINE_CLOCK_FREEZE=1` is the Decision-10 ablation arm. It replaces the clock's
view of the offence's margin with ZERO on every row, and every margin-derived
column is then computed from that zero: `score_diff`, its clock interaction, the
score-state leg of round 2's fine-bucket cross, and P3's end-game indicators
alike. That is stronger than L23's two-column ablation, and it is what "frozen
at its pregame value" means for a model whose margin enters through several
derived columns. Under P2 it is a no-op by construction, which is the
implementation's own consistency check. It freezes the CLOCK ONLY: every other
sub-model still sees the live margin.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cbb_sim.engine.inputs import EngineInputs
from cbb_sim.engine.manifest import ArtifactManifest
from cbb_sim.models import clock as CK

CK_DIR = Path("data/processed/models/clock")

#: `ENGINE_CLOCK` value -> (manifest file relative to CK_DIR, base arm,
#: parametrisation). `gamma_P3` points at ROUND 3b's directory on purpose: that
#: schedule is served byte-identical to the one round 3b scored, and round 3c
#: never writes there.
V3C_MODES: dict[str, dict] = {
    "v3c_gamma_P1_s1": {"manifest": "v3c_s1/manifest_gamma_P1.json",
                        "base_arm": "gamma_aft", "parametrisation": "P1"},
    "v3c_gamma_P2_s1": {"manifest": "v3c_s1/manifest_gamma_P2.json",
                        "base_arm": "gamma_aft", "parametrisation": "P2"},
    "v3c_gamma_P3_s1": {"manifest": "v3b_s1/manifest.json",
                        "base_arm": "gamma_aft", "parametrisation": "P3"},
    "v3c_srfloor_P1_s1": {"manifest": "v3c_s1/manifest_srfloor_P1.json",
                          "base_arm": "empirical_km3_srfloor", "parametrisation": "P1"},
    "v3c_srfloor_P3_s1": {"manifest": "v3c_s1/manifest_srfloor_P3.json",
                          "base_arm": "empirical_km3_srfloor", "parametrisation": "P3"},
}

#: Team-static columns any round-3c arm reads. `tempo_prior_game` is on the
#: list twice over: it is a feature of every R2 set AND the cell arms' tempo
#: tercile, which is coded from the RAW value and not from the feature matrix.
TEAM_COLS: tuple[str, ...] = (
    "off_tempo_rel", "def_tempo_rel", "tempo_prior_game",
    "off_rating_off_c", "off_rating_def_c", "def_rating_off_c", "def_rating_def_c",
    "site_home", "site_away",
)

#: Period buckets the end-of-half accumulator reports separately.
PERIOD_KEYS: tuple[str, ...] = ("H1", "H2", "OT")

#: The offline end-of-half statistic's window (`clock.SECS_BUCKET_EDGES[1]`).
EOH_WINDOW_S = 35


def _manifest_obj(path: Path) -> dict:
    """Translate a clock S1 manifest into `ArtifactManifest`'s format.

    The clock trainer writes `months: [{refit_date, model_file, max_train_date,
    ...}]`; the engine's manifest reader wants `artifacts: [{refit_date, path,
    max_train_date}]`. Translating is a rename, so the SELECTION RULE and both
    honest-backtest checks stay in `cbb_sim.engine.manifest` where every
    sub-model passes through them."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    months = doc.get("months") or []
    if not months:
        raise ValueError(f"{path} lists no months")
    return {
        "model": "clock", "scheme": doc.get("scheme", "S1"),
        "fold": doc.get("fold", "F2"), "season": doc.get("season"),
        "artifacts": [{"refit_date": m["refit_date"], "path": m["model_file"],
                       "max_train_date": m["max_train_date"],
                       "month": m.get("month"), "n_train": m.get("n_train"),
                       "n_train_from_test_season": m.get("n_train_from_test_season")}
                      for m in months],
    }


@dataclass
class EohAccumulator:
    """The engine's analogue of the offline end-of-half statistic.

    A possession whose INTENDED duration reaches or exceeds the seconds left is
    the period's last possession by construction, because `loop.py` truncates it
    at the horn and the period then ends. For each such possession we record the
    seconds that were left at its start -- which is also the duration it
    actually consumed -- so the two offline quantities (share of halves whose
    last possession starts inside 35 s, and that possession's mean duration)
    both fall out. The engine writes no possession-level file, so this cannot be
    recomputed after a run; it is accumulated live or not at all."""

    n_poss: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))
    n_horn: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))
    n_horn_short: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))
    sum_last_dur: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    sum_intended: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def add(self, period: np.ndarray, left: np.ndarray, intended: np.ndarray) -> None:
        p = np.where(period <= 1.0, 0, np.where(period <= 2.0, 1, 2)).astype(np.int64)
        horn = intended >= left
        np.add.at(self.n_poss, p, 1)
        np.add.at(self.n_horn, p[horn], 1)
        np.add.at(self.sum_last_dur, p[horn], left[horn].astype(np.float64))
        np.add.at(self.sum_intended, p, np.minimum(intended, left).astype(np.float64))
        short = horn & (left < EOH_WINDOW_S)
        np.add.at(self.n_horn_short, p[short], 1)

    def snapshot(self) -> dict:
        return {k: getattr(self, k).tolist() for k in
                ("n_poss", "n_horn", "n_horn_short", "sum_last_dur", "sum_intended")}

    def drain(self) -> dict:
        """Snapshot and reset. A worker process runs many blocks and each block
        returns its OWN counts, so folding the blocks can never double-count."""
        out = self.snapshot()
        self.n_poss = np.zeros(3, dtype=np.int64)
        self.n_horn = np.zeros(3, dtype=np.int64)
        self.n_horn_short = np.zeros(3, dtype=np.int64)
        self.sum_last_dur = np.zeros(3, dtype=np.float64)
        self.sum_intended = np.zeros(3, dtype=np.float64)
        return out


@dataclass
class ClockAdapterV3:
    """One round-3c clock arm, batched, over the engine's own state block."""

    mode: str
    arm: object                      # the segment served when `segment` is pinned
    manifests: dict
    segment: int | None              # None = route per game through the manifest
    freeze: bool
    source: dict
    provisional: bool = True
    team_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    state_idx: dict = field(default_factory=dict)
    eoh: EohAccumulator = field(default_factory=EohAccumulator)
    #: every fitted object of the schedule, indexed by manifest entry. Length 1
    #: and equal to `(arm,)` when a segment is pinned.
    arms: tuple = ()

    #: `loop.py` reads this and passes the active rows' game index to `draw`.
    #: An adapter without it keeps the old three-argument call, so the
    #: incumbent `ClockAdapter` needs no change.
    wants_game_index: bool = True

    # -- construction -----------------------------------------------------
    @classmethod
    def load(cls, inp: EngineInputs, mode: str, season: int = 2025) -> ClockAdapterV3:
        if mode not in V3C_MODES:
            raise NotImplementedError(
                f"ENGINE_CLOCK={mode!r} is not a round-3c arm; known arms are "
                f"{sorted(V3C_MODES)} (docs/models/clock/experiments.md section 12)")
        spec = V3C_MODES[mode]
        mpath = CK_DIR / spec["manifest"]
        if not mpath.exists():
            raise FileNotFoundError(
                f"{mpath} missing; run scripts/train_clock_v3c_s1.py "
                f"(gamma_P3 is served from round 3b's v3b_s1/manifest.json)")
        man = ArtifactManifest.from_obj(_manifest_obj(mpath), CK_DIR, inp.games)

        seg_env = os.environ.get("ENGINE_CLOCK_SEGMENT", "")
        if seg_env == "":
            # Per-GAME routing: every entry is loaded and `pmf` dispatches each
            # row to its own game's artifact. This is the default path and the
            # one an engine DEFAULT needs.
            segs = list(range(len(man.entries)))
            pinned: int | None = None
        else:
            pinned = int(seg_env)
            if not 0 <= pinned < len(man.entries):
                raise ValueError(
                    f"ENGINE_CLOCK_SEGMENT={pinned} outside 0..{len(man.entries) - 1}")
            segs = [pinned]

        arms: list[object] = []
        for k in segs:
            entry = man.entries[k]
            with open(entry.path, "rb") as f:
                a = pickle.load(f)
            # The pickle must BE the parametrisation the mode names. A P2/P3
            # arm is a `StateWrapArm` that re-derives its own state columns; a
            # P1 arm is the bare fitted object. Serving one under the other's
            # name would be a silent train/serve skew, which is the whole
            # failure L23 is about.
            got = getattr(a, "parametrisation", "P1")
            if got != spec["parametrisation"]:
                raise ValueError(
                    f"{mode} names parametrisation {spec['parametrisation']} but "
                    f"{entry.path} holds {got}")
            arms.append(a)

        team_idx = np.array([inp.team_names[c] for c in TEAM_COLS], dtype=np.int64)
        from cbb_sim.engine.adapters import STATE_INDEX
        freeze = os.environ.get("ENGINE_CLOCK_FREEZE", "0") == "1"
        served = [man.entries[k] for k in segs]
        src = {
            "path": str(served[0].path) if pinned is not None else
            f"{len(served)} dated artifacts under {mpath.parent}",
            "arm": spec["base_arm"],
            "parametrisation": spec["parametrisation"], "scheme": "S1",
            "manifest": str(mpath),
            "segment": pinned, "routing": "pinned_segment" if pinned is not None
            else "per_game_manifest",
            "refit_date": [str(e.refit_date.date()) for e in served],
            "max_train_date": [None if e.max_train_date is None
                               else str(e.max_train_date.date()) for e in served],
            "n_segments": len(man.entries), "freeze_score_diff": freeze,
            "adopted": False,
            "note": ("round-3c closed-loop candidate (experiments.md section 12); "
                     "live fitted object, no lookup table, so no binning error (L26)"),
        }
        return cls(mode=mode, arm=arms[0], manifests={"clock": man}, segment=pinned,
                   freeze=freeze, source=src, team_idx=team_idx,
                   state_idx=dict(STATE_INDEX), arms=tuple(arms))

    # -- the schedule the runner has to honour ----------------------------
    def game_segments(self) -> np.ndarray:
        """Per game of the engine slate, the manifest entry that may serve it."""
        return self.manifests["clock"].seg_of_game

    # -- prediction -------------------------------------------------------
    def _frame(self, team: np.ndarray, state: np.ndarray) -> pd.DataFrame:
        """The design frame, built from the engine's own blocks.

        Every derived column is recomputed here from the LIVE clock and the
        (possibly frozen) margin using the clock module's own
        `add_r2_state`, exactly as `apply_clock_override` does offline -- so a
        simulated possession can never inherit a real one's bucket, and the
        engine's definition of a state column cannot drift from training's."""
        idx = self.state_idx
        sr = state[:, idx["seconds_remaining"]].astype(np.float64)
        per = state[:, idx["period"]].astype(np.float64)
        sd = (np.zeros(len(sr), dtype=np.float64) if self.freeze
              else state[:, idx["score_diff"]].astype(np.float64))
        cols: dict[str, np.ndarray] = {
            name: team[:, j].astype(np.float64)
            for name, j in zip(TEAM_COLS, self.team_idx, strict=True)
        }
        cols["period"] = per
        cols["seconds_remaining"] = sr
        cols["score_diff"] = sd
        cols["is_ot"] = state[:, idx["is_ot"]].astype(np.float64)
        cols["in_bonus"] = state[:, idx["in_bonus"]].astype(np.float64)
        cols["x_score_diff__seconds_remaining"] = sd * sr / 1200.0
        for name in CK.PREV_END_DUMMIES:
            cols[name] = state[:, idx[name]].astype(np.float64)
        df = pd.DataFrame(cols, copy=False)
        CK.add_r2_state(df)
        # The cell arms code their prev-end dimension off the STRING level. The
        # dummies and `clock.PREV_END_LEVELS` are built from the same tuple in
        # the same order, so this is a decode of a value the engine already
        # holds, not a new modelling choice.
        d = np.column_stack([cols[c] for c in CK.PREV_END_DUMMIES])
        code = np.where(d.any(axis=1), d.argmax(axis=1) + 1, 0)
        df["prev_end"] = np.asarray(CK.PREV_END_LEVELS, dtype=object)[code]
        return df

    def pmf(self, team: np.ndarray, state: np.ndarray,
            gidx: np.ndarray | None = None) -> np.ndarray:
        """The predictive law per row.

        One batched `pmf` when a segment is pinned; otherwise one batched `pmf`
        PER MANIFEST ENTRY over disjoint row sets, selected by each row's own
        game through `ArtifactManifest.segments`. Same total row count, no
        per-game model call, and a game can only ever be served the artifact
        the manifest's honest-backtest checks already cleared for it."""
        df = self._frame(team, state)
        if self.segment is not None or len(self.arms) == 1:
            return self.arms[0].pmf(df)
        if gidx is None:
            raise ValueError(
                f"ENGINE_CLOCK={self.mode} serves a dated S1 schedule of "
                f"{len(self.arms)} artifacts and needs the per-row game index to "
                "select each game's refit; pass gidx, or pin the run with "
                "ENGINE_CLOCK_SEGMENT=k. Refusing to silently serve one month's "
                "fit to a whole season.")
        segs = self.manifests["clock"].segments(np.asarray(gidx))
        out = np.empty((len(df), CK.DURATION_CAP + 1), dtype=np.float64)
        for k in np.unique(segs):
            r = np.flatnonzero(segs == k)
            out[r] = self.arms[int(k)].pmf(df.iloc[r].reset_index(drop=True))
        return out

    def draw(self, team: np.ndarray, state: np.ndarray, u: np.ndarray,
             gidx: np.ndarray | None = None) -> np.ndarray:
        """One INTENDED duration per row, by inverse CDF on the engine's own
        uniforms. `loop.py` truncates at the horn (L20); nothing is clipped
        here."""
        dur = CK.sample_from_pmf(self.pmf(team, state, gidx), u)
        left = state[:, self.state_idx["seconds_remaining"]].astype(np.int64)
        self.eoh.add(state[:, self.state_idx["period"]], left, dur)
        return dur

    # -- diagnostics ------------------------------------------------------
    def eoh_snapshot(self) -> dict:
        return self.eoh.snapshot()

    def eoh_drain(self) -> dict:
        return self.eoh.drain()


@dataclass
class RecordingClock:
    """A thin proxy that gives ANY clock adapter the same end-of-half counters.

    The incumbent `ClockAdapter` has no accumulator, and the round-3c table has
    to report the incumbent on exactly the same metric as the candidates or the
    comparison is not one. This wraps rather than edits: `adapters.py` is
    untouched, the inner adapter's `draw` is called unchanged, and the proxy
    only watches what goes past."""

    inner: object
    state_idx: dict
    eoh: EohAccumulator = field(default_factory=EohAccumulator)

    @property
    def wants_game_index(self) -> bool:
        """Mirror the inner adapter, so wrapping never changes what `loop.py`
        passes: a v3 arm still gets its `gidx`, the incumbent still does not."""
        return bool(getattr(self.inner, "wants_game_index", False))

    @property
    def provisional(self) -> bool:
        return bool(getattr(self.inner, "provisional", True))

    @property
    def source(self) -> dict:
        return dict(getattr(self.inner, "source", {}))

    @property
    def manifests(self) -> dict:
        return getattr(self.inner, "manifests", {}) or {}

    def pmf(self, team: np.ndarray, state: np.ndarray,
            gidx: np.ndarray | None = None) -> np.ndarray:
        if self.wants_game_index:
            return self.inner.pmf(team, state, gidx)
        return self.inner.pmf(team, state)

    def draw(self, team: np.ndarray, state: np.ndarray, u: np.ndarray,
             gidx: np.ndarray | None = None) -> np.ndarray:
        dur = (self.inner.draw(team, state, u, gidx) if self.wants_game_index
               else self.inner.draw(team, state, u))
        left = state[:, self.state_idx["seconds_remaining"]].astype(np.int64)
        self.eoh.add(state[:, self.state_idx["period"]], left, np.asarray(dur))
        return dur

    def eoh_snapshot(self) -> dict:
        return self.eoh.snapshot()

    def eoh_drain(self) -> dict:
        return self.eoh.drain()


def manifest_for(inp: EngineInputs, mode: str) -> ArtifactManifest | None:
    """The dated schedule behind an `ENGINE_CLOCK` value, without loading any
    fitted object. The runner needs it to dispatch each month's games to that
    month's run; returning None means the arm is static and takes one run."""
    if mode not in V3C_MODES:
        return None
    mpath = CK_DIR / V3C_MODES[mode]["manifest"]
    return ArtifactManifest.from_obj(_manifest_obj(mpath), CK_DIR, inp.games)


def summarise_eoh(snaps: list[dict]) -> dict:
    """Fold worker snapshots into the two offline end-of-half quantities."""
    tot = {k: np.zeros(3, dtype=np.float64) for k in
           ("n_poss", "n_horn", "n_horn_short", "sum_last_dur", "sum_intended")}
    for s in snaps:
        for k, v in tot.items():
            v += np.asarray(s[k], dtype=np.float64)
    out: dict = {}
    for i, key in enumerate(PERIOD_KEYS):
        nh = tot["n_horn"][i]
        out[key] = {
            "n_possessions": int(tot["n_poss"][i]),
            "n_period_ending": int(nh),
            "share_last_starts_under_35s": float(tot["n_horn_short"][i] / nh) if nh else float("nan"),
            "mean_last_possession_duration_s": float(tot["sum_last_dur"][i] / nh) if nh else float("nan"),
            "mean_possession_duration_s": (float(tot["sum_intended"][i] / tot["n_poss"][i])
                                           if tot["n_poss"][i] else float("nan")),
        }
    nh = tot["n_horn"][:2].sum()
    out["REG"] = {
        "n_possessions": int(tot["n_poss"][:2].sum()),
        "n_period_ending": int(nh),
        "share_last_starts_under_35s": float(tot["n_horn_short"][:2].sum() / nh) if nh else float("nan"),
        "mean_last_possession_duration_s": float(tot["sum_last_dur"][:2].sum() / nh) if nh else float("nan"),
        "mean_possession_duration_s": (float(tot["sum_intended"][:2].sum() / tot["n_poss"][:2].sum())
                                       if tot["n_poss"][:2].sum() else float("nan")),
    }
    return out
