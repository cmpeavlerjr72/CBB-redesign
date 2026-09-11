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
    # --- round 4 (experiments.md section 14) -----------------------------
    # Same family, same P3 state, same S1 schedule; they differ only in how the
    # fit weights or partitions CALENDAR TIME, and A4 in the clock-bucket floor.
    "v4_recency_P3_s1": {"manifest": "v4_s1/manifest_recency.json",
                         "base_arm": "empirical_km3_srfloor", "parametrisation": "P3"},
    "v4_curseason_P3_s1": {"manifest": "v4_s1/manifest_curseason.json",
                           "base_arm": "empirical_km3_srfloor", "parametrisation": "P3"},
    "v4_calpart_P3_s1": {"manifest": "v4_s1/manifest_calpart.json",
                         "base_arm": "empirical_km3_srfloor", "parametrisation": "P3"},
    "v4_nofloor_P3_s1": {"manifest": "v4_s1/manifest_nofloor.json",
                         "base_arm": "empirical_km3", "parametrisation": "P3"},
    "v4_ref_P3_s1": {"manifest": "v4_s1/manifest_ref.json",
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

#: Round-4 diagnosis cell grid (`ENGINE_CLOCK_DIAG=1`). The engine writes no
#: possession-level file, so the state COMPOSITION it visits -- as opposed to
#: the conditional law it draws from -- can only be accumulated live. These are
#: the round-2 cell dimensions the offline tables are cut on, so the sim and the
#: data are read on the same grid.
CELL_DIMS: tuple[tuple[str, int], ...] = (
    ("prev_end", len(CK.PREV_END_LEVELS)),
    ("r2_bucket", len(CK.R2_SR_LABELS)),
    ("period_group", 3),
    ("bonus", 2),
    ("tempo_tercile", 3),
)
CELL_SHAPE: tuple[int, ...] = tuple(n for _, n in CELL_DIMS)
CELL_N: int = int(np.prod(CELL_SHAPE))


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
class CellAccumulator:
    """Counts and consumed seconds per round-2 state cell, accumulated LIVE.

    Round 3c's residual is a mean-duration shortfall (L31). It has exactly two
    sources: the conditional law the model draws from, and the distribution of
    STATES the engine visits. The first is measurable offline on the real
    possessions; the second is only measurable here, because the engine writes
    no possession-level file. Enabled by `ENGINE_CLOCK_DIAG=1`; off, this costs
    one `if`."""

    n: np.ndarray = field(default_factory=lambda: np.zeros(CELL_N, dtype=np.int64))
    sum_consumed: np.ndarray = field(default_factory=lambda: np.zeros(CELL_N))
    sum_intended: np.ndarray = field(default_factory=lambda: np.zeros(CELL_N))

    def add(self, codes: np.ndarray, consumed: np.ndarray, intended: np.ndarray) -> None:
        np.add.at(self.n, codes, 1)
        np.add.at(self.sum_consumed, codes, consumed)
        np.add.at(self.sum_intended, codes, intended)

    def snapshot(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("n", "sum_consumed", "sum_intended")}

    def drain(self) -> dict:
        out = self.snapshot()
        self.n = np.zeros(CELL_N, dtype=np.int64)
        self.sum_consumed = np.zeros(CELL_N)
        self.sum_intended = np.zeros(CELL_N)
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
    #: The season being simulated, and each slate game's calendar month. Round
    #: 4's calendar arms code a cell from them; every other arm ignores them.
    season: int = 2025
    game_month: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    #: `ENGINE_CLOCK_DIAG=1`: accumulate the round-4 state-composition cells.
    diag: bool = False
    cells: CellAccumulator = field(default_factory=CellAccumulator)
    #: every fitted object of the schedule, indexed by manifest entry. Length 1
    #: and equal to `(arm,)` when a segment is pinned.
    arms: tuple = ()

    #: With a pinned segment and no game index, every row is served that
    #: segment's own refit month -- which is what a pinned run IS. Never a guess.
    pinned_month: int | None = None

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
        gm = pd.to_datetime(inp.games["game_date"]).dt.month.to_numpy().astype(np.int64)
        return cls(mode=mode, arm=arms[0], manifests={"clock": man}, segment=pinned,
                   freeze=freeze, source=src, team_idx=team_idx,
                   state_idx=dict(STATE_INDEX), arms=tuple(arms),
                   season=int(season), game_month=gm,
                   pinned_month=(None if pinned is None
                                 else int(man.entries[pinned].refit_date.month)),
                   diag=os.environ.get("ENGINE_CLOCK_DIAG", "0") == "1")

    # -- the schedule the runner has to honour ----------------------------
    def game_segments(self) -> np.ndarray:
        """Per game of the engine slate, the manifest entry that may serve it."""
        return self.manifests["clock"].seg_of_game

    # -- prediction -------------------------------------------------------
    def _frame(self, team: np.ndarray, state: np.ndarray,
               gidx: np.ndarray | None = None) -> pd.DataFrame:
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
        # Round 4's calendar arms read these two; every other arm ignores them.
        df["season"] = np.int64(self.season)
        if gidx is not None and len(self.game_month):
            df["month"] = self.game_month[np.asarray(gidx)]
        elif self.pinned_month is not None:
            df["month"] = np.int64(self.pinned_month)
        return df

    def pmf(self, team: np.ndarray, state: np.ndarray,
            gidx: np.ndarray | None = None) -> np.ndarray:
        """The predictive law per row.

        One batched `pmf` when a segment is pinned; otherwise one batched `pmf`
        PER MANIFEST ENTRY over disjoint row sets, selected by each row's own
        game through `ArtifactManifest.segments`. Same total row count, no
        per-game model call, and a game can only ever be served the artifact
        the manifest's honest-backtest checks already cleared for it."""
        return self._pmf_from_frame(self._frame(team, state, gidx), gidx)

    def _pmf_from_frame(self, df: pd.DataFrame, gidx: np.ndarray | None) -> np.ndarray:
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
        df = self._frame(team, state, gidx)
        dur = CK.sample_from_pmf(self._pmf_from_frame(df, gidx), u)
        left = state[:, self.state_idx["seconds_remaining"]].astype(np.int64)
        self.eoh.add(state[:, self.state_idx["period"]], left, dur)
        if self.diag:
            self.cells.add(self._cell_codes(df),
                           np.minimum(dur, left).astype(np.float64),
                           dur.astype(np.float64))
        return dur

    def _cell_codes(self, df: pd.DataFrame) -> np.ndarray:
        """Round-2 cell index per row, from the frame the model itself saw."""
        prev = df["prev_end"].map(CK.PREV_END_INDEX).to_numpy().astype(np.int64)
        bucket = CK.r2_bucket_id(df["seconds_remaining"].to_numpy()).astype(np.int64)
        per = df["period"].to_numpy()
        pg = np.where(per <= 1.0, 0, np.where(per <= 2.0, 1, 2)).astype(np.int64)
        bonus = (df["in_bonus"].to_numpy() > 0).astype(np.int64)
        edges = tempo_edges_of(self.arms[0])
        tempo = (np.searchsorted(np.asarray(edges),
                                 df["tempo_prior_game"].to_numpy(), side="right")
                 .astype(np.int64) if edges is not None
                 else np.zeros(len(df), dtype=np.int64))
        return np.ravel_multi_index((prev, bucket, pg, bonus, tempo), CELL_SHAPE)

    # -- diagnostics ------------------------------------------------------
    def cell_snapshot(self) -> dict:  # noqa: D401
        return self.cells.snapshot()

    def cell_drain(self) -> dict:
        return self.cells.drain()

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


def tempo_edges_of(arm) -> tuple | None:
    """The fitted tempo-tercile cut points, through any state wrapper.

    A P2/P3 arm is a `StateWrapArm` around the fitted cell table, so the edges
    live one level in. Walked rather than duplicated: the diagnosis must code
    its cells on the SAME cut points the model itself used, or the composition
    table would be measuring a different grid from the one the engine drew on."""
    seen = set()
    while arm is not None and id(arm) not in seen:
        seen.add(id(arm))
        e = getattr(arm, "tempo_edges", None)
        if e is not None:
            return tuple(e)
        arm = getattr(arm, "inner", None)
    return None


def summarise_cells(snaps: list[dict]) -> pd.DataFrame:
    """Fold worker cell snapshots into one long table.

    One row per occupied round-2 cell with the engine's own possession count,
    mean CONSUMED duration (what `loop.py` subtracts from the clock) and mean
    INTENDED duration (what the model drew). Read against the same cut of the
    real possessions, this separates the conditional law from the state
    composition."""
    tot = {k: np.zeros(CELL_N, dtype=np.float64) for k in
           ("n", "sum_consumed", "sum_intended")}
    for s in snaps:
        for k, v in tot.items():
            v += np.asarray(s[k], dtype=np.float64)
    idx = np.flatnonzero(tot["n"] > 0)
    codes = np.array(np.unravel_index(idx, CELL_SHAPE)).T
    out = pd.DataFrame(codes, columns=[d for d, _ in CELL_DIMS])
    out["prev_end"] = np.asarray(CK.PREV_END_LEVELS, dtype=object)[out["prev_end"]]
    out["r2_bucket"] = np.asarray(CK.R2_SR_LABELS, dtype=object)[out["r2_bucket"]]
    out["period_group"] = np.asarray(PERIOD_KEYS, dtype=object)[out["period_group"]]
    out["n"] = tot["n"][idx]
    out["mean_consumed"] = tot["sum_consumed"][idx] / tot["n"][idx]
    out["mean_intended"] = tot["sum_intended"][idx] / tot["n"][idx]
    out["sum_consumed"] = tot["sum_consumed"][idx]
    return out


# ===========================================================================
# ROUND 5 (experiments.md section 16): the within-game duration latent
# ===========================================================================
#: `ENGINE_CLOCK` value -> the round-4 reference arm it wraps and the fitted
#: parameter it reads. NOT ADOPTED and NOT a default: `adapters.py` still
#: serves `v3c_srfloor_P3_s1` and this lane does not change it.
V5_MODES: dict[str, dict] = {
    "v5_glat_shared": {"base_mode": "v3c_srfloor_P3_s1", "unit": "game",
                       "param": "A1_sigma", "loc": "minus_half",
                       "params_file": "v5_bakeoff/v5_bakeoff_report.json"},
    # Round 5b (experiments.md section 21): the SAME latent, moved so that the
    # POSSESSION COUNT's expectation is preserved rather than the duration
    # mean's. B1 is the analytic identity E[1/A]=1; B2 additionally re-estimates
    # the law's mean level with the latent present, on TRAINING rows.
    "v5b_glat_pmean": {"base_mode": "v3c_srfloor_P3_s1", "unit": "game",
                       "param": "B1_sigma", "loc": "plus_half",
                       "params_file": "v5b_bakeoff/v5b_bakeoff_report.json"},
    "v5b_glat_joint": {"base_mode": "v3c_srfloor_P3_s1", "unit": "game",
                       "param": "B2_sigma", "loc": "log_c",
                       "log_c_param": "B2_log_c",
                       "params_file": "v5b_bakeoff/v5b_bakeoff_report.json"},
    # Round 5d (experiments.md section 26): round 5c's arm C4, the B1 latent
    # with `sigma^2` a QUADRATIC in the PREGAME tempo feature. Same location
    # (`m = +sigma^2/2`, row-wise, so `E[1/A] = 1` holds per game), same stream,
    # same ordinal -- only the map from the shared uniform to `A` differs, which
    # is what makes a C4 run pair with a B1 run game by game. DEFAULT-OFF.
    "v5d_glat_pquad": {"base_mode": "v3c_srfloor_P3_s1", "unit": "game",
                       "param": "B1_sigma", "loc": "plus_half",
                       "sigma_fn": "pace2", "beta_param": "C4_beta",
                       "centre_param": "C4_tbar",
                       "params_file": "v5c_bakeoff/v5c_params.json",
                       "params_root": None},
}

#: The pregame tempo feature the round-5d dispersion function is a function of.
#: It is a TEAM_COLS member the served round-3c frame already carries, and it is
#: game-level in the engine inputs (both team rows of a game carry the same
#: value), so one draw is one pace realisation for the whole game and BOTH
#: teams -- the CLAUDE.md modeling rule the latent exists to honour.
TEMPO_COL: str = "tempo_prior_game"

#: `exp_clk5c_dispersion_function.main` floors the fitted `sigma^2` at this
#: value before taking a square root. Carried here so the engine's per-row
#: sigma is the same function of the same coefficients as the offline grade's.
#: Over the engine's observed tempo range the C4 quadratic is strictly
#: positive, so the floor is a guard against a degenerate coefficient set and
#: never active -- it is not a clip on engine output.
SIGMA2_FLOOR: float = 1e-8

#: The bake-off's fitted parameters, read rather than re-derived. Same rule as
#: round 3c's "the fitted object is READ, never reimplemented": the engine draws
#: from the object the offline grade scored, with no second implementation.
V5_PARAMS = CK_DIR / "v5_bakeoff" / "v5_bakeoff_report.json"

#: The ordinal the game latent is drawn at on the `clock` stream. The
#: possession counter advances once per possession and never approaches 2^40,
#: so the latent's uniform can never collide with a duration draw -- and no new
#: RNG family is introduced, so every other sub-model's stream is untouched and
#: a paired arm still differences game by game (CLAUDE.md's RNG rule).
LATENT_ORDINAL: int = 1 << 40


@dataclass
class LatentClockAdapter:
    """The served cell law, scale-mixed by ONE pace realisation per simulation.

    `CLAUDE.md`: "One pace realisation per simulated game, both teams scaled by
    it. Dispersion comes from the model's own variance function and is
    validated against realised residual SD." The clock has never implemented
    the first sentence -- `clock.sample_from_pmf` draws every possession
    independently -- and the measurement
    (`docs/tests/clock_duration_dispersion_2026-09-11.md`) prices that omission
    at 98.9% of the per-game possession-SD gap.

    `A = exp(sigma*z - sigma^2/2)` so `E[A] = 1` exactly: the conditional MEAN
    of every possession is unchanged and this arm cannot move round 4's mean
    gate in its own favour. `sigma` is fitted by method of moments on TRAINING
    rows only (`scripts/exp_clk5_dispersion_bakeoff.py`) and read from the
    bake-off report; it is a random-effects parameter, not a multiplier on
    engine output (`docs/SIM_GUARDRAILS.md` section 5).

    Only `unit="game"` (arm A1) is wired. Arm A2's per-offence latent needs the
    offensive side at the call site, which `loop.py` does not pass, and A1 is
    the arm the pre-registered simplicity order puts first among the arms that
    landed the primary -- so the closed loop runs A1 and A2 stays offline.
    """

    inner: ClockAdapterV3
    sigma: float
    unit: str
    mode: str
    source: dict
    eoh: EohAccumulator = field(default_factory=EohAccumulator)
    wants_game_index: bool = True
    #: `loop.py` reads this and passes the active rows' (seed, game_id, "clock")
    #: stream keys, which is the only per-SIMULATION identity the adapter has.
    wants_sim_keys: bool = True
    #: round-5b latent LOCATION (section 21.2). The default reproduces round 5.
    loc_kind: str = "minus_half"
    log_c: float = 0.0
    #: round-5d DISPERSION FUNCTION (section 26.1). `None` -- every round-5 and
    #: round-5b arm -- keeps `sigma` a fitted scalar and the scalar code path
    #: below BIT-IDENTICAL. `"pace2"` makes `sigma^2` the round-5c C4 quadratic
    #: in the pregame tempo feature, with `beta` and `tbar` READ from the
    #: offline fit, never re-derived.
    sigma_fn: str | None = None
    beta: tuple = ()
    tbar: float = 0.0

    @classmethod
    def load(cls, inp: EngineInputs, mode: str, season: int = 2025) -> LatentClockAdapter:
        if mode not in V5_MODES:
            raise NotImplementedError(
                f"ENGINE_CLOCK={mode!r} is not a round-5 arm; known arms are "
                f"{sorted(V5_MODES)} (docs/models/clock/experiments.md section 16)")
        spec = V5_MODES[mode]
        if not V5_PARAMS.exists():
            raise FileNotFoundError(
                f"{V5_PARAMS} missing; run scripts/exp_clk5_dispersion_bakeoff.py")
        pf = CK_DIR / spec.get("params_file", "v5_bakeoff/v5_bakeoff_report.json")
        if not pf.exists():
            raise FileNotFoundError(
                f"{pf} missing; run the round-5/5b bake-off script")
        rep = json.loads(pf.read_text(encoding="utf-8"))
        # Round 5's and round 5b's reports nest their fitted values under
        # `params`; round 5c's `v5c_params.json` is already keyed by fold. The
        # spec names the shape rather than the loader guessing it.
        fitted = rep["F2"] if spec.get("params_root", "params") is None \
            else rep["params"]["F2"]
        sigma = float(fitted[spec["param"]])
        log_c = float(fitted.get(spec.get("log_c_param", ""), 0.0)
                      ) if spec.get("log_c_param") else 0.0
        loc_kind = str(spec.get("loc", "minus_half"))
        sigma_fn = spec.get("sigma_fn")
        beta: tuple = ()
        tbar = 0.0
        if sigma_fn is not None:
            beta = tuple(float(b) for b in fitted[spec["beta_param"]])
            tbar = float(fitted[spec["centre_param"]])
        inner = ClockAdapterV3.load(inp, spec["base_mode"], season)
        src = dict(inner.source)
        src.update({
            "round5_arm": mode, "latent_unit": spec["unit"],
            "latent_sigma": sigma, "latent_sigma_source": str(pf),
            "latent_loc_kind": loc_kind, "latent_log_c": log_c,
            "latent_sigma_fold": "F2 train {2022,2023,2024}, method of moments",
            "adopted": False,
            "note": ("round-5 closed-loop candidate (experiments.md section 16); "
                     "E[A]=1 scale mixture on the served cell law, NOT a default"),
        })
        if sigma_fn is not None:
            src.update({
                "latent_sigma_fn": sigma_fn, "latent_sigma_beta": list(beta),
                "latent_sigma_centre": tbar, "latent_sigma_feature": TEMPO_COL,
                "note": ("round-5d closed-loop candidate (experiments.md section "
                         "26); round-5c arm C4, sigma^2 quadratic in the PREGAME "
                         "tempo feature under the B1 location so E[1/A]=1 holds "
                         "per game. NOT a default, NOT adopted."),
            })
        return cls(inner=inner, sigma=sigma, unit=str(spec["unit"]), mode=mode,
                   source=src, loc_kind=loc_kind, log_c=log_c,
                   sigma_fn=sigma_fn, beta=beta, tbar=tbar)

    def __getattr__(self, name: str):
        """Everything this wrapper does not override is the inner adapter's.

        Only reached when normal attribute lookup fails, so the dataclass's own
        fields always win and the delegation cannot shadow them."""
        return getattr(self.__dict__["inner"], name)

    def _sigma_rows(self, team: np.ndarray) -> np.ndarray:
        """Per-ROW `sigma` under a round-5d dispersion function.

        `sigma^2 = b0 + b1*(t - tbar) + b2*(t - tbar)^2` with `t` the PREGAME
        tempo feature the served frame already carries. The coefficients are the
        offline fit's, read at load time; nothing is estimated here and nothing
        reads an outcome. `t` is game-level, so every row of a game gets the
        same `sigma` and the latent stays ONE pace realisation per game."""
        if self.sigma_fn != "pace2":
            raise NotImplementedError(
                f"ENGINE_CLOCK={self.mode} names dispersion function "
                f"{self.sigma_fn!r}, which this adapter does not implement")
        j = self.inner.team_idx[TEAM_COLS.index(TEMPO_COL)]
        t = team[:, j].astype(np.float64) - self.tbar
        b0, b1, b2 = self.beta
        s2 = np.clip(b0 + b1 * t + b2 * t * t, SIGMA2_FLOOR, None)
        return np.sqrt(s2)

    def _latent(self, keys: np.ndarray,
                team: np.ndarray | None = None) -> np.ndarray:
        from scipy.special import ndtri

        from cbb_sim.engine.rng import uniforms_at
        k = np.asarray(keys, dtype=np.uint64)
        u = uniforms_at(k, np.full(len(k), LATENT_ORDINAL, dtype=np.int64))
        from cbb_sim.models.clock_v5 import loc_for
        if self.sigma_fn is None:
            # UNCHANGED scalar path: every round-5 and round-5b arm, including
            # the served `v5b_glat_pmean`, reaches exactly these three lines.
            m = float(loc_for(self.sigma, self.loc_kind, self.log_c))
            return np.exp(self.sigma * ndtri(u) + m)
        if team is None:
            raise ValueError(
                f"ENGINE_CLOCK={self.mode} needs the per-row team-static block "
                f"to evaluate its dispersion function on {TEMPO_COL}")
        s = self._sigma_rows(team)
        # The B1 location, applied ROW-WISE at that row's own sigma, which is
        # what makes `E[1/A] = 1` hold per game rather than only on average.
        return np.exp(s * ndtri(u) + 0.5 * s * s)

    def pmf(self, team: np.ndarray, state: np.ndarray,
            gidx: np.ndarray | None = None) -> np.ndarray:
        """The BASE conditional law. The latent is a joint-law object and does
        not belong in a per-row pmf the engine never integrates over; the
        offline marginal that IS integrated lives in
        `cbb_sim.models.clock_v5.LatentArm.pmf` and is what the blind scorer
        read."""
        return self.inner.pmf(team, state, gidx)

    def draw(self, team: np.ndarray, state: np.ndarray, u: np.ndarray,
             gidx: np.ndarray | None = None,
             keys: np.ndarray | None = None) -> np.ndarray:
        if keys is None:
            raise ValueError(
                f"ENGINE_CLOCK={self.mode} needs the per-simulation stream keys; "
                "loop.py passes them when the adapter sets wants_sim_keys. "
                "Refusing to draw a game latent that is constant across seeds.")
        t = np.asarray(self.inner.draw(team, state, u, gidx), dtype=np.float64)
        d = np.clip(np.rint(self._latent(keys, team) * t), 0.0,
                    float(CK.DURATION_CAP))
        left = state[:, self.inner.state_idx["seconds_remaining"]].astype(np.int64)
        self.eoh.add(state[:, self.inner.state_idx["period"]], left, d)
        return d

    def eoh_snapshot(self) -> dict:
        return self.eoh.snapshot()

    def eoh_drain(self) -> dict:
        return self.eoh.drain()
