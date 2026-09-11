"""
inputs.py -- the engine's per-game lookup tables, and the static/state splice.

CLAUDE.md: "Sim loop uses lookup tables and vectorized NumPy, never live model
calls." Every sub-model feature is one of exactly two kinds:

  STATIC   constant for the whole simulated game: as-of team form, as-of
           player form, site, season index, days since the season start, the
           fitted rotation prior. All of it is pregame by construction (every
           as-of column is a strictly-before expanding statistic), so it is
           computed ONCE per (game, team) or (game, team, roster slot) by
           `scripts/build_engine_inputs.py` and then only INDEXED in the loop.

  STATE    a function of the live simulation: period, seconds remaining, score
           difference, bonus, chance number, previous possession end, miss
           type, elapsed chance time. Recomputed per step, vectorised over the
           whole batch.

`FeaturePlan` is the splice: for one sub-model's declared feature order it
records, per column, whether the value comes from the static team block, the
static slot block, or the per-step state block, and at which index. Building a
design matrix for a step is then two fancy-index assignments and no pandas.

This is why there are no live model calls in a Python loop: the only thing the
loop does per step is gather static rows it already has and write the handful of
state columns, then hand ONE contiguous matrix to ONE batched predict.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIR = Path("data/processed/models/engine")

#: The inputs BUILD the engine loads, selected by `ENGINE_INPUTS_VERSION`.
#: `v1` is the original build; `v2` (2026-09-11) is the same arrays with the
#: fg_make shooter block re-keyed on `shot_shooter_id`, the fg_make round-4 slot
#: columns appended and `usage_rate` rebuilt off the usage round-2 panel
#: (`scripts/build_engine_inputs.py --version v2`). `v2` is the DEFAULT because
#: `ENGINE_FG_MAKE=round4_B1` cannot resolve `shooter_shrunk_dev_c` without it,
#: and because serving the round-3/4 models against a `participant_1_id` block
#: is the train/serve skew L32 priced at 0.97 pp of three-point make rate.
DEFAULT_INPUTS_VERSION = "v2"
INPUTS_VERSION_ENV = "ENGINE_INPUTS_VERSION"


def resolve_tag(in_dir: Path | str, tag: str, version: str | None = None) -> tuple[str, str]:
    """(`tag on disk`, `version actually loaded`).

    `v1` means the bare tag; any other version means `{tag}_{version}`.

    If the caller (or the environment) names a version EXPLICITLY, that version
    must exist or this raises -- a run must never silently get different inputs
    from the ones it asked for. If nothing is named, the default is preferred
    and the bare tag is used when the default's files are absent, which is what
    lets a directory built before versioning existed (e.g.
    `data/processed/models/engine_fgm4/`) keep loading unchanged. The version
    that was actually loaded is recorded in `meta["inputs_version_loaded"]` and
    printed, so the fallback is never silent."""
    d = Path(in_dir)
    asked = version if version is not None else os.environ.get(INPUTS_VERSION_ENV)
    explicit = asked is not None and asked != ""
    v = asked if explicit else DEFAULT_INPUTS_VERSION
    if v in ("v1", "base"):
        return tag, "v1"
    vt = f"{tag}_{v}"
    if (d / f"arrays_{vt}.npz").exists():
        return vt, v
    if explicit:
        raise FileNotFoundError(
            f"{d}/arrays_{vt}.npz missing, but inputs version {v!r} was asked for "
            f"explicitly. Build it (scripts/build_engine_inputs.py --version {v}) or "
            f"set {INPUTS_VERSION_ENV}=v1 deliberately; the engine will not quietly "
            "serve a different build from the one the run declared.")
    return tag, "v1"

#: Roster slots carried per team. `rotation.MAX_CANDIDATES` is 15 and a
#: TeamPrior is extended to `fit.n_profile` (15) slots, so 15 is exact rather
#: than a truncation.
N_SLOTS = 15


@dataclass
class FeaturePlan:
    """How to fill one sub-model's feature matrix for a batch of rows."""

    features: tuple[str, ...]
    team_src: np.ndarray      # (k,) int32 column index into the static team block
    team_dst: np.ndarray      # (k,) int32 column index into the output matrix
    slot_src: np.ndarray
    slot_dst: np.ndarray
    state_src: np.ndarray
    state_dst: np.ndarray

    @property
    def width(self) -> int:
        return len(self.features)


def plan_features(features: list[str] | tuple[str, ...],
                  team_names: dict[str, int],
                  slot_names: dict[str, int],
                  state_names: dict[str, int]) -> FeaturePlan:
    """Resolve a sub-model's declared feature order against the three blocks.

    A feature that resolves in none of them is a hard error: the engine must
    never hand a model a silently-zero column (`cbb_sim.control.simulate`'s
    preflight discipline, generalised)."""
    t_s, t_d, s_s, s_d, x_s, x_d = [], [], [], [], [], []
    missing = []
    for j, f in enumerate(features):
        if f in state_names:          # state wins: it is the live value
            x_s.append(state_names[f])
            x_d.append(j)
        elif f in slot_names:
            s_s.append(slot_names[f])
            s_d.append(j)
        elif f in team_names:
            t_s.append(team_names[f])
            t_d.append(j)
        else:
            missing.append(f)
    if missing:
        raise KeyError(
            f"engine inputs do not carry feature(s) {missing}; the sim would hand the "
            "model a column it never saw in training. Add it to build_engine_inputs.py "
            "or to the state block -- never default it to zero here.")
    i32 = lambda v: np.asarray(v, dtype=np.int32)  # noqa: E731
    return FeaturePlan(tuple(features), i32(t_s), i32(t_d), i32(s_s), i32(s_d),
                       i32(x_s), i32(x_d))


@dataclass
class EngineInputs:
    """Everything the loop needs, as dense arrays indexed by game and side.

    Side index 0 is ALWAYS home. Every `*_off` block is the OFFENCE's view of
    the matchup, so side `s` of `team_static` is the block to use when side `s`
    has the ball.
    """

    games: pd.DataFrame                 # G rows: game_id, season, teams, site, dates
    team_static: np.ndarray             # (G, 2, Ft) float32
    team_names: dict[str, int]
    slot_static: np.ndarray             # (G, 2, S, Fs) float32
    slot_names: dict[str, int]
    roster_cbbd: np.ndarray             # (G, 2, S) int64, negative = anonymous tail slot
    roster_espn: np.ndarray             # (G, 2, S) int64, -1 where the crosswalk misses
    roster_valid: np.ndarray            # (G, 2, S) bool, slot is a real candidate
    rot_share: np.ndarray               # (G, 2, S) float32 as-of minute share (target basis)
    rot_srank: np.ndarray               # (G, 2, S) int16 start-rank order (1-based)
    rot_start: np.ndarray               # (G, 2, S) int16 start priority, 0 = first
    rot_fpm: np.ndarray                 # (G, 2, S) float32 fouls per on-floor minute
    rot_pavail: np.ndarray              # (G, 2, S) float32 P(plays at all)
    usage_rate: np.ndarray              # (G, 2, S, 5) float32 shrunk as-of class rate
    usage_classes: tuple[str, ...]
    reb_rate: np.ndarray                # (G, 2, S, 2) float32 as-of (oreb, dreb) rate
    rules: dict                         # data-derived engine rule constants
    meta: dict                          # provenance and provisional flags

    @property
    def n_games(self) -> int:
        return len(self.games)

    @property
    def n_slots(self) -> int:
        return self.roster_cbbd.shape[2]

    # -- persistence ------------------------------------------------------
    def save(self, out_dir: Path | str, tag: str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.games.to_parquet(out / f"games_{tag}.parquet", index=False)
        np.savez_compressed(
            out / f"arrays_{tag}.npz",
            team_static=self.team_static, slot_static=self.slot_static,
            roster_cbbd=self.roster_cbbd, roster_espn=self.roster_espn,
            roster_valid=self.roster_valid, rot_share=self.rot_share,
            rot_srank=self.rot_srank, rot_start=self.rot_start, rot_fpm=self.rot_fpm,
            rot_pavail=self.rot_pavail, usage_rate=self.usage_rate, reb_rate=self.reb_rate,
        )
        (out / f"names_{tag}.json").write_text(json.dumps({
            "team_names": self.team_names, "slot_names": self.slot_names,
            "usage_classes": list(self.usage_classes),
            "rules": self.rules, "meta": self.meta,
        }, indent=1, default=str), encoding="utf-8")
        return out

    @classmethod
    def load(cls, in_dir: Path | str, tag: str, version: str | None = None) -> EngineInputs:
        d = Path(in_dir)
        disk_tag, loaded = resolve_tag(d, tag, version)
        games = pd.read_parquet(d / f"games_{disk_tag}.parquet")
        z = np.load(d / f"arrays_{disk_tag}.npz")
        names = json.loads((d / f"names_{disk_tag}.json").read_text(encoding="utf-8"))
        names.setdefault("meta", {})
        names["meta"]["inputs_version_loaded"] = loaded
        names["meta"]["inputs_tag_loaded"] = disk_tag
        tag = disk_tag
        return cls(
            games=games,
            team_static=z["team_static"], team_names={k: int(v) for k, v in names["team_names"].items()},
            slot_static=z["slot_static"], slot_names={k: int(v) for k, v in names["slot_names"].items()},
            roster_cbbd=z["roster_cbbd"], roster_espn=z["roster_espn"],
            roster_valid=z["roster_valid"], rot_share=z["rot_share"], rot_srank=z["rot_srank"],
            rot_start=z["rot_start"], rot_fpm=z["rot_fpm"], rot_pavail=z["rot_pavail"],
            usage_rate=z["usage_rate"], usage_classes=tuple(names["usage_classes"]),
            reb_rate=z["reb_rate"], rules=names["rules"], meta=names["meta"],
        )
