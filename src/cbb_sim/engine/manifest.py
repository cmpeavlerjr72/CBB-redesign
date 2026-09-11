"""
manifest.py -- dated model artifacts, and which one a given game may use.

WHY THIS IS GENERIC AND NOT PART OF THE EVENT ADAPTER
-----------------------------------------------------
The possession-outcome round-2 bake-off adopted training scheme **S1** --
in-season monthly walk-forward refit -- and its pre-registration says the
winning scheme becomes the default for every later sub-model unless that
sub-model's own bake-off overrides it (`docs/models/possession_outcome/
experiments.md` section 4; `docs/models/change_ledger.md`, the TRAINING SCHEME
row; L21). So an S1 winner is not a special case of the event layer: clock,
rotation, usage, fg_make, free_throw and rebound are each expected to arrive as
a SCHEDULE of dated artifacts rather than as one fitted object.

A deployed simulator cannot run the bake-off's `fit_predict_scheme`: it has no
test frame to partition. What it needs, per sub-model, is a manifest of dated
artifacts and one rule for choosing between them. That rule is here, once, so
that seven adapters cannot implement it seven ways -- and so the honest-backtest
property is enforced in ONE place that every sub-model passes through.

THE SELECTION RULE
------------------
For a game tipping at time `t`, use the artifact with the LATEST `refit_date`
that is STRICTLY BEFORE `t`. Refit dates are calendar dates read as midnight at
the start of their day, so a game on the first of the month uses THAT month's
refit -- which was fitted only on games strictly before the first, and whose
midnight timestamp is strictly before an evening tipoff -- and never the next
one. This is also exactly the partition the bake-off scored, so the calibration
measured there is the calibration the engine gets.

TWO INDEPENDENT SAFETY CHECKS, BOTH ENFORCED AT LOAD
----------------------------------------------------
1. `refit_date < tipoff` -- the selection rule itself.
2. `max_train_date < game_date` -- the property that actually matters, checked
   against what each artifact says it was TRAINED on rather than against when
   it was scheduled. An artifact whose manifest does not declare
   `max_train_date` cannot be checked and is REJECTED unless the caller passes
   `require_max_train_date=False`, because `CLAUDE.md`'s honest-backtest rule
   ("every backtest row must satisfy created_at < tipoff, enforced in code")
   is not satisfied by a schedule that merely looks right.

A STATIC MODEL IS A MANIFEST OF LENGTH ONE
------------------------------------------
A sub-model whose bake-off adopted a single undated artifact is served by
`ArtifactManifest.static(...)`: every game maps to entry 0 and `is_static` is
True, which the engine writes into `run_meta.json` as
`scheme_static_<model>=True`. That flag exists so a reader can tell a model
that is deliberately static from one that is silently serving a stale S1
artifact -- the failure mode the change ledger calls out ("a stale artifact
degrades toward S0, which fails the gate").

MANIFEST FORMAT (JSON)
----------------------
    {
      "model": "possession_outcome",   # sub-model name; names the run_meta flag
      "scheme": "S1",                  # "S1" | "static" | any scheme label
      "fold": "F2",
      "season": 2025,
      "key": "first",                  # OPTIONAL: population / shot class / target
      "artifacts": [
        {"refit_date": "2024-11-01",   # required, ISO date
         "path": "first_2024-11-01.joblib",   # relative to the manifest's dir
         "max_train_date": "2024-04-08",      # required unless waived
         "n_train": 1908534},                 # optional, provenance only
        ...
      ]
    }

`artifacts` need not be sorted; this module sorts and rejects duplicates.
A model with several keys (populations, shot classes) writes one manifest per
key, or one file whose top level maps key -> the object above.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArtifactEntry:
    refit_date: pd.Timestamp
    path: Path
    max_train_date: pd.Timestamp | None
    meta: dict = field(default_factory=dict)


@dataclass
class ArtifactManifest:
    """A sub-model's dated artifacts plus the per-game choice among them."""

    model: str
    scheme: str
    entries: tuple[ArtifactEntry, ...]
    seg_of_game: np.ndarray          # (G,) int index into `entries`
    key: str | None = None
    source: dict = field(default_factory=dict)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_obj(cls, obj: dict, base: Path, games: pd.DataFrame,
                 require_max_train_date: bool = True) -> ArtifactManifest:
        arts = obj.get("artifacts") or []
        if not arts:
            raise ValueError(f"manifest for {obj.get('model')!r} lists no artifacts")
        entries = []
        for a in arts:
            mt = a.get("max_train_date")
            if mt is None and require_max_train_date:
                raise ValueError(
                    f"artifact {a.get('path')!r} of {obj.get('model')!r} declares no "
                    "max_train_date, so its training window cannot be checked against "
                    "tipoff; declare it or pass require_max_train_date=False and say why")
            entries.append(ArtifactEntry(
                refit_date=pd.Timestamp(a["refit_date"]),
                path=base / a["path"],
                max_train_date=None if mt is None else pd.Timestamp(mt),
                meta={k: v for k, v in a.items()
                      if k not in ("refit_date", "path", "max_train_date")}))
        entries.sort(key=lambda e: e.refit_date)
        dates = [e.refit_date for e in entries]
        if len(set(dates)) != len(dates):
            raise ValueError(f"{obj.get('model')!r} manifest repeats a refit_date")
        for e in entries:
            if not e.path.exists():
                raise FileNotFoundError(f"manifest names a missing artifact: {e.path}")

        seg = _select(entries, games, model=obj.get("model", "?"),
                      require_max_train_date=require_max_train_date)
        return cls(model=obj.get("model", "?"), scheme=obj.get("scheme", "S1"),
                   entries=tuple(entries), seg_of_game=seg, key=obj.get("key"),
                   source={"fold": obj.get("fold"), "season": obj.get("season"),
                           "n_artifacts": len(entries),
                           "refit_dates": [str(d.date()) for d in dates],
                           "max_train_dates": [None if e.max_train_date is None
                                               else str(e.max_train_date.date())
                                               for e in entries]})

    @classmethod
    def from_json(cls, path: Path | str, games: pd.DataFrame, key: str | None = None,
                  require_max_train_date: bool = True) -> ArtifactManifest:
        p = Path(path)
        doc = json.loads(p.read_text(encoding="utf-8"))
        obj = doc if "artifacts" in doc else doc[key]
        if key is not None:
            obj = {**obj, "key": key, "model": obj.get("model", doc.get("model", "?")),
                   "scheme": obj.get("scheme", doc.get("scheme", "S1")),
                   "fold": obj.get("fold", doc.get("fold")),
                   "season": obj.get("season", doc.get("season"))}
        return cls.from_obj(obj, p.parent, games, require_max_train_date)

    @classmethod
    def static(cls, model: str, path: Path | str, n_games: int,
               key: str | None = None, why: str = "") -> ArtifactManifest:
        """A single undated artifact used for every game.

        `is_static` is True, which the engine reports as
        `scheme_static_<model>=True`."""
        return cls(model=model, scheme="static",
                   entries=(ArtifactEntry(pd.Timestamp.min, Path(path), None,
                                          {"why": why}),),
                   seg_of_game=np.zeros(int(n_games), dtype=np.int64),
                   key=key, source={"n_artifacts": 1, "static": True, "why": why})

    # -- use --------------------------------------------------------------
    @property
    def is_static(self) -> bool:
        return len(self.entries) == 1 and self.scheme == "static"

    @property
    def flag_name(self) -> str:
        return f"scheme_static_{self.model}" + (f"_{self.key}" if self.key else "")

    def segments(self, gidx: np.ndarray) -> np.ndarray:
        """The artifact index for each row's game."""
        return self.seg_of_game[gidx]

    def used_segments(self, gidx: np.ndarray) -> np.ndarray:
        return np.unique(self.seg_of_game[gidx])

    def provenance(self) -> dict:
        return {"model": self.model, "key": self.key, "scheme": self.scheme,
                "is_static": self.is_static, **self.source}


def _select(entries: list[ArtifactEntry], games: pd.DataFrame, model: str,
            require_max_train_date: bool) -> np.ndarray:
    """Per game, the index of the latest artifact whose refit_date is STRICTLY
    before that game's tipoff -- then both safety checks."""
    tip = _tipoff(games)
    cuts = np.array([np.datetime64(e.refit_date.tz_localize(None)
                                   if e.refit_date.tzinfo else e.refit_date, "ns")
                     for e in entries])
    t = tip.to_numpy()
    # STRICTLY BEFORE TIPOFF, at date granularity. `t` is midnight at the start
    # of the game's own day and a refit date is midnight of ITS day, so a refit
    # dated the SAME day as the game is dated midnight and the game tips later
    # that day: it is strictly before tipoff and is therefore eligible. Hence
    # `side="right"` (cut <= t), not `side="left"` (cut < t).
    #
    # This also matches what the bake-off measured, which is the real
    # constraint. `possession_outcome.fit_predict_scheme`'s S1 assigns a test
    # game to a segment by `game_date >= cut`, and that segment's fit used only
    # `game_date < cut` -- so a game ON the first of the month is scored by
    # that month's refit, trained strictly before the first. Selecting the
    # PREVIOUS month's artifact there would be safe but would serve a model the
    # bake-off never scored, and its measured calibration would not apply.
    seg = np.searchsorted(cuts, t, side="right") - 1
    if (seg < 0).any():
        i = int(np.flatnonzero(seg < 0)[0])
        raise ValueError(
            f"{model}: game {games.iloc[i].get('game_id')} tips {pd.Timestamp(t[i])}, "
            f"before the earliest artifact refit date {entries[0].refit_date.date()}; "
            "the manifest does not cover this season's start")

    if require_max_train_date:
        mt = np.array([np.datetime64(e.max_train_date, "ns") for e in entries])
        gd = _game_date(games).to_numpy()
        bad = mt[seg] >= gd
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise AssertionError(
                f"{model}: game {games.iloc[i].get('game_id')} on {pd.Timestamp(gd[i]).date()} "
                f"would be scored by an artifact trained through "
                f"{pd.Timestamp(mt[seg][i]).date()} -- that is a leak "
                "(CLAUDE.md: every backtest row must satisfy created_at < tipoff)")
    return seg.astype(np.int64)


def _tipoff(games: pd.DataFrame) -> pd.Series:
    """The instant a refit date must be strictly earlier than.

    THIS IS `game_date` AT MIDNIGHT, NOT `tipoff_utc`, AND THE REASON IS A BUG
    THIS FUNCTION ALREADY CAUGHT ONCE. A refit date is a CALENDAR date on the
    same local convention the training frame's `game_date` uses. `tipoff_utc`
    is on a different clock: a game tipping the evening of 31 March US time
    carries `tipoff_utc` of 1 April, so comparing the two selected the April
    refit -- an artifact trained on games through 31 March -- for a 31 March
    game. Mixing the two clocks silently converts a correct schedule into a
    leak for every late-evening game in the last week of a month.

    Comparing midnight-of-`game_date` against the refit date is the same
    question asked on one clock, and it is the STRICTER of the two readings: a
    refit dated the same calendar day as the game is excluded, which is what
    "fitted only on games strictly before this date" means. `tipoff_utc` is
    still used by the engine elsewhere; it is simply not the right key for a
    date-scheduled artifact."""
    return _game_date(games)


def _game_date(games: pd.DataFrame) -> pd.Series:
    for c in ("game_date", "date"):
        if c in games:
            return pd.to_datetime(games[c]).dt.normalize()
    raise KeyError("games frame carries neither `game_date` nor `date`")
