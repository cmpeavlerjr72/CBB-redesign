"""Default-off start-type label after an and-one (clock round 6, arm L1).

`docs/models/clock/experiments.md` section 28 and
`docs/tests/g1_g5_possessions_corr_diagnostic_2026-09-30.md` section 5.

The pbp possession layer the clock is trained on labels the possession that
follows an and-one `made_FG` whether the single free throw was made or missed.
`loop.py` labels it `made_FT` (FT made) or by the rebound outcome (FT missed).
With `ENGINE_ANDONE_LABEL=training` the engine uses the training table's
convention instead: when a chance that carried an and-one ends the possession
(no offensive-rebound continuation), the next possession's `prev_end` is
`made_FG`.  The default `engine` leaves every code path and every RNG draw
exactly as before (no draw is added or removed in either mode; only an integer
label changes).
"""
from __future__ import annotations

import os

import numpy as np

FLAG = "ENGINE_ANDONE_LABEL"
MODES = ("engine", "training")


def mode() -> str:
    m = os.environ.get(FLAG, "engine")
    if m not in MODES:
        raise ValueError(f"{FLAG}={m!r}; known modes: {MODES}")
    return m


def active() -> bool:
    return mode() == "training"


def relabel(end_code: np.ndarray, andone_rows: np.ndarray, cont: np.ndarray, made_fg_code: int) -> None:
    """In place: step-space rows whose chance carried an and-one and whose
    possession ends on this chance get `made_fg_code`."""
    if len(andone_rows) == 0:
        return
    r = andone_rows[~cont[andone_rows]]
    end_code[r] = made_fg_code
