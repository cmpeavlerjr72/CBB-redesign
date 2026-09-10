"""
seal.py -- the 2025-26 season seal guard.

Standing rule (CLAUDE.md / docs/FRAMEWORK_PLAN.md section 1, item 8): season 2026
(hoopR/CBBD convention: season = ending year, so 2026 == the 2025-26 season, the
one with real closing lines) is the blind final test. It stays sealed -- untouched
by any training run, model-selection fold, or gate-target baseline used to pick a
model -- until a sub-model has been chosen on the fold-2 walk-forward holdout
(train through 2023-24, test 2024-25).

`assert_not_sealed(df_or_season)` is the single guard every training or
selection script must call before it consumes a table that might carry season
2026 rows. It is deliberately narrow: it does not stop season 2026 from being
*read* for pure data-audit / reference-table purposes (e.g. the per-season
descriptive gate-reference tables in scripts/build_gate_reference.py report
2026 numbers openly, same as the hoopR/CBBD audits already do) -- it stops
season 2026 from silently entering a *trainable* or *selection* artifact, such
as a fold's training window, a fitted model's training slice, or a bake-off
decision table.

Usage:
    from cbb_sim.data.seal import assert_not_sealed

    assert_not_sealed(2026)                       # raises SealedSeasonError
    assert_not_sealed([2022, 2023, 2024])          # ok
    assert_not_sealed(training_df)                 # raises if training_df has any season==2026 row
    assert_not_sealed(training_df, context="fold2 train slice")

Override (deliberate, logged): set the environment variable CBB_UNSEAL=1. This
is meant for the one moment in the timeline (FRAMEWORK_PLAN section 7, week 6)
where the season is unsealed on purpose after fold-2 selection is done -- not
for routine development convenience.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

SEALED_SEASON = 2026
UNSEAL_ENV_VAR = "CBB_UNSEAL"


class SealedSeasonError(RuntimeError):
    """Raised when a training/selection script touches sealed season 2026 data
    without an explicit CBB_UNSEAL=1 override."""


def _is_unsealed() -> bool:
    return os.environ.get(UNSEAL_ENV_VAR, "") == "1"


def assert_not_sealed(df_or_season, context: str = "") -> None:
    """Raise SealedSeasonError if `df_or_season` touches season 2026, unless
    CBB_UNSEAL=1 is set in the environment.

    Accepts:
      - an int/np.integer season value, e.g. 2026
      - an iterable of season ints/strings, e.g. [2022, 2023, 2024]
      - a pandas DataFrame with a 'season' column (checked for any row == 2026)

    `context` is an optional short string identifying the caller/output
    (e.g. "fold2 train slice", "gate_targets_fold2_train.parquet") included in
    the raised error message and safe to omit.
    """
    if _is_unsealed():
        return

    seasons_touched: set[int] = set()

    # pandas DataFrame (duck-typed to avoid a hard pandas import requirement here)
    if hasattr(df_or_season, "columns") and hasattr(df_or_season, "__getitem__"):
        if "season" in df_or_season.columns:
            try:
                seasons_touched = {int(s) for s in df_or_season["season"].dropna().unique()}
            except (TypeError, ValueError):
                seasons_touched = set()
    elif isinstance(df_or_season, (int,)):
        seasons_touched = {int(df_or_season)}
    elif isinstance(df_or_season, Iterable) and not isinstance(df_or_season, (str, bytes)):
        for s in df_or_season:
            try:
                seasons_touched.add(int(s))
            except (TypeError, ValueError):
                continue
    else:
        try:
            seasons_touched = {int(df_or_season)}
        except (TypeError, ValueError):
            seasons_touched = set()

    if SEALED_SEASON in seasons_touched:
        where = f" ({context})" if context else ""
        raise SealedSeasonError(
            f"season {SEALED_SEASON} (2025-26, the sealed blind final test) was touched by a "
            f"training/selection step{where}. Set {UNSEAL_ENV_VAR}=1 to override deliberately "
            f"once fold-2 model selection is complete (FRAMEWORK_PLAN.md section 7, week 6)."
        )
