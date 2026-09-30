"""team_rate_adapter.py -- substitute E3 team-rate features into a sub-model's frame (Stage B).

The mapping below is the one in docs/models/team_rate_estimator/experiments.md section 6.5 and
section 7. It swaps the served as-of team-rate features for the E3 estimator's values, and it works
on training frames and inference frames alike.

    apply(frame, table_path, submodel, fold=None, missing="raise") -> frame

  * `submodel` is one of "possession_outcome", "fg_make", "rebound".
  * Exactly the mapped columns are replaced; every other column is untouched.
  * Row count and row order are preserved (asserted), and no NaN is introduced (asserted).
  * Key coverage is asserted: every (season, game_id, team) in the frame must have a row in the
    table. With the default missing="raise" an uncovered key raises. With missing="keep_served",
    an uncovered row keeps its served value and the count is reported in `frame.attrs`. That is
    an explicit, logged choice for the caller; it never happens silently.
  * One exception to "untouched": possession_outcome's interaction columns
    `x_off_<r>_c__opp_def_<r>_c` are recomputed as the product of the two replaced factors. Leaving
    the served product next to replaced factors would feed the model an inconsistent pair. The
    rating interaction `x_off_rating_off_c__def_rating_def_c` is untouched, because its factors are.

Scales: possession_outcome's style rates are x100 (`possession_outcome.RATE_SCALE`); fg_make and
rebound use 0-1.

The table (`data/processed/team_rate_features_<arm>_v<k>.parquet`) is keyed on
(fold, season, game_id, team_id) and has `<rate>_<side>_c` columns. A table with more than one fold
needs `fold=` ("F1" / "F2"): the trainer for fold k reads the rows fitted under fold k's
parameters.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUBMODELS = ("possession_outcome", "fg_make", "rebound")

#: possession_outcome: served column -> (table column, which team's row, scale)
PO_MAP: dict[str, tuple[str, str, float]] = {
    "off_tov_c": ("tov_off_c", "off", 100.0),
    "opp_def_tov_c": ("tov_def_c", "def", 100.0),
    "off_ftr_c": ("ftr_off_c", "off", 100.0),
    "opp_def_ftr_c": ("ftr_def_c", "def", 100.0),
    "off_rim_c": ("share_rim_off_c", "off", 100.0),
    "opp_def_rim_c": ("share_rim_def_c", "def", 100.0),
    "off_3pa_c": ("pa3_off_c", "off", 100.0),          # 3PA per possession; table v2 onwards
    "opp_def_3pa_c": ("pa3_def_c", "def", 100.0),
}
PO_INTERACTIONS = {f"x_off_{r}_c__opp_def_{r}_c": (f"off_{r}_c", f"opp_def_{r}_c")
                   for r in ("3pa", "rim", "tov", "ftr")}

#: fg_make: one served column per side whose meaning depends on the row's shot class
FG_CLASS_RATE = {"FGA_rim": "make_rim", "FGA_jump2": "make_jump", "FGA_3": "make3"}
FG_MAP = {"off_make_c": ("off", "off"), "def_allow_c": ("def", "def")}   # column -> (table side, team row)

#: rebound: served column -> (table column, which team's row, sign)
RB_MAP: dict[str, tuple[str, str, float]] = {
    "off_oreb_c": ("oreb_off_c", "off", 1.0),
    "opp_def_dreb_c": ("oreb_def_c", "def", -1.0),   # DREB% - league = -(OREB% allowed - league)
}

KEYS = {  # submodel -> (offence team column, defence team column)
    "possession_outcome": ("offense_team_id", "defense_team_id"),
    "fg_make": ("off_team_id", "def_team_id"),
    "rebound": ("off_team_id", "def_team_id"),
}


def load_table(table_path, fold: str | None) -> pd.DataFrame:
    t = pd.read_parquet(table_path)
    if "fold" in t.columns:
        folds = sorted(t["fold"].unique())
        if len(folds) > 1:
            if fold is None:
                raise ValueError(f"table has folds {folds}; pass fold=")
            t = t[t["fold"] == fold]
        t = t.drop(columns="fold")
    if t.duplicated(["season", "game_id", "team_id"]).any():
        raise ValueError("table keys (season, game_id, team_id) are not unique")
    return t


def _lookup(frame: pd.DataFrame, t: pd.DataFrame, team_col: str, cols: list[str]) -> pd.DataFrame:
    key = frame[["season", "game_id", team_col]].rename(columns={team_col: "team_id"})
    key = key.astype({"season": "int64", "game_id": "int64", "team_id": "int64"})
    tt = t[["season", "game_id", "team_id"] + cols].astype({"season": "int64", "game_id": "int64", "team_id": "int64"})
    out = key.merge(tt, on=["season", "game_id", "team_id"], how="left")
    assert len(out) == len(frame)
    out.index = frame.index
    return out[cols]


def apply(frame: pd.DataFrame, table_path, submodel: str, fold: str | None = None,
          missing: str = "raise") -> pd.DataFrame:
    if submodel not in SUBMODELS:
        raise KeyError(f"submodel must be one of {SUBMODELS}")
    if missing not in ("raise", "keep_served"):
        raise ValueError("missing must be 'raise' or 'keep_served'")
    t = load_table(table_path, fold)
    off_col, def_col = KEYS[submodel]
    n0 = len(frame)
    out = frame.copy()
    new: dict[str, pd.Series] = {}
    if submodel == "possession_outcome":
        need = [v[0] for v in PO_MAP.values()]
        miss = [c for c in need if c not in t.columns]
        if miss:
            raise KeyError(f"table lacks {miss} (off_3pa_c needs table v2 or later)")
        lk = {"off": _lookup(frame, t, off_col, [v[0] for v in PO_MAP.values() if v[1] == "off"]),
              "def": _lookup(frame, t, def_col, [v[0] for v in PO_MAP.values() if v[1] == "def"])}
        for served, (tc, who, scale) in PO_MAP.items():
            if served in frame.columns:
                new[served] = lk[who][tc] * scale
    elif submodel == "fg_make":
        if "shot_class" not in frame.columns:
            raise KeyError("fg_make frame needs shot_class")
        cls = frame["shot_class"].astype(str)
        unknown = set(cls.unique()) - set(FG_CLASS_RATE)
        if unknown:
            raise KeyError(f"unknown shot classes {unknown}")
        for served, (tside, who) in FG_MAP.items():
            if served not in frame.columns:
                continue
            cols = [f"{r}_{tside}_c" for r in FG_CLASS_RATE.values()]
            lk = _lookup(frame, t, off_col if who == "off" else def_col, cols)
            v = pd.Series(np.nan, index=frame.index)
            for c, r in FG_CLASS_RATE.items():
                m = (cls == c).to_numpy()
                v[m] = lk.loc[m, f"{r}_{tside}_c"]
            new[served] = v
    else:
        lk = {"off": _lookup(frame, t, off_col, ["oreb_off_c"]), "def": _lookup(frame, t, def_col, ["oreb_def_c"])}
        for served, (tc, who, sign) in RB_MAP.items():
            if served in frame.columns:
                new[served] = sign * lk[who][tc]
    if not new:
        raise KeyError(f"frame carries none of the mapped {submodel} columns")
    n_missing = 0
    for served, v in new.items():
        bad = v.isna().to_numpy()
        if bad.any():
            if missing == "raise":
                ex = frame.loc[bad, ["season", "game_id"]].drop_duplicates().head(5).to_dict("records")
                raise AssertionError(f"{int(bad.sum())} rows of {served} have no table key (e.g. {ex}); "
                                     "pass missing='keep_served' to keep their served value, explicitly")
            v = v.where(~bad, frame[served])
            n_missing = max(n_missing, int(bad.sum()))
        out[served] = v.astype(frame[served].dtype)
    if submodel == "possession_outcome":
        for x, (a, b) in PO_INTERACTIONS.items():
            if x in out.columns and a in new and b in new:
                out[x] = (out[a] * out[b]).astype(frame[x].dtype)
    assert len(out) == n0 and out.index.equals(frame.index), "row count/order changed"
    for c in new:
        assert not out[c].isna().any() or frame[c].isna().any(), f"NaN introduced in {c}"
    out.attrs["team_rate_adapter"] = {"table": str(Path(table_path)), "submodel": submodel, "fold": fold,
                                      "replaced": sorted(new), "rows_kept_served": n_missing}
    return out
