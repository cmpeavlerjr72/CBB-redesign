"""team_rate_adapter_v2.py -- `team_rate_adapter.apply` PLUS every column DERIVED from a swapped rate (lane D, 2026-10-01).

Versioned sibling of `cbb_sim.team_rate_adapter` (not edited). v1 swaps the mapped centred rates but leaves columns
computed FROM them at their served values. Audit of every sub-model the adapter serves (docs/ops/full_retrain_chain
section 3):

  * possession_outcome: the interaction columns `x_off_<r>_c__opp_def_<r>_c` -- v1 already recomputes them. Nothing to add.
  * fg_make: `off_make_raw` / `def_allow_raw` (= centred rate + `lg_make_asof`) stay at the served expanding means, and the
    trainer builds `fit_m` and `shooter_shrunk_dev_c` from `off_make_raw`. The engine inputs builder derives the shooter
    dev from the SWAPPED rate, so trained and served features disagree (corr 0.94-0.97; lane A,
    docs/tests/aggregation_overspread_decomposition_2026-09-30.md). v2 re-derives both raw columns as c + lg_make_asof,
    the rule the `--feature-table` path of `train_fg_make_v4_par_v1.py` already uses.
  * rebound: the round-3 design carries rate-derived extras (`rawc_off`, `off_oreb_g*`, `off_priorc`, `*_oa*_c`, ...) that
    v1 also leaves stale, but NO served arm reads them (the served A0B0C0 = lgbm / C_plus_state features are the two
    swapped rates plus non-rate columns). v2 does not invent a re-derivation for them; it REFUSES to be used with an arm
    whose feature list contains one (`assert_rebound_features`).

With the same arguments as v1 and a frame without those columns, v2 returns exactly v1's frame.
"""
from __future__ import annotations

import numpy as np

from cbb_sim import team_rate_adapter as V1

SUBMODELS = V1.SUBMODELS
#: captured at import, so a caller that rebinds `team_rate_adapter.apply` to this module's `apply` cannot recurse
_V1_APPLY = V1.apply
FG_RAW = (("off_make_c", "off_make_raw"), ("def_allow_c", "def_allow_raw"))
#: rebound design columns derived from the swapped OREB / DREB rates that v2 does NOT re-derive
RB_STALE_DERIVED = ("rawc_off", "rawc_def", "off_oreb_g1", "off_oreb_g2", "off_oreb_g3", "off_priorc", "off_Dprev",
                    "off_oreb_oa1_c", "off_oreb_oa2_c", "opp_def_dreb_g1", "opp_def_dreb_g2", "opp_def_dreb_g3",
                    "def_priorc", "opp_def_dreb_oa1_c", "opp_def_dreb_oa2_c")


def apply(frame, table_path, submodel, fold=None, missing="raise"):
    out = _V1_APPLY(frame, table_path, submodel, fold=fold, missing=missing)
    rederived = []
    if submodel == "fg_make":
        if "lg_make_asof" not in out.columns and any(r in out.columns for _, r in FG_RAW):
            raise KeyError("fg_make frame carries a raw make rate but no lg_make_asof to re-derive it from")
        for c, raw in FG_RAW:
            if c in out.columns and raw in out.columns:
                out[raw] = (out[c].to_numpy(dtype="float64")
                            + out["lg_make_asof"].to_numpy(dtype="float64")).astype(out[raw].dtype)
                rederived.append(raw)
    attrs = dict(out.attrs.get("team_rate_adapter", {}))
    attrs["v2_rederived"] = rederived
    attrs["v2_stale_unused"] = [c for c in RB_STALE_DERIVED if c in out.columns] if submodel == "rebound" else []
    out.attrs["team_rate_adapter"] = attrs
    return out


def assert_rebound_features(features) -> None:
    bad = [f for f in features if f in RB_STALE_DERIVED]
    if bad:
        raise ValueError(f"rebound arm reads rate-derived columns {bad} that the adapter does not re-derive")
