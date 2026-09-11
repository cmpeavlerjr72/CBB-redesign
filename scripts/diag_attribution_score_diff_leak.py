#!/usr/bin/env python
"""
diag_attribution_score_diff_leak.py -- is attribution's `score_diff` the same
post-outcome leak L27 found in fg_make, and if so, how much of its apparent
effect does it manufacture?

    .venv/Scripts/python.exe scripts/diag_attribution_score_diff_leak.py

`docs/models/change_ledger.md`'s "flagged for that model's owner" row
(2026-09-10) named the suspect: `attribution.py`'s `_state_block` (line ~534
before this round's fix) built `score_diff` from the candidate's own-row
`homeScore`/`awayScore`, the same construction fg_make's round-2 bake-off
proved was POST-play (L27). This script proves, independently of the fix
already applied to `attribution.py` in this round, WHICH of the eight targets
that column actually reaches, using the identical own-row delta test L27 used
on fg_make: how often the candidate side's own score on its own row moves by
exactly the event's point value (leak) vs by zero (clean).

PART 0 -- the own-row delta test, off the raw feed, for all FIVE populations.
PART 1 -- the "manufactured pp" bucket-span read for the THREE binaries
          (`assisted`/`stolen`/`blocked`): the outcome rate by score_diff
          decile, LEAKED vs PRE-PLAY, exactly fg_make's Part 0b.
PART 2 -- the conditional-logit's own score_diff-interaction coefficients for
          the FIVE choice targets, LEAKED vs PRE-PLAY, fit once each on the F1
          training slice (2024) -- the model's own read of "how much of the
          apparent effect vanishes."

Output: `data/processed/models/attribution/score_diff_leak_2026-09-10.json`;
the markdown built from it is
`docs/tests/attribution_score_diff_leak_2026-09-10.md`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbb_sim.models import attribution as A  # noqa: E402
from cbb_sim.models import event_stream as ES  # noqa: E402

OUT_JSON = Path("data/processed/models/attribution/score_diff_leak_2026-09-10.json")
SEASONS = [2024, 2025]
RIM = ES.rim_override_for_version("v2")
DECILE_LABELS = [f"D{i}" for i in range(1, 11)]

POP_OF_MASK = {
    "reb_off": ("REB_off",), "reb_def": ("REB_def",), "made_fga": ("assist", "assisted"),
    "tov": ("steal", "stolen"), "miss_fga": ("block", "blocked"),
}
CHOICE_TARGET_OF_POP = {"reb_off": "REB_off", "reb_def": "REB_def",
                        "made_fga": "assist", "tov": "steal", "miss_fga": "block"}
BINARY_TARGET_OF_POP = {"made_fga": "assisted", "tov": "stolen", "miss_fga": "blocked"}


# ---------------------------------------------------------------------------
# PART 0: the own-row delta test, off the raw feed (independent of the fix)
# ---------------------------------------------------------------------------
def own_row_delta_test(st: pd.DataFrame) -> dict:
    """For every row of the (already-cleaned) attribution stream: how much did
    the row's OWN side's score move on the row's own record, relative to the
    row immediately before it in the same game? A rebound, a turnover and a
    missed/blocked attempt never change the score by construction; a made
    field goal's own row already carries its own points. This is L27's test
    ("the shooting team's own-row score moving by exactly the shot's value"),
    generalised to every population `build_attr_events` constructs, and it does
    not call `_state_block` or any other attribution-model code -- only the
    stream's raw `home_score` / `away_score` / `side` / `cls` / `made`."""
    g = st["cbbd_game_id"].to_numpy()
    hs = st["home_score"].to_numpy(dtype="float64")
    as_ = st["away_score"].to_numpy(dtype="float64")
    same = np.concatenate([[False], g[1:] == g[:-1]])
    dh = np.where(same, np.concatenate([[0.0], np.diff(hs)]), 0.0)
    da = np.where(same, np.concatenate([[0.0], np.diff(as_)]), 0.0)
    side = st["side"].to_numpy()
    is_home_row = side == 0
    # the ROW'S OWN team's score delta (not yet the candidate side -- that
    # differs by population below; a rebound/tov/miss row's team is `side`)
    own_delta_row_team = np.where(is_home_row, dh, da)

    cls = st["cls"].to_numpy(dtype=object)
    made = st["made"].to_numpy().astype(bool)
    pts = np.where(cls == "FGA_3", 3.0, 2.0)
    made_fga = np.isin(cls, ES.FGA_CLASSES) & made
    expected = np.where(made_fga, pts, 0.0)

    masks = {
        "reb_off": cls == "OREB",
        "reb_def": cls == "DREB",
        "made_fga": made_fga,
        "tov": cls == "TOV",
        "miss_fga": np.isin(cls, ES.FGA_CLASSES) & ~made,
    }
    out = {}
    for pop, m in masks.items():
        d, e = own_delta_row_team[m], expected[m]
        n = int(m.sum())
        out[pop] = {
            "n": n,
            "mean_own_row_delta": round(float(d.mean()), 4) if n else None,
            "pct_delta_equals_expected_value": round(float((d == e).mean() * 100), 3) if n else None,
            "pct_delta_zero": round(float((d == 0).mean() * 100), 3) if n else None,
        }
    return out


# ---------------------------------------------------------------------------
# PART 1: binaries -- outcome-rate bucket span, leaked vs pre-play
# ---------------------------------------------------------------------------
def decile_span(score_diff: np.ndarray, y: np.ndarray) -> tuple[dict, float]:
    try:
        buckets = pd.qcut(score_diff, 10, labels=DECILE_LABELS, duplicates="drop")
    except ValueError:
        buckets = pd.qcut(score_diff, 5, duplicates="drop")
    rates = pd.Series(y).groupby(buckets, observed=True).mean() * 100.0
    d = {str(k): round(float(v), 3) for k, v in rates.items()}
    return d, float(rates.max() - rates.min())


def binary_bucket_report(pops_leaked: dict, pops_pre: dict) -> dict:
    out = {}
    for pop, target in BINARY_TARGET_OF_POP.items():
        pl, pp = pops_leaked[pop], pops_pre[pop]
        assert len(pl) == len(pp), "leaked/pre-play population tables must be row-aligned"
        y = pl["b"].to_numpy()
        curve_l, span_l = decile_span(pl["score_diff"].to_numpy(dtype="float64"), y)
        curve_p, span_p = decile_span(pp["score_diff"].to_numpy(dtype="float64"), y)
        # rows where the fix actually changed the value (should be ~100%: every
        # row of these three populations' event IS the scoring/turnover/miss
        # row itself only for made_fga; tov/miss_fga should show 0 changed)
        changed_pct = round(float(
            (pl["score_diff"].to_numpy() != pp["score_diff"].to_numpy()).mean() * 100), 3)
        out[target] = {
            "n": int(len(pl)), "base_rate_pct": round(float(y.mean() * 100), 4),
            "pct_rows_score_diff_changed_by_fix": changed_pct,
            "leaked_decile_curve_pct": curve_l, "leaked_span_pp": round(span_l, 3),
            "pre_play_decile_curve_pct": curve_p, "pre_play_span_pp": round(span_p, 3),
            "manufactured_pp": round(span_l - span_p, 3),
            "manufactured_share_pct": (round((span_l - span_p) / span_l * 100, 1)
                                       if span_l > 1e-9 else None),
        }
    return out


# ---------------------------------------------------------------------------
# PART 2: choice targets -- the P2 conditional logit's own score_diff terms
# ---------------------------------------------------------------------------
def choice_coefficient_report(pooled_leaked: dict, pooled_pre: dict,
                              asof: pd.DataFrame) -> dict:
    """The P2 conditional logit's OWN read of the score_diff interaction terms,
    fit on the real pipeline (`usable` -> `build_choice_design` -> `cl_design`
    -> `CondLogitArm`) on the F1 TRAINING slice (season 2024), leaked vs
    pre-play. `l2=1.0` and `(prior_kind='position', m=25)` are fixed rather than
    grid-searched -- this is an effect-size read, not a bake-off arm."""
    out = {}
    for pop, target in CHOICE_TARGET_OF_POP.items():
        rows = {}
        for mode, pops in (("leaked", pooled_leaked), ("pre_play", pooled_pre)):
            ev = A.usable(pops[pop], target)
            d = A.build_choice_design(ev, asof, target)
            tr, _ = A.fold_slices(d)
            if not len(tr):
                rows[mode] = {"n": 0}
                continue
            unid = A.unidentified_features(tr, target)
            X, names = A.cl_design(tr, target, "position", 25.0)
            keep = [i for i, n in enumerate(names) if n not in unid]
            X, names = X[:, :, keep], [names[i] for i in keep]
            X, names, const = A.drop_constant_features(X, names)
            arm = A.CondLogitArm(l2=1.0).fit(X, tr["y"].to_numpy())
            coef = dict(zip(names, [round(float(b), 5) for b in arm.beta_], strict=False))
            rows[mode] = {"n": int(len(tr)), "dropped_unidentified": unid,
                          "dropped_constant": const, "coefficients": coef}
        l, p = rows.get("leaked", {}), rows.get("pre_play", {})
        delta = {}
        for term in ("log_share_x_scorediff", "is_C_x_scorediff"):
            bl = l.get("coefficients", {}).get(term)
            bp = p.get("coefficients", {}).get(term)
            if bl is not None and bp is not None:
                delta[term] = {"leaked": bl, "pre_play": bp,
                               "abs_shrinkage_pct": (round((abs(bl) - abs(bp)) / abs(bl) * 100, 1)
                                                     if abs(bl) > 1e-9 else None)}
        out[target] = {"leaked": l, "pre_play": p, "scorediff_term_delta": delta}
    return out


def main() -> int:
    t0 = time.time()
    universe = ES.load_universe(require_pbp_complete=True)
    rep: dict = {"created_at": pd.Timestamp.now("UTC").isoformat(), "seasons": SEASONS,
                "rim_override_max_ft": RIM}

    streams, pops_leaked, pops_pre, own_row = {}, {}, {}, {}
    for s in SEASONS:
        print(f"building stream {s}...", flush=True)
        streams[s] = A.build_attr_stream(s, universe, rim_override_max_ft=RIM)
        own_row[s] = own_row_delta_test(streams[s])
        print(f"  {s} own-row delta: " + json.dumps(own_row[s]), flush=True)
        pops_leaked[s] = A.build_attr_events(s, universe, rim_override_max_ft=RIM,
                                             stream=streams[s], score_diff_mode="leaked")
        pops_pre[s] = A.build_attr_events(s, universe, rim_override_max_ft=RIM,
                                          stream=streams[s], score_diff_mode="pre_play")
    rep["part0_own_row_delta_test"] = own_row

    pooled_leaked = {pop: pd.concat([pops_leaked[s][pop] for s in SEASONS], ignore_index=True)
                     for pop in ("reb_off", "reb_def", "made_fga", "tov", "miss_fga")}
    pooled_pre = {pop: pd.concat([pops_pre[s][pop] for s in SEASONS], ignore_index=True)
                  for pop in ("reb_off", "reb_def", "made_fga", "tov", "miss_fga")}

    print("\nPART 1: binary outcome-rate bucket span, leaked vs pre-play", flush=True)
    rep["part1_binary_bucket_span"] = binary_bucket_report(pooled_leaked, pooled_pre)
    for t, v in rep["part1_binary_bucket_span"].items():
        print(f"  {t}: rows changed by fix {v['pct_rows_score_diff_changed_by_fix']}%, "
              f"span leaked {v['leaked_span_pp']} pp -> pre-play {v['pre_play_span_pp']} pp "
              f"(manufactured {v['manufactured_pp']} pp, "
              f"{v['manufactured_share_pct']}%)", flush=True)

    print("\nPART 2: choice targets, P2 score_diff-interaction coefficients "
          "(2024 train)", flush=True)
    asof = pd.read_parquet("data/processed/models/attribution/asof_v2.parquet")
    rep["part2_choice_coefficients"] = choice_coefficient_report(pooled_leaked, pooled_pre, asof)
    for t, v in rep["part2_choice_coefficients"].items():
        print(f"  {t}: {json.dumps(v['scorediff_term_delta'])}", flush=True)

    rep["runtime_s"] = round(time.time() - t0, 1)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON} ({rep['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
