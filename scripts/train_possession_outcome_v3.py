#!/usr/bin/env python
"""
train_possession_outcome_v3.py -- ROUND 3 of the pre-registered L3
POSSESSION-OUTCOME bake-off: REFIT ALIGNMENT and OPPONENT ADJUSTMENT
(ARCHITECTURE_DECISIONS.md Decision 9).

    .venv/Scripts/python.exe scripts/train_possession_outcome_v3.py
    ... --stages 1,2,3,4      # subset of the pre-registered stage order
    ... --budget-hours 6.5    # hard wall clock on the tree stages
    ... --no-append           # do not touch experiments.md

Pre-registration (written to `docs/models/possession_outcome/experiments.md` and
COMMITTED before this script ran): section "Round 3: refit alignment and
opponent adjustment (Decision 9)".

WHAT CHANGES FROM ROUND 2, and nothing else does:

  (i)   the SCHEME dimension gains ALIGNMENT: S1 is no longer only the calendar
        month. Arms: `S1_monthly` (round 2's, the reference), `S1_weekly`,
        `S1_conf_aligned`, `S1_conf_aligned_weekly`.
  (ii)  the FEATURE dimension gains the conference flag and opponent-adjusted
        style rates: F0 (round 2's features, the reference), F1 = F0 + the
        conference-game flag, F2 = F1 with one-pass opponent-adjusted style
        rates, F3 = F1 with the rates adjusted by alternating least squares to
        convergence (`cbb_sim.features.opponent_adjust`).

Model class is HELD FIXED at round 2's winners -- `lgbm` on first chances,
`cascade` on continuation chances. The event layer, the universe, the folds and
the seal are round 2's, untouched.

GRADING IS STILL LITERALLY UNCHANGED: `score()` is imported from
`train_possession_outcome_v1` and called, exactly as round 2 did, so all three
rounds go through one scoring function.

RUNTIME IS A FIRST-CLASS CONSTRAINT AND IT SHAPES THE DESIGN. Four workers share
this machine, so this run is capped at four threads. A tree fit on the fold-2
`first` training slice costs minutes at that cap, and the alignment arms multiply
FITS, not rows: 6 refits a season for monthly, ~23 for weekly, ~29 for
conference-aligned, ~40 for both. A full 4 x 4 x 2-fold cross on the tree arm is
several hundred fits and is not affordable. The pre-registration therefore states
a STAGED design with a fixed stage order and a hard wall clock, and this script
executes exactly that order; anything the clock does not reach is written to the
results as NOT RUN and never as a result.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import train_possession_outcome_v1 as R1  # noqa: E402

from cbb_sim.analysis import leak_test as LT  # noqa: E402
from cbb_sim.features import conference as CF  # noqa: E402
from cbb_sim.features import opponent_adjust as OA  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

OUT_DIR = _ROOT / "data/processed/models/possession_outcome/round3"
EXPERIMENTS_MD = _ROOT / "docs/models/possession_outcome/experiments.md"
SEASONS = [2022, 2023, 2024, 2025]
POSSESSIONS_VERSION = "v2"
STYLE_SOURCE = "first_chance"
#: Round 2's winners, held fixed. One arm per population.
POP_ARM = {"first": "lgbm", "cont": "cascade"}
N_JOBS = 4

# ---------------------------------------------------------------------------
# 1. Features
# ---------------------------------------------------------------------------
CONF_FEATURE = "is_conf_game"
#: suffix of the adjusted style columns per method
ADJ_SUFFIX = {"one_pass": "a1", "iterative": "a3"}
FEATURE_ARMS: tuple[str, ...] = ("F0", "F1", "F2", "F3")
#: Complexity order for "ties go to the simpler arm".
FEATURE_SIMPLICITY = {"F0": 0, "F1": 1, "F2": 2, "F3": 3}
SCHEME_SIMPLICITY = {"S1_monthly": 0, "S1_conf_aligned": 1,
                     "S1_weekly": 2, "S1_conf_aligned_weekly": 3}
SCHEME_ARMS: tuple[str, ...] = tuple(SCHEME_SIMPLICITY)
REFERENCE_CELL = ("F0", "S1_monthly")


def feature_set_v3(arm: str, population: str) -> list[str]:
    """The round-3 feature bundles.

    F0 is round 2's `C_plus_state`, character for character. F1 appends the
    conference flag. F2 and F3 REPLACE each raw-centred style column with its
    opponent-adjusted sibling -- replace, not append, because the adjusted
    column is the same quantity measured better, and carrying both would let a
    tree reconstruct the raw one and turn a feature-bundle comparison into a
    superset comparison."""
    f0 = PO.feature_set("C_plus_state", population)
    if arm == "F0":
        return f0
    f1 = f0 + [CONF_FEATURE]
    if arm == "F1":
        return f1
    suffix = ADJ_SUFFIX["one_pass" if arm == "F2" else "iterative"]
    if arm not in ("F2", "F3"):
        raise KeyError(f"unknown feature arm {arm!r}; known: {FEATURE_ARMS}")
    out = []
    for c in f1:
        if c.startswith("off_") and c.endswith("_c") and not c.startswith("off_rating_"):
            out.append(c[:-1] + suffix)
        elif c.startswith("opp_def_") and c.endswith("_c"):
            out.append(c[:-1] + suffix)
        else:
            out.append(c)
    return out


def build_design_v3(cache: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Round 2's design plus the conference flag and both adjusted style
    bundles. The round-2 columns are produced by round 2's own code path
    (`PO.build_design`) and are not recomputed here, so F0 is bit-identical to
    round 2's feature bundle."""
    meta_path = (Path(cache).with_suffix(".meta.json") if cache else None)
    if cache and Path(cache).exists():
        print(f"reusing design cache {cache}", flush=True)
        return pd.read_parquet(cache), json.loads(meta_path.read_text())

    print("building round-2 design (possessions v2, first-chance style rates, "
          "pbp_complete universe) ...", flush=True)
    d = PO.build_design(SEASONS, version=POSSESSIONS_VERSION,
                        style_source=STYLE_SOURCE, require_pbp_complete=True)

    # --- conference flag ---------------------------------------------------
    conf = CF.build_conference_flags(SEASONS)
    n_missing_conf = int(conf.attrs["n_missing_conference_id"])
    # merged under a private name: CONF_FEATURE is itself "is_conf_game", so
    # dropping the merge column by that name would drop the feature.
    d = d.merge(conf[["game_id", "is_conf_game"]].rename(columns={"is_conf_game": "_conf"}),
                on="game_id", how="left")
    n_unmatched = int(d["_conf"].isna().sum())
    d[CONF_FEATURE] = d["_conf"].fillna(False).astype("float32")
    d = d.drop(columns=["_conf"])

    # --- opponent-adjusted style rates ------------------------------------
    universe = pd.read_parquet(PO.DEFAULT_UNIVERSE)
    universe = universe[universe["is_d1_game"] & ~universe["pbp_truncated"]
                        & universe["pbp_complete"]].copy()
    universe["game_date"] = pd.to_datetime(universe["game_date"])
    ch = PO.load_chances(SEASONS, version=POSSESSIONS_VERSION)
    ch = ch[ch["game_id"].isin(set(universe["game_id"]))]
    boxes = pd.concat([PO.team_game_box_first_chance(ch[ch["season"] == s]) for s in SEASONS],
                      ignore_index=True)
    boxes = boxes.merge(universe[["game_id", "game_date"]], on="game_id", how="left")
    assert boxes["game_date"].notna().all(), "a team-game box row has no date"

    adj_meta = {}
    for method, suffix in ADJ_SUFFIX.items():
        t0 = time.time()
        res = OA.adjust_team_form(boxes, PO.RATE_DEFS, PO.RATE_SCALE, method)
        adj_meta[method] = {"seconds": round(time.time() - t0, 1), "per_rate": res.meta}
        corr = boxes[["season", "game_id", "team_id"]].copy()
        for r in PO.RATE_DEFS:
            corr[f"off_corr_{r}"] = res.off_adj[r].to_numpy()
            corr[f"def_corr_{r}"] = res.def_adj[r].to_numpy()
        off = corr.rename(columns={"team_id": "offense_team_id"})
        d = d.merge(off[["season", "game_id", "offense_team_id"]
                        + [f"off_corr_{r}" for r in PO.RATE_DEFS]],
                    on=["season", "game_id", "offense_team_id"], how="left")
        dfd = corr.rename(columns={"team_id": "defense_team_id"})
        d = d.merge(dfd[["season", "game_id", "defense_team_id"]
                        + [f"def_corr_{r}" for r in PO.RATE_DEFS]],
                    on=["season", "game_id", "defense_team_id"], how="left")
        for r in PO.RATE_DEFS:
            # A row whose form join failed already sits at the league mean
            # (0.0) in round 2; the correction for it is 0.0 too, so the
            # adjusted column equals the raw one there rather than inventing a
            # level (L22 -- a builder never manufactures data).
            d[f"off_{r}_{suffix}"] = (d[f"off_{r}_c"].astype("float32")
                                      - d[f"off_corr_{r}"].fillna(0.0).astype("float32"))
            d[f"opp_def_{r}_{suffix}"] = (d[f"opp_def_{r}_c"].astype("float32")
                                          - d[f"def_corr_{r}"].fillna(0.0).astype("float32"))
        d = d.drop(columns=[f"off_corr_{r}" for r in PO.RATE_DEFS]
                   + [f"def_corr_{r}" for r in PO.RATE_DEFS])

    # --- descriptive stratifier for the responsiveness check ---------------
    # Non-conference schedule strength: the mean net quality of the opponents a
    # team met in its NON-CONFERENCE games, from the as-of ratings already in
    # the design. Descriptive only -- it is never a model feature.
    d["_opp_net"] = (d["def_rating_off_c"] - d["def_rating_def_c"]).astype("float32")
    nc = d[d[CONF_FEATURE] == 0.0]
    ncss = (nc.groupby(["season", "game_id", "offense_team_id"], as_index=False)["_opp_net"].first()
              .groupby(["season", "offense_team_id"], as_index=False)["_opp_net"].mean()
              .rename(columns={"_opp_net": "ncss"}))
    d = d.merge(ncss, on=["season", "offense_team_id"], how="left")
    d["ncss"] = d["ncss"].astype("float32").fillna(0.0)
    d = d.drop(columns=["_opp_net"])

    for c in d.columns:
        if c.endswith(("_a1", "_a3")) or c in (CONF_FEATURE, "ncss"):
            assert d[c].notna().all(), f"{c} has NaN after the build; the builder must raise (L22)"
    assert d[CONF_FEATURE].nunique() > 1, "conference flag is constant"

    meta = {"n_missing_conference_id": n_missing_conf,
            "n_chances_without_schedule_row": n_unmatched,
            "conf_share_of_chances": round(float(d[CONF_FEATURE].mean()), 4),
            "adjust": adj_meta,
            "possessions_version": POSSESSIONS_VERSION, "style_source": STYLE_SOURCE}
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(cache, index=False)
        meta_path.write_text(json.dumps(meta, indent=1, default=str))
    return d, meta


def design_columns(d: pd.DataFrame) -> list[str]:
    """Every column any arm, gate or metric reads. Dropping the rest keeps the
    S1 concatenations affordable and cannot change a number."""
    cols: set[str] = set()
    for pop in PO.POPULATIONS:
        for arm in FEATURE_ARMS:
            cols |= set(feature_set_v3(arm, pop))
    cols |= {f for f, _c in PO.RESPONSIVENESS_SPECS}
    cols |= {"y", "season", "game_id", "game_date", "population",
             "offense_team_id", "defense_team_id", "ncss", CONF_FEATURE}
    return sorted(c for c in cols if c in d.columns)


# ---------------------------------------------------------------------------
# 2. Schemes: refit-date calendars
# ---------------------------------------------------------------------------
def refit_dates(scheme: str, te: pd.DataFrame, conf: pd.DataFrame) -> list[pd.Timestamp]:
    """The refit calendar of `scheme` for a test slice.

    All four arms keep round 2's S1 contract exactly: each refit uses games
    STRICTLY BEFORE its own date, and each test game is scored by the most
    recent refit at or before its own date, so no game is ever in its own fit.
    Only the CALENDAR changes.

    The conference-aligned calendars use the DATES ON WHICH TEAMS FIRST PLAY A
    CONFERENCE GAME. Those dates come from the published schedule, which is
    known before the season is played and contains no result, so choosing refit
    dates from them is not a leak -- and refitting once per distinct boundary
    date rather than once per team is what makes per-team alignment free."""
    dates = pd.to_datetime(te["game_date"])
    season = int(te["season"].iloc[0])
    monthly = PO.month_boundaries(dates)
    if scheme == "S1_monthly":
        return monthly
    weekly = CF.weekly_boundaries(dates)
    if scheme == "S1_weekly":
        return CF.union_boundaries(weekly, monthly[:1])
    bounds = CF.conference_boundary_dates(conf, season)
    if scheme == "S1_conf_aligned":
        return CF.union_boundaries(monthly, bounds)
    if scheme == "S1_conf_aligned_weekly":
        return CF.union_boundaries(weekly, monthly[:1], bounds)
    raise KeyError(f"unknown scheme {scheme!r}; known: {SCHEME_ARMS}")


def fit_predict_walkforward(arm: str, tr: pd.DataFrame, te: pd.DataFrame,
                            features: list[str], cuts: list[pd.Timestamp],
                            seed: int = 0) -> tuple[np.ndarray, dict]:
    """Round 2's `PO.fit_predict_scheme` S1 branch, generalised from the month
    boundaries to an arbitrary refit calendar. The partition logic, the
    strictly-before rule and the column subsetting are transcribed unchanged;
    only `cuts` is a parameter instead of `month_boundaries(te_dates)`."""
    te_dates = pd.to_datetime(te["game_date"])
    tr_dates = pd.to_datetime(tr["game_date"])
    fit_cols = [*features, "y", "season"]
    p = np.zeros((len(te), len(PO.CLASSES)), dtype="float64")
    segments = []
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else None
        seg = (te_dates >= cut) if nxt is None else ((te_dates >= cut) & (te_dates < nxt))
        seg = seg.to_numpy()
        if not seg.any():
            continue
        before = (te_dates < cut).to_numpy()
        prior_test = te.loc[before, fit_cols]
        fit_rows = (tr if not len(prior_test)
                    else pd.concat([tr[fit_cols], prior_test], ignore_index=True))
        model = PO.fit_arm(arm, fit_rows, features, seed=seed)
        p[seg] = PO.predict_arm(arm, model, te.loc[seg], features)
        max_train = (tr_dates.max() if not before.any()
                     else max(tr_dates.max(), te_dates[before].max()))
        segments.append({"refit_date": str(pd.Timestamp(cut).date()),
                         "n_train": int(len(fit_rows)),
                         "n_train_from_test_season": int(len(prior_test)),
                         "n_scored": int(seg.sum()),
                         "max_train_date": str(pd.Timestamp(max_train).date())})
    if (p.sum(axis=1) == 0).any():
        raise AssertionError("the refit calendar is not a cover of the test slice")
    return p, {"n_fits": len(segments), "segments": segments,
               "earliest_train_date": str(tr_dates.min().date()) if len(tr_dates) else None}


# ---------------------------------------------------------------------------
# 3. Extra metrics the round-3 pre-registration adds
# ---------------------------------------------------------------------------
def conf_window_calibration(te: pd.DataFrame, p: np.ndarray, firsts: pd.DataFrame) -> dict:
    """Worst gated decile calibration gap on the regime segments.

    `first4_conf_weeks` is the segment Decision 9 predicted a mis-aligned refit
    damages, by the offence team's own boundary. `non_conference` is the segment
    the STAGE-A DIAGNOSTIC actually found the damage in
    (`docs/tests/possession_outcome_conference_regime_2026-09-10.md`): under the
    round-2 S1 winner the overall gap is 0.98 pp but the non-conference gap is
    2.49 pp and the pre-boundary gap 2.95 pp, both outside the 2.0 pp gate, while
    conference games sit at 0.98 pp. Both are pre-registered decision quantities
    for round 3; the rest are reported evidence."""
    t = te[["offense_team_id", "game_date", "y"]].copy()
    t["game_date"] = pd.to_datetime(t["game_date"])
    t = t.merge(firsts.rename(columns={"team_id": "offense_team_id"}),
                on=["offense_team_id"], how="left")
    wk = (t["game_date"] - t["first_conf_date"]).dt.days / 7.0
    season_wk = (t["game_date"] - t["game_date"].min()).dt.days / 7.0
    conf = te[CONF_FEATURE].to_numpy() > 0.5
    segs = (("first4_conf_weeks", (wk >= 0) & (wk < 4)),
            ("conf_weeks_4plus", wk >= 4),
            ("pre_boundary", wk < 0),
            ("non_conference", pd.Series(~conf, index=t.index)),
            ("conference", pd.Series(conf, index=t.index)),
            ("first4_season_weeks", season_wk < 4))
    out = {}
    for name, m in segs:
        m = m.fillna(False).to_numpy()
        if m.sum() < 20_000:
            out[name] = {"n": int(m.sum()), "worst_gated_gap_pp": None, "underpowered": True}
            continue
        cal = PO.decile_calibration(te["y"].to_numpy()[m], p[m])
        gated = [v["max_abs_gap_pp"] for v in cal.values()
                 if v["share_pct"] >= R1.CAL_GATE_MIN_SHARE]
        out[name] = {"n": int(m.sum()),
                     "worst_gated_gap_pp": round(float(max(gated)), 3) if gated else None,
                     "underpowered": False}
    return out


SLOPE_BAND = (0.8, 1.2)          # Decision 8
MIN_SPAN_PP = 2.0                # Decision 8 exemption threshold
NCSS_SPECS: tuple[tuple[str, str], ...] = (
    ("ncss", "FGA_3"), ("ncss", "FGA_rim"), ("ncss", "TOV"),
)


def responsiveness_by(te: pd.DataFrame, p: np.ndarray, driver: str, cls: str,
                      n_q: int = 5) -> dict:
    """Decision 8's responsiveness reading for one (driver, class) pair.

    Same shape as `PO.responsiveness` but the driver column is a parameter, so
    the non-conference-schedule-strength check can reuse it."""
    j = PO.CLASS_INDEX[cls]
    x = te[driver].to_numpy()
    try:
        q = pd.qcut(pd.Series(x), n_q, labels=False, duplicates="drop")
    except ValueError:
        return {"driver": driver, "class": cls, "status": "degenerate driver"}
    pred, act, n = [], [], []
    for b in sorted(pd.Series(q).dropna().unique()):
        m = (q == b).to_numpy()
        pred.append(float(p[m, j].mean()))
        act.append(float((te["y"].to_numpy()[m] == j).mean()))
        n.append(int(m.sum()))
    pred, act = np.array(pred), np.array(act)
    span_a = float(act[-1] - act[0])
    span_p = float(pred[-1] - pred[0])
    steps = int(np.sum(np.sign(np.diff(pred)) == np.sign(np.diff(act))))
    exempt = abs(span_a) * 100.0 < MIN_SPAN_PP
    ratio = float(span_p / span_a) if abs(span_a) > 1e-9 else float("nan")
    return {"driver": driver, "class": cls, "n": n,
            "pred": [round(v, 5) for v in pred.tolist()],
            "act": [round(v, 5) for v in act.tolist()],
            "span_pred_pp": round(span_p * 100, 3), "span_act_pp": round(span_a * 100, 3),
            "slope_ratio": round(ratio, 4) if np.isfinite(ratio) else None,
            "steps_with_actual": steps, "n_steps": len(pred) - 1,
            "exempt_narrow_span": bool(exempt),
            "pass": bool(exempt or (SLOPE_BAND[0] <= ratio <= SLOPE_BAND[1]
                                    and steps >= PO.RESPONSIVENESS_MIN_STEPS))}


# ---------------------------------------------------------------------------
# 4. Leak test on the adjusted features
# ---------------------------------------------------------------------------
def leak_test_adjusted(design: pd.DataFrame) -> pd.DataFrame:
    """The standing INV-45 change-form leak test on every column round 3 adds.

    The panel is one row per (season, team, game) on the OFFENCE side, with that
    team's own margin, exactly as `cbb_sim.analysis.leak_test` wants it. The
    reference arm's raw-centred columns are tested alongside the adjusted ones,
    because a leak number is only interpretable next to the number the same
    statistic gives on a column already accepted."""
    u = pd.read_parquet(PO.DEFAULT_UNIVERSE)
    u = u[u["season"].isin(SEASONS) & u["is_d1_game"]].copy()
    u["game_date"] = pd.to_datetime(u["game_date"])
    home = u[["game_id", "season", "game_date", "home_team_id", "home_score", "away_score"]].rename(
        columns={"home_team_id": "team", "home_score": "own", "away_score": "opp"})
    away = u[["game_id", "season", "game_date", "away_team_id", "away_score", "home_score"]].rename(
        columns={"away_team_id": "team", "away_score": "own", "home_score": "opp"})
    marg = pd.concat([home, away], ignore_index=True)
    marg["margin"] = marg["own"] - marg["opp"]

    cols = [f"off_{r}_c" for r in PO.RATE_DEFS] + \
           [f"off_{r}_{s}" for r in PO.RATE_DEFS for s in ADJ_SUFFIX.values()] + [CONF_FEATURE]
    feat = (design.groupby(["season", "game_id", "offense_team_id"], as_index=False)[cols]
            .first().rename(columns={"offense_team_id": "team"}))
    panel = marg.merge(feat, on=["season", "game_id", "team"], how="inner")
    rows = []
    for c in cols:
        r = LT.leak_test_column(panel, c, team_col="team")
        r["column"] = c
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. The pre-registered stage order
# ---------------------------------------------------------------------------
def planned_cells() -> list[dict]:
    """Every grid cell round 3 pre-registers, IN THE ORDER the pre-registration
    fixes. The budget rule drops from the bottom up, so the order IS the drop
    order: `S1_conf_aligned_weekly` goes first, then F3, exactly as instructed.

    The order is not the obvious one (all of dimension A, then all of dimension
    B) and the reason is measured, not guessed. At the four-thread cap this
    machine gives a shared worker, one tree fit on the fold-2 `first` training
    slice costs about eight minutes, and the arms multiply FITS: `S1_monthly` is
    6 refits, `S1_weekly` about 23, `S1_conf_aligned` about 29. So a single tree
    ALIGNMENT cell costs roughly four hours while a tree FEATURE cell costs
    fifty minutes. Running the cheap cells first buys the reference, the noise
    floor and the whole opponent-adjustment question inside four hours, and
    leaves the alignment arm to run against the clock rather than the other way
    round."""
    cells: list[dict] = []

    def add(stage, pop, arm, fold, fa, sa, role, seed=0):
        cells.append({"stage": stage, "population": pop, "arm": arm, "fold": fold,
                      "feature_arm": fa, "scheme": sa, "role": role, "seed": seed})

    # Stage 1 -- the cheap linear cells: the FULL 4 x 4 cross.
    #   * on `cont` this IS the selection grid (round 2's winner there is cascade)
    #   * on `first` it is an INTERACTION PROBE, fold 2 only, reported in full and
    #     explicitly not the selection metric. It is the only affordable way to see
    #     the whole scheme x feature interaction surface on that population.
    for fa in FEATURE_ARMS:
        for sa in SCHEME_ARMS:
            for fold in ("F1", "F2"):
                add(1, "cont", "cascade", fold, fa, sa, "selection")
            add(1, "first", "cascade", "F2", fa, sa, "interaction probe (not selection)")
    # Stage 2, 3 -- the tree reference cell, selection fold then fold 1.
    add(2, "first", "lgbm", "F2", *REFERENCE_CELL, "reference")
    add(3, "first", "lgbm", "F1", *REFERENCE_CELL, "reference")
    # Stage 4 -- the noise floor: the reference cell under a second seed.
    for pop in ("first", "cont"):
        add(4, pop, POP_ARM[pop], "F2", *REFERENCE_CELL, "noise floor", seed=1)
    # Stages 5-7 -- the tree FEATURE ladder at the reference scheme (Decision 9a/9b).
    for fa in ("F1", "F2", "F3"):
        add(4 + FEATURE_SIMPLICITY[fa], "first", "lgbm", "F2", fa, "S1_monthly", "selection")
    # Stages 8, 9 -- the tree SCHEME ladder at the reference features (Decision 9c).
    add(8, "first", "lgbm", "F2", "F0", "S1_conf_aligned", "selection")
    add(9, "first", "lgbm", "F2", "F0", "S1_weekly", "selection")
    # Stage 11 -- the densest alignment: the first thing the drop order sheds.
    add(11, "first", "lgbm", "F2", "F0", "S1_conf_aligned_weekly", "selection")
    return cells


def cell_key(c: dict) -> str:
    return (f"{c['population']}|{c['fold']}|{c['arm']}|{c['feature_arm']}|{c['scheme']}"
            f"|s{c.get('seed', 0)}")


class Checkpoint:
    """Finished measurements on disk, so a killed run recomputes nothing.
    Round 2's rationale (two runs lost to the environment mid-LightGBM)
    applies with more force here: round 3's tree cells are hours."""

    def __init__(self, path: Path, enabled: bool = True):
        self.path, self.enabled = Path(path), enabled
        self.d: dict = {"grid": {}, "detail": {}, "meta": {}, "floors": {}, "extra": {}}
        if enabled and self.path.exists():
            loaded = json.loads(self.path.read_text())
            self.d.update({k: loaded.get(k, v) for k, v in self.d.items()})
            print(f"resuming from {self.path}: {len(self.d['grid'])} cells", flush=True)

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.d, indent=1, default=str))
        os.replace(tmp, self.path)

    def has(self, key: str) -> bool:
        return key in self.d["grid"]

    def put(self, key: str, row: dict, detail: dict, meta: dict, extra: dict) -> None:
        self.d["grid"][key] = row
        self.d["detail"][key] = detail
        self.d["meta"][key] = meta
        self.d["extra"][key] = extra
        self.save()


def run_cell(c: dict, design: pd.DataFrame, conf: pd.DataFrame, firsts: pd.DataFrame,
             seed: int = 0) -> tuple[dict, dict, dict, dict, np.ndarray]:
    pop, fold = c["population"], c["fold"]
    feats = feature_set_v3(c["feature_arm"], pop)
    tr, te = PO.fold_slices(design, fold, pop)
    cuts = refit_dates(c["scheme"], te, conf)
    t0 = time.time()
    p, meta = fit_predict_walkforward(c["arm"], tr, te, feats, cuts, seed=seed)
    s = R1.score(te, p)
    dt = time.time() - t0
    extra = {
        "conf_window": conf_window_calibration(te, p, firsts),
        "responsiveness_ncss": [responsiveness_by(te, p, d, cl) for d, cl in NCSS_SPECS],
        "responsiveness_own": [responsiveness_by(te, p, d, cl) for d, cl in PO.RESPONSIVENESS_SPECS],
    }
    w = extra["conf_window"]["first4_conf_weeks"]
    nc = extra["conf_window"]["non_conference"]
    row = {"stage": c["stage"], "role": c["role"], "population": pop, "fold": fold,
           "arm": c["arm"], "feature_arm": c["feature_arm"], "scheme": c["scheme"],
           "seed": c.get("seed", 0),
           "n_fits": meta["n_fits"], "n_train": len(tr), "n_test": len(te),
           "log_loss": round(s["log_loss"], 6),
           "worst_gated_gap_pp": round(s["worst_gated_gap_pp"], 3),
           "worst_gated_level_pp": round(s["worst_gated_level_pp"], 3),
           "worst_gated_shape_pp": round(s["worst_gated_shape_pp"], 3),
           "conf4_gap_pp": w["worst_gated_gap_pp"],
           "nonconf_gap_pp": nc["worst_gated_gap_pp"],
           "first4_season_gap_pp": extra["conf_window"]["first4_season_weeks"]["worst_gated_gap_pp"],
           "calibration_pass": s["calibration_pass"],
           "responsiveness_pass": s["responsiveness_pass"],
           "ncss_slope_pass": all(r.get("pass", False) for r in extra["responsiveness_ncss"]),
           **{f"brier_{k}": round(v, 6) for k, v in s["brier"].items()},
           "fit_seconds": round(dt, 1)}
    return row, s, meta, extra, p


# ---------------------------------------------------------------------------
# 6. Decision rule
# ---------------------------------------------------------------------------
def decide_v3(grid: pd.DataFrame, population: str, nf: float,
              fold: str = PO.SELECTION_FOLD) -> dict:
    """The pre-registered round-3 decision rule, transcribed.

    Within each dimension the SIMPLEST arm wins unless a more complex one beats
    it beyond the noise floor on the primary metric (F2 log loss) OR on the
    first-four-conference-weeks calibration gap, and every arm must still clear
    round 1's calibration and responsiveness gates. Adoption of the adjusted /
    aligned arms is PENDING EVIDENCE: a tie or a loss for them is a result, not
    a failure to fix, and the reference cell stands in that case."""
    sub = grid[(grid["population"] == population) & (grid["fold"] == fold)
               & (grid["arm"] == POP_ARM[population])
               & (grid["role"] != "interaction probe (not selection)")].copy()
    out: dict = {"population": population, "fold": fold, "noise_floor": nf,
                 "n_cells": int(len(sub))}
    if not len(sub):
        out["winner"] = None
        out["reason"] = "no cell of the selection arm was run"
        return out
    ref = sub[(sub["feature_arm"] == REFERENCE_CELL[0]) & (sub["scheme"] == REFERENCE_CELL[1])]
    out["reference"] = (ref.iloc[0].to_dict() if len(ref) else None)

    def _ladder(dim: str, held: str, held_value: str, order: dict) -> dict:
        lad = sub[sub[held] == held_value].copy()
        lad = lad.sort_values(dim, key=lambda s: s.map(order))
        if not len(lad):
            return {"cells": [], "winner": None, "reason": "ladder not run"}
        base = lad.iloc[0]
        rows, beaters = [], []
        for _, r in lad.iterrows():
            d_ll = float(base["log_loss"] - r["log_loss"])
            b4, r4 = base["conf4_gap_pp"], r["conf4_gap_pp"]
            d_c4 = (float(b4) - float(r4)) if (b4 is not None and r4 is not None) else None
            bn, rn = base["nonconf_gap_pp"], r["nonconf_gap_pp"]
            d_nc = (float(bn) - float(rn)) if (bn is not None and rn is not None) else None
            gates = bool(r["calibration_pass"] and r["responsiveness_pass"])
            beats = bool(gates and (d_ll > nf
                                    or (d_c4 is not None and d_c4 > SEGMENT_FLOOR_PP)
                                    or (d_nc is not None and d_nc > SEGMENT_FLOOR_PP)))
            rows.append({dim: r[dim], "log_loss": float(r["log_loss"]),
                         "gain_vs_reference": round(d_ll, 6),
                         "conf4_gap_pp": r4,
                         "conf4_gain_vs_reference_pp": (round(d_c4, 3) if d_c4 is not None else None),
                         "nonconf_gap_pp": rn,
                         "nonconf_gain_vs_reference_pp": (round(d_nc, 3) if d_nc is not None else None),
                         "gates_pass": gates,
                         "beats_reference_beyond_floor": beats})
            if beats and r[dim] != base[dim]:
                beaters.append(r)
        if not beaters:
            return {"cells": rows, "winner": str(base[dim]), "reason": (
                f"the simplest arm `{base[dim]}` stands -- no more complex arm beat it by more "
                f"than the noise floor {nf:.5f} on log loss or by more than "
                f"{SEGMENT_FLOOR_PP} pp on the first-4-conference-weeks gap. Under the "
                f"pre-registration that is a RESULT, not a failure")}
        # among the arms that DO beat the reference, ties still go to the simpler one:
        # take the simplest whose log loss is within the floor of the best beater's.
        best_ll = min(float(r["log_loss"]) for r in beaters)
        chosen = next(r for r in beaters if float(r["log_loss"]) <= best_ll + nf)
        d_ll = float(base["log_loss"] - chosen["log_loss"])
        return {"cells": rows, "winner": str(chosen[dim]), "reason": (
            f"`{chosen[dim]}` beats the reference `{base[dim]}` beyond the floor "
            f"(log-loss gain {d_ll:.5f} against floor {nf:.5f}; first-4-conference-weeks gap "
            f"{base['conf4_gap_pp']} -> {chosen['conf4_gap_pp']} pp) and is the simplest arm "
            f"within the floor of the best beater ({best_ll:.6f})")}

    out["scheme_ladder"] = _ladder("scheme", "feature_arm", REFERENCE_CELL[0], SCHEME_SIMPLICITY)
    out["feature_ladder"] = _ladder("feature_arm", "scheme", REFERENCE_CELL[1], FEATURE_SIMPLICITY)
    ws = out["scheme_ladder"]["winner"] or REFERENCE_CELL[1]
    wf = out["feature_ladder"]["winner"] or REFERENCE_CELL[0]
    inter = sub[(sub["scheme"] == ws) & (sub["feature_arm"] == wf)]
    out["winner"] = {"feature_arm": wf, "scheme": ws,
                     "log_loss": (float(inter.iloc[0]["log_loss"]) if len(inter) else None),
                     "conf4_gap_pp": (inter.iloc[0]["conf4_gap_pp"] if len(inter) else None),
                     "interaction_cell_run": bool(len(inter))}
    out["adopted_vs_reference"] = bool((wf, ws) != REFERENCE_CELL)
    return out


#: A SEGMENT calibration-gap improvement (first-4-conference-weeks, or
#: non-conference) must clear this to count. It is the pre-registered analogue of
#: the log-loss noise floor; the second-seed refit of the reference cell reports
#: its own segment gaps so the number can be checked against a measured spread
#: rather than asserted.
SEGMENT_FLOOR_PP = 0.25


# ---------------------------------------------------------------------------
# 7. Artifacts: the per-refit-date manifest the engine's selector reads
# ---------------------------------------------------------------------------
def write_manifest(design: pd.DataFrame, conf: pd.DataFrame, verdicts: dict,
                   out_dir: Path, fold: str = PO.SELECTION_FOLD,
                   write_artifacts: bool = True) -> dict:
    """Refit the selected cell per population, persist one fitted object per
    refit date, and write `manifest.json` mapping refit_date -> artifact path.

    That mapping IS the deployed model: L21's second consequence is that the
    artifact is a SCHEDULE, and the engine's per-game selector picks the latest
    entry at or before a game's date."""
    art = out_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    man: dict = {"schema_version": 1,
                 "created_at": time.strftime("%Y-%m-%d %H:%M"),
                 "fold": fold, "test_season": PO.FOLDS[fold]["test"][0],
                 "selector": "latest refit_date <= game_date",
                 "populations": {}}
    for pop, v in verdicts.items():
        w = v.get("winner")
        if not w:
            continue
        fa, sa = w["feature_arm"], w["scheme"]
        arm = POP_ARM[pop]
        feats = feature_set_v3(fa, pop)
        tr, te = PO.fold_slices(design, fold, pop)
        cuts = refit_dates(sa, te, conf)
        te_dates = pd.to_datetime(te["game_date"])
        tr_max = pd.to_datetime(tr["game_date"]).max()
        entries = []
        for cut in cuts:
            before = (te_dates < cut).to_numpy()
            prior = te.loc[before, [*feats, "y", "season"]]
            rows = tr if not len(prior) else pd.concat([tr[[*feats, "y", "season"]], prior],
                                                       ignore_index=True)
            rel = f"artifacts/{pop}_{arm}_{fa}_{sa}_{pd.Timestamp(cut).date()}.pkl.gz"
            e = {"refit_date": str(pd.Timestamp(cut).date()), "artifact": rel,
                 "n_train": int(len(rows)),
                 "max_train_date": str(pd.Timestamp(
                     tr_max if not before.any() else max(tr_max, te_dates[before].max())).date())}
            if write_artifacts:
                model = PO.fit_arm(arm, rows, feats, seed=0)
                with gzip.open(out_dir / rel, "wb") as fh:
                    pickle.dump({"arm": arm, "population": pop, "feature_arm": fa,
                                 "features": feats, "scheme": sa,
                                 "refit_date": e["refit_date"],
                                 "classes": list(PO.CLASSES), "model": model}, fh)
                e["bytes"] = int((out_dir / rel).stat().st_size)
            entries.append(e)
            print(f"    manifest {pop} {rel} "
                  f"({e.get('bytes', 0)/1e6:.1f} MB)", flush=True)
        man["populations"][pop] = {"arm": arm, "feature_arm": fa, "scheme": sa,
                                   "features": feats, "classes": list(PO.CLASSES),
                                   "artifacts_written": bool(write_artifacts),
                                   "refits": entries,
                                   "total_bytes": int(sum(e.get("bytes", 0) for e in entries))}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


# ---------------------------------------------------------------------------
# 7b. experiments.md rendering
# ---------------------------------------------------------------------------
GRID_COLS = ["stage", "role", "population", "fold", "arm", "feature_arm", "scheme",
             "n_fits", "log_loss", "worst_gated_gap_pp", "worst_gated_level_pp",
             "worst_gated_shape_pp", "conf4_gap_pp", "nonconf_gap_pp",
             "first4_season_gap_pp", "calibration_pass",
             "responsiveness_pass", "ncss_slope_pass", "fit_seconds"]


def render(grid: pd.DataFrame, detail: dict, extras: dict, floors: dict,
           verdicts: dict, leak: pd.DataFrame, not_run: list, run_meta: dict) -> str:
    def T(df: pd.DataFrame) -> list[str]:
        """One markdown table, as lines, from whatever columns the frame has."""
        return [R1._md_table(df, list(df.columns))]

    L: list[str] = ["", "---", ""]
    A = L.append
    A(f"## 7. Round 3 full results (run {run_meta['run_at']}, "
      f"`scripts/train_possession_outcome_v3.py`)")
    A("")
    A(f"Wall clock {run_meta['runtime_hours']:.2f} h against a pre-registered budget of "
      f"{run_meta['budget_hours']} h, at {run_meta['threads']} threads (four workers share the "
      f"machine). Cells run: {run_meta['n_cells']}. Cells the budget did not reach: "
      f"{run_meta['n_not_run']} -- listed in section 7.6 as NOT RUN, never as a result.")
    A("")
    A(f"Design: conference flag on {run_meta['design'].get('conf_share_of_chances')} of chances; "
      f"{run_meta['design'].get('n_missing_conference_id')} D-I games carry no hoopR conference id "
      f"on one side and are flagged non-conference and counted; "
      f"{run_meta['design'].get('n_chances_without_schedule_row')} chances matched no schedule row.")
    A("")
    A("### 7.1 The cross")
    A("")
    for pop in ("first", "cont"):
        for fold in ("F2", "F1"):
            sub = grid[(grid["population"] == pop) & (grid["fold"] == fold)]
            if not len(sub):
                continue
            A(f"**`{pop}` / {fold}**" + ("  (SELECTION)" if fold == "F2" else ""))
            A("")
            L.extend(T(sub[[c for c in GRID_COLS if c in sub.columns]]
                       .sort_values(["arm", "feature_arm", "scheme"])))
            A("")
    A("`conf4_gap_pp` is the worst gated decile calibration gap over the chances in the FIRST FOUR "
      "WEEKS OF CONFERENCE PLAY, by the offence team's own boundary -- the segment Decision 9 "
      "predicts a mis-aligned refit damages. `ncss_slope_pass` is Decision 8's slope reading "
      "against the non-conference-schedule-strength quintile, which is the direct test of "
      "Decision 9's claim.")
    A("")
    A("### 7.2 Noise floor")
    A("")
    for pop, f in floors.items():
        A(f"* **`{pop}`**: reference cell `{REFERENCE_CELL[0]} x {REFERENCE_CELL[1]}` refit under a "
          f"second seed -- log loss {f['seed0_log_loss']:.6f} (seed 0) vs {f['seed1_log_loss']:.6f} "
          f"(seed 1), **spread {f['seed_spread']:.6f}**; first-4-conference-weeks gap "
          f"{f['seed0_conf4_gap_pp']} vs {f['seed1_conf4_gap_pp']} pp; "
          f"{R1.N_BOOTSTRAP}-replicate game-block bootstrap SE {f['block_bootstrap_se']:.6f}. "
          f"Applied floor **{f['applied']:.6f}**. PARTIAL: {f['partial_reason']}.")
    A("")
    A("### 7.3 Decision")
    A("")
    for pop, v in verdicts.items():
        A(f"**`{pop}`** (fold F2, arm `{POP_ARM[pop]}`, noise floor {v['noise_floor']:.6f})")
        A("")
        for dim in ("scheme_ladder", "feature_ladder"):
            lad = v.get(dim) or {}
            if lad.get("cells"):
                A(f"*{dim.replace('_', ' ')}*")
                A("")
                L.extend(T(pd.DataFrame(lad["cells"])))
                A("")
                A(f"Winner: `{lad['winner']}` -- {lad['reason']}.")
                A("")
        w = v.get("winner") or {}
        A(f"**Selected cell: `{w.get('feature_arm')}` x `{w.get('scheme')}`.** "
          + ("This is the reference cell: the round-3 arms did not beat it beyond the floor, "
             "which under the pre-registration is a RESULT and not a failure -- opponent "
             "adjustment and conference alignment stay PENDING EVIDENCE."
             if not v.get("adopted_vs_reference") else
             "The round-3 arm beat the reference beyond the floor and is adopted for this "
             "sub-model."))
        if w and not w.get("interaction_cell_run"):
            A("")
            A("The two ladder winners' INTERACTION cell was not reached by the budget; the "
              "selected cell's own metrics are therefore read from its ladder rows and the "
              "interaction is recorded as unmeasured.")
        A("")
    A("### 7.4 Responsiveness, both drivers")
    A("")
    rows = []
    for key, e in extras.items():
        for r in e.get("responsiveness_own", []) + e.get("responsiveness_ncss", []):
            if "slope_ratio" not in r:
                continue
            rows.append({"cell": key, "driver": r["driver"], "class": r["class"],
                         "span_pred_pp": r["span_pred_pp"], "span_act_pp": r["span_act_pp"],
                         "slope_ratio": r["slope_ratio"],
                         "steps": f"{r['steps_with_actual']}/{r['n_steps']}",
                         "exempt_narrow_span": r["exempt_narrow_span"], "pass": r["pass"]})
    if rows:
        L.extend(T(pd.DataFrame(rows)))
    A("")
    A("### 7.5 Leak test on every column round 3 adds")
    A("")
    lk = leak.copy()
    keep = [c for c in ("column", "corr_asjoined", "corr_update", "corr_level",
                        "n", "static", "verdict", "level_verdict") if c in lk.columns]
    if keep:
        lk = lk[keep].copy()
        for c in ("corr_asjoined", "corr_update", "corr_level"):
            if c in lk.columns:
                lk[c] = lk[c].astype(float).round(4)
    L.extend(T(lk))
    A("")
    A(f"Gate: |as-joined change-form corr| <= {LT.GATE} (CLAUDE.md, standing rule "
      "'backtests must be honest'). The raw-centred reference columns are shown alongside the "
      "adjusted ones so the adjusted numbers are read against a column already accepted.")
    A("")
    A("### 7.6 Cells the budget did not reach (NOT RUN, not a result)")
    A("")
    if not_run:
        L.extend(T(pd.DataFrame(not_run)))
    else:
        A("None -- every pre-registered cell ran.")
    A("")
    return chr(10).join(L) + chr(10)


# ---------------------------------------------------------------------------
# 8. main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-hours", type=float, default=6.5)
    ap.add_argument("--stages", default="1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--manifest-no-artifacts", action="store_true",
                    help="write manifest.json as a SCHEDULE only (refit dates, train sizes, "
                         "artifact paths) without refitting to persist the objects. Used when a "
                         "hard stop lands before the manifest pass: the format the engine's "
                         "selector reads is still delivered and the fits can be produced later "
                         "without re-deciding anything.")
    a = ap.parse_args()
    stages = {int(x) for x in a.stages.split(",") if x.strip()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    design, dmeta = build_design_v3(OUT_DIR / "design_v3.parquet")
    design = design[design_columns(design)]
    conf = CF.build_conference_flags(SEASONS)
    firsts = CF.first_conference_game_dates(conf)
    firsts = firsts[firsts["season"] == PO.FOLDS[PO.SELECTION_FOLD]["test"][0]][
        ["team_id", "first_conf_date"]]
    print(json.dumps(dmeta, indent=1, default=str)[:1200], flush=True)

    ckpt = Checkpoint(OUT_DIR / "checkpoint.json", enabled=not a.no_resume)
    cells = [c for c in planned_cells() if c["stage"] in stages]
    rows, detail, metas, extras = [], {}, {}, {}
    not_run: list[dict] = []

    def _execute(c: dict) -> None:
        key = cell_key(c)
        if ckpt.has(key):
            rows.append(ckpt.d["grid"][key])
            detail[key] = ckpt.d["detail"][key]
            metas[key] = ckpt.d["meta"][key]
            extras[key] = ckpt.d["extra"][key]
            print(f"  {key:52s} (checkpoint)", flush=True)
            return
        elapsed = (time.time() - t_start) / 3600.0
        if c["arm"] == "lgbm" and elapsed > a.budget_hours:
            not_run.append({**c, "reason": f"wall clock {elapsed:.2f}h over budget "
                                           f"{a.budget_hours}h"})
            print(f"  {key:52s} NOT RUN (budget)", flush=True)
            return
        row, s, meta, extra, p = run_cell(c, design, conf, firsts, seed=c.get("seed", 0))
        rows.append(row)
        detail[key], metas[key], extras[key] = s, meta, extra
        if (c["fold"] == PO.SELECTION_FOLD and c.get("seed", 0) == 0
                and c["arm"] == POP_ARM[c["population"]]
                and (c["feature_arm"], c["scheme"]) == REFERENCE_CELL):
            np.save(OUT_DIR / f"ref_pred_{c['population']}_F2_seed0.npy", p.astype("float32"))
        ckpt.put(key, row, s, meta, extra)
        print(f"  {key:52s} n_fits {row['n_fits']:3d}  logloss {row['log_loss']:.6f}  "
              f"cal {'PASS' if row['calibration_pass'] else 'FAIL'} "
              f"({row['worst_gated_gap_pp']:.2f}pp)  conf4 {row['conf4_gap_pp']}  "
              f"[{row['fit_seconds']:.0f}s]", flush=True)

    for c in cells:
        _execute(c)

    # --- stages 5 and 6: constructed only once the ladders have run ---------
    grid = pd.DataFrame(rows)
    if len(grid):
        for pop in ("first", "cont"):
            nf_tmp = 0.0
            v = decide_v3(grid, pop, nf_tmp)
            ws = (v.get("scheme_ladder") or {}).get("winner") or REFERENCE_CELL[1]
            wf = (v.get("feature_ladder") or {}).get("winner") or REFERENCE_CELL[0]
            extra_cells = []
            if 10 in stages and (wf, ws) != REFERENCE_CELL:
                extra_cells.append({"stage": 10, "population": pop, "arm": POP_ARM[pop],
                                    "fold": PO.SELECTION_FOLD, "feature_arm": wf,
                                    "scheme": ws, "seed": 0,
                                    "role": "selection (ladder interaction)"})
            for c in extra_cells:
                if not any(cell_key(c) == cell_key(x) for x in cells):
                    _execute(c)
                    cells.append(c)
    grid = pd.DataFrame(rows)
    # --- noise floor: the reference cell under a second seed (stage 3) -------
    floors = {}
    for pop in ("first", "cont"):
        base = grid[(grid["population"] == pop) & (grid["fold"] == PO.SELECTION_FOLD)
                    & (grid["arm"] == POP_ARM[pop]) & (grid["seed"] == 0)
                    & (grid["feature_arm"] == REFERENCE_CELL[0])
                    & (grid["scheme"] == REFERENCE_CELL[1])]
        alt = grid[(grid["population"] == pop) & (grid["fold"] == PO.SELECTION_FOLD)
                   & (grid["arm"] == POP_ARM[pop]) & (grid["seed"] == 1)
                   & (grid["feature_arm"] == REFERENCE_CELL[0])
                   & (grid["scheme"] == REFERENCE_CELL[1])]
        if not len(base) or not len(alt):
            continue
        b, a2 = base.iloc[0], alt.iloc[0]
        _tr, te_ = PO.fold_slices(design, PO.SELECTION_FOLD, pop)
        ref_npy = OUT_DIR / f"ref_pred_{pop}_F2_seed0.npy"
        se = (round(PO.block_bootstrap_se(te_, np.load(ref_npy).astype("float64"),
                                          n_rep=R1.N_BOOTSTRAP), 6)
              if ref_npy.exists() else None)
        spread = round(abs(float(b["log_loss"]) - float(a2["log_loss"])), 6)
        floors[pop] = {
            "population": pop, "seed0_log_loss": float(b["log_loss"]),
            "seed1_log_loss": float(a2["log_loss"]), "seed_spread": spread,
            "seed0_conf4_gap_pp": b["conf4_gap_pp"], "seed1_conf4_gap_pp": a2["conf4_gap_pp"],
            "seed0_nonconf_gap_pp": b["nonconf_gap_pp"], "seed1_nonconf_gap_pp": a2["nonconf_gap_pp"],
            "conf4_seed_spread_pp": (round(abs(float(b["conf4_gap_pp"]) - float(a2["conf4_gap_pp"])), 3)
                                     if (b["conf4_gap_pp"] is not None
                                         and a2["conf4_gap_pp"] is not None) else None),
            "block_bootstrap_se": se, "n_seeds": 2,
            "partial_reason": ("2 seeds, against the 5 round 1 pre-registered; the round-3 "
                               "pre-registration asks for one second seed and the run is "
                               "wall-clock bound"),
        }
        floors[pop]["applied"] = max([x for x in (spread, se) if x is not None])

    verdicts = {}
    for pop in ("first", "cont"):
        nf = float(floors.get(pop, {}).get("applied", float("nan")))
        verdicts[pop] = decide_v3(grid, pop, nf)

    leak = leak_test_adjusted(design)
    leak.to_csv(OUT_DIR / "leak_test_adjusted.csv", index=False)
    grid.to_csv(OUT_DIR / "grid_results.csv", index=False)
    (OUT_DIR / "metrics_detail.json").write_text(json.dumps(detail, indent=1, default=str))
    (OUT_DIR / "scheme_meta.json").write_text(json.dumps(metas, indent=1, default=str))
    (OUT_DIR / "extra_metrics.json").write_text(json.dumps(extras, indent=1, default=str))
    (OUT_DIR / "noise_floor.json").write_text(json.dumps(floors, indent=1, default=str))
    (OUT_DIR / "verdict.json").write_text(json.dumps(verdicts, indent=1, default=str))
    (OUT_DIR / "not_run.json").write_text(json.dumps(not_run, indent=1, default=str))
    run_meta = {"run_at": time.strftime("%Y-%m-%d %H:%M"),
                "runtime_hours": round((time.time() - t_start) / 3600.0, 3),
                "budget_hours": a.budget_hours, "stages": sorted(stages),
                "n_cells": int(len(grid)), "n_not_run": len(not_run),
                "threads": N_JOBS, "design": dmeta}
    (OUT_DIR / "run_meta.json").write_text(json.dumps(run_meta, indent=1, default=str))

    if not a.no_manifest:
        # Refitting the selected cell to persist one object per refit date costs
        # another pass over its whole calendar. If the clock is already far past
        # budget the SCHEDULE is still written -- refit dates, train sizes and
        # the artifact paths the engine's selector expects -- with
        # `artifacts_written: false`, so the format is delivered and the fits can
        # be produced later without re-deciding anything.
        over = (a.manifest_no_artifacts
                or (time.time() - t_start) / 3600.0 > a.budget_hours + 1.0)
        write_manifest(design, conf, verdicts, OUT_DIR, write_artifacts=not over)

    if not a.no_append:
        EXPERIMENTS_MD.write_text(
            EXPERIMENTS_MD.read_text(encoding="utf-8")
            + render(grid, detail, extras, floors, verdicts, leak, not_run, run_meta),
            encoding="utf-8")
    print(f"done in {run_meta['runtime_hours']:.2f} h; {len(grid)} cells, "
          f"{len(not_run)} not run", flush=True)


if __name__ == "__main__":
    main()
