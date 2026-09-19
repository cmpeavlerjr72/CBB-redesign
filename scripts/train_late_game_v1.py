"""train_late_game_v1.py -- L7 late-game regime, round 1: the EVENT half.

Pre-registration: `docs/models/late_game/experiments.md` section 1 and the
section-2 amendment (both committed BEFORE this file ran).  Design:
`scripts/build_late_game_design_v1.py`.  Grading:
`scripts/grade_late_game_v1.py` -- this trainer computes NO metric and writes
NO verdict.  It fits cells and writes probability matrices; one grading script
scores every cell by the same code path.

THE FOUR ARMS (experiments.md 1.2, arm D's reading fixed by the amendment):

  A  reference       fit on ALL regulation possessions, bundle `L0_reference`
                     (= the served `C_plus_state`), served model classes
                     (lgbm on `first`, cascade on `cont`).
  B  state enrichment  the same pooled fit on ALL regulation possessions with
                     the late-game columns added: bundles L1..L4.
  C  regime refit    the same model classes and bundles, fitted on WINDOW rows
                     only, served behind the gate: bundles L0..L4.
  D  dedicated       a model that exists only for the window: populations
                     POOLED (with `chance_number` and `is_cont` as features)
                     and the conditional law estimated SEPARATELY PER ROLE
                     (trailing / tied / leading).  That is the direct test of
                     "the regime is a conditional law a pooled fit averages
                     away"; bundles L3_gates and L4_team.

ALL FOUR ARE GRADED ON WINDOW ROWS ONLY, by `grade_late_game_v1.py`.

CADENCE.  Every arm's primary cell is S0 (static), so the grid varies the
CONDITIONING and holds the refit cadence fixed -- an arm cannot win the primary
by being refit more often than its reference.  The Decision-9 cadence cell (S1
monthly, the served scheme) is run for arm A and for whichever arm the grader
nominates, fold 2 only, by `--cadence s1`.

HYPERPARAMETERS.  Two fixed sets, both declared in the amendment before any fit
and neither searched: `PO.LgbmArm.PARAMS` for every fit on the full design, and
`WINDOW_PARAMS` for every fit on window rows only (50k rows cannot carry
`min_child_samples=400`; using the full-data value there would sandbag arms C
and D rather than test them).  Both are identical across every bundle, fold and
seed within their half of the grid.

SEEDS.  Seed 0 is the arm; seed 1 is its own spec-identical retrain, the
offline noise floor.  Deterministic arms (the cascade's logits) are unmoved by
a seed, so the grader ALSO computes a game-level block-bootstrap SE and takes
the larger of the two as the binding floor.

SEAL.  `PO.fold_slices` guards every slice; season 2026 can never enter a fold.

Usage:
  train_late_game_v1.py [--shard k --nshards n] [--folds F1,F2] [--cadence s0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from cbb_sim.data.seal import assert_not_sealed                      # noqa: E402
from cbb_sim.models import possession_outcome as PO                  # noqa: E402

import importlib.util as _ilu                                        # noqa: E402
_spec = _ilu.spec_from_file_location(
    "build_late_game_design_v1", ROOT / "scripts" / "build_late_game_design_v1.py")
LG = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(LG)

DESIGN = ROOT / "data/processed/models/late_game/design_v1.parquet"
OUT = ROOT / "data/processed/models/late_game/round1"

#: The window-only hyperparameter set.  FIXED before any fit, shared by every
#: window-only arm (C and D), every bundle, fold and seed.  Not searched.
WINDOW_PARAMS = dict(
    objective="multiclass", num_class=len(PO.CLASSES), n_estimators=300,
    learning_rate=0.05, num_leaves=15, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
    reg_lambda=1.0, verbose=-1,
)

SEEDS = (0, 1)
ROLES = (-1.0, 0.0, 1.0)
POPS = ("first", "cont")
#: The served model class per population (possession_outcome round 2 winners).
SERVED_CLASS = {"first": "lgbm", "cont": "cascade"}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ===========================================================================
# 1. The cell grid
# ===========================================================================
def cells(folds: tuple[str, ...], cadence: str) -> list[dict]:
    out = []
    for fold in folds:
        for seed in SEEDS:
            out.append({"arm": "A", "bundle": "L0_reference", "scope": "full",
                        "fold": fold, "seed": seed, "cadence": cadence})
            for b in ("L1_role", "L2_role_poss", "L3_gates", "L4_team"):
                out.append({"arm": "B", "bundle": b, "scope": "full",
                            "fold": fold, "seed": seed, "cadence": cadence})
            for b in LG.BUNDLES:
                out.append({"arm": "C", "bundle": b, "scope": "window",
                            "fold": fold, "seed": seed, "cadence": "s0"})
            for b in ("L3_gates", "L4_team"):
                out.append({"arm": "D", "bundle": b, "scope": "window_role",
                            "fold": fold, "seed": seed, "cadence": "s0"})
    for i, c in enumerate(out):
        c["cell_id"] = "c%03d" % i
        c["tag"] = "{arm}_{bundle}_{fold}_s{seed}_{cadence}".format(**c)
    return out


class WindowLgbm:
    """LightGBM multiclass under `WINDOW_PARAMS`.  Same class as
    `PO.LgbmArm`, different fixed parameter set for the small window fits."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y):
        import lightgbm as lgb
        self.clf_ = lgb.LGBMClassifier(random_state=self.seed, **WINDOW_PARAMS)
        self.clf_.fit(X, y)
        return self

    def predict_proba(self, X):
        return PO._align_classes(self.clf_.predict_proba(X), self.clf_.classes_)


def _fit_predict_full(arm_class: str, scheme: str, tr: pd.DataFrame,
                      te: pd.DataFrame, feats: list[str], seed: int):
    """Arms A and B: fit on the whole training slice, predict the window test
    rows.  `te` is already restricted to the window -- S1 needs the FULL test
    season to build its month partition, so the caller passes the full test
    slice and this function sub-selects afterwards."""
    if scheme == "s0":
        m = PO.fit_arm(arm_class, tr, feats, seed=seed)
        return m
    raise KeyError(scheme)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--folds", default="F1,F2")
    ap.add_argument("--cadence", default="s0")
    ap.add_argument("--only", default="",
                    help="comma-separated cell ids; overrides --shard/--nshards. "
                         "Used to re-dispatch leftovers across free workers; the "
                         "cell id is a pure function of the grid, so a cell run "
                         "this way is the same cell.")
    ap.add_argument("--pops", default="first,cont",
                    help="populations to FIT and score for the full/window "
                         "scopes. `first` alone leaves continuation rows NaN; "
                         "the first-chance predictions are BIT-IDENTICAL either "
                         "way, because each population is fitted on its own rows "
                         "only. Used under a wall-clock deadline; the graded "
                         "population is then declared and identical for every arm.")
    ap.add_argument("--meta-suffix", default="",
                    help="suffix for this process's own cells_shard*.json, so two "
                         "processes never write the same metadata file")
    a = ap.parse_args()
    folds = tuple(x.strip() for x in a.folds.split(",") if x.strip())
    pops = tuple(x.strip() for x in a.pops.split(",") if x.strip())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pred").mkdir(exist_ok=True)

    # Only the columns a fit, a gate or the canonical index reads.  Four shards
    # share one 20-core box with other lanes, and the full-width 3M-row design
    # is several GB per process; pruning cannot change a fitted model, because
    # `fit_arm` reads exactly `features` and `y`.
    need = {"season", "game_id", "game_date", "period", "poss_index",
            "chance_number", "population", "y", "points",
            "offense_team_id", "defense_team_id", "in_bonus", "is_transition",
            "in_window", "in_window_sens1", "in_window_sens2", "is_regulation",
            "seconds_remaining", "score_diff", "role"}
    for b in LG.BUNDLES:
        need |= set(LG.bundle(b, "first")) | set(LG.bundle(b, "cont"))
    log(f"loading {DESIGN.name} ({len(need)} columns)")
    d = pd.read_parquet(DESIGN, columns=sorted(need))
    d["game_date"] = pd.to_datetime(d["game_date"])
    d = d[d["is_regulation"].to_numpy()].reset_index(drop=True)
    d["is_cont"] = (d["population"].to_numpy() == "cont").astype("float32")
    log(f"  {len(d):,} regulation rows, {int(d['in_window'].sum()):,} in window")

    # --- the canonical test index, one per fold ----------------------------
    # It is the WIDEST sensitivity gate (150 s, 8 pts), which is a strict
    # superset of the selection gate (120 s, 6 pts) and of the narrow
    # sensitivity gate (90 s, 5 pts), PLUS a fixed random sample of
    # reference-window rows.  Predicting the widest gate makes R7 (cut-point
    # robustness) computable without a second pass; the reference sample makes
    # R6 (no upstream damage) computable for the arms that refit on all
    # possessions.  Selection is still on the (120, 6) subset only.
    order = ["game_id", "period", "poss_index", "chance_number"]
    idx, ref_n = {}, 60_000
    for fold in folds:
        spec = PO.FOLDS[fold]
        assert_not_sealed(spec["train"], context=f"{fold} train seasons")
        assert_not_sealed(spec["test"], context=f"{fold} test seasons")
        tef = d[d["season"].isin(spec["test"])]
        win = tef[tef["in_window_sens1"].to_numpy()]
        rest = tef[~tef["in_window_sens1"].to_numpy()]
        rng = np.random.default_rng(20260918)
        take = rng.choice(len(rest), size=min(ref_n, len(rest)), replace=False)
        ref = rest.iloc[np.sort(take)]
        te = pd.concat([win.assign(is_reference=False),
                        ref.assign(is_reference=True)])
        te = te.sort_values(order + ["is_reference"], kind="stable")
        idx[fold] = te.index.to_numpy()
        p = OUT / f"window_test_{fold}.parquet"
        if not p.exists():
            keep = [c for c in te.columns if not c.startswith("x_off_")]
            te[keep].to_parquet(p, index=False)
        log(f"  {fold}: scored rows {len(te):,} "
            f"(selection gate {int(te['in_window'].sum()):,}, "
            f"sens1 {int((~te['is_reference']).sum()):,}, reference {len(ref):,})")

    grid = cells(folds, a.cadence)
    if a.only:
        want_ids = [x.strip() for x in a.only.split(",") if x.strip()]
        mine = [c for c in grid if c["cell_id"] in want_ids]
        log(f"explicit cells: {len(mine)} of {len(grid)}")
    else:
        mine = grid[a.shard::a.nshards]
        log(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(grid)} cells")

    meta_path = OUT / f"cells_shard{a.shard}{a.meta_suffix}.json"
    # Resume: a cell whose prediction matrix and metadata already exist is not
    # refitted.  Cell ids are a pure function of the grid, so a resumed cell is
    # the same cell; this only stops paying twice for it.
    prior = {}
    for f in OUT.glob("cells_shard*.json"):
        for r in json.loads(f.read_text(encoding="utf-8")):
            prior[r["cell_id"]] = r
    done = []
    for c in mine:
        t0 = time.time()
        if c["cell_id"] in prior and (OUT / "pred" / f"{c['cell_id']}.npy").exists():
            done.append(prior[c["cell_id"]])
            meta_path.write_text(json.dumps(done, indent=1, default=str), encoding="utf-8")
            log(f"  {c['cell_id']} {c['tag']}: already fitted, reused")
            continue
        fold, seed, bundle = c["fold"], c["seed"], c["bundle"]
        spec = PO.FOLDS[fold]
        te_all = d[d["season"].isin(spec["test"])]
        tr_all = d[d["season"].isin(spec["train"])]
        assert_not_sealed(tr_all, context=f"{fold} train slice")
        assert_not_sealed(te_all, context=f"{fold} test slice")
        want = idx[fold]
        P = np.full((len(want), len(PO.CLASSES)), np.nan, dtype="float64")
        pos = pd.Series(np.arange(len(want)), index=want)
        info = {"n_fits": 0, "n_train": {}, "max_train_date": {}}
        # A `full`-scope arm is served everywhere, so it is scored everywhere
        # (R6 needs its reference-window rates).  A window-only arm would be
        # GATED OFF outside the window in the engine, so scoring it there would
        # be scoring a model nobody would ever run; those rows stay NaN and the
        # grader records R6 as satisfied by construction.
        scored = (d.index.isin(want) if c["scope"] == "full"
                  else d.index.isin(want) & d["in_window_sens1"].to_numpy())
        te_scored = d[scored & d["season"].isin(spec["test"]).to_numpy()]

        if c["scope"] in ("full", "window"):
            for pop in pops:
                feats = LG.bundle(bundle, pop)
                if c["scope"] == "full":
                    tr = tr_all[tr_all["population"] == pop]
                else:
                    tr = tr_all[(tr_all["population"] == pop)
                                & tr_all["in_window"].to_numpy()]
                te_rows = te_scored[te_scored["population"] == pop]
                if not len(te_rows):
                    continue
                if c["scope"] == "full":
                    model = PO.fit_arm(SERVED_CLASS[pop], tr, feats, seed=seed)
                    p = PO.predict_arm(SERVED_CLASS[pop], model, te_rows, feats)
                else:
                    X = np.ascontiguousarray(tr[feats].to_numpy(dtype="float32"))
                    y = tr["y"].to_numpy()
                    if SERVED_CLASS[pop] == "lgbm":
                        model = WindowLgbm(seed=seed).fit(X, y)
                    else:
                        model = PO.CascadeArm(seed=seed).fit(X, y)
                    p = model.predict_proba(
                        np.ascontiguousarray(te_rows[feats].to_numpy(dtype="float32")))
                P[pos.loc[te_rows.index].to_numpy()] = p
                info["n_fits"] += 1
                info["n_train"][pop] = int(len(tr))
                info["max_train_date"][pop] = str(pd.Timestamp(tr["game_date"].max()).date())

        elif c["scope"] == "window_role":
            feats = LG.bundle(bundle, "cont") + ["is_cont"]
            trw = tr_all[tr_all["in_window"].to_numpy()]
            tew = te_scored
            for r in ROLES:
                tr = trw[trw["role"].to_numpy() == r]
                te_rows = tew[tew["role"].to_numpy() == r]
                if not len(te_rows):
                    continue
                X = np.ascontiguousarray(tr[feats].to_numpy(dtype="float32"))
                model = WindowLgbm(seed=seed).fit(X, tr["y"].to_numpy())
                p = model.predict_proba(
                    np.ascontiguousarray(te_rows[feats].to_numpy(dtype="float32")))
                P[pos.loc[te_rows.index].to_numpy()] = p
                info["n_fits"] += 1
                info["n_train"]["role%+d" % int(r)] = int(len(tr))
                info["max_train_date"]["role%+d" % int(r)] = \
                    str(pd.Timestamp(tr["game_date"].max()).date())
        else:
            raise KeyError(c["scope"])

        if c["scope"] in ("full", "window") and set(pops) != set(POPS):
            te_scored = te_scored[te_scored["population"].isin(pops)]
        need = np.isin(want, te_scored.index.to_numpy())
        if np.isnan(P[need]).any():
            raise AssertionError(f"{c['tag']}: rows this arm must score were left unscored")
        c["n_scored"] = int(need.sum())
        c["populations_fitted"] = list(pops)
        np.save(OUT / "pred" / f"{c['cell_id']}.npy", P.astype("float32"))
        c.update(info)
        c["features"] = (LG.bundle(bundle, "cont") + ["is_cont"]
                         if c["scope"] == "window_role" else LG.bundle(bundle, "first"))
        c["runtime_s"] = round(time.time() - t0, 1)
        done.append(c)
        meta_path.write_text(json.dumps(done, indent=1, default=str), encoding="utf-8")
        log(f"  {c['cell_id']} {c['tag']}: {info['n_fits']} fits, {c['runtime_s']}s")

    log(f"shard {a.shard} done, {len(done)} cells -> {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
