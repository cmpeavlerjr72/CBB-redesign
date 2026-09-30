"""Season-drift anchor `O` for the box-parallel possession_outcome S1 trainer (lane C, 2026-09-30).

Versioned sibling of `scripts/train_possession_outcome_s1_par_v1.py` (lane J), which is NOT
edited; this reuses J's design loading contract, `C.run_tasks`, the monthly S1 partition and
`score_populations`, and adds `--anchor O`:

  * population `first` (served arm lgbm): each fit gets a LightGBM init_score of
    `log L_c(t) - log Lbar_c` per class c, where `L_c(t)` is the as-of league share of class c
    in the current season before the chance's date (day 0 = prior season's end shares) from
    `cbb_sim.season_anchor.anchor_O(kind="multi")`, computed on the fold's `first` rows
    (train + test); prediction adds the same offset back (softmax of raw + offset). This is
    exactly season-drift round 1's arm `O` for po.
  * population `cont` (served arm cascade, a chain of sklearn logits that takes no offset):
    fitted UNCHANGED through J's own worker. Round 1 tested `O` on `first` only.
  * NO engine artifact directory is written with an anchor: the engine has no offset feed
    yet, so an anchored model dumped for serving would be served without its offset. The
    wrapper forces `--no-artifacts` and says so.
With `--team-rate-table` this is the Stage B `TO` arm (E3 team features + anchor O).
Ruling: `docs/models/season_drift/experiments.md` section 3.

    python scripts/train_possession_outcome_s1_par_anchor_v1.py --anchor O --fold F2 \
        --season 2025 --team-rate-table data/processed/team_rate_features_E3_v2.parquet \
        --out-root data/processed/models/engine_s1_TO_v1 --n-jobs 24
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
import train_par_common_v1 as C  # noqa: E402

C.pin_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_engine_event_round2 as B  # noqa: E402
import train_possession_outcome_s1_par_v1 as J  # noqa: E402
from cbb_sim import season_anchor as SA  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402

OFF_COLS = [f"_anc_off_{k}" for k in range(len(PO.CLASSES))]


def _worker_anchor(arm, seed, feats, rows, te_seg, meta, n_estimators=None):
    """One `first` refit with the anchor offset, single-threaded, own process."""
    C.pin_threads()
    import lightgbm as lgb
    assert arm == "lgbm", f"anchor O is defined for the lgbm arm, not {arm}"
    params = dict(PO.LgbmArm.PARAMS, n_jobs=1)
    if n_estimators:
        params["n_estimators"] = int(n_estimators)
    t = time.time()
    K = len(PO.CLASSES)
    init = SA.lgbm_init_score(rows[OFF_COLS].to_numpy(), K, list(range(K)))
    clf = lgb.LGBMClassifier(random_state=seed, **params)
    clf.fit(np.ascontiguousarray(rows[feats].to_numpy(dtype="float32")),
            rows["y"].to_numpy().astype(int), init_score=init)
    assert list(clf.classes_) == list(range(K)), clf.classes_
    pred = None
    if len(te_seg):
        it = SA.lgbm_init_score(te_seg[OFF_COLS].to_numpy(), K, list(range(K)))
        pred = SA.predict_proba_with_offset(
            clf, np.ascontiguousarray(te_seg[feats].to_numpy(dtype="float32")), it)
    return {"model": clf, "pred": pred, "key": f"anchorO|{arm}|{seed}|{len(rows)}",
            "fit_s": round(time.time() - t, 1), "meta": meta}


def add_anchor_columns(design: pd.DataFrame, fold: str, season: int) -> tuple[pd.DataFrame, dict]:
    d = design.copy()
    d[OFF_COLS] = 0.0
    m = ((d["population"] == "first") & d["season"].isin([*B.FOLD_TRAIN[fold], season])).to_numpy()
    sub = d.loc[m]
    num, den = SA.po_inputs(sub, len(PO.CLASSES))
    a = SA.anchor_O(sub["season"].to_numpy(), sub["game_date"].to_numpy(), num, den,
                    B.FOLD_TRAIN[fold], "multi")
    d.loc[m, OFF_COLS] = a.offset()
    return d, {"Lbar": a.meta["Lbar"],
               "prior_by_season": {str(k): v for k, v in a.meta["prior_by_season"].items()}}


def build_tasks_anchor(design, fold, season, win, seed, max_cuts=None, n_estimators=None):
    """J.build_tasks with the offset columns carried into the `first` fits and segments."""
    tasks, layout, cuts = J.build_tasks(design, fold, season, win, seed, max_cuts, n_estimators)
    d = design.copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    tr = d[(d["season"].isin(B.FOLD_TRAIN[fold])) & (d["population"] == "first")]
    te = layout["_te_first"]
    feats = PO.feature_set(win["first"]["feature_set"], "first")
    fit_cols = [*feats, "y", "season", *OFF_COLS]
    te_dates = te["game_date"]
    for key in [k for k in tasks if k.startswith("first_")]:
        cut = pd.Timestamp(key.split("_", 1)[1])
        before = (te_dates < cut).to_numpy()
        prior = te.loc[before, fit_cols]
        rows = tr[fit_cols] if not len(prior) else pd.concat([tr[fit_cols], prior], ignore_index=True)
        tasks[key] = dict(tasks[key], rows=rows, te_seg=te.loc[layout[key], [*feats, *OFF_COLS]],
                          _anchor=True)
    return tasks, layout, cuts


def _dispatch(_anchor=False, **kw):
    return _worker_anchor(**kw) if _anchor else J._worker(**kw)


def run(a) -> dict:
    t0 = time.time()
    out_root = Path(a.out_root)
    if a.team_rate_table:
        out_root = out_root / Path(a.team_rate_table).stem
    C.assert_fresh_out_dir(out_root)
    C.stamp_owner(out_root, "train_possession_outcome_s1_par_anchor_v1", vars(a))
    for s in [*B.FOLD_TRAIN[a.fold], a.season]:
        assert_not_sealed(s)
    if a.season != B.FOLD_TEST[a.fold]:
        raise SystemExit(f"fold {a.fold} tests season {B.FOLD_TEST[a.fold]}, not {a.season}")
    B.R2_VERDICT = Path(a.verdict)
    win = B.winners_from_verdict()
    if win["first"]["arm"] != "lgbm":
        raise SystemExit(f"anchor O needs the lgbm arm on `first`; verdict says {win['first']['arm']}")
    design = pd.read_parquet(a.design)
    trt = None
    if a.team_rate_table:
        from cbb_sim.team_rate_adapter import apply as team_rate_apply
        design = team_rate_apply(design, a.team_rate_table, "possession_outcome", fold=a.fold,
                                 missing=a.team_rate_missing)
        trt = {"path": str(a.team_rate_table), "sha256": C.sha256_file(Path(a.team_rate_table))}
    design["game_date"] = pd.to_datetime(design["game_date"])
    design, anc_meta = add_anchor_columns(design, a.fold, a.season)
    if a.test_n_estimators:
        PO.LgbmArm.PARAMS = dict(PO.LgbmArm.PARAMS, n_estimators=int(a.test_n_estimators))
    tasks, layout, cuts = build_tasks_anchor(design, a.fold, a.season, win, a.seed, a.max_cuts,
                                             int(a.test_n_estimators) or None)
    if a.only_pop:
        tasks = {k: v for k, v in tasks.items() if k.startswith(a.only_pop + "_")}
    print(f"[{time.time()-t0:.0f}s] {len(tasks)} fit tasks over {len(cuts)} refit dates "
          f"(anchor O on `first`)", flush=True)
    results = C.run_tasks(tasks, _dispatch, a.n_jobs, out_root / "cuts",
                          stop_at=pd.Timestamp(a.stop_at) if a.stop_at else None)
    if len(results) < len(tasks):
        print("INCOMPLETE; rerun the same command to resume", flush=True)
        return {"complete": False}
    rep = {"fold": a.fold, "season": a.season, "seed": a.seed, "scheme": "S1", "anchor": "O",
           "anchor_meta": anc_meta, "team_rate_table": trt, "n_jobs": a.n_jobs,
           "artifacts": "NOT WRITTEN (engine has no anchor-offset feed)",
           "test_hooks": {"max_cuts": a.max_cuts, "test_n_estimators": a.test_n_estimators,
                          "only_pop": a.only_pop}}
    if not a.max_cuts and not a.only_pop:
        scores, preds = J.score_populations(results, layout, win, cuts, a.fold)
        for pop, p in preds.items():
            np.save(out_root / f"pred_{pop}_{a.fold}.npy", p.astype("float32"))
        rep["scores"] = scores
    else:
        rep["smoke"] = {k: {"fit_s": r["fit_s"], "n_scored": r["meta"]["n_scored"],
                            "pred_rowsum_ok": bool(r["pred"] is None or np.allclose(r["pred"].sum(1), 1)),
                            "pred_mean": (None if r["pred"] is None else
                                          [round(float(x), 5) for x in r["pred"].mean(0)])}
                        for k, r in results.items()}
    (out_root / "par_anchor_v1_report.json").write_text(json.dumps(rep, indent=1, default=str),
                                                        encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k in ("scores", "smoke")}, default=str), flush=True)
    return {"complete": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", choices=["O"], required=True)
    ap.add_argument("--fold", default="F2", choices=sorted(B.FOLD_TRAIN))
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--design", default=str(B.R2_DESIGN))
    ap.add_argument("--verdict", default=str(B.R2_VERDICT))
    ap.add_argument("--team-rate-table", default="")
    ap.add_argument("--team-rate-missing", choices=["raise", "keep_served"], default="raise")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--stop-at", default="")
    ap.add_argument("--max-cuts", type=int, default=0, help="TEST HOOK")
    ap.add_argument("--test-n-estimators", type=int, default=0, help="TEST HOOK")
    ap.add_argument("--only-pop", choices=["", "first", "cont"], default="", help="TEST HOOK")
    a = ap.parse_args()
    return 0 if run(a).get("complete") else 3


if __name__ == "__main__":
    raise SystemExit(main())
