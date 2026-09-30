"""train_possession_outcome_s1_par_anchor_artifacts_v1.py -- anchored possession_outcome S1 (season-drift anchor O)
PLUS the engine artifact directory. Lane M, 2026-09-30. Versioned sibling; nothing existing is edited.

`train_possession_outcome_s1_par_anchor_v1.py` (lane C) fits the anchored `first` population and the unchanged `cont`
population but forces `--no-artifacts`, because the engine had no offset feed. The engine now has one
(`ENGINE_SEASON_ANCHOR`, `src/cbb_sim/engine/season_anchor_serving.py`). This wrapper takes the SAME arguments as lane
C's trainer (plus `--engine-dir`, and the test hook `--sample-games`), runs the same fits through lane C's own
functions (`add_anchor_columns`, `build_tasks_anchor`, `_dispatch`), and then writes the engine artifact directory
through the UNMODIFIED round-2 builder with the pool's fits memoised, exactly as lane J's
`train_possession_outcome_s1_par_v1.py` does:

    <out-root>/[<table stem>/]event_round2_s1_F2_2025/{first_<date>.joblib x6, cont_<date>.joblib x6, index.json,
                                                        team_block.npz}

Each `first` joblib holds `PO.LgbmArm` wrapping the anchored LGBMClassifier and is MARKED
    "anchor": {"arm": "O", "kind": "multi", "cols": [0..5], "Lbar": [...], "prior_by_season": {...}}
(also in `index.json` populations.first.anchor). The engine refuses a marked artifact unless the per-game offsets
(`scripts/build_engine_anchor_offsets_v1.py`, family po) are fed. `cont` joblibs are unmarked (cascade, no offset).
`scripts/build_engine_inputs_v3_tag_v1.py --po-artifacts <that dir>` consumes it unchanged.

    python scripts/train_possession_outcome_s1_par_anchor_artifacts_v1.py --anchor O --fold F2 --season 2025 \
        --team-rate-table data/processed/team_rate_features_E3_v4.parquet --n-jobs 24 \
        --out-root data/processed/models/engine_s1_TO_art_v1
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
import shutil  # noqa: E402
import time  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_engine_event_round2 as B  # noqa: E402
import train_possession_outcome_s1_par_anchor_v1 as A  # noqa: E402
import train_possession_outcome_s1_par_v1 as J  # noqa: E402
from cbb_sim.data.seal import assert_not_sealed  # noqa: E402
from cbb_sim.models import possession_outcome as PO  # noqa: E402


def _t(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def mark_first(built: Path, anc_meta: dict) -> dict:
    mark = {"arm": "O", "kind": "multi", "cols": list(range(len(PO.CLASSES))), "classes": list(PO.CLASSES),
            "definition": "cbb_sim.season_anchor.anchor_O on the design's `first` rows; offset = log L_asof - log Lbar",
            "Lbar": anc_meta["Lbar"], "prior_by_season": anc_meta["prior_by_season"],
            "serve": "ENGINE_SEASON_ANCHOR=<anchor_offsets .npz with po_first>",
            "writer": "train_possession_outcome_s1_par_anchor_artifacts_v1"}
    idx = json.loads((built / "index.json").read_text(encoding="utf-8"))
    for sg in idx["populations"]["first"]["segments"]:
        p = built / sg["file"]
        w = joblib.load(p)
        assert list(w["model"].clf_.classes_) == list(range(len(PO.CLASSES)))
        w["anchor"] = dict(mark)
        joblib.dump(w, p, compress=3)
    idx["populations"]["first"]["anchor"] = dict(mark)
    (built / "index.json").write_text(json.dumps(idx, indent=1), encoding="utf-8")
    return mark


def run(a) -> dict:
    t0 = time.time()
    out_root = Path(a.out_root)
    if a.team_rate_table:
        out_root = out_root / Path(a.team_rate_table).stem
    C.assert_fresh_out_dir(out_root)
    C.stamp_owner(out_root, "train_possession_outcome_s1_par_anchor_artifacts_v1", vars(a))
    for s in [*B.FOLD_TRAIN[a.fold], a.season]:
        assert_not_sealed(s)
    if a.season != B.FOLD_TEST[a.fold]:
        raise SystemExit(f"fold {a.fold} tests season {B.FOLD_TEST[a.fold]}, not {a.season}")
    B.R2_VERDICT = Path(a.verdict)
    win = B.winners_from_verdict()
    if win["first"]["arm"] != "lgbm":
        raise SystemExit(f"anchor O needs the lgbm arm on `first`; verdict says {win['first']['arm']}")
    design = pd.read_parquet(a.design)
    design_path = Path(a.design)
    trt = None
    if a.team_rate_table:
        from cbb_sim.team_rate_adapter import apply as team_rate_apply
        design = team_rate_apply(design, a.team_rate_table, "possession_outcome", fold=a.fold,
                                 missing=a.team_rate_missing)
        trt = {"path": str(a.team_rate_table), "sha256": C.sha256_file(Path(a.team_rate_table))}
        design_path = out_root / "design_overlay.parquet"
        design.to_parquet(design_path, index=False)
    if a.sample_games:
        gid = np.sort(design["game_id"].unique())[:: a.sample_games]
        design = design[design["game_id"].isin(gid)].reset_index(drop=True)
        design_path = out_root / "design_sample.parquet"
        design.to_parquet(design_path, index=False)
        _t(f"TEST HOOK: design cut to every {a.sample_games}th game ({len(design):,} rows)", t0)
    design["game_date"] = pd.to_datetime(design["game_date"])
    design, anc_meta = A.add_anchor_columns(design, a.fold, a.season)
    if a.test_n_estimators:
        PO.LgbmArm.PARAMS = dict(PO.LgbmArm.PARAMS, n_estimators=int(a.test_n_estimators))
        _t(f"TEST HOOK: n_estimators -> {a.test_n_estimators} (never on the box)", t0)
    tasks, layout, cuts = A.build_tasks_anchor(design, a.fold, a.season, win, a.seed, a.max_cuts,
                                               int(a.test_n_estimators) or None)
    # the builder memo is keyed on lane J's fit identity of the rows each fit actually sees
    keys = {k: J.fit_key(t["arm"], t["seed"], t["rows"], t["feats"]) for k, t in tasks.items()}
    _t(f"{len(tasks)} fit tasks over {len(cuts)} refit dates (anchor O on `first`)", t0)
    results = C.run_tasks(tasks, A._dispatch, a.n_jobs, out_root / "cuts",
                          stop_at=pd.Timestamp(a.stop_at) if a.stop_at else None)
    if len(results) < len(tasks):
        _t("INCOMPLETE; rerun the same command to resume", t0)
        return {"complete": False}
    del tasks
    rep = {"fold": a.fold, "season": a.season, "seed": a.seed, "scheme": "S1", "anchor": "O",
           "anchor_meta": anc_meta, "team_rate_table": trt, "design": str(design_path), "n_jobs": a.n_jobs,
           "test_hooks": {"max_cuts": a.max_cuts, "test_n_estimators": a.test_n_estimators,
                          "sample_games": a.sample_games}}
    if not a.max_cuts:
        scores, preds = J.score_populations(results, layout, win, cuts, a.fold)
        for pop, p in preds.items():
            np.save(out_root / f"pred_{pop}_{a.fold}.npy", p.astype("float32"))
        rep["scores"] = scores
    built = None
    games_src = Path(a.engine_dir) / f"games_{a.fold}_{a.season}.parquet"
    if games_src.exists() and not a.max_cuts:
        shutil.copyfile(games_src, out_root / games_src.name)
        memo = {}
        for k, r in results.items():
            m = r["model"]
            if k.startswith("first_"):
                arm = PO.LgbmArm(a.seed)
                arm.clf_ = m
                m = arm
            memo[keys[k]] = m
        real_fit = PO.fit_arm

        def _memo_fit(arm, tr, features, seed=0, sample_weight=None):
            k = J.fit_key(arm, seed, tr, features)
            if sample_weight is not None or k not in memo:
                raise AssertionError(f"builder asked for a fit the pool did not run ({k}); refusing to fit serially")
            return memo[k]

        B.ENGINE_DIR = out_root
        B.R2_DESIGN = design_path
        PO.fit_arm = _memo_fit
        try:
            built = B.build(a.fold, a.season, a.seed)
        finally:
            PO.fit_arm = real_fit
        rep["anchor_mark"] = mark_first(built, anc_meta)
        _t(f"engine artifact dir (first MARKED anchored): {built}", t0)
    else:
        _t(f"no engine artifact dir written (no {games_src} or max-cuts hook)", t0)
    rep["artifact_dir"] = str(built) if built else None
    rep["wall_s"] = round(time.time() - t0, 1)
    (out_root / "par_anchor_artifacts_v1_report.json").write_text(json.dumps(rep, indent=1, default=str),
                                                                  encoding="utf-8")
    print(json.dumps({k: rep.get(k) for k in ("scores", "artifact_dir")}, default=str), flush=True)
    return {"complete": True, "built": built}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", choices=["O"], required=True)
    ap.add_argument("--fold", default="F2", choices=sorted(B.FOLD_TRAIN))
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--design", default=str(B.R2_DESIGN))
    ap.add_argument("--verdict", default=str(B.R2_VERDICT))
    ap.add_argument("--engine-dir", default=str(B.ENGINE_DIR))
    ap.add_argument("--team-rate-table", default="")
    ap.add_argument("--team-rate-missing", choices=["raise", "keep_served"], default="raise")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--stop-at", default="")
    ap.add_argument("--max-cuts", type=int, default=0, help="TEST HOOK (no artifacts)")
    ap.add_argument("--test-n-estimators", type=int, default=0, help="TEST HOOK")
    ap.add_argument("--sample-games", type=int, default=0, help="TEST HOOK: every Nth game")
    a = ap.parse_args()
    return 0 if run(a).get("complete") else 3


if __name__ == "__main__":
    raise SystemExit(main())
