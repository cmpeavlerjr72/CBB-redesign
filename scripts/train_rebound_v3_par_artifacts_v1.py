"""train_rebound_v3_par_artifacts_v1.py -- versioned wrapper: lane J's box-parallel rebound trainer PLUS engine artifacts.
Lane G, 2026-09-30.  `train_rebound_v3_par_v1.py` (not edited) fits every weekly S1 cut but returns only predictions:
its worker drops the fitted model, so no rebound artifact a retrained arm could be SERVED from exists. This wrapper
swaps that one worker for a copy that also returns the model, then (after J's main) writes, per finished cell,

    <out-dir>/[<table stem>/]artifacts/<cell tag>/manifest.json + seg_<refit date>.joblib

in exactly the layout `ReboundAdapter._load_dated` reads (served copy: rebound/s1_confirm/S1_weekly/F2/). Each joblib
is {arm, feature_set, features, model (RB.LgbmArm holding the fitted classifier, class-aligned predict_proba), scheme,
fold, refit_date, max_train_date}. Every other argument is J's (`--stage 2 --folds F2 --arms A0B0C0 --n-jobs N --out-dir D
[--team-rate-table T] ...`) and passes through unchanged.
REFUSED: arms with an offset (A5, anchor O): the engine has no offset feed, a model served without its offset is wrong.
TEST HOOK (never on the box): env REB_TEST_N_ESTIMATORS=N trains N-tree models.
Only stage 2 (S1_weekly) writes artifacts.
"""
import os
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts")); sys.path.insert(0, str(_ROOT / "src"))
import train_rebound_v3_par_v1 as J  # noqa: E402  (pins threads)
import train_par_common_v1 as C  # noqa: E402
import train_rebound_v3_round3 as R3  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from cbb_sim.models import rebound as RB  # noqa: E402


def _worker_art(spec, feats, seed, pool, sub, feed_sub, refit, n_scored, params_override):
    """J._worker, plus the model and the pool's last date."""
    C.pin_threads()
    over = dict(params_override or {})
    if os.environ.get("REB_TEST_N_ESTIMATORS"):
        over["n_estimators"] = int(os.environ["REB_TEST_N_ESTIMATORS"])
    if over:
        R3.PARAMS = {**R3.PARAMS, **over}
    R3.PARAMS = {**R3.PARAMS, "n_jobs": 1}
    clf, m = R3.fit_arm(spec, pool, feats, seed)
    p = R3.predict_arm(spec, clf, sub, feats)
    pf = {k: R3.predict_with_feed(spec, clf, sub, feats, v) for k, v in feed_sub.items()}
    arm = RB.LgbmArm(seed)
    arm.clf_ = clf
    return {"p": p, "pf": pf, "seg": {"refit": refit, "n_scored": n_scored, **m},
            "model": arm, "max_train_date": str(pd.to_datetime(pool["game_date"]).max().date())}


def write_artifacts(out_dir: Path, stem: str, stage: int, folds: str, arms: str, seed: int, feats_of) -> list:
    base = out_dir / stem if stem else out_dir
    written = []
    for fold in folds.split(","):
        for arm in [a for a in arms.split(",") if a]:
            tag = f"s{stage}_{fold}_{arm}_seed{seed}"
            ck = base / "cuts" / tag
            files = sorted(ck.glob("*.pkl"))
            if not files:
                continue
            ad = base / "artifacts" / tag
            ad.mkdir(parents=True, exist_ok=True)
            entries = []
            for f in files:
                r = pickle.load(open(f, "rb"))
                refit = r["seg"]["refit"]
                name = f"seg_{refit}.joblib"
                joblib.dump({"arm": "lgbm", "feature_set": "C_plus_state", "features": list(feats_of(arm)),
                             "model": r["model"], "scheme": "S1_weekly", "fold": fold, "refit_date": refit,
                             "max_train_date": r["max_train_date"], "note": f"par_artifacts_v1 cell {tag}"}, ad / name)
                entries.append({"refit_date": refit, "path": name, "max_train_date": r["max_train_date"],
                                "n_train": int(r["seg"].get("n_train", 0))})
            entries.sort(key=lambda e: e["refit_date"])
            (ad / "manifest.json").write_text(json.dumps({"model": "rebound", "scheme": "S1_weekly", "fold": fold,
                                                          "season": 2025, "key": "C_plus_state", "artifacts": entries}, indent=2),
                                              encoding="utf-8")
            written.append(str(ad))
    return written


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--stage", type=int, default=1); ap.add_argument("--arms", default="")
    ap.add_argument("--folds", default="F2"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=""); ap.add_argument("--team-rate-table", default="")
    ap.add_argument("--mode", default="run")
    a, _ = ap.parse_known_args()
    bad = [x for x in a.arms.split(",") if x and R3.arm_spec(x).get("offset")]
    if bad:
        raise SystemExit(f"arms {bad} use an init_score offset; the engine cannot serve them (no offset feed)")
    J._worker = _worker_art
    rc = J.main()
    if rc in (0, None) and a.mode == "run" and a.stage == 2:
        stem = Path(a.team_rate_table).stem if a.team_rate_table else ""
        w = write_artifacts(Path(a.out_dir).resolve(), stem, a.stage, a.folds, a.arms, a.seed,
                            lambda arm: R3.features_for(R3.arm_spec(arm)))
        print("artifact dirs:", *w, sep="\n  ")
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
