"""diag_full_retrain_identity_v1.py -- does a sibling trainer, run with the OLD inputs, reproduce the SERVED artifact?

    python scripts/diag_full_retrain_identity_v1.py po --new <event dir> [--served data/processed/models/engine/event_round2_s1_F2_2025]
    python scripts/diag_full_retrain_identity_v1.py fg --new <B1 dir>    [--served data/processed/models/fg_make/round4/B1]
    python scripts/diag_full_retrain_identity_v1.py rb --new <artifact dir> [--served data/processed/models/rebound/s1_confirm/S1_weekly/F2]

For every artifact present in BOTH (matched by refit date): LightGBM booster text equality (`model_to_string`, the
strongest test), else the max |p_new - p_served| on a fixed 20,000-row sample of the model's own design (season 2025
rows). Writes a JSON next to --new and prints a table. Exit 0 always (the doc states the verdict and the tolerance).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

M = ROOT / "data/processed/models"


def booster_text(model):
    clf = getattr(model, "clf_", None)
    if clf is None or not hasattr(clf, "booster_"):
        return None
    return clf.booster_.model_to_string()


def _strip_header(s: str | None):
    # the text dump carries the training parameters line (n_jobs differs between a serial and a 1-thread fit)
    if s is None:
        return None
    return "\n".join(x for x in s.splitlines() if not x.startswith("[num_threads") and not x.startswith("[n_jobs"))


def cmp_pair(a, b, X_fn):
    ta, tb = booster_text(a), booster_text(b)
    out = {"booster_text_equal": (ta == tb) if ta is not None and tb is not None else None}
    if ta is not None and tb is not None and ta != tb:
        out["booster_text_equal_ignoring_thread_params"] = _strip_header(ta) == _strip_header(tb)
    if X_fn is not None:
        pa, pb = X_fn(a), X_fn(b)
        out["max_abs_pred_diff"] = float(np.abs(np.asarray(pa, dtype="float64") - np.asarray(pb, dtype="float64")).max())
    return out


def po(new: Path, served: Path) -> dict:
    from cbb_sim.models import possession_outcome as PO
    design = pd.read_parquet(M / "possession_outcome/round2/design.parquet")
    design = design[design["season"] == 2025]
    res = {}
    for pop in ("first", "cont"):
        samp = design[design["population"] == pop].sample(20000, random_state=11)
        for f in sorted(served.glob(f"{pop}_*.joblib")):
            g = new / f.name
            if not g.exists():
                continue
            A, B = joblib.load(f), joblib.load(g)
            feats = A["features"]
            r = cmp_pair(A["model"], B["model"], lambda m: PO.predict_arm(A["arm"], m, samp, feats))
            res[f.name] = r
    return res


def fg(new: Path, served: Path) -> dict:
    from cbb_sim.models import fg_make as FG
    import train_fg_make_v4_shooter_block as R4M
    design = pd.read_parquet(R4M.DESIGN)
    extra = pd.read_parquet(R4M.EXTRA_CACHE)
    for c in extra.columns:
        design[c] = extra[c].to_numpy()
    design = design[design["season"] == 2025]
    res = {}
    for f in sorted(served.glob("FGA_*.joblib")):
        g = new / f.name
        if not g.exists():
            continue
        A, B = joblib.load(f), joblib.load(g)
        cls = A["shot_class"]
        samp = FG.class_slice(design, cls)
        samp = samp.sample(min(20000, len(samp)), random_state=11)
        X = FG.design_matrix(samp, A["features"])
        res[f.name] = cmp_pair(A["model"], B["model"], lambda m: m.predict_proba(X)[:, FG.CLASS_INDEX["MAKE"]])
    return res


def rb(new: Path, served: Path) -> dict:
    import train_rebound_v3_round3 as R3
    design = pd.read_parquet(R3.DESIGN)
    design = design[design["season"] == 2025].sample(20000, random_state=11)
    ms = json.loads((served / "manifest.json").read_text())["artifacts"]
    mn = json.loads((new / "manifest.json").read_text())["artifacts"]
    by_date = {e["refit_date"]: e for e in mn}
    res = {}
    for e in ms:
        if e["refit_date"] not in by_date:
            continue
        A = joblib.load(served / e["path"])
        B = joblib.load(new / by_date[e["refit_date"]]["path"])
        feats = A["features"]
        if list(B["features"]) != list(feats):
            res[e["refit_date"]] = {"features_differ": [list(feats), list(B["features"])]}
            continue
        X = design[feats].to_numpy(dtype="float32")
        r = cmp_pair(A["model"], B["model"], lambda m: m.predict_proba(X))
        r.update({"n_train_served": e.get("n_train"), "n_train_new": by_date[e["refit_date"]].get("n_train"),
                  "max_train_served": e.get("max_train_date"), "max_train_new": by_date[e["refit_date"]].get("max_train_date")})
        res[e["refit_date"]] = r
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["po", "fg", "rb"])
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--served", type=Path, default=None)
    a = ap.parse_args()
    served = a.served or {"po": M / "engine/event_round2_s1_F2_2025", "fg": M / "fg_make/round4/B1",
                          "rb": M / "rebound/s1_confirm/S1_weekly/F2"}[a.family]
    res = {"po": po, "fg": fg, "rb": rb}[a.family](a.new, served)
    out = {"family": a.family, "new": str(a.new), "served": str(served), "n_compared": len(res), "results": res}
    (a.new / f"identity_vs_served_{a.family}.json").write_text(json.dumps(out, indent=1, default=str))
    for k, v in res.items():
        print(f"{k:32s} {v}")
    vals = [v.get("max_abs_pred_diff") for v in res.values() if isinstance(v, dict) and v.get("max_abs_pred_diff") is not None]
    print(f"compared {len(res)}; booster text equal {sum(1 for v in res.values() if v.get('booster_text_equal'))}; "
          f"max |dp| over all {max(vals) if vals else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
