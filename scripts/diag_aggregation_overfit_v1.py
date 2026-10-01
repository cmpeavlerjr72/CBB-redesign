"""diag_aggregation_overfit_v1.py -- lane A 2026-09-30, docs/models/aggregation/experiments.md section 2.2
(addendum E): in-sample (training seasons 2022-2024) vs out-of-sample (fold 2) team-game make-rate
calibration slope of the PRE-SEASON (2024-11-01) fg_make refit, Stage B `T` (E3 v4 features) vs `R`
(served features). DIAGNOSTIC ONLY.

The design is rebuilt exactly as `train_fg_make_v4_par_v1.py` builds it (same design file, same
team-rate adapter, same extra cache: the E3 cache rebuilt locally by the G0 run, which reproduces the
box `T` artifacts' harness margin exactly; the served cache for R).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import train_par_common_v1 as C  # noqa: E402

C.pin_threads()
import joblib  # noqa: E402
import train_fg_make_v4_shooter_block as R4M  # noqa: E402
from cbb_sim.models import fg_make as FG  # noqa: E402
from cbb_sim.team_rate_adapter import apply as trt_apply  # noqa: E402

HF = Path("C:/Users/devuser/AppData/Local/Temp/claude/C--Users-devuser-CBB-clean-sheet/"
          "f2e4a65e-7674-468c-ac10-92b191e2329c/scratchpad/hf/model_artifacts/fg_make/round_stageb")
ARMS = {
    "T": {"table": ROOT / "data/processed/team_rate_features_E3_v4.parquet",
          "cache": ROOT / "data/processed/models/fg_make/round_aggfix/design_v4_extra_E3_local_g0.parquet",
          "dir": ROOT / "data/processed/models/fg_make/round_aggfix/G0_seed0/team_rate_features_E3_v4/B1"},
    "R": {"table": None, "cache": R4M.EXTRA_CACHE, "dir": HF / "R_seed0/B1"},
}


def wslope(y, x, w):
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))


def main() -> int:
    out = {}
    base = pd.read_parquet(R4M.DESIGN)
    for arm, spec in ARMS.items():
        design = base.copy()
        if spec["table"] is not None:
            design = trt_apply(design, spec["table"], "fg_make", fold="F2", missing="raise")
        extra = pd.read_parquet(spec["cache"])
        for c in extra.columns:
            design[c] = extra[c].to_numpy()
        tr_all, te_all = FG.fold_slices(design, "F2")
        res = {}
        for cls in FG.SHOT_CLASSES:
            w = joblib.load(spec["dir"] / f"{cls}_2024-11-01.joblib")
            feats, model = w["features"], w["model"]
            for part, df in (("in_sample", FG.class_slice(tr_all, cls)), ("out_of_sample", FG.class_slice(te_all, cls))):
                p = model.predict_proba(np.ascontiguousarray(FG.design_matrix(df, feats), dtype=np.float32))[:, FG.CLASS_INDEX["MAKE"]]
                g = pd.DataFrame({"game_id": df["game_id"].to_numpy(), "team": df["off_team_id"].to_numpy(),
                                  "y": (df["y"].to_numpy() == FG.CLASS_INDEX["MAKE"]).astype(float),
                                  "p": p})
                agg = g.groupby(["game_id", "team"]).agg(y=("y", "mean"), p=("p", "mean"), n=("p", "size")).reset_index()
                agg = agg[agg["n"] >= 3]
                s = wslope(agg["y"].to_numpy(), agg["p"].to_numpy(), agg["n"].to_numpy(float))
                res[f"{cls}_{part}"] = {"slope": s, "n_team_games": int(len(agg)),
                                        "sd_pred": float(np.sqrt(np.cov(agg["p"], aweights=agg["n"]))),
                                        "mean_y": float(np.average(agg["y"], weights=agg["n"])),
                                        "mean_p": float(np.average(agg["p"], weights=agg["n"]))}
                print(f"{arm} {cls} {part}: slope {s:.3f} n {len(agg)} sd_pred {res[f'{cls}_{part}']['sd_pred']:.4f} "
                      f"mean y {res[f'{cls}_{part}']['mean_y']:.4f} p {res[f'{cls}_{part}']['mean_p']:.4f}", flush=True)
            res[f"{cls}_gap"] = res[f"{cls}_in_sample"]["slope"] - res[f"{cls}_out_of_sample"]["slope"]
        out[arm] = res
    out["T_minus_R_gap"] = {c: out["T"][f"{c}_gap"] - out["R"][f"{c}_gap"] for c in FG.SHOT_CLASSES}
    print("T gap - R gap:", {k: round(v, 3) for k, v in out["T_minus_R_gap"].items()})
    (ROOT / "results/aggregation_v1/analysis_overfit_v1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
