"""diag_train_serve_parity_fg_v1.py -- train/serve parity of fg_make's team-rate-DERIVED shooter feature (lane D, 2026-10-01).

The fg_make trainer writes, for every (shooter, game date), the `shooter_shrunk_dev_c__{rim,jump2,three}` it TRAINED on
(`<fg out>/slot_source_v2.parquet`). The engine inputs builder writes, for every (game, side, roster slot), the value it
SERVES (`slot_static`, keyed by `roster_cbbd`). This joins the two on (cbbd player id, game date) for the test season
and reports, per class: matched cells, correlation, mean / p99 / max |served - trained|.

Lane A (`docs/tests/aggregation_overspread_decomposition_2026-09-30.md`) found the E3 path trained on a stale raw rate
(served vs trained corr 0.94-0.97). The rule used as the chain's `parity` stage:
  FAIL if any class has corr < --min-corr (default 0.999) or p99 |diff| > --max-p99 (default 1e-3).
The thresholds were set from the F_R run (served path, no table), which must pass them as the reference
(docs/ops/full_retrain_chain_2026-09-30.md section 3).

    python scripts/diag_train_serve_parity_fg_v1.py --fg-out <fg out dir holding slot_source_v2.parquet> \
        --inputs <tagged input dir> [--out report.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLS = ("rim", "jump2", "three")
TAG = "F2_2025"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg-out", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--min-corr", type=float, default=0.999)
    ap.add_argument("--max-p99", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rb-arm", default="", help="also assert this rebound arm reads no rate-derived column the adapter leaves stale")
    a = ap.parse_args()
    tr = pd.read_parquet(a.fg_out / "slot_source_v2.parquet")
    tr["game_date"] = pd.to_datetime(tr["game_date"]).dt.normalize()
    z = np.load(a.inputs / f"arrays_{TAG}.npz")
    names = json.loads((a.inputs / f"names_{TAG}.json").read_text(encoding="utf-8"))
    games = pd.read_parquet(a.inputs / f"games_{TAG}.parquet")
    sn = names["slot_names"]
    ros, valid, sl = z["roster_cbbd"], z["roster_valid"], z["slot_static"]
    G, _, S = ros.shape
    gd = pd.to_datetime(games["game_date"]).dt.normalize().to_numpy()
    sv = pd.DataFrame({"shooter_id": ros.reshape(-1),
                       "game_date": np.repeat(gd, 2 * S),
                       "valid": valid.reshape(-1)})
    for c in CLS:
        sv[f"srv_{c}"] = sl[..., sn[f"shooter_shrunk_dev_c__{c}"]].reshape(-1).astype("float64")
    sv = sv[sv["valid"] & (sv["shooter_id"] > 0)].drop_duplicates(["shooter_id", "game_date"])
    m = sv.merge(tr, on=["shooter_id", "game_date"], how="inner")
    rep = {"fg_out": str(a.fg_out), "inputs": str(a.inputs), "served_cells": int(len(sv)), "matched": int(len(m)),
           "thresholds": {"min_corr": a.min_corr, "max_p99": a.max_p99}, "by_class": {}}
    fail = []
    for c in CLS:
        x, y = m[f"srv_{c}"].to_numpy(), m[f"shooter_shrunk_dev_c__{c}"].to_numpy(dtype="float64")
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        d = np.abs(x - y)
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")
        r = {"n": int(ok.sum()), "corr": round(corr, 6), "mean_abs": float(d.mean()),
             "p99_abs": float(np.quantile(d, 0.99)), "max_abs": float(d.max()), "share_exact": float((d == 0).mean())}
        rep["by_class"][c] = r
        if not (corr >= a.min_corr) or r["p99_abs"] > a.max_p99:
            fail.append(c)
    if a.rb_arm:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(root / "src"))
        import train_rebound_v3_round3 as R3
        from cbb_sim.team_rate_adapter_v2 import RB_STALE_DERIVED
        feats = list(R3.features_for(R3.arm_spec(a.rb_arm)))
        stale = [f for f in feats if f in RB_STALE_DERIVED]
        rep["rebound"] = {"arm": a.rb_arm, "features": feats, "stale_derived_read": stale}
        if stale:
            fail.append("rebound")
    rep["verdict"] = "FAIL" if fail else "PASS"
    rep["failing_classes"] = fail
    js = json.dumps(rep, indent=1)
    print(js)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(js)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
