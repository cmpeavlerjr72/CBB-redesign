"""grade_late_game_r4_clock_offline_v1.py -- late-game ROUND 4 (b), offline line (experiments.md 9.2).

Composites on round 1's held-out window rows, both folds, one code path:
    A      served-family law everywhere
    Dt     D on tied rows, A elsewhere            (round 3)
    DtL    D on tied rows, LGL on leading rows, A on trailing rows
    LGLall LGL on every row (reported)
CRPS of the horn-truncated law on uncensored rows and censored log-likelihood, per role and role-band x
bucket; floor = game-block bootstrap SE of the paired delta vs A. GUARD: on leading rows LGL must not be worse
than A by > 1 floor on F2.

    .venv/Scripts/python.exe scripts/grade_late_game_r4_clock_offline_v1.py --out results/late_game/round4/clock_offline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cbb_sim.models import clock as ck                               # noqa: E402
from cbb_sim.models import clock_v3 as c3                            # noqa: E402
from grade_late_game_v1 import CK_DIR, block_se, cell_label         # noqa: E402

R4 = ROOT / "data/processed/models/late_game/round4"
BANDS = (("trail_4_6", -6, -4), ("trail_1_3", -3, -1), ("tied", 0, 0), ("lead_1_3", 1, 3), ("lead_4_6", 4, 6))
BUCKETS = (("(60,120]", 60, 120), ("(30,60]", 30, 60), ("(10,30]", 10, 30), ("(0,10]", -1, 10))


def grade(fold: str) -> dict:
    te = pd.read_parquet(CK_DIR / f"window_test_{fold}.parquet")
    y = te["duration_s"].to_numpy(dtype="int64")
    cen = te["censored"].to_numpy(dtype=bool)
    rl = te["seconds_remaining"].to_numpy(dtype="float64")
    g = te["game_id"].to_numpy()
    sd = te["score_diff"].to_numpy()
    role = np.sign(sd)
    unc = ~cen
    A = np.load(CK_DIR / "pmf" / f"A_clk_{fold}.npy").astype("float64")
    D = np.load(CK_DIR / "pmf" / f"D_clk_{fold}.npy").astype("float64")
    L = np.load(R4 / f"clk_LGL_{fold}.npy").astype("float64")
    arms = {"A": A, "Dt": np.where((role == 0)[:, None], D, A),
            "DtL": np.where((role == 0)[:, None], D, np.where((role > 0)[:, None], L, A)), "LGLall": L,
            "DtLL": np.where((role == 0)[:, None], D, L)}   # round 5 (experiments.md s11)
    sc = {}
    for k, pmf in arms.items():
        tp, _ = c3.truncate_pmf(pmf, rl)
        cr = np.full(len(te), np.nan)
        cr[unc] = ck.crps(tp[unc], y[unc])
        sc[k] = {"crps": cr, "ll": c3.censored_loglik_rows(pmf, y, cen, rl), "mt": ck.pmf_mean_sd(tp)[0]}
    out = {"fold": fold, "n": int(len(te)), "arms": {}}
    for k in arms:
        row = {}
        for scope, m in [("all", np.ones(len(te), bool)), ("trailing", role < 0), ("tied", role == 0), ("leading", role > 0)]:
            d = np.where(m, sc[k]["crps"] - sc["A"]["crps"], np.nan)
            dl = np.where(m, sc[k]["ll"] - sc["A"]["ll"], np.nan)
            se = block_se(d, g) if k != "A" else None
            sel = block_se(dl, g) if k != "A" else None
            row[scope] = {"n": int(m.sum()), "crps": float(np.nanmean(np.where(m, sc[k]["crps"], np.nan))),
                          "d_crps": float(np.nanmean(d)), "floors_crps": (-float(np.nanmean(d)) / se) if se else None,
                          "d_ll": float(np.nanmean(dl)), "floors_ll": (float(np.nanmean(dl)) / sel) if sel else None}
        prof = {}
        for bn, lo, hi in BANDS:
            for kn, blo, bhi in BUCKETS:
                m = unc & (sd >= lo) & (sd <= hi) & (rl > blo) & (rl <= bhi)
                n = int(m.sum())
                prof[f"{bn}|{kn}"] = {"n": n, "label": cell_label(n),
                                      "pred": float(sc[k]["mt"][m].mean()) if n else None,
                                      "actual": float(y[m].mean()) if n else None}
        row["profile"] = prof
        out["arms"][k] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {f: grade(f) for f in ("F1", "F2")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for f, r in res.items():
        print(f"== {f}")
        for k, row in r["arms"].items():
            print(f"  {k:7s} " + "  ".join(
                f"{s}: {row[s]['crps']:.4f} ({(row[s]['floors_crps'] or 0):+.2f} fl, ll {(row[s]['floors_ll'] or 0):+.2f})"
                for s in ("all", "trailing", "tied", "leading")))
        for cell in r["arms"]["A"]["profile"]:
            if not (cell.startswith("lead") or cell.startswith("trail")):
                continue
            print(f"   {cell:20s} act {r['arms']['A']['profile'][cell]['actual'] or 0:6.2f} "
                  + " ".join(f"{k} {r['arms'][k]['profile'][cell]['pred'] or 0:6.2f}" for k in r["arms"])
                  + f" n={r['arms']['A']['profile'][cell]['n']} {r['arms']['A']['profile'][cell]['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
