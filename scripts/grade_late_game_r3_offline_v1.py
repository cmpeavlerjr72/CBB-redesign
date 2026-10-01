"""grade_late_game_r3_offline_v1.py -- late-game ROUND 3, offline line (experiments.md section 7.5).

The round fits nothing. Its offline evidence is the COMPOSED window duration law on round 1's held-out
window rows, both folds: A = round 1's A_clk (served-family P3 / floor-5 law), D = D_clk on every row,
Dt = D on tied offence rows and A elsewhere, Dtt = D on tied and trailing rows and A elsewhere.
One code path grades all four; the arm's identity only labels the row.

Metric: round 1's duration primary (CRPS of the horn-truncated law on uncensored rows) and the censored
log-likelihood, overall and per role. Floor: game-level block-bootstrap SE of the PAIRED per-row delta vs A
(Kaplan-Meier laws are deterministic; the reseed floor is zero, section 2.5.3).
GUARD (7.5): a candidate worse than A on fold 2 by more than one floor on the rows it replaces is out.

    .venv/Scripts/python.exe scripts/grade_late_game_r3_offline_v1.py --out results/late_game/round3/offline.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from cbb_sim.models import clock as ck                               # noqa: E402
from cbb_sim.models import clock_v3 as c3                            # noqa: E402
from grade_late_game_v1 import CK_DIR, block_se, cell_label         # noqa: E402

ROLES = {"trailing": -1, "tied": 0, "leading": 1}
ARMS = {"A": None, "D": (-1, 0, 1), "Dt": (0,), "Dtt": (0, -1)}
BUCKETS = (("(60,120]", 60, 120), ("(30,60]", 30, 60), ("(10,30]", 10, 30), ("(0,10]", -1, 10))


def grade_fold(fold: str) -> dict:
    te = pd.read_parquet(CK_DIR / f"window_test_{fold}.parquet")
    y = te["duration_s"].to_numpy(dtype="int64")
    cen = te["censored"].to_numpy(dtype=bool)
    rleft = te["seconds_remaining"].to_numpy(dtype="float64")
    games = te["game_id"].to_numpy()
    role = np.sign(te["score_diff"].to_numpy())
    unc = ~cen
    pa = np.load(CK_DIR / "pmf" / f"A_clk_{fold}.npy").astype("float64")
    pd_ = np.load(CK_DIR / "pmf" / f"D_clk_{fold}.npy").astype("float64")
    per_row: dict = {}
    trunc_mean: dict = {}
    for arm, gate in ARMS.items():
        use_d = np.zeros(len(te), dtype=bool) if gate is None else np.isin(role, gate)
        pmf = np.where(use_d[:, None], pd_, pa)
        tp, _ = c3.truncate_pmf(pmf, rleft)
        cr = np.full(len(te), np.nan)
        cr[unc] = ck.crps(tp[unc], y[unc])
        ll = c3.censored_loglik_rows(pmf, y, cen, rleft)
        per_row[arm] = {"crps": cr, "ll": ll, "replaced": use_d}
        trunc_mean[arm] = ck.pmf_mean_sd(tp)[0]
    out: dict = {"fold": fold, "n_rows": int(len(te)), "n_uncensored": int(unc.sum()),
                 "n_games": int(len(np.unique(games))), "arms": {}}
    for arm in ARMS:
        r = per_row[arm]
        a = per_row["A"]
        row: dict = {}
        for scope, m in [("all", np.ones(len(te), bool))] + [(k, role == v) for k, v in ROLES.items()]:
            n_unc = int((m & unc).sum())
            d_cr = np.where(m, r["crps"] - a["crps"], np.nan)
            d_ll = np.where(m, r["ll"] - a["ll"], np.nan)
            se_cr = block_se(d_cr, games) if arm != "A" else None
            se_ll = block_se(d_ll, games) if arm != "A" else None
            dcr = float(np.nanmean(d_cr)) if n_unc else None
            dll = float(np.nanmean(d_ll))
            row[scope] = {
                "n": int(m.sum()), "n_uncensored": n_unc, "label": cell_label(n_unc),
                "crps": float(np.nanmean(np.where(m, r["crps"], np.nan))) if n_unc else None,
                "censored_loglik": float(np.mean(r["ll"][m])),
                "d_crps_vs_A": dcr, "se_d_crps": se_cr,
                "floors_crps": (-dcr / se_cr) if (se_cr and dcr is not None) else None,
                "d_loglik_vs_A": dll, "se_d_loglik": se_ll,
                "floors_loglik": (dll / se_ll) if se_ll else None,
            }
        # the guard: on the rows the arm replaces, CRPS must not be worse than A by > 1 floor
        rep = r["replaced"]
        if rep.any():
            d = np.where(rep, r["crps"] - a["crps"], np.nan)
            se = block_se(d, games)
            dm = float(np.nanmean(d))
            row["guard_replaced_rows"] = {"n": int(rep.sum()), "d_crps": dm, "se": se,
                                          "worse_floors": dm / se if se else None,
                                          "pass": bool(dm <= se)}
        prof = {}
        for rn, rv in ROLES.items():
            for bn, lo, hi in BUCKETS:
                m = (role == rv) & unc & (rleft > lo) & (rleft <= hi)
                n = int(m.sum())
                prof[f"{rn}|{bn}"] = {"n": n, "label": cell_label(n),
                                      "pred_trunc_mean": float(trunc_mean[arm][m].mean()) if n else None,
                                      "actual_unc_mean": float(y[m].mean()) if n else None}
        row["profile"] = prof
        out["arms"][arm] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"spec": "docs/models/late_game/experiments.md section 7.5",
           "folds": {f: grade_fold(f) for f in ("F1", "F2")}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for f, r in res["folds"].items():
        print(f"== {f}: {r['n_rows']} rows, {r['n_uncensored']} uncensored, {r['n_games']} games")
        for arm, row in r["arms"].items():
            s = "  ".join(
                f"{sc}: crps {row[sc]['crps']:.4f} d {row[sc]['d_crps_vs_A'] if row[sc]['d_crps_vs_A'] is not None else 0:+.4f}"
                f" ({row[sc]['floors_crps'] if row[sc]['floors_crps'] is not None else 0:+.2f} fl) "
                f"ll {row[sc]['censored_loglik']:.4f} ({row[sc]['floors_loglik'] if row[sc]['floors_loglik'] is not None else 0:+.2f} fl)"
                f"{' ' + row[sc]['label'] if row[sc]['label'] else ''}"
                for sc in ("all", "trailing", "tied", "leading"))
            g = row.get("guard_replaced_rows")
            gs = (f"  GUARD {'PASS' if g['pass'] else 'FAIL'} ({g['worse_floors']:+.2f} fl worse, n={g['n']})"
                  if g else "")
            print(f"  {arm:4s} {s}{gs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
