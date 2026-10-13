"""train_endgame_clock_v1.py -- block C (clock) of late-game round 7, late_game/experiments.md section 15.

Arms (s15.A.2; the law code lives in `cbb_sim.models.endgame`, shared with the engine):
  GT0    the served K2 first-chance law (`r8_K2/<fold>/manifest.json`, S1 monthly), marginal and end-class
         conditional. Control. No fit.
  GT0s1  the K2 law refitted under seed 1 by its own trainer (`train_clock_r8_chance_v1.py --arm K2 --seed 1`,
         `r8_K2/<fold>_seed1/`): the noise-floor arm, never selectable.
  GT2    smooth margin response (AFT scale, 2 parameters per half), maximum likelihood on training window rows.
  GT1    role-band x time-bucket Kaplan-Meier cell law, hierarchical m = 50 shrink to the bucket then to GT0.
Training rows: window rows (periods 1-2, start <= 240 s, regulation) of the fold's TRAINING seasons only (F1 2022-23,
F2 2022-24); the served law on those rows is the fold's FIRST S1 refit, whose training data are exactly those
seasons. Test rows: the test season's window rows, the served law from the S1 schedule (refit at or before the game
date; `max_train_date < game_date` asserted per row). 2025-26 is never read. `--seed` seeds numpy before the fit;
the fits are deterministic, so a seed-1 refit must equal the seed-0 fit (reseed proof).

Writes: test predictions `results/endgame/offline/<fold>/C/<arm>.npz` (for `grade_endgame_v1.py`), engine artifacts
`data/processed/models/late_game/endgame_v1/<fold>/{gt1_law.npz, gt2_law.json}` (seed 0 only; seed k writes
`..._seed<k>` siblings), never overwriting another worker's file.

    .venv/Scripts/python.exe scripts/train_endgame_clock_v1.py --fold F2 --arm GT2
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import build_endgame_window_v1 as W  # noqa: E402

from cbb_sim.models import endgame as EG  # noqa: E402

CKD = ROOT / "data/processed/models/clock"
ART = ROOT / "data/processed/models/late_game/endgame_v1"
OFF = ROOT / "results/endgame/offline"
CHUNK = 40000


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_schedule(fold: str, seed: int = 0) -> list:
    d = CKD / "r8_K2" / (fold if seed == 0 else f"{fold}_seed{seed}")
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    out = []
    for mth in man["months"]:
        with open(CKD / mth["model_file"].replace("r8_K2/" + fold + "/", f"r8_K2/{d.name}/"), "rb") as f:
            out.append((pd.Timestamp(mth["refit_date"]), pd.Timestamp(mth["max_train_date"]), pickle.load(f)))
    return out


def served_pmfs(df: pd.DataFrame, sched: list, first_only: bool) -> tuple[np.ndarray, np.ndarray]:
    """(marginal, conditional) served pmfs per row. first_only: every row gets the fold's first refit (training rows);
    else the S1 rule, asserting max_train_date < game_date."""
    n = len(df)
    pm = np.empty((n, 91), dtype=np.float64)
    pe = np.empty((n, 91), dtype=np.float64)
    if first_only:
        seg = np.zeros(n, dtype=np.int64)
    else:
        gd = pd.to_datetime(df["game_date"]).to_numpy()
        cuts = np.array([s[0].to_datetime64() for s in sched])
        seg = np.searchsorted(cuts, gd, side="right") - 1
        if (seg < 0).any():
            raise SystemExit("a test row precedes the first refit")
        mtd = np.array([s[1].to_datetime64() for s in sched])[seg]
        if not (mtd < gd).all():
            raise SystemExit("created_at < tipoff violated: a refit's training data reach the game date")
    for k in np.unique(seg):
        rows = np.flatnonzero(seg == k)
        arm = sched[int(k)][2]
        for lo in range(0, len(rows), CHUNK):
            r = rows[lo:lo + CHUNK]
            blk = df.iloc[r].reset_index(drop=True)
            pm[r] = np.asarray(arm.pmf(blk), dtype=np.float64)
            pe[r] = np.asarray(arm.pmf_end(blk), dtype=np.float64)
    return pm, pe


def nll_gt2(b2, p0, y, cen, sec, m, period):
    b = np.zeros((2, 2))
    b[int(period[0]) - 1] = b2
    p = EG.scale_pmf(p0, np.exp(EG.GT2Law.log_k(b, period, sec, m)))
    ii = np.arange(len(y))
    pu = p[ii, y]
    S = 1.0 - np.cumsum(p, axis=1)[ii, y] + pu
    return -np.log(np.clip(np.where(cen, S, pu), 1e-12, None)).sum()


def fit_gt2(tr: pd.DataFrame, p0: np.ndarray) -> EG.GT2Law:
    from scipy.optimize import minimize
    b = np.zeros((2, 2))
    info = {}
    for h in (0, 1):
        sel = (tr["period"].to_numpy() == h + 1) & (tr["score_diff"].to_numpy() != 0)
        y = np.clip(tr["duration_s"].to_numpy(np.int64)[sel], 0, 90)
        cen = tr["censored"].to_numpy(bool)[sel]
        sec = tr["seconds_remaining"].to_numpy(float)[sel]
        m = tr["score_diff"].to_numpy(float)[sel]
        per = np.full(sel.sum(), h + 1)
        f = lambda x, a_=(p0[sel], y, cen, sec, m, per): nll_gt2(x, *a_)  # noqa: E731
        r = minimize(f, np.zeros(2), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 400})
        b[h] = r.x
        info[f"period{h + 1}"] = {"b_L": float(r.x[0]), "b_T": float(r.x[1]), "nll": float(r.fun),
                                  "nll_at_0": float(f(np.zeros(2))), "n_rows": int(sel.sum()), "iters": int(r.nit),
                                  "converged": bool(r.success)}
        log(f"GT2 period {h + 1}: b_L {r.x[0]:+.4f} b_T {r.x[1]:+.4f} (dNLL {r.fun - f(np.zeros(2)):+.1f}, n {sel.sum():,})")
    return EG.GT2Law(b=b, info=info)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["F1", "F2"], required=True)
    ap.add_argument("--arm", choices=["GT0", "GT0s1", "GT1", "GT2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed)
    t0 = time.time()
    tr, te = W.split(W.clock_window(), a.fold)
    log(f"{a.fold} {a.arm} seed {a.seed}: train {len(tr):,} rows {sorted(tr.season.unique())}, test {len(te):,} rows")
    sched = load_schedule(a.fold, 1 if a.arm == "GT0s1" else 0)
    p0t, p0te = served_pmfs(te, sched, first_only=False)
    meta = {"arm": a.arm, "fold": a.fold, "seed": a.seed, "created_at": datetime.now(UTC).isoformat(),
            "train_seasons": sorted(int(s) for s in tr["season"].unique()), "test_season": int(te["season"].iloc[0]),
            "served_schedule": [str(s[0].date()) for s in sched], "spec": "late_game/experiments.md s15"}
    if a.arm in ("GT0", "GT0s1"):
        pm, pe = p0t, p0te
    else:
        p0, p0e = served_pmfs(tr, sched, first_only=True)
        if a.arm == "GT2":
            law = fit_gt2(tr, p0)
        else:
            law = EG.fit_gt1(tr["period"].to_numpy(), tr["seconds_remaining"].to_numpy(), tr["score_diff"].to_numpy(),
                             tr["end1_code"].to_numpy(), tr["duration_s"].to_numpy(), tr["censored"].to_numpy(bool),
                             p0, p0e)
        meta["law"] = law.info
        args = (te["period"].to_numpy(), te["seconds_remaining"].to_numpy(), te["score_diff"].to_numpy())
        pm = law.apply(p0t, *args)
        pe = law.apply_end(p0te, *args, te["end1_code"].to_numpy())
        d = ART / (a.fold if a.seed == 0 else f"{a.fold}_seed{a.seed}")
        d.mkdir(parents=True, exist_ok=True)
        f = d / ("gt2_law.json" if a.arm == "GT2" else "gt1_law.npz")
        if f.exists() and a.seed == 0 and os.environ.get("EG_OVERWRITE") != "1":
            raise SystemExit(f"{f} exists; refusing to overwrite (EG_OVERWRITE=1 for an own rerun)")
        law.save(f)
        meta["artifact"] = str(f.relative_to(ROOT)).replace("\\", "/")
    out = OFF / a.fold / "C"
    out.mkdir(parents=True, exist_ok=True)
    name = a.arm if a.seed == 0 else f"{a.arm}_seed{a.seed}"
    np.savez_compressed(out / f"{name}.npz", game_id=te["game_id"].to_numpy(np.int64),
                        period=te["period"].to_numpy(np.int64), poss_index=te["poss_index"].to_numpy(np.int64),
                        pmf=pm.astype(np.float32), pmf_end=pe.astype(np.float32), meta=np.array(json.dumps(meta)))
    log(f"wrote {out / name}.npz ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
